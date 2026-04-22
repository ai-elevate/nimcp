#include <stddef.h>  /* for NULL */
#include <time.h>    /* for clock_gettime */
#include <zlib.h>    /* for gzopen/gzwrite/gzread — compress .snn checkpoint */
//=============================================================================
// nimcp_snn_network.c - SNN Network Implementation
//=============================================================================
/**
 * @file nimcp_snn_network.c
 * @brief Implementation of SNN network operations
 *
 * WHAT: Core SNN network creation, simulation, and training
 * WHY:  Provide spiking neural network functionality on existing infrastructure
 * HOW:  Orchestrates neural_network_t, manages populations, spike trains
 *
 * @author NIMCP Development Team
 * @date 2025-12-20
 * @version 1.0.0
 */

#include "snn/nimcp_snn_network.h"
#include "snn/nimcp_snn_synapse.h"
#include "snn/nimcp_snn_config.h"
#include "snn/nimcp_snn_adaptation.h"
#include "snn/nimcp_snn_basket.h"
#include "snn/nimcp_snn_training.h"  /* snn_tune_get_substrate_* */
#include "core/substrate/nimcp_substrate_effects.h"  /* substrate_compute_effects, apply helpers */
#include "gpu/snn/nimcp_snn_gpu.h"
#include "gpu/tensor/nimcp_tensor_gpu.h"
#include "gpu/context/nimcp_gpu_context.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <stdint.h>  /* UINT32_MAX (neural_network_add_neuron sentinel) */

/*=============================================================================
 * Health Agent Forward Declarations (Phase 8: Heartbeat for Long Operations)
 *============================================================================*/
#include "utils/fault_tolerance/nimcp_health_agent_macros.h"
#include "utils/thread/nimcp_thread_rand.h"
#include "constants/nimcp_math_constants.h"

NIMCP_DECLARE_HEALTH_AGENT_ATOMIC(snn_network)

//=============================================================================
// Internal Helper Functions
//=============================================================================

/* Forward declarations */
static void snn_population_destroy_internal(snn_population_t* pop);

/**
 * @brief Allocate and initialize a population structure
 */
static snn_population_t* snn_population_create_internal(
    uint32_t id,
    uint32_t n_neurons,
    neuron_type_t type,
    const char* name,
    uint32_t start_neuron_id
) {
    snn_population_t* pop = (snn_population_t*)nimcp_malloc(sizeof(snn_population_t));
    if (!pop) {
        NIMCP_THROW_MEMORY(NIMCP_ERROR_NO_MEMORY, sizeof(snn_population_t),
            "snn_population_create: allocation failed");
        return NULL;
    }

    memset(pop, 0, sizeof(snn_population_t));
    pop->id = id;
    pop->n_neurons = n_neurons;
    pop->neuron_type = type;

    if (name) {
        strncpy(pop->name, name, sizeof(pop->name) - 1);
    }

    /* Allocate neuron ID array */
    pop->neuron_ids = (uint32_t*)nimcp_malloc(n_neurons * sizeof(uint32_t));
    if (!pop->neuron_ids) {
        NIMCP_THROW_MEMORY(NIMCP_ERROR_NO_MEMORY, n_neurons * sizeof(uint32_t),
            "snn_population_create: neuron_ids allocation failed");
        nimcp_free(pop);
        return NULL;
    }

    /* Allocate spike trains */
    pop->spike_trains = (snn_spike_train_t*)nimcp_malloc(
        n_neurons * sizeof(snn_spike_train_t));
    if (!pop->spike_trains) {
        NIMCP_THROW_MEMORY(NIMCP_ERROR_NO_MEMORY, n_neurons * sizeof(snn_spike_train_t),
            "snn_population_create: spike_trains allocation failed");
        nimcp_free(pop->neuron_ids);
        nimcp_free(pop);
        return NULL;
    }
    memset(pop->spike_trains, 0, n_neurons * sizeof(snn_spike_train_t));

    /* Initialize spike trains and neuron IDs */
    for (uint32_t i = 0; i < n_neurons; i++) {
        pop->neuron_ids[i] = start_neuron_id + i;
        pop->spike_trains[i].neuron_id = start_neuron_id + i;
    }

    /* Allocate tensor views for efficient computation */
    uint32_t dims[1] = {n_neurons};
    pop->membrane_v = nimcp_tensor_create(dims, 1, NIMCP_DTYPE_F32);
    pop->spike_output = nimcp_tensor_create(dims, 1, NIMCP_DTYPE_F32);
    pop->refractory = nimcp_tensor_create(dims, 1, NIMCP_DTYPE_F32);

    if (!pop->membrane_v || !pop->spike_output || !pop->refractory) {
        NIMCP_THROW_MEMORY(NIMCP_ERROR_NO_MEMORY, n_neurons * sizeof(float),
            "snn_population_create: tensor allocation failed");
        if (pop->membrane_v) nimcp_tensor_destroy(pop->membrane_v);
        if (pop->spike_output) nimcp_tensor_destroy(pop->spike_output);
        if (pop->refractory) nimcp_tensor_destroy(pop->refractory);
        nimcp_free(pop->spike_trains);
        nimcp_free(pop->neuron_ids);
        nimcp_free(pop);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "snn_population_create_internal: validation failed");
        return NULL;
    }

    pop->topology = SNN_TOPO_FULL;
    pop->connectivity = 1.0f;

    /* Homeostatic + intrinsic-plasticity + biophysical state.
     *
     * Bug fix (pre-existing Wave A+B gap): the lightweight path (below)
     * allocates ahp, pump, basket, threshold_offset, neuron_rate_ema,
     * depression, and seeds EMAs. The non-lightweight path (this
     * function) previously left those NULL even though the same step
     * code runs for both. Non-lightweight pops silently lacked their
     * biophysical stability mechanisms.
     *
     * Fix: mirror the same allocations here. All allocation failures
     * are non-fatal — the step code branches on NULL, so a missing
     * mechanism simply becomes a no-op. */
    pop->firing_rate_ema = 0.03f;
    pop->rate_samples = 0;
    memset(pop->spike_count_history, 0, sizeof(pop->spike_count_history));
    pop->history_write_idx = 0;
    pop->history_total_steps = 0;

    pop->threshold_offset = nimcp_calloc(n_neurons, sizeof(float));
    pop->neuron_rate_ema  = nimcp_calloc(n_neurons, sizeof(float));
    pop->depression       = nimcp_calloc(n_neurons, sizeof(float));
    if (!pop->threshold_offset || !pop->neuron_rate_ema || !pop->depression) {
        NIMCP_LOGGING_WARN(
            "snn_population_create_internal: partial allocation of intrinsic "
            "plasticity / depression buffers for pop '%s' (%u neurons); "
            "mechanism will be no-op", name ? name : "unnamed", n_neurons);
    } else {
        const float ip_target = 0.03f;
        for (uint32_t i = 0; i < n_neurons; i++) {
            pop->neuron_rate_ema[i] = ip_target;
        }
    }

    extern float snn_tune_get_ahp_enabled(void);
    extern float snn_tune_get_pump_enabled(void);
    extern float snn_tune_get_basket_enabled(void);
    extern float snn_tune_get_ahp_tau_ms(void);
    extern float snn_tune_get_ahp_gain_mv(void);
    extern float snn_tune_get_pump_tau_ms(void);
    extern float snn_tune_get_pump_gain_mv(void);
    extern float snn_tune_get_basket_fraction(void);

    if (snn_tune_get_ahp_enabled() != 0.0f) {
        pop->ahp = snn_adaptation_create(n_neurons,
                                         snn_tune_get_ahp_tau_ms(),
                                         snn_tune_get_ahp_gain_mv(),
                                         1.0f);
        if (!pop->ahp) {
            NIMCP_LOGGING_WARN(
                "snn_population_create_internal: AHP allocation failed for "
                "pop '%s'; adaptation will be no-op",
                name ? name : "unnamed");
        }
    }
    if (snn_tune_get_pump_enabled() != 0.0f) {
        pop->pump = snn_adaptation_create(n_neurons,
                                          snn_tune_get_pump_tau_ms(),
                                          snn_tune_get_pump_gain_mv(),
                                          1.0f);
        if (!pop->pump) {
            NIMCP_LOGGING_WARN(
                "snn_population_create_internal: pump allocation failed for "
                "pop '%s'; adaptation will be no-op",
                name ? name : "unnamed");
        }
    }
    if (snn_tune_get_basket_enabled() != 0.0f) {
        pop->basket = snn_basket_pool_create(pop->id,
                                             n_neurons,
                                             snn_tune_get_basket_fraction());
        if (!pop->basket) {
            NIMCP_LOGGING_WARN(
                "snn_population_create_internal: basket-pool allocation failed "
                "for pop '%s'; inhibitory pool will be no-op",
                name ? name : "unnamed");
        }
    }

    return pop;
}

/**
 * @brief Create a lightweight population (no neuron_t allocation)
 *
 * Same as snn_population_create_internal but allocates external_current[]
 * and incoming_csr instead of relying on neural_network_t neurons.
 * Neuron IDs are logical: (pop_id << 20) | neuron_index.
 */
static snn_population_t* snn_population_create_lightweight(
    uint32_t id,
    uint32_t n_neurons,
    neuron_type_t type,
    const char* name)
{
    /* Reuse the standard create with dummy start_neuron_id */
    uint32_t logical_base = ((uint32_t)id << 20);
    snn_population_t* pop = snn_population_create_internal(
        id, n_neurons, type, name, logical_base);
    if (!pop) return NULL;

    /* Allocate lightweight-specific storage.
     *
     * NOTE: Homeostatic state, intrinsic plasticity buffers
     * (threshold_offset, neuron_rate_ema, depression) and the
     * biophysical stability mechanisms (ahp, pump, basket) are
     * allocated and seeded by snn_population_create_internal() itself
     * — lightweight does NOT re-allocate them here. (Previously
     * lightweight was the only path that allocated them; the
     * non-lightweight path left them NULL. That gap has been closed
     * in snn_population_create_internal.) */
    pop->lightweight = true;
    pop->external_current = nimcp_calloc(n_neurons, sizeof(float));
    pop->incoming_csr = snn_csr_create(n_neurons, n_neurons * 20);

    if (!pop->external_current || !pop->incoming_csr) {
        snn_population_destroy_internal(pop);
        return NULL;
    }

    /* Initialize membrane potential to resting potential */
    float* v = (float*)nimcp_tensor_data(pop->membrane_v);
    if (v) {
        for (uint32_t i = 0; i < n_neurons; i++) {
            v[i] = -65.0f;  /* v_rest default */
        }
    }

    return pop;
}

/**
 * @brief Destroy a population structure
 */
static void snn_population_destroy_internal(snn_population_t* pop) {
    if (!pop) return;

    if (pop->membrane_v) nimcp_tensor_destroy(pop->membrane_v);
    if (pop->spike_output) nimcp_tensor_destroy(pop->spike_output);
    if (pop->refractory) nimcp_tensor_destroy(pop->refractory);
    if (pop->spike_trains) nimcp_free(pop->spike_trains);
    if (pop->neuron_ids) nimcp_free(pop->neuron_ids);

    /* Lightweight mode cleanup */
    if (pop->external_current) nimcp_free(pop->external_current);
    if (pop->incoming_csr) {
        snn_csr_destroy(pop->incoming_csr);
    }

    /* Intrinsic plasticity + short-term depression state */
    if (pop->threshold_offset) nimcp_free(pop->threshold_offset);
    if (pop->neuron_rate_ema) nimcp_free(pop->neuron_rate_ema);
    if (pop->depression) nimcp_free(pop->depression);

    /* Biophysical stability mechanisms. Destroyers tolerate NULL. */
    if (pop->ahp)    { snn_adaptation_destroy(pop->ahp);     pop->ahp = NULL; }
    if (pop->pump)   { snn_adaptation_destroy(pop->pump);    pop->pump = NULL; }
    if (pop->basket) { snn_basket_pool_destroy(pop->basket); pop->basket = NULL; }

    nimcp_free(pop);
}

/**
 * @brief Initialize simulation context
 */
static snn_simulation_t* snn_simulation_create_internal(float dt_ms, uint32_t total_neurons) {
    snn_simulation_t* sim = (snn_simulation_t*)nimcp_malloc(sizeof(snn_simulation_t));
    if (!sim) {
        NIMCP_THROW_MEMORY(NIMCP_ERROR_NO_MEMORY, sizeof(snn_simulation_t),
            "snn_simulation_create: allocation failed");
        return NULL;
    }

    memset(sim, 0, sizeof(snn_simulation_t));
    sim->dt_ms = dt_ms;
    sim->health = SNN_STATE_HEALTHY;

    /* Allocate spike queue — scale with total neuron count.
     * At 1.8M SNN neurons with ~5% firing rate per step, need ~90K queue slots.
     * queue = total/2 handles bursts; cap at 2M prevents runaway alloc. */
    sim->queue_capacity = (total_neurons > 20000) ? total_neurons / 2 : 10000;
    if (sim->queue_capacity > 2000000) sim->queue_capacity = 2000000;
    sim->spike_queue = (snn_spike_t*)nimcp_malloc(
        sim->queue_capacity * sizeof(snn_spike_t));
    if (!sim->spike_queue) {
        NIMCP_THROW_MEMORY(NIMCP_ERROR_NO_MEMORY, sim->queue_capacity * sizeof(snn_spike_t),
            "snn_simulation_create: spike_queue allocation failed");
        nimcp_free(sim);
        return NULL;
    }

    /* Initialize RNG */
    sim->rng_state = 12345ULL;

    return sim;
}

/**
 * @brief Destroy simulation context
 */
static void snn_simulation_destroy_internal(snn_simulation_t* sim) {
    if (!sim) return;
    if (sim->spike_queue) nimcp_free(sim->spike_queue);
    nimcp_free(sim);
}

/** Track spike buffer overflow events for monitoring */
static volatile uint64_t g_spike_buffer_overflow_count = 0;
static volatile uint64_t g_last_overflow_warning_count = 0;

/**
 * @brief Record spike to spike train with overflow protection
 *
 * WHAT: Record spike timestamp to circular buffer
 * WHY:  Track spike history for STDP, analysis, and decoding
 * HOW:  Circular buffer wraps when full, overwrites oldest spikes
 *
 * NOTE: When buffer is full, old spikes are overwritten (circular buffer).
 *       This is intentional behavior for bounded memory, but we log
 *       warnings to detect pathological high-firing scenarios.
 */
static void record_spike(snn_spike_train_t* train, uint64_t time_us) {
    uint32_t idx = train->write_idx;
    train->spike_times[idx] = time_us;
    train->write_idx = (idx + 1) % SNN_SPIKE_BUFFER_SIZE;

    if (train->count < SNN_SPIKE_BUFFER_SIZE) {
        train->count++;
    } else {
        /* Buffer overflow - old spikes being overwritten
         * WHAT: Track overflow events
         * WHY:  High spike rates may indicate network instability
         * HOW:  Log periodic warnings (not every overflow to avoid log spam)
         */
        uint64_t overflow_count = __atomic_add_fetch(&g_spike_buffer_overflow_count, 1, __ATOMIC_RELAXED);

        /* Log warning every 1000 overflows to avoid log spam */
        uint64_t last_warning = __atomic_load_n(&g_last_overflow_warning_count, __ATOMIC_RELAXED);
        if (overflow_count - last_warning >= 1000) {
            NIMCP_LOGGING_WARN("Spike buffer overflow: neuron %u buffer full "
                                  "(capacity=%u), overwriting old spikes. "
                                  "Total overflows: %lu. Consider increasing "
                                  "SNN_SPIKE_BUFFER_SIZE or reducing simulation time.",
                                  train->neuron_id, SNN_SPIKE_BUFFER_SIZE,
                                  (unsigned long)overflow_count);
            __atomic_store_n(&g_last_overflow_warning_count, overflow_count, __ATOMIC_RELAXED);
        }
    }
    train->total_spikes++;
}

/**
 * @brief Compute instantaneous firing rate
 */
static float compute_firing_rate(const snn_spike_train_t* train,
                                 uint64_t current_time_us,
                                 float window_ms) {
    if (train->count == 0) return 0.0f;

    uint64_t window_us = (uint64_t)(window_ms * 1000.0f);
    uint64_t cutoff = (current_time_us > window_us) ? current_time_us - window_us : 0;

    uint32_t count = 0;
    for (uint32_t i = 0; i < train->count; i++) {
        if (train->spike_times[i] >= cutoff) {
            count++;
        }
    }

    return (float)count / (window_ms / 1000.0f);  /* Convert to Hz */
}

//=============================================================================
// Network Lifecycle
//=============================================================================

snn_network_t* snn_network_create(const snn_config_t* config) {
    /* Guard clause: validate config */
    if (!config) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
            "snn_network_create: NULL config");
        return NULL;
    }

    int validate_result = snn_config_validate(config);
    if (validate_result != SNN_SUCCESS) {
        /* Exception already thrown by snn_config_validate */
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "snn_network_create: validation failed");
        return NULL;
    }

    /* Allocate network structure */
    snn_network_t* network = (snn_network_t*)nimcp_malloc(sizeof(snn_network_t));
    if (!network) {
        NIMCP_THROW_MEMORY(NIMCP_ERROR_NO_MEMORY, sizeof(snn_network_t),
            "snn_network_create: allocation failed");
        return NULL;
    }
    memset(network, 0, sizeof(snn_network_t));

    /* Set magic number and copy config */
    network->magic = SNN_MAGIC;
    snn_config_clone(config, &network->config);

    /* Create underlying neural network */
    network_config_t nn_config;
    memset(&nn_config, 0, sizeof(network_config_t));
    nn_config.num_neurons = config->n_inputs + config->n_hidden + config->n_outputs;
    nn_config.input_size = config->n_inputs;
    nn_config.output_size = config->n_outputs;
    nn_config.enable_stdp = config->enable_stdp;
    nn_config.refractory_period = config->t_ref;
    nn_config.neuron_model = NEURON_MODEL_LIF;  /* Default to LIF */
    nn_config.enable_bio_async = config->enable_bio_async;

    network->neural_net = neural_network_create(&nn_config);
    if (!network->neural_net) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NETWORK_CREATION,
            "snn_network_create: neural_network_create failed");
        nimcp_free(network);
        return NULL;
    }

    /* Allocate population array.
     *
     * Bug fix (pre-existing heap overflow): the old code sized this array by
     * `config->n_populations` when > 0. For a feedforward config with a
     * hidden layer, that yields exactly 3 slots — and calling
     * snn_network_add_population() later would then write populations[3],
     * corrupting the heap. `snn_network_add_population{,_lightweight}()`
     * only guarded against `SNN_MAX_POPULATIONS`, not the actual alloc size.
     *
     * Fix: ALWAYS allocate SNN_MAX_POPULATIONS slots regardless of config.
     * Cost: 128 * sizeof(ptr) = 1 KB per network — trivial at 2M-neuron
     * scale. We also record the actual allocation size in
     * `populations_capacity` so the add-population guards can use it
     * directly (decoupling the bound from the constant). */
    uint32_t max_populations = SNN_MAX_POPULATIONS;
    network->populations = (snn_population_t**)nimcp_malloc(
        max_populations * sizeof(snn_population_t*));
    if (!network->populations) {
        NIMCP_THROW_MEMORY(NIMCP_ERROR_NO_MEMORY, max_populations * sizeof(snn_population_t*),
            "snn_network_create: populations array allocation failed");
        neural_network_destroy(network->neural_net);
        nimcp_free(network);
        return NULL;
    }
    memset(network->populations, 0, max_populations * sizeof(snn_population_t*));
    network->populations_capacity = max_populations;

    /* Create simulation context */
    uint32_t total_snn_neurons = config->n_inputs + config->n_hidden + config->n_outputs;
    network->sim = snn_simulation_create_internal(config->dt, total_snn_neurons);
    if (!network->sim) {
        /* Exception already thrown by snn_simulation_create_internal */
        nimcp_free(network->populations);
        neural_network_destroy(network->neural_net);
        nimcp_free(network);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "snn_network_create: network->sim is NULL");
        return NULL;
    }

    /* Create input population */
    network->input_pop = snn_population_create_internal(
        0, config->n_inputs, NEURON_GENERIC_LIF, "input", 0);
    if (!network->input_pop) {
        /* Exception already thrown by snn_population_create_internal */
        snn_network_destroy(network);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "snn_network_create: network->input_pop is NULL");
        return NULL;
    }
    network->populations[0] = network->input_pop;
    network->n_populations = 1;

    /* Create hidden populations if n_hidden > 0 */
    uint32_t next_neuron_id = config->n_inputs;
    uint32_t next_pop_id = 1;

    if (config->n_hidden > 0) {
        snn_population_t* hidden_pop = snn_population_create_internal(
            next_pop_id, config->n_hidden, NEURON_GENERIC_LIF, "hidden", next_neuron_id);
        if (!hidden_pop) {
            snn_network_destroy(network);
            NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "snn_network_create: hidden_pop is NULL");
            return NULL;
        }
        network->populations[next_pop_id] = hidden_pop;
        next_neuron_id += config->n_hidden;
        next_pop_id++;
        network->n_populations = next_pop_id;
    }

    /* Create output population */
    network->output_pop = snn_population_create_internal(
        next_pop_id, config->n_outputs, NEURON_GENERIC_LIF, "output", next_neuron_id);
    if (!network->output_pop) {
        /* Exception already thrown by snn_population_create_internal */
        snn_network_destroy(network);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "snn_network_create: network->output_pop is NULL");
        return NULL;
    }
    network->populations[next_pop_id] = network->output_pop;
    network->n_populations = next_pop_id + 1;

    /* Create mutex for thread safety */
    network->mutex = nimcp_malloc(sizeof(nimcp_mutex_t));
    if (network->mutex) {
        nimcp_mutex_init((nimcp_mutex_t*)network->mutex, NULL);
    }

    /* GPU acceleration: try to create LIF state on GPU */
    network->gpu_lif_state = NULL;
    network->gpu_ctx = NULL;
    {
        nimcp_gpu_context_t* gpu = nimcp_gpu_context_create_auto();
        if (gpu) {
            size_t total_neurons = (size_t)config->n_inputs + config->n_hidden + config->n_outputs;
            nimcp_lif_params_t lif_params = {
                .tau_mem   = config->tau_mem,
                .tau_syn   = config->tau_syn > 0.0f ? config->tau_syn : 5.0f,
                .v_thresh  = config->v_thresh,
                .v_reset   = config->v_reset,
                .v_rest    = config->v_rest,
                .dt        = config->dt,
                .hard_reset = true
            };
            nimcp_lif_state_t* lif_state = nimcp_lif_state_create(gpu, total_neurons, &lif_params);
            if (lif_state) {
                network->gpu_lif_state = lif_state;
                network->gpu_ctx = gpu;
                NIMCP_LOGGING_INFO("snn_network_create: GPU LIF state created for %zu neurons",
                                   total_neurons);
            } else {
                NIMCP_LOGGING_INFO("snn_network_create: GPU LIF state creation failed, using CPU fallback");
                nimcp_gpu_context_destroy(gpu);
            }
        } else {
            NIMCP_LOGGING_DEBUG("snn_network_create: no GPU available, using CPU path");
        }
    }

    NIMCP_LOGGING_INFO("snn_network_create: created SNN with %u inputs, %u outputs",
                       config->n_inputs, config->n_outputs);

    return network;
}

void snn_network_destroy(snn_network_t* network) {
    if (!network) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "snn_network_destroy: null network pointer");
        return;
    }

    /* Destroy populations */
    if (network->populations) {
        for (uint32_t i = 0; i < network->n_populations; i++) {
            if (network->populations[i]) {
                snn_population_destroy_internal(network->populations[i]);
            }
        }
        nimcp_free(network->populations);
    }

    /* Destroy simulation context */
    if (network->sim) {
        snn_simulation_destroy_internal(network->sim);
    }

    /* Destroy training context */
    if (network->train_ctx) {
        if (network->train_ctx->eligibility) {
            nimcp_tensor_destroy(network->train_ctx->eligibility);
        }
        if (network->train_ctx->grad_membrane) {
            nimcp_tensor_destroy(network->train_ctx->grad_membrane);
        }
        if (network->train_ctx->grad_weights) {
            nimcp_tensor_destroy(network->train_ctx->grad_weights);
        }
        nimcp_free(network->train_ctx);
    }

    /* Destroy encoder/decoder (when implemented) */

    /* Destroy underlying neural network */
    if (network->neural_net) {
        neural_network_destroy(network->neural_net);
    }

    /* Destroy GPU LIF state */
    if (network->gpu_lif_state) {
        nimcp_lif_state_destroy((nimcp_lif_state_t*)network->gpu_lif_state);
        network->gpu_lif_state = NULL;
    }
    if (network->gpu_ctx) {
        nimcp_gpu_context_destroy((nimcp_gpu_context_t*)network->gpu_ctx);
        network->gpu_ctx = NULL;
    }

    /* Destroy mutex */
    if (network->mutex) {
        nimcp_mutex_free((nimcp_mutex_t*)network->mutex);
    }

    /* Clear and free */
    memset(network, 0, sizeof(snn_network_t));
    nimcp_free(network);

    NIMCP_LOGGING_DEBUG("snn_network_destroy: network destroyed");
}

void snn_network_attach_substrate(snn_network_t* net,
                                  struct neural_substrate* sub) {
    if (!net) return;  /* null-tolerant: silent drop */
    net->substrate = sub;
    /* Reset cached effects + counter so the next step recomputes fresh. */
    memset(&net->cached_axon_effects, 0, sizeof(net->cached_axon_effects));
    memset(&net->cached_dend_effects, 0, sizeof(net->cached_dend_effects));
    net->substrate_steps_since_update = 0;
    /* Bug #1 fix: populate cache IMMEDIATELY so downstream consumers
     * (snn_rstdp_apply) that read cached_dend_effects.plasticity_mod
     * before the first step don't multiply LR by 0. Without this,
     * R-STDP silently dies on the GPU path (where the CPU-fallback
     * refresh block never runs) and on any pre-step inference. */
    if (sub) {
        substrate_compute_effects(sub,
                                  &net->cached_axon_effects,
                                  &net->cached_dend_effects);
    }
    NIMCP_LOGGING_DEBUG("snn_network_attach_substrate: substrate=%p",
                        (void*)sub);
}

/* Accessor for wire-up regression tests — returns the substrate pointer
 * the network currently holds (borrowed). NULL-tolerant. */
struct neural_substrate* snn_network_get_substrate(snn_network_t* net) {
    return net ? net->substrate : NULL;
}

int snn_network_reset(snn_network_t* network) {
    if (!network) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "snn_network_reset: null network pointer");
        return SNN_ERROR_NULL_POINTER;
    }

    /* Reset all populations */
    for (uint32_t p = 0; p < network->n_populations; p++) {
        snn_population_t* pop = network->populations[p];
        if (!pop) continue;

        /* Reset spike trains */
        for (uint32_t n = 0; n < pop->n_neurons; n++) {
            memset(&pop->spike_trains[n], 0, sizeof(snn_spike_train_t));
            pop->spike_trains[n].neuron_id = n;
        }

        /* Reset membrane potentials to resting */
        float* v_data = (float*)nimcp_tensor_data(pop->membrane_v);
        for (uint32_t n = 0; n < pop->n_neurons; n++) {
            v_data[n] = network->config.v_rest;
        }

        /* Clear spike output and refractory */
        memset(nimcp_tensor_data(pop->spike_output), 0,
               pop->n_neurons * sizeof(float));
        memset(nimcp_tensor_data(pop->refractory), 0,
               pop->n_neurons * sizeof(float));

        pop->total_spikes = 0;
        pop->mean_rate = 0.0f;
    }

    /* Reset simulation context */
    if (network->sim) {
        network->sim->current_time_us = 0;
        network->sim->step_count = 0;
        network->sim->queue_size = 0;
        network->sim->health = SNN_STATE_HEALTHY;
        network->sim->total_energy = 0.0f;
    }

    /* Reset underlying neural network */
    if (network->neural_net) {
        neural_network_reset(network->neural_net);
    }

    /* Reset statistics */
    memset(&network->stats, 0, sizeof(snn_stats_t));

    NIMCP_LOGGING_DEBUG("snn_network_reset: network reset to initial state");
    return SNN_SUCCESS;
}

//=============================================================================
// Simulation
//=============================================================================

int snn_network_step(snn_network_t* network, float dt) {
    if (!network) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "snn_network_step: null network pointer");
        return SNN_ERROR_NULL_POINTER;
    }
    if (!network->sim) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_STATE, "snn_network_step: null simulation context");
        return SNN_ERROR_INVALID_STATE;
    }

    /* Phase 8: Send heartbeat at start of network step */
    snn_network_heartbeat("snn_step", 0.0f);

    float dt_ms = (dt > 0.0f) ? dt : network->config.dt;
    uint64_t dt_us = (uint64_t)(dt_ms * 1000.0f);

    int total_spikes = 0;
    bool gpu_executed = false;
    struct timespec _step_t0, _step_t1, _phase_t0, _phase_t1;
    double _isyn_ms = 0, _lif_ms = 0, _readback_ms = 0;
    clock_gettime(CLOCK_MONOTONIC, &_step_t0);

    /* ===== Substrate adapter: refresh cache EVERY step, BOTH paths =====
     * Bug #2 fix: previously this lived inside `if (!gpu_executed)`, which
     * meant the production GPU path never refreshed the cache and every
     * R-STDP weight update multiplied by a stale (possibly zero) value.
     * Hoist it to the top of snn_network_step so both CPU and GPU paths
     * see up-to-date plasticity_mod / spike_reliability / etc.
     *
     * GPU-path limitation: per-neuron tau_eff/tref_eff modulation still
     * only applies on the CPU fallback — those values feed the LIF
     * integration loop which runs inside a CUDA kernel that doesn't read
     * the cache. Plasticity modulation still works on GPU because it
     * happens in snn_rstdp_apply on the host side after the step.
     * Future work: upload the cache struct to GPU and thread it through
     * the LIF + surrogate-gradient kernels. */
    if (network->substrate && snn_tune_get_substrate_enabled() != 0.0f) {
        const uint32_t period =
            (uint32_t)snn_tune_get_substrate_update_period();
        if (network->substrate_steps_since_update == 0 || period == 0) {
            substrate_compute_effects(network->substrate,
                                      &network->cached_axon_effects,
                                      &network->cached_dend_effects);
        }
        network->substrate_steps_since_update++;
        /* Bug #5 verification: cap counter to avoid uint32 overflow on
         * very long runs + ensure deterministic modulo rollover. */
        if (period > 0 && network->substrate_steps_since_update >= period) {
            network->substrate_steps_since_update = 0;
        }
    }

    /* ===== GPU FAST PATH ===== */
    if (network->gpu_lif_state && network->gpu_ctx) {
        nimcp_gpu_context_t* gpu = (nimcp_gpu_context_t*)network->gpu_ctx;
        nimcp_lif_state_t* lif_state = (nimcp_lif_state_t*)network->gpu_lif_state;

        /* Compute total neuron count across all populations */
        size_t total_neurons = 0;
        for (uint32_t p = 0; p < network->n_populations; p++) {
            if (network->populations[p]) {
                total_neurons += network->populations[p]->n_neurons;
            }
        }

        /* Build input current vector for LIF kernel.
         * GPU CSR path: use kernel_snn_isyn_csr per population.
         * CPU fallback: iterate synapses sequentially. */
        float* h_input = (float*)nimcp_calloc(total_neurons, sizeof(float));
        if (h_input) {
            clock_gettime(CLOCK_MONOTONIC, &_phase_t0);
            /* Check if any lightweight pop has GPU-ready CSR */
            bool has_gpu_csr = false;
            for (uint32_t p = 0; p < network->n_populations && !has_gpu_csr; p++) {
                snn_population_t* pop = network->populations[p];
                if (pop && pop->lightweight && pop->incoming_csr &&
                    pop->incoming_csr->gpu_ready)
                    has_gpu_csr = true;
            }

            if (has_gpu_csr) {
                /* === GPU I_SYN PATH === */
                /* 1. Build flat spike vector on host from population tensors */
                float* h_spk_flat = (float*)nimcp_calloc(total_neurons, sizeof(float));
                if (h_spk_flat) {
                    size_t off = 0;
                    for (uint32_t p = 0; p < network->n_populations; p++) {
                        snn_population_t* pop = network->populations[p];
                        if (!pop) continue;
                        float* sd = (float*)nimcp_tensor_data(pop->spike_output);
                        if (sd) memcpy(h_spk_flat + off, sd, pop->n_neurons * sizeof(float));
                        off += pop->n_neurons;
                    }

                    /* 2. Upload flat spike vector to GPU */
                    size_t spk_dims[1] = { total_neurons };
                    nimcp_gpu_tensor_t* d_spk = nimcp_gpu_tensor_from_host(
                        gpu, h_spk_flat, spk_dims, 1, NIMCP_GPU_PRECISION_FP32);

                    if (d_spk) {
                        /* 3. Per-population: upload CSR + external_current, call kernel */
                        size_t pop_offset = 0;
                        for (uint32_t p = 0; p < network->n_populations; p++) {
                            snn_population_t* pop = network->populations[p];
                            if (!pop) continue;

                            if (pop->lightweight && pop->incoming_csr &&
                                pop->incoming_csr->gpu_ready &&
                                pop->incoming_csr->n_synapses > 0) {
                                snn_csr_storage_t* csr = pop->incoming_csr;

                                /* === V2 Fast path: persistent GPU CSR ===
                                 * Upload once, reuse every timestep. Eliminates
                                 * ~12 GB PCIe transfer per step for 1.45B-synapse
                                 * populations. Falls back to per-step upload if
                                 * upload fails or is disabled. */
                                bool use_persistent = false;
                                if (!csr->gpu_resident) {
                                    /* Lazy upload on first use */
                                    if (snn_csr_upload_to_gpu(csr, gpu) == 0) {
                                        use_persistent = true;
                                    }
                                } else {
                                    use_persistent = true;
                                }

                                size_t n_dims[1] = { pop->n_neurons };
                                float* ext_cur = pop->external_current;
                                float* h_ext = ext_cur ? ext_cur :
                                    (float*)nimcp_calloc(pop->n_neurons, sizeof(float));
                                nimcp_gpu_tensor_t* d_ext = nimcp_gpu_tensor_from_host(
                                    gpu, h_ext, n_dims, 1, NIMCP_GPU_PRECISION_FP32);
                                nimcp_gpu_tensor_t* d_isyn = nimcp_gpu_tensor_create(
                                    gpu, n_dims, 1, NIMCP_GPU_PRECISION_FP32);

                                extern bool nimcp_gpu_snn_isyn_csr(
                                    nimcp_gpu_context_t*, const float*,
                                    const float*, const unsigned int*,
                                    const unsigned int*, const float*,
                                    float*, size_t);

                                if (use_persistent && d_ext && d_isyn) {
                                    /* Fast path: use persistent device pointers */
                                    nimcp_gpu_snn_isyn_csr(gpu,
                                        (const float*)d_spk->data,
                                        (const float*)csr->d_weights,
                                        (const unsigned int*)csr->d_flat_col_idx,
                                        (const unsigned int*)csr->d_row_ptr,
                                        (const float*)d_ext->data,
                                        (float*)d_isyn->data,
                                        pop->n_neurons);
                                    nimcp_gpu_tensor_to_host(d_isyn,
                                        h_input + pop_offset);
                                } else if (d_ext && d_isyn) {
                                    /* Slow fallback: per-step CSR upload (legacy) */
                                    size_t w_dims[1] = { csr->n_synapses };
                                    size_t r_dims[1] = { (size_t)csr->n_neurons + 1 };
                                    nimcp_gpu_tensor_t* d_w = nimcp_gpu_tensor_from_host(
                                        gpu, csr->weights, w_dims, 1, NIMCP_GPU_PRECISION_FP32);
                                    nimcp_gpu_tensor_t* d_ci = nimcp_gpu_tensor_from_host(
                                        gpu, csr->flat_col_idx, w_dims, 1, NIMCP_GPU_PRECISION_UINT32);
                                    nimcp_gpu_tensor_t* d_rp = nimcp_gpu_tensor_from_host(
                                        gpu, csr->row_ptr, r_dims, 1, NIMCP_GPU_PRECISION_UINT32);
                                    if (d_w && d_ci && d_rp) {
                                        nimcp_gpu_snn_isyn_csr(gpu,
                                            (const float*)d_spk->data,
                                            (const float*)d_w->data,
                                            (const unsigned int*)d_ci->data,
                                            (const unsigned int*)d_rp->data,
                                            (const float*)d_ext->data,
                                            (float*)d_isyn->data,
                                            pop->n_neurons);
                                        nimcp_gpu_tensor_to_host(d_isyn,
                                            h_input + pop_offset);
                                    }
                                    if (d_w) nimcp_gpu_tensor_destroy(d_w);
                                    if (d_ci) nimcp_gpu_tensor_destroy(d_ci);
                                    if (d_rp) nimcp_gpu_tensor_destroy(d_rp);
                                }

                                if (d_ext) nimcp_gpu_tensor_destroy(d_ext);
                                if (d_isyn) nimcp_gpu_tensor_destroy(d_isyn);
                                if (!ext_cur && h_ext) nimcp_free(h_ext);
                            } else {
                                /* CPU fallback for non-lightweight or non-GPU-ready pops */
                                for (uint32_t n = 0; n < pop->n_neurons; n++) {
                                    float I_syn = 0.0f;
                                    if (pop->lightweight && pop->incoming_csr &&
                                        pop->incoming_csr->finalized) {
                                        I_syn = pop->external_current ?
                                            pop->external_current[n] : 0.0f;
                                        uint32_t sc;
                                        snn_csr_synapse_t* syns = snn_csr_get_incoming(
                                            pop->incoming_csr, n, &sc);
                                        for (uint32_t s = 0; s < sc; s++) {
                                            if (syns[s].src_pop >= network->n_populations) continue;
                                            snn_population_t* sp = network->populations[syns[s].src_pop];
                                            if (!sp || !sp->spike_output) continue;
                                            float* ss = (float*)nimcp_tensor_data(sp->spike_output);
                                            if (ss && syns[s].src_neuron < sp->n_neurons
                                                && ss[syns[s].src_neuron] > 0.5f)
                                                I_syn += syns[s].weight;
                                        }
                                    } else if (network->neural_net) {
                                        neuron_t* neuron = neural_network_get_neuron(
                                            network->neural_net, pop->neuron_ids[n]);
                                        if (neuron) {
                                            I_syn = neuron->external_current;
                                            uint32_t ic = neuron->incoming.embedded_count
                                                        + neuron->incoming.overflow_count;
                                            for (uint32_t s = 0; s < ic; s++) {
                                                synapse_handle_t* h = sparse_synapse_get(&neuron->incoming, s);
                                                if (!h) continue;
                                                neuron_t* pre = neural_network_get_neuron(
                                                    network->neural_net, h->target_neuron_id);
                                                if (pre && pre->state > 0.5f)
                                                    I_syn += h->weight;
                                            }
                                        }
                                    }
                                    h_input[pop_offset + n] = I_syn;
                                }
                            }
                            pop_offset += pop->n_neurons;
                        }
                        nimcp_gpu_tensor_destroy(d_spk);
                    }
                    nimcp_free(h_spk_flat);
                }
            } else {
                /* === CPU-ONLY I_SYN PATH (no GPU CSR available) === */
                size_t neuron_offset = 0;
                for (uint32_t p = 0; p < network->n_populations; p++) {
                    snn_population_t* pop = network->populations[p];
                    if (!pop) continue;
                    for (uint32_t n = 0; n < pop->n_neurons; n++) {
                        float I_syn = 0.0f;
                        if (pop->lightweight && pop->incoming_csr && pop->incoming_csr->finalized) {
                            I_syn = pop->external_current ? pop->external_current[n] : 0.0f;
                            uint32_t sc;
                            snn_csr_synapse_t* syns = snn_csr_get_incoming(
                                pop->incoming_csr, n, &sc);
                            for (uint32_t s = 0; s < sc; s++) {
                                if (syns[s].src_pop >= network->n_populations) continue;
                                snn_population_t* sp = network->populations[syns[s].src_pop];
                                if (!sp || !sp->spike_output) continue;
                                float* ss = (float*)nimcp_tensor_data(sp->spike_output);
                                if (ss && syns[s].src_neuron < sp->n_neurons
                                    && ss[syns[s].src_neuron] > 0.5f)
                                    I_syn += syns[s].weight;
                            }
                        } else if (network->neural_net) {
                            neuron_t* neuron = neural_network_get_neuron(
                                network->neural_net, pop->neuron_ids[n]);
                            if (neuron) {
                                I_syn = neuron->external_current;
                                uint32_t ic = neuron->incoming.embedded_count
                                            + neuron->incoming.overflow_count;
                                for (uint32_t s = 0; s < ic; s++) {
                                    synapse_handle_t* h = sparse_synapse_get(&neuron->incoming, s);
                                    if (!h) continue;
                                    neuron_t* pre = neural_network_get_neuron(
                                        network->neural_net, h->target_neuron_id);
                                    if (pre && pre->state > 0.5f)
                                        I_syn += h->weight;
                                }
                            }
                        }
                        h_input[neuron_offset + n] = I_syn;
                    }
                    neuron_offset += pop->n_neurons;
                }
            }

            clock_gettime(CLOCK_MONOTONIC, &_phase_t1);
            _isyn_ms = (_phase_t1.tv_sec - _phase_t0.tv_sec) * 1000.0
                     + (_phase_t1.tv_nsec - _phase_t0.tv_nsec) / 1e6;

            /* Adaptive Poisson background noise — inject per-population
             * scaled depolarizing pulses into h_input before GPU LIF
             * forward. Dead pops get full noise to escape the absorbing
             * zero state; pops at or above target get zero noise so noise
             * doesn't saturate healthy populations. The per-pop factor
             * comes from snn_noise_factor_for_pop() which reads each
             * pop's firing_rate_ema. noise_rate_hz=0 disables entirely.
             *
             * See project_snn_structural_fixes_2026-04-21 for why fixed-
             * amplitude noise caused SATURATION ↔ FULL_COLLAPSE cycling. */
            {
                extern float snn_tune_get_noise_rate_hz(void);
                extern float snn_tune_get_noise_pulse_mv(void);
                extern float snn_tune_get_noise_ei_ratio(void);
                extern float snn_noise_factor_for_pop(const snn_population_t*);
                const float nrate = snn_tune_get_noise_rate_hz();
                if (nrate > 0.0f) {
                    const float npulse = snn_tune_get_noise_pulse_mv();
                    const float p_base = nrate * dt_ms * 0.001f;
                    const float ei_ratio = snn_tune_get_noise_ei_ratio();
                    static __thread unsigned int _gpu_noise_seed = 0;
                    if (_gpu_noise_seed == 0) {
                        struct timespec _nt;
                        clock_gettime(CLOCK_MONOTONIC, &_nt);
                        _gpu_noise_seed = (unsigned int)(_nt.tv_nsec
                            ^ ((uintptr_t)&_gpu_noise_seed >> 4));
                        if (_gpu_noise_seed == 0) _gpu_noise_seed = 0xDEADBEEFu;
                    }
                    uint32_t neuron_idx = 0;
                    for (uint32_t p = 0; p < network->n_populations; p++) {
                        snn_population_t* pop = network->populations[p];
                        if (!pop) continue;
                        const float factor = snn_noise_factor_for_pop(pop);
                        const unsigned int pop_thresh =
                            (unsigned int)(p_base * factor * (float)RAND_MAX);
                        /* Dead-pop exception: if pop is (nearly) fully in
                         * rescue mode, force excitatory-only noise so it
                         * can escape the absorbing-zero state. Otherwise
                         * apply ei_ratio as the fraction of pulses that
                         * are inhibitory. */
                        const int exc_only = (factor >= 0.9f);
                        for (uint32_t n = 0; n < pop->n_neurons; n++) {
                            if (pop_thresh > 0 &&
                                (unsigned int)rand_r(&_gpu_noise_seed) < pop_thresh) {
                                if (exc_only) {
                                    h_input[neuron_idx] += npulse;
                                } else {
                                    float sign_r = (float)rand_r(&_gpu_noise_seed)
                                                 * (1.0f / (float)RAND_MAX);
                                    if (sign_r < ei_ratio) {
                                        h_input[neuron_idx] -= npulse;
                                    } else {
                                        h_input[neuron_idx] += npulse;
                                    }
                                }
                            }
                            neuron_idx++;
                        }
                    }
                }
            }

            /* Upload input to GPU tensor */
            clock_gettime(CLOCK_MONOTONIC, &_phase_t0);
            size_t dims[1] = { total_neurons };
            nimcp_gpu_tensor_t* input_tensor = nimcp_gpu_tensor_from_host(
                gpu, h_input, dims, 1, NIMCP_GPU_PRECISION_FP32);

            if (input_tensor) {
                /* Run GPU LIF forward pass */
                bool gpu_ok = nimcp_gpu_lif_forward(gpu, lif_state, input_tensor);

                if (gpu_ok && lif_state->spikes) {
                    /* Read back spikes from GPU */
                    float* h_spikes = (float*)nimcp_malloc(total_neurons * sizeof(float));
                    if (h_spikes) {
                        nimcp_gpu_tensor_to_host(lif_state->spikes, h_spikes);

                        /* Distribute GPU results back to population tensors and neural_net */
                        size_t neuron_offset = 0;
                        for (uint32_t p = 0; p < network->n_populations; p++) {
                            snn_population_t* pop = network->populations[p];
                            if (!pop) continue;

                            float* spike_data = (float*)nimcp_tensor_data(pop->spike_output);
                            float* v_data = (float*)nimcp_tensor_data(pop->membrane_v);

                            for (uint32_t n = 0; n < pop->n_neurons; n++) {
                                float spiked = h_spikes[neuron_offset + n];
                                spike_data[n] = spiked;

                                if (spiked > 0.5f) {
                                    record_spike(&pop->spike_trains[n], network->sim->current_time_us);
                                    total_spikes++;
                                    pop->total_spikes++;

                                    if (!pop->lightweight && network->neural_net) {
                                        neuron_t* nn = neural_network_get_neuron(
                                            network->neural_net, pop->neuron_ids[n]);
                                        if (nn) nn->state = 1.0f;
                                    }
                                } else {
                                    if (!pop->lightweight && network->neural_net) {
                                        neuron_t* nn = neural_network_get_neuron(
                                            network->neural_net, pop->neuron_ids[n]);
                                        if (nn) nn->state = 0.0f;
                                    }
                                }
                            }

                            /* Sync membrane potential back from GPU */
                            if (lif_state->v) {
                                float* h_v = (float*)nimcp_malloc(total_neurons * sizeof(float));
                                if (h_v) {
                                    nimcp_gpu_tensor_to_host(lif_state->v, h_v);
                                    for (uint32_t n = 0; n < pop->n_neurons; n++) {
                                        v_data[n] = h_v[neuron_offset + n];
                                    }
                                    nimcp_free(h_v);
                                }
                            }

                            neuron_offset += pop->n_neurons;
                        }
                        nimcp_free(h_spikes);
                        gpu_executed = true;
                    }
                } else if (!gpu_ok) {
                    NIMCP_LOGGING_WARN("snn_network_step: GPU LIF forward failed, falling back to CPU");
                }

                nimcp_gpu_tensor_destroy(input_tensor);
            }
            clock_gettime(CLOCK_MONOTONIC, &_phase_t1);
            _lif_ms = (_phase_t1.tv_sec - _phase_t0.tv_sec) * 1000.0
                    + (_phase_t1.tv_nsec - _phase_t0.tv_nsec) / 1e6;
            nimcp_free(h_input);
        }

        /* Clear external_current for lightweight pops regardless of GPU success */
        for (uint32_t p = 0; p < network->n_populations; p++) {
            snn_population_t* pop = network->populations[p];
            if (pop && pop->lightweight && pop->external_current) {
                memset(pop->external_current, 0, pop->n_neurons * sizeof(float));
            }
        }
    }

    if (!gpu_executed) {
    /* ===== CPU FALLBACK PATH ===== */

    /* Substrate adapter: cache was already refreshed at the top of
     * snn_network_step (so GPU path gets the same treatment). We just
     * need to pick up the cached pointers here for tau_eff / tref_eff
     * / spike-survival — the three CPU-only per-neuron hooks. */
    const axon_substrate_effects_t*     se_axon = NULL;
    const dendrite_substrate_effects_t* se_dend = NULL;
    if (network->substrate && snn_tune_get_substrate_enabled() != 0.0f) {
        se_axon = &network->cached_axon_effects;
        se_dend = &network->cached_dend_effects;
    }
    const int substrate_spike_dropout_on =
        (snn_tune_get_substrate_spike_dropout_on() != 0.0f);

    /* Process each population */
    for (uint32_t p = 0; p < network->n_populations; p++) {
        snn_population_t* pop = network->populations[p];
        if (!pop) continue;

        float* v_data = (float*)nimcp_tensor_data(pop->membrane_v);
        float* spike_data = (float*)nimcp_tensor_data(pop->spike_output);
        float* ref_data = (float*)nimcp_tensor_data(pop->refractory);

        float v_thresh = network->config.v_thresh;
        float v_reset = network->config.v_reset;
        float v_rest = network->config.v_rest;
        float tau_mem = network->config.tau_mem;

        /* Substrate-modulated tau (dendritic membrane time constant) and
         * refractory period (axonal Na+/K+ pump recovery). When no
         * substrate is attached these collapse to the config values. */
        float tau_eff  = se_dend ? substrate_apply_tau(tau_mem, se_dend) : tau_mem;
        float tref_eff = se_axon ? substrate_apply_tref(network->config.t_ref, se_axon)
                                 : network->config.t_ref;
        /* Emergency silence: when axonal capacity collapses (ATP or ion
         * gradient critically low), the neuron cannot propagate an AP.
         * We skip the entire LIF update for this step on that pop. */
        const int substrate_emergency_silence =
            (se_axon && se_axon->overall_capacity < 0.1f);

        /* Fetch Poisson noise parameters ONCE per population to avoid
         * the extern-call overhead inside the inner per-neuron loop.
         * Apply per-pop adaptive factor (dead=1.0, at-target=0.0). */
        extern float snn_tune_get_noise_rate_hz(void);
        extern float snn_tune_get_noise_pulse_mv(void);
        extern float snn_noise_factor_for_pop(const snn_population_t*);
        const float noise_rate_hz = snn_tune_get_noise_rate_hz();
        const float noise_pulse_mv = snn_tune_get_noise_pulse_mv();
        const float noise_factor = snn_noise_factor_for_pop(pop);
        /* Per-step spike probability: p = rate_hz × dt_s × adaptive_factor. */
        const float noise_p = noise_rate_hz * dt_ms * 0.001f * noise_factor;
        /* Thread-local PRNG seed. Seeded once on first call from
         * clock-based entropy + a per-thread spread so different worker
         * threads don't produce correlated noise. */
        static __thread unsigned int _noise_seed = 0;
        if (_noise_seed == 0) {
            struct timespec _t;
            clock_gettime(CLOCK_MONOTONIC, &_t);
            _noise_seed = (unsigned int)(_t.tv_nsec ^ ((uintptr_t)&_noise_seed >> 4));
            if (_noise_seed == 0) _noise_seed = 0xDEADBEEFu;
        }

        /* Update each neuron */
        if (pop->lightweight && pop->incoming_csr && pop->incoming_csr->finalized) {
            /* ===== LIGHTWEIGHT CSR PATH ===== */

            /* Biophysical stability mechanisms — fetch knobs once per pop.
             *
             * AHP + pump: spike-triggered hyperpolarizing currents. Bulk-
             *   compute the per-neuron contribution into pre-allocated
             *   buffers before the inner loop; subtract from I_syn inside
             *   the loop; update state (decay + spike bump) after the loop.
             *
             * Basket pool: fast-spiking inhibitory interneurons. Uniform
             *   contribution (same mV to every parent neuron) so we compute
             *   one scalar before the loop. emit_inhibition uses the PREVIOUS
             *   basket step's mean spike output — the call to
             *   snn_basket_pool_step at the end of this pop's work advances
             *   it for next step.
             *
             * E/I noise: a fraction g_snn_noise_ei_ratio of Poisson pulses
             *   are negative. Exception: if pop is effectively dead
             *   (noise_factor >= 0.9) we force excitatory-only so silent
             *   pops can still escape the absorbing-zero state. */
            extern float snn_tune_get_ahp_enabled(void);
            extern float snn_tune_get_pump_enabled(void);
            extern float snn_tune_get_basket_enabled(void);
            extern float snn_tune_get_noise_ei_ratio(void);
            extern float snn_tune_get_ahp_pump_substrate_coupling(void);
            const int   ahp_on    = (pop->ahp  && snn_tune_get_ahp_enabled()  != 0.0f);
            const int   pump_on   = (pop->pump && snn_tune_get_pump_enabled() != 0.0f);
            const int   basket_on = (pop->basket && snn_tune_get_basket_enabled() != 0.0f);
            const float ei_ratio  = snn_tune_get_noise_ei_ratio();
            const int   noise_exc_only = (noise_factor >= 0.9f);  /* dead-pop exception */

            /* F8: biological feedback — scale AHP/pump gains by substrate
             * pump_activity (ATP proxy). Real Na/K-ATPase is ATP-dependent:
             * low ATP slows the pumps and weakens both AHP (pump-driven
             * component) and pump adaptation. Clamp to [0.1, 1.0] since
             * pumps slow but don't stop entirely. Gated on the dedicated
             * knob so users can disable the new coupling without losing
             * the existing substrate modulation of tau/tref/spike-survival.
             * When se_axon is NULL (no substrate attached) or the knob is
             * off, factor stays at 1.0 (identity). */
            float substrate_pump_factor = 1.0f;
            if (se_axon
                && snn_tune_get_ahp_pump_substrate_coupling() != 0.0f) {
                substrate_pump_factor = se_axon->pump_activity;
                if (substrate_pump_factor < 0.1f) substrate_pump_factor = 0.1f;
                if (substrate_pump_factor > 1.0f) substrate_pump_factor = 1.0f;
            }

            float* ahp_hyp  = ahp_on
                ? (float*)nimcp_calloc(pop->n_neurons, sizeof(float)) : NULL;
            float* pump_hyp = pump_on
                ? (float*)nimcp_calloc(pop->n_neurons, sizeof(float)) : NULL;
            if (ahp_hyp)  snn_adaptation_compute_hyperpol(pop->ahp,  ahp_hyp,  dt_ms);
            if (pump_hyp) snn_adaptation_compute_hyperpol(pop->pump, pump_hyp, dt_ms);

            /* Basket contribution is uniform across parent. Using
             * gain_inhib_to_parent × basket_mean_rate reproduces exactly
             * what snn_basket_pool_emit_inhibition would write per-neuron,
             * without allocating a temporary buffer. */
            float basket_contrib = 0.0f;
            if (basket_on) {
                basket_contrib = pop->basket->gain_inhib_to_parent
                               * snn_basket_pool_mean_rate(pop->basket);
            }

            for (uint32_t n = 0; n < pop->n_neurons; n++) {
                spike_data[n] = 0.0f;

                /* Substrate emergency silence: overall axon capacity has
                 * collapsed (ATP/ion critical). The neuron cannot
                 * generate or propagate an action potential this step.
                 * Decrement refractory so it continues to tick down
                 * correctly when the substrate recovers. */
                if (substrate_emergency_silence) {
                    if (ref_data[n] > 0.0f) ref_data[n] -= dt_ms;
                    continue;
                }

                if (ref_data[n] > 0.0f) {
                    ref_data[n] -= dt_ms;
                    continue;
                }

                /* I_syn from external_current + CSR incoming synapses */
                float I_syn = pop->external_current ? pop->external_current[n] : 0.0f;

                uint32_t syn_count;
                snn_csr_synapse_t* syns = snn_csr_get_incoming(
                    pop->incoming_csr, n, &syn_count);
                for (uint32_t s = 0; s < syn_count; s++) {
                    if (syns[s].src_pop >= network->n_populations) continue;
                    snn_population_t* src_pop = network->populations[syns[s].src_pop];
                    if (!src_pop || !src_pop->spike_output) continue;
                    float* src_spikes = (float*)nimcp_tensor_data(src_pop->spike_output);
                    if (src_spikes && syns[s].src_neuron < src_pop->n_neurons
                        && src_spikes[syns[s].src_neuron] > 0.5f) {
                        /* Short-term synaptic depression: scale contribution by
                         * (1 - depression[src_neuron]). Recently-firing sources
                         * deliver less drive, preventing hot-pathway runaway. */
                        float dep = 0.0f;
                        if (src_pop->depression) dep = src_pop->depression[syns[s].src_neuron];
                        I_syn += syns[s].weight * (1.0f - dep);
                    }
                }

                /* Basket feedforward inhibition — uniform across pop. */
                I_syn += basket_contrib;

                /* E/I-balanced Poisson background noise — structural fix
                 * for the absorbing-zero state. Dead pops (factor >= 0.9)
                 * receive only excitatory pulses to escape; balanced pops
                 * receive ei_ratio fraction as inhibitory so mean drive
                 * stays near zero and neurons sit fluctuation-driven
                 * instead of ramping up. */
                if (noise_p > 0.0f) {
                    float r = (float)rand_r(&_noise_seed) * (1.0f / (float)RAND_MAX);
                    if (r < noise_p) {
                        if (!noise_exc_only) {
                            float sign_r = (float)rand_r(&_noise_seed)
                                         * (1.0f / (float)RAND_MAX);
                            if (sign_r < ei_ratio) {
                                I_syn -= noise_pulse_mv;
                            } else {
                                I_syn += noise_pulse_mv;
                            }
                        } else {
                            I_syn += noise_pulse_mv;
                        }
                    }
                }

                /* AHP + pump hyperpolarization — subtract from drive.
                 * Numerically equivalent to subtracting from dV since
                 * dV = (v_rest - v + I_syn)/tau × dt. F8: the combined
                 * gain is scaled by substrate_pump_factor so metabolic
                 * state feeds back into spike-rate adaptation (weak pumps
                 * → weak adaptation → membrane integrates longer). */
                float hyp = 0.0f;
                if (ahp_hyp)  hyp += ahp_hyp[n];
                if (pump_hyp) hyp += pump_hyp[n];
                I_syn -= hyp * substrate_pump_factor;

                /* LIF dynamics — use substrate-modulated tau (tau_eff
                 * collapses to tau_mem when no substrate is attached). */
                float dv = (v_rest - v_data[n] + I_syn) / tau_eff * dt_ms;
                v_data[n] += dv;

                /* Intrinsic plasticity: effective threshold = config + per-neuron offset.
                 * offset is adjusted below based on firing vs target rate. */
                float v_thresh_n = v_thresh;
                if (pop->threshold_offset) v_thresh_n += pop->threshold_offset[n];

                if (v_data[n] >= v_thresh_n) {
                    /* Substrate spike-survival gate: a fraction of spikes
                     * drop at the axon hillock when reliability is low
                     * (weak AP amplitude, low ion gradient). When the
                     * substrate is absent or the knob is off, every
                     * threshold-crossing spike survives. When a spike
                     * fails to survive we do NOT set spike_data, do NOT
                     * reset v, and do NOT enter refractory — the
                     * membrane simply sits above threshold and the
                     * neuron will try again next step. */
                    int spike_ok = 1;
                    if (substrate_spike_dropout_on && se_axon) {
                        float rand01 = (float)rand_r(&_noise_seed) * (1.0f / (float)RAND_MAX);
                        if (!substrate_spike_survives(se_axon, rand01)) {
                            spike_ok = 0;
                        }
                    }
                    if (spike_ok) {
                        spike_data[n] = 1.0f;
                        v_data[n] = v_reset;
                        /* Substrate-modulated refractory period (tref_eff
                         * collapses to config.t_ref when no substrate). */
                        ref_data[n] = tref_eff;
                        record_spike(&pop->spike_trains[n], network->sim->current_time_us);
                        total_spikes++;
                        pop->total_spikes++;
                    }
                }
            }

            /* Advance AHP + pump state: decay adapt_var by exp(-dt/tau),
             * then bump on fresh spikes. Must run AFTER the fire decisions
             * above, since spike_data was just populated. */
            if (ahp_on)  snn_adaptation_update(pop->ahp,  spike_data, dt_ms);
            if (pump_on) snn_adaptation_update(pop->pump, spike_data, dt_ms);
            if (ahp_hyp)  nimcp_free(ahp_hyp);
            if (pump_hyp) nimcp_free(pump_hyp);

            /* Advance basket pool with this step's parent mean fire rate.
             * The next step's emit_inhibition will see the spikes this
             * basket step just produced. */
            if (basket_on) {
                uint32_t spike_count = 0;
                for (uint32_t n = 0; n < pop->n_neurons; n++) {
                    if (spike_data[n] > 0.5f) spike_count++;
                }
                float mean_rate = (pop->n_neurons > 0)
                    ? (float)spike_count / (float)pop->n_neurons : 0.0f;
                snn_basket_pool_step(pop->basket, mean_rate, dt_ms);
            }

            /* Intrinsic plasticity update: per-neuron threshold adaptation.
             * Each neuron's threshold drifts toward a value that would make
             * it fire at target_neuron_rate. Fast timescale (alpha=0.01)
             * catches runaway fast; slow drift (alpha=0.001 used elsewhere
             * for homeostasis) would take too long.
             * Short-term depression update: depressing factor exponentially
             * decays to 0 with tau ~50ms. On firing, depression jumps by
             * 0.2 (capped at 0.5) — effective weight is (1 - depression). */
            if (pop->threshold_offset && pop->neuron_rate_ema && pop->depression) {
                const float target_neuron_rate = 0.03f;   /* per step */
                const float ip_alpha = 0.01f;             /* rate EMA smoothing */
                const float ip_gain  = 0.5f;              /* mV per unit rate error */
                const float ip_cap   = 10.0f;             /* clamp offset ± */
                /* Short-term depression dynamics — runtime-tunable via
                 * snn_tune_set_depression_*. Fetch ONCE per pop step
                 * (not per neuron) to avoid extern-call overhead in the
                 * hot loop. dep_decay = exp(-dt_ms / tau) so tuning tau
                 * directly controls recovery timescale in ms. */
                extern float snn_tune_get_depression_inc(void);
                extern float snn_tune_get_depression_tau_ms(void);
                extern float snn_tune_get_depression_cap(void);
                const float dep_inc   = snn_tune_get_depression_inc();
                const float dep_tau   = snn_tune_get_depression_tau_ms();
                const float dep_cap   = snn_tune_get_depression_cap();
                const float dep_decay = expf(-dt_ms / dep_tau);

                /* === Phase 4.1 wiring: when the batch-safe flag is set,
                 * route through the batch-safe C functions (B=1 here).
                 * Behavior is identity-equivalent to the legacy inline loop,
                 * which lets us validate that the wiring produces identical
                 * trainning dynamics before enabling larger batches. */
                extern bool nimcp_snn_batch_safe_is_enabled(void);
                extern int nimcp_snn_scaling_apply_batch(float*, const float*, uint32_t, uint32_t, float);
                extern int nimcp_snn_depression_apply_batch(float*, const float*, uint32_t, uint32_t, float, float, float);
                extern int nimcp_snn_ip_apply_batch(float*, const float*, const float*, uint32_t, uint32_t, float, float, float);

                if (nimcp_snn_batch_safe_is_enabled()) {
                    /* Build a fire-vector for this step (B=1, N=n_neurons).
                     * Allocated on stack via VLA. */
                    float fired_buf[1024];  /* most pops fit; fallback to malloc if larger */
                    float* fired_vec = fired_buf;
                    bool heap_alloc = false;
                    if (pop->n_neurons > 1024) {
                        fired_vec = (float*)malloc(pop->n_neurons * sizeof(float));
                        heap_alloc = true;
                    }
                    if (fired_vec) {
                        for (uint32_t n = 0; n < pop->n_neurons; n++) {
                            fired_vec[n] = spike_data[n] > 0.5f ? 1.0f : 0.0f;
                        }
                        /* Scaling: rate EMA update */
                        nimcp_snn_scaling_apply_batch(
                            pop->neuron_rate_ema, fired_vec, 1, pop->n_neurons,
                            1.0f - ip_alpha);
                        /* IP: threshold adaptation (uses updated rate_ema) */
                        nimcp_snn_ip_apply_batch(
                            pop->threshold_offset, fired_vec,
                            pop->neuron_rate_ema, 1, pop->n_neurons,
                            ip_gain, target_neuron_rate, ip_cap);
                        /* Depression */
                        nimcp_snn_depression_apply_batch(
                            pop->depression, fired_vec, 1, pop->n_neurons,
                            dep_decay, dep_inc, dep_cap);
                        if (heap_alloc) free(fired_vec);
                    }
                } else {
                    /* Legacy sequential path (DEFAULT) */
                    for (uint32_t n = 0; n < pop->n_neurons; n++) {
                        /* Update per-neuron rate EMA */
                        float fired = spike_data[n] > 0.5f ? 1.0f : 0.0f;
                        pop->neuron_rate_ema[n] = (1.0f - ip_alpha) * pop->neuron_rate_ema[n]
                                                + ip_alpha * fired;
                        /* If rate > target, raise threshold (harder to fire).
                         * If rate < target, lower threshold (easier to fire). */
                        float err = pop->neuron_rate_ema[n] - target_neuron_rate;
                        pop->threshold_offset[n] += ip_gain * err;
                        if (pop->threshold_offset[n] > ip_cap) pop->threshold_offset[n] = ip_cap;
                        if (pop->threshold_offset[n] < -ip_cap) pop->threshold_offset[n] = -ip_cap;
                        /* Short-term depression decay + jump on firing */
                        pop->depression[n] *= dep_decay;
                        if (fired > 0.5f) {
                            pop->depression[n] += dep_inc;
                            if (pop->depression[n] > dep_cap) pop->depression[n] = dep_cap;
                        }
                    }
                }
            }

            /* Clear external_current for next step (input is per-step) */
            if (pop->external_current) {
                memset(pop->external_current, 0, pop->n_neurons * sizeof(float));
            }

            /* Report this pop's activity to the substrate. n_plasticity=0
             * here; any plasticity application downstream (homeostasis,
             * R-STDP) debits separately. region_id=0 since pop-level
             * region tagging is not yet plumbed. */
            if (network->substrate &&
                snn_tune_get_substrate_enabled() != 0.0f) {
                uint32_t pop_spikes_this_step = 0;
                for (uint32_t n = 0; n < pop->n_neurons; n++) {
                    if (spike_data[n] > 0.5f) pop_spikes_this_step++;
                }
                substrate_debit_activity(network->substrate,
                                         0 /* region_id */,
                                         pop_spikes_this_step,
                                         0 /* n_plasticity */);
            }
        } else {
        /* ===== LEGACY NEURON_T PATH =====
         * Bug #3 fix: mirror the lightweight path's substrate modulation.
         * Use tau_eff / tref_eff (already computed above in this pop's
         * loop iteration) instead of the raw config values so the
         * biological substrate feedback also modulates legacy pops. */
        for (uint32_t n = 0; n < pop->n_neurons; n++) {
            spike_data[n] = 0.0f;

            if (ref_data[n] > 0.0f) {
                ref_data[n] -= dt_ms;
                continue;
            }

            float I_syn = 0.0f;
            if (network->neural_net && n < pop->n_neurons) {
                neuron_t* neuron = neural_network_get_neuron(
                    network->neural_net, pop->neuron_ids[n]);
                if (neuron) {
                    I_syn = neuron->external_current;
                    uint32_t in_count = neuron->incoming.embedded_count
                                      + neuron->incoming.overflow_count;
                    for (uint32_t s = 0; s < in_count; s++) {
                        synapse_handle_t* h = sparse_synapse_get(&neuron->incoming, s);
                        if (!h) continue;
                        uint32_t pre_id = h->target_neuron_id;
                        neuron_t* pre = neural_network_get_neuron(network->neural_net, pre_id);
                        if (pre && pre->state > 0.5f) {
                            I_syn += h->weight;
                        }
                    }
                }
            }

            /* LIF dynamics — use substrate-modulated tau (tau_eff
             * collapses to tau_mem when no substrate is attached). */
            float dv = (v_rest - v_data[n] + I_syn) / tau_eff * dt_ms;
            v_data[n] += dv;

            if (v_data[n] >= v_thresh) {
                spike_data[n] = 1.0f;
                v_data[n] = v_reset;
                /* Refractory uses substrate-modulated tref_eff
                 * (collapses to config.t_ref when no substrate). */
                ref_data[n] = tref_eff;
                record_spike(&pop->spike_trains[n], network->sim->current_time_us);
                total_spikes++;
                pop->total_spikes++;

                if (network->neural_net) {
                    neuron_t* spiked = neural_network_get_neuron(
                        network->neural_net, pop->neuron_ids[n]);
                    if (spiked) spiked->state = 1.0f;
                }
            } else {
                if (network->neural_net) {
                    neuron_t* quiet = neural_network_get_neuron(
                        network->neural_net, pop->neuron_ids[n]);
                    if (quiet) quiet->state = 0.0f;
                }
            }
        }

        /* Report this pop's activity to the substrate (mirrors the
         * lightweight branch). Bug #3 fix: legacy neuron_t pops were
         * previously silent contributors — substrate never saw their
         * metabolic cost. */
        if (network->substrate &&
            snn_tune_get_substrate_enabled() != 0.0f) {
            uint32_t pop_spikes_this_step = 0;
            for (uint32_t n = 0; n < pop->n_neurons; n++) {
                if (spike_data[n] > 0.5f) pop_spikes_this_step++;
            }
            substrate_debit_activity(network->substrate,
                                     0 /* region_id */,
                                     pop_spikes_this_step,
                                     0 /* n_plasticity */);
        }
        } /* end legacy */
    }
    } /* end CPU fallback */

    /* ===== GPU-path substrate activity debit =====
     * Bug #2 fix: previously substrate_debit_activity only ran inside the
     * CPU fallback, so the GPU fast path never reported spike cost back
     * to the substrate. Result: ATP never drained, plasticity_mod stayed
     * pinned at ~1.0, and the metabolic feedback loop was silently open.
     * When the GPU executed the step, walk each population's spike_output
     * (already synced back to host above) and debit the substrate once
     * per pop. */
    if (gpu_executed && network->substrate &&
        snn_tune_get_substrate_enabled() != 0.0f) {
        for (uint32_t p = 0; p < network->n_populations; p++) {
            snn_population_t* pop = network->populations[p];
            if (!pop || !pop->spike_output) continue;
            const float* sp = (const float*)nimcp_tensor_data_const(
                pop->spike_output);
            if (!sp) continue;
            uint32_t pop_spikes_this_step = 0;
            for (uint32_t n = 0; n < pop->n_neurons; n++) {
                if (sp[n] > 0.5f) pop_spikes_this_step++;
            }
            substrate_debit_activity(network->substrate,
                                     0 /* region_id */,
                                     pop_spikes_this_step,
                                     0 /* n_plasticity */);
        }
    }

    /* Update simulation time */
    network->sim->current_time_us += dt_us;
    network->sim->step_count++;

    /* Per-population firing-rate EMA for homeostatic plasticity.
     * Counts fresh spike_output from this step and folds into an EMA
     * with alpha=0.01 (≈100-step time constant). The EMA is consumed
     * by snn_homeostatic_apply() which runs every N training steps. */
    for (uint32_t p = 0; p < network->n_populations; p++) {
        snn_population_t* pop = network->populations[p];
        if (!pop || !pop->lightweight || !pop->spike_output) continue;
        const float* sp = (const float*)nimcp_tensor_data_const(pop->spike_output);
        if (!sp) continue;
        uint32_t n_spk = 0;
        for (uint32_t n = 0; n < pop->n_neurons; n++) {
            if (sp[n] > 0.5f) n_spk++;
        }
        float rate = (pop->n_neurons > 0)
                   ? (float)n_spk / (float)pop->n_neurons : 0.0f;
        /* EMA with warm-up: for first 100 samples weight new data more */
        float alpha = (pop->rate_samples < 100) ? 0.05f : 0.01f;
        pop->firing_rate_ema = (1.0f - alpha) * pop->firing_rate_ema
                             + alpha * rate;
        pop->rate_samples++;

        /* Temporal history ring buffer — store per-step spike count.
         * Consumed by Python-side FFT/correlation analysis via the
         * snn_get_population_history RPC. */
        pop->spike_count_history[pop->history_write_idx] = n_spk;
        pop->history_write_idx = (pop->history_write_idx + 1) % SNN_POP_HISTORY_LEN;
        pop->history_total_steps++;
    }

    /* Update statistics */
    clock_gettime(CLOCK_MONOTONIC, &_step_t1);
    double step_ms = (_step_t1.tv_sec - _step_t0.tv_sec) * 1000.0
                   + (_step_t1.tv_nsec - _step_t0.tv_nsec) / 1e6;
    network->stats.total_steps = network->sim->step_count;
    network->stats.total_spikes += total_spikes;
    network->stats.total_compute_time_ms += step_ms;
    network->stats.avg_step_time_ms = network->stats.total_compute_time_ms
                                    / (double)network->stats.total_steps;
    network->stats.last_step_time_ms = step_ms;
    network->stats.last_isyn_time_ms = _isyn_ms;
    network->stats.last_step_gpu = gpu_executed;
    if (gpu_executed) {
        network->stats.gpu_steps++;
    } else {
        network->stats.cpu_fallback_steps++;
    }

    /* Log timing breakdown every step for profiling */
    {
        NIMCP_LOGGING_INFO("SNN step %llu: total=%.1fms (isyn=%.1fms lif+readback=%.1fms) "
                           "spikes=%d gpu=%s avg=%.1fms",
                           (unsigned long long)network->stats.total_steps,
                           step_ms, _isyn_ms, _lif_ms,
                           total_spikes, gpu_executed ? "yes" : "no",
                           network->stats.avg_step_time_ms);
    }

    /* === PER-POPULATION DIAGNOSTICS (NIMCP_DEBUG_SNN=1) ===
     * Every 100 steps, dump per-population spike counts, mean V, mean I_syn.
     * Tells us exactly which populations are firing, which are dead, and
     * why (flat V = no input current; threshold V = input adequate). */
    {
        static int _snn_dbg_checked = 0, _snn_dbg = 0;
        if (!_snn_dbg_checked) {
            const char* env = getenv("NIMCP_DEBUG_SNN");
            _snn_dbg = (env && env[0] == '1');
            _snn_dbg_checked = 1;
        }
        /* Allow on-demand triggering: another thread can set a flag (or
         * we re-check env var) to force an immediate dump. For now, the
         * env var NIMCP_SNN_DUMP_NOW=1 triggers a one-shot dump that
         * resets to 0 after firing. */
        bool dump_now = false;
        const char* dump_env = getenv("NIMCP_SNN_DUMP_NOW");
        if (dump_env && dump_env[0] == '1') {
            dump_now = true;
            unsetenv("NIMCP_SNN_DUMP_NOW");  /* one-shot */
        }
        if (_snn_dbg && ((network->stats.total_steps % 100) == 0 || dump_now)) {
            for (uint32_t p = 0; p < network->n_populations; p++) {
                snn_population_t* pop = network->populations[p];
                if (!pop || !pop->spike_output || !pop->membrane_v) continue;
                const float* spikes = (const float*)nimcp_tensor_data_const(pop->spike_output);
                const float* v = (const float*)nimcp_tensor_data_const(pop->membrane_v);
                if (!spikes || !v) continue;

                uint32_t n_spiked = 0;
                float v_sum = 0, v_min = 1e9, v_max = -1e9;
                for (uint32_t i = 0; i < pop->n_neurons; i++) {
                    if (spikes[i] > 0.5f) n_spiked++;
                    v_sum += v[i];
                    if (v[i] < v_min) v_min = v[i];
                    if (v[i] > v_max) v_max = v[i];
                }
                float v_mean = v_sum / (float)pop->n_neurons;

                /* External current stats (lightweight pops only) */
                float ext_mean = 0, ext_max = 0;
                if (pop->lightweight && pop->external_current) {
                    for (uint32_t i = 0; i < pop->n_neurons; i++) {
                        float ec = pop->external_current[i];
                        ext_mean += ec;
                        if (ec > ext_max) ext_max = ec;
                    }
                    ext_mean /= (float)pop->n_neurons;
                }

                /* Incoming CSR weight stats + computed I_syn from current spikes.
                 * I_syn = ext_current[n] + sum(weight × pre_spike) for each n.
                 * This is the EXPECTED I_syn this neuron will see next step,
                 * computed independently of GPU/CPU execution. Tells us
                 * whether signal is reaching this layer. */
                float w_mean = 0, w_max = 0;
                float isyn_mean = 0, isyn_max = 0;
                uint32_t n_syn = 0, n_neurons_with_isyn = 0;
                if (pop->lightweight && pop->incoming_csr && pop->incoming_csr->entries) {
                    n_syn = pop->incoming_csr->n_synapses;
                    for (uint32_t e = 0; e < n_syn; e++) {
                        float w = pop->incoming_csr->entries[e].weight;
                        w_mean += w;
                        if (w > w_max) w_max = w;
                    }
                    if (n_syn > 0) w_mean /= (float)n_syn;

                    /* Compute I_syn per neuron from pre-spikes × weights */
                    snn_csr_storage_t* csr = pop->incoming_csr;
                    if (csr->row_ptr) {
                        for (uint32_t n = 0; n < pop->n_neurons; n++) {
                            float isyn = pop->external_current ? pop->external_current[n] : 0.0f;
                            uint32_t s_lo = csr->row_ptr[n];
                            uint32_t s_hi = csr->row_ptr[n + 1];
                            for (uint32_t e = s_lo; e < s_hi; e++) {
                                snn_csr_synapse_t* syn = &csr->entries[e];
                                if (syn->src_pop >= network->n_populations) continue;
                                snn_population_t* src = network->populations[syn->src_pop];
                                if (!src || !src->spike_output) continue;
                                if (syn->src_neuron >= src->n_neurons) continue;
                                /* Verify tensor dtype before casting — non-lightweight
                                 * pops may use different dtypes (audit bug #2) */
                                if (nimcp_tensor_dtype(src->spike_output) != NIMCP_DTYPE_F32)
                                    continue;
                                const float* src_sp =
                                    (const float*)nimcp_tensor_data_const(src->spike_output);
                                if (src_sp && src_sp[syn->src_neuron] > 0.5f) {
                                    isyn += syn->weight;
                                }
                            }
                            isyn_mean += isyn;
                            if (isyn > isyn_max) isyn_max = isyn;
                            if (isyn != 0.0f) n_neurons_with_isyn++;
                        }
                        if (pop->n_neurons > 0) isyn_mean /= (float)pop->n_neurons;
                    }
                }

                NIMCP_LOGGING_INFO(
                    "[SNN-POP] %s n=%u spk=%u V[%.1f..%.1f]μ=%.1f "
                    "ext[μ=%.2f max=%.2f] w[μ=%.3f max=%.3f] "
                    "I_syn[μ=%.3f max=%.2f n_recv=%u/%u] nsyn=%u",
                    pop->name, pop->n_neurons, n_spiked,
                    v_min, v_max, v_mean,
                    ext_mean, ext_max, w_mean, w_max,
                    isyn_mean, isyn_max, n_neurons_with_isyn, pop->n_neurons,
                    n_syn);
            }

            /* Per-tier propagation summary: aggregate spike counts by name prefix */
            uint32_t tier_spikes[10] = {0};
            uint32_t tier_neurons[10] = {0};
            const char* tier_names[10] = {
                "input", "L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7", "output"
            };
            for (uint32_t p = 0; p < network->n_populations; p++) {
                snn_population_t* pop = network->populations[p];
                if (!pop || !pop->spike_output) continue;
                const float* sp = (const float*)nimcp_tensor_data_const(pop->spike_output);
                if (!sp) continue;
                int tier = -1;
                if (strncmp(pop->name, "input", 5) == 0) tier = 0;
                else if (strncmp(pop->name, "output", 6) == 0) tier = 9;
                /* Strict L<digit>_ pattern — avoid matching "L10..." or "Ldifferent" */
                else if (pop->name[0] == 'L' && pop->name[1] >= '0' && pop->name[1] <= '7'
                         && (pop->name[2] == '\0' || pop->name[2] == '_'))
                    tier = (pop->name[1] - '0') + 1;  /* L0→1, L1→2, ... L7→8 */
                if (tier < 0 || tier >= 10) continue;
                uint32_t s = 0;
                for (uint32_t i = 0; i < pop->n_neurons; i++) if (sp[i] > 0.5f) s++;
                tier_spikes[tier] += s;
                tier_neurons[tier] += pop->n_neurons;
            }
            char tier_summary[512] = {0};
            int off = 0;
            for (int t = 0; t < 10; t++) {
                if (tier_neurons[t] == 0) continue;
                int n = snprintf(tier_summary + off, sizeof(tier_summary) - off,
                    "%s%s=%u/%u", off > 0 ? " " : "", tier_names[t],
                    tier_spikes[t], tier_neurons[t]);
                if (n > 0) off += n;
                if (off >= (int)sizeof(tier_summary) - 1) break;
            }
            NIMCP_LOGGING_INFO("[SNN-TIER] %s", tier_summary);
        }
    }

    return total_spikes;
}

/**
 * @brief Compute per-population derived firing-rate stats and write to network->stats.
 *
 * Shared helper so both the inference path (snn_network_run) and the BPTT
 * training path (snn_backprop_forward in src/training/nimcp_snn_backprop.c)
 * can update the same visible metrics. Before this was extracted, only
 * snn_network_run updated network->stats, so get_snn_stats RPC reflected
 * inference activity exclusively — training-time spiking was invisible.
 * Discovered Apr 11 2026 while verifying the unroll_steps=1000 fix.
 *
 * @param network     The network whose populations have just been stepped.
 * @param total_spikes Total spike count from the run (sum across populations).
 * @param duration_ms  Simulated duration in milliseconds (for Hz conversion).
 */
void snn_network_update_stats(snn_network_t* network, int total_spikes, float duration_ms) {
    if (!network) return;

    float duration_s = duration_ms / 1000.0f;
    uint32_t total_neurons = 0;
    uint32_t silent = 0;
    uint32_t hyperactive = 0;
    float max_rate = 0.0f;

    for (uint32_t p = 0; p < network->n_populations; p++) {
        snn_population_t* pop = network->populations[p];
        if (!pop) continue;

        /* Per-neuron firing rate from spike trains */
        for (uint32_t n = 0; n < pop->n_neurons; n++) {
            float rate = (duration_s > 0.0f) ?
                (float)pop->spike_trains[n].total_spikes / duration_s : 0.0f;
            if (rate == 0.0f) silent++;
            if (rate > 200.0f) hyperactive++;  /* >200 Hz is pathological */
            if (rate > max_rate) max_rate = rate;
        }

        /* Population mean rate */
        pop->mean_rate = (duration_s > 0.0f && pop->n_neurons > 0) ?
            (float)pop->total_spikes / (float)pop->n_neurons / duration_s : 0.0f;
        total_neurons += pop->n_neurons;
    }

    if (total_neurons > 0) {
        network->stats.mean_firing_rate = (duration_s > 0.0f) ?
            (float)total_spikes / (float)total_neurons / duration_s : 0.0f;
        network->stats.max_firing_rate = max_rate;
        network->stats.sparsity = (float)silent / (float)total_neurons;
        network->stats.silent_neurons = silent;
        network->stats.hyperactive_neurons = hyperactive;
        network->stats.spikes_per_sample = (float)total_spikes / (float)total_neurons;
    }
}

int snn_network_run(snn_network_t* network, float duration_ms) {
    if (!network) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "snn_network_run: null network pointer");
        return SNN_ERROR_NULL_POINTER;
    }

    float dt_ms = network->config.dt;
    int n_steps = (int)(duration_ms / dt_ms);
    int total_spikes = 0;

    for (int i = 0; i < n_steps; i++) {
        int spikes = snn_network_step(network, dt_ms);
        if (spikes < 0) {
            return spikes;  /* Error code */
        }
        total_spikes += spikes;
    }

    snn_network_update_stats(network, total_spikes, duration_ms);
    return total_spikes;
}

//=============================================================================
// Performance-Optimized Stepping
//=============================================================================

/** Default threshold margin for sparse stepping (mV) */
#define SNN_SPARSE_THRESHOLD_MARGIN_DEFAULT 5.0f

/**
 * @brief Spike-driven sparse step — skip quiescent neurons
 *
 * WHAT: Only update neurons that are near threshold or received input
 * WHY:  At 2-5% firing rate, most neurons decay passively → skip them
 * HOW:  Two-pass: (1) identify active neurons, (2) update only those
 *
 * A neuron is "active" if any of:
 * - In refractory period (needs countdown)
 * - Has non-zero external_current (received a spike)
 * - Membrane potential is within threshold_margin of v_thresh
 * - Was spiking last step (needs reset propagation)
 */
int snn_network_step_sparse(snn_network_t* network, float dt,
                             float threshold_margin,
                             snn_step_stats_t* stats) {
    if (!network) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
            "snn_network_step_sparse: null network pointer");
        return SNN_ERROR_NULL_POINTER;
    }
    if (!network->sim) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_STATE,
            "snn_network_step_sparse: null simulation context");
        return SNN_ERROR_INVALID_STATE;
    }

    snn_network_heartbeat("snn_step_sparse", 0.0f);

    float dt_ms = (dt > 0.0f) ? dt : network->config.dt;
    uint64_t dt_us = (uint64_t)(dt_ms * 1000.0f);
    float margin = (threshold_margin > 0.0f) ? threshold_margin
                                              : SNN_SPARSE_THRESHOLD_MARGIN_DEFAULT;

    int total_spikes = 0;
    uint32_t total_neurons = 0;
    uint32_t neurons_updated = 0;
    uint32_t neurons_skipped = 0;
    uint32_t neurons_refractory = 0;

    float v_thresh = network->config.v_thresh;
    float v_reset = network->config.v_reset;
    float v_rest = network->config.v_rest;
    float tau_mem = network->config.tau_mem;
    float active_threshold = v_thresh - margin;

    for (uint32_t p = 0; p < network->n_populations; p++) {
        snn_population_t* pop = network->populations[p];
        if (!pop) continue;

        float* v_data = (float*)nimcp_tensor_data(pop->membrane_v);
        float* spike_data = (float*)nimcp_tensor_data(pop->spike_output);
        float* ref_data = (float*)nimcp_tensor_data(pop->refractory);

        total_neurons += pop->n_neurons;

        for (uint32_t n = 0; n < pop->n_neurons; n++) {
            spike_data[n] = 0.0f;

            /* Refractory neurons: just decrement, count as refractory */
            if (ref_data[n] > 0.0f) {
                ref_data[n] -= dt_ms;
                neurons_refractory++;
                continue;
            }

            /* Get synaptic input */
            float I_syn = 0.0f;
            if (network->neural_net) {
                neuron_t* neuron = neural_network_get_neuron(
                    network->neural_net, pop->neuron_ids[n]);
                if (neuron) {
                    I_syn = neuron->external_current;
                }
            }

            /* Sparse optimization: skip neuron if quiescent
             * - No synaptic input AND
             * - Membrane potential far below threshold (passive decay only)
             *
             * For passive decay: dV = (v_rest - V) / tau * dt
             * If V < active_threshold and I_syn == 0, the neuron is decaying
             * toward rest and won't spike. Apply the decay analytically. */
            if (I_syn == 0.0f && v_data[n] < active_threshold) {
                /* Analytical exponential decay: V → v_rest + (V-v_rest)*exp(-dt/tau) */
                float alpha = expf(-dt_ms / tau_mem);
                v_data[n] = v_rest + (v_data[n] - v_rest) * alpha;
                neurons_skipped++;
                continue;
            }

            /* Full LIF update for active neurons */
            neurons_updated++;
            float dv = (v_rest - v_data[n] + I_syn) / tau_mem * dt_ms;
            v_data[n] += dv;

            /* Spike generation */
            if (v_data[n] >= v_thresh) {
                spike_data[n] = 1.0f;
                v_data[n] = v_reset;
                ref_data[n] = network->config.t_ref;

                record_spike(&pop->spike_trains[n], network->sim->current_time_us);
                total_spikes++;
                pop->total_spikes++;
            }
        }
    }

    /* Update simulation time */
    network->sim->current_time_us += dt_us;
    network->sim->step_count++;
    network->stats.total_steps = network->sim->step_count;
    network->stats.total_spikes += total_spikes;

    /* Fill statistics if requested */
    if (stats) {
        stats->total_neurons = total_neurons;
        stats->neurons_updated = neurons_updated;
        stats->neurons_skipped = neurons_skipped;
        stats->neurons_refractory = neurons_refractory;
        stats->spikes_generated = (uint32_t)total_spikes;
        stats->compute_ratio = (total_neurons > 0)
            ? (float)neurons_updated / (float)total_neurons
            : 0.0f;
    }

    return total_spikes;
}

/**
 * @brief Worker context for parallel population stepping
 */
typedef struct {
    snn_population_t* pop;
    neural_network_t neural_net;
    float dt_ms;
    float v_thresh;
    float v_reset;
    float v_rest;
    float tau_mem;
    float t_ref;
    uint64_t current_time_us;
    int spikes;  /* output */
} snn_pop_step_ctx_t;

/**
 * @brief Step a single population (thread worker function)
 */
static void snn_pop_step_worker(snn_pop_step_ctx_t* ctx) {
    snn_population_t* pop = ctx->pop;
    if (!pop) { ctx->spikes = 0; return; }

    float* v_data = (float*)nimcp_tensor_data(pop->membrane_v);
    float* spike_data = (float*)nimcp_tensor_data(pop->spike_output);
    float* ref_data = (float*)nimcp_tensor_data(pop->refractory);

    int spikes = 0;

    for (uint32_t n = 0; n < pop->n_neurons; n++) {
        spike_data[n] = 0.0f;

        if (ref_data[n] > 0.0f) {
            ref_data[n] -= ctx->dt_ms;
            continue;
        }

        float I_syn = 0.0f;
        if (ctx->neural_net) {
            neuron_t* neuron = neural_network_get_neuron(
                ctx->neural_net, pop->neuron_ids[n]);
            if (neuron) {
                I_syn = neuron->external_current;
            }
        }

        float dv = (ctx->v_rest - v_data[n] + I_syn) / ctx->tau_mem * ctx->dt_ms;
        v_data[n] += dv;

        if (v_data[n] >= ctx->v_thresh) {
            spike_data[n] = 1.0f;
            v_data[n] = ctx->v_reset;
            ref_data[n] = ctx->t_ref;
            record_spike(&pop->spike_trains[n], ctx->current_time_us);
            spikes++;
            pop->total_spikes++;
        }
    }

    ctx->spikes = spikes;
}

/**
 * @brief Population-parallel step using pthreads
 *
 * WHAT: Step independent populations concurrently
 * WHY:  Populations within a step don't have cross-dependencies
 * HOW:  Launch one thread per population (up to n_threads), join all
 */
int snn_network_step_parallel(snn_network_t* network, float dt,
                               uint32_t n_threads) {
    if (!network) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
            "snn_network_step_parallel: null network pointer");
        return SNN_ERROR_NULL_POINTER;
    }
    if (!network->sim) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_STATE,
            "snn_network_step_parallel: null simulation context");
        return SNN_ERROR_INVALID_STATE;
    }

    snn_network_heartbeat("snn_step_parallel", 0.0f);

    float dt_ms = (dt > 0.0f) ? dt : network->config.dt;
    uint64_t dt_us = (uint64_t)(dt_ms * 1000.0f);

    uint32_t n_pops = network->n_populations;
    if (n_pops == 0) return 0;

    /* For single population or 1 thread, fall back to serial */
    uint32_t actual_threads = (n_threads > 0) ? n_threads : network->config.n_threads;
    if (actual_threads == 0) actual_threads = 4;  /* Default */
    if (n_pops == 1 || actual_threads <= 1) {
        return snn_network_step(network, dt);
    }

    /* Cap threads to population count */
    if (actual_threads > n_pops) actual_threads = n_pops;

    /* Prepare per-population contexts */
    snn_pop_step_ctx_t* ctxs = (snn_pop_step_ctx_t*)nimcp_malloc(
        n_pops * sizeof(snn_pop_step_ctx_t));
    if (!ctxs) return snn_network_step(network, dt);  /* Fallback */

    for (uint32_t p = 0; p < n_pops; p++) {
        ctxs[p].pop = network->populations[p];
        ctxs[p].neural_net = network->neural_net;
        ctxs[p].dt_ms = dt_ms;
        ctxs[p].v_thresh = network->config.v_thresh;
        ctxs[p].v_reset = network->config.v_reset;
        ctxs[p].v_rest = network->config.v_rest;
        ctxs[p].tau_mem = network->config.tau_mem;
        ctxs[p].t_ref = network->config.t_ref;
        ctxs[p].current_time_us = network->sim->current_time_us;
        ctxs[p].spikes = 0;
    }

    /* Launch threads for populations (use nimcp_thread API) */
    nimcp_thread_t* threads = (nimcp_thread_t*)nimcp_malloc(
        actual_threads * sizeof(nimcp_thread_t));
    if (!threads) {
        /* Fallback: run serially */
        for (uint32_t p = 0; p < n_pops; p++) {
            snn_pop_step_worker(&ctxs[p]);
        }
    } else {
        /* Round-robin populations to threads in batches */
        uint32_t launched = 0;
        uint32_t p = 0;

        while (p < n_pops) {
            launched = 0;
            uint32_t batch = (n_pops - p < actual_threads) ? (n_pops - p) : actual_threads;

            for (uint32_t t = 0; t < batch; t++) {
                int rc = nimcp_thread_create(&threads[t],
                    (void*(*)(void*))snn_pop_step_worker, &ctxs[p + t], NULL);
                if (rc != 0) {
                    /* Thread creation failed — run serially */
                    snn_pop_step_worker(&ctxs[p + t]);
                } else {
                    launched++;
                }
            }

            /* Join all launched threads */
            for (uint32_t t = 0; t < batch; t++) {
                if (launched > 0) {
                    nimcp_thread_join(threads[t], NULL);
                }
            }
            p += batch;
        }
        nimcp_free(threads);
    }

    /* Accumulate results */
    int total_spikes = 0;
    for (uint32_t p = 0; p < n_pops; p++) {
        total_spikes += ctxs[p].spikes;
    }
    nimcp_free(ctxs);

    /* Update simulation time */
    network->sim->current_time_us += dt_us;
    network->sim->step_count++;
    network->stats.total_steps = network->sim->step_count;
    network->stats.total_spikes += total_spikes;

    return total_spikes;
}

int snn_network_run_sparse(snn_network_t* network, float duration_ms,
                            float threshold_margin,
                            snn_step_stats_t* stats) {
    if (!network) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
            "snn_network_run_sparse: null network pointer");
        return SNN_ERROR_NULL_POINTER;
    }

    float dt_ms = network->config.dt;
    int n_steps = (int)(duration_ms / dt_ms);
    int total_spikes = 0;

    /* Accumulate stats across steps */
    snn_step_stats_t acc = {0};

    for (int i = 0; i < n_steps; i++) {
        snn_step_stats_t step_stats = {0};
        int spikes = snn_network_step_sparse(network, dt_ms,
                                              threshold_margin, &step_stats);
        if (spikes < 0) return spikes;
        total_spikes += spikes;

        acc.total_neurons = step_stats.total_neurons;  /* Same each step */
        acc.neurons_updated += step_stats.neurons_updated;
        acc.neurons_skipped += step_stats.neurons_skipped;
        acc.neurons_refractory += step_stats.neurons_refractory;
        acc.spikes_generated += step_stats.spikes_generated;
    }

    if (stats && n_steps > 0) {
        stats->total_neurons = acc.total_neurons;
        stats->neurons_updated = acc.neurons_updated / (uint32_t)n_steps;
        stats->neurons_skipped = acc.neurons_skipped / (uint32_t)n_steps;
        stats->neurons_refractory = acc.neurons_refractory / (uint32_t)n_steps;
        stats->spikes_generated = acc.spikes_generated;
        stats->compute_ratio = (acc.total_neurons > 0)
            ? (float)acc.neurons_updated /
              (float)(acc.total_neurons * (uint32_t)n_steps)
            : 0.0f;
    }

    /* Update network-level stats so get_snn_stats reflects sparse-path runs too. */
    snn_network_update_stats(network, total_spikes, duration_ms);
    return total_spikes;
}

//=============================================================================
// Input/Output
//=============================================================================

int snn_network_set_inputs(snn_network_t* network,
                           const float* inputs,
                           uint32_t n_inputs) {
    if (!network) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "snn_network_set_inputs: null network pointer");
        return SNN_ERROR_NULL_POINTER;
    }
    if (!inputs) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "snn_network_set_inputs: null inputs pointer");
        return SNN_ERROR_NULL_POINTER;
    }
    if (!network->input_pop) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_STATE, "snn_network_set_inputs: null input population");
        return SNN_ERROR_INVALID_STATE;
    }
    if (n_inputs != network->input_pop->n_neurons) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_SIZE,
            "snn_network_set_inputs: dimension mismatch");
        NIMCP_LOGGING_ERROR("snn_network_set_inputs: dimension mismatch (%u != %u)",
                           n_inputs, network->input_pop->n_neurons);
        return SNN_ERROR_INVALID_DIMENSION;
    }

    /* For now, set inputs as external currents on input neurons */
    /* A full implementation would use the encoder to generate spikes */
    /* Lightweight path: write directly to population's external_current array */
    if (network->input_pop->lightweight && network->input_pop->external_current) {
        float scale = network->config.input_current_scale;
        for (uint32_t i = 0; i < n_inputs && i < network->input_pop->n_neurons; i++) {
            float inp = inputs[i];
            network->input_pop->external_current[i] = (inp > 0.0f) ? inp * scale : 0.0f;
        }
        return SNN_SUCCESS;
    }

    /* Legacy path: write to neuron_t external_current */
    for (uint32_t i = 0; i < n_inputs; i++) {
        if (network->neural_net && i < network->config.n_inputs) {
            neuron_t* neuron = neural_network_get_neuron(
                network->neural_net, network->input_pop->neuron_ids[i]);
            if (neuron) {
                float inp = inputs[i];
                if (inp > 0.0f) {
                    neuron->external_current = inp * network->config.input_current_scale;
                } else {
                    neuron->external_current = 0.0f;
                }
            }
        }
    }

    return SNN_SUCCESS;
}

int snn_network_set_input_tensor(snn_network_t* network,
                                 const nimcp_tensor_t* input) {
    if (!network) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "snn_network_set_input_tensor: null network pointer");
        return SNN_ERROR_NULL_POINTER;
    }
    if (!input) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "snn_network_set_input_tensor: null input tensor");
        return SNN_ERROR_NULL_POINTER;
    }

    const float* data = (const float*)nimcp_tensor_data((nimcp_tensor_t*)input);
    uint64_t size = nimcp_tensor_numel(input);

    return snn_network_set_inputs(network, data, (uint32_t)size);
}

int snn_network_get_outputs(snn_network_t* network,
                            float* outputs,
                            uint32_t n_outputs) {
    if (!network) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "snn_network_get_outputs: null network pointer");
        return SNN_ERROR_NULL_POINTER;
    }
    if (!outputs) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "snn_network_get_outputs: null outputs pointer");
        return SNN_ERROR_NULL_POINTER;
    }
    if (!network->output_pop) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_STATE, "snn_network_get_outputs: null output population");
        return SNN_ERROR_INVALID_STATE;
    }
    if (n_outputs != network->output_pop->n_neurons) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_SIZE,
            "snn_network_get_outputs: dimension mismatch");
        return SNN_ERROR_INVALID_DIMENSION;
    }

    /* Rate decoding: count spikes in time window */
    float window_ms = network->config.decoder.time_window;
    uint64_t current_time = network->sim->current_time_us;

    for (uint32_t i = 0; i < n_outputs; i++) {
        outputs[i] = compute_firing_rate(
            &network->output_pop->spike_trains[i],
            current_time,
            window_ms);
    }

    return SNN_SUCCESS;
}

int snn_network_get_output_tensor(snn_network_t* network,
                                  nimcp_tensor_t* output) {
    if (!network) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "snn_network_get_output_tensor: null network pointer");
        return SNN_ERROR_NULL_POINTER;
    }
    if (!output) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "snn_network_get_output_tensor: null output tensor");
        return SNN_ERROR_NULL_POINTER;
    }

    float* data = (float*)nimcp_tensor_data(output);
    uint64_t size = nimcp_tensor_numel(output);

    return snn_network_get_outputs(network, data, (uint32_t)size);
}

int snn_network_forward(snn_network_t* network,
                        const float* inputs,
                        uint32_t n_inputs,
                        float* outputs,
                        uint32_t n_outputs,
                        float duration_ms) {
    if (!network) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "snn_network_forward: null network pointer");
        return SNN_ERROR_NULL_POINTER;
    }
    if (!inputs) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "snn_network_forward: null inputs pointer");
        return SNN_ERROR_NULL_POINTER;
    }
    if (!outputs) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "snn_network_forward: null outputs pointer");
        return SNN_ERROR_NULL_POINTER;
    }

    /* Phase 8: Send heartbeat at start of forward pass */
    snn_network_heartbeat("snn_forward", 0.0f);

    /* Reset network state */
    int result = snn_network_reset(network);
    if (result != SNN_SUCCESS) return result;

    /* Set inputs */
    result = snn_network_set_inputs(network, inputs, n_inputs);
    if (result != SNN_SUCCESS) return result;

    /* Run simulation */
    result = snn_network_run(network, duration_ms);
    if (result < 0) return result;

    /* Get outputs */
    return snn_network_get_outputs(network, outputs, n_outputs);
}

//=============================================================================
// Training
//=============================================================================

int snn_network_set_training(snn_network_t* network, bool training) {
    if (!network) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "snn_network_set_training: null network pointer");
        return SNN_ERROR_NULL_POINTER;
    }

    network->is_training = training;

    if (training && !network->train_ctx) {
        /* Allocate training context */
        network->train_ctx = (snn_training_ctx_t*)nimcp_malloc(sizeof(snn_training_ctx_t));
        if (!network->train_ctx) return SNN_ERROR_OUT_OF_MEMORY;

        memset(network->train_ctx, 0, sizeof(snn_training_ctx_t));
        network->train_ctx->mode = network->config.train_mode;
        network->train_ctx->surrogate = network->config.surrogate;
        network->train_ctx->surrogate_beta = network->config.surrogate_beta;
        network->train_ctx->eligibility_decay = 0.99f;
    }

    return SNN_SUCCESS;
}

int snn_network_apply_stdp(snn_network_t* network) {
    if (!network) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "snn_network_apply_stdp: null network pointer");
        return SNN_ERROR_NULL_POINTER;
    }
    if (!network->neural_net) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_STATE, "snn_network_apply_stdp: null neural network");
        return SNN_ERROR_INVALID_STATE;
    }

    /* Delegate to neural_network_apply_stdp which already exists */
    uint64_t timestamp = network->sim ? network->sim->current_time_us : 0;
    uint32_t n_neurons = neural_network_get_num_neurons(network->neural_net);
    int total_modified = 0;

    for (uint32_t i = 0; i < n_neurons; i++) {
        int modified = neural_network_apply_stdp(network->neural_net, i, timestamp);
        if (modified > 0) {
            total_modified += modified;
        }
    }

    return total_modified;
}

int snn_network_apply_rstdp(snn_network_t* network, float reward) {
    if (!network) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "snn_network_apply_rstdp: null network pointer");
        return SNN_ERROR_NULL_POINTER;
    }
    if (!network->train_ctx) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_STATE, "snn_network_apply_rstdp: null training context");
        return SNN_ERROR_INVALID_STATE;
    }

    network->train_ctx->reward = reward;

    /* Apply reward-modulated learning using existing infrastructure */
    uint64_t timestamp = network->sim ? network->sim->current_time_us : 0;
    float learning_rate = network->config.learning_rate;

    uint32_t modified = neural_network_apply_reward_learning(
        network->neural_net, reward, learning_rate, timestamp);

    return (int)modified;
}

int snn_network_compute_gradients(snn_network_t* network,
                                  const float* target,
                                  uint32_t n_targets) {
    if (!network) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "snn_network_compute_gradients: null network pointer");
        return SNN_ERROR_NULL_POINTER;
    }
    if (!target) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "snn_network_compute_gradients: null target pointer");
        return SNN_ERROR_NULL_POINTER;
    }
    if (!network->train_ctx) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_STATE, "snn_network_compute_gradients: null training context");
        return SNN_ERROR_INVALID_STATE;
    }

    /* Surrogate gradient computation - placeholder for full BPTT */
    /* A complete implementation would unroll the simulation and compute
       gradients using the chain rule with surrogate gradient functions */

    /* For now, compute simple error signal */
    if (!network->output_pop || n_targets != network->output_pop->n_neurons) {
        return SNN_ERROR_INVALID_DIMENSION;
    }

    float total_error = 0.0f;
    for (uint32_t i = 0; i < n_targets; i++) {
        float rate = compute_firing_rate(
            &network->output_pop->spike_trains[i],
            network->sim->current_time_us,
            network->config.decoder.time_window);

        float error = target[i] - rate;
        total_error += error * error;
    }

    network->train_ctx->current_loss = total_error / (float)n_targets;

    /* Exponential moving average of loss */
    float alpha = 0.1f;
    network->train_ctx->smoothed_loss =
        alpha * network->train_ctx->current_loss +
        (1.0f - alpha) * network->train_ctx->smoothed_loss;

    return SNN_SUCCESS;
}

int snn_network_apply_gradients(snn_network_t* network, float learning_rate) {
    if (!network) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "snn_network_apply_gradients: null network pointer");
        return SNN_ERROR_NULL_POINTER;
    }

    float lr = (learning_rate > 0.0f) ? learning_rate : network->config.learning_rate;

    /* Temporarily override config LR so STDP uses the caller's rate */
    float saved_lr = network->config.learning_rate;
    network->config.learning_rate = lr;
    int rc = snn_network_apply_stdp(network);
    network->config.learning_rate = saved_lr;
    return rc;
}

float snn_network_train_step(snn_network_t* network,
                             const float* inputs,
                             uint32_t n_inputs,
                             const float* targets,
                             uint32_t n_targets,
                             float duration_ms) {
    if (!network) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "snn_network_train_step: null network pointer");
        return -1.0f;
    }
    if (!inputs) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "snn_network_train_step: null inputs pointer");
        return -1.0f;
    }
    if (!targets) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "snn_network_train_step: null targets pointer");
        return -1.0f;
    }

    /* Enable training mode */
    snn_network_set_training(network, true);

    /* Forward pass — dynamic allocation instead of fixed 4096 stack buffer */
    float* outputs = (float*)nimcp_calloc(n_targets, sizeof(float));
    if (!outputs) return -1.0f;

    int result = snn_network_forward(network, inputs, n_inputs,
                                     outputs, n_targets, duration_ms);
    if (result < 0) { nimcp_free(outputs); return -1.0f; }

    /* Compute gradients */
    result = snn_network_compute_gradients(network, targets, n_targets);
    if (result != SNN_SUCCESS) { nimcp_free(outputs); return -1.0f; }

    /* Apply gradients (STDP for now) */
    snn_network_apply_gradients(network, 0.0f);

    nimcp_free(outputs);
    return network->train_ctx->current_loss;
}

//=============================================================================
// Population Management
//=============================================================================

int snn_network_add_population(snn_network_t* network,
                               uint32_t n_neurons,
                               neuron_type_t neuron_type,
                               const char* name) {
    if (!network) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "snn_network_add_population: null network pointer");
        return SNN_ERROR_NULL_POINTER;
    }
    if (n_neurons == 0) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_SIZE, "snn_network_add_population: zero neurons");
        return SNN_ERROR_INVALID_DIMENSION;
    }
    /* Bug fix: guard against the actual allocation size, not just
     * SNN_MAX_POPULATIONS. Legacy code tripped over this when
     * populations[] was sized by config->n_populations. */
    uint32_t capacity = network->populations_capacity;
    if (capacity == 0) capacity = SNN_MAX_POPULATIONS; /* legacy safety */
    if (network->n_populations >= capacity) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_SIZE,
            "snn_network_add_population: max populations exceeded");
        return SNN_ERROR_OUT_OF_MEMORY;
    }

    uint32_t pop_id = network->n_populations;

    snn_population_t* pop = snn_population_create_internal(
        pop_id, n_neurons, neuron_type, name, 0);
    if (!pop) return SNN_ERROR_OUT_OF_MEMORY;

    /* Add neurons to underlying neural network.
     * Bug fix: neural_network_add_neuron returns UINT32_MAX on capacity
     * exhaustion. The old code stored that sentinel into
     * pop->neuron_ids[i] without checking, and later
     * neural_net->neurons[pop->neuron_ids[i]] segfaulted. Now we detect
     * the sentinel, destroy the partially-built population, and return
     * SNN_ERROR_OUT_OF_MEMORY. Any neurons already added to the
     * underlying neural_network_t are leak-tolerated: no
     * neural_network_remove_neuron exists, and those slots will be
     * freed on neural_network_destroy with the rest of the network. */
    for (uint32_t i = 0; i < n_neurons; i++) {
        uint32_t neuron_id = neural_network_add_neuron(
            network->neural_net, ACTIVATION_SIGMOID);
        if (neuron_id == UINT32_MAX) {
            NIMCP_LOGGING_ERROR(
                "snn_network_add_population: neural_network_add_neuron "
                "exhausted at slot %u/%u for pop '%s'; %u neurons already "
                "added to neural_network are leak-tolerated "
                "(no remove API)",
                i, n_neurons, name ? name : "unnamed", i);
            snn_population_destroy_internal(pop);
            return SNN_ERROR_OUT_OF_MEMORY;
        }
        pop->neuron_ids[i] = neuron_id;
    }

    network->populations[pop_id] = pop;
    network->n_populations++;

    NIMCP_LOGGING_DEBUG("snn_network_add_population: added '%s' with %u neurons",
                       name ? name : "unnamed", n_neurons);

    return (int)pop_id;
}

int snn_network_add_population_lightweight(snn_network_t* network,
                                           uint32_t n_neurons,
                                           neuron_type_t neuron_type,
                                           const char* name) {
    if (!network) return -1;
    if (n_neurons == 0) return -1;
    /* Bug fix: guard against the actual allocation size, not just
     * SNN_MAX_POPULATIONS. See snn_network_add_population for details. */
    uint32_t capacity = network->populations_capacity;
    if (capacity == 0) capacity = SNN_MAX_POPULATIONS; /* legacy safety */
    if (network->n_populations >= capacity) return -1;

    uint32_t pop_id = network->n_populations;
    snn_population_t* pop = snn_population_create_lightweight(
        pop_id, n_neurons, neuron_type, name);
    if (!pop) return -1;

    network->populations[pop_id] = pop;
    network->n_populations++;

    NIMCP_LOGGING_DEBUG("snn_network_add_population_lightweight: added '%s' "
                        "with %u neurons (CSR mode)", name ? name : "unnamed", n_neurons);

    return (int)pop_id;
}

int snn_network_connect_populations(snn_network_t* network,
                                    uint32_t src_pop,
                                    uint32_t dst_pop,
                                    snn_topology_t topology,
                                    float connectivity,
                                    synapse_type_t synapse_type,
                                    float weight_mean,
                                    float weight_std) {
    if (!network) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "snn_network_connect_populations: null network pointer");
        return SNN_ERROR_NULL_POINTER;
    }
    if (src_pop >= network->n_populations || dst_pop >= network->n_populations) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM,
            "snn_network_connect_populations: invalid population index");
        return SNN_ERROR_INVALID_POPULATION;
    }

    snn_population_t* src = network->populations[src_pop];
    snn_population_t* dst = network->populations[dst_pop];
    if (!src || !dst) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
            "snn_network_connect_populations: null population pointer");
        return SNN_ERROR_INVALID_POPULATION;
    }

    int n_connections = 0;

    /* Helper: generate Gaussian random weight */
    #define RANDOM_WEIGHT(mean, std) ({ \
        float _w = (mean); \
        if ((std) > 0.0f) { \
            float _u1 = (float)nimcp_tl_rand() / (float)RAND_MAX; \
            float _u2 = (float)nimcp_tl_rand() / (float)RAND_MAX; \
            if (_u1 < 1e-7f) _u1 = 1e-7f; \
            _w += sqrtf(-2.0f * logf(_u1)) * cosf(NIMCP_TWO_PI_F * _u2) * (std); \
        } _w; })

    /* Lightweight CSR path: store synapses in dst population's CSR storage
     * instead of the neural_network_t sparse synapse system. */
    if (dst->lightweight && dst->incoming_csr) {
        bool use_ds = (topology == SNN_TOPO_RANDOM && connectivity < 0.1f &&
                       (uint64_t)src->n_neurons * dst->n_neurons > 1000000);
        if (use_ds) {
            uint32_t k = (uint32_t)(dst->n_neurons * connectivity + 0.5f);
            if (k == 0) k = 1;
            if (k > dst->n_neurons) k = dst->n_neurons;
            for (uint32_t i = 0; i < src->n_neurons; i++) {
                for (uint32_t c = 0; c < k; c++) {
                    uint32_t j = (uint32_t)(nimcp_tl_rand() % dst->n_neurons);
                    float w = RANDOM_WEIGHT(weight_mean, weight_std);
                    if (snn_csr_add_entry(dst->incoming_csr, j,
                                          src_pop, i, w) == 0)
                        n_connections++;
                }
            }
        } else {
            for (uint32_t i = 0; i < src->n_neurons; i++) {
                for (uint32_t j = 0; j < dst->n_neurons; j++) {
                    if (topology == SNN_TOPO_RANDOM) {
                        float r = (float)nimcp_tl_rand() / (float)RAND_MAX;
                        if (r > connectivity) continue;
                    }
                    float w = RANDOM_WEIGHT(weight_mean, weight_std);
                    if (snn_csr_add_entry(dst->incoming_csr, j,
                                          src_pop, i, w) == 0)
                        n_connections++;
                }
            }
        }
        goto done;
    }

    /* Legacy neuron_t path (for non-lightweight populations) */
    {
    bool use_direct_sampling = (topology == SNN_TOPO_RANDOM &&
                                 connectivity < 0.1f &&
                                 (uint64_t)src->n_neurons * dst->n_neurons > 1000000);

    if (use_direct_sampling) {
        uint32_t k = (uint32_t)(dst->n_neurons * connectivity + 0.5f);
        if (k == 0) k = 1;
        if (k > dst->n_neurons) k = dst->n_neurons;

        for (uint32_t i = 0; i < src->n_neurons; i++) {
            for (uint32_t c = 0; c < k; c++) {
                uint32_t j = (uint32_t)(nimcp_tl_rand() % dst->n_neurons);
                float weight = RANDOM_WEIGHT(weight_mean, weight_std);

                bool success = neural_network_add_connection_typed(
                    network->neural_net,
                    src->neuron_ids[i],
                    dst->neuron_ids[j],
                    weight,
                    synapse_type);
                if (success) n_connections++;
            }
        }
    } else {
        for (uint32_t i = 0; i < src->n_neurons; i++) {
            for (uint32_t j = 0; j < dst->n_neurons; j++) {
                if (topology == SNN_TOPO_RANDOM) {
                    float r = (float)nimcp_tl_rand() / (float)RAND_MAX;
                    if (r > connectivity) continue;
                }
                float weight = RANDOM_WEIGHT(weight_mean, weight_std);

                bool success = neural_network_add_connection_typed(
                    network->neural_net,
                    src->neuron_ids[i],
                    dst->neuron_ids[j],
                    weight,
                    synapse_type);
                if (success) n_connections++;
            }
        }
    }
    } /* end legacy block */

done:
    #undef RANDOM_WEIGHT

    NIMCP_LOGGING_DEBUG("snn_network_connect_populations: created %d connections",
                       n_connections);

    return n_connections;
}

int snn_network_finalize_connections(snn_network_t* network) {
    if (!network) return -1;
    int finalized = 0;
    for (uint32_t p = 0; p < network->n_populations; p++) {
        snn_population_t* pop = network->populations[p];
        if (pop && pop->lightweight && pop->incoming_csr && !pop->incoming_csr->finalized) {
            if (snn_csr_finalize(pop->incoming_csr) == 0) {
                finalized++;
            }
        }
    }
    NIMCP_LOGGING_INFO("Finalized CSR for %d lightweight populations", finalized);

    /* Re-create GPU LIF state with correct total neuron count.
     * The initial snn_network_create() only counted n_inputs+n_hidden+n_outputs,
     * but lightweight populations add millions more neurons that need GPU state. */
    if (finalized > 0) {
        size_t total_neurons = 0;
        for (uint32_t p = 0; p < network->n_populations; p++) {
            if (network->populations[p])
                total_neurons += network->populations[p]->n_neurons;
        }

        /* Destroy old undersized GPU state */
        if (network->gpu_lif_state) {
            nimcp_lif_state_destroy((nimcp_lif_state_t*)network->gpu_lif_state);
            network->gpu_lif_state = NULL;
        }

        /* Create new state for full neuron count */
        if (network->gpu_ctx) {
            nimcp_lif_params_t lif_params = {
                .tau_mem   = network->config.tau_mem,
                .tau_syn   = network->config.tau_syn > 0.0f ? network->config.tau_syn : 5.0f,
                .v_thresh  = network->config.v_thresh,
                .v_reset   = network->config.v_reset,
                .v_rest    = network->config.v_rest,
                .dt        = network->config.dt,
                .hard_reset = true
            };
            nimcp_lif_state_t* lif_state = nimcp_lif_state_create(
                (nimcp_gpu_context_t*)network->gpu_ctx, total_neurons, &lif_params);
            if (lif_state) {
                network->gpu_lif_state = lif_state;
                NIMCP_LOGGING_INFO("GPU LIF state re-created for %zu total neurons "
                                   "(including lightweight)", total_neurons);
            } else {
                NIMCP_LOGGING_WARN("GPU LIF state creation failed for %zu neurons — "
                                   "falling back to CPU", total_neurons);
            }
        }

        /* Build population offset table and prepare GPU-friendly CSR arrays.
         * Flat index = pop_offsets[src_pop] + src_neuron */
        uint32_t* pop_offsets = nimcp_calloc(network->n_populations + 1, sizeof(uint32_t));
        if (pop_offsets) {
            for (uint32_t p = 0; p < network->n_populations; p++) {
                pop_offsets[p + 1] = pop_offsets[p] +
                    (network->populations[p] ? network->populations[p]->n_neurons : 0);
            }
            int gpu_prepared = 0;
            for (uint32_t p = 0; p < network->n_populations; p++) {
                snn_population_t* pop = network->populations[p];
                if (pop && pop->lightweight && pop->incoming_csr) {
                    if (snn_csr_prepare_gpu(pop->incoming_csr, pop_offsets,
                                            network->n_populations) == 0) {
                        gpu_prepared++;
                    }
                }
            }
            NIMCP_LOGGING_INFO("GPU CSR prepared for %d populations "
                               "(%zu total neurons, flat index range 0-%u)",
                               gpu_prepared, total_neurons,
                               pop_offsets[network->n_populations]);
            nimcp_free(pop_offsets);
        }
    }

    return finalized;
}

snn_population_t* snn_network_get_population(snn_network_t* network,
                                             uint32_t pop_id) {
    if (!network) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "snn_network_get_population: null network pointer");
        return NULL;
    }
    if (pop_id >= network->n_populations) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM,
            "snn_network_get_population: invalid population index");
        return NULL;
    }
    return network->populations[pop_id];
}

//=============================================================================
// Statistics and Monitoring
//=============================================================================

int snn_network_get_stats(snn_network_t* network, snn_stats_t* stats) {
    if (!network) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "snn_network_get_stats: null network pointer");
        return SNN_ERROR_NULL_POINTER;
    }
    if (!stats) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "snn_network_get_stats: null stats pointer");
        return SNN_ERROR_NULL_POINTER;
    }

    memcpy(stats, &network->stats, sizeof(snn_stats_t));

    /* Update computed stats */
    stats->health = snn_network_check_health(network);

    /* Compute mean firing rate across all populations */
    float total_rate = 0.0f;
    uint32_t total_neurons = 0;
    for (uint32_t p = 0; p < network->n_populations; p++) {
        snn_population_t* pop = network->populations[p];
        if (!pop) continue;

        total_rate += pop->mean_rate * pop->n_neurons;
        total_neurons += pop->n_neurons;
    }
    stats->mean_firing_rate = (total_neurons > 0) ? total_rate / total_neurons : 0.0f;

    return SNN_SUCCESS;
}

snn_state_health_t snn_network_check_health(snn_network_t* network) {
    if (!network) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "snn_network_check_health: null network pointer");
        return SNN_STATE_SILENT;
    }

    /* Check for silence (no spikes in recent window) */
    if (network->stats.total_spikes == 0 && network->sim->step_count > 1000) {
        return SNN_STATE_SILENT;
    }

    /* Check for explosion (too many spikes) */
    float max_rate = 1000.0f;  /* Hz */
    for (uint32_t p = 0; p < network->n_populations; p++) {
        snn_population_t* pop = network->populations[p];
        if (!pop) continue;

        if (pop->mean_rate > max_rate) {
            return SNN_STATE_EXPLOSION;
        }
    }

    return SNN_STATE_HEALTHY;
}

float snn_network_get_firing_rate(snn_network_t* network,
                                  uint32_t pop_id,
                                  uint32_t neuron_idx,
                                  float window_ms) {
    if (!network) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "snn_network_get_firing_rate: null network pointer");
        return 0.0f;
    }
    if (pop_id >= network->n_populations) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM,
            "snn_network_get_firing_rate: invalid population index");
        return 0.0f;
    }

    snn_population_t* pop = network->populations[pop_id];
    if (!pop || neuron_idx >= pop->n_neurons) return 0.0f;

    return compute_firing_rate(
        &pop->spike_trains[neuron_idx],
        network->sim->current_time_us,
        window_ms);
}

float snn_network_get_population_rate(snn_network_t* network,
                                      uint32_t pop_id,
                                      float window_ms) {
    if (!network) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "snn_network_get_population_rate: null network pointer");
        return 0.0f;
    }
    if (pop_id >= network->n_populations) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM,
            "snn_network_get_population_rate: invalid population index");
        return 0.0f;
    }

    snn_population_t* pop = network->populations[pop_id];
    if (!pop) return 0.0f;

    float total_rate = 0.0f;
    for (uint32_t n = 0; n < pop->n_neurons; n++) {
        total_rate += compute_firing_rate(
            &pop->spike_trains[n],
            network->sim->current_time_us,
            window_ms);
    }

    return total_rate / (float)pop->n_neurons;
}

uint32_t snn_network_get_population_history(snn_network_t* network,
                                            uint32_t pop_id,
                                            uint32_t* out_counts,
                                            uint64_t* out_total_steps) {
    if (!network || !out_counts) return 0;
    if (pop_id >= network->n_populations) return 0;
    snn_population_t* pop = network->populations[pop_id];
    if (!pop) return 0;

    /* Copy in time order: start at write_idx (oldest), wrap around.
     * If history_total_steps < SNN_POP_HISTORY_LEN, only the first
     * history_total_steps entries are valid — caller gets that count. */
    uint64_t total = pop->history_total_steps;
    uint32_t valid = (total < SNN_POP_HISTORY_LEN)
                   ? (uint32_t)total : SNN_POP_HISTORY_LEN;
    uint32_t start = (valid == SNN_POP_HISTORY_LEN)
                   ? pop->history_write_idx  /* ring wrapped, oldest is write head */
                   : 0;                      /* not wrapped yet, oldest is at 0 */
    for (uint32_t i = 0; i < valid; i++) {
        uint32_t idx = (start + i) % SNN_POP_HISTORY_LEN;
        out_counts[i] = pop->spike_count_history[idx];
    }
    if (out_total_steps) *out_total_steps = total;
    return valid;
}

/* Emergency saturation rescue: run snn_homeostatic_apply in a tight loop
 * without waiting for R-STDP cadence. Used when the network has drifted
 * far from biological range and we don't want to wait for the normal
 * every-10-steps homeostatic cadence (which can take hours at 1 learn_vector
 * per minute). Each iteration scales all lightweight population weights by
 * target_rate/current_rate (clamped to [0.90, 1.10] in emergency range),
 * so log(0.9) ≈ -0.105 per iteration — 20 iterations drops firing 8×.
 *
 * @param network   SNN network (must have snn_training_ctx attached)
 * @param ctx       Training context (from dispatch or test harness)
 * @param n_iter    Number of homeostatic applies to run back-to-back
 * @return Total number of populations scaled across all iterations
 */
uint32_t snn_network_force_homeostasis(snn_network_t* network,
                                       void* ctx,
                                       uint32_t n_iter) {
    if (!network || !ctx || n_iter == 0) return 0;
    /* snn_homeostatic_apply is declared in nimcp_snn_training.h (included
     * at the top of this file). ctx is a snn_training_ctx_t*; we take
     * void* through the force-homeostasis signature to keep the public
     * force-homeostasis API loose, but cast on the call site. */
    uint32_t total_scaled = 0;
    for (uint32_t i = 0; i < n_iter; i++) {
        total_scaled += snn_homeostatic_apply((snn_training_ctx_t*)ctx, network);
    }
    return total_scaled;
}

float snn_population_get_firing_rate(const snn_population_t* population) {
    if (!population) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "snn_population_get_firing_rate: null population pointer");
        return 0.0f;
    }

    /* Return the cached mean_rate maintained by the population */
    return population->mean_rate;
}

//=============================================================================
// Integration
//=============================================================================

int snn_network_connect_bio_async(snn_network_t* network) {
    if (!network) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "snn_network_connect_bio_async: null network pointer");
        return SNN_ERROR_NULL_POINTER;
    }

    /* Bio-async registration - placeholder */
    /* Full implementation would register with bio_router */
    NIMCP_LOGGING_INFO("snn_network_connect_bio_async: bio-async connection (stub)");

    return SNN_SUCCESS;
}

int snn_network_disconnect_bio_async(snn_network_t* network) {
    if (!network) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "snn_network_disconnect_bio_async: null network pointer");
        return SNN_ERROR_NULL_POINTER;
    }

    network->bio_ctx = NULL;
    return SNN_SUCCESS;
}

int snn_network_connect_immune(snn_network_t* network, void* immune) {
    if (!network) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "snn_network_connect_immune: null network pointer");
        return SNN_ERROR_NULL_POINTER;
    }

    network->immune_bridge = immune;
    NIMCP_LOGGING_INFO("snn_network_connect_immune: immune system connected");

    return SNN_SUCCESS;
}

int snn_network_apply_immune_modulation(snn_network_t* network) {
    if (!network) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "snn_network_apply_immune_modulation: null network pointer");
        return SNN_ERROR_NULL_POINTER;
    }
    if (!network->immune_bridge) return SNN_SUCCESS;  /* No-op if not connected */

    /* Immune modulation - placeholder */
    /* Full implementation would query cytokine levels and modulate
       firing threshold, time constants, and learning rates */

    return SNN_SUCCESS;
}

//=============================================================================
// Utility Functions
//=============================================================================

neural_network_t snn_network_get_neural_net(snn_network_t* network) {
    if (!network) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "snn_network_get_neural_net: null network pointer");
        return NULL;
    }
    return network->neural_net;
}

int snn_network_validate(const snn_network_t* network) {
    if (!network) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "snn_network_validate: null network pointer");
        return SNN_ERROR_NULL_POINTER;
    }

    /* Check magic number */
    if (network->magic != SNN_MAGIC) {
        NIMCP_LOGGING_ERROR("snn_network_validate: invalid magic number");
        return SNN_ERROR_INVALID_STATE;
    }

    /* Check neural network */
    if (!network->neural_net) {
        NIMCP_LOGGING_ERROR("snn_network_validate: no underlying neural network");
        return SNN_ERROR_INVALID_STATE;
    }

    /* Check populations */
    if (!network->populations) {
        NIMCP_LOGGING_ERROR("snn_network_validate: no populations array");
        return SNN_ERROR_INVALID_STATE;
    }

    return SNN_SUCCESS;
}

//=============================================================================
// Persistence (save/load)
//=============================================================================

#include <stdio.h>
#include <errno.h>
#include "core/neuralnet/nimcp_sparse_synapse.h"

/* SNN checkpoint format constants */
#define SNN_CHECKPOINT_MAGIC    0x534E4E53  /**< "SNNS" */
#define SNN_CHECKPOINT_VERSION  4           /**< v4: adds wiring_schema_version field */

/* Wiring schema version: bump whenever the hierarchical wiring topology or
 * weight-initialization formula changes. A mismatch on load causes the cached
 * .snn file to be rejected so the network is rebuilt with the new scheme.
 *
 *   1 = original hardcoded weights (in=0.3, FF=1.0, inhib=-0.6, etc.)
 *   2 = fluctuation-driven weights per arxiv 2206.10226 (2026-04-15)
 *   3 = v2 + stronger recurrent quench (60E/40I at 4× I, 2026-04-15)
 *   4 = v3 reverted to 80E/20I + 4× I, budget reduced 0.8→0.3 × gap
 *       (rate recalibration; 2026-04-15)
 *   5 = v4 + Turrigiano-style homeostatic synaptic scaling to prevent
 *       R-STDP-induced drift into silent/saturated regimes (2026-04-16)
 *   6 = v5 + intrinsic plasticity (per-neuron threshold adaptation),
 *       short-term synaptic depression, separate E/I plasticity rules,
 *       per-neuron metabolic weight budget, aggressive emergency
 *       homeostatic range (±10% when rate > 3× or < 1/3 target),
 *       R-STDP LR halved 0.001→0.0005, and force_quench RPC.
 *       Full biological stability package. (2026-04-19) */
/* SCHEMA VERSION HISTORY
 *   v5: fluctuation-driven weight init (2026-04-18)
 *   v6: full biological stability package — IP + depression + inhibitory
 *       plasticity + metabolic budget + emergency homeostatic range (2026-04-19)
 *   v7: persist per-neuron homeostatic state across restarts —
 *       threshold_offset, neuron_rate_ema, depression (2026-04-19)
 *       Without this, every restart wiped hours of homeostatic settling.
 */
#define SNN_WIRING_SCHEMA_VERSION  7

/* Checked fwrite helper — returns -1 on failure.
 * Saves raw (uncompressed) — compression happens at rsync transit time
 * via `rsync -z`. Raw write on NVMe is ~1.6 GB/s vs single-threaded gzip-3
 * at ~70 MB/s, so save wall time drops from ~16 min to ~15 s. The on-disk
 * file grows from 10.5 GB to ~17 GB. */
#define SNN_FWRITE(ptr, size, count, stream) \
    do { size_t _items = (count); \
         size_t _w = fwrite((ptr), (size), _items, (stream)); \
         if (_w != _items) { \
            NIMCP_LOGGING_ERROR("snn_network_save: fwrite failed (wrote %zu of %zu items)", \
                                _w, _items); \
            fclose(stream); if (tmp_path[0]) remove(tmp_path); return -1; \
         } } while (0)

int snn_network_save(snn_network_t* network, const char* path) {
    if (!network || !path) return -1;

    /* Atomic write: write to temp file, then rename */
    char tmp_path[1024];
    snprintf(tmp_path, sizeof(tmp_path), "%s.tmp", path);

    FILE* f = fopen(tmp_path, "wb");
    if (!f) {
        NIMCP_LOGGING_ERROR("snn_network_save: failed to open %s", tmp_path);
        return -1;
    }

    uint32_t magic = SNN_CHECKPOINT_MAGIC;
    uint32_t version = SNN_CHECKPOINT_VERSION;
    uint32_t wiring_schema = SNN_WIRING_SCHEMA_VERSION;
    SNN_FWRITE(&magic, sizeof(uint32_t), 1, f);
    SNN_FWRITE(&version, sizeof(uint32_t), 1, f);
    SNN_FWRITE(&wiring_schema, sizeof(uint32_t), 1, f);

    /* Save config (contains all architecture parameters) */
    SNN_FWRITE(&network->config, sizeof(snn_config_t), 1, f);
    SNN_FWRITE(&network->is_training, sizeof(bool), 1, f);

    /* Save neuron state and outgoing synapse weights via sparse synapse API */
    uint32_t total_neurons = network->neural_net
        ? neural_network_get_num_neurons(network->neural_net) : 0;
    SNN_FWRITE(&total_neurons, sizeof(uint32_t), 1, f);

    for (uint32_t i = 0; i < total_neurons && network->neural_net; i++) {
        neuron_t* n = neural_network_get_neuron(network->neural_net, i);
        if (!n) {
            float zero = 0.0f;
            uint32_t zero_u = 0;
            SNN_FWRITE(&zero, sizeof(float), 1, f);
            SNN_FWRITE(&zero, sizeof(float), 1, f);
            SNN_FWRITE(&zero, sizeof(float), 1, f);
            SNN_FWRITE(&zero_u, sizeof(uint32_t), 1, f);
            continue;
        }

        SNN_FWRITE(&n->state, sizeof(float), 1, f);
        SNN_FWRITE(&n->threshold, sizeof(float), 1, f);
        SNN_FWRITE(&n->bias, sizeof(float), 1, f);

        uint32_t n_synapses = sparse_synapse_count(&n->outgoing);
        SNN_FWRITE(&n_synapses, sizeof(uint32_t), 1, f);
        for (uint32_t s = 0; s < n_synapses; s++) {
            synapse_handle_t* h = sparse_synapse_get(&n->outgoing, s);
            float w = h ? h->weight : 0.0f;
            uint32_t tid = h ? h->target_neuron_id : 0;
            SNN_FWRITE(&w, sizeof(float), 1, f);
            SNN_FWRITE(&tid, sizeof(uint32_t), 1, f);
        }
    }

    /* === V3: Save lightweight CSR population data === */
    uint32_t n_lightweight = 0;
    for (uint32_t p = 0; p < network->n_populations; p++) {
        snn_population_t* pop = network->populations[p];
        if (pop && pop->lightweight && pop->incoming_csr && pop->incoming_csr->finalized)
            n_lightweight++;
    }
    SNN_FWRITE(&n_lightweight, sizeof(uint32_t), 1, f);
    SNN_FWRITE(&network->n_populations, sizeof(uint32_t), 1, f);

    if (n_lightweight > 0) {
        /* Save population metadata */
        for (uint32_t p = 0; p < network->n_populations; p++) {
            snn_population_t* pop = network->populations[p];
            if (!pop || !pop->lightweight || !pop->incoming_csr) continue;

            SNN_FWRITE(&pop->id, sizeof(uint32_t), 1, f);
            SNN_FWRITE(&pop->n_neurons, sizeof(uint32_t), 1, f);
            SNN_FWRITE(&pop->neuron_type, sizeof(neuron_type_t), 1, f);
            SNN_FWRITE(pop->name, sizeof(pop->name), 1, f);

            /* CSR data */
            snn_csr_storage_t* csr = pop->incoming_csr;
            SNN_FWRITE(&csr->n_neurons, sizeof(uint32_t), 1, f);
            SNN_FWRITE(&csr->n_synapses, sizeof(uint32_t), 1, f);
            SNN_FWRITE(csr->row_ptr, sizeof(uint32_t), csr->n_neurons + 1, f);
            SNN_FWRITE(csr->entries, sizeof(snn_csr_synapse_t), csr->n_synapses, f);

            /* GPU arrays (weights[], flat_col_idx[]) are derived from entries[]
             * — skip writing them to save ~8 bytes/synapse (~11.6 GB for 1.45B).
             * They will be regenerated on first GPU step via snn_csr_prepare_gpu().
             * Always write has_gpu=0 to signal "regenerate on demand". */
            uint8_t has_gpu = 0;
            SNN_FWRITE(&has_gpu, sizeof(uint8_t), 1, f);

            /* === V7: Persist per-neuron homeostatic state ===
             * These arrays capture hours of biological adaptation. Without
             * persisting them, every restart wiped threshold adaptation,
             * rate EMAs, and depression levels — forcing the network to
             * re-discover its operating point from scratch every time. */
            uint8_t has_homeostatic = (pop->threshold_offset &&
                                       pop->neuron_rate_ema &&
                                       pop->depression) ? 1 : 0;
            SNN_FWRITE(&has_homeostatic, sizeof(uint8_t), 1, f);
            if (has_homeostatic) {
                SNN_FWRITE(pop->threshold_offset, sizeof(float), pop->n_neurons, f);
                SNN_FWRITE(pop->neuron_rate_ema, sizeof(float), pop->n_neurons, f);
                SNN_FWRITE(pop->depression, sizeof(float), pop->n_neurons, f);
            }
        }
        NIMCP_LOGGING_INFO("SNN save: %u lightweight CSR populations saved "
                           "(with per-neuron homeostatic state)", n_lightweight);
    }

    fclose(f);

    /* Atomic rename: temp → final path.
     * If the destination already exists as a directory (misconfig/stale symlink),
     * rename(2) fails with EISDIR/ENOTDIR. Surface the actual errno so ops can
     * fix the underlying condition instead of guessing. */
    if (rename(tmp_path, path) != 0) {
        int saved_errno = errno;
        NIMCP_LOGGING_ERROR("snn_network_save: rename %s → %s failed: %s (errno=%d)",
                            tmp_path, path, strerror(saved_errno), saved_errno);
        remove(tmp_path);
        return -1;
    }

    NIMCP_LOGGING_INFO("SNN network saved to %s (raw, %u neurons, %u CSR pops)",
                       path, total_neurons, n_lightweight);
    return 0;
}

#undef SNN_FWRITE

/* fread-like wrapper for gzFile. Returns count on success, less on EOF/error.
 * gzopen for read auto-detects gzip header and falls back to plain reads, so
 * this works for both compressed AND existing uncompressed .snn files. */
static inline size_t gz_fread(void* ptr, size_t sz, size_t count, gzFile f) {
    if (sz == 0 || count == 0) return 0;
    int got = gzread(f, ptr, (unsigned)(sz * count));
    if (got <= 0) return 0;
    return (size_t)got / sz;
}

snn_network_t* snn_network_load(const char* path) {
    if (!path) return NULL;

    /* gzopen handles both gzip-compressed and plain files transparently —
     * existing uncompressed .snn files load fine without conversion. */
    gzFile f = gzopen(path, "rb");
    if (!f) {
        NIMCP_LOGGING_WARN("snn_network_load: file not found %s", path);
        return NULL;
    }
    #define fread(p, s, c, fp) gz_fread((p), (s), (c), (fp))
    #define fclose(fp) gzclose(fp)
    #define fseek(fp, off, whence) gzseek((fp), (off), (whence))

    uint32_t magic = 0, version = 0;
    if (fread(&magic, sizeof(uint32_t), 1, f) != 1 || magic != SNN_CHECKPOINT_MAGIC) {
        NIMCP_LOGGING_ERROR("snn_network_load: invalid magic");
        fclose(f);
        return NULL;
    }
    if (fread(&version, sizeof(uint32_t), 1, f) != 1 || version > SNN_CHECKPOINT_VERSION) {
        NIMCP_LOGGING_ERROR("snn_network_load: unsupported version %u", version);
        fclose(f); return NULL;
    }

    /* Wiring-schema version gating. v4+ files embed the schema tag; older files
     * predate the fluctuation-driven init and must be rejected so the caller
     * rebuilds from scratch with the current formula. */
    if (version < 4) {
        NIMCP_LOGGING_WARN("snn_network_load: checkpoint v%u predates wiring schema "
                           "tracking — rejecting so network is rebuilt with current "
                           "wiring schema v%u", version, SNN_WIRING_SCHEMA_VERSION);
        fclose(f); return NULL;
    }
    uint32_t wiring_schema = 0;
    if (fread(&wiring_schema, sizeof(uint32_t), 1, f) != 1) {
        NIMCP_LOGGING_ERROR("snn_network_load: truncated wiring_schema field");
        fclose(f); return NULL;
    }
    /* Schema compatibility policy:
     *   - Forward-compatible: newer file schema than binary → reject (we
     *     don't know what fields were added, can't safely load)
     *   - Backward-compatible: older file schema than binary → ACCEPT for
     *     schemas ≥ SNN_MIN_COMPATIBLE_SCHEMA.
     *     * v5 → v6: added per-neuron in-memory fields (threshold_offset,
     *       neuron_rate_ema, depression) but didn't change on-disk format.
     *     * v6 → v7: on-disk format extended with optional homeostatic-state
     *       block per population. Reads guarded by `if (wiring_schema >= 7)`
     *       so v5/v6 files skip it and use seed defaults.
     *     Older saves still load cleanly; missing fields default-init. This
     *     preserves learned synaptic patterns across schema bumps.
     *   - For format-incompatible bumps in future, bump SNN_MIN_COMPATIBLE
     *     and force a rebuild. */
    #define SNN_MIN_COMPATIBLE_SCHEMA 5
    if (wiring_schema > SNN_WIRING_SCHEMA_VERSION) {
        NIMCP_LOGGING_WARN("snn_network_load: file schema v%u is NEWER than "
                           "binary v%u — rejecting (can't load future format)",
                           wiring_schema, SNN_WIRING_SCHEMA_VERSION);
        fclose(f); return NULL;
    }
    if (wiring_schema < SNN_MIN_COMPATIBLE_SCHEMA) {
        NIMCP_LOGGING_WARN("snn_network_load: file schema v%u is below "
                           "minimum compatible v%u — rejecting, will rebuild",
                           wiring_schema, SNN_MIN_COMPATIBLE_SCHEMA);
        fclose(f); return NULL;
    }
    if (wiring_schema < SNN_WIRING_SCHEMA_VERSION) {
        NIMCP_LOGGING_INFO("snn_network_load: migrating v%u → v%u "
                           "(format-compatible, new in-memory fields default-init). "
                           "If loaded weights were saturated, call "
                           "brain.snn_force_quench(25) after load to rescue.",
                           wiring_schema, SNN_WIRING_SCHEMA_VERSION);
    }

    snn_config_t config;
    bool is_training = false;
    if (fread(&config, sizeof(snn_config_t), 1, f) != 1 ||
        fread(&is_training, sizeof(bool), 1, f) != 1) {
        NIMCP_LOGGING_ERROR("snn_network_load: truncated config");
        fclose(f); return NULL;
    }

    /* Recreate network from config */
    snn_network_t* net = snn_network_create(&config);
    if (!net) {
        NIMCP_LOGGING_ERROR("snn_network_load: failed to recreate network");
        fclose(f);
        return NULL;
    }
    net->is_training = is_training;

    /* Restore neuron state and synapse weights */
    uint32_t total_neurons = 0;
    if (fread(&total_neurons, sizeof(uint32_t), 1, f) != 1) {
        NIMCP_LOGGING_ERROR("snn_network_load: truncated neuron count");
        snn_network_destroy(net); fclose(f); return NULL;
    }

    uint32_t actual_neurons = net->neural_net
        ? neural_network_get_num_neurons(net->neural_net) : 0;
    uint32_t restore_count = (total_neurons < actual_neurons) ? total_neurons : actual_neurons;

    for (uint32_t i = 0; i < total_neurons; i++) {
        float state_val, threshold, bias;
        if (fread(&state_val, sizeof(float), 1, f) != 1 ||
            fread(&threshold, sizeof(float), 1, f) != 1 ||
            fread(&bias, sizeof(float), 1, f) != 1) {
            NIMCP_LOGGING_WARN("snn_network_load: truncated at neuron %u", i);
            break;
        }

        uint32_t n_synapses = 0;
        if (fread(&n_synapses, sizeof(uint32_t), 1, f) != 1) {
            NIMCP_LOGGING_WARN("snn_network_load: truncated synapse count at neuron %u", i);
            break;
        }

        /* Read per-synapse data: v1=weight only, v2=weight+target_id */
        float* weights = NULL;
        uint32_t* target_ids = NULL;
        if (n_synapses > 0) {
            weights = nimcp_malloc(n_synapses * sizeof(float));
            if (version >= 2) {
                target_ids = nimcp_malloc(n_synapses * sizeof(uint32_t));
            }
            bool synapse_read_truncated = false;
            for (uint32_t s = 0; s < n_synapses; s++) {
                float w = 0.0f;
                if (fread(&w, sizeof(float), 1, f) != 1) {
                    synapse_read_truncated = true;
                    break;
                }
                if (weights) weights[s] = w;
                if (version >= 2) {
                    uint32_t tid = 0;
                    if (fread(&tid, sizeof(uint32_t), 1, f) != 1) {
                        synapse_read_truncated = true;
                        break;
                    }
                    if (target_ids) target_ids[s] = tid;
                }
            }
            if (synapse_read_truncated) {
                NIMCP_LOGGING_WARN("snn_network_load: truncated synapse data at neuron %u", i);
                nimcp_free(weights);
                nimcp_free(target_ids);
                break;
            }
        }

        /* Apply to matching neuron in recreated network */
        if (i < restore_count && net->neural_net) {
            neuron_t* n = neural_network_get_neuron(net->neural_net, i);
            if (n) {
                n->state = state_val;
                n->threshold = threshold;
                n->bias = bias;
                if (weights && target_ids) {
                    /* v2: restore connections with weights and target IDs.
                     * If the network has no connections yet (fresh from snn_network_create),
                     * CREATE them from the checkpoint data. This preserves BPTT-trained
                     * weights across daemon restarts. */
                    uint32_t cur_syn = sparse_synapse_count(&n->outgoing);
                    if (cur_syn > 0) {
                        /* Network already has connections — match by target ID */
                        for (uint32_t s = 0; s < n_synapses; s++) {
                            for (uint32_t cs = 0; cs < cur_syn; cs++) {
                                synapse_handle_t* h = sparse_synapse_get(&n->outgoing, cs);
                                if (h && h->target_neuron_id == target_ids[s]) {
                                    h->weight = weights[s];
                                    break;
                                }
                            }
                        }
                    } else {
                        /* No connections — create from checkpoint data.
                         * Uses neural_network_add_connection to properly wire
                         * both outgoing and incoming synapse handles. */
                        for (uint32_t s = 0; s < n_synapses; s++) {
                            neural_network_add_connection(
                                net->neural_net, i, target_ids[s], weights[s]);
                        }
                    }
                } else if (weights) {
                    /* v1 fallback: positional restore */
                    uint32_t cur_syn = sparse_synapse_count(&n->outgoing);
                    uint32_t s_count = (n_synapses < cur_syn) ? n_synapses : cur_syn;
                    for (uint32_t s = 0; s < s_count; s++) {
                        synapse_handle_t* h = sparse_synapse_get(&n->outgoing, s);
                        if (h) h->weight = weights[s];
                    }
                }
            }
        }
        nimcp_free(weights);
        nimcp_free(target_ids);
    }

    /* === V3: Load lightweight CSR populations === */
    if (version >= 3) {
        uint32_t n_lightweight = 0, n_pops_saved = 0;
        if (fread(&n_lightweight, sizeof(uint32_t), 1, f) == 1 &&
            fread(&n_pops_saved, sizeof(uint32_t), 1, f) == 1 &&
            n_lightweight > 0) {

            NIMCP_LOGGING_INFO("Loading %u lightweight CSR populations...", n_lightweight);

            for (uint32_t lp = 0; lp < n_lightweight; lp++) {
                uint32_t pop_id, pop_n;
                neuron_type_t ntype;
                char pop_name[64];

                if (fread(&pop_id, sizeof(uint32_t), 1, f) != 1 ||
                    fread(&pop_n, sizeof(uint32_t), 1, f) != 1 ||
                    fread(&ntype, sizeof(neuron_type_t), 1, f) != 1 ||
                    fread(pop_name, sizeof(pop_name), 1, f) != 1) {
                    NIMCP_LOGGING_WARN("snn_network_load: truncated CSR pop header at %u", lp);
                    break;
                }

                /* Create lightweight population */
                int rc = snn_network_add_population_lightweight(net, pop_n, ntype, pop_name);
                if (rc < 0) {
                    NIMCP_LOGGING_ERROR("snn_network_load: failed to create CSR pop '%s'", pop_name);
                    /* Skip this pop's CSR data.
                     * Layout: [csr_n][csr_nnz][row_ptr][entries][has_gpu]
                     *         [if v7: has_homeostatic][if set: 3 × n_neurons floats] */
                    uint32_t csr_n, csr_nnz;
                    if (fread(&csr_n, 4, 1, f) == 1 && fread(&csr_nnz, 4, 1, f) == 1) {
                        fseek(f, (long)((csr_n + 1) * 4 + csr_nnz * sizeof(snn_csr_synapse_t)), SEEK_CUR);
                        uint8_t has_gpu;
                        if (fread(&has_gpu, 1, 1, f) == 1 && has_gpu)
                            fseek(f, (long)(csr_nnz * (4 + 4)), SEEK_CUR);
                        /* V7: also skip homeostatic block if present.
                         * Without this, the file cursor lands mid-array and
                         * subsequent population reads corrupt. */
                        if (wiring_schema >= 7) {
                            uint8_t has_homeo;
                            if (fread(&has_homeo, 1, 1, f) == 1 && has_homeo) {
                                /* 3 arrays × pop_n × sizeof(float) */
                                fseek(f, (long)(3 * pop_n * sizeof(float)),
                                       SEEK_CUR);
                            }
                        }
                    }
                    continue;
                }

                snn_population_t* pop = net->populations[rc];
                snn_csr_storage_t* csr = pop->incoming_csr;

                /* Read CSR arrays directly */
                uint32_t csr_n = 0, csr_nnz = 0;
                if (fread(&csr_n, sizeof(uint32_t), 1, f) != 1 ||
                    fread(&csr_nnz, sizeof(uint32_t), 1, f) != 1) {
                    NIMCP_LOGGING_WARN("snn_network_load: truncated CSR header for '%s'", pop_name);
                    break;
                }

                /* Resize CSR storage if needed */
                if (csr_nnz > csr->capacity) {
                    snn_csr_synapse_t* new_entries = nimcp_realloc(
                        csr->entries, csr_nnz * sizeof(snn_csr_synapse_t));
                    if (!new_entries) {
                        NIMCP_LOGGING_ERROR("snn_network_load: CSR realloc failed for %u synapses", csr_nnz);
                        break;
                    }
                    csr->entries = new_entries;
                    csr->capacity = csr_nnz;
                }

                /* Read row_ptr and entries */
                if (fread(csr->row_ptr, sizeof(uint32_t), csr_n + 1, f) != csr_n + 1 ||
                    fread(csr->entries, sizeof(snn_csr_synapse_t), csr_nnz, f) != csr_nnz) {
                    NIMCP_LOGGING_WARN("snn_network_load: truncated CSR data for '%s'", pop_name);
                    break;
                }
                csr->n_neurons = csr_n;
                csr->n_synapses = csr_nnz;
                csr->finalized = true;

                /* Read GPU-ready arrays if present */
                uint8_t has_gpu = 0;
                if (fread(&has_gpu, sizeof(uint8_t), 1, f) == 1 && has_gpu) {
                    csr->weights = nimcp_malloc(csr_nnz * sizeof(float));
                    csr->flat_col_idx = nimcp_malloc(csr_nnz * sizeof(uint32_t));
                    if (csr->weights && csr->flat_col_idx) {
                        if (fread(csr->weights, sizeof(float), csr_nnz, f) == csr_nnz &&
                            fread(csr->flat_col_idx, sizeof(uint32_t), csr_nnz, f) == csr_nnz) {
                            csr->gpu_ready = true;
                        }
                    }
                }

                /* === V7: Restore per-neuron homeostatic state ===
                 * Older schemas don't have these — keep the seed values
                 * initialized when the population was created. */
                if (wiring_schema >= 7) {
                    uint8_t has_homeostatic = 0;
                    if (fread(&has_homeostatic, sizeof(uint8_t), 1, f) == 1 && has_homeostatic) {
                        if (pop->threshold_offset) {
                            size_t r1 = fread(pop->threshold_offset, sizeof(float),
                                              pop->n_neurons, f);
                            if (r1 != pop->n_neurons) {
                                NIMCP_LOGGING_WARN("  pop '%s': truncated threshold_offset "
                                                    "(%zu of %u)", pop_name, r1, pop->n_neurons);
                            }
                        } else {
                            fseek(f, (long)(pop->n_neurons * sizeof(float)), SEEK_CUR);
                        }
                        if (pop->neuron_rate_ema) {
                            size_t r2 = fread(pop->neuron_rate_ema, sizeof(float),
                                              pop->n_neurons, f);
                            if (r2 != pop->n_neurons) {
                                NIMCP_LOGGING_WARN("  pop '%s': truncated rate_ema", pop_name);
                            }
                        } else {
                            fseek(f, (long)(pop->n_neurons * sizeof(float)), SEEK_CUR);
                        }
                        if (pop->depression) {
                            size_t r3 = fread(pop->depression, sizeof(float),
                                              pop->n_neurons, f);
                            if (r3 != pop->n_neurons) {
                                NIMCP_LOGGING_WARN("  pop '%s': truncated depression", pop_name);
                            }
                        } else {
                            fseek(f, (long)(pop->n_neurons * sizeof(float)), SEEK_CUR);
                        }
                        NIMCP_LOGGING_INFO("  pop '%s': restored homeostatic state "
                                           "(threshold_offset + rate_ema + depression)",
                                           pop_name);
                    }
                }

                NIMCP_LOGGING_INFO("  CSR pop '%s': %u neurons, %u synapses (gpu=%s)",
                                   pop_name, csr_n, csr_nnz, csr->gpu_ready ? "yes" : "no");
            }

            /* Re-create GPU LIF state for full neuron count */
            size_t total_n = 0;
            for (uint32_t p = 0; p < net->n_populations; p++) {
                if (net->populations[p]) total_n += net->populations[p]->n_neurons;
            }
            if (net->gpu_lif_state) {
                nimcp_lif_state_destroy((nimcp_lif_state_t*)net->gpu_lif_state);
                net->gpu_lif_state = NULL;
            }
            if (net->gpu_ctx) {
                nimcp_lif_params_t lp = {
                    .tau_mem = net->config.tau_mem,
                    .tau_syn = net->config.tau_syn > 0.0f ? net->config.tau_syn : 5.0f,
                    .v_thresh = net->config.v_thresh,
                    .v_reset = net->config.v_reset,
                    .v_rest = net->config.v_rest,
                    .dt = net->config.dt,
                    .hard_reset = true
                };
                nimcp_lif_state_t* ls = nimcp_lif_state_create(
                    (nimcp_gpu_context_t*)net->gpu_ctx, total_n, &lp);
                if (ls) {
                    net->gpu_lif_state = ls;
                    NIMCP_LOGGING_INFO("GPU LIF state created for %zu neurons (from checkpoint)", total_n);
                }
            }

            /* Prepare GPU CSR arrays if not loaded from checkpoint */
            uint32_t* pop_offsets = nimcp_calloc(net->n_populations + 1, sizeof(uint32_t));
            if (pop_offsets) {
                for (uint32_t p = 0; p < net->n_populations; p++) {
                    pop_offsets[p + 1] = pop_offsets[p] +
                        (net->populations[p] ? net->populations[p]->n_neurons : 0);
                }
                /* Audit bug #7: track GPU prep success/failure per pop.
                 * Without this, silent CPU fallback degrades perf 10-100x. */
                uint32_t gpu_ok = 0, gpu_fail = 0;
                for (uint32_t p = 0; p < net->n_populations; p++) {
                    snn_population_t* pop = net->populations[p];
                    if (pop && pop->lightweight && pop->incoming_csr && !pop->incoming_csr->gpu_ready) {
                        if (snn_csr_prepare_gpu(pop->incoming_csr, pop_offsets,
                                                net->n_populations) == 0) {
                            gpu_ok++;
                        } else {
                            gpu_fail++;
                            NIMCP_LOGGING_WARN("GPU prep failed for pop '%s' — "
                                               "falling back to CPU (perf hit)",
                                               pop->name);
                        }
                    }
                }
                if (gpu_fail > 0) {
                    NIMCP_LOGGING_WARN("snn_load: %u/%u CSR populations FAILED GPU prep",
                                       gpu_fail, gpu_ok + gpu_fail);
                } else if (gpu_ok > 0) {
                    NIMCP_LOGGING_INFO("snn_load: %u CSR populations GPU-ready",
                                       gpu_ok);
                }
                nimcp_free(pop_offsets);
            }
        }
    }

    fclose(f);
    NIMCP_LOGGING_INFO("SNN network loaded from %s (%u neurons restored, v%u)", path, restore_count, version);
    #undef fread
    #undef fclose
    #undef fseek
    return net;
}
