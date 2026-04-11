#include <stddef.h>  /* for NULL */
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
#include "snn/nimcp_snn_config.h"
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

    nimcp_free(pop);
}

/**
 * @brief Initialize simulation context
 */
static snn_simulation_t* snn_simulation_create_internal(float dt_ms) {
    snn_simulation_t* sim = (snn_simulation_t*)nimcp_malloc(sizeof(snn_simulation_t));
    if (!sim) {
        NIMCP_THROW_MEMORY(NIMCP_ERROR_NO_MEMORY, sizeof(snn_simulation_t),
            "snn_simulation_create: allocation failed");
        return NULL;
    }

    memset(sim, 0, sizeof(snn_simulation_t));
    sim->dt_ms = dt_ms;
    sim->health = SNN_STATE_HEALTHY;

    /* Allocate spike queue */
    sim->queue_capacity = 10000;
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

    /* Allocate population array */
    uint32_t max_populations = (config->n_populations > 0) ? config->n_populations : SNN_MAX_POPULATIONS;
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

    /* Create simulation context */
    network->sim = snn_simulation_create_internal(config->dt);
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

        /* Build input current vector on host from synaptic inputs */
        float* h_input = (float*)nimcp_malloc(total_neurons * sizeof(float));
        if (h_input) {
            size_t neuron_offset = 0;
            for (uint32_t p = 0; p < network->n_populations; p++) {
                snn_population_t* pop = network->populations[p];
                if (!pop) continue;

                for (uint32_t n = 0; n < pop->n_neurons; n++) {
                    float I_syn = 0.0f;
                    if (network->neural_net) {
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
                    h_input[neuron_offset + n] = I_syn;
                }
                neuron_offset += pop->n_neurons;
            }

            /* Upload input to GPU tensor */
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
                        neuron_offset = 0;
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

                                    if (network->neural_net) {
                                        neuron_t* nn = neural_network_get_neuron(
                                            network->neural_net, pop->neuron_ids[n]);
                                        if (nn) nn->state = 1.0f;
                                    }
                                } else {
                                    if (network->neural_net) {
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
                    }
                } else if (!gpu_ok) {
                    NIMCP_LOGGING_WARN("snn_network_step: GPU LIF forward failed, will use CPU next step");
                }

                nimcp_gpu_tensor_destroy(input_tensor);
            }
            nimcp_free(h_input);
        }
    } else {
    /* ===== CPU FALLBACK PATH ===== */

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

        /* Update each neuron */
        for (uint32_t n = 0; n < pop->n_neurons; n++) {
            spike_data[n] = 0.0f;

            /* Check refractory period */
            if (ref_data[n] > 0.0f) {
                ref_data[n] -= dt_ms;
                continue;
            }

            /* Get synaptic input: external current + weighted sum of presynaptic spikes */
            float I_syn = 0.0f;
            if (network->neural_net && n < pop->n_neurons) {
                neuron_t* neuron = neural_network_get_neuron(
                    network->neural_net, pop->neuron_ids[n]);
                if (neuron) {
                    I_syn = neuron->external_current;
                    /* Sum incoming synaptic currents from neurons that spiked last step.
                     * Each incoming synapse contributes weight * presynaptic_spike. */
                    uint32_t in_count = neuron->incoming.embedded_count
                                      + neuron->incoming.overflow_count;
                    for (uint32_t s = 0; s < in_count; s++) {
                        synapse_handle_t* h = sparse_synapse_get(&neuron->incoming, s);
                        if (!h) continue;
                        /* Check if presynaptic neuron spiked (look up its spike output) */
                        uint32_t pre_id = h->target_neuron_id; /* incoming: target = source */
                        neuron_t* pre = neural_network_get_neuron(network->neural_net, pre_id);
                        if (pre && pre->state > 0.5f) {
                            I_syn += h->weight;
                        }
                    }
                }
            }

            /* LIF dynamics: dV/dt = (V_rest - V + I) / tau_mem */
            float dv = (v_rest - v_data[n] + I_syn) / tau_mem * dt_ms;
            v_data[n] += dv;

            /* Spike generation */
            if (v_data[n] >= v_thresh) {
                spike_data[n] = 1.0f;
                v_data[n] = v_reset;
                ref_data[n] = network->config.t_ref;

                /* Record spike + propagate to neural_net for synaptic current summation */
                record_spike(&pop->spike_trains[n], network->sim->current_time_us);
                total_spikes++;
                pop->total_spikes++;

                /* Set activation=1 on underlying neuron so downstream neurons detect the spike */
                if (network->neural_net) {
                    neuron_t* spiked = neural_network_get_neuron(
                        network->neural_net, pop->neuron_ids[n]);
                    if (spiked) spiked->state = 1.0f;
                }
            } else {
                /* No spike: clear activation for this step */
                if (network->neural_net) {
                    neuron_t* quiet = neural_network_get_neuron(
                        network->neural_net, pop->neuron_ids[n]);
                    if (quiet) quiet->state = 0.0f;
                }
            }
        }
    }
    } /* end CPU fallback */

    /* Update simulation time */
    network->sim->current_time_us += dt_us;
    network->sim->step_count++;

    /* Update statistics */
    network->stats.total_steps = network->sim->step_count;
    network->stats.total_spikes += total_spikes;

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
    for (uint32_t i = 0; i < n_inputs; i++) {
        if (network->neural_net && i < network->config.n_inputs) {
            neuron_t* neuron = neural_network_get_neuron(
                network->neural_net, network->input_pop->neuron_ids[i]);
            if (neuron) {
                /* Scale input to current.
             * Neurons need ~20mV above resting (-70mV) to reach threshold (-50mV).
             * After average-pooling, BERT features are typically [-0.5, 0.5].
             * Apply ReLU (only positive features drive spikes) then scale.
             * Scale factor maps: input 0.3 → 21mV (fires), 0.1 → 7mV (sub).
             * This creates sparse activation from the strongest features. */
                float inp = inputs[i];
                if (inp > 0.0f) {
                    neuron->external_current = inp * network->config.input_current_scale;
                } else {
                    neuron->external_current = 0.0f;  /* No inhibitory drive from negative features */
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
    if (network->n_populations >= SNN_MAX_POPULATIONS) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_SIZE,
            "snn_network_add_population: max populations exceeded");
        return SNN_ERROR_INVALID_DIMENSION;
    }

    uint32_t pop_id = network->n_populations;

    snn_population_t* pop = snn_population_create_internal(
        pop_id, n_neurons, neuron_type, name, 0);
    if (!pop) return SNN_ERROR_OUT_OF_MEMORY;

    /* Add neurons to underlying neural network */
    for (uint32_t i = 0; i < n_neurons; i++) {
        uint32_t neuron_id = neural_network_add_neuron(
            network->neural_net, ACTIVATION_SIGMOID);
        pop->neuron_ids[i] = neuron_id;
    }

    network->populations[pop_id] = pop;
    network->n_populations++;

    NIMCP_LOGGING_DEBUG("snn_network_add_population: added '%s' with %u neurons",
                       name ? name : "unnamed", n_neurons);

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

    for (uint32_t i = 0; i < src->n_neurons; i++) {
        for (uint32_t j = 0; j < dst->n_neurons; j++) {
            /* Apply connectivity probability */
            if (topology == SNN_TOPO_RANDOM) {
                float r = (float)nimcp_tl_rand() / (float)RAND_MAX;
                if (r > connectivity) continue;
            }

            /* Compute weight */
            float weight = weight_mean;
            if (weight_std > 0.0f) {
                /* Box-Muller for normal distribution */
                float u1 = (float)nimcp_tl_rand() / (float)RAND_MAX;
                float u2 = (float)nimcp_tl_rand() / (float)RAND_MAX;
                if (u1 < 1e-7f) u1 = 1e-7f;
                float z = sqrtf(-2.0f * logf(u1)) * cosf(NIMCP_TWO_PI_F * u2);
                weight += z * weight_std;
            }

            /* Add connection with synapse type */
            bool success = neural_network_add_connection_typed(
                network->neural_net,
                src->neuron_ids[i],
                dst->neuron_ids[j],
                weight,
                synapse_type);

            if (success) n_connections++;
        }
    }

    NIMCP_LOGGING_DEBUG("snn_network_connect_populations: created %d connections",
                       n_connections);

    return n_connections;
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
#include "core/neuralnet/nimcp_sparse_synapse.h"

/* Checked fwrite helper — returns -1 on failure */
#define SNN_FWRITE(ptr, size, count, stream) \
    do { if (fwrite((ptr), (size), (count), (stream)) != (count)) { \
        NIMCP_LOGGING_ERROR("snn_network_save: fwrite failed"); \
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

    uint32_t magic = 0x534E4E53; /* "SNNS" */
    uint32_t version = 2;  /* v2: saves target neuron IDs per synapse */
    SNN_FWRITE(&magic, sizeof(uint32_t), 1, f);
    SNN_FWRITE(&version, sizeof(uint32_t), 1, f);

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

    fclose(f);

    /* Atomic rename: temp → final path */
    if (rename(tmp_path, path) != 0) {
        NIMCP_LOGGING_ERROR("snn_network_save: rename %s → %s failed", tmp_path, path);
        remove(tmp_path);
        return -1;
    }

    NIMCP_LOGGING_INFO("SNN network saved to %s (%u neurons)", path, total_neurons);
    return 0;
}

#undef SNN_FWRITE

snn_network_t* snn_network_load(const char* path) {
    if (!path) return NULL;

    FILE* f = fopen(path, "rb");
    if (!f) {
        NIMCP_LOGGING_WARN("snn_network_load: file not found %s", path);
        return NULL;
    }

    uint32_t magic = 0, version = 0;
    if (fread(&magic, sizeof(uint32_t), 1, f) != 1 || magic != 0x534E4E53) {
        NIMCP_LOGGING_ERROR("snn_network_load: invalid magic");
        fclose(f);
        return NULL;
    }
    if (fread(&version, sizeof(uint32_t), 1, f) != 1 || version > 2) {
        NIMCP_LOGGING_ERROR("snn_network_load: unsupported version %u", version);
        fclose(f); return NULL;
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

    fclose(f);
    NIMCP_LOGGING_INFO("SNN network loaded from %s (%u neurons restored)", path, restore_count);
    return net;
}
