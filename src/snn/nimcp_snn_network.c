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
    const char* name
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

    /* Initialize spike trains */
    for (uint32_t i = 0; i < n_neurons; i++) {
        pop->spike_trains[i].neuron_id = i;
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
    nn_config.num_neurons = config->n_inputs + config->n_outputs;
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
        0, config->n_inputs, NEURON_GENERIC_LIF, "input");
    if (!network->input_pop) {
        /* Exception already thrown by snn_population_create_internal */
        snn_network_destroy(network);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "snn_network_create: network->input_pop is NULL");
        return NULL;
    }
    network->populations[0] = network->input_pop;
    network->n_populations = 1;

    /* Create output population */
    network->output_pop = snn_population_create_internal(
        1, config->n_outputs, NEURON_GENERIC_LIF, "output");
    if (!network->output_pop) {
        /* Exception already thrown by snn_population_create_internal */
        snn_network_destroy(network);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "snn_network_create: network->output_pop is NULL");
        return NULL;
    }
    network->populations[1] = network->output_pop;
    network->n_populations = 2;

    /* Create mutex for thread safety */
    network->mutex = nimcp_malloc(sizeof(nimcp_mutex_t));
    if (network->mutex) {
        nimcp_mutex_init((nimcp_mutex_t*)network->mutex, NULL);
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

            /* Get synaptic input from underlying neural network */
            float I_syn = 0.0f;
            if (network->neural_net && n < pop->n_neurons) {
                neuron_t* neuron = neural_network_get_neuron(
                    network->neural_net, pop->neuron_ids[n]);
                if (neuron) {
                    I_syn = neuron->external_current;
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

                /* Record spike */
                record_spike(&pop->spike_trains[n], network->sim->current_time_us);
                total_spikes++;
                pop->total_spikes++;
            }
        }
    }

    /* Update simulation time */
    network->sim->current_time_us += dt_us;
    network->sim->step_count++;

    /* Update statistics */
    network->stats.total_steps = network->sim->step_count;
    network->stats.total_spikes += total_spikes;

    return total_spikes;
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
                /* Scale input to current (simple linear mapping) */
                neuron->external_current = inputs[i] * 10.0f;  /* Scale factor */
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
    (void)lr;  /* TODO: Use lr in gradient descent when surrogate gradients are implemented */

    /* For now, just apply STDP as gradient approximation */
    return snn_network_apply_stdp(network);
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

    /* Forward pass */
    float outputs[256];  /* Max outputs */
    if (n_targets > 256) return -1.0f;

    int result = snn_network_forward(network, inputs, n_inputs,
                                     outputs, n_targets, duration_ms);
    if (result < 0) return -1.0f;

    /* Compute gradients */
    result = snn_network_compute_gradients(network, targets, n_targets);
    if (result != SNN_SUCCESS) return -1.0f;

    /* Apply gradients (STDP for now) */
    snn_network_apply_gradients(network, 0.0f);

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
        pop_id, n_neurons, neuron_type, name);
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
                float r = (float)rand() / (float)RAND_MAX;
                if (r > connectivity) continue;
            }

            /* Compute weight */
            float weight = weight_mean;
            if (weight_std > 0.0f) {
                /* Box-Muller for normal distribution */
                float u1 = (float)rand() / (float)RAND_MAX;
                float u2 = (float)rand() / (float)RAND_MAX;
                float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159f * u2);
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
