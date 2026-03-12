/**
 * @file nimcp_lnn_network.c
 * @brief LNN Network Implementation
 *
 * WHAT: Multi-layer Liquid Neural Network implementation
 * WHY:  Provides continuous-time temporal processing for NIMCP
 * HOW:  Sequential layer propagation, state history management, training integration
 *
 * @author NIMCP Development Team
 * @date 2025-12-20
 */

#include "lnn/nimcp_lnn_network.h"
#include "lnn/nimcp_lnn_layer.h"
#include "lnn/nimcp_lnn_config.h"
#include "utils/tensor/nimcp_tensor.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/platform/nimcp_platform_mutex.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/thread/nimcp_atomic.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

/*=============================================================================
 * Health Agent Forward Declarations (Phase 8: Heartbeat for Long Operations)
 *============================================================================*/
#include "utils/fault_tolerance/nimcp_health_agent_macros.h"

NIMCP_DECLARE_HEALTH_AGENT_ATOMIC(lnn_network)

#include <stddef.h>  /* for NULL */
// Thread-safe network ID counter
static nimcp_atomic_uint32_t g_lnn_network_id_counter = {0};

/*=============================================================================
 * Network Lifecycle
 *===========================================================================*/

/**
 * @brief Create LNN network from configuration
 *
 * WHAT: Allocate and initialize multi-layer LNN network
 * WHY:  Primary network creation method
 * HOW:  Validate config, create layers, allocate history buffer, initialize mutex
 */
lnn_network_t* lnn_network_create(const lnn_config_t* config) {
    // Guard: Validate inputs
    if (!config) {
        NIMCP_LOGGING_ERROR("lnn_network_create: NULL config");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "lnn_network_create: config is NULL");
        return NULL;
    }

    if (lnn_config_validate(config) != LNN_SUCCESS) {
        NIMCP_LOGGING_ERROR("lnn_network_create: Invalid config");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "lnn_network_create: validation failed");
        return NULL;
    }

    // Allocate network structure
    lnn_network_t* network = (lnn_network_t*)nimcp_calloc(1, sizeof(lnn_network_t));
    if (!network) {
        NIMCP_LOGGING_ERROR("lnn_network_create: Failed to allocate network");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "lnn_network_create: failed to allocate network");
        return NULL;
    }

    // Initialize basic fields with thread-safe ID generation
    network->id = nimcp_atomic_fetch_add_u32(&g_lnn_network_id_counter, 1, NIMCP_MEMORY_ORDER_SEQ_CST);
    snprintf(network->name, sizeof(network->name), "lnn_network_%u", network->id);
    network->n_layers = config->n_layers;
    network->n_inputs = config->n_inputs;
    network->n_outputs = config->n_outputs;
    network->train_mode = config->train_mode;
    network->is_training = false;

    // Deep copy configuration (shallow copy would leave shared pointers)
    network->config = (lnn_config_t*)nimcp_calloc(1, sizeof(lnn_config_t));
    if (!network->config) {
        NIMCP_LOGGING_ERROR("lnn_network_create: Failed to allocate config");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "lnn_network_create: failed to allocate config");
        nimcp_free(network);
        return NULL;
    }
    memcpy(network->config, config, sizeof(lnn_config_t));

    // Deep copy layer_configs if present
    if (config->layer_configs && config->n_layers > 0) {
        network->config->layer_configs = (lnn_layer_config_t*)nimcp_calloc(
            config->n_layers, sizeof(lnn_layer_config_t));
        if (!network->config->layer_configs) {
            NIMCP_LOGGING_ERROR("lnn_network_create: Failed to allocate layer configs");
            NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "lnn_network_create: failed to allocate layer configs");
            nimcp_free(network->config);
            nimcp_free(network);
            return NULL;
        }
        memcpy(network->config->layer_configs, config->layer_configs,
               config->n_layers * sizeof(lnn_layer_config_t));
    }

    // Allocate layer array
    network->layers = (lnn_layer_t**)nimcp_calloc(network->n_layers, sizeof(lnn_layer_t*));
    if (!network->layers) {
        NIMCP_LOGGING_ERROR("lnn_network_create: Failed to allocate layer array");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "lnn_network_create: failed to allocate layer array");
        lnn_config_destroy(network->config);
        nimcp_free(network->config);
        nimcp_free(network);
        return NULL;
    }

    // Create layers
    for (uint32_t i = 0; i < network->n_layers; i++) {
        uint32_t layer_input_size = (i == 0) ? config->n_inputs :
                                     config->layer_configs[i-1].n_neurons;

        network->layers[i] = lnn_layer_create(&config->layer_configs[i], layer_input_size);
        if (!network->layers[i]) {
            NIMCP_LOGGING_ERROR("lnn_network_create: Failed to create layer %u", i);
            // Cleanup previously created layers
            for (uint32_t j = 0; j < i; j++) {
                lnn_layer_destroy(network->layers[j]);
            }
            nimcp_free(network->layers);
            lnn_config_destroy(network->config);
            nimcp_free(network->config);
            nimcp_free(network);
            NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "lnn_network_create: failed to create layer");
            return NULL;
        }
    }

    // Allocate state history if preallocate enabled
    if (config->preallocate_history && config->bptt_truncation > 0) {
        network->history_capacity = config->bptt_truncation;
        network->state_history = (nimcp_tensor_t**)nimcp_calloc(
            network->history_capacity, sizeof(nimcp_tensor_t*)
        );
        if (!network->state_history) {
            NIMCP_LOGGING_WARN("lnn_network_create: Failed to preallocate history");
            network->history_capacity = 0;
        }
    }

    // Create mutex
    network->mutex = nimcp_platform_mutex_create();
    if (!network->mutex) {
        NIMCP_LOGGING_WARN("lnn_network_create: Failed to create mutex");
    }

    // Initialize statistics
    memset(&network->stats, 0, sizeof(lnn_network_stats_t));
    network->stats.health = LNN_STATE_VALID;

    NIMCP_LOGGING_INFO("Created LNN network '%s' with %u layers (%u inputs, %u outputs)",
                       network->name, network->n_layers, network->n_inputs, network->n_outputs);

    return network;
}

/**
 * @brief Create NCP (Neural Circuit Policy) network
 *
 * WHAT: Convenience constructor for NCP architecture
 * WHY:  NCP is standard LNN architecture from original paper
 * HOW:  Create config with lnn_config_ncp, then call lnn_network_create
 */
lnn_network_t* lnn_network_create_ncp(
    uint32_t n_inputs,
    uint32_t n_inter,
    uint32_t n_command,
    uint32_t n_outputs
) {
    // Guard: Validate inputs
    if (n_inputs == 0 || n_inter == 0 || n_command == 0 || n_outputs == 0) {
        NIMCP_LOGGING_ERROR("lnn_network_create_ncp: Invalid dimensions");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "lnn_network_create_ncp: n_inputs is zero");
        return NULL;
    }

    // Create NCP configuration
    lnn_config_t config;
    if (lnn_config_ncp(&config, n_inputs, n_inter, n_command, n_outputs) != LNN_SUCCESS) {
        NIMCP_LOGGING_ERROR("lnn_network_create_ncp: Failed to create NCP config");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "lnn_network_create_ncp: validation failed");
        return NULL;
    }

    // Create network from config
    lnn_network_t* network = lnn_network_create(&config);

    // Cleanup config (network owns a copy)
    lnn_config_destroy(&config);

    return network;
}

/**
 * @brief Destroy network and free all resources
 *
 * WHAT: Clean shutdown of network
 * WHY:  Prevent memory leaks
 * HOW:  Free layers, history, config, mutex, network struct
 */
void lnn_network_destroy(lnn_network_t* network) {
    // Guard: NULL-safe
    if (!network) {
        return;
    }

    // Destroy layers
    if (network->layers) {
        for (uint32_t i = 0; i < network->n_layers; i++) {
            lnn_layer_destroy(network->layers[i]);
            network->layers[i] = NULL;  // Prevent use-after-free
        }
        nimcp_free(network->layers);
        network->layers = NULL;  // Prevent use-after-free
    }

    // Clear and free state history
    lnn_network_clear_history(network);
    if (network->state_history) {
        nimcp_free(network->state_history);
        network->state_history = NULL;  // Prevent use-after-free
    }

    // Destroy config
    if (network->config) {
        lnn_config_destroy(network->config);
        nimcp_free(network->config);
        network->config = NULL;  // Prevent use-after-free
    }

    // Destroy mutex
    if (network->mutex) {
        nimcp_platform_mutex_destroy((nimcp_platform_mutex_t*)network->mutex);
        nimcp_free(network->mutex);
        network->mutex = NULL;
        network->mutex = NULL;  // Prevent use-after-free
    }

    // Free network structure
    nimcp_free(network);
}

/**
 * @brief Initialize network weights
 *
 * WHAT: Initialize all layer weights with random values
 * WHY:  Proper initialization critical for training convergence
 * HOW:  Call lnn_layer_init_weights on each layer
 */
int lnn_network_init_weights(lnn_network_t* network, uint64_t seed) {
    // Guard: Validate inputs
    if (!network) {
        NIMCP_LOGGING_ERROR("lnn_network_init_weights: NULL network");
        return LNN_ERROR_NULL_POINTER;
    }

    // Use current time if seed is 0
    if (seed == 0) {
        seed = (uint64_t)time(NULL);
    }

    // Initialize each layer with unique seed
    for (uint32_t i = 0; i < network->n_layers; i++) {
        float std = network->config->layer_configs[i].weight_init_std;
        uint64_t layer_seed = seed + i;

        int result = lnn_layer_init_weights(network->layers[i], std, layer_seed);
        if (result != LNN_SUCCESS) {
            NIMCP_LOGGING_ERROR("lnn_network_init_weights: Failed to init layer %u", i);
            return result;
        }
    }

    NIMCP_LOGGING_INFO("Initialized weights for network '%s' with seed %lu",
                       network->name, seed);
    return LNN_SUCCESS;
}

/*=============================================================================
 * Forward Pass
 *===========================================================================*/

/**
 * @brief Single forward step with given input
 *
 * WHAT: Propagate input through all layers for one time step
 * WHY:  Core inference operation
 * HOW:  Sequential layer forward, record state if training
 */
int lnn_network_forward_step(
    lnn_network_t* network,
    const nimcp_tensor_t* input,
    nimcp_tensor_t* output,
    float dt
) {
    // Guard: Validate inputs
    if (!network || !input || !output) {
        NIMCP_LOGGING_ERROR("lnn_network_forward_step: NULL argument");
        return LNN_ERROR_NULL_POINTER;
    }

    /* Phase 8: Send heartbeat at start of forward step */
    lnn_network_heartbeat("lnn_forward_step", 0.0f);

    // Use default dt if not specified
    if (dt <= 0.0f) {
        dt = network->config->default_dt;
    }

    // Lock for thread safety
    if (network->mutex) {
        nimcp_platform_mutex_lock((nimcp_platform_mutex_t*)network->mutex);
    }

    // Temporary storage for layer outputs
    const nimcp_tensor_t* layer_input = input;
    nimcp_tensor_t* layer_output = NULL;

    // Forward through each layer
    for (uint32_t i = 0; i < network->n_layers; i++) {
        // For last layer, write directly to output
        if (i == network->n_layers - 1) {
            layer_output = output;
        } else {
            // Allocate temporary output for intermediate layers
            uint32_t next_size = network->layers[i]->n_neurons;
            uint32_t dims[1] = {next_size};
            layer_output = nimcp_tensor_create(dims, 1, NIMCP_DTYPE_F32);
            if (!layer_output) {
                NIMCP_LOGGING_ERROR("lnn_network_forward_step: Failed to allocate layer output");
                if (network->mutex) {
                    nimcp_platform_mutex_unlock((nimcp_platform_mutex_t*)network->mutex);
                }
                return LNN_ERROR_OUT_OF_MEMORY;
            }
        }

        // Forward through layer
        int result = lnn_layer_forward(network->layers[i], layer_input, layer_output, dt);
        if (result != LNN_SUCCESS) {
            NIMCP_LOGGING_ERROR("lnn_network_forward_step: Layer %u forward failed", i);
            if (i < network->n_layers - 1) {
                nimcp_tensor_destroy(layer_output);
            }
            if (network->mutex) {
                nimcp_platform_mutex_unlock((nimcp_platform_mutex_t*)network->mutex);
            }
            return result;
        }

        // Free previous layer input if it was temporary
        if (i > 0 && layer_input != input) {
            nimcp_tensor_destroy((nimcp_tensor_t*)layer_input);
        }

        // Move output to input for next layer
        layer_input = layer_output;
    }

    // Record state if training
    if (network->is_training) {
        lnn_network_record_state(network);
    }

    // Update statistics
    network->stats.forward_steps++;

    if (network->mutex) {
        nimcp_platform_mutex_unlock((nimcp_platform_mutex_t*)network->mutex);
    }

    return LNN_SUCCESS;
}

/**
 * @brief Forward pass over sequence
 *
 * WHAT: Process sequence of inputs, producing sequence of outputs
 * WHY:  Temporal sequence processing
 * HOW:  Loop over sequence, calling forward_step for each timestep
 */
int lnn_network_forward_sequence(
    lnn_network_t* network,
    const nimcp_tensor_t* inputs,
    nimcp_tensor_t* outputs,
    uint32_t seq_len,
    float dt
) {
    // Guard: Validate inputs
    if (!network || !inputs || !outputs) {
        NIMCP_LOGGING_ERROR("lnn_network_forward_sequence: NULL argument");
        return LNN_ERROR_NULL_POINTER;
    }

    if (seq_len == 0) {
        NIMCP_LOGGING_ERROR("lnn_network_forward_sequence: Zero sequence length");
        return LNN_ERROR_INVALID_DIMENSION;
    }

    // Process each time step
    for (uint32_t t = 0; t < seq_len; t++) {
        // Extract input at time t
        uint32_t input_dims[1] = {network->n_inputs};
        nimcp_tensor_t* input_t = nimcp_tensor_create(input_dims, 1, NIMCP_DTYPE_F32);
        if (!input_t) {
            NIMCP_LOGGING_ERROR("lnn_network_forward_sequence: Failed to allocate input");
            return LNN_ERROR_OUT_OF_MEMORY;
        }

        // Copy input slice [t, :] to input_t
        for (uint32_t i = 0; i < network->n_inputs; i++) {
            float val = nimcp_tensor_get_flat(inputs, t * network->n_inputs + i);
            nimcp_tensor_set_flat(input_t, i, val);
        }

        // Allocate output for this timestep
        uint32_t output_dims[1] = {network->n_outputs};
        nimcp_tensor_t* output_t = nimcp_tensor_create(output_dims, 1, NIMCP_DTYPE_F32);
        if (!output_t) {
            NIMCP_LOGGING_ERROR("lnn_network_forward_sequence: Failed to allocate output");
            nimcp_tensor_destroy(input_t);
            return LNN_ERROR_OUT_OF_MEMORY;
        }

        // Forward step
        int result = lnn_network_forward_step(network, input_t, output_t, dt);
        nimcp_tensor_destroy(input_t);

        if (result != LNN_SUCCESS) {
            NIMCP_LOGGING_ERROR("lnn_network_forward_sequence: Failed at timestep %u", t);
            nimcp_tensor_destroy(output_t);
            return result;
        }

        // Copy output_t to outputs[t, :]
        for (uint32_t i = 0; i < network->n_outputs; i++) {
            float val = nimcp_tensor_get_flat(output_t, i);
            nimcp_tensor_set_flat(outputs, t * network->n_outputs + i, val);
        }

        nimcp_tensor_destroy(output_t);
    }

    return LNN_SUCCESS;
}

/**
 * @brief Parallel forward over batch of sequences
 *
 * WHAT: Process batch of sequences in parallel
 * WHY:  Efficient batch processing
 * HOW:  Distribute batch across threads (if thread pool connected)
 *
 * NOTE: Simplified implementation - sequential for now (parallel requires thread pool integration)
 */
int lnn_network_forward_batch(
    lnn_network_t* network,
    const nimcp_tensor_t* inputs,
    nimcp_tensor_t* outputs,
    uint32_t batch_size,
    uint32_t seq_len,
    float dt
) {
    // Guard: Validate inputs
    if (!network || !inputs || !outputs) {
        NIMCP_LOGGING_ERROR("lnn_network_forward_batch: NULL argument");
        return LNN_ERROR_NULL_POINTER;
    }

    // For now, process sequentially (TODO: parallelize with thread pool)
    for (uint32_t b = 0; b < batch_size; b++) {
        // Extract batch element [b, :, :]
        uint32_t batch_input_dims[1] = {seq_len * network->n_inputs};
        nimcp_tensor_t* batch_input = nimcp_tensor_create(batch_input_dims, 1, NIMCP_DTYPE_F32);

        uint32_t batch_output_dims[1] = {seq_len * network->n_outputs};
        nimcp_tensor_t* batch_output = nimcp_tensor_create(batch_output_dims, 1, NIMCP_DTYPE_F32);

        if (!batch_input || !batch_output) {
            NIMCP_LOGGING_ERROR("lnn_network_forward_batch: Failed to allocate batch tensors");
            if (batch_input) nimcp_tensor_destroy(batch_input);
            if (batch_output) nimcp_tensor_destroy(batch_output);
            return LNN_ERROR_OUT_OF_MEMORY;
        }

        // Copy batch slice
        size_t batch_offset = b * seq_len * network->n_inputs;
        for (uint32_t i = 0; i < seq_len * network->n_inputs; i++) {
            float val = nimcp_tensor_get_flat(inputs, batch_offset + i);
            nimcp_tensor_set_flat(batch_input, i, val);
        }

        // Process sequence
        int result = lnn_network_forward_sequence(network, batch_input, batch_output, seq_len, dt);

        if (result != LNN_SUCCESS) {
            NIMCP_LOGGING_ERROR("lnn_network_forward_batch: Failed at batch %u", b);
            nimcp_tensor_destroy(batch_input);
            nimcp_tensor_destroy(batch_output);
            return result;
        }

        // Copy result back
        size_t output_offset = b * seq_len * network->n_outputs;
        for (uint32_t i = 0; i < seq_len * network->n_outputs; i++) {
            float val = nimcp_tensor_get_flat(batch_output, i);
            nimcp_tensor_set_flat(outputs, output_offset + i, val);
        }

        nimcp_tensor_destroy(batch_input);
        nimcp_tensor_destroy(batch_output);
    }

    return LNN_SUCCESS;
}

/*=============================================================================
 * State Management
 *===========================================================================*/

/**
 * @brief Get current network state (all layers)
 *
 * WHAT: Concatenate state from all layers
 * WHY:  State checkpointing, introspection
 * HOW:  Gather x from each layer, concatenate
 */
/**
 * Internal unlocked helper — caller MUST hold network->mutex.
 */
static int lnn_network_get_state_unlocked(const lnn_network_t* network, nimcp_tensor_t** state) {
    // Calculate total state size
    uint32_t total_size = 0;
    for (uint32_t i = 0; i < network->n_layers; i++) {
        total_size += network->layers[i]->n_neurons;
    }

    // Allocate output tensor
    uint32_t dims[1] = {total_size};
    *state = nimcp_tensor_create(dims, 1, NIMCP_DTYPE_F32);
    if (!*state) {
        NIMCP_LOGGING_ERROR("lnn_network_get_state: Failed to allocate state tensor");
        return LNN_ERROR_OUT_OF_MEMORY;
    }

    // Concatenate layer states
    uint32_t offset = 0;
    for (uint32_t i = 0; i < network->n_layers; i++) {
        nimcp_tensor_t* layer_state = NULL;
        int result = lnn_layer_get_state(network->layers[i], &layer_state);
        if (result != LNN_SUCCESS) {
            NIMCP_LOGGING_ERROR("lnn_network_get_state: Failed to get layer %u state", i);
            nimcp_tensor_destroy(*state);
            *state = NULL;
            return result;
        }

        // Copy layer state to output
        for (uint32_t j = 0; j < network->layers[i]->n_neurons; j++) {
            float val = nimcp_tensor_get_flat(layer_state, j);
            nimcp_tensor_set_flat(*state, offset + j, val);
        }

        offset += network->layers[i]->n_neurons;
        nimcp_tensor_destroy(layer_state);
    }

    return LNN_SUCCESS;
}

int lnn_network_get_state(const lnn_network_t* network, nimcp_tensor_t** state) {
    // Guard: Validate inputs
    if (!network || !state) {
        NIMCP_LOGGING_ERROR("lnn_network_get_state: NULL argument");
        return LNN_ERROR_NULL_POINTER;
    }

    /* Lock mutex to prevent race with forward_step which destroys/replaces
     * layer->x tensors.  Without this lock, a concurrent forward_step can
     * free layer->x while we are cloning it, leading to use-after-free
     * (manifests as "nimcp_tensor_create: t->shape.numel is zero"). */
    if (network->mutex) {
        nimcp_platform_mutex_lock((nimcp_platform_mutex_t*)network->mutex);
    }

    int result = lnn_network_get_state_unlocked(network, state);

    if (network->mutex) {
        nimcp_platform_mutex_unlock((nimcp_platform_mutex_t*)network->mutex);
    }

    return result;
}

/**
 * @brief Set network state (all layers)
 *
 * WHAT: Restore network state from tensor
 * WHY:  State restoration from checkpoint
 * HOW:  Split state tensor, set each layer's state
 */
int lnn_network_set_state(lnn_network_t* network, const nimcp_tensor_t* state) {
    // Guard: Validate inputs
    if (!network || !state) {
        NIMCP_LOGGING_ERROR("lnn_network_set_state: NULL argument");
        return LNN_ERROR_NULL_POINTER;
    }

    /* Lock mutex — set_state writes to layer->x which forward_step also modifies. */
    if (network->mutex) {
        nimcp_platform_mutex_lock((nimcp_platform_mutex_t*)network->mutex);
    }

    // Split state and set each layer
    uint32_t offset = 0;
    for (uint32_t i = 0; i < network->n_layers; i++) {
        uint32_t layer_size = network->layers[i]->n_neurons;

        // Extract layer state slice
        uint32_t dims[1] = {layer_size};
        nimcp_tensor_t* layer_state = nimcp_tensor_create(dims, 1, NIMCP_DTYPE_F32);
        if (!layer_state) {
            NIMCP_LOGGING_ERROR("lnn_network_set_state: Failed to allocate layer state");
            if (network->mutex) {
                nimcp_platform_mutex_unlock((nimcp_platform_mutex_t*)network->mutex);
            }
            return LNN_ERROR_OUT_OF_MEMORY;
        }

        // Copy slice
        for (uint32_t j = 0; j < layer_size; j++) {
            float val = nimcp_tensor_get_flat(state, offset + j);
            nimcp_tensor_set_flat(layer_state, j, val);
        }

        // Set layer state
        int result = lnn_layer_set_state(network->layers[i], layer_state);
        nimcp_tensor_destroy(layer_state);

        if (result != LNN_SUCCESS) {
            NIMCP_LOGGING_ERROR("lnn_network_set_state: Failed to set layer %u state", i);
            if (network->mutex) {
                nimcp_platform_mutex_unlock((nimcp_platform_mutex_t*)network->mutex);
            }
            return result;
        }

        offset += layer_size;
    }

    if (network->mutex) {
        nimcp_platform_mutex_unlock((nimcp_platform_mutex_t*)network->mutex);
    }

    return LNN_SUCCESS;
}

/**
 * @brief Get current time constants (all layers)
 *
 * WHAT: Concatenate τ from all layers
 * WHY:  Introspection of learned dynamics
 * HOW:  Gather tau from each layer, concatenate
 */
int lnn_network_get_tau(const lnn_network_t* network, nimcp_tensor_t** tau) {
    // Guard: Validate inputs
    if (!network || !tau) {
        NIMCP_LOGGING_ERROR("lnn_network_get_tau: NULL argument");
        return LNN_ERROR_NULL_POINTER;
    }

    /* Lock mutex — tau tensors are replaced during lnn_layer_compute_tau
     * which runs inside forward_step under the same mutex. */
    if (network->mutex) {
        nimcp_platform_mutex_lock((nimcp_platform_mutex_t*)network->mutex);
    }

    // Calculate total size
    uint32_t total_size = 0;
    for (uint32_t i = 0; i < network->n_layers; i++) {
        total_size += network->layers[i]->n_neurons;
    }

    // Allocate output tensor
    uint32_t dims[1] = {total_size};
    *tau = nimcp_tensor_create(dims, 1, NIMCP_DTYPE_F32);
    if (!*tau) {
        NIMCP_LOGGING_ERROR("lnn_network_get_tau: Failed to allocate tau tensor");
        if (network->mutex) {
            nimcp_platform_mutex_unlock((nimcp_platform_mutex_t*)network->mutex);
        }
        return LNN_ERROR_OUT_OF_MEMORY;
    }

    // Concatenate layer tau values
    uint32_t offset = 0;
    for (uint32_t i = 0; i < network->n_layers; i++) {
        nimcp_tensor_t* layer_tau = NULL;
        int result = lnn_layer_get_tau(network->layers[i], &layer_tau);
        if (result != LNN_SUCCESS) {
            NIMCP_LOGGING_ERROR("lnn_network_get_tau: Failed to get layer %u tau", i);
            nimcp_tensor_destroy(*tau);
            *tau = NULL;
            if (network->mutex) {
                nimcp_platform_mutex_unlock((nimcp_platform_mutex_t*)network->mutex);
            }
            return result;
        }

        // Copy layer tau to output
        for (uint32_t j = 0; j < network->layers[i]->n_neurons; j++) {
            float val = nimcp_tensor_get_flat(layer_tau, j);
            nimcp_tensor_set_flat(*tau, offset + j, val);
        }

        offset += network->layers[i]->n_neurons;
        nimcp_tensor_destroy(layer_tau);
    }

    if (network->mutex) {
        nimcp_platform_mutex_unlock((nimcp_platform_mutex_t*)network->mutex);
    }

    return LNN_SUCCESS;
}

/**
 * @brief Reset network state to zero
 *
 * WHAT: Zero out all layer states
 * WHY:  Between sequences
 * HOW:  Call lnn_layer_reset on each layer
 */
void lnn_network_reset_state(lnn_network_t* network) {
    // Guard: NULL-safe
    if (!network) {
        return;
    }

    // Reset each layer
    for (uint32_t i = 0; i < network->n_layers; i++) {
        lnn_layer_reset(network->layers[i]);
    }
}

/**
 * @brief Reset accumulated gradients to zero
 *
 * WHAT: Zero all gradient tensors
 * WHY:  Between training batches
 * HOW:  Call lnn_layer_reset_gradients on each layer
 */
void lnn_network_reset_gradients(lnn_network_t* network) {
    // Guard: NULL-safe
    if (!network) {
        return;
    }

    // Reset each layer
    for (uint32_t i = 0; i < network->n_layers; i++) {
        lnn_layer_reset_gradients(network->layers[i]);
    }
}

/*=============================================================================
 * Training Mode
 *===========================================================================*/

/**
 * @brief Set training mode
 *
 * WHAT: Enable/disable training mode
 * WHY:  Training mode records state history for BPTT
 * HOW:  Set flag, allocate history if needed
 */
void lnn_network_set_training(lnn_network_t* network, bool training) {
    // Guard: Validate input
    if (!network) {
        return;
    }

    network->is_training = training;

    // Allocate history buffer if entering training mode and not preallocated
    if (training && !network->state_history && network->config->bptt_truncation > 0) {
        network->history_capacity = network->config->bptt_truncation;
        network->state_history = (nimcp_tensor_t**)nimcp_calloc(
            network->history_capacity, sizeof(nimcp_tensor_t*)
        );
        if (!network->state_history) {
            NIMCP_LOGGING_WARN("lnn_network_set_training: Failed to allocate history");
            network->history_capacity = 0;
        }
    }

    // Clear history when exiting training mode
    if (!training) {
        lnn_network_clear_history(network);
    }
}

/**
 * @brief Check if in training mode
 *
 * WHAT: Query current training mode
 * WHY:  Conditional behavior based on train/inference
 * HOW:  Return is_training flag
 */
bool lnn_network_is_training(const lnn_network_t* network) {
    // Guard: NULL-safe
    if (!network) {
        return false;
    }

    return network->is_training;
}

/*=============================================================================
 * History Management (for BPTT)
 *===========================================================================*/

/**
 * @brief Record current state in history
 *
 * WHAT: Save current network state to history buffer
 * WHY:  BPTT requires state at each time step
 * HOW:  Get state, store in circular buffer
 */
int lnn_network_record_state(lnn_network_t* network) {
    // Guard: Validate input
    if (!network) {
        NIMCP_LOGGING_ERROR("lnn_network_record_state: NULL network");
        return LNN_ERROR_NULL_POINTER;
    }

    // Check if history is allocated
    if (!network->state_history || network->history_capacity == 0) {
        return LNN_SUCCESS;  // No history buffer, skip silently
    }

    // Get current state — use unlocked helper since caller (forward_step)
    // already holds network->mutex.
    nimcp_tensor_t* state = NULL;
    int result = lnn_network_get_state_unlocked(network, &state);
    if (result != LNN_SUCCESS) {
        NIMCP_LOGGING_ERROR("lnn_network_record_state: Failed to get state");
        return result;
    }

    // Free old state if buffer is full
    if (network->state_history[network->history_write_idx]) {
        nimcp_tensor_destroy(network->state_history[network->history_write_idx]);
    }

    // Store state in circular buffer
    network->state_history[network->history_write_idx] = state;
    network->history_write_idx = (network->history_write_idx + 1) % network->history_capacity;

    // Update length
    if (network->history_len < network->history_capacity) {
        network->history_len++;
    }

    return LNN_SUCCESS;
}

/**
 * @brief Get state from history at given time step
 *
 * WHAT: Retrieve previously recorded state
 * WHY:  Access intermediate states for gradient computation
 * HOW:  Index into circular buffer
 */
int lnn_network_get_history(
    const lnn_network_t* network,
    uint32_t step,
    nimcp_tensor_t** state
) {
    // Guard: Validate inputs
    if (!network || !state) {
        NIMCP_LOGGING_ERROR("lnn_network_get_history: NULL argument");
        return LNN_ERROR_NULL_POINTER;
    }

    if (step >= network->history_len) {
        NIMCP_LOGGING_ERROR("lnn_network_get_history: Invalid step %u (max %u)",
                           step, network->history_len - 1);
        return LNN_ERROR_INVALID_DIMENSION;
    }

    if (!network->state_history) {
        NIMCP_LOGGING_ERROR("lnn_network_get_history: No history buffer");
        return LNN_ERROR_NOT_INITIALIZED;
    }

    // Calculate circular buffer index
    uint32_t idx = (network->history_write_idx + network->history_capacity - network->history_len + step)
                   % network->history_capacity;

    // Clone state tensor
    *state = nimcp_tensor_clone(network->state_history[idx]);
    if (!*state) {
        NIMCP_LOGGING_ERROR("lnn_network_get_history: Failed to clone state");
        return LNN_ERROR_OUT_OF_MEMORY;
    }

    return LNN_SUCCESS;
}

/**
 * @brief Clear state history
 *
 * WHAT: Reset history buffer to empty
 * WHY:  Between training epochs, free memory
 * HOW:  Free all history tensors, reset indices
 */
void lnn_network_clear_history(lnn_network_t* network) {
    // Guard: NULL-safe
    if (!network || !network->state_history) {
        return;
    }

    // Free all stored states
    for (uint32_t i = 0; i < network->history_capacity; i++) {
        if (network->state_history[i]) {
            nimcp_tensor_destroy(network->state_history[i]);
            network->state_history[i] = NULL;
        }
    }

    // Reset indices
    network->history_len = 0;
    network->history_write_idx = 0;
}

/*=============================================================================
 * Integration Connections
 *===========================================================================*/

/**
 * @brief Connect to NIMCP optimizer
 *
 * WHAT: Link network to optimizer for parameter updates
 * WHY:  Enable training with Adam, SGD, etc.
 * HOW:  Store optimizer handle
 */
int lnn_network_connect_optimizer(lnn_network_t* network, void* optimizer) {
    // Guard: Validate inputs
    if (!network) {
        NIMCP_LOGGING_ERROR("lnn_network_connect_optimizer: NULL network");
        return LNN_ERROR_NULL_POINTER;
    }

    network->optimizer = optimizer;

    NIMCP_LOGGING_INFO("Connected optimizer to network '%s'", network->name);
    return LNN_SUCCESS;
}

/**
 * @brief Connect to gradient manager
 *
 * WHAT: Link network to gradient manager
 * WHY:  Enable gradient accumulation, clipping
 * HOW:  Store gradient manager handle
 */
int lnn_network_connect_gradient_manager(lnn_network_t* network, void* grad_mgr) {
    // Guard: Validate inputs
    if (!network) {
        NIMCP_LOGGING_ERROR("lnn_network_connect_gradient_manager: NULL network");
        return LNN_ERROR_NULL_POINTER;
    }

    network->gradient_manager = grad_mgr;

    NIMCP_LOGGING_INFO("Connected gradient manager to network '%s'", network->name);
    return LNN_SUCCESS;
}

/**
 * @brief Connect to thread pool
 *
 * WHAT: Link network to thread pool for parallel execution
 * WHY:  Enable batch parallelism
 * HOW:  Store thread pool handle
 */
int lnn_network_connect_thread_pool(lnn_network_t* network, void* pool) {
    // Guard: Validate inputs
    if (!network) {
        NIMCP_LOGGING_ERROR("lnn_network_connect_thread_pool: NULL network");
        return LNN_ERROR_NULL_POINTER;
    }

    network->thread_pool = pool;

    NIMCP_LOGGING_INFO("Connected thread pool to network '%s'", network->name);
    return LNN_SUCCESS;
}

/*=============================================================================
 * Statistics
 *===========================================================================*/

/**
 * @brief Get network statistics
 *
 * WHAT: Compute current network stats
 * WHY:  Monitor training, detect issues
 * HOW:  Aggregate stats from all layers
 */
int lnn_network_get_stats(const lnn_network_t* network, lnn_network_stats_t* stats) {
    // Guard: Validate inputs
    if (!network || !stats) {
        NIMCP_LOGGING_ERROR("lnn_network_get_stats: NULL argument");
        return LNN_ERROR_NULL_POINTER;
    }

    // Copy current stats
    memcpy(stats, &network->stats, sizeof(lnn_network_stats_t));

    // Compute average tau across all layers
    float total_tau = 0.0f;
    uint32_t total_neurons = 0;

    for (uint32_t i = 0; i < network->n_layers; i++) {
        float avg_tau, min_tau, max_tau, state_norm;
        lnn_layer_get_stats(network->layers[i], &avg_tau, &min_tau, &max_tau, &state_norm);

        total_tau += avg_tau * network->layers[i]->n_neurons;
        total_neurons += network->layers[i]->n_neurons;
    }

    stats->avg_tau_network = (total_neurons > 0) ? (total_tau / total_neurons) : 0.0f;

    // Compute state norm
    nimcp_tensor_t* state = NULL;
    if (lnn_network_get_state(network, &state) == LNN_SUCCESS) {
        stats->state_norm = nimcp_tensor_norm_p(state, 2.0f);
        nimcp_tensor_destroy(state);
    }

    // Compute memory usage
    stats->memory_usage_bytes = lnn_network_memory_usage(network);

    return LNN_SUCCESS;
}

/**
 * @brief Get total number of trainable parameters
 *
 * WHAT: Count all learnable weights across all layers
 * WHY:  Model capacity reporting
 * HOW:  Sum lnn_layer_param_count across layers
 */
size_t lnn_network_param_count(const lnn_network_t* network) {
    // Guard: NULL-safe
    if (!network) {
        return 0;
    }

    size_t total = 0;
    for (uint32_t i = 0; i < network->n_layers; i++) {
        total += lnn_layer_param_count(network->layers[i]);
    }

    return total;
}

/**
 * @brief Get total memory usage
 *
 * WHAT: Compute total memory consumed by network
 * WHY:  Memory profiling
 * HOW:  Sum memory from layers, history, metadata
 */
size_t lnn_network_memory_usage(const lnn_network_t* network) {
    // Guard: NULL-safe
    if (!network) {
        return 0;
    }

    size_t total = sizeof(lnn_network_t);

    // Config
    total += sizeof(lnn_config_t);
    if (network->config && network->config->layer_configs) {
        total += network->n_layers * sizeof(lnn_layer_config_t);
    }

    // Layers array
    total += network->n_layers * sizeof(lnn_layer_t*);

    // Layer memory (simplified - would need layer-specific memory calculation)
    for (uint32_t i = 0; i < network->n_layers; i++) {
        total += sizeof(lnn_layer_t);
        // Add tensor memory (rough estimate)
        total += network->layers[i]->n_neurons * sizeof(float) * 10;  // State + weights
    }

    // History buffer
    total += network->history_capacity * sizeof(nimcp_tensor_t*);

    return total;
}

/*=============================================================================
 * Parameter Access
 *===========================================================================*/

/**
 * @brief Helper to copy tensor data to flat buffer
 */
static size_t copy_tensor_to_buffer(const nimcp_tensor_t* t, float* buf, size_t offset) {
    if (!t) return 0;
    size_t n = nimcp_tensor_numel(t);
    const float* data = (const float*)nimcp_tensor_data_const(t);
    if (data) {
        memcpy(buf + offset, data, n * sizeof(float));
    }
    return n;
}

/**
 * @brief Helper to copy flat buffer data to tensor
 */
static size_t copy_buffer_to_tensor(nimcp_tensor_t* t, const float* buf, size_t offset) {
    if (!t) return 0;
    size_t n = nimcp_tensor_numel(t);
    float* data = (float*)nimcp_tensor_data(t);
    if (data) {
        memcpy(data, buf + offset, n * sizeof(float));
    }
    return n;
}

/**
 * @brief Get all network parameters as flat array
 *
 * WHAT: Extract all trainable weights into contiguous buffer
 * WHY:  Interface with optimizers
 * HOW:  Iterate layers, copy W_in, W_rec, W_tau, b_in, b_tau, tau_base
 */
int lnn_network_get_params(
    const lnn_network_t* network,
    float* params,
    size_t* n_params
) {
    /* Guard clauses */
    if (!network) {
        NIMCP_LOGGING_ERROR("lnn_network_get_params: NULL network");
        return LNN_ERROR_NULL_POINTER;
    }
    if (!params || !n_params) {
        NIMCP_LOGGING_ERROR("lnn_network_get_params: NULL output pointer");
        return LNN_ERROR_NULL_POINTER;
    }

    size_t offset = 0;

    /* Copy parameters from each layer */
    for (uint32_t i = 0; i < network->n_layers; i++) {
        lnn_layer_t* layer = network->layers[i];
        if (!layer) continue;

        /* Order: W_in, W_rec, W_tau, b_in, b_tau, tau_base */
        offset += copy_tensor_to_buffer(layer->W_in, params, offset);
        offset += copy_tensor_to_buffer(layer->W_rec, params, offset);
        offset += copy_tensor_to_buffer(layer->W_tau, params, offset);
        offset += copy_tensor_to_buffer(layer->b_in, params, offset);
        offset += copy_tensor_to_buffer(layer->b_tau, params, offset);
        offset += copy_tensor_to_buffer(layer->tau_base, params, offset);
    }

    *n_params = offset;
    return LNN_SUCCESS;
}

/**
 * @brief Set all network parameters from flat array
 *
 * WHAT: Load trainable weights from contiguous buffer
 * WHY:  Apply optimizer updates
 * HOW:  Iterate layers, write to weight tensors
 */
int lnn_network_set_params(
    lnn_network_t* network,
    const float* params,
    size_t n_params
) {
    /* Guard clauses */
    if (!network) {
        NIMCP_LOGGING_ERROR("lnn_network_set_params: NULL network");
        return LNN_ERROR_NULL_POINTER;
    }
    if (!params) {
        NIMCP_LOGGING_ERROR("lnn_network_set_params: NULL params");
        return LNN_ERROR_NULL_POINTER;
    }

    size_t expected = lnn_network_param_count(network);
    if (n_params != expected) {
        NIMCP_LOGGING_ERROR("lnn_network_set_params: param count mismatch (%zu vs %zu)",
                           n_params, expected);
        return LNN_ERROR_INVALID_PARAM;
    }

    size_t offset = 0;

    /* Copy parameters to each layer */
    for (uint32_t i = 0; i < network->n_layers; i++) {
        lnn_layer_t* layer = network->layers[i];
        if (!layer) continue;

        /* Order: W_in, W_rec, W_tau, b_in, b_tau, tau_base */
        offset += copy_buffer_to_tensor(layer->W_in, params, offset);
        offset += copy_buffer_to_tensor(layer->W_rec, params, offset);
        offset += copy_buffer_to_tensor(layer->W_tau, params, offset);
        offset += copy_buffer_to_tensor(layer->b_in, params, offset);
        offset += copy_buffer_to_tensor(layer->b_tau, params, offset);
        offset += copy_buffer_to_tensor(layer->tau_base, params, offset);
    }

    return LNN_SUCCESS;
}

/**
 * @brief Get all gradient values as flat array
 *
 * WHAT: Extract accumulated gradients from all layers
 * WHY:  Interface with optimizers
 * HOW:  Iterate layers, copy grad_W_in, grad_W_rec, etc.
 */
int lnn_network_get_gradients(
    const lnn_network_t* network,
    float* gradients,
    size_t* n_grads
) {
    /* Guard clauses */
    if (!network) {
        NIMCP_LOGGING_ERROR("lnn_network_get_gradients: NULL network");
        return LNN_ERROR_NULL_POINTER;
    }
    if (!gradients || !n_grads) {
        NIMCP_LOGGING_ERROR("lnn_network_get_gradients: NULL output pointer");
        return LNN_ERROR_NULL_POINTER;
    }

    size_t offset = 0;

    /* Copy gradients from each layer */
    for (uint32_t i = 0; i < network->n_layers; i++) {
        lnn_layer_t* layer = network->layers[i];
        if (!layer) continue;

        /* Order: same as params */
        offset += copy_tensor_to_buffer(layer->grad_W_in, gradients, offset);
        offset += copy_tensor_to_buffer(layer->grad_W_rec, gradients, offset);
        offset += copy_tensor_to_buffer(layer->grad_W_tau, gradients, offset);
        offset += copy_tensor_to_buffer(layer->grad_b_in, gradients, offset);
        offset += copy_tensor_to_buffer(layer->grad_b_tau, gradients, offset);
        offset += copy_tensor_to_buffer(layer->grad_tau_base, gradients, offset);
    }

    *n_grads = offset;
    return LNN_SUCCESS;
}

/**
 * @brief Zero all gradient tensors in network
 *
 * WHAT: Reset gradients to zero
 * WHY:  Prevent gradient accumulation
 * HOW:  Call nimcp_tensor_zero on all grad_* tensors
 */
int lnn_network_zero_gradients(lnn_network_t* network) {
    /* Guard clause */
    if (!network) {
        NIMCP_LOGGING_ERROR("lnn_network_zero_gradients: NULL network");
        return LNN_ERROR_NULL_POINTER;
    }

    for (uint32_t i = 0; i < network->n_layers; i++) {
        lnn_layer_t* layer = network->layers[i];
        if (!layer) continue;

        if (layer->grad_W_in) nimcp_tensor_zero_grad(layer->grad_W_in);
        if (layer->grad_W_rec) nimcp_tensor_zero_grad(layer->grad_W_rec);
        if (layer->grad_W_tau) nimcp_tensor_zero_grad(layer->grad_W_tau);
        if (layer->grad_b_in) nimcp_tensor_zero_grad(layer->grad_b_in);
        if (layer->grad_b_tau) nimcp_tensor_zero_grad(layer->grad_b_tau);
        if (layer->grad_tau_base) nimcp_tensor_zero_grad(layer->grad_tau_base);
    }

    return LNN_SUCCESS;
}

/*=============================================================================
 * Serialization
 *===========================================================================*/

/**
 * @brief Save network to file
 *
 * WHAT: Serialize network to disk
 * WHY:  Model persistence
 * HOW:  Write magic number, config, layer weights
 *
 * NOTE: Placeholder implementation - full serialization requires binary format
 */
int lnn_network_save(const lnn_network_t* network, const char* path) {
    if (!network || !path) {
        NIMCP_LOGGING_ERROR("lnn_network_save: NULL argument");
        return LNN_ERROR_NULL_POINTER;
    }

    FILE* f = fopen(path, "wb");
    if (!f) {
        NIMCP_LOGGING_ERROR("lnn_network_save: failed to open %s", path);
        return -1;
    }

    /* Header */
    uint32_t magic = 0x4C4E4E53; /* "LNNS" */
    uint32_t version = 1;
    fwrite(&magic, sizeof(uint32_t), 1, f);
    fwrite(&version, sizeof(uint32_t), 1, f);

    /* Network metadata */
    fwrite(&network->n_layers, sizeof(uint32_t), 1, f);
    fwrite(&network->n_inputs, sizeof(uint32_t), 1, f);
    fwrite(&network->n_outputs, sizeof(uint32_t), 1, f);
    fwrite(&network->is_training, sizeof(bool), 1, f);

    /* Per-layer weights and state */
    for (uint32_t i = 0; i < network->n_layers; i++) {
        lnn_layer_t* layer = network->layers[i];
        if (!layer) {
            uint32_t zero = 0;
            fwrite(&zero, sizeof(uint32_t), 1, f);
            continue;
        }
        fwrite(&layer->n_neurons, sizeof(uint32_t), 1, f);
        fwrite(&layer->ode_method, sizeof(uint32_t), 1, f);
        fwrite(&layer->dt, sizeof(float), 1, f);
        fwrite(&layer->use_layer_norm, sizeof(bool), 1, f);
        fwrite(&layer->layer_norm_eps, sizeof(float), 1, f);

        /* Save weight tensors (NULL-safe via nimcp_tensor_save) */
        nimcp_tensor_save(layer->W_in, f);
        nimcp_tensor_save(layer->W_rec, f);
        nimcp_tensor_save(layer->W_tau, f);
        nimcp_tensor_save(layer->b_in, f);
        nimcp_tensor_save(layer->b_tau, f);
        nimcp_tensor_save(layer->tau_base, f);

        /* Save state tensors */
        nimcp_tensor_save(layer->x, f);
        nimcp_tensor_save(layer->tau, f);
    }

    fclose(f);
    NIMCP_LOGGING_INFO("LNN network saved to %s (%u layers)", path, network->n_layers);
    return 0;
}

/**
 * @brief Load network from file
 *
 * WHAT: Deserialize network from disk
 * WHY:  Restore saved model
 * HOW:  Read file, reconstruct config, create network, load weights
 *
 * NOTE: Placeholder implementation
 */
lnn_network_t* lnn_network_load(const char* path) {
    if (!path) {
        NIMCP_LOGGING_ERROR("lnn_network_load: NULL path");
        return NULL;
    }

    FILE* f = fopen(path, "rb");
    if (!f) {
        NIMCP_LOGGING_WARN("lnn_network_load: file not found %s", path);
        return NULL;
    }

    /* Header */
    uint32_t magic = 0, version = 0;
    if (fread(&magic, sizeof(uint32_t), 1, f) != 1 || magic != 0x4C4E4E53) {
        NIMCP_LOGGING_ERROR("lnn_network_load: invalid magic");
        fclose(f);
        return NULL;
    }
    fread(&version, sizeof(uint32_t), 1, f);

    /* Network metadata */
    uint32_t n_layers = 0, n_inputs = 0, n_outputs = 0;
    bool is_training = false;
    fread(&n_layers, sizeof(uint32_t), 1, f);
    fread(&n_inputs, sizeof(uint32_t), 1, f);
    fread(&n_outputs, sizeof(uint32_t), 1, f);
    fread(&is_training, sizeof(bool), 1, f);

    /* Recreate network using NCP topology (matches brain_enable_multi_network_training) */
    uint32_t n_inter = (n_inputs + n_outputs) / 2;
    if (n_inter < 8) n_inter = 8;
    uint32_t n_command = n_outputs * 2;
    if (n_command < 8) n_command = 8;

    lnn_network_t* net = lnn_network_create_ncp(n_inputs, n_inter, n_command, n_outputs);
    if (!net) {
        NIMCP_LOGGING_ERROR("lnn_network_load: failed to create NCP network");
        fclose(f);
        return NULL;
    }

    /* Restore per-layer weights from file */
    uint32_t restore_layers = (n_layers < net->n_layers) ? n_layers : net->n_layers;
    for (uint32_t i = 0; i < n_layers; i++) {
        uint32_t n_neurons = 0;
        uint32_t ode_method = 0;
        float dt = 0.0f;
        bool use_layer_norm = false;
        float layer_norm_eps = 0.0f;

        fread(&n_neurons, sizeof(uint32_t), 1, f);
        if (n_neurons == 0) continue;

        fread(&ode_method, sizeof(uint32_t), 1, f);
        fread(&dt, sizeof(float), 1, f);
        fread(&use_layer_norm, sizeof(bool), 1, f);
        fread(&layer_norm_eps, sizeof(float), 1, f);

        /* Load weight tensors */
        nimcp_tensor_t* W_in = nimcp_tensor_load(f);
        nimcp_tensor_t* W_rec = nimcp_tensor_load(f);
        nimcp_tensor_t* W_tau = nimcp_tensor_load(f);
        nimcp_tensor_t* b_in = nimcp_tensor_load(f);
        nimcp_tensor_t* b_tau = nimcp_tensor_load(f);
        nimcp_tensor_t* tau_base = nimcp_tensor_load(f);
        nimcp_tensor_t* x = nimcp_tensor_load(f);
        nimcp_tensor_t* tau = nimcp_tensor_load(f);

        /* Apply to matching layer in recreated network */
        if (i < restore_layers && net->layers[i]) {
            lnn_layer_t* layer = net->layers[i];

            /* Replace weight tensors if dimensions match */
            #define SWAP_TENSOR(dst, src) do { \
                if ((src) && (dst) && nimcp_tensor_numel(src) == nimcp_tensor_numel(dst)) { \
                    nimcp_tensor_destroy(dst); \
                    (dst) = (src); \
                    (src) = NULL; \
                } \
            } while(0)

            SWAP_TENSOR(layer->W_in, W_in);
            SWAP_TENSOR(layer->W_rec, W_rec);
            SWAP_TENSOR(layer->W_tau, W_tau);
            SWAP_TENSOR(layer->b_in, b_in);
            SWAP_TENSOR(layer->b_tau, b_tau);
            SWAP_TENSOR(layer->tau_base, tau_base);
            SWAP_TENSOR(layer->x, x);
            SWAP_TENSOR(layer->tau, tau);
            #undef SWAP_TENSOR

            layer->ode_method = (lnn_ode_method_t)ode_method;
            layer->dt = dt;
            layer->use_layer_norm = use_layer_norm;
            layer->layer_norm_eps = layer_norm_eps;
        }

        /* Clean up any tensors not consumed */
        nimcp_tensor_destroy(W_in);
        nimcp_tensor_destroy(W_rec);
        nimcp_tensor_destroy(W_tau);
        nimcp_tensor_destroy(b_in);
        nimcp_tensor_destroy(b_tau);
        nimcp_tensor_destroy(tau_base);
        nimcp_tensor_destroy(x);
        nimcp_tensor_destroy(tau);
    }

    net->is_training = is_training;
    fclose(f);
    NIMCP_LOGGING_INFO("LNN network loaded from %s (%u layers restored)", path, restore_layers);
    return net;
}
