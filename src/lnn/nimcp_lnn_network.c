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
#include "utils/memory/nimcp_memory.h"
#include "utils/platform/nimcp_platform_mutex.h"
#include "utils/logging/nimcp_logging.h"
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

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
        return NULL;
    }

    if (lnn_config_validate(config) != LNN_SUCCESS) {
        NIMCP_LOGGING_ERROR("lnn_network_create: Invalid config");
        return NULL;
    }

    // Allocate network structure
    lnn_network_t* network = (lnn_network_t*)nimcp_calloc(1, sizeof(lnn_network_t));
    if (!network) {
        NIMCP_LOGGING_ERROR("lnn_network_create: Failed to allocate network");
        return NULL;
    }

    // Initialize basic fields
    static uint32_t next_id = 0;
    network->id = next_id++;
    snprintf(network->name, sizeof(network->name), "lnn_network_%u", network->id);
    network->n_layers = config->n_layers;
    network->n_inputs = config->n_inputs;
    network->n_outputs = config->n_outputs;
    network->train_mode = config->train_mode;
    network->is_training = false;

    // Copy configuration
    network->config = (lnn_config_t*)nimcp_calloc(1, sizeof(lnn_config_t));
    if (!network->config) {
        NIMCP_LOGGING_ERROR("lnn_network_create: Failed to allocate config");
        nimcp_free(network);
        return NULL;
    }
    memcpy(network->config, config, sizeof(lnn_config_t));

    // Allocate layer array
    network->layers = (lnn_layer_t**)nimcp_calloc(network->n_layers, sizeof(lnn_layer_t*));
    if (!network->layers) {
        NIMCP_LOGGING_ERROR("lnn_network_create: Failed to allocate layer array");
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
            nimcp_free(network->config);
            nimcp_free(network);
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
        return NULL;
    }

    // Create NCP configuration
    lnn_config_t config;
    if (lnn_config_ncp(&config, n_inputs, n_inter, n_command, n_outputs) != LNN_SUCCESS) {
        NIMCP_LOGGING_ERROR("lnn_network_create_ncp: Failed to create NCP config");
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
        }
        nimcp_free(network->layers);
    }

    // Clear and free state history
    lnn_network_clear_history(network);
    if (network->state_history) {
        nimcp_free(network->state_history);
    }

    // Destroy config
    if (network->config) {
        lnn_config_destroy(network->config);
        nimcp_free(network->config);
    }

    // Destroy mutex
    if (network->mutex) {
        nimcp_platform_mutex_destroy((nimcp_platform_mutex_t*)network->mutex);
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
int lnn_network_get_state(const lnn_network_t* network, nimcp_tensor_t** state) {
    // Guard: Validate inputs
    if (!network || !state) {
        NIMCP_LOGGING_ERROR("lnn_network_get_state: NULL argument");
        return LNN_ERROR_NULL_POINTER;
    }

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

    // Split state and set each layer
    uint32_t offset = 0;
    for (uint32_t i = 0; i < network->n_layers; i++) {
        uint32_t layer_size = network->layers[i]->n_neurons;

        // Extract layer state slice
        uint32_t dims[1] = {layer_size};
        nimcp_tensor_t* layer_state = nimcp_tensor_create(dims, 1, NIMCP_DTYPE_F32);
        if (!layer_state) {
            NIMCP_LOGGING_ERROR("lnn_network_set_state: Failed to allocate layer state");
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
            return result;
        }

        offset += layer_size;
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

    // Get current state
    nimcp_tensor_t* state = NULL;
    int result = lnn_network_get_state(network, &state);
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
    // Guard: Validate inputs
    if (!network || !path) {
        NIMCP_LOGGING_ERROR("lnn_network_save: NULL argument");
        return LNN_ERROR_NULL_POINTER;
    }

    NIMCP_LOGGING_WARN("lnn_network_save: Not yet implemented");
    return LNN_ERROR_NOT_INITIALIZED;
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
    // Guard: Validate input
    if (!path) {
        NIMCP_LOGGING_ERROR("lnn_network_load: NULL path");
        return NULL;
    }

    NIMCP_LOGGING_WARN("lnn_network_load: Not yet implemented");
    return NULL;
}
