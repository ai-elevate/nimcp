/**
 * @file nimcp_lnn_network.h
 * @brief LNN Network API - Multi-layer Liquid Neural Network
 * @version 1.0.0
 * @date 2025-12-20
 *
 * WHAT: Complete Liquid Neural Network with multiple layers and training support
 * WHY:  Provides continuous-time temporal processing for NIMCP modules
 * HOW:  Stacks multiple LNN layers, manages state history for BPTT, integrates with training pipeline
 *
 * ARCHITECTURE:
 * ```
 * ┌─────────────────────────────────────────────────────────────────────────┐
 * │                        LNN NETWORK STRUCTURE                             │
 * ├─────────────────────────────────────────────────────────────────────────┤
 * │                                                                          │
 * │   INPUT [n_inputs]                                                       │
 * │      │                                                                   │
 * │      ▼                                                                   │
 * │   ┌────────────────────────────────────────────────────────────────┐    │
 * │   │                    LAYER 0                                      │    │
 * │   │  [n_neurons_0] LTC neurons with recurrent connections          │    │
 * │   │  Wiring: Full/Random/Small-World/Scale-Free/NCP                │    │
 * │   └──────────────────────────┬─────────────────────────────────────┘    │
 * │                              │                                           │
 * │                              ▼                                           │
 * │   ┌────────────────────────────────────────────────────────────────┐    │
 * │   │                    LAYER 1                                      │    │
 * │   │  [n_neurons_1] LTC neurons                                     │    │
 * │   └──────────────────────────┬─────────────────────────────────────┘    │
 * │                              │                                           │
 * │                              ⋮                                           │
 * │                              │                                           │
 * │                              ▼                                           │
 * │   ┌────────────────────────────────────────────────────────────────┐    │
 * │   │                    LAYER N-1                                    │    │
 * │   │  [n_outputs] neurons                                           │    │
 * │   └──────────────────────────┬─────────────────────────────────────┘    │
 * │                              │                                           │
 * │                              ▼                                           │
 * │   OUTPUT [n_outputs]                                                     │
 * │                                                                          │
 * │   STATE HISTORY (if training):                                           │
 * │   [t=0] → [t=1] → [t=2] → ... → [t=T]                                  │
 * │   For backpropagation through time (BPTT)                               │
 * │                                                                          │
 * └─────────────────────────────────────────────────────────────────────────┘
 * ```
 *
 * FORWARD PASS:
 * =============
 * Sequential layer-by-layer computation:
 *   1. Layer 0: output_0 = layer_forward(layer_0, input, dt)
 *   2. Layer 1: output_1 = layer_forward(layer_1, output_0, dt)
 *   3. ...
 *   4. Layer N-1: output_N-1 = layer_forward(layer_N-1, output_N-2, dt)
 *   5. network_output = output_N-1
 *
 * If training mode enabled, save state at each step for BPTT.
 *
 * TRAINING INTEGRATION:
 * =====================
 * LNN networks integrate with NIMCP training infrastructure:
 *   - Optimizer: Adam, SGD, RMSprop for parameter updates
 *   - Gradient Manager: Gradient accumulation, clipping, scaling
 *   - Loss Functions: MSE, CrossEntropy for supervised learning
 *   - Cognitive Training Bridge: Meta-learning, curriculum learning
 *   - Training Logic Bridge: Validation, early stopping
 *   - Training Immune: Detects/responds to training instabilities
 *
 * BIOLOGICAL GROUNDING:
 * =====================
 * - Continuous-time dynamics model real neural membrane potentials
 * - Learnable time constants capture diverse neural timescales (1ms - 1000ms)
 * - Sparse recurrent wiring reflects biological connectivity constraints
 * - Multi-layer hierarchy models cortical depth (L1 → L6)
 *
 * NIMCP STANDARDS:
 * ================
 * - All functions < 50 lines (guard clauses)
 * - WHAT-WHY-HOW documentation on all functions
 * - Thread-safe via nimcp_mutex_t
 * - Memory management via nimcp_malloc/nimcp_free
 * - Error codes: LNN_SUCCESS, LNN_ERROR_*
 *
 * @author NIMCP Development Team
 */

#ifndef NIMCP_LNN_NETWORK_H
#define NIMCP_LNN_NETWORK_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "lnn/nimcp_lnn_types.h"
#include "lnn/nimcp_lnn_config.h"
#include "lnn/nimcp_lnn_layer.h"
#include "utils/tensor/nimcp_tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Types lnn_network_t, lnn_network_stats_t, and struct lnn_network_s
 * are defined in nimcp_lnn_types.h */

/*=============================================================================
 * Network Lifecycle
 *===========================================================================*/

/**
 * @brief Create LNN network from configuration
 *
 * WHAT: Allocate and initialize multi-layer LNN network
 * WHY:  Primary network creation method
 * HOW:  1. Validate config  2. Create layers  3. Initialize weights  4. Allocate history
 *
 * @param config Network configuration (must be valid)
 * @return Network handle or NULL on failure
 */
lnn_network_t* lnn_network_create(const lnn_config_t* config);

/**
 * @brief Create NCP (Neural Circuit Policy) network
 *
 * WHAT: Convenience constructor for NCP architecture
 * WHY:  NCP is standard LNN architecture from original paper
 * HOW:  Create 4-layer network: sensory → inter → command → motor
 *
 * ARCHITECTURE:
 *   Input → Sensory Layer → Inter Layer → Command Layer → Motor Layer → Output
 *           (n_inputs)       (n_inter)     (n_command)     (n_outputs)
 *
 * @param n_inputs Number of input features (sensory layer size)
 * @param n_inter Number of interneurons (hidden layer size)
 * @param n_command Number of command neurons (decision layer size)
 * @param n_outputs Number of output features (motor layer size)
 * @return Network handle or NULL on failure
 */
lnn_network_t* lnn_network_create_ncp(
    uint32_t n_inputs,
    uint32_t n_inter,
    uint32_t n_command,
    uint32_t n_outputs
);

/**
 * @brief Destroy network and free all resources
 *
 * WHAT: Clean shutdown of network
 * WHY:  Prevent memory leaks
 * HOW:  Free layers, history, config, gradient context, mutex
 *
 * @param network Network to destroy (NULL-safe)
 */
void lnn_network_destroy(lnn_network_t* network);

/**
 * @brief Initialize network weights
 *
 * WHAT: Initialize all layer weights with random values
 * WHY:  Proper initialization critical for training convergence
 * HOW:  Call lnn_layer_init_weights on each layer with seed
 *
 * @param network LNN network
 * @param seed Random seed (0 = use time)
 * @return 0 on success, negative on error
 */
int lnn_network_init_weights(lnn_network_t* network, uint64_t seed);

/*=============================================================================
 * Forward Pass
 *===========================================================================*/

/**
 * @brief Single forward step with given input
 *
 * WHAT: Propagate input through all layers for one time step
 * WHY:  Core inference operation
 * HOW:  Sequential layer forward: layer_i(output_{i-1})
 *
 * If training mode enabled, records state in history.
 *
 * @param network LNN network
 * @param input Input tensor [n_inputs]
 * @param output Output tensor [n_outputs] (pre-allocated)
 * @param dt Time step in ms (0 = use network default)
 * @return 0 on success, negative on error
 */
int lnn_network_forward_step(
    lnn_network_t* network,
    const nimcp_tensor_t* input,
    nimcp_tensor_t* output,
    float dt
);

/**
 * @brief Forward pass over sequence
 *
 * WHAT: Process sequence of inputs, producing sequence of outputs
 * WHY:  Temporal sequence processing (time series, speech, etc.)
 * HOW:  Loop over sequence, calling forward_step for each time step
 *
 * @param network LNN network
 * @param inputs Input sequence tensor [seq_len, n_inputs]
 * @param outputs Output sequence tensor [seq_len, n_outputs] (pre-allocated)
 * @param seq_len Sequence length
 * @param dt Time step per sample in ms
 * @return 0 on success, negative on error
 */
int lnn_network_forward_sequence(
    lnn_network_t* network,
    const nimcp_tensor_t* inputs,
    nimcp_tensor_t* outputs,
    uint32_t seq_len,
    float dt
);

/**
 * @brief Parallel forward over batch of sequences
 *
 * WHAT: Process batch of sequences in parallel using thread pool
 * WHY:  Efficient batch processing for training
 * HOW:  Distribute batch across threads, each processes one sequence
 *
 * Requires network->thread_pool to be connected.
 *
 * @param network LNN network
 * @param inputs Batch input tensor [batch_size, seq_len, n_inputs]
 * @param outputs Batch output tensor [batch_size, seq_len, n_outputs] (pre-allocated)
 * @param batch_size Number of sequences in batch
 * @param seq_len Sequence length
 * @param dt Time step per sample in ms
 * @return 0 on success, negative on error
 */
int lnn_network_forward_batch(
    lnn_network_t* network,
    const nimcp_tensor_t* inputs,
    nimcp_tensor_t* outputs,
    uint32_t batch_size,
    uint32_t seq_len,
    float dt
);

/*=============================================================================
 * State Management
 *===========================================================================*/

/**
 * @brief Get current network state (all layers)
 *
 * WHAT: Concatenate state from all layers into single tensor
 * WHY:  State checkpointing, introspection
 * HOW:  Gather x from each layer, concatenate into output
 *
 * @param network LNN network
 * @param state Output state tensor (allocated by function)
 * @return 0 on success, negative on error
 */
int lnn_network_get_state(const lnn_network_t* network, nimcp_tensor_t** state);

/**
 * @brief Set network state (all layers)
 *
 * WHAT: Restore network state from tensor
 * WHY:  State restoration from checkpoint
 * HOW:  Split state tensor, set each layer's state
 *
 * @param network LNN network
 * @param state State tensor (must match total network state size)
 * @return 0 on success, negative on error
 */
int lnn_network_set_state(lnn_network_t* network, const nimcp_tensor_t* state);

/**
 * @brief Get current time constants (all layers)
 *
 * WHAT: Concatenate τ from all layers
 * WHY:  Introspection of learned dynamics
 * HOW:  Gather tau from each layer, concatenate
 *
 * @param network LNN network
 * @param tau Output tau tensor (allocated by function)
 * @return 0 on success, negative on error
 */
int lnn_network_get_tau(const lnn_network_t* network, nimcp_tensor_t** tau);

/**
 * @brief Reset network state to zero
 *
 * WHAT: Zero out all layer states
 * WHY:  Between sequences in training/inference
 * HOW:  Call lnn_layer_reset on each layer
 *
 * @param network LNN network
 */
void lnn_network_reset_state(lnn_network_t* network);

/**
 * @brief Reset accumulated gradients to zero
 *
 * WHAT: Zero all gradient tensors in all layers
 * WHY:  Between training batches
 * HOW:  Call lnn_layer_reset_gradients on each layer
 *
 * @param network LNN network
 */
void lnn_network_reset_gradients(lnn_network_t* network);

/*=============================================================================
 * Training Mode
 *===========================================================================*/

/**
 * @brief Set training mode
 *
 * WHAT: Enable/disable training mode
 * WHY:  Training mode records state history for BPTT
 * HOW:  Set is_training flag, allocate/clear history if needed
 *
 * @param network LNN network
 * @param training True to enable training mode, false for inference
 */
void lnn_network_set_training(lnn_network_t* network, bool training);

/**
 * @brief Check if in training mode
 *
 * WHAT: Query current training mode
 * WHY:  Conditional behavior based on train/inference
 * HOW:  Return is_training flag
 *
 * @param network LNN network
 * @return True if in training mode, false otherwise
 */
bool lnn_network_is_training(const lnn_network_t* network);

/*=============================================================================
 * History Management (for BPTT)
 *===========================================================================*/

/**
 * @brief Record current state in history
 *
 * WHAT: Save current network state to history buffer
 * WHY:  BPTT requires state at each time step
 * HOW:  Get state, store in circular buffer, increment index
 *
 * Called automatically during forward pass if training mode enabled.
 *
 * @param network LNN network
 * @return 0 on success, negative on error
 */
int lnn_network_record_state(lnn_network_t* network);

/**
 * @brief Get state from history at given time step
 *
 * WHAT: Retrieve previously recorded state
 * WHY:  Access intermediate states for gradient computation
 * HOW:  Index into circular buffer
 *
 * @param network LNN network
 * @param step Time step index (0 = oldest, history_len-1 = newest)
 * @param state Output state tensor (allocated by function)
 * @return 0 on success, negative on error
 */
int lnn_network_get_history(
    const lnn_network_t* network,
    uint32_t step,
    nimcp_tensor_t** state
);

/**
 * @brief Clear state history
 *
 * WHAT: Reset history buffer to empty
 * WHY:  Between training epochs, free memory
 * HOW:  Free all history tensors, reset indices
 *
 * @param network LNN network
 */
void lnn_network_clear_history(lnn_network_t* network);

/*=============================================================================
 * Integration Connections
 *===========================================================================*/

/**
 * @brief Connect to NIMCP optimizer
 *
 * WHAT: Link network to optimizer for parameter updates
 * WHY:  Enable training with Adam, SGD, RMSprop, etc.
 * HOW:  Store optimizer handle, register parameters
 *
 * @param network LNN network
 * @param optimizer Optimizer context (nimcp_optimizer_context_t*)
 * @return 0 on success, negative on error
 */
int lnn_network_connect_optimizer(lnn_network_t* network, void* optimizer);

/**
 * @brief Connect to gradient manager
 *
 * WHAT: Link network to gradient manager
 * WHY:  Enable gradient accumulation, clipping, scaling
 * HOW:  Store gradient manager handle
 *
 * @param network LNN network
 * @param grad_mgr Gradient manager context (nimcp_gradient_manager_ctx_t*)
 * @return 0 on success, negative on error
 */
int lnn_network_connect_gradient_manager(lnn_network_t* network, void* grad_mgr);

/**
 * @brief Connect to thread pool
 *
 * WHAT: Link network to thread pool for parallel execution
 * WHY:  Enable batch parallelism, layer pipelining
 * HOW:  Store thread pool handle
 *
 * @param network LNN network
 * @param pool Thread pool (nimcp_thread_pool_t*)
 * @return 0 on success, negative on error
 */
int lnn_network_connect_thread_pool(lnn_network_t* network, void* pool);

/*=============================================================================
 * Statistics
 *===========================================================================*/

/**
 * @brief Get network statistics
 *
 * WHAT: Compute current network stats (performance, health)
 * WHY:  Monitor training, detect issues
 * HOW:  Aggregate stats from all layers, check health
 *
 * @param network LNN network
 * @param stats Output statistics structure
 * @return 0 on success, negative on error
 */
int lnn_network_get_stats(const lnn_network_t* network, lnn_network_stats_t* stats);

/**
 * @brief Get total number of trainable parameters
 *
 * WHAT: Count all learnable weights across all layers
 * WHY:  Model capacity reporting
 * HOW:  Sum lnn_layer_param_count across layers
 *
 * @param network LNN network
 * @return Total parameter count
 */
size_t lnn_network_param_count(const lnn_network_t* network);

/**
 * @brief Get total memory usage
 *
 * WHAT: Compute total memory consumed by network
 * WHY:  Memory profiling, resource management
 * HOW:  Sum memory from layers, history, metadata
 *
 * @param network LNN network
 * @return Memory usage in bytes
 */
size_t lnn_network_memory_usage(const lnn_network_t* network);

/*=============================================================================
 * Serialization
 *===========================================================================*/

/**
 * @brief Save network to file
 *
 * WHAT: Serialize network (architecture + weights) to disk
 * WHY:  Model persistence, checkpointing
 * HOW:  Write config, layer weights, metadata to binary file
 *
 * FILE FORMAT:
 * - Magic number: "LNN\0" (4 bytes)
 * - Version: uint32_t
 * - Config: lnn_config_t serialized
 * - Layers: Each layer's weights in order
 *
 * @param network LNN network
 * @param path File path to save to
 * @return 0 on success, negative on error
 */
int lnn_network_save(const lnn_network_t* network, const char* path);

/**
 * @brief Load network from file
 *
 * WHAT: Deserialize network from disk
 * WHY:  Restore saved model
 * HOW:  Read file, reconstruct config, create network, load weights
 *
 * @param path File path to load from
 * @return Network handle or NULL on failure
 */
lnn_network_t* lnn_network_load(const char* path);

#ifdef __cplusplus
}
#endif

#endif /* NIMCP_LNN_NETWORK_H */
