//=============================================================================
// nimcp_training_bridge.h - GPU Weight Cache & Forward Pass Bridge
//=============================================================================
/**
 * @file nimcp_training_bridge.h
 * @brief Bridge between neuron/synapse AoS layout and GPU tensor operations
 *
 * WHAT: Extracts contiguous weight matrices from neuron_t/synapse_t structs,
 *       runs forward pass + loss computation on GPU, writes results back.
 * WHY:  Neural network uses Array-of-Structures (each neuron has incoming_synapses[]),
 *       but GPU GEMM needs contiguous weight matrices. This bridge layer handles
 *       the AoS <-> matrix conversion.
 * HOW:  Upload: iterate neurons per layer, extract weight*strength into row-major matrix.
 *       Download: write GPU matrix values back to synapse structs.
 *
 * ARCHITECTURE:
 *   CPU                           GPU
 *   -----                         -----
 *   input data ---upload--->  [input tensor]
 *                                  |
 *                             GEMV per layer: W[l] @ a[l] + b[l]
 *                             Activation (sigmoid/tanh/relu)
 *                             MSE loss
 *                                  |
 *   output, loss <--download--     |
 *        |
 *   reward = 1/(1+loss)
 *   Sync activations -> neuron states
 *   STDP/BCM/eligibility (CPU)
 *   Mark weights_dirty = true
 *
 * @version 1.0
 * @date 2026
 */

#ifndef NIMCP_TRAINING_BRIDGE_H
#define NIMCP_TRAINING_BRIDGE_H

#include "gpu/context/nimcp_gpu_context.h"
#include "gpu/tensor/nimcp_tensor_gpu.h"
#include "core/neuralnet/nimcp_neuralnet.h"

#ifdef __cplusplus
extern "C" {
#endif

#include "common/nimcp_export.h"
#include <stdbool.h>
#include <stdint.h>

//=============================================================================
// GPU Weight Cache Structure
//=============================================================================

/**
 * @brief Cache of GPU tensors mirroring neural network weight matrices
 *
 * Maintains GPU-resident copies of weight matrices extracted from the
 * neuron/synapse AoS layout. Tracks dirty state so weights are only
 * re-uploaded when CPU biological learning modifies them.
 */
typedef struct nimcp_gpu_weight_cache_s {
    nimcp_gpu_context_t* ctx;              /**< GPU context for operations */

    uint32_t num_layers;                   /**< Number of layers (from config) */
    uint32_t* layer_sizes;                 /**< Size of each layer (owned copy) */
    uint32_t* layer_offsets;               /**< Neuron index offset per layer */

    // Per-layer GPU tensors (num_layers - 1 transitions)
    nimcp_gpu_tensor_t** weights;          /**< W[l] = (layer_sizes[l+1] x layer_sizes[l]) */
    nimcp_gpu_tensor_t** biases;           /**< b[l] = (layer_sizes[l+1]) */
    nimcp_gpu_tensor_t** activations;      /**< a[l] = (layer_sizes[l]) per-layer output */

    // Per-layer activation types
    activation_type_t* layer_activations;  /**< Activation function per layer */

    // Sync state
    bool weights_dirty_on_cpu;             /**< CPU modified weights, need re-upload */
    uint64_t last_sync_step;               /**< Last step weights were synced */

    // Host-side scratch buffers (reused across uploads)
    float* host_weight_buf;                /**< Largest weight matrix buffer */
    float* host_bias_buf;                  /**< Largest bias vector buffer */
    float* host_activation_buf;            /**< Largest activation vector buffer */
    size_t host_weight_buf_size;           /**< Size of weight buffer in floats */
    size_t host_bias_buf_size;             /**< Size of bias buffer in floats */
    size_t host_activation_buf_size;       /**< Size of activation buffer in floats */
} nimcp_gpu_weight_cache_t;

//=============================================================================
// Lifecycle
//=============================================================================

/**
 * @brief Create GPU weight cache from a neural network
 *
 * Allocates GPU tensors for all layer weight matrices, bias vectors,
 * and activation vectors. Does NOT upload weights (call upload separately).
 *
 * @param ctx GPU context (must be valid)
 * @param net Neural network to mirror
 * @param layer_sizes Array of layer sizes (num_layers elements)
 * @param num_layers Number of layers
 * @return Weight cache or NULL on failure
 */
NIMCP_EXPORT nimcp_gpu_weight_cache_t* nimcp_gpu_weight_cache_create(
    nimcp_gpu_context_t* ctx,
    neural_network_t net,
    const uint32_t* layer_sizes,
    uint32_t num_layers
);

/**
 * @brief Destroy GPU weight cache and free all GPU tensors
 *
 * @param cache Weight cache to destroy (NULL-safe)
 */
NIMCP_EXPORT void nimcp_gpu_weight_cache_destroy(nimcp_gpu_weight_cache_t* cache);

//=============================================================================
// Weight Synchronization
//=============================================================================

/**
 * @brief Upload weights from CPU neuron/synapse structs to GPU tensors
 *
 * Extracts effective weight (weight * strength) from each synapse's
 * incoming_synapses array and builds row-major matrices for GPU GEMV.
 * Also uploads neuron biases.
 *
 * @param cache Weight cache
 * @param net Neural network source
 * @return true on success
 */
NIMCP_EXPORT bool nimcp_gpu_weight_cache_upload(
    nimcp_gpu_weight_cache_t* cache,
    neural_network_t net
);

/**
 * @brief Download weights from GPU tensors back to CPU synapse structs
 *
 * Writes GPU weight matrix values back to synapse weight fields.
 * Divides by synapse strength to preserve the strength factor separately.
 *
 * @param cache Weight cache
 * @param net Neural network destination
 * @return true on success
 */
NIMCP_EXPORT bool nimcp_gpu_weight_cache_download(
    nimcp_gpu_weight_cache_t* cache,
    neural_network_t net
);

/**
 * @brief Sync GPU activation values back to neuron state fields
 *
 * After GPU forward pass, downloads activation tensors and writes
 * values to neuron->state so CPU biological learning (STDP/BCM)
 * can read current neuron activations.
 *
 * @param cache Weight cache with computed activations
 * @param net Neural network to update
 */
NIMCP_EXPORT void nimcp_gpu_weight_cache_sync_activations(
    nimcp_gpu_weight_cache_t* cache,
    neural_network_t net
);

//=============================================================================
// GPU Compute Operations
//=============================================================================

/**
 * @brief Run forward pass on GPU
 *
 * Uploads input, runs per-layer GEMV + bias + activation + clamp,
 * downloads output. Uses existing GPU tensor API functions.
 *
 * @param cache Weight cache (weights must be uploaded)
 * @param input Input data (host memory)
 * @param input_size Input vector size
 * @param output Output buffer (host memory, pre-allocated)
 * @param output_size Output vector size
 * @return true on success
 */
NIMCP_EXPORT bool nimcp_gpu_forward_pass(
    nimcp_gpu_weight_cache_t* cache,
    const float* input,
    uint32_t input_size,
    float* output,
    uint32_t output_size
);

/**
 * @brief Compute MSE loss on GPU
 *
 * @param cache Weight cache (for GPU context)
 * @param output Network output (host memory)
 * @param target Target values (host memory)
 * @param size Vector size
 * @return MSE loss value, or -1.0f on error
 */
NIMCP_EXPORT float nimcp_gpu_compute_loss(
    nimcp_gpu_weight_cache_t* cache,
    const float* output,
    const float* target,
    uint32_t size
);

#ifdef __cplusplus
}
#endif

#endif /* NIMCP_TRAINING_BRIDGE_H */
