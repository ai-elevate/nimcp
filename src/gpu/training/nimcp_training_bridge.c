//=============================================================================
// nimcp_training_bridge.c - GPU Weight Cache & Forward Pass Bridge
//=============================================================================
/**
 * @file nimcp_training_bridge.c
 * @brief Bridges neuron/synapse AoS layout to GPU contiguous tensors
 *
 * WHAT: Extracts weight matrices from neuron_t.incoming_synapses[] into
 *       row-major GPU tensors, runs GEMV forward pass + MSE loss on GPU,
 *       and syncs activations/weights back to CPU structs.
 *
 * KEY ALGORITHM (weight extraction):
 *   For layer transition l -> l+1:
 *     For each neuron i in layer l+1:
 *       For each incoming_synapse j:
 *         src_local = source_neuron_id - layer_offset[l]
 *         W[i][src_local] = syn->weight * syn->strength
 *
 * PERFORMANCE:
 *   Upload/download are O(N*S) where N=neurons, S=avg synapses per neuron.
 *
 *   Forward pass is GPU-bound: O(sum(layer_sizes[l+1] * layer_sizes[l])).
 *   Loss computation is O(output_size) on GPU.
 */

#include "gpu/training/nimcp_training_bridge.h"
#include "core/neuralnet/nimcp_neuron_synapse_access.h"
#include "gpu/training/nimcp_training_gpu.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/exception/nimcp_exception_macros.h"

#include <math.h>
#include <string.h>

#define LOG_MODULE "gpu_training_bridge"

//=============================================================================
// Internal Helpers
//=============================================================================

/**
 * @brief Compute layer neuron offsets from layer_sizes
 *
 * layer_offsets[0] = 0 (input layer starts at neuron 0)
 * layer_offsets[1] = layer_sizes[0] (first hidden layer starts after input)
 * etc.
 */
static uint32_t* compute_layer_offsets(const uint32_t* layer_sizes, uint32_t num_layers)
{
    uint32_t* offsets = nimcp_calloc(num_layers, sizeof(uint32_t));
    if (!offsets) return NULL;

    offsets[0] = 0;
    for (uint32_t l = 1; l < num_layers; l++) {
        offsets[l] = offsets[l - 1] + layer_sizes[l - 1];
    }
    return offsets;
}

/**
 * @brief Find the maximum layer sizes for scratch buffer allocation
 */
static void find_max_sizes(const uint32_t* layer_sizes, uint32_t num_layers,
                           size_t* max_weight, size_t* max_bias, size_t* max_activation)
{
    *max_weight = 0;
    *max_bias = 0;
    *max_activation = 0;

    for (uint32_t l = 0; l < num_layers; l++) {
        if (layer_sizes[l] > *max_activation) {
            *max_activation = layer_sizes[l];
        }
    }

    for (uint32_t l = 0; l < num_layers - 1; l++) {
        size_t w = (size_t)layer_sizes[l + 1] * layer_sizes[l];
        if (w > *max_weight) *max_weight = w;
        if (layer_sizes[l + 1] > *max_bias) *max_bias = layer_sizes[l + 1];
    }
}

/**
 * @brief Apply activation function to host buffer (for GPU clamp step)
 *
 * Used only as fallback when GPU activation returns false.
 * The GPU path applies activation via nimcp_gpu_sigmoid/tanh/relu.
 */
static void clamp_host_buffer(float* buf, uint32_t size)
{
    for (uint32_t i = 0; i < size; i++) {
        if (buf[i] > 1.0f) buf[i] = 1.0f;
        else if (buf[i] < -1.0f) buf[i] = -1.0f;
    }
}

//=============================================================================
// Lifecycle
//=============================================================================

nimcp_gpu_weight_cache_t* nimcp_gpu_weight_cache_create(
    nimcp_gpu_context_t* ctx,
    neural_network_t net,
    const uint32_t* layer_sizes,
    uint32_t num_layers)
{
    if (!ctx || !net || !layer_sizes || num_layers < 2) {
        return NULL;
    }

    nimcp_gpu_weight_cache_t* cache = nimcp_calloc(1, sizeof(nimcp_gpu_weight_cache_t));
    if (!cache) return NULL;

    cache->ctx = ctx;
    cache->num_layers = num_layers;
    cache->weights_dirty_on_cpu = true;
    cache->last_sync_step = 0;

    // Copy layer sizes
    cache->layer_sizes = nimcp_calloc(num_layers, sizeof(uint32_t));
    if (!cache->layer_sizes) goto fail;
    memcpy(cache->layer_sizes, layer_sizes, num_layers * sizeof(uint32_t));

    // Compute layer offsets
    cache->layer_offsets = compute_layer_offsets(layer_sizes, num_layers);
    if (!cache->layer_offsets) goto fail;

    // Determine activation type per layer from first neuron in each layer
    cache->layer_activations = nimcp_calloc(num_layers, sizeof(activation_type_t));
    if (!cache->layer_activations) goto fail;
    for (uint32_t l = 1; l < num_layers; l++) {
        neuron_t* neuron = neural_network_get_neuron(net, cache->layer_offsets[l]);
        if (neuron) {
            cache->layer_activations[l] = neuron->activation_type;
        } else {
            cache->layer_activations[l] = ACTIVATION_TANH;  // safe default
        }
    }

    // Allocate per-transition tensor arrays (num_layers - 1 transitions)
    uint32_t num_transitions = num_layers - 1;
    cache->weights = nimcp_calloc(num_transitions, sizeof(nimcp_gpu_tensor_t*));
    cache->biases = nimcp_calloc(num_transitions, sizeof(nimcp_gpu_tensor_t*));
    if (!cache->weights || !cache->biases) goto fail;

    // Allocate per-layer activation arrays
    cache->activations = nimcp_calloc(num_layers, sizeof(nimcp_gpu_tensor_t*));
    if (!cache->activations) goto fail;

    // Create GPU tensors for each layer transition
    for (uint32_t l = 0; l < num_transitions; l++) {
        uint32_t rows = layer_sizes[l + 1];  // output neurons
        uint32_t cols = layer_sizes[l];       // input neurons

        // Weight matrix: (rows x cols) = (layer_sizes[l+1] x layer_sizes[l])
        size_t w_dims[2] = { rows, cols };
        cache->weights[l] = nimcp_gpu_tensor_create(ctx, w_dims, 2, NIMCP_GPU_PRECISION_FP32);
        if (!cache->weights[l]) goto fail;

        // Bias vector: (rows)
        size_t b_dims[1] = { rows };
        cache->biases[l] = nimcp_gpu_tensor_create(ctx, b_dims, 1, NIMCP_GPU_PRECISION_FP32);
        if (!cache->biases[l]) goto fail;
    }

    // Create activation tensors for each layer
    for (uint32_t l = 0; l < num_layers; l++) {
        size_t a_dims[1] = { layer_sizes[l] };
        cache->activations[l] = nimcp_gpu_tensor_create(ctx, a_dims, 1, NIMCP_GPU_PRECISION_FP32);
        if (!cache->activations[l]) goto fail;
    }

    // Allocate host scratch buffers (reused across upload/download calls)
    find_max_sizes(layer_sizes, num_layers,
                   &cache->host_weight_buf_size,
                   &cache->host_bias_buf_size,
                   &cache->host_activation_buf_size);

    cache->host_weight_buf = nimcp_calloc(cache->host_weight_buf_size, sizeof(float));
    cache->host_bias_buf = nimcp_calloc(cache->host_bias_buf_size, sizeof(float));
    cache->host_activation_buf = nimcp_calloc(cache->host_activation_buf_size, sizeof(float));
    if (!cache->host_weight_buf || !cache->host_bias_buf || !cache->host_activation_buf) goto fail;

    NIMCP_LOG_INFO("GPU weight cache created: %u layers, %u transitions",
                   num_layers, num_transitions);
    return cache;

fail:
    nimcp_gpu_weight_cache_destroy(cache);
    return NULL;
}

void nimcp_gpu_weight_cache_destroy(nimcp_gpu_weight_cache_t* cache)
{
    if (!cache) return;

    uint32_t num_transitions = (cache->num_layers > 1) ? cache->num_layers - 1 : 0;

    // Free GPU weight/bias tensors
    if (cache->weights) {
        for (uint32_t l = 0; l < num_transitions; l++) {
            if (cache->weights[l]) nimcp_gpu_tensor_destroy(cache->weights[l]);
        }
        nimcp_free(cache->weights);
    }
    if (cache->biases) {
        for (uint32_t l = 0; l < num_transitions; l++) {
            if (cache->biases[l]) nimcp_gpu_tensor_destroy(cache->biases[l]);
        }
        nimcp_free(cache->biases);
    }

    // Free GPU activation tensors
    if (cache->activations) {
        for (uint32_t l = 0; l < cache->num_layers; l++) {
            if (cache->activations[l]) nimcp_gpu_tensor_destroy(cache->activations[l]);
        }
        nimcp_free(cache->activations);
    }

    // Free CPU arrays
    nimcp_free(cache->layer_sizes);
    nimcp_free(cache->layer_offsets);
    nimcp_free(cache->layer_activations);
    nimcp_free(cache->host_weight_buf);
    nimcp_free(cache->host_bias_buf);
    nimcp_free(cache->host_activation_buf);

    nimcp_free(cache);
}

//=============================================================================
// Weight Synchronization
//=============================================================================

bool nimcp_gpu_weight_cache_upload(nimcp_gpu_weight_cache_t* cache, neural_network_t net)
{
    if (!cache || !net) return false;

    uint32_t num_transitions = cache->num_layers - 1;

    for (uint32_t l = 0; l < num_transitions; l++) {
        uint32_t dst_layer_size = cache->layer_sizes[l + 1];  // post-synaptic
        uint32_t src_layer_size = cache->layer_sizes[l];       // pre-synaptic
        uint32_t dst_offset = cache->layer_offsets[l + 1];
        uint32_t src_offset = cache->layer_offsets[l];

        // Zero the scratch buffers
        size_t w_count = (size_t)dst_layer_size * src_layer_size;
        memset(cache->host_weight_buf, 0, w_count * sizeof(float));
        memset(cache->host_bias_buf, 0, dst_layer_size * sizeof(float));

        // Extract weights from neuron incoming_synapses into row-major matrix
        // W[i][j] = effective weight from pre-neuron j to post-neuron i
        for (uint32_t i = 0; i < dst_layer_size; i++) {
            uint32_t neuron_id = dst_offset + i;
            neuron_t* neuron = neural_network_get_neuron(net, neuron_id);
            if (!neuron) continue;

            // Extract bias
            cache->host_bias_buf[i] = neuron->bias;

            // Extract incoming synapse weights
            for (uint32_t j = 0; j < NEURON_IN_COUNT(neuron); j++) {
                synapse_handle_t* in_h = NEURON_IN_HANDLE(neuron, j);
                if (!in_h) continue;

                // Convert global source_neuron_id to layer-local index
                // In incoming handles, target_neuron_id stores the source neuron ID
                uint32_t source_id = in_h->target_neuron_id;
                if (source_id >= src_offset &&
                    source_id < src_offset + src_layer_size) {
                    uint32_t src_local = source_id - src_offset;

                    // Effective weight = weight * strength (same as CPU forward pass)
                    float eff_weight = in_h->weight * in_h->strength;

                    // Row-major: W[i][src_local] = W[i * src_layer_size + src_local]
                    cache->host_weight_buf[i * src_layer_size + src_local] = eff_weight;
                }
                // Synapses from other layers are skip connections — not in this matrix
            }
        }

        // Upload weight matrix to GPU
        size_t w_dims[2] = { dst_layer_size, src_layer_size };
        nimcp_gpu_tensor_t* w_upload = nimcp_gpu_tensor_from_host(
            cache->ctx, cache->host_weight_buf, w_dims, 2, NIMCP_GPU_PRECISION_FP32);
        if (!w_upload) return false;

        // Swap: destroy old GPU tensor, replace with uploaded one
        nimcp_gpu_tensor_destroy(cache->weights[l]);
        cache->weights[l] = w_upload;

        // Upload bias vector to GPU
        size_t b_dims[1] = { dst_layer_size };
        nimcp_gpu_tensor_t* b_upload = nimcp_gpu_tensor_from_host(
            cache->ctx, cache->host_bias_buf, b_dims, 1, NIMCP_GPU_PRECISION_FP32);
        if (!b_upload) return false;

        nimcp_gpu_tensor_destroy(cache->biases[l]);
        cache->biases[l] = b_upload;
    }

    cache->weights_dirty_on_cpu = false;
    cache->last_sync_step++;
    return true;
}

bool nimcp_gpu_weight_cache_download(nimcp_gpu_weight_cache_t* cache, neural_network_t net)
{
    if (!cache || !net) return false;

    uint32_t num_transitions = cache->num_layers - 1;

    for (uint32_t l = 0; l < num_transitions; l++) {
        uint32_t dst_layer_size = cache->layer_sizes[l + 1];
        uint32_t src_layer_size = cache->layer_sizes[l];
        uint32_t dst_offset = cache->layer_offsets[l + 1];
        uint32_t src_offset = cache->layer_offsets[l];

        // Download weight matrix from GPU
        size_t w_count = (size_t)dst_layer_size * src_layer_size;
        memset(cache->host_weight_buf, 0, w_count * sizeof(float));

        if (!nimcp_gpu_tensor_to_host(cache->weights[l], cache->host_weight_buf)) {
            return false;
        }

        // Write back to synapse structs
        for (uint32_t i = 0; i < dst_layer_size; i++) {
            uint32_t neuron_id = dst_offset + i;
            neuron_t* neuron = neural_network_get_neuron(net, neuron_id);
            if (!neuron) continue;

            for (uint32_t j = 0; j < NEURON_IN_COUNT(neuron); j++) {
                synapse_handle_t* in_h = NEURON_IN_HANDLE(neuron, j);
                if (!in_h) continue;

                uint32_t source_id = in_h->target_neuron_id;
                if (source_id >= src_offset &&
                    source_id < src_offset + src_layer_size) {
                    uint32_t src_local = source_id - src_offset;
                    float eff_weight = cache->host_weight_buf[i * src_layer_size + src_local];

                    // Divide by strength to preserve strength separately
                    // Guard against division by zero
                    if (fabsf(in_h->strength) > 1e-8f) {
                        in_h->weight = eff_weight / in_h->strength;
                    } else {
                        in_h->weight = eff_weight;
                    }
                }
            }
        }
    }

    return true;
}

void nimcp_gpu_weight_cache_sync_activations(nimcp_gpu_weight_cache_t* cache, neural_network_t net)
{
    if (!cache || !net) return;

    for (uint32_t l = 0; l < cache->num_layers; l++) {
        uint32_t layer_size = cache->layer_sizes[l];
        uint32_t offset = cache->layer_offsets[l];

        memset(cache->host_activation_buf, 0, layer_size * sizeof(float));

        if (!nimcp_gpu_tensor_to_host(cache->activations[l], cache->host_activation_buf)) {
            continue;
        }

        // Write activation values back to neuron->state
        for (uint32_t i = 0; i < layer_size; i++) {
            neuron_t* neuron = neural_network_get_neuron(net, offset + i);
            if (neuron) {
                neuron->state = cache->host_activation_buf[i];
            }
        }
    }
}

//=============================================================================
// GPU Forward Pass
//=============================================================================

bool nimcp_gpu_forward_pass(
    nimcp_gpu_weight_cache_t* cache,
    const float* input,
    uint32_t input_size,
    float* output,
    uint32_t output_size)
{
    if (!cache || !input || !output) return false;
    if (input_size != cache->layer_sizes[0]) return false;
    if (output_size != cache->layer_sizes[cache->num_layers - 1]) return false;

    // Step 1: Upload input to layer 0 activation tensor
    size_t in_dims[1] = { input_size };
    nimcp_gpu_tensor_t* input_tensor = nimcp_gpu_tensor_from_host(
        cache->ctx, input, in_dims, 1, NIMCP_GPU_PRECISION_FP32);
    if (!input_tensor) return false;

    // Replace layer 0 activation with input
    nimcp_gpu_tensor_destroy(cache->activations[0]);
    cache->activations[0] = input_tensor;

    // Step 2: Forward pass through each layer transition
    uint32_t num_transitions = cache->num_layers - 1;
    for (uint32_t l = 0; l < num_transitions; l++) {
        // y = W[l] @ a[l]  (GEMV: matrix-vector multiply)
        // activations[l+1] = alpha * weights[l] @ activations[l] + beta * activations[l+1]
        // With alpha=1.0, beta=0.0: pure GEMV
        if (!nimcp_gpu_gemv(cache->ctx,
                            cache->weights[l],      // A = W[l] (rows x cols)
                            cache->activations[l],  // x = a[l] (cols)
                            cache->activations[l + 1], // y = a[l+1] (rows)
                            1.0f, 0.0f, false)) {
            NIMCP_LOG_ERROR("GPU GEMV failed for layer %u", l);
            return false;
        }

        // Add bias: a[l+1] = a[l+1] + b[l]
        if (!nimcp_gpu_add(cache->ctx,
                           cache->activations[l + 1],
                           cache->biases[l],
                           cache->activations[l + 1])) {
            NIMCP_LOG_ERROR("GPU bias add failed for layer %u", l);
            return false;
        }

        // Apply activation function based on per-layer type
        activation_type_t act = cache->layer_activations[l + 1];

        // We need a temporary tensor for in-place activation
        // Use activations[l+1] as both input and output (most GPU ops support this)
        bool act_ok = false;
        switch (act) {
            case ACTIVATION_SIGMOID:
                act_ok = nimcp_gpu_sigmoid(cache->ctx,
                    cache->activations[l + 1], cache->activations[l + 1]);
                break;
            case ACTIVATION_TANH:
                act_ok = nimcp_gpu_tanh(cache->ctx,
                    cache->activations[l + 1], cache->activations[l + 1]);
                break;
            case ACTIVATION_RELU:
                act_ok = nimcp_gpu_relu(cache->ctx,
                    cache->activations[l + 1], cache->activations[l + 1]);
                break;
            case ACTIVATION_LEAKY_RELU:
                act_ok = nimcp_gpu_leaky_relu(cache->ctx,
                    cache->activations[l + 1], cache->activations[l + 1], 0.01f);
                break;
            case ACTIVATION_ADAPTIVE:
                // Adaptive activation: tanh((x - threshold) / 10) for x > threshold, else 0
                // Fall back to tanh for GPU (threshold handling is CPU-specific)
                act_ok = nimcp_gpu_tanh(cache->ctx,
                    cache->activations[l + 1], cache->activations[l + 1]);
                break;
            default:
                act_ok = nimcp_gpu_tanh(cache->ctx,
                    cache->activations[l + 1], cache->activations[l + 1]);
                break;
        }

        if (!act_ok) {
            NIMCP_LOG_ERROR("GPU activation failed for layer %u", l);
            return false;
        }

        // Clamp to [-1, 1] — match CPU behavior (line 2798 of nimcp_neuralnet.c)
        // Download, clamp on CPU, re-upload
        // This is a minor overhead but ensures numerical parity with CPU path
        uint32_t layer_size = cache->layer_sizes[l + 1];
        if (!nimcp_gpu_tensor_to_host(cache->activations[l + 1],
                                      cache->host_activation_buf)) {
            return false;
        }
        clamp_host_buffer(cache->host_activation_buf, layer_size);

        size_t a_dims[1] = { layer_size };
        nimcp_gpu_tensor_t* clamped = nimcp_gpu_tensor_from_host(
            cache->ctx, cache->host_activation_buf, a_dims, 1, NIMCP_GPU_PRECISION_FP32);
        if (!clamped) return false;

        nimcp_gpu_tensor_destroy(cache->activations[l + 1]);
        cache->activations[l + 1] = clamped;
    }

    // Step 3: Download output from last layer activation
    uint32_t last_layer = cache->num_layers - 1;
    if (!nimcp_gpu_tensor_to_host(cache->activations[last_layer], output)) {
        return false;
    }

    return true;
}

//=============================================================================
// GPU Loss Computation
//=============================================================================

float nimcp_gpu_compute_loss(
    nimcp_gpu_weight_cache_t* cache,
    const float* output,
    const float* target,
    uint32_t size)
{
    if (!cache || !output || !target || size == 0) return -1.0f;

    // Upload output and target to GPU tensors
    size_t dims[1] = { size };

    nimcp_gpu_tensor_t* pred_tensor = nimcp_gpu_tensor_from_host(
        cache->ctx, output, dims, 1, NIMCP_GPU_PRECISION_FP32);
    if (!pred_tensor) return -1.0f;

    nimcp_gpu_tensor_t* target_tensor = nimcp_gpu_tensor_from_host(
        cache->ctx, target, dims, 1, NIMCP_GPU_PRECISION_FP32);
    if (!target_tensor) {
        nimcp_gpu_tensor_destroy(pred_tensor);
        return -1.0f;
    }

    // Compute MSE loss
    float loss = -1.0f;
    bool ok = nimcp_gpu_loss_mse(cache->ctx, pred_tensor, target_tensor, &loss, NULL);

    nimcp_gpu_tensor_destroy(pred_tensor);
    nimcp_gpu_tensor_destroy(target_tensor);

    return ok ? loss : -1.0f;
}
