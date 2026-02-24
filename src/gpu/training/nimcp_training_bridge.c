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
#include "gpu/sparse/nimcp_sparse_gpu.h"
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
 *
 * Only computes max_bias and max_activation — weights are now sparse
 * with dynamically-grown COO/CSR scratch buffers.
 */
static void find_max_sizes(const uint32_t* layer_sizes, uint32_t num_layers,
                           size_t* max_bias, size_t* max_activation)
{
    *max_bias = 0;
    *max_activation = 0;

    for (uint32_t l = 0; l < num_layers; l++) {
        if (layer_sizes[l] > *max_activation) {
            *max_activation = layer_sizes[l];
        }
    }

    for (uint32_t l = 0; l < num_layers - 1; l++) {
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

/**
 * @brief Ensure COO/CSR scratch buffers can hold the given sizes
 *
 * Uses 2x doubling strategy to amortize reallocation cost.
 *
 * @param cache Weight cache owning the scratch buffers
 * @param nnz Required COO entry count
 * @param rows Required row count (for CSR row_ptrs = rows+1)
 * @return true on success, false on allocation failure
 */
static bool ensure_coo_capacity(nimcp_gpu_weight_cache_t* cache, size_t nnz, size_t rows)
{
    // Grow COO arrays (values, row_idx, col_idx) if needed
    if (nnz > cache->host_coo_capacity) {
        size_t new_cap = cache->host_coo_capacity ? cache->host_coo_capacity : 1024;
        while (new_cap < nnz) new_cap *= 2;

        float* new_vals = nimcp_realloc(cache->host_coo_values, new_cap * sizeof(float));
        int* new_rows   = nimcp_realloc(cache->host_coo_row_idx, new_cap * sizeof(int));
        int* new_cols   = nimcp_realloc(cache->host_coo_col_idx, new_cap * sizeof(int));
        if (!new_vals || !new_rows || !new_cols) {
            // Partial realloc — keep old pointers (realloc preserves old on failure)
            if (new_vals) cache->host_coo_values  = new_vals;
            if (new_rows) cache->host_coo_row_idx = new_rows;
            if (new_cols) cache->host_coo_col_idx = new_cols;
            return false;
        }
        cache->host_coo_values  = new_vals;
        cache->host_coo_row_idx = new_rows;
        cache->host_coo_col_idx = new_cols;
        cache->host_coo_capacity = new_cap;
    }

    // Grow CSR row_ptrs if needed (rows + 1 entries)
    size_t row_ptrs_needed = rows + 1;
    if (row_ptrs_needed > cache->host_csr_row_ptrs_capacity) {
        size_t new_cap = cache->host_csr_row_ptrs_capacity ? cache->host_csr_row_ptrs_capacity : 256;
        while (new_cap < row_ptrs_needed) new_cap *= 2;

        int* new_ptrs = nimcp_realloc(cache->host_csr_row_ptrs, new_cap * sizeof(int));
        if (!new_ptrs) return false;
        cache->host_csr_row_ptrs = new_ptrs;
        cache->host_csr_row_ptrs_capacity = new_cap;
    }

    return true;
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

    // Create sparse context for cuSPARSE operations
    cache->sparse_ctx = nimcp_sparse_ctx_create(ctx);
    if (!cache->sparse_ctx) goto fail;

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

    // Allocate per-transition arrays (num_layers - 1 transitions)
    uint32_t num_transitions = num_layers - 1;

    // Sparse weights: NULL-initialized, populated during upload
    cache->sparse_weights = nimcp_calloc(num_transitions, sizeof(nimcp_sparse_tensor_t*));
    if (!cache->sparse_weights) goto fail;

    // Dense bias tensors
    cache->biases = nimcp_calloc(num_transitions, sizeof(nimcp_gpu_tensor_t*));
    if (!cache->biases) goto fail;

    // Allocate per-layer activation arrays
    cache->activations = nimcp_calloc(num_layers, sizeof(nimcp_gpu_tensor_t*));
    if (!cache->activations) goto fail;

    // Create GPU tensors for bias vectors (sparse weights created during upload)
    for (uint32_t l = 0; l < num_transitions; l++) {
        uint32_t rows = layer_sizes[l + 1];  // output neurons
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

    // Allocate host scratch buffers for bias and activation (reused across calls)
    // COO/CSR scratch buffers are grown dynamically on first upload
    find_max_sizes(layer_sizes, num_layers,
                   &cache->host_bias_buf_size,
                   &cache->host_activation_buf_size);

    cache->host_coo_values  = NULL;
    cache->host_coo_row_idx = NULL;
    cache->host_coo_col_idx = NULL;
    cache->host_csr_row_ptrs = NULL;
    cache->host_coo_capacity = 0;
    cache->host_csr_row_ptrs_capacity = 0;

    cache->host_bias_buf = nimcp_calloc(cache->host_bias_buf_size, sizeof(float));
    cache->host_activation_buf = nimcp_calloc(cache->host_activation_buf_size, sizeof(float));
    if (!cache->host_bias_buf || !cache->host_activation_buf) goto fail;

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

    // Free sparse weight tensors
    if (cache->sparse_weights) {
        for (uint32_t l = 0; l < num_transitions; l++) {
            if (cache->sparse_weights[l]) nimcp_sparse_tensor_destroy(cache->sparse_weights[l]);
        }
        nimcp_free(cache->sparse_weights);
    }

    // Free sparse context
    if (cache->sparse_ctx) {
        nimcp_sparse_ctx_destroy(cache->sparse_ctx);
    }

    // Free GPU bias tensors
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
    nimcp_free(cache->host_coo_values);
    nimcp_free(cache->host_coo_row_idx);
    nimcp_free(cache->host_coo_col_idx);
    nimcp_free(cache->host_csr_row_ptrs);
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

        memset(cache->host_bias_buf, 0, dst_layer_size * sizeof(float));

        // --- Pass 1: Count nnz and extract biases ---
        size_t nnz = 0;
        for (uint32_t i = 0; i < dst_layer_size; i++) {
            uint32_t neuron_id = dst_offset + i;
            neuron_t* neuron = neural_network_get_neuron(net, neuron_id);
            if (!neuron) continue;

            cache->host_bias_buf[i] = neuron->bias;

            for (uint32_t j = 0; j < NEURON_IN_COUNT(neuron); j++) {
                synapse_handle_t* in_h = NEURON_IN_HANDLE(neuron, j);
                if (!in_h) continue;

                uint32_t source_id = in_h->target_neuron_id;
                if (source_id >= src_offset &&
                    source_id < src_offset + src_layer_size) {
                    nnz++;
                }
            }
        }

        // Ensure COO/CSR scratch buffers are large enough
        if (nnz > 0 && !ensure_coo_capacity(cache, nnz, dst_layer_size)) {
            return false;
        }

        // --- Pass 2: Fill COO arrays ---
        size_t k = 0;
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
                    float eff_weight = in_h->weight * in_h->strength;

                    cache->host_coo_values[k]  = eff_weight;
                    cache->host_coo_row_idx[k] = (int)i;
                    cache->host_coo_col_idx[k] = (int)src_local;
                    k++;
                }
            }
        }

        // --- Upload sparse weights via COO → CSR ---
        // Destroy previous sparse tensor for this layer
        if (cache->sparse_weights[l]) {
            nimcp_sparse_tensor_destroy(cache->sparse_weights[l]);
            cache->sparse_weights[l] = NULL;
        }

        if (nnz > 0) {
            cache->sparse_weights[l] = nimcp_sparse_from_coo(
                cache->sparse_ctx,
                cache->host_coo_values,
                cache->host_coo_row_idx,
                cache->host_coo_col_idx,
                (int)dst_layer_size,
                (int)src_layer_size,
                (int)nnz,
                SPARSE_FORMAT_CSR);
            if (!cache->sparse_weights[l]) return false;
        }
        // nnz == 0: sparse_weights[l] stays NULL (valid empty transition)

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

        // Skip empty transitions
        if (!cache->sparse_weights[l]) continue;

        int nnz = nimcp_sparse_nnz(cache->sparse_weights[l]);
        if (nnz == 0) continue;

        // Ensure scratch buffers can hold the CSR data
        if (!ensure_coo_capacity(cache, (size_t)nnz, (size_t)dst_layer_size)) {
            return false;
        }

        // Download CSR data from GPU to host
        if (!nimcp_sparse_to_host_csr(cache->sparse_weights[l],
                                       cache->host_coo_values,
                                       cache->host_coo_col_idx,
                                       cache->host_csr_row_ptrs)) {
            return false;
        }

        // Write back to synapse structs using CSR structure
        for (uint32_t i = 0; i < dst_layer_size; i++) {
            uint32_t neuron_id = dst_offset + i;
            neuron_t* neuron = neural_network_get_neuron(net, neuron_id);
            if (!neuron) continue;

            int row_start = cache->host_csr_row_ptrs[i];
            int row_end   = cache->host_csr_row_ptrs[i + 1];

            for (uint32_t j = 0; j < NEURON_IN_COUNT(neuron); j++) {
                synapse_handle_t* in_h = NEURON_IN_HANDLE(neuron, j);
                if (!in_h) continue;

                uint32_t source_id = in_h->target_neuron_id;
                if (source_id >= src_offset &&
                    source_id < src_offset + src_layer_size) {
                    int src_local = (int)(source_id - src_offset);

                    // Linear scan CSR row for matching column
                    float eff_weight = 0.0f;
                    for (int p = row_start; p < row_end; p++) {
                        if (cache->host_coo_col_idx[p] == src_local) {
                            eff_weight = cache->host_coo_values[p];
                            break;
                        }
                    }

                    // Divide by strength to preserve strength separately
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
        // y = W[l] @ a[l]  (SpMV: sparse matrix-vector multiply)
        if (cache->sparse_weights[l]) {
            nimcp_gpu_tensor_t* mv_result = nimcp_sparse_mv(
                cache->sparse_ctx,
                cache->sparse_weights[l],       // A = W[l] sparse CSR
                cache->activations[l],          // x = a[l] (cols)
                1.0f, 0.0f,
                cache->activations[l + 1]);     // y = a[l+1] (rows)

            if (!mv_result) {
                NIMCP_LOG_ERROR("GPU SpMV failed for layer %u", l);
                return false;
            }
            // If SpMV returned a new tensor, swap it in
            if (mv_result != cache->activations[l + 1]) {
                nimcp_gpu_tensor_destroy(cache->activations[l + 1]);
                cache->activations[l + 1] = mv_result;
            }
        } else {
            // sparse_weights[l] is NULL (nnz=0): zero the activation tensor
            // so bias add starts from 0 rather than uninitialized GPU memory
            uint32_t ls = cache->layer_sizes[l + 1];
            memset(cache->host_activation_buf, 0, ls * sizeof(float));
            size_t a_dims[1] = { ls };
            nimcp_gpu_tensor_t* zero_tensor = nimcp_gpu_tensor_from_host(
                cache->ctx, cache->host_activation_buf, a_dims, 1, NIMCP_GPU_PRECISION_FP32);
            if (!zero_tensor) return false;
            nimcp_gpu_tensor_destroy(cache->activations[l + 1]);
            cache->activations[l + 1] = zero_tensor;

            NIMCP_LOG_WARN("GPU forward: sparse_weights[%u] is NULL (no connections L%u->L%u), bias-only",
                          l, l, l+1);
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

        // Clamp unbounded activations to prevent float overflow
        // ReLU/Leaky ReLU need wider range to preserve discrimination
        // Sigmoid/tanh already self-bound — skip clamping for them
        if (act == ACTIVATION_RELU || act == ACTIVATION_LEAKY_RELU) {
            // Wide clamp for unbounded activations
            uint32_t layer_size = cache->layer_sizes[l + 1];
            if (!nimcp_gpu_tensor_to_host(cache->activations[l + 1],
                                          cache->host_activation_buf)) {
                return false;
            }
            for (uint32_t ci = 0; ci < layer_size; ci++) {
                if (cache->host_activation_buf[ci] > 100.0f)
                    cache->host_activation_buf[ci] = 100.0f;
                else if (cache->host_activation_buf[ci] < -100.0f)
                    cache->host_activation_buf[ci] = -100.0f;
            }
            size_t a_dims[1] = { layer_size };
            nimcp_gpu_tensor_t* clamped = nimcp_gpu_tensor_from_host(
                cache->ctx, cache->host_activation_buf, a_dims, 1, NIMCP_GPU_PRECISION_FP32);
            if (!clamped) return false;
            nimcp_gpu_tensor_destroy(cache->activations[l + 1]);
            cache->activations[l + 1] = clamped;
        }
        // sigmoid [0,1], tanh [-1,1] — already bounded, no clamp needed

        // (debug prints removed — GPU forward pass verified working)
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

//=============================================================================
// Batched GPU Forward Pass
//=============================================================================

bool nimcp_gpu_forward_pass_batch(
    nimcp_gpu_weight_cache_t* cache,
    const float* inputs,
    uint32_t batch_size,
    uint32_t input_size,
    float* outputs,
    uint32_t output_size)
{
    if (!cache || !inputs || !outputs || batch_size == 0) return false;
    if (input_size != cache->layer_sizes[0]) return false;
    if (output_size != cache->layer_sizes[cache->num_layers - 1]) return false;

    // For batch_size == 1, delegate to single forward pass
    if (batch_size == 1) {
        return nimcp_gpu_forward_pass(cache, inputs, input_size, outputs, output_size);
    }

    // Upload input batch as 2D tensor [batch_size x input_size]
    size_t in_dims[2] = { batch_size, input_size };
    nimcp_gpu_tensor_t* act = nimcp_gpu_tensor_from_host(
        cache->ctx, inputs, in_dims, 2, NIMCP_GPU_PRECISION_FP32);
    if (!act) return false;

    uint32_t num_transitions = cache->num_layers - 1;

    // Allocate host scratch for batched clamp
    size_t max_layer = 0;
    for (uint32_t l = 0; l <= num_transitions; l++) {
        if (cache->layer_sizes[l] > max_layer)
            max_layer = cache->layer_sizes[l];
    }
    float* clamp_buf = (float*)malloc(batch_size * max_layer * sizeof(float));
    if (!clamp_buf) {
        nimcp_gpu_tensor_destroy(act);
        return false;
    }

    for (uint32_t l = 0; l < num_transitions; l++) {
        uint32_t out_size = cache->layer_sizes[l + 1];

        // SpMM: result = W[l] @ act^T, then transpose back
        // nimcp_sparse_mm_batched expects: A [out x in] sparse, B [in x batch] dense
        // Our act is [batch x in], so we need B = act^T [in x batch]
        // Result C = [out x batch], then transpose to [batch x out]
        //
        // Alternative: process per-sample on GPU (still faster than CPU)
        // Use per-sample SpMV loop on GPU for correctness, given sparse_mm_batched
        // expects column-major B which our row-major doesn't match directly.

        // Per-sample SpMV within the batch — still GPU-accelerated
        size_t out_dims[2] = { batch_size, out_size };
        nimcp_gpu_tensor_t* next_act = nimcp_gpu_tensor_create(
            cache->ctx, out_dims, 2, NIMCP_GPU_PRECISION_FP32);
        if (!next_act) {
            nimcp_gpu_tensor_destroy(act);
            free(clamp_buf);
            return false;
        }

        // Download current activations to host for per-sample processing
        float* act_host = (float*)malloc(batch_size * cache->layer_sizes[l] * sizeof(float));
        if (!act_host) {
            nimcp_gpu_tensor_destroy(act);
            nimcp_gpu_tensor_destroy(next_act);
            free(clamp_buf);
            return false;
        }
        nimcp_gpu_tensor_to_host(act, act_host);

        float* out_host = (float*)malloc(batch_size * out_size * sizeof(float));
        if (!out_host) {
            nimcp_gpu_tensor_destroy(act);
            nimcp_gpu_tensor_destroy(next_act);
            free(act_host);
            free(clamp_buf);
            return false;
        }

        // Process each sample through the single-layer GPU SpMV
        for (uint32_t s = 0; s < batch_size; s++) {
            const float* sample_in = act_host + s * cache->layer_sizes[l];
            float* sample_out = out_host + s * out_size;

            size_t sv_in_dims[1] = { cache->layer_sizes[l] };
            nimcp_gpu_tensor_t* sv_input = nimcp_gpu_tensor_from_host(
                cache->ctx, sample_in, sv_in_dims, 1, NIMCP_GPU_PRECISION_FP32);
            if (!sv_input) { sample_out[0] = 0; continue; }

            size_t sv_out_dims[1] = { out_size };
            nimcp_gpu_tensor_t* sv_output = nimcp_gpu_tensor_create(
                cache->ctx, sv_out_dims, 1, NIMCP_GPU_PRECISION_FP32);
            if (!sv_output) { nimcp_gpu_tensor_destroy(sv_input); continue; }

            // SpMV
            if (cache->sparse_weights[l]) {
                nimcp_gpu_tensor_t* mv_result = nimcp_sparse_mv(
                    cache->sparse_ctx,
                    cache->sparse_weights[l],
                    sv_input, 1.0f, 0.0f, sv_output);
                if (mv_result && mv_result != sv_output) {
                    nimcp_gpu_tensor_destroy(sv_output);
                    sv_output = mv_result;
                }
            }

            // Add bias
            nimcp_gpu_add(cache->ctx, sv_output, cache->biases[l], sv_output);

            // Activation
            activation_type_t atype = cache->layer_activations[l + 1];
            switch (atype) {
                case ACTIVATION_SIGMOID:
                    nimcp_gpu_sigmoid(cache->ctx, sv_output, sv_output); break;
                case ACTIVATION_TANH:
                    nimcp_gpu_tanh(cache->ctx, sv_output, sv_output); break;
                case ACTIVATION_RELU:
                    nimcp_gpu_relu(cache->ctx, sv_output, sv_output); break;
                case ACTIVATION_LEAKY_RELU:
                    nimcp_gpu_leaky_relu(cache->ctx, sv_output, sv_output, 0.01f); break;
                default:
                    nimcp_gpu_tanh(cache->ctx, sv_output, sv_output); break;
            }

            // Download — only clamp unbounded activations
            nimcp_gpu_tensor_to_host(sv_output, sample_out);
            if (atype == ACTIVATION_RELU || atype == ACTIVATION_LEAKY_RELU) {
                for (uint32_t ci = 0; ci < out_size; ci++) {
                    if (sample_out[ci] > 100.0f) sample_out[ci] = 100.0f;
                    else if (sample_out[ci] < -100.0f) sample_out[ci] = -100.0f;
                }
            }

            nimcp_gpu_tensor_destroy(sv_input);
            nimcp_gpu_tensor_destroy(sv_output);
        }

        // Upload clamped batch result back to GPU
        nimcp_gpu_tensor_destroy(act);
        nimcp_gpu_tensor_destroy(next_act);
        free(act_host);

        act = nimcp_gpu_tensor_from_host(
            cache->ctx, out_host, out_dims, 2, NIMCP_GPU_PRECISION_FP32);
        free(out_host);

        if (!act) {
            free(clamp_buf);
            return false;
        }
    }

    // Download final output
    nimcp_gpu_tensor_to_host(act, outputs);
    nimcp_gpu_tensor_destroy(act);
    free(clamp_buf);
    return true;
}
