//=============================================================================
// nimcp_backprop_kernel.c - Shared Backprop with Parallel + SIMD
//=============================================================================
/**
 * @file nimcp_backprop_kernel.c
 * @brief Sparse backpropagation kernel extracted from nimcp_adaptive.c
 *
 * OPTIMIZATIONS:
 *   Phase 1 — Within-layer parallel dispatch for nsparse_cur > 256
 *   Phase 2 — SIMD activation derivatives (AVX2 gather→process→scatter)
 *
 * THREAD SAFETY:
 *   - Workers touch disjoint neurons from sparse_cur (no race on weights/bias)
 *   - delta_prev is per-worker (reduced after join)
 *   - grad_norm_sq is per-worker accumulator (summed after join)
 */

#include "plasticity/adaptive/nimcp_backprop_kernel.h"
#include "core/neuralnet/nimcp_neuralnet.h"
#include "core/neuralnet/nimcp_neuron_synapse_access.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/memory/nimcp_memory_pool.h"
#include "utils/thread/nimcp_thread_pool.h"
#include "utils/platform/nimcp_platform_once.h"
#include "utils/tensor/nimcp_tensor_simd.h"

#include <math.h>
#include <string.h>
#include <stdatomic.h>

//=============================================================================
// Constants
//=============================================================================

#define OUTPUT_LR_BOOST 10.0f
#define BP_PARALLEL_THRESHOLD 256
#define BP_NEURONS_PER_THREAD 64
#define BP_MAX_WORKERS 4

#define BP_POOL_BLOCK_SIZE 4096
#define BP_POOL_NUM_BLOCKS 256

//=============================================================================
// Memory Pool (mirrors nimcp_adaptive.c pattern)
//=============================================================================

static _Atomic(memory_pool_t) g_bp_pool = NULL;
static nimcp_platform_once_t g_bp_pool_once = NIMCP_PLATFORM_ONCE_INIT;

static void init_bp_pool(void) {
    memory_pool_config_t config = memory_pool_default_config(
        BP_POOL_BLOCK_SIZE, BP_POOL_NUM_BLOCKS);
    memory_pool_t pool = memory_pool_create(&config);
    atomic_store(&g_bp_pool, pool);
}

static memory_pool_t get_bp_pool(void) {
    nimcp_platform_once(&g_bp_pool_once, init_bp_pool);
    return atomic_load(&g_bp_pool);
}

void* bp_alloc_hot_buffer(size_t size) {
    if (size <= BP_POOL_BLOCK_SIZE) {
        memory_pool_t pool = get_bp_pool();
        if (pool) {
            void* buf = memory_pool_acquire(pool);
            if (buf) {
                memset(buf, 0, size);
                return buf;
            }
        }
    }
    return nimcp_calloc(1, size);
}

void bp_free_hot_buffer(void* buf) {
    if (!buf) return;
    memory_pool_t pool = get_bp_pool();
    if (pool && memory_pool_owns(pool, buf)) {
        memory_pool_release(pool, buf);
    } else {
        nimcp_free(buf);
    }
}

//=============================================================================
// Thread Pool (Phase 1: lazy-init, module-level)
//=============================================================================

static _Atomic(nimcp_thread_pool_t*) g_bp_thread_pool = NULL;
static nimcp_platform_once_t g_bp_thread_pool_once = NIMCP_PLATFORM_ONCE_INIT;

static void init_bp_thread_pool(void) {
    nimcp_thread_pool_t* pool = nimcp_pool_create(BP_MAX_WORKERS);
    atomic_store(&g_bp_thread_pool, pool);
}

static nimcp_thread_pool_t* get_bp_thread_pool(void) {
    nimcp_platform_once(&g_bp_thread_pool_once, init_bp_thread_pool);
    return atomic_load(&g_bp_thread_pool);
}

void backprop_kernel_cleanup(void) {
    nimcp_thread_pool_t* pool = atomic_load(&g_bp_thread_pool);
    if (pool) {
        nimcp_pool_destroy(pool);
        atomic_store(&g_bp_thread_pool, (nimcp_thread_pool_t*)NULL);
    }
    g_bp_thread_pool_once = NIMCP_PLATFORM_ONCE_INIT;

    memory_pool_t mpool = atomic_load(&g_bp_pool);
    if (mpool) {
        memory_pool_destroy(mpool);
        atomic_store(&g_bp_pool, (memory_pool_t)NULL);
    }
    g_bp_pool_once = NIMCP_PLATFORM_ONCE_INIT;
}

//=============================================================================
// Phase 2: SIMD Activation Derivatives
//=============================================================================

#ifdef __x86_64__
#include <immintrin.h>

__attribute__((target("avx2,fma")))
static void bp_relu_deriv_avx2(float* delta, const float* states, size_t n)
{
    __m256 zero = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 s = _mm256_loadu_ps(states + i);
        __m256 d = _mm256_loadu_ps(delta + i);
        __m256 mask = _mm256_cmp_ps(s, zero, _CMP_GT_OQ);
        d = _mm256_and_ps(d, mask);
        _mm256_storeu_ps(delta + i, d);
    }
    for (; i < n; i++) {
        if (states[i] <= 0.0f) delta[i] = 0.0f;
    }
}

__attribute__((target("avx2,fma")))
static void bp_leaky_relu_deriv_avx2(float* delta, const float* states, size_t n)
{
    __m256 zero = _mm256_setzero_ps();
    __m256 leak = _mm256_set1_ps(0.01f);
    __m256 one  = _mm256_set1_ps(1.0f);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 s = _mm256_loadu_ps(states + i);
        __m256 d = _mm256_loadu_ps(delta + i);
        __m256 mask = _mm256_cmp_ps(s, zero, _CMP_GT_OQ);
        __m256 deriv = _mm256_blendv_ps(leak, one, mask);
        d = _mm256_mul_ps(d, deriv);
        _mm256_storeu_ps(delta + i, d);
    }
    for (; i < n; i++) {
        delta[i] *= (states[i] > 0.0f) ? 1.0f : 0.01f;
    }
}

__attribute__((target("avx2,fma")))
static void bp_tanh_deriv_avx2(float* delta, const float* states, size_t n)
{
    __m256 one = _mm256_set1_ps(1.0f);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 s = _mm256_loadu_ps(states + i);
        __m256 d = _mm256_loadu_ps(delta + i);
        __m256 s2 = _mm256_mul_ps(s, s);
        __m256 deriv = _mm256_sub_ps(one, s2);
        d = _mm256_mul_ps(d, deriv);
        _mm256_storeu_ps(delta + i, d);
    }
    for (; i < n; i++) {
        float s = states[i];
        delta[i] *= (1.0f - s * s);
    }
}

__attribute__((target("avx2,fma")))
static void bp_sigmoid_deriv_avx2(float* delta, const float* states, size_t n)
{
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 floor_val = _mm256_set1_ps(0.01f);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 s = _mm256_loadu_ps(states + i);
        __m256 d = _mm256_loadu_ps(delta + i);
        __m256 deriv = _mm256_mul_ps(s, _mm256_sub_ps(one, s));
        deriv = _mm256_max_ps(deriv, floor_val);
        d = _mm256_mul_ps(d, deriv);
        _mm256_storeu_ps(delta + i, d);
    }
    for (; i < n; i++) {
        float s = states[i];
        float deriv = s * (1.0f - s);
        if (deriv < 0.01f) deriv = 0.01f;
        delta[i] *= deriv;
    }
}
#endif // __x86_64__

// Scalar fallbacks (always available)
static void bp_relu_deriv_scalar(float* delta, const float* states, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        if (states[i] <= 0.0f) delta[i] = 0.0f;
    }
}

static void bp_leaky_relu_deriv_scalar(float* delta, const float* states, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        delta[i] *= (states[i] > 0.0f) ? 1.0f : 0.01f;
    }
}

static void bp_tanh_deriv_scalar(float* delta, const float* states, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        float s = states[i];
        delta[i] *= (1.0f - s * s);
    }
}

static void bp_sigmoid_deriv_scalar(float* delta, const float* states, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        float s = states[i];
        float deriv = s * (1.0f - s);
        if (deriv < 0.01f) deriv = 0.01f;
        delta[i] *= deriv;
    }
}

// Runtime-dispatched SIMD derivative
typedef void (*bp_deriv_fn)(float* delta, const float* states, size_t n);

static bp_deriv_fn get_deriv_fn(int activation_type)
{
#ifdef __x86_64__
    tensor_simd_backend_t backend = tensor_simd_get_backend();
    if (backend >= TENSOR_SIMD_AVX2) {
        switch (activation_type) {
            case 2: /* ACTIVATION_RELU */       return bp_relu_deriv_avx2;
            case 3: /* ACTIVATION_LEAKY_RELU */ return bp_leaky_relu_deriv_avx2;
            case 1: /* ACTIVATION_TANH */       return bp_tanh_deriv_avx2;
            case 0: /* ACTIVATION_SIGMOID */    return bp_sigmoid_deriv_avx2;
            default:                            return bp_leaky_relu_deriv_avx2;
        }
    }
#endif
    switch (activation_type) {
        case 2: /* ACTIVATION_RELU */       return bp_relu_deriv_scalar;
        case 3: /* ACTIVATION_LEAKY_RELU */ return bp_leaky_relu_deriv_scalar;
        case 1: /* ACTIVATION_TANH */       return bp_tanh_deriv_scalar;
        case 0: /* ACTIVATION_SIGMOID */    return bp_sigmoid_deriv_scalar;
        default:                            return bp_leaky_relu_deriv_scalar;
    }
}

//=============================================================================
// Phase 1: Parallel Worker
//=============================================================================

typedef struct {
    neural_network_t net;
    uint32_t cur_offset;
    uint32_t prev_offset;
    uint32_t prev_size;
    float layer_lr;
    float min_weight;
    float max_weight;
    const uint32_t* sparse_cur;
    uint32_t chunk_start;   // index into sparse_cur
    uint32_t chunk_end;
    const float* delta_cur;
    bool is_output_layer;
    uint32_t target_size;
    // Per-worker outputs (caller allocates, zeroed)
    float* local_delta_prev;
    uint32_t* local_sparse_prev;
    uint8_t* local_dedup;
    uint32_t local_nsparse_prev;
    double local_grad_norm_sq;
} backprop_worker_arg_t;

static void backprop_worker(void* arg)
{
    backprop_worker_arg_t* w = (backprop_worker_arg_t*)arg;
    double gnorm = 0.0;
    uint32_t nsp = 0;

    for (uint32_t ai = w->chunk_start; ai < w->chunk_end; ai++) {
        uint32_t j = w->sparse_cur[ai];
        if (w->is_output_layer && j >= w->target_size)
            continue;

        float dj = w->delta_cur[j];
        if (dj > 1.0f) dj = 1.0f;
        if (dj < -1.0f) dj = -1.0f;

        neuron_t* cur_n = neural_network_get_neuron(w->net, w->cur_offset + j);
        if (!cur_n) continue;

        uint32_t in_count = NEURON_IN_COUNT(cur_n);
        for (uint32_t k = 0; k < in_count; k++) {
            synapse_handle_t* in_syn = NEURON_IN_HANDLE(cur_n, k);
            if (!in_syn) continue;

            uint32_t src_id = in_syn->target_neuron_id;
            neuron_t* src = neural_network_get_neuron(w->net, src_id);
            if (!src) continue;

            float weight_delta = w->layer_lr * dj * src->state;
            if (weight_delta > 0.1f) weight_delta = 0.1f;
            if (weight_delta < -0.1f) weight_delta = -0.1f;

            gnorm += (double)weight_delta * weight_delta;

            if (src_id >= w->prev_offset &&
                src_id < w->prev_offset + w->prev_size) {
                uint32_t prev_idx = src_id - w->prev_offset;
                w->local_delta_prev[prev_idx] += in_syn->weight * dj;
                if (!w->local_dedup[prev_idx]) {
                    w->local_dedup[prev_idx] = 1;
                    w->local_sparse_prev[nsp++] = prev_idx;
                }
            }

            in_syn->weight += weight_delta;
            if (in_syn->weight < w->min_weight) in_syn->weight = w->min_weight;
            if (in_syn->weight > w->max_weight) in_syn->weight = w->max_weight;
        }

        float bias_delta = w->layer_lr * dj;
        gnorm += (double)bias_delta * bias_delta;
        cur_n->bias += bias_delta;
        if (cur_n->bias > 10.0f) cur_n->bias = 10.0f;
        if (cur_n->bias < -10.0f) cur_n->bias = -10.0f;
    }

    w->local_nsparse_prev = nsp;
    w->local_grad_norm_sq = gnorm;
}

//=============================================================================
// Core: backprop_sparse_full
//=============================================================================

int backprop_sparse_full(
    neural_network_t net,
    uint32_t num_layers,
    const uint32_t* layer_sizes,
    float learning_rate,
    float min_weight, float max_weight,
    const float* target, const float* output,
    uint32_t target_size,
    float* out_grad_norm)
{
    if (!net || num_layers < 2 || !layer_sizes || !target || !output || !out_grad_norm)
        return -1;

    double grad_norm_sq = 0.0;

    // Compute layer offsets
    uint32_t* layer_offsets = (uint32_t*)nimcp_malloc(num_layers * sizeof(uint32_t));
    if (!layer_offsets) return -1;
    layer_offsets[0] = 0;
    for (uint32_t l = 1; l < num_layers; l++)
        layer_offsets[l] = layer_offsets[l - 1] + layer_sizes[l - 1];

    uint32_t output_size = layer_sizes[num_layers - 1];

    // Find max layer size for buffer allocation
    uint32_t max_layer = 0;
    for (uint32_t l = 0; l < num_layers; l++)
        if (layer_sizes[l] > max_layer) max_layer = layer_sizes[l];

    // Allocate delta buffers
    float* delta_cur = (float*)bp_alloc_hot_buffer(max_layer * sizeof(float));
    float* delta_prev = (float*)bp_alloc_hot_buffer(max_layer * sizeof(float));
    if (!delta_cur || !delta_prev) {
        if (delta_cur) bp_free_hot_buffer(delta_cur);
        if (delta_prev) bp_free_hot_buffer(delta_prev);
        nimcp_free(layer_offsets);
        return -1;
    }

    // Sparse active set tracking
    uint32_t* sparse_cur = (uint32_t*)nimcp_malloc(max_layer * sizeof(uint32_t));
    uint32_t* sparse_prev = (uint32_t*)nimcp_malloc(max_layer * sizeof(uint32_t));
    uint8_t* sparse_dedup = (uint8_t*)nimcp_calloc(max_layer, sizeof(uint8_t));
    uint32_t nsparse_cur = 0, nsparse_prev = 0;
    if (!sparse_cur || !sparse_prev || !sparse_dedup) {
        nimcp_free(sparse_cur);
        nimcp_free(sparse_prev);
        nimcp_free(sparse_dedup);
        bp_free_hot_buffer(delta_cur);
        bp_free_hot_buffer(delta_prev);
        nimcp_free(layer_offsets);
        return -1;
    }

    // Zero delta buffers once
    memset(delta_cur, 0, max_layer * sizeof(float));
    memset(delta_prev, 0, max_layer * sizeof(float));

    // Output layer deltas (CE + sigmoid): delta = target - output
    for (uint32_t j = 0; j < output_size && j < target_size; j++) {
        delta_cur[j] = target[j] - output[j];
        if (fabsf(delta_cur[j]) > 1e-10f) {
            sparse_cur[nsparse_cur++] = j;
        }
    }

    // Get thread pool for parallel dispatch
    nimcp_thread_pool_t* tpool = get_bp_thread_pool();

    // Backprop layer by layer (output → first hidden)
    for (int32_t layer = (int32_t)num_layers - 1; layer >= 1; layer--) {
        uint32_t cur_offset = layer_offsets[layer];
        uint32_t prev_size = layer_sizes[layer - 1];
        uint32_t prev_offset = layer_offsets[layer - 1];
        bool is_output = (layer == (int32_t)num_layers - 1);

        float layer_lr;
        if (is_output) {
            layer_lr = learning_rate * OUTPUT_LR_BOOST;
        } else {
            float fan_in = (float)prev_size;
            layer_lr = learning_rate / sqrtf(fmaxf(fan_in, 1.0f));
        }

        nsparse_prev = 0;

        //--------------------------------------------------------------
        // Phase 1: Parallel or serial weight update
        //--------------------------------------------------------------
        if (nsparse_cur > BP_PARALLEL_THRESHOLD && tpool) {
            uint32_t nw = nsparse_cur / BP_NEURONS_PER_THREAD;
            if (nw > BP_MAX_WORKERS) nw = BP_MAX_WORKERS;
            if (nw < 2) nw = 2;

            // Allocate per-worker arrays
            backprop_worker_arg_t* workers = (backprop_worker_arg_t*)
                nimcp_calloc(nw, sizeof(backprop_worker_arg_t));
            float** worker_delta = (float**)nimcp_calloc(nw, sizeof(float*));
            uint32_t** worker_sparse = (uint32_t**)nimcp_calloc(nw, sizeof(uint32_t*));
            uint8_t** worker_dedup = (uint8_t**)nimcp_calloc(nw, sizeof(uint8_t*));

            if (!workers || !worker_delta || !worker_sparse || !worker_dedup) {
                // Fallback to serial
                nimcp_free(workers);
                nimcp_free(worker_delta);
                nimcp_free(worker_sparse);
                nimcp_free(worker_dedup);
                goto serial_path;
            }

            bool alloc_ok = true;
            for (uint32_t w = 0; w < nw; w++) {
                worker_delta[w] = (float*)nimcp_calloc(prev_size, sizeof(float));
                worker_sparse[w] = (uint32_t*)nimcp_malloc(prev_size * sizeof(uint32_t));
                worker_dedup[w] = (uint8_t*)nimcp_calloc(prev_size, sizeof(uint8_t));
                if (!worker_delta[w] || !worker_sparse[w] || !worker_dedup[w]) {
                    alloc_ok = false;
                    break;
                }
            }

            if (!alloc_ok) {
                for (uint32_t w = 0; w < nw; w++) {
                    nimcp_free(worker_delta[w]);
                    nimcp_free(worker_sparse[w]);
                    nimcp_free(worker_dedup[w]);
                }
                nimcp_free(workers);
                nimcp_free(worker_delta);
                nimcp_free(worker_sparse);
                nimcp_free(worker_dedup);
                goto serial_path;
            }

            // Partition sparse_cur into chunks
            uint32_t chunk = nsparse_cur / nw;
            for (uint32_t w = 0; w < nw; w++) {
                workers[w].net = net;
                workers[w].cur_offset = cur_offset;
                workers[w].prev_offset = prev_offset;
                workers[w].prev_size = prev_size;
                workers[w].layer_lr = layer_lr;
                workers[w].min_weight = min_weight;
                workers[w].max_weight = max_weight;
                workers[w].sparse_cur = sparse_cur;
                workers[w].chunk_start = w * chunk;
                workers[w].chunk_end = (w == nw - 1) ? nsparse_cur : (w + 1) * chunk;
                workers[w].delta_cur = delta_cur;
                workers[w].is_output_layer = is_output;
                workers[w].target_size = target_size;
                workers[w].local_delta_prev = worker_delta[w];
                workers[w].local_sparse_prev = worker_sparse[w];
                workers[w].local_dedup = worker_dedup[w];
                workers[w].local_nsparse_prev = 0;
                workers[w].local_grad_norm_sq = 0.0;
            }

            // Submit all workers
            for (uint32_t w = 0; w < nw; w++) {
                nimcp_pool_submit(tpool, backprop_worker, &workers[w]);
            }
            nimcp_pool_wait(tpool);

            // Reduce: merge per-worker results into delta_prev
            // First worker → direct copy, subsequent → SIMD add
            for (uint32_t w = 0; w < nw; w++) {
                grad_norm_sq += workers[w].local_grad_norm_sq;

                if (w == 0) {
                    // Copy first worker's delta_prev
                    memcpy(delta_prev, worker_delta[0],
                           prev_size * sizeof(float));
                } else {
                    // SIMD add subsequent workers
                    tensor_simd_add_f32(delta_prev, worker_delta[w], prev_size);
                }

                // Merge sparse_prev indices using dedup
                for (uint32_t z = 0; z < workers[w].local_nsparse_prev; z++) {
                    uint32_t idx = workers[w].local_sparse_prev[z];
                    if (!sparse_dedup[idx]) {
                        sparse_dedup[idx] = 1;
                        sparse_prev[nsparse_prev++] = idx;
                    }
                }
            }

            // Cleanup per-worker arrays
            for (uint32_t w = 0; w < nw; w++) {
                nimcp_free(worker_delta[w]);
                nimcp_free(worker_sparse[w]);
                nimcp_free(worker_dedup[w]);
            }
            nimcp_free(workers);
            nimcp_free(worker_delta);
            nimcp_free(worker_sparse);
            nimcp_free(worker_dedup);

            goto after_weight_update;
        }

        //--------------------------------------------------------------
        // Serial path (small layer or no thread pool)
        //--------------------------------------------------------------
serial_path:
        for (uint32_t ai = 0; ai < nsparse_cur; ai++) {
            uint32_t j = sparse_cur[ai];
            if (is_output && j >= target_size)
                continue;

            float dj = delta_cur[j];
            if (dj > 1.0f) dj = 1.0f;
            if (dj < -1.0f) dj = -1.0f;

            neuron_t* cur_n = neural_network_get_neuron(net, cur_offset + j);
            if (!cur_n) continue;

            uint32_t in_count = NEURON_IN_COUNT(cur_n);
            for (uint32_t k = 0; k < in_count; k++) {
                synapse_handle_t* in_syn = NEURON_IN_HANDLE(cur_n, k);
                if (!in_syn) continue;

                uint32_t src_id = in_syn->target_neuron_id;
                neuron_t* src = neural_network_get_neuron(net, src_id);
                if (!src) continue;

                float weight_delta = layer_lr * dj * src->state;
                if (weight_delta > 0.1f) weight_delta = 0.1f;
                if (weight_delta < -0.1f) weight_delta = -0.1f;

                grad_norm_sq += (double)weight_delta * weight_delta;

                if (src_id >= prev_offset && src_id < prev_offset + prev_size) {
                    uint32_t prev_idx = src_id - prev_offset;
                    delta_prev[prev_idx] += in_syn->weight * dj;
                    if (!sparse_dedup[prev_idx]) {
                        sparse_dedup[prev_idx] = 1;
                        sparse_prev[nsparse_prev++] = prev_idx;
                    }
                }

                in_syn->weight += weight_delta;
                if (in_syn->weight < min_weight) in_syn->weight = min_weight;
                if (in_syn->weight > max_weight) in_syn->weight = max_weight;
            }

            float bias_delta = layer_lr * dj;
            grad_norm_sq += (double)bias_delta * bias_delta;
            cur_n->bias += bias_delta;
            if (cur_n->bias > 10.0f) cur_n->bias = 10.0f;
            if (cur_n->bias < -10.0f) cur_n->bias = -10.0f;
        }

after_weight_update:
        // Clear dedup flags (only active entries)
        for (uint32_t z = 0; z < nsparse_prev; z++) {
            sparse_dedup[sparse_prev[z]] = 0;
        }

        //--------------------------------------------------------------
        // Phase 2: SIMD Activation derivative (if not backprop into input)
        //--------------------------------------------------------------
        if (layer > 1) {
            // Try SIMD path: gather states, apply vectorized derivative, scatter
            if (nsparse_prev > 128) {
                // Check if uniform activation type (common case)
                neuron_t* first_n = neural_network_get_neuron(net, prev_offset + sparse_prev[0]);
                if (first_n) {
                    int act_type = (int)first_n->activation_type;
                    bool uniform = true;

                    // Gather states + deltas into contiguous arrays
                    float* gathered_states = (float*)nimcp_malloc(nsparse_prev * sizeof(float));
                    float* gathered_deltas = (float*)nimcp_malloc(nsparse_prev * sizeof(float));

                    if (gathered_states && gathered_deltas) {
                        for (uint32_t ai = 0; ai < nsparse_prev; ai++) {
                            uint32_t idx = sparse_prev[ai];
                            neuron_t* prev_n = neural_network_get_neuron(
                                net, prev_offset + idx);
                            if (prev_n) {
                                gathered_states[ai] = prev_n->state;
                                gathered_deltas[ai] = delta_prev[idx];
                                if ((int)prev_n->activation_type != act_type)
                                    uniform = false;
                            } else {
                                gathered_states[ai] = 0.0f;
                                gathered_deltas[ai] = 0.0f;
                                uniform = false;
                            }
                        }

                        if (uniform) {
                            // Vectorized derivative on contiguous arrays
                            bp_deriv_fn deriv = get_deriv_fn(act_type);
                            deriv(gathered_deltas, gathered_states, nsparse_prev);

                            // Scatter back
                            for (uint32_t ai = 0; ai < nsparse_prev; ai++) {
                                delta_prev[sparse_prev[ai]] = gathered_deltas[ai];
                            }

                            nimcp_free(gathered_states);
                            nimcp_free(gathered_deltas);
                            goto after_act_deriv;
                        }
                    }
                    nimcp_free(gathered_states);
                    nimcp_free(gathered_deltas);
                }
            }

            // Scalar fallback: per-neuron pointer chase + switch
            for (uint32_t ai = 0; ai < nsparse_prev; ai++) {
                uint32_t i = sparse_prev[ai];
                neuron_t* prev_n = neural_network_get_neuron(net, prev_offset + i);
                if (!prev_n) continue;
                float s = prev_n->state;
                float act_deriv;
                switch (prev_n->activation_type) {
                    case ACTIVATION_RELU:
                        act_deriv = (s > 0.0f) ? 1.0f : 0.0f; break;
                    case ACTIVATION_LEAKY_RELU:
                        act_deriv = (s > 0.0f) ? 1.0f : 0.01f; break;
                    case ACTIVATION_TANH:
                        act_deriv = 1.0f - s * s; break;
                    case ACTIVATION_SIGMOID:
                        act_deriv = s * (1.0f - s);
                        if (act_deriv < 0.01f) act_deriv = 0.01f; break;
                    default:
                        act_deriv = (s > 0.0f) ? 1.0f : 0.01f; break;
                }
                delta_prev[i] *= act_deriv;
            }

after_act_deriv:
            // Targeted cleanup: zero delta_cur at active indices before swap
            for (uint32_t z = 0; z < nsparse_cur; z++) {
                delta_cur[sparse_cur[z]] = 0.0f;
            }

            // Swap delta buffers + sparse tracking
            float* tmp_d = delta_cur;
            delta_cur = delta_prev;
            delta_prev = tmp_d;
            uint32_t* tmp_s = sparse_cur;
            sparse_cur = sparse_prev;
            sparse_prev = tmp_s;
            nsparse_cur = nsparse_prev;
            nsparse_prev = 0;
        }
    }

    // Cleanup
    bp_free_hot_buffer(delta_cur);
    bp_free_hot_buffer(delta_prev);
    nimcp_free(sparse_cur);
    nimcp_free(sparse_prev);
    nimcp_free(sparse_dedup);
    nimcp_free(layer_offsets);

    *out_grad_norm = sqrtf((float)grad_norm_sq);
    return 0;
}
