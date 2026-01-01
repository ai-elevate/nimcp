/**
 * @file nimcp_working_memory_gpu.cu
 * @brief GPU-accelerated working memory decay kernel
 *
 * WHAT: CUDA kernel for exponential decay of salience in working memory
 * WHY:  GPU acceleration for API consistency with other accelerated modules
 * HOW:  Parallel exponential decay computation across working memory items
 *
 * HOT PATH ACCELERATED:
 * - Exponential decay on salience array: salience[i] *= exp(-elapsed / tau)
 *
 * NOTE ON DATA SIZE:
 * Working memory typically has 7-32 items (Miller's 7+/-2). While small data
 * size means limited single-operation benefit, GPU acceleration is valuable:
 * 1. API consistency with other GPU-accelerated modules
 * 2. Enables batching with other cognitive operations
 * 3. Prepares for future extensions (larger associative buffers)
 * 4. Zero-copy integration when data already on GPU (e.g., after attention)
 *
 * BIOLOGICAL BASIS:
 * - Prefrontal cortex (PFC) maintains ~7 items in active state
 * - Exponential decay without rehearsal (tau approx 1-2 seconds)
 * - Items untouched decay to threshold, not zero (MIN_DECAY_EXPONENT clamp)
 *
 * @version 1.0
 * @author NIMCP Development Team - Phase 2.3 GPU Integration
 * @date 2025
 */

#ifdef NIMCP_ENABLE_CUDA

#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>

// Our headers (after CUDA headers to avoid extern "C" issues)
#include "gpu/context/nimcp_gpu_context.h"
#include "utils/logging/nimcp_logging.h"

#define LOG_MODULE "WM_GPU"

//=============================================================================
// CUDA Error Handling
//=============================================================================

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        LOG_ERROR("CUDA error: %s", cudaGetErrorString(err)); \
        return -1; \
    } \
} while(0)

#define CUDA_CHECK_BOOL(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        LOG_ERROR("CUDA error: %s", cudaGetErrorString(err)); \
        return false; \
    } \
} while(0)

// Block size - optimized for small workloads (32-256 items)
#define WM_BLOCK_SIZE 32

//=============================================================================
// Working Memory Decay Kernel
//=============================================================================

/**
 * @brief GPU kernel for exponential decay of working memory salience
 *
 * WHAT: Apply time-based exponential decay to salience array
 * WHY:  Model biological forgetting curve (without attention refresh)
 * HOW:  Each thread handles one item: salience[i] *= exp(-elapsed / tau)
 *
 * FORMULA: salience_new = salience_old * exp(-elapsed_ms / decay_tau_ms)
 *
 * NUMERICAL STABILITY:
 * - Exponent clamped to MIN_DECAY_EXPONENT (-80) to prevent underflow
 * - Items decay to threshold, not zero (biological plausibility)
 *
 * @param salience      [in/out] Salience values for each item
 * @param timestamps    [in]     Last access time for each item (ms)
 * @param current_time  [in]     Current time in milliseconds
 * @param decay_tau_ms  [in]     Decay time constant (default: 1000ms)
 * @param min_salience  [in]     Minimum salience threshold for eviction check
 * @param num_items     [in]     Number of items in working memory
 */
__global__ void kernel_working_memory_decay(
    float* salience,
    const uint64_t* timestamps,
    uint64_t current_time,
    float decay_tau_ms,
    float min_salience,
    uint32_t num_items
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_items) return;

    // Calculate elapsed time since last access
    float elapsed = (float)(current_time - timestamps[idx]);

    // Compute decay exponent with clamp to prevent underflow
    // MIN_DECAY_EXPONENT = -80 ensures meaningful decay factor (~10^-35)
    float exponent = -elapsed / decay_tau_ms;
    float clamped_exponent = fmaxf(exponent, -80.0f);

    // Apply exponential decay
    float decay = expf(clamped_exponent);
    salience[idx] *= decay;
}

/**
 * @brief Extended decay kernel with attention refresh handling
 *
 * WHAT: Apply decay only to non-refreshed items, then clear refresh flags
 * WHY:  Attention refresh (rehearsal) prevents decay (biological)
 * HOW:  Skip refreshed items, apply decay to others, clear flags
 *
 * @param salience           [in/out] Salience values
 * @param timestamps         [in]     Last access timestamps
 * @param attention_refreshed [in/out] Refresh flags (cleared after use)
 * @param current_time       [in]     Current time in milliseconds
 * @param decay_tau_ms       [in]     Decay time constant
 * @param min_salience       [in]     Minimum salience threshold
 * @param num_items          [in]     Number of items
 */
__global__ void kernel_working_memory_decay_with_attention(
    float* salience,
    const uint64_t* timestamps,
    bool* attention_refreshed,
    uint64_t current_time,
    float decay_tau_ms,
    float min_salience,
    uint32_t num_items
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_items) return;

    // Check if item was attention-refreshed (skip decay)
    if (attention_refreshed[idx]) {
        attention_refreshed[idx] = false;  // Clear refresh flag
        return;  // No decay for refreshed items
    }

    // Calculate elapsed time and decay
    float elapsed = (float)(current_time - timestamps[idx]);
    float exponent = fmaxf(-elapsed / decay_tau_ms, -80.0f);
    float decay = expf(exponent);
    salience[idx] *= decay;
}

/**
 * @brief Kernel to count items below threshold (for eviction counting)
 *
 * WHAT: Count how many items have salience below threshold
 * WHY:  Determine how many items will be evicted due to decay
 * HOW:  Parallel reduction with atomic add
 *
 * @param salience      [in]  Salience values
 * @param min_salience  [in]  Minimum threshold
 * @param num_items     [in]  Number of items
 * @param below_count   [out] Count of items below threshold
 */
__global__ void kernel_count_below_threshold(
    const float* salience,
    float min_salience,
    uint32_t num_items,
    uint32_t* below_count
) {
    __shared__ uint32_t shared_count;
    if (threadIdx.x == 0) shared_count = 0;
    __syncthreads();

    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_items && salience[idx] < min_salience) {
        atomicAdd(&shared_count, 1);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        atomicAdd(below_count, shared_count);
    }
}

//=============================================================================
// C-Callable Launch Wrappers
//=============================================================================

extern "C" {

/**
 * @brief Launch working memory decay kernel on GPU
 *
 * WHAT: C-callable wrapper for GPU decay kernel
 * WHY:  Enable GPU acceleration from C code
 * HOW:  Upload data, launch kernel, synchronize
 *
 * MEMORY STRATEGY:
 * For small working memory (7-32 items), we use synchronous transfers
 * for simplicity. For integration with larger GPU pipelines, data can
 * be kept on device across operations.
 *
 * @param salience       Host salience array (modified in-place)
 * @param timestamps     Host timestamp array
 * @param current_time   Current time in milliseconds
 * @param decay_tau_ms   Decay time constant
 * @param min_salience   Minimum salience threshold
 * @param num_items      Number of items in working memory
 * @return 0 on success, -1 on error
 */
int nimcp_gpu_working_memory_decay(
    float* salience,
    const uint64_t* timestamps,
    uint64_t current_time,
    float decay_tau_ms,
    float min_salience,
    uint32_t num_items
) {
    if (!salience || !timestamps || num_items == 0) {
        LOG_ERROR("Invalid parameters for GPU working memory decay");
        return -1;
    }

    // For very small arrays, GPU overhead may exceed benefit
    // But we proceed for API consistency and batching potential
    if (num_items < 4) {
        LOG_DEBUG("WM decay: small array (%u items), GPU overhead may exceed benefit", num_items);
    }

    // Allocate device memory
    float* d_salience = NULL;
    uint64_t* d_timestamps = NULL;
    size_t salience_size = num_items * sizeof(float);
    size_t timestamps_size = num_items * sizeof(uint64_t);

    CUDA_CHECK(cudaMalloc(&d_salience, salience_size));
    CUDA_CHECK(cudaMalloc(&d_timestamps, timestamps_size));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_salience, salience, salience_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_timestamps, timestamps, timestamps_size, cudaMemcpyHostToDevice));

    // Launch kernel
    uint32_t grid_size = (num_items + WM_BLOCK_SIZE - 1) / WM_BLOCK_SIZE;
    kernel_working_memory_decay<<<grid_size, WM_BLOCK_SIZE>>>(
        d_salience, d_timestamps, current_time, decay_tau_ms, min_salience, num_items
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        LOG_ERROR("Kernel launch failed: %s", cudaGetErrorString(err));
        cudaFree(d_salience);
        cudaFree(d_timestamps);
        return -1;
    }

    // Copy result back
    CUDA_CHECK(cudaMemcpy(salience, d_salience, salience_size, cudaMemcpyDeviceToHost));

    // Cleanup
    cudaFree(d_salience);
    cudaFree(d_timestamps);

    LOG_DEBUG("GPU working memory decay completed: %u items", num_items);
    return 0;
}

/**
 * @brief Launch decay kernel with attention refresh handling
 *
 * @param salience            Host salience array (modified)
 * @param timestamps          Host timestamp array
 * @param attention_refreshed Host attention refresh flags (modified)
 * @param current_time        Current time in milliseconds
 * @param decay_tau_ms        Decay time constant
 * @param min_salience        Minimum salience threshold
 * @param num_items           Number of items
 * @return 0 on success, -1 on error
 */
int nimcp_gpu_working_memory_decay_with_attention(
    float* salience,
    const uint64_t* timestamps,
    bool* attention_refreshed,
    uint64_t current_time,
    float decay_tau_ms,
    float min_salience,
    uint32_t num_items
) {
    if (!salience || !timestamps || !attention_refreshed || num_items == 0) {
        LOG_ERROR("Invalid parameters for GPU working memory decay with attention");
        return -1;
    }

    // Allocate device memory
    float* d_salience = NULL;
    uint64_t* d_timestamps = NULL;
    bool* d_attention = NULL;
    size_t salience_size = num_items * sizeof(float);
    size_t timestamps_size = num_items * sizeof(uint64_t);
    size_t attention_size = num_items * sizeof(bool);

    CUDA_CHECK(cudaMalloc(&d_salience, salience_size));
    CUDA_CHECK(cudaMalloc(&d_timestamps, timestamps_size));
    CUDA_CHECK(cudaMalloc(&d_attention, attention_size));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_salience, salience, salience_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_timestamps, timestamps, timestamps_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_attention, attention_refreshed, attention_size, cudaMemcpyHostToDevice));

    // Launch kernel
    uint32_t grid_size = (num_items + WM_BLOCK_SIZE - 1) / WM_BLOCK_SIZE;
    kernel_working_memory_decay_with_attention<<<grid_size, WM_BLOCK_SIZE>>>(
        d_salience, d_timestamps, d_attention, current_time, decay_tau_ms, min_salience, num_items
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        LOG_ERROR("Kernel launch failed: %s", cudaGetErrorString(err));
        cudaFree(d_salience);
        cudaFree(d_timestamps);
        cudaFree(d_attention);
        return -1;
    }

    // Copy results back
    CUDA_CHECK(cudaMemcpy(salience, d_salience, salience_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(attention_refreshed, d_attention, attention_size, cudaMemcpyDeviceToHost));

    // Cleanup
    cudaFree(d_salience);
    cudaFree(d_timestamps);
    cudaFree(d_attention);

    return 0;
}

/**
 * @brief Count items below threshold on GPU
 *
 * @param salience     Host salience array
 * @param min_salience Minimum salience threshold
 * @param num_items    Number of items
 * @param count        Output: count of items below threshold
 * @return 0 on success, -1 on error
 */
int nimcp_gpu_working_memory_count_below_threshold(
    const float* salience,
    float min_salience,
    uint32_t num_items,
    uint32_t* count
) {
    if (!salience || !count || num_items == 0) {
        return -1;
    }

    float* d_salience = NULL;
    uint32_t* d_count = NULL;

    CUDA_CHECK(cudaMalloc(&d_salience, num_items * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_count, sizeof(uint32_t)));

    CUDA_CHECK(cudaMemcpy(d_salience, salience, num_items * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_count, 0, sizeof(uint32_t)));

    uint32_t grid_size = (num_items + WM_BLOCK_SIZE - 1) / WM_BLOCK_SIZE;
    kernel_count_below_threshold<<<grid_size, WM_BLOCK_SIZE>>>(
        d_salience, min_salience, num_items, d_count
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(count, d_count, sizeof(uint32_t), cudaMemcpyDeviceToHost));

    cudaFree(d_salience);
    cudaFree(d_count);

    return 0;
}

/**
 * @brief Check if GPU working memory acceleration is available
 *
 * @return true if CUDA is available, false otherwise
 */
bool nimcp_gpu_working_memory_available(void) {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0);
}

} // extern "C"

#else // !NIMCP_ENABLE_CUDA

//=============================================================================
// CPU Fallback Stubs (when CUDA not available)
//=============================================================================

#include <stdint.h>
#include <stdbool.h>
#include <math.h>

#include "utils/logging/nimcp_logging.h"

#define LOG_MODULE "WM_GPU"

extern "C" {

int nimcp_gpu_working_memory_decay(
    float* salience,
    const uint64_t* timestamps,
    uint64_t current_time,
    float decay_tau_ms,
    float min_salience,
    uint32_t num_items
) {
    // CPU fallback implementation
    if (!salience || !timestamps || num_items == 0) {
        return -1;
    }

    for (uint32_t i = 0; i < num_items; i++) {
        float elapsed = (float)(current_time - timestamps[i]);
        float exponent = -elapsed / decay_tau_ms;
        if (exponent < -80.0f) exponent = -80.0f;
        float decay = expf(exponent);
        salience[i] *= decay;
    }

    LOG_DEBUG("CPU fallback: working memory decay for %u items", num_items);
    return 0;
}

int nimcp_gpu_working_memory_decay_with_attention(
    float* salience,
    const uint64_t* timestamps,
    bool* attention_refreshed,
    uint64_t current_time,
    float decay_tau_ms,
    float min_salience,
    uint32_t num_items
) {
    if (!salience || !timestamps || !attention_refreshed || num_items == 0) {
        return -1;
    }

    for (uint32_t i = 0; i < num_items; i++) {
        if (attention_refreshed[i]) {
            attention_refreshed[i] = false;
            continue;
        }
        float elapsed = (float)(current_time - timestamps[i]);
        float exponent = -elapsed / decay_tau_ms;
        if (exponent < -80.0f) exponent = -80.0f;
        float decay = expf(exponent);
        salience[i] *= decay;
    }

    return 0;
}

int nimcp_gpu_working_memory_count_below_threshold(
    const float* salience,
    float min_salience,
    uint32_t num_items,
    uint32_t* count
) {
    if (!salience || !count || num_items == 0) {
        return -1;
    }

    *count = 0;
    for (uint32_t i = 0; i < num_items; i++) {
        if (salience[i] < min_salience) {
            (*count)++;
        }
    }
    return 0;
}

bool nimcp_gpu_working_memory_available(void) {
    return false;  // CUDA not available
}

} // extern "C"

#endif // NIMCP_ENABLE_CUDA
