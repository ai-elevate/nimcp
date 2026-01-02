/**
 * @file nimcp_cuda_utils.h
 * @brief Standardized CUDA Error Handling and Utilities
 *
 * WHAT: Common CUDA error checking macros and utilities
 * WHY:  Eliminate inconsistent error handling across 60+ GPU kernel files
 * HOW:  Standardized macros with consistent error reporting
 *
 * USAGE:
 *   #include "gpu/common/nimcp_cuda_utils.h"
 *
 *   NIMCP_CUDA_CHECK(cudaMalloc(&ptr, size));
 *   NIMCP_CUDA_CHECK_RETURN(cudaMemcpy(...));
 *
 * CONSOLIDATES DUPLICATES FROM:
 *   - ALL .cu files in src/gpu/
 *
 * @version 1.0
 * @author NIMCP Development Team
 * @date 2025
 */

#ifndef NIMCP_CUDA_UTILS_H
#define NIMCP_CUDA_UTILS_H

#include "common/nimcp_export.h"
#include "common/nimcp_error.h"

#ifdef NIMCP_ENABLE_CUDA
#include <cuda_runtime.h>
#include <stdio.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// Standard Kernel Configuration
//=============================================================================

/** Default CUDA block size (threads per block) */
#define NIMCP_CUDA_BLOCK_SIZE 256

/** Maximum supported block size */
#define NIMCP_CUDA_MAX_BLOCK_SIZE 1024

/** Warp size (fixed on all NVIDIA GPUs) */
#define NIMCP_CUDA_WARP_SIZE 32

/** Calculate grid size for n elements */
#define NIMCP_CUDA_GRID_SIZE(n, block) (((n) + (block) - 1) / (block))

/** Calculate grid size with default block */
#define NIMCP_CUDA_GRID_SIZE_DEFAULT(n) NIMCP_CUDA_GRID_SIZE(n, NIMCP_CUDA_BLOCK_SIZE)

//=============================================================================
// Error Checking Macros
//=============================================================================

#ifdef NIMCP_ENABLE_CUDA

/**
 * @brief Check CUDA call, log error, and return false on failure
 *
 * Use in functions returning bool.
 */
#define NIMCP_CUDA_CHECK(call) do { \
    cudaError_t _err = (call); \
    if (_err != cudaSuccess) { \
        fprintf(stderr, "[NIMCP CUDA ERROR] %s:%d: %s returned %s\n", \
                __FILE__, __LINE__, #call, cudaGetErrorString(_err)); \
        return false; \
    } \
} while(0)

/**
 * @brief Check CUDA call, log error, and return NULL on failure
 *
 * Use in functions returning pointers.
 */
#define NIMCP_CUDA_CHECK_NULL(call) do { \
    cudaError_t _err = (call); \
    if (_err != cudaSuccess) { \
        fprintf(stderr, "[NIMCP CUDA ERROR] %s:%d: %s returned %s\n", \
                __FILE__, __LINE__, #call, cudaGetErrorString(_err)); \
        return NULL; \
    } \
} while(0)

/**
 * @brief Check CUDA call, log error, and return error code on failure
 *
 * Use in functions returning nimcp_error_t.
 */
#define NIMCP_CUDA_CHECK_ERROR(call) do { \
    cudaError_t _err = (call); \
    if (_err != cudaSuccess) { \
        fprintf(stderr, "[NIMCP CUDA ERROR] %s:%d: %s returned %s\n", \
                __FILE__, __LINE__, #call, cudaGetErrorString(_err)); \
        return NIMCP_ERROR_GPU; \
    } \
} while(0)

/**
 * @brief Check CUDA call, log error, and goto cleanup label on failure
 *
 * Use when cleanup is needed before return.
 */
#define NIMCP_CUDA_CHECK_GOTO(call, label) do { \
    cudaError_t _err = (call); \
    if (_err != cudaSuccess) { \
        fprintf(stderr, "[NIMCP CUDA ERROR] %s:%d: %s returned %s\n", \
                __FILE__, __LINE__, #call, cudaGetErrorString(_err)); \
        goto label; \
    } \
} while(0)

/**
 * @brief Check CUDA call and store result, but don't return
 *
 * For non-critical errors or when custom handling is needed.
 */
#define NIMCP_CUDA_CHECK_WARN(call) do { \
    cudaError_t _err = (call); \
    if (_err != cudaSuccess) { \
        fprintf(stderr, "[NIMCP CUDA WARN] %s:%d: %s returned %s\n", \
                __FILE__, __LINE__, #call, cudaGetErrorString(_err)); \
    } \
} while(0)

/**
 * @brief Check last CUDA error (for kernel launches)
 */
#define NIMCP_CUDA_CHECK_LAST() do { \
    cudaError_t _err = cudaGetLastError(); \
    if (_err != cudaSuccess) { \
        fprintf(stderr, "[NIMCP CUDA ERROR] %s:%d: kernel error: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(_err)); \
        return false; \
    } \
} while(0)

/**
 * @brief Check last CUDA error, return NULL
 */
#define NIMCP_CUDA_CHECK_LAST_NULL() do { \
    cudaError_t _err = cudaGetLastError(); \
    if (_err != cudaSuccess) { \
        fprintf(stderr, "[NIMCP CUDA ERROR] %s:%d: kernel error: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(_err)); \
        return NULL; \
    } \
} while(0)

/**
 * @brief Synchronize and check for errors
 */
#define NIMCP_CUDA_SYNC_CHECK() do { \
    NIMCP_CUDA_CHECK(cudaDeviceSynchronize()); \
    NIMCP_CUDA_CHECK_LAST(); \
} while(0)

#else // !NIMCP_ENABLE_CUDA

// No-op macros when CUDA is disabled
#define NIMCP_CUDA_CHECK(call) ((void)0)
#define NIMCP_CUDA_CHECK_NULL(call) ((void)0)
#define NIMCP_CUDA_CHECK_ERROR(call) (NIMCP_ERROR_GPU)
#define NIMCP_CUDA_CHECK_GOTO(call, label) ((void)0)
#define NIMCP_CUDA_CHECK_WARN(call) ((void)0)
#define NIMCP_CUDA_CHECK_LAST() ((void)0)
#define NIMCP_CUDA_CHECK_LAST_NULL() ((void)0)
#define NIMCP_CUDA_SYNC_CHECK() ((void)0)

#endif // NIMCP_ENABLE_CUDA

//=============================================================================
// cuBLAS/cuSPARSE/cuRAND Error Checking
//=============================================================================

#ifdef NIMCP_ENABLE_CUDA

/**
 * @brief Check cuBLAS call
 */
#define NIMCP_CUBLAS_CHECK(call) do { \
    cublasStatus_t _status = (call); \
    if (_status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "[NIMCP cuBLAS ERROR] %s:%d: %s returned %d\n", \
                __FILE__, __LINE__, #call, _status); \
        return false; \
    } \
} while(0)

/**
 * @brief Check cuSPARSE call
 */
#define NIMCP_CUSPARSE_CHECK(call) do { \
    cusparseStatus_t _status = (call); \
    if (_status != CUSPARSE_STATUS_SUCCESS) { \
        fprintf(stderr, "[NIMCP cuSPARSE ERROR] %s:%d: %s returned %d\n", \
                __FILE__, __LINE__, #call, _status); \
        return false; \
    } \
} while(0)

/**
 * @brief Check cuRAND call
 */
#define NIMCP_CURAND_CHECK(call) do { \
    curandStatus_t _status = (call); \
    if (_status != CURAND_STATUS_SUCCESS) { \
        fprintf(stderr, "[NIMCP cuRAND ERROR] %s:%d: %s returned %d\n", \
                __FILE__, __LINE__, #call, _status); \
        return false; \
    } \
} while(0)

#else

#define NIMCP_CUBLAS_CHECK(call) ((void)0)
#define NIMCP_CUSPARSE_CHECK(call) ((void)0)
#define NIMCP_CURAND_CHECK(call) ((void)0)

#endif // NIMCP_ENABLE_CUDA

//=============================================================================
// Kernel Launch Helpers
//=============================================================================

#ifdef NIMCP_ENABLE_CUDA

/**
 * @brief Calculate optimal block size for occupancy
 *
 * @param kernel Kernel function pointer
 * @param dynamic_smem Dynamic shared memory per block
 * @return Recommended block size
 */
static inline int nimcp_cuda_optimal_block_size(void* kernel, size_t dynamic_smem)
{
    int min_grid_size, block_size;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, kernel, dynamic_smem, 0);
    (void)min_grid_size;
    return block_size;
}

/**
 * @brief Get current device compute capability
 *
 * @param major Output: major version
 * @param minor Output: minor version
 * @return true on success
 */
static inline bool nimcp_cuda_get_compute_capability(int* major, int* minor)
{
    int device;
    if (cudaGetDevice(&device) != cudaSuccess) return false;

    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) return false;

    *major = prop.major;
    *minor = prop.minor;
    return true;
}

/**
 * @brief Get available GPU memory
 *
 * @param free_bytes Output: free memory in bytes
 * @param total_bytes Output: total memory in bytes
 * @return true on success
 */
static inline bool nimcp_cuda_get_memory_info(size_t* free_bytes, size_t* total_bytes)
{
    return cudaMemGetInfo(free_bytes, total_bytes) == cudaSuccess;
}

#endif // NIMCP_ENABLE_CUDA

//=============================================================================
// Debug Helpers
//=============================================================================

#ifdef NIMCP_ENABLE_CUDA

/**
 * @brief Print GPU memory usage to stderr
 */
static inline void nimcp_cuda_print_memory_usage(void)
{
    size_t free_mem, total_mem;
    if (cudaMemGetInfo(&free_mem, &total_mem) == cudaSuccess) {
        fprintf(stderr, "[NIMCP GPU Memory] Free: %.2f GB / Total: %.2f GB (%.1f%% used)\n",
                free_mem / (1024.0 * 1024.0 * 1024.0),
                total_mem / (1024.0 * 1024.0 * 1024.0),
                100.0 * (1.0 - (double)free_mem / total_mem));
    }
}

/**
 * @brief Print device info to stderr
 */
static inline void nimcp_cuda_print_device_info(void)
{
    int device;
    if (cudaGetDevice(&device) != cudaSuccess) return;

    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) return;

    fprintf(stderr, "[NIMCP GPU] Device %d: %s (SM %d.%d, %.2f GB, %d SMs)\n",
            device, prop.name, prop.major, prop.minor,
            prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0),
            prop.multiProcessorCount);
}

#else

static inline void nimcp_cuda_print_memory_usage(void) {}
static inline void nimcp_cuda_print_device_info(void) {}

#endif // NIMCP_ENABLE_CUDA

#ifdef __cplusplus
}
#endif

#endif // NIMCP_CUDA_UTILS_H
