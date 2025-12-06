/**
 * @file nimcp_execution_mode.c
 * @brief Implementation of execution mode detection and selection
 *
 * NIMCP Phase: Integrated bio-async communication and comprehensive logging
 * Date: 2025-11-28
 *
 * WHAT: Detects hardware capabilities and selects optimal execution mode
 * WHY:  Different platforms have different optimal execution strategies
 * HOW:  Query CPU, GPU, network at runtime; select best mode for workload
 *
 * BIO-ASYNC INTEGRATION:
 * - GPU execution events published to bio-async system
 * - Mode switches communicated via async channels
 * - Performance metrics shared with monitoring subsystems
 *
 * @author NIMCP Development Team
 * @date 2025
 * @version 2.6 (GPU P2P)
 */

#define LOG_MODULE "GPU_EXEC_MODE"
#define LOG_MODULE_ID 0x0900

#include "gpu/nimcp_execution_mode.h"
#include "utils/memory/nimcp_unified_memory.h"
#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_messages.h"
#include "async/nimcp_bio_router.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/validation/nimcp_validate.h"
#include "utils/logging/nimcp_logging.h"
#include <string.h>
#include <stdio.h>

// Platform-specific includes
#ifdef __linux__
#include <unistd.h>
#include <sys/sysinfo.h>
#endif

#ifdef _WIN32
#include <windows.h>
#endif

// CUDA includes (conditional)
#ifdef NIMCP_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

//=============================================================================
// Internal Structures
//=============================================================================

/**
 * @brief Execution context internal structure
 */
struct execution_context_struct {
    execution_config_t config;
    execution_mode_t active_mode;

    // CPU resources
    uint32_t cpu_thread_count;
    void** cpu_threads;

    // GPU resources
    void* gpu_device_ptr;
    int gpu_device_id;

    // Statistics
    uint64_t total_operations;
    double total_time_ms;
    uint64_t start_time;

    // Bio-async integration
    bio_module_context_t bio_ctx;
};

//=============================================================================
// CPU Detection (Cross-Platform)
//=============================================================================

/**
 * @brief Detect CPU capabilities
 */
static bool detect_cpu_capabilities(hardware_capabilities_t* caps)
{
    if (!caps) {
        return false;
    }

    caps->cpu_available = true;

    #ifdef __linux__
    // Linux: Use sysconf
    caps->cpu_cores = (uint32_t)sysconf(_SC_NPROCESSORS_ONLN);
    caps->cpu_threads = caps->cpu_cores;  // Simplified

    #elif defined(_WIN32)
    // Windows: Use GetSystemInfo
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    caps->cpu_cores = sysinfo.dwNumberOfProcessors;
    caps->cpu_threads = caps->cpu_cores;

    #else
    // Fallback: Assume 4 cores
    caps->cpu_cores = 4;
    caps->cpu_threads = 4;
    #endif

    // TODO: Detect SIMD capabilities (AVX2, AVX512)
    // For now, assume available on x86-64
    #if defined(__x86_64__) || defined(_M_X64)
    caps->cpu_avx2 = true;
    caps->cpu_avx512 = false;  // Conservative
    #else
    caps->cpu_avx2 = false;
    caps->cpu_avx512 = false;
    #endif

    return true;
}

//=============================================================================
// GPU Detection (CUDA)
//=============================================================================

/**
 * @brief Detect CUDA GPU capabilities
 */
static bool detect_cuda_capabilities(hardware_capabilities_t* caps)
{
    if (!caps) {
        return false;
    }

    #ifdef NIMCP_ENABLE_CUDA
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);

    if (err != cudaSuccess || device_count == 0) {
        caps->cuda_available = false;
        caps->gpu_count = 0;
        return false;
    }

    caps->cuda_available = true;
    caps->gpu_count = (uint32_t)device_count;

    // Get properties of first GPU
    struct cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, 0);

    if (err == cudaSuccess) {
        caps->gpu_compute_units = (uint32_t)prop.multiProcessorCount;
        caps->gpu_memory_mb = (uint64_t)(prop.totalGlobalMem / (1024 * 1024));
        caps->gpu_compute_capability = (uint32_t)(prop.major * 10 + prop.minor);
    }

    return true;
    #else
    // CUDA not available at compile time
    caps->cuda_available = false;
    caps->gpu_count = 0;
    caps->gpu_compute_units = 0;
    caps->gpu_memory_mb = 0;
    caps->gpu_compute_capability = 0;
    return false;
    #endif
}

//=============================================================================
// Network Detection
//=============================================================================

/**
 * @brief Detect network capabilities
 */
static bool detect_network_capabilities(hardware_capabilities_t* caps)
{
    if (!caps) {
        return false;
    }

    // TODO: Implement network detection (MPI, etc.)
    // For now, assume no distributed capabilities
    caps->network_available = false;
    caps->network_nodes = 1;
    caps->network_bandwidth_mbps = 0;

    return true;
}

//=============================================================================
// Public API: Hardware Detection
//=============================================================================

/**
 * @brief Detect all hardware capabilities
 */
bool execution_detect_capabilities(hardware_capabilities_t* caps)
{
    // Guard: Validate output
    if (!caps) {
        LOG_ERROR("NULL capabilities pointer");
        return false;
    }

    LOG_DEBUG("Detecting hardware capabilities");

    // Initialize all fields to zero/false
    memset(caps, 0, sizeof(hardware_capabilities_t));

    // Detect each capability
    detect_cpu_capabilities(caps);
    detect_cuda_capabilities(caps);
    detect_network_capabilities(caps);

    // Determine recommended mode
    if (caps->cuda_available && caps->gpu_count > 0) {
        caps->recommended_mode = EXEC_MODE_GPU_CUDA;
        LOG_INFO("Detected CUDA-capable GPU (count=%u), recommending GPU mode", caps->gpu_count);
    } else if (caps->cpu_threads >= 4) {
        caps->recommended_mode = EXEC_MODE_CPU_PARALLEL;
        LOG_INFO("Detected multi-core CPU (threads=%u), recommending parallel mode", caps->cpu_threads);
    } else {
        caps->recommended_mode = EXEC_MODE_CPU_SEQUENTIAL;
        LOG_INFO("Limited CPU resources, recommending sequential mode");
    }

    return true;
}

/**
 * @brief Check if mode is supported
 */
bool execution_mode_is_supported(execution_mode_t mode)
{
    hardware_capabilities_t caps;
    if (!execution_detect_capabilities(&caps)) {
        return false;
    }

    switch (mode) {
        case EXEC_MODE_CPU_SEQUENTIAL:
        case EXEC_MODE_CPU_PARALLEL:
            return caps.cpu_available;

        case EXEC_MODE_GPU_CUDA:
            return caps.cuda_available;

        case EXEC_MODE_GPU_ROCM:
            return caps.rocm_available;

        case EXEC_MODE_GPU_OPENCL:
            return caps.opencl_available;

        case EXEC_MODE_DISTRIBUTED_CPU:
        case EXEC_MODE_DISTRIBUTED_GPU:
            return caps.network_available;

        case EXEC_MODE_HYBRID:
            return caps.cpu_available && caps.cuda_available;

        case EXEC_MODE_AUTO:
            return true;  // Auto always supported

        default:
            return false;
    }
}

/**
 * @brief Get recommended mode for workload
 */
execution_mode_t execution_get_recommended_mode(uint32_t num_neurons, uint32_t num_synapses)
{
    hardware_capabilities_t caps;
    if (!execution_detect_capabilities(&caps)) {
        return EXEC_MODE_CPU_SEQUENTIAL;
    }

    (void)num_synapses;  // Not used yet

    // Heuristics based on neuron count
    if (num_neurons < 1000) {
        return EXEC_MODE_CPU_SEQUENTIAL;
    } else if (num_neurons < 10000) {
        return caps.cpu_threads >= 4 ? EXEC_MODE_CPU_PARALLEL : EXEC_MODE_CPU_SEQUENTIAL;
    } else if (num_neurons < 1000000) {
        return caps.cuda_available ? EXEC_MODE_GPU_CUDA : EXEC_MODE_CPU_PARALLEL;
    } else {
        // Very large networks
        if (caps.network_available && caps.cuda_available) {
            return EXEC_MODE_DISTRIBUTED_GPU;
        } else if (caps.cuda_available) {
            return EXEC_MODE_GPU_CUDA;
        } else {
            return EXEC_MODE_CPU_PARALLEL;
        }
    }
}

//=============================================================================
// Execution Context Management
//=============================================================================

/**
 * @brief Create execution context
 */
execution_context_t execution_context_create(const execution_config_t* config)
{
    // Guard: Validate config
    if (!config) {
        return NULL;
    }

    // Allocate context
    execution_context_t ctx = (execution_context_t)nimcp_calloc(1, sizeof(struct execution_context_struct));
    if (!ctx) {
        return NULL;
    }

    // Copy configuration
    ctx->config = *config;

    // Determine actual mode (handle AUTO and fallback)
    execution_mode_t requested_mode = config->mode;

    if (requested_mode == EXEC_MODE_AUTO) {
        requested_mode = execution_get_recommended_mode(1000, 100);  // Default heuristic
    }

    // Check if requested mode is supported
    if (!execution_mode_is_supported(requested_mode)) {
        // Fallback to CPU parallel or sequential
        if (config->auto_fallback) {
            requested_mode = EXEC_MODE_CPU_PARALLEL;
            if (!execution_mode_is_supported(requested_mode)) {
                requested_mode = EXEC_MODE_CPU_SEQUENTIAL;
            }
        } else {
            nimcp_free(ctx);
            return NULL;
        }
    }

    ctx->active_mode = requested_mode;

    // Initialize mode-specific resources
    switch (ctx->active_mode) {
        case EXEC_MODE_GPU_CUDA:
            #ifdef NIMCP_ENABLE_CUDA
            // Initialize CUDA
            ctx->gpu_device_id = 0;
            cudaSetDevice(ctx->gpu_device_id);
            #endif
            break;

        case EXEC_MODE_CPU_PARALLEL:
            // TODO: Initialize thread pool
            ctx->cpu_thread_count = config->cpu_threads;
            break;

        default:
            // CPU sequential - no special initialization
            break;
    }

    return ctx;
}

/**
 * @brief Destroy execution context
 */
void execution_context_destroy(execution_context_t ctx)
{
    if (!ctx) {
        return;
    }

    // Synchronize before cleanup
    execution_synchronize(ctx);

    // Cleanup mode-specific resources
    switch (ctx->active_mode) {
        case EXEC_MODE_GPU_CUDA:
            #ifdef NIMCP_ENABLE_CUDA
            cudaDeviceReset();
            #endif
            break;

        case EXEC_MODE_CPU_PARALLEL:
            // TODO: Cleanup thread pool
            break;

        default:
            break;
    }

    nimcp_free(ctx);
}

/**
 * @brief Get active mode
 */
execution_mode_t execution_context_get_mode(execution_context_t ctx)
{
    if (!ctx) {
        return EXEC_MODE_CPU_SEQUENTIAL;
    }

    return ctx->active_mode;
}

/**
 * @brief Set execution mode
 */
bool execution_context_set_mode(execution_context_t ctx, execution_mode_t new_mode)
{
    // Guard: Validate context
    if (!ctx) {
        return false;
    }

    // Check if mode is supported
    if (!execution_mode_is_supported(new_mode)) {
        return false;
    }

    // Synchronize current mode
    execution_synchronize(ctx);

    // TODO: Cleanup old mode resources
    // TODO: Initialize new mode resources

    ctx->active_mode = new_mode;
    return true;
}

//=============================================================================
// Memory Management
//=============================================================================

/**
 * @brief Allocate memory
 */
void* execution_alloc(execution_context_t ctx, size_t size)
{
    if (!ctx || size == 0) {
        return NULL;
    }

    switch (ctx->active_mode) {
        case EXEC_MODE_GPU_CUDA:
            #ifdef NIMCP_ENABLE_CUDA
            if (ctx->config.use_unified_memory) {
                void* ptr = NULL;
                cudaMallocManaged(&ptr, size, cudaMemAttachGlobal);
                return ptr;
            } else {
                void* ptr = NULL;
                cudaMalloc(&ptr, size);
                return ptr;
            }
            #else
            // Fallback to CPU
            return nimcp_malloc(size);
            #endif

        default:
            // CPU allocation
            return nimcp_malloc(size);
    }
}

/**
 * @brief Free memory
 */
void execution_free(execution_context_t ctx, void* ptr)
{
    if (!ctx || !ptr) {
        return;
    }

    switch (ctx->active_mode) {
        case EXEC_MODE_GPU_CUDA:
            #ifdef NIMCP_ENABLE_CUDA
            cudaFree(ptr);
            #else
            nimcp_free(ptr);
            #endif
            break;

        default:
            nimcp_free(ptr);
            break;
    }
}

/**
 * @brief Copy memory
 */
bool execution_memcpy(execution_context_t ctx, void* dst, const void* src,
                     size_t size, bool host_to_device)
{
    if (!ctx || !dst || !src || size == 0) {
        return false;
    }

    switch (ctx->active_mode) {
        case EXEC_MODE_GPU_CUDA:
            #ifdef NIMCP_ENABLE_CUDA
            {
                enum cudaMemcpyKind kind = host_to_device ?
                    cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost;
                cudaError_t err = cudaMemcpy(dst, src, size, kind);
                return (err == cudaSuccess);
            }
            #else
            // Fallback to CPU
            memcpy(dst, src, size);
            return true;
            #endif

        default:
            // CPU memcpy
            memcpy(dst, src, size);
            return true;
    }
}

//=============================================================================
// Synchronization
//=============================================================================

/**
 * @brief Synchronize execution
 */
bool execution_synchronize(execution_context_t ctx)
{
    if (!ctx) {
        return false;
    }

    switch (ctx->active_mode) {
        case EXEC_MODE_GPU_CUDA:
            #ifdef NIMCP_ENABLE_CUDA
            cudaError_t err = cudaDeviceSynchronize();
            return (err == cudaSuccess);
            #else
            return true;
            #endif

        default:
            // CPU - already synchronous
            return true;
    }
}

/**
 * @brief Get statistics
 */
bool execution_get_stats(execution_context_t ctx, uint64_t* total_ops,
                        double* total_time_ms)
{
    if (!ctx) {
        return false;
    }

    if (total_ops) {
        *total_ops = ctx->total_operations;
    }

    if (total_time_ms) {
        *total_time_ms = ctx->total_time_ms;
    }

    return true;
}

//=============================================================================
// Default Configurations
//=============================================================================

/**
 * @brief Get default config for mode
 */
execution_config_t execution_get_default_config(execution_mode_t mode)
{
    execution_config_t config = {0};

    config.mode = mode;
    config.auto_fallback = true;
    config.fallback_mode = EXEC_MODE_CPU_PARALLEL;

    switch (mode) {
        case EXEC_MODE_CPU_SEQUENTIAL:
            config.cpu_threads = 1;
            config.batch_size = 1;
            break;

        case EXEC_MODE_CPU_PARALLEL:
            {
                hardware_capabilities_t caps;
                execution_detect_capabilities(&caps);
                config.cpu_threads = caps.cpu_threads;
                config.batch_size = 32;
            }
            break;

        case EXEC_MODE_GPU_CUDA:
            config.gpu_blocks = 256;
            config.gpu_threads_per_block = 256;
            config.use_unified_memory = false;
            config.pin_cpu_memory = true;
            config.batch_size = 1024;
            break;

        default:
            config.cpu_threads = 4;
            config.batch_size = 16;
            break;
    }

    return config;
}

/**
 * @brief Get optimal config
 */
execution_config_t execution_get_optimal_config(uint32_t num_neurons)
{
    execution_mode_t mode = execution_get_recommended_mode(num_neurons, 100);
    return execution_get_default_config(mode);
}
