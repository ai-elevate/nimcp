/**
 * @file nimcp_brain_init_gpu.c
 * @brief GPU Context Initialization for Brain Factory
 *
 * WHAT: Initializes GPU context during brain creation
 * WHY:  Enables automatic GPU acceleration for brains
 * HOW:  Detects GPU availability and creates context if present
 *
 * ARCHITECTURE:
 * - Auto-detection: Checks if CUDA GPU is available
 * - Auto-selection: Chooses best GPU device automatically
 * - Graceful fallback: Works without GPU (CPU-only mode)
 *
 * @author NIMCP Development Team
 * @date 2025
 * @version 1.0
 */

#include "core/brain/factory/init/nimcp_brain_init.h"
#include "core/brain/nimcp_brain_internal.h"
#include "gpu/context/nimcp_gpu_context.h"
#include "gpu/execution/nimcp_gpu_detect.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/exception/nimcp_exception_macros.h"

#include <string.h>

#define LOG_MODULE "BRAIN_INIT_GPU"

/**
 * @brief Initialize GPU subsystem for brain
 *
 * WHAT: Creates GPU context with GPU-first default policy
 * WHY:  Phase 1 GPU Integration - GPU is now the default backend
 * HOW:  Always try GPU first unless force_cpu_only is set
 *
 * @param brain Brain instance to initialize GPU for
 * @return true on success (including CPU fallback), false on error
 *
 * COMPLEXITY: O(1) - GPU detection and context creation
 *
 * THREAD SAFETY: NOT thread-safe (call during brain creation only)
 *
 * GPU-FIRST POLICY (Phase 1 GPU Integration):
 * - GPU is the DEFAULT backend, not an optional enhancement
 * - Only skips GPU if force_cpu_only=true OR GPU unavailable
 * - Logs warning when falling back to CPU to encourage GPU setup
 *
 * BACKWARD COMPATIBILITY:
 * - disable_gpu is still checked for legacy code
 * - force_cpu_only takes precedence if both are set
 */
bool nimcp_brain_factory_init_gpu_subsystem(brain_t brain)
{
    if (!brain) {
        LOG_ERROR("NULL brain provided to GPU init");
        return false;
    }

    // Initialize GPU fields to safe defaults
    brain->gpu_ctx = NULL;
    brain->gpu_enabled = false;
    brain->last_gpu_sync_us = 0;

    // Check if user explicitly requested CPU-only mode
    // force_cpu_only takes precedence over disable_gpu
    bool skip_gpu = brain->config.force_cpu_only || brain->config.disable_gpu;

    if (skip_gpu) {
        LOG_INFO("GPU disabled by configuration (force_cpu_only=%s, disable_gpu=%s) - using CPU-only execution",
                 brain->config.force_cpu_only ? "true" : "false",
                 brain->config.disable_gpu ? "true" : "false");
        return true;
    }

    // GPU-FIRST POLICY: Always try to use GPU unless explicitly disabled
    LOG_INFO("Initializing GPU subsystem (GPU-first policy)...");

    // Check if any GPU is available
    if (!gpu_is_available()) {
        LOG_WARN("No GPU available - falling back to CPU-only execution");
        LOG_WARN("For optimal performance, install CUDA, ROCm, or OpenCL drivers");
        return true;  // Success - CPU fallback is valid
    }

    // Create GPU context with auto-device selection
    brain->gpu_ctx = nimcp_gpu_context_create_auto();
    if (!brain->gpu_ctx) {
        // GPU detection said available, but context creation failed
        // This can happen if driver is outdated or GPU is busy
        LOG_WARN("GPU detected but context creation failed - falling back to CPU");
        LOG_WARN("Check GPU driver version and availability");
        return true;  // Still success - CPU fallback is valid
    }

    // GPU context created successfully
    brain->gpu_enabled = true;
    brain->last_gpu_sync_us = 0;

    // Log GPU info
    char info_buffer[256];
    nimcp_gpu_context_get_info_string(brain->gpu_ctx, info_buffer, sizeof(info_buffer));
    LOG_INFO("GPU acceleration enabled (GPU-first policy): %s", info_buffer);

    return true;
}

/**
 * @brief Destroy GPU subsystem for brain
 *
 * WHAT: Cleans up GPU context and resources
 * WHY:  Proper resource cleanup during brain destruction
 * HOW:  Destroys GPU context if it exists
 *
 * @param brain Brain instance to cleanup GPU for
 *
 * COMPLEXITY: O(1) - GPU context destruction
 *
 * THREAD SAFETY: NOT thread-safe (call during brain destruction only)
 */
void nimcp_brain_factory_destroy_gpu_subsystem(brain_t brain)
{
    if (!brain) {
        return;
    }

    if (brain->gpu_ctx) {
        // Synchronize before destruction to ensure all GPU ops complete
        nimcp_gpu_context_synchronize(brain->gpu_ctx);

        // Get final memory stats for logging
        size_t allocated = 0, peak = 0;
        nimcp_gpu_memory_stats(brain->gpu_ctx, &allocated, &peak, NULL);

        if (allocated > 0) {
            LOG_WARN("GPU memory leak detected: %zu bytes still allocated", allocated);
        }

        LOG_INFO("Destroying GPU context (peak memory: %zu MB)",
                 peak / (1024 * 1024));

        nimcp_gpu_context_destroy(brain->gpu_ctx);
        brain->gpu_ctx = NULL;
    }

    brain->gpu_enabled = false;
    brain->last_gpu_sync_us = 0;
}
