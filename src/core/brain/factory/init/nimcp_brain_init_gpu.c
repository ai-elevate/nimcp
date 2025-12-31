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

#include <string.h>

#define LOG_MODULE "BRAIN_INIT_GPU"

/**
 * @brief Initialize GPU subsystem for brain
 *
 * WHAT: Creates GPU context if GPU is available
 * WHY:  Enables automatic GPU acceleration for brain operations
 * HOW:  Uses gpu_is_available() to detect, creates context if present
 *
 * @param brain Brain instance to initialize GPU for
 * @return true on success (including when no GPU is available), false on error
 *
 * COMPLEXITY: O(1) - GPU detection and context creation
 *
 * THREAD SAFETY: NOT thread-safe (call during brain creation only)
 *
 * NOTE: Success is returned even if no GPU is available - the brain
 * will simply operate in CPU-only mode. This is not an error.
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

    // Check if GPU is available
    if (!gpu_is_available()) {
        LOG_INFO("No GPU available - brain will use CPU-only execution");
        return true;  // Success - CPU-only mode is valid
    }

    // Check user config for GPU preference
    // (Allow users to disable GPU even if available)
    if (brain->config.disable_gpu) {
        LOG_INFO("GPU disabled by configuration - using CPU-only execution");
        return true;
    }

    // Create GPU context with auto-device selection
    brain->gpu_ctx = nimcp_gpu_context_create_auto();
    if (!brain->gpu_ctx) {
        // GPU detection said available, but context creation failed
        // This can happen if driver is outdated or GPU is busy
        LOG_WARN("GPU detected but context creation failed - falling back to CPU");
        return true;  // Still success - CPU fallback is valid
    }

    // GPU context created successfully
    brain->gpu_enabled = true;
    brain->last_gpu_sync_us = 0;

    // Log GPU info
    char info_buffer[256];
    nimcp_gpu_context_get_info_string(brain->gpu_ctx, info_buffer, sizeof(info_buffer));
    LOG_INFO("GPU acceleration enabled: %s", info_buffer);

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
