/**
 * @file nimcp.c
 * @brief Implementation of unified NIMCP API (refactored)
 *
 * This file wraps the internal APIs and provides a consistent, stable public interface.
 *
 * REFACTORING NOTE (2025-12-08):
 * - Extracted brain core API to nimcp_brain_api.c (lines 184-718)
 * - Extracted snapshot/COW API to nimcp_snapshot_api.c (lines 481-935)
 * - Extracted cognitive APIs to nimcp_cognitive_api.c (lines 936-2014)
 * - Extracted subsystems APIs to nimcp_subsystems_api.c (lines 1386-1697)
 * - Extracted oscillation API to nimcp_oscillation_api.c (lines 1698-1969)
 * - Training API remains here (lines 2015-end) due to complex state management
 */

#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/memory/nimcp_unified_memory.h"
#include "utils/error/nimcp_error_codes.h"
#include "api/nimcp_api_exception.h"

#define LOG_MODULE "API"

#include <stddef.h>  /* for NULL */
//=============================================================================
// Health Agent Integration (Phase 8: System-Wide Health Integration)
//=============================================================================
struct nimcp_health_agent;
typedef struct nimcp_health_agent nimcp_health_agent_t;
extern void nimcp_health_agent_heartbeat_ex(nimcp_health_agent_t* agent,
                                             const char* operation,
                                             float progress);

/** Global health agent for refactored module */
static nimcp_health_agent_t* g_refactored_health_agent = NULL;

/**
 * @brief Set health agent for refactored heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
static void refactored_set_health_agent(nimcp_health_agent_t* agent) {
    g_refactored_health_agent = agent;
}

/** @brief Send heartbeat from refactored module */
static inline void refactored_heartbeat(const char* operation, float progress) {
    if (g_refactored_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_refactored_health_agent, operation, progress);
    }
}


#include "nimcp.h"
#include "core/brain/nimcp_brain.h"
#include "core/neuralnet/nimcp_neuralnet.h"
#include "cognitive/ethics/nimcp_ethics.h"
#include "cognitive/knowledge/nimcp_knowledge.h"
#include "cognitive/nimcp_working_memory.h"
#include "cognitive/global_workspace/nimcp_global_workspace.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/config/nimcp_config.h"
#include "utils/cache/nimcp_cache.h"
#include "utils/time/nimcp_time.h"
#include "security/nimcp_blood_brain_barrier.h"
#include "core/brain/nimcp_brain_internal.h"
#include "middleware/training/nimcp_brain_training_integration.h"
#include "middleware/training/nimcp_loss_functions.h"
#include "middleware/training/nimcp_optimizers.h"
#include "middleware/training/nimcp_lr_scheduler.h"
#include "middleware/training/nimcp_training_callbacks.h"
#include "plasticity/adaptive/nimcp_adaptive.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <unistd.h>

//=============================================================================
// Internal Handle Structures (shared with extracted modules)
//=============================================================================

struct nimcp_brain_handle {
    brain_t internal_brain;
};

struct nimcp_network_handle {
    neural_network_t internal_network;
};

struct nimcp_ethics_handle {
    ethics_engine_t internal_ethics;
};

struct nimcp_knowledge_handle {
    knowledge_system_t internal_knowledge;
};

struct nimcp_brain_snapshot_handle {
    brain_t internal_brain_snapshot;
    uint64_t timestamp_us;
    size_t shared_memory_size;
    uint32_t snapshot_refcount;
    bool is_isolated;
};

//=============================================================================
// Global State
//=============================================================================

static char g_last_error[256] = "No error";
static bool g_initialized = false;

//=============================================================================
// Error Handling (shared with extracted modules)
//=============================================================================

void set_error(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vsnprintf(g_last_error, sizeof(g_last_error), fmt, args);
    va_end(args);
}

const char* nimcp_get_error(void) {
    return g_last_error;
}

//=============================================================================
// Version Functions
//=============================================================================

const char* nimcp_version(void) {
    return NIMCP_VERSION_STRING;
}

int nimcp_version_int(void) {
    return NIMCP_VERSION_MAJOR * 10000 + NIMCP_VERSION_MINOR * 100 + NIMCP_VERSION_PATCH;
}

//=============================================================================
// Initialization
//=============================================================================

nimcp_status_t nimcp_init(void) {
    LOG_INFO("Initializing NIMCP library version %s", NIMCP_VERSION_STRING);

    if (g_initialized) {
        LOG_DEBUG("NIMCP already initialized, skipping");
        return NIMCP_OK;
    }

    // Initialize memory tracking (unified memory management)
    LOG_DEBUG("Initializing memory tracking system");
    nimcp_memory_init();

    // Initialize bio-async system (core async communication infrastructure)
    LOG_INFO("Initializing bio-async communication system");
    nimcp_bio_async_config_t bio_async_config = {0};
    if (nimcp_bio_async_init(&bio_async_config) != NIMCP_SUCCESS) {
        LOG_ERROR("Failed to initialize bio-async system");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NOT_INITIALIZED, "Failed to initialize bio-async system");
        nimcp_memory_cleanup();
        set_error("Failed to initialize bio-async system");
        return NIMCP_ERROR;
    }

    // Initialize bio-async router (message routing for modules)
    LOG_DEBUG("Initializing bio-async router");
    if (bio_router_init(NULL) != NIMCP_SUCCESS) {
        LOG_ERROR("Failed to initialize bio-async router");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NOT_INITIALIZED, "Failed to initialize bio-async router");
        nimcp_bio_async_shutdown();
        nimcp_memory_cleanup();
        set_error("Failed to initialize bio-async router");
        return NIMCP_ERROR;
    }

    // Initialize COW cache system
    LOG_DEBUG("Initializing COW cache system");
    nimcp_cache_init();

    g_initialized = true;
    set_error("No error");
    LOG_INFO("NIMCP library initialized successfully");
    return NIMCP_OK;
}

void nimcp_shutdown(void) {
    LOG_INFO("Shutting down NIMCP library");

    if (!g_initialized) {
        LOG_DEBUG("NIMCP not initialized, nothing to shutdown");
        return;
    }

    // Cleanup cache system
    LOG_DEBUG("Cleaning up COW cache system");
    nimcp_cache_cleanup();

    // Shutdown bio-async router
    LOG_DEBUG("Shutting down bio-async router");
    bio_router_shutdown();

    // Shutdown bio-async system
    LOG_DEBUG("Shutting down bio-async communication system");
    nimcp_bio_async_shutdown();

    // Cleanup memory tracking (last)
    LOG_DEBUG("Cleaning up memory tracking");
    nimcp_memory_cleanup();

    g_initialized = false;
    LOG_INFO("NIMCP library shutdown complete");
}

//=============================================================================
// NOTE: The following APIs are now in separate modules:
// - Brain API: nimcp_brain_api.c (create, destroy, learn, predict, save/load)
// - Snapshot API: nimcp_snapshot_api.c (snapshot save/restore, COW clone)
// - Cognitive API: nimcp_cognitive_api.c (working memory, workspace, resize)
// - Subsystems API: nimcp_subsystems_api.c (network, ethics, knowledge)
// - Oscillation API: nimcp_oscillation_api.c (phasor, PAC, coherence)
//
// Training API remains below due to complex internal state management
//=============================================================================
