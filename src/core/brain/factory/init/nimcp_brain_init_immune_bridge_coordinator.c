//=============================================================================
// nimcp_brain_init_immune_bridge_coordinator.c - Immune Bridge Coordinator Init
//=============================================================================
/**
 * @file nimcp_brain_init_immune_bridge_coordinator.c
 * @brief Immune Bridge Coordinator subsystem initialization for brain
 *
 * WHAT: Immune Bridge Coordinator initialization and bridge registration setup
 * WHY:  Central registry and manager for 27+ immune bridges across NIMCP
 * HOW:  Creates coordinator, connects brain immune and bio-async, starts monitoring
 *
 * BIOLOGICAL BASIS:
 * Models systemic immune coordination where:
 * - Local responses: Tissue-specific immune activation
 * - Regional coordination: Lymph node integration of signals
 * - Systemic coordination: Cytokine cascades coordinate whole-body response
 * The coordinator manages bridge-level (module) coordination analogous to
 * organ-level immune integration in biology.
 *
 * DESIGN PATTERNS:
 * - Factory: Created by brain factory during lifecycle init
 * - Registry: Dynamic bridge registration with category tracking
 * - Observer: Health monitoring and event notification across bridges
 * - Mediator: Coordinates cross-bridge communication via bio-async
 *
 * @version 1.0.0
 * @author NIMCP Development Team
 * @date 2025-12-15
 */

//=============================================================================
// Includes
//=============================================================================

#include "core/brain/factory/init/nimcp_brain_init_subsystems.h"
#include "core/brain/nimcp_brain.h"
#include "core/brain/nimcp_brain_internal.h"
#include "cognitive/immune/nimcp_immune_bridge_coordinator.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/exception/nimcp_exception_macros.h"

#define LOG_MODULE "BRAIN_INIT_IMMUNE_BRIDGE_COORD"

//=============================================================================
#include <stddef.h>  /* for NULL */
// Health Agent Integration (Phase 8: System-Wide Health Integration)
//=============================================================================
struct nimcp_health_agent;
typedef struct nimcp_health_agent nimcp_health_agent_t;
extern void nimcp_health_agent_heartbeat_ex(nimcp_health_agent_t* agent,
                                             const char* operation,
                                             float progress);

/** Global health agent for brain_init_immune_bridge_coordinator module */
static nimcp_health_agent_t* g_brain_init_immune_bridge_coordinator_health_agent = NULL;

/**
 * @brief Set health agent for brain_init_immune_bridge_coordinator heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
static void brain_init_immune_bridge_coordinator_set_health_agent(nimcp_health_agent_t* agent) {
    g_brain_init_immune_bridge_coordinator_health_agent = agent;
}

/** @brief Send heartbeat from brain_init_immune_bridge_coordinator module */
static inline void brain_init_immune_bridge_coordinator_heartbeat(const char* operation, float progress) {
    if (g_brain_init_immune_bridge_coordinator_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_brain_init_immune_bridge_coordinator_health_agent, operation, progress);
    }
}


//=============================================================================
// Main Initialization Function
//=============================================================================

/**
 * @brief Initialize immune bridge coordinator subsystem
 *
 * WHAT: Creates and configures immune bridge coordinator for brain
 * WHY:  Centralizes management and monitoring of all immune bridges
 * HOW:  Guard clause checks, create coordinator, wire dependencies, start
 *
 * @param brain Brain instance to initialize coordinator for
 * @return true on success or non-fatal failure, false on critical error
 */
bool nimcp_brain_factory_init_immune_bridge_coordinator_subsystem(brain_t brain) {
    /* Guard clause: NULL check */
    if (!brain) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
            "nimcp_brain_factory_init_immune_bridge_coordinator_subsystem: brain is NULL");
        return false;
    }

    /* Initialize fields to defaults */
    brain->immune_bridge_coordinator = NULL;
    brain->immune_bridge_coordinator_enabled = false;

    /* Check if immune system is enabled */
    if (!brain->immune_enabled) {
        NIMCP_LOGGING_DEBUG("Immune bridge coordinator skipped (immune disabled)");
        return true;
    }

    /* Create coordinator configuration */
    immune_bridge_coordinator_config_t config;
    immune_bridge_coordinator_default_config(&config);

    /* Configure based on brain settings */
    config.enable_bio_async = brain->bio_async_enabled;
    config.enable_brain_immune = true;  /* Always enable if immune is on */
    config.enable_statistics = true;
    config.enable_logging = true;
    config.enable_auto_update = true;

    /* Create coordinator */
    immune_bridge_coordinator_t* coord = immune_bridge_coordinator_create(&config);
    if (!coord) {
        NIMCP_LOGGING_WARN("Failed to create immune bridge coordinator - "
                          "continuing without bridge coordination");
        return true;  /* Non-fatal */
    }

    /* Connect to brain immune system */
    if (brain->immune_system) {
        if (immune_bridge_coordinator_connect_brain_immune(coord, brain->immune_system) == 0) {
            NIMCP_LOGGING_DEBUG("Immune bridge coordinator connected to brain immune");
        } else {
            NIMCP_LOGGING_WARN("Failed to connect immune bridge coordinator to immune");
        }
    }

    /* Connect to bio-async router if available */
    if (brain->bio_async_enabled) {
        if (immune_bridge_coordinator_connect_bio_async(coord) == 0) {
            NIMCP_LOGGING_DEBUG("Immune bridge coordinator connected to bio-async");
        } else {
            NIMCP_LOGGING_WARN("Failed to connect immune bridge coordinator to bio-async");
        }
    }

    /* Start the coordinator */
    if (immune_bridge_coordinator_start(coord) != 0) {
        NIMCP_LOGGING_WARN("Failed to start immune bridge coordinator");
        immune_bridge_coordinator_destroy(coord);
        return true;  /* Non-fatal */
    }

    /* Store coordinator in brain */
    brain->immune_bridge_coordinator = coord;
    brain->immune_bridge_coordinator_enabled = true;

    /* Log success */
    immune_coordinator_stats_t stats;
    if (immune_bridge_coordinator_get_stats(coord, &stats) == 0) {
        NIMCP_LOGGING_INFO("Immune bridge coordinator initialized: "
                          "bridges=%u, health=%.2f, immune=%s, bio_async=%s",
                          stats.total_bridges,
                          stats.system_health,
                          brain->immune_enabled ? "connected" : "disabled",
                          brain->bio_async_enabled ? "connected" : "disabled");
    } else {
        NIMCP_LOGGING_INFO("Immune bridge coordinator initialized");
    }

    return true;
}
