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
#include "utils/fault_tolerance/nimcp_health_agent_macros.h"
#include "mesh/nimcp_mesh_participant.h"
#include "mesh/nimcp_mesh_adapter.h"

NIMCP_DECLARE_HEALTH_AGENT_ATOMIC(brain_init_immune_bridge_coordinator)
//=============================================================================
// Mesh Participant Registration
//=============================================================================

static mesh_participant_id_t g_brain_init_immune_bridge_coordinator_mesh_id = 0;
static mesh_participant_registry_t* g_brain_init_immune_bridge_coordinator_mesh_registry = NULL;

nimcp_error_t brain_init_immune_bridge_coordinator_mesh_register(mesh_participant_registry_t* registry) {
    if (!registry) return NIMCP_ERROR_NULL_POINTER;
    if (g_brain_init_immune_bridge_coordinator_mesh_id != 0) return NIMCP_SUCCESS;
    mesh_participant_interface_t iface;
    mesh_participant_interface_init(&iface);
    strncpy(iface.module_name, "brain_init_immune_bridge_coordinator", MESH_MAX_NAME_LEN - 1);
    iface.type = MESH_PARTICIPANT_MODULE;
    iface.home_channel = mesh_adapter_get_default_channel(MESH_ADAPTER_CATEGORY_SYSTEM);
    mesh_participant_config_t config;
    mesh_participant_config_init(&config);
    config.module_name = "brain_init_immune_bridge_coordinator";
    config.type = MESH_PARTICIPANT_MODULE;
    config.home_channel = iface.home_channel;
    nimcp_error_t err = mesh_participant_register(registry, &iface, &config, &g_brain_init_immune_bridge_coordinator_mesh_id);
    if (err == NIMCP_SUCCESS) g_brain_init_immune_bridge_coordinator_mesh_registry = registry;
    return err;
}

void brain_init_immune_bridge_coordinator_mesh_unregister(void) {
    if (g_brain_init_immune_bridge_coordinator_mesh_registry && g_brain_init_immune_bridge_coordinator_mesh_id != 0) {
        mesh_participant_unregister(g_brain_init_immune_bridge_coordinator_mesh_registry, g_brain_init_immune_bridge_coordinator_mesh_id);
        g_brain_init_immune_bridge_coordinator_mesh_id = 0;
        g_brain_init_immune_bridge_coordinator_mesh_registry = NULL;
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
