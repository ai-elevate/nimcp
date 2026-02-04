//=============================================================================
// nimcp_brain_init_bio_async_orchestrator.c - Bio-Async Orchestrator Init
//=============================================================================
/**
 * @file nimcp_brain_init_bio_async_orchestrator.c
 * @brief Bio-Async Orchestrator subsystem initialization for brain
 *
 * WHAT: Bio-Async Orchestrator initialization and integration setup
 * WHY:  Central coordination of 200+ bio-async modules for inter-module messaging
 * HOW:  Creates orchestrator, connects to bio-router, wires integrations
 *
 * BIOLOGICAL BASIS:
 * Models the brain's glial network coordination where astrocytes orchestrate
 * synaptic communication and calcium wave propagation across brain regions.
 *
 * DESIGN PATTERNS:
 * - Factory: Created by brain factory during lifecycle init
 * - Coordinator: Manages distributed bio-async module communication
 * - Observer: Receives and routes messages between modules
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
#include "async/nimcp_bio_async_orchestrator.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/exception/nimcp_exception_macros.h"

#define LOG_MODULE "BRAIN_INIT_BIO_ASYNC_ORCH"
#include "utils/fault_tolerance/nimcp_health_agent_macros.h"
#include "mesh/nimcp_mesh_participant.h"
#include "mesh/nimcp_mesh_adapter.h"

NIMCP_DECLARE_HEALTH_AGENT_ATOMIC(brain_init_bio_async_orchestrator)
//=============================================================================
// Mesh Participant Registration
//=============================================================================

static mesh_participant_id_t g_brain_init_bio_async_orchestrator_mesh_id = 0;
static mesh_participant_registry_t* g_brain_init_bio_async_orchestrator_mesh_registry = NULL;

nimcp_error_t brain_init_bio_async_orchestrator_mesh_register(mesh_participant_registry_t* registry) {
    if (!registry) return NIMCP_ERROR_NULL_POINTER;
    if (g_brain_init_bio_async_orchestrator_mesh_id != 0) return NIMCP_SUCCESS;
    mesh_participant_interface_t iface;
    mesh_participant_interface_init(&iface);
    strncpy(iface.module_name, "brain_init_bio_async_orchestrator", MESH_MAX_NAME_LEN - 1);
    iface.type = MESH_PARTICIPANT_MODULE;
    iface.home_channel = mesh_adapter_get_default_channel(MESH_ADAPTER_CATEGORY_SYSTEM);
    mesh_participant_config_t config;
    mesh_participant_config_init(&config);
    config.module_name = "brain_init_bio_async_orchestrator";
    config.type = MESH_PARTICIPANT_MODULE;
    config.home_channel = iface.home_channel;
    nimcp_error_t err = mesh_participant_register(registry, &iface, &config, &g_brain_init_bio_async_orchestrator_mesh_id);
    if (err == NIMCP_SUCCESS) g_brain_init_bio_async_orchestrator_mesh_registry = registry;
    return err;
}

void brain_init_bio_async_orchestrator_mesh_unregister(void) {
    if (g_brain_init_bio_async_orchestrator_mesh_registry && g_brain_init_bio_async_orchestrator_mesh_id != 0) {
        mesh_participant_unregister(g_brain_init_bio_async_orchestrator_mesh_registry, g_brain_init_bio_async_orchestrator_mesh_id);
        g_brain_init_bio_async_orchestrator_mesh_id = 0;
        g_brain_init_bio_async_orchestrator_mesh_registry = NULL;
    }
}


//=============================================================================
// Main Initialization Function
//=============================================================================

/**
 * @brief Initialize bio-async orchestrator subsystem
 *
 * WHAT: Creates and configures bio-async orchestrator for brain
 * WHY:  Foundation layer for all inter-module bio-async communication
 * HOW:  Guard clause checks, create orchestrator, connect integrations
 *
 * @param brain Brain instance to initialize orchestrator for
 * @return true on success or non-fatal failure, false on critical error
 */
bool nimcp_brain_factory_init_bio_async_orchestrator_subsystem(brain_t brain) {
    /* Guard clause: NULL check */
    if (!brain) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
            "nimcp_brain_factory_init_bio_async_orchestrator_subsystem: brain is NULL");
        return false;
    }

    /* Initialize fields to defaults */
    brain->bio_async_orchestrator = NULL;
    brain->bio_async_orchestrator_enabled = false;

    /* Check if bio-async is enabled for this brain */
    if (!brain->bio_async_enabled) {
        NIMCP_LOGGING_DEBUG("Bio-async orchestrator skipped (bio-async disabled)");
        return true;  /* Not an error - just not needed */
    }

    /* Create orchestrator configuration */
    bio_orchestrator_config_t config;
    bio_orchestrator_default_config(&config);

    /* Configure based on brain settings */
    config.enable_statistics = true;
    config.enable_logging = true;
    config.enable_bio_async = true;
    config.enable_auto_health_check = true;

    /* Create orchestrator */
    bio_async_orchestrator_t* orch = bio_orchestrator_create(&config);
    if (!orch) {
        NIMCP_LOGGING_WARN("Failed to create bio-async orchestrator - "
                          "continuing without orchestration");
        return true;  /* Non-fatal: bio-async still works without coordinator */
    }

    /* Start the orchestrator */
    if (bio_orchestrator_start(orch) != 0) {
        NIMCP_LOGGING_WARN("Failed to start bio-async orchestrator");
        bio_orchestrator_destroy(orch);
        return true;  /* Non-fatal */
    }

    /* Store orchestrator in brain */
    brain->bio_async_orchestrator = orch;
    brain->bio_async_orchestrator_enabled = true;

    /* Link orchestrator to router for KG-driven wiring callbacks */
    extern nimcp_error_t bio_router_set_orchestrator(struct bio_async_orchestrator*);
    bio_router_set_orchestrator(orch);

    /* Log success */
    bio_orchestrator_stats_t stats;
    if (bio_orchestrator_get_stats(orch, &stats) == 0) {
        NIMCP_LOGGING_INFO("Bio-async orchestrator initialized: "
                          "modules=%u, health=%.1f%%",
                          stats.total_modules,
                          stats.system_health_score * 100.0f);
    } else {
        NIMCP_LOGGING_INFO("Bio-async orchestrator initialized");
    }

    return true;
}
