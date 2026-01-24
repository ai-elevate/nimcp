//=============================================================================
// nimcp_brain_init_plasticity_coordinator.c - Plasticity Coordinator Init
//=============================================================================
/**
 * @file nimcp_brain_init_plasticity_coordinator.c
 * @brief Plasticity Coordinator subsystem initialization for brain
 *
 * WHAT: Plasticity Coordinator initialization and mechanism registration
 * WHY:  Unified management of 8 plasticity mechanisms with conflict resolution
 * HOW:  Creates coordinator, connects bio-async and immune, registers mechanisms
 *
 * BIOLOGICAL BASIS:
 * The brain employs multiple plasticity mechanisms at different timescales:
 * - Fast (ms): STP - vesicle depletion/facilitation
 * - Medium (100ms-1s): STDP, BCM - spike-timing and rate-based rules
 * - Slow (minutes-hours): Homeostatic, synaptic scaling
 * This coordinator resolves conflicts when mechanisms disagree (STDP→LTP vs BCM→LTD).
 *
 * DESIGN PATTERNS:
 * - Factory: Created by brain factory during lifecycle init
 * - Registry: Stores and manages plasticity mechanisms
 * - Strategy: Configurable conflict resolution strategies
 * - Mediator: Coordinates cross-mechanism interactions
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
#include "plasticity/nimcp_plasticity_coordinator.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/exception/nimcp_exception_macros.h"

#define LOG_MODULE "BRAIN_INIT_PLASTICITY_COORD"

//=============================================================================
// Main Initialization Function
//=============================================================================

/**
 * @brief Initialize plasticity coordinator subsystem
 *
 * WHAT: Creates and configures plasticity coordinator for brain
 * WHY:  Centralizes management of all plasticity mechanisms
 * HOW:  Guard clause checks, create coordinator, wire dependencies
 *
 * @param brain Brain instance to initialize coordinator for
 * @return true on success or non-fatal failure, false on critical error
 */
bool nimcp_brain_factory_init_plasticity_coordinator_subsystem(brain_t brain) {
    /* Guard clause: NULL check */
    if (!brain) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
            "nimcp_brain_factory_init_plasticity_coordinator_subsystem: brain is NULL");
        return false;
    }

    /* Initialize fields to defaults */
    brain->plasticity_coordinator = NULL;
    brain->plasticity_coordinator_enabled = false;

    /* Check if plasticity is relevant for this brain */
    /* Enable if we have any plasticity-related subsystems */
    bool should_enable = brain->bio_async_enabled || brain->immune_enabled;

    if (!should_enable) {
        NIMCP_LOGGING_DEBUG("Plasticity coordinator skipped (no dependencies)");
        return true;
    }

    /* Create coordinator configuration */
    plasticity_coordinator_config_t config;
    plasticity_coordinator_default_config(&config);

    /* Configure based on brain settings */
    config.enable_bio_async = brain->bio_async_enabled;
    config.enable_brain_immune = brain->immune_enabled;
    config.enable_statistics = true;
    config.enable_logging = true;
    config.enable_energy_tracking = true;

    /* Create coordinator */
    plasticity_coordinator_t* coord = plasticity_coordinator_create(&config);
    if (!coord) {
        NIMCP_LOGGING_WARN("Failed to create plasticity coordinator - "
                          "continuing without plasticity coordination");
        return true;  /* Non-fatal */
    }

    /* Connect to brain immune system if available */
    if (brain->immune_enabled && brain->immune_system) {
        if (plasticity_coordinator_connect_brain_immune(coord, brain->immune_system) == 0) {
            NIMCP_LOGGING_DEBUG("Plasticity coordinator connected to brain immune");
        } else {
            NIMCP_LOGGING_WARN("Failed to connect plasticity coordinator to immune");
        }
    }

    /* Connect to bio-async router if available */
    if (brain->bio_async_enabled) {
        if (plasticity_coordinator_connect_bio_async(coord) == 0) {
            NIMCP_LOGGING_DEBUG("Plasticity coordinator connected to bio-async");
        } else {
            NIMCP_LOGGING_WARN("Failed to connect plasticity coordinator to bio-async");
        }
    }

    /* Store coordinator in brain */
    brain->plasticity_coordinator = coord;
    brain->plasticity_coordinator_enabled = true;

    /* Log success */
    plasticity_coordinator_stats_t stats;
    if (plasticity_coordinator_get_stats(coord, &stats) == 0) {
        NIMCP_LOGGING_INFO("Plasticity coordinator initialized: "
                          "mechanisms=%u, state=%s, immune=%s, bio_async=%s",
                          stats.total_mechanisms,
                          plasticity_coordinator_state_to_string(stats.current_state),
                          brain->immune_enabled ? "connected" : "disabled",
                          brain->bio_async_enabled ? "connected" : "disabled");
    } else {
        NIMCP_LOGGING_INFO("Plasticity coordinator initialized");
    }

    return true;
}
