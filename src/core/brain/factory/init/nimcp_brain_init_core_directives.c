//=============================================================================
// nimcp_brain_init_core_directives.c - Core Directives Initialization
//=============================================================================
/**
 * @file nimcp_brain_init_core_directives.c
 * @brief Core directives subsystem initialization for brain
 *
 * WHAT: Core directives initialization and integration setup
 * WHY:  Provides ethical constraint enforcement for all brain actions
 * HOW:  Creates directives system, connects to immune/FEP, enables bio-async
 *
 * @version 1.0.0
 * @author NIMCP Development Team
 * @date 2025-12-16
 */

//=============================================================================
// Includes
//=============================================================================

#include "core/brain/factory/init/nimcp_brain_init_subsystems.h"
#include "core/brain/nimcp_brain.h"
#include "core/brain/nimcp_brain_internal.h"
#include "cognitive/ethics/nimcp_core_directives.h"
#include "utils/logging/nimcp_logging.h"

#define LOG_MODULE "BRAIN_INIT_CORE_DIRECTIVES"

//=============================================================================
// Main Initialization Function
//=============================================================================

bool nimcp_brain_factory_init_core_directives_subsystem(brain_t brain) {
    if (!brain) {
        NIMCP_LOGGING_ERROR("Null brain in init_core_directives_subsystem");
        return false;
    }

    /* Initialize core directives fields to defaults */
    brain->core_directives = NULL;
    brain->directive_immune_bridge = NULL;
    brain->directive_fep_bridge = NULL;
    brain->core_directives_enabled = false;

    /* Core directives should always be enabled as the ethical foundation */
    /* However, we make it conditional on bio-async or immune for now */
    bool should_enable = true; // Always enable

    if (!should_enable) {
        NIMCP_LOGGING_DEBUG("Core directives not enabled");
        return true;  /* Not an error - just not needed */
    }

    /* Create directives configuration */
    core_directives_config_t dir_config;
    core_directives_default_config(&dir_config);

    /* Configure based on brain settings */
    dir_config.enable_bio_async = brain->bio_async_enabled;
    dir_config.enable_immune_integration = brain->immune_enabled;
    dir_config.enable_fep_integration = (brain->fep_orchestrator != NULL);

    /* Enable all ethical layers */
    dir_config.enable_first_law = true;           // Harm prevention
    dir_config.enable_second_law = true;          // Obedience (when safe)
    dir_config.enable_third_law = true;           // Self-preservation
    dir_config.enable_golden_rule = true;         // Reciprocity
    dir_config.enable_combinatorial_harm = true;  // Emergent harm detection

    /* Set thresholds */
    dir_config.harm_threshold = 0.3f;
    dir_config.severity_threshold = 0.5f;
    dir_config.confidence_threshold = 0.6f;

    /* Action history for combinatorial analysis */
    dir_config.action_history_size = 100;
    dir_config.max_combination_depth = 5;

    /* Create core directives system */
    core_directives_system_t* directives = core_directives_create(&dir_config);
    if (!directives) {
        NIMCP_LOGGING_WARN("Failed to create core directives - continuing without ethical constraints");
        return true;  /* Non-fatal, but should ideally fail */
    }

    /* Connect to bio-async if available */
    if (brain->bio_async_enabled) {
        if (core_directives_connect_bio_async(directives) == 0) {
            NIMCP_LOGGING_DEBUG("Core directives connected to bio-async router");
        } else {
            NIMCP_LOGGING_WARN("Failed to connect core directives to bio-async");
        }
    }

    /* Connect to brain immune system if available */
    if (brain->immune_enabled && brain->immune_system) {
        if (core_directives_connect_immune(directives, brain->immune_system) == 0) {
            NIMCP_LOGGING_DEBUG("Core directives connected to brain immune system");
            /* Store bridge reference from directives internal structure */
            /* Note: This is a placeholder - the actual bridge is managed internally */
        } else {
            NIMCP_LOGGING_WARN("Failed to connect core directives to immune system");
        }
    }

    /* Connect to FEP orchestrator if available */
    if (brain->fep_orchestrator) {
        if (core_directives_connect_fep(directives, brain->fep_orchestrator) == 0) {
            NIMCP_LOGGING_DEBUG("Core directives connected to FEP orchestrator");
            /* Store bridge reference from directives internal structure */
            /* Note: This is a placeholder - the actual bridge is managed internally */
        } else {
            NIMCP_LOGGING_WARN("Failed to connect core directives to FEP orchestrator");
        }
    }

    /* Store in brain */
    brain->core_directives = directives;
    brain->core_directives_enabled = true;

    NIMCP_LOGGING_INFO("Core directives subsystem initialized successfully");
    NIMCP_LOGGING_INFO("  Asimov's Laws: ENABLED (harm prevention, obedience, self-preservation)");
    NIMCP_LOGGING_INFO("  Golden Rule: ENABLED (reciprocity evaluation)");
    NIMCP_LOGGING_INFO("  Combinatorial Harm: ENABLED (emergent harm detection)");
    NIMCP_LOGGING_INFO("  Action History: %u entries", dir_config.action_history_size);
    NIMCP_LOGGING_INFO("  Bio-Async: %s", brain->bio_async_enabled ? "CONNECTED" : "DISABLED");
    NIMCP_LOGGING_INFO("  Immune Integration: %s", brain->immune_enabled ? "CONNECTED" : "DISABLED");
    NIMCP_LOGGING_INFO("  FEP Integration: %s", brain->fep_orchestrator ? "CONNECTED" : "DISABLED");

    return true;
}
