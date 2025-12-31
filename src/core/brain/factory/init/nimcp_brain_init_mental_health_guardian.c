//=============================================================================
// nimcp_brain_init_mental_health_guardian.c - Mental Health Guardian Initialization
//=============================================================================
/**
 * @file nimcp_brain_init_mental_health_guardian.c
 * @brief Mental Health Guardian subsystem initialization for brain factory
 *
 * WHAT: Initializes the Mental Health Guardian background monitoring agent
 * WHY:  Proactively detect and correct mental health abnormalities
 * HOW:  Create guardian, connect to integrations, auto-start if configured
 *
 * INITIALIZATION ORDER:
 * This init must run AFTER:
 * - Mental health monitor (provides disorder detection)
 * - Immune system (optional, for threat reporting)
 * - Internal KG (optional, for topology awareness)
 *
 * @version 1.0.0
 * @author NIMCP Development Team
 * @date 2025-12-31
 */

//=============================================================================
// Includes
//=============================================================================

#include "core/brain/factory/init/nimcp_brain_init_subsystems.h"
#include "core/brain/nimcp_brain_internal.h"
#include "cognitive/mental_health/nimcp_mental_health_guardian.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

//=============================================================================
// Logging
//=============================================================================

#define LOG_TAG "[BRAIN_INIT_GUARDIAN]"

//=============================================================================
// Integration Helper Functions
//=============================================================================

/**
 * @brief Connect guardian to immune system for threat reporting
 */
static void connect_guardian_to_immune(
    mental_health_guardian_t* guardian,
    brain_t brain)
{
    if (!brain->immune_system || !brain->immune_enabled) {
        fprintf(stderr, LOG_TAG " Immune system not available - skipping connection\n");
        return;
    }

    if (mental_health_guardian_connect_immune(guardian, brain->immune_system)) {
        fprintf(stderr, LOG_TAG " Connected to immune system\n");
    } else {
        fprintf(stderr, LOG_TAG " WARNING: Failed to connect to immune system\n");
    }
}

/**
 * @brief Connect guardian to internal knowledge graph
 */
static void connect_guardian_to_kg(
    mental_health_guardian_t* guardian,
    brain_t brain)
{
    if (!brain->internal_kg || !brain->internal_kg_enabled) {
        fprintf(stderr, LOG_TAG " Internal KG not available - skipping connection\n");
        return;
    }

    if (mental_health_guardian_connect_kg(
            guardian,
            brain->internal_kg,
            brain->internal_kg_admin_token)) {
        fprintf(stderr, LOG_TAG " Connected to internal knowledge graph\n");
    } else {
        fprintf(stderr, LOG_TAG " WARNING: Failed to connect to internal KG\n");
    }
}

//=============================================================================
// Main Initialization Function
//=============================================================================

/**
 * @brief Initialize mental health guardian subsystem for brain
 *
 * Creates the Mental Health Guardian background agent and connects
 * it to relevant subsystems (immune, KG).
 *
 * @param brain Brain instance to initialize guardian for
 * @return true on success, false on failure
 *
 * PROCESS:
 * 1. Check if guardian is enabled in brain config
 * 2. Verify mental health monitor exists
 * 3. Create guardian with default config
 * 4. Connect to immune system (if available)
 * 5. Connect to internal KG (if available)
 * 6. Auto-start if brain is in running state
 */
bool nimcp_brain_factory_init_mental_health_guardian_subsystem(brain_t brain) {
    if (!brain) {
        return false;
    }

    /* Check if guardian is disabled in config */
    if (!brain->config.enable_mental_health_guardian) {
        brain->mental_health_guardian = NULL;
        fprintf(stderr, LOG_TAG " Mental Health Guardian disabled by config\n");
        return true;  /* Success - disabled by config */
    }

    /* Verify mental health monitor exists */
    if (!brain->mental_health_monitor) {
        fprintf(stderr, LOG_TAG " ERROR: Mental health monitor not initialized\n");
        brain->mental_health_guardian = NULL;
        return false;
    }

    fprintf(stderr, LOG_TAG " Initializing Mental Health Guardian...\n");

    /* ====================================================================== */
    /* CREATE GUARDIAN WITH DEFAULT CONFIG                                    */
    /* ====================================================================== */

    mental_health_guardian_config_t config = mental_health_guardian_default_config();

    /* Adjust config based on brain settings */
    config.immune_integration = brain->immune_enabled;
    config.kg_integration = brain->internal_kg_enabled;

    mental_health_guardian_t* guardian = mental_health_guardian_create(brain, &config);
    if (!guardian) {
        fprintf(stderr, LOG_TAG " ERROR: Failed to create guardian\n");
        brain->mental_health_guardian = NULL;
        return false;
    }

    /* Store in brain */
    brain->mental_health_guardian = guardian;

    fprintf(stderr, LOG_TAG " Guardian created (interval=%ums)\n",
            config.monitoring_interval_ms);

    /* ====================================================================== */
    /* CONNECT TO RELEVANT SUBSYSTEMS                                         */
    /* ====================================================================== */

    /* 1. Immune System - Threat reporting */
    connect_guardian_to_immune(guardian, brain);

    /* 2. Internal KG - Topology awareness */
    connect_guardian_to_kg(guardian, brain);

    /* ====================================================================== */
    /* AUTO-START IF BRAIN IS RUNNING                                         */
    /* ====================================================================== */

    /* Note: Brain state check would go here if we had a running state */
    /* For now, we don't auto-start - caller should start explicitly */

    fprintf(stderr, LOG_TAG " Mental Health Guardian initialization complete\n");
    fprintf(stderr, LOG_TAG "   Call mental_health_guardian_start() to begin monitoring\n");

    return true;
}

//=============================================================================
// Accessor Functions
//=============================================================================

/**
 * @brief Get mental health guardian from brain
 *
 * @param brain Brain instance
 * @return Guardian handle or NULL if not enabled
 */
mental_health_guardian_t* brain_get_mental_health_guardian(brain_t brain) {
    if (!brain) {
        return NULL;
    }
    return brain->mental_health_guardian;
}

/**
 * @brief Start the mental health guardian
 *
 * Convenience function to start the guardian from brain reference.
 *
 * @param brain Brain instance
 * @return true on success
 */
bool brain_start_mental_health_guardian(brain_t brain) {
    if (!brain || !brain->mental_health_guardian) {
        return false;
    }
    return mental_health_guardian_start(brain->mental_health_guardian);
}

/**
 * @brief Stop the mental health guardian
 *
 * @param brain Brain instance
 * @return true on success
 */
bool brain_stop_mental_health_guardian(brain_t brain) {
    if (!brain || !brain->mental_health_guardian) {
        return false;
    }
    return mental_health_guardian_stop(brain->mental_health_guardian);
}

/**
 * @brief Get guardian status
 *
 * @param brain Brain instance
 * @param status Output status structure
 * @return true on success
 */
bool brain_get_mental_health_guardian_status(
    brain_t brain,
    mental_health_guardian_status_t* status)
{
    if (!brain || !brain->mental_health_guardian || !status) {
        return false;
    }
    return mental_health_guardian_get_status(brain->mental_health_guardian, status);
}

//=============================================================================
// Destruction (called during brain_destroy)
//=============================================================================

/**
 * @brief Destroy mental health guardian subsystem
 *
 * @param brain Brain instance
 */
void nimcp_brain_factory_destroy_mental_health_guardian_subsystem(brain_t brain) {
    if (!brain) {
        return;
    }

    if (brain->mental_health_guardian) {
        fprintf(stderr, LOG_TAG " Destroying Mental Health Guardian\n");
        mental_health_guardian_destroy(brain->mental_health_guardian);
        brain->mental_health_guardian = NULL;
    }
}
