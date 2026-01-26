//=============================================================================
// nimcp_brain_init_medulla.c - Medulla Oblongata Initialization
//=============================================================================
/**
 * @file nimcp_brain_init_medulla.c
 * @brief Medulla oblongata subsystem initialization for brain
 *
 * WHAT: Medulla initialization, update, and destruction functions
 * WHY:  Provides foundational brainstem autonomic regulation for brain
 * HOW:  Creates medulla, connects integrations, manages lifecycle
 *
 * BIOLOGICAL BASIS:
 * The medulla oblongata is the most primitive part of the brain, essential
 * for survival. It must be initialized early and destroyed last.
 *
 * @version 1.0.0
 * @author NIMCP Development Team
 * @date 2025-12-17
 */

//=============================================================================
// Includes
//=============================================================================

#include "core/brain/factory/init/nimcp_brain_init_medulla.h"
#include "core/brain/factory/init/nimcp_brain_init_subsystems.h"
#include "core/brain/nimcp_brain.h"
#include "core/brain/nimcp_brain_internal.h"
#include "core/medulla/nimcp_medulla.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/error/nimcp_error_codes.h"
#include "utils/exception/nimcp_exception_macros.h"

#define LOG_MODULE "BRAIN_INIT_MEDULLA"

//=============================================================================
#include <stddef.h>  /* for NULL */
// Health Agent Integration (Phase 8: System-Wide Health Integration)
//=============================================================================
struct nimcp_health_agent;
typedef struct nimcp_health_agent nimcp_health_agent_t;
extern void nimcp_health_agent_heartbeat_ex(nimcp_health_agent_t* agent,
                                             const char* operation,
                                             float progress);

/** Global health agent for brain_init_medulla module */
static nimcp_health_agent_t* g_brain_init_medulla_health_agent = NULL;

/**
 * @brief Set health agent for brain_init_medulla heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
static void brain_init_medulla_set_health_agent(nimcp_health_agent_t* agent) {
    g_brain_init_medulla_health_agent = agent;
}

/** @brief Send heartbeat from brain_init_medulla module */
static inline void brain_init_medulla_heartbeat(const char* operation, float progress) {
    if (g_brain_init_medulla_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_brain_init_medulla_health_agent, operation, progress);
    }
}


//=============================================================================
// Main Initialization Function
//=============================================================================

bool nimcp_brain_factory_init_medulla_subsystem(brain_t brain) {
    if (!brain) {
        NIMCP_LOGGING_ERROR("Null brain in init_medulla_subsystem");
        return false;
    }

    /* Initialize medulla fields to defaults */
    brain->medulla = NULL;
    brain->medulla_enabled = false;
    brain->last_medulla_update_us = 0;

    /* Medulla should always be enabled - it's foundational */
    bool should_enable = true;

    if (!should_enable) {
        NIMCP_LOGGING_DEBUG("Medulla not enabled");
        return true;  /* Not an error - just not needed */
    }

    /* Create medulla configuration */
    medulla_config_t config = medulla_default_config();

    /* Configure based on brain settings */
    config.enable_bio_async = brain->bio_async_enabled;

    /* Set biologically-plausible update interval (50ms = 20Hz) */
    config.update_interval_ms = 50;

    /* Configure arousal for balanced baseline */
    config.arousal.baseline_arousal = 0.5f;
    config.arousal.arousal_decay_rate = 0.02f;
    config.arousal.min_arousal = 0.1f;
    config.arousal.max_arousal = 0.95f;

    /* Configure protection for reasonable thresholds */
    config.protection.health_threshold_critical = 0.2f;
    config.protection.health_threshold_defensive = 0.4f;
    config.protection.recovery_time_ms = 1000.0f;

    /* Configure circadian for 24-hour cycle */
    config.circadian.period_hours = 24.0f;
    config.circadian.phase_offset_hours = 0.0f;
    config.circadian.amplitude = 0.3f;

    /* Configure coupling for moderate strength */
    config.coupling.coupling_strength = 0.5f;
    config.coupling.latency_ms = 10.0f;

    /* Create medulla */
    medulla_t medulla = medulla_create(&config);
    if (!medulla) {
        NIMCP_LOGGING_WARN("Failed to create medulla - continuing without brainstem regulation");
        return true;  /* Non-fatal for now, but limits functionality */
    }

    /* Connect to bio-async if available */
    if (brain->bio_async_enabled) {
        if (medulla_connect_bio_async(medulla) == 0) {
            NIMCP_LOGGING_DEBUG("Medulla connected to bio-async router");
        } else {
            NIMCP_LOGGING_WARN("Failed to connect medulla to bio-async");
        }
    }

    /* Start the medulla */
    int result = medulla_start(medulla);
    if (result != NIMCP_SUCCESS) {
        NIMCP_LOGGING_WARN("Failed to start medulla (error %d)", result);
        medulla_destroy(medulla);
        return true;  /* Non-fatal */
    }

    /* Store in brain */
    brain->medulla = medulla;
    brain->medulla_enabled = true;
    brain->last_medulla_update_us = brain->current_time_us;

    /* Log initialization */
    medulla_stats_t stats;
    medulla_get_stats(medulla, &stats);

    NIMCP_LOGGING_INFO("Medulla oblongata subsystem initialized successfully");
    NIMCP_LOGGING_INFO("  State: %s",
        stats.state == MEDULLA_STATE_RUNNING ? "RUNNING" :
        stats.state == MEDULLA_STATE_STOPPED ? "STOPPED" :
        stats.state == MEDULLA_STATE_DEGRADED ? "DEGRADED" : "UNKNOWN");
    NIMCP_LOGGING_INFO("  Arousal: %.2f (baseline: %.2f)",
        stats.current_arousal, config.arousal.baseline_arousal);
    NIMCP_LOGGING_INFO("  Protection Level: %s",
        stats.protection_level == PROTECTION_LEVEL_NORMAL ? "NORMAL" :
        stats.protection_level == PROTECTION_LEVEL_CAUTIOUS ? "CAUTIOUS" :
        stats.protection_level == PROTECTION_LEVEL_GUARDED ? "GUARDED" :
        stats.protection_level == PROTECTION_LEVEL_DEFENSIVE ? "DEFENSIVE" :
        stats.protection_level == PROTECTION_LEVEL_CRITICAL ? "CRITICAL" : "SHUTDOWN");
    NIMCP_LOGGING_INFO("  Circadian Phase: %d (%.1f hours)",
        (int)medulla_get_circadian_phase(medulla), stats.circadian_time_hours);
    NIMCP_LOGGING_INFO("  Bio-Async: %s",
        medulla_is_bio_async_connected(medulla) ? "CONNECTED" : "DISABLED");

    return true;
}

//=============================================================================
// Update Function
//=============================================================================

int nimcp_brain_update_medulla_subsystem(brain_t brain, float delta_time_s) {
    if (!brain) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    if (!brain->medulla_enabled || !brain->medulla) {
        return NIMCP_SUCCESS;  /* No medulla to update */
    }

    /* Update medulla */
    int result = medulla_update(brain->medulla, delta_time_s);
    if (result != NIMCP_SUCCESS) {
        NIMCP_LOGGING_WARN("Medulla update failed (error %d)", result);
        return result;
    }

    /* Update timestamp */
    uint64_t delta_us = (uint64_t)(delta_time_s * 1000000.0f);
    brain->last_medulla_update_us = brain->current_time_us;

    return NIMCP_SUCCESS;
}

//=============================================================================
// Destruction Function
//=============================================================================

void nimcp_brain_destroy_medulla_subsystem(brain_t brain) {
    if (!brain) {
        return;
    }

    if (brain->medulla) {
        /* Stop medulla first */
        medulla_stop(brain->medulla);

        /* Destroy medulla */
        medulla_destroy(brain->medulla);
        brain->medulla = NULL;

        NIMCP_LOGGING_DEBUG("Medulla subsystem destroyed");
    }

    brain->medulla_enabled = false;
    brain->last_medulla_update_us = 0;
}

//=============================================================================
// State Query Functions
//=============================================================================

float nimcp_brain_get_arousal_level(brain_t brain) {
    if (!brain || !brain->medulla_enabled || !brain->medulla) {
        return 0.5f;  /* Default to neutral arousal */
    }

    medulla_stats_t stats;
    if (medulla_get_stats(brain->medulla, &stats) != NIMCP_SUCCESS) {
        return 0.5f;
    }

    return stats.current_arousal;
}

circadian_phase_t nimcp_brain_get_circadian_phase(brain_t brain) {
    if (!brain || !brain->medulla_enabled || !brain->medulla) {
        return CIRCADIAN_PHASE_MORNING;  /* Default to morning */
    }

    return medulla_get_circadian_phase(brain->medulla);
}

protection_level_t nimcp_brain_get_protection_level(brain_t brain) {
    if (!brain || !brain->medulla_enabled || !brain->medulla) {
        return PROTECTION_LEVEL_NORMAL;  /* Default to normal */
    }

    return medulla_get_protection_level(brain->medulla);
}

bool nimcp_brain_is_medulla_emergency(brain_t brain) {
    if (!brain || !brain->medulla_enabled || !brain->medulla) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM,

                "nimcp_brain_is_medulla_emergency: invalid parameters");

            return false;
    }

    medulla_stats_t stats;
    if (medulla_get_stats(brain->medulla, &stats) != NIMCP_SUCCESS) {
        return false;
    }

    return (stats.state == MEDULLA_STATE_EMERGENCY ||
            stats.protection_level >= PROTECTION_LEVEL_CRITICAL);
}

medulla_t brain_get_medulla(brain_t brain) {
    if (!brain || !brain->medulla_enabled) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM,

                "brain_get_medulla: invalid parameters");

            return NULL;
    }
    return brain->medulla;
}

//=============================================================================
// Control Functions
//=============================================================================

int nimcp_brain_trigger_emergency(brain_t brain, const char* reason) {
    if (!brain) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    if (!brain->medulla_enabled || !brain->medulla) {
        NIMCP_LOGGING_WARN("Cannot trigger emergency - medulla not enabled");
        return NIMCP_ERROR_INVALID_STATE;
    }

    NIMCP_LOGGING_WARN("Brain triggering medulla emergency: %s",
        reason ? reason : "unspecified");

    return medulla_emergency_shutdown(brain->medulla, reason);
}

int nimcp_brain_request_medulla_state(brain_t brain, medulla_state_t new_state) {
    if (!brain) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    if (!brain->medulla_enabled || !brain->medulla) {
        NIMCP_LOGGING_WARN("Cannot change medulla state - medulla not enabled");
        return NIMCP_ERROR_INVALID_STATE;
    }

    return medulla_request_state_change(brain->medulla, new_state);
}
