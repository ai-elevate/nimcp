//=============================================================================
// nimcp_dragonfly_medulla_bridge.c - Dragonfly-Medulla Integration Bridge
//=============================================================================
/**
 * @file nimcp_dragonfly_medulla_bridge.c
 * @brief Implementation of dragonfly-medulla integration bridge
 *
 * BIOLOGICAL RATIONALE:
 * This bridge implements the biological reality that hunting behavior in
 * dragonflies (and all predators) is modulated by:
 * - Arousal state: Alert animals hunt better than drowsy ones
 * - Circadian rhythm: Diurnal hunters are inactive at night
 * - Stress/protection: Threatened animals prioritize escape over hunting
 *
 * @version 1.0.0
 * @author NIMCP Development Team
 * @date 2025-12-29
 */

#include "dragonfly/nimcp_dragonfly_medulla_bridge.h"
#include "dragonfly/nimcp_dragonfly.h"
#include "core/medulla/nimcp_medulla.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/time/nimcp_time.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

//=============================================================================
// Internal Structure
//=============================================================================

struct dragonfly_medulla_bridge_s {
    /* Configuration */
    dragonfly_medulla_config_t config;

    /* Connected systems */
    dragonfly_system_t* dragonfly;
    medulla_t medulla;
    bool connected;

    /* Current modulation state */
    dragonfly_medulla_modulation_t modulation;

    /* Statistics */
    dragonfly_medulla_stats_t stats;

    /* Timing */
    uint64_t last_update_us;
    uint64_t creation_time_us;

    /* Thread safety */
    nimcp_mutex_t* mutex;
};

//=============================================================================
// Arousal Modulation Tables
//=============================================================================

/**
 * Arousal level modulation factors
 * Index corresponds to arousal_level_t enum value
 */
static const float AROUSAL_NAV_GAIN_FACTOR[] = {
    0.0f,   /* COMA */
    0.0f,   /* DEEP_SLEEP */
    0.2f,   /* LIGHT_SLEEP */
    0.5f,   /* DROWSY */
    1.0f,   /* AWAKE */
    1.2f,   /* ALERT */
    1.5f    /* HYPERAROUSAL */
};

static const float AROUSAL_URGENCY_FACTOR[] = {
    0.0f,   /* COMA */
    0.0f,   /* DEEP_SLEEP */
    0.1f,   /* LIGHT_SLEEP */
    0.3f,   /* DROWSY */
    1.0f,   /* AWAKE */
    1.2f,   /* ALERT */
    1.5f    /* HYPERAROUSAL */
};

static const float AROUSAL_REACTION_FACTOR[] = {
    0.0f,   /* COMA */
    0.0f,   /* DEEP_SLEEP */
    0.3f,   /* LIGHT_SLEEP */
    0.6f,   /* DROWSY */
    1.0f,   /* AWAKE */
    1.3f,   /* ALERT */
    1.5f    /* HYPERAROUSAL */
};

static const float AROUSAL_ACCURACY_FACTOR[] = {
    0.0f,   /* COMA */
    0.0f,   /* DEEP_SLEEP */
    0.5f,   /* LIGHT_SLEEP */
    0.7f,   /* DROWSY */
    1.0f,   /* AWAKE */
    1.1f,   /* ALERT */
    0.8f    /* HYPERAROUSAL - jittery, less accurate */
};

//=============================================================================
// Circadian Modulation Table
//=============================================================================

/**
 * Circadian phase performance factors
 * Dragonflies are diurnal - active during day, inactive at night
 * Index corresponds to circadian_phase_t enum value
 */
static const float CIRCADIAN_PERFORMANCE_FACTOR[] = {
    0.7f,   /* EARLY_MORNING - warming up */
    1.0f,   /* MORNING - peak hunting */
    0.85f,  /* AFTERNOON - post-peak dip */
    0.95f,  /* EVENING - second peak */
    0.7f,   /* LATE_EVENING - declining */
    0.3f,   /* NIGHT - mostly inactive */
    0.1f,   /* DEEP_NIGHT - minimal activity */
    0.4f    /* PRE_DAWN - starting to wake */
};

//=============================================================================
// Protection Level Effects
//=============================================================================

/**
 * Protection level effects on hunting
 */
typedef struct {
    bool hunting_allowed;
    bool should_abort;
    float duration_scale;
} protection_effect_t;

static const protection_effect_t PROTECTION_EFFECTS[] = {
    { true,  false, 1.0f },  /* NORMAL */
    { true,  false, 0.9f },  /* CAUTIOUS */
    { true,  false, 0.7f },  /* GUARDED */
    { true,  false, 0.5f },  /* DEFENSIVE - limited hunting */
    { false, true,  0.0f },  /* CRITICAL - abort */
    { false, true,  0.0f }   /* SHUTDOWN - abort */
};

//=============================================================================
// Helper Functions
//=============================================================================

static uint64_t get_time_us(void) {
    return nimcp_time_get_us();
}

static float clampf(float value, float min, float max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
}

//=============================================================================
// Lifecycle Functions
//=============================================================================

dragonfly_medulla_config_t dragonfly_medulla_default_config(void) {
    dragonfly_medulla_config_t config = {
        /* Arousal modulation */
        .arousal_nav_gain_min = 0.0f,
        .arousal_nav_gain_max = 1.5f,
        .arousal_urgency_scale_min = 0.0f,
        .arousal_urgency_scale_max = 1.5f,
        .arousal_reaction_min = 0.0f,
        .arousal_reaction_max = 1.5f,

        /* Hyperarousal penalty */
        .hyperarousal_accuracy_penalty = 0.2f,

        /* Protection thresholds */
        .abort_on_critical = true,
        .abort_on_shutdown = true,
        .defensive_duration_scale = 0.5f,
        .guarded_duration_scale = 0.7f,
        .cautious_duration_scale = 0.9f,

        /* Circadian */
        .enable_circadian_modulation = true,
        .night_performance_floor = 0.1f,

        /* Feedback */
        .enable_arousal_feedback = true,
        .pursuit_arousal_boost = 0.1f,
        .intercept_arousal_boost = 0.2f,
        .failure_arousal_penalty = 0.05f,

        /* Update settings */
        .update_interval_ms = 50,
        .enable_logging = false
    };
    return config;
}

dragonfly_medulla_bridge_t dragonfly_medulla_bridge_create(
    const dragonfly_medulla_config_t* config
) {
    dragonfly_medulla_bridge_t bridge = nimcp_calloc(1, sizeof(struct dragonfly_medulla_bridge_s));
    if (!bridge) {
        return NULL;
    }

    /* Apply configuration */
    if (config) {
        bridge->config = *config;
    } else {
        bridge->config = dragonfly_medulla_default_config();
    }

    /* Create mutex */
    mutex_attr_t attr = { .type = MUTEX_TYPE_NORMAL };
    bridge->mutex = nimcp_mutex_create(&attr);
    if (!bridge->mutex) {
        nimcp_free(bridge);
        return NULL;
    }

    /* Initialize state */
    bridge->dragonfly = NULL;
    bridge->medulla = NULL;
    bridge->connected = false;
    bridge->creation_time_us = get_time_us();
    bridge->last_update_us = 0;

    /* Initialize modulation to defaults */
    bridge->modulation.nav_gain_scale = 1.0f;
    bridge->modulation.urgency_scale = 1.0f;
    bridge->modulation.reaction_scale = 1.0f;
    bridge->modulation.accuracy_scale = 1.0f;
    bridge->modulation.max_duration_scale = 1.0f;
    bridge->modulation.hunting_allowed = true;
    bridge->modulation.should_abort = false;

    /* Clear statistics */
    memset(&bridge->stats, 0, sizeof(bridge->stats));

    if (bridge->config.enable_logging) {
        fprintf(stderr, "[DRAGONFLY-MEDULLA] Bridge created\n");
    }

    return bridge;
}

void dragonfly_medulla_bridge_destroy(dragonfly_medulla_bridge_t bridge) {
    if (!bridge) return;

    if (bridge->connected) {
        dragonfly_medulla_bridge_disconnect(bridge);
    }

    if (bridge->mutex) {
        nimcp_mutex_destroy(bridge->mutex);
    }

    nimcp_free(bridge);
}

//=============================================================================
// Connection Functions
//=============================================================================

int dragonfly_medulla_bridge_connect(
    dragonfly_medulla_bridge_t bridge,
    dragonfly_system_t* dragonfly,
    medulla_t medulla
) {
    if (!bridge || !dragonfly || !medulla) {
        return -1;
    }

    nimcp_mutex_lock(bridge->mutex);

    if (bridge->connected) {
        nimcp_mutex_unlock(bridge->mutex);
        return -1;  /* Already connected */
    }

    bridge->dragonfly = dragonfly;
    bridge->medulla = medulla;
    bridge->connected = true;
    bridge->stats.is_connected = true;
    bridge->stats.connection_time_us = get_time_us();

    if (bridge->config.enable_logging) {
        fprintf(stderr, "[DRAGONFLY-MEDULLA] Bridge connected to dragonfly and medulla\n");
    }

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int dragonfly_medulla_bridge_disconnect(dragonfly_medulla_bridge_t bridge) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);

    bridge->dragonfly = NULL;
    bridge->medulla = NULL;
    bridge->connected = false;
    bridge->stats.is_connected = false;

    if (bridge->config.enable_logging) {
        fprintf(stderr, "[DRAGONFLY-MEDULLA] Bridge disconnected\n");
    }

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

bool dragonfly_medulla_bridge_is_connected(const dragonfly_medulla_bridge_t bridge) {
    if (!bridge) return false;
    return bridge->connected;
}

//=============================================================================
// Update Function
//=============================================================================

int dragonfly_medulla_bridge_update(dragonfly_medulla_bridge_t bridge, float dt) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);

    if (!bridge->connected) {
        nimcp_mutex_unlock(bridge->mutex);
        return -1;
    }

    uint64_t start_time = get_time_us();

    /* Get current medulla states */
    float arousal = medulla_get_arousal_level(bridge->medulla);
    protection_level_t protection = medulla_get_protection_level(bridge->medulla);
    circadian_phase_t circadian = medulla_get_circadian_phase(bridge->medulla);

    /* Clamp arousal to valid range and convert to level */
    arousal = clampf(arousal, 0.0f, 1.0f);

    /* Map continuous arousal [0-1] to discrete level [0-6] */
    int arousal_level = (int)(arousal * 6.0f + 0.5f);
    if (arousal_level > 6) arousal_level = 6;

    /* Store source states */
    bridge->modulation.arousal_level = arousal_level;
    bridge->modulation.protection_level = (int)protection;
    bridge->modulation.circadian_phase = (int)circadian;

    /* =====================================================================
     * AROUSAL MODULATION
     * ===================================================================== */

    float arousal_nav = AROUSAL_NAV_GAIN_FACTOR[arousal_level];
    float arousal_urgency = AROUSAL_URGENCY_FACTOR[arousal_level];
    float arousal_reaction = AROUSAL_REACTION_FACTOR[arousal_level];
    float arousal_accuracy = AROUSAL_ACCURACY_FACTOR[arousal_level];

    bridge->stats.current_arousal_modifier = arousal_nav;

    /* =====================================================================
     * CIRCADIAN MODULATION
     * ===================================================================== */

    float circadian_mod = 1.0f;
    if (bridge->config.enable_circadian_modulation) {
        circadian_mod = CIRCADIAN_PERFORMANCE_FACTOR[circadian];
        /* Apply floor for night performance */
        if (circadian_mod < bridge->config.night_performance_floor) {
            circadian_mod = bridge->config.night_performance_floor;
        }
    }

    bridge->stats.current_circadian_modifier = circadian_mod;

    /* =====================================================================
     * PROTECTION MODULATION
     * ===================================================================== */

    protection_effect_t prot_effect = PROTECTION_EFFECTS[protection];
    bridge->modulation.hunting_allowed = prot_effect.hunting_allowed;
    bridge->modulation.should_abort = prot_effect.should_abort;
    bridge->modulation.max_duration_scale = prot_effect.duration_scale;

    bridge->stats.current_protection_modifier = prot_effect.duration_scale;

    /* Check if we should abort current pursuit */
    if (bridge->modulation.should_abort) {
        /* Abort the dragonfly pursuit */
        dragonfly_abort_pursuit(bridge->dragonfly);
        bridge->stats.pursuits_aborted_protection++;

        if (bridge->config.enable_logging) {
            fprintf(stderr, "[DRAGONFLY-MEDULLA] Pursuit aborted due to protection level %d\n",
                    (int)protection);
        }
    }

    /* =====================================================================
     * COMBINED MODULATION
     * ===================================================================== */

    /* Combine arousal and circadian for final modifiers */
    bridge->modulation.nav_gain_scale = arousal_nav * circadian_mod;
    bridge->modulation.urgency_scale = arousal_urgency * circadian_mod;
    bridge->modulation.reaction_scale = arousal_reaction * circadian_mod;
    bridge->modulation.accuracy_scale = arousal_accuracy * circadian_mod;

    /* Apply protection duration scaling */
    bridge->modulation.max_duration_scale *= circadian_mod;

    /* Store effective scales */
    bridge->stats.effective_nav_gain_scale = bridge->modulation.nav_gain_scale;
    bridge->stats.effective_urgency_scale = bridge->modulation.urgency_scale;

    /* =====================================================================
     * APPLY TO DRAGONFLY SYSTEM
     * ===================================================================== */

    /* Apply modulations to dragonfly system */
    if (bridge->dragonfly) {
        /* Get current dragonfly config */
        dragonfly_config_t dragonfly_config;
        if (dragonfly_get_config(bridge->dragonfly, &dragonfly_config) == 0) {
            /* Scale navigation gain by arousal and circadian modulation */
            dragonfly_config.intercept_config.pn_gain =
                intercept_default_config().pn_gain * bridge->modulation.nav_gain_scale;

            /* Scale pursuit timeout by protection and circadian */
            dragonfly_config.pursuit_timeout_s =
                dragonfly_default_config().pursuit_timeout_s * bridge->modulation.max_duration_scale;

            /* Adjust confidence thresholds based on arousal */
            if (bridge->modulation.arousal_level >= 5) {  /* ALERT or HYPERAROUSAL */
                /* Lower thresholds when alert - more aggressive targeting */
                dragonfly_config.lock_threshold *= 0.8f;
                dragonfly_config.pursue_threshold *= 0.8f;
            } else if (bridge->modulation.arousal_level <= 3) {  /* DROWSY or below */
                /* Higher thresholds when drowsy - more conservative */
                dragonfly_config.lock_threshold *= 1.3f;
                dragonfly_config.pursue_threshold *= 1.3f;
            }

            /* Apply hyperarousal accuracy penalty */
            if (bridge->modulation.arousal_level == 6) {  /* HYPERAROUSAL */
                /* Reduce intercept precision due to jitteriness */
                dragonfly_config.intercept_threshold *= (1.0f + bridge->config.hyperarousal_accuracy_penalty);
            }

            /* Update dragonfly configuration */
            dragonfly_set_config(bridge->dragonfly, &dragonfly_config);
        }
    }

    /* =====================================================================
     * UPDATE STATISTICS
     * ===================================================================== */

    bridge->stats.total_updates++;
    bridge->last_update_us = start_time;

    uint64_t elapsed = get_time_us() - start_time;
    float update_time = (float)elapsed;

    /* Running average of update time */
    if (bridge->stats.total_updates == 1) {
        bridge->stats.avg_update_time_us = update_time;
    } else {
        bridge->stats.avg_update_time_us =
            0.99f * bridge->stats.avg_update_time_us + 0.01f * update_time;
    }
    bridge->stats.last_update_us = start_time;

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int dragonfly_medulla_bridge_get_modulation(
    const dragonfly_medulla_bridge_t bridge,
    dragonfly_medulla_modulation_t* modulation
) {
    if (!bridge || !modulation) return -1;

    nimcp_mutex_lock(bridge->mutex);
    *modulation = bridge->modulation;
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

//=============================================================================
// Feedback Functions
//=============================================================================

int dragonfly_medulla_bridge_notify_pursuit_start(dragonfly_medulla_bridge_t bridge) {
    if (!bridge || !bridge->connected) return -1;

    nimcp_mutex_lock(bridge->mutex);

    if (bridge->config.enable_arousal_feedback && bridge->medulla) {
        /* Pursuit increases arousal - send boost to medulla */
        medulla_boost_arousal(bridge->medulla, bridge->config.pursuit_arousal_boost);

        bridge->stats.arousal_boosts_sent++;
        bridge->stats.total_arousal_contribution += bridge->config.pursuit_arousal_boost;

        if (bridge->config.enable_logging) {
            fprintf(stderr, "[DRAGONFLY-MEDULLA] Pursuit started - arousal boost %.2f\n",
                    bridge->config.pursuit_arousal_boost);
        }
    }

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int dragonfly_medulla_bridge_notify_intercept_success(dragonfly_medulla_bridge_t bridge) {
    if (!bridge || !bridge->connected) return -1;

    nimcp_mutex_lock(bridge->mutex);

    if (bridge->config.enable_arousal_feedback && bridge->medulla) {
        /* Success gives bigger arousal boost */
        medulla_boost_arousal(bridge->medulla, bridge->config.intercept_arousal_boost);

        bridge->stats.arousal_boosts_sent++;
        bridge->stats.total_arousal_contribution += bridge->config.intercept_arousal_boost;

        if (bridge->config.enable_logging) {
            fprintf(stderr, "[DRAGONFLY-MEDULLA] Intercept success - arousal boost %.2f\n",
                    bridge->config.intercept_arousal_boost);
        }
    }

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int dragonfly_medulla_bridge_notify_pursuit_failure(
    dragonfly_medulla_bridge_t bridge,
    const char* reason
) {
    if (!bridge || !bridge->connected) return -1;

    nimcp_mutex_lock(bridge->mutex);

    if (bridge->config.enable_arousal_feedback && bridge->medulla) {
        /* Failure slightly decreases arousal */
        medulla_reduce_arousal(bridge->medulla, bridge->config.failure_arousal_penalty);

        bridge->stats.total_arousal_contribution -= bridge->config.failure_arousal_penalty;

        if (bridge->config.enable_logging) {
            fprintf(stderr, "[DRAGONFLY-MEDULLA] Pursuit failed (%s) - arousal penalty %.2f\n",
                    reason ? reason : "unknown", bridge->config.failure_arousal_penalty);
        }
    }

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

//=============================================================================
// Query Functions
//=============================================================================

bool dragonfly_medulla_bridge_hunting_allowed(const dragonfly_medulla_bridge_t bridge) {
    if (!bridge || !bridge->connected) return false;

    /* Quick check without lock for performance */
    return bridge->modulation.hunting_allowed;
}

int dragonfly_medulla_bridge_get_stats(
    const dragonfly_medulla_bridge_t bridge,
    dragonfly_medulla_stats_t* stats
) {
    if (!bridge || !stats) return -1;

    nimcp_mutex_lock(bridge->mutex);
    *stats = bridge->stats;
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

int dragonfly_medulla_bridge_reset_stats(dragonfly_medulla_bridge_t bridge) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);

    bool was_connected = bridge->stats.is_connected;
    uint64_t conn_time = bridge->stats.connection_time_us;

    memset(&bridge->stats, 0, sizeof(bridge->stats));

    /* Preserve connection state */
    bridge->stats.is_connected = was_connected;
    bridge->stats.connection_time_us = conn_time;

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}
