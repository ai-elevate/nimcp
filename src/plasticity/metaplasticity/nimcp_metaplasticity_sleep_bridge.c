/**
 * @file nimcp_metaplasticity_sleep_bridge.c
 * @brief Sleep-Metaplasticity Integration Bridge Implementation
 * @version 1.0.0
 * @date 2025-12-19
 */

#include "plasticity/metaplasticity/nimcp_metaplasticity_sleep_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/platform/nimcp_platform_mutex.h"
#include <math.h>
#include <string.h>

/* ============================================================================
 * Helper Functions
 * ============================================================================ */

float metaplasticity_sleep_state_to_reset_factor(sleep_state_t state) {
    switch (state) {
        case SLEEP_STATE_AWAKE:
            return METAPLASTICITY_SLEEP_BRIDGE_RESET_AWAKE;
        case SLEEP_STATE_DROWSY:
            return METAPLASTICITY_SLEEP_BRIDGE_RESET_DROWSY;
        case SLEEP_STATE_LIGHT_NREM:
            return METAPLASTICITY_SLEEP_BRIDGE_RESET_LIGHT_NREM;
        case SLEEP_STATE_DEEP_NREM:
            return METAPLASTICITY_SLEEP_BRIDGE_RESET_DEEP_NREM;
        case SLEEP_STATE_REM:
            return METAPLASTICITY_SLEEP_BRIDGE_RESET_REM;
        default:
            return 0.0f;
    }
}

float metaplasticity_sleep_state_to_adapt_factor(sleep_state_t state) {
    switch (state) {
        case SLEEP_STATE_AWAKE:
            return METAPLASTICITY_SLEEP_ADAPT_AWAKE;
        case SLEEP_STATE_DROWSY:
            return METAPLASTICITY_SLEEP_ADAPT_DROWSY;
        case SLEEP_STATE_LIGHT_NREM:
            return METAPLASTICITY_SLEEP_ADAPT_LIGHT_NREM;
        case SLEEP_STATE_DEEP_NREM:
            return METAPLASTICITY_SLEEP_ADAPT_DEEP_NREM;
        case SLEEP_STATE_REM:
            return METAPLASTICITY_SLEEP_ADAPT_REM;
        default:
            return 1.0f;
    }
}

/* ============================================================================
 * Lifecycle API
 * ============================================================================ */

int metaplasticity_sleep_default_config(metaplasticity_sleep_config_t* config) {
    /* Guard clause */
    if (!config) {
        NIMCP_LOGGING_ERROR("NULL config in default_config");
        return -1;
    }

    config->enable_threshold_reset = true;
    config->enable_adaptation_freeze = true;
    config->enable_drift_monitoring = true;
    config->enable_sleep_pressure_feedback = true;
    config->reset_strength_multiplier = 1.0f;
    config->adaptation_strength_multiplier = 1.0f;

    return 0;
}

metaplasticity_sleep_bridge_t* metaplasticity_sleep_bridge_create(
    const metaplasticity_sleep_config_t* config,
    sleep_system_t sleep_system,
    metaplasticity_controller_t metaplasticity_controller
) {
    /* Guard clauses */
    if (!sleep_system || !metaplasticity_controller) {
        NIMCP_LOGGING_ERROR("NULL parameters in bridge create");
        return NULL;
    }

    /* Allocate bridge */
    metaplasticity_sleep_bridge_t* bridge =
        (metaplasticity_sleep_bridge_t*)nimcp_malloc(sizeof(metaplasticity_sleep_bridge_t));
    if (!bridge) {
        NIMCP_LOGGING_ERROR("Failed to allocate sleep bridge");
        return NULL;
    }

    memset(bridge, 0, sizeof(metaplasticity_sleep_bridge_t));

    /* Store system handles */
    bridge->sleep_system = sleep_system;
    bridge->metaplasticity_controller = metaplasticity_controller;

    /* Store configuration */
    if (config) {
        bridge->config = *config;
    } else {
        metaplasticity_sleep_default_config(&bridge->config);
    }

    /* Initialize effects */
    bridge->effects.current_state = SLEEP_STATE_AWAKE;
    bridge->effects.threshold_reset_factor = 0.0f;
    bridge->effects.adaptation_rate_factor = 1.0f;
    bridge->effects.adaptation_frozen = false;

    /* Allocate mutex */
    bridge->mutex = nimcp_malloc(sizeof(nimcp_platform_mutex_t));
    if (!bridge->mutex) {
        NIMCP_LOGGING_ERROR("Failed to allocate mutex");
        nimcp_free(bridge);
        return NULL;
    }

    if (nimcp_platform_mutex_init((nimcp_platform_mutex_t*)bridge->mutex, false) != 0) {
        NIMCP_LOGGING_ERROR("Failed to initialize mutex");
        nimcp_free(bridge->mutex);
        nimcp_free(bridge);
        return NULL;
    }

    NIMCP_LOGGING_INFO("Created metaplasticity-sleep bridge");

    return bridge;
}

void metaplasticity_sleep_bridge_destroy(metaplasticity_sleep_bridge_t* bridge) {
    if (!bridge) return;

    /* Destroy mutex */
    if (bridge->mutex) {
        nimcp_platform_mutex_destroy((nimcp_platform_mutex_t*)bridge->mutex);
        nimcp_free(bridge->mutex);
    }

    /* Free bridge */
    nimcp_free(bridge);

    NIMCP_LOGGING_INFO("Destroyed metaplasticity-sleep bridge");
}

/* ============================================================================
 * Sleep → Metaplasticity API
 * ============================================================================ */

int metaplasticity_sleep_apply_threshold_reset(metaplasticity_sleep_bridge_t* bridge) {
    /* Guard clause */
    if (!bridge) {
        NIMCP_LOGGING_ERROR("NULL bridge in apply_threshold_reset");
        return -1;
    }

    /* Skip if disabled */
    if (!bridge->config.enable_threshold_reset) {
        return 0;
    }

    nimcp_platform_mutex_lock((nimcp_platform_mutex_t*)bridge->mutex);

    /* Get current sleep state */
    sleep_state_t sleep_state = sleep_get_current_state(bridge->sleep_system);
    bridge->effects.current_state = sleep_state;

    /* Compute reset factor */
    float base_reset = metaplasticity_sleep_state_to_reset_factor(sleep_state);
    bridge->effects.threshold_reset_factor =
        base_reset * bridge->config.reset_strength_multiplier;

    /* Apply reset to metaplasticity controller */
    if (bridge->effects.threshold_reset_factor > 0.0f) {
        if (metaplasticity_controller_set_sleep_state(
                bridge->metaplasticity_controller, sleep_state) != 0) {
            nimcp_platform_mutex_unlock((nimcp_platform_mutex_t*)bridge->mutex);
            NIMCP_LOGGING_ERROR("Failed to apply sleep state");
            return -1;
        }

        /* Track resets */
        bridge->total_resets++;
        if (sleep_state == SLEEP_STATE_DEEP_NREM) {
            bridge->deep_nrem_resets++;
        } else if (sleep_state == SLEEP_STATE_REM) {
            bridge->rem_resets++;
        }

        NIMCP_LOGGING_DEBUG("Applied threshold reset: state=%d, factor=%.2f",
                           sleep_state, bridge->effects.threshold_reset_factor);
    }

    nimcp_platform_mutex_unlock((nimcp_platform_mutex_t*)bridge->mutex);

    return 0;
}

int metaplasticity_sleep_freeze_adaptation(metaplasticity_sleep_bridge_t* bridge) {
    /* Guard clause */
    if (!bridge) {
        NIMCP_LOGGING_ERROR("NULL bridge in freeze_adaptation");
        return -1;
    }

    /* Skip if disabled */
    if (!bridge->config.enable_adaptation_freeze) {
        bridge->effects.adaptation_frozen = false;
        bridge->effects.adaptation_rate_factor = 1.0f;
        return 0;
    }

    nimcp_platform_mutex_lock((nimcp_platform_mutex_t*)bridge->mutex);

    /* Get current sleep state */
    sleep_state_t sleep_state = sleep_get_current_state(bridge->sleep_system);

    /* Compute adaptation factor */
    float base_adapt = metaplasticity_sleep_state_to_adapt_factor(sleep_state);
    bridge->effects.adaptation_rate_factor =
        base_adapt * bridge->config.adaptation_strength_multiplier;

    /* Check if frozen */
    bridge->effects.adaptation_frozen =
        (sleep_state == SLEEP_STATE_DEEP_NREM) &&
        (bridge->effects.adaptation_rate_factor < 0.1f);

    nimcp_platform_mutex_unlock((nimcp_platform_mutex_t*)bridge->mutex);

    return 0;
}

float metaplasticity_sleep_get_adaptation_rate(
    const metaplasticity_sleep_bridge_t* bridge,
    float base_rate
) {
    if (!bridge) return base_rate;
    return base_rate * bridge->effects.adaptation_rate_factor;
}

float metaplasticity_sleep_get_reset_factor(const metaplasticity_sleep_bridge_t* bridge) {
    if (!bridge) return 0.0f;
    return bridge->effects.threshold_reset_factor;
}

/* ============================================================================
 * Metaplasticity → Sleep API
 * ============================================================================ */

int metaplasticity_sleep_monitor_threshold_drift(
    metaplasticity_sleep_bridge_t* bridge,
    float* drift
) {
    /* Guard clauses */
    if (!bridge || !drift) {
        NIMCP_LOGGING_ERROR("NULL parameters in monitor_threshold_drift");
        return -1;
    }

    /* Skip if disabled */
    if (!bridge->config.enable_drift_monitoring) {
        *drift = 0.0f;
        return 0;
    }

    nimcp_platform_mutex_lock((nimcp_platform_mutex_t*)bridge->mutex);

    /* Get metaplasticity statistics */
    metaplasticity_stats_t stats;
    if (metaplasticity_controller_get_stats(bridge->metaplasticity_controller, &stats) != 0) {
        nimcp_platform_mutex_unlock((nimcp_platform_mutex_t*)bridge->mutex);
        NIMCP_LOGGING_ERROR("Failed to get metaplasticity stats");
        return -1;
    }

    /* Compute drift as difference between effective and baseline */
    float current_drift = fabsf(stats.mean_theta_effective - stats.mean_theta_baseline);
    *drift = current_drift;

    /* Update bridge state */
    bridge->effects.baseline_drift = current_drift;
    bridge->total_drift_accumulated += current_drift;

    if (current_drift > bridge->max_drift_observed) {
        bridge->max_drift_observed = current_drift;
    }

    nimcp_platform_mutex_unlock((nimcp_platform_mutex_t*)bridge->mutex);

    NIMCP_LOGGING_DEBUG("Threshold drift: %.3f (baseline: %.3f, effective: %.3f)",
                       current_drift, stats.mean_theta_baseline,
                       stats.mean_theta_effective);

    return 0;
}

int metaplasticity_sleep_compute_sleep_pressure(
    metaplasticity_sleep_bridge_t* bridge,
    float* sleep_pressure
) {
    /* Guard clauses */
    if (!bridge || !sleep_pressure) {
        NIMCP_LOGGING_ERROR("NULL parameters in compute_sleep_pressure");
        return -1;
    }

    /* Skip if disabled */
    if (!bridge->config.enable_sleep_pressure_feedback) {
        *sleep_pressure = 0.0f;
        return 0;
    }

    nimcp_platform_mutex_lock((nimcp_platform_mutex_t*)bridge->mutex);

    /* Compute pressure from drift */
    float drift_ratio = bridge->effects.baseline_drift /
                        METAPLASTICITY_THRESHOLD_DRIFT_TOLERANCE;

    /* Clamp to [0, 1] */
    if (drift_ratio < 0.0f) drift_ratio = 0.0f;
    if (drift_ratio > 1.0f) drift_ratio = 1.0f;

    /* Store in effects */
    bridge->effects.sleep_pressure_contribution = drift_ratio;
    *sleep_pressure = drift_ratio;

    nimcp_platform_mutex_unlock((nimcp_platform_mutex_t*)bridge->mutex);

    if (drift_ratio > 0.5f) {
        NIMCP_LOGGING_DEBUG("High sleep pressure from threshold drift: %.2f", drift_ratio);
    }

    return 0;
}

bool metaplasticity_sleep_is_reset_complete(const metaplasticity_sleep_bridge_t* bridge) {
    if (!bridge) return false;
    return bridge->effects.baseline_drift < (METAPLASTICITY_THRESHOLD_DRIFT_TOLERANCE * 0.1f);
}

/* ============================================================================
 * Bidirectional Update API
 * ============================================================================ */

int metaplasticity_sleep_bridge_update(metaplasticity_sleep_bridge_t* bridge) {
    /* Guard clause */
    if (!bridge) {
        NIMCP_LOGGING_ERROR("NULL bridge in bridge_update");
        return -1;
    }

    /* Apply sleep → metaplasticity effects */
    if (metaplasticity_sleep_apply_threshold_reset(bridge) != 0) {
        NIMCP_LOGGING_WARN("Failed to apply threshold reset");
    }

    if (metaplasticity_sleep_freeze_adaptation(bridge) != 0) {
        NIMCP_LOGGING_WARN("Failed to freeze adaptation");
    }

    /* Monitor metaplasticity → sleep feedback */
    float drift;
    if (metaplasticity_sleep_monitor_threshold_drift(bridge, &drift) != 0) {
        NIMCP_LOGGING_WARN("Failed to monitor drift");
    }

    float sleep_pressure;
    if (metaplasticity_sleep_compute_sleep_pressure(bridge, &sleep_pressure) != 0) {
        NIMCP_LOGGING_WARN("Failed to compute sleep pressure");
    }

    return 0;
}

/* ============================================================================
 * Query API
 * ============================================================================ */

int metaplasticity_sleep_get_effects(
    const metaplasticity_sleep_bridge_t* bridge,
    metaplasticity_sleep_effects_t* effects
) {
    /* Guard clauses */
    if (!bridge || !effects) {
        NIMCP_LOGGING_ERROR("NULL parameters in get_effects");
        return -1;
    }

    *effects = bridge->effects;
    return 0;
}

int metaplasticity_sleep_get_drift_stats(
    const metaplasticity_sleep_bridge_t* bridge,
    float* mean_drift,
    float* max_drift
) {
    /* Guard clauses */
    if (!bridge) {
        NIMCP_LOGGING_ERROR("NULL bridge in get_drift_stats");
        return -1;
    }

    if (mean_drift) {
        *mean_drift = bridge->effects.baseline_drift;
    }

    if (max_drift) {
        *max_drift = bridge->max_drift_observed;
    }

    return 0;
}

bool metaplasticity_sleep_is_adaptation_frozen(const metaplasticity_sleep_bridge_t* bridge) {
    if (!bridge) return false;
    return bridge->effects.adaptation_frozen;
}

uint64_t metaplasticity_sleep_get_reset_count(const metaplasticity_sleep_bridge_t* bridge) {
    if (!bridge) return 0;
    return bridge->total_resets;
}
