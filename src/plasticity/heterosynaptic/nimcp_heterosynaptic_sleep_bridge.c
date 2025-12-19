/**
 * @file nimcp_heterosynaptic_sleep_bridge.c
 * @brief Sleep-Heterosynaptic Integration Bridge Implementation
 * @version 1.0.0
 * @date 2025-12-19
 */

#include "plasticity/heterosynaptic/nimcp_heterosynaptic_sleep_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/error/nimcp_error_codes.h"
#include <string.h>

/* ============================================================================
 * Internal Structure
 * ============================================================================ */

struct hetero_sleep_bridge_struct {
    sleep_system_t sleep_system;
    hetero_system_t* hetero_system;
    hetero_sleep_config_t config;
    hetero_sleep_effects_t effects;
    nimcp_platform_mutex_t* mutex;
};

/* ============================================================================
 * Lifecycle API Implementation
 * ============================================================================ */

int hetero_sleep_default_config(hetero_sleep_config_t* config) {
    if (!config) return NIMCP_ERROR_NULL_POINTER;

    config->enable_competition_modulation = true;
    config->enable_depression_modulation = true;
    config->enable_radius_modulation = true;
    config->modulation_strength = 1.0f;

    return 0;
}

hetero_sleep_bridge_t hetero_sleep_bridge_create(
    const hetero_sleep_config_t* config,
    sleep_system_t sleep_system,
    hetero_system_t* hetero_system)
{
    if (!hetero_system) {
        NIMCP_LOGGING_ERROR("Heterosynaptic system is NULL");
        return NULL;
    }

    /* Use defaults if no config */
    hetero_sleep_config_t default_config;
    if (!config) {
        hetero_sleep_default_config(&default_config);
        config = &default_config;
    }

    /* Allocate bridge */
    hetero_sleep_bridge_t bridge = nimcp_malloc(sizeof(struct hetero_sleep_bridge_struct));
    if (!bridge) {
        NIMCP_LOGGING_ERROR("Failed to allocate hetero-sleep bridge");
        return NULL;
    }

    /* Initialize fields */
    bridge->sleep_system = sleep_system;
    bridge->hetero_system = hetero_system;
    memcpy(&bridge->config, config, sizeof(hetero_sleep_config_t));

    /* Initialize effects to awake state */
    bridge->effects.competition_factor = HETERO_SLEEP_COMPETITION_AWAKE;
    bridge->effects.depression_factor = HETERO_SLEEP_DEPRESSION_AWAKE;
    bridge->effects.radius_factor = HETERO_SLEEP_RADIUS_AWAKE;
    bridge->effects.current_state = SLEEP_STATE_AWAKE;
    bridge->effects.sleep_pressure = 0.0f;
    bridge->effects.competition_enabled = true;

    /* Create mutex */
    bridge->mutex = nimcp_platform_mutex_create();
    if (!bridge->mutex) {
        NIMCP_LOGGING_ERROR("Failed to create mutex");
        nimcp_free(bridge);
        return NULL;
    }

    NIMCP_LOGGING_INFO("Created hetero-sleep bridge");
    return bridge;
}

void hetero_sleep_bridge_destroy(hetero_sleep_bridge_t bridge) {
    if (!bridge) return;

    nimcp_platform_mutex_destroy(bridge->mutex);
    nimcp_free(bridge);

    NIMCP_LOGGING_INFO("Destroyed hetero-sleep bridge");
}

/* ============================================================================
 * Helper Functions Implementation
 * ============================================================================ */

float hetero_sleep_get_competition_factor_for_state(sleep_state_t state) {
    switch (state) {
        case SLEEP_STATE_AWAKE:
            return HETERO_SLEEP_COMPETITION_AWAKE;
        case SLEEP_STATE_DROWSY:
            return HETERO_SLEEP_COMPETITION_DROWSY;
        case SLEEP_STATE_LIGHT_NREM:
            return HETERO_SLEEP_COMPETITION_LIGHT_NREM;
        case SLEEP_STATE_DEEP_NREM:
            return HETERO_SLEEP_COMPETITION_DEEP_NREM;
        case SLEEP_STATE_REM:
            return HETERO_SLEEP_COMPETITION_REM;
        default:
            return HETERO_SLEEP_COMPETITION_AWAKE;
    }
}

float hetero_sleep_get_depression_factor_for_state(sleep_state_t state) {
    switch (state) {
        case SLEEP_STATE_AWAKE:
            return HETERO_SLEEP_DEPRESSION_AWAKE;
        case SLEEP_STATE_DROWSY:
            return HETERO_SLEEP_DEPRESSION_DROWSY;
        case SLEEP_STATE_LIGHT_NREM:
            return HETERO_SLEEP_DEPRESSION_LIGHT_NREM;
        case SLEEP_STATE_DEEP_NREM:
            return HETERO_SLEEP_DEPRESSION_DEEP_NREM;
        case SLEEP_STATE_REM:
            return HETERO_SLEEP_DEPRESSION_REM;
        default:
            return HETERO_SLEEP_DEPRESSION_AWAKE;
    }
}

float hetero_sleep_get_radius_factor_for_state(sleep_state_t state) {
    switch (state) {
        case SLEEP_STATE_AWAKE:
            return HETERO_SLEEP_RADIUS_AWAKE;
        case SLEEP_STATE_DROWSY:
            return HETERO_SLEEP_RADIUS_DROWSY;
        case SLEEP_STATE_LIGHT_NREM:
            return HETERO_SLEEP_RADIUS_LIGHT_NREM;
        case SLEEP_STATE_DEEP_NREM:
            return HETERO_SLEEP_RADIUS_DEEP_NREM;
        case SLEEP_STATE_REM:
            return HETERO_SLEEP_RADIUS_REM;
        default:
            return HETERO_SLEEP_RADIUS_AWAKE;
    }
}

/* ============================================================================
 * Update API Implementation
 * ============================================================================ */

int hetero_sleep_update(hetero_sleep_bridge_t bridge) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;

    nimcp_platform_mutex_lock(bridge->mutex);

    /* Get current sleep state */
    sleep_state_t state = SLEEP_STATE_AWAKE;
    if (bridge->sleep_system) {
        state = sleep_get_current_state(bridge->sleep_system);
    }

    /* Update effects */
    if (bridge->config.enable_competition_modulation) {
        float factor = hetero_sleep_get_competition_factor_for_state(state);
        bridge->effects.competition_factor = factor * bridge->config.modulation_strength +
                                             (1.0f - bridge->config.modulation_strength);
    } else {
        bridge->effects.competition_factor = 1.0f;
    }

    if (bridge->config.enable_depression_modulation) {
        float factor = hetero_sleep_get_depression_factor_for_state(state);
        bridge->effects.depression_factor = factor * bridge->config.modulation_strength +
                                            (1.0f - bridge->config.modulation_strength);
    } else {
        bridge->effects.depression_factor = 1.0f;
    }

    if (bridge->config.enable_radius_modulation) {
        float factor = hetero_sleep_get_radius_factor_for_state(state);
        bridge->effects.radius_factor = factor * bridge->config.modulation_strength +
                                        (1.0f - bridge->config.modulation_strength);
    } else {
        bridge->effects.radius_factor = 1.0f;
    }

    bridge->effects.current_state = state;

    /* Disable competition during deep sleep */
    bridge->effects.competition_enabled = (state != SLEEP_STATE_DEEP_NREM);

    /* Apply to heterosynaptic system */
    if (bridge->hetero_system) {
        hetero_set_sleep_state(bridge->hetero_system, state);
    }

    nimcp_platform_mutex_unlock(bridge->mutex);
    return 0;
}

int hetero_sleep_get_effects(const hetero_sleep_bridge_t bridge, hetero_sleep_effects_t* effects) {
    if (!bridge || !effects) return NIMCP_ERROR_NULL_POINTER;

    nimcp_platform_mutex_lock(bridge->mutex);
    memcpy(effects, &bridge->effects, sizeof(hetero_sleep_effects_t));
    nimcp_platform_mutex_unlock(bridge->mutex);

    return 0;
}

float hetero_sleep_get_competition_factor(const hetero_sleep_bridge_t bridge, float base_competition) {
    if (!bridge) return base_competition;
    return base_competition * bridge->effects.competition_factor;
}

float hetero_sleep_get_depression_factor(const hetero_sleep_bridge_t bridge, float base_depression) {
    if (!bridge) return base_depression;
    return base_depression * bridge->effects.depression_factor;
}

float hetero_sleep_get_radius(const hetero_sleep_bridge_t bridge, float base_radius) {
    if (!bridge) return base_radius;
    return base_radius * bridge->effects.radius_factor;
}
