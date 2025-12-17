/**
 * @file nimcp_bcm_sleep_bridge.c
 * @brief Sleep-BCM Integration Bridge Implementation
 * @version 1.0.0
 * @date 2025-12-17
 */

#include "plasticity/bcm/nimcp_bcm_sleep_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/platform/nimcp_platform_mutex.h"
#include <string.h>

struct bcm_sleep_bridge_struct {
    bcm_sleep_config_t config;
    sleep_system_t sleep_system;
    bcm_sleep_effects_t effects;
    nimcp_mutex_t* mutex;
};

int bcm_sleep_default_config(bcm_sleep_config_t* config) {
    if (!config) return -1;
    config->enable_theta_modulation = true;
    config->enable_lr_modulation = true;
    config->modulation_strength = 1.0f;
    return 0;
}

bcm_sleep_bridge_t bcm_sleep_bridge_create(
    const bcm_sleep_config_t* config,
    sleep_system_t sleep_system)
{
    if (!sleep_system) {
        NIMCP_LOGGING_ERROR("bcm_sleep_bridge_create: NULL sleep_system");
        return NULL;
    }

    struct bcm_sleep_bridge_struct* bridge =
        (struct bcm_sleep_bridge_struct*)nimcp_malloc(sizeof(struct bcm_sleep_bridge_struct));
    if (!bridge) return NULL;

    memset(bridge, 0, sizeof(struct bcm_sleep_bridge_struct));

    if (config) {
        bridge->config = *config;
    } else {
        bcm_sleep_default_config(&bridge->config);
    }

    bridge->sleep_system = sleep_system;
    bridge->effects.theta_factor = 1.0f;
    bridge->effects.learning_rate_factor = 1.0f;
    bridge->effects.favors_ltd = false;

    bridge->mutex = nimcp_platform_mutex_create();
    if (!bridge->mutex) {
        nimcp_free(bridge);
        return NULL;
    }

    NIMCP_LOGGING_INFO("BCM-sleep bridge created");
    return bridge;
}

void bcm_sleep_bridge_destroy(bcm_sleep_bridge_t bridge) {
    if (!bridge) return;
    if (bridge->mutex) nimcp_mutex_destroy(bridge->mutex);
    nimcp_free(bridge);
}

int bcm_sleep_update(bcm_sleep_bridge_t bridge) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);

    sleep_state_t state = sleep_get_current_state(bridge->sleep_system);
    float pressure = sleep_get_pressure(bridge->sleep_system);

    bridge->effects.current_state = state;
    bridge->effects.sleep_pressure = pressure;

    if (bridge->config.enable_theta_modulation) {
        float theta_base = bcm_sleep_theta_for_state(state);
        bridge->effects.theta_factor =
            1.0f + (theta_base - 1.0f) * bridge->config.modulation_strength;
    }

    if (bridge->config.enable_lr_modulation) {
        float lr_base = bcm_sleep_lr_for_state(state);
        bridge->effects.learning_rate_factor =
            1.0f + (lr_base - 1.0f) * bridge->config.modulation_strength;
    }

    /* Elevated theta favors LTD (activity must exceed higher threshold for LTP) */
    bridge->effects.favors_ltd = (bridge->effects.theta_factor > 1.0f);

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int bcm_sleep_get_effects(const bcm_sleep_bridge_t bridge, bcm_sleep_effects_t* effects) {
    if (!bridge || !effects) return -1;
    nimcp_mutex_lock(bridge->mutex);
    *effects = bridge->effects;
    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

float bcm_sleep_get_theta_factor(const bcm_sleep_bridge_t bridge) {
    if (!bridge) return 1.0f;
    nimcp_mutex_lock(bridge->mutex);
    float result = bridge->effects.theta_factor;
    nimcp_mutex_unlock(bridge->mutex);
    return result;
}

float bcm_sleep_get_lr_factor(const bcm_sleep_bridge_t bridge) {
    if (!bridge) return 1.0f;
    nimcp_mutex_lock(bridge->mutex);
    float result = bridge->effects.learning_rate_factor;
    nimcp_mutex_unlock(bridge->mutex);
    return result;
}

float bcm_sleep_theta_for_state(sleep_state_t state) {
    switch (state) {
        case SLEEP_STATE_AWAKE:      return BCM_SLEEP_THETA_AWAKE;
        case SLEEP_STATE_DROWSY:     return BCM_SLEEP_THETA_DROWSY;
        case SLEEP_STATE_LIGHT_NREM: return BCM_SLEEP_THETA_LIGHT_NREM;
        case SLEEP_STATE_DEEP_NREM:  return BCM_SLEEP_THETA_DEEP_NREM;
        case SLEEP_STATE_REM:        return BCM_SLEEP_THETA_REM;
        default:                     return BCM_SLEEP_THETA_AWAKE;
    }
}

float bcm_sleep_lr_for_state(sleep_state_t state) {
    switch (state) {
        case SLEEP_STATE_AWAKE:      return BCM_SLEEP_LR_AWAKE;
        case SLEEP_STATE_DROWSY:     return BCM_SLEEP_LR_DROWSY;
        case SLEEP_STATE_LIGHT_NREM: return BCM_SLEEP_LR_LIGHT_NREM;
        case SLEEP_STATE_DEEP_NREM:  return BCM_SLEEP_LR_DEEP_NREM;
        case SLEEP_STATE_REM:        return BCM_SLEEP_LR_REM;
        default:                     return BCM_SLEEP_LR_AWAKE;
    }
}
