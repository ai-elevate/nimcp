/**
 * @file nimcp_bbb_sleep_bridge.c
 * @brief Sleep-BBB Integration Bridge Implementation
 * @version 1.0.0
 * @date 2025-12-18
 */

#include "security/sleep/nimcp_bbb_sleep_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/platform/nimcp_platform_mutex.h"
#include <string.h>

struct bbb_sleep_bridge_struct {
    bbb_sleep_config_t config;
    sleep_system_t sleep_system;
    bbb_sleep_effects_t effects;
    nimcp_mutex_t* mutex;
    bool callback_registered;
};

static void bbb_on_sleep_state_change(sleep_state_t new_state, void* user_data)
{
    bbb_sleep_bridge_t bridge = (bbb_sleep_bridge_t)user_data;
    if (!bridge) return;

    nimcp_mutex_lock(bridge->mutex);
    bridge->effects.current_state = new_state;

    if (bridge->config.enable_permeability_modulation) {
        bridge->effects.permeability_factor = bbb_sleep_permeability_for_state(new_state);
    }
    if (bridge->config.enable_detection_modulation) {
        bridge->effects.detection_threshold_factor = bbb_sleep_detection_for_state(new_state);
    }
    if (bridge->config.enable_response_modulation) {
        bridge->effects.response_urgency_factor = bbb_sleep_urgency_for_state(new_state);
    }

    bridge->effects.glymphatic_active = (new_state == SLEEP_STATE_LIGHT_NREM ||
                                         new_state == SLEEP_STATE_DEEP_NREM);
    bridge->effects.critical_protection_only = (new_state == SLEEP_STATE_DEEP_NREM) &&
                                                bridge->config.maintain_critical_protection;
    nimcp_mutex_unlock(bridge->mutex);
}

int bbb_sleep_default_config(bbb_sleep_config_t* config) {
    if (!config) return -1;
    config->enable_permeability_modulation = true;
    config->enable_detection_modulation = true;
    config->enable_response_modulation = true;
    config->modulation_strength = 1.0f;
    config->maintain_critical_protection = true;
    return 0;
}

bbb_sleep_bridge_t bbb_sleep_bridge_create(
    const bbb_sleep_config_t* config, sleep_system_t sleep_system)
{
    if (!sleep_system) return NULL;

    struct bbb_sleep_bridge_struct* bridge = nimcp_malloc(sizeof(*bridge));
    if (!bridge) return NULL;
    memset(bridge, 0, sizeof(*bridge));

    if (config) bridge->config = *config;
    else bbb_sleep_default_config(&bridge->config);

    bridge->sleep_system = sleep_system;
    bridge->effects.permeability_factor = 1.0f;
    bridge->effects.detection_threshold_factor = 1.0f;
    bridge->effects.response_urgency_factor = 1.0f;
    bridge->effects.glymphatic_active = false;

    bridge->mutex = nimcp_platform_mutex_create();
    if (!bridge->mutex) { nimcp_free(bridge); return NULL; }

    bridge->callback_registered = sleep_register_state_callback(
        sleep_system, bbb_on_sleep_state_change, bridge);

    sleep_state_t initial = sleep_get_current_state(sleep_system);
    bbb_on_sleep_state_change(initial, bridge);

    NIMCP_LOGGING_INFO("BBB-sleep bridge created");
    return bridge;
}

void bbb_sleep_bridge_destroy(bbb_sleep_bridge_t bridge) {
    if (!bridge) return;
    if (bridge->callback_registered && bridge->sleep_system) {
        sleep_unregister_state_callback(bridge->sleep_system,
            bbb_on_sleep_state_change, bridge);
    }
    if (bridge->mutex) nimcp_mutex_destroy(bridge->mutex);
    nimcp_free(bridge);
}

int bbb_sleep_update(bbb_sleep_bridge_t bridge) {
    if (!bridge) return -1;
    sleep_state_t state = sleep_get_current_state(bridge->sleep_system);
    bbb_on_sleep_state_change(state, bridge);
    return 0;
}

int bbb_sleep_get_effects(const bbb_sleep_bridge_t bridge, bbb_sleep_effects_t* effects) {
    if (!bridge || !effects) return -1;
    nimcp_mutex_lock(bridge->mutex);
    *effects = bridge->effects;
    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

float bbb_sleep_get_permeability(const bbb_sleep_bridge_t bridge) {
    if (!bridge) return 1.0f;
    nimcp_mutex_lock(bridge->mutex);
    float result = bridge->effects.permeability_factor;
    nimcp_mutex_unlock(bridge->mutex);
    return result;
}

bool bbb_sleep_is_glymphatic_active(const bbb_sleep_bridge_t bridge) {
    if (!bridge) return false;
    nimcp_mutex_lock(bridge->mutex);
    bool result = bridge->effects.glymphatic_active;
    nimcp_mutex_unlock(bridge->mutex);
    return result;
}

float bbb_sleep_get_detection_threshold(const bbb_sleep_bridge_t bridge, float base) {
    if (!bridge) return base;
    nimcp_mutex_lock(bridge->mutex);
    float result = base * bridge->effects.detection_threshold_factor;
    nimcp_mutex_unlock(bridge->mutex);
    return result;
}

float bbb_sleep_permeability_for_state(sleep_state_t state) {
    switch (state) {
        case SLEEP_STATE_AWAKE:      return BBB_SLEEP_PERMEABILITY_AWAKE;
        case SLEEP_STATE_DROWSY:     return BBB_SLEEP_PERMEABILITY_DROWSY;
        case SLEEP_STATE_LIGHT_NREM: return BBB_SLEEP_PERMEABILITY_LIGHT_NREM;
        case SLEEP_STATE_DEEP_NREM:  return BBB_SLEEP_PERMEABILITY_DEEP_NREM;
        case SLEEP_STATE_REM:        return BBB_SLEEP_PERMEABILITY_REM;
        default:                     return BBB_SLEEP_PERMEABILITY_AWAKE;
    }
}

float bbb_sleep_detection_for_state(sleep_state_t state) {
    switch (state) {
        case SLEEP_STATE_AWAKE:      return BBB_SLEEP_DETECTION_AWAKE;
        case SLEEP_STATE_DROWSY:     return BBB_SLEEP_DETECTION_DROWSY;
        case SLEEP_STATE_LIGHT_NREM: return BBB_SLEEP_DETECTION_LIGHT_NREM;
        case SLEEP_STATE_DEEP_NREM:  return BBB_SLEEP_DETECTION_DEEP_NREM;
        case SLEEP_STATE_REM:        return BBB_SLEEP_DETECTION_REM;
        default:                     return BBB_SLEEP_DETECTION_AWAKE;
    }
}

float bbb_sleep_urgency_for_state(sleep_state_t state) {
    switch (state) {
        case SLEEP_STATE_AWAKE:      return BBB_SLEEP_URGENCY_AWAKE;
        case SLEEP_STATE_DROWSY:     return BBB_SLEEP_URGENCY_DROWSY;
        case SLEEP_STATE_LIGHT_NREM: return BBB_SLEEP_URGENCY_LIGHT_NREM;
        case SLEEP_STATE_DEEP_NREM:  return BBB_SLEEP_URGENCY_DEEP_NREM;
        case SLEEP_STATE_REM:        return BBB_SLEEP_URGENCY_REM;
        default:                     return BBB_SLEEP_URGENCY_AWAKE;
    }
}
