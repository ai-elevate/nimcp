/**
 * @file nimcp_executive_sleep_bridge.c
 * @brief Sleep-Executive Function Integration Bridge Implementation
 * @version 1.0.0
 * @date 2025-12-17
 */

#include "cognitive/executive/nimcp_executive_sleep_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/platform/nimcp_platform_mutex.h"
#include <string.h>

struct executive_sleep_bridge_struct {
    executive_sleep_config_t config;
    sleep_system_t sleep_system;
    executive_sleep_effects_t effects;
    nimcp_mutex_t* mutex;
};

int executive_sleep_default_config(executive_sleep_config_t* config) {
    if (!config) return -1;
    config->enable_inhibition_modulation = true;
    config->enable_flexibility_modulation = true;
    config->enable_switch_cost_modulation = true;
    config->modulation_strength = 1.0f;
    return 0;
}

executive_sleep_bridge_t executive_sleep_bridge_create(
    const executive_sleep_config_t* config,
    sleep_system_t sleep)
{
    if (!sleep) return NULL;

    struct executive_sleep_bridge_struct* bridge =
        (struct executive_sleep_bridge_struct*)nimcp_malloc(
            sizeof(struct executive_sleep_bridge_struct));
    if (!bridge) return NULL;

    memset(bridge, 0, sizeof(struct executive_sleep_bridge_struct));

    if (config) bridge->config = *config;
    else executive_sleep_default_config(&bridge->config);

    bridge->sleep_system = sleep;
    bridge->effects.inhibition_factor = 1.0f;
    bridge->effects.flexibility_factor = 1.0f;
    bridge->effects.switch_cost_factor = 1.0f;
    bridge->effects.executive_offline = false;

    bridge->mutex = nimcp_platform_mutex_create();
    if (!bridge->mutex) { nimcp_free(bridge); return NULL; }

    NIMCP_LOGGING_INFO("Executive-sleep bridge created");
    return bridge;
}

void executive_sleep_bridge_destroy(executive_sleep_bridge_t bridge) {
    if (!bridge) return;
    if (bridge->mutex) nimcp_mutex_destroy(bridge->mutex);
    nimcp_free(bridge);
}

int executive_sleep_update(executive_sleep_bridge_t bridge) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);

    sleep_state_t state = sleep_get_current_state(bridge->sleep_system);
    float pressure = sleep_get_pressure(bridge->sleep_system);

    bridge->effects.current_state = state;
    bridge->effects.sleep_pressure = pressure;

    if (bridge->config.enable_inhibition_modulation) {
        bridge->effects.inhibition_factor = executive_sleep_inhibition_for_state(state);
        /* Sleep pressure impairs inhibition even when awake */
        if (state == SLEEP_STATE_AWAKE && pressure > 0.5f) {
            bridge->effects.inhibition_factor *= (1.0f - 0.4f * (pressure - 0.5f) / 0.5f);
        }
    }

    if (bridge->config.enable_flexibility_modulation) {
        bridge->effects.flexibility_factor = executive_sleep_flexibility_for_state(state);
    }

    if (bridge->config.enable_switch_cost_modulation) {
        bridge->effects.switch_cost_factor = executive_sleep_switch_cost_for_state(state);
    }

    bridge->effects.executive_offline = (state == SLEEP_STATE_DEEP_NREM ||
                                         state == SLEEP_STATE_LIGHT_NREM);

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int executive_sleep_get_effects(
    const executive_sleep_bridge_t bridge,
    executive_sleep_effects_t* effects)
{
    if (!bridge || !effects) return -1;
    nimcp_mutex_lock(bridge->mutex);
    *effects = bridge->effects;
    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

float executive_sleep_get_inhibition(const executive_sleep_bridge_t bridge) {
    if (!bridge) return 1.0f;
    nimcp_mutex_lock(bridge->mutex);
    float result = bridge->effects.inhibition_factor;
    nimcp_mutex_unlock(bridge->mutex);
    return result;
}

bool executive_sleep_is_offline(const executive_sleep_bridge_t bridge) {
    if (!bridge) return false;
    nimcp_mutex_lock(bridge->mutex);
    bool result = bridge->effects.executive_offline;
    nimcp_mutex_unlock(bridge->mutex);
    return result;
}

float executive_sleep_inhibition_for_state(sleep_state_t state) {
    switch (state) {
        case SLEEP_STATE_AWAKE:      return EXEC_SLEEP_INHIBITION_AWAKE;
        case SLEEP_STATE_DROWSY:     return EXEC_SLEEP_INHIBITION_DROWSY;
        case SLEEP_STATE_LIGHT_NREM: return EXEC_SLEEP_INHIBITION_LIGHT_NREM;
        case SLEEP_STATE_DEEP_NREM:  return EXEC_SLEEP_INHIBITION_DEEP_NREM;
        case SLEEP_STATE_REM:        return EXEC_SLEEP_INHIBITION_REM;
        default:                     return EXEC_SLEEP_INHIBITION_AWAKE;
    }
}

float executive_sleep_flexibility_for_state(sleep_state_t state) {
    switch (state) {
        case SLEEP_STATE_AWAKE:      return EXEC_SLEEP_FLEXIBILITY_AWAKE;
        case SLEEP_STATE_DROWSY:     return EXEC_SLEEP_FLEXIBILITY_DROWSY;
        case SLEEP_STATE_LIGHT_NREM:
        case SLEEP_STATE_DEEP_NREM:  return EXEC_SLEEP_FLEXIBILITY_NREM;
        case SLEEP_STATE_REM:        return EXEC_SLEEP_FLEXIBILITY_REM;
        default:                     return EXEC_SLEEP_FLEXIBILITY_AWAKE;
    }
}

float executive_sleep_switch_cost_for_state(sleep_state_t state) {
    switch (state) {
        case SLEEP_STATE_AWAKE:      return EXEC_SLEEP_SWITCH_COST_AWAKE;
        case SLEEP_STATE_DROWSY:     return EXEC_SLEEP_SWITCH_COST_DROWSY;
        case SLEEP_STATE_LIGHT_NREM:
        case SLEEP_STATE_DEEP_NREM:  return EXEC_SLEEP_SWITCH_COST_NREM;
        case SLEEP_STATE_REM:        return EXEC_SLEEP_SWITCH_COST_REM;
        default:                     return EXEC_SLEEP_SWITCH_COST_AWAKE;
    }
}
