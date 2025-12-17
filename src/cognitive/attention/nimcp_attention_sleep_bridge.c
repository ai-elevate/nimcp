/**
 * @file nimcp_attention_sleep_bridge.c
 * @brief Sleep-Attention Integration Bridge Implementation
 * @version 1.0.0
 * @date 2025-12-17
 */

#include "cognitive/attention/nimcp_attention_sleep_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/platform/nimcp_platform_mutex.h"
#include <string.h>

struct attention_sleep_bridge_struct {
    attention_sleep_config_t config;
    sleep_system_t sleep_system;
    attention_sleep_effects_t effects;
    nimcp_mutex_t* mutex;
};

int attention_sleep_default_config(attention_sleep_config_t* config) {
    if (!config) return -1;
    config->enable_capacity_modulation = true;
    config->enable_vigilance_modulation = true;
    config->modulation_strength = 1.0f;
    return 0;
}

attention_sleep_bridge_t attention_sleep_bridge_create(
    const attention_sleep_config_t* config,
    sleep_system_t sleep)
{
    if (!sleep) return NULL;

    struct attention_sleep_bridge_struct* bridge =
        (struct attention_sleep_bridge_struct*)nimcp_malloc(sizeof(struct attention_sleep_bridge_struct));
    if (!bridge) return NULL;

    memset(bridge, 0, sizeof(struct attention_sleep_bridge_struct));

    if (config) bridge->config = *config;
    else attention_sleep_default_config(&bridge->config);

    bridge->sleep_system = sleep;
    bridge->effects.capacity_factor = 1.0f;
    bridge->effects.vigilance_factor = 1.0f;
    bridge->effects.spotlight_size_factor = 1.0f;
    bridge->effects.attention_offline = false;

    bridge->mutex = nimcp_platform_mutex_create();
    if (!bridge->mutex) { nimcp_free(bridge); return NULL; }

    NIMCP_LOGGING_INFO("Attention-sleep bridge created");
    return bridge;
}

void attention_sleep_bridge_destroy(attention_sleep_bridge_t bridge) {
    if (!bridge) return;
    if (bridge->mutex) nimcp_mutex_destroy(bridge->mutex);
    nimcp_free(bridge);
}

int attention_sleep_update(attention_sleep_bridge_t bridge) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);

    sleep_state_t state = sleep_get_current_state(bridge->sleep_system);
    float pressure = sleep_get_pressure(bridge->sleep_system);

    bridge->effects.current_state = state;
    bridge->effects.sleep_pressure = pressure;

    if (bridge->config.enable_capacity_modulation) {
        float cap_base = attention_sleep_capacity_for_state(state);
        bridge->effects.capacity_factor = cap_base * bridge->config.modulation_strength +
                                          (1.0f - bridge->config.modulation_strength);
        /* Sleep pressure further reduces capacity */
        if (state == SLEEP_STATE_AWAKE && pressure > 0.7f) {
            bridge->effects.capacity_factor *= (1.0f - (pressure - 0.7f));
        }
    }

    if (bridge->config.enable_vigilance_modulation) {
        bridge->effects.vigilance_factor = attention_sleep_vigilance_for_state(state);
    }

    /* Spotlight narrows with drowsiness, expands in REM */
    bridge->effects.spotlight_size_factor = (state == SLEEP_STATE_DROWSY) ? 0.7f :
                                            (state == SLEEP_STATE_REM) ? 1.3f : 1.0f;

    bridge->effects.attention_offline = (state == SLEEP_STATE_DEEP_NREM ||
                                         state == SLEEP_STATE_LIGHT_NREM);

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int attention_sleep_get_effects(const attention_sleep_bridge_t bridge, attention_sleep_effects_t* effects) {
    if (!bridge || !effects) return -1;
    nimcp_mutex_lock(bridge->mutex);
    *effects = bridge->effects;
    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

float attention_sleep_get_capacity(const attention_sleep_bridge_t bridge) {
    if (!bridge) return 1.0f;
    nimcp_mutex_lock(bridge->mutex);
    float result = bridge->effects.capacity_factor;
    nimcp_mutex_unlock(bridge->mutex);
    return result;
}

bool attention_sleep_is_offline(const attention_sleep_bridge_t bridge) {
    if (!bridge) return false;
    nimcp_mutex_lock(bridge->mutex);
    bool result = bridge->effects.attention_offline;
    nimcp_mutex_unlock(bridge->mutex);
    return result;
}

float attention_sleep_capacity_for_state(sleep_state_t state) {
    switch (state) {
        case SLEEP_STATE_AWAKE:      return ATTN_SLEEP_CAPACITY_AWAKE;
        case SLEEP_STATE_DROWSY:     return ATTN_SLEEP_CAPACITY_DROWSY;
        case SLEEP_STATE_LIGHT_NREM: return ATTN_SLEEP_CAPACITY_LIGHT_NREM;
        case SLEEP_STATE_DEEP_NREM:  return ATTN_SLEEP_CAPACITY_DEEP_NREM;
        case SLEEP_STATE_REM:        return ATTN_SLEEP_CAPACITY_REM;
        default:                     return ATTN_SLEEP_CAPACITY_AWAKE;
    }
}

float attention_sleep_vigilance_for_state(sleep_state_t state) {
    switch (state) {
        case SLEEP_STATE_AWAKE:      return ATTN_SLEEP_VIGILANCE_AWAKE;
        case SLEEP_STATE_DROWSY:     return ATTN_SLEEP_VIGILANCE_DROWSY;
        case SLEEP_STATE_LIGHT_NREM:
        case SLEEP_STATE_DEEP_NREM:  return ATTN_SLEEP_VIGILANCE_NREM;
        case SLEEP_STATE_REM:        return ATTN_SLEEP_VIGILANCE_REM;
        default:                     return ATTN_SLEEP_VIGILANCE_AWAKE;
    }
}
