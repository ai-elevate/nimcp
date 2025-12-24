/**
 * @file nimcp_swarm_immune_sleep_bridge.c
 * @brief Swarm Immune-Sleep Integration Bridge Implementation
 * @version 1.0.0
 * @date 2025-12-18
 *
 * WHAT: Modulates swarm immune system based on sleep state
 * WHY:  Detection and response should vary with consciousness level
 * HOW:  Sleep state callbacks dynamically adjust immune parameters
 *
 * @author NIMCP Development Team
 */

#include "swarm/sleep/nimcp_swarm_immune_sleep_bridge.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/platform/nimcp_platform.h"
#include "utils/logging/nimcp_logging.h"
#include <stdlib.h>
#include <string.h>

struct swarm_immune_sleep_bridge_struct {
    bridge_base_t base;               /**< MUST be first: base bridge infrastructure */

    swarm_immune_sleep_config_t config;
    sleep_system_t sleep_system;
    swarm_immune_sleep_effects_t effects;
    bool callback_registered;
};

static void swarm_immune_on_sleep_state_change(sleep_state_t new_state, void* user_data)
{
    swarm_immune_sleep_bridge_t bridge = (swarm_immune_sleep_bridge_t)user_data;
    if (!bridge) return;

    nimcp_mutex_lock(bridge->base.mutex);

    bridge->effects.current_state = new_state;

    if (bridge->config.enable_detection_modulation) {
        float base = swarm_immune_sleep_get_detect_factor(new_state);
        bridge->effects.detection_sensitivity = 1.0f + (base - 1.0f) * bridge->config.modulation_strength;
    }
    if (bridge->config.enable_response_modulation) {
        float base = swarm_immune_sleep_get_response_factor(new_state);
        bridge->effects.response_intensity = 1.0f + (base - 1.0f) * bridge->config.modulation_strength;
    }
    if (bridge->config.enable_memory_consolidation) {
        float base = swarm_immune_sleep_get_memory_factor(new_state);
        bridge->effects.memory_consolidation = 1.0f + (base - 1.0f) * bridge->config.modulation_strength;
    }
    bridge->effects.suppress_non_critical = (new_state == SLEEP_STATE_DEEP_NREM);

    NIMCP_LOGGING_DEBUG("Swarm immune sleep state changed to %d, detect=%.2f, resp=%.2f, mem=%.2f",
                        new_state, bridge->effects.detection_sensitivity,
                        bridge->effects.response_intensity,
                        bridge->effects.memory_consolidation);

    nimcp_mutex_unlock(bridge->base.mutex);
}

int swarm_immune_sleep_default_config(swarm_immune_sleep_config_t* config)
{
    if (!config) return -1;

    config->enable_detection_modulation = true;
    config->enable_response_modulation = true;
    config->enable_memory_consolidation = true;
    config->modulation_strength = 1.0f;

    return 0;
}

swarm_immune_sleep_bridge_t swarm_immune_sleep_bridge_create(
    const swarm_immune_sleep_config_t* config,
    sleep_system_t sleep_system)
{
    if (!config || !sleep_system) {
        NIMCP_LOGGING_ERROR("Invalid parameters for swarm immune sleep bridge creation");
        return NULL;
    }

    swarm_immune_sleep_bridge_t bridge =
        (swarm_immune_sleep_bridge_t)nimcp_malloc(sizeof(struct swarm_immune_sleep_bridge_struct));
    if (!bridge) {
        NIMCP_LOGGING_ERROR("Failed to allocate swarm immune sleep bridge");
        return NULL;
    }

    memset(bridge, 0, sizeof(struct swarm_immune_sleep_bridge_struct));
    memcpy(&bridge->config, config, sizeof(swarm_immune_sleep_config_t));
    bridge->sleep_system = sleep_system;

    bridge->base.mutex = nimcp_platform_mutex_create();
    if (!bridge->base.mutex) {
        NIMCP_LOGGING_ERROR("Failed to create mutex for swarm immune sleep bridge");
        nimcp_free(bridge);
        return NULL;
    }

    bridge->effects.detection_sensitivity = SWARM_IMMUNE_SLEEP_DETECT_AWAKE;
    bridge->effects.response_intensity = SWARM_IMMUNE_SLEEP_RESPONSE_AWAKE;
    bridge->effects.memory_consolidation = SWARM_IMMUNE_SLEEP_MEMORY_AWAKE;
    bridge->effects.current_state = SLEEP_STATE_AWAKE;
    bridge->effects.sleep_pressure = 0.0f;
    bridge->effects.suppress_non_critical = false;

    bridge->callback_registered = sleep_register_state_callback(
        sleep_system, swarm_immune_on_sleep_state_change, bridge);

    if (bridge->callback_registered) {
        sleep_state_t initial = sleep_get_current_state(sleep_system);
        swarm_immune_on_sleep_state_change(initial, bridge);
    }

    NIMCP_LOGGING_INFO("Swarm immune sleep bridge created");
    return bridge;
}

void swarm_immune_sleep_bridge_destroy(swarm_immune_sleep_bridge_t bridge)
{
    if (!bridge) return;

    if (bridge->callback_registered && bridge->sleep_system) {
        sleep_unregister_state_callback(bridge->sleep_system,
                                        swarm_immune_on_sleep_state_change, bridge);
    }

    if (bridge->base.mutex) {
        nimcp_platform_mutex_destroy(bridge->base.mutex);
    }

    nimcp_free(bridge);
    NIMCP_LOGGING_INFO("Swarm immune sleep bridge destroyed");
}

int swarm_immune_sleep_update(swarm_immune_sleep_bridge_t bridge)
{
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->base.mutex);
    if (bridge->sleep_system) {
        bridge->effects.sleep_pressure = sleep_get_pressure(bridge->sleep_system);
    }
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

int swarm_immune_sleep_get_effects(const swarm_immune_sleep_bridge_t bridge,
                                    swarm_immune_sleep_effects_t* effects)
{
    if (!bridge || !effects) return -1;

    nimcp_mutex_lock(bridge->base.mutex);
    memcpy(effects, &bridge->effects, sizeof(swarm_immune_sleep_effects_t));
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

float swarm_immune_sleep_get_detection_sensitivity(const swarm_immune_sleep_bridge_t bridge, float base)
{
    if (!bridge) return base;

    nimcp_mutex_lock(bridge->base.mutex);
    float result = base * bridge->effects.detection_sensitivity;
    nimcp_mutex_unlock(bridge->base.mutex);

    return result;
}

float swarm_immune_sleep_get_response_intensity(const swarm_immune_sleep_bridge_t bridge, float base)
{
    if (!bridge) return base;

    nimcp_mutex_lock(bridge->base.mutex);
    float result = base * bridge->effects.response_intensity;
    nimcp_mutex_unlock(bridge->base.mutex);

    return result;
}

float swarm_immune_sleep_get_memory_boost(const swarm_immune_sleep_bridge_t bridge, float base)
{
    if (!bridge) return base;

    nimcp_mutex_lock(bridge->base.mutex);
    float result = base * bridge->effects.memory_consolidation;
    nimcp_mutex_unlock(bridge->base.mutex);

    return result;
}

float swarm_immune_sleep_get_detect_factor(sleep_state_t state)
{
    switch (state) {
        case SLEEP_STATE_AWAKE:      return SWARM_IMMUNE_SLEEP_DETECT_AWAKE;
        case SLEEP_STATE_DROWSY:     return SWARM_IMMUNE_SLEEP_DETECT_DROWSY;
        case SLEEP_STATE_LIGHT_NREM: return SWARM_IMMUNE_SLEEP_DETECT_LIGHT_NREM;
        case SLEEP_STATE_DEEP_NREM:  return SWARM_IMMUNE_SLEEP_DETECT_DEEP_NREM;
        case SLEEP_STATE_REM:        return SWARM_IMMUNE_SLEEP_DETECT_REM;
        default:                     return SWARM_IMMUNE_SLEEP_DETECT_AWAKE;
    }
}

float swarm_immune_sleep_get_response_factor(sleep_state_t state)
{
    switch (state) {
        case SLEEP_STATE_AWAKE:      return SWARM_IMMUNE_SLEEP_RESPONSE_AWAKE;
        case SLEEP_STATE_DROWSY:     return SWARM_IMMUNE_SLEEP_RESPONSE_DROWSY;
        case SLEEP_STATE_LIGHT_NREM: return SWARM_IMMUNE_SLEEP_RESPONSE_LIGHT_NREM;
        case SLEEP_STATE_DEEP_NREM:  return SWARM_IMMUNE_SLEEP_RESPONSE_DEEP_NREM;
        case SLEEP_STATE_REM:        return SWARM_IMMUNE_SLEEP_RESPONSE_REM;
        default:                     return SWARM_IMMUNE_SLEEP_RESPONSE_AWAKE;
    }
}

float swarm_immune_sleep_get_memory_factor(sleep_state_t state)
{
    switch (state) {
        case SLEEP_STATE_AWAKE:      return SWARM_IMMUNE_SLEEP_MEMORY_AWAKE;
        case SLEEP_STATE_DROWSY:     return SWARM_IMMUNE_SLEEP_MEMORY_DROWSY;
        case SLEEP_STATE_LIGHT_NREM: return SWARM_IMMUNE_SLEEP_MEMORY_LIGHT_NREM;
        case SLEEP_STATE_DEEP_NREM:  return SWARM_IMMUNE_SLEEP_MEMORY_DEEP_NREM;
        case SLEEP_STATE_REM:        return SWARM_IMMUNE_SLEEP_MEMORY_REM;
        default:                     return SWARM_IMMUNE_SLEEP_MEMORY_AWAKE;
    }
}
