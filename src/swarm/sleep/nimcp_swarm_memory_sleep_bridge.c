/**
 * @file nimcp_swarm_memory_sleep_bridge.c
 * @brief Swarm Memory-Sleep Integration Bridge Implementation
 * @version 1.0.0
 * @date 2025-12-18
 *
 * WHAT: Modulates swarm memory based on sleep state
 * WHY:  Memory consolidation and replay are enhanced during sleep
 * HOW:  Sleep state callbacks dynamically adjust memory parameters
 *
 * @author NIMCP Development Team
 */

#include "swarm/sleep/nimcp_swarm_memory_sleep_bridge.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/platform/nimcp_platform.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <stdlib.h>
#include <string.h>

struct swarm_memory_sleep_bridge_struct {
    bridge_base_t base;               /**< MUST be first: base bridge infrastructure */

    swarm_memory_sleep_config_t config;
    sleep_system_t sleep_system;
    swarm_memory_sleep_effects_t effects;
    bool callback_registered;
};

static void swarm_memory_on_sleep_state_change(sleep_state_t new_state, void* user_data)
{
    swarm_memory_sleep_bridge_t bridge = (swarm_memory_sleep_bridge_t)user_data;
    if (!bridge) return;

    nimcp_mutex_lock(bridge->base.mutex);

    bridge->effects.current_state = new_state;
    bridge->effects.consolidation_factor = swarm_memory_sleep_get_consol_factor(new_state);
    bridge->effects.replay_priority_factor = swarm_memory_sleep_get_replay_factor(new_state);
    bridge->effects.forgetting_rate_factor = swarm_memory_sleep_get_forget_factor(new_state);
    bridge->effects.consolidation_active = (new_state == SLEEP_STATE_LIGHT_NREM ||
                                            new_state == SLEEP_STATE_DEEP_NREM);

    NIMCP_LOGGING_DEBUG("Swarm memory sleep state changed to %d, consol=%.2f, replay=%.2f, forget=%.2f",
                        new_state, bridge->effects.consolidation_factor,
                        bridge->effects.replay_priority_factor,
                        bridge->effects.forgetting_rate_factor);

    nimcp_mutex_unlock(bridge->base.mutex);
}

int swarm_memory_sleep_default_config(swarm_memory_sleep_config_t* config)
{
    if (!config) return -1;

    config->enable_consolidation_modulation = true;
    config->enable_replay_modulation = true;
    config->enable_forgetting_modulation = true;
    config->modulation_strength = 1.0f;

    return 0;
}

swarm_memory_sleep_bridge_t swarm_memory_sleep_bridge_create(
    const swarm_memory_sleep_config_t* config,
    sleep_system_t sleep_system)
{
    if (!config || !sleep_system) {
        NIMCP_LOGGING_ERROR("Invalid parameters for swarm memory sleep bridge creation");
        return NULL;
    }

    swarm_memory_sleep_bridge_t bridge =
        (swarm_memory_sleep_bridge_t)nimcp_malloc(sizeof(struct swarm_memory_sleep_bridge_struct));
    if (!bridge) {
        NIMCP_LOGGING_ERROR("Failed to allocate swarm memory sleep bridge");
        return NULL;
    }

    memset(bridge, 0, sizeof(struct swarm_memory_sleep_bridge_struct));
    memcpy(&bridge->config, config, sizeof(swarm_memory_sleep_config_t));
    bridge->sleep_system = sleep_system;

    bridge->base.mutex = nimcp_platform_mutex_create();
    if (!bridge->base.mutex) {
        NIMCP_LOGGING_ERROR("Failed to create mutex for swarm memory sleep bridge");
        nimcp_free(bridge);
        return NULL;
    }

    bridge->effects.consolidation_factor = SWARM_MEMORY_SLEEP_CONSOL_AWAKE;
    bridge->effects.replay_priority_factor = SWARM_MEMORY_SLEEP_REPLAY_AWAKE;
    bridge->effects.forgetting_rate_factor = SWARM_MEMORY_SLEEP_FORGET_AWAKE;
    bridge->effects.current_state = SLEEP_STATE_AWAKE;
    bridge->effects.sleep_pressure = 0.0f;
    bridge->effects.consolidation_active = false;

    bridge->callback_registered = sleep_register_state_callback(
        sleep_system, swarm_memory_on_sleep_state_change, bridge);

    if (bridge->callback_registered) {
        sleep_state_t initial = sleep_get_current_state(sleep_system);
        swarm_memory_on_sleep_state_change(initial, bridge);
    }

    NIMCP_LOGGING_INFO("Swarm memory sleep bridge created");
    return bridge;
}

void swarm_memory_sleep_bridge_destroy(swarm_memory_sleep_bridge_t bridge)
{
    if (!bridge) return;

    if (bridge->callback_registered && bridge->sleep_system) {
        sleep_unregister_state_callback(bridge->sleep_system,
                                        swarm_memory_on_sleep_state_change, bridge);
    }

    if (bridge->base.mutex) {
        nimcp_platform_mutex_destroy(bridge->base.mutex);
    }

    nimcp_free(bridge);
    NIMCP_LOGGING_INFO("Swarm memory sleep bridge destroyed");
}

int swarm_memory_sleep_update(swarm_memory_sleep_bridge_t bridge)
{
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->base.mutex);
    if (bridge->sleep_system) {
        bridge->effects.sleep_pressure = sleep_get_pressure(bridge->sleep_system);
    }
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

int swarm_memory_sleep_get_effects(const swarm_memory_sleep_bridge_t bridge,
                                    swarm_memory_sleep_effects_t* effects)
{
    if (!bridge || !effects) return -1;

    nimcp_mutex_lock(bridge->base.mutex);
    memcpy(effects, &bridge->effects, sizeof(swarm_memory_sleep_effects_t));
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

float swarm_memory_sleep_get_consolidation(const swarm_memory_sleep_bridge_t bridge, float base)
{
    if (!bridge) return base;

    nimcp_mutex_lock(bridge->base.mutex);
    float factor = bridge->config.enable_consolidation_modulation ?
        bridge->effects.consolidation_factor : 1.0f;
    float result = base * (1.0f + (factor - 1.0f) * bridge->config.modulation_strength);
    nimcp_mutex_unlock(bridge->base.mutex);

    return result;
}

float swarm_memory_sleep_get_replay_priority(const swarm_memory_sleep_bridge_t bridge, float base)
{
    if (!bridge) return base;

    nimcp_mutex_lock(bridge->base.mutex);
    float factor = bridge->config.enable_replay_modulation ?
        bridge->effects.replay_priority_factor : 1.0f;
    float result = base * (1.0f + (factor - 1.0f) * bridge->config.modulation_strength);
    nimcp_mutex_unlock(bridge->base.mutex);

    return result;
}

float swarm_memory_sleep_get_forgetting_rate(const swarm_memory_sleep_bridge_t bridge, float base)
{
    if (!bridge) return base;

    nimcp_mutex_lock(bridge->base.mutex);
    float factor = bridge->config.enable_forgetting_modulation ?
        bridge->effects.forgetting_rate_factor : 1.0f;
    float result = base * (1.0f + (factor - 1.0f) * bridge->config.modulation_strength);
    nimcp_mutex_unlock(bridge->base.mutex);

    return result;
}

float swarm_memory_sleep_get_consol_factor(sleep_state_t state)
{
    switch (state) {
        case SLEEP_STATE_AWAKE:      return SWARM_MEMORY_SLEEP_CONSOL_AWAKE;
        case SLEEP_STATE_DROWSY:     return SWARM_MEMORY_SLEEP_CONSOL_DROWSY;
        case SLEEP_STATE_LIGHT_NREM: return SWARM_MEMORY_SLEEP_CONSOL_LIGHT_NREM;
        case SLEEP_STATE_DEEP_NREM:  return SWARM_MEMORY_SLEEP_CONSOL_DEEP_NREM;
        case SLEEP_STATE_REM:        return SWARM_MEMORY_SLEEP_CONSOL_REM;
        default:                     return SWARM_MEMORY_SLEEP_CONSOL_AWAKE;
    }
}

float swarm_memory_sleep_get_replay_factor(sleep_state_t state)
{
    switch (state) {
        case SLEEP_STATE_AWAKE:      return SWARM_MEMORY_SLEEP_REPLAY_AWAKE;
        case SLEEP_STATE_DROWSY:     return SWARM_MEMORY_SLEEP_REPLAY_DROWSY;
        case SLEEP_STATE_LIGHT_NREM: return SWARM_MEMORY_SLEEP_REPLAY_LIGHT_NREM;
        case SLEEP_STATE_DEEP_NREM:  return SWARM_MEMORY_SLEEP_REPLAY_DEEP_NREM;
        case SLEEP_STATE_REM:        return SWARM_MEMORY_SLEEP_REPLAY_REM;
        default:                     return SWARM_MEMORY_SLEEP_REPLAY_AWAKE;
    }
}

float swarm_memory_sleep_get_forget_factor(sleep_state_t state)
{
    switch (state) {
        case SLEEP_STATE_AWAKE:      return SWARM_MEMORY_SLEEP_FORGET_AWAKE;
        case SLEEP_STATE_DROWSY:     return SWARM_MEMORY_SLEEP_FORGET_DROWSY;
        case SLEEP_STATE_LIGHT_NREM: return SWARM_MEMORY_SLEEP_FORGET_LIGHT_NREM;
        case SLEEP_STATE_DEEP_NREM:  return SWARM_MEMORY_SLEEP_FORGET_DEEP_NREM;
        case SLEEP_STATE_REM:        return SWARM_MEMORY_SLEEP_FORGET_REM;
        default:                     return SWARM_MEMORY_SLEEP_FORGET_AWAKE;
    }
}
