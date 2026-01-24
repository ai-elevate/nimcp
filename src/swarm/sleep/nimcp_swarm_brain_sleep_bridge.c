/**
 * @file nimcp_swarm_brain_sleep_bridge.c
 * @brief Sleep-Swarm Brain Integration Bridge Implementation
 * @version 1.0.0
 * @date 2025-12-18
 *
 * WHAT: Modulates swarm brain coordination based on sleep state
 * WHY:  Coordination strength and heartbeat frequency should vary with alertness
 * HOW:  Sleep state callbacks dynamically adjust coordination parameters
 *
 * @author NIMCP Development Team
 */

#include "swarm/sleep/nimcp_swarm_brain_sleep_bridge.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/platform/nimcp_platform.h"
#include "utils/logging/nimcp_logging.h"
#include "api/nimcp_api_exception.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <stdlib.h>
#include <string.h>

struct swarm_brain_sleep_bridge_struct {
    bridge_base_t base;               /**< MUST be first: base bridge infrastructure */

    swarm_brain_sleep_config_t config;
    sleep_system_t sleep_system;
    swarm_brain_sleep_effects_t effects;
    bool callback_registered;
};

static void swarm_brain_on_sleep_state_change(sleep_state_t new_state, void* user_data)
{
    swarm_brain_sleep_bridge_t bridge = (swarm_brain_sleep_bridge_t)user_data;
    if (!bridge) return;

    nimcp_mutex_lock(bridge->base.mutex);
    bridge->effects.current_state = new_state;

    if (bridge->config.enable_coord_modulation) {
        float base = swarm_brain_sleep_get_coord_factor(new_state);
        bridge->effects.coordination_factor = 1.0f + (base - 1.0f) * bridge->config.modulation_strength;
    }
    if (bridge->config.enable_heartbeat_modulation) {
        float base = swarm_brain_sleep_get_heartbeat_factor(new_state);
        bridge->effects.heartbeat_multiplier = 1.0f + (base - 1.0f) * bridge->config.modulation_strength;
    }
    if (bridge->config.enable_coherence_modulation) {
        float base = swarm_brain_sleep_get_coherence_factor(new_state);
        bridge->effects.coherence_factor = 1.0f + (base - 1.0f) * bridge->config.modulation_strength;
    }
    bridge->effects.coordination_enabled = (new_state != SLEEP_STATE_DEEP_NREM);
    nimcp_mutex_unlock(bridge->base.mutex);
}

int swarm_brain_sleep_default_config(swarm_brain_sleep_config_t* config)
{
    if (!config) return -1;
    config->enable_coord_modulation = true;
    config->enable_heartbeat_modulation = true;
    config->enable_coherence_modulation = true;
    config->modulation_strength = 1.0f;
    return 0;
}

swarm_brain_sleep_bridge_t swarm_brain_sleep_bridge_create(
    const swarm_brain_sleep_config_t* config, sleep_system_t sleep_system)
{
    if (!sleep_system) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "sleep_system is NULL");

        return NULL;

    }

    struct swarm_brain_sleep_bridge_struct* bridge = nimcp_malloc(sizeof(*bridge));
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return NULL;

    }
    memset(bridge, 0, sizeof(*bridge));

    if (config) bridge->config = *config;
    else swarm_brain_sleep_default_config(&bridge->config);

    bridge->sleep_system = sleep_system;
    bridge->effects.coordination_factor = 1.0f;
    bridge->effects.heartbeat_multiplier = 1.0f;
    bridge->effects.coherence_factor = 1.0f;
    bridge->effects.coordination_enabled = true;

    if (bridge_base_init(&bridge->base, 0, "swarm_brain_sleep") != 0) { nimcp_free(bridge); return NULL; }
    if (!bridge->base.mutex) { nimcp_free(bridge); return NULL; }

    bridge->callback_registered = sleep_register_state_callback(
        sleep_system, swarm_brain_on_sleep_state_change, bridge);

    sleep_state_t initial = sleep_get_current_state(sleep_system);
    swarm_brain_on_sleep_state_change(initial, bridge);

    NIMCP_LOGGING_INFO("Swarm brain-sleep bridge created");
    return bridge;
}

void swarm_brain_sleep_bridge_destroy(swarm_brain_sleep_bridge_t bridge)
{
    if (!bridge) return;
    if (bridge->callback_registered && bridge->sleep_system) {
        sleep_unregister_state_callback(bridge->sleep_system,
            swarm_brain_on_sleep_state_change, bridge);
    }
    if (bridge->base.mutex) bridge_base_cleanup(&bridge->base);
    nimcp_free(bridge);
}

int swarm_brain_sleep_update(swarm_brain_sleep_bridge_t bridge)
{
    if (!bridge) return -1;
    sleep_state_t state = sleep_get_current_state(bridge->sleep_system);
    swarm_brain_on_sleep_state_change(state, bridge);
    return 0;
}

int swarm_brain_sleep_get_effects(const swarm_brain_sleep_bridge_t bridge,
                                   swarm_brain_sleep_effects_t* effects)
{
    if (!bridge || !effects) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    *effects = bridge->effects;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

float swarm_brain_sleep_get_coordination(const swarm_brain_sleep_bridge_t bridge, float base)
{
    if (!bridge) return base;
    nimcp_mutex_lock(bridge->base.mutex);
    float result = base * bridge->effects.coordination_factor;
    nimcp_mutex_unlock(bridge->base.mutex);
    return result;
}

uint32_t swarm_brain_sleep_get_heartbeat_interval(const swarm_brain_sleep_bridge_t bridge, uint32_t base_ms)
{
    if (!bridge) return base_ms;
    nimcp_mutex_lock(bridge->base.mutex);
    uint32_t result = (uint32_t)(base_ms * bridge->effects.heartbeat_multiplier);
    nimcp_mutex_unlock(bridge->base.mutex);
    return result;
}

float swarm_brain_sleep_get_coherence(const swarm_brain_sleep_bridge_t bridge, float base)
{
    if (!bridge) return base;
    nimcp_mutex_lock(bridge->base.mutex);
    float result = base * bridge->effects.coherence_factor;
    nimcp_mutex_unlock(bridge->base.mutex);
    return result;
}

float swarm_brain_sleep_get_coord_factor(sleep_state_t state)
{
    switch (state) {
        case SLEEP_STATE_AWAKE:      return SWARM_BRAIN_SLEEP_COORD_AWAKE;
        case SLEEP_STATE_DROWSY:     return SWARM_BRAIN_SLEEP_COORD_DROWSY;
        case SLEEP_STATE_LIGHT_NREM: return SWARM_BRAIN_SLEEP_COORD_LIGHT_NREM;
        case SLEEP_STATE_DEEP_NREM:  return SWARM_BRAIN_SLEEP_COORD_DEEP_NREM;
        case SLEEP_STATE_REM:        return SWARM_BRAIN_SLEEP_COORD_REM;
        default:                     return SWARM_BRAIN_SLEEP_COORD_AWAKE;
    }
}

float swarm_brain_sleep_get_heartbeat_factor(sleep_state_t state)
{
    switch (state) {
        case SLEEP_STATE_AWAKE:      return SWARM_BRAIN_SLEEP_HB_AWAKE;
        case SLEEP_STATE_DROWSY:     return SWARM_BRAIN_SLEEP_HB_DROWSY;
        case SLEEP_STATE_LIGHT_NREM: return SWARM_BRAIN_SLEEP_HB_LIGHT_NREM;
        case SLEEP_STATE_DEEP_NREM:  return SWARM_BRAIN_SLEEP_HB_DEEP_NREM;
        case SLEEP_STATE_REM:        return SWARM_BRAIN_SLEEP_HB_REM;
        default:                     return SWARM_BRAIN_SLEEP_HB_AWAKE;
    }
}

float swarm_brain_sleep_get_coherence_factor(sleep_state_t state)
{
    return swarm_brain_sleep_get_coord_factor(state);
}
