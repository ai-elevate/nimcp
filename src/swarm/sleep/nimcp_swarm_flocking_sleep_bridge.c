/**
 * @file nimcp_swarm_flocking_sleep_bridge.c
 * @brief Swarm Flocking-Sleep Integration Bridge Implementation
 * @version 1.0.0
 * @date 2025-12-18
 *
 * WHAT: Modulates swarm flocking behavior based on sleep state
 * WHY:  Separation, alignment, and cohesion forces should vary with alertness
 * HOW:  Sleep state callbacks dynamically adjust flocking parameters
 *
 * @author NIMCP Development Team
 */

#include "swarm/sleep/nimcp_swarm_flocking_sleep_bridge.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/platform/nimcp_platform.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <stdlib.h>
#include <string.h>

#include <stddef.h>  /* for NULL */
//=============================================================================
// Health Agent Integration (Phase 8: System-Wide Health Integration)
//=============================================================================
struct nimcp_health_agent;
typedef struct nimcp_health_agent nimcp_health_agent_t;
extern void nimcp_health_agent_heartbeat_ex(nimcp_health_agent_t* agent,
                                             const char* operation,
                                             float progress);

/** Global health agent for swarm_flocking_sleep_bridge module */
static nimcp_health_agent_t* g_swarm_flocking_sleep_bridge_health_agent = NULL;

/**
 * @brief Set health agent for swarm_flocking_sleep_bridge heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
static void swarm_flocking_sleep_bridge_set_health_agent(nimcp_health_agent_t* agent) {
    g_swarm_flocking_sleep_bridge_health_agent = agent;
}

/** @brief Send heartbeat from swarm_flocking_sleep_bridge module */
static inline void swarm_flocking_sleep_bridge_heartbeat(const char* operation, float progress) {
    if (g_swarm_flocking_sleep_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_swarm_flocking_sleep_bridge_health_agent, operation, progress);
    }
}


struct swarm_flocking_sleep_bridge_struct {
    bridge_base_t base;               /**< MUST be first: base bridge infrastructure */

    swarm_flocking_sleep_config_t config;
    sleep_system_t sleep_system;
    swarm_flocking_sleep_effects_t effects;
    bool callback_registered;
};

static void swarm_flocking_on_sleep_state_change(sleep_state_t new_state, void* user_data)
{
    swarm_flocking_sleep_bridge_t bridge = (swarm_flocking_sleep_bridge_t)user_data;
    if (!bridge) return;

    nimcp_mutex_lock(bridge->base.mutex);

    bridge->effects.current_state = new_state;

    float force = swarm_flocking_sleep_get_force_factor(new_state);
    float update = swarm_flocking_sleep_get_update_factor(new_state);

    if (bridge->config.enable_force_modulation) {
        bridge->effects.separation_factor = 1.0f + (force - 1.0f) * bridge->config.modulation_strength;
        bridge->effects.alignment_factor = 1.0f + (force - 1.0f) * bridge->config.modulation_strength;
        bridge->effects.cohesion_factor = 1.0f + (force - 1.0f) * bridge->config.modulation_strength;
    }
    if (bridge->config.enable_update_modulation) {
        bridge->effects.update_frequency_factor = 1.0f + (update - 1.0f) * bridge->config.modulation_strength;
    }
    bridge->effects.flocking_enabled = (new_state != SLEEP_STATE_DEEP_NREM);

    NIMCP_LOGGING_DEBUG("Swarm flocking sleep state changed to %d, force=%.2f, update=%.2f",
                        new_state, force, update);

    nimcp_mutex_unlock(bridge->base.mutex);
}

int swarm_flocking_sleep_default_config(swarm_flocking_sleep_config_t* config)
{
    if (!config) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "config is NULL");

        return -1;

    }

    config->enable_force_modulation = true;
    config->enable_update_modulation = true;
    config->enable_formation_modulation = true;
    config->modulation_strength = 1.0f;

    return 0;
}

swarm_flocking_sleep_bridge_t swarm_flocking_sleep_bridge_create(
    const swarm_flocking_sleep_config_t* config,
    sleep_system_t sleep_system)
{
    if (!config || !sleep_system) {
        NIMCP_LOGGING_ERROR("Invalid parameters for swarm flocking sleep bridge creation");
        return NULL;
    }

    swarm_flocking_sleep_bridge_t bridge =
        (swarm_flocking_sleep_bridge_t)nimcp_malloc(sizeof(struct swarm_flocking_sleep_bridge_struct));
    if (!bridge) {
        NIMCP_LOGGING_ERROR("Failed to allocate swarm flocking sleep bridge");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return NULL;
    }

    memset(bridge, 0, sizeof(struct swarm_flocking_sleep_bridge_struct));
    memcpy(&bridge->config, config, sizeof(swarm_flocking_sleep_config_t));
    bridge->sleep_system = sleep_system;

    if (bridge_base_init(&bridge->base, 0, "swarm_flocking_sleep") != 0) { nimcp_free(bridge); return NULL; }
    if (!bridge->base.mutex) {
        NIMCP_LOGGING_ERROR("Failed to create mutex for swarm flocking sleep bridge");
        nimcp_free(bridge);
        return NULL;
    }

    bridge->effects.separation_factor = SWARM_FLOCKING_SLEEP_FORCE_AWAKE;
    bridge->effects.alignment_factor = SWARM_FLOCKING_SLEEP_FORCE_AWAKE;
    bridge->effects.cohesion_factor = SWARM_FLOCKING_SLEEP_FORCE_AWAKE;
    bridge->effects.update_frequency_factor = SWARM_FLOCKING_SLEEP_UPDATE_AWAKE;
    bridge->effects.current_state = SLEEP_STATE_AWAKE;
    bridge->effects.sleep_pressure = 0.0f;
    bridge->effects.flocking_enabled = true;

    bridge->callback_registered = sleep_register_state_callback(
        sleep_system, swarm_flocking_on_sleep_state_change, bridge);

    if (bridge->callback_registered) {
        sleep_state_t initial = sleep_get_current_state(sleep_system);
        swarm_flocking_on_sleep_state_change(initial, bridge);
    }

    NIMCP_LOGGING_INFO("Swarm flocking sleep bridge created");
    return bridge;
}

void swarm_flocking_sleep_bridge_destroy(swarm_flocking_sleep_bridge_t bridge)
{
    if (!bridge) return;

    if (bridge->callback_registered && bridge->sleep_system) {
        sleep_unregister_state_callback(bridge->sleep_system,
                                        swarm_flocking_on_sleep_state_change, bridge);
    }

    if (bridge->base.mutex) {
        bridge_base_cleanup(&bridge->base);
    }

    nimcp_free(bridge);
    NIMCP_LOGGING_INFO("Swarm flocking sleep bridge destroyed");
}

int swarm_flocking_sleep_update(swarm_flocking_sleep_bridge_t bridge)
{
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return -1;

    }

    nimcp_mutex_lock(bridge->base.mutex);
    if (bridge->sleep_system) {
        bridge->effects.sleep_pressure = sleep_get_pressure(bridge->sleep_system);
    }
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

int swarm_flocking_sleep_get_effects(const swarm_flocking_sleep_bridge_t bridge,
                                      swarm_flocking_sleep_effects_t* effects)
{
    if (!bridge || !effects) return -1;

    nimcp_mutex_lock(bridge->base.mutex);
    memcpy(effects, &bridge->effects, sizeof(swarm_flocking_sleep_effects_t));
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

float swarm_flocking_sleep_get_separation(const swarm_flocking_sleep_bridge_t bridge, float base)
{
    if (!bridge) return base;

    nimcp_mutex_lock(bridge->base.mutex);
    float result = base * bridge->effects.separation_factor;
    nimcp_mutex_unlock(bridge->base.mutex);

    return result;
}

float swarm_flocking_sleep_get_alignment(const swarm_flocking_sleep_bridge_t bridge, float base)
{
    if (!bridge) return base;

    nimcp_mutex_lock(bridge->base.mutex);
    float result = base * bridge->effects.alignment_factor;
    nimcp_mutex_unlock(bridge->base.mutex);

    return result;
}

float swarm_flocking_sleep_get_cohesion(const swarm_flocking_sleep_bridge_t bridge, float base)
{
    if (!bridge) return base;

    nimcp_mutex_lock(bridge->base.mutex);
    float result = base * bridge->effects.cohesion_factor;
    nimcp_mutex_unlock(bridge->base.mutex);

    return result;
}

float swarm_flocking_sleep_get_force_factor(sleep_state_t state)
{
    switch (state) {
        case SLEEP_STATE_AWAKE:      return SWARM_FLOCKING_SLEEP_FORCE_AWAKE;
        case SLEEP_STATE_DROWSY:     return SWARM_FLOCKING_SLEEP_FORCE_DROWSY;
        case SLEEP_STATE_LIGHT_NREM: return SWARM_FLOCKING_SLEEP_FORCE_LIGHT_NREM;
        case SLEEP_STATE_DEEP_NREM:  return SWARM_FLOCKING_SLEEP_FORCE_DEEP_NREM;
        case SLEEP_STATE_REM:        return SWARM_FLOCKING_SLEEP_FORCE_REM;
        default:                     return SWARM_FLOCKING_SLEEP_FORCE_AWAKE;
    }
}

float swarm_flocking_sleep_get_update_factor(sleep_state_t state)
{
    switch (state) {
        case SLEEP_STATE_AWAKE:      return SWARM_FLOCKING_SLEEP_UPDATE_AWAKE;
        case SLEEP_STATE_DROWSY:     return SWARM_FLOCKING_SLEEP_UPDATE_DROWSY;
        case SLEEP_STATE_LIGHT_NREM: return SWARM_FLOCKING_SLEEP_UPDATE_LIGHT_NREM;
        case SLEEP_STATE_DEEP_NREM:  return SWARM_FLOCKING_SLEEP_UPDATE_DEEP_NREM;
        case SLEEP_STATE_REM:        return SWARM_FLOCKING_SLEEP_UPDATE_REM;
        default:                     return SWARM_FLOCKING_SLEEP_UPDATE_AWAKE;
    }
}
