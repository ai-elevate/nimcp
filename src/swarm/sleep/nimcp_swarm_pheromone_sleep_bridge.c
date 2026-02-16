/**
 * @file nimcp_swarm_pheromone_sleep_bridge.c
 * @brief Swarm Pheromone-Sleep Integration Bridge Implementation
 * @version 1.0.0
 * @date 2025-12-18
 *
 * WHAT: Modulates swarm pheromone system based on sleep state
 * WHY:  Pheromone dynamics should vary with alertness level
 * HOW:  Sleep state callbacks dynamically adjust pheromone parameters
 *
 * @author NIMCP Development Team
 */

#include "swarm/sleep/nimcp_swarm_pheromone_sleep_bridge.h"
#include "constants/nimcp_constants.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/platform/nimcp_platform.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <stdlib.h>
#include <string.h>
#include "utils/fault_tolerance/nimcp_health_agent_macros.h"

NIMCP_DECLARE_HEALTH_AGENT_ATOMIC(swarm_pheromone_sleep_bridge)

struct swarm_pheromone_sleep_bridge_struct {
    bridge_base_t base;               /**< MUST be first: base bridge infrastructure */

    swarm_pheromone_sleep_config_t config;
    sleep_system_t sleep_system;
    swarm_pheromone_sleep_effects_t effects;
    bool callback_registered;
};

static void swarm_pheromone_on_sleep_state_change(sleep_state_t new_state, void* user_data)
{
    swarm_pheromone_sleep_bridge_t bridge = (swarm_pheromone_sleep_bridge_t)user_data;
    if (!bridge) return;

    nimcp_mutex_lock(bridge->base.mutex);

    bridge->effects.current_state = new_state;
    bridge->effects.decay_rate_factor = swarm_pheromone_sleep_get_decay_factor(new_state);
    bridge->effects.diffusion_rate_factor = swarm_pheromone_sleep_get_diff_factor(new_state);
    bridge->effects.detection_threshold_factor = swarm_pheromone_sleep_get_detect_factor(new_state);
    bridge->effects.signaling_enabled = (new_state != SLEEP_STATE_DEEP_NREM);

    NIMCP_LOGGING_DEBUG("Swarm pheromone sleep state changed to %d, decay=%.2f, diff=%.2f, detect=%.2f",
                        new_state, bridge->effects.decay_rate_factor,
                        bridge->effects.diffusion_rate_factor,
                        bridge->effects.detection_threshold_factor);

    nimcp_mutex_unlock(bridge->base.mutex);
}

int swarm_pheromone_sleep_default_config(swarm_pheromone_sleep_config_t* config)
{
    if (!config) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "config is NULL");

        return -1;

    }

    config->enable_decay_modulation = true;
    config->enable_diffusion_modulation = true;
    config->enable_detection_modulation = true;
    config->modulation_strength = NIMCP_SENSITIVITY_DEFAULT;

    return 0;
}

swarm_pheromone_sleep_bridge_t swarm_pheromone_sleep_bridge_create(
    const swarm_pheromone_sleep_config_t* config,
    sleep_system_t sleep_system)
{
    if (!config || !sleep_system) {
        NIMCP_LOGGING_ERROR("Invalid parameters for swarm pheromone sleep bridge creation");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "swarm_pheromone_sleep_bridge_create: required parameter is NULL (config, sleep_system)");
        return NULL;
    }

    swarm_pheromone_sleep_bridge_t bridge =
        (swarm_pheromone_sleep_bridge_t)nimcp_malloc(sizeof(struct swarm_pheromone_sleep_bridge_struct));
    if (!bridge) {
        NIMCP_LOGGING_ERROR("Failed to allocate swarm pheromone sleep bridge");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return NULL;
    }

    memset(bridge, 0, sizeof(struct swarm_pheromone_sleep_bridge_struct));
    memcpy(&bridge->config, config, sizeof(swarm_pheromone_sleep_config_t));
    bridge->sleep_system = sleep_system;

    if (bridge_base_init(&bridge->base, 0, "swarm_pheromone_sleep") != 0) { nimcp_free(bridge); return NULL; }
    if (!bridge->base.mutex) {
        NIMCP_LOGGING_ERROR("Failed to create mutex for swarm pheromone sleep bridge");
        nimcp_free(bridge);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "swarm_pheromone_sleep_bridge_create: bridge->base is NULL");
        return NULL;
    }

    bridge->effects.decay_rate_factor = SWARM_PHEROMONE_SLEEP_DECAY_AWAKE;
    bridge->effects.diffusion_rate_factor = SWARM_PHEROMONE_SLEEP_DIFF_AWAKE;
    bridge->effects.detection_threshold_factor = SWARM_PHEROMONE_SLEEP_DETECT_AWAKE;
    bridge->effects.current_state = SLEEP_STATE_AWAKE;
    bridge->effects.sleep_pressure = 0.0f;
    bridge->effects.signaling_enabled = true;

    bridge->callback_registered = sleep_register_state_callback(
        sleep_system, swarm_pheromone_on_sleep_state_change, bridge);

    if (bridge->callback_registered) {
        sleep_state_t initial = sleep_get_current_state(sleep_system);
        swarm_pheromone_on_sleep_state_change(initial, bridge);
    }

    NIMCP_LOGGING_INFO("Swarm pheromone sleep bridge created");
    return bridge;
}

void swarm_pheromone_sleep_bridge_destroy(swarm_pheromone_sleep_bridge_t bridge)
{
    if (!bridge) return;

    if (bridge->callback_registered && bridge->sleep_system) {
        sleep_unregister_state_callback(bridge->sleep_system,
                                        swarm_pheromone_on_sleep_state_change, bridge);
    }

    if (bridge->base.mutex) {
        bridge_base_cleanup(&bridge->base);
    }

    nimcp_free(bridge);
    NIMCP_LOGGING_INFO("Swarm pheromone sleep bridge destroyed");
}

int swarm_pheromone_sleep_update(swarm_pheromone_sleep_bridge_t bridge)
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

int swarm_pheromone_sleep_get_effects(const swarm_pheromone_sleep_bridge_t bridge,
                                       swarm_pheromone_sleep_effects_t* effects)
{
    if (!bridge || !effects) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "swarm_pheromone_sleep_get_effects: required parameter is NULL (bridge, effects)");
        return -1;
    }

    nimcp_mutex_lock(bridge->base.mutex);
    memcpy(effects, &bridge->effects, sizeof(swarm_pheromone_sleep_effects_t));
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

float swarm_pheromone_sleep_get_decay_rate(const swarm_pheromone_sleep_bridge_t bridge, float base)
{
    if (!bridge) return base;

    nimcp_mutex_lock(bridge->base.mutex);
    float factor = bridge->config.enable_decay_modulation ?
        bridge->effects.decay_rate_factor : 1.0f;
    float result = base * (1.0f + (factor - 1.0f) * bridge->config.modulation_strength);
    nimcp_mutex_unlock(bridge->base.mutex);

    return result;
}

float swarm_pheromone_sleep_get_diffusion_rate(const swarm_pheromone_sleep_bridge_t bridge, float base)
{
    if (!bridge) return base;

    nimcp_mutex_lock(bridge->base.mutex);
    float factor = bridge->config.enable_diffusion_modulation ?
        bridge->effects.diffusion_rate_factor : 1.0f;
    float result = base * (1.0f + (factor - 1.0f) * bridge->config.modulation_strength);
    nimcp_mutex_unlock(bridge->base.mutex);

    return result;
}

float swarm_pheromone_sleep_get_detection_threshold(const swarm_pheromone_sleep_bridge_t bridge, float base)
{
    if (!bridge) return base;

    nimcp_mutex_lock(bridge->base.mutex);
    float factor = bridge->config.enable_detection_modulation ?
        bridge->effects.detection_threshold_factor : 1.0f;
    float result = base * (1.0f + (factor - 1.0f) * bridge->config.modulation_strength);
    nimcp_mutex_unlock(bridge->base.mutex);

    return result;
}

float swarm_pheromone_sleep_get_decay_factor(sleep_state_t state)
{
    switch (state) {
        case SLEEP_STATE_AWAKE:      return SWARM_PHEROMONE_SLEEP_DECAY_AWAKE;
        case SLEEP_STATE_DROWSY:     return SWARM_PHEROMONE_SLEEP_DECAY_DROWSY;
        case SLEEP_STATE_LIGHT_NREM: return SWARM_PHEROMONE_SLEEP_DECAY_LIGHT_NREM;
        case SLEEP_STATE_DEEP_NREM:  return SWARM_PHEROMONE_SLEEP_DECAY_DEEP_NREM;
        case SLEEP_STATE_REM:        return SWARM_PHEROMONE_SLEEP_DECAY_REM;
        default:                     return SWARM_PHEROMONE_SLEEP_DECAY_AWAKE;
    }
}

float swarm_pheromone_sleep_get_diff_factor(sleep_state_t state)
{
    switch (state) {
        case SLEEP_STATE_AWAKE:      return SWARM_PHEROMONE_SLEEP_DIFF_AWAKE;
        case SLEEP_STATE_DROWSY:     return SWARM_PHEROMONE_SLEEP_DIFF_DROWSY;
        case SLEEP_STATE_LIGHT_NREM: return SWARM_PHEROMONE_SLEEP_DIFF_LIGHT_NREM;
        case SLEEP_STATE_DEEP_NREM:  return SWARM_PHEROMONE_SLEEP_DIFF_DEEP_NREM;
        case SLEEP_STATE_REM:        return SWARM_PHEROMONE_SLEEP_DIFF_REM;
        default:                     return SWARM_PHEROMONE_SLEEP_DIFF_AWAKE;
    }
}

float swarm_pheromone_sleep_get_detect_factor(sleep_state_t state)
{
    switch (state) {
        case SLEEP_STATE_AWAKE:      return SWARM_PHEROMONE_SLEEP_DETECT_AWAKE;
        case SLEEP_STATE_DROWSY:     return SWARM_PHEROMONE_SLEEP_DETECT_DROWSY;
        case SLEEP_STATE_LIGHT_NREM: return SWARM_PHEROMONE_SLEEP_DETECT_LIGHT_NREM;
        case SLEEP_STATE_DEEP_NREM:  return SWARM_PHEROMONE_SLEEP_DETECT_DEEP_NREM;
        case SLEEP_STATE_REM:        return SWARM_PHEROMONE_SLEEP_DETECT_REM;
        default:                     return SWARM_PHEROMONE_SLEEP_DETECT_AWAKE;
    }
}
