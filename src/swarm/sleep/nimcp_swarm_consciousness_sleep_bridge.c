/**
 * @file nimcp_swarm_consciousness_sleep_bridge.c
 * @brief Swarm Consciousness-Sleep Integration Bridge Implementation
 * @version 1.0.0
 * @date 2025-12-18
 *
 * WHAT: Modulates swarm consciousness based on sleep state
 * WHY:  Collective phi and integration should vary with consciousness level
 * HOW:  Sleep state callbacks dynamically adjust consciousness parameters
 *
 * @author NIMCP Development Team
 */

#include "swarm/sleep/nimcp_swarm_consciousness_sleep_bridge.h"
#include "utils/platform/nimcp_platform.h"
#include "utils/logging/nimcp_logging.h"
#include <stdlib.h>
#include <string.h>

struct swarm_consciousness_sleep_bridge_struct {
    swarm_consciousness_sleep_config_t config;
    sleep_system_t sleep_system;
    swarm_consciousness_sleep_effects_t effects;
    nimcp_mutex_t* mutex;
    bool callback_registered;
};

static void swarm_consciousness_on_sleep_state_change(sleep_state_t new_state, void* user_data)
{
    swarm_consciousness_sleep_bridge_t bridge = (swarm_consciousness_sleep_bridge_t)user_data;
    if (!bridge) return;

    nimcp_mutex_lock(bridge->mutex);

    bridge->effects.current_state = new_state;
    bridge->effects.phi_factor = swarm_consciousness_sleep_get_phi_factor(new_state);
    bridge->effects.integration_factor = swarm_consciousness_sleep_get_integration_factor(new_state);
    bridge->effects.coherence_factor = swarm_consciousness_sleep_get_coherence_factor(new_state);
    bridge->effects.consciousness_enabled = (new_state != SLEEP_STATE_DEEP_NREM);

    NIMCP_LOGGING_DEBUG("Swarm consciousness sleep state changed to %d, phi=%.2f, integration=%.2f",
                        new_state, bridge->effects.phi_factor, bridge->effects.integration_factor);

    nimcp_mutex_unlock(bridge->mutex);
}

int swarm_consciousness_sleep_default_config(swarm_consciousness_sleep_config_t* config)
{
    if (!config) return -1;

    config->enable_phi_modulation = true;
    config->enable_integration_modulation = true;
    config->enable_coherence_modulation = true;
    config->modulation_strength = 1.0f;

    return 0;
}

swarm_consciousness_sleep_bridge_t swarm_consciousness_sleep_bridge_create(
    const swarm_consciousness_sleep_config_t* config,
    sleep_system_t sleep_system)
{
    if (!config || !sleep_system) {
        NIMCP_LOGGING_ERROR("Invalid parameters for swarm consciousness sleep bridge creation");
        return NULL;
    }

    swarm_consciousness_sleep_bridge_t bridge =
        (swarm_consciousness_sleep_bridge_t)nimcp_malloc(sizeof(struct swarm_consciousness_sleep_bridge_struct));
    if (!bridge) {
        NIMCP_LOGGING_ERROR("Failed to allocate swarm consciousness sleep bridge");
        return NULL;
    }

    memset(bridge, 0, sizeof(struct swarm_consciousness_sleep_bridge_struct));
    memcpy(&bridge->config, config, sizeof(swarm_consciousness_sleep_config_t));
    bridge->sleep_system = sleep_system;

    bridge->mutex = nimcp_platform_mutex_create();
    if (!bridge->mutex) {
        NIMCP_LOGGING_ERROR("Failed to create mutex for swarm consciousness sleep bridge");
        nimcp_free(bridge);
        return NULL;
    }

    bridge->effects.phi_factor = SWARM_CONSCIOUSNESS_SLEEP_PHI_AWAKE;
    bridge->effects.integration_factor = SWARM_CONSCIOUSNESS_SLEEP_INT_AWAKE;
    bridge->effects.coherence_factor = 1.0f;
    bridge->effects.current_state = SLEEP_STATE_AWAKE;
    bridge->effects.sleep_pressure = 0.0f;
    bridge->effects.consciousness_enabled = true;

    bridge->callback_registered = sleep_register_state_callback(
        sleep_system, swarm_consciousness_on_sleep_state_change, bridge);

    if (bridge->callback_registered) {
        sleep_state_t initial = sleep_get_current_state(sleep_system);
        swarm_consciousness_on_sleep_state_change(initial, bridge);
    }

    NIMCP_LOGGING_INFO("Swarm consciousness sleep bridge created");
    return bridge;
}

void swarm_consciousness_sleep_bridge_destroy(swarm_consciousness_sleep_bridge_t bridge)
{
    if (!bridge) return;

    if (bridge->callback_registered && bridge->sleep_system) {
        sleep_unregister_state_callback(bridge->sleep_system,
                                        swarm_consciousness_on_sleep_state_change, bridge);
    }

    if (bridge->mutex) {
        nimcp_platform_mutex_destroy(bridge->mutex);
    }

    nimcp_free(bridge);
    NIMCP_LOGGING_INFO("Swarm consciousness sleep bridge destroyed");
}

int swarm_consciousness_sleep_update(swarm_consciousness_sleep_bridge_t bridge)
{
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);
    if (bridge->sleep_system) {
        bridge->effects.sleep_pressure = sleep_get_pressure(bridge->sleep_system);
    }
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

int swarm_consciousness_sleep_get_effects(const swarm_consciousness_sleep_bridge_t bridge,
                                           swarm_consciousness_sleep_effects_t* effects)
{
    if (!bridge || !effects) return -1;

    nimcp_mutex_lock(bridge->mutex);
    memcpy(effects, &bridge->effects, sizeof(swarm_consciousness_sleep_effects_t));
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

float swarm_consciousness_sleep_get_phi(const swarm_consciousness_sleep_bridge_t bridge, float base)
{
    if (!bridge) return base;

    nimcp_mutex_lock(bridge->mutex);
    float factor = bridge->config.enable_phi_modulation ?
        bridge->effects.phi_factor : 1.0f;
    float result = base * (1.0f + (factor - 1.0f) * bridge->config.modulation_strength);
    nimcp_mutex_unlock(bridge->mutex);

    return result;
}

float swarm_consciousness_sleep_get_integration(const swarm_consciousness_sleep_bridge_t bridge, float base)
{
    if (!bridge) return base;

    nimcp_mutex_lock(bridge->mutex);
    float factor = bridge->config.enable_integration_modulation ?
        bridge->effects.integration_factor : 1.0f;
    float result = base * (1.0f + (factor - 1.0f) * bridge->config.modulation_strength);
    nimcp_mutex_unlock(bridge->mutex);

    return result;
}

float swarm_consciousness_sleep_get_coherence(const swarm_consciousness_sleep_bridge_t bridge, float base)
{
    if (!bridge) return base;

    nimcp_mutex_lock(bridge->mutex);
    float factor = bridge->config.enable_coherence_modulation ?
        bridge->effects.coherence_factor : 1.0f;
    float result = base * (1.0f + (factor - 1.0f) * bridge->config.modulation_strength);
    nimcp_mutex_unlock(bridge->mutex);

    return result;
}

float swarm_consciousness_sleep_get_phi_factor(sleep_state_t state)
{
    switch (state) {
        case SLEEP_STATE_AWAKE:      return SWARM_CONSCIOUSNESS_SLEEP_PHI_AWAKE;
        case SLEEP_STATE_DROWSY:     return SWARM_CONSCIOUSNESS_SLEEP_PHI_DROWSY;
        case SLEEP_STATE_LIGHT_NREM: return SWARM_CONSCIOUSNESS_SLEEP_PHI_LIGHT_NREM;
        case SLEEP_STATE_DEEP_NREM:  return SWARM_CONSCIOUSNESS_SLEEP_PHI_DEEP_NREM;
        case SLEEP_STATE_REM:        return SWARM_CONSCIOUSNESS_SLEEP_PHI_REM;
        default:                     return SWARM_CONSCIOUSNESS_SLEEP_PHI_AWAKE;
    }
}

float swarm_consciousness_sleep_get_integration_factor(sleep_state_t state)
{
    switch (state) {
        case SLEEP_STATE_AWAKE:      return SWARM_CONSCIOUSNESS_SLEEP_INT_AWAKE;
        case SLEEP_STATE_DROWSY:     return SWARM_CONSCIOUSNESS_SLEEP_INT_DROWSY;
        case SLEEP_STATE_LIGHT_NREM: return SWARM_CONSCIOUSNESS_SLEEP_INT_LIGHT_NREM;
        case SLEEP_STATE_DEEP_NREM:  return SWARM_CONSCIOUSNESS_SLEEP_INT_DEEP_NREM;
        case SLEEP_STATE_REM:        return SWARM_CONSCIOUSNESS_SLEEP_INT_REM;
        default:                     return SWARM_CONSCIOUSNESS_SLEEP_INT_AWAKE;
    }
}

float swarm_consciousness_sleep_get_coherence_factor(sleep_state_t state)
{
    /* Coherence uses same pattern as phi */
    return swarm_consciousness_sleep_get_phi_factor(state);
}
