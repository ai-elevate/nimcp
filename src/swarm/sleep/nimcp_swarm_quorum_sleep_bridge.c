/**
 * @file nimcp_swarm_quorum_sleep_bridge.c
 * @brief Swarm Quorum-Sleep Integration Bridge Implementation
 * @version 1.0.0
 * @date 2025-12-18
 *
 * WHAT: Modulates swarm quorum decisions based on sleep state
 * WHY:  Decision thresholds and signal dynamics should vary with alertness
 * HOW:  Sleep state callbacks dynamically adjust quorum parameters
 *
 * @author NIMCP Development Team
 */

#include "swarm/sleep/nimcp_swarm_quorum_sleep_bridge.h"
#include "utils/platform/nimcp_platform.h"
#include "utils/logging/nimcp_logging.h"
#include <stdlib.h>
#include <string.h>

struct swarm_quorum_sleep_bridge_struct {
    swarm_quorum_sleep_config_t config;
    sleep_system_t sleep_system;
    swarm_quorum_sleep_effects_t effects;
    nimcp_mutex_t* mutex;
    bool callback_registered;
};

static void swarm_quorum_on_sleep_state_change(sleep_state_t new_state, void* user_data)
{
    swarm_quorum_sleep_bridge_t bridge = (swarm_quorum_sleep_bridge_t)user_data;
    if (!bridge) return;

    nimcp_mutex_lock(bridge->mutex);

    bridge->effects.current_state = new_state;
    bridge->effects.decision_threshold_factor = swarm_quorum_sleep_get_thresh_factor(new_state);
    bridge->effects.signal_decay_factor = swarm_quorum_sleep_get_decay_factor(new_state);
    bridge->effects.commitment_rate_factor = swarm_quorum_sleep_get_commit_factor(new_state);
    bridge->effects.quorum_enabled = (new_state != SLEEP_STATE_DEEP_NREM);

    NIMCP_LOGGING_DEBUG("Swarm quorum sleep state changed to %d, thresh=%.2f, decay=%.2f, commit=%.2f",
                        new_state, bridge->effects.decision_threshold_factor,
                        bridge->effects.signal_decay_factor,
                        bridge->effects.commitment_rate_factor);

    nimcp_mutex_unlock(bridge->mutex);
}

int swarm_quorum_sleep_default_config(swarm_quorum_sleep_config_t* config)
{
    if (!config) return -1;

    config->enable_threshold_modulation = true;
    config->enable_decay_modulation = true;
    config->enable_commitment_modulation = true;
    config->modulation_strength = 1.0f;

    return 0;
}

swarm_quorum_sleep_bridge_t swarm_quorum_sleep_bridge_create(
    const swarm_quorum_sleep_config_t* config,
    sleep_system_t sleep_system)
{
    if (!config || !sleep_system) {
        NIMCP_LOGGING_ERROR("Invalid parameters for swarm quorum sleep bridge creation");
        return NULL;
    }

    swarm_quorum_sleep_bridge_t bridge =
        (swarm_quorum_sleep_bridge_t)nimcp_malloc(sizeof(struct swarm_quorum_sleep_bridge_struct));
    if (!bridge) {
        NIMCP_LOGGING_ERROR("Failed to allocate swarm quorum sleep bridge");
        return NULL;
    }

    memset(bridge, 0, sizeof(struct swarm_quorum_sleep_bridge_struct));
    memcpy(&bridge->config, config, sizeof(swarm_quorum_sleep_config_t));
    bridge->sleep_system = sleep_system;

    bridge->mutex = nimcp_platform_mutex_create();
    if (!bridge->mutex) {
        NIMCP_LOGGING_ERROR("Failed to create mutex for swarm quorum sleep bridge");
        nimcp_free(bridge);
        return NULL;
    }

    bridge->effects.decision_threshold_factor = SWARM_QUORUM_SLEEP_THRESH_AWAKE;
    bridge->effects.signal_decay_factor = SWARM_QUORUM_SLEEP_DECAY_AWAKE;
    bridge->effects.commitment_rate_factor = SWARM_QUORUM_SLEEP_COMMIT_AWAKE;
    bridge->effects.current_state = SLEEP_STATE_AWAKE;
    bridge->effects.sleep_pressure = 0.0f;
    bridge->effects.quorum_enabled = true;

    bridge->callback_registered = sleep_register_state_callback(
        sleep_system, swarm_quorum_on_sleep_state_change, bridge);

    if (bridge->callback_registered) {
        sleep_state_t initial = sleep_get_current_state(sleep_system);
        swarm_quorum_on_sleep_state_change(initial, bridge);
    }

    NIMCP_LOGGING_INFO("Swarm quorum sleep bridge created");
    return bridge;
}

void swarm_quorum_sleep_bridge_destroy(swarm_quorum_sleep_bridge_t bridge)
{
    if (!bridge) return;

    if (bridge->callback_registered && bridge->sleep_system) {
        sleep_unregister_state_callback(bridge->sleep_system,
                                        swarm_quorum_on_sleep_state_change, bridge);
    }

    if (bridge->mutex) {
        nimcp_platform_mutex_destroy(bridge->mutex);
    }

    nimcp_free(bridge);
    NIMCP_LOGGING_INFO("Swarm quorum sleep bridge destroyed");
}

int swarm_quorum_sleep_update(swarm_quorum_sleep_bridge_t bridge)
{
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);
    if (bridge->sleep_system) {
        bridge->effects.sleep_pressure = sleep_get_pressure(bridge->sleep_system);
    }
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

int swarm_quorum_sleep_get_effects(const swarm_quorum_sleep_bridge_t bridge,
                                    swarm_quorum_sleep_effects_t* effects)
{
    if (!bridge || !effects) return -1;

    nimcp_mutex_lock(bridge->mutex);
    memcpy(effects, &bridge->effects, sizeof(swarm_quorum_sleep_effects_t));
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

float swarm_quorum_sleep_get_decision_threshold(const swarm_quorum_sleep_bridge_t bridge, float base)
{
    if (!bridge) return base;

    nimcp_mutex_lock(bridge->mutex);
    float factor = bridge->config.enable_threshold_modulation ?
        bridge->effects.decision_threshold_factor : 1.0f;
    float result = base * (1.0f + (factor - 1.0f) * bridge->config.modulation_strength);
    nimcp_mutex_unlock(bridge->mutex);

    return result;
}

float swarm_quorum_sleep_get_signal_decay(const swarm_quorum_sleep_bridge_t bridge, float base)
{
    if (!bridge) return base;

    nimcp_mutex_lock(bridge->mutex);
    float factor = bridge->config.enable_decay_modulation ?
        bridge->effects.signal_decay_factor : 1.0f;
    float result = base * (1.0f + (factor - 1.0f) * bridge->config.modulation_strength);
    nimcp_mutex_unlock(bridge->mutex);

    return result;
}

float swarm_quorum_sleep_get_commitment_rate(const swarm_quorum_sleep_bridge_t bridge, float base)
{
    if (!bridge) return base;

    nimcp_mutex_lock(bridge->mutex);
    float factor = bridge->config.enable_commitment_modulation ?
        bridge->effects.commitment_rate_factor : 1.0f;
    float result = base * (1.0f + (factor - 1.0f) * bridge->config.modulation_strength);
    nimcp_mutex_unlock(bridge->mutex);

    return result;
}

float swarm_quorum_sleep_get_thresh_factor(sleep_state_t state)
{
    switch (state) {
        case SLEEP_STATE_AWAKE:      return SWARM_QUORUM_SLEEP_THRESH_AWAKE;
        case SLEEP_STATE_DROWSY:     return SWARM_QUORUM_SLEEP_THRESH_DROWSY;
        case SLEEP_STATE_LIGHT_NREM: return SWARM_QUORUM_SLEEP_THRESH_LIGHT_NREM;
        case SLEEP_STATE_DEEP_NREM:  return SWARM_QUORUM_SLEEP_THRESH_DEEP_NREM;
        case SLEEP_STATE_REM:        return SWARM_QUORUM_SLEEP_THRESH_REM;
        default:                     return SWARM_QUORUM_SLEEP_THRESH_AWAKE;
    }
}

float swarm_quorum_sleep_get_decay_factor(sleep_state_t state)
{
    switch (state) {
        case SLEEP_STATE_AWAKE:      return SWARM_QUORUM_SLEEP_DECAY_AWAKE;
        case SLEEP_STATE_DROWSY:     return SWARM_QUORUM_SLEEP_DECAY_DROWSY;
        case SLEEP_STATE_LIGHT_NREM: return SWARM_QUORUM_SLEEP_DECAY_LIGHT_NREM;
        case SLEEP_STATE_DEEP_NREM:  return SWARM_QUORUM_SLEEP_DECAY_DEEP_NREM;
        case SLEEP_STATE_REM:        return SWARM_QUORUM_SLEEP_DECAY_REM;
        default:                     return SWARM_QUORUM_SLEEP_DECAY_AWAKE;
    }
}

float swarm_quorum_sleep_get_commit_factor(sleep_state_t state)
{
    switch (state) {
        case SLEEP_STATE_AWAKE:      return SWARM_QUORUM_SLEEP_COMMIT_AWAKE;
        case SLEEP_STATE_DROWSY:     return SWARM_QUORUM_SLEEP_COMMIT_DROWSY;
        case SLEEP_STATE_LIGHT_NREM: return SWARM_QUORUM_SLEEP_COMMIT_LIGHT_NREM;
        case SLEEP_STATE_DEEP_NREM:  return SWARM_QUORUM_SLEEP_COMMIT_DEEP_NREM;
        case SLEEP_STATE_REM:        return SWARM_QUORUM_SLEEP_COMMIT_REM;
        default:                     return SWARM_QUORUM_SLEEP_COMMIT_AWAKE;
    }
}
