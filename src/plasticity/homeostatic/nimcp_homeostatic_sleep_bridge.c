/**
 * @file nimcp_homeostatic_sleep_bridge.c
 * @brief Sleep-Homeostatic Plasticity Integration Bridge Implementation
 * @version 1.0.0
 * @date 2025-12-17
 *
 * Implements Tononi's Synaptic Homeostasis Hypothesis (SHY):
 * Sleep is the primary time for homeostatic synaptic downscaling.
 */

#include "plasticity/homeostatic/nimcp_homeostatic_sleep_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/platform/nimcp_platform_mutex.h"
#include <string.h>

struct homeostatic_sleep_bridge_struct {
    homeostatic_sleep_config_t config;
    sleep_system_t sleep_system;
    homeostatic_sleep_effects_t effects;
    nimcp_mutex_t* mutex;
    bool callback_registered;  /* Track if callback is registered for cleanup */
};

/* Forward declarations */
static void homeostatic_on_sleep_state_change(sleep_state_t new_state, void* user_data);

/**
 * WHAT: Callback invoked when sleep state changes
 * WHY:  Immediately update homeostatic scaling for new sleep state
 * HOW:  Called by sleep system via observer pattern
 *
 * BIOLOGICAL BASIS:
 * - Implements Tononi's Synaptic Homeostasis Hypothesis (SHY)
 * - Sleep is the primary time for homeostatic synaptic downscaling
 * - Deep NREM is most critical for synaptic renormalization
 */
static void homeostatic_on_sleep_state_change(sleep_state_t new_state, void* user_data)
{
    homeostatic_sleep_bridge_t bridge = (homeostatic_sleep_bridge_t)user_data;

    if (!bridge) {
        NIMCP_LOGGING_ERROR("NULL bridge in sleep state callback");
        return;
    }

    NIMCP_LOGGING_DEBUG("Homeostatic bridge received sleep state: %d", new_state);

    nimcp_mutex_lock(bridge->mutex);

    bridge->effects.current_state = new_state;
    bridge->effects.is_deep_nrem = (new_state == SLEEP_STATE_DEEP_NREM);

    if (bridge->config.enable_scaling_modulation) {
        float scale_base = homeostatic_sleep_scaling_for_state(new_state);
        bridge->effects.scaling_rate_factor = scale_base * bridge->config.modulation_strength;

        /* Apply deep NREM boost */
        if (bridge->effects.is_deep_nrem) {
            bridge->effects.scaling_rate_factor *= bridge->config.deep_nrem_scaling_boost;
        }
    }

    if (bridge->config.enable_target_modulation) {
        bridge->effects.target_rate_modifier = homeostatic_sleep_target_for_state(new_state);
    }

    if (bridge->config.enable_pruning_modulation) {
        bridge->effects.pruning_threshold_mod =
            homeostatic_sleep_pruning_for_state(new_state) * bridge->config.modulation_strength;
    }

    /* Scaling is active if rate factor > 0 */
    bridge->effects.scaling_active = (bridge->effects.scaling_rate_factor > 0.01f);

    nimcp_mutex_unlock(bridge->mutex);

    NIMCP_LOGGING_DEBUG("Homeostatic modulated: scaling=%.2f, target=%.2f, pruning=%.2f",
                        bridge->effects.scaling_rate_factor,
                        bridge->effects.target_rate_modifier,
                        bridge->effects.pruning_threshold_mod);
}

int homeostatic_sleep_default_config(homeostatic_sleep_config_t* config) {
    if (!config) return -1;
    config->enable_scaling_modulation = true;
    config->enable_target_modulation = true;
    config->enable_pruning_modulation = true;
    config->modulation_strength = 1.0f;
    config->deep_nrem_scaling_boost = 1.2f;  /* Extra 20% boost in deep NREM */
    return 0;
}

homeostatic_sleep_bridge_t homeostatic_sleep_bridge_create(
    const homeostatic_sleep_config_t* config,
    sleep_system_t sleep_system)
{
    if (!sleep_system) {
        NIMCP_LOGGING_ERROR("homeostatic_sleep_bridge_create: NULL sleep_system");
        return NULL;
    }

    struct homeostatic_sleep_bridge_struct* bridge =
        (struct homeostatic_sleep_bridge_struct*)nimcp_malloc(
            sizeof(struct homeostatic_sleep_bridge_struct));
    if (!bridge) return NULL;

    memset(bridge, 0, sizeof(struct homeostatic_sleep_bridge_struct));

    if (config) {
        bridge->config = *config;
    } else {
        homeostatic_sleep_default_config(&bridge->config);
    }

    bridge->sleep_system = sleep_system;
    bridge->effects.scaling_rate_factor = 0.0f;
    bridge->effects.target_rate_modifier = 1.0f;
    bridge->effects.pruning_threshold_mod = 0.0f;
    bridge->effects.scaling_active = false;
    bridge->effects.is_deep_nrem = false;

    bridge->mutex = nimcp_platform_mutex_create();
    if (!bridge->mutex) {
        nimcp_free(bridge);
        return NULL;
    }

    /* Register callback for automatic state updates */
    bridge->callback_registered = sleep_register_state_callback(
        sleep_system,
        homeostatic_on_sleep_state_change,
        bridge);

    if (!bridge->callback_registered) {
        NIMCP_LOGGING_WARN("Failed to register sleep state callback - will use polling");
    } else {
        NIMCP_LOGGING_DEBUG("Registered sleep state callback for homeostatic bridge");
    }

    /* Get initial state immediately */
    sleep_state_t initial_state = sleep_get_current_state(sleep_system);
    homeostatic_on_sleep_state_change(initial_state, bridge);

    NIMCP_LOGGING_INFO("Homeostatic-sleep bridge created");
    return bridge;
}

void homeostatic_sleep_bridge_destroy(homeostatic_sleep_bridge_t bridge) {
    if (!bridge) return;

    /* Unregister callback if it was registered */
    if (bridge->callback_registered && bridge->sleep_system) {
        bool unregistered = sleep_unregister_state_callback(
            bridge->sleep_system,
            homeostatic_on_sleep_state_change,
            bridge);

        if (unregistered) {
            NIMCP_LOGGING_DEBUG("Unregistered sleep state callback for homeostatic bridge");
        }
    }

    if (bridge->mutex) nimcp_mutex_destroy(bridge->mutex);
    nimcp_free(bridge);
}

int homeostatic_sleep_update(homeostatic_sleep_bridge_t bridge) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);

    sleep_state_t state = sleep_get_current_state(bridge->sleep_system);
    float pressure = sleep_get_pressure(bridge->sleep_system);

    bridge->effects.current_state = state;
    bridge->effects.sleep_pressure = pressure;
    bridge->effects.is_deep_nrem = (state == SLEEP_STATE_DEEP_NREM);

    if (bridge->config.enable_scaling_modulation) {
        float scale_base = homeostatic_sleep_scaling_for_state(state);
        bridge->effects.scaling_rate_factor = scale_base * bridge->config.modulation_strength;

        /* Apply deep NREM boost */
        if (bridge->effects.is_deep_nrem) {
            bridge->effects.scaling_rate_factor *= bridge->config.deep_nrem_scaling_boost;
        }
    }

    if (bridge->config.enable_target_modulation) {
        bridge->effects.target_rate_modifier = homeostatic_sleep_target_for_state(state);
    }

    if (bridge->config.enable_pruning_modulation) {
        bridge->effects.pruning_threshold_mod =
            homeostatic_sleep_pruning_for_state(state) * bridge->config.modulation_strength;
    }

    /* Scaling is active if rate factor > 0 */
    bridge->effects.scaling_active = (bridge->effects.scaling_rate_factor > 0.01f);

    nimcp_mutex_unlock(bridge->mutex);

    NIMCP_LOGGING_DEBUG("Homeostatic sleep update: state=%d, scaling=%.2f, target=%.2f, pruning=%.2f",
                       state, bridge->effects.scaling_rate_factor,
                       bridge->effects.target_rate_modifier,
                       bridge->effects.pruning_threshold_mod);

    return 0;
}

int homeostatic_sleep_get_effects(
    const homeostatic_sleep_bridge_t bridge,
    homeostatic_sleep_effects_t* effects)
{
    if (!bridge || !effects) return -1;
    nimcp_mutex_lock(bridge->mutex);
    *effects = bridge->effects;
    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

float homeostatic_sleep_get_scaling_rate(const homeostatic_sleep_bridge_t bridge) {
    if (!bridge) return 0.0f;
    nimcp_mutex_lock(bridge->mutex);
    float result = bridge->effects.scaling_rate_factor;
    nimcp_mutex_unlock(bridge->mutex);
    return result;
}

bool homeostatic_sleep_is_scaling_active(const homeostatic_sleep_bridge_t bridge) {
    if (!bridge) return false;
    nimcp_mutex_lock(bridge->mutex);
    bool result = bridge->effects.scaling_active;
    nimcp_mutex_unlock(bridge->mutex);
    return result;
}

float homeostatic_sleep_scaling_for_state(sleep_state_t state) {
    switch (state) {
        case SLEEP_STATE_AWAKE:      return HOMEO_SLEEP_SCALE_RATE_AWAKE;
        case SLEEP_STATE_DROWSY:     return HOMEO_SLEEP_SCALE_RATE_DROWSY;
        case SLEEP_STATE_LIGHT_NREM: return HOMEO_SLEEP_SCALE_RATE_LIGHT_NREM;
        case SLEEP_STATE_DEEP_NREM:  return HOMEO_SLEEP_SCALE_RATE_DEEP_NREM;
        case SLEEP_STATE_REM:        return HOMEO_SLEEP_SCALE_RATE_REM;
        default:                     return HOMEO_SLEEP_SCALE_RATE_AWAKE;
    }
}

float homeostatic_sleep_target_for_state(sleep_state_t state) {
    switch (state) {
        case SLEEP_STATE_AWAKE:      return HOMEO_SLEEP_TARGET_AWAKE;
        case SLEEP_STATE_DROWSY:     return HOMEO_SLEEP_TARGET_DROWSY;
        case SLEEP_STATE_LIGHT_NREM: return HOMEO_SLEEP_TARGET_LIGHT_NREM;
        case SLEEP_STATE_DEEP_NREM:  return HOMEO_SLEEP_TARGET_DEEP_NREM;
        case SLEEP_STATE_REM:        return HOMEO_SLEEP_TARGET_REM;
        default:                     return HOMEO_SLEEP_TARGET_AWAKE;
    }
}

float homeostatic_sleep_pruning_for_state(sleep_state_t state) {
    switch (state) {
        case SLEEP_STATE_AWAKE:      return HOMEO_SLEEP_PRUNE_AWAKE;
        case SLEEP_STATE_DROWSY:     return HOMEO_SLEEP_PRUNE_DROWSY;
        case SLEEP_STATE_LIGHT_NREM: return HOMEO_SLEEP_PRUNE_LIGHT_NREM;
        case SLEEP_STATE_DEEP_NREM:  return HOMEO_SLEEP_PRUNE_DEEP_NREM;
        case SLEEP_STATE_REM:        return HOMEO_SLEEP_PRUNE_REM;
        default:                     return HOMEO_SLEEP_PRUNE_AWAKE;
    }
}
