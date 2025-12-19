/**
 * @file nimcp_adaptive_sleep_bridge.c
 * @brief Sleep-Adaptive Plasticity Integration Bridge Implementation
 * @version 1.0.0
 * @date 2025-12-17
 */

#include "plasticity/adaptive/nimcp_adaptive_sleep_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/platform/nimcp_platform_mutex.h"
#include <string.h>

struct adaptive_sleep_bridge_struct {
    adaptive_sleep_config_t config;
    sleep_system_t sleep_system;
    adaptive_sleep_effects_t effects;
    nimcp_mutex_t* mutex;
    bool callback_registered;  /* Track if callback is registered for cleanup */
};

/* Forward declarations */
static void adaptive_on_sleep_state_change(sleep_state_t new_state, void* user_data);

/**
 * WHAT: Callback invoked when sleep state changes
 * WHY:  Immediately update adaptive plasticity parameters for new sleep state
 * HOW:  Called by sleep system via observer pattern
 *
 * BIOLOGICAL BASIS:
 * - Firing rate homeostasis adapts slower during sleep
 * - Thresholds are stabilized during consolidation to preserve learned patterns
 * - Sparsity increases during NREM to focus on relevant memories
 * - Deep NREM freezes thresholds for stable replay
 */
static void adaptive_on_sleep_state_change(sleep_state_t new_state, void* user_data)
{
    adaptive_sleep_bridge_t bridge = (adaptive_sleep_bridge_t)user_data;

    if (!bridge) {
        NIMCP_LOGGING_ERROR("NULL bridge in sleep state callback");
        return;
    }

    NIMCP_LOGGING_DEBUG("Adaptive bridge received sleep state: %d", new_state);

    nimcp_mutex_lock(bridge->mutex);

    bridge->effects.current_state = new_state;

    if (bridge->config.enable_adaptation_modulation) {
        float adapt_base = adaptive_sleep_get_adapt_factor(new_state);
        bridge->effects.adaptation_rate_factor =
            1.0f + (adapt_base - 1.0f) * bridge->config.modulation_strength;
    }

    if (bridge->config.enable_sparsity_modulation) {
        float sparsity_base = adaptive_sleep_get_sparsity_factor(new_state);
        bridge->effects.sparsity_target = sparsity_base;
    }

    if (bridge->config.enable_reset_modulation) {
        float reset_base = adaptive_sleep_get_reset_factor(new_state);
        bridge->effects.soft_reset_factor =
            1.0f + (reset_base - 1.0f) * bridge->config.modulation_strength;
    }

    bridge->effects.freeze_thresholds = (new_state == SLEEP_STATE_DEEP_NREM);

    nimcp_mutex_unlock(bridge->mutex);

    NIMCP_LOGGING_DEBUG("Adaptive modulated: adapt=%.2f, sparsity=%.2f, reset=%.2f",
                        bridge->effects.adaptation_rate_factor,
                        bridge->effects.sparsity_target,
                        bridge->effects.soft_reset_factor);
}

int adaptive_sleep_default_config(adaptive_sleep_config_t* config) {
    if (!config) return -1;
    config->enable_adaptation_modulation = true;
    config->enable_sparsity_modulation = true;
    config->enable_reset_modulation = true;
    config->modulation_strength = 1.0f;
    return 0;
}

adaptive_sleep_bridge_t adaptive_sleep_bridge_create(
    const adaptive_sleep_config_t* config,
    sleep_system_t sleep_system)
{
    if (!sleep_system) {
        NIMCP_LOGGING_ERROR("adaptive_sleep_bridge_create: NULL sleep_system");
        return NULL;
    }

    struct adaptive_sleep_bridge_struct* bridge =
        (struct adaptive_sleep_bridge_struct*)nimcp_malloc(
            sizeof(struct adaptive_sleep_bridge_struct));
    if (!bridge) return NULL;

    memset(bridge, 0, sizeof(struct adaptive_sleep_bridge_struct));

    if (config) {
        bridge->config = *config;
    } else {
        adaptive_sleep_default_config(&bridge->config);
    }

    bridge->sleep_system = sleep_system;
    bridge->effects.adaptation_rate_factor = 1.0f;
    bridge->effects.sparsity_target = 0.75f;
    bridge->effects.soft_reset_factor = 1.0f;
    bridge->effects.freeze_thresholds = false;

    bridge->mutex = nimcp_platform_mutex_create();
    if (!bridge->mutex) {
        nimcp_free(bridge);
        return NULL;
    }

    /* Register callback for automatic state updates */
    bridge->callback_registered = sleep_register_state_callback(
        sleep_system,
        adaptive_on_sleep_state_change,
        bridge);

    if (!bridge->callback_registered) {
        NIMCP_LOGGING_WARN("Failed to register sleep state callback - will use polling");
    } else {
        NIMCP_LOGGING_DEBUG("Registered sleep state callback for adaptive bridge");
    }

    /* Get initial state immediately */
    sleep_state_t initial_state = sleep_get_current_state(sleep_system);
    adaptive_on_sleep_state_change(initial_state, bridge);

    NIMCP_LOGGING_INFO("Adaptive-sleep bridge created");
    return bridge;
}

void adaptive_sleep_bridge_destroy(adaptive_sleep_bridge_t bridge) {
    if (!bridge) return;

    /* Unregister callback if it was registered */
    if (bridge->callback_registered && bridge->sleep_system) {
        bool unregistered = sleep_unregister_state_callback(
            bridge->sleep_system,
            adaptive_on_sleep_state_change,
            bridge);

        if (unregistered) {
            NIMCP_LOGGING_DEBUG("Unregistered sleep state callback for adaptive bridge");
        }
    }

    if (bridge->mutex) nimcp_mutex_destroy(bridge->mutex);
    nimcp_free(bridge);
}

int adaptive_sleep_update(adaptive_sleep_bridge_t bridge) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);

    sleep_state_t state = sleep_get_current_state(bridge->sleep_system);
    float pressure = sleep_get_pressure(bridge->sleep_system);

    bridge->effects.current_state = state;
    bridge->effects.sleep_pressure = pressure;

    if (bridge->config.enable_adaptation_modulation) {
        float adapt_base = adaptive_sleep_get_adapt_factor(state);
        bridge->effects.adaptation_rate_factor =
            1.0f + (adapt_base - 1.0f) * bridge->config.modulation_strength;
    }

    if (bridge->config.enable_sparsity_modulation) {
        float sparsity_base = adaptive_sleep_get_sparsity_factor(state);
        bridge->effects.sparsity_target = sparsity_base;
    }

    if (bridge->config.enable_reset_modulation) {
        float reset_base = adaptive_sleep_get_reset_factor(state);
        bridge->effects.soft_reset_factor =
            1.0f + (reset_base - 1.0f) * bridge->config.modulation_strength;
    }

    bridge->effects.freeze_thresholds = (state == SLEEP_STATE_DEEP_NREM);

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int adaptive_sleep_get_effects(const adaptive_sleep_bridge_t bridge,
                                adaptive_sleep_effects_t* effects) {
    if (!bridge || !effects) return -1;
    nimcp_mutex_lock(bridge->mutex);
    *effects = bridge->effects;
    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

float adaptive_sleep_get_adaptation_rate(const adaptive_sleep_bridge_t bridge,
                                          float base_rate) {
    if (!bridge) return base_rate;
    nimcp_mutex_lock(bridge->mutex);
    float result = base_rate * bridge->effects.adaptation_rate_factor;
    nimcp_mutex_unlock(bridge->mutex);
    return result;
}

float adaptive_sleep_get_sparsity_target(const adaptive_sleep_bridge_t bridge,
                                          float base_sparsity) {
    if (!bridge) return base_sparsity;
    nimcp_mutex_lock(bridge->mutex);
    float result = bridge->effects.sparsity_target;
    nimcp_mutex_unlock(bridge->mutex);
    return result;
}

float adaptive_sleep_get_soft_reset(const adaptive_sleep_bridge_t bridge,
                                     float base_reset) {
    if (!bridge) return base_reset;
    nimcp_mutex_lock(bridge->mutex);
    float result = base_reset * bridge->effects.soft_reset_factor;
    nimcp_mutex_unlock(bridge->mutex);
    return result;
}

float adaptive_sleep_get_adapt_factor(sleep_state_t state) {
    switch (state) {
        case SLEEP_STATE_AWAKE:      return ADAPTIVE_SLEEP_ADAPT_AWAKE;
        case SLEEP_STATE_DROWSY:     return ADAPTIVE_SLEEP_ADAPT_DROWSY;
        case SLEEP_STATE_LIGHT_NREM: return ADAPTIVE_SLEEP_ADAPT_LIGHT_NREM;
        case SLEEP_STATE_DEEP_NREM:  return ADAPTIVE_SLEEP_ADAPT_DEEP_NREM;
        case SLEEP_STATE_REM:        return ADAPTIVE_SLEEP_ADAPT_REM;
        default:                     return ADAPTIVE_SLEEP_ADAPT_AWAKE;
    }
}

float adaptive_sleep_get_sparsity_factor(sleep_state_t state) {
    switch (state) {
        case SLEEP_STATE_AWAKE:      return ADAPTIVE_SLEEP_SPARSITY_AWAKE;
        case SLEEP_STATE_DROWSY:     return ADAPTIVE_SLEEP_SPARSITY_DROWSY;
        case SLEEP_STATE_LIGHT_NREM: return ADAPTIVE_SLEEP_SPARSITY_LIGHT_NREM;
        case SLEEP_STATE_DEEP_NREM:  return ADAPTIVE_SLEEP_SPARSITY_DEEP_NREM;
        case SLEEP_STATE_REM:        return ADAPTIVE_SLEEP_SPARSITY_REM;
        default:                     return ADAPTIVE_SLEEP_SPARSITY_AWAKE;
    }
}

float adaptive_sleep_get_reset_factor(sleep_state_t state) {
    switch (state) {
        case SLEEP_STATE_AWAKE:      return ADAPTIVE_SLEEP_RESET_AWAKE;
        case SLEEP_STATE_DROWSY:     return ADAPTIVE_SLEEP_RESET_DROWSY;
        case SLEEP_STATE_LIGHT_NREM: return ADAPTIVE_SLEEP_RESET_LIGHT_NREM;
        case SLEEP_STATE_DEEP_NREM:  return ADAPTIVE_SLEEP_RESET_DEEP_NREM;
        case SLEEP_STATE_REM:        return ADAPTIVE_SLEEP_RESET_REM;
        default:                     return ADAPTIVE_SLEEP_RESET_AWAKE;
    }
}
