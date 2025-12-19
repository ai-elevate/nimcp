/**
 * @file nimcp_mental_health_sleep_bridge.c
 * @brief Sleep-Mental Health Integration Bridge Implementation
 * @version 1.0.0
 * @date 2025-12-17
 */

#include "cognitive/mental_health/nimcp_mental_health_sleep_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/platform/nimcp_platform_mutex.h"
#include <string.h>

struct mental_health_sleep_bridge_struct {
    mental_health_sleep_config_t config;
    sleep_system_t sleep_system;
    mental_health_sleep_effects_t effects;
    nimcp_mutex_t* mutex;
    bool callback_registered;
};

/* Forward declarations */
static void mental_health_on_sleep_state_change(sleep_state_t new_state, void* user_data);

/**
 * WHAT: Callback invoked when sleep state changes
 * WHY:  Immediately update mental health parameters for new sleep state
 * HOW:  Called by sleep system via observer pattern
 *
 * BIOLOGICAL BASIS:
 * - Sleep state changes affect psychiatric stability immediately
 * - Sleep deprivation rapidly increases disorder risk
 * - NREM sleep actively restores emotional regulation circuits
 */
static void mental_health_on_sleep_state_change(sleep_state_t new_state, void* user_data)
{
    mental_health_sleep_bridge_t bridge = (mental_health_sleep_bridge_t)user_data;

    /* Guard clause: Validate bridge */
    if (!bridge) {
        NIMCP_LOGGING_ERROR("NULL bridge in sleep state callback");
        return;
    }

    NIMCP_LOGGING_DEBUG("Mental health bridge received sleep state: %d", new_state);

    nimcp_mutex_lock(bridge->mutex);

    bridge->effects.current_state = new_state;

    /* Update psychiatric stability factor */
    if (bridge->config.enable_stability_modulation) {
        float stability_base = mental_health_sleep_stability_for_state(new_state);
        bridge->effects.psychiatric_stability_factor = stability_base * bridge->config.modulation_strength +
                                                       (1.0f - bridge->config.modulation_strength);
    }

    /* Update disorder risk */
    if (bridge->config.enable_risk_modulation) {
        bridge->effects.disorder_risk_factor = mental_health_sleep_disorder_risk_for_state(new_state);
    }

    /* Emotional regulation improves during sleep */
    bridge->effects.emotional_regulation_factor = (new_state == SLEEP_STATE_AWAKE) ? 1.0f :
                                                   (new_state == SLEEP_STATE_DROWSY) ? 0.7f :
                                                   (new_state == SLEEP_STATE_DEEP_NREM) ? 1.2f :
                                                   (new_state == SLEEP_STATE_LIGHT_NREM) ? 1.1f : 0.9f;

    /* Reality testing degrades with sleep deprivation */
    bridge->effects.reality_testing_factor = (new_state == SLEEP_STATE_AWAKE) ? 1.0f :
                                             (new_state == SLEEP_STATE_DROWSY) ? 0.8f : 1.0f;

    /* Update restoration status */
    bridge->effects.restoration_active = (new_state == SLEEP_STATE_DEEP_NREM ||
                                          new_state == SLEEP_STATE_LIGHT_NREM);

    nimcp_mutex_unlock(bridge->mutex);

    NIMCP_LOGGING_DEBUG("Mental health modulated: stability=%.2f, risk=%.2f, restoration=%d",
                        bridge->effects.psychiatric_stability_factor,
                        bridge->effects.disorder_risk_factor,
                        bridge->effects.restoration_active);
}

int mental_health_sleep_default_config(mental_health_sleep_config_t* config) {
    if (!config) return -1;
    config->enable_stability_modulation = true;
    config->enable_risk_modulation = true;
    config->modulation_strength = 1.0f;
    return 0;
}

mental_health_sleep_bridge_t mental_health_sleep_bridge_create(
    const mental_health_sleep_config_t* config,
    sleep_system_t sleep)
{
    if (!sleep) return NULL;

    struct mental_health_sleep_bridge_struct* bridge =
        (struct mental_health_sleep_bridge_struct*)nimcp_malloc(sizeof(struct mental_health_sleep_bridge_struct));
    if (!bridge) return NULL;

    memset(bridge, 0, sizeof(struct mental_health_sleep_bridge_struct));

    if (config) bridge->config = *config;
    else mental_health_sleep_default_config(&bridge->config);

    bridge->sleep_system = sleep;
    bridge->effects.psychiatric_stability_factor = 1.0f;
    bridge->effects.disorder_risk_factor = 1.0f;
    bridge->effects.emotional_regulation_factor = 1.0f;
    bridge->effects.reality_testing_factor = 1.0f;
    bridge->effects.restoration_active = false;

    bridge->mutex = nimcp_platform_mutex_create();
    if (!bridge->mutex) { nimcp_free(bridge); return NULL; }

    /* Register callback for automatic state updates */
    bridge->callback_registered = sleep_register_state_callback(
        sleep,
        mental_health_on_sleep_state_change,
        bridge);

    if (!bridge->callback_registered) {
        NIMCP_LOGGING_WARN("Failed to register sleep state callback - will use polling");
    } else {
        NIMCP_LOGGING_DEBUG("Registered sleep state callback for mental health bridge");
    }

    /* Get initial state immediately */
    sleep_state_t initial_state = sleep_get_current_state(sleep);
    mental_health_on_sleep_state_change(initial_state, bridge);

    NIMCP_LOGGING_INFO("Mental health-sleep bridge created");
    return bridge;
}

void mental_health_sleep_bridge_destroy(mental_health_sleep_bridge_t bridge) {
    if (!bridge) return;

    /* Unregister callback if it was registered */
    if (bridge->callback_registered && bridge->sleep_system) {
        bool unregistered = sleep_unregister_state_callback(
            bridge->sleep_system,
            mental_health_on_sleep_state_change,
            bridge);

        if (unregistered) {
            NIMCP_LOGGING_DEBUG("Unregistered sleep state callback for mental health bridge");
        }
    }

    if (bridge->mutex) nimcp_mutex_destroy(bridge->mutex);
    nimcp_free(bridge);
}

int mental_health_sleep_update(mental_health_sleep_bridge_t bridge) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);

    sleep_state_t state = sleep_get_current_state(bridge->sleep_system);
    float pressure = sleep_get_pressure(bridge->sleep_system);

    bridge->effects.current_state = state;
    bridge->effects.sleep_pressure = pressure;

    if (bridge->config.enable_stability_modulation) {
        float stability_base = mental_health_sleep_stability_for_state(state);
        bridge->effects.psychiatric_stability_factor = stability_base * bridge->config.modulation_strength +
                                                       (1.0f - bridge->config.modulation_strength);
        /* High sleep pressure reduces stability */
        if (state == SLEEP_STATE_AWAKE && pressure > 0.7f) {
            bridge->effects.psychiatric_stability_factor *= (1.0f - (pressure - 0.7f) * 0.7f);
        }
    }

    if (bridge->config.enable_risk_modulation) {
        float risk_base = mental_health_sleep_disorder_risk_for_state(state);
        bridge->effects.disorder_risk_factor = risk_base;
        /* High sleep pressure increases disorder risk */
        if (state == SLEEP_STATE_AWAKE && pressure > 0.7f) {
            bridge->effects.disorder_risk_factor *= (1.0f + (pressure - 0.7f) * 0.5f);
        }
    }

    /* Emotional regulation improves during sleep */
    bridge->effects.emotional_regulation_factor = (state == SLEEP_STATE_AWAKE) ? 1.0f :
                                                   (state == SLEEP_STATE_DROWSY) ? 0.7f :
                                                   (state == SLEEP_STATE_DEEP_NREM) ? 1.2f :
                                                   (state == SLEEP_STATE_LIGHT_NREM) ? 1.1f : 0.9f;

    /* Reality testing degrades with sleep deprivation */
    bridge->effects.reality_testing_factor = (state == SLEEP_STATE_AWAKE) ? 1.0f :
                                             (state == SLEEP_STATE_DROWSY) ? 0.8f : 1.0f;
    if (state == SLEEP_STATE_AWAKE && pressure > 0.8f) {
        bridge->effects.reality_testing_factor *= 0.6f;  /* Severe impairment */
    }

    bridge->effects.restoration_active = (state == SLEEP_STATE_DEEP_NREM ||
                                          state == SLEEP_STATE_LIGHT_NREM);

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int mental_health_sleep_get_effects(const mental_health_sleep_bridge_t bridge, mental_health_sleep_effects_t* effects) {
    if (!bridge || !effects) return -1;
    nimcp_mutex_lock(bridge->mutex);
    *effects = bridge->effects;
    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

float mental_health_sleep_get_stability(const mental_health_sleep_bridge_t bridge) {
    if (!bridge) return 1.0f;
    nimcp_mutex_lock(bridge->mutex);
    float result = bridge->effects.psychiatric_stability_factor;
    nimcp_mutex_unlock(bridge->mutex);
    return result;
}

bool mental_health_sleep_is_restoration_active(const mental_health_sleep_bridge_t bridge) {
    if (!bridge) return false;
    nimcp_mutex_lock(bridge->mutex);
    bool result = bridge->effects.restoration_active;
    nimcp_mutex_unlock(bridge->mutex);
    return result;
}

float mental_health_sleep_stability_for_state(sleep_state_t state) {
    switch (state) {
        case SLEEP_STATE_AWAKE:      return MENTAL_HEALTH_SLEEP_STABILITY_AWAKE;
        case SLEEP_STATE_DROWSY:     return MENTAL_HEALTH_SLEEP_STABILITY_DROWSY;
        case SLEEP_STATE_LIGHT_NREM: return MENTAL_HEALTH_SLEEP_STABILITY_LIGHT_NREM;
        case SLEEP_STATE_DEEP_NREM:  return MENTAL_HEALTH_SLEEP_STABILITY_DEEP_NREM;
        case SLEEP_STATE_REM:        return MENTAL_HEALTH_SLEEP_STABILITY_REM;
        default:                     return MENTAL_HEALTH_SLEEP_STABILITY_AWAKE;
    }
}

float mental_health_sleep_disorder_risk_for_state(sleep_state_t state) {
    switch (state) {
        case SLEEP_STATE_AWAKE:      return MENTAL_HEALTH_SLEEP_DISORDER_RISK_AWAKE;
        case SLEEP_STATE_DROWSY:     return MENTAL_HEALTH_SLEEP_DISORDER_RISK_DROWSY;
        case SLEEP_STATE_LIGHT_NREM:
        case SLEEP_STATE_DEEP_NREM:  return MENTAL_HEALTH_SLEEP_DISORDER_RISK_NREM;
        case SLEEP_STATE_REM:        return MENTAL_HEALTH_SLEEP_DISORDER_RISK_REM;
        default:                     return MENTAL_HEALTH_SLEEP_DISORDER_RISK_AWAKE;
    }
}
