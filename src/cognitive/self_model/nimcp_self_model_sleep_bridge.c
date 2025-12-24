/**
 * @file nimcp_self_model_sleep_bridge.c
 * @brief Sleep-Self Model Integration Bridge Implementation
 * @version 1.0.0
 * @date 2025-12-17
 */

#include "cognitive/self_model/nimcp_self_model_sleep_bridge.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/platform/nimcp_platform_mutex.h"
#include <string.h>

struct self_model_sleep_bridge_struct {
    bridge_base_t base;               /**< MUST be first: base bridge infrastructure */

    self_model_sleep_config_t config;
    sleep_system_t sleep_system;
    self_model_sleep_effects_t effects;
    bool callback_registered;
};

/* Forward declarations */
static void self_model_on_sleep_state_change(sleep_state_t new_state, void* user_data);

/**
 * WHAT: Callback invoked when sleep state changes
 * WHY:  Immediately update self-model parameters for new sleep state
 * HOW:  Called by sleep system via observer pattern
 *
 * BIOLOGICAL BASIS:
 * - Sleep state changes affect self-awareness immediately
 * - Consciousness level depends on arousal state
 * - Metacognition requires wakefulness
 */
static void self_model_on_sleep_state_change(sleep_state_t new_state, void* user_data)
{
    self_model_sleep_bridge_t bridge = (self_model_sleep_bridge_t)user_data;

    /* Guard clause: Validate bridge */
    if (!bridge) {
        NIMCP_LOGGING_ERROR("NULL bridge in sleep state callback");
        return;
    }

    NIMCP_LOGGING_DEBUG("Self-model bridge received sleep state: %d", new_state);

    nimcp_mutex_lock(bridge->base.mutex);

    bridge->effects.current_state = new_state;

    /* Update self-awareness */
    if (bridge->config.enable_awareness_modulation) {
        float awareness_base = self_model_sleep_awareness_for_state(new_state);
        bridge->effects.self_awareness_factor = awareness_base * bridge->config.modulation_strength +
                                                (1.0f - bridge->config.modulation_strength);
    }

    /* Update self-reflection */
    if (bridge->config.enable_reflection_modulation) {
        bridge->effects.self_reflection_factor = self_model_sleep_reflection_for_state(new_state);
    }

    /* Metacognition requires full consciousness */
    bridge->effects.metacognition_factor = (new_state == SLEEP_STATE_AWAKE) ? 1.0f :
                                           (new_state == SLEEP_STATE_DROWSY) ? 0.4f : 0.0f;

    /* Self-monitoring accuracy */
    bridge->effects.self_monitoring_factor = (new_state == SLEEP_STATE_AWAKE) ? 1.0f :
                                             (new_state == SLEEP_STATE_DROWSY) ? 0.5f :
                                             (new_state == SLEEP_STATE_REM) ? 0.2f : 0.0f;

    /* Update offline status */
    bridge->effects.self_awareness_offline = (new_state == SLEEP_STATE_DEEP_NREM ||
                                              new_state == SLEEP_STATE_LIGHT_NREM);

    nimcp_mutex_unlock(bridge->base.mutex);

    NIMCP_LOGGING_DEBUG("Self-model modulated: awareness=%.2f, reflection=%.2f, offline=%d",
                        bridge->effects.self_awareness_factor,
                        bridge->effects.self_reflection_factor,
                        bridge->effects.self_awareness_offline);
}

int self_model_sleep_default_config(self_model_sleep_config_t* config) {
    if (!config) return -1;
    config->enable_awareness_modulation = true;
    config->enable_reflection_modulation = true;
    config->modulation_strength = 1.0f;
    return 0;
}

self_model_sleep_bridge_t self_model_sleep_bridge_create(
    const self_model_sleep_config_t* config,
    sleep_system_t sleep)
{
    if (!sleep) return NULL;

    struct self_model_sleep_bridge_struct* bridge =
        (struct self_model_sleep_bridge_struct*)nimcp_malloc(sizeof(struct self_model_sleep_bridge_struct));
    if (!bridge) return NULL;

    memset(bridge, 0, sizeof(struct self_model_sleep_bridge_struct));

    if (config) bridge->config = *config;
    else self_model_sleep_default_config(&bridge->config);

    bridge->sleep_system = sleep;
    bridge->effects.self_awareness_factor = 1.0f;
    bridge->effects.metacognition_factor = 1.0f;
    bridge->effects.self_reflection_factor = 1.0f;
    bridge->effects.self_monitoring_factor = 1.0f;
    bridge->effects.self_awareness_offline = false;

    bridge->base.mutex = nimcp_platform_mutex_create();
    if (!bridge->base.mutex) { nimcp_free(bridge); return NULL; }

    /* Register callback for automatic state updates */
    bridge->callback_registered = sleep_register_state_callback(
        sleep,
        self_model_on_sleep_state_change,
        bridge);

    if (!bridge->callback_registered) {
        NIMCP_LOGGING_WARN("Failed to register sleep state callback - will use polling");
    } else {
        NIMCP_LOGGING_DEBUG("Registered sleep state callback for self-model bridge");
    }

    /* Get initial state immediately */
    sleep_state_t initial_state = sleep_get_current_state(sleep);
    self_model_on_sleep_state_change(initial_state, bridge);

    NIMCP_LOGGING_INFO("Self-model-sleep bridge created");
    return bridge;
}

void self_model_sleep_bridge_destroy(self_model_sleep_bridge_t bridge) {
    if (!bridge) return;

    /* Unregister callback if it was registered */
    if (bridge->callback_registered && bridge->sleep_system) {
        bool unregistered = sleep_unregister_state_callback(
            bridge->sleep_system,
            self_model_on_sleep_state_change,
            bridge);

        if (unregistered) {
            NIMCP_LOGGING_DEBUG("Unregistered sleep state callback for self-model bridge");
        }
    }

    if (bridge->base.mutex) nimcp_mutex_destroy(bridge->base.mutex);
    nimcp_free(bridge);
}

int self_model_sleep_update(self_model_sleep_bridge_t bridge) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->base.mutex);

    sleep_state_t state = sleep_get_current_state(bridge->sleep_system);
    float pressure = sleep_get_pressure(bridge->sleep_system);

    bridge->effects.current_state = state;
    bridge->effects.sleep_pressure = pressure;

    if (bridge->config.enable_awareness_modulation) {
        float awareness_base = self_model_sleep_awareness_for_state(state);
        bridge->effects.self_awareness_factor = awareness_base * bridge->config.modulation_strength +
                                                (1.0f - bridge->config.modulation_strength);
        /* Sleep pressure impairs self-awareness */
        if (state == SLEEP_STATE_AWAKE && pressure > 0.7f) {
            bridge->effects.self_awareness_factor *= (1.0f - (pressure - 0.7f) * 0.5f);
        }
    }

    if (bridge->config.enable_reflection_modulation) {
        bridge->effects.self_reflection_factor = self_model_sleep_reflection_for_state(state);
        /* Sleep pressure reduces reflection */
        if (state == SLEEP_STATE_AWAKE && pressure > 0.7f) {
            bridge->effects.self_reflection_factor *= (1.0f - (pressure - 0.7f) * 0.6f);
        }
    }

    /* Metacognition requires full consciousness */
    bridge->effects.metacognition_factor = (state == SLEEP_STATE_AWAKE) ? 1.0f :
                                           (state == SLEEP_STATE_DROWSY) ? 0.4f : 0.0f;

    /* Self-monitoring accuracy */
    bridge->effects.self_monitoring_factor = (state == SLEEP_STATE_AWAKE) ? 1.0f :
                                             (state == SLEEP_STATE_DROWSY) ? 0.5f :
                                             (state == SLEEP_STATE_REM) ? 0.2f : 0.0f;

    bridge->effects.self_awareness_offline = (state == SLEEP_STATE_DEEP_NREM ||
                                              state == SLEEP_STATE_LIGHT_NREM);

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int self_model_sleep_get_effects(const self_model_sleep_bridge_t bridge, self_model_sleep_effects_t* effects) {
    if (!bridge || !effects) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    *effects = bridge->effects;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

float self_model_sleep_get_awareness(const self_model_sleep_bridge_t bridge) {
    if (!bridge) return 1.0f;
    nimcp_mutex_lock(bridge->base.mutex);
    float result = bridge->effects.self_awareness_factor;
    nimcp_mutex_unlock(bridge->base.mutex);
    return result;
}

bool self_model_sleep_is_offline(const self_model_sleep_bridge_t bridge) {
    if (!bridge) return false;
    nimcp_mutex_lock(bridge->base.mutex);
    bool result = bridge->effects.self_awareness_offline;
    nimcp_mutex_unlock(bridge->base.mutex);
    return result;
}

float self_model_sleep_awareness_for_state(sleep_state_t state) {
    switch (state) {
        case SLEEP_STATE_AWAKE:      return SELF_MODEL_SLEEP_AWARENESS_AWAKE;
        case SLEEP_STATE_DROWSY:     return SELF_MODEL_SLEEP_AWARENESS_DROWSY;
        case SLEEP_STATE_LIGHT_NREM: return SELF_MODEL_SLEEP_AWARENESS_LIGHT_NREM;
        case SLEEP_STATE_DEEP_NREM:  return SELF_MODEL_SLEEP_AWARENESS_DEEP_NREM;
        case SLEEP_STATE_REM:        return SELF_MODEL_SLEEP_AWARENESS_REM;
        default:                     return SELF_MODEL_SLEEP_AWARENESS_AWAKE;
    }
}

float self_model_sleep_reflection_for_state(sleep_state_t state) {
    switch (state) {
        case SLEEP_STATE_AWAKE:      return SELF_MODEL_SLEEP_REFLECTION_AWAKE;
        case SLEEP_STATE_DROWSY:     return SELF_MODEL_SLEEP_REFLECTION_DROWSY;
        case SLEEP_STATE_LIGHT_NREM:
        case SLEEP_STATE_DEEP_NREM:  return SELF_MODEL_SLEEP_REFLECTION_NREM;
        case SLEEP_STATE_REM:        return SELF_MODEL_SLEEP_REFLECTION_REM;
        default:                     return SELF_MODEL_SLEEP_REFLECTION_AWAKE;
    }
}
