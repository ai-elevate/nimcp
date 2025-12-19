/**
 * @file nimcp_retina_sleep_bridge.c
 * @brief Sleep-Retina Integration Bridge Implementation
 * @version 1.0.0
 * @date 2025-12-17
 */

#include "perception/sleep/nimcp_retina_sleep_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/platform/nimcp_platform_mutex.h"
#include <string.h>

struct retina_sleep_bridge_struct {
    retina_sleep_config_t config;
    sleep_system_t sleep_system;
    retina_sleep_effects_t effects;
    nimcp_mutex_t* mutex;
    bool callback_registered;  /* Track if callback is registered for cleanup */
};

/* Forward declarations */
static void retina_on_sleep_state_change(sleep_state_t new_state, void* user_data);

/**
 * WHAT: Callback invoked when sleep state changes
 * WHY:  Immediately update retinal parameters for new sleep state
 * HOW:  Called by sleep system via observer pattern
 *
 * BIOLOGICAL BASIS:
 * - Pupillary light reflex mediated by pretectal olivary nucleus
 * - Miosis during NREM sleep (parasympathetic dominance)
 * - Eyes close during sleep onset (protective mechanism)
 * - Retinal recovery from phototoxicity during dark rest
 * - REM sleep shows rapid eye movements but eyelids remain closed
 */
static void retina_on_sleep_state_change(sleep_state_t new_state, void* user_data)
{
    retina_sleep_bridge_t bridge = (retina_sleep_bridge_t)user_data;

    if (!bridge) {
        NIMCP_LOGGING_ERROR("NULL bridge in sleep state callback");
        return;
    }

    NIMCP_LOGGING_DEBUG("Retina bridge received sleep state: %d", new_state);

    nimcp_mutex_lock(bridge->mutex);

    bridge->effects.current_state = new_state;

    if (bridge->config.enable_pupil_modulation) {
        float pupil_base = retina_sleep_get_pupil_factor(new_state);
        bridge->effects.pupil_response_factor =
            1.0f + (pupil_base - 1.0f) * bridge->config.modulation_strength;
    }

    if (bridge->config.enable_sensitivity_modulation) {
        float sensitivity_base = retina_sleep_get_sensitivity_factor(new_state);
        bridge->effects.light_sensitivity_factor =
            1.0f + (sensitivity_base - 1.0f) * bridge->config.modulation_strength;
    }

    if (bridge->config.enable_adaptation_modulation) {
        float adaptation_base = retina_sleep_get_adaptation_factor(new_state);
        bridge->effects.adaptation_rate_factor =
            1.0f + (adaptation_base - 1.0f) * bridge->config.modulation_strength;
    }

    bridge->effects.retinal_processing_enabled = (new_state == SLEEP_STATE_AWAKE) ||
                                                   (new_state == SLEEP_STATE_DROWSY) ||
                                                   bridge->effects.light_sensitivity_factor > 0.3f;

    nimcp_mutex_unlock(bridge->mutex);

    NIMCP_LOGGING_DEBUG("Retina modulated: pupil=%.2f, sensitivity=%.2f, adaptation=%.2f",
                        bridge->effects.pupil_response_factor,
                        bridge->effects.light_sensitivity_factor,
                        bridge->effects.adaptation_rate_factor);
}

int retina_sleep_default_config(retina_sleep_config_t* config) {
    if (!config) return -1;
    config->enable_pupil_modulation = true;
    config->enable_sensitivity_modulation = true;
    config->enable_adaptation_modulation = true;
    config->modulation_strength = 1.0f;
    return 0;
}

retina_sleep_bridge_t retina_sleep_bridge_create(
    const retina_sleep_config_t* config,
    sleep_system_t sleep_system)
{
    if (!sleep_system) {
        NIMCP_LOGGING_ERROR("retina_sleep_bridge_create: NULL sleep_system");
        return NULL;
    }

    struct retina_sleep_bridge_struct* bridge =
        (struct retina_sleep_bridge_struct*)nimcp_malloc(sizeof(struct retina_sleep_bridge_struct));
    if (!bridge) return NULL;

    memset(bridge, 0, sizeof(struct retina_sleep_bridge_struct));

    if (config) {
        bridge->config = *config;
    } else {
        retina_sleep_default_config(&bridge->config);
    }

    bridge->sleep_system = sleep_system;
    bridge->effects.pupil_response_factor = 1.0f;
    bridge->effects.light_sensitivity_factor = 1.0f;
    bridge->effects.adaptation_rate_factor = 1.0f;
    bridge->effects.retinal_processing_enabled = true;

    bridge->mutex = nimcp_platform_mutex_create();
    if (!bridge->mutex) {
        nimcp_free(bridge);
        return NULL;
    }

    /* Register callback for automatic state updates */
    bridge->callback_registered = sleep_register_state_callback(
        sleep_system,
        retina_on_sleep_state_change,
        bridge);

    if (!bridge->callback_registered) {
        NIMCP_LOGGING_WARN("Failed to register sleep state callback - will use polling");
    } else {
        NIMCP_LOGGING_DEBUG("Registered sleep state callback for retina bridge");
    }

    /* Get initial state immediately */
    sleep_state_t initial_state = sleep_get_current_state(sleep_system);
    retina_on_sleep_state_change(initial_state, bridge);

    NIMCP_LOGGING_INFO("Retina-sleep bridge created");
    return bridge;
}

void retina_sleep_bridge_destroy(retina_sleep_bridge_t bridge) {
    if (!bridge) return;

    /* Unregister callback if it was registered */
    if (bridge->callback_registered && bridge->sleep_system) {
        bool unregistered = sleep_unregister_state_callback(
            bridge->sleep_system,
            retina_on_sleep_state_change,
            bridge);

        if (unregistered) {
            NIMCP_LOGGING_DEBUG("Unregistered sleep state callback for retina bridge");
        }
    }

    if (bridge->mutex) nimcp_mutex_destroy(bridge->mutex);
    nimcp_free(bridge);
}

int retina_sleep_update(retina_sleep_bridge_t bridge) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);

    sleep_state_t state = sleep_get_current_state(bridge->sleep_system);
    float pressure = sleep_get_pressure(bridge->sleep_system);

    bridge->effects.current_state = state;
    bridge->effects.sleep_pressure = pressure;

    if (bridge->config.enable_pupil_modulation) {
        float pupil_base = retina_sleep_get_pupil_factor(state);
        bridge->effects.pupil_response_factor =
            1.0f + (pupil_base - 1.0f) * bridge->config.modulation_strength;
    }

    if (bridge->config.enable_sensitivity_modulation) {
        float sensitivity_base = retina_sleep_get_sensitivity_factor(state);
        bridge->effects.light_sensitivity_factor =
            1.0f + (sensitivity_base - 1.0f) * bridge->config.modulation_strength;
    }

    if (bridge->config.enable_adaptation_modulation) {
        float adaptation_base = retina_sleep_get_adaptation_factor(state);
        bridge->effects.adaptation_rate_factor =
            1.0f + (adaptation_base - 1.0f) * bridge->config.modulation_strength;
    }

    bridge->effects.retinal_processing_enabled = (state == SLEEP_STATE_AWAKE) ||
                                                   (state == SLEEP_STATE_DROWSY) ||
                                                   bridge->effects.light_sensitivity_factor > 0.3f;

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int retina_sleep_get_effects(const retina_sleep_bridge_t bridge, retina_sleep_effects_t* effects) {
    if (!bridge || !effects) return -1;
    nimcp_mutex_lock(bridge->mutex);
    *effects = bridge->effects;
    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

float retina_sleep_get_pupil_response(const retina_sleep_bridge_t bridge, float base_response) {
    if (!bridge) return base_response;
    nimcp_mutex_lock(bridge->mutex);
    float result = base_response * bridge->effects.pupil_response_factor;
    nimcp_mutex_unlock(bridge->mutex);
    return result;
}

float retina_sleep_get_light_sensitivity(const retina_sleep_bridge_t bridge, float base_sensitivity) {
    if (!bridge) return base_sensitivity;
    nimcp_mutex_lock(bridge->mutex);
    float result = base_sensitivity * bridge->effects.light_sensitivity_factor;
    nimcp_mutex_unlock(bridge->mutex);
    return result;
}

float retina_sleep_get_adaptation_rate(const retina_sleep_bridge_t bridge, float base_rate) {
    if (!bridge) return base_rate;
    nimcp_mutex_lock(bridge->mutex);
    float result = base_rate * bridge->effects.adaptation_rate_factor;
    nimcp_mutex_unlock(bridge->mutex);
    return result;
}

float retina_sleep_get_pupil_factor(sleep_state_t state) {
    switch (state) {
        case SLEEP_STATE_AWAKE:      return RETINA_SLEEP_PUPIL_AWAKE;
        case SLEEP_STATE_DROWSY:     return RETINA_SLEEP_PUPIL_DROWSY;
        case SLEEP_STATE_LIGHT_NREM: return RETINA_SLEEP_PUPIL_LIGHT_NREM;
        case SLEEP_STATE_DEEP_NREM:  return RETINA_SLEEP_PUPIL_DEEP_NREM;
        case SLEEP_STATE_REM:        return RETINA_SLEEP_PUPIL_REM;
        default:                     return RETINA_SLEEP_PUPIL_AWAKE;
    }
}

float retina_sleep_get_sensitivity_factor(sleep_state_t state) {
    switch (state) {
        case SLEEP_STATE_AWAKE:      return RETINA_SLEEP_SENSITIVITY_AWAKE;
        case SLEEP_STATE_DROWSY:     return RETINA_SLEEP_SENSITIVITY_DROWSY;
        case SLEEP_STATE_LIGHT_NREM: return RETINA_SLEEP_SENSITIVITY_LIGHT_NREM;
        case SLEEP_STATE_DEEP_NREM:  return RETINA_SLEEP_SENSITIVITY_DEEP_NREM;
        case SLEEP_STATE_REM:        return RETINA_SLEEP_SENSITIVITY_REM;
        default:                     return RETINA_SLEEP_SENSITIVITY_AWAKE;
    }
}

float retina_sleep_get_adaptation_factor(sleep_state_t state) {
    switch (state) {
        case SLEEP_STATE_AWAKE:      return RETINA_SLEEP_ADAPTATION_AWAKE;
        case SLEEP_STATE_DROWSY:     return RETINA_SLEEP_ADAPTATION_DROWSY;
        case SLEEP_STATE_LIGHT_NREM: return RETINA_SLEEP_ADAPTATION_LIGHT_NREM;
        case SLEEP_STATE_DEEP_NREM:  return RETINA_SLEEP_ADAPTATION_DEEP_NREM;
        case SLEEP_STATE_REM:        return RETINA_SLEEP_ADAPTATION_REM;
        default:                     return RETINA_SLEEP_ADAPTATION_AWAKE;
    }
}
