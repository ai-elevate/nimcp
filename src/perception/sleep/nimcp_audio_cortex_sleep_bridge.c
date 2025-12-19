/**
 * @file nimcp_audio_cortex_sleep_bridge.c
 * @brief Sleep-Audio Cortex Integration Bridge Implementation
 * @version 1.0.0
 * @date 2025-12-17
 */

#include "perception/sleep/nimcp_audio_cortex_sleep_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/platform/nimcp_platform_mutex.h"
#include <string.h>

struct audio_sleep_bridge_struct {
    audio_sleep_config_t config;
    sleep_system_t sleep_system;
    audio_sleep_effects_t effects;
    nimcp_mutex_t* mutex;
    bool callback_registered;  /* Track if callback is registered for cleanup */
};

/* Forward declarations */
static void audio_on_sleep_state_change(sleep_state_t new_state, void* user_data);

/**
 * WHAT: Callback invoked when sleep state changes
 * WHY:  Immediately update audio cortex parameters for new sleep state
 * HOW:  Called by sleep system via observer pattern
 *
 * BIOLOGICAL BASIS:
 * - Auditory threshold increases during sleep (protective deafness)
 * - Frequency discrimination degrades with sleep pressure
 * - Temporal processing slows during NREM stages
 * - K-complexes in NREM can be triggered by salient sounds (baby crying)
 */
static void audio_on_sleep_state_change(sleep_state_t new_state, void* user_data)
{
    audio_sleep_bridge_t bridge = (audio_sleep_bridge_t)user_data;

    if (!bridge) {
        NIMCP_LOGGING_ERROR("NULL bridge in sleep state callback");
        return;
    }

    NIMCP_LOGGING_DEBUG("Audio cortex bridge received sleep state: %d", new_state);

    nimcp_mutex_lock(bridge->mutex);

    bridge->effects.current_state = new_state;

    if (bridge->config.enable_threshold_modulation) {
        float threshold_base = audio_sleep_get_threshold_factor(new_state);
        bridge->effects.threshold_factor =
            1.0f + (threshold_base - 1.0f) * bridge->config.modulation_strength;
    }

    if (bridge->config.enable_frequency_modulation) {
        float frequency_base = audio_sleep_get_frequency_factor(new_state);
        bridge->effects.frequency_selectivity_factor =
            1.0f + (frequency_base - 1.0f) * bridge->config.modulation_strength;
    }

    if (bridge->config.enable_speed_modulation) {
        float speed_base = audio_sleep_get_speed_factor(new_state);
        bridge->effects.processing_speed_factor =
            1.0f + (speed_base - 1.0f) * bridge->config.modulation_strength;
    }

    bridge->effects.audio_processing_enabled = (new_state != SLEEP_STATE_DEEP_NREM) ||
                                                 bridge->effects.threshold_factor > 0.3f;

    nimcp_mutex_unlock(bridge->mutex);

    NIMCP_LOGGING_DEBUG("Audio modulated: threshold=%.2f, frequency=%.2f, speed=%.2f",
                        bridge->effects.threshold_factor,
                        bridge->effects.frequency_selectivity_factor,
                        bridge->effects.processing_speed_factor);
}

int audio_sleep_default_config(audio_sleep_config_t* config) {
    if (!config) return -1;
    config->enable_threshold_modulation = true;
    config->enable_frequency_modulation = true;
    config->enable_speed_modulation = true;
    config->modulation_strength = 1.0f;
    return 0;
}

audio_sleep_bridge_t audio_sleep_bridge_create(
    const audio_sleep_config_t* config,
    sleep_system_t sleep_system)
{
    if (!sleep_system) {
        NIMCP_LOGGING_ERROR("audio_sleep_bridge_create: NULL sleep_system");
        return NULL;
    }

    struct audio_sleep_bridge_struct* bridge =
        (struct audio_sleep_bridge_struct*)nimcp_malloc(sizeof(struct audio_sleep_bridge_struct));
    if (!bridge) return NULL;

    memset(bridge, 0, sizeof(struct audio_sleep_bridge_struct));

    if (config) {
        bridge->config = *config;
    } else {
        audio_sleep_default_config(&bridge->config);
    }

    bridge->sleep_system = sleep_system;
    bridge->effects.threshold_factor = 1.0f;
    bridge->effects.frequency_selectivity_factor = 1.0f;
    bridge->effects.processing_speed_factor = 1.0f;
    bridge->effects.audio_processing_enabled = true;

    bridge->mutex = nimcp_platform_mutex_create();
    if (!bridge->mutex) {
        nimcp_free(bridge);
        return NULL;
    }

    /* Register callback for automatic state updates */
    bridge->callback_registered = sleep_register_state_callback(
        sleep_system,
        audio_on_sleep_state_change,
        bridge);

    if (!bridge->callback_registered) {
        NIMCP_LOGGING_WARN("Failed to register sleep state callback - will use polling");
    } else {
        NIMCP_LOGGING_DEBUG("Registered sleep state callback for audio cortex bridge");
    }

    /* Get initial state immediately */
    sleep_state_t initial_state = sleep_get_current_state(sleep_system);
    audio_on_sleep_state_change(initial_state, bridge);

    NIMCP_LOGGING_INFO("Audio cortex-sleep bridge created");
    return bridge;
}

void audio_sleep_bridge_destroy(audio_sleep_bridge_t bridge) {
    if (!bridge) return;

    /* Unregister callback if it was registered */
    if (bridge->callback_registered && bridge->sleep_system) {
        bool unregistered = sleep_unregister_state_callback(
            bridge->sleep_system,
            audio_on_sleep_state_change,
            bridge);

        if (unregistered) {
            NIMCP_LOGGING_DEBUG("Unregistered sleep state callback for audio cortex bridge");
        }
    }

    if (bridge->mutex) nimcp_mutex_destroy(bridge->mutex);
    nimcp_free(bridge);
}

int audio_sleep_update(audio_sleep_bridge_t bridge) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);

    sleep_state_t state = sleep_get_current_state(bridge->sleep_system);
    float pressure = sleep_get_pressure(bridge->sleep_system);

    bridge->effects.current_state = state;
    bridge->effects.sleep_pressure = pressure;

    if (bridge->config.enable_threshold_modulation) {
        float threshold_base = audio_sleep_get_threshold_factor(state);
        bridge->effects.threshold_factor =
            1.0f + (threshold_base - 1.0f) * bridge->config.modulation_strength;
    }

    if (bridge->config.enable_frequency_modulation) {
        float frequency_base = audio_sleep_get_frequency_factor(state);
        bridge->effects.frequency_selectivity_factor =
            1.0f + (frequency_base - 1.0f) * bridge->config.modulation_strength;
    }

    if (bridge->config.enable_speed_modulation) {
        float speed_base = audio_sleep_get_speed_factor(state);
        bridge->effects.processing_speed_factor =
            1.0f + (speed_base - 1.0f) * bridge->config.modulation_strength;
    }

    bridge->effects.audio_processing_enabled = (state != SLEEP_STATE_DEEP_NREM) ||
                                                 bridge->effects.threshold_factor > 0.3f;

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int audio_sleep_get_effects(const audio_sleep_bridge_t bridge, audio_sleep_effects_t* effects) {
    if (!bridge || !effects) return -1;
    nimcp_mutex_lock(bridge->mutex);
    *effects = bridge->effects;
    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

float audio_sleep_get_threshold(const audio_sleep_bridge_t bridge, float base_threshold) {
    if (!bridge) return base_threshold;
    nimcp_mutex_lock(bridge->mutex);
    float result = base_threshold * bridge->effects.threshold_factor;
    nimcp_mutex_unlock(bridge->mutex);
    return result;
}

float audio_sleep_get_frequency_selectivity(const audio_sleep_bridge_t bridge, float base_selectivity) {
    if (!bridge) return base_selectivity;
    nimcp_mutex_lock(bridge->mutex);
    float result = base_selectivity * bridge->effects.frequency_selectivity_factor;
    nimcp_mutex_unlock(bridge->mutex);
    return result;
}

float audio_sleep_get_processing_speed(const audio_sleep_bridge_t bridge, float base_speed) {
    if (!bridge) return base_speed;
    nimcp_mutex_lock(bridge->mutex);
    float result = base_speed * bridge->effects.processing_speed_factor;
    nimcp_mutex_unlock(bridge->mutex);
    return result;
}

float audio_sleep_get_threshold_factor(sleep_state_t state) {
    switch (state) {
        case SLEEP_STATE_AWAKE:      return AUDIO_SLEEP_THRESHOLD_AWAKE;
        case SLEEP_STATE_DROWSY:     return AUDIO_SLEEP_THRESHOLD_DROWSY;
        case SLEEP_STATE_LIGHT_NREM: return AUDIO_SLEEP_THRESHOLD_LIGHT_NREM;
        case SLEEP_STATE_DEEP_NREM:  return AUDIO_SLEEP_THRESHOLD_DEEP_NREM;
        case SLEEP_STATE_REM:        return AUDIO_SLEEP_THRESHOLD_REM;
        default:                     return AUDIO_SLEEP_THRESHOLD_AWAKE;
    }
}

float audio_sleep_get_frequency_factor(sleep_state_t state) {
    switch (state) {
        case SLEEP_STATE_AWAKE:      return AUDIO_SLEEP_FREQUENCY_AWAKE;
        case SLEEP_STATE_DROWSY:     return AUDIO_SLEEP_FREQUENCY_DROWSY;
        case SLEEP_STATE_LIGHT_NREM: return AUDIO_SLEEP_FREQUENCY_LIGHT_NREM;
        case SLEEP_STATE_DEEP_NREM:  return AUDIO_SLEEP_FREQUENCY_DEEP_NREM;
        case SLEEP_STATE_REM:        return AUDIO_SLEEP_FREQUENCY_REM;
        default:                     return AUDIO_SLEEP_FREQUENCY_AWAKE;
    }
}

float audio_sleep_get_speed_factor(sleep_state_t state) {
    switch (state) {
        case SLEEP_STATE_AWAKE:      return AUDIO_SLEEP_SPEED_AWAKE;
        case SLEEP_STATE_DROWSY:     return AUDIO_SLEEP_SPEED_DROWSY;
        case SLEEP_STATE_LIGHT_NREM: return AUDIO_SLEEP_SPEED_LIGHT_NREM;
        case SLEEP_STATE_DEEP_NREM:  return AUDIO_SLEEP_SPEED_DEEP_NREM;
        case SLEEP_STATE_REM:        return AUDIO_SLEEP_SPEED_REM;
        default:                     return AUDIO_SLEEP_SPEED_AWAKE;
    }
}
