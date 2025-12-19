/**
 * @file nimcp_curiosity_sleep_bridge.c
 * @brief Sleep-Curiosity Integration Bridge Implementation
 * @version 1.0.0
 * @date 2025-12-17
 */

#include "cognitive/curiosity/nimcp_curiosity_sleep_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/platform/nimcp_platform_mutex.h"
#include <string.h>

struct curiosity_sleep_bridge_struct {
    curiosity_sleep_config_t config;
    sleep_system_t sleep_system;
    curiosity_sleep_effects_t effects;
    nimcp_mutex_t* mutex;
    bool callback_registered;
};

/* Forward declarations */
static void curiosity_on_sleep_state_change(sleep_state_t new_state, void* user_data);

/**
 * WHAT: Callback invoked when sleep state changes
 * WHY:  Immediately update curiosity parameters for new sleep state
 * HOW:  Called by sleep system via observer pattern
 *
 * BIOLOGICAL BASIS:
 * - Sleep state changes affect exploration drive immediately
 * - Alertness and novelty seeking depend on arousal level
 * - REM allows internal creative exploration while external exploration is suppressed
 */
static void curiosity_on_sleep_state_change(sleep_state_t new_state, void* user_data)
{
    curiosity_sleep_bridge_t bridge = (curiosity_sleep_bridge_t)user_data;

    /* Guard clause: Validate bridge */
    if (!bridge) {
        NIMCP_LOGGING_ERROR("NULL bridge in sleep state callback");
        return;
    }

    NIMCP_LOGGING_DEBUG("Curiosity bridge received sleep state: %d", new_state);

    nimcp_mutex_lock(bridge->mutex);

    bridge->effects.current_state = new_state;

    /* Update curiosity drive factor */
    if (bridge->config.enable_drive_modulation) {
        float drive_base = curiosity_sleep_drive_for_state(new_state);
        bridge->effects.curiosity_drive_factor = drive_base * bridge->config.modulation_strength +
                                                 (1.0f - bridge->config.modulation_strength);
    }

    /* Update exploration threshold */
    if (bridge->config.enable_threshold_modulation) {
        bridge->effects.exploration_threshold_factor = curiosity_sleep_threshold_for_state(new_state);
    }

    /* Learning potential varies by state */
    bridge->effects.learning_potential_factor = (new_state == SLEEP_STATE_AWAKE) ? 1.0f :
                                                (new_state == SLEEP_STATE_DROWSY) ? 0.7f :
                                                (new_state == SLEEP_STATE_REM) ? 0.5f : 0.0f;

    /* Update offline status */
    bridge->effects.exploration_offline = (new_state == SLEEP_STATE_DEEP_NREM ||
                                           new_state == SLEEP_STATE_LIGHT_NREM);

    nimcp_mutex_unlock(bridge->mutex);

    NIMCP_LOGGING_DEBUG("Curiosity modulated: drive=%.2f, threshold=%.2f, offline=%d",
                        bridge->effects.curiosity_drive_factor,
                        bridge->effects.exploration_threshold_factor,
                        bridge->effects.exploration_offline);
}

int curiosity_sleep_default_config(curiosity_sleep_config_t* config) {
    if (!config) return -1;
    config->enable_drive_modulation = true;
    config->enable_threshold_modulation = true;
    config->modulation_strength = 1.0f;
    return 0;
}

curiosity_sleep_bridge_t curiosity_sleep_bridge_create(
    const curiosity_sleep_config_t* config,
    sleep_system_t sleep)
{
    if (!sleep) return NULL;

    struct curiosity_sleep_bridge_struct* bridge =
        (struct curiosity_sleep_bridge_struct*)nimcp_malloc(sizeof(struct curiosity_sleep_bridge_struct));
    if (!bridge) return NULL;

    memset(bridge, 0, sizeof(struct curiosity_sleep_bridge_struct));

    if (config) bridge->config = *config;
    else curiosity_sleep_default_config(&bridge->config);

    bridge->sleep_system = sleep;
    bridge->effects.curiosity_drive_factor = 1.0f;
    bridge->effects.exploration_threshold_factor = 0.3f;
    bridge->effects.learning_potential_factor = 1.0f;
    bridge->effects.exploration_offline = false;

    bridge->mutex = nimcp_platform_mutex_create();
    if (!bridge->mutex) { nimcp_free(bridge); return NULL; }

    /* Register callback for automatic state updates */
    bridge->callback_registered = sleep_register_state_callback(
        sleep,
        curiosity_on_sleep_state_change,
        bridge);

    if (!bridge->callback_registered) {
        NIMCP_LOGGING_WARN("Failed to register sleep state callback - will use polling");
    } else {
        NIMCP_LOGGING_DEBUG("Registered sleep state callback for curiosity bridge");
    }

    /* Get initial state immediately */
    sleep_state_t initial_state = sleep_get_current_state(sleep);
    curiosity_on_sleep_state_change(initial_state, bridge);

    NIMCP_LOGGING_INFO("Curiosity-sleep bridge created");
    return bridge;
}

void curiosity_sleep_bridge_destroy(curiosity_sleep_bridge_t bridge) {
    if (!bridge) return;

    /* Unregister callback if it was registered */
    if (bridge->callback_registered && bridge->sleep_system) {
        bool unregistered = sleep_unregister_state_callback(
            bridge->sleep_system,
            curiosity_on_sleep_state_change,
            bridge);

        if (unregistered) {
            NIMCP_LOGGING_DEBUG("Unregistered sleep state callback for curiosity bridge");
        }
    }

    if (bridge->mutex) nimcp_mutex_destroy(bridge->mutex);
    nimcp_free(bridge);
}

int curiosity_sleep_update(curiosity_sleep_bridge_t bridge) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);

    sleep_state_t state = sleep_get_current_state(bridge->sleep_system);
    float pressure = sleep_get_pressure(bridge->sleep_system);

    bridge->effects.current_state = state;
    bridge->effects.sleep_pressure = pressure;

    if (bridge->config.enable_drive_modulation) {
        float drive_base = curiosity_sleep_drive_for_state(state);
        bridge->effects.curiosity_drive_factor = drive_base * bridge->config.modulation_strength +
                                                 (1.0f - bridge->config.modulation_strength);
        /* Sleep pressure further reduces drive */
        if (state == SLEEP_STATE_AWAKE && pressure > 0.7f) {
            bridge->effects.curiosity_drive_factor *= (1.0f - (pressure - 0.7f) * 0.5f);
        }
    }

    if (bridge->config.enable_threshold_modulation) {
        bridge->effects.exploration_threshold_factor = curiosity_sleep_threshold_for_state(state);
    }

    /* Learning potential varies by state */
    bridge->effects.learning_potential_factor = (state == SLEEP_STATE_AWAKE) ? 1.0f :
                                                (state == SLEEP_STATE_DROWSY) ? 0.7f :
                                                (state == SLEEP_STATE_REM) ? 0.5f : 0.0f;

    bridge->effects.exploration_offline = (state == SLEEP_STATE_DEEP_NREM ||
                                           state == SLEEP_STATE_LIGHT_NREM);

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int curiosity_sleep_get_effects(const curiosity_sleep_bridge_t bridge, curiosity_sleep_effects_t* effects) {
    if (!bridge || !effects) return -1;
    nimcp_mutex_lock(bridge->mutex);
    *effects = bridge->effects;
    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

float curiosity_sleep_get_drive(const curiosity_sleep_bridge_t bridge) {
    if (!bridge) return 1.0f;
    nimcp_mutex_lock(bridge->mutex);
    float result = bridge->effects.curiosity_drive_factor;
    nimcp_mutex_unlock(bridge->mutex);
    return result;
}

bool curiosity_sleep_is_offline(const curiosity_sleep_bridge_t bridge) {
    if (!bridge) return false;
    nimcp_mutex_lock(bridge->mutex);
    bool result = bridge->effects.exploration_offline;
    nimcp_mutex_unlock(bridge->mutex);
    return result;
}

float curiosity_sleep_drive_for_state(sleep_state_t state) {
    switch (state) {
        case SLEEP_STATE_AWAKE:      return CURIOSITY_SLEEP_DRIVE_AWAKE;
        case SLEEP_STATE_DROWSY:     return CURIOSITY_SLEEP_DRIVE_DROWSY;
        case SLEEP_STATE_LIGHT_NREM: return CURIOSITY_SLEEP_DRIVE_LIGHT_NREM;
        case SLEEP_STATE_DEEP_NREM:  return CURIOSITY_SLEEP_DRIVE_DEEP_NREM;
        case SLEEP_STATE_REM:        return CURIOSITY_SLEEP_DRIVE_REM;
        default:                     return CURIOSITY_SLEEP_DRIVE_AWAKE;
    }
}

float curiosity_sleep_threshold_for_state(sleep_state_t state) {
    switch (state) {
        case SLEEP_STATE_AWAKE:      return CURIOSITY_SLEEP_THRESHOLD_AWAKE;
        case SLEEP_STATE_DROWSY:     return CURIOSITY_SLEEP_THRESHOLD_DROWSY;
        case SLEEP_STATE_LIGHT_NREM:
        case SLEEP_STATE_DEEP_NREM:  return CURIOSITY_SLEEP_THRESHOLD_NREM;
        case SLEEP_STATE_REM:        return CURIOSITY_SLEEP_THRESHOLD_REM;
        default:                     return CURIOSITY_SLEEP_THRESHOLD_AWAKE;
    }
}
