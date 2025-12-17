/**
 * @file nimcp_oscillations_sleep_bridge.c
 * @brief Sleep-Oscillations Integration Bridge Implementation
 * @version 1.0.0
 * @date 2025-12-17
 *
 * Brain oscillations define sleep stages - this is the core interface.
 */

#include "core/brain_oscillations/nimcp_oscillations_sleep_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/platform/nimcp_platform_mutex.h"
#include <string.h>

struct oscillations_sleep_bridge_struct {
    oscillations_sleep_config_t config;
    sleep_system_t sleep_system;
    oscillations_sleep_effects_t effects;
    nimcp_mutex_t* mutex;
};

int oscillations_sleep_default_config(oscillations_sleep_config_t* config) {
    if (!config) return -1;
    config->enable_frequency_modulation = true;
    config->enable_power_modulation = true;
    config->enable_spindle_generation = true;
    config->modulation_strength = 1.0f;
    return 0;
}

oscillations_sleep_bridge_t oscillations_sleep_bridge_create(
    const oscillations_sleep_config_t* config,
    sleep_system_t sleep)
{
    if (!sleep) return NULL;

    struct oscillations_sleep_bridge_struct* bridge =
        (struct oscillations_sleep_bridge_struct*)nimcp_malloc(
            sizeof(struct oscillations_sleep_bridge_struct));
    if (!bridge) return NULL;

    memset(bridge, 0, sizeof(struct oscillations_sleep_bridge_struct));

    if (config) bridge->config = *config;
    else oscillations_sleep_default_config(&bridge->config);

    bridge->sleep_system = sleep;
    bridge->effects.dominant_frequency = OSC_SLEEP_FREQ_AWAKE;
    bridge->effects.dominant_band = OSC_BAND_BETA;
    bridge->effects.spindle_activity = 0.0f;
    bridge->effects.ripple_activity = 0.0f;

    bridge->mutex = nimcp_platform_mutex_create();
    if (!bridge->mutex) { nimcp_free(bridge); return NULL; }

    NIMCP_LOGGING_INFO("Oscillations-sleep bridge created");
    return bridge;
}

void oscillations_sleep_bridge_destroy(oscillations_sleep_bridge_t bridge) {
    if (!bridge) return;
    if (bridge->mutex) nimcp_mutex_destroy(bridge->mutex);
    nimcp_free(bridge);
}

int oscillations_sleep_update(oscillations_sleep_bridge_t bridge) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);

    sleep_state_t state = sleep_get_current_state(bridge->sleep_system);
    float pressure = sleep_get_pressure(bridge->sleep_system);

    bridge->effects.current_state = state;
    bridge->effects.sleep_pressure = pressure;

    if (bridge->config.enable_frequency_modulation) {
        bridge->effects.dominant_frequency = oscillations_sleep_freq_for_state(state);
        bridge->effects.dominant_band = oscillations_sleep_band_for_state(state);
    }

    if (bridge->config.enable_power_modulation) {
        /* Set band power distribution based on sleep state */
        bridge->effects.delta_power = (state == SLEEP_STATE_DEEP_NREM) ? 0.8f :
                                      (state == SLEEP_STATE_LIGHT_NREM) ? 0.3f : 0.1f;
        bridge->effects.theta_power = (state == SLEEP_STATE_REM ||
                                       state == SLEEP_STATE_LIGHT_NREM) ? 0.6f :
                                      (state == SLEEP_STATE_DROWSY) ? 0.3f : 0.2f;
        bridge->effects.alpha_power = (state == SLEEP_STATE_DROWSY) ? 0.7f :
                                      (state == SLEEP_STATE_AWAKE) ? 0.3f : 0.1f;
        bridge->effects.beta_power = (state == SLEEP_STATE_AWAKE) ? 0.6f :
                                     (state == SLEEP_STATE_DROWSY) ? 0.3f : 0.1f;
        bridge->effects.gamma_power = (state == SLEEP_STATE_AWAKE) ? 0.5f : 0.1f;
    }

    if (bridge->config.enable_spindle_generation) {
        bridge->effects.spindle_activity = oscillations_sleep_spindle_for_state(state);
        /* Sharp wave ripples during NREM for memory replay */
        bridge->effects.ripple_activity = (state == SLEEP_STATE_DEEP_NREM) ? 0.7f :
                                          (state == SLEEP_STATE_LIGHT_NREM) ? 0.4f : 0.0f;
    }

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int oscillations_sleep_get_effects(
    const oscillations_sleep_bridge_t bridge,
    oscillations_sleep_effects_t* effects)
{
    if (!bridge || !effects) return -1;
    nimcp_mutex_lock(bridge->mutex);
    *effects = bridge->effects;
    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

float oscillations_sleep_get_frequency(const oscillations_sleep_bridge_t bridge) {
    if (!bridge) return OSC_SLEEP_FREQ_AWAKE;
    nimcp_mutex_lock(bridge->mutex);
    float result = bridge->effects.dominant_frequency;
    nimcp_mutex_unlock(bridge->mutex);
    return result;
}

oscillation_band_t oscillations_sleep_get_band(const oscillations_sleep_bridge_t bridge) {
    if (!bridge) return OSC_BAND_BETA;
    nimcp_mutex_lock(bridge->mutex);
    oscillation_band_t result = bridge->effects.dominant_band;
    nimcp_mutex_unlock(bridge->mutex);
    return result;
}

float oscillations_sleep_freq_for_state(sleep_state_t state) {
    switch (state) {
        case SLEEP_STATE_AWAKE:      return OSC_SLEEP_FREQ_AWAKE;
        case SLEEP_STATE_DROWSY:     return OSC_SLEEP_FREQ_DROWSY;
        case SLEEP_STATE_LIGHT_NREM: return OSC_SLEEP_FREQ_LIGHT_NREM;
        case SLEEP_STATE_DEEP_NREM:  return OSC_SLEEP_FREQ_DEEP_NREM;
        case SLEEP_STATE_REM:        return OSC_SLEEP_FREQ_REM;
        default:                     return OSC_SLEEP_FREQ_AWAKE;
    }
}

oscillation_band_t oscillations_sleep_band_for_state(sleep_state_t state) {
    switch (state) {
        case SLEEP_STATE_AWAKE:      return OSC_BAND_BETA;
        case SLEEP_STATE_DROWSY:     return OSC_BAND_ALPHA;
        case SLEEP_STATE_LIGHT_NREM: return OSC_BAND_THETA;
        case SLEEP_STATE_DEEP_NREM:  return OSC_BAND_DELTA;
        case SLEEP_STATE_REM:        return OSC_BAND_THETA;
        default:                     return OSC_BAND_BETA;
    }
}

float oscillations_sleep_spindle_for_state(sleep_state_t state) {
    switch (state) {
        case SLEEP_STATE_AWAKE:      return OSC_SLEEP_SPINDLE_AWAKE;
        case SLEEP_STATE_DROWSY:     return OSC_SLEEP_SPINDLE_DROWSY;
        case SLEEP_STATE_LIGHT_NREM: return OSC_SLEEP_SPINDLE_LIGHT;
        case SLEEP_STATE_DEEP_NREM:  return OSC_SLEEP_SPINDLE_DEEP;
        case SLEEP_STATE_REM:        return OSC_SLEEP_SPINDLE_REM;
        default:                     return OSC_SLEEP_SPINDLE_AWAKE;
    }
}
