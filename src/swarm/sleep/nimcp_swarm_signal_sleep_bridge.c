/**
 * @file nimcp_swarm_signal_sleep_bridge.c
 * @brief Sleep-Swarm Signal Integration Bridge Implementation
 * @version 1.0.0
 * @date 2025-12-18
 *
 * WHAT: Modulates swarm signal transmission based on sleep state
 * WHY:  Transmission power and sensitivity should vary with alertness
 * HOW:  Sleep state callbacks dynamically adjust signal parameters
 *
 * @author NIMCP Development Team
 */

#include "swarm/sleep/nimcp_swarm_signal_sleep_bridge.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/platform/nimcp_platform.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <stdlib.h>
#include <string.h>

struct swarm_signal_sleep_bridge_struct {
    bridge_base_t base;               /**< MUST be first: base bridge infrastructure */

    swarm_signal_sleep_config_t config;
    sleep_system_t sleep_system;
    swarm_signal_sleep_effects_t effects;
    bool callback_registered;
};

static void swarm_signal_on_sleep_state_change(sleep_state_t new_state, void* user_data)
{
    swarm_signal_sleep_bridge_t bridge = (swarm_signal_sleep_bridge_t)user_data;
    if (!bridge) return;

    nimcp_mutex_lock(bridge->base.mutex);
    bridge->effects.current_state = new_state;

    if (bridge->config.enable_power_modulation) {
        bridge->effects.transmission_power_factor = swarm_signal_sleep_get_power_factor(new_state);
    }
    if (bridge->config.enable_reception_modulation) {
        bridge->effects.reception_sensitivity_factor = swarm_signal_sleep_get_recv_factor(new_state);
    }
    if (bridge->config.enable_latency_modulation) {
        bridge->effects.latency_tolerance_factor = swarm_signal_sleep_get_latency_factor(new_state);
    }
    bridge->effects.signaling_enabled = (new_state != SLEEP_STATE_DEEP_NREM);
    nimcp_mutex_unlock(bridge->base.mutex);
}

int swarm_signal_sleep_default_config(swarm_signal_sleep_config_t* config)
{
    if (!config) return -1;
    config->enable_power_modulation = true;
    config->enable_reception_modulation = true;
    config->enable_latency_modulation = true;
    config->modulation_strength = 1.0f;
    return 0;
}

swarm_signal_sleep_bridge_t swarm_signal_sleep_bridge_create(
    const swarm_signal_sleep_config_t* config, sleep_system_t sleep_system)
{
    if (!sleep_system) return NULL;

    struct swarm_signal_sleep_bridge_struct* bridge = nimcp_malloc(sizeof(*bridge));
    if (!bridge) return NULL;
    memset(bridge, 0, sizeof(*bridge));

    if (config) bridge->config = *config;
    else swarm_signal_sleep_default_config(&bridge->config);

    bridge->sleep_system = sleep_system;
    bridge->effects.transmission_power_factor = 1.0f;
    bridge->effects.reception_sensitivity_factor = 1.0f;
    bridge->effects.latency_tolerance_factor = 1.0f;
    bridge->effects.signaling_enabled = true;

    if (bridge_base_init(&bridge->base, 0, "swarm_signal_sleep") != 0) { nimcp_free(bridge); return NULL; }
    if (!bridge->base.mutex) { nimcp_free(bridge); return NULL; }

    bridge->callback_registered = sleep_register_state_callback(
        sleep_system, swarm_signal_on_sleep_state_change, bridge);

    sleep_state_t initial = sleep_get_current_state(sleep_system);
    swarm_signal_on_sleep_state_change(initial, bridge);

    NIMCP_LOGGING_INFO("Swarm signal-sleep bridge created");
    return bridge;
}

void swarm_signal_sleep_bridge_destroy(swarm_signal_sleep_bridge_t bridge)
{
    if (!bridge) return;
    if (bridge->callback_registered && bridge->sleep_system) {
        sleep_unregister_state_callback(bridge->sleep_system,
            swarm_signal_on_sleep_state_change, bridge);
    }
    if (bridge->base.mutex) bridge_base_cleanup(&bridge->base);
    nimcp_free(bridge);
}

int swarm_signal_sleep_update(swarm_signal_sleep_bridge_t bridge)
{
    if (!bridge) return -1;
    sleep_state_t state = sleep_get_current_state(bridge->sleep_system);
    swarm_signal_on_sleep_state_change(state, bridge);
    return 0;
}

int swarm_signal_sleep_get_effects(const swarm_signal_sleep_bridge_t bridge,
                                    swarm_signal_sleep_effects_t* effects)
{
    if (!bridge || !effects) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    *effects = bridge->effects;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

float swarm_signal_sleep_get_transmission_power(const swarm_signal_sleep_bridge_t bridge, float base)
{
    if (!bridge) return base;
    nimcp_mutex_lock(bridge->base.mutex);
    float result = base * bridge->effects.transmission_power_factor;
    nimcp_mutex_unlock(bridge->base.mutex);
    return result;
}

float swarm_signal_sleep_get_reception_sensitivity(const swarm_signal_sleep_bridge_t bridge, float base)
{
    if (!bridge) return base;
    nimcp_mutex_lock(bridge->base.mutex);
    float result = base * bridge->effects.reception_sensitivity_factor;
    nimcp_mutex_unlock(bridge->base.mutex);
    return result;
}

float swarm_signal_sleep_get_latency_tolerance(const swarm_signal_sleep_bridge_t bridge, float base)
{
    if (!bridge) return base;
    nimcp_mutex_lock(bridge->base.mutex);
    float result = base * bridge->effects.latency_tolerance_factor;
    nimcp_mutex_unlock(bridge->base.mutex);
    return result;
}

float swarm_signal_sleep_get_power_factor(sleep_state_t state)
{
    switch (state) {
        case SLEEP_STATE_AWAKE:      return SWARM_SIGNAL_SLEEP_POWER_AWAKE;
        case SLEEP_STATE_DROWSY:     return SWARM_SIGNAL_SLEEP_POWER_DROWSY;
        case SLEEP_STATE_LIGHT_NREM: return SWARM_SIGNAL_SLEEP_POWER_LIGHT_NREM;
        case SLEEP_STATE_DEEP_NREM:  return SWARM_SIGNAL_SLEEP_POWER_DEEP_NREM;
        case SLEEP_STATE_REM:        return SWARM_SIGNAL_SLEEP_POWER_REM;
        default:                     return SWARM_SIGNAL_SLEEP_POWER_AWAKE;
    }
}

float swarm_signal_sleep_get_recv_factor(sleep_state_t state)
{
    switch (state) {
        case SLEEP_STATE_AWAKE:      return SWARM_SIGNAL_SLEEP_RECV_AWAKE;
        case SLEEP_STATE_DROWSY:     return SWARM_SIGNAL_SLEEP_RECV_DROWSY;
        case SLEEP_STATE_LIGHT_NREM: return SWARM_SIGNAL_SLEEP_RECV_LIGHT_NREM;
        case SLEEP_STATE_DEEP_NREM:  return SWARM_SIGNAL_SLEEP_RECV_DEEP_NREM;
        case SLEEP_STATE_REM:        return SWARM_SIGNAL_SLEEP_RECV_REM;
        default:                     return SWARM_SIGNAL_SLEEP_RECV_AWAKE;
    }
}

float swarm_signal_sleep_get_latency_factor(sleep_state_t state)
{
    switch (state) {
        case SLEEP_STATE_AWAKE:      return SWARM_SIGNAL_SLEEP_LATENCY_AWAKE;
        case SLEEP_STATE_DROWSY:     return SWARM_SIGNAL_SLEEP_LATENCY_DROWSY;
        case SLEEP_STATE_LIGHT_NREM: return SWARM_SIGNAL_SLEEP_LATENCY_LIGHT_NREM;
        case SLEEP_STATE_DEEP_NREM:  return SWARM_SIGNAL_SLEEP_LATENCY_DEEP_NREM;
        case SLEEP_STATE_REM:        return SWARM_SIGNAL_SLEEP_LATENCY_REM;
        default:                     return SWARM_SIGNAL_SLEEP_LATENCY_AWAKE;
    }
}
