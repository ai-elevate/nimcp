/**
 * @file nimcp_working_memory_sleep_bridge.c
 * @brief Sleep-Working Memory Integration Bridge Implementation
 * @version 1.0.0
 * @date 2025-12-17
 */

#include "cognitive/working_memory/nimcp_working_memory_sleep_bridge.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/platform/nimcp_platform_mutex.h"
#include <string.h>

struct working_memory_sleep_bridge_struct {
    bridge_base_t base;               /**< MUST be first: base bridge infrastructure */

    working_memory_sleep_config_t config;
    sleep_system_t sleep_system;
    working_memory_sleep_effects_t effects;
    bool callback_registered;  /* Track if callback is registered for cleanup */
};

/* Forward declarations */
static void working_memory_on_sleep_state_change(sleep_state_t new_state, void* user_data);

/**
 * WHAT: Callback invoked when sleep state changes
 * WHY:  Immediately update WM parameters for new sleep state
 * HOW:  Called by sleep system via observer pattern
 *
 * BIOLOGICAL BASIS:
 * - WM capacity drops sharply with drowsiness (prefrontal hypofunction)
 * - Deep NREM enables consolidation to long-term memory
 * - Sleep deprivation severely impairs working memory span
 */
static void working_memory_on_sleep_state_change(sleep_state_t new_state, void* user_data)
{
    working_memory_sleep_bridge_t bridge = (working_memory_sleep_bridge_t)user_data;

    if (!bridge) {
        NIMCP_LOGGING_ERROR("NULL bridge in sleep state callback");
        return;
    }

    NIMCP_LOGGING_DEBUG("Working memory bridge received sleep state: %d", new_state);

    nimcp_mutex_lock(bridge->base.mutex);

    bridge->effects.current_state = new_state;

    if (bridge->config.enable_capacity_modulation) {
        bridge->effects.capacity_factor = working_memory_sleep_capacity_for_state(new_state);
    }

    if (bridge->config.enable_decay_modulation) {
        bridge->effects.decay_rate_factor = working_memory_sleep_decay_for_state(new_state);
    }

    /* Rehearsal efficiency drops with drowsiness */
    bridge->effects.rehearsal_efficiency = (new_state == SLEEP_STATE_AWAKE) ? 1.0f :
                                           (new_state == SLEEP_STATE_DROWSY) ? 0.5f : 0.0f;

    bridge->effects.wm_offline = (new_state == SLEEP_STATE_DEEP_NREM);
    bridge->effects.consolidation_active = (new_state == SLEEP_STATE_DEEP_NREM ||
                                            new_state == SLEEP_STATE_LIGHT_NREM);

    nimcp_mutex_unlock(bridge->base.mutex);

    NIMCP_LOGGING_DEBUG("WM modulated: capacity=%.2f, decay=%.2f, offline=%d",
                        bridge->effects.capacity_factor,
                        bridge->effects.decay_rate_factor,
                        bridge->effects.wm_offline);
}

int working_memory_sleep_default_config(working_memory_sleep_config_t* config) {
    if (!config) return -1;
    config->enable_capacity_modulation = true;
    config->enable_decay_modulation = true;
    config->enable_transfer_on_sleep = true;
    config->modulation_strength = 1.0f;
    return 0;
}

working_memory_sleep_bridge_t working_memory_sleep_bridge_create(
    const working_memory_sleep_config_t* config,
    sleep_system_t sleep)
{
    if (!sleep) return NULL;

    struct working_memory_sleep_bridge_struct* bridge =
        (struct working_memory_sleep_bridge_struct*)nimcp_malloc(
            sizeof(struct working_memory_sleep_bridge_struct));
    if (!bridge) return NULL;

    memset(bridge, 0, sizeof(struct working_memory_sleep_bridge_struct));

    if (config) bridge->config = *config;
    else working_memory_sleep_default_config(&bridge->config);

    bridge->sleep_system = sleep;
    bridge->effects.capacity_factor = 1.0f;
    bridge->effects.decay_rate_factor = 1.0f;
    bridge->effects.rehearsal_efficiency = 1.0f;
    bridge->effects.wm_offline = false;
    bridge->effects.consolidation_active = false;

    bridge->base.mutex = nimcp_platform_mutex_create();
    if (!bridge->base.mutex) { nimcp_free(bridge); return NULL; }

    /* Register callback for automatic state updates */
    bridge->callback_registered = sleep_register_state_callback(
        sleep,
        working_memory_on_sleep_state_change,
        bridge);

    if (!bridge->callback_registered) {
        NIMCP_LOGGING_WARN("Failed to register sleep state callback - will use polling");
    } else {
        NIMCP_LOGGING_DEBUG("Registered sleep state callback for working memory bridge");
    }

    /* Get initial state immediately */
    sleep_state_t initial_state = sleep_get_current_state(sleep);
    working_memory_on_sleep_state_change(initial_state, bridge);

    NIMCP_LOGGING_INFO("Working memory-sleep bridge created");
    return bridge;
}

void working_memory_sleep_bridge_destroy(working_memory_sleep_bridge_t bridge) {
    if (!bridge) return;

    /* Unregister callback if it was registered */
    if (bridge->callback_registered && bridge->sleep_system) {
        bool unregistered = sleep_unregister_state_callback(
            bridge->sleep_system,
            working_memory_on_sleep_state_change,
            bridge);

        if (unregistered) {
            NIMCP_LOGGING_DEBUG("Unregistered sleep state callback for working memory bridge");
        }
    }

    if (bridge->base.mutex) nimcp_mutex_destroy(bridge->base.mutex);
    nimcp_free(bridge);
}

int working_memory_sleep_update(working_memory_sleep_bridge_t bridge) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->base.mutex);

    sleep_state_t state = sleep_get_current_state(bridge->sleep_system);
    float pressure = sleep_get_pressure(bridge->sleep_system);

    bridge->effects.current_state = state;
    bridge->effects.sleep_pressure = pressure;

    if (bridge->config.enable_capacity_modulation) {
        float cap_base = working_memory_sleep_capacity_for_state(state);
        bridge->effects.capacity_factor = cap_base;
        /* High sleep pressure reduces WM capacity even when awake */
        if (state == SLEEP_STATE_AWAKE && pressure > 0.6f) {
            bridge->effects.capacity_factor *= (1.0f - 0.3f * (pressure - 0.6f) / 0.4f);
        }
    }

    if (bridge->config.enable_decay_modulation) {
        bridge->effects.decay_rate_factor = working_memory_sleep_decay_for_state(state);
    }

    /* Rehearsal efficiency drops with drowsiness */
    bridge->effects.rehearsal_efficiency = (state == SLEEP_STATE_AWAKE) ? 1.0f :
                                           (state == SLEEP_STATE_DROWSY) ? 0.5f : 0.0f;

    bridge->effects.wm_offline = (state == SLEEP_STATE_DEEP_NREM);
    bridge->effects.consolidation_active = (state == SLEEP_STATE_DEEP_NREM ||
                                            state == SLEEP_STATE_LIGHT_NREM);

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int working_memory_sleep_get_effects(
    const working_memory_sleep_bridge_t bridge,
    working_memory_sleep_effects_t* effects)
{
    if (!bridge || !effects) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    *effects = bridge->effects;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

float working_memory_sleep_get_capacity(const working_memory_sleep_bridge_t bridge) {
    if (!bridge) return 1.0f;
    nimcp_mutex_lock(bridge->base.mutex);
    float result = bridge->effects.capacity_factor;
    nimcp_mutex_unlock(bridge->base.mutex);
    return result;
}

bool working_memory_sleep_is_offline(const working_memory_sleep_bridge_t bridge) {
    if (!bridge) return false;
    nimcp_mutex_lock(bridge->base.mutex);
    bool result = bridge->effects.wm_offline;
    nimcp_mutex_unlock(bridge->base.mutex);
    return result;
}

float working_memory_sleep_capacity_for_state(sleep_state_t state) {
    switch (state) {
        case SLEEP_STATE_AWAKE:      return WM_SLEEP_CAPACITY_AWAKE;
        case SLEEP_STATE_DROWSY:     return WM_SLEEP_CAPACITY_DROWSY;
        case SLEEP_STATE_LIGHT_NREM: return WM_SLEEP_CAPACITY_LIGHT_NREM;
        case SLEEP_STATE_DEEP_NREM:  return WM_SLEEP_CAPACITY_DEEP_NREM;
        case SLEEP_STATE_REM:        return WM_SLEEP_CAPACITY_REM;
        default:                     return WM_SLEEP_CAPACITY_AWAKE;
    }
}

float working_memory_sleep_decay_for_state(sleep_state_t state) {
    switch (state) {
        case SLEEP_STATE_AWAKE:      return WM_SLEEP_DECAY_AWAKE;
        case SLEEP_STATE_DROWSY:     return WM_SLEEP_DECAY_DROWSY;
        case SLEEP_STATE_LIGHT_NREM: return WM_SLEEP_DECAY_LIGHT_NREM;
        case SLEEP_STATE_DEEP_NREM:  return WM_SLEEP_DECAY_DEEP_NREM;
        case SLEEP_STATE_REM:        return WM_SLEEP_DECAY_REM;
        default:                     return WM_SLEEP_DECAY_AWAKE;
    }
}
