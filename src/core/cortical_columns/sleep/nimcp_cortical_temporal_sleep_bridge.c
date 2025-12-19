/**
 * @file nimcp_cortical_temporal_sleep_bridge.c
 * @brief Sleep-Cortical Temporal Processing Integration Bridge Implementation
 * @version 1.0.0
 * @date 2025-12-17
 */

#include "core/cortical_columns/sleep/nimcp_cortical_temporal_sleep_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/platform/nimcp_platform_mutex.h"
#include <string.h>

struct cortical_temporal_sleep_bridge_struct {
    cortical_temporal_sleep_config_t config;
    void* temporal_module;
    sleep_system_t sleep_system;
    cortical_temporal_sleep_effects_t effects;
    nimcp_mutex_t* mutex;
    bool callback_registered;
};

static void cortical_temporal_on_sleep_state_change(sleep_state_t new_state, void* user_data)
{
    cortical_temporal_sleep_bridge_t bridge = (cortical_temporal_sleep_bridge_t)user_data;
    if (!bridge) return;

    nimcp_mutex_lock(bridge->mutex);
    bridge->effects.current_state = new_state;

    if (bridge->config.enable_timescale_modulation) {
        switch (new_state) {
            case SLEEP_STATE_AWAKE:
            case SLEEP_STATE_DROWSY:
                bridge->effects.timescale_factor = TEMPORAL_SLEEP_TIMESCALE_AWAKE;
                break;
            case SLEEP_STATE_LIGHT_NREM:
            case SLEEP_STATE_DEEP_NREM:
                bridge->effects.timescale_factor = TEMPORAL_SLEEP_TIMESCALE_NREM;
                break;
            case SLEEP_STATE_REM:
                bridge->effects.timescale_factor = TEMPORAL_SLEEP_TIMESCALE_REM;
                break;
        }
    }

    bridge->effects.temporal_offline = (new_state == SLEEP_STATE_DEEP_NREM);
    nimcp_mutex_unlock(bridge->mutex);
}

int cortical_temporal_sleep_default_config(cortical_temporal_sleep_config_t* config)
{
    if (!config) return -1;
    config->enable_timescale_modulation = true;
    config->modulation_strength = 1.0f;
    return 0;
}

cortical_temporal_sleep_bridge_t cortical_temporal_sleep_bridge_create(
    const cortical_temporal_sleep_config_t* config,
    void* temporal_module,
    sleep_system_t sleep)
{
    if (!temporal_module || !sleep) return NULL;

    struct cortical_temporal_sleep_bridge_struct* bridge =
        (struct cortical_temporal_sleep_bridge_struct*)nimcp_malloc(
            sizeof(struct cortical_temporal_sleep_bridge_struct));
    if (!bridge) return NULL;

    memset(bridge, 0, sizeof(struct cortical_temporal_sleep_bridge_struct));

    if (config) bridge->config = *config;
    else cortical_temporal_sleep_default_config(&bridge->config);

    bridge->temporal_module = temporal_module;
    bridge->sleep_system = sleep;

    bridge->mutex = nimcp_mutex_create();
    if (!bridge->mutex) {
        nimcp_free(bridge);
        return NULL;
    }

    bridge->effects.current_state = SLEEP_STATE_AWAKE;
    bridge->effects.timescale_factor = 1.0f;
    bridge->effects.temporal_offline = false;

    bool registered = sleep_register_state_callback(sleep, cortical_temporal_on_sleep_state_change, bridge);
    if (registered) bridge->callback_registered = true;

    return bridge;
}

void cortical_temporal_sleep_bridge_destroy(cortical_temporal_sleep_bridge_t bridge)
{
    if (!bridge) return;
    if (bridge->callback_registered) {
        sleep_unregister_state_callback(bridge->sleep_system, cortical_temporal_on_sleep_state_change, bridge);
    }
    if (bridge->mutex) nimcp_mutex_destroy(bridge->mutex);
    nimcp_free(bridge);
}

int cortical_temporal_sleep_update(cortical_temporal_sleep_bridge_t bridge)
{
    if (!bridge) return -1;
    return 0;
}

float cortical_temporal_sleep_get_timescale(const cortical_temporal_sleep_bridge_t bridge)
{
    if (!bridge) return -1.0f;
    nimcp_mutex_lock(bridge->mutex);
    float timescale = bridge->effects.timescale_factor;
    nimcp_mutex_unlock(bridge->mutex);
    return timescale;
}
