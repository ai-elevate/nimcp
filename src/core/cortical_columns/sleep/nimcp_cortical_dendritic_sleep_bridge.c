/**
 * @file nimcp_cortical_dendritic_sleep_bridge.c
 * @brief Sleep-Cortical Dendritic Processing Integration Bridge Implementation
 * @version 1.0.0
 * @date 2025-12-17
 */

#include "core/cortical_columns/sleep/nimcp_cortical_dendritic_sleep_bridge.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/platform/nimcp_platform_mutex.h"
#include <string.h>

struct cortical_dendritic_sleep_bridge_struct {
    bridge_base_t base;               /**< MUST be first: base bridge infrastructure */

    cortical_dendritic_sleep_config_t config;
    void* dendritic_module;
    sleep_system_t sleep_system;
    cortical_dendritic_sleep_effects_t effects;
    bool callback_registered;
};

static void cortical_dendritic_on_sleep_state_change(sleep_state_t new_state, void* user_data)
{
    cortical_dendritic_sleep_bridge_t bridge = (cortical_dendritic_sleep_bridge_t)user_data;
    if (!bridge) return;

    nimcp_mutex_lock(bridge->base.mutex);
    bridge->effects.current_state = new_state;

    if (bridge->config.enable_excitability_modulation) {
        switch (new_state) {
            case SLEEP_STATE_AWAKE:
            case SLEEP_STATE_DROWSY:
                bridge->effects.dendritic_excitability = DENDRITIC_SLEEP_EXCITABILITY_AWAKE;
                bridge->effects.integration_window = 1.0f;
                break;
            case SLEEP_STATE_LIGHT_NREM:
            case SLEEP_STATE_DEEP_NREM:
                bridge->effects.dendritic_excitability = DENDRITIC_SLEEP_EXCITABILITY_NREM;
                bridge->effects.integration_window = 0.5f;  /* Slower integration */
                break;
            case SLEEP_STATE_REM:
                bridge->effects.dendritic_excitability = DENDRITIC_SLEEP_EXCITABILITY_REM;
                bridge->effects.integration_window = 0.8f;
                break;
        }
    }

    bridge->effects.dendritic_offline = (new_state == SLEEP_STATE_DEEP_NREM);
    nimcp_mutex_unlock(bridge->base.mutex);
}

int cortical_dendritic_sleep_default_config(cortical_dendritic_sleep_config_t* config)
{
    if (!config) return -1;
    config->enable_excitability_modulation = true;
    config->modulation_strength = 1.0f;
    return 0;
}

cortical_dendritic_sleep_bridge_t cortical_dendritic_sleep_bridge_create(
    const cortical_dendritic_sleep_config_t* config,
    void* dendritic_module,
    sleep_system_t sleep)
{
    if (!dendritic_module || !sleep) return NULL;

    struct cortical_dendritic_sleep_bridge_struct* bridge =
        (struct cortical_dendritic_sleep_bridge_struct*)nimcp_malloc(
            sizeof(struct cortical_dendritic_sleep_bridge_struct));
    if (!bridge) return NULL;

    memset(bridge, 0, sizeof(struct cortical_dendritic_sleep_bridge_struct));

    if (config) bridge->config = *config;
    else cortical_dendritic_sleep_default_config(&bridge->config);

    bridge->dendritic_module = dendritic_module;
    bridge->sleep_system = sleep;

    bridge->base.mutex = nimcp_platform_mutex_create();
    if (!bridge->base.mutex) {
        nimcp_free(bridge);
        return NULL;
    }

    bridge->effects.current_state = SLEEP_STATE_AWAKE;
    bridge->effects.dendritic_excitability = 1.0f;
    bridge->effects.integration_window = 1.0f;
    bridge->effects.dendritic_offline = false;

    bool registered = sleep_register_state_callback(sleep, cortical_dendritic_on_sleep_state_change, bridge);
    if (registered) bridge->callback_registered = true;

    return bridge;
}

void cortical_dendritic_sleep_bridge_destroy(cortical_dendritic_sleep_bridge_t bridge)
{
    if (!bridge) return;
    if (bridge->callback_registered) {
        sleep_unregister_state_callback(bridge->sleep_system, cortical_dendritic_on_sleep_state_change, bridge);
    }
    if (bridge->base.mutex) nimcp_platform_mutex_destroy(bridge->base.mutex);
    nimcp_free(bridge);
}

int cortical_dendritic_sleep_update(cortical_dendritic_sleep_bridge_t bridge)
{
    if (!bridge) return -1;
    return 0;
}

float cortical_dendritic_sleep_get_excitability(const cortical_dendritic_sleep_bridge_t bridge)
{
    if (!bridge) return -1.0f;
    nimcp_mutex_lock(bridge->base.mutex);
    float excitability = bridge->effects.dendritic_excitability;
    nimcp_mutex_unlock(bridge->base.mutex);
    return excitability;
}
