/**
 * @file nimcp_swarm_emergence_sleep_bridge.c
 * @brief Swarm Emergence-Sleep Integration Bridge Implementation
 * @version 1.0.0
 * @date 2025-12-18
 *
 * WHAT: Modulates swarm emergence based on sleep state
 * WHY:  Tier transitions and emergence capabilities should vary with alertness
 * HOW:  Sleep state callbacks dynamically adjust emergence parameters
 *
 * @author NIMCP Development Team
 */

#include "swarm/sleep/nimcp_swarm_emergence_sleep_bridge.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/platform/nimcp_platform.h"
#include "utils/logging/nimcp_logging.h"
#include "api/nimcp_api_exception.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <stdlib.h>
#include <string.h>

#include <stddef.h>  /* for NULL */
//=============================================================================
// Health Agent Integration (Phase 8: System-Wide Health Integration)
//=============================================================================
struct nimcp_health_agent;
typedef struct nimcp_health_agent nimcp_health_agent_t;
extern void nimcp_health_agent_heartbeat_ex(nimcp_health_agent_t* agent,
                                             const char* operation,
                                             float progress);

/** Global health agent for swarm_emergence_sleep_bridge module */
static nimcp_health_agent_t* g_swarm_emergence_sleep_bridge_health_agent = NULL;

/**
 * @brief Set health agent for swarm_emergence_sleep_bridge heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
static void swarm_emergence_sleep_bridge_set_health_agent(nimcp_health_agent_t* agent) {
    g_swarm_emergence_sleep_bridge_health_agent = agent;
}

/** @brief Send heartbeat from swarm_emergence_sleep_bridge module */
static inline void swarm_emergence_sleep_bridge_heartbeat(const char* operation, float progress) {
    if (g_swarm_emergence_sleep_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_swarm_emergence_sleep_bridge_health_agent, operation, progress);
    }
}


struct swarm_emergence_sleep_bridge_struct {
    bridge_base_t base;               /**< MUST be first: base bridge infrastructure */

    swarm_emergence_sleep_config_t config;
    sleep_system_t sleep_system;
    swarm_emergence_sleep_effects_t effects;
    bool callback_registered;
};

static void swarm_emergence_on_sleep_state_change(sleep_state_t new_state, void* user_data)
{
    swarm_emergence_sleep_bridge_t bridge = (swarm_emergence_sleep_bridge_t)user_data;
    if (!bridge) return;

    nimcp_mutex_lock(bridge->base.mutex);

    bridge->effects.current_state = new_state;
    bridge->effects.transition_rate_factor = swarm_emergence_sleep_get_trans_factor(new_state);
    bridge->effects.capability_threshold_factor = swarm_emergence_sleep_get_cap_factor(new_state);
    bridge->effects.coherence_factor = swarm_emergence_sleep_get_trans_factor(new_state);
    bridge->effects.emergence_enabled = (new_state != SLEEP_STATE_DEEP_NREM);

    NIMCP_LOGGING_DEBUG("Swarm emergence sleep state changed to %d, trans=%.2f, cap=%.2f",
                        new_state, bridge->effects.transition_rate_factor,
                        bridge->effects.capability_threshold_factor);

    nimcp_mutex_unlock(bridge->base.mutex);
}

int swarm_emergence_sleep_default_config(swarm_emergence_sleep_config_t* config)
{
    if (!config) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "config is NULL");

        return -1;

    }

    config->enable_transition_modulation = true;
    config->enable_capability_modulation = true;
    config->enable_coherence_modulation = true;
    config->modulation_strength = 1.0f;

    return 0;
}

swarm_emergence_sleep_bridge_t swarm_emergence_sleep_bridge_create(
    const swarm_emergence_sleep_config_t* config,
    sleep_system_t sleep_system)
{
    if (!config || !sleep_system) {
        NIMCP_LOGGING_ERROR("Invalid parameters for swarm emergence sleep bridge creation");
        return NULL;
    }

    swarm_emergence_sleep_bridge_t bridge =
        (swarm_emergence_sleep_bridge_t)nimcp_malloc(sizeof(struct swarm_emergence_sleep_bridge_struct));
    if (!bridge) {
        NIMCP_LOGGING_ERROR("Failed to allocate swarm emergence sleep bridge");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return NULL;
    }

    memset(bridge, 0, sizeof(struct swarm_emergence_sleep_bridge_struct));
    memcpy(&bridge->config, config, sizeof(swarm_emergence_sleep_config_t));
    bridge->sleep_system = sleep_system;

    if (bridge_base_init(&bridge->base, 0, "swarm_emergence_sleep") != 0) { nimcp_free(bridge); return NULL; }
    if (!bridge->base.mutex) {
        NIMCP_LOGGING_ERROR("Failed to create mutex for swarm emergence sleep bridge");
        nimcp_free(bridge);
        return NULL;
    }

    bridge->effects.transition_rate_factor = SWARM_EMERGENCE_SLEEP_TRANS_AWAKE;
    bridge->effects.capability_threshold_factor = SWARM_EMERGENCE_SLEEP_CAP_AWAKE;
    bridge->effects.coherence_factor = 1.0f;
    bridge->effects.current_state = SLEEP_STATE_AWAKE;
    bridge->effects.sleep_pressure = 0.0f;
    bridge->effects.emergence_enabled = true;

    bridge->callback_registered = sleep_register_state_callback(
        sleep_system, swarm_emergence_on_sleep_state_change, bridge);

    if (bridge->callback_registered) {
        sleep_state_t initial = sleep_get_current_state(sleep_system);
        swarm_emergence_on_sleep_state_change(initial, bridge);
    }

    NIMCP_LOGGING_INFO("Swarm emergence sleep bridge created");
    return bridge;
}

void swarm_emergence_sleep_bridge_destroy(swarm_emergence_sleep_bridge_t bridge)
{
    if (!bridge) return;

    if (bridge->callback_registered && bridge->sleep_system) {
        sleep_unregister_state_callback(bridge->sleep_system,
                                        swarm_emergence_on_sleep_state_change, bridge);
    }

    if (bridge->base.mutex) {
        bridge_base_cleanup(&bridge->base);
    }

    nimcp_free(bridge);
    NIMCP_LOGGING_INFO("Swarm emergence sleep bridge destroyed");
}

int swarm_emergence_sleep_update(swarm_emergence_sleep_bridge_t bridge)
{
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return -1;

    }

    nimcp_mutex_lock(bridge->base.mutex);
    if (bridge->sleep_system) {
        bridge->effects.sleep_pressure = sleep_get_pressure(bridge->sleep_system);
    }
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

int swarm_emergence_sleep_get_effects(const swarm_emergence_sleep_bridge_t bridge,
                                       swarm_emergence_sleep_effects_t* effects)
{
    if (!bridge || !effects) return -1;

    nimcp_mutex_lock(bridge->base.mutex);
    memcpy(effects, &bridge->effects, sizeof(swarm_emergence_sleep_effects_t));
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

float swarm_emergence_sleep_get_transition_rate(const swarm_emergence_sleep_bridge_t bridge, float base)
{
    if (!bridge) return base;

    nimcp_mutex_lock(bridge->base.mutex);
    float factor = bridge->config.enable_transition_modulation ?
        bridge->effects.transition_rate_factor : 1.0f;
    float result = base * (1.0f + (factor - 1.0f) * bridge->config.modulation_strength);
    nimcp_mutex_unlock(bridge->base.mutex);

    return result;
}

float swarm_emergence_sleep_get_capability_threshold(const swarm_emergence_sleep_bridge_t bridge, float base)
{
    if (!bridge) return base;

    nimcp_mutex_lock(bridge->base.mutex);
    float factor = bridge->config.enable_capability_modulation ?
        bridge->effects.capability_threshold_factor : 1.0f;
    float result = base * (1.0f + (factor - 1.0f) * bridge->config.modulation_strength);
    nimcp_mutex_unlock(bridge->base.mutex);

    return result;
}

float swarm_emergence_sleep_get_trans_factor(sleep_state_t state)
{
    switch (state) {
        case SLEEP_STATE_AWAKE:      return SWARM_EMERGENCE_SLEEP_TRANS_AWAKE;
        case SLEEP_STATE_DROWSY:     return SWARM_EMERGENCE_SLEEP_TRANS_DROWSY;
        case SLEEP_STATE_LIGHT_NREM: return SWARM_EMERGENCE_SLEEP_TRANS_LIGHT_NREM;
        case SLEEP_STATE_DEEP_NREM:  return SWARM_EMERGENCE_SLEEP_TRANS_DEEP_NREM;
        case SLEEP_STATE_REM:        return SWARM_EMERGENCE_SLEEP_TRANS_REM;
        default:                     return SWARM_EMERGENCE_SLEEP_TRANS_AWAKE;
    }
}

float swarm_emergence_sleep_get_cap_factor(sleep_state_t state)
{
    switch (state) {
        case SLEEP_STATE_AWAKE:      return SWARM_EMERGENCE_SLEEP_CAP_AWAKE;
        case SLEEP_STATE_DROWSY:     return SWARM_EMERGENCE_SLEEP_CAP_DROWSY;
        case SLEEP_STATE_LIGHT_NREM: return SWARM_EMERGENCE_SLEEP_CAP_LIGHT_NREM;
        case SLEEP_STATE_DEEP_NREM:  return SWARM_EMERGENCE_SLEEP_CAP_DEEP_NREM;
        case SLEEP_STATE_REM:        return SWARM_EMERGENCE_SLEEP_CAP_REM;
        default:                     return SWARM_EMERGENCE_SLEEP_CAP_AWAKE;
    }
}
