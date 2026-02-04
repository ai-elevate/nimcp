/**
 * @file nimcp_swarm_consensus_sleep_bridge.c
 * @brief Sleep-Swarm Consensus Integration Bridge Implementation
 * @version 1.0.0
 * @date 2025-12-18
 *
 * WHAT: Modulates swarm consensus voting based on sleep state
 * WHY:  Voting frequency and quorum thresholds should vary with alertness
 * HOW:  Sleep state callbacks dynamically adjust consensus parameters
 *
 * @author NIMCP Development Team
 */

#include "swarm/sleep/nimcp_swarm_consensus_sleep_bridge.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/platform/nimcp_platform.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <stdlib.h>
#include <string.h>
#include "utils/fault_tolerance/nimcp_health_agent_macros.h"

NIMCP_DECLARE_HEALTH_AGENT_ATOMIC(swarm_consensus_sleep_bridge)

struct swarm_consensus_sleep_bridge_struct {
    bridge_base_t base;               /**< MUST be first: base bridge infrastructure */

    swarm_consensus_sleep_config_t config;
    sleep_system_t sleep_system;
    swarm_consensus_sleep_effects_t effects;
    bool callback_registered;
};

static void swarm_consensus_on_sleep_state_change(sleep_state_t new_state, void* user_data)
{
    swarm_consensus_sleep_bridge_t bridge = (swarm_consensus_sleep_bridge_t)user_data;
    if (!bridge) return;

    nimcp_mutex_lock(bridge->base.mutex);
    bridge->effects.current_state = new_state;

    if (bridge->config.enable_voting_modulation) {
        bridge->effects.voting_frequency_factor = swarm_consensus_sleep_get_vote_factor(new_state);
    }
    if (bridge->config.enable_quorum_modulation) {
        bridge->effects.quorum_threshold_factor = swarm_consensus_sleep_get_quorum_factor(new_state);
    }
    if (bridge->config.enable_timeout_modulation) {
        bridge->effects.timeout_multiplier = swarm_consensus_sleep_get_quorum_factor(new_state);
    }
    bridge->effects.consensus_enabled = (new_state != SLEEP_STATE_DEEP_NREM);
    nimcp_mutex_unlock(bridge->base.mutex);
}

int swarm_consensus_sleep_default_config(swarm_consensus_sleep_config_t* config)
{
    if (!config) return -1;
    config->enable_voting_modulation = true;
    config->enable_quorum_modulation = true;
    config->enable_timeout_modulation = true;
    config->modulation_strength = 1.0f;
    return 0;
}

swarm_consensus_sleep_bridge_t swarm_consensus_sleep_bridge_create(
    const swarm_consensus_sleep_config_t* config, sleep_system_t sleep_system)
{
    if (!sleep_system) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "sleep_system is NULL");

        return NULL;

    }

    struct swarm_consensus_sleep_bridge_struct* bridge = nimcp_malloc(sizeof(*bridge));
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return NULL;

    }
    memset(bridge, 0, sizeof(*bridge));

    if (config) bridge->config = *config;
    else swarm_consensus_sleep_default_config(&bridge->config);

    bridge->sleep_system = sleep_system;
    bridge->effects.voting_frequency_factor = 1.0f;
    bridge->effects.quorum_threshold_factor = 1.0f;
    bridge->effects.timeout_multiplier = 1.0f;
    bridge->effects.consensus_enabled = true;

    if (bridge_base_init(&bridge->base, 0, "swarm_consensus_sleep") != 0) { nimcp_free(bridge); return NULL; }
    if (!bridge->base.mutex) { nimcp_free(bridge); return NULL; }

    bridge->callback_registered = sleep_register_state_callback(
        sleep_system, swarm_consensus_on_sleep_state_change, bridge);

    sleep_state_t initial = sleep_get_current_state(sleep_system);
    swarm_consensus_on_sleep_state_change(initial, bridge);

    NIMCP_LOGGING_INFO("Swarm consensus-sleep bridge created");
    return bridge;
}

void swarm_consensus_sleep_bridge_destroy(swarm_consensus_sleep_bridge_t bridge)
{
    if (!bridge) return;
    if (bridge->callback_registered && bridge->sleep_system) {
        sleep_unregister_state_callback(bridge->sleep_system,
            swarm_consensus_on_sleep_state_change, bridge);
    }
    if (bridge->base.mutex) bridge_base_cleanup(&bridge->base);
    nimcp_free(bridge);
}

int swarm_consensus_sleep_update(swarm_consensus_sleep_bridge_t bridge)
{
    if (!bridge) return -1;
    sleep_state_t state = sleep_get_current_state(bridge->sleep_system);
    swarm_consensus_on_sleep_state_change(state, bridge);
    return 0;
}

int swarm_consensus_sleep_get_effects(const swarm_consensus_sleep_bridge_t bridge,
                                       swarm_consensus_sleep_effects_t* effects)
{
    if (!bridge || !effects) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    *effects = bridge->effects;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

float swarm_consensus_sleep_get_voting_frequency(const swarm_consensus_sleep_bridge_t bridge, float base)
{
    if (!bridge) return base;
    nimcp_mutex_lock(bridge->base.mutex);
    float result = base * bridge->effects.voting_frequency_factor;
    nimcp_mutex_unlock(bridge->base.mutex);
    return result;
}

float swarm_consensus_sleep_get_quorum_threshold(const swarm_consensus_sleep_bridge_t bridge, float base)
{
    if (!bridge) return base;
    nimcp_mutex_lock(bridge->base.mutex);
    float result = base * bridge->effects.quorum_threshold_factor;
    nimcp_mutex_unlock(bridge->base.mutex);
    return result;
}

uint32_t swarm_consensus_sleep_get_timeout(const swarm_consensus_sleep_bridge_t bridge, uint32_t base_ms)
{
    if (!bridge) return base_ms;
    nimcp_mutex_lock(bridge->base.mutex);
    uint32_t result = (uint32_t)(base_ms * bridge->effects.timeout_multiplier);
    nimcp_mutex_unlock(bridge->base.mutex);
    return result;
}

float swarm_consensus_sleep_get_vote_factor(sleep_state_t state)
{
    switch (state) {
        case SLEEP_STATE_AWAKE:      return SWARM_CONSENSUS_SLEEP_VOTE_AWAKE;
        case SLEEP_STATE_DROWSY:     return SWARM_CONSENSUS_SLEEP_VOTE_DROWSY;
        case SLEEP_STATE_LIGHT_NREM: return SWARM_CONSENSUS_SLEEP_VOTE_LIGHT_NREM;
        case SLEEP_STATE_DEEP_NREM:  return SWARM_CONSENSUS_SLEEP_VOTE_DEEP_NREM;
        case SLEEP_STATE_REM:        return SWARM_CONSENSUS_SLEEP_VOTE_REM;
        default:                     return SWARM_CONSENSUS_SLEEP_VOTE_AWAKE;
    }
}

float swarm_consensus_sleep_get_quorum_factor(sleep_state_t state)
{
    switch (state) {
        case SLEEP_STATE_AWAKE:      return SWARM_CONSENSUS_SLEEP_QUORUM_AWAKE;
        case SLEEP_STATE_DROWSY:     return SWARM_CONSENSUS_SLEEP_QUORUM_DROWSY;
        case SLEEP_STATE_LIGHT_NREM: return SWARM_CONSENSUS_SLEEP_QUORUM_LIGHT_NREM;
        case SLEEP_STATE_DEEP_NREM:  return SWARM_CONSENSUS_SLEEP_QUORUM_DEEP_NREM;
        case SLEEP_STATE_REM:        return SWARM_CONSENSUS_SLEEP_QUORUM_REM;
        default:                     return SWARM_CONSENSUS_SLEEP_QUORUM_AWAKE;
    }
}
