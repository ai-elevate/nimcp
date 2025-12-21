/**
 * @file nimcp_swarm_consensus_immune_bridge.c
 * @brief Swarm Consensus-Immune System Integration Bridge Implementation
 * @version 1.0.0
 * @date 2025-12-18
 *
 * WHAT: Modulates swarm consensus voting based on immune state
 * WHY:  Inflammation affects collective decision-making; failures trigger immune
 * HOW:  Cytokines delay voting; failed consensus triggers inflammation
 *
 * @author NIMCP Development Team
 */

#include "swarm/immune/nimcp_swarm_consensus_immune_bridge.h"
#include "utils/platform/nimcp_platform.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/memory/nimcp_memory.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"
#include <stdlib.h>
#include <string.h>

struct swarm_consensus_immune_bridge_struct {
    swarm_consensus_immune_config_t config;
    brain_immune_system_t* immune_system;
    void* swarm_consensus;
    cytokine_consensus_effects_t cytokine_effects;
    inflammation_consensus_state_t inflammation_state;
    consensus_immune_modulation_t modulation;
    nimcp_mutex_t* mutex;
    void* bio_ctx;
    bool bio_async_connected;
};

static float get_quorum_factor_for_level(brain_inflammation_level_t level)
{
    switch (level) {
        case INFLAMMATION_NONE:     return INFLAMMATION_NONE_QUORUM_FACTOR;
        case INFLAMMATION_LOCAL:    return INFLAMMATION_LOCAL_QUORUM_FACTOR;
        case INFLAMMATION_REGIONAL: return INFLAMMATION_REGIONAL_QUORUM_FACTOR;
        case INFLAMMATION_SYSTEMIC: return INFLAMMATION_SYSTEMIC_QUORUM_FACTOR;
        case INFLAMMATION_STORM:    return INFLAMMATION_STORM_QUORUM_FACTOR;
        default:                          return INFLAMMATION_NONE_QUORUM_FACTOR;
    }
}

int swarm_consensus_immune_default_config(swarm_consensus_immune_config_t* config)
{
    if (!config) return -1;

    config->enable_cytokine_effects = true;
    config->enable_inflammation_effects = true;
    config->enable_failure_stress = true;
    config->enable_success_boost = true;
    config->cytokine_sensitivity = 1.0f;

    return 0;
}

swarm_consensus_immune_bridge_t* swarm_consensus_immune_bridge_create(
    const swarm_consensus_immune_config_t* config,
    brain_immune_system_t* immune_system,
    void* swarm_consensus)
{
    if (!config || !immune_system) {
        NIMCP_LOGGING_ERROR("Invalid parameters for swarm consensus immune bridge creation");
        return NULL;
    }

    swarm_consensus_immune_bridge_t* bridge =
        (swarm_consensus_immune_bridge_t*)nimcp_malloc(sizeof(swarm_consensus_immune_bridge_t));
    if (!bridge) {
        NIMCP_LOGGING_ERROR("Failed to allocate swarm consensus immune bridge");
        return NULL;
    }

    memset(bridge, 0, sizeof(swarm_consensus_immune_bridge_t));
    memcpy(&bridge->config, config, sizeof(swarm_consensus_immune_config_t));
    bridge->immune_system = immune_system;
    bridge->swarm_consensus = swarm_consensus;

    bridge->mutex = nimcp_platform_mutex_create();
    if (!bridge->mutex) {
        NIMCP_LOGGING_ERROR("Failed to create mutex for swarm consensus immune bridge");
        nimcp_free(bridge);
        return NULL;
    }

    bridge->cytokine_effects.voting_delay_factor = 1.0f;
    bridge->cytokine_effects.quorum_increase_factor = 1.0f;
    bridge->cytokine_effects.agreement_threshold_factor = 1.0f;
    bridge->cytokine_effects.timeout_multiplier = 1.0f;

    bridge->inflammation_state.current_level = INFLAMMATION_NONE;
    bridge->inflammation_state.quorum_factor = 1.0f;
    bridge->inflammation_state.decision_impairment = 0.0f;
    bridge->inflammation_state.consensus_blocked = false;

    bridge->modulation.vote_success_rate = 1.0f;
    bridge->modulation.failed_votes = 0;
    bridge->modulation.stress_triggered = false;
    bridge->modulation.il10_from_success = 0.0f;

    NIMCP_LOGGING_INFO("Swarm consensus immune bridge created");
    return bridge;
}

void swarm_consensus_immune_bridge_destroy(swarm_consensus_immune_bridge_t* bridge)
{
    if (!bridge) return;

    if (bridge->mutex) {
        nimcp_platform_mutex_destroy(bridge->mutex);
    }

    nimcp_free(bridge);
    NIMCP_LOGGING_INFO("Swarm consensus immune bridge destroyed");
}

int swarm_consensus_immune_apply_cytokine_effects(swarm_consensus_immune_bridge_t* bridge)
{
    if (!bridge || !bridge->immune_system) return -1;
    if (!bridge->config.enable_cytokine_effects) return 0;

    nimcp_mutex_lock(bridge->mutex);

    float sensitivity = bridge->config.cytokine_sensitivity;

    float delay = (CYTOKINE_IL1_VOTING_DELAY - 1.0f) * sensitivity +
                  (CYTOKINE_IL6_VOTING_DELAY - 1.0f) * sensitivity +
                  (CYTOKINE_TNF_VOTING_DELAY - 1.0f) * sensitivity +
                  (CYTOKINE_IL10_VOTING_BOOST - 1.0f) * sensitivity;

    bridge->cytokine_effects.voting_delay_factor = 1.0f + delay;
    bridge->cytokine_effects.quorum_increase_factor = bridge->cytokine_effects.voting_delay_factor;
    bridge->cytokine_effects.agreement_threshold_factor = 1.0f + (delay * 0.5f);
    bridge->cytokine_effects.timeout_multiplier = bridge->cytokine_effects.voting_delay_factor;

    if (bridge->cytokine_effects.voting_delay_factor < 0.5f) {
        bridge->cytokine_effects.voting_delay_factor = 0.5f;
    }

    nimcp_mutex_unlock(bridge->mutex);

    NIMCP_LOGGING_DEBUG("Applied cytokine consensus effects: delay=%.2f",
                        bridge->cytokine_effects.voting_delay_factor);
    return 0;
}

int swarm_consensus_immune_apply_inflammation_effects(swarm_consensus_immune_bridge_t* bridge)
{
    if (!bridge || !bridge->immune_system) return -1;
    if (!bridge->config.enable_inflammation_effects) return 0;

    nimcp_mutex_lock(bridge->mutex);

    brain_immune_stats_t stats;
    if (brain_immune_get_stats(bridge->immune_system, &stats) != 0) {
        nimcp_mutex_unlock(bridge->mutex);
        return -1;
    }
    brain_inflammation_level_t level = stats.inflammation_level;
    bridge->inflammation_state.current_level = level;
    bridge->inflammation_state.quorum_factor = get_quorum_factor_for_level(level);
    bridge->inflammation_state.decision_impairment =
        bridge->inflammation_state.quorum_factor - 1.0f;
    bridge->inflammation_state.consensus_blocked =
        (level >= INFLAMMATION_STORM);

    nimcp_mutex_unlock(bridge->mutex);

    NIMCP_LOGGING_DEBUG("Applied inflammation consensus effects: level=%d, quorum=%.2f",
                        level, bridge->inflammation_state.quorum_factor);
    return 0;
}

int swarm_consensus_immune_trigger_from_failure(swarm_consensus_immune_bridge_t* bridge)
{
    if (!bridge || !bridge->immune_system) return -1;
    if (!bridge->config.enable_failure_stress) return 0;

    nimcp_mutex_lock(bridge->mutex);

    bridge->modulation.failed_votes++;
    if (bridge->modulation.failed_votes > 3) {
        bridge->modulation.stress_triggered = true;
        NIMCP_LOGGING_INFO("Consensus failures triggered stress response: count=%d",
                          bridge->modulation.failed_votes);
    }

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int swarm_consensus_immune_boost_from_success(swarm_consensus_immune_bridge_t* bridge)
{
    if (!bridge || !bridge->immune_system) return -1;
    if (!bridge->config.enable_success_boost) return 0;

    nimcp_mutex_lock(bridge->mutex);

    bridge->modulation.il10_from_success += 0.05f;
    bridge->modulation.failed_votes = 0;
    bridge->modulation.stress_triggered = false;

    NIMCP_LOGGING_DEBUG("Consensus success boosted IL-10: total=%.2f",
                       bridge->modulation.il10_from_success);

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int swarm_consensus_immune_bridge_update(swarm_consensus_immune_bridge_t* bridge, uint64_t delta_ms)
{
    if (!bridge) return -1;
    (void)delta_ms;

    swarm_consensus_immune_apply_cytokine_effects(bridge);
    swarm_consensus_immune_apply_inflammation_effects(bridge);

    return 0;
}

float swarm_consensus_immune_get_voting_delay(const swarm_consensus_immune_bridge_t* bridge)
{
    if (!bridge) return 1.0f;

    nimcp_mutex_lock(bridge->mutex);
    float delay = bridge->cytokine_effects.voting_delay_factor;
    nimcp_mutex_unlock(bridge->mutex);

    return delay;
}

float swarm_consensus_immune_get_quorum_factor(const swarm_consensus_immune_bridge_t* bridge)
{
    if (!bridge) return 1.0f;

    nimcp_mutex_lock(bridge->mutex);
    float factor = bridge->inflammation_state.quorum_factor;
    nimcp_mutex_unlock(bridge->mutex);

    return factor;
}

bool swarm_consensus_immune_is_blocked(const swarm_consensus_immune_bridge_t* bridge)
{
    if (!bridge) return false;

    nimcp_mutex_lock(bridge->mutex);
    bool blocked = bridge->inflammation_state.consensus_blocked;
    nimcp_mutex_unlock(bridge->mutex);

    return blocked;
}

int swarm_consensus_immune_connect_bio_async(swarm_consensus_immune_bridge_t* bridge)
{
    if (!bridge) {
        NIMCP_LOGGING_ERROR("Cannot connect to bio-async: NULL bridge");
        return -1;
    }

    if (bridge->bio_async_connected) {
        NIMCP_LOGGING_INFO("Swarm consensus-immune bridge already connected to bio-async");
        return 0;
    }

    bio_module_info_t info = {
        .module_id = BIO_MODULE_IMMUNE_SWARM_CONSENSUS,
        .module_name = "swarm_consensus_immune_bridge",
        .inbox_capacity = 32,
        .user_data = bridge
    };

    bridge->bio_ctx = bio_router_register_module(&info);
    if (bridge->bio_ctx) {
        bridge->bio_async_connected = true;
        NIMCP_LOGGING_INFO("Swarm consensus-immune bridge connected to bio-async router");
    } else {
        NIMCP_LOGGING_INFO("Bio-async router not available, skipping registration");
    }

    return 0;
}

int swarm_consensus_immune_disconnect_bio_async(swarm_consensus_immune_bridge_t* bridge)
{
    if (!bridge) {
        NIMCP_LOGGING_ERROR("Cannot disconnect from bio-async: NULL bridge");
        return -1;
    }

    if (!bridge->bio_async_connected) {
        return 0;
    }

    if (bridge->bio_ctx) {
        bio_router_unregister_module(bridge->bio_ctx);
        bridge->bio_ctx = NULL;
    }

    bridge->bio_async_connected = false;
    NIMCP_LOGGING_INFO("Swarm consensus-immune bridge disconnected from bio-async router");

    return 0;
}

bool swarm_consensus_immune_is_bio_async_connected(const swarm_consensus_immune_bridge_t* bridge)
{
    if (!bridge) return false;
    return bridge->bio_async_connected;
}
