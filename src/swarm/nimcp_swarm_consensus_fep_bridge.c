/**
 * @file nimcp_swarm_consensus_fep_bridge.c
 */

#include "swarm/nimcp_swarm_consensus_fep_bridge.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/error/nimcp_error_codes.h"
#include "utils/platform/nimcp_platform_time.h"
#include "utils/validation/nimcp_common.h"
#include "api/nimcp_api_exception.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>
#include <math.h>

void swarm_consensus_fep_default_config(swarm_consensus_fep_config_t* config) {
    if (!config) return;
    config->confidence_precision_weight = 0.85f;
    config->agreement_fe_threshold = 0.4f;
    config->quorum_certainty_gain = 1.3f;
    config->enable_bayesian_voting = true;
}

swarm_consensus_fep_bridge_t* swarm_consensus_fep_create(const swarm_consensus_fep_config_t* config, swarm_consensus_t consensus_ctx, fep_system_t* fep_system) {
    if (!consensus_ctx || !fep_system) return NULL;
    swarm_consensus_fep_bridge_t* bridge = (swarm_consensus_fep_bridge_t*)nimcp_malloc(sizeof(swarm_consensus_fep_bridge_t));
    if (!bridge) return NULL;
    memset(bridge, 0, sizeof(swarm_consensus_fep_bridge_t));
    if (config) bridge->config = *config;
    else swarm_consensus_fep_default_config(&bridge->config);
    bridge->fep_system = fep_system;
    bridge->consensus_ctx = consensus_ctx;
    bridge->base.mutex = nimcp_platform_mutex_create();
    if (!bridge->base.mutex) { nimcp_free(bridge); return NULL; }
    return bridge;
}

void swarm_consensus_fep_destroy(swarm_consensus_fep_bridge_t* bridge) {
    if (!bridge) return;
    if (bridge->base.bio_async_enabled) swarm_consensus_fep_disconnect_bio_async(bridge);
    if (bridge->base.mutex) nimcp_platform_mutex_destroy(bridge->base.mutex);
    nimcp_free(bridge);
}

int swarm_consensus_fep_update(swarm_consensus_fep_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    nimcp_platform_mutex_lock(bridge->base.mutex);
    float fe = fep_get_free_energy(bridge->fep_system);
    // Compute precision as inverse of free energy (high FE = low precision)
    float precision = 1.0f / (1.0f + fe);
    bridge->fep_effects.vote_confidence_boost = precision * bridge->config.confidence_precision_weight;
    bridge->fep_effects.quorum_threshold_adjustment = (fe < bridge->config.agreement_fe_threshold) ? -0.1f : 0.1f;
    bridge->fep_effects.agreement_bias = fmaxf(0.3f, 1.0f - fe * 0.3f);
    float consensus_strength = 0.7f;
    bridge->consensus_effects.precision_from_consensus = 0.5f + consensus_strength * 1.0f;
    bridge->consensus_effects.belief_strength_from_quorum = consensus_strength * bridge->config.quorum_certainty_gain;
    bridge->consensus_effects.consensus_state = (consensus_strength > 0.66f) ? 1 : 0;
    bridge->state.last_consensus_fe = fe;
    bridge->state.last_update_time = nimcp_platform_time_monotonic_ms();
    bridge->stats.total_updates++;
    bridge->stats.avg_consensus_fe = (bridge->stats.avg_consensus_fe * (bridge->stats.total_updates - 1) + fe) / bridge->stats.total_updates;
    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return 0;
}

int swarm_consensus_fep_apply_modulation(swarm_consensus_fep_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    return 0;
}

int swarm_consensus_fep_get_effects(const swarm_consensus_fep_bridge_t* bridge, swarm_consensus_fep_effects_t* effects) {
    if (!bridge || !effects) return NIMCP_ERROR_NULL_POINTER;
    nimcp_platform_mutex_lock(bridge->base.mutex);
    *effects = bridge->fep_effects;
    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return 0;
}

int swarm_consensus_fep_get_consensus_effects(const swarm_consensus_fep_bridge_t* bridge, fep_swarm_consensus_effects_t* effects) {
    if (!bridge || !effects) return NIMCP_ERROR_NULL_POINTER;
    nimcp_platform_mutex_lock(bridge->base.mutex);
    *effects = bridge->consensus_effects;
    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return 0;
}

int swarm_consensus_fep_get_stats(const swarm_consensus_fep_bridge_t* bridge, swarm_consensus_fep_stats_t* stats) {
    if (!bridge || !stats) return NIMCP_ERROR_NULL_POINTER;
    nimcp_platform_mutex_lock(bridge->base.mutex);
    *stats = bridge->stats;
    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return 0;
}

int swarm_consensus_fep_connect_bio_async(swarm_consensus_fep_bridge_t* bridge) {
    if (!bridge || bridge->base.bio_async_enabled) return 0;
    bio_module_info_t info = { .module_id = BIO_MODULE_FEP_SWARM_CONSENSUS, .module_name = "swarm_consensus_fep_bridge", .inbox_capacity = NIMCP_INBOX_CAPACITY_SMALL, .user_data = bridge };
    bridge->base.bio_ctx = bio_router_register_module(&info);
    if (bridge->base.bio_ctx) bridge->base.bio_async_enabled = true;
    return 0;
}

int swarm_consensus_fep_disconnect_bio_async(swarm_consensus_fep_bridge_t* bridge) {
    if (!bridge || !bridge->base.bio_async_enabled) return 0;
    if (bridge->base.bio_ctx) bio_router_unregister_module(bridge->base.bio_ctx);
    bridge->base.bio_ctx = NULL;
    bridge->base.bio_async_enabled = false;
    return 0;
}

bool swarm_consensus_fep_is_bio_async_connected(const swarm_consensus_fep_bridge_t* bridge) {
    return bridge && bridge->base.bio_async_enabled;
}
