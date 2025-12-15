/**
 * @file nimcp_swarm_emergence_fep_bridge.c
 */

#include "swarm/nimcp_swarm_emergence_fep_bridge.h"
#include "utils/error/nimcp_error_codes.h"
#include <string.h>
#include <math.h>

void swarm_emergence_fep_default_config(swarm_emergence_fep_config_t* config) {
    if (!config) return;
    config->tier_fe_threshold = 0.5f;
    config->coherence_precision_gain = 1.2f;
    config->hierarchy_depth_scaling = 1.0f;
    config->enable_emergence_detection = true;
}

swarm_emergence_fep_bridge_t* swarm_emergence_fep_create(const swarm_emergence_fep_config_t* config, swarm_emergence_ctx_t* emergence_ctx, fep_system_t* fep_system) {
    if (!emergence_ctx || !fep_system) return NULL;
    swarm_emergence_fep_bridge_t* bridge = (swarm_emergence_fep_bridge_t*)nimcp_malloc(sizeof(swarm_emergence_fep_bridge_t));
    if (!bridge) return NULL;
    memset(bridge, 0, sizeof(swarm_emergence_fep_bridge_t));
    if (config) bridge->config = *config;
    else swarm_emergence_fep_default_config(&bridge->config);
    bridge->fep_system = fep_system;
    bridge->emergence_ctx = emergence_ctx;
    bridge->mutex = nimcp_platform_mutex_create();
    if (!bridge->mutex) { nimcp_free(bridge); return NULL; }
    return bridge;
}

void swarm_emergence_fep_destroy(swarm_emergence_fep_bridge_t* bridge) {
    if (!bridge) return;
    if (bridge->bio_async_enabled) swarm_emergence_fep_disconnect_bio_async(bridge);
    if (bridge->mutex) nimcp_platform_mutex_destroy(bridge->mutex);
    nimcp_free(bridge);
}

int swarm_emergence_fep_update(swarm_emergence_fep_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    nimcp_platform_mutex_lock(bridge->mutex);
    float fe = fep_get_free_energy(bridge->fep_system);
    swarm_emergence_tier_t tier = swarm_emergence_get_tier(bridge->emergence_ctx);
    float coherence = swarm_emergence_get_coherence(bridge->emergence_ctx);
    bridge->fep_effects.tier_advancement_bias = (fe < bridge->config.tier_fe_threshold) ? 0.2f : -0.1f;
    bridge->fep_effects.coherence_boost = fmaxf(0.0f, 0.3f - fe * 0.2f);
    bridge->fep_effects.capability_activation_threshold = 0.6f + fe * 0.1f;
    bridge->emergence_effects.precision_from_tier = 0.5f + ((float)tier / (float)SWARM_TIER_COUNT) * 1.0f;
    bridge->emergence_effects.hierarchy_depth = (uint32_t)tier + 1;
    bridge->emergence_effects.model_complexity = (float)tier / (float)SWARM_TIER_COUNT;
    if (bridge->state.last_tier != tier) {
        bridge->state.tier_transitions++;
        bridge->state.last_tier = tier;
        bridge->stats.emergence_events++;
    }
    bridge->state.last_coherence = coherence;
    bridge->state.last_update_time = nimcp_platform_get_time_ns();
    bridge->stats.total_updates++;
    bridge->stats.avg_tier_fe = (bridge->stats.avg_tier_fe * (bridge->stats.total_updates - 1) + fe) / bridge->stats.total_updates;
    nimcp_platform_mutex_unlock(bridge->mutex);
    return 0;
}

int swarm_emergence_fep_apply_modulation(swarm_emergence_fep_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    return 0;
}

int swarm_emergence_fep_get_effects(const swarm_emergence_fep_bridge_t* bridge, swarm_emergence_fep_effects_t* effects) {
    if (!bridge || !effects) return NIMCP_ERROR_NULL_POINTER;
    nimcp_platform_mutex_lock(bridge->mutex);
    *effects = bridge->fep_effects;
    nimcp_platform_mutex_unlock(bridge->mutex);
    return 0;
}

int swarm_emergence_fep_get_emergence_effects(const swarm_emergence_fep_bridge_t* bridge, fep_swarm_emergence_effects_t* effects) {
    if (!bridge || !effects) return NIMCP_ERROR_NULL_POINTER;
    nimcp_platform_mutex_lock(bridge->mutex);
    *effects = bridge->emergence_effects;
    nimcp_platform_mutex_unlock(bridge->mutex);
    return 0;
}

int swarm_emergence_fep_get_stats(const swarm_emergence_fep_bridge_t* bridge, swarm_emergence_fep_stats_t* stats) {
    if (!bridge || !stats) return NIMCP_ERROR_NULL_POINTER;
    nimcp_platform_mutex_lock(bridge->mutex);
    *stats = bridge->stats;
    nimcp_platform_mutex_unlock(bridge->mutex);
    return 0;
}

int swarm_emergence_fep_connect_bio_async(swarm_emergence_fep_bridge_t* bridge) {
    if (!bridge || bridge->bio_async_enabled) return 0;
    bio_module_info_t info = { .module_id = BIO_MODULE_FEP_SWARM_EMERGENCE, .module_name = "swarm_emergence_fep_bridge", .inbox_capacity = 32, .user_data = bridge };
    bridge->bio_ctx = bio_router_register_module(&info);
    if (bridge->bio_ctx) bridge->bio_async_enabled = true;
    return 0;
}

int swarm_emergence_fep_disconnect_bio_async(swarm_emergence_fep_bridge_t* bridge) {
    if (!bridge || !bridge->bio_async_enabled) return 0;
    if (bridge->bio_ctx) bio_router_unregister_module(bridge->bio_ctx);
    bridge->bio_ctx = NULL;
    bridge->bio_async_enabled = false;
    return 0;
}

bool swarm_emergence_fep_is_bio_async_connected(const swarm_emergence_fep_bridge_t* bridge) {
    return bridge && bridge->bio_async_enabled;
}
