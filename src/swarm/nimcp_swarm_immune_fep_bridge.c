/**
 * @file nimcp_swarm_immune_fep_bridge.c
 */

#include "swarm/nimcp_swarm_immune_fep_bridge.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/error/nimcp_error_codes.h"
#include "utils/platform/nimcp_platform_time.h"
#include <string.h>
#include <math.h>

void swarm_immune_fep_default_config(swarm_immune_fep_config_t* config) {
    if (!config) return;
    config->threat_pe_weight = 0.9f;
    config->response_action_gain = 1.3f;
    config->inflammation_precision_mod = 0.7f;
    config->enable_adaptive_response = true;
}

swarm_immune_fep_bridge_t* swarm_immune_fep_create(const swarm_immune_fep_config_t* config, void* immune_system, fep_system_t* fep_system) {
    if (!immune_system || !fep_system) return NULL;
    swarm_immune_fep_bridge_t* bridge = (swarm_immune_fep_bridge_t*)nimcp_malloc(sizeof(swarm_immune_fep_bridge_t));
    if (!bridge) return NULL;
    memset(bridge, 0, sizeof(swarm_immune_fep_bridge_t));
    if (config) bridge->config = *config;
    else swarm_immune_fep_default_config(&bridge->config);
    bridge->fep_system = fep_system;
    bridge->immune_system = immune_system;
    bridge->base.mutex = nimcp_platform_mutex_create();
    if (!bridge->base.mutex) { nimcp_free(bridge); return NULL; }
    return bridge;
}

void swarm_immune_fep_destroy(swarm_immune_fep_bridge_t* bridge) {
    if (!bridge) return;
    if (bridge->base.bio_async_enabled) swarm_immune_fep_disconnect_bio_async(bridge);
    if (bridge->base.mutex) nimcp_platform_mutex_destroy(bridge->base.mutex);
    nimcp_free(bridge);
}

int swarm_immune_fep_update(swarm_immune_fep_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    nimcp_platform_mutex_lock(bridge->base.mutex);
    float fe = fep_get_free_energy(bridge->fep_system);
    bridge->fep_effects.threat_sensitivity = 0.5f + fe * 0.5f;
    bridge->fep_effects.response_urgency = fminf(1.0f, fe * bridge->config.response_action_gain);
    bridge->fep_effects.inflammation_level = fe * bridge->config.inflammation_precision_mod;
    bridge->immune_effects.precision_from_threat = 0.3f + bridge->fep_effects.threat_sensitivity * 0.7f;
    bridge->immune_effects.action_bias_from_immune = bridge->fep_effects.response_urgency;
    bridge->immune_effects.learning_suppression = bridge->fep_effects.inflammation_level;
    bridge->state.last_threat_fe = fe;
    bridge->state.last_update_time = nimcp_platform_time_monotonic_ms();
    bridge->stats.total_updates++;
    bridge->stats.avg_threat_fe = (bridge->stats.avg_threat_fe * (bridge->stats.total_updates - 1) + fe) / bridge->stats.total_updates;
    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return 0;
}

int swarm_immune_fep_apply_modulation(swarm_immune_fep_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    return 0;
}

int swarm_immune_fep_get_effects(const swarm_immune_fep_bridge_t* bridge, swarm_immune_fep_effects_t* effects) {
    if (!bridge || !effects) return NIMCP_ERROR_NULL_POINTER;
    nimcp_platform_mutex_lock(bridge->base.mutex);
    *effects = bridge->fep_effects;
    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return 0;
}

int swarm_immune_fep_get_immune_effects(const swarm_immune_fep_bridge_t* bridge, fep_swarm_immune_effects_t* effects) {
    if (!bridge || !effects) return NIMCP_ERROR_NULL_POINTER;
    nimcp_platform_mutex_lock(bridge->base.mutex);
    *effects = bridge->immune_effects;
    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return 0;
}

int swarm_immune_fep_get_stats(const swarm_immune_fep_bridge_t* bridge, swarm_immune_fep_stats_t* stats) {
    if (!bridge || !stats) return NIMCP_ERROR_NULL_POINTER;
    nimcp_platform_mutex_lock(bridge->base.mutex);
    *stats = bridge->stats;
    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return 0;
}

int swarm_immune_fep_connect_bio_async(swarm_immune_fep_bridge_t* bridge) {
    if (!bridge || bridge->base.bio_async_enabled) return 0;
    bio_module_info_t info = { .module_id = BIO_MODULE_FEP_SWARM_IMMUNE, .module_name = "swarm_immune_fep_bridge", .inbox_capacity = 32, .user_data = bridge };
    bridge->base.bio_ctx = bio_router_register_module(&info);
    if (bridge->base.bio_ctx) bridge->base.bio_async_enabled = true;
    return 0;
}

int swarm_immune_fep_disconnect_bio_async(swarm_immune_fep_bridge_t* bridge) {
    if (!bridge || !bridge->base.bio_async_enabled) return 0;
    if (bridge->base.bio_ctx) bio_router_unregister_module(bridge->base.bio_ctx);
    bridge->base.bio_ctx = NULL;
    bridge->base.bio_async_enabled = false;
    return 0;
}

bool swarm_immune_fep_is_bio_async_connected(const swarm_immune_fep_bridge_t* bridge) {
    return bridge && bridge->base.bio_async_enabled;
}
