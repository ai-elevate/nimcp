/**
 * @file nimcp_homeostatic_fep_bridge.c
 * @brief FEP-Homeostatic Integration Bridge Implementation
 */

#include "plasticity/homeostatic/nimcp_homeostatic_fep_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/logging/nimcp_logging.h"
#include <math.h>
#include <string.h>

#define LOG_MODULE_HOMEOSTATIC_FEP "HOMEOSTATIC_FEP_BRIDGE"

int homeostatic_fep_bridge_default_config(homeostatic_fep_config_t* config) {
    if (!config) return -1;
    config->precision_target_gain = HOMEOSTATIC_FEP_TARGET_RATE_SCALING;
    config->free_energy_scaling_factor = 1.0f;
    config->stability_threshold = 0.01f;
    config->enable_precision_normalization = true;
    config->enable_fe_modulation = true;
    config->enable_stability_tracking = true;
    return 0;
}

homeostatic_fep_bridge_t* homeostatic_fep_bridge_create(const homeostatic_fep_config_t* config) {
    homeostatic_fep_bridge_t* bridge = (homeostatic_fep_bridge_t*)nimcp_malloc(sizeof(homeostatic_fep_bridge_t));
    if (!bridge) return NULL;
    memset(bridge, 0, sizeof(homeostatic_fep_bridge_t));
    if (config) bridge->config = *config;
    else homeostatic_fep_bridge_default_config(&bridge->config);
    bridge->mutex = nimcp_platform_mutex_create();
    if (!bridge->mutex) { nimcp_free(bridge); return NULL; }
    bridge->effects.total_normalization_factor = 1.0f;
    NIMCP_LOGGING_INFO("Homeostatic-FEP bridge created");
    return bridge;
}

void homeostatic_fep_bridge_destroy(homeostatic_fep_bridge_t* bridge) {
    if (!bridge) return;
    if (bridge->bio_async_enabled) homeostatic_fep_bridge_disconnect_bio_async(bridge);
    if (bridge->mutex) nimcp_platform_mutex_destroy(bridge->mutex);
    nimcp_free(bridge);
}

int homeostatic_fep_bridge_connect_fep(homeostatic_fep_bridge_t* bridge, fep_system_t* fep) {
    if (!bridge || !fep) return -1;
    nimcp_platform_mutex_lock(bridge->mutex);
    bridge->fep_system = fep;
    nimcp_platform_mutex_unlock(bridge->mutex);
    return 0;
}

int homeostatic_fep_bridge_connect_homeostatic(homeostatic_fep_bridge_t* bridge, homeostatic_controller_t homeostatic) {
    if (!bridge || !homeostatic) return -1;
    nimcp_platform_mutex_lock(bridge->mutex);
    bridge->homeostatic_system = homeostatic;
    nimcp_platform_mutex_unlock(bridge->mutex);
    return 0;
}

int homeostatic_fep_bridge_disconnect(homeostatic_fep_bridge_t* bridge) {
    if (!bridge) return -1;
    nimcp_platform_mutex_lock(bridge->mutex);
    bridge->fep_system = NULL;
    bridge->homeostatic_system = NULL;
    nimcp_platform_mutex_unlock(bridge->mutex);
    return 0;
}

float homeostatic_fep_apply_precision_normalization(homeostatic_fep_bridge_t* bridge, float precision) {
    if (!bridge || !bridge->config.enable_precision_normalization) return 1.0f;
    return fminf(fmaxf(precision, HOMEOSTATIC_FEP_PRECISION_MIN), HOMEOSTATIC_FEP_PRECISION_MAX);
}

float homeostatic_fep_apply_fe_modulation(homeostatic_fep_bridge_t* bridge, float free_energy) {
    if (!bridge || !bridge->config.enable_fe_modulation) return 1.0f;
    return 1.0f + free_energy * bridge->config.free_energy_scaling_factor;
}

float homeostatic_fep_get_effective_target_rate(const homeostatic_fep_bridge_t* bridge, float base_rate) {
    if (!bridge) return base_rate;
    return base_rate * bridge->effects.precision_target_rate;
}

int homeostatic_fep_report_scaling(homeostatic_fep_bridge_t* bridge, float scaling_factor) {
    if (!bridge) return -1;
    nimcp_platform_mutex_lock(bridge->mutex);
    bridge->stats.avg_scaling_factor = (bridge->stats.avg_scaling_factor * bridge->stats.total_updates + scaling_factor) /
                                        (bridge->stats.total_updates + 1);
    nimcp_platform_mutex_unlock(bridge->mutex);
    return 0;
}

int homeostatic_fep_bridge_update(homeostatic_fep_bridge_t* bridge, uint64_t delta_ms) {
    if (!bridge) return -1;
    nimcp_platform_mutex_lock(bridge->mutex);
    if (bridge->fep_system) {
        float precision_norm = homeostatic_fep_apply_precision_normalization(bridge, bridge->effects.precision_value);
        float fe_mod = homeostatic_fep_apply_fe_modulation(bridge, bridge->effects.free_energy_value);
        bridge->effects.precision_target_rate = precision_norm;
        bridge->effects.scaling_time_modulation = fe_mod;
        bridge->effects.total_normalization_factor = precision_norm * fe_mod;
    }
    bridge->stats.total_updates++;
    bridge->state.last_update_time = delta_ms;
    nimcp_platform_mutex_unlock(bridge->mutex);
    return 0;
}

int homeostatic_fep_bridge_get_state(const homeostatic_fep_bridge_t* bridge, homeostatic_fep_state_t* state) {
    if (!bridge || !state) return -1;
    *state = bridge->state;
    return 0;
}

int homeostatic_fep_bridge_get_stats(const homeostatic_fep_bridge_t* bridge, homeostatic_fep_stats_t* stats) {
    if (!bridge || !stats) return -1;
    *stats = bridge->stats;
    return 0;
}

int homeostatic_fep_bridge_connect_bio_async(homeostatic_fep_bridge_t* bridge) {
    if (!bridge || bridge->bio_async_enabled) return 0;
    bio_module_info_t info = {
        .module_id = BIO_MODULE_FEP_HOMEOSTATIC_BRIDGE,
        .module_name = "homeostatic_fep_bridge",
        .inbox_capacity = 32,
        .user_data = bridge
    };
    bridge->bio_ctx = bio_router_register_module(&info);
    if (bridge->bio_ctx) {
        bridge->bio_async_enabled = true;
        NIMCP_LOGGING_INFO("Connected to bio-async router");
    }
    return 0;
}

int homeostatic_fep_bridge_disconnect_bio_async(homeostatic_fep_bridge_t* bridge) {
    if (!bridge || !bridge->bio_async_enabled) return -1;
    bio_router_unregister_module(bridge->bio_ctx);
    bridge->bio_ctx = NULL;
    bridge->bio_async_enabled = false;
    return 0;
}

bool homeostatic_fep_bridge_is_bio_async_connected(const homeostatic_fep_bridge_t* bridge) {
    return bridge ? bridge->bio_async_enabled : false;
}
