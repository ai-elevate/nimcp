/**
 * @file nimcp_free_energy_thalamic_bridge.c
 * @brief Free Energy Principle-Thalamic Bridge Implementation
 */

#include "cognitive/free_energy/nimcp_free_energy_thalamic_bridge.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/memory/nimcp_memory.h"
#include <string.h>

struct free_energy_thalamic_bridge {
    bridge_base_t base;
    void* free_energy;
    thalamic_router_t* router;
    free_energy_thalamic_config_t config;
    free_energy_thalamic_stats_t stats;
    float attention_weight;
};

free_energy_thalamic_config_t free_energy_thalamic_default_config(void) {
    free_energy_thalamic_config_t cfg = {
        .enable_attention_gating = true,
        .enable_precision_boost = true,
        .min_error_threshold = 0.2f,
        .precision_boost = 0.4f
    };
    return cfg;
}

free_energy_thalamic_bridge_t* free_energy_thalamic_bridge_create(void* free_energy, thalamic_router_t* router, const free_energy_thalamic_config_t* config) {
    free_energy_thalamic_bridge_t* bridge = nimcp_calloc(1, sizeof(free_energy_thalamic_bridge_t));
    if (!bridge) return NULL;
    bridge->base.mutex = nimcp_platform_mutex_create();
    if (!bridge->base.mutex) { nimcp_free(bridge); return NULL; }
    bridge->free_energy = free_energy;
    bridge->router = router;
    bridge->config = config ? *config : free_energy_thalamic_default_config();
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return bridge;
}

void free_energy_thalamic_bridge_destroy(free_energy_thalamic_bridge_t* bridge) {
    if (!bridge) return;
    if (bridge->base.mutex) nimcp_platform_mutex_destroy(bridge->base.mutex);
    nimcp_free(bridge);
}

int free_energy_thalamic_bridge_reset(free_energy_thalamic_bridge_t* bridge) {
    if (!bridge) return -1;
    nimcp_platform_mutex_lock(bridge->base.mutex);
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return 0;
}

int free_energy_thalamic_route_prediction_error(free_energy_thalamic_bridge_t* bridge, const free_energy_thalamic_signal_t* signal) {
    if (!bridge || !signal) return -1;
    nimcp_platform_mutex_lock(bridge->base.mutex);
    /* High-precision errors bypass gating */
    if (bridge->config.enable_attention_gating &&
        signal->prediction_error < bridge->config.min_error_threshold &&
        signal->precision < 0.7f) {
        nimcp_platform_mutex_unlock(bridge->base.mutex);
        return 0;
    }
    bridge->stats.errors_routed++;
    bridge->stats.avg_precision = (bridge->stats.avg_precision * (bridge->stats.errors_routed - 1) +
                                   signal->precision) / bridge->stats.errors_routed;
    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return 0;
}

int free_energy_thalamic_route_model_update(free_energy_thalamic_bridge_t* bridge, const void* model, float importance) {
    if (!bridge) return -1;
    nimcp_platform_mutex_lock(bridge->base.mutex);
    bridge->stats.model_updates++;
    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return 0;
}

int free_energy_thalamic_set_attention(free_energy_thalamic_bridge_t* bridge, float attention) {
    if (!bridge) return -1;
    nimcp_platform_mutex_lock(bridge->base.mutex);
    bridge->attention_weight = attention < 0.0f ? 0.0f : (attention > 1.0f ? 1.0f : attention);
    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return 0;
}

int free_energy_thalamic_get_attention(const free_energy_thalamic_bridge_t* bridge, float* attention) {
    if (!bridge || !attention) return -1;
    nimcp_platform_mutex_lock(bridge->base.mutex);
    *attention = bridge->attention_weight;
    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return 0;
}

int free_energy_thalamic_bridge_get_stats(const free_energy_thalamic_bridge_t* bridge, free_energy_thalamic_stats_t* stats) {
    if (!bridge || !stats) return -1;
    nimcp_platform_mutex_lock(bridge->base.mutex);
    *stats = bridge->stats;
    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return 0;
}
