/**
 * @file nimcp_salience_thalamic_bridge.c
 * @brief Salience-Thalamic Bridge Implementation
 */

#include "cognitive/salience/nimcp_salience_thalamic_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include <string.h>

struct salience_thalamic_bridge {
    void* salience;
    thalamic_router_t* router;
    salience_thalamic_config_t config;
    salience_thalamic_stats_t stats;
    float attention_weight;
};

salience_thalamic_config_t salience_thalamic_default_config(void) {
    salience_thalamic_config_t cfg = {
        .enable_attention_gating = true,
        .enable_priority_override = true,
        .min_salience_threshold = 0.3f,
        .switch_threshold = 0.7f
    };
    return cfg;
}

salience_thalamic_bridge_t* salience_thalamic_bridge_create(void* salience, thalamic_router_t* router, const salience_thalamic_config_t* config) {
    salience_thalamic_bridge_t* bridge = nimcp_calloc(1, sizeof(salience_thalamic_bridge_t));
    if (!bridge) return NULL;
    bridge->salience = salience;
    bridge->router = router;
    bridge->config = config ? *config : salience_thalamic_default_config();
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return bridge;
}

void salience_thalamic_bridge_destroy(salience_thalamic_bridge_t* bridge) {
    if (bridge) nimcp_free(bridge);
}

int salience_thalamic_bridge_reset(salience_thalamic_bridge_t* bridge) {
    if (!bridge) return -1;
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return 0;
}

int salience_thalamic_route_detection(salience_thalamic_bridge_t* bridge, const salience_thalamic_signal_t* signal) {
    if (!bridge || !signal) return -1;
    if (bridge->config.enable_attention_gating && signal->salience_value < bridge->config.min_salience_threshold) {
        return 0;
    }
    bridge->stats.detections_routed++;
    bridge->stats.avg_salience_value = (bridge->stats.avg_salience_value * (bridge->stats.detections_routed - 1) +
                                        signal->salience_value) / bridge->stats.detections_routed;
    if (signal->signal_type == SALIENCE_SIGNAL_SWITCH && signal->salience_value >= bridge->config.switch_threshold) {
        bridge->stats.attention_switches++;
    }
    return 0;
}

int salience_thalamic_route_priority(salience_thalamic_bridge_t* bridge, const void* stimulus, float priority) {
    if (!bridge) return -1;
    if (bridge->config.enable_priority_override && priority >= bridge->config.switch_threshold) {
        bridge->stats.priority_overrides++;
    }
    return 0;
}

int salience_thalamic_set_attention(salience_thalamic_bridge_t* bridge, float attention) {
    if (!bridge) return -1;
    bridge->attention_weight = attention < 0.0f ? 0.0f : (attention > 1.0f ? 1.0f : attention);
    return 0;
}

int salience_thalamic_get_attention(const salience_thalamic_bridge_t* bridge, float* attention) {
    if (!bridge || !attention) return -1;
    *attention = bridge->attention_weight;
    return 0;
}

int salience_thalamic_bridge_get_stats(const salience_thalamic_bridge_t* bridge, salience_thalamic_stats_t* stats) {
    if (!bridge || !stats) return -1;
    *stats = bridge->stats;
    return 0;
}
