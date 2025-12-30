/**
 * @file nimcp_remorse_thalamic_bridge.c
 * @brief Remorse-Thalamic Bridge Implementation
 */

#include "cognitive/remorse/nimcp_remorse_thalamic_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include <string.h>

struct remorse_thalamic_bridge {
    void* remorse;
    thalamic_router_t* router;
    remorse_thalamic_config_t config;
    remorse_thalamic_stats_t stats;
    float attention_weight;
};

remorse_thalamic_config_t remorse_thalamic_default_config(void) {
    remorse_thalamic_config_t cfg = {
        .enable_attention_gating = true,
        .enable_repair_routing = true,
        .min_guilt_threshold = 0.3f,
        .repair_threshold = 0.5f
    };
    return cfg;
}

remorse_thalamic_bridge_t* remorse_thalamic_bridge_create(void* remorse, thalamic_router_t* router, const remorse_thalamic_config_t* config) {
    remorse_thalamic_bridge_t* bridge = nimcp_calloc(1, sizeof(remorse_thalamic_bridge_t));
    if (!bridge) return NULL;
    bridge->remorse = remorse;
    bridge->router = router;
    bridge->config = config ? *config : remorse_thalamic_default_config();
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return bridge;
}

void remorse_thalamic_bridge_destroy(remorse_thalamic_bridge_t* bridge) {
    if (bridge) nimcp_free(bridge);
}

int remorse_thalamic_bridge_reset(remorse_thalamic_bridge_t* bridge) {
    if (!bridge) return -1;
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return 0;
}

int remorse_thalamic_route_guilt(remorse_thalamic_bridge_t* bridge, const remorse_thalamic_signal_t* signal) {
    if (!bridge || !signal) return -1;
    if (bridge->config.enable_attention_gating && signal->guilt_intensity < bridge->config.min_guilt_threshold) {
        return 0;
    }
    bridge->stats.guilt_signals_routed++;
    bridge->stats.avg_guilt_intensity = (bridge->stats.avg_guilt_intensity * (bridge->stats.guilt_signals_routed - 1) +
                                         signal->guilt_intensity) / bridge->stats.guilt_signals_routed;
    if (signal->signal_type == REMORSE_SIGNAL_FORGIVENESS) {
        bridge->stats.forgiveness_achieved++;
    }
    return 0;
}

int remorse_thalamic_route_repair(remorse_thalamic_bridge_t* bridge, const void* action, float motivation) {
    if (!bridge) return -1;
    if (bridge->config.enable_repair_routing && motivation >= bridge->config.repair_threshold) {
        bridge->stats.repair_actions++;
    }
    return 0;
}

int remorse_thalamic_set_attention(remorse_thalamic_bridge_t* bridge, float attention) {
    if (!bridge) return -1;
    bridge->attention_weight = attention < 0.0f ? 0.0f : (attention > 1.0f ? 1.0f : attention);
    return 0;
}

int remorse_thalamic_get_attention(const remorse_thalamic_bridge_t* bridge, float* attention) {
    if (!bridge || !attention) return -1;
    *attention = bridge->attention_weight;
    return 0;
}

int remorse_thalamic_bridge_get_stats(const remorse_thalamic_bridge_t* bridge, remorse_thalamic_stats_t* stats) {
    if (!bridge || !stats) return -1;
    *stats = bridge->stats;
    return 0;
}
