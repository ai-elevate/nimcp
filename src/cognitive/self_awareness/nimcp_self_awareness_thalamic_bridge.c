/**
 * @file nimcp_self_awareness_thalamic_bridge.c
 * @brief Self-Awareness-Thalamic Bridge Implementation
 */

#include "cognitive/self_awareness/nimcp_self_awareness_thalamic_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include <string.h>

struct self_awareness_thalamic_bridge {
    void* self_awareness;
    thalamic_router_t* router;
    self_awareness_thalamic_config_t config;
    self_awareness_thalamic_stats_t stats;
    float attention_weight;
};

self_awareness_thalamic_config_t self_awareness_thalamic_default_config(void) {
    self_awareness_thalamic_config_t cfg = {
        .enable_attention_gating = true,
        .enable_metacognitive_boost = true,
        .min_introspection_depth = 0.3f,
        .identity_threshold = 0.5f
    };
    return cfg;
}

self_awareness_thalamic_bridge_t* self_awareness_thalamic_bridge_create(void* self_awareness, thalamic_router_t* router, const self_awareness_thalamic_config_t* config) {
    self_awareness_thalamic_bridge_t* bridge = nimcp_calloc(1, sizeof(self_awareness_thalamic_bridge_t));
    if (!bridge) return NULL;
    bridge->self_awareness = self_awareness;
    bridge->router = router;
    bridge->config = config ? *config : self_awareness_thalamic_default_config();
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return bridge;
}

void self_awareness_thalamic_bridge_destroy(self_awareness_thalamic_bridge_t* bridge) {
    if (bridge) nimcp_free(bridge);
}

int self_awareness_thalamic_bridge_reset(self_awareness_thalamic_bridge_t* bridge) {
    if (!bridge) return -1;
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return 0;
}

int self_awareness_thalamic_route_introspection(self_awareness_thalamic_bridge_t* bridge, const self_awareness_thalamic_signal_t* signal) {
    if (!bridge || !signal) return -1;
    if (bridge->config.enable_attention_gating && signal->introspection_depth < bridge->config.min_introspection_depth) {
        return 0;
    }
    bridge->stats.introspections_routed++;
    bridge->stats.avg_introspection_depth = (bridge->stats.avg_introspection_depth * (bridge->stats.introspections_routed - 1) +
                                             signal->introspection_depth) / bridge->stats.introspections_routed;
    if (signal->signal_type == SELF_SIGNAL_IDENTITY) {
        bridge->stats.identity_updates++;
    }
    return 0;
}

int self_awareness_thalamic_route_metacognition(self_awareness_thalamic_bridge_t* bridge, const void* metacog, float accuracy) {
    if (!bridge) return -1;
    bridge->stats.metacognitions_routed++;
    return 0;
}

int self_awareness_thalamic_set_attention(self_awareness_thalamic_bridge_t* bridge, float attention) {
    if (!bridge) return -1;
    bridge->attention_weight = attention < 0.0f ? 0.0f : (attention > 1.0f ? 1.0f : attention);
    return 0;
}

int self_awareness_thalamic_get_attention(const self_awareness_thalamic_bridge_t* bridge, float* attention) {
    if (!bridge || !attention) return -1;
    *attention = bridge->attention_weight;
    return 0;
}

int self_awareness_thalamic_bridge_get_stats(const self_awareness_thalamic_bridge_t* bridge, self_awareness_thalamic_stats_t* stats) {
    if (!bridge || !stats) return -1;
    *stats = bridge->stats;
    return 0;
}
