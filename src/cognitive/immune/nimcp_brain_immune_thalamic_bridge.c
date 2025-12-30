/**
 * @file nimcp_brain_immune_thalamic_bridge.c
 * @brief Brain Immune-Thalamic Bridge Implementation
 */

#include "cognitive/immune/nimcp_brain_immune_thalamic_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include <string.h>

struct brain_immune_thalamic_bridge {
    void* brain_immune;
    thalamic_router_t* router;
    brain_immune_thalamic_config_t config;
    brain_immune_thalamic_stats_t stats;
    float attention_weight;
};

brain_immune_thalamic_config_t brain_immune_thalamic_default_config(void) {
    brain_immune_thalamic_config_t cfg = {
        .enable_attention_gating = true,
        .enable_severity_boost = true,
        .min_threat_threshold = 0.2f,
        .inflammation_boost = 0.4f
    };
    return cfg;
}

brain_immune_thalamic_bridge_t* brain_immune_thalamic_bridge_create(void* brain_immune, thalamic_router_t* router, const brain_immune_thalamic_config_t* config) {
    brain_immune_thalamic_bridge_t* bridge = nimcp_calloc(1, sizeof(brain_immune_thalamic_bridge_t));
    if (!bridge) return NULL;
    bridge->brain_immune = brain_immune;
    bridge->router = router;
    bridge->config = config ? *config : brain_immune_thalamic_default_config();
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return bridge;
}

void brain_immune_thalamic_bridge_destroy(brain_immune_thalamic_bridge_t* bridge) {
    if (bridge) nimcp_free(bridge);
}

int brain_immune_thalamic_bridge_reset(brain_immune_thalamic_bridge_t* bridge) {
    if (!bridge) return -1;
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return 0;
}

int brain_immune_thalamic_route_threat(brain_immune_thalamic_bridge_t* bridge, const brain_immune_thalamic_signal_t* signal) {
    if (!bridge || !signal) return -1;
    /* High-severity threats bypass attention gating */
    if (bridge->config.enable_attention_gating &&
        signal->threat_severity < bridge->config.min_threat_threshold &&
        signal->inflammation_level < 0.6f) {
        return 0;
    }
    bridge->stats.threats_routed++;
    bridge->stats.avg_threat_severity = (bridge->stats.avg_threat_severity * (bridge->stats.threats_routed - 1) +
                                         signal->threat_severity) / bridge->stats.threats_routed;
    return 0;
}

int brain_immune_thalamic_route_response(brain_immune_thalamic_bridge_t* bridge, const void* response, float intensity) {
    if (!bridge) return -1;
    bridge->stats.responses_triggered++;
    return 0;
}

int brain_immune_thalamic_set_attention(brain_immune_thalamic_bridge_t* bridge, float attention) {
    if (!bridge) return -1;
    bridge->attention_weight = attention < 0.0f ? 0.0f : (attention > 1.0f ? 1.0f : attention);
    return 0;
}

int brain_immune_thalamic_get_attention(const brain_immune_thalamic_bridge_t* bridge, float* attention) {
    if (!bridge || !attention) return -1;
    *attention = bridge->attention_weight;
    return 0;
}

int brain_immune_thalamic_bridge_get_stats(const brain_immune_thalamic_bridge_t* bridge, brain_immune_thalamic_stats_t* stats) {
    if (!bridge || !stats) return -1;
    *stats = bridge->stats;
    return 0;
}
