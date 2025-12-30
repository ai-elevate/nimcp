/**
 * @file nimcp_jepa_thalamic_bridge.c
 * @brief JEPA-Thalamic Bridge Implementation
 */

#include "cognitive/jepa/nimcp_jepa_thalamic_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include <string.h>

struct jepa_thalamic_bridge {
    void* jepa;
    thalamic_router_t* router;
    jepa_thalamic_config_t config;
    jepa_thalamic_stats_t stats;
    float attention_weight;
};

jepa_thalamic_config_t jepa_thalamic_default_config(void) {
    jepa_thalamic_config_t cfg = {
        .enable_attention_gating = true,
        .enable_error_amplification = true,
        .min_prediction_confidence = 0.3f,
        .max_temporal_horizon = 100
    };
    return cfg;
}

jepa_thalamic_bridge_t* jepa_thalamic_bridge_create(void* jepa, thalamic_router_t* router, const jepa_thalamic_config_t* config) {
    jepa_thalamic_bridge_t* bridge = nimcp_calloc(1, sizeof(jepa_thalamic_bridge_t));
    if (!bridge) return NULL;
    bridge->jepa = jepa;
    bridge->router = router;
    bridge->config = config ? *config : jepa_thalamic_default_config();
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return bridge;
}

void jepa_thalamic_bridge_destroy(jepa_thalamic_bridge_t* bridge) {
    if (bridge) nimcp_free(bridge);
}

int jepa_thalamic_bridge_reset(jepa_thalamic_bridge_t* bridge) {
    if (!bridge) return -1;
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return 0;
}

int jepa_thalamic_route_prediction(jepa_thalamic_bridge_t* bridge, const jepa_thalamic_signal_t* signal) {
    if (!bridge || !signal) return -1;
    if (bridge->config.enable_attention_gating && signal->prediction_confidence < bridge->config.min_prediction_confidence) {
        return 0;
    }
    if (signal->temporal_horizon > bridge->config.max_temporal_horizon) {
        return 0;
    }
    bridge->stats.predictions_routed++;
    bridge->stats.avg_prediction_confidence = (bridge->stats.avg_prediction_confidence * (bridge->stats.predictions_routed - 1) +
                                               signal->prediction_confidence) / bridge->stats.predictions_routed;
    if (signal->signal_type == JEPA_SIGNAL_EMBEDDING) {
        bridge->stats.embeddings_updated++;
    }
    return 0;
}

int jepa_thalamic_route_error(jepa_thalamic_bridge_t* bridge, const void* error, float magnitude) {
    if (!bridge) return -1;
    bridge->stats.errors_propagated++;
    return 0;
}

int jepa_thalamic_set_attention(jepa_thalamic_bridge_t* bridge, float attention) {
    if (!bridge) return -1;
    bridge->attention_weight = attention < 0.0f ? 0.0f : (attention > 1.0f ? 1.0f : attention);
    return 0;
}

int jepa_thalamic_get_attention(const jepa_thalamic_bridge_t* bridge, float* attention) {
    if (!bridge || !attention) return -1;
    *attention = bridge->attention_weight;
    return 0;
}

int jepa_thalamic_bridge_get_stats(const jepa_thalamic_bridge_t* bridge, jepa_thalamic_stats_t* stats) {
    if (!bridge || !stats) return -1;
    *stats = bridge->stats;
    return 0;
}
