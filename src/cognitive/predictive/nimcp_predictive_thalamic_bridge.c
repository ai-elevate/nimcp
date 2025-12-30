/**
 * @file nimcp_predictive_thalamic_bridge.c
 * @brief Predictive Coding-Thalamic Bridge Implementation
 */

#include "cognitive/predictive/nimcp_predictive_thalamic_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include <string.h>

struct predictive_thalamic_bridge {
    void* predictive;
    thalamic_router_t* router;
    predictive_thalamic_config_t config;
    predictive_thalamic_stats_t stats;
    float attention_weight;
};

predictive_thalamic_config_t predictive_thalamic_default_config(void) {
    predictive_thalamic_config_t cfg = {
        .enable_attention_gating = true,
        .enable_error_amplification = true,
        .min_error_threshold = 0.1f,
        .precision_threshold = 0.5f
    };
    return cfg;
}

predictive_thalamic_bridge_t* predictive_thalamic_bridge_create(void* predictive, thalamic_router_t* router, const predictive_thalamic_config_t* config) {
    predictive_thalamic_bridge_t* bridge = nimcp_calloc(1, sizeof(predictive_thalamic_bridge_t));
    if (!bridge) return NULL;
    bridge->predictive = predictive;
    bridge->router = router;
    bridge->config = config ? *config : predictive_thalamic_default_config();
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return bridge;
}

void predictive_thalamic_bridge_destroy(predictive_thalamic_bridge_t* bridge) {
    if (bridge) nimcp_free(bridge);
}

int predictive_thalamic_bridge_reset(predictive_thalamic_bridge_t* bridge) {
    if (!bridge) return -1;
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return 0;
}

int predictive_thalamic_route_error(predictive_thalamic_bridge_t* bridge, const predictive_thalamic_signal_t* signal) {
    if (!bridge || !signal) return -1;
    if (bridge->config.enable_attention_gating && signal->error_magnitude < bridge->config.min_error_threshold) {
        return 0;
    }
    bridge->stats.errors_routed++;
    bridge->stats.avg_error_magnitude = (bridge->stats.avg_error_magnitude * (bridge->stats.errors_routed - 1) +
                                         signal->error_magnitude) / bridge->stats.errors_routed;
    if (signal->signal_type == PREDICTIVE_SIGNAL_PREDICTION) {
        bridge->stats.predictions_routed++;
    }
    return 0;
}

int predictive_thalamic_route_update(predictive_thalamic_bridge_t* bridge, const void* update, uint32_t level) {
    if (!bridge) return -1;
    bridge->stats.updates_triggered++;
    return 0;
}

int predictive_thalamic_set_attention(predictive_thalamic_bridge_t* bridge, float attention) {
    if (!bridge) return -1;
    bridge->attention_weight = attention < 0.0f ? 0.0f : (attention > 1.0f ? 1.0f : attention);
    return 0;
}

int predictive_thalamic_get_attention(const predictive_thalamic_bridge_t* bridge, float* attention) {
    if (!bridge || !attention) return -1;
    *attention = bridge->attention_weight;
    return 0;
}

int predictive_thalamic_bridge_get_stats(const predictive_thalamic_bridge_t* bridge, predictive_thalamic_stats_t* stats) {
    if (!bridge || !stats) return -1;
    *stats = bridge->stats;
    return 0;
}
