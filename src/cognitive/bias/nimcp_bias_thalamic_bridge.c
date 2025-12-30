/**
 * @file nimcp_bias_thalamic_bridge.c
 * @brief Cognitive Bias-Thalamic Bridge Implementation
 */

#include "cognitive/bias/nimcp_bias_thalamic_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include <string.h>

struct bias_thalamic_bridge {
    void* bias;
    thalamic_router_t* router;
    bias_thalamic_config_t config;
    bias_thalamic_stats_t stats;
    float attention_weight;
};

bias_thalamic_config_t bias_thalamic_default_config(void) {
    bias_thalamic_config_t cfg = {
        .enable_attention_gating = true,
        .enable_automatic_correction = true,
        .min_detection_confidence = 0.3f,
        .correction_threshold = 0.5f
    };
    return cfg;
}

bias_thalamic_bridge_t* bias_thalamic_bridge_create(void* bias, thalamic_router_t* router, const bias_thalamic_config_t* config) {
    bias_thalamic_bridge_t* bridge = nimcp_calloc(1, sizeof(bias_thalamic_bridge_t));
    if (!bridge) return NULL;
    bridge->bias = bias;
    bridge->router = router;
    bridge->config = config ? *config : bias_thalamic_default_config();
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return bridge;
}

void bias_thalamic_bridge_destroy(bias_thalamic_bridge_t* bridge) {
    if (bridge) nimcp_free(bridge);
}

int bias_thalamic_bridge_reset(bias_thalamic_bridge_t* bridge) {
    if (!bridge) return -1;
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return 0;
}

int bias_thalamic_route_detection(bias_thalamic_bridge_t* bridge, const bias_thalamic_signal_t* signal) {
    if (!bridge || !signal) return -1;
    if (bridge->config.enable_attention_gating && signal->detection_confidence < bridge->config.min_detection_confidence) {
        return 0;
    }
    bridge->stats.biases_detected++;
    bridge->stats.avg_detection_confidence = (bridge->stats.avg_detection_confidence * (bridge->stats.biases_detected - 1) +
                                              signal->detection_confidence) / bridge->stats.biases_detected;
    if (signal->signal_type == BIAS_SIGNAL_OVERRIDE) {
        bridge->stats.overrides_successful++;
    }
    return 0;
}

int bias_thalamic_route_correction(bias_thalamic_bridge_t* bridge, const void* correction, float strength) {
    if (!bridge) return -1;
    if (bridge->config.enable_automatic_correction && strength >= bridge->config.correction_threshold) {
        bridge->stats.corrections_applied++;
    }
    return 0;
}

int bias_thalamic_set_attention(bias_thalamic_bridge_t* bridge, float attention) {
    if (!bridge) return -1;
    bridge->attention_weight = attention < 0.0f ? 0.0f : (attention > 1.0f ? 1.0f : attention);
    return 0;
}

int bias_thalamic_get_attention(const bias_thalamic_bridge_t* bridge, float* attention) {
    if (!bridge || !attention) return -1;
    *attention = bridge->attention_weight;
    return 0;
}

int bias_thalamic_bridge_get_stats(const bias_thalamic_bridge_t* bridge, bias_thalamic_stats_t* stats) {
    if (!bridge || !stats) return -1;
    *stats = bridge->stats;
    return 0;
}
