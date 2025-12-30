/**
 * @file nimcp_emotion_recognition_thalamic_bridge.c
 * @brief Emotion Recognition-Thalamic Bridge Implementation
 */

#include "cognitive/emotion_recognition/nimcp_emotion_recognition_thalamic_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include <string.h>

struct emotion_recognition_thalamic_bridge {
    void* emotion_rec;
    thalamic_router_t* router;
    emotion_recognition_thalamic_config_t config;
    emotion_recognition_thalamic_stats_t stats;
    float attention_weight;
};

emotion_recognition_thalamic_config_t emotion_recognition_thalamic_default_config(void) {
    emotion_recognition_thalamic_config_t cfg = {
        .enable_attention_gating = true,
        .enable_threat_priority = true,
        .min_recognition_confidence = 0.3f,
        .threat_boost = 0.3f
    };
    return cfg;
}

emotion_recognition_thalamic_bridge_t* emotion_recognition_thalamic_bridge_create(void* emotion_rec, thalamic_router_t* router, const emotion_recognition_thalamic_config_t* config) {
    emotion_recognition_thalamic_bridge_t* bridge = nimcp_calloc(1, sizeof(emotion_recognition_thalamic_bridge_t));
    if (!bridge) return NULL;
    bridge->emotion_rec = emotion_rec;
    bridge->router = router;
    bridge->config = config ? *config : emotion_recognition_thalamic_default_config();
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return bridge;
}

void emotion_recognition_thalamic_bridge_destroy(emotion_recognition_thalamic_bridge_t* bridge) {
    if (bridge) nimcp_free(bridge);
}

int emotion_recognition_thalamic_bridge_reset(emotion_recognition_thalamic_bridge_t* bridge) {
    if (!bridge) return -1;
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return 0;
}

int emotion_recognition_thalamic_route_recognition(emotion_recognition_thalamic_bridge_t* bridge, const emotion_recognition_thalamic_signal_t* signal) {
    if (!bridge || !signal) return -1;
    if (bridge->config.enable_attention_gating && signal->recognition_confidence < bridge->config.min_recognition_confidence) {
        return 0;
    }
    bridge->stats.recognitions_routed++;
    bridge->stats.avg_recognition_confidence = (bridge->stats.avg_recognition_confidence * (bridge->stats.recognitions_routed - 1) +
                                                signal->recognition_confidence) / bridge->stats.recognitions_routed;
    /* Threat detection based on high emotional intensity with threat priority enabled */
    if (bridge->config.enable_threat_priority && signal->emotional_intensity > 0.7f) {
        bridge->stats.threat_detections++;
    }
    return 0;
}

int emotion_recognition_thalamic_route_context(emotion_recognition_thalamic_bridge_t* bridge, const void* context, float relevance) {
    if (!bridge) return -1;
    bridge->stats.context_integrations++;
    return 0;
}

int emotion_recognition_thalamic_set_attention(emotion_recognition_thalamic_bridge_t* bridge, float attention) {
    if (!bridge) return -1;
    bridge->attention_weight = attention < 0.0f ? 0.0f : (attention > 1.0f ? 1.0f : attention);
    return 0;
}

int emotion_recognition_thalamic_get_attention(const emotion_recognition_thalamic_bridge_t* bridge, float* attention) {
    if (!bridge || !attention) return -1;
    *attention = bridge->attention_weight;
    return 0;
}

int emotion_recognition_thalamic_bridge_get_stats(const emotion_recognition_thalamic_bridge_t* bridge, emotion_recognition_thalamic_stats_t* stats) {
    if (!bridge || !stats) return -1;
    *stats = bridge->stats;
    return 0;
}
