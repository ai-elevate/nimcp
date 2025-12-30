/**
 * @file nimcp_emotion_thalamic_bridge.c
 * @brief Emotion-Thalamic Bridge Implementation
 */

#include "cognitive/emotion/nimcp_emotion_thalamic_bridge.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/time/nimcp_time.h"
#include <string.h>

struct emotion_thalamic_bridge {
    bridge_base_t base;  /* MUST be first - provides mutex protection */
    void* emotion;
    thalamic_router_t* router;
    emotion_thalamic_config_t config;
    emotion_thalamic_stats_t stats;
    float attention_weight;
};

emotion_thalamic_config_t emotion_thalamic_default_config(void) {
    return (emotion_thalamic_config_t){
        .enable_attention_gating = true,
        .enable_intensity_boost = true,
        .enable_regulation_priority = true,
        .min_intensity_threshold = 0.15f,
        .regulation_boost = 1.3f
    };
}

emotion_thalamic_bridge_t* emotion_thalamic_bridge_create(
    void* emotion,
    thalamic_router_t* router,
    const emotion_thalamic_config_t* config
) {
    emotion_thalamic_bridge_t* bridge = nimcp_calloc(1, sizeof(emotion_thalamic_bridge_t));
    if (!bridge) return NULL;

    /* Initialize mutex for thread safety */
    bridge->base.mutex = nimcp_mutex_create(NULL);
    if (!bridge->base.mutex) {
        nimcp_free(bridge);
        return NULL;
    }

    bridge->emotion = emotion;
    bridge->router = router;
    bridge->config = config ? *config : emotion_thalamic_default_config();
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));

    return bridge;
}

void emotion_thalamic_bridge_destroy(emotion_thalamic_bridge_t* bridge) {
    if (bridge) {
        if (bridge->base.mutex) {
            nimcp_mutex_destroy(bridge->base.mutex);
        }
        nimcp_free(bridge);
    }
}

int emotion_thalamic_bridge_reset(emotion_thalamic_bridge_t* bridge) {
    if (!bridge) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int emotion_thalamic_route_signal(
    emotion_thalamic_bridge_t* bridge,
    const emotion_thalamic_signal_t* signal
) {
    if (!bridge || !signal) return -1;

    nimcp_mutex_lock(bridge->base.mutex);

    if (bridge->config.enable_attention_gating) {
        float effective_intensity = signal->emotional_intensity * bridge->attention_weight;

        if (bridge->config.enable_regulation_priority &&
            signal->signal_type == EMOTION_SIGNAL_REGULATION) {
            effective_intensity *= bridge->config.regulation_boost;
            if (effective_intensity > 1.0f) effective_intensity = 1.0f;
        }

        if (effective_intensity < bridge->config.min_intensity_threshold) {
            bridge->stats.signals_gated++;
            nimcp_mutex_unlock(bridge->base.mutex);
            return 0;
        }
    }

    switch (signal->signal_type) {
        case EMOTION_SIGNAL_AROUSAL:
            bridge->stats.arousal_signals++;
            break;
        case EMOTION_SIGNAL_VALENCE:
            bridge->stats.valence_updates++;
            break;
        case EMOTION_SIGNAL_REGULATION:
            bridge->stats.regulations_attempted++;
            break;
        case EMOTION_SIGNAL_EXPRESSION:
            bridge->stats.expressions_routed++;
            break;
        default:
            nimcp_mutex_unlock(bridge->base.mutex);
            return -1;
    }

    uint64_t total = bridge->stats.arousal_signals + bridge->stats.valence_updates +
                     bridge->stats.regulations_attempted + bridge->stats.expressions_routed;
    if (total > 0) {
        bridge->stats.avg_intensity =
            (bridge->stats.avg_intensity * (total - 1) + signal->emotional_intensity) / total;
        bridge->stats.avg_arousal =
            (bridge->stats.avg_arousal * (total - 1) + signal->arousal_level) / total;
    }

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int emotion_thalamic_route_arousal(
    emotion_thalamic_bridge_t* bridge,
    float intensity,
    float arousal
) {
    if (!bridge) return -1;

    emotion_thalamic_signal_t signal = {
        .signal_type = EMOTION_SIGNAL_AROUSAL,
        .emotional_intensity = intensity < 0.0f ? 0.0f : (intensity > 1.0f ? 1.0f : intensity),
        .valence = 0.0f,
        .arousal_level = arousal < 0.0f ? 0.0f : (arousal > 1.0f ? 1.0f : arousal),
        .regulation_effort = 0.0f,
        .content = NULL,
        .content_size = 0,
        .timestamp_us = nimcp_time_get_us()
    };

    return emotion_thalamic_route_signal(bridge, &signal);
}

int emotion_thalamic_route_regulation(
    emotion_thalamic_bridge_t* bridge,
    float effort,
    float urgency
) {
    if (!bridge) return -1;

    emotion_thalamic_signal_t signal = {
        .signal_type = EMOTION_SIGNAL_REGULATION,
        .emotional_intensity = urgency,
        .valence = 0.0f,
        .arousal_level = 0.5f,
        .regulation_effort = effort < 0.0f ? 0.0f : (effort > 1.0f ? 1.0f : effort),
        .content = NULL,
        .content_size = 0,
        .timestamp_us = nimcp_time_get_us()
    };

    return emotion_thalamic_route_signal(bridge, &signal);
}

int emotion_thalamic_set_attention(emotion_thalamic_bridge_t* bridge, float attention) {
    if (!bridge) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    bridge->attention_weight = attention < 0.0f ? 0.0f : (attention > 1.0f ? 1.0f : attention);
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int emotion_thalamic_get_attention(const emotion_thalamic_bridge_t* bridge, float* attention) {
    if (!bridge || !attention) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    *attention = bridge->attention_weight;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int emotion_thalamic_bridge_get_stats(
    const emotion_thalamic_bridge_t* bridge,
    emotion_thalamic_stats_t* stats
) {
    if (!bridge || !stats) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    *stats = bridge->stats;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}
