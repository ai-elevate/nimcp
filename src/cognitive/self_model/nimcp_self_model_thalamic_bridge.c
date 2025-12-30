/**
 * @file nimcp_self_model_thalamic_bridge.c
 * @brief Self-Model-Thalamic Bridge Implementation
 */

#include "cognitive/self_model/nimcp_self_model_thalamic_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/time/nimcp_time.h"
#include <string.h>

struct self_model_thalamic_bridge {
    void* self_model;
    thalamic_router_t* router;
    self_model_thalamic_config_t config;
    self_model_thalamic_stats_t stats;
    float attention_weight;
};

self_model_thalamic_config_t self_model_thalamic_default_config(void) {
    return (self_model_thalamic_config_t){
        .enable_attention_gating = true,
        .enable_conflict_priority = true,
        .min_urgency_threshold = 0.2f,
        .conflict_boost = 1.5f
    };
}

self_model_thalamic_bridge_t* self_model_thalamic_bridge_create(
    void* self_model,
    thalamic_router_t* router,
    const self_model_thalamic_config_t* config
) {
    self_model_thalamic_bridge_t* bridge = nimcp_calloc(1, sizeof(self_model_thalamic_bridge_t));
    if (!bridge) return NULL;

    bridge->self_model = self_model;
    bridge->router = router;
    bridge->config = config ? *config : self_model_thalamic_default_config();
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));

    return bridge;
}

void self_model_thalamic_bridge_destroy(self_model_thalamic_bridge_t* bridge) {
    if (bridge) nimcp_free(bridge);
}

int self_model_thalamic_bridge_reset(self_model_thalamic_bridge_t* bridge) {
    if (!bridge) return -1;
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return 0;
}

int self_model_thalamic_route_signal(
    self_model_thalamic_bridge_t* bridge,
    const self_model_thalamic_signal_t* signal
) {
    if (!bridge || !signal) return -1;

    if (bridge->config.enable_attention_gating) {
        float effective_urgency = signal->model_urgency * bridge->attention_weight;

        /* Conflicts get priority boost (require attention) */
        if (bridge->config.enable_conflict_priority &&
            signal->signal_type == SELF_MODEL_SIGNAL_CONFLICT) {
            effective_urgency *= bridge->config.conflict_boost;
            if (effective_urgency > 1.0f) effective_urgency = 1.0f;
        }

        if (effective_urgency < bridge->config.min_urgency_threshold) {
            bridge->stats.signals_gated++;
            return 0;
        }
    }

    switch (signal->signal_type) {
        case SELF_MODEL_SIGNAL_UPDATE:
            bridge->stats.updates_routed++;
            break;
        case SELF_MODEL_SIGNAL_PREDICTION:
            bridge->stats.predictions++;
            break;
        case SELF_MODEL_SIGNAL_CONFLICT:
            bridge->stats.conflicts_detected++;
            break;
        case SELF_MODEL_SIGNAL_INTEGRATION:
            bridge->stats.integrations++;
            break;
        default:
            return -1;
    }

    uint64_t total = bridge->stats.updates_routed + bridge->stats.predictions +
                     bridge->stats.conflicts_detected + bridge->stats.integrations;
    if (total > 0) {
        bridge->stats.avg_coherence =
            (bridge->stats.avg_coherence * (total - 1) + signal->coherence) / total;
        bridge->stats.avg_prediction_error =
            (bridge->stats.avg_prediction_error * (total - 1) + signal->prediction_error) / total;
    }

    return 0;
}

int self_model_thalamic_route_update(
    self_model_thalamic_bridge_t* bridge,
    float coherence,
    float urgency
) {
    if (!bridge) return -1;

    self_model_thalamic_signal_t signal = {
        .signal_type = SELF_MODEL_SIGNAL_UPDATE,
        .model_urgency = urgency < 0.0f ? 0.0f : (urgency > 1.0f ? 1.0f : urgency),
        .coherence = coherence < 0.0f ? 0.0f : (coherence > 1.0f ? 1.0f : coherence),
        .prediction_error = 0.0f,
        .self_relevance = 0.8f,
        .content = NULL,
        .content_size = 0,
        .timestamp_us = nimcp_time_get_us()
    };

    return self_model_thalamic_route_signal(bridge, &signal);
}

int self_model_thalamic_route_conflict(
    self_model_thalamic_bridge_t* bridge,
    float prediction_error,
    float urgency
) {
    if (!bridge) return -1;

    self_model_thalamic_signal_t signal = {
        .signal_type = SELF_MODEL_SIGNAL_CONFLICT,
        .model_urgency = urgency < 0.0f ? 0.0f : (urgency > 1.0f ? 1.0f : urgency),
        .coherence = 1.0f - prediction_error,
        .prediction_error = prediction_error < 0.0f ? 0.0f : (prediction_error > 1.0f ? 1.0f : prediction_error),
        .self_relevance = 1.0f,
        .content = NULL,
        .content_size = 0,
        .timestamp_us = nimcp_time_get_us()
    };

    return self_model_thalamic_route_signal(bridge, &signal);
}

int self_model_thalamic_set_attention(self_model_thalamic_bridge_t* bridge, float attention) {
    if (!bridge) return -1;
    bridge->attention_weight = attention < 0.0f ? 0.0f : (attention > 1.0f ? 1.0f : attention);
    return 0;
}

int self_model_thalamic_get_attention(const self_model_thalamic_bridge_t* bridge, float* attention) {
    if (!bridge || !attention) return -1;
    *attention = bridge->attention_weight;
    return 0;
}

int self_model_thalamic_bridge_get_stats(
    const self_model_thalamic_bridge_t* bridge,
    self_model_thalamic_stats_t* stats
) {
    if (!bridge || !stats) return -1;
    *stats = bridge->stats;
    return 0;
}
