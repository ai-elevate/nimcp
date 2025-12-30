/**
 * @file nimcp_analysis_thalamic_bridge.c
 * @brief Analysis-Thalamic Bridge Implementation
 *
 * Routes analytical processing signals through thalamic relay
 * for prefrontal coordination and attention gating.
 */

#include "cognitive/analysis/nimcp_analysis_thalamic_bridge.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/time/nimcp_time.h"
#include <string.h>

struct analysis_thalamic_bridge {
    bridge_base_t base;  /* MUST be first - provides mutex protection */
    void* analysis;
    thalamic_router_t* router;
    analysis_thalamic_config_t config;
    analysis_thalamic_stats_t stats;
    float attention_weight;
};

analysis_thalamic_config_t analysis_thalamic_default_config(void) {
    return (analysis_thalamic_config_t){
        .enable_attention_gating = true,
        .enable_complexity_routing = true,
        .min_urgency_threshold = 0.2f,
        .complexity_boost = 1.3f
    };
}

analysis_thalamic_bridge_t* analysis_thalamic_bridge_create(
    void* analysis,
    thalamic_router_t* router,
    const analysis_thalamic_config_t* config
) {
    analysis_thalamic_bridge_t* bridge = nimcp_calloc(1, sizeof(analysis_thalamic_bridge_t));
    if (!bridge) return NULL;

    /* Initialize mutex for thread safety */
    bridge->base.mutex = nimcp_mutex_create(NULL);
    if (!bridge->base.mutex) {
        nimcp_free(bridge);
        return NULL;
    }

    bridge->analysis = analysis;
    bridge->router = router;
    bridge->config = config ? *config : analysis_thalamic_default_config();
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));

    return bridge;
}

void analysis_thalamic_bridge_destroy(analysis_thalamic_bridge_t* bridge) {
    if (bridge) {
        if (bridge->base.mutex) {
            nimcp_mutex_destroy(bridge->base.mutex);
        }
        nimcp_free(bridge);
    }
}

int analysis_thalamic_bridge_reset(analysis_thalamic_bridge_t* bridge) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->base.mutex);
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int analysis_thalamic_route_signal(
    analysis_thalamic_bridge_t* bridge,
    const analysis_thalamic_signal_t* signal
) {
    if (!bridge || !signal) return -1;

    nimcp_mutex_lock(bridge->base.mutex);

    /* Apply attention gating */
    if (bridge->config.enable_attention_gating) {
        float effective_urgency = signal->analysis_urgency * bridge->attention_weight;

        /* Complex analyses get priority boost */
        if (bridge->config.enable_complexity_routing &&
            signal->complexity > 0.7f) {
            effective_urgency *= bridge->config.complexity_boost;
            if (effective_urgency > 1.0f) effective_urgency = 1.0f;
        }

        if (effective_urgency < bridge->config.min_urgency_threshold) {
            bridge->stats.signals_gated++;
            nimcp_mutex_unlock(bridge->base.mutex);
            return 0;
        }
    }

    /* Route based on signal type */
    switch (signal->signal_type) {
        case ANALYSIS_SIGNAL_DECOMPOSITION:
            bridge->stats.decompositions_routed++;
            break;

        case ANALYSIS_SIGNAL_DEPTH_REQUEST:
            bridge->stats.depth_requests++;
            break;

        case ANALYSIS_SIGNAL_PATTERN_FOUND:
            bridge->stats.patterns_found++;
            break;

        case ANALYSIS_SIGNAL_COMPLETION:
            bridge->stats.completions++;
            break;

        default:
            nimcp_mutex_unlock(bridge->base.mutex);
            return -1;
    }

    /* Update average urgency */
    uint64_t total = bridge->stats.decompositions_routed +
                     bridge->stats.depth_requests +
                     bridge->stats.patterns_found +
                     bridge->stats.completions;

    if (total > 0) {
        bridge->stats.avg_analysis_urgency =
            (bridge->stats.avg_analysis_urgency * (total - 1) +
             signal->analysis_urgency) / total;
    }

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int analysis_thalamic_route_decomposition(
    analysis_thalamic_bridge_t* bridge,
    const void* problem,
    uint32_t problem_size,
    float complexity
) {
    if (!bridge) return -1;

    analysis_thalamic_signal_t signal = {
        .signal_type = ANALYSIS_SIGNAL_DECOMPOSITION,
        .analysis_urgency = 0.6f + (complexity * 0.3f),
        .depth_required = complexity,
        .complexity = complexity,
        .content = (void*)problem,
        .content_size = problem_size,
        .timestamp_us = nimcp_time_get_us()
    };

    return analysis_thalamic_route_signal(bridge, &signal);
}

int analysis_thalamic_request_depth(
    analysis_thalamic_bridge_t* bridge,
    float depth_required,
    float urgency
) {
    if (!bridge) return -1;

    analysis_thalamic_signal_t signal = {
        .signal_type = ANALYSIS_SIGNAL_DEPTH_REQUEST,
        .analysis_urgency = urgency < 0.0f ? 0.0f : (urgency > 1.0f ? 1.0f : urgency),
        .depth_required = depth_required,
        .complexity = 0.5f,
        .content = NULL,
        .content_size = 0,
        .timestamp_us = nimcp_time_get_us()
    };

    return analysis_thalamic_route_signal(bridge, &signal);
}

int analysis_thalamic_set_attention(analysis_thalamic_bridge_t* bridge, float attention) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->base.mutex);
    bridge->attention_weight = attention < 0.0f ? 0.0f :
                               (attention > 1.0f ? 1.0f : attention);
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int analysis_thalamic_get_attention(const analysis_thalamic_bridge_t* bridge, float* attention) {
    if (!bridge || !attention) return -1;

    nimcp_mutex_lock(bridge->base.mutex);
    *attention = bridge->attention_weight;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int analysis_thalamic_bridge_get_stats(
    const analysis_thalamic_bridge_t* bridge,
    analysis_thalamic_stats_t* stats
) {
    if (!bridge || !stats) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    *stats = bridge->stats;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

void analysis_thalamic_bridge_reset_stats(analysis_thalamic_bridge_t* bridge) {
    if (bridge) {
        nimcp_mutex_lock(bridge->base.mutex);
        memset(&bridge->stats, 0, sizeof(bridge->stats));
        nimcp_mutex_unlock(bridge->base.mutex);
    }
}
