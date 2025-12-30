/**
 * @file nimcp_intuition_thalamic_bridge.c
 * @brief Intuition-Thalamic Bridge Implementation
 */

#include "cognitive/parietal/nimcp_intuition_thalamic_bridge.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/memory/nimcp_memory.h"
#include <string.h>

struct intuition_thalamic_bridge {
    bridge_base_t base;
    intuition_system_t* intuition;
    thalamic_router_t* router;
    intuition_thalamic_config_t config;
    intuition_thalamic_stats_t stats;
    float attention_weight;
    float signal_type_boosts[8]; /* Boost per signal type */
};

intuition_thalamic_config_t intuition_thalamic_default_config(void) {
    intuition_thalamic_config_t cfg = {
        .enable_attention_gating = true,
        .enable_priority_routing = true,
        .enable_broadcast = true,
        .min_confidence_threshold = 0.3f,
        .min_attention_threshold = 0.2f,
        .emotional_attention_boost = 0.2f,
        .novelty_attention_boost = 0.15f,
        .num_hunch_targets = 0,
        .num_insight_targets = 0,
        .num_meta_targets = 0
    };
    memset(cfg.default_hunch_targets, 0, sizeof(cfg.default_hunch_targets));
    memset(cfg.default_insight_targets, 0, sizeof(cfg.default_insight_targets));
    memset(cfg.default_meta_targets, 0, sizeof(cfg.default_meta_targets));
    return cfg;
}

intuition_thalamic_bridge_t* intuition_thalamic_bridge_create(
    intuition_system_t* intuition,
    thalamic_router_t* router,
    const intuition_thalamic_config_t* config
) {
    intuition_thalamic_bridge_t* bridge = nimcp_calloc(1, sizeof(intuition_thalamic_bridge_t));
    if (!bridge) return NULL;
    bridge->base.mutex = nimcp_platform_mutex_create();
    if (!bridge->base.mutex) { nimcp_free(bridge); return NULL; }
    bridge->intuition = intuition;
    bridge->router = router;
    bridge->config = config ? *config : intuition_thalamic_default_config();
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));

    /* Set default attention boosts for each signal type */
    bridge->signal_type_boosts[0] = INTUITION_ATTENTION_HUNCH_DEFAULT;
    bridge->signal_type_boosts[1] = INTUITION_ATTENTION_INSIGHT_DEFAULT;
    bridge->signal_type_boosts[2] = INTUITION_ATTENTION_ANALOGY_DEFAULT;
    bridge->signal_type_boosts[3] = INTUITION_ATTENTION_HYPOTHESIS_DEFAULT;
    bridge->signal_type_boosts[4] = INTUITION_ATTENTION_BLEND_DEFAULT;
    bridge->signal_type_boosts[5] = INTUITION_ATTENTION_COUNTERFACTUAL_DEFAULT;
    bridge->signal_type_boosts[6] = INTUITION_ATTENTION_META_DEFAULT;
    bridge->signal_type_boosts[7] = 0.5f; /* extrapolation default */

    return bridge;
}

void intuition_thalamic_bridge_destroy(intuition_thalamic_bridge_t* bridge) {
    if (!bridge) return;
    if (bridge->base.mutex) nimcp_platform_mutex_destroy(bridge->base.mutex);
    nimcp_free(bridge);
}

int intuition_thalamic_bridge_reset(intuition_thalamic_bridge_t* bridge) {
    if (!bridge) return -1;
    nimcp_platform_mutex_lock(bridge->base.mutex);
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return 0;
}

int intuition_thalamic_route_hunch(
    intuition_thalamic_bridge_t* bridge,
    const hunch_t* hunch
) {
    if (!bridge || !hunch) return -1;
    nimcp_platform_mutex_lock(bridge->base.mutex);

    float confidence = hunch->score.confidence;
    if (bridge->config.enable_attention_gating && confidence < bridge->config.min_confidence_threshold) {
        bridge->stats.signals_dropped++;
        nimcp_platform_mutex_unlock(bridge->base.mutex);
        return 0;
    }

    bridge->stats.hunches_routed++;
    bridge->stats.signals_routed++;
    bridge->stats.avg_confidence = (bridge->stats.avg_confidence * (bridge->stats.signals_routed - 1) +
                                    confidence) / bridge->stats.signals_routed;

    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return 0;
}

int intuition_thalamic_route_insight(
    intuition_thalamic_bridge_t* bridge,
    const void* insight,
    float novelty
) {
    if (!bridge) return -1;
    nimcp_platform_mutex_lock(bridge->base.mutex);

    float effective_attention = bridge->attention_weight;
    if (novelty > 0.5f) {
        effective_attention += bridge->config.novelty_attention_boost;
    }

    bridge->stats.insights_routed++;
    bridge->stats.signals_routed++;
    bridge->stats.avg_attention_weight = (bridge->stats.avg_attention_weight * (bridge->stats.signals_routed - 1) +
                                          effective_attention) / bridge->stats.signals_routed;

    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return 0;
}

int intuition_thalamic_route_analogy(
    intuition_thalamic_bridge_t* bridge,
    const void* analogy,
    float strength
) {
    if (!bridge) return -1;
    nimcp_platform_mutex_lock(bridge->base.mutex);

    if (bridge->config.enable_attention_gating && strength < bridge->config.min_confidence_threshold) {
        bridge->stats.signals_dropped++;
        nimcp_platform_mutex_unlock(bridge->base.mutex);
        return 0;
    }

    bridge->stats.analogies_routed++;
    bridge->stats.signals_routed++;

    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return 0;
}

int intuition_thalamic_route_hypothesis(
    intuition_thalamic_bridge_t* bridge,
    const hypogen_theory_t* theory
) {
    if (!bridge || !theory) return -1;
    nimcp_platform_mutex_lock(bridge->base.mutex);

    bridge->stats.hypotheses_routed++;
    bridge->stats.signals_routed++;

    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return 0;
}

int intuition_thalamic_route_blend(
    intuition_thalamic_bridge_t* bridge,
    const void* blend,
    float creativity
) {
    if (!bridge) return -1;
    nimcp_platform_mutex_lock(bridge->base.mutex);

    bridge->stats.blends_routed++;
    bridge->stats.signals_routed++;

    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return 0;
}

int intuition_thalamic_route_meta(
    intuition_thalamic_bridge_t* bridge,
    const void* meta_signal
) {
    if (!bridge) return -1;
    nimcp_platform_mutex_lock(bridge->base.mutex);

    bridge->stats.meta_signals_routed++;
    bridge->stats.signals_routed++;

    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return 0;
}

int intuition_thalamic_route_signal(
    intuition_thalamic_bridge_t* bridge,
    const intuition_signal_t* signal,
    const intuition_routing_target_t* targets,
    uint32_t num_targets
) {
    if (!bridge || !signal) return -1;
    nimcp_platform_mutex_lock(bridge->base.mutex);

    float effective_attention = bridge->attention_weight;

    /* Apply emotional boost */
    if (signal->emotional_valence > 0.3f || signal->emotional_valence < -0.3f) {
        effective_attention += bridge->config.emotional_attention_boost;
    }

    /* Apply novelty boost */
    if (signal->novelty > 0.5f) {
        effective_attention += bridge->config.novelty_attention_boost;
    }

    /* Check gating */
    if (bridge->config.enable_attention_gating) {
        if (signal->confidence < bridge->config.min_confidence_threshold) {
            bridge->stats.signals_dropped++;
            nimcp_platform_mutex_unlock(bridge->base.mutex);
            return 0;
        }
        if (effective_attention < bridge->config.min_attention_threshold) {
            bridge->stats.signals_dropped++;
            nimcp_platform_mutex_unlock(bridge->base.mutex);
            return 0;
        }
    }

    /* Check for priority bypass */
    if (bridge->config.enable_priority_routing && signal->confidence > 0.9f) {
        bridge->stats.signals_bypassed++;
    }

    bridge->stats.signals_routed++;
    bridge->stats.avg_attention_weight = (bridge->stats.avg_attention_weight * (bridge->stats.signals_routed - 1) +
                                          effective_attention) / bridge->stats.signals_routed;
    bridge->stats.avg_confidence = (bridge->stats.avg_confidence * (bridge->stats.signals_routed - 1) +
                                    signal->confidence) / bridge->stats.signals_routed;

    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return 0;
}

int intuition_thalamic_set_attention(
    intuition_thalamic_bridge_t* bridge,
    float attention
) {
    if (!bridge) return -1;
    nimcp_platform_mutex_lock(bridge->base.mutex);
    bridge->attention_weight = attention < 0.0f ? 0.0f : (attention > 1.0f ? 1.0f : attention);
    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return 0;
}

int intuition_thalamic_get_attention(
    const intuition_thalamic_bridge_t* bridge,
    float* attention
) {
    if (!bridge || !attention) return -1;
    nimcp_platform_mutex_lock(bridge->base.mutex);
    *attention = bridge->attention_weight;
    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return 0;
}

int intuition_thalamic_boost_attention(
    intuition_thalamic_bridge_t* bridge,
    uint32_t signal_type,
    float boost
) {
    if (!bridge) return -1;
    nimcp_platform_mutex_lock(bridge->base.mutex);

    /* Map signal type to array index */
    uint32_t index = 0;
    switch (signal_type) {
        case INTUITION_SIGNAL_HUNCH:        index = 0; break;
        case INTUITION_SIGNAL_INSIGHT:      index = 1; break;
        case INTUITION_SIGNAL_ANALOGY:      index = 2; break;
        case INTUITION_SIGNAL_HYPOTHESIS:   index = 3; break;
        case INTUITION_SIGNAL_BLEND:        index = 4; break;
        case INTUITION_SIGNAL_COUNTERFACTUAL: index = 5; break;
        case INTUITION_SIGNAL_META:         index = 6; break;
        case INTUITION_SIGNAL_EXTRAPOLATION: index = 7; break;
        default:
            nimcp_platform_mutex_unlock(bridge->base.mutex);
            return -1;
    }

    bridge->signal_type_boosts[index] += boost;
    if (bridge->signal_type_boosts[index] > 1.0f) {
        bridge->signal_type_boosts[index] = 1.0f;
    }
    if (bridge->signal_type_boosts[index] < 0.0f) {
        bridge->signal_type_boosts[index] = 0.0f;
    }

    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return 0;
}

int intuition_thalamic_bridge_get_stats(
    const intuition_thalamic_bridge_t* bridge,
    intuition_thalamic_stats_t* stats
) {
    if (!bridge || !stats) return -1;
    nimcp_platform_mutex_lock(bridge->base.mutex);
    *stats = bridge->stats;
    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return 0;
}

void intuition_thalamic_bridge_reset_stats(intuition_thalamic_bridge_t* bridge) {
    if (bridge) {
        nimcp_platform_mutex_lock(bridge->base.mutex);
        memset(&bridge->stats, 0, sizeof(bridge->stats));
        nimcp_platform_mutex_unlock(bridge->base.mutex);
    }
}
