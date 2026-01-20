/**
 * @file nimcp_explanations_thalamic_bridge.c
 * @brief Explanations-Thalamic Bridge Implementation
 */

#include "cognitive/explanations/nimcp_explanations_thalamic_bridge.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/time/nimcp_time.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include <string.h>

struct explanations_thalamic_bridge {
    bridge_base_t base;  /* MUST be first - provides mutex protection */
    void* explanations;
    thalamic_router_t* router;
    explanations_thalamic_config_t config;
    explanations_thalamic_stats_t stats;
    float attention_weight;
};

explanations_thalamic_config_t explanations_thalamic_default_config(void) {
    return (explanations_thalamic_config_t){
        .enable_attention_gating = true,
        .enable_complexity_routing = true,
        .min_urgency_threshold = 0.25f,
        .complexity_boost = 1.2f
    };
}

explanations_thalamic_bridge_t* explanations_thalamic_bridge_create(
    void* explanations,
    thalamic_router_t* router,
    const explanations_thalamic_config_t* config
) {
    explanations_thalamic_bridge_t* bridge = nimcp_calloc(1, sizeof(explanations_thalamic_bridge_t));
    if (!bridge) return NULL;

    /* Initialize mutex for thread safety */
    bridge->base.mutex = nimcp_mutex_create(NULL);
    if (!bridge->base.mutex) {
        nimcp_free(bridge);
        return NULL;
    }

    bridge->explanations = explanations;
    bridge->router = router;
    bridge->config = config ? *config : explanations_thalamic_default_config();
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));

    return bridge;
}

void explanations_thalamic_bridge_destroy(explanations_thalamic_bridge_t* bridge) {
    if (bridge) {
        if (bridge->base.mutex) {
            nimcp_mutex_free(bridge->base.mutex);
        }
        nimcp_free(bridge);
    }
}

int explanations_thalamic_bridge_reset(explanations_thalamic_bridge_t* bridge) {
    if (!bridge) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int explanations_thalamic_route_signal(
    explanations_thalamic_bridge_t* bridge,
    const explanations_thalamic_signal_t* signal
) {
    if (!bridge || !signal) return -1;

    nimcp_mutex_lock(bridge->base.mutex);

    if (bridge->config.enable_attention_gating) {
        float effective_urgency = signal->explanation_urgency * bridge->attention_weight;

        if (bridge->config.enable_complexity_routing && signal->complexity > 0.6f) {
            effective_urgency *= bridge->config.complexity_boost;
            if (effective_urgency > 1.0f) effective_urgency = 1.0f;
        }

        if (effective_urgency < bridge->config.min_urgency_threshold) {
            bridge->stats.signals_gated++;
            nimcp_mutex_unlock(bridge->base.mutex);
            return 0;
        }
    }

    switch (signal->signal_type) {
        case EXPLANATION_SIGNAL_CAUSAL:
            bridge->stats.causal_explanations++;
            break;
        case EXPLANATION_SIGNAL_MECHANISTIC:
            bridge->stats.mechanistic_explanations++;
            break;
        case EXPLANATION_SIGNAL_CONTRASTIVE:
            bridge->stats.contrastive_explanations++;
            break;
        case EXPLANATION_SIGNAL_COMPLETION:
            bridge->stats.completions++;
            break;
        default:
            nimcp_mutex_unlock(bridge->base.mutex);
            return -1;
    }

    uint64_t total = bridge->stats.causal_explanations + bridge->stats.mechanistic_explanations +
                     bridge->stats.contrastive_explanations + bridge->stats.completions;
    if (total > 0) {
        bridge->stats.avg_complexity =
            (bridge->stats.avg_complexity * (total - 1) + signal->complexity) / total;
        bridge->stats.avg_coherence =
            (bridge->stats.avg_coherence * (total - 1) + signal->coherence) / total;
    }

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int explanations_thalamic_route_causal(
    explanations_thalamic_bridge_t* bridge,
    float complexity,
    float urgency
) {
    if (!bridge) return -1;

    explanations_thalamic_signal_t signal = {
        .signal_type = EXPLANATION_SIGNAL_CAUSAL,
        .explanation_urgency = urgency < 0.0f ? 0.0f : (urgency > 1.0f ? 1.0f : urgency),
        .complexity = complexity < 0.0f ? 0.0f : (complexity > 1.0f ? 1.0f : complexity),
        .coherence = 0.7f,
        .completeness = 0.0f,
        .content = NULL,
        .content_size = 0,
        .timestamp_us = nimcp_time_get_us()
    };

    return explanations_thalamic_route_signal(bridge, &signal);
}

int explanations_thalamic_set_attention(explanations_thalamic_bridge_t* bridge, float attention) {
    if (!bridge) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    bridge->attention_weight = attention < 0.0f ? 0.0f : (attention > 1.0f ? 1.0f : attention);
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int explanations_thalamic_get_attention(const explanations_thalamic_bridge_t* bridge, float* attention) {
    if (!bridge || !attention) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    *attention = bridge->attention_weight;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int explanations_thalamic_bridge_get_stats(
    const explanations_thalamic_bridge_t* bridge,
    explanations_thalamic_stats_t* stats
) {
    if (!bridge || !stats) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    *stats = bridge->stats;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

int explanations_thalamic_bridge_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;

    const kg_entity_t* self = kg_reader_get_entity(kg, "Explanations_Thalamic_Bridge");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            (void)self->observations[i];
        }
    }

    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Explanations_Thalamic_Bridge");
    if (connections) {
        kg_relation_list_destroy(connections);
    }

    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Explanations_Thalamic_Bridge");
    if (incoming) {
        kg_relation_list_destroy(incoming);
    }

    return self ? 1 : 0;
}
