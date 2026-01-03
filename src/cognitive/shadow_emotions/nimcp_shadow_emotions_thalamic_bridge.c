/**
 * @file nimcp_shadow_emotions_thalamic_bridge.c
 * @brief Shadow Emotions-Thalamic Bridge Implementation
 */

#include "cognitive/shadow_emotions/nimcp_shadow_emotions_thalamic_bridge.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/time/nimcp_time.h"
#include <string.h>

struct shadow_emotions_thalamic_bridge {
    void* shadow_emotions;
    thalamic_router_t* router;
    shadow_emotions_thalamic_config_t config;
    shadow_emotions_thalamic_stats_t stats;
    float attention_weight;
};

shadow_emotions_thalamic_config_t shadow_emotions_thalamic_default_config(void) {
    return (shadow_emotions_thalamic_config_t){
        .enable_attention_gating = true,
        .enable_emergence_priority = true,
        .min_urgency_threshold = 0.15f,
        .emergence_boost = 1.4f
    };
}

shadow_emotions_thalamic_bridge_t* shadow_emotions_thalamic_bridge_create(
    void* shadow_emotions,
    thalamic_router_t* router,
    const shadow_emotions_thalamic_config_t* config
) {
    shadow_emotions_thalamic_bridge_t* bridge = nimcp_calloc(1, sizeof(shadow_emotions_thalamic_bridge_t));
    if (!bridge) return NULL;

    bridge->shadow_emotions = shadow_emotions;
    bridge->router = router;
    bridge->config = config ? *config : shadow_emotions_thalamic_default_config();
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));

    return bridge;
}

void shadow_emotions_thalamic_bridge_destroy(shadow_emotions_thalamic_bridge_t* bridge) {
    if (bridge) nimcp_free(bridge);
}

int shadow_emotions_thalamic_bridge_reset(shadow_emotions_thalamic_bridge_t* bridge) {
    if (!bridge) return -1;
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return 0;
}

int shadow_emotions_thalamic_route_signal(
    shadow_emotions_thalamic_bridge_t* bridge,
    const shadow_emotions_thalamic_signal_t* signal
) {
    if (!bridge || !signal) return -1;

    if (bridge->config.enable_attention_gating) {
        float effective_urgency = signal->shadow_urgency * bridge->attention_weight;

        /* Emergence gets priority (breaking through suppression) */
        if (bridge->config.enable_emergence_priority &&
            signal->signal_type == SHADOW_EMO_SIGNAL_EMERGENCE) {
            effective_urgency *= bridge->config.emergence_boost;
            if (effective_urgency > 1.0f) effective_urgency = 1.0f;
        }

        if (effective_urgency < bridge->config.min_urgency_threshold) {
            bridge->stats.signals_gated++;
            return 0;
        }
    }

    switch (signal->signal_type) {
        case SHADOW_EMO_SIGNAL_EMERGENCE:
            bridge->stats.emergences++;
            break;
        case SHADOW_EMO_SIGNAL_SUPPRESSION:
            bridge->stats.suppressions++;
            break;
        case SHADOW_EMO_SIGNAL_INTEGRATION:
            bridge->stats.integrations++;
            break;
        case SHADOW_EMO_SIGNAL_PROJECTION:
            bridge->stats.projections++;
            break;
        default:
            return -1;
    }

    uint64_t total = bridge->stats.emergences + bridge->stats.suppressions +
                     bridge->stats.integrations + bridge->stats.projections;
    if (total > 0) {
        bridge->stats.avg_suppression_strength =
            (bridge->stats.avg_suppression_strength * (total - 1) + signal->suppression_strength) / total;
        bridge->stats.avg_emergence_pressure =
            (bridge->stats.avg_emergence_pressure * (total - 1) + signal->emergence_pressure) / total;
    }

    return 0;
}

int shadow_emotions_thalamic_route_emergence(
    shadow_emotions_thalamic_bridge_t* bridge,
    float pressure,
    float urgency
) {
    if (!bridge) return -1;

    shadow_emotions_thalamic_signal_t signal = {
        .signal_type = SHADOW_EMO_SIGNAL_EMERGENCE,
        .shadow_urgency = urgency < 0.0f ? 0.0f : (urgency > 1.0f ? 1.0f : urgency),
        .suppression_strength = 0.3f,
        .emergence_pressure = pressure < 0.0f ? 0.0f : (pressure > 1.0f ? 1.0f : pressure),
        .integration_readiness = 0.5f,
        .content = NULL,
        .content_size = 0,
        .timestamp_us = nimcp_time_get_us()
    };

    return shadow_emotions_thalamic_route_signal(bridge, &signal);
}

int shadow_emotions_thalamic_set_attention(shadow_emotions_thalamic_bridge_t* bridge, float attention) {
    if (!bridge) return -1;
    bridge->attention_weight = attention < 0.0f ? 0.0f : (attention > 1.0f ? 1.0f : attention);
    return 0;
}

int shadow_emotions_thalamic_get_attention(const shadow_emotions_thalamic_bridge_t* bridge, float* attention) {
    if (!bridge || !attention) return -1;
    *attention = bridge->attention_weight;
    return 0;
}

int shadow_emotions_thalamic_bridge_get_stats(
    const shadow_emotions_thalamic_bridge_t* bridge,
    shadow_emotions_thalamic_stats_t* stats
) {
    if (!bridge || !stats) return -1;
    *stats = bridge->stats;
    return 0;
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

int shadow_emotions_thalamic_bridge_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;

    const kg_entity_t* self = kg_reader_get_entity(kg, "Shadow_Emotions_Thalamic_Bridge");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            (void)self->observations[i];
        }
    }

    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Shadow_Emotions_Thalamic_Bridge");
    if (connections) {
        kg_relation_list_destroy(connections);
    }

    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Shadow_Emotions_Thalamic_Bridge");
    if (incoming) {
        kg_relation_list_destroy(incoming);
    }

    return self ? 1 : 0;
}
