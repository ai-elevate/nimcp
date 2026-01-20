/**
 * @file nimcp_introspection_thalamic_bridge.c
 * @brief Introspection-Thalamic Bridge Implementation
 */

#include "cognitive/introspection/nimcp_introspection_thalamic_bridge.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/time/nimcp_time.h"
#include <string.h>

struct introspection_thalamic_bridge {
    bridge_base_t base;
    void* introspection;
    thalamic_router_t* router;
    introspection_thalamic_config_t config;
    introspection_thalamic_stats_t stats;
    float attention_weight;
};

introspection_thalamic_config_t introspection_thalamic_default_config(void) {
    return (introspection_thalamic_config_t){
        .enable_attention_gating = true,
        .enable_depth_routing = true,
        .min_urgency_threshold = 0.2f,
        .depth_boost = 1.25f
    };
}

introspection_thalamic_bridge_t* introspection_thalamic_bridge_create(
    void* introspection,
    thalamic_router_t* router,
    const introspection_thalamic_config_t* config
) {
    introspection_thalamic_bridge_t* bridge = nimcp_calloc(1, sizeof(introspection_thalamic_bridge_t));
    if (!bridge) return NULL;
    bridge->base.mutex = nimcp_mutex_create(NULL);
    if (!bridge->base.mutex) {
        nimcp_free(bridge);
        return NULL;
    }

    bridge->introspection = introspection;
    bridge->router = router;
    bridge->config = config ? *config : introspection_thalamic_default_config();
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));

    return bridge;
}

void introspection_thalamic_bridge_destroy(introspection_thalamic_bridge_t* bridge) {
    if (!bridge) return;
    if (bridge->base.mutex) {
        nimcp_mutex_free(bridge->base.mutex);
    }
    nimcp_free(bridge);
}

int introspection_thalamic_bridge_reset(introspection_thalamic_bridge_t* bridge) {
    if (!bridge) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int introspection_thalamic_route_signal(
    introspection_thalamic_bridge_t* bridge,
    const introspection_thalamic_signal_t* signal
) {
    if (!bridge || !signal) return -1;
    nimcp_mutex_lock(bridge->base.mutex);

    if (bridge->config.enable_attention_gating) {
        float effective_urgency = signal->introspection_urgency * bridge->attention_weight;

        if (bridge->config.enable_depth_routing && signal->depth > 0.6f) {
            effective_urgency *= bridge->config.depth_boost;
            if (effective_urgency > 1.0f) effective_urgency = 1.0f;
        }

        if (effective_urgency < bridge->config.min_urgency_threshold) {
            bridge->stats.signals_gated++;
            nimcp_mutex_unlock(bridge->base.mutex);
            return 0;
        }
    }

    switch (signal->signal_type) {
        case INTROSPECTION_SIGNAL_MONITOR:
            bridge->stats.monitors_routed++;
            break;
        case INTROSPECTION_SIGNAL_REFLECT:
            bridge->stats.reflections++;
            break;
        case INTROSPECTION_SIGNAL_META:
            bridge->stats.meta_insights++;
            break;
        case INTROSPECTION_SIGNAL_AWARENESS:
            bridge->stats.awareness_updates++;
            break;
        default:
            nimcp_mutex_unlock(bridge->base.mutex);
            return -1;
    }

    uint64_t total = bridge->stats.monitors_routed + bridge->stats.reflections +
                     bridge->stats.meta_insights + bridge->stats.awareness_updates;
    if (total > 0) {
        bridge->stats.avg_depth =
            (bridge->stats.avg_depth * (total - 1) + signal->depth) / total;
        bridge->stats.avg_clarity =
            (bridge->stats.avg_clarity * (total - 1) + signal->clarity) / total;
    }

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int introspection_thalamic_route_monitor(
    introspection_thalamic_bridge_t* bridge,
    float depth,
    float urgency
) {
    if (!bridge) return -1;

    introspection_thalamic_signal_t signal = {
        .signal_type = INTROSPECTION_SIGNAL_MONITOR,
        .introspection_urgency = urgency < 0.0f ? 0.0f : (urgency > 1.0f ? 1.0f : urgency),
        .depth = depth < 0.0f ? 0.0f : (depth > 1.0f ? 1.0f : depth),
        .clarity = 0.7f,
        .self_relevance = 0.8f,
        .content = NULL,
        .content_size = 0,
        .timestamp_us = nimcp_time_get_us()
    };

    return introspection_thalamic_route_signal(bridge, &signal);
}

int introspection_thalamic_route_reflection(
    introspection_thalamic_bridge_t* bridge,
    float clarity,
    float urgency
) {
    if (!bridge) return -1;

    introspection_thalamic_signal_t signal = {
        .signal_type = INTROSPECTION_SIGNAL_REFLECT,
        .introspection_urgency = urgency < 0.0f ? 0.0f : (urgency > 1.0f ? 1.0f : urgency),
        .depth = 0.6f,
        .clarity = clarity < 0.0f ? 0.0f : (clarity > 1.0f ? 1.0f : clarity),
        .self_relevance = 0.9f,
        .content = NULL,
        .content_size = 0,
        .timestamp_us = nimcp_time_get_us()
    };

    return introspection_thalamic_route_signal(bridge, &signal);
}

int introspection_thalamic_set_attention(introspection_thalamic_bridge_t* bridge, float attention) {
    if (!bridge) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    bridge->attention_weight = attention < 0.0f ? 0.0f : (attention > 1.0f ? 1.0f : attention);
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int introspection_thalamic_get_attention(const introspection_thalamic_bridge_t* bridge, float* attention) {
    if (!bridge || !attention) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    *attention = bridge->attention_weight;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int introspection_thalamic_bridge_get_stats(
    const introspection_thalamic_bridge_t* bridge,
    introspection_thalamic_stats_t* stats
) {
    if (!bridge || !stats) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    *stats = bridge->stats;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

/* ========================================================================
 * KG SELF-AWARENESS INTEGRATION
 * ======================================================================== */

/**
 * WHAT: Query knowledge graph for self-knowledge about introspection thalamic bridge
 * WHY:  Enable self-awareness - module can introspect its own capabilities
 * HOW:  Query entity by name, get relations from/to
 *
 * @param kg Knowledge graph reader
 * @return 1 if entity found, 0 if not
 */
int introspection_thalamic_bridge_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;

    /* Query our own entity from the knowledge graph */
    const kg_entity_t* self = kg_reader_get_entity(kg, "Introspection_Thalamic_Bridge");
    if (self) {
        /* Module now knows its own capabilities from KG */
        for (uint32_t i = 0; i < self->num_observations; i++) {
            NIMCP_LOGGING_DEBUG("Introspection thalamic bridge self-knowledge: %s", self->observations[i]);
        }
    }

    /* Query connections to understand integration points */
    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Introspection_Thalamic_Bridge");
    if (connections) {
        NIMCP_LOGGING_DEBUG("Introspection thalamic bridge has %u outgoing connections", connections->count);
        kg_relation_list_destroy(connections);
    }

    /* Query incoming connections */
    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Introspection_Thalamic_Bridge");
    if (incoming) {
        NIMCP_LOGGING_DEBUG("Introspection thalamic bridge has %u incoming connections", incoming->count);
        kg_relation_list_destroy(incoming);
    }

    return self ? 1 : 0;
}
