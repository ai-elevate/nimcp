/**
 * @file nimcp_self_awareness_thalamic_bridge.c
 * @brief Self-Awareness-Thalamic Bridge Implementation
 */

#include "cognitive/self_awareness/nimcp_self_awareness_thalamic_bridge.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>

struct self_awareness_thalamic_bridge {
    bridge_base_t base;
    void* self_awareness;
    thalamic_router_t* router;
    self_awareness_thalamic_config_t config;
    self_awareness_thalamic_stats_t stats;
    float attention_weight;
};

self_awareness_thalamic_config_t self_awareness_thalamic_default_config(void) {
    self_awareness_thalamic_config_t cfg = {
        .enable_attention_gating = true,
        .enable_metacognitive_boost = true,
        .min_introspection_depth = 0.3f,
        .identity_threshold = 0.5f
    };
    return cfg;
}

self_awareness_thalamic_bridge_t* self_awareness_thalamic_bridge_create(void* self_awareness, thalamic_router_t* router, const self_awareness_thalamic_config_t* config) {
    self_awareness_thalamic_bridge_t* bridge = nimcp_calloc(1, sizeof(self_awareness_thalamic_bridge_t));
    if (!bridge) return NULL;
    bridge->base.mutex = nimcp_mutex_create(NULL);
    if (!bridge->base.mutex) {
        nimcp_free(bridge);
        return NULL;
    }
    bridge->self_awareness = self_awareness;
    bridge->router = router;
    bridge->config = config ? *config : self_awareness_thalamic_default_config();
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return bridge;
}

void self_awareness_thalamic_bridge_destroy(self_awareness_thalamic_bridge_t* bridge) {
    if (!bridge) return;
    if (bridge->base.mutex) {
        nimcp_mutex_free(bridge->base.mutex);
    }
    nimcp_free(bridge);
}

int self_awareness_thalamic_bridge_reset(self_awareness_thalamic_bridge_t* bridge) {
    if (!bridge) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int self_awareness_thalamic_route_introspection(self_awareness_thalamic_bridge_t* bridge, const self_awareness_thalamic_signal_t* signal) {
    if (!bridge || !signal) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    if (bridge->config.enable_attention_gating && signal->introspection_depth < bridge->config.min_introspection_depth) {
        nimcp_mutex_unlock(bridge->base.mutex);
        return 0;
    }
    bridge->stats.introspections_routed++;
    bridge->stats.avg_introspection_depth = (bridge->stats.avg_introspection_depth * (bridge->stats.introspections_routed - 1) +
                                             signal->introspection_depth) / bridge->stats.introspections_routed;
    if (signal->signal_type == SELF_SIGNAL_IDENTITY) {
        bridge->stats.identity_updates++;
    }
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int self_awareness_thalamic_route_metacognition(self_awareness_thalamic_bridge_t* bridge, const void* metacog, float accuracy) {
    if (!bridge) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    bridge->stats.metacognitions_routed++;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int self_awareness_thalamic_set_attention(self_awareness_thalamic_bridge_t* bridge, float attention) {
    if (!bridge) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    bridge->attention_weight = attention < 0.0f ? 0.0f : (attention > 1.0f ? 1.0f : attention);
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int self_awareness_thalamic_get_attention(const self_awareness_thalamic_bridge_t* bridge, float* attention) {
    if (!bridge || !attention) return -1;
    *attention = bridge->attention_weight;
    return 0;
}

int self_awareness_thalamic_bridge_get_stats(const self_awareness_thalamic_bridge_t* bridge, self_awareness_thalamic_stats_t* stats) {
    if (!bridge || !stats) return -1;
    *stats = bridge->stats;
    return 0;
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

int self_awareness_thalamic_bridge_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;

    const kg_entity_t* self = kg_reader_get_entity(kg, "Self_Awareness_Thalamic_Bridge");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            (void)self->observations[i];
        }
    }

    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Self_Awareness_Thalamic_Bridge");
    if (connections) {
        kg_relation_list_destroy(connections);
    }

    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Self_Awareness_Thalamic_Bridge");
    if (incoming) {
        kg_relation_list_destroy(incoming);
    }

    return self ? 1 : 0;
}
