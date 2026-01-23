/**
 * @file nimcp_autobio_thalamic_bridge.c
 * @brief Autobiographical Memory-Thalamic Bridge Implementation
 */

#include "utils/bridge/nimcp_bridge_base.h"
#include "cognitive/autobiographical_memory/nimcp_autobio_thalamic_bridge.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>

struct autobio_thalamic_bridge {
    bridge_base_t base;              /**< MUST be first: base bridge infrastructure */
    void* autobio;
    thalamic_router_t* router;
    autobio_thalamic_config_t config;
    autobio_thalamic_stats_t stats;
    float attention_weight;
};

autobio_thalamic_config_t autobio_thalamic_default_config(void) {
    autobio_thalamic_config_t cfg = {
        .enable_attention_gating = true,
        .enable_emotional_boost = true,
        .min_vividness_threshold = 0.3f,
        .emotional_threshold = 0.5f
    };
    return cfg;
}

autobio_thalamic_bridge_t* autobio_thalamic_bridge_create(void* autobio, thalamic_router_t* router, const autobio_thalamic_config_t* config) {
    autobio_thalamic_bridge_t* bridge = nimcp_calloc(1, sizeof(autobio_thalamic_bridge_t));
    if (!bridge) return NULL;
    bridge->autobio = autobio;
    bridge->router = router;
    bridge->config = config ? *config : autobio_thalamic_default_config();
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return bridge;
}

void autobio_thalamic_bridge_destroy(autobio_thalamic_bridge_t* bridge) {
    if (bridge) nimcp_free(bridge);
}

int autobio_thalamic_bridge_reset(autobio_thalamic_bridge_t* bridge) {
    if (!bridge) return -1;
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return 0;
}

int autobio_thalamic_route_recall(autobio_thalamic_bridge_t* bridge, const autobio_thalamic_signal_t* signal) {
    if (!bridge || !signal) return -1;
    if (bridge->config.enable_attention_gating && signal->vividness < bridge->config.min_vividness_threshold) {
        return 0;
    }
    bridge->stats.recalls_routed++;
    bridge->stats.avg_vividness = (bridge->stats.avg_vividness * (bridge->stats.recalls_routed - 1) +
                                   signal->vividness) / bridge->stats.recalls_routed;
    if (signal->signal_type == AUTOBIO_SIGNAL_NARRATIVE) {
        bridge->stats.narrative_updates++;
    }
    return 0;
}

int autobio_thalamic_route_encoding(autobio_thalamic_bridge_t* bridge, const void* event, float importance) {
    if (!bridge) return -1;
    bridge->stats.encodings_routed++;
    return 0;
}

int autobio_thalamic_set_attention(autobio_thalamic_bridge_t* bridge, float attention) {
    if (!bridge) return -1;
    bridge->attention_weight = attention < 0.0f ? 0.0f : (attention > 1.0f ? 1.0f : attention);
    return 0;
}

int autobio_thalamic_get_attention(const autobio_thalamic_bridge_t* bridge, float* attention) {
    if (!bridge || !attention) return -1;
    *attention = bridge->attention_weight;
    return 0;
}

int autobio_thalamic_bridge_get_stats(const autobio_thalamic_bridge_t* bridge, autobio_thalamic_stats_t* stats) {
    if (!bridge || !stats) return -1;
    *stats = bridge->stats;
    return 0;
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

int autobio_thalamic_bridge_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;

    const kg_entity_t* self = kg_reader_get_entity(kg, "Autobiographical_Thalamic_Bridge");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            (void)self->observations[i];
        }
    }

    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Autobiographical_Thalamic_Bridge");
    if (connections) {
        kg_relation_list_destroy(connections);
    }

    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Autobiographical_Thalamic_Bridge");
    if (incoming) {
        kg_relation_list_destroy(incoming);
    }

    return self ? 1 : 0;
}
