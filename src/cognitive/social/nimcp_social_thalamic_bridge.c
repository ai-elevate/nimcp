/**
 * @file nimcp_social_thalamic_bridge.c
 * @brief Social Cognition-Thalamic Bridge Implementation
 */

#include "utils/bridge/nimcp_bridge_base.h"
#include "cognitive/social/nimcp_social_thalamic_bridge.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>

struct social_thalamic_bridge {
    bridge_base_t base;              /**< MUST be first: base bridge infrastructure */
    void* social;
    thalamic_router_t* router;
    social_thalamic_config_t config;
    social_thalamic_stats_t stats;
    float attention_weight;
};

social_thalamic_config_t social_thalamic_default_config(void) {
    social_thalamic_config_t cfg = {
        .enable_attention_gating = true,
        .enable_salience_boost = true,
        .min_salience_threshold = 0.25f,
        .betrayal_boost = 0.5f  /* Betrayal signals get strong priority */
    };
    return cfg;
}

social_thalamic_bridge_t* social_thalamic_bridge_create(void* social, thalamic_router_t* router, const social_thalamic_config_t* config) {
    social_thalamic_bridge_t* bridge = nimcp_calloc(1, sizeof(social_thalamic_bridge_t));
    if (!bridge) return NULL;
    bridge->social = social;
    bridge->router = router;
    bridge->config = config ? *config : social_thalamic_default_config();
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return bridge;
}

void social_thalamic_bridge_destroy(social_thalamic_bridge_t* bridge) {
    if (bridge) nimcp_free(bridge);
}

int social_thalamic_bridge_reset(social_thalamic_bridge_t* bridge) {
    if (!bridge) return -1;
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return 0;
}

int social_thalamic_route_bond(social_thalamic_bridge_t* bridge, const social_thalamic_signal_t* signal) {
    if (!bridge || !signal) return -1;
    /* Betrayal signals bypass attention gating */
    if (bridge->config.enable_attention_gating &&
        signal->social_salience < bridge->config.min_salience_threshold &&
        signal->signal_type != SOCIAL_SIGNAL_BETRAYAL) {
        return 0;
    }
    bridge->stats.bonds_routed++;
    bridge->stats.avg_social_salience = (bridge->stats.avg_social_salience * (bridge->stats.bonds_routed - 1) +
                                         signal->social_salience) / bridge->stats.bonds_routed;
    if (signal->signal_type == SOCIAL_SIGNAL_BETRAYAL) {
        bridge->stats.betrayals_detected++;
    }
    return 0;
}

int social_thalamic_route_trust(social_thalamic_bridge_t* bridge, const void* trust_event, float significance) {
    if (!bridge) return -1;
    bridge->stats.trust_events++;
    return 0;
}

int social_thalamic_set_attention(social_thalamic_bridge_t* bridge, float attention) {
    if (!bridge) return -1;
    bridge->attention_weight = attention < 0.0f ? 0.0f : (attention > 1.0f ? 1.0f : attention);
    return 0;
}

int social_thalamic_get_attention(const social_thalamic_bridge_t* bridge, float* attention) {
    if (!bridge || !attention) return -1;
    *attention = bridge->attention_weight;
    return 0;
}

int social_thalamic_bridge_get_stats(const social_thalamic_bridge_t* bridge, social_thalamic_stats_t* stats) {
    if (!bridge || !stats) return -1;
    *stats = bridge->stats;
    return 0;
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

int social_thalamic_bridge_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;

    const kg_entity_t* self = kg_reader_get_entity(kg, "Social_Thalamic_Bridge");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            (void)self->observations[i];
        }
    }

    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Social_Thalamic_Bridge");
    if (connections) {
        kg_relation_list_destroy(connections);
    }

    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Social_Thalamic_Bridge");
    if (incoming) {
        kg_relation_list_destroy(incoming);
    }

    return self ? 1 : 0;
}
