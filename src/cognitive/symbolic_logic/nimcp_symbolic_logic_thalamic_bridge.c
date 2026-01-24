/**
 * @file nimcp_symbolic_logic_thalamic_bridge.c
 * @brief Symbolic Logic-Thalamic Bridge Implementation
 */

#include "utils/bridge/nimcp_bridge_base.h"
#include "cognitive/symbolic_logic/nimcp_symbolic_logic_thalamic_bridge.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/time/nimcp_time.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>

struct symbolic_logic_thalamic_bridge {
    bridge_base_t base;              /**< MUST be first: base bridge infrastructure */
    void* symbolic_logic;
    thalamic_router_t* router;
    symbolic_logic_thalamic_config_t config;
    symbolic_logic_thalamic_stats_t stats;
    float attention_weight;
};

symbolic_logic_thalamic_config_t symbolic_logic_thalamic_default_config(void) {
    return (symbolic_logic_thalamic_config_t){
        .enable_attention_gating = true,
        .enable_complexity_routing = true,
        .min_urgency_threshold = 0.25f,
        .complexity_boost = 1.3f
    };
}

symbolic_logic_thalamic_bridge_t* symbolic_logic_thalamic_bridge_create(
    void* symbolic_logic,
    thalamic_router_t* router,
    const symbolic_logic_thalamic_config_t* config
) {
    symbolic_logic_thalamic_bridge_t* bridge = nimcp_calloc(1, sizeof(symbolic_logic_thalamic_bridge_t));
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return NULL;

    }

    bridge->symbolic_logic = symbolic_logic;
    bridge->router = router;
    bridge->config = config ? *config : symbolic_logic_thalamic_default_config();
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));

    return bridge;
}

void symbolic_logic_thalamic_bridge_destroy(symbolic_logic_thalamic_bridge_t* bridge) {
    if (bridge) nimcp_free(bridge);
}

int symbolic_logic_thalamic_bridge_reset(symbolic_logic_thalamic_bridge_t* bridge) {
    if (!bridge) return -1;
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return 0;
}

int symbolic_logic_thalamic_route_signal(
    symbolic_logic_thalamic_bridge_t* bridge,
    const symbolic_logic_thalamic_signal_t* signal
) {
    if (!bridge || !signal) return -1;

    if (bridge->config.enable_attention_gating) {
        float effective_urgency = signal->logic_urgency * bridge->attention_weight;

        if (bridge->config.enable_complexity_routing && signal->complexity > 0.6f) {
            effective_urgency *= bridge->config.complexity_boost;
            if (effective_urgency > 1.0f) effective_urgency = 1.0f;
        }

        if (effective_urgency < bridge->config.min_urgency_threshold) {
            bridge->stats.signals_gated++;
            return 0;
        }
    }

    switch (signal->signal_type) {
        case SYMBOLIC_SIGNAL_RULE_APPLY:
            bridge->stats.rules_applied++;
            break;
        case SYMBOLIC_SIGNAL_UNIFICATION:
            bridge->stats.unifications++;
            break;
        case SYMBOLIC_SIGNAL_INFERENCE:
            bridge->stats.inferences++;
            break;
        case SYMBOLIC_SIGNAL_PROOF:
            bridge->stats.proof_steps++;
            break;
        default:
            return -1;
    }

    uint64_t total = bridge->stats.rules_applied + bridge->stats.unifications +
                     bridge->stats.inferences + bridge->stats.proof_steps;
    if (total > 0) {
        bridge->stats.avg_complexity =
            (bridge->stats.avg_complexity * (total - 1) + signal->complexity) / total;
        bridge->stats.avg_proof_depth =
            (bridge->stats.avg_proof_depth * (total - 1) + signal->proof_depth) / total;
    }

    return 0;
}

int symbolic_logic_thalamic_apply_rule(
    symbolic_logic_thalamic_bridge_t* bridge,
    float complexity,
    float urgency
) {
    if (!bridge) return -1;

    symbolic_logic_thalamic_signal_t signal = {
        .signal_type = SYMBOLIC_SIGNAL_RULE_APPLY,
        .logic_urgency = urgency < 0.0f ? 0.0f : (urgency > 1.0f ? 1.0f : urgency),
        .complexity = complexity < 0.0f ? 0.0f : (complexity > 1.0f ? 1.0f : complexity),
        .confidence = 0.9f,
        .proof_depth = 0.0f,
        .content = NULL,
        .content_size = 0,
        .timestamp_us = nimcp_time_get_us()
    };

    return symbolic_logic_thalamic_route_signal(bridge, &signal);
}

int symbolic_logic_thalamic_set_attention(symbolic_logic_thalamic_bridge_t* bridge, float attention) {
    if (!bridge) return -1;
    bridge->attention_weight = attention < 0.0f ? 0.0f : (attention > 1.0f ? 1.0f : attention);
    return 0;
}

int symbolic_logic_thalamic_get_attention(const symbolic_logic_thalamic_bridge_t* bridge, float* attention) {
    if (!bridge || !attention) return -1;
    *attention = bridge->attention_weight;
    return 0;
}

int symbolic_logic_thalamic_bridge_get_stats(
    const symbolic_logic_thalamic_bridge_t* bridge,
    symbolic_logic_thalamic_stats_t* stats
) {
    if (!bridge || !stats) return -1;
    *stats = bridge->stats;
    return 0;
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

int symbolic_logic_thalamic_bridge_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;

    const kg_entity_t* self = kg_reader_get_entity(kg, "Symbolic_Logic_Thalamic_Bridge");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            (void)self->observations[i];
        }
    }

    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Symbolic_Logic_Thalamic_Bridge");
    if (connections) {
        kg_relation_list_destroy(connections);
    }

    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Symbolic_Logic_Thalamic_Bridge");
    if (incoming) {
        kg_relation_list_destroy(incoming);
    }

    return self ? 1 : 0;
}
