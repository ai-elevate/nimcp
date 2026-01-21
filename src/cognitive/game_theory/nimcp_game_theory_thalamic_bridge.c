/**
 * @file nimcp_game_theory_thalamic_bridge.c
 * @brief Game Theory-Thalamic Bridge Implementation
 */

#include "cognitive/game_theory/nimcp_game_theory_thalamic_bridge.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>

struct game_theory_thalamic_bridge {
    void* game_theory;
    thalamic_router_t* router;
    game_theory_thalamic_config_t config;
    game_theory_thalamic_stats_t stats;
    float attention_weight;
};

game_theory_thalamic_config_t game_theory_thalamic_default_config(void) {
    game_theory_thalamic_config_t cfg = {
        .enable_attention_gating = true,
        .enable_stakes_boost = true,
        .min_stakes_threshold = 0.25f,
        .coalition_boost = 0.35f
    };
    return cfg;
}

game_theory_thalamic_bridge_t* game_theory_thalamic_bridge_create(void* game_theory, thalamic_router_t* router, const game_theory_thalamic_config_t* config) {
    game_theory_thalamic_bridge_t* bridge = nimcp_calloc(1, sizeof(game_theory_thalamic_bridge_t));
    if (!bridge) return NULL;
    bridge->game_theory = game_theory;
    bridge->router = router;
    bridge->config = config ? *config : game_theory_thalamic_default_config();
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return bridge;
}

void game_theory_thalamic_bridge_destroy(game_theory_thalamic_bridge_t* bridge) {
    if (bridge) nimcp_free(bridge);
}

int game_theory_thalamic_bridge_reset(game_theory_thalamic_bridge_t* bridge) {
    if (!bridge) return -1;
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return 0;
}

int game_theory_thalamic_route_strategy(game_theory_thalamic_bridge_t* bridge, const game_theory_thalamic_signal_t* signal) {
    if (!bridge || !signal) return -1;
    if (bridge->config.enable_attention_gating && signal->stakes_level < bridge->config.min_stakes_threshold) {
        return 0;
    }
    bridge->stats.strategies_routed++;
    bridge->stats.avg_stakes_level = (bridge->stats.avg_stakes_level * (bridge->stats.strategies_routed - 1) +
                                      signal->stakes_level) / bridge->stats.strategies_routed;
    return 0;
}

int game_theory_thalamic_route_outcome(game_theory_thalamic_bridge_t* bridge, const void* outcome, float payoff) {
    if (!bridge) return -1;
    bridge->stats.outcomes_processed++;
    return 0;
}

int game_theory_thalamic_set_attention(game_theory_thalamic_bridge_t* bridge, float attention) {
    if (!bridge) return -1;
    bridge->attention_weight = attention < 0.0f ? 0.0f : (attention > 1.0f ? 1.0f : attention);
    return 0;
}

int game_theory_thalamic_get_attention(const game_theory_thalamic_bridge_t* bridge, float* attention) {
    if (!bridge || !attention) return -1;
    *attention = bridge->attention_weight;
    return 0;
}

int game_theory_thalamic_bridge_get_stats(const game_theory_thalamic_bridge_t* bridge, game_theory_thalamic_stats_t* stats) {
    if (!bridge || !stats) return -1;
    *stats = bridge->stats;
    return 0;
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

/**
 * WHAT: Query knowledge graph for Game Theory Thalamic Bridge self-knowledge
 * WHY:  Enable self-awareness about module's role and connections
 * HOW:  Query KG for entity observations and relations
 */
int game_theory_thalamic_bridge_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;
    const kg_entity_t* self = kg_reader_get_entity(kg, "Game_Theory_Thalamic_Bridge");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            /* GT Thalamic bridge self-knowledge logged */
        }
    }
    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Game_Theory_Thalamic_Bridge");
    if (connections) { kg_relation_list_destroy(connections); }
    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Game_Theory_Thalamic_Bridge");
    if (incoming) { kg_relation_list_destroy(incoming); }
    return self ? 1 : 0;
}
