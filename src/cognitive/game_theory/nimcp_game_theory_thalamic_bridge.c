/**
 * @file nimcp_game_theory_thalamic_bridge.c
 * @brief Game Theory-Thalamic Bridge Implementation
 */

#include "utils/bridge/nimcp_bridge_base.h"
#include "cognitive/game_theory/nimcp_game_theory_thalamic_bridge.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>

//=============================================================================
#include <stddef.h>  /* for NULL */
#include "utils/logging/nimcp_logging.h"
// Health Agent Integration (Phase 8: System-Wide Health Integration)
//=============================================================================
struct nimcp_health_agent;
typedef struct nimcp_health_agent nimcp_health_agent_t;
extern void nimcp_health_agent_heartbeat_ex(nimcp_health_agent_t* agent,
                                             const char* operation,
                                             float progress);

/** Global health agent for game_theory_thalamic_bridge module */
static nimcp_health_agent_t* g_game_theory_thalamic_bridge_health_agent = NULL;

/**
 * @brief Set health agent for game_theory_thalamic_bridge heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
void game_theory_thalamic_bridge_set_health_agent(nimcp_health_agent_t* agent) {
    g_game_theory_thalamic_bridge_health_agent = agent;
}

/** @brief Send heartbeat from game_theory_thalamic_bridge module */
static inline void game_theory_thalamic_bridge_heartbeat(const char* operation, float progress) {
    if (g_game_theory_thalamic_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_game_theory_thalamic_bridge_health_agent, operation, progress);
    }
}

/** @brief Send heartbeat from game_theory_thalamic_bridge module (instance-level) */
static inline void game_theory_thalamic_bridge_heartbeat_instance(
    nimcp_health_agent_t* instance_agent, const char* operation, float progress)
{
    if (g_game_theory_thalamic_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_game_theory_thalamic_bridge_health_agent, operation, progress);
    }
    if (instance_agent && instance_agent != g_game_theory_thalamic_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(instance_agent, operation, progress);
    }
}


#define LOG_MODULE "GAME_THEORY_THALAMIC_BRIDGE"


struct game_theory_thalamic_bridge {
    bridge_base_t base;              /**< MUST be first: base bridge infrastructure */
    nimcp_health_agent_t* health_agent;  /**< Phase 8: instance-level health agent */
    void* game_theory;
    thalamic_router_t* router;
    game_theory_thalamic_config_t config;
    game_theory_thalamic_stats_t stats;
    float attention_weight;
};

game_theory_thalamic_config_t game_theory_thalamic_default_config(void) {
    /* Phase 8: Heartbeat at operation start */
    game_theory_thalamic_bridge_heartbeat("game_theory__game_theory_thalamic", 0.0f);


    game_theory_thalamic_config_t cfg = {
        .enable_attention_gating = true,
        .enable_stakes_boost = true,
        .min_stakes_threshold = 0.25f,
        .coalition_boost = 0.35f
    };
    return cfg;
}

game_theory_thalamic_bridge_t* game_theory_thalamic_bridge_create(void* game_theory, thalamic_router_t* router, const game_theory_thalamic_config_t* config) {
    /* Phase 8: Heartbeat at operation start */
    game_theory_thalamic_bridge_heartbeat("game_theory__create", 0.0f);


    game_theory_thalamic_bridge_t* bridge = nimcp_calloc(1, sizeof(game_theory_thalamic_bridge_t));
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "Failed to allocate bridge");

        return NULL;

    }
    bridge->game_theory = game_theory;
    bridge->router = router;
    bridge->config = config ? *config : game_theory_thalamic_default_config();
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    NIMCP_LOGGING_INFO("Created %s bridge", "game_theory_thalamic");
    return bridge;
}

void game_theory_thalamic_bridge_destroy(game_theory_thalamic_bridge_t* bridge) {
    /* Phase 8: Heartbeat at operation start */
    game_theory_thalamic_bridge_heartbeat("game_theory__destroy", 0.0f);


    if (bridge) nimcp_free(bridge);
}

int game_theory_thalamic_bridge_reset(game_theory_thalamic_bridge_t* bridge) {
    if (!bridge) return -1;
    /* Phase 8: Heartbeat at operation start */
    game_theory_thalamic_bridge_heartbeat("game_theory__reset", 0.0f);


    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return 0;
}

int game_theory_thalamic_route_strategy(game_theory_thalamic_bridge_t* bridge, const game_theory_thalamic_signal_t* signal) {
    if (!bridge || !signal) return -1;
    /* Phase 8: Heartbeat at operation start */
    game_theory_thalamic_bridge_heartbeat("game_theory__game_theory_thalamic", 0.0f);


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
    /* Phase 8: Heartbeat at operation start */
    game_theory_thalamic_bridge_heartbeat("game_theory__game_theory_thalamic", 0.0f);


    bridge->stats.outcomes_processed++;
    return 0;
}

int game_theory_thalamic_set_attention(game_theory_thalamic_bridge_t* bridge, float attention) {
    if (!bridge) return -1;
    /* Phase 8: Heartbeat at operation start */
    game_theory_thalamic_bridge_heartbeat("game_theory__game_theory_thalamic", 0.0f);


    bridge->attention_weight = attention < 0.0f ? 0.0f : (attention > 1.0f ? 1.0f : attention);
    return 0;
}

int game_theory_thalamic_get_attention(const game_theory_thalamic_bridge_t* bridge, float* attention) {
    if (!bridge || !attention) return -1;
    *attention = bridge->attention_weight;
    /* Phase 8: Heartbeat at operation start */
    game_theory_thalamic_bridge_heartbeat("game_theory__game_theory_thalamic", 0.0f);


    return 0;
}

int game_theory_thalamic_bridge_get_stats(const game_theory_thalamic_bridge_t* bridge, game_theory_thalamic_stats_t* stats) {
    if (!bridge || !stats) return -1;
    *stats = bridge->stats;
    /* Phase 8: Heartbeat at operation start */
    game_theory_thalamic_bridge_heartbeat("game_theory__get_stats", 0.0f);


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
    /* Phase 8: Heartbeat at operation start */
    game_theory_thalamic_bridge_heartbeat("game_theory__query_self_knowledge", 0.0f);


    const kg_entity_t* self = kg_reader_get_entity(kg, "Game_Theory_Thalamic_Bridge");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            /* Phase 8: Loop progress heartbeat */
            if ((i & 0xFF) == 0 && self->num_observations > 256) {
                game_theory_thalamic_bridge_heartbeat("game_theory__loop",
                                 (float)(i + 1) / (float)self->num_observations);
            }

            /* GT Thalamic bridge self-knowledge logged */
        }
    }
    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Game_Theory_Thalamic_Bridge");
    if (connections) { kg_relation_list_destroy(connections); }
    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Game_Theory_Thalamic_Bridge");
    if (incoming) { kg_relation_list_destroy(incoming); }
    return self ? 1 : 0;
}

/* ============================================================================
 * Phase 8: Instance-Level Health Agent
 * ============================================================================ */

void game_theory_thalamic_bridge_set_instance_health_agent(game_theory_thalamic_bridge_t* bridge, nimcp_health_agent_t* agent) {
    if (!bridge) {
        NIMCP_THROW(NIMCP_ERROR_NULL_POINTER,
                    "game_theory_thalamic_bridge_set_instance_health_agent: NULL bridge");
        return;
    }
    bridge->health_agent = agent;
}

/* ============================================================================
 * Phase 8: Training Integration (Full Implementation)
 * ============================================================================ */

int game_theory_thalamic_bridge_training_begin(game_theory_thalamic_bridge_t* bridge) {
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "game_theory_thalamic_bridge_training_begin: NULL argument");
        return -1;
    }
    game_theory_thalamic_bridge_heartbeat_instance(bridge->health_agent, "game_theory_thalamic_bridge_training_begin", 0.0f);
    return 0;
}

int game_theory_thalamic_bridge_training_end(game_theory_thalamic_bridge_t* bridge) {
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "game_theory_thalamic_bridge_training_end: NULL argument");
        return -1;
    }
    game_theory_thalamic_bridge_heartbeat_instance(bridge->health_agent, "game_theory_thalamic_bridge_training_end", 1.0f);
    return 0;
}

int game_theory_thalamic_bridge_training_step(game_theory_thalamic_bridge_t* bridge, float progress) {
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "game_theory_thalamic_bridge_training_step: NULL argument");
        return -1;
    }
    if (progress < 0.0f) progress = 0.0f;
    if (progress > 1.0f) progress = 1.0f;
    game_theory_thalamic_bridge_heartbeat_instance(bridge->health_agent, "game_theory_thalamic_bridge_training_step", progress);
    return 0;
}
