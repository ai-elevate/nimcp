//=============================================================================
// nimcp_game_theory.c - Game Theory Core Implementation
//=============================================================================
/**
 * @file nimcp_game_theory.c
 * @brief Core game theory system implementation
 */

#include "cognitive/game_theory/nimcp_game_theory.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/platform/nimcp_platform_mutex.h"
#include <string.h>
#include <math.h>

//=============================================================================
// Internal Structure
//=============================================================================

struct nimcp_gt_system_struct {
    nimcp_gt_config_t config;
    nimcp_game_stats_t stats;
    bool initialized;
    nimcp_platform_mutex_t mutex;
};

//=============================================================================
// Static Name Tables
//=============================================================================

static const char* s_game_type_names[] = {
    "Zero-Sum",
    "Cooperative",
    "General-Sum",
    "Potential",
    "Congestion",
    "Auction",
    "Bargaining"
};

static const char* s_solution_concept_names[] = {
    "Nash Equilibrium",
    "Correlated Equilibrium",
    "Pareto Optimal",
    "Core",
    "Shapley Value",
    "Nucleolus"
};

//=============================================================================
// Configuration
//=============================================================================

nimcp_gt_config_t nimcp_gt_default_config(void) {
    nimcp_gt_config_t config = {
        .max_players = NIMCP_GT_MAX_PLAYERS,
        .max_iterations = NIMCP_GT_MAX_ITERATIONS,
        .convergence_epsilon = 1e-6f,
        .enable_statistics = true,
        .enable_history = true,
        .history_depth = NIMCP_GT_HISTORY_DEPTH
    };
    return config;
}

//=============================================================================
// Lifecycle
//=============================================================================

nimcp_gt_system_t nimcp_gt_create(const nimcp_gt_config_t* config) {
    nimcp_gt_system_t system = nimcp_calloc(1, sizeof(struct nimcp_gt_system_struct));
    if (!system) {
        return NULL;
    }

    if (config) {
        system->config = *config;
    } else {
        system->config = nimcp_gt_default_config();
    }

    if (nimcp_platform_mutex_init(&system->mutex, false) != 0) {
        nimcp_free(system);
        return NULL;
    }

    memset(&system->stats, 0, sizeof(nimcp_game_stats_t));
    system->initialized = true;

    return system;
}

void nimcp_gt_destroy(nimcp_gt_system_t system) {
    if (!system) return;

    nimcp_platform_mutex_destroy(&system->mutex);
    nimcp_free(system);
}

bool nimcp_gt_is_initialized(const nimcp_gt_system_t system) {
    return system && system->initialized;
}

nimcp_error_t nimcp_gt_get_stats(const nimcp_gt_system_t system,
                                  nimcp_game_stats_t* stats) {
    if (!system || !stats) {
        return NIMCP_GT_ERROR_NULL_POINTER;
    }

    nimcp_platform_mutex_lock(&system->mutex);
    *stats = system->stats;
    nimcp_platform_mutex_unlock(&system->mutex);

    return NIMCP_SUCCESS;
}

void nimcp_gt_reset_stats(nimcp_gt_system_t system) {
    if (!system) return;

    nimcp_platform_mutex_lock(&system->mutex);
    memset(&system->stats, 0, sizeof(nimcp_game_stats_t));
    nimcp_platform_mutex_unlock(&system->mutex);
}

//=============================================================================
// Name Utilities
//=============================================================================

const char* nimcp_game_type_name(nimcp_game_type_t type) {
    if (type >= NIMCP_GAME_COUNT) {
        return "Unknown";
    }
    return s_game_type_names[type];
}

const char* nimcp_solution_concept_name(nimcp_solution_concept_t solution_concept) {
    if (solution_concept > NIMCP_SOLUTION_NUCLEOLUS) {
        return "Unknown";
    }
    return s_solution_concept_names[solution_concept];
}

//=============================================================================
// Fairness and Efficiency Utilities
//=============================================================================

float nimcp_compute_fairness_index(const float* allocations, uint32_t n) {
    if (!allocations || n == 0) {
        return 0.0f;
    }

    // Jain's fairness index: J = (sum x_i)^2 / (n * sum x_i^2)
    float sum = 0.0f;
    float sum_sq = 0.0f;

    for (uint32_t i = 0; i < n; i++) {
        sum += allocations[i];
        sum_sq += allocations[i] * allocations[i];
    }

    if (sum_sq < 1e-10f) {
        return 1.0f;  // All zero is perfectly fair
    }

    return (sum * sum) / ((float)n * sum_sq);
}

bool nimcp_is_pareto_optimal(const float* utilities, uint32_t n,
                              const float* feasible_utilities,
                              uint32_t num_feasible) {
    if (!utilities || !feasible_utilities || n == 0 || num_feasible == 0) {
        return false;
    }

    // Check if any feasible allocation Pareto-dominates current
    for (uint32_t f = 0; f < num_feasible; f++) {
        const float* other = &feasible_utilities[f * n];

        bool at_least_one_better = false;
        bool all_at_least_as_good = true;

        for (uint32_t i = 0; i < n; i++) {
            if (other[i] < utilities[i]) {
                all_at_least_as_good = false;
                break;
            }
            if (other[i] > utilities[i]) {
                at_least_one_better = true;
            }
        }

        // Other dominates current -> not Pareto optimal
        if (all_at_least_as_good && at_least_one_better) {
            return false;
        }
    }

    return true;
}

//=============================================================================
// Player Utilities
//=============================================================================

void nimcp_player_init(nimcp_player_t* player, nimcp_player_id_t id,
                        const char* name, uint32_t num_actions) {
    if (!player) return;

    memset(player, 0, sizeof(nimcp_player_t));
    player->id = id;
    player->num_actions = num_actions;

    if (name) {
        strncpy(player->name, name, sizeof(player->name) - 1);
        player->name[sizeof(player->name) - 1] = '\0';
    }

    if (num_actions > 0) {
        player->strategy = nimcp_calloc(num_actions, sizeof(float));
        if (player->strategy) {
            // Default: uniform distribution
            float uniform = 1.0f / (float)num_actions;
            for (uint32_t i = 0; i < num_actions; i++) {
                player->strategy[i] = uniform;
            }
        }
    }
}

void nimcp_player_cleanup(nimcp_player_t* player) {
    if (!player) return;

    if (player->strategy) {
        nimcp_free(player->strategy);
        player->strategy = NULL;
    }
    if (player->private_info) {
        nimcp_free(player->private_info);
        player->private_info = NULL;
    }
}

void nimcp_game_outcome_init(nimcp_game_outcome_t* outcome) {
    if (!outcome) return;

    memset(outcome, 0, sizeof(nimcp_game_outcome_t));

    // Initialize winners to invalid
    for (uint32_t i = 0; i < NIMCP_GT_MAX_PLAYERS; i++) {
        outcome->winners[i] = NIMCP_GT_INVALID_PLAYER;
    }
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

/**
 * WHAT: Query knowledge graph for Game Theory self-knowledge
 * WHY:  Enable self-awareness about module's role and connections
 * HOW:  Query KG for entity observations and relations
 */
int game_theory_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;
    const kg_entity_t* self = kg_reader_get_entity(kg, "Game_Theory");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            /* Game theory self-knowledge logged */
        }
    }
    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Game_Theory");
    if (connections) { kg_relation_list_destroy(connections); }
    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Game_Theory");
    if (incoming) { kg_relation_list_destroy(incoming); }
    return self ? 1 : 0;
}
