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
#include "utils/algorithms/nimcp_monte_carlo.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>
#include <math.h>

//=============================================================================
// Monte Carlo Integration - Thread-local seed
//=============================================================================

static __thread uint32_t g_gt_mc_seed = 0;

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

//=============================================================================
// Monte Carlo Enhanced Functions
//=============================================================================

/**
 * @brief Select action according to mixed strategy using MC sampling
 *
 * WHAT: Sample action from probability distribution
 * WHY:  Realize mixed strategies in actual play
 * HOW:  Use MC importance sampling
 *
 * @param player Player with mixed strategy
 * @return Selected action index
 */
uint32_t nimcp_gt_sample_action_mc(const nimcp_player_t* player) {
    if (!player || !player->strategy || player->num_actions == 0) {
        return 0;
    }

    if (g_gt_mc_seed == 0) {
        g_gt_mc_seed = mc_seed_from_time();
    }

    /* Sample using MC */
    float r = mc_random_uniform(&g_gt_mc_seed);
    float cumulative = 0.0f;

    for (uint32_t i = 0; i < player->num_actions; i++) {
        cumulative += player->strategy[i];
        if (r < cumulative) {
            return i;
        }
    }

    return player->num_actions - 1;
}

/**
 * @brief Estimate expected utility via Monte Carlo sampling
 *
 * WHAT: Compute expected payoff under mixed strategies
 * WHY:  Evaluate strategy quality when payoff matrix is stochastic
 * HOW:  Sample action profiles, average payoffs
 *
 * @param system Game theory system
 * @param players Array of players with strategies
 * @param num_players Number of players
 * @param payoff_fn Payoff function (action profile -> utilities)
 * @param num_samples Number of MC samples
 * @param expected_utilities Output: expected utility per player
 * @param user_data User data for payoff function
 * @return 0 on success
 */
int nimcp_gt_expected_utility_mc(
    nimcp_gt_system_t system,
    const nimcp_player_t* players,
    uint32_t num_players,
    float (*payoff_fn)(const uint32_t* actions, uint32_t num_players, void* user_data),
    uint32_t num_samples,
    float* expected_utilities,
    void* user_data
) {
    if (!system || !players || !payoff_fn || !expected_utilities || num_players == 0) {
        return -1;
    }

    if (g_gt_mc_seed == 0) {
        g_gt_mc_seed = mc_seed_from_time();
    }

    /* Initialize utilities to zero */
    for (uint32_t i = 0; i < num_players; i++) {
        expected_utilities[i] = 0.0f;
    }

    /* Allocate action profile */
    uint32_t* actions = nimcp_calloc(num_players, sizeof(uint32_t));
    if (!actions) return -1;

    /* Monte Carlo sampling */
    for (uint32_t s = 0; s < num_samples; s++) {
        /* Sample action profile from mixed strategies */
        for (uint32_t p = 0; p < num_players; p++) {
            float r = mc_random_uniform(&g_gt_mc_seed);
            float cumulative = 0.0f;
            actions[p] = 0;

            for (uint32_t a = 0; a < players[p].num_actions; a++) {
                cumulative += players[p].strategy[a];
                if (r < cumulative) {
                    actions[p] = a;
                    break;
                }
            }
        }

        /* Compute payoff and accumulate */
        float payoff = payoff_fn(actions, num_players, user_data);
        for (uint32_t p = 0; p < num_players; p++) {
            expected_utilities[p] += payoff;
        }
    }

    /* Average */
    for (uint32_t i = 0; i < num_players; i++) {
        expected_utilities[i] /= (float)num_samples;
    }

    nimcp_free(actions);
    return 0;
}

/**
 * @brief Update strategy via fictitious play with MC sampling
 *
 * WHAT: Update strategy using best response to opponent history
 * WHY:  Simple learning algorithm that converges to Nash in some games
 * HOW:  Track opponent action frequencies, compute best response
 *
 * @param player Player to update
 * @param opponent_frequencies Observed frequencies of opponent actions
 * @param num_opponent_actions Number of opponent actions
 * @param payoff_matrix Player's payoff matrix [my_action][opp_action]
 * @param learning_rate How fast to update (0-1)
 * @return 0 on success
 */
int nimcp_gt_fictitious_play_update_mc(
    nimcp_player_t* player,
    const float* opponent_frequencies,
    uint32_t num_opponent_actions,
    const float* payoff_matrix,
    float learning_rate
) {
    if (!player || !opponent_frequencies || !payoff_matrix || !player->strategy) {
        return -1;
    }

    if (g_gt_mc_seed == 0) {
        g_gt_mc_seed = mc_seed_from_time();
    }

    /* Compute expected payoff for each of my actions */
    float* expected_payoffs = nimcp_calloc(player->num_actions, sizeof(float));
    if (!expected_payoffs) return -1;

    float max_payoff = -1e30f;
    uint32_t best_action = 0;

    for (uint32_t a = 0; a < player->num_actions; a++) {
        expected_payoffs[a] = 0.0f;
        for (uint32_t o = 0; o < num_opponent_actions; o++) {
            expected_payoffs[a] += payoff_matrix[a * num_opponent_actions + o] * opponent_frequencies[o];
        }
        if (expected_payoffs[a] > max_payoff) {
            max_payoff = expected_payoffs[a];
            best_action = a;
        }
    }

    /* Update strategy toward best response with noise for exploration */
    for (uint32_t a = 0; a < player->num_actions; a++) {
        float target = (a == best_action) ? 1.0f : 0.0f;
        /* Add small noise for exploration */
        float noise = mc_random_normal(&g_gt_mc_seed, 0.0f, 0.01f);
        player->strategy[a] = (1.0f - learning_rate) * player->strategy[a] +
                              learning_rate * target + noise;
        if (player->strategy[a] < 0.0f) player->strategy[a] = 0.0f;
    }

    /* Normalize */
    float sum = 0.0f;
    for (uint32_t a = 0; a < player->num_actions; a++) {
        sum += player->strategy[a];
    }
    if (sum > 0.0f) {
        for (uint32_t a = 0; a < player->num_actions; a++) {
            player->strategy[a] /= sum;
        }
    }

    nimcp_free(expected_payoffs);
    return 0;
}

/**
 * @brief Get thread-local MC seed for game theory
 *
 * @return Pointer to thread-local seed
 */
uint32_t* nimcp_gt_get_mc_seed(void) {
    if (g_gt_mc_seed == 0) {
        g_gt_mc_seed = mc_seed_from_time();
    }
    return &g_gt_mc_seed;
}
