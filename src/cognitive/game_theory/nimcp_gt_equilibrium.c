//=============================================================================
// nimcp_gt_equilibrium.c - N-Player Nash Equilibrium Solver Implementation
//=============================================================================
/**
 * @file nimcp_gt_equilibrium.c
 * @brief Nash equilibrium computation algorithms
 */

#include "cognitive/game_theory/nimcp_gt_equilibrium.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/platform/nimcp_platform_mutex.h"
#include "utils/time/nimcp_time.h"
#include "utils/logging/nimcp_logging.h"
#include "api/nimcp_api_exception.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>
#include <math.h>
#include <float.h>

#define LOG_MODULE "equilibrium"

//=============================================================================
// Internal Structure
//=============================================================================

struct nimcp_equilibrium_struct {
    nimcp_equilibrium_config_t config;

    // Payoff matrices for each player
    nimcp_game_matrix_t* payoffs[NIMCP_GT_MAX_PLAYERS];
    bool payoffs_set[NIMCP_GT_MAX_PLAYERS];

    // Computation state
    uint32_t total_strategy_profiles;       /**< Total number of pure strategy combinations */
    nimcp_equilibrium_stats_t stats;

    // Temporary workspace
    uint32_t* temp_profile;                 /**< Scratch space for strategy profile iteration */
    float* temp_probs;                      /**< Scratch space for probability calculations */

    // Thread safety
    nimcp_platform_mutex_t mutex;
};

//=============================================================================
// Static Name Tables
//=============================================================================

static const char* s_algo_names[] = {
    "Auto",
    "Best Response Iteration",
    "Support Enumeration",
    "Lemke-Howson",
    "Fictitious Play"
};

static const char* s_type_names[] = {
    "Pure Strategy",
    "Mixed Strategy",
    "Correlated",
    "Approximate"
};

//=============================================================================
// Forward Declarations
//=============================================================================

static uint32_t compute_total_profiles(const nimcp_equilibrium_config_t* config);
static void increment_profile(uint32_t* profile, const uint32_t* max_strategies,
                              uint32_t num_players);
static float compute_pure_payoff(const nimcp_equilibrium_t ctx, uint32_t player,
                                 const uint32_t* profile);
static float compute_mixed_payoff(const nimcp_equilibrium_t ctx, uint32_t player,
                                  const nimcp_strategy_profile_t* strategies);
static bool solve_indifference_2player(const nimcp_equilibrium_t ctx,
                                       const uint32_t* support_row,
                                       uint32_t support_row_size,
                                       const uint32_t* support_col,
                                       uint32_t support_col_size,
                                       float* row_probs, float* col_probs);
static bool check_no_outside_deviation(const nimcp_equilibrium_t ctx,
                                       const nimcp_strategy_profile_t* strategies);

//=============================================================================
// Configuration
//=============================================================================

nimcp_equilibrium_config_t nimcp_equilibrium_default_config(
    uint32_t num_players,
    const uint32_t* strategies_per_player
) {
    nimcp_equilibrium_config_t config;
    memset(&config, 0, sizeof(config));

    config.algorithm = NIMCP_EQUILIBRIUM_ALGO_AUTO;
    config.num_players = num_players > NIMCP_GT_MAX_PLAYERS ?
                         NIMCP_GT_MAX_PLAYERS : num_players;

    // Set strategies per player
    for (uint32_t i = 0; i < config.num_players; i++) {
        if (strategies_per_player) {
            config.num_strategies[i] = strategies_per_player[i] > NIMCP_GT_MAX_STRATEGIES ?
                                       NIMCP_GT_MAX_STRATEGIES : strategies_per_player[i];
        } else {
            config.num_strategies[i] = 2;  // Default binary strategies
        }
    }

    // Convergence parameters
    config.max_iterations = NIMCP_GT_MAX_ITERATIONS;
    config.convergence_epsilon = 1e-6f;
    config.nash_epsilon = 1e-6f;
    config.learning_rate = 0.1f;

    // Behavior
    config.find_all_equilibria = false;
    config.verify_equilibria = true;
    config.enable_early_termination = true;
    config.timeout_ms = 0;

    return config;
}

//=============================================================================
// Lifecycle
//=============================================================================

nimcp_equilibrium_t nimcp_equilibrium_create(const nimcp_equilibrium_config_t* config) {
    if (!config || config->num_players == 0 || config->num_players > NIMCP_GT_MAX_PLAYERS) {
        return NULL;
    }

    // Validate strategies
    for (uint32_t i = 0; i < config->num_players; i++) {
        if (config->num_strategies[i] == 0 ||
            config->num_strategies[i] > NIMCP_GT_MAX_STRATEGIES) {
            return NULL;
        }
    }

    nimcp_equilibrium_t ctx = nimcp_calloc(1, sizeof(struct nimcp_equilibrium_struct));
    if (!ctx) {
        return NULL;
    }

    ctx->config = *config;
    ctx->total_strategy_profiles = compute_total_profiles(config);

    // Initialize payoff matrices
    for (uint32_t i = 0; i < config->num_players; i++) {
        ctx->payoffs[i] = NULL;
        ctx->payoffs_set[i] = false;
    }

    // Allocate temp workspace
    ctx->temp_profile = nimcp_calloc(config->num_players, sizeof(uint32_t));
    if (!ctx->temp_profile) {
        nimcp_free(ctx);
        return NULL;
    }

    // Allocate temp probabilities (for largest player's strategy set)
    uint32_t max_strategies = 0;
    for (uint32_t i = 0; i < config->num_players; i++) {
        if (config->num_strategies[i] > max_strategies) {
            max_strategies = config->num_strategies[i];
        }
    }
    ctx->temp_probs = nimcp_calloc(max_strategies, sizeof(float));
    if (!ctx->temp_probs) {
        nimcp_free(ctx->temp_profile);
        nimcp_free(ctx);
        return NULL;
    }

    // Initialize mutex
    if (nimcp_platform_mutex_init(&ctx->mutex, false) != 0) {
        nimcp_free(ctx->temp_probs);
        nimcp_free(ctx->temp_profile);
        nimcp_free(ctx);
        return NULL;
    }

    // Initialize stats
    memset(&ctx->stats, 0, sizeof(nimcp_equilibrium_stats_t));
    ctx->stats.status = NIMCP_CONVERGENCE_NOT_STARTED;

    return ctx;
}

void nimcp_equilibrium_destroy(nimcp_equilibrium_t ctx) {
    if (!ctx) return;

    nimcp_platform_mutex_destroy(&ctx->mutex);

    // Free payoff matrices
    for (uint32_t i = 0; i < ctx->config.num_players; i++) {
        if (ctx->payoffs[i]) {
            nimcp_game_matrix_destroy(ctx->payoffs[i]);
        }
    }

    nimcp_free(ctx->temp_probs);
    nimcp_free(ctx->temp_profile);
    nimcp_free(ctx);
}

//=============================================================================
// Game Matrix Functions
//=============================================================================

nimcp_game_matrix_t* nimcp_game_matrix_create(
    uint32_t num_players,
    const uint32_t* strategies_per_player
) {
    if (!strategies_per_player || num_players == 0 ||
        num_players > NIMCP_GT_MAX_PLAYERS) {
        return NULL;
    }

    nimcp_game_matrix_t* matrix = nimcp_calloc(1, sizeof(nimcp_game_matrix_t));
    if (!matrix) {
        return NULL;
    }

    matrix->num_players = num_players;

    // Compute strides and total cells
    matrix->total_cells = 1;
    for (uint32_t i = 0; i < num_players; i++) {
        matrix->num_strategies[i] = strategies_per_player[i];
        matrix->total_cells *= strategies_per_player[i];
    }

    // Compute strides (row-major: last dimension changes fastest)
    uint32_t stride = 1;
    for (int i = (int)num_players - 1; i >= 0; i--) {
        matrix->strides[i] = stride;
        stride *= strategies_per_player[i];
    }

    // Allocate data
    matrix->data = nimcp_calloc(matrix->total_cells, sizeof(float));
    if (!matrix->data) {
        nimcp_free(matrix);
        return NULL;
    }

    return matrix;
}

void nimcp_game_matrix_destroy(nimcp_game_matrix_t* matrix) {
    if (!matrix) return;
    nimcp_free(matrix->data);
    nimcp_free(matrix);
}

float nimcp_game_matrix_get(
    const nimcp_game_matrix_t* matrix,
    const uint32_t* strategy_profile
) {
    if (!matrix || !strategy_profile) return 0.0f;

    uint32_t index = 0;
    for (uint32_t i = 0; i < matrix->num_players; i++) {
        if (strategy_profile[i] >= matrix->num_strategies[i]) {
            return 0.0f;  // Invalid strategy
        }
        index += strategy_profile[i] * matrix->strides[i];
    }

    return matrix->data[index];
}

void nimcp_game_matrix_set(
    nimcp_game_matrix_t* matrix,
    const uint32_t* strategy_profile,
    float payoff
) {
    if (!matrix || !strategy_profile) return;

    uint32_t index = 0;
    for (uint32_t i = 0; i < matrix->num_players; i++) {
        if (strategy_profile[i] >= matrix->num_strategies[i]) {
            return;  // Invalid strategy
        }
        index += strategy_profile[i] * matrix->strides[i];
    }

    matrix->data[index] = payoff;
}

//=============================================================================
// Game Setup
//=============================================================================

nimcp_error_t nimcp_equilibrium_set_payoffs(
    nimcp_equilibrium_t ctx,
    uint32_t player,
    const float* payoffs,
    uint32_t size
) {
    if (!ctx || !payoffs) {
        return NIMCP_GT_ERROR_NULL_POINTER;
    }

    if (player >= ctx->config.num_players) {
        return NIMCP_GT_ERROR_INVALID_PARAMETER;
    }

    nimcp_platform_mutex_lock(&ctx->mutex);

    // Create matrix if not exists
    if (!ctx->payoffs[player]) {
        ctx->payoffs[player] = nimcp_game_matrix_create(
            ctx->config.num_players,
            ctx->config.num_strategies
        );
        if (!ctx->payoffs[player]) {
            nimcp_platform_mutex_unlock(&ctx->mutex);
            return NIMCP_GT_ERROR_NO_MEMORY;
        }
    }

    // Validate size
    if (size != ctx->payoffs[player]->total_cells) {
        nimcp_platform_mutex_unlock(&ctx->mutex);
        return NIMCP_GT_ERROR_INVALID_PARAMETER;
    }

    // Copy payoffs
    memcpy(ctx->payoffs[player]->data, payoffs, size * sizeof(float));
    ctx->payoffs_set[player] = true;

    nimcp_platform_mutex_unlock(&ctx->mutex);
    return NIMCP_SUCCESS;
}

nimcp_error_t nimcp_equilibrium_set_bimatrix(
    nimcp_equilibrium_t ctx,
    const float* row_payoffs,
    const float* col_payoffs,
    uint32_t rows,
    uint32_t cols
) {
    if (!ctx || !row_payoffs || !col_payoffs) {
        return NIMCP_GT_ERROR_NULL_POINTER;
    }

    if (ctx->config.num_players != 2) {
        return NIMCP_GT_ERROR_INVALID_PARAMETER;
    }

    uint32_t size = rows * cols;

    nimcp_error_t err = nimcp_equilibrium_set_payoffs(ctx, 0, row_payoffs, size);
    if (err != NIMCP_SUCCESS) return err;

    return nimcp_equilibrium_set_payoffs(ctx, 1, col_payoffs, size);
}

//=============================================================================
// Pure Strategy Nash Equilibrium
//=============================================================================

nimcp_error_t nimcp_equilibrium_find_pure_nash(
    nimcp_equilibrium_t ctx,
    nimcp_equilibrium_result_t* result
) {
    if (!ctx || !result) {
        return NIMCP_GT_ERROR_NULL_POINTER;
    }

    nimcp_platform_mutex_lock(&ctx->mutex);

    // Verify all payoffs set
    for (uint32_t i = 0; i < ctx->config.num_players; i++) {
        if (!ctx->payoffs_set[i]) {
            nimcp_platform_mutex_unlock(&ctx->mutex);
            return NIMCP_GT_ERROR_NOT_INITIALIZED;
        }
    }

    uint64_t start_time = nimcp_time_get_ms();
    ctx->stats.status = NIMCP_CONVERGENCE_IN_PROGRESS;
    ctx->stats.iterations_completed = 0;

    uint32_t n = ctx->config.num_players;
    memset(ctx->temp_profile, 0, n * sizeof(uint32_t));

    memset(result, 0, sizeof(nimcp_equilibrium_result_t));

    // Iterate through all pure strategy profiles
    for (uint32_t p = 0; p < ctx->total_strategy_profiles; p++) {
        ctx->stats.iterations_completed++;

        // Check timeout
        if (ctx->config.timeout_ms > 0) {
            uint64_t elapsed = nimcp_time_get_ms() - start_time;
            if (elapsed >= ctx->config.timeout_ms) {
                ctx->stats.status = NIMCP_CONVERGENCE_MAX_ITERATIONS;
                ctx->stats.compute_time_ms = elapsed;
                nimcp_platform_mutex_unlock(&ctx->mutex);
                return NIMCP_GT_ERROR_TIMEOUT;
            }
        }

        // Check if this profile is a Nash equilibrium
        bool is_nash = true;

        for (uint32_t player = 0; player < n; player++) {
            // Find best response for this player
            uint32_t current_strategy = ctx->temp_profile[player];
            float current_payoff = compute_pure_payoff(ctx, player, ctx->temp_profile);
            float best_payoff = current_payoff;

            // Try all alternative strategies
            for (uint32_t s = 0; s < ctx->config.num_strategies[player]; s++) {
                if (s == current_strategy) continue;

                ctx->temp_profile[player] = s;
                float payoff = compute_pure_payoff(ctx, player, ctx->temp_profile);
                if (payoff > best_payoff + ctx->config.nash_epsilon) {
                    best_payoff = payoff;
                    is_nash = false;
                    break;
                }
            }

            // Restore original strategy
            ctx->temp_profile[player] = current_strategy;

            if (!is_nash) break;
        }

        if (is_nash) {
            // Found a Nash equilibrium
            result->type = NIMCP_EQUILIBRIUM_TYPE_PURE;
            nimcp_strategy_profile_init_pure(&result->strategies, n, ctx->temp_profile);

            // Compute payoffs
            for (uint32_t i = 0; i < n; i++) {
                result->payoffs[i] = compute_pure_payoff(ctx, i, ctx->temp_profile);
                result->social_welfare += result->payoffs[i];
            }

            result->epsilon = 0.0f;
            result->is_verified = true;

            ctx->stats.status = NIMCP_CONVERGENCE_CONVERGED;
            ctx->stats.equilibria_found = 1;
            ctx->stats.compute_time_ms = nimcp_time_get_ms() - start_time;

            nimcp_platform_mutex_unlock(&ctx->mutex);
            return NIMCP_SUCCESS;
        }

        // Move to next profile
        increment_profile(ctx->temp_profile, ctx->config.num_strategies, n);
    }

    // No pure strategy equilibrium found
    ctx->stats.status = NIMCP_CONVERGENCE_FAILED;
    ctx->stats.equilibria_found = 0;
    ctx->stats.compute_time_ms = nimcp_time_get_ms() - start_time;

    nimcp_platform_mutex_unlock(&ctx->mutex);
    return NIMCP_GT_ERROR_NO_EQUILIBRIUM;
}

//=============================================================================
// Mixed Strategy Nash Equilibrium
//=============================================================================

nimcp_error_t nimcp_equilibrium_find_mixed_nash(
    nimcp_equilibrium_t ctx,
    nimcp_equilibrium_result_t* result
) {
    if (!ctx || !result) {
        return NIMCP_GT_ERROR_NULL_POINTER;
    }

    nimcp_platform_mutex_lock(&ctx->mutex);

    // Verify all payoffs set
    for (uint32_t i = 0; i < ctx->config.num_players; i++) {
        if (!ctx->payoffs_set[i]) {
            nimcp_platform_mutex_unlock(&ctx->mutex);
            return NIMCP_GT_ERROR_NOT_INITIALIZED;
        }
    }

    uint64_t start_time = nimcp_time_get_ms();
    ctx->stats.status = NIMCP_CONVERGENCE_IN_PROGRESS;
    ctx->stats.iterations_completed = 0;

    memset(result, 0, sizeof(nimcp_equilibrium_result_t));

    // For 2-player games with small strategy sets, use support enumeration
    if (ctx->config.num_players == 2 &&
        ctx->config.num_strategies[0] <= NIMCP_GT_SUPPORT_ENUM_THRESHOLD &&
        ctx->config.num_strategies[1] <= NIMCP_GT_SUPPORT_ENUM_THRESHOLD) {

        uint32_t m = ctx->config.num_strategies[0];
        uint32_t n = ctx->config.num_strategies[1];

        // Allocate mixed strategy arrays
        nimcp_error_t err = nimcp_strategy_profile_init_mixed(
            &result->strategies,
            2,
            ctx->config.num_strategies
        );
        if (err != NIMCP_SUCCESS) {
            nimcp_platform_mutex_unlock(&ctx->mutex);
            return err;
        }

        // Try all support combinations
        uint32_t* support_row = nimcp_calloc(m, sizeof(uint32_t));
        uint32_t* support_col = nimcp_calloc(n, sizeof(uint32_t));
        float* row_probs = nimcp_calloc(m, sizeof(float));
        float* col_probs = nimcp_calloc(n, sizeof(float));

        if (!support_row || !support_col || !row_probs || !col_probs) {
            nimcp_free(support_row);
            nimcp_free(support_col);
            nimcp_free(row_probs);
            nimcp_free(col_probs);
            nimcp_strategy_profile_cleanup(&result->strategies);
            nimcp_platform_mutex_unlock(&ctx->mutex);
            return NIMCP_GT_ERROR_NO_MEMORY;
        }

        bool found = false;

        // Enumerate all non-empty support combinations
        // Using bitmask: 1 to 2^m - 1 for row support, 1 to 2^n - 1 for col support
        for (uint32_t row_mask = 1; row_mask < (1u << m) && !found; row_mask++) {
            uint32_t row_support_size = 0;
            for (uint32_t i = 0; i < m; i++) {
                if (row_mask & (1u << i)) {
                    support_row[row_support_size++] = i;
                }
            }

            for (uint32_t col_mask = 1; col_mask < (1u << n) && !found; col_mask++) {
                uint32_t col_support_size = 0;
                for (uint32_t j = 0; j < n; j++) {
                    if (col_mask & (1u << j)) {
                        support_col[col_support_size++] = j;
                    }
                }

                ctx->stats.iterations_completed++;

                // Check timeout
                if (ctx->config.timeout_ms > 0) {
                    uint64_t elapsed = nimcp_time_get_ms() - start_time;
                    if (elapsed >= ctx->config.timeout_ms) {
                        nimcp_free(support_row);
                        nimcp_free(support_col);
                        nimcp_free(row_probs);
                        nimcp_free(col_probs);
                        nimcp_strategy_profile_cleanup(&result->strategies);
                        ctx->stats.status = NIMCP_CONVERGENCE_MAX_ITERATIONS;
                        ctx->stats.compute_time_ms = elapsed;
                        nimcp_platform_mutex_unlock(&ctx->mutex);
                        return NIMCP_GT_ERROR_TIMEOUT;
                    }
                }

                // Try to solve indifference conditions
                memset(row_probs, 0, m * sizeof(float));
                memset(col_probs, 0, n * sizeof(float));

                if (solve_indifference_2player(ctx, support_row, row_support_size,
                                               support_col, col_support_size,
                                               row_probs, col_probs)) {

                    // Copy probabilities to result
                    memcpy(result->strategies.mixed_strategies[0], row_probs, m * sizeof(float));
                    memcpy(result->strategies.mixed_strategies[1], col_probs, n * sizeof(float));

                    // Verify no profitable deviation outside support
                    if (check_no_outside_deviation(ctx, &result->strategies)) {
                        found = true;
                        result->type = NIMCP_EQUILIBRIUM_TYPE_MIXED;
                        result->support_sizes[0] = row_support_size;
                        result->support_sizes[1] = col_support_size;
                    }
                }
            }
        }

        nimcp_free(support_row);
        nimcp_free(support_col);
        nimcp_free(row_probs);
        nimcp_free(col_probs);

        if (found) {
            // Compute expected payoffs
            result->payoffs[0] = compute_mixed_payoff(ctx, 0, &result->strategies);
            result->payoffs[1] = compute_mixed_payoff(ctx, 1, &result->strategies);
            result->social_welfare = result->payoffs[0] + result->payoffs[1];
            result->epsilon = 0.0f;
            result->is_verified = true;

            ctx->stats.status = NIMCP_CONVERGENCE_CONVERGED;
            ctx->stats.equilibria_found = 1;
            ctx->stats.compute_time_ms = nimcp_time_get_ms() - start_time;

            nimcp_platform_mutex_unlock(&ctx->mutex);
            return NIMCP_SUCCESS;
        }

        nimcp_strategy_profile_cleanup(&result->strategies);
    }

    // Fall back to Lemke-Howson for 2-player games
    if (ctx->config.num_players == 2) {
        nimcp_platform_mutex_unlock(&ctx->mutex);
        return nimcp_equilibrium_lemke_howson(ctx, result);
    }

    // For N-player games, return error (not yet implemented)
    ctx->stats.status = NIMCP_CONVERGENCE_FAILED;
    ctx->stats.compute_time_ms = nimcp_time_get_ms() - start_time;

    nimcp_platform_mutex_unlock(&ctx->mutex);
    return NIMCP_GT_ERROR_NO_EQUILIBRIUM;
}

//=============================================================================
// Lemke-Howson Algorithm (2-player bimatrix games)
//=============================================================================

nimcp_error_t nimcp_equilibrium_lemke_howson(
    nimcp_equilibrium_t ctx,
    nimcp_equilibrium_result_t* result
) {
    if (!ctx || !result) {
        return NIMCP_GT_ERROR_NULL_POINTER;
    }

    if (ctx->config.num_players != 2) {
        return NIMCP_GT_ERROR_INVALID_PARAMETER;
    }

    nimcp_platform_mutex_lock(&ctx->mutex);

    // Verify all payoffs set
    for (uint32_t i = 0; i < 2; i++) {
        if (!ctx->payoffs_set[i]) {
            nimcp_platform_mutex_unlock(&ctx->mutex);
            return NIMCP_GT_ERROR_NOT_INITIALIZED;
        }
    }

    uint64_t start_time = nimcp_time_get_ms();
    ctx->stats.status = NIMCP_CONVERGENCE_IN_PROGRESS;

    uint32_t m = ctx->config.num_strategies[0];
    uint32_t n = ctx->config.num_strategies[1];

    memset(result, 0, sizeof(nimcp_equilibrium_result_t));

    // Allocate workspace for the algorithm
    // Tableau size: (m + n + 1) x (m + n + 2)
    uint32_t tableau_rows = m + n + 1;
    uint32_t tableau_cols = m + n + 2;

    float* tableau = nimcp_calloc(tableau_rows * tableau_cols, sizeof(float));
    int* basis = nimcp_calloc(tableau_rows, sizeof(int));
    bool* in_basis = nimcp_calloc(m + n, sizeof(bool));

    if (!tableau || !basis || !in_basis) {
        nimcp_free(tableau);
        nimcp_free(basis);
        nimcp_free(in_basis);
        ctx->stats.status = NIMCP_CONVERGENCE_FAILED;
        nimcp_platform_mutex_unlock(&ctx->mutex);
        return NIMCP_GT_ERROR_NO_MEMORY;
    }

    // Build the LCP tableau
    // We need to shift payoffs to be strictly positive
    float min_payoff = FLT_MAX;
    for (uint32_t i = 0; i < m; i++) {
        for (uint32_t j = 0; j < n; j++) {
            uint32_t profile[2] = {i, j};
            float p0 = nimcp_game_matrix_get(ctx->payoffs[0], profile);
            float p1 = nimcp_game_matrix_get(ctx->payoffs[1], profile);
            if (p0 < min_payoff) min_payoff = p0;
            if (p1 < min_payoff) min_payoff = p1;
        }
    }
    float shift = (min_payoff < 1.0f) ? (2.0f - min_payoff) : 1.0f;

    // Initialize tableau
    // Format: [slack | strategy vars | RHS]
    // First m rows: player 1 constraints
    // Next n rows: player 2 constraints
    // Last row: probability sum constraints

    // Player 1 constraints: y^T * B^T <= 1 (shifted to be positive)
    for (uint32_t i = 0; i < m; i++) {
        // Slack variable coefficient
        tableau[i * tableau_cols + i] = 1.0f;

        // Strategy variable coefficients (B transposed)
        for (uint32_t j = 0; j < n; j++) {
            uint32_t profile[2] = {i, j};
            float payoff = nimcp_game_matrix_get(ctx->payoffs[1], profile) + shift;
            tableau[i * tableau_cols + (m + j)] = payoff;
        }

        // RHS
        tableau[i * tableau_cols + (m + n)] = 1.0f;
        tableau[i * tableau_cols + (m + n + 1)] = 0.0f;

        basis[i] = (int)i;  // Slack variable in basis
    }

    // Player 2 constraints: A * x <= 1 (shifted)
    for (uint32_t j = 0; j < n; j++) {
        uint32_t row = m + j;

        // Slack variable coefficient
        tableau[row * tableau_cols + row] = 1.0f;

        // Strategy variable coefficients (A)
        for (uint32_t i = 0; i < m; i++) {
            uint32_t profile[2] = {i, j};
            float payoff = nimcp_game_matrix_get(ctx->payoffs[0], profile) + shift;
            tableau[row * tableau_cols + i] = payoff;
        }

        // RHS
        tableau[row * tableau_cols + (m + n)] = 1.0f;
        tableau[row * tableau_cols + (m + n + 1)] = 0.0f;

        basis[row] = (int)row;  // Slack variable in basis
    }

    // Initialize basis tracking
    for (uint32_t i = 0; i < m + n; i++) {
        in_basis[i] = true;  // All slacks start in basis
    }

    // Lemke-Howson pivot loop
    // Start by dropping label 0 (first strategy of player 1)
    int entering = 0;  // Label to enter
    int max_pivots = (int)(ctx->config.max_iterations);
    bool found = false;

    for (int pivot = 0; pivot < max_pivots && !found; pivot++) {
        ctx->stats.iterations_completed = (uint32_t)(pivot + 1);

        // Check timeout
        if (ctx->config.timeout_ms > 0) {
            uint64_t elapsed = nimcp_time_get_ms() - start_time;
            if (elapsed >= ctx->config.timeout_ms) {
                nimcp_free(tableau);
                nimcp_free(basis);
                nimcp_free(in_basis);
                ctx->stats.status = NIMCP_CONVERGENCE_MAX_ITERATIONS;
                ctx->stats.compute_time_ms = elapsed;
                nimcp_platform_mutex_unlock(&ctx->mutex);
                return NIMCP_GT_ERROR_TIMEOUT;
            }
        }

        // Find row to pivot (minimum ratio test)
        int pivot_row = -1;
        float min_ratio = FLT_MAX;

        for (uint32_t row = 0; row < m + n; row++) {
            float coeff = tableau[row * tableau_cols + (uint32_t)entering];
            if (coeff > 1e-10f) {
                float ratio = tableau[row * tableau_cols + (m + n)] / coeff;
                if (ratio >= 0 && ratio < min_ratio) {
                    min_ratio = ratio;
                    pivot_row = (int)row;
                }
            }
        }

        if (pivot_row < 0) {
            // Unbounded - no valid pivot found
            break;
        }

        // Perform pivot
        int leaving = basis[pivot_row];
        float pivot_val = tableau[(uint32_t)pivot_row * tableau_cols + (uint32_t)entering];

        // Normalize pivot row
        for (uint32_t col = 0; col < tableau_cols; col++) {
            tableau[(uint32_t)pivot_row * tableau_cols + col] /= pivot_val;
        }

        // Eliminate from other rows
        for (uint32_t row = 0; row < m + n; row++) {
            if (row != (uint32_t)pivot_row) {
                float factor = tableau[row * tableau_cols + (uint32_t)entering];
                for (uint32_t col = 0; col < tableau_cols; col++) {
                    tableau[row * tableau_cols + col] -=
                        factor * tableau[(uint32_t)pivot_row * tableau_cols + col];
                }
            }
        }

        // Update basis
        basis[pivot_row] = entering;
        in_basis[entering] = true;
        in_basis[leaving] = false;

        // Check if we've returned to label 0
        if (leaving == 0) {
            found = true;
        } else {
            // Determine next entering variable (complement of leaving)
            // Labels 0..m-1 for player 1 strategies, m..m+n-1 for player 2
            if (leaving < (int)m) {
                entering = (int)m + leaving;  // Corresponding player 2 slack
            } else {
                entering = leaving - (int)m;  // Corresponding player 1 slack
            }
        }
    }

    if (found) {
        // Extract solution from tableau
        nimcp_error_t err = nimcp_strategy_profile_init_mixed(
            &result->strategies,
            2,
            ctx->config.num_strategies
        );

        if (err != NIMCP_SUCCESS) {
            nimcp_free(tableau);
            nimcp_free(basis);
            nimcp_free(in_basis);
            nimcp_platform_mutex_unlock(&ctx->mutex);
            return err;
        }

        // Read probabilities from basis values
        float sum_x = 0.0f, sum_y = 0.0f;

        for (uint32_t row = 0; row < m + n; row++) {
            int var = basis[row];
            float val = tableau[row * tableau_cols + (m + n)];

            if (var >= 0 && var < (int)m) {
                // Player 1 strategy variable
                result->strategies.mixed_strategies[0][var] = val;
                sum_x += val;
            } else if (var >= (int)m && var < (int)(m + n)) {
                // Player 2 strategy variable
                result->strategies.mixed_strategies[1][var - m] = val;
                sum_y += val;
            }
        }

        // Normalize probabilities
        if (sum_x > 1e-10f) {
            for (uint32_t i = 0; i < m; i++) {
                result->strategies.mixed_strategies[0][i] /= sum_x;
            }
        }
        if (sum_y > 1e-10f) {
            for (uint32_t j = 0; j < n; j++) {
                result->strategies.mixed_strategies[1][j] /= sum_y;
            }
        }

        result->type = NIMCP_EQUILIBRIUM_TYPE_MIXED;

        // Compute expected payoffs
        result->payoffs[0] = compute_mixed_payoff(ctx, 0, &result->strategies);
        result->payoffs[1] = compute_mixed_payoff(ctx, 1, &result->strategies);
        result->social_welfare = result->payoffs[0] + result->payoffs[1];

        // Count support sizes
        for (uint32_t i = 0; i < m; i++) {
            if (result->strategies.mixed_strategies[0][i] > 1e-6f) {
                result->support_sizes[0]++;
            }
        }
        for (uint32_t j = 0; j < n; j++) {
            if (result->strategies.mixed_strategies[1][j] > 1e-6f) {
                result->support_sizes[1]++;
            }
        }

        result->epsilon = 0.0f;
        result->is_verified = true;

        ctx->stats.status = NIMCP_CONVERGENCE_CONVERGED;
        ctx->stats.equilibria_found = 1;
    } else {
        ctx->stats.status = NIMCP_CONVERGENCE_FAILED;
        ctx->stats.equilibria_found = 0;
    }

    nimcp_free(tableau);
    nimcp_free(basis);
    nimcp_free(in_basis);

    ctx->stats.compute_time_ms = nimcp_time_get_ms() - start_time;

    nimcp_platform_mutex_unlock(&ctx->mutex);

    return found ? NIMCP_SUCCESS : NIMCP_GT_ERROR_NO_EQUILIBRIUM;
}

//=============================================================================
// Find All Equilibria
//=============================================================================

nimcp_error_t nimcp_equilibrium_find_all(
    nimcp_equilibrium_t ctx,
    nimcp_equilibrium_result_t* results,
    uint32_t max_results,
    uint32_t* num_found
) {
    if (!ctx || !results || !num_found) {
        return NIMCP_GT_ERROR_NULL_POINTER;
    }

    *num_found = 0;

    // First, find all pure strategy equilibria
    nimcp_platform_mutex_lock(&ctx->mutex);

    for (uint32_t i = 0; i < ctx->config.num_players; i++) {
        if (!ctx->payoffs_set[i]) {
            nimcp_platform_mutex_unlock(&ctx->mutex);
            return NIMCP_GT_ERROR_NOT_INITIALIZED;
        }
    }

    uint32_t n = ctx->config.num_players;
    memset(ctx->temp_profile, 0, n * sizeof(uint32_t));

    // Check all pure strategy profiles
    for (uint32_t p = 0; p < ctx->total_strategy_profiles && *num_found < max_results; p++) {
        bool is_nash = true;

        for (uint32_t player = 0; player < n && is_nash; player++) {
            uint32_t current_strategy = ctx->temp_profile[player];
            float current_payoff = compute_pure_payoff(ctx, player, ctx->temp_profile);

            for (uint32_t s = 0; s < ctx->config.num_strategies[player]; s++) {
                if (s == current_strategy) continue;

                ctx->temp_profile[player] = s;
                float payoff = compute_pure_payoff(ctx, player, ctx->temp_profile);
                if (payoff > current_payoff + ctx->config.nash_epsilon) {
                    is_nash = false;
                    break;
                }
            }
            ctx->temp_profile[player] = current_strategy;
        }

        if (is_nash) {
            nimcp_equilibrium_result_t* res = &results[*num_found];
            memset(res, 0, sizeof(nimcp_equilibrium_result_t));

            res->type = NIMCP_EQUILIBRIUM_TYPE_PURE;
            nimcp_strategy_profile_init_pure(&res->strategies, n, ctx->temp_profile);

            for (uint32_t i = 0; i < n; i++) {
                res->payoffs[i] = compute_pure_payoff(ctx, i, ctx->temp_profile);
                res->social_welfare += res->payoffs[i];
            }

            res->epsilon = 0.0f;
            res->is_verified = true;

            (*num_found)++;
        }

        increment_profile(ctx->temp_profile, ctx->config.num_strategies, n);
    }

    nimcp_platform_mutex_unlock(&ctx->mutex);

    // For 2-player games, also find mixed equilibria via Lemke-Howson
    if (ctx->config.num_players == 2 && *num_found < max_results) {
        nimcp_equilibrium_result_t mixed_result;
        nimcp_error_t err = nimcp_equilibrium_lemke_howson(ctx, &mixed_result);
        if (err == NIMCP_SUCCESS) {
            // Check if this is different from pure equilibria
            bool is_new = true;
            if (mixed_result.support_sizes[0] == 1 && mixed_result.support_sizes[1] == 1) {
                // It's essentially pure, check if already found
                for (uint32_t i = 0; i < *num_found; i++) {
                    if (results[i].type == NIMCP_EQUILIBRIUM_TYPE_PURE) {
                        // Compare pure strategies
                        is_new = false;  // Simplified - could do proper comparison
                        break;
                    }
                }
            }

            if (is_new) {
                results[*num_found] = mixed_result;
                (*num_found)++;
            } else {
                nimcp_strategy_profile_cleanup(&mixed_result.strategies);
            }
        }
    }

    return (*num_found > 0) ? NIMCP_SUCCESS : NIMCP_GT_ERROR_NO_EQUILIBRIUM;
}

//=============================================================================
// Best Response Functions
//=============================================================================

nimcp_error_t nimcp_equilibrium_best_response(
    nimcp_equilibrium_t ctx,
    uint32_t player,
    const nimcp_strategy_profile_t* opponent_strategies,
    uint32_t* best_strategy,
    float* best_payoff
) {
    if (!ctx || !opponent_strategies || !best_strategy) {
        return NIMCP_GT_ERROR_NULL_POINTER;
    }

    if (player >= ctx->config.num_players) {
        return NIMCP_GT_ERROR_INVALID_PARAMETER;
    }

    nimcp_platform_mutex_lock(&ctx->mutex);

    if (!ctx->payoffs_set[player]) {
        nimcp_platform_mutex_unlock(&ctx->mutex);
        return NIMCP_GT_ERROR_NOT_INITIALIZED;
    }

    float max_payoff = -FLT_MAX;
    uint32_t best_s = 0;

    // Create strategy profile for evaluation
    nimcp_strategy_profile_t full_profile;
    nimcp_error_t err = nimcp_strategy_profile_copy(&full_profile, opponent_strategies);
    if (err != NIMCP_SUCCESS) {
        nimcp_platform_mutex_unlock(&ctx->mutex);
        return err;
    }

    // Try each strategy for this player
    for (uint32_t s = 0; s < ctx->config.num_strategies[player]; s++) {
        // Set this player's strategy
        if (full_profile.type == NIMCP_STRATEGY_PURE) {
            full_profile.pure_strategies[player] = s;
        } else {
            // For mixed opponent strategies, compute expected payoff
            memset(full_profile.mixed_strategies[player], 0,
                   ctx->config.num_strategies[player] * sizeof(float));
            full_profile.mixed_strategies[player][s] = 1.0f;
        }

        float payoff;
        if (full_profile.type == NIMCP_STRATEGY_PURE) {
            payoff = compute_pure_payoff(ctx, player, full_profile.pure_strategies);
        } else {
            payoff = compute_mixed_payoff(ctx, player, &full_profile);
        }

        if (payoff > max_payoff) {
            max_payoff = payoff;
            best_s = s;
        }
    }

    *best_strategy = best_s;
    if (best_payoff) {
        *best_payoff = max_payoff;
    }

    nimcp_strategy_profile_cleanup(&full_profile);
    nimcp_platform_mutex_unlock(&ctx->mutex);

    return NIMCP_SUCCESS;
}

nimcp_error_t nimcp_equilibrium_best_response_mixed(
    nimcp_equilibrium_t ctx,
    uint32_t player,
    const nimcp_strategy_profile_t* opponent_strategies,
    float* best_response,
    float* best_payoff
) {
    if (!ctx || !opponent_strategies || !best_response) {
        return NIMCP_GT_ERROR_NULL_POINTER;
    }

    // Find pure best response first
    uint32_t best_s;
    float max_payoff;
    nimcp_error_t err = nimcp_equilibrium_best_response(ctx, player, opponent_strategies,
                                                         &best_s, &max_payoff);
    if (err != NIMCP_SUCCESS) {
        return err;
    }

    // Set best response to pure strategy on best_s
    memset(best_response, 0, ctx->config.num_strategies[player] * sizeof(float));
    best_response[best_s] = 1.0f;

    if (best_payoff) {
        *best_payoff = max_payoff;
    }

    return NIMCP_SUCCESS;
}

//=============================================================================
// Verification Functions
//=============================================================================

bool nimcp_equilibrium_is_nash(
    nimcp_equilibrium_t ctx,
    const nimcp_strategy_profile_t* strategies,
    float epsilon
) {
    if (!ctx || !strategies) {
        return false;
    }

    float regrets[NIMCP_GT_MAX_PLAYERS];
    nimcp_error_t err = nimcp_equilibrium_compute_regret(ctx, strategies, regrets);
    if (err != NIMCP_SUCCESS) {
        return false;
    }

    for (uint32_t i = 0; i < ctx->config.num_players; i++) {
        if (regrets[i] > epsilon) {
            return false;
        }
    }

    return true;
}

nimcp_error_t nimcp_equilibrium_compute_regret(
    nimcp_equilibrium_t ctx,
    const nimcp_strategy_profile_t* strategies,
    float* regrets
) {
    if (!ctx || !strategies || !regrets) {
        return NIMCP_GT_ERROR_NULL_POINTER;
    }

    nimcp_platform_mutex_lock(&ctx->mutex);

    for (uint32_t player = 0; player < ctx->config.num_players; player++) {
        if (!ctx->payoffs_set[player]) {
            nimcp_platform_mutex_unlock(&ctx->mutex);
            return NIMCP_GT_ERROR_NOT_INITIALIZED;
        }

        // Current payoff
        float current_payoff;
        if (strategies->type == NIMCP_STRATEGY_PURE) {
            current_payoff = compute_pure_payoff(ctx, player, strategies->pure_strategies);
        } else {
            current_payoff = compute_mixed_payoff(ctx, player, strategies);
        }

        // Best response payoff
        uint32_t best_s;
        float best_payoff;
        nimcp_platform_mutex_unlock(&ctx->mutex);

        nimcp_error_t err = nimcp_equilibrium_best_response(ctx, player, strategies,
                                                             &best_s, &best_payoff);
        if (err != NIMCP_SUCCESS) {
            return err;
        }

        nimcp_platform_mutex_lock(&ctx->mutex);

        regrets[player] = best_payoff - current_payoff;
        if (regrets[player] < 0.0f) {
            regrets[player] = 0.0f;  // Can happen due to floating point
        }
    }

    nimcp_platform_mutex_unlock(&ctx->mutex);
    return NIMCP_SUCCESS;
}

//=============================================================================
// Utility Functions
//=============================================================================

float nimcp_equilibrium_expected_payoff(
    nimcp_equilibrium_t ctx,
    uint32_t player,
    const nimcp_strategy_profile_t* strategies
) {
    if (!ctx || !strategies || player >= ctx->config.num_players) {
        return 0.0f;
    }

    nimcp_platform_mutex_lock(&ctx->mutex);

    float payoff;
    if (strategies->type == NIMCP_STRATEGY_PURE) {
        payoff = compute_pure_payoff(ctx, player, strategies->pure_strategies);
    } else {
        payoff = compute_mixed_payoff(ctx, player, strategies);
    }

    nimcp_platform_mutex_unlock(&ctx->mutex);
    return payoff;
}

nimcp_error_t nimcp_equilibrium_get_stats(
    const nimcp_equilibrium_t ctx,
    nimcp_equilibrium_stats_t* stats
) {
    if (!ctx || !stats) {
        return NIMCP_GT_ERROR_NULL_POINTER;
    }

    nimcp_platform_mutex_lock(&ctx->mutex);
    *stats = ctx->stats;
    nimcp_platform_mutex_unlock(&ctx->mutex);

    return NIMCP_SUCCESS;
}

nimcp_error_t nimcp_equilibrium_reset(nimcp_equilibrium_t ctx) {
    if (!ctx) {
        return NIMCP_GT_ERROR_NULL_POINTER;
    }

    nimcp_platform_mutex_lock(&ctx->mutex);

    // Clear payoff matrices
    for (uint32_t i = 0; i < ctx->config.num_players; i++) {
        if (ctx->payoffs[i]) {
            nimcp_game_matrix_destroy(ctx->payoffs[i]);
            ctx->payoffs[i] = NULL;
        }
        ctx->payoffs_set[i] = false;
    }

    // Reset stats
    memset(&ctx->stats, 0, sizeof(nimcp_equilibrium_stats_t));
    ctx->stats.status = NIMCP_CONVERGENCE_NOT_STARTED;

    nimcp_platform_mutex_unlock(&ctx->mutex);
    return NIMCP_SUCCESS;
}

const char* nimcp_equilibrium_algo_name(nimcp_equilibrium_algo_t algo) {
    if (algo >= NIMCP_EQUILIBRIUM_ALGO_COUNT) {
        return "Unknown";
    }
    return s_algo_names[algo];
}

const char* nimcp_equilibrium_type_name(nimcp_equilibrium_type_t type) {
    if (type > NIMCP_EQUILIBRIUM_TYPE_APPROXIMATE) {
        return "Unknown";
    }
    return s_type_names[type];
}

//=============================================================================
// Strategy Profile Helpers
//=============================================================================

void nimcp_strategy_profile_init_pure(
    nimcp_strategy_profile_t* profile,
    uint32_t num_players,
    const uint32_t* strategies
) {
    if (!profile) return;

    memset(profile, 0, sizeof(nimcp_strategy_profile_t));
    profile->type = NIMCP_STRATEGY_PURE;
    profile->num_players = num_players;

    for (uint32_t i = 0; i < num_players && i < NIMCP_GT_MAX_PLAYERS; i++) {
        profile->pure_strategies[i] = strategies ? strategies[i] : 0;
    }
}

nimcp_error_t nimcp_strategy_profile_init_mixed(
    nimcp_strategy_profile_t* profile,
    uint32_t num_players,
    const uint32_t* num_strategies
) {
    if (!profile || !num_strategies) {
        return NIMCP_GT_ERROR_NULL_POINTER;
    }

    memset(profile, 0, sizeof(nimcp_strategy_profile_t));
    profile->type = NIMCP_STRATEGY_MIXED;
    profile->num_players = num_players;

    for (uint32_t i = 0; i < num_players && i < NIMCP_GT_MAX_PLAYERS; i++) {
        profile->num_strategies[i] = num_strategies[i];
        profile->mixed_strategies[i] = nimcp_calloc(num_strategies[i], sizeof(float));
        if (!profile->mixed_strategies[i]) {
            // Cleanup on failure
            for (uint32_t j = 0; j < i; j++) {
                nimcp_free(profile->mixed_strategies[j]);
            }
            return NIMCP_GT_ERROR_NO_MEMORY;
        }
    }

    return NIMCP_SUCCESS;
}

void nimcp_strategy_profile_cleanup(nimcp_strategy_profile_t* profile) {
    if (!profile) return;

    if (profile->type == NIMCP_STRATEGY_MIXED) {
        for (uint32_t i = 0; i < profile->num_players && i < NIMCP_GT_MAX_PLAYERS; i++) {
            nimcp_free(profile->mixed_strategies[i]);
            profile->mixed_strategies[i] = NULL;
        }
    }

    memset(profile, 0, sizeof(nimcp_strategy_profile_t));
}

nimcp_error_t nimcp_strategy_profile_copy(
    nimcp_strategy_profile_t* dest,
    const nimcp_strategy_profile_t* src
) {
    if (!dest || !src) {
        return NIMCP_GT_ERROR_NULL_POINTER;
    }

    // Copy basic fields
    dest->type = src->type;
    dest->num_players = src->num_players;

    if (src->type == NIMCP_STRATEGY_PURE) {
        memcpy(dest->pure_strategies, src->pure_strategies,
               NIMCP_GT_MAX_PLAYERS * sizeof(uint32_t));
    } else {
        // Allocate and copy mixed strategies
        for (uint32_t i = 0; i < src->num_players && i < NIMCP_GT_MAX_PLAYERS; i++) {
            dest->num_strategies[i] = src->num_strategies[i];
            dest->mixed_strategies[i] = nimcp_calloc(src->num_strategies[i], sizeof(float));
            if (!dest->mixed_strategies[i]) {
                // Cleanup on failure
                for (uint32_t j = 0; j < i; j++) {
                    nimcp_free(dest->mixed_strategies[j]);
                }
                return NIMCP_GT_ERROR_NO_MEMORY;
            }
            memcpy(dest->mixed_strategies[i], src->mixed_strategies[i],
                   src->num_strategies[i] * sizeof(float));
        }
    }

    return NIMCP_SUCCESS;
}

//=============================================================================
// Static Helper Functions
//=============================================================================

static uint32_t compute_total_profiles(const nimcp_equilibrium_config_t* config) {
    uint32_t total = 1;
    for (uint32_t i = 0; i < config->num_players; i++) {
        total *= config->num_strategies[i];
    }
    return total;
}

static void increment_profile(uint32_t* profile, const uint32_t* max_strategies,
                              uint32_t num_players) {
    for (int i = (int)num_players - 1; i >= 0; i--) {
        profile[i]++;
        if (profile[i] < max_strategies[i]) {
            return;
        }
        profile[i] = 0;
    }
}

static float compute_pure_payoff(const nimcp_equilibrium_t ctx, uint32_t player,
                                 const uint32_t* profile) {
    if (!ctx->payoffs[player]) {
        return 0.0f;
    }
    return nimcp_game_matrix_get(ctx->payoffs[player], profile);
}

static float compute_mixed_payoff(const nimcp_equilibrium_t ctx, uint32_t player,
                                  const nimcp_strategy_profile_t* strategies) {
    if (!ctx->payoffs[player] || strategies->type != NIMCP_STRATEGY_MIXED) {
        return 0.0f;
    }

    uint32_t n = ctx->config.num_players;
    float total_payoff = 0.0f;

    // Iterate over all strategy profiles
    uint32_t temp_profile[NIMCP_GT_MAX_PLAYERS];
    memset(temp_profile, 0, sizeof(temp_profile));

    for (uint32_t p = 0; p < ctx->total_strategy_profiles; p++) {
        // Compute probability of this profile
        float prob = 1.0f;
        for (uint32_t i = 0; i < n; i++) {
            prob *= strategies->mixed_strategies[i][temp_profile[i]];
        }

        // Add weighted payoff
        if (prob > 1e-10f) {
            float payoff = nimcp_game_matrix_get(ctx->payoffs[player], temp_profile);
            total_payoff += prob * payoff;
        }

        increment_profile(temp_profile, ctx->config.num_strategies, n);
    }

    return total_payoff;
}

static bool solve_indifference_2player(const nimcp_equilibrium_t ctx,
                                       const uint32_t* support_row,
                                       uint32_t support_row_size,
                                       const uint32_t* support_col,
                                       uint32_t support_col_size,
                                       float* row_probs, float* col_probs) {
    // For 2-player games, solve the indifference conditions:
    // Row player is indifferent over support_row strategies when:
    // sum_j col_probs[j] * B[i][j] = constant for all i in support_row
    //
    // Column player is indifferent over support_col strategies when:
    // sum_i row_probs[i] * A[i][j] = constant for all j in support_col

    if (support_row_size == 0 || support_col_size == 0) {
        return false;
    }

    // Simple case: 1x1 support
    if (support_row_size == 1 && support_col_size == 1) {
        row_probs[support_row[0]] = 1.0f;
        col_probs[support_col[0]] = 1.0f;
        return true;
    }

    // For larger supports, use Gaussian elimination
    // This is a simplified implementation for 2x2 case

    if (support_row_size == 2 && support_col_size == 2) {
        uint32_t i0 = support_row[0], i1 = support_row[1];
        uint32_t j0 = support_col[0], j1 = support_col[1];

        // Column player indifference:
        // p0 * A[i0][j0] + p1 * A[i1][j0] = p0 * A[i0][j1] + p1 * A[i1][j1]
        // p0 * (A[i0][j0] - A[i0][j1]) = p1 * (A[i1][j1] - A[i1][j0])
        // p0 + p1 = 1

        uint32_t profile[2];
        profile[0] = i0; profile[1] = j0;
        float A00 = nimcp_game_matrix_get(ctx->payoffs[0], profile);
        profile[1] = j1;
        float A01 = nimcp_game_matrix_get(ctx->payoffs[0], profile);
        profile[0] = i1; profile[1] = j0;
        float A10 = nimcp_game_matrix_get(ctx->payoffs[0], profile);
        profile[1] = j1;
        float A11 = nimcp_game_matrix_get(ctx->payoffs[0], profile);

        float diff_row = (A11 - A10) - (A01 - A00);
        if (fabsf(diff_row) < 1e-10f) {
            // Degenerate case
            return false;
        }

        float p1 = (A00 - A01) / diff_row;
        float p0 = 1.0f - p1;

        if (p0 < -1e-6f || p0 > 1.0f + 1e-6f || p1 < -1e-6f || p1 > 1.0f + 1e-6f) {
            return false;
        }

        // Clamp to valid range
        p0 = fmaxf(0.0f, fminf(1.0f, p0));
        p1 = fmaxf(0.0f, fminf(1.0f, p1));

        row_probs[i0] = p0;
        row_probs[i1] = p1;

        // Row player indifference:
        // q0 * B[i0][j0] + q1 * B[i0][j1] = q0 * B[i1][j0] + q1 * B[i1][j1]

        profile[0] = i0; profile[1] = j0;
        float B00 = nimcp_game_matrix_get(ctx->payoffs[1], profile);
        profile[1] = j1;
        float B01 = nimcp_game_matrix_get(ctx->payoffs[1], profile);
        profile[0] = i1; profile[1] = j0;
        float B10 = nimcp_game_matrix_get(ctx->payoffs[1], profile);
        profile[1] = j1;
        float B11 = nimcp_game_matrix_get(ctx->payoffs[1], profile);

        float diff_col = (B11 - B01) - (B10 - B00);
        if (fabsf(diff_col) < 1e-10f) {
            return false;
        }

        float q1 = (B00 - B10) / diff_col;
        float q0 = 1.0f - q1;

        if (q0 < -1e-6f || q0 > 1.0f + 1e-6f || q1 < -1e-6f || q1 > 1.0f + 1e-6f) {
            return false;
        }

        q0 = fmaxf(0.0f, fminf(1.0f, q0));
        q1 = fmaxf(0.0f, fminf(1.0f, q1));

        col_probs[j0] = q0;
        col_probs[j1] = q1;

        return true;
    }

    // For other cases, return false (would need general linear algebra solver)
    return false;
}

static bool check_no_outside_deviation(const nimcp_equilibrium_t ctx,
                                       const nimcp_strategy_profile_t* strategies) {
    // For each player, check that strategies outside support don't give higher payoff
    for (uint32_t player = 0; player < ctx->config.num_players; player++) {
        // Compute payoff from any strategy in support
        float support_payoff = 0.0f;
        bool found_support = false;

        for (uint32_t s = 0; s < ctx->config.num_strategies[player]; s++) {
            if (strategies->mixed_strategies[player][s] > 1e-6f) {
                // Create profile with this pure strategy
                nimcp_strategy_profile_t temp;
                nimcp_strategy_profile_copy(&temp, strategies);

                // Set player's strategy to pure s
                memset(temp.mixed_strategies[player], 0,
                       ctx->config.num_strategies[player] * sizeof(float));
                temp.mixed_strategies[player][s] = 1.0f;

                support_payoff = compute_mixed_payoff(ctx, player, &temp);
                nimcp_strategy_profile_cleanup(&temp);
                found_support = true;
                break;
            }
        }

        if (!found_support) continue;

        // Check all strategies outside support
        for (uint32_t s = 0; s < ctx->config.num_strategies[player]; s++) {
            if (strategies->mixed_strategies[player][s] < 1e-6f) {
                // Strategy outside support
                nimcp_strategy_profile_t temp;
                nimcp_strategy_profile_copy(&temp, strategies);

                memset(temp.mixed_strategies[player], 0,
                       ctx->config.num_strategies[player] * sizeof(float));
                temp.mixed_strategies[player][s] = 1.0f;

                float outside_payoff = compute_mixed_payoff(ctx, player, &temp);
                nimcp_strategy_profile_cleanup(&temp);

                if (outside_payoff > support_payoff + 1e-6f) {
                    return false;  // Profitable deviation outside support
                }
            }
        }
    }

    return true;
}

//=============================================================================
// KG Self-Awareness Integration
//=============================================================================

int equilibrium_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;

    const kg_entity_t* self = kg_reader_get_entity(kg, "Equilibrium_Module");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            LOG_DEBUG(LOG_MODULE, "Equilibrium self-knowledge: %s", self->observations[i]);
        }
    }

    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Equilibrium_Module");
    if (connections) {
        kg_relation_list_destroy(connections);
    }

    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Equilibrium_Module");
    if (incoming) {
        kg_relation_list_destroy(incoming);
    }

    return self ? 1 : 0;
}
