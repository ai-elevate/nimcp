//=============================================================================
// nimcp_gt_hemispheric.c - Game-Theoretic Hemispheric Brain Integration
//=============================================================================
/**
 * @file nimcp_gt_hemispheric.c
 * @brief Nash bargaining and Shapley credit for hemispheric brain
 *
 * WHAT: Game-theoretic hemisphere coordination
 * WHY:  Fair resource allocation and credit assignment
 * HOW:  Nash bargaining for resources, Shapley for credit
 *
 * @author NIMCP Development Team
 * @date 2024-12-27
 * @version 1.0.0
 */

#include "cognitive/game_theory/integration/nimcp_gt_hemispheric.h"
#include "cognitive/game_theory/nimcp_bargaining.h"
#include "cognitive/game_theory/nimcp_credit_assignment.h"
#include "utils/memory/nimcp_memory.h"
#include "core/nimcp_error.h"
#include <string.h>
#include <math.h>

//=============================================================================
// Constants
//=============================================================================

#define NUM_HEMISPHERES 2
#define LEFT_HEMI 0
#define RIGHT_HEMI 1

//=============================================================================
// Internal Structures
//=============================================================================

/**
 * @brief Opaque context structure
 */
struct gt_hemi_bargaining_ctx_struct {
    hemispheric_brain_t* brain;
    gt_hemi_config_t config;

    // Bargaining state
    nimcp_bargaining_t bargaining;
    nimcp_credit_ctx_t credit_ctx;

    // Last results
    gt_hemi_outcome_t last_outcome;
    gt_hemi_credit_t last_credit;

    // Statistics
    uint64_t negotiations_completed;
    uint64_t agreements_reached;
    float total_left_allocation;
    float total_right_allocation;
    float total_left_credit;
    float total_right_credit;

    bool active;
};

//=============================================================================
// Coalition Value Callback for Shapley
//=============================================================================

/**
 * @brief User data for coalition value computation
 */
typedef struct {
    gt_hemi_bargaining_ctx_t ctx;
    const float* combined_output;
    uint32_t output_size;
    float left_only_value;
    float right_only_value;
} hemi_coalition_data_t;

/**
 * @brief Compute coalition value for hemispheric credit assignment
 *
 * Coalition values:
 * - Empty: 0
 * - {Left}: Left hemisphere solo contribution
 * - {Right}: Right hemisphere solo contribution
 * - {Left, Right}: Combined output value
 */
static float hemi_coalition_value(
    const bool* coalition,
    uint32_t num_players,
    void* user_data
) {
    hemi_coalition_data_t* data = (hemi_coalition_data_t*)user_data;

    bool has_left = coalition[LEFT_HEMI];
    bool has_right = coalition[RIGHT_HEMI];

    if (!has_left && !has_right) {
        return 0.0f;  // Empty coalition
    }

    if (has_left && has_right) {
        // Grand coalition - compute L2 norm of combined output
        float sum_sq = 0.0f;
        for (uint32_t i = 0; i < data->output_size; i++) {
            sum_sq += data->combined_output[i] * data->combined_output[i];
        }
        return sqrtf(sum_sq);
    }

    if (has_left) {
        return data->left_only_value;
    }

    // has_right
    return data->right_only_value;
}

//=============================================================================
// Lifecycle
//=============================================================================

gt_hemi_config_t gt_hemi_default_config(void) {
    gt_hemi_config_t config = {
        .left_bargaining_power = 0.5f,
        .right_bargaining_power = 0.5f,
        .disagreement_left = 0.0f,
        .disagreement_right = 0.0f,
        .discount_factor = 0.9f,
        .max_rounds = 10,
        .use_shapley_credit = true,
        .bargain_type = NIMCP_BARGAINING_NASH
    };
    return config;
}

gt_hemi_bargaining_ctx_t gt_hemi_create(
    hemispheric_brain_t* brain,
    const gt_hemi_config_t* config
) {
    if (!brain) {
        return NULL;
    }

    gt_hemi_bargaining_ctx_t ctx = nimcp_calloc(1, sizeof(struct gt_hemi_bargaining_ctx_struct));
    if (!ctx) {
        return NULL;
    }

    ctx->brain = brain;
    ctx->config = config ? *config : gt_hemi_default_config();

    // Create bargaining context
    nimcp_bargaining_config_t bargain_config = nimcp_bargaining_default_config();
    bargain_config.type = ctx->config.bargain_type;
    bargain_config.num_players = NUM_HEMISPHERES;
    bargain_config.discount_factor = ctx->config.discount_factor;
    bargain_config.max_rounds = ctx->config.max_rounds;
    bargain_config.bargaining_powers[LEFT_HEMI] = ctx->config.left_bargaining_power;
    bargain_config.bargaining_powers[RIGHT_HEMI] = ctx->config.right_bargaining_power;
    bargain_config.disagreement_point[LEFT_HEMI] = ctx->config.disagreement_left;
    bargain_config.disagreement_point[RIGHT_HEMI] = ctx->config.disagreement_right;

    ctx->bargaining = nimcp_bargaining_create(&bargain_config);
    if (!ctx->bargaining) {
        nimcp_free(ctx);
        return NULL;
    }

    // Create credit assignment context
    nimcp_credit_config_t credit_config = nimcp_credit_default_config();
    credit_config.method = NIMCP_CREDIT_SHAPLEY;
    credit_config.num_players = NUM_HEMISPHERES;

    ctx->credit_ctx = nimcp_credit_create(&credit_config);
    if (!ctx->credit_ctx) {
        nimcp_bargaining_destroy(ctx->bargaining);
        nimcp_free(ctx);
        return NULL;
    }

    memset(&ctx->last_outcome, 0, sizeof(gt_hemi_outcome_t));
    memset(&ctx->last_credit, 0, sizeof(gt_hemi_credit_t));

    ctx->negotiations_completed = 0;
    ctx->agreements_reached = 0;
    ctx->total_left_allocation = 0.0f;
    ctx->total_right_allocation = 0.0f;
    ctx->total_left_credit = 0.0f;
    ctx->total_right_credit = 0.0f;

    ctx->active = true;

    return ctx;
}

void gt_hemi_destroy(gt_hemi_bargaining_ctx_t ctx) {
    if (!ctx) {
        return;
    }

    if (ctx->bargaining) {
        nimcp_bargaining_destroy(ctx->bargaining);
    }

    if (ctx->credit_ctx) {
        nimcp_credit_destroy(ctx->credit_ctx);
    }

    nimcp_free(ctx);
}

//=============================================================================
// Core Operations
//=============================================================================

nimcp_error_t gt_hemi_process_bargaining(
    gt_hemi_bargaining_ctx_t ctx,
    const float* input,
    uint32_t input_size,
    float* output,
    uint32_t output_size,
    gt_hemi_outcome_t* outcome
) {
    if (!ctx || !input || !output || !outcome) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    if (!ctx->active) {
        return NIMCP_GT_ERROR_GAME_OVER;
    }

    // First negotiate resource allocation
    gt_hemi_outcome_t allocation_outcome;
    nimcp_error_t err = gt_hemi_negotiate_resources(ctx, 1.0f, &allocation_outcome);
    if (err != NIMCP_SUCCESS) {
        return err;
    }

    // TODO: Process input through hemispheric brain using negotiated allocation
    // For now, copy input to output as placeholder
    uint32_t copy_size = (input_size < output_size) ? input_size : output_size;
    memcpy(output, input, copy_size * sizeof(float));

    // Fill remaining output with zeros
    if (output_size > input_size) {
        memset(output + input_size, 0, (output_size - input_size) * sizeof(float));
    }

    *outcome = allocation_outcome;
    return NIMCP_SUCCESS;
}

nimcp_error_t gt_hemi_negotiate_resources(
    gt_hemi_bargaining_ctx_t ctx,
    float total_resources,
    gt_hemi_outcome_t* outcome
) {
    if (!ctx || !outcome || total_resources <= 0.0f) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    memset(outcome, 0, sizeof(gt_hemi_outcome_t));

    // Create feasible set - all (x, y) such that x + y <= total_resources
    // For simplicity, sample the Pareto frontier
    const uint32_t num_samples = 11;
    nimcp_feasible_point_t feasible[11];

    for (uint32_t i = 0; i < num_samples; i++) {
        float left_share = (float)i / (num_samples - 1);
        feasible[i].utilities[LEFT_HEMI] = left_share * total_resources;
        feasible[i].utilities[RIGHT_HEMI] = (1.0f - left_share) * total_resources;
    }

    nimcp_error_t err = nimcp_bargaining_set_feasible(ctx->bargaining, feasible, num_samples);
    if (err != NIMCP_SUCCESS) {
        return err;
    }

    // Solve bargaining problem
    nimcp_bargaining_outcome_t bargain_outcome;
    err = nimcp_bargaining_solve(ctx->bargaining, &bargain_outcome);
    if (err != NIMCP_SUCCESS) {
        return err;
    }

    // Populate outcome
    outcome->left_allocation = bargain_outcome.allocations[LEFT_HEMI];
    outcome->right_allocation = bargain_outcome.allocations[RIGHT_HEMI];
    outcome->left_utility = bargain_outcome.allocations[LEFT_HEMI];  // In this case utility = allocation
    outcome->right_utility = bargain_outcome.allocations[RIGHT_HEMI];
    outcome->rounds_taken = bargain_outcome.rounds_used;
    outcome->agreement_reached = bargain_outcome.agreement_reached;
    outcome->nash_product = bargain_outcome.nash_product;

    // Update statistics
    ctx->negotiations_completed++;
    if (outcome->agreement_reached) {
        ctx->agreements_reached++;
        ctx->total_left_allocation += outcome->left_allocation;
        ctx->total_right_allocation += outcome->right_allocation;
    }

    ctx->last_outcome = *outcome;

    return NIMCP_SUCCESS;
}

nimcp_error_t gt_hemi_compute_credit(
    gt_hemi_bargaining_ctx_t ctx,
    const float* combined_output,
    uint32_t output_size,
    gt_hemi_credit_t* credit
) {
    if (!ctx || !combined_output || output_size == 0 || !credit) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    memset(credit, 0, sizeof(gt_hemi_credit_t));

    // Compute combined output value (L2 norm)
    float total_value = 0.0f;
    for (uint32_t i = 0; i < output_size; i++) {
        total_value += combined_output[i] * combined_output[i];
    }
    total_value = sqrtf(total_value);
    credit->total_value = total_value;

    // Estimate solo contributions (heuristic: based on bargaining power)
    float left_solo = total_value * ctx->config.left_bargaining_power * 0.7f;
    float right_solo = total_value * ctx->config.right_bargaining_power * 0.7f;

    // Set up coalition value callback
    hemi_coalition_data_t callback_data = {
        .ctx = ctx,
        .combined_output = combined_output,
        .output_size = output_size,
        .left_only_value = left_solo,
        .right_only_value = right_solo
    };

    nimcp_error_t err = nimcp_credit_set_value_fn(
        ctx->credit_ctx,
        hemi_coalition_value,
        &callback_data
    );
    if (err != NIMCP_SUCCESS) {
        return err;
    }

    // Compute Shapley values
    float shapley[NUM_HEMISPHERES];
    err = nimcp_credit_compute(ctx->credit_ctx, shapley);
    if (err != NIMCP_SUCCESS) {
        return err;
    }

    credit->left_credit = shapley[LEFT_HEMI];
    credit->right_credit = shapley[RIGHT_HEMI];
    credit->synergy_bonus = total_value - (left_solo + right_solo);
    credit->is_superadditive = (credit->synergy_bonus > 0.0f);

    // Update statistics
    ctx->total_left_credit += credit->left_credit;
    ctx->total_right_credit += credit->right_credit;

    ctx->last_credit = *credit;

    return NIMCP_SUCCESS;
}

nimcp_error_t gt_hemi_set_bargaining_power(
    gt_hemi_bargaining_ctx_t ctx,
    float left_power
) {
    if (!ctx) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    if (left_power < 0.0f || left_power > 1.0f) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    ctx->config.left_bargaining_power = left_power;
    ctx->config.right_bargaining_power = 1.0f - left_power;

    // Update bargaining context
    return nimcp_bargaining_set_power(
        ctx->bargaining,
        LEFT_HEMI,
        left_power
    );
}

nimcp_error_t gt_hemi_set_disagreement(
    gt_hemi_bargaining_ctx_t ctx,
    float left_disagree,
    float right_disagree
) {
    if (!ctx) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    ctx->config.disagreement_left = left_disagree;
    ctx->config.disagreement_right = right_disagree;

    float disagreement[NUM_HEMISPHERES] = {left_disagree, right_disagree};
    return nimcp_bargaining_set_disagreement(ctx->bargaining, disagreement);
}

//=============================================================================
// Query Functions
//=============================================================================

hemispheric_brain_t* gt_hemi_get_brain(const gt_hemi_bargaining_ctx_t ctx) {
    return ctx ? ctx->brain : NULL;
}

nimcp_error_t gt_hemi_get_last_outcome(
    const gt_hemi_bargaining_ctx_t ctx,
    gt_hemi_outcome_t* outcome
) {
    if (!ctx || !outcome) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    *outcome = ctx->last_outcome;
    return NIMCP_SUCCESS;
}

nimcp_error_t gt_hemi_get_last_credit(
    const gt_hemi_bargaining_ctx_t ctx,
    gt_hemi_credit_t* credit
) {
    if (!ctx || !credit) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    *credit = ctx->last_credit;
    return NIMCP_SUCCESS;
}

nimcp_error_t gt_hemi_get_stats(
    const gt_hemi_bargaining_ctx_t ctx,
    nimcp_game_stats_t* stats
) {
    if (!ctx || !stats) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    memset(stats, 0, sizeof(nimcp_game_stats_t));

    stats->games_played = ctx->negotiations_completed;
    stats->cooperation_rate = (ctx->negotiations_completed > 0) ?
        (float)ctx->agreements_reached / ctx->negotiations_completed : 0.0f;

    // Compute fairness based on allocation history
    if (ctx->agreements_reached > 0) {
        float avg_left = ctx->total_left_allocation / ctx->agreements_reached;
        float avg_right = ctx->total_right_allocation / ctx->agreements_reached;
        float total = avg_left + avg_right;
        if (total > 0.0f) {
            // Jain's fairness index for 2 players
            float sum_sq = (avg_left / total) * (avg_left / total) +
                           (avg_right / total) * (avg_right / total);
            stats->avg_fairness = (avg_left + avg_right) * (avg_left + avg_right) /
                                  (2.0f * total * total * sum_sq);
        }
    }

    stats->avg_payoff = (ctx->negotiations_completed > 0) ?
        (ctx->total_left_allocation + ctx->total_right_allocation) /
        ctx->negotiations_completed : 0.0f;

    return NIMCP_SUCCESS;
}

bool gt_hemi_is_active(const gt_hemi_bargaining_ctx_t ctx) {
    return ctx ? ctx->active : false;
}
