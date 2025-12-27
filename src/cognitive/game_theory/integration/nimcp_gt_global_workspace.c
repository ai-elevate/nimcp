//=============================================================================
// nimcp_gt_global_workspace.c - Game-Theoretic Global Workspace Integration
//=============================================================================
/**
 * @file nimcp_gt_global_workspace.c
 * @brief Auction-based competition for Global Workspace broadcast access
 *
 * WHAT: Second-price auction for GW slot allocation
 * WHY:  Incentive-compatible, efficient access allocation
 * HOW:  Modules bid with salience, highest bidder wins, pays second price
 *
 * @author NIMCP Development Team
 * @date 2024-12-27
 * @version 1.0.0
 */

#include "cognitive/game_theory/integration/nimcp_gt_global_workspace.h"
#include "cognitive/game_theory/nimcp_auction.h"
#include "utils/memory/nimcp_memory.h"
#include "core/nimcp_error.h"
#include <string.h>
#include <math.h>

//=============================================================================
// Constants
//=============================================================================

#define MAX_GW_MODULES 64
#define DEFAULT_HISTORY_DEPTH 100

//=============================================================================
// Internal Structures
//=============================================================================

/**
 * @brief Per-module state tracking
 */
typedef struct {
    cognitive_module_t module;
    float current_budget;
    float total_spent;
    uint64_t bids_submitted;
    uint64_t wins;
    float total_winning_bids;
    float total_payments;
    uint64_t last_win_time_ms;
    bool active;
} module_state_internal_t;

/**
 * @brief Pending bid for current round
 */
typedef struct {
    cognitive_module_t module;
    float* content;
    uint32_t content_dim;
    float bid;
    bool valid;
} pending_bid_t;

/**
 * @brief Opaque context structure
 */
struct gt_gw_auction_ctx_struct {
    global_workspace_t* workspace;
    gt_gw_config_t config;
    nimcp_auction_t auction;

    // Module states
    module_state_internal_t modules[MAX_GW_MODULES];
    uint32_t num_modules;

    // Current round bids
    pending_bid_t pending_bids[MAX_GW_MODULES];
    uint32_t num_pending_bids;

    // Statistics
    uint64_t total_rounds;
    uint64_t total_allocations;
    float total_revenue;

    // Timing
    uint64_t last_replenish_time_ms;

    bool active;
};

//=============================================================================
// Helper Functions
//=============================================================================

static uint64_t get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000 + (uint64_t)ts.tv_nsec / 1000000;
}

static int find_module_index(gt_gw_auction_ctx_t ctx, cognitive_module_t module) {
    for (uint32_t i = 0; i < ctx->num_modules; i++) {
        if (ctx->modules[i].module == module) {
            return (int)i;
        }
    }
    return -1;
}

static nimcp_error_t ensure_module_registered(gt_gw_auction_ctx_t ctx, cognitive_module_t module) {
    int idx = find_module_index(ctx, module);
    if (idx >= 0) {
        return NIMCP_SUCCESS;
    }

    if (ctx->num_modules >= MAX_GW_MODULES) {
        return NIMCP_GT_ERROR_CAPACITY;
    }

    module_state_internal_t* state = &ctx->modules[ctx->num_modules];
    state->module = module;
    state->current_budget = ctx->config.initial_budget;
    state->total_spent = 0.0f;
    state->bids_submitted = 0;
    state->wins = 0;
    state->total_winning_bids = 0.0f;
    state->total_payments = 0.0f;
    state->last_win_time_ms = 0;
    state->active = true;

    ctx->num_modules++;
    return NIMCP_SUCCESS;
}

//=============================================================================
// Lifecycle
//=============================================================================

gt_gw_config_t gt_gw_default_config(void) {
    gt_gw_config_t config = {
        .strategy = GT_GW_STRATEGY_SECOND_PRICE,
        .reserve_price = 0.1f,
        .initial_budget = 100.0f,
        .budget_replenish_rate = 1.0f,
        .budget_replenish_interval_ms = 1000,
        .enable_budget_constraints = true,
        .track_bid_history = false,
        .history_depth = DEFAULT_HISTORY_DEPTH
    };
    return config;
}

gt_gw_auction_ctx_t gt_gw_create(
    global_workspace_t* workspace,
    const gt_gw_config_t* config
) {
    if (!workspace) {
        return NULL;
    }

    gt_gw_auction_ctx_t ctx = nimcp_calloc(1, sizeof(struct gt_gw_auction_ctx_struct));
    if (!ctx) {
        return NULL;
    }

    ctx->workspace = workspace;
    ctx->config = config ? *config : gt_gw_default_config();

    // Create underlying auction
    nimcp_auction_config_t auction_config = nimcp_auction_default_config();
    auction_config.type = NIMCP_AUCTION_SECOND_PRICE;
    auction_config.reserve_price = ctx->config.reserve_price;
    auction_config.max_bidders = MAX_GW_MODULES;

    ctx->auction = nimcp_auction_create(&auction_config);
    if (!ctx->auction) {
        nimcp_free(ctx);
        return NULL;
    }

    ctx->num_modules = 0;
    ctx->num_pending_bids = 0;
    ctx->total_rounds = 0;
    ctx->total_allocations = 0;
    ctx->total_revenue = 0.0f;
    ctx->last_replenish_time_ms = get_time_ms();
    ctx->active = true;

    return ctx;
}

void gt_gw_destroy(gt_gw_auction_ctx_t ctx) {
    if (!ctx) {
        return;
    }

    // Free any pending bid content
    for (uint32_t i = 0; i < ctx->num_pending_bids; i++) {
        if (ctx->pending_bids[i].content) {
            nimcp_free(ctx->pending_bids[i].content);
        }
    }

    if (ctx->auction) {
        nimcp_auction_destroy(ctx->auction);
    }

    nimcp_free(ctx);
}

//=============================================================================
// Core Operations
//=============================================================================

nimcp_error_t gt_gw_bid(
    gt_gw_auction_ctx_t ctx,
    cognitive_module_t module,
    const float* content,
    uint32_t content_dim,
    float bid
) {
    if (!ctx || !content || content_dim == 0) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    if (!ctx->active) {
        return NIMCP_GT_ERROR_GAME_OVER;
    }

    // Ensure module is registered
    nimcp_error_t err = ensure_module_registered(ctx, module);
    if (err != NIMCP_SUCCESS) {
        return err;
    }

    int module_idx = find_module_index(ctx, module);
    if (module_idx < 0) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    module_state_internal_t* state = &ctx->modules[module_idx];

    // Check budget constraint
    if (ctx->config.enable_budget_constraints && bid > state->current_budget) {
        return NIMCP_GT_ERROR_BUDGET;
    }

    // Check if module already has a pending bid this round
    for (uint32_t i = 0; i < ctx->num_pending_bids; i++) {
        if (ctx->pending_bids[i].module == module) {
            // Update existing bid
            nimcp_free(ctx->pending_bids[i].content);
            ctx->pending_bids[i].content = nimcp_malloc(content_dim * sizeof(float));
            if (!ctx->pending_bids[i].content) {
                return NIMCP_ERROR_MEMORY;
            }
            memcpy(ctx->pending_bids[i].content, content, content_dim * sizeof(float));
            ctx->pending_bids[i].content_dim = content_dim;
            ctx->pending_bids[i].bid = bid;
            return NIMCP_SUCCESS;
        }
    }

    // Add new pending bid
    if (ctx->num_pending_bids >= MAX_GW_MODULES) {
        return NIMCP_GT_ERROR_CAPACITY;
    }

    pending_bid_t* pb = &ctx->pending_bids[ctx->num_pending_bids];
    pb->module = module;
    pb->content = nimcp_malloc(content_dim * sizeof(float));
    if (!pb->content) {
        return NIMCP_ERROR_MEMORY;
    }
    memcpy(pb->content, content, content_dim * sizeof(float));
    pb->content_dim = content_dim;
    pb->bid = bid;
    pb->valid = true;

    ctx->num_pending_bids++;
    state->bids_submitted++;

    return NIMCP_SUCCESS;
}

nimcp_error_t gt_gw_resolve(
    gt_gw_auction_ctx_t ctx,
    gt_gw_round_result_t* result
) {
    if (!ctx || !result) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    memset(result, 0, sizeof(gt_gw_round_result_t));

    if (ctx->num_pending_bids == 0) {
        result->reserve_met = false;
        return NIMCP_SUCCESS;
    }

    // Reset auction for this round
    nimcp_auction_reset(ctx->auction);

    // Submit all pending bids to auction
    for (uint32_t i = 0; i < ctx->num_pending_bids; i++) {
        pending_bid_t* pb = &ctx->pending_bids[i];
        if (pb->valid) {
            nimcp_bid_t bid = {
                .bidder_id = (nimcp_player_id_t)pb->module,
                .bid_amount = pb->bid,
                .timestamp_ms = get_time_ms()
            };
            nimcp_auction_bid(ctx->auction, &bid);
        }
    }

    // Resolve auction
    nimcp_auction_result_t auction_result;
    nimcp_error_t err = nimcp_auction_resolve(ctx->auction, &auction_result);
    if (err != NIMCP_SUCCESS) {
        // Clear pending bids
        for (uint32_t i = 0; i < ctx->num_pending_bids; i++) {
            if (ctx->pending_bids[i].content) {
                nimcp_free(ctx->pending_bids[i].content);
                ctx->pending_bids[i].content = NULL;
            }
        }
        ctx->num_pending_bids = 0;
        return err;
    }

    // Populate result
    result->winner = (cognitive_module_t)auction_result.winner_id;
    result->winning_bid = auction_result.winning_bid;
    result->payment = auction_result.payment;
    result->second_highest_bid = auction_result.second_price;
    result->num_bidders = ctx->num_pending_bids;
    result->reserve_met = auction_result.sold;

    // Calculate total bids
    result->total_bids = 0.0f;
    for (uint32_t i = 0; i < ctx->num_pending_bids; i++) {
        result->total_bids += ctx->pending_bids[i].bid;
    }

    // If winner exists, broadcast their content
    if (result->reserve_met) {
        // Find winner's pending bid
        for (uint32_t i = 0; i < ctx->num_pending_bids; i++) {
            if (ctx->pending_bids[i].module == result->winner) {
                // TODO: Call actual GW broadcast when available
                // global_workspace_broadcast(ctx->workspace,
                //     ctx->pending_bids[i].content,
                //     ctx->pending_bids[i].content_dim);
                break;
            }
        }

        // Update winner state
        int winner_idx = find_module_index(ctx, result->winner);
        if (winner_idx >= 0) {
            module_state_internal_t* winner_state = &ctx->modules[winner_idx];
            winner_state->wins++;
            winner_state->total_winning_bids += result->winning_bid;
            winner_state->total_payments += result->payment;
            if (ctx->config.enable_budget_constraints) {
                winner_state->current_budget -= result->payment;
            }
            winner_state->last_win_time_ms = get_time_ms();
        }

        ctx->total_allocations++;
        ctx->total_revenue += result->payment;
    }

    // Clear pending bids
    for (uint32_t i = 0; i < ctx->num_pending_bids; i++) {
        if (ctx->pending_bids[i].content) {
            nimcp_free(ctx->pending_bids[i].content);
            ctx->pending_bids[i].content = NULL;
        }
    }
    ctx->num_pending_bids = 0;
    ctx->total_rounds++;

    return NIMCP_SUCCESS;
}

bool gt_gw_compete(
    gt_gw_auction_ctx_t ctx,
    cognitive_module_t module,
    const float* content,
    uint32_t content_dim,
    float bid
) {
    if (!ctx) {
        return false;
    }

    // Submit bid
    nimcp_error_t err = gt_gw_bid(ctx, module, content, content_dim, bid);
    if (err != NIMCP_SUCCESS) {
        return false;
    }

    // Resolve immediately
    gt_gw_round_result_t result;
    err = gt_gw_resolve(ctx, &result);
    if (err != NIMCP_SUCCESS) {
        return false;
    }

    return result.reserve_met && result.winner == module;
}

void gt_gw_replenish_budgets(gt_gw_auction_ctx_t ctx, float amount) {
    if (!ctx) {
        return;
    }

    float replenish = (amount > 0.0f) ? amount : ctx->config.budget_replenish_rate;

    for (uint32_t i = 0; i < ctx->num_modules; i++) {
        ctx->modules[i].current_budget += replenish;
        // Cap at initial budget
        if (ctx->modules[i].current_budget > ctx->config.initial_budget) {
            ctx->modules[i].current_budget = ctx->config.initial_budget;
        }
    }

    ctx->last_replenish_time_ms = get_time_ms();
}

void gt_gw_reset_budgets(gt_gw_auction_ctx_t ctx) {
    if (!ctx) {
        return;
    }

    for (uint32_t i = 0; i < ctx->num_modules; i++) {
        ctx->modules[i].current_budget = ctx->config.initial_budget;
    }
}

//=============================================================================
// Query Functions
//=============================================================================

nimcp_error_t gt_gw_get_module_state(
    const gt_gw_auction_ctx_t ctx,
    cognitive_module_t module,
    gt_gw_module_state_t* state
) {
    if (!ctx || !state) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    int idx = find_module_index((gt_gw_auction_ctx_t)ctx, module);
    if (idx < 0) {
        return NIMCP_GT_ERROR_PLAYER_NOT_FOUND;
    }

    const module_state_internal_t* internal = &ctx->modules[idx];

    state->module = internal->module;
    state->current_budget = internal->current_budget;
    state->total_spent = internal->total_spent;
    state->bids_submitted = internal->bids_submitted;
    state->wins = internal->wins;
    state->avg_winning_bid = (internal->wins > 0) ?
        internal->total_winning_bids / internal->wins : 0.0f;
    state->avg_payment = (internal->wins > 0) ?
        internal->total_payments / internal->wins : 0.0f;
    state->last_win_time_ms = internal->last_win_time_ms;

    return NIMCP_SUCCESS;
}

nimcp_auction_t gt_gw_get_auction(const gt_gw_auction_ctx_t ctx) {
    return ctx ? ctx->auction : NULL;
}

global_workspace_t* gt_gw_get_workspace(const gt_gw_auction_ctx_t ctx) {
    return ctx ? ctx->workspace : NULL;
}

bool gt_gw_is_auction_active(const gt_gw_auction_ctx_t ctx) {
    return ctx ? ctx->active : false;
}

nimcp_error_t gt_gw_get_stats(
    const gt_gw_auction_ctx_t ctx,
    nimcp_game_stats_t* stats
) {
    if (!ctx || !stats) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    memset(stats, 0, sizeof(nimcp_game_stats_t));

    stats->games_played = ctx->total_rounds;

    // Calculate wins per module
    float total_wins = 0.0f;
    float total_payments = 0.0f;
    for (uint32_t i = 0; i < ctx->num_modules; i++) {
        total_wins += ctx->modules[i].wins;
        total_payments += ctx->modules[i].total_payments;
    }

    // Compute fairness (Jain's index on win distribution)
    if (ctx->num_modules > 0 && total_wins > 0.0f) {
        float sum = 0.0f;
        float sum_sq = 0.0f;
        for (uint32_t i = 0; i < ctx->num_modules; i++) {
            float share = (float)ctx->modules[i].wins / total_wins;
            sum += share;
            sum_sq += share * share;
        }
        stats->avg_fairness = (sum * sum) / (ctx->num_modules * sum_sq);
    }

    // Average payoff = average payment saved (bid - payment)
    stats->avg_payoff = (ctx->total_allocations > 0) ?
        ctx->total_revenue / ctx->total_allocations : 0.0f;

    return NIMCP_SUCCESS;
}
