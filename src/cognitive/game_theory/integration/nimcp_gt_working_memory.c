//=============================================================================
// nimcp_gt_working_memory.c - Priority Auction for Working Memory Slots
//=============================================================================
/**
 * @file nimcp_gt_working_memory.c
 * @brief Auction-based eviction policy for working memory
 *
 * WHAT: Auction-based eviction policy for working memory
 * WHY:  Market mechanism for efficient slot allocation
 * HOW:  Items bid with salience, lowest bidders evicted
 *
 * BIOLOGICAL INSPIRATION:
 * - Attentional competition among memory traces
 * - Rehearsal as bidding for retention
 * - Decay as declining bid strength
 *
 * @author NIMCP Development Team
 * @date 2024-12-27
 * @version 1.0.0
 */

#include "cognitive/game_theory/integration/nimcp_gt_working_memory.h"
#include "utils/memory/nimcp_memory.h"
#include "core/nimcp_error.h"
#include <string.h>
#include <math.h>
#include <time.h>

//=============================================================================
// Constants
//=============================================================================

#define MAX_WM_SLOTS 64
#define MAX_EVICTIONS_PER_ROUND 16

//=============================================================================
// Internal Structures
//=============================================================================

/**
 * @brief Per-slot auction state
 */
typedef struct {
    bool occupied;
    float current_bid;
    float initial_bid;
    uint32_t age_cycles;
    uint32_t rehearsal_count;
    uint64_t added_time_ms;
    uint64_t last_access_time_ms;
} slot_auction_state_t;

/**
 * @brief Opaque context structure
 */
struct gt_wm_auction_ctx_struct {
    working_memory_t* wm;
    gt_wm_config_t config;

    // Slot states
    slot_auction_state_t slots[MAX_WM_SLOTS];
    uint32_t capacity;
    uint32_t occupancy;

    // Statistics
    uint64_t eviction_rounds;
    uint64_t total_evictions;
    uint64_t total_additions;

    // Timing
    uint64_t last_eviction_time_ms;
    uint64_t last_decay_time_ms;

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

/**
 * @brief Compare slots by effective value (for sorting during eviction)
 */
static int compare_slots_by_value(const void* a, const void* b) {
    const uint32_t* idx_a = (const uint32_t*)a;
    const uint32_t* idx_b = (const uint32_t*)b;
    // This is a placeholder - actual comparison would need context access
    return (*idx_a < *idx_b) ? -1 : ((*idx_a > *idx_b) ? 1 : 0);
}

/**
 * @brief Compute congestion cost based on current occupancy
 */
static float compute_congestion_cost(gt_wm_auction_ctx_t ctx) {
    if (!ctx || ctx->capacity == 0) {
        return 0.0f;
    }

    float occupancy_ratio = (float)ctx->occupancy / ctx->capacity;
    return occupancy_ratio * occupancy_ratio * ctx->config.congestion_cost_factor;
}

//=============================================================================
// Lifecycle
//=============================================================================

gt_wm_config_t gt_wm_default_config(void) {
    gt_wm_config_t config = {
        .policy = GT_WM_EVICTION_PRIORITY_AUCTION,
        .slot_reserve_price = 0.1f,
        .congestion_cost_factor = 0.5f,
        .decay_rate = 0.05f,
        .enable_preemption = true,
        .preemption_premium = 0.2f,
        .eviction_interval_ms = 100
    };
    return config;
}

gt_wm_auction_ctx_t gt_wm_create(
    working_memory_t* wm,
    const gt_wm_config_t* config
) {
    if (!wm) {
        return NULL;
    }

    gt_wm_auction_ctx_t ctx = nimcp_calloc(1, sizeof(struct gt_wm_auction_ctx_struct));
    if (!ctx) {
        return NULL;
    }

    ctx->wm = wm;
    ctx->config = config ? *config : gt_wm_default_config();

    // Get capacity from working memory
    // For now, use a reasonable default
    ctx->capacity = MAX_WM_SLOTS;
    ctx->occupancy = 0;

    // Initialize slot states
    for (uint32_t i = 0; i < MAX_WM_SLOTS; i++) {
        ctx->slots[i].occupied = false;
        ctx->slots[i].current_bid = 0.0f;
        ctx->slots[i].initial_bid = 0.0f;
        ctx->slots[i].age_cycles = 0;
        ctx->slots[i].rehearsal_count = 0;
        ctx->slots[i].added_time_ms = 0;
        ctx->slots[i].last_access_time_ms = 0;
    }

    ctx->eviction_rounds = 0;
    ctx->total_evictions = 0;
    ctx->total_additions = 0;
    ctx->last_eviction_time_ms = get_time_ms();
    ctx->last_decay_time_ms = get_time_ms();

    ctx->active = true;

    return ctx;
}

void gt_wm_destroy(gt_wm_auction_ctx_t ctx) {
    if (!ctx) {
        return;
    }
    nimcp_free(ctx);
}

//=============================================================================
// Core Operations
//=============================================================================

nimcp_error_t gt_wm_add(
    gt_wm_auction_ctx_t ctx,
    const float* item,
    uint32_t item_size,
    float bid,
    int32_t* slot_index
) {
    if (!ctx || !item || item_size == 0 || !slot_index) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    *slot_index = -1;

    if (!ctx->active) {
        return NIMCP_GT_ERROR_GAME_OVER;
    }

    // Check if bid meets reserve price
    float congestion_cost = compute_congestion_cost(ctx);
    float effective_reserve = ctx->config.slot_reserve_price + congestion_cost;

    if (bid < effective_reserve) {
        return NIMCP_GT_ERROR_BID_TOO_LOW;
    }

    uint64_t now = get_time_ms();

    // Look for empty slot first
    for (uint32_t i = 0; i < ctx->capacity; i++) {
        if (!ctx->slots[i].occupied) {
            // Found empty slot
            ctx->slots[i].occupied = true;
            ctx->slots[i].current_bid = bid;
            ctx->slots[i].initial_bid = bid;
            ctx->slots[i].age_cycles = 0;
            ctx->slots[i].rehearsal_count = 0;
            ctx->slots[i].added_time_ms = now;
            ctx->slots[i].last_access_time_ms = now;

            ctx->occupancy++;
            ctx->total_additions++;
            *slot_index = (int32_t)i;

            // TODO: Actually add to working memory
            // working_memory_add(ctx->wm, item, item_size);

            return NIMCP_SUCCESS;
        }
    }

    // No empty slots - check if preemption is allowed
    if (!ctx->config.enable_preemption) {
        return NIMCP_GT_ERROR_CAPACITY;
    }

    // Find lowest bidder for preemption
    float lowest_bid = bid - ctx->config.preemption_premium;
    int32_t lowest_idx = -1;

    for (uint32_t i = 0; i < ctx->capacity; i++) {
        if (ctx->slots[i].occupied) {
            float effective_value = ctx->slots[i].current_bid - congestion_cost;
            if (effective_value < lowest_bid) {
                lowest_bid = effective_value;
                lowest_idx = (int32_t)i;
            }
        }
    }

    if (lowest_idx < 0) {
        // No slot can be preempted
        return NIMCP_GT_ERROR_CAPACITY;
    }

    // Preempt lowest bidder
    // TODO: Actually remove from working memory
    // working_memory_remove(ctx->wm, lowest_idx);

    ctx->slots[lowest_idx].current_bid = bid;
    ctx->slots[lowest_idx].initial_bid = bid;
    ctx->slots[lowest_idx].age_cycles = 0;
    ctx->slots[lowest_idx].rehearsal_count = 0;
    ctx->slots[lowest_idx].added_time_ms = now;
    ctx->slots[lowest_idx].last_access_time_ms = now;

    ctx->total_additions++;
    ctx->total_evictions++;  // Count preemption as eviction
    *slot_index = lowest_idx;

    return NIMCP_SUCCESS;
}

nimcp_error_t gt_wm_refresh(
    gt_wm_auction_ctx_t ctx,
    uint32_t slot_index,
    float new_bid
) {
    if (!ctx) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    if (slot_index >= ctx->capacity) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    if (!ctx->slots[slot_index].occupied) {
        return NIMCP_GT_ERROR_PLAYER_NOT_FOUND;
    }

    uint64_t now = get_time_ms();

    // Refresh bid
    if (new_bid > 0.0f) {
        ctx->slots[slot_index].current_bid = new_bid;
    } else {
        // Restore to initial bid
        ctx->slots[slot_index].current_bid = ctx->slots[slot_index].initial_bid;
    }

    ctx->slots[slot_index].last_access_time_ms = now;
    ctx->slots[slot_index].rehearsal_count++;

    return NIMCP_SUCCESS;
}

nimcp_error_t gt_wm_run_eviction(
    gt_wm_auction_ctx_t ctx,
    gt_wm_eviction_result_t* result
) {
    if (!ctx || !result) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    memset(result, 0, sizeof(gt_wm_eviction_result_t));

    result->occupancy_before = ctx->occupancy;

    if (ctx->occupancy == 0) {
        result->occupancy_after = 0;
        return NIMCP_SUCCESS;
    }

    float congestion_cost = compute_congestion_cost(ctx);
    result->total_congestion_cost = congestion_cost;

    // Collect effective values and find eviction candidates
    typedef struct {
        uint32_t index;
        float effective_value;
    } slot_value_t;

    slot_value_t values[MAX_WM_SLOTS];
    uint32_t num_occupied = 0;

    for (uint32_t i = 0; i < ctx->capacity; i++) {
        if (ctx->slots[i].occupied) {
            values[num_occupied].index = i;
            values[num_occupied].effective_value =
                ctx->slots[i].current_bid - congestion_cost;
            num_occupied++;
        }
    }

    // Sort by effective value (lowest first)
    for (uint32_t i = 0; i < num_occupied; i++) {
        for (uint32_t j = i + 1; j < num_occupied; j++) {
            if (values[j].effective_value < values[i].effective_value) {
                slot_value_t tmp = values[i];
                values[i] = values[j];
                values[j] = tmp;
            }
        }
    }

    // Evict items below reserve price
    result->num_evicted = 0;
    float effective_reserve = ctx->config.slot_reserve_price;

    for (uint32_t i = 0; i < num_occupied && result->num_evicted < MAX_EVICTIONS_PER_ROUND; i++) {
        if (values[i].effective_value < effective_reserve) {
            uint32_t slot_idx = values[i].index;

            // Evict this slot
            result->evicted_indices[result->num_evicted++] = slot_idx;

            ctx->slots[slot_idx].occupied = false;
            ctx->slots[slot_idx].current_bid = 0.0f;
            ctx->occupancy--;
            ctx->total_evictions++;

            // TODO: Actually remove from working memory
            // working_memory_remove(ctx->wm, slot_idx);
        }
    }

    // Find lowest surviving bid
    result->lowest_surviving_bid = 0.0f;
    bool found_survivor = false;
    for (uint32_t i = 0; i < ctx->capacity; i++) {
        if (ctx->slots[i].occupied) {
            if (!found_survivor || ctx->slots[i].current_bid < result->lowest_surviving_bid) {
                result->lowest_surviving_bid = ctx->slots[i].current_bid;
                found_survivor = true;
            }
        }
    }

    result->occupancy_after = ctx->occupancy;

    // Increment age for all remaining items
    for (uint32_t i = 0; i < ctx->capacity; i++) {
        if (ctx->slots[i].occupied) {
            ctx->slots[i].age_cycles++;
        }
    }

    ctx->eviction_rounds++;
    ctx->last_eviction_time_ms = get_time_ms();

    return NIMCP_SUCCESS;
}

nimcp_error_t gt_wm_apply_decay(
    gt_wm_auction_ctx_t ctx,
    float decay_amount
) {
    if (!ctx) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    float decay = (decay_amount > 0.0f) ? decay_amount : ctx->config.decay_rate;

    for (uint32_t i = 0; i < ctx->capacity; i++) {
        if (ctx->slots[i].occupied) {
            ctx->slots[i].current_bid *= (1.0f - decay);
            if (ctx->slots[i].current_bid < 0.0f) {
                ctx->slots[i].current_bid = 0.0f;
            }
        }
    }

    ctx->last_decay_time_ms = get_time_ms();

    return NIMCP_SUCCESS;
}

float gt_wm_get_congestion_cost(const gt_wm_auction_ctx_t ctx) {
    return compute_congestion_cost((gt_wm_auction_ctx_t)ctx);
}

//=============================================================================
// Query Functions
//=============================================================================

nimcp_error_t gt_wm_get_slot_state(
    const gt_wm_auction_ctx_t ctx,
    uint32_t slot_index,
    gt_wm_slot_state_t* state
) {
    if (!ctx || !state) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    if (slot_index >= ctx->capacity) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    if (!ctx->slots[slot_index].occupied) {
        return NIMCP_GT_ERROR_PLAYER_NOT_FOUND;
    }

    const slot_auction_state_t* internal = &ctx->slots[slot_index];

    state->item_index = slot_index;
    state->current_bid = internal->current_bid;
    state->initial_bid = internal->initial_bid;
    state->congestion_cost = compute_congestion_cost((gt_wm_auction_ctx_t)ctx);
    state->effective_value = internal->current_bid - state->congestion_cost;
    state->age_cycles = internal->age_cycles;
    state->rehearsal_count = internal->rehearsal_count;
    state->last_access_time_ms = internal->last_access_time_ms;

    return NIMCP_SUCCESS;
}

working_memory_t* gt_wm_get_wm(const gt_wm_auction_ctx_t ctx) {
    return ctx ? ctx->wm : NULL;
}

uint32_t gt_wm_get_occupancy(const gt_wm_auction_ctx_t ctx) {
    return ctx ? ctx->occupancy : 0;
}

float gt_wm_get_highest_bid(const gt_wm_auction_ctx_t ctx) {
    if (!ctx || ctx->occupancy == 0) {
        return 0.0f;
    }

    float highest = 0.0f;
    bool found = false;

    for (uint32_t i = 0; i < ctx->capacity; i++) {
        if (ctx->slots[i].occupied) {
            if (!found || ctx->slots[i].current_bid > highest) {
                highest = ctx->slots[i].current_bid;
                found = true;
            }
        }
    }

    return highest;
}

float gt_wm_get_lowest_bid(const gt_wm_auction_ctx_t ctx) {
    if (!ctx || ctx->occupancy == 0) {
        return 0.0f;
    }

    float lowest = 0.0f;
    bool found = false;

    for (uint32_t i = 0; i < ctx->capacity; i++) {
        if (ctx->slots[i].occupied) {
            if (!found || ctx->slots[i].current_bid < lowest) {
                lowest = ctx->slots[i].current_bid;
                found = true;
            }
        }
    }

    return lowest;
}

bool gt_wm_is_active(const gt_wm_auction_ctx_t ctx) {
    return ctx ? ctx->active : false;
}
