/**
 * @file nimcp_global_workspace.c
 * @brief Implementation of Global Workspace Theory for conscious access
 *
 * IMPLEMENTATION NOTES:
 * - Winner-take-all competition via max(signal_strengths)
 * - Circular buffer for broadcast history
 * - Competition decay via exponential forgetting
 * - Refractory period enforced via timestamp checking
 * - Statistics accumulated atomically (no threading, single brain)
 *
 * PERFORMANCE:
 * - Competition: O(N) where N = competitors (typically <10)
 * - Broadcast read: O(D) where D = content dim (memcpy)
 * - History: O(1) append, O(H) retrieval
 *
 * MEMORY:
 * - Workspace: ~300 bytes base
 * - Content: capacity_dim × sizeof(float) × 2 (current + temp)
 * - History: capacity_dim × sizeof(float) × history_depth
 * - Competitors: ~100 bytes × MAX_COMPETITORS
 * Total: ~50KB typical (256 dim, 10 history, 32 competitors)
 *
 * @author NIMCP Development Team - Part J
 * @date 2025-11-11
 */

#include "nimcp_global_workspace.h"
#include "utils/memory/nimcp_memory.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <assert.h>

//=============================================================================
// Internal Structures
//=============================================================================

/**
 * @brief Global workspace implementation (opaque)
 *
 * WHAT: Internal representation of workspace
 * WHY:  Encapsulation - users see only opaque pointer
 * HOW:  Struct contains all state and buffers
 */
struct global_workspace_struct {
    // Configuration (immutable after creation)
    global_workspace_config_t config;

    // Current broadcast state
    workspace_broadcast_t current_broadcast;
    float* broadcast_content;            /**< Content buffer [capacity_dim] */

    // Competition pool
    competitor_entry_t competitors[GLOBAL_WORKSPACE_MAX_COMPETITORS];
    uint32_t num_active_competitors;

    // Subscribers
    cognitive_module_t subscribers[GLOBAL_WORKSPACE_MAX_SUBSCRIBERS];
    uint32_t num_subscribers;

    // Broadcast history (circular buffer)
    workspace_broadcast_t* history;      /**< [history_depth] */
    float** history_content;             /**< [history_depth][capacity_dim] */
    uint32_t history_head;               /**< Circular buffer index */
    uint32_t history_count;              /**< How many in history */

    // Statistics
    workspace_statistics_t stats;

    // Timing state
    uint64_t last_broadcast_time_ms;
    uint64_t pool_activation_time_ms;  /**< When pool went from empty to non-empty */
    uint32_t next_broadcast_id;

    // Round-robin state (for ROUND_ROBIN strategy)
    uint32_t last_winner_idx;
};

//=============================================================================
// Internal Helper Functions
//=============================================================================

/**
 * @brief Get current time in milliseconds
 *
 * WHAT: Monotonic timestamp for timing operations
 * WHY:  Need consistent time source for refractory period, decay
 * HOW:  clock_gettime(CLOCK_MONOTONIC)
 *
 * @return Milliseconds since arbitrary epoch
 */
static uint64_t get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000ULL + (uint64_t)ts.tv_nsec / 1000000ULL;
}

/**
 * @brief Clamp float value to range [min, max]
 */
static inline float clamp_float(float value, float min, float max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
}

/**
 * @brief Apply exponential decay to competitor signal
 *
 * WHAT: Reduce strength of stale competition entries
 * WHY:  Prevent old submissions from lingering indefinitely
 * HOW:  strength *= exp(-dt / tau)
 *
 * @param original_strength Initial strength
 * @param age_ms How long since submission
 * @param tau_ms Time constant
 * @return Decayed strength
 */
static float apply_decay(float original_strength, uint64_t age_ms, float tau_ms) {
    if (tau_ms <= 0.0f) return original_strength;
    float decay_factor = expf(-(float)age_ms / tau_ms);
    return original_strength * decay_factor;
}

/**
 * @brief Resolve winner-take-all competition
 *
 * WHAT: Find strongest competitor above threshold
 * WHY:  Most biologically realistic competition mechanism
 * HOW:  argmax(strengths) where strength >= ignition_threshold
 *
 * @param workspace Workspace instance
 * @param winner_idx Output: index of winner in competitors array
 * @param winner_strength Output: strength of winner (after decay)
 * @return true if winner found (above threshold), false otherwise
 */
static bool resolve_winner_take_all(
    struct global_workspace_struct* workspace,
    uint32_t* winner_idx,
    float* winner_strength)
{
    float max_strength = 0.0f;
    uint32_t max_idx = 0;
    bool found_any = false;
    uint64_t current_time = get_time_ms();

    // Find strongest competitor (with decay applied)
    for (uint32_t i = 0; i < GLOBAL_WORKSPACE_MAX_COMPETITORS; i++) {
        if (!workspace->competitors[i].is_active) continue;

        // Apply decay
        uint64_t age_ms = current_time - workspace->competitors[i].timestamp_ms;
        float decayed_strength = apply_decay(
            workspace->competitors[i].strength,
            age_ms,
            workspace->config.competition_decay_tau_ms
        );

        // Prune if decayed below minimum
        if (decayed_strength < GLOBAL_WORKSPACE_MIN_IGNITION_THRESHOLD) {
            workspace->competitors[i].is_active = false;
            workspace->num_active_competitors--;
            continue;
        }

        // Track maximum
        if (decayed_strength > max_strength) {
            max_strength = decayed_strength;
            max_idx = i;
            found_any = true;
        }
    }

    // Check if winner exceeds ignition threshold
    if (found_any && max_strength >= workspace->config.ignition_threshold) {
        *winner_idx = max_idx;
        *winner_strength = max_strength;
        return true;
    }

    return false;  // No winner above threshold
}

/**
 * @brief Resolve priority-based competition
 *
 * WHAT: Select highest priority module, then strongest
 * WHY:  Some modules are more important (safety, pain, etc.)
 * HOW:  argmax(priorities), tiebreak with strength
 *
 * @param workspace Workspace instance
 * @param winner_idx Output: index of winner
 * @param winner_strength Output: strength of winner
 * @return true if winner found
 */
static bool resolve_priority_based(
    struct global_workspace_struct* workspace,
    uint32_t* winner_idx,
    float* winner_strength)
{
    float max_priority = -1.0f;
    float max_strength_at_priority = 0.0f;
    uint32_t max_idx = 0;
    bool found_any = false;
    uint64_t current_time = get_time_ms();

    for (uint32_t i = 0; i < GLOBAL_WORKSPACE_MAX_COMPETITORS; i++) {
        if (!workspace->competitors[i].is_active) continue;

        // Apply decay
        uint64_t age_ms = current_time - workspace->competitors[i].timestamp_ms;
        float decayed_strength = apply_decay(
            workspace->competitors[i].strength,
            age_ms,
            workspace->config.competition_decay_tau_ms
        );

        if (decayed_strength < GLOBAL_WORKSPACE_MIN_IGNITION_THRESHOLD) {
            workspace->competitors[i].is_active = false;
            workspace->num_active_competitors--;
            continue;
        }

        // Get module priority
        cognitive_module_t module = workspace->competitors[i].module;
        float priority = (module < MODULE_CUSTOM_START) ?
                          workspace->config.module_priorities[module] : 0.5f;

        // Compare priority first, strength second
        if (priority > max_priority ||
            (priority == max_priority && decayed_strength > max_strength_at_priority)) {
            max_priority = priority;
            max_strength_at_priority = decayed_strength;
            max_idx = i;
            found_any = true;
        }
    }

    // Check ignition threshold
    if (found_any && max_strength_at_priority >= workspace->config.ignition_threshold) {
        *winner_idx = max_idx;
        *winner_strength = max_strength_at_priority;
        return true;
    }

    return false;
}

/**
 * @brief Resolve round-robin competition
 *
 * WHAT: Take turns among competitors
 * WHY:  Ensure fairness, prevent starvation
 * HOW:  Select next in sequence after last winner
 *
 * @param workspace Workspace instance
 * @param winner_idx Output: index of winner
 * @param winner_strength Output: strength of winner
 * @return true if winner found
 */
static bool resolve_round_robin(
    struct global_workspace_struct* workspace,
    uint32_t* winner_idx,
    float* winner_strength)
{
    if (workspace->num_active_competitors == 0) return false;

    uint64_t current_time = get_time_ms();
    uint32_t start_idx = (workspace->last_winner_idx + 1) % GLOBAL_WORKSPACE_MAX_COMPETITORS;

    // Find next active competitor after last winner
    for (uint32_t offset = 0; offset < GLOBAL_WORKSPACE_MAX_COMPETITORS; offset++) {
        uint32_t idx = (start_idx + offset) % GLOBAL_WORKSPACE_MAX_COMPETITORS;
        if (!workspace->competitors[idx].is_active) continue;

        // Apply decay
        uint64_t age_ms = current_time - workspace->competitors[idx].timestamp_ms;
        float decayed_strength = apply_decay(
            workspace->competitors[idx].strength,
            age_ms,
            workspace->config.competition_decay_tau_ms
        );

        if (decayed_strength < GLOBAL_WORKSPACE_MIN_IGNITION_THRESHOLD) {
            workspace->competitors[idx].is_active = false;
            workspace->num_active_competitors--;
            continue;
        }

        // Found next active competitor
        *winner_idx = idx;
        *winner_strength = decayed_strength;
        workspace->last_winner_idx = idx;
        return true;
    }

    return false;
}

/**
 * @brief Broadcast winner's content to workspace
 *
 * WHAT: Update workspace with winning content, notify subscribers
 * WHY:  Make content globally available
 * HOW:  Copy content, update broadcast state, add to history
 *
 * @param workspace Workspace instance
 * @param winner_idx Index of winning competitor
 * @param winner_strength Decayed strength of winner
 */
static void broadcast_winner(
    struct global_workspace_struct* workspace,
    uint32_t winner_idx,
    float winner_strength)
{
    competitor_entry_t* winner = &workspace->competitors[winner_idx];

    // Copy content to broadcast buffer
    memcpy(workspace->broadcast_content, winner->content,
           workspace->config.capacity_dim * sizeof(float));

    // Update broadcast state
    workspace->current_broadcast.content = workspace->broadcast_content;
    workspace->current_broadcast.content_dim = workspace->config.capacity_dim;
    workspace->current_broadcast.source_module = winner->module;
    workspace->current_broadcast.source_strength = winner_strength;
    workspace->current_broadcast.broadcast_timestamp_ms = get_time_ms();
    workspace->current_broadcast.broadcast_id = workspace->next_broadcast_id++;
    workspace->current_broadcast.num_competitors = workspace->num_active_competitors;
    workspace->current_broadcast.is_valid = true;

    // Find runner-up strength (for statistics)
    float runner_up = 0.0f;
    for (uint32_t i = 0; i < GLOBAL_WORKSPACE_MAX_COMPETITORS; i++) {
        if (i == winner_idx || !workspace->competitors[i].is_active) continue;
        if (workspace->competitors[i].strength > runner_up) {
            runner_up = workspace->competitors[i].strength;
        }
    }
    workspace->current_broadcast.runner_up_strength = runner_up;

    // Update timing
    workspace->last_broadcast_time_ms = workspace->current_broadcast.broadcast_timestamp_ms;

    // Add to history (if enabled)
    if (workspace->config.enable_history && workspace->history != NULL) {
        uint32_t hist_idx = workspace->history_head;

        // Copy broadcast metadata
        workspace->history[hist_idx] = workspace->current_broadcast;

        // Copy content
        memcpy(workspace->history_content[hist_idx], workspace->broadcast_content,
               workspace->config.capacity_dim * sizeof(float));

        // Update history to point to copied content
        workspace->history[hist_idx].content = workspace->history_content[hist_idx];

        // Advance circular buffer
        workspace->history_head = (workspace->history_head + 1) % workspace->config.history_depth;
        if (workspace->history_count < workspace->config.history_depth) {
            workspace->history_count++;
        }
    }

    // Update statistics
    if (workspace->config.enable_statistics) {
        workspace->stats.total_broadcasts++;
        if (winner->module < MODULE_CUSTOM_START) {
            workspace->stats.broadcasts_per_module[winner->module]++;
        }
    }

    // Note: Winner removal now happens in global_workspace_compete() after broadcast
}

//=============================================================================
// Lifecycle Functions
//=============================================================================

global_workspace_t* global_workspace_create(void) {
    global_workspace_config_t config = global_workspace_default_config();
    return global_workspace_create_custom(&config);
}

global_workspace_t* global_workspace_create_custom(
    const global_workspace_config_t* config)
{
    // Validate configuration
    if (config != NULL) {
        char error[256];
        if (!global_workspace_validate_config(config, error, sizeof(error))) {
            fprintf(stderr, "Global workspace creation failed: %s\n", error);
            return NULL;
        }
    }

    // Use defaults if NULL config
    global_workspace_config_t actual_config;
    if (config != NULL) {
        actual_config = *config;
    } else {
        actual_config = global_workspace_default_config();
    }

    // Allocate workspace structure
    struct global_workspace_struct* workspace =
        (struct global_workspace_struct*)nimcp_calloc(1, sizeof(struct global_workspace_struct));
    if (workspace == NULL) {
        fprintf(stderr, "Failed to allocate workspace structure\n");
        return NULL;
    }

    // Copy configuration
    workspace->config = actual_config;

    // Allocate broadcast content buffer
    workspace->broadcast_content =
        (float*)nimcp_calloc(actual_config.capacity_dim, sizeof(float));
    if (workspace->broadcast_content == NULL) {
        fprintf(stderr, "Failed to allocate broadcast content buffer\n");
        nimcp_free(workspace);
        return NULL;
    }

    // Initialize broadcast state
    workspace->current_broadcast.content = workspace->broadcast_content;
    workspace->current_broadcast.content_dim = actual_config.capacity_dim;
    workspace->current_broadcast.is_valid = false;

    // Allocate history (if enabled)
    if (actual_config.enable_history && actual_config.history_depth > 0) {
        workspace->history = (workspace_broadcast_t*)nimcp_calloc(
            actual_config.history_depth, sizeof(workspace_broadcast_t));
        if (workspace->history == NULL) {
            fprintf(stderr, "Failed to allocate history buffer\n");
            nimcp_free(workspace->broadcast_content);
            nimcp_free(workspace);
            return NULL;
        }

        // Allocate content buffers for history
        workspace->history_content = (float**)nimcp_calloc(
            actual_config.history_depth, sizeof(float*));
        if (workspace->history_content == NULL) {
            fprintf(stderr, "Failed to allocate history content array\n");
            nimcp_free(workspace->history);
            nimcp_free(workspace->broadcast_content);
            nimcp_free(workspace);
            return NULL;
        }

        for (uint32_t i = 0; i < actual_config.history_depth; i++) {
            workspace->history_content[i] = (float*)nimcp_calloc(
                actual_config.capacity_dim, sizeof(float));
            if (workspace->history_content[i] == NULL) {
                fprintf(stderr, "Failed to allocate history content buffer %u\n", i);
                // Clean up previously allocated
                for (uint32_t j = 0; j < i; j++) {
                    nimcp_free(workspace->history_content[j]);
                }
                nimcp_free(workspace->history_content);
                nimcp_free(workspace->history);
                nimcp_free(workspace->broadcast_content);
                nimcp_free(workspace);
                return NULL;
            }
        }
    }

    // Initialize competitor pool
    for (uint32_t i = 0; i < GLOBAL_WORKSPACE_MAX_COMPETITORS; i++) {
        workspace->competitors[i].is_active = false;
    }
    workspace->num_active_competitors = 0;

    // Initialize subscribers
    workspace->num_subscribers = 0;

    // Initialize timing
    workspace->last_broadcast_time_ms = 0;
    workspace->pool_activation_time_ms = 0;
    workspace->next_broadcast_id = 1;
    workspace->last_winner_idx = 0;

    // Initialize statistics
    memset(&workspace->stats, 0, sizeof(workspace_statistics_t));

    // Note: global_workspace_t is typedef'd as a pointer, so function signature
    // expects global_workspace_t* (double pointer). Cast to match.
    return (global_workspace_t*)workspace;
}

void global_workspace_destroy(global_workspace_t* workspace) {
    if (workspace == NULL) return;

    struct global_workspace_struct* ws = (struct global_workspace_struct*)workspace;

    // Free history content buffers
    if (ws->history_content != NULL) {
        for (uint32_t i = 0; i < ws->config.history_depth; i++) {
            nimcp_free(ws->history_content[i]);
        }
        nimcp_free(ws->history_content);
    }

    // Free history metadata
    nimcp_free(ws->history);

    // Free broadcast content
    nimcp_free(ws->broadcast_content);

    // Free workspace structure
    nimcp_free(ws);
}

//=============================================================================
// Core Operations
//=============================================================================

bool global_workspace_compete(
    global_workspace_t* workspace,
    cognitive_module_t module,
    const float* content,
    uint32_t content_dim,
    float strength)
{
    // WHAT: Submit content and immediately resolve competition
    // WHY:  Backward compatibility - original API auto-resolves
    // HOW:  Use submit() + resolve() internally

    // Guard: NULL checks
    if (workspace == NULL || content == NULL) {
        return false;
    }

    // Submit to competition pool
    if (!global_workspace_submit(workspace, module, content, content_dim, strength)) {
        return false;
    }

    // Immediately resolve competition (backward compatible behavior)
    cognitive_module_t winner = MODULE_NONE;
    bool broadcast_occurred = global_workspace_resolve(workspace, &winner);

    // Return true only if THIS module won and was broadcast
    return (broadcast_occurred && winner == module);
}

bool global_workspace_submit(
    global_workspace_t* workspace,
    cognitive_module_t module,
    const float* content,
    uint32_t content_dim,
    float strength)
{
    // Guard: NULL checks
    if (workspace == NULL) {
        fprintf(stderr, "NULL workspace in global_workspace_submit\n");
        return false;
    }

    if (content == NULL) {
        fprintf(stderr, "NULL content in global_workspace_submit\n");
        return false;
    }

    struct global_workspace_struct* ws = (struct global_workspace_struct*)workspace;

    // Guard: Validate content dimension
    if (content_dim != ws->config.capacity_dim) {
        fprintf(stderr, "Content dimension mismatch in global_workspace_submit: "
                "expected %u, got %u\n", ws->config.capacity_dim, content_dim);
        return false;
    }

    // Guard: Validate strength range
    if (strength < 0.0f || strength > 1.0f) {
        fprintf(stderr, "Invalid strength in global_workspace_submit: %.2f "
                "(must be 0.0-1.0)\n", strength);
        return false;
    }

    // Update statistics
    if (ws->config.enable_statistics) {
        ws->stats.total_competitions++;
        if (module < MODULE_CUSTOM_START) {
            ws->stats.competitions_per_module[module]++;
        }
    }

    // Find slot for this module (update existing or find empty)
    uint32_t slot_idx = GLOBAL_WORKSPACE_MAX_COMPETITORS;

    // Check if module already in pool (update case)
    for (uint32_t i = 0; i < GLOBAL_WORKSPACE_MAX_COMPETITORS; i++) {
        if (ws->competitors[i].is_active && ws->competitors[i].module == module) {
            slot_idx = i;
            break;
        }
    }

    // If not found, find empty slot
    if (slot_idx == GLOBAL_WORKSPACE_MAX_COMPETITORS) {
        for (uint32_t i = 0; i < GLOBAL_WORKSPACE_MAX_COMPETITORS; i++) {
            if (!ws->competitors[i].is_active) {
                slot_idx = i;
                break;
            }
        }
    }

    // Guard: Check if pool is full
    if (slot_idx == GLOBAL_WORKSPACE_MAX_COMPETITORS) {
        fprintf(stderr, "Competition pool full in global_workspace_submit "
                "(%u competitors)\n", GLOBAL_WORKSPACE_MAX_COMPETITORS);
        return false;
    }

    // Track if we're adding new competitor vs updating existing
    bool was_inactive = !ws->competitors[slot_idx].is_active;
    bool pool_was_empty = (ws->num_active_competitors == 0);

    // Add/update competitor in pool
    ws->competitors[slot_idx].module = module;
    ws->competitors[slot_idx].content = (float*)content;  // Caller owns content
    ws->competitors[slot_idx].content_dim = content_dim;
    ws->competitors[slot_idx].strength = strength;
    ws->competitors[slot_idx].timestamp_ms = get_time_ms();
    ws->competitors[slot_idx].is_active = true;

    // Update pool counts
    if (was_inactive) {
        ws->num_active_competitors++;

        // Track when pool first becomes active
        if (pool_was_empty) {
            ws->pool_activation_time_ms = get_time_ms();
        }
    }

    return true;
}

bool global_workspace_resolve(
    global_workspace_t* workspace,
    cognitive_module_t* winning_module)
{
    // Guard: NULL checks
    if (workspace == NULL) {
        fprintf(stderr, "NULL workspace in global_workspace_resolve\n");
        return false;
    }

    struct global_workspace_struct* ws = (struct global_workspace_struct*)workspace;

    // Set default output (no winner)
    if (winning_module != NULL) {
        *winning_module = MODULE_NONE;
    }

    // Check if pool is empty
    if (ws->num_active_competitors == 0) {
        // No competitors to resolve
        return false;
    }

    uint64_t current_time = get_time_ms();

    // Run competition resolution based on strategy
    uint32_t winner_idx;
    float winner_strength;
    bool found_winner = false;

    uint64_t competition_start = get_time_ms();

    switch (ws->config.strategy) {
        case COMPETITION_WINNER_TAKE_ALL:
            found_winner = resolve_winner_take_all(ws, &winner_idx, &winner_strength);
            break;

        case COMPETITION_PRIORITY_BASED:
            found_winner = resolve_priority_based(ws, &winner_idx, &winner_strength);
            break;

        case COMPETITION_ROUND_ROBIN:
            found_winner = resolve_round_robin(ws, &winner_idx, &winner_strength);
            break;

        case COMPETITION_WEIGHTED_FUSION:
            fprintf(stderr, "WEIGHTED_FUSION strategy not yet implemented\n");
            found_winner = resolve_winner_take_all(ws, &winner_idx, &winner_strength);
            break;

        default:
            fprintf(stderr, "Unknown competition strategy: %d\n", ws->config.strategy);
            found_winner = false;
            break;
    }

    // Update competition latency statistics
    if (ws->config.enable_statistics) {
        uint64_t competition_end = get_time_ms();
        uint64_t latency_us = (competition_end - competition_start) * 1000;
        if (ws->stats.total_competitions == 1) {
            ws->stats.avg_competition_latency_us = latency_us;
            ws->stats.max_competition_latency_us = latency_us;
        } else {
            ws->stats.avg_competition_latency_us =
                (ws->stats.avg_competition_latency_us * (ws->stats.total_competitions - 1) +
                 latency_us) / ws->stats.total_competitions;
            if (latency_us > ws->stats.max_competition_latency_us) {
                ws->stats.max_competition_latency_us = latency_us;
            }
        }
    }

    // If no winner found, clear pool and return
    if (!found_winner) {
        // Clear pool (all below threshold or pruned by decay)
        for (uint32_t i = 0; i < GLOBAL_WORKSPACE_MAX_COMPETITORS; i++) {
            if (ws->competitors[i].is_active) {
                ws->competitors[i].is_active = false;
            }
        }
        ws->num_active_competitors = 0;
        ws->pool_activation_time_ms = 0;

        if (ws->config.enable_statistics) {
            ws->stats.rejected_submissions++;
        }

        return false;
    }

    // Check refractory period before broadcasting
    bool can_broadcast = true;
    if (ws->last_broadcast_time_ms > 0) {
        uint64_t time_since_broadcast = current_time - ws->last_broadcast_time_ms;
        if (time_since_broadcast < ws->config.refractory_period_ms) {
            can_broadcast = false;
            if (ws->config.enable_statistics) {
                ws->stats.refractory_violations++;
            }
        }
    }

    // Broadcast if allowed
    if (can_broadcast) {
        broadcast_winner(ws, winner_idx, winner_strength);

        // Set output: which module won
        if (winning_module != NULL) {
            *winning_module = ws->competitors[winner_idx].module;
        }

        // Clear the entire competition pool after broadcasting
        for (uint32_t i = 0; i < GLOBAL_WORKSPACE_MAX_COMPETITORS; i++) {
            if (ws->competitors[i].is_active) {
                ws->competitors[i].is_active = false;
            }
        }
        ws->num_active_competitors = 0;
        ws->pool_activation_time_ms = 0;

        return true;
    } else {
        // Winner found but blocked by refractory period
        // Keep competitors in pool for next resolve attempt
        return false;
    }
}

bool global_workspace_read_broadcast(
    global_workspace_t* workspace,
    float* content,
    uint32_t max_dim,
    uint32_t* actual_dim,
    cognitive_module_t* source)
{
    if (workspace == NULL || content == NULL) return false;

    struct global_workspace_struct* ws = (struct global_workspace_struct*)workspace;

    // Check if broadcast available
    if (!ws->current_broadcast.is_valid) {
        return false;
    }

    // Check buffer size
    if (max_dim < ws->current_broadcast.content_dim) {
        fprintf(stderr, "Buffer too small: need %u, have %u\n",
                ws->current_broadcast.content_dim, max_dim);
        return false;
    }

    // Copy content
    memcpy(content, ws->current_broadcast.content,
           ws->current_broadcast.content_dim * sizeof(float));

    // Return metadata
    if (actual_dim != NULL) {
        *actual_dim = ws->current_broadcast.content_dim;
    }
    if (source != NULL) {
        *source = ws->current_broadcast.source_module;
    }

    return true;
}

bool global_workspace_subscribe(
    global_workspace_t* workspace,
    cognitive_module_t module)
{
    if (workspace == NULL) return false;

    struct global_workspace_struct* ws = (struct global_workspace_struct*)workspace;

    // Check if already subscribed
    for (uint32_t i = 0; i < ws->num_subscribers; i++) {
        if (ws->subscribers[i] == module) {
            return true;  // Already subscribed (idempotent)
        }
    }

    // Check if room for more
    if (ws->num_subscribers >= GLOBAL_WORKSPACE_MAX_SUBSCRIBERS) {
        fprintf(stderr, "Subscriber list full (%u max)\n", GLOBAL_WORKSPACE_MAX_SUBSCRIBERS);
        return false;
    }

    // Add subscriber
    ws->subscribers[ws->num_subscribers++] = module;

    // Update statistics
    if (ws->config.enable_statistics) {
        ws->stats.current_subscribers = ws->num_subscribers;
    }

    return true;
}

bool global_workspace_unsubscribe(
    global_workspace_t* workspace,
    cognitive_module_t module)
{
    if (workspace == NULL) return false;

    struct global_workspace_struct* ws = (struct global_workspace_struct*)workspace;

    // Find module in subscriber list
    for (uint32_t i = 0; i < ws->num_subscribers; i++) {
        if (ws->subscribers[i] == module) {
            // Remove by shifting remaining elements
            for (uint32_t j = i; j < ws->num_subscribers - 1; j++) {
                ws->subscribers[j] = ws->subscribers[j + 1];
            }
            ws->num_subscribers--;

            // Update statistics
            if (ws->config.enable_statistics) {
                ws->stats.current_subscribers = ws->num_subscribers;
            }

            return true;
        }
    }

    return false;  // Not subscribed
}

//=============================================================================
// Query Functions
//=============================================================================

bool global_workspace_has_broadcast(const global_workspace_t* workspace) {
    if (workspace == NULL) return false;
    const struct global_workspace_struct* ws =
        (const struct global_workspace_struct*)workspace;
    return ws->current_broadcast.is_valid;
}

cognitive_module_t global_workspace_get_broadcast_source(
    const global_workspace_t* workspace)
{
    if (workspace == NULL) return MODULE_NONE;
    const struct global_workspace_struct* ws =
        (const struct global_workspace_struct*)workspace;
    if (!ws->current_broadcast.is_valid) return MODULE_NONE;
    return ws->current_broadcast.source_module;
}

float global_workspace_get_broadcast_strength(
    const global_workspace_t* workspace)
{
    if (workspace == NULL) return 0.0f;
    const struct global_workspace_struct* ws =
        (const struct global_workspace_struct*)workspace;
    if (!ws->current_broadcast.is_valid) return 0.0f;
    return ws->current_broadcast.source_strength;
}

uint32_t global_workspace_get_subscriber_count(const global_workspace_t* workspace) {
    if (workspace == NULL) return 0;
    const struct global_workspace_struct* ws =
        (const struct global_workspace_struct*)workspace;
    return ws->num_subscribers;
}

uint32_t global_workspace_get_competitor_count(const global_workspace_t* workspace) {
    if (workspace == NULL) return 0;
    const struct global_workspace_struct* ws =
        (const struct global_workspace_struct*)workspace;
    return ws->num_active_competitors;
}

bool global_workspace_is_competing(
    const global_workspace_t* workspace,
    cognitive_module_t module)
{
    if (workspace == NULL) return false;
    const struct global_workspace_struct* ws =
        (const struct global_workspace_struct*)workspace;

    for (uint32_t i = 0; i < GLOBAL_WORKSPACE_MAX_COMPETITORS; i++) {
        if (ws->competitors[i].is_active && ws->competitors[i].module == module) {
            return true;
        }
    }
    return false;
}

//=============================================================================
// History Functions
//=============================================================================

bool global_workspace_get_history(
    const global_workspace_t* workspace,
    workspace_broadcast_t* history,
    uint32_t max_history,
    uint32_t* actual_count)
{
    if (workspace == NULL || history == NULL || actual_count == NULL) return false;

    const struct global_workspace_struct* ws =
        (const struct global_workspace_struct*)workspace;

    if (!ws->config.enable_history || ws->history == NULL) {
        *actual_count = 0;
        return false;
    }

    // Copy history (most recent first)
    uint32_t count = (ws->history_count < max_history) ? ws->history_count : max_history;
    *actual_count = count;

    for (uint32_t i = 0; i < count; i++) {
        // Calculate circular buffer index (most recent first)
        uint32_t idx = (ws->history_head + ws->config.history_depth - 1 - i) %
                       ws->config.history_depth;
        history[i] = ws->history[idx];
    }

    return true;
}

uint64_t global_workspace_time_since_broadcast(
    const global_workspace_t* workspace,
    uint64_t current_time_ms)
{
    if (workspace == NULL) return UINT64_MAX;

    const struct global_workspace_struct* ws =
        (const struct global_workspace_struct*)workspace;

    if (ws->last_broadcast_time_ms == 0) {
        return UINT64_MAX;  // Never broadcast
    }

    if (current_time_ms < ws->last_broadcast_time_ms) {
        return 0;  // Time went backwards?
    }

    return current_time_ms - ws->last_broadcast_time_ms;
}

//=============================================================================
// Statistics Functions
//=============================================================================

bool global_workspace_get_statistics(
    const global_workspace_t* workspace,
    workspace_statistics_t* stats)
{
    if (workspace == NULL || stats == NULL) return false;

    const struct global_workspace_struct* ws =
        (const struct global_workspace_struct*)workspace;

    if (!ws->config.enable_statistics) {
        return false;
    }

    *stats = ws->stats;
    return true;
}

void global_workspace_reset_statistics(global_workspace_t* workspace) {
    if (workspace == NULL) return;

    struct global_workspace_struct* ws = (struct global_workspace_struct*)workspace;
    memset(&ws->stats, 0, sizeof(workspace_statistics_t));

    // Restore current counts
    ws->stats.current_subscribers = ws->num_subscribers;
    ws->stats.current_competitors = ws->num_active_competitors;
}

void global_workspace_print_state(
    const global_workspace_t* workspace,
    bool verbose)
{
    if (workspace == NULL) {
        fprintf(stderr, "Workspace is NULL\n");
        return;
    }

    const struct global_workspace_struct* ws =
        (const struct global_workspace_struct*)workspace;

    fprintf(stderr, "=== Global Workspace State ===\n");

    // Current broadcast
    if (ws->current_broadcast.is_valid) {
        fprintf(stderr, "Broadcast: %s (strength=%.2f, dim=%u, id=%u)\n",
                cognitive_module_to_string(ws->current_broadcast.source_module),
                ws->current_broadcast.source_strength,
                ws->current_broadcast.content_dim,
                ws->current_broadcast.broadcast_id);
        fprintf(stderr, "  Time: %lu ms ago\n",
                get_time_ms() - ws->current_broadcast.broadcast_timestamp_ms);
        fprintf(stderr, "  Competitors: %u (runner-up: %.2f)\n",
                ws->current_broadcast.num_competitors,
                ws->current_broadcast.runner_up_strength);

        if (verbose && ws->current_broadcast.content != NULL) {
            fprintf(stderr, "  Content (first 10): ");
            uint32_t print_count = (ws->current_broadcast.content_dim < 10) ?
                                    ws->current_broadcast.content_dim : 10;
            for (uint32_t i = 0; i < print_count; i++) {
                fprintf(stderr, "%.3f ", ws->current_broadcast.content[i]);
            }
            fprintf(stderr, "...\n");
        }
    } else {
        fprintf(stderr, "Broadcast: (none)\n");
    }

    // Competitors
    fprintf(stderr, "Competitors (%u active):\n", ws->num_active_competitors);
    for (uint32_t i = 0; i < GLOBAL_WORKSPACE_MAX_COMPETITORS; i++) {
        if (ws->competitors[i].is_active) {
            uint64_t age = get_time_ms() - ws->competitors[i].timestamp_ms;
            fprintf(stderr, "  %s (strength=%.2f, age=%lu ms)\n",
                    cognitive_module_to_string(ws->competitors[i].module),
                    ws->competitors[i].strength,
                    age);
        }
    }

    // Subscribers
    fprintf(stderr, "Subscribers (%u):\n", ws->num_subscribers);
    for (uint32_t i = 0; i < ws->num_subscribers; i++) {
        fprintf(stderr, "  %s\n", cognitive_module_to_string(ws->subscribers[i]));
    }

    // Configuration
    fprintf(stderr, "Configuration:\n");
    fprintf(stderr, "  Strategy: %s\n",
            competition_strategy_to_string(ws->config.strategy));
    fprintf(stderr, "  Ignition threshold: %.2f\n", ws->config.ignition_threshold);
    fprintf(stderr, "  Refractory period: %u ms\n", ws->config.refractory_period_ms);
    fprintf(stderr, "  Capacity: %u floats\n", ws->config.capacity_dim);

    // Statistics
    if (ws->config.enable_statistics) {
        fprintf(stderr, "Statistics:\n");
        fprintf(stderr, "  Total broadcasts: %lu\n", ws->stats.total_broadcasts);
        fprintf(stderr, "  Total competitions: %lu\n", ws->stats.total_competitions);
        fprintf(stderr, "  Rejected: %lu\n", ws->stats.rejected_submissions);
        fprintf(stderr, "  Refractory violations: %lu\n", ws->stats.refractory_violations);
        fprintf(stderr, "  Avg competition latency: %lu us\n",
                ws->stats.avg_competition_latency_us);
        fprintf(stderr, "  Max competition latency: %lu us\n",
                ws->stats.max_competition_latency_us);

        if (verbose) {
            fprintf(stderr, "  Per-module broadcasts:\n");
            for (cognitive_module_t m = MODULE_PERCEPTION; m < MODULE_CUSTOM_START; m++) {
                if (ws->stats.broadcasts_per_module[m] > 0) {
                    fprintf(stderr, "    %s: %lu\n",
                            cognitive_module_to_string(m),
                            ws->stats.broadcasts_per_module[m]);
                }
            }
        }
    }

    fprintf(stderr, "==============================\n");
}

//=============================================================================
// Configuration Functions
//=============================================================================

global_workspace_config_t global_workspace_default_config(void) {
    global_workspace_config_t config;

    config.capacity_dim = GLOBAL_WORKSPACE_DEFAULT_DIM;
    config.strategy = COMPETITION_WINNER_TAKE_ALL;
    config.ignition_threshold = GLOBAL_WORKSPACE_DEFAULT_IGNITION_THRESHOLD;
    config.refractory_period_ms = GLOBAL_WORKSPACE_REFRACTORY_PERIOD_MS;
    config.competition_decay_tau_ms = GLOBAL_WORKSPACE_COMPETITION_DECAY_TAU_MS;
    config.history_depth = GLOBAL_WORKSPACE_HISTORY_DEPTH;
    config.enable_history = true;
    config.enable_statistics = true;

    // Initialize all module priorities to 0.5 (normal)
    for (uint32_t i = 0; i < MODULE_CUSTOM_START; i++) {
        config.module_priorities[i] = 0.5f;
    }

    return config;
}

bool global_workspace_set_ignition_threshold(
    global_workspace_t* workspace,
    float new_threshold)
{
    if (workspace == NULL) return false;

    struct global_workspace_struct* ws = (struct global_workspace_struct*)workspace;

    // Clamp to valid range
    new_threshold = clamp_float(new_threshold,
                                 GLOBAL_WORKSPACE_MIN_IGNITION_THRESHOLD,
                                 GLOBAL_WORKSPACE_MAX_IGNITION_THRESHOLD);

    ws->config.ignition_threshold = new_threshold;
    return true;
}

float global_workspace_get_ignition_threshold(const global_workspace_t* workspace) {
    if (workspace == NULL) return 0.0f;
    const struct global_workspace_struct* ws =
        (const struct global_workspace_struct*)workspace;
    return ws->config.ignition_threshold;
}

bool global_workspace_set_module_priority(
    global_workspace_t* workspace,
    cognitive_module_t module,
    float priority)
{
    if (workspace == NULL) return false;
    if (module >= MODULE_CUSTOM_START) return false;  // Only for standard modules

    struct global_workspace_struct* ws = (struct global_workspace_struct*)workspace;

    // Clamp to valid range
    priority = clamp_float(priority, 0.0f, 1.0f);

    ws->config.module_priorities[module] = priority;
    return true;
}

//=============================================================================
// Utility Functions
//=============================================================================

const char* cognitive_module_to_string(cognitive_module_t module) {
    switch (module) {
        case MODULE_NONE: return "NONE";
        case MODULE_PERCEPTION: return "PERCEPTION";
        case MODULE_WORKING_MEMORY: return "WORKING_MEMORY";
        case MODULE_EXECUTIVE: return "EXECUTIVE";
        case MODULE_THEORY_OF_MIND: return "THEORY_OF_MIND";
        case MODULE_ETHICS: return "ETHICS";
        case MODULE_EPISODIC_MEMORY: return "EPISODIC_MEMORY";
        case MODULE_SEMANTIC_MEMORY: return "SEMANTIC_MEMORY";
        case MODULE_LANGUAGE: return "LANGUAGE";
        case MODULE_EMOTION: return "EMOTION";
        case MODULE_SALIENCE: return "SALIENCE";
        case MODULE_MOTOR: return "MOTOR";
        case MODULE_ATTENTION: return "ATTENTION";
        case MODULE_METACOGNITION: return "METACOGNITION";
        case MODULE_CURIOSITY: return "CURIOSITY";
        case MODULE_INTROSPECTION: return "INTROSPECTION";
        case MODULE_PREDICTIVE: return "PREDICTIVE";
        case MODULE_CONSOLIDATION: return "CONSOLIDATION";
        case MODULE_WELLBEING: return "WELLBEING";
        case MODULE_MENTAL_HEALTH: return "MENTAL_HEALTH";
        case MODULE_GOAL_MOTIVATION: return "GOAL_MOTIVATION";
        case MODULE_COGNITIVE_CONTROL: return "COGNITIVE_CONTROL";
        default:
            // Custom modules in reasonable range
            if (module >= MODULE_CUSTOM_START && module < MODULE_CUSTOM_START + 1000) {
                return "CUSTOM";
            }
            return "UNKNOWN";
    }
}

const char* competition_strategy_to_string(competition_strategy_t strategy) {
    switch (strategy) {
        case COMPETITION_WINNER_TAKE_ALL: return "WINNER_TAKE_ALL";
        case COMPETITION_WEIGHTED_FUSION: return "WEIGHTED_FUSION";
        case COMPETITION_PRIORITY_BASED: return "PRIORITY_BASED";
        case COMPETITION_ROUND_ROBIN: return "ROUND_ROBIN";
        default: return "UNKNOWN";
    }
}

bool global_workspace_validate_config(
    const global_workspace_config_t* config,
    char* error_msg,
    size_t error_msg_len)
{
    if (config == NULL) {
        if (error_msg != NULL && error_msg_len > 0) {
            snprintf(error_msg, error_msg_len, "Configuration is NULL");
        }
        return false;
    }

    // Check capacity_dim
    if (config->capacity_dim == 0) {
        if (error_msg != NULL && error_msg_len > 0) {
            snprintf(error_msg, error_msg_len, "capacity_dim must be > 0");
        }
        return false;
    }
    if (config->capacity_dim > GLOBAL_WORKSPACE_MAX_DIM) {
        if (error_msg != NULL && error_msg_len > 0) {
            snprintf(error_msg, error_msg_len, "capacity_dim %u exceeds maximum %u",
                     config->capacity_dim, GLOBAL_WORKSPACE_MAX_DIM);
        }
        return false;
    }

    // Check ignition_threshold
    if (config->ignition_threshold < GLOBAL_WORKSPACE_MIN_IGNITION_THRESHOLD ||
        config->ignition_threshold > GLOBAL_WORKSPACE_MAX_IGNITION_THRESHOLD) {
        if (error_msg != NULL && error_msg_len > 0) {
            snprintf(error_msg, error_msg_len,
                     "ignition_threshold %.2f out of range [%.2f, %.2f]",
                     config->ignition_threshold,
                     GLOBAL_WORKSPACE_MIN_IGNITION_THRESHOLD,
                     GLOBAL_WORKSPACE_MAX_IGNITION_THRESHOLD);
        }
        return false;
    }

    // Check refractory_period_ms
    if (config->refractory_period_ms == 0) {
        if (error_msg != NULL && error_msg_len > 0) {
            snprintf(error_msg, error_msg_len, "refractory_period_ms must be > 0");
        }
        return false;
    }

    // Check competition_decay_tau_ms
    if (config->competition_decay_tau_ms <= 0.0f) {
        if (error_msg != NULL && error_msg_len > 0) {
            snprintf(error_msg, error_msg_len, "competition_decay_tau_ms must be > 0");
        }
        return false;
    }

    // Check history_depth
    if (config->enable_history && config->history_depth == 0) {
        if (error_msg != NULL && error_msg_len > 0) {
            snprintf(error_msg, error_msg_len,
                     "history_depth must be > 0 when enable_history is true");
        }
        return false;
    }

    // All checks passed
    if (error_msg != NULL && error_msg_len > 0) {
        error_msg[0] = '\0';
    }
    return true;
}
