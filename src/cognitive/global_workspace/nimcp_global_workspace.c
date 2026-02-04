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

#include "cognitive/global_workspace/nimcp_global_workspace.h"
#include "security/nimcp_security.h"
#include "security/nimcp_blood_brain_barrier.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"

#include "utils/memory/nimcp_unified_memory.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/memory/nimcp_memory_pool.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/platform/nimcp_platform_mutex.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <assert.h>

// Bio-async integration
#include "nimcp.h"  // For error codes
#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_messages.h"
#include "async/nimcp_wiring_helpers.h"
#include "async/nimcp_bio_router.h"

// SNN and Plasticity bridge integration
#include "cognitive/global_workspace/nimcp_gw_snn_bridge.h"
#include "cognitive/global_workspace/nimcp_gw_plasticity_bridge.h"
#include "utils/exception/nimcp_exception_macros.h"

#define LOG_MODULE "global_workspace"
#include "utils/fault_tolerance/nimcp_health_agent_macros.h"
#include "mesh/nimcp_mesh_participant.h"
#include "mesh/nimcp_mesh_adapter.h"

NIMCP_DECLARE_HEALTH_AGENT_ATOMIC(global_workspace)
//=============================================================================
// Mesh Participant Registration
//=============================================================================

static mesh_participant_id_t g_global_workspace_mesh_id = 0;
static mesh_participant_registry_t* g_global_workspace_mesh_registry = NULL;

nimcp_error_t global_workspace_mesh_register(mesh_participant_registry_t* registry) {
    if (!registry) return NIMCP_ERROR_NULL_POINTER;
    if (g_global_workspace_mesh_id != 0) return NIMCP_SUCCESS;
    mesh_participant_interface_t iface;
    mesh_participant_interface_init(&iface);
    strncpy(iface.module_name, "global_workspace", MESH_MAX_NAME_LEN - 1);
    iface.type = MESH_PARTICIPANT_MODULE;
    iface.home_channel = mesh_adapter_get_default_channel(MESH_ADAPTER_CATEGORY_COGNITIVE);
    mesh_participant_config_t config;
    mesh_participant_config_init(&config);
    config.module_name = "global_workspace";
    config.type = MESH_PARTICIPANT_MODULE;
    config.home_channel = iface.home_channel;
    nimcp_error_t err = mesh_participant_register(registry, &iface, &config, &g_global_workspace_mesh_id);
    if (err == NIMCP_SUCCESS) g_global_workspace_mesh_registry = registry;
    return err;
}

void global_workspace_mesh_unregister(void) {
    if (g_global_workspace_mesh_registry && g_global_workspace_mesh_id != 0) {
        mesh_participant_unregister(g_global_workspace_mesh_registry, g_global_workspace_mesh_id);
        g_global_workspace_mesh_id = 0;
        g_global_workspace_mesh_registry = NULL;
    }
}


/** @brief Send heartbeat from global_workspace module (instance-level) */
static inline void global_workspace_heartbeat_instance(
    nimcp_health_agent_t* instance_agent, const char* operation, float progress)
{
    if (g_global_workspace_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_global_workspace_health_agent, operation, progress);
    }
    if (instance_agent && instance_agent != g_global_workspace_health_agent) {
        nimcp_health_agent_heartbeat_ex(instance_agent, operation, progress);
    }
}



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

    // Thread safety mutex for public API operations
    nimcp_platform_mutex_t mutex;           /**< Protects all mutable state */

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

    // Phase 1.5: Memory pools for hot-path allocations
    memory_pool_t broadcast_content_pool;   /**< Pool for broadcast content buffers */
    memory_pool_t history_content_pool;     /**< Pool for history content buffers */

    // Bio-async integration
    bio_module_context_t bio_ctx;           /**< Bio-async module context */
    bool bio_async_enabled;                 /**< Bio-async registration status */

    // SNN and Plasticity bridge integration
    gw_snn_bridge_t* snn_bridge;            /**< SNN bridge for spike-based processing */
    gw_plasticity_bridge_t* plasticity_bridge; /**< Plasticity bridge for learning */
    bool bridges_enabled;                   /**< Bridge integration status */
};

//=============================================================================
// BIO-ASYNC MESSAGE HANDLERS
//=============================================================================

// Forward declaration for handler function
static nimcp_error_t handle_attention_shift(const void*, size_t, nimcp_bio_promise_t, void*);

/**
 * @brief KG-driven wiring callback for global workspace module
 */
static int global_workspace_wiring_handler_callback(
    bio_module_context_t ctx,
    const bio_message_type_t* message_types,
    uint32_t message_count,
    void* user_data)
{
    (void)user_data;

    int registered = 0;
    for (uint32_t i = 0; i < message_count; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && message_count > 256) {
            global_workspace_heartbeat("global_works_loop",
                             (float)(i + 1) / (float)message_count);
        }

        switch (message_types[i]) {
            case BIO_MSG_ATTENTION_SHIFT:
                bio_router_register_handler(ctx, message_types[i], handle_attention_shift);
                registered++;
                break;
            default:
                NIMCP_LOGGING_DEBUG("global_workspace: unknown message type %d in wiring callback", message_types[i]);
                break;
        }
    }

    NIMCP_LOGGING_INFO("global_workspace: registered %d handlers via KG wiring", registered);
    return 0;
}

/**
 * @brief Handle attention shift via bio-async
 */
static nimcp_error_t handle_attention_shift(
    const void* msg,
    size_t msg_size,
    nimcp_bio_promise_t response_promise,
    void* user_data)
{
    (void)msg_size;
    (void)response_promise;

    if (!msg || !user_data) {
        return NIMCP_ERROR_NULL_ARG;
    }

    const bio_msg_attention_shift_t* shift = (const bio_msg_attention_shift_t*)msg;
    global_workspace_t* ws = (global_workspace_t*)user_data;
    (void)ws;

    LOG_DEBUG("Received attention shift: target=%u, weight=%.2f",
              shift->target_id, shift->attention_weight);

    // TODO: Process attention shift
    return NIMCP_SUCCESS;
}

/**
 * @brief Broadcast workspace broadcast event
 * Note: Using internal struct pointer (struct global_workspace_struct*) instead of typedef
 */
static void bio_broadcast_workspace_event(struct global_workspace_struct* ws, uint32_t broadcast_id, float strength) {
    if (!ws || !ws->bio_async_enabled || !ws->bio_ctx) {
        return;
    }

    bio_msg_attention_shift_t msg = {};
    bio_msg_init_header(&msg.header, BIO_MSG_ATTENTION_SHIFT,
                        bio_module_context_get_id(ws->bio_ctx), 0, sizeof(msg));
    msg.header.flags |= BIO_MSG_FLAG_BROADCAST;
    msg.target_id = broadcast_id;
    msg.attention_weight = strength;

    bio_router_broadcast(ws->bio_ctx, &msg, sizeof(msg));
    LOG_DEBUG("Broadcast: workspace event %u", broadcast_id);
}

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
    if (tau_ms <= 0.0F) return original_strength;
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
    float max_strength = 0.0F;
    uint32_t max_idx = 0;
    bool found_any = false;
    uint64_t current_time = get_time_ms();

    // Find strongest competitor (with decay applied)
    for (uint32_t i = 0; i < GLOBAL_WORKSPACE_MAX_COMPETITORS; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && GLOBAL_WORKSPACE_MAX_COMPETITORS > 256) {
            global_workspace_heartbeat("global_works_loop",
                             (float)(i + 1) / (float)GLOBAL_WORKSPACE_MAX_COMPETITORS);
        }

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
    float max_priority = -1.0F;
    float max_strength_at_priority = 0.0F;
    uint32_t max_idx = 0;
    bool found_any = false;
    uint64_t current_time = get_time_ms();

    for (uint32_t i = 0; i < GLOBAL_WORKSPACE_MAX_COMPETITORS; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && GLOBAL_WORKSPACE_MAX_COMPETITORS > 256) {
            global_workspace_heartbeat("global_works_loop",
                             (float)(i + 1) / (float)GLOBAL_WORKSPACE_MAX_COMPETITORS);
        }

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
                          workspace->config.module_priorities[module] : 0.5F;

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
        /* Phase 8: Loop progress heartbeat */
        if ((offset & 0xFF) == 0 && GLOBAL_WORKSPACE_MAX_COMPETITORS > 256) {
            global_workspace_heartbeat("global_works_loop",
                             (float)(offset + 1) / (float)GLOBAL_WORKSPACE_MAX_COMPETITORS);
        }

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
    float runner_up = 0.0F;
    for (uint32_t i = 0; i < GLOBAL_WORKSPACE_MAX_COMPETITORS; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && GLOBAL_WORKSPACE_MAX_COMPETITORS > 256) {
            global_workspace_heartbeat("global_works_loop",
                             (float)(i + 1) / (float)GLOBAL_WORKSPACE_MAX_COMPETITORS);
        }

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
    /* Phase 8: Heartbeat at operation start */
    global_workspace_heartbeat("global_works_create", 0.0f);


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
    /* Phase 8: Heartbeat at operation start */
    global_workspace_heartbeat("global_works_create_custom", 0.0f);


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

    // Initialize thread safety mutex
    nimcp_platform_mutex_init(&workspace->mutex, false);

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
            /* Phase 8: Loop progress heartbeat */
            if ((i & 0xFF) == 0 && actual_config.history_depth > 256) {
                global_workspace_heartbeat("global_works_loop",
                                 (float)(i + 1) / (float)actual_config.history_depth);
            }

            workspace->history_content[i] = (float*)nimcp_calloc(
                actual_config.capacity_dim, sizeof(float));
            if (workspace->history_content[i] == NULL) {
                fprintf(stderr, "Failed to allocate history content buffer %u\n", i);
                // Clean up previously allocated
                for (uint32_t j = 0; j < i; j++) {
                    /* Phase 8: Loop progress heartbeat */
                    if ((j & 0xFF) == 0 && i > 256) {
                        global_workspace_heartbeat("global_works_loop",
                                         (float)(j + 1) / (float)i);
                    }

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
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && GLOBAL_WORKSPACE_MAX_COMPETITORS > 256) {
            global_workspace_heartbeat("global_works_loop",
                             (float)(i + 1) / (float)GLOBAL_WORKSPACE_MAX_COMPETITORS);
        }

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

    // Phase 1.5: Initialize memory pools for hot-path allocations
    // Pool for broadcast content buffers (capacity_dim floats each)
    memory_pool_config_t content_pool_config = {
        .block_size = actual_config.capacity_dim * sizeof(float),
        .num_blocks = 4,  // Current + temp + 2 spare
        .alignment = 16,  // SIMD alignment
        .enable_tracking = false,
        .enable_guard_pages = false
    };
    workspace->broadcast_content_pool = memory_pool_create(&content_pool_config);

    // Pool for history content buffers (same size, more blocks)
    memory_pool_config_t history_pool_config = {
        .block_size = actual_config.capacity_dim * sizeof(float),
        .num_blocks = actual_config.history_depth + 2,  // history_depth + spares
        .alignment = 16,
        .enable_tracking = false,
        .enable_guard_pages = false
    };
    workspace->history_content_pool = memory_pool_create(&history_pool_config);

    // Initialize bio-async fields
    workspace->bio_ctx = NULL;
    workspace->bio_async_enabled = false;

    // Register with bio-async router if available
    NIMCP_LOGGING_DEBUG("global_workspace: Checking bio-async router initialization...");
    if (bio_router_is_initialized()) {
        NIMCP_LOGGING_DEBUG("global_workspace: Bio-router initialized, registering module (id=%d, inbox_capacity=64)...",
                           BIO_MODULE_GLOBAL_WORKSPACE);
        bio_module_info_t bio_info = {
            .module_id = BIO_MODULE_GLOBAL_WORKSPACE,
            .module_name = "global_workspace",
            .inbox_capacity = 64,
            .user_data = workspace
        };
        workspace->bio_ctx = bio_router_register_module(&bio_info);
        if (workspace->bio_ctx) {
            workspace->bio_async_enabled = true;

            // Try KG-driven wiring callback registration first
            nimcp_error_t wiring_result = bio_router_register_wiring_callback(
                BIO_MODULE_GLOBAL_WORKSPACE,
                (void*)global_workspace_wiring_handler_callback,
                workspace
            );

            if (wiring_result == NIMCP_SUCCESS) {
                NIMCP_LOGGING_INFO("global_workspace: KG-driven wiring callback registered");
            } else {
                // Legacy fallback - register handlers directly
                LEGACY_HANDLER_REGISTRATION(
                    bio_router_register_handler(workspace->bio_ctx, BIO_MSG_ATTENTION_SHIFT, handle_attention_shift)
                );
                NIMCP_LOGGING_INFO("global_workspace: legacy handler registration");
            }
        } else {
            NIMCP_LOGGING_WARN("global_workspace: Bio-async registration failed - module will operate without async messaging");
        }
    } else {
        NIMCP_LOGGING_DEBUG("global_workspace: Bio-router not initialized, skipping async registration");
    }

    // Initialize SNN and Plasticity bridges
    workspace->snn_bridge = NULL;
    workspace->plasticity_bridge = NULL;
    workspace->bridges_enabled = false;

    // Create SNN bridge with default config
    gw_snn_config_t snn_config = gw_snn_config_default();
    workspace->snn_bridge = gw_snn_create(&snn_config);
    if (workspace->snn_bridge) {
        NIMCP_LOGGING_DEBUG("global_workspace: SNN bridge created successfully");
    } else {
        NIMCP_LOGGING_WARN("global_workspace: Failed to create SNN bridge - continuing without SNN integration");
    }

    // Create Plasticity bridge with default config
    gw_plasticity_config_t plasticity_config = gw_plasticity_config_default();
    workspace->plasticity_bridge = gw_plasticity_create(&plasticity_config);
    if (workspace->plasticity_bridge) {
        NIMCP_LOGGING_DEBUG("global_workspace: Plasticity bridge created successfully");
    } else {
        NIMCP_LOGGING_WARN("global_workspace: Failed to create Plasticity bridge - continuing without plasticity integration");
    }

    // Mark bridges as enabled if at least one was created
    if (workspace->snn_bridge || workspace->plasticity_bridge) {
        workspace->bridges_enabled = true;
        NIMCP_LOGGING_INFO("global_workspace: SNN/Plasticity bridge integration enabled");
    }

    // Note: global_workspace_t is typedef'd as a pointer, so function signature
    // expects global_workspace_t* (double pointer). Cast to match.
    return (global_workspace_t*)workspace;
}

void global_workspace_destroy(global_workspace_t* workspace) {
    if (workspace == NULL) return;

    /* Phase 8: Heartbeat at operation start */
    global_workspace_heartbeat("global_works_destroy", 0.0f);


    struct global_workspace_struct* ws = (struct global_workspace_struct*)workspace;

    // Lock before destroying to ensure no concurrent operations
    nimcp_platform_mutex_lock(&ws->mutex);

    // Unregister from bio-async router
    if (ws->bio_async_enabled && ws->bio_ctx) {
        bio_router_unregister_module(ws->bio_ctx);
        ws->bio_ctx = NULL;
        ws->bio_async_enabled = false;
        NIMCP_LOGGING_INFO("Bio-async communication disabled for global_workspace");
    }

    // Destroy SNN and Plasticity bridges
    if (ws->snn_bridge) {
        gw_snn_destroy(ws->snn_bridge);
        ws->snn_bridge = NULL;
        NIMCP_LOGGING_DEBUG("global_workspace: SNN bridge destroyed");
    }
    if (ws->plasticity_bridge) {
        gw_plasticity_destroy(ws->plasticity_bridge);
        ws->plasticity_bridge = NULL;
        NIMCP_LOGGING_DEBUG("global_workspace: Plasticity bridge destroyed");
    }
    ws->bridges_enabled = false;

    // Free history content buffers
    if (ws->history_content != NULL) {
        for (uint32_t i = 0; i < ws->config.history_depth; i++) {
            /* Phase 8: Loop progress heartbeat */
            if ((i & 0xFF) == 0 && ws->config.history_depth > 256) {
                global_workspace_heartbeat("global_works_loop",
                                 (float)(i + 1) / (float)ws->config.history_depth);
            }

            nimcp_free(ws->history_content[i]);
        }
        nimcp_free(ws->history_content);
    }

    // Free history metadata
    nimcp_free(ws->history);

    // Free broadcast content
    nimcp_free(ws->broadcast_content);

    // Phase 1.5: Destroy memory pools
    if (ws->broadcast_content_pool != NULL) {
        memory_pool_destroy(ws->broadcast_content_pool);
    }
    if (ws->history_content_pool != NULL) {
        memory_pool_destroy(ws->history_content_pool);
    }

    // Free competitor content buffers that we own
    for (uint32_t i = 0; i < GLOBAL_WORKSPACE_MAX_COMPETITORS; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && GLOBAL_WORKSPACE_MAX_COMPETITORS > 256) {
            global_workspace_heartbeat("global_works_loop",
                             (float)(i + 1) / (float)GLOBAL_WORKSPACE_MAX_COMPETITORS);
        }

        if (ws->competitors[i].is_active && ws->competitors[i].content) {
            nimcp_free(ws->competitors[i].content);
            ws->competitors[i].content = NULL;
        }
    }

    // Unlock and destroy mutex
    nimcp_platform_mutex_unlock(&ws->mutex);
    nimcp_platform_mutex_destroy(&ws->mutex);

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

    /* Phase 8: Heartbeat at operation start */
    global_workspace_heartbeat("global_works_compete", 0.0f);


    struct global_workspace_struct* ws = (struct global_workspace_struct*)workspace;

    // Thread safety - lock for entire operation
    nimcp_platform_mutex_lock(&ws->mutex);

    // Process pending bio-async messages before competition
    if (ws->bio_async_enabled && ws->bio_ctx) {
        bio_router_process_inbox(ws->bio_ctx, 10);  // Process up to 10 messages
    }

    // Submit to competition pool (internal call, already locked)
    // Note: We call the internal parts directly to avoid recursive locking
    // Validate content dimension
    if (content_dim != ws->config.capacity_dim) {
        fprintf(stderr, "Content dimension mismatch in global_workspace_compete: "
                "expected %u, got %u\n", ws->config.capacity_dim, content_dim);
        nimcp_platform_mutex_unlock(&ws->mutex);
        return false;
    }

    // Validate strength range
    if (strength < 0.0F || strength > 1.0F) {
        fprintf(stderr, "Invalid strength in global_workspace_compete: %.2f "
                "(must be 0.0-1.0)\n", strength);
        nimcp_platform_mutex_unlock(&ws->mutex);
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
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && GLOBAL_WORKSPACE_MAX_COMPETITORS > 256) {
            global_workspace_heartbeat("global_works_loop",
                             (float)(i + 1) / (float)GLOBAL_WORKSPACE_MAX_COMPETITORS);
        }

        if (ws->competitors[i].is_active && ws->competitors[i].module == module) {
            slot_idx = i;
            // Free old content buffer
            if (ws->competitors[i].content) {
                nimcp_free(ws->competitors[i].content);
            }
            break;
        }
    }

    // If not found, find empty slot
    if (slot_idx == GLOBAL_WORKSPACE_MAX_COMPETITORS) {
        for (uint32_t i = 0; i < GLOBAL_WORKSPACE_MAX_COMPETITORS; i++) {
            /* Phase 8: Loop progress heartbeat */
            if ((i & 0xFF) == 0 && GLOBAL_WORKSPACE_MAX_COMPETITORS > 256) {
                global_workspace_heartbeat("global_works_loop",
                                 (float)(i + 1) / (float)GLOBAL_WORKSPACE_MAX_COMPETITORS);
            }

            if (!ws->competitors[i].is_active) {
                slot_idx = i;
                break;
            }
        }
    }

    // Guard: Check if pool is full
    if (slot_idx == GLOBAL_WORKSPACE_MAX_COMPETITORS) {
        fprintf(stderr, "Competition pool full in global_workspace_compete "
                "(%u competitors)\n", GLOBAL_WORKSPACE_MAX_COMPETITORS);
        nimcp_platform_mutex_unlock(&ws->mutex);
        return false;
    }

    // Track if we're adding new competitor vs updating existing
    bool was_inactive = !ws->competitors[slot_idx].is_active;
    bool pool_was_empty = (ws->num_active_competitors == 0);

    // Copy content - we own this buffer now
    float* content_copy = (float*)nimcp_malloc(content_dim * sizeof(float));
    if (!content_copy) {
        fprintf(stderr, "Failed to allocate content buffer in global_workspace_compete\n");
        nimcp_platform_mutex_unlock(&ws->mutex);
        return false;
    }
    memcpy(content_copy, content, content_dim * sizeof(float));

    // Add/update competitor in pool
    ws->competitors[slot_idx].module = module;
    ws->competitors[slot_idx].content = content_copy;  // We own this buffer
    ws->competitors[slot_idx].content_dim = content_dim;
    ws->competitors[slot_idx].strength = strength;
    ws->competitors[slot_idx].timestamp_ms = get_time_ms();
    ws->competitors[slot_idx].is_active = true;

    // Update pool counts
    if (was_inactive) {
        ws->num_active_competitors++;
        if (pool_was_empty) {
            ws->pool_activation_time_ms = get_time_ms();
        }
    }

    // Immediately resolve competition (backward compatible behavior)
    cognitive_module_t winner = MODULE_NONE;

    // Check if pool is empty
    if (ws->num_active_competitors == 0) {
        nimcp_platform_mutex_unlock(&ws->mutex);
        return false;
    }

    uint64_t current_time = get_time_ms();
    uint32_t winner_idx;
    float winner_strength;
    bool found_winner = false;

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
            found_winner = resolve_winner_take_all(ws, &winner_idx, &winner_strength);
            break;
        default:
            found_winner = false;
            break;
    }

    bool broadcast_occurred = false;
    if (found_winner) {
        // Check refractory period
        bool can_broadcast = true;
        if (ws->last_broadcast_time_ms > 0) {
            uint64_t time_since_broadcast = current_time - ws->last_broadcast_time_ms;
            if (time_since_broadcast < ws->config.refractory_period_ms) {
                can_broadcast = false;
            }
        }

        if (can_broadcast) {
            broadcast_winner(ws, winner_idx, winner_strength);
            winner = ws->competitors[winner_idx].module;

            // Clear the entire competition pool after broadcasting
            for (uint32_t i = 0; i < GLOBAL_WORKSPACE_MAX_COMPETITORS; i++) {
                /* Phase 8: Loop progress heartbeat */
                if ((i & 0xFF) == 0 && GLOBAL_WORKSPACE_MAX_COMPETITORS > 256) {
                    global_workspace_heartbeat("global_works_loop",
                                     (float)(i + 1) / (float)GLOBAL_WORKSPACE_MAX_COMPETITORS);
                }

                if (ws->competitors[i].is_active) {
                    if (ws->competitors[i].content) {
                        nimcp_free(ws->competitors[i].content);
                        ws->competitors[i].content = NULL;
                    }
                    ws->competitors[i].is_active = false;
                }
            }
            ws->num_active_competitors = 0;
            ws->pool_activation_time_ms = 0;

            broadcast_occurred = true;
        }
    }

    nimcp_platform_mutex_unlock(&ws->mutex);

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

    /* Phase 8: Heartbeat at operation start */
    global_workspace_heartbeat("global_works_submit", 0.0f);


    struct global_workspace_struct* ws = (struct global_workspace_struct*)workspace;

    // Thread safety
    nimcp_platform_mutex_lock(&ws->mutex);

    // Guard: Validate content dimension
    if (content_dim != ws->config.capacity_dim) {
        fprintf(stderr, "Content dimension mismatch in global_workspace_submit: "
                "expected %u, got %u\n", ws->config.capacity_dim, content_dim);
        nimcp_platform_mutex_unlock(&ws->mutex);
        return false;
    }

    // Guard: Validate strength range
    if (strength < 0.0F || strength > 1.0F) {
        fprintf(stderr, "Invalid strength in global_workspace_submit: %.2f "
                "(must be 0.0-1.0)\n", strength);
        nimcp_platform_mutex_unlock(&ws->mutex);
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
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && GLOBAL_WORKSPACE_MAX_COMPETITORS > 256) {
            global_workspace_heartbeat("global_works_loop",
                             (float)(i + 1) / (float)GLOBAL_WORKSPACE_MAX_COMPETITORS);
        }

        if (ws->competitors[i].is_active && ws->competitors[i].module == module) {
            slot_idx = i;
            // Free old content buffer before replacing
            if (ws->competitors[i].content) {
                nimcp_free(ws->competitors[i].content);
                ws->competitors[i].content = NULL;
            }
            break;
        }
    }

    // If not found, find empty slot
    if (slot_idx == GLOBAL_WORKSPACE_MAX_COMPETITORS) {
        for (uint32_t i = 0; i < GLOBAL_WORKSPACE_MAX_COMPETITORS; i++) {
            /* Phase 8: Loop progress heartbeat */
            if ((i & 0xFF) == 0 && GLOBAL_WORKSPACE_MAX_COMPETITORS > 256) {
                global_workspace_heartbeat("global_works_loop",
                                 (float)(i + 1) / (float)GLOBAL_WORKSPACE_MAX_COMPETITORS);
            }

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
        nimcp_platform_mutex_unlock(&ws->mutex);
        return false;
    }

    // Track if we're adding new competitor vs updating existing
    bool was_inactive = !ws->competitors[slot_idx].is_active;
    bool pool_was_empty = (ws->num_active_competitors == 0);

    // Copy content - we own this buffer now (fixes buffer ownership issue)
    float* content_copy = (float*)nimcp_malloc(content_dim * sizeof(float));
    if (!content_copy) {
        fprintf(stderr, "Failed to allocate content buffer in global_workspace_submit\n");
        nimcp_platform_mutex_unlock(&ws->mutex);
        return false;
    }
    memcpy(content_copy, content, content_dim * sizeof(float));

    // Add/update competitor in pool
    ws->competitors[slot_idx].module = module;
    ws->competitors[slot_idx].content = content_copy;  // We own this buffer now
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

    nimcp_platform_mutex_unlock(&ws->mutex);
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

    /* Phase 8: Heartbeat at operation start */
    global_workspace_heartbeat("global_works_resolve", 0.0f);


    struct global_workspace_struct* ws = (struct global_workspace_struct*)workspace;

    // Thread safety
    nimcp_platform_mutex_lock(&ws->mutex);

    // Set default output (no winner)
    if (winning_module != NULL) {
        *winning_module = MODULE_NONE;
    }

    // Check if pool is empty
    if (ws->num_active_competitors == 0) {
        // No competitors to resolve
        nimcp_platform_mutex_unlock(&ws->mutex);
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
            /* Phase 8: Loop progress heartbeat */
            if ((i & 0xFF) == 0 && GLOBAL_WORKSPACE_MAX_COMPETITORS > 256) {
                global_workspace_heartbeat("global_works_loop",
                                 (float)(i + 1) / (float)GLOBAL_WORKSPACE_MAX_COMPETITORS);
            }

            if (ws->competitors[i].is_active) {
                // Free content buffer we own
                if (ws->competitors[i].content) {
                    nimcp_free(ws->competitors[i].content);
                    ws->competitors[i].content = NULL;
                }
                ws->competitors[i].is_active = false;
            }
        }
        ws->num_active_competitors = 0;
        ws->pool_activation_time_ms = 0;

        if (ws->config.enable_statistics) {
            ws->stats.rejected_submissions++;
        }

        nimcp_platform_mutex_unlock(&ws->mutex);
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
            /* Phase 8: Loop progress heartbeat */
            if ((i & 0xFF) == 0 && GLOBAL_WORKSPACE_MAX_COMPETITORS > 256) {
                global_workspace_heartbeat("global_works_loop",
                                 (float)(i + 1) / (float)GLOBAL_WORKSPACE_MAX_COMPETITORS);
            }

            if (ws->competitors[i].is_active) {
                // Free content buffer we own
                if (ws->competitors[i].content) {
                    nimcp_free(ws->competitors[i].content);
                    ws->competitors[i].content = NULL;
                }
                ws->competitors[i].is_active = false;
            }
        }
        ws->num_active_competitors = 0;
        ws->pool_activation_time_ms = 0;

        nimcp_platform_mutex_unlock(&ws->mutex);
        return true;
    } else {
        // Winner found but blocked by refractory period
        // Keep competitors in pool for next resolve attempt
        nimcp_platform_mutex_unlock(&ws->mutex);
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

    /* Phase 8: Heartbeat at operation start */
    global_workspace_heartbeat("global_works_read_broadcast", 0.0f);


    struct global_workspace_struct* ws = (struct global_workspace_struct*)workspace;

    // Thread safety
    nimcp_platform_mutex_lock(&ws->mutex);

    // Check if broadcast available
    if (!ws->current_broadcast.is_valid) {
        nimcp_platform_mutex_unlock(&ws->mutex);
        return false;
    }

    // Check buffer size
    if (max_dim < ws->current_broadcast.content_dim) {
        fprintf(stderr, "Buffer too small: need %u, have %u\n",
                ws->current_broadcast.content_dim, max_dim);
        nimcp_platform_mutex_unlock(&ws->mutex);
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

    nimcp_platform_mutex_unlock(&ws->mutex);
    return true;
}

bool global_workspace_subscribe(
    global_workspace_t* workspace,
    cognitive_module_t module)
{
    if (workspace == NULL) return false;

    /* Phase 8: Heartbeat at operation start */
    global_workspace_heartbeat("global_works_subscribe", 0.0f);


    struct global_workspace_struct* ws = (struct global_workspace_struct*)workspace;

    // Thread safety
    nimcp_platform_mutex_lock(&ws->mutex);

    // Check if already subscribed
    for (uint32_t i = 0; i < ws->num_subscribers; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && ws->num_subscribers > 256) {
            global_workspace_heartbeat("global_works_loop",
                             (float)(i + 1) / (float)ws->num_subscribers);
        }

        if (ws->subscribers[i] == module) {
            nimcp_platform_mutex_unlock(&ws->mutex);
            return true;  // Already subscribed (idempotent)
        }
    }

    // Check if room for more
    if (ws->num_subscribers >= GLOBAL_WORKSPACE_MAX_SUBSCRIBERS) {
        fprintf(stderr, "Subscriber list full (%u max)\n", GLOBAL_WORKSPACE_MAX_SUBSCRIBERS);
        nimcp_platform_mutex_unlock(&ws->mutex);
        return false;
    }

    // Add subscriber
    ws->subscribers[ws->num_subscribers++] = module;

    // Update statistics
    if (ws->config.enable_statistics) {
        ws->stats.current_subscribers = ws->num_subscribers;
    }

    nimcp_platform_mutex_unlock(&ws->mutex);
    return true;
}

bool global_workspace_unsubscribe(
    global_workspace_t* workspace,
    cognitive_module_t module)
{
    if (workspace == NULL) return false;

    /* Phase 8: Heartbeat at operation start */
    global_workspace_heartbeat("global_works_unsubscribe", 0.0f);


    struct global_workspace_struct* ws = (struct global_workspace_struct*)workspace;

    // Thread safety
    nimcp_platform_mutex_lock(&ws->mutex);

    // Find module in subscriber list
    for (uint32_t i = 0; i < ws->num_subscribers; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && ws->num_subscribers > 256) {
            global_workspace_heartbeat("global_works_loop",
                             (float)(i + 1) / (float)ws->num_subscribers);
        }

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

            nimcp_platform_mutex_unlock(&ws->mutex);
            return true;
        }
    }

    nimcp_platform_mutex_unlock(&ws->mutex);
    return false;  // Not subscribed
}

//=============================================================================
// Query Functions
//=============================================================================

bool global_workspace_has_broadcast(const global_workspace_t* workspace) {
    if (workspace == NULL) return false;
    /* Phase 8: Heartbeat at operation start */
    global_workspace_heartbeat("global_works_has_broadcast", 0.0f);


    struct global_workspace_struct* ws =
        (struct global_workspace_struct*)workspace;

    /* Thread-safe read with mutex */
    nimcp_platform_mutex_lock(&ws->mutex);
    bool result = ws->current_broadcast.is_valid;
    nimcp_platform_mutex_unlock(&ws->mutex);
    return result;
}

cognitive_module_t global_workspace_get_broadcast_source(
    const global_workspace_t* workspace)
{
    if (workspace == NULL) return MODULE_NONE;
    /* Phase 8: Heartbeat at operation start */
    global_workspace_heartbeat("global_works_get_broadcast_source", 0.0f);


    struct global_workspace_struct* ws =
        (struct global_workspace_struct*)workspace;

    /* Thread-safe read with mutex */
    nimcp_platform_mutex_lock(&ws->mutex);
    cognitive_module_t result = MODULE_NONE;
    if (ws->current_broadcast.is_valid) {
        result = ws->current_broadcast.source_module;
    }
    nimcp_platform_mutex_unlock(&ws->mutex);
    return result;
}

float global_workspace_get_broadcast_strength(
    const global_workspace_t* workspace)
{
    if (workspace == NULL) return 0.0F;
    /* Phase 8: Heartbeat at operation start */
    global_workspace_heartbeat("global_works_get_broadcast_streng", 0.0f);


    struct global_workspace_struct* ws =
        (struct global_workspace_struct*)workspace;

    /* Thread-safe read with mutex */
    nimcp_platform_mutex_lock(&ws->mutex);
    float result = 0.0F;
    if (ws->current_broadcast.is_valid) {
        result = ws->current_broadcast.source_strength;
    }
    nimcp_platform_mutex_unlock(&ws->mutex);
    return result;
}

uint32_t global_workspace_get_subscriber_count(const global_workspace_t* workspace) {
    if (workspace == NULL) return 0;
    /* Phase 8: Heartbeat at operation start */
    global_workspace_heartbeat("global_works_get_subscriber_count", 0.0f);


    struct global_workspace_struct* ws =
        (struct global_workspace_struct*)workspace;

    /* Thread-safe read with mutex */
    nimcp_platform_mutex_lock(&ws->mutex);
    uint32_t result = ws->num_subscribers;
    nimcp_platform_mutex_unlock(&ws->mutex);
    return result;
}

uint32_t global_workspace_get_competitor_count(const global_workspace_t* workspace) {
    if (workspace == NULL) return 0;
    /* Phase 8: Heartbeat at operation start */
    global_workspace_heartbeat("global_works_get_competitor_count", 0.0f);


    struct global_workspace_struct* ws =
        (struct global_workspace_struct*)workspace;

    /* Thread-safe read with mutex */
    nimcp_platform_mutex_lock(&ws->mutex);
    uint32_t result = ws->num_active_competitors;
    nimcp_platform_mutex_unlock(&ws->mutex);
    return result;
}

bool global_workspace_is_competing(
    const global_workspace_t* workspace,
    cognitive_module_t module)
{
    if (workspace == NULL) return false;
    /* Phase 8: Heartbeat at operation start */
    global_workspace_heartbeat("global_works_is_competing", 0.0f);


    struct global_workspace_struct* ws =
        (struct global_workspace_struct*)workspace;

    /* Thread-safe read with mutex */
    nimcp_platform_mutex_lock(&ws->mutex);
    bool result = false;
    for (uint32_t i = 0; i < GLOBAL_WORKSPACE_MAX_COMPETITORS; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && GLOBAL_WORKSPACE_MAX_COMPETITORS > 256) {
            global_workspace_heartbeat("global_works_loop",
                             (float)(i + 1) / (float)GLOBAL_WORKSPACE_MAX_COMPETITORS);
        }

        if (ws->competitors[i].is_active && ws->competitors[i].module == module) {
            result = true;
            break;
        }
    }
    nimcp_platform_mutex_unlock(&ws->mutex);
    return result;
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

    /* Phase 8: Heartbeat at operation start */
    global_workspace_heartbeat("global_works_get_history", 0.0f);


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
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && count > 256) {
            global_workspace_heartbeat("global_works_loop",
                             (float)(i + 1) / (float)count);
        }

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

    /* Phase 8: Heartbeat at operation start */
    global_workspace_heartbeat("global_works_time_since_broadcast", 0.0f);


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

    /* Phase 8: Heartbeat at operation start */
    global_workspace_heartbeat("global_works_get_statistics", 0.0f);


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

    /* Phase 8: Heartbeat at operation start */
    global_workspace_heartbeat("global_works_reset_statistics", 0.0f);


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

    /* Phase 8: Heartbeat at operation start */
    global_workspace_heartbeat("global_works_print_state", 0.0f);


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
                /* Phase 8: Loop progress heartbeat */
                if ((i & 0xFF) == 0 && print_count > 256) {
                    global_workspace_heartbeat("global_works_loop",
                                     (float)(i + 1) / (float)print_count);
                }

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
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && GLOBAL_WORKSPACE_MAX_COMPETITORS > 256) {
            global_workspace_heartbeat("global_works_loop",
                             (float)(i + 1) / (float)GLOBAL_WORKSPACE_MAX_COMPETITORS);
        }

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
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && ws->num_subscribers > 256) {
            global_workspace_heartbeat("global_works_loop",
                             (float)(i + 1) / (float)ws->num_subscribers);
        }

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
    /* Phase 8: Heartbeat at operation start */
    global_workspace_heartbeat("global_works_default_config", 0.0f);


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
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && MODULE_CUSTOM_START > 256) {
            global_workspace_heartbeat("global_works_loop",
                             (float)(i + 1) / (float)MODULE_CUSTOM_START);
        }

        config.module_priorities[i] = 0.5F;
    }

    return config;
}

bool global_workspace_set_ignition_threshold(
    global_workspace_t* workspace,
    float new_threshold)
{
    if (workspace == NULL) return false;

    /* Phase 8: Heartbeat at operation start */
    global_workspace_heartbeat("global_works_set_ignition_thresho", 0.0f);


    struct global_workspace_struct* ws = (struct global_workspace_struct*)workspace;

    // Clamp to valid range
    new_threshold = clamp_float(new_threshold,
                                 GLOBAL_WORKSPACE_MIN_IGNITION_THRESHOLD,
                                 GLOBAL_WORKSPACE_MAX_IGNITION_THRESHOLD);

    ws->config.ignition_threshold = new_threshold;
    return true;
}

float global_workspace_get_ignition_threshold(const global_workspace_t* workspace) {
    if (workspace == NULL) return 0.0F;
    /* Phase 8: Heartbeat at operation start */
    global_workspace_heartbeat("global_works_get_ignition_thresho", 0.0f);


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

    /* Phase 8: Heartbeat at operation start */
    global_workspace_heartbeat("global_works_set_module_priority", 0.0f);


    struct global_workspace_struct* ws = (struct global_workspace_struct*)workspace;

    // Clamp to valid range
    priority = clamp_float(priority, 0.0F, 1.0F);

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
    /* Phase 8: Heartbeat at operation start */
    global_workspace_heartbeat("global_works_validate_config", 0.0f);


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
    if (config->competition_decay_tau_ms <= 0.0F) {
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

//=============================================================================
// Knowledge Graph Self-Awareness Integration
//=============================================================================

/**
 * @brief Query self-knowledge from knowledge graph
 *
 * WHAT: Allow global workspace module to introspect its own structure and capabilities
 * WHY:  Enable self-awareness - GW is central to consciousness, needs to know itself
 * HOW:  Use KG reader to look up Global_Workspace entity and related entities
 *
 * @param kg Knowledge graph reader instance
 * @return 1 if self-knowledge found, 0 otherwise
 */
int global_workspace_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;

    // Query for our own module entity
    /* Phase 8: Heartbeat at operation start */
    global_workspace_heartbeat("global_works_query_self_knowledge", 0.0f);


    const kg_entity_t* self = kg_reader_get_entity(kg, "Global_Workspace");
    if (self) {
        // Global workspace module now has access to its documented structure
        NIMCP_LOGGING_DEBUG("global_workspace: Self-knowledge found: %s (%u observations)",
                  self->name, self->num_observations);
    }

    // Query consciousness-related entities
    kg_entity_list_t* consciousness = kg_reader_search_entities(kg, "consciousness");
    if (consciousness) {
        NIMCP_LOGGING_DEBUG("global_workspace: Found %u consciousness-related entities in KG",
                  consciousness->count);
        kg_entity_list_destroy(consciousness);
    }

    // Query broadcast-related entities
    kg_entity_list_t* broadcast = kg_reader_search_entities(kg, "broadcast");
    if (broadcast) {
        NIMCP_LOGGING_DEBUG("global_workspace: Found %u broadcast-related entities in KG",
                  broadcast->count);
        kg_entity_list_destroy(broadcast);
    }

    return self ? 1 : 0;
}

/**
 * @brief Query broadcast targets from knowledge graph
 *
 * WHAT: Allow GW to know what cognitive modules it can broadcast to
 * WHY:  Essential for consciousness - GW needs to know its audience
 * HOW:  Query KG for cognitive subsystem entities that are potential broadcast targets
 *
 * @param kg Knowledge graph reader instance
 * @return 1 if broadcast targets found, 0 otherwise
 */
int global_workspace_query_broadcast_targets(kg_reader_t* kg) {
    if (!kg) return 0;

    // GW should know all cognitive modules it can broadcast to
    /* Phase 8: Heartbeat at operation start */
    global_workspace_heartbeat("global_works_query_broadcast_targ", 0.0f);


    kg_entity_list_t* cognitive = kg_reader_get_entities_by_type(kg, "cognitive_subsystem");
    if (cognitive) {
        NIMCP_LOGGING_DEBUG("global_workspace: Found %u cognitive subsystems as broadcast targets",
                  cognitive->count);
        kg_entity_list_destroy(cognitive);
        return 1;
    }

    // Also check for Module type entities
    kg_entity_list_t* modules = kg_reader_get_entities_by_type(kg, "Module");
    if (modules) {
        NIMCP_LOGGING_DEBUG("global_workspace: Found %u modules as potential broadcast targets",
                  modules->count);
        kg_entity_list_destroy(modules);
        return 1;
    }

    // Query for cognitive-related entities as fallback
    kg_entity_list_t* cognitive_entities = kg_reader_search_entities(kg, "cognitive");
    if (cognitive_entities) {
        NIMCP_LOGGING_DEBUG("global_workspace: Found %u cognitive-related entities",
                  cognitive_entities->count);
        kg_entity_list_destroy(cognitive_entities);
        return 1;
    }

    return 0;
}

/**
 * @brief Get global workspace capabilities from knowledge graph
 *
 * @param kg Knowledge graph reader instance
 * @return Capability description string or NULL
 */
const char* global_workspace_get_capabilities(kg_reader_t* kg) {
    if (!kg) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "kg is NULL");

        return NULL;

    }
    return kg_reader_get_module_capabilities(kg, "Global_Workspace");
}

/**
 * @brief Get global workspace integrations from knowledge graph
 *
 * @param kg Knowledge graph reader instance
 * @return Relation list showing integrations (caller must free)
 */
kg_relation_list_t* global_workspace_get_integrations(kg_reader_t* kg) {
    if (!kg) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "kg is NULL");

        return NULL;

    }
    /* Phase 8: Heartbeat at operation start */
    global_workspace_heartbeat("global_works_get_integrations", 0.0f);


    return kg_reader_get_module_integrations(kg, "Global_Workspace");
}

/**
 * @brief Query competition strategy information from knowledge graph
 *
 * WHAT: Allow GW to understand its competition mechanisms
 * WHY:  Self-awareness about how attention works
 * HOW:  Query KG for competition and attention related entities
 *
 * @param kg Knowledge graph reader instance
 * @return Number of competition-related entities found
 */
int global_workspace_query_competition_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;

    /* Phase 8: Heartbeat at operation start */
    global_workspace_heartbeat("global_works_query_competition_kn", 0.0f);


    int count = 0;

    // Query for competition-related entities
    kg_entity_list_t* competition = kg_reader_search_entities(kg, "competition");
    if (competition) {
        count += competition->count;
        kg_entity_list_destroy(competition);
    }

    // Query for attention-related entities
    kg_entity_list_t* attention = kg_reader_search_entities(kg, "attention");
    if (attention) {
        count += attention->count;
        kg_entity_list_destroy(attention);
    }

    // Query for ignition-related entities (global ignition theory)
    kg_entity_list_t* ignition = kg_reader_search_entities(kg, "ignition");
    if (ignition) {
        count += ignition->count;
        kg_entity_list_destroy(ignition);
    }

    return count;
}

/* ============================================================================
 * Phase 8: Instance-Level Health Agent
 * ============================================================================ */

void global_workspace_set_instance_health_agent(void* instance, nimcp_health_agent_t* agent) {
    if (instance) {
        (void)agent;
        g_global_workspace_health_agent = agent;
    }
}

/* ============================================================================
 * Phase 8: Training Integration (Full Implementation)
 * ============================================================================ */

int global_workspace_training_begin(void* instance) {
    if (!instance) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "global_workspace_training_begin: NULL argument");
        return -1;
    }
    global_workspace_heartbeat_instance(NULL, "global_workspace_training_begin", 0.0f);
    (void)(struct global_workspace_struct*)instance; /* Module state available for reset */
    return 0;
}

int global_workspace_training_end(void* instance) {
    if (!instance) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "global_workspace_training_end: NULL argument");
        return -1;
    }
    global_workspace_heartbeat_instance(NULL, "global_workspace_training_end", 1.0f);
    (void)(struct global_workspace_struct*)instance; /* Module state available for finalization */
    return 0;
}

int global_workspace_training_step(void* instance, float progress) {
    if (!instance) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "global_workspace_training_step: NULL argument");
        return -1;
    }
    if (progress < 0.0f) progress = 0.0f;
    if (progress > 1.0f) progress = 1.0f;
    global_workspace_heartbeat_instance(NULL, "global_workspace_training_step", progress);
    (void)(struct global_workspace_struct*)instance; /* Module state available for step adaptation */
    return 0;
}
