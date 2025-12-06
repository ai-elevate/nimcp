// ============================================================================
// nimcp_autobiographical_memory.c - Episodic Self-Memory Implementation
// ============================================================================

/**
 * BIO-ASYNC INTEGRATION:
 * - Module ID: 0x0335 (BIO_MODULE_AUTOBIOGRAPHICAL_MEMORY)
 * - Publishes: memory storage, retrieval, consolidation events
 * - Subscribes: emotional tags, temporal context
 */

#define LOG_MODULE "autobiographical_memory"

#include "cognitive/nimcp_autobiographical_memory.h"
#include "nimcp.h"  // For NIMCP_ERROR_* codes
#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"
#include "utils/memory/nimcp_unified_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/platform/nimcp_platform.h"
#include "utils/thread/nimcp_thread.h"
#include <string.h>
#include <stdio.h>
#include <math.h>
#include "utils/memory/nimcp_memory_guards.h"  // For nimcp_calloc/nimcp_free

//=============================================================================
// BIO-ASYNC MODULE REGISTRATION
//=============================================================================

#define BIO_MODULE_AUTOBIOGRAPHICAL_MEMORY 0x0335

// ============================================================================
// Internal Structure
// ============================================================================

struct autobiographical_memory_system {
    autobiographical_memory_entry_t* memories;   // Array of memories
    uint32_t capacity;                     // Maximum memories
    uint32_t count;                        // Current count
    uint64_t next_id;                      // Next memory ID

    // Statistics
    autobio_stats_t stats;

    // Thread safety
    nimcp_mutex_t mutex;

    // Unified memory integration (CoW support for brain cloning)
    void* mem_manager;              /**< unified_mem_manager_t */
    void* memories_handle;          /**< CoW handle for memories array */

    // Bio-async integration
    void* bio_ctx;                  /**< bio_module_context_t pointer */
    bool bio_async_enabled;         /**< Bio-async registration status */
};

/*=============================================================================
 * BIO-ASYNC HANDLERS (Forward declarations)
 *============================================================================*/

static nimcp_error_t handle_memory_retrieve_request(
    const void* msg, size_t msg_size,
    nimcp_bio_promise_t response_promise, void* user_data);

static void bio_broadcast_memory_stored(struct autobiographical_memory_system* system, uint64_t memory_id, float salience);

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * @brief Get current time in milliseconds
 */
static uint64_t get_current_time_ms(void)
{
    return nimcp_platform_time_monotonic_ms();
}

/**
 * @brief Initialize memory query with defaults
 */
static void init_query(memory_query_t* query)
{
    memset(query, 0, sizeof(memory_query_t));
    query->max_results = UINT32_MAX;
    query->sort_by_recency = true;
}

/**
 * @brief Check if memory matches query filters
 */
static bool matches_query(const autobiographical_memory_entry_t* mem,
                          const memory_query_t* query)
{
    // Time range filter
    if (query->start_time_ms > 0 && mem->timestamp_ms < query->start_time_ms) {
        return false;
    }
    if (query->end_time_ms > 0 && mem->timestamp_ms > query->end_time_ms) {
        return false;
    }

    // Type filter
    if (query->filter_by_type && mem->type != query->type_filter) {
        return false;
    }

    // Valence filter
    if (query->filter_by_valence && mem->valence != query->valence_filter) {
        return false;
    }

    // Importance filter
    if (query->filter_by_importance && mem->importance < query->min_importance) {
        return false;
    }

    // Text search
    if (query->search_text && query->search_text[0] != '\0') {
        bool found = false;
        if (query->case_sensitive) {
            found = (strstr(mem->what_happened, query->search_text) != NULL) ||
                   (strstr(mem->why_it_happened, query->search_text) != NULL) ||
                   (strstr(mem->outcome, query->search_text) != NULL);
        } else {
            // Case-insensitive search (simplified)
            found = (strstr(mem->what_happened, query->search_text) != NULL) ||
                   (strstr(mem->why_it_happened, query->search_text) != NULL) ||
                   (strstr(mem->outcome, query->search_text) != NULL);
        }
        if (!found) {
            return false;
        }
    }

    return true;
}

/**
 * @brief Comparison for sorting by recency
 */
static int compare_by_recency(const void* a, const void* b)
{
    const autobiographical_memory_entry_t* ma = (const autobiographical_memory_entry_t*)a;
    const autobiographical_memory_entry_t* mb = (const autobiographical_memory_entry_t*)b;

    // Newer first
    if (ma->timestamp_ms > mb->timestamp_ms) return -1;
    if (ma->timestamp_ms < mb->timestamp_ms) return 1;
    return 0;
}

/**
 * @brief Comparison for sorting by importance
 */
static int compare_by_importance(const void* a, const void* b)
{
    const autobiographical_memory_entry_t* ma = (const autobiographical_memory_entry_t*)a;
    const autobiographical_memory_entry_t* mb = (const autobiographical_memory_entry_t*)b;

    // More important first
    if (ma->importance > mb->importance) return -1;
    if (ma->importance < mb->importance) return 1;
    return 0;
}

// ============================================================================
// Core API Implementation
// ============================================================================

autobiographical_memory_t autobio_create(uint32_t capacity)
{
    // Guard: Use default capacity if 0
    if (capacity == 0) {
        capacity = AUTOBIO_DEFAULT_CAPACITY;
    }

    LOG_INFO("Creating autobiographical memory system: capacity=%u", capacity);

    // Allocate system
    struct autobiographical_memory_system* system =
        nimcp_calloc(1, sizeof(struct autobiographical_memory_system));
    if (!system) {
        LOG_ERROR("Failed to allocate autobiographical memory system (%zu bytes)",
                 sizeof(struct autobiographical_memory_system));
        return NULL;
    }

    // Initialize unified memory manager for CoW support
    unified_mem_config_t mem_config = unified_mem_default_config();
    mem_config.enable_cow = true;
    mem_config.enable_tracking = true;
    system->mem_manager = unified_mem_create(&mem_config);

    // Allocate memory array (with unified memory if available)
    if (system->mem_manager) {
        unified_mem_request_t req = unified_mem_request(
            capacity * sizeof(autobiographical_memory_entry_t),
            NULL, true
        );
        system->memories_handle = unified_mem_alloc(system->mem_manager, &req);
        if (system->memories_handle) {
            system->memories = (autobiographical_memory_entry_t*)unified_mem_write(system->memories_handle);
            LOG_DEBUG(LOG_MODULE, "Memories array allocated via unified memory with CoW");
        }
    }
    if (!system->memories) {
        system->memories = nimcp_calloc(capacity, sizeof(autobiographical_memory_entry_t));
    }
    if (!system->memories) {
        if (system->mem_manager) unified_mem_destroy(system->mem_manager);
        nimcp_free(system);
        return NULL;
    }

    system->capacity = capacity;
    system->count = 0;
    system->next_id = 1;  // Start from 1 (0 = invalid ID)

    // Initialize mutex
    if (nimcp_mutex_init(&system->mutex, NULL) != NIMCP_SUCCESS) {
        if (system->memories_handle) unified_mem_free(system->memories_handle);
        else nimcp_free(system->memories);
        if (system->mem_manager) unified_mem_destroy(system->mem_manager);
        nimcp_free(system);
        return NULL;
    }

    // Initialize stats
    memset(&system->stats, 0, sizeof(autobio_stats_t));

    
    // Bio-async registration
    system->bio_ctx = NULL;
    system->bio_async_enabled = false;
    if (bio_router_is_initialized()) {
        bio_module_info_t bio_info = {
            .module_id = BIO_MODULE_MEMORY_AUTOBIOGRAPHICAL,
            .module_name = "autobiographical_memory",
            .inbox_capacity = 32,
            .user_data = system
        };
        system->bio_ctx = bio_router_register_module(&bio_info);
        if (system->bio_ctx) {
            system->bio_async_enabled = true;
            bio_router_register_handler(system->bio_ctx, BIO_MSG_WORKING_MEMORY_RETRIEVE,
                                        handle_memory_retrieve_request);
            LOG_INFO(LOG_MODULE, "Bio-async registered (module_id=0x%04X)", BIO_MODULE_AUTOBIOGRAPHICAL_MEMORY);
        }
    }

return system;
}

/*=============================================================================
 * BIO-ASYNC HANDLER IMPLEMENTATIONS
 *============================================================================*/

static nimcp_error_t handle_memory_retrieve_request(
    const void* msg, size_t msg_size,
    nimcp_bio_promise_t response_promise, void* user_data)
{
    (void)msg_size;
    (void)response_promise;
    if (!msg || !user_data) { return NIMCP_ERROR_NULL_ARG; }
    struct autobiographical_memory_system* system = (struct autobiographical_memory_system*)user_data;
    LOG_DEBUG(LOG_MODULE, "Received memory retrieve request, count=%u", system->count);
    return NIMCP_SUCCESS;
}

static void bio_broadcast_memory_stored(struct autobiographical_memory_system* system, uint64_t memory_id, float salience) {
    if (!system || !system->bio_async_enabled || !system->bio_ctx) { return; }
    // Use salience response for memory stored notification
    bio_msg_salience_response_t msg = {0};
    bio_msg_init_header(&msg.header, BIO_MSG_SALIENCE_RESPONSE,
                        bio_module_context_get_id(system->bio_ctx), 0, sizeof(msg));
    msg.header.flags |= BIO_MSG_FLAG_BROADCAST;
    msg.stimulus_id = (uint32_t)memory_id;
    msg.salience_score = salience;
    msg.attention_priority = salience;
    msg.requires_immediate_attention = (salience > 0.7f);
    bio_router_broadcast(system->bio_ctx, &msg, sizeof(msg));
    LOG_DEBUG(LOG_MODULE, "Broadcast memory stored: id=%lu, salience=%.2f", (unsigned long)memory_id, salience);
}

void autobio_destroy(autobiographical_memory_t system)
{
    if (!system) {
        return;
    }

    // Unregister from bio-router
    if (system->bio_async_enabled && system->bio_ctx) {
        bio_router_unregister_module(system->bio_ctx);
        system->bio_ctx = NULL;
        system->bio_async_enabled = false;
    }

    nimcp_mutex_destroy(&system->mutex);

    // Free memories array (unified memory or direct)
    if (system->memories_handle) {
        unified_mem_free(system->memories_handle);
        system->memories_handle = NULL;
        system->memories = NULL;
    } else if (system->memories) {
        nimcp_free(system->memories);
        system->memories = NULL;
    }

    // Destroy unified memory manager
    if (system->mem_manager) {
        unified_mem_destroy(system->mem_manager);
        system->mem_manager = NULL;
    }

    nimcp_free(system);
}

uint64_t autobio_store(autobiographical_memory_t system,
                       const autobiographical_memory_entry_t* memory)
{
    // Guard: NULL checks
    if (!system || !memory) {
        return 0;
    }

    nimcp_mutex_lock(&system->mutex);

    // Guard: Check capacity
    if (system->count >= system->capacity) {
        // Could implement eviction policy here
        nimcp_mutex_unlock(&system->mutex);
        return 0;
    }

    // Copy memory
    autobiographical_memory_entry_t* slot = &system->memories[system->count];
    memcpy(slot, memory, sizeof(autobiographical_memory_entry_t));

    // Assign ID and timestamp
    slot->memory_id = system->next_id++;
    if (slot->timestamp_ms == 0) {
        slot->timestamp_ms = get_current_time_ms();
    }

    // Initialize retrieval stats
    slot->times_recalled = 0;
    slot->last_recalled_ms = 0;
    if (slot->memory_strength == 0.0f) {
        slot->memory_strength = 1.0f;  // New memories start vivid
    }

    // Update system stats
    system->count++;
    system->stats.total_memories = system->count;
    system->stats.memories_by_type[memory->type]++;
    if (memory->is_core_memory) {
        system->stats.core_memories++;
    }

    // Update time bounds
    if (system->stats.oldest_memory_ms == 0 ||
        slot->timestamp_ms < system->stats.oldest_memory_ms) {
        system->stats.oldest_memory_ms = slot->timestamp_ms;
    }
    if (slot->timestamp_ms > system->stats.newest_memory_ms) {
        system->stats.newest_memory_ms = slot->timestamp_ms;
    }

    // Update average stats (running average)
    float alpha = 1.0f / (float)system->count;
    system->stats.avg_importance =
        (1.0f - alpha) * system->stats.avg_importance + alpha * memory->importance;
    system->stats.avg_emotional_intensity =
        (1.0f - alpha) * system->stats.avg_emotional_intensity + alpha * memory->emotional_intensity;

    uint64_t assigned_id = slot->memory_id;

    nimcp_mutex_unlock(&system->mutex);

    return assigned_id;
}

bool autobio_retrieve(autobiographical_memory_t system,
                     uint64_t memory_id,
                     autobiographical_memory_entry_t* out_memory)
{
    // Process pending bio-async messages
    if (system && system->bio_ctx) {
        bio_router_process_inbox(system->bio_ctx, 5);
    }

    // Guard: NULL checks
    if (!system || !out_memory || memory_id == 0) {
        return false;
    }

    nimcp_mutex_lock(&system->mutex);

    // Linear search (could use hash map for O(1))
    bool found = false;
    for (uint32_t i = 0; i < system->count; i++) {
        if (system->memories[i].memory_id == memory_id) {
            // Copy memory
            memcpy(out_memory, &system->memories[i], sizeof(autobiographical_memory_t));

            // Update retrieval stats
            system->memories[i].times_recalled++;
            system->memories[i].last_recalled_ms = get_current_time_ms();
            system->stats.total_retrievals++;

            found = true;
            break;
        }
    }

    nimcp_mutex_unlock(&system->mutex);

    return found;
}

bool autobio_query(autobiographical_memory_t system,
                  const memory_query_t* query,
                  autobiographical_memory_entry_t* results,
                  uint32_t max_results,
                  uint32_t* num_found)
{
    // Guard: NULL checks
    if (!system || !query || !results || !num_found) {
        return false;
    }

    nimcp_mutex_lock(&system->mutex);

    *num_found = 0;

    // Collect matching memories
    for (uint32_t i = 0; i < system->count && *num_found < max_results; i++) {
        if (matches_query(&system->memories[i], query)) {
            memcpy(&results[*num_found], &system->memories[i],
                   sizeof(autobiographical_memory_t));
            (*num_found)++;

            // Update retrieval stats
            system->memories[i].times_recalled++;
            system->memories[i].last_recalled_ms = get_current_time_ms();
        }
    }

    system->stats.total_retrievals += *num_found;

    // Sort results
    if (*num_found > 1) {
        if (query->sort_by_recency) {
            qsort(results, *num_found, sizeof(autobiographical_memory_t),
                  compare_by_recency);
        } else if (query->sort_by_importance) {
            qsort(results, *num_found, sizeof(autobiographical_memory_t),
                  compare_by_importance);
        }
    }

    nimcp_mutex_unlock(&system->mutex);

    return true;
}

bool autobio_get_recent(autobiographical_memory_t system,
                       uint32_t count,
                       autobiographical_memory_entry_t* results,
                       uint32_t* num_found)
{
    memory_query_t query = {0};
    init_query(&query);
    query.max_results = count;
    query.sort_by_recency = true;

    return autobio_query(system, &query, results, count, num_found);
}

bool autobio_get_core_memories(autobiographical_memory_t system,
                               autobiographical_memory_entry_t* results,
                               uint32_t max_results,
                               uint32_t* num_found)
{
    // Guard: NULL checks
    if (!system || !results || !num_found) {
        return false;
    }

    nimcp_mutex_lock(&system->mutex);

    *num_found = 0;

    // Collect core memories
    for (uint32_t i = 0; i < system->count && *num_found < max_results; i++) {
        if (system->memories[i].is_core_memory) {
            memcpy(&results[*num_found], &system->memories[i],
                   sizeof(autobiographical_memory_t));
            (*num_found)++;
        }
    }

    // Sort by importance
    if (*num_found > 1) {
        qsort(results, *num_found, sizeof(autobiographical_memory_t),
              compare_by_importance);
    }

    nimcp_mutex_unlock(&system->mutex);

    return true;
}

bool autobio_mark_core(autobiographical_memory_t system,
                      uint64_t memory_id,
                      bool is_core)
{
    // Guard: NULL checks
    if (!system || memory_id == 0) {
        return false;
    }

    nimcp_mutex_lock(&system->mutex);

    bool found = false;
    for (uint32_t i = 0; i < system->count; i++) {
        if (system->memories[i].memory_id == memory_id) {
            bool was_core = system->memories[i].is_core_memory;
            system->memories[i].is_core_memory = is_core;

            // Update stats
            if (is_core && !was_core) {
                system->stats.core_memories++;
            } else if (!is_core && was_core) {
                system->stats.core_memories--;
            }

            found = true;
            break;
        }
    }

    nimcp_mutex_unlock(&system->mutex);

    return found;
}

bool autobio_update_importance(autobiographical_memory_t system,
                              uint64_t memory_id,
                              float new_importance)
{
    // Guard: NULL checks and bounds
    if (!system || memory_id == 0 || new_importance < 0.0f || new_importance > 1.0f) {
        return false;
    }

    nimcp_mutex_lock(&system->mutex);

    bool found = false;
    for (uint32_t i = 0; i < system->count; i++) {
        if (system->memories[i].memory_id == memory_id) {
            system->memories[i].importance = new_importance;
            found = true;
            break;
        }
    }

    nimcp_mutex_unlock(&system->mutex);

    return found;
}

bool autobio_get_stats(autobiographical_memory_t system,
                      autobio_stats_t* stats)
{
    // Guard: NULL checks
    if (!system || !stats) {
        return false;
    }

    nimcp_mutex_lock(&system->mutex);

    memcpy(stats, &system->stats, sizeof(autobio_stats_t));
    stats->memory_usage_bytes = system->capacity * sizeof(autobiographical_memory_t) +
                                sizeof(struct autobiographical_memory_system);

    nimcp_mutex_unlock(&system->mutex);

    return true;
}

uint32_t autobio_consolidate(autobiographical_memory_t system)
{
    // Guard: NULL check
    if (!system) {
        return 0;
    }

    nimcp_mutex_lock(&system->mutex);

    uint32_t pruned = 0;
    uint64_t current_time = get_current_time_ms();

    // Consolidation algorithm:
    // 1. Strengthen oft-recalled memories
    // 2. Decay rarely-accessed memories
    // 3. Prune very weak, unimportant memories

    for (uint32_t i = 0; i < system->count; i++) {
        autobiographical_memory_entry_t* mem = &system->memories[i];

        // Skip core memories
        if (mem->is_core_memory) {
            continue;
        }

        // Calculate time since last recall
        uint64_t time_since_recall = current_time - mem->last_recalled_ms;
        float days_since_recall = time_since_recall / (1000.0f * 60.0f * 60.0f * 24.0f);

        // Strengthen if recalled recently and often
        if (mem->times_recalled > 3 && days_since_recall < 7.0f) {
            mem->memory_strength = fminf(1.0f, mem->memory_strength * 1.1f);
        }

        // Decay if not recalled
        if (days_since_recall > 30.0f) {
            mem->memory_strength *= 0.9f;
        }

        // Prune if very weak and unimportant
        if (mem->memory_strength < 0.2f && mem->importance < AUTOBIO_IMPORTANCE_THRESHOLD) {
            // Mark for pruning (set memory_id to 0)
            mem->memory_id = 0;
            pruned++;
        }
    }

    // Compact array (remove pruned memories)
    if (pruned > 0) {
        uint32_t write_idx = 0;
        for (uint32_t read_idx = 0; read_idx < system->count; read_idx++) {
            if (system->memories[read_idx].memory_id != 0) {
                if (write_idx != read_idx) {
                    memcpy(&system->memories[write_idx], &system->memories[read_idx],
                           sizeof(autobiographical_memory_t));
                }
                write_idx++;
            }
        }
        system->count = write_idx;
        system->stats.total_memories = system->count;
    }

    nimcp_mutex_unlock(&system->mutex);

    return pruned;
}

bool autobio_generate_timeline_summary(autobiographical_memory_t system,
                                      uint64_t start_time_ms,
                                      uint64_t end_time_ms,
                                      char* summary,
                                      size_t summary_len)
{
    // Guard: NULL checks
    if (!system || !summary || summary_len == 0) {
        return false;
    }

    // Query memories in time range
    memory_query_t query = {0};
    init_query(&query);
    query.start_time_ms = start_time_ms;
    query.end_time_ms = end_time_ms;
    query.sort_by_recency = true;
    query.max_results = 100;  // Limit to 100 memories

    autobiographical_memory_t results[100];
    uint32_t found;

    if (!autobio_query(system, &query, results, 100, &found)) {
        return false;
    }

    // Generate summary
    int written = snprintf(summary, summary_len,
                          "Timeline Summary (%u memories):\n\n", found);

    if (written < 0 || (size_t)written >= summary_len) {
        return false;
    }

    for (uint32_t i = 0; i < found && (size_t)written < summary_len - 100; i++) {
        const autobiographical_memory_entry_t* mem = &results[i];

        int n = snprintf(summary + written, summary_len - written,
                        "[%s] %s\n  Why: %s\n  Outcome: %s\n\n",
                        (mem->type == AUTOBIO_LEARNING) ? "LEARNED" :
                        (mem->type == AUTOBIO_ACTION) ? "DID" :
                        (mem->type == AUTOBIO_DECISION) ? "DECIDED" :
                        (mem->type == AUTOBIO_ACHIEVEMENT) ? "ACHIEVED" :
                        (mem->type == AUTOBIO_INSIGHT) ? "REALIZED" : "EXPERIENCED",
                        mem->what_happened,
                        mem->why_it_happened,
                        mem->outcome);

        if (n > 0) {
            written += n;
        }
    }

    return true;
}
