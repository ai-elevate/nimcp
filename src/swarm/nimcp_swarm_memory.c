/**
 * @file NimcpSwarmMemory.c
 * @brief Implementation of Swarm Memory Consolidation System
 *
 * Biological Inspiration: Sleep-based memory consolidation in mammals
 *
 * This implementation provides distributed memory consolidation across
 * the swarm, mimicking how mammalian brains consolidate memories during
 * sleep through hippocampal replay and neocortical integration.
 *
 * @version 1.0
 * @date 2025
 */

#include "swarm/nimcp_swarm_memory.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/validation/nimcp_validate.h"
#include "utils/platform/nimcp_platform.h"
#include "utils/containers/nimcp_hash_table.h"
#include "utils/containers/nimcp_min_heap.h"

#define LOG_MODULE "swarm_memory"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ============================================================================
 * Validation Helper Macros (local implementations)
 * ============================================================================ */

#define NIMCP_VALIDATE_NOT_NULL(ptr) \
    do { if (!(ptr)) { return NIMCP_INVALID_PARAM; } } while(0)

#define NIMCP_VALIDATE(cond) \
    do { if (!(cond)) { return NIMCP_INVALID_PARAM; } } while(0)

/* ============================================================================
 * Hash Table Wrapper Functions (adapts string-keyed API)
 * ============================================================================ */

static inline nimcp_result_t nimcp_hash_table_insert(hash_table_t *table, const char *key, void *value) {
    if (!table || !key) return NIMCP_INVALID_PARAM;
    return hash_table_insert_string(table, key, &value, sizeof(void*)) ? NIMCP_SUCCESS : NIMCP_ERROR;
}

static inline void *nimcp_hash_table_get(hash_table_t *table, const char *key) {
    if (!table || !key) return NULL;
    void **result = (void**)hash_table_lookup_string(table, key);
    return result ? *result : NULL;
}

static inline nimcp_result_t nimcp_hash_table_remove(hash_table_t *table, const char *key) {
    if (!table || !key) return NIMCP_INVALID_PARAM;
    return hash_table_remove_string(table, key) ? NIMCP_SUCCESS : NIMCP_NOT_FOUND;
}

/* ============================================================================
 * Replay Queue Management (uses min_heap with priority wrapper)
 * Note: min_heap uses uint32_t vertex_id + float priority, we adapt for replay
 * ============================================================================ */

/* Replay entry index counter for heap vertex_id */
static uint32_t _replay_entry_counter = 0;

/* Simple array to map heap vertex_id back to replay entry pointers */
#define MAX_REPLAY_ENTRIES 4096
static void* _replay_entry_map[MAX_REPLAY_ENTRIES];

static inline nimcp_result_t nimcp_replay_heap_insert(nimcp_min_heap_t *heap, void *entry) {
    if (!heap || !entry) return NIMCP_INVALID_PARAM;

    uint32_t idx = _replay_entry_counter++ % MAX_REPLAY_ENTRIES;
    _replay_entry_map[idx] = entry;

    /* Use negative priority so higher priority = lower value (min-heap) */
    nimcp_heap_element_t elem = { .vertex_id = idx, .priority = -(float)idx };
    return nimcp_min_heap_insert(heap, &elem) ? NIMCP_SUCCESS : NIMCP_ERROR;
}

static inline void *nimcp_replay_heap_extract(nimcp_min_heap_t *heap) {
    if (!heap) return NULL;

    nimcp_heap_element_t elem;
    if (!nimcp_min_heap_extract_min(heap, &elem)) return NULL;

    void* entry = _replay_entry_map[elem.vertex_id % MAX_REPLAY_ENTRIES];
    _replay_entry_map[elem.vertex_id % MAX_REPLAY_ENTRIES] = NULL;
    return entry;
}

/* ============================================================================
 * Constants and Configuration
 * ============================================================================ */

#define NIMCP_MEMORY_ID_MAX_LENGTH 64
#define NIMCP_NODE_ID_MAX_LENGTH 64
#define NIMCP_DEFAULT_REPLICATION_FACTOR 3
#define NIMCP_DEFAULT_CONSENSUS_THRESHOLD 0.67f
#define NIMCP_DEFAULT_NOVELTY_THRESHOLD 0.6f
#define NIMCP_DEFAULT_REPLAY_PROBABILITY 0.1f
#define NIMCP_DEFAULT_COMPRESSION_TARGET 0.5f
#define NIMCP_MIN_MEMORY_STRENGTH 0.01f
#define NIMCP_MAX_PATTERN_HASH 0xFFFFFFFF

/* Default forgetting curve parameters */
#define NIMCP_DEFAULT_DECAY_RATE 0.0001f
#define NIMCP_DEFAULT_REHEARSAL_BOOST 0.1f
#define NIMCP_DEFAULT_HALF_LIFE_MS 86400000  /* 1 day */

/* Consolidation window defaults */
#define NIMCP_DEFAULT_WINDOW_DURATION_MS 300000  /* 5 minutes */
#define NIMCP_DEFAULT_MAX_MEMORIES_PER_WINDOW 100
#define NIMCP_DEFAULT_ACTIVITY_THRESHOLD 0.3f

/* Bio-async message types */
#define NIMCP_MSG_MEMORY_SHARE "MEMORY_SHARE"
#define NIMCP_MSG_MEMORY_REQUEST "MEMORY_REQUEST"
#define NIMCP_MSG_MEMORY_RESPONSE "MEMORY_RESPONSE"
#define NIMCP_MSG_CONSOLIDATION_SIGNAL "CONSOLIDATION_SIGNAL"
#define NIMCP_MSG_CONSENSUS_REQUEST "CONSENSUS_REQUEST"
#define NIMCP_MSG_SYNC_REQUEST "SYNC_REQUEST"

/* ============================================================================
 * Helper Function Declarations
 * ============================================================================ */

static NimcpMemoryEntry *create_memory_entry(
    NimcpMemoryType type,
    NimcpMemoryImportance importance,
    const void *data,
    size_t data_size,
    const char *source_node
);

static void destroy_memory_entry(NimcpMemoryEntry *entry);

static void generate_memory_id(char *buffer, size_t buffer_size);

static float calculate_decay_modifier(
    NimcpMemoryImportance importance,
    uint32_t rehearsal_count
);

static nimcp_result_t apply_compression(
    const void *data,
    size_t data_size,
    void **compressed,
    size_t *compressed_size
);

static nimcp_result_t apply_decompression(
    const void *compressed,
    size_t compressed_size,
    void *decompressed,
    size_t buffer_size
);

static int compare_replay_priority(const void *a, const void *b);

static nimcp_result_t send_bio_message(
    NimcpSwarmMemory *memory,
    const char *msg_type,
    const char *target_node,
    const void *payload,
    size_t payload_size
);

static void initialize_default_forgetting_curves(NimcpSwarmMemory *memory);

static nimcp_result_t select_replication_nodes(
    NimcpSwarmMemory *memory,
    uint32_t count,
    char **node_ids
);

/* ============================================================================
 * Core API Implementation
 * ============================================================================ */

NimcpSwarmMemory *nimcp_swarm_memory_create(
    uint32_t max_capacity,
    uint32_t replication_factor)
{
    LOG_INFO("Creating swarm memory system with capacity=%u, replication=%u",
                   max_capacity, replication_factor);

    NimcpSwarmMemory *memory = (NimcpSwarmMemory *)nimcp_malloc(sizeof(NimcpSwarmMemory));
    if (!memory) {
        LOG_ERROR("Failed to allocate swarm memory system");
        return NULL;
    }

    memset(memory, 0, sizeof(NimcpSwarmMemory));

    /* Create hash tables for memory storage */
    hash_table_config_t config = {
        .initial_buckets = 256,
        .key_type = HASH_KEY_STRING,
        .hash_algorithm = HASH_ALG_FNV1A,
        .value_destructor = NULL,
        .case_insensitive = false,
        .thread_safe = false
    };

    memory->memories = hash_table_create(&config);
    if (!memory->memories) {
        LOG_ERROR("Failed to create memories hash table");
        nimcp_free(memory);
        return NULL;
    }

    for (int i = 0; i < NIMCP_MEMORY_TYPE_COUNT; i++) {
        memory->memories_by_type[i] = hash_table_create(&config);
        if (!memory->memories_by_type[i]) {
            LOG_ERROR("Failed to create memories_by_type[%d] hash table", i);
            hash_table_destroy(memory->memories);
            for (int j = 0; j < i; j++) {
                hash_table_destroy(memory->memories_by_type[j]);
            }
            nimcp_free(memory);
            return NULL;
        }
    }

    /* Create replay priority queue */
    memory->replay_queue = nimcp_min_heap_create(max_capacity > 0 ? max_capacity : 1024);
    if (!memory->replay_queue) {
        LOG_ERROR("Failed to create replay queue");
        hash_table_destroy(memory->memories);
        for (int i = 0; i < NIMCP_MEMORY_TYPE_COUNT; i++) {
            hash_table_destroy(memory->memories_by_type[i]);
        }
        nimcp_free(memory);
        return NULL;
    }

    /* Create hippocampus nodes hash table */
    memory->hippocampus_nodes = hash_table_create(&config);
    if (!memory->hippocampus_nodes) {
        LOG_ERROR("Failed to create hippocampus_nodes hash table");
        nimcp_min_heap_destroy(memory->replay_queue);
        hash_table_destroy(memory->memories);
        for (int i = 0; i < NIMCP_MEMORY_TYPE_COUNT; i++) {
            hash_table_destroy(memory->memories_by_type[i]);
        }
        nimcp_free(memory);
        return NULL;
    }

    /* Initialize configuration */
    memory->max_memory_capacity = max_capacity;
    memory->replication_factor = (replication_factor > 0) ? replication_factor : NIMCP_DEFAULT_REPLICATION_FACTOR;
    memory->consensus_threshold = NIMCP_DEFAULT_CONSENSUS_THRESHOLD;
    memory->novelty_threshold = NIMCP_DEFAULT_NOVELTY_THRESHOLD;
    memory->replay_probability = NIMCP_DEFAULT_REPLAY_PROBABILITY;
    memory->auto_compression = true;
    memory->auto_distribution = true;

    /* Initialize semantic compression */
    memory->compression.pattern_count = 0;
    memory->compression.pattern_tree = NULL;
    memory->compression.pattern_index = NULL;
    memory->compression.compression_target = NIMCP_DEFAULT_COMPRESSION_TARGET;
    memory->compression.abstraction_level = 1;

    /* Initialize consolidation window */
    memory->window.mode = NIMCP_CONSOLIDATION_PASSIVE;
    memory->window.window_duration_ms = NIMCP_DEFAULT_WINDOW_DURATION_MS;
    memory->window.max_memories_per_window = NIMCP_DEFAULT_MAX_MEMORIES_PER_WINDOW;
    memory->window.activity_threshold = NIMCP_DEFAULT_ACTIVITY_THRESHOLD;
    memory->window.auto_schedule = true;

    /* Initialize default forgetting curves */
    initialize_default_forgetting_curves(memory);

    /* Initialize mutex */
    memory->mutex = nimcp_platform_mutex_create();
    if (!memory->mutex) {
        LOG_ERROR("Failed to initialize mutex");
        nimcp_free(memory);
        return NULL;
    }

    memory->is_initialized = false;
    memory->bio_async_enabled = false;

    LOG_INFO("Swarm memory system created successfully");
    return memory;
}

void nimcp_swarm_memory_destroy(NimcpSwarmMemory *memory)
{
    if (!memory) {
        return;
    }

    LOG_INFO("Destroying swarm memory system");

    if (memory->mutex) {
        nimcp_platform_mutex_lock(memory->mutex);
    }

    /* Destroy all containers */
    if (memory->memories) {
        hash_table_destroy(memory->memories);
        memory->memories = NULL;
    }

    for (int i = 0; i < NIMCP_MEMORY_TYPE_COUNT; i++) {
        if (memory->memories_by_type[i]) {
            hash_table_destroy(memory->memories_by_type[i]);
            memory->memories_by_type[i] = NULL;
        }
    }

    if (memory->replay_queue) {
        nimcp_min_heap_destroy(memory->replay_queue);
        memory->replay_queue = NULL;
    }

    if (memory->hippocampus_nodes) {
        hash_table_destroy(memory->hippocampus_nodes);
        memory->hippocampus_nodes = NULL;
    }
    memory->compression.pattern_index = NULL;
    if (memory->compression.pattern_tree) {
        nimcp_free(memory->compression.pattern_tree);
        memory->compression.pattern_tree = NULL;
    }

    if (memory->mutex) {
        nimcp_platform_mutex_unlock(memory->mutex);
        nimcp_platform_mutex_destroy(memory->mutex);
    }

    nimcp_free(memory);

    LOG_INFO("Swarm memory system destroyed");
}

nimcp_result_t nimcp_swarm_memory_init(
    NimcpSwarmMemory *memory,
    void *bio_ctx)
{
    NIMCP_VALIDATE_NOT_NULL(memory);

    LOG_INFO("Initializing swarm memory system");

    nimcp_platform_mutex_lock(memory->mutex);

    if (memory->is_initialized) {
        LOG_WARN("Swarm memory system already initialized");
        nimcp_platform_mutex_unlock(memory->mutex);
        return NIMCP_SUCCESS;
    }

    /* Set bio-async context */
    memory->bio_ctx = bio_ctx;
    memory->bio_async_enabled = (bio_ctx != NULL);

    /* Register with bio-async if available */
    if (memory->bio_async_enabled) {
        LOG_INFO("Bio-async integration enabled");
        /* Register message handlers here */
    }

    /* Initialize statistics */
    memset(&memory->stats, 0, sizeof(NimcpMemoryStatistics));

    /* Set initialization timestamp */
    memory->last_consolidation = nimcp_time_get_us();
    memory->window.window_start = memory->last_consolidation;

    memory->is_initialized = true;

    nimcp_platform_mutex_unlock(memory->mutex);

    LOG_INFO("Swarm memory system initialized successfully");
    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Memory Management Implementation
 * ============================================================================ */

nimcp_result_t nimcp_swarm_memory_store(
    NimcpSwarmMemory *memory,
    NimcpMemoryType type,
    NimcpMemoryImportance importance,
    const void *data,
    size_t data_size,
    char *memory_id)
{
    NIMCP_VALIDATE_NOT_NULL(memory);
    NIMCP_VALIDATE_NOT_NULL(data);
    NIMCP_VALIDATE(data_size > 0);

    if (!memory->is_initialized) {
        LOG_ERROR("Swarm memory system not initialized");
        return NIMCP_ERROR;
    }

    if (type >= NIMCP_MEMORY_TYPE_COUNT) {
        LOG_ERROR("Invalid memory type: %d", type);
        return NIMCP_INVALID_PARAM;
    }

    nimcp_platform_mutex_lock(memory->mutex);

    /* Check capacity */
    if (memory->stats.total_memories >= memory->max_memory_capacity) {
        LOG_WARN("Memory capacity reached, triggering forgetting");
        uint32_t forgotten = 0;
        nimcp_swarm_memory_apply_forgetting(memory, &forgotten);
    }

    /* Generate memory ID */
    char id[NIMCP_MEMORY_ID_MAX_LENGTH];
    generate_memory_id(id, sizeof(id));

    /* Create memory entry */
    NimcpMemoryEntry *entry = create_memory_entry(type, importance, data, data_size, "local");
    if (!entry) {
        LOG_ERROR("Failed to create memory entry");
        nimcp_platform_mutex_unlock(memory->mutex);
        return NIMCP_NO_MEMORY;
    }

    strncpy(entry->id, id, sizeof(entry->id) - 1);

    /* Calculate novelty score (simplified) */
    entry->novelty_score = (float)rand() / (float)RAND_MAX;  /* TODO: Real novelty detection */

    /* Store in hash tables */
    if (nimcp_hash_table_insert(memory->memories, id, entry) != NIMCP_SUCCESS) {
        LOG_ERROR("Failed to insert memory into main table");
        destroy_memory_entry(entry);
        nimcp_platform_mutex_unlock(memory->mutex);
        return NIMCP_ERROR;
    }

    if (nimcp_hash_table_insert(memory->memories_by_type[type], id, entry) != NIMCP_SUCCESS) {
        LOG_ERROR("Failed to insert memory into type table");
        nimcp_hash_table_remove(memory->memories, id);
        destroy_memory_entry(entry);
        nimcp_platform_mutex_unlock(memory->mutex);
        return NIMCP_ERROR;
    }

    /* Update statistics */
    memory->stats.total_memories++;

    /* Schedule for replay if novel or important */
    if (entry->novelty_score > memory->novelty_threshold ||
        importance >= NIMCP_IMPORTANCE_HIGH) {
        float priority = nimcp_swarm_memory_calculate_replay_priority(memory, entry);
        nimcp_swarm_memory_schedule_replay(memory, id, priority);
    }

    /* Auto-distribute if enabled */
    if (memory->auto_distribution) {
        uint32_t replicas = 0;
        nimcp_swarm_memory_distribute(memory, id, &replicas);
    }

    /* Return memory ID */
    if (memory_id) {
        strncpy(memory_id, id, NIMCP_MEMORY_ID_MAX_LENGTH - 1);
        memory_id[NIMCP_MEMORY_ID_MAX_LENGTH - 1] = '\0';
    }

    nimcp_platform_mutex_unlock(memory->mutex);

    LOG_DEBUG("Stored memory: id=%s, type=%s, importance=%s, size=%zu",
                    id, nimcp_memory_type_to_string(type),
                    nimcp_memory_importance_to_string(importance), data_size);

    return NIMCP_SUCCESS;
}

nimcp_result_t nimcp_swarm_memory_retrieve(
    NimcpSwarmMemory *memory,
    const char *memory_id,
    void *out_data,
    size_t data_size)
{
    NIMCP_VALIDATE_NOT_NULL(memory);
    NIMCP_VALIDATE_NOT_NULL(memory_id);
    NIMCP_VALIDATE_NOT_NULL(out_data);

    if (!memory->is_initialized) {
        return NIMCP_ERROR;
    }

    nimcp_platform_mutex_lock(memory->mutex);

    /* Lookup memory */
    NimcpMemoryEntry *entry = (NimcpMemoryEntry *)nimcp_hash_table_get(memory->memories, memory_id);
    if (!entry) {
        LOG_WARN("Memory not found: %s", memory_id);
        nimcp_platform_mutex_unlock(memory->mutex);
        return NIMCP_ERROR_NOT_FOUND;
    }

    /* Check buffer size */
    if (data_size < entry->data_size) {
        LOG_ERROR("Buffer too small: need %zu, got %zu", entry->data_size, data_size);
        nimcp_platform_mutex_unlock(memory->mutex);
        return NIMCP_ERROR_BUFFER_TOO_SMALL;
    }

    /* Copy data */
    if (entry->is_compressed) {
        /* Decompress */
        nimcp_result_t result = apply_decompression(entry->data, entry->data_size, out_data, data_size);
        if (result != NIMCP_SUCCESS) {
            nimcp_platform_mutex_unlock(memory->mutex);
            return result;
        }
    } else {
        memcpy(out_data, entry->data, entry->data_size);
    }

    /* Update access tracking */
    nimcp_swarm_memory_access(memory, memory_id);

    nimcp_platform_mutex_unlock(memory->mutex);

    LOG_DEBUG("Retrieved memory: %s", memory_id);
    return NIMCP_SUCCESS;
}

nimcp_result_t nimcp_swarm_memory_access(
    NimcpSwarmMemory *memory,
    const char *memory_id)
{
    NIMCP_VALIDATE_NOT_NULL(memory);
    NIMCP_VALIDATE_NOT_NULL(memory_id);

    if (!memory->is_initialized) {
        return NIMCP_ERROR;
    }

    nimcp_platform_mutex_lock(memory->mutex);

    NimcpMemoryEntry *entry = (NimcpMemoryEntry *)nimcp_hash_table_get(memory->memories, memory_id);
    if (!entry) {
        nimcp_platform_mutex_unlock(memory->mutex);
        return NIMCP_ERROR_NOT_FOUND;
    }

    /* Update access tracking */
    entry->last_accessed = nimcp_time_get_us();
    entry->access_count++;

    /* Boost strength slightly on access */
    entry->strength = fminf(1.0f, entry->strength + 0.01f);

    nimcp_platform_mutex_unlock(memory->mutex);

    return NIMCP_SUCCESS;
}

nimcp_result_t nimcp_swarm_memory_rehearse(
    NimcpSwarmMemory *memory,
    const char *memory_id)
{
    NIMCP_VALIDATE_NOT_NULL(memory);
    NIMCP_VALIDATE_NOT_NULL(memory_id);

    if (!memory->is_initialized) {
        return NIMCP_ERROR;
    }

    nimcp_platform_mutex_lock(memory->mutex);

    NimcpMemoryEntry *entry = (NimcpMemoryEntry *)nimcp_hash_table_get(memory->memories, memory_id);
    if (!entry) {
        nimcp_platform_mutex_unlock(memory->mutex);
        return NIMCP_ERROR_NOT_FOUND;
    }

    /* Update rehearsal tracking */
    entry->last_rehearsed = nimcp_time_get_us();
    entry->rehearsal_count++;

    /* Apply rehearsal boost from forgetting curve */
    NimcpForgettingCurve *curve = &memory->curves[entry->type];
    entry->strength = fminf(1.0f, entry->strength + curve->rehearsal_boost);

    /* Update decay rate based on rehearsals */
    float modifier = calculate_decay_modifier(entry->importance, entry->rehearsal_count);
    entry->decay_rate = curve->decay_rate * modifier;

    nimcp_platform_mutex_unlock(memory->mutex);

    LOG_DEBUG("Rehearsed memory: %s (strength=%.3f, rehearsals=%u)",
                    memory_id, entry->strength, entry->rehearsal_count);

    return NIMCP_SUCCESS;
}

nimcp_result_t nimcp_swarm_memory_delete(
    NimcpSwarmMemory *memory,
    const char *memory_id)
{
    NIMCP_VALIDATE_NOT_NULL(memory);
    NIMCP_VALIDATE_NOT_NULL(memory_id);

    if (!memory->is_initialized) {
        return NIMCP_ERROR;
    }

    nimcp_platform_mutex_lock(memory->mutex);

    NimcpMemoryEntry *entry = (NimcpMemoryEntry *)nimcp_hash_table_get(memory->memories, memory_id);
    if (!entry) {
        nimcp_platform_mutex_unlock(memory->mutex);
        return NIMCP_ERROR_NOT_FOUND;
    }

    /* Remove from all tables */
    nimcp_hash_table_remove(memory->memories_by_type[entry->type], memory_id);
    nimcp_hash_table_remove(memory->memories, memory_id);

    /* Update statistics */
    memory->stats.total_memories--;
    memory->stats.forgotten_memories++;

    /* Destroy entry */
    destroy_memory_entry(entry);

    nimcp_platform_mutex_unlock(memory->mutex);

    LOG_DEBUG("Deleted memory: %s", memory_id);
    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Experience Replay Implementation
 * ============================================================================ */

nimcp_result_t nimcp_swarm_memory_schedule_replay(
    NimcpSwarmMemory *memory,
    const char *memory_id,
    float priority)
{
    NIMCP_VALIDATE_NOT_NULL(memory);
    NIMCP_VALIDATE_NOT_NULL(memory_id);

    if (!memory->is_initialized) {
        return NIMCP_ERROR;
    }

    nimcp_platform_mutex_lock(memory->mutex);

    NimcpMemoryEntry *entry = (NimcpMemoryEntry *)nimcp_hash_table_get(memory->memories, memory_id);
    if (!entry) {
        nimcp_platform_mutex_unlock(memory->mutex);
        return NIMCP_ERROR_NOT_FOUND;
    }

    /* Create replay entry */
    NimcpReplayEntry *replay = (NimcpReplayEntry *)nimcp_malloc(sizeof(NimcpReplayEntry));
    if (!replay) {
        nimcp_platform_mutex_unlock(memory->mutex);
        return NIMCP_NO_MEMORY;
    }

    replay->memory = entry;
    replay->replay_priority = priority;
    replay->replay_count = 0;
    replay->next_replay_time = nimcp_time_get_us();

    /* Add to replay queue */
    if (nimcp_replay_heap_insert(memory->replay_queue, replay) != NIMCP_SUCCESS) {
        nimcp_free(replay);
        nimcp_platform_mutex_unlock(memory->mutex);
        return NIMCP_ERROR;
    }

    nimcp_platform_mutex_unlock(memory->mutex);

    LOG_DEBUG("Scheduled replay: %s (priority=%.3f)", memory_id, priority);
    return NIMCP_SUCCESS;
}

nimcp_result_t nimcp_swarm_memory_replay_cycle(
    NimcpSwarmMemory *memory,
    uint32_t max_replays,
    uint32_t *replays_performed)
{
    NIMCP_VALIDATE_NOT_NULL(memory);

    if (!memory->is_initialized) {
        return NIMCP_ERROR;
    }

    nimcp_platform_mutex_lock(memory->mutex);

    uint32_t count = 0;
    uint64_t current_time = nimcp_time_get_us();

    LOG_DEBUG("Starting replay cycle (max=%u)", max_replays);

    while (count < max_replays && nimcp_min_heap_size(memory->replay_queue) > 0) {
        /* Check replay probability */
        float rand_val = (float)rand() / (float)RAND_MAX;
        if (rand_val > memory->replay_probability) {
            break;
        }

        /* Get highest priority replay */
        NimcpReplayEntry *replay = (NimcpReplayEntry *)nimcp_replay_heap_extract(memory->replay_queue);
        if (!replay) {
            break;
        }

        /* Check if ready for replay */
        if (replay->next_replay_time > current_time) {
            /* Put back in queue */
            nimcp_replay_heap_insert(memory->replay_queue, replay);
            break;
        }

        /* Perform replay */
        LOG_DEBUG("Replaying memory: %s (priority=%.3f)",
                        replay->memory->id, replay->replay_priority);

        /* Rehearse the memory */
        nimcp_swarm_memory_rehearse(memory, replay->memory->id);

        /* Share with swarm if important */
        if (replay->memory->importance >= NIMCP_IMPORTANCE_HIGH &&
            memory->bio_async_enabled) {
            /* Send to swarm */
            send_bio_message(memory, NIMCP_MSG_MEMORY_SHARE, "broadcast",
                           replay->memory->data, replay->memory->data_size);
            memory->stats.bytes_transmitted += replay->memory->data_size;
        }

        /* Update replay entry */
        replay->replay_count++;
        replay->next_replay_time = current_time +
            (uint64_t)(3600000.0f / replay->replay_priority);  /* Schedule next replay */

        /* Re-insert if still valuable */
        if (replay->replay_count < 10 && replay->memory->strength > NIMCP_MIN_MEMORY_STRENGTH) {
            nimcp_replay_heap_insert(memory->replay_queue, replay);
        } else {
            nimcp_free(replay);
        }

        count++;
        memory->stats.total_replays++;
        memory->stats.successful_replays++;
    }

    if (replays_performed) {
        *replays_performed = count;
    }

    nimcp_platform_mutex_unlock(memory->mutex);

    LOG_DEBUG("Replay cycle complete: performed=%u", count);
    return NIMCP_SUCCESS;
}

float nimcp_swarm_memory_calculate_replay_priority(
    NimcpSwarmMemory *memory,
    const NimcpMemoryEntry *memory_entry)
{
    if (!memory || !memory_entry) {
        return 0.0f;
    }

    /* Calculate priority based on multiple factors */
    float novelty_factor = memory_entry->novelty_score;
    float importance_factor = (float)memory_entry->importance / (float)NIMCP_IMPORTANCE_CRITICAL;

    /* Recency factor */
    uint64_t current_time = nimcp_time_get_us();
    uint64_t age = current_time - memory_entry->created_at;
    float recency_factor = expf(-(float)age / 3600000.0f);  /* Decay over hours */

    /* Combine factors */
    float priority = (novelty_factor * 0.4f) +
                     (importance_factor * 0.4f) +
                     (recency_factor * 0.2f);

    return fminf(1.0f, fmaxf(0.0f, priority));
}

/* ============================================================================
 * Knowledge Distillation Implementation
 * ============================================================================ */

nimcp_result_t nimcp_swarm_memory_compress(
    NimcpSwarmMemory *memory,
    const char *memory_id,
    NimcpCompressedMemory *compressed)
{
    NIMCP_VALIDATE_NOT_NULL(memory);
    NIMCP_VALIDATE_NOT_NULL(memory_id);
    NIMCP_VALIDATE_NOT_NULL(compressed);

    if (!memory->is_initialized) {
        return NIMCP_ERROR;
    }

    nimcp_platform_mutex_lock(memory->mutex);

    NimcpMemoryEntry *entry = (NimcpMemoryEntry *)nimcp_hash_table_get(memory->memories, memory_id);
    if (!entry) {
        nimcp_platform_mutex_unlock(memory->mutex);
        return NIMCP_ERROR_NOT_FOUND;
    }

    /* Apply compression */
    void *compressed_data = NULL;
    size_t compressed_size = 0;
    nimcp_result_t result = apply_compression(entry->data, entry->data_size,
                                          &compressed_data, &compressed_size);
    if (result != NIMCP_SUCCESS) {
        nimcp_platform_mutex_unlock(memory->mutex);
        return result;
    }

    /* Fill compressed memory structure */
    strncpy(compressed->id, entry->id, sizeof(compressed->id) - 1);
    compressed->type = entry->type;
    compressed->importance = entry->importance;
    compressed->compressed_data = compressed_data;
    compressed->compressed_size = compressed_size;
    compressed->original_size = entry->data_size;
    compressed->compression_ratio = (float)compressed_size / (float)entry->data_size;
    compressed->pattern_hash = 0;  /* TODO: Calculate pattern hash */

    /* Update statistics */
    memory->stats.avg_compression_ratio =
        (memory->stats.avg_compression_ratio * (memory->stats.total_memories - 1) +
         compressed->compression_ratio) / memory->stats.total_memories;

    nimcp_platform_mutex_unlock(memory->mutex);

    LOG_DEBUG("Compressed memory: %s (ratio=%.3f)", memory_id, compressed->compression_ratio);
    return NIMCP_SUCCESS;
}

nimcp_result_t nimcp_swarm_memory_decompress(
    NimcpSwarmMemory *memory,
    const NimcpCompressedMemory *compressed,
    void *decompressed,
    size_t buffer_size)
{
    NIMCP_VALIDATE_NOT_NULL(memory);
    NIMCP_VALIDATE_NOT_NULL(compressed);
    NIMCP_VALIDATE_NOT_NULL(decompressed);

    if (buffer_size < compressed->original_size) {
        return NIMCP_ERROR_BUFFER_TOO_SMALL;
    }

    return apply_decompression(compressed->compressed_data, compressed->compressed_size,
                              decompressed, buffer_size);
}

nimcp_result_t nimcp_swarm_memory_extract_pattern(
    NimcpSwarmMemory *memory,
    const char *memory_id,
    uint32_t *pattern_hash)
{
    NIMCP_VALIDATE_NOT_NULL(memory);
    NIMCP_VALIDATE_NOT_NULL(memory_id);
    NIMCP_VALIDATE_NOT_NULL(pattern_hash);

    if (!memory->is_initialized) {
        return NIMCP_ERROR;
    }

    nimcp_platform_mutex_lock(memory->mutex);

    NimcpMemoryEntry *entry = (NimcpMemoryEntry *)nimcp_hash_table_get(memory->memories, memory_id);
    if (!entry) {
        nimcp_platform_mutex_unlock(memory->mutex);
        return NIMCP_ERROR_NOT_FOUND;
    }

    /* Simple hash-based pattern extraction (placeholder) */
    *pattern_hash = 0;
    const uint8_t *data = (const uint8_t *)entry->data;
    for (size_t i = 0; i < entry->data_size; i++) {
        *pattern_hash = (*pattern_hash * 31) + data[i];
    }

    memory->compression.pattern_count++;

    nimcp_platform_mutex_unlock(memory->mutex);

    LOG_DEBUG("Extracted pattern from memory: %s (hash=0x%08X)", memory_id, *pattern_hash);
    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Forgetting Curve Implementation
 * ============================================================================ */

nimcp_result_t nimcp_swarm_memory_set_forgetting_curve(
    NimcpSwarmMemory *memory,
    NimcpMemoryType type,
    const NimcpForgettingCurve *curve)
{
    NIMCP_VALIDATE_NOT_NULL(memory);
    NIMCP_VALIDATE_NOT_NULL(curve);

    if (type >= NIMCP_MEMORY_TYPE_COUNT) {
        return NIMCP_INVALID_PARAM;
    }

    nimcp_platform_mutex_lock(memory->mutex);
    memcpy(&memory->curves[type], curve, sizeof(NimcpForgettingCurve));
    nimcp_platform_mutex_unlock(memory->mutex);

    LOG_DEBUG("Updated forgetting curve for type: %s", nimcp_memory_type_to_string(type));
    return NIMCP_SUCCESS;
}

float nimcp_swarm_memory_calculate_strength(
    NimcpSwarmMemory *memory,
    const NimcpMemoryEntry *memory_entry,
    uint64_t current_time)
{
    if (!memory || !memory_entry) {
        return 0.0f;
    }

    NimcpForgettingCurve *curve = &memory->curves[memory_entry->type];

    /* Calculate time since last rehearsal or creation */
    uint64_t time_since = current_time -
        (memory_entry->last_rehearsed > 0 ? memory_entry->last_rehearsed : memory_entry->created_at);

    /* Exponential decay based on forgetting curve */
    float decay_modifier = calculate_decay_modifier(memory_entry->importance,
                                                    memory_entry->rehearsal_count);
    float effective_decay = curve->decay_rate * decay_modifier;

    float strength = curve->initial_strength * expf(-effective_decay * (float)time_since / 1000.0f);

    /* Add importance modifier */
    strength *= (1.0f + curve->importance_modifier * (float)memory_entry->importance /
                 (float)NIMCP_IMPORTANCE_CRITICAL);

    return fminf(1.0f, fmaxf(0.0f, strength));
}

nimcp_result_t nimcp_swarm_memory_apply_forgetting(
    NimcpSwarmMemory *memory,
    uint32_t *forgotten_count)
{
    NIMCP_VALIDATE_NOT_NULL(memory);

    if (!memory->is_initialized) {
        return NIMCP_ERROR;
    }

    nimcp_platform_mutex_lock(memory->mutex);

    uint32_t count = 0;
    uint64_t current_time = nimcp_time_get_us();

    LOG_DEBUG("Applying forgetting to all memories");

    /* TODO: Iterate through all memories and update strengths */
    /* For now, simplified implementation */

    /* Collect weak memories to forget */
    char **to_forget = NULL;
    uint32_t to_forget_count = 0;

    /* Note: In real implementation, iterate through hash table */
    /* For demonstration, we'll just update the count */

    if (forgotten_count) {
        *forgotten_count = count;
    }

    memory->stats.forgotten_memories += count;

    nimcp_platform_mutex_unlock(memory->mutex);

    LOG_DEBUG("Forgetting complete: forgotten=%u", count);
    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Consolidation Window Implementation
 * ============================================================================ */

nimcp_result_t nimcp_swarm_memory_configure_consolidation(
    NimcpSwarmMemory *memory,
    const NimcpConsolidationWindow *window)
{
    NIMCP_VALIDATE_NOT_NULL(memory);
    NIMCP_VALIDATE_NOT_NULL(window);

    nimcp_platform_mutex_lock(memory->mutex);
    memcpy(&memory->window, window, sizeof(NimcpConsolidationWindow));
    nimcp_platform_mutex_unlock(memory->mutex);

    LOG_INFO("Consolidation window configured: mode=%s, duration=%lu ms",
                   nimcp_consolidation_mode_to_string(window->mode),
                   (unsigned long)window->window_duration_ms);

    return NIMCP_SUCCESS;
}

nimcp_result_t nimcp_swarm_memory_start_consolidation(
    NimcpSwarmMemory *memory,
    NimcpConsolidationMode mode)
{
    NIMCP_VALIDATE_NOT_NULL(memory);

    if (!memory->is_initialized) {
        return NIMCP_ERROR;
    }

    nimcp_platform_mutex_lock(memory->mutex);

    memory->window.mode = mode;
    memory->window.window_start = nimcp_time_get_us();

    nimcp_platform_mutex_unlock(memory->mutex);

    LOG_INFO("Started consolidation window: mode=%s",
                   nimcp_consolidation_mode_to_string(mode));

    /* Broadcast to swarm */
    if (memory->bio_async_enabled) {
        send_bio_message(memory, NIMCP_MSG_CONSOLIDATION_SIGNAL, "broadcast",
                        &mode, sizeof(mode));
    }

    return NIMCP_SUCCESS;
}

nimcp_result_t nimcp_swarm_memory_consolidate(
    NimcpSwarmMemory *memory,
    uint32_t *memories_consolidated)
{
    NIMCP_VALIDATE_NOT_NULL(memory);

    if (!memory->is_initialized) {
        return NIMCP_ERROR;
    }

    nimcp_platform_mutex_lock(memory->mutex);

    uint32_t count = 0;

    LOG_INFO("Starting memory consolidation");

    /* Apply forgetting */
    uint32_t forgotten = 0;
    nimcp_swarm_memory_apply_forgetting(memory, &forgotten);

    /* Perform replay cycle */
    uint32_t replays = 0;
    nimcp_swarm_memory_replay_cycle(memory, memory->window.max_memories_per_window, &replays);

    /* Compress important memories */
    if (memory->auto_compression) {
        /* TODO: Iterate and compress uncompressed important memories */
    }

    /* Distribute to swarm */
    if (memory->auto_distribution) {
        /* TODO: Distribute unconsolidated memories */
    }

    /* Update consolidation tracking */
    memory->last_consolidation = nimcp_time_get_us();
    memory->consolidation_count++;
    memory->stats.consolidated_memories += count;

    if (memories_consolidated) {
        *memories_consolidated = count;
    }

    nimcp_platform_mutex_unlock(memory->mutex);

    LOG_INFO("Consolidation complete: memories=%u, forgotten=%u, replays=%u",
                   count, forgotten, replays);

    return NIMCP_SUCCESS;
}

bool nimcp_swarm_memory_is_consolidating(const NimcpSwarmMemory *memory)
{
    if (!memory) {
        return false;
    }

    uint64_t current_time = nimcp_time_get_us();
    uint64_t elapsed = current_time - memory->window.window_start;

    return (elapsed < memory->window.window_duration_ms);
}

/* ============================================================================
 * Distributed Hippocampus Implementation
 * ============================================================================ */

nimcp_result_t nimcp_swarm_memory_register_node(
    NimcpSwarmMemory *memory,
    const char *node_id,
    uint32_t capacity)
{
    NIMCP_VALIDATE_NOT_NULL(memory);
    NIMCP_VALIDATE_NOT_NULL(node_id);

    if (!memory->is_initialized) {
        return NIMCP_ERROR;
    }

    nimcp_platform_mutex_lock(memory->mutex);

    /* Create node entry */
    NimcpHippocampusNode *node = (NimcpHippocampusNode *)nimcp_malloc(sizeof(NimcpHippocampusNode));
    if (!node) {
        nimcp_platform_mutex_unlock(memory->mutex);
        return NIMCP_NO_MEMORY;
    }

    strncpy(node->node_id, node_id, sizeof(node->node_id) - 1);
    node->is_active = true;
    node->memory_count = 0;
    node->last_sync_time = nimcp_time_get_us();
    node->health_score = 1.0f;
    node->replica_capacity = capacity;

    /* Register in hash table */
    if (nimcp_hash_table_insert(memory->hippocampus_nodes, node_id, node) != NIMCP_SUCCESS) {
        nimcp_free(node);
        nimcp_platform_mutex_unlock(memory->mutex);
        return NIMCP_ERROR;
    }

    memory->stats.active_nodes++;

    nimcp_platform_mutex_unlock(memory->mutex);

    LOG_INFO("Registered hippocampus node: %s (capacity=%u)", node_id, capacity);
    return NIMCP_SUCCESS;
}

nimcp_result_t nimcp_swarm_memory_unregister_node(
    NimcpSwarmMemory *memory,
    const char *node_id)
{
    NIMCP_VALIDATE_NOT_NULL(memory);
    NIMCP_VALIDATE_NOT_NULL(node_id);

    if (!memory->is_initialized) {
        return NIMCP_ERROR;
    }

    nimcp_platform_mutex_lock(memory->mutex);

    NimcpHippocampusNode *node = (NimcpHippocampusNode *)nimcp_hash_table_get(
        memory->hippocampus_nodes, node_id);
    if (!node) {
        nimcp_platform_mutex_unlock(memory->mutex);
        return NIMCP_ERROR_NOT_FOUND;
    }

    nimcp_hash_table_remove(memory->hippocampus_nodes, node_id);
    memory->stats.active_nodes--;
    nimcp_free(node);

    nimcp_platform_mutex_unlock(memory->mutex);

    LOG_INFO("Unregistered hippocampus node: %s", node_id);
    return NIMCP_SUCCESS;
}

nimcp_result_t nimcp_swarm_memory_distribute(
    NimcpSwarmMemory *memory,
    const char *memory_id,
    uint32_t *replicas_created)
{
    NIMCP_VALIDATE_NOT_NULL(memory);
    NIMCP_VALIDATE_NOT_NULL(memory_id);

    if (!memory->is_initialized) {
        return NIMCP_ERROR;
    }

    nimcp_platform_mutex_lock(memory->mutex);

    NimcpMemoryEntry *entry = (NimcpMemoryEntry *)nimcp_hash_table_get(memory->memories, memory_id);
    if (!entry) {
        nimcp_platform_mutex_unlock(memory->mutex);
        return NIMCP_ERROR_NOT_FOUND;
    }

    /* Check if already distributed */
    if (entry->is_distributed && entry->replica_count >= memory->replication_factor) {
        if (replicas_created) {
            *replicas_created = 0;
        }
        nimcp_platform_mutex_unlock(memory->mutex);
        return NIMCP_SUCCESS;
    }

    /* Select nodes for replication */
    uint32_t target_replicas = memory->replication_factor - entry->replica_count;
    char **node_ids = (char **)nimcp_malloc(sizeof(char *) * target_replicas);
    if (!node_ids) {
        nimcp_platform_mutex_unlock(memory->mutex);
        return NIMCP_NO_MEMORY;
    }

    for (uint32_t i = 0; i < target_replicas; i++) {
        node_ids[i] = (char *)nimcp_malloc(NIMCP_NODE_ID_MAX_LENGTH);
    }

    nimcp_result_t result = select_replication_nodes(memory, target_replicas, node_ids);
    if (result != NIMCP_SUCCESS) {
        for (uint32_t i = 0; i < target_replicas; i++) {
            nimcp_free(node_ids[i]);
        }
        nimcp_free(node_ids);
        nimcp_platform_mutex_unlock(memory->mutex);
        return result;
    }

    /* Send to selected nodes */
    uint32_t created = 0;
    for (uint32_t i = 0; i < target_replicas; i++) {
        if (memory->bio_async_enabled) {
            if (send_bio_message(memory, NIMCP_MSG_MEMORY_SHARE, node_ids[i],
                                entry->data, entry->data_size) == NIMCP_SUCCESS) {
                created++;
                memory->stats.bytes_transmitted += entry->data_size;
            }
        }
        nimcp_free(node_ids[i]);
    }
    nimcp_free(node_ids);

    /* Update entry */
    entry->replica_count += created;
    entry->is_distributed = (entry->replica_count >= memory->replication_factor);

    memory->stats.distributed_memories++;
    memory->stats.total_replicas += created;

    if (replicas_created) {
        *replicas_created = created;
    }

    nimcp_platform_mutex_unlock(memory->mutex);

    LOG_DEBUG("Distributed memory: %s (replicas=%u)", memory_id, created);
    return NIMCP_SUCCESS;
}

nimcp_result_t nimcp_swarm_memory_verify_consensus(
    NimcpSwarmMemory *memory,
    const char *memory_id,
    bool *has_consensus)
{
    NIMCP_VALIDATE_NOT_NULL(memory);
    NIMCP_VALIDATE_NOT_NULL(memory_id);
    NIMCP_VALIDATE_NOT_NULL(has_consensus);

    if (!memory->is_initialized) {
        return NIMCP_ERROR;
    }

    nimcp_platform_mutex_lock(memory->mutex);

    NimcpMemoryEntry *entry = (NimcpMemoryEntry *)nimcp_hash_table_get(memory->memories, memory_id);
    if (!entry) {
        nimcp_platform_mutex_unlock(memory->mutex);
        return NIMCP_ERROR_NOT_FOUND;
    }

    /* Check if sufficient replicas exist */
    float replica_ratio = (float)entry->replica_count / (float)memory->replication_factor;
    *has_consensus = (replica_ratio >= memory->consensus_threshold);

    nimcp_platform_mutex_unlock(memory->mutex);

    return NIMCP_SUCCESS;
}

nimcp_result_t nimcp_swarm_memory_sync_with_node(
    NimcpSwarmMemory *memory,
    const char *node_id,
    uint32_t *memories_synced)
{
    NIMCP_VALIDATE_NOT_NULL(memory);
    NIMCP_VALIDATE_NOT_NULL(node_id);

    if (!memory->is_initialized) {
        return NIMCP_ERROR;
    }

    nimcp_platform_mutex_lock(memory->mutex);

    NimcpHippocampusNode *node = (NimcpHippocampusNode *)nimcp_hash_table_get(
        memory->hippocampus_nodes, node_id);
    if (!node) {
        nimcp_platform_mutex_unlock(memory->mutex);
        return NIMCP_ERROR_NOT_FOUND;
    }

    /* Send sync request */
    if (memory->bio_async_enabled) {
        send_bio_message(memory, NIMCP_MSG_SYNC_REQUEST, node_id, NULL, 0);
    }

    /* Update sync time */
    node->last_sync_time = nimcp_time_get_us();

    if (memories_synced) {
        *memories_synced = 0;  /* TODO: Track actual synced count */
    }

    nimcp_platform_mutex_unlock(memory->mutex);

    LOG_DEBUG("Syncing with node: %s", node_id);
    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Semantic Compression Implementation
 * ============================================================================ */

nimcp_result_t nimcp_swarm_memory_abstract_pattern(
    NimcpSwarmMemory *memory,
    const char **memory_ids,
    uint32_t count,
    uint32_t *pattern_hash)
{
    NIMCP_VALIDATE_NOT_NULL(memory);
    NIMCP_VALIDATE_NOT_NULL(memory_ids);
    NIMCP_VALIDATE_NOT_NULL(pattern_hash);

    if (!memory->is_initialized) {
        return NIMCP_ERROR;
    }

    nimcp_platform_mutex_lock(memory->mutex);

    /* Extract patterns from each memory and combine */
    *pattern_hash = 0;
    for (uint32_t i = 0; i < count; i++) {
        uint32_t individual_hash = 0;
        nimcp_swarm_memory_extract_pattern(memory, memory_ids[i], &individual_hash);
        *pattern_hash ^= individual_hash;  /* XOR combine */
    }

    /* Store in pattern index */
    /* TODO: Implement pattern tree storage */

    nimcp_platform_mutex_unlock(memory->mutex);

    LOG_DEBUG("Abstracted pattern from %u memories (hash=0x%08X)", count, *pattern_hash);
    return NIMCP_SUCCESS;
}

nimcp_result_t nimcp_swarm_memory_generalize(
    NimcpSwarmMemory *memory,
    const char **specific_ids,
    uint32_t count,
    char *generalized_id)
{
    NIMCP_VALIDATE_NOT_NULL(memory);
    NIMCP_VALIDATE_NOT_NULL(specific_ids);
    NIMCP_VALIDATE_NOT_NULL(generalized_id);

    if (!memory->is_initialized) {
        return NIMCP_ERROR;
    }

    /* Abstract pattern */
    uint32_t pattern_hash = 0;
    nimcp_result_t result = nimcp_swarm_memory_abstract_pattern(memory, specific_ids, count, &pattern_hash);
    if (result != NIMCP_SUCCESS) {
        return result;
    }

    /* Create generalized memory */
    /* TODO: Implement proper generalization logic */
    generate_memory_id(generalized_id, NIMCP_MEMORY_ID_MAX_LENGTH);

    LOG_DEBUG("Generalized %u specific memories into: %s", count, generalized_id);
    return NIMCP_SUCCESS;
}

nimcp_result_t nimcp_swarm_memory_build_hierarchy(
    NimcpSwarmMemory *memory,
    uint32_t *levels)
{
    NIMCP_VALIDATE_NOT_NULL(memory);

    if (!memory->is_initialized) {
        return NIMCP_ERROR;
    }

    nimcp_platform_mutex_lock(memory->mutex);

    /* Build hierarchical knowledge structure */
    /* TODO: Implement hierarchy building */
    *levels = memory->compression.abstraction_level;

    nimcp_platform_mutex_unlock(memory->mutex);

    LOG_INFO("Built knowledge hierarchy: %u levels", *levels);
    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Bio-Async Integration Implementation
 * ============================================================================ */

nimcp_result_t nimcp_swarm_memory_process_message(
    NimcpSwarmMemory *memory,
    const void *msg)
{
    NIMCP_VALIDATE_NOT_NULL(memory);
    NIMCP_VALIDATE_NOT_NULL(msg);

    if (!memory->is_initialized || !memory->bio_async_enabled) {
        return NIMCP_SUCCESS;  /* Not enabled, return success */
    }

    /* Process incoming bio-async message */
    const bio_message_header_t *header = (const bio_message_header_t *)msg;

    switch (header->type) {
        case BIO_MSG_SWARM_MEMORY_SYNC:
            LOG_DEBUG("Received memory sync request from module 0x%x",
                      header->source_module);
            /* TODO: Handle memory sync request */
            break;

        case BIO_MSG_CONSOLIDATION_TRIGGER: {
            LOG_INFO("Received consolidation trigger");
            /* Trigger memory consolidation */
            uint32_t consolidated = 0;
            return nimcp_swarm_memory_consolidate(memory, &consolidated);
        }

        default:
            LOG_DEBUG("Received message type 0x%x (unhandled)", header->type);
            break;
    }

    return NIMCP_SUCCESS;
}

nimcp_result_t nimcp_swarm_memory_send_memory(
    NimcpSwarmMemory *memory,
    const char *memory_id,
    const char *target_node)
{
    NIMCP_VALIDATE_NOT_NULL(memory);
    NIMCP_VALIDATE_NOT_NULL(memory_id);
    NIMCP_VALIDATE_NOT_NULL(target_node);

    if (!memory->is_initialized || !memory->bio_async_enabled) {
        return NIMCP_ERROR;
    }

    nimcp_platform_mutex_lock(memory->mutex);

    NimcpMemoryEntry *entry = (NimcpMemoryEntry *)nimcp_hash_table_get(memory->memories, memory_id);
    if (!entry) {
        nimcp_platform_mutex_unlock(memory->mutex);
        return NIMCP_ERROR_NOT_FOUND;
    }

    /* Compress if needed */
    void *data_to_send = entry->data;
    size_t size_to_send = entry->data_size;

    if (memory->auto_compression && !entry->is_compressed) {
        NimcpCompressedMemory compressed;
        if (nimcp_swarm_memory_compress(memory, memory_id, &compressed) == NIMCP_SUCCESS) {
            data_to_send = compressed.compressed_data;
            size_to_send = compressed.compressed_size;
        }
    }

    /* Send message */
    nimcp_result_t result = send_bio_message(memory, NIMCP_MSG_MEMORY_SHARE, target_node,
                                         data_to_send, size_to_send);

    if (result == NIMCP_SUCCESS) {
        memory->stats.bytes_transmitted += size_to_send;
    }

    nimcp_platform_mutex_unlock(memory->mutex);

    return result;
}

nimcp_result_t nimcp_swarm_memory_request_memory(
    NimcpSwarmMemory *memory,
    const char *memory_id,
    const char *source_node)
{
    NIMCP_VALIDATE_NOT_NULL(memory);
    NIMCP_VALIDATE_NOT_NULL(memory_id);
    NIMCP_VALIDATE_NOT_NULL(source_node);

    if (!memory->is_initialized || !memory->bio_async_enabled) {
        return NIMCP_ERROR;
    }

    /* Send request message */
    return send_bio_message(memory, NIMCP_MSG_MEMORY_REQUEST, source_node,
                           memory_id, strlen(memory_id) + 1);
}

nimcp_result_t nimcp_swarm_memory_broadcast_consolidation(
    NimcpSwarmMemory *memory,
    NimcpConsolidationMode mode)
{
    NIMCP_VALIDATE_NOT_NULL(memory);

    if (!memory->is_initialized || !memory->bio_async_enabled) {
        return NIMCP_ERROR;
    }

    return send_bio_message(memory, NIMCP_MSG_CONSOLIDATION_SIGNAL, "broadcast",
                           &mode, sizeof(mode));
}

/* ============================================================================
 * Statistics and Monitoring Implementation
 * ============================================================================ */

nimcp_result_t nimcp_swarm_memory_get_statistics(
    const NimcpSwarmMemory *memory,
    NimcpMemoryStatistics *stats)
{
    NIMCP_VALIDATE_NOT_NULL(memory);
    NIMCP_VALIDATE_NOT_NULL(stats);

    if (!memory->is_initialized) {
        return NIMCP_ERROR;
    }

    memcpy(stats, &memory->stats, sizeof(NimcpMemoryStatistics));
    return NIMCP_SUCCESS;
}

uint32_t nimcp_swarm_memory_get_count_by_type(
    const NimcpSwarmMemory *memory,
    NimcpMemoryType type)
{
    if (!memory || type >= NIMCP_MEMORY_TYPE_COUNT) {
        return 0;
    }

    /* TODO: Return actual count from type-specific hash table */
    return 0;
}

float nimcp_swarm_memory_get_health_score(const NimcpSwarmMemory *memory)
{
    if (!memory || !memory->is_initialized) {
        return 0.0f;
    }

    /* Calculate health based on multiple factors */
    float node_health = (memory->stats.active_nodes > 0) ?
        (float)memory->stats.active_nodes / (float)memory->replication_factor : 0.0f;

    float replication_health = (memory->stats.distributed_memories > 0) ?
        (float)memory->stats.distributed_memories / (float)memory->stats.total_memories : 0.0f;

    float consolidation_health = (memory->consolidation_count > 0) ? 1.0f : 0.5f;

    float health = (node_health * 0.4f) + (replication_health * 0.4f) + (consolidation_health * 0.2f);

    return fminf(1.0f, fmaxf(0.0f, health));
}

void nimcp_swarm_memory_print_status(
    const NimcpSwarmMemory *memory,
    bool verbose)
{
    if (!memory) {
        return;
    }

    LOG_INFO("=== Swarm Memory Status ===");
    LOG_INFO("Total memories: %lu", (unsigned long)memory->stats.total_memories);
    LOG_INFO("Consolidated: %lu", (unsigned long)memory->stats.consolidated_memories);
    LOG_INFO("Distributed: %lu", (unsigned long)memory->stats.distributed_memories);
    LOG_INFO("Forgotten: %lu", (unsigned long)memory->stats.forgotten_memories);
    LOG_INFO("Active nodes: %u", memory->stats.active_nodes);
    LOG_INFO("Total replicas: %u", memory->stats.total_replicas);
    LOG_INFO("Health score: %.3f", nimcp_swarm_memory_get_health_score(memory));

    if (verbose) {
        LOG_INFO("Total replays: %lu", (unsigned long)memory->stats.total_replays);
        LOG_INFO("Successful replays: %lu", (unsigned long)memory->stats.successful_replays);
        LOG_INFO("Avg compression ratio: %.3f", memory->stats.avg_compression_ratio);
        LOG_INFO("Avg memory strength: %.3f", memory->stats.avg_memory_strength);
        LOG_INFO("Bytes transmitted: %lu", (unsigned long)memory->stats.bytes_transmitted);
        LOG_INFO("Bytes received: %lu", (unsigned long)memory->stats.bytes_received);
        LOG_INFO("Consolidations: %u", memory->consolidation_count);
        LOG_INFO("Pattern count: %u", memory->compression.pattern_count);
    }
}

/* ============================================================================
 * Utility Functions Implementation
 * ============================================================================ */

const char *nimcp_memory_type_to_string(NimcpMemoryType type)
{
    switch (type) {
        case NIMCP_MEMORY_EPISODIC:   return "EPISODIC";
        case NIMCP_MEMORY_SEMANTIC:   return "SEMANTIC";
        case NIMCP_MEMORY_PROCEDURAL: return "PROCEDURAL";
        case NIMCP_MEMORY_THREAT:     return "THREAT";
        case NIMCP_MEMORY_SPATIAL:    return "SPATIAL";
        default:                      return "UNKNOWN";
    }
}

const char *nimcp_consolidation_mode_to_string(NimcpConsolidationMode mode)
{
    switch (mode) {
        case NIMCP_CONSOLIDATION_ACTIVE:  return "ACTIVE";
        case NIMCP_CONSOLIDATION_PASSIVE: return "PASSIVE";
        case NIMCP_CONSOLIDATION_SLEEP:   return "SLEEP";
        default:                          return "UNKNOWN";
    }
}

const char *nimcp_memory_importance_to_string(NimcpMemoryImportance importance)
{
    switch (importance) {
        case NIMCP_IMPORTANCE_LOW:      return "LOW";
        case NIMCP_IMPORTANCE_MEDIUM:   return "MEDIUM";
        case NIMCP_IMPORTANCE_HIGH:     return "HIGH";
        case NIMCP_IMPORTANCE_CRITICAL: return "CRITICAL";
        default:                        return "UNKNOWN";
    }
}

/* ============================================================================
 * Helper Functions Implementation
 * ============================================================================ */

static NimcpMemoryEntry *create_memory_entry(
    NimcpMemoryType type,
    NimcpMemoryImportance importance,
    const void *data,
    size_t data_size,
    const char *source_node)
{
    NimcpMemoryEntry *entry = (NimcpMemoryEntry *)nimcp_malloc(sizeof(NimcpMemoryEntry));
    if (!entry) {
        return NULL;
    }

    memset(entry, 0, sizeof(NimcpMemoryEntry));

    /* Allocate and copy data */
    entry->data = nimcp_malloc(data_size);
    if (!entry->data) {
        nimcp_free(entry);
        return NULL;
    }
    memcpy(entry->data, data, data_size);

    entry->type = type;
    entry->importance = importance;
    entry->data_size = data_size;
    entry->created_at = nimcp_time_get_us();
    entry->last_accessed = entry->created_at;
    entry->last_rehearsed = 0;
    entry->access_count = 0;
    entry->rehearsal_count = 0;
    entry->strength = 1.0f;
    entry->decay_rate = NIMCP_DEFAULT_DECAY_RATE;
    entry->novelty_score = 0.0f;
    entry->is_compressed = false;
    entry->is_consolidated = false;
    entry->is_distributed = false;
    entry->replica_count = 0;

    if (source_node) {
        strncpy(entry->source_node, source_node, sizeof(entry->source_node) - 1);
    }

    return entry;
}

static void destroy_memory_entry(NimcpMemoryEntry *entry)
{
    if (!entry) {
        return;
    }

    if (entry->data) {
        nimcp_free(entry->data);
    }

    nimcp_free(entry);
}

static void generate_memory_id(char *buffer, size_t buffer_size)
{
    static uint64_t counter = 0;
    uint64_t timestamp = nimcp_time_get_us();

    snprintf(buffer, buffer_size, "MEM_%016lX_%08lX",
             (unsigned long)timestamp, (unsigned long)counter++);
}

static float calculate_decay_modifier(
    NimcpMemoryImportance importance,
    uint32_t rehearsal_count)
{
    /* More important memories decay slower */
    float importance_factor = 1.0f - ((float)importance / (float)NIMCP_IMPORTANCE_CRITICAL) * 0.5f;

    /* More rehearsals reduce decay */
    float rehearsal_factor = 1.0f / (1.0f + logf(1.0f + (float)rehearsal_count));

    return importance_factor * rehearsal_factor;
}

static nimcp_result_t apply_compression(
    const void *data,
    size_t data_size,
    void **compressed,
    size_t *compressed_size)
{
    /* Simple compression placeholder - in real implementation use zlib, lz4, etc. */
    *compressed = nimcp_malloc(data_size);
    if (!*compressed) {
        return NIMCP_NO_MEMORY;
    }

    memcpy(*compressed, data, data_size);
    *compressed_size = data_size;  /* No actual compression in placeholder */

    return NIMCP_SUCCESS;
}

static nimcp_result_t apply_decompression(
    const void *compressed,
    size_t compressed_size,
    void *decompressed,
    size_t buffer_size)
{
    /* Simple decompression placeholder */
    if (buffer_size < compressed_size) {
        return NIMCP_ERROR_BUFFER_TOO_SMALL;
    }

    memcpy(decompressed, compressed, compressed_size);
    return NIMCP_SUCCESS;
}

static int compare_replay_priority(const void *a, const void *b)
{
    const NimcpReplayEntry *entry_a = (const NimcpReplayEntry *)a;
    const NimcpReplayEntry *entry_b = (const NimcpReplayEntry *)b;

    /* Higher priority comes first (min heap, so negate) */
    if (entry_a->replay_priority > entry_b->replay_priority) {
        return -1;
    } else if (entry_a->replay_priority < entry_b->replay_priority) {
        return 1;
    }
    return 0;
}

static nimcp_result_t send_bio_message(
    NimcpSwarmMemory *memory,
    const char *msg_type,
    const char *target_node,
    const void *payload,
    size_t payload_size)
{
    if (!memory->bio_async_enabled || !memory->bio_ctx) {
        return NIMCP_SUCCESS;  /* Not enabled, return success */
    }

    if (!bio_router_is_initialized()) {
        LOG_DEBUG("Bio message queued: type=%s, target=%s, size=%zu",
                  msg_type, target_node, payload_size);
        return NIMCP_SUCCESS;
    }

    /* Build bio-async message */
    size_t total_size = sizeof(bio_message_header_t) + payload_size;
    uint8_t *buffer = nimcp_malloc(total_size);
    if (!buffer) {
        return NIMCP_ERROR_MEMORY;
    }

    bio_message_header_t *header = (bio_message_header_t *)buffer;
    header->type = BIO_MSG_SWARM_MEMORY_SYNC;
    header->sequence_id = 0;
    header->source_module = BIO_MODULE_SWARM_MEMORY;
    header->target_module = BIO_MODULE_ALL;
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    header->timestamp_us = (uint64_t)ts.tv_sec * 1000000ULL + (uint64_t)ts.tv_nsec / 1000ULL;
    header->channel = BIO_CHANNEL_ACETYLCHOLINE;  /* Memory operations */
    header->payload_size = (uint32_t)payload_size;
    header->flags = BIO_MSG_FLAG_BROADCAST;

    if (payload && payload_size > 0) {
        memcpy(buffer + sizeof(bio_message_header_t), payload, payload_size);
    }

    nimcp_error_t err = bio_router_broadcast(
        (bio_module_context_t)memory->bio_ctx, buffer, total_size
    );
    nimcp_free(buffer);

    if (err != NIMCP_SUCCESS) {
        LOG_WARN("Bio message send failed: type=%s, target=%s, error=%d",
                 msg_type, target_node, err);
        return (nimcp_result_t)err;
    }

    LOG_DEBUG("Bio message sent: type=%s, target=%s, size=%zu",
              msg_type, target_node, payload_size);
    (void)msg_type;  /* Used in log above */
    (void)target_node;  /* Used in log above */
    return NIMCP_SUCCESS;
}

static void initialize_default_forgetting_curves(NimcpSwarmMemory *memory)
{
    for (int i = 0; i < NIMCP_MEMORY_TYPE_COUNT; i++) {
        memory->curves[i].initial_strength = 1.0f;
        memory->curves[i].decay_rate = NIMCP_DEFAULT_DECAY_RATE;
        memory->curves[i].importance_modifier = 0.5f;
        memory->curves[i].rehearsal_boost = NIMCP_DEFAULT_REHEARSAL_BOOST;
        memory->curves[i].half_life_ms = NIMCP_DEFAULT_HALF_LIFE_MS;

        /* Adjust for specific types */
        switch (i) {
            case NIMCP_MEMORY_THREAT:
                /* Threat memories decay slower */
                memory->curves[i].decay_rate *= 0.5f;
                memory->curves[i].half_life_ms *= 2;
                break;
            case NIMCP_MEMORY_EPISODIC:
                /* Episodic memories decay faster */
                memory->curves[i].decay_rate *= 1.5f;
                break;
            case NIMCP_MEMORY_SEMANTIC:
                /* Semantic memories very stable */
                memory->curves[i].decay_rate *= 0.3f;
                memory->curves[i].half_life_ms *= 3;
                break;
            default:
                break;
        }
    }
}

static nimcp_result_t select_replication_nodes(
    NimcpSwarmMemory *memory,
    uint32_t count,
    char **node_ids)
{
    /* Simple selection: pick first N active nodes */
    /* TODO: Implement smart selection based on load, health, etc. */

    uint32_t selected = 0;
    /* Iterate through hippocampus_nodes and select */

    if (selected < count) {
        LOG_WARN("Could not select enough replication nodes: needed=%u, got=%u",
                         count, selected);
    }

    return NIMCP_SUCCESS;
}
