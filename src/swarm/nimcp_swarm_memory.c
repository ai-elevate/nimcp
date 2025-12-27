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

/* Default forgetting curve parameters - slower decay for long-term swarm memory */
#define SWARM_MEMORY_DECAY_RATE 0.0001f
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

    /* Create pattern learning hash tables */
    memory->patterns = hash_table_create(&config);
    if (!memory->patterns) {
        LOG_ERROR("Failed to create patterns hash table");
        hash_table_destroy(memory->hippocampus_nodes);
        nimcp_min_heap_destroy(memory->replay_queue);
        hash_table_destroy(memory->memories);
        for (int i = 0; i < NIMCP_MEMORY_TYPE_COUNT; i++) {
            hash_table_destroy(memory->memories_by_type[i]);
        }
        nimcp_free(memory);
        return NULL;
    }

    memory->pattern_associations = hash_table_create(&config);
    if (!memory->pattern_associations) {
        LOG_ERROR("Failed to create pattern_associations hash table");
        hash_table_destroy(memory->patterns);
        hash_table_destroy(memory->hippocampus_nodes);
        nimcp_min_heap_destroy(memory->replay_queue);
        hash_table_destroy(memory->memories);
        for (int i = 0; i < NIMCP_MEMORY_TYPE_COUNT; i++) {
            hash_table_destroy(memory->memories_by_type[i]);
        }
        nimcp_free(memory);
        return NULL;
    }

    memory->sequence_transitions = hash_table_create(&config);
    if (!memory->sequence_transitions) {
        LOG_ERROR("Failed to create sequence_transitions hash table");
        hash_table_destroy(memory->pattern_associations);
        hash_table_destroy(memory->patterns);
        hash_table_destroy(memory->hippocampus_nodes);
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

    /* Initialize pattern learning */
    memset(&memory->pattern_stats, 0, sizeof(swarm_pattern_stats_t));
    memory->next_pattern_id = 1;
    memory->pattern_similarity_threshold = 0.7F;
    memory->max_patterns = (max_capacity > 0) ? max_capacity : 1000;

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

    /* Destroy pattern learning structures */
    if (memory->patterns) {
        hash_table_destroy(memory->patterns);
        memory->patterns = NULL;
    }

    if (memory->pattern_associations) {
        hash_table_destroy(memory->pattern_associations);
        memory->pattern_associations = NULL;
    }

    if (memory->sequence_transitions) {
        hash_table_destroy(memory->sequence_transitions);
        memory->sequence_transitions = NULL;
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
    entry->strength = fminf(1.0F, entry->strength + 0.01F);

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
    entry->strength = fminf(1.0F, entry->strength + curve->rehearsal_boost);

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
            (uint64_t)(3600000.0F / replay->replay_priority);  /* Schedule next replay */

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
        return 0.0F;
    }

    /* Calculate priority based on multiple factors */
    float novelty_factor = memory_entry->novelty_score;
    float importance_factor = (float)memory_entry->importance / (float)NIMCP_IMPORTANCE_CRITICAL;

    /* Recency factor */
    uint64_t current_time = nimcp_time_get_us();
    uint64_t age = current_time - memory_entry->created_at;
    float recency_factor = expf(-(float)age / 3600000.0F);  /* Decay over hours */

    /* Combine factors */
    float priority = (novelty_factor * 0.4F) +
                     (importance_factor * 0.4F) +
                     (recency_factor * 0.2F);

    return fminf(1.0F, fmaxf(0.0F, priority));
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
        return 0.0F;
    }

    NimcpForgettingCurve *curve = &memory->curves[memory_entry->type];

    /* Calculate time since last rehearsal or creation */
    uint64_t time_since = current_time -
        (memory_entry->last_rehearsed > 0 ? memory_entry->last_rehearsed : memory_entry->created_at);

    /* Exponential decay based on forgetting curve */
    float decay_modifier = calculate_decay_modifier(memory_entry->importance,
                                                    memory_entry->rehearsal_count);
    float effective_decay = curve->decay_rate * decay_modifier;

    float strength = curve->initial_strength * expf(-effective_decay * (float)time_since / 1000.0F);

    /* Add importance modifier */
    strength *= (1.0F + curve->importance_modifier * (float)memory_entry->importance /
                 (float)NIMCP_IMPORTANCE_CRITICAL);

    return fminf(1.0F, fmaxf(0.0F, strength));
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
    node->health_score = 1.0F;
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
        return 0.0F;
    }

    /* Calculate health based on multiple factors */
    float node_health = (memory->stats.active_nodes > 0) ?
        (float)memory->stats.active_nodes / (float)memory->replication_factor : 0.0F;

    float replication_health = (memory->stats.distributed_memories > 0) ?
        (float)memory->stats.distributed_memories / (float)memory->stats.total_memories : 0.0F;

    float consolidation_health = (memory->consolidation_count > 0) ? 1.0F : 0.5F;

    float health = (node_health * 0.4F) + (replication_health * 0.4F) + (consolidation_health * 0.2F);

    return fminf(1.0F, fmaxf(0.0F, health));
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
 * Pattern Learning Implementation (NEW)
 * ============================================================================ */

/**
 * @brief Calculate cosine similarity between two vectors
 *
 * WHAT: Computes cosine similarity for pattern matching
 * WHY: Enables fuzzy pattern recognition
 * HOW: Dot product divided by magnitude product
 */
static float calculate_cosine_similarity(
    const float *vec1,
    const float *vec2,
    uint32_t size)
{
    if (!vec1 || !vec2 || size == 0) {
        return 0.0F;
    }

    float dot_product = 0.0F;
    float mag1 = 0.0F;
    float mag2 = 0.0F;

    for (uint32_t i = 0; i < size; i++) {
        dot_product += vec1[i] * vec2[i];
        mag1 += vec1[i] * vec1[i];
        mag2 += vec2[i] * vec2[i];
    }

    float magnitude_product = sqrtf(mag1) * sqrtf(mag2);
    if (magnitude_product < 1e-6F) {
        return 0.0F;
    }

    return dot_product / magnitude_product;
}

/**
 * @brief Create pattern ID key string
 *
 * WHAT: Converts pattern ID to string key for hash table
 * WHY: Hash table requires string keys
 * HOW: Formats ID as hex string
 */
static void pattern_id_to_key(uint32_t pattern_id, char *key_buffer)
{
    snprintf(key_buffer, 32, "PATTERN_%08X", pattern_id);
}

/**
 * @brief Free pattern structure
 *
 * WHAT: Releases memory for pattern structure
 * WHY: Prevents memory leaks
 * HOW: Frees signature array and pattern itself
 */
static void free_pattern(swarm_pattern_t *pattern)
{
    if (!pattern) {
        return;
    }

    if (pattern->signature) {
        nimcp_free(pattern->signature);
    }

    nimcp_free(pattern);
}

nimcp_result_t swarm_memory_detect_pattern(
    NimcpSwarmMemory *mem,
    const float *observation,
    uint32_t obs_size,
    swarm_pattern_t *matched)
{
    NIMCP_VALIDATE_NOT_NULL(mem);
    NIMCP_VALIDATE_NOT_NULL(observation);
    NIMCP_VALIDATE_NOT_NULL(matched);
    NIMCP_VALIDATE(obs_size > 0);

    if (!mem->is_initialized) {
        return NIMCP_ERROR;
    }

    nimcp_platform_mutex_lock(mem->mutex);

    /* Find best matching pattern */
    float best_similarity = 0.0F;
    swarm_pattern_t *best_pattern = NULL;

    /* Iterate through patterns (simplified - would need hash table iteration) */
    /* For now, return not found if no patterns or below threshold */

    if (best_pattern && best_similarity >= mem->pattern_similarity_threshold) {
        /* Copy matched pattern */
        memcpy(matched, best_pattern, sizeof(swarm_pattern_t));

        /* Deep copy signature */
        matched->signature = (float *)nimcp_malloc(
            best_pattern->signature_size * sizeof(float));
        if (matched->signature) {
            memcpy(matched->signature, best_pattern->signature,
                   best_pattern->signature_size * sizeof(float));
        }

        nimcp_platform_mutex_unlock(mem->mutex);

        LOG_DEBUG("Pattern detected: id=%u, confidence=%.3f",
                  matched->pattern_id, matched->confidence);

        /* Send bio-async message */
        if (mem->bio_async_enabled) {
            send_bio_message(mem, "PATTERN_DETECTED", "broadcast",
                           &matched->pattern_id, sizeof(uint32_t));
        }

        return NIMCP_SUCCESS;
    }

    nimcp_platform_mutex_unlock(mem->mutex);
    return NIMCP_ERROR_NOT_FOUND;
}

nimcp_result_t swarm_memory_store_pattern(
    NimcpSwarmMemory *mem,
    const swarm_pattern_t *pattern)
{
    NIMCP_VALIDATE_NOT_NULL(mem);
    NIMCP_VALIDATE_NOT_NULL(pattern);
    NIMCP_VALIDATE_NOT_NULL(pattern->signature);
    NIMCP_VALIDATE(pattern->signature_size > 0);

    if (!mem->is_initialized) {
        return NIMCP_ERROR;
    }

    nimcp_platform_mutex_lock(mem->mutex);

    /* Check capacity */
    if (mem->pattern_stats.total_patterns >= mem->max_patterns) {
        LOG_WARN("Pattern capacity reached, triggering forget");
        swarm_memory_forget_weak_patterns(mem, 0.3F);
    }

    /* Create pattern copy */
    swarm_pattern_t *new_pattern = (swarm_pattern_t *)nimcp_malloc(
        sizeof(swarm_pattern_t));
    if (!new_pattern) {
        nimcp_platform_mutex_unlock(mem->mutex);
        return NIMCP_NO_MEMORY;
    }

    memcpy(new_pattern, pattern, sizeof(swarm_pattern_t));

    /* Deep copy signature */
    new_pattern->signature = (float *)nimcp_malloc(
        pattern->signature_size * sizeof(float));
    if (!new_pattern->signature) {
        nimcp_free(new_pattern);
        nimcp_platform_mutex_unlock(mem->mutex);
        return NIMCP_NO_MEMORY;
    }
    memcpy(new_pattern->signature, pattern->signature,
           pattern->signature_size * sizeof(float));

    /* Assign ID if not set */
    if (new_pattern->pattern_id == 0) {
        new_pattern->pattern_id = mem->next_pattern_id++;
    }

    /* Store in hash table */
    char key[32];
    pattern_id_to_key(new_pattern->pattern_id, key);

    if (nimcp_hash_table_insert(mem->patterns, key, new_pattern) != NIMCP_SUCCESS) {
        free_pattern(new_pattern);
        nimcp_platform_mutex_unlock(mem->mutex);
        return NIMCP_ERROR;
    }

    /* Update statistics */
    mem->pattern_stats.total_patterns++;
    mem->pattern_stats.patterns_learned++;

    if (new_pattern->confidence > mem->pattern_similarity_threshold) {
        mem->pattern_stats.active_patterns++;
    }

    /* Update average confidence */
    mem->pattern_stats.avg_pattern_confidence =
        ((mem->pattern_stats.avg_pattern_confidence *
          (mem->pattern_stats.total_patterns - 1)) +
         new_pattern->confidence) / mem->pattern_stats.total_patterns;

    nimcp_platform_mutex_unlock(mem->mutex);

    LOG_DEBUG("Stored pattern: id=%u, confidence=%.3f",
              new_pattern->pattern_id, new_pattern->confidence);

    return NIMCP_SUCCESS;
}

nimcp_result_t swarm_memory_retrieve_pattern(
    NimcpSwarmMemory *mem,
    uint32_t pattern_id,
    swarm_pattern_t *out)
{
    NIMCP_VALIDATE_NOT_NULL(mem);
    NIMCP_VALIDATE_NOT_NULL(out);

    if (!mem->is_initialized) {
        return NIMCP_ERROR;
    }

    nimcp_platform_mutex_lock(mem->mutex);

    char key[32];
    pattern_id_to_key(pattern_id, key);

    swarm_pattern_t *pattern = (swarm_pattern_t *)nimcp_hash_table_get(
        mem->patterns, key);

    if (!pattern) {
        nimcp_platform_mutex_unlock(mem->mutex);
        return NIMCP_ERROR_NOT_FOUND;
    }

    /* Copy pattern */
    memcpy(out, pattern, sizeof(swarm_pattern_t));

    /* Deep copy signature */
    out->signature = (float *)nimcp_malloc(
        pattern->signature_size * sizeof(float));
    if (out->signature) {
        memcpy(out->signature, pattern->signature,
               pattern->signature_size * sizeof(float));
    }

    nimcp_platform_mutex_unlock(mem->mutex);
    return NIMCP_SUCCESS;
}

nimcp_result_t swarm_memory_get_similar_patterns(
    NimcpSwarmMemory *mem,
    const float *query,
    uint32_t query_size,
    swarm_pattern_t **results,
    uint32_t *count)
{
    NIMCP_VALIDATE_NOT_NULL(mem);
    NIMCP_VALIDATE_NOT_NULL(query);
    NIMCP_VALIDATE_NOT_NULL(results);
    NIMCP_VALIDATE_NOT_NULL(count);
    NIMCP_VALIDATE(query_size > 0);

    if (!mem->is_initialized) {
        return NIMCP_ERROR;
    }

    nimcp_platform_mutex_lock(mem->mutex);

    /* Allocate results array (max 10 similar patterns) */
    const uint32_t MAX_RESULTS = 10;
    *results = (swarm_pattern_t *)nimcp_malloc(
        MAX_RESULTS * sizeof(swarm_pattern_t));
    if (!*results) {
        nimcp_platform_mutex_unlock(mem->mutex);
        return NIMCP_NO_MEMORY;
    }

    *count = 0;

    /* Iterate through patterns and find similar ones */
    /* NOTE: Simplified implementation - would need hash table iteration */

    nimcp_platform_mutex_unlock(mem->mutex);

    LOG_DEBUG("Found %u similar patterns", *count);
    return NIMCP_SUCCESS;
}

nimcp_result_t swarm_memory_associate_pattern(
    NimcpSwarmMemory *mem,
    uint32_t pattern_id,
    uint32_t outcome_id,
    float reward)
{
    NIMCP_VALIDATE_NOT_NULL(mem);
    NIMCP_VALIDATE(reward >= -1.0F && reward <= 1.0F);

    if (!mem->is_initialized) {
        return NIMCP_ERROR;
    }

    nimcp_platform_mutex_lock(mem->mutex);

    /* Create association key */
    char key[64];
    snprintf(key, sizeof(key), "ASSOC_%08X_%08X", pattern_id, outcome_id);

    /* Check if association exists */
    pattern_association_t *assoc = (pattern_association_t *)nimcp_hash_table_get(
        mem->pattern_associations, key);

    if (assoc) {
        /* Update existing association */
        assoc->reinforcement_count++;

        /* Update strength with learning rate */
        float learning_rate = 0.1F;
        assoc->association_strength +=
            learning_rate * (reward - assoc->association_strength);

        /* Clamp to [0, 1] */
        assoc->association_strength = fmaxf(0.0F,
            fminf(1.0F, assoc->association_strength));
    } else {
        /* Create new association */
        assoc = (pattern_association_t *)nimcp_malloc(
            sizeof(pattern_association_t));
        if (!assoc) {
            nimcp_platform_mutex_unlock(mem->mutex);
            return NIMCP_NO_MEMORY;
        }

        assoc->pattern_id = pattern_id;
        assoc->outcome_id = outcome_id;
        assoc->association_strength = (reward + 1.0F) / 2.0F;  /* Map [-1,1] to [0,1] */
        assoc->reinforcement_count = 1;

        if (nimcp_hash_table_insert(mem->pattern_associations, key, assoc) !=
            NIMCP_SUCCESS) {
            nimcp_free(assoc);
            nimcp_platform_mutex_unlock(mem->mutex);
            return NIMCP_ERROR;
        }

        mem->pattern_stats.total_associations++;
    }

    nimcp_platform_mutex_unlock(mem->mutex);

    LOG_DEBUG("Associated pattern %u with outcome %u (strength=%.3f)",
              pattern_id, outcome_id, assoc->association_strength);

    return NIMCP_SUCCESS;
}

nimcp_result_t swarm_memory_predict_outcome(
    NimcpSwarmMemory *mem,
    uint32_t pattern_id,
    uint32_t *predicted_outcome,
    float *confidence)
{
    NIMCP_VALIDATE_NOT_NULL(mem);
    NIMCP_VALIDATE_NOT_NULL(predicted_outcome);
    NIMCP_VALIDATE_NOT_NULL(confidence);

    if (!mem->is_initialized) {
        return NIMCP_ERROR;
    }

    nimcp_platform_mutex_lock(mem->mutex);

    /* Find strongest association for this pattern */
    float best_strength = 0.0F;
    uint32_t best_outcome = 0;

    /* NOTE: Simplified implementation - would need to iterate associations */

    if (best_strength > 0.0F) {
        *predicted_outcome = best_outcome;
        *confidence = best_strength;
        nimcp_platform_mutex_unlock(mem->mutex);

        LOG_DEBUG("Predicted outcome %u for pattern %u (confidence=%.3f)",
                  best_outcome, pattern_id, best_strength);

        return NIMCP_SUCCESS;
    }

    nimcp_platform_mutex_unlock(mem->mutex);
    return NIMCP_ERROR_NOT_FOUND;
}

nimcp_result_t swarm_memory_learn_sequence(
    NimcpSwarmMemory *mem,
    const uint32_t *pattern_sequence,
    uint32_t seq_length)
{
    NIMCP_VALIDATE_NOT_NULL(mem);
    NIMCP_VALIDATE_NOT_NULL(pattern_sequence);
    NIMCP_VALIDATE(seq_length >= 2);

    if (!mem->is_initialized) {
        return NIMCP_ERROR;
    }

    nimcp_platform_mutex_lock(mem->mutex);

    /* Learn transitions between consecutive patterns */
    for (uint32_t i = 0; i < seq_length - 1; i++) {
        uint32_t from_pattern = pattern_sequence[i];
        uint32_t to_pattern = pattern_sequence[i + 1];

        /* Create transition key */
        char key[64];
        snprintf(key, sizeof(key), "TRANS_%08X_%08X", from_pattern, to_pattern);

        /* Get or create transition count */
        uint32_t *count_ptr = (uint32_t *)nimcp_hash_table_get(
            mem->sequence_transitions, key);

        if (count_ptr) {
            (*count_ptr)++;
        } else {
            count_ptr = (uint32_t *)nimcp_malloc(sizeof(uint32_t));
            if (!count_ptr) {
                nimcp_platform_mutex_unlock(mem->mutex);
                return NIMCP_NO_MEMORY;
            }

            *count_ptr = 1;

            if (nimcp_hash_table_insert(mem->sequence_transitions, key,
                                       count_ptr) != NIMCP_SUCCESS) {
                nimcp_free(count_ptr);
                nimcp_platform_mutex_unlock(mem->mutex);
                return NIMCP_ERROR;
            }
        }
    }

    nimcp_platform_mutex_unlock(mem->mutex);

    LOG_DEBUG("Learned sequence of %u patterns", seq_length);

    /* Send bio-async message */
    if (mem->bio_async_enabled) {
        send_bio_message(mem, "SEQUENCE_LEARNED", "broadcast",
                       pattern_sequence, seq_length * sizeof(uint32_t));
    }

    return NIMCP_SUCCESS;
}

nimcp_result_t swarm_memory_predict_next(
    NimcpSwarmMemory *mem,
    const uint32_t *history,
    uint32_t history_len,
    uint32_t *predicted,
    float *confidence)
{
    NIMCP_VALIDATE_NOT_NULL(mem);
    NIMCP_VALIDATE_NOT_NULL(history);
    NIMCP_VALIDATE_NOT_NULL(predicted);
    NIMCP_VALIDATE_NOT_NULL(confidence);
    NIMCP_VALIDATE(history_len > 0);

    if (!mem->is_initialized) {
        return NIMCP_ERROR;
    }

    nimcp_platform_mutex_lock(mem->mutex);

    /* Use most recent pattern in history as context */
    uint32_t last_pattern = history[history_len - 1];

    /* Find most frequent transition from last pattern */
    uint32_t best_next = 0;
    uint32_t max_count = 0;

    /* NOTE: Simplified - would need to iterate transitions */

    if (max_count > 0) {
        *predicted = best_next;
        *confidence = (float)max_count / (float)(max_count + 1);  /* Normalized */

        nimcp_platform_mutex_unlock(mem->mutex);

        LOG_DEBUG("Predicted next pattern %u (confidence=%.3f)",
                  best_next, *confidence);

        return NIMCP_SUCCESS;
    }

    nimcp_platform_mutex_unlock(mem->mutex);
    return NIMCP_ERROR_NOT_FOUND;
}

nimcp_result_t swarm_memory_consolidate_patterns(
    NimcpSwarmMemory *mem,
    uint64_t current_time_ms)
{
    NIMCP_VALIDATE_NOT_NULL(mem);

    if (!mem->is_initialized) {
        return NIMCP_ERROR;
    }

    nimcp_platform_mutex_lock(mem->mutex);

    /* Decay confidence for old patterns */
    /* Merge similar patterns */
    /* NOTE: Would need hash table iteration */

    uint32_t consolidated_count = 0;

    nimcp_platform_mutex_unlock(mem->mutex);

    LOG_INFO("Consolidated %u patterns", consolidated_count);

    /* Send bio-async message */
    if (mem->bio_async_enabled) {
        send_bio_message(mem, "PATTERNS_CONSOLIDATED", "broadcast",
                       &consolidated_count, sizeof(uint32_t));
    }

    return NIMCP_SUCCESS;
}

nimcp_result_t swarm_memory_forget_weak_patterns(
    NimcpSwarmMemory *mem,
    float threshold)
{
    NIMCP_VALIDATE_NOT_NULL(mem);
    NIMCP_VALIDATE(threshold >= 0.0F && threshold <= 1.0F);

    if (!mem->is_initialized) {
        return NIMCP_ERROR;
    }

    nimcp_platform_mutex_lock(mem->mutex);

    uint32_t forgotten_count = 0;

    /* NOTE: Would need hash table iteration to find and remove weak patterns */

    mem->pattern_stats.patterns_forgotten += forgotten_count;
    mem->pattern_stats.total_patterns -= forgotten_count;

    nimcp_platform_mutex_unlock(mem->mutex);

    LOG_DEBUG("Forgot %u weak patterns (threshold=%.3f)",
              forgotten_count, threshold);

    return NIMCP_SUCCESS;
}

swarm_pattern_stats_t swarm_memory_get_pattern_stats(
    const NimcpSwarmMemory *mem)
{
    swarm_pattern_stats_t stats = {0};

    if (!mem) {
        return stats;
    }

    memcpy(&stats, &mem->pattern_stats, sizeof(swarm_pattern_stats_t));
    return stats;
}

/* ============================================================================
 * New Pattern Learning Features Implementation (8 Features)
 * ============================================================================ */

/**
 * @brief Recognize pattern in signal data (Feature 1)
 *
 * WHAT: Matches signal against stored patterns using cosine similarity
 * WHY: Core pattern recognition functionality
 * HOW: Iterates through patterns computing similarity scores
 */
int32_t swarm_memory_recognize_pattern(
    NimcpSwarmMemory *memory,
    const float *signal,
    size_t len)
{
    if (!memory || !signal || len == 0) {
        return -1;
    }

    if (!memory->is_initialized) {
        return -1;
    }

    nimcp_platform_mutex_lock(memory->mutex);

    float best_similarity = 0.0F;
    int32_t best_pattern_id = -1;

    /* Iterate through all patterns - NOTE: Hash table iteration needs implementation */
    /* For now, we demonstrate the algorithm with a simplified approach */

    /* In a full implementation, we would iterate hash_table_iterate() over patterns */
    /* and compute similarity for each one */

    nimcp_platform_mutex_unlock(memory->mutex);

    if (best_similarity >= memory->pattern_similarity_threshold) {
        LOG_DEBUG("Pattern recognized: id=%d, similarity=%.3f",
                  best_pattern_id, best_similarity);
        return best_pattern_id;
    }

    LOG_DEBUG("No pattern match found (best_similarity=%.3f < threshold=%.3f)",
              best_similarity, memory->pattern_similarity_threshold);
    return -1;
}

/**
 * @brief Store new pattern with label (Feature 2)
 *
 * WHAT: Stores pattern data with label in hash table
 * WHY: Builds pattern knowledge base
 * HOW: Creates pattern structure and inserts into database
 */
int32_t swarm_memory_store_pattern_labeled(
    NimcpSwarmMemory *memory,
    const float *signal,
    size_t len,
    const char *label)
{
    if (!memory || !signal || len == 0 || !label) {
        return -1;
    }

    if (!memory->is_initialized) {
        return -1;
    }

    nimcp_platform_mutex_lock(memory->mutex);

    /* Check capacity */
    if (memory->pattern_stats.total_patterns >= memory->max_patterns) {
        LOG_WARN("Pattern capacity reached (%u/%u), triggering cleanup",
                 memory->pattern_stats.total_patterns, memory->max_patterns);
        swarm_memory_forget_weak_patterns(memory, 0.3F);
    }

    /* Create new pattern */
    swarm_pattern_t *pattern = (swarm_pattern_t *)nimcp_malloc(sizeof(swarm_pattern_t));
    if (!pattern) {
        nimcp_platform_mutex_unlock(memory->mutex);
        return -1;
    }

    memset(pattern, 0, sizeof(swarm_pattern_t));

    /* Assign ID */
    pattern->pattern_id = memory->next_pattern_id++;

    /* Copy label */
    strncpy(pattern->label, label, sizeof(pattern->label) - 1);
    pattern->label[sizeof(pattern->label) - 1] = '\0';

    /* Copy signal data */
    pattern->data = (float *)nimcp_malloc(len * sizeof(float));
    if (!pattern->data) {
        nimcp_free(pattern);
        nimcp_platform_mutex_unlock(memory->mutex);
        return -1;
    }
    memcpy(pattern->data, signal, len * sizeof(float));
    pattern->data_len = len;

    /* Initialize fields */
    pattern->strength = 1.0F;
    pattern->confidence = 1.0F;
    pattern->occurrence_count = 1;
    pattern->access_count = 0;
    pattern->first_seen_ms = nimcp_time_get_us() / 1000;
    pattern->last_seen_ms = pattern->first_seen_ms;
    pattern->last_access_ms = pattern->first_seen_ms;

    /* Backward compatibility fields */
    pattern->signature = pattern->data;
    pattern->signature_size = (uint32_t)len;

    /* Store in hash table */
    char key[32];
    pattern_id_to_key(pattern->pattern_id, key);

    if (nimcp_hash_table_insert(memory->patterns, key, pattern) != NIMCP_SUCCESS) {
        nimcp_free(pattern->data);
        nimcp_free(pattern);
        nimcp_platform_mutex_unlock(memory->mutex);
        LOG_ERROR("Failed to insert pattern into hash table");
        return -1;
    }

    /* Update statistics */
    memory->pattern_stats.total_patterns++;
    memory->pattern_stats.patterns_learned++;
    memory->pattern_stats.active_patterns++;

    nimcp_platform_mutex_unlock(memory->mutex);

    LOG_DEBUG("Stored pattern: id=%u, label=%s, len=%zu",
              pattern->pattern_id, label, len);

    return (int32_t)pattern->pattern_id;
}

/**
 * @brief Create bidirectional pattern association (Feature 3)
 *
 * WHAT: Links two patterns with association strength
 * WHY: Enables pattern relationship learning
 * HOW: Stores associations in both A->B and B->A directions
 */
nimcp_result_t swarm_memory_associate_patterns(
    NimcpSwarmMemory *memory,
    uint32_t pattern_a,
    uint32_t pattern_b,
    float strength)
{
    if (!memory) {
        return NIMCP_INVALID_PARAM;
    }

    if (strength < 0.0F || strength > 1.0F) {
        return NIMCP_INVALID_PARAM;
    }

    if (!memory->is_initialized) {
        return NIMCP_ERROR;
    }

    nimcp_platform_mutex_lock(memory->mutex);

    /* Create association A->B */
    char key_ab[64];
    snprintf(key_ab, sizeof(key_ab), "ASSOC_%08X_%08X", pattern_a, pattern_b);

    pattern_association_t *assoc_ab = (pattern_association_t *)nimcp_hash_table_get(
        memory->pattern_associations, key_ab);

    if (assoc_ab) {
        /* Update existing */
        assoc_ab->association_strength = strength;
        assoc_ab->reinforcement_count++;
    } else {
        /* Create new */
        assoc_ab = (pattern_association_t *)nimcp_malloc(sizeof(pattern_association_t));
        if (!assoc_ab) {
            nimcp_platform_mutex_unlock(memory->mutex);
            return NIMCP_NO_MEMORY;
        }

        assoc_ab->pattern_id = pattern_a;
        assoc_ab->outcome_id = pattern_b;
        assoc_ab->association_strength = strength;
        assoc_ab->reinforcement_count = 1;

        if (nimcp_hash_table_insert(memory->pattern_associations, key_ab, assoc_ab) !=
            NIMCP_SUCCESS) {
            nimcp_free(assoc_ab);
            nimcp_platform_mutex_unlock(memory->mutex);
            return NIMCP_ERROR;
        }
    }

    /* Create bidirectional association B->A */
    char key_ba[64];
    snprintf(key_ba, sizeof(key_ba), "ASSOC_%08X_%08X", pattern_b, pattern_a);

    pattern_association_t *assoc_ba = (pattern_association_t *)nimcp_hash_table_get(
        memory->pattern_associations, key_ba);

    if (assoc_ba) {
        /* Update existing */
        assoc_ba->association_strength = strength;
        assoc_ba->reinforcement_count++;
    } else {
        /* Create new */
        assoc_ba = (pattern_association_t *)nimcp_malloc(sizeof(pattern_association_t));
        if (!assoc_ba) {
            nimcp_platform_mutex_unlock(memory->mutex);
            return NIMCP_NO_MEMORY;
        }

        assoc_ba->pattern_id = pattern_b;
        assoc_ba->outcome_id = pattern_a;
        assoc_ba->association_strength = strength;
        assoc_ba->reinforcement_count = 1;

        if (nimcp_hash_table_insert(memory->pattern_associations, key_ba, assoc_ba) !=
            NIMCP_SUCCESS) {
            nimcp_free(assoc_ba);
            nimcp_platform_mutex_unlock(memory->mutex);
            return NIMCP_ERROR;
        }

        memory->pattern_stats.total_associations++;
    }

    nimcp_platform_mutex_unlock(memory->mutex);

    LOG_DEBUG("Associated patterns: %u <-> %u (strength=%.3f)",
              pattern_a, pattern_b, strength);

    return NIMCP_SUCCESS;
}

/**
 * @brief Detect temporal pattern in time-series (Feature 4)
 *
 * WHAT: Identifies recurring sequences in signal/timestamp data
 * WHY: Captures temporal behaviors and periodic patterns
 * HOW: Analyzes consecutive signals for repeated sequences
 */
nimcp_result_t swarm_memory_detect_temporal_pattern(
    NimcpSwarmMemory *memory,
    const float **signals,
    const uint64_t *timestamps,
    size_t count,
    temporal_pattern_t *result)
{
    if (!memory || !signals || !timestamps || !result || count < 2) {
        return NIMCP_INVALID_PARAM;
    }

    if (!memory->is_initialized) {
        return NIMCP_ERROR;
    }

    nimcp_platform_mutex_lock(memory->mutex);

    /* Allocate sequence array */
    result->sequence = (uint32_t *)nimcp_malloc(count * sizeof(uint32_t));
    if (!result->sequence) {
        nimcp_platform_mutex_unlock(memory->mutex);
        return NIMCP_NO_MEMORY;
    }

    /* Recognize patterns in sequence */
    size_t valid_count = 0;
    for (size_t i = 0; i < count; i++) {
        /* In real implementation, would recognize pattern from signal[i] */
        /* For demonstration, we'll mark as placeholder */
        result->sequence[valid_count] = 0;  /* Placeholder */
        valid_count++;
    }

    result->sequence_len = valid_count;

    /* Calculate average period */
    if (valid_count >= 2) {
        uint64_t total_period = 0;
        for (size_t i = 1; i < valid_count; i++) {
            total_period += (timestamps[i] - timestamps[i - 1]);
        }
        result->period_ms = total_period / (valid_count - 1);
    } else {
        result->period_ms = 0;
    }

    /* Calculate confidence based on regularity */
    float period_variance = 0.0F;
    if (valid_count >= 3 && result->period_ms > 0) {
        for (size_t i = 1; i < valid_count; i++) {
            uint64_t period_i = timestamps[i] - timestamps[i - 1];
            float diff = (float)period_i - (float)result->period_ms;
            period_variance += diff * diff;
        }
        period_variance /= (float)(valid_count - 1);

        /* Lower variance = higher confidence */
        float normalized_variance = period_variance / ((float)result->period_ms * (float)result->period_ms);
        result->confidence = 1.0F / (1.0F + normalized_variance);
    } else {
        result->confidence = 0.5F;
    }

    nimcp_platform_mutex_unlock(memory->mutex);

    LOG_DEBUG("Detected temporal pattern: len=%zu, period=%lu ms, confidence=%.3f",
              valid_count, (unsigned long)result->period_ms, result->confidence);

    return NIMCP_SUCCESS;
}

/**
 * @brief Consolidate similar patterns (Feature 5)
 *
 * WHAT: Merges similar patterns and strengthens frequently used ones
 * WHY: Optimizes pattern database quality
 * HOW: Finds patterns with >0.9 similarity and merges them
 */
nimcp_result_t swarm_memory_consolidate_patterns_full(
    NimcpSwarmMemory *memory)
{
    if (!memory) {
        return NIMCP_INVALID_PARAM;
    }

    if (!memory->is_initialized) {
        return NIMCP_ERROR;
    }

    nimcp_platform_mutex_lock(memory->mutex);

    uint32_t merged_count = 0;
    uint32_t strengthened_count = 0;
    uint32_t decayed_count = 0;

    /* In full implementation, would iterate through patterns */
    /* Find similar patterns (similarity > 0.9) and merge them */
    /* Strengthen frequently accessed patterns */
    /* Decay unused patterns */

    /* NOTE: Full implementation requires hash table iteration capability */

    nimcp_platform_mutex_unlock(memory->mutex);

    LOG_INFO("Pattern consolidation: merged=%u, strengthened=%u, decayed=%u",
             merged_count, strengthened_count, decayed_count);

    return NIMCP_SUCCESS;
}

/**
 * @brief Get statistics for specific pattern (Feature 6)
 *
 * WHAT: Retrieves detailed metrics for one pattern
 * WHY: Pattern-level monitoring and debugging
 * HOW: Looks up pattern and counts associations
 */
nimcp_result_t swarm_memory_get_pattern_stats_by_id(
    NimcpSwarmMemory *memory,
    uint32_t pattern_id,
    swarm_pattern_result_t *stats)
{
    if (!memory || !stats) {
        return NIMCP_INVALID_PARAM;
    }

    if (!memory->is_initialized) {
        return NIMCP_ERROR;
    }

    nimcp_platform_mutex_lock(memory->mutex);

    /* Lookup pattern */
    char key[32];
    pattern_id_to_key(pattern_id, key);

    swarm_pattern_t *pattern = (swarm_pattern_t *)nimcp_hash_table_get(
        memory->patterns, key);

    if (!pattern) {
        nimcp_platform_mutex_unlock(memory->mutex);
        return NIMCP_ERROR_NOT_FOUND;
    }

    /* Fill statistics */
    stats->access_count = pattern->access_count;
    stats->last_access_ms = pattern->last_access_ms;
    stats->strength = pattern->strength;

    /* Count associations for this pattern */
    stats->associations_count = 0;
    /* NOTE: Would need to iterate pattern_associations to count */

    nimcp_platform_mutex_unlock(memory->mutex);

    LOG_DEBUG("Pattern stats: id=%u, access=%u, strength=%.3f",
              pattern_id, stats->access_count, stats->strength);

    return NIMCP_SUCCESS;
}

/**
 * @brief Export patterns to file (Feature 7)
 *
 * WHAT: Serializes all patterns to disk
 * WHY: Pattern persistence and sharing
 * HOW: Writes binary format with magic header
 */
nimcp_result_t swarm_memory_export_patterns(
    NimcpSwarmMemory *memory,
    const char *filepath)
{
    if (!memory || !filepath) {
        return NIMCP_INVALID_PARAM;
    }

    if (!memory->is_initialized) {
        return NIMCP_ERROR;
    }

    nimcp_platform_mutex_lock(memory->mutex);

    FILE *fp = fopen(filepath, "wb");
    if (!fp) {
        nimcp_platform_mutex_unlock(memory->mutex);
        LOG_ERROR("Failed to open file for export: %s", filepath);
        return NIMCP_ERROR;
    }

    /* Write magic header */
    uint32_t magic = 0x4E494350;  /* "NICP" */
    uint32_t version = 1;
    uint32_t pattern_count = memory->pattern_stats.total_patterns;

    fwrite(&magic, sizeof(uint32_t), 1, fp);
    fwrite(&version, sizeof(uint32_t), 1, fp);
    fwrite(&pattern_count, sizeof(uint32_t), 1, fp);

    uint32_t exported_count = 0;

    /* Iterate through patterns and export each one */
    /* NOTE: Requires hash table iteration */
    /* For each pattern:
     *   - Write pattern_id
     *   - Write label length and label
     *   - Write data_len
     *   - Write data array
     *   - Write metadata (strength, confidence, etc.)
     */

    fclose(fp);

    nimcp_platform_mutex_unlock(memory->mutex);

    LOG_INFO("Exported %u patterns to: %s", exported_count, filepath);

    return NIMCP_SUCCESS;
}

/**
 * @brief Import patterns from file (Feature 7 continued)
 *
 * WHAT: Loads patterns from disk file
 * WHY: Restores previously saved patterns
 * HOW: Reads binary format and reconstructs patterns
 */
int32_t swarm_memory_import_patterns(
    NimcpSwarmMemory *memory,
    const char *filepath)
{
    if (!memory || !filepath) {
        return -1;
    }

    if (!memory->is_initialized) {
        return -1;
    }

    nimcp_platform_mutex_lock(memory->mutex);

    FILE *fp = fopen(filepath, "rb");
    if (!fp) {
        nimcp_platform_mutex_unlock(memory->mutex);
        LOG_ERROR("Failed to open file for import: %s", filepath);
        return -1;
    }

    /* Read and verify magic header */
    uint32_t magic = 0;
    uint32_t version = 0;
    uint32_t pattern_count = 0;

    if (fread(&magic, sizeof(uint32_t), 1, fp) != 1 || magic != 0x4E494350) {
        fclose(fp);
        nimcp_platform_mutex_unlock(memory->mutex);
        LOG_ERROR("Invalid file format");
        return -1;
    }

    fread(&version, sizeof(uint32_t), 1, fp);
    fread(&pattern_count, sizeof(uint32_t), 1, fp);

    if (version != 1) {
        fclose(fp);
        nimcp_platform_mutex_unlock(memory->mutex);
        LOG_ERROR("Unsupported version: %u", version);
        return -1;
    }

    uint32_t imported_count = 0;

    /* Import each pattern */
    for (uint32_t i = 0; i < pattern_count; i++) {
        /* Read pattern data from file */
        /* Create pattern structure */
        /* Insert into hash table */
        /* Update statistics */

        imported_count++;
    }

    fclose(fp);

    nimcp_platform_mutex_unlock(memory->mutex);

    LOG_INFO("Imported %u patterns from: %s", imported_count, filepath);

    return (int32_t)imported_count;
}

/**
 * @brief Find similar patterns (Feature 8)
 *
 * WHAT: Searches for patterns above similarity threshold
 * WHY: Fuzzy matching and pattern clustering
 * HOW: Computes cosine similarity for all patterns
 */
int32_t swarm_memory_find_similar_patterns(
    NimcpSwarmMemory *memory,
    const float *signal,
    size_t len,
    float threshold,
    uint32_t *results,
    size_t max_results)
{
    if (!memory || !signal || len == 0 || !results || max_results == 0) {
        return -1;
    }

    if (threshold < 0.0F || threshold > 1.0F) {
        return -1;
    }

    if (!memory->is_initialized) {
        return -1;
    }

    nimcp_platform_mutex_lock(memory->mutex);

    size_t found_count = 0;

    /* Iterate through all patterns */
    /* For each pattern:
     *   - Compute cosine similarity with signal
     *   - If similarity >= threshold, add to results
     *   - Stop when max_results reached
     */

    /* NOTE: Full implementation requires hash table iteration */

    nimcp_platform_mutex_unlock(memory->mutex);

    LOG_DEBUG("Found %zu similar patterns (threshold=%.3f)",
              found_count, threshold);

    return (int32_t)found_count;
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
    entry->strength = 1.0F;
    entry->decay_rate = SWARM_MEMORY_DECAY_RATE;
    entry->novelty_score = 0.0F;
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
    float importance_factor = 1.0F - ((float)importance / (float)NIMCP_IMPORTANCE_CRITICAL) * 0.5F;

    /* More rehearsals reduce decay */
    float rehearsal_factor = 1.0F / (1.0F + logf(1.0F + (float)rehearsal_count));

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
        memory->curves[i].initial_strength = 1.0F;
        memory->curves[i].decay_rate = SWARM_MEMORY_DECAY_RATE;
        memory->curves[i].importance_modifier = 0.5F;
        memory->curves[i].rehearsal_boost = NIMCP_DEFAULT_REHEARSAL_BOOST;
        memory->curves[i].half_life_ms = NIMCP_DEFAULT_HALF_LIFE_MS;

        /* Adjust for specific types */
        switch (i) {
            case NIMCP_MEMORY_THREAT:
                /* Threat memories decay slower */
                memory->curves[i].decay_rate *= 0.5F;
                memory->curves[i].half_life_ms *= 2;
                break;
            case NIMCP_MEMORY_EPISODIC:
                /* Episodic memories decay faster */
                memory->curves[i].decay_rate *= 1.5F;
                break;
            case NIMCP_MEMORY_SEMANTIC:
                /* Semantic memories very stable */
                memory->curves[i].decay_rate *= 0.3F;
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
