/**
 * @file nimcp_swarm_memory.h
 * @brief Swarm Memory Consolidation System for NIMCP
 *
 * Biological Inspiration: Sleep-based memory consolidation in mammals
 *
 * This system implements distributed memory consolidation across the swarm,
 * inspired by how mammalian brains consolidate memories during sleep through
 * hippocampal replay and neocortical integration.
 *
 * Key Features:
 * - Multiple memory types (episodic, semantic, procedural, threat, spatial)
 * - Experience replay during low-activity periods
 * - Knowledge distillation and compression
 * - Time-based forgetting curves with importance weighting
 * - Periodic consolidation windows
 * - Distributed hippocampus architecture
 * - Semantic compression and pattern abstraction
 * - Full bio-async integration
 *
 * @version 1.0
 * @date 2025
 */

#ifndef NIMCP_SWARM_MEMORY_H
#define NIMCP_SWARM_MEMORY_H

#include "core/brain/nimcp_brain.h"
#include "async/nimcp_bio_messages.h"
#include "utils/time/nimcp_time.h"
#include "utils/containers/nimcp_hash_table.h"
#include "utils/containers/nimcp_min_heap.h"
#include <stdint.h>
#include <stdbool.h>

#include "async/nimcp_bio_router.h"
#include "utils/thread/nimcp_thread.h"
#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Memory Type Definitions
 * ============================================================================ */

/**
 * @brief Types of memories that can be consolidated
 */
typedef enum {
    NIMCP_MEMORY_EPISODIC,    /**< Specific events/experiences */
    NIMCP_MEMORY_SEMANTIC,    /**< General knowledge/patterns */
    NIMCP_MEMORY_PROCEDURAL,  /**< Learned behaviors */
    NIMCP_MEMORY_THREAT,      /**< Threat patterns */
    NIMCP_MEMORY_SPATIAL,     /**< Location/map data */
    NIMCP_MEMORY_TYPE_COUNT
} NimcpMemoryType;

/**
 * @brief Consolidation mode
 */
typedef enum {
    NIMCP_CONSOLIDATION_ACTIVE,   /**< Active consolidation (high priority) */
    NIMCP_CONSOLIDATION_PASSIVE,  /**< Passive consolidation (background) */
    NIMCP_CONSOLIDATION_SLEEP     /**< Sleep-like consolidation (offline) */
} NimcpConsolidationMode;

/**
 * @brief Memory importance level
 */
typedef enum {
    NIMCP_IMPORTANCE_LOW,
    NIMCP_IMPORTANCE_MEDIUM,
    NIMCP_IMPORTANCE_HIGH,
    NIMCP_IMPORTANCE_CRITICAL
} NimcpMemoryImportance;

/* ============================================================================
 * Memory Structure Definitions
 * ============================================================================ */

/**
 * @brief Individual memory entry
 */
typedef struct {
    char id[64];                      /**< Unique memory identifier */
    NimcpMemoryType type;             /**< Type of memory */
    NimcpMemoryImportance importance; /**< Importance level */

    void *data;                       /**< Memory data payload */
    size_t data_size;                 /**< Size of data in bytes */

    uint64_t created_at;              /**< Creation timestamp */
    uint64_t last_accessed;           /**< Last access timestamp */
    uint64_t last_rehearsed;          /**< Last rehearsal timestamp */

    uint32_t access_count;            /**< Number of accesses */
    uint32_t rehearsal_count;         /**< Number of rehearsals */

    float strength;                   /**< Memory strength (0.0-1.0) */
    float decay_rate;                 /**< Decay rate per time unit */
    float novelty_score;              /**< Novelty score (0.0-1.0) */

    bool is_compressed;               /**< Whether data is compressed */
    bool is_consolidated;             /**< Whether memory is consolidated */
    bool is_distributed;              /**< Whether memory is distributed to swarm */

    uint32_t replica_count;           /**< Number of replicas in swarm */
    char source_node[64];             /**< Node that created this memory */
} NimcpMemoryEntry;

/**
 * @brief Compressed memory for efficient transmission
 */
typedef struct {
    char id[64];                      /**< Memory identifier */
    NimcpMemoryType type;             /**< Memory type */
    NimcpMemoryImportance importance; /**< Importance level */

    void *compressed_data;            /**< Compressed data */
    size_t compressed_size;           /**< Compressed data size */
    size_t original_size;             /**< Original uncompressed size */

    float compression_ratio;          /**< Compression ratio achieved */
    uint32_t pattern_hash;            /**< Hash of extracted pattern */
} NimcpCompressedMemory;

/**
 * @brief Experience replay entry
 */
typedef struct {
    NimcpMemoryEntry *memory;         /**< Memory to replay */
    float replay_priority;            /**< Priority for replay */
    uint32_t replay_count;            /**< Number of times replayed */
    uint64_t next_replay_time;        /**< Next scheduled replay time */
} NimcpReplayEntry;

/**
 * @brief Forgetting curve parameters
 */
typedef struct {
    float initial_strength;           /**< Initial memory strength */
    float decay_rate;                 /**< Base decay rate */
    float importance_modifier;        /**< Importance-based modifier */
    float rehearsal_boost;            /**< Strength boost per rehearsal */
    uint64_t half_life_ms;            /**< Half-life in milliseconds */
} NimcpForgettingCurve;

/**
 * @brief Consolidation window configuration
 */
typedef struct {
    NimcpConsolidationMode mode;      /**< Consolidation mode */
    uint64_t window_start;            /**< Window start time */
    uint64_t window_duration_ms;      /**< Window duration */
    uint32_t max_memories_per_window; /**< Max memories to consolidate */
    float activity_threshold;         /**< Activity threshold for triggering */
    bool auto_schedule;               /**< Automatic scheduling enabled */
} NimcpConsolidationWindow;

/**
 * @brief Distributed hippocampus node
 */
typedef struct {
    char node_id[64];                 /**< Node identifier */
    bool is_active;                   /**< Whether node is active */
    uint32_t memory_count;            /**< Number of memories stored */
    uint64_t last_sync_time;          /**< Last synchronization time */
    float health_score;               /**< Node health (0.0-1.0) */
    uint32_t replica_capacity;        /**< Max replicas this node can hold */
} NimcpHippocampusNode;

/**
 * @brief Semantic compression context
 */
typedef struct {
    uint32_t pattern_count;           /**< Number of patterns extracted */
    void *pattern_tree;               /**< Hierarchical pattern tree */
    NimcpHashTable *pattern_index;    /**< Pattern lookup index */
    float compression_target;         /**< Target compression ratio */
    uint32_t abstraction_level;       /**< Level of abstraction */
} NimcpSemanticCompression;

/**
 * @brief Memory statistics
 */
typedef struct {
    uint64_t total_memories;          /**< Total memories stored */
    uint64_t consolidated_memories;   /**< Consolidated memories */
    uint64_t distributed_memories;    /**< Distributed memories */
    uint64_t forgotten_memories;      /**< Forgotten memories */

    uint64_t total_replays;           /**< Total replay operations */
    uint64_t successful_replays;      /**< Successful replays */

    float avg_compression_ratio;      /**< Average compression ratio */
    float avg_memory_strength;        /**< Average memory strength */

    uint64_t bytes_transmitted;       /**< Bytes transmitted to swarm */
    uint64_t bytes_received;          /**< Bytes received from swarm */

    uint32_t active_nodes;            /**< Active hippocampus nodes */
    uint32_t total_replicas;          /**< Total memory replicas */
} NimcpMemoryStatistics;

/**
 * @brief Main swarm memory consolidation system
 */
typedef struct {
    /* Memory storage */
    NimcpHashTable *memories;         /**< All memories indexed by ID */
    NimcpHashTable *memories_by_type[NIMCP_MEMORY_TYPE_COUNT]; /**< By type */
    NimcpMinHeap *replay_queue;       /**< Priority queue for replay */

    /* Forgetting curves */
    NimcpForgettingCurve curves[NIMCP_MEMORY_TYPE_COUNT]; /**< Per-type curves */

    /* Consolidation */
    NimcpConsolidationWindow window;  /**< Current consolidation window */
    uint64_t last_consolidation;      /**< Last consolidation time */
    uint32_t consolidation_count;     /**< Total consolidations performed */

    /* Distributed hippocampus */
    NimcpHashTable *hippocampus_nodes; /**< Active hippocampus nodes */
    uint32_t replication_factor;      /**< Target replication factor */
    float consensus_threshold;        /**< Consensus threshold (0.0-1.0) */

    /* Semantic compression */
    NimcpSemanticCompression compression; /**< Compression context */

    /* Bio-async integration */
    NimcpBioContext *bio_ctx;         /**< Bio-async context */
    bool bio_async_enabled;           /**< Bio-async enabled flag */

    /* Statistics */
    NimcpMemoryStatistics stats;      /**< System statistics */

    /* Configuration */
    float novelty_threshold;          /**< Threshold for novelty detection */
    float replay_probability;         /**< Probability of replay per cycle */
    uint32_t max_memory_capacity;     /**< Max memories to store */
    bool auto_compression;            /**< Automatic compression enabled */
    bool auto_distribution;           /**< Automatic distribution enabled */

    /* Synchronization */
    nimcp_mutex_t mutex;                 /**< Thread-safety mutex */
    bool is_initialized;              /**< Initialization flag */
} NimcpSwarmMemory;

/* ============================================================================
 * Core API Functions
 * ============================================================================ */

/**
 * @brief Create a new swarm memory system
 *
 * @param max_capacity Maximum memory capacity
 * @param replication_factor Target replication factor
 * @return Pointer to created system, or NULL on failure
 */
NimcpSwarmMemory *nimcp_swarm_memory_create(
    uint32_t max_capacity,
    uint32_t replication_factor
);

/**
 * @brief Destroy swarm memory system and free resources
 *
 * @param memory Swarm memory system to destroy
 */
void nimcp_swarm_memory_destroy(NimcpSwarmMemory *memory);

/**
 * @brief Initialize swarm memory system
 *
 * @param memory Swarm memory system
 * @param bio_ctx Bio-async context (optional, can be NULL)
 * @return NIMCP_OK on success, error code otherwise
 */
nimcp_result_t nimcp_swarm_memory_init(
    NimcpSwarmMemory *memory,
    NimcpBioContext *bio_ctx
);

/* ============================================================================
 * Memory Management API
 * ============================================================================ */

/**
 * @brief Store a new memory
 *
 * @param memory Swarm memory system
 * @param type Memory type
 * @param importance Memory importance
 * @param data Memory data
 * @param data_size Size of data
 * @param memory_id Output: assigned memory ID
 * @return NIMCP_OK on success, error code otherwise
 */
nimcp_result_t nimcp_swarm_memory_store(
    NimcpSwarmMemory *memory,
    NimcpMemoryType type,
    NimcpMemoryImportance importance,
    const void *data,
    size_t data_size,
    char *memory_id
);

/**
 * @brief Retrieve a memory by ID
 *
 * @param memory Swarm memory system
 * @param memory_id Memory identifier
 * @param out_data Output buffer for data
 * @param data_size Size of output buffer
 * @return NIMCP_OK on success, error code otherwise
 */
nimcp_result_t nimcp_swarm_memory_retrieve(
    NimcpSwarmMemory *memory,
    const char *memory_id,
    void *out_data,
    size_t data_size
);

/**
 * @brief Update memory access tracking
 *
 * @param memory Swarm memory system
 * @param memory_id Memory identifier
 * @return NIMCP_OK on success, error code otherwise
 */
nimcp_result_t nimcp_swarm_memory_access(
    NimcpSwarmMemory *memory,
    const char *memory_id
);

/**
 * @brief Rehearse a memory to strengthen it
 *
 * @param memory Swarm memory system
 * @param memory_id Memory identifier
 * @return NIMCP_OK on success, error code otherwise
 */
nimcp_result_t nimcp_swarm_memory_rehearse(
    NimcpSwarmMemory *memory,
    const char *memory_id
);

/**
 * @brief Delete a memory
 *
 * @param memory Swarm memory system
 * @param memory_id Memory identifier
 * @return NIMCP_OK on success, error code otherwise
 */
nimcp_result_t nimcp_swarm_memory_delete(
    NimcpSwarmMemory *memory,
    const char *memory_id
);

/* ============================================================================
 * Experience Replay API
 * ============================================================================ */

/**
 * @brief Schedule a memory for replay
 *
 * @param memory Swarm memory system
 * @param memory_id Memory identifier
 * @param priority Replay priority
 * @return NIMCP_OK on success, error code otherwise
 */
nimcp_result_t nimcp_swarm_memory_schedule_replay(
    NimcpSwarmMemory *memory,
    const char *memory_id,
    float priority
);

/**
 * @brief Execute replay cycle
 *
 * Replays high-priority memories to strengthen them and share with swarm
 *
 * @param memory Swarm memory system
 * @param max_replays Maximum number of replays to perform
 * @param replays_performed Output: number of replays performed
 * @return NIMCP_OK on success, error code otherwise
 */
nimcp_result_t nimcp_swarm_memory_replay_cycle(
    NimcpSwarmMemory *memory,
    uint32_t max_replays,
    uint32_t *replays_performed
);

/**
 * @brief Calculate replay priority for a memory
 *
 * Based on novelty, importance, and recency
 *
 * @param memory Swarm memory system
 * @param memory_entry Memory entry
 * @return Replay priority (0.0-1.0)
 */
float nimcp_swarm_memory_calculate_replay_priority(
    NimcpSwarmMemory *memory,
    const NimcpMemoryEntry *memory_entry
);

/* ============================================================================
 * Knowledge Distillation API
 * ============================================================================ */

/**
 * @brief Compress a memory for efficient transmission
 *
 * @param memory Swarm memory system
 * @param memory_id Memory identifier
 * @param compressed Output: compressed memory
 * @return NIMCP_OK on success, error code otherwise
 */
nimcp_result_t nimcp_swarm_memory_compress(
    NimcpSwarmMemory *memory,
    const char *memory_id,
    NimcpCompressedMemory *compressed
);

/**
 * @brief Decompress a received memory
 *
 * @param memory Swarm memory system
 * @param compressed Compressed memory
 * @param decompressed Output buffer
 * @param buffer_size Size of output buffer
 * @return NIMCP_OK on success, error code otherwise
 */
nimcp_result_t nimcp_swarm_memory_decompress(
    NimcpSwarmMemory *memory,
    const NimcpCompressedMemory *compressed,
    void *decompressed,
    size_t buffer_size
);

/**
 * @brief Extract essential patterns from a memory
 *
 * @param memory Swarm memory system
 * @param memory_id Memory identifier
 * @param pattern_hash Output: hash of extracted pattern
 * @return NIMCP_OK on success, error code otherwise
 */
nimcp_result_t nimcp_swarm_memory_extract_pattern(
    NimcpSwarmMemory *memory,
    const char *memory_id,
    uint32_t *pattern_hash
);

/* ============================================================================
 * Forgetting Curve API
 * ============================================================================ */

/**
 * @brief Update forgetting curve parameters for a memory type
 *
 * @param memory Swarm memory system
 * @param type Memory type
 * @param curve Forgetting curve parameters
 * @return NIMCP_OK on success, error code otherwise
 */
nimcp_result_t nimcp_swarm_memory_set_forgetting_curve(
    NimcpSwarmMemory *memory,
    NimcpMemoryType type,
    const NimcpForgettingCurve *curve
);

/**
 * @brief Calculate current memory strength based on forgetting curve
 *
 * @param memory Swarm memory system
 * @param memory_entry Memory entry
 * @param current_time Current timestamp
 * @return Current memory strength (0.0-1.0)
 */
float nimcp_swarm_memory_calculate_strength(
    NimcpSwarmMemory *memory,
    const NimcpMemoryEntry *memory_entry,
    uint64_t current_time
);

/**
 * @brief Apply forgetting to all memories
 *
 * Updates memory strengths and removes forgotten memories
 *
 * @param memory Swarm memory system
 * @param forgotten_count Output: number of memories forgotten
 * @return NIMCP_OK on success, error code otherwise
 */
nimcp_result_t nimcp_swarm_memory_apply_forgetting(
    NimcpSwarmMemory *memory,
    uint32_t *forgotten_count
);

/* ============================================================================
 * Consolidation Window API
 * ============================================================================ */

/**
 * @brief Configure consolidation window
 *
 * @param memory Swarm memory system
 * @param window Consolidation window configuration
 * @return NIMCP_OK on success, error code otherwise
 */
nimcp_result_t nimcp_swarm_memory_configure_consolidation(
    NimcpSwarmMemory *memory,
    const NimcpConsolidationWindow *window
);

/**
 * @brief Start a consolidation window
 *
 * @param memory Swarm memory system
 * @param mode Consolidation mode
 * @return NIMCP_OK on success, error code otherwise
 */
nimcp_result_t nimcp_swarm_memory_start_consolidation(
    NimcpSwarmMemory *memory,
    NimcpConsolidationMode mode
);

/**
 * @brief Execute consolidation process
 *
 * Consolidates memories, applies forgetting, and distributes to swarm
 *
 * @param memory Swarm memory system
 * @param memories_consolidated Output: number of memories consolidated
 * @return NIMCP_OK on success, error code otherwise
 */
nimcp_result_t nimcp_swarm_memory_consolidate(
    NimcpSwarmMemory *memory,
    uint32_t *memories_consolidated
);

/**
 * @brief Check if consolidation window is active
 *
 * @param memory Swarm memory system
 * @return true if window is active, false otherwise
 */
bool nimcp_swarm_memory_is_consolidating(const NimcpSwarmMemory *memory);

/* ============================================================================
 * Distributed Hippocampus API
 * ============================================================================ */

/**
 * @brief Register a hippocampus node
 *
 * @param memory Swarm memory system
 * @param node_id Node identifier
 * @param capacity Node capacity
 * @return NIMCP_OK on success, error code otherwise
 */
nimcp_result_t nimcp_swarm_memory_register_node(
    NimcpSwarmMemory *memory,
    const char *node_id,
    uint32_t capacity
);

/**
 * @brief Unregister a hippocampus node
 *
 * @param memory Swarm memory system
 * @param node_id Node identifier
 * @return NIMCP_OK on success, error code otherwise
 */
nimcp_result_t nimcp_swarm_memory_unregister_node(
    NimcpSwarmMemory *memory,
    const char *node_id
);

/**
 * @brief Distribute a memory to hippocampus nodes
 *
 * @param memory Swarm memory system
 * @param memory_id Memory identifier
 * @param replicas_created Output: number of replicas created
 * @return NIMCP_OK on success, error code otherwise
 */
nimcp_result_t nimcp_swarm_memory_distribute(
    NimcpSwarmMemory *memory,
    const char *memory_id,
    uint32_t *replicas_created
);

/**
 * @brief Verify memory consensus across nodes
 *
 * @param memory Swarm memory system
 * @param memory_id Memory identifier
 * @param has_consensus Output: whether consensus was reached
 * @return NIMCP_OK on success, error code otherwise
 */
nimcp_result_t nimcp_swarm_memory_verify_consensus(
    NimcpSwarmMemory *memory,
    const char *memory_id,
    bool *has_consensus
);

/**
 * @brief Synchronize with a specific node
 *
 * @param memory Swarm memory system
 * @param node_id Node identifier
 * @param memories_synced Output: number of memories synchronized
 * @return NIMCP_OK on success, error code otherwise
 */
nimcp_result_t nimcp_swarm_memory_sync_with_node(
    NimcpSwarmMemory *memory,
    const char *node_id,
    uint32_t *memories_synced
);

/* ============================================================================
 * Semantic Compression API
 * ============================================================================ */

/**
 * @brief Abstract patterns from raw experiences
 *
 * @param memory Swarm memory system
 * @param memory_ids Array of memory IDs to abstract from
 * @param count Number of memory IDs
 * @param pattern_hash Output: hash of abstracted pattern
 * @return NIMCP_OK on success, error code otherwise
 */
nimcp_result_t nimcp_swarm_memory_abstract_pattern(
    NimcpSwarmMemory *memory,
    const char **memory_ids,
    uint32_t count,
    uint32_t *pattern_hash
);

/**
 * @brief Generalize specific memories into broader knowledge
 *
 * @param memory Swarm memory system
 * @param specific_ids Array of specific memory IDs
 * @param count Number of specific memories
 * @param generalized_id Output: ID of generalized memory
 * @return NIMCP_OK on success, error code otherwise
 */
nimcp_result_t nimcp_swarm_memory_generalize(
    NimcpSwarmMemory *memory,
    const char **specific_ids,
    uint32_t count,
    char *generalized_id
);

/**
 * @brief Build hierarchical knowledge structure
 *
 * @param memory Swarm memory system
 * @param levels Output: number of hierarchy levels created
 * @return NIMCP_OK on success, error code otherwise
 */
nimcp_result_t nimcp_swarm_memory_build_hierarchy(
    NimcpSwarmMemory *memory,
    uint32_t *levels
);

/* ============================================================================
 * Bio-Async Integration API
 * ============================================================================ */

/**
 * @brief Process incoming bio-async message
 *
 * @param memory Swarm memory system
 * @param msg Incoming message
 * @return NIMCP_OK on success, error code otherwise
 */
nimcp_result_t nimcp_swarm_memory_process_message(
    NimcpSwarmMemory *memory,
    const NimcpBioMessage *msg
);

/**
 * @brief Send memory to another node via bio-async
 *
 * @param memory Swarm memory system
 * @param memory_id Memory identifier
 * @param target_node Target node identifier
 * @return NIMCP_OK on success, error code otherwise
 */
nimcp_result_t nimcp_swarm_memory_send_memory(
    NimcpSwarmMemory *memory,
    const char *memory_id,
    const char *target_node
);

/**
 * @brief Request memory from another node
 *
 * @param memory Swarm memory system
 * @param memory_id Memory identifier
 * @param source_node Source node identifier
 * @return NIMCP_OK on success, error code otherwise
 */
nimcp_result_t nimcp_swarm_memory_request_memory(
    NimcpSwarmMemory *memory,
    const char *memory_id,
    const char *source_node
);

/**
 * @brief Broadcast consolidation signal to swarm
 *
 * @param memory Swarm memory system
 * @param mode Consolidation mode
 * @return NIMCP_OK on success, error code otherwise
 */
nimcp_result_t nimcp_swarm_memory_broadcast_consolidation(
    NimcpSwarmMemory *memory,
    NimcpConsolidationMode mode
);

/* ============================================================================
 * Statistics and Monitoring API
 * ============================================================================ */

/**
 * @brief Get system statistics
 *
 * @param memory Swarm memory system
 * @param stats Output: statistics structure
 * @return NIMCP_OK on success, error code otherwise
 */
nimcp_result_t nimcp_swarm_memory_get_statistics(
    const NimcpSwarmMemory *memory,
    NimcpMemoryStatistics *stats
);

/**
 * @brief Get memory count by type
 *
 * @param memory Swarm memory system
 * @param type Memory type
 * @return Number of memories of specified type
 */
uint32_t nimcp_swarm_memory_get_count_by_type(
    const NimcpSwarmMemory *memory,
    NimcpMemoryType type
);

/**
 * @brief Get system health score
 *
 * Based on node availability, replication factor, and consolidation status
 *
 * @param memory Swarm memory system
 * @return Health score (0.0-1.0)
 */
float nimcp_swarm_memory_get_health_score(const NimcpSwarmMemory *memory);

/**
 * @brief Print system status to log
 *
 * @param memory Swarm memory system
 * @param verbose Whether to print verbose information
 */
void nimcp_swarm_memory_print_status(
    const NimcpSwarmMemory *memory,
    bool verbose
);

/* ============================================================================
 * Utility Functions
 * ============================================================================ */

/**
 * @brief Get string name for memory type
 *
 * @param type Memory type
 * @return String name
 */
const char *nimcp_memory_type_to_string(NimcpMemoryType type);

/**
 * @brief Get string name for consolidation mode
 *
 * @param mode Consolidation mode
 * @return String name
 */
const char *nimcp_consolidation_mode_to_string(NimcpConsolidationMode mode);

/**
 * @brief Get string name for memory importance
 *
 * @param importance Memory importance
 * @return String name
 */
const char *nimcp_memory_importance_to_string(NimcpMemoryImportance importance);

#include "async/nimcp_bio_router.h"
#include "utils/thread/nimcp_thread.h"
#ifdef __cplusplus
}
#endif

#endif /* NIMCP_SWARM_MEMORY_H */
