//=============================================================================
// nimcp_z_ladder.c - Z-Ladder Memory Consolidation System Implementation
//=============================================================================
/**
 * @file nimcp_z_ladder.c
 * @brief Implementation of four-tier memory consolidation system
 *
 * WHAT: Full implementation of Z-Ladder memory management
 * WHY:  Enable biologically-inspired memory consolidation
 * HOW:  Manages nodes across tiers with decay, promotion, demotion, eviction
 *
 * IMPLEMENTATION NOTES:
 * - Thread-safe via internal mutex
 * - Hash table for O(1) node lookup
 * - Dynamic arrays for tier storage
 * - Circular buffer for consolidation events
 *
 * @author NIMCP Development Team
 * @date 2026-01-09
 * @version 1.0.0
 */

#include "cognitive/memory/core/nimcp_z_ladder.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <pthread.h>

//=============================================================================
// Internal Constants
//=============================================================================

/** Hash table bucket count (prime number for better distribution) */
#define HASH_BUCKETS 1021

/** Growth factor when tier storage needs expansion */
#define GROWTH_FACTOR 2

/** Small strength boost on access */
#define ACCESS_STRENGTH_BOOST 0.01f

/** Emotional boost multiplier */
#define EMOTIONAL_BOOST_FACTOR 0.15f

/** Sleep consolidation promotion threshold reduction */
#define SLEEP_PROMOTION_REDUCTION 0.2f

/** Number of sleep consolidation passes */
#define SLEEP_CONSOLIDATION_PASSES 3

//=============================================================================
// Internal Data Structures
//=============================================================================

/**
 * @brief Hash table entry for node lookup
 */
typedef struct z_hash_entry_struct {
    uint64_t node_id;                   /**< Node identifier */
    pr_memory_node_t* node;             /**< Node pointer */
    pr_memory_tier_t tier;              /**< Current tier */
    size_t tier_index;                  /**< Index in tier storage */
    struct z_hash_entry_struct* next;   /**< Chain for collision handling */
} z_hash_entry_t;

/**
 * @brief Tier storage structure
 */
typedef struct {
    pr_memory_node_t** nodes;           /**< Array of node pointers */
    size_t count;                       /**< Current count */
    size_t capacity;                    /**< Allocated capacity */
    z_tier_config_t config;             /**< Tier configuration */
} z_tier_storage_t;

/**
 * @brief Internal Z-Ladder structure
 */
struct z_ladder_struct {
    //-------------------------------------------------------------------------
    // Tier Storage
    //-------------------------------------------------------------------------
    z_tier_storage_t tiers[Z_LADDER_NUM_TIERS];

    //-------------------------------------------------------------------------
    // Hash Table for Fast Lookup
    //-------------------------------------------------------------------------
    z_hash_entry_t* hash_buckets[HASH_BUCKETS];
    size_t total_nodes;

    //-------------------------------------------------------------------------
    // Statistics
    //-------------------------------------------------------------------------
    uint64_t promotions[3];             /**< Z0->Z1, Z1->Z2, Z2->Z3 */
    uint64_t demotions[3];              /**< Z1->Z0, Z2->Z1, Z3->Z2 */
    uint64_t evictions[Z_LADDER_NUM_TIERS];
    uint64_t last_decay_time_ms;
    uint64_t last_consolidation_time_ms;
    uint64_t total_consolidations;

    //-------------------------------------------------------------------------
    // Consolidation Events
    //-------------------------------------------------------------------------
    z_consolidation_event_t* events;
    size_t event_count;
    size_t event_capacity;
    size_t event_head;                  /**< Circular buffer head */
    bool enable_events;

    //-------------------------------------------------------------------------
    // Callbacks
    //-------------------------------------------------------------------------
    z_promotion_callback_t promotion_callback;
    void* promotion_user_data;
    z_eviction_callback_t eviction_callback;
    void* eviction_user_data;
    bool enable_callbacks;

    //-------------------------------------------------------------------------
    // Node Manager
    //-------------------------------------------------------------------------
    pr_node_manager_t node_manager;

    //-------------------------------------------------------------------------
    // Thread Safety
    //-------------------------------------------------------------------------
    pthread_mutex_t mutex;
    bool mutex_initialized;
};

//=============================================================================
// Internal Helper Functions - Time
//=============================================================================

/**
 * @brief Get current time in milliseconds (monotonic)
 */
static uint64_t get_current_time_ms(void) {
    struct timespec ts;

#ifdef CLOCK_MONOTONIC
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0) {
        return (uint64_t)ts.tv_sec * 1000ULL + (uint64_t)ts.tv_nsec / 1000000ULL;
    }
#endif

    // Fallback to real time
    if (clock_gettime(CLOCK_REALTIME, &ts) == 0) {
        return (uint64_t)ts.tv_sec * 1000ULL + (uint64_t)ts.tv_nsec / 1000000ULL;
    }

    // Last resort
    return (uint64_t)time(NULL) * 1000ULL;
}

//=============================================================================
// Internal Helper Functions - Math
//=============================================================================

/**
 * @brief Clamp float to range [min, max]
 */
static inline float clampf(float x, float min_val, float max_val) {
    if (x < min_val) return min_val;
    if (x > max_val) return max_val;
    return x;
}

/**
 * @brief Absolute value of float
 */
static inline float absf(float x) {
    return (x < 0.0f) ? -x : x;
}

//=============================================================================
// Internal Helper Functions - Hash Table
//=============================================================================

/**
 * @brief Compute hash bucket index for node ID
 */
static inline size_t hash_node_id(uint64_t node_id) {
    // FNV-1a inspired hash
    uint64_t hash = 14695981039346656037ULL;
    hash ^= node_id;
    hash *= 1099511628211ULL;
    return (size_t)(hash % HASH_BUCKETS);
}

/**
 * @brief Find hash entry for node ID
 */
static z_hash_entry_t* hash_find(z_ladder_t ladder, uint64_t node_id) {
    size_t bucket = hash_node_id(node_id);
    z_hash_entry_t* entry = ladder->hash_buckets[bucket];

    while (entry) {
        if (entry->node_id == node_id) {
            return entry;
        }
        entry = entry->next;
    }

    return NULL;
}

/**
 * @brief Insert hash entry
 */
static bool hash_insert(z_ladder_t ladder, uint64_t node_id,
                        pr_memory_node_t* node, pr_memory_tier_t tier,
                        size_t tier_index) {
    // Check if already exists
    if (hash_find(ladder, node_id)) {
        return false;
    }

    // Create new entry
    z_hash_entry_t* entry = (z_hash_entry_t*)malloc(sizeof(z_hash_entry_t));
    if (!entry) {
        return false;
    }

    entry->node_id = node_id;
    entry->node = node;
    entry->tier = tier;
    entry->tier_index = tier_index;

    // Insert at head of bucket chain
    size_t bucket = hash_node_id(node_id);
    entry->next = ladder->hash_buckets[bucket];
    ladder->hash_buckets[bucket] = entry;

    return true;
}

/**
 * @brief Remove hash entry
 */
static bool hash_remove(z_ladder_t ladder, uint64_t node_id) {
    size_t bucket = hash_node_id(node_id);
    z_hash_entry_t** prev_ptr = &ladder->hash_buckets[bucket];
    z_hash_entry_t* entry = *prev_ptr;

    while (entry) {
        if (entry->node_id == node_id) {
            *prev_ptr = entry->next;
            free(entry);
            return true;
        }
        prev_ptr = &entry->next;
        entry = entry->next;
    }

    return false;
}

/**
 * @brief Update hash entry tier/index
 */
static bool hash_update(z_ladder_t ladder, uint64_t node_id,
                        pr_memory_tier_t tier, size_t tier_index) {
    z_hash_entry_t* entry = hash_find(ladder, node_id);
    if (!entry) {
        return false;
    }

    entry->tier = tier;
    entry->tier_index = tier_index;
    return true;
}

/**
 * @brief Clear all hash entries
 */
static void hash_clear(z_ladder_t ladder) {
    for (size_t i = 0; i < HASH_BUCKETS; i++) {
        z_hash_entry_t* entry = ladder->hash_buckets[i];
        while (entry) {
            z_hash_entry_t* next = entry->next;
            free(entry);
            entry = next;
        }
        ladder->hash_buckets[i] = NULL;
    }
}

//=============================================================================
// Internal Helper Functions - Tier Storage
//=============================================================================

/**
 * @brief Initialize tier storage
 */
static bool tier_storage_init(z_tier_storage_t* tier, const z_tier_config_t* config) {
    tier->config = *config;
    tier->count = 0;

    // Determine initial capacity
    size_t initial_cap = (config->capacity > 0) ? config->capacity : 64;
    if (initial_cap < 16) {
        initial_cap = 16;
    }

    tier->capacity = initial_cap;
    tier->nodes = (pr_memory_node_t**)calloc(tier->capacity, sizeof(pr_memory_node_t*));

    return tier->nodes != NULL;
}

/**
 * @brief Destroy tier storage (does not destroy nodes)
 */
static void tier_storage_destroy(z_tier_storage_t* tier) {
    if (tier->nodes) {
        free(tier->nodes);
        tier->nodes = NULL;
    }
    tier->count = 0;
    tier->capacity = 0;
}

/**
 * @brief Grow tier storage capacity
 */
static bool tier_storage_grow(z_tier_storage_t* tier) {
    size_t new_capacity = tier->capacity * GROWTH_FACTOR;

    // Check for unlimited capacity (Z3)
    if (tier->config.capacity > 0 && new_capacity > tier->config.capacity * 2) {
        new_capacity = tier->config.capacity * 2;
    }

    pr_memory_node_t** new_nodes = (pr_memory_node_t**)realloc(
        tier->nodes, new_capacity * sizeof(pr_memory_node_t*));

    if (!new_nodes) {
        return false;
    }

    tier->nodes = new_nodes;
    tier->capacity = new_capacity;

    return true;
}

/**
 * @brief Add node to tier storage
 * Returns index of inserted node, or SIZE_MAX on failure
 */
static size_t tier_storage_add(z_tier_storage_t* tier, pr_memory_node_t* node) {
    // Check capacity (0 = unlimited)
    if (tier->config.capacity > 0 && tier->count >= tier->config.capacity) {
        return SIZE_MAX;  // At capacity
    }

    // Grow if needed
    if (tier->count >= tier->capacity) {
        if (!tier_storage_grow(tier)) {
            return SIZE_MAX;
        }
    }

    size_t index = tier->count;
    tier->nodes[index] = node;
    tier->count++;

    return index;
}

/**
 * @brief Remove node from tier storage by index
 */
static bool tier_storage_remove(z_tier_storage_t* tier, size_t index) {
    if (index >= tier->count) {
        return false;
    }

    // Move last element to fill gap (swap-and-pop)
    if (index < tier->count - 1) {
        tier->nodes[index] = tier->nodes[tier->count - 1];
    }

    tier->nodes[tier->count - 1] = NULL;
    tier->count--;

    return true;
}

/**
 * @brief Find node index in tier storage
 */
static size_t tier_storage_find(z_tier_storage_t* tier, pr_memory_node_t* node) {
    for (size_t i = 0; i < tier->count; i++) {
        if (tier->nodes[i] == node) {
            return i;
        }
    }
    return SIZE_MAX;
}

/**
 * @brief Find node with minimum strength (for eviction)
 */
static size_t tier_storage_find_weakest(z_tier_storage_t* tier) {
    if (tier->count == 0) {
        return SIZE_MAX;
    }

    size_t min_index = 0;
    float min_strength = tier->nodes[0]->current_strength;

    for (size_t i = 1; i < tier->count; i++) {
        if (tier->nodes[i]->current_strength < min_strength) {
            min_strength = tier->nodes[i]->current_strength;
            min_index = i;
        }
    }

    return min_index;
}

/**
 * @brief Find oldest node (for LRU eviction)
 */
static size_t tier_storage_find_oldest(z_tier_storage_t* tier) {
    if (tier->count == 0) {
        return SIZE_MAX;
    }

    size_t oldest_index = 0;
    uint64_t oldest_time = tier->nodes[0]->created_time_ms;

    for (size_t i = 1; i < tier->count; i++) {
        if (tier->nodes[i]->created_time_ms < oldest_time) {
            oldest_time = tier->nodes[i]->created_time_ms;
            oldest_index = i;
        }
    }

    return oldest_index;
}

/**
 * @brief Find least recently used node
 */
static size_t tier_storage_find_lru(z_tier_storage_t* tier) {
    if (tier->count == 0) {
        return SIZE_MAX;
    }

    size_t lru_index = 0;
    uint64_t lru_time = tier->nodes[0]->last_accessed_ms;

    for (size_t i = 1; i < tier->count; i++) {
        if (tier->nodes[i]->last_accessed_ms < lru_time) {
            lru_time = tier->nodes[i]->last_accessed_ms;
            lru_index = i;
        }
    }

    return lru_index;
}

/**
 * @brief Find least frequently used node
 */
static size_t tier_storage_find_lfu(z_tier_storage_t* tier) {
    if (tier->count == 0) {
        return SIZE_MAX;
    }

    size_t lfu_index = 0;
    uint64_t lfu_count = atomic_load(&tier->nodes[0]->access_count);

    for (size_t i = 1; i < tier->count; i++) {
        uint64_t count = atomic_load(&tier->nodes[i]->access_count);
        if (count < lfu_count) {
            lfu_count = count;
            lfu_index = i;
        }
    }

    return lfu_index;
}

/**
 * @brief Find node to evict based on policy
 */
static size_t tier_storage_find_evict_target(z_tier_storage_t* tier) {
    switch (tier->config.eviction_policy) {
        case Z_EVICT_OLDEST:
            return tier_storage_find_oldest(tier);
        case Z_EVICT_LRU:
            return tier_storage_find_lru(tier);
        case Z_EVICT_LFU:
            return tier_storage_find_lfu(tier);
        case Z_EVICT_COMBINED: {
            // Combined scoring: lower is more likely to evict
            if (tier->count == 0) return SIZE_MAX;

            size_t best_index = 0;
            float best_score = FLT_MAX;
            uint64_t now = get_current_time_ms();

            for (size_t i = 0; i < tier->count; i++) {
                pr_memory_node_t* node = tier->nodes[i];

                // Score: strength + recency + frequency (all normalized 0-1)
                float strength_score = node->current_strength;

                float age_ms = (float)(now - node->last_accessed_ms);
                float recency_score = 1.0f / (1.0f + age_ms / 60000.0f);  // Half at 1 min

                uint64_t acc = atomic_load(&node->access_count);
                float freq_score = (float)acc / (float)(acc + 10);  // Saturates at ~10

                float score = 0.4f * strength_score +
                             0.3f * recency_score +
                             0.3f * freq_score;

                if (score < best_score) {
                    best_score = score;
                    best_index = i;
                }
            }

            return best_index;
        }
        case Z_EVICT_WEAKEST:
        default:
            return tier_storage_find_weakest(tier);
    }
}

//=============================================================================
// Internal Helper Functions - Events
//=============================================================================

/**
 * @brief Record a consolidation event
 */
static void record_event(z_ladder_t ladder, uint64_t node_id,
                        pr_memory_tier_t from_tier, pr_memory_tier_t to_tier,
                        float strength_before, float strength_after,
                        bool is_promotion) {
    if (!ladder->enable_events || !ladder->events) {
        return;
    }

    // Circular buffer insertion
    size_t index = ladder->event_head;
    ladder->event_head = (ladder->event_head + 1) % ladder->event_capacity;

    if (ladder->event_count < ladder->event_capacity) {
        ladder->event_count++;
    }

    z_consolidation_event_t* event = &ladder->events[index];
    event->node_id = node_id;
    event->from_tier = from_tier;
    event->to_tier = to_tier;
    event->strength_before = strength_before;
    event->strength_after = strength_after;
    event->timestamp_ms = get_current_time_ms();
    event->is_promotion = is_promotion;
}

//=============================================================================
// Internal Helper Functions - Unlocked Operations
//=============================================================================

/**
 * @brief Internal remove without locking (caller holds mutex)
 */
static z_ladder_error_t remove_node_unlocked(z_ladder_t ladder, uint64_t node_id,
                                             bool destroy_node) {
    z_hash_entry_t* entry = hash_find(ladder, node_id);
    if (!entry) {
        return Z_LADDER_ERROR_NOT_FOUND;
    }

    pr_memory_tier_t tier = entry->tier;
    pr_memory_node_t* node = entry->node;
    size_t index = entry->tier_index;

    // Remove from tier storage
    z_tier_storage_t* storage = &ladder->tiers[tier];

    if (index < storage->count && storage->nodes[index] == node) {
        // Move last element to fill gap
        if (index < storage->count - 1) {
            storage->nodes[index] = storage->nodes[storage->count - 1];
            // Update hash entry for moved node
            if (storage->nodes[index]) {
                hash_update(ladder, storage->nodes[index]->node_id, tier, index);
            }
        }
        storage->nodes[storage->count - 1] = NULL;
        storage->count--;
    }

    // Remove from hash
    hash_remove(ladder, node_id);
    ladder->total_nodes--;

    // Destroy node if requested
    if (destroy_node && node) {
        pr_memory_node_destroy(node);
    }

    return Z_LADDER_SUCCESS;
}

/**
 * @brief Internal insert without locking (caller holds mutex)
 */
static z_ladder_error_t insert_node_unlocked(z_ladder_t ladder,
                                              pr_memory_node_t* node,
                                              pr_memory_tier_t tier) {
    if (!node) {
        return Z_LADDER_ERROR_NULL_POINTER;
    }

    if (tier >= Z_LADDER_NUM_TIERS) {
        return Z_LADDER_ERROR_INVALID_TIER;
    }

    uint64_t node_id = node->node_id;

    // Check if already exists
    if (hash_find(ladder, node_id)) {
        return Z_LADDER_ERROR_ALREADY_EXISTS;
    }

    z_tier_storage_t* storage = &ladder->tiers[tier];

    // Check capacity
    if (storage->config.capacity > 0 && storage->count >= storage->config.capacity) {
        return Z_LADDER_ERROR_CAPACITY;
    }

    // Add to tier storage
    size_t index = tier_storage_add(storage, node);
    if (index == SIZE_MAX) {
        return Z_LADDER_ERROR_NO_MEMORY;
    }

    // Add to hash table
    if (!hash_insert(ladder, node_id, node, tier, index)) {
        tier_storage_remove(storage, index);
        return Z_LADDER_ERROR_NO_MEMORY;
    }

    // Update node's tier
    node->tier = tier;
    node->decay_rate = storage->config.decay_rate;

    ladder->total_nodes++;

    return Z_LADDER_SUCCESS;
}

/**
 * @brief Internal promote without locking
 */
static z_ladder_error_t promote_unlocked(z_ladder_t ladder, uint64_t node_id) {
    z_hash_entry_t* entry = hash_find(ladder, node_id);
    if (!entry) {
        return Z_LADDER_ERROR_NOT_FOUND;
    }

    pr_memory_tier_t from_tier = entry->tier;

    if (from_tier >= PR_MEMORY_TIER_Z3) {
        return Z_LADDER_ERROR_ALREADY_TOP;
    }

    pr_memory_tier_t to_tier = (pr_memory_tier_t)(from_tier + 1);
    pr_memory_node_t* node = entry->node;
    float strength_before = node->current_strength;

    // Remove from current tier
    z_ladder_error_t err = remove_node_unlocked(ladder, node_id, false);
    if (err != Z_LADDER_SUCCESS) {
        return err;
    }

    // Insert into new tier
    err = insert_node_unlocked(ladder, node, to_tier);
    if (err != Z_LADDER_SUCCESS) {
        // Rollback: reinsert into original tier
        insert_node_unlocked(ladder, node, from_tier);
        return err;
    }

    // Update statistics
    ladder->promotions[from_tier]++;

    // Record event
    record_event(ladder, node_id, from_tier, to_tier,
                 strength_before, node->current_strength, true);

    // Invoke callback
    if (ladder->enable_callbacks && ladder->promotion_callback) {
        ladder->promotion_callback(node, from_tier, to_tier,
                                   ladder->promotion_user_data);
    }

    return Z_LADDER_SUCCESS;
}

/**
 * @brief Internal demote without locking
 */
static z_ladder_error_t demote_unlocked(z_ladder_t ladder, uint64_t node_id) {
    z_hash_entry_t* entry = hash_find(ladder, node_id);
    if (!entry) {
        return Z_LADDER_ERROR_NOT_FOUND;
    }

    pr_memory_tier_t from_tier = entry->tier;

    if (from_tier <= PR_MEMORY_TIER_Z0) {
        return Z_LADDER_ERROR_ALREADY_BOTTOM;
    }

    pr_memory_tier_t to_tier = (pr_memory_tier_t)(from_tier - 1);
    pr_memory_node_t* node = entry->node;
    float strength_before = node->current_strength;

    // Remove from current tier
    z_ladder_error_t err = remove_node_unlocked(ladder, node_id, false);
    if (err != Z_LADDER_SUCCESS) {
        return err;
    }

    // Insert into new tier
    err = insert_node_unlocked(ladder, node, to_tier);
    if (err != Z_LADDER_SUCCESS) {
        // Rollback: reinsert into original tier
        insert_node_unlocked(ladder, node, from_tier);
        return err;
    }

    // Update statistics
    ladder->demotions[from_tier - 1]++;

    // Record event
    record_event(ladder, node_id, from_tier, to_tier,
                 strength_before, node->current_strength, false);

    return Z_LADDER_SUCCESS;
}

/**
 * @brief Internal evict without locking
 */
static z_ladder_error_t evict_unlocked(z_ladder_t ladder, pr_memory_tier_t tier,
                                        size_t evict_index) {
    z_tier_storage_t* storage = &ladder->tiers[tier];

    if (evict_index >= storage->count) {
        return Z_LADDER_ERROR_NOT_FOUND;
    }

    pr_memory_node_t* node = storage->nodes[evict_index];
    uint64_t node_id = node->node_id;

    // Invoke callback before destruction
    if (ladder->enable_callbacks && ladder->eviction_callback) {
        ladder->eviction_callback(node, tier, ladder->eviction_user_data);
    }

    // Remove and destroy
    z_ladder_error_t err = remove_node_unlocked(ladder, node_id, true);

    if (err == Z_LADDER_SUCCESS) {
        ladder->evictions[tier]++;
    }

    return err;
}

/**
 * @brief Check promotion eligibility for a node (unlocked)
 */
static bool check_promotion_unlocked(z_ladder_t ladder, const pr_memory_node_t* node) {
    if (!node) {
        return false;
    }

    pr_memory_tier_t tier = node->tier;

    if (tier >= PR_MEMORY_TIER_Z3) {
        return false;  // Already at top
    }

    const z_tier_config_t* config = &ladder->tiers[tier].config;
    uint64_t now = get_current_time_ms();
    uint64_t age_ms = (now >= node->created_time_ms) ? (now - node->created_time_ms) : 0;

    // Check minimum age
    if (age_ms < config->min_age_for_promotion_ms) {
        return false;
    }

    // Check minimum access count
    uint64_t access_count = atomic_load(&node->access_count);
    if (access_count < config->min_access_count) {
        return false;
    }

    // Check strength threshold
    if (node->current_strength < config->promotion_threshold) {
        return false;
    }

    // Tier-specific criteria
    switch (tier) {
        case PR_MEMORY_TIER_Z0:
            // Z0->Z1: Check salience
            if (node->state.y < config->min_salience) {
                return false;
            }
            break;

        case PR_MEMORY_TIER_Z1:
            // Z1->Z2: Check consolidation
            if (node->state.w < config->min_consolidation) {
                return false;
            }
            break;

        case PR_MEMORY_TIER_Z2:
            // Z2->Z3: Check consolidation and entanglement
            if (node->state.w < config->min_consolidation) {
                return false;
            }
            if (atomic_load(&node->entanglement_count) < (uint32_t)config->min_entanglement) {
                return false;
            }
            break;

        default:
            return false;
    }

    return true;
}

/**
 * @brief Check demotion criteria for a node (unlocked)
 */
static bool check_demotion_unlocked(z_ladder_t ladder, const pr_memory_node_t* node) {
    if (!node) {
        return false;
    }

    pr_memory_tier_t tier = node->tier;

    if (tier <= PR_MEMORY_TIER_Z0) {
        return false;  // Z0 nodes get evicted, not demoted
    }

    const z_tier_config_t* config = &ladder->tiers[tier].config;

    // Demote if strength falls below threshold
    return (node->current_strength < config->demotion_threshold);
}

//=============================================================================
// Configuration Functions Implementation
//=============================================================================

z_tier_config_t z_ladder_default_tier_config(pr_memory_tier_t tier) {
    z_tier_config_t config = {0};

    switch (tier) {
        case PR_MEMORY_TIER_Z0:
            config.capacity = Z_LADDER_Z0_CAPACITY;
            config.decay_rate = Z_LADDER_DECAY_Z0;
            config.promotion_threshold = Z_LADDER_PROMOTE_Z0_STRENGTH;
            config.demotion_threshold = 0.0f;  // Z0 doesn't demote, it evicts
            config.min_age_for_promotion_ms = Z_LADDER_MIN_AGE_Z0_MS;
            config.min_access_count = Z_LADDER_MIN_ACCESS_Z0;
            config.min_entanglement = 0.0f;
            config.min_salience = Z_LADDER_MIN_SALIENCE_Z0;
            config.min_consolidation = 0.0f;
            config.eviction_policy = Z_EVICT_COMBINED;
            break;

        case PR_MEMORY_TIER_Z1:
            config.capacity = Z_LADDER_Z1_CAPACITY;
            config.decay_rate = Z_LADDER_DECAY_Z1;
            config.promotion_threshold = Z_LADDER_PROMOTE_Z1_STRENGTH;
            config.demotion_threshold = Z_LADDER_DEMOTE_THRESHOLD;
            config.min_age_for_promotion_ms = Z_LADDER_MIN_AGE_Z1_MS;
            config.min_access_count = Z_LADDER_MIN_ACCESS_Z1;
            config.min_entanglement = 0.0f;
            config.min_salience = 0.0f;
            config.min_consolidation = Z_LADDER_MIN_CONSOL_Z1;
            config.eviction_policy = Z_EVICT_WEAKEST;
            break;

        case PR_MEMORY_TIER_Z2:
            config.capacity = Z_LADDER_Z2_CAPACITY;
            config.decay_rate = Z_LADDER_DECAY_Z2;
            config.promotion_threshold = Z_LADDER_PROMOTE_Z2_STRENGTH;
            config.demotion_threshold = Z_LADDER_DEMOTE_THRESHOLD;
            config.min_age_for_promotion_ms = Z_LADDER_MIN_AGE_Z2_MS;
            config.min_access_count = 0;
            config.min_entanglement = (float)Z_LADDER_MIN_ENTANGLE_Z2;
            config.min_salience = 0.0f;
            config.min_consolidation = Z_LADDER_MIN_CONSOL_Z2;
            config.eviction_policy = Z_EVICT_WEAKEST;
            break;

        case PR_MEMORY_TIER_Z3:
            config.capacity = Z_LADDER_Z3_CAPACITY;  // 0 = unlimited
            config.decay_rate = Z_LADDER_DECAY_Z3;   // No decay
            config.promotion_threshold = 1.0f;       // Cannot promote further
            config.demotion_threshold = 0.05f;       // Very rarely demote
            config.min_age_for_promotion_ms = 0;
            config.min_access_count = 0;
            config.min_entanglement = 0.0f;
            config.min_salience = 0.0f;
            config.min_consolidation = 0.0f;
            config.eviction_policy = Z_EVICT_WEAKEST;
            break;

        default:
            // Return Z0 defaults for unknown tier
            return z_ladder_default_tier_config(PR_MEMORY_TIER_Z0);
    }

    return config;
}

z_ladder_config_t z_ladder_default_config(void) {
    z_ladder_config_t config = {0};

    // Initialize tier configs
    for (int i = 0; i < Z_LADDER_NUM_TIERS; i++) {
        config.tier_configs[i] = z_ladder_default_tier_config((pr_memory_tier_t)i);
    }

    config.hash_initial_capacity = Z_LADDER_HASH_INITIAL_CAP;
    config.enable_callbacks = true;
    config.enable_event_tracking = true;
    config.max_events = Z_LADDER_MAX_EVENTS;
    config.node_manager = NULL;

    return config;
}

bool z_ladder_config_validate(const z_ladder_config_t* config) {
    if (!config) {
        return false;
    }

    // Validate tier configs
    for (int i = 0; i < Z_LADDER_NUM_TIERS; i++) {
        const z_tier_config_t* tc = &config->tier_configs[i];

        // Decay rate must be non-negative
        if (tc->decay_rate < 0.0f) {
            return false;
        }

        // Thresholds must be in [0, 1]
        if (tc->promotion_threshold < 0.0f || tc->promotion_threshold > 1.0f) {
            return false;
        }
        if (tc->demotion_threshold < 0.0f || tc->demotion_threshold > 1.0f) {
            return false;
        }
    }

    return true;
}

//=============================================================================
// Manager API Implementation
//=============================================================================

z_ladder_t z_ladder_create(const z_ladder_config_t* config) {
    z_ladder_config_t cfg;
    if (config) {
        cfg = *config;
    } else {
        cfg = z_ladder_default_config();
    }

    // Validate configuration
    if (!z_ladder_config_validate(&cfg)) {
        return NULL;
    }

    // Allocate ladder structure
    z_ladder_t ladder = (z_ladder_t)calloc(1, sizeof(struct z_ladder_struct));
    if (!ladder) {
        return NULL;
    }

    // Initialize mutex
    if (pthread_mutex_init(&ladder->mutex, NULL) != 0) {
        free(ladder);
        return NULL;
    }
    ladder->mutex_initialized = true;

    // Initialize tier storage
    for (int i = 0; i < Z_LADDER_NUM_TIERS; i++) {
        if (!tier_storage_init(&ladder->tiers[i], &cfg.tier_configs[i])) {
            // Cleanup on failure
            for (int j = 0; j < i; j++) {
                tier_storage_destroy(&ladder->tiers[j]);
            }
            pthread_mutex_destroy(&ladder->mutex);
            free(ladder);
            return NULL;
        }
    }

    // Initialize hash buckets (already zeroed by calloc)

    // Initialize event tracking
    ladder->enable_events = cfg.enable_event_tracking;
    if (cfg.enable_event_tracking && cfg.max_events > 0) {
        ladder->events = (z_consolidation_event_t*)calloc(
            cfg.max_events, sizeof(z_consolidation_event_t));
        if (ladder->events) {
            ladder->event_capacity = cfg.max_events;
        }
    }

    // Initialize callbacks
    ladder->enable_callbacks = cfg.enable_callbacks;

    // Store node manager
    ladder->node_manager = cfg.node_manager;

    // Initialize timestamps
    uint64_t now = get_current_time_ms();
    ladder->last_decay_time_ms = now;
    ladder->last_consolidation_time_ms = now;

    return ladder;
}

void z_ladder_destroy(z_ladder_t ladder) {
    if (!ladder) {
        return;
    }

    // Lock to ensure no concurrent access
    if (ladder->mutex_initialized) {
        pthread_mutex_lock(&ladder->mutex);
    }

    // Destroy all nodes in all tiers
    for (int i = 0; i < Z_LADDER_NUM_TIERS; i++) {
        z_tier_storage_t* tier = &ladder->tiers[i];
        for (size_t j = 0; j < tier->count; j++) {
            if (tier->nodes[j]) {
                pr_memory_node_destroy(tier->nodes[j]);
                tier->nodes[j] = NULL;
            }
        }
        tier_storage_destroy(tier);
    }

    // Clear hash table
    hash_clear(ladder);

    // Free events buffer
    if (ladder->events) {
        free(ladder->events);
    }

    // Unlock before destroying mutex
    if (ladder->mutex_initialized) {
        pthread_mutex_unlock(&ladder->mutex);
        pthread_mutex_destroy(&ladder->mutex);
    }

    free(ladder);
}

z_ladder_error_t z_ladder_clear(z_ladder_t ladder) {
    if (!ladder) {
        return Z_LADDER_ERROR_NULL_POINTER;
    }

    pthread_mutex_lock(&ladder->mutex);

    // Destroy all nodes
    for (int i = 0; i < Z_LADDER_NUM_TIERS; i++) {
        z_tier_storage_t* tier = &ladder->tiers[i];
        for (size_t j = 0; j < tier->count; j++) {
            if (tier->nodes[j]) {
                pr_memory_node_destroy(tier->nodes[j]);
                tier->nodes[j] = NULL;
            }
        }
        tier->count = 0;
    }

    // Clear hash table
    hash_clear(ladder);
    ladder->total_nodes = 0;

    // Reset statistics
    memset(ladder->promotions, 0, sizeof(ladder->promotions));
    memset(ladder->demotions, 0, sizeof(ladder->demotions));
    memset(ladder->evictions, 0, sizeof(ladder->evictions));

    // Clear events
    if (ladder->events) {
        memset(ladder->events, 0, ladder->event_capacity * sizeof(z_consolidation_event_t));
        ladder->event_count = 0;
        ladder->event_head = 0;
    }

    pthread_mutex_unlock(&ladder->mutex);

    return Z_LADDER_SUCCESS;
}

//=============================================================================
// Node Management API Implementation
//=============================================================================

z_ladder_error_t z_ladder_insert(z_ladder_t ladder, pr_memory_node_t* node,
                                  pr_memory_tier_t tier) {
    if (!ladder) {
        return Z_LADDER_ERROR_NULL_POINTER;
    }

    if (!node) {
        return Z_LADDER_ERROR_NULL_POINTER;
    }

    if (tier >= Z_LADDER_NUM_TIERS) {
        return Z_LADDER_ERROR_INVALID_TIER;
    }

    pthread_mutex_lock(&ladder->mutex);

    z_ladder_error_t err = insert_node_unlocked(ladder, node, tier);

    pthread_mutex_unlock(&ladder->mutex);

    return err;
}

z_ladder_error_t z_ladder_remove(z_ladder_t ladder, uint64_t node_id) {
    if (!ladder) {
        return Z_LADDER_ERROR_NULL_POINTER;
    }

    pthread_mutex_lock(&ladder->mutex);

    z_ladder_error_t err = remove_node_unlocked(ladder, node_id, true);

    pthread_mutex_unlock(&ladder->mutex);

    return err;
}

pr_memory_node_t* z_ladder_find(z_ladder_t ladder, uint64_t node_id) {
    if (!ladder) {
        return NULL;
    }

    pthread_mutex_lock(&ladder->mutex);

    z_hash_entry_t* entry = hash_find(ladder, node_id);
    pr_memory_node_t* node = entry ? entry->node : NULL;

    // Update access metadata
    if (node) {
        node->last_accessed_ms = get_current_time_ms();
        atomic_fetch_add(&node->access_count, 1);
    }

    pthread_mutex_unlock(&ladder->mutex);

    return node;
}

z_ladder_error_t z_ladder_get_tier(z_ladder_t ladder, uint64_t node_id,
                                    pr_memory_tier_t* tier) {
    if (!ladder || !tier) {
        return Z_LADDER_ERROR_NULL_POINTER;
    }

    pthread_mutex_lock(&ladder->mutex);

    z_hash_entry_t* entry = hash_find(ladder, node_id);
    z_ladder_error_t err;

    if (entry) {
        *tier = entry->tier;
        err = Z_LADDER_SUCCESS;
    } else {
        err = Z_LADDER_ERROR_NOT_FOUND;
    }

    pthread_mutex_unlock(&ladder->mutex);

    return err;
}

z_ladder_error_t z_ladder_move(z_ladder_t ladder, uint64_t node_id,
                                pr_memory_tier_t new_tier) {
    if (!ladder) {
        return Z_LADDER_ERROR_NULL_POINTER;
    }

    if (new_tier >= Z_LADDER_NUM_TIERS) {
        return Z_LADDER_ERROR_INVALID_TIER;
    }

    pthread_mutex_lock(&ladder->mutex);

    z_hash_entry_t* entry = hash_find(ladder, node_id);
    if (!entry) {
        pthread_mutex_unlock(&ladder->mutex);
        return Z_LADDER_ERROR_NOT_FOUND;
    }

    // If already in target tier, nothing to do
    if (entry->tier == new_tier) {
        pthread_mutex_unlock(&ladder->mutex);
        return Z_LADDER_SUCCESS;
    }

    pr_memory_node_t* node = entry->node;
    pr_memory_tier_t old_tier = entry->tier;

    // Remove from current tier
    z_ladder_error_t err = remove_node_unlocked(ladder, node_id, false);
    if (err != Z_LADDER_SUCCESS) {
        pthread_mutex_unlock(&ladder->mutex);
        return err;
    }

    // Insert into new tier
    err = insert_node_unlocked(ladder, node, new_tier);
    if (err != Z_LADDER_SUCCESS) {
        // Rollback
        insert_node_unlocked(ladder, node, old_tier);
        pthread_mutex_unlock(&ladder->mutex);
        return err;
    }

    pthread_mutex_unlock(&ladder->mutex);

    return Z_LADDER_SUCCESS;
}

//=============================================================================
// Promotion/Demotion API Implementation
//=============================================================================

bool z_ladder_check_promotion(z_ladder_t ladder, const pr_memory_node_t* node) {
    if (!ladder || !node) {
        return false;
    }

    pthread_mutex_lock(&ladder->mutex);

    bool eligible = check_promotion_unlocked(ladder, node);

    pthread_mutex_unlock(&ladder->mutex);

    return eligible;
}

bool z_ladder_check_demotion(z_ladder_t ladder, const pr_memory_node_t* node) {
    if (!ladder || !node) {
        return false;
    }

    pthread_mutex_lock(&ladder->mutex);

    bool should_demote = check_demotion_unlocked(ladder, node);

    pthread_mutex_unlock(&ladder->mutex);

    return should_demote;
}

z_ladder_error_t z_ladder_promote(z_ladder_t ladder, uint64_t node_id) {
    if (!ladder) {
        return Z_LADDER_ERROR_NULL_POINTER;
    }

    pthread_mutex_lock(&ladder->mutex);

    z_ladder_error_t err = promote_unlocked(ladder, node_id);

    pthread_mutex_unlock(&ladder->mutex);

    return err;
}

z_ladder_error_t z_ladder_demote(z_ladder_t ladder, uint64_t node_id) {
    if (!ladder) {
        return Z_LADDER_ERROR_NULL_POINTER;
    }

    pthread_mutex_lock(&ladder->mutex);

    z_hash_entry_t* entry = hash_find(ladder, node_id);
    if (!entry) {
        pthread_mutex_unlock(&ladder->mutex);
        return Z_LADDER_ERROR_NOT_FOUND;
    }

    z_ladder_error_t err;

    if (entry->tier == PR_MEMORY_TIER_Z0) {
        // At Z0, demotion means eviction
        size_t index = entry->tier_index;
        err = evict_unlocked(ladder, PR_MEMORY_TIER_Z0, index);
    } else {
        err = demote_unlocked(ladder, node_id);
    }

    pthread_mutex_unlock(&ladder->mutex);

    return err;
}

size_t z_ladder_process_promotions(z_ladder_t ladder) {
    if (!ladder) {
        return 0;
    }

    pthread_mutex_lock(&ladder->mutex);

    size_t promoted_count = 0;

    // Process tiers Z0 through Z2 (Z3 cannot be promoted)
    for (int tier_idx = 0; tier_idx < 3; tier_idx++) {
        z_tier_storage_t* tier = &ladder->tiers[tier_idx];

        // Collect eligible nodes first (to avoid modifying while iterating)
        size_t eligible_count = 0;
        uint64_t* eligible_ids = (uint64_t*)malloc(tier->count * sizeof(uint64_t));
        if (!eligible_ids) {
            continue;
        }

        for (size_t i = 0; i < tier->count; i++) {
            if (check_promotion_unlocked(ladder, tier->nodes[i])) {
                eligible_ids[eligible_count++] = tier->nodes[i]->node_id;
            }
        }

        // Now promote each eligible node
        for (size_t i = 0; i < eligible_count; i++) {
            if (promote_unlocked(ladder, eligible_ids[i]) == Z_LADDER_SUCCESS) {
                promoted_count++;
            }
        }

        free(eligible_ids);
    }

    pthread_mutex_unlock(&ladder->mutex);

    return promoted_count;
}

size_t z_ladder_process_demotions(z_ladder_t ladder) {
    if (!ladder) {
        return 0;
    }

    pthread_mutex_lock(&ladder->mutex);

    size_t demoted_count = 0;

    // Process tiers Z3 down to Z1 (Z0 doesn't demote, it evicts)
    for (int tier_idx = 3; tier_idx >= 1; tier_idx--) {
        z_tier_storage_t* tier = &ladder->tiers[tier_idx];

        // Collect nodes to demote
        size_t demote_count = 0;
        uint64_t* demote_ids = (uint64_t*)malloc(tier->count * sizeof(uint64_t));
        if (!demote_ids) {
            continue;
        }

        for (size_t i = 0; i < tier->count; i++) {
            if (check_demotion_unlocked(ladder, tier->nodes[i])) {
                demote_ids[demote_count++] = tier->nodes[i]->node_id;
            }
        }

        // Demote each node
        for (size_t i = 0; i < demote_count; i++) {
            if (demote_unlocked(ladder, demote_ids[i]) == Z_LADDER_SUCCESS) {
                demoted_count++;
            }
        }

        free(demote_ids);
    }

    // Handle Z0 evictions (nodes below threshold get evicted)
    z_tier_storage_t* z0 = &ladder->tiers[PR_MEMORY_TIER_Z0];
    float demotion_threshold = z0->config.demotion_threshold;

    // Collect nodes to evict from Z0
    size_t evict_count = 0;
    size_t* evict_indices = (size_t*)malloc(z0->count * sizeof(size_t));
    if (evict_indices) {
        for (size_t i = 0; i < z0->count; i++) {
            if (z0->nodes[i]->current_strength < demotion_threshold) {
                evict_indices[evict_count++] = i;
            }
        }

        // Evict in reverse order (to preserve indices)
        for (size_t i = evict_count; i > 0; i--) {
            if (evict_unlocked(ladder, PR_MEMORY_TIER_Z0, evict_indices[i - 1])
                == Z_LADDER_SUCCESS) {
                demoted_count++;
            }
        }

        free(evict_indices);
    }

    pthread_mutex_unlock(&ladder->mutex);

    return demoted_count;
}

//=============================================================================
// Decay API Implementation
//=============================================================================

z_ladder_error_t z_ladder_apply_decay(z_ladder_t ladder, float dt_seconds) {
    if (!ladder) {
        return Z_LADDER_ERROR_NULL_POINTER;
    }

    if (dt_seconds <= 0.0f) {
        return Z_LADDER_SUCCESS;  // No decay to apply
    }

    pthread_mutex_lock(&ladder->mutex);

    // Apply decay to each tier (except Z3 which has zero decay)
    for (int tier_idx = 0; tier_idx < Z_LADDER_NUM_TIERS; tier_idx++) {
        z_tier_storage_t* tier = &ladder->tiers[tier_idx];

        if (tier->config.decay_rate <= 0.0f) {
            continue;  // No decay for this tier (Z3)
        }

        float decay_factor = expf(-tier->config.decay_rate * dt_seconds);

        for (size_t i = 0; i < tier->count; i++) {
            pr_memory_node_t* node = tier->nodes[i];
            node->current_strength *= decay_factor;
            node->current_strength = clampf(node->current_strength, 0.0f, 1.0f);

            // Sync with quaternion consolidation component
            node->state.w = node->current_strength;
        }
    }

    ladder->last_decay_time_ms = get_current_time_ms();

    pthread_mutex_unlock(&ladder->mutex);

    return Z_LADDER_SUCCESS;
}

z_ladder_error_t z_ladder_decay_tier(z_ladder_t ladder, pr_memory_tier_t tier,
                                      float dt_seconds) {
    if (!ladder) {
        return Z_LADDER_ERROR_NULL_POINTER;
    }

    if (tier >= Z_LADDER_NUM_TIERS) {
        return Z_LADDER_ERROR_INVALID_TIER;
    }

    if (dt_seconds <= 0.0f) {
        return Z_LADDER_SUCCESS;
    }

    pthread_mutex_lock(&ladder->mutex);

    z_tier_storage_t* storage = &ladder->tiers[tier];

    if (storage->config.decay_rate > 0.0f) {
        float decay_factor = expf(-storage->config.decay_rate * dt_seconds);

        for (size_t i = 0; i < storage->count; i++) {
            pr_memory_node_t* node = storage->nodes[i];
            node->current_strength *= decay_factor;
            node->current_strength = clampf(node->current_strength, 0.0f, 1.0f);
            node->state.w = node->current_strength;
        }
    }

    pthread_mutex_unlock(&ladder->mutex);

    return Z_LADDER_SUCCESS;
}

float z_ladder_get_decay_rate(z_ladder_t ladder, pr_memory_tier_t tier) {
    if (!ladder || tier >= Z_LADDER_NUM_TIERS) {
        return 0.0f;
    }

    // No lock needed - tier config is read-only after creation
    return ladder->tiers[tier].config.decay_rate;
}

z_ladder_error_t z_ladder_set_decay_rate(z_ladder_t ladder, pr_memory_tier_t tier,
                                          float rate) {
    if (!ladder) {
        return Z_LADDER_ERROR_NULL_POINTER;
    }

    if (tier >= Z_LADDER_NUM_TIERS) {
        return Z_LADDER_ERROR_INVALID_TIER;
    }

    if (rate < 0.0f) {
        return Z_LADDER_ERROR_INVALID_CONFIG;
    }

    pthread_mutex_lock(&ladder->mutex);

    ladder->tiers[tier].config.decay_rate = rate;

    // Update decay rate on all nodes in this tier
    z_tier_storage_t* storage = &ladder->tiers[tier];
    for (size_t i = 0; i < storage->count; i++) {
        storage->nodes[i]->decay_rate = rate;
    }

    pthread_mutex_unlock(&ladder->mutex);

    return Z_LADDER_SUCCESS;
}

//=============================================================================
// Eviction API Implementation
//=============================================================================

z_ladder_error_t z_ladder_evict_weakest(z_ladder_t ladder, pr_memory_tier_t tier) {
    if (!ladder) {
        return Z_LADDER_ERROR_NULL_POINTER;
    }

    if (tier >= Z_LADDER_NUM_TIERS) {
        return Z_LADDER_ERROR_INVALID_TIER;
    }

    pthread_mutex_lock(&ladder->mutex);

    z_tier_storage_t* storage = &ladder->tiers[tier];

    if (storage->count == 0) {
        pthread_mutex_unlock(&ladder->mutex);
        return Z_LADDER_ERROR_NOT_FOUND;
    }

    size_t evict_index = tier_storage_find_evict_target(storage);
    z_ladder_error_t err = evict_unlocked(ladder, tier, evict_index);

    pthread_mutex_unlock(&ladder->mutex);

    return err;
}

size_t z_ladder_evict_if_full(z_ladder_t ladder, pr_memory_tier_t tier) {
    if (!ladder || tier >= Z_LADDER_NUM_TIERS) {
        return 0;
    }

    pthread_mutex_lock(&ladder->mutex);

    z_tier_storage_t* storage = &ladder->tiers[tier];
    size_t evicted = 0;

    // Keep evicting while over capacity (0 = unlimited)
    while (storage->config.capacity > 0 && storage->count > storage->config.capacity) {
        size_t evict_index = tier_storage_find_evict_target(storage);
        if (evict_unlocked(ladder, tier, evict_index) == Z_LADDER_SUCCESS) {
            evicted++;
        } else {
            break;  // Failed to evict
        }
    }

    pthread_mutex_unlock(&ladder->mutex);

    return evicted;
}

z_ladder_error_t z_ladder_set_eviction_policy(z_ladder_t ladder, pr_memory_tier_t tier,
                                               z_eviction_policy_t policy) {
    if (!ladder) {
        return Z_LADDER_ERROR_NULL_POINTER;
    }

    if (tier >= Z_LADDER_NUM_TIERS) {
        return Z_LADDER_ERROR_INVALID_TIER;
    }

    pthread_mutex_lock(&ladder->mutex);

    ladder->tiers[tier].config.eviction_policy = policy;

    pthread_mutex_unlock(&ladder->mutex);

    return Z_LADDER_SUCCESS;
}

//=============================================================================
// Query API Implementation
//=============================================================================

z_ladder_error_t z_ladder_get_nodes(z_ladder_t ladder, pr_memory_tier_t tier,
                                     pr_memory_node_t** nodes, size_t max_nodes,
                                     size_t* count) {
    if (!ladder || !nodes || !count) {
        return Z_LADDER_ERROR_NULL_POINTER;
    }

    if (tier >= Z_LADDER_NUM_TIERS) {
        return Z_LADDER_ERROR_INVALID_TIER;
    }

    pthread_mutex_lock(&ladder->mutex);

    z_tier_storage_t* storage = &ladder->tiers[tier];
    size_t to_copy = (storage->count < max_nodes) ? storage->count : max_nodes;

    for (size_t i = 0; i < to_copy; i++) {
        nodes[i] = storage->nodes[i];
    }

    *count = to_copy;

    pthread_mutex_unlock(&ladder->mutex);

    return Z_LADDER_SUCCESS;
}

size_t z_ladder_get_count(z_ladder_t ladder, pr_memory_tier_t tier) {
    if (!ladder || tier >= Z_LADDER_NUM_TIERS) {
        return 0;
    }

    pthread_mutex_lock(&ladder->mutex);

    size_t count = ladder->tiers[tier].count;

    pthread_mutex_unlock(&ladder->mutex);

    return count;
}

size_t z_ladder_get_total_count(z_ladder_t ladder) {
    if (!ladder) {
        return 0;
    }

    pthread_mutex_lock(&ladder->mutex);

    size_t total = ladder->total_nodes;

    pthread_mutex_unlock(&ladder->mutex);

    return total;
}

/**
 * @brief Comparison function for sorting nodes by strength (descending)
 */
static int compare_strength_desc(const void* a, const void* b) {
    pr_memory_node_t* na = *(pr_memory_node_t**)a;
    pr_memory_node_t* nb = *(pr_memory_node_t**)b;

    if (na->current_strength > nb->current_strength) return -1;
    if (na->current_strength < nb->current_strength) return 1;
    return 0;
}

/**
 * @brief Comparison function for sorting nodes by strength (ascending)
 */
static int compare_strength_asc(const void* a, const void* b) {
    return -compare_strength_desc(a, b);
}

z_ladder_error_t z_ladder_get_strongest(z_ladder_t ladder, pr_memory_tier_t tier,
                                         size_t k, pr_memory_node_t** nodes,
                                         size_t* count) {
    if (!ladder || !nodes || !count) {
        return Z_LADDER_ERROR_NULL_POINTER;
    }

    if (tier >= Z_LADDER_NUM_TIERS) {
        return Z_LADDER_ERROR_INVALID_TIER;
    }

    pthread_mutex_lock(&ladder->mutex);

    z_tier_storage_t* storage = &ladder->tiers[tier];

    // Copy pointers to temporary array
    size_t n = storage->count;
    pr_memory_node_t** temp = (pr_memory_node_t**)malloc(n * sizeof(pr_memory_node_t*));
    if (!temp) {
        pthread_mutex_unlock(&ladder->mutex);
        return Z_LADDER_ERROR_NO_MEMORY;
    }

    memcpy(temp, storage->nodes, n * sizeof(pr_memory_node_t*));

    // Sort by strength (descending)
    qsort(temp, n, sizeof(pr_memory_node_t*), compare_strength_desc);

    // Copy top k
    size_t result_count = (k < n) ? k : n;
    for (size_t i = 0; i < result_count; i++) {
        nodes[i] = temp[i];
    }

    *count = result_count;

    free(temp);

    pthread_mutex_unlock(&ladder->mutex);

    return Z_LADDER_SUCCESS;
}

z_ladder_error_t z_ladder_get_weakest(z_ladder_t ladder, pr_memory_tier_t tier,
                                       size_t k, pr_memory_node_t** nodes,
                                       size_t* count) {
    if (!ladder || !nodes || !count) {
        return Z_LADDER_ERROR_NULL_POINTER;
    }

    if (tier >= Z_LADDER_NUM_TIERS) {
        return Z_LADDER_ERROR_INVALID_TIER;
    }

    pthread_mutex_lock(&ladder->mutex);

    z_tier_storage_t* storage = &ladder->tiers[tier];

    size_t n = storage->count;
    pr_memory_node_t** temp = (pr_memory_node_t**)malloc(n * sizeof(pr_memory_node_t*));
    if (!temp) {
        pthread_mutex_unlock(&ladder->mutex);
        return Z_LADDER_ERROR_NO_MEMORY;
    }

    memcpy(temp, storage->nodes, n * sizeof(pr_memory_node_t*));

    // Sort by strength (ascending)
    qsort(temp, n, sizeof(pr_memory_node_t*), compare_strength_asc);

    // Copy bottom k
    size_t result_count = (k < n) ? k : n;
    for (size_t i = 0; i < result_count; i++) {
        nodes[i] = temp[i];
    }

    *count = result_count;

    free(temp);

    pthread_mutex_unlock(&ladder->mutex);

    return Z_LADDER_SUCCESS;
}

//=============================================================================
// Consolidation API Implementation
//=============================================================================

z_ladder_error_t z_ladder_consolidate(z_ladder_t ladder) {
    if (!ladder) {
        return Z_LADDER_ERROR_NULL_POINTER;
    }

    pthread_mutex_lock(&ladder->mutex);

    uint64_t now = get_current_time_ms();
    float dt_seconds = (float)(now - ladder->last_decay_time_ms) / 1000.0f;

    // Phase 1: Apply decay
    for (int tier_idx = 0; tier_idx < Z_LADDER_NUM_TIERS; tier_idx++) {
        z_tier_storage_t* tier = &ladder->tiers[tier_idx];

        if (tier->config.decay_rate > 0.0f && dt_seconds > 0.0f) {
            float decay_factor = expf(-tier->config.decay_rate * dt_seconds);

            for (size_t i = 0; i < tier->count; i++) {
                pr_memory_node_t* node = tier->nodes[i];
                node->current_strength *= decay_factor;
                node->current_strength = clampf(node->current_strength, 0.0f, 1.0f);
                node->state.w = node->current_strength;
            }
        }
    }
    ladder->last_decay_time_ms = now;

    // Phase 2: Process demotions (Z3 down to Z1)
    for (int tier_idx = 3; tier_idx >= 1; tier_idx--) {
        z_tier_storage_t* tier = &ladder->tiers[tier_idx];

        // Collect nodes to demote
        size_t demote_count = 0;
        uint64_t* demote_ids = (uint64_t*)malloc(tier->count * sizeof(uint64_t));
        if (demote_ids) {
            for (size_t i = 0; i < tier->count; i++) {
                if (check_demotion_unlocked(ladder, tier->nodes[i])) {
                    demote_ids[demote_count++] = tier->nodes[i]->node_id;
                }
            }

            for (size_t i = 0; i < demote_count; i++) {
                demote_unlocked(ladder, demote_ids[i]);
            }

            free(demote_ids);
        }
    }

    // Evict weak Z0 nodes
    z_tier_storage_t* z0 = &ladder->tiers[PR_MEMORY_TIER_Z0];
    for (size_t i = z0->count; i > 0; i--) {
        if (z0->nodes[i - 1]->current_strength < Z_LADDER_DEMOTE_THRESHOLD) {
            evict_unlocked(ladder, PR_MEMORY_TIER_Z0, i - 1);
        }
    }

    // Phase 3: Process promotions (Z0 to Z2)
    for (int tier_idx = 0; tier_idx < 3; tier_idx++) {
        z_tier_storage_t* tier = &ladder->tiers[tier_idx];

        size_t promote_count = 0;
        uint64_t* promote_ids = (uint64_t*)malloc(tier->count * sizeof(uint64_t));
        if (promote_ids) {
            for (size_t i = 0; i < tier->count; i++) {
                if (check_promotion_unlocked(ladder, tier->nodes[i])) {
                    promote_ids[promote_count++] = tier->nodes[i]->node_id;
                }
            }

            for (size_t i = 0; i < promote_count; i++) {
                promote_unlocked(ladder, promote_ids[i]);
            }

            free(promote_ids);
        }
    }

    // Phase 4: Evict over-capacity tiers
    for (int tier_idx = 0; tier_idx < Z_LADDER_NUM_TIERS; tier_idx++) {
        z_tier_storage_t* tier = &ladder->tiers[tier_idx];

        while (tier->config.capacity > 0 && tier->count > tier->config.capacity) {
            size_t evict_index = tier_storage_find_evict_target(tier);
            if (evict_unlocked(ladder, (pr_memory_tier_t)tier_idx, evict_index)
                != Z_LADDER_SUCCESS) {
                break;
            }
        }
    }

    ladder->last_consolidation_time_ms = now;
    ladder->total_consolidations++;

    pthread_mutex_unlock(&ladder->mutex);

    return Z_LADDER_SUCCESS;
}

z_ladder_error_t z_ladder_consolidate_tier(z_ladder_t ladder, pr_memory_tier_t tier) {
    if (!ladder) {
        return Z_LADDER_ERROR_NULL_POINTER;
    }

    if (tier >= Z_LADDER_NUM_TIERS) {
        return Z_LADDER_ERROR_INVALID_TIER;
    }

    pthread_mutex_lock(&ladder->mutex);

    uint64_t now = get_current_time_ms();
    float dt_seconds = (float)(now - ladder->last_decay_time_ms) / 1000.0f;

    z_tier_storage_t* storage = &ladder->tiers[tier];

    // Apply decay
    if (storage->config.decay_rate > 0.0f && dt_seconds > 0.0f) {
        float decay_factor = expf(-storage->config.decay_rate * dt_seconds);

        for (size_t i = 0; i < storage->count; i++) {
            pr_memory_node_t* node = storage->nodes[i];
            node->current_strength *= decay_factor;
            node->current_strength = clampf(node->current_strength, 0.0f, 1.0f);
            node->state.w = node->current_strength;
        }
    }

    // Process demotions (if not Z0)
    if (tier > PR_MEMORY_TIER_Z0) {
        size_t demote_count = 0;
        uint64_t* demote_ids = (uint64_t*)malloc(storage->count * sizeof(uint64_t));
        if (demote_ids) {
            for (size_t i = 0; i < storage->count; i++) {
                if (check_demotion_unlocked(ladder, storage->nodes[i])) {
                    demote_ids[demote_count++] = storage->nodes[i]->node_id;
                }
            }

            for (size_t i = 0; i < demote_count; i++) {
                demote_unlocked(ladder, demote_ids[i]);
            }

            free(demote_ids);
        }
    }

    // Process promotions (if not Z3)
    if (tier < PR_MEMORY_TIER_Z3) {
        size_t promote_count = 0;
        uint64_t* promote_ids = (uint64_t*)malloc(storage->count * sizeof(uint64_t));
        if (promote_ids) {
            for (size_t i = 0; i < storage->count; i++) {
                if (check_promotion_unlocked(ladder, storage->nodes[i])) {
                    promote_ids[promote_count++] = storage->nodes[i]->node_id;
                }
            }

            for (size_t i = 0; i < promote_count; i++) {
                promote_unlocked(ladder, promote_ids[i]);
            }

            free(promote_ids);
        }
    }

    // Evict if over capacity
    while (storage->config.capacity > 0 && storage->count > storage->config.capacity) {
        size_t evict_index = tier_storage_find_evict_target(storage);
        if (evict_unlocked(ladder, tier, evict_index) != Z_LADDER_SUCCESS) {
            break;
        }
    }

    pthread_mutex_unlock(&ladder->mutex);

    return Z_LADDER_SUCCESS;
}

z_ladder_error_t z_ladder_sleep_consolidate(z_ladder_t ladder) {
    if (!ladder) {
        return Z_LADDER_ERROR_NULL_POINTER;
    }

    pthread_mutex_lock(&ladder->mutex);

    // Sleep consolidation: multiple passes with relaxed thresholds
    for (int pass = 0; pass < SLEEP_CONSOLIDATION_PASSES; pass++) {

        // Focus on Z1->Z2 consolidation (hippocampal -> neocortical transfer)
        z_tier_storage_t* z1 = &ladder->tiers[PR_MEMORY_TIER_Z1];

        // Temporarily reduce promotion thresholds
        float orig_threshold = z1->config.promotion_threshold;
        z1->config.promotion_threshold -= SLEEP_PROMOTION_REDUCTION;
        if (z1->config.promotion_threshold < 0.0f) {
            z1->config.promotion_threshold = 0.0f;
        }

        // Sort by emotional salience (emotional memories consolidate first)
        size_t n = z1->count;
        pr_memory_node_t** sorted = (pr_memory_node_t**)malloc(n * sizeof(pr_memory_node_t*));
        if (sorted) {
            memcpy(sorted, z1->nodes, n * sizeof(pr_memory_node_t*));

            // Sort by absolute emotional valence (descending)
            for (size_t i = 0; i < n; i++) {
                for (size_t j = i + 1; j < n; j++) {
                    float ai = absf(sorted[i]->state.x);
                    float aj = absf(sorted[j]->state.x);
                    if (aj > ai) {
                        pr_memory_node_t* tmp = sorted[i];
                        sorted[i] = sorted[j];
                        sorted[j] = tmp;
                    }
                }
            }

            // Promote eligible (emotional memories get priority)
            for (size_t i = 0; i < n; i++) {
                if (check_promotion_unlocked(ladder, sorted[i])) {
                    promote_unlocked(ladder, sorted[i]->node_id);
                }
            }

            free(sorted);
        }

        // Restore threshold
        z1->config.promotion_threshold = orig_threshold;

        // Also process Z0->Z1 with some boost
        z_tier_storage_t* z0 = &ladder->tiers[PR_MEMORY_TIER_Z0];
        for (size_t i = 0; i < z0->count; i++) {
            pr_memory_node_t* node = z0->nodes[i];
            // Give rehearsal boost during "sleep"
            node->current_strength += 0.1f;
            node->current_strength = clampf(node->current_strength, 0.0f, 1.0f);
            node->state.w = node->current_strength;
        }

        // Check for promotions after boost
        size_t promote_count = 0;
        uint64_t* promote_ids = (uint64_t*)malloc(z0->count * sizeof(uint64_t));
        if (promote_ids) {
            for (size_t i = 0; i < z0->count; i++) {
                if (check_promotion_unlocked(ladder, z0->nodes[i])) {
                    promote_ids[promote_count++] = z0->nodes[i]->node_id;
                }
            }

            for (size_t i = 0; i < promote_count; i++) {
                promote_unlocked(ladder, promote_ids[i]);
            }

            free(promote_ids);
        }
    }

    ladder->last_consolidation_time_ms = get_current_time_ms();
    ladder->total_consolidations++;

    pthread_mutex_unlock(&ladder->mutex);

    return Z_LADDER_SUCCESS;
}

z_ladder_error_t z_ladder_get_consolidation_events(z_ladder_t ladder,
                                                    z_consolidation_event_t* events,
                                                    size_t max_events,
                                                    size_t* count) {
    if (!ladder || !events || !count) {
        return Z_LADDER_ERROR_NULL_POINTER;
    }

    pthread_mutex_lock(&ladder->mutex);

    if (!ladder->events || ladder->event_count == 0) {
        *count = 0;
        pthread_mutex_unlock(&ladder->mutex);
        return Z_LADDER_SUCCESS;
    }

    size_t to_copy = (ladder->event_count < max_events) ? ladder->event_count : max_events;

    // Copy events in chronological order
    size_t start = (ladder->event_head >= ladder->event_count) ?
                   (ladder->event_head - ladder->event_count) :
                   (ladder->event_capacity - ladder->event_count + ladder->event_head);

    for (size_t i = 0; i < to_copy; i++) {
        size_t idx = (start + i) % ladder->event_capacity;
        events[i] = ladder->events[idx];
    }

    *count = to_copy;

    pthread_mutex_unlock(&ladder->mutex);

    return Z_LADDER_SUCCESS;
}

z_ladder_error_t z_ladder_clear_events(z_ladder_t ladder) {
    if (!ladder) {
        return Z_LADDER_ERROR_NULL_POINTER;
    }

    pthread_mutex_lock(&ladder->mutex);

    if (ladder->events) {
        memset(ladder->events, 0,
               ladder->event_capacity * sizeof(z_consolidation_event_t));
        ladder->event_count = 0;
        ladder->event_head = 0;
    }

    pthread_mutex_unlock(&ladder->mutex);

    return Z_LADDER_SUCCESS;
}

//=============================================================================
// Reinforcement API Implementation
//=============================================================================

float z_ladder_reinforce(z_ladder_t ladder, uint64_t node_id, float amount) {
    if (!ladder) {
        return -1.0f;
    }

    pthread_mutex_lock(&ladder->mutex);

    z_hash_entry_t* entry = hash_find(ladder, node_id);
    if (!entry) {
        pthread_mutex_unlock(&ladder->mutex);
        return -1.0f;
    }

    pr_memory_node_t* node = entry->node;

    // Add reinforcement
    node->current_strength += clampf(amount, 0.0f, 1.0f);
    node->current_strength = clampf(node->current_strength, 0.0f, 1.0f);

    // Sync with quaternion
    node->state.w = node->current_strength;

    float result = node->current_strength;

    pthread_mutex_unlock(&ladder->mutex);

    return result;
}

z_ladder_error_t z_ladder_access(z_ladder_t ladder, uint64_t node_id) {
    if (!ladder) {
        return Z_LADDER_ERROR_NULL_POINTER;
    }

    pthread_mutex_lock(&ladder->mutex);

    z_hash_entry_t* entry = hash_find(ladder, node_id);
    if (!entry) {
        pthread_mutex_unlock(&ladder->mutex);
        return Z_LADDER_ERROR_NOT_FOUND;
    }

    pr_memory_node_t* node = entry->node;

    // Update access metadata
    node->last_accessed_ms = get_current_time_ms();
    atomic_fetch_add(&node->access_count, 1);

    // Small strength boost on access
    node->current_strength += ACCESS_STRENGTH_BOOST;
    node->current_strength = clampf(node->current_strength, 0.0f, 1.0f);
    node->state.w = node->current_strength;

    pthread_mutex_unlock(&ladder->mutex);

    return Z_LADDER_SUCCESS;
}

float z_ladder_emotional_boost(z_ladder_t ladder, uint64_t node_id, float valence) {
    if (!ladder) {
        return -1.0f;
    }

    pthread_mutex_lock(&ladder->mutex);

    z_hash_entry_t* entry = hash_find(ladder, node_id);
    if (!entry) {
        pthread_mutex_unlock(&ladder->mutex);
        return -1.0f;
    }

    pr_memory_node_t* node = entry->node;

    // Update emotional valence
    valence = clampf(valence, -1.0f, 1.0f);
    node->state.x = valence;

    // Boost strength based on emotional intensity (both positive and negative)
    float boost = absf(valence) * EMOTIONAL_BOOST_FACTOR;
    node->current_strength += boost;
    node->current_strength = clampf(node->current_strength, 0.0f, 1.0f);
    node->state.w = node->current_strength;

    // Also boost salience for emotional memories
    node->state.y += absf(valence) * 0.1f;
    node->state.y = clampf(node->state.y, 0.0f, 1.0f);

    float result = node->current_strength;

    pthread_mutex_unlock(&ladder->mutex);

    return result;
}

//=============================================================================
// Statistics API Implementation
//=============================================================================

z_ladder_error_t z_ladder_get_stats(z_ladder_t ladder, z_ladder_stats_t* stats) {
    if (!ladder || !stats) {
        return Z_LADDER_ERROR_NULL_POINTER;
    }

    pthread_mutex_lock(&ladder->mutex);

    memset(stats, 0, sizeof(z_ladder_stats_t));

    // Per-tier statistics
    for (int i = 0; i < Z_LADDER_NUM_TIERS; i++) {
        z_tier_storage_t* tier = &ladder->tiers[i];

        stats->tier_counts[i] = tier->count;
        stats->tier_capacities[i] = tier->config.capacity;

        if (tier->count > 0) {
            float sum = 0.0f;
            float min_s = tier->nodes[0]->current_strength;
            float max_s = tier->nodes[0]->current_strength;

            for (size_t j = 0; j < tier->count; j++) {
                float s = tier->nodes[j]->current_strength;
                sum += s;
                if (s < min_s) min_s = s;
                if (s > max_s) max_s = s;
            }

            stats->avg_strength[i] = sum / (float)tier->count;
            stats->min_strength[i] = min_s;
            stats->max_strength[i] = max_s;
        }
    }

    // Total counts
    stats->total_nodes = ladder->total_nodes;

    // Transition counts
    memcpy(stats->promotions, ladder->promotions, sizeof(stats->promotions));
    memcpy(stats->demotions, ladder->demotions, sizeof(stats->demotions));
    memcpy(stats->evictions, ladder->evictions, sizeof(stats->evictions));

    // Time statistics
    stats->last_decay_time_ms = ladder->last_decay_time_ms;
    stats->last_consolidation_time_ms = ladder->last_consolidation_time_ms;
    stats->total_consolidations = ladder->total_consolidations;

    // Memory usage estimate
    size_t mem = sizeof(struct z_ladder_struct);
    for (int i = 0; i < Z_LADDER_NUM_TIERS; i++) {
        mem += ladder->tiers[i].capacity * sizeof(pr_memory_node_t*);
    }
    if (ladder->events) {
        mem += ladder->event_capacity * sizeof(z_consolidation_event_t);
    }
    stats->memory_bytes = mem;

    pthread_mutex_unlock(&ladder->mutex);

    return Z_LADDER_SUCCESS;
}

void z_ladder_print_summary(z_ladder_t ladder) {
    if (!ladder) {
        printf("Z-Ladder: NULL\n");
        return;
    }

    z_ladder_stats_t stats;
    z_ladder_get_stats(ladder, &stats);

    printf("============================================\n");
    printf("           Z-Ladder Memory Summary\n");
    printf("============================================\n");
    printf("Total Nodes: %zu\n", stats.total_nodes);
    printf("Total Consolidations: %lu\n", (unsigned long)stats.total_consolidations);
    printf("\n");

    printf("Tier Distribution:\n");
    printf("  +------+--------+----------+----------+----------+\n");
    printf("  | Tier | Count  | Capacity | Avg Str  | Min-Max  |\n");
    printf("  +------+--------+----------+----------+----------+\n");

    const char* tier_names[] = {"Z0", "Z1", "Z2", "Z3"};
    for (int i = 0; i < Z_LADDER_NUM_TIERS; i++) {
        const char* cap_str = (stats.tier_capacities[i] == 0) ? "   inf" : "";
        if (stats.tier_capacities[i] > 0) {
            printf("  | %s   | %6zu | %8zu | %8.3f | %.2f-%.2f |\n",
                   tier_names[i],
                   stats.tier_counts[i],
                   stats.tier_capacities[i],
                   stats.avg_strength[i],
                   stats.min_strength[i],
                   stats.max_strength[i]);
        } else {
            printf("  | %s   | %6zu |   inf    | %8.3f | %.2f-%.2f |\n",
                   tier_names[i],
                   stats.tier_counts[i],
                   stats.avg_strength[i],
                   stats.min_strength[i],
                   stats.max_strength[i]);
        }
    }
    printf("  +------+--------+----------+----------+----------+\n");
    printf("\n");

    printf("Transitions:\n");
    printf("  Promotions:  Z0->Z1: %lu  Z1->Z2: %lu  Z2->Z3: %lu\n",
           (unsigned long)stats.promotions[0],
           (unsigned long)stats.promotions[1],
           (unsigned long)stats.promotions[2]);
    printf("  Demotions:   Z1->Z0: %lu  Z2->Z1: %lu  Z3->Z2: %lu\n",
           (unsigned long)stats.demotions[0],
           (unsigned long)stats.demotions[1],
           (unsigned long)stats.demotions[2]);
    printf("  Evictions:   Z0: %lu  Z1: %lu  Z2: %lu  Z3: %lu\n",
           (unsigned long)stats.evictions[0],
           (unsigned long)stats.evictions[1],
           (unsigned long)stats.evictions[2],
           (unsigned long)stats.evictions[3]);
    printf("\n");

    printf("Memory Usage: ~%zu bytes\n", stats.memory_bytes);
    printf("============================================\n");
}

void z_ladder_reset_stats(z_ladder_t ladder) {
    if (!ladder) {
        return;
    }

    pthread_mutex_lock(&ladder->mutex);

    memset(ladder->promotions, 0, sizeof(ladder->promotions));
    memset(ladder->demotions, 0, sizeof(ladder->demotions));
    memset(ladder->evictions, 0, sizeof(ladder->evictions));
    ladder->total_consolidations = 0;

    pthread_mutex_unlock(&ladder->mutex);
}

//=============================================================================
// Callback API Implementation
//=============================================================================

z_ladder_error_t z_ladder_set_promotion_callback(z_ladder_t ladder,
                                                  z_promotion_callback_t callback,
                                                  void* user_data) {
    if (!ladder) {
        return Z_LADDER_ERROR_NULL_POINTER;
    }

    pthread_mutex_lock(&ladder->mutex);

    ladder->promotion_callback = callback;
    ladder->promotion_user_data = user_data;

    pthread_mutex_unlock(&ladder->mutex);

    return Z_LADDER_SUCCESS;
}

z_ladder_error_t z_ladder_set_eviction_callback(z_ladder_t ladder,
                                                 z_eviction_callback_t callback,
                                                 void* user_data) {
    if (!ladder) {
        return Z_LADDER_ERROR_NULL_POINTER;
    }

    pthread_mutex_lock(&ladder->mutex);

    ladder->eviction_callback = callback;
    ladder->eviction_user_data = user_data;

    pthread_mutex_unlock(&ladder->mutex);

    return Z_LADDER_SUCCESS;
}

//=============================================================================
// Utility Functions Implementation
//=============================================================================

const char* z_ladder_error_string(z_ladder_error_t error) {
    switch (error) {
        case Z_LADDER_SUCCESS:
            return "Success";
        case Z_LADDER_ERROR_NULL_POINTER:
            return "Null pointer";
        case Z_LADDER_ERROR_INVALID_TIER:
            return "Invalid tier";
        case Z_LADDER_ERROR_NOT_FOUND:
            return "Node not found";
        case Z_LADDER_ERROR_ALREADY_EXISTS:
            return "Node already exists";
        case Z_LADDER_ERROR_CAPACITY:
            return "Capacity exceeded";
        case Z_LADDER_ERROR_NO_MEMORY:
            return "Memory allocation failed";
        case Z_LADDER_ERROR_LOCKED:
            return "Node is locked";
        case Z_LADDER_ERROR_ALREADY_TOP:
            return "Already at top tier (Z3)";
        case Z_LADDER_ERROR_ALREADY_BOTTOM:
            return "Already at bottom tier (Z0)";
        case Z_LADDER_ERROR_INVALID_CONFIG:
            return "Invalid configuration";
        default:
            return "Unknown error";
    }
}

const char* z_ladder_tier_name(pr_memory_tier_t tier) {
    switch (tier) {
        case PR_MEMORY_TIER_Z0:
            return "Working Memory (Z0)";
        case PR_MEMORY_TIER_Z1:
            return "Short-Term (Z1)";
        case PR_MEMORY_TIER_Z2:
            return "Long-Term (Z2)";
        case PR_MEMORY_TIER_Z3:
            return "Permanent (Z3)";
        default:
            return "Unknown Tier";
    }
}

const char* z_ladder_eviction_policy_name(z_eviction_policy_t policy) {
    switch (policy) {
        case Z_EVICT_WEAKEST:
            return "Weakest";
        case Z_EVICT_OLDEST:
            return "Oldest";
        case Z_EVICT_LRU:
            return "Least Recently Used";
        case Z_EVICT_LFU:
            return "Least Frequently Used";
        case Z_EVICT_COMBINED:
            return "Combined Score";
        default:
            return "Unknown";
    }
}

uint64_t z_ladder_current_time_ms(void) {
    return get_current_time_ms();
}

bool z_ladder_validate(z_ladder_t ladder) {
    if (!ladder) {
        return false;
    }

    pthread_mutex_lock(&ladder->mutex);

    bool valid = true;
    size_t counted_total = 0;

    // Validate tier storage
    for (int i = 0; i < Z_LADDER_NUM_TIERS; i++) {
        z_tier_storage_t* tier = &ladder->tiers[i];

        // Check count vs capacity
        if (tier->count > tier->capacity) {
            valid = false;
            break;
        }

        counted_total += tier->count;

        // Verify all nodes in tier
        for (size_t j = 0; j < tier->count; j++) {
            pr_memory_node_t* node = tier->nodes[j];

            if (!node) {
                valid = false;
                break;
            }

            // Verify node's tier matches
            if (node->tier != (pr_memory_tier_t)i) {
                valid = false;
                break;
            }

            // Verify hash entry exists
            z_hash_entry_t* entry = hash_find(ladder, node->node_id);
            if (!entry || entry->node != node || entry->tier != (pr_memory_tier_t)i) {
                valid = false;
                break;
            }
        }

        if (!valid) break;
    }

    // Verify total count
    if (valid && counted_total != ladder->total_nodes) {
        valid = false;
    }

    pthread_mutex_unlock(&ladder->mutex);

    return valid;
}
