//=============================================================================
// nimcp_attention_wm_bridge.c - Attention-Working Memory Integration Bridge
//=============================================================================
/**
 * @file nimcp_attention_wm_bridge.c
 * @brief Bidirectional integration between attention and working memory systems
 *
 * WHAT: Attention-gated working memory entry and priority management
 * WHY:  Attention gates entry into WM; focus shifts update item priorities
 * HOW:  Threshold-based gating, capacity management, priority decay
 *
 * BIOLOGICAL BASIS:
 * - Prefrontal cortex maintains WM representations via sustained attention
 * - Cowan's embedded-processes model: attention selects WM contents
 * - Basal ganglia gates WM updates based on attentional signals
 *
 * @author NIMCP Development Team
 * @date 2025-12-31
 * @version 1.0.0
 */

#include "cognitive/integration/nimcp_attention_wm_bridge.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/time/nimcp_time.h"
#include <string.h>

//=============================================================================
// Constants
//=============================================================================

#define DEFAULT_CAPACITY_LIMIT     7      // Miller's magical number
#define DEFAULT_ATTENTION_THRESHOLD 0.3f
#define DEFAULT_DECAY_RATE         0.1f
#define PRIORITY_BOOST_FACTOR      1.2f   // Boost when gaining focus

//=============================================================================
// Internal Structures
//=============================================================================

/**
 * @brief Internal working memory item representation
 */
typedef struct wm_item {
    uint64_t item_id;              /**< Unique item identifier */
    float attention_strength;      /**< Attention when gated in */
    float priority;                /**< Current priority [0, 1] */
    uint64_t entry_time;           /**< When item entered WM */
    uint64_t last_access_time;     /**< Last time item was accessed */
    bool valid;                    /**< Whether slot is occupied */
} wm_item_t;

/**
 * @brief Opaque attention-WM bridge structure
 */
struct attention_wm_bridge {
    attention_wm_config_t config;       /**< Configuration parameters */
    wm_item_t* items;                   /**< Working memory items */
    size_t item_capacity;               /**< Maximum items (from config) */
    size_t item_count;                  /**< Current item count */
    uint64_t current_focus;             /**< Currently focused item ID */
    attention_wm_stats_t stats;         /**< Bridge statistics */
    nimcp_mutex_t* mutex;               /**< Thread safety mutex */
    bool initialized;                   /**< Initialization flag */
};

//=============================================================================
// Static Helpers
//=============================================================================

/**
 * @brief Find item index by ID (must be called with mutex held)
 */
static int find_item_unlocked(const attention_wm_bridge_t* bridge, uint64_t item_id) {
    if (!bridge || !bridge->items) {
        return -1;
    }

    for (size_t i = 0; i < bridge->item_capacity; i++) {
        if (bridge->items[i].valid && bridge->items[i].item_id == item_id) {
            return (int)i;
        }
    }
    return -1;
}

/**
 * @brief Find lowest priority item index (must be called with mutex held)
 */
static int find_lowest_priority_unlocked(const attention_wm_bridge_t* bridge) {
    if (!bridge || !bridge->items || bridge->item_count == 0) {
        return -1;
    }

    int lowest_idx = -1;
    float lowest_priority = 2.0f;  // Higher than max priority

    for (size_t i = 0; i < bridge->item_capacity; i++) {
        if (bridge->items[i].valid && bridge->items[i].priority < lowest_priority) {
            lowest_priority = bridge->items[i].priority;
            lowest_idx = (int)i;
        }
    }
    return lowest_idx;
}

/**
 * @brief Find first free slot (must be called with mutex held)
 */
static int find_free_slot_unlocked(const attention_wm_bridge_t* bridge) {
    if (!bridge || !bridge->items) {
        return -1;
    }

    for (size_t i = 0; i < bridge->item_capacity; i++) {
        if (!bridge->items[i].valid) {
            return (int)i;
        }
    }
    return -1;
}

/**
 * @brief Clamp value to range [min, max]
 */
static float clamp_priority(float value) {
    if (value < ATTENTION_WM_MIN_PRIORITY) {
        return ATTENTION_WM_MIN_PRIORITY;
    }
    if (value > ATTENTION_WM_MAX_PRIORITY) {
        return ATTENTION_WM_MAX_PRIORITY;
    }
    return value;
}

/**
 * @brief Compute average priority (must be called with mutex held)
 */
static float compute_avg_priority_unlocked(const attention_wm_bridge_t* bridge) {
    if (!bridge || bridge->item_count == 0) {
        return 0.0f;
    }

    float sum = 0.0f;
    for (size_t i = 0; i < bridge->item_capacity; i++) {
        if (bridge->items[i].valid) {
            sum += bridge->items[i].priority;
        }
    }
    return sum / (float)bridge->item_count;
}

//=============================================================================
// Lifecycle API
//=============================================================================

int attention_wm_bridge_default_config(attention_wm_config_t* config) {
    if (!config) {
        return -1;
    }

    config->capacity_limit = DEFAULT_CAPACITY_LIMIT;
    config->attention_threshold = DEFAULT_ATTENTION_THRESHOLD;
    config->decay_rate = DEFAULT_DECAY_RATE;

    return 0;
}

attention_wm_bridge_t* attention_wm_bridge_create(
    const attention_wm_config_t* config
) {
    // Allocate bridge structure
    attention_wm_bridge_t* bridge = nimcp_calloc(1, sizeof(attention_wm_bridge_t));
    if (!bridge) {
        return NULL;
    }

    // Apply configuration
    if (config) {
        bridge->config = *config;
    } else {
        attention_wm_bridge_default_config(&bridge->config);
    }

    // Validate and bound capacity
    if (bridge->config.capacity_limit == 0) {
        bridge->config.capacity_limit = DEFAULT_CAPACITY_LIMIT;
    }
    if (bridge->config.capacity_limit > ATTENTION_WM_MAX_ITEMS) {
        bridge->config.capacity_limit = ATTENTION_WM_MAX_ITEMS;
    }

    bridge->item_capacity = bridge->config.capacity_limit;

    // Allocate items array
    bridge->items = nimcp_calloc(bridge->item_capacity, sizeof(wm_item_t));
    if (!bridge->items) {
        nimcp_free(bridge);
        return NULL;
    }

    // Initialize all items as invalid
    for (size_t i = 0; i < bridge->item_capacity; i++) {
        bridge->items[i].valid = false;
    }

    // Create mutex
    bridge->mutex = nimcp_mutex_create(NULL);
    if (!bridge->mutex) {
        nimcp_free(bridge->items);
        nimcp_free(bridge);
        return NULL;
    }

    // Initialize state
    bridge->item_count = 0;
    bridge->current_focus = 0;

    // Initialize stats
    memset(&bridge->stats, 0, sizeof(attention_wm_stats_t));

    bridge->initialized = true;

    return bridge;
}

void attention_wm_bridge_destroy(attention_wm_bridge_t* bridge) {
    if (!bridge) {
        return;
    }

    // Free items array
    if (bridge->items) {
        nimcp_free(bridge->items);
        bridge->items = NULL;
    }

    // Destroy mutex
    if (bridge->mutex) {
        nimcp_mutex_free(bridge->mutex);
        bridge->mutex = NULL;
    }

    bridge->initialized = false;

    // Free bridge
    nimcp_free(bridge);
}

//=============================================================================
// Attention -> WM Direction
//=============================================================================

int attention_wm_gate_entry(
    attention_wm_bridge_t* bridge,
    uint64_t item_id,
    float attention_strength
) {
    if (!bridge || !bridge->initialized) {
        return -1;
    }

    nimcp_mutex_lock(bridge->mutex);

    // Check attention threshold - reject if below threshold
    if (attention_strength < bridge->config.attention_threshold) {
        nimcp_mutex_unlock(bridge->mutex);
        return -1;  // Rejected due to insufficient attention
    }

    // Check if item already exists
    int existing_idx = find_item_unlocked(bridge, item_id);
    if (existing_idx >= 0) {
        // Update existing item
        bridge->items[existing_idx].attention_strength = attention_strength;
        bridge->items[existing_idx].priority = clamp_priority(attention_strength);
        bridge->items[existing_idx].last_access_time = nimcp_time_get_ms();
        nimcp_mutex_unlock(bridge->mutex);
        return 0;
    }

    // Find free slot or evict if at capacity
    int slot_idx = find_free_slot_unlocked(bridge);

    if (slot_idx < 0) {
        // At capacity - evict lowest priority item
        slot_idx = find_lowest_priority_unlocked(bridge);
        if (slot_idx < 0) {
            nimcp_mutex_unlock(bridge->mutex);
            return -1;  // Should not happen, but safety check
        }

        // Mark evicted item as invalid
        bridge->items[slot_idx].valid = false;
        bridge->item_count--;
        bridge->stats.items_evicted++;
    }

    // Add new item
    uint64_t now = nimcp_time_get_ms();
    bridge->items[slot_idx].item_id = item_id;
    bridge->items[slot_idx].attention_strength = attention_strength;
    bridge->items[slot_idx].priority = clamp_priority(attention_strength);
    bridge->items[slot_idx].entry_time = now;
    bridge->items[slot_idx].last_access_time = now;
    bridge->items[slot_idx].valid = true;

    bridge->item_count++;
    bridge->stats.items_gated_in++;

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int attention_wm_on_focus_shift(
    attention_wm_bridge_t* bridge,
    uint64_t old_focus,
    uint64_t new_focus
) {
    if (!bridge || !bridge->initialized) {
        return -1;
    }

    nimcp_mutex_lock(bridge->mutex);

    // Apply decay to old focus item (if it exists and is not 0)
    if (old_focus != 0) {
        int old_idx = find_item_unlocked(bridge, old_focus);
        if (old_idx >= 0) {
            float decayed = bridge->items[old_idx].priority * (1.0f - bridge->config.decay_rate);
            bridge->items[old_idx].priority = clamp_priority(decayed);
            bridge->items[old_idx].last_access_time = nimcp_time_get_ms();
        }
    }

    // Boost priority of new focus item
    int new_idx = find_item_unlocked(bridge, new_focus);
    if (new_idx >= 0) {
        float boosted = bridge->items[new_idx].priority * PRIORITY_BOOST_FACTOR;
        bridge->items[new_idx].priority = clamp_priority(boosted);
        bridge->items[new_idx].last_access_time = nimcp_time_get_ms();
    }

    // Update current focus
    bridge->current_focus = new_focus;
    bridge->stats.focus_shifts++;

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

//=============================================================================
// WM Management
//=============================================================================

int attention_wm_update_priority(
    attention_wm_bridge_t* bridge,
    uint64_t item_id,
    float new_priority
) {
    if (!bridge || !bridge->initialized) {
        return -1;
    }

    nimcp_mutex_lock(bridge->mutex);

    int idx = find_item_unlocked(bridge, item_id);
    if (idx < 0) {
        nimcp_mutex_unlock(bridge->mutex);
        return -1;  // Item not found
    }

    bridge->items[idx].priority = clamp_priority(new_priority);
    bridge->items[idx].last_access_time = nimcp_time_get_ms();
    bridge->stats.priority_updates++;

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

//=============================================================================
// Query API
//=============================================================================

int attention_wm_get_attended_items(
    attention_wm_bridge_t* bridge,
    attention_wm_item_t items[],
    size_t max_count
) {
    if (!bridge || !bridge->initialized || !items || max_count == 0) {
        return -1;
    }

    nimcp_mutex_lock(bridge->mutex);

    int count = 0;
    for (size_t i = 0; i < bridge->item_capacity && (size_t)count < max_count; i++) {
        if (bridge->items[i].valid &&
            bridge->items[i].attention_strength >= bridge->config.attention_threshold) {

            items[count].item_id = bridge->items[i].item_id;
            items[count].priority = bridge->items[i].priority;
            items[count].attention_strength = bridge->items[i].attention_strength;
            items[count].entry_time = bridge->items[i].entry_time;
            items[count].last_access_time = bridge->items[i].last_access_time;
            count++;
        }
    }

    nimcp_mutex_unlock(bridge->mutex);
    return count;
}

//=============================================================================
// Stats API
//=============================================================================

int attention_wm_bridge_get_stats(
    const attention_wm_bridge_t* bridge,
    attention_wm_stats_t* stats
) {
    if (!bridge || !bridge->initialized || !stats) {
        return -1;
    }

    // Cast away const to acquire mutex (thread-safe read)
    attention_wm_bridge_t* mutable_bridge = (attention_wm_bridge_t*)bridge;

    nimcp_mutex_lock(mutable_bridge->mutex);

    // Copy stats
    memcpy(stats, &bridge->stats, sizeof(attention_wm_stats_t));

    // Update dynamic fields
    stats->current_item_count = bridge->item_count;
    stats->avg_priority = compute_avg_priority_unlocked(bridge);

    nimcp_mutex_unlock(mutable_bridge->mutex);
    return 0;
}
