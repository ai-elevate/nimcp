/**
 * @file nimcp_working_memory.c
 * @brief Working memory implementation with temporal decay and attention refresh
 *
 * WHAT: Miller's 7±2 working memory buffer with salience-based eviction
 * WHY:  Maintain active representations for reasoning, planning, and decision-making
 * HOW:  Dynamic buffer with exponential decay, attention refresh, and priority eviction
 *
 * BIOLOGICAL BASIS:
 * - Prefrontal cortex (PFC) maintains ~7 items in active state
 * - Exponential decay without rehearsal (τ ≈ 1-2 seconds)
 * - Attention refresh prevents decay (frontal-parietal networks)
 * - Salience determines eviction priority (thalamic gating)
 *
 * PHASE: 10.2 (Working Memory)
 * DEPENDENCIES: None (standalone module)
 * TRAINING_IMPACT: None (inference-only, no weight modification)
 *
 * @author Claude Code
 * @date 2025-11
 */

#include "cognitive/nimcp_working_memory.h"
#include "utils/time/nimcp_time.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

// ============================================================================
// CONSTANTS
// ============================================================================

#define MAX_ITEM_SIZE_BYTES (1024 * 1024)  // 1MB max per item
#define MIN_CAPACITY 1
#define MAX_CAPACITY 32  // Pathological cases beyond 7±2

// ============================================================================
// INTERNAL STRUCTURE
// ============================================================================

/**
 * @brief Internal working memory structure
 *
 * WHAT: Complete working memory state with items, metadata, and statistics
 * WHY:  Encapsulate all data needed for temporal decay and eviction
 * HOW:  Parallel arrays for items, salience, timestamps, and attention flags
 */
struct working_memory {
    // Item storage
    float** items;                  // Array of item pointers
    uint32_t* item_sizes;           // Size of each item in floats

    // Capacity management
    uint32_t capacity;              // Maximum items (default: 7)
    uint32_t current_size;          // Current item count

    // Metadata
    float* salience;                // Importance scores [0.0, 1.0]
    uint64_t* timestamps;           // Last access time (ms)
    bool* attention_refreshed;      // Rehearsal flag (prevents decay)

    // Emotional tagging (Phase 10.3)
    emotional_tag_t* emotions;      // Emotional context for each item
    bool* has_emotion;              // Whether item has emotional tag

    // Configuration
    float decay_tau_ms;             // Decay time constant (default: 1000ms)
    float min_salience;             // Eviction threshold (default: 0.01)
    bool enable_attention_refresh;  // Allow rehearsal to prevent decay
    bool enable_temporal_decay;     // Enable exponential decay

    // Statistics
    uint32_t total_additions;       // Lifetime item additions
    uint32_t total_evictions;       // Lifetime evictions
    uint32_t total_refreshes;       // Lifetime attention refreshes
    uint32_t total_decay_removals;  // Items removed by decay
};

// ============================================================================
// ERROR HANDLING
// ============================================================================

static char last_error[256] = {0};

static void set_error(const char* msg) {
    snprintf(last_error, sizeof(last_error), "%s", msg);
}

const char* working_memory_get_last_error(void) {
    return last_error;
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * @brief Find index of item with lowest salience
 *
 * WHAT: Search for lowest-priority item for eviction
 * WHY:  Evict least important item when buffer is full
 * HOW:  Linear scan with min tracking
 *
 * COMPLEXITY: O(n) where n = current_size
 *
 * @param wm Working memory instance
 * @return Index of lowest salience item, or -1 if empty
 */
static int find_lowest_salience_index(const working_memory_t* wm) {
    // Guard: Empty buffer
    if (wm->current_size == 0) {
        return -1;
    }

    int min_index = 0;
    float min_salience = wm->salience[0];

    for (uint32_t i = 1; i < wm->current_size; i++) {
        if (wm->salience[i] < min_salience) {
            min_salience = wm->salience[i];
            min_index = i;
        }
    }

    return min_index;
}

/**
 * @brief Evict item at specific index
 *
 * WHAT: Remove item and compact buffer
 * WHY:  Make space for new item
 * HOW:  Free memory → Shift arrays left → Decrement size
 *
 * COMPLEXITY: O(n) where n = current_size (due to shift)
 *
 * @param wm Working memory instance
 * @param index Index to evict
 */
static void evict_item_at_index(working_memory_t* wm, uint32_t index) {
    // Guard: Invalid index
    if (index >= wm->current_size) {
        return;
    }

    // Free item memory
    free(wm->items[index]);
    wm->items[index] = NULL;

    // Shift arrays left
    uint32_t shift_count = wm->current_size - index - 1;
    if (shift_count > 0) {
        memmove(&wm->items[index], &wm->items[index + 1],
                shift_count * sizeof(float*));
        memmove(&wm->item_sizes[index], &wm->item_sizes[index + 1],
                shift_count * sizeof(uint32_t));
        memmove(&wm->salience[index], &wm->salience[index + 1],
                shift_count * sizeof(float));
        memmove(&wm->timestamps[index], &wm->timestamps[index + 1],
                shift_count * sizeof(uint64_t));
        memmove(&wm->attention_refreshed[index],
                &wm->attention_refreshed[index + 1],
                shift_count * sizeof(bool));
        memmove(&wm->emotions[index], &wm->emotions[index + 1],  // Phase 10.3
                shift_count * sizeof(emotional_tag_t));
        memmove(&wm->has_emotion[index], &wm->has_emotion[index + 1],  // Phase 10.3
                shift_count * sizeof(bool));
    }

    wm->current_size--;
    wm->total_evictions++;
}

/**
 * @brief Get current time in milliseconds
 *
 * WHAT: System clock for temporal decay
 * WHY:  Track item age for exponential decay
 * HOW:  Use monotonic clock for consistent timing
 *
 * @return Current time in milliseconds
 *
 * COMPLEXITY: O(1) - direct system call
 */
static uint64_t get_current_time_ms(void) {
    return nimcp_time_monotonic_ms();
}

// ============================================================================
// LIFECYCLE FUNCTIONS
// ============================================================================

/**
 * @brief Get default working memory configuration
 *
 * WHAT: Standard configuration matching biological constraints
 * WHY:  Provide sensible defaults (Miller's 7±2, 1s decay)
 * HOW:  Initialize struct with empirically validated values
 *
 * @return Default configuration
 */
working_memory_config_t working_memory_default_config(void) {
    working_memory_config_t config = {
        .capacity = WORKING_MEMORY_DEFAULT_CAPACITY,  // 7
        .decay_tau_ms = WORKING_MEMORY_DECAY_TAU_MS,  // 1000ms
        .min_salience = WORKING_MEMORY_MIN_SALIENCE,  // 0.01
        .enable_attention_refresh = true,
        .enable_temporal_decay = true
    };
    return config;
}

/**
 * @brief Create working memory with default configuration
 *
 * WHAT: Allocate and initialize working memory buffer
 * WHY:  Provide simple creation for standard use cases
 * HOW:  Delegate to custom creation with default config
 *
 * COMPLEXITY: O(capacity) for array allocation
 * MEMORY: ~capacity × (ptr + uint32 + float + uint64 + bool) bytes
 *
 * @return New working memory instance, or NULL on allocation failure
 */
working_memory_t* working_memory_create(void) {
    working_memory_config_t config = working_memory_default_config();
    return working_memory_create_custom(&config);
}

/**
 * @brief Create working memory with custom configuration
 *
 * WHAT: Allocate and initialize working memory with custom parameters
 * WHY:  Allow experimentation with non-standard capacities and decay
 * HOW:  Validate config → Allocate struct → Allocate arrays → Initialize
 *
 * COMPLEXITY: O(capacity) for array allocation
 * MEMORY: capacity × (24 bytes per item + item data)
 *
 * @param config Configuration parameters (non-NULL)
 * @return New working memory instance, or NULL on error
 */
working_memory_t* working_memory_create_custom(
    const working_memory_config_t* config
)
{
    // Guard: NULL config
    if (!config) {
        set_error("NULL config");
        return NULL;
    }

    // Guard: Invalid capacity
    if (config->capacity < MIN_CAPACITY || config->capacity > MAX_CAPACITY) {
        set_error("Invalid capacity (must be 1-32)");
        return NULL;
    }

    // Guard: Invalid decay tau
    if (config->decay_tau_ms <= 0.0f) {
        set_error("Invalid decay_tau_ms (must be > 0)");
        return NULL;
    }

    // Allocate main structure
    working_memory_t* wm = calloc(1, sizeof(working_memory_t));
    if (!wm) {
        set_error("Failed to allocate working_memory_t");
        return NULL;
    }

    // Allocate arrays
    wm->items = calloc(config->capacity, sizeof(float*));
    wm->item_sizes = calloc(config->capacity, sizeof(uint32_t));
    wm->salience = calloc(config->capacity, sizeof(float));
    wm->timestamps = calloc(config->capacity, sizeof(uint64_t));
    wm->attention_refreshed = calloc(config->capacity, sizeof(bool));
    wm->emotions = calloc(config->capacity, sizeof(emotional_tag_t));  // Phase 10.3
    wm->has_emotion = calloc(config->capacity, sizeof(bool));          // Phase 10.3

    // Check all allocations
    if (!wm->items || !wm->item_sizes || !wm->salience ||
        !wm->timestamps || !wm->attention_refreshed ||
        !wm->emotions || !wm->has_emotion) {
        set_error("Failed to allocate arrays");
        working_memory_destroy(wm);
        return NULL;
    }

    // Initialize configuration
    wm->capacity = config->capacity;
    wm->current_size = 0;
    wm->decay_tau_ms = config->decay_tau_ms;
    wm->min_salience = config->min_salience;
    wm->enable_attention_refresh = config->enable_attention_refresh;
    wm->enable_temporal_decay = config->enable_temporal_decay;

    // Initialize statistics
    wm->total_additions = 0;
    wm->total_evictions = 0;
    wm->total_refreshes = 0;
    wm->total_decay_removals = 0;

    return wm;
}

/**
 * @brief Destroy working memory and free all resources
 *
 * WHAT: Free all allocated memory
 * WHY:  Prevent memory leaks
 * HOW:  Free items → Free arrays → Free struct
 *
 * COMPLEXITY: O(n) where n = current_size
 *
 * @param wm Working memory instance (nullable)
 */
void working_memory_destroy(working_memory_t* wm) {
    // Guard: NULL pointer (safe to call on NULL)
    if (!wm) {
        return;
    }

    // Free all items
    if (wm->items) {
        for (uint32_t i = 0; i < wm->current_size; i++) {
            free(wm->items[i]);
        }
        free(wm->items);
    }

    // Free arrays
    free(wm->item_sizes);
    free(wm->salience);
    free(wm->timestamps);
    free(wm->attention_refreshed);
    free(wm->emotions);      // Phase 10.3
    free(wm->has_emotion);   // Phase 10.3

    // Free main structure
    free(wm);
}

// ============================================================================
// ITEM MANAGEMENT
// ============================================================================

/**
 * @brief Add item to working memory with salience-based eviction
 *
 * WHAT: Insert new item into buffer, evicting if necessary
 * WHY:  Maintain active representations for reasoning
 * HOW:  Validate → Check capacity → Evict if full → Copy item → Store metadata
 *
 * COMPLEXITY: O(n) where n = capacity (eviction search)
 * MEMORY: Allocates item_size × sizeof(float) bytes
 *
 * @param wm Working memory instance (non-NULL)
 * @param item Item data array (non-NULL)
 * @param item_size Size of item in floats (> 0)
 * @param salience Importance [0.0, 1.0] for eviction priority
 * @return true on success, false on error
 */
bool working_memory_add(
    working_memory_t* wm,
    const float* item,
    uint32_t item_size,
    float salience
)
{
    // Guard: NULL working memory
    if (!wm) {
        set_error("NULL working_memory_t");
        return false;
    }

    // Guard: NULL item
    if (!item) {
        set_error("NULL item");
        return false;
    }

    // Guard: Invalid size
    if (item_size == 0) {
        set_error("item_size must be > 0");
        return false;
    }

    // Guard: Size overflow check
    if (item_size > (MAX_ITEM_SIZE_BYTES / sizeof(float))) {
        set_error("item_size exceeds maximum");
        return false;
    }

    // Guard: Invalid salience
    if (salience < 0.0f || salience > 1.0f) {
        set_error("salience must be in [0.0, 1.0]");
        return false;
    }

    // Check if full → evict lowest salience item
    if (wm->current_size >= wm->capacity) {
        int evict_index = find_lowest_salience_index(wm);
        if (evict_index >= 0) {
            evict_item_at_index(wm, evict_index);
        }
    }

    // Allocate and copy item
    float* item_copy = malloc(item_size * sizeof(float));
    if (!item_copy) {
        set_error("Failed to allocate item memory");
        return false;
    }
    memcpy(item_copy, item, item_size * sizeof(float));

    // Insert at end
    uint32_t index = wm->current_size;
    wm->items[index] = item_copy;
    wm->item_sizes[index] = item_size;
    wm->salience[index] = salience;
    wm->timestamps[index] = get_current_time_ms();
    wm->attention_refreshed[index] = false;
    wm->has_emotion[index] = false;  // Phase 10.3: No emotion by default

    wm->current_size++;
    wm->total_additions++;

    return true;
}

/**
 * @brief Add item to working memory with emotional tag (Phase 10.3)
 *
 * WHAT: Insert new item with emotional context for enhanced salience
 * WHY:  Emotional events receive memory priority (biological)
 * HOW:  Store emotional tag → Compute boosted salience → Add item
 *
 * COMPLEXITY: O(n) where n = capacity (eviction search)
 *
 * @param wm Working memory instance (non-NULL)
 * @param item Item data array (non-NULL)
 * @param item_size Size of item in floats (> 0)
 * @param base_salience Base importance [0.0, 1.0]
 * @param emotion Emotional tag (non-NULL)
 * @return true on success, false on error
 */
bool working_memory_add_with_emotion(
    working_memory_t* wm,
    const float* item,
    uint32_t item_size,
    float base_salience,
    const emotional_tag_t* emotion
)
{
    // Guard: NULL emotion
    if (!emotion) {
        set_error("NULL emotional_tag_t");
        return false;
    }

    // Guard: Invalid emotion
    if (!emotional_tag_is_valid(emotion)) {
        set_error("Invalid emotional tag");
        return false;
    }

    // Compute emotional salience boost
    float emotional_boost = emotional_compute_salience_boost(emotion);
    float total_salience = base_salience * emotional_boost;

    // Clamp to valid range
    if (total_salience > 1.0f) {
        total_salience = 1.0f;
    }

    // Add item with boosted salience
    if (!working_memory_add(wm, item, item_size, total_salience)) {
        return false;
    }

    // Attach emotional tag to the just-added item
    uint32_t index = wm->current_size - 1;  // Last added
    wm->emotions[index] = *emotion;
    wm->has_emotion[index] = true;

    return true;
}

/**
 * @brief Get item from working memory without removing
 *
 * WHAT: Read-only access to item by index
 * WHY:  Allow inspection of working memory contents
 * HOW:  Validate → Return pointer to internal data
 *
 * COMPLEXITY: O(1)
 *
 * @param wm Working memory instance (non-NULL)
 * @param index Item index [0, current_size)
 * @param size Output parameter for item size (nullable)
 * @return Pointer to item data, or NULL on error
 */
const float* working_memory_get(
    const working_memory_t* wm,
    uint32_t index,
    uint32_t* size
)
{
    // Guard: NULL working memory
    if (!wm) {
        set_error("NULL working_memory_t");
        return NULL;
    }

    // Guard: Invalid index
    if (index >= wm->current_size) {
        set_error("Index out of bounds");
        return NULL;
    }

    // Set size if requested
    if (size) {
        *size = wm->item_sizes[index];
    }

    return wm->items[index];
}

/**
 * @brief Remove item from working memory
 *
 * WHAT: Delete item at specific index
 * WHY:  Manual removal of irrelevant items
 * HOW:  Validate → Evict → Compact
 *
 * COMPLEXITY: O(n) where n = current_size (shift)
 *
 * @param wm Working memory instance (non-NULL)
 * @param index Item index to remove [0, current_size)
 * @return true on success, false on error
 */
bool working_memory_remove(working_memory_t* wm, uint32_t index) {
    // Guard: NULL working memory
    if (!wm) {
        set_error("NULL working_memory_t");
        return false;
    }

    // Guard: Invalid index
    if (index >= wm->current_size) {
        set_error("Index out of bounds");
        return false;
    }

    evict_item_at_index(wm, index);
    return true;
}

/**
 * @brief Clear all items from working memory
 *
 * WHAT: Remove all items and reset to empty state
 * WHY:  Task switching, context reset
 * HOW:  Free all items → Reset size counter
 *
 * COMPLEXITY: O(n) where n = current_size
 *
 * @param wm Working memory instance (non-NULL)
 */
void working_memory_clear(working_memory_t* wm) {
    // Guard: NULL working memory
    if (!wm) {
        return;
    }

    // Free all items
    for (uint32_t i = 0; i < wm->current_size; i++) {
        free(wm->items[i]);
        wm->items[i] = NULL;
    }

    wm->current_size = 0;
}

/**
 * @brief Get emotional tag of item (Phase 10.3)
 *
 * WHAT: Retrieve emotional context attached to working memory item
 * WHY:  Access emotional state for decision-making and memory retrieval
 * HOW:  Validate → Copy emotional tag to output
 *
 * COMPLEXITY: O(1)
 *
 * @param wm Working memory instance (non-NULL)
 * @param index Item index [0, current_size)
 * @param emotion Output: emotional tag (non-NULL)
 * @return true on success, false on invalid index
 */
bool working_memory_get_emotion(
    const working_memory_t* wm,
    uint32_t index,
    emotional_tag_t* emotion
)
{
    // Guard: NULL working memory
    if (!wm) {
        set_error("NULL working_memory_t");
        return false;
    }

    // Guard: NULL output
    if (!emotion) {
        set_error("NULL emotion output");
        return false;
    }

    // Guard: Invalid index
    if (index >= wm->current_size) {
        set_error("Index out of bounds");
        return false;
    }

    // Copy emotional tag (or neutral if none)
    if (wm->has_emotion[index]) {
        *emotion = wm->emotions[index];
    } else {
        *emotion = emotional_tag_neutral();
    }

    return true;
}

/**
 * @brief Get total salience including emotional boost (Phase 10.3)
 *
 * WHAT: Return effective salience (already boosted if emotion present)
 * WHY:  Priority decisions should use final salience value
 * HOW:  Return stored salience (pre-boosted during add_with_emotion)
 *
 * COMPLEXITY: O(1)
 *
 * DESIGN NOTE:
 * When items are added with working_memory_add_with_emotion(), the
 * emotional boost is computed and applied ONCE during insertion.
 * The stored salience is already the "total" salience.
 * This function is a convenience accessor that returns the stored value.
 *
 * @param wm Working memory instance (non-NULL)
 * @param index Item index [0, current_size)
 * @param total_salience Output: total salience value (non-NULL)
 * @return true on success, false on invalid index
 */
bool working_memory_get_total_salience(
    const working_memory_t* wm,
    uint32_t index,
    float* total_salience
)
{
    // Guard: NULL working memory
    if (!wm) {
        set_error("NULL working_memory_t");
        return false;
    }

    // Guard: NULL output
    if (!total_salience) {
        set_error("NULL total_salience output");
        return false;
    }

    // Guard: Invalid index
    if (index >= wm->current_size) {
        set_error("Index out of bounds");
        return false;
    }

    // Return stored salience (already boosted if emotion present)
    *total_salience = wm->salience[index];

    return true;
}

// ============================================================================
// ATTENTION AND DECAY
// ============================================================================

/**
 * @brief Refresh item via attention (prevent decay)
 *
 * WHAT: Mark item as rehearsed to prevent temporal decay
 * WHY:  Simulate attention-based maintenance (frontal-parietal loop)
 * HOW:  Validate → Update timestamp → Set refresh flag
 *
 * COMPLEXITY: O(1)
 *
 * @param wm Working memory instance (non-NULL)
 * @param index Item index to refresh [0, current_size)
 * @return true on success, false on error
 */
bool working_memory_refresh(working_memory_t* wm, uint32_t index) {
    // Guard: NULL working memory
    if (!wm) {
        set_error("NULL working_memory_t");
        return false;
    }

    // Guard: Invalid index
    if (index >= wm->current_size) {
        set_error("Index out of bounds");
        return false;
    }

    // Guard: Feature disabled
    if (!wm->enable_attention_refresh) {
        set_error("Attention refresh disabled");
        return false;
    }

    // Refresh timestamp and set flag
    wm->timestamps[index] = get_current_time_ms();
    wm->attention_refreshed[index] = true;
    wm->total_refreshes++;

    return true;
}

/**
 * @brief Apply temporal decay to all items
 *
 * WHAT: Exponentially decay salience based on time elapsed
 * WHY:  Simulate natural forgetting without rehearsal
 * HOW:  For each item: Calculate decay → Update salience → Remove if below threshold
 *
 * FORMULA: salience_new = salience_old × exp(-Δt / τ)
 *
 * COMPLEXITY: O(n) where n = current_size
 *
 * @param wm Working memory instance (non-NULL)
 * @param current_time_ms Current time in milliseconds
 * @return Number of items removed due to decay
 */
uint32_t working_memory_decay(
    working_memory_t* wm,
    uint64_t current_time_ms
)
{
    // Guard: NULL working memory
    if (!wm) {
        set_error("NULL working_memory_t");
        return 0;
    }

    // Guard: Feature disabled
    if (!wm->enable_temporal_decay) {
        return 0;
    }

    uint32_t removed_count = 0;

    // Iterate backwards to safely remove items
    for (int i = (int)wm->current_size - 1; i >= 0; i--) {
        // Skip if attention-refreshed
        if (wm->attention_refreshed[i]) {
            wm->attention_refreshed[i] = false;  // Reset flag
            continue;
        }

        // Calculate time elapsed
        uint64_t elapsed_ms = current_time_ms - wm->timestamps[i];

        // Apply exponential decay: s_new = s_old × exp(-t/τ)
        float decay_factor = expf(-(float)elapsed_ms / wm->decay_tau_ms);
        wm->salience[i] *= decay_factor;

        // Remove if below threshold
        if (wm->salience[i] < wm->min_salience) {
            evict_item_at_index(wm, i);
            wm->total_decay_removals++;
            removed_count++;
        }
    }

    return removed_count;
}

// ============================================================================
// QUERY AND STATISTICS
// ============================================================================

/**
 * @brief Get current size of working memory
 *
 * WHAT: Return number of items currently stored
 * WHY:  Monitor buffer utilization
 * HOW:  Return current_size field
 *
 * COMPLEXITY: O(1)
 *
 * @param wm Working memory instance (non-NULL)
 * @return Current item count, or 0 on error
 */
uint32_t working_memory_get_size(const working_memory_t* wm) {
    // Guard: NULL working memory
    if (!wm) {
        return 0;
    }

    return wm->current_size;
}

/**
 * @brief Get current number of items (alias for get_size)
 *
 * WHAT: Return count of currently stored items
 * WHY:  Provide alternative naming for consistency with other APIs
 * HOW:  Call working_memory_get_size
 *
 * COMPLEXITY: O(1)
 */
uint32_t working_memory_get_count(const working_memory_t* wm) {
    return working_memory_get_size(wm);
}

/**
 * @brief Get working memory utilization percentage
 *
 * WHAT: Return percentage of capacity currently in use
 * WHY:  Monitor memory pressure and capacity usage
 * HOW:  Return (current_size / capacity) as float [0.0, 1.0]
 *
 * COMPLEXITY: O(1)
 */
float working_memory_get_utilization(const working_memory_t* wm) {
    // Guard: NULL working memory
    if (!wm) {
        return 0.0f;
    }

    // Guard: Zero capacity (shouldn't happen but guard anyway)
    if (wm->capacity == 0) {
        return 0.0f;
    }

    return (float)wm->current_size / (float)wm->capacity;
}

/**
 * @brief Get capacity of working memory
 *
 * WHAT: Return maximum item capacity
 * WHY:  Determine buffer limits
 * HOW:  Return capacity field
 *
 * COMPLEXITY: O(1)
 *
 * @param wm Working memory instance (non-NULL)
 * @return Maximum capacity, or 0 on error
 */
uint32_t working_memory_get_capacity(const working_memory_t* wm) {
    // Guard: NULL working memory
    if (!wm) {
        return 0;
    }

    return wm->capacity;
}

/**
 * @brief Check if working memory is full
 *
 * WHAT: Test if buffer has reached capacity
 * WHY:  Determine if next add will trigger eviction
 * HOW:  Compare current_size to capacity
 *
 * COMPLEXITY: O(1)
 *
 * @param wm Working memory instance (non-NULL)
 * @return true if full, false otherwise
 */
bool working_memory_is_full(const working_memory_t* wm) {
    // Guard: NULL working memory
    if (!wm) {
        return false;
    }

    return wm->current_size >= wm->capacity;
}

/**
 * @brief Find index of item with highest salience
 *
 * WHAT: Search for most important item
 * WHY:  Identify priority item for processing
 * HOW:  Linear scan with max tracking
 *
 * COMPLEXITY: O(n) where n = current_size
 *
 * @param wm Working memory instance (non-NULL)
 * @param salience Output parameter for salience value (nullable)
 * @return Index of highest salience item, or -1 if empty
 */
int working_memory_find_highest_salience(
    const working_memory_t* wm,
    float* salience
)
{
    // Guard: NULL working memory
    if (!wm) {
        set_error("NULL working_memory_t");
        return -1;
    }

    // Guard: Empty buffer
    if (wm->current_size == 0) {
        return -1;
    }

    int max_index = 0;
    float max_salience = wm->salience[0];

    for (uint32_t i = 1; i < wm->current_size; i++) {
        if (wm->salience[i] > max_salience) {
            max_salience = wm->salience[i];
            max_index = i;
        }
    }

    // Set salience if requested
    if (salience) {
        *salience = max_salience;
    }

    return max_index;
}

/**
 * @brief Get working memory statistics
 *
 * WHAT: Retrieve lifetime usage statistics
 * WHY:  Monitor performance and utilization patterns
 * HOW:  Copy internal statistics to output struct
 *
 * COMPLEXITY: O(n) for avg salience calculation
 *
 * @param wm Working memory instance (non-NULL)
 * @param stats Output statistics structure (non-NULL)
 */
void working_memory_get_stats(
    const working_memory_t* wm,
    working_memory_stats_t* stats
)
{
    // Guard: NULL pointers
    if (!wm || !stats) {
        return;
    }

    stats->current_size = wm->current_size;
    stats->capacity = wm->capacity;
    stats->total_additions = wm->total_additions;
    stats->total_evictions = wm->total_evictions;
    stats->total_refreshes = wm->total_refreshes;

    // Calculate average salience
    stats->avg_salience = 0.0f;
    if (wm->current_size > 0) {
        float sum = 0.0f;
        for (uint32_t i = 0; i < wm->current_size; i++) {
            sum += wm->salience[i];
        }
        stats->avg_salience = sum / wm->current_size;
    }

    // Find oldest item age
    stats->oldest_item_age_ms = 0.0f;
    if (wm->current_size > 0) {
        uint64_t current_time = get_current_time_ms();
        uint64_t oldest_time = wm->timestamps[0];
        for (uint32_t i = 1; i < wm->current_size; i++) {
            if (wm->timestamps[i] < oldest_time) {
                oldest_time = wm->timestamps[i];
            }
        }
        stats->oldest_item_age_ms = (float)(current_time - oldest_time);
    }
}
