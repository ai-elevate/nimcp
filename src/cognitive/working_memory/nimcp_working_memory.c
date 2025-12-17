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
 * BIO-ASYNC INTEGRATION:
 * - Module ID: 0x0334 (BIO_MODULE_WORKING_MEMORY)
 * - Publishes: item additions, evictions, refreshes
 * - Subscribes: attention updates, decay triggers
 *
 * PHASE: 10.2 (Working Memory)
 * DEPENDENCIES: None (standalone module)
 * TRAINING_IMPACT: None (inference-only, no weight modification)
 *
 * @author Claude Code
 * @date 2025-11
 */

#define LOG_MODULE "working_memory"

#include "cognitive/nimcp_working_memory.h"
#include "plasticity/nimcp_second_messengers.h"
#include "security/nimcp_security.h"
#include "security/nimcp_blood_brain_barrier.h"
#include "cognitive/global_workspace/nimcp_global_workspace.h"  // Global Workspace integration
#include "cognitive/immune/nimcp_brain_immune.h"  // Brain immune integration (cytokine enums)
#include "cognitive/nimcp_sleep_wake.h"  // Sleep state integration
#include "cognitive/working_memory/nimcp_working_memory_sleep_bridge.h"  // Sleep bridge for modulation

#include "nimcp.h"  // For error codes
#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"
#include "utils/memory/nimcp_unified_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/time/nimcp_time.h"
#include "utils/platform/nimcp_platform_mutex.h"  // Thread safety
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include "utils/memory/nimcp_memory_guards.h"  // For nimcp_calloc/nimcp_free

//=============================================================================
// BIO-ASYNC MODULE REGISTRATION
//=============================================================================

#define BIO_MODULE_WORKING_MEMORY 0x0334

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

    // Thread safety (added 2025-11-17)
    nimcp_platform_mutex_t mutex;   // Protects all working memory operations

    // Bio-async integration
    bio_module_context_t bio_ctx;   // Bio-async module context
    bool bio_async_enabled;         // Bio-async registration status

    // Second messenger integration
    second_messenger_system_t* sm_system;  // Second messenger cascade system
    bool enable_second_messengers;         // Whether cascade modulation is enabled
    uint32_t num_neurons;                  // Number of neurons for cascade tracking

    // Global Workspace integration (Phase 10.x)
    global_workspace_t* workspace;           // Global workspace for conscious access
    bool workspace_integration_enabled;       // Workspace integration active
    float workspace_salience_threshold;       // Threshold for triggering ignition

    // Positional encoding integration
    nimcp_pos_encoder_t* pos_encoder;        // Positional encoder for serial position effects
    bool enable_positional_encoding;          // Whether position encoding is active
    nimcp_pos_encoding_type_t pe_type;       // Type of positional encoding
    uint32_t pe_embedding_dim;               // Dimension of position embeddings
    float* pe_buffer;                        // Temporary buffer for position encodings

    // Brain immune integration
    struct brain_immune_system* immune;      // Connected immune system
    bool immune_integration_enabled;         // Immune integration active
    uint32_t inflammation_capacity_penalty;  // Capacity reduction from inflammation
    float last_stress_signal_time_ms;        // Last stress cytokine release time

    // Sleep state integration
    sleep_state_t current_sleep_state;       // Current sleep/wake state for modulation
};

//=============================================================================
// BIO-ASYNC MESSAGE HANDLERS
//=============================================================================

/**
 * @brief Handle working memory store request via bio-async
 *
 * WHAT: Process incoming request to store data in working memory
 * WHY:  Enable distributed systems to add items to working memory via bio-async
 * HOW:  Extract payload data, call working_memory_add with salience from priority
 *
 * NOTE: Data follows immediately after bio_msg_wm_store_t in message buffer
 */
static nimcp_error_t handle_wm_store_request(
    const void* msg,
    size_t msg_size,
    nimcp_bio_promise_t response_promise,
    void* user_data)
{
    (void)response_promise;

    if (!msg || !user_data) {
        return NIMCP_ERROR_NULL_ARG;
    }

    if (msg_size < sizeof(bio_msg_wm_store_t)) {
        LOG_ERROR("Store request too small: %zu bytes", msg_size);
        return NIMCP_ERROR_INVALID;
    }

    const bio_msg_wm_store_t* store_msg = (const bio_msg_wm_store_t*)msg;
    working_memory_t* wm = (working_memory_t*)user_data;

    LOG_DEBUG("Received WM store request: slot=%u, size=%u, priority=%.2f",
              store_msg->slot_id, store_msg->data_size, store_msg->priority);

    // Extract payload data (follows immediately after header)
    const uint8_t* payload = (const uint8_t*)msg + sizeof(bio_msg_wm_store_t);
    size_t expected_size = sizeof(bio_msg_wm_store_t) + store_msg->data_size;

    if (msg_size < expected_size) {
        LOG_ERROR("Incomplete store request: expected %zu, got %zu", expected_size, msg_size);
        return NIMCP_ERROR_INVALID;
    }

    // Convert byte data to float array (assuming data_size is in bytes)
    uint32_t num_floats = store_msg->data_size / sizeof(float);
    if (num_floats == 0) {
        LOG_WARN("Empty data in store request");
        return NIMCP_ERROR_INVALID;
    }

    const float* data = (const float*)payload;

    // Add to working memory with priority as salience
    bool success = working_memory_add(wm, data, num_floats, store_msg->priority);

    if (!success) {
        LOG_ERROR("Failed to add item to working memory");
        return NIMCP_ERROR_MEMORY;
    }

    LOG_DEBUG("Successfully stored %u floats in working memory", num_floats);
    return NIMCP_SUCCESS;
}

/**
 * @brief Handle working memory retrieve request via bio-async
 *
 * WHAT: Process incoming request to retrieve data from working memory
 * WHY:  Enable distributed systems to query working memory contents via bio-async
 * HOW:  Lookup item by slot_id, send response with data via promise
 *
 * NOTE: Response includes retrieved data in payload
 */
static nimcp_error_t handle_wm_retrieve_request(
    const void* msg,
    size_t msg_size,
    nimcp_bio_promise_t response_promise,
    void* user_data)
{
    (void)msg_size;

    if (!msg || !user_data) {
        return NIMCP_ERROR_NULL_ARG;
    }

    const bio_msg_wm_retrieve_t* retrieve_msg = (const bio_msg_wm_retrieve_t*)msg;
    working_memory_t* wm = (working_memory_t*)user_data;

    LOG_DEBUG("Received WM retrieve request: slot=%u, min_confidence=%.2f",
              retrieve_msg->slot_id, retrieve_msg->min_confidence);

    // Get item from working memory
    uint32_t item_size = 0;
    const float* item = working_memory_get(wm, retrieve_msg->slot_id, &item_size);

    if (!item) {
        LOG_WARN("Item not found at slot %u", retrieve_msg->slot_id);
        // Send empty response via promise to indicate failure
        if (response_promise) {
            bio_msg_wm_store_t response = {0};
            bio_msg_init_header(&response.header, BIO_MSG_WORKING_MEMORY_STORE,
                                BIO_MODULE_WORKING_MEMORY, 0, sizeof(response));
            nimcp_bio_promise_complete_sized(response_promise, &response, sizeof(response));
        }
        return NIMCP_ERROR_INVALID;
    }

    // Get salience for confidence check
    float salience = 0.0F;
    working_memory_get_total_salience(wm, retrieve_msg->slot_id, &salience);

    if (salience < retrieve_msg->min_confidence) {
        LOG_DEBUG("Item salience %.2f below threshold %.2f",
                  salience, retrieve_msg->min_confidence);
        // Send empty response to indicate confidence too low
        if (response_promise) {
            bio_msg_wm_store_t response = {0};
            bio_msg_init_header(&response.header, BIO_MSG_WORKING_MEMORY_STORE,
                                BIO_MODULE_WORKING_MEMORY, 0, sizeof(response));
            nimcp_bio_promise_complete_sized(response_promise, &response, sizeof(response));
        }
        return NIMCP_ERROR_INVALID;
    }

    // Send response with retrieved data via promise
    if (response_promise) {
        size_t response_size = sizeof(bio_msg_wm_store_t) + (item_size * sizeof(float));
        uint8_t* response_buf = nimcp_malloc(response_size);
        if (!response_buf) {
            LOG_ERROR("Failed to allocate response buffer");
            return NIMCP_ERROR_MEMORY;
        }

        bio_msg_wm_store_t* response = (bio_msg_wm_store_t*)response_buf;
        bio_msg_init_header(&response->header, BIO_MSG_WORKING_MEMORY_STORE,
                            BIO_MODULE_WORKING_MEMORY, 0, response_size);
        response->slot_id = retrieve_msg->slot_id;
        response->data_size = item_size * sizeof(float);
        response->priority = salience;

        // Copy item data to response payload
        memcpy(response_buf + sizeof(bio_msg_wm_store_t), item, item_size * sizeof(float));

        nimcp_bio_promise_complete_sized(response_promise, response_buf, response_size);
        nimcp_free(response_buf);

        LOG_DEBUG("Retrieved %u floats from slot %u", item_size, retrieve_msg->slot_id);
    }

    return NIMCP_SUCCESS;
}

/**
 * @brief Broadcast item stored event
 */
static void bio_broadcast_item_stored(working_memory_t* wm, uint32_t slot_id, float salience) {
    if (!wm || !wm->bio_async_enabled || !wm->bio_ctx) {
        return;
    }

    bio_msg_wm_store_t msg = {};
    bio_msg_init_header(&msg.header, BIO_MSG_WORKING_MEMORY_STORE,
                        bio_module_context_get_id(wm->bio_ctx), 0, sizeof(msg));
    msg.header.flags |= BIO_MSG_FLAG_BROADCAST;
    msg.slot_id = slot_id;
    msg.priority = salience;

    bio_router_broadcast(wm->bio_ctx, &msg, sizeof(msg));
    LOG_DEBUG("Broadcast: item stored in slot %u", slot_id);
}

/**
 * @brief Broadcast item evicted event (attention shift)
 */
static void bio_broadcast_item_evicted(working_memory_t* wm, uint32_t slot_id) {
    if (!wm || !wm->bio_async_enabled || !wm->bio_ctx) {
        return;
    }

    bio_msg_attention_shift_t msg = {};
    bio_msg_init_header(&msg.header, BIO_MSG_ATTENTION_SHIFT,
                        bio_module_context_get_id(wm->bio_ctx), 0, sizeof(msg));
    msg.header.flags |= BIO_MSG_FLAG_BROADCAST;
    msg.target_id = slot_id;
    msg.attention_weight = 0.0F;  // Item is gone
    msg.preemptive = false;

    bio_router_broadcast(wm->bio_ctx, &msg, sizeof(msg));
    LOG_DEBUG("Broadcast: item evicted from slot %u", slot_id);
}

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
 * HOW:  Free memory → Shift arrays left → Decrement size → NULL last slot
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
    nimcp_free(wm->items[index]);

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

    // NULL out the last slot to prevent double-free
    // After memmove and size decrement, items[current_size] contains a stale pointer
    wm->items[wm->current_size] = NULL;

    wm->total_evictions++;

    // Signal immune system on eviction (TNF-alpha for failure)
    if (wm->immune_integration_enabled && wm->immune) {
        // Release TNF-alpha (eviction = resource failure)
        uint32_t cytokine_id = 0;
        brain_immune_release_cytokine(
            wm->immune,
            CYTOKINE_TNFA,
            0,  // Working memory module
            0.5f,  // Moderate signal
            0,  // Broadcast
            &cytokine_id
        );
    }

    // Broadcast eviction event via bio-async
    bio_broadcast_item_evicted(wm, index);
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
        .enable_temporal_decay = true,
        .enable_positional_encoding = true,           // Enable position encoding
        .pe_type = NIMCP_POS_SINUSOIDAL,              // Sinusoidal (no training needed)
        .pe_embedding_dim = 64                        // 64-dim position embeddings
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
        LOG_ERROR("NULL config provided to working_memory_create_custom");
        return NULL;
    }

    // Guard: Invalid capacity
    if (config->capacity < MIN_CAPACITY || config->capacity > MAX_CAPACITY) {
        set_error("Invalid capacity (must be 1-32)");
        LOG_ERROR("Invalid capacity: %u (must be %d-%d)", config->capacity, MIN_CAPACITY, MAX_CAPACITY);
        return NULL;
    }

    // Guard: Invalid decay tau
    if (config->decay_tau_ms <= 0.0F) {
        set_error("Invalid decay_tau_ms (must be > 0)");
        LOG_ERROR("Invalid decay_tau_ms: %.2f (must be > 0)", config->decay_tau_ms);
        return NULL;
    }

    LOG_INFO("Creating working memory: capacity=%u, decay_tau_ms=%.2f",
             config->capacity, config->decay_tau_ms);

    // Allocate main structure
    working_memory_t* wm = nimcp_calloc(1, sizeof(working_memory_t));
    if (!wm) {
        set_error("Failed to allocate working_memory_t");
        LOG_ERROR("Failed to allocate working_memory_t (%zu bytes)", sizeof(working_memory_t));
        return NULL;
    }

    // Allocate arrays
    wm->items = nimcp_calloc(config->capacity, sizeof(float*));
    wm->item_sizes = nimcp_calloc(config->capacity, sizeof(uint32_t));
    wm->salience = nimcp_calloc(config->capacity, sizeof(float));
    wm->timestamps = nimcp_calloc(config->capacity, sizeof(uint64_t));
    wm->attention_refreshed = nimcp_calloc(config->capacity, sizeof(bool));
    wm->emotions = nimcp_calloc(config->capacity, sizeof(emotional_tag_t));  // Phase 10.3
    wm->has_emotion = nimcp_calloc(config->capacity, sizeof(bool));          // Phase 10.3

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

    // Initialize mutex for thread safety
    if (nimcp_platform_mutex_init(&wm->mutex, false) != 0) {
        set_error("Failed to initialize mutex");
        working_memory_destroy(wm);
        return NULL;
    }

    // Bio-async registration
    wm->bio_ctx = NULL;
    wm->bio_async_enabled = false;
    if (bio_router_is_initialized()) {
        bio_module_info_t bio_info = {
            .module_id = BIO_MODULE_WORKING_MEMORY,
            .module_name = "working_memory",
            .inbox_capacity = 32,
            .user_data = wm
        };
        wm->bio_ctx = bio_router_register_module(&bio_info);
        if (wm->bio_ctx) {
            wm->bio_async_enabled = true;
            // Register message handlers
            bio_router_register_handler(wm->bio_ctx, BIO_MSG_WORKING_MEMORY_STORE, handle_wm_store_request);
            bio_router_register_handler(wm->bio_ctx, BIO_MSG_WORKING_MEMORY_RETRIEVE, handle_wm_retrieve_request);
            // Subscribe to neuromodulator release messages for cascade triggering
            bio_router_register_handler(wm->bio_ctx, BIO_MSG_NEUROMODULATOR_RELEASE, handle_wm_store_request);
            LOG_INFO("Bio-async registered for working_memory module with handlers");
        }
    }

    // Second messenger integration (optional, must be set via working_memory_integrate_second_messengers)
    wm->sm_system = NULL;
    wm->enable_second_messengers = false;
    wm->num_neurons = 0;

    // Initialize positional encoding
    wm->pos_encoder = NULL;
    wm->pe_buffer = NULL;
    wm->enable_positional_encoding = config->enable_positional_encoding;
    wm->pe_type = config->pe_type;
    wm->pe_embedding_dim = config->pe_embedding_dim;

    // Initialize sleep state (default: awake)
    wm->current_sleep_state = SLEEP_STATE_AWAKE;

    if (wm->enable_positional_encoding) {
        // Create positional encoder configuration
        nimcp_pos_config_t pe_config;
        pe_config.type = config->pe_type;

        // Configure based on type
        if (config->pe_type == NIMCP_POS_SINUSOIDAL) {
            pe_config.config.sinusoidal = nimcp_pos_sinusoidal_default_config();
            pe_config.config.sinusoidal.base.max_seq_length = config->capacity;
            pe_config.config.sinusoidal.base.embedding_dim = config->pe_embedding_dim;
            pe_config.config.sinusoidal.base.cache_enabled = true;
        } else if (config->pe_type == NIMCP_POS_RELATIVE) {
            pe_config.config.relative = nimcp_pos_relative_default_config();
            pe_config.config.relative.base.max_seq_length = config->capacity;
            pe_config.config.relative.base.embedding_dim = config->pe_embedding_dim;
            pe_config.config.relative.base.cache_enabled = true;
            pe_config.config.relative.max_relative_pos = config->capacity;
        } else {
            // For other types, use sinusoidal as fallback
            LOG_WARN("Unsupported PE type %d, using SINUSOIDAL", config->pe_type);
            pe_config.type = NIMCP_POS_SINUSOIDAL;
            pe_config.config.sinusoidal = nimcp_pos_sinusoidal_default_config();
            pe_config.config.sinusoidal.base.max_seq_length = config->capacity;
            pe_config.config.sinusoidal.base.embedding_dim = config->pe_embedding_dim;
            pe_config.config.sinusoidal.base.cache_enabled = true;
            wm->pe_type = NIMCP_POS_SINUSOIDAL;
        }

        // Create encoder
        wm->pos_encoder = nimcp_pos_encoder_create(&pe_config);
        if (!wm->pos_encoder) {
            LOG_ERROR("Failed to create positional encoder");
            working_memory_destroy(wm);
            return NULL;
        }

        // Allocate PE buffer for temporary position encodings
        wm->pe_buffer = nimcp_malloc(config->pe_embedding_dim * sizeof(float));
        if (!wm->pe_buffer) {
            LOG_ERROR("Failed to allocate PE buffer");
            working_memory_destroy(wm);
            return NULL;
        }

        LOG_INFO("Positional encoding enabled: type=%d, dim=%u",
                 wm->pe_type, wm->pe_embedding_dim);
    }

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
            nimcp_free(wm->items[i]);
        }
        nimcp_free(wm->items);
    }

    // Free arrays
    nimcp_free(wm->item_sizes);
    nimcp_free(wm->salience);
    nimcp_free(wm->timestamps);
    nimcp_free(wm->attention_refreshed);
    nimcp_free(wm->emotions);      // Phase 10.3
    nimcp_free(wm->has_emotion);   // Phase 10.3

    // Unregister from bio-router
    if (wm->bio_async_enabled && wm->bio_ctx) {
        bio_router_unregister_module(wm->bio_ctx);
        wm->bio_ctx = NULL;
        wm->bio_async_enabled = false;
    }

    // Destroy positional encoder
    if (wm->pos_encoder) {
        nimcp_pos_encoder_destroy(wm->pos_encoder);
        wm->pos_encoder = NULL;
    }
    if (wm->pe_buffer) {
        nimcp_free(wm->pe_buffer);
        wm->pe_buffer = NULL;
    }

    // Destroy mutex
    nimcp_platform_mutex_destroy(&wm->mutex);

    // Free main structure
    nimcp_free(wm);
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
    if (salience < 0.0F || salience > 1.0F) {
        set_error("salience must be in [0.0, 1.0]");
        return false;
    }

    // Lock mutex for thread-safe access
    nimcp_platform_mutex_lock(&wm->mutex);

    // Check if full → evict lowest salience item
    if (wm->current_size >= wm->capacity) {
        int evict_index = find_lowest_salience_index(wm);
        if (evict_index >= 0) {
            evict_item_at_index(wm, evict_index);
        }
    }

    // Allocate and copy item
    float* item_copy = nimcp_malloc(item_size * sizeof(float));
    if (!item_copy) {
        set_error("Failed to allocate item memory");
        nimcp_platform_mutex_unlock(&wm->mutex);
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

    // Check for high utilization and signal stress to immune system
    if (wm->immune_integration_enabled && wm->immune) {
        float utilization = (float)wm->current_size / (float)working_memory_get_effective_capacity(wm);
        if (utilization > 0.9f) {
            // High utilization - signal IL-6 (cognitive load)
            uint32_t cytokine_id = 0;
            brain_immune_release_cytokine(
                wm->immune,
                CYTOKINE_IL6,
                0,  // Working memory module
                utilization,  // Signal strength = utilization
                0,  // Broadcast
                &cytokine_id
            );
        }
    }

    // Broadcast item stored event via bio-async
    bio_broadcast_item_stored(wm, index, salience);

    nimcp_platform_mutex_unlock(&wm->mutex);
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
    // Guard: NULL working memory
    if (!wm) {
        set_error("NULL working_memory_t");
        return false;
    }

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
    if (total_salience > 1.0F) {
        total_salience = 1.0F;
    }

    // Add item with boosted salience (this will acquire the lock)
    if (!working_memory_add(wm, item, item_size, total_salience)) {
        return false;
    }

    // Lock mutex to attach emotional tag
    nimcp_platform_mutex_lock(&wm->mutex);

    // Attach emotional tag to the just-added item
    uint32_t index = wm->current_size - 1;  // Last added
    wm->emotions[index] = *emotion;
    wm->has_emotion[index] = true;

    nimcp_platform_mutex_unlock(&wm->mutex);
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

    // Lock mutex for thread-safe access
    // NOTE: This is a const function but we need to lock for thread safety
    // The mutex itself is mutable via nimcp_platform_mutex_lock
    nimcp_platform_mutex_lock((nimcp_platform_mutex_t*)&wm->mutex);

    // Guard: Invalid index
    if (index >= wm->current_size) {
        set_error("Index out of bounds");
        nimcp_platform_mutex_unlock((nimcp_platform_mutex_t*)&wm->mutex);
        return NULL;
    }

    // Set size if requested
    if (size) {
        *size = wm->item_sizes[index];
    }

    const float* result = wm->items[index];
    nimcp_platform_mutex_unlock((nimcp_platform_mutex_t*)&wm->mutex);
    return result;
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

    // Lock mutex for thread-safe access
    nimcp_platform_mutex_lock(&wm->mutex);

    // Guard: Invalid index
    if (index >= wm->current_size) {
        set_error("Index out of bounds");
        nimcp_platform_mutex_unlock(&wm->mutex);
        return false;
    }

    evict_item_at_index(wm, index);
    nimcp_platform_mutex_unlock(&wm->mutex);
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

    // Lock mutex for thread-safe access
    nimcp_platform_mutex_lock(&wm->mutex);

    // Free all items
    for (uint32_t i = 0; i < wm->current_size; i++) {
        nimcp_free(wm->items[i]);
        wm->items[i] = NULL;
    }

    wm->current_size = 0;

    nimcp_platform_mutex_unlock(&wm->mutex);
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

    // Lock mutex for thread-safe access
    nimcp_platform_mutex_lock((nimcp_platform_mutex_t*)&wm->mutex);

    // Guard: Invalid index
    if (index >= wm->current_size) {
        set_error("Index out of bounds");
        nimcp_platform_mutex_unlock((nimcp_platform_mutex_t*)&wm->mutex);
        return false;
    }

    // Copy emotional tag (or neutral if none)
    if (wm->has_emotion[index]) {
        *emotion = wm->emotions[index];
    } else {
        *emotion = emotional_tag_neutral();
    }

    nimcp_platform_mutex_unlock((nimcp_platform_mutex_t*)&wm->mutex);
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

    // Lock mutex for thread-safe access
    nimcp_platform_mutex_lock((nimcp_platform_mutex_t*)&wm->mutex);

    // Guard: Invalid index
    if (index >= wm->current_size) {
        set_error("Index out of bounds");
        nimcp_platform_mutex_unlock((nimcp_platform_mutex_t*)&wm->mutex);
        return false;
    }

    // Return stored salience (already boosted if emotion present)
    *total_salience = wm->salience[index];

    nimcp_platform_mutex_unlock((nimcp_platform_mutex_t*)&wm->mutex);
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

    // Guard: Feature disabled (check before locking)
    if (!wm->enable_attention_refresh) {
        set_error("Attention refresh disabled");
        return false;
    }

    // Lock mutex for thread-safe access
    nimcp_platform_mutex_lock(&wm->mutex);

    // Guard: Invalid index
    if (index >= wm->current_size) {
        set_error("Index out of bounds");
        nimcp_platform_mutex_unlock(&wm->mutex);
        return false;
    }

    // Refresh timestamp and set flag
    wm->timestamps[index] = get_current_time_ms();
    wm->attention_refreshed[index] = true;
    wm->total_refreshes++;

    nimcp_platform_mutex_unlock(&wm->mutex);
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

    // Process pending bio-async messages before decay processing
    if (wm->bio_async_enabled && wm->bio_ctx) {
        bio_router_process_inbox(wm->bio_ctx, 10);  // Process up to 10 messages
    }

    // Lock mutex for thread-safe access
    nimcp_platform_mutex_lock(&wm->mutex);

    uint32_t removed_count = 0;

    // Get sleep-modulated decay rate
    float decay_rate_modulation = working_memory_sleep_decay_for_state(wm->current_sleep_state);

    // Iterate backwards to safely remove items
    for (int i = (int)wm->current_size - 1; i >= 0; i--) {
        // Skip if attention-refreshed
        if (wm->attention_refreshed[i]) {
            wm->attention_refreshed[i] = false;  // Reset flag
            continue;
        }

        // Calculate time elapsed
        uint64_t elapsed_ms = current_time_ms - wm->timestamps[i];

        // Apply exponential decay with sleep-modulated rate: s_new = s_old × exp(-t×decay_rate/τ)
        float decay_factor = expf(-(float)elapsed_ms * decay_rate_modulation / wm->decay_tau_ms);
        wm->salience[i] *= decay_factor;

        // Remove if below threshold
        if (wm->salience[i] < wm->min_salience) {
            evict_item_at_index(wm, i);
            wm->total_decay_removals++;
            removed_count++;
        }
    }

    // Signal immune system if items were removed by decay (IL-1 for resource scarcity)
    if (removed_count > 0 && wm->immune_integration_enabled && wm->immune) {
        uint32_t cytokine_id = 0;
        float signal_strength = (float)removed_count / (float)wm->capacity;
        if (signal_strength > 1.0f) signal_strength = 1.0f;

        brain_immune_release_cytokine(
            wm->immune,
            CYTOKINE_IL1B,
            0,  // Working memory module
            signal_strength,
            0,  // Broadcast
            &cytokine_id
        );
    }

    nimcp_platform_mutex_unlock(&wm->mutex);
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

    // Lock mutex for thread-safe access
    nimcp_platform_mutex_lock((nimcp_platform_mutex_t*)&wm->mutex);
    uint32_t size = wm->current_size;
    nimcp_platform_mutex_unlock((nimcp_platform_mutex_t*)&wm->mutex);

    return size;
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
        return 0.0F;
    }

    // Guard: Zero capacity (shouldn't happen but guard anyway)
    if (wm->capacity == 0) {
        return 0.0F;
    }

    // Lock mutex for thread-safe access
    nimcp_platform_mutex_lock((nimcp_platform_mutex_t*)&wm->mutex);
    float utilization = (float)wm->current_size / (float)wm->capacity;
    nimcp_platform_mutex_unlock((nimcp_platform_mutex_t*)&wm->mutex);

    return utilization;
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

    // Lock mutex for thread-safe access
    nimcp_platform_mutex_lock((nimcp_platform_mutex_t*)&wm->mutex);
    uint32_t capacity = wm->capacity;
    nimcp_platform_mutex_unlock((nimcp_platform_mutex_t*)&wm->mutex);

    return capacity;
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

    // Lock mutex for thread-safe access
    nimcp_platform_mutex_lock((nimcp_platform_mutex_t*)&wm->mutex);
    bool is_full = wm->current_size >= wm->capacity;
    nimcp_platform_mutex_unlock((nimcp_platform_mutex_t*)&wm->mutex);

    return is_full;
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

    // Lock mutex for thread-safe access
    nimcp_platform_mutex_lock((nimcp_platform_mutex_t*)&wm->mutex);

    // Guard: Empty buffer
    if (wm->current_size == 0) {
        nimcp_platform_mutex_unlock((nimcp_platform_mutex_t*)&wm->mutex);
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

    nimcp_platform_mutex_unlock((nimcp_platform_mutex_t*)&wm->mutex);
    return max_index;
}

/**
 * @brief Find index of item with lowest salience
 *
 * WHAT: Search for least important item
 * WHY:  Identify item for eviction when memory full
 * HOW:  Linear scan with min tracking
 *
 * COMPLEXITY: O(n) where n = current_size
 *
 * @param wm Working memory instance (non-NULL)
 * @param salience Output parameter for salience value (nullable)
 * @return Index of lowest salience item, or -1 if empty
 */
int working_memory_find_lowest_salience(
    const working_memory_t* wm,
    float* salience
)
{
    /* Guard: NULL working memory */
    if (!wm) {
        return -1;
    }

    /* Lock mutex for thread-safe access */
    nimcp_platform_mutex_lock((nimcp_platform_mutex_t*)&wm->mutex);

    /* Guard: Empty buffer */
    if (wm->current_size == 0) {
        nimcp_platform_mutex_unlock((nimcp_platform_mutex_t*)&wm->mutex);
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

    /* Set salience if requested */
    if (salience) {
        *salience = min_salience;
    }

    nimcp_platform_mutex_unlock((nimcp_platform_mutex_t*)&wm->mutex);
    return min_index;
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

    // Lock mutex for thread-safe access
    nimcp_platform_mutex_lock((nimcp_platform_mutex_t*)&wm->mutex);

    stats->current_size = wm->current_size;
    stats->capacity = wm->capacity;
    stats->total_additions = wm->total_additions;
    stats->total_evictions = wm->total_evictions;
    stats->total_refreshes = wm->total_refreshes;

    // Calculate average salience
    stats->avg_salience = 0.0F;
    if (wm->current_size > 0) {
        float sum = 0.0F;
        for (uint32_t i = 0; i < wm->current_size; i++) {
            sum += wm->salience[i];
        }
        stats->avg_salience = sum / wm->current_size;
    }

    // Find oldest item age
    stats->oldest_item_age_ms = 0.0F;
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

    nimcp_platform_mutex_unlock((nimcp_platform_mutex_t*)&wm->mutex);
}

//=============================================================================
// POSITIONAL ENCODING FUNCTIONS
//=============================================================================

/**
 * @brief Apply positional encodings to all items in working memory
 *
 * WHAT: Add position embeddings to each item based on its slot position
 * WHY:  Capture serial position effects (primacy, recency) in working memory
 * HOW:  For each item at position i, apply PE(i) additively to item data
 *
 * ALGORITHM:
 * 1. Check if positional encoding is enabled
 * 2. For each item in buffer:
 *    a. Get position encoding PE(i) for slot i
 *    b. Add PE to item data: item[j] += PE(i)[j % pe_dim]
 * 3. Handle dimension mismatch (cycle or truncate PE)
 *
 * BIOLOGICAL BASIS:
 * - Serial position effects: primacy (early items) and recency (late items)
 * - Prefrontal cortex encodes temporal order of representations
 * - Position information aids working memory retrieval and manipulation
 *
 * COMPLEXITY: O(n × d) where n = current_size, d = embedding_dim
 *
 * @param wm Working memory instance (non-NULL)
 * @return true on success, false if PE disabled or error
 */
bool working_memory_encode_positions(working_memory_t* wm) {
    // Guard: NULL working memory
    if (!wm) {
        set_error("NULL working_memory_t");
        return false;
    }

    // Guard: Positional encoding disabled
    if (!wm->enable_positional_encoding || !wm->pos_encoder) {
        return false;
    }

    // Lock mutex for thread-safe access
    nimcp_platform_mutex_lock(&wm->mutex);

    // Guard: Empty buffer
    if (wm->current_size == 0) {
        nimcp_platform_mutex_unlock(&wm->mutex);
        return true;  // Success: nothing to encode
    }

    // Apply position encoding to each item
    for (uint32_t pos = 0; pos < wm->current_size; pos++) {
        // Get position encoding for this slot
        int result = nimcp_pos_encode_position(wm->pos_encoder, pos, wm->pe_buffer);
        if (result != NIMCP_POS_SUCCESS) {
            LOG_WARN("Failed to encode position %u: error %d", pos, result);
            continue;  // Skip this position, continue with others
        }

        // Add position encoding to item data
        // Handle dimension mismatch: cycle PE if item is larger than PE dim
        float* item_data = wm->items[pos];
        uint32_t item_size = wm->item_sizes[pos];

        for (uint32_t j = 0; j < item_size; j++) {
            // Cycle through PE dimensions if item is larger
            uint32_t pe_idx = j % wm->pe_embedding_dim;
            item_data[j] += wm->pe_buffer[pe_idx];
        }
    }

    nimcp_platform_mutex_unlock(&wm->mutex);
    return true;
}

/**
 * @brief Get positional embedding for specific slot
 *
 * WHAT: Retrieve position encoding vector for a working memory slot
 * WHY:  Inspect position information, external position-aware processing
 * HOW:  Query internal positional encoder for slot's position encoding
 *
 * COMPLEXITY: O(1) if cached, O(d) if computed where d = embedding_dim
 *
 * BIOLOGICAL BASIS:
 * - Position codes in prefrontal working memory representations
 * - Enables comparison of relative positions between items
 * - Supports temporal reasoning over working memory contents
 *
 * @param wm Working memory instance (non-NULL)
 * @param slot_index Slot position [0, capacity)
 * @param output Output buffer for position embedding (pe_embedding_dim floats)
 * @return true on success, false on invalid slot or PE disabled
 */
bool working_memory_get_position_embedding(
    const working_memory_t* wm,
    uint32_t slot_index,
    float* output
)
{
    // Guard: NULL working memory
    if (!wm) {
        set_error("NULL working_memory_t");
        return false;
    }

    // Guard: NULL output
    if (!output) {
        set_error("NULL output buffer");
        return false;
    }

    // Guard: Positional encoding disabled
    if (!wm->enable_positional_encoding || !wm->pos_encoder) {
        set_error("Positional encoding disabled");
        return false;
    }

    // Guard: Invalid slot index
    if (slot_index >= wm->capacity) {
        set_error("Slot index out of bounds");
        return false;
    }

    // Get position encoding (no lock needed, encoder is thread-safe)
    int result = nimcp_pos_encode_position(wm->pos_encoder, slot_index, output);
    if (result != NIMCP_POS_SUCCESS) {
        set_error("Failed to encode position");
        LOG_ERROR("Position encoding failed for slot %u: error %d", slot_index, result);
        return false;
    }

    return true;
}

/**
 * @brief Configure positional encoding type
 *
 * WHAT: Change the type of positional encoding used
 * WHY:  Allow runtime switching between encoding strategies
 * HOW:  Destroy old encoder, create new encoder with specified type
 *
 * ALGORITHM:
 * 1. Validate new PE type
 * 2. Lock mutex for thread safety
 * 3. Destroy existing encoder
 * 4. Create new encoder with specified type
 * 5. Reapply encodings to existing items
 * 6. Unlock mutex
 *
 * COMPLEXITY: O(capacity × embedding_dim) - must rebuild encoder cache
 *
 * @param wm Working memory instance (non-NULL)
 * @param pe_type New positional encoding type
 * @return true on success, false on invalid type or allocation failure
 */
bool working_memory_set_pe_type(
    working_memory_t* wm,
    nimcp_pos_encoding_type_t pe_type
)
{
    // Guard: NULL working memory
    if (!wm) {
        set_error("NULL working_memory_t");
        return false;
    }

    // Guard: Positional encoding disabled
    if (!wm->enable_positional_encoding) {
        set_error("Positional encoding disabled");
        return false;
    }

    // Guard: Invalid PE type
    if (pe_type >= NIMCP_POS_TYPE_COUNT) {
        set_error("Invalid positional encoding type");
        return false;
    }

    // Lock mutex for thread-safe access
    nimcp_platform_mutex_lock(&wm->mutex);

    // If already using this type, nothing to do
    if (wm->pe_type == pe_type) {
        nimcp_platform_mutex_unlock(&wm->mutex);
        return true;
    }

    LOG_INFO("Switching PE type from %d to %d", wm->pe_type, pe_type);

    // Destroy old encoder
    if (wm->pos_encoder) {
        nimcp_pos_encoder_destroy(wm->pos_encoder);
        wm->pos_encoder = NULL;
    }

    // Create new encoder configuration
    nimcp_pos_config_t pe_config;
    pe_config.type = pe_type;

    // Configure based on type
    if (pe_type == NIMCP_POS_SINUSOIDAL) {
        pe_config.config.sinusoidal = nimcp_pos_sinusoidal_default_config();
        pe_config.config.sinusoidal.base.max_seq_length = wm->capacity;
        pe_config.config.sinusoidal.base.embedding_dim = wm->pe_embedding_dim;
        pe_config.config.sinusoidal.base.cache_enabled = true;
    } else if (pe_type == NIMCP_POS_RELATIVE) {
        pe_config.config.relative = nimcp_pos_relative_default_config();
        pe_config.config.relative.base.max_seq_length = wm->capacity;
        pe_config.config.relative.base.embedding_dim = wm->pe_embedding_dim;
        pe_config.config.relative.base.cache_enabled = true;
        pe_config.config.relative.max_relative_pos = wm->capacity;
    } else if (pe_type == NIMCP_POS_LEARNED) {
        pe_config.config.learned = nimcp_pos_learned_default_config();
        pe_config.config.learned.base.max_seq_length = wm->capacity;
        pe_config.config.learned.base.embedding_dim = wm->pe_embedding_dim;
        pe_config.config.learned.base.cache_enabled = true;
    } else if (pe_type == NIMCP_POS_ROTARY) {
        pe_config.config.rope = nimcp_pos_rope_default_config();
        pe_config.config.rope.base.max_seq_length = wm->capacity;
        pe_config.config.rope.base.embedding_dim = wm->pe_embedding_dim;
        pe_config.config.rope.base.cache_enabled = true;
    } else if (pe_type == NIMCP_POS_ALIBI) {
        pe_config.config.alibi = nimcp_pos_alibi_default_config();
        pe_config.config.alibi.base.max_seq_length = wm->capacity;
        pe_config.config.alibi.base.embedding_dim = wm->pe_embedding_dim;
        pe_config.config.alibi.base.cache_enabled = true;
    } else {
        // Unsupported type, use sinusoidal as fallback
        LOG_WARN("Unsupported PE type %d, using SINUSOIDAL", pe_type);
        pe_config.type = NIMCP_POS_SINUSOIDAL;
        pe_config.config.sinusoidal = nimcp_pos_sinusoidal_default_config();
        pe_config.config.sinusoidal.base.max_seq_length = wm->capacity;
        pe_config.config.sinusoidal.base.embedding_dim = wm->pe_embedding_dim;
        pe_config.config.sinusoidal.base.cache_enabled = true;
        pe_type = NIMCP_POS_SINUSOIDAL;
    }

    // Create new encoder
    wm->pos_encoder = nimcp_pos_encoder_create(&pe_config);
    if (!wm->pos_encoder) {
        set_error("Failed to create new positional encoder");
        LOG_ERROR("Failed to create PE encoder for type %d", pe_type);
        wm->enable_positional_encoding = false;
        nimcp_platform_mutex_unlock(&wm->mutex);
        return false;
    }

    // Update PE type
    wm->pe_type = pe_type;

    LOG_INFO("Successfully switched to PE type %d", pe_type);

    nimcp_platform_mutex_unlock(&wm->mutex);
    return true;
}

//=============================================================================
// BRAIN IMMUNE INTEGRATION
//=============================================================================

/**
 * @brief Callback for inflammation state changes
 *
 * WHAT: Update working memory capacity based on inflammation level
 * WHY:  Model cytokine impairment of prefrontal cortex function
 * HOW:  Apply capacity penalty based on inflammation severity
 *
 * BIOLOGICAL BASIS:
 * IL-6 and TNF-alpha impair prefrontal working memory representations
 * during systemic inflammation (illness, stress, immune activation)
 */
static void wm_inflammation_callback(
    brain_immune_system_t* system,
    const brain_inflammation_site_t* site,
    void* user_data)
{
    (void)system;

    if (!user_data || !site) {
        return;
    }

    working_memory_t* wm = (working_memory_t*)user_data;

    nimcp_platform_mutex_lock(&wm->mutex);

    // Map inflammation level to capacity penalty
    uint32_t penalty = 0;
    switch (site->level) {
        case INFLAMMATION_NONE:
            penalty = 0;
            break;
        case INFLAMMATION_LOCAL:
            penalty = 1;  // -1 item (7 → 6)
            break;
        case INFLAMMATION_REGIONAL:
            penalty = 2;  // -2 items (7 → 5)
            break;
        case INFLAMMATION_SYSTEMIC:
            penalty = 3;  // -3 items (7 → 4)
            break;
        case INFLAMMATION_STORM:
            penalty = 4;  // -4 items (7 → 3, minimum)
            break;
    }

    wm->inflammation_capacity_penalty = penalty;

    LOG_INFO("WM inflammation callback: level=%s, penalty=%u, effective_capacity=%u",
             brain_immune_inflammation_to_string(site->level),
             penalty,
             wm->capacity > penalty ? wm->capacity - penalty : 3);

    // If current size exceeds new effective capacity, evict lowest-salience items
    uint32_t effective_capacity = wm->capacity > penalty ? wm->capacity - penalty : 3;
    if (effective_capacity < 3) {
        effective_capacity = 3;  // Minimum 3 items even under cytokine storm
    }

    while (wm->current_size > effective_capacity) {
        // Find and evict lowest-salience item
        int lowest_idx = working_memory_find_lowest_salience(wm, NULL);
        if (lowest_idx < 0) {
            break;  // No more items
        }

        LOG_DEBUG("Evicting item %d due to inflammation (size=%u, effective_cap=%u)",
                  lowest_idx, wm->current_size, effective_capacity);

        working_memory_remove(wm, (uint32_t)lowest_idx);
    }

    nimcp_platform_mutex_unlock(&wm->mutex);
}

bool working_memory_connect_immune(
    working_memory_t* wm,
    struct brain_immune_system* immune)
{
    // WHAT: Connect working memory to brain immune system
    // WHY:  Enable bidirectional immune-cognitive integration
    // HOW:  Store immune pointer, register inflammation callback

    if (!wm) {
        set_error("NULL working memory");
        return false;
    }

    if (!immune) {
        set_error("NULL immune system");
        return false;
    }

    nimcp_platform_mutex_lock(&wm->mutex);

    wm->immune = immune;
    wm->immune_integration_enabled = true;
    wm->inflammation_capacity_penalty = 0;
    wm->last_stress_signal_time_ms = 0.0f;

    // Register callback for inflammation events
    brain_immune_set_inflammation_callback(immune, wm_inflammation_callback, wm);

    LOG_INFO("Working memory connected to brain immune system");

    nimcp_platform_mutex_unlock(&wm->mutex);
    return true;
}

void working_memory_disconnect_immune(working_memory_t* wm)
{
    // WHAT: Disconnect from immune system
    // WHY:  Clean shutdown or disable modulation
    // HOW:  Clear pointer, restore full capacity

    if (!wm) {
        return;
    }

    nimcp_platform_mutex_lock(&wm->mutex);

    if (wm->immune) {
        // Unregister callback
        brain_immune_set_inflammation_callback(wm->immune, NULL, NULL);
    }

    wm->immune = NULL;
    wm->immune_integration_enabled = false;
    wm->inflammation_capacity_penalty = 0;

    LOG_INFO("Working memory disconnected from brain immune system");

    nimcp_platform_mutex_unlock(&wm->mutex);
}

uint32_t working_memory_get_effective_capacity(const working_memory_t* wm)
{
    // WHAT: Get current capacity after inflammation and sleep penalties
    // WHY:  Check available slots for new items
    // HOW:  base_capacity × sleep_factor - inflammation_penalty, minimum 3

    if (!wm) {
        return 0;
    }

    // Start with base capacity
    float effective = (float)wm->capacity;

    // Apply sleep state modulation
    float sleep_capacity_factor = working_memory_sleep_capacity_for_state(wm->current_sleep_state);
    effective *= sleep_capacity_factor;

    // Apply inflammation penalty
    if (wm->immune_integration_enabled) {
        effective -= (float)wm->inflammation_capacity_penalty;
    }

    // Floor to integer, minimum 3 items
    uint32_t final_capacity = (uint32_t)effective;
    if (final_capacity < 3) {
        final_capacity = 3;
    }

    return final_capacity;
}

bool working_memory_is_immune_impaired(const working_memory_t* wm)
{
    // WHAT: Check if inflammation is reducing capacity
    // WHY:  Detect cognitive impairment
    // HOW:  Compare effective to base capacity

    if (!wm) {
        return false;
    }

    return wm->immune_integration_enabled && wm->inflammation_capacity_penalty > 0;
}

bool working_memory_signal_stress(
    working_memory_t* wm,
    float stress_level)
{
    // WHAT: Signal immune system about cognitive stress
    // WHY:  Working memory overload triggers immune response
    // HOW:  Release IL-6 cytokine via brain immune system

    if (!wm) {
        return false;
    }

    if (!wm->immune_integration_enabled || !wm->immune) {
        return false;
    }

    // Clamp stress level
    if (stress_level < 0.0f) stress_level = 0.0f;
    if (stress_level > 1.0f) stress_level = 1.0f;

    nimcp_platform_mutex_lock(&wm->mutex);

    // Release IL-6 cytokine (cognitive load signal)
    uint32_t cytokine_id = 0;
    int result = brain_immune_release_cytokine(
        wm->immune,
        CYTOKINE_IL6,
        0,  // No specific source cell (working memory itself)
        stress_level,
        0,  // Broadcast (target_region = 0)
        &cytokine_id
    );

    if (result == 0) {
        wm->last_stress_signal_time_ms = (float)nimcp_time_get_ms();
        LOG_DEBUG("WM stress signal: level=%.2f, cytokine_id=%u", stress_level, cytokine_id);
    }

    nimcp_platform_mutex_unlock(&wm->mutex);

    return result == 0;
}

//=============================================================================
// SLEEP STATE INTEGRATION
//=============================================================================

/**
 * @brief Set sleep state for working memory modulation
 *
 * WHAT: Update current sleep state to modulate WM capacity and decay
 * WHY:  Sleep state affects working memory performance (biological)
 * HOW:  Store state, apply modulation factors from sleep bridge
 *
 * BIOLOGICAL BASIS:
 * - AWAKE: Full WM capacity (~7±2 items), normal decay
 * - DROWSY: Reduced capacity (~5 items), faster decay
 * - NREM: Minimal capacity (offline processing)
 * - REM: Limited capacity (dream narrative)
 *
 * @param wm Working memory instance (non-NULL)
 * @param state New sleep state
 * @return true on success, false on NULL pointer
 */
bool working_memory_set_sleep_state(working_memory_t* wm, sleep_state_t state)
{
    if (!wm) {
        set_error("NULL working_memory_t");
        return false;
    }

    nimcp_platform_mutex_lock(&wm->mutex);
    wm->current_sleep_state = state;
    nimcp_platform_mutex_unlock(&wm->mutex);

    LOG_DEBUG("WM sleep state changed to %d", state);
    return true;
}

/**
 * @brief Get current sleep state
 *
 * WHAT: Query current sleep/wake state
 * WHY:  Check what modulation is being applied
 * HOW:  Return current_sleep_state field
 *
 * @param wm Working memory instance (non-NULL)
 * @return Current sleep state, or SLEEP_STATE_AWAKE if NULL
 */
sleep_state_t working_memory_get_sleep_state(const working_memory_t* wm)
{
    if (!wm) {
        return SLEEP_STATE_AWAKE;
    }

    nimcp_platform_mutex_lock((nimcp_platform_mutex_t*)&wm->mutex);
    sleep_state_t state = wm->current_sleep_state;
    nimcp_platform_mutex_unlock((nimcp_platform_mutex_t*)&wm->mutex);

    return state;
}

//=============================================================================
// UTILITY FUNCTIONS
//=============================================================================

/**
 * @brief Check if working memory is empty
 *
 * WHAT: Returns true if no items are stored
 * WHY:  Useful for quick emptiness checks before operations
 * HOW:  Check current_size == 0
 *
 * @param wm Working memory instance
 * @return true if empty, false if has items or NULL
 */
bool working_memory_is_empty(const working_memory_t* wm)
{
    if (!wm) {
        return true;  // NULL is considered empty
    }

    return wm->current_size == 0;
}

/**
 * @brief Get salience value for an item at given index
 *
 * WHAT: Retrieve the salience (importance) score for a specific item
 * WHY:  Allow external modules to query item priorities
 * HOW:  Direct array access with bounds checking
 *
 * @param wm Working memory instance (non-NULL)
 * @param index Item index [0, current_size)
 * @param salience Output: salience value [0.0, 1.0] (non-NULL)
 * @return true on success, false on invalid index or NULL params
 */
bool working_memory_get_salience(
    const working_memory_t* wm,
    uint32_t index,
    float* salience
)
{
    // Guard: NULL working memory
    if (!wm) {
        return false;
    }

    // Guard: NULL output
    if (!salience) {
        return false;
    }

    // Guard: Invalid index
    if (index >= wm->current_size) {
        return false;
    }

    // Thread-safe access
    nimcp_platform_mutex_lock((nimcp_platform_mutex_t*)&wm->mutex);
    *salience = wm->salience[index];
    nimcp_platform_mutex_unlock((nimcp_platform_mutex_t*)&wm->mutex);

    return true;
}
