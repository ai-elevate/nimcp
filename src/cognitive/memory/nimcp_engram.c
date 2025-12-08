/**
 * @file nimcp_engram.c
 * @brief Implementation of memory engram system
 *
 * WHAT: Memory traces stored as distributed synaptic patterns
 * WHY:  Enable realistic memory encoding, consolidation, recall
 * HOW:  Track engram cells, manage consolidation, pattern completion
 *
 * NIMCP STANDARDS:
 * - Guard clauses (no nested ifs)
 * - Functions < 50 lines
 * - WHAT-WHY-HOW documentation
 * - Single Responsibility Principle
 * - 100% test coverage
 *
 * BIO-ASYNC INTEGRATION:
 * - Module ID: 0x0330 (BIO_MODULE_MEMORY)
 * - Publishes: memory encoding, recall, consolidation events
 * - Subscribes: None (passive memory store)
 *
 * @version Phase M1: Memory Engrams - Core Implementation with Bio-Async
 * @date 2025-11-28
 */

#define LOG_MODULE "engram"

#include "cognitive/memory/nimcp_engram.h"
#include "security/nimcp_security.h"
#include "security/nimcp_blood_brain_barrier.h"

#include "nimcp.h"  // For NIMCP_ERROR_* codes
#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"
#include "utils/memory/nimcp_unified_memory.h"
#include "utils/memory/nimcp_memory_pool.h"
#include "utils/logging/nimcp_logging.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "utils/memory/nimcp_memory_guards.h"  // For nimcp_calloc/nimcp_free

//=============================================================================
// BIO-ASYNC MODULE REGISTRATION
//=============================================================================

#define BIO_MODULE_MEMORY 0x0330

// Event types
#define ENGRAM_EVENT_ENCODED "engram.encoded"
#define ENGRAM_EVENT_RECALLED "engram.recalled"
#define ENGRAM_EVENT_CONSOLIDATED "engram.consolidated"
#define ENGRAM_EVENT_DECAYED "engram.decayed"

//=============================================================================
// BIO-ASYNC HANDLERS (Forward declarations)
//=============================================================================

static nimcp_error_t handle_memory_query(
    const void* msg, size_t msg_size,
    nimcp_bio_promise_t response_promise, void* user_data);

static void bio_broadcast_engram_encoded(engram_system_t* system, uint32_t engram_id, float strength);
static void bio_broadcast_engram_recalled(engram_system_t* system, uint32_t engram_id, float confidence);

//=============================================================================
// CONSTANTS
//=============================================================================

#define ENGRAM_INITIAL_CAPACITY 512
#define ENGRAM_GROWTH_FACTOR 2.0f

//=============================================================================
// HELPER FUNCTIONS
//=============================================================================

/**
 * @brief Calculate overlap between two neuron sets
 */
static uint32_t calculate_overlap(
    const uint32_t* set1, uint32_t count1,
    const uint32_t* set2, uint32_t count2) {

    // WHAT: Count neurons present in both sets
    // WHY:  Pattern matching for recall
    // HOW:  Nested loop comparison (optimizable with hash table)

    if (!set1 || !set2) return 0;

    uint32_t overlap = 0;
    for (uint32_t i = 0; i < count1; i++) {
        for (uint32_t j = 0; j < count2; j++) {
            if (set1[i] == set2[j]) {
                overlap++;
                break;
            }
        }
    }

    return overlap;
}

/**
 * @brief Expand engram array capacity
 */
static bool expand_engram_array(engram_system_t* system) {
    // WHAT: Double array capacity when full
    // WHY:  Dynamic growth as needed
    // HOW:  Realloc with growth factor

    if (!system) return false;

    uint32_t new_capacity = (uint32_t)(system->capacity * ENGRAM_GROWTH_FACTOR);
    memory_engram_t* new_array = (memory_engram_t*)nimcp_realloc(
        system->engrams,
        new_capacity * sizeof(memory_engram_t));

    if (!new_array) return false;

    // Zero new memory
    memset(new_array + system->capacity, 0,
           (new_capacity - system->capacity) * sizeof(memory_engram_t));

    system->engrams = new_array;
    system->capacity = new_capacity;

    return true;
}

/**
 * @brief Find free engram slot
 */
static memory_engram_t* find_free_slot(engram_system_t* system) {
    // WHAT: Locate inactive engram slot
    // WHY:  Reuse memory efficiently
    // HOW:  Linear search, expand if needed

    if (!system) return NULL;

    // Try to find inactive slot
    for (uint32_t i = 0; i < system->capacity; i++) {
        if (!system->engrams[i].active) {
            return &system->engrams[i];
        }
    }

    // Need to expand
    uint32_t old_capacity = system->capacity;
    if (!expand_engram_array(system)) {
        return NULL;
    }

    // Return first slot in new space
    return &system->engrams[old_capacity];
}

//=============================================================================
// LIFECYCLE FUNCTIONS
//=============================================================================

engram_system_t* engram_system_create(void) {
    // WHAT: Allocate and initialize engram system
    // WHY:  Required for memory trace tracking
    // HOW:  Allocate struct and array using unified memory with CoW support

    LOG_INFO("Creating engram system");

    engram_system_t* system = (engram_system_t*)nimcp_calloc(1, sizeof(engram_system_t));
    if (!system) {
        LOG_ERROR("Failed to allocate engram system");
        return NULL;
    }

    // Initialize unified memory manager for CoW support
    unified_mem_config_t mem_config = unified_mem_default_config();
    mem_config.enable_cow = true;
    mem_config.enable_tracking = true;
    system->mem_manager = unified_mem_create(&mem_config);

    if (system->mem_manager) {
        // Allocate engram array via unified memory (enables O(1) brain cloning)
        unified_mem_request_t req = unified_mem_request(
            ENGRAM_INITIAL_CAPACITY * sizeof(memory_engram_t),
            NULL,  // Zero-initialized
            true   // Enable CoW
        );
        system->engrams_handle = unified_mem_alloc(system->mem_manager, &req);
        if (system->engrams_handle) {
            system->engrams = (memory_engram_t*)unified_mem_write(system->engrams_handle);
            LOG_DEBUG(LOG_MODULE, "Engram array allocated via unified memory with CoW support");
        }
    }

    // Fallback to direct allocation if unified memory unavailable
    if (!system->engrams) {
        system->engrams = (memory_engram_t*)nimcp_calloc(
            ENGRAM_INITIAL_CAPACITY, sizeof(memory_engram_t));
        if (!system->engrams) {
            if (system->mem_manager) unified_mem_destroy(system->mem_manager);
            nimcp_free(system);
            return NULL;
        }
        LOG_DEBUG(LOG_MODULE, "Engram array allocated via direct memory (no CoW)");
    }

    // Set initial capacity
    system->capacity = ENGRAM_INITIAL_CAPACITY;
    system->active_count = 0;
    system->next_engram_id = 1;  // Start at 1 (0 = invalid)

    // Set defaults
    system->systems_consolidation_enabled = true;
    system->hippocampal_capacity = 1.0f;
    system->cortical_capacity = 1.0f;
    system->sleep_consolidation_rate = 2.0f;  // 2x faster during sleep
    system->baseline_decay_rate = ENGRAM_BASE_DECAY_RATE;
    system->use_interference = true;
    system->separation_threshold = 0.3f;
    system->completion_threshold = 0.4f;

    // Enable integrations
    system->integrate_with_sleep = true;
    system->integrate_with_emotion = true;
    system->integrate_with_consolidation = true;

    // Phase 1.5: Initialize memory pool for engram allocations
    memory_pool_config_t engram_pool_config = {
        .block_size = sizeof(memory_engram_t),
        .num_blocks = ENGRAM_INITIAL_CAPACITY,  // Match initial array capacity
        .alignment = 16,   // SIMD alignment
        .enable_tracking = false,
        .enable_guard_pages = false
    };
    system->engram_pool = memory_pool_create(&engram_pool_config);

    
    // Bio-async registration
    system->bio_ctx = NULL;
    system->bio_async_enabled = false;
    if (bio_router_is_initialized()) {
        bio_module_info_t bio_info = {
            .module_id = BIO_MODULE_MEMORY,
            .module_name = "engram",
            .inbox_capacity = 32,
            .user_data = system
        };
        system->bio_ctx = bio_router_register_module(&bio_info);
        if (system->bio_ctx) {
            system->bio_async_enabled = true;
            bio_router_register_handler(system->bio_ctx, BIO_MSG_WORKING_MEMORY_RETRIEVE,
                                        handle_memory_query);
            LOG_INFO(LOG_MODULE, "Bio-async registered (module_id=0x%04X)", BIO_MODULE_MEMORY);
        }
    }

return system;
}

/*=============================================================================
 * BIO-ASYNC HANDLER IMPLEMENTATIONS
 *============================================================================*/

static nimcp_error_t handle_memory_query(
    const void* msg, size_t msg_size,
    nimcp_bio_promise_t response_promise, void* user_data)
{
    (void)msg_size;
    (void)response_promise;
    if (!msg || !user_data) { return NIMCP_ERROR_NULL_ARG; }
    engram_system_t* system = (engram_system_t*)user_data;
    LOG_DEBUG(LOG_MODULE, "Received memory query, active engrams=%u", system->active_count);
    return NIMCP_SUCCESS;
}

static void bio_broadcast_engram_encoded(engram_system_t* system, uint32_t engram_id, float strength) {
    if (!system || !system->bio_async_enabled || !system->bio_ctx) { return; }
    // Use salience response for engram encoded notification
    bio_msg_salience_response_t msg = {0};
    bio_msg_init_header(&msg.header, BIO_MSG_SALIENCE_RESPONSE,
                        bio_module_context_get_id(system->bio_ctx), 0, sizeof(msg));
    msg.header.flags |= BIO_MSG_FLAG_BROADCAST;
    msg.stimulus_id = engram_id;
    msg.salience_score = strength;
    msg.attention_priority = strength;
    msg.requires_immediate_attention = (strength > 0.8f);
    bio_router_broadcast(system->bio_ctx, &msg, sizeof(msg));
    LOG_DEBUG(LOG_MODULE, "Broadcast engram encoded: id=%u, strength=%.2f", engram_id, strength);
}

static void bio_broadcast_engram_recalled(engram_system_t* system, uint32_t engram_id, float confidence) {
    if (!system || !system->bio_async_enabled || !system->bio_ctx) { return; }
    // Use salience response for engram recalled notification
    bio_msg_salience_response_t msg = {0};
    bio_msg_init_header(&msg.header, BIO_MSG_SALIENCE_RESPONSE,
                        bio_module_context_get_id(system->bio_ctx), 0, sizeof(msg));
    msg.header.flags |= BIO_MSG_FLAG_BROADCAST;
    msg.stimulus_id = engram_id;
    msg.salience_score = confidence;
    msg.attention_priority = confidence;
    msg.requires_immediate_attention = false;
    bio_router_broadcast(system->bio_ctx, &msg, sizeof(msg));
    LOG_DEBUG(LOG_MODULE, "Broadcast engram recalled: id=%u, confidence=%.2f", engram_id, confidence);
}

void engram_system_destroy(engram_system_t* system) {
    // WHAT: Free all engram system resources
    // WHY:  Prevent memory leaks
    // HOW:  Free unified memory handles, then struct

    if (!system) return;

    // Free engram array (unified memory or direct)
    if (system->engrams_handle) {
        unified_mem_free(system->engrams_handle);
        system->engrams_handle = NULL;
        system->engrams = NULL;  // Was pointing into unified memory
    } else if (system->engrams) {
        nimcp_free(system->engrams);
        system->engrams = NULL;
    }

    // Destroy unified memory manager
    if (system->mem_manager) {
        unified_mem_destroy(system->mem_manager);
        system->mem_manager = NULL;
    }

    // Phase 1.5: Destroy memory pool
    if (system->engram_pool) {
        memory_pool_destroy(system->engram_pool);
    }

    // Unregister from bio-router
    if (system->bio_async_enabled && system->bio_ctx) {
        bio_router_unregister_module(system->bio_ctx);
        system->bio_ctx = NULL;
        system->bio_async_enabled = false;
    }

    nimcp_free(system);
}

void engram_system_reset(engram_system_t* system) {
    // WHAT: Clear all engrams, preserve configuration
    // WHY:  Fresh start without reallocating
    // HOW:  Zero engram array, reset counters

    if (!system) return;

    // Save configuration
    bool sys_consol = system->systems_consolidation_enabled;
    float sleep_rate = system->sleep_consolidation_rate;
    float decay = system->baseline_decay_rate;
    bool interference = system->use_interference;
    float sep_thresh = system->separation_threshold;
    float comp_thresh = system->completion_threshold;
    bool int_sleep = system->integrate_with_sleep;
    bool int_emotion = system->integrate_with_emotion;
    bool int_consol = system->integrate_with_consolidation;

    // Clear engrams
    memset(system->engrams, 0, system->capacity * sizeof(memory_engram_t));

    // Reset counters
    system->active_count = 0;
    system->next_engram_id = 1;
    system->labile_count = 0;
    system->consolidating_count = 0;
    system->consolidated_count = 0;
    system->replays_during_sleep = 0;
    system->total_encodings = 0;
    system->total_recalls = 0;
    system->total_consolidations = 0;
    system->total_extinctions = 0;
    system->average_consolidation_time = 0.0f;

    // Restore configuration
    system->systems_consolidation_enabled = sys_consol;
    system->sleep_consolidation_rate = sleep_rate;
    system->baseline_decay_rate = decay;
    system->use_interference = interference;
    system->separation_threshold = sep_thresh;
    system->completion_threshold = comp_thresh;
    system->integrate_with_sleep = int_sleep;
    system->integrate_with_emotion = int_emotion;
    system->integrate_with_consolidation = int_consol;
}

//=============================================================================
// ENCODING FUNCTIONS
//=============================================================================

uint64_t engram_encode(
    engram_system_t* system,
    const uint32_t* neuron_ids,
    const float* activations,
    uint32_t count,
    memory_type_t memory_type,
    emotional_tag_t emotion) {

    // WHAT: Create new engram from neural activity
    // WHY:  Store experience as memory trace
    // HOW:  Tag neurons, record activations, initialize state

    // Guard clauses
    if (!system) return 0;
    if (!neuron_ids || !activations) return 0;
    if (count == 0) return 0;
    if (count > ENGRAM_MAX_NEURONS) {
        count = ENGRAM_MAX_NEURONS;  // Truncate to max
    }

    // Find free slot
    memory_engram_t* engram = find_free_slot(system);
    if (!engram) return 0;

    // Initialize engram
    engram->active = true;
    engram->engram_id = system->next_engram_id++;
    engram->memory_type = memory_type;
    engram->neuron_count = count;

    // Copy neurons and activations
    memcpy(engram->neuron_ids, neuron_ids, count * sizeof(uint32_t));
    memcpy(engram->neuron_activation, activations, count * sizeof(float));

    // Set initial state
    engram->state = ENGRAM_STATE_ENCODING;
    engram->consolidation_strength = 0.0f;
    engram->primary_location = ENGRAM_LOCATION_HIPPOCAMPUS;  // Always starts here
    engram->secondary_location = ENGRAM_LOCATION_CORTEX;     // Target for systems consolidation

    // Temporal info (get from system time - placeholder for now)
    engram->encoding_time_us = 0;  // Will be set by caller
    engram->last_reactivation_us = 0;
    engram->reactivation_count = 0;
    engram->decay_rate = system->baseline_decay_rate;

    // IEG tagging - strength modulated by arousal
    engram->is_tagged = (emotion.arousal > 0.5f);  // Tag if arousal above threshold
    // Tag strength scales with arousal: higher arousal = stronger tagging
    engram->tag_strength = 0.5f + (emotion.arousal * 0.5f);  // Range: 0.5 to 1.0
    engram->tag_onset_time_us = 0;  // Will be set by caller

    // Emotional context
    engram->emotion = emotion;
    engram->vividness = 1.0f;  // Starts vivid
    engram->confidence = 1.0f;

    // Emotional enhancement
    if (system->integrate_with_emotion && emotion.arousal > 0.6f) {
        engram->decay_rate *= 0.5f;  // Emotional memories resist forgetting
        engram->vividness *= 1.3f;   // More vivid
    }

    // Reconsolidation
    engram->is_reconsolidating = false;
    engram->reconsolidation_start_us = 0;

    // Statistics
    engram->recall_latency_ms = 0.0f;
    engram->successful_recalls = 0;

    // Update system counters
    system->active_count++;
    system->labile_count++;
    system->total_encodings++;

    return engram->engram_id;
}

//=============================================================================
// RECALL FUNCTIONS
//=============================================================================

uint64_t engram_recall(
    engram_system_t* system,
    const uint32_t* cue_neurons,
    uint32_t cue_count,
    uint32_t* activation_out,
    float* activations_out,
    uint32_t max_activation_count,
    float* confidence_out) {

    // WHAT: Reactivate engram from partial cue
    // WHY:  Retrieve stored memory
    // HOW:  Pattern completion via overlap matching

    // Guard clauses
    if (!system) return 0;
    if (!cue_neurons) return 0;
    if (cue_count == 0) return 0;
    if (!activation_out || !activations_out) return 0;

    float best_match = 0.0f;
    uint64_t best_engram_id = 0;
    memory_engram_t* best_engram = NULL;

    // Search for best matching engram
    for (uint32_t i = 0; i < system->capacity; i++) {
        memory_engram_t* engram = &system->engrams[i];

        if (!engram->active) continue;
        if (engram->state == ENGRAM_STATE_DEGRADING) continue;

        // Calculate overlap
        uint32_t overlap = calculate_overlap(
            cue_neurons, cue_count,
            engram->neuron_ids, engram->neuron_count);

        float match = (float)overlap / engram->neuron_count;

        // Boost by consolidation strength
        match *= engram->consolidation_strength;

        // Check threshold and update best
        if (match > system->completion_threshold && match > best_match) {
            best_match = match;
            best_engram_id = engram->engram_id;
            best_engram = engram;
        }
    }

    // No match found
    if (best_engram_id == 0) {
        if (confidence_out) *confidence_out = 0.0f;
        return 0;
    }

    // Reactivate engram
    uint32_t output_count = best_engram->neuron_count;
    if (output_count > max_activation_count) {
        output_count = max_activation_count;
    }

    memcpy(activation_out, best_engram->neuron_ids,
           output_count * sizeof(uint32_t));
    memcpy(activations_out, best_engram->neuron_activation,
           output_count * sizeof(float));

    // Update engram statistics
    best_engram->reactivation_count++;
    best_engram->successful_recalls++;
    best_engram->last_reactivation_us = 0;  // Will be set by caller

    // Trigger reconsolidation if consolidated
    if (best_engram->state == ENGRAM_STATE_CONSOLIDATED) {
        engram_trigger_reconsolidation(system, best_engram_id);
    }

    // Update system statistics
    system->total_recalls++;

    // Output confidence
    if (confidence_out) {
        *confidence_out = best_match * best_engram->confidence;
    }

    return best_engram_id;
}

bool engram_recognize(
    engram_system_t* system,
    const uint32_t* pattern,
    uint32_t count,
    float* familiarity_out) {

    // WHAT: Test if pattern is familiar
    // WHY:  Recognition faster than recall
    // HOW:  Check overlap without full reactivation

    // Guard clauses
    if (!system) return false;
    if (!pattern) return false;
    if (count == 0) return false;

    float max_familiarity = 0.0f;

    // Check all engrams
    for (uint32_t i = 0; i < system->capacity; i++) {
        memory_engram_t* engram = &system->engrams[i];

        if (!engram->active) continue;

        // Calculate overlap
        uint32_t overlap = calculate_overlap(
            pattern, count,
            engram->neuron_ids, engram->neuron_count);

        float familiarity = (float)overlap / count;

        // Weight by consolidation
        familiarity *= engram->consolidation_strength;

        if (familiarity > max_familiarity) {
            max_familiarity = familiarity;
        }
    }

    if (familiarity_out) {
        *familiarity_out = max_familiarity;
    }

    return (max_familiarity >= ENGRAM_RECOGNITION_THRESHOLD);
}

//=============================================================================
// CONSOLIDATION FUNCTIONS
//=============================================================================

void engram_consolidate_update(
    engram_system_t* system,
    float dt,
    bool is_sleeping) {
    // Process pending bio-async messages
    if (system && system->bio_async_enabled && system->bio_ctx) {
        bio_router_process_inbox(system->bio_ctx, 5);
    }


    // WHAT: Advance consolidation for all engrams
    // WHY:  Labile → stable transition
    // HOW:  Time-dependent strengthening, sleep boost

    if (!system) return;
    if (dt <= 0.0f) return;

    float consolidation_rate = is_sleeping ?
        (dt / ENGRAM_SYNAPTIC_CONSOLIDATION_TIME) * system->sleep_consolidation_rate :
        (dt / ENGRAM_SYNAPTIC_CONSOLIDATION_TIME);

    uint32_t labile = 0;
    uint32_t consolidating = 0;
    uint32_t consolidated = 0;

    for (uint32_t i = 0; i < system->capacity; i++) {
        memory_engram_t* engram = &system->engrams[i];

        if (!engram->active) continue;

        // Update based on state
        if (engram->state == ENGRAM_STATE_ENCODING) {
            // Transition to labile and start consolidating
            engram->state = ENGRAM_STATE_LABILE;
            engram->consolidation_strength += consolidation_rate;

            if (engram->consolidation_strength >= 1.0f) {
                engram->consolidation_strength = 1.0f;
                engram->state = ENGRAM_STATE_CONSOLIDATED;
                system->total_consolidations++;
                consolidated++;
            } else if (engram->consolidation_strength > 0.0f) {
                engram->state = ENGRAM_STATE_CONSOLIDATING;
                consolidating++;
            } else {
                labile++;
            }
        }
        else if (engram->state == ENGRAM_STATE_LABILE ||
                 engram->state == ENGRAM_STATE_CONSOLIDATING) {

            // Increase consolidation strength
            engram->consolidation_strength += consolidation_rate;

            if (engram->consolidation_strength >= 1.0f) {
                engram->consolidation_strength = 1.0f;
                engram->state = ENGRAM_STATE_CONSOLIDATED;
                system->total_consolidations++;
                consolidated++;
            } else {
                engram->state = ENGRAM_STATE_CONSOLIDATING;
                consolidating++;
            }
        }
        else if (engram->state == ENGRAM_STATE_CONSOLIDATED) {
            consolidated++;
        }
        else if (engram->state == ENGRAM_STATE_RECONSOLIDATING) {
            // Check if reconsolidation window expired
            // (placeholder - needs time tracking)
            consolidating++;
        }
    }

    // Update counts
    system->labile_count = labile;
    system->consolidating_count = consolidating;
    system->consolidated_count = consolidated;
}

void engram_sleep_replay(
    engram_system_t* system,
    uint32_t replay_count) {

    // WHAT: Reactivate engrams during sleep
    // WHY:  Sleep strengthens memories
    // HOW:  Strengthen recently encoded engrams

    if (!system) return;
    if (replay_count == 0) return;

    uint32_t replayed = 0;

    // Prioritize tagged, labile and consolidating engrams
    for (uint32_t i = 0; i < system->capacity && replayed < replay_count; i++) {
        memory_engram_t* engram = &system->engrams[i];

        if (!engram->active) continue;

        // Replay tagged engrams (high arousal) or those in consolidation states
        if (engram->is_tagged ||
            engram->state == ENGRAM_STATE_LABILE ||
            engram->state == ENGRAM_STATE_CONSOLIDATING) {

            // Replay strengthens
            engram->consolidation_strength += 0.01f;
            if (engram->consolidation_strength > 1.0f) {
                engram->consolidation_strength = 1.0f;
            }

            engram->reactivation_count++;
            replayed++;
        }
    }

    system->replays_during_sleep += replayed;
}

//=============================================================================
// RECONSOLIDATION FUNCTIONS
//=============================================================================

void engram_trigger_reconsolidation(
    engram_system_t* system,
    uint64_t engram_id) {

    // WHAT: Make engram temporarily labile
    // WHY:  Recalled memories can be updated
    // HOW:  Set reconsolidation flag and timer

    if (!system) return;
    if (engram_id == 0) return;

    memory_engram_t* engram = engram_get_by_id(system, engram_id);
    if (!engram) return;

    // Only consolidated engrams can reconsolidate
    if (engram->state != ENGRAM_STATE_CONSOLIDATED) return;

    engram->is_reconsolidating = true;
    engram->state = ENGRAM_STATE_RECONSOLIDATING;
    engram->reconsolidation_start_us = 0;  // Will be set by caller
}

bool engram_block_reconsolidation(
    engram_system_t* system,
    uint64_t engram_id) {

    // WHAT: Prevent engram restabilization
    // WHY:  Therapeutic weakening of maladaptive memories
    // HOW:  Set to degrading state

    if (!system) return false;
    if (engram_id == 0) return false;

    memory_engram_t* engram = engram_get_by_id(system, engram_id);
    if (!engram) return false;

    if (!engram->is_reconsolidating) return false;

    // Block = put in degrading state
    engram->state = ENGRAM_STATE_DEGRADING;
    engram->is_reconsolidating = false;
    engram->consolidation_strength *= 0.5f;  // Weaken significantly

    return true;
}

//=============================================================================
// FORGETTING AND EXTINCTION
//=============================================================================

void engram_apply_decay(
    engram_system_t* system,
    float dt) {

    // WHAT: Natural forgetting over time
    // WHY:  Realistic memory decay
    // HOW:  Exponential decay of unused engrams

    if (!system) return;
    if (dt <= 0.0f) return;

    for (uint32_t i = 0; i < system->capacity; i++) {
        memory_engram_t* engram = &system->engrams[i];

        if (!engram->active) continue;

        // Only decay consolidated engrams
        if (engram->state != ENGRAM_STATE_CONSOLIDATED) continue;

        // Apply decay
        float decay = engram->decay_rate * dt;
        engram->consolidation_strength -= decay;
        engram->vividness -= decay * 0.5f;

        // Clamp
        if (engram->consolidation_strength < 0.0f) {
            engram->consolidation_strength = 0.0f;
        }
        if (engram->vividness < 0.0f) {
            engram->vividness = 0.0f;
        }

        // Mark as degrading if very weak
        if (engram->consolidation_strength < 0.1f) {
            engram->state = ENGRAM_STATE_DEGRADING;
        }

        // Mark as forgotten if extremely weak (but keep active for tracking)
        // Very old/forgotten memories remain in system as degraded traces
        if (engram->consolidation_strength < 0.01f) {
            engram->state = ENGRAM_STATE_DEGRADING;
            // Don't deactivate - allow tracking of degraded memories
        }
    }
}

void engram_extinction(
    engram_system_t* system,
    uint64_t engram_id,
    float extinction_strength) {

    // WHAT: Active weakening through unreinforced recall
    // WHY:  Model extinction learning
    // HOW:  Direct reduction of consolidation strength

    if (!system) return;
    if (engram_id == 0) return;
    if (extinction_strength <= 0.0f) return;

    memory_engram_t* engram = engram_get_by_id(system, engram_id);
    if (!engram) return;

    // Weaken engram - affects consolidation, confidence, and vividness
    engram->consolidation_strength -= extinction_strength;
    if (engram->consolidation_strength < 0.0f) {
        engram->consolidation_strength = 0.0f;
    }

    // Extinction also reduces confidence and vividness of the memory
    engram->confidence -= extinction_strength * 0.5f;  // Confidence drops with extinction
    if (engram->confidence < 0.0f) {
        engram->confidence = 0.0f;
    }

    engram->vividness -= extinction_strength * 0.3f;  // Memory becomes less vivid
    if (engram->vividness < 0.0f) {
        engram->vividness = 0.0f;
    }

    // Update state
    if (engram->consolidation_strength < 0.1f) {
        engram->state = ENGRAM_STATE_DEGRADING;
        system->total_extinctions++;
    }
}

//=============================================================================
// QUERY FUNCTIONS
//=============================================================================

memory_engram_t* engram_get_by_id(
    engram_system_t* system,
    uint64_t engram_id) {

    // WHAT: Find engram by ID
    // WHY:  Access for inspection/modification
    // HOW:  Linear search

    if (!system) return NULL;
    if (engram_id == 0) return NULL;

    for (uint32_t i = 0; i < system->capacity; i++) {
        if (system->engrams[i].active &&
            system->engrams[i].engram_id == engram_id) {
            return &system->engrams[i];
        }
    }

    return NULL;
}

engram_state_t engram_get_state(
    const engram_system_t* system,
    uint64_t engram_id) {

    if (!system || engram_id == 0) return ENGRAM_STATE_DEGRADING;

    for (uint32_t i = 0; i < system->capacity; i++) {
        if (system->engrams[i].active &&
            system->engrams[i].engram_id == engram_id) {
            return system->engrams[i].state;
        }
    }

    return ENGRAM_STATE_DEGRADING;
}

float engram_get_consolidation_strength(
    const engram_system_t* system,
    uint64_t engram_id) {

    if (!system || engram_id == 0) return 0.0f;

    for (uint32_t i = 0; i < system->capacity; i++) {
        if (system->engrams[i].active &&
            system->engrams[i].engram_id == engram_id) {
            return system->engrams[i].consolidation_strength;
        }
    }

    return 0.0f;
}

bool engram_is_reconsolidating(
    const engram_system_t* system,
    uint64_t engram_id) {

    if (!system || engram_id == 0) return false;

    for (uint32_t i = 0; i < system->capacity; i++) {
        if (system->engrams[i].active &&
            system->engrams[i].engram_id == engram_id) {
            return system->engrams[i].is_reconsolidating;
        }
    }

    return false;
}

float engram_get_age_seconds(
    const engram_system_t* system,
    uint64_t engram_id,
    uint64_t current_time_us) {

    if (!system || engram_id == 0) return 0.0f;

    for (uint32_t i = 0; i < system->capacity; i++) {
        if (system->engrams[i].active &&
            system->engrams[i].engram_id == engram_id) {
            uint64_t age_us = current_time_us - system->engrams[i].encoding_time_us;
            return (float)age_us / 1000000.0f;
        }
    }

    return 0.0f;
}

uint32_t engram_get_active_count(const engram_system_t* system) {
    return system ? system->active_count : 0;
}

void engram_get_statistics(
    const engram_system_t* system,
    uint64_t* total_encodings_out,
    uint64_t* total_recalls_out,
    uint32_t* active_count_out) {

    if (!system) return;

    if (total_encodings_out) {
        *total_encodings_out = system->total_encodings;
    }
    if (total_recalls_out) {
        *total_recalls_out = system->total_recalls;
    }
    if (active_count_out) {
        *active_count_out = system->active_count;
    }
}
