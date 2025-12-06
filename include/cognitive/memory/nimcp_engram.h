/**
 * @file nimcp_engram.h
 * @brief Memory engram system - explicit memory trace representation
 *
 * WHAT: Physical traces of memory stored as synaptic patterns across neurons
 * WHY:  Enable realistic memory encoding, consolidation, recall, manipulation
 * HOW:  Track engram cells, weights, consolidation state, reactivation
 *
 * BIOLOGICAL BASIS:
 * - Engram cells: Neurons active during encoding and reactivated during recall
 * - Synaptic plasticity: LTP/LTD creates lasting structural changes
 * - Consolidation: Labile → stable transition over hours/days
 * - Systems consolidation: Hippocampus → cortex transfer over weeks/months
 *
 * NEUROSCIENCE REFERENCES:
 * - Tonegawa et al. (2015): "Memory engram cells have come of age"
 * - Liu et al. (2012): "Optogenetic stimulation of engram cells triggers recall"
 * - Ryan et al. (2015): "Engram cells retain memory under retrograde amnesia"
 * - Josselyn & Tonegawa (2020): "Memory engrams: Recalling the past"
 *
 * NIMCP STANDARDS:
 * - Guard clauses (no nested ifs)
 * - WHAT-WHY-HOW documentation
 * - Functions < 50 lines
 * - Single Responsibility Principle
 * - 100% test coverage required
 *
 * @version Phase M1: Memory Engrams - Core System
 * @date 2025-11-13
 */

#ifndef NIMCP_ENGRAM_H
#define NIMCP_ENGRAM_H

#include <stdint.h>
#include <stdbool.h>
#include "cognitive/nimcp_emotional_tagging.h"
#include "utils/memory/nimcp_unified_memory.h"

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// CONSTANTS
//=============================================================================

/* Maximum neurons per engram (distributed trace) */
#define ENGRAM_MAX_NEURONS 256

/* Maximum engrams tracked simultaneously */
#define ENGRAM_MAX_COUNT 512

/* Consolidation time constants (seconds) */
#define ENGRAM_SYNAPTIC_CONSOLIDATION_TIME (3600.0f * 6.0f)    /* 6 hours */
#define ENGRAM_SYSTEMS_CONSOLIDATION_TIME (3600.0f * 24.0f * 30.0f)  /* 30 days */

/* Tagging constants (c-fos/Arc expression) */
#define ENGRAM_TAG_DURATION (3600.0f * 4.0f)  /* 4 hours for IEG expression */
#define ENGRAM_TAG_WINDOW (3600.0f * 6.0f)    /* 6 hour tagging window */

/* Reactivation thresholds */
#define ENGRAM_RECALL_THRESHOLD 0.4f          /* Min activation for recall */
#define ENGRAM_RECOGNITION_THRESHOLD 0.6f     /* Min for strong recognition */

/* Decay and interference */
#define ENGRAM_BASE_DECAY_RATE 0.001f         /* Default forgetting rate */
#define ENGRAM_RECONSOLIDATION_WINDOW (3600.0f * 6.0f)  /* 6 hour labile period */

//=============================================================================
// ENUMERATIONS
//=============================================================================

/**
 * @brief Engram consolidation state
 */
typedef enum {
    ENGRAM_STATE_ENCODING,          /**< Currently being formed (seconds) */
    ENGRAM_STATE_LABILE,            /**< Unstable, vulnerable (minutes-hours) */
    ENGRAM_STATE_CONSOLIDATING,     /**< Protein synthesis dependent (hours) */
    ENGRAM_STATE_CONSOLIDATED,      /**< Stable, long-term (days-lifetime) */
    ENGRAM_STATE_RECONSOLIDATING,   /**< Temporarily labile after recall */
    ENGRAM_STATE_DEGRADING          /**< Actively weakening/forgotten */
} engram_state_t;

/**
 * @brief Memory system location
 */
typedef enum {
    ENGRAM_LOCATION_HIPPOCAMPUS,    /**< Short-term, spatial, episodic */
    ENGRAM_LOCATION_CORTEX,         /**< Long-term, semantic, consolidated */
    ENGRAM_LOCATION_AMYGDALA,       /**< Emotional memories */
    ENGRAM_LOCATION_STRIATUM,       /**< Procedural, motor memories */
    ENGRAM_LOCATION_DISTRIBUTED     /**< Across multiple regions */
} engram_location_t;

/**
 * @brief Memory type (Tulving, 1972)
 */
typedef enum {
    MEMORY_TYPE_EPISODIC,           /**< Personal events (what/where/when) */
    MEMORY_TYPE_SEMANTIC,           /**< Facts and knowledge */
    MEMORY_TYPE_PROCEDURAL,         /**< Skills and habits */
    MEMORY_TYPE_EMOTIONAL,          /**< Emotional associations */
    MEMORY_TYPE_WORKING             /**< Temporary active memory */
} memory_type_t;

//=============================================================================
// DATA STRUCTURES
//=============================================================================

/**
 * @brief Individual memory engram
 *
 * WHAT: Distributed pattern of neurons encoding a memory
 * WHY:  Explicit representation enables targeted memory operations
 * HOW:  Track neurons, consolidation state, reactivation
 */
typedef struct {
    // Identification
    uint64_t engram_id;             /**< Unique identifier */
    bool active;                    /**< Is this slot in use? */
    memory_type_t memory_type;      /**< What kind of memory? */

    // Neuron ensemble
    uint32_t neuron_ids[ENGRAM_MAX_NEURONS];  /**< Engram cells */
    float neuron_activation[ENGRAM_MAX_NEURONS]; /**< Current activation */
    uint32_t neuron_count;          /**< How many neurons */

    // Consolidation state
    engram_state_t state;           /**< Current consolidation state */
    float consolidation_strength;   /**< 0.0 (labile) to 1.0 (consolidated) */
    engram_location_t primary_location;  /**< Main storage location */
    engram_location_t secondary_location; /**< Systems consolidation target */

    // Temporal dynamics
    uint64_t encoding_time_us;      /**< When was this encoded? */
    uint64_t last_reactivation_us;  /**< Last recall time */
    uint32_t reactivation_count;    /**< How many times recalled */
    float decay_rate;               /**< Forgetting rate */

    // Activity-dependent tagging (IEG expression)
    bool is_tagged;                 /**< c-fos/Arc expression? */
    float tag_strength;             /**< Tag intensity [0-1] */
    uint64_t tag_onset_time_us;     /**< When was tag created? */

    // Content and context
    emotional_tag_t emotion;        /**< Emotional context */
    float vividness;                /**< Subjective clarity [0-1] */
    float confidence;               /**< Retrieval confidence [0-1] */

    // Reconsolidation
    bool is_reconsolidating;        /**< Temporarily labile? */
    uint64_t reconsolidation_start_us; /**< When recall made it labile */

    // Statistics
    float recall_latency_ms;        /**< Time to retrieve */
    uint32_t successful_recalls;    /**< Count of successful retrievals */

} memory_engram_t;

/**
 * @brief Complete engram memory system
 */
typedef struct {
    // Engram storage
    memory_engram_t* engrams;       /**< Dynamic array of engrams */
    uint32_t active_count;          /**< How many engrams active */
    uint32_t capacity;              /**< Current array capacity */
    uint64_t next_engram_id;        /**< ID counter */

    // Consolidation tracking
    uint32_t labile_count;          /**< Engrams in labile state */
    uint32_t consolidating_count;   /**< Engrams consolidating */
    uint32_t consolidated_count;    /**< Stable engrams */

    // Systems consolidation (hippocampus → cortex)
    bool systems_consolidation_enabled;
    float hippocampal_capacity;     /**< Hippocampus space [0-1] */
    float cortical_capacity;        /**< Cortex space [0-1] */

    // Sleep-dependent consolidation
    float sleep_consolidation_rate; /**< Consolidation during sleep */
    uint32_t replays_during_sleep;  /**< Replay count per sleep cycle */

    // Forgetting
    float baseline_decay_rate;      /**< Default forgetting rate */
    bool use_interference;          /**< Enable interference effects */

    // Pattern separation/completion
    float separation_threshold;     /**< Min difference for separate engrams */
    float completion_threshold;     /**< Min overlap for pattern completion */

    // Integration flags
    bool integrate_with_sleep;      /**< Consolidate during sleep? */
    bool integrate_with_emotion;    /**< Emotional enhancement? */
    bool integrate_with_consolidation; /**< Use consolidation system? */

    // Statistics
    uint64_t total_encodings;       /**< Lifetime encoding count */
    uint64_t total_recalls;         /**< Lifetime recall count */
    uint64_t total_consolidations;  /**< Successful consolidations */
    uint64_t total_extinctions;     /**< Memories extinguished */
    float average_consolidation_time; /**< Mean time to consolidate */

    // Phase 1.5: Memory pool for hot-path engram allocations
    void* engram_pool;              /**< Pool for memory_engram_t structs */

    // Unified memory integration (CoW support for brain cloning)
    unified_mem_manager_t mem_manager;  /**< Unified memory manager */
    unified_mem_handle_t engrams_handle; /**< CoW handle for engram array */

    // Bio-async integration
    void* bio_ctx;                  /**< bio_module_context_t pointer */
    bool bio_async_enabled;         /**< Bio-async registration status */

} engram_system_t;

//=============================================================================
// LIFECYCLE FUNCTIONS
//=============================================================================

/**
 * @brief Create engram system
 *
 * WHAT: Allocate and initialize memory engram system
 * WHY:  Required for explicit memory trace tracking
 * HOW:  Allocate dynamic array, set defaults
 *
 * @return Engram system pointer (NULL on failure)
 */
engram_system_t* engram_system_create(void);

/**
 * @brief Destroy engram system
 *
 * WHAT: Free all engram system resources
 * WHY:  Prevent memory leaks
 * HOW:  Free dynamic arrays, then system struct
 *
 * @param system Engram system to destroy
 */
void engram_system_destroy(engram_system_t* system);

/**
 * @brief Reset engram system
 *
 * WHAT: Clear all engrams, reset to initial state
 * WHY:  Enable fresh start while preserving configuration
 * HOW:  Zero engram array, reset counters
 *
 * @param system Engram system to reset
 */
void engram_system_reset(engram_system_t* system);

//=============================================================================
// ENCODING FUNCTIONS
//=============================================================================

/**
 * @brief Encode new memory engram
 *
 * WHAT: Create new engram from current neural activity pattern
 * WHY:  Store experience as explicit memory trace
 * HOW:  Tag active neurons, record activations, apply emotion
 *
 * @param system Engram system
 * @param neuron_ids Array of active neuron IDs
 * @param activations Array of neuron activations [0-1]
 * @param count Number of neurons (max ENGRAM_MAX_NEURONS)
 * @param memory_type Type of memory being encoded
 * @param emotion Emotional context
 * @return Engram ID (0 if failed)
 */
uint64_t engram_encode(
    engram_system_t* system,
    const uint32_t* neuron_ids,
    const float* activations,
    uint32_t count,
    memory_type_t memory_type,
    emotional_tag_t emotion);

//=============================================================================
// RECALL FUNCTIONS
//=============================================================================

/**
 * @brief Recall memory engram
 *
 * WHAT: Reactivate engram neurons based on cue pattern
 * WHY:  Retrieve stored memory
 * HOW:  Pattern completion from partial cue, reactivate ensemble
 *
 * @param system Engram system
 * @param cue_neurons Partial cue pattern
 * @param cue_count Number of cue neurons
 * @param activation_out Output: reactivated neurons (allocated by caller)
 * @param activations_out Output: activation levels
 * @param max_activation_count Max output neurons
 * @param confidence_out Output: recall confidence [0-1]
 * @return Engram ID of recalled memory (0 if no match)
 */
uint64_t engram_recall(
    engram_system_t* system,
    const uint32_t* cue_neurons,
    uint32_t cue_count,
    uint32_t* activation_out,
    float* activations_out,
    uint32_t max_activation_count,
    float* confidence_out);

/**
 * @brief Recognition test (familiarity)
 *
 * WHAT: Test if pattern has been seen before
 * WHY:  Recognition vs recall distinction
 * HOW:  Match against engrams without full reactivation
 *
 * @param system Engram system
 * @param pattern Pattern to recognize
 * @param count Pattern length
 * @param familiarity_out Output: familiarity strength [0-1]
 * @return True if recognized (above threshold)
 */
bool engram_recognize(
    engram_system_t* system,
    const uint32_t* pattern,
    uint32_t count,
    float* familiarity_out);

//=============================================================================
// CONSOLIDATION FUNCTIONS
//=============================================================================

/**
 * @brief Update consolidation state
 *
 * WHAT: Advance synaptic and systems consolidation
 * WHY:  Labile → stable transition, hippocampus → cortex transfer
 * HOW:  Time-dependent strengthening, protein synthesis simulation
 *
 * @param system Engram system
 * @param dt Time step (seconds)
 * @param is_sleeping Currently in sleep state?
 */
void engram_consolidate_update(
    engram_system_t* system,
    float dt,
    bool is_sleeping);

/**
 * @brief Sleep-dependent replay
 *
 * WHAT: Reactivate engrams during sleep for consolidation
 * WHY:  Sleep strengthens memories (Wilson & McNaughton, 1994)
 * HOW:  Replay sequences, strengthen synapses
 *
 * @param system Engram system
 * @param replay_count Number of replays to perform
 */
void engram_sleep_replay(
    engram_system_t* system,
    uint32_t replay_count);

//=============================================================================
// RECONSOLIDATION FUNCTIONS
//=============================================================================

/**
 * @brief Trigger reconsolidation
 *
 * WHAT: Make consolidated engram temporarily labile
 * WHY:  Recalled memories can be updated (Nader et al., 2000)
 * HOW:  Recall triggers reconsolidation window (~6 hours)
 *
 * @param system Engram system
 * @param engram_id Engram to reconsolidate
 */
void engram_trigger_reconsolidation(
    engram_system_t* system,
    uint64_t engram_id);

/**
 * @brief Block reconsolidation (therapeutic)
 *
 * WHAT: Prevent engram from restabilizing
 * WHY:  PTSD treatment, maladaptive memory weakening
 * HOW:  Simulate protein synthesis inhibitor
 *
 * @param system Engram system
 * @param engram_id Engram to block
 * @return True if successfully blocked
 */
bool engram_block_reconsolidation(
    engram_system_t* system,
    uint64_t engram_id);

//=============================================================================
// FORGETTING AND EXTINCTION
//=============================================================================

/**
 * @brief Natural forgetting (decay)
 *
 * WHAT: Gradual weakening of unused engrams
 * WHY:  Realistic forgetting curves (Ebbinghaus)
 * HOW:  Time-dependent strength decay
 *
 * @param system Engram system
 * @param dt Time step (seconds)
 */
void engram_apply_decay(
    engram_system_t* system,
    float dt);

/**
 * @brief Extinction (active unlearning)
 *
 * WHAT: Weaken engram through repeated unreinforced reactivation
 * WHY:  Model extinction learning (e.g., fear extinction)
 * HOW:  LTD-like weakening with each unreinforced recall
 *
 * @param system Engram system
 * @param engram_id Engram to extinguish
 * @param extinction_strength Strength of extinction [0-1]
 */
void engram_extinction(
    engram_system_t* system,
    uint64_t engram_id,
    float extinction_strength);

//=============================================================================
// QUERY FUNCTIONS
//=============================================================================

/**
 * @brief Get engram by ID
 *
 * WHAT: Retrieve engram pointer from ID
 * WHY:  Access engram for inspection/modification
 * HOW:  Linear search through active engrams
 *
 * @param system Engram system
 * @param engram_id Engram ID to find
 * @return Engram pointer (NULL if not found)
 */
memory_engram_t* engram_get_by_id(
    engram_system_t* system,
    uint64_t engram_id);

/**
 * @brief Get consolidation state
 *
 * @param system Engram system
 * @param engram_id Engram ID
 * @return Consolidation state
 */
engram_state_t engram_get_state(
    const engram_system_t* system,
    uint64_t engram_id);

/**
 * @brief Get consolidation strength
 *
 * @param system Engram system
 * @param engram_id Engram ID
 * @return Consolidation strength [0-1]
 */
float engram_get_consolidation_strength(
    const engram_system_t* system,
    uint64_t engram_id);

/**
 * @brief Check if engram is reconsolidating
 *
 * @param system Engram system
 * @param engram_id Engram ID
 * @return True if currently in reconsolidation window
 */
bool engram_is_reconsolidating(
    const engram_system_t* system,
    uint64_t engram_id);

/**
 * @brief Get memory age
 *
 * @param system Engram system
 * @param engram_id Engram ID
 * @param current_time_us Current time in microseconds
 * @return Age in seconds
 */
float engram_get_age_seconds(
    const engram_system_t* system,
    uint64_t engram_id,
    uint64_t current_time_us);

/**
 * @brief Get active engram count
 *
 * @param system Engram system
 * @return Number of active engrams
 */
uint32_t engram_get_active_count(const engram_system_t* system);

/**
 * @brief Get engram statistics
 *
 * WHAT: Retrieve system-wide statistics
 * WHY:  Monitor engram system performance
 * HOW:  Return counters and averages
 *
 * @param system Engram system
 * @param total_encodings_out Output: lifetime encodings
 * @param total_recalls_out Output: lifetime recalls
 * @param active_count_out Output: current active engrams
 */
void engram_get_statistics(
    const engram_system_t* system,
    uint64_t* total_encodings_out,
    uint64_t* total_recalls_out,
    uint32_t* active_count_out);

#ifdef __cplusplus
}
#endif

#endif /* NIMCP_ENGRAM_H */
