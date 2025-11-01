/**
 * @file nimcp_microglia.h
 * @brief Microglia glial cell module - Synaptic surveillance and pruning
 *
 * BIOLOGICAL BASIS:
 * - Microglia are immune cells that continuously survey the brain
 * - Prune weak/inactive synapses during development and learning
 * - Activity-dependent refinement: preserve active, remove inactive
 * - Critical for circuit optimization and plasticity
 * - Surveillance: monitor synapses within radius (~50-100 µm)
 * - Pruning: remove synapses below activity threshold
 *
 * DESIGN PRINCIPLES (SOLID):
 * - Single Responsibility: ONLY handles synaptic surveillance & pruning
 * - Open/Closed: Extends neural network without modifying synapse code
 * - Liskov Substitution: Implements glial_cell interface
 * - Interface Segregation: Focused API for surveillance/pruning
 * - Dependency Inversion: Depends on NIMCP utils abstractions
 *
 * INTEGRATION POINTS:
 * - nimcp_neuralnet.c: Synapse removal/pruning
 * - nimcp_brain.c: Assign microglia to spatial regions
 * - nimcp_adaptive.c: Activity monitoring
 * - nimcp_introspection.c: Monitor pruning statistics
 *
 * PERFORMANCE CHARACTERISTICS:
 * - Activity score update: O(N) where N = monitored synapses
 * - Weak synapse identification: O(N) linear scan
 * - Pruning: O(W) where W = weak synapses (typically small)
 * - Thread-safe with per-microglia spinlocks
 *
 * TDD STATUS: RED phase (tests written, implementation pending)
 */

#ifndef NIMCP_MICROGLIA_H
#define NIMCP_MICROGLIA_H

#include "utils/nimcp_common.h"
#include "utils/nimcp_memory.h"
#include "utils/nimcp_thread.h"
#include "utils/nimcp_time.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// CONSTANTS & BIOLOGICAL PARAMETERS
// ============================================================================

/** @brief Default surveillance radius (µm) */
#define NIMCP_MICROGLIA_SURVEILLANCE_RADIUS_UM 100.0f

/** @brief Default pruning threshold (activity score) */
#define NIMCP_MICROGLIA_PRUNING_THRESHOLD 0.1f

/** @brief Default pruning rate (max synapses per step) */
#define NIMCP_MICROGLIA_PRUNING_RATE 5.0f

/** @brief Activity score decay time constant (seconds) */
#define NIMCP_MICROGLIA_ACTIVITY_DECAY_TAU_S 10.0f

/** @brief Minimum activity window for assessment (milliseconds) */
#define NIMCP_MICROGLIA_MIN_ACTIVITY_WINDOW_MS 1000.0f

// ============================================================================
// DATA STRUCTURES
// ============================================================================

/**
 * @brief Microglia cell state
 *
 * Models a single microglia glial cell that surveys and prunes synapses.
 * Tracks per-synapse activity scores and prunes weak/inactive connections.
 *
 * BIOLOGICAL ACCURACY:
 * - Surveillance radius: 50-100 µm (typical microglial territory)
 * - Continuous monitoring of synaptic activity
 * - Prunes synapses with activity below threshold
 * - Activity-dependent refinement during development/learning
 */
typedef struct {
    uint32_t id;                         /**< Unique microglia ID */

    // Spatial location
    float position[3];                   /**< x, y, z coordinates (µm) */
    float surveillance_radius;           /**< Monitoring radius (µm) */

    // Monitored synapses (HOT PATH)
    uint32_t num_monitored_synapses;     /**< Number of synapses monitored */
    uint32_t max_monitored_synapses;     /**< Capacity */
    uint32_t* monitored_synapse_ids;     /**< Array of synapse IDs */
    float* synapse_activity_scores;      /**< Activity level per synapse (0-1+) */
    uint64_t* last_activity_times;       /**< Last active timestamp (µs) */

    // Pruning policy
    float pruning_threshold;             /**< Activity below this → candidate for pruning */
    float pruning_rate;                  /**< Max synapses to prune per timestep */
    uint64_t last_pruning_time;          /**< When last pruned (µs) */

    // Statistics
    uint32_t total_synapses_pruned;      /**< Cumulative pruned count */

    // Thread safety
    nimcp_spinlock_t lock;               /**< Lock for concurrent access */
} microglia_t;

/**
 * @brief Network of microglia cells
 *
 * Container for managing multiple microglia and coordinating
 * network-wide surveillance and pruning operations.
 */
typedef struct {
    uint32_t num_microglia;              /**< Current number of microglia */
    uint32_t capacity;                   /**< Max microglia */
    microglia_t** microglia;             /**< Array of microglia pointers */

    // Global parameters
    float global_pruning_threshold;      /**< Default pruning threshold */
    float min_activity_window_ms;        /**< Min time window for activity assessment */

    // Thread safety
    nimcp_mutex_t lock;                  /**< Network-level lock */
} microglia_network_t;

// ============================================================================
// CREATION & DESTRUCTION
// ============================================================================

/**
 * @brief Create a new microglia cell
 *
 * @param id Unique identifier for this microglia
 * @param x X coordinate (µm)
 * @param y Y coordinate (µm)
 * @param z Z coordinate (µm)
 * @param surveillance_radius Monitoring radius (µm)
 *
 * @return Pointer to microglia or NULL on failure
 *
 * INITIAL STATE:
 * - No synapses monitored (num_monitored_synapses = 0)
 * - Pruning threshold = NIMCP_MICROGLIA_PRUNING_THRESHOLD
 * - Pruning rate = NIMCP_MICROGLIA_PRUNING_RATE
 * - Total pruned = 0
 */
microglia_t* microglia_create(uint32_t id, float x, float y, float z, float surveillance_radius);

/**
 * @brief Destroy microglia and free resources
 *
 * @param mg Microglia to destroy (NULL safe)
 */
void microglia_destroy(microglia_t* mg);

// ============================================================================
// SYNAPSE MONITORING
// ============================================================================

/**
 * @brief Add a synapse to this microglia's surveillance list
 *
 * @param mg Microglia
 * @param synapse_id ID of synapse to monitor
 *
 * @return NIMCP_SUCCESS on success, error code otherwise
 *
 * BEHAVIOR:
 * - Adds synapse to monitored_synapse_ids array
 * - Initializes activity score to 0.0
 * - Fails if at capacity
 * - Duplicate monitoring is idempotent (updates existing entry)
 */
nimcp_result_t microglia_monitor_synapse(microglia_t* mg, uint32_t synapse_id);

/**
 * @brief Track activity for a monitored synapse
 *
 * @param mg Microglia
 * @param synapse_id ID of synapse
 * @param activity Activity level (e.g., 1.0 for spike, 0.0 for silent)
 * @param timestamp Current timestamp (µs)
 *
 * BEHAVIOR:
 * - Updates activity score using exponential moving average
 * - Updates last_activity_times timestamp
 * - Ignores unknown synapses (not monitored)
 * - Thread-safe with spinlock
 */
void microglia_track_synapse_activity(microglia_t* mg,
                                      uint32_t synapse_id,
                                      float activity,
                                      uint64_t timestamp);

/**
 * @brief Update activity scores for all monitored synapses
 *
 * @param mg Microglia
 * @param current_time Current timestamp (µs)
 *
 * BEHAVIOR:
 * - Decays activity scores over time (inactive synapses → 0)
 * - Decay formula: score *= exp(-dt / tau)
 * - Call periodically to maintain accurate scores
 */
void microglia_update_activity_scores(microglia_t* mg, uint64_t current_time);

/**
 * @brief Get activity score for a synapse
 *
 * @param mg Microglia
 * @param synapse_id ID of synapse to query
 *
 * @return Activity score (0-1+), or 0.0 if synapse not monitored
 *
 * INTERPRETATION:
 * - 0.0 = Completely inactive
 * - 0.5 = Moderate activity
 * - 1.0+ = High activity
 */
float microglia_get_synapse_activity_score(microglia_t* mg, uint32_t synapse_id);

// ============================================================================
// WEAK SYNAPSE IDENTIFICATION & PRUNING
// ============================================================================

/**
 * @brief Identify weak synapses candidates for pruning
 *
 * @param mg Microglia
 * @param weak_synapse_ids Output buffer for weak synapse IDs
 * @param max_count Maximum synapses to identify (buffer size)
 *
 * @return Number of weak synapses identified
 *
 * ALGORITHM:
 * - Scan all monitored synapses
 * - If activity_score < pruning_threshold → mark as weak
 * - Return up to max_count weak synapses
 * - Sort by activity score (weakest first) if possible
 *
 * PERFORMANCE: O(N) where N = num_monitored_synapses
 */
uint32_t microglia_identify_weak_synapses(microglia_t* mg,
                                          uint32_t* weak_synapse_ids,
                                          uint32_t max_count);

/**
 * @brief Prune weak synapses from monitored list
 *
 * @param mg Microglia
 *
 * @return Number of synapses pruned
 *
 * BEHAVIOR:
 * - Identifies weak synapses (activity < threshold)
 * - Removes up to pruning_rate synapses per call
 * - Removes from monitored_synapse_ids list
 * - Updates num_monitored_synapses
 * - Increments total_synapses_pruned counter
 * - Thread-safe with spinlock
 *
 * NOTE: This removes from MONITORING list, not the actual neural network.
 * For actual synapse deletion, call neural network pruning functions.
 *
 * PERFORMANCE: O(W) where W = number of weak synapses (typically small)
 */
uint32_t microglia_prune_weak_synapses(microglia_t* mg);

// ============================================================================
// NETWORK MANAGEMENT
// ============================================================================

/**
 * @brief Create a network of microglia
 *
 * @param capacity Maximum number of microglia
 *
 * @return Pointer to network or NULL on failure
 */
microglia_network_t* microglia_network_create(uint32_t capacity);

/**
 * @brief Destroy microglia network
 *
 * @param network Network to destroy (NULL safe)
 *
 * NOTE: Also destroys all contained microglia
 */
void microglia_network_destroy(microglia_network_t* network);

/**
 * @brief Add a microglia to the network
 *
 * @param network Network
 * @param mg Microglia to add
 *
 * @return NIMCP_SUCCESS on success, error code otherwise
 *
 * NOTE: Network takes ownership of the microglia
 */
nimcp_result_t microglia_network_add(microglia_network_t* network, microglia_t* mg);

/**
 * @brief Update all microglia in network (surveillance & pruning step)
 *
 * @param network Network
 * @param current_time Current timestamp (µs)
 *
 * OPERATIONS:
 * - Update activity scores for all microglia
 * - Prune weak synapses across all microglia
 * - Thread-safe: each microglia uses its own spinlock
 *
 * PERFORMANCE:
 * - O(M×N) where M = num_microglia, N = synapses per microglia
 * - Can be parallelized (each microglia is independent)
 */
void microglia_network_step(microglia_network_t* network, uint64_t current_time);

/**
 * @brief Find microglia monitoring a specific synapse
 *
 * @param network Network
 * @param synapse_id ID of synapse to query
 *
 * @return Pointer to microglia or NULL if not found
 *
 * PERFORMANCE: O(M×N) linear search (can be optimized with spatial indexing)
 */
microglia_t* microglia_network_find_by_synapse(microglia_network_t* network,
                                                uint32_t synapse_id);

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * @brief Get total number of synapses pruned by this microglia
 *
 * @param mg Microglia
 *
 * @return Total pruned count
 *
 * USE CASE: Monitoring pruning statistics, debugging
 */
uint32_t microglia_get_total_pruned(microglia_t* mg);

#ifdef __cplusplus
}
#endif

#endif // NIMCP_MICROGLIA_H
