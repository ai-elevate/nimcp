/**
 * @file nimcp_oligodendrocytes.h
 * @brief Oligodendrocyte glial cell module - Axon myelination and conduction velocity optimization
 *
 * BIOLOGICAL BASIS:
 * - Oligodendrocytes produce myelin sheaths that wrap around axons
 * - Each oligodendrocyte can myelinate 10-50 axons in the CNS
 * - Myelin increases conduction velocity from 0.5-2 m/s to 50-100 m/s (10-100x faster)
 * - Adaptive myelination: high-activity axons receive more myelin
 * - Myelin remodeling occurs over hours to days in response to activity patterns
 * - Myelination is metabolically expensive (ATP cost)
 *
 * DESIGN PRINCIPLES (SOLID):
 * - Single Responsibility: ONLY handles myelination and conduction velocity
 * - Open/Closed: Extends neural network without modifying neuron code
 * - Liskov Substitution: Implements glial_cell interface
 * - Interface Segregation: Focused API for myelination functions
 * - Dependency Inversion: Depends on NIMCP utils abstractions
 *
 * INTEGRATION POINTS:
 * - nimcp_neuralnet.c: Modify signal propagation delays based on myelination
 * - nimcp_brain.c: Assign oligodendrocytes to neurons during network construction
 * - nimcp_adaptive.c: Activity tracking for adaptive myelination
 * - nimcp_introspection.c: Monitor myelination state
 *
 * PERFORMANCE CHARACTERISTICS:
 * - Remodeling: O(N) where N = number of myelinated neurons per oligodendrocyte (typically 10-50)
 * - Network step: O(M×N) where M = oligodendrocytes, N = neurons per oligo
 * - Velocity lookup: O(log N) with binary search
 * - Thread-safe with per-oligodendrocyte spinlocks
 *
 * TDD STATUS: RED phase (tests written, implementation pending)
 */

#ifndef NIMCP_OLIGODENDROCYTES_H
#define NIMCP_OLIGODENDROCYTES_H

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

/** @brief Maximum axons one oligodendrocyte can myelinate */
#define NIMCP_OLIGO_MAX_AXONS 50

/** @brief Baseline unmyelinated conduction velocity (m/s) */
#define NIMCP_OLIGO_BASE_VELOCITY_MS 1.0f

/** @brief Myelin velocity multiplier (fully myelinated) */
#define NIMCP_OLIGO_MYELIN_MULTIPLIER 50.0f

/** @brief Activity threshold for triggering myelination (Hz) */
#define NIMCP_OLIGO_ACTIVITY_THRESHOLD_HZ 1.0f

/** @brief ATP cost per unit myelination per second */
#define NIMCP_OLIGO_ATP_COST_PER_MYELIN 0.01f

/** @brief ATP regeneration rate (per second) */
#define NIMCP_OLIGO_ATP_REGEN_RATE 0.1f

/** @brief Myelination remodeling time constant (seconds)
 *
 * BIOLOGICAL NOTE: Real myelination remodeling occurs over hours to days.
 * We use a faster time constant (1s) for demonstration and testing purposes.
 * In production with real-time simulation, consider using 3600s (1 hour) or longer.
 */
#define NIMCP_OLIGO_REMODEL_TAU_S 1.0f // 1 second (accelerated for demos/tests)

/** @brief Activity history window (number of samples) */
#define NIMCP_OLIGO_ACTIVITY_WINDOW 100

// ============================================================================
// DATA STRUCTURES
// ============================================================================

/**
 * @brief Oligodendrocyte cell state
 *
 * Models a single oligodendrocyte glial cell that can myelinate multiple axons.
 * Tracks per-neuron myelination levels, activity history, and metabolic state.
 *
 * BIOLOGICAL ACCURACY:
 * - Each oligodendrocyte myelinates 10-50 axons (CNS)
 * - Myelination levels: 0 (unmyelinated) to 1 (fully myelinated)
 * - Adaptive myelination: high-activity axons prioritized
 * - Metabolic cost: producing myelin requires ATP
 * - Remodeling: myelination changes over hours to days
 */
typedef struct {
    uint32_t id;                         /**< Unique oligodendrocyte ID */

    // Myelinated neurons (HOT PATH)
    uint32_t num_myelinated_neurons;     /**< Number of neurons this oligo covers */
    uint32_t max_neurons;                /**< Capacity (typically 10-50) */
    uint32_t* myelinated_neuron_ids;     /**< Array of neuron IDs [max_neurons] */
    float* myelination_levels;           /**< Per-neuron myelination 0-1 [max_neurons] */

    // Metabolic state
    float atp_level;                     /**< Energy available (0-1) */
    float metabolic_cost;                /**< Current ATP consumption rate */
    float max_myelination_capacity;      /**< Total myelin this oligo can produce */

    // Activity tracking for adaptive myelination
    float* neuron_activity_history;      /**< Rolling average activity [max_neurons] */
    uint64_t* last_spike_times;          /**< Last spike timestamp [max_neurons] */

    // Remodeling state
    uint64_t last_remodeling_time;       /**< When last adjusted myelination (µs) */
    float remodeling_interval_ms;        /**< How often to remodel (typically hours) */

    // Spatial location
    float position[3];                   /**< x, y, z coordinates (µm) */

    // Thread safety
    nimcp_spinlock_t lock;               /**< Lock for concurrent access */
} oligodendrocyte_t;

/**
 * @brief Network of oligodendrocytes
 *
 * Container for managing multiple oligodendrocytes and coordinating
 * network-wide myelination updates.
 */
typedef struct {
    uint32_t num_oligodendrocytes;       /**< Current number of oligodendrocytes */
    uint32_t capacity;                   /**< Max oligodendrocytes */
    oligodendrocyte_t** oligodendrocytes; /**< Array of oligodendrocyte pointers */

    // Global parameters
    float base_conduction_velocity;      /**< Unmyelinated velocity (m/s) */
    float myelinated_velocity_multiplier; /**< Myelin boost factor (10-100x) */
    float activity_threshold;            /**< Activity level to trigger myelination (Hz) */

    // Thread safety
    nimcp_mutex_t lock;                  /**< Network-level lock */
} oligodendrocyte_network_t;

// ============================================================================
// CREATION & DESTRUCTION
// ============================================================================

/**
 * @brief Create a new oligodendrocyte
 *
 * @param id Unique identifier for this oligodendrocyte
 * @param max_axons Maximum number of axons this oligo can myelinate (typically 10-50)
 *
 * @return Pointer to oligodendrocyte or NULL on failure
 *
 * INITIAL STATE:
 * - No neurons assigned (num_myelinated_neurons = 0)
 * - ATP level = 1.0 (fully energized)
 * - All myelination levels = 0.0 (unmyelinated)
 * - Activity history = 0.0 (no activity tracked yet)
 */
oligodendrocyte_t* oligodendrocyte_create(uint32_t id, uint32_t max_axons);

/**
 * @brief Destroy oligodendrocyte and free resources
 *
 * @param oligo Oligodendrocyte to destroy (NULL safe)
 */
void oligodendrocyte_destroy(oligodendrocyte_t* oligo);

// ============================================================================
// NEURON ASSIGNMENT & MYELINATION
// ============================================================================

/**
 * @brief Assign a neuron to this oligodendrocyte for myelination
 *
 * @param oligo Oligodendrocyte
 * @param neuron_id ID of neuron/axon to myelinate
 *
 * @return NIMCP_SUCCESS on success, error code otherwise
 *
 * BEHAVIOR:
 * - Adds neuron to myelinated_neuron_ids array
 * - Initializes myelination_level to 0.0 (will increase with activity)
 * - Fails if at capacity (num_myelinated_neurons >= max_neurons)
 * - Duplicate assignments update existing entry
 */
nimcp_result_t oligodendrocyte_assign_neuron(oligodendrocyte_t* oligo, uint32_t neuron_id);

/**
 * @brief Get current myelination level for a neuron
 *
 * @param oligo Oligodendrocyte
 * @param neuron_id ID of neuron to query
 *
 * @return Myelination level 0-1, or 0.0 if neuron not found
 *
 * INTERPRETATION:
 * - 0.0 = Unmyelinated
 * - 0.5 = Partially myelinated
 * - 1.0 = Fully myelinated
 */
float oligodendrocyte_get_myelination_level(oligodendrocyte_t* oligo, uint32_t neuron_id);

/**
 * @brief Compute effective conduction velocity for a neuron
 *
 * @param oligo Oligodendrocyte
 * @param neuron_id ID of neuron
 * @param base_velocity Unmyelinated conduction velocity (m/s)
 *
 * @return Effective velocity with myelin boost (m/s)
 *
 * FORMULA:
 * velocity = base_velocity × (1 + myelin_level × (multiplier - 1))
 *
 * EXAMPLES:
 * - Unmyelinated (myelin=0): velocity = base_velocity
 * - Fully myelinated (myelin=1, multiplier=50): velocity = base_velocity × 50
 */
float oligodendrocyte_compute_conduction_velocity(oligodendrocyte_t* oligo,
                                                   uint32_t neuron_id,
                                                   float base_velocity);

// ============================================================================
// ACTIVITY TRACKING & ADAPTIVE MYELINATION
// ============================================================================

/**
 * @brief Track activity for a neuron
 *
 * @param oligo Oligodendrocyte
 * @param neuron_id ID of active neuron
 * @param activity Activity level (e.g., spike count, firing rate)
 * @param timestamp Timestamp of this activity sample (µs)
 *
 * BEHAVIOR:
 * - Updates rolling average of neuron activity
 * - Used by remodeling to determine myelination adjustments
 * - Higher activity → prioritized for myelination
 * - Thread-safe (uses spinlock)
 */
void oligodendrocyte_track_activity(oligodendrocyte_t* oligo,
                                    uint32_t neuron_id,
                                    float activity,
                                    uint64_t timestamp);

/**
 * @brief Remodel myelination based on activity history
 *
 * @param oligo Oligodendrocyte
 * @param dt Time step (seconds)
 *
 * ADAPTIVE MYELINATION ALGORITHM:
 * 1. Compute target myelination for each neuron based on activity
 * 2. High-activity neurons → increase myelination (up to 1.0)
 * 3. Low-activity neurons → decrease myelination (down to 0.0)
 * 4. Rate-limited by time constant (hours) and ATP availability
 * 5. Total myelination constrained by max_myelination_capacity
 *
 * BIOLOGICAL ACCURACY:
 * - Myelination changes slowly (hours to days)
 * - Resource allocation: limited total myelin per oligodendrocyte
 * - Metabolic cost: ATP required for myelin production
 *
 * PERFORMANCE:
 * - O(N) where N = num_myelinated_neurons (typically 10-50)
 * - Thread-safe with spinlock
 */
void oligodendrocyte_remodel_myelination(oligodendrocyte_t* oligo, float dt);

// ============================================================================
// METABOLIC MANAGEMENT
// ============================================================================

/**
 * @brief Update ATP level based on metabolic cost
 *
 * @param oligo Oligodendrocyte
 * @param dt Time step (seconds)
 *
 * DYNAMICS:
 * - ATP depleted by myelination cost
 * - ATP regenerated over time
 * - Low ATP limits myelination remodeling rate
 *
 * FORMULA:
 * cost = sum(myelination_levels) × ATP_COST_PER_MYELIN
 * atp_level += dt × (ATP_REGEN_RATE - cost)
 * atp_level = clamp(atp_level, 0.0, 1.0)
 */
void oligodendrocyte_update_atp(oligodendrocyte_t* oligo, float dt);

// ============================================================================
// NETWORK MANAGEMENT
// ============================================================================

/**
 * @brief Create a network of oligodendrocytes
 *
 * @param capacity Maximum number of oligodendrocytes
 *
 * @return Pointer to network or NULL on failure
 */
oligodendrocyte_network_t* oligodendrocyte_network_create(uint32_t capacity);

/**
 * @brief Destroy oligodendrocyte network
 *
 * @param network Network to destroy (NULL safe)
 *
 * NOTE: Also destroys all contained oligodendrocytes
 */
void oligodendrocyte_network_destroy(oligodendrocyte_network_t* network);

/**
 * @brief Add an oligodendrocyte to the network
 *
 * @param network Network
 * @param oligo Oligodendrocyte to add
 *
 * @return NIMCP_SUCCESS on success, error code otherwise
 *
 * NOTE: Network takes ownership of the oligodendrocyte
 */
nimcp_result_t oligodendrocyte_network_add(oligodendrocyte_network_t* network,
                                            oligodendrocyte_t* oligo);

/**
 * @brief Update all oligodendrocytes in network
 *
 * @param network Network
 * @param dt Time step (seconds)
 *
 * OPERATIONS:
 * - Remodel myelination for all oligodendrocytes
 * - Update ATP levels
 * - Thread-safe: each oligodendrocyte uses its own spinlock
 *
 * PERFORMANCE:
 * - O(M×N) where M = num_oligodendrocytes, N = neurons per oligo
 * - Can be parallelized (each oligo is independent)
 */
void oligodendrocyte_network_step(oligodendrocyte_network_t* network, float dt);

/**
 * @brief Find oligodendrocyte responsible for a neuron
 *
 * @param network Network
 * @param neuron_id ID of neuron to query
 *
 * @return Pointer to oligodendrocyte or NULL if not found
 *
 * PERFORMANCE: O(M×N) linear search (can be optimized with hash table)
 */
oligodendrocyte_t* oligodendrocyte_network_find_by_neuron(oligodendrocyte_network_t* network,
                                                           uint32_t neuron_id);

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * @brief Get total myelination for this oligodendrocyte
 *
 * @param oligo Oligodendrocyte
 *
 * @return Sum of myelination levels across all neurons
 *
 * USE CASE: Monitor metabolic load, resource allocation
 */
float oligodendrocyte_get_total_myelination(oligodendrocyte_t* oligo);

#ifdef __cplusplus
}
#endif

#endif // NIMCP_OLIGODENDROCYTES_H
