//=============================================================================
// nimcp_brain.h - Unified Brain API
//=============================================================================
/**
 * @file nimcp_brain.h
 * @brief Unified interface coordinating all NIMCP subsystems
 *
 * WHAT: High-level brain API that integrates all cognitive components
 * WHY:  Simplify usage and ensure correct interaction between subsystems
 * HOW:  Single brain_update() call coordinates:
 *       - Spike encoding (NLP → spikes)
 *       - Network dynamics (spike propagation)
 *       - Plasticity (STDP, eligibility traces)
 *       - Neuromodulation (dopamine, etc.)
 *       - Output decoding (spikes → predictions)
 *
 * ARCHITECTURE:
 * ```
 * Brain {
 *   Network (fractal topology)
 *   STDP Learner (spike-timing plasticity)
 *   Eligibility Traces (temporal credit assignment)
 *   Neuromodulation (dopamine, serotonin, ACh, NE)
 *   Input/Output Mappings
 * }
 *
 * brain_update(input, reward) {
 *   1. Encode input → spikes
 *   2. Propagate spikes through network
 *   3. Apply STDP on spike pairs
 *   4. Update neuromodulators
 *   5. Update eligibility traces with reward
 *   6. Decode output spikes
 * }
 * ```
 *
 * DESIGN GOALS:
 * - Simplicity: One function call per timestep
 * - Correctness: Automatic coordination prevents integration errors
 * - Performance: Minimize overhead, optimize critical paths
 * - Flexibility: Enable/disable subsystems via config
 *
 * @author NIMCP Development Team
 * @date 2025-11-08
 * @version 2.7.0 Phase 5
 */

#ifndef NIMCP_BRAIN_H
#define NIMCP_BRAIN_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// Forward Declarations
//=============================================================================

typedef struct brain_struct* brain_t;

//=============================================================================
// Configuration
//=============================================================================

/**
 * @brief Brain configuration
 */
typedef struct {
    // Network topology
    uint32_t num_neurons;           /**< Total neurons in network */
    float scale_free_exponent;      /**< Power-law exponent for scale-free (-2.1 typical) */

    // I/O dimensions
    uint32_t input_dim;             /**< Input embedding dimension */
    uint32_t output_dim;            /**< Output prediction dimension */

    // Subsystem enable flags
    bool enable_stdp;               /**< Enable spike-timing plasticity */
    bool enable_eligibility_traces; /**< Enable temporal credit assignment */
    bool enable_neuromodulation;    /**< Enable dopamine/serotonin/etc. */

    // Learning rates
    float stdp_learning_rate;       /**< STDP learning rate multiplier */
    float eligibility_learning_rate;/**< Eligibility trace learning rate */
} brain_config_t;

/**
 * @brief Brain statistics
 */
typedef struct {
    uint64_t total_spikes;          /**< Total spikes generated */
    uint64_t stdp_ltp_events;       /**< LTP events (potentiation) */
    uint64_t stdp_ltd_events;       /**< LTD events (depression) */
    float avg_weight_change;        /**< Average synaptic weight change */
    float dopamine_level;           /**< Current dopamine level */
    float serotonin_level;          /**< Current serotonin level */
    float acetylcholine_level;      /**< Current acetylcholine level */
    float norepinephrine_level;     /**< Current norepinephrine level */
} brain_stats_t;

//=============================================================================
// Lifecycle
//=============================================================================

/**
 * @brief Get default brain configuration
 *
 * WHAT: Return sensible defaults for brain parameters
 * WHY:  Provide starting point for experimentation
 * HOW:  Static initialization with empirically chosen values
 *
 * DEFAULTS:
 * - 500 neurons (scale-free topology)
 * - Input/output: 50 dimensions each
 * - All subsystems enabled
 * - Learning rates: STDP=1.0, Eligibility=0.001
 *
 * @return Default configuration
 */
brain_config_t brain_default_config(void);

/**
 * @brief Create integrated brain system
 *
 * WHAT: Allocate and initialize all subsystems
 * WHY:  One-stop initialization for complete cognitive system
 * HOW:  Create network, STDP, eligibility traces, neuromodulation
 *
 * SUBSYSTEMS CREATED:
 * 1. Fractal neural network (scale-free topology)
 * 2. STDP learner (if enabled)
 * 3. Eligibility trace system (if enabled)
 * 4. Pink noise neuromodulation (if enabled)
 *
 * ERROR HANDLING:
 * - Returns NULL if config is NULL
 * - Returns NULL if any subsystem fails to initialize
 * - Cleanup: brain_destroy() called on partial initialization
 *
 * @param config Brain configuration (NULL = use defaults)
 * @return Brain handle or NULL on error
 */
brain_t brain_create(const brain_config_t* config);

/**
 * @brief Destroy brain and free all resources
 *
 * WHAT: Clean up all subsystems
 * WHY:  Prevent memory leaks
 * HOW:  Destroy each subsystem in reverse order of creation
 *
 * CLEANUP ORDER:
 * 1. Neuromodulation system
 * 2. Eligibility traces
 * 3. STDP learner
 * 4. Neural network
 * 5. Brain structure
 *
 * THREAD_SAFETY: Not thread-safe, must not be called concurrently
 *
 * @param brain Brain to destroy (NULL-safe)
 */
void brain_destroy(brain_t brain);

//=============================================================================
// Unified Update Loop
//=============================================================================

/**
 * @brief Update brain for single timestep
 *
 * WHAT: Coordinate all subsystems for one simulation step
 * WHY:  Ensure correct interaction between learning mechanisms
 * HOW:  Sequential pipeline through all subsystems
 *
 * ALGORITHM:
 * ```
 * 1. Encode input embedding → spikes (spike NLP)
 * 2. Propagate spikes through network (network dynamics)
 * 3. Apply STDP to active synapses (spike-timing plasticity)
 * 4. Update neuromodulators (dopamine response to reward)
 * 5. Apply eligibility traces (temporal credit assignment)
 * 6. Decode output spikes → predictions (spike NLP)
 * ```
 *
 * LEARNING:
 * - STDP: Applied automatically on all spike pairs
 * - Eligibility: Traces updated, consolidated with reward
 * - Neuromodulation: Dopamine gates learning strength
 *
 * PERFORMANCE: O(N×S) where N=neurons, S=avg synapses/neuron
 *
 * @param brain Brain system
 * @param input_embedding Input pattern (NULL = no input)
 * @param input_dim Dimension of input (must match config)
 * @param reward Reward signal (0.0 = no reward, +1.0 = reward, -1.0 = punishment)
 * @param timestamp Current simulation time (ms)
 * @param output_prediction Output buffer (NULL = no output) [output_dim]
 * @return Number of spikes generated
 */
uint32_t brain_update(
    brain_t brain,
    const float* input_embedding,
    uint32_t input_dim,
    float reward,
    uint64_t timestamp,
    float* output_prediction
);

//=============================================================================
// Statistics and Monitoring
//=============================================================================

/**
 * @brief Get brain statistics
 *
 * WHAT: Retrieve current stats from all subsystems
 * WHY:  Monitor learning progress and system health
 * HOW:  Query each subsystem and aggregate results
 *
 * @param brain Brain system
 * @param stats Output statistics structure
 * @return true on success, false on error
 */
bool brain_get_stats(brain_t brain, brain_stats_t* stats);

/**
 * @brief Reset brain statistics
 *
 * WHAT: Clear accumulated statistics
 * WHY:  Start new measurement period
 * HOW:  Reset counters in all subsystems
 *
 * @param brain Brain system
 */
void brain_reset_stats(brain_t brain);

/**
 * @brief Get neuromodulator levels
 *
 * WHAT: Retrieve current neuromodulator concentrations
 * WHY:  Monitor arousal, attention, learning state
 * HOW:  Direct query to neuromodulation subsystem
 *
 * @param brain Brain system
 * @param dopamine Output dopamine level (can be NULL)
 * @param serotonin Output serotonin level (can be NULL)
 * @param acetylcholine Output acetylcholine level (can be NULL)
 * @param norepinephrine Output norepinephrine level (can be NULL)
 * @return true on success
 */
bool brain_get_neuromodulators(
    brain_t brain,
    float* dopamine,
    float* serotonin,
    float* acetylcholine,
    float* norepinephrine
);

#ifdef __cplusplus
}
#endif

#endif // NIMCP_BRAIN_H
