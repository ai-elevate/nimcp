//=============================================================================
// nimcp_stdp.h - Spike-Timing-Dependent Plasticity
//=============================================================================
/**
 * @file nimcp_stdp.h
 * @brief Spike-timing-dependent plasticity for temporal learning
 *
 * WHAT: Biological learning rule based on spike timing causality
 * WHY:
 *   - Captures temporal relationships in data
 *   - Biologically plausible unsupervised learning
 *   - Enables sequence learning and temporal credit assignment
 *
 * HOW:
 *   Δw = A+ × exp(-Δt / τ+)   if pre before post (LTP)
 *   Δw = A- × exp(Δt / τ-)    if post before pre (LTD)
 *
 * BIOLOGICAL BASIS:
 *   - Bi & Poo (1998): Discovered in hippocampal neurons
 *   - LTP (Long-Term Potentiation): Strengthens causal connections
 *   - LTD (Long-Term Depression): Weakens anti-causal connections
 *   - Asymmetric time window: ±20-50ms typical
 *
 * USE CASES:
 *   - Sequence learning: Learn temporal patterns
 *   - Sensory processing: Extract temporal features
 *   - Reinforcement learning: Credit assignment (with eligibility traces)
 *   - Pattern completion: Predict next spike from history
 *
 * @author NIMCP Development Team
 * @date 2025-11-08
 * @version 2.7.0 Phase 4
 */

#ifndef NIMCP_STDP_H
#define NIMCP_STDP_H

#include <stdint.h>
#include <stdbool.h>
#include "core/neuralnet/nimcp_neuralnet.h"

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// STDP Configuration
//=============================================================================

/**
 * @brief STDP learning rule configuration
 *
 * WHAT: Parameters for exponential STDP window
 * WHY: Control plasticity timescales and magnitudes
 * HOW: Time constants and amplitude factors
 */
typedef struct {
    // Time constants
    float tau_plus;      /**< LTP time constant in ms (default: 20.0) */
    float tau_minus;     /**< LTD time constant in ms (default: 20.0) */

    // Amplitude factors
    float A_plus;        /**< LTP amplitude (default: 0.01) */
    float A_minus;       /**< LTD amplitude (default: -0.012, asymmetric) */

    // Weight bounds
    float w_min;         /**< Minimum synaptic weight (default: 0.0) */
    float w_max;         /**< Maximum synaptic weight (default: 10.0) */

    // Learning rate modulation
    float learning_rate; /**< Global learning rate multiplier (default: 1.0) */

    // Nearest-spike vs all-pairs
    bool nearest_spike;  /**< Use only nearest spike pair (default: true) */

} stdp_config_t;

/**
 * @brief STDP learner state
 *
 * WHAT: Runtime state for STDP learning
 * WHY: Track configuration and statistics
 * HOW: Store config, update counters, exponential lookup tables
 */
typedef struct {
    // Configuration
    stdp_config_t config;

    // Statistics
    uint64_t updates_applied;  /**< Total STDP updates */
    uint64_t ltp_count;        /**< LTP events */
    uint64_t ltd_count;        /**< LTD events */
    float avg_weight_change;   /**< Exponential moving average of |Δw| */

    // Optimization: Exponential lookup tables (optional)
    float* ltp_lookup;         /**< Pre-computed exp(-t/τ+) for t=0..200ms */
    float* ltd_lookup;         /**< Pre-computed exp(t/τ-) for t=0..200ms */
    uint32_t lookup_size;      /**< Size of lookup tables */

} stdp_learner_t;

//=============================================================================
// Lifecycle Functions
//=============================================================================

/**
 * @brief Create STDP learner with configuration
 *
 * WHAT: Initialize STDP system with time constants and amplitudes
 * WHY: Set up learning parameters for network
 * HOW: Allocate structure, copy config, create lookup tables
 *
 * @param config STDP configuration (or NULL for defaults)
 * @return STDP learner handle, or NULL on error
 */
stdp_learner_t* stdp_create(const stdp_config_t* config);

/**
 * @brief Destroy STDP learner
 *
 * WHAT: Free all resources associated with STDP learner
 * WHY: Prevent memory leaks
 * HOW: Free lookup tables, then main structure
 *
 * @param learner STDP learner to destroy (can be NULL)
 */
void stdp_destroy(stdp_learner_t* learner);

/**
 * @brief Get default STDP configuration
 *
 * WHAT: Return sensible default STDP parameters
 * WHY: Provide biologically plausible starting point
 * HOW: Static initialization with literature values
 *
 * DEFAULTS (based on Bi & Poo 1998, Song et al. 2000):
 * - τ+ = τ- = 20ms (symmetric time constants)
 * - A+ = 0.01, A- = -0.012 (slightly asymmetric for stability)
 * - w ∈ [0, 10] (non-negative weights)
 * - learning_rate = 1.0 (no scaling)
 * - nearest_spike = true (computational efficiency)
 *
 * @return Default configuration
 */
stdp_config_t stdp_default_config(void);

//=============================================================================
// STDP Update Functions
//=============================================================================

/**
 * @brief Apply STDP to synapse given spike pair
 *
 * WHAT: Update synapse weight based on pre/post spike timing
 * WHY: Implement spike-timing-dependent plasticity
 * HOW: Compute Δt, apply exponential STDP rule, clamp weight
 *
 * ALGORITHM:
 * ```
 * Δt = t_post - t_pre
 * if (Δt > 0):  // pre before post (causal)
 *     Δw = A+ × exp(-Δt / τ+)  // LTP
 * else:         // post before pre (anti-causal)
 *     Δw = A- × exp(|Δt| / τ-) // LTD
 * w_new = clamp(w_old + Δw, w_min, w_max)
 * ```
 *
 * PERFORMANCE: O(1) - single exponential computation (or lookup)
 *
 * @param learner STDP learner with configuration
 * @param synapse Synapse to update
 * @param t_pre Presynaptic spike time (ms)
 * @param t_post Postsynaptic spike time (ms)
 * @return Weight change (Δw), or 0.0 if no update
 */
float stdp_apply(
    stdp_learner_t* learner,
    synapse_t* synapse,
    uint64_t t_pre,
    uint64_t t_post
);

/**
 * @brief Apply STDP using spike histories
 *
 * WHAT: Update synapse based on recent spike history
 * WHY: Handle multiple spike pairs (all-pairs STDP)
 * HOW: Iterate through spike history, apply STDP for each pair
 *
 * MODES:
 * - nearest_spike=true: Use only most recent spike pair (fast)
 * - nearest_spike=false: Use all spike pairs within window (accurate)
 *
 * PERFORMANCE:
 * - Nearest spike: O(1)
 * - All pairs: O(H²) where H = history length
 *
 * @param learner STDP learner
 * @param synapse Synapse to update
 * @param pre_neuron Presynaptic neuron (for spike history)
 * @param post_neuron Postsynaptic neuron (for spike history)
 * @return Total weight change
 */
float stdp_apply_from_history(
    stdp_learner_t* learner,
    synapse_t* synapse,
    const neuron_t* pre_neuron,
    const neuron_t* post_neuron
);

/**
 * @brief Apply STDP to all synapses in network
 *
 * WHAT: Batch STDP update for entire network
 * WHY: Update all synapses after network step
 * HOW: Iterate through all neurons, update incoming synapses
 *
 * TYPICAL USAGE:
 * ```
 * // After each network timestep:
 * neural_network_compute_step(network, t);
 * stdp_apply_to_network(learner, network);
 * ```
 *
 * PERFORMANCE: O(S) where S = total synapses in network
 *
 * @param learner STDP learner
 * @param network Neural network to update
 */
void stdp_apply_to_network(
    stdp_learner_t* learner,
    void* network  // neural_network_t (opaque)
);

//=============================================================================
// Modulated STDP (Integration with Neuromodulation)
//=============================================================================

/**
 * @brief Apply dopamine-modulated STDP
 *
 * WHAT: Scale STDP learning by dopamine level
 * WHY: Dopamine gates plasticity (reward-modulated learning)
 * HOW: Multiply weight change by dopamine concentration
 *
 * BIOLOGICAL BASIS:
 * - Dopamine D1 receptors enhance LTP
 * - Low dopamine → weak/no plasticity (Schultz et al., 1997)
 *
 * ALGORITHM:
 * ```
 * Δw_base = stdp_apply(...)
 * Δw_modulated = Δw_base × dopamine_level
 * ```
 *
 * @param learner STDP learner
 * @param synapse Synapse to update
 * @param t_pre Presynaptic spike time
 * @param t_post Postsynaptic spike time
 * @param dopamine_level Dopamine concentration [0, 1]
 * @return Modulated weight change
 */
float stdp_apply_modulated(
    stdp_learner_t* learner,
    synapse_t* synapse,
    uint64_t t_pre,
    uint64_t t_post,
    float dopamine_level
);

//=============================================================================
// Statistics and Analysis
//=============================================================================

/**
 * @brief Get STDP learning statistics
 *
 * WHAT: Query learning activity (LTP/LTD counts, avg weight change)
 * WHY: Monitor learning dynamics
 * HOW: Return statistics structure
 *
 * @param learner STDP learner
 * @param ltp_count Output: Number of LTP events (can be NULL)
 * @param ltd_count Output: Number of LTD events (can be NULL)
 * @param avg_weight_change Output: Average |Δw| (can be NULL)
 */
void stdp_get_statistics(
    const stdp_learner_t* learner,
    uint64_t* ltp_count,
    uint64_t* ltd_count,
    float* avg_weight_change
);

/**
 * @brief Reset STDP statistics
 *
 * WHAT: Zero all learning counters
 * WHY: Start fresh measurement period
 * HOW: Set all counters to zero
 *
 * @param learner STDP learner
 */
void stdp_reset_statistics(stdp_learner_t* learner);

//=============================================================================
// Weight-Dependent STDP (Advanced)
//=============================================================================

/**
 * @brief Apply weight-dependent STDP
 *
 * WHAT: Scale plasticity by current weight (soft bounds)
 * WHY: Prevent runaway potentiation/depression
 * HOW: Multiply by (1 - w/w_max) for LTP, (w/w_max) for LTD
 *
 * ALGORITHM:
 * ```
 * if (Δt > 0):  // LTP
 *     Δw = A+ × exp(-Δt/τ+) × (1 - w/w_max)
 * else:         // LTD
 *     Δw = A- × exp(|Δt|/τ-) × (w/w_max)
 * ```
 *
 * BENEFIT: Prevents saturation at bounds while allowing exploration
 *
 * @param learner STDP learner
 * @param synapse Synapse to update
 * @param t_pre Presynaptic spike time
 * @param t_post Postsynaptic spike time
 * @return Weight change with soft bounds
 */
float stdp_apply_weight_dependent(
    stdp_learner_t* learner,
    synapse_t* synapse,
    uint64_t t_pre,
    uint64_t t_post
);

#ifdef __cplusplus
}
#endif

#endif // NIMCP_STDP_H
