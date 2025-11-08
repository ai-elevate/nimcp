//=============================================================================
// nimcp_eligibility_trace.h - Eligibility Traces for Temporal Credit Assignment
//=============================================================================
/**
 * @file nimcp_eligibility_trace.h
 * @brief Eligibility traces for bridging temporal gap in reinforcement learning
 *
 * WHAT: Exponentially decaying memory of synaptic activity
 * WHY: Solve distal reward problem (assign credit to past actions)
 * HOW: e(t) = λ × e(t-1) + δ(spike), then Δw = η × e × reward × dopamine
 *
 * BIOLOGICAL BASIS:
 *   - Izhikevich (2007): "Solving the distal reward problem through STDP"
 *   - Traces persist 100-1000ms after spike (λ = 0.9-0.99)
 *   - Dopamine gates trace-based learning (3-factor rule)
 *
 * USE CASES:
 *   - Temporal difference learning: Q-learning, SARSA
 *   - Policy gradients: Actor-critic algorithms
 *   - Sequence learning: Credit assignment across time
 *   - Motor control: Delayed feedback (e.g., reaching task)
 *
 * @author NIMCP Development Team
 * @date 2025-11-08
 * @version 2.7.0 Phase 4
 */

#ifndef NIMCP_ELIGIBILITY_TRACE_H
#define NIMCP_ELIGIBILITY_TRACE_H

#include <stdint.h>
#include <stdbool.h>
#include "core/neuralnet/nimcp_neuralnet.h"

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// Configuration
//=============================================================================

/**
 * @brief Eligibility trace configuration
 *
 * WHAT: Parameters for trace decay and learning
 * WHY: Control temporal credit assignment window
 * HOW: Decay constant, learning rate, neuromodulation gating
 */
typedef struct {
    float decay_lambda;       /**< Trace decay per timestep (default: 0.95) */
    float learning_rate;      /**< Base learning rate (default: 0.001) */
    bool use_neuromodulation; /**< Gate by dopamine (default: true) */
    float trace_threshold;    /**< Minimum trace for update (default: 0.01) */
} eligibility_config_t;

/**
 * @brief Eligibility trace state for a synapse
 *
 * WHAT: Runtime eligibility trace value
 * WHY: Track synaptic eligibility for credit assignment
 * HOW: Decays exponentially, reset by spikes
 */
typedef struct {
    float trace;              /**< Current trace value [0, 1] */
    uint64_t last_update;     /**< Last time trace was updated (ms) */
} eligibility_trace_t;

//=============================================================================
// Lifecycle Functions
//=============================================================================

/**
 * @brief Get default eligibility trace configuration
 *
 * WHAT: Return sensible default parameters
 * WHY: Based on RL literature (Sutton & Barto 2018)
 * HOW: Static initialization
 *
 * DEFAULTS:
 * - λ = 0.95 (decay over ~20 timesteps, half-life ~14ms)
 * - η = 0.001 (modest learning rate)
 * - Use neuromodulation: true (dopamine gating)
 * - Trace threshold: 0.01 (ignore tiny traces)
 *
 * @return Default configuration
 */
eligibility_config_t eligibility_default_config(void);

/**
 * @brief Initialize eligibility trace for synapse
 *
 * WHAT: Set trace to zero, record timestamp
 * WHY: Fresh start for learning
 * HOW: Direct initialization
 *
 * @param trace Trace structure to initialize
 * @param current_time Current simulation time (ms)
 */
void eligibility_trace_init(eligibility_trace_t* trace, uint64_t current_time);

//=============================================================================
// Trace Update Functions
//=============================================================================

/**
 * @brief Update eligibility trace with spike
 *
 * WHAT: Decay trace and add spike contribution
 * WHY: Mark synapse as recently active
 * HOW: e(t) = λ^Δt × e(t-1) + δ(spike)
 *
 * ALGORITHM:
 * ```
 * Δt = current_time - last_update
 * e = λ^Δt × e_old + spike_contribution
 * ```
 *
 * PERFORMANCE: O(1) - single power computation
 *
 * @param trace Eligibility trace
 * @param config Configuration
 * @param current_time Current time (ms)
 * @param spike_contribution Spike contribution (typically 1.0)
 */
void eligibility_trace_update(
    eligibility_trace_t* trace,
    const eligibility_config_t* config,
    uint64_t current_time,
    float spike_contribution
);

/**
 * @brief Decay eligibility trace without spike
 *
 * WHAT: Apply exponential decay for elapsed time
 * WHY: Traces naturally decay over time
 * HOW: e(t) = λ^Δt × e(t-1)
 *
 * USE CASE: Call every timestep to decay all traces
 *
 * @param trace Eligibility trace
 * @param config Configuration
 * @param current_time Current time (ms)
 */
void eligibility_trace_decay(
    eligibility_trace_t* trace,
    const eligibility_config_t* config,
    uint64_t current_time
);

//=============================================================================
// Learning Functions
//=============================================================================

/**
 * @brief Apply trace-based weight update
 *
 * WHAT: Update synapse weight using eligibility trace and reward
 * WHY: Assign credit to recently active synapses
 * HOW: Δw = η × e × reward × dopamine
 *
 * ALGORITHM:
 * ```
 * if (use_neuromodulation):
 *     Δw = learning_rate × trace × reward × dopamine
 * else:
 *     Δw = learning_rate × trace × reward
 * w_new = w_old + Δw
 * ```
 *
 * THREE-FACTOR RULE:
 * 1. Trace (eligibility): Was synapse recently active?
 * 2. Reward: Was outcome good/bad?
 * 3. Dopamine: Should we learn from this?
 *
 * PERFORMANCE: O(1) - simple arithmetic
 *
 * @param synapse Synapse to update
 * @param trace Eligibility trace for this synapse
 * @param config Configuration
 * @param reward Reward signal [-1, 1]
 * @param dopamine_level Dopamine concentration [0, 1]
 * @return Weight change (Δw)
 */
float eligibility_apply_reward(
    synapse_t* synapse,
    const eligibility_trace_t* trace,
    const eligibility_config_t* config,
    float reward,
    float dopamine_level
);

/**
 * @brief Combined trace update and learning
 *
 * WHAT: Decay trace, then apply reward-based learning
 * WHY: Convenience function for typical use case
 * HOW: Call eligibility_trace_decay(), then eligibility_apply_reward()
 *
 * TYPICAL USAGE:
 * ```
 * // Each timestep:
 * for (each synapse with trace):
 *     eligibility_update_and_learn(
 *         synapse, trace, config,
 *         current_time, reward, dopamine
 *     );
 * ```
 *
 * @param synapse Synapse to update
 * @param trace Eligibility trace
 * @param config Configuration
 * @param current_time Current time (ms)
 * @param reward Reward signal
 * @param dopamine_level Dopamine concentration
 * @return Weight change
 */
float eligibility_update_and_learn(
    synapse_t* synapse,
    eligibility_trace_t* trace,
    const eligibility_config_t* config,
    uint64_t current_time,
    float reward,
    float dopamine_level
);

//=============================================================================
// Query Functions
//=============================================================================

/**
 * @brief Get current trace value
 *
 * WHAT: Return trace strength
 * WHY: Query eligibility for debugging/analysis
 * HOW: Direct field access
 *
 * @param trace Eligibility trace
 * @return Trace value [0, 1]
 */
float eligibility_get_trace(const eligibility_trace_t* trace);

/**
 * @brief Check if trace is significant
 *
 * WHAT: Test if trace exceeds threshold
 * WHY: Skip updates for negligible traces (performance)
 * HOW: Compare to config.trace_threshold
 *
 * @param trace Eligibility trace
 * @param config Configuration
 * @return true if trace > threshold
 */
bool eligibility_is_significant(
    const eligibility_trace_t* trace,
    const eligibility_config_t* config
);

#ifdef __cplusplus
}
#endif

#endif // NIMCP_ELIGIBILITY_TRACE_H
