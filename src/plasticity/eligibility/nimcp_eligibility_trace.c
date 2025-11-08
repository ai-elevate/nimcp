//=============================================================================
// nimcp_eligibility_trace.c - Eligibility Trace Implementation
//=============================================================================
/**
 * @file nimcp_eligibility_trace.c
 * @brief Implementation of eligibility traces for temporal credit assignment
 *
 * WHAT: Exponentially decaying memory of synaptic activity
 * WHY: Bridge temporal gap between actions and rewards
 * HOW: e(t) = λ^Δt × e(t-1) + δ(spike)
 *
 * ALGORITHM:
 * ```
 * // Update trace with spike:
 * Δt = current_time - last_update
 * e = λ^Δt × e_old + spike_contribution
 *
 * // Apply reward:
 * Δw = η × e × reward × dopamine
 * ```
 *
 * PERFORMANCE:
 * - eligibility_trace_update: O(1) - single power computation
 * - eligibility_apply_reward: O(1) - arithmetic only
 *
 * @author NIMCP Development Team
 * @date 2025-11-08
 * @version 2.7.0 Phase 4
 */

#include "plasticity/eligibility/nimcp_eligibility_trace.h"
#include <math.h>

// Note: synapse_t structure is defined in the headers included
// from nimcp_eligibility_trace.h and forward-declared there

//=============================================================================
// Lifecycle Functions
//=============================================================================

/**
 * @brief Get default eligibility trace configuration
 *
 * WHAT: Return RL-standard parameters
 * WHY: Based on Sutton & Barto (2018) recommendations
 * HOW: Static initialization with empirical values
 *
 * @return Default configuration structure
 */
eligibility_config_t eligibility_default_config(void) {
    eligibility_config_t config = {
        .decay_lambda = 0.95f,        // 95% decay per timestep (~20 step memory)
        .learning_rate = 0.001f,      // Modest learning rate
        .use_neuromodulation = true,  // Dopamine gating enabled
        .trace_threshold = 0.01f      // Ignore traces < 1%
    };
    return config;
}

/**
 * @brief Initialize eligibility trace for synapse
 *
 * WHAT: Set trace to zero, record timestamp
 * WHY: Fresh start for learning
 * HOW: Direct field initialization
 *
 * @param trace Trace structure to initialize
 * @param current_time Current simulation time (ms)
 */
void eligibility_trace_init(eligibility_trace_t* trace, uint64_t current_time) {
    // Guard: NULL trace
    if (!trace) {
        return;
    }

    // WHAT: Initialize trace to zero
    // WHY: No eligibility at start
    // HOW: Direct assignment
    trace->trace = 0.0f;
    trace->last_update = current_time;
}

//=============================================================================
// Trace Update Functions
//=============================================================================

/**
 * @brief Update eligibility trace with spike
 *
 * WHAT: Decay trace for elapsed time, add spike contribution
 * WHY: Mark synapse as recently active
 * HOW: e(t) = λ^Δt × e(t-1) + spike_contribution
 *
 * ALGORITHM:
 * 1. Compute Δt = current_time - last_update
 * 2. Decay: e × λ^Δt (exponential decay)
 * 3. Add spike: e + spike_contribution
 * 4. Update timestamp
 *
 * PERFORMANCE: O(1) - single pow() call
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
) {
    // Guard: NULL trace or config
    if (!trace || !config) {
        return;
    }

    // STEP 1: Compute time elapsed since last update
    // WHAT: Δt = current_time - last_update
    // WHY: Determine how much decay to apply
    // HOW: Unsigned subtraction (time always increases)
    uint64_t delta_t = current_time - trace->last_update;

    // STEP 2: Apply exponential decay
    // WHAT: Decay trace by λ^Δt
    // WHY: Traces fade over time
    // HOW: Use powf() for exact exponential
    // OPTIMIZATION: Could use lookup table for common Δt values
    if (delta_t > 0) {
        float decay_factor = powf(config->decay_lambda, (float)delta_t);
        trace->trace *= decay_factor;
    }

    // STEP 3: Add spike contribution
    // WHAT: Increment trace by spike amount
    // WHY: Spike marks synapse as eligible
    // HOW: Simple addition
    trace->trace += spike_contribution;

    // STEP 4: Clamp trace to [0, 1]
    // WHAT: Prevent trace from exceeding 1.0
    // WHY: Traces represent eligibility probability
    // HOW: fminf() with 1.0 upper bound
    if (trace->trace > 1.0f) {
        trace->trace = 1.0f;
    }

    // STEP 5: Update timestamp
    // WHAT: Record this update time
    // WHY: Needed for next decay calculation
    // HOW: Direct assignment
    trace->last_update = current_time;
}

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
) {
    // Guard: NULL trace or config
    if (!trace || !config) {
        return;
    }

    // WHAT: Compute and apply exponential decay
    // WHY: Update trace to current time
    // HOW: Same as eligibility_trace_update but without spike contribution
    uint64_t delta_t = current_time - trace->last_update;

    if (delta_t > 0) {
        float decay_factor = powf(config->decay_lambda, (float)delta_t);
        trace->trace *= decay_factor;
        trace->last_update = current_time;
    }

    // OPTIMIZATION: If trace becomes negligible, zero it out
    // WHAT: Set very small traces to zero
    // WHY: Avoid denormal floating-point performance issues
    // HOW: Threshold at 0.0001 (0.01%)
    if (trace->trace < 0.0001f) {
        trace->trace = 0.0f;
    }
}

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
 * THREE-FACTOR RULE:
 * 1. Eligibility (e): Was synapse recently active?
 * 2. Reward (r): Was outcome good/bad?
 * 3. Dopamine (d): Should we learn from this?
 *
 * ALGORITHM:
 * ```
 * if (trace < threshold): return 0  // Skip negligible traces
 * if (use_neuromodulation):
 *     Δw = η × e × r × d
 * else:
 *     Δw = η × e × r
 * w_new = w_old + Δw
 * ```
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
) {
    // Guard: NULL synapse, trace, or config
    if (!synapse || !trace || !config) {
        return 0.0f;
    }

    // OPTIMIZATION: Skip negligible traces
    // WHAT: Check if trace exceeds threshold
    // WHY: Avoid wasting computation on tiny weight changes
    // HOW: Compare to config.trace_threshold
    if (trace->trace < config->trace_threshold) {
        return 0.0f;
    }

    // STEP 1: Compute base weight change
    // WHAT: Δw = learning_rate × trace × reward
    // WHY: Scale by eligibility and reward magnitude
    // HOW: Simple multiplication
    float delta_w = config->learning_rate * trace->trace * reward;

    // STEP 2: Apply neuromodulation gating (if enabled)
    // WHAT: Multiply by dopamine level
    // WHY: Dopamine gates plasticity (3-factor rule)
    // HOW: Conditional multiplication
    if (config->use_neuromodulation) {
        delta_w *= dopamine_level;
    }

    // STEP 3: Update synapse weight
    // WHAT: Apply weight change
    // WHY: Implement learning
    // HOW: w_new = w_old + Δw
    synapse->weight += delta_w;

    // NOTE: No hard weight bounds here - caller can apply if needed
    // This allows flexibility in weight range constraints

    return delta_w;
}

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
) {
    // Guard: NULL checks
    if (!synapse || !trace || !config) {
        return 0.0f;
    }

    // STEP 1: Decay trace to current time
    // WHAT: Update trace for elapsed time
    // WHY: Ensure trace reflects current eligibility
    // HOW: Call eligibility_trace_decay()
    eligibility_trace_decay(trace, config, current_time);

    // STEP 2: Apply reward-based learning
    // WHAT: Update weight based on trace and reward
    // WHY: Implement three-factor learning rule
    // HOW: Call eligibility_apply_reward()
    float delta_w = eligibility_apply_reward(
        synapse, trace, config, reward, dopamine_level
    );

    return delta_w;
}

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
 * @return Trace value [0, 1], or 0.0 if trace is NULL
 */
float eligibility_get_trace(const eligibility_trace_t* trace) {
    return trace ? trace->trace : 0.0f;
}

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
) {
    // Guard: NULL checks
    if (!trace || !config) {
        return false;
    }

    // WHAT: Compare trace to threshold
    // WHY: Determine if trace is worth processing
    // HOW: Simple comparison
    return trace->trace >= config->trace_threshold;
}
