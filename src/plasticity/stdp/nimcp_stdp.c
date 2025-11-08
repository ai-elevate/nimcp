//=============================================================================
// nimcp_stdp.c - Spike-Timing-Dependent Plasticity Implementation
//=============================================================================
/**
 * @file nimcp_stdp.c
 * @brief Implementation of STDP learning rule
 *
 * WHAT: Exponential STDP window for temporal learning
 * WHY: Biological learning rule that captures causality
 * HOW: Compute weight changes based on spike time differences
 *
 * ALGORITHM:
 * ```
 * Δt = t_post - t_pre
 * if (Δt > 0):
 *     Δw = A+ × exp(-Δt / τ+)  // LTP (causal)
 * else:
 *     Δw = A- × exp(|Δt| / τ-) // LTD (anti-causal)
 * w_new = clamp(w_old + Δw, w_min, w_max)
 * ```
 *
 * PERFORMANCE:
 * - stdp_apply: O(1) - single exponential computation
 * - stdp_apply_from_history: O(H) or O(H²) depending on mode
 * - stdp_apply_to_network: O(S) where S = total synapses
 *
 * @author NIMCP Development Team
 * @date 2025-11-08
 * @version 2.7.0 Phase 4
 */

#include "plasticity/stdp/nimcp_stdp.h"
#include "utils/memory/nimcp_memory.h"
#include "core/neuralnet/nimcp_neuralnet.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

// Note: neuron_t, synapse_t, and neural_network_t structures are defined
// in core/neuralnet/nimcp_neuralnet.h and included above

//=============================================================================
// Lifecycle Functions
//=============================================================================

/**
 * @brief Get default STDP configuration
 *
 * WHAT: Return biologically plausible STDP parameters
 * WHY: Based on literature (Bi & Poo 1998, Song et al. 2000)
 * HOW: Static initialization with empirical values
 *
 * @return Default configuration structure
 */
stdp_config_t stdp_default_config(void) {
    stdp_config_t config = {
        .tau_plus = 20.0f,         // 20ms LTP window
        .tau_minus = 20.0f,        // 20ms LTD window
        .A_plus = 0.01f,           // 1% max weight change for LTP
        .A_minus = -0.012f,        // 1.2% for LTD (slight asymmetry)
        .w_min = 0.0f,             // Non-negative weights
        .w_max = 10.0f,            // Maximum weight
        .learning_rate = 1.0f,     // No global scaling
        .nearest_spike = true      // Use nearest spike pair (fast)
    };
    return config;
}

/**
 * @brief Create STDP learner with configuration
 *
 * WHAT: Initialize STDP system with lookup tables for efficiency
 * WHY: Pre-compute exponentials for common time differences
 * HOW: Allocate structure, copy config, create lookup tables
 *
 * OPTIMIZATION:
 * - Lookup tables for exp(-t/τ) cover 0-200ms in 1ms steps
 * - Saves ~100-1000x vs repeated exp() calls
 * - Memory cost: 2 × 200 × 4 bytes = 1.6KB
 *
 * @param config Configuration (or NULL for defaults)
 * @return STDP learner handle, or NULL on error
 */
stdp_learner_t* stdp_create(const stdp_config_t* config) {
    // WHAT: Allocate main structure (zero-initialized)
    // WHY: nimcp_calloc ensures memory tracking and leak detection
    // HOW: Single allocation for learner structure
    stdp_learner_t* learner = (stdp_learner_t*)nimcp_calloc(1, sizeof(stdp_learner_t));

    // Guard: Allocation failure
    if (!learner) {
        fprintf(stderr, "ERROR: stdp_create: Failed to allocate learner\n");
        return NULL;
    }

    // WHAT: Copy configuration or use defaults
    // WHY: Allow user customization while providing sensible defaults
    // HOW: Direct copy if config provided, else use stdp_default_config()
    if (config) {
        learner->config = *config;
    } else {
        learner->config = stdp_default_config();
    }

    // WHAT: Create exponential lookup tables for performance
    // WHY: exp() is expensive (~100 cycles), lookups are fast (~4 cycles)
    // HOW: Pre-compute exp(-t/τ) for t = 0..200ms in 1ms steps
    learner->lookup_size = 200;  // 200ms window (covers > 99% of STDP window)

    // Allocate LTP lookup table
    learner->ltp_lookup = (float*)nimcp_malloc(learner->lookup_size * sizeof(float));
    if (!learner->ltp_lookup) {
        fprintf(stderr, "ERROR: stdp_create: Failed to allocate LTP lookup\n");
        stdp_destroy(learner);
        return NULL;
    }

    // Allocate LTD lookup table
    learner->ltd_lookup = (float*)nimcp_malloc(learner->lookup_size * sizeof(float));
    if (!learner->ltd_lookup) {
        fprintf(stderr, "ERROR: stdp_create: Failed to allocate LTD lookup\n");
        stdp_destroy(learner);
        return NULL;
    }

    // WHAT: Pre-compute exponentials
    // WHY: Amortize cost over many STDP updates
    // HOW: Loop through time steps, compute exp(-t/τ)
    for (uint32_t t = 0; t < learner->lookup_size; t++) {
        // LTP: exp(-t / τ+)
        learner->ltp_lookup[t] = expf(-(float)t / learner->config.tau_plus);

        // LTD: exp(-t / τ-)
        learner->ltd_lookup[t] = expf(-(float)t / learner->config.tau_minus);
    }

    return learner;
}

/**
 * @brief Destroy STDP learner
 *
 * WHAT: Free all resources (lookup tables, main structure)
 * WHY: Prevent memory leaks
 * HOW: Free in reverse order of allocation
 *
 * SAFETY: NULL-safe, can call on partially initialized learner
 *
 * @param learner STDP learner to destroy (can be NULL)
 */
void stdp_destroy(stdp_learner_t* learner) {
    // Guard: NULL learner (safe to destroy NULL)
    if (!learner) {
        return;
    }

    // WHAT: Free lookup tables
    // WHY: Release pre-computed exponential tables (~1.6KB)
    // HOW: nimcp_free() for memory tracking
    if (learner->ltp_lookup) nimcp_free(learner->ltp_lookup);
    if (learner->ltd_lookup) nimcp_free(learner->ltd_lookup);

    // WHAT: Free main structure
    // WHY: Release learner allocation
    // HOW: Use nimcp_free() for leak detection
    nimcp_free(learner);
}

//=============================================================================
// Core STDP Update Functions
//=============================================================================

/**
 * @brief Apply STDP to synapse given spike pair
 *
 * WHAT: Update synapse weight based on pre/post spike timing
 * WHY: Implement biological plasticity rule
 * HOW: Compute Δt, lookup exponential, apply weight change
 *
 * ALGORITHM:
 * 1. Compute Δt = t_post - t_pre
 * 2. If Δt > 0 (pre before post): LTP via ltp_lookup
 * 3. If Δt < 0 (post before pre): LTD via ltd_lookup
 * 4. Apply weight change: w += Δw × learning_rate
 * 5. Clamp to [w_min, w_max]
 * 6. Update statistics
 *
 * PERFORMANCE: O(1) - single lookup + arithmetic
 *
 * @param learner STDP learner
 * @param synapse Synapse to update
 * @param t_pre Presynaptic spike time (ms)
 * @param t_post Postsynaptic spike time (ms)
 * @return Weight change (Δw)
 */
float stdp_apply(
    stdp_learner_t* learner,
    synapse_t* synapse,
    uint64_t t_pre,
    uint64_t t_post
) {
    // Guard: NULL learner or synapse
    if (!learner || !synapse) {
        return 0.0f;
    }

    // STEP 1: Compute spike time difference
    // WHAT: Δt = t_post - t_pre (signed)
    // WHY: Positive Δt = pre before post (causal, LTP)
    //      Negative Δt = post before pre (anti-causal, LTD)
    // HOW: Cast to signed int64_t for subtraction
    int64_t delta_t = (int64_t)t_post - (int64_t)t_pre;

    float delta_w = 0.0f;

    // STEP 2: Apply STDP rule based on causality
    if (delta_t > 0) {
        // WHAT: LTP (Long-Term Potentiation) - strengthen synapse
        // WHY: Pre spiked before post (causality)
        // HOW: Δw = A+ × exp(-Δt / τ+)

        // Use lookup table if Δt is within range
        if (delta_t < (int64_t)learner->lookup_size) {
            delta_w = learner->config.A_plus * learner->ltp_lookup[delta_t];
        } else {
            // Fallback to direct exponential for very large Δt
            delta_w = learner->config.A_plus * expf(-(float)delta_t / learner->config.tau_plus);
        }

        learner->ltp_count++;

    } else if (delta_t < 0) {
        // WHAT: LTD (Long-Term Depression) - weaken synapse
        // WHY: Post spiked before pre (anti-causality)
        // HOW: Δw = A- × exp(|Δt| / τ-)

        uint64_t abs_delta_t = (uint64_t)(-delta_t);

        // Use lookup table if |Δt| is within range
        if (abs_delta_t < learner->lookup_size) {
            delta_w = learner->config.A_minus * learner->ltd_lookup[abs_delta_t];
        } else {
            // Fallback to direct exponential
            delta_w = learner->config.A_minus * expf(-(float)abs_delta_t / learner->config.tau_minus);
        }

        learner->ltd_count++;
    }
    // Note: If delta_t == 0 (simultaneous spikes), no change (delta_w = 0)

    // STEP 3: Scale by global learning rate
    // WHAT: Apply learning rate multiplier
    // WHY: Allow global control of plasticity strength
    // HOW: Simple multiplication
    delta_w *= learner->config.learning_rate;

    // STEP 4: Update synapse weight with clamping
    // WHAT: Apply weight change and enforce bounds
    // WHY: Prevent runaway potentiation/depression
    // HOW: w_new = clamp(w_old + Δw, w_min, w_max)
    float new_weight = synapse->weight + delta_w;
    new_weight = fmaxf(learner->config.w_min, fminf(learner->config.w_max, new_weight));
    synapse->weight = new_weight;

    // STEP 5: Update statistics
    // WHAT: Track learning activity
    // WHY: Monitor STDP dynamics
    // HOW: Exponential moving average of |Δw|
    learner->updates_applied++;
    float abs_delta_w = fabsf(delta_w);
    learner->avg_weight_change = 0.99f * learner->avg_weight_change + 0.01f * abs_delta_w;

    return delta_w;
}

/**
 * @brief Apply dopamine-modulated STDP
 *
 * WHAT: Scale STDP by neuromodulator level
 * WHY: Dopamine gates plasticity (reward-modulated learning)
 * HOW: Compute base STDP, then multiply by dopamine
 *
 * BIOLOGICAL BASIS:
 * - Dopamine D1 receptors enhance LTP
 * - Low dopamine → no plasticity (Schultz et al., 1997)
 *
 * @param learner STDP learner
 * @param synapse Synapse to update
 * @param t_pre Presynaptic spike time
 * @param t_post Postsynaptic spike time
 * @param dopamine_level Dopamine [0, 1]
 * @return Modulated weight change
 */
float stdp_apply_modulated(
    stdp_learner_t* learner,
    synapse_t* synapse,
    uint64_t t_pre,
    uint64_t t_post,
    float dopamine_level
) {
    // Guard: NULL checks
    if (!learner || !synapse) {
        return 0.0f;
    }

    // WHAT: Save original weight for delta calculation
    // WHY: Need to return actual change, not base change
    // HOW: Store before STDP application
    float weight_before = synapse->weight;

    // STEP 1: Apply base STDP rule
    // WHAT: Compute unmodulated weight change
    // WHY: Get baseline plasticity signal
    // HOW: Call stdp_apply()
    float base_delta = stdp_apply(learner, synapse, t_pre, t_post);

    // STEP 2: Modulate by dopamine
    // WHAT: Scale weight change by dopamine level
    // WHY: Dopamine gates synaptic plasticity
    // HOW: Interpolate between old weight and new weight
    // NOTE: This is done after stdp_apply, so we need to reverse and re-apply

    // Reverse the base update
    synapse->weight = weight_before;

    // Apply modulated update
    float modulated_delta = base_delta * dopamine_level;
    float new_weight = synapse->weight + modulated_delta;
    new_weight = fmaxf(learner->config.w_min, fminf(learner->config.w_max, new_weight));
    synapse->weight = new_weight;

    return modulated_delta;
}

/**
 * @brief Apply weight-dependent STDP
 *
 * WHAT: Scale plasticity by current weight (soft bounds)
 * WHY: Prevent saturation at weight bounds
 * HOW: Multiply LTP by (1 - w/w_max), LTD by (w/w_max)
 *
 * BENEFIT: Smooth approach to bounds, no hard saturation
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
) {
    // Guard: NULL checks
    if (!learner || !synapse) {
        return 0.0f;
    }

    // STEP 1: Compute spike time difference
    int64_t delta_t = (int64_t)t_post - (int64_t)t_pre;

    float delta_w = 0.0f;
    float w_normalized = (synapse->weight - learner->config.w_min) /
                         (learner->config.w_max - learner->config.w_min);

    // STEP 2: Apply weight-dependent STDP
    if (delta_t > 0) {
        // WHAT: LTP with soft upper bound
        // WHY: Plasticity decreases as weight approaches w_max
        // HOW: Multiply by (1 - w/w_max)
        float exp_factor;
        if (delta_t < (int64_t)learner->lookup_size) {
            exp_factor = learner->ltp_lookup[delta_t];
        } else {
            exp_factor = expf(-(float)delta_t / learner->config.tau_plus);
        }
        delta_w = learner->config.A_plus * exp_factor * (1.0f - w_normalized);
        learner->ltp_count++;

    } else if (delta_t < 0) {
        // WHAT: LTD with soft lower bound
        // WHY: Depression decreases as weight approaches w_min
        // HOW: Multiply by (w/w_max)
        uint64_t abs_delta_t = (uint64_t)(-delta_t);
        float exp_factor;
        if (abs_delta_t < learner->lookup_size) {
            exp_factor = learner->ltd_lookup[abs_delta_t];
        } else {
            exp_factor = expf(-(float)abs_delta_t / learner->config.tau_minus);
        }
        delta_w = learner->config.A_minus * exp_factor * w_normalized;
        learner->ltd_count++;
    }

    // STEP 3: Apply weight change (no clamping needed with soft bounds)
    delta_w *= learner->config.learning_rate;
    synapse->weight += delta_w;

    // Still clamp as safety net
    synapse->weight = fmaxf(learner->config.w_min,
                           fminf(learner->config.w_max, synapse->weight));

    // Update statistics
    learner->updates_applied++;
    learner->avg_weight_change = 0.99f * learner->avg_weight_change + 0.01f * fabsf(delta_w);

    return delta_w;
}

//=============================================================================
// Statistics Functions
//=============================================================================

/**
 * @brief Get STDP learning statistics
 *
 * WHAT: Query learning activity metrics
 * WHY: Monitor plasticity dynamics
 * HOW: Copy statistics to output parameters
 *
 * @param learner STDP learner
 * @param ltp_count Output: LTP event count (can be NULL)
 * @param ltd_count Output: LTD event count (can be NULL)
 * @param avg_weight_change Output: Average |Δw| (can be NULL)
 */
void stdp_get_statistics(
    const stdp_learner_t* learner,
    uint64_t* ltp_count,
    uint64_t* ltd_count,
    float* avg_weight_change
) {
    // Guard: NULL learner
    if (!learner) {
        return;
    }

    // WHAT: Copy statistics to outputs (if provided)
    // WHY: Allow caller to request subset of statistics
    // HOW: NULL-check each pointer before writing
    if (ltp_count) *ltp_count = learner->ltp_count;
    if (ltd_count) *ltd_count = learner->ltd_count;
    if (avg_weight_change) *avg_weight_change = learner->avg_weight_change;
}

/**
 * @brief Reset STDP statistics
 *
 * WHAT: Zero all learning counters
 * WHY: Start fresh measurement period
 * HOW: Set counters to zero
 *
 * @param learner STDP learner
 */
void stdp_reset_statistics(stdp_learner_t* learner) {
    // Guard: NULL learner
    if (!learner) {
        return;
    }

    // WHAT: Reset all statistics counters
    // WHY: Enable profiling of specific training periods
    // HOW: Direct assignment to zero
    learner->updates_applied = 0;
    learner->ltp_count = 0;
    learner->ltd_count = 0;
    learner->avg_weight_change = 0.0f;
}

//=============================================================================
// Network-Wide STDP Application (Phase 5)
//=============================================================================

/**
 * @brief Apply STDP to all synapses in network
 *
 * WHAT: Batch STDP update for entire network based on spike histories
 * WHY: Coordinate learning across all connections after network timestep
 * HOW: Iterate all neurons → synapses, apply STDP from spike history
 *
 * ALGORITHM:
 * ```
 * for each neuron N in network:
 *     for each outgoing synapse S from N:
 *         pre_neuron = N
 *         post_neuron = S.target
 *         if both neurons have recent spikes:
 *             apply STDP based on spike timing
 *             update statistics
 * ```
 *
 * TYPICAL USAGE:
 * ```
 * // After each network timestep:
 * neural_network_compute_step(network, t);
 * stdp_apply_to_network(learner, network);
 * ```
 *
 * PERFORMANCE: O(S) where S = total synapses in network
 * COMPLEXITY: Single pass through all synapses
 *
 * @param learner STDP learner with configuration
 * @param network Neural network (opaque pointer, cast to neural_network_t)
 */
void stdp_apply_to_network(
    stdp_learner_t* learner,
    void* network  // neural_network_t (opaque)
) {
    // Guard: NULL inputs
    if (!learner || !network) {
        return;
    }

    // WHAT: Cast opaque pointer to neural_network_t
    // WHY: Function signature uses void* to avoid circular dependencies
    // HOW: Safe cast - caller guarantees correct type
    neural_network_t net = (neural_network_t)network;

    // WHAT: Get network statistics to determine neuron count
    // WHY: Cannot access internal structure (opaque type)
    // HOW: Use public API neural_network_get_stats()
    network_stats_t stats;
    if (!neural_network_get_stats(net, &stats)) {
        return;  // Failed to get stats
    }

    // Guard: Network has no neurons
    if (stats.num_neurons == 0) {
        return;
    }

    // WHAT: Apply STDP to all neurons using public API
    // WHY: Use neural_network_apply_stdp() instead of direct access
    // HOW: Call per-neuron STDP function for each neuron
    // PERFORMANCE: O(S) where S = total synapses
    // NOTE: neural_network_apply_stdp() handles synapse iteration internally
    uint32_t total_updates = 0;
    for (uint32_t neuron_id = 0; neuron_id < stats.num_neurons; neuron_id++) {
        // WHAT: Apply STDP for this neuron's outgoing synapses
        // WHY: Update weights based on spike history
        // HOW: Delegate to network's STDP implementation
        // RESULT: Returns number of synapses updated
        uint32_t updates = neural_network_apply_stdp(net, neuron_id, stats.network_time);
        total_updates += updates;
    }

    // WHAT: Update learner statistics based on STDP applications
    // WHY: Track learning activity for monitoring
    // HOW: Increment update counter
    // NOTE: Network's STDP doesn't distinguish LTP/LTD, so we count total updates
    // TODO: Enhance network API to report LTP/LTD separately
    learner->updates_applied += total_updates;

    // ASSUMPTION: Roughly half of updates are LTP, half are LTD (average case)
    // This is a simplification; real ratio depends on spike timing distribution
    learner->ltp_count += total_updates / 2;
    learner->ltd_count += total_updates / 2;
}
