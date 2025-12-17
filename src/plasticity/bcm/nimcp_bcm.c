/**
 * @file nimcp_bcm.c
 * @brief Implementation of BCM learning rule
 *
 * ARCHITECTURAL OVERVIEW:
 * - Strategy Pattern: Configurable threshold update strategies
 * - Factory Pattern: Preset parameter configurations
 * - Value Object: Immutable parameter structs
 *
 * PERFORMANCE OPTIMIZATIONS:
 * - Branchless computation in hot paths
 * - SIMD-friendly memory layout
 * - Inlined fast-path functions
 * - Cache-coherent data structures
 *
 * COMPLEXITY ANALYSIS:
 * - bcm_apply_rule: O(1) - constant time per synapse
 * - bcm_update_threshold: O(1) - single exponential update
 * - bcm_compute_stats: O(n) - linear scan over synapses
 *
 * DESIGN PRINCIPLES:
 * - Guard clauses for early returns (no nested ifs)
 * - Functions < 50 lines
 * - Explicit what/why comments
 * - TDD verified
 */

#include "plasticity/bcm/nimcp_bcm.h"
#include "plasticity/bcm/nimcp_bcm_sleep_bridge.h"
#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_messages.h"
#include "utils/logging/nimcp_logging.h"
#include "security/nimcp_security.h"
#include <math.h>
#include <string.h>

#define LOG_MODULE "plasticity_bcm"

//=============================================================================
// Constants
//=============================================================================

#define BCM_EPSILON 1e-8f          // Prevent division by zero
#define BCM_WEIGHT_MIN 0.0f        // Minimum synaptic weight
#define BCM_WEIGHT_MAX 1.0f        // Maximum synaptic weight

//=============================================================================
// Inline Helper Functions
//=============================================================================

/**
 * @brief Clamp value to range [min, max]
 *
 * WHAT: Branchless clamp using fmin/fmax
 * WHY:  Faster than if-else on modern CPUs
 *
 * COMPLEXITY: O(1)
 * PERFORMANCE: ~2 CPU cycles (no branch misprediction)
 */
static inline float clamp_f(float value, float min_val, float max_val) {
    return fminf(fmaxf(value, min_val), max_val);
}

/**
 * @brief Compute BCM plasticity factor
 *
 * WHAT: φ(post, θ) = post × (post - θ)
 * WHY:  Core of BCM rule - supralinear in post-synaptic activity
 *
 * BIOLOGICAL:
 * - post > θ: φ > 0 → LTP
 * - post < θ: φ < 0 → LTD
 * - post = θ: φ = 0 → No change
 *
 * COMPLEXITY: O(1)
 * PERFORMANCE: ~3 CPU cycles
 */
static inline float bcm_plasticity_factor(float post_activity, float threshold) {
    /* WHAT: Compute deviation from threshold
     * WHY:  Determines direction of plasticity
     */
    float deviation = post_activity - threshold;

    /* WHAT: Multiply by post-synaptic activity
     * WHY:  Creates supralinear dependence (quadratic in post)
     */
    return post_activity * deviation;
}

//=============================================================================
// Core BCM Functions
//=============================================================================

bcm_synapse_t bcm_synapse_init(float initial_weight, float initial_threshold) {
    /* WHAT: Initialize BCM synapse with factory method (thread-safe version)
     * WHY:  Encapsulates initialization, ensures valid state with synchronization
     * COMPLEXITY: O(1)
     * THREAD SAFETY: Initializes spinlock for concurrent access protection
     */

    /* WHAT: Create synapse with zeroed memory
     * WHY:  Ensures all fields have defined values
     */
    bcm_synapse_t synapse = {0};

    /* WHAT: Clamp initial weight to valid range
     * WHY:  Prevent invalid initial states
     */
    synapse.weight = clamp_f(initial_weight, BCM_WEIGHT_MIN, BCM_WEIGHT_MAX);

    /* WHAT: Set initial threshold
     * WHY:  Starting point for sliding threshold dynamics
     * NOTE: Threshold will adapt based on activity
     */
    synapse.threshold = initial_threshold;

    /* WHAT: Initialize average activity to zero
     * WHY:  Will be computed from actual activity
     */
    synapse.avg_post_activity = 0.0F;

    /* WHAT: Initialize eligibility to zero
     * WHY:  Used for delayed reward assignment
     */
    synapse.eligibility = 0.0F;

    /* WHAT: Initialize sleep state to awake
     * WHY:  Default to normal (awake) learning dynamics
     */
    synapse.current_sleep_state = SLEEP_STATE_AWAKE;

    /* WHAT: Initialize spinlock for thread-safe updates
     * WHY:  Allow concurrent access when synapse is shared across threads
     * WHEN NEEDED: Only if multiple threads update same synapse (rare)
     * OVERHEAD: Minimal (~4 bytes), no performance cost if not shared
     */
    nimcp_spinlock_init(&synapse.lock);

    return synapse;
}

void bcm_update_threshold(bcm_synapse_t* synapse, float post_activity, float dt,
                         const bcm_params_t* params) {
    /* WHAT: Update sliding modification threshold (thread-safe)
     * WHY:  BCM threshold adapts to average post-synaptic activity²
     *
     * BIOLOGICAL: θ̇ = (post² - θ) / τ
     * - Threshold tracks E[post²]
     * - Self-stabilizing without explicit normalization
     *
     * COMPLEXITY: O(1)
     * THREAD SAFETY: Acquires spinlock (brief critical section ~30-50ns)
     */

    /* WHAT: Guard clause - validate inputs
     * WHY:  Prevent null pointer dereference
     */
    if (!synapse || !params) return;

    /* WHAT: Acquire spinlock for atomic threshold update
     * WHY:  Prevent race conditions when multiple threads access same synapse
     * PERFORMANCE: ~10-20ns overhead (acceptable for per-synapse lock)
     * NOTE: Only contends if threads share synapses (rare in typical use)
     */
    nimcp_spinlock_lock(&synapse->lock);

    /* WHAT: Get sleep modulation factor for threshold
     * WHY:  Sleep state modulates threshold (e.g., NREM raises it for LTD bias)
     */
    float theta_factor = bcm_sleep_theta_for_state(synapse->current_sleep_state);

    /* WHAT: Compute target threshold (post-synaptic activity squared)
     * WHY:  Biological BCM uses quadratic dependence on activity
     * RATIONALE: E[post²] sets crossover point between LTP and LTD
     */
    float target_threshold = post_activity * post_activity * theta_factor;

    /* WHAT: Compute time constant for exponential decay
     * WHY:  Convert dt from ms to decay fraction
     * FORMULA: decay = 1 - exp(-dt/τ)
     */
    float decay = 1.0F - expf(-dt / (params->threshold_time_constant + BCM_EPSILON));

    /* WHAT: Exponential moving average toward target
     * WHY:  Smooth threshold adaptation, prevents oscillations
     * FORMULA: θ_new = θ_old + decay × (target - θ_old)
     */
    synapse->threshold += decay * (target_threshold - synapse->threshold);

    /* WHAT: Clamp threshold to physiological range
     * WHY:  Prevent numerical instabilities and unrealistic values
     */
    synapse->threshold = clamp_f(synapse->threshold, params->min_threshold,
                                 params->max_threshold);

    /* WHAT: Update running average of post-synaptic activity
     * WHY:  Used for statistics and monitoring
     */
    float activity_decay = 1.0F - expf(-dt / (params->activity_time_constant + BCM_EPSILON));
    synapse->avg_post_activity += activity_decay * (post_activity - synapse->avg_post_activity);

    /* WHAT: Release spinlock
     * WHY:  Allow other threads to access synapse
     * PERFORMANCE: ~10-20ns
     */
    nimcp_spinlock_unlock(&synapse->lock);
}

void bcm_apply_rule(bcm_synapse_t* synapse, float pre_activity, float post_activity,
                   float dt, const bcm_params_t* params) {
    /* WHAT: Apply BCM plasticity rule to update synaptic weight (thread-safe)
     * WHY:  Implement sliding threshold LTP/LTD with concurrent access protection
     *
     * BIOLOGICAL: Δw = η × post × (post - θ) × pre
     * - Supralinear in post-synaptic activity
     * - Hebbian-like (requires pre and post co-activity)
     * - Self-stabilizing via sliding threshold
     *
     * COMPLEXITY: O(1)
     * PERFORMANCE: ~10 CPU cycles (branchless hot path) + spinlock overhead
     * THREAD SAFETY: Acquires spinlock (brief critical section ~50-80ns)
     */

    /* WHAT: Guard clause - validate inputs
     * WHY:  Prevent null pointer dereference
     */
    if (!synapse || !params) return;

    /* WHAT: Acquire spinlock for atomic weight update
     * WHY:  Prevent race conditions when multiple threads access same synapse
     * PERFORMANCE: ~10-20ns overhead (acceptable for per-synapse lock)
     * NOTE: Only contends if threads share synapses (rare in typical use)
     */
    nimcp_spinlock_lock(&synapse->lock);

    /* WHAT: Get sleep modulation factor for learning rate
     * WHY:  Sleep state modulates learning rate (e.g., reduced during sleep)
     */
    float lr_factor = bcm_sleep_lr_for_state(synapse->current_sleep_state);

    /* WHAT: Compute BCM plasticity factor φ(post, θ)
     * WHY:  Determines direction and magnitude of weight change
     * FORMULA: φ = post × (post - θ)
     */
    float plasticity_factor = bcm_plasticity_factor(post_activity, synapse->threshold);

    /* WHAT: Compute weight change with sleep modulation
     * WHY:  BCM rule with Hebbian pre-synaptic gating and sleep modulation
     * FORMULA: Δw = η × lr_factor × φ(post, θ) × pre × dt
     * NOTE: dt scaling makes rule time-step independent
     */
    float delta_w = params->learning_rate * lr_factor * plasticity_factor * pre_activity * dt;

    /* WHAT: Apply weight update
     * WHY:  Integrate weight change into current weight
     */
    float new_weight = synapse->weight + delta_w;

    /* WHAT: Clamp weight to valid physiological range
     * WHY:  Weights must stay in [0, 1] for biological realism
     * NOTE: Branchless clamp for performance
     */
    synapse->weight = clamp_f(new_weight, BCM_WEIGHT_MIN, BCM_WEIGHT_MAX);

    /* WHAT: Update eligibility trace for delayed reward
     * WHY:  Allows credit assignment when reward is delayed
     * FORMULA: e_new = e_old × decay + Δw
     */
    float eligibility_decay = 0.95F;  // Fast decay (~20ms time constant)
    synapse->eligibility = synapse->eligibility * eligibility_decay + delta_w;

    /* WHAT: Release spinlock
     * WHY:  Allow other threads to access synapse
     * PERFORMANCE: ~10-20ns
     */
    nimcp_spinlock_unlock(&synapse->lock);
}

void bcm_apply_rule_modulated(bcm_synapse_t* synapse, float pre_activity,
                             float post_activity, float dt, const bcm_params_t* params,
                             float neuromodulator_level) {
    /* WHAT: Apply BCM rule with neuromodulator gating
     * WHY:  Reward signals should gate cortical plasticity
     *
     * BIOLOGICAL:
     * - Dopamine gates LTP in cortex and striatum
     * - No dopamine → minimal plasticity
     * - High dopamine → enhanced plasticity
     *
     * COMPLEXITY: O(1)
     */

    /* WHAT: Guard clause - validate inputs
     * WHY:  Prevent null pointer dereference
     */
    if (!synapse || !params) return;

    /* WHAT: Clamp neuromodulator to valid range
     * WHY:  Ensure modulation factor is in [0, 1]
     */
    float modulation = clamp_f(neuromodulator_level, 0.0F, 1.0F);

    /* WHAT: Get sleep modulation factor for learning rate
     * WHY:  Sleep state modulates learning rate (e.g., reduced during sleep)
     */
    float lr_factor = bcm_sleep_lr_for_state(synapse->current_sleep_state);

    /* WHAT: Compute BCM plasticity factor
     * WHY:  Same as standard BCM rule
     */
    float plasticity_factor = bcm_plasticity_factor(post_activity, synapse->threshold);

    /* WHAT: Compute modulated weight change with sleep modulation
     * WHY:  Neuromodulator and sleep both scale learning rate
     * FORMULA: Δw = η × lr_factor × modulation × φ(post, θ) × pre × dt
     * BIOLOGICAL: DA concentration and sleep state both scale plasticity magnitude
     */
    float delta_w = params->learning_rate * lr_factor * modulation * plasticity_factor * pre_activity * dt;

    /* WHAT: Apply weight update with clamping
     * WHY:  Same as standard BCM, but modulated magnitude
     */
    synapse->weight = clamp_f(synapse->weight + delta_w, BCM_WEIGHT_MIN, BCM_WEIGHT_MAX);

    /* WHAT: Update eligibility trace
     * WHY:  Track recent plasticity for delayed reward
     */
    float eligibility_decay = 0.95F;
    synapse->eligibility = synapse->eligibility * eligibility_decay + delta_w;
}

//=============================================================================
// Factory Methods for Parameter Presets
//=============================================================================

bcm_params_t bcm_params_cortical(void) {
    /* WHAT: Factory method for cortical BCM parameters
     * WHY:  Provides biologically plausible defaults
     *
     * BIOLOGICAL: Based on visual cortex measurements
     * - Learning rate: moderate (0.01)
     * - Threshold τ: slow adaptation (~1 second)
     * - Activity τ: moderate averaging (~100 ms)
     *
     * COMPLEXITY: O(1)
     */

    bcm_params_t params;

    /* WHAT: Set learning rate for adult cortex
     * WHY:  Adult cortex has moderate plasticity
     */
    params.learning_rate = 0.01F;

    /* WHAT: Set threshold adaptation time constant
     * WHY:  Threshold should adapt slowly relative to activity
     * BIOLOGICAL: ~1 second timescale in cortex
     */
    params.threshold_time_constant = 1000.0F;  // ms

    /* WHAT: Set activity averaging time constant
     * WHY:  Average over multiple spikes for stability
     * BIOLOGICAL: ~100 ms integration window
     */
    params.activity_time_constant = 100.0F;  // ms

    /* WHAT: Set minimum threshold
     * WHY:  Prevent over-depression (all weights → 0)
     */
    params.min_threshold = 0.1F;

    /* WHAT: Set maximum threshold
     * WHY:  Prevent runaway potentiation
     */
    params.max_threshold = 10.0F;

    return params;
}

bcm_params_t bcm_params_critical_period(void) {
    /* WHAT: Factory method for critical period learning
     * WHY:  Developmental plasticity is higher than adult
     *
     * BIOLOGICAL: Models early postnatal development
     * - High learning rate (10x adult)
     * - Faster threshold adaptation
     * - Wider threshold range
     *
     * COMPLEXITY: O(1)
     */

    bcm_params_t params;

    /* WHAT: High learning rate for critical period
     * WHY:  Rapid learning during development
     * BIOLOGICAL: Critical periods have elevated plasticity
     */
    params.learning_rate = 0.1F;  // 10x adult

    /* WHAT: Faster threshold adaptation
     * WHY:  Development requires rapid tuning
     */
    params.threshold_time_constant = 500.0F;  // ms (2x faster)

    /* WHAT: Faster activity averaging
     * WHY:  Developmental plasticity responds to shorter timescales
     */
    params.activity_time_constant = 50.0F;  // ms

    /* WHAT: Lower minimum threshold
     * WHY:  Allow stronger depression during refinement
     */
    params.min_threshold = 0.05F;

    /* WHAT: Higher maximum threshold
     * WHY:  Allow stronger potentiation during development
     */
    params.max_threshold = 20.0F;

    return params;
}

bcm_params_t bcm_params_mature(void) {
    /* WHAT: Factory method for mature/adult learning
     * WHY:  Adult cortex has lower plasticity than critical period
     *
     * BIOLOGICAL: Models adult cortical plasticity
     * - Low learning rate (stable)
     * - Slow threshold adaptation
     * - Narrow threshold range
     *
     * COMPLEXITY: O(1)
     */

    bcm_params_t params;

    /* WHAT: Low learning rate for mature brain
     * WHY:  Stability over plasticity in adults
     * BIOLOGICAL: Adult cortex resists change
     */
    params.learning_rate = 0.001F;  // 10x lower than developmental

    /* WHAT: Slow threshold adaptation
     * WHY:  Mature brain has stable activity statistics
     */
    params.threshold_time_constant = 2000.0F;  // ms (very slow)

    /* WHAT: Slow activity averaging
     * WHY:  Integrate over longer timescales for stability
     */
    params.activity_time_constant = 200.0F;  // ms

    /* WHAT: Narrow threshold range
     * WHY:  Prevent large weight fluctuations in mature brain
     */
    params.min_threshold = 0.2F;
    params.max_threshold = 5.0F;

    return params;
}

//=============================================================================
// Statistics and Analysis
//=============================================================================

bool bcm_compute_stats(const bcm_synapse_t* synapses, uint32_t num_synapses,
                      bcm_stats_t* stats) {
    /* WHAT: Compute aggregate statistics over synapse population
     * WHY:  Monitor learning progress, detect instabilities
     *
     * COMPLEXITY: O(n) where n = num_synapses
     * PERFORMANCE: ~2n cache misses (linear scan)
     */

    /* WHAT: Guard clause - validate inputs
     * WHY:  Prevent null pointer dereference
     */
    if (!synapses || !stats || num_synapses == 0) return false;

    /* WHAT: Zero out statistics structure
     * WHY:  Clean slate for accumulation
     */
    memset(stats, 0, sizeof(bcm_stats_t));

    /* WHAT: Accumulate sums for mean computation
     * WHY:  Single pass for efficiency
     */
    float sum_weight = 0.0F;
    float sum_threshold = 0.0F;
    float sum_weight_squared = 0.0F;

    for (uint32_t i = 0; i < num_synapses; i++) {
        const bcm_synapse_t* syn = &synapses[i];

        sum_weight += syn->weight;
        sum_threshold += syn->threshold;
        sum_weight_squared += syn->weight * syn->weight;

        /* WHAT: Count LTP vs LTD events (approximate)
         * WHY:  Balance indicates learning dynamics
         * NOTE: Eligibility > 0 suggests recent LTP, < 0 suggests LTD
         */
        if (syn->eligibility > 0.0F) {
            stats->ltp_events++;
        } else if (syn->eligibility < 0.0F) {
            stats->ltd_events++;
        }
    }

    /* WHAT: Compute averages
     * WHY:  Mean weight and threshold indicate population state
     */
    stats->avg_weight = sum_weight / num_synapses;
    stats->avg_threshold = sum_threshold / num_synapses;

    /* WHAT: Compute weight variance
     * WHY:  Variance indicates weight distribution spread
     * FORMULA: Var(w) = E[w²] - E[w]²
     */
    float mean_squared = stats->avg_weight * stats->avg_weight;
    float mean_of_squares = sum_weight_squared / num_synapses;
    stats->weight_variance = mean_of_squares - mean_squared;

    /* WHAT: Record total update count (per synapse basis)
     * WHY:  Indicates total plasticity activity
     */
    stats->total_updates = num_synapses;

    return true;
}

//=============================================================================
// Sleep Integration
//=============================================================================

void bcm_set_sleep_state(bcm_synapse_t* synapse, sleep_state_t state) {
    /* WHAT: Update sleep state for synapse
     * WHY:  Sleep state modulates threshold and learning rate
     * HOW:  Set current_sleep_state field, will be applied during next update
     */
    if (!synapse) return;
    synapse->current_sleep_state = state;
}
