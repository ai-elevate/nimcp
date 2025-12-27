/**
 * @file nimcp_stdp.c
 * @brief Implementation of dopamine-modulated STDP
 *
 * NIMCP Phase: Option 2.1 (Integration with Phase C2.2)
 * Date: 2025-11-12
 */

#include "plasticity/stdp/nimcp_stdp.h"
#include "plasticity/stdp/nimcp_stdp_sleep_bridge.h"
#include "plasticity/neuromodulators/nimcp_neuromodulators.h"
#include "plasticity/neuromodulators/nimcp_phasic_tonic.h"
#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_messages.h"
#include "utils/logging/nimcp_logging.h"
#include "security/nimcp_security.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

#define LOG_MODULE "plasticity_stdp"

/* ============================================================================
 * Configuration
 * ============================================================================ */

stdp_config_t stdp_config_default(void) {
    stdp_config_t config = {
        .w_max = 1.0F,
        .learning_rate = 0.01F,
        .a_plus = 0.005F,           /* LTP amplitude (Bi & Poo, 1998) */
        .a_minus = 0.00525F,        /* LTD amplitude (slightly larger for balance) */
        .tau_plus = 0.020F,         /* 20 ms (seconds) */
        .tau_minus = 0.020F,        /* 20 ms */
        .enable_da_modulation = true,
        .da_modulation_gain = 100.0F,   /* DA concentration [0-1] → LR multiplier */
        .burst_amplification = 3.0F     /* 3x learning during bursts */
    };
    return config;
}

/* ============================================================================
 * Initialization
 * ============================================================================ */

void stdp_synapse_init(stdp_synapse_t* synapse) {
    stdp_config_t config = stdp_config_default();
    stdp_synapse_init_with_config(synapse, &config);
}

void stdp_synapse_init_with_config(stdp_synapse_t* synapse, const stdp_config_t* config) {
    LOG_DEBUG("Initializing STDP synapse with config");

    if (!synapse || !config) {
        LOG_ERROR("NULL pointer passed to stdp_synapse_init_with_config");
        return;
    }

    memset(synapse, 0, sizeof(stdp_synapse_t));

    synapse->weight = 0.5F;  /* Initialize to 50% of max */
    synapse->w_max = config->w_max;
    synapse->w_min = 0.0F;

    synapse->learning_rate = config->learning_rate;
    synapse->a_plus = config->a_plus;
    synapse->a_minus = config->a_minus;
    synapse->tau_plus = config->tau_plus;
    synapse->tau_minus = config->tau_minus;

    synapse->enable_da_modulation = config->enable_da_modulation;
    synapse->da_modulation_gain = config->da_modulation_gain;
    synapse->burst_amplification = config->burst_amplification;

    synapse->current_sleep_state = SLEEP_STATE_AWAKE;  /* Default to awake */

    /* Initialize spinlock for thread safety */
    nimcp_spinlock_init(&synapse->lock);

    synapse->pre_trace = 0.0F;
    synapse->post_trace = 0.0F;

    synapse->num_potentiation_events = 0;
    synapse->num_depression_events = 0;
    synapse->total_ltp = 0.0F;
    synapse->total_ltd = 0.0F;
}

/* ============================================================================
 * Trace Updates
 * ============================================================================ */

void stdp_update_traces(stdp_synapse_t* synapse, float dt) {
    /* Exponential decay of traces */
    synapse->pre_trace *= expf(-dt / synapse->tau_plus);
    synapse->post_trace *= expf(-dt / synapse->tau_minus);
}

/* ============================================================================
 * Classic STDP (No Neuromodulation)
 * ============================================================================ */

float stdp_pre_spike(stdp_synapse_t* synapse, float current_time) {
    (void)current_time;  /* Unused - kept for API consistency */

    /* Get sleep modulation factors */
    float lr_factor = stdp_sleep_get_lr_factor(synapse->current_sleep_state);
    float ratio_factor = stdp_sleep_get_ratio_factor(synapse->current_sleep_state);

    /* LTD: post trace indicates recent postsynaptic spike */
    /* Apply sleep modulation: LR factor and ratio factor (ratio affects A-) */
    float modulated_a_minus = synapse->a_minus / ratio_factor;  /* Lower ratio = higher A- */
    float weight_change = -modulated_a_minus * synapse->learning_rate * lr_factor * synapse->post_trace;

    /* Thread-safe weight modification */
    nimcp_spinlock_lock(&synapse->lock);

    /* P0 fix: Validate numerical stability before weight update
     * WHY:  NaN/Inf can propagate and corrupt weights
     */
    if (isnan(weight_change) || isinf(weight_change)) {
        nimcp_spinlock_unlock(&synapse->lock);
        return 0.0F;  /* Return zero to indicate no valid change */
    }

    if (weight_change < 0.0F) {
        synapse->num_depression_events++;
        synapse->total_ltd += fabsf(weight_change);
    }

    /* Apply weight change with atomic clamp (P0 fix: single operation to prevent torn reads)
     * WHY:  Separate update and clamp creates window for unclamped values to be observed
     */
    float new_weight = fmaxf(synapse->w_min, fminf(synapse->w_max, synapse->weight + weight_change));
    synapse->weight = new_weight;

    /* Increment presynaptic trace with upper bound to prevent unbounded growth
     * WHY:  Without clamping, rapid spike bursts can cause trace overflow
     * HOW:  Cap trace at 10.0 (typical biological saturation)
     */
    synapse->pre_trace = fminf(synapse->pre_trace + 1.0F, 10.0F);

    nimcp_spinlock_unlock(&synapse->lock);

    return weight_change;
}

float stdp_post_spike(stdp_synapse_t* synapse, float current_time) {
    (void)current_time;  /* Unused - kept for API consistency */

    /* Get sleep modulation factors */
    float lr_factor = stdp_sleep_get_lr_factor(synapse->current_sleep_state);
    float ratio_factor = stdp_sleep_get_ratio_factor(synapse->current_sleep_state);

    /* LTP: pre trace indicates recent presynaptic spike */
    /* Apply sleep modulation: LR factor and ratio factor (ratio affects A+) */
    float modulated_a_plus = synapse->a_plus * ratio_factor;  /* Higher ratio = higher A+ */
    float weight_change = modulated_a_plus * synapse->learning_rate * lr_factor * synapse->pre_trace;

    /* Thread-safe weight modification */
    nimcp_spinlock_lock(&synapse->lock);

    /* P0 fix: Validate numerical stability before weight update
     * WHY:  NaN/Inf can propagate and corrupt weights
     */
    if (isnan(weight_change) || isinf(weight_change)) {
        nimcp_spinlock_unlock(&synapse->lock);
        return 0.0F;  /* Return zero to indicate no valid change */
    }

    if (weight_change > 0.0F) {
        synapse->num_potentiation_events++;
        synapse->total_ltp += weight_change;
    }

    /* Apply weight change with atomic clamp (P0 fix: single operation to prevent torn reads)
     * WHY:  Separate update and clamp creates window for unclamped values to be observed
     */
    float new_weight = fmaxf(synapse->w_min, fminf(synapse->w_max, synapse->weight + weight_change));
    synapse->weight = new_weight;

    /* Increment postsynaptic trace with upper bound to prevent unbounded growth
     * WHY:  Without clamping, rapid spike bursts can cause trace overflow
     * HOW:  Cap trace at 10.0 (typical biological saturation)
     */
    synapse->post_trace = fminf(synapse->post_trace + 1.0F, 10.0F);

    nimcp_spinlock_unlock(&synapse->lock);

    return weight_change;
}

/* ============================================================================
 * Dopamine-Modulated STDP (THREE-FACTOR LEARNING)
 * ============================================================================ */

float stdp_get_da_modulation_factor(const stdp_synapse_t* synapse,
                                     neuromodulator_system_t neuromod) {
    if (!synapse->enable_da_modulation || neuromod == NULL) {
        return 1.0F;  /* No modulation */
    }

    /* Get dopamine concentration (tonic + phasic) */
    float da_concentration = neuromodulator_get_level(neuromod, NEUROMOD_DOPAMINE);

    /* Base modulation: DA concentration → LR multiplier */
    float modulation_factor = 1.0F + da_concentration * synapse->da_modulation_gain;

    /* Burst amplification: High DA concentration indicates burst */
    /* Baseline DA ≈ 0.05 in [0,1] range, burst ≈ 0.8+ */
    if (da_concentration > 0.3F) {  /* Above 30% = likely in burst */
        modulation_factor *= synapse->burst_amplification;
    }

    return modulation_factor;
}

float stdp_apply_modulated_weight_change(stdp_synapse_t* synapse,
                                         float base_weight_change,
                                         neuromodulator_system_t neuromod) {
    /* Get dopamine modulation factor */
    float modulation = stdp_get_da_modulation_factor(synapse, neuromod);

    /* Apply modulation */
    float modulated_weight_change = base_weight_change * modulation;

    /* Thread-safe weight modification */
    nimcp_spinlock_lock(&synapse->lock);

    /* P0 fix: Validate numerical stability before weight update
     * WHY:  NaN/Inf can propagate and corrupt weights
     */
    if (isnan(modulated_weight_change) || isinf(modulated_weight_change)) {
        nimcp_spinlock_unlock(&synapse->lock);
        return 0.0F;  /* Return zero to indicate no valid change */
    }

    /* Update statistics */
    if (modulated_weight_change > 0.0F) {
        synapse->num_potentiation_events++;
        synapse->total_ltp += modulated_weight_change;
    } else if (modulated_weight_change < 0.0F) {
        synapse->num_depression_events++;
        synapse->total_ltd += fabsf(modulated_weight_change);
    }

    /* Apply weight change with atomic clamp (P0 fix: single operation to prevent torn reads) */
    float new_weight = fmaxf(synapse->w_min, fminf(synapse->w_max, synapse->weight + modulated_weight_change));
    synapse->weight = new_weight;

    nimcp_spinlock_unlock(&synapse->lock);

    return modulated_weight_change;
}

float stdp_pre_spike_modulated(stdp_synapse_t* synapse,
                                float current_time,
                                neuromodulator_system_t neuromod) {
    (void)current_time;  /* Unused - kept for API consistency */

    /* Get sleep modulation factors */
    float lr_factor = stdp_sleep_get_lr_factor(synapse->current_sleep_state);
    float ratio_factor = stdp_sleep_get_ratio_factor(synapse->current_sleep_state);

    /* Compute base weight change (LTD from post trace) with sleep modulation */
    float modulated_a_minus = synapse->a_minus / ratio_factor;
    float base_weight_change = -modulated_a_minus * synapse->learning_rate * lr_factor * synapse->post_trace;

    /* Apply dopamine modulation */
    float modulated_weight_change = stdp_apply_modulated_weight_change(
        synapse, base_weight_change, neuromod);

    /* Increment presynaptic trace with upper bound to prevent unbounded growth
     * WHY:  Without clamping, rapid spike bursts can cause trace overflow
     * HOW:  Cap trace at 10.0 (typical biological saturation)
     */
    synapse->pre_trace = fminf(synapse->pre_trace + 1.0F, 10.0F);

    return modulated_weight_change;
}

float stdp_post_spike_modulated(stdp_synapse_t* synapse,
                                 float current_time,
                                 neuromodulator_system_t neuromod) {
    (void)current_time;  /* Unused - kept for API consistency */

    /* Get sleep modulation factors */
    float lr_factor = stdp_sleep_get_lr_factor(synapse->current_sleep_state);
    float ratio_factor = stdp_sleep_get_ratio_factor(synapse->current_sleep_state);

    /* Compute base weight change (LTP from pre trace) with sleep modulation */
    float modulated_a_plus = synapse->a_plus * ratio_factor;
    float base_weight_change = modulated_a_plus * synapse->learning_rate * lr_factor * synapse->pre_trace;

    /* Apply dopamine modulation */
    float modulated_weight_change = stdp_apply_modulated_weight_change(
        synapse, base_weight_change, neuromod);

    /* Increment postsynaptic trace with upper bound to prevent unbounded growth
     * WHY:  Without clamping, rapid spike bursts can cause trace overflow
     * HOW:  Cap trace at 10.0 (typical biological saturation)
     */
    synapse->post_trace = fminf(synapse->post_trace + 1.0F, 10.0F);

    return modulated_weight_change;
}

/* ============================================================================
 * Utilities
 * ============================================================================ */

void stdp_set_sleep_state(stdp_synapse_t* synapse, sleep_state_t state) {
    /* WHAT: Update sleep state for synapse
     * WHY:  Sleep state modulates learning rate, LTP/LTD ratio, timing windows
     * HOW:  Set current_sleep_state field, will be queried during next spike
     */
    if (!synapse) return;

    /* P2 fix: Validate sleep state bounds to prevent array index overflow
     * WHY:  Sleep bridge uses state as array index; invalid state = UB
     */
    if (state < SLEEP_STATE_AWAKE || state > SLEEP_STATE_REM) {
        LOG_WARN("Invalid sleep state %d, defaulting to AWAKE", (int)state);
        state = SLEEP_STATE_AWAKE;
    }
    synapse->current_sleep_state = state;
}

void stdp_synapse_reset(stdp_synapse_t* synapse) {
    synapse->pre_trace = 0.0F;
    synapse->post_trace = 0.0F;
    synapse->num_potentiation_events = 0;
    synapse->num_depression_events = 0;
    synapse->total_ltp = 0.0F;
    synapse->total_ltd = 0.0F;
}

void stdp_synapse_print_stats(const stdp_synapse_t* synapse) {
    printf("STDP Synapse Statistics:\n");
    printf("  Weight: %.4f / %.4f\n", synapse->weight, synapse->w_max);
    printf("  Potentiation events: %lu (total LTP: %.6f)\n",
           synapse->num_potentiation_events, synapse->total_ltp);
    printf("  Depression events: %lu (total LTD: %.6f)\n",
           synapse->num_depression_events, synapse->total_ltd);
    printf("  Net change: %.6f\n", synapse->total_ltp - synapse->total_ltd);
    printf("  Pre trace: %.6f, Post trace: %.6f\n",
           synapse->pre_trace, synapse->post_trace);
    printf("  DA modulation: %s (gain: %.1f, burst amp: %.1fx)\n",
           synapse->enable_da_modulation ? "ENABLED" : "DISABLED",
           synapse->da_modulation_gain, synapse->burst_amplification);
}
