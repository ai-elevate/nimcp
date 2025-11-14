/**
 * @file nimcp_stdp.c
 * @brief Implementation of dopamine-modulated STDP
 *
 * NIMCP Phase: Option 2.1 (Integration with Phase C2.2)
 * Date: 2025-11-12
 */

#include "plasticity/nimcp_stdp.h"
#include "plasticity/neuromodulators/nimcp_neuromodulators.h"
#include "plasticity/neuromodulators/nimcp_phasic_tonic.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

/* ============================================================================
 * Configuration
 * ============================================================================ */

stdp_config_t stdp_config_default(void) {
    stdp_config_t config = {
        .w_max = 1.0f,
        .learning_rate = 0.01f,
        .a_plus = 0.005f,           /* LTP amplitude (Bi & Poo, 1998) */
        .a_minus = 0.00525f,        /* LTD amplitude (slightly larger for balance) */
        .tau_plus = 0.020f,         /* 20 ms (seconds) */
        .tau_minus = 0.020f,        /* 20 ms */
        .enable_da_modulation = true,
        .da_modulation_gain = 100.0f,   /* DA concentration [0-1] → LR multiplier */
        .burst_amplification = 3.0f     /* 3x learning during bursts */
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
    memset(synapse, 0, sizeof(stdp_synapse_t));

    synapse->weight = 0.5f;  /* Initialize to 50% of max */
    synapse->w_max = config->w_max;
    synapse->w_min = 0.0f;

    synapse->learning_rate = config->learning_rate;
    synapse->a_plus = config->a_plus;
    synapse->a_minus = config->a_minus;
    synapse->tau_plus = config->tau_plus;
    synapse->tau_minus = config->tau_minus;

    synapse->enable_da_modulation = config->enable_da_modulation;
    synapse->da_modulation_gain = config->da_modulation_gain;
    synapse->burst_amplification = config->burst_amplification;

    synapse->pre_trace = 0.0f;
    synapse->post_trace = 0.0f;

    synapse->num_potentiation_events = 0;
    synapse->num_depression_events = 0;
    synapse->total_ltp = 0.0f;
    synapse->total_ltd = 0.0f;
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

    /* LTD: post trace indicates recent postsynaptic spike */
    float weight_change = -synapse->a_minus * synapse->learning_rate * synapse->post_trace;

    if (weight_change < 0.0f) {
        synapse->num_depression_events++;
        synapse->total_ltd += fabsf(weight_change);
    }

    /* Apply weight change */
    synapse->weight += weight_change;
    if (synapse->weight < synapse->w_min) synapse->weight = synapse->w_min;
    if (synapse->weight > synapse->w_max) synapse->weight = synapse->w_max;

    /* Increment presynaptic trace */
    synapse->pre_trace += 1.0f;

    return weight_change;
}

float stdp_post_spike(stdp_synapse_t* synapse, float current_time) {
    (void)current_time;  /* Unused - kept for API consistency */

    /* LTP: pre trace indicates recent presynaptic spike */
    float weight_change = synapse->a_plus * synapse->learning_rate * synapse->pre_trace;

    if (weight_change > 0.0f) {
        synapse->num_potentiation_events++;
        synapse->total_ltp += weight_change;
    }

    /* Apply weight change */
    synapse->weight += weight_change;
    if (synapse->weight < synapse->w_min) synapse->weight = synapse->w_min;
    if (synapse->weight > synapse->w_max) synapse->weight = synapse->w_max;

    /* Increment postsynaptic trace */
    synapse->post_trace += 1.0f;

    return weight_change;
}

/* ============================================================================
 * Dopamine-Modulated STDP (THREE-FACTOR LEARNING)
 * ============================================================================ */

float stdp_get_da_modulation_factor(const stdp_synapse_t* synapse,
                                     neuromodulator_system_t neuromod) {
    if (!synapse->enable_da_modulation || neuromod == NULL) {
        return 1.0f;  /* No modulation */
    }

    /* Get dopamine concentration (tonic + phasic) */
    float da_concentration = neuromodulator_get_level(neuromod, NEUROMOD_DOPAMINE);

    /* Base modulation: DA concentration → LR multiplier */
    float modulation_factor = 1.0f + da_concentration * synapse->da_modulation_gain;

    /* Burst amplification: High DA concentration indicates burst */
    /* Baseline DA ≈ 0.05 in [0,1] range, burst ≈ 0.8+ */
    if (da_concentration > 0.3f) {  /* Above 30% = likely in burst */
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

    /* Update statistics */
    if (modulated_weight_change > 0.0f) {
        synapse->num_potentiation_events++;
        synapse->total_ltp += modulated_weight_change;
    } else if (modulated_weight_change < 0.0f) {
        synapse->num_depression_events++;
        synapse->total_ltd += fabsf(modulated_weight_change);
    }

    /* Apply weight change */
    synapse->weight += modulated_weight_change;

    /* Clamp to [w_min, w_max] */
    if (synapse->weight < synapse->w_min) synapse->weight = synapse->w_min;
    if (synapse->weight > synapse->w_max) synapse->weight = synapse->w_max;

    return modulated_weight_change;
}

float stdp_pre_spike_modulated(stdp_synapse_t* synapse,
                                float current_time,
                                neuromodulator_system_t neuromod) {
    (void)current_time;  /* Unused - kept for API consistency */

    /* Compute base weight change (LTD from post trace) */
    float base_weight_change = -synapse->a_minus * synapse->learning_rate * synapse->post_trace;

    /* Apply dopamine modulation */
    float modulated_weight_change = stdp_apply_modulated_weight_change(
        synapse, base_weight_change, neuromod);

    /* Increment presynaptic trace */
    synapse->pre_trace += 1.0f;

    return modulated_weight_change;
}

float stdp_post_spike_modulated(stdp_synapse_t* synapse,
                                 float current_time,
                                 neuromodulator_system_t neuromod) {
    (void)current_time;  /* Unused - kept for API consistency */

    /* Compute base weight change (LTP from pre trace) */
    float base_weight_change = synapse->a_plus * synapse->learning_rate * synapse->pre_trace;

    /* Apply dopamine modulation */
    float modulated_weight_change = stdp_apply_modulated_weight_change(
        synapse, base_weight_change, neuromod);

    /* Increment postsynaptic trace */
    synapse->post_trace += 1.0f;

    return modulated_weight_change;
}

/* ============================================================================
 * Utilities
 * ============================================================================ */

void stdp_synapse_reset(stdp_synapse_t* synapse) {
    synapse->pre_trace = 0.0f;
    synapse->post_trace = 0.0f;
    synapse->num_potentiation_events = 0;
    synapse->num_depression_events = 0;
    synapse->total_ltp = 0.0f;
    synapse->total_ltd = 0.0f;
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
