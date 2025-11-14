/**
 * @file nimcp_stdp.h
 * @brief Spike-Timing-Dependent Plasticity with Dopamine Modulation
 *
 * Implementation of STDP with neuromodulator-gated learning, integrating
 * Phase C2.2 phasic-tonic dopamine dynamics for biologically realistic
 * reward learning.
 *
 * Key Features:
 * - Classic STDP (Bi & Poo, 1998): LTP for pre-before-post, LTD for post-before-pre
 * - Dopamine modulation: Learning rate scaled by DA concentration
 * - Burst amplification: 3x learning during phasic dopamine bursts
 * - Three-factor learning: Hebbian + Timing + Reward
 *
 * NIMCP Phase: Option 2.1 (Integration)
 * Date: 2025-11-12
 */

#ifndef NIMCP_STDP_H
#define NIMCP_STDP_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Forward declarations */
struct neuromodulator_system_struct;
typedef struct neuromodulator_system_struct neuromodulator_system_t;

/**
 * STDP synapse state
 */
typedef struct {
    /* Synaptic weight */
    float weight;               /* Current weight [0, w_max] */
    float w_max;                /* Maximum weight (default: 1.0) */
    float w_min;                /* Minimum weight (default: 0.0) */

    /* Learning parameters */
    float learning_rate;        /* Base learning rate (default: 0.01) */
    float a_plus;               /* LTP amplitude (default: 0.005) */
    float a_minus;              /* LTD amplitude (default: 0.00525) */
    float tau_plus;             /* LTP time constant [ms] (default: 20) */
    float tau_minus;            /* LTD time constant [ms] (default: 20) */

    /* Spike timing traces */
    float pre_trace;            /* Presynaptic trace */
    float post_trace;           /* Postsynaptic trace */

    /* Dopamine modulation parameters */
    bool enable_da_modulation;  /* Use dopamine modulation (default: true) */
    float da_modulation_gain;   /* DA concentration → LR scaling (default: 100.0) */
    float burst_amplification;  /* LR multiplier during bursts (default: 3.0) */

    /* Statistics */
    uint64_t num_potentiation_events;
    uint64_t num_depression_events;
    float total_ltp;            /* Cumulative LTP */
    float total_ltd;            /* Cumulative LTD */

} stdp_synapse_t;

/**
 * STDP configuration
 */
typedef struct {
    float w_max;
    float learning_rate;
    float a_plus;
    float a_minus;
    float tau_plus;
    float tau_minus;
    bool enable_da_modulation;
    float da_modulation_gain;
    float burst_amplification;
} stdp_config_t;

/**
 * Initialize STDP synapse with default parameters
 *
 * @param synapse Synapse to initialize
 */
void stdp_synapse_init(stdp_synapse_t* synapse);

/**
 * Initialize STDP synapse with custom configuration
 *
 * @param synapse Synapse to initialize
 * @param config Configuration parameters
 */
void stdp_synapse_init_with_config(stdp_synapse_t* synapse, const stdp_config_t* config);

/**
 * Get default STDP configuration
 *
 * @return Default configuration (Bi & Poo parameters)
 */
stdp_config_t stdp_config_default(void);

/**
 * Update STDP traces (called every timestep)
 *
 * @param synapse Synapse to update
 * @param dt Time step [seconds]
 */
void stdp_update_traces(stdp_synapse_t* synapse, float dt);

/**
 * Process presynaptic spike (classic STDP, no neuromodulation)
 *
 * @param synapse Synapse to update
 * @param current_time Current time [ms]
 * @return Weight change applied
 */
float stdp_pre_spike(stdp_synapse_t* synapse, float current_time);

/**
 * Process postsynaptic spike (classic STDP, no neuromodulation)
 *
 * @param synapse Synapse to update
 * @param current_time Current time [ms]
 * @return Weight change applied
 */
float stdp_post_spike(stdp_synapse_t* synapse, float current_time);

/**
 * Process presynaptic spike with dopamine modulation (THREE-FACTOR LEARNING)
 *
 * This is the key function for reward-modulated learning. The weight change
 * is modulated by:
 * 1. Dopamine concentration (tonic + phasic)
 * 2. Burst state (3x amplification during bursts)
 *
 * @param synapse Synapse to update
 * @param current_time Current time [ms]
 * @param neuromod Neuromodulator system (for dopamine level)
 * @return Weight change applied
 */
float stdp_pre_spike_modulated(stdp_synapse_t* synapse,
                                float current_time,
                                neuromodulator_system_t* neuromod);

/**
 * Process postsynaptic spike with dopamine modulation
 *
 * @param synapse Synapse to update
 * @param current_time Current time [ms]
 * @param neuromod Neuromodulator system (for dopamine level)
 * @return Weight change applied
 */
float stdp_post_spike_modulated(stdp_synapse_t* synapse,
                                 float current_time,
                                 neuromodulator_system_t* neuromod);

/**
 * Apply weight change with dopamine modulation
 *
 * Internal function that applies DA-gated learning. Called by
 * stdp_pre_spike_modulated() and stdp_post_spike_modulated().
 *
 * @param synapse Synapse to update
 * @param base_weight_change Unmodulated weight change (from STDP rule)
 * @param neuromod Neuromodulator system
 * @return Actual weight change applied (after modulation)
 */
float stdp_apply_modulated_weight_change(stdp_synapse_t* synapse,
                                         float base_weight_change,
                                         neuromodulator_system_t* neuromod);

/**
 * Get current dopamine modulation factor
 *
 * Computes the multiplicative factor applied to learning rate based on:
 * - Dopamine concentration (tonic + phasic)
 * - Burst state (amplification if bursting)
 *
 * @param synapse Synapse configuration
 * @param neuromod Neuromodulator system
 * @return Modulation factor [0, 10+] (typically 1.0 baseline, 5.0+ during bursts)
 */
float stdp_get_da_modulation_factor(const stdp_synapse_t* synapse,
                                     neuromodulator_system_t* neuromod);

/**
 * Reset STDP synapse (clear traces and statistics)
 *
 * @param synapse Synapse to reset
 */
void stdp_synapse_reset(stdp_synapse_t* synapse);

/**
 * Print STDP synapse statistics
 *
 * @param synapse Synapse to print
 */
void stdp_synapse_print_stats(const stdp_synapse_t* synapse);

#ifdef __cplusplus
}
#endif

#endif /* NIMCP_STDP_H */
