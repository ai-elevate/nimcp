//=============================================================================
// nimcp_plasticity_ternary.h - Ternary Plasticity Updates
//=============================================================================
/**
 * @file nimcp_plasticity_ternary.h
 * @brief Ternary representation for synaptic plasticity
 *
 * WHAT: Three-state plasticity {LTD, STABLE, LTP}
 * WHY:  Discrete plasticity updates for efficient hardware/simulation
 * HOW:  Map continuous STDP/BCM to ternary state transitions
 *
 * BIOLOGICAL BASIS:
 * - Synaptic weights change in discrete steps at molecular level
 * - LTP: Long-term potentiation (strengthening)
 * - LTD: Long-term depression (weakening)
 * - Stable: Maintenance of current weight (homeostasis)
 *
 * INTEGRATION WITH PLASTICITY RULES:
 * - STDP: Pre-post timing → LTP/LTD decision
 * - BCM: Post > threshold → LTP, Post < threshold → LTD
 * - Homeostatic: Nudge toward target firing rate
 *
 * @author NIMCP Development Team
 * @date 2025-12-21
 */

#ifndef NIMCP_PLASTICITY_TERNARY_H
#define NIMCP_PLASTICITY_TERNARY_H

#include "utils/ternary/nimcp_ternary.h"

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// Plasticity Ternary Types
//=============================================================================

/**
 * @brief Ternary plasticity direction
 */
typedef trit_t trit_plasticity_t;

#define TRIT_PLASTICITY_LTD     TRIT_NEGATIVE  /**< Long-term depression */
#define TRIT_PLASTICITY_STABLE  TRIT_UNKNOWN   /**< No change */
#define TRIT_PLASTICITY_LTP     TRIT_POSITIVE  /**< Long-term potentiation */

//=============================================================================
// STDP Ternary Functions
//=============================================================================

/**
 * @brief Compute ternary STDP update from spike timing
 *
 * WHAT: Determine LTP/LTD from pre-post timing difference
 * WHY:  Discretize STDP for ternary weight updates
 * HOW:  Standard STDP window with dead zone
 *
 * STDP RULE:
 * - dt = t_post - t_pre
 * - dt > 0 (causal): LTP (pre fires before post)
 * - dt < 0 (anti-causal): LTD (post fires before pre)
 * - |dt| too small or large: STABLE
 *
 * @param dt_ms Time difference (t_post - t_pre) in ms
 * @param tau_plus LTP time constant (ms)
 * @param tau_minus LTD time constant (ms)
 * @param dead_zone Minimum |dt| for plasticity
 * @return Ternary plasticity direction
 */
static inline trit_plasticity_t trit_stdp_compute(
    float dt_ms,
    float tau_plus,
    float tau_minus,
    float dead_zone
) {
    float abs_dt = (dt_ms >= 0) ? dt_ms : -dt_ms;

    /* Dead zone: too close in time */
    if (abs_dt < dead_zone) return TRIT_PLASTICITY_STABLE;

    /* Too far apart: outside STDP window */
    float cutoff = 5.0f * ((tau_plus > tau_minus) ? tau_plus : tau_minus);
    if (abs_dt > cutoff) return TRIT_PLASTICITY_STABLE;

    /* Causal: LTP */
    if (dt_ms > 0) return TRIT_PLASTICITY_LTP;

    /* Anti-causal: LTD */
    return TRIT_PLASTICITY_LTD;
}

/**
 * @brief Compute ternary STDP with magnitude threshold
 *
 * @param delta Continuous STDP delta value
 * @param threshold Minimum magnitude for change
 * @return Ternary plasticity direction
 */
static inline trit_plasticity_t trit_stdp_from_delta(
    float delta,
    float threshold
) {
    if (delta >= threshold) return TRIT_PLASTICITY_LTP;
    if (delta <= -threshold) return TRIT_PLASTICITY_LTD;
    return TRIT_PLASTICITY_STABLE;
}

//=============================================================================
// BCM Ternary Functions
//=============================================================================

/**
 * @brief Compute ternary BCM update
 *
 * WHAT: Bienenstock-Cooper-Munro plasticity rule
 * WHY:  Rate-based plasticity with sliding threshold
 * HOW:  Compare postsynaptic activity to threshold
 *
 * BCM RULE:
 * - activity > threshold: LTP
 * - activity < threshold: LTD
 * - activity ≈ threshold: STABLE
 *
 * @param post_activity Postsynaptic firing rate/activity
 * @param theta_m Modification threshold
 * @param dead_zone Width of STABLE region around theta
 * @return Ternary plasticity direction
 */
static inline trit_plasticity_t trit_bcm_compute(
    float post_activity,
    float theta_m,
    float dead_zone
) {
    float diff = post_activity - theta_m;

    if (diff > dead_zone) return TRIT_PLASTICITY_LTP;
    if (diff < -dead_zone) return TRIT_PLASTICITY_LTD;
    return TRIT_PLASTICITY_STABLE;
}

//=============================================================================
// Homeostatic Ternary Functions
//=============================================================================

/**
 * @brief Compute ternary homeostatic update
 *
 * WHAT: Nudge weights to restore target firing rate
 * WHY:  Maintain network stability
 * HOW:  Compare current rate to target
 *
 * @param current_rate Current firing rate
 * @param target_rate Target firing rate
 * @param tolerance Rate difference tolerance
 * @return Ternary plasticity direction
 */
static inline trit_plasticity_t trit_homeostatic_compute(
    float current_rate,
    float target_rate,
    float tolerance
) {
    float diff = current_rate - target_rate;

    /* Rate too high: decrease weights (LTD) */
    if (diff > tolerance) return TRIT_PLASTICITY_LTD;

    /* Rate too low: increase weights (LTP) */
    if (diff < -tolerance) return TRIT_PLASTICITY_LTP;

    /* Rate in acceptable range */
    return TRIT_PLASTICITY_STABLE;
}

//=============================================================================
// Combined Plasticity Rules
//=============================================================================

/**
 * @brief Combine multiple plasticity signals
 *
 * WHAT: Aggregate STDP, BCM, homeostatic signals
 * WHY:  Multiple rules may act on same synapse
 * HOW:  Voting with configurable weights
 *
 * @param signals Array of plasticity signals
 * @param weights Array of signal weights
 * @param n_signals Number of signals
 * @param threshold Weighted sum threshold
 * @return Combined ternary direction
 */
static inline trit_plasticity_t trit_plasticity_combine(
    const trit_plasticity_t* signals,
    const float* weights,
    size_t n_signals,
    float threshold
) {
    if (!signals || n_signals == 0) return TRIT_PLASTICITY_STABLE;

    float weighted_sum = 0.0f;

    for (size_t i = 0; i < n_signals; i++) {
        float w = (weights) ? weights[i] : 1.0f;
        weighted_sum += (float)signals[i] * w;
    }

    if (weighted_sum >= threshold) return TRIT_PLASTICITY_LTP;
    if (weighted_sum <= -threshold) return TRIT_PLASTICITY_LTD;
    return TRIT_PLASTICITY_STABLE;
}

/**
 * @brief Apply neuromodulator gating to plasticity
 *
 * WHAT: Modulate plasticity based on neuromodulator state
 * WHY:  Dopamine, ACh, etc. gate learning
 * HOW:  Multiply plasticity by modulator level, then discretize
 *
 * @param base_plasticity Base ternary plasticity
 * @param modulator_level Neuromodulator level [0, 1]
 * @param gate_threshold Threshold for gating
 * @return Gated plasticity (may become STABLE if modulator low)
 */
static inline trit_plasticity_t trit_plasticity_gate(
    trit_plasticity_t base_plasticity,
    float modulator_level,
    float gate_threshold
) {
    /* Low modulator blocks all plasticity */
    if (modulator_level < gate_threshold) return TRIT_PLASTICITY_STABLE;

    /* Otherwise pass through */
    return base_plasticity;
}

//=============================================================================
// Statistics
//=============================================================================

/**
 * @brief Plasticity event statistics
 */
typedef struct {
    uint64_t n_ltp;      /**< Count of LTP events */
    uint64_t n_stable;   /**< Count of STABLE events */
    uint64_t n_ltd;      /**< Count of LTD events */
    float ltp_ratio;     /**< Fraction of LTP */
    float ltd_ratio;     /**< Fraction of LTD */
} trit_plasticity_stats_t;

/**
 * @brief Count plasticity events
 *
 * @param events Array of plasticity events
 * @param n_events Number of events
 * @param stats Output statistics
 */
static inline void trit_plasticity_count(
    const trit_plasticity_t* events,
    size_t n_events,
    trit_plasticity_stats_t* stats
) {
    if (!stats) return;

    stats->n_ltp = 0;
    stats->n_stable = 0;
    stats->n_ltd = 0;

    if (!events || n_events == 0) {
        stats->ltp_ratio = 0.0f;
        stats->ltd_ratio = 0.0f;
        return;
    }

    for (size_t i = 0; i < n_events; i++) {
        if (events[i] == TRIT_PLASTICITY_LTP) stats->n_ltp++;
        else if (events[i] == TRIT_PLASTICITY_LTD) stats->n_ltd++;
        else stats->n_stable++;
    }

    stats->ltp_ratio = (float)stats->n_ltp / (float)n_events;
    stats->ltd_ratio = (float)stats->n_ltd / (float)n_events;
}

/**
 * @brief Get plasticity name string
 *
 * @param p Plasticity direction
 * @return String name
 */
static inline const char* trit_plasticity_name(trit_plasticity_t p) {
    switch (p) {
        case TRIT_PLASTICITY_LTD:    return "LTD";
        case TRIT_PLASTICITY_STABLE: return "STABLE";
        case TRIT_PLASTICITY_LTP:    return "LTP";
        default:                      return "INVALID";
    }
}

#ifdef __cplusplus
}
#endif

#endif /* NIMCP_PLASTICITY_TERNARY_H */
