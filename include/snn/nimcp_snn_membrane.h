/**
 * @file nimcp_snn_membrane.h
 * @brief SNN membrane integration — current-mode and conductance-mode kernels.
 *
 * Single-responsibility module owning the LIF membrane equation. Both the
 * legacy current-based and the new conductance-based paths share the same
 * voltage clamp, NaN guards, and integration scheme — this module is the
 * single source of truth for the math.
 *
 * The functions are header-inline `static inline` so the SNN hot loop pays
 * no call overhead. Pure functions of their inputs (no globals, no I/O) so
 * they are trivially unit-testable and thread-safe.
 *
 * See docs/claude/cb-phase0-design.md for the full migration design.
 */

#ifndef NIMCP_SNN_MEMBRANE_H
#define NIMCP_SNN_MEMBRANE_H

#include <stdbool.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Compute one timestep of LIF membrane voltage change (dv).
 *
 * Selects current- or conductance-based dynamics by the @p conductance_mode
 * flag. The same equation tail (leak term + integration / tau_eff * dt) runs
 * in both branches; only the synaptic-drive term differs.
 *
 * Current mode:    dv = ( (v_rest - v) + i_syn )            / tau_eff * dt
 * Conductance:     dv = ( (v_rest - v) + g_e*(E_e - v)
 *                                      + g_i*(E_i - v) )    / tau_eff * dt
 *
 * @param v               Current membrane voltage (mV)
 * @param v_rest          Resting potential (mV)
 * @param tau_eff_ms      Effective membrane time constant (ms; > 0)
 * @param dt_ms           Integration timestep (ms; > 0)
 * @param i_syn           Total synaptic current (mV-equivalent; current mode only)
 * @param g_exc           Excitatory conductance (dimensionless; CB mode only)
 * @param g_inh           Inhibitory conductance (dimensionless; CB mode only)
 * @param e_exc           Excitatory reversal potential (mV; CB mode only)
 * @param e_inh           Inhibitory reversal potential (mV; CB mode only)
 * @param conductance_mode Select CB (true) or legacy current (false)
 * @return dv (mV)
 *
 * Pure function. No NaN/Inf produced for finite inputs with tau_eff > 0.
 */
static inline float snn_membrane_compute_dv(
    float v, float v_rest,
    float tau_eff_ms, float dt_ms,
    float i_syn,
    float g_exc, float g_inh,
    float e_exc, float e_inh,
    bool conductance_mode)
{
    /* Defensive: tau_eff is bounded > 0 by all upstream callers (snn_step
     * computes it from a strictly-positive substrate-modulated tau_mem).
     * If a future caller passes a non-positive tau_eff we'd get inf/-inf;
     * clamp to 1e-3 to keep the integration stable. */
    if (tau_eff_ms < 1e-3f) tau_eff_ms = 1e-3f;

    const float leak = (v_rest - v);
    float drive;
    if (conductance_mode) {
        drive = leak
              + g_exc * (e_exc - v)
              + g_inh * (e_inh - v);
    } else {
        drive = leak + i_syn;
    }
    float dv = drive * (dt_ms / tau_eff_ms);

    /* Biophysical clamp: a single integration step should not move the
     * membrane by more than ±100 mV. Real cortical neurons swing
     * 60–80 mV across an action potential cycle; values larger than
     * 100 mV per dt indicate pathological tau collapse, runaway g_exc,
     * or an unrescaled-weight CB run. Clamping prevents the LIF state
     * from blowing up (NaN propagation, threshold-crossing storms,
     * GPU sync failures) while leaving normal dynamics untouched
     * (typical |dv| << 5 mV per step at dt=1 ms). */
    if (dv >  100.0f) dv =  100.0f;
    if (dv < -100.0f) dv = -100.0f;

    /* CB-mode reversal-potential bound: enforce the key conductance-based
     * invariant — V cannot cross a reversal potential, regardless of how
     * the dv clamp above interacted with extreme g_exc/g_inh. Without
     * this, a very large g_exc with the ±100 mV dv clamp could let V
     * leapfrog over E_exc on a single step (the conductance "should"
     * have produced a larger dv that, after clamp, no longer respects
     * the saturation). We cap dv so that v + dv stays within
     * [E_inh, E_exc] (with a tiny slop for FP rounding). */
    if (conductance_mode) {
        const float v_after = v + dv;
        if (v_after > e_exc) dv = e_exc - v;
        if (v_after < e_inh) dv = e_inh - v;
    }
    return dv;
}

/**
 * @brief Decay one neuron's conductances by precomputed factors.
 *
 * Caller computes @p decay_exc = expf(-dt/tau_exc) and @p decay_inh
 * once per population step (NOT per neuron) to avoid expf() in the
 * inner loop — this matches the existing `dep_decay` pattern in
 * nimcp_snn_network.c:1504.
 *
 * @param g_exc       Pointer to one neuron's excitatory conductance (in/out)
 * @param g_inh       Pointer to one neuron's inhibitory conductance (in/out)
 * @param decay_exc   Excitatory decay factor (e.g., expf(-dt/tau_exc))
 * @param decay_inh   Inhibitory decay factor
 *
 * Both pointers may be NULL — function no-ops, matching the pop-level
 * "alloc failed → silent degrade" contract from snn_population_create.
 */
static inline void snn_membrane_decay_one(
    float* g_exc, float* g_inh,
    float decay_exc, float decay_inh)
{
    if (g_exc) *g_exc *= decay_exc;
    if (g_inh) *g_inh *= decay_inh;
}

/**
 * @brief Deposit a synaptic weight into either I_syn or g_exc/g_inh
 *        depending on @p conductance_mode and @p weight sign.
 *
 * In current mode, all weights add to *i_syn (sign-preserving).
 * In CB mode, positive weights add to *g_exc; negative weights add
 * |weight| to *g_inh. Receptor type is derived from sign — no
 * struct-field change needed for the synapse itself.
 *
 * @param i_syn            In/out: current-mode accumulator (may be NULL in CB mode)
 * @param g_exc            In/out: excitatory conductance for one neuron (may be NULL)
 * @param g_inh            In/out: inhibitory conductance for one neuron (may be NULL)
 * @param weight           Effective weight (already depression-modulated by caller)
 * @param conductance_mode Select CB (true) or legacy current (false)
 *
 * Branch-free path is hoisted by the compiler when @p conductance_mode is a
 * loop-invariant captured before the inner loop (LICM). Do not pass it
 * per-synapse from a global getter.
 */
static inline void snn_membrane_deposit_synapse(
    float* i_syn,
    float* g_exc, float* g_inh,
    float weight,
    bool conductance_mode)
{
    if (conductance_mode) {
        if (weight >= 0.0f) {
            if (g_exc) *g_exc += weight;
        } else {
            if (g_inh) *g_inh += -weight;
        }
    } else {
        if (i_syn) *i_syn += weight;
    }
}

#ifdef __cplusplus
}
#endif

#endif /* NIMCP_SNN_MEMBRANE_H */
