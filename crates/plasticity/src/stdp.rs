//! Spike-timing-dependent plasticity (STDP).
//!
//! # Math
//!
//! Let `Δt = post_spike_time − pre_spike_time`.
//! - If `Δt > 0` (pre leads post — causal):   `Δw = +A_plus  · exp(−Δt / τ_plus)`
//! - If `Δt < 0` (post leads pre — anti-causal): `Δw = −A_minus · exp(+Δt / τ_minus)`
//! - If `Δt == 0`: `Δw = 0` (neutral; symmetric kernels disagree on sign).
//!
//! The exponential kernels are the standard Bi & Poo (1998) / Song, Miller,
//! Abbott (2000) form. All constants are positive; the minus-branch sign is
//! encoded in the equation above, not in `A_minus`.

use serde::{Deserialize, Serialize};

/// STDP kernel parameters.
///
/// All magnitudes are positive. The sign of the weight change is determined
/// by the kernel branch selected from `Δt`, not by these fields.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct StdpParams {
    /// Potentiation amplitude (`A⁺`). Must be `≥ 0`.
    pub a_plus: f32,
    /// Depression amplitude (`A⁻`). Must be `≥ 0`.
    pub a_minus: f32,
    /// Potentiation time constant `τ⁺` in ms. Must be `> 0`.
    pub tau_plus_ms: f32,
    /// Depression time constant `τ⁻` in ms. Must be `> 0`.
    pub tau_minus_ms: f32,
}

impl Default for StdpParams {
    /// Biologically plausible defaults (Song et al. 2000): small-amplitude,
    /// slightly asymmetric kernel favoring depression.
    fn default() -> Self {
        Self {
            a_plus: 0.01,
            a_minus: 0.012,
            tau_plus_ms: 20.0,
            tau_minus_ms: 20.0,
        }
    }
}

/// Compute the STDP weight delta for a pre/post spike pair.
///
/// `Δw = +A_plus · exp(−Δt/τ_plus)` when pre leads post (causal), negative and
/// symmetric when post leads pre (anti-causal). Simultaneous spikes (`Δt = 0`)
/// yield `Δw = 0`.
///
/// # Panics
///
/// Does not panic on malformed params — `τ ≤ 0` short-circuits to `0.0`.
/// We do not panic in plasticity kernels; callers are hot-path code.
#[must_use]
pub fn stdp_weight_delta(
    pre_spike_time_ms: f32,
    post_spike_time_ms: f32,
    params: &StdpParams,
) -> f32 {
    // Defensive: malformed τ should not crash the hot path.
    if params.tau_plus_ms <= 0.0 || params.tau_minus_ms <= 0.0 {
        return 0.0;
    }

    let dt = post_spike_time_ms - pre_spike_time_ms;
    if dt > 0.0 {
        params.a_plus * (-dt / params.tau_plus_ms).exp()
    } else if dt < 0.0 {
        // dt is negative here, so (dt / τ⁻) is negative → exp() ∈ (0, 1).
        -params.a_minus * (dt / params.tau_minus_ms).exp()
    } else {
        0.0
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)] // exact-equality asserts are deliberate here
mod tests {
    use super::*;

    /// Regression: pre leading post by 5ms must strengthen (positive Δw).
    #[test]
    fn pre_leads_post_potentiates() {
        let params = StdpParams::default();
        let dw = stdp_weight_delta(0.0, 5.0, &params);
        assert!(dw > 0.0, "pre→post (5ms) should potentiate, got Δw = {dw}");
    }

    /// Regression: post leading pre by 5ms must weaken (negative Δw).
    #[test]
    fn post_leads_pre_depresses() {
        let params = StdpParams::default();
        let dw = stdp_weight_delta(5.0, 0.0, &params);
        assert!(dw < 0.0, "post→pre (5ms) should depress, got Δw = {dw}");
    }

    /// Simultaneous spikes produce no change.
    #[test]
    fn simultaneous_spikes_no_change() {
        let params = StdpParams::default();
        let dw = stdp_weight_delta(3.5, 3.5, &params);
        assert_eq!(dw, 0.0);
    }

    /// Exponential decay: |Δw| shrinks as |Δt| grows.
    #[test]
    fn magnitude_decays_with_time_difference() {
        let params = StdpParams::default();
        let dw_near = stdp_weight_delta(0.0, 5.0, &params);
        let dw_far = stdp_weight_delta(0.0, 50.0, &params);
        assert!(dw_near > dw_far);
        assert!(dw_far > 0.0);
    }

    /// Exact arithmetic at `Δt = τ`: `Δw = A · exp(−1)`.
    #[test]
    fn kernel_matches_analytic_formula() {
        let params = StdpParams::default();
        let dw = stdp_weight_delta(0.0, params.tau_plus_ms, &params);
        let expected = params.a_plus * (-1.0_f32).exp();
        assert!((dw - expected).abs() < 1e-6);
    }

    /// Malformed τ must not panic — return 0 and move on.
    #[test]
    fn zero_tau_returns_zero() {
        let params = StdpParams {
            a_plus: 1.0,
            a_minus: 1.0,
            tau_plus_ms: 0.0,
            tau_minus_ms: 0.0,
        };
        assert_eq!(stdp_weight_delta(0.0, 5.0, &params), 0.0);
        assert_eq!(stdp_weight_delta(5.0, 0.0, &params), 0.0);
    }
}
