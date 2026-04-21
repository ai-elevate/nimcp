//! NIMCP V2 — plasticity rules.
//!
//! Each rule is a pure function of (pre-activity, post-activity, weights,
//! neuromodulators) → weight deltas. The equations are well-understood;
//! the V1 bugs were mostly in how they were wired, not in the math.
//!
//! # Rules (ported from V1, each in its own module for unit testability)
//!
//! - [`stdp`]: spike-timing-dependent plasticity
//! - [`homeostatic`]: synaptic scaling with tight `[0.98, 1.02]` bounds
//! - [`quiet_start`]: load-time weight transform for saturation recovery
//! - [`warmup`]: warmup gate for rate-dependent rules
//! - [`bcm`]: Bienenstock–Cooper–Munro rate-based rule
//!
//! # Regression tests shipped from day one
//!
//! These regressions encode V1 bugs we never want to see again:
//!
//! - Homeostatic **tight bounds** `[0.98, 1.02]` are the default; a unit
//!   test asserts exact-clamp behavior on saturated rates.
//! - Homeostatic **anti-oscillation**: a 100-step simulation would fail
//!   under V1's old `[0.90, 1.10]` bounds.
//! - R-STDP **warmup gate** is exposed as an explicit helper
//!   (`rate_samples_ready`). Callers can't forget it.
//! - **Quiet-start as a transform**, not a code path — the SNN crate in
//!   Phase 3 will call it at load time as well as at init.
//!
//! # Conventions
//!
//! - All functions are pure. No actors, no async, no shared state.
//! - All `*Params` structs implement `Default` with biologically plausible
//!   defaults. Callers override explicitly; V2 has no silent magic numbers.
//! - No panics in hot paths. Malformed inputs (e.g. `τ ≤ 0`) short-circuit
//!   to a neutral result (`Δw = 0`, `scale = 1.0`).

#![forbid(unsafe_code)]

pub mod bcm;
pub mod homeostatic;
pub mod quiet_start;
pub mod stdp;
pub mod warmup;

// Flatten the common surface for callers that just want the functions.
pub use bcm::bcm_weight_delta;
pub use homeostatic::{HomeostaticParams, homeostatic_scale};
pub use quiet_start::{QuietStartBand, quiet_start_scale, quiet_start_scale_with_band};
pub use stdp::{StdpParams, stdp_weight_delta};
pub use warmup::{DEFAULT_WARMUP_THRESHOLD, rate_samples_ready};

#[cfg(test)]
mod integration_tests {
    //! Cross-module regression tests covering the V1 "bad state → bad state"
    //! failure modes. These are the tests that would have caught the real
    //! V1 bugs before they shipped.

    use super::*;

    /// A fresh population + R-STDP-style rule: warmup gate must keep us
    /// from applying STDP updates before enough samples exist.
    #[test]
    fn warmup_gate_blocks_early_stdp() {
        let samples_so_far = 50_u32;
        let ready = rate_samples_ready(samples_so_far, DEFAULT_WARMUP_THRESHOLD);
        assert!(!ready, "R-STDP should be blocked during warmup");

        // After enough samples, the gate opens and STDP proceeds.
        let ready = rate_samples_ready(100, DEFAULT_WARMUP_THRESHOLD);
        assert!(ready);
        let dw = stdp_weight_delta(0.0, 5.0, &StdpParams::default());
        assert!(dw > 0.0);
    }

    /// End-to-end recovery simulation: load a saturated checkpoint, apply
    /// `quiet_start_scale` once, then run homeostatic scaling. Within a
    /// bounded number of steps, the pop should land near target.
    ///
    /// This is the V1 "saturated-checkpoint resume trap" test.
    #[test]
    fn saturated_checkpoint_recovers_after_quiet_start() {
        let target = 0.03_f32;
        let observed = [1.0_f32]; // one very saturated pop

        // Step 1 — one-shot quiet-start transform on load.
        let scales = quiet_start_scale(&observed, target);
        assert!(scales[0] < 0.2, "quiet-start did not downscale enough");
        let mut rate = observed[0] * scales[0];

        // Step 2 — homeostatic scaling drives the rest (linear-rate model).
        let p = HomeostaticParams::default();
        let mut converged = false;
        for _ in 0..500 {
            let s = homeostatic_scale(rate, target, &p);
            rate *= s;
            if (rate - target).abs() <= p.deadband_frac * target {
                converged = true;
                break;
            }
        }
        assert!(
            converged,
            "pop failed to recover to target {target} from r = {rate}"
        );
    }

    /// Without quiet-start, recovery from deep saturation via the tight
    /// homeostatic loop alone is slow but still monotonic — crucially, never
    /// oscillates. This documents why we keep the tight bounds even with
    /// quiet-start available.
    #[test]
    fn homeostatic_alone_recovers_monotonically_from_saturation() {
        let target = 0.03_f32;
        let mut rate = 1.0_f32;
        let p = HomeostaticParams::default();
        let mut last = rate;
        for _ in 0..2000 {
            let s = homeostatic_scale(rate, target, &p);
            rate *= s;
            // Monotonic: above target, never increase.
            if rate > target {
                assert!(rate <= last + 1e-9);
            }
            last = rate;
            if (rate - target).abs() <= p.deadband_frac * target {
                return;
            }
        }
        panic!("homeostatic alone failed to converge in 2000 steps; r={rate}");
    }
}
