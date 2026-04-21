//! Homeostatic synaptic scaling.
//!
//! # Math
//!
//! Drives population firing rate `r` toward target `r*` by multiplicatively
//! scaling all incoming weights. The raw control law is
//!
//! ```text
//! s = 1 + η · (r* − r) / r*
//! ```
//!
//! where `η` is the gain. The output is clamped to `[min_scale, max_scale]`,
//! and a deadband around `r*` returns exactly `1.0` to prevent micro-dither.
//!
//! # V1 history — "emergency band oscillation"
//!
//! V1 shipped with `[0.90, 1.10]` bounds, which gave the loop enough gain to
//! overshoot the target every tick and oscillate bang-bang forever. The fix
//! in V1 commit `776511957` tightened bounds to `[0.98, 1.02]`. In V2 the
//! tight bounds are the *default and only* production values; regression
//! tests below would have caught the V1 bug on day one.

use serde::{Deserialize, Serialize};

/// Default lower bound on per-tick scale factor.
///
/// Tight bound matters: `0.90` gave the V1 loop enough gain to oscillate
/// bang-bang. Keep this tight.
pub const DEFAULT_MIN_SCALE: f32 = 0.98;
/// Default upper bound on per-tick scale factor.
pub const DEFAULT_MAX_SCALE: f32 = 1.02;
/// Default deadband fraction — below this relative error, no change.
/// `0.01` = 1% of target. In scale space this corresponds to roughly
/// `[0.99, 1.01]` with unit gain; micro-adjustments are suppressed so the
/// loop doesn't dither forever.
pub const DEFAULT_DEADBAND_FRAC: f32 = 0.01;
/// Default gain on the proportional term.
pub const DEFAULT_GAIN: f32 = 1.0;

/// Homeostatic scaling parameters.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HomeostaticParams {
    /// Lower bound on the returned scale factor. Default `0.98`.
    pub min_scale: f32,
    /// Upper bound on the returned scale factor. Default `1.02`.
    pub max_scale: f32,
    /// Fractional deadband around the target (e.g. `0.01` = 1%).
    pub deadband_frac: f32,
    /// Proportional gain. Default `1.0`.
    pub gain: f32,
}

impl Default for HomeostaticParams {
    fn default() -> Self {
        Self {
            min_scale: DEFAULT_MIN_SCALE,
            max_scale: DEFAULT_MAX_SCALE,
            deadband_frac: DEFAULT_DEADBAND_FRAC,
            gain: DEFAULT_GAIN,
        }
    }
}

/// Compute a per-tick multiplicative scale factor that drives `current_rate`
/// toward `target_rate`.
///
/// - Returns `1.0` exactly when `|r − r*| ≤ deadband_frac · r*`.
/// - Otherwise returns `clamp(1 + gain · (r* − r)/r*, min_scale, max_scale)`.
/// - A non-positive `target_rate` is a no-op (returns `1.0`); plasticity on a
///   dead target is undefined.
#[must_use]
pub fn homeostatic_scale(
    current_rate: f32,
    target_rate: f32,
    params: &HomeostaticParams,
) -> f32 {
    // Invalid target → no scaling. Do not panic in the hot path.
    if !(target_rate > 0.0) {
        return 1.0;
    }

    let err = target_rate - current_rate;
    let deadband = params.deadband_frac * target_rate;
    if err.abs() <= deadband {
        return 1.0;
    }

    // Proportional control law, clamped.
    let raw = 1.0 + params.gain * (err / target_rate);
    raw.clamp(params.min_scale, params.max_scale)
}

#[cfg(test)]
#[allow(clippy::float_cmp)] // exact-equality asserts are the point — we
                            // test that tight bounds clamp to exact values.
mod tests {
    use super::*;

    /// Default bounds are the tight V2 values, NOT the legacy `[0.90, 1.10]`.
    #[test]
    fn default_bounds_are_tight() {
        let p = HomeostaticParams::default();
        assert_eq!(p.min_scale, 0.98);
        assert_eq!(p.max_scale, 1.02);
    }

    /// Regression: saturated rate (r=1.0, r*=0.03) clamps to exactly `0.98`,
    /// not something wider. This would have caught the V1 emergency-band bug.
    #[test]
    fn saturated_rate_clamps_to_lower_bound_exactly_098() {
        let p = HomeostaticParams::default();
        let s = homeostatic_scale(1.0, 0.03, &p);
        assert_eq!(
            s, 0.98,
            "expected exact lower-bound clamp at 0.98, got {s}"
        );
    }

    /// Dead rate (r=0) clamps to upper bound exactly.
    #[test]
    fn dead_rate_clamps_to_upper_bound() {
        let p = HomeostaticParams::default();
        let s = homeostatic_scale(0.0, 0.03, &p);
        assert_eq!(s, 1.02);
    }

    /// Rate within deadband returns exactly 1.0 (no micro-adjustment).
    /// 0.0302 is within 0.5% of 0.03 → deadband hit.
    #[test]
    fn deadband_returns_exactly_one() {
        let p = HomeostaticParams::default();
        let s = homeostatic_scale(0.0302, 0.03, &p);
        assert_eq!(s, 1.0, "deadband should return exactly 1.0, got {s}");
    }

    /// Just-outside-deadband returns != 1.0 (crisp boundary).
    #[test]
    fn outside_deadband_adjusts() {
        let p = HomeostaticParams::default();
        // 0.029 is ~3.3% low — well outside 0.5% deadband.
        let s = homeostatic_scale(0.029, 0.03, &p);
        assert!(s > 1.0);
        assert!(s <= p.max_scale);
    }

    /// Invalid target (≤ 0) is a no-op.
    #[test]
    fn invalid_target_is_noop() {
        let p = HomeostaticParams::default();
        assert_eq!(homeostatic_scale(0.5, 0.0, &p), 1.0);
        assert_eq!(homeostatic_scale(0.5, -1.0, &p), 1.0);
    }

    /// Regression: **anti-oscillation**. This test would have caught the V1
    /// `[0.90, 1.10]` emergency-band bang-bang bug. Start saturated, iterate
    /// 100 times. Rate is assumed to scale linearly with weights (first-order
    /// approx). Expect monotonic convergence: no overshoot past target, no
    /// oscillation.
    ///
    /// With the tight `[0.98, 1.02]` bounds, a single step can at most shave
    /// 2% off the weight — we don't expect full convergence in 100 steps
    /// (that takes `ln(0.03/1)/ln(0.98)` ≈ 174 steps from saturation). The
    /// invariant we DO assert is: rate monotonically decreases toward target
    /// on every step, never overshooting.
    #[test]
    fn anti_oscillation_monotonic_convergence() {
        let p = HomeostaticParams::default();
        let target = 0.03_f32;
        let mut rate = 1.0_f32; // saturated start
        let mut prev_rate = rate;

        for step in 0..100 {
            let s = homeostatic_scale(rate, target, &p);
            rate *= s; // linear-rate assumption

            // Rate is above target the whole time (tight bounds, slow decay),
            // so we should see monotonic decrease — no oscillation.
            assert!(
                rate <= prev_rate + 1e-9,
                "step {step}: rate increased ({prev_rate} → {rate}) — oscillation!"
            );
            // And must never overshoot past target (that would be bang-bang).
            assert!(
                rate >= target - 1e-6,
                "step {step}: rate overshot target ({rate} < {target}) — \
                 bang-bang bug returned"
            );
            prev_rate = rate;
        }

        // After 100 steps we should have made serious progress.
        assert!(
            rate < 0.5,
            "rate {rate} barely moved in 100 steps — gain too low"
        );
    }

    /// Companion to anti-oscillation: given enough iterations, the tight
    /// loop DOES eventually converge. This proves monotonic convergence is
    /// convergence, not just "monotonically stalled".
    #[test]
    fn tight_loop_converges_eventually() {
        let p = HomeostaticParams::default();
        let target = 0.03_f32;
        let mut rate = 1.0_f32;
        for _ in 0..2000 {
            let s = homeostatic_scale(rate, target, &p);
            rate *= s;
            if (rate - target).abs() <= p.deadband_frac * target {
                return;
            }
        }
        panic!("rate {rate} failed to converge in 2000 steps");
    }

    /// Under V1's wide bounds, the same initial condition oscillates. Here
    /// we invert the regression: if bounds were `[0.90, 1.10]`, the loop
    /// undershoots → this is why we forbid that config.
    #[test]
    fn wide_bounds_would_overshoot_demo() {
        // Reproduce the V1 bug condition for documentation purposes.
        let bad = HomeostaticParams {
            min_scale: 0.90,
            max_scale: 1.10,
            ..HomeostaticParams::default()
        };
        let target = 0.03_f32;
        let rate = 1.0_f32;
        let s = homeostatic_scale(rate, target, &bad);
        // s clamps to 0.90 → new rate = 0.9, still way above target.
        let new_rate = rate * s;
        assert_eq!(s, 0.90);
        // With the wide band, a single step overshoots target in a few more
        // iterations and rings — we just show the first step difference from
        // the tight-bounds case to make intent clear.
        let tight_s = homeostatic_scale(rate, target, &HomeostaticParams::default());
        assert!(
            (new_rate - 0.9).abs() < 1e-6 && (rate * tight_s - 0.98).abs() < 1e-6,
            "wide bounds step to 0.9 vs tight bounds step to 0.98"
        );
    }
}
