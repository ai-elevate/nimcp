//! Lorentz-factor helpers (1+1-dim special-relativistic boost).
//!
//! Port of V1's `utils/geometry/nimcp_lorentz.h` in minimum-viable
//! form. V2 only needs the 1-D boost primitives (the full 4-vector
//! algebra isn't load-bearing anywhere in the V2 network core).
//!
//! Speeds are given as fractions of `c` (unit speed). A speed `|v|`
//! strictly less than `1.0` is required; boundary speeds are clamped
//! to `1 − EPS`.

/// Maximum usable speed — keeps `1 − v²` away from zero.
pub const LORENTZ_V_MAX: f32 = 1.0 - 1.0e-6;

/// Lorentz factor `γ = 1 / sqrt(1 − v²)`. Clamps `|v|` to
/// `LORENTZ_V_MAX` to avoid `+∞`.
#[must_use]
pub fn lorentz_factor(v: f32) -> f32 {
    let vc = v.clamp(-LORENTZ_V_MAX, LORENTZ_V_MAX);
    1.0 / (1.0 - vc * vc).sqrt()
}

/// 1-D Lorentz boost of the event `(t, x)` by velocity `v`.
/// Returns the transformed `(t', x')`.
#[must_use]
pub fn lorentz_boost(t: f32, x: f32, v: f32) -> (f32, f32) {
    let vc = v.clamp(-LORENTZ_V_MAX, LORENTZ_V_MAX);
    let gamma = 1.0 / (1.0 - vc * vc).sqrt();
    let t_new = gamma * (t - vc * x);
    let x_new = gamma * (x - vc * t);
    (t_new, x_new)
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn gamma_at_rest_is_one() {
        assert!((lorentz_factor(0.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn gamma_at_half_c() {
        // γ(0.5) = 1/sqrt(0.75) ≈ 1.1547
        assert!((lorentz_factor(0.5) - 1.1547).abs() < 1e-3);
    }

    #[test]
    fn gamma_clamps_at_extreme() {
        // At v = 1.0 (unclamped) γ would be ∞. Clamp keeps it finite.
        assert!(lorentz_factor(1.0).is_finite());
        assert!(lorentz_factor(-1.0).is_finite());
    }

    #[test]
    fn boost_zero_is_identity() {
        let (t, x) = lorentz_boost(3.0, 4.0, 0.0);
        assert!((t - 3.0).abs() < 1e-6);
        assert!((x - 4.0).abs() < 1e-6);
    }

    #[test]
    fn boost_inverse_recovers_event() {
        let (t, x) = lorentz_boost(3.0, 4.0, 0.3);
        let (t2, x2) = lorentz_boost(t, x, -0.3);
        assert!((t2 - 3.0).abs() < 1e-5);
        assert!((x2 - 4.0).abs() < 1e-5);
    }

    #[test]
    fn boost_preserves_interval() {
        // s² = t² − x² is a Lorentz invariant.
        let (t1, x1) = (5.0_f32, 3.0);
        let interval_before = t1 * t1 - x1 * x1;
        let (t2, x2) = lorentz_boost(t1, x1, 0.4);
        let interval_after = t2 * t2 - x2 * x2;
        assert!(
            (interval_before - interval_after).abs() < 1e-4,
            "interval {interval_before} != {interval_after}"
        );
    }
}
