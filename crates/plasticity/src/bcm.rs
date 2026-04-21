//! Bienenstock–Cooper–Munro (BCM) rate-based plasticity.
//!
//! # Math
//!
//! ```text
//! Δw = η · pre_rate · post_rate · (post_rate − θ_m)
//! ```
//!
//! Sign of `Δw` flips at `post_rate = θ_m`:
//! - `post_rate > θ_m` → potentiation
//! - `post_rate < θ_m` → depression
//!
//! The threshold `θ_m` ("modification threshold") is typically the
//! time-averaged square of `post_rate`; we treat it as an input here,
//! letting callers manage its own sliding-average state.

/// Compute the BCM weight delta.
///
/// `Δw = η · pre_rate · post_rate · (post_rate − θ_m)`.
#[must_use]
#[inline]
pub fn bcm_weight_delta(pre_rate: f32, post_rate: f32, theta_m: f32, learning_rate: f32) -> f32 {
    learning_rate * pre_rate * post_rate * (post_rate - theta_m)
}

#[cfg(test)]
#[allow(clippy::float_cmp)] // exact-equality asserts are intentional
mod tests {
    use super::*;

    /// Post above threshold → positive Δw (potentiation).
    #[test]
    fn post_above_threshold_potentiates() {
        let dw = bcm_weight_delta(1.0, 2.0, 1.0, 0.1);
        assert!(dw > 0.0, "post > θ should potentiate, got {dw}");
    }

    /// Post below threshold → negative Δw (depression).
    #[test]
    fn post_below_threshold_depresses() {
        let dw = bcm_weight_delta(1.0, 0.5, 1.0, 0.1);
        assert!(dw < 0.0, "post < θ should depress, got {dw}");
    }

    /// Post equals threshold → zero Δw.
    #[test]
    fn post_at_threshold_zero() {
        let dw = bcm_weight_delta(1.0, 1.0, 1.0, 0.1);
        assert_eq!(dw, 0.0);
    }

    /// Zero pre_rate → zero Δw (no pre firing, no learning).
    #[test]
    fn zero_pre_rate_no_change() {
        let dw = bcm_weight_delta(0.0, 5.0, 1.0, 0.1);
        assert_eq!(dw, 0.0);
    }

    /// Zero post_rate → zero Δw.
    #[test]
    fn zero_post_rate_no_change() {
        let dw = bcm_weight_delta(1.0, 0.0, 1.0, 0.1);
        assert_eq!(dw, 0.0);
    }

    /// Sign depends strictly on `sign(post - θ)` when pre > 0, post > 0.
    #[test]
    fn sign_depends_on_post_minus_theta() {
        let lr = 0.1;
        let pre = 1.0;
        let theta = 0.5;
        for post in [0.1, 0.2, 0.4, 0.6, 0.8, 1.0_f32] {
            let dw = bcm_weight_delta(pre, post, theta, lr);
            if post > theta {
                assert!(dw > 0.0, "post={post} > θ={theta}: expected +, got {dw}");
            } else if post < theta {
                assert!(dw < 0.0, "post={post} < θ={theta}: expected -, got {dw}");
            } else {
                assert_eq!(dw, 0.0);
            }
        }
    }

    /// Exact arithmetic check.
    #[test]
    fn exact_formula() {
        // η=0.1, pre=2.0, post=3.0, θ=1.0 → 0.1 · 2 · 3 · (3-1) = 1.2
        let dw = bcm_weight_delta(2.0, 3.0, 1.0, 0.1);
        assert!((dw - 1.2).abs() < 1e-6);
    }
}
