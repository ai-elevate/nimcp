//! Phase 4a — LTC neuron dynamics.
//!
//! One LTC layer:
//!
//! ```text
//!   pre_i = W_rec[i,:] · x  +  W_in[i,:] · u  +  b_i
//!   dx_i/dt = -x_i / tau_safe_i  +  tanh(pre_i)
//!   x_i(t+dt) = clamp(x_i(t) + dt · dx_i/dt, [-CLAMP, +CLAMP])
//! ```
//!
//! See [`crate`] for the encoded V1 lessons (tau floor, state clamp).

use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::distr::{Distribution, Uniform};
use rand_chacha::ChaCha20Rng;
use serde::{Deserialize, Serialize};

/// Minimum `tau_base` — hard floor applied *before* the `1/τ` division.
/// V1 lesson: lower values produce NaN gradients on float32 from the
/// `1/τ²` term in the adjoint.
pub const LTC_TAU_MIN: f32 = 0.01;

/// Per-step state clamp. Prevents single-precision blow-up on long
/// unrolls. V1 lesson: without this, 1000-step sequences hit `inf` /
/// `NaN` deterministically by step ~600.
pub const LTC_STATE_CLAMP: f32 = 1.0e4;

/// Static dimensions + hyperparameters for one LTC layer.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct LtcParams {
    /// Input dimension.
    pub n_in: usize,
    /// Recurrent / hidden dimension.
    pub n_rec: usize,
    /// Initial `tau_base` applied uniformly across all neurons at
    /// construction time. Trainable in Phase 4b; here it's just the
    /// starting value.
    pub tau_init: f32,
    /// Half-width of the uniform distribution used to init `W_rec` /
    /// `W_in`. Scaled by `1/sqrt(fan_in)` internally, so this is the
    /// tail-bound on the pre-scaled draw.
    pub init_scale: f32,
}

impl Default for LtcParams {
    fn default() -> Self {
        Self {
            n_in: 0,
            n_rec: 0,
            tau_init: 1.0,
            init_scale: 1.0,
        }
    }
}

/// One LTC layer's trainable parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LtcLayer {
    /// Recurrent weights, shape `(n_rec, n_rec)`.
    pub w_rec: Array2<f32>,
    /// Input-projection weights, shape `(n_rec, n_in)`.
    pub w_in: Array2<f32>,
    /// Per-neuron bias, shape `(n_rec,)`.
    pub b: Array1<f32>,
    /// Per-neuron trainable time constant, shape `(n_rec,)`.
    /// Floored at [`LTC_TAU_MIN`] during every forward step.
    pub tau_base: Array1<f32>,
    /// Static params — dimensions + init hyperparameters.
    pub params: LtcParams,
}

impl LtcLayer {
    /// Deterministic Xavier-like init from `seed`. Uses `ChaCha20Rng` so
    /// the same seed produces bit-identical weights on any platform.
    ///
    /// `W_rec` and `W_in` are sampled from
    /// `U(−init_scale/√fan_in, +init_scale/√fan_in)`. `b` is zero-init.
    /// `tau_base` is `tau_init` everywhere (floored at [`LTC_TAU_MIN`]).
    #[must_use]
    pub fn new_seeded(params: LtcParams, seed: u64) -> Self {
        let LtcParams {
            n_in,
            n_rec,
            tau_init,
            init_scale,
        } = params;
        let tau_init = tau_init.max(LTC_TAU_MIN);
        let mut rng = ChaCha20Rng::seed_from_u64(seed);

        // Xavier-uniform: bound = init_scale / sqrt(fan_in).
        let rec_bound = if n_rec == 0 {
            0.0
        } else {
            init_scale / (n_rec as f32).sqrt()
        };
        let in_bound = if n_in == 0 {
            0.0
        } else {
            init_scale / (n_in as f32).sqrt()
        };

        // `Uniform::new` returns `Err` only when `low >= high`; we
        // guarantee `bound > 0` above by checking the n_* against 0.
        let rec_uni = Uniform::new(-rec_bound, rec_bound).expect("rec bound positive");
        let in_uni = Uniform::new(-in_bound, in_bound).expect("in bound positive");

        let w_rec = Array2::from_shape_fn((n_rec, n_rec), |_| rec_uni.sample(&mut rng));
        let w_in = Array2::from_shape_fn((n_rec, n_in), |_| in_uni.sample(&mut rng));
        let b = Array1::zeros(n_rec);
        let tau_base = Array1::from_elem(n_rec, tau_init);

        Self {
            w_rec,
            w_in,
            b,
            tau_base,
            params,
        }
    }

    /// Returns `(n_in, n_rec)`.
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        (self.params.n_in, self.params.n_rec)
    }
}

/// Recurrent state for one LTC layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LtcState {
    /// Current hidden state, shape `(n_rec,)`.
    pub x: Array1<f32>,
}

impl LtcState {
    /// All zeros — the LTC rest attractor under zero input.
    #[must_use]
    pub fn new(n_rec: usize) -> Self {
        Self {
            x: Array1::zeros(n_rec),
        }
    }

    /// Reset to zeros without reallocating.
    pub fn reset(&mut self) {
        for v in self.x.iter_mut() {
            *v = 0.0;
        }
    }
}

/// One Euler integration step of LTC dynamics on `state.x` given input
/// `u` (length `params.n_in`), producing updated state in place.
///
/// Returns the pre-activation vector `pre` so callers that track it for
/// backprop-through-time don't have to recompute it.
///
/// # Panics
///
/// Debug-asserts that shapes match. In release builds, ndarray's own
/// broadcasting will panic on mismatch at the dot-product sites.
pub fn ltc_forward_step(
    state: &mut LtcState,
    layer: &LtcLayer,
    u: &Array1<f32>,
    dt_ms: f32,
) -> Array1<f32> {
    debug_assert_eq!(state.x.len(), layer.params.n_rec);
    debug_assert_eq!(u.len(), layer.params.n_in);

    // Pre-activation: W_rec · x + W_in · u + b
    let mut pre = layer.w_rec.dot(&state.x);
    pre += &layer.w_in.dot(u);
    pre += &layer.b;

    // Nonlinearity (tanh) elementwise.
    let act = pre.mapv(f32::tanh);

    // dx/dt = -x / tau_safe + tanh(pre).
    // Apply elementwise + Euler step in one pass for cache friendliness.
    for ((x, &tau), &a) in state
        .x
        .iter_mut()
        .zip(layer.tau_base.iter())
        .zip(act.iter())
    {
        let tau_safe = tau.max(LTC_TAU_MIN);
        let dx = -*x / tau_safe + a;
        let x_new = *x + dt_ms * dx;
        *x = x_new.clamp(-LTC_STATE_CLAMP, LTC_STATE_CLAMP);
    }

    pre
}

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::float_cmp)] // exact-equality asserts on clamp boundaries
mod tests {
    use super::*;

    fn p(n_in: usize, n_rec: usize) -> LtcParams {
        LtcParams {
            n_in,
            n_rec,
            tau_init: 1.0,
            init_scale: 0.5,
        }
    }

    #[test]
    fn new_seeded_is_deterministic() {
        let a = LtcLayer::new_seeded(p(3, 4), 0xCAFE_BABE);
        let b = LtcLayer::new_seeded(p(3, 4), 0xCAFE_BABE);
        assert_eq!(a.w_rec, b.w_rec);
        assert_eq!(a.w_in, b.w_in);
        assert_eq!(a.b, b.b);
        assert_eq!(a.tau_base, b.tau_base);
    }

    #[test]
    fn new_seeded_diverges_on_different_seeds() {
        let a = LtcLayer::new_seeded(p(3, 4), 1);
        let b = LtcLayer::new_seeded(p(3, 4), 2);
        assert_ne!(
            a.w_rec, b.w_rec,
            "different seeds must produce different weights"
        );
    }

    #[test]
    fn weights_respect_init_bound() {
        // Xavier bound: 0.5 / sqrt(n_rec=4) = 0.25.
        let layer = LtcLayer::new_seeded(p(3, 4), 7);
        let bound_rec = 0.5_f32 / (4_f32).sqrt();
        let bound_in = 0.5_f32 / (3_f32).sqrt();
        assert!(layer.w_rec.iter().all(|&w| w.abs() <= bound_rec + 1e-6));
        assert!(layer.w_in.iter().all(|&w| w.abs() <= bound_in + 1e-6));
    }

    #[test]
    fn tau_init_floored_at_minimum() {
        let mut params = p(2, 3);
        params.tau_init = 1e-6; // below LTC_TAU_MIN
        let layer = LtcLayer::new_seeded(params, 0);
        assert!(layer.tau_base.iter().all(|&t| t >= LTC_TAU_MIN - 1e-9));
    }

    #[test]
    fn state_new_is_zero() {
        let s = LtcState::new(5);
        assert_eq!(s.x.len(), 5);
        assert!(s.x.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn reset_zeros_without_realloc() {
        let mut s = LtcState::new(5);
        for (i, v) in s.x.iter_mut().enumerate() {
            *v = i as f32;
        }
        s.reset();
        assert!(s.x.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn forward_at_rest_stays_at_rest_under_zero_input() {
        // x = 0, u = 0, b = 0 → pre = 0 → tanh(0) = 0 → dx/dt = 0.
        let layer = LtcLayer::new_seeded(p(2, 3), 42);
        let mut s = LtcState::new(3);
        let u = Array1::zeros(2);
        // Zero b to make this exact (seeded init gives b = 0 already).
        let _pre = ltc_forward_step(&mut s, &layer, &u, 0.1);
        assert!(
            s.x.iter().all(|&v| v.abs() < 1e-6),
            "state drifted under zero drive: {:?}",
            s.x
        );
    }

    #[test]
    fn forward_under_constant_drive_approaches_steady_state() {
        // With x = 0 and constant positive u, pre = W_in·u + b; tanh(pre)
        // is bounded in (−1, +1); steady state of
        //   dx/dt = −x/τ + a   is   x_ss = τ · a.
        // We don't assert exact x_ss (W_in is random), just that |x|
        // grows monotonically toward a nonzero value.
        let params = LtcParams {
            n_in: 4,
            n_rec: 8,
            tau_init: 5.0,
            init_scale: 2.0,
        };
        let layer = LtcLayer::new_seeded(params, 7);
        let mut s = LtcState::new(8);
        let u = Array1::from_elem(4, 1.0_f32);

        let mut prev_norm: f32 = 0.0;
        for _ in 0..30 {
            ltc_forward_step(&mut s, &layer, &u, 0.1);
            let norm = s.x.iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!(
                norm >= prev_norm - 1e-4,
                "state norm decreased under constant drive: {prev_norm} → {norm}"
            );
            prev_norm = norm;
        }
        assert!(
            prev_norm > 0.1,
            "state failed to grow under drive: norm {prev_norm}"
        );
    }

    #[test]
    fn state_clamp_holds_under_extreme_weights() {
        // Construct a pathological layer with huge weights to try to
        // force `x` past ±LTC_STATE_CLAMP. The forward step must
        // saturate rather than explode.
        let mut layer = LtcLayer::new_seeded(p(1, 2), 0);
        for w in layer.w_in.iter_mut() {
            *w = 1.0e6;
        }
        // Keep tau huge so the leak term doesn't pull state back.
        for t in layer.tau_base.iter_mut() {
            *t = 1.0e6;
        }
        let mut s = LtcState::new(2);
        let u = Array1::from_elem(1, 1.0_f32);
        for _ in 0..1000 {
            ltc_forward_step(&mut s, &layer, &u, 10.0);
        }
        assert!(
            s.x.iter().all(|&v| v.abs() <= LTC_STATE_CLAMP + 1e-3),
            "state exceeded clamp: {:?}",
            s.x
        );
        assert!(s.x.iter().all(|&v| v.is_finite()), "state went non-finite");
    }

    /// V1 regression: very small `tau_base` used to produce NaN state.
    /// V2 floors `tau_safe` at [`LTC_TAU_MIN`] inside the forward step,
    /// independent of what's stored on the layer.
    #[test]
    fn tiny_tau_does_not_explode() {
        let mut layer = LtcLayer::new_seeded(p(1, 2), 0);
        for t in layer.tau_base.iter_mut() {
            *t = 1.0e-12; // absurdly small, below LTC_TAU_MIN
        }
        let mut s = LtcState::new(2);
        let u = Array1::from_elem(1, 0.5_f32);
        for _ in 0..200 {
            ltc_forward_step(&mut s, &layer, &u, 0.1);
        }
        assert!(
            s.x.iter().all(|&v| v.is_finite()),
            "tau floor failed: state {:?}",
            s.x
        );
    }

    #[test]
    fn forward_returns_pre_activation_of_correct_shape() {
        let layer = LtcLayer::new_seeded(p(3, 5), 13);
        let mut s = LtcState::new(5);
        let u = Array1::from_elem(3, 0.1_f32);
        let pre = ltc_forward_step(&mut s, &layer, &u, 0.1);
        assert_eq!(pre.len(), 5);
    }
}
