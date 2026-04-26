//! Scalar `H(q, p)` MLP with hand-rolled reverse-mode autodiff.
//!
//! Forward stack:
//!
//! ```text
//! z_0 = concat(q, p)                              [2*dof]
//! for l in 1..=n_hidden:
//!     pre_l = W_l · z_{l-1} + b_l
//!     z_l   = act(pre_l)
//! H = w_out · z_L + b_out                          (scalar)
//! ```
//!
//! Backward (gradient of `H` w.r.t. the input `[q; p]`):
//!
//! ```text
//! dH/dz_L = w_out
//! for l in L..1:
//!     dH/dpre_l   = dH/dz_l ⊙ act'(pre_l)
//!     dH/dz_{l-1} = W_l^T · dH/dpre_l
//! dH/d[q;p] = dH/dz_0
//! ```
//!
//! The backward only computes input gradients — *not* parameter
//! gradients. Training is out of scope for Phase 11c (it would also
//! need a full backward through `step()`, which symplectic-Euler
//! makes non-trivial). Inference + integrator-driven dynamics is what
//! we ship here.

use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::distr::{Distribution, Uniform};
use rand_chacha::ChaCha20Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum MlpActivation {
    #[default]
    Tanh,
}

impl MlpActivation {
    fn apply(&self, x: f32) -> f32 {
        match self {
            MlpActivation::Tanh => x.tanh(),
        }
    }
    /// Derivative evaluated at the *output* of the activation (for
    /// `tanh`: `1 - y²`). Caller must pass the post-activation value,
    /// not the pre-activation.
    fn derivative_at_output(&self, y: f32) -> f32 {
        match self {
            MlpActivation::Tanh => 1.0 - y * y,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HamiltonianMlp {
    /// Per-layer weight matrices `[out, in]`, hidden stack only.
    pub weights: Vec<Array2<f32>>,
    /// Per-layer biases `[out]`.
    pub biases: Vec<Array1<f32>>,
    /// Output projection `[hidden_last]` and bias scalar.
    pub out_weight: Array1<f32>,
    pub out_bias: f32,
    pub activation: MlpActivation,
    /// Number of canonical coordinates per side; input is `[q; p]` of
    /// total length `2 * dof`.
    pub dof: usize,
}

impl HamiltonianMlp {
    /// `hidden_layers` — list of hidden widths. Empty → linear in
    /// `[q; p]` (still useful for tiny harmonic-oscillator tests).
    pub fn new(dof: usize, hidden_layers: &[usize], seed: u64) -> Self {
        assert!(dof > 0, "hnn: dof must be > 0");
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let in_dim = 2 * dof;

        let mut weights: Vec<Array2<f32>> = Vec::with_capacity(hidden_layers.len());
        let mut biases: Vec<Array1<f32>> = Vec::with_capacity(hidden_layers.len());
        let mut prev = in_dim;
        for &h in hidden_layers {
            assert!(h > 0, "hnn: zero-width hidden layer");
            let bound = (6.0_f32 / (prev + h) as f32).sqrt();
            let dist = Uniform::new(-bound, bound).expect("xavier bound > 0");
            let mut w = Array2::<f32>::zeros((h, prev));
            for v in w.iter_mut() {
                *v = dist.sample(&mut rng);
            }
            weights.push(w);
            biases.push(Array1::<f32>::zeros(h));
            prev = h;
        }

        let bound = (6.0_f32 / (prev + 1) as f32).sqrt();
        let dist = Uniform::new(-bound, bound).expect("xavier bound > 0");
        let mut out_w = Array1::<f32>::zeros(prev);
        for v in out_w.iter_mut() {
            *v = dist.sample(&mut rng);
        }

        Self {
            weights,
            biases,
            out_weight: out_w,
            out_bias: 0.0,
            activation: MlpActivation::Tanh,
            dof,
        }
    }

    /// Returns `(H, dH/dq, dH/dp)`. `q.len() == p.len() == self.dof`.
    pub fn evaluate(&self, q: &Array1<f32>, p: &Array1<f32>) -> (f32, Array1<f32>, Array1<f32>) {
        assert_eq!(q.len(), self.dof, "hnn: q length mismatch");
        assert_eq!(p.len(), self.dof, "hnn: p length mismatch");

        // Build z_0 = [q; p] in one allocation.
        let mut z0 = Array1::<f32>::zeros(2 * self.dof);
        for i in 0..self.dof {
            z0[i] = q[i];
            z0[self.dof + i] = p[i];
        }

        // Forward; cache post-activation outputs for backward.
        let mut activations: Vec<Array1<f32>> = Vec::with_capacity(self.weights.len() + 1);
        activations.push(z0);
        for (w, b) in self.weights.iter().zip(self.biases.iter()) {
            let prev = activations.last().expect("seed activation");
            let mut pre = w.dot(prev);
            for (v, bv) in pre.iter_mut().zip(b.iter()) {
                *v += *bv;
            }
            let post = pre.mapv(|x| self.activation.apply(x));
            activations.push(post);
        }
        let last = activations.last().expect("at least the input is present");
        let h_scalar = self.out_weight.dot(last) + self.out_bias;

        // Backward: dH/dz_L = w_out.
        let mut grad = self.out_weight.clone();
        // Walk layers in reverse, applying chain rule.
        for layer_idx in (0..self.weights.len()).rev() {
            // Multiply by act'(pre) — using the *post-activation* form
            // act'(out) for tanh.
            let post = &activations[layer_idx + 1];
            for (g, y) in grad.iter_mut().zip(post.iter()) {
                *g *= self.activation.derivative_at_output(*y);
            }
            // Push grad through W^T to get gradient w.r.t. previous activation.
            let w = &self.weights[layer_idx];
            grad = w.t().dot(&grad);
        }

        // Split grad over [q; p].
        let mut dh_dq = Array1::<f32>::zeros(self.dof);
        let mut dh_dp = Array1::<f32>::zeros(self.dof);
        for i in 0..self.dof {
            dh_dq[i] = grad[i];
            dh_dp[i] = grad[self.dof + i];
        }
        (h_scalar, dh_dq, dh_dp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Validate the analytic gradient against a centred finite-
    /// difference for a small random network. Tolerance is loose
    /// because we're working in `f32`.
    #[test]
    fn evaluate_gradients_match_finite_difference() {
        let mlp = HamiltonianMlp::new(3, &[8, 8], 0xBEEF);
        let q = Array1::from_vec(vec![0.3, -0.4, 0.1]);
        let p = Array1::from_vec(vec![-0.2, 0.5, 0.0]);

        let (_h0, dh_dq, dh_dp) = mlp.evaluate(&q, &p);
        let eps = 1e-3_f32;
        for i in 0..mlp.dof {
            // Numerical dH/dq[i].
            let mut q_plus = q.clone();
            let mut q_minus = q.clone();
            q_plus[i] += eps;
            q_minus[i] -= eps;
            let (h_plus, _, _) = mlp.evaluate(&q_plus, &p);
            let (h_minus, _, _) = mlp.evaluate(&q_minus, &p);
            let num = (h_plus - h_minus) / (2.0 * eps);
            assert!(
                (num - dh_dq[i]).abs() < 1e-2,
                "dH/dq[{i}] analytic {} vs FD {}",
                dh_dq[i],
                num
            );

            // Numerical dH/dp[i].
            let mut p_plus = p.clone();
            let mut p_minus = p.clone();
            p_plus[i] += eps;
            p_minus[i] -= eps;
            let (h_plus, _, _) = mlp.evaluate(&q, &p_plus);
            let (h_minus, _, _) = mlp.evaluate(&q, &p_minus);
            let num = (h_plus - h_minus) / (2.0 * eps);
            assert!(
                (num - dh_dp[i]).abs() < 1e-2,
                "dH/dp[{i}] analytic {} vs FD {}",
                dh_dp[i],
                num
            );
        }
    }

    #[test]
    fn deterministic_weights_under_seed() {
        let a = HamiltonianMlp::new(2, &[6, 4], 0x42);
        let b = HamiltonianMlp::new(2, &[6, 4], 0x42);
        for (wa, wb) in a.weights.iter().zip(b.weights.iter()) {
            assert_eq!(wa, wb);
        }
        assert_eq!(a.out_weight, b.out_weight);
    }
}
