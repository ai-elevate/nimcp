//! Dense linear layer — `y = x · W^T + b`.
//!
//! Operates on a 2-D `[batch, in_features]` input (the output of
//! [`crate::FlattenLayer`]) and produces `[batch, out_features]`.
//!
//! Init: Xavier-uniform from a [`rand_chacha::ChaCha20Rng`] so seed →
//! weights is bit-identical across runs and platforms (matches the LNN
//! convention).

use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::distr::{Distribution, Uniform};
use rand_chacha::ChaCha20Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearLayer {
    pub in_features: usize,
    pub out_features: usize,
    /// `[out_features, in_features]`.
    pub weight: Array2<f32>,
    /// `[out_features]`.
    pub bias: Array1<f32>,
}

impl LinearLayer {
    /// Allocate + Xavier-initialise. `seed` drives the ChaCha20 RNG so
    /// the same `seed` always produces the same weights.
    pub fn new(in_features: usize, out_features: usize, seed: u64) -> Self {
        assert!(in_features > 0 && out_features > 0, "linear: zero dim");
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let bound = (6.0_f32 / (in_features + out_features) as f32).sqrt();
        let dist = Uniform::new(-bound, bound).expect("xavier bound > 0");

        let mut w = Array2::<f32>::zeros((out_features, in_features));
        for v in w.iter_mut() {
            *v = dist.sample(&mut rng);
        }
        Self {
            in_features,
            out_features,
            weight: w,
            bias: Array1::<f32>::zeros(out_features),
        }
    }

    /// `input: [batch, in_features]` → `[batch, out_features]`.
    pub fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
        let (n, in_f) = input.dim();
        assert_eq!(
            in_f, self.in_features,
            "linear input feature count mismatch"
        );
        // `x · W^T`. ndarray's `.dot` dispatches to BLAS-style nested
        // loops; for the small dims we exercise here it's fine.
        let mut out = input.dot(&self.weight.t());
        // Broadcast bias over rows.
        for mut row in out.rows_mut() {
            for (o, b) in row.iter_mut().zip(self.bias.iter()) {
                *o += *b;
            }
        }
        let _ = n; // silences unused warn under some lint settings
        out
    }
}
