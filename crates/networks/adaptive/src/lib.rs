//! NIMCP V2 — adaptive network (multi-layer perceptron).
//!
//! This is V2's successor to `adaptive_network_learn` in V1. CPU-first in
//! Phase 1; GPU offload lands in Phase 2.
//!
//! # What's different from V1
//!
//! - Pure `Kernel` implementation — no shared `neural_network_t` struct.
//! - Forward + backward emit events (weight updates) to the scheduler.
//! - No global learning rate in the network; passed in per-call.
//! - Seeded RNG for reproducible init.

#![forbid(unsafe_code)]
#![allow(dead_code)] // Phase 1 stub — methods land incrementally

use ndarray::Array2;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use serde::{Deserialize, Serialize};

/// Configuration for an adaptive network.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveConfig {
    /// Layer widths including input + output. E.g. `[784, 128, 10]` for MNIST.
    pub layers: Vec<usize>,
    /// Seed for weight initialization. Same seed ⇒ same init, bit-for-bit.
    pub rng_seed: u64,
    /// Activation between hidden layers. Output layer is linear.
    pub activation: Activation,
}

/// Activation function choices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Activation {
    /// Rectified linear: `max(0, x)`. Default.
    Relu,
    /// Hyperbolic tangent.
    Tanh,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            layers: vec![64, 32, 10],
            rng_seed: 0xABCDEF,
            activation: Activation::Relu,
        }
    }
}

/// Weights for one layer transition: `W @ x + b`.
#[derive(Debug, Clone)]
pub struct LayerWeights {
    /// Weight matrix shaped (out_features, in_features).
    pub w: Array2<f32>,
    /// Bias vector shaped (out_features,).
    pub b: ndarray::Array1<f32>,
}

/// Adaptive network state.
pub struct AdaptiveNet {
    config: AdaptiveConfig,
    weights: Vec<LayerWeights>,
}

impl AdaptiveNet {
    /// Create a new network with weights initialized per `config.rng_seed`.
    pub fn new(config: AdaptiveConfig) -> Self {
        let mut rng = ChaCha20Rng::seed_from_u64(config.rng_seed);
        let mut weights = Vec::with_capacity(config.layers.len().saturating_sub(1));

        for w in config.layers.windows(2) {
            let (in_dim, out_dim) = (w[0], w[1]);
            weights.push(init_layer(in_dim, out_dim, &mut rng));
        }

        Self { config, weights }
    }

    /// Number of weight tensors (= layers - 1).
    pub fn num_transitions(&self) -> usize {
        self.weights.len()
    }

    /// Read-only access to the config.
    pub fn config(&self) -> &AdaptiveConfig {
        &self.config
    }
}

/// Initialize one layer with Kaiming-like scaling (ReLU-friendly).
fn init_layer(in_dim: usize, out_dim: usize, rng: &mut ChaCha20Rng) -> LayerWeights {
    use rand::distr::{Distribution, Uniform};
    let scale = (2.0_f32 / in_dim as f32).sqrt();
    let dist = Uniform::new(-scale, scale).unwrap();
    let w = Array2::from_shape_fn((out_dim, in_dim), |_| dist.sample(rng));
    let b = ndarray::Array1::zeros(out_dim);
    LayerWeights { w, b }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_builds_expected_layer_count() {
        let net = AdaptiveNet::new(AdaptiveConfig {
            layers: vec![10, 20, 30, 5],
            ..Default::default()
        });
        assert_eq!(net.num_transitions(), 3);
        assert_eq!(net.weights[0].w.dim(), (20, 10));
        assert_eq!(net.weights[1].w.dim(), (30, 20));
        assert_eq!(net.weights[2].w.dim(), (5, 30));
    }

    #[test]
    fn same_seed_same_weights() {
        let cfg = AdaptiveConfig {
            layers: vec![4, 8, 2],
            rng_seed: 42,
            activation: Activation::Relu,
        };
        let a = AdaptiveNet::new(cfg.clone());
        let b = AdaptiveNet::new(cfg);
        for (la, lb) in a.weights.iter().zip(b.weights.iter()) {
            assert_eq!(la.w, lb.w);
            assert_eq!(la.b, lb.b);
        }
    }
}
