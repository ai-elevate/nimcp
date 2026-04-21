//! NIMCP V2 — adaptive network (multi-layer perceptron).
//!
//! This is V2's successor to `adaptive_network_learn` in V1. CPU-first in
//! Phase 1; GPU offload lands in Phase 2.
//!
//! # What's different from V1
//!
//! - Pure forward + backward — no shared `neural_network_t` struct.
//! - Training emits [`WeightUpdateEvent`] values (conceptually; Phase 1
//!   logs them via `tracing` and does not yet wire into the event log).
//! - No global learning rate in the network; passed in per-call.
//! - Seeded RNG for reproducible init.
//!
//! # Phase 1 scope
//!
//! - MLP forward: `h = activation(W @ h_prev + b)`; output layer linear.
//! - ReLU + Tanh activations (output layer is always linear).
//! - Standard backprop against MSE loss; in-place SGD weight update.
//! - Save / load via rkyv.
//! - Bit-for-bit determinism given an RNG seed.

#![forbid(unsafe_code)]

use ndarray::{Array1, Array2};
use nimcp_core::Event;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rkyv::rancor;
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
    pub b: Array1<f32>,
}

/// Error type for the adaptive crate.
#[derive(Debug, thiserror::Error)]
pub enum AdaptiveError {
    /// Input dimensionality doesn't match the first layer width.
    #[error("shape mismatch: expected input of size {expected}, got {got}")]
    ShapeMismatch {
        /// Expected dimension.
        expected: usize,
        /// Actual dimension received.
        got: usize,
    },
    /// Serialization failed during save or load.
    #[error("rkyv serialization: {0}")]
    Serialization(String),
    /// Loaded checkpoint disagrees with current config (layer shapes mismatch).
    #[error("checkpoint shape mismatch: {0}")]
    Checkpoint(String),
}

/// Adaptive network state.
#[derive(Debug)]
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

    /// Read-only borrow of layer weights. Ordering matches `config.layers`
    /// windows; `weights()[i]` maps `layers[i]` → `layers[i + 1]`.
    pub fn weights(&self) -> &[LayerWeights] {
        &self.weights
    }

    /// Forward pass: returns the output of the final (linear) layer.
    ///
    /// Panics if `x.len()` doesn't match the input width.
    ///
    /// # Correctness
    ///
    /// - Hidden layers apply `self.config.activation` after affine.
    /// - Output layer is always linear (no activation).
    pub fn forward(&self, x: &Array1<f32>) -> Array1<f32> {
        assert_eq!(
            x.len(),
            self.config.layers[0],
            "input size {} doesn't match first layer {}",
            x.len(),
            self.config.layers[0]
        );
        let mut h = x.clone();
        let last = self.weights.len() - 1;
        for (i, layer) in self.weights.iter().enumerate() {
            let mut z = layer.w.dot(&h) + &layer.b;
            if i != last {
                apply_activation(&mut z, self.config.activation);
            }
            h = z;
        }
        h
    }

    /// Backward pass + in-place SGD update against MSE loss.
    ///
    /// Returns the loss computed **before** the weight update — useful for
    /// logging monotone-ish convergence.
    ///
    /// MSE is `mean((pred - y)^2)` so `dL/dpred = 2 * (pred - y) / n`.
    pub fn learn(&mut self, x: &Array1<f32>, y: &Array1<f32>, lr: f32) -> f32 {
        let n_layers = self.weights.len();
        assert_eq!(x.len(), self.config.layers[0]);
        assert_eq!(y.len(), self.config.layers[n_layers]);

        // Forward, saving pre-activation z and activated h per layer.
        // h_pre[i] is the input to weights[i]; z[i] is pre-activation output;
        // h_post[i] is z[i] passed through activation (except final layer).
        let mut activations: Vec<Array1<f32>> = Vec::with_capacity(n_layers + 1);
        let mut pre_activations: Vec<Array1<f32>> = Vec::with_capacity(n_layers);

        activations.push(x.clone());
        let last = n_layers - 1;
        for (i, layer) in self.weights.iter().enumerate() {
            let z = layer.w.dot(&activations[i]) + &layer.b;
            pre_activations.push(z.clone());
            if i != last {
                let mut h_post = z;
                apply_activation(&mut h_post, self.config.activation);
                activations.push(h_post);
            } else {
                activations.push(z);
            }
        }
        let pred = activations.last().expect("at least one layer").clone();

        // MSE loss (mean over output dims) pre-update.
        let n_out = pred.len() as f32;
        let diff = &pred - y;
        let loss = diff.iter().map(|&d| d * d).sum::<f32>() / n_out;

        // Backward: delta at final layer is dL/dz_last = 2*(pred - y) / n.
        // No activation on output layer, so dL/dz == dL/dpred.
        let mut grad_z = diff.mapv(|d| 2.0 * d / n_out);

        // Store per-layer deltas applied (for WeightUpdateEvent logging).
        let mut per_layer_delta_norms: Vec<f32> = Vec::with_capacity(n_layers);

        for i in (0..n_layers).rev() {
            let input = &activations[i]; // shape (in_dim,)
            // grad_w = grad_z ⊗ input  (outer product, shape (out, in))
            let grad_w = outer(&grad_z, input);
            let grad_b = grad_z.clone();

            // Compute delta norm before mutating weights (for logging).
            let delta_w_norm = grad_w.iter().map(|&g| (lr * g).powi(2)).sum::<f32>().sqrt();
            per_layer_delta_norms.push(delta_w_norm);

            // Propagate grad to previous layer's post-activation, THEN through its
            // activation derivative. For layer 0 we don't need to propagate further.
            let w_clone = if i > 0 {
                Some(self.weights[i].w.clone())
            } else {
                None
            };

            // Apply SGD update in place.
            self.weights[i].w.scaled_add(-lr, &grad_w);
            self.weights[i].b.scaled_add(-lr, &grad_b);

            if let Some(w_prev) = w_clone {
                // dL/dh_{i-1} = W_i^T @ grad_z
                let grad_h_prev = w_prev.t().dot(&grad_z);
                // Apply derivative of activation on h_{i-1} == activated z_{i-1}.
                // We activated z_{i-1} into activations[i], so derivative uses
                // pre_activations[i-1].
                let z_prev = &pre_activations[i - 1];
                let mut new_grad_z = Array1::<f32>::zeros(z_prev.len());
                for (k, (&gh, &zp)) in grad_h_prev.iter().zip(z_prev.iter()).enumerate() {
                    new_grad_z[k] = gh * activation_deriv(zp, self.config.activation);
                }
                grad_z = new_grad_z;
            }
        }

        // Event emission (conceptual in Phase 1). Reverse so index 0 is first layer.
        per_layer_delta_norms.reverse();
        let event = WeightUpdateEvent {
            loss,
            lr,
            per_layer_delta_norm: per_layer_delta_norms,
        };
        tracing::trace!(?event, "weight update");

        loss
    }

    /// Serialize `self.weights` to a portable byte payload via rkyv.
    ///
    /// Only the weights are persisted; the caller restores config
    /// separately (the config is a build-time thing, not a runtime thing).
    pub fn save(&self) -> Result<Vec<u8>, AdaptiveError> {
        let owned: Vec<LayerWeightsOwned> = self
            .weights
            .iter()
            .map(LayerWeightsOwned::from_view)
            .collect();
        let bytes = rkyv::to_bytes::<rancor::Error>(&owned)
            .map_err(|e| AdaptiveError::Serialization(e.to_string()))?;
        Ok(bytes.to_vec())
    }

    /// Load weights previously produced by [`AdaptiveNet::save`].
    ///
    /// Fails if the loaded shape doesn't match the current network's
    /// configured layer widths — callers must instantiate a net with the
    /// same `AdaptiveConfig.layers` before loading.
    pub fn load(&mut self, bytes: &[u8]) -> Result<(), AdaptiveError> {
        let owned: Vec<LayerWeightsOwned> =
            rkyv::from_bytes::<Vec<LayerWeightsOwned>, rancor::Error>(bytes)
                .map_err(|e| AdaptiveError::Serialization(e.to_string()))?;
        if owned.len() != self.weights.len() {
            return Err(AdaptiveError::Checkpoint(format!(
                "loaded {} layers, network has {}",
                owned.len(),
                self.weights.len()
            )));
        }
        let mut new_weights = Vec::with_capacity(owned.len());
        for (i, lw) in owned.into_iter().enumerate() {
            let expected = (self.weights[i].w.nrows(), self.weights[i].w.ncols());
            if lw.w_shape != [expected.0, expected.1] || lw.b.len() != expected.0 {
                return Err(AdaptiveError::Checkpoint(format!(
                    "layer {i} shape mismatch: got {:?}/{}, expected {:?}/{}",
                    lw.w_shape,
                    lw.b.len(),
                    [expected.0, expected.1],
                    expected.0
                )));
            }
            let w = Array2::from_shape_vec((lw.w_shape[0], lw.w_shape[1]), lw.w)
                .map_err(|e| AdaptiveError::Checkpoint(format!("layer {i} w reshape: {e}")))?;
            let b = Array1::from_vec(lw.b);
            new_weights.push(LayerWeights { w, b });
        }
        self.weights = new_weights;
        Ok(())
    }
}

/// Serializable plain-data mirror of [`LayerWeights`] used by rkyv.
///
/// `ndarray::Array2` doesn't implement rkyv's `Archive` trait, so we
/// flatten to `Vec<f32>` + shape for persistence.
#[derive(Debug, Clone, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
struct LayerWeightsOwned {
    /// Row-major flattened weight matrix.
    w: Vec<f32>,
    /// Shape `[out_dim, in_dim]`.
    w_shape: [usize; 2],
    /// Bias vector.
    b: Vec<f32>,
}

impl LayerWeightsOwned {
    fn from_view(lw: &LayerWeights) -> Self {
        let (out_dim, in_dim) = lw.w.dim();
        // Array2 is row-major (standard layout); iter() walks in row-major order.
        let w = lw.w.iter().copied().collect::<Vec<f32>>();
        let b = lw.b.iter().copied().collect::<Vec<f32>>();
        Self {
            w,
            w_shape: [out_dim, in_dim],
            b,
        }
    }
}

/// Event emitted on every SGD step.
///
/// Records per-layer weight-update magnitude so an observer (audit log,
/// monitoring dashboard, replay tool) can reconstruct training dynamics
/// without seeing every individual weight.
///
/// # Phase 1 note
///
/// This type implements [`Event`] but is only logged via `tracing` today;
/// the event log wiring lands alongside the scheduler integration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightUpdateEvent {
    /// MSE loss observed **before** the update.
    pub loss: f32,
    /// Learning rate used for this step.
    pub lr: f32,
    /// `||lr * grad_w||_2` for each layer, ordered first-layer-first.
    pub per_layer_delta_norm: Vec<f32>,
}

/// The state mutated by weight-update events: a bare summary of total
/// movement + count. The real weights live on the actor; for the `Event`
/// contract we only need a minimal materialized view.
#[derive(Debug, Default)]
pub struct WeightUpdateStats {
    /// Number of updates applied.
    pub steps: u64,
    /// Sum of last-step per-layer delta norms.
    pub last_total_delta: f32,
    /// Last observed loss.
    pub last_loss: f32,
}

impl Event for WeightUpdateEvent {
    type State = WeightUpdateStats;

    fn apply(self, state: &mut Self::State) {
        state.steps = state.steps.saturating_add(1);
        state.last_total_delta = self.per_layer_delta_norm.iter().sum();
        state.last_loss = self.loss;
    }
}

/// Initialize one layer with Kaiming-like scaling (ReLU-friendly).
fn init_layer(in_dim: usize, out_dim: usize, rng: &mut ChaCha20Rng) -> LayerWeights {
    use rand::distr::{Distribution, Uniform};
    let scale = (2.0_f32 / in_dim as f32).sqrt();
    let dist = Uniform::new(-scale, scale).expect("valid uniform range");
    let w = Array2::from_shape_fn((out_dim, in_dim), |_| dist.sample(rng));
    let b = Array1::zeros(out_dim);
    LayerWeights { w, b }
}

/// Apply activation in place.
fn apply_activation(v: &mut Array1<f32>, kind: Activation) {
    match kind {
        Activation::Relu => v.iter_mut().for_each(|x| *x = x.max(0.0)),
        Activation::Tanh => v.iter_mut().for_each(|x| *x = x.tanh()),
    }
}

/// Derivative of activation evaluated at pre-activation value `z`.
fn activation_deriv(z: f32, kind: Activation) -> f32 {
    match kind {
        Activation::Relu => {
            if z > 0.0 {
                1.0
            } else {
                0.0
            }
        }
        Activation::Tanh => {
            let t = z.tanh();
            1.0 - t * t
        }
    }
}

/// Outer product `a (x) b` → shape (a.len(), b.len()).
fn outer(a: &Array1<f32>, b: &Array1<f32>) -> Array2<f32> {
    let rows = a.len();
    let cols = b.len();
    Array2::from_shape_fn((rows, cols), |(i, j)| a[i] * b[j])
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
            assert_eq!(la.w.shape(), lb.w.shape());
            for (va, vb) in la.w.iter().zip(lb.w.iter()) {
                assert_eq!(va.to_bits(), vb.to_bits(), "weight init diverged");
            }
            for (va, vb) in la.b.iter().zip(lb.b.iter()) {
                assert_eq!(va.to_bits(), vb.to_bits(), "bias init diverged");
            }
        }
    }

    #[test]
    fn forward_shape_is_correct() {
        let net = AdaptiveNet::new(AdaptiveConfig {
            layers: vec![4, 8, 3],
            rng_seed: 1,
            activation: Activation::Relu,
        });
        let x = Array1::from_vec(vec![1.0, -2.0, 0.5, 3.0]);
        let out = net.forward(&x);
        assert_eq!(out.shape(), &[3]);
    }

    #[test]
    fn xor_converges() {
        // 2 -> 8 -> 1 MLP, tanh. XOR is the classic non-linear test.
        let mut net = AdaptiveNet::new(AdaptiveConfig {
            layers: vec![2, 8, 1],
            rng_seed: 0x0BAD_C0FF_EE0D_DF00,
            activation: Activation::Tanh,
        });

        let samples: [(Array1<f32>, Array1<f32>); 4] = [
            (
                Array1::from_vec(vec![0.0, 0.0]),
                Array1::from_vec(vec![0.0]),
            ),
            (
                Array1::from_vec(vec![0.0, 1.0]),
                Array1::from_vec(vec![1.0]),
            ),
            (
                Array1::from_vec(vec![1.0, 0.0]),
                Array1::from_vec(vec![1.0]),
            ),
            (
                Array1::from_vec(vec![1.0, 1.0]),
                Array1::from_vec(vec![0.0]),
            ),
        ];

        let lr = 0.1_f32;
        let mut final_mean_loss = f32::INFINITY;
        for step in 0..5000 {
            let mut mean = 0.0_f32;
            for (x, y) in &samples {
                mean += net.learn(x, y, lr);
            }
            mean /= samples.len() as f32;
            final_mean_loss = mean;
            if mean < 0.05 {
                // Converged early.
                eprintln!("XOR converged at step {step} with mean loss {mean:.6}");
                break;
            }
        }
        assert!(
            final_mean_loss < 0.05,
            "XOR did not converge: final mean loss = {final_mean_loss}"
        );
    }

    #[test]
    fn same_seed_same_loss_trajectory() {
        let cfg = AdaptiveConfig {
            layers: vec![3, 6, 2],
            rng_seed: 123_456,
            activation: Activation::Tanh,
        };
        let mut a = AdaptiveNet::new(cfg.clone());
        let mut b = AdaptiveNet::new(cfg);
        // Deterministic input/target stream.
        let mut rng = ChaCha20Rng::seed_from_u64(7);
        use rand::distr::{Distribution, Uniform};
        let u = Uniform::new(-1.0_f32, 1.0).expect("valid range");

        let mut losses_a: Vec<f32> = Vec::with_capacity(50);
        let mut losses_b: Vec<f32> = Vec::with_capacity(50);
        for _ in 0..50 {
            let x = Array1::from_shape_fn(3, |_| u.sample(&mut rng));
            let y = Array1::from_shape_fn(2, |_| u.sample(&mut rng));
            losses_a.push(a.learn(&x, &y, 0.05));
            losses_b.push(b.learn(&x, &y, 0.05));
        }
        // Bit-for-bit equality is the entire point of this test. We intentionally
        // rely on f32 `==` here — see CONTRIBUTING_V2.md §5 on determinism.
        assert_eq!(losses_a.len(), losses_b.len());
        for (la, lb) in losses_a.iter().zip(losses_b.iter()) {
            assert_eq!(la.to_bits(), lb.to_bits(), "loss trajectory divergence");
        }
    }

    #[test]
    fn save_load_round_trip() {
        let cfg = AdaptiveConfig {
            layers: vec![3, 5, 2],
            rng_seed: 2024,
            activation: Activation::Relu,
        };
        let mut net = AdaptiveNet::new(cfg.clone());

        // Train for 100 steps on a synthetic dataset.
        let mut rng = ChaCha20Rng::seed_from_u64(99);
        use rand::distr::{Distribution, Uniform};
        let u = Uniform::new(-1.0_f32, 1.0).expect("valid range");
        for _ in 0..100 {
            let x = Array1::from_shape_fn(3, |_| u.sample(&mut rng));
            let y = Array1::from_shape_fn(2, |_| u.sample(&mut rng));
            net.learn(&x, &y, 0.01);
        }

        let bytes = net.save().expect("save ok");
        // Load into a *fresh* net with same config.
        let mut loaded = AdaptiveNet::new(cfg);
        loaded.load(&bytes).expect("load ok");

        // Forward outputs must match for a handful of inputs.
        for _ in 0..5 {
            let x = Array1::from_shape_fn(3, |_| u.sample(&mut rng));
            let a = net.forward(&x);
            let b = loaded.forward(&x);
            for (va, vb) in a.iter().zip(b.iter()) {
                assert!(
                    (va - vb).abs() < 1e-7,
                    "forward mismatch after load: {va} vs {vb}"
                );
            }
        }
    }

    #[test]
    fn gradient_check_finite_differences() {
        // Tiny net: 3 -> 4 -> 2, tanh hidden (smooth).
        let cfg = AdaptiveConfig {
            layers: vec![3, 4, 2],
            rng_seed: 314_159,
            activation: Activation::Tanh,
        };
        let net = AdaptiveNet::new(cfg);

        let x = Array1::from_vec(vec![0.3_f32, -0.7, 0.1]);
        let y = Array1::from_vec(vec![0.2_f32, -0.5]);

        // Analytical gradients: compute via a one-step "learn at lr=0" pattern.
        // Since `learn` applies lr*grad, we instead reconstruct grads by hand
        // using the same math but without mutating weights.
        let (grad_w, grad_b) = analytical_grads(&net, &x, &y);

        // Finite-difference check on a sample of entries.
        let eps = 1e-3_f32;
        let tol = 1e-3_f32;

        for layer in 0..net.weights.len() {
            let (out_dim, in_dim) = net.weights[layer].w.dim();
            // Check a few entries to keep the test fast.
            let positions: Vec<(usize, usize)> = (0..out_dim)
                .flat_map(|i| (0..in_dim).map(move |j| (i, j)))
                .step_by(3)
                .collect();
            for (i, j) in positions {
                let mut plus = clone_net(&net);
                plus.weights[layer].w[[i, j]] += eps;
                let loss_plus = mse(&plus.forward(&x), &y);

                let mut minus = clone_net(&net);
                minus.weights[layer].w[[i, j]] -= eps;
                let loss_minus = mse(&minus.forward(&x), &y);

                let numeric = (loss_plus - loss_minus) / (2.0 * eps);
                let analytic = grad_w[layer][[i, j]];
                assert!(
                    (numeric - analytic).abs() < tol,
                    "w grad mismatch layer={layer} [{i},{j}]: numeric={numeric} analytic={analytic}"
                );
            }

            #[allow(clippy::needless_range_loop)]
            for k in 0..out_dim {
                let mut plus = clone_net(&net);
                plus.weights[layer].b[k] += eps;
                let loss_plus = mse(&plus.forward(&x), &y);

                let mut minus = clone_net(&net);
                minus.weights[layer].b[k] -= eps;
                let loss_minus = mse(&minus.forward(&x), &y);

                let numeric = (loss_plus - loss_minus) / (2.0 * eps);
                let analytic = grad_b[layer][k];
                assert!(
                    (numeric - analytic).abs() < tol,
                    "b grad mismatch layer={layer} [{k}]: numeric={numeric} analytic={analytic}"
                );
            }
        }
    }

    #[test]
    fn weight_update_event_applies() {
        let mut stats = WeightUpdateStats::default();
        WeightUpdateEvent {
            loss: 0.5,
            lr: 0.01,
            per_layer_delta_norm: vec![0.1, 0.2, 0.3],
        }
        .apply(&mut stats);
        assert_eq!(stats.steps, 1);
        assert!((stats.last_total_delta - 0.6).abs() < 1e-6);
        assert!((stats.last_loss - 0.5).abs() < 1e-6);
    }

    // ---------- helpers for the gradient-check test ----------

    fn mse(pred: &Array1<f32>, y: &Array1<f32>) -> f32 {
        let n = pred.len() as f32;
        pred.iter()
            .zip(y.iter())
            .map(|(p, t)| (p - t) * (p - t))
            .sum::<f32>()
            / n
    }

    fn clone_net(src: &AdaptiveNet) -> AdaptiveNet {
        AdaptiveNet {
            config: src.config.clone(),
            weights: src.weights.clone(),
        }
    }

    /// Pure analytical gradient computation (no weight mutation).
    fn analytical_grads(
        net: &AdaptiveNet,
        x: &Array1<f32>,
        y: &Array1<f32>,
    ) -> (Vec<Array2<f32>>, Vec<Array1<f32>>) {
        let n_layers = net.weights.len();
        let mut activations: Vec<Array1<f32>> = Vec::with_capacity(n_layers + 1);
        let mut pre: Vec<Array1<f32>> = Vec::with_capacity(n_layers);
        activations.push(x.clone());
        let last = n_layers - 1;
        for (i, layer) in net.weights.iter().enumerate() {
            let z = layer.w.dot(&activations[i]) + &layer.b;
            pre.push(z.clone());
            if i != last {
                let mut h = z;
                apply_activation(&mut h, net.config.activation);
                activations.push(h);
            } else {
                activations.push(z);
            }
        }
        let pred = activations.last().expect("at least one layer").clone();
        let n_out = pred.len() as f32;
        let diff = &pred - y;
        let mut grad_z = diff.mapv(|d| 2.0 * d / n_out);

        let mut grad_w: Vec<Option<Array2<f32>>> = vec![None; n_layers];
        let mut grad_b: Vec<Option<Array1<f32>>> = vec![None; n_layers];

        for i in (0..n_layers).rev() {
            let input = &activations[i];
            grad_w[i] = Some(outer(&grad_z, input));
            grad_b[i] = Some(grad_z.clone());
            if i > 0 {
                let w_prev = &net.weights[i].w;
                let grad_h_prev = w_prev.t().dot(&grad_z);
                let z_prev = &pre[i - 1];
                let mut new_grad_z = Array1::<f32>::zeros(z_prev.len());
                for (k, (&gh, &zp)) in grad_h_prev.iter().zip(z_prev.iter()).enumerate() {
                    new_grad_z[k] = gh * activation_deriv(zp, net.config.activation);
                }
                grad_z = new_grad_z;
            }
        }
        (
            grad_w.into_iter().map(|o| o.expect("filled")).collect(),
            grad_b.into_iter().map(|o| o.expect("filled")).collect(),
        )
    }
}
