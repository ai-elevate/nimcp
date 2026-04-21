//! Phase 4a — top-level [`LnnNetwork`] composition.
//!
//! Stacks one or more [`LtcLayer`]s and appends a linear readout. This
//! is the forward-pass-only shape; Phase 4b adds BPTT + SGD.
//!
//! # Forward
//!
//! For input sequence `u_1, u_2, ...`:
//!
//! ```text
//! state_0 = 0
//! for t in 1..=T:
//!   h = u_t
//!   for L in layers:
//!       state_L <- ltc_forward_step(state_L, L, h, dt_ms)
//!       h = state_L.x         // layer output = layer state
//!   y_t = W_out · h + b_out   // linear readout
//! ```
//!
//! The readout is dense (no nonlinearity) — downstream tasks can compose
//! their own (softmax for classification, identity for regression).
//!
//! # Checkpointing
//!
//! `serde` round-trips the whole network (layers + readout). State is
//! not part of the checkpoint — it's transient and resets to zeros on
//! `reset()`.

use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::distr::{Distribution, Uniform};
use rand_chacha::ChaCha20Rng;
use serde::{Deserialize, Serialize};

use crate::ltc::{LtcLayer, LtcParams, LtcState, ltc_forward_step};

/// Config for building an [`LnnNetwork`].
///
/// Sizes are validated against adjacency: `layers[k].n_in` must equal
/// `layers[k-1].n_rec` (the hidden state of the previous layer is the
/// input to the next), with `layers[0].n_in = input_dim`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LnnConfig {
    /// Network input dimension.
    pub input_dim: usize,
    /// Network output dimension (readout width).
    pub output_dim: usize,
    /// Per-layer LTC params. `layers.len() >= 1`.
    pub layers: Vec<LtcParams>,
    /// Deterministic seed for weight init.
    pub rng_seed: u64,
    /// Integration timestep, ms.
    pub dt_ms: f32,
}

impl Default for LnnConfig {
    fn default() -> Self {
        Self {
            input_dim: 0,
            output_dim: 0,
            layers: Vec::new(),
            rng_seed: 0,
            dt_ms: 1.0,
        }
    }
}

/// Construction error for [`LnnNetwork`].
#[derive(Debug, thiserror::Error)]
pub enum LnnError {
    /// No layers declared.
    #[error("config declares zero layers; at least one required")]
    NoLayers,
    /// `layers[0].n_in` does not match the declared `input_dim`.
    #[error("layer 0 n_in ({got}) does not match input_dim ({expected})")]
    InputDimMismatch {
        /// Value found on `layers[0].n_in`.
        got: usize,
        /// Expected value from `input_dim`.
        expected: usize,
    },
    /// Two consecutive layers disagree on the hidden-state dimension.
    #[error("layer {layer_idx} n_in ({got}) does not match previous layer n_rec ({expected})")]
    LayerAdjacencyMismatch {
        /// Offending layer index.
        layer_idx: usize,
        /// Value on the offending layer's `n_in`.
        got: usize,
        /// Expected value (previous layer's `n_rec`).
        expected: usize,
    },
}

/// Stacked LTC layers + linear readout.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LnnNetwork {
    /// Per-layer parameters.
    pub layers: Vec<LtcLayer>,
    /// Readout weights `(output_dim, last_layer.n_rec)`.
    pub w_out: Array2<f32>,
    /// Readout bias `(output_dim,)`.
    pub b_out: Array1<f32>,
    /// Integration timestep, ms.
    pub dt_ms: f32,
    /// Input dimension (copy of the build-time config).
    pub input_dim: usize,
    /// Output dimension (copy of the build-time config).
    pub output_dim: usize,
}

impl LnnNetwork {
    /// Construct from `config`. Validates layer adjacency and seeds the
    /// weights deterministically.
    #[allow(clippy::needless_pass_by_value)] // `config.layers` is consumed.
    pub fn new(config: LnnConfig) -> Result<Self, LnnError> {
        if config.layers.is_empty() {
            return Err(LnnError::NoLayers);
        }
        if config.layers[0].n_in != config.input_dim {
            return Err(LnnError::InputDimMismatch {
                got: config.layers[0].n_in,
                expected: config.input_dim,
            });
        }
        for (idx, win) in config.layers.windows(2).enumerate() {
            let prev_n_rec = win[0].n_rec;
            let next_n_in = win[1].n_in;
            if prev_n_rec != next_n_in {
                return Err(LnnError::LayerAdjacencyMismatch {
                    layer_idx: idx + 1,
                    got: next_n_in,
                    expected: prev_n_rec,
                });
            }
        }

        // Seed weights — use `rng_seed` for layers, `rng_seed ^ 0xFE..`
        // for the readout so changing the number of layers doesn't
        // perturb the readout's RNG draw.
        let layers: Vec<LtcLayer> = config
            .layers
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let layer_seed = config
                    .rng_seed
                    .wrapping_add(i as u64)
                    .wrapping_mul(0x9E37_79B9);
                LtcLayer::new_seeded(*p, layer_seed)
            })
            .collect();

        let last_n_rec = layers.last().expect("checked non-empty").params.n_rec;

        let readout_seed = config.rng_seed ^ 0xFEED_FACE_DEAD_BEEF;
        let mut rng = ChaCha20Rng::seed_from_u64(readout_seed);
        let bound = if last_n_rec == 0 {
            0.0
        } else {
            1.0_f32 / (last_n_rec as f32).sqrt()
        };
        let w_out = if bound > 0.0 {
            let uni = Uniform::new(-bound, bound).expect("bound positive");
            Array2::from_shape_fn((config.output_dim, last_n_rec), |_| uni.sample(&mut rng))
        } else {
            Array2::zeros((config.output_dim, last_n_rec))
        };
        let b_out = Array1::zeros(config.output_dim);

        Ok(Self {
            layers,
            w_out,
            b_out,
            dt_ms: config.dt_ms,
            input_dim: config.input_dim,
            output_dim: config.output_dim,
        })
    }

    /// Allocate fresh zeroed per-layer state vectors.
    #[must_use]
    pub fn new_state(&self) -> Vec<LtcState> {
        self.layers
            .iter()
            .map(|l| LtcState::new(l.params.n_rec))
            .collect()
    }

    /// Advance the network by one timestep.
    ///
    /// `state` must be the vector returned by [`new_state`](Self::new_state)
    /// (one [`LtcState`] per layer, in order). `input` must have length
    /// [`input_dim`](Self::input_dim).
    ///
    /// Returns the readout `y = W_out · h + b_out`, where `h` is the
    /// final layer's updated state vector.
    pub fn forward_step(&self, state: &mut [LtcState], input: &Array1<f32>) -> Array1<f32> {
        debug_assert_eq!(state.len(), self.layers.len(), "state/layer count mismatch");
        debug_assert_eq!(input.len(), self.input_dim, "input dim mismatch");

        // Walk layers forward, passing each layer's post-step state as
        // the next layer's input. `curr_input` is owned because we need
        // to rebind it to each layer's state as we go.
        let mut curr_input: Array1<f32> = input.clone();
        for (layer, st) in self.layers.iter().zip(state.iter_mut()) {
            let _ = ltc_forward_step(st, layer, &curr_input, self.dt_ms);
            curr_input = st.x.clone();
        }

        // Readout: W_out · h + b_out.
        let mut y = self.w_out.dot(&curr_input);
        y += &self.b_out;
        y
    }

    /// Advance the network over a sequence of `T` inputs; returns the
    /// readout at every timestep (shape `(T, output_dim)`). Useful for
    /// sequence-classification and regression tasks.
    #[must_use]
    pub fn forward_sequence(&self, inputs: &[Array1<f32>]) -> Vec<Array1<f32>> {
        let mut state = self.new_state();
        let mut out = Vec::with_capacity(inputs.len());
        for u in inputs {
            out.push(self.forward_step(&mut state, u));
        }
        out
    }
}

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    fn two_layer_cfg(input_dim: usize, hidden: usize, output_dim: usize) -> LnnConfig {
        LnnConfig {
            input_dim,
            output_dim,
            layers: vec![
                LtcParams {
                    n_in: input_dim,
                    n_rec: hidden,
                    tau_init: 1.0,
                    init_scale: 1.0,
                },
                LtcParams {
                    n_in: hidden,
                    n_rec: hidden,
                    tau_init: 1.0,
                    init_scale: 1.0,
                },
            ],
            rng_seed: 0x1234,
            dt_ms: 0.1,
        }
    }

    #[test]
    fn builds_valid_config() {
        let net = LnnNetwork::new(two_layer_cfg(3, 8, 2)).expect("build");
        assert_eq!(net.layers.len(), 2);
        assert_eq!(net.w_out.shape(), &[2, 8]);
        assert_eq!(net.b_out.len(), 2);
    }

    #[test]
    fn rejects_empty_layers() {
        let cfg = LnnConfig {
            input_dim: 2,
            output_dim: 1,
            layers: vec![],
            rng_seed: 0,
            dt_ms: 1.0,
        };
        match LnnNetwork::new(cfg) {
            Err(LnnError::NoLayers) => {}
            other => panic!("expected NoLayers, got {other:?}"),
        }
    }

    #[test]
    fn rejects_input_dim_mismatch() {
        let mut cfg = two_layer_cfg(3, 8, 2);
        cfg.input_dim = 5;
        match LnnNetwork::new(cfg) {
            Err(LnnError::InputDimMismatch { got, expected }) => {
                assert_eq!(got, 3);
                assert_eq!(expected, 5);
            }
            other => panic!("expected InputDimMismatch, got {other:?}"),
        }
    }

    #[test]
    fn rejects_layer_adjacency_mismatch() {
        let mut cfg = two_layer_cfg(3, 8, 2);
        cfg.layers[1].n_in = 999;
        match LnnNetwork::new(cfg) {
            Err(LnnError::LayerAdjacencyMismatch {
                layer_idx,
                got,
                expected,
            }) => {
                assert_eq!(layer_idx, 1);
                assert_eq!(got, 999);
                assert_eq!(expected, 8);
            }
            other => panic!("expected LayerAdjacencyMismatch, got {other:?}"),
        }
    }

    #[test]
    fn forward_is_deterministic_on_same_seed() {
        let a = LnnNetwork::new(two_layer_cfg(3, 8, 2)).expect("a");
        let b = LnnNetwork::new(two_layer_cfg(3, 8, 2)).expect("b");
        let inputs: Vec<Array1<f32>> = (0..20)
            .map(|t| Array1::from_elem(3, (t as f32).sin()))
            .collect();
        let ya = a.forward_sequence(&inputs);
        let yb = b.forward_sequence(&inputs);
        assert_eq!(ya.len(), yb.len());
        for (a_t, b_t) in ya.iter().zip(yb.iter()) {
            assert_eq!(a_t, b_t);
        }
    }

    #[test]
    fn forward_shape_is_output_dim() {
        let net = LnnNetwork::new(two_layer_cfg(3, 8, 5)).expect("build");
        let inputs: Vec<Array1<f32>> = (0..4).map(|_| Array1::zeros(3)).collect();
        let out = net.forward_sequence(&inputs);
        assert_eq!(out.len(), 4);
        for y in &out {
            assert_eq!(y.len(), 5);
        }
    }

    #[test]
    fn state_is_non_zero_after_drive() {
        let net = LnnNetwork::new(two_layer_cfg(2, 8, 1)).expect("build");
        let mut state = net.new_state();
        let u = Array1::from_elem(2, 1.0_f32);
        for _ in 0..30 {
            net.forward_step(&mut state, &u);
        }
        let l0_norm = state[0].x.iter().map(|v| v * v).sum::<f32>().sqrt();
        let l1_norm = state[1].x.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(l0_norm > 0.1, "layer 0 state did not grow: {l0_norm}");
        assert!(l1_norm > 0.0, "layer 1 state did not grow: {l1_norm}");
    }

    #[test]
    fn forward_stays_finite_on_long_noise_sequence() {
        // 2000-step noise sequence — V1 used to NaN by step ~600 without
        // state clamping. V2 regression guard.
        let net = LnnNetwork::new(two_layer_cfg(4, 16, 2)).expect("build");
        let mut state = net.new_state();
        let mut rng = ChaCha20Rng::seed_from_u64(0xDEAD_BEEF);
        let uni = Uniform::new(-3.0_f32, 3.0_f32).expect("bound positive");
        for _ in 0..2000 {
            let u: Array1<f32> = Array1::from_shape_fn(4, |_| uni.sample(&mut rng));
            let y = net.forward_step(&mut state, &u);
            assert!(y.iter().all(|v| v.is_finite()), "readout went NaN/Inf");
            assert!(
                state.iter().all(|s| s.x.iter().all(|v| v.is_finite())),
                "state went NaN/Inf"
            );
        }
    }

    #[test]
    fn serde_round_trip_preserves_weights_and_output() {
        use serde_json;

        let original = LnnNetwork::new(two_layer_cfg(3, 8, 2)).expect("build");
        let json = serde_json::to_string(&original).expect("serialize");
        let restored: LnnNetwork = serde_json::from_str(&json).expect("deserialize");

        // Same weights.
        assert_eq!(original.w_out, restored.w_out);
        assert_eq!(original.b_out, restored.b_out);
        assert_eq!(original.layers.len(), restored.layers.len());
        for (a, b) in original.layers.iter().zip(restored.layers.iter()) {
            assert_eq!(a.w_rec, b.w_rec);
            assert_eq!(a.w_in, b.w_in);
            assert_eq!(a.b, b.b);
            assert_eq!(a.tau_base, b.tau_base);
        }

        // Same forward output on identical inputs.
        let inputs: Vec<Array1<f32>> = (0..10)
            .map(|t| Array1::from_elem(3, (t as f32 * 0.3).cos()))
            .collect();
        let y_orig = original.forward_sequence(&inputs);
        let y_new = restored.forward_sequence(&inputs);
        assert_eq!(y_orig, y_new);
    }
}
