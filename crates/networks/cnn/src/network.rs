//! Top-level [`CnnNetwork`] — stacks a heterogeneous list of layers
//! and runs them in declared order.
//!
//! # Spec → network
//!
//! [`CnnConfig`] describes the network as a `Vec<CnnLayerSpec>`; the
//! constructor walks the spec, allocates each layer with a sub-seed
//! derived from the network seed, and threads the running shape through
//! to validate adjacency.
//!
//! # Forward contract
//!
//! - Input must be a 4-D `[batch, in_channels, height, width]` tensor
//!   matching the config's `input_shape`.
//! - Output is the dense vector `[batch, output_dim]` (the last linear
//!   layer's width).
//! - The forward pass holds a single live tensor at any moment; no
//!   activations are cached (training is a follow-up phase).
//!
//! # V1 lessons carried forward
//!
//! - Deterministic seed everywhere — same `rng_seed` produces bit-
//!   identical weights across runs and platforms.
//! - Shape errors caught at config time rather than at first forward —
//!   `CnnNetwork::new()` returns a [`CnnError::ShapeMismatch`] if any
//!   layer's expected input doesn't match the previous layer's output.

use ndarray::{Array2, Array4};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::activation::ReluLayer;
use crate::conv::Conv2dLayer;
use crate::flatten::FlattenLayer;
use crate::linear::LinearLayer;
use crate::pool::MaxPool2dLayer;

/// Declarative spec for a single CNN layer. The constructor maps each
/// variant to its concrete `*Layer` type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CnnLayerSpec {
    /// `(out_channels, kh, kw, stride, padding)`.
    Conv2d {
        out_channels: usize,
        kh: usize,
        kw: usize,
        stride: usize,
        padding: usize,
    },
    /// `(kernel, stride)` — square pool with matching stride.
    MaxPool2d { kernel: usize, stride: usize },
    Relu,
    Flatten,
    /// Dense layer width. The constructor infers `in_features` from the
    /// running flattened shape; `out_features` is the spec value.
    Linear { out_features: usize },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CnnConfig {
    /// `(in_channels, height, width)` of the network input — the batch
    /// axis is flexible.
    pub input_shape: (usize, usize, usize),
    pub layers: Vec<CnnLayerSpec>,
    pub rng_seed: u64,
}

#[derive(Debug, Error)]
pub enum CnnError {
    #[error("cnn: empty layer list")]
    Empty,
    #[error("cnn: shape mismatch in layer {layer_index} ({hint})")]
    ShapeMismatch { layer_index: usize, hint: String },
    #[error(
        "cnn: linear-after-non-flatten in layer {0} — call FlattenLayer before any LinearLayer"
    )]
    LinearWithoutFlatten(usize),
    #[error("cnn: pool/conv on a flattened tensor in layer {0}")]
    SpatialAfterFlatten(usize),
}

/// Internal layer enum — the constructed counterpart to [`CnnLayerSpec`].
/// `serde` is derived so the whole network round-trips through JSON.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CnnLayer {
    Conv(Conv2dLayer),
    Pool(MaxPool2dLayer),
    Relu(ReluLayer),
    Flatten(FlattenLayer),
    Linear(LinearLayer),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CnnNetwork {
    pub input_shape: (usize, usize, usize),
    pub output_dim: usize,
    pub layers: Vec<CnnLayer>,
}

impl CnnNetwork {
    pub fn new(config: CnnConfig) -> Result<Self, CnnError> {
        if config.layers.is_empty() {
            return Err(CnnError::Empty);
        }
        let (mut c, mut h, mut w) = config.input_shape;
        let mut flat_dim: Option<usize> = None;
        let mut output_dim = 0;
        let mut layers: Vec<CnnLayer> = Vec::with_capacity(config.layers.len());

        for (idx, spec) in config.layers.iter().enumerate() {
            // Sub-seed each layer so reordering / inserting layers
            // doesn't shift every downstream init.
            let sub_seed = config.rng_seed.wrapping_add(idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            match spec {
                CnnLayerSpec::Conv2d {
                    out_channels,
                    kh,
                    kw,
                    stride,
                    padding,
                } => {
                    if flat_dim.is_some() {
                        return Err(CnnError::SpatialAfterFlatten(idx));
                    }
                    let layer = Conv2dLayer::new(
                        c,
                        *out_channels,
                        *kh,
                        *kw,
                        *stride,
                        *stride,
                        *padding,
                        *padding,
                        sub_seed,
                    );
                    let (h_out, w_out) = layer.output_hw(h, w);
                    if h_out == 0 || w_out == 0 {
                        return Err(CnnError::ShapeMismatch {
                            layer_index: idx,
                            hint: format!(
                                "conv produces zero spatial dim from {h}x{w} (kernel {kh}x{kw}, stride {stride}, pad {padding})"
                            ),
                        });
                    }
                    c = *out_channels;
                    h = h_out;
                    w = w_out;
                    layers.push(CnnLayer::Conv(layer));
                }
                CnnLayerSpec::MaxPool2d { kernel, stride } => {
                    if flat_dim.is_some() {
                        return Err(CnnError::SpatialAfterFlatten(idx));
                    }
                    let layer = MaxPool2dLayer::new(*kernel, *kernel, *stride, *stride);
                    let (h_out, w_out) = layer.output_hw(h, w);
                    if h_out == 0 || w_out == 0 {
                        return Err(CnnError::ShapeMismatch {
                            layer_index: idx,
                            hint: format!(
                                "pool produces zero spatial dim from {h}x{w} (kernel {kernel}, stride {stride})"
                            ),
                        });
                    }
                    h = h_out;
                    w = w_out;
                    layers.push(CnnLayer::Pool(layer));
                }
                CnnLayerSpec::Relu => {
                    layers.push(CnnLayer::Relu(ReluLayer::new()));
                }
                CnnLayerSpec::Flatten => {
                    if flat_dim.is_some() {
                        return Err(CnnError::ShapeMismatch {
                            layer_index: idx,
                            hint: "flatten called twice".into(),
                        });
                    }
                    flat_dim = Some(c * h * w);
                    layers.push(CnnLayer::Flatten(FlattenLayer::new()));
                }
                CnnLayerSpec::Linear { out_features } => {
                    let in_f = flat_dim.ok_or(CnnError::LinearWithoutFlatten(idx))?;
                    let layer = LinearLayer::new(in_f, *out_features, sub_seed);
                    flat_dim = Some(*out_features);
                    output_dim = *out_features;
                    layers.push(CnnLayer::Linear(layer));
                }
            }
        }

        if output_dim == 0 {
            // Network ended on a non-Linear; output is whatever the
            // last spatial / flat layer produced. Treat the running
            // flattened size (if flatten ran) or c*h*w as the dim.
            output_dim = flat_dim.unwrap_or(c * h * w);
        }

        Ok(Self {
            input_shape: config.input_shape,
            output_dim,
            layers,
        })
    }

    /// Forward over a 4-D input. Returns `[batch, output_dim]`.
    ///
    /// # Panics
    /// If `input.shape()[1..]` doesn't match `self.input_shape`.
    pub fn forward(&self, input: &Array4<f32>) -> Array2<f32> {
        let (_n, c, h, w) = input.dim();
        assert_eq!(
            (c, h, w),
            self.input_shape,
            "cnn input shape mismatch (expected {:?}, got {:?})",
            self.input_shape,
            (c, h, w)
        );

        // Carry both possible representations — at most one is "live"
        // at a time, transitioning at the Flatten layer.
        let mut spatial: Option<Array4<f32>> = Some(input.clone());
        let mut flat: Option<Array2<f32>> = None;

        for layer in &self.layers {
            match layer {
                CnnLayer::Conv(c) => {
                    let cur = spatial.take().expect("conv after flatten");
                    spatial = Some(c.forward(&cur));
                }
                CnnLayer::Pool(p) => {
                    let cur = spatial.take().expect("pool after flatten");
                    spatial = Some(p.forward(&cur));
                }
                CnnLayer::Relu(r) => {
                    if let Some(cur) = spatial.take() {
                        spatial = Some(r.forward(&cur));
                    } else if let Some(cur) = flat.take() {
                        // Apply ReLU on flat tensor too — promote to a
                        // 4-D shape `[N, C, 1, 1]`-equivalent isn't
                        // needed; do it in place.
                        flat = Some(cur.mapv(|x| if x > 0.0 { x } else { 0.0 }));
                    }
                }
                CnnLayer::Flatten(f) => {
                    let cur = spatial.take().expect("flatten without spatial");
                    flat = Some(f.forward(&cur));
                }
                CnnLayer::Linear(l) => {
                    let cur = flat.take().expect("linear without flatten");
                    flat = Some(l.forward(&cur));
                }
            }
        }

        if let Some(f) = flat {
            f
        } else {
            // Network ended on a spatial layer — flatten implicitly so
            // the caller always sees `[batch, out_dim]`.
            let s = spatial.expect("forward produced no output");
            FlattenLayer::new().forward(&s)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array4;

    #[test]
    fn deterministic_init_round_trips_bit_for_bit() {
        let cfg = CnnConfig {
            input_shape: (1, 8, 8),
            layers: vec![
                CnnLayerSpec::Conv2d {
                    out_channels: 4,
                    kh: 3,
                    kw: 3,
                    stride: 1,
                    padding: 1,
                },
                CnnLayerSpec::Relu,
                CnnLayerSpec::MaxPool2d { kernel: 2, stride: 2 },
                CnnLayerSpec::Flatten,
                CnnLayerSpec::Linear { out_features: 5 },
            ],
            rng_seed: 0xC1A551C,
        };
        let a = CnnNetwork::new(cfg.clone()).unwrap();
        let b = CnnNetwork::new(cfg).unwrap();

        // Compare conv weights.
        let (CnnLayer::Conv(ca), CnnLayer::Conv(cb)) = (&a.layers[0], &b.layers[0]) else {
            panic!("conv expected at index 0");
        };
        assert_eq!(ca.weight, cb.weight, "conv weights differ across builds");

        // Linear weights.
        let (CnnLayer::Linear(la), CnnLayer::Linear(lb)) = (&a.layers[4], &b.layers[4]) else {
            panic!("linear expected at index 4");
        };
        assert_eq!(la.weight, lb.weight, "linear weights differ across builds");
    }

    #[test]
    fn conv_output_shape_matches_formula() {
        let layer = Conv2dLayer::new(3, 8, 3, 3, 1, 1, 1, 1, 0xABC);
        let inp = Array4::<f32>::zeros((2, 3, 16, 16));
        let out = layer.forward(&inp);
        assert_eq!(out.dim(), (2, 8, 16, 16));

        let strided = Conv2dLayer::new(3, 4, 3, 3, 2, 2, 0, 0, 0xDEF);
        let out2 = strided.forward(&inp);
        assert_eq!(out2.dim(), (2, 4, 7, 7));
    }

    #[test]
    fn maxpool_downsamples_correctly() {
        let pool = MaxPool2dLayer::square(2);
        let inp = Array4::from_shape_fn((1, 1, 4, 4), |(_, _, h, w)| (h * 4 + w) as f32);
        let out = pool.forward(&inp);
        assert_eq!(out.dim(), (1, 1, 2, 2));
        // 4×4 grid 0..15 with 2×2 max pool → corners 5, 7, 13, 15.
        assert_eq!(out[[0, 0, 0, 0]], 5.0);
        assert_eq!(out[[0, 0, 0, 1]], 7.0);
        assert_eq!(out[[0, 0, 1, 0]], 13.0);
        assert_eq!(out[[0, 0, 1, 1]], 15.0);
    }

    #[test]
    fn lenet_shaped_forward_is_finite() {
        // Conv → Pool → Conv → Pool → Flatten → Linear.
        let cfg = CnnConfig {
            input_shape: (1, 28, 28),
            layers: vec![
                CnnLayerSpec::Conv2d {
                    out_channels: 6,
                    kh: 5,
                    kw: 5,
                    stride: 1,
                    padding: 0,
                }, // 28 → 24
                CnnLayerSpec::Relu,
                CnnLayerSpec::MaxPool2d { kernel: 2, stride: 2 }, // 24 → 12
                CnnLayerSpec::Conv2d {
                    out_channels: 16,
                    kh: 5,
                    kw: 5,
                    stride: 1,
                    padding: 0,
                }, // 12 → 8
                CnnLayerSpec::Relu,
                CnnLayerSpec::MaxPool2d { kernel: 2, stride: 2 }, // 8 → 4
                CnnLayerSpec::Flatten,
                CnnLayerSpec::Linear { out_features: 10 },
            ],
            rng_seed: 0x1E_4E_57,
        };
        let net = CnnNetwork::new(cfg).unwrap();
        assert_eq!(net.output_dim, 10);

        // Random-ish input — deterministic via shape-fn so the test is
        // reproducible without dragging in `rand`.
        let inp = Array4::from_shape_fn((1, 1, 28, 28), |(_, _, h, w)| {
            ((h * 28 + w) as f32 * 0.001).sin()
        });
        let out = net.forward(&inp);
        assert_eq!(out.dim(), (1, 10));
        for v in out.iter() {
            assert!(v.is_finite(), "lenet forward produced non-finite: {v}");
        }
    }

    #[test]
    fn config_round_trips_through_serde() {
        let cfg = CnnConfig {
            input_shape: (3, 16, 16),
            layers: vec![
                CnnLayerSpec::Conv2d {
                    out_channels: 4,
                    kh: 3,
                    kw: 3,
                    stride: 1,
                    padding: 1,
                },
                CnnLayerSpec::Flatten,
                CnnLayerSpec::Linear { out_features: 2 },
            ],
            rng_seed: 99,
        };
        let net = CnnNetwork::new(cfg).unwrap();
        let json = serde_json::to_string(&net).unwrap();
        let restored: CnnNetwork = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.input_shape, net.input_shape);
        assert_eq!(restored.output_dim, net.output_dim);
        assert_eq!(restored.layers.len(), net.layers.len());
    }
}
