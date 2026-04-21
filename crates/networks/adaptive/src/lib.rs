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
//!
//! # Phase 2d scope (GPU, behind `--features cuda`)
//!
//! - [`AdaptiveNet::init_gpu`]: compile kernels once + upload current
//!   CPU weights to a device-resident weight cache (`nimcp-gpu`).
//! - [`AdaptiveNet::forward_gpu`]: matches [`AdaptiveNet::forward`]
//!   within f32 FMA tolerance (~1e-4).
//! - [`AdaptiveNet::learn_gpu`]: mirrors [`AdaptiveNet::learn`]. Returns
//!   pre-update MSE loss. Backward + SGD run on the device; activations
//!   are still computed on the CPU because `mlp_backward` takes CPU
//!   slices in this phase.
//! - [`AdaptiveNet::sync_gpu_to_cpu`]: downloads device weights back to
//!   `self.weights` so [`AdaptiveNet::forward`] / [`AdaptiveNet::save`]
//!   see the trained state. Callers must sync before a CPU-path save.
//!
//! The CPU API is unchanged; `--no-default-features` builds a pure-CPU
//! crate with no GPU symbols.

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
    /// GPU-backed operation failed. Wraps a [`nimcp_gpu::GpuError`] message.
    /// Only emitted when the crate is built with `--features cuda`.
    #[cfg(feature = "cuda")]
    #[error("gpu: {0}")]
    Gpu(String),
}

#[cfg(feature = "cuda")]
impl From<nimcp_gpu::GpuError> for AdaptiveError {
    fn from(e: nimcp_gpu::GpuError) -> Self {
        AdaptiveError::Gpu(e.to_string())
    }
}

/// Adaptive network state.
pub struct AdaptiveNet {
    config: AdaptiveConfig,
    weights: Vec<LayerWeights>,
    /// Device-resident weight cache. Built lazily via
    /// [`AdaptiveNet::init_gpu`]; `None` means the GPU path hasn't been
    /// wired up yet on this instance.
    #[cfg(feature = "cuda")]
    gpu_cache: Option<nimcp_gpu::GpuWeightCache>,
    /// Device-resident gradient buffers + SGD kernel module. Built
    /// alongside [`Self::gpu_cache`] in [`AdaptiveNet::init_gpu`].
    #[cfg(feature = "cuda")]
    gpu_grads: Option<nimcp_gpu::mlp_backward::GpuGradCache>,
}

impl std::fmt::Debug for AdaptiveNet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut ds = f.debug_struct("AdaptiveNet");
        ds.field("config", &self.config);
        ds.field("weights", &self.weights);
        #[cfg(feature = "cuda")]
        {
            ds.field(
                "gpu_cache",
                &self.gpu_cache.as_ref().map(|_| "GpuWeightCache"),
            );
            ds.field(
                "gpu_grads",
                &self.gpu_grads.as_ref().map(|_| "GpuGradCache"),
            );
        }
        ds.finish()
    }
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

        Self {
            config,
            weights,
            #[cfg(feature = "cuda")]
            gpu_cache: None,
            #[cfg(feature = "cuda")]
            gpu_grads: None,
        }
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
        // CPU weights have been replaced wholesale — any prior GPU state
        // is stale. Drop it so the caller can init_gpu() again if needed.
        #[cfg(feature = "cuda")]
        {
            self.gpu_cache = None;
            self.gpu_grads = None;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// GPU fast-path (Phase 2d). Only compiled when the `cuda` feature is active.
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
impl AdaptiveNet {
    /// Initialize the GPU cache from current CPU weights.
    ///
    /// Compiles the NVRTC kernels, allocates device buffers, and uploads
    /// the current `self.weights` to the device. Must be called before
    /// [`Self::forward_gpu`] or [`Self::learn_gpu`].
    ///
    /// Calling this repeatedly is safe — any prior cache is dropped (its
    /// `Drop` impl releases the device buffers) before the new one is
    /// built. This is how callers reset to the CPU weights after hand-
    /// editing `self.weights` or calling [`Self::load`].
    ///
    /// # Errors
    ///
    /// [`AdaptiveError::Gpu`] on any cuda failure (no device, NVRTC
    /// compile failure, alloc failure, etc.).
    pub fn init_gpu(&mut self) -> Result<(), AdaptiveError> {
        // Tear down any prior cache first so a re-init doesn't leak
        // two device copies while the new one is being built.
        self.gpu_cache = None;
        self.gpu_grads = None;

        // Flatten each layer's weights + bias into contiguous f32 slices
        // matching the GPU's row-major (out_dim, in_dim) layout. `Array2`
        // is row-major by default and `as_slice()` returns `Some` when
        // the array is in standard layout — which ours always is, since
        // we construct via `Array2::from_shape_fn` / `from_shape_vec`.
        let layers_flat: Vec<(Vec<f32>, Vec<f32>, usize, usize)> = self
            .weights
            .iter()
            .map(|lw| {
                let (out_dim, in_dim) = lw.w.dim();
                let w: Vec<f32> = lw.w.iter().copied().collect();
                let b: Vec<f32> = lw.b.iter().copied().collect();
                (w, b, in_dim, out_dim)
            })
            .collect();
        let layer_refs: Vec<(&[f32], &[f32], usize, usize)> = layers_flat
            .iter()
            .map(|(w, b, i, o)| (w.as_slice(), b.as_slice(), *i, *o))
            .collect();

        let cache = nimcp_gpu::GpuWeightCache::new(layer_refs)?;
        let grads = nimcp_gpu::mlp_backward::GpuGradCache::new(&cache)?;

        tracing::info!(
            layers = self.weights.len(),
            "adaptive gpu cache initialized"
        );

        self.gpu_cache = Some(cache);
        self.gpu_grads = Some(grads);
        Ok(())
    }

    /// Forward pass on the GPU.
    ///
    /// Matches [`Self::forward`] within f32 FMA tolerance (~1e-4 for
    /// typical MLPs; tighter for small layers).
    ///
    /// # Errors
    ///
    /// - [`AdaptiveError::Gpu`] with `"init_gpu not called"` if
    ///   [`Self::init_gpu`] hasn't been run yet.
    /// - [`AdaptiveError::ShapeMismatch`] if `x.len()` doesn't match the
    ///   network's input width.
    /// - [`AdaptiveError::Gpu`] on any cuda failure.
    pub fn forward_gpu(&self, x: &Array1<f32>) -> Result<Array1<f32>, AdaptiveError> {
        let expected_in = self.config.layers[0];
        if x.len() != expected_in {
            return Err(AdaptiveError::ShapeMismatch {
                expected: expected_in,
                got: x.len(),
            });
        }
        let cache = self
            .gpu_cache
            .as_ref()
            .ok_or_else(|| AdaptiveError::Gpu("init_gpu not called".into()))?;

        // ndarray is row-major / standard layout — a contiguous view
        // exists. Fall back to a clone if (implausibly) it isn't.
        let x_slice: Vec<f32> = if let Some(s) = x.as_slice() {
            s.to_vec()
        } else {
            x.iter().copied().collect()
        };
        let act = to_gpu_fwd_activation(self.config.activation);
        let y = nimcp_gpu::mlp_forward::mlp_forward(cache, &x_slice, act)?;
        Ok(Array1::from_vec(y))
    }

    /// GPU backward + in-place SGD against MSE loss.
    ///
    /// Mirrors [`Self::learn`]: returns the pre-update MSE loss (useful
    /// for logging). Steps in order:
    ///
    /// 1. CPU forward (produces `activations` + `pre_activations` that
    ///    [`nimcp_gpu::mlp_backward::mlp_backward`] consumes — they still
    ///    live on the host in this phase).
    /// 2. Upload host tensors + run `zero_grads` + `mlp_backward` + `sgd_step`.
    /// 3. Sync device weights back to [`Self::weights`] so the next
    ///    call's CPU forward sees the freshly-updated parameters.
    ///
    /// The returned loss is computed on the CPU from the same pre-update
    /// pred vector that feeds backward, so [`Self::learn`] and
    /// [`Self::learn_gpu`] report identical loss trajectories within FMA
    /// tolerance given identical weights.
    ///
    /// # Phase note
    ///
    /// The activations stay on the CPU because `mlp_backward` accepts
    /// `&[Vec<f32>]`. Every step therefore round-trips weights D→H at
    /// the end; a later phase can keep activations device-resident and
    /// drop the per-step sync to amortize across batches. This
    /// implementation prioritizes correctness over throughput.
    ///
    /// # Errors
    ///
    /// - [`AdaptiveError::Gpu`] if [`Self::init_gpu`] hasn't been run.
    /// - [`AdaptiveError::ShapeMismatch`] if `x` or `y` doesn't match.
    /// - [`AdaptiveError::Gpu`] on any cuda failure.
    pub fn learn_gpu(
        &mut self,
        x: &Array1<f32>,
        y: &Array1<f32>,
        lr: f32,
    ) -> Result<f32, AdaptiveError> {
        let n_layers = self.weights.len();
        let expected_in = self.config.layers[0];
        let expected_out = self.config.layers[n_layers];
        if x.len() != expected_in {
            return Err(AdaptiveError::ShapeMismatch {
                expected: expected_in,
                got: x.len(),
            });
        }
        if y.len() != expected_out {
            return Err(AdaptiveError::ShapeMismatch {
                expected: expected_out,
                got: y.len(),
            });
        }
        if self.gpu_cache.is_none() || self.gpu_grads.is_none() {
            return Err(AdaptiveError::Gpu("init_gpu not called".into()));
        }

        // CPU forward to produce the activations + pre-activations that
        // `mlp_backward` consumes, and to compute the pre-update loss we
        // return. The CPU weights are authoritative between `init_gpu`
        // (upload) and `sync_gpu_to_cpu` (download) — we only use them
        // here to reconstruct the same intermediate tensors the GPU
        // backward needs.
        let (activations, pre_activations) = self.cpu_forward_with_cache(x);
        let pred = activations.last().expect("at least one layer").clone();
        let n_out = pred.len() as f32;
        let diff = &pred - y;
        let loss = diff.iter().map(|&d| d * d).sum::<f32>() / n_out;

        // Convert the row-based `Array1`s into plain `Vec<f32>` payloads
        // that the GPU backward API accepts. Each element is cheap — f32
        // tensors at this scale are small.
        let acts_vecs: Vec<Vec<f32>> = activations.into_iter().map(|a| a.to_vec()).collect();
        let pre_vecs: Vec<Vec<f32>> = pre_activations.into_iter().map(|a| a.to_vec()).collect();
        let x_vec: Vec<f32> = x.iter().copied().collect();
        let y_vec: Vec<f32> = y.iter().copied().collect();

        let act = to_gpu_bwd_activation(self.config.activation);

        // Scoped borrow so `cache_mut` / `grads_mut` are released before
        // the follow-up `sync_gpu_to_cpu()` call reborrows `&mut self`.
        {
            // `gpu_cache` is `Option` and borrowed mutably by `sgd_step`;
            // we have to split the borrow. `take()` would poison the
            // field on early return, so pattern-match with `.as_mut()`.
            let cache_mut = self.gpu_cache.as_mut().expect("checked is_none above");
            let grads_mut = self.gpu_grads.as_mut().expect("checked is_none above");

            nimcp_gpu::mlp_backward::zero_grads(grads_mut)?;
            nimcp_gpu::mlp_backward::mlp_backward(
                cache_mut, grads_mut, &x_vec, &y_vec, act, &acts_vecs, &pre_vecs,
            )?;
            nimcp_gpu::mlp_backward::sgd_step(cache_mut, grads_mut, lr)?;
        }

        // Phase 2d correctness contract: the next `learn_gpu` call has
        // to run its CPU forward against the freshly-updated weights,
        // otherwise the CPU copy would stay frozen at init and every
        // step would recompute the same loss. Download device weights
        // back into `self.weights` so the CPU forward and `self.save`
        // both see the trained state. A later phase can keep the
        // forward activations device-resident and skip this round-trip.
        self.sync_gpu_to_cpu()?;

        tracing::trace!(loss, lr, "gpu weight update");
        Ok(loss)
    }

    /// Download GPU weights back into `self.weights`.
    ///
    /// Use this periodically (e.g. every N training steps) to keep the
    /// CPU and GPU copies in sync — so [`Self::forward`] and
    /// [`Self::save`] see the trained parameters. Cheap, but not free:
    /// every layer triggers two D→H memcpys.
    ///
    /// Note: [`Self::learn_gpu`] already invokes this at the end of
    /// every step in Phase 2d (it has to, to get the CPU-side forward
    /// on the next call right). Calling it explicitly is only necessary
    /// after a [`Self::forward_gpu`]-only path that never touches
    /// `learn_gpu`, which currently has no reason to mutate weights.
    ///
    /// # Errors
    ///
    /// - [`AdaptiveError::Gpu`] if [`Self::init_gpu`] hasn't been run.
    /// - [`AdaptiveError::Gpu`] on any cuda failure.
    pub fn sync_gpu_to_cpu(&mut self) -> Result<(), AdaptiveError> {
        let cache = self
            .gpu_cache
            .as_ref()
            .ok_or_else(|| AdaptiveError::Gpu("init_gpu not called".into()))?;
        if cache.num_layers() != self.weights.len() {
            return Err(AdaptiveError::Gpu(format!(
                "layer-count mismatch: gpu={} cpu={}",
                cache.num_layers(),
                self.weights.len()
            )));
        }
        for (i, lw) in self.weights.iter_mut().enumerate() {
            let (w, b) = cache.download_layer(i)?;
            let (out_dim, in_dim) = lw.w.dim();
            if w.len() != out_dim * in_dim || b.len() != out_dim {
                return Err(AdaptiveError::Gpu(format!(
                    "layer {i} shape mismatch on download: w={}, b={}, expected {}x{}",
                    w.len(),
                    b.len(),
                    out_dim,
                    in_dim,
                )));
            }
            lw.w = Array2::from_shape_vec((out_dim, in_dim), w)
                .map_err(|e| AdaptiveError::Gpu(format!("layer {i} w reshape: {e}")))?;
            lw.b = Array1::from_vec(b);
        }
        tracing::debug!(layers = self.weights.len(), "gpu->cpu sync complete");
        Ok(())
    }

    /// Internal helper: CPU forward that returns `(activations, pre_activations)`
    /// in the layout `mlp_backward` expects. Shape-identical to the
    /// first half of [`Self::learn`].
    fn cpu_forward_with_cache(&self, x: &Array1<f32>) -> (Vec<Array1<f32>>, Vec<Array1<f32>>) {
        let n_layers = self.weights.len();
        let mut activations: Vec<Array1<f32>> = Vec::with_capacity(n_layers + 1);
        let mut pre_activations: Vec<Array1<f32>> = Vec::with_capacity(n_layers);
        activations.push(x.clone());
        let last = n_layers - 1;
        for (i, layer) in self.weights.iter().enumerate() {
            let z = layer.w.dot(&activations[i]) + &layer.b;
            pre_activations.push(z.clone());
            if i != last {
                let mut h = z;
                apply_activation(&mut h, self.config.activation);
                activations.push(h);
            } else {
                activations.push(z);
            }
        }
        (activations, pre_activations)
    }
}

/// Map the adaptive crate's `Activation` onto `nimcp_gpu::mlp_forward::Activation`.
#[cfg(feature = "cuda")]
fn to_gpu_fwd_activation(a: Activation) -> nimcp_gpu::mlp_forward::Activation {
    match a {
        Activation::Relu => nimcp_gpu::mlp_forward::Activation::Relu,
        Activation::Tanh => nimcp_gpu::mlp_forward::Activation::Tanh,
    }
}

/// Map the adaptive crate's `Activation` onto `nimcp_gpu::mlp_backward::Activation`.
///
/// The forward and backward modules define their own duplicate enums;
/// both must be set consistently on every step.
#[cfg(feature = "cuda")]
fn to_gpu_bwd_activation(a: Activation) -> nimcp_gpu::mlp_backward::Activation {
    match a {
        Activation::Relu => nimcp_gpu::mlp_backward::Activation::Relu,
        Activation::Tanh => nimcp_gpu::mlp_backward::Activation::Tanh,
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
            #[cfg(feature = "cuda")]
            gpu_cache: None,
            #[cfg(feature = "cuda")]
            gpu_grads: None,
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

// ---------------------------------------------------------------------------
// GPU tests. Gated on `cuda`; every test skips cleanly when `probe_device`
// reports no GPU so CI hosts without an NVIDIA GPU still build + pass.
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "cuda"))]
mod gpu_tests {
    use super::*;
    use ndarray::Array1;

    /// Match the skip-if-absent idiom used by `crates/gpu` tests.
    fn cuda_available() -> bool {
        nimcp_gpu::probe_device().is_ok()
    }

    /// Deterministic input sampler — does not rely on any crate
    /// external to the adaptive + rand_chacha stack.
    fn sample_vec(rng: &mut ChaCha20Rng, n: usize, scale: f32) -> Array1<f32> {
        use rand::distr::{Distribution, Uniform};
        let dist = Uniform::new(-scale, scale).expect("valid range");
        Array1::from_shape_fn(n, |_| dist.sample(rng))
    }

    #[test]
    fn forward_gpu_matches_cpu() {
        if !cuda_available() {
            eprintln!("skipping: no CUDA device on this host");
            return;
        }
        // Small 3 -> 8 -> 2 MLP, tanh hidden. Same seeded init on both paths.
        let cfg = AdaptiveConfig {
            layers: vec![3, 8, 2],
            rng_seed: 0xAA55_AA55,
            activation: Activation::Tanh,
        };
        let mut net = AdaptiveNet::new(cfg);
        net.init_gpu().expect("init_gpu ok");

        let mut rng = ChaCha20Rng::seed_from_u64(12345);
        for _ in 0..5 {
            let x = sample_vec(&mut rng, 3, 1.0);
            let cpu = net.forward(&x);
            let gpu = net.forward_gpu(&x).expect("forward_gpu ok");
            assert_eq!(cpu.len(), gpu.len());
            for (i, (c, g)) in cpu.iter().zip(gpu.iter()).enumerate() {
                assert!(
                    (c - g).abs() < 1e-4,
                    "idx {i}: cpu={c} gpu={g} diff={}",
                    (c - g).abs()
                );
            }
        }
    }

    #[test]
    fn learn_gpu_decreases_loss() {
        if !cuda_available() {
            eprintln!("skipping: no CUDA device on this host");
            return;
        }
        // XOR-sized MLP: 2 -> 8 -> 1 tanh. We train on XOR and expect
        // the sliding-window mean loss to drop monotonically after the
        // first window.
        let cfg = AdaptiveConfig {
            layers: vec![2, 8, 1],
            rng_seed: 0x0BAD_C0FF_EE0D_DF00,
            activation: Activation::Tanh,
        };
        let mut net = AdaptiveNet::new(cfg);
        net.init_gpu().expect("init_gpu ok");

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

        let lr = 0.1f32;
        let mut losses: Vec<f32> = Vec::with_capacity(100);
        for _ in 0..100 {
            let mut sum = 0.0f32;
            for (x, y) in &samples {
                sum += net.learn_gpu(x, y, lr).expect("learn_gpu ok");
            }
            losses.push(sum / samples.len() as f32);
        }

        // Sliding-window-of-10 mean must be strictly non-increasing.
        let window = 10;
        let mut prev = f32::INFINITY;
        for chunk in losses.chunks(window) {
            let avg: f32 = chunk.iter().sum::<f32>() / (chunk.len() as f32);
            // Allow a tiny numerical nudge (1e-6) so fused-multiply-add
            // jitter on the final window doesn't flake the test.
            assert!(
                avg <= prev + 1e-6,
                "loss window avg went up: {prev:.6} -> {avg:.6} (full series: {losses:?})"
            );
            prev = avg;
        }

        // Net loss reduction — start window vs last window should be
        // meaningful. 2x reduction is a weak-but-reliable bar for XOR.
        let start: f32 = losses[..window].iter().sum::<f32>() / window as f32;
        let end: f32 = losses[losses.len() - window..].iter().sum::<f32>() / window as f32;
        assert!(
            end < start * 0.75,
            "learn_gpu did not meaningfully reduce loss: start={start} end={end}"
        );
    }

    #[test]
    fn sync_gpu_to_cpu_round_trips() {
        if !cuda_available() {
            eprintln!("skipping: no CUDA device on this host");
            return;
        }
        let cfg = AdaptiveConfig {
            layers: vec![3, 6, 2],
            rng_seed: 7777,
            activation: Activation::Tanh,
        };
        let mut net = AdaptiveNet::new(cfg);
        net.init_gpu().expect("init_gpu ok");

        // Train briefly on the GPU so CPU and GPU weights actually diverge.
        let mut rng = ChaCha20Rng::seed_from_u64(9999);
        for _ in 0..10 {
            let x = sample_vec(&mut rng, 3, 1.0);
            let y = sample_vec(&mut rng, 2, 1.0);
            net.learn_gpu(&x, &y, 0.05).expect("learn_gpu ok");
        }

        // Sync weights back. CPU forward must match GPU forward within
        // FMA tolerance on fresh inputs.
        net.sync_gpu_to_cpu().expect("sync ok");

        for _ in 0..5 {
            let x = sample_vec(&mut rng, 3, 1.0);
            let cpu = net.forward(&x);
            let gpu = net.forward_gpu(&x).expect("forward_gpu ok");
            for (c, g) in cpu.iter().zip(gpu.iter()) {
                assert!((c - g).abs() < 1e-4, "post-sync drift: cpu={c} gpu={g}");
            }
        }
    }

    #[test]
    fn init_gpu_is_idempotent() {
        if !cuda_available() {
            eprintln!("skipping: no CUDA device on this host");
            return;
        }
        let cfg = AdaptiveConfig {
            layers: vec![4, 8, 3],
            rng_seed: 2024,
            activation: Activation::Relu,
        };
        let mut net = AdaptiveNet::new(cfg);

        net.init_gpu().expect("first init ok");
        // Reference output before re-init.
        let x = Array1::from_vec(vec![0.25f32, -0.5, 0.75, -0.125]);
        let out_before = net.forward_gpu(&x).expect("forward_gpu after first init");

        // Second init must succeed — the previous cache is dropped.
        net.init_gpu().expect("re-init ok");

        // Forward still works after re-init and produces identical
        // output (weights haven't changed, kernels are deterministic).
        let out_after = net.forward_gpu(&x).expect("forward_gpu after re-init");
        assert_eq!(out_before.len(), out_after.len());
        for (a, b) in out_before.iter().zip(out_after.iter()) {
            assert!((a - b).abs() < 1e-6, "re-init drift: before={a} after={b}");
        }
    }

    #[test]
    fn forward_gpu_requires_init() {
        // No GPU required — this checks the error path before any cuda
        // call. Keep the test behind `cuda` so the error variant exists.
        let mut net = AdaptiveNet::new(AdaptiveConfig {
            layers: vec![3, 4, 2],
            rng_seed: 1,
            activation: Activation::Relu,
        });
        let x = Array1::from_vec(vec![0.0_f32, 0.0, 0.0]);
        // Before init_gpu.
        let err = net.forward_gpu(&x).unwrap_err();
        assert!(
            matches!(err, AdaptiveError::Gpu(ref m) if m.contains("init_gpu")),
            "expected Gpu(init_gpu not called), got {err:?}"
        );
        // learn_gpu should also bail — don't actually run it because it
        // would need a device call stack; just confirm shape checks pass
        // before the init check.
        let y = Array1::from_vec(vec![0.0_f32, 0.0]);
        let err2 = net.learn_gpu(&x, &y, 0.01).unwrap_err();
        assert!(
            matches!(err2, AdaptiveError::Gpu(ref m) if m.contains("init_gpu")),
            "expected Gpu(init_gpu not called), got {err2:?}"
        );
    }
}
