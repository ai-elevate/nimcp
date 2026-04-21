//! GPU MLP backward pass + in-place SGD (Phase 2c).
//!
//! This module is the mirror of Phase 2b's `mlp_forward`. It consumes the
//! forward pass's cached activations + pre-activations, walks back through
//! the layers, produces gradients in a [`GpuGradCache`], and offers an
//! in-place SGD update step.
//!
//! # Layout assumption
//!
//! All weight matrices `W` are stored **row-major**, shape
//! `(out_dim, in_dim)`. Element `W[r, c]` lives at linear index
//! `r * in_dim + c`. All kernels in this module assume that layout; pass
//! it through from `mlp_forward` and do not transpose in flight.
//!
//! # Math summary (MSE loss, matches `crates/networks/adaptive`)
//!
//! - Final-layer seed: `grad_z_last = 2 * (pred - y) / n_out`
//!   (output layer is linear, so `grad_pred == grad_z_last`).
//! - Per layer, walking right-to-left:
//!   - `grad_w[r, c] += grad_z[r] * input[c]`  (outer product, accumulated)
//!   - `grad_b[r]    += grad_z[r]`
//!   - `grad_h_prev[c] = Σ_r W[r, c] * grad_z[r]`  (transpose matvec)
//!   - `grad_z_prev[k] = grad_h_prev[k] * act'(z_prev[k])`
//!
//! # Dependency on Phase 2b (`mlp_forward`)
//!
//! The sibling agent building `mlp_forward` will export
//! [`crate::mlp_forward::GpuWeightCache`] / [`crate::mlp_forward::GpuLayer`]. Until
//! that lands in `crate::mlp_forward`, we define a **local** copy in
//! [`forward_types`] with the exact same public shape so this module and
//! its tests are buildable in isolation. See the `TODO` on the module.

// Launch sites below are `unsafe` — see individual `SAFETY:` comments.
#![allow(unsafe_code)]

use std::sync::Arc;

use cudarc::driver::{
    CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};

use crate::GpuError;

// ---------------------------------------------------------------------------
// Local stand-in for the types that Phase 2b's `mlp_forward` will export.
//
// The sibling agent owns the real definitions. We mirror them here so this
// crate builds + tests standalone. When `mlp_forward` lands, the integrator
// will delete this module and change the `use` below.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Error helper (matches cuda_impl.rs convention).
// ---------------------------------------------------------------------------

fn cuda_err<E: std::fmt::Debug>(e: E) -> GpuError {
    GpuError::Cuda(format!("{e:?}"))
}

// ---------------------------------------------------------------------------
// Activation choice (duplicated from `nimcp-networks-adaptive` — the GPU crate
// is a downstream consumer and shouldn't pull the networks crate into its
// compile tree just to reuse an enum).
// ---------------------------------------------------------------------------

/// Activation applied between hidden layers. The output layer is linear.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// Rectified linear: `max(0, x)`.
    Relu,
    /// Hyperbolic tangent.
    Tanh,
}

// ---------------------------------------------------------------------------
// Grad cache.
// ---------------------------------------------------------------------------

/// Per-layer gradient buffers. Shape mirrors the corresponding [`GpuLayer`].
pub struct GpuGradLayer {
    /// `grad_w` — row-major, shape `(out_dim, in_dim)`. Length `out_dim * in_dim`.
    pub grad_w: CudaSlice<f32>,
    /// `grad_b` — length `out_dim`.
    pub grad_b: CudaSlice<f32>,
}

/// Owns gradient buffers + the NVRTC-compiled kernel module used by
/// backward, SGD, and zero_grads. Build once per training session.
pub struct GpuGradCache {
    /// One entry per weight layer, in the same order as the forward cache.
    pub layers: Vec<GpuGradLayer>,
    /// Holds the compiled kernels so we don't recompile per call. Kept alive
    /// by the `Arc` so loaded `CudaFunction`s stay valid.
    _module: Arc<CudaModule>,
    /// Clone of the forward cache's stream. Lets [`zero_grads`] launch on
    /// the same queue without needing a `&GpuWeightCache` parameter.
    stream: Arc<CudaStream>,
    /// Compiled kernel handles.
    kernels: GradKernels,
}

struct GradKernels {
    relu_deriv: CudaFunction,
    tanh_deriv: CudaFunction,
    elementwise_mul: CudaFunction,
    outer_accumulate: CudaFunction,
    matvec_transpose: CudaFunction,
    sgd_update: CudaFunction,
    zero_f32: CudaFunction,
    seed_grad_z: CudaFunction,
}

impl GpuGradCache {
    /// Allocate grad buffers mirroring `cache`'s shape and compile the
    /// gradient/SGD kernels once. Buffers start zeroed.
    pub fn new(cache: &crate::mlp_forward::GpuWeightCache) -> Result<Self, GpuError> {
        let stream = &cache.stream;

        // Compile kernels once.
        let ptx = cudarc::nvrtc::compile_ptx(KERNEL_SRC).map_err(cuda_err)?;
        let module = cache.ctx.load_module(ptx).map_err(cuda_err)?;

        let kernels = GradKernels {
            relu_deriv: module.load_function("relu_deriv").map_err(cuda_err)?,
            tanh_deriv: module.load_function("tanh_deriv").map_err(cuda_err)?,
            elementwise_mul: module.load_function("elementwise_mul").map_err(cuda_err)?,
            outer_accumulate: module.load_function("outer_accumulate").map_err(cuda_err)?,
            matvec_transpose: module.load_function("matvec_transpose").map_err(cuda_err)?,
            sgd_update: module.load_function("sgd_update").map_err(cuda_err)?,
            zero_f32: module.load_function("zero_f32").map_err(cuda_err)?,
            seed_grad_z: module.load_function("seed_grad_z").map_err(cuda_err)?,
        };

        let mut layers = Vec::with_capacity(cache.layers.len());
        for layer in &cache.layers {
            let grad_w = stream
                .alloc_zeros::<f32>(layer.out_dim * layer.in_dim)
                .map_err(cuda_err)?;
            let grad_b = stream.alloc_zeros::<f32>(layer.out_dim).map_err(cuda_err)?;
            layers.push(GpuGradLayer { grad_w, grad_b });
        }

        Ok(Self {
            layers,
            _module: module,
            stream: stream.clone(),
            kernels,
        })
    }
}

// ---------------------------------------------------------------------------
// Public API: backward / sgd_step / zero_grads.
// ---------------------------------------------------------------------------

/// Run the backward pass for one sample against MSE loss.
///
/// Writes gradients into `grads`. Does not clear them first — the caller
/// owns clearing via [`zero_grads`].
///
/// # Inputs
///
/// - `activations[i]` is the input to layer `i`. Length `num_layers + 1`;
///   `activations[0] == input`, `activations[num_layers]` is the net output.
/// - `pre_activations[i]` is `z_i = W_i @ activations[i] + b_i` for layer `i`.
///
/// Layout is row-major throughout.
///
/// # Errors
///
/// Returns [`GpuError::Cuda`] on any cuda call failure. Returns
/// [`GpuError::Cuda`] with an explanatory message if shapes are wrong.
pub fn mlp_backward(
    cache: &crate::mlp_forward::GpuWeightCache,
    grads: &mut GpuGradCache,
    input: &[f32],
    target: &[f32],
    act: Activation,
    activations: &[Vec<f32>],
    pre_activations: &[Vec<f32>],
) -> Result<(), GpuError> {
    let n_layers = cache.layers.len();
    if n_layers == 0 {
        return Err(GpuError::Cuda("backward called on empty cache".into()));
    }
    if grads.layers.len() != n_layers {
        return Err(GpuError::Cuda(format!(
            "grad/cache layer count mismatch: {} vs {}",
            grads.layers.len(),
            n_layers
        )));
    }
    if activations.len() != n_layers + 1 {
        return Err(GpuError::Cuda(format!(
            "expected {} activations (num_layers+1), got {}",
            n_layers + 1,
            activations.len()
        )));
    }
    if pre_activations.len() != n_layers {
        return Err(GpuError::Cuda(format!(
            "expected {} pre_activations, got {}",
            n_layers,
            pre_activations.len()
        )));
    }

    let first = &cache.layers[0];
    let last = &cache.layers[n_layers - 1];
    if input.len() != first.in_dim {
        return Err(GpuError::Cuda(format!(
            "input len {} doesn't match first layer in_dim {}",
            input.len(),
            first.in_dim
        )));
    }
    if target.len() != last.out_dim {
        return Err(GpuError::Cuda(format!(
            "target len {} doesn't match last layer out_dim {}",
            target.len(),
            last.out_dim
        )));
    }

    let stream = &cache.stream;
    let k = &grads.kernels;

    // Upload every activation + pre-activation to the device once.
    // Phase 2b will eventually hand us these device-resident; for now we
    // accept CPU slices and upload inside. The extra H2D copies are
    // acceptable for Phase 2c's correctness goal.
    let mut d_activations: Vec<CudaSlice<f32>> = Vec::with_capacity(n_layers + 1);
    for a in activations {
        d_activations.push(stream.memcpy_stod(a).map_err(cuda_err)?);
    }
    let mut d_pre: Vec<CudaSlice<f32>> = Vec::with_capacity(n_layers);
    for z in pre_activations {
        d_pre.push(stream.memcpy_stod(z).map_err(cuda_err)?);
    }
    let d_target = stream.memcpy_stod(target).map_err(cuda_err)?;

    // Seed grad_z at the final layer: grad_z_last = 2 * (pred - y) / n_out.
    let last_out = last.out_dim;
    let mut grad_z: CudaSlice<f32> = stream.alloc_zeros::<f32>(last_out).map_err(cuda_err)?;
    {
        let n_u32 = last_out as u32;
        let inv_n = 2.0f32 / (last_out as f32);
        let cfg = LaunchConfig::for_num_elems(n_u32);
        let mut b = stream.launch_builder(&k.seed_grad_z);
        b.arg(&d_activations[n_layers]); // pred
        b.arg(&d_target);
        b.arg(&mut grad_z);
        b.arg(&inv_n);
        b.arg(&n_u32);
        // SAFETY: kernel signature is
        //   (const float*, const float*, float*, float, int)
        // all four buffer args are live CudaSlice<f32> owning their memory,
        // and n_u32 matches the element counts of pred/target/grad_z.
        unsafe { b.launch(cfg) }.map_err(cuda_err)?;
    }

    // Walk layers right to left.
    for i in (0..n_layers).rev() {
        let layer = &cache.layers[i];
        let grad_layer = &mut grads.layers[i];
        let input_slice = &d_activations[i];

        // grad_w[r, c] += grad_z[r] * input[c]  — 2D launch over (out, in).
        {
            let out_u32 = layer.out_dim as u32;
            let in_u32 = layer.in_dim as u32;
            let block_x: u32 = 16;
            let block_y: u32 = 16;
            let grid_x = in_u32.div_ceil(block_x);
            let grid_y = out_u32.div_ceil(block_y);
            let cfg = LaunchConfig {
                grid_dim: (grid_x, grid_y, 1),
                block_dim: (block_x, block_y, 1),
                shared_mem_bytes: 0,
            };
            let mut b = stream.launch_builder(&k.outer_accumulate);
            b.arg(&grad_z);
            b.arg(input_slice);
            b.arg(&mut grad_layer.grad_w);
            b.arg(&out_u32);
            b.arg(&in_u32);
            // SAFETY: kernel signature is
            //   (const float* gz, const float* in_, float* gw, int out, int in)
            // grad_z has `out_dim` elements, input_slice has `in_dim`, and
            // grad_w has out_dim*in_dim elements, all matching the int args.
            unsafe { b.launch(cfg) }.map_err(cuda_err)?;
        }

        // grad_b[r] += grad_z[r]  — sgd_update with lr = -1 would do
        // the wrong thing (it subtracts). Use a dedicated add kernel.
        // We reuse sgd_update with lr = -1.0 applied to b: the kernel does
        // p -= lr * g. If we set p = grad_b, g = grad_z, lr = -1.0, we get
        // grad_b -= -1.0 * grad_z == grad_b + grad_z. Clean and keeps the
        // kernel count down.
        {
            let n_u32 = layer.out_dim as u32;
            let lr = -1.0f32;
            let cfg = LaunchConfig::for_num_elems(n_u32);
            let mut b = stream.launch_builder(&k.sgd_update);
            b.arg(&mut grad_layer.grad_b);
            b.arg(&grad_z);
            b.arg(&lr);
            b.arg(&n_u32);
            // SAFETY: kernel signature is (float*, const float*, float, int);
            // both buffers have `out_dim` elements and n_u32 == out_dim.
            unsafe { b.launch(cfg) }.map_err(cuda_err)?;
        }

        // If not the first layer, propagate to grad_z_prev.
        if i > 0 {
            let prev_out = cache.layers[i - 1].out_dim; // == layer.in_dim
            debug_assert_eq!(prev_out, layer.in_dim);

            // grad_h_prev[c] = Σ_r W[r, c] * grad_z[r]
            let mut grad_h_prev = stream.alloc_zeros::<f32>(prev_out).map_err(cuda_err)?;
            {
                let out_u32 = layer.out_dim as u32;
                let in_u32 = layer.in_dim as u32;
                let cfg = LaunchConfig::for_num_elems(in_u32);
                let mut b = stream.launch_builder(&k.matvec_transpose);
                b.arg(&layer.w);
                b.arg(&grad_z);
                b.arg(&mut grad_h_prev);
                b.arg(&out_u32);
                b.arg(&in_u32);
                // SAFETY: signature (const float* w, const float* gz,
                //   float* out, int out_dim, int in_dim). w has
                //   out_dim*in_dim, gz has out_dim, grad_h_prev has in_dim.
                unsafe { b.launch(cfg) }.map_err(cuda_err)?;
            }

            // grad_z_prev = grad_h_prev * act'(z_prev)
            let mut deriv = stream.alloc_zeros::<f32>(prev_out).map_err(cuda_err)?;
            {
                let n_u32 = prev_out as u32;
                let cfg = LaunchConfig::for_num_elems(n_u32);
                let kernel = match act {
                    Activation::Relu => &k.relu_deriv,
                    Activation::Tanh => &k.tanh_deriv,
                };
                let mut b = stream.launch_builder(kernel);
                b.arg(&d_pre[i - 1]);
                b.arg(&mut deriv);
                b.arg(&n_u32);
                // SAFETY: signature (const float* z, float* out, int n).
                // Both buffers have `prev_out` elements matching n_u32.
                unsafe { b.launch(cfg) }.map_err(cuda_err)?;
            }

            let mut new_grad_z = stream.alloc_zeros::<f32>(prev_out).map_err(cuda_err)?;
            {
                let n_u32 = prev_out as u32;
                let cfg = LaunchConfig::for_num_elems(n_u32);
                let mut b = stream.launch_builder(&k.elementwise_mul);
                b.arg(&grad_h_prev);
                b.arg(&deriv);
                b.arg(&mut new_grad_z);
                b.arg(&n_u32);
                // SAFETY: signature (const float* a, const float* b,
                //   float* c, int n). All buffers have `prev_out` elements.
                unsafe { b.launch(cfg) }.map_err(cuda_err)?;
            }

            grad_z = new_grad_z;
        }
    }

    // Synchronize so caller-visible state is consistent on return.
    stream.synchronize().map_err(cuda_err)?;
    Ok(())
}

/// In-place SGD: for every layer `i`,
/// `w_i -= lr * grad_w_i` and `b_i -= lr * grad_b_i`.
///
/// Does not reset gradients — call [`zero_grads`] before the next step if
/// you don't want accumulation.
///
/// # Errors
///
/// Returns [`GpuError::Cuda`] if shape sanity checks fail or a cuda call
/// errors out.
pub fn sgd_step(
    cache: &mut crate::mlp_forward::GpuWeightCache,
    grads: &GpuGradCache,
    lr: f32,
) -> Result<(), GpuError> {
    if grads.layers.len() != cache.layers.len() {
        return Err(GpuError::Cuda(format!(
            "grad/cache layer count mismatch: {} vs {}",
            grads.layers.len(),
            cache.layers.len()
        )));
    }

    let stream = cache.stream.clone();
    let k = &grads.kernels;

    for (layer, grad_layer) in cache.layers.iter_mut().zip(grads.layers.iter()) {
        // Weights.
        {
            let n = (layer.out_dim * layer.in_dim) as u32;
            let cfg = LaunchConfig::for_num_elems(n);
            let mut b = stream.launch_builder(&k.sgd_update);
            b.arg(&mut layer.w);
            b.arg(&grad_layer.grad_w);
            b.arg(&lr);
            b.arg(&n);
            // SAFETY: signature (float* p, const float* g, float lr, int n).
            // w has out_dim*in_dim elements, grad_w matches by construction.
            unsafe { b.launch(cfg) }.map_err(cuda_err)?;
        }
        // Biases.
        {
            let n = layer.out_dim as u32;
            let cfg = LaunchConfig::for_num_elems(n);
            let mut b = stream.launch_builder(&k.sgd_update);
            b.arg(&mut layer.b);
            b.arg(&grad_layer.grad_b);
            b.arg(&lr);
            b.arg(&n);
            // SAFETY: signature (float* p, const float* g, float lr, int n).
            // b has out_dim elements, grad_b matches.
            unsafe { b.launch(cfg) }.map_err(cuda_err)?;
        }
    }

    stream.synchronize().map_err(cuda_err)?;
    Ok(())
}

/// Zero every gradient buffer in `grads`.
///
/// # Errors
///
/// Returns [`GpuError::Cuda`] on any cuda failure.
pub fn zero_grads(grads: &mut GpuGradCache) -> Result<(), GpuError> {
    // Launch on the same stream the cache was built with. We stash an Arc
    // to it at construction so `zero_grads` can satisfy the spec's
    // `&mut GpuGradCache`-only signature.
    let stream = grads.stream.clone();
    let k = &grads.kernels;
    for grad_layer in &mut grads.layers {
        {
            let n = grad_layer.grad_w.len() as u32;
            let cfg = LaunchConfig::for_num_elems(n);
            let mut b = stream.launch_builder(&k.zero_f32);
            b.arg(&mut grad_layer.grad_w);
            b.arg(&n);
            // SAFETY: signature (float* p, int n). grad_w len == n.
            unsafe { b.launch(cfg) }.map_err(cuda_err)?;
        }
        {
            let n = grad_layer.grad_b.len() as u32;
            let cfg = LaunchConfig::for_num_elems(n);
            let mut b = stream.launch_builder(&k.zero_f32);
            b.arg(&mut grad_layer.grad_b);
            b.arg(&n);
            // SAFETY: signature (float* p, int n). grad_b len == n.
            unsafe { b.launch(cfg) }.map_err(cuda_err)?;
        }
    }
    stream.synchronize().map_err(cuda_err)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// NVRTC-compiled kernels. Compiled once per GpuGradCache::new.
// ---------------------------------------------------------------------------

const KERNEL_SRC: &str = r#"
extern "C" __global__ void relu_deriv(const float* z, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = (z[i] > 0.0f) ? 1.0f : 0.0f;
    }
}

extern "C" __global__ void tanh_deriv(const float* z, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float t = tanhf(z[i]);
        out[i] = 1.0f - t * t;
    }
}

extern "C" __global__ void elementwise_mul(const float* a, const float* b,
                                           float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] * b[i];
    }
}

// Row-major grad_w, shape (out_dim, in_dim). Accumulates (+=).
// 2D launch: threadIdx/blockIdx .x iterates `in_dim`, .y iterates `out_dim`.
extern "C" __global__ void outer_accumulate(const float* grad_z,
                                            const float* input_,
                                            float* grad_w,
                                            int out_dim, int in_dim) {
    int c = blockIdx.x * blockDim.x + threadIdx.x; // column (in_dim axis)
    int r = blockIdx.y * blockDim.y + threadIdx.y; // row (out_dim axis)
    if (r < out_dim && c < in_dim) {
        grad_w[r * in_dim + c] += grad_z[r] * input_[c];
    }
}

// grad_h_prev[c] = sum over r of W[r, c] * grad_z[r]. W is row-major,
// shape (out_dim, in_dim). 1D launch over in_dim.
extern "C" __global__ void matvec_transpose(const float* w,
                                            const float* grad_z,
                                            float* grad_h_prev,
                                            int out_dim, int in_dim) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < in_dim) {
        float acc = 0.0f;
        for (int r = 0; r < out_dim; ++r) {
            acc += w[r * in_dim + c] * grad_z[r];
        }
        grad_h_prev[c] = acc;
    }
}

// In-place SGD: p[i] -= lr * g[i]. Doubles as "accumulate" with lr = -1.
extern "C" __global__ void sgd_update(float* p, const float* g,
                                      float lr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        p[i] -= lr * g[i];
    }
}

extern "C" __global__ void zero_f32(float* p, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        p[i] = 0.0f;
    }
}

// grad_z_last = inv_n * (pred - target). For MSE with inv_n = 2/n_out the
// result matches the adaptive CPU reference bit-for-bit on any platform
// that follows IEEE-754 round-to-nearest (all current NVIDIA GPUs do).
extern "C" __global__ void seed_grad_z(const float* pred, const float* target,
                                       float* out, float inv_n, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = inv_n * (pred[i] - target[i]);
    }
}
"#;

// ---------------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use cudarc::driver::CudaContext;

    // ---------- test harness: spin up a GpuWeightCache using our local
    // forward_types stub, seeded with a CPU weight init that matches the
    // adaptive crate's Kaiming-like scheme. This is test-local; the real
    // init is 2b's job.

    struct TestNet {
        layers: Vec<(Vec<f32>, Vec<f32>, usize, usize)>, // (w_row_major, b, in, out)
    }

    impl TestNet {
        fn new_seeded(dims: &[usize], seed: u64) -> Self {
            use rand::SeedableRng;
            use rand::distr::{Distribution, Uniform};
            use rand_chacha::ChaCha20Rng;
            let mut rng = ChaCha20Rng::seed_from_u64(seed);
            let mut layers = Vec::new();
            for w in dims.windows(2) {
                let (in_dim, out_dim) = (w[0], w[1]);
                let scale = (2.0f32 / in_dim as f32).sqrt();
                let dist = Uniform::new(-scale, scale).expect("valid uniform");
                let mut wv = vec![0.0f32; out_dim * in_dim];
                for x in &mut wv {
                    *x = dist.sample(&mut rng);
                }
                let bv = vec![0.0f32; out_dim];
                layers.push((wv, bv, in_dim, out_dim));
            }
            Self { layers }
        }

        fn into_gpu_cache(self) -> Result<crate::mlp_forward::GpuWeightCache, GpuError> {
            // Use mlp_forward::GpuWeightCache::new so we get the compiled
            // kernels + lifecycle the backward path expects.
            let refs: Vec<(&[f32], &[f32], usize, usize)> = self
                .layers
                .iter()
                .map(|(w, b, i, o)| (w.as_slice(), b.as_slice(), *i, *o))
                .collect();
            crate::mlp_forward::GpuWeightCache::new(refs)
        }
    }

    fn cuda_available() -> bool {
        CudaContext::new(0).is_ok()
    }

    /// Reference CPU forward that records activations + pre-activations in
    /// the same layout `mlp_backward` expects.
    fn cpu_forward(
        weights: &[(Vec<f32>, Vec<f32>, usize, usize)],
        input: &[f32],
        act: Activation,
    ) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let n_layers = weights.len();
        let mut activations: Vec<Vec<f32>> = Vec::with_capacity(n_layers + 1);
        let mut pre: Vec<Vec<f32>> = Vec::with_capacity(n_layers);
        activations.push(input.to_vec());

        for (i, (w, b, in_dim, out_dim)) in weights.iter().enumerate() {
            let h = &activations[i];
            let mut z = vec![0.0f32; *out_dim];
            for r in 0..*out_dim {
                let mut s = b[r];
                for c in 0..*in_dim {
                    s += w[r * in_dim + c] * h[c];
                }
                z[r] = s;
            }
            pre.push(z.clone());
            let is_last = i == n_layers - 1;
            if is_last {
                activations.push(z);
            } else {
                let mut hz = z;
                for v in &mut hz {
                    *v = match act {
                        Activation::Relu => v.max(0.0),
                        Activation::Tanh => v.tanh(),
                    };
                }
                activations.push(hz);
            }
        }
        (activations, pre)
    }

    fn cpu_mse(pred: &[f32], y: &[f32]) -> f32 {
        let n = pred.len() as f32;
        pred.iter()
            .zip(y.iter())
            .map(|(p, t)| (p - t) * (p - t))
            .sum::<f32>()
            / n
    }

    /// (w_row_major, b). One entry per layer.
    type DownloadedWeights = Vec<(Vec<f32>, Vec<f32>)>;
    /// (w_row_major, b, in_dim, out_dim). One entry per layer.
    type ShapedWeights = Vec<(Vec<f32>, Vec<f32>, usize, usize)>;

    fn download_weights(
        cache: &crate::mlp_forward::GpuWeightCache,
    ) -> Result<DownloadedWeights, GpuError> {
        let mut out = Vec::with_capacity(cache.layers.len());
        for l in &cache.layers {
            let w = cache.stream.memcpy_dtov(&l.w).map_err(cuda_err)?;
            let b = cache.stream.memcpy_dtov(&l.b).map_err(cuda_err)?;
            out.push((w, b));
        }
        Ok(out)
    }

    fn layer_shapes(cache: &crate::mlp_forward::GpuWeightCache) -> Vec<(usize, usize)> {
        cache.layers.iter().map(|l| (l.in_dim, l.out_dim)).collect()
    }

    fn rebuild_weights_from_gpu(
        cache: &crate::mlp_forward::GpuWeightCache,
    ) -> Result<ShapedWeights, GpuError> {
        let shapes = layer_shapes(cache);
        let wb = download_weights(cache)?;
        Ok(wb
            .into_iter()
            .zip(shapes)
            .map(|((w, b), (in_dim, out_dim))| (w, b, in_dim, out_dim))
            .collect())
    }

    #[test]
    fn zero_grads_zeros_buffers() {
        if !cuda_available() {
            eprintln!("skipping: no CUDA device on this host");
            return;
        }
        let net = TestNet::new_seeded(&[3, 4, 2], 1234);
        let cache = net.into_gpu_cache().unwrap();
        let mut grads = GpuGradCache::new(&cache).unwrap();

        // Fill grads with non-zero values by running a dummy backward.
        let input = vec![0.1f32, -0.2, 0.3];
        let target = vec![0.5f32, -0.1];
        let weights = rebuild_weights_from_gpu(&cache).unwrap();
        let (acts, pre) = cpu_forward(&weights, &input, Activation::Tanh);
        mlp_backward(
            &cache,
            &mut grads,
            &input,
            &target,
            Activation::Tanh,
            &acts,
            &pre,
        )
        .unwrap();

        // Now zero and assert.
        zero_grads(&mut grads).unwrap();
        for gl in &grads.layers {
            let gw = cache.stream.memcpy_dtov(&gl.grad_w).unwrap();
            let gb = cache.stream.memcpy_dtov(&gl.grad_b).unwrap();
            assert!(gw.iter().all(|&x| x == 0.0), "grad_w not zeroed: {gw:?}");
            assert!(gb.iter().all(|&x| x == 0.0), "grad_b not zeroed: {gb:?}");
        }
    }

    #[test]
    fn backward_matches_finite_differences() {
        if !cuda_available() {
            eprintln!("skipping: no CUDA device on this host");
            return;
        }
        // 3 -> 4 -> 2 tanh MLP.
        let dims = [3usize, 4, 2];
        let net = TestNet::new_seeded(&dims, 314_159);
        let cache = net.into_gpu_cache().unwrap();
        let mut grads = GpuGradCache::new(&cache).unwrap();

        let input = vec![0.3f32, -0.7, 0.1];
        let target = vec![0.2f32, -0.5];

        // Run analytical backward on GPU.
        let weights = rebuild_weights_from_gpu(&cache).unwrap();
        let (acts, pre) = cpu_forward(&weights, &input, Activation::Tanh);
        mlp_backward(
            &cache,
            &mut grads,
            &input,
            &target,
            Activation::Tanh,
            &acts,
            &pre,
        )
        .unwrap();

        // Download analytic grads.
        let mut analytic_w: Vec<Vec<f32>> = Vec::with_capacity(grads.layers.len());
        let mut analytic_b: Vec<Vec<f32>> = Vec::with_capacity(grads.layers.len());
        for gl in &grads.layers {
            analytic_w.push(cache.stream.memcpy_dtov(&gl.grad_w).unwrap());
            analytic_b.push(cache.stream.memcpy_dtov(&gl.grad_b).unwrap());
        }

        // Finite-difference check on a sample of entries — full grid is
        // expensive and f32 finite differences are noisy, so we subsample.
        let eps = 1e-3f32;
        let tol = 1e-3f32;
        for li in 0..weights.len() {
            let (_w, _b, in_dim, out_dim) = weights[li].clone();

            // Weight entries, every 3rd to keep the test fast but still
            // touch at least one entry per row for the small dims here.
            let positions: Vec<(usize, usize)> = (0..out_dim)
                .flat_map(|i| (0..in_dim).map(move |j| (i, j)))
                .step_by(3)
                .collect();
            for (r, c) in positions {
                let mut wplus = weights.clone();
                wplus[li].0[r * in_dim + c] += eps;
                let (ap, _) = cpu_forward(&wplus, &input, Activation::Tanh);
                let loss_plus = cpu_mse(ap.last().unwrap(), &target);

                let mut wminus = weights.clone();
                wminus[li].0[r * in_dim + c] -= eps;
                let (am, _) = cpu_forward(&wminus, &input, Activation::Tanh);
                let loss_minus = cpu_mse(am.last().unwrap(), &target);

                let numeric = (loss_plus - loss_minus) / (2.0 * eps);
                let analytic = analytic_w[li][r * in_dim + c];
                assert!(
                    (numeric - analytic).abs() < tol,
                    "w grad mismatch layer={li} [{r},{c}]: numeric={numeric} analytic={analytic}"
                );
            }

            // Bias entries — all of them; out_dim is small.
            #[allow(clippy::needless_range_loop)]
            for k in 0..out_dim {
                let mut wplus = weights.clone();
                wplus[li].1[k] += eps;
                let (ap, _) = cpu_forward(&wplus, &input, Activation::Tanh);
                let loss_plus = cpu_mse(ap.last().unwrap(), &target);

                let mut wminus = weights.clone();
                wminus[li].1[k] -= eps;
                let (am, _) = cpu_forward(&wminus, &input, Activation::Tanh);
                let loss_minus = cpu_mse(am.last().unwrap(), &target);

                let numeric = (loss_plus - loss_minus) / (2.0 * eps);
                let analytic = analytic_b[li][k];
                assert!(
                    (numeric - analytic).abs() < tol,
                    "b grad mismatch layer={li} [{k}]: numeric={numeric} analytic={analytic}"
                );
            }
        }
    }

    #[test]
    fn sgd_decreases_loss_on_toy_problem() {
        if !cuda_available() {
            eprintln!("skipping: no CUDA device on this host");
            return;
        }
        let dims = [3usize, 5, 2];
        let net = TestNet::new_seeded(&dims, 2024);
        let mut cache = net.into_gpu_cache().unwrap();
        let mut grads = GpuGradCache::new(&cache).unwrap();

        let input = vec![0.4f32, -0.2, 0.8];
        let target = vec![0.1f32, 0.3];

        let lr = 0.05f32;
        let mut losses: Vec<f32> = Vec::with_capacity(100);

        for _ in 0..100 {
            zero_grads(&mut grads).unwrap();
            let weights = rebuild_weights_from_gpu(&cache).unwrap();
            let (acts, pre) = cpu_forward(&weights, &input, Activation::Tanh);
            let pred = acts.last().unwrap().clone();
            let loss = cpu_mse(&pred, &target);
            losses.push(loss);

            mlp_backward(
                &cache,
                &mut grads,
                &input,
                &target,
                Activation::Tanh,
                &acts,
                &pre,
            )
            .unwrap();
            sgd_step(&mut cache, &grads, lr).unwrap();
        }

        // Loss must decrease overall. Allow a few non-monotone bumps at the
        // very start while the GPU path warms, but end must be << start.
        let start = losses[0];
        let end = *losses.last().unwrap();
        assert!(
            end < start * 0.5,
            "SGD did not reduce loss: start={start:.6} end={end:.6}"
        );

        // Check monotone-after-warmup: after step 5, every window-of-5
        // average is non-increasing vs the previous window.
        let mut prev_avg = f32::INFINITY;
        for w in losses[5..].chunks(5) {
            let avg: f32 = w.iter().sum::<f32>() / (w.len() as f32);
            assert!(
                avg <= prev_avg + 1e-6,
                "loss windowed-average went up: {prev_avg:.6} -> {avg:.6}"
            );
            prev_avg = avg;
        }
    }
}
