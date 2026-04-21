//! MLP forward on GPU — Phase 2b.
//!
//! Implements the compute path equivalent to [`nimcp_adaptive::AdaptiveNet::forward`]:
//! for each layer transition `(W, b)`, compute `h_out = act(W · h_in + b)`
//! with the output layer linear. Weights stay resident on the device
//! across forward calls via [`GpuWeightCache`]; intermediates never leave
//! device memory.
//!
//! # Weight layout
//!
//! Row-major, shape `(out_dim, in_dim)`. Element `(i, j)` — the weight
//! from input `j` to output `i` — lives at index `i * in_dim + j`. This
//! matches `ndarray::Array2` standard layout and V1's CUDA weight cache,
//! so uploads from either source are a plain memcpy.
//!
//! # Design notes
//!
//! - Three kernels (`matmul_bias`, `relu_inplace`, `tanh_inplace`) live
//!   as NVRTC source strings and are compiled once per cache via
//!   [`cudarc::nvrtc::compile_ptx`]. The compiled module is stored on
//!   the cache so forward calls reuse it.
//! - Forward allocates two device ping-pong scratch buffers per call
//!   (sized to the widest layer) and swaps between them across
//!   transitions. Keeping scratch on the cache would require interior
//!   mutability to match the `&GpuWeightCache` signature; the per-call
//!   alloc is stream-ordered and dominated by the actual matmul work.
//!   A future `mlp_forward_batch` can amortize if profiling demands it.
//! - The naive `matmul_bias` kernel dedicates one thread per output
//!   neuron and accumulates across `in_dim`. This is fine for MLPs
//!   with `in_dim` up to a few thousand and is the baseline Phase 2b
//!   ships — a tiled / cuBLAS version lands if profiling demands it.

use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};

use crate::GpuError;

fn cuda_err<E: std::fmt::Debug>(e: E) -> GpuError {
    GpuError::Cuda(format!("{e:?}"))
}

/// Activation applied after each hidden layer. Output layer is always linear.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// Rectified linear: `max(0, x)`.
    Relu,
    /// Hyperbolic tangent.
    Tanh,
}

/// Device-resident weights for one layer transition.
///
/// `w` is row-major shape `(out_dim, in_dim)`; `b` has length `out_dim`.
pub struct GpuLayer {
    /// Weight matrix, flattened row-major, length `out_dim * in_dim`.
    pub w: CudaSlice<f32>,
    /// Bias vector, length `out_dim`.
    pub b: CudaSlice<f32>,
    /// Input dimension.
    pub in_dim: usize,
    /// Output dimension.
    pub out_dim: usize,
}

impl std::fmt::Debug for GpuLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuLayer")
            .field("in_dim", &self.in_dim)
            .field("out_dim", &self.out_dim)
            .finish_non_exhaustive()
    }
}

/// Persistent device-side weight cache for an MLP.
///
/// Owns:
/// - A CUDA context + default stream (via `Arc`, so cheap to clone for
///   auxiliary work).
/// - One [`GpuLayer`] per layer transition.
/// - A compiled kernel module (`matmul_bias`, `relu_inplace`, `tanh_inplace`).
///
/// Recommended lifetime: build once at model-init, reuse across many
/// [`mlp_forward`] calls. Recompiling kernels per call would dominate
/// runtime.
pub struct GpuWeightCache {
    // `pub(crate)` so sibling modules (mlp_backward, future optimisers)
    // can touch the device buffers directly — same crate, same trust
    // boundary, same invariants.
    pub(crate) layers: Vec<GpuLayer>,
    pub(crate) ctx: Arc<CudaContext>,
    pub(crate) stream: Arc<CudaStream>,
    #[allow(dead_code)] // Held to keep the loaded PTX alive for the kernel handles.
    module: Arc<CudaModule>,
    kernel_matmul_bias: CudaFunction,
    kernel_relu: CudaFunction,
    kernel_tanh: CudaFunction,
    /// Max activation width across all layers — scratch capacity for forward.
    scratch_cap: usize,
}

impl std::fmt::Debug for GpuWeightCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuWeightCache")
            .field("num_layers", &self.layers.len())
            .field("scratch_cap", &self.scratch_cap)
            .finish_non_exhaustive()
    }
}

impl GpuWeightCache {
    /// Allocate + upload weights onto the device.
    ///
    /// Each tuple is `(w_row_major, b, in_dim, out_dim)`. Requirements:
    ///
    /// - `w.len() == in_dim * out_dim` (row-major, `(out_dim, in_dim)`).
    /// - `b.len() == out_dim`.
    /// - Adjacent layers must chain: `layers[i].out_dim == layers[i+1].in_dim`.
    /// - At least one layer.
    ///
    /// Compiles the kernel module once and stores it on the cache.
    pub fn new(layers: Vec<(&[f32], &[f32], usize, usize)>) -> Result<Self, GpuError> {
        if layers.is_empty() {
            return Err(GpuError::Cuda("GpuWeightCache::new: empty layers".into()));
        }
        // Validate shapes + chaining up front so we never allocate on an invalid spec.
        for (idx, (w, b, in_dim, out_dim)) in layers.iter().enumerate() {
            let in_dim = *in_dim;
            let out_dim = *out_dim;
            if in_dim == 0 || out_dim == 0 {
                return Err(GpuError::Cuda(format!(
                    "layer {idx}: zero dimension (in={in_dim}, out={out_dim})"
                )));
            }
            if w.len() != in_dim * out_dim {
                return Err(GpuError::Cuda(format!(
                    "layer {idx}: w.len()={} but expected {}*{}={}",
                    w.len(),
                    out_dim,
                    in_dim,
                    in_dim * out_dim
                )));
            }
            if b.len() != out_dim {
                return Err(GpuError::Cuda(format!(
                    "layer {idx}: b.len()={} but expected {out_dim}",
                    b.len()
                )));
            }
        }
        for pair in layers.windows(2) {
            let (_, _, _, prev_out) = pair[0];
            let (_, _, next_in, _) = pair[1];
            if prev_out != next_in {
                return Err(GpuError::Cuda(format!(
                    "layer chain: prev out_dim={prev_out} != next in_dim={next_in}"
                )));
            }
        }

        let ctx = CudaContext::new(0).map_err(cuda_err)?;
        let stream = ctx.default_stream();

        // Compile kernels once, stash the module + function handles.
        let ptx = cudarc::nvrtc::compile_ptx(KERNEL_SRC).map_err(cuda_err)?;
        let module = ctx.load_module(ptx).map_err(cuda_err)?;
        let kernel_matmul_bias = module.load_function("matmul_bias").map_err(cuda_err)?;
        let kernel_relu = module.load_function("relu_inplace").map_err(cuda_err)?;
        let kernel_tanh = module.load_function("tanh_inplace").map_err(cuda_err)?;

        // Upload every layer.
        let mut gpu_layers: Vec<GpuLayer> = Vec::with_capacity(layers.len());
        let mut scratch_cap: usize = 0;
        for (w, b, in_dim, out_dim) in layers {
            let w_dev = stream.memcpy_stod(w).map_err(cuda_err)?;
            let b_dev = stream.memcpy_stod(b).map_err(cuda_err)?;
            scratch_cap = scratch_cap.max(in_dim).max(out_dim);
            gpu_layers.push(GpuLayer {
                w: w_dev,
                b: b_dev,
                in_dim,
                out_dim,
            });
        }

        tracing::info!(
            num_layers = gpu_layers.len(),
            scratch_cap,
            "gpu weight cache built"
        );

        Ok(Self {
            layers: gpu_layers,
            ctx,
            stream,
            module,
            kernel_matmul_bias,
            kernel_relu,
            kernel_tanh,
            scratch_cap,
        })
    }

    /// Number of layer transitions held by this cache.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Read-only view of one cached layer's shape.
    pub fn layer(&self, i: usize) -> Option<&GpuLayer> {
        self.layers.get(i)
    }

    /// Update layer `i`'s weights + bias from CPU slices.
    ///
    /// Shapes must match exactly: `w.len() == in_dim * out_dim`,
    /// `b.len() == out_dim`. Reallocating with different shape is not
    /// supported — rebuild the cache instead.
    pub fn upload_layer(&mut self, i: usize, w: &[f32], b: &[f32]) -> Result<(), GpuError> {
        let layer = self
            .layers
            .get_mut(i)
            .ok_or_else(|| GpuError::Cuda(format!("upload_layer: index {i} out of range")))?;
        if w.len() != layer.in_dim * layer.out_dim {
            return Err(GpuError::Cuda(format!(
                "upload_layer {i}: w.len()={} but expected {}*{}={}",
                w.len(),
                layer.out_dim,
                layer.in_dim,
                layer.in_dim * layer.out_dim
            )));
        }
        if b.len() != layer.out_dim {
            return Err(GpuError::Cuda(format!(
                "upload_layer {i}: b.len()={} but expected {}",
                b.len(),
                layer.out_dim
            )));
        }
        self.stream.memcpy_htod(w, &mut layer.w).map_err(cuda_err)?;
        self.stream.memcpy_htod(b, &mut layer.b).map_err(cuda_err)?;
        Ok(())
    }

    /// Download layer `i`'s weights + bias back to host memory.
    ///
    /// Returns `(w_row_major, b)`. Synchronizes the stream implicitly
    /// via `memcpy_dtov`.
    pub fn download_layer(&self, i: usize) -> Result<(Vec<f32>, Vec<f32>), GpuError> {
        let layer = self
            .layers
            .get(i)
            .ok_or_else(|| GpuError::Cuda(format!("download_layer: index {i} out of range")))?;
        let w = self.stream.memcpy_dtov(&layer.w).map_err(cuda_err)?;
        let b = self.stream.memcpy_dtov(&layer.b).map_err(cuda_err)?;
        Ok((w, b))
    }

    /// Borrow the CUDA context (for callers that need to share it).
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// Borrow the default stream.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }
}

/// Run the MLP forward pass using cached device weights.
///
/// `h_0 = input`, `h_{i+1} = act(W_i · h_i + b_i)` for hidden layers,
/// `h_N = W_{N-1} · h_{N-1} + b_{N-1}` for the final (linear) layer.
///
/// Input is uploaded once into a scratch buffer; all intermediates stay
/// on the device. Output is copied back to a fresh `Vec<f32>`.
///
/// # Errors
///
/// - `GpuError::Cuda` if `input.len() != layers[0].in_dim` or if any
///   kernel launch / memcpy fails.
pub fn mlp_forward(
    cache: &GpuWeightCache,
    input: &[f32],
    act: Activation,
) -> Result<Vec<f32>, GpuError> {
    if cache.layers.is_empty() {
        return Err(GpuError::Cuda("mlp_forward: empty cache".into()));
    }
    let expected_in = cache.layers[0].in_dim;
    if input.len() != expected_in {
        return Err(GpuError::Cuda(format!(
            "mlp_forward: input.len()={} but first layer in_dim={}",
            input.len(),
            expected_in
        )));
    }

    let stream = &cache.stream;

    // Allocate two per-call scratch buffers sized to the widest layer.
    // Kept off the cache because `mlp_forward` takes `&GpuWeightCache` and
    // cudarc's stream-ordered writes need `&mut CudaSlice`. The alloc is
    // stream-ordered and dwarfed by the matmul itself.
    let cap = cache.scratch_cap;
    let mut buf_a: CudaSlice<f32> = stream.alloc_zeros::<f32>(cap).map_err(cuda_err)?;
    let mut buf_b: CudaSlice<f32> = stream.alloc_zeros::<f32>(cap).map_err(cuda_err)?;

    // Upload input into buf_a. memcpy_htod copies into the leading elements
    // of buf_a; we only touch up to input.len() in kernels.
    // buf_a is zero-padded so any untouched tail stays 0.
    // cudarc's memcpy_htod with a smaller host slice copies exactly that many
    // elements to the start of the device slice.
    stream.memcpy_htod(input, &mut buf_a).map_err(cuda_err)?;

    let n_layers = cache.layers.len();
    let last = n_layers - 1;

    // On each iteration: src_buf -> dst_buf. After the loop, `src_buf`
    // holds the final output (we swap at the end of every step).
    let mut use_a_as_src = true;

    for (i, layer) in cache.layers.iter().enumerate() {
        let in_dim = layer.in_dim as u32;
        let out_dim = layer.out_dim as u32;

        // Partition (src, dst) based on current ping-pong state.
        let (src_ref, dst_ref) = if use_a_as_src {
            (&buf_a, &mut buf_b)
        } else {
            (&buf_b, &mut buf_a)
        };

        // Launch matmul_bias: one thread per output element, dot over in_dim.
        let cfg = LaunchConfig::for_num_elems(out_dim);
        let mut builder = stream.launch_builder(&cache.kernel_matmul_bias);
        builder.arg(&layer.w);
        builder.arg(&layer.b);
        builder.arg(src_ref);
        builder.arg(&*dst_ref);
        builder.arg(&in_dim);
        builder.arg(&out_dim);
        // SAFETY: the kernel signature is
        //   (const float* w, const float* b, const float* x, float* y,
        //    int in_dim, int out_dim)
        // which matches the six args pushed above in order + type
        // (all `CudaSlice<f32>` map to `float*`, `u32` maps to CUDA `int`
        // for positive dims). Out-of-range writes are prevented by the
        // `if (row < out_dim)` guard inside the kernel.
        unsafe { builder.launch(cfg) }.map_err(cuda_err)?;

        // Activation on hidden layers only. Output layer stays linear.
        if i != last {
            let kernel = match act {
                Activation::Relu => &cache.kernel_relu,
                Activation::Tanh => &cache.kernel_tanh,
            };
            let cfg_act = LaunchConfig::for_num_elems(out_dim);
            let mut b2 = stream.launch_builder(kernel);
            b2.arg(&*dst_ref);
            b2.arg(&out_dim);
            // SAFETY: kernel is (float* x, int n); we pass a `CudaSlice<f32>`
            // as `float*` and a `u32` as `int`. The kernel guards `i < n`.
            unsafe { b2.launch(cfg_act) }.map_err(cuda_err)?;
        }

        // Swap roles for the next layer.
        use_a_as_src = !use_a_as_src;
    }

    // After the final swap, the final output is in whichever buffer
    // `use_a_as_src` now points at (the "src" for a hypothetical next layer).
    let final_out_dim = cache.layers[last].out_dim;
    let final_buf: &CudaSlice<f32> = if use_a_as_src { &buf_a } else { &buf_b };

    let mut host = stream.memcpy_dtov(final_buf).map_err(cuda_err)?;
    // Trim to the meaningful prefix — scratch is padded to scratch_cap.
    host.truncate(final_out_dim);
    Ok(host)
}

/// Kernel source, compiled once per cache via NVRTC.
///
/// All three kernels are standard boilerplate — nothing device-specific,
/// so NVRTC's runtime arch detection picks the right target on both
/// the dev host (sm_89) and the pod (sm_120).
const KERNEL_SRC: &str = r#"
extern "C" __global__ void matmul_bias(
    const float* __restrict__ w,
    const float* __restrict__ b,
    const float* __restrict__ x,
    float* __restrict__ y,
    int in_dim,
    int out_dim)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= out_dim) return;
    float acc = b[row];
    const float* w_row = w + row * in_dim;
    for (int j = 0; j < in_dim; ++j) {
        acc += w_row[j] * x[j];
    }
    y[row] = acc;
}

extern "C" __global__ void relu_inplace(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    x[i] = v > 0.0f ? v : 0.0f;
}

extern "C" __global__ void tanh_inplace(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] = tanhf(x[i]);
}
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::probe_device;

    fn cuda_available() -> bool {
        probe_device().is_ok()
    }

    // --- CPU reference implementations ---

    fn cpu_matmul_bias(w: &[f32], b: &[f32], x: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
        assert_eq!(w.len(), in_dim * out_dim);
        assert_eq!(b.len(), out_dim);
        assert_eq!(x.len(), in_dim);
        let mut y = vec![0.0_f32; out_dim];
        for i in 0..out_dim {
            let mut acc = b[i];
            for j in 0..in_dim {
                acc += w[i * in_dim + j] * x[j];
            }
            y[i] = acc;
        }
        y
    }

    fn apply_act_cpu(v: &mut [f32], act: Activation) {
        match act {
            Activation::Relu => v.iter_mut().for_each(|x| *x = x.max(0.0)),
            Activation::Tanh => v.iter_mut().for_each(|x| *x = x.tanh()),
        }
    }

    fn cpu_forward(
        layers: &[(Vec<f32>, Vec<f32>, usize, usize)],
        input: &[f32],
        act: Activation,
    ) -> Vec<f32> {
        let mut h = input.to_vec();
        let last = layers.len() - 1;
        for (i, (w, b, in_dim, out_dim)) in layers.iter().enumerate() {
            let mut z = cpu_matmul_bias(w, b, &h, *in_dim, *out_dim);
            if i != last {
                apply_act_cpu(&mut z, act);
            }
            h = z;
        }
        h
    }

    // Simple linear congruential RNG — tests need reproducible weights
    // without pulling rand into the gpu crate's dev-deps.
    struct Lcg(u64);
    impl Lcg {
        fn new(seed: u64) -> Self {
            Self(seed | 1)
        }
        fn next_f32(&mut self) -> f32 {
            // Knuth MMIX constants → 32-bit-ish uniform in [-0.5, 0.5).
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let x = ((self.0 >> 33) as u32) as f32;
            x / (u32::MAX as f32) - 0.5
        }
        fn vec(&mut self, n: usize) -> Vec<f32> {
            (0..n).map(|_| self.next_f32()).collect()
        }
    }

    fn layer_refs(
        owned: &[(Vec<f32>, Vec<f32>, usize, usize)],
    ) -> Vec<(&[f32], &[f32], usize, usize)> {
        owned
            .iter()
            .map(|(w, b, i, o)| (w.as_slice(), b.as_slice(), *i, *o))
            .collect()
    }

    #[test]
    fn forward_small_relu_matches_cpu() {
        if !cuda_available() {
            eprintln!("skipping: no CUDA device on this host");
            return;
        }
        // 2 -> 8 -> 3 with ReLU hidden, linear output.
        let mut rng = Lcg::new(0xDEAD_BEEF);
        let w0 = rng.vec(2 * 8);
        let b0 = rng.vec(8);
        let w1 = rng.vec(8 * 3);
        let b1 = rng.vec(3);
        let owned = vec![(w0, b0, 2, 8), (w1, b1, 8, 3)];

        let x = vec![0.5_f32, -1.25];
        let expected = cpu_forward(&owned, &x, Activation::Relu);

        let cache = GpuWeightCache::new(layer_refs(&owned)).unwrap();
        let got = mlp_forward(&cache, &x, Activation::Relu).unwrap();

        assert_eq!(got.len(), expected.len());
        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-4,
                "idx {i}: gpu={g} cpu={e} diff={}",
                (g - e).abs()
            );
        }
    }

    #[test]
    fn forward_larger_tanh_matches_cpu() {
        if !cuda_available() {
            eprintln!("skipping: no CUDA device on this host");
            return;
        }
        // 100 -> 200 -> 50 -> 10 with tanh hidden.
        let mut rng = Lcg::new(0xC0FF_EE42);
        let sizes = [100_usize, 200, 50, 10];
        let owned: Vec<(Vec<f32>, Vec<f32>, usize, usize)> = sizes
            .windows(2)
            .map(|w| {
                let (in_dim, out_dim) = (w[0], w[1]);
                // Scale weights down a bit so tanh outputs don't saturate
                // uniformly and we actually exercise the activation.
                let scale = (1.0_f32 / in_dim as f32).sqrt();
                let w_vec: Vec<f32> = (0..in_dim * out_dim)
                    .map(|_| rng.next_f32() * scale)
                    .collect();
                let b_vec: Vec<f32> = (0..out_dim).map(|_| rng.next_f32() * 0.1).collect();
                (w_vec, b_vec, in_dim, out_dim)
            })
            .collect();

        let x: Vec<f32> = (0..sizes[0]).map(|_| rng.next_f32()).collect();
        let expected = cpu_forward(&owned, &x, Activation::Tanh);

        let cache = GpuWeightCache::new(layer_refs(&owned)).unwrap();
        let got = mlp_forward(&cache, &x, Activation::Tanh).unwrap();

        assert_eq!(got.len(), expected.len());
        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-4,
                "idx {i}: gpu={g} cpu={e} diff={}",
                (g - e).abs()
            );
        }
    }

    #[test]
    fn upload_download_round_trip_is_bit_identical() {
        if !cuda_available() {
            eprintln!("skipping: no CUDA device on this host");
            return;
        }
        let mut rng = Lcg::new(0x1234_5678);
        let w = rng.vec(4 * 6);
        let b = rng.vec(6);
        let owned = vec![(w.clone(), b.clone(), 4, 6)];

        let cache = GpuWeightCache::new(layer_refs(&owned)).unwrap();
        let (w_back, b_back) = cache.download_layer(0).unwrap();
        assert_eq!(w.len(), w_back.len());
        assert_eq!(b.len(), b_back.len());
        for (a, b2) in w.iter().zip(w_back.iter()) {
            assert_eq!(a.to_bits(), b2.to_bits(), "weight mismatch: {a} vs {b2}");
        }
        for (a, b2) in b.iter().zip(b_back.iter()) {
            assert_eq!(a.to_bits(), b2.to_bits(), "bias mismatch: {a} vs {b2}");
        }

        // Upload a different pattern and verify it round-trips too.
        let mut rng2 = Lcg::new(0xFEED_FACE);
        let w2 = rng2.vec(4 * 6);
        let b2 = rng2.vec(6);
        let mut cache = cache;
        cache.upload_layer(0, &w2, &b2).unwrap();
        let (w_back2, b_back2) = cache.download_layer(0).unwrap();
        for (a, c) in w2.iter().zip(w_back2.iter()) {
            assert_eq!(a.to_bits(), c.to_bits(), "re-upload weight mismatch");
        }
        for (a, c) in b2.iter().zip(b_back2.iter()) {
            assert_eq!(a.to_bits(), c.to_bits(), "re-upload bias mismatch");
        }
    }

    #[test]
    fn persistent_cache_1000_forwards_stable() {
        if !cuda_available() {
            eprintln!("skipping: no CUDA device on this host");
            return;
        }
        let mut rng = Lcg::new(0xAAAA_BBBB);
        let w0 = rng.vec(16 * 32);
        let b0 = rng.vec(32);
        let w1 = rng.vec(32 * 8);
        let b1 = rng.vec(8);
        let owned = vec![(w0, b0, 16, 32), (w1, b1, 32, 8)];

        let cache = GpuWeightCache::new(layer_refs(&owned)).unwrap();

        let x: Vec<f32> = (0..16).map(|_| rng.next_f32()).collect();
        let reference = mlp_forward(&cache, &x, Activation::Relu).unwrap();

        // Same input should produce identical output every time — kernels
        // are deterministic for fixed weights + input at a given block
        // layout, and NVRTC compiles once.
        for step in 0..1000 {
            let got = mlp_forward(&cache, &x, Activation::Relu).unwrap();
            assert_eq!(got.len(), reference.len(), "step {step} length drift");
            for (i, (g, r)) in got.iter().zip(reference.iter()).enumerate() {
                assert_eq!(g.to_bits(), r.to_bits(), "step {step} idx {i}: {g} vs {r}");
            }
        }
    }

    #[test]
    fn new_rejects_bad_shapes() {
        // These validations run before any device call, so run them even
        // when no GPU is present — they still exercise the error path.
        let w = vec![0.0_f32; 6];
        let b = vec![0.0_f32; 2];
        // in_dim*out_dim mismatch (wants 3*2=6 but says 4*2=8).
        let bad = vec![(w.as_slice(), b.as_slice(), 4, 2)];
        if cuda_available() {
            assert!(GpuWeightCache::new(bad).is_err());
        }
    }

    #[test]
    fn new_rejects_broken_chain() {
        if !cuda_available() {
            eprintln!("skipping: no CUDA device on this host");
            return;
        }
        // layer 0: 3 -> 4; layer 1: 5 -> 2 (doesn't chain with layer 0).
        let w0 = vec![0.0_f32; 3 * 4];
        let b0 = vec![0.0_f32; 4];
        let w1 = vec![0.0_f32; 5 * 2];
        let b1 = vec![0.0_f32; 2];
        let layers = vec![
            (w0.as_slice(), b0.as_slice(), 3, 4),
            (w1.as_slice(), b1.as_slice(), 5, 2),
        ];
        assert!(GpuWeightCache::new(layers).is_err());
    }

    #[test]
    fn forward_rejects_wrong_input_length() {
        if !cuda_available() {
            eprintln!("skipping: no CUDA device on this host");
            return;
        }
        let mut rng = Lcg::new(42);
        let w = rng.vec(3 * 2);
        let b = rng.vec(2);
        let owned = vec![(w, b, 3, 2)];
        let cache = GpuWeightCache::new(layer_refs(&owned)).unwrap();
        // Input of length 5, expected 3.
        let bad_input = vec![0.0_f32; 5];
        assert!(mlp_forward(&cache, &bad_input, Activation::Relu).is_err());
    }
}
