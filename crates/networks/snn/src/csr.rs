//! Phase 3b — CSR synapse storage + `I_syn` forward kernel.
//!
//! Row-major CSR, one instance per destination population. Layout mirrors
//! V1's lightweight SNN, which was the pattern that actually worked at
//! 1.8M-neuron scale (see V1 commit `3a7aa5f7d` for the homeostatic
//! perf fix that depends on this flat layout).
//!
//! # Layout
//!
//! - `row_ptr[0..=n_post]` — `row_ptr[i]..row_ptr[i+1]` is the slice in
//!   `col_idx` / `weights` holding incoming synapses to post-neuron `i`.
//! - `col_idx[k]` — pre-neuron index in the source population's local
//!   numbering (NOT a global brain-wide id).
//! - `weights[k]` — synaptic efficacy, same length as `col_idx`.
//!
//! # I_syn forward
//!
//! For each post-neuron `i`,
//! `I_syn[i] = sum over k in [row_ptr[i], row_ptr[i+1])  of
//!             weights[k] * (pre_spikes[col_idx[k]] != 0)`.
//!
//! The CPU path is a plain scatter-gather loop; the GPU path (feature
//! `cuda`) compiles an NVRTC kernel once per [`CsrGpu`] and launches one
//! thread per post-neuron.

use rand::SeedableRng;
use rand::distr::{Distribution, Uniform};
use rand_chacha::ChaCha20Rng;
use thiserror::Error;

/// Stable population handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PopulationId(pub u32);

/// Errors returned when building a [`CsrSynapses`] from user input.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum CsrError {
    /// Triple references a pre-neuron index outside `[0, n_pre)`.
    #[error("pre-neuron index {got} >= n_pre {limit}")]
    PreOutOfRange {
        /// Offending index.
        got: u32,
        /// Declared `n_pre`.
        limit: u32,
    },
    /// Triple references a post-neuron index outside `[0, n_post)`.
    #[error("post-neuron index {got} >= n_post {limit}")]
    PostOutOfRange {
        /// Offending index.
        got: u32,
        /// Declared `n_post`.
        limit: u32,
    },
    /// More triples than a `u32` row-pointer can index.
    #[error("too many synapses: {0}")]
    Overflow(usize),
}

/// A neuron population — the unit CSR addresses source and destination by.
#[derive(Debug, Clone)]
pub struct Population {
    /// Stable identifier (unique within an SNN network).
    pub id: PopulationId,
    /// Number of neurons in this population.
    pub n_neurons: u32,
    /// Human-readable name (e.g. `"L4_exc"`).
    pub name: String,
    /// LIF parameters shared by every neuron in the population.
    pub lif: crate::lif::LifParams,
}

impl Population {
    /// Construct a population with the given id, size, name, and LIF params.
    pub fn new(
        id: PopulationId,
        n_neurons: u32,
        name: impl Into<String>,
        lif: crate::lif::LifParams,
    ) -> Self {
        Self {
            id,
            n_neurons,
            name: name.into(),
            lif,
        }
    }
}

/// CSR-formatted incoming synapses for a single destination population.
///
/// See the module docs for the layout invariant.
#[derive(Debug, Clone)]
pub struct CsrSynapses {
    /// `row_ptr[i]..row_ptr[i+1]` is the index range in [`col_idx`] /
    /// [`weights`] for incoming synapses to post-neuron `i`.
    /// Length `n_post + 1`; monotonically non-decreasing; `row_ptr[0] == 0`;
    /// `row_ptr[n_post] == col_idx.len()`.
    ///
    /// [`col_idx`]: Self::col_idx
    /// [`weights`]: Self::weights
    pub row_ptr: Vec<u32>,
    /// Flat pre-neuron indices in the source population's local numbering.
    /// Length equals `row_ptr[n_post]`.
    pub col_idx: Vec<u32>,
    /// Synapse weights, same length as [`col_idx`](Self::col_idx).
    pub weights: Vec<f32>,
    /// Source population — where pre-spikes come from.
    pub src: PopulationId,
    /// Destination population — which neurons these synapses project to.
    pub dst: PopulationId,
    /// Cached size of the source population.
    pub n_pre: u32,
    /// Cached size of the destination population.
    pub n_post: u32,
}

impl CsrSynapses {
    /// Build a CSR matrix from an unordered list of `(pre, post, weight)`
    /// triples.
    ///
    /// Triples are validated against `n_pre` / `n_post` and then bucketed
    /// by `post` into the CSR layout. Input order within a row is preserved
    /// relative to insertion order (a stable bucket sort), so identical
    /// triples + same `n_pre` / `n_post` give a bit-identical CSR.
    ///
    /// # Errors
    ///
    /// - [`CsrError::PreOutOfRange`] if any triple has `pre >= n_pre`.
    /// - [`CsrError::PostOutOfRange`] if any triple has `post >= n_post`.
    /// - [`CsrError::Overflow`] if the triple count exceeds `u32::MAX`.
    pub fn from_triples(
        src: PopulationId,
        dst: PopulationId,
        n_pre: u32,
        n_post: u32,
        triples: Vec<(u32, u32, f32)>,
    ) -> Result<Self, CsrError> {
        if triples.len() > u32::MAX as usize {
            return Err(CsrError::Overflow(triples.len()));
        }

        // First pass: validate bounds. Fails fast on the first bad triple.
        for &(pre, post, _w) in &triples {
            if pre >= n_pre {
                return Err(CsrError::PreOutOfRange {
                    got: pre,
                    limit: n_pre,
                });
            }
            if post >= n_post {
                return Err(CsrError::PostOutOfRange {
                    got: post,
                    limit: n_post,
                });
            }
        }

        // Second pass: count per-row to size row_ptr exactly.
        let n_post_usize = n_post as usize;
        let mut row_counts: Vec<u32> = vec![0; n_post_usize];
        for &(_pre, post, _w) in &triples {
            row_counts[post as usize] += 1;
        }

        // Cumulative sum into row_ptr. Length n_post + 1.
        let mut row_ptr: Vec<u32> = Vec::with_capacity(n_post_usize + 1);
        row_ptr.push(0);
        let mut running: u32 = 0;
        for c in &row_counts {
            running = running.checked_add(*c).ok_or(CsrError::Overflow(
                // Shouldn't happen: triples.len() already fit in u32.
                triples.len(),
            ))?;
            row_ptr.push(running);
        }

        // Third pass: scatter triples into col_idx / weights using a
        // running write-cursor per row (start-of-row copy).
        let total = row_ptr[n_post_usize] as usize;
        let mut col_idx: Vec<u32> = vec![0; total];
        let mut weights: Vec<f32> = vec![0.0; total];
        let mut cursor: Vec<u32> = row_ptr[..n_post_usize].to_vec();

        for (pre, post, w) in triples {
            let slot = cursor[post as usize] as usize;
            col_idx[slot] = pre;
            weights[slot] = w;
            cursor[post as usize] += 1;
        }

        Ok(Self {
            row_ptr,
            col_idx,
            weights,
            src,
            dst,
            n_pre,
            n_post,
        })
    }

    /// Random fan-in connectivity: each post-neuron samples exactly
    /// `fan_in` distinct pre-neurons (uniform without replacement), with
    /// weights drawn uniformly from `weight_init ± weight_jitter`.
    ///
    /// Deterministic for a given `rng` state: reseed with the same value
    /// to reproduce the CSR bit-identically.
    ///
    /// `fan_in` is clamped to `min(fan_in, n_pre)` so small source
    /// populations don't deadlock sampling-without-replacement.
    #[allow(clippy::too_many_arguments)] // API shape dictated by V2_PLAN Phase 3b.
    pub fn random_uniform(
        src: PopulationId,
        dst: PopulationId,
        n_pre: u32,
        n_post: u32,
        fan_in: u32,
        weight_init: f32,
        weight_jitter: f32,
        rng: &mut ChaCha20Rng,
    ) -> Self {
        let fan_in = fan_in.min(n_pre);
        let n_post_usize = n_post as usize;
        let total = (fan_in as usize).saturating_mul(n_post_usize);

        let mut row_ptr: Vec<u32> = Vec::with_capacity(n_post_usize + 1);
        let mut col_idx: Vec<u32> = Vec::with_capacity(total);
        let mut weights: Vec<f32> = Vec::with_capacity(total);

        // `Uniform::new` rejects an empty range, so short-circuit for the
        // zero-jitter case rather than failing at runtime.
        let w_lo = weight_init - weight_jitter;
        let w_hi = weight_init + weight_jitter;
        let weight_dist: Option<Uniform<f32>> = if weight_jitter > 0.0 {
            // Uniform::new is half-open [lo, hi); that's fine for our use.
            Some(Uniform::new(w_lo, w_hi).expect("valid weight range"))
        } else {
            None
        };

        let mut running: u32 = 0;
        row_ptr.push(0);

        // Reservoir-style sampling without replacement, implemented by
        // partial Fisher-Yates on a scratch index vector. `fan_in <= n_pre`
        // by construction, so `fan_in` swaps are enough.
        let mut scratch: Vec<u32> = (0..n_pre).collect();

        for _post in 0..n_post {
            // Partial Fisher-Yates: pick `fan_in` draws from `scratch`.
            // For each pick i, swap scratch[i] with scratch[rand in [i, n_pre))].
            for i in 0..fan_in as usize {
                // rand 0.9: `random_range` on the RNG picks a uniform int
                // over the half-open range. No `Uniform::new` needed.
                use rand::Rng;
                let j = rng.random_range(i..n_pre as usize);
                scratch.swap(i, j);
                let pre = scratch[i];
                let w = match &weight_dist {
                    Some(d) => d.sample(rng),
                    None => weight_init,
                };
                col_idx.push(pre);
                weights.push(w);
            }
            running += fan_in;
            row_ptr.push(running);
        }

        Self {
            row_ptr,
            col_idx,
            weights,
            src,
            dst,
            n_pre,
            n_post,
        }
    }

    /// Total number of synapses stored.
    pub fn n_synapses(&self) -> usize {
        self.col_idx.len()
    }

    /// Compute `I_syn[post] = sum over k in row i of weights[k] * spike[col_idx[k]]`
    /// on the CPU.
    ///
    /// `pre_spikes` is the source population's spike vector — a non-zero
    /// value at index `j` means pre-neuron `j` spiked this step.
    /// `out` must have length `self.n_post as usize`; it is overwritten
    /// (not accumulated) on each call.
    ///
    /// # Panics
    ///
    /// Panics if `pre_spikes.len() < self.n_pre as usize` or if
    /// `out.len() != self.n_post as usize`. Panicking beats silent
    /// corruption for a tight inner-loop precondition.
    pub fn i_syn_cpu(&self, pre_spikes: &[u8], out: &mut [f32]) {
        self.i_syn_cpu_with_pre_scale(pre_spikes, None, out);
    }

    /// Like [`Self::i_syn_cpu`] but scales each contributing synapse by
    /// an optional per-pre-neuron multiplier (short-term depression).
    /// `pre_scale[i]` is applied to all outgoing contributions of
    /// neuron `i`. `None` is equivalent to all-ones (no scaling).
    ///
    /// # Panics
    /// Same as [`Self::i_syn_cpu`], plus: when `Some(pre_scale)` is
    /// supplied, its length must be `>= n_pre`.
    // HOT PATH: called per edge per step.
    pub fn i_syn_cpu_with_pre_scale(
        &self,
        pre_spikes: &[u8],
        pre_scale: Option<&[f32]>,
        out: &mut [f32],
    ) {
        assert_eq!(
            out.len(),
            self.n_post as usize,
            "i_syn_cpu: out.len()={} but n_post={}",
            out.len(),
            self.n_post
        );
        assert!(
            pre_spikes.len() >= self.n_pre as usize,
            "i_syn_cpu: pre_spikes.len()={} but n_pre={}",
            pre_spikes.len(),
            self.n_pre
        );
        if let Some(ps) = pre_scale {
            assert!(
                ps.len() >= self.n_pre as usize,
                "i_syn_cpu: pre_scale.len()={} but n_pre={}",
                ps.len(),
                self.n_pre
            );
        }

        let row_pairs = self.row_ptr.windows(2);
        for (out_slot, row) in out.iter_mut().zip(row_pairs) {
            let row_start = row[0] as usize;
            let row_end = row[1] as usize;
            let mut s: f32 = 0.0;
            for k in row_start..row_end {
                let pre = self.col_idx[k] as usize;
                if pre_spikes[pre] != 0 {
                    let scale = pre_scale.map_or(1.0, |ps| ps[pre]);
                    s += self.weights[k] * scale;
                }
            }
            *out_slot = s;
        }
    }
}

// --- Convenience constructor for tests + higher-level callers ---

impl CsrSynapses {
    /// Seeded random fan-in constructor — wraps `random_uniform` with a
    /// `ChaCha20Rng` derived from `seed`. Bit-identical across platforms
    /// because `ChaCha20Rng` is portable.
    #[allow(clippy::too_many_arguments)] // Convenience wrapper for test ergonomics.
    pub fn random_uniform_seeded(
        src: PopulationId,
        dst: PopulationId,
        n_pre: u32,
        n_post: u32,
        fan_in: u32,
        weight_init: f32,
        weight_jitter: f32,
        seed: u64,
    ) -> Self {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        Self::random_uniform(
            src,
            dst,
            n_pre,
            n_post,
            fan_in,
            weight_init,
            weight_jitter,
            &mut rng,
        )
    }

    /// Build a topology-less CSR from a flat weights vector.
    ///
    /// Used by sibling modules (homeostatic, tests) that only need the
    /// flat `weights` buffer to exist — `row_ptr` / `col_idx` are left
    /// empty. The result is NOT valid for `I_syn` forward; use
    /// [`from_triples`](Self::from_triples) or
    /// [`random_uniform`](Self::random_uniform) for anything that
    /// propagates spikes.
    #[must_use]
    pub fn from_weights(weights: Vec<f32>) -> Self {
        Self {
            row_ptr: vec![0],
            col_idx: Vec::new(),
            weights,
            src: PopulationId(0),
            dst: PopulationId(0),
            n_pre: 0,
            n_post: 0,
        }
    }
}

// --- GPU backend (feature-gated) ---

#[cfg(feature = "cuda")]
mod gpu {
    use std::sync::Arc;

    use cudarc::driver::{
        CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
    };
    use nimcp_gpu::GpuError;

    use super::CsrSynapses;

    fn cuda_err<E: std::fmt::Debug>(e: E) -> GpuError {
        GpuError::Cuda(format!("{e:?}"))
    }

    /// Device-resident CSR matrix + the compiled `csr_i_syn` kernel.
    ///
    /// One instance per destination population. Holds `row_ptr`,
    /// `col_idx`, `weights`, and a pre-allocated `i_syn` output buffer on
    /// the device. The kernel module is compiled once at `new`.
    pub struct CsrGpu {
        pub(crate) row_ptr: CudaSlice<u32>,
        pub(crate) col_idx: CudaSlice<u32>,
        pub(crate) weights: CudaSlice<f32>,
        /// Scratch input buffer for `pre_spikes` — length `n_pre`.
        pub(crate) pre_spikes_buf: CudaSlice<u8>,
        /// Scratch output buffer — length `n_post`.
        pub(crate) i_syn_buf: CudaSlice<f32>,
        pub(crate) n_pre: u32,
        pub(crate) n_post: u32,
        pub(crate) n_syn: u32,

        pub(crate) ctx: Arc<CudaContext>,
        pub(crate) stream: Arc<CudaStream>,
        // Held so the loaded PTX stays alive for the kernel handle.
        #[allow(dead_code)]
        module: Arc<CudaModule>,
        kernel: CudaFunction,
    }

    impl std::fmt::Debug for CsrGpu {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("CsrGpu")
                .field("n_pre", &self.n_pre)
                .field("n_post", &self.n_post)
                .field("n_syn", &self.n_syn)
                .finish_non_exhaustive()
        }
    }

    impl CsrGpu {
        /// Upload a host-side [`CsrSynapses`] and compile the kernel.
        ///
        /// Fails with [`GpuError::Cuda`] if no CUDA device is available,
        /// if NVRTC compilation fails, or if any upload fails.
        pub fn new(csr: &CsrSynapses) -> Result<Self, GpuError> {
            let ctx = CudaContext::new(0).map_err(cuda_err)?;
            let stream = ctx.default_stream();

            let ptx = cudarc::nvrtc::compile_ptx(KERNEL_SRC).map_err(cuda_err)?;
            let module = ctx.load_module(ptx).map_err(cuda_err)?;
            let kernel = module.load_function("csr_i_syn").map_err(cuda_err)?;

            let row_ptr = stream.memcpy_stod(&csr.row_ptr).map_err(cuda_err)?;
            let col_idx = stream.memcpy_stod(&csr.col_idx).map_err(cuda_err)?;
            let weights = stream.memcpy_stod(&csr.weights).map_err(cuda_err)?;

            let pre_spikes_buf: CudaSlice<u8> = stream
                .alloc_zeros::<u8>(csr.n_pre as usize)
                .map_err(cuda_err)?;
            let i_syn_buf: CudaSlice<f32> = stream
                .alloc_zeros::<f32>(csr.n_post as usize)
                .map_err(cuda_err)?;

            let n_syn = u32::try_from(csr.col_idx.len()).map_err(|_| {
                GpuError::Cuda(format!("synapse count {} exceeds u32", csr.col_idx.len()))
            })?;

            tracing::info!(
                n_pre = csr.n_pre,
                n_post = csr.n_post,
                n_syn,
                "csr gpu buffers uploaded"
            );

            Ok(Self {
                row_ptr,
                col_idx,
                weights,
                pre_spikes_buf,
                i_syn_buf,
                n_pre: csr.n_pre,
                n_post: csr.n_post,
                n_syn,
                ctx,
                stream,
                module,
                kernel,
            })
        }

        /// Upload `pre_spikes`, launch the `csr_i_syn` kernel, then
        /// download the result into `out`.
        ///
        /// `pre_spikes.len()` must equal `self.n_pre as usize`. `out` is
        /// resized to `self.n_post as usize` on success (the caller may
        /// pass an empty `Vec`). Errors return without touching `out`.
        pub fn i_syn(&mut self, pre_spikes: &[u8], out: &mut Vec<f32>) -> Result<(), GpuError> {
            if pre_spikes.len() != self.n_pre as usize {
                return Err(GpuError::Cuda(format!(
                    "i_syn: pre_spikes.len()={} but n_pre={}",
                    pre_spikes.len(),
                    self.n_pre
                )));
            }

            // Upload spikes.
            self.stream
                .memcpy_htod(pre_spikes, &mut self.pre_spikes_buf)
                .map_err(cuda_err)?;

            // One thread per post-neuron.
            let n_post_u32 = self.n_post;
            let cfg = LaunchConfig::for_num_elems(n_post_u32);
            let mut builder = self.stream.launch_builder(&self.kernel);
            builder.arg(&self.row_ptr);
            builder.arg(&self.col_idx);
            builder.arg(&self.weights);
            builder.arg(&self.pre_spikes_buf);
            // Output buffer passed as `&CudaSlice` like in mlp_forward; the
            // kernel writes through the raw device pointer — writability is
            // a kernel-level concern, not a Rust borrow concern.
            builder.arg(&self.i_syn_buf);
            builder.arg(&n_post_u32);
            // SAFETY: the kernel signature is
            //   (const unsigned int* row_ptr,
            //    const unsigned int* col_idx,
            //    const float* weights,
            //    const unsigned char* pre_spikes,
            //    float* i_syn,
            //    int n_post)
            // which matches the six args pushed above in order + type:
            // - 3× `CudaSlice<u32>` / `CudaSlice<f32>` map to the pointer args;
            // - `CudaSlice<u8>` maps to `const unsigned char*`;
            // - `&mut CudaSlice<f32>` maps to `float*`;
            // - `u32` is passed through cudarc as a 4-byte CUDA `int` for the
            //   loop bound (n_post is non-negative and bounded by the buffer
            //   lengths we allocated, so no sign-issue).
            // The kernel's `if (i >= n_post) return` guard prevents OOB writes.
            unsafe { builder.launch(cfg) }.map_err(cuda_err)?;

            // Download result into `out`.
            let host = self.stream.memcpy_dtov(&self.i_syn_buf).map_err(cuda_err)?;
            out.clear();
            out.extend_from_slice(&host);
            Ok(())
        }

        /// Overwrite device-side `weights` from a host slice. Length must
        /// match the original synapse count.
        pub fn upload_weights(&mut self, weights: &[f32]) -> Result<(), GpuError> {
            if weights.len() != self.n_syn as usize {
                return Err(GpuError::Cuda(format!(
                    "upload_weights: got {} but expected {}",
                    weights.len(),
                    self.n_syn
                )));
            }
            self.stream
                .memcpy_htod(weights, &mut self.weights)
                .map_err(cuda_err)?;
            Ok(())
        }

        /// Download device-side `weights` back to host memory.
        pub fn download_weights(&self) -> Result<Vec<f32>, GpuError> {
            self.stream.memcpy_dtov(&self.weights).map_err(cuda_err)
        }

        /// Number of synapses held on device.
        pub fn n_synapses(&self) -> usize {
            self.n_syn as usize
        }

        /// Borrow the underlying CUDA context.
        pub fn context(&self) -> &Arc<CudaContext> {
            &self.ctx
        }

        /// Borrow the default stream.
        pub fn stream(&self) -> &Arc<CudaStream> {
            &self.stream
        }
    }

    /// NVRTC kernel source. One thread per post-neuron, scatter-gather
    /// over that neuron's incoming row in CSR.
    const KERNEL_SRC: &str = r#"
extern "C" __global__ void csr_i_syn(
    const unsigned int* __restrict__ row_ptr,
    const unsigned int* __restrict__ col_idx,
    const float* __restrict__ weights,
    const unsigned char* __restrict__ pre_spikes,
    float* __restrict__ i_syn,
    int n_post)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_post) return;
    float s = 0.0f;
    unsigned int row_start = row_ptr[i];
    unsigned int row_end   = row_ptr[i + 1];
    for (unsigned int k = row_start; k < row_end; ++k) {
        if (pre_spikes[col_idx[k]]) {
            s += weights[k];
        }
    }
    i_syn[i] = s;
}
"#;
}

#[cfg(feature = "cuda")]
pub use gpu::CsrGpu;

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;

    fn pid(n: u32) -> PopulationId {
        PopulationId(n)
    }

    #[test]
    fn from_triples_basic() {
        // 3 pre-neurons, 2 post-neurons, 5 synapses.
        // Rows:
        //   post=0: (pre=0, w=1.0), (pre=2, w=3.0)
        //   post=1: (pre=1, w=2.0), (pre=0, w=4.0), (pre=2, w=5.0)
        let triples = vec![
            (0_u32, 0_u32, 1.0_f32),
            (1, 1, 2.0),
            (2, 0, 3.0),
            (0, 1, 4.0),
            (2, 1, 5.0),
        ];
        let csr = CsrSynapses::from_triples(pid(0), pid(1), 3, 2, triples).unwrap();
        assert_eq!(csr.row_ptr, vec![0, 2, 5]);
        // Within each row the order is stable w.r.t. input order.
        assert_eq!(csr.col_idx, vec![0, 2, 1, 0, 2]);
        assert_eq!(csr.weights, vec![1.0, 3.0, 2.0, 4.0, 5.0]);
        assert_eq!(csr.n_synapses(), 5);
        assert_eq!(csr.src, pid(0));
        assert_eq!(csr.dst, pid(1));
        assert_eq!(csr.n_pre, 3);
        assert_eq!(csr.n_post, 2);
    }

    #[test]
    fn from_triples_empty_is_ok() {
        // Zero synapses, 2 post — row_ptr must still be [0, 0, 0].
        let csr = CsrSynapses::from_triples(pid(0), pid(1), 4, 2, Vec::new()).unwrap();
        assert_eq!(csr.row_ptr, vec![0, 0, 0]);
        assert!(csr.col_idx.is_empty());
        assert!(csr.weights.is_empty());
    }

    #[test]
    fn from_triples_bounds_checked() {
        // pre out of range
        let bad_pre = vec![(5_u32, 0_u32, 1.0_f32)];
        let err = CsrSynapses::from_triples(pid(0), pid(1), 3, 2, bad_pre).unwrap_err();
        assert_eq!(err, CsrError::PreOutOfRange { got: 5, limit: 3 });

        // post out of range
        let bad_post = vec![(0_u32, 7_u32, 1.0_f32)];
        let err = CsrSynapses::from_triples(pid(0), pid(1), 3, 2, bad_post).unwrap_err();
        assert_eq!(err, CsrError::PostOutOfRange { got: 7, limit: 2 });

        // exactly-at-boundary is invalid (indices are 0-based, exclusive of limit).
        let bad_pre_eq = vec![(3_u32, 0_u32, 1.0_f32)];
        let err = CsrSynapses::from_triples(pid(0), pid(1), 3, 2, bad_pre_eq).unwrap_err();
        assert_eq!(err, CsrError::PreOutOfRange { got: 3, limit: 3 });
    }

    #[test]
    fn random_uniform_respects_fan_in() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let n_pre: u32 = 64;
        let n_post: u32 = 16;
        let fan_in: u32 = 8;
        let csr =
            CsrSynapses::random_uniform(pid(0), pid(1), n_pre, n_post, fan_in, 0.5, 0.1, &mut rng);
        assert_eq!(csr.row_ptr.len(), n_post as usize + 1);
        assert_eq!(csr.n_synapses(), (fan_in as usize) * (n_post as usize));
        // Every row has exactly `fan_in` synapses...
        for i in 0..n_post as usize {
            let start = csr.row_ptr[i];
            let end = csr.row_ptr[i + 1];
            assert_eq!(end - start, fan_in, "row {i} has wrong fan-in");
            // ...and every pre-index in the row is distinct (sampling
            // without replacement).
            let row = &csr.col_idx[start as usize..end as usize];
            let mut sorted = row.to_vec();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(sorted.len(), row.len(), "row {i} has duplicate pre");
            // ...and every pre-index is in [0, n_pre).
            for &p in row {
                assert!(p < n_pre);
            }
        }
        // Weights live in the jittered band.
        for &w in &csr.weights {
            assert!(
                (0.4_f32..0.6_f32).contains(&w),
                "weight {w} outside jitter band"
            );
        }
    }

    #[test]
    fn random_uniform_handles_fan_in_equal_to_npre() {
        // Edge case: fan_in == n_pre means every post gets *all* pre's.
        let mut rng = ChaCha20Rng::seed_from_u64(7);
        let csr = CsrSynapses::random_uniform(pid(0), pid(1), 5, 3, 5, 1.0, 0.0, &mut rng);
        for i in 0..3 {
            let start = csr.row_ptr[i] as usize;
            let end = csr.row_ptr[i + 1] as usize;
            let mut row = csr.col_idx[start..end].to_vec();
            row.sort_unstable();
            assert_eq!(row, vec![0, 1, 2, 3, 4]);
        }
        // Zero jitter -> every weight exactly equals weight_init.
        for &w in &csr.weights {
            assert_eq!(w.to_bits(), 1.0_f32.to_bits());
        }
    }

    #[test]
    fn random_uniform_clamps_fan_in_above_npre() {
        // fan_in > n_pre is clamped rather than failing; behave as fan_in = n_pre.
        let mut rng = ChaCha20Rng::seed_from_u64(11);
        let csr = CsrSynapses::random_uniform(pid(0), pid(1), 3, 2, 10, 0.25, 0.0, &mut rng);
        assert_eq!(csr.n_synapses(), 3 * 2);
        for i in 0..2 {
            let start = csr.row_ptr[i] as usize;
            let end = csr.row_ptr[i + 1] as usize;
            assert_eq!(end - start, 3);
        }
    }

    #[test]
    fn random_uniform_deterministic() {
        // Same seed -> bit-identical CSR.
        let a = CsrSynapses::random_uniform_seeded(pid(0), pid(1), 50, 20, 6, 0.3, 0.2, 0xABCD);
        let b = CsrSynapses::random_uniform_seeded(pid(0), pid(1), 50, 20, 6, 0.3, 0.2, 0xABCD);
        assert_eq!(a.row_ptr, b.row_ptr);
        assert_eq!(a.col_idx, b.col_idx);
        assert_eq!(a.weights.len(), b.weights.len());
        for (x, y) in a.weights.iter().zip(b.weights.iter()) {
            assert_eq!(x.to_bits(), y.to_bits(), "weight drift at same seed");
        }

        // Different seed -> different CSR (with overwhelming probability).
        let c = CsrSynapses::random_uniform_seeded(pid(0), pid(1), 50, 20, 6, 0.3, 0.2, 0xDEAD);
        // At least the col_idx must differ somewhere.
        assert_ne!(a.col_idx, c.col_idx);
    }

    #[test]
    fn i_syn_cpu_matches_manual_sum() {
        // 4 pre, 3 post.
        // post=0: pre 0 (w=0.5), pre 2 (w=-1.0)
        // post=1: pre 1 (w=2.0)
        // post=2: pre 0 (w=1.5), pre 3 (w=-0.5), pre 2 (w=0.25)
        let triples = vec![
            (0_u32, 0_u32, 0.5_f32),
            (2, 0, -1.0),
            (1, 1, 2.0),
            (0, 2, 1.5),
            (3, 2, -0.5),
            (2, 2, 0.25),
        ];
        let csr = CsrSynapses::from_triples(pid(0), pid(1), 4, 3, triples).unwrap();

        // Case A: all spiked.
        let pre_spikes_all = vec![1_u8, 1, 1, 1];
        let mut out = vec![f32::NAN; 3];
        csr.i_syn_cpu(&pre_spikes_all, &mut out);
        // post 0: 0.5 + (-1.0) = -0.5
        // post 1: 2.0
        // post 2: 1.5 + (-0.5) + 0.25 = 1.25
        assert!((out[0] - (-0.5)).abs() < 1e-6);
        assert!((out[1] - 2.0).abs() < 1e-6);
        assert!((out[2] - 1.25).abs() < 1e-6);

        // Case B: only pre=2 spiked.
        let pre_spikes_2 = vec![0_u8, 0, 1, 0];
        let mut out = vec![0.0_f32; 3];
        csr.i_syn_cpu(&pre_spikes_2, &mut out);
        assert!((out[0] - (-1.0)).abs() < 1e-6);
        assert!((out[1] - 0.0).abs() < 1e-6);
        assert!((out[2] - 0.25).abs() < 1e-6);

        // Case C: pre=0 and pre=3 spiked.
        let pre_spikes_03 = vec![1_u8, 0, 0, 1];
        let mut out = vec![0.0_f32; 3];
        csr.i_syn_cpu(&pre_spikes_03, &mut out);
        assert!((out[0] - 0.5).abs() < 1e-6);
        assert!((out[1] - 0.0).abs() < 1e-6);
        assert!((out[2] - 1.0).abs() < 1e-6); // 1.5 + (-0.5)
    }

    #[test]
    fn i_syn_cpu_no_spikes_zero_current() {
        let mut rng = ChaCha20Rng::seed_from_u64(99);
        let csr = CsrSynapses::random_uniform(pid(0), pid(1), 32, 16, 5, 0.5, 0.2, &mut rng);
        let pre_spikes = vec![0_u8; 32];
        let mut out = vec![f32::NAN; 16];
        csr.i_syn_cpu(&pre_spikes, &mut out);
        for (i, &v) in out.iter().enumerate() {
            // Bit-exact zero: every row sums nothing when no pre spiked.
            assert_eq!(
                v.to_bits(),
                0.0_f32.to_bits(),
                "post {i} got non-zero current with no spikes"
            );
        }
    }

    #[test]
    fn i_syn_cpu_treats_nonzero_byte_as_spiked() {
        // u8 spike vector uses any non-zero byte as "spiked".
        let triples = vec![(0_u32, 0_u32, 1.0_f32), (1, 0, 2.0)];
        let csr = CsrSynapses::from_triples(pid(0), pid(1), 2, 1, triples).unwrap();
        let pre_spikes = vec![7_u8, 42]; // neither zero -> both count.
        let mut out = vec![0.0_f32; 1];
        csr.i_syn_cpu(&pre_spikes, &mut out);
        assert!((out[0] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn population_new_smoke() {
        let lif = crate::lif::LifParams::default();
        let p = Population::new(pid(7), 128, "L4_exc", lif);
        assert_eq!(p.id, pid(7));
        assert_eq!(p.n_neurons, 128);
        assert_eq!(p.name, "L4_exc");
        // Float field plumbing check — bit-exact compare avoids clippy::float_cmp.
        assert_eq!(p.lif.v_thresh.to_bits(), lif.v_thresh.to_bits());
    }

    // --- GPU tests (feature "cuda" only). Skip gracefully without a device. ---

    #[cfg(feature = "cuda")]
    mod gpu_tests {
        use super::*;

        fn cuda_available() -> bool {
            nimcp_gpu::probe_device().is_ok()
        }

        #[test]
        fn i_syn_cpu_gpu_match() {
            if !cuda_available() {
                eprintln!("skipping: no CUDA device on this host");
                return;
            }
            // 1000 pre, 500 post, fan-in 50, random spikes.
            let n_pre: u32 = 1000;
            let n_post: u32 = 500;
            let fan_in: u32 = 50;
            let csr = CsrSynapses::random_uniform_seeded(
                pid(0),
                pid(1),
                n_pre,
                n_post,
                fan_in,
                0.4,
                0.15,
                0xBEEF_CAFE,
            );

            // Random spike vector (~20% spiking).
            let mut rng = ChaCha20Rng::seed_from_u64(0xF00D_D00D);
            use rand::Rng;
            let pre_spikes: Vec<u8> = (0..n_pre as usize)
                .map(|_| if rng.random_bool(0.2) { 1_u8 } else { 0 })
                .collect();

            // CPU reference.
            let mut cpu_out = vec![0.0_f32; n_post as usize];
            csr.i_syn_cpu(&pre_spikes, &mut cpu_out);

            // GPU path.
            let mut gpu = CsrGpu::new(&csr).expect("CsrGpu::new");
            let mut gpu_out: Vec<f32> = Vec::new();
            gpu.i_syn(&pre_spikes, &mut gpu_out).expect("i_syn");

            assert_eq!(gpu_out.len(), cpu_out.len());
            for (i, (g, c)) in gpu_out.iter().zip(cpu_out.iter()).enumerate() {
                let diff = (g - c).abs();
                assert!(diff < 1e-5, "post {i}: gpu={g} cpu={c} diff={diff}");
            }
        }

        #[test]
        fn i_syn_gpu_zero_spikes() {
            if !cuda_available() {
                eprintln!("skipping: no CUDA device on this host");
                return;
            }
            let csr = CsrSynapses::random_uniform_seeded(
                pid(0),
                pid(1),
                128,
                64,
                12,
                0.3,
                0.05,
                0x1234_5678,
            );
            let pre_spikes = vec![0_u8; 128];
            let mut gpu = CsrGpu::new(&csr).unwrap();
            let mut out: Vec<f32> = Vec::new();
            gpu.i_syn(&pre_spikes, &mut out).unwrap();
            assert_eq!(out.len(), 64);
            for (i, &v) in out.iter().enumerate() {
                // Bit-exact zero on GPU too — deterministic zero-sum kernel path.
                assert_eq!(
                    v.to_bits(),
                    0.0_f32.to_bits(),
                    "post {i} got non-zero current with no spikes"
                );
            }
        }

        #[test]
        fn i_syn_gpu_rejects_wrong_spike_len() {
            if !cuda_available() {
                eprintln!("skipping: no CUDA device on this host");
                return;
            }
            let csr =
                CsrSynapses::random_uniform_seeded(pid(0), pid(1), 32, 16, 4, 0.2, 0.0, 0x999);
            let mut gpu = CsrGpu::new(&csr).unwrap();
            let bad = vec![0_u8; 31]; // wrong length
            let mut out = Vec::new();
            assert!(gpu.i_syn(&bad, &mut out).is_err());
        }

        #[test]
        fn upload_download_round_trips() {
            if !cuda_available() {
                eprintln!("skipping: no CUDA device on this host");
                return;
            }
            let csr = CsrSynapses::random_uniform_seeded(
                pid(0),
                pid(1),
                64,
                32,
                8,
                0.5,
                0.2,
                0xAABB_CCDD,
            );
            let original = csr.weights.clone();
            let gpu = CsrGpu::new(&csr).unwrap();

            // Download -> compare bit-identical to source.
            let round_trip = gpu.download_weights().unwrap();
            assert_eq!(round_trip.len(), original.len());
            for (a, b) in original.iter().zip(round_trip.iter()) {
                assert_eq!(a.to_bits(), b.to_bits(), "download mismatch");
            }

            // Upload a new pattern, download again, still bit-identical.
            let mut rng = ChaCha20Rng::seed_from_u64(0x5A5A_5A5A);
            use rand::Rng;
            let replacement: Vec<f32> = (0..csr.weights.len())
                .map(|_| rng.random_range(-1.0_f32..1.0))
                .collect();
            let mut gpu = gpu;
            gpu.upload_weights(&replacement).unwrap();
            let downloaded = gpu.download_weights().unwrap();
            for (a, b) in replacement.iter().zip(downloaded.iter()) {
                assert_eq!(a.to_bits(), b.to_bits(), "re-upload mismatch");
            }
        }

        #[test]
        fn upload_weights_rejects_wrong_len() {
            if !cuda_available() {
                eprintln!("skipping: no CUDA device on this host");
                return;
            }
            let csr =
                CsrSynapses::random_uniform_seeded(pid(0), pid(1), 16, 8, 4, 0.1, 0.0, 0xDEAD);
            let mut gpu = CsrGpu::new(&csr).unwrap();
            let bad = vec![0.0_f32; 7]; // wrong length
            assert!(gpu.upload_weights(&bad).is_err());
        }
    }
}
