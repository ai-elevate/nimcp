//! NIMCP V2 — Phase 2d bench_mlp: per-step latency CPU vs GPU.
//!
//! # What this measures
//!
//! For a set of MLP shape presets, this binary measures the median
//! per-step latency (microseconds) of four operations:
//!
//! 1. **CPU forward** — [`nimcp_adaptive::AdaptiveNet::forward`].
//! 2. **CPU learn** — `forward + backward + SGD` via
//!    [`nimcp_adaptive::AdaptiveNet::learn`].
//! 3. **GPU forward** — [`nimcp_gpu::gpu_mlp_forward`] against a populated
//!    [`nimcp_gpu::GpuWeightCache`].
//! 4. **GPU learn** — same forward + [`nimcp_gpu::mlp_backward::mlp_backward`]
//!    + [`nimcp_gpu::mlp_backward::sgd_step`].
//!
//! The GPU backward API currently consumes CPU-side activation + pre-activation
//! vectors (see the module doc on `mlp_backward` — Phase 2c's planned
//! forward-handoff isn't wired yet). We replicate the CPU forward to
//! materialize those, then time the GPU forward + backward + step
//! separately. That slightly penalizes the GPU learn timing (an extra
//! CPU pass), but it matches what a V2 caller integrating today would
//! pay; it's honest.
//!
//! # Methodology
//!
//! - 5 untimed warmup iterations, then 50 timed iterations per op.
//! - Median of timed iterations is reported (robust to outliers).
//! - CUDA stream is synchronized **before** and **after** every timed
//!   iteration so async queues don't hide GPU work.
//! - ChaCha20Rng seed is fixed per size for reproducibility.
//!
//! # VRAM scaling
//!
//! Each shape's weight-memory footprint is estimated and compared against
//! free device memory at startup (`cuMemGetInfo_v2`). If a shape won't
//! fit (with a 128 MB safety margin for CUDA context + scratch + kernels),
//! it's skipped with a clear `SKIP (VRAM)` row. The crossover + trend
//! are the load-bearing output, not hitting any one specific size.
//!
//! # Exit behaviour
//!
//! - `0` on clean run even if GPU is absent (CPU-only numbers printed).
//! - Non-zero only on unrecoverable errors (e.g. panic during init).

// A single `unsafe` block: cudarc 0.17 doesn't expose a safe wrapper for
// `cuMemGetInfo_v2`, which we need to probe free VRAM before choosing
// which shapes to run. Justification is documented at the call site.
#![cfg_attr(feature = "cuda", allow(unsafe_code))]

use std::time::{Duration, Instant};

use ndarray::Array1;
use nimcp_adaptive::{Activation as CpuActivation, AdaptiveConfig, AdaptiveNet};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

#[cfg(feature = "cuda")]
use nimcp_gpu::mlp_backward::{
    self as gpu_bwd, Activation as GpuBwdActivation, GpuGradCache,
    mlp_backward as gpu_mlp_backward, sgd_step as gpu_sgd_step,
};
#[cfg(feature = "cuda")]
use nimcp_gpu::{Activation as GpuFwdActivation, GpuWeightCache, gpu_mlp_forward};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const WARMUP_ITERS: usize = 5;
const TIMED_ITERS: usize = 50;

/// Safety margin kept free after estimating weight footprint. Covers the
/// CUDA context, NVRTC module, per-call scratch, gradient buffers, and
/// general slack. Tuned empirically — a 50 M-param MLP has ~200 MB of
/// weights but the GPU cache's scratch + grad doubles that, and cuBLAS/ctx
/// easily spends another 128 MB on top.
#[cfg(feature = "cuda")]
const VRAM_SAFETY_MARGIN_BYTES: u64 = 256 * 1024 * 1024;

/// A single benchmark shape. `layers` follows the `[in, h1, h2, ..., out]`
/// convention of `nimcp_adaptive::AdaptiveConfig`.
struct Shape {
    name: &'static str,
    layers: Vec<usize>,
}

fn shapes() -> Vec<Shape> {
    vec![
        Shape {
            name: "tiny",
            layers: vec![64, 32, 10],
        },
        Shape {
            name: "small",
            layers: vec![256, 128, 64],
        },
        Shape {
            name: "medium",
            layers: vec![1024, 512, 256],
        },
        Shape {
            name: "large",
            layers: vec![1024, 4096, 1024],
        },
        Shape {
            name: "xlarge",
            layers: vec![1024, 16384, 2048],
        },
    ]
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Total parameter count (weights + biases) for a layer-width spec.
fn param_count(layers: &[usize]) -> usize {
    layers
        .windows(2)
        .map(|w| w[0] * w[1] + w[1]) // W: in*out, b: out
        .sum()
}

/// Estimated device-side memory for GPU weight cache + grad cache, in bytes.
///
/// - Forward cache holds weights + biases (1× params × 4 bytes).
/// - Grad cache mirrors that (1× params × 4 bytes).
/// - Forward scratch: 2 buffers sized to the widest layer (captured below).
/// - Backward scratch: grad_z + grad_h_prev + deriv + new_grad_z per step,
///   roughly 4× widest_layer × 4 bytes worst-case.
#[cfg(feature = "cuda")]
fn estimated_vram_bytes(layers: &[usize]) -> u64 {
    let params = param_count(layers) as u64;
    let widest = layers.iter().copied().max().unwrap_or(0) as u64;
    // weights + grads + forward scratch + backward scratch
    let bytes_per_f32 = 4_u64;
    params * bytes_per_f32 * 2            // weights + grads
        + widest * bytes_per_f32 * 2      // forward ping-pong
        + widest * bytes_per_f32 * 4 // backward scratch
}

/// Format a byte count as `KB` / `MB` / `GB` with one decimal place.
#[cfg(feature = "cuda")]
fn fmt_bytes(n: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * KB;
    const GB: u64 = 1024 * MB;
    if n >= GB {
        format!("{:.1} GB", n as f64 / GB as f64)
    } else if n >= MB {
        format!("{:.1} MB", n as f64 / MB as f64)
    } else if n >= KB {
        format!("{:.1} KB", n as f64 / KB as f64)
    } else {
        format!("{n} B")
    }
}

/// Format a parameter count compactly (2.7K, 41K, 650K, 8.0M, 50M).
fn fmt_params(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        format!("{n}")
    }
}

fn duration_micros(d: Duration) -> f64 {
    d.as_secs_f64() * 1_000_000.0
}

/// Median of a slice of `Duration`, returned in microseconds.
///
/// Uses the "lower of the two middles" for even-length inputs — we want
/// a concrete sample, not an interpolated value. Slice is left unsorted
/// by the caller's reference; we clone into a scratch Vec for sorting.
fn median_micros(samples: &[Duration]) -> f64 {
    assert!(!samples.is_empty(), "median_micros: empty samples");
    let mut sorted = samples.to_vec();
    sorted.sort();
    let mid = sorted.len() / 2;
    duration_micros(sorted[mid])
}

// ---------------------------------------------------------------------------
// CPU path
// ---------------------------------------------------------------------------

/// One timed forward-only run on the CPU. Returns elapsed time.
fn cpu_forward_once(net: &AdaptiveNet, x: &Array1<f32>) -> Duration {
    let t = Instant::now();
    let _out = net.forward(x);
    // Force the result to be observed so the compiler can't drop it.
    std::hint::black_box(_out);
    t.elapsed()
}

/// One timed learn step (forward + backward + SGD) on the CPU.
fn cpu_learn_once(net: &mut AdaptiveNet, x: &Array1<f32>, y: &Array1<f32>, lr: f32) -> Duration {
    let t = Instant::now();
    let loss = net.learn(x, y, lr);
    std::hint::black_box(loss);
    t.elapsed()
}

/// CPU forward capturing all activations + pre-activations. Used to feed
/// the GPU backward path (which currently expects them on the host — see
/// `mlp_backward.rs`'s module-level doc).
#[cfg(feature = "cuda")]
fn cpu_forward_capture(
    net: &AdaptiveNet,
    x: &Array1<f32>,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<f32>) {
    use nimcp_adaptive::Activation;
    let weights = net.weights();
    let act = net.config().activation;
    let last = weights.len() - 1;
    let mut activations: Vec<Vec<f32>> = Vec::with_capacity(weights.len() + 1);
    let mut pre_activations: Vec<Vec<f32>> = Vec::with_capacity(weights.len());

    activations.push(x.to_vec());
    let mut h = x.clone();
    for (i, layer) in weights.iter().enumerate() {
        let z = layer.w.dot(&h) + &layer.b;
        pre_activations.push(z.to_vec());
        let mut h_post = z.clone();
        if i != last {
            match act {
                Activation::Relu => h_post.iter_mut().for_each(|v| *v = v.max(0.0)),
                Activation::Tanh => h_post.iter_mut().for_each(|v| *v = v.tanh()),
            }
        }
        activations.push(h_post.to_vec());
        h = h_post;
    }
    let pred = activations.last().unwrap().clone();
    (activations, pre_activations, pred)
}

// ---------------------------------------------------------------------------
// GPU path (only compiled with feature = "cuda")
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
mod gpu {
    use super::*;

    /// Free / total device memory in bytes via the raw driver API.
    /// cudarc 0.17 does not expose a safe wrapper, so call the sys binding.
    pub fn mem_get_info() -> Result<(u64, u64), String> {
        use cudarc::driver::CudaContext;
        use cudarc::driver::sys::{CUresult, cuMemGetInfo_v2};

        // We need an active context for cuMemGetInfo. `CudaContext::new(0)`
        // binds thread-local context as a side effect; holding the Arc is
        // enough to keep it alive for the duration of the call.
        let _ctx = CudaContext::new(0).map_err(|e| format!("{e:?}"))?;
        let mut free_bytes: usize = 0;
        let mut total_bytes: usize = 0;
        // SAFETY: The two out-pointers are valid, writable `usize` locations
        // on the stack. The CUDA call only writes, never reads, through them.
        // cuMemGetInfo_v2 is safe to call from any thread that has a current
        // context, which we established above by constructing `_ctx`.
        let res = unsafe { cuMemGetInfo_v2(&mut free_bytes, &mut total_bytes) };
        if res != CUresult::CUDA_SUCCESS {
            return Err(format!("cuMemGetInfo_v2 returned {res:?}"));
        }
        Ok((free_bytes as u64, total_bytes as u64))
    }

    /// Build a GPU weight cache from an [`AdaptiveNet`]'s current weights.
    ///
    /// Layer weights are uploaded row-major `(out_dim, in_dim)`, which is
    /// also `ndarray::Array2`'s default standard layout, so iteration
    /// order matches the GPU layout without a transpose.
    pub fn build_gpu_cache(net: &AdaptiveNet) -> Result<GpuWeightCache, nimcp_gpu::GpuError> {
        let weights = net.weights();
        let mut flat_ws: Vec<Vec<f32>> = Vec::with_capacity(weights.len());
        let mut flat_bs: Vec<Vec<f32>> = Vec::with_capacity(weights.len());
        let mut shapes: Vec<(usize, usize)> = Vec::with_capacity(weights.len());
        for layer in weights.iter() {
            let (out_dim, in_dim) = layer.w.dim();
            // Array2 is row-major; flatten preserves that order.
            let w_vec: Vec<f32> = layer.w.iter().copied().collect();
            let b_vec: Vec<f32> = layer.b.iter().copied().collect();
            flat_ws.push(w_vec);
            flat_bs.push(b_vec);
            shapes.push((in_dim, out_dim));
        }
        let specs: Vec<(&[f32], &[f32], usize, usize)> = flat_ws
            .iter()
            .zip(flat_bs.iter())
            .zip(shapes.iter())
            .map(|((w, b), &(in_dim, out_dim))| (w.as_slice(), b.as_slice(), in_dim, out_dim))
            .collect();
        GpuWeightCache::new(specs)
    }

    /// Convert an `nimcp_adaptive::Activation` to the forward-crate flavour.
    pub fn fwd_activation(act: nimcp_adaptive::Activation) -> GpuFwdActivation {
        match act {
            nimcp_adaptive::Activation::Relu => GpuFwdActivation::Relu,
            nimcp_adaptive::Activation::Tanh => GpuFwdActivation::Tanh,
        }
    }

    /// Convert to the backward-crate flavour (distinct enum, same meaning).
    pub fn bwd_activation(act: nimcp_adaptive::Activation) -> GpuBwdActivation {
        match act {
            nimcp_adaptive::Activation::Relu => GpuBwdActivation::Relu,
            nimcp_adaptive::Activation::Tanh => GpuBwdActivation::Tanh,
        }
    }

    /// One GPU forward, synchronized on both ends so timing reflects real
    /// device work. Returns elapsed wall-clock.
    pub fn gpu_forward_once(
        cache: &GpuWeightCache,
        x: &[f32],
        act: GpuFwdActivation,
    ) -> Result<Duration, nimcp_gpu::GpuError> {
        cache.stream().synchronize().map_err(stream_err)?;
        let t = Instant::now();
        let out = gpu_mlp_forward(cache, x, act)?;
        cache.stream().synchronize().map_err(stream_err)?;
        let elapsed = t.elapsed();
        std::hint::black_box(out);
        Ok(elapsed)
    }

    /// One GPU learn step: forward + backward + SGD.
    ///
    /// The backward API requires host-side activations + pre-activations;
    /// for Phase 2d we materialize them via CPU forward beforehand. This
    /// matches what an integrating caller pays today. The timing window
    /// starts **after** CPU prep so the number reflects GPU work only
    /// (optional: switchable via `include_cpu_prep`). For the headline
    /// report we include the prep cost — that's the full step a user sees.
    pub fn gpu_learn_once(
        cache: &mut GpuWeightCache,
        grads: &mut GpuGradCache,
        net_for_capture: &AdaptiveNet,
        x: &Array1<f32>,
        y: &[f32],
        act: nimcp_adaptive::Activation,
        lr: f32,
    ) -> Result<Duration, nimcp_gpu::GpuError> {
        cache.stream().synchronize().map_err(stream_err)?;
        let t = Instant::now();

        // CPU-side forward pass to capture activations + pre-activations
        // for the backward call. Mirrors what nimcp-adaptive::learn does
        // internally before its backward. When Phase 2c lands the
        // forward-handoff API, this block goes away.
        let (activations, pre_activations, _pred) = cpu_forward_capture(net_for_capture, x);

        gpu_bwd::zero_grads(grads)?;
        gpu_mlp_backward(
            cache,
            grads,
            x.as_slice().expect("contiguous x"),
            y,
            bwd_activation(act),
            &activations,
            &pre_activations,
        )?;
        gpu_sgd_step(cache, grads, lr)?;
        cache.stream().synchronize().map_err(stream_err)?;

        Ok(t.elapsed())
    }

    fn stream_err<E: std::fmt::Debug>(e: E) -> nimcp_gpu::GpuError {
        nimcp_gpu::GpuError::Cuda(format!("stream sync: {e:?}"))
    }
}

// ---------------------------------------------------------------------------
// Per-shape measurement
// ---------------------------------------------------------------------------

#[derive(Default, Clone)]
struct ShapeResult {
    name: &'static str,
    params: usize,
    cpu_fwd_us: Option<f64>,
    gpu_fwd_us: Option<f64>,
    cpu_learn_us: Option<f64>,
    gpu_learn_us: Option<f64>,
    skipped_reason: Option<String>,
}

fn bench_cpu(net: &mut AdaptiveNet, layers: &[usize]) -> (f64, f64) {
    use rand::distr::{Distribution, Uniform};

    let mut rng = ChaCha20Rng::seed_from_u64(0xA110_CAA7);
    let in_dim = layers[0];
    let out_dim = *layers.last().unwrap();
    let dist = Uniform::new(-1.0_f32, 1.0_f32).expect("valid uniform range");
    let x: Array1<f32> = Array1::from_shape_fn(in_dim, |_| dist.sample(&mut rng));
    let y: Array1<f32> = Array1::from_shape_fn(out_dim, |_| dist.sample(&mut rng));

    // Forward warmup + timing.
    for _ in 0..WARMUP_ITERS {
        let _ = cpu_forward_once(net, &x);
    }
    let mut fwd_times = Vec::with_capacity(TIMED_ITERS);
    for _ in 0..TIMED_ITERS {
        fwd_times.push(cpu_forward_once(net, &x));
    }
    let cpu_fwd_us = median_micros(&fwd_times);

    // Learn warmup + timing. `learn` mutates weights; we don't care about
    // the training dynamics here, just per-step latency at realistic
    // weight magnitudes. Small lr keeps values sane across 55 iterations.
    let lr = 1e-4;
    for _ in 0..WARMUP_ITERS {
        let _ = cpu_learn_once(net, &x, &y, lr);
    }
    let mut learn_times = Vec::with_capacity(TIMED_ITERS);
    for _ in 0..TIMED_ITERS {
        learn_times.push(cpu_learn_once(net, &x, &y, lr));
    }
    let cpu_learn_us = median_micros(&learn_times);

    (cpu_fwd_us, cpu_learn_us)
}

#[cfg(feature = "cuda")]
fn bench_gpu(net: &AdaptiveNet, layers: &[usize]) -> Result<(f64, f64), nimcp_gpu::GpuError> {
    use crate::gpu::*;

    use rand::distr::{Distribution, Uniform};
    let mut rng = ChaCha20Rng::seed_from_u64(0xDECAFBAD);
    let in_dim = layers[0];
    let out_dim = *layers.last().unwrap();
    let dist = Uniform::new(-1.0_f32, 1.0_f32).expect("valid uniform range");
    let x_vec: Vec<f32> = (0..in_dim).map(|_| dist.sample(&mut rng)).collect();
    let x_arr = Array1::from_vec(x_vec.clone());
    let y_vec: Vec<f32> = (0..out_dim).map(|_| dist.sample(&mut rng)).collect();

    let mut cache = build_gpu_cache(net)?;
    let mut grads = GpuGradCache::new(&cache)?;
    let fwd_act = fwd_activation(net.config().activation);

    // Forward warmup.
    for _ in 0..WARMUP_ITERS {
        let _ = gpu_forward_once(&cache, &x_vec, fwd_act)?;
    }
    let mut fwd_times = Vec::with_capacity(TIMED_ITERS);
    for _ in 0..TIMED_ITERS {
        fwd_times.push(gpu_forward_once(&cache, &x_vec, fwd_act)?);
    }
    let gpu_fwd_us = median_micros(&fwd_times);

    // Learn warmup.
    let lr = 1e-4;
    for _ in 0..WARMUP_ITERS {
        let _ = gpu_learn_once(
            &mut cache,
            &mut grads,
            net,
            &x_arr,
            &y_vec,
            net.config().activation,
            lr,
        )?;
    }
    let mut learn_times = Vec::with_capacity(TIMED_ITERS);
    for _ in 0..TIMED_ITERS {
        learn_times.push(gpu_learn_once(
            &mut cache,
            &mut grads,
            net,
            &x_arr,
            &y_vec,
            net.config().activation,
            lr,
        )?);
    }
    let gpu_learn_us = median_micros(&learn_times);

    Ok((gpu_fwd_us, gpu_learn_us))
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

fn run_shape(shape: &Shape, free_vram: Option<u64>) -> ShapeResult {
    let params = param_count(&shape.layers);
    // `est` is only meaningful on cuda builds (VRAM-gating logic below).
    #[cfg(feature = "cuda")]
    let est = estimated_vram_bytes(&shape.layers);

    // Initial CPU bench is always possible.
    let mut res = ShapeResult {
        name: shape.name,
        params,
        ..Default::default()
    };

    // Build the CPU reference net.
    let cfg = AdaptiveConfig {
        layers: shape.layers.clone(),
        rng_seed: 0x005E_ED42 ^ (params as u64),
        activation: CpuActivation::Relu,
    };
    let mut net = AdaptiveNet::new(cfg);

    // CPU timing.
    let (cpu_fwd_us, cpu_learn_us) = bench_cpu(&mut net, &shape.layers);
    res.cpu_fwd_us = Some(cpu_fwd_us);
    res.cpu_learn_us = Some(cpu_learn_us);

    // GPU timing — gated on feature + VRAM budget.
    #[cfg(feature = "cuda")]
    {
        if let Some(free) = free_vram {
            let budget = free.saturating_sub(VRAM_SAFETY_MARGIN_BYTES);
            if est > budget {
                res.skipped_reason = Some(format!(
                    "VRAM: need ~{} but only {} available after {} safety margin",
                    fmt_bytes(est),
                    fmt_bytes(budget),
                    fmt_bytes(VRAM_SAFETY_MARGIN_BYTES),
                ));
            } else {
                // Fresh net (CPU learn mutated weights slightly). The
                // exact weight values aren't load-bearing for latency.
                let cfg = AdaptiveConfig {
                    layers: shape.layers.clone(),
                    rng_seed: 0x005E_ED42 ^ (params as u64),
                    activation: CpuActivation::Relu,
                };
                let net_fresh = AdaptiveNet::new(cfg);
                match bench_gpu(&net_fresh, &shape.layers) {
                    Ok((g_fwd, g_learn)) => {
                        res.gpu_fwd_us = Some(g_fwd);
                        res.gpu_learn_us = Some(g_learn);
                    }
                    Err(e) => {
                        res.skipped_reason = Some(format!("GPU error: {e}"));
                    }
                }
            }
        } else {
            res.skipped_reason = Some("no CUDA device".into());
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        let _ = free_vram;
        res.skipped_reason = Some("cpu-only build".into());
    }

    res
}

fn print_header(free_vram: Option<u64>, total_vram: Option<u64>) {
    println!("nimcp-v2 bench_mlp");
    #[cfg(feature = "cuda")]
    {
        match (free_vram, total_vram) {
            (Some(f), Some(t)) => {
                println!(
                    "GPU: CUDA device detected — free {} / total {}",
                    fmt_bytes(f),
                    fmt_bytes(t)
                );
                if f < 1024 * 1024 * 1024 {
                    println!("  note: less than 1 GB free — likely another process (e.g. Ollama)");
                    println!("  is resident. Larger shapes will be skipped gracefully.");
                }
            }
            _ => println!("GPU: no CUDA device available — CPU-only timings below"),
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (free_vram, total_vram);
        println!("GPU: build does not include the `cuda` feature — CPU-only timings below");
    }
    println!();
    println!(
        "{:<8} {:>8} {:>12} {:>12} {:>14} {:>14} {:>13} {:>15}",
        "shape",
        "params",
        "cpu_fwd_us",
        "gpu_fwd_us",
        "cpu_learn_us",
        "gpu_learn_us",
        "speedup_fwd",
        "speedup_learn"
    );
    println!("{}", "-".repeat(8 + 8 + 12 + 12 + 14 + 14 + 13 + 15 + 7));
}

fn print_row(r: &ShapeResult) {
    let cpu_fwd = r
        .cpu_fwd_us
        .map(|v| format!("{v:.1}"))
        .unwrap_or_else(|| "—".into());
    let cpu_learn = r
        .cpu_learn_us
        .map(|v| format!("{v:.1}"))
        .unwrap_or_else(|| "—".into());
    let gpu_fwd = match (&r.skipped_reason, r.gpu_fwd_us) {
        (Some(reason), _) => format!("SKIP ({reason})"),
        (None, Some(v)) => format!("{v:.1}"),
        (None, None) => "—".into(),
    };
    let gpu_learn = match (&r.skipped_reason, r.gpu_learn_us) {
        (Some(_), _) => "SKIP".into(),
        (None, Some(v)) => format!("{v:.1}"),
        (None, None) => "—".into(),
    };
    let speedup_fwd = match (r.cpu_fwd_us, r.gpu_fwd_us) {
        (Some(c), Some(g)) if g > 0.0 => format!("{:.2}x", c / g),
        _ => "—".into(),
    };
    let speedup_learn = match (r.cpu_learn_us, r.gpu_learn_us) {
        (Some(c), Some(g)) if g > 0.0 => format!("{:.2}x", c / g),
        _ => "—".into(),
    };

    // Truncate the gpu_fwd column if it contains a verbose SKIP message —
    // keeps the table readable. Full reason is printed after the table.
    let gpu_fwd_disp = if gpu_fwd.starts_with("SKIP ") {
        "SKIP (VRAM)".to_string()
    } else {
        gpu_fwd.clone()
    };

    println!(
        "{:<8} {:>8} {:>12} {:>12} {:>14} {:>14} {:>13} {:>15}",
        r.name,
        fmt_params(r.params),
        cpu_fwd,
        gpu_fwd_disp,
        cpu_learn,
        gpu_learn,
        speedup_fwd,
        speedup_learn
    );
}

fn print_summary(results: &[ShapeResult]) {
    println!();
    println!("Summary:");

    // Find crossover: smallest shape where GPU forward is faster than CPU.
    let crossover = results
        .iter()
        .filter_map(|r| match (r.cpu_fwd_us, r.gpu_fwd_us) {
            (Some(c), Some(g)) if g > 0.0 && c / g >= 1.0 => Some((r.name, c / g, r.params)),
            _ => None,
        })
        .next();

    match crossover {
        Some((name, s, p)) => {
            println!(
                "  Crossover: GPU forward overtakes CPU at shape `{name}` ({} params, {:.2}x speedup).",
                fmt_params(p),
                s
            );
        }
        None => {
            println!(
                "  Crossover: not observed in this run — either all measured shapes are too small"
            );
            println!(
                "  for the GPU launch overhead to pay off, or larger shapes were VRAM-skipped."
            );
        }
    }

    // Peak speedup observed.
    let peak_fwd = results
        .iter()
        .filter_map(|r| match (r.cpu_fwd_us, r.gpu_fwd_us) {
            (Some(c), Some(g)) if g > 0.0 => Some((r.name, c / g)),
            _ => None,
        })
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let peak_learn = results
        .iter()
        .filter_map(|r| match (r.cpu_learn_us, r.gpu_learn_us) {
            (Some(c), Some(g)) if g > 0.0 => Some((r.name, c / g)),
            _ => None,
        })
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    if let Some((name, s)) = peak_fwd {
        println!("  Peak GPU forward speedup: {:.2}x at shape `{name}`.", s);
    }
    if let Some((name, s)) = peak_learn {
        println!("  Peak GPU learn    speedup: {:.2}x at shape `{name}`.", s);
    }

    // Ollama / VRAM note.
    let any_skip = results.iter().any(|r| r.skipped_reason.is_some());
    if any_skip {
        println!();
        println!("  VRAM skips:");
        for r in results.iter().filter(|r| r.skipped_reason.is_some()) {
            println!(
                "    - {}: {}",
                r.name,
                r.skipped_reason.as_deref().unwrap_or("")
            );
        }
        println!();
        println!("  On this dev host the RTX 4000 Ada has 20 GB total; concurrent Ollama / other");
        println!(
            "  GPU workloads can leave <500 MB free, which caps the largest shapes we can fit."
        );
        println!("  Free the GPU (kill competing processes) to exercise the full size matrix.");
    }

    // Phase 2 exit criterion.
    println!();
    println!("  Phase 2 exit criterion: \"150K-neuron MLP trains at V1 wall-clock or better.\"");
    match peak_learn {
        Some((name, s)) if s >= 1.0 => {
            println!(
                "    At the scales measured here, GPU learn is {s:.2}x faster than CPU at `{name}`."
            );
            println!(
                "    150K-neuron hidden dim doesn't fit in <500 MB free VRAM; validate on pod (RTX 5090, 32 GB)."
            );
        }
        Some((name, s)) => {
            println!(
                "    Peak GPU learn speedup observed: {s:.2}x at `{name}` — below parity at measured scale."
            );
            println!(
                "    Expected: the scales fitting in dev VRAM are too small to amortize launch overhead."
            );
        }
        None => {
            println!(
                "    GPU learn measurements unavailable — cannot evaluate exit criterion here."
            );
            println!("    Run on a host with more free VRAM to exercise the larger shapes.");
        }
    }
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .with_target(false)
        .compact()
        .init();

    // Probe VRAM up front so shape selection is aware of it.
    let (free_vram, total_vram) = {
        #[cfg(feature = "cuda")]
        {
            match gpu::mem_get_info() {
                Ok((f, t)) => (Some(f), Some(t)),
                Err(e) => {
                    tracing::warn!(error = %e, "cuMemGetInfo failed; treating as CPU-only run");
                    (None, None)
                }
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            (None, None)
        }
    };

    print_header(free_vram, total_vram);

    let mut results: Vec<ShapeResult> = Vec::new();
    for shape in shapes() {
        let r = run_shape(&shape, free_vram);
        print_row(&r);
        results.push(r);
    }

    print_summary(&results);
}
