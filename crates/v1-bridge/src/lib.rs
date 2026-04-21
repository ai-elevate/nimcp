//! NIMCP V2 — FFI bridge to V1's C `libnimcp`.
//!
//! # Purpose
//!
//! V1 (the C implementation at `src/`) is a large, battle-tested compute
//! engine: 2M-neuron brain, STDP plasticity, CUDA kernels, 60+ cognitive
//! modules. V2 is not re-implementing any of that. Instead V2 treats V1 as
//! a black-box neural substrate and layers on top of it:
//!
//! - deterministic scheduling (`nimcp-scheduler`),
//! - an append-only event log (`nimcp-eventlog`),
//! - structured checkpoints (`nimcp-checkpoint`),
//! - actor-style isolation (`nimcp-core`).
//!
//! This crate is the boundary between those two worlds. It:
//!
//! 1. Runs `bindgen` at build time to generate raw Rust declarations for
//!    the V1 public C API (see `build.rs`).
//! 2. Wraps the opaque C handles in RAII types so V2 code cannot leak a
//!    brain, decision, or checkpoint.
//! 3. Converts every V1 status code into a typed [`BridgeError`].
//!
//! # Unsafety policy
//!
//! This is the FFI boundary, so `#![allow(unsafe_code)]` is set crate-wide.
//! Every individual `unsafe` block still carries a `// SAFETY:` comment
//! justifying the invariant.
//!
//! # Feature flags
//!
//! - `cuda` (default) — link against `libnimcp.so`, run bindgen, expose a
//!   real `Brain`.
//! - Without `cuda` — the crate still builds (stub types only), but every
//!   wrapper returns [`BridgeError::CudaUnavailable`]. This path exists so
//!   CPU-only CI hosts can run `cargo test --no-default-features` against
//!   the rest of the workspace.

#![allow(unsafe_code)]
#![allow(clippy::missing_safety_doc)]
// FFI wrappers document safety inline.
// The `cfg(not(feature = "cuda"))` early-return pattern used throughout the
// file would otherwise trip this lint on every wrapper method.
#![allow(clippy::needless_return)]

#[cfg(feature = "cuda")]
use std::ffi::CString;
use std::ffi::NulError;
use std::path::Path;

use thiserror::Error;

// -----------------------------------------------------------------------------
// Raw bindings
// -----------------------------------------------------------------------------

/// Raw bindgen output (with the `cuda` feature) or stub types (without it).
///
/// Kept in its own module so the `allow(non_camel_case_types)` and friends
/// only apply here.
#[allow(
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    unsafe_op_in_unsafe_fn,
    dead_code,
    clippy::all,
    missing_docs
)]
pub mod sys {
    #[cfg(feature = "cuda")]
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

    // Stub path: keep a few handle typedefs so the rest of the crate compiles.
    // Every public wrapper short-circuits to `BridgeError::CudaUnavailable`
    // before touching these, so they are never actually dereferenced.
    #[cfg(not(feature = "cuda"))]
    mod stub {
        #[repr(C)]
        pub struct nimcp_brain_handle {
            _private: [u8; 0],
        }
        pub type nimcp_brain_t = *mut nimcp_brain_handle;

        #[repr(u32)]
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum nimcp_status_t {
            NIMCP_OK = 0,
        }
    }

    #[cfg(not(feature = "cuda"))]
    pub use stub::*;
}

// -----------------------------------------------------------------------------
// Error type
// -----------------------------------------------------------------------------

/// Errors surfaced by the V1 bridge.
///
/// Any non-zero V1 `nimcp_status_t` is wrapped in `Ffi(code)`. We keep the
/// raw integer rather than mapping to a Rust enum so we don't lose resolution
/// when V1 adds new codes.
#[derive(Debug, Error)]
pub enum BridgeError {
    /// A C function returned a null pointer where a valid handle was expected.
    #[error("V1 returned a null handle")]
    Null,

    /// A C function returned a non-zero status code.
    #[error("V1 returned error code {0}")]
    Ffi(i32),

    /// A path or label contained an interior NUL byte and could not be
    /// converted to a C string.
    #[error("serialization error: {0}")]
    Serialization(String),

    /// The `cuda` feature is disabled, so `libnimcp.so` is not linked.
    #[error("V1 libnimcp is not linked (build with `--features cuda`)")]
    CudaUnavailable,
}

impl From<NulError> for BridgeError {
    fn from(err: NulError) -> Self {
        BridgeError::Serialization(err.to_string())
    }
}

/// Result type used throughout the bridge.
pub type Result<T, E = BridgeError> = std::result::Result<T, E>;

// -----------------------------------------------------------------------------
// Value types mirrored from V1 enums
// -----------------------------------------------------------------------------

/// Brain size preset. Mirrors `nimcp_brain_size_t` from `include/nimcp.h`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BrainSize {
    /// ~100 neurons, <1 MB, sub-millisecond inference.
    Tiny,
    /// ~1K neurons, ~10 MB.
    Small,
    /// ~10K neurons, ~50 MB.
    Medium,
    /// ~100K neurons, ~500 MB.
    Large,
}

impl BrainSize {
    #[allow(dead_code)] // Only used on the `cuda` path.
    fn as_raw(self) -> u32 {
        match self {
            BrainSize::Tiny => 0,
            BrainSize::Small => 1,
            BrainSize::Medium => 2,
            BrainSize::Large => 3,
        }
    }
}

/// Task template. Mirrors `nimcp_brain_task_t`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BrainTask {
    /// Multi-class classification — argmax over output vector.
    Classification,
    /// Continuous regression.
    Regression,
    /// Pattern recognition.
    PatternMatching,
    /// Temporal sequence learning.
    Sequence,
    /// Association learning.
    Association,
}

impl BrainTask {
    #[allow(dead_code)] // Only used on the `cuda` path.
    fn as_raw(self) -> u32 {
        match self {
            BrainTask::Classification => 0,
            BrainTask::Regression => 1,
            BrainTask::PatternMatching => 2,
            BrainTask::Sequence => 3,
            BrainTask::Association => 4,
        }
    }
}

/// Sensory modality passed to [`Brain::submit_sensory`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Modality {
    /// Visual input (pixel features).
    Visual,
    /// Auditory input (audio features / spectrogram).
    Auditory,
    /// Somatosensory / touch input.
    Somatosensory,
    /// Proprioceptive / motor input.
    Proprioceptive,
}

impl Modality {
    #[allow(dead_code)] // Only used on the `cuda` path.
    fn as_c_str(self) -> &'static std::ffi::CStr {
        // Every literal here ends in an explicit NUL, so `from_bytes_with_nul`
        // can't fail — we unwrap at compile-time-equivalent runtime cost.
        match self {
            Modality::Visual => c"visual",
            Modality::Auditory => c"auditory",
            Modality::Somatosensory => c"somatosensory",
            Modality::Proprioceptive => c"proprioceptive",
        }
    }
}

/// Spatial / structural shape of a sensory payload.
///
/// Mirrors the `width`, `height`, `channels`, `n_segments` parameters of
/// `nimcp_brain_submit_sensory`. Use [`SensoryShape::flat`] for 1D feature
/// vectors where only the element count matters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SensoryShape {
    /// Width in pixels or samples.
    pub width: u32,
    /// Height in pixels / 1 for audio.
    pub height: u32,
    /// Channel count (e.g. 3 for RGB, 1 for mono audio).
    pub channels: u32,
    /// Segment count (for temporal/segmented inputs); 0 when unused.
    pub n_segments: u32,
}

impl SensoryShape {
    /// A flat 1D feature vector — caller only knows the length.
    pub fn flat(len: u32) -> Self {
        Self {
            width: len,
            height: 1,
            channels: 1,
            n_segments: 0,
        }
    }
}

/// Configuration for [`Brain::new`].
#[derive(Debug, Clone)]
pub struct BrainConfig {
    /// Human-readable brain name (used for logs and audit entries).
    pub name: String,
    /// Size preset. Ignored if `neuron_count` is set.
    pub size: BrainSize,
    /// Task template.
    pub task: BrainTask,
    /// Input feature dimension.
    pub num_inputs: u32,
    /// Output dimension (class count for classification).
    pub num_outputs: u32,
    /// Exact neuron count; `None` selects the count implied by `size`.
    pub neuron_count: Option<u32>,
}

impl BrainConfig {
    /// Convenience constructor for a small tiny brain used in smoke tests.
    pub fn tiny(name: impl Into<String>, num_inputs: u32, num_outputs: u32) -> Self {
        Self {
            name: name.into(),
            size: BrainSize::Tiny,
            task: BrainTask::Classification,
            num_inputs,
            num_outputs,
            neuron_count: None,
        }
    }
}

/// Result of a full cognitive decision — what `nimcp_brain_decide_full`
/// returns, repackaged into owned Rust types.
#[derive(Debug, Clone)]
pub struct DecisionResult {
    /// Predicted label (first NUL-terminated portion of the output buffer).
    pub label: String,
    /// Confidence score in `[0.0, 1.0]`.
    pub confidence: f32,
    /// Human-readable explanation emitted by the cognitive stack.
    pub explanation: String,
    /// Raw output activation vector.
    pub output: Vec<f32>,
    /// Number of neurons that were active in the forward pass.
    pub active_neurons: u32,
    /// Fraction of neurons that were inactive (`[0.0, 1.0]`, higher = sparser).
    pub sparsity: f32,
    /// Wall-clock inference time reported by V1, in microseconds.
    pub inference_time_us: u64,
}

// -----------------------------------------------------------------------------
// Brain RAII wrapper
// -----------------------------------------------------------------------------

/// Safe wrapper around a V1 brain handle.
///
/// The handle is owned — `Drop` calls `nimcp_brain_destroy`. `Brain` is not
/// `Clone` because V1 brain handles are not cheap to duplicate (and the V1
/// API has no "clone" function; saving + loading is the closest analogue).
///
/// # Thread safety
///
/// The V1 brain is documented as single-writer. We mark `Brain: Send` so a
/// V2 actor can own and move it between threads, but we deliberately do not
/// mark it `Sync` — concurrent calls from multiple threads are unsafe at the
/// V1 level.
pub struct Brain {
    raw: sys::nimcp_brain_t,
}

// SAFETY: The V1 brain handle is safe to move across threads as long as only
// one thread calls into it at a time. `Brain` does not implement `Sync`, so
// the borrow checker enforces that constraint for us.
unsafe impl Send for Brain {}

impl std::fmt::Debug for Brain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Brain")
            .field("raw", &(self.raw as *const ()))
            .finish()
    }
}

impl Brain {
    /// Create a new brain from `config`.
    ///
    /// Routes to either `nimcp_brain_create` or `nimcp_brain_create_with_neurons`
    /// depending on whether `config.neuron_count` is set.
    pub fn new(config: &BrainConfig) -> Result<Self> {
        #[cfg(not(feature = "cuda"))]
        {
            let _ = config;
            return Err(BridgeError::CudaUnavailable);
        }

        #[cfg(feature = "cuda")]
        {
            let c_name = CString::new(config.name.as_str())?;

            // SAFETY: c_name is a valid NUL-terminated C string living for the
            // duration of the call. The V1 API copies the string internally;
            // we can drop c_name after this returns. All other args are
            // plain values (enum discriminants, u32 dims).
            // bindgen emits `nimcp_brain_size_t` and `nimcp_brain_task_t` as
            // `#[repr(transparent)] struct(c_uint)` newtypes, so we must
            // construct them explicitly rather than rely on an integer cast.
            let size_raw = sys::nimcp_brain_size_t(config.size.as_raw() as _);
            let task_raw = sys::nimcp_brain_task_t(config.task.as_raw() as _);

            let raw = unsafe {
                match config.neuron_count {
                    None => sys::nimcp_brain_create(
                        c_name.as_ptr(),
                        size_raw,
                        task_raw,
                        config.num_inputs,
                        config.num_outputs,
                    ),
                    Some(n) => sys::nimcp_brain_create_with_neurons(
                        c_name.as_ptr(),
                        task_raw,
                        config.num_inputs,
                        config.num_outputs,
                        n,
                    ),
                }
            };

            if raw.is_null() {
                Err(BridgeError::Null)
            } else {
                tracing::debug!(name = %config.name, "created V1 brain");
                Ok(Brain { raw })
            }
        }
    }

    /// Learn from one `(features, target)` example.
    ///
    /// Wraps `nimcp_brain_learn_vector`. Returns the last-observed loss
    /// after the step (`nimcp_brain_get_last_loss`). Pass `label=None` for
    /// unlabeled distillation-style training.
    pub fn learn_vector(
        &mut self,
        features: &[f32],
        target: &[f32],
        _lr: f32,
        label: Option<&str>,
        confidence: f32,
    ) -> Result<f32> {
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (features, target, label, confidence);
            return Err(BridgeError::CudaUnavailable);
        }

        #[cfg(feature = "cuda")]
        {
            // Hold the CString alive for the whole call. We can't collapse
            // this into a one-liner because `as_ptr()` would dangle.
            let c_label = match label {
                Some(l) => Some(CString::new(l)?),
                None => None,
            };
            let label_ptr = c_label
                .as_ref()
                .map(|s| s.as_ptr())
                .unwrap_or(std::ptr::null());

            // SAFETY: `features`/`target` live for the whole call via the
            // slice borrow. V1 only reads these. `label_ptr` is either null
            // or points into `c_label`, which outlives the call.
            let status = unsafe {
                sys::nimcp_brain_learn_vector(
                    self.raw,
                    features.as_ptr(),
                    features.len() as u32,
                    target.as_ptr(),
                    target.len() as u32,
                    label_ptr,
                    confidence,
                )
            };
            check_status(status)?;

            // SAFETY: `self.raw` is still alive — Drop hasn't fired. V1
            // guarantees `get_last_loss` is safe even if no learning has
            // happened (returns -1 in that case).
            let loss = unsafe { sys::nimcp_brain_get_last_loss(self.raw) };
            Ok(loss)
        }
    }

    /// Full cognitive decision — label, confidence, explanation, raw output.
    ///
    /// Wraps `nimcp_brain_decide_full`. Allocates output buffers sized for
    /// `NIMCP_MAX_LABEL_SIZE` (64) and `NIMCP_MAX_EXPLANATION_SIZE` (256).
    /// Output vector is sized to the brain's full output dimension (cap of
    /// 4096 floats).
    pub fn decide_full(&mut self, features: &[f32]) -> Result<DecisionResult> {
        #[cfg(not(feature = "cuda"))]
        {
            let _ = features;
            return Err(BridgeError::CudaUnavailable);
        }

        #[cfg(feature = "cuda")]
        {
            // Constants mirror the NIMCP_MAX_* defines in `include/nimcp.h`.
            const LABEL_CAP: usize = 64;
            const EXPLANATION_CAP: usize = 256;
            const OUTPUT_CAP: usize = 4096;

            let mut label_buf = vec![0u8; LABEL_CAP];
            let mut explanation_buf = vec![0u8; EXPLANATION_CAP];
            let mut output_buf = vec![0.0f32; OUTPUT_CAP];
            let mut output_size: u32 = OUTPUT_CAP as u32;
            let mut confidence: f32 = 0.0;
            let mut active_neurons: u32 = 0;
            let mut sparsity: f32 = 0.0;
            let mut inference_time_us: u64 = 0;

            // SAFETY: Every pointer below refers to a buffer or stack local
            // that lives for the whole call. The V1 API writes into the
            // buffers but does not retain the pointers.
            let status = unsafe {
                sys::nimcp_brain_decide_full(
                    self.raw,
                    features.as_ptr(),
                    features.len() as u32,
                    label_buf.as_mut_ptr() as *mut _,
                    &mut confidence,
                    explanation_buf.as_mut_ptr() as *mut _,
                    output_buf.as_mut_ptr(),
                    &mut output_size,
                    &mut active_neurons,
                    &mut sparsity,
                    &mut inference_time_us,
                )
            };
            check_status(status)?;

            output_buf.truncate(output_size as usize);

            Ok(DecisionResult {
                label: cstr_buf_to_string(&label_buf),
                confidence,
                explanation: cstr_buf_to_string(&explanation_buf),
                output: output_buf,
                active_neurons,
                sparsity,
                inference_time_us,
            })
        }
    }

    /// Stage sensory data for the next [`Brain::decide_full`] call.
    ///
    /// Wraps `nimcp_brain_submit_sensory`.
    pub fn submit_sensory(
        &mut self,
        modality: Modality,
        data: &[f32],
        shape: SensoryShape,
    ) -> Result<()> {
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (modality, data, shape);
            return Err(BridgeError::CudaUnavailable);
        }

        #[cfg(feature = "cuda")]
        {
            let modality_c = modality.as_c_str();

            // SAFETY: `modality_c` is a 'static C literal. `data.as_ptr()`
            // is valid for `data.len()` reads for the duration of the call.
            let status = unsafe {
                sys::nimcp_brain_submit_sensory(
                    self.raw,
                    modality_c.as_ptr(),
                    data.as_ptr(),
                    data.len() as u32,
                    shape.width,
                    shape.height,
                    shape.channels,
                    shape.n_segments,
                )
            };
            check_status(status)
        }
    }

    /// Persist the brain to `path`. Wraps `nimcp_brain_save`.
    pub fn save(&self, path: &Path) -> Result<()> {
        #[cfg(not(feature = "cuda"))]
        {
            let _ = path;
            return Err(BridgeError::CudaUnavailable);
        }

        #[cfg(feature = "cuda")]
        {
            let c_path = path_to_cstring(path)?;

            // SAFETY: c_path lives for the call. V1 copies the string.
            let status = unsafe { sys::nimcp_brain_save(self.raw, c_path.as_ptr()) };
            check_status(status)
        }
    }

    /// Load a previously saved brain from `path`. Wraps `nimcp_brain_load`.
    pub fn load(path: &Path) -> Result<Self> {
        #[cfg(not(feature = "cuda"))]
        {
            let _ = path;
            return Err(BridgeError::CudaUnavailable);
        }

        #[cfg(feature = "cuda")]
        {
            let c_path = path_to_cstring(path)?;

            // SAFETY: c_path lives for the call. The returned handle is
            // either null (error) or a valid brain owned by us.
            let raw = unsafe { sys::nimcp_brain_load(c_path.as_ptr()) };
            if raw.is_null() {
                Err(BridgeError::Null)
            } else {
                Ok(Brain { raw })
            }
        }
    }

    /// Lightweight liveness check: returns `true` iff the brain reports a
    /// non-zero neuron count.
    ///
    /// V1 has no explicit `ping` function, so we use
    /// `nimcp_brain_get_neuron_count() > 0` as a proxy.
    pub fn ping(&self) -> bool {
        self.neuron_count() > 0
    }

    /// Current hidden-layer neuron count. Wraps `nimcp_brain_get_neuron_count`.
    pub fn neuron_count(&self) -> u32 {
        #[cfg(not(feature = "cuda"))]
        {
            return 0;
        }

        #[cfg(feature = "cuda")]
        {
            if self.raw.is_null() {
                return 0;
            }
            // SAFETY: self.raw is non-null and owned by us.
            unsafe { sys::nimcp_brain_get_neuron_count(self.raw) }
        }
    }
}

impl Drop for Brain {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        {
            if !self.raw.is_null() {
                // SAFETY: self.raw was returned by V1 `nimcp_brain_create*` or
                // `nimcp_brain_load`, neither returned before. `Drop` fires
                // exactly once, so double-free is impossible.
                unsafe { sys::nimcp_brain_destroy(self.raw) };
                self.raw = std::ptr::null_mut();
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

/// Convert a V1 status code into a `Result<()>`.
#[cfg(feature = "cuda")]
fn check_status(status: sys::nimcp_status_t) -> Result<()> {
    // bindgen emits `nimcp_status_t` as a NewType struct wrapping a u32.
    // The NIMCP_OK constant on it is 0. Treat any non-zero value as an error.
    let code = status.0 as i32;
    if code == 0 {
        Ok(())
    } else {
        Err(BridgeError::Ffi(code))
    }
}

/// Convert a `Path` into an owned `CString`. Fails if the path is not valid
/// UTF-8 or contains an interior NUL.
#[cfg(feature = "cuda")]
fn path_to_cstring(path: &Path) -> Result<CString> {
    let s = path
        .to_str()
        .ok_or_else(|| BridgeError::Serialization(format!("non-UTF-8 path: {}", path.display())))?;
    Ok(CString::new(s)?)
}

/// Convert a NUL-terminated byte buffer (as V1 writes into `char*` outputs)
/// into an owned `String`, stopping at the first NUL. Non-UTF-8 bytes are
/// replaced with U+FFFD rather than returning an error — V1 labels are
/// guaranteed ASCII in practice but we don't want to panic on a stray byte.
#[cfg(feature = "cuda")]
fn cstr_buf_to_string(buf: &[u8]) -> String {
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    String::from_utf8_lossy(&buf[..end]).into_owned()
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn brain_size_raw_matches_c_enum() {
        assert_eq!(BrainSize::Tiny.as_raw(), 0);
        assert_eq!(BrainSize::Small.as_raw(), 1);
        assert_eq!(BrainSize::Medium.as_raw(), 2);
        assert_eq!(BrainSize::Large.as_raw(), 3);
    }

    #[test]
    fn brain_task_raw_matches_c_enum() {
        assert_eq!(BrainTask::Classification.as_raw(), 0);
        assert_eq!(BrainTask::Regression.as_raw(), 1);
        assert_eq!(BrainTask::PatternMatching.as_raw(), 2);
        assert_eq!(BrainTask::Sequence.as_raw(), 3);
        assert_eq!(BrainTask::Association.as_raw(), 4);
    }

    #[test]
    fn sensory_shape_flat() {
        let s = SensoryShape::flat(128);
        assert_eq!(s.width, 128);
        assert_eq!(s.height, 1);
        assert_eq!(s.channels, 1);
        assert_eq!(s.n_segments, 0);
    }

    #[test]
    fn bridge_config_tiny_defaults() {
        let cfg = BrainConfig::tiny("unit_test", 4, 2);
        assert_eq!(cfg.num_inputs, 4);
        assert_eq!(cfg.num_outputs, 2);
        assert!(matches!(cfg.size, BrainSize::Tiny));
        assert!(matches!(cfg.task, BrainTask::Classification));
        assert!(cfg.neuron_count.is_none());
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn stub_build_returns_cuda_unavailable() {
        let cfg = BrainConfig::tiny("stub", 2, 2);
        let err = Brain::new(&cfg).expect_err("stub build must refuse to create a brain");
        assert!(matches!(err, BridgeError::CudaUnavailable));
    }
}
