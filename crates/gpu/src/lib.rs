//! NIMCP V2 — GPU compute backend.
//!
//! # Dual-target design
//!
//! NIMCP V2 must run on two different NVIDIA GPUs with different CUDA toolkit
//! versions:
//!
//! | Host              | GPU                  | Compute cap | CUDA toolkit |
//! |-------------------|----------------------|-------------|--------------|
//! | Dev (Hetzner)     | RTX 4000 SFF Ada     | sm_89       | 12.0         |
//! | Pod (RunPod)      | RTX 5090 Blackwell   | sm_120      | 12.8+        |
//!
//! We solve the version skew by using [`cudarc`] with `dynamic-loading`:
//! libcudart.so is loaded at runtime, not link time. The same binary works
//! against any CUDA 12.x on either host.
//!
//! We solve the arch skew by compiling every `.cu` kernel with
//! `-gencode arch=compute_89,code=sm_89` and `-gencode arch=compute_120,code=sm_120`
//! (+ PTX fallback for forward compat). The CUDA runtime picks the right
//! variant per device automatically.
//!
//! # Capability probe
//!
//! [`probe_device`] queries the GPU's compute capability at startup so
//! higher layers can pick kernel variants. No hardcoded assumptions.
//!
//! # Phase 0 scope
//!
//! Types + capability probe. Actual kernels land in Phase 2.

#![forbid(unsafe_code)]

use thiserror::Error;

/// GPU backend errors.
#[derive(Debug, Error)]
pub enum GpuError {
    /// CUDA is compiled out. Build with `--features cuda`.
    #[error("CUDA support not compiled in (rebuild with --features cuda)")]
    NotAvailable,

    /// No CUDA-capable device found on this host.
    #[error("no CUDA device found")]
    NoDevice,

    /// Device capability is below the minimum this binary supports.
    #[error("device compute capability {found:?} below minimum {required:?}")]
    CapabilityTooLow {
        /// Capability detected on this device.
        found: (u32, u32),
        /// Minimum capability the binary was compiled for.
        required: (u32, u32),
    },

    /// CUDA runtime API returned an error.
    #[error("cuda error: {0}")]
    Cuda(String),
}

/// Device probe result — enough to choose kernel variants at runtime.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// Device index (0-based).
    pub ordinal: u32,
    /// Human-readable name (e.g. "NVIDIA RTX 4000 SFF Ada Generation").
    pub name: String,
    /// Compute capability as (major, minor). sm_89 => (8, 9); sm_120 => (12, 0).
    pub compute_cap: (u32, u32),
    /// Total global memory in bytes.
    pub total_memory: u64,
}

/// Probe the default CUDA device.
///
/// # Errors
///
/// - [`GpuError::NotAvailable`] if the crate was built without `--features cuda`.
/// - [`GpuError::NoDevice`] if no CUDA-capable device is present.
/// - [`GpuError::Cuda`] for any runtime failure.
pub fn probe_device() -> Result<DeviceInfo, GpuError> {
    #[cfg(not(feature = "cuda"))]
    {
        Err(GpuError::NotAvailable)
    }
    #[cfg(feature = "cuda")]
    {
        // Phase 0 stub — Phase 2 will implement via cudarc::driver::CudaDevice
        // + cudarc::driver::result::device_get_attribute for capability.
        tracing::warn!("probe_device: stub implementation (Phase 0)");
        Err(GpuError::NotAvailable)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_returns_error_in_phase_0() {
        let r = probe_device();
        assert!(r.is_err(), "Phase 0 stub always errors");
    }

    #[test]
    fn device_info_is_debug() {
        let d = DeviceInfo {
            ordinal: 0,
            name: "RTX 4000 Ada".into(),
            compute_cap: (8, 9),
            total_memory: 20 * 1024 * 1024 * 1024,
        };
        let s = format!("{:?}", d);
        assert!(s.contains("4000"));
    }
}
