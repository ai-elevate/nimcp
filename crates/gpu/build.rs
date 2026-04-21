//! Build script for the GPU crate.
//!
//! In Phase 0 this is nearly a no-op. In Phase 2 it will:
//!
//! 1. Detect whether nvcc is available on the build host.
//! 2. Compile any `.cu` kernel source files with multi-arch flags.
//!
//! # Build targets
//!
//! The Rust binary is dev-first, pod-validated. Two targets matter:
//!
//! | Host        | GPU                      | Arch     | CUDA |
//! |-------------|--------------------------|----------|------|
//! | Dev (here)  | RTX 4000 SFF Ada         | sm_89    | 12.0 |
//! | Pod         | RTX 5090 Blackwell       | sm_120   | 12.8 |
//!
//! `NIMCP_GPU_ARCH` env var controls which archs nvcc builds for:
//!
//! | Value       | Meaning                                             |
//! |-------------|-----------------------------------------------------|
//! | unset (default) | Auto-detect from local `nvidia-smi` compute cap |
//! | `dev`       | sm_89 only — fast compile, local dev iteration      |
//! | `pod`       | sm_120 only — release build for pod deploy          |
//! | `all`       | sm_89 + sm_120 + PTX fallback — CI / release tarball |
//! | `sm_XX`     | Pass directly to nvcc `-gencode arch=compute_XX,code=sm_XX` |
//!
//! We use `cudarc` for the CUDA runtime API (dynamic loading) rather than
//! `cust`: cudarc loads libcudart.so at startup, letting one Rust binary
//! run against CUDA 12.0 on dev or 12.8 on pod without a rebuild.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=kernels");
    println!("cargo:rerun-if-env-changed=NIMCP_GPU_ARCH");

    if cfg!(feature = "cuda") {
        println!("cargo:rustc-cfg=has_cuda");

        // Document the chosen arch so downstream code can log it.
        let arch = std::env::var("NIMCP_GPU_ARCH").unwrap_or_else(|_| "dev".to_string());
        println!("cargo:rustc-env=NIMCP_GPU_ARCH_RESOLVED={arch}");
    }
}
