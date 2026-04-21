//! Build script for the GPU crate.
//!
//! In Phase 0 this is nearly a no-op. In Phase 2 it will:
//!
//! 1. Detect whether nvcc is available on the build host.
//! 2. Compile any `.cu` kernel source files with multi-arch flags:
//!
//!    ```text
//!    nvcc -gencode arch=compute_89,code=sm_89       # dev: RTX 4000 Ada
//!         -gencode arch=compute_120,code=sm_120     # pod: RTX 5090 Blackwell
//!         -gencode arch=compute_89,code=compute_89  # PTX fallback
//!    ```
//!
//! 3. Emit link flags so cargo picks up the compiled object files.
//!
//! We use `cudarc` for the runtime API (dynamic loading) rather than
//! `cust` because cudarc loads libcudart.so at startup, letting one
//! binary run against CUDA 12.0 (dev) or 12.8 (pod) transparently.

fn main() {
    // Emit rerun hints.
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=kernels");

    // Phase 0: nothing to compile. Just announce our feature flags.
    if cfg!(feature = "cuda") {
        println!("cargo:rustc-cfg=has_cuda");
    }
}
