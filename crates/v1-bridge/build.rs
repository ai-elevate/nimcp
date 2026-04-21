//! Build script for `nimcp-v1-bridge`.
//!
//! Behaviour depends on the `cuda` feature (default ON):
//!
//! | Feature        | What this script does                                     |
//! |----------------|-----------------------------------------------------------|
//! | `cuda` (on)    | Run bindgen against V1 `include/nimcp.h`; link libnimcp.so |
//! | `cuda` (off)   | Emit nothing; `lib.rs` exposes stub types only             |
//!
//! The `cuda` off path is there for CPU-only CI hosts where the V1 C library
//! hasn't been built and `libnimcp.so` isn't available. In that mode the
//! crate still compiles so workspace-wide `cargo check`/`cargo test
//! --no-default-features` works.
//!
//! # Paths
//!
//! This crate lives at `<workspace_root>/crates/v1-bridge/`. Paths emitted to
//! cargo are relative to `CARGO_MANIFEST_DIR`:
//!
//! - Headers:  `../../include/nimcp.h`
//! - Library:  `../../build/lib/libnimcp.so`
//!
//! The rpath points one more level up (`$ORIGIN/../../../build/lib`) because
//! built test binaries live under `target/<profile>/deps/`.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=wrapper.h");

    #[cfg(feature = "cuda")]
    run_bindgen_and_link();
}

#[cfg(feature = "cuda")]
fn run_bindgen_and_link() {
    use std::env;
    use std::path::PathBuf;

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir
        .join("..")
        .join("..")
        .canonicalize()
        .expect("failed to canonicalize workspace root");

    let include_dir = workspace_root.join("include");
    let umbrella_header = include_dir.join("nimcp.h");
    let lib_dir = workspace_root.join("build").join("lib");

    // Re-run when the umbrella header or the built library changes.
    println!("cargo:rerun-if-changed={}", umbrella_header.display());
    println!(
        "cargo:rerun-if-changed={}",
        lib_dir.join("libnimcp.so").display()
    );

    // ---- Link flags ---------------------------------------------------------
    // Search path for link-time resolution.
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    // Dynamic link to libnimcp.so.
    println!("cargo:rustc-link-lib=dylib=nimcp");

    // Runtime search path so the built test binary finds libnimcp.so without
    // needing LD_LIBRARY_PATH. $ORIGIN resolves relative to the binary at load
    // time; binaries under target/<profile>/deps/ need 3 ups to reach the
    // workspace root, then /build/lib.
    println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/../../../build/lib");
    // Also add the absolute path as a fallback — helps when tests run from
    // other cwds (e.g. cargo nextest, IDE test runners).
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());

    // ---- bindgen ------------------------------------------------------------
    if !umbrella_header.exists() {
        panic!(
            "V1 umbrella header not found at {}. Build V1 first or disable the `cuda` feature.",
            umbrella_header.display()
        );
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let bindings_path = out_dir.join("bindings.rs");

    let bindings = bindgen::Builder::default()
        .header(umbrella_header.to_string_lossy())
        // nimcp.h does `#include "core/brain/learning/nimcp_brain_experience.h"`
        // etc. Give clang the include root so those resolve.
        .clang_arg(format!("-I{}", include_dir.display()))
        // The V1 code uses C99 + a few GNU extensions. Parse as plain C.
        .clang_arg("-std=c11")
        // Allowlist: types and functions we actually need. Leaving this
        // narrow keeps `bindings.rs` small and avoids pulling in every
        // internal type the umbrella header transitively exposes.
        .allowlist_function("nimcp_brain_.*")
        .allowlist_function("brain_.*")
        .allowlist_function("nimcp_version.*")
        .allowlist_function("nimcp_abi_layout_hash")
        .allowlist_type("nimcp_brain_.*")
        .allowlist_type("brain_.*")
        .allowlist_type("nimcp_status_t")
        .allowlist_type("nimcp_brain_size_t")
        .allowlist_type("nimcp_brain_task_t")
        .allowlist_type("nimcp_network_type_t")
        .allowlist_type("training_example_.*")
        .allowlist_type("brain_config_.*")
        .allowlist_var("NIMCP_.*")
        // Generate Rust enums as `#[repr(u32)]` structs with constants —
        // matches the C layout exactly and avoids UB when the C side
        // returns a value not in our Rust enum.
        .default_enum_style(bindgen::EnumVariation::NewType {
            is_bitfield: false,
            is_global: false,
        })
        .derive_debug(true)
        .derive_default(true)
        .derive_copy(true)
        // Treat every `typedef` as an opaque handle unless the type is in
        // our allowlist. This prevents bindgen chasing pointers into
        // headers it shouldn't (atomic types, pthread types, etc).
        .layout_tests(false)
        // Warn instead of fail on unknown clang warnings.
        .formatter(bindgen::Formatter::Rustfmt)
        .generate()
        .unwrap_or_else(|e| panic!("bindgen failed on {}: {}", umbrella_header.display(), e));

    bindings
        .write_to_file(&bindings_path)
        .unwrap_or_else(|e| panic!("failed to write {}: {}", bindings_path.display(), e));
}
