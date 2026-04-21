//! Smoke tests for the V1 FFI bridge.
//!
//! Two test paths:
//!
//! - **CUDA enabled**: create a tiny V1 brain, confirm it pings, destroy it.
//!   Requires `libnimcp.so` built at `<workspace>/build/lib/`. Gated by
//!   `#[cfg(feature = "cuda")]` so `cargo test --no-default-features`
//!   skips it.
//!
//! - **Stub build**: confirms the crate compiles and the API surface is
//!   reachable when V1 is not linked, by asking for a brain and expecting
//!   `BridgeError::CudaUnavailable`.

use nimcp_v1_bridge::BrainConfig;
#[cfg(not(feature = "cuda"))]
use nimcp_v1_bridge::BridgeError;

#[cfg(feature = "cuda")]
#[test]
fn tiny_brain_lifecycle() {
    use nimcp_v1_bridge::Brain;

    let cfg = BrainConfig::tiny("v2_bridge_smoke", 4, 2);
    let brain = Brain::new(&cfg).expect("create tiny brain");

    // ping should succeed and neuron count should be non-zero for TINY.
    assert!(brain.ping(), "fresh brain must ping alive");
    assert!(
        brain.neuron_count() > 0,
        "neuron count must be positive after creation"
    );

    // Drop fires nimcp_brain_destroy — success is "no crash".
    drop(brain);
}

#[cfg(not(feature = "cuda"))]
#[test]
fn stub_build_refuses_to_create_brain() {
    use nimcp_v1_bridge::Brain;

    let cfg = BrainConfig::tiny("stub_smoke", 4, 2);
    let err = Brain::new(&cfg).expect_err("stub build must refuse");
    assert!(matches!(err, BridgeError::CudaUnavailable));
}
