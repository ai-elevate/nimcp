//! NIMCP V2 — spiking neural network.
//!
//! Phase 3 crate. V2 design takeaways from V1's pain:
//!
//! 1. **Quiet-start is a load-time weight transform**, not an init-time branch.
//!    V1's quiet-start protocol only applied to fresh init; resuming from a
//!    saturated checkpoint inherited saturation. V2's transform keys on
//!    observed firing statistics at load time and scales weights regardless
//!    of whether the network is fresh or resumed.
//!
//! 2. **CSR synapses owned per population**. V1 got this right in the
//!    lightweight-SNN refactor; we keep it.
//!
//! 3. **Homeostatic scaling operates on the flat weights array, not strided
//!    entries**. Ports the Phase E perf fix from V1 (commit `3a7aa5f7d`).
//!
//! 4. **Tight bounds [0.98, 1.02] from day one**. The "emergency band"
//!    oscillation bug is in V2's regression tests from the start.

#![forbid(unsafe_code)]
#![allow(dead_code)] // Phase 3 stub

pub mod csr;
pub mod homeostatic;

pub use csr::{CsrSynapses, Population, PopulationId};
pub use homeostatic::{PopulationRateEma, apply_quiet_start_transform, step_homeostatic};

/// Phase 3 placeholder.
pub struct SnnNetwork;
