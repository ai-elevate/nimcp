//! NIMCP V2 — biological substrate.
//!
//! Port of V1's `neural_substrate_t` + `axon_dendrite_substrate_bridge`
//! infrastructure, reduced to the minimum surface needed by the
//! multi-network substrate integration plan (see
//! `docs/design/substrate_thalamic_integration.md`).
//!
//! # What lives here
//!
//! - [`NeuralSubstrate`] — chemistry + physical + modulation state for
//!   one compartment (typically one brain region).
//! - [`AxonSubstrateEffects`] — 14 multipliers the axon side of any
//!   network reads each step.
//! - [`DendriteSubstrateEffects`] — 15 multipliers the dendrite side
//!   reads each step.
//! - [`compute_effects`] — pure function producing both effect structs
//!   from a [`NeuralSubstrate`]. No allocation, no side effects.
//! - [`debit_activity`] — bidirectional feedback: each network tick
//!   decrements ATP by a per-spike / per-plasticity cost.
//!
//! # Design invariants
//!
//! 1. All multipliers are `1.0` at full health (atp=1, o2=1, membrane=1,
//!    temperature=37 °C, ion_balance=1). Networks see identity → the
//!    substrate layer is a no-op on a healthy brain.
//! 2. Multipliers are **clamped** to documented ranges. No adapter ever
//!    has to defend against NaN or extreme values.
//! 3. Every effect field is monotone in the underlying chemistry — if
//!    ATP drops, ATP-gated fields never go UP. Makes adapter testing
//!    straightforward.
//! 4. Serde-serializable so checkpoints round-trip.

#![forbid(unsafe_code)]

pub mod compute;
pub mod effects;
pub mod state;

pub use compute::{compute_effects, debit_activity};
pub use effects::{AxonSubstrateEffects, DendriteSubstrateEffects};
pub use state::{NeuralSubstrate, NeuralSubstrateConfig};
