//! Phase 3b — CSR synapse storage.
//!
//! **Placeholder stub owned by the 3b agent.** This file intentionally
//! carries only the fields referenced by sibling Phase 3 modules so they
//! can compile and unit-test against a stable shape. The real CSR storage
//! (`row_ptr`, `col_idx`, forward kernel, etc.) is the 3b agent's work;
//! the integrator step reconciles any field additions without touching
//! the semantics exposed here.
//!
//! # Invariants
//!
//! - `weights.len()` equals the number of synapses in the population's
//!   incoming CSR — Phase 3e's homeostatic scaling multiplies every
//!   entry in-place.
#![allow(dead_code)]

/// Stable population handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PopulationId(pub u32);

/// Placeholder — full population definition lives with the 3b agent.
#[derive(Debug, Default)]
pub struct Population;

/// CSR-format synapse storage for a single destination population.
///
/// The only field Phase 3e (homeostatic / quiet-start) relies on is
/// [`CsrSynapses::weights`]: a flat, row-major array of synaptic
/// efficacies that the scaling transforms multiply in place.
#[derive(Debug, Default, Clone)]
pub struct CsrSynapses {
    /// Flat row-major weights, one per synapse. Scaled in place by
    /// homeostatic and quiet-start.
    pub weights: Vec<f32>,
}

impl CsrSynapses {
    /// Build a CSR with the given weight vector. Test helper for sibling
    /// modules until the full 3b implementation lands.
    #[must_use]
    pub fn from_weights(weights: Vec<f32>) -> Self {
        Self { weights }
    }
}
