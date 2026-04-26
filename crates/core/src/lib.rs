//! NIMCP V2 — core traits and types.
//!
//! This crate defines the fundamental abstractions that everything else
//! in the workspace builds on:
//!
//! - [`Event`] — the unit of state mutation. Every change in V2 is an event
//!   appended to an event log; state is a materialized view over that log.
//! - [`Actor`] — a unit of isolated state that processes typed messages.
//!   Actors own their state; they do not share memory with peers.
//! - [`Kernel`] — a pure compute function. Forward + backward. Neither
//!   stateful nor async; the scheduler and networks wrap these into actors.
//! - [`Id`] / [`ActorId`] / [`EventId`] — newtype wrappers for IDs so the
//!   compiler catches mis-typed IDs.
//!
//! Keep this crate **tiny**. Anything that could live elsewhere should.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub mod alloc_stats;
pub mod ids;

pub use alloc_stats::{
    AllocSnapshot, AllocTag, ProcStatusBytes, TaggedBytes, account, release, snapshot,
};
pub use ids::{ActorId, EventId, Id};

/// Result alias used across the workspace.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Top-level error type. Domain crates can wrap this in their own errors.
#[derive(Debug, Error)]
pub enum Error {
    /// An internal invariant was violated. Indicates a bug; caller cannot recover.
    #[error("invariant violation: {0}")]
    Invariant(String),

    /// An actor mailbox refused a message (closed or full).
    #[error("mailbox: {0}")]
    Mailbox(String),

    /// A state serialization or deserialization failed.
    #[error("serialization: {0}")]
    Serialization(String),

    /// I/O failure (event log, checkpoint, etc.).
    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    /// Config was rejected at validation time.
    #[error("config: {0}")]
    Config(String),
}

/// The universal state-mutation primitive.
///
/// Every change in V2 — a weight update, a synapse addition, a firing event —
/// is modeled as an `Event` appended to the event log. State is the result of
/// replaying the event log through each event's `apply` method.
///
/// Events MUST be:
/// - Pure (no side effects beyond `apply`'s mutation of `State`)
/// - Deterministic (same event + same state → same result, always)
/// - Serializable via rkyv (enables zero-copy replay)
pub trait Event: Serialize + for<'a> Deserialize<'a> + Send + Sync + 'static {
    /// The state type this event mutates.
    type State: Send + 'static;

    /// Apply this event to the given state. Must be pure + deterministic.
    fn apply(self, state: &mut Self::State);
}

/// A compute kernel: pure forward + backward, no state, no async.
///
/// Kernels are the innermost hot path. The scheduler wraps them in actors;
/// actors are responsible for dispatch + event emission. Keep kernels simple.
pub trait Kernel: Send + Sync {
    /// Input tensor type (e.g., `ndarray::ArrayD<f32>`).
    type Input: Send + Sync;
    /// Weights tensor type.
    type Weights: Send + Sync;
    /// Output tensor type.
    type Output: Send + Sync;

    /// Forward pass. Must be pure: same (input, weights) → same output.
    fn forward(&self, input: &Self::Input, weights: &Self::Weights) -> Self::Output;

    /// Backward pass. Returns `(input_grad, weights_grad)`.
    fn backward(
        &self,
        input: &Self::Input,
        output_grad: &Self::Output,
        weights: &Self::Weights,
    ) -> (Self::Input, Self::Weights);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Serialize, Deserialize)]
    struct Counter(u64);

    #[derive(Serialize, Deserialize)]
    struct Inc(u64);

    impl Event for Inc {
        type State = Counter;
        fn apply(self, state: &mut Self::State) {
            state.0 += self.0;
        }
    }

    #[test]
    fn event_apply_mutates_state() {
        let mut state = Counter(0);
        Inc(5).apply(&mut state);
        Inc(3).apply(&mut state);
        assert_eq!(state.0, 8);
    }
}
