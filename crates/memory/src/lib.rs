//! NIMCP V2 — Z-Ladder memory system.
//!
//! # Phase 5 scope
//!
//! Port of V1's Phase E Z-Ladder — a four-tier consolidating memory
//! system (`Z0 → Z1 → Z2 → Z3`). V1's design was sound, so this is a
//! translation to Rust rather than a redesign, keeping the biological
//! analogy (working / short-term / long-term / permanent) and the
//! promote / demote / decay / consolidate flow.
//!
//! V2-specific changes:
//!
//! 1. **No hand-rolled hash tables or mutexes.** One `HashMap` per
//!    tier; the whole ladder is `Send` so it lives inside an actor.
//! 2. **Checkpoint preserves full feature payload by default.** V1's
//!    E6 lesson was that landmark save/load must round-trip every
//!    feature vector; V2 bakes that in — [`ZLadder`]'s whole state,
//!    including every node's `features: Vec<f32>`, serializes via
//!    serde.
//! 3. **Similarity queries use cosine distance on the feature vector.**
//!    V1 used a "prime signature" — equivalent end-to-end but simpler
//!    to port.
//! 4. **Landmarks are protected from demotion.** A landmark-flagged
//!    node stays at `Z3` until explicitly unmarked; decay still
//!    applies to strength (for introspection) but the demotion gate
//!    always returns `false` for landmarks.
//!
//! # Module layout
//!
//! - [`node`] — [`MemoryNode`], [`Tier`], their serde impls.
//! - [`config`] — [`ZLadderConfig`], [`TierConfig`], defaults.
//! - [`ladder`] — [`ZLadder`] itself, basic ops + consolidation +
//!   landmark API.
//! - [`query`] — cosine-similarity retrieval.

#![forbid(unsafe_code)]

pub mod config;
pub mod ladder;
pub mod node;
pub mod query;

pub use config::{TierConfig, ZLadderConfig, ZLadderError};
pub use ladder::{ZLadder, ZLadderStats};
pub use node::{MemoryNode, Tier};
pub use query::QueryHit;
