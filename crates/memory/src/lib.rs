//! NIMCP V2 — Z-Ladder memory system.
//!
//! Phase 5 crate. Port of V1's Phase E work — the design was sound, so we
//! translate rather than redesign. Key changes for V2:
//!
//! - Actor, not a set of callbacks. Single-threaded, no races.
//! - Landmark check + demotion is a single `apply()` on a tier event.
//! - Checkpoint includes full feature payload by default (V1's E6 lesson).
//! - Full-tier query and landmark query are both supported from day one.

#![forbid(unsafe_code)]
#![allow(dead_code)]

/// Four-tier memory ladder: Z0 working → Z1 short → Z2 long → Z3 permanent.
pub struct ZLadder;
