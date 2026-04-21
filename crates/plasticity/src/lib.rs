//! NIMCP V2 — plasticity rules.
//!
//! Each rule is a pure function of (pre-activity, post-activity, weights,
//! neuromodulators) → weight deltas. The equations are well-understood;
//! the V1 bugs were mostly in how they were wired, not in the math.
//!
//! # Rules (ported from V1, each in its own module for unit testability)
//!
//! - STDP: spike-timing-dependent plasticity
//! - R-STDP: reward-modulated STDP (with warmup gate)
//! - Homeostatic: synaptic scaling with tight [0.98, 1.02] bounds
//! - BCM: Bienenstock-Cooper-Munro (rate-based)
//! - Eligibility traces
//!
//! # Regression tests shipped from day one
//!
//! - Homeostatic stability: the "emergency band oscillation" bug is a test.
//! - Quiet-start-on-resume: saturated weights load → recovery within N steps.

#![forbid(unsafe_code)]
#![allow(dead_code)]

/// Phase 3 placeholder.
pub struct Plasticity;
