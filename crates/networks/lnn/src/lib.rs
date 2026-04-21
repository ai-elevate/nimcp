//! NIMCP V2 — Liquid Time-Constant (LTC) neural network.
//!
//! # Phase 4a scope
//!
//! This crate ships the forward-pass core of an LTC-style recurrent
//! network — per-neuron trainable time constants, Euler integration,
//! deterministic seeded init, serde checkpoint. Backward pass through
//! time + SGD land in Phase 4b (see [`V2_PLAN.md`]).
//!
//! # Math
//!
//! For neuron `i` with recurrent state `x_i`:
//!
//! ```text
//! pre_i = W_rec_i · x  +  W_in_i · u  +  b_i
//! dx_i/dt = -x_i / tau_safe_i  +  tanh(pre_i)
//! x_i(t+dt) = x_i(t) + dt · dx_i/dt
//! ```
//!
//! `tau_safe_i = max(tau_base_i, TAU_MIN)` keeps the `1/τ` term stable
//! — this is the V1 "1/τ² explosion" lesson encoded as an invariant,
//! not a runtime check.
//!
//! Per-step state is clamped to `[-STATE_CLAMP, +STATE_CLAMP]` after
//! every integration; without this, single-precision NaN creeps in on
//! long unrolls. Again — V1's hard-won rule.
//!
//! # Module layout
//!
//! - [`ltc`] — LTC neuron math (`LtcParams`, `LtcState`, `LtcLayer`,
//!   `ltc_forward_step`).
//! - [`network`] — top-level [`LnnNetwork`]: stacks layers, handles
//!   input projection + readout, exposes `forward` / `reset`.
//!
//! # V1 rules carried forward
//!
//! - **`tau_safe` floor of 0.01** — not tunable. Lower values put
//!   `1/τ²` into territory that produces NaN gradients on float32.
//! - **State clamping `[−1e4, +1e4]` per step** — prevents blow-up on
//!   long sequences; documented as part of the forward contract so
//!   backward-pass authors (Phase 4b) don't have to rediscover it.
//! - **`tanh` nonlinearity by default** — matches V1's LNN layer. ReLU
//!   is not a drop-in substitute (breaks symmetry around rest).

#![forbid(unsafe_code)]

pub mod ltc;
pub mod network;

pub use ltc::{LTC_STATE_CLAMP, LTC_TAU_MIN, LtcLayer, LtcParams, LtcState, ltc_forward_step};
pub use network::{LnnConfig, LnnNetwork};
