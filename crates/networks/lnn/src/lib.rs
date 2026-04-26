//! NIMCP V2 — Liquid Time-Constant (LTC) neural network.
//!
//! # Phase 4a + 4b scope
//!
//! Phase 4a shipped the forward-pass core: per-neuron trainable time
//! constants, Euler integration, deterministic seeded init, serde
//! checkpoint. Phase 4b (this revision) adds backpropagation through
//! time via the adjoint method, MSE sequence loss, global gradient
//! clipping, and an SGD step that respects the `τ_min` floor.
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
//!   input projection + readout, exposes `forward_step` /
//!   `forward_sequence` / `new_state`.
//! - [`train`] — BPTT adjoint, MSE loss, gradient clipping, SGD step,
//!   and the `train_step_mse` convenience that bundles all three.
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

// CPU build forbids unsafe. With `--features cuda`, the LTC GPU kernel
// launch path requires one `unsafe` block per call (cudarc's
// `.launch(cfg)` is `unsafe fn`). Scoped to the cuda-feature `gpu`
// submodule of `ltc.rs`.
#![cfg_attr(not(feature = "cuda"), forbid(unsafe_code))]
#![cfg_attr(feature = "cuda", deny(unsafe_code))]
#![cfg_attr(feature = "cuda", allow(unsafe_code))]

pub mod ltc;
pub mod network;
pub mod substrate_adapter;
pub mod train;

pub use ltc::{LTC_STATE_CLAMP, LTC_TAU_MIN, LtcLayer, LtcParams, LtcState, ltc_forward_step};
pub use network::{LnnConfig, LnnNetwork, LnnSubstrateCfg, LnnThalamicCfg};
pub use substrate_adapter::{
    attention_scale_input, burst_tau_clamp_ms, effective_dt_ms, effective_lr, effective_tau,
    ltc_forward_step_with_tau_override, ltp_ltd_gates,
};
pub use train::{
    LnnGradients, LnnTrace, TrainParams, clip_gradients, mse_sequence_loss, sgd_step,
    train_step_mse,
};
