//! Phase 4b — LNN training: BPTT adjoint + SGD.
//!
//! # The equations (unrolled per timestep)
//!
//! Let `pre_t = W_rec·x_{t-1} + W_in·u_t + b`,
//!     `act_t = tanh(pre_t)`,
//!     `x_t   = x_{t-1} · (1 − dt/τ_safe) + dt · act_t` (per neuron).
//!
//! Gradients accumulate backward in time (last timestep first):
//!
//! ```text
//! dL/dpre_t   = dL/dx_t ⊙ (dt · (1 − act_t²))          // sech²(pre) = 1 − tanh²
//! dL/dW_rec  += outer(dL/dpre_t, x_{t-1})
//! dL/dW_in   += outer(dL/dpre_t, u_t)
//! dL/db      += dL/dpre_t
//! dL/dτ_base += dL/dx_t ⊙ (dt · x_{t-1} / τ_safe²)     // only where τ_base ≥ τ_min
//! dL/dx_{t-1} = dL/dx_t ⊙ (1 − dt/τ_safe)  +  W_rec ᵀ · dL/dpre_t
//! ```
//!
//! Readout gradients (dense `y_t = W_out · h_t + b_out`):
//!
//! ```text
//! dL/dy_t     = y_t − target_t                    // MSE
//! dL/dW_out  += outer(dL/dy_t, h_t)
//! dL/db_out  += dL/dy_t
//! dL/dh_t     = W_outᵀ · dL/dy_t
//! ```
//!
//! `dL/dh_t` enters the top layer's adjoint as the initial `dL/dx_t`.
//!
//! # Stability rules ported from V1
//!
//! - `τ_safe = max(τ_base, LTC_TAU_MIN)` everywhere; τ_base gradient is
//!   zeroed on neurons that are currently at the floor (no "push below
//!   floor" pressure).
//! - Gradient clipping by global L2 norm after the whole backward pass
//!   ([`clip_gradients`]); default norm 1.0 matches V1's setting.
//! - No per-step tau/state clamp on the gradient side — forward clamps
//!   already prevent `x_{t-1}` from reaching the explosion region, so
//!   the backward pass stays finite as long as forward did.

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

use crate::ltc::{LTC_TAU_MIN, LtcLayer, LtcState, ltc_forward_step};
use crate::network::LnnNetwork;

/// Accumulated per-layer parameter gradients.
#[derive(Debug, Clone)]
pub struct LayerGrads {
    /// Recurrent-weight gradient, same shape as `w_rec`.
    pub w_rec: Array2<f32>,
    /// Input-projection gradient, same shape as `w_in`.
    pub w_in: Array2<f32>,
    /// Bias gradient, same shape as `b`.
    pub b: Array1<f32>,
    /// Time-constant gradient, same shape as `tau_base`.
    pub tau_base: Array1<f32>,
}

impl LayerGrads {
    /// All-zero gradients shaped like `layer`.
    #[must_use]
    pub fn zeros_like(layer: &LtcLayer) -> Self {
        Self {
            w_rec: Array2::zeros(layer.w_rec.raw_dim()),
            w_in: Array2::zeros(layer.w_in.raw_dim()),
            b: Array1::zeros(layer.b.len()),
            tau_base: Array1::zeros(layer.tau_base.len()),
        }
    }
}

/// Full-network gradient bundle — per layer + readout.
#[derive(Debug, Clone)]
pub struct LnnGradients {
    /// One `LayerGrads` per LTC layer.
    pub layers: Vec<LayerGrads>,
    /// Readout weight gradient.
    pub w_out: Array2<f32>,
    /// Readout bias gradient.
    pub b_out: Array1<f32>,
}

impl LnnGradients {
    /// All-zero gradient bundle shaped like `net`.
    #[must_use]
    pub fn zeros_like(net: &LnnNetwork) -> Self {
        Self {
            layers: net.layers.iter().map(LayerGrads::zeros_like).collect(),
            w_out: Array2::zeros(net.w_out.raw_dim()),
            b_out: Array1::zeros(net.b_out.len()),
        }
    }

    /// Global L2 norm across every parameter gradient — the scalar
    /// [`clip_gradients`] divides by.
    #[must_use]
    pub fn global_norm(&self) -> f32 {
        let mut sumsq: f64 = 0.0;
        sumsq += self
            .w_out
            .iter()
            .map(|v| (*v as f64) * (*v as f64))
            .sum::<f64>();
        sumsq += self
            .b_out
            .iter()
            .map(|v| (*v as f64) * (*v as f64))
            .sum::<f64>();
        for g in &self.layers {
            sumsq += g
                .w_rec
                .iter()
                .map(|v| (*v as f64) * (*v as f64))
                .sum::<f64>();
            sumsq += g
                .w_in
                .iter()
                .map(|v| (*v as f64) * (*v as f64))
                .sum::<f64>();
            sumsq += g.b.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>();
            sumsq += g
                .tau_base
                .iter()
                .map(|v| (*v as f64) * (*v as f64))
                .sum::<f64>();
        }
        sumsq.sqrt() as f32
    }

    /// Scale every gradient in place by `s`. Used by [`clip_gradients`].
    pub fn scale(&mut self, s: f32) {
        self.w_out *= s;
        self.b_out *= s;
        for g in &mut self.layers {
            g.w_rec *= s;
            g.w_in *= s;
            g.b *= s;
            g.tau_base *= s;
        }
    }
}

/// One unrolled-trajectory record used as input to the backward pass.
///
/// Forward fills this as it walks the sequence; backward reads from the
/// end. Keeping `pre`, `act`, `x_prev`, and `h_last` per layer per
/// timestep lets the adjoint run without recomputing any forward state.
#[derive(Debug, Clone)]
pub struct LnnTrace {
    /// Per-timestep, per-layer: `pre[t][L]` is the layer-L pre-activation
    /// at step t. Shape `(T, n_layers)` conceptually; stored as a `Vec<Vec<..>>`.
    pub pre: Vec<Vec<Array1<f32>>>,
    /// Per-timestep, per-layer: `act[t][L]` = `tanh(pre[t][L])`. Kept
    /// separately so the backward pass avoids recomputing `tanh` only
    /// to then take `(1 − tanh²)`.
    pub act: Vec<Vec<Array1<f32>>>,
    /// Per-timestep, per-layer: `x_prev[t][L]` = layer-L state *before*
    /// step t's integration. Needed for `outer(dL/dpre, x_{t-1})` and
    /// the τ-gradient `x_{t-1} / τ²` term.
    pub x_prev: Vec<Vec<Array1<f32>>>,
    /// Per-timestep: input to the whole network at step t.
    pub u: Vec<Array1<f32>>,
    /// Per-timestep: final-layer state *after* step t (the vector fed
    /// into the readout). `h_last[t]` has length equal to the last
    /// layer's `n_rec`.
    pub h_last: Vec<Array1<f32>>,
    /// Per-timestep readout output `y_t = W_out · h_last[t] + b_out`.
    pub y: Vec<Array1<f32>>,
}

impl LnnNetwork {
    /// Forward pass that also records everything the backward pass needs.
    ///
    /// Returns `(outputs, trace)`. `outputs[t] == trace.y[t]`.
    pub fn forward_sequence_traced(&self, inputs: &[Array1<f32>]) -> (Vec<Array1<f32>>, LnnTrace) {
        let t_max = inputs.len();
        let n_layers = self.layers.len();
        let mut trace = LnnTrace {
            pre: Vec::with_capacity(t_max),
            act: Vec::with_capacity(t_max),
            x_prev: Vec::with_capacity(t_max),
            u: Vec::with_capacity(t_max),
            h_last: Vec::with_capacity(t_max),
            y: Vec::with_capacity(t_max),
        };

        let mut state: Vec<LtcState> = self.new_state();

        for u in inputs {
            let mut pre_t: Vec<Array1<f32>> = Vec::with_capacity(n_layers);
            let mut act_t: Vec<Array1<f32>> = Vec::with_capacity(n_layers);
            let mut x_prev_t: Vec<Array1<f32>> = Vec::with_capacity(n_layers);

            let mut curr_input: Array1<f32> = u.clone();
            for (layer, st) in self.layers.iter().zip(state.iter_mut()) {
                x_prev_t.push(st.x.clone());
                let pre = ltc_forward_step(st, layer, &curr_input, self.dt_ms);
                let act = pre.mapv(f32::tanh);
                pre_t.push(pre);
                act_t.push(act);
                curr_input = st.x.clone();
            }

            let mut y = self.w_out.dot(&curr_input);
            y += &self.b_out;
            trace.pre.push(pre_t);
            trace.act.push(act_t);
            trace.x_prev.push(x_prev_t);
            trace.u.push(u.clone());
            trace.h_last.push(curr_input);
            trace.y.push(y.clone());
        }

        (trace.y.clone(), trace)
    }

    /// Backward pass — computes parameter gradients for a MSE objective
    /// summed over the sequence. Adjoint runs timestep-last → first and
    /// layer-last → first inside each timestep.
    ///
    /// `targets.len()` must equal `trace.y.len()`. Each target must have
    /// length `self.output_dim`. Shape mismatches silently truncate to
    /// the shorter length (a no-op on fully-aligned inputs).
    pub fn backward_mse(&self, trace: &LnnTrace, targets: &[Array1<f32>]) -> LnnGradients {
        let mut grads = LnnGradients::zeros_like(self);
        let t_max = trace.y.len().min(targets.len());
        let n_layers = self.layers.len();
        let dt = self.dt_ms;

        // Pre-compute W_rec transposes once (they don't change
        // mid-backward). Avoids rebuilding them at every t.
        let w_rec_t: Vec<Array2<f32>> =
            self.layers.iter().map(|l| l.w_rec.t().to_owned()).collect();
        let w_out_t = self.w_out.t().to_owned();

        // Per-layer running adjoint — dL/dx at the *end* of the previous
        // (later-in-time) step, held over from step t+1's backward.
        // Initialised to zero at the sequence tail.
        let mut dl_dx_next: Vec<Array1<f32>> = self
            .layers
            .iter()
            .map(|l| Array1::zeros(l.params.n_rec))
            .collect();

        for t in (0..t_max).rev() {
            // --- readout backward: dL/dy = y - target, dL/dh = W_outᵀ · dL/dy
            let dy: Array1<f32> = &trace.y[t] - &targets[t];
            let h = &trace.h_last[t];
            // dL/dW_out += outer(dy, h)
            for (i, &dy_i) in dy.iter().enumerate() {
                for (j, &h_j) in h.iter().enumerate() {
                    grads.w_out[[i, j]] += dy_i * h_j;
                }
            }
            for (gb, &v) in grads.b_out.iter_mut().zip(dy.iter()) {
                *gb += v;
            }
            let dh: Array1<f32> = w_out_t.dot(&dy);

            // --- LTC layers backward, last → first
            // dL/dx at the top layer (for this timestep) = readout adjoint
            // + whatever carried forward from step t+1.
            let mut upstream: Array1<f32> = dh;

            for l_idx in (0..n_layers).rev() {
                let layer = &self.layers[l_idx];
                let x_prev = &trace.x_prev[t][l_idx];
                let act = &trace.act[t][l_idx];

                // Combine this timestep's upstream with the temporal
                // adjoint from step t+1 for this same layer.
                let mut dl_dx: Array1<f32> = &upstream + &dl_dx_next[l_idx];

                // dL/dpre = dL/dx * dt * (1 - act²)
                let mut dl_dpre: Array1<f32> = Array1::zeros(layer.params.n_rec);
                for (((dp, &dx), &a), _) in dl_dpre
                    .iter_mut()
                    .zip(dl_dx.iter())
                    .zip(act.iter())
                    .zip(0..layer.params.n_rec)
                {
                    *dp = dx * dt * (1.0 - a * a);
                }

                // Parameter grads for this layer:
                //   dW_rec += outer(dl_dpre, x_prev)
                //   dW_in  += outer(dl_dpre, u at this timestep for layer)
                //   db     += dl_dpre
                let layer_input: &Array1<f32> = if l_idx == 0 {
                    &trace.u[t]
                } else {
                    // For layers > 0, the input at step t is the
                    // previous layer's *post-step* state. Reconstruct
                    // from trace: x_prev[t][l_idx] was the state at the
                    // *start* of step t; the input came from layer
                    // l_idx-1's state at the *end* of step t, which is
                    // `x_prev[t+1][l_idx-1]` if t+1 < T, else the
                    // *current* state after step t — unknown here, so
                    // the simpler choice is to reuse x_prev[t][l_idx-1]
                    // as a proxy. That's wrong for BPTT correctness —
                    // so we record the actual layer input by recomputing
                    // from `h_last` structure is inconvenient. Instead,
                    // we walk forward once more to capture per-layer
                    // inputs; see the auxiliary helper below.
                    unreachable!(
                        "multi-layer BPTT requires per-layer input trace; use forward_sequence_traced \
                         and inputs_per_layer — see train.rs note."
                    )
                };
                let g = &mut grads.layers[l_idx];
                for (i, &dp_i) in dl_dpre.iter().enumerate() {
                    for (j, &lj) in layer_input.iter().enumerate() {
                        g.w_in[[i, j]] += dp_i * lj;
                    }
                    for (j, &xj) in x_prev.iter().enumerate() {
                        g.w_rec[[i, j]] += dp_i * xj;
                    }
                    g.b[i] += dp_i;
                }

                // τ-gradient:  dL/dτ_base_i  =  dL/dx_t[i] * dt * x_prev[i] / τ_safe²
                // Only where τ_base_i >= τ_min (otherwise tau_safe was
                // floored and dτ_safe/dτ_base = 0).
                for (i, (gt, &tau_b)) in
                    g.tau_base.iter_mut().zip(layer.tau_base.iter()).enumerate()
                {
                    if tau_b >= LTC_TAU_MIN {
                        let tau_safe = tau_b;
                        let contrib = dl_dx[i] * dt * x_prev[i] / (tau_safe * tau_safe);
                        *gt += contrib;
                    }
                }

                // Compute dl_dx_prev for this layer (used as this
                // layer's temporal adjoint at step t-1, AND as the
                // upstream for the layer below at this timestep).
                // dl_dx_prev = dl_dx * (1 - dt/tau_safe) + W_recᵀ · dl_dpre
                let mut dl_dx_prev: Array1<f32> = w_rec_t[l_idx].dot(&dl_dpre);
                for (i, dxp) in dl_dx_prev.iter_mut().enumerate() {
                    let tau_safe = layer.tau_base[i].max(LTC_TAU_MIN);
                    *dxp += dl_dx[i] * (1.0 - dt / tau_safe);
                }

                // Consume dl_dx for this layer at this timestep.
                let _ = &mut dl_dx;

                // Temporal carry for next iteration of the outer `t` loop.
                dl_dx_next[l_idx] = dl_dx_prev.clone();

                // Spatial carry to the layer below (at this same t).
                // For layer 0 this is the gradient w.r.t. the network
                // input — we drop it.
                upstream = dl_dx_prev;
            }
        }

        grads
    }
}

/// Clip gradients so the global L2 norm is at most `max_norm`. Returns
/// the unclipped global norm — useful for logging.
///
/// `max_norm <= 0` is a no-op.
pub fn clip_gradients(grads: &mut LnnGradients, max_norm: f32) -> f32 {
    let norm = grads.global_norm();
    if max_norm > 0.0 && norm > max_norm && norm.is_finite() {
        grads.scale(max_norm / norm);
    }
    norm
}

/// SGD step: `θ <- θ − lr · ∇θ` over every parameter, including
/// `tau_base` which is then re-floored to `LTC_TAU_MIN` so subsequent
/// forward passes don't have to repeat the check.
pub fn sgd_step(net: &mut LnnNetwork, grads: &LnnGradients, lr: f32) {
    net.w_out.zip_mut_with(&grads.w_out, |p, &g| *p -= lr * g);
    net.b_out.zip_mut_with(&grads.b_out, |p, &g| *p -= lr * g);
    for (layer, g) in net.layers.iter_mut().zip(grads.layers.iter()) {
        layer.w_rec.zip_mut_with(&g.w_rec, |p, &gv| *p -= lr * gv);
        layer.w_in.zip_mut_with(&g.w_in, |p, &gv| *p -= lr * gv);
        layer.b.zip_mut_with(&g.b, |p, &gv| *p -= lr * gv);
        layer
            .tau_base
            .zip_mut_with(&g.tau_base, |p, &gv| *p = (*p - lr * gv).max(LTC_TAU_MIN));
    }
}

/// Training hyperparameters.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TrainParams {
    /// Learning rate.
    pub lr: f32,
    /// Gradient clip L2 norm (`<= 0` → no clip). V1 default: 1.0.
    pub grad_clip: f32,
}

impl Default for TrainParams {
    fn default() -> Self {
        Self {
            lr: 1.0e-2,
            grad_clip: 1.0,
        }
    }
}

/// One full train step on a single sequence: forward (traced) + backward
/// (MSE) + optional gradient clip + SGD. Returns `(loss, grad_norm)`.
///
/// `loss` is `0.5 · Σₜ ||y_t − target_t||²` summed across all timesteps.
pub fn train_step_mse(
    net: &mut LnnNetwork,
    inputs: &[Array1<f32>],
    targets: &[Array1<f32>],
    params: &TrainParams,
) -> (f32, f32) {
    assert_eq!(inputs.len(), targets.len(), "sequence length mismatch");

    // Multi-layer BPTT needs the per-layer input at every step, and the
    // single-layer path has a cheap shortcut. Pick the correct one.
    if net.layers.len() == 1 {
        train_step_mse_single(net, inputs, targets, params)
    } else {
        train_step_mse_multi(net, inputs, targets, params)
    }
}

fn train_step_mse_single(
    net: &mut LnnNetwork,
    inputs: &[Array1<f32>],
    targets: &[Array1<f32>],
    params: &TrainParams,
) -> (f32, f32) {
    let (outputs, trace) = net.forward_sequence_traced(inputs);
    let loss = mse_sequence_loss(&outputs, targets);
    let mut grads = net.backward_mse(&trace, targets);
    let norm = clip_gradients(&mut grads, params.grad_clip);
    sgd_step(net, &grads, params.lr);
    (loss, norm)
}

/// Multi-layer BPTT. Carries per-layer inputs alongside the trace so the
/// backward pass can form `outer(dl_dpre, layer_input)` correctly for
/// layers beyond the first.
fn train_step_mse_multi(
    net: &mut LnnNetwork,
    inputs: &[Array1<f32>],
    targets: &[Array1<f32>],
    params: &TrainParams,
) -> (f32, f32) {
    // Trace-with-layer-inputs: identical to `forward_sequence_traced` plus
    // an extra `Vec<Vec<Array1<f32>>>` storing the *input* to each layer
    // at each step.
    let t_max = inputs.len();
    let n_layers = net.layers.len();
    let mut trace = LnnTrace {
        pre: Vec::with_capacity(t_max),
        act: Vec::with_capacity(t_max),
        x_prev: Vec::with_capacity(t_max),
        u: Vec::with_capacity(t_max),
        h_last: Vec::with_capacity(t_max),
        y: Vec::with_capacity(t_max),
    };
    // Per timestep, per layer, the vector fed into that layer.
    let mut layer_inputs: Vec<Vec<Array1<f32>>> = Vec::with_capacity(t_max);
    let mut state: Vec<LtcState> = net.new_state();

    for u in inputs {
        let mut pre_t = Vec::with_capacity(n_layers);
        let mut act_t = Vec::with_capacity(n_layers);
        let mut x_prev_t = Vec::with_capacity(n_layers);
        let mut li_t = Vec::with_capacity(n_layers);

        let mut curr_input = u.clone();
        for (layer, st) in net.layers.iter().zip(state.iter_mut()) {
            x_prev_t.push(st.x.clone());
            li_t.push(curr_input.clone());
            let pre = ltc_forward_step(st, layer, &curr_input, net.dt_ms);
            let act = pre.mapv(f32::tanh);
            pre_t.push(pre);
            act_t.push(act);
            curr_input = st.x.clone();
        }

        let mut y = net.w_out.dot(&curr_input);
        y += &net.b_out;
        trace.pre.push(pre_t);
        trace.act.push(act_t);
        trace.x_prev.push(x_prev_t);
        trace.u.push(u.clone());
        trace.h_last.push(curr_input);
        trace.y.push(y);
        layer_inputs.push(li_t);
    }

    // Sequence MSE.
    let loss = mse_sequence_loss(&trace.y, targets);

    // Backward pass — same math as `backward_mse` but uses the
    // per-layer inputs we just captured.
    let mut grads = LnnGradients::zeros_like(net);
    let dt = net.dt_ms;
    let w_out_t = net.w_out.t().to_owned();
    let w_rec_t: Vec<Array2<f32>> = net.layers.iter().map(|l| l.w_rec.t().to_owned()).collect();
    let mut dl_dx_next: Vec<Array1<f32>> = net
        .layers
        .iter()
        .map(|l| Array1::zeros(l.params.n_rec))
        .collect();

    let t_steps = trace.y.len().min(targets.len());
    for t in (0..t_steps).rev() {
        let dy: Array1<f32> = &trace.y[t] - &targets[t];
        let h = &trace.h_last[t];
        for (i, &dy_i) in dy.iter().enumerate() {
            for (j, &h_j) in h.iter().enumerate() {
                grads.w_out[[i, j]] += dy_i * h_j;
            }
        }
        for (gb, &v) in grads.b_out.iter_mut().zip(dy.iter()) {
            *gb += v;
        }
        let dh: Array1<f32> = w_out_t.dot(&dy);

        let mut upstream: Array1<f32> = dh;
        for l_idx in (0..n_layers).rev() {
            let layer = &net.layers[l_idx];
            let x_prev = &trace.x_prev[t][l_idx];
            let act = &trace.act[t][l_idx];
            let layer_input = &layer_inputs[t][l_idx];

            let dl_dx: Array1<f32> = &upstream + &dl_dx_next[l_idx];
            let mut dl_dpre: Array1<f32> = Array1::zeros(layer.params.n_rec);
            for (dp, (&dx, &a)) in dl_dpre.iter_mut().zip(dl_dx.iter().zip(act.iter())) {
                *dp = dx * dt * (1.0 - a * a);
            }

            let g = &mut grads.layers[l_idx];
            for (i, &dp_i) in dl_dpre.iter().enumerate() {
                for (j, &lj) in layer_input.iter().enumerate() {
                    g.w_in[[i, j]] += dp_i * lj;
                }
                for (j, &xj) in x_prev.iter().enumerate() {
                    g.w_rec[[i, j]] += dp_i * xj;
                }
                g.b[i] += dp_i;
            }
            for (i, (gt, &tau_b)) in g.tau_base.iter_mut().zip(layer.tau_base.iter()).enumerate() {
                if tau_b >= LTC_TAU_MIN {
                    *gt += dl_dx[i] * dt * x_prev[i] / (tau_b * tau_b);
                }
            }

            let mut dl_dx_prev: Array1<f32> = w_rec_t[l_idx].dot(&dl_dpre);
            for (i, dxp) in dl_dx_prev.iter_mut().enumerate() {
                let tau_safe = layer.tau_base[i].max(LTC_TAU_MIN);
                *dxp += dl_dx[i] * (1.0 - dt / tau_safe);
            }

            // Upstream for the layer below at this timestep.
            // For layer 0, the gradient is w.r.t. the network input —
            // discard. For layers > 0, propagate to the previous layer's
            // x at this same step (which is the current layer's input).
            upstream = if l_idx == 0 {
                Array1::zeros(net.layers[0].params.n_in)
            } else {
                // dL/d(layer_input) = W_inᵀ · dl_dpre
                layer.w_in.t().dot(&dl_dpre)
            };

            dl_dx_next[l_idx] = dl_dx_prev;
        }
    }

    let norm = clip_gradients(&mut grads, params.grad_clip);
    sgd_step(net, &grads, params.lr);
    (loss, norm)
}

/// `0.5 · Σₜ ||y_t − target_t||²` — the sequence-MSE loss used throughout.
#[must_use]
pub fn mse_sequence_loss(outputs: &[Array1<f32>], targets: &[Array1<f32>]) -> f32 {
    let mut loss = 0.0_f32;
    let n = outputs.len().min(targets.len());
    for (y, t) in outputs.iter().take(n).zip(targets.iter().take(n)) {
        let diff = y - t;
        loss += 0.5 * diff.iter().map(|v| v * v).sum::<f32>();
    }
    loss
}

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use crate::ltc::LtcParams;
    use crate::network::LnnConfig;

    fn single_layer_cfg(input_dim: usize, hidden: usize, output_dim: usize) -> LnnConfig {
        LnnConfig {
            input_dim,
            output_dim,
            layers: vec![LtcParams {
                n_in: input_dim,
                n_rec: hidden,
                tau_init: 1.0,
                init_scale: 1.0,
            }],
            rng_seed: 0x9999,
            dt_ms: 0.1,
            substrate: crate::network::LnnSubstrateCfg::default(),
            thalamic: None,
        }
    }

    fn two_layer_cfg(input_dim: usize, hidden: usize, output_dim: usize) -> LnnConfig {
        LnnConfig {
            input_dim,
            output_dim,
            layers: vec![
                LtcParams {
                    n_in: input_dim,
                    n_rec: hidden,
                    tau_init: 1.0,
                    init_scale: 1.0,
                },
                LtcParams {
                    n_in: hidden,
                    n_rec: hidden,
                    tau_init: 1.0,
                    init_scale: 1.0,
                },
            ],
            rng_seed: 0x1357,
            dt_ms: 0.1,
            substrate: crate::network::LnnSubstrateCfg::default(),
            thalamic: None,
        }
    }

    #[test]
    fn zero_target_zero_output_gives_zero_loss() {
        let outputs: Vec<Array1<f32>> = (0..5).map(|_| Array1::zeros(3)).collect();
        let targets: Vec<Array1<f32>> = (0..5).map(|_| Array1::zeros(3)).collect();
        assert_eq!(mse_sequence_loss(&outputs, &targets), 0.0);
    }

    #[test]
    fn mse_loss_exact_on_known_pair() {
        // [1,2,3] vs [4,5,6] → diff [-3,-3,-3] → 0.5*27 = 13.5
        let outputs = vec![Array1::from_vec(vec![1.0, 2.0, 3.0])];
        let targets = vec![Array1::from_vec(vec![4.0, 5.0, 6.0])];
        assert!((mse_sequence_loss(&outputs, &targets) - 13.5).abs() < 1e-6);
    }

    #[test]
    fn gradients_are_zero_on_zero_loss() {
        let net = LnnNetwork::new(single_layer_cfg(2, 4, 1)).expect("build");
        let inputs: Vec<Array1<f32>> = (0..3).map(|_| Array1::zeros(2)).collect();
        let (y, trace) = net.forward_sequence_traced(&inputs);
        // Use y as the target so dL/dy == 0 exactly.
        let grads = net.backward_mse(&trace, &y);
        assert!((grads.global_norm()).abs() < 1e-6);
    }

    #[test]
    fn grad_clip_respects_max_norm() {
        let net = LnnNetwork::new(single_layer_cfg(2, 4, 1)).expect("build");
        let mut grads = LnnGradients::zeros_like(&net);
        // Set one weight to a huge value so global norm is >> 1.
        grads.w_out[[0, 0]] = 100.0;
        let unclipped = clip_gradients(&mut grads, 1.0);
        assert!(unclipped > 99.0);
        assert!(grads.global_norm() <= 1.0 + 1e-5);
    }

    #[test]
    fn grad_clip_noop_below_threshold() {
        let net = LnnNetwork::new(single_layer_cfg(2, 4, 1)).expect("build");
        let mut grads = LnnGradients::zeros_like(&net);
        grads.w_out[[0, 0]] = 0.3;
        let before = grads.w_out[[0, 0]];
        let _ = clip_gradients(&mut grads, 1.0);
        assert_eq!(grads.w_out[[0, 0]], before, "sub-threshold grad was scaled");
    }

    #[test]
    fn sgd_moves_params_in_negative_gradient_direction() {
        let mut net = LnnNetwork::new(single_layer_cfg(2, 4, 1)).expect("build");
        let mut grads = LnnGradients::zeros_like(&net);
        grads.w_out[[0, 0]] = 0.5;
        let before = net.w_out[[0, 0]];
        sgd_step(&mut net, &grads, 0.1);
        assert_eq!(net.w_out[[0, 0]], before - 0.05);
    }

    #[test]
    fn sgd_respects_tau_floor() {
        let mut net = LnnNetwork::new(single_layer_cfg(2, 4, 1)).expect("build");
        // Force tau_base to exactly the floor, then apply a positive
        // gradient so lr*g subtracts from it — the floor must hold.
        for t in net.layers[0].tau_base.iter_mut() {
            *t = LTC_TAU_MIN;
        }
        let mut grads = LnnGradients::zeros_like(&net);
        for g in grads.layers[0].tau_base.iter_mut() {
            *g = 1.0;
        }
        sgd_step(&mut net, &grads, 1.0);
        assert!(net.layers[0].tau_base.iter().all(|&v| v >= LTC_TAU_MIN));
    }

    /// Single-layer sanity: one train step on a trivial constant-target
    /// problem should reduce loss. Seeds chosen so the initial direction
    /// isn't degenerate.
    #[test]
    fn train_step_reduces_loss_single_layer() {
        let mut net = LnnNetwork::new(single_layer_cfg(2, 8, 1)).expect("build");
        let inputs: Vec<Array1<f32>> = (0..10)
            .map(|t| Array1::from_vec(vec![(t as f32 * 0.3).sin(), (t as f32 * 0.2).cos()]))
            .collect();
        let targets: Vec<Array1<f32>> = (0..10).map(|_| Array1::from_vec(vec![0.5])).collect();

        let (loss_before, trace) = {
            let (y, tr) = net.forward_sequence_traced(&inputs);
            (mse_sequence_loss(&y, &targets), tr)
        };
        let _ = trace;

        let params = TrainParams {
            lr: 1.0e-2,
            grad_clip: 1.0,
        };
        for _ in 0..20 {
            train_step_mse(&mut net, &inputs, &targets, &params);
        }
        let (y, _) = net.forward_sequence_traced(&inputs);
        let loss_after = mse_sequence_loss(&y, &targets);
        assert!(
            loss_after < loss_before,
            "loss did not decrease: before {loss_before}, after {loss_after}"
        );
    }

    /// Two-layer sanity test — exercises the multi-layer BPTT path.
    #[test]
    fn train_step_reduces_loss_two_layer() {
        let mut net = LnnNetwork::new(two_layer_cfg(2, 8, 1)).expect("build");
        let inputs: Vec<Array1<f32>> = (0..10)
            .map(|t| Array1::from_vec(vec![(t as f32 * 0.3).sin(), (t as f32 * 0.2).cos()]))
            .collect();
        let targets: Vec<Array1<f32>> = (0..10).map(|_| Array1::from_vec(vec![0.5])).collect();

        let (y, _) = net.forward_sequence_traced(&inputs);
        let loss_before = mse_sequence_loss(&y, &targets);

        let params = TrainParams {
            lr: 1.0e-2,
            grad_clip: 1.0,
        };
        for _ in 0..30 {
            train_step_mse(&mut net, &inputs, &targets, &params);
        }
        let (y, _) = net.forward_sequence_traced(&inputs);
        let loss_after = mse_sequence_loss(&y, &targets);
        assert!(
            loss_after < loss_before,
            "loss did not decrease: before {loss_before}, after {loss_after}"
        );
    }

    /// Convergence test: learn a short constant-delay task. The network
    /// must output the input at t=0 at every subsequent step. Trained
    /// over many epochs, the single-step MSE should drop meaningfully.
    #[test]
    fn converges_on_constant_target_regression() {
        let mut net = LnnNetwork::new(single_layer_cfg(1, 12, 1)).expect("build");
        let inputs: Vec<Array1<f32>> = (0..20).map(|_| Array1::from_vec(vec![0.7])).collect();
        let targets: Vec<Array1<f32>> = (0..20).map(|_| Array1::from_vec(vec![0.7])).collect();

        let (y, _) = net.forward_sequence_traced(&inputs);
        let loss_before = mse_sequence_loss(&y, &targets);

        let params = TrainParams {
            lr: 2.0e-2,
            grad_clip: 1.0,
        };
        for _ in 0..200 {
            train_step_mse(&mut net, &inputs, &targets, &params);
        }
        let (y, _) = net.forward_sequence_traced(&inputs);
        let loss_after = mse_sequence_loss(&y, &targets);

        // Demand at least 50% reduction — crude but unambiguous.
        assert!(
            loss_after < 0.5 * loss_before,
            "loss reduction insufficient: before {loss_before}, after {loss_after}"
        );
    }

    #[test]
    fn grad_clip_zero_or_negative_is_noop() {
        let net = LnnNetwork::new(single_layer_cfg(2, 4, 1)).expect("build");
        let mut grads = LnnGradients::zeros_like(&net);
        grads.w_out[[0, 0]] = 50.0;
        let norm_before = grads.global_norm();
        let returned = clip_gradients(&mut grads, 0.0);
        assert_eq!(returned, norm_before);
        // Nothing should have been scaled down.
        assert_eq!(grads.w_out[[0, 0]], 50.0);
    }
}
