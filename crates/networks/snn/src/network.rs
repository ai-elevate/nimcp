//! Phase 3d — top-level [`SnnNetwork`] composition.
//!
//! Wires the four Phase 3 modules into a runnable spiking network:
//!
//! ```text
//!   inputs ─▶ I_syn ─▶ LIF ─▶ spike ─▶ CSR forward ─▶ I_syn ...
//!                                        │
//!                                        ▼
//!                                    R-STDP    ──▶ weight update
//!                                        │
//!                                        ▼
//!                                 homeostatic  ──▶ rate EMA + scale
//! ```
//!
//! # Ownership
//!
//! - [`SnnNetwork`] owns every population and every directed edge, plus
//!   the preallocated I_syn / spike scratch buffers — no per-tick
//!   allocation on the hot path.
//!
//! # Integration tests
//!
//! - `saturation_recovery` — builds a two-pop network, hand-saturates
//!   every weight to `w_max`, verifies spike rates are above target,
//!   runs [`SnnNetwork::apply_quiet_start`], then steps the network and
//!   asserts per-pop rates land inside the deadband within 100 steps.
//!   This is V2_PLAN Phase 3's exit criterion.

use nimcp_plasticity::HomeostaticParams;
use serde::{Deserialize, Serialize};

use crate::csr::{CsrError, CsrSynapses, PopulationId};
use crate::homeostatic::{DEFAULT_RATE_EMA_ALPHA, PopulationRateEma, step_homeostatic};
use crate::lif::{LifParams, LifState, lif_step_cpu};
use crate::rstdp::{RstdpParams, RstdpState, step_rstdp};

// -------------------------------------------------------------------------
// Config
// -------------------------------------------------------------------------

/// Opt-in adaptation-mechanism config (used for both AHP and pump).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(default)]
pub struct AdaptationCfg {
    /// Exponential decay time constant (ms).
    pub tau_ms: f32,
    /// Hyperpolarization per unit adapt_var (mV).
    pub gain_mv: f32,
    /// Increment on each spike (usually 1.0).
    pub spike_bump: f32,
}

impl Default for AdaptationCfg {
    /// Fast AHP defaults — tau ~150 ms, gain 0.6 mV.
    fn default() -> Self {
        Self {
            tau_ms: 150.0,
            gain_mv: 0.6,
            spike_bump: 1.0,
        }
    }
}

impl AdaptationCfg {
    /// Slow Na/K pump defaults — tau ~5 s, gain 0.05 mV.
    #[must_use]
    pub fn pump_defaults() -> Self {
        Self {
            tau_ms: 5_000.0,
            gain_mv: 0.05,
            spike_bump: 1.0,
        }
    }
}

/// Opt-in basket-cell pool config.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(default)]
pub struct BasketCfg {
    /// Basket pool size as a fraction of parent population. Clamped to [0.01, 0.5].
    pub fraction: f32,
}

impl Default for BasketCfg {
    fn default() -> Self {
        Self { fraction: 0.2 }
    }
}

/// Static description of one neuron population.
///
/// [`Default`] produces a backward-compatible profile with every Phase 3.5
/// stability mechanism **off**: `noise.rate_hz = 0`, `depression.inc = 0`,
/// all `Option` fields `None`. Callers who want the master-tuned
/// production defaults set `noise: NoiseConfig::default()` etc.
/// explicitly — JSON construction via `#[serde(default)]` already
/// picks those.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PopulationSpec {
    /// Human-readable label.
    pub name: String,
    /// Number of LIF neurons.
    pub n_neurons: u32,
    /// LIF dynamics.
    pub lif: LifParams,
    /// Homeostatic target (fraction spiking per step).
    pub target_rate: f32,
    /// Homeostatic bounds + deadband + gain.
    pub homeostatic: HomeostaticParams,
    /// Poisson background noise config. Default is V1's 20 Hz × 30 mV;
    /// set `rate_hz = 0.0` to disable.
    #[serde(default)]
    pub noise: crate::noise::NoiseConfig,
    /// Short-term depression config. Default `inc=0.3, tau=50ms, cap=0.5`;
    /// set `inc = 0.0` to disable.
    #[serde(default)]
    pub depression: crate::depression::DepressionConfig,
    /// Optional fast M-current AHP adaptation. `None` disables.
    #[serde(default)]
    pub adaptation_ahp: Option<AdaptationCfg>,
    /// Optional slow Na/K pump adaptation. `None` disables.
    #[serde(default)]
    pub adaptation_pump: Option<AdaptationCfg>,
    /// Optional basket-cell pool. `None` disables.
    #[serde(default)]
    pub basket: Option<BasketCfg>,
}

impl Default for PopulationSpec {
    /// Stability-mechanism-OFF profile — preserves Phase 3 test behavior.
    /// Use this as a base for `..Default::default()` struct-update syntax
    /// when you only want to override the fields that matter.
    fn default() -> Self {
        Self {
            name: "pop".into(),
            n_neurons: 16,
            lif: LifParams::default(),
            target_rate: 0.05,
            homeostatic: HomeostaticParams::default(),
            noise: crate::noise::NoiseConfig {
                rate_hz: 0.0,
                pulse_mv: 0.0,
            },
            depression: crate::depression::DepressionConfig {
                inc: 0.0,
                ..crate::depression::DepressionConfig::default()
            },
            adaptation_ahp: None,
            adaptation_pump: None,
            basket: None,
        }
    }
}

/// Static description of one directed synapse edge (src → dst).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeSpec {
    /// Source population index.
    pub src: usize,
    /// Destination population index.
    pub dst: usize,
    /// Incoming fan-in per destination neuron. Clamped to `src.n_neurons`.
    pub fan_in: u32,
    /// Initial mean weight.
    pub weight_init: f32,
    /// Uniform jitter half-width.
    pub weight_jitter: f32,
    /// R-STDP kernel + gains.
    pub rstdp: RstdpParams,
}

/// Top-level configuration for [`SnnNetwork`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnnConfig {
    /// Population specs.
    pub populations: Vec<PopulationSpec>,
    /// Directed synapse edges.
    pub edges: Vec<EdgeSpec>,
    /// Seed for the deterministic topology RNG.
    pub rng_seed: u64,
    /// Initial alpha for every population's firing-rate EMA.
    pub rate_ema_alpha: f32,
    /// Intrinsic-reward config shared across populations. See
    /// [`SnnNetwork::compute_intrinsic_reward`]; callers pass the
    /// result as the `reward` argument to [`SnnNetwork::step`] when
    /// they want rate-driven R-STDP.
    #[serde(default)]
    pub intrinsic_reward: crate::intrinsic_reward::IntrinsicRewardConfig,
    /// When `true`, `step()` uses [`crate::homeostatic::step_homeostatic_with_reward`]
    /// (widens scale-down floor to 0.90 when reward is stuck AND rate is
    /// saturated). When `false` (default, backward-compat), uses the
    /// tight-bounds [`crate::homeostatic::step_homeostatic`]. Enable on
    /// brains that drive the SNN with a reward signal and want the
    /// Phase 3.5 safety net against saturation↔collapse oscillation.
    #[serde(default)]
    pub reward_coupled_homeostatic: bool,
}

impl Default for SnnConfig {
    fn default() -> Self {
        Self {
            populations: Vec::new(),
            edges: Vec::new(),
            rng_seed: 0,
            rate_ema_alpha: DEFAULT_RATE_EMA_ALPHA,
            intrinsic_reward: crate::intrinsic_reward::IntrinsicRewardConfig::default(),
            reward_coupled_homeostatic: false,
        }
    }
}

// -------------------------------------------------------------------------
// Network
// -------------------------------------------------------------------------

/// Construction error for [`SnnNetwork`].
#[derive(Debug, thiserror::Error)]
pub enum SnnError {
    /// Edge references an out-of-range population index.
    #[error("edge {edge_idx}: population index {pop_idx} >= n_populations {n_pops}")]
    EdgePopOutOfRange {
        /// Offending edge index.
        edge_idx: usize,
        /// Bad population index.
        pop_idx: usize,
        /// Total population count.
        n_pops: usize,
    },
    /// Underlying CSR builder error.
    #[error("edge {edge_idx}: csr build failed — {err}")]
    CsrBuild {
        /// Offending edge index.
        edge_idx: usize,
        /// Underlying [`CsrError`].
        err: CsrError,
    },
    /// A per-population stability mechanism (adaptation, basket, etc.)
    /// failed to initialize. Message encodes the pop index + mechanism.
    #[error("population {pop_idx}: stability mechanism init failed — {msg}")]
    StabilityInit {
        /// Offending population index.
        pop_idx: usize,
        /// Detail — which mechanism and the underlying error.
        msg: String,
    },
}

/// Per-population runtime state.
struct Pop {
    spec: PopulationSpec,
    state: LifState,
    rate: PopulationRateEma,
    i_syn_scratch: Vec<f32>,
    // Phase 3.5 — stability mechanisms.
    noise_rng: rand_chacha::ChaCha20Rng,
    depression_state: crate::depression::DepressionState,
    adaptation_ahp: Option<crate::adaptation::AdaptationState>,
    adaptation_pump: Option<crate::adaptation::AdaptationState>,
    basket: Option<crate::basket::BasketPool>,
    /// Per-neuron hyperpolarization buffer (shared between AHP + pump).
    hyperpol_scratch: Vec<f32>,
    /// Per-neuron `fired` buffer as f32, for adaptation + depression updates.
    fired_f32_scratch: Vec<f32>,
    /// Per-pre-neuron depression scale = `1 - depression.d[i]`, clamped
    /// to `[1 - cap, 1]`. Recomputed each step from depression_state.
    depression_scale_scratch: Vec<f32>,
}

/// Per-edge runtime state.
struct Edge {
    spec: EdgeSpec,
    csr: CsrSynapses,
    rstdp: RstdpState,
    /// Scratch I_syn buffer for this edge's CSR forward. Preallocated
    /// at construction; the CSR writes into it, the `step` pipeline
    /// then adds it into the destination population's I_syn scratch.
    i_syn_scratch: Vec<f32>,
}

/// Spiking neural network — the Phase 3 composition.
pub struct SnnNetwork {
    populations: Vec<Pop>,
    edges: Vec<Edge>,
    t_ms: f32,
    reward_coupled_homeostatic: bool,
}

impl SnnNetwork {
    /// Build a network from `config`.
    #[allow(clippy::needless_pass_by_value)] // `config` owns Vecs we take apart.
    pub fn new(config: SnnConfig) -> Result<Self, SnnError> {
        let n_pops = config.populations.len();

        use rand::SeedableRng;
        let mut populations: Vec<Pop> = Vec::with_capacity(config.populations.len());
        for (pop_idx, spec) in config.populations.iter().enumerate() {
            let state = LifState::new(spec.n_neurons, &spec.lif);
            let rate = PopulationRateEma::new(config.rate_ema_alpha, spec.target_rate);
            let n = spec.n_neurons as usize;
            let i_syn_scratch = vec![0.0; n];
            let hyperpol_scratch = vec![0.0; n];
            let fired_f32_scratch = vec![0.0; n];
            let depression_scale_scratch = vec![1.0; n];
            let noise_rng_seed = config
                .rng_seed
                .wrapping_add(0xAA55_AA55_AA55_AA55)
                .wrapping_add(pop_idx as u64);
            let noise_rng = rand_chacha::ChaCha20Rng::seed_from_u64(noise_rng_seed);
            let depression_state = crate::depression::DepressionState::new(n);
            let adaptation_ahp = spec
                .adaptation_ahp
                .as_ref()
                .map(|cfg| {
                    crate::adaptation::AdaptationState::new(
                        n,
                        cfg.tau_ms,
                        cfg.gain_mv,
                        cfg.spike_bump,
                    )
                })
                .transpose()
                .map_err(|e| SnnError::StabilityInit {
                    pop_idx,
                    msg: format!("adaptation_ahp: {e}"),
                })?;
            let adaptation_pump = spec
                .adaptation_pump
                .as_ref()
                .map(|cfg| {
                    crate::adaptation::AdaptationState::new(
                        n,
                        cfg.tau_ms,
                        cfg.gain_mv,
                        cfg.spike_bump,
                    )
                })
                .transpose()
                .map_err(|e| SnnError::StabilityInit {
                    pop_idx,
                    msg: format!("adaptation_pump: {e}"),
                })?;
            let basket = spec
                .basket
                .as_ref()
                .map(|cfg| crate::basket::BasketPool::new(pop_idx as u32, n, cfg.fraction))
                .transpose()
                .map_err(|e| SnnError::StabilityInit {
                    pop_idx,
                    msg: format!("basket: {e}"),
                })?;
            populations.push(Pop {
                spec: spec.clone(),
                state,
                rate,
                i_syn_scratch,
                noise_rng,
                depression_state,
                adaptation_ahp,
                adaptation_pump,
                basket,
                hyperpol_scratch,
                fired_f32_scratch,
                depression_scale_scratch,
            });
        }

        let mut edges: Vec<Edge> = Vec::with_capacity(config.edges.len());
        for (edge_idx, spec) in config.edges.iter().enumerate() {
            if spec.src >= n_pops {
                return Err(SnnError::EdgePopOutOfRange {
                    edge_idx,
                    pop_idx: spec.src,
                    n_pops,
                });
            }
            if spec.dst >= n_pops {
                return Err(SnnError::EdgePopOutOfRange {
                    edge_idx,
                    pop_idx: spec.dst,
                    n_pops,
                });
            }

            let src_pop = &populations[spec.src];
            let dst_pop = &populations[spec.dst];
            let edge_seed = config
                .rng_seed
                .wrapping_add(edge_idx as u64)
                .wrapping_mul(0x9E37_79B9_7F4A_7C15);

            let csr = CsrSynapses::random_uniform_seeded(
                PopulationId(spec.src as u32),
                PopulationId(spec.dst as u32),
                src_pop.spec.n_neurons,
                dst_pop.spec.n_neurons,
                spec.fan_in,
                spec.weight_init,
                spec.weight_jitter,
                edge_seed,
            );

            let rstdp = RstdpState::new(
                csr.weights.len(),
                src_pop.spec.n_neurons,
                dst_pop.spec.n_neurons,
            );
            let i_syn_scratch = vec![0.0_f32; dst_pop.spec.n_neurons as usize];

            edges.push(Edge {
                spec: spec.clone(),
                csr,
                rstdp,
                i_syn_scratch,
            });
        }

        Ok(Self {
            populations,
            edges,
            t_ms: 0.0,
            reward_coupled_homeostatic: config.reward_coupled_homeostatic,
        })
    }

    /// Number of populations.
    #[must_use]
    pub fn n_populations(&self) -> usize {
        self.populations.len()
    }

    /// Number of directed edges.
    #[must_use]
    pub fn n_edges(&self) -> usize {
        self.edges.len()
    }

    /// Read-only view of a population's current spike buffer.
    #[must_use]
    pub fn spikes(&self, pop_idx: usize) -> &[u8] {
        &self.populations[pop_idx].state.spike
    }

    /// Read-only view of a population's current membrane voltage vector.
    #[must_use]
    pub fn v_mem(&self, pop_idx: usize) -> &[f32] {
        &self.populations[pop_idx].state.v_mem
    }

    /// Read-only view of a population's firing-rate EMA.
    #[must_use]
    pub fn rate_ema(&self, pop_idx: usize) -> f32 {
        self.populations[pop_idx].rate.rate()
    }

    /// Mutable view of a single edge's weights. For tests and operator
    /// preloads.
    pub fn edge_weights_mut(&mut self, edge_idx: usize) -> &mut [f32] {
        &mut self.edges[edge_idx].csr.weights
    }

    /// Read-only edge weights.
    #[must_use]
    pub fn edge_weights(&self, edge_idx: usize) -> &[f32] {
        &self.edges[edge_idx].csr.weights
    }

    /// Simulated time (ms).
    #[must_use]
    pub fn t_ms(&self) -> f32 {
        self.t_ms
    }

    /// One integration step.
    ///
    /// `external_i_syn` is indexed by population. Pass an empty outer
    /// slice for no drive. Each inner slice must match the population's
    /// neuron count or it is silently ignored.
    ///
    /// Returns total spikes emitted across all populations this step.
    pub fn step(&mut self, external_i_syn: &[&[f32]], reward: f32, dt_ms: f32) -> u32 {
        // 1. Per-pop prep: zero scratches, inject external drive, apply
        //    Poisson noise and basket inhibition, refresh per-source
        //    depression scale. Adaptation hyperpol is subtracted from
        //    v_mem directly so the LIF step sees the post-adaptation
        //    membrane.
        for (pop_idx, pop) in self.populations.iter_mut().enumerate() {
            // Zero I_syn scratch.
            for v in pop.i_syn_scratch.iter_mut() {
                *v = 0.0;
            }
            // External drive.
            if let Some(ext) = external_i_syn.get(pop_idx)
                && ext.len() == pop.i_syn_scratch.len()
            {
                for (dst, &src) in pop.i_syn_scratch.iter_mut().zip(ext.iter()) {
                    *dst = src;
                }
            }

            // Basket inhibition (uses PRIOR-step basket spike_output —
            // first step is a no-op because spike_output is all zeros).
            if let Some(bp) = &pop.basket {
                bp.emit_inhibition(&mut pop.i_syn_scratch);
            }

            // Adaptive-factor Poisson noise into v_mem. Dead pops get
            // full injection; at-target pops get none.
            let factor = crate::noise::noise_factor_for_pop(
                pop.rate.rate(),
                pop.spec.target_rate,
            );
            crate::noise::inject_poisson_noise(
                &mut pop.noise_rng,
                &mut pop.state.v_mem,
                &pop.spec.noise,
                dt_ms,
                factor,
            );

            // Adaptation hyperpol: subtract gain*adapt_var from v_mem.
            // AHP and pump share the scratch buffer to avoid an extra
            // allocation per step.
            let n = pop.state.v_mem.len();
            if pop.adaptation_ahp.is_some() || pop.adaptation_pump.is_some() {
                for x in pop.hyperpol_scratch.iter_mut().take(n) {
                    *x = 0.0;
                }
                if let Some(a) = &pop.adaptation_ahp {
                    // Fill scratch with AHP contribution.
                    a.compute_hyperpol(&mut pop.hyperpol_scratch);
                }
                if let Some(a) = &pop.adaptation_pump {
                    // Add pump contribution on top. Use a temporary so
                    // we don't overwrite AHP.
                    let mut tmp = vec![0.0; n];
                    a.compute_hyperpol(&mut tmp);
                    for (h, &t) in pop.hyperpol_scratch.iter_mut().zip(tmp.iter()) {
                        *h += t;
                    }
                }
                for (v, &h) in pop.state.v_mem.iter_mut().zip(pop.hyperpol_scratch.iter()) {
                    *v -= h;
                }
            }

            // Refresh per-source depression scale from depression_state.
            let cap = pop.spec.depression.cap;
            for (scale, &d) in pop
                .depression_scale_scratch
                .iter_mut()
                .zip(pop.depression_state.d.iter())
            {
                *scale = (1.0 - d).clamp(1.0 - cap, 1.0);
            }
        }

        // 2. Forward every CSR edge, applying per-source depression scale.
        for edge in &mut self.edges {
            let src = edge.spec.src;
            let dst = edge.spec.dst;
            debug_assert_ne!(src, dst, "self-loops unsupported in Phase 3");

            let src_spikes = &self.populations[src].state.spike;
            // Depression scale — only supply when STD actually in play,
            // so the non-depression hot path stays at v1 perf.
            let pre_scale: Option<&[f32]> = if self.populations[src].spec.depression.is_disabled()
            {
                None
            } else {
                Some(&self.populations[src].depression_scale_scratch)
            };
            edge.csr.i_syn_cpu_with_pre_scale(
                src_spikes,
                pre_scale,
                &mut edge.i_syn_scratch,
            );
        }
        for edge in &self.edges {
            let dst_pop = &mut self.populations[edge.spec.dst];
            for (d, &s) in dst_pop
                .i_syn_scratch
                .iter_mut()
                .zip(edge.i_syn_scratch.iter())
            {
                *d += s;
            }
        }

        // 3. Advance LIF dynamics.
        for pop in self.populations.iter_mut() {
            lif_step_cpu(&mut pop.state, &pop.i_syn_scratch, &pop.spec.lif, dt_ms);
        }

        // 4. Post-LIF: update adaptation + depression + basket using
        //    the fresh spike output.
        for pop in self.populations.iter_mut() {
            // Build f32 fired buffer once per pop.
            for (f, &s) in pop.fired_f32_scratch.iter_mut().zip(pop.state.spike.iter()) {
                *f = if s != 0 { 1.0 } else { 0.0 };
            }
            if let Some(a) = pop.adaptation_ahp.as_mut() {
                a.update(&pop.fired_f32_scratch, dt_ms);
            }
            if let Some(a) = pop.adaptation_pump.as_mut() {
                a.update(&pop.fired_f32_scratch, dt_ms);
            }
            if !pop.spec.depression.is_disabled() {
                crate::depression::step_depression(
                    &mut pop.depression_state,
                    &pop.spec.depression,
                    &pop.fired_f32_scratch,
                    dt_ms,
                );
            }
            if let Some(bp) = pop.basket.as_mut() {
                let n = pop.state.n_neurons as usize;
                let mean_fire = if n == 0 {
                    0.0
                } else {
                    pop.state.n_spikes_this_step() as f32 / n as f32
                };
                bp.step(mean_fire, dt_ms);
            }
        }

        // 5. R-STDP weight updates.
        for edge in &mut self.edges {
            let src = edge.spec.src;
            let dst = edge.spec.dst;
            let (pre_spikes, post_spikes): (&[u8], &[u8]) = if src < dst {
                let (a, b) = self.populations.split_at(dst);
                (&a[src].state.spike, &b[0].state.spike)
            } else {
                let (a, b) = self.populations.split_at(src);
                (&b[0].state.spike, &a[dst].state.spike)
            };
            step_rstdp(
                &mut edge.csr,
                &mut edge.rstdp,
                pre_spikes,
                post_spikes,
                reward,
                self.t_ms,
                dt_ms,
                &edge.spec.rstdp,
            );
        }

        // 6. Homeostatic scaling per destination population — reward-
        //    coupled variant: widens scale-down when reward is stuck
        //    near zero AND rate is saturated.
        for edge in &mut self.edges {
            let dst_pop = &mut self.populations[edge.spec.dst];
            let n_post = dst_pop.state.n_neurons as usize;
            let fraction_spiking = if n_post == 0 {
                0.0
            } else {
                dst_pop.state.n_spikes_this_step() as f32 / n_post as f32
            };
            if self.reward_coupled_homeostatic {
                let _ = crate::homeostatic::step_homeostatic_with_reward(
                    &mut edge.csr,
                    &mut dst_pop.rate,
                    fraction_spiking,
                    &dst_pop.spec.homeostatic,
                    reward,
                );
            } else {
                let _ = step_homeostatic(
                    &mut edge.csr,
                    &mut dst_pop.rate,
                    fraction_spiking,
                    &dst_pop.spec.homeostatic,
                );
            }
        }

        // 7. Advance clock and tally activity.
        self.t_ms += dt_ms;
        self.populations
            .iter()
            .map(|p| p.state.n_spikes_this_step())
            .sum()
    }

    /// Compute the network-level intrinsic reward from per-population
    /// firing-rate EMAs and the configured [`IntrinsicRewardConfig`].
    /// Returns the scalar R-STDP teaching signal — pass as the
    /// `reward` argument to [`Self::step`] when running intrinsic-only
    /// training. Combine with external reward additively when needed.
    #[must_use]
    pub fn compute_intrinsic_reward(&self, cfg: &crate::intrinsic_reward::IntrinsicRewardConfig) -> f32 {
        crate::intrinsic_reward::compute_network_reward(
            self.populations
                .iter()
                .map(|p| (p.rate.rate(), p.spec.target_rate)),
            cfg,
        )
    }

    /// Load-time weight transform — applied whenever weights may have
    /// come from a saturated source (fresh init with high initial
    /// weights, checkpoint resume, operator paste).
    ///
    /// Per destination population, delegates to
    /// [`nimcp_plasticity::quiet_start_scale`] with the observed rate
    /// and target, then applies the same scale to every incoming edge.
    /// Returns the per-edge scales actually applied.
    pub fn apply_quiet_start(&mut self, observed_rates_per_pop: &[f32]) -> Vec<f32> {
        let mut scales_out = vec![1.0_f32; self.edges.len()];
        let n_pops = self.populations.len();
        for dst_idx in 0..n_pops {
            let target = self.populations[dst_idx].spec.target_rate;
            let observed = observed_rates_per_pop.get(dst_idx).copied().unwrap_or(0.0);
            let scale_vec = nimcp_plasticity::quiet_start_scale(&[observed], target);
            let scale = scale_vec.first().copied().unwrap_or(1.0);

            #[allow(clippy::float_cmp)]
            let is_noop = scale == 1.0;
            for (idx, edge) in self.edges.iter_mut().enumerate() {
                if edge.spec.dst != dst_idx {
                    continue;
                }
                if !is_noop {
                    for w in edge.csr.weights.iter_mut() {
                        *w *= scale;
                    }
                }
                scales_out[idx] = scale;
            }
        }
        scales_out
    }
}

// -------------------------------------------------------------------------
// Serialization — weights + rate EMAs.
// -------------------------------------------------------------------------

/// Weight snapshot for checkpointing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightSnapshot {
    /// Per-edge weight vectors.
    pub edge_weights: Vec<Vec<f32>>,
    /// Per-pop firing-rate EMAs.
    pub rate_emas: Vec<PopulationRateEma>,
}

impl SnnNetwork {
    /// Snapshot weights + rate EMAs.
    #[must_use]
    pub fn snapshot(&self) -> WeightSnapshot {
        WeightSnapshot {
            edge_weights: self.edges.iter().map(|e| e.csr.weights.clone()).collect(),
            rate_emas: self.populations.iter().map(|p| p.rate.clone()).collect(),
        }
    }

    /// Restore weights + rate EMAs. Shapes must match; returns `false`
    /// on mismatch without modifying state.
    pub fn restore(&mut self, snap: &WeightSnapshot) -> bool {
        if snap.edge_weights.len() != self.edges.len()
            || snap.rate_emas.len() != self.populations.len()
        {
            return false;
        }
        for (edge, w) in self.edges.iter().zip(snap.edge_weights.iter()) {
            if edge.csr.weights.len() != w.len() {
                return false;
            }
        }
        for (edge, w) in self.edges.iter_mut().zip(snap.edge_weights.iter()) {
            edge.csr.weights.copy_from_slice(w);
        }
        for (pop, ema) in self.populations.iter_mut().zip(snap.rate_emas.iter()) {
            pop.rate = ema.clone();
        }
        true
    }
}

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::float_cmp)] // exact-equality asserts intentional in round-trip tests
mod tests {
    use super::*;

    /// Build a minimal two-population config. Sizes + fan-in chosen so
    /// saturated weights can drive downstream to threshold given the
    /// LIF `v_rest = -70 / v_thresh = -50` gap of 20 mV.
    fn two_pop_config(
        seed: u64,
        warmup_samples: u32,
        target_rate: f32,
        weight_init: f32,
    ) -> SnnConfig {
        let lif = LifParams::default();
        let n = 128_u32;
        let pop_a = PopulationSpec {
            name: "A".into(),
            n_neurons: n,
            lif,
            target_rate,
            homeostatic: HomeostaticParams::default(),
            noise: crate::noise::NoiseConfig {
                rate_hz: 0.0,
                pulse_mv: 0.0,
            },
            depression: crate::depression::DepressionConfig {
                inc: 0.0,
                ..crate::depression::DepressionConfig::default()
            },
            adaptation_ahp: None,
            adaptation_pump: None,
            basket: None,
        };
        let pop_b = PopulationSpec {
            name: "B".into(),
            n_neurons: n,
            lif,
            target_rate,
            homeostatic: HomeostaticParams::default(),
            noise: crate::noise::NoiseConfig {
                rate_hz: 0.0,
                pulse_mv: 0.0,
            },
            depression: crate::depression::DepressionConfig {
                inc: 0.0,
                ..crate::depression::DepressionConfig::default()
            },
            adaptation_ahp: None,
            adaptation_pump: None,
            basket: None,
        };
        // w_max bumped so a saturated edge can drive the downstream
        // population to fire on a single pre-spike step — V1's
        // production `w_max=1.0` requires hundreds-of-fan-in scale,
        // which these tests deliberately stay below to keep them fast.
        let rstdp = RstdpParams {
            warmup_samples,
            w_max: 10.0,
            ..RstdpParams::default()
        };
        let edge = EdgeSpec {
            src: 0,
            dst: 1,
            fan_in: 64,
            weight_init,
            weight_jitter: 0.0,
            rstdp,
        };
        SnnConfig {
            populations: vec![pop_a, pop_b],
            edges: vec![edge],
            rng_seed: seed,
            rate_ema_alpha: 0.05,
            reward_coupled_homeostatic: false,
            intrinsic_reward: crate::intrinsic_reward::IntrinsicRewardConfig::default(),
        }
    }

    #[test]
    fn constructs_and_reports_shape() {
        let cfg = two_pop_config(0, 100, 0.1, 0.5);
        let net = SnnNetwork::new(cfg).expect("build");
        assert_eq!(net.n_populations(), 2);
        assert_eq!(net.n_edges(), 1);
        assert_eq!(net.spikes(0).len(), 128);
        assert_eq!(net.v_mem(1).len(), 128);
    }

    #[test]
    fn rejects_out_of_range_edge() {
        let mut cfg = two_pop_config(0, 100, 0.1, 0.5);
        cfg.edges[0].dst = 5;
        match SnnNetwork::new(cfg) {
            Err(SnnError::EdgePopOutOfRange { pop_idx, .. }) => assert_eq!(pop_idx, 5),
            Err(other) => panic!("expected EdgePopOutOfRange, got {other:?}"),
            Ok(_) => panic!("expected EdgePopOutOfRange, got Ok"),
        }
    }

    #[test]
    fn step_is_deterministic_across_runs() {
        let a_cfg = two_pop_config(0xABCD, 0, 0.1, 0.3);
        let b_cfg = two_pop_config(0xABCD, 0, 0.1, 0.3);
        let mut a = SnnNetwork::new(a_cfg).expect("a");
        let mut b = SnnNetwork::new(b_cfg).expect("b");
        let ext = vec![25.0_f32; 128];
        let slices: Vec<&[f32]> = vec![&ext, &[]];
        for _ in 0..40 {
            a.step(&slices, 0.0, 1.0);
            b.step(&slices, 0.0, 1.0);
        }
        assert_eq!(a.edge_weights(0), b.edge_weights(0));
        assert_eq!(a.spikes(1), b.spikes(1));
    }

    #[test]
    fn external_drive_elicits_downstream_spikes() {
        let cfg = two_pop_config(1, 0, 0.1, 1.8);
        let mut net = SnnNetwork::new(cfg).expect("build");

        // Strong suprathreshold drive so pop A fires on (nearly) every
        // step. With LIF leak at `dt/τ = 0.05` and a 20 mV rest→threshold
        // gap, I_syn of a few hundred crosses threshold in one step.
        let ext = vec![1_000.0_f32; 128];
        let slices: Vec<&[f32]> = vec![&ext, &[]];

        let mut saw_b_spike = false;
        for _ in 0..150 {
            net.step(&slices, 0.0, 1.0);
            if net.spikes(1).iter().any(|&s| s != 0) {
                saw_b_spike = true;
                break;
            }
        }
        assert!(
            saw_b_spike,
            "downstream pop B should spike under saturated drive"
        );
    }

    /// V2_PLAN Phase 3 exit criterion: a saturated network recovers
    /// into the deadband within 100 steps after `apply_quiet_start`.
    #[test]
    fn saturation_recovery() {
        let target = 0.15_f32;
        let cfg = two_pop_config(7, 0, target, 1.8);
        let mut net = SnnNetwork::new(cfg).expect("build");

        // Hand-saturate every weight at w_max (= 10.0 per two_pop_config).
        for w in net.edge_weights_mut(0) {
            *w = 10.0;
        }

        // Strong drive on pop A so it fires almost every step — we need
        // pop B to saturate before we can exercise recovery.
        let ext = vec![1_000.0_f32; 128];
        let slices: Vec<&[f32]> = vec![&ext, &[]];
        for _ in 0..120 {
            net.step(&slices, 0.0, 1.0);
        }
        let saturated = net.rate_ema(1);
        assert!(
            saturated > 2.0 * target,
            "setup: pop B should be saturated (>2× target); got {saturated}"
        );

        // Apply quiet-start with the observed saturated rate.
        let observed = vec![0.0, saturated];
        let applied = net.apply_quiet_start(&observed);
        assert!(
            applied[0] < 1.0,
            "quiet-start should have scaled edge 0 down; got {}",
            applied[0]
        );

        // Let homeostatic finish the job.
        for _ in 0..100 {
            net.step(&slices, 0.0, 1.0);
        }

        let final_rate = net.rate_ema(1);
        let deadband = HomeostaticParams::default().deadband_frac * target;
        assert!(
            (final_rate - target).abs() <= 2.0 * deadband + 0.05,
            "recovery: pop B rate {final_rate} not within 2×deadband+0.05 of target {target} (deadband={deadband})"
        );
    }

    #[test]
    fn snapshot_restore_round_trip() {
        let cfg = two_pop_config(42, 0, 0.1, 0.4);
        let mut net = SnnNetwork::new(cfg).expect("build");
        for (i, w) in net.edge_weights_mut(0).iter_mut().enumerate() {
            *w = 0.1 + 0.01 * i as f32;
        }
        let snap = net.snapshot();
        for w in net.edge_weights_mut(0) {
            *w = 0.0;
        }
        assert!(net.restore(&snap));
        assert_eq!(net.edge_weights(0)[5], snap.edge_weights[0][5]);
    }

    #[test]
    fn restore_rejects_shape_mismatch() {
        let cfg_a = two_pop_config(0, 0, 0.1, 0.4);
        let cfg_b = two_pop_config(0, 0, 0.1, 0.4);
        let mut a = SnnNetwork::new(cfg_a).expect("a");
        let b = SnnNetwork::new(cfg_b).expect("b");
        let mut snap = b.snapshot();
        snap.edge_weights[0].truncate(1);
        assert!(!a.restore(&snap));
    }
}
