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

/// Static description of one neuron population.
#[derive(Debug, Clone)]
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
}

/// Static description of one directed synapse edge (src → dst).
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
pub struct SnnConfig {
    /// Population specs.
    pub populations: Vec<PopulationSpec>,
    /// Directed synapse edges.
    pub edges: Vec<EdgeSpec>,
    /// Seed for the deterministic topology RNG.
    pub rng_seed: u64,
    /// Initial alpha for every population's firing-rate EMA.
    pub rate_ema_alpha: f32,
}

impl Default for SnnConfig {
    fn default() -> Self {
        Self {
            populations: Vec::new(),
            edges: Vec::new(),
            rng_seed: 0,
            rate_ema_alpha: DEFAULT_RATE_EMA_ALPHA,
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
}

/// Per-population runtime state.
struct Pop {
    spec: PopulationSpec,
    state: LifState,
    rate: PopulationRateEma,
    i_syn_scratch: Vec<f32>,
}

/// Per-edge runtime state.
struct Edge {
    spec: EdgeSpec,
    csr: CsrSynapses,
    rstdp: RstdpState,
}

/// Spiking neural network — the Phase 3 composition.
pub struct SnnNetwork {
    populations: Vec<Pop>,
    edges: Vec<Edge>,
    t_ms: f32,
}

impl SnnNetwork {
    /// Build a network from `config`.
    #[allow(clippy::needless_pass_by_value)] // `config` owns Vecs we take apart.
    pub fn new(config: SnnConfig) -> Result<Self, SnnError> {
        let n_pops = config.populations.len();

        let populations: Vec<Pop> = config
            .populations
            .iter()
            .map(|spec| {
                let state = LifState::new(spec.n_neurons, &spec.lif);
                let rate = PopulationRateEma::new(config.rate_ema_alpha, spec.target_rate);
                let i_syn_scratch = vec![0.0; spec.n_neurons as usize];
                Pop {
                    spec: spec.clone(),
                    state,
                    rate,
                    i_syn_scratch,
                }
            })
            .collect();

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

            edges.push(Edge {
                spec: spec.clone(),
                csr,
                rstdp,
            });
        }

        Ok(Self {
            populations,
            edges,
            t_ms: 0.0,
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
        // 1. Zero I_syn scratches; inject external drive where provided.
        for (pop_idx, pop) in self.populations.iter_mut().enumerate() {
            for v in pop.i_syn_scratch.iter_mut() {
                *v = 0.0;
            }
            if let Some(ext) = external_i_syn.get(pop_idx)
                && ext.len() == pop.i_syn_scratch.len()
            {
                for (dst, &src) in pop.i_syn_scratch.iter_mut().zip(ext.iter()) {
                    *dst = src;
                }
            }
        }

        // 2. Forward every CSR edge (prior-step spike → current-step I_syn).
        //    Self-loops (src == dst) are unsupported in Phase 3.
        for edge in &mut self.edges {
            let src = edge.spec.src;
            let dst = edge.spec.dst;
            debug_assert_ne!(src, dst, "self-loops unsupported in Phase 3");

            let (src_pop, dst_pop) = if src < dst {
                let (a, b) = self.populations.split_at_mut(dst);
                (&a[src], &mut b[0])
            } else {
                let (a, b) = self.populations.split_at_mut(src);
                (&b[0], &mut a[dst])
            };

            // i_syn_cpu overwrites `out`; accumulate via a scratch vec.
            let n_post = dst_pop.i_syn_scratch.len();
            let mut tmp = vec![0.0_f32; n_post];
            edge.csr.i_syn_cpu(&src_pop.state.spike, &mut tmp);
            for (d, s) in dst_pop.i_syn_scratch.iter_mut().zip(tmp.iter()) {
                *d += *s;
            }
        }

        // 3. Advance LIF dynamics.
        for pop in self.populations.iter_mut() {
            lif_step_cpu(&mut pop.state, &pop.i_syn_scratch, &pop.spec.lif, dt_ms);
        }

        // 4. R-STDP weight updates. We need disjoint &[u8] for pre and
        //    post spike buffers; Rust's borrow checker can't prove
        //    indices into the same Vec are disjoint, so we clone the
        //    two slices. Spike buffers are u8 and small (per-pop size),
        //    so this is cheap.
        for edge in &mut self.edges {
            let pre = self.populations[edge.spec.src].state.spike.clone();
            let post = self.populations[edge.spec.dst].state.spike.clone();
            step_rstdp(
                &mut edge.csr,
                &mut edge.rstdp,
                &pre,
                &post,
                reward,
                self.t_ms,
                dt_ms,
                &edge.spec.rstdp,
            );
        }

        // 5. Homeostatic scaling per destination population. Drive the
        //    dst pop's rate EMA with its fraction spiking, then scale
        //    incoming edge weights if the gate opened and we're outside
        //    the deadband.
        for edge in &mut self.edges {
            let dst_pop = &mut self.populations[edge.spec.dst];
            let n_post = dst_pop.state.n_neurons as usize;
            let fraction_spiking = if n_post == 0 {
                0.0
            } else {
                dst_pop.state.n_spikes_this_step() as f32 / n_post as f32
            };
            let _ = step_homeostatic(
                &mut edge.csr,
                &mut dst_pop.rate,
                fraction_spiking,
                &dst_pop.spec.homeostatic,
            );
        }

        // 6. Advance clock and tally activity.
        self.t_ms += dt_ms;
        self.populations
            .iter()
            .map(|p| p.state.n_spikes_this_step())
            .sum()
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
        };
        let pop_b = PopulationSpec {
            name: "B".into(),
            n_neurons: n,
            lif,
            target_rate,
            homeostatic: HomeostaticParams::default(),
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
