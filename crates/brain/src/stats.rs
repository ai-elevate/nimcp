//! Phase 6 — read-only brain introspection.
//!
//! [`BrainStats`] bundles one snapshot of every configured subsystem.
//! It's a pure read over the networks — no mutation, no side effects,
//! no allocation beyond the owned fields — so callers can poll it on
//! any cadence (dashboards, training monitors, Python REPLs).
//!
//! # Sections
//!
//! - [`AdaptiveStats`] — MLP layer widths, parameter counts, per-layer
//!   weight + bias distribution summaries (min / max / mean / std).
//! - [`SnnStats`] — population and edge counts, per-population firing
//!   rate EMA + current spike count, per-edge weight distribution.
//! - [`LnnStats`] — LTC layer shapes, tau-base distribution, current
//!   transient state L2 norm per layer.
//! - [`MemoryStats`] — Z-Ladder tier counts, landmark count, cumulative
//!   promotion / demotion / eviction tallies.
//! - [`LossStats`] — per-network last training loss + EMA + call count.
//!   Populated by [`crate::Brain::learn`] (adaptive) and
//!   [`crate::Brain::lnn_train_step_mse`] (LNN). SNN is reward-driven
//!   with no scalar loss, so it is absent from this section by design.
//!
//! Every field is serde-serializable so [`Brain::stats_json`] can
//! return a JSON string the Python binding hands back as a dict.

use std::collections::HashMap;

use ndarray::Array1;
use nimcp_adaptive::{AdaptiveNet, LayerWeights};
use nimcp_lnn::{LnnNetwork, LtcState};
use nimcp_memory::ZLadder;
use nimcp_snn::SnnNetwork;
use serde::{Deserialize, Serialize};

/// Top-level stats bundle returned by [`crate::Brain::stats`].
///
/// Sections are `Option` — a section is `Some` iff that subsystem was
/// configured on the brain at build time. Callers should not assume
/// every section is populated.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainStats {
    /// The brain's configured `rng_seed`.
    pub rng_seed: u64,
    /// Adaptive (MLP) subsystem stats.
    pub adaptive: Option<AdaptiveStats>,
    /// SNN subsystem stats.
    pub snn: Option<SnnStats>,
    /// LNN subsystem stats.
    pub lnn: Option<LnnStats>,
    /// Memory (Z-Ladder) subsystem stats.
    pub memory: Option<MemoryStats>,
    /// Training loss summary across networks that report scalar loss.
    pub loss: LossStats,
}

// -------------------------------------------------------------------------
// Loss
// -------------------------------------------------------------------------

/// Default EMA smoothing factor for [`LossTracker`]. Picked to match the
/// SNN's `rate_ema_alpha` default — fast enough to track step-level
/// variation, slow enough that a single outlier doesn't dominate.
pub const DEFAULT_LOSS_EMA_ALPHA: f32 = 0.05;

/// Rolling loss tracker for one network. Updated on every training step
/// that produces a scalar loss; read from [`crate::Brain::stats`].
///
/// `last` and `ema` are `None` until the first update — a brain that
/// has never been trained reports `count == 0` with both summary
/// fields absent.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct LossTracker {
    /// Most recent observed loss. `None` before the first update.
    pub last: Option<f32>,
    /// EMA of observed losses: `ema ← α · loss + (1 − α) · ema`.
    /// `None` before the first update. Initialized to the first loss
    /// observation (no cold-start bias).
    pub ema: Option<f32>,
    /// Number of `observe` calls since construction (monotonic).
    pub count: u64,
    /// EMA smoothing factor in `(0, 1]`. Higher α tracks recent losses
    /// more aggressively; lower α is smoother but slower to respond.
    pub ema_alpha: f32,
}

impl Default for LossTracker {
    fn default() -> Self {
        Self {
            last: None,
            ema: None,
            count: 0,
            ema_alpha: DEFAULT_LOSS_EMA_ALPHA,
        }
    }
}

impl LossTracker {
    /// Construct an empty tracker with a custom smoothing factor.
    /// Panics if `ema_alpha` is not in `(0, 1]` — caller programming error.
    #[must_use]
    pub fn with_alpha(ema_alpha: f32) -> Self {
        assert!(
            ema_alpha > 0.0 && ema_alpha <= 1.0,
            "ema_alpha must be in (0, 1], got {ema_alpha}"
        );
        Self {
            ema_alpha,
            ..Self::default()
        }
    }

    /// Fold one loss observation into the tracker.
    pub fn observe(&mut self, loss: f32) {
        self.last = Some(loss);
        self.ema = Some(match self.ema {
            None => loss,
            Some(prev) => self.ema_alpha * loss + (1.0 - self.ema_alpha) * prev,
        });
        self.count = self.count.saturating_add(1);
    }
}

/// Bundle of per-network loss trackers.
///
/// A section is `Some` iff the matching network is configured on the
/// brain — so a Phase 1 adaptive-only brain reports `adaptive = Some(...)`
/// and `lnn = None` regardless of whether training has happened yet.
/// Whether training *has* happened is conveyed by `count > 0` inside
/// the tracker.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct LossStats {
    /// Adaptive (MLP) training loss tracker.
    pub adaptive: Option<LossTracker>,
    /// LNN training loss tracker. `None` on brains built without an LNN.
    pub lnn: Option<LossTracker>,
}

// -------------------------------------------------------------------------
// Adaptive
// -------------------------------------------------------------------------

/// One layer's weight + bias distribution summary.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct LayerWeightStats {
    /// Layer index (0 = first weight matrix after the input layer).
    pub layer_idx: usize,
    /// Output features (rows of `w`).
    pub out_features: usize,
    /// Input features (cols of `w`).
    pub in_features: usize,
    /// Minimum weight.
    pub w_min: f32,
    /// Maximum weight.
    pub w_max: f32,
    /// Mean weight.
    pub w_mean: f32,
    /// Standard deviation of the weight distribution.
    pub w_std: f32,
    /// Minimum bias.
    pub b_min: f32,
    /// Maximum bias.
    pub b_max: f32,
    /// Mean bias.
    pub b_mean: f32,
    /// Bias standard deviation.
    pub b_std: f32,
}

/// Adaptive (MLP) subsystem stats.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveStats {
    /// Layer widths (input + hidden + output).
    pub layer_widths: Vec<usize>,
    /// Number of weight transitions (`layer_widths.len() - 1`).
    pub num_transitions: usize,
    /// Total learnable parameters (weights + biases).
    pub total_params: usize,
    /// Per-layer weight/bias distribution summaries.
    pub layers: Vec<LayerWeightStats>,
}

fn stats_of(xs: &[f32]) -> (f32, f32, f32, f32) {
    if xs.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0_f64;
    for &x in xs {
        if x < min {
            min = x;
        }
        if x > max {
            max = x;
        }
        sum += x as f64;
    }
    let mean = (sum / xs.len() as f64) as f32;
    let mut var_sum = 0.0_f64;
    for &x in xs {
        let d = (x - mean) as f64;
        var_sum += d * d;
    }
    let std = (var_sum / xs.len() as f64).sqrt() as f32;
    (min, max, mean, std)
}

fn layer_weight_stats(idx: usize, w: &LayerWeights) -> LayerWeightStats {
    let (rows, cols) = (w.w.shape()[0], w.w.shape()[1]);
    let flat: Vec<f32> = w.w.iter().copied().collect();
    let (w_min, w_max, w_mean, w_std) = stats_of(&flat);
    let b: Vec<f32> = w.b.iter().copied().collect();
    let (b_min, b_max, b_mean, b_std) = stats_of(&b);
    LayerWeightStats {
        layer_idx: idx,
        out_features: rows,
        in_features: cols,
        w_min,
        w_max,
        w_mean,
        w_std,
        b_min,
        b_max,
        b_mean,
        b_std,
    }
}

/// Gather stats from an [`AdaptiveNet`].
#[must_use]
pub fn collect_adaptive(net: &AdaptiveNet) -> AdaptiveStats {
    let cfg = net.config();
    let weights = net.weights();
    let layers: Vec<LayerWeightStats> = weights
        .iter()
        .enumerate()
        .map(|(i, w)| layer_weight_stats(i, w))
        .collect();
    let total_params: usize = layers
        .iter()
        .map(|l| l.out_features * l.in_features + l.out_features)
        .sum();
    AdaptiveStats {
        layer_widths: cfg.layers.clone(),
        num_transitions: net.num_transitions(),
        total_params,
        layers,
    }
}

// -------------------------------------------------------------------------
// SNN
// -------------------------------------------------------------------------

/// Per-population snapshot.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct PopulationStats {
    /// Population index.
    pub pop_idx: usize,
    /// Neuron count in this population.
    pub n_neurons: usize,
    /// Rate-EMA as tracked by the SNN's homeostatic system.
    pub rate_ema: f32,
    /// Non-zero entries in the current spike buffer (fraction spiking
    /// this step × `n_neurons`).
    pub spikes_this_step: usize,
}

/// Per-edge synapse stats.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct EdgeWeightStats {
    /// Edge index.
    pub edge_idx: usize,
    /// Number of synapses (length of the weight vector).
    pub n_synapses: usize,
    /// Minimum weight.
    pub w_min: f32,
    /// Maximum weight.
    pub w_max: f32,
    /// Mean weight.
    pub w_mean: f32,
    /// Standard deviation.
    pub w_std: f32,
}

/// SNN subsystem stats.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnnStats {
    /// Populations.
    pub populations: Vec<PopulationStats>,
    /// Directed edges.
    pub edges: Vec<EdgeWeightStats>,
    /// Total synapses across every edge.
    pub total_synapses: usize,
    /// Current simulated time (ms).
    pub t_ms: f32,
}

/// Gather stats from an [`SnnNetwork`].
#[must_use]
pub fn collect_snn(net: &SnnNetwork) -> SnnStats {
    let populations: Vec<PopulationStats> = (0..net.n_populations())
        .map(|i| {
            let spikes = net.spikes(i);
            PopulationStats {
                pop_idx: i,
                n_neurons: spikes.len(),
                rate_ema: net.rate_ema(i),
                spikes_this_step: spikes.iter().filter(|&&s| s != 0).count(),
            }
        })
        .collect();
    let edges: Vec<EdgeWeightStats> = (0..net.n_edges())
        .map(|i| {
            let w = net.edge_weights(i);
            let (w_min, w_max, w_mean, w_std) = stats_of(w);
            EdgeWeightStats {
                edge_idx: i,
                n_synapses: w.len(),
                w_min,
                w_max,
                w_mean,
                w_std,
            }
        })
        .collect();
    let total_synapses = edges.iter().map(|e| e.n_synapses).sum();
    SnnStats {
        populations,
        edges,
        total_synapses,
        t_ms: net.t_ms(),
    }
}

// -------------------------------------------------------------------------
// LNN
// -------------------------------------------------------------------------

/// Per-LTC-layer tau-base distribution summary + current state norm.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct LtcLayerStats {
    /// Layer index.
    pub layer_idx: usize,
    /// Hidden dimension (`n_rec`).
    pub n_rec: usize,
    /// Input dimension (`n_in`).
    pub n_in: usize,
    /// Minimum tau_base across the layer's neurons.
    pub tau_min: f32,
    /// Maximum tau_base.
    pub tau_max: f32,
    /// Mean tau_base.
    pub tau_mean: f32,
    /// Tau_base standard deviation.
    pub tau_std: f32,
    /// Current transient-state L2 norm.
    pub state_norm: f32,
}

/// LNN subsystem stats.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LnnStats {
    /// Network input dimension.
    pub input_dim: usize,
    /// Network output dimension.
    pub output_dim: usize,
    /// Integration timestep, ms.
    pub dt_ms: f32,
    /// Per-layer LTC stats.
    pub layers: Vec<LtcLayerStats>,
    /// Readout weight distribution summary.
    pub readout_w_min: f32,
    /// Readout weight max.
    pub readout_w_max: f32,
    /// Readout weight mean.
    pub readout_w_mean: f32,
    /// Readout weight std.
    pub readout_w_std: f32,
}

fn array1_norm(x: &Array1<f32>) -> f32 {
    let sumsq: f32 = x.iter().map(|v| v * v).sum();
    sumsq.sqrt()
}

/// Gather stats from an [`LnnNetwork`] + optional current state (if the
/// caller tracks one). `state` being `None` means we report zero
/// state-norms (the network has a fresh state vector in that case).
#[must_use]
pub fn collect_lnn(net: &LnnNetwork, state: Option<&[LtcState]>) -> LnnStats {
    let layers: Vec<LtcLayerStats> = net
        .layers
        .iter()
        .enumerate()
        .map(|(i, l)| {
            let taus: Vec<f32> = l.tau_base.iter().copied().collect();
            let (tau_min, tau_max, tau_mean, tau_std) = stats_of(&taus);
            let state_norm = state
                .and_then(|s| s.get(i))
                .map(|st| array1_norm(&st.x))
                .unwrap_or(0.0);
            LtcLayerStats {
                layer_idx: i,
                n_rec: l.params.n_rec,
                n_in: l.params.n_in,
                tau_min,
                tau_max,
                tau_mean,
                tau_std,
                state_norm,
            }
        })
        .collect();

    let w_out_flat: Vec<f32> = net.w_out.iter().copied().collect();
    let (readout_w_min, readout_w_max, readout_w_mean, readout_w_std) = stats_of(&w_out_flat);

    LnnStats {
        input_dim: net.input_dim,
        output_dim: net.output_dim,
        dt_ms: net.dt_ms,
        layers,
        readout_w_min,
        readout_w_max,
        readout_w_mean,
        readout_w_std,
    }
}

// -------------------------------------------------------------------------
// Memory
// -------------------------------------------------------------------------

/// Memory (Z-Ladder) subsystem stats.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    /// Nodes per tier (index 0 = Z0, …, 3 = Z3).
    pub tier_counts: [usize; 4],
    /// Total nodes across every tier.
    pub total_nodes: usize,
    /// Landmark count.
    pub landmark_count: usize,
    /// Cumulative per-transition promotion tallies (`Z0→Z1`, `Z1→Z2`, `Z2→Z3`).
    pub promotions: [u64; 3],
    /// Cumulative per-transition demotion tallies (`Z1→Z0`, `Z2→Z1`, `Z3→Z2`).
    pub demotions: [u64; 3],
    /// Cumulative per-tier evictions.
    pub evictions: [u64; 4],
    /// Consolidate-cycle counter.
    pub consolidate_cycles: u64,
    /// Simulated clock (ms).
    pub clock_ms: u64,
    /// Average node strength per tier.
    pub avg_strength_per_tier: [f32; 4],
    /// Landmark reason counts — how many landmarks carry each reason tag.
    pub landmark_reasons: HashMap<String, usize>,
}

/// Gather stats from a [`ZLadder`].
#[must_use]
pub fn collect_memory(mem: &ZLadder) -> MemoryStats {
    let ls = mem.stats();

    // Average strength per tier requires a pass over the nodes; the
    // ladder's `iter_nodes` gives us everything in one shot.
    let mut per_tier_sum = [0.0_f64; 4];
    let mut per_tier_count = [0_usize; 4];
    for n in mem.iter_nodes() {
        let idx = n.tier.index();
        per_tier_sum[idx] += n.strength as f64;
        per_tier_count[idx] += 1;
    }
    let mut avg_strength_per_tier = [0.0_f32; 4];
    for i in 0..4 {
        if per_tier_count[i] > 0 {
            avg_strength_per_tier[i] = (per_tier_sum[i] / per_tier_count[i] as f64) as f32;
        }
    }

    // Landmark reason histogram.
    let mut landmark_reasons: HashMap<String, usize> = HashMap::new();
    for (_id, reason) in mem.landmarks() {
        *landmark_reasons.entry(reason.to_string()).or_insert(0) += 1;
    }

    MemoryStats {
        tier_counts: ls.tier_counts,
        total_nodes: ls.total_nodes,
        landmark_count: ls.landmark_count,
        promotions: ls.promotions,
        demotions: ls.demotions,
        evictions: ls.evictions,
        consolidate_cycles: ls.consolidate_cycles,
        clock_ms: mem.clock_ms(),
        avg_strength_per_tier,
        landmark_reasons,
    }
}

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn stats_of_handles_empty() {
        assert_eq!(stats_of(&[]), (0.0, 0.0, 0.0, 0.0));
    }

    #[test]
    fn stats_of_single_value() {
        let (min, max, mean, std) = stats_of(&[3.0]);
        assert_eq!(min, 3.0);
        assert_eq!(max, 3.0);
        assert_eq!(mean, 3.0);
        assert_eq!(std, 0.0);
    }

    #[test]
    fn stats_of_known_distribution() {
        // mean(1..=5) = 3; var = ((1-3)^2 + (2-3)^2 + 0 + 1 + 4)/5 = 10/5 = 2; std = sqrt(2).
        let (min, max, mean, std) = stats_of(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(min, 1.0);
        assert_eq!(max, 5.0);
        assert_eq!(mean, 3.0);
        assert!((std - 2.0_f32.sqrt()).abs() < 1e-6);
    }

    #[test]
    fn loss_tracker_starts_empty() {
        let t = LossTracker::default();
        assert!(t.last.is_none());
        assert!(t.ema.is_none());
        assert_eq!(t.count, 0);
        assert_eq!(t.ema_alpha, DEFAULT_LOSS_EMA_ALPHA);
    }

    #[test]
    fn loss_tracker_first_observation_seeds_ema() {
        let mut t = LossTracker::default();
        t.observe(0.5);
        assert_eq!(t.last, Some(0.5));
        // No cold-start bias — first EMA == first observation.
        assert_eq!(t.ema, Some(0.5));
        assert_eq!(t.count, 1);
    }

    #[test]
    fn loss_tracker_ema_lags_last() {
        let mut t = LossTracker::with_alpha(0.1);
        t.observe(1.0);
        t.observe(0.0);
        // ema = 0.1 * 0.0 + 0.9 * 1.0 = 0.9; last = 0.0.
        assert_eq!(t.last, Some(0.0));
        assert!((t.ema.unwrap() - 0.9).abs() < 1e-6);
        assert_eq!(t.count, 2);
    }

    #[test]
    fn loss_tracker_alpha_one_is_no_smoothing() {
        let mut t = LossTracker::with_alpha(1.0);
        t.observe(2.0);
        t.observe(7.0);
        // α = 1 → ema tracks last exactly.
        assert_eq!(t.last, t.ema);
        assert_eq!(t.last, Some(7.0));
    }

    #[test]
    #[should_panic(expected = "ema_alpha must be in (0, 1]")]
    fn loss_tracker_alpha_zero_rejected() {
        let _ = LossTracker::with_alpha(0.0);
    }

    #[test]
    #[should_panic(expected = "ema_alpha must be in (0, 1]")]
    fn loss_tracker_alpha_above_one_rejected() {
        let _ = LossTracker::with_alpha(1.5);
    }
}
