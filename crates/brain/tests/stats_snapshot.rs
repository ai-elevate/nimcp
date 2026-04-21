//! Phase 6 — `brain.stats()` coverage.
//!
//! Exercises the introspection surface against every subsystem
//! combination:
//!
//! - adaptive-only (Phase 1 brain)
//! - adaptive + SNN + LNN + memory (full V2 ensemble)
//!
//! The V2_PLAN Phase 6 exit criterion is "`brain.stats()` returns a
//! comprehensive, documented dict" — the Rust side exposes a typed
//! [`BrainStats`] plus a `stats_json()` companion; the Python binding
//! (Phase 6b) turns the JSON into a dict.

use nimcp_adaptive::{Activation, AdaptiveConfig};
use nimcp_brain::stats::BrainStats;
use nimcp_brain::{Brain, BrainConfig};
use nimcp_lnn::{LnnConfig, LtcParams};
use nimcp_memory::{MemoryNode, Tier, ZLadderConfig};
use nimcp_plasticity::HomeostaticParams;
use nimcp_snn::network::{EdgeSpec, PopulationSpec};
use nimcp_snn::{LifParams, RstdpParams, SnnConfig};

fn full_ensemble_cfg(seed: u64) -> BrainConfig {
    let adaptive = AdaptiveConfig {
        layers: vec![3, 8, 2],
        rng_seed: seed,
        activation: Activation::Tanh,
    };

    let snn = Some(SnnConfig {
        populations: vec![
            PopulationSpec {
                name: "in".into(),
                n_neurons: 16,
                lif: LifParams::default(),
                target_rate: 0.1,
                homeostatic: HomeostaticParams::default(),
            },
            PopulationSpec {
                name: "out".into(),
                n_neurons: 16,
                lif: LifParams::default(),
                target_rate: 0.1,
                homeostatic: HomeostaticParams::default(),
            },
        ],
        edges: vec![EdgeSpec {
            src: 0,
            dst: 1,
            fan_in: 4,
            weight_init: 1.0,
            weight_jitter: 0.2,
            rstdp: RstdpParams {
                warmup_samples: 0,
                w_max: 5.0,
                ..RstdpParams::default()
            },
        }],
        rng_seed: seed + 1,
        rate_ema_alpha: 0.05,
    });

    let lnn = Some(LnnConfig {
        input_dim: 3,
        output_dim: 1,
        layers: vec![LtcParams {
            n_in: 3,
            n_rec: 8,
            tau_init: 1.0,
            init_scale: 1.0,
        }],
        rng_seed: seed + 2,
        dt_ms: 0.1,
    });

    BrainConfig {
        rng_seed: seed,
        deterministic: true,
        adaptive,
        snn,
        lnn,
        memory: Some(ZLadderConfig::default()),
        ..Default::default()
    }
}

#[tokio::test]
async fn stats_on_adaptive_only_brain_has_adaptive_section() {
    let cfg = BrainConfig {
        adaptive: AdaptiveConfig {
            layers: vec![4, 8, 3],
            rng_seed: 1,
            activation: Activation::Tanh,
        },
        ..Default::default()
    };
    let brain = Brain::new(cfg).unwrap();
    let stats = brain.stats();

    let ad = stats.adaptive.as_ref().expect("adaptive section");
    assert_eq!(ad.layer_widths, vec![4, 8, 3]);
    assert_eq!(ad.num_transitions, 2);
    assert_eq!(ad.layers.len(), 2);
    // Params: 4*8+8 + 8*3+3 = 40 + 27 = 67
    assert_eq!(ad.total_params, 4 * 8 + 8 + 8 * 3 + 3);
    // Layer shapes line up.
    assert_eq!(ad.layers[0].in_features, 4);
    assert_eq!(ad.layers[0].out_features, 8);
    assert_eq!(ad.layers[1].in_features, 8);
    assert_eq!(ad.layers[1].out_features, 3);

    // Every other section is absent.
    assert!(stats.snn.is_none());
    assert!(stats.lnn.is_none());
    assert!(stats.memory.is_none());
}

#[tokio::test]
async fn stats_on_full_ensemble_covers_every_subsystem() {
    let mut brain = Brain::new(full_ensemble_cfg(42)).unwrap();

    // Move state around so the snapshot isn't just the init defaults.
    brain
        .memory_insert({
            let mut n = MemoryNode::new(1, vec![0.1, 0.2, 0.3], 0);
            n.tier = Tier::Z0;
            n
        })
        .unwrap();
    brain
        .memory_insert({
            let mut n = MemoryNode::new(2, vec![0.4, 0.5, 0.6], 0);
            n.tier = Tier::Z3;
            n
        })
        .unwrap();
    brain.memory_mark_landmark(2, "bootstrap").unwrap();

    // Step the SNN once so the t_ms field isn't zero.
    let drive: Vec<f32> = vec![200.0; 16];
    let empty: Vec<f32> = Vec::new();
    let slices: Vec<&[f32]> = vec![&drive, &empty];
    brain.snn_step(&slices, 0.0, 1.0).unwrap();

    let stats = brain.stats();
    assert_eq!(stats.rng_seed, 42);

    // Adaptive.
    let ad = stats.adaptive.as_ref().unwrap();
    assert_eq!(ad.layer_widths, vec![3, 8, 2]);
    assert_eq!(ad.layers.len(), 2);
    for l in &ad.layers {
        assert!(l.w_std >= 0.0, "w_std must be non-negative");
        assert!(l.w_min <= l.w_max);
    }

    // SNN.
    let sn = stats.snn.as_ref().unwrap();
    assert_eq!(sn.populations.len(), 2);
    assert_eq!(sn.populations[0].n_neurons, 16);
    assert_eq!(sn.edges.len(), 1);
    assert_eq!(sn.total_synapses, sn.edges[0].n_synapses);
    assert!(sn.t_ms > 0.0, "t_ms should have advanced");

    // LNN.
    let ln = stats.lnn.as_ref().unwrap();
    assert_eq!(ln.input_dim, 3);
    assert_eq!(ln.output_dim, 1);
    assert_eq!(ln.layers.len(), 1);
    assert_eq!(ln.layers[0].n_rec, 8);
    assert_eq!(ln.layers[0].n_in, 3);
    // tau_init was 1.0 for every LTC neuron.
    assert!((ln.layers[0].tau_mean - 1.0).abs() < 1e-6);

    // Memory.
    let m = stats.memory.as_ref().unwrap();
    assert_eq!(m.tier_counts[0], 1, "node 1 at Z0");
    assert_eq!(m.tier_counts[3], 1, "node 2 at Z3 (landmark)");
    assert_eq!(m.total_nodes, 2);
    assert_eq!(m.landmark_count, 1);
    assert_eq!(m.landmark_reasons.get("bootstrap"), Some(&1));
}

#[tokio::test]
async fn stats_json_round_trips() {
    let brain = Brain::new(full_ensemble_cfg(7)).unwrap();
    let json = brain.stats_json().expect("stats_json");
    let restored: BrainStats = serde_json::from_str(&json).expect("round-trip");

    // Structural match on the top-level `Some` presence.
    let live = brain.stats();
    assert_eq!(restored.rng_seed, live.rng_seed);
    assert_eq!(restored.adaptive.is_some(), live.adaptive.is_some());
    assert_eq!(restored.snn.is_some(), live.snn.is_some());
    assert_eq!(restored.lnn.is_some(), live.lnn.is_some());
    assert_eq!(restored.memory.is_some(), live.memory.is_some());

    // One deep value, to prove the full tree made it through.
    assert_eq!(
        restored.adaptive.as_ref().unwrap().total_params,
        live.adaptive.as_ref().unwrap().total_params,
    );
}

#[tokio::test]
async fn stats_is_stable_across_back_to_back_calls() {
    // Pure read → two back-to-back calls on an untouched brain return
    // identical snapshots. The test guards against anyone stapling a
    // side effect (access recording, clock advance, etc.) into
    // `stats()`.
    let brain = Brain::new(full_ensemble_cfg(11)).unwrap();
    let a = brain.stats_json().unwrap();
    let b = brain.stats_json().unwrap();
    assert_eq!(a, b);
}
