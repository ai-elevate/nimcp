//! NIMCP V2 — top-level brain integration.
//!
//! This crate composes the smaller crates into a runnable `Brain`. It owns:
//!
//! - The scheduler (hosts all actors)
//! - The event log (the source of truth)
//! - A collection of network actors (adaptive / SNN / LNN)
//! - The memory actor (Z-Ladder)
//! - The checkpoint coordinator
//!
//! **No 800-field struct.** A `Brain` is a small handle that routes requests
//! to the right actor; each actor owns its own state.
//!
//! # Phase 1 scope
//!
//! The Brain composes an [`AdaptiveNet`] (MLP) for CPU forward + backward +
//! SGD. That's the only compute path wired here so far. SNN / LNN / memory
//! land in later phases.

#![forbid(unsafe_code)]

use std::path::Path;

use ndarray::Array1;
use nimcp_adaptive::{AdaptiveConfig, AdaptiveError, AdaptiveNet};
use nimcp_core::{Error, Result};
use nimcp_scheduler::{Scheduler, SchedulerConfig};
use serde::{Deserialize, Serialize};

/// Top-level brain configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainConfig {
    /// Seed for deterministic init.
    pub rng_seed: u64,
    /// Whether to run in deterministic (single-threaded, virtual time) mode.
    pub deterministic: bool,
    /// Path where the event log + checkpoints live.
    pub state_dir: std::path::PathBuf,
    /// Adaptive network config — the MLP the brain trains against.
    pub adaptive: AdaptiveConfig,
}

impl Default for BrainConfig {
    fn default() -> Self {
        Self {
            rng_seed: 0x5EED,
            deterministic: false,
            state_dir: std::path::PathBuf::from("./nimcp-state"),
            adaptive: AdaptiveConfig::default(),
        }
    }
}

/// The top-level brain handle.
pub struct Brain {
    config: BrainConfig,
    #[allow(dead_code)] // scheduler wires up in later phases
    scheduler: Scheduler,
    adaptive: AdaptiveNet,
}

impl Brain {
    /// Boot a new brain with the given config.
    pub fn new(config: BrainConfig) -> Result<Self> {
        let sched_cfg = SchedulerConfig {
            deterministic: config.deterministic,
            mailbox_capacity: 1024,
            rng_seed: config.rng_seed,
            ..SchedulerConfig::default()
        };
        let scheduler = Scheduler::new(sched_cfg);

        // Propagate the brain's rng_seed into adaptive unless the caller
        // overrode it explicitly. Same seed → same init, bit-for-bit.
        let mut adaptive_cfg = config.adaptive.clone();
        if adaptive_cfg.rng_seed == AdaptiveConfig::default().rng_seed {
            adaptive_cfg.rng_seed = config.rng_seed;
        }
        let adaptive = AdaptiveNet::new(adaptive_cfg);

        tracing::info!(
            layers = ?config.adaptive.layers,
            seed = config.rng_seed,
            "brain created"
        );
        Ok(Self {
            config,
            scheduler,
            adaptive,
        })
    }

    /// Accessor for the config.
    pub fn config(&self) -> &BrainConfig {
        &self.config
    }

    /// One training step against an MSE target. Returns pre-update loss.
    ///
    /// # Panics
    /// Panics if `features.len()` or `target.len()` don't match the first
    /// or last configured layer width. Callers that can't guarantee shape
    /// should validate before calling — the bindings layer does this.
    pub fn learn(&mut self, features: &Array1<f32>, target: &Array1<f32>, lr: f32) -> f32 {
        self.adaptive.learn(features, target, lr)
    }

    /// Forward pass. Returns the brain's output vector.
    pub fn predict(&self, features: &Array1<f32>) -> Array1<f32> {
        self.adaptive.forward(features)
    }

    /// Persist the brain's weights to `path`. Phase 1 only saves the
    /// adaptive net; later phases extend via CheckpointCoordinator.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let bytes = self.adaptive.save().map_err(adaptive_to_core_err)?;
        std::fs::write(path.as_ref(), bytes).map_err(Error::from)
    }

    /// Reload weights from a previous [`Brain::save`]. Shape is inferred
    /// from disk; the config's `layers` must match (shape-mismatched
    /// loads are a `Error::Config`).
    pub fn load<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let bytes = std::fs::read(path.as_ref()).map_err(Error::from)?;
        self.adaptive.load(&bytes).map_err(adaptive_to_core_err)?;
        Ok(())
    }
}

fn adaptive_to_core_err(e: AdaptiveError) -> Error {
    match e {
        AdaptiveError::ShapeMismatch { expected, got } => Error::Config(format!(
            "adaptive shape mismatch: expected {expected}, got {got}"
        )),
        AdaptiveError::Serialization(msg) => Error::Serialization(msg),
        AdaptiveError::Checkpoint(msg) => Error::Config(format!("checkpoint: {msg}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[tokio::test]
    async fn brain_boots_with_default_config() {
        let brain = Brain::new(BrainConfig::default()).unwrap();
        assert_eq!(brain.config().rng_seed, 0x5EED);
    }

    /// End-to-end XOR inside the Brain API — this is the Phase 1 exit
    /// criterion: 100-neuron toy brain trains on XOR in <5s.
    #[tokio::test]
    async fn brain_trains_xor_end_to_end() {
        let cfg = BrainConfig {
            rng_seed: 0x42,
            deterministic: true,
            adaptive: AdaptiveConfig {
                layers: vec![2, 16, 1],
                rng_seed: 0x42,
                activation: nimcp_adaptive::Activation::Tanh,
            },
            ..Default::default()
        };
        let mut brain = Brain::new(cfg).unwrap();

        let samples: [(Array1<f32>, Array1<f32>); 4] = [
            (
                Array1::from_vec(vec![0.0, 0.0]),
                Array1::from_vec(vec![0.0]),
            ),
            (
                Array1::from_vec(vec![0.0, 1.0]),
                Array1::from_vec(vec![1.0]),
            ),
            (
                Array1::from_vec(vec![1.0, 0.0]),
                Array1::from_vec(vec![1.0]),
            ),
            (
                Array1::from_vec(vec![1.0, 1.0]),
                Array1::from_vec(vec![0.0]),
            ),
        ];

        let start = std::time::Instant::now();
        let mut final_loss = f32::INFINITY;
        for _step in 0..5000 {
            let mut mean = 0.0;
            for (x, y) in &samples {
                mean += brain.learn(x, y, 0.1);
            }
            final_loss = mean / 4.0;
            if final_loss < 0.05 {
                break;
            }
        }
        let elapsed = start.elapsed();
        assert!(
            final_loss < 0.05,
            "XOR didn't converge: final_loss={final_loss}"
        );
        assert!(elapsed.as_secs() < 5, "XOR took too long: {:?}", elapsed);

        // Prediction sanity: (1, 0) should be close to 1; (1, 1) close to 0.
        let p10 = brain.predict(&Array1::from_vec(vec![1.0, 0.0]))[0];
        let p11 = brain.predict(&Array1::from_vec(vec![1.0, 1.0]))[0];
        assert!(p10 > 0.5, "predict(1,0)={p10}, expected >0.5");
        assert!(p11 < 0.5, "predict(1,1)={p11}, expected <0.5");
    }

    #[tokio::test]
    async fn save_load_round_trip() {
        let cfg = BrainConfig {
            adaptive: AdaptiveConfig {
                layers: vec![3, 5, 2],
                rng_seed: 7,
                activation: nimcp_adaptive::Activation::Relu,
            },
            ..Default::default()
        };
        let mut a = Brain::new(cfg.clone()).unwrap();
        let x = Array1::from_vec(vec![1.0, -0.5, 0.25]);

        // Train for a few steps so a != b initially.
        for _ in 0..20 {
            a.learn(&x, &Array1::from_vec(vec![0.0, 1.0]), 0.05);
        }
        let y_a = a.predict(&x);

        let tmp = tempfile::NamedTempFile::new().unwrap();
        a.save(tmp.path()).unwrap();

        let mut b = Brain::new(cfg).unwrap();
        b.load(tmp.path()).unwrap();
        let y_b = b.predict(&x);

        for (pa, pb) in y_a.iter().zip(y_b.iter()) {
            assert!((pa - pb).abs() < 1e-6, "save/load drift: {pa} vs {pb}");
        }
    }
}
