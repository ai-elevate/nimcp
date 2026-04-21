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

#![forbid(unsafe_code)]
#![allow(dead_code)]

use nimcp_core::Result;
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
}

impl Default for BrainConfig {
    fn default() -> Self {
        Self {
            rng_seed: 0x5EED,
            deterministic: false,
            state_dir: std::path::PathBuf::from("./nimcp-state"),
        }
    }
}

/// The top-level brain handle.
pub struct Brain {
    config: BrainConfig,
    scheduler: Scheduler,
}

impl Brain {
    /// Boot a new brain with the given config. Phase 0 stub: only spins up
    /// the scheduler. Networks wire up in later phases.
    pub fn new(config: BrainConfig) -> Result<Self> {
        let sched_cfg = SchedulerConfig {
            deterministic: config.deterministic,
            mailbox_capacity: 1024,
            rng_seed: config.rng_seed,
        };
        let scheduler = Scheduler::new(sched_cfg);

        tracing::info!(?config, "brain created");
        Ok(Self { config, scheduler })
    }

    /// Accessor for the config.
    pub fn config(&self) -> &BrainConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn brain_boots_with_default_config() {
        let brain = Brain::new(BrainConfig::default()).unwrap();
        assert_eq!(brain.config().rng_seed, 0x5EED);
    }
}
