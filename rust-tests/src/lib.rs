//! Workspace-level integration tests for NIMCP V2.
//!
//! Unit tests live with their crates. This crate exercises interactions
//! **across** crates that neither owns. Examples:
//!
//! - Event flows from the scheduler → event log → checkpoint replay.
//! - Adaptive network trains, saves, loads into fresh net, predictions match.
//! - Plasticity rules applied to network weights produce expected firing.
//!
//! # Phase 0 status
//!
//! This is a stub: it only verifies that the top-level `Brain` boots and
//! the workspace links correctly. Real integration tests land as the
//! per-crate implementations come online (Phase 1+).

#![forbid(unsafe_code)]

#[cfg(test)]
mod tests {
    use nimcp_brain::{Brain, BrainConfig};

    #[tokio::test]
    async fn brain_boots_with_default_config() {
        let brain = Brain::new(BrainConfig::default()).expect("brain boots");
        assert_eq!(brain.config().rng_seed, 0x5EED);
    }

    #[tokio::test]
    async fn brain_boots_deterministic() {
        let cfg = BrainConfig {
            deterministic: true,
            ..Default::default()
        };
        let brain = Brain::new(cfg).expect("deterministic brain boots");
        assert!(brain.config().deterministic);
    }
}
