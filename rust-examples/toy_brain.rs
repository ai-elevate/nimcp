//! Toy brain example — Phase 0 stub.
//!
//! In Phase 1 this becomes a 100-neuron MLP learning XOR in <5 seconds.
//! Right now it just boots a Brain to prove the workspace compiles + runs.

use nimcp_brain::{Brain, BrainConfig};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let brain = Brain::new(BrainConfig::default()).map_err(|e| anyhow::anyhow!("{e}"))?;
    tracing::info!(cfg = ?brain.config(), "brain up");
    Ok(())
}
