//! NIMCP V2 — checkpoint save/load.
//!
//! V1 had scattered checkpoint logic across ~10 subsystems, each with
//! its own sidecar file, producing confusing .snn/.cnn/.lnn/.cortex_*
//! artifacts. V2 centralizes: every stateful actor exposes a
//! `Checkpointable` trait; a single `CheckpointCoordinator` drives
//! save + load under atomic-rename semantics.
//!
//! # Format
//!
//! rkyv-serialized, versioned, schema-evolution-safe. Version bumps
//! ship with an explicit migration function.
//!
//! # Atomic save
//!
//! Write to `foo.tmp` → fsync → rename `foo.tmp` → `foo`. Crash during
//! write leaves `foo` untouched; caller reverts to the previous save.

#![forbid(unsafe_code)]
#![allow(dead_code)]

/// Trait every checkpointable actor implements.
pub trait Checkpointable {
    /// Serialize current state to bytes.
    fn save(&self) -> Vec<u8>;
    /// Restore from bytes saved by an earlier `save()`.
    fn load(&mut self, bytes: &[u8]) -> Result<(), String>;
    /// Stable name for the sidecar file / log entry.
    fn checkpoint_name(&self) -> &'static str;
}
