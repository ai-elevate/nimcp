//! NIMCP V2 — checkpoint save/load.
//!
//! V1 had scattered checkpoint logic across ~10 subsystems, each with
//! its own sidecar file, producing confusing `.snn` / `.cnn` / `.lnn` /
//! `.cortex_*` artifacts. V2 centralizes: every stateful actor exposes a
//! [`Checkpointable`] trait; a single [`CheckpointCoordinator`] drives
//! save + load under atomic-rename semantics.
//!
//! # Layout on disk
//!
//! ```text
//! <dir>/
//!   manifest.json            # human-readable index
//!   <name>.sidecar           # one opaque blob per registered object
//!   <name>.sidecar
//!   ...
//! ```
//!
//! During a save in progress, files land in `<dir>/.tmp/` and are
//! rename-promoted into `<dir>/` once all sidecars have been written.
//!
//! # Versioning + migration
//!
//! Every sidecar carries a schema version. When [`CheckpointCoordinator::load_all`]
//! finds a sidecar whose recorded version is below the type's current
//! [`Checkpointable::VERSION`], it applies any migrations the caller has
//! registered with [`CheckpointCoordinator::register_migration`] in order
//! (`from_version` → `from_version + 1` → … → current) before handing the
//! final bytes to [`Checkpointable::load`]. A sidecar whose recorded
//! version is **above** the current `VERSION` is a hard error — the caller
//! is running older code than the data.
//!
//! # Atomicity
//!
//! Writes go to `<dir>/.tmp/` first; only once every sidecar + the
//! manifest have been written do we promote them. A crash partway through
//! leaves the previous committed checkpoint in `<dir>/` untouched, and
//! leaves stale files in `<dir>/.tmp/`. [`CheckpointCoordinator::load_all`]
//! treats a lingering `.tmp/` as evidence of a prior aborted save: it
//! warns and removes it.
//!
//! # Scope
//!
//! This crate is the mechanism. It does not implement `Checkpointable` for
//! any domain type — that is each domain crate's responsibility (SNN,
//! LNN, memory, etc.). See each crate's docs for its concrete impl.

#![forbid(unsafe_code)]

mod coordinator;
mod error;
mod manifest;
mod traits;

pub use coordinator::{CheckpointCoordinator, MigrationFn, read_sidecar_file};
pub use error::CheckpointError;
pub use manifest::{MANIFEST_FILE, MANIFEST_VERSION, Manifest, SIDECAR_SUFFIX, SidecarEntry};
pub use traits::Checkpointable;
