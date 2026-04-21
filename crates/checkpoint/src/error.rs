//! Errors surfaced by the checkpoint crate.

use std::io;
use std::path::PathBuf;

use nimcp_core::Error as CoreError;
use thiserror::Error;

/// Error returned by [`Checkpointable`](crate::Checkpointable) methods and
/// by the [`CheckpointCoordinator`](crate::CheckpointCoordinator).
///
/// Converts into [`nimcp_core::Error`] via `From` so domain crates can
/// bubble it up through their own error enums.
#[derive(Debug, Error)]
pub enum CheckpointError {
    /// Underlying filesystem I/O failed.
    #[error("io: {0}")]
    Io(#[from] io::Error),

    /// A sidecar blob could not be (de)serialized.
    #[error("serialization: {0}")]
    Serialization(String),

    /// The manifest on disk is not valid JSON or is missing required fields.
    #[error("malformed manifest at {path}: {reason}")]
    MalformedManifest {
        /// Path to the manifest file that failed to parse.
        path: PathBuf,
        /// Human-readable reason the parse failed.
        reason: String,
    },

    /// The manifest references a sidecar that isn't on disk.
    #[error("sidecar `{name}` referenced by manifest is missing on disk")]
    MissingSidecar {
        /// Registered name of the missing sidecar.
        name: String,
    },

    /// An on-disk sidecar carries a schema version newer than this build
    /// knows how to load. The caller is running old code against new data.
    #[error(
        "sidecar `{name}` has version {found} but current code only understands up to {current}; \
         refusing to load to avoid silent data loss — upgrade the binary"
    )]
    UnknownFutureVersion {
        /// Registered name.
        name: String,
        /// Version recorded in the manifest.
        found: u32,
        /// Current version declared by the `Checkpointable` impl.
        current: u32,
    },

    /// No migration path exists to bring an older sidecar up to the current version.
    #[error(
        "cannot migrate sidecar `{name}` from version {from} to version {to}: \
         no migration registered for step {stuck_at} -> {next}"
    )]
    MissingMigration {
        /// Registered name.
        name: String,
        /// Starting version on disk.
        from: u32,
        /// Target version (current).
        to: u32,
        /// Version that had no outgoing migration.
        stuck_at: u32,
        /// Version we could not reach.
        next: u32,
    },

    /// A migration function returned an error.
    #[error("migration `{name}` {from}->{to} failed: {reason}")]
    MigrationFailed {
        /// Registered name.
        name: String,
        /// Version the migration stepped from.
        from: u32,
        /// Version the migration stepped to.
        to: u32,
        /// Error message from the migration function.
        reason: String,
    },

    /// The caller tried to save or load an object that was never registered.
    #[error("no object registered under name `{0}`")]
    NotRegistered(String),

    /// Two registrations collided on the same name.
    #[error("name `{0}` is already registered")]
    DuplicateName(String),

    /// The coordinator's internal lock was poisoned by a panicking thread.
    #[error("internal lock poisoned: {0}")]
    Poisoned(String),

    /// Catch-all for caller-supplied `Checkpointable` impl errors that don't
    /// fit anywhere more specific.
    #[error("{0}")]
    Other(String),
}

impl From<CheckpointError> for CoreError {
    fn from(err: CheckpointError) -> Self {
        match err {
            CheckpointError::Io(e) => CoreError::Io(e),
            CheckpointError::Serialization(s) => CoreError::Serialization(s),
            other => CoreError::Invariant(other.to_string()),
        }
    }
}
