//! The [`Checkpointable`] trait.

use crate::error::CheckpointError;

/// Every stateful actor that wants to persist across restarts implements
/// this trait.
///
/// Implementations are free to serialize however they like — rkyv,
/// bincode, hand-rolled bytes — as long as `load` accepts any byte buffer
/// previously produced by `save` for the same version.
///
/// # Versioning
///
/// Bump [`VERSION`](Self::VERSION) whenever the on-wire layout changes in
/// a way that can't be round-tripped by the old code. Register a
/// migration with
/// [`CheckpointCoordinator::register_migration`](crate::CheckpointCoordinator::register_migration)
/// so old sidecars can still be loaded.
///
/// # Naming
///
/// [`NAME`](Self::NAME) is a type-wide default; at registration time the
/// caller may override it — useful when you register several instances of
/// the same type (e.g. two independent SNN populations).
pub trait Checkpointable: Send {
    /// Stable default name for sidecar files of this type. Must be a
    /// valid filename fragment (no `/`, no null bytes, not `.` / `..`).
    const NAME: &'static str;

    /// Schema version for [`save`](Self::save)'s output. Bump on breaking
    /// layout changes.
    const VERSION: u32;

    /// Serialize current state to an opaque byte buffer.
    ///
    /// The returned bytes become the body of a sidecar file. They should
    /// fully describe the object's state at the moment of the call; the
    /// coordinator will not re-enter the object to fetch anything extra.
    fn save(&self) -> Result<Vec<u8>, CheckpointError>;

    /// Replace current state with whatever `bytes` describe.
    ///
    /// `version` is the schema version of `bytes`. It is either
    /// [`Self::VERSION`] (fresh save) or has already been migrated up to
    /// `Self::VERSION` by the coordinator before this call — implementations
    /// may therefore assume `version == Self::VERSION` in nearly all cases,
    /// but MUST validate the assumption and return a clear error if not.
    fn load(&mut self, bytes: &[u8], version: u32) -> Result<(), CheckpointError>;
}
