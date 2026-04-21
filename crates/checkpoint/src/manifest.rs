//! On-disk manifest format.
//!
//! The manifest is intentionally JSON rather than rkyv: it is small,
//! human-auditable, and the one part of a checkpoint that a human or an
//! external tool may want to inspect without the Rust code that wrote it.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::error::CheckpointError;

/// Filename of the manifest within a checkpoint directory.
pub const MANIFEST_FILE: &str = "manifest.json";

/// Suffix appended to registered names to form sidecar filenames.
pub const SIDECAR_SUFFIX: &str = ".sidecar";

/// Format version of the manifest itself (not of any individual sidecar).
///
/// This number only changes if the *manifest schema* changes — adding a
/// field, renaming one, etc. Sidecar schema changes do NOT bump this.
pub const MANIFEST_VERSION: u32 = 1;

/// One entry in the manifest for each registered `Checkpointable`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SidecarEntry {
    /// Registered name. Used as the sidecar filename stem.
    pub name: String,
    /// Schema version the sidecar was written at.
    pub version: u32,
    /// Byte length of the sidecar body on disk. Used as a sanity check
    /// at load time; mismatch indicates truncation or tampering.
    pub bytes: u64,
}

/// Root of the manifest JSON document.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Manifest {
    /// Manifest format version (see [`MANIFEST_VERSION`]).
    pub nimcp_checkpoint_version: u32,
    /// ISO-8601 UTC timestamp of when the save completed (roughly —
    /// generated just before the rename). Purely informational.
    pub saved_at: String,
    /// One entry per sidecar on disk.
    pub sidecars: Vec<SidecarEntry>,
}

impl Manifest {
    /// Serialize to pretty JSON (easy to diff / read).
    pub fn to_bytes(&self) -> Result<Vec<u8>, CheckpointError> {
        serde_json::to_vec_pretty(self)
            .map_err(|e| CheckpointError::Serialization(format!("manifest encode: {e}")))
    }

    /// Parse manifest bytes, attributing errors to `path` for diagnostics.
    pub fn from_bytes(bytes: &[u8], path: PathBuf) -> Result<Self, CheckpointError> {
        let mf: Manifest =
            serde_json::from_slice(bytes).map_err(|e| CheckpointError::MalformedManifest {
                path: path.clone(),
                reason: e.to_string(),
            })?;
        if mf.nimcp_checkpoint_version == 0 {
            return Err(CheckpointError::MalformedManifest {
                path,
                reason: "nimcp_checkpoint_version must be >= 1".to_string(),
            });
        }
        if mf.nimcp_checkpoint_version > MANIFEST_VERSION {
            return Err(CheckpointError::MalformedManifest {
                path,
                reason: format!(
                    "manifest version {found} is newer than this build supports ({supported})",
                    found = mf.nimcp_checkpoint_version,
                    supported = MANIFEST_VERSION,
                ),
            });
        }
        Ok(mf)
    }
}
