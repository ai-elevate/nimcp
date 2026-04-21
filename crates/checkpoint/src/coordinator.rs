//! The [`CheckpointCoordinator`] — central save/load for everything.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::{Read, Write};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use tracing::{info, warn};

use crate::error::CheckpointError;
use crate::manifest::{MANIFEST_FILE, MANIFEST_VERSION, Manifest, SIDECAR_SUFFIX, SidecarEntry};
use crate::traits::Checkpointable;

/// Directory name (relative to the checkpoint root) where in-flight save
/// artifacts live before being promoted.
const TMP_DIR: &str = ".tmp";

/// A migration function from version `from` → `from + 1`, applied to the
/// raw sidecar bytes as recorded on disk.
pub type MigrationFn = Arc<dyn Fn(&[u8]) -> Result<Vec<u8>, String> + Send + Sync>;

/// Object-safe dynamic façade over [`Checkpointable`]. Keeps the
/// associated consts reachable through accessor methods so the coordinator
/// can store heterogeneous objects in one `HashMap`.
trait CheckpointableDyn: Send {
    fn current_version(&self) -> u32;
    fn save_bytes(&self) -> Result<Vec<u8>, CheckpointError>;
    fn load_bytes(&mut self, bytes: &[u8], version: u32) -> Result<(), CheckpointError>;
}

impl<T: Checkpointable> CheckpointableDyn for T {
    fn current_version(&self) -> u32 {
        T::VERSION
    }

    fn save_bytes(&self) -> Result<Vec<u8>, CheckpointError> {
        <T as Checkpointable>::save(self)
    }

    fn load_bytes(&mut self, bytes: &[u8], version: u32) -> Result<(), CheckpointError> {
        <T as Checkpointable>::load(self, bytes, version)
    }
}

/// Internal record of one registered object.
struct Registration {
    /// Type-erased handle. `Arc<Mutex<_>>` because:
    /// - `Arc`: the caller wants to keep its own reference to the live
    ///   actor; the coordinator is only a co-owner.
    /// - `Mutex`: save + load touch object state; we need exclusive
    ///   access for the duration of each operation. Locks are held for
    ///   the single object only — never across the whole save_all.
    obj: Arc<Mutex<dyn CheckpointableDyn>>,
}

/// Central save/load across every registered [`Checkpointable`].
///
/// One `CheckpointCoordinator` per brain. Not intended to be used
/// concurrently across threads for the same operation — callers should
/// drive save/load from a single checkpoint-owning task. Concurrent
/// `register` and reads of registration state are protected internally.
pub struct CheckpointCoordinator {
    /// Registered objects keyed by their (user-chosen) sidecar name.
    /// `BTreeMap` for deterministic iteration order: save_all should
    /// produce bit-identical manifests given identical input.
    registry: Mutex<BTreeMap<String, Registration>>,

    /// Migration functions keyed by `(name, from_version)`. Each
    /// function advances bytes from `from_version` to `from_version + 1`.
    /// Chain them together for multi-step migrations.
    migrations: Mutex<HashMap<(String, u32), MigrationFn>>,
}

impl Default for CheckpointCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for CheckpointCoordinator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let registry_names = self
            .registry
            .lock()
            .map(|r| r.keys().cloned().collect::<Vec<_>>())
            .unwrap_or_default();
        let migration_count = self.migrations.lock().map(|m| m.len()).unwrap_or(0);
        f.debug_struct("CheckpointCoordinator")
            .field("registered", &registry_names)
            .field("migrations", &migration_count)
            .finish()
    }
}

impl CheckpointCoordinator {
    /// Create an empty coordinator with no registrations.
    pub fn new() -> Self {
        Self {
            registry: Mutex::new(BTreeMap::new()),
            migrations: Mutex::new(HashMap::new()),
        }
    }

    /// Register an object for participation in save + load.
    ///
    /// `name` is the sidecar filename stem; it must be unique within this
    /// coordinator and must be a valid filename fragment (no `/`, no null
    /// bytes, not `.` or `..`, not empty).
    pub fn register<T: Checkpointable + 'static>(
        &self,
        name: &str,
        obj: Arc<Mutex<T>>,
    ) -> Result<(), CheckpointError> {
        validate_name(name)?;

        // Upcast the `Arc<Mutex<T>>` to `Arc<Mutex<dyn CheckpointableDyn>>`
        // by way of a typed intermediate. We cannot `as`-cast directly but
        // we can go through an explicit coercion.
        let erased: Arc<Mutex<dyn CheckpointableDyn>> = obj;

        let mut reg = self
            .registry
            .lock()
            .map_err(|e| CheckpointError::Poisoned(e.to_string()))?;
        if reg.contains_key(name) {
            return Err(CheckpointError::DuplicateName(name.to_string()));
        }
        reg.insert(name.to_string(), Registration { obj: erased });
        Ok(())
    }

    /// Unregister an object by name. Returns `Ok(false)` if the name was
    /// not registered.
    pub fn unregister(&self, name: &str) -> Result<bool, CheckpointError> {
        let mut reg = self
            .registry
            .lock()
            .map_err(|e| CheckpointError::Poisoned(e.to_string()))?;
        Ok(reg.remove(name).is_some())
    }

    /// Register a migration function for `name` that advances a sidecar
    /// from `from_version` to `from_version + 1`.
    ///
    /// Migrations are chained by the coordinator: if a sidecar on disk is
    /// at v1 and the current `VERSION` is v3, the coordinator calls the
    /// v1→v2 migration, then the v2→v3 migration, in that order.
    ///
    /// Errors:
    /// - [`CheckpointError::Invariant`](nimcp_core::Error::Invariant) is
    ///   not raised; duplicate migration registrations simply overwrite,
    ///   which is convenient for testing and also harmless since
    ///   migration is a pure byte-to-byte function.
    ///
    /// # Note on API shape
    ///
    /// We only accept single-step migrations (`from` → `from + 1`). Callers
    /// who want to skip versions should register each intermediate step.
    /// This keeps the composition rules simple (no graph search at load
    /// time; just a linear chain) and means each migration can be tested
    /// in isolation.
    pub fn register_migration<F>(&self, name: &str, from_version: u32, migrate_fn: F)
    where
        F: Fn(&[u8]) -> Result<Vec<u8>, String> + Send + Sync + 'static,
    {
        let Ok(mut migs) = self.migrations.lock() else {
            warn!(
                name,
                from_version, "checkpoint migrations lock poisoned; dropping registration"
            );
            return;
        };
        migs.insert((name.to_string(), from_version), Arc::new(migrate_fn));
    }

    /// Snapshot every registered object to `dir`, atomically.
    ///
    /// Writes all sidecars + the manifest into `<dir>/.tmp/`, then
    /// promotes them into `<dir>/` one file at a time. A crash partway
    /// through leaves `<dir>/` at its previous good state; the next
    /// [`Self::load_all`] will detect and clean up `<dir>/.tmp/`.
    pub fn save_all(&self, dir: &Path) -> Result<(), CheckpointError> {
        self.save_all_inner(dir, /*abort_before_promote=*/ false)
    }

    /// Testing hook: perform a save up to the point of promotion, then
    /// leave `.tmp/` in place instead of renaming. Used by the
    /// fault-injection test to verify that an interrupted save does not
    /// corrupt a previously good `<dir>/`.
    #[doc(hidden)]
    pub fn save_all_abort_before_promote(&self, dir: &Path) -> Result<(), CheckpointError> {
        self.save_all_inner(dir, /*abort_before_promote=*/ true)
    }

    fn save_all_inner(
        &self,
        dir: &Path,
        abort_before_promote: bool,
    ) -> Result<(), CheckpointError> {
        fs::create_dir_all(dir)?;

        let tmp = dir.join(TMP_DIR);
        // If a previous abort left `.tmp/` around, nuke it before re-using.
        if tmp.exists() {
            fs::remove_dir_all(&tmp)?;
        }
        fs::create_dir_all(&tmp)?;

        // --- 1. Serialize every registered object into bytes. ---
        //
        // We do this under per-object locks held ONLY for the serialize
        // call; we never hold more than one actor lock at a time, and we
        // never hold any actor lock across the manifest / promote steps.
        let entries: Vec<(String, u32)>;
        {
            let reg = self
                .registry
                .lock()
                .map_err(|e| CheckpointError::Poisoned(e.to_string()))?;
            entries = reg.keys().cloned().map(|k| (k, 0)).collect();
            drop(reg);
        }

        let mut sidecars = Vec::with_capacity(entries.len());
        for (name, _) in entries {
            // Re-acquire the registry lock just long enough to clone the Arc.
            let arc = {
                let reg = self
                    .registry
                    .lock()
                    .map_err(|e| CheckpointError::Poisoned(e.to_string()))?;
                let Some(entry) = reg.get(&name) else {
                    // Could happen if another thread unregistered between
                    // the name snapshot and now. Treat as a mid-save race
                    // and skip.
                    warn!(%name, "registered object vanished mid-save; skipping");
                    continue;
                };
                entry.obj.clone()
            };

            let (bytes, version) = {
                let guard = arc
                    .lock()
                    .map_err(|e| CheckpointError::Poisoned(e.to_string()))?;
                let version = guard.current_version();
                let bytes = guard.save_bytes()?;
                (bytes, version)
            };

            // Write the sidecar into `.tmp/`.
            let sidecar_path = tmp.join(sidecar_filename(&name));
            write_file_durable(&sidecar_path, &bytes)?;

            sidecars.push(SidecarEntry {
                name,
                version,
                bytes: bytes.len() as u64,
            });
        }

        // --- 2. Write the manifest into `.tmp/manifest.json`. ---
        let manifest = Manifest {
            nimcp_checkpoint_version: MANIFEST_VERSION,
            saved_at: iso8601_utc_now(),
            sidecars,
        };
        let manifest_bytes = manifest.to_bytes()?;
        let manifest_tmp = tmp.join(MANIFEST_FILE);
        write_file_durable(&manifest_tmp, &manifest_bytes)?;

        // Testing hook: stop here, leaving `.tmp/` populated but
        // `<dir>/` unchanged.
        if abort_before_promote {
            warn!(
                dir = %dir.display(),
                "save_all_inner: aborting before promote (test hook)"
            );
            return Ok(());
        }

        // --- 3. Promote every file from `.tmp/` into `<dir>/`. ---
        //
        // Rename is atomic per-file on POSIX. We promote sidecars first
        // and the manifest LAST, so that if we crash mid-promote, the
        // leftover manifest (if present) still describes files we have
        // actually moved. The reverse order would risk a manifest that
        // references sidecars not yet promoted.
        for entry in &manifest.sidecars {
            let src = tmp.join(sidecar_filename(&entry.name));
            let dst = dir.join(sidecar_filename(&entry.name));
            atomic_rename(&src, &dst)?;
        }
        let manifest_dst = dir.join(MANIFEST_FILE);
        atomic_rename(&manifest_tmp, &manifest_dst)?;

        // --- 4. Remove the now-empty `.tmp/` dir. ---
        if let Err(e) = fs::remove_dir_all(&tmp) {
            // Not fatal; a stale `.tmp/` will be cleaned on next load.
            warn!(
                tmp = %tmp.display(),
                error = %e,
                "failed to remove temp dir after successful save"
            );
        }

        info!(
            dir = %dir.display(),
            sidecars = manifest.sidecars.len(),
            "checkpoint saved"
        );
        Ok(())
    }

    /// Restore every registered object from `dir`.
    ///
    /// Reads the manifest, then for each sidecar entry:
    /// 1. Reads the sidecar bytes.
    /// 2. Applies migrations to bring the bytes up to the current
    ///    [`Checkpointable::VERSION`].
    /// 3. Calls [`Checkpointable::load`] with the migrated bytes.
    ///
    /// Entries whose names are not registered are ignored with a warning
    /// — this supports graceful deprecation of retired subsystems.
    ///
    /// If a lingering `<dir>/.tmp/` is found, it is evidence of an
    /// aborted save. We log a warning and remove it; the committed
    /// `<dir>/` contents are what we actually load from.
    pub fn load_all(&self, dir: &Path) -> Result<(), CheckpointError> {
        // Crash recovery: remove any leftover `.tmp/`.
        let tmp = dir.join(TMP_DIR);
        if tmp.exists() {
            warn!(
                tmp = %tmp.display(),
                "found leftover temp dir from aborted save; cleaning up before load"
            );
            fs::remove_dir_all(&tmp)?;
        }

        // Read + parse manifest.
        let manifest_path = dir.join(MANIFEST_FILE);
        let manifest_bytes = fs::read(&manifest_path)?;
        let manifest = Manifest::from_bytes(&manifest_bytes, manifest_path.clone())?;

        // For each sidecar entry, load + migrate + hand to the object.
        for entry in &manifest.sidecars {
            let sidecar_path = dir.join(sidecar_filename(&entry.name));
            if !sidecar_path.exists() {
                return Err(CheckpointError::MissingSidecar {
                    name: entry.name.clone(),
                });
            }
            let disk_bytes = fs::read(&sidecar_path)?;
            if disk_bytes.len() as u64 != entry.bytes {
                return Err(CheckpointError::MalformedManifest {
                    path: manifest_path.clone(),
                    reason: format!(
                        "sidecar `{}` size mismatch: manifest says {} bytes, on-disk is {}",
                        entry.name,
                        entry.bytes,
                        disk_bytes.len(),
                    ),
                });
            }

            // Find the registered object (if any). If the name is no
            // longer registered, we skip with a warning — this supports
            // dropping retired subsystems.
            let maybe_arc = {
                let reg = self
                    .registry
                    .lock()
                    .map_err(|e| CheckpointError::Poisoned(e.to_string()))?;
                reg.get(&entry.name).map(|r| r.obj.clone())
            };
            let Some(arc) = maybe_arc else {
                warn!(
                    name = %entry.name,
                    "sidecar is present on disk but no object is registered for it; skipping"
                );
                continue;
            };

            let current_version = {
                let guard = arc
                    .lock()
                    .map_err(|e| CheckpointError::Poisoned(e.to_string()))?;
                guard.current_version()
            };

            // Versioning + migration decision tree.
            if entry.version > current_version {
                return Err(CheckpointError::UnknownFutureVersion {
                    name: entry.name.clone(),
                    found: entry.version,
                    current: current_version,
                });
            }

            let migrated_bytes = if entry.version < current_version {
                self.apply_migrations(&entry.name, disk_bytes, entry.version, current_version)?
            } else {
                disk_bytes
            };

            {
                let mut guard = arc
                    .lock()
                    .map_err(|e| CheckpointError::Poisoned(e.to_string()))?;
                guard.load_bytes(&migrated_bytes, current_version)?;
            }
        }

        info!(
            dir = %dir.display(),
            sidecars = manifest.sidecars.len(),
            "checkpoint loaded"
        );
        Ok(())
    }

    /// Apply registered migrations in sequence to advance `bytes` from
    /// `from_version` up to `to_version`. Errors if any step is missing.
    fn apply_migrations(
        &self,
        name: &str,
        mut bytes: Vec<u8>,
        from_version: u32,
        to_version: u32,
    ) -> Result<Vec<u8>, CheckpointError> {
        let mut current = from_version;
        while current < to_version {
            let key = (name.to_string(), current);
            let migration = {
                let migs = self
                    .migrations
                    .lock()
                    .map_err(|e| CheckpointError::Poisoned(e.to_string()))?;
                migs.get(&key).cloned()
            };
            let Some(migration) = migration else {
                return Err(CheckpointError::MissingMigration {
                    name: name.to_string(),
                    from: from_version,
                    to: to_version,
                    stuck_at: current,
                    next: current + 1,
                });
            };
            bytes = migration(&bytes).map_err(|reason| CheckpointError::MigrationFailed {
                name: name.to_string(),
                from: current,
                to: current + 1,
                reason,
            })?;
            current += 1;
        }
        Ok(bytes)
    }

    /// Number of registered objects. Primarily for tests + diagnostics.
    pub fn len(&self) -> usize {
        self.registry.lock().map(|r| r.len()).unwrap_or(0)
    }

    /// `true` if no objects are registered.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// --- Filesystem helpers --------------------------------------------------

/// Write `bytes` to `path` with best-effort durability: we write in full
/// and call `sync_all` so that a crash after return doesn't leave a
/// half-written file.
fn write_file_durable(path: &Path, bytes: &[u8]) -> Result<(), CheckpointError> {
    let mut f = fs::File::create(path)?;
    f.write_all(bytes)?;
    f.sync_all()?;
    Ok(())
}

/// Move `src` over `dst`, overwriting `dst` if it exists. `fs::rename` on
/// POSIX atomically replaces the destination; on Windows we let the
/// default semantics apply (same behavior for files on NTFS since
/// Windows 10 1607, which we target via MSRV 1.85).
fn atomic_rename(src: &Path, dst: &Path) -> Result<(), CheckpointError> {
    fs::rename(src, dst).map_err(CheckpointError::from)
}

/// Compose the filename for a sidecar with registered `name`.
fn sidecar_filename(name: &str) -> String {
    format!("{name}{SIDECAR_SUFFIX}")
}

/// Validate a proposed sidecar name.
fn validate_name(name: &str) -> Result<(), CheckpointError> {
    if name.is_empty() {
        return Err(CheckpointError::Other("sidecar name is empty".into()));
    }
    if name == "." || name == ".." {
        return Err(CheckpointError::Other(format!(
            "sidecar name `{name}` is reserved"
        )));
    }
    if name.contains('/') || name.contains('\\') || name.contains('\0') {
        return Err(CheckpointError::Other(format!(
            "sidecar name `{name}` contains illegal characters"
        )));
    }
    Ok(())
}

/// Current time formatted as an RFC 3339 / ISO 8601 UTC string, e.g.
/// `2026-04-21T12:34:56Z`. We format by hand to avoid pulling in `chrono`
/// just for the manifest header.
fn iso8601_utc_now() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // Convert seconds-since-epoch into components. Standard algorithm
    // from Howard Hinnant, public domain:
    //   https://howardhinnant.github.io/date_algorithms.html
    let days = (secs / 86_400) as i64;
    let secs_of_day = (secs % 86_400) as u32;
    let (year, month, day) = civil_from_days(days);
    let hour = secs_of_day / 3_600;
    let minute = (secs_of_day % 3_600) / 60;
    let second = secs_of_day % 60;
    format!(
        "{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z",
        year = year,
        month = month,
        day = day,
        hour = hour,
        minute = minute,
        second = second,
    )
}

/// `days` = days since 1970-01-01 (Unix epoch). Returns `(year, month, day)`
/// in the proleptic Gregorian calendar.
#[allow(clippy::many_single_char_names)]
fn civil_from_days(days: i64) -> (i32, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = (z - era * 146_097) as u64; // [0, 146096]
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    let year = if m <= 2 { y + 1 } else { y };
    (year as i32, m as u32, d as u32)
}

/// Read a sidecar body straight off disk. Exposed for tests + tooling.
///
/// Returns the raw bytes as stored; no version negotiation is performed.
pub fn read_sidecar_file(dir: &Path, name: &str) -> Result<Vec<u8>, CheckpointError> {
    validate_name(name)?;
    let path = dir.join(sidecar_filename(name));
    let mut f = fs::File::open(&path)?;
    let mut out = Vec::new();
    f.read_to_end(&mut out)?;
    Ok(out)
}

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use tempfile::tempdir;

    // A tiny fake `Checkpointable` used for round-trip + migration tests.
    //
    // We use an `AtomicU32` rather than a plain `u32` so we can mutate the
    // "current VERSION" between registrations — simulating a code upgrade.
    struct Fake {
        payload: Vec<u8>,
    }

    // Version-1 impl: bytes = payload verbatim.
    impl Checkpointable for Fake {
        const NAME: &'static str = "fake";
        const VERSION: u32 = 1;
        fn save(&self) -> Result<Vec<u8>, CheckpointError> {
            Ok(self.payload.clone())
        }
        fn load(&mut self, bytes: &[u8], version: u32) -> Result<(), CheckpointError> {
            assert_eq!(version, Self::VERSION);
            self.payload = bytes.to_vec();
            Ok(())
        }
    }

    // A second fake for the multi-object save test.
    struct Other(String);

    impl Checkpointable for Other {
        const NAME: &'static str = "other";
        const VERSION: u32 = 1;
        fn save(&self) -> Result<Vec<u8>, CheckpointError> {
            Ok(self.0.as_bytes().to_vec())
        }
        fn load(&mut self, bytes: &[u8], _version: u32) -> Result<(), CheckpointError> {
            self.0 = std::str::from_utf8(bytes)
                .map_err(|e| CheckpointError::Serialization(e.to_string()))?
                .to_string();
            Ok(())
        }
    }

    #[test]
    fn round_trip_single_object() {
        let tmp = tempdir().unwrap();

        // Save phase.
        let coord = CheckpointCoordinator::new();
        let obj = Arc::new(Mutex::new(Fake {
            payload: b"hello world".to_vec(),
        }));
        coord.register("fake", obj.clone()).unwrap();
        coord.save_all(tmp.path()).unwrap();

        // Mutate in-memory state so we can tell the load worked.
        obj.lock().unwrap().payload = b"TRASH".to_vec();

        // Load phase.
        coord.load_all(tmp.path()).unwrap();
        assert_eq!(obj.lock().unwrap().payload, b"hello world");
    }

    #[test]
    fn round_trip_multiple_objects() {
        let tmp = tempdir().unwrap();
        let coord = CheckpointCoordinator::new();
        let a = Arc::new(Mutex::new(Fake {
            payload: vec![1, 2, 3, 4],
        }));
        let b = Arc::new(Mutex::new(Other("snapshot".to_string())));
        coord.register("fake", a.clone()).unwrap();
        coord.register("other", b.clone()).unwrap();
        coord.save_all(tmp.path()).unwrap();

        a.lock().unwrap().payload.clear();
        b.lock().unwrap().0 = String::new();

        coord.load_all(tmp.path()).unwrap();
        assert_eq!(a.lock().unwrap().payload, vec![1, 2, 3, 4]);
        assert_eq!(b.lock().unwrap().0, "snapshot");
    }

    #[test]
    fn atomic_save_leaves_previous_untouched_on_abort() {
        let tmp = tempdir().unwrap();
        let dir = tmp.path();

        // First successful save with payload A.
        let coord = CheckpointCoordinator::new();
        let obj = Arc::new(Mutex::new(Fake {
            payload: b"FIRST".to_vec(),
        }));
        coord.register("fake", obj.clone()).unwrap();
        coord.save_all(dir).unwrap();

        // Mutate the in-memory state to something different.
        obj.lock().unwrap().payload = b"SECOND".to_vec();

        // Start a second save but abort before the promote step.
        coord.save_all_abort_before_promote(dir).unwrap();

        // The committed directory should still contain FIRST, untouched.
        let raw = read_sidecar_file(dir, "fake").unwrap();
        assert_eq!(raw, b"FIRST");

        // `.tmp/` should still be present (the abort is the whole point).
        assert!(dir.join(".tmp").exists());

        // Now load — the crash-recovery path should remove `.tmp/` and
        // succeed with the committed FIRST payload.
        obj.lock().unwrap().payload.clear();
        coord.load_all(dir).unwrap();
        assert_eq!(obj.lock().unwrap().payload, b"FIRST");
        assert!(
            !dir.join(".tmp").exists(),
            ".tmp/ should have been cleaned up"
        );
    }

    #[test]
    fn missing_sidecar_errors_cleanly() {
        let tmp = tempdir().unwrap();
        let dir = tmp.path();

        let coord = CheckpointCoordinator::new();
        let obj = Arc::new(Mutex::new(Fake {
            payload: b"data".to_vec(),
        }));
        coord.register("fake", obj.clone()).unwrap();
        coord.save_all(dir).unwrap();

        // Delete the sidecar behind the coordinator's back.
        fs::remove_file(dir.join("fake.sidecar")).unwrap();

        match coord.load_all(dir) {
            Err(CheckpointError::MissingSidecar { name }) => {
                assert_eq!(name, "fake");
            }
            other => panic!("expected MissingSidecar, got {other:?}"),
        }
    }

    // `VersionedFake` lets us pretend to have shipped a v2 of the schema
    // after a v1 was already saved to disk. Its VERSION is controlled by
    // a shared atomic so the test can bump it between save and load.
    static SHARED_VERSION: AtomicU32 = AtomicU32::new(1);

    struct VersionedFake {
        payload: Vec<u8>,
    }

    impl Checkpointable for VersionedFake {
        const NAME: &'static str = "ver";
        // The trait const itself is fixed per-type. We emulate upgrades
        // by having load() honor a different "current" via the shared
        // atomic, but the trait-level VERSION still needs to be stable
        // at compile time. So we keep VERSION at 2 and simulate a v1
        // save by invoking save on a value with a manually-set pre-v2
        // byte layout. See the test for details.
        const VERSION: u32 = 2;
        fn save(&self) -> Result<Vec<u8>, CheckpointError> {
            // v2 format: payload prefixed with a 1-byte schema tag 0x02.
            let mut out = vec![0x02];
            out.extend_from_slice(&self.payload);
            Ok(out)
        }
        fn load(&mut self, bytes: &[u8], version: u32) -> Result<(), CheckpointError> {
            assert_eq!(version, 2, "coordinator must migrate before calling load");
            if bytes.first() != Some(&0x02) {
                return Err(CheckpointError::Serialization(
                    "expected v2 tag 0x02".into(),
                ));
            }
            self.payload = bytes[1..].to_vec();
            Ok(())
        }
    }

    /// Helper: write a v1-format sidecar by hand (no tag prefix), then
    /// hand-build a manifest that claims version 1. Exercises the
    /// migration path in the coordinator.
    #[test]
    fn version_migration_v1_to_v2() {
        let tmp = tempdir().unwrap();
        let dir = tmp.path();

        // Manually lay down a v1 checkpoint: just raw payload, no tag.
        let v1_payload = b"legacy-state".to_vec();
        fs::write(dir.join("ver.sidecar"), &v1_payload).unwrap();
        let manifest = Manifest {
            nimcp_checkpoint_version: MANIFEST_VERSION,
            saved_at: "2026-01-01T00:00:00Z".into(),
            sidecars: vec![SidecarEntry {
                name: "ver".into(),
                version: 1,
                bytes: v1_payload.len() as u64,
            }],
        };
        fs::write(dir.join(MANIFEST_FILE), manifest.to_bytes().unwrap()).unwrap();

        // Register a v2 object + a v1 -> v2 migration that adds the tag.
        let coord = CheckpointCoordinator::new();
        let obj = Arc::new(Mutex::new(VersionedFake {
            payload: b"will be overwritten".to_vec(),
        }));
        coord.register("ver", obj.clone()).unwrap();
        coord.register_migration("ver", 1, |bytes| {
            let mut out = vec![0x02];
            out.extend_from_slice(bytes);
            Ok(out)
        });

        coord.load_all(dir).unwrap();
        assert_eq!(obj.lock().unwrap().payload, b"legacy-state");
    }

    #[test]
    fn missing_migration_errors_cleanly() {
        let tmp = tempdir().unwrap();
        let dir = tmp.path();

        // Lay down a v1 sidecar + manifest, then try to load v2 without
        // registering a migration.
        let v1_payload = b"legacy".to_vec();
        fs::write(dir.join("ver.sidecar"), &v1_payload).unwrap();
        let manifest = Manifest {
            nimcp_checkpoint_version: MANIFEST_VERSION,
            saved_at: "2026-01-01T00:00:00Z".into(),
            sidecars: vec![SidecarEntry {
                name: "ver".into(),
                version: 1,
                bytes: v1_payload.len() as u64,
            }],
        };
        fs::write(dir.join(MANIFEST_FILE), manifest.to_bytes().unwrap()).unwrap();

        let coord = CheckpointCoordinator::new();
        let obj = Arc::new(Mutex::new(VersionedFake {
            payload: Vec::new(),
        }));
        coord.register("ver", obj).unwrap();

        match coord.load_all(dir) {
            Err(CheckpointError::MissingMigration {
                name,
                from,
                to,
                stuck_at,
                next,
            }) => {
                assert_eq!(name, "ver");
                assert_eq!(from, 1);
                assert_eq!(to, 2);
                assert_eq!(stuck_at, 1);
                assert_eq!(next, 2);
            }
            other => panic!("expected MissingMigration, got {other:?}"),
        }
    }

    #[test]
    fn future_version_rejected() {
        let tmp = tempdir().unwrap();
        let dir = tmp.path();

        // Lay down a sidecar whose version (99) is ahead of our code (v2).
        fs::write(dir.join("ver.sidecar"), b"future-bytes").unwrap();
        let manifest = Manifest {
            nimcp_checkpoint_version: MANIFEST_VERSION,
            saved_at: "2026-01-01T00:00:00Z".into(),
            sidecars: vec![SidecarEntry {
                name: "ver".into(),
                version: 99,
                bytes: 12,
            }],
        };
        fs::write(dir.join(MANIFEST_FILE), manifest.to_bytes().unwrap()).unwrap();

        let coord = CheckpointCoordinator::new();
        let obj = Arc::new(Mutex::new(VersionedFake {
            payload: Vec::new(),
        }));
        coord.register("ver", obj).unwrap();

        match coord.load_all(dir) {
            Err(CheckpointError::UnknownFutureVersion {
                name,
                found,
                current,
            }) => {
                assert_eq!(name, "ver");
                assert_eq!(found, 99);
                assert_eq!(current, 2);
            }
            other => panic!("expected UnknownFutureVersion, got {other:?}"),
        }
    }

    #[test]
    fn malformed_manifest_errors_cleanly() {
        let tmp = tempdir().unwrap();
        let dir = tmp.path();
        fs::write(dir.join(MANIFEST_FILE), b"{not valid json").unwrap();

        let coord = CheckpointCoordinator::new();
        match coord.load_all(dir) {
            Err(CheckpointError::MalformedManifest { path, reason: _ }) => {
                assert_eq!(path, dir.join(MANIFEST_FILE));
            }
            other => panic!("expected MalformedManifest, got {other:?}"),
        }
    }

    #[test]
    fn duplicate_registration_errors() {
        let coord = CheckpointCoordinator::new();
        let a = Arc::new(Mutex::new(Fake { payload: vec![] }));
        let b = Arc::new(Mutex::new(Fake { payload: vec![] }));
        coord.register("fake", a).unwrap();
        match coord.register("fake", b) {
            Err(CheckpointError::DuplicateName(n)) => assert_eq!(n, "fake"),
            other => panic!("expected DuplicateName, got {other:?}"),
        }
    }

    #[test]
    fn invalid_names_rejected() {
        let coord = CheckpointCoordinator::new();
        let obj = Arc::new(Mutex::new(Fake { payload: vec![] }));
        assert!(matches!(
            coord.register("", obj.clone()),
            Err(CheckpointError::Other(_))
        ));
        assert!(matches!(
            coord.register("..", obj.clone()),
            Err(CheckpointError::Other(_))
        ));
        assert!(matches!(
            coord.register("a/b", obj.clone()),
            Err(CheckpointError::Other(_))
        ));
    }

    #[test]
    fn unregistered_sidecar_is_skipped_with_warning() {
        let tmp = tempdir().unwrap();
        let dir = tmp.path();

        // Save with both `fake` and `other` registered.
        {
            let coord = CheckpointCoordinator::new();
            let a = Arc::new(Mutex::new(Fake {
                payload: b"A".to_vec(),
            }));
            let b = Arc::new(Mutex::new(Other("B".into())));
            coord.register("fake", a).unwrap();
            coord.register("other", b).unwrap();
            coord.save_all(dir).unwrap();
        }

        // Load with only `fake` registered — `other` should be silently
        // skipped (with a warning in tracing).
        let coord2 = CheckpointCoordinator::new();
        let a2 = Arc::new(Mutex::new(Fake { payload: vec![] }));
        coord2.register("fake", a2.clone()).unwrap();
        coord2.load_all(dir).unwrap();
        assert_eq!(a2.lock().unwrap().payload, b"A");
    }

    #[test]
    fn sidecar_truncation_detected() {
        let tmp = tempdir().unwrap();
        let dir = tmp.path();

        let coord = CheckpointCoordinator::new();
        let obj = Arc::new(Mutex::new(Fake {
            payload: b"important-payload".to_vec(),
        }));
        coord.register("fake", obj).unwrap();
        coord.save_all(dir).unwrap();

        // Truncate the sidecar on disk.
        fs::write(dir.join("fake.sidecar"), b"short").unwrap();

        match coord.load_all(dir) {
            Err(CheckpointError::MalformedManifest { reason, .. }) => {
                assert!(
                    reason.contains("size mismatch"),
                    "unexpected reason: {reason}"
                );
            }
            other => panic!("expected MalformedManifest for size mismatch, got {other:?}"),
        }
    }

    #[test]
    fn empty_coordinator_saves_empty_manifest() {
        let tmp = tempdir().unwrap();
        let coord = CheckpointCoordinator::new();
        coord.save_all(tmp.path()).unwrap();

        let mf_bytes = fs::read(tmp.path().join(MANIFEST_FILE)).unwrap();
        let mf = Manifest::from_bytes(&mf_bytes, tmp.path().join(MANIFEST_FILE)).unwrap();
        assert!(mf.sidecars.is_empty());
        assert_eq!(mf.nimcp_checkpoint_version, MANIFEST_VERSION);
    }

    #[test]
    fn coordinator_len_and_empty() {
        let coord = CheckpointCoordinator::new();
        assert!(coord.is_empty());
        let a = Arc::new(Mutex::new(Fake { payload: vec![] }));
        coord.register("fake", a).unwrap();
        assert_eq!(coord.len(), 1);
        assert!(!coord.is_empty());
        coord.unregister("fake").unwrap();
        assert!(coord.is_empty());
    }

    #[test]
    fn multi_step_migration_v1_to_v3() {
        // Synthetic scenario: current VERSION is 2 (per VersionedFake),
        // so this test uses a separate helper flow. To exercise the
        // chained-migration path, we load a v1 sidecar and register TWO
        // migrations (v1->v2) — proving the coordinator chains, not just
        // that it runs a single step. For a true 3-step chain we'd need
        // a type with VERSION=3; we emulate by verifying the chain logic
        // exits cleanly after one step when the target is reached.
        //
        // The migration LOOP itself (apply_migrations) is the thing under
        // test. Construct the scenario directly against apply_migrations.
        let coord = CheckpointCoordinator::new();

        // Chain: v1 -> v2 prepends "A"; v2 -> v3 prepends "B". End result
        // of migrating "x" from v1 should be "BAx".
        coord.register_migration("chain", 1, |b| {
            let mut out = b"A".to_vec();
            out.extend_from_slice(b);
            Ok(out)
        });
        coord.register_migration("chain", 2, |b| {
            let mut out = b"B".to_vec();
            out.extend_from_slice(b);
            Ok(out)
        });

        let migrated = coord
            .apply_migrations("chain", b"x".to_vec(), 1, 3)
            .expect("chained migrations should succeed");
        assert_eq!(migrated, b"BAx");

        // Also verify that stopping short works (v1 -> v1 is a no-op).
        let noop = coord
            .apply_migrations("chain", b"x".to_vec(), 1, 1)
            .unwrap();
        assert_eq!(noop, b"x");

        // Ensure the SHARED_VERSION noise above didn't leak.
        assert_eq!(SHARED_VERSION.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn failed_migration_propagates() {
        let coord = CheckpointCoordinator::new();
        coord.register_migration("bad", 1, |_| Err("intentional".into()));
        match coord.apply_migrations("bad", b"x".to_vec(), 1, 2) {
            Err(CheckpointError::MigrationFailed {
                name,
                from,
                to,
                reason,
            }) => {
                assert_eq!(name, "bad");
                assert_eq!(from, 1);
                assert_eq!(to, 2);
                assert_eq!(reason, "intentional");
            }
            other => panic!("expected MigrationFailed, got {other:?}"),
        }
    }
}
