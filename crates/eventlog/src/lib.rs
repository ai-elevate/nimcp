//! NIMCP V2 — append-only persistent event log.
//!
//! # Purpose
//!
//! In V2, every state mutation is an event. The event log is the canonical
//! source of truth. Benefits:
//!
//! 1. **Determinism.** Replay the log, get bit-identical state.
//! 2. **Crash recovery.** Truncate to last valid event, replay, resume.
//! 3. **Debugging.** Every weight update, every synapse creation is a
//!    durable, inspectable record.
//! 4. **No races on shared state** — the log is append-only + serialized.
//!
//! # Phase 0 scope
//!
//! - Append operation with atomic flush
//! - Read operation by range
//! - Truncate (for checkpoint compaction)
//! - Local fs backend only; replication is a Phase 8 concern
//!
//! # Anti-goals
//!
//! - Distributed consensus (not yet)
//! - Multi-writer (single-writer by design, avoid coordination overhead)

#![forbid(unsafe_code)]

use nimcp_core::EventId;
use std::path::PathBuf;
use thiserror::Error;

/// Errors specific to the event log.
#[derive(Debug, Error)]
pub enum LogError {
    /// I/O failure reading or writing to the log file.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    /// Log file header or entry framing is invalid.
    #[error("corrupt log: {0}")]
    Corrupt(String),
    /// Requested event ID is past the end of the log.
    #[error("event {0:?} not in log")]
    NotFound(EventId),
}

/// Configuration for an event log.
#[derive(Debug, Clone)]
pub struct LogConfig {
    /// Directory where the log file lives.
    pub dir: PathBuf,
    /// fsync after every append (safe but slow). If false, flushes only on
    /// `flush()`. Set true for production; false for tests.
    pub fsync_on_append: bool,
}

impl Default for LogConfig {
    fn default() -> Self {
        Self {
            dir: PathBuf::from("./nimcp-eventlog"),
            fsync_on_append: true,
        }
    }
}

/// Placeholder event log handle. Phase 0 implementation follows.
///
/// TODO(phase-0): implement append/read/truncate over a single append-only
/// file with a CRC32 per entry for corruption detection.
pub struct EventLog {
    #[allow(dead_code)]
    config: LogConfig,
    next_id: u64,
}

impl EventLog {
    /// Open (or create) an event log at the configured path.
    pub fn open(config: LogConfig) -> Result<Self, LogError> {
        std::fs::create_dir_all(&config.dir)?;
        tracing::info!(dir = ?config.dir, "opened event log (stub)");
        Ok(Self { config, next_id: 0 })
    }

    /// Append a raw (already-serialized) event payload. Returns the assigned ID.
    ///
    /// TODO(phase-0): actually persist to disk.
    pub fn append(&mut self, _payload: &[u8]) -> Result<EventId, LogError> {
        let id = EventId(self.next_id);
        self.next_id += 1;
        Ok(id)
    }

    /// Total number of events currently in the log.
    pub fn len(&self) -> u64 {
        self.next_id
    }

    /// True if the log contains no events.
    pub fn is_empty(&self) -> bool {
        self.next_id == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn opens_and_appends() {
        let dir = tempdir().unwrap();
        let mut log = EventLog::open(LogConfig {
            dir: dir.path().to_path_buf(),
            fsync_on_append: false,
        })
        .unwrap();
        assert!(log.is_empty());
        let id0 = log.append(b"hello").unwrap();
        let id1 = log.append(b"world").unwrap();
        assert_eq!(id0, EventId(0));
        assert_eq!(id1, EventId(1));
        assert_eq!(log.len(), 2);
    }
}
