//! Per-actor runtime metrics.
//!
//! Metrics are kept on a shared `Arc` because multiple tasks (the actor
//! loop, the mailbox producer, the scheduler's introspection API) observe
//! the same counters. All updates are atomic; no locks.
//!
//! The handle-duration histogram is a simple fixed-bucket counter — we
//! don't need HDR-level precision here; we care about spotting actors
//! that have blown their budget. Buckets are in microseconds.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// Upper bounds of the handle-duration histogram buckets, in microseconds.
///
/// A measurement of X us increments the first bucket whose bound >= X; the
/// final `u64::MAX` bucket catches pathological outliers.
pub const HISTOGRAM_BUCKETS_US: &[u64] = &[
    10,
    50,
    100,
    500,
    1_000,
    5_000,
    10_000,
    50_000,
    100_000,
    500_000,
    1_000_000,
    u64::MAX,
];

/// Snapshot of one actor's metrics at an instant in time.
///
/// All fields are monotonic (except `mailbox_depth`, which is a live gauge).
#[derive(Debug, Clone, Default)]
pub struct ActorMetrics {
    /// Actor's logical name (from `Actor::name()`).
    pub name: &'static str,
    /// Number of messages currently queued in the mailbox.
    pub mailbox_depth: u64,
    /// Total messages delivered to the handler since spawn (across restarts).
    pub messages_handled: u64,
    /// Total times the supervisor restarted this actor.
    pub restarts: u64,
    /// Total handler errors observed (whether or not they caused a restart).
    pub errors: u64,
    /// Histogram of handler durations, aligned with [`HISTOGRAM_BUCKETS_US`].
    pub handle_duration_us: Vec<u64>,
}

/// Shared metric cells owned by the scheduler and updated by the actor loop.
///
/// Cloning shares state (`Arc` internally) so the scheduler's introspection
/// code and the hot-path actor loop see identical numbers.
#[derive(Debug, Clone)]
pub(crate) struct MetricsCell {
    pub name: &'static str,
    pub mailbox_depth: Arc<AtomicU64>,
    pub messages_handled: Arc<AtomicU64>,
    pub restarts: Arc<AtomicU64>,
    pub errors: Arc<AtomicU64>,
    pub histogram: Arc<Vec<AtomicU64>>,
}

impl MetricsCell {
    pub(crate) fn new(name: &'static str) -> Self {
        let histogram: Vec<AtomicU64> = HISTOGRAM_BUCKETS_US
            .iter()
            .map(|_| AtomicU64::new(0))
            .collect();
        Self {
            name,
            mailbox_depth: Arc::new(AtomicU64::new(0)),
            messages_handled: Arc::new(AtomicU64::new(0)),
            restarts: Arc::new(AtomicU64::new(0)),
            errors: Arc::new(AtomicU64::new(0)),
            histogram: Arc::new(histogram),
        }
    }

    pub(crate) fn record_handle_duration(&self, elapsed_us: u64) {
        // First bucket whose upper bound >= elapsed wins.
        for (i, bound) in HISTOGRAM_BUCKETS_US.iter().enumerate() {
            if elapsed_us <= *bound {
                self.histogram[i].fetch_add(1, Ordering::Relaxed);
                return;
            }
        }
        // Unreachable: last bucket is u64::MAX. Defensive only.
        let last = self.histogram.len() - 1;
        self.histogram[last].fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn snapshot(&self) -> ActorMetrics {
        ActorMetrics {
            name: self.name,
            mailbox_depth: self.mailbox_depth.load(Ordering::Relaxed),
            messages_handled: self.messages_handled.load(Ordering::Relaxed),
            restarts: self.restarts.load(Ordering::Relaxed),
            errors: self.errors.load(Ordering::Relaxed),
            handle_duration_us: self
                .histogram
                .iter()
                .map(|cell| cell.load(Ordering::Relaxed))
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn histogram_records_into_correct_bucket() {
        let cell = MetricsCell::new("test");
        cell.record_handle_duration(5); // first bucket (<= 10)
        cell.record_handle_duration(25); // second bucket (<= 50)
        cell.record_handle_duration(10_000_000); // last bucket
        let snap = cell.snapshot();
        assert_eq!(snap.handle_duration_us[0], 1);
        assert_eq!(snap.handle_duration_us[1], 1);
        assert_eq!(*snap.handle_duration_us.last().unwrap(), 1);
    }

    #[test]
    fn snapshot_captures_all_counters() {
        let cell = MetricsCell::new("n");
        cell.messages_handled.fetch_add(7, Ordering::Relaxed);
        cell.restarts.fetch_add(2, Ordering::Relaxed);
        cell.errors.fetch_add(3, Ordering::Relaxed);
        cell.mailbox_depth.fetch_add(5, Ordering::Relaxed);
        let snap = cell.snapshot();
        assert_eq!(snap.messages_handled, 7);
        assert_eq!(snap.restarts, 2);
        assert_eq!(snap.errors, 3);
        assert_eq!(snap.mailbox_depth, 5);
    }
}
