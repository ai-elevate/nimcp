//! Warmup gate for rate-dependent plasticity (R-STDP, homeostatic).
//!
//! # Why
//!
//! V1 ran R-STDP from the first tick, so freshly-initialized networks saw
//! huge Hebbian updates on near-random activity, producing runaway
//! potentiation. The V1 fix required `rate_samples ≥ 100` before R-STDP
//! could apply.
//!
//! V2 ships this gate as an explicit helper. Rate-dependent rules call
//! `rate_samples_ready(samples, threshold)` before computing a delta;
//! callers can't silently skip the check.
//!
//! # Math
//!
//! Trivially, `samples >= threshold`. This is a one-liner on purpose —
//! the *explicitness* is the point.

/// Default warmup threshold matching V1's post-fix behavior.
pub const DEFAULT_WARMUP_THRESHOLD: u32 = 100;

/// Return `true` iff enough rate samples have been observed for a
/// rate-dependent plasticity rule to fire safely.
#[must_use]
#[inline]
pub const fn rate_samples_ready(samples: u32, threshold: u32) -> bool {
    samples >= threshold
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_blocks_below_threshold() {
        assert!(!rate_samples_ready(0, DEFAULT_WARMUP_THRESHOLD));
        assert!(!rate_samples_ready(1, DEFAULT_WARMUP_THRESHOLD));
        assert!(!rate_samples_ready(99, DEFAULT_WARMUP_THRESHOLD));
    }

    #[test]
    fn gate_opens_at_threshold() {
        assert!(rate_samples_ready(DEFAULT_WARMUP_THRESHOLD, DEFAULT_WARMUP_THRESHOLD));
    }

    #[test]
    fn gate_stays_open_past_threshold() {
        assert!(rate_samples_ready(101, DEFAULT_WARMUP_THRESHOLD));
        assert!(rate_samples_ready(u32::MAX, DEFAULT_WARMUP_THRESHOLD));
    }

    /// Zero-threshold: always open. (Caller opted out.)
    #[test]
    fn zero_threshold_always_open() {
        assert!(rate_samples_ready(0, 0));
    }

    #[test]
    fn default_threshold_matches_v1_fix() {
        assert_eq!(DEFAULT_WARMUP_THRESHOLD, 100);
    }
}
