//! Tunable parameter state with baseline for symmetric recovery.
//!
//! Port of master commit `1a495f51d feat(snn): three control fixes` —
//! specifically Fix 2: watchdog captures baseline params on startup and,
//! after N consecutive healthy polls, steps each tunable back toward
//! baseline by a fixed fraction per tick.
//!
//! # Why this exists
//!
//! V1's watchdog could only monotonically shrink parameters (half
//! noise_rate, half pulse_mv, etc.) when it saw saturation. Once
//! healthy, there was no recovery path: the parameters stayed at
//! their reduced values indefinitely, leaving the system one knob-
//! collapse away from collapse. Pair that with a transient saturation
//! event and the oscillation becomes self-sustaining.
//!
//! This module owns the "where are we relative to factory settings"
//! bookkeeping. The watchdog policy layer decides when to invoke
//! [`TunableState::nudge_toward_baseline`] (typically after 3
//! consecutive HEALTHY or TRANSIENT polls, matching master's default).
//!
//! # Thread-safety
//!
//! [`TunableState`] is `Send + !Sync` — the scheduler pins watchdog
//! work to a single actor, so no lock is needed.

use serde::{Deserialize, Serialize};

/// Number of consecutive healthy polls V1 requires before the watchdog
/// starts nudging parameters back toward baseline. Exposed as a public
/// constant so callers match V1's policy without magic numbers.
pub const HEALTHY_POLLS_BEFORE_RECOVERY: u32 = 3;
/// Fractional step per recovery tick — master `1a495f51d` uses 0.20.
/// Applied multiplicatively: `new = current + fraction × (baseline − current)`.
pub const RECOVERY_STEP_FRACTION: f32 = 0.20;

/// One tunable parameter with its own baseline.
///
/// `current` is what the watchdog / operator currently set the parameter
/// to; `baseline` is the value captured on startup (or any other
/// "known-good" moment). `nudge_toward_baseline` moves `current` a
/// fraction of the way back toward `baseline`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TunableState {
    /// Value the rest of the system reads.
    pub current: f32,
    /// Factory / startup value to recover toward.
    pub baseline: f32,
}

impl TunableState {
    /// Snapshot a value as both current and baseline — use at startup.
    #[must_use]
    pub fn snapshot(value: f32) -> Self {
        Self {
            current: value,
            baseline: value,
        }
    }

    /// Force `current` to a new value without touching baseline —
    /// use when the watchdog actively reduces a tunable.
    pub fn set_current(&mut self, v: f32) {
        self.current = v;
    }

    /// Update the baseline in-place — use only when operator explicitly
    /// redefines "factory settings" (e.g. after a stable 24-hour run).
    pub fn set_baseline(&mut self, v: f32) {
        self.baseline = v;
    }

    /// Step `current` one fraction of the way back toward `baseline`.
    /// Standard call: `nudge_toward_baseline(RECOVERY_STEP_FRACTION)`.
    ///
    /// `fraction` is clamped to `[0, 1]`. At `fraction == 0.0` this is
    /// a no-op; at `fraction == 1.0` current snaps fully to baseline.
    pub fn nudge_toward_baseline(&mut self, fraction: f32) {
        let f = fraction.clamp(0.0, 1.0);
        self.current += f * (self.baseline - self.current);
    }

    /// Distance between `current` and `baseline`. Useful for telemetry
    /// — watchdog emits this so operators can see how far from factory
    /// the tunables have drifted.
    #[must_use]
    pub fn drift(&self) -> f32 {
        self.current - self.baseline
    }

    /// Absolute drift normalized by baseline magnitude. Returns 0.0
    /// when baseline is exactly zero (avoids `NaN`).
    #[must_use]
    pub fn drift_fraction(&self) -> f32 {
        if self.baseline.abs() < f32::EPSILON {
            return 0.0;
        }
        self.drift().abs() / self.baseline.abs()
    }
}

/// Simple counter that fires [`HEALTHY_POLLS_BEFORE_RECOVERY`] polls
/// after the last non-healthy observation. Wrapping helper for
/// watchdog policies that implement the full recovery loop.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct HealthyStreak {
    /// Number of consecutive healthy polls.
    pub count: u32,
}

impl HealthyStreak {
    /// Call with `true` each time the watchdog observed a HEALTHY /
    /// TRANSIENT poll; with `false` for SATURATED / COLLAPSED.
    /// Returns `true` the first poll after the streak reaches the
    /// recovery threshold — recovery should tick once and then fall
    /// silent until the streak resets. The caller is responsible
    /// for calling once per watchdog poll.
    pub fn observe(&mut self, healthy: bool) -> bool {
        if !healthy {
            self.count = 0;
            return false;
        }
        self.count = self.count.saturating_add(1);
        self.count >= HEALTHY_POLLS_BEFORE_RECOVERY
    }

    /// Reset the streak — use when operator manually overrides tunables.
    pub fn reset(&mut self) {
        self.count = 0;
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_sets_both_values() {
        let t = TunableState::snapshot(20.0);
        assert_eq!(t.current, 20.0);
        assert_eq!(t.baseline, 20.0);
        assert_eq!(t.drift(), 0.0);
    }

    #[test]
    fn nudge_steps_fraction_of_way_back() {
        let mut t = TunableState {
            current: 5.0,
            baseline: 20.0,
        };
        t.nudge_toward_baseline(0.2);
        // new = 5 + 0.2 × (20 − 5) = 5 + 3 = 8
        assert!((t.current - 8.0).abs() < 1e-5);
        assert_eq!(t.baseline, 20.0);
    }

    #[test]
    fn nudge_fraction_one_snaps_to_baseline() {
        let mut t = TunableState {
            current: 1.0,
            baseline: 20.0,
        };
        t.nudge_toward_baseline(1.0);
        assert!((t.current - 20.0).abs() < 1e-5);
    }

    #[test]
    fn nudge_fraction_zero_no_op() {
        let mut t = TunableState {
            current: 5.0,
            baseline: 20.0,
        };
        t.nudge_toward_baseline(0.0);
        assert_eq!(t.current, 5.0);
    }

    #[test]
    fn nudge_clamps_out_of_range_fraction() {
        let mut t = TunableState {
            current: 5.0,
            baseline: 20.0,
        };
        t.nudge_toward_baseline(-1.0);
        assert_eq!(t.current, 5.0); // clamped to 0
        t.nudge_toward_baseline(1000.0);
        assert_eq!(t.current, 20.0); // clamped to 1
    }

    #[test]
    fn nudge_converges_geometrically() {
        let mut t = TunableState {
            current: 0.0,
            baseline: 100.0,
        };
        // 20% per tick — after 20 ticks drift should be near zero.
        for _ in 0..20 {
            t.nudge_toward_baseline(RECOVERY_STEP_FRACTION);
        }
        assert!(t.drift_fraction() < 0.02);
    }

    #[test]
    fn drift_fraction_safe_at_zero_baseline() {
        let t = TunableState {
            current: 5.0,
            baseline: 0.0,
        };
        assert_eq!(t.drift_fraction(), 0.0);
    }

    #[test]
    fn streak_fires_only_on_threshold_cross() {
        let mut s = HealthyStreak::default();
        assert!(!s.observe(true)); // 1
        assert!(!s.observe(true)); // 2
        assert!(s.observe(true)); // 3 — first fire
        assert!(s.observe(true)); // 4 — saturating, keeps firing
    }

    #[test]
    fn streak_resets_on_unhealthy() {
        let mut s = HealthyStreak::default();
        assert!(!s.observe(true));
        assert!(!s.observe(true));
        assert!(!s.observe(false)); // unhealthy resets count to 0
        assert_eq!(s.count, 0);
        assert!(!s.observe(true)); // 1 again
    }

    #[test]
    fn streak_manual_reset() {
        let mut s = HealthyStreak::default();
        s.observe(true);
        s.observe(true);
        s.reset();
        assert_eq!(s.count, 0);
    }

    /// Integration: simulate a watchdog poll loop — 5 saturated polls,
    /// parameter is halved, then healthy for 10 polls with recovery
    /// nudging. Parameter should return to ~baseline.
    #[test]
    fn watchdog_loop_restores_parameter() {
        let baseline = 20.0;
        let mut param = TunableState::snapshot(baseline);
        let mut streak = HealthyStreak::default();

        // Saturation event — watchdog halves the tunable.
        param.set_current(param.current * 0.5);
        assert_eq!(param.current, 10.0);
        for _ in 0..5 {
            let _ = streak.observe(false);
        }
        assert_eq!(streak.count, 0);

        // Recovery sequence.
        for _ in 0..20 {
            let should_recover = streak.observe(true);
            if should_recover {
                param.nudge_toward_baseline(RECOVERY_STEP_FRACTION);
            }
        }
        // 18 recovery ticks at 20% → current ~= baseline within 2%.
        assert!(param.drift_fraction() < 0.02);
    }
}
