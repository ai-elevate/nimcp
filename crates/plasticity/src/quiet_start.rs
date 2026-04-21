//! Quiet-start load-time weight transform.
//!
//! # Why this exists
//!
//! In V1, quiet-start was an *initialization-only* protocol: on fresh brain
//! creation it scaled weights down so new networks wouldn't spike-runaway.
//! When a checkpoint saved mid-saturation was then reloaded, quiet-start did
//! not re-apply and the brain re-entered the saturated state. See
//! `project_snn_quiet_start_2026-04-20.md` in V1 notes.
//!
//! V2 fixes this by making quiet-start a **pure transform** keyed on
//! *observed* firing stats — callable at init, at load, or any time a
//! population is detected to be saturated.
//!
//! # Math
//!
//! For each population with observed rate `r_i` and target `r*`:
//!
//! ```text
//! ratio_i = r_i / r*
//! scale_i = clamp(r* / r_i,  0.01, 1.0)      if r_i > upper_band · r*
//!         = clamp(sqrt(r* / max(r_i, ε)), 1.0, max_up)  if r_i < lower_band · r*
//!         = 1.0                               otherwise (healthy)
//! ```
//!
//! The saturated branch uses a **linear** correction (`r* / r_i`) so that a
//! population firing at 33× target gets scaled to roughly 1/33× of its
//! original weight — enough to drop its rate directly onto target under the
//! linear-rate assumption. Aggressive downscale is allowed (floor `0.01`)
//! because this is a one-shot transform applied under operator control, not
//! a feedback loop. A dying population gets a modest upscale (cap
//! `max_up = 5.0`) — large upscales on silent populations cause runaway
//! on resume.

use serde::{Deserialize, Serialize};

/// Lower bound on the returned scale. Aggressive downscale is intentional.
pub const MIN_QUIET_SCALE: f32 = 0.01;
/// Upper bound on the returned scale. Conservative upscale — a dead pop
/// needs coaxing, not shocking.
pub const MAX_QUIET_SCALE: f32 = 1.0;
/// Upper upscale cap used when `observed < lower_band · target`. Kept modest
/// to avoid runaway; compound applications handle the rest.
pub const MAX_UPSCALE_CAP: f32 = 5.0;
/// Floor to prevent divide-by-zero when a population is totally silent.
const EPS: f32 = 1e-9;

/// Thresholds for the "healthy" band around `target_rate`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct QuietStartBand {
    /// Fraction of target below which the pop is "too quiet".
    /// Default `0.5` → `r < 0.5 · r*` triggers upscale.
    pub lower_band: f32,
    /// Fraction of target above which the pop is "saturated".
    /// Default `2.0` → `r > 2.0 · r*` triggers downscale.
    pub upper_band: f32,
}

impl Default for QuietStartBand {
    fn default() -> Self {
        Self {
            lower_band: 0.5,
            upper_band: 2.0,
        }
    }
}

/// Compute a per-population weight scale factor to apply as a one-shot
/// transform. Output shape matches `observed_rates` input shape.
///
/// - Healthy pops (within the band) get exactly `1.0`.
/// - Saturated pops get `scale ∈ [0.01, 1.0)`.
/// - Quiet pops get `scale ∈ (1.0, 5.0]`.
///
/// # Notes
///
/// This is *not* a feedback loop — the returned scales are wider than the
/// homeostatic `[0.98, 1.02]` range because the operator is explicitly
/// asking for a one-shot correction.
#[must_use]
pub fn quiet_start_scale(observed_rates: &[f32], target_rate: f32) -> Vec<f32> {
    quiet_start_scale_with_band(observed_rates, target_rate, &QuietStartBand::default())
}

/// Variant of [`quiet_start_scale`] with explicit band thresholds.
#[must_use]
pub fn quiet_start_scale_with_band(
    observed_rates: &[f32],
    target_rate: f32,
    band: &QuietStartBand,
) -> Vec<f32> {
    if !(target_rate > 0.0) {
        return vec![1.0; observed_rates.len()];
    }

    let upper = band.upper_band * target_rate;
    let lower = band.lower_band * target_rate;

    observed_rates
        .iter()
        .map(|&r| {
            if r > upper {
                // Saturated — downscale linearly so r*k → target_rate under
                // the linear-rate assumption. For r = 1.0, target = 0.03 this
                // yields scale = 0.03. Clamped to [0.01, 1.0] so no-op when
                // above-upper by a hair (≤ 100× still floors at 0.01).
                (target_rate / r.max(EPS)).clamp(MIN_QUIET_SCALE, MAX_QUIET_SCALE)
            } else if r < lower {
                // Too quiet — gentle upscale, modest cap.
                let s = (target_rate / r.max(EPS)).sqrt();
                s.clamp(1.0, MAX_UPSCALE_CAP)
            } else {
                1.0
            }
        })
        .collect()
}

#[cfg(test)]
#[allow(clippy::float_cmp)] // exact-equality asserts are intentional
mod tests {
    use super::*;

    /// Regression: saturated pops (1.0, 0.99, 0.5) vs target 0.03 must all
    /// receive aggressive downscale — scale < 0.1.
    #[test]
    fn saturated_pops_downscaled_hard() {
        let observed = [1.0, 0.99, 0.5];
        let scales = quiet_start_scale(&observed, 0.03);
        for (i, s) in scales.iter().enumerate() {
            assert!(
                *s < 0.1,
                "pop {i}: r={} → scale={s}, expected < 0.1",
                observed[i]
            );
            assert!(*s >= MIN_QUIET_SCALE, "scale {s} below min {MIN_QUIET_SCALE}");
        }
    }

    /// Regression: healthy pops (0.03, 0.025, 0.035) vs target 0.03 must
    /// receive scales ≈ 1.0 (within 5%).
    #[test]
    fn healthy_pops_preserved() {
        let observed = [0.03, 0.025, 0.035];
        let scales = quiet_start_scale(&observed, 0.03);
        for (i, s) in scales.iter().enumerate() {
            assert!(
                (s - 1.0).abs() < 0.05,
                "pop {i}: r={} → scale={s}, expected ≈ 1.0 (±5%)",
                observed[i]
            );
        }
    }

    /// Quiet pops receive an upscale — but capped so a dead population
    /// doesn't detonate on first step.
    #[test]
    fn quiet_pops_upscaled_capped() {
        let observed = [0.0, 0.001, 0.005];
        let scales = quiet_start_scale(&observed, 0.03);
        for s in &scales {
            assert!(*s >= 1.0, "quiet pop scale {s} should be >= 1.0");
            assert!(*s <= MAX_UPSCALE_CAP, "quiet pop scale {s} over cap");
        }
        // Silent pop hits the cap, not infinity.
        assert_eq!(scales[0], MAX_UPSCALE_CAP);
    }

    /// Invalid target is a no-op: all scales return 1.0.
    #[test]
    fn invalid_target_is_identity() {
        let scales = quiet_start_scale(&[0.1, 0.5, 1.0], 0.0);
        assert_eq!(scales, vec![1.0, 1.0, 1.0]);
        let scales = quiet_start_scale(&[0.1], -0.01);
        assert_eq!(scales, vec![1.0]);
    }

    /// Empty input is handled cleanly.
    #[test]
    fn empty_input() {
        let scales = quiet_start_scale(&[], 0.03);
        assert!(scales.is_empty());
    }

    /// Output length exactly matches input length.
    #[test]
    fn output_shape_matches_input() {
        let observed = [0.03, 0.03, 0.03, 0.03];
        let scales = quiet_start_scale(&observed, 0.03);
        assert_eq!(scales.len(), observed.len());
    }
}
