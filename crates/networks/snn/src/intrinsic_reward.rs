//! Intrinsic reward — firing-rate-target-based R-STDP teaching signal.
//!
//! The base reward is a **Gaussian** centred on each population's
//! `target_rate`: peak `1.0` when `rate_ema == target`, falling off
//! with width `sigma = 0.5 × target`. Use this as the primary R-STDP
//! signal so SNN learning is decoupled from ANN-loss signal — the SNN
//! rewards itself when its populations fire close to their targets.
//!
//! # Anti-reward
//!
//! Port of master commit `cf793e0f0 feat(snn): anti-reward`.
//!
//! The Gaussian bottoms out at `0` once `rate_ema >> target`, which
//! leaves R-STDP powerless against saturation (`Δw = lr · r · e → 0`
//! when `r = 0`). The anti-reward branch subtracts a linear penalty
//! when `rate_ema > threshold_ratio × target`, so reward goes
//! **negative** and R-STDP can push saturated pathways down.
//!
//! Penalty: `r -= gain · (rate − thr_ratio · target) / target`.
//!
//! # Multi-population aggregation
//!
//! Networks with multiple populations average the per-population
//! rewards. Callers pass the iterator of `(rate_ema, target)` pairs;
//! this module doesn't care whether it's input pops, cascade pops,
//! or both.

use serde::{Deserialize, Serialize};

/// Default: anti-reward is on at construction. Match master `cf793e0f0`.
pub const DEFAULT_ANTI_REWARD_ENABLED: bool = true;
/// Default `threshold_ratio` — penalty engages above `2 × target_rate`.
pub const DEFAULT_ANTI_REWARD_THRESHOLD_RATIO: f32 = 2.0;
/// Default anti-reward gain — linear penalty slope.
pub const DEFAULT_ANTI_REWARD_GAIN: f32 = 0.5;
/// Default Gaussian `sigma` as a fraction of `target_rate`. Wide enough
/// to give substantial reward within ±target of the set-point.
pub const DEFAULT_GAUSSIAN_SIGMA_RATIO: f32 = 0.5;

/// Inclusive min for `threshold_ratio`. Below this, the trigger would
/// engage at / before target — meaningless.
pub const ANTI_REWARD_THRESHOLD_RATIO_MIN: f32 = 1.0;
/// Inclusive max for `threshold_ratio`.
pub const ANTI_REWARD_THRESHOLD_RATIO_MAX: f32 = 10.0;
/// Inclusive min for `gain`. Zero disables anti-reward without
/// flipping `enabled`.
pub const ANTI_REWARD_GAIN_MIN: f32 = 0.0;
/// Inclusive max for `gain`.
pub const ANTI_REWARD_GAIN_MAX: f32 = 10.0;

/// Configuration for the per-population intrinsic reward computation.
///
/// All fields are `#[serde(default)]`-friendly — a sparse JSON config
/// picks sane V1-tuned defaults for every field.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(default)]
pub struct IntrinsicRewardConfig {
    /// `sigma` for the Gaussian, as a fraction of `target_rate`.
    /// Default `0.5` — wide enough that rates within ±target give
    /// substantial reward, narrow enough to fall off at 2×target.
    pub gaussian_sigma_ratio: f32,
    /// Whether the anti-reward branch applies a penalty above
    /// `threshold_ratio × target`.
    pub anti_reward_enabled: bool,
    /// Multiplier on `target_rate` above which the anti-reward
    /// penalty engages. Default `2.0`, clamped to
    /// `(ANTI_REWARD_THRESHOLD_RATIO_MIN, ANTI_REWARD_THRESHOLD_RATIO_MAX)`.
    pub anti_reward_threshold_ratio: f32,
    /// Linear penalty gain. Default `0.5`, clamped to
    /// `[ANTI_REWARD_GAIN_MIN, ANTI_REWARD_GAIN_MAX)`.
    pub anti_reward_gain: f32,
}

impl Default for IntrinsicRewardConfig {
    fn default() -> Self {
        Self {
            gaussian_sigma_ratio: DEFAULT_GAUSSIAN_SIGMA_RATIO,
            anti_reward_enabled: DEFAULT_ANTI_REWARD_ENABLED,
            anti_reward_threshold_ratio: DEFAULT_ANTI_REWARD_THRESHOLD_RATIO,
            anti_reward_gain: DEFAULT_ANTI_REWARD_GAIN,
        }
    }
}

impl IntrinsicRewardConfig {
    /// Clamp out-of-range fields to the V1-documented valid ranges.
    /// Silent — out-of-range values are bugs in configuration, but
    /// clamping is nicer than refusing to construct.
    pub fn clamp(&mut self) {
        if self.gaussian_sigma_ratio <= 0.0 {
            self.gaussian_sigma_ratio = DEFAULT_GAUSSIAN_SIGMA_RATIO;
        }
        self.anti_reward_threshold_ratio = self.anti_reward_threshold_ratio.clamp(
            ANTI_REWARD_THRESHOLD_RATIO_MIN,
            ANTI_REWARD_THRESHOLD_RATIO_MAX,
        );
        self.anti_reward_gain = self
            .anti_reward_gain
            .clamp(ANTI_REWARD_GAIN_MIN, ANTI_REWARD_GAIN_MAX);
    }
}

/// Per-population intrinsic reward.
///
/// Returns a scalar in roughly `[-large, 1.0]`:
/// - `1.0` at `rate_ema == target`.
/// - Gaussian decay around the target (width = `sigma_ratio × target`).
/// - If anti-reward is on and `rate > threshold_ratio × target`,
///   subtract `gain × (rate − thr_ratio × target) / target`. This can
///   push the return value arbitrarily negative for extreme saturation.
#[must_use]
pub fn compute_per_pop_reward(rate_ema: f32, target: f32, cfg: &IntrinsicRewardConfig) -> f32 {
    if target <= 0.0 {
        return 0.0;
    }
    let sigma = target * cfg.gaussian_sigma_ratio;
    if sigma <= 0.0 {
        return 0.0;
    }
    let two_sigma_sq = 2.0 * sigma * sigma;
    let err = rate_ema - target;
    let mut r = (-(err * err) / two_sigma_sq).exp();

    if cfg.anti_reward_enabled {
        let trigger = cfg.anti_reward_threshold_ratio * target;
        if rate_ema > trigger {
            r -= cfg.anti_reward_gain * (rate_ema - trigger) / target;
        }
    }
    r
}

/// Average intrinsic reward across multiple populations. Populations
/// with `target <= 0` are skipped (they don't contribute). Returns
/// `0.0` if no population contributes (e.g. every target is zero).
///
/// Input is an iterator of `(rate_ema, target)` pairs — the caller is
/// free to pull from any source (per-pop rate EMA, per-pop target from
/// the pop name prefix, etc.).
#[must_use]
pub fn compute_network_reward<I>(pops: I, cfg: &IntrinsicRewardConfig) -> f32
where
    I: IntoIterator<Item = (f32, f32)>,
{
    let mut sum = 0.0_f32;
    let mut n = 0_u32;
    for (rate, target) in pops {
        if target <= 0.0 {
            continue;
        }
        sum += compute_per_pop_reward(rate, target, cfg);
        n += 1;
    }
    #[allow(clippy::cast_precision_loss)]
    if n > 0 { sum / n as f32 } else { 0.0 }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    fn default_cfg() -> IntrinsicRewardConfig {
        IntrinsicRewardConfig::default()
    }

    #[test]
    fn at_target_peak_reward_one() {
        let r = compute_per_pop_reward(0.05, 0.05, &default_cfg());
        assert!((r - 1.0).abs() < 1e-6);
    }

    #[test]
    fn zero_target_returns_zero() {
        let r = compute_per_pop_reward(0.1, 0.0, &default_cfg());
        assert_eq!(r, 0.0);
    }

    #[test]
    fn gaussian_decays_symmetrically() {
        let cfg = default_cfg();
        let below = compute_per_pop_reward(0.025, 0.05, &cfg);
        let above = compute_per_pop_reward(0.075, 0.05, &cfg);
        // Below-target doesn't trigger anti-reward, so the two are
        // mirror images of the Gaussian.
        assert!((below - above).abs() < 1e-6);
        assert!(below < 1.0);
    }

    #[test]
    fn anti_reward_triggers_above_threshold() {
        let cfg = default_cfg(); // threshold_ratio = 2.0, gain = 0.5
        let target = 0.05;
        // 2× threshold = 0.10. Above that: anti-reward kicks in.
        let saturated = compute_per_pop_reward(0.20, target, &cfg);
        // Gaussian at rate=0.20, target=0.05, sigma=0.025 →
        // err=0.15, two_sigma_sq=0.00125, err^2=0.0225, ratio=18 →
        // exp(-18) ≈ 1.5e-8 ≈ 0.
        // Anti-reward: -0.5 × (0.20 - 0.10) / 0.05 = -1.0
        assert!(
            saturated < -0.5,
            "saturated reward must be strongly negative, got {saturated}"
        );
    }

    #[test]
    fn anti_reward_disabled_keeps_gaussian_only() {
        let mut cfg = default_cfg();
        cfg.anti_reward_enabled = false;
        let r = compute_per_pop_reward(0.20, 0.05, &cfg);
        // Gaussian only — approaches 0 from above.
        assert!((0.0..0.001).contains(&r));
    }

    #[test]
    fn just_below_threshold_no_penalty() {
        let cfg = default_cfg();
        let target = 0.05;
        // Right at 2× target: penalty term is 0 (rate - 2×target == 0).
        // Just below: the Gaussian is slightly larger; penalty
        // discontinuity is zero. So r_below > r_at by a small amount,
        // and the function is continuous from the left at threshold.
        let r_at = compute_per_pop_reward(0.10, target, &cfg);
        let r_just_above = compute_per_pop_reward(0.1001, target, &cfg);
        // Right at threshold == right above (penalty is linear in excess).
        assert!((r_at - r_just_above).abs() < 0.01);
    }

    #[test]
    fn network_average_matches_per_pop() {
        let cfg = default_cfg();
        let pops = [(0.05, 0.05), (0.05, 0.05), (0.05, 0.05)];
        let avg = compute_network_reward(pops, &cfg);
        assert!((avg - 1.0).abs() < 1e-6);
    }

    #[test]
    fn network_skips_zero_target_pops() {
        let cfg = default_cfg();
        let pops = [(0.05, 0.05), (0.10, 0.0), (0.05, 0.05)];
        let avg = compute_network_reward(pops, &cfg);
        // Only two valid pops, both at target → 1.0 average.
        assert!((avg - 1.0).abs() < 1e-6);
    }

    #[test]
    fn network_empty_returns_zero() {
        let cfg = default_cfg();
        let pops: [(f32, f32); 0] = [];
        assert_eq!(compute_network_reward(pops, &cfg), 0.0);
    }

    #[test]
    fn clamp_rejects_out_of_range_values() {
        let mut cfg = IntrinsicRewardConfig {
            gaussian_sigma_ratio: -0.5, // invalid — reset to default
            anti_reward_enabled: true,
            anti_reward_threshold_ratio: 0.5, // below min → MIN
            anti_reward_gain: 20.0,           // above max → MAX
        };
        cfg.clamp();
        assert_eq!(cfg.gaussian_sigma_ratio, DEFAULT_GAUSSIAN_SIGMA_RATIO);
        assert_eq!(
            cfg.anti_reward_threshold_ratio,
            ANTI_REWARD_THRESHOLD_RATIO_MIN
        );
        assert_eq!(cfg.anti_reward_gain, ANTI_REWARD_GAIN_MAX);
    }

    #[test]
    fn anti_reward_scales_linearly_with_gain() {
        let mut cfg = default_cfg();
        cfg.anti_reward_gain = 0.5;
        let r_half = compute_per_pop_reward(0.20, 0.05, &cfg);
        cfg.anti_reward_gain = 1.0;
        let r_full = compute_per_pop_reward(0.20, 0.05, &cfg);
        // Difference between the two is just the extra penalty:
        // extra = 0.5 × (0.20 - 0.10) / 0.05 = 1.0
        let diff = r_half - r_full;
        assert!((diff - 1.0).abs() < 1e-6);
    }
}
