//! Poisson background noise injection for the SNN.
//!
//! Without persistent background drive, a population that loses its
//! input cascade falls into the absorbing-zero state (`v ≪ v_thresh`,
//! never spikes, R-STDP can't lift weights, dead). Poisson noise is
//! the structural fix — every neuron, every step, has a small chance
//! of receiving a depolarizing pulse, so quiescent neurons drift
//! upward and occasionally spike. Live populations get a small
//! background that doesn't matter.
//!
//! Defaults are V1's tuned values from master commit `2bd4099ff`:
//! **20 Hz × 30 mV** — empirically rescues collapsed populations
//! without visibly perturbing healthy ones. Earlier defaults of
//! 1 Hz × 15 mV were ~50× too weak.
//!
//! # Adaptive scaling (Phase 3.5)
//!
//! The base injection probability is multiplied by a per-population
//! factor in `[0, 1]`: `1.0` when the population is dead (rate=0),
//! `0.0` at-target, linear in between. See [`noise_factor_for_pop`] —
//! ports the adaptive-noise fix from master commit `1a495f51d`. This
//! gives dead populations the full structural rescue while leaving
//! healthy populations untouched.

use rand::Rng;
use rand_chacha::ChaCha20Rng;
use serde::{Deserialize, Serialize};

/// Default Poisson rate (Hz) per neuron. Master commit `2bd4099ff`.
pub const DEFAULT_NOISE_RATE_HZ: f32 = 20.0;
/// Default Poisson pulse amplitude (mV). Master commit `2bd4099ff`.
pub const DEFAULT_NOISE_PULSE_MV: f32 = 30.0;
/// Setter upper bound for `rate_hz`. Wide enough to support exploratory
/// sweeps without silent rejection (master `2bd4099ff` widened this
/// from 100 to 500 after the old range bit operators).
pub const NOISE_RATE_HZ_MAX: f32 = 500.0;
/// Setter upper bound for `pulse_mv`. Widened from 50 to 200 alongside
/// rate (same commit).
pub const NOISE_PULSE_MV_MAX: f32 = 200.0;

/// Per-population Poisson noise configuration.
///
/// Two knobs only. The injection per step per neuron is a Bernoulli draw
/// with probability `rate_hz × dt_ms / 1000`; on success, `pulse_mv`
/// is added to the membrane.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(default)]
pub struct NoiseConfig {
    /// Bernoulli rate per neuron (Hz). Range `[0, NOISE_RATE_HZ_MAX]`.
    pub rate_hz: f32,
    /// Membrane voltage bump on each noise event (mV). Range
    /// `[0, NOISE_PULSE_MV_MAX]`. Use `0.0` to disable injection
    /// without changing the rate.
    pub pulse_mv: f32,
}

impl Default for NoiseConfig {
    fn default() -> Self {
        Self {
            rate_hz: DEFAULT_NOISE_RATE_HZ,
            pulse_mv: DEFAULT_NOISE_PULSE_MV,
        }
    }
}

impl NoiseConfig {
    /// Construct + validate. Out-of-range values are clamped to the
    /// allowed range rather than rejected — the noise system is
    /// best-effort and a bad rate shouldn't fail brain construction.
    #[must_use]
    pub fn new(rate_hz: f32, pulse_mv: f32) -> Self {
        Self {
            rate_hz: rate_hz.clamp(0.0, NOISE_RATE_HZ_MAX),
            pulse_mv: pulse_mv.clamp(0.0, NOISE_PULSE_MV_MAX),
        }
    }

    /// Per-neuron per-step Bernoulli probability for the given timestep.
    #[must_use]
    pub fn bernoulli_p(&self, dt_ms: f32) -> f32 {
        if dt_ms <= 0.0 {
            return 0.0;
        }
        // dt_ms / 1000 → seconds; × rate_hz → expected events per step.
        // For rate × dt_ms small, this is a good approximation to the
        // Poisson probability of ≥1 event.
        (self.rate_hz * dt_ms / 1000.0).clamp(0.0, 1.0)
    }
}

/// Adaptive scaling factor in `[0, 1]` based on observed firing rate
/// vs target. Dead populations (`rate_ema ~= 0`) get full noise
/// (`factor = 1.0`); at-target populations get none (`factor = 0.0`);
/// linear interpolation in between.
///
/// Above target the factor is `0.0` — we don't suppress noise into
/// negatives; if the population is over-firing, the homeostatic +
/// anti-reward + basket layers handle that.
///
/// Port of master commit `1a495f51d` `snn_noise_factor_for_pop()`.
#[must_use]
pub fn noise_factor_for_pop(rate_ema: f32, target_rate: f32) -> f32 {
    if target_rate <= 0.0 {
        // Degenerate target — do nothing rather than divide by zero.
        return 0.0;
    }
    let ratio = rate_ema / target_rate;
    // Linear ramp: ratio=0 → 1.0; ratio=1 → 0.0; ratio>1 → 0.0.
    (1.0 - ratio).clamp(0.0, 1.0)
}

/// Inject Poisson noise into a population's membrane voltages, scaled
/// by `factor` (typically the result of [`noise_factor_for_pop`]).
///
/// Each neuron, each step, has probability `factor × p_base` of
/// receiving a `pulse_mv` bump. RNG state is mutated; the same RNG
/// seed across runs gives bit-identical noise patterns.
///
/// `factor = 0.0` is a fast no-op (skips the per-neuron loop).
pub fn inject_poisson_noise(
    rng: &mut ChaCha20Rng,
    membrane_v: &mut [f32],
    config: &NoiseConfig,
    dt_ms: f32,
    factor: f32,
) {
    let p = config.bernoulli_p(dt_ms) * factor.clamp(0.0, 1.0);
    if p <= 0.0 || config.pulse_mv == 0.0 {
        return;
    }
    let pulse = config.pulse_mv;
    for v in membrane_v.iter_mut() {
        // ChaCha20Rng's `random` returns f32 in [0, 1) uniformly.
        if rng.random::<f32>() < p {
            *v += pulse;
        }
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn default_config_uses_master_2bd4099ff_values() {
        let cfg = NoiseConfig::default();
        assert_eq!(cfg.rate_hz, 20.0);
        assert_eq!(cfg.pulse_mv, 30.0);
    }

    #[test]
    fn new_clamps_out_of_range() {
        let cfg = NoiseConfig::new(-5.0, -10.0);
        assert_eq!(cfg.rate_hz, 0.0);
        assert_eq!(cfg.pulse_mv, 0.0);

        let cfg = NoiseConfig::new(99999.0, 99999.0);
        assert_eq!(cfg.rate_hz, NOISE_RATE_HZ_MAX);
        assert_eq!(cfg.pulse_mv, NOISE_PULSE_MV_MAX);
    }

    #[test]
    fn bernoulli_p_zero_for_zero_dt() {
        let cfg = NoiseConfig::default();
        assert_eq!(cfg.bernoulli_p(0.0), 0.0);
        assert_eq!(cfg.bernoulli_p(-1.0), 0.0);
    }

    #[test]
    fn bernoulli_p_matches_formula() {
        let cfg = NoiseConfig {
            rate_hz: 100.0,
            pulse_mv: 30.0,
        };
        // 100 Hz × 1 ms / 1000 = 0.1
        assert!((cfg.bernoulli_p(1.0) - 0.1).abs() < 1e-6);
    }

    #[test]
    fn bernoulli_p_clamps_to_one() {
        let cfg = NoiseConfig {
            rate_hz: 500.0,
            pulse_mv: 30.0,
        };
        // 500 Hz × 10 ms / 1000 = 5.0 → clamp to 1.0
        assert_eq!(cfg.bernoulli_p(10.0), 1.0);
    }

    #[test]
    fn factor_dead_pop_full_noise() {
        assert_eq!(noise_factor_for_pop(0.0, 0.05), 1.0);
    }

    #[test]
    fn factor_at_target_zero_noise() {
        assert!((noise_factor_for_pop(0.05, 0.05) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn factor_above_target_clamps_to_zero() {
        assert_eq!(noise_factor_for_pop(0.10, 0.05), 0.0);
    }

    #[test]
    fn factor_half_target_half_noise() {
        let f = noise_factor_for_pop(0.025, 0.05);
        assert!((f - 0.5).abs() < 1e-6);
    }

    #[test]
    fn factor_zero_target_returns_zero() {
        assert_eq!(noise_factor_for_pop(0.05, 0.0), 0.0);
        assert_eq!(noise_factor_for_pop(0.05, -0.1), 0.0);
    }

    #[test]
    fn injection_with_zero_factor_no_op() {
        let mut rng = ChaCha20Rng::seed_from_u64(1);
        let mut v = vec![-65.0_f32; 100];
        let cfg = NoiseConfig::default();
        inject_poisson_noise(&mut rng, &mut v, &cfg, 1.0, 0.0);
        assert!(v.iter().all(|&m| m == -65.0));
    }

    #[test]
    fn injection_with_zero_pulse_no_op() {
        let mut rng = ChaCha20Rng::seed_from_u64(1);
        let mut v = vec![-65.0_f32; 100];
        let cfg = NoiseConfig {
            rate_hz: 100.0,
            pulse_mv: 0.0,
        };
        inject_poisson_noise(&mut rng, &mut v, &cfg, 1.0, 1.0);
        assert!(v.iter().all(|&m| m == -65.0));
    }

    #[test]
    fn injection_changes_some_neurons_at_high_rate() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let mut v = vec![-65.0_f32; 1000];
        // 100 Hz × 1 ms = 10% probability — about 100 of 1000 should be hit.
        let cfg = NoiseConfig {
            rate_hz: 100.0,
            pulse_mv: 30.0,
        };
        inject_poisson_noise(&mut rng, &mut v, &cfg, 1.0, 1.0);
        let n_hit = v.iter().filter(|&&m| m > -65.0).count();
        // Allow generous slack — Bernoulli, finite sample.
        assert!(
            (50..200).contains(&n_hit),
            "expected ~100 hits at p=0.1, got {n_hit}"
        );
        // Hits get exactly +pulse_mv.
        for &m in &v {
            assert!(m == -65.0 || (m - (-35.0)).abs() < 1e-5);
        }
    }

    #[test]
    fn injection_is_deterministic_for_same_seed() {
        let mut rng_a = ChaCha20Rng::seed_from_u64(7);
        let mut rng_b = ChaCha20Rng::seed_from_u64(7);
        let cfg = NoiseConfig::default();
        let mut va = vec![-65.0_f32; 200];
        let mut vb = vec![-65.0_f32; 200];
        inject_poisson_noise(&mut rng_a, &mut va, &cfg, 1.0, 1.0);
        inject_poisson_noise(&mut rng_b, &mut vb, &cfg, 1.0, 1.0);
        assert_eq!(va, vb, "same seed must give bit-identical noise");
    }
}
