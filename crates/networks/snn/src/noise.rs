//! Background noise injection for the SNN.
//!
//! Without persistent background drive, a population that loses its
//! input cascade falls into the absorbing-zero state (`v ≪ v_thresh`,
//! never spikes, R-STDP can't lift weights, dead). Background noise is
//! the structural fix — every neuron, every step, has a small chance
//! of receiving a depolarizing pulse, so quiescent neurons drift
//! upward and occasionally spike. Live populations get a small
//! background that doesn't matter.
//!
//! Defaults are V1's tuned values from master commit `2bd4099ff`:
//! **20 Hz × 30 mV** Poisson — empirically rescues collapsed populations
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
//!
//! # Alternative sources — 1/f (pink) noise
//!
//! Opt-in via [`NoiseKind::Pink`]. Pink noise preserves the rescue
//! property (non-zero mean perturbation over any window) while adding
//! temporal correlations that better match the 1/f spectrum observed
//! in cortical LFPs. Selecting pink routes each per-neuron bump through
//! a shared [`nimcp_pinknoise::PinkNoiseGen`] sample and stashes the
//! sample into a rolling ring buffer; [`SnnNetwork::pink_alpha`] then
//! runs `nimcp_fractal::dfa_alpha` over that buffer to verify the 1/f
//! property online.
//!
//! The Poisson path is bit-identical to pre-pink behavior — selecting
//! [`NoiseKind::Poisson`] (the `Default`) preserves every existing test.

use nimcp_pinknoise::PinkNoiseGen;
use rand::Rng;
use rand_chacha::ChaCha20Rng;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

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
/// Default octave count for [`NoiseKind::Pink`]. Matches V1's production
/// default (see [`nimcp_pinknoise::DEFAULT_N_OCTAVES`]).
pub const DEFAULT_PINK_N_OCTAVES: u32 = 16;
/// Ring-buffer capacity used by [`PinkNoiseState`] for online DFA.
/// 1024 samples is wide enough to span ~3 decades for
/// `dfa_alpha(&ring, 8, 256)` while staying cheap per-step.
pub const PINK_RING_CAPACITY: usize = 1024;

/// Which background-noise algorithm drives a population.
///
/// `Default` returns [`NoiseKind::Poisson`] — the pre-pink behavior —
/// so `NoiseConfig::default()` stays bit-identical to pre-port code.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum NoiseKind {
    /// Classic Bernoulli-per-step Poisson pulses (V1 default).
    #[default]
    Poisson,
    /// 1/f (Voss-McCartney) pink noise. `n_octaves` is forwarded to
    /// [`nimcp_pinknoise::PinkNoiseGen::new`] — clamped to
    /// `[1, nimcp_pinknoise::MAX_N_OCTAVES]`. Typical values: 8–16.
    Pink {
        /// Number of Voss-McCartney sources. More octaves → cleaner 1/f
        /// over more decades, but slightly more work per step.
        n_octaves: u32,
    },
}

/// Per-population background-noise configuration.
///
/// Three knobs. For the [`NoiseKind::Poisson`] path (the default), the
/// injection per step per neuron is a Bernoulli draw with probability
/// `rate_hz × dt_ms / 1000`; on success, `pulse_mv` is added to the
/// membrane. For [`NoiseKind::Pink`], each step draws one sample from
/// the shared [`nimcp_pinknoise::PinkNoiseGen`] and adds
/// `sample × pulse_mv × factor` to every neuron's membrane; `rate_hz`
/// is unused by that path but preserved in the config so flipping
/// `kind` round-trips cleanly.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(default)]
pub struct NoiseConfig {
    /// Poisson-path Bernoulli rate per neuron (Hz). Range
    /// `[0, NOISE_RATE_HZ_MAX]`. Unused when `kind == Pink`.
    pub rate_hz: f32,
    /// Membrane voltage bump (mV) — scalar multiplier in both paths.
    /// Range `[0, NOISE_PULSE_MV_MAX]`. Use `0.0` to disable injection
    /// without changing the rate/kind.
    pub pulse_mv: f32,
    /// Noise algorithm. Defaults to [`NoiseKind::Poisson`] — preserves
    /// every pre-pink call-site behavior bit-for-bit.
    #[serde(default)]
    pub kind: NoiseKind,
}

impl Default for NoiseConfig {
    fn default() -> Self {
        Self {
            rate_hz: DEFAULT_NOISE_RATE_HZ,
            pulse_mv: DEFAULT_NOISE_PULSE_MV,
            kind: NoiseKind::default(),
        }
    }
}

impl NoiseConfig {
    /// Construct + validate a [`NoiseKind::Poisson`] config. Out-of-range
    /// values are clamped to the allowed range rather than rejected —
    /// the noise system is best-effort and a bad rate shouldn't fail
    /// brain construction.
    #[must_use]
    pub fn new(rate_hz: f32, pulse_mv: f32) -> Self {
        Self {
            rate_hz: rate_hz.clamp(0.0, NOISE_RATE_HZ_MAX),
            pulse_mv: pulse_mv.clamp(0.0, NOISE_PULSE_MV_MAX),
            kind: NoiseKind::Poisson,
        }
    }

    /// Construct a [`NoiseKind::Pink`] config with the given octave count
    /// and pulse amplitude. `rate_hz` is set to the default Poisson rate
    /// so round-tripping through JSON preserves a sensible fallback,
    /// but the pink path ignores it.
    #[must_use]
    pub fn new_pink(n_octaves: u32, pulse_mv: f32) -> Self {
        Self {
            rate_hz: DEFAULT_NOISE_RATE_HZ,
            pulse_mv: pulse_mv.clamp(0.0, NOISE_PULSE_MV_MAX),
            kind: NoiseKind::Pink { n_octaves },
        }
    }

    /// Per-neuron per-step Bernoulli probability for the given timestep.
    /// Only meaningful for the Poisson path.
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

/// Runtime state for a [`NoiseKind::Pink`] population.
///
/// Bundles a [`PinkNoiseGen`] (one-sample-per-step source) with a
/// rolling ring buffer of the last [`PINK_RING_CAPACITY`] samples. The
/// ring feeds [`SnnNetwork::pink_alpha`], which runs
/// `nimcp_fractal::dfa_alpha` over it for online spectrum health
/// monitoring (α ≈ 1.0 confirms 1/f is actually being produced).
#[derive(Debug, Clone)]
pub struct PinkNoiseState {
    /// The underlying generator. Owns its RNG.
    pub generator: PinkNoiseGen,
    /// Bounded ring of the most recent samples (oldest at front).
    /// Capacity is fixed at [`PINK_RING_CAPACITY`]; pushes past capacity
    /// evict the oldest sample.
    pub ring: VecDeque<f32>,
}

impl PinkNoiseState {
    /// Construct with `n_octaves` and a deterministic seed. The ring
    /// is empty until the first step; [`SnnNetwork::pink_alpha`] only
    /// returns `Some` once it has filled to [`PINK_RING_CAPACITY`].
    #[must_use]
    pub fn new(n_octaves: u32, seed: u64) -> Self {
        Self {
            generator: PinkNoiseGen::new(n_octaves, seed),
            ring: VecDeque::with_capacity(PINK_RING_CAPACITY),
        }
    }

    /// Advance the generator once, push the sample into the ring
    /// (evicting the oldest if full), and return the sample.
    pub fn step_sample(&mut self) -> f32 {
        let s = self.generator.step();
        if self.ring.len() == PINK_RING_CAPACITY {
            self.ring.pop_front();
        }
        self.ring.push_back(s);
        s
    }

    /// Number of samples buffered so far (saturates at
    /// [`PINK_RING_CAPACITY`]).
    #[must_use]
    pub fn len(&self) -> usize {
        self.ring.len()
    }

    /// `true` while the ring still has room.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ring.is_empty()
    }

    /// `true` once the ring has reached [`PINK_RING_CAPACITY`].
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.ring.len() >= PINK_RING_CAPACITY
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

/// Inject one pink-noise sample, scaled by `pulse_mv × factor`, into
/// every neuron's membrane voltage, and record the raw sample into
/// the state's ring buffer.
///
/// Unlike the Poisson path, every neuron receives the same increment
/// on every step — the perturbation's structure comes from the 1/f
/// temporal correlation rather than per-neuron Bernoulli draws. This
/// matches V1's `nimcp_pink_noise_inject_population` contract.
///
/// `factor = 0.0` still advances the generator so the ring continues
/// to fill — monitoring must work on quiet populations — but skips
/// the per-neuron membrane loop.
pub fn inject_pink_noise(
    pink: &mut PinkNoiseState,
    membrane_v: &mut [f32],
    pulse_mv: f32,
    dt_ms: f32,
    factor: f32,
) {
    if dt_ms <= 0.0 {
        return;
    }
    let sample = pink.step_sample();
    let scaled_factor = factor.clamp(0.0, 1.0);
    if pulse_mv == 0.0 || scaled_factor == 0.0 {
        return;
    }
    let bump = sample * pulse_mv * scaled_factor;
    if bump == 0.0 {
        return;
    }
    for v in membrane_v.iter_mut() {
        *v += bump;
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
            kind: NoiseKind::Poisson,
        };
        // 100 Hz × 1 ms / 1000 = 0.1
        assert!((cfg.bernoulli_p(1.0) - 0.1).abs() < 1e-6);
    }

    #[test]
    fn bernoulli_p_clamps_to_one() {
        let cfg = NoiseConfig {
            rate_hz: 500.0,
            pulse_mv: 30.0,
            kind: NoiseKind::Poisson,
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
            kind: NoiseKind::Poisson,
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
            kind: NoiseKind::Poisson,
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

    // -----------------------------------------------------------------
    // Pink-noise unit tests (state machinery only — integration with
    // SnnNetwork::step is exercised from network.rs tests).
    // -----------------------------------------------------------------

    #[test]
    fn noise_kind_default_is_poisson() {
        // Backward-compat contract: old code that relied on
        // `NoiseConfig::default()` being Poisson must still observe it.
        assert_eq!(NoiseKind::default(), NoiseKind::Poisson);
        assert_eq!(NoiseConfig::default().kind, NoiseKind::Poisson);
    }

    #[test]
    fn new_pink_constructor_sets_kind() {
        let cfg = NoiseConfig::new_pink(16, 30.0);
        match cfg.kind {
            NoiseKind::Pink { n_octaves } => assert_eq!(n_octaves, 16),
            other => panic!("expected Pink, got {other:?}"),
        }
        assert_eq!(cfg.pulse_mv, 30.0);
    }

    #[test]
    fn pink_state_ring_fills_and_bounds_at_capacity() {
        let mut s = PinkNoiseState::new(8, 0xDEAD_BEEF);
        assert!(s.is_empty());
        for _ in 0..PINK_RING_CAPACITY {
            s.step_sample();
        }
        assert!(s.is_full());
        assert_eq!(s.len(), PINK_RING_CAPACITY);
        // Pushing past capacity must evict the oldest, keeping length fixed.
        for _ in 0..10 {
            s.step_sample();
        }
        assert_eq!(s.len(), PINK_RING_CAPACITY);
    }

    #[test]
    fn inject_pink_zero_pulse_leaves_membrane_unchanged() {
        let mut s = PinkNoiseState::new(8, 1);
        let mut v = vec![-65.0_f32; 64];
        inject_pink_noise(&mut s, &mut v, 0.0, 1.0, 1.0);
        // Zero pulse → membrane untouched. Sample is still recorded
        // into the ring so monitoring continues to work.
        assert!(v.iter().all(|&m| m == -65.0));
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn inject_pink_zero_factor_advances_ring_but_not_membrane() {
        // Quiet populations (factor=0) must still fill the ring so
        // pink_alpha keeps working — see doc comment on inject_pink_noise.
        let mut s = PinkNoiseState::new(8, 2);
        let mut v = vec![-65.0_f32; 32];
        inject_pink_noise(&mut s, &mut v, 30.0, 1.0, 0.0);
        assert!(v.iter().all(|&m| m == -65.0));
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn inject_pink_applies_same_bump_to_all_neurons() {
        // Per-step, every neuron gets the same additive sample×pulse
        // perturbation. Matches V1's `inject_population` contract.
        let mut s = PinkNoiseState::new(8, 3);
        let mut v = vec![-65.0_f32; 16];
        inject_pink_noise(&mut s, &mut v, 30.0, 1.0, 1.0);
        let first = v[0];
        assert!(v.iter().all(|&m| (m - first).abs() < 1e-6));
    }

    #[test]
    fn inject_pink_is_deterministic_for_same_seed() {
        let mut a = PinkNoiseState::new(8, 7);
        let mut b = PinkNoiseState::new(8, 7);
        let mut va = vec![-65.0_f32; 200];
        let mut vb = vec![-65.0_f32; 200];
        for _ in 0..50 {
            inject_pink_noise(&mut a, &mut va, 30.0, 1.0, 1.0);
            inject_pink_noise(&mut b, &mut vb, 30.0, 1.0, 1.0);
        }
        assert_eq!(va, vb);
        assert_eq!(
            a.ring.iter().copied().collect::<Vec<_>>(),
            b.ring.iter().copied().collect::<Vec<_>>()
        );
    }
}
