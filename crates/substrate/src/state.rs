//! Per-region substrate state.
//!
//! One [`NeuralSubstrate`] covers one chemistry compartment — typically
//! one brain region on V1, but finer granularity is allowed. The
//! struct bundles three logical groups:
//!
//! - **Chemistry**: ATP, oxygen, glucose, metabolic rate / recovery.
//!   Time-evolving outputs from the chemistry tick.
//! - **Physical**: temperature, membrane integrity, ion balance,
//!   Na+/K+ pump, Ca²⁺ homeostasis. Time-evolving outputs from the
//!   cell physics tick.
//! - **Modulation summary**: derived scalars a network can read
//!   directly if it doesn't need the full effect bundle (compact
//!   telemetry path).
//!
//! Default values represent a fully-healthy cortical compartment at
//! body temperature — [`NeuralSubstrate::default()`] returns identity
//! substrate, and [`compute_effects`](crate::compute::compute_effects)
//! on it emits all-ones multipliers.

use serde::{Deserialize, Serialize};

/// Rest temperature in degrees Celsius — body-core reference.
pub const REST_TEMPERATURE_C: f32 = 37.0;

/// Per-region substrate state.
///
/// Fields are plain scalars in `[0, 1]` unless otherwise noted. A fresh
/// instance (via [`Self::default`] or [`Self::new`]) is in the
/// full-health attractor.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(default)]
pub struct NeuralSubstrate {
    // Chemistry.
    /// ATP concentration normalized to [0, 1]. 1.0 = full; 0.0 = depleted.
    pub atp_level: f32,
    /// Oxygen saturation in [0, 1].
    pub oxygen_saturation: f32,
    /// Glucose availability in [0, 1].
    pub glucose_level: f32,
    /// Current metabolic rate (arbitrary units; used by debit to
    /// estimate ATP consumption cadence).
    pub metabolic_rate: f32,
    /// Recovery rate — how quickly ATP / glucose refill on the next
    /// chemistry tick.
    pub recovery_rate: f32,

    // Physical.
    /// Temperature in degrees Celsius.
    pub temperature: f32,
    /// Membrane health in [0, 1] — degraded by sustained activity or
    /// metabolic stress.
    pub membrane_integrity: f32,
    /// Ion gradient quality in [0, 1].
    pub ion_balance: f32,
    /// Na+/K+-ATPase activity in [0, 1].
    pub na_k_pump_activity: f32,
    /// Ca2+ regulation quality in [0, 1].
    pub ca_homeostasis: f32,

    // Glial / myelin summaries (compact; full glial state lives in a
    // sibling crate once we port it).
    /// Myelin sheath integrity in [0, 1]. Determines conduction velocity.
    pub myelin_integrity: f32,
    /// Astrocytic support in [0, 1]. Buffers ions / recycles glutamate.
    pub astrocytic_support: f32,

    /// Serializable region identifier — the single piece of identity
    /// that survives a checkpoint. Pointers (thalamic channel, router)
    /// are rebuilt at load time, same pattern as the Wave A+B
    /// adaptation / basket states.
    pub region_id: u32,
}

impl NeuralSubstrate {
    /// Construct a full-health substrate for a given region.
    #[must_use]
    pub fn new(region_id: u32) -> Self {
        Self {
            region_id,
            ..Self::default()
        }
    }

    /// Clamp every field into its documented range. Called automatically
    /// by [`crate::compute::compute_effects`] before reading, so
    /// out-of-range inputs never propagate.
    pub fn clamp(&mut self) {
        self.atp_level = self.atp_level.clamp(0.0, 1.0);
        self.oxygen_saturation = self.oxygen_saturation.clamp(0.0, 1.0);
        self.glucose_level = self.glucose_level.clamp(0.0, 1.0);
        self.metabolic_rate = self.metabolic_rate.clamp(0.0, 10.0);
        self.recovery_rate = self.recovery_rate.clamp(0.0, 10.0);
        self.temperature = self.temperature.clamp(25.0, 45.0);
        self.membrane_integrity = self.membrane_integrity.clamp(0.0, 1.0);
        self.ion_balance = self.ion_balance.clamp(0.0, 1.0);
        self.na_k_pump_activity = self.na_k_pump_activity.clamp(0.0, 1.0);
        self.ca_homeostasis = self.ca_homeostasis.clamp(0.0, 1.0);
        self.myelin_integrity = self.myelin_integrity.clamp(0.0, 1.0);
        self.astrocytic_support = self.astrocytic_support.clamp(0.0, 1.0);
    }
}

impl Default for NeuralSubstrate {
    /// Full-health substrate at body temperature.
    fn default() -> Self {
        Self {
            atp_level: 1.0,
            oxygen_saturation: 1.0,
            glucose_level: 1.0,
            metabolic_rate: 1.0,
            recovery_rate: 1.0,
            temperature: REST_TEMPERATURE_C,
            membrane_integrity: 1.0,
            ion_balance: 1.0,
            na_k_pump_activity: 1.0,
            ca_homeostasis: 1.0,
            myelin_integrity: 1.0,
            astrocytic_support: 1.0,
            region_id: 0,
        }
    }
}

/// Tuning knobs for substrate dynamics (debit costs + recovery rates).
/// Defaults match V1's literature-derived costs.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(default)]
pub struct NeuralSubstrateConfig {
    /// ATP cost per spike (fraction of `atp_level`). V1 default `1e-6`
    /// — millions of spikes to deplete a fresh region.
    pub atp_cost_per_spike: f32,
    /// ATP cost per plasticity update — roughly 10× a spike (STDP is
    /// AMPAR insertion, ~expensive).
    pub atp_cost_per_plasticity: f32,
    /// Ion-gradient degradation per spike.
    pub ion_cost_per_spike: f32,
    /// Membrane wear per spike.
    pub membrane_cost_per_spike: f32,
    /// Default ATP recovery per `debit_activity` call (offsets
    /// activity-driven depletion; callers tuning the chemistry tick
    /// set this to zero).
    pub atp_passive_recovery: f32,
}

impl Default for NeuralSubstrateConfig {
    fn default() -> Self {
        Self {
            atp_cost_per_spike: 1.0e-6,
            atp_cost_per_plasticity: 1.0e-5,
            ion_cost_per_spike: 2.0e-7,
            membrane_cost_per_spike: 1.0e-8,
            atp_passive_recovery: 1.0e-4,
        }
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn default_is_full_health() {
        let s = NeuralSubstrate::default();
        assert_eq!(s.atp_level, 1.0);
        assert_eq!(s.oxygen_saturation, 1.0);
        assert_eq!(s.temperature, REST_TEMPERATURE_C);
        assert_eq!(s.membrane_integrity, 1.0);
    }

    #[test]
    fn new_preserves_region_id_and_defaults_rest() {
        let s = NeuralSubstrate::new(42);
        assert_eq!(s.region_id, 42);
        assert_eq!(s.atp_level, 1.0);
    }

    #[test]
    fn clamp_reigns_in_out_of_range() {
        let mut s = NeuralSubstrate {
            atp_level: 2.0,
            oxygen_saturation: -0.5,
            temperature: 100.0,
            ..Default::default()
        };
        s.clamp();
        assert_eq!(s.atp_level, 1.0);
        assert_eq!(s.oxygen_saturation, 0.0);
        assert_eq!(s.temperature, 45.0);
    }
}
