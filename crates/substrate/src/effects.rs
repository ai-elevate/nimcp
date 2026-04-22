//! Per-network substrate-effect bundles.
//!
//! The substrate produces two structured outputs, one for each side of
//! a neuron's cable model:
//!
//! - [`AxonSubstrateEffects`] — 14 multipliers read by the spike
//!   propagation / refractory / ion-gradient side.
//! - [`DendriteSubstrateEffects`] — 15 multipliers read by the
//!   integration / plasticity / Ca²⁺-handling side.
//!
//! Each struct is pure data. The producer (a `compute_effects` call)
//! owns the conversion from [`NeuralSubstrate`](crate::NeuralSubstrate)
//! chemistry → these multipliers. Consumers (SNN, LNN, CNN, FNO, HNN
//! adapters) multiply their native dynamics by the relevant fields.
//!
//! # Documented ranges
//!
//! Each field comment records V1's documented range. Adapters can
//! assume these bounds after [`compute_effects`](crate::compute::compute_effects)
//! — no NaN, no infinities, no negative gains.

use serde::{Deserialize, Serialize};

/// 14 scalars consumed by the **axon / propagation** side of a neuron.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(default)]
pub struct AxonSubstrateEffects {
    // --- Conduction velocity ---
    /// Q10 temperature effect. Range `[0.5, 1.5]`. `1.0` at 37 °C.
    pub temperature_q10_factor: f32,
    /// ATP effect on conduction velocity. Range `[0, 1]`. `1.0` at full ATP.
    pub atp_velocity_factor: f32,
    /// Myelination ATP efficiency. Range `[0, 1]`. `1.0` with intact myelin.
    pub myelin_efficiency: f32,
    /// Combined velocity multiplier. Range `[0, 1.5]`.
    pub overall_velocity_mod: f32,

    // --- Action potential quality ---
    /// Na⁺/K⁺ gradient quality. Range `[0, 1]`.
    pub ion_gradient_strength: f32,
    /// AP amplitude modulation. Range `[0, 1]`.
    pub ap_amplitude_mod: f32,
    /// Propagation reliability (probability an AP survives an axon
    /// branch point under the current ion gradient). Range `[0, 1]`.
    pub spike_reliability: f32,

    // --- Refractory + pump ---
    /// Na⁺/K⁺-ATPase activity. Range `[0, 1]`.
    pub pump_activity: f32,
    /// Refractory period multiplier. Range `[0.7, 1.3]`.
    /// `> 1.0` = longer refractory under ATP stress.
    pub refractory_period_mod: f32,

    // --- Vesicle transport ---
    /// Vesicle transport efficiency. Range `[0, 1]`.
    pub transport_efficiency: f32,
    /// Motor protein (kinesin) activity. Range `[0, 1]`.
    pub kinesin_activity: f32,

    // --- Membrane properties (axonal) ---
    /// Membrane capacitance multiplier. Range `[0.8, 1.2]`.
    pub membrane_capacitance_mod: f32,
    /// Leak conductance multiplier. Range `[1.0, 3.0]` — degraded
    /// membrane leaks more.
    pub membrane_leak_mod: f32,

    /// Scalar summary of axonal health. Range `[0, 1]`.
    pub overall_capacity: f32,
}

impl Default for AxonSubstrateEffects {
    /// Identity — all multipliers at full-health values. Networks
    /// multiplying by `Default::default()` see no modulation.
    fn default() -> Self {
        Self {
            temperature_q10_factor: 1.0,
            atp_velocity_factor: 1.0,
            myelin_efficiency: 1.0,
            overall_velocity_mod: 1.0,
            ion_gradient_strength: 1.0,
            ap_amplitude_mod: 1.0,
            spike_reliability: 1.0,
            pump_activity: 1.0,
            refractory_period_mod: 1.0,
            transport_efficiency: 1.0,
            kinesin_activity: 1.0,
            membrane_capacitance_mod: 1.0,
            membrane_leak_mod: 1.0,
            overall_capacity: 1.0,
        }
    }
}

impl AxonSubstrateEffects {
    /// Sanity-clamp into the documented ranges. Producer calls this
    /// before returning so consumers can skip defensive checks.
    pub fn clamp(&mut self) {
        self.temperature_q10_factor = self.temperature_q10_factor.clamp(0.5, 1.5);
        self.atp_velocity_factor = self.atp_velocity_factor.clamp(0.0, 1.0);
        self.myelin_efficiency = self.myelin_efficiency.clamp(0.0, 1.0);
        self.overall_velocity_mod = self.overall_velocity_mod.clamp(0.0, 1.5);
        self.ion_gradient_strength = self.ion_gradient_strength.clamp(0.0, 1.0);
        self.ap_amplitude_mod = self.ap_amplitude_mod.clamp(0.0, 1.0);
        self.spike_reliability = self.spike_reliability.clamp(0.0, 1.0);
        self.pump_activity = self.pump_activity.clamp(0.0, 1.0);
        self.refractory_period_mod = self.refractory_period_mod.clamp(0.7, 1.3);
        self.transport_efficiency = self.transport_efficiency.clamp(0.0, 1.0);
        self.kinesin_activity = self.kinesin_activity.clamp(0.0, 1.0);
        self.membrane_capacitance_mod = self.membrane_capacitance_mod.clamp(0.8, 1.2);
        self.membrane_leak_mod = self.membrane_leak_mod.clamp(1.0, 3.0);
        self.overall_capacity = self.overall_capacity.clamp(0.0, 1.0);
    }

    /// True when all fields equal the identity defaults. Adapters can
    /// cheap-check this to skip per-neuron work in the common full-
    /// health case.
    #[must_use]
    #[allow(clippy::float_cmp)]
    pub fn is_identity(&self) -> bool {
        let d = Self::default();
        self.temperature_q10_factor == d.temperature_q10_factor
            && self.atp_velocity_factor == d.atp_velocity_factor
            && self.myelin_efficiency == d.myelin_efficiency
            && self.overall_velocity_mod == d.overall_velocity_mod
            && self.ion_gradient_strength == d.ion_gradient_strength
            && self.ap_amplitude_mod == d.ap_amplitude_mod
            && self.spike_reliability == d.spike_reliability
            && self.pump_activity == d.pump_activity
            && self.refractory_period_mod == d.refractory_period_mod
            && self.transport_efficiency == d.transport_efficiency
            && self.kinesin_activity == d.kinesin_activity
            && self.membrane_capacitance_mod == d.membrane_capacitance_mod
            && self.membrane_leak_mod == d.membrane_leak_mod
            && self.overall_capacity == d.overall_capacity
    }
}

/// 15 scalars consumed by the **dendrite / integration** side of a neuron.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(default)]
pub struct DendriteSubstrateEffects {
    // --- Passive properties ---
    /// Membrane time constant τ_m modulation. Range `[0.5, 1.5]`.
    pub membrane_time_constant_mod: f32,
    /// Space constant λ modulation. Range `[0.5, 1.5]`.
    pub space_constant_mod: f32,
    /// Overall integration efficiency. Range `[0, 1]`.
    pub integration_efficiency: f32,
    /// Voltage attenuation along the dendrite. Range `[1.0, 2.0]`.
    pub attenuation_mod: f32,

    // --- Active properties ---
    /// NMDA Mg²⁺ block sensitivity. Range `[0.8, 1.2]`.
    pub nmda_mg_block_mod: f32,
    /// Spike threshold voltage shift (multiplicative on the resting
    /// threshold distance). Range `[0.8, 1.2]`.
    pub spike_threshold_mod: f32,
    /// Dendritic Na⁺ channel availability. Range `[0, 1]`.
    pub na_channel_availability: f32,

    // --- Calcium handling ---
    /// Ca²⁺ extrusion efficiency. Range `[0, 1]`.
    pub ca_pump_efficiency: f32,
    /// Buffering capacity. Range `[0, 1]`.
    pub ca_buffer_capacity: f32,
    /// Overall Ca²⁺ handling. Range `[0, 1]`.
    pub ca_handling_mod: f32,

    // --- Plasticity ---
    /// LTP induction ability. Range `[0, 1]`.
    pub ltp_capacity: f32,
    /// LTD induction ability. Range `[0, 1]`.
    pub ltd_capacity: f32,
    /// Structural plasticity (spine growth). Range `[0, 1]`.
    pub spine_growth_capacity: f32,
    /// Overall plasticity gate (scalar LR multiplier). Range `[0, 1]`.
    pub plasticity_mod: f32,

    /// Scalar summary of dendritic health. Range `[0, 1]`.
    pub overall_capacity: f32,
}

impl Default for DendriteSubstrateEffects {
    /// Identity — full health, no modulation.
    fn default() -> Self {
        Self {
            membrane_time_constant_mod: 1.0,
            space_constant_mod: 1.0,
            integration_efficiency: 1.0,
            attenuation_mod: 1.0,
            nmda_mg_block_mod: 1.0,
            spike_threshold_mod: 1.0,
            na_channel_availability: 1.0,
            ca_pump_efficiency: 1.0,
            ca_buffer_capacity: 1.0,
            ca_handling_mod: 1.0,
            ltp_capacity: 1.0,
            ltd_capacity: 1.0,
            spine_growth_capacity: 1.0,
            plasticity_mod: 1.0,
            overall_capacity: 1.0,
        }
    }
}

impl DendriteSubstrateEffects {
    /// Clamp into documented ranges.
    pub fn clamp(&mut self) {
        self.membrane_time_constant_mod = self.membrane_time_constant_mod.clamp(0.5, 1.5);
        self.space_constant_mod = self.space_constant_mod.clamp(0.5, 1.5);
        self.integration_efficiency = self.integration_efficiency.clamp(0.0, 1.0);
        self.attenuation_mod = self.attenuation_mod.clamp(1.0, 2.0);
        self.nmda_mg_block_mod = self.nmda_mg_block_mod.clamp(0.8, 1.2);
        self.spike_threshold_mod = self.spike_threshold_mod.clamp(0.8, 1.2);
        self.na_channel_availability = self.na_channel_availability.clamp(0.0, 1.0);
        self.ca_pump_efficiency = self.ca_pump_efficiency.clamp(0.0, 1.0);
        self.ca_buffer_capacity = self.ca_buffer_capacity.clamp(0.0, 1.0);
        self.ca_handling_mod = self.ca_handling_mod.clamp(0.0, 1.0);
        self.ltp_capacity = self.ltp_capacity.clamp(0.0, 1.0);
        self.ltd_capacity = self.ltd_capacity.clamp(0.0, 1.0);
        self.spine_growth_capacity = self.spine_growth_capacity.clamp(0.0, 1.0);
        self.plasticity_mod = self.plasticity_mod.clamp(0.0, 1.0);
        self.overall_capacity = self.overall_capacity.clamp(0.0, 1.0);
    }

    /// Cheap identity check — used by adapters to skip per-neuron
    /// multiplies in the common full-health case.
    #[must_use]
    #[allow(clippy::float_cmp)]
    pub fn is_identity(&self) -> bool {
        let d = Self::default();
        self.membrane_time_constant_mod == d.membrane_time_constant_mod
            && self.space_constant_mod == d.space_constant_mod
            && self.integration_efficiency == d.integration_efficiency
            && self.attenuation_mod == d.attenuation_mod
            && self.nmda_mg_block_mod == d.nmda_mg_block_mod
            && self.spike_threshold_mod == d.spike_threshold_mod
            && self.na_channel_availability == d.na_channel_availability
            && self.ca_pump_efficiency == d.ca_pump_efficiency
            && self.ca_buffer_capacity == d.ca_buffer_capacity
            && self.ca_handling_mod == d.ca_handling_mod
            && self.ltp_capacity == d.ltp_capacity
            && self.ltd_capacity == d.ltd_capacity
            && self.spine_growth_capacity == d.spine_growth_capacity
            && self.plasticity_mod == d.plasticity_mod
            && self.overall_capacity == d.overall_capacity
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn axon_default_is_identity() {
        let a = AxonSubstrateEffects::default();
        assert!(a.is_identity());
        assert_eq!(a.temperature_q10_factor, 1.0);
        assert_eq!(a.overall_capacity, 1.0);
    }

    #[test]
    fn dendrite_default_is_identity() {
        let d = DendriteSubstrateEffects::default();
        assert!(d.is_identity());
        assert_eq!(d.plasticity_mod, 1.0);
    }

    #[test]
    fn axon_clamp_enforces_ranges() {
        let mut a = AxonSubstrateEffects {
            refractory_period_mod: 10.0,
            membrane_leak_mod: 0.1,
            ..Default::default()
        };
        a.clamp();
        assert_eq!(a.refractory_period_mod, 1.3);
        assert_eq!(a.membrane_leak_mod, 1.0);
    }

    #[test]
    fn dendrite_clamp_enforces_ranges() {
        let mut d = DendriteSubstrateEffects {
            membrane_time_constant_mod: 3.0,
            attenuation_mod: 0.1,
            ..Default::default()
        };
        d.clamp();
        assert_eq!(d.membrane_time_constant_mod, 1.5);
        assert_eq!(d.attenuation_mod, 1.0);
    }

    #[test]
    fn non_default_is_not_identity() {
        let mut a = AxonSubstrateEffects::default();
        a.spike_reliability = 0.9;
        assert!(!a.is_identity());
    }
}
