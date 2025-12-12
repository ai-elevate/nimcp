/**
 * @file nimcp_axon_dendrite_substrate_bridge.h
 * @brief Axon-Dendrite Neural Substrate Integration Bridge
 * @version 1.0.0
 * @date 2025-12-12
 *
 * WHAT: Bidirectional integration between neural substrate and axon/dendrite modules
 * WHY:  Substrate conditions (ATP, temperature, ions) directly affect signal propagation,
 *       integration, and plasticity in axons and dendrites through biophysical mechanisms.
 * HOW:  Substrate modulates axonal conduction and dendritic integration; axonal/dendritic
 *       activity depletes substrate resources.
 *
 * BIOLOGICAL BASIS:
 * ==================================================================================
 *
 * SUBSTRATE → AXON PATHWAYS:
 * ---------------------------
 * 1. Conduction Velocity Modulation:
 *    - Temperature: Q10 coefficient (2-3 for channel kinetics)
 *    - Myelination efficiency requires ATP for Na+/K+-ATPase
 *    - Hodgkin-Huxley: velocity ∝ exp(ΔT/10°C * ln(Q10))
 *    - Hypoxia/ATP depletion → failed action potential propagation
 *    - Reference: Hodgkin & Huxley (1952) temperature coefficients
 *
 * 2. Action Potential Reliability:
 *    - Na+/K+ gradient maintenance requires ATP (1 ATP → 3 Na+ out, 2 K+ in)
 *    - Ion imbalance → depolarized resting potential, inactivation
 *    - Membrane integrity affects capacitance and leak conductance
 *    - Reference: Attwell & Laughlin (2001) "Energy Budget for Signaling"
 *
 * 3. Refractory Period:
 *    - Na+ channel recovery depends on Na+/K+-ATPase activity
 *    - ATP depletion → prolonged refractory (slower recovery)
 *    - Hyperthermia → shortened refractory (faster kinetics)
 *    - Reference: Clay (1998) "Excitability of the squid giant axon"
 *
 * 4. Axonal Transport:
 *    - Kinesin/dynein motors require ATP
 *    - Fast transport: 200-400 mm/day (mitochondria, vesicles)
 *    - Slow transport: 0.2-1 mm/day (cytoskeletal elements)
 *    - ATP depletion → impaired vesicle delivery to terminals
 *    - Reference: Hirokawa et al. (2010) "Kinesin superfamily motor proteins"
 *
 * SUBSTRATE → DENDRITE PATHWAYS:
 * -------------------------------
 * 1. EPSP/IPSP Integration:
 *    - Membrane integrity determines capacitance and time constant
 *    - Degraded membrane → leaky integration, reduced EPSP amplitude
 *    - Cable theory: λ = √(R_m / R_a), τ = R_m * C_m
 *    - Ion balance affects reversal potentials (E_syn)
 *    - Reference: Rall (1967) "Distinguishing theoretical synaptic potentials"
 *
 * 2. Calcium Dynamics:
 *    - Ca2+ ATPase pumps require ATP for calcium extrusion
 *    - Ca2+ homeostasis critical for plasticity (LTP/LTD thresholds)
 *    - ATP depletion → elevated basal [Ca2+], impaired signaling
 *    - Reference: Berridge et al. (2003) "Calcium signalling: dynamics, homeostasis"
 *
 * 3. Dendritic Spike Threshold:
 *    - NMDA receptors: Mg2+ block removal depends on depolarization
 *    - Ion imbalance → altered NMDA spike threshold
 *    - Na+ channel density in dendrites affected by ATP availability
 *    - Hyperthermia → lower spike threshold (Q10 effect)
 *    - Reference: Larkum et al. (1999) "Dendritic mechanisms underlying plateau potentials"
 *
 * 4. Spine Plasticity:
 *    - Actin polymerization requires ATP (spine growth during LTP)
 *    - Protein synthesis for receptor trafficking needs energy
 *    - ATP depletion → suppressed structural plasticity
 *    - Metabolic stress → increased spine elimination
 *    - Reference: Honkura et al. (2008) "The subspine organization of actin"
 *
 * AXON/DENDRITE → SUBSTRATE PATHWAYS:
 * ------------------------------------
 * 1. Energy Consumption:
 *    - Action potential: ~10^8 ATP molecules per spike
 *    - Synaptic transmission: ~1.6 × 10^5 ATP per vesicle
 *    - Dendritic integration: continuous ATP for Na+/K+-ATPase
 *    - High activity → ATP depletion, lactate accumulation
 *    - Reference: Rangaraju et al. (2014) "Activity-driven local ATP synthesis"
 *
 * 2. Ion Gradient Disruption:
 *    - Each spike: 3-4 pmol/cm2 Na+ influx, K+ efflux
 *    - Sustained activity → accumulated ionic imbalance
 *    - Requires ATP-dependent recovery between bursts
 *    - Reference: Alle et al. (2009) "Energy-efficient action potentials"
 *
 * 3. Temperature Increase:
 *    - Metabolic heat generation from ATP hydrolysis
 *    - Active regions can be 0.1-0.3°C warmer
 *    - Affects local channel kinetics and conduction
 *    - Reference: Yablonskiy et al. (2000) "Coupling between changes in temperature"
 *
 * ARCHITECTURE:
 * ```
 * ╔═══════════════════════════════════════════════════════════════════════════╗
 * ║        AXON-DENDRITE SUBSTRATE INTEGRATION BRIDGE                          ║
 * ╠═══════════════════════════════════════════════════════════════════════════╣
 * ║                                                                            ║
 * ║   ┌────────────────────────────────────────────────────────────────────┐  ║
 * ║   │              SUBSTRATE → AXON MODULATION                            │  ║
 * ║   │                                                                     │  ║
 * ║   │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐            │  ║
 * ║   │   │ Temperature  │  │ ATP Level    │  │ Ion Balance  │            │  ║
 * ║   │   │ ──────────── │  │ ──────────── │  │ ──────────── │            │  ║
 * ║   │   │ Q10 effect   │  │ Pump function│  │ Gradient     │            │  ║
 * ║   │   │ ±50% velocity│  │ Spike reliab.│  │ maintenance  │            │  ║
 * ║   │   └──────┬───────┘  └──────┬───────┘  └──────┬───────┘            │  ║
 * ║   │          └──────────────────┴──────────────────┘                   │  ║
 * ║   │                             ↓                                      │  ║
 * ║   │   ┌─────────────────────────────────────────────────────────────┐ │  ║
 * ║   │   │           AXON EFFECTS                                      │ │  ║
 * ║   │   │  • Conduction velocity: velocity_mod ∈ [0.5, 1.5]          │ │  ║
 * ║   │   │  • AP reliability: reliability ∈ [0, 1]                    │ │  ║
 * ║   │   │  • Refractory period: refractory_mod ∈ [0.7, 1.3]          │ │  ║
 * ║   │   │  • Transport efficiency: transport_mod ∈ [0, 1]            │ │  ║
 * ║   │   └─────────────────────────────────────────────────────────────┘ │  ║
 * ║   └────────────────────────────────────────────────────────────────────┘  ║
 * ║                                                                            ║
 * ║   ┌────────────────────────────────────────────────────────────────────┐  ║
 * ║   │              SUBSTRATE → DENDRITE MODULATION                        │  ║
 * ║   │                                                                     │  ║
 * ║   │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐            │  ║
 * ║   │   │ Membrane     │  │ ATP/Ca2+     │  │ Temperature  │            │  ║
 * ║   │   │ Integrity    │  │ Homeostasis  │  │ Effect       │            │  ║
 * ║   │   │ ──────────── │  │ ──────────── │  │ ──────────── │            │  ║
 * ║   │   │ Cable props  │  │ Plasticity   │  │ Spike thresh │            │  ║
 * ║   │   └──────┬───────┘  └──────┬───────┘  └──────┬───────┘            │  ║
 * ║   │          └──────────────────┴──────────────────┘                   │  ║
 * ║   │                             ↓                                      │  ║
 * ║   │   ┌─────────────────────────────────────────────────────────────┐ │  ║
 * ║   │   │           DENDRITE EFFECTS                                  │ │  ║
 * ║   │   │  • Integration efficiency: integration_mod ∈ [0, 1]        │ │  ║
 * ║   │   │  • Spike threshold: threshold_mod ∈ [0.8, 1.2]             │ │  ║
 * ║   │   │  • Plasticity capacity: plasticity_mod ∈ [0, 1]            │ │  ║
 * ║   │   │  • Calcium dynamics: ca_handling_mod ∈ [0, 1]              │ │  ║
 * ║   │   └─────────────────────────────────────────────────────────────┘ │  ║
 * ║   └────────────────────────────────────────────────────────────────────┘  ║
 * ║                                                                            ║
 * ║   ┌────────────────────────────────────────────────────────────────────┐  ║
 * ║   │              AXON/DENDRITE → SUBSTRATE FEEDBACK                     │  ║
 * ║   │                                                                     │  ║
 * ║   │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐            │  ║
 * ║   │   │ Spike Count  │  │ Synaptic     │  │ Dendritic    │            │  ║
 * ║   │   │              │  │ Events       │  │ Activity     │            │  ║
 * ║   │   └──────┬───────┘  └──────┬───────┘  └──────┬───────┘            │  ║
 * ║   │          └──────────────────┴──────────────────┘                   │  ║
 * ║   │                             ↓                                      │  ║
 * ║   │   ┌─────────────────────────────────────────────────────────────┐ │  ║
 * ║   │   │           SUBSTRATE DEPLETION                               │ │  ║
 * ║   │   │  • ATP consumption: activity × cost_per_event               │ │  ║
 * ║   │   │  • Ion accumulation: K+ external, Na+ internal              │ │  ║
 * ║   │   │  • Thermal load: metabolic heat generation                  │ │  ║
 * ║   │   └─────────────────────────────────────────────────────────────┘ │  ║
 * ║   └────────────────────────────────────────────────────────────────────┘  ║
 * ║                                                                            ║
 * ╚═══════════════════════════════════════════════════════════════════════════╝
 * ```
 *
 * @author NIMCP Development Team
 */

#ifndef NIMCP_AXON_DENDRITE_SUBSTRATE_BRIDGE_H
#define NIMCP_AXON_DENDRITE_SUBSTRATE_BRIDGE_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

/* Module integrations */
#include "core/neural_substrate/nimcp_neural_substrate.h"
#include "core/axon/nimcp_axon.h"
#include "core/dendrite/nimcp_dendrite.h"

/* Bio-async integration */
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"

/* Thread utilities */
#include "utils/thread/nimcp_thread.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Constants
 * ============================================================================ */

/* Hodgkin-Huxley Q10 coefficients for temperature effects */
#define SUBSTRATE_Q10_NA_CHANNEL           2.8f     /**< Na+ channel kinetics Q10 */
#define SUBSTRATE_Q10_K_CHANNEL            2.5f     /**< K+ channel kinetics Q10 */
#define SUBSTRATE_Q10_CA_CHANNEL           2.3f     /**< Ca2+ channel kinetics Q10 */
#define SUBSTRATE_Q10_PUMP                 2.1f     /**< Na+/K+-ATPase Q10 */
#define SUBSTRATE_REFERENCE_TEMP           37.0f    /**< Reference temperature (°C) */

/* Axon substrate effects - ATP dependence */
#define SUBSTRATE_ATP_THRESHOLD_SPIKE      0.5f     /**< Min ATP for reliable spikes */
#define SUBSTRATE_ATP_THRESHOLD_TRANSPORT  0.4f     /**< Min ATP for transport */
#define SUBSTRATE_ATP_COST_PER_SPIKE       0.002f   /**< ATP consumed per spike */

/* Axon substrate effects - Ion balance */
#define SUBSTRATE_ION_THRESHOLD_CONDUCT    0.6f     /**< Min ion balance for conduction */
#define SUBSTRATE_ION_ACCUMULATION_RATE    0.001f   /**< Ion imbalance per spike */

/* Axon substrate effects - Membrane integrity */
#define SUBSTRATE_MEMBRANE_THRESHOLD       0.7f     /**< Min membrane for AP propagation */
#define SUBSTRATE_MEMBRANE_LEAK_FACTOR     2.0f     /**< Leak increase factor */

/* Dendrite substrate effects - Integration */
#define SUBSTRATE_DENDRITE_ATP_THRESHOLD   0.4f     /**< Min ATP for plasticity */
#define SUBSTRATE_DENDRITE_CA_THRESHOLD    0.5f     /**< Min for Ca2+ handling */
#define SUBSTRATE_DENDRITE_MEMBRANE_MIN    0.6f     /**< Min membrane for integration */

/* Dendrite substrate effects - Plasticity */
#define SUBSTRATE_PLASTICITY_ATP_COST      0.005f   /**< ATP cost for LTP event */
#define SUBSTRATE_SPINE_GROWTH_ATP_COST    0.003f   /**< ATP cost for spine growth */

/* Temperature effect ranges (Celsius) */
#define SUBSTRATE_TEMP_NORMAL_LOW          36.0f    /**< Lower bound normal */
#define SUBSTRATE_TEMP_NORMAL_HIGH         38.0f    /**< Upper bound normal */
#define SUBSTRATE_TEMP_HYPOTHERMIA         32.0f    /**< Hypothermia threshold */
#define SUBSTRATE_TEMP_HYPERTHERMIA        40.0f    /**< Hyperthermia threshold */

/* Velocity modulation bounds */
#define SUBSTRATE_VELOCITY_MIN             0.5f     /**< Min velocity multiplier */
#define SUBSTRATE_VELOCITY_MAX             1.5f     /**< Max velocity multiplier */

/* Bio-async module IDs */
#define BIO_MODULE_AXON_SUBSTRATE          0x0E00   /**< Bio-async axon-substrate */
#define BIO_MODULE_DENDRITE_SUBSTRATE      0x0E01   /**< Bio-async dendrite-substrate */

/* ============================================================================
 * Structures
 * ============================================================================ */

/**
 * @brief Substrate effects on axon conduction and transmission
 *
 * WHAT: Quantified substrate modulation of axonal properties
 * WHY:  Substrate conditions directly affect action potential propagation
 * HOW:  Compute modulation factors from substrate state
 */
typedef struct {
    /* Conduction velocity modulation */
    float temperature_q10_factor;     /**< Q10 temperature effect [0.5-1.5] */
    float atp_velocity_factor;        /**< ATP effect on velocity [0-1] */
    float myelin_efficiency;          /**< Myelination ATP efficiency [0-1] */
    float overall_velocity_mod;       /**< Combined velocity multiplier */

    /* Action potential reliability */
    float ion_gradient_strength;     /**< Na+/K+ gradient quality [0-1] */
    float ap_amplitude_mod;           /**< AP amplitude modulation [0-1] */
    float spike_reliability;          /**< Propagation reliability [0-1] */

    /* Refractory period modulation */
    float pump_activity;              /**< Na+/K+-ATPase activity [0-1] */
    float refractory_period_mod;      /**< Refractory multiplier [0.7-1.3] */

    /* Axonal transport */
    float transport_efficiency;       /**< Vesicle transport [0-1] */
    float kinesin_activity;           /**< Motor protein activity [0-1] */

    /* Membrane effects */
    float membrane_capacitance_mod;   /**< Capacitance change [0.8-1.2] */
    float membrane_leak_mod;          /**< Leak conductance [1.0-3.0] */

    /* Overall capacity */
    float overall_capacity;           /**< Combined axon capacity [0-1] */
} axon_substrate_effects_t;

/**
 * @brief Substrate effects on dendritic integration and plasticity
 *
 * WHAT: Quantified substrate modulation of dendritic properties
 * WHY:  Substrate conditions affect integration, spikes, and plasticity
 * HOW:  Cable theory and biophysical constraints
 */
typedef struct {
    /* Integration efficiency */
    float membrane_time_constant_mod; /**< τ_m modulation [0.5-1.5] */
    float space_constant_mod;         /**< λ modulation [0.5-1.5] */
    float integration_efficiency;     /**< Overall integration [0-1] */
    float attenuation_mod;            /**< Voltage attenuation [1.0-2.0] */

    /* Dendritic spike threshold */
    float nmda_mg_block_mod;          /**< Mg2+ block sensitivity [0.8-1.2] */
    float spike_threshold_mod;        /**< Threshold voltage shift [0.8-1.2] */
    float na_channel_availability;    /**< Dendritic Na+ channels [0-1] */

    /* Calcium dynamics */
    float ca_pump_efficiency;         /**< Ca2+ extrusion [0-1] */
    float ca_buffer_capacity;         /**< Buffering capacity [0-1] */
    float ca_handling_mod;            /**< Overall Ca2+ handling [0-1] */

    /* Plasticity capacity */
    float ltp_capacity;               /**< LTP induction ability [0-1] */
    float ltd_capacity;               /**< LTD induction ability [0-1] */
    float spine_growth_capacity;      /**< Structural plasticity [0-1] */
    float plasticity_mod;             /**< Overall plasticity [0-1] */

    /* Overall capacity */
    float overall_capacity;           /**< Combined dendrite capacity [0-1] */
} dendrite_substrate_effects_t;

/**
 * @brief Bridge statistics
 */
typedef struct {
    uint64_t total_updates;
    uint64_t axon_spikes_processed;
    uint64_t dendrite_events_processed;
    uint32_t atp_depletion_events;
    uint32_t ion_imbalance_events;
    uint32_t conduction_failures;
    uint32_t plasticity_suppressions;
    float peak_velocity_mod;
    float min_velocity_mod;
    float avg_integration_efficiency;
} axon_dendrite_substrate_stats_t;

/**
 * @brief Bridge configuration
 */
typedef struct {
    /* Feature enables */
    bool enable_axon_modulation;
    bool enable_dendrite_modulation;
    bool enable_bidirectional_feedback;
    bool enable_temperature_effects;
    bool enable_atp_dynamics;
    bool enable_ion_dynamics;
    bool enable_bio_async;

    /* Sensitivity multipliers */
    float temperature_sensitivity;
    float atp_sensitivity;
    float ion_sensitivity;
    float membrane_sensitivity;

    /* Thresholds */
    float min_atp_for_spikes;
    float min_ion_balance_for_conduct;
    float min_membrane_for_integration;
} axon_dendrite_substrate_config_t;

/**
 * @brief Complete axon-dendrite substrate bridge state
 */
typedef struct {
    /* System handles */
    neural_substrate_t* substrate;
    axon_t* axon;                     /**< Optional: single axon integration */
    dendrite_t* dendrite;             /**< Optional: single dendrite integration */

    /* Current effects */
    axon_substrate_effects_t axon_effects;
    dendrite_substrate_effects_t dendrite_effects;

    /* Configuration */
    axon_dendrite_substrate_config_t config;

    /* Activity tracking (for feedback) */
    uint64_t recent_axon_spikes;
    uint64_t recent_dendrite_events;
    float accumulated_atp_debt;
    float accumulated_ion_imbalance;

    /* Bio-async integration */
    bio_module_context_t bio_ctx;
    bool bio_async_enabled;

    /* Statistics */
    axon_dendrite_substrate_stats_t stats;

    /* Thread safety */
    nimcp_mutex_t* mutex;
} axon_dendrite_substrate_bridge_t;

/* ============================================================================
 * Lifecycle API
 * ============================================================================ */

/**
 * @brief Get default configuration
 *
 * WHAT: Provide sensible default configuration
 * WHY:  Easy initialization with biological defaults
 * HOW:  Set parameters based on physiological values
 *
 * @param config Output configuration
 * @return 0 on success, -1 on error
 */
int axon_dendrite_substrate_default_config(axon_dendrite_substrate_config_t* config);

/**
 * @brief Create axon-dendrite substrate bridge
 *
 * WHAT: Initialize bridge between substrate and axon/dendrite modules
 * WHY:  Enable substrate modulation of neural signal processing
 * HOW:  Allocate structure, connect modules, initialize state
 *
 * @param config Configuration (NULL for defaults)
 * @param substrate Neural substrate
 * @param axon Axon (NULL if dendrite-only)
 * @param dendrite Dendrite (NULL if axon-only)
 * @return New bridge or NULL on failure
 */
axon_dendrite_substrate_bridge_t* axon_dendrite_substrate_bridge_create(
    const axon_dendrite_substrate_config_t* config,
    neural_substrate_t* substrate,
    axon_t* axon,
    dendrite_t* dendrite
);

/**
 * @brief Destroy axon-dendrite substrate bridge
 *
 * WHAT: Clean up bridge resources
 * WHY:  Proper resource deallocation
 * HOW:  Free structure, release mutex
 *
 * @param bridge Bridge to destroy (NULL safe)
 */
void axon_dendrite_substrate_bridge_destroy(axon_dendrite_substrate_bridge_t* bridge);

/* ============================================================================
 * Bio-async Integration API
 * ============================================================================ */

/**
 * @brief Connect to bio-async router
 *
 * @param bridge Axon-dendrite substrate bridge
 * @return 0 on success, -1 on error
 */
int axon_dendrite_substrate_connect_bio_async(axon_dendrite_substrate_bridge_t* bridge);

/**
 * @brief Disconnect from bio-async router
 *
 * @param bridge Axon-dendrite substrate bridge
 * @return 0 on success, -1 on error
 */
int axon_dendrite_substrate_disconnect_bio_async(axon_dendrite_substrate_bridge_t* bridge);

/**
 * @brief Check if bio-async is connected
 *
 * @param bridge Axon-dendrite substrate bridge
 * @return true if connected
 */
bool axon_dendrite_substrate_is_bio_async_connected(
    const axon_dendrite_substrate_bridge_t* bridge
);

/* ============================================================================
 * Substrate → Axon API
 * ============================================================================ */

/**
 * @brief Update axon effects from substrate state
 *
 * WHAT: Compute substrate modulation of axon properties
 * WHY:  Substrate conditions directly affect conduction
 * HOW:  Apply Q10, ATP, ion, and membrane effects
 *
 * BIOLOGICAL:
 * - Q10 coefficient modulates channel kinetics
 * - ATP depletion impairs Na+/K+-ATPase → gradient loss
 * - Ion imbalance → altered resting potential
 * - Membrane damage → increased leak, reduced capacitance
 *
 * @param bridge Axon-dendrite substrate bridge
 * @return 0 on success, -1 on error
 */
int axon_dendrite_substrate_update_axon_effects(axon_dendrite_substrate_bridge_t* bridge);

/**
 * @brief Get conduction velocity modulation factor
 *
 * WHAT: Retrieve current velocity multiplier for axon
 * WHY:  Axon module needs this for spike propagation delay
 * HOW:  Return pre-computed velocity_mod from effects
 *
 * @param bridge Axon-dendrite substrate bridge
 * @return Velocity multiplier [0.5-1.5]
 */
float axon_dendrite_substrate_get_conduction_mod(
    const axon_dendrite_substrate_bridge_t* bridge
);

/**
 * @brief Get spike reliability factor
 *
 * WHAT: Probability that spike successfully propagates
 * WHY:  Low ATP/ion balance → conduction failures
 * HOW:  Combine ion gradient and membrane integrity
 *
 * @param bridge Axon-dendrite substrate bridge
 * @return Reliability [0-1], 0 = always fails, 1 = always succeeds
 */
float axon_dendrite_substrate_get_spike_reliability(
    const axon_dendrite_substrate_bridge_t* bridge
);

/**
 * @brief Get refractory period modulation
 *
 * WHAT: Multiplier for axon refractory period
 * WHY:  Pump activity affects Na+ channel recovery
 * HOW:  ATP → pump activity → faster/slower recovery
 *
 * @param bridge Axon-dendrite substrate bridge
 * @return Refractory multiplier [0.7-1.3]
 */
float axon_dendrite_substrate_get_refractory_mod(
    const axon_dendrite_substrate_bridge_t* bridge
);

/* ============================================================================
 * Substrate → Dendrite API
 * ============================================================================ */

/**
 * @brief Update dendrite effects from substrate state
 *
 * WHAT: Compute substrate modulation of dendrite properties
 * WHY:  Substrate affects integration, spikes, plasticity
 * HOW:  Apply cable theory, calcium, and plasticity constraints
 *
 * BIOLOGICAL:
 * - Membrane integrity → R_m, C_m → λ, τ_m
 * - ATP → Ca2+ pumps → plasticity capacity
 * - Temperature → channel kinetics → spike threshold
 * - Ion balance → reversal potentials → integration
 *
 * @param bridge Axon-dendrite substrate bridge
 * @return 0 on success, -1 on error
 */
int axon_dendrite_substrate_update_dendrite_effects(
    axon_dendrite_substrate_bridge_t* bridge
);

/**
 * @brief Get dendritic integration efficiency
 *
 * WHAT: How effectively dendrite integrates synaptic inputs
 * WHY:  Membrane damage → leaky integration
 * HOW:  Combine time constant and space constant mods
 *
 * @param bridge Axon-dendrite substrate bridge
 * @return Integration efficiency [0-1]
 */
float axon_dendrite_substrate_get_integration_mod(
    const axon_dendrite_substrate_bridge_t* bridge
);

/**
 * @brief Get dendritic spike threshold modulation
 *
 * WHAT: Voltage shift for NMDA spike initiation
 * WHY:  Ion balance and temperature affect threshold
 * HOW:  Combine NMDA Mg2+ block and Na+ availability
 *
 * @param bridge Axon-dendrite substrate bridge
 * @return Threshold multiplier [0.8-1.2]
 */
float axon_dendrite_substrate_get_spike_threshold_mod(
    const axon_dendrite_substrate_bridge_t* bridge
);

/**
 * @brief Get plasticity capacity
 *
 * WHAT: Ability to undergo LTP/LTD
 * WHY:  ATP required for spine growth, receptor trafficking
 * HOW:  ATP level → plasticity_mod
 *
 * @param bridge Axon-dendrite substrate bridge
 * @return Plasticity capacity [0-1]
 */
float axon_dendrite_substrate_get_plasticity_mod(
    const axon_dendrite_substrate_bridge_t* bridge
);

/**
 * @brief Get calcium handling capacity
 *
 * WHAT: Efficiency of Ca2+ extrusion and buffering
 * WHY:  Ca2+ dynamics critical for plasticity
 * HOW:  ATP → pump efficiency, buffering capacity
 *
 * @param bridge Axon-dendrite substrate bridge
 * @return Ca2+ handling [0-1]
 */
float axon_dendrite_substrate_get_ca_handling_mod(
    const axon_dendrite_substrate_bridge_t* bridge
);

/* ============================================================================
 * Axon/Dendrite → Substrate Feedback API
 * ============================================================================ */

/**
 * @brief Record axon spike (depletes ATP, disrupts ions)
 *
 * WHAT: Update substrate after action potential
 * WHY:  Spikes consume energy and disrupt gradients
 * HOW:  Decrement ATP, accumulate ion imbalance
 *
 * BIOLOGICAL:
 * - 10^8 ATP molecules per spike
 * - 3-4 pmol/cm2 Na+ influx per spike
 * - Requires Na+/K+-ATPase recovery
 *
 * @param bridge Axon-dendrite substrate bridge
 * @param spike_count Number of spikes fired
 * @return 0 on success, -1 on error
 */
int axon_dendrite_substrate_record_axon_spikes(
    axon_dendrite_substrate_bridge_t* bridge,
    uint32_t spike_count
);

/**
 * @brief Record dendritic synaptic events
 *
 * WHAT: Update substrate after dendritic activity
 * WHY:  Synaptic integration consumes ATP
 * HOW:  Decrement ATP based on event count
 *
 * @param bridge Axon-dendrite substrate bridge
 * @param event_count Number of synaptic events
 * @return 0 on success, -1 on error
 */
int axon_dendrite_substrate_record_dendrite_events(
    axon_dendrite_substrate_bridge_t* bridge,
    uint32_t event_count
);

/**
 * @brief Record plasticity event (high ATP cost)
 *
 * WHAT: Update substrate after LTP/LTD or spine growth
 * WHY:  Plasticity is energetically expensive
 * HOW:  Deplete ATP proportional to plasticity magnitude
 *
 * @param bridge Axon-dendrite substrate bridge
 * @param magnitude Plasticity magnitude [0-1]
 * @return 0 on success, -1 on error
 */
int axon_dendrite_substrate_record_plasticity(
    axon_dendrite_substrate_bridge_t* bridge,
    float magnitude
);

/* ============================================================================
 * Bidirectional Update API
 * ============================================================================ */

/**
 * @brief Update bridge (both directions)
 *
 * WHAT: Process all substrate-axon-dendrite interactions
 * WHY:  Advance coupled state machine
 * HOW:  Compute substrate effects, apply feedback
 *
 * @param bridge Axon-dendrite substrate bridge
 * @param delta_ms Time since last update
 * @return 0 on success, -1 on error
 */
int axon_dendrite_substrate_bridge_update(
    axon_dendrite_substrate_bridge_t* bridge,
    uint64_t delta_ms
);

/* ============================================================================
 * Query API
 * ============================================================================ */

/**
 * @brief Get current axon effects
 *
 * @param bridge Axon-dendrite substrate bridge
 * @param effects Output effects structure
 * @return 0 on success, -1 on error
 */
int axon_dendrite_substrate_get_axon_effects(
    const axon_dendrite_substrate_bridge_t* bridge,
    axon_substrate_effects_t* effects
);

/**
 * @brief Get current dendrite effects
 *
 * @param bridge Axon-dendrite substrate bridge
 * @param effects Output effects structure
 * @return 0 on success, -1 on error
 */
int axon_dendrite_substrate_get_dendrite_effects(
    const axon_dendrite_substrate_bridge_t* bridge,
    dendrite_substrate_effects_t* effects
);

/**
 * @brief Check if axon is substrate-limited
 *
 * WHAT: Determine if substrate constraints impair axon
 * WHY:  Detect when substrate stress affects signaling
 * HOW:  Check ATP, ion, membrane thresholds
 *
 * @param bridge Axon-dendrite substrate bridge
 * @return true if substrate-limited
 */
bool axon_dendrite_substrate_is_axon_limited(
    const axon_dendrite_substrate_bridge_t* bridge
);

/**
 * @brief Check if dendrite is substrate-limited
 *
 * @param bridge Axon-dendrite substrate bridge
 * @return true if substrate-limited
 */
bool axon_dendrite_substrate_is_dendrite_limited(
    const axon_dendrite_substrate_bridge_t* bridge
);

/**
 * @brief Get bridge statistics
 *
 * @param bridge Axon-dendrite substrate bridge
 * @param stats Output statistics
 * @return 0 on success, -1 on error
 */
int axon_dendrite_substrate_get_stats(
    const axon_dendrite_substrate_bridge_t* bridge,
    axon_dendrite_substrate_stats_t* stats
);

#ifdef __cplusplus
}
#endif

#endif /* NIMCP_AXON_DENDRITE_SUBSTRATE_BRIDGE_H */
