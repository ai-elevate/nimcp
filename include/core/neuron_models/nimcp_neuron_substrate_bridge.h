/**
 * @file nimcp_neuron_substrate_bridge.h
 * @brief Neuron Model-Substrate Integration Bridge
 * @version 1.0.0
 * @date 2025-12-12
 *
 * WHAT: Bidirectional integration between neuron models and neural substrate
 * WHY:  Neuron firing dynamics depend on metabolic/physical substrate conditions
 *       (ATP availability, temperature, oxygen). Neuron activity consumes substrate resources.
 * HOW:  Substrate modulates firing rate and excitability; neurons consume ATP per spike.
 *
 * BIOLOGICAL BASIS:
 * ==================================================================================
 *
 * SUBSTRATE → NEURON PATHWAYS:
 * ----------------------------
 * 1. ATP-Dependent Excitability:
 *    - Na+/K+-ATPase maintains resting potential (-70mV)
 *    - ATP depletion → depolarization → altered firing threshold
 *    - Severe ATP depletion → complete loss of excitability
 *    - Reference: Erecinska & Silver (1989) "ATP and brain function"
 *
 * 2. Temperature Effects (Q10 Rule):
 *    - Q10 ≈ 2-3 for neural processes (rate doubles per 10°C)
 *    - Hyperthermia (>40°C) → hyperexcitability → seizures
 *    - Hypothermia (<32°C) → slowed dynamics → reduced firing
 *    - Reference: Hodgkin & Huxley (1952) temperature coefficients
 *
 * 3. Oxygen-Dependent Function:
 *    - Neurons require continuous O2 for ATP synthesis
 *    - Hypoxia → rapid ATP depletion → spreading depression
 *    - O2 < 50% → reduced synaptic transmission
 *    - Reference: Ames (2000) "CNS energy metabolism"
 *
 * 4. Ion Balance Effects:
 *    - Na+/K+ gradient determines resting potential
 *    - Ca2+ overload → excitotoxicity
 *    - Ion imbalance → altered firing patterns
 *    - Reference: Stys (1998) "Ionic mechanisms of neuronal injury"
 *
 * NEURON → SUBSTRATE PATHWAYS:
 * ----------------------------
 * 1. Energy Cost of Spiking:
 *    - Action potential: ~10^8 ATP molecules per spike
 *    - Na+ influx restored by Na+/K+-ATPase
 *    - High-frequency firing → rapid ATP depletion
 *    - Reference: Attwell & Laughlin (2001) "Energy budget for signaling"
 *
 * 2. Metabolic Demand Scaling:
 *    - Firing rate correlates with metabolic rate
 *    - 100 Hz firing → 10x baseline ATP consumption
 *    - Sustained activity → local hypoxia
 *    - Reference: Alle et al. (2009) "Energy-efficient action potentials"
 *
 * 3. Heat Generation:
 *    - Ion flux generates heat
 *    - High activity regions show temperature increase
 *    - Reference: Kiyatkin (2010) "Brain hyperthermia"
 *
 * ARCHITECTURE:
 * ```
 * ╔═══════════════════════════════════════════════════════════════════════════╗
 * ║              NEURON-SUBSTRATE INTEGRATION BRIDGE                           ║
 * ╠═══════════════════════════════════════════════════════════════════════════╣
 * ║                                                                            ║
 * ║   ┌────────────────────────────────────────────────────────────────────┐  ║
 * ║   │               SUBSTRATE → NEURON PATHWAYS                           │  ║
 * ║   │                                                                     │  ║
 * ║   │   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐          │  ║
 * ║   │   │ ATP Level    │   │ Temperature  │   │ O2 Saturation│          │  ║
 * ║   │   │ ──────────── │   │ ──────────── │   │ ──────────── │          │  ║
 * ║   │   │ Excitability │   │ Q10 Scaling  │   │ Transmission │          │  ║
 * ║   │   │ Modulation   │   │ (firing rate)│   │ Efficiency   │          │  ║
 * ║   │   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘          │  ║
 * ║   │          └──────────────────┴──────────────────┘                   │  ║
 * ║   │                             ↓                                      │  ║
 * ║   │   ┌─────────────────────────────────────────────────────────────┐ │  ║
 * ║   │   │              NEURON MODULATION                              │ │  ║
 * ║   │   │  • firing_rate_mod: Q10 temperature scaling                │ │  ║
 * ║   │   │  • excitability_mod: ATP-dependent threshold shift         │ │  ║
 * ║   │   │  • input_scaling: Overall substrate capacity               │ │  ║
 * ║   │   └─────────────────────────────────────────────────────────────┘ │  ║
 * ║   └────────────────────────────────────────────────────────────────────┘  ║
 * ║                                                                            ║
 * ║   ┌────────────────────────────────────────────────────────────────────┐  ║
 * ║   │               NEURON → SUBSTRATE PATHWAYS                           │  ║
 * ║   │                                                                     │  ║
 * ║   │   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐          │  ║
 * ║   │   │ Spike Event  │   │ Firing Rate  │   │ Activity     │          │  ║
 * ║   │   │ ──────────── │   │ ──────────── │   │ ──────────── │          │  ║
 * ║   │   │ ATP Cost     │   │ Metabolic    │   │ Heat         │          │  ║
 * ║   │   │ (~10^8 ATP)  │   │ Demand       │   │ Generation   │          │  ║
 * ║   │   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘          │  ║
 * ║   │          └──────────────────┴──────────────────┘                   │  ║
 * ║   │                             ↓                                      │  ║
 * ║   │   ┌─────────────────────────────────────────────────────────────┐ │  ║
 * ║   │   │             SUBSTRATE CONSUMPTION                           │ │  ║
 * ║   │   │  • ATP depletion per spike (configurable cost)              │ │  ║
 * ║   │   │  • Metabolic rate tracking                                  │ │  ║
 * ║   │   │  • Temperature increase from activity                       │ │  ║
 * ║   │   └─────────────────────────────────────────────────────────────┘ │  ║
 * ║   └────────────────────────────────────────────────────────────────────┘  ║
 * ║                                                                            ║
 * ╚═══════════════════════════════════════════════════════════════════════════╝
 * ```
 *
 * DESIGN PATTERNS:
 * - Observer: Neurons notify bridge of spike events for ATP consumption
 * - Strategy: Modulation modes (additive, multiplicative, mixed)
 *
 * @author NIMCP Development Team
 */

#ifndef NIMCP_NEURON_SUBSTRATE_BRIDGE_H
#define NIMCP_NEURON_SUBSTRATE_BRIDGE_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

/* Module integrations */
#include "core/neuron_models/nimcp_neuron_model.h"
#include "core/neural_substrate/nimcp_neural_substrate.h"

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

/* ATP cost per spike (normalized units, configurable) */
#define NEURON_SPIKE_ATP_COST_DEFAULT       0.0008f   /**< Default ATP cost per spike */
#define NEURON_SPIKE_ATP_COST_FAST          0.0005f   /**< Fast-spiking interneurons */
#define NEURON_SPIKE_ATP_COST_PYRAMIDAL     0.001f    /**< Pyramidal neurons */

/* Q10 temperature coefficients for neuron dynamics */
#define NEURON_Q10_FIRING_RATE              2.5f      /**< Q10 for firing rate */
#define NEURON_Q10_MEMBRANE_TIME_CONSTANT   2.2f      /**< Q10 for membrane tau */
#define NEURON_Q10_THRESHOLD                1.5f      /**< Q10 for threshold dynamics */

/* Temperature reference (normal brain temperature) */
#define NEURON_REFERENCE_TEMPERATURE        37.0f     /**< Normal brain temp (°C) */

/* ATP-dependent modulation parameters */
#define ATP_EXCITABILITY_THRESHOLD          0.5f      /**< Below this, excitability drops */
#define ATP_CRITICAL_THRESHOLD              0.2f      /**< Below this, neuron nearly silent */

/* Oxygen-dependent modulation */
#define O2_TRANSMISSION_THRESHOLD           0.5f      /**< Below this, transmission impaired */
#define O2_CRITICAL_THRESHOLD               0.3f      /**< Below this, severe impairment */

/* Bio-async module ID - defined in nimcp_bio_messages.h as BIO_MODULE_NEURON_SUBSTRATE */

/* ============================================================================
 * Enumerations
 * ============================================================================ */

/**
 * @brief Modulation application mode
 */
typedef enum {
    MODULATION_MODE_ADDITIVE = 0,     /**< Add modulation to base value */
    MODULATION_MODE_MULTIPLICATIVE,   /**< Multiply base value by modulation */
    MODULATION_MODE_MIXED             /**< Combination of additive and multiplicative */
} neuron_modulation_mode_t;

/* ============================================================================
 * Structures
 * ============================================================================ */

/**
 * @brief Substrate effects on neuron dynamics
 */
typedef struct {
    /* Temperature effects (Q10 scaling) */
    float q10_firing_rate_mod;        /**< Q10 scaling for firing rate [0.5-2.0] */
    float q10_tau_mod;                /**< Q10 scaling for membrane tau [0.5-2.0] */
    float q10_threshold_mod;          /**< Q10 scaling for threshold [0.8-1.2] */

    /* ATP-dependent effects */
    float atp_excitability_mod;       /**< ATP modulation of excitability [0-1.2] */
    float atp_threshold_shift;        /**< Threshold shift from ATP (mV) */

    /* Oxygen-dependent effects */
    float o2_transmission_mod;        /**< O2 modulation of transmission [0-1] */
    float o2_recovery_mod;            /**< O2 modulation of recovery [0-1] */

    /* Ion balance effects */
    float ion_resting_potential_shift; /**< Shift in resting potential (mV) */

    /* Composite modulation factors */
    float firing_rate_mod;            /**< Overall firing rate multiplier [0-2] */
    float excitability_mod;           /**< Overall excitability [0-1.5] */
    float input_scaling;              /**< Input current scaling [0-1] */
} neuron_substrate_effects_t;

/**
 * @brief Energy consumption tracking
 */
typedef struct {
    uint64_t total_spikes;            /**< Total spikes processed */
    float total_atp_consumed;         /**< Total ATP consumed */
    float current_metabolic_rate;     /**< Current metabolic rate */
    float peak_metabolic_rate;        /**< Peak metabolic rate observed */
    float avg_firing_rate;            /**< Average firing rate (Hz) */
    float atp_per_spike;              /**< ATP cost per spike (current) */
} neuron_energy_tracking_t;

/**
 * @brief Bridge statistics
 */
typedef struct {
    uint64_t total_updates;
    uint64_t spikes_consumed;
    uint64_t modulation_applications;
    float total_atp_depleted;
    float max_firing_rate_mod;
    float min_excitability_mod;
    uint32_t substrate_critical_events;
} neuron_substrate_stats_t;

/**
 * @brief Bridge configuration
 */
typedef struct {
    /* Feature enables */
    bool enable_temperature_effects;
    bool enable_atp_modulation;
    bool enable_o2_modulation;
    bool enable_ion_effects;
    bool enable_spike_consumption;
    bool enable_bio_async;

    /* Energy parameters */
    float atp_cost_per_spike;         /**< ATP consumed per spike */
    float baseline_metabolic_cost;    /**< Baseline ATP drain */

    /* Modulation parameters */
    neuron_modulation_mode_t modulation_mode;
    float temperature_sensitivity;    /**< Q10 sensitivity multiplier */
    float atp_sensitivity;            /**< ATP effect sensitivity */
    float o2_sensitivity;             /**< O2 effect sensitivity */

    /* Safety limits */
    float max_firing_rate_mod;        /**< Maximum firing rate multiplier */
    float min_excitability;           /**< Minimum excitability allowed */
} neuron_substrate_config_t;

/**
 * @brief Complete neuron-substrate bridge state
 */
typedef struct {
    /* System handles */
    neuron_model_state_t neuron_model;
    neural_substrate_t* substrate;

    /* Current effects */
    neuron_substrate_effects_t substrate_effects;
    neuron_energy_tracking_t energy_tracking;

    /* Configuration */
    neuron_substrate_config_t config;

    /* Bio-async integration */
    bio_module_context_t bio_ctx;
    bool bio_async_enabled;

    /* Statistics */
    neuron_substrate_stats_t stats;

    /* Thread safety */
    nimcp_mutex_t* mutex;
} neuron_substrate_bridge_t;

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
 * @return 0 on success, NIMCP_ERROR_NULL_POINTER on error
 */
int neuron_substrate_default_config(neuron_substrate_config_t* config);

/**
 * @brief Create neuron-substrate bridge
 *
 * WHAT: Initialize neuron-substrate integration bridge
 * WHY:  Connect neuron dynamics to metabolic/physical substrate
 * HOW:  Allocate structure, link systems, initialize state
 *
 * @param config Configuration (NULL for defaults)
 * @param neuron_model Neuron model state
 * @param substrate Neural substrate
 * @return New bridge or NULL on failure
 */
neuron_substrate_bridge_t* neuron_substrate_bridge_create(
    const neuron_substrate_config_t* config,
    neuron_model_state_t neuron_model,
    neural_substrate_t* substrate
);

/**
 * @brief Destroy neuron-substrate bridge
 *
 * WHAT: Clean up bridge resources
 * WHY:  Proper resource deallocation
 * HOW:  Disconnect bio-async, destroy mutex, free structure
 *
 * @param bridge Bridge to destroy (NULL safe)
 */
void neuron_substrate_bridge_destroy(neuron_substrate_bridge_t* bridge);

/* ============================================================================
 * Bio-async Integration API
 * ============================================================================ */

/**
 * @brief Connect to bio-async router
 *
 * WHAT: Register bridge with bio-async messaging system
 * WHY:  Enable inter-module communication via bio-async
 * HOW:  Register module, get context, enable flag
 *
 * @param bridge Neuron substrate bridge
 * @return 0 on success, NIMCP_ERROR_NULL_POINTER on error
 */
int neuron_substrate_connect_bio_async(neuron_substrate_bridge_t* bridge);

/**
 * @brief Disconnect from bio-async router
 *
 * WHAT: Unregister bridge from bio-async messaging
 * WHY:  Clean shutdown of bio-async integration
 * HOW:  Unregister module, clear context, disable flag
 *
 * @param bridge Neuron substrate bridge
 * @return 0 on success, NIMCP_ERROR_NULL_POINTER on error
 */
int neuron_substrate_disconnect_bio_async(neuron_substrate_bridge_t* bridge);

/**
 * @brief Check if bio-async is connected
 *
 * @param bridge Neuron substrate bridge
 * @return true if connected
 */
bool neuron_substrate_is_bio_async_connected(const neuron_substrate_bridge_t* bridge);

/* ============================================================================
 * Substrate → Neuron API
 * ============================================================================ */

/**
 * @brief Update substrate effects on neuron
 *
 * WHAT: Compute substrate modulation factors for neuron dynamics
 * WHY:  Substrate state (ATP, O2, temperature) affects neural computation
 * HOW:  Query substrate, compute Q10 scaling, ATP/O2 effects
 *
 * Biological Basis:
 * - Temperature: Q10 rule (rate doubles per 10°C)
 * - ATP: Maintains Na+/K+ gradients, powers pumps
 * - O2: Required for ATP synthesis via oxidative phosphorylation
 *
 * @param bridge Neuron substrate bridge
 * @return 0 on success, NIMCP_ERROR_NULL_POINTER/INVALID_STATE on error
 */
int neuron_substrate_update_effects(neuron_substrate_bridge_t* bridge);

/**
 * @brief Apply substrate modulation to neuron model
 *
 * WHAT: Modify neuron model parameters based on substrate state
 * WHY:  Substrate conditions directly affect neuron dynamics
 * HOW:  Scale firing rate, shift threshold, modulate input
 *
 * Modulation applied to:
 * - Firing rate (Q10 temperature scaling)
 * - Excitability (ATP-dependent threshold shift)
 * - Input current (overall substrate capacity)
 *
 * @param bridge Neuron substrate bridge
 * @return 0 on success, NIMCP_ERROR_NULL_POINTER/INVALID_STATE on error
 */
int neuron_substrate_apply_modulation(neuron_substrate_bridge_t* bridge);

/**
 * @brief Get modulated input current
 *
 * WHAT: Scale input current by substrate capacity
 * WHY:  Substrate depletion reduces effective synaptic input
 * HOW:  Multiply input by composite modulation factor
 *
 * @param bridge Neuron substrate bridge
 * @param base_input Base input current
 * @return Modulated input current
 */
float neuron_substrate_get_modulated_input(
    const neuron_substrate_bridge_t* bridge,
    float base_input
);

/**
 * @brief Get modulated firing rate
 *
 * WHAT: Scale firing rate by substrate effects
 * WHY:  Temperature and ATP affect spike generation rate
 * HOW:  Apply Q10 scaling and ATP modulation
 *
 * @param bridge Neuron substrate bridge
 * @param base_rate Base firing rate (Hz)
 * @return Modulated firing rate (Hz)
 */
float neuron_substrate_get_modulated_firing_rate(
    const neuron_substrate_bridge_t* bridge,
    float base_rate
);

/* ============================================================================
 * Neuron → Substrate API
 * ============================================================================ */

/**
 * @brief Consume substrate resources for spike
 *
 * WHAT: Deplete ATP and update metabolic state for spike event
 * WHY:  Action potentials cost ~10^8 ATP molecules per spike
 * HOW:  Subtract ATP cost from substrate, update tracking
 *
 * Biological Basis:
 * - Na+ influx during spike must be pumped out by Na+/K+-ATPase
 * - Each pump cycle consumes 1 ATP to move 3 Na+ out, 2 K+ in
 * - Total ATP cost: ~10^8 molecules per spike
 *
 * @param bridge Neuron substrate bridge
 * @return 0 on success, NIMCP_ERROR_NULL_POINTER on error
 */
int neuron_substrate_consume_spike(neuron_substrate_bridge_t* bridge);

/**
 * @brief Update metabolic rate tracking
 *
 * WHAT: Compute current metabolic rate from firing activity
 * WHY:  Track energy consumption for monitoring
 * HOW:  Calculate ATP consumption per time based on firing rate
 *
 * @param bridge Neuron substrate bridge
 * @param delta_ms Time since last update (milliseconds)
 * @return 0 on success, NIMCP_ERROR_NULL_POINTER on error
 */
int neuron_substrate_update_metabolic_rate(
    neuron_substrate_bridge_t* bridge,
    uint64_t delta_ms
);

/* ============================================================================
 * Bidirectional Update API
 * ============================================================================ */

/**
 * @brief Update neuron-substrate bridge (both directions)
 *
 * WHAT: Process all neuron-substrate interactions
 * WHY:  Advance coupled state machine
 * HOW:  Update substrate effects, apply modulation, track metabolism
 *
 * Call this each simulation step to:
 * 1. Compute substrate effects on neuron (temperature, ATP, O2)
 * 2. Apply modulation to neuron model
 * 3. Update metabolic tracking
 *
 * @param bridge Neuron substrate bridge
 * @param delta_ms Time since last update (milliseconds)
 * @return 0 on success, NIMCP_ERROR_NULL_POINTER on error
 */
int neuron_substrate_bridge_update(
    neuron_substrate_bridge_t* bridge,
    uint64_t delta_ms
);

/* ============================================================================
 * Query API
 * ============================================================================ */

/**
 * @brief Get current substrate effects on neuron
 *
 * @param bridge Neuron substrate bridge
 * @param effects Output effects structure
 * @return 0 on success, NIMCP_ERROR_NULL_POINTER on error
 */
int neuron_substrate_get_effects(
    const neuron_substrate_bridge_t* bridge,
    neuron_substrate_effects_t* effects
);

/**
 * @brief Get energy tracking data
 *
 * @param bridge Neuron substrate bridge
 * @param tracking Output tracking structure
 * @return 0 on success, NIMCP_ERROR_NULL_POINTER on error
 */
int neuron_substrate_get_energy_tracking(
    const neuron_substrate_bridge_t* bridge,
    neuron_energy_tracking_t* tracking
);

/**
 * @brief Check if neuron is under substrate modulation
 *
 * @param bridge Neuron substrate bridge
 * @return true if significantly modulated (>5% deviation from baseline)
 */
bool neuron_substrate_is_modulated(const neuron_substrate_bridge_t* bridge);

/**
 * @brief Get current excitability level
 *
 * @param bridge Neuron substrate bridge
 * @return Excitability [0-1.5] (1.0 = normal)
 */
float neuron_substrate_get_excitability(const neuron_substrate_bridge_t* bridge);

/**
 * @brief Get bridge statistics
 *
 * @param bridge Neuron substrate bridge
 * @param stats Output statistics
 * @return 0 on success, NIMCP_ERROR_NULL_POINTER on error
 */
int neuron_substrate_get_stats(
    const neuron_substrate_bridge_t* bridge,
    neuron_substrate_stats_t* stats
);

#ifdef __cplusplus
}
#endif

#endif /* NIMCP_NEURON_SUBSTRATE_BRIDGE_H */
