/**
 * @file nimcp_synapse_substrate_bridge.h
 * @brief Synapse-Neural Substrate Integration Bridge
 * @version 1.0.0
 * @date 2025-12-12
 *
 * WHAT: Bidirectional integration between neural substrate and synaptic transmission
 * WHY:  Substrate state (ATP, temperature, Ca2+) fundamentally constrains synaptic function.
 *       Energy depletion reduces vesicle release, temperature affects kinetics, Ca2+ gates transmission.
 * HOW:  Substrate modulates transmission probability, release efficiency, receptor kinetics per synapse type.
 *
 * BIOLOGICAL BASIS:
 * ==================================================================================
 *
 * SUBSTRATE → SYNAPSE PATHWAYS:
 * ------------------------------
 * 1. ATP-Dependent Vesicle Release:
 *    - Vesicle mobilization requires ATP (SNARE complex priming)
 *    - Vesicle recycling via endocytosis requires ATP
 *    - Low ATP → reduced release probability (fewer vesicles available)
 *    - Reference: Harris et al. (2012) "The energetics of CNS white matter"
 *
 * 2. Calcium-Dependent Transmission:
 *    - Ca2+ influx triggers vesicle fusion (synaptotagmin)
 *    - Ca2+ homeostasis regulates release probability
 *    - Ca2+ overload → excitotoxicity, impaired transmission
 *    - Reference: Sudhof (2013) "Neurotransmitter release: The last millisecond"
 *
 * 3. Temperature Effects on Receptor Kinetics:
 *    - Q10 = 2-3 for most channel kinetics
 *    - Hyperthermia → faster rise/decay (less temporal precision)
 *    - Hypothermia → slower kinetics (impaired transmission)
 *    - Reference: Hodgkin & Huxley (1952) "Temperature coefficients"
 *
 * 4. Membrane Integrity and Receptor Function:
 *    - Receptor insertion/anchoring requires intact membrane
 *    - Lipid composition affects channel gating
 *    - Membrane damage → reduced receptor density
 *    - Reference: Bhargava et al. (2013) "Lipid-protein interactions"
 *
 * 5. Ion Gradients and Synaptic Driving Force:
 *    - Ion imbalance alters reversal potentials
 *    - Reduced driving force → smaller synaptic currents
 *    - Na+/K+ pump dysfunction → excitability changes
 *    - Reference: Bhardwaj et al. (2016) "Ion homeostasis"
 *
 * SYNAPSE → SUBSTRATE PATHWAYS:
 * ------------------------------
 * 1. Synaptic Transmission Energy Cost:
 *    - Each transmission consumes ATP (vesicle recycling, pumps)
 *    - High-frequency firing → ATP depletion
 *    - Substrate must track synaptic energy consumption
 *    - Reference: Attwell & Laughlin (2001) "Energy budget for signaling"
 *
 * 2. Calcium Accumulation from Repeated Transmission:
 *    - Repeated Ca2+ influx can overwhelm buffering
 *    - Sustained high Ca2+ → mitochondrial stress
 *    - Substrate Ca2+ homeostasis tracks accumulated load
 *    - Reference: Berridge et al. (2003) "Calcium signaling"
 *
 * SYNAPSE TYPE-SPECIFIC MODULATION:
 * ----------------------------------
 * 1. AMPA (Fast Excitatory):
 *    - Most sensitive to temperature (fast kinetics)
 *    - Q10 ~ 2.5 for rise/decay times
 *    - Less ATP-dependent (simple ionotropic)
 *
 * 2. NMDA (Slow Excitatory):
 *    - Mg2+ block temperature-dependent
 *    - Ca2+ permeability → substrate Ca2+ load
 *    - Moderate ATP dependence
 *
 * 3. GABA-A (Fast Inhibitory):
 *    - Temperature-sensitive kinetics (Q10 ~ 2.0)
 *    - Low ATP dependence (ionotropic)
 *    - Critical for temperature-regulated oscillations
 *
 * 4. GABA-B (Slow Inhibitory):
 *    - G-protein cascade → high ATP dependence
 *    - Slower temperature sensitivity (Q10 ~ 1.5)
 *    - Metabotropic machinery sensitive to energy state
 *
 * 5. Neuromodulatory (Dopamine, Serotonin, ACh):
 *    - Very high ATP dependence (second messenger cascades)
 *    - Temperature affects cascade kinetics
 *    - Most vulnerable to metabolic stress
 *
 * ARCHITECTURE:
 * ```
 * ╔═══════════════════════════════════════════════════════════════════════════╗
 * ║              SYNAPSE-SUBSTRATE INTEGRATION BRIDGE                          ║
 * ╠═══════════════════════════════════════════════════════════════════════════╣
 * ║                                                                            ║
 * ║   ┌────────────────────────────────────────────────────────────────────┐  ║
 * ║   │               SUBSTRATE → SYNAPSE PATHWAYS                          │  ║
 * ║   │                                                                     │  ║
 * ║   │   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐          │  ║
 * ║   │   │   ATP Level  │   │ Temperature  │   │ Ca2+ Balance │          │  ║
 * ║   │   │ ──────────── │   │ ──────────── │   │ ──────────── │          │  ║
 * ║   │   │ Release Prob │   │ Kinetics Q10 │   │ Transmission │          │  ║
 * ║   │   │ Vesicle Pool │   │ Rise/Decay τ │   │ Probability  │          │  ║
 * ║   │   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘          │  ║
 * ║   │          └──────────────────┴──────────────────┘                   │  ║
 * ║   │                             ↓                                      │  ║
 * ║   │   ┌─────────────────────────────────────────────────────────────┐ │  ║
 * ║   │   │         PER-SYNAPSE-TYPE MODULATION                         │ │  ║
 * ║   │   │  • transmission_efficiency: Overall strength [0-1]          │ │  ║
 * ║   │   │  • release_probability: Vesicle release pr [0-1]            │ │  ║
 * ║   │   │  • receptor_kinetics: τ scaling factor [0.5-1.5]            │ │  ║
 * ║   │   │  • driving_force_factor: Ion gradient effect [0-1]          │ │  ║
 * ║   │   └─────────────────────────────────────────────────────────────┘ │  ║
 * ║   └────────────────────────────────────────────────────────────────────┘  ║
 * ║                                                                            ║
 * ║   ┌────────────────────────────────────────────────────────────────────┐  ║
 * ║   │               SYNAPSE → SUBSTRATE PATHWAYS                          │  ║
 * ║   │                                                                     │  ║
 * ║   │   ┌──────────────┐   ┌──────────────┐                             │  ║
 * ║   │   │ Transmission │   │ NMDA Ca2+    │                             │  ║
 * ║   │   │ Event        │   │ Influx       │                             │  ║
 * ║   │   └──────┬───────┘   └──────┬───────┘                             │  ║
 * ║   │          └──────────────────┘                                      │  ║
 * ║   │                   ↓                                                │  ║
 * ║   │   ┌─────────────────────────────────────────────────────────────┐ │  ║
 * ║   │   │          SUBSTRATE CONSUMPTION                              │ │  ║
 * ║   │   │  • ATP depletion (vesicle recycling)                        │ │  ║
 * ║   │   │  • Ca2+ accumulation (NMDA influx)                          │ │  ║
 * ║   │   └─────────────────────────────────────────────────────────────┘ │  ║
 * ║   └────────────────────────────────────────────────────────────────────┘  ║
 * ║                                                                            ║
 * ╚═══════════════════════════════════════════════════════════════════════════╝
 * ```
 *
 * @author NIMCP Development Team
 */

#ifndef NIMCP_SYNAPSE_SUBSTRATE_BRIDGE_H
#define NIMCP_SYNAPSE_SUBSTRATE_BRIDGE_H

#include <stdint.h>
#include "utils/bridge/nimcp_bridge_base.h"
#include <stdbool.h>
#include <stddef.h>

/* Module integrations */
#include "core/neural_substrate/nimcp_neural_substrate.h"
#include "core/synapse_types/nimcp_synapse_types.h"
#include "core/synapse_compute/nimcp_synapse_compute.h"

/* Thread utilities */
#include "utils/thread/nimcp_thread.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Constants
 * ============================================================================ */

/* ATP effects on transmission */
#define ATP_RELEASE_THRESHOLD           0.3f    /**< Below this, release impaired */
#define ATP_NORMAL_RELEASE              0.8f    /**< Above this, full release capacity */
#define ATP_VESICLE_POOL_SENSITIVITY    0.5f    /**< Vesicle pool scaling factor */

/* Calcium effects on transmission */
#define CA_NORMAL_LEVEL                 0.95f   /**< Normal Ca2+ homeostasis */
#define CA_DEPLETION_THRESHOLD          0.5f    /**< Low Ca2+ impairs transmission */
#define CA_OVERLOAD_THRESHOLD           0.7f    /**< High Ca2+ indicates stress */
#define CA_TRANSMISSION_SENSITIVITY     0.8f    /**< Ca2+ transmission scaling */

/* Temperature Q10 coefficients per synapse type */
#define Q10_AMPA                        2.5f    /**< AMPA temperature coefficient */
#define Q10_NMDA                        2.0f    /**< NMDA temperature coefficient */
#define Q10_GABA_A                      2.0f    /**< GABA-A temperature coefficient */
#define Q10_GABA_B                      1.5f    /**< GABA-B temperature coefficient */
#define Q10_NEUROMOD                    1.8f    /**< Neuromodulator Q10 */
#define Q10_ELECTRICAL                  1.2f    /**< Gap junction Q10 */

/* Membrane integrity effects */
#define MEMBRANE_RECEPTOR_THRESHOLD     0.6f    /**< Below this, receptors degraded */
#define MEMBRANE_RECEPTOR_SENSITIVITY   0.4f    /**< Receptor density scaling */

/* Ion balance effects */
#define ION_DRIVING_FORCE_THRESHOLD     0.5f    /**< Ion imbalance impairs driving force */
#define ION_DRIVING_FORCE_SENSITIVITY   0.3f    /**< Driving force scaling */

/* ATP costs per transmission (synapse type dependent) */
#define ATP_COST_AMPA                   0.0002f /**< AMPA: low cost (ionotropic) */
#define ATP_COST_NMDA                   0.0003f /**< NMDA: moderate cost (Ca2+ handling) */
#define ATP_COST_GABA_A                 0.0002f /**< GABA-A: low cost (ionotropic) */
#define ATP_COST_GABA_B                 0.0005f /**< GABA-B: high cost (metabotropic) */
#define ATP_COST_DOPAMINE               0.0008f /**< Dopamine: very high (cascades) */
#define ATP_COST_SEROTONIN              0.0008f /**< Serotonin: very high (cascades) */
#define ATP_COST_ACETYLCHOLINE          0.0006f /**< ACh: high (dual receptors) */
#define ATP_COST_ELECTRICAL             0.0001f /**< Gap junction: minimal */

/* NMDA calcium influx per transmission */
#define NMDA_CA_INFLUX_PER_EVENT        0.001f  /**< Ca2+ per NMDA transmission */

/* ============================================================================
 * Structures
 * ============================================================================ */

/**
 * @brief Substrate effects on synapse transmission
 */
typedef struct {
    /* Overall modulation factors */
    float transmission_efficiency;      /**< Combined transmission strength [0-1] */
    float release_probability;          /**< Vesicle release probability [0-1] */
    float receptor_kinetics_factor;     /**< Time constant scaling [0.5-1.5] */
    float driving_force_factor;         /**< Ion gradient effect [0-1] */

    /* Component effects */
    float atp_release_effect;           /**< ATP effect on release [0-1] */
    float ca_transmission_effect;       /**< Ca2+ effect on transmission [0-1] */
    float temperature_kinetics_q10;     /**< Q10 factor for current temp */
    float membrane_receptor_effect;     /**< Membrane integrity effect [0-1] */
    float ion_driving_force_effect;     /**< Ion balance effect [0-1] */

    /* Per-synapse-type modulation (Strategy Pattern) */
    float ampa_modulation;              /**< AMPA-specific factor [0-1] */
    float nmda_modulation;              /**< NMDA-specific factor [0-1] */
    float gaba_a_modulation;            /**< GABA-A-specific factor [0-1] */
    float gaba_b_modulation;            /**< GABA-B-specific factor [0-1] */
    float dopamine_modulation;          /**< Dopamine-specific factor [0-1] */
    float serotonin_modulation;         /**< Serotonin-specific factor [0-1] */
    float acetylcholine_modulation;     /**< ACh-specific factor [0-1] */
    float electrical_modulation;        /**< Gap junction-specific factor [0-1] */
} substrate_synapse_effects_t;

/**
 * @brief Bridge statistics
 */
typedef struct {
    uint64_t total_updates;
    uint64_t transmissions_processed;
    uint64_t nmda_ca_influx_events;
    float total_atp_consumed_by_synapses;
    float min_transmission_efficiency;
    float max_temperature_kinetics_factor;
    uint32_t low_atp_impairments;
    uint32_t ca_overload_events;
} synapse_substrate_stats_t;

/**
 * @brief Bridge configuration
 */
typedef struct {
    /* Feature enables */
    bool enable_atp_modulation;
    bool enable_ca_modulation;
    bool enable_temperature_modulation;
    bool enable_membrane_modulation;
    bool enable_ion_modulation;
    bool enable_transmission_cost;
    bool enable_nmda_ca_tracking;

    /* Sensitivity multipliers */
    float atp_sensitivity;
    float ca_sensitivity;
    float temperature_sensitivity;
    float membrane_sensitivity;
    float ion_sensitivity;

    /* ATP cost scaling */
    float atp_cost_multiplier;
} synapse_substrate_config_t;

/**
 * @brief Complete synapse-substrate bridge state
 */
typedef struct {
    
    bridge_base_t base;                 /* MUST be first: base bridge infrastructure */
    /* System handles */
    synapse_compute_context_t* synapse_context;
    neural_substrate_t* substrate;

    /* Current effects */
    substrate_synapse_effects_t substrate_effects;

    /* Configuration */
    synapse_substrate_config_t config;

    /* Statistics */
    synapse_substrate_stats_t stats;

    } synapse_substrate_bridge_t;

/* ============================================================================
 * Lifecycle API
 * ============================================================================ */

/**
 * @brief Get default configuration
 *
 * WHAT: Provide sensible default bridge configuration
 * WHY:  Easy initialization with biological defaults
 * HOW:  Set all features enabled, sensitivities to 1.0
 *
 * @param config Output configuration
 * @return 0 on success, NIMCP_ERROR_NULL_POINTER if config is NULL
 */
int synapse_substrate_default_config(synapse_substrate_config_t* config);

/**
 * @brief Create synapse-substrate bridge
 *
 * WHAT: Initialize synapse-substrate integration bridge
 * WHY:  Enable bidirectional substrate-synapse coupling
 * HOW:  Allocate structure, link systems, initialize mutex
 *
 * @param config Configuration (NULL for defaults)
 * @param synapse_context Synapse computation context
 * @param substrate Neural substrate
 * @return New bridge or NULL on failure
 */
synapse_substrate_bridge_t* synapse_substrate_bridge_create(
    const synapse_substrate_config_t* config,
    synapse_compute_context_t* synapse_context,
    neural_substrate_t* substrate
);

/**
 * @brief Destroy synapse-substrate bridge
 *
 * WHAT: Clean up bridge resources
 * WHY:  Proper resource deallocation
 * HOW:  Destroy mutex, free structure (NULL safe)
 *
 * @param bridge Bridge to destroy (NULL safe)
 */
void synapse_substrate_bridge_destroy(synapse_substrate_bridge_t* bridge);

/* ============================================================================
 * Substrate → Synapse API
 * ============================================================================ */

/**
 * @brief Update substrate effects on synaptic transmission
 *
 * WHAT: Compute how substrate state affects transmission
 * WHY:  ATP, temperature, Ca2+, membrane, ions all modulate synapses
 * HOW:  Query substrate state, compute per-synapse-type modulation
 *
 * BIOLOGICAL BASIS:
 * - Low ATP → reduced vesicle pool → lower release probability
 * - High/low temperature → altered receptor kinetics (Q10)
 * - Ca2+ imbalance → impaired transmission
 * - Membrane damage → reduced receptor density
 * - Ion imbalance → reduced driving force
 *
 * DESIGN PATTERN: Strategy (per-synapse-type modulation)
 *
 * @param bridge Synapse substrate bridge
 * @return 0 on success, NIMCP_ERROR_NULL_POINTER if bridge is NULL
 */
int synapse_substrate_update(synapse_substrate_bridge_t* bridge);

/**
 * @brief Apply substrate modulation to synapse type
 *
 * WHAT: Get modulation factor for specific synapse type
 * WHY:  Different synapse types have different substrate sensitivities
 * HOW:  Return precomputed type-specific modulation from substrate_effects
 *
 * USAGE:
 * ```c
 * float mod = synapse_substrate_apply_modulation(bridge, SYNAPSE_AMPA);
 * float effective_weight = base_weight * mod;
 * ```
 *
 * @param bridge Synapse substrate bridge
 * @param synapse_type Type of synapse (AMPA, NMDA, etc.)
 * @return Modulation factor [0-1], or 1.0 on error (no modulation)
 */
float synapse_substrate_apply_modulation(
    const synapse_substrate_bridge_t* bridge,
    synapse_type_t synapse_type
);

/* ============================================================================
 * Synapse → Substrate API
 * ============================================================================ */

/**
 * @brief Record synaptic transmission (consumes ATP)
 *
 * WHAT: Notify substrate of synaptic transmission event
 * WHY:  Transmission has ATP cost (vesicle recycling, pumps)
 * HOW:  Compute ATP cost based on synapse type, call substrate_record_transmissions
 *
 * BIOLOGICAL BASIS:
 * - Vesicle recycling costs ATP (endocytosis, repriming)
 * - Metabotropic synapses cost more (second messenger cascades)
 * - Gap junctions cost least (passive current flow)
 *
 * ATP COSTS (per transmission):
 * - AMPA: 0.0002 (ionotropic)
 * - NMDA: 0.0003 (Ca2+ handling)
 * - GABA-A: 0.0002 (ionotropic)
 * - GABA-B: 0.0005 (metabotropic)
 * - Dopamine: 0.0008 (cascades)
 * - Serotonin: 0.0008 (cascades)
 * - ACh: 0.0006 (dual receptors)
 * - Electrical: 0.0001 (minimal)
 *
 * @param bridge Synapse substrate bridge
 * @param synapse_type Type of synapse
 * @param count Number of transmission events
 * @return 0 on success
 */
int synapse_substrate_consume_transmission(
    synapse_substrate_bridge_t* bridge,
    synapse_type_t synapse_type,
    uint32_t count
);

/**
 * @brief Record NMDA calcium influx
 *
 * WHAT: Notify substrate of Ca2+ influx from NMDA transmission
 * WHY:  NMDA Ca2+ influx loads substrate Ca2+ homeostasis
 * HOW:  Accumulate Ca2+ based on NMDA transmission events
 *
 * BIOLOGICAL BASIS:
 * - NMDA receptors are Ca2+-permeable
 * - Repeated Ca2+ influx can overwhelm buffering
 * - Substrate tracks Ca2+ accumulation for homeostasis
 *
 * @param bridge Synapse substrate bridge
 * @param transmission_count Number of NMDA transmissions
 * @return 0 on success
 */
int synapse_substrate_record_nmda_calcium(
    synapse_substrate_bridge_t* bridge,
    uint32_t transmission_count
);

/* ============================================================================
 * Query API
 * ============================================================================ */

/**
 * @brief Get current substrate effects on synapses
 *
 * @param bridge Synapse substrate bridge
 * @param effects Output substrate effects
 * @return 0 on success
 */
int synapse_substrate_get_effects(
    const synapse_substrate_bridge_t* bridge,
    substrate_synapse_effects_t* effects
);

/**
 * @brief Get transmission efficiency (overall)
 *
 * @param bridge Synapse substrate bridge
 * @return Transmission efficiency [0-1]
 */
float synapse_substrate_get_transmission_efficiency(
    const synapse_substrate_bridge_t* bridge
);

/**
 * @brief Get release probability (ATP-dependent)
 *
 * @param bridge Synapse substrate bridge
 * @return Release probability [0-1]
 */
float synapse_substrate_get_release_probability(
    const synapse_substrate_bridge_t* bridge
);

/**
 * @brief Get receptor kinetics factor (temperature-dependent)
 *
 * @param bridge Synapse substrate bridge
 * @return Kinetics scaling factor [0.5-1.5]
 */
float synapse_substrate_get_receptor_kinetics_factor(
    const synapse_substrate_bridge_t* bridge
);

/**
 * @brief Get bridge statistics
 *
 * @param bridge Synapse substrate bridge
 * @param stats Output statistics
 * @return 0 on success
 */
int synapse_substrate_get_stats(
    const synapse_substrate_bridge_t* bridge,
    synapse_substrate_stats_t* stats
);

#ifdef __cplusplus
}
#endif

#endif /* NIMCP_SYNAPSE_SUBSTRATE_BRIDGE_H */
