/**
 * @file nimcp_glial_substrate_bridge.h
 * @brief Neural Substrate-Glial System Integration Bridge
 * @version 1.0.0
 * @date 2025-12-12
 *
 * WHAT: Bidirectional integration between neural substrate and all glial cell types
 *       (astrocytes, microglia, oligodendrocytes, myelin)
 * WHY:  Substrate conditions affect glial function; glial cells provide metabolic
 *       support and regulate substrate health
 * HOW:  Bridge monitors substrate state and modulates glial activity based on ATP,
 *       temperature, oxygen. Glial cells (esp. astrocytes) provide lactate shuttle
 *       metabolic support back to substrate.
 *
 * BIOLOGICAL BASIS:
 * ==================================================================================
 *
 * SUBSTRATE → GLIAL PATHWAYS:
 * ----------------------------
 * 1. ATP Availability Affects Glial Function:
 *    - Astrocyte calcium waves require ATP for IP3/Ca2+ pumps
 *    - Oligodendrocyte myelin synthesis is ATP-expensive
 *    - Microglia surveillance and phagocytosis consume ATP
 *    - Reference: Harris et al. (2012) "The neurovascular unit"
 *
 * 2. Temperature Effects on Glial Cells:
 *    - Hyperthermia (fever) slows glial calcium dynamics
 *    - Hypothermia reduces glial metabolic activity
 *    - Q10 effects similar to neurons (Q10 = 2-3)
 *    - Reference: Hodgkin & Huxley (1952) temperature coefficients
 *
 * 3. Oxygen/Glucose Supply to Glia:
 *    - Astrocytes store glycogen but need continuous glucose
 *    - Oligodendrocytes have high metabolic demand for myelin
 *    - Hypoxia triggers microglial activation (stress response)
 *    - Reference: Attwell & Laughlin (2001) "Energy Budget"
 *
 * GLIAL → SUBSTRATE PATHWAYS:
 * ----------------------------
 * 1. Astrocyte-Neuron Lactate Shuttle (ANLS):
 *    - Astrocytes convert glucose → lactate
 *    - Lactate transported to neurons via MCT1/MCT2
 *    - Lactate provides ~30% of neuronal energy during activity
 *    - Increases substrate ATP availability
 *    - Reference: Magistretti & Allaman (2015) "Lactate shuttle"
 *
 * 2. Microglia Metabolic Support:
 *    - Microglia clear debris → reduce metabolic burden
 *    - Activated microglia consume oxygen (may reduce substrate O2)
 *    - Pruning weak synapses improves network efficiency
 *    - Reference: Kettenmann et al. (2013) "Microglia physiology"
 *
 * 3. Oligodendrocyte Lactate Support:
 *    - Oligodendrocytes provide lactate to axons
 *    - Similar shuttle mechanism to astrocytes
 *    - Critical for long-term axon health
 *    - Reference: Saab et al. (2013) "Myelin lactate transfer"
 *
 * 4. Myelin Integrity Affects Substrate:
 *    - Healthy myelin reduces action potential ATP cost (10-100x more efficient)
 *    - Damaged myelin increases metabolic demand
 *    - Conduction block increases heat dissipation
 *    - Reference: Harris & Attwell (2012) "Energetics of CNS white matter"
 *
 * ARCHITECTURE:
 * ```
 * ╔═══════════════════════════════════════════════════════════════════════════╗
 * ║              GLIAL-SUBSTRATE INTEGRATION BRIDGE                            ║
 * ╠═══════════════════════════════════════════════════════════════════════════╣
 * ║                                                                            ║
 * ║   ┌────────────────────────────────────────────────────────────────────┐  ║
 * ║   │               SUBSTRATE → GLIAL PATHWAYS                            │  ║
 * ║   │                                                                     │  ║
 * ║   │   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐          │  ║
 * ║   │   │   Low ATP    │   │  High Temp   │   │   Low O2     │          │  ║
 * ║   │   │ ──────────── │   │ ──────────── │   │ ──────────── │          │  ║
 * ║   │   │ -Calcium     │   │ -Calcium     │   │ +Microglia   │          │  ║
 * ║   │   │ -Myelin      │   │ -Myelin      │   │ -Myelin      │          │  ║
 * ║   │   │ -Surveillance│   │ -Metabolism  │   │ Stress       │          │  ║
 * ║   │   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘          │  ║
 * ║   │          └──────────────────┴──────────────────┘                   │  ║
 * ║   │                             ↓                                      │  ║
 * ║   │   ┌─────────────────────────────────────────────────────────────┐ │  ║
 * ║   │   │              GLIAL FUNCTION MODULATION                      │ │  ║
 * ║   │   │  • Astrocyte calcium wave propagation scaled                │ │  ║
 * ║   │   │  • Oligodendrocyte myelin production rate limited           │ │  ║
 * ║   │   │  • Microglia surveillance activity modulated                │ │  ║
 * ║   │   │  • Myelin integrity maintenance affected                    │ │  ║
 * ║   │   └─────────────────────────────────────────────────────────────┘ │  ║
 * ║   └────────────────────────────────────────────────────────────────────┘  ║
 * ║                                                                            ║
 * ║   ┌────────────────────────────────────────────────────────────────────┐  ║
 * ║   │               GLIAL → SUBSTRATE PATHWAYS                            │  ║
 * ║   │                                                                     │  ║
 * ║   │   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐          │  ║
 * ║   │   │  Astrocyte   │   │    Oligo     │   │   Myelin     │          │  ║
 * ║   │   │  Lactate     │   │   Lactate    │   │  Efficiency  │          │  ║
 * ║   │   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘          │  ║
 * ║   │          └──────────────────┴──────────────────┘                   │  ║
 * ║   │                             ↓                                      │  ║
 * ║   │   ┌─────────────────────────────────────────────────────────────┐ │  ║
 * ║   │   │             SUBSTRATE METABOLIC SUPPORT                     │ │  ║
 * ║   │   │  • ATP level increase (lactate shuttle)                     │ │  ║
 * ║   │   │  • Metabolic efficiency improvement (myelin)                │ │  ║
 * ║   │   │  • Network pruning reduces baseline cost (microglia)        │ │  ║
 * ║   │   └─────────────────────────────────────────────────────────────┘ │  ║
 * ║   └────────────────────────────────────────────────────────────────────┘  ║
 * ║                                                                            ║
 * ╚═══════════════════════════════════════════════════════════════════════════╝
 * ```
 *
 * DESIGN PATTERNS:
 * - Facade: Unified interface for all glial-substrate interactions
 * - Observer: Substrate state changes notify glial cells
 * - Strategy: Different glial types respond differently to substrate state
 *
 * @author NIMCP Development Team
 */

#ifndef NIMCP_GLIAL_SUBSTRATE_BRIDGE_H
#define NIMCP_GLIAL_SUBSTRATE_BRIDGE_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

/* Module integrations */
#include "core/neural_substrate/nimcp_neural_substrate.h"
#include "glial/astrocytes/nimcp_astrocytes.h"
#include "glial/microglia/nimcp_microglia.h"
#include "glial/oligodendrocytes/nimcp_oligodendrocytes.h"
#include "glial/myelin_sheath/nimcp_myelin_sheath.h"

/* Bio-async integration */
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"

/* Thread utilities */
#include "utils/thread/nimcp_thread.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/validation/nimcp_common.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Constants
 * ============================================================================ */

/* Substrate effects on astrocytes */
#define SUBSTRATE_ATP_ASTRO_CALCIUM_FACTOR      0.8f    /**< ATP effect on Ca2+ wave */
#define SUBSTRATE_ATP_ASTRO_THRESHOLD           0.3f    /**< ATP threshold for impairment */
#define SUBSTRATE_TEMP_ASTRO_Q10                2.5f    /**< Q10 for astrocyte dynamics */
#define SUBSTRATE_O2_ASTRO_THRESHOLD            0.5f    /**< O2 threshold for stress */

/* Substrate effects on oligodendrocytes */
#define SUBSTRATE_ATP_OLIGO_MYELIN_FACTOR       0.5f    /**< ATP effect on myelination */
#define SUBSTRATE_ATP_OLIGO_THRESHOLD           0.4f    /**< ATP threshold for myelin */
#define SUBSTRATE_TEMP_OLIGO_Q10                2.2f    /**< Q10 for myelin synthesis */
#define SUBSTRATE_O2_OLIGO_THRESHOLD            0.6f    /**< O2 threshold for function */

/* Substrate effects on microglia */
#define SUBSTRATE_ATP_MICRO_SURVEILLANCE_FACTOR 0.7f    /**< ATP effect on surveillance */
#define SUBSTRATE_ATP_MICRO_THRESHOLD           0.35f   /**< ATP threshold for activity */
#define SUBSTRATE_O2_MICRO_ACTIVATION           0.5f    /**< Hypoxia activates microglia */
#define SUBSTRATE_TEMP_MICRO_Q10                2.3f    /**< Q10 for microglia */

/* Substrate effects on myelin */
#define SUBSTRATE_ATP_MYELIN_MAINTENANCE        0.001f  /**< ATP cost per segment per ms */
#define SUBSTRATE_TEMP_MYELIN_INTEGRITY_LOSS    0.01f   /**< Integrity loss per °C above 40 */
#define SUBSTRATE_O2_MYELIN_DAMAGE_RATE         0.005f  /**< Hypoxia damage rate */

/* Glial support to substrate */
#define ASTRO_LACTATE_ATP_CONVERSION            0.30f   /**< Lactate → ATP efficiency */
#define ASTRO_LACTATE_PRODUCTION_RATE           0.10f   /**< Lactate per astrocyte per ms */
#define OLIGO_LACTATE_ATP_CONVERSION            0.25f   /**< Oligo lactate → ATP */
#define OLIGO_LACTATE_PRODUCTION_RATE           0.05f   /**< Lactate per oligo per ms */
#define MYELIN_EFFICIENCY_ATP_SAVINGS           0.90f   /**< ATP savings with full myelin */
#define MICROGLIA_PRUNING_ATP_SAVINGS           0.001f  /**< ATP saved per pruned synapse */

/* Bio-async module ID */
#define BIO_MODULE_GLIAL_SUBSTRATE              0x0D30  /**< Bio-async module ID */

/* ============================================================================
 * Structures
 * ============================================================================ */

/**
 * @brief Substrate effects on astrocyte function
 */
typedef struct {
    /* Metabolic effects */
    float atp_modulation;           /**< ATP effect on calcium dynamics [0-1] */
    float calcium_wave_factor;      /**< Wave propagation speed factor */
    float glutamate_release_factor; /**< Glutamate release modulation */

    /* Temperature effects */
    float temp_q10_factor;          /**< Q10 temperature correction */
    float metabolic_rate_factor;    /**< Overall metabolic rate */

    /* Oxygen effects */
    bool hypoxia_stress;            /**< Hypoxia stress state */
    float o2_modulation;            /**< O2 effect on function [0-1] */
} substrate_astrocyte_effects_t;

/**
 * @brief Substrate effects on oligodendrocyte function
 */
typedef struct {
    /* Metabolic effects */
    float atp_modulation;           /**< ATP effect on myelin synthesis [0-1] */
    float myelin_production_rate;   /**< Myelin synthesis rate factor */
    float g_ratio_optimization;     /**< G-ratio optimization capacity */

    /* Temperature effects */
    float temp_q10_factor;          /**< Q10 temperature correction */
    float maturation_rate;          /**< Maturation speed factor */

    /* Oxygen effects */
    bool hypoxia_stress;            /**< Hypoxia stress state */
    float o2_modulation;            /**< O2 effect on function [0-1] */
} substrate_oligodendrocyte_effects_t;

/**
 * @brief Substrate effects on microglia function
 */
typedef struct {
    /* Metabolic effects */
    float atp_modulation;           /**< ATP effect on surveillance [0-1] */
    float surveillance_radius;      /**< Effective surveillance radius */
    float pruning_threshold;        /**< Activity threshold for pruning */

    /* Temperature effects */
    float temp_q10_factor;          /**< Q10 temperature correction */
    float activation_rate;          /**< State transition speed */

    /* Oxygen effects */
    bool hypoxia_activation;        /**< Hypoxia-triggered activation */
    float o2_modulation;            /**< O2 effect on function [0-1] */
} substrate_microglia_effects_t;

/**
 * @brief Substrate effects on myelin integrity
 */
typedef struct {
    /* Metabolic effects */
    float atp_maintenance_cost;     /**< ATP cost per segment per ms */
    bool insufficient_atp;          /**< ATP below maintenance threshold */
    float integrity_decay_rate;     /**< Integrity loss rate */

    /* Temperature effects */
    bool hyperthermia_damage;       /**< Temperature-induced damage */
    float temp_damage_rate;         /**< Damage accumulation rate */
    float conduction_block_prob;    /**< Block probability increase */

    /* Oxygen effects */
    bool hypoxia_damage;            /**< Hypoxia-induced damage */
    float o2_damage_rate;           /**< Hypoxia damage rate */
} substrate_myelin_effects_t;

/**
 * @brief Glial metabolic support to substrate
 */
typedef struct {
    /* Astrocyte lactate shuttle */
    float astro_lactate_total;      /**< Total astrocyte lactate (mM) */
    float astro_atp_contribution;   /**< ATP from astrocyte lactate */
    uint32_t astro_active_count;    /**< Active astrocytes */

    /* Oligodendrocyte lactate shuttle */
    float oligo_lactate_total;      /**< Total oligo lactate (mM) */
    float oligo_atp_contribution;   /**< ATP from oligo lactate */
    uint32_t oligo_active_count;    /**< Active oligodendrocytes */

    /* Myelin efficiency savings */
    float myelin_atp_savings;       /**< ATP saved by efficient conduction */
    float avg_myelination_factor;   /**< Average myelination [0-1] */

    /* Microglia pruning savings */
    float pruning_atp_savings;      /**< ATP saved by pruning */
    uint32_t synapses_pruned;       /**< Synapses pruned this cycle */

    /* Total support */
    float total_atp_support;        /**< Total ATP provided to substrate */
    float total_metabolic_boost;    /**< Overall metabolic capacity boost */
} glial_substrate_support_t;

/**
 * @brief Bridge statistics
 */
typedef struct {
    uint64_t total_updates;
    uint32_t astrocyte_modulations;
    uint32_t oligo_modulations;
    uint32_t microglia_modulations;
    uint32_t myelin_modulations;
    uint32_t lactate_shuttles;
    float max_atp_support;
    float avg_substrate_health;
    uint32_t stress_events;
} glial_substrate_stats_t;

/**
 * @brief Bridge configuration
 */
typedef struct {
    /* Feature enables */
    bool enable_astrocyte_substrate;
    bool enable_oligo_substrate;
    bool enable_microglia_substrate;
    bool enable_myelin_substrate;
    bool enable_lactate_shuttle;
    bool enable_bio_async;

    /* Sensitivity multipliers */
    float atp_sensitivity;
    float temperature_sensitivity;
    float oxygen_sensitivity;

    /* Support multipliers */
    float lactate_efficiency;
    float myelin_savings_factor;
    float pruning_savings_factor;
} glial_substrate_config_t;

/**
 * @brief Complete glial-substrate bridge state
 */
typedef struct {
    /* System handles */
    neural_substrate_t* substrate;
    astrocyte_network_t* astrocyte_network;
    oligodendrocyte_network_t* oligo_network;
    microglia_network_t* microglia_network;
    myelin_sheath_network_t* myelin_network;

    /* Current effects */
    substrate_astrocyte_effects_t astro_effects;
    substrate_oligodendrocyte_effects_t oligo_effects;
    substrate_microglia_effects_t micro_effects;
    substrate_myelin_effects_t myelin_effects;

    /* Current support */
    glial_substrate_support_t glial_support;

    /* Configuration */
    glial_substrate_config_t config;

    /* Bio-async integration */
    bio_module_context_t bio_ctx;
    bool bio_async_enabled;

    /* Statistics */
    glial_substrate_stats_t stats;

    /* Thread safety */
    nimcp_mutex_t* mutex;
} glial_substrate_bridge_t;

/* ============================================================================
 * Lifecycle API
 * ============================================================================ */

/**
 * @brief Get default configuration
 *
 * WHAT: Provide sensible defaults for bridge
 * WHY:  Easy initialization with biological defaults
 * HOW:  Pre-filled configuration structure
 *
 * @param config Output configuration
 * @return 0 on success, -1 on error
 */
int glial_substrate_default_config(glial_substrate_config_t* config);

/**
 * @brief Create glial-substrate bridge
 *
 * WHAT: Initialize bridge between neural substrate and glial systems
 * WHY:  Enable bidirectional substrate-glial interactions
 * HOW:  Allocate structure, connect systems, initialize effects
 *
 * @param config Configuration (NULL for defaults)
 * @param substrate Neural substrate
 * @param astro_network Astrocyte network (can be NULL)
 * @param oligo_network Oligodendrocyte network (can be NULL)
 * @param micro_network Microglia network (can be NULL)
 * @param myelin_network Myelin sheath network (can be NULL)
 * @return New bridge or NULL on failure
 */
glial_substrate_bridge_t* glial_substrate_bridge_create(
    const glial_substrate_config_t* config,
    neural_substrate_t* substrate,
    astrocyte_network_t* astro_network,
    oligodendrocyte_network_t* oligo_network,
    microglia_network_t* micro_network,
    myelin_sheath_network_t* myelin_network
);

/**
 * @brief Destroy glial-substrate bridge
 *
 * WHAT: Clean up bridge resources
 * WHY:  Proper resource deallocation
 * HOW:  Free structure, disconnect bio-async
 *
 * @param bridge Bridge to destroy (NULL safe)
 */
void glial_substrate_bridge_destroy(glial_substrate_bridge_t* bridge);

/* ============================================================================
 * Connection API
 * ============================================================================ */

/**
 * @brief Connect astrocyte network to bridge
 *
 * @param bridge Glial substrate bridge
 * @param astro_network Astrocyte network
 * @return 0 on success, -1 on error
 */
int glial_substrate_connect_astrocytes(
    glial_substrate_bridge_t* bridge,
    astrocyte_network_t* astro_network
);

/**
 * @brief Connect oligodendrocyte network to bridge
 *
 * @param bridge Glial substrate bridge
 * @param oligo_network Oligodendrocyte network
 * @return 0 on success, -1 on error
 */
int glial_substrate_connect_oligodendrocytes(
    glial_substrate_bridge_t* bridge,
    oligodendrocyte_network_t* oligo_network
);

/**
 * @brief Connect microglia network to bridge
 *
 * @param bridge Glial substrate bridge
 * @param micro_network Microglia network
 * @return 0 on success, -1 on error
 */
int glial_substrate_connect_microglia(
    glial_substrate_bridge_t* bridge,
    microglia_network_t* micro_network
);

/**
 * @brief Connect myelin sheath network to bridge
 *
 * @param bridge Glial substrate bridge
 * @param myelin_network Myelin sheath network
 * @return 0 on success, -1 on error
 */
int glial_substrate_connect_myelin(
    glial_substrate_bridge_t* bridge,
    myelin_sheath_network_t* myelin_network
);

/* ============================================================================
 * Bio-async Integration API
 * ============================================================================ */

/**
 * @brief Connect to bio-async router
 *
 * WHAT: Register bridge with bio-async messaging
 * WHY:  Enable event-driven substrate-glial communication
 * HOW:  Register module with bio-router
 *
 * @param bridge Glial substrate bridge
 * @return 0 on success, -1 on error
 */
int glial_substrate_connect_bio_async(glial_substrate_bridge_t* bridge);

/**
 * @brief Disconnect from bio-async router
 *
 * @param bridge Glial substrate bridge
 * @return 0 on success, -1 on error
 */
int glial_substrate_disconnect_bio_async(glial_substrate_bridge_t* bridge);

/**
 * @brief Check if bio-async is connected
 *
 * @param bridge Glial substrate bridge
 * @return true if connected
 */
bool glial_substrate_is_bio_async_connected(const glial_substrate_bridge_t* bridge);

/* ============================================================================
 * Substrate → Glial API (Substrate affects glial function)
 * ============================================================================ */

/**
 * @brief Update astrocyte effects from substrate
 *
 * WHAT: Compute substrate effects on astrocyte function
 * WHY:  ATP, temperature, oxygen affect astrocyte calcium dynamics
 * HOW:  Query substrate state, compute modulation factors
 *
 * BIOLOGICAL BASIS:
 * - Low ATP impairs calcium pumps → reduced wave propagation
 * - Temperature affects enzyme kinetics (Q10 = 2.5)
 * - Hypoxia triggers astrocyte stress response
 *
 * @param bridge Glial substrate bridge
 * @return 0 on success, -1 on error
 */
int glial_substrate_update_astrocyte_effects(glial_substrate_bridge_t* bridge);

/**
 * @brief Update oligodendrocyte effects from substrate
 *
 * WHAT: Compute substrate effects on oligodendrocyte function
 * WHY:  ATP, temperature, oxygen affect myelin synthesis
 * HOW:  Query substrate state, compute myelin production rate
 *
 * BIOLOGICAL BASIS:
 * - Myelin synthesis is ATP-expensive (requires continuous supply)
 * - Temperature affects maturation rate (Q10 = 2.2)
 * - Hypoxia impairs myelin production
 *
 * @param bridge Glial substrate bridge
 * @return 0 on success, -1 on error
 */
int glial_substrate_update_oligodendrocyte_effects(glial_substrate_bridge_t* bridge);

/**
 * @brief Update microglia effects from substrate
 *
 * WHAT: Compute substrate effects on microglia function
 * WHY:  ATP, oxygen affect surveillance; hypoxia activates microglia
 * HOW:  Query substrate state, modulate surveillance and pruning
 *
 * BIOLOGICAL BASIS:
 * - Surveillance requires ATP for process extension
 * - Hypoxia is a danger signal → activates microglia
 * - Temperature affects motility and phagocytosis
 *
 * @param bridge Glial substrate bridge
 * @return 0 on success, -1 on error
 */
int glial_substrate_update_microglia_effects(glial_substrate_bridge_t* bridge);

/**
 * @brief Update myelin effects from substrate
 *
 * WHAT: Compute substrate effects on myelin integrity
 * WHY:  ATP, temperature, oxygen affect myelin maintenance
 * HOW:  Query substrate state, compute damage/maintenance rates
 *
 * BIOLOGICAL BASIS:
 * - Myelin maintenance requires continuous ATP
 * - Hyperthermia causes myelin damage (protein denaturation)
 * - Hypoxia leads to demyelination
 *
 * @param bridge Glial substrate bridge
 * @return 0 on success, -1 on error
 */
int glial_substrate_update_myelin_effects(glial_substrate_bridge_t* bridge);

/* ============================================================================
 * Glial → Substrate API (Glial cells support substrate)
 * ============================================================================ */

/**
 * @brief Compute astrocyte lactate shuttle support
 *
 * WHAT: Calculate ATP support from astrocyte lactate production
 * WHY:  Astrocytes provide 30% of neuronal energy via lactate
 * HOW:  Sum lactate production across astrocytes, convert to ATP
 *
 * BIOLOGICAL BASIS:
 * - Astrocyte-Neuron Lactate Shuttle (ANLS)
 * - Glucose → lactate in astrocytes
 * - Lactate → pyruvate → ATP in neurons
 * - Reference: Magistretti & Allaman (2015)
 *
 * @param bridge Glial substrate bridge
 * @return 0 on success, -1 on error
 */
int glial_substrate_compute_astrocyte_support(glial_substrate_bridge_t* bridge);

/**
 * @brief Compute oligodendrocyte lactate shuttle support
 *
 * WHAT: Calculate ATP support from oligodendrocyte lactate
 * WHY:  Oligodendrocytes provide lactate to axons
 * HOW:  Sum lactate production across oligodendrocytes
 *
 * BIOLOGICAL BASIS:
 * - Similar to ANLS but for axons
 * - Critical for long-term axon health
 * - Reference: Saab et al. (2013)
 *
 * @param bridge Glial substrate bridge
 * @return 0 on success, -1 on error
 */
int glial_substrate_compute_oligodendrocyte_support(glial_substrate_bridge_t* bridge);

/**
 * @brief Compute myelin efficiency savings
 *
 * WHAT: Calculate ATP savings from myelinated conduction
 * WHY:  Myelin reduces action potential ATP cost by 10-100x
 * HOW:  Query myelin network, compute efficiency factor
 *
 * BIOLOGICAL BASIS:
 * - Saltatory conduction bypasses most axon membrane
 * - Reduces Na+/K+-ATPase pump activity
 * - Reference: Harris & Attwell (2012)
 *
 * @param bridge Glial substrate bridge
 * @return 0 on success, -1 on error
 */
int glial_substrate_compute_myelin_support(glial_substrate_bridge_t* bridge);

/**
 * @brief Compute microglia pruning savings
 *
 * WHAT: Calculate ATP savings from synaptic pruning
 * WHY:  Pruning weak synapses reduces baseline metabolic cost
 * HOW:  Track pruned synapses, sum ATP savings
 *
 * BIOLOGICAL BASIS:
 * - Each synapse has maintenance ATP cost
 * - Pruning reduces unnecessary energy expenditure
 * - Improves network efficiency
 *
 * @param bridge Glial substrate bridge
 * @return 0 on success, -1 on error
 */
int glial_substrate_compute_microglia_support(glial_substrate_bridge_t* bridge);

/**
 * @brief Apply glial metabolic support to substrate
 *
 * WHAT: Increase substrate ATP based on glial support
 * WHY:  Glial cells provide metabolic support to neurons
 * HOW:  Sum all support sources, increase substrate ATP
 *
 * @param bridge Glial substrate bridge
 * @return 0 on success, -1 on error
 */
int glial_substrate_apply_glial_support(glial_substrate_bridge_t* bridge);

/* ============================================================================
 * Bidirectional Update API
 * ============================================================================ */

/**
 * @brief Update all substrate effects on glial cells
 *
 * WHAT: Update effects on all glial types
 * WHY:  Substrate state affects all glial functions
 * HOW:  Call individual update functions
 *
 * @param bridge Glial substrate bridge
 * @return 0 on success, -1 on error
 */
int glial_substrate_update_all_effects(glial_substrate_bridge_t* bridge);

/**
 * @brief Compute all glial support to substrate
 *
 * WHAT: Compute support from all glial types
 * WHY:  Multiple glial types contribute to substrate health
 * HOW:  Call individual support functions, sum contributions
 *
 * @param bridge Glial substrate bridge
 * @return 0 on success, -1 on error
 */
int glial_substrate_compute_all_support(glial_substrate_bridge_t* bridge);

/**
 * @brief Update glial-substrate bridge (bidirectional)
 *
 * WHAT: Process all glial-substrate interactions
 * WHY:  Advance coupled state machine
 * HOW:  Update substrate effects, compute support, apply to substrate
 *
 * @param bridge Glial substrate bridge
 * @param delta_ms Time since last update
 * @return 0 on success, -1 on error
 */
int glial_substrate_bridge_update(
    glial_substrate_bridge_t* bridge,
    uint64_t delta_ms
);

/* ============================================================================
 * Query API
 * ============================================================================ */

/**
 * @brief Get astrocyte effects
 *
 * @param bridge Glial substrate bridge
 * @param effects Output effects structure
 * @return 0 on success, -1 on error
 */
int glial_substrate_get_astrocyte_effects(
    const glial_substrate_bridge_t* bridge,
    substrate_astrocyte_effects_t* effects
);

/**
 * @brief Get oligodendrocyte effects
 *
 * @param bridge Glial substrate bridge
 * @param effects Output effects structure
 * @return 0 on success, -1 on error
 */
int glial_substrate_get_oligodendrocyte_effects(
    const glial_substrate_bridge_t* bridge,
    substrate_oligodendrocyte_effects_t* effects
);

/**
 * @brief Get microglia effects
 *
 * @param bridge Glial substrate bridge
 * @param effects Output effects structure
 * @return 0 on success, -1 on error
 */
int glial_substrate_get_microglia_effects(
    const glial_substrate_bridge_t* bridge,
    substrate_microglia_effects_t* effects
);

/**
 * @brief Get myelin effects
 *
 * @param bridge Glial substrate bridge
 * @param effects Output effects structure
 * @return 0 on success, -1 on error
 */
int glial_substrate_get_myelin_effects(
    const glial_substrate_bridge_t* bridge,
    substrate_myelin_effects_t* effects
);

/**
 * @brief Get glial support summary
 *
 * @param bridge Glial substrate bridge
 * @param support Output support structure
 * @return 0 on success, -1 on error
 */
int glial_substrate_get_support(
    const glial_substrate_bridge_t* bridge,
    glial_substrate_support_t* support
);

/**
 * @brief Get total ATP support from glia
 *
 * @param bridge Glial substrate bridge
 * @return Total ATP support (arbitrary units)
 */
float glial_substrate_get_total_atp_support(const glial_substrate_bridge_t* bridge);

/**
 * @brief Get bridge statistics
 *
 * @param bridge Glial substrate bridge
 * @param stats Output statistics
 * @return 0 on success, -1 on error
 */
int glial_substrate_get_stats(
    const glial_substrate_bridge_t* bridge,
    glial_substrate_stats_t* stats
);

#ifdef __cplusplus
}
#endif

#endif /* NIMCP_GLIAL_SUBSTRATE_BRIDGE_H */
