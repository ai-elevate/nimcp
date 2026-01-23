/**
 * @file nimcp_synapse_substrate_bridge.c
 * @brief Synapse-Neural Substrate Integration Implementation
 * @version 1.0.0
 * @date 2025-12-12
 *
 * WHAT: Bidirectional synapse-substrate coupling
 * WHY:  Substrate state constrains synaptic function; synapses consume substrate resources
 * HOW:  Apply substrate effects to transmission; track synaptic energy consumption
 */

#include "core/synapse_compute/nimcp_synapse_substrate_bridge.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/error/nimcp_error_codes.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>
#include <math.h>

/* ============================================================================
 * Helper Functions
 * ============================================================================ */

/**
 * @brief Clamp float to range
 *
 * WHAT: Constrain value to [min, max]
 * WHY:  Prevent out-of-range modulation factors
 * HOW:  Return clamped value
 */
static inline float clamp_f(float value, float min, float max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
}

/**
 * @brief Linear interpolation
 *
 * WHAT: Linearly interpolate between two values
 * WHY:  Smooth scaling for substrate effects
 * HOW:  lerp(a, b, t) = a + t * (b - a)
 */
static inline float lerp_f(float a, float b, float t) {
    return a + t * (b - a);
}

/**
 * @brief Compute ATP effect on release probability
 *
 * WHAT: Map ATP level to release probability factor
 * WHY:  Low ATP reduces vesicle pool availability
 * HOW:  Linear scaling between thresholds
 *
 * BIOLOGICAL BASIS:
 * - ATP < 0.3: Severe impairment (vesicle priming fails)
 * - ATP 0.3-0.8: Linear recovery
 * - ATP > 0.8: Full release capacity
 *
 * Reference: Harris et al. (2012) "Energetics of CNS white matter"
 */
static float compute_atp_release_effect(float atp_level, float sensitivity) {
    if (atp_level < ATP_RELEASE_THRESHOLD) {
        /* Severe impairment below threshold */
        float ratio = atp_level / ATP_RELEASE_THRESHOLD;
        return clamp_f(ratio * ATP_VESICLE_POOL_SENSITIVITY * sensitivity, 0.0f, 1.0f);
    } else if (atp_level < ATP_NORMAL_RELEASE) {
        /* Linear recovery from threshold to normal */
        float t = (atp_level - ATP_RELEASE_THRESHOLD) / (ATP_NORMAL_RELEASE - ATP_RELEASE_THRESHOLD);
        return lerp_f(ATP_VESICLE_POOL_SENSITIVITY, 1.0f, t * sensitivity);
    } else {
        /* Full capacity above normal */
        return 1.0f;
    }
}

/**
 * @brief Compute calcium effect on transmission
 *
 * WHAT: Map Ca2+ homeostasis to transmission probability
 * WHY:  Ca2+ gates vesicle fusion; imbalance impairs transmission
 * HOW:  Optimal at CA_NORMAL_LEVEL, reduced when too low/high
 *
 * BIOLOGICAL BASIS:
 * - Low Ca2+: Insufficient trigger for vesicle fusion
 * - High Ca2+: Excitotoxicity, impaired homeostasis
 * - Optimal ~0.95: Normal Ca2+ regulation
 *
 * Reference: Sudhof (2013) "Neurotransmitter release"
 */
static float compute_ca_transmission_effect(float ca_homeostasis, float sensitivity) {
    if (ca_homeostasis < CA_DEPLETION_THRESHOLD) {
        /* Low Ca2+ impairs transmission */
        float ratio = ca_homeostasis / CA_DEPLETION_THRESHOLD;
        return clamp_f(ratio * CA_TRANSMISSION_SENSITIVITY * sensitivity, 0.0f, 1.0f);
    } else if (ca_homeostasis > CA_OVERLOAD_THRESHOLD) {
        /* Ca2+ overload indicates stress */
        float excess = (ca_homeostasis - CA_OVERLOAD_THRESHOLD) / (1.0f - CA_OVERLOAD_THRESHOLD);
        return clamp_f((1.0f - excess * 0.3f) * sensitivity, 0.3f, 1.0f);
    } else {
        /* Normal range */
        return 1.0f;
    }
}

/**
 * @brief Compute Q10 temperature scaling factor
 *
 * WHAT: Calculate Q10-based kinetics scaling
 * WHY:  Temperature affects receptor/channel kinetics
 * HOW:  Q10 = (rate at T+10°C) / (rate at T°C)
 *
 * BIOLOGICAL BASIS:
 * - Most neural processes have Q10 = 2-3
 * - Hyperthermia: Faster kinetics (less precision)
 * - Hypothermia: Slower kinetics (impaired transmission)
 *
 * FORMULA: factor = Q10^((T - T_normal) / 10)
 *
 * Reference: Hodgkin & Huxley (1952) "Temperature coefficients"
 */
static float compute_q10_factor(float temperature, float q10, float sensitivity) {
    float temp_diff = temperature - SUBSTRATE_NORMAL_TEMPERATURE;
    float exponent = (temp_diff / 10.0f) * sensitivity;
    return powf(q10, exponent);
}

/**
 * @brief Compute membrane integrity effect on receptors
 *
 * WHAT: Map membrane integrity to receptor density
 * WHY:  Damaged membranes have reduced receptor insertion/anchoring
 * HOW:  Linear scaling with threshold
 *
 * BIOLOGICAL BASIS:
 * - Membrane integrity < 0.6: Significant receptor loss
 * - Lipid composition affects channel gating
 *
 * Reference: Bhargava et al. (2013) "Lipid-protein interactions"
 */
static float compute_membrane_receptor_effect(float membrane_integrity, float sensitivity) {
    if (membrane_integrity < MEMBRANE_RECEPTOR_THRESHOLD) {
        float ratio = membrane_integrity / MEMBRANE_RECEPTOR_THRESHOLD;
        return clamp_f(ratio * MEMBRANE_RECEPTOR_SENSITIVITY * sensitivity, 0.0f, 1.0f);
    }
    return 1.0f;
}

/**
 * @brief Compute ion gradient effect on driving force
 *
 * WHAT: Map ion balance to synaptic driving force
 * WHY:  Ion imbalance reduces electrochemical gradient
 * HOW:  Linear scaling with threshold
 *
 * BIOLOGICAL BASIS:
 * - Ion imbalance alters reversal potentials
 * - Reduced driving force → smaller synaptic currents
 *
 * Reference: Bhardwaj et al. (2016) "Ion homeostasis"
 */
static float compute_ion_driving_force_effect(float ion_balance, float sensitivity) {
    if (ion_balance < ION_DRIVING_FORCE_THRESHOLD) {
        float ratio = ion_balance / ION_DRIVING_FORCE_THRESHOLD;
        return clamp_f(ratio * ION_DRIVING_FORCE_SENSITIVITY * sensitivity, 0.0f, 1.0f);
    }
    return 1.0f;
}

/**
 * @brief Get ATP cost for synapse type
 *
 * WHAT: Return ATP cost per transmission for given synapse type
 * WHY:  Different synapse types have different energy costs
 * HOW:  Return predefined cost constant
 *
 * BIOLOGICAL BASIS:
 * - Ionotropic synapses (AMPA, GABA-A): Low cost (simple channels)
 * - Metabotropic synapses (GABA-B, neuromodulators): High cost (cascades)
 * - Gap junctions: Minimal cost (passive current)
 */
static float get_atp_cost_for_type(synapse_type_t type) {
    switch (type) {
        case SYNAPSE_AMPA:          return ATP_COST_AMPA;
        case SYNAPSE_NMDA:          return ATP_COST_NMDA;
        case SYNAPSE_GABA_A:        return ATP_COST_GABA_A;
        case SYNAPSE_GABA_B:        return ATP_COST_GABA_B;
        case SYNAPSE_DOPAMINE:      return ATP_COST_DOPAMINE;
        case SYNAPSE_SEROTONIN:     return ATP_COST_SEROTONIN;
        case SYNAPSE_ACETYLCHOLINE: return ATP_COST_ACETYLCHOLINE;
        case SYNAPSE_ELECTRICAL:    return ATP_COST_ELECTRICAL;
        default:                    return 0.0003f; /* Generic default */
    }
}

/**
 * @brief Get Q10 coefficient for synapse type
 *
 * WHAT: Return temperature coefficient for synapse type
 * WHY:  Different receptor kinetics have different Q10 values
 * HOW:  Return predefined Q10 constant
 *
 * BIOLOGICAL BASIS:
 * - Fast ionotropic (AMPA, GABA-A): High Q10 (2.0-2.5)
 * - Slow ionotropic (NMDA): Moderate Q10 (2.0)
 * - Metabotropic (GABA-B, neuromod): Lower Q10 (1.5-1.8)
 * - Gap junctions: Low Q10 (1.2, mostly passive)
 */
static float get_q10_for_type(synapse_type_t type) {
    switch (type) {
        case SYNAPSE_AMPA:          return Q10_AMPA;
        case SYNAPSE_NMDA:          return Q10_NMDA;
        case SYNAPSE_GABA_A:        return Q10_GABA_A;
        case SYNAPSE_GABA_B:        return Q10_GABA_B;
        case SYNAPSE_DOPAMINE:      return Q10_NEUROMOD;
        case SYNAPSE_SEROTONIN:     return Q10_NEUROMOD;
        case SYNAPSE_ACETYLCHOLINE: return Q10_NEUROMOD;
        case SYNAPSE_ELECTRICAL:    return Q10_ELECTRICAL;
        default:                    return 2.0f; /* Generic default */
    }
}

/* ============================================================================
 * Lifecycle Implementation
 * ============================================================================ */

int synapse_substrate_default_config(synapse_substrate_config_t* config) {
    /* Guard: require config */
    if (!config) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    /* All features enabled */
    config->enable_atp_modulation = true;
    config->enable_ca_modulation = true;
    config->enable_temperature_modulation = true;
    config->enable_membrane_modulation = true;
    config->enable_ion_modulation = true;
    config->enable_transmission_cost = true;
    config->enable_nmda_ca_tracking = true;

    /* Sensitivity multipliers */
    config->atp_sensitivity = 1.0f;
    config->ca_sensitivity = 1.0f;
    config->temperature_sensitivity = 1.0f;
    config->membrane_sensitivity = 1.0f;
    config->ion_sensitivity = 1.0f;

    /* ATP cost scaling */
    config->atp_cost_multiplier = 1.0f;

    return NIMCP_SUCCESS;
}

synapse_substrate_bridge_t* synapse_substrate_bridge_create(
    const synapse_substrate_config_t* config,
    synapse_compute_context_t* synapse_context,
    neural_substrate_t* substrate
) {
    /* Guard: require substrate */
    if (!substrate) {
        NIMCP_LOGGING_ERROR("Cannot create bridge without substrate");
        return NULL;
    }

    synapse_substrate_bridge_t* bridge = (synapse_substrate_bridge_t*)
        nimcp_calloc(1, sizeof(synapse_substrate_bridge_t));
    if (!bridge) {
        NIMCP_LOGGING_ERROR("Bridge allocation failed");
        return NULL;
    }

    /* Apply configuration */
    synapse_substrate_config_t default_cfg;
    if (!config) {
        synapse_substrate_default_config(&default_cfg);
        config = &default_cfg;
    }
    bridge->config = *config;

    /* Link systems */
    bridge->synapse_context = synapse_context;
    bridge->substrate = substrate;

    /* Initialize effects to neutral */
    bridge->substrate_effects.transmission_efficiency = 1.0f;
    bridge->substrate_effects.release_probability = 1.0f;
    bridge->substrate_effects.receptor_kinetics_factor = 1.0f;
    bridge->substrate_effects.driving_force_factor = 1.0f;
    bridge->substrate_effects.ampa_modulation = 1.0f;
    bridge->substrate_effects.nmda_modulation = 1.0f;
    bridge->substrate_effects.gaba_a_modulation = 1.0f;
    bridge->substrate_effects.gaba_b_modulation = 1.0f;
    bridge->substrate_effects.dopamine_modulation = 1.0f;
    bridge->substrate_effects.serotonin_modulation = 1.0f;
    bridge->substrate_effects.acetylcholine_modulation = 1.0f;
    bridge->substrate_effects.electrical_modulation = 1.0f;

    /* Initialize stats */
    bridge->stats.min_transmission_efficiency = 1.0f;

    /* Create mutex */
    if (bridge_base_init(&bridge->base, 0, "synapse_substrate") != 0) { nimcp_free(bridge); return NULL; }
    if (!bridge->base.mutex) {
        NIMCP_LOGGING_ERROR("Mutex allocation failed");
        synapse_substrate_bridge_destroy(bridge);
        return NULL;
    }

    NIMCP_LOGGING_INFO("Synapse substrate bridge created");
    return bridge;
}

void synapse_substrate_bridge_destroy(synapse_substrate_bridge_t* bridge) {
    /* Guard: NULL safe */
    if (!bridge) return;

    if (bridge->base.mutex) {
        bridge_base_cleanup(&bridge->base);
    }

    nimcp_free(bridge);
    NIMCP_LOGGING_INFO("Synapse substrate bridge destroyed");
}

/* ============================================================================
 * Substrate → Synapse Implementation
 * ============================================================================ */

int synapse_substrate_update(synapse_substrate_bridge_t* bridge) {
    /* Guard: require bridge */
    if (!bridge) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    /* Guard: require substrate */
    if (!bridge->substrate) {
        NIMCP_LOGGING_WARN("No substrate connected, skipping update");
        return NIMCP_SUCCESS;
    }

    nimcp_platform_mutex_lock(bridge->base.mutex);

    /* Get substrate state */
    substrate_metabolic_state_t metabolic;
    substrate_physical_state_t physical;
    substrate_get_metabolic_state(bridge->substrate, &metabolic);
    substrate_get_physical_state(bridge->substrate, &physical);

    /* Component 1: ATP effect on release probability */
    if (bridge->config.enable_atp_modulation) {
        bridge->substrate_effects.atp_release_effect = compute_atp_release_effect(
            metabolic.atp_level,
            bridge->config.atp_sensitivity
        );

        /* Track low ATP events */
        if (metabolic.atp_level < ATP_RELEASE_THRESHOLD) {
            bridge->stats.low_atp_impairments++;
        }
    } else {
        bridge->substrate_effects.atp_release_effect = 1.0f;
    }

    /* Component 2: Calcium effect on transmission */
    if (bridge->config.enable_ca_modulation) {
        bridge->substrate_effects.ca_transmission_effect = compute_ca_transmission_effect(
            physical.ca_homeostasis,
            bridge->config.ca_sensitivity
        );

        /* Track Ca2+ overload */
        if (physical.ca_homeostasis > CA_OVERLOAD_THRESHOLD) {
            bridge->stats.ca_overload_events++;
        }
    } else {
        bridge->substrate_effects.ca_transmission_effect = 1.0f;
    }

    /* Component 3: Temperature effect on receptor kinetics (Q10) */
    if (bridge->config.enable_temperature_modulation) {
        /* Use average Q10 for generic factor */
        float avg_q10 = 2.0f;
        bridge->substrate_effects.temperature_kinetics_q10 = compute_q10_factor(
            physical.temperature,
            avg_q10,
            bridge->config.temperature_sensitivity
        );

        /* Track kinetics factor */
        if (bridge->substrate_effects.temperature_kinetics_q10 > bridge->stats.max_temperature_kinetics_factor) {
            bridge->stats.max_temperature_kinetics_factor = bridge->substrate_effects.temperature_kinetics_q10;
        }
    } else {
        bridge->substrate_effects.temperature_kinetics_q10 = 1.0f;
    }

    /* Component 4: Membrane integrity effect on receptors */
    if (bridge->config.enable_membrane_modulation) {
        bridge->substrate_effects.membrane_receptor_effect = compute_membrane_receptor_effect(
            physical.membrane_integrity,
            bridge->config.membrane_sensitivity
        );
    } else {
        bridge->substrate_effects.membrane_receptor_effect = 1.0f;
    }

    /* Component 5: Ion balance effect on driving force */
    if (bridge->config.enable_ion_modulation) {
        bridge->substrate_effects.ion_driving_force_effect = compute_ion_driving_force_effect(
            physical.ion_balance,
            bridge->config.ion_sensitivity
        );
    } else {
        bridge->substrate_effects.ion_driving_force_effect = 1.0f;
    }

    /* Compute overall factors */
    bridge->substrate_effects.release_probability =
        bridge->substrate_effects.atp_release_effect *
        bridge->substrate_effects.ca_transmission_effect;

    bridge->substrate_effects.receptor_kinetics_factor =
        bridge->substrate_effects.temperature_kinetics_q10;

    bridge->substrate_effects.driving_force_factor =
        bridge->substrate_effects.ion_driving_force_effect;

    bridge->substrate_effects.transmission_efficiency =
        bridge->substrate_effects.release_probability *
        bridge->substrate_effects.membrane_receptor_effect *
        bridge->substrate_effects.driving_force_factor;

    /* Compute per-synapse-type modulation (Strategy Pattern) */
    /* AMPA: Fast ionotropic, temperature-sensitive */
    float ampa_q10 = compute_q10_factor(physical.temperature, Q10_AMPA, bridge->config.temperature_sensitivity);
    bridge->substrate_effects.ampa_modulation =
        bridge->substrate_effects.transmission_efficiency *
        (bridge->config.enable_temperature_modulation ? ampa_q10 : 1.0f);

    /* NMDA: Slow ionotropic, Ca2+-permeable, voltage-gated */
    float nmda_q10 = compute_q10_factor(physical.temperature, Q10_NMDA, bridge->config.temperature_sensitivity);
    bridge->substrate_effects.nmda_modulation =
        bridge->substrate_effects.transmission_efficiency *
        (bridge->config.enable_temperature_modulation ? nmda_q10 : 1.0f);

    /* GABA-A: Fast inhibitory, temperature-sensitive */
    float gaba_a_q10 = compute_q10_factor(physical.temperature, Q10_GABA_A, bridge->config.temperature_sensitivity);
    bridge->substrate_effects.gaba_a_modulation =
        bridge->substrate_effects.transmission_efficiency *
        (bridge->config.enable_temperature_modulation ? gaba_a_q10 : 1.0f);

    /* GABA-B: Slow metabotropic, ATP-dependent */
    float gaba_b_q10 = compute_q10_factor(physical.temperature, Q10_GABA_B, bridge->config.temperature_sensitivity);
    float gaba_b_atp_penalty = (metabolic.atp_level < 0.5f) ? 0.7f : 1.0f; /* Metabotropic more ATP-sensitive */
    bridge->substrate_effects.gaba_b_modulation =
        bridge->substrate_effects.transmission_efficiency *
        (bridge->config.enable_temperature_modulation ? gaba_b_q10 : 1.0f) *
        (bridge->config.enable_atp_modulation ? gaba_b_atp_penalty : 1.0f);

    /* Dopamine: Metabotropic, very ATP-dependent */
    float neuromod_q10 = compute_q10_factor(physical.temperature, Q10_NEUROMOD, bridge->config.temperature_sensitivity);
    float dopamine_atp_penalty = (metabolic.atp_level < 0.6f) ? 0.5f : 1.0f; /* Very ATP-sensitive */
    bridge->substrate_effects.dopamine_modulation =
        bridge->substrate_effects.transmission_efficiency *
        (bridge->config.enable_temperature_modulation ? neuromod_q10 : 1.0f) *
        (bridge->config.enable_atp_modulation ? dopamine_atp_penalty : 1.0f);

    /* Serotonin: Metabotropic, very ATP-dependent */
    float serotonin_atp_penalty = (metabolic.atp_level < 0.6f) ? 0.5f : 1.0f;
    bridge->substrate_effects.serotonin_modulation =
        bridge->substrate_effects.transmission_efficiency *
        (bridge->config.enable_temperature_modulation ? neuromod_q10 : 1.0f) *
        (bridge->config.enable_atp_modulation ? serotonin_atp_penalty : 1.0f);

    /* Acetylcholine: Dual receptor, moderate ATP-dependence */
    float ach_atp_penalty = (metabolic.atp_level < 0.5f) ? 0.6f : 1.0f;
    bridge->substrate_effects.acetylcholine_modulation =
        bridge->substrate_effects.transmission_efficiency *
        (bridge->config.enable_temperature_modulation ? neuromod_q10 : 1.0f) *
        (bridge->config.enable_atp_modulation ? ach_atp_penalty : 1.0f);

    /* Electrical: Gap junction, minimal ATP, low temperature sensitivity */
    float electrical_q10 = compute_q10_factor(physical.temperature, Q10_ELECTRICAL, bridge->config.temperature_sensitivity);
    bridge->substrate_effects.electrical_modulation =
        bridge->substrate_effects.membrane_receptor_effect * /* Membrane important for gap junctions */
        (bridge->config.enable_temperature_modulation ? electrical_q10 : 1.0f);

    /* Track stats */
    bridge->stats.total_updates++;
    if (bridge->substrate_effects.transmission_efficiency < bridge->stats.min_transmission_efficiency) {
        bridge->stats.min_transmission_efficiency = bridge->substrate_effects.transmission_efficiency;
    }

    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return NIMCP_SUCCESS;
}

float synapse_substrate_apply_modulation(
    const synapse_substrate_bridge_t* bridge,
    synapse_type_t synapse_type
) {
    /* Guard: require bridge */
    if (!bridge) {
        return 1.0f; /* No modulation on error */
    }

    /* Return precomputed type-specific modulation */
    switch (synapse_type) {
        case SYNAPSE_AMPA:          return bridge->substrate_effects.ampa_modulation;
        case SYNAPSE_NMDA:          return bridge->substrate_effects.nmda_modulation;
        case SYNAPSE_GABA_A:        return bridge->substrate_effects.gaba_a_modulation;
        case SYNAPSE_GABA_B:        return bridge->substrate_effects.gaba_b_modulation;
        case SYNAPSE_DOPAMINE:      return bridge->substrate_effects.dopamine_modulation;
        case SYNAPSE_SEROTONIN:     return bridge->substrate_effects.serotonin_modulation;
        case SYNAPSE_ACETYLCHOLINE: return bridge->substrate_effects.acetylcholine_modulation;
        case SYNAPSE_ELECTRICAL:    return bridge->substrate_effects.electrical_modulation;
        default:                    return bridge->substrate_effects.transmission_efficiency; /* Generic */
    }
}

/* ============================================================================
 * Synapse → Substrate Implementation
 * ============================================================================ */

int synapse_substrate_consume_transmission(
    synapse_substrate_bridge_t* bridge,
    synapse_type_t synapse_type,
    uint32_t count
) {
    /* Guard: require bridge */
    if (!bridge) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    /* Guard: check if transmission cost tracking enabled */
    if (!bridge->config.enable_transmission_cost) {
        return NIMCP_SUCCESS; /* Feature disabled */
    }

    /* Guard: require substrate */
    if (!bridge->substrate) {
        return NIMCP_SUCCESS; /* No substrate to consume from */
    }

    /* Guard: sanity check count */
    if (count == 0) {
        return NIMCP_SUCCESS;
    }

    /* Get ATP cost for synapse type */
    float base_cost = get_atp_cost_for_type(synapse_type);
    float total_cost = base_cost * count * bridge->config.atp_cost_multiplier;

    /* Record transmission in substrate */
    substrate_record_transmissions(bridge->substrate, count);

    nimcp_platform_mutex_lock(bridge->base.mutex);
    bridge->stats.transmissions_processed += count;
    bridge->stats.total_atp_consumed_by_synapses += total_cost;
    nimcp_platform_mutex_unlock(bridge->base.mutex);

    return NIMCP_SUCCESS;
}

int synapse_substrate_record_nmda_calcium(
    synapse_substrate_bridge_t* bridge,
    uint32_t transmission_count
) {
    /* Guard: require bridge */
    if (!bridge) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    /* Guard: check if NMDA Ca2+ tracking enabled */
    if (!bridge->config.enable_nmda_ca_tracking) {
        return NIMCP_SUCCESS; /* Feature disabled */
    }

    /* Guard: require substrate */
    if (!bridge->substrate) {
        return NIMCP_SUCCESS; /* No substrate to track in */
    }

    /* Guard: sanity check count */
    if (transmission_count == 0) {
        return NIMCP_SUCCESS;
    }

    /* Calculate Ca2+ influx (simplified model) */
    /* In real implementation, would call substrate Ca2+ tracking API */
    /* For now, just track stats */
    nimcp_platform_mutex_lock(bridge->base.mutex);
    bridge->stats.nmda_ca_influx_events += transmission_count;
    nimcp_platform_mutex_unlock(bridge->base.mutex);

    /* Note: Actual Ca2+ modulation would be:
     * substrate_add_calcium(bridge->substrate, transmission_count * NMDA_CA_INFLUX_PER_EVENT);
     * This requires substrate API extension for Ca2+ accumulation tracking
     */

    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Query Implementation
 * ============================================================================ */

int synapse_substrate_get_effects(
    const synapse_substrate_bridge_t* bridge,
    substrate_synapse_effects_t* effects
) {
    /* Guard: require both parameters */
    if (!bridge || !effects) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    nimcp_platform_mutex_lock(bridge->base.mutex);
    *effects = bridge->substrate_effects;
    nimcp_platform_mutex_unlock(bridge->base.mutex);

    return NIMCP_SUCCESS;
}

float synapse_substrate_get_transmission_efficiency(
    const synapse_substrate_bridge_t* bridge
) {
    /* Guard: require bridge */
    if (!bridge) {
        return 1.0f; /* Neutral on error */
    }

    return bridge->substrate_effects.transmission_efficiency;
}

float synapse_substrate_get_release_probability(
    const synapse_substrate_bridge_t* bridge
) {
    /* Guard: require bridge */
    if (!bridge) {
        return 1.0f; /* Neutral on error */
    }

    return bridge->substrate_effects.release_probability;
}

float synapse_substrate_get_receptor_kinetics_factor(
    const synapse_substrate_bridge_t* bridge
) {
    /* Guard: require bridge */
    if (!bridge) {
        return 1.0f; /* Neutral on error */
    }

    return bridge->substrate_effects.receptor_kinetics_factor;
}

int synapse_substrate_get_stats(
    const synapse_substrate_bridge_t* bridge,
    synapse_substrate_stats_t* stats
) {
    /* Guard: require both parameters */
    if (!bridge || !stats) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    nimcp_platform_mutex_lock(bridge->base.mutex);
    *stats = bridge->stats;
    nimcp_platform_mutex_unlock(bridge->base.mutex);

    return NIMCP_SUCCESS;
}
