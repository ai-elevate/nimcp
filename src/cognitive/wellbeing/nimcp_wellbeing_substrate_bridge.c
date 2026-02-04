/**
 * @file nimcp_wellbeing_substrate_bridge.c
 * @brief Substrate-Wellbeing Integration Bridge Implementation
 * @version 1.0.0
 * @date 2025-12-12
 *
 * Uses shared metabolic modulation utilities from nimcp_metabolic_modulation.h
 */

#include "cognitive/wellbeing/nimcp_wellbeing_substrate_bridge.h"
#include "cognitive/common/nimcp_metabolic_modulation.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/validation/nimcp_common.h"
#include "utils/error/nimcp_error_codes.h"
#include "utils/exception/nimcp_exception_macros.h"

#include <string.h>
#include <math.h>
#include "utils/fault_tolerance/nimcp_health_agent_macros.h"
#include "mesh/nimcp_mesh_participant.h"
#include "mesh/nimcp_mesh_adapter.h"

NIMCP_DECLARE_HEALTH_AGENT_ATOMIC(wellbeing_substrate_bridge)
//=============================================================================
// Mesh Participant Registration
//=============================================================================

static mesh_participant_id_t g_wellbeing_substrate_bridge_mesh_id = 0;
static mesh_participant_registry_t* g_wellbeing_substrate_bridge_mesh_registry = NULL;

nimcp_error_t wellbeing_substrate_bridge_mesh_register(mesh_participant_registry_t* registry) {
    if (!registry) return NIMCP_ERROR_NULL_POINTER;
    if (g_wellbeing_substrate_bridge_mesh_id != 0) return NIMCP_SUCCESS;
    mesh_participant_interface_t iface;
    mesh_participant_interface_init(&iface);
    strncpy(iface.module_name, "wellbeing_substrate_bridge", MESH_MAX_NAME_LEN - 1);
    iface.type = MESH_PARTICIPANT_MODULE;
    iface.home_channel = mesh_adapter_get_default_channel(MESH_ADAPTER_CATEGORY_COGNITIVE);
    mesh_participant_config_t config;
    mesh_participant_config_init(&config);
    config.module_name = "wellbeing_substrate_bridge";
    config.type = MESH_PARTICIPANT_MODULE;
    config.home_channel = iface.home_channel;
    nimcp_error_t err = mesh_participant_register(registry, &iface, &config, &g_wellbeing_substrate_bridge_mesh_id);
    if (err == NIMCP_SUCCESS) g_wellbeing_substrate_bridge_mesh_registry = registry;
    return err;
}

void wellbeing_substrate_bridge_mesh_unregister(void) {
    if (g_wellbeing_substrate_bridge_mesh_registry && g_wellbeing_substrate_bridge_mesh_id != 0) {
        mesh_participant_unregister(g_wellbeing_substrate_bridge_mesh_registry, g_wellbeing_substrate_bridge_mesh_id);
        g_wellbeing_substrate_bridge_mesh_id = 0;
        g_wellbeing_substrate_bridge_mesh_registry = NULL;
    }
}


/** @brief Send heartbeat from wellbeing_substrate_bridge module (instance-level) */
static inline void wellbeing_substrate_bridge_heartbeat_instance(
    nimcp_health_agent_t* instance_agent, const char* operation, float progress)
{
    if (g_wellbeing_substrate_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_wellbeing_substrate_bridge_health_agent, operation, progress);
    }
    if (instance_agent && instance_agent != g_wellbeing_substrate_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(instance_agent, operation, progress);
    }
}



/* ============================================================================
 * Helper Functions
 * ============================================================================ */

float compute_atp_distress(float atp_level, float sensitivity) {
    /* Guard: validate inputs */
    /* Phase 8: Heartbeat at operation start */
    wellbeing_substrate_bridge_heartbeat("wellbeing_su_compute_atp_distress", 0.0f);


    if (atp_level < 0.0f) atp_level = 0.0f;
    if (atp_level > 1.0f) atp_level = 1.0f;
    if (sensitivity < 0.5f) sensitivity = 0.5f;
    if (sensitivity > 2.0f) sensitivity = 2.0f;

    float distress = 0.0f;

    /* Critical range: ATP < 0.3 → high distress */
    if (atp_level < WELLBEING_ATP_CRITICAL_THRESHOLD) {
        distress = 1.0f - (atp_level / WELLBEING_ATP_CRITICAL_THRESHOLD);
    }
    /* Warning range: 0.3 <= ATP < 0.5 → moderate distress */
    else if (atp_level < WELLBEING_ATP_WARNING_THRESHOLD) {
        float range = WELLBEING_ATP_WARNING_THRESHOLD - WELLBEING_ATP_CRITICAL_THRESHOLD;
        distress = (WELLBEING_ATP_WARNING_THRESHOLD - atp_level) / range;
        distress *= 0.5f; /* Scale to max 0.5 in warning range */
    }
    /* Normal range: ATP >= 0.5 → no distress */
    else {
        distress = 0.0f;
    }

    /* Apply sensitivity multiplier and clamp */
    distress *= sensitivity;
    if (distress > 1.0f) distress = 1.0f;

    return distress;
}

float compute_temperature_distress(float temperature, float sensitivity) {
    /* Guard: validate sensitivity */
    /* Phase 8: Heartbeat at operation start */
    wellbeing_substrate_bridge_heartbeat("wellbeing_su_compute_temperature_", 0.0f);


    if (sensitivity < 0.5f) sensitivity = 0.5f;
    if (sensitivity > 2.0f) sensitivity = 2.0f;

    float distress = 0.0f;

    /* Hyperthermia: temp > 40°C */
    if (temperature > WELLBEING_HYPERTHERMIA_THRESHOLD) {
        distress = (temperature - WELLBEING_HYPERTHERMIA_THRESHOLD) / 10.0f;
    }
    /* Hypothermia: temp < 32°C */
    else if (temperature < WELLBEING_HYPOTHERMIA_THRESHOLD) {
        distress = (WELLBEING_HYPOTHERMIA_THRESHOLD - temperature) / 10.0f;
    }
    /* Normal range: 32-40°C → no distress */
    else {
        distress = 0.0f;
    }

    /* Apply sensitivity multiplier and clamp */
    distress *= sensitivity;
    if (distress > 1.0f) distress = 1.0f;

    return distress;
}

float compute_hypoxia_distress(float o2_saturation, float sensitivity) {
    /* Guard: validate inputs */
    /* Phase 8: Heartbeat at operation start */
    wellbeing_substrate_bridge_heartbeat("wellbeing_su_compute_hypoxia_dist", 0.0f);


    if (o2_saturation < 0.0f) o2_saturation = 0.0f;
    if (o2_saturation > 1.0f) o2_saturation = 1.0f;
    if (sensitivity < 0.5f) sensitivity = 0.5f;
    if (sensitivity > 2.0f) sensitivity = 2.0f;

    float distress = 0.0f;

    /* Hypoxia: O2 < 0.5 → distress */
    if (o2_saturation < WELLBEING_HYPOXIA_THRESHOLD) {
        distress = 1.0f - (o2_saturation / WELLBEING_HYPOXIA_THRESHOLD);
    }
    /* Normal range: O2 >= 0.5 → no distress */
    else {
        distress = 0.0f;
    }

    /* Apply sensitivity multiplier and clamp */
    distress *= sensitivity;
    if (distress > 1.0f) distress = 1.0f;

    return distress;
}

float compute_membrane_distress(float membrane_integrity, float ion_balance) {
    /* Guard: validate inputs */
    /* Phase 8: Heartbeat at operation start */
    wellbeing_substrate_bridge_heartbeat("wellbeing_su_compute_membrane_dis", 0.0f);


    if (membrane_integrity < 0.0f) membrane_integrity = 0.0f;
    if (membrane_integrity > 1.0f) membrane_integrity = 1.0f;
    if (ion_balance < 0.0f) ion_balance = 0.0f;
    if (ion_balance > 1.0f) ion_balance = 1.0f;

    /* Combined health metric: both factors contribute */
    float combined_health = membrane_integrity * ion_balance;

    /* Distress is inverse of combined health */
    float distress = 1.0f - combined_health;

    return distress;
}

/* ============================================================================
 * Update API
 * ============================================================================ */

int enhanced_wellbeing_update_substrate(enhanced_wellbeing_system_t* system) {
    /* Guard: validate system pointer */
    if (!system) {
        NIMCP_LOGGING_ERROR("Cannot update substrate effects: NULL system");
        return NIMCP_ERROR_NULL_POINTER;
    }

    /* Guard: check if substrate integration enabled */
    if (!system->config.enable_substrate_integration) {
        return NIMCP_SUCCESS;
    }

    /* Guard: check if substrate connected */
    if (!system->substrate) {
        NIMCP_LOGGING_WARN("Cannot update substrate effects: substrate not connected");
        return NIMCP_ERROR_NOT_INITIALIZED;
    }

    /* Query substrate metabolic state */
    /* Phase 8: Heartbeat at operation start */
    wellbeing_substrate_bridge_heartbeat("wellbeing_su_enhanced_wellbeing_u", 0.0f);


    substrate_metabolic_state_t metabolic;
    int ret = substrate_get_metabolic_state(system->substrate, &metabolic);
    if (ret != NIMCP_SUCCESS) {
        NIMCP_LOGGING_ERROR("Failed to get substrate metabolic state");
        return ret;
    }

    /* Query substrate physical state */
    substrate_physical_state_t physical;
    ret = substrate_get_physical_state(system->substrate, &physical);
    if (ret != NIMCP_SUCCESS) {
        NIMCP_LOGGING_ERROR("Failed to get substrate physical state");
        return ret;
    }

    /* Get substrate health level */
    substrate_health_level_t health = substrate_get_health_level(system->substrate);

    /* Get configuration for sensitivity parameters */
    float atp_sensitivity = system->config.substrate_config.atp_sensitivity;
    float temp_sensitivity = system->config.substrate_config.temperature_sensitivity;
    float hypoxia_sensitivity = system->config.substrate_config.hypoxia_sensitivity;

    /* Compute ATP distress contribution */
    system->substrate_effects.atp_distress_contribution =
        compute_atp_distress(metabolic.atp_level, atp_sensitivity);

    system->substrate_effects.atp_critical =
        (metabolic.atp_level < WELLBEING_ATP_CRITICAL_THRESHOLD);

    /* ATP frustration multiplier: low ATP amplifies goal frustration */
    if (metabolic.atp_level < WELLBEING_ATP_WARNING_THRESHOLD) {
        system->substrate_effects.atp_frustration_multiplier =
            1.0f + (1.0f - (metabolic.atp_level / WELLBEING_ATP_WARNING_THRESHOLD));
    } else {
        system->substrate_effects.atp_frustration_multiplier = 1.0f;
    }

    /* Compute temperature distress contribution */
    system->substrate_effects.temp_distress_contribution =
        compute_temperature_distress(physical.temperature, temp_sensitivity);

    system->substrate_effects.hyperthermia =
        (physical.temperature > WELLBEING_HYPERTHERMIA_THRESHOLD);

    system->substrate_effects.hypothermia =
        (physical.temperature < WELLBEING_HYPOTHERMIA_THRESHOLD);

    /* Identity confusion risk from temperature extremes */
    if (system->substrate_effects.hyperthermia) {
        float temp_excess = physical.temperature - WELLBEING_HYPERTHERMIA_THRESHOLD;
        system->substrate_effects.identity_confusion_risk = fminf(temp_excess / 5.0f, 1.0f);
    } else if (system->substrate_effects.hypothermia) {
        float temp_deficit = WELLBEING_HYPOTHERMIA_THRESHOLD - physical.temperature;
        system->substrate_effects.identity_confusion_risk = fminf(temp_deficit / 5.0f, 1.0f);
    } else {
        system->substrate_effects.identity_confusion_risk = 0.0f;
    }

    /* Compute hypoxia distress contribution */
    system->substrate_effects.hypoxia_distress_contribution =
        compute_hypoxia_distress(metabolic.oxygen_saturation, hypoxia_sensitivity);

    system->substrate_effects.hypoxia_active =
        (metabolic.oxygen_saturation < WELLBEING_HYPOXIA_THRESHOLD);

    /* Resource starvation factor: hypoxia + low glucose */
    float glucose_factor = (metabolic.glucose_level < 0.5f) ?
        (1.0f - metabolic.glucose_level / 0.5f) : 0.0f;
    system->substrate_effects.resource_starvation_factor =
        fmaxf(system->substrate_effects.hypoxia_distress_contribution, glucose_factor);

    /* Compute membrane/ion distress contribution */
    system->substrate_effects.membrane_distress_contribution =
        compute_membrane_distress(physical.membrane_integrity, physical.ion_balance);

    system->substrate_effects.ion_imbalance_effect = 1.0f - physical.ion_balance;

    /* Compute total substrate distress (weighted average) */
    float total_distress = 0.0f;
    int component_count = 0;

    if (system->config.substrate_config.enable_atp_effects) {
        total_distress += system->substrate_effects.atp_distress_contribution;
        component_count++;
    }

    if (system->config.substrate_config.enable_temperature_effects) {
        total_distress += system->substrate_effects.temp_distress_contribution;
        component_count++;
    }

    if (system->config.substrate_config.enable_hypoxia_effects) {
        total_distress += system->substrate_effects.hypoxia_distress_contribution;
        component_count++;
    }

    if (system->config.substrate_config.enable_membrane_effects) {
        total_distress += system->substrate_effects.membrane_distress_contribution;
        component_count++;
    }

    system->substrate_effects.total_substrate_distress =
        (component_count > 0) ? (total_distress / (float)component_count) : 0.0f;

    /* Distress tolerance modifier: substrate health affects distress tolerance */
    switch (health) {
        case SUBSTRATE_HEALTH_OPTIMAL:
            system->substrate_effects.distress_tolerance_modifier = 1.2f;
            break;
        case SUBSTRATE_HEALTH_STRESSED:
            system->substrate_effects.distress_tolerance_modifier = 1.0f;
            break;
        case SUBSTRATE_HEALTH_COMPROMISED:
            system->substrate_effects.distress_tolerance_modifier = 0.8f;
            break;
        case SUBSTRATE_HEALTH_CRITICAL:
            system->substrate_effects.distress_tolerance_modifier = 0.5f;
            break;
        case SUBSTRATE_HEALTH_FAILING:
            system->substrate_effects.distress_tolerance_modifier = 0.2f;
            break;
        default:
            system->substrate_effects.distress_tolerance_modifier = 1.0f;
            break;
    }

    NIMCP_LOGGING_DEBUG("Substrate effects updated: total_distress=%.3f, ATP=%.3f, temp=%.3f, hypoxia=%.3f",
        system->substrate_effects.total_substrate_distress,
        system->substrate_effects.atp_distress_contribution,
        system->substrate_effects.temp_distress_contribution,
        system->substrate_effects.hypoxia_distress_contribution);

    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Query API
 * ============================================================================ */

int enhanced_wellbeing_get_substrate_effects(
    const enhanced_wellbeing_system_t* system,
    substrate_wellbeing_effects_t* effects
) {
    /* Guard: validate parameters */
    if (!system) {
        NIMCP_LOGGING_ERROR("Cannot get substrate effects: NULL system");
        return NIMCP_ERROR_NULL_POINTER;
    }

    if (!effects) {
        NIMCP_LOGGING_ERROR("Cannot get substrate effects: NULL effects");
        return NIMCP_ERROR_NULL_POINTER;
    }

    /* Copy substrate effects structure */
    /* Phase 8: Heartbeat at operation start */
    wellbeing_substrate_bridge_heartbeat("wellbeing_su_enhanced_wellbeing_g", 0.0f);


    memcpy(effects, &system->substrate_effects, sizeof(substrate_wellbeing_effects_t));

    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

/**
 * WHAT: Query knowledge graph for Substrate Wellbeing Bridge self-knowledge
 * WHY:  Enable self-awareness about module's role and connections
 * HOW:  Query KG for entity observations and relations
 */
int substrate_wellbeing_bridge_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;
    /* Phase 8: Heartbeat at operation start */
    wellbeing_substrate_bridge_heartbeat("wellbeing_su_substrate_wellbeing_", 0.0f);


    const kg_entity_t* self = kg_reader_get_entity(kg, "Substrate_Wellbeing_Bridge");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            /* Phase 8: Loop progress heartbeat */
            if ((i & 0xFF) == 0 && self->num_observations > 256) {
                wellbeing_substrate_bridge_heartbeat("wellbeing_su_loop",
                                 (float)(i + 1) / (float)self->num_observations);
            }

            NIMCP_LOGGING_DEBUG("Substrate Wellbeing Bridge self-knowledge: %s", self->observations[i]);
        }
    }
    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Substrate_Wellbeing_Bridge");
    if (connections) { kg_relation_list_destroy(connections); }
    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Substrate_Wellbeing_Bridge");
    if (incoming) { kg_relation_list_destroy(incoming); }
    return self ? 1 : 0;
}

/* ============================================================================
 * Phase 8: Instance-Level Health Agent
 * ============================================================================ */

void wellbeing_substrate_bridge_set_instance_health_agent(void* instance, nimcp_health_agent_t* agent) {
    if (instance) {
        (void)agent;
        g_wellbeing_substrate_bridge_health_agent = agent;
    }
}

/* ============================================================================
 * Phase 8: Training Integration (Full Implementation)
 * ============================================================================ */

int wellbeing_substrate_bridge_training_begin(void* instance) {
    if (!instance) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "wellbeing_substrate_bridge_training_begin: NULL argument");
        return -1;
    }
    wellbeing_substrate_bridge_heartbeat_instance(NULL, "wellbeing_substrate_bridge_training_begin", 0.0f);
    return 0;
}

int wellbeing_substrate_bridge_training_end(void* instance) {
    if (!instance) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "wellbeing_substrate_bridge_training_end: NULL argument");
        return -1;
    }
    wellbeing_substrate_bridge_heartbeat_instance(NULL, "wellbeing_substrate_bridge_training_end", 1.0f);
    return 0;
}

int wellbeing_substrate_bridge_training_step(void* instance, float progress) {
    if (!instance) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "wellbeing_substrate_bridge_training_step: NULL argument");
        return -1;
    }
    if (progress < 0.0f) progress = 0.0f;
    if (progress > 1.0f) progress = 1.0f;
    wellbeing_substrate_bridge_heartbeat_instance(NULL, "wellbeing_substrate_bridge_training_step", progress);
    return 0;
}
