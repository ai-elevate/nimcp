/**
 * @file nimcp_broca_substrate_bridge.c
 * @brief Broca-Substrate Bridge Implementation
 *
 * Links language production to metabolic state for biologically
 * realistic speech production under fatigue/stress conditions.
 */

#include "utils/bridge/nimcp_bridge_base.h"
#include "core/brain/regions/broca/nimcp_broca_substrate_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>

//=============================================================================
#include <stddef.h>  /* for NULL */
#include "utils/logging/nimcp_logging.h"
#include "utils/fault_tolerance/nimcp_health_agent_macros.h"
#include "mesh/nimcp_mesh_participant.h"
#include "mesh/nimcp_mesh_adapter.h"

NIMCP_DECLARE_HEALTH_AGENT_ATOMIC(broca_substrate_bridge)
//=============================================================================
// Mesh Participant Registration
//=============================================================================

static mesh_participant_id_t g_broca_substrate_bridge_mesh_id = 0;
static mesh_participant_registry_t* g_broca_substrate_bridge_mesh_registry = NULL;

nimcp_error_t broca_substrate_bridge_mesh_register(mesh_participant_registry_t* registry) {
    if (!registry) return NIMCP_ERROR_NULL_POINTER;
    if (g_broca_substrate_bridge_mesh_id != 0) return NIMCP_SUCCESS;
    mesh_participant_interface_t iface;
    mesh_participant_interface_init(&iface);
    strncpy(iface.module_name, "broca_substrate_bridge", MESH_MAX_NAME_LEN - 1);
    iface.type = MESH_PARTICIPANT_MODULE;
    iface.home_channel = mesh_adapter_get_default_channel(MESH_ADAPTER_CATEGORY_SYSTEM);
    mesh_participant_config_t config;
    mesh_participant_config_init(&config);
    config.module_name = "broca_substrate_bridge";
    config.type = MESH_PARTICIPANT_MODULE;
    config.home_channel = iface.home_channel;
    nimcp_error_t err = mesh_participant_register(registry, &iface, &config, &g_broca_substrate_bridge_mesh_id);
    if (err == NIMCP_SUCCESS) g_broca_substrate_bridge_mesh_registry = registry;
    return err;
}

void broca_substrate_bridge_mesh_unregister(void) {
    if (g_broca_substrate_bridge_mesh_registry && g_broca_substrate_bridge_mesh_id != 0) {
        mesh_participant_unregister(g_broca_substrate_bridge_mesh_registry, g_broca_substrate_bridge_mesh_id);
        g_broca_substrate_bridge_mesh_id = 0;
        g_broca_substrate_bridge_mesh_registry = NULL;
    }
}


#define LOG_MODULE "BROCA_SUBSTRATE_BRIDGE"


//=============================================================================
// Internal Constants
//=============================================================================

#define LOW_ATP_THRESHOLD 0.3f
#define HIGH_FATIGUE_THRESHOLD 0.7f

//=============================================================================
// Internal Structure
//=============================================================================

struct broca_substrate_bridge {
    bridge_base_t base;              /**< MUST be first: base bridge infrastructure */
    void* broca;                         /**< Broca adapter handle */
    neural_substrate_t* substrate;       /**< Neural substrate handle */
    broca_substrate_config_t config;     /**< Configuration */
    broca_substrate_effects_t effects;   /**< Current effects */
    broca_substrate_stats_t stats;       /**< Statistics */
    bio_router_t* router;                /**< Bio-async router */
    bool registered;                     /**< Registered with router */
};

//=============================================================================
// Configuration API
//=============================================================================

broca_substrate_config_t broca_substrate_default_config(void) {
    return (broca_substrate_config_t){
        .enable_atp_modulation = true,
        .enable_fatigue_modulation = true,
        .enable_bio_async = true,
        .atp_sensitivity = 1.0f,
        .fatigue_sensitivity = 1.0f,
        .min_capacity = 0.1f,
        .fluency_atp_weight = 1.0f,
        .syntax_fatigue_weight = 1.0f
    };
}

//=============================================================================
// Lifecycle API
//=============================================================================

broca_substrate_bridge_t* broca_substrate_bridge_create(
    void* broca,
    neural_substrate_t* substrate,
    const broca_substrate_config_t* config
) {
    broca_substrate_bridge_t* bridge = nimcp_calloc(1, sizeof(broca_substrate_bridge_t));
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return NULL;

    }

    bridge->broca = broca;
    bridge->substrate = substrate;
    bridge->config = config ? *config : broca_substrate_default_config();

    /* Initialize effects to full capacity */
    bridge->effects.speech_fluency = 1.0f;
    bridge->effects.word_retrieval = 1.0f;
    bridge->effects.syntax_complexity = 1.0f;
    bridge->effects.articulation_precision = 1.0f;
    bridge->effects.phonological_accuracy = 1.0f;
    bridge->effects.overall_capacity = 1.0f;

    memset(&bridge->stats, 0, sizeof(bridge->stats));
    bridge->stats.min_observed_capacity = 1.0f;
    bridge->router = NULL;
    bridge->registered = false;

    return bridge;
}

void broca_substrate_bridge_destroy(broca_substrate_bridge_t* bridge) {
    if (bridge) {
        nimcp_free(bridge);
    }
}

//=============================================================================
// Update API
//=============================================================================

int broca_substrate_bridge_update(broca_substrate_bridge_t* bridge) {
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");
        return -1;
    }

    float atp_level = 1.0f;
    float fatigue_level = 0.0f;

    /* Get metabolic state from substrate if available */
    if (bridge->substrate) {
        substrate_metabolic_state_t metabolic_state;
        if (substrate_get_metabolic_state(bridge->substrate, &metabolic_state) == 0) {
            atp_level = metabolic_state.atp_level;
            fatigue_level = 1.0f - metabolic_state.metabolic_capacity;
        }
    }

    /* Compute ATP-modulated effects */
    float atp_factor = 1.0f;
    if (bridge->config.enable_atp_modulation) {
        /* ATP affects word retrieval and fluency most */
        atp_factor = atp_level * bridge->config.atp_sensitivity;
        if (atp_factor < bridge->config.min_capacity) {
            atp_factor = bridge->config.min_capacity;
        }
        if (atp_factor > 1.0f) atp_factor = 1.0f;

        if (atp_level < LOW_ATP_THRESHOLD) {
            bridge->stats.low_atp_events++;
        }
    }

    /* Compute fatigue-modulated effects */
    float fatigue_factor = 1.0f;
    if (bridge->config.enable_fatigue_modulation) {
        /* Fatigue affects syntax complexity and articulation precision */
        fatigue_factor = 1.0f - (fatigue_level * bridge->config.fatigue_sensitivity);
        if (fatigue_factor < bridge->config.min_capacity) {
            fatigue_factor = bridge->config.min_capacity;
        }
        if (fatigue_factor > 1.0f) fatigue_factor = 1.0f;

        if (fatigue_level > HIGH_FATIGUE_THRESHOLD) {
            bridge->stats.high_fatigue_events++;
        }
    }

    /* Apply effects with appropriate weights */
    bridge->effects.speech_fluency = atp_factor * bridge->config.fluency_atp_weight;
    if (bridge->effects.speech_fluency > 1.0f) bridge->effects.speech_fluency = 1.0f;

    bridge->effects.word_retrieval = atp_factor * 1.05f;  /* ATP-dependent */
    if (bridge->effects.word_retrieval > 1.0f) bridge->effects.word_retrieval = 1.0f;

    bridge->effects.syntax_complexity = fatigue_factor * bridge->config.syntax_fatigue_weight;
    if (bridge->effects.syntax_complexity > 1.0f) bridge->effects.syntax_complexity = 1.0f;

    bridge->effects.articulation_precision = fatigue_factor * 0.95f;  /* Fatigue-sensitive */
    if (bridge->effects.articulation_precision > 1.0f) bridge->effects.articulation_precision = 1.0f;

    bridge->effects.phonological_accuracy = (atp_factor + fatigue_factor) * 0.5f;  /* Both affect */
    if (bridge->effects.phonological_accuracy > 1.0f) bridge->effects.phonological_accuracy = 1.0f;

    /* Compute overall capacity */
    bridge->effects.overall_capacity = (
        bridge->effects.speech_fluency +
        bridge->effects.word_retrieval +
        bridge->effects.syntax_complexity +
        bridge->effects.articulation_precision +
        bridge->effects.phonological_accuracy
    ) / 5.0f;

    /* Update statistics */
    bridge->stats.updates_processed++;

    float prev_avg_fluency = bridge->stats.avg_speech_fluency;
    bridge->stats.avg_speech_fluency = (prev_avg_fluency * (bridge->stats.updates_processed - 1) +
                                        bridge->effects.speech_fluency) / bridge->stats.updates_processed;

    float prev_avg_syntax = bridge->stats.avg_syntax_complexity;
    bridge->stats.avg_syntax_complexity = (prev_avg_syntax * (bridge->stats.updates_processed - 1) +
                                           bridge->effects.syntax_complexity) / bridge->stats.updates_processed;

    if (bridge->effects.overall_capacity < bridge->stats.min_observed_capacity) {
        bridge->stats.min_observed_capacity = bridge->effects.overall_capacity;
    }

    return 0;
}

int broca_substrate_bridge_get_effects(
    const broca_substrate_bridge_t* bridge,
    broca_substrate_effects_t* effects
) {
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");
        return -1;
    }
    if (!effects) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "effects is NULL");
        return -1;
    }
    *effects = bridge->effects;
    return 0;
}

int broca_substrate_bridge_apply_effects(broca_substrate_bridge_t* bridge) {
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");
        return -1;
    }

    /* Update effects from substrate first */
    broca_substrate_bridge_update(bridge);

    /*
     * TODO: Apply effects to Broca adapter
     * Would call something like:
     * broca_adapter_set_fluency_factor(bridge->broca, bridge->effects.speech_fluency);
     * broca_adapter_set_syntax_capacity(bridge->broca, bridge->effects.syntax_complexity);
     */

    return 0;
}

//=============================================================================
// Bio-Async API
//=============================================================================

int broca_substrate_bridge_register_bio_async(
    broca_substrate_bridge_t* bridge,
    bio_router_t* router
) {
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");
        return -1;
    }
    if (!router) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "router is NULL");
        return -1;
    }
    if (!bridge->config.enable_bio_async) return 0;

    bridge->router = router;
    bridge->registered = true;

    /* Register as module with bio router */
    /* bio_router_register_module(router, BIO_MODULE_SUBSTRATE_BROCA, bridge); */

    return 0;
}

//=============================================================================
// Statistics API
//=============================================================================

int broca_substrate_bridge_get_stats(
    const broca_substrate_bridge_t* bridge,
    broca_substrate_stats_t* stats
) {
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");
        return -1;
    }
    if (!stats) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "stats is NULL");
        return -1;
    }
    *stats = bridge->stats;
    return 0;
}

void broca_substrate_bridge_reset_stats(broca_substrate_bridge_t* bridge) {
    if (bridge) {
        memset(&bridge->stats, 0, sizeof(bridge->stats));
        bridge->stats.min_observed_capacity = 1.0f;
    }
}
