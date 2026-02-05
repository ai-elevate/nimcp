//=============================================================================
// nimcp_hypothalamus_kg_wiring.c - Hypothalamus Knowledge Graph Registration Implementation
//=============================================================================
/**
 * @file nimcp_hypothalamus_kg_wiring.c
 * @brief Implementation of Hypothalamus Knowledge Graph registration
 *
 * WHAT: Implements KG node/edge creation for Hypothalamus module
 * WHY:  Enables semantic queries about homeostasis, circadian, and stress
 * HOW:  Creates hierarchical node structure with typed relationships
 */

#include "core/brain/regions/hypothalamus/bridges/nimcp_hypothalamus_kg_wiring.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>
#include <stdio.h>
#include "utils/fault_tolerance/nimcp_health_agent_macros.h"
#include "mesh/nimcp_mesh_participant.h"
#include "mesh/nimcp_mesh_adapter.h"

NIMCP_DECLARE_HEALTH_AGENT_ATOMIC(hypothalamus_kg_wiring)

//=============================================================================
// Mesh Participant Registration
//=============================================================================

static mesh_participant_id_t g_hypothalamus_kg_wiring_mesh_id = 0;
static mesh_participant_registry_t* g_hypothalamus_kg_wiring_mesh_registry = NULL;

nimcp_error_t hypothalamus_kg_wiring_mesh_register(mesh_participant_registry_t* registry) {
    if (!registry) return NIMCP_ERROR_NULL_POINTER;
    if (g_hypothalamus_kg_wiring_mesh_id != 0) return NIMCP_SUCCESS;
    mesh_participant_interface_t iface;
    mesh_participant_interface_init(&iface);
    strncpy(iface.module_name, "hypothalamus_kg_wiring", MESH_MAX_NAME_LEN - 1);
    iface.type = MESH_PARTICIPANT_MODULE;
    iface.home_channel = mesh_adapter_get_default_channel(MESH_ADAPTER_CATEGORY_SYSTEM);
    mesh_participant_config_t config;
    mesh_participant_config_init(&config);
    config.module_name = "hypothalamus_kg_wiring";
    config.type = MESH_PARTICIPANT_MODULE;
    config.home_channel = iface.home_channel;
    nimcp_error_t err = mesh_participant_register(registry, &iface, &config, &g_hypothalamus_kg_wiring_mesh_id);
    if (err == NIMCP_SUCCESS) g_hypothalamus_kg_wiring_mesh_registry = registry;
    return err;
}

void hypothalamus_kg_wiring_mesh_unregister(void) {
    if (g_hypothalamus_kg_wiring_mesh_registry && g_hypothalamus_kg_wiring_mesh_id != 0) {
        mesh_participant_unregister(g_hypothalamus_kg_wiring_mesh_registry, g_hypothalamus_kg_wiring_mesh_id);
        g_hypothalamus_kg_wiring_mesh_id = 0;
        g_hypothalamus_kg_wiring_mesh_registry = NULL;
    }
}

//=============================================================================
// Helper Functions
//=============================================================================

static brain_kg_node_id_t create_hypothalamus_node(
    brain_kg_t* kg,
    const char* name,
    brain_kg_node_type_t type,
    const char* description,
    uint64_t admin_token
) {
    (void)admin_token;
    if (!kg || !name) return BRAIN_KG_INVALID_NODE;

    brain_kg_node_id_t id = brain_kg_add_node(kg, name, type, description);
    if (id != BRAIN_KG_INVALID_NODE) {
        NIMCP_LOG_DEBUG(HYPOTHALAMUS_KG_MODULE_NAME,
            "Created node '%s' (id=%u)", name, id);
    }
    return id;
}

static brain_kg_edge_id_t create_hypothalamus_edge(
    brain_kg_t* kg,
    brain_kg_node_id_t from,
    brain_kg_node_id_t to,
    brain_kg_edge_type_t type,
    const char* description,
    float weight,
    uint64_t admin_token
) {
    (void)admin_token;
    if (!kg) return BRAIN_KG_INVALID_NODE;
    if (from == BRAIN_KG_INVALID_NODE || to == BRAIN_KG_INVALID_NODE) {
        return BRAIN_KG_INVALID_NODE;
    }
    return brain_kg_add_edge(kg, from, to, type, description, weight);
}

//=============================================================================
// Configuration API
//=============================================================================

int hypothalamus_kg_default_config(hypothalamus_kg_config_t* config) {
    if (!config) return -1;

    config->register_scn = true;
    config->register_pvn = true;
    config->register_vmh = true;
    config->register_lh = true;
    config->register_homeostasis = true;
    config->register_circadian_rhythm = true;
    config->register_stress_response = true;
    config->register_cross_edges = true;
    config->include_state_metadata = true;

    return 0;
}

//=============================================================================
// Registration API
//=============================================================================

int hypothalamus_kg_register_all(
    brain_kg_t* kg,
    const hypothalamus_kg_config_t* config,
    hypothalamus_kg_state_t* state,
    uint64_t admin_token
) {
    if (!kg) return -1;

    hypothalamus_kg_config_t local_config;
    if (config) {
        local_config = *config;
    } else {
        hypothalamus_kg_default_config(&local_config);
    }

    hypothalamus_kg_state_t local_state;
    memset(&local_state, 0, sizeof(local_state));

    /* Create root node */
    local_state.root_id = create_hypothalamus_node(
        kg, HYPOTHALAMUS_KG_ROOT_NAME,
        BRAIN_KG_NODE_SUBCORTICAL,
        "Hypothalamus - homeostasis, circadian rhythm, HPA axis, neuroendocrine control",
        admin_token
    );
    if (local_state.root_id == BRAIN_KG_INVALID_NODE) {
        NIMCP_LOG_ERROR(HYPOTHALAMUS_KG_MODULE_NAME, "Failed to create root node");
        return -1;
    }
    local_state.node_count++;

    /* SCN */
    if (local_config.register_scn) {
        local_state.scn_id = create_hypothalamus_node(
            kg, "SCN",
            (brain_kg_node_type_t)HYPOTHALAMUS_KG_NODE_SCN,
            "Suprachiasmatic nucleus - master circadian clock, light entrainment",
            admin_token
        );
        if (local_state.scn_id != BRAIN_KG_INVALID_NODE) {
            local_state.node_count++;
            create_hypothalamus_edge(kg, local_state.root_id, local_state.scn_id,
                BRAIN_KG_EDGE_CONNECTS_TO, "contains SCN", 1.0f, admin_token);
            local_state.edge_count++;
        }
    }

    /* PVN */
    if (local_config.register_pvn) {
        local_state.pvn_id = create_hypothalamus_node(
            kg, "PVN",
            (brain_kg_node_type_t)HYPOTHALAMUS_KG_NODE_PVN,
            "Paraventricular nucleus - CRH release, autonomic control, HPA axis",
            admin_token
        );
        if (local_state.pvn_id != BRAIN_KG_INVALID_NODE) {
            local_state.node_count++;
            create_hypothalamus_edge(kg, local_state.root_id, local_state.pvn_id,
                BRAIN_KG_EDGE_CONNECTS_TO, "contains PVN", 1.0f, admin_token);
            local_state.edge_count++;
        }
    }

    /* VMH */
    if (local_config.register_vmh) {
        local_state.vmh_id = create_hypothalamus_node(
            kg, "VMH",
            (brain_kg_node_type_t)HYPOTHALAMUS_KG_NODE_VMH,
            "Ventromedial hypothalamus - satiety center, defensive behavior",
            admin_token
        );
        if (local_state.vmh_id != BRAIN_KG_INVALID_NODE) {
            local_state.node_count++;
            create_hypothalamus_edge(kg, local_state.root_id, local_state.vmh_id,
                BRAIN_KG_EDGE_CONNECTS_TO, "contains VMH", 1.0f, admin_token);
            local_state.edge_count++;
        }
    }

    /* LH */
    if (local_config.register_lh) {
        local_state.lh_id = create_hypothalamus_node(
            kg, "LH",
            (brain_kg_node_type_t)HYPOTHALAMUS_KG_NODE_LH,
            "Lateral hypothalamus - hunger center, arousal, reward seeking",
            admin_token
        );
        if (local_state.lh_id != BRAIN_KG_INVALID_NODE) {
            local_state.node_count++;
            create_hypothalamus_edge(kg, local_state.root_id, local_state.lh_id,
                BRAIN_KG_EDGE_CONNECTS_TO, "contains LH", 1.0f, admin_token);
            local_state.edge_count++;
        }
    }

    /* Homeostasis */
    if (local_config.register_homeostasis) {
        local_state.homeostasis_id = create_hypothalamus_node(
            kg, "homeostasis",
            (brain_kg_node_type_t)HYPOTHALAMUS_KG_NODE_HOMEOSTASIS,
            "Homeostasis - temperature, osmolality, energy balance regulation",
            admin_token
        );
        if (local_state.homeostasis_id != BRAIN_KG_INVALID_NODE) {
            local_state.node_count++;
            create_hypothalamus_edge(kg, local_state.root_id, local_state.homeostasis_id,
                BRAIN_KG_EDGE_CONNECTS_TO, "contains homeostasis", 1.0f, admin_token);
            local_state.edge_count++;
        }
    }

    /* Circadian rhythm */
    if (local_config.register_circadian_rhythm) {
        local_state.circadian_rhythm_id = create_hypothalamus_node(
            kg, "circadian_rhythm",
            (brain_kg_node_type_t)HYPOTHALAMUS_KG_NODE_CIRCADIAN,
            "Circadian rhythm - sleep-wake cycle, hormonal oscillation, photoperiod",
            admin_token
        );
        if (local_state.circadian_rhythm_id != BRAIN_KG_INVALID_NODE) {
            local_state.node_count++;
            create_hypothalamus_edge(kg, local_state.root_id, local_state.circadian_rhythm_id,
                BRAIN_KG_EDGE_CONNECTS_TO, "contains circadian rhythm", 1.0f, admin_token);
            local_state.edge_count++;
        }
    }

    /* Stress response */
    if (local_config.register_stress_response) {
        local_state.stress_response_id = create_hypothalamus_node(
            kg, "stress_response",
            (brain_kg_node_type_t)HYPOTHALAMUS_KG_NODE_STRESS,
            "Stress response - HPA axis activation, cortisol regulation, fight-or-flight",
            admin_token
        );
        if (local_state.stress_response_id != BRAIN_KG_INVALID_NODE) {
            local_state.node_count++;
            create_hypothalamus_edge(kg, local_state.root_id, local_state.stress_response_id,
                BRAIN_KG_EDGE_CONNECTS_TO, "contains stress response", 1.0f, admin_token);
            local_state.edge_count++;
        }
    }

    /* Cross edges */
    if (local_config.register_cross_edges) {
        /* SCN entrains circadian rhythm */
        if (local_state.scn_id != BRAIN_KG_INVALID_NODE &&
            local_state.circadian_rhythm_id != BRAIN_KG_INVALID_NODE) {
            create_hypothalamus_edge(kg, local_state.scn_id,
                local_state.circadian_rhythm_id,
                (brain_kg_edge_type_t)HYPOTHALAMUS_KG_EDGE_ENTRAINS,
                "SCN entrains circadian rhythm", 0.95f, admin_token);
            local_state.edge_count++;
        }
        /* PVN activates HPA stress response */
        if (local_state.pvn_id != BRAIN_KG_INVALID_NODE &&
            local_state.stress_response_id != BRAIN_KG_INVALID_NODE) {
            create_hypothalamus_edge(kg, local_state.pvn_id,
                local_state.stress_response_id,
                (brain_kg_edge_type_t)HYPOTHALAMUS_KG_EDGE_ACTIVATES_HPA,
                "PVN activates HPA axis stress response", 0.9f, admin_token);
            local_state.edge_count++;
        }
        /* VMH regulates homeostasis */
        if (local_state.vmh_id != BRAIN_KG_INVALID_NODE &&
            local_state.homeostasis_id != BRAIN_KG_INVALID_NODE) {
            create_hypothalamus_edge(kg, local_state.vmh_id,
                local_state.homeostasis_id,
                (brain_kg_edge_type_t)HYPOTHALAMUS_KG_EDGE_REGULATES,
                "VMH regulates satiety and energy balance", 0.85f, admin_token);
            local_state.edge_count++;
        }
        /* LH drives feeding and arousal */
        if (local_state.lh_id != BRAIN_KG_INVALID_NODE &&
            local_state.homeostasis_id != BRAIN_KG_INVALID_NODE) {
            create_hypothalamus_edge(kg, local_state.lh_id,
                local_state.homeostasis_id,
                (brain_kg_edge_type_t)HYPOTHALAMUS_KG_EDGE_DRIVES,
                "LH drives hunger and arousal signaling", 0.85f, admin_token);
            local_state.edge_count++;
        }
        /* Circadian modulates homeostasis */
        if (local_state.circadian_rhythm_id != BRAIN_KG_INVALID_NODE &&
            local_state.homeostasis_id != BRAIN_KG_INVALID_NODE) {
            create_hypothalamus_edge(kg, local_state.circadian_rhythm_id,
                local_state.homeostasis_id,
                BRAIN_KG_EDGE_MODULATES,
                "circadian rhythm modulates homeostatic setpoints", 0.8f, admin_token);
            local_state.edge_count++;
        }
    }

    local_state.registered = true;
    if (state) *state = local_state;

    NIMCP_LOG_INFO(HYPOTHALAMUS_KG_MODULE_NAME,
        "Registered %u nodes, %u edges",
        local_state.node_count, local_state.edge_count);

    return 0;
}

//=============================================================================
// Query API
//=============================================================================

brain_kg_node_id_t hypothalamus_kg_get_root(brain_kg_t* kg) {
    if (!kg) return BRAIN_KG_INVALID_NODE;
    return brain_kg_find_node(kg, HYPOTHALAMUS_KG_ROOT_NAME);
}

brain_kg_node_id_t hypothalamus_kg_find_subsystem(brain_kg_t* kg, const char* name) {
    if (!kg || !name) return BRAIN_KG_INVALID_NODE;
    return brain_kg_find_node(kg, name);
}

int hypothalamus_kg_unregister_all(
    brain_kg_t* kg,
    hypothalamus_kg_state_t* state,
    uint64_t admin_token
) {
    (void)admin_token;
    if (!kg || !state) return -1;

    state->registered = false;
    state->node_count = 0;
    state->edge_count = 0;

    NIMCP_LOG_INFO(HYPOTHALAMUS_KG_MODULE_NAME, "Unregistered Hypothalamus KG nodes");
    return 0;
}
