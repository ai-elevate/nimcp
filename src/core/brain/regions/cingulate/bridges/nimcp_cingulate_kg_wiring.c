//=============================================================================
// nimcp_cingulate_kg_wiring.c - Cingulate Cortex Knowledge Graph Registration Implementation
//=============================================================================
/**
 * @file nimcp_cingulate_kg_wiring.c
 * @brief Implementation of Cingulate Cortex Knowledge Graph registration
 *
 * WHAT: Implements KG node/edge creation for Cingulate Cortex module
 * WHY:  Enables semantic queries about conflict monitoring and error detection
 * HOW:  Creates hierarchical node structure with typed relationships
 */

#include "core/brain/regions/cingulate/bridges/nimcp_cingulate_kg_wiring.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>
#include <stdio.h>
#include "utils/fault_tolerance/nimcp_health_agent_macros.h"
#include "mesh/nimcp_mesh_participant.h"
#include "mesh/nimcp_mesh_adapter.h"

NIMCP_DECLARE_HEALTH_AGENT_ATOMIC(cingulate_kg_wiring)

//=============================================================================
// Mesh Participant Registration
//=============================================================================

static mesh_participant_id_t g_cingulate_kg_wiring_mesh_id = 0;
static mesh_participant_registry_t* g_cingulate_kg_wiring_mesh_registry = NULL;

nimcp_error_t cingulate_kg_wiring_mesh_register(mesh_participant_registry_t* registry) {
    if (!registry) return NIMCP_ERROR_NULL_POINTER;
    if (g_cingulate_kg_wiring_mesh_id != 0) return NIMCP_SUCCESS;
    mesh_participant_interface_t iface;
    mesh_participant_interface_init(&iface);
    strncpy(iface.module_name, "cingulate_kg_wiring", MESH_MAX_NAME_LEN - 1);
    iface.type = MESH_PARTICIPANT_MODULE;
    iface.home_channel = mesh_adapter_get_default_channel(MESH_ADAPTER_CATEGORY_SYSTEM);
    mesh_participant_config_t config;
    mesh_participant_config_init(&config);
    config.module_name = "cingulate_kg_wiring";
    config.type = MESH_PARTICIPANT_MODULE;
    config.home_channel = iface.home_channel;
    nimcp_error_t err = mesh_participant_register(registry, &iface, &config, &g_cingulate_kg_wiring_mesh_id);
    if (err == NIMCP_SUCCESS) g_cingulate_kg_wiring_mesh_registry = registry;
    return err;
}

void cingulate_kg_wiring_mesh_unregister(void) {
    if (g_cingulate_kg_wiring_mesh_registry && g_cingulate_kg_wiring_mesh_id != 0) {
        mesh_participant_unregister(g_cingulate_kg_wiring_mesh_registry, g_cingulate_kg_wiring_mesh_id);
        g_cingulate_kg_wiring_mesh_id = 0;
        g_cingulate_kg_wiring_mesh_registry = NULL;
    }
}

//=============================================================================
// Helper Functions
//=============================================================================

static brain_kg_node_id_t create_cingulate_node(
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
        NIMCP_LOG_DEBUG(CINGULATE_KG_MODULE_NAME,
            "Created node '%s' (id=%u)", name, id);
    }
    return id;
}

static brain_kg_edge_id_t create_cingulate_edge(
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

int cingulate_kg_default_config(cingulate_kg_config_t* config) {
    if (!config) return -1;

    config->register_dacc = true;
    config->register_vacc = true;
    config->register_pcc = true;
    config->register_conflict_detection = true;
    config->register_error_monitoring = true;
    config->register_reward_processing = true;
    config->register_cross_edges = true;
    config->include_state_metadata = true;

    return 0;
}

//=============================================================================
// Registration API
//=============================================================================

int cingulate_kg_register_all(
    brain_kg_t* kg,
    const cingulate_kg_config_t* config,
    cingulate_kg_state_t* state,
    uint64_t admin_token
) {
    if (!kg) return -1;

    cingulate_kg_config_t local_config;
    if (config) {
        local_config = *config;
    } else {
        cingulate_kg_default_config(&local_config);
    }

    cingulate_kg_state_t local_state;
    memset(&local_state, 0, sizeof(local_state));

    /* Create root node */
    local_state.root_id = create_cingulate_node(
        kg, CINGULATE_KG_ROOT_NAME,
        BRAIN_KG_NODE_CORTICAL,
        "Cingulate cortex - conflict monitoring, error detection, reward evaluation",
        admin_token
    );
    if (local_state.root_id == BRAIN_KG_INVALID_NODE) {
        NIMCP_LOG_ERROR(CINGULATE_KG_MODULE_NAME, "Failed to create root node");
        return -1;
    }
    local_state.node_count++;

    /* dACC */
    if (local_config.register_dacc) {
        local_state.dacc_id = create_cingulate_node(
            kg, "dACC",
            (brain_kg_node_type_t)CINGULATE_KG_NODE_DORSAL_ACC,
            "Dorsal ACC - cognitive conflict monitoring, response selection",
            admin_token
        );
        if (local_state.dacc_id != BRAIN_KG_INVALID_NODE) {
            local_state.node_count++;
            create_cingulate_edge(kg, local_state.root_id, local_state.dacc_id,
                BRAIN_KG_EDGE_CONNECTS_TO, "contains dACC", 1.0f, admin_token);
            local_state.edge_count++;
        }
    }

    /* vACC */
    if (local_config.register_vacc) {
        local_state.vacc_id = create_cingulate_node(
            kg, "vACC",
            (brain_kg_node_type_t)CINGULATE_KG_NODE_VENTRAL_ACC,
            "Ventral ACC - emotional regulation, autonomic control",
            admin_token
        );
        if (local_state.vacc_id != BRAIN_KG_INVALID_NODE) {
            local_state.node_count++;
            create_cingulate_edge(kg, local_state.root_id, local_state.vacc_id,
                BRAIN_KG_EDGE_CONNECTS_TO, "contains vACC", 1.0f, admin_token);
            local_state.edge_count++;
        }
    }

    /* PCC */
    if (local_config.register_pcc) {
        local_state.pcc_id = create_cingulate_node(
            kg, "PCC",
            (brain_kg_node_type_t)CINGULATE_KG_NODE_PCC,
            "Posterior cingulate - self-referential processing, default mode network",
            admin_token
        );
        if (local_state.pcc_id != BRAIN_KG_INVALID_NODE) {
            local_state.node_count++;
            create_cingulate_edge(kg, local_state.root_id, local_state.pcc_id,
                BRAIN_KG_EDGE_CONNECTS_TO, "contains PCC", 1.0f, admin_token);
            local_state.edge_count++;
        }
    }

    /* Conflict detection */
    if (local_config.register_conflict_detection) {
        local_state.conflict_detection_id = create_cingulate_node(
            kg, "conflict_detection",
            (brain_kg_node_type_t)CINGULATE_KG_NODE_CONFLICT,
            "Conflict detection - response conflict monitoring, cognitive control signaling",
            admin_token
        );
        if (local_state.conflict_detection_id != BRAIN_KG_INVALID_NODE) {
            local_state.node_count++;
            create_cingulate_edge(kg, local_state.root_id, local_state.conflict_detection_id,
                BRAIN_KG_EDGE_CONNECTS_TO, "contains conflict detection", 1.0f, admin_token);
            local_state.edge_count++;
        }
    }

    /* Error monitoring */
    if (local_config.register_error_monitoring) {
        local_state.error_monitoring_id = create_cingulate_node(
            kg, "error_monitoring",
            (brain_kg_node_type_t)CINGULATE_KG_NODE_ERROR_MONITOR,
            "Error monitoring - error-related negativity, performance adjustment",
            admin_token
        );
        if (local_state.error_monitoring_id != BRAIN_KG_INVALID_NODE) {
            local_state.node_count++;
            create_cingulate_edge(kg, local_state.root_id, local_state.error_monitoring_id,
                BRAIN_KG_EDGE_CONNECTS_TO, "contains error monitoring", 1.0f, admin_token);
            local_state.edge_count++;
        }
    }

    /* Reward processing */
    if (local_config.register_reward_processing) {
        local_state.reward_processing_id = create_cingulate_node(
            kg, "reward_processing",
            (brain_kg_node_type_t)CINGULATE_KG_NODE_REWARD,
            "Reward processing - outcome evaluation, reward-based decision making",
            admin_token
        );
        if (local_state.reward_processing_id != BRAIN_KG_INVALID_NODE) {
            local_state.node_count++;
            create_cingulate_edge(kg, local_state.root_id, local_state.reward_processing_id,
                BRAIN_KG_EDGE_CONNECTS_TO, "contains reward processing", 1.0f, admin_token);
            local_state.edge_count++;
        }
    }

    /* Cross edges */
    if (local_config.register_cross_edges) {
        /* dACC drives conflict detection */
        if (local_state.dacc_id != BRAIN_KG_INVALID_NODE &&
            local_state.conflict_detection_id != BRAIN_KG_INVALID_NODE) {
            create_cingulate_edge(kg, local_state.dacc_id,
                local_state.conflict_detection_id,
                (brain_kg_edge_type_t)CINGULATE_KG_EDGE_DETECTS,
                "dACC detects response conflict", 0.9f, admin_token);
            local_state.edge_count++;
        }
        /* dACC monitors errors */
        if (local_state.dacc_id != BRAIN_KG_INVALID_NODE &&
            local_state.error_monitoring_id != BRAIN_KG_INVALID_NODE) {
            create_cingulate_edge(kg, local_state.dacc_id,
                local_state.error_monitoring_id,
                (brain_kg_edge_type_t)CINGULATE_KG_EDGE_MONITORS,
                "dACC monitors performance errors", 0.85f, admin_token);
            local_state.edge_count++;
        }
        /* vACC evaluates reward */
        if (local_state.vacc_id != BRAIN_KG_INVALID_NODE &&
            local_state.reward_processing_id != BRAIN_KG_INVALID_NODE) {
            create_cingulate_edge(kg, local_state.vacc_id,
                local_state.reward_processing_id,
                (brain_kg_edge_type_t)CINGULATE_KG_EDGE_EVALUATES,
                "vACC evaluates reward outcomes", 0.85f, admin_token);
            local_state.edge_count++;
        }
        /* Error monitoring signals conflict detection */
        if (local_state.error_monitoring_id != BRAIN_KG_INVALID_NODE &&
            local_state.conflict_detection_id != BRAIN_KG_INVALID_NODE) {
            create_cingulate_edge(kg, local_state.error_monitoring_id,
                local_state.conflict_detection_id,
                (brain_kg_edge_type_t)CINGULATE_KG_EDGE_SIGNALS_ERROR,
                "errors increase conflict sensitivity", 0.8f, admin_token);
            local_state.edge_count++;
        }
    }

    local_state.registered = true;
    if (state) *state = local_state;

    NIMCP_LOG_INFO(CINGULATE_KG_MODULE_NAME,
        "Registered %u nodes, %u edges",
        local_state.node_count, local_state.edge_count);

    return 0;
}

//=============================================================================
// Query API
//=============================================================================

brain_kg_node_id_t cingulate_kg_get_root(brain_kg_t* kg) {
    if (!kg) return BRAIN_KG_INVALID_NODE;
    return brain_kg_find_node(kg, CINGULATE_KG_ROOT_NAME);
}

brain_kg_node_id_t cingulate_kg_find_subsystem(brain_kg_t* kg, const char* name) {
    if (!kg || !name) return BRAIN_KG_INVALID_NODE;
    return brain_kg_find_node(kg, name);
}

int cingulate_kg_unregister_all(
    brain_kg_t* kg,
    cingulate_kg_state_t* state,
    uint64_t admin_token
) {
    (void)admin_token;
    if (!kg || !state) return -1;

    state->registered = false;
    state->node_count = 0;
    state->edge_count = 0;

    NIMCP_LOG_INFO(CINGULATE_KG_MODULE_NAME, "Unregistered Cingulate KG nodes");
    return 0;
}
