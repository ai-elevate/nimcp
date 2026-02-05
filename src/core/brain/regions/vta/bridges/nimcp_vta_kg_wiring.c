//=============================================================================
// nimcp_vta_kg_wiring.c - VTA Knowledge Graph Registration Implementation
//=============================================================================
/**
 * @file nimcp_vta_kg_wiring.c
 * @brief Implementation of VTA Knowledge Graph registration
 *
 * WHAT: Implements KG node/edge creation for VTA module
 * WHY:  Enables semantic queries about reward, motivation, and reinforcement
 * HOW:  Creates hierarchical node structure with typed relationships
 */

#include "core/brain/regions/vta/bridges/nimcp_vta_kg_wiring.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>
#include <stdio.h>
#include "utils/fault_tolerance/nimcp_health_agent_macros.h"
#include "mesh/nimcp_mesh_participant.h"
#include "mesh/nimcp_mesh_adapter.h"

NIMCP_DECLARE_HEALTH_AGENT_ATOMIC(vta_kg_wiring)

//=============================================================================
// Mesh Participant Registration
//=============================================================================

static mesh_participant_id_t g_vta_kg_wiring_mesh_id = 0;
static mesh_participant_registry_t* g_vta_kg_wiring_mesh_registry = NULL;

nimcp_error_t vta_kg_wiring_mesh_register(mesh_participant_registry_t* registry) {
    if (!registry) return NIMCP_ERROR_NULL_POINTER;
    if (g_vta_kg_wiring_mesh_id != 0) return NIMCP_SUCCESS;
    mesh_participant_interface_t iface;
    mesh_participant_interface_init(&iface);
    strncpy(iface.module_name, "vta_kg_wiring", MESH_MAX_NAME_LEN - 1);
    iface.type = MESH_PARTICIPANT_MODULE;
    iface.home_channel = mesh_adapter_get_default_channel(MESH_ADAPTER_CATEGORY_SYSTEM);
    mesh_participant_config_t config;
    mesh_participant_config_init(&config);
    config.module_name = "vta_kg_wiring";
    config.type = MESH_PARTICIPANT_MODULE;
    config.home_channel = iface.home_channel;
    nimcp_error_t err = mesh_participant_register(registry, &iface, &config, &g_vta_kg_wiring_mesh_id);
    if (err == NIMCP_SUCCESS) g_vta_kg_wiring_mesh_registry = registry;
    return err;
}

void vta_kg_wiring_mesh_unregister(void) {
    if (g_vta_kg_wiring_mesh_registry && g_vta_kg_wiring_mesh_id != 0) {
        mesh_participant_unregister(g_vta_kg_wiring_mesh_registry, g_vta_kg_wiring_mesh_id);
        g_vta_kg_wiring_mesh_id = 0;
        g_vta_kg_wiring_mesh_registry = NULL;
    }
}

//=============================================================================
// Helper Functions
//=============================================================================

static brain_kg_node_id_t create_vta_node(
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
        NIMCP_LOG_DEBUG(VTA_KG_MODULE_NAME,
            "Created node '%s' (id=%u)", name, id);
    }
    return id;
}

static brain_kg_edge_id_t create_vta_edge(
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

int vta_kg_default_config(vta_kg_config_t* config) {
    if (!config) return -1;

    config->register_dopaminergic_neurons = true;
    config->register_reward_prediction = true;
    config->register_motivation_drive = true;
    config->register_reinforcement_learning = true;
    config->register_cross_edges = true;
    config->include_state_metadata = true;

    return 0;
}

//=============================================================================
// Registration API
//=============================================================================

int vta_kg_register_all(
    brain_kg_t* kg,
    const vta_kg_config_t* config,
    vta_kg_state_t* state,
    uint64_t admin_token
) {
    if (!kg) return -1;

    vta_kg_config_t local_config;
    if (config) {
        local_config = *config;
    } else {
        vta_kg_default_config(&local_config);
    }

    vta_kg_state_t local_state;
    memset(&local_state, 0, sizeof(local_state));

    /* Create root node */
    local_state.root_id = create_vta_node(
        kg, VTA_KG_ROOT_NAME,
        BRAIN_KG_NODE_SUBCORTICAL,
        "Ventral Tegmental Area - dopamine release, reward prediction, motivation",
        admin_token
    );
    if (local_state.root_id == BRAIN_KG_INVALID_NODE) {
        NIMCP_LOG_ERROR(VTA_KG_MODULE_NAME, "Failed to create root node");
        return -1;
    }
    local_state.node_count++;

    /* Dopaminergic neurons */
    if (local_config.register_dopaminergic_neurons) {
        local_state.dopaminergic_neurons_id = create_vta_node(
            kg, "dopaminergic_neurons",
            (brain_kg_node_type_t)VTA_KG_NODE_DOPAMINERGIC,
            "Dopaminergic neurons - DA release to NAc and PFC, reward signaling",
            admin_token
        );
        if (local_state.dopaminergic_neurons_id != BRAIN_KG_INVALID_NODE) {
            local_state.node_count++;
            create_vta_edge(kg, local_state.root_id, local_state.dopaminergic_neurons_id,
                BRAIN_KG_EDGE_CONNECTS_TO, "contains dopaminergic neurons", 1.0f, admin_token);
            local_state.edge_count++;
        }
    }

    /* Reward prediction */
    if (local_config.register_reward_prediction) {
        local_state.reward_prediction_id = create_vta_node(
            kg, "reward_prediction",
            (brain_kg_node_type_t)VTA_KG_NODE_REWARD_PREDICTION,
            "Reward prediction - reward prediction error, expected value computation",
            admin_token
        );
        if (local_state.reward_prediction_id != BRAIN_KG_INVALID_NODE) {
            local_state.node_count++;
            create_vta_edge(kg, local_state.root_id, local_state.reward_prediction_id,
                BRAIN_KG_EDGE_CONNECTS_TO, "contains reward prediction", 1.0f, admin_token);
            local_state.edge_count++;
        }
    }

    /* Motivation drive */
    if (local_config.register_motivation_drive) {
        local_state.motivation_drive_id = create_vta_node(
            kg, "motivation_drive",
            (brain_kg_node_type_t)VTA_KG_NODE_MOTIVATION,
            "Motivation drive - incentive salience, wanting, approach behavior",
            admin_token
        );
        if (local_state.motivation_drive_id != BRAIN_KG_INVALID_NODE) {
            local_state.node_count++;
            create_vta_edge(kg, local_state.root_id, local_state.motivation_drive_id,
                BRAIN_KG_EDGE_CONNECTS_TO, "contains motivation drive", 1.0f, admin_token);
            local_state.edge_count++;
        }
    }

    /* Reinforcement learning */
    if (local_config.register_reinforcement_learning) {
        local_state.reinforcement_learning_id = create_vta_node(
            kg, "reinforcement_learning",
            (brain_kg_node_type_t)VTA_KG_NODE_REINFORCEMENT,
            "Reinforcement learning - TD learning, policy update, action-value mapping",
            admin_token
        );
        if (local_state.reinforcement_learning_id != BRAIN_KG_INVALID_NODE) {
            local_state.node_count++;
            create_vta_edge(kg, local_state.root_id, local_state.reinforcement_learning_id,
                BRAIN_KG_EDGE_CONNECTS_TO, "contains reinforcement learning", 1.0f, admin_token);
            local_state.edge_count++;
        }
    }

    /* Cross edges */
    if (local_config.register_cross_edges) {
        /* Dopaminergic neurons release DA for reward prediction */
        if (local_state.dopaminergic_neurons_id != BRAIN_KG_INVALID_NODE &&
            local_state.reward_prediction_id != BRAIN_KG_INVALID_NODE) {
            create_vta_edge(kg, local_state.dopaminergic_neurons_id,
                local_state.reward_prediction_id,
                (brain_kg_edge_type_t)VTA_KG_EDGE_RELEASES_DA,
                "DA neurons encode reward prediction error", 0.95f, admin_token);
            local_state.edge_count++;
        }
        /* Reward prediction drives motivation */
        if (local_state.reward_prediction_id != BRAIN_KG_INVALID_NODE &&
            local_state.motivation_drive_id != BRAIN_KG_INVALID_NODE) {
            create_vta_edge(kg, local_state.reward_prediction_id,
                local_state.motivation_drive_id,
                (brain_kg_edge_type_t)VTA_KG_EDGE_DRIVES_MOTIVATION,
                "reward prediction drives motivational salience", 0.9f, admin_token);
            local_state.edge_count++;
        }
        /* Reward prediction error reinforces learning */
        if (local_state.reward_prediction_id != BRAIN_KG_INVALID_NODE &&
            local_state.reinforcement_learning_id != BRAIN_KG_INVALID_NODE) {
            create_vta_edge(kg, local_state.reward_prediction_id,
                local_state.reinforcement_learning_id,
                (brain_kg_edge_type_t)VTA_KG_EDGE_REINFORCES,
                "RPE signal reinforces action-value learning", 0.9f, admin_token);
            local_state.edge_count++;
        }
        /* Motivation modulates reinforcement learning */
        if (local_state.motivation_drive_id != BRAIN_KG_INVALID_NODE &&
            local_state.reinforcement_learning_id != BRAIN_KG_INVALID_NODE) {
            create_vta_edge(kg, local_state.motivation_drive_id,
                local_state.reinforcement_learning_id,
                BRAIN_KG_EDGE_MODULATES,
                "motivation level modulates learning rate", 0.8f, admin_token);
            local_state.edge_count++;
        }
    }

    local_state.registered = true;
    if (state) *state = local_state;

    NIMCP_LOG_INFO(VTA_KG_MODULE_NAME,
        "Registered %u nodes, %u edges",
        local_state.node_count, local_state.edge_count);

    return 0;
}

//=============================================================================
// Query API
//=============================================================================

brain_kg_node_id_t vta_kg_get_root(brain_kg_t* kg) {
    if (!kg) return BRAIN_KG_INVALID_NODE;
    return brain_kg_find_node(kg, VTA_KG_ROOT_NAME);
}

brain_kg_node_id_t vta_kg_find_subsystem(brain_kg_t* kg, const char* name) {
    if (!kg || !name) return BRAIN_KG_INVALID_NODE;
    return brain_kg_find_node(kg, name);
}

int vta_kg_unregister_all(
    brain_kg_t* kg,
    vta_kg_state_t* state,
    uint64_t admin_token
) {
    (void)admin_token;
    if (!kg || !state) return -1;

    state->registered = false;
    state->node_count = 0;
    state->edge_count = 0;

    NIMCP_LOG_INFO(VTA_KG_MODULE_NAME, "Unregistered VTA KG nodes");
    return 0;
}
