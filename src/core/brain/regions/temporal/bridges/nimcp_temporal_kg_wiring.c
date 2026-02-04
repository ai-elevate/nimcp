//=============================================================================
// nimcp_temporal_kg_wiring.c - Temporal Lobe Knowledge Graph Registration Implementation
//=============================================================================
/**
 * @file nimcp_temporal_kg_wiring.c
 * @brief Implementation of Temporal Lobe Knowledge Graph registration
 *
 * WHAT: Implements KG node/edge creation for Temporal Lobe module
 * WHY:  Enables semantic queries about auditory and recognition processing
 * HOW:  Creates hierarchical node structure with typed relationships
 */

#include "core/brain/regions/temporal/bridges/nimcp_temporal_kg_wiring.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>
#include <stdio.h>
#include "utils/fault_tolerance/nimcp_health_agent_macros.h"
#include "mesh/nimcp_mesh_participant.h"
#include "mesh/nimcp_mesh_adapter.h"

NIMCP_DECLARE_HEALTH_AGENT_ATOMIC(temporal_kg_wiring)

//=============================================================================
// Mesh Participant Registration
//=============================================================================

static mesh_participant_id_t g_temporal_kg_wiring_mesh_id = 0;
static mesh_participant_registry_t* g_temporal_kg_wiring_mesh_registry = NULL;

nimcp_error_t temporal_kg_wiring_mesh_register(mesh_participant_registry_t* registry) {
    if (!registry) return NIMCP_ERROR_NULL_POINTER;
    if (g_temporal_kg_wiring_mesh_id != 0) return NIMCP_SUCCESS;
    mesh_participant_interface_t iface;
    mesh_participant_interface_init(&iface);
    strncpy(iface.module_name, "temporal_kg_wiring", MESH_MAX_NAME_LEN - 1);
    iface.type = MESH_PARTICIPANT_MODULE;
    iface.home_channel = mesh_adapter_get_default_channel(MESH_ADAPTER_CATEGORY_SYSTEM);
    mesh_participant_config_t config;
    mesh_participant_config_init(&config);
    config.module_name = "temporal_kg_wiring";
    config.type = MESH_PARTICIPANT_MODULE;
    config.home_channel = iface.home_channel;
    nimcp_error_t err = mesh_participant_register(registry, &iface, &config, &g_temporal_kg_wiring_mesh_id);
    if (err == NIMCP_SUCCESS) g_temporal_kg_wiring_mesh_registry = registry;
    return err;
}

void temporal_kg_wiring_mesh_unregister(void) {
    if (g_temporal_kg_wiring_mesh_registry && g_temporal_kg_wiring_mesh_id != 0) {
        mesh_participant_unregister(g_temporal_kg_wiring_mesh_registry, g_temporal_kg_wiring_mesh_id);
        g_temporal_kg_wiring_mesh_id = 0;
        g_temporal_kg_wiring_mesh_registry = NULL;
    }
}

//=============================================================================
// Helper Functions
//=============================================================================

static brain_kg_node_id_t create_temporal_node(
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
        NIMCP_LOG_DEBUG(TEMPORAL_KG_MODULE_NAME,
            "Created node '%s' (id=%u)", name, id);
    }
    return id;
}

static brain_kg_edge_id_t create_temporal_edge(
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

int temporal_kg_default_config(temporal_kg_config_t* config) {
    if (!config) return -1;

    config->register_auditory_processing = true;
    config->register_object_recognition = true;
    config->register_face_processing = true;
    config->register_memory_encoding = true;
    config->register_cross_edges = true;
    config->include_state_metadata = true;

    return 0;
}

//=============================================================================
// Registration API
//=============================================================================

int temporal_kg_register_all(
    brain_kg_t* kg,
    const temporal_kg_config_t* config,
    temporal_kg_state_t* state,
    uint64_t admin_token
) {
    if (!kg) return -1;

    temporal_kg_config_t local_config;
    if (config) {
        local_config = *config;
    } else {
        temporal_kg_default_config(&local_config);
    }

    temporal_kg_state_t local_state;
    memset(&local_state, 0, sizeof(local_state));

    /* Create root node */
    local_state.root_id = create_temporal_node(
        kg, TEMPORAL_KG_ROOT_NAME,
        BRAIN_KG_NODE_CORTICAL,
        "Temporal lobe - auditory processing, object recognition, memory encoding",
        admin_token
    );
    if (local_state.root_id == BRAIN_KG_INVALID_NODE) {
        NIMCP_LOG_ERROR(TEMPORAL_KG_MODULE_NAME, "Failed to create root node");
        return -1;
    }
    local_state.node_count++;

    /* Auditory processing */
    if (local_config.register_auditory_processing) {
        local_state.auditory_processing_id = create_temporal_node(
            kg, "auditory_processing",
            (brain_kg_node_type_t)TEMPORAL_KG_NODE_AUDITORY,
            "Auditory processing - sound analysis, frequency decomposition, temporal patterns",
            admin_token
        );
        if (local_state.auditory_processing_id != BRAIN_KG_INVALID_NODE) {
            local_state.node_count++;
            create_temporal_edge(kg, local_state.root_id, local_state.auditory_processing_id,
                BRAIN_KG_EDGE_CONNECTS_TO, "contains auditory processing", 1.0f, admin_token);
            local_state.edge_count++;
        }
    }

    /* Object recognition */
    if (local_config.register_object_recognition) {
        local_state.object_recognition_id = create_temporal_node(
            kg, "object_recognition",
            (brain_kg_node_type_t)TEMPORAL_KG_NODE_RECOGNITION,
            "Object recognition - visual object identification, categorization",
            admin_token
        );
        if (local_state.object_recognition_id != BRAIN_KG_INVALID_NODE) {
            local_state.node_count++;
            create_temporal_edge(kg, local_state.root_id, local_state.object_recognition_id,
                BRAIN_KG_EDGE_CONNECTS_TO, "contains object recognition", 1.0f, admin_token);
            local_state.edge_count++;
        }
    }

    /* Face processing */
    if (local_config.register_face_processing) {
        local_state.face_processing_id = create_temporal_node(
            kg, "face_processing",
            (brain_kg_node_type_t)TEMPORAL_KG_NODE_FACE_AREA,
            "Face processing - fusiform face area, identity recognition, expression analysis",
            admin_token
        );
        if (local_state.face_processing_id != BRAIN_KG_INVALID_NODE) {
            local_state.node_count++;
            create_temporal_edge(kg, local_state.root_id, local_state.face_processing_id,
                BRAIN_KG_EDGE_CONNECTS_TO, "contains face processing", 1.0f, admin_token);
            local_state.edge_count++;
        }
    }

    /* Memory encoding */
    if (local_config.register_memory_encoding) {
        local_state.memory_encoding_id = create_temporal_node(
            kg, "memory_encoding",
            (brain_kg_node_type_t)TEMPORAL_KG_NODE_MEMORY_ENCODER,
            "Memory encoding - medial temporal lobe encoding, declarative memory formation",
            admin_token
        );
        if (local_state.memory_encoding_id != BRAIN_KG_INVALID_NODE) {
            local_state.node_count++;
            create_temporal_edge(kg, local_state.root_id, local_state.memory_encoding_id,
                BRAIN_KG_EDGE_CONNECTS_TO, "contains memory encoding", 1.0f, admin_token);
            local_state.edge_count++;
        }
    }

    /* Cross edges */
    if (local_config.register_cross_edges) {
        if (local_state.auditory_processing_id != BRAIN_KG_INVALID_NODE &&
            local_state.object_recognition_id != BRAIN_KG_INVALID_NODE) {
            create_temporal_edge(kg, local_state.auditory_processing_id,
                local_state.object_recognition_id,
                BRAIN_KG_EDGE_INTEGRATES_WITH,
                "auditory cues support object recognition", 0.7f, admin_token);
            local_state.edge_count++;
        }
        if (local_state.object_recognition_id != BRAIN_KG_INVALID_NODE &&
            local_state.face_processing_id != BRAIN_KG_INVALID_NODE) {
            create_temporal_edge(kg, local_state.object_recognition_id,
                local_state.face_processing_id,
                (brain_kg_edge_type_t)TEMPORAL_KG_EDGE_RECOGNIZES,
                "object recognition specializes into face processing", 0.85f, admin_token);
            local_state.edge_count++;
        }
        if (local_state.object_recognition_id != BRAIN_KG_INVALID_NODE &&
            local_state.memory_encoding_id != BRAIN_KG_INVALID_NODE) {
            create_temporal_edge(kg, local_state.object_recognition_id,
                local_state.memory_encoding_id,
                (brain_kg_edge_type_t)TEMPORAL_KG_EDGE_ENCODES,
                "recognized objects encoded into memory", 0.8f, admin_token);
            local_state.edge_count++;
        }
        if (local_state.face_processing_id != BRAIN_KG_INVALID_NODE &&
            local_state.memory_encoding_id != BRAIN_KG_INVALID_NODE) {
            create_temporal_edge(kg, local_state.face_processing_id,
                local_state.memory_encoding_id,
                (brain_kg_edge_type_t)TEMPORAL_KG_EDGE_ENCODES,
                "face identities encoded into memory", 0.85f, admin_token);
            local_state.edge_count++;
        }
    }

    local_state.registered = true;
    if (state) *state = local_state;

    NIMCP_LOG_INFO(TEMPORAL_KG_MODULE_NAME,
        "Registered %u nodes, %u edges",
        local_state.node_count, local_state.edge_count);

    return 0;
}

//=============================================================================
// Query API
//=============================================================================

brain_kg_node_id_t temporal_kg_get_root(brain_kg_t* kg) {
    if (!kg) return BRAIN_KG_INVALID_NODE;
    return brain_kg_find_node(kg, TEMPORAL_KG_ROOT_NAME);
}

brain_kg_node_id_t temporal_kg_find_subsystem(brain_kg_t* kg, const char* name) {
    if (!kg || !name) return BRAIN_KG_INVALID_NODE;
    return brain_kg_find_node(kg, name);
}

int temporal_kg_unregister_all(
    brain_kg_t* kg,
    temporal_kg_state_t* state,
    uint64_t admin_token
) {
    (void)admin_token;
    if (!kg || !state) return -1;

    state->registered = false;
    state->node_count = 0;
    state->edge_count = 0;

    NIMCP_LOG_INFO(TEMPORAL_KG_MODULE_NAME, "Unregistered Temporal KG nodes");
    return 0;
}
