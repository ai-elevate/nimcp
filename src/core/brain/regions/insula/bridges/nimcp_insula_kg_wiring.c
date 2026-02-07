//=============================================================================
// nimcp_insula_kg_wiring.c - Insula Knowledge Graph Registration Implementation
//=============================================================================
/**
 * @file nimcp_insula_kg_wiring.c
 * @brief Implementation of Insula Knowledge Graph registration
 *
 * WHAT: Implements KG node/edge creation for Insula module
 * WHY:  Enables semantic queries about interoception and emotion-cognition
 * HOW:  Creates hierarchical node structure with typed relationships
 */

#include "core/brain/regions/insula/bridges/nimcp_insula_kg_wiring.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>
#include <stdio.h>
#include "utils/fault_tolerance/nimcp_health_agent_macros.h"
#include "mesh/nimcp_mesh_participant.h"
#include "mesh/nimcp_mesh_adapter.h"

NIMCP_DECLARE_HEALTH_AGENT_ATOMIC(insula_kg_wiring)

//=============================================================================
// Mesh Participant Registration
//=============================================================================

static mesh_participant_id_t g_insula_kg_wiring_mesh_id = 0;
static mesh_participant_registry_t* g_insula_kg_wiring_mesh_registry = NULL;

nimcp_error_t insula_kg_wiring_mesh_register(mesh_participant_registry_t* registry) {
    if (!registry) return NIMCP_ERROR_NULL_POINTER;
    if (g_insula_kg_wiring_mesh_id != 0) return NIMCP_SUCCESS;
    mesh_participant_interface_t iface;
    mesh_participant_interface_init(&iface);
    strncpy(iface.module_name, "insula_kg_wiring", MESH_MAX_NAME_LEN - 1);
    iface.type = MESH_PARTICIPANT_MODULE;
    iface.home_channel = mesh_adapter_get_default_channel(MESH_ADAPTER_CATEGORY_SYSTEM);
    mesh_participant_config_t config;
    mesh_participant_config_init(&config);
    config.module_name = "insula_kg_wiring";
    config.type = MESH_PARTICIPANT_MODULE;
    config.home_channel = iface.home_channel;
    nimcp_error_t err = mesh_participant_register(registry, &iface, &config, &g_insula_kg_wiring_mesh_id);
    if (err == NIMCP_SUCCESS) g_insula_kg_wiring_mesh_registry = registry;
    return err;
}

void insula_kg_wiring_mesh_unregister(void) {
    if (g_insula_kg_wiring_mesh_registry && g_insula_kg_wiring_mesh_id != 0) {
        mesh_participant_unregister(g_insula_kg_wiring_mesh_registry, g_insula_kg_wiring_mesh_id);
        g_insula_kg_wiring_mesh_id = 0;
        g_insula_kg_wiring_mesh_registry = NULL;
    }
}

//=============================================================================
// Helper Functions
//=============================================================================

static brain_kg_node_id_t create_insula_node(
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
        NIMCP_LOG_DEBUG(INSULA_KG_MODULE_NAME,
            "Created node '%s' (id=%u)", name, id);
    }
    return id;
}

static brain_kg_edge_id_t create_insula_edge(
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

int insula_kg_default_config(insula_kg_config_t* config) {
    if (!config) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "insula_kg_default_config: config is NULL");
        return -1;
    }

    config->register_anterior_insula = true;
    config->register_posterior_insula = true;
    config->register_interoception = true;
    config->register_pain_processing = true;
    config->register_emotion_awareness = true;
    config->register_cross_edges = true;
    config->include_state_metadata = true;

    return 0;
}

//=============================================================================
// Registration API
//=============================================================================

int insula_kg_register_all(
    brain_kg_t* kg,
    const insula_kg_config_t* config,
    insula_kg_state_t* state,
    uint64_t admin_token
) {
    if (!kg) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "insula_kg_register_all: kg is NULL");
        return -1;
    }

    insula_kg_config_t local_config;
    if (config) {
        local_config = *config;
    } else {
        insula_kg_default_config(&local_config);
    }

    insula_kg_state_t local_state;
    memset(&local_state, 0, sizeof(local_state));

    /* Create root node */
    local_state.root_id = create_insula_node(
        kg, INSULA_KG_ROOT_NAME,
        BRAIN_KG_NODE_CORTICAL,
        "Insula - interoception, emotion awareness, pain processing, self-awareness",
        admin_token
    );
    if (local_state.root_id == BRAIN_KG_INVALID_NODE) {
        NIMCP_LOG_ERROR(INSULA_KG_MODULE_NAME, "Failed to create root node");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "insula_kg_register_all: validation failed");
        return -1;
    }
    local_state.node_count++;

    /* Anterior insula */
    if (local_config.register_anterior_insula) {
        local_state.anterior_insula_id = create_insula_node(
            kg, "anterior_insula",
            (brain_kg_node_type_t)INSULA_KG_NODE_ANTERIOR,
            "Anterior insula - subjective feeling, awareness, salience detection",
            admin_token
        );
        if (local_state.anterior_insula_id != BRAIN_KG_INVALID_NODE) {
            local_state.node_count++;
            create_insula_edge(kg, local_state.root_id, local_state.anterior_insula_id,
                BRAIN_KG_EDGE_CONNECTS_TO, "contains anterior insula", 1.0f, admin_token);
            local_state.edge_count++;
        }
    }

    /* Posterior insula */
    if (local_config.register_posterior_insula) {
        local_state.posterior_insula_id = create_insula_node(
            kg, "posterior_insula",
            (brain_kg_node_type_t)INSULA_KG_NODE_POSTERIOR,
            "Posterior insula - primary interoceptive cortex, body-state representation",
            admin_token
        );
        if (local_state.posterior_insula_id != BRAIN_KG_INVALID_NODE) {
            local_state.node_count++;
            create_insula_edge(kg, local_state.root_id, local_state.posterior_insula_id,
                BRAIN_KG_EDGE_CONNECTS_TO, "contains posterior insula", 1.0f, admin_token);
            local_state.edge_count++;
        }
    }

    /* Interoception */
    if (local_config.register_interoception) {
        local_state.interoception_id = create_insula_node(
            kg, "interoception",
            (brain_kg_node_type_t)INSULA_KG_NODE_INTEROCEPTIVE,
            "Interoception - internal body state sensing, visceral awareness",
            admin_token
        );
        if (local_state.interoception_id != BRAIN_KG_INVALID_NODE) {
            local_state.node_count++;
            create_insula_edge(kg, local_state.root_id, local_state.interoception_id,
                BRAIN_KG_EDGE_CONNECTS_TO, "contains interoception", 1.0f, admin_token);
            local_state.edge_count++;
        }
    }

    /* Pain processing */
    if (local_config.register_pain_processing) {
        local_state.pain_processing_id = create_insula_node(
            kg, "pain_processing",
            (brain_kg_node_type_t)INSULA_KG_NODE_PAIN,
            "Pain processing - nociceptive integration, pain awareness",
            admin_token
        );
        if (local_state.pain_processing_id != BRAIN_KG_INVALID_NODE) {
            local_state.node_count++;
            create_insula_edge(kg, local_state.root_id, local_state.pain_processing_id,
                BRAIN_KG_EDGE_CONNECTS_TO, "contains pain processing", 1.0f, admin_token);
            local_state.edge_count++;
        }
    }

    /* Emotion awareness */
    if (local_config.register_emotion_awareness) {
        local_state.emotion_awareness_id = create_insula_node(
            kg, "emotion_awareness",
            (brain_kg_node_type_t)INSULA_KG_NODE_EMOTION_AWARENESS,
            "Emotion awareness - conscious feeling states, emotional self-awareness",
            admin_token
        );
        if (local_state.emotion_awareness_id != BRAIN_KG_INVALID_NODE) {
            local_state.node_count++;
            create_insula_edge(kg, local_state.root_id, local_state.emotion_awareness_id,
                BRAIN_KG_EDGE_CONNECTS_TO, "contains emotion awareness", 1.0f, admin_token);
            local_state.edge_count++;
        }
    }

    /* Cross edges */
    if (local_config.register_cross_edges) {
        /* Posterior insula senses body state -> interoception */
        if (local_state.posterior_insula_id != BRAIN_KG_INVALID_NODE &&
            local_state.interoception_id != BRAIN_KG_INVALID_NODE) {
            create_insula_edge(kg, local_state.posterior_insula_id,
                local_state.interoception_id,
                (brain_kg_edge_type_t)INSULA_KG_EDGE_SENSES,
                "posterior insula provides interoceptive signals", 0.9f, admin_token);
            local_state.edge_count++;
        }
        /* Interoception -> anterior insula for awareness */
        if (local_state.interoception_id != BRAIN_KG_INVALID_NODE &&
            local_state.anterior_insula_id != BRAIN_KG_INVALID_NODE) {
            create_insula_edge(kg, local_state.interoception_id,
                local_state.anterior_insula_id,
                BRAIN_KG_EDGE_SENDS_TO,
                "interoceptive signals reach conscious awareness", 0.85f, admin_token);
            local_state.edge_count++;
        }
        /* Anterior insula integrates emotion awareness */
        if (local_state.anterior_insula_id != BRAIN_KG_INVALID_NODE &&
            local_state.emotion_awareness_id != BRAIN_KG_INVALID_NODE) {
            create_insula_edge(kg, local_state.anterior_insula_id,
                local_state.emotion_awareness_id,
                (brain_kg_edge_type_t)INSULA_KG_EDGE_INTEGRATES,
                "anterior insula integrates emotional awareness", 0.85f, admin_token);
            local_state.edge_count++;
        }
        /* Pain processing -> posterior insula */
        if (local_state.pain_processing_id != BRAIN_KG_INVALID_NODE &&
            local_state.posterior_insula_id != BRAIN_KG_INVALID_NODE) {
            create_insula_edge(kg, local_state.pain_processing_id,
                local_state.posterior_insula_id,
                (brain_kg_edge_type_t)INSULA_KG_EDGE_SIGNALS,
                "pain signals processed by posterior insula", 0.9f, admin_token);
            local_state.edge_count++;
        }
    }

    local_state.registered = true;
    if (state) *state = local_state;

    NIMCP_LOG_INFO(INSULA_KG_MODULE_NAME,
        "Registered %u nodes, %u edges",
        local_state.node_count, local_state.edge_count);

    return 0;
}

//=============================================================================
// Query API
//=============================================================================

brain_kg_node_id_t insula_kg_get_root(brain_kg_t* kg) {
    if (!kg) return BRAIN_KG_INVALID_NODE;
    return brain_kg_find_node(kg, INSULA_KG_ROOT_NAME);
}

brain_kg_node_id_t insula_kg_find_subsystem(brain_kg_t* kg, const char* name) {
    if (!kg || !name) return BRAIN_KG_INVALID_NODE;
    return brain_kg_find_node(kg, name);
}

int insula_kg_unregister_all(
    brain_kg_t* kg,
    insula_kg_state_t* state,
    uint64_t admin_token
) {
    (void)admin_token;
    if (!kg || !state) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "insula_kg_unregister_all: required parameter is NULL (kg, state)");
        return -1;
    }

    state->registered = false;
    state->node_count = 0;
    state->edge_count = 0;

    NIMCP_LOG_INFO(INSULA_KG_MODULE_NAME, "Unregistered Insula KG nodes");
    return 0;
}
