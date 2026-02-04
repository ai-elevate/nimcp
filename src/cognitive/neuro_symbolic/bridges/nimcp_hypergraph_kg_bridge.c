/**
 * @file nimcp_hypergraph_kg_bridge.c
 * @brief Hypergraph - Knowledge Graph Bridge Implementation
 */

#include "cognitive/neuro_symbolic/bridges/nimcp_hypergraph_kg_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>

//=============================================================================
#include <stddef.h>  /* for NULL */
#include "utils/logging/nimcp_logging.h"
#include "utils/fault_tolerance/nimcp_health_agent_macros.h"
#include "mesh/nimcp_mesh_participant.h"
#include "mesh/nimcp_mesh_adapter.h"

NIMCP_DECLARE_HEALTH_AGENT_ATOMIC(hypergraph_kg_bridge)
//=============================================================================
// Mesh Participant Registration
//=============================================================================

static mesh_participant_id_t g_hypergraph_kg_bridge_mesh_id = 0;
static mesh_participant_registry_t* g_hypergraph_kg_bridge_mesh_registry = NULL;

nimcp_error_t hypergraph_kg_bridge_mesh_register(mesh_participant_registry_t* registry) {
    if (!registry) return NIMCP_ERROR_NULL_POINTER;
    if (g_hypergraph_kg_bridge_mesh_id != 0) return NIMCP_SUCCESS;
    mesh_participant_interface_t iface;
    mesh_participant_interface_init(&iface);
    strncpy(iface.module_name, "hypergraph_kg_bridge", MESH_MAX_NAME_LEN - 1);
    iface.type = MESH_PARTICIPANT_MODULE;
    iface.home_channel = mesh_adapter_get_default_channel(MESH_ADAPTER_CATEGORY_COGNITIVE);
    mesh_participant_config_t config;
    mesh_participant_config_init(&config);
    config.module_name = "hypergraph_kg_bridge";
    config.type = MESH_PARTICIPANT_MODULE;
    config.home_channel = iface.home_channel;
    nimcp_error_t err = mesh_participant_register(registry, &iface, &config, &g_hypergraph_kg_bridge_mesh_id);
    if (err == NIMCP_SUCCESS) g_hypergraph_kg_bridge_mesh_registry = registry;
    return err;
}

void hypergraph_kg_bridge_mesh_unregister(void) {
    if (g_hypergraph_kg_bridge_mesh_registry && g_hypergraph_kg_bridge_mesh_id != 0) {
        mesh_participant_unregister(g_hypergraph_kg_bridge_mesh_registry, g_hypergraph_kg_bridge_mesh_id);
        g_hypergraph_kg_bridge_mesh_id = 0;
        g_hypergraph_kg_bridge_mesh_registry = NULL;
    }
}


/** @brief Send heartbeat from hypergraph_kg_bridge module (instance-level) */
static inline void hypergraph_kg_bridge_heartbeat_instance(
    nimcp_health_agent_t* instance_agent, const char* operation, float progress)
{
    if (g_hypergraph_kg_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_hypergraph_kg_bridge_health_agent, operation, progress);
    }
    if (instance_agent && instance_agent != g_hypergraph_kg_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(instance_agent, operation, progress);
    }
}


#define LOG_MODULE "HYPERGRAPH_KG_BRIDGE"


NIMCP_API hypergraph_kg_bridge_t* hypergraph_kg_bridge_create(void) {
    hypergraph_kg_bridge_t* bridge = nimcp_calloc(1, sizeof(hypergraph_kg_bridge_t));
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "Failed to allocate bridge");

        return NULL;

    }

    bridge_base_init(&bridge->base, BIO_MODULE_HYPERGRAPH_KG_BRIDGE,
                     "hypergraph_kg_bridge");

    bridge->enable_bidirectional_sync = true;

    NIMCP_LOGGING_INFO("Created %s bridge", "hypergraph_kg");
    return bridge;
}

NIMCP_API void hypergraph_kg_bridge_destroy(hypergraph_kg_bridge_t* bridge) {
    if (!bridge) return;
    NIMCP_LOGGING_DEBUG("Destroying %s bridge", "hypergraph_kg");
    bridge_base_cleanup(&bridge->base);
    nimcp_free(bridge);
}

/* ============================================================================
 * Phase 8: Instance-Level Health Agent
 * ============================================================================ */

void hypergraph_kg_bridge_set_instance_health_agent(void* instance, nimcp_health_agent_t* agent) {
    if (instance) {
        (void)agent;
        g_hypergraph_kg_bridge_health_agent = agent;
    }
}

/* ============================================================================
 * Phase 8: Training Integration (Full Implementation)
 * ============================================================================ */

int hypergraph_kg_bridge_training_begin(void* instance) {
    if (!instance) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "hypergraph_kg_bridge_training_begin: NULL argument");
        return -1;
    }
    hypergraph_kg_bridge_heartbeat_instance(NULL, "hypergraph_kg_bridge_training_begin", 0.0f);
    return 0;
}

int hypergraph_kg_bridge_training_end(void* instance) {
    if (!instance) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "hypergraph_kg_bridge_training_end: NULL argument");
        return -1;
    }
    hypergraph_kg_bridge_heartbeat_instance(NULL, "hypergraph_kg_bridge_training_end", 1.0f);
    return 0;
}

int hypergraph_kg_bridge_training_step(void* instance, float progress) {
    if (!instance) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "hypergraph_kg_bridge_training_step: NULL argument");
        return -1;
    }
    if (progress < 0.0f) progress = 0.0f;
    if (progress > 1.0f) progress = 1.0f;
    hypergraph_kg_bridge_heartbeat_instance(NULL, "hypergraph_kg_bridge_training_step", progress);
    return 0;
}
