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
// Health Agent Integration (Phase 8: System-Wide Health Integration)
//=============================================================================
struct nimcp_health_agent;
typedef struct nimcp_health_agent nimcp_health_agent_t;
extern void nimcp_health_agent_heartbeat_ex(nimcp_health_agent_t* agent,
                                             const char* operation,
                                             float progress);

/** Global health agent for hypergraph_kg_bridge module */
static nimcp_health_agent_t* g_hypergraph_kg_bridge_health_agent = NULL;

/**
 * @brief Set health agent for hypergraph_kg_bridge heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
void hypergraph_kg_bridge_set_health_agent(nimcp_health_agent_t* agent) {
    g_hypergraph_kg_bridge_health_agent = agent;
}

/** @brief Send heartbeat from hypergraph_kg_bridge module */
static inline void hypergraph_kg_bridge_heartbeat(const char* operation, float progress) {
    if (g_hypergraph_kg_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_hypergraph_kg_bridge_health_agent, operation, progress);
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
