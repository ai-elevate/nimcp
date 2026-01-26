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


NIMCP_API hypergraph_kg_bridge_t* hypergraph_kg_bridge_create(void) {
    hypergraph_kg_bridge_t* bridge = nimcp_calloc(1, sizeof(hypergraph_kg_bridge_t));
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return NULL;

    }

    bridge_base_init(&bridge->base, BIO_MODULE_HYPERGRAPH_KG_BRIDGE,
                     "hypergraph_kg_bridge");

    bridge->enable_bidirectional_sync = true;

    return bridge;
}

NIMCP_API void hypergraph_kg_bridge_destroy(hypergraph_kg_bridge_t* bridge) {
    if (!bridge) return;
    bridge_base_cleanup(&bridge->base);
    nimcp_free(bridge);
}
