/**
 * @file nimcp_evolutionary_proof_logic_bridge.c
 * @brief Evolutionary Proof - Logic Bridge Implementation
 */

#include "cognitive/neuro_symbolic/bridges/nimcp_evolutionary_proof_logic_bridge.h"
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

/** Global health agent for evolutionary_proof_logic_bridge module */
static nimcp_health_agent_t* g_evolutionary_proof_logic_bridge_health_agent = NULL;

/**
 * @brief Set health agent for evolutionary_proof_logic_bridge heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
static void evolutionary_proof_logic_bridge_set_health_agent(nimcp_health_agent_t* agent) {
    g_evolutionary_proof_logic_bridge_health_agent = agent;
}

/** @brief Send heartbeat from evolutionary_proof_logic_bridge module */
static inline void evolutionary_proof_logic_bridge_heartbeat(const char* operation, float progress) {
    if (g_evolutionary_proof_logic_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_evolutionary_proof_logic_bridge_health_agent, operation, progress);
    }
}


NIMCP_API evoproof_logic_bridge_t* evoproof_logic_bridge_create(void) {
    evoproof_logic_bridge_t* bridge = nimcp_calloc(1, sizeof(evoproof_logic_bridge_t));
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return NULL;

    }

    bridge_base_init(&bridge->base, BIO_MODULE_EVOPROOF_LOGIC_BRIDGE,
                     "evoproof_logic_bridge");

    bridge->enable_axiom_expansion = true;
    bridge->enable_lemma_caching = true;

    return bridge;
}

NIMCP_API void evoproof_logic_bridge_destroy(evoproof_logic_bridge_t* bridge) {
    if (!bridge) return;
    bridge_base_cleanup(&bridge->base);
    nimcp_free(bridge);
}
