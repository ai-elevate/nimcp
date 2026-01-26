/**
 * @file nimcp_insula_substrate_bridge.c
 * @brief Insula-Neural Substrate Bridge Implementation
 */

#include "utils/bridge/nimcp_bridge_base.h"
#include "core/insula/nimcp_insula_substrate_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>

#include <stddef.h>  /* for NULL */

//=============================================================================
// Health Agent Integration (Phase 8: System-Wide Health Integration)
//=============================================================================
struct nimcp_health_agent;
typedef struct nimcp_health_agent nimcp_health_agent_t;
extern void nimcp_health_agent_heartbeat_ex(nimcp_health_agent_t* agent,
                                             const char* operation,
                                             float progress);

/** Global health agent for insula_substrate_bridge module */
static nimcp_health_agent_t* g_insula_substrate_bridge_health_agent = NULL;

/**
 * @brief Set health agent for insula_substrate_bridge heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
static void insula_substrate_bridge_set_health_agent(nimcp_health_agent_t* agent) {
    g_insula_substrate_bridge_health_agent = agent;
}

/** @brief Send heartbeat from insula_substrate_bridge module */
static inline void insula_substrate_bridge_heartbeat(const char* operation, float progress) {
    if (g_insula_substrate_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_insula_substrate_bridge_health_agent, operation, progress);
    }
}

//=============================================================================
