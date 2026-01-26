/**
 * @file nimcp_temporal_substrate_bridge.c
 * @brief Temporal Cortex-Neural Substrate Bridge Implementation
 */

#include "utils/bridge/nimcp_bridge_base.h"
#include "core/temporal/nimcp_temporal_substrate_bridge.h"
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

/** Global health agent for temporal_substrate_bridge module */
static nimcp_health_agent_t* g_temporal_substrate_bridge_health_agent = NULL;

/**
 * @brief Set health agent for temporal_substrate_bridge heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
static void temporal_substrate_bridge_set_health_agent(nimcp_health_agent_t* agent) {
    g_temporal_substrate_bridge_health_agent = agent;
}

/** @brief Send heartbeat from temporal_substrate_bridge module */
static inline void temporal_substrate_bridge_heartbeat(const char* operation, float progress) {
    if (g_temporal_substrate_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_temporal_substrate_bridge_health_agent, operation, progress);
    }
}

//=============================================================================
