/**
 * @file nimcp_theory_of_mind_substrate_bridge.c
 * @brief Theory of Mind-Neural Substrate Bridge - Health Agent Integration
 *
 * NOTE: The actual bridge implementation lives in cognitive/tom/nimcp_tom_substrate_bridge.c.
 * This file provides only the health agent setter for the theory_of_mind_substrate_bridge
 * module identity, used by Phase 8 system-wide health monitoring.
 */

#include <stddef.h>  /* for NULL */
#include "utils/logging/nimcp_logging.h"

//=============================================================================
// Health Agent Integration (Phase 8: System-Wide Health Integration)
//=============================================================================
struct nimcp_health_agent;
typedef struct nimcp_health_agent nimcp_health_agent_t;
extern void nimcp_health_agent_heartbeat_ex(nimcp_health_agent_t* agent,
                                             const char* operation,
                                             float progress);

/** Global health agent for theory_of_mind_substrate_bridge module */
static nimcp_health_agent_t* g_theory_of_mind_substrate_bridge_health_agent = NULL;

/**
 * @brief Set health agent for theory_of_mind_substrate_bridge heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
void theory_of_mind_substrate_bridge_set_health_agent(nimcp_health_agent_t* agent) {
    g_theory_of_mind_substrate_bridge_health_agent = agent;
}

/** @brief Send heartbeat from theory_of_mind_substrate_bridge module */
static inline void theory_of_mind_substrate_bridge_heartbeat(const char* operation, float progress) {
    if (g_theory_of_mind_substrate_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_theory_of_mind_substrate_bridge_health_agent, operation, progress);
    }
}

#define LOG_MODULE "THEORY_OF_MIND_SUBSTRATE_BRIDGE"
