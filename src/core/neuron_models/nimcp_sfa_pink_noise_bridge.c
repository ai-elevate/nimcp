#include <stddef.h>  /* for NULL */

//=============================================================================
// Health Agent Integration (Phase 8: System-Wide Health Integration)
//=============================================================================
struct nimcp_health_agent;
typedef struct nimcp_health_agent nimcp_health_agent_t;
extern void nimcp_health_agent_heartbeat_ex(nimcp_health_agent_t* agent,
                                             const char* operation,
                                             float progress);

/** Global health agent for sfa_pink_noise_bridge module */
static nimcp_health_agent_t* g_sfa_pink_noise_bridge_health_agent = NULL;

/**
 * @brief Set health agent for sfa_pink_noise_bridge heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
static void sfa_pink_noise_bridge_set_health_agent(nimcp_health_agent_t* agent) {
    g_sfa_pink_noise_bridge_health_agent = agent;
}

/** @brief Send heartbeat from sfa_pink_noise_bridge module */
static inline void sfa_pink_noise_bridge_heartbeat(const char* operation, float progress) {
    if (g_sfa_pink_noise_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_sfa_pink_noise_bridge_health_agent, operation, progress);
    }
}

//=============================================================================
