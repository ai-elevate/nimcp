#include <stddef.h>  /* for NULL */
#include "security/nimcp_bbb_helpers.h"

//=============================================================================
// Health Agent Integration (Phase 8: System-Wide Health Integration)
//=============================================================================
struct nimcp_health_agent;
typedef struct nimcp_health_agent nimcp_health_agent_t;
extern void nimcp_health_agent_heartbeat_ex(nimcp_health_agent_t* agent,
                                             const char* operation,
                                             float progress);

/** Global health agent for stdp_quantum_bridge module */
static nimcp_health_agent_t* g_stdp_quantum_bridge_health_agent = NULL;

/**
 * @brief Set health agent for stdp_quantum_bridge heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
static void stdp_quantum_bridge_set_health_agent(nimcp_health_agent_t* agent) {
    g_stdp_quantum_bridge_health_agent = agent;
}

/** @brief Send heartbeat from stdp_quantum_bridge module */
static inline void stdp_quantum_bridge_heartbeat(const char* operation, float progress) {
    if (g_stdp_quantum_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_stdp_quantum_bridge_health_agent, operation, progress);
    }
}

/* Security integration */
BRIDGE_DEFINE_SECURITY_SETTERS(stdp_quantum_bridge)

#define LOG_MODULE "STDP_QUANTUM_BRIDGE"

//=============================================================================
// STDP Quantum Bridge Implementation
//=============================================================================

/*
 * Define NIMCP_STDP_QUANTUM_BRIDGE_IMPLEMENTATION to pull in the struct
 * definition and all function bodies from the header.
 */
#define NIMCP_STDP_QUANTUM_BRIDGE_IMPLEMENTATION
#include "plasticity/stdp/nimcp_stdp_quantum_bridge.h"
#include "utils/logging/nimcp_logging.h"
