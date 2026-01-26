/**
 * @file nimcp_neural_logic_quantum_bridge.c
 * @brief Implementation of quantum-accelerated neural logic bridge
 *
 * WHAT: Quantum SAT solving for neural logic circuits
 * WHY:  √N speedup for satisfiability search
 * HOW:  Grover's algorithm on circuit CNF encoding
 *
 * @author NIMCP Development Team
 * @date 2025-01-20
 */

#define NIMCP_NEURAL_LOGIC_QUANTUM_BRIDGE_IMPLEMENTATION
#include "core/logic/nimcp_neural_logic_quantum_bridge.h"
#include "utils/exception/nimcp_exception_macros.h"

#include <stddef.h>  /* for NULL */
#include <string.h>  /* for memcpy */

//=============================================================================
// Opaque wrapper for neural_logic.c (avoids header dependency)
//=============================================================================

/**
 * @brief Opaque wrapper for neural_logic_quantum_default_config()
 * @param config_out Pointer to neural_logic_quantum_config_t buffer
 *
 * Called by nimcp_neural_logic.c which forward-declares this as
 * extern void neural_logic_quantum_default_config_opaque(void* config_out);
 */
void neural_logic_quantum_default_config_opaque(void* config_out) {
    if (!config_out) return;
    neural_logic_quantum_config_t cfg = neural_logic_quantum_default_config();
    *(neural_logic_quantum_config_t*)config_out = cfg;
}

//=============================================================================
// Health Agent Integration (Phase 8: System-Wide Health Integration)
//=============================================================================
struct nimcp_health_agent;
typedef struct nimcp_health_agent nimcp_health_agent_t;
extern void nimcp_health_agent_heartbeat_ex(nimcp_health_agent_t* agent,
                                             const char* operation,
                                             float progress);

/** Global health agent for neural_logic_quantum_bridge module */
static nimcp_health_agent_t* g_neural_logic_quantum_bridge_health_agent = NULL;

/**
 * @brief Set health agent for neural_logic_quantum_bridge heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
static void neural_logic_quantum_bridge_set_health_agent(nimcp_health_agent_t* agent) {
    g_neural_logic_quantum_bridge_health_agent = agent;
}

/** @brief Send heartbeat from neural_logic_quantum_bridge module */
static inline void neural_logic_quantum_bridge_heartbeat(const char* operation, float progress) {
    if (g_neural_logic_quantum_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_neural_logic_quantum_bridge_health_agent, operation, progress);
    }
}

//=============================================================================
