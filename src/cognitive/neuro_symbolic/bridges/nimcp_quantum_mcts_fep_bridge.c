/**
 * @file nimcp_quantum_mcts_fep_bridge.c
 * @brief Quantum MCTS - FEP Bridge Implementation
 */

#include "cognitive/neuro_symbolic/bridges/nimcp_quantum_mcts_fep_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <math.h>
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

/** Global health agent for quantum_mcts_fep_bridge module */
static nimcp_health_agent_t* g_quantum_mcts_fep_bridge_health_agent = NULL;

/**
 * @brief Set health agent for quantum_mcts_fep_bridge heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
void quantum_mcts_fep_bridge_set_health_agent(nimcp_health_agent_t* agent) {
    g_quantum_mcts_fep_bridge_health_agent = agent;
}

/** @brief Send heartbeat from quantum_mcts_fep_bridge module */
static inline void quantum_mcts_fep_bridge_heartbeat(const char* operation, float progress) {
    if (g_quantum_mcts_fep_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_quantum_mcts_fep_bridge_health_agent, operation, progress);
    }
}


NIMCP_API qmcts_fep_bridge_t* qmcts_fep_bridge_create(void) {
    qmcts_fep_bridge_t* bridge = nimcp_calloc(1, sizeof(qmcts_fep_bridge_t));
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return NULL;

    }

    bridge_base_init(&bridge->base, BIO_MODULE_QMCTS_FEP_BRIDGE,
                     "qmcts_fep_bridge");

    bridge->quantum_exploration_boost = 1.5f;

    return bridge;
}

NIMCP_API void qmcts_fep_bridge_destroy(qmcts_fep_bridge_t* bridge) {
    if (!bridge) return;
    bridge_base_cleanup(&bridge->base);
    nimcp_free(bridge);
}

NIMCP_API float qmcts_fep_bridge_expected_value(
    qmcts_fep_bridge_t* bridge, const float* state, uint32_t dim) {

    if (!bridge || !state || dim == 0) return 0.0f;

    /* Compute expected value based on state energy */
    float sum = 0.0f;
    for (uint32_t i = 0; i < dim; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && dim > 256) {
            quantum_mcts_fep_bridge_heartbeat("quantum_mcts_loop",
                             (float)(i + 1) / (float)dim);
        }

        sum += state[i] * state[i];
    }

    /* Lower energy = higher expected value */
    float energy = sqrtf(sum);
    float value = 1.0f / (1.0f + energy);

    bridge->plans_executed++;

    return value * bridge->quantum_exploration_boost;
}
