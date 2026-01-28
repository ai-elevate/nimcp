/**
 * @file nimcp_theory_of_mind_thalamic_bridge.c
 * @brief Theory of Mind-Thalamic Bridge - Health Agent Integration
 *
 * NOTE: The actual bridge implementation lives in cognitive/tom/nimcp_tom_thalamic_bridge.c.
 * This file provides only the health agent setter for the theory_of_mind_thalamic_bridge
 * module identity, used by Phase 8 system-wide health monitoring.
 */

#include <stddef.h>  /* for NULL */
#include "utils/logging/nimcp_logging.h"
#include "utils/exception/nimcp_exception_macros.h"


//=============================================================================
// Health Agent Integration (Phase 8: System-Wide Health Integration)
//=============================================================================
struct nimcp_health_agent;
typedef struct nimcp_health_agent nimcp_health_agent_t;
extern void nimcp_health_agent_heartbeat_ex(nimcp_health_agent_t* agent,
                                             const char* operation,
                                             float progress);

/** Global health agent for theory_of_mind_thalamic_bridge module */
static nimcp_health_agent_t* g_theory_of_mind_thalamic_bridge_health_agent = NULL;

/**
 * @brief Set health agent for theory_of_mind_thalamic_bridge heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
void theory_of_mind_thalamic_bridge_set_health_agent(nimcp_health_agent_t* agent) {
    g_theory_of_mind_thalamic_bridge_health_agent = agent;
}

/** @brief Send heartbeat from theory_of_mind_thalamic_bridge module */
static inline void theory_of_mind_thalamic_bridge_heartbeat(const char* operation, float progress) {
    if (g_theory_of_mind_thalamic_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_theory_of_mind_thalamic_bridge_health_agent, operation, progress);
    }
}

/** @brief Send heartbeat from theory_of_mind_thalamic_bridge module (instance-level) */
static inline void theory_of_mind_thalamic_bridge_heartbeat_instance(
    nimcp_health_agent_t* instance_agent, const char* operation, float progress)
{
    if (g_theory_of_mind_thalamic_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_theory_of_mind_thalamic_bridge_health_agent, operation, progress);
    }
    if (instance_agent && instance_agent != g_theory_of_mind_thalamic_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(instance_agent, operation, progress);
    }
}

#define LOG_MODULE "THEORY_OF_MIND_THALAMIC_BRIDGE"

/* ============================================================================
 * Phase 8: Instance-level health agent setter
 * ============================================================================ */
void theory_of_mind_thalamic_bridge_set_instance_health_agent(void* instance, nimcp_health_agent_t* agent) {
    if (instance) {
        (void)agent;
        g_theory_of_mind_thalamic_bridge_health_agent = agent;
    }
}

/* ============================================================================
 * Phase 8: Training stubs
 * ============================================================================ */
int theory_of_mind_thalamic_bridge_training_begin(void* instance) {
    if (!instance) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "theory_of_mind_thalamic_bridge_training_begin: NULL argument");
        return -1;
    }
    theory_of_mind_thalamic_bridge_heartbeat_instance(NULL, "theory_of_mind_thalamic_bridge_training_begin", 0.0f);
    return 0;
}

int theory_of_mind_thalamic_bridge_training_end(void* instance) {
    if (!instance) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "theory_of_mind_thalamic_bridge_training_end: NULL argument");
        return -1;
    }
    theory_of_mind_thalamic_bridge_heartbeat_instance(NULL, "theory_of_mind_thalamic_bridge_training_end", 1.0f);
    return 0;
}

int theory_of_mind_thalamic_bridge_training_step(void* instance, float progress) {
    if (!instance) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "theory_of_mind_thalamic_bridge_training_step: NULL argument");
        return -1;
    }
    theory_of_mind_thalamic_bridge_heartbeat_instance(NULL, "theory_of_mind_thalamic_bridge_training_step", progress);
    return 0;
}
