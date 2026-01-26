//=============================================================================
// nimcp_brain_state.c - Brain State Accessors and COW Handling
//=============================================================================
/**
 * @file nimcp_brain_state.c
 * @brief Brain state accessors and copy-on-write network handling
 *
 * This module contains approximately 1200 lines extracted from nimcp_brain.c:
 * - brain_get_network() - Network accessor
 * - brain_get_neuromodulator_system() - Neuromodulator accessor
 * - brain_get_sleep_system() - Sleep system accessor
 * - brain_get_theory_of_mind() - Theory of mind accessor
 * - brain_get_explanation_generator() - Explanation generator accessor
 * - ensure_writable_network() - COW network cloning
 *
 * @version 1.0.0
 * @date 2025-12-08
 */

#include "core/brain/nimcp_brain_state.h"
#include "core/brain/nimcp_brain.h"
#include "core/brain/nimcp_brain_internal.h"
#include "api/nimcp_api_exception.h"
#include "utils/logging/nimcp_logging.h"
#include "plasticity/neuromodulators/nimcp_neuromodulators.h"
#include "cognitive/nimcp_sleep_wake.h"
#include "cognitive/nimcp_theory_of_mind.h"
#include "cognitive/nimcp_explanations.h"
#include "utils/exception/nimcp_exception_macros.h"

#define LOG_MODULE "BRAIN_STATE"

//=============================================================================
#include <stddef.h>  /* for NULL */
// Health Agent Integration (Phase 8: System-Wide Health Integration)
//=============================================================================
struct nimcp_health_agent;
typedef struct nimcp_health_agent nimcp_health_agent_t;
extern void nimcp_health_agent_heartbeat_ex(nimcp_health_agent_t* agent,
                                             const char* operation,
                                             float progress);

/** Global health agent for brain_state module */
static nimcp_health_agent_t* g_brain_state_health_agent = NULL;

/**
 * @brief Set health agent for brain_state heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
static void brain_state_set_health_agent(nimcp_health_agent_t* agent) {
    g_brain_state_health_agent = agent;
}

/** @brief Send heartbeat from brain_state module */
static inline void brain_state_heartbeat(const char* operation, float progress) {
    if (g_brain_state_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_brain_state_health_agent, operation, progress);
    }
}


// NOTE: Implementation functions are currently in nimcp_brain.c
// External declarations for linking
extern adaptive_network_t brain_get_network(brain_t brain);
extern neuromodulator_system_t brain_get_neuromodulator_system(brain_t brain);
extern sleep_system_t brain_get_sleep_system(brain_t brain);
extern theory_of_mind_t brain_get_theory_of_mind(brain_t brain);
extern explanation_generator_t brain_get_explanation_generator(brain_t brain);
extern bool ensure_writable_network(brain_t brain);
