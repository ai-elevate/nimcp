//=============================================================================
// nimcp_brain_processing.c - Brain Processing and Decision Making
//=============================================================================
/**
 * @file nimcp_brain_processing.c
 * @brief Forward pass computation and decision making logic
 *
 * This module contains approximately 1400 lines extracted from nimcp_brain.c:
 * - perform_forward_pass() - Neural network forward propagation
 * - brain_decide() - Decision making and inference
 * - Helper functions for label determination and statistics
 *
 * @version 1.0.0
 * @date 2025-12-08
 */

#include "core/brain/nimcp_brain_processing.h"
#include "core/brain/nimcp_brain.h"
#include "core/brain/nimcp_brain_internal.h"
#include "api/nimcp_api_exception.h"
#include "utils/logging/nimcp_logging.h"
#include "core/neuralnet/nimcp_neuralnet.h"
#include "plasticity/adaptive/nimcp_adaptive.h"
#include "utils/exception/nimcp_exception_macros.h"

#define LOG_MODULE "BRAIN_PROC"

//=============================================================================
#include <stddef.h>  /* for NULL */
// Health Agent Integration (Phase 8: System-Wide Health Integration)
//=============================================================================
struct nimcp_health_agent;
typedef struct nimcp_health_agent nimcp_health_agent_t;
extern void nimcp_health_agent_heartbeat_ex(nimcp_health_agent_t* agent,
                                             const char* operation,
                                             float progress);

/** Global health agent for brain_processing module */
static nimcp_health_agent_t* g_brain_processing_health_agent = NULL;

/**
 * @brief Set health agent for brain_processing heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
static void brain_processing_set_health_agent(nimcp_health_agent_t* agent) {
    g_brain_processing_health_agent = agent;
}

/** @brief Send heartbeat from brain_processing module */
static inline void brain_processing_heartbeat(const char* operation, float progress) {
    if (g_brain_processing_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_brain_processing_health_agent, operation, progress);
    }
}


// NOTE: Implementation functions are currently in nimcp_brain.c
// This file provides modular organization and will be fully populated
// in a future migration phase.
