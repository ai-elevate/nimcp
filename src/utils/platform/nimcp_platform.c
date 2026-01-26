/**
 * @file nimcp_platform.c
 * @brief Cross-platform utility implementations
 *
 * WHAT: Platform detection and utility functions
 * WHY:  Provide runtime information about platform/compiler/architecture
 * HOW:  Return compile-time constants
 *
 * @author NIMCP Development Team
 * @date 2025
 */

#include "utils/platform/nimcp_platform.h"
#include "security/nimcp_security.h"
#include "security/nimcp_blood_brain_barrier.h"

#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_messages.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>
#include "utils/memory/nimcp_unified_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "api/nimcp_api_exception.h"

#include <stddef.h>  /* for NULL */

//=============================================================================
// Health Agent Integration (Phase 8: System-Wide Health Integration)
//=============================================================================
struct nimcp_health_agent;
typedef struct nimcp_health_agent nimcp_health_agent_t;
extern void nimcp_health_agent_heartbeat_ex(nimcp_health_agent_t* agent,
                                             const char* operation,
                                             float progress);

/** Global health agent for platform module */
static nimcp_health_agent_t* g_platform_health_agent = NULL;

/**
 * @brief Set health agent for platform heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
static void platform_set_health_agent(nimcp_health_agent_t* agent) {
    g_platform_health_agent = agent;
}

/** @brief Send heartbeat from platform module */
static inline void platform_heartbeat(const char* operation, float progress) {
    if (g_platform_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_platform_health_agent, operation, progress);
    }
}

//=============================================================================
