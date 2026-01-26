#include <stddef.h>  /* for NULL */
//=============================================================================
// nimcp_pattern_cow.c - Copy-on-Write Pattern Data Implementation
//=============================================================================

#include "middleware/patterns/nimcp_pattern_cow.h"
#include "security/nimcp_security.h"
#include "security/nimcp_blood_brain_barrier.h"
#include "api/nimcp_api_exception.h"

#include "utils/memory/nimcp_memory.h"
#include <string.h>
#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/memory/nimcp_unified_memory.h"
#include "utils/exception/nimcp_exception_macros.h"



#define LOG_MODULE "nimcp_pattern_cow"
#define LOG_MODULE_ID 0x0525

//=============================================================================
// Health Agent Integration (Phase 8: System-Wide Health Integration)
//=============================================================================
struct nimcp_health_agent;
typedef struct nimcp_health_agent nimcp_health_agent_t;
extern void nimcp_health_agent_heartbeat_ex(nimcp_health_agent_t* agent,
                                             const char* operation,
                                             float progress);

/** Global health agent for pattern_cow module */
static nimcp_health_agent_t* g_pattern_cow_health_agent = NULL;

/**
 * @brief Set health agent for pattern_cow heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
static void pattern_cow_set_health_agent(nimcp_health_agent_t* agent) {
    g_pattern_cow_health_agent = agent;
}

/** @brief Send heartbeat from pattern_cow module */
static inline void pattern_cow_heartbeat(const char* operation, float progress) {
    if (g_pattern_cow_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_pattern_cow_health_agent, operation, progress);
    }
}


//=============================================================================
// CoW Pattern Data Lifecycle
//=============================================================================

pattern_cow_t* pattern_cow_create(const float* data, uint32_t dimension) {
    if (!data || dimension == 0) return NULL;

    // Allocate CoW wrapper
    pattern_cow_t* cow = (pattern_cow_t*)nimcp_malloc(sizeof(pattern_cow_t));
    if (!cow) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "cow is NULL");

        return NULL;

    }

    // Allocate pattern data
    cow->data = (float*)nimcp_malloc(dimension * sizeof(float));
    if (!cow->data) {
        nimcp_free(cow);
        return NULL;
    }

    // Copy pattern data
    memcpy(cow->data, data, dimension * sizeof(float));
    cow->dimension = dimension;

    // Initialize refcount to 1
    atomic_init(&cow->refcount, 1);

    return cow;
}

pattern_cow_t* pattern_cow_clone(pattern_cow_t* cow) {
    if (!cow) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "cow is NULL");

        return NULL;

    }

    // Atomically increment reference count
    atomic_fetch_add(&cow->refcount, 1);

    return cow;
}

void pattern_cow_release(pattern_cow_t* cow) {
    if (!cow) return;

    // Atomically decrement reference count
    uint32_t old_count = atomic_fetch_sub(&cow->refcount, 1);

    // If we were the last reference (old_count was 1, now 0), free resources
    if (old_count == 1) {
        nimcp_free(cow->data);
        nimcp_free(cow);
    }
}

const float* pattern_cow_data(const pattern_cow_t* cow) {
    if (!cow) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "cow is NULL");

        return NULL;

    }
    return cow->data;
}

uint32_t pattern_cow_dimension(const pattern_cow_t* cow) {
    if (!cow) return 0;
    return cow->dimension;
}

uint32_t pattern_cow_refcount(const pattern_cow_t* cow) {
    if (!cow) return 0;
    return atomic_load(&cow->refcount);
}
