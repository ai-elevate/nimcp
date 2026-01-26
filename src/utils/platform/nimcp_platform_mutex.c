/**
 * @file nimcp_platform_mutex.c
 * @brief Cross-platform mutex implementation (SRP: Mutex operations only)
 *
 * WHAT: Platform-agnostic mutex implementation
 * WHY:  pthread_mutex doesn't exist on Windows
 * HOW:  Wrap pthread_mutex (POSIX) and CRITICAL_SECTION (Windows)
 *
 * SRP: This module has ONE responsibility - mutex operations
 * COMPLEXITY: O(1) for all operations
 * THREAD-SAFE: Yes - all functions thread-safe
 *
 * @author NIMCP Development Team
 * @date 2025
 */

#include "utils/platform/nimcp_platform_mutex.h"
#include "security/nimcp_security.h"
#include "security/nimcp_blood_brain_barrier.h"

#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_messages.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <errno.h>
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

/** Global health agent for platform_mutex module */
static nimcp_health_agent_t* g_platform_mutex_health_agent = NULL;

/**
 * @brief Set health agent for platform_mutex heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
static void platform_mutex_set_health_agent(nimcp_health_agent_t* agent) {
    g_platform_mutex_health_agent = agent;
}

/** @brief Send heartbeat from platform_mutex module */
static inline void platform_mutex_heartbeat(const char* operation, float progress) {
    if (g_platform_mutex_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_platform_mutex_health_agent, operation, progress);
    }
}


/* ========================================================================
 * MUTEX FUNCTIONS
 * ======================================================================== */

int nimcp_platform_mutex_init(nimcp_platform_mutex_t* mutex, bool recursive)
{
    if (!mutex) {
        LOG_ERROR("nimcp_platform_mutex_init: mutex pointer is NULL");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "Mutex init failed: NULL pointer");
        return EINVAL;
    }

#if defined(NIMCP_PLATFORM_POSIX)
    /* POSIX: pthread_mutex with optional recursive attribute */
    if (recursive) {
        pthread_mutexattr_t attr;
        pthread_mutexattr_init(&attr);
        pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
        int result = pthread_mutex_init(mutex, &attr);
        pthread_mutexattr_destroy(&attr);
        return result;
    } else {
        return pthread_mutex_init(mutex, NULL);
    }

#elif defined(NIMCP_PLATFORM_WINDOWS)
    /* Windows: CRITICAL_SECTION (always recursive) */
    InitializeCriticalSection(mutex);
    return 0;

#else
    #error "Unsupported platform"
#endif
}

int nimcp_platform_mutex_destroy(nimcp_platform_mutex_t* mutex)
{
    if (!mutex) {
        LOG_ERROR("nimcp_platform_mutex_destroy: mutex pointer is NULL");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "Mutex destroy failed: NULL pointer");
        return EINVAL;
    }

#if defined(NIMCP_PLATFORM_POSIX)
    return pthread_mutex_destroy(mutex);

#elif defined(NIMCP_PLATFORM_WINDOWS)
    DeleteCriticalSection(mutex);
    return 0;

#else
    #error "Unsupported platform"
#endif
}

int nimcp_platform_mutex_lock(nimcp_platform_mutex_t* mutex)
{
    if (!mutex) {
        LOG_ERROR("nimcp_platform_mutex_lock: mutex pointer is NULL");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "Mutex lock failed: NULL pointer");
        return EINVAL;
    }

#if defined(NIMCP_PLATFORM_POSIX)
    return pthread_mutex_lock(mutex);

#elif defined(NIMCP_PLATFORM_WINDOWS)
    EnterCriticalSection(mutex);
    return 0;

#else
    #error "Unsupported platform"
#endif
}

int nimcp_platform_mutex_trylock(nimcp_platform_mutex_t* mutex)
{
    if (!mutex) {
        LOG_ERROR("nimcp_platform_mutex_trylock: mutex pointer is NULL");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "Mutex trylock failed: NULL pointer");
        return EINVAL;
    }

#if defined(NIMCP_PLATFORM_POSIX)
    return pthread_mutex_trylock(mutex);

#elif defined(NIMCP_PLATFORM_WINDOWS)
    BOOL result = TryEnterCriticalSection(mutex);
    return result ? 0 : EBUSY;

#else
    #error "Unsupported platform"
#endif
}

int nimcp_platform_mutex_unlock(nimcp_platform_mutex_t* mutex)
{
    if (!mutex) {
        LOG_ERROR("nimcp_platform_mutex_unlock: mutex pointer is NULL");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "Mutex unlock failed: NULL pointer");
        return EINVAL;
    }

#if defined(NIMCP_PLATFORM_POSIX)
    return pthread_mutex_unlock(mutex);

#elif defined(NIMCP_PLATFORM_WINDOWS)
    LeaveCriticalSection(mutex);
    return 0;

#else
    #error "Unsupported platform"
#endif
}

nimcp_platform_mutex_t* nimcp_platform_mutex_create(void)
{
    nimcp_platform_mutex_t* mutex = (nimcp_platform_mutex_t*)nimcp_malloc(
        sizeof(nimcp_platform_mutex_t)
    );
    NIMCP_API_CHECK_ALLOC(mutex, "Failed to allocate mutex structure");

    if (nimcp_platform_mutex_init(mutex, false) != 0) {
        LOG_ERROR("nimcp_platform_mutex_create: failed to initialize mutex");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_MUTEX_INIT, "Mutex create failed: init error");
        nimcp_free(mutex);
        return NULL;
    }

    return mutex;
}
