/**
 * @file nimcp_platform_rwlock.c
 * @brief Cross-platform read-write lock implementation (SRP: RWLock operations only)
 *
 * WHAT: Platform-agnostic read-write lock implementation
 * WHY:  pthread_rwlock doesn't exist on Windows
 * HOW:  Wrap pthread_rwlock (POSIX) and SRWLOCK (Windows Vista+)
 *
 * SRP: This module has ONE responsibility - read-write lock operations
 * COMPLEXITY: O(1) for all operations (excluding wait time)
 * THREAD-SAFE: Yes - all functions thread-safe
 *
 * PLATFORM NOTES:
 * POSIX (Linux, macOS, BSD):
 * - Uses pthread_rwlock_t and pthread_rwlock_* functions
 * - Provides both blocking and non-blocking variants
 * - Writer starvation possible depending on platform scheduler
 *
 * Windows (Vista+):
 * - Uses SRWLOCK (Slim Read/Write Lock)
 * - Lightweight, kernel-based synchronization
 * - Readers share lock, writers are exclusive
 * - TryAcquireSRWLockShared/Exclusive available on Windows 7+
 * - Different functions for releasing shared vs exclusive locks
 *
 * @author NIMCP Development Team
 * @date 2025
 */

#include "nimcp_platform_rwlock.h"
#include <errno.h>

/* ========================================================================
 * READ-WRITE LOCK FUNCTIONS
 * ======================================================================== */

int nimcp_platform_rwlock_init(nimcp_platform_rwlock_t* rwlock)
{
    if (!rwlock) {
        return EINVAL;
    }

#if defined(NIMCP_PLATFORM_POSIX)
    /* POSIX: Initialize pthread_rwlock_t */
    return pthread_rwlock_init(rwlock, NULL);

#elif defined(NIMCP_PLATFORM_WINDOWS)
    /* Windows: SRWLOCK is initialized to 0 (no-op needed) */
    /* InitializeSRWLock macro sets it to 0, but we do it explicitly for clarity */
    *rwlock = SRWLOCK_INIT;
    return 0;

#else
    #error "Unsupported platform"
#endif
}

int nimcp_platform_rwlock_destroy(nimcp_platform_rwlock_t* rwlock)
{
    if (!rwlock) {
        return EINVAL;
    }

#if defined(NIMCP_PLATFORM_POSIX)
    /* POSIX: Destroy pthread_rwlock_t */
    return pthread_rwlock_destroy(rwlock);

#elif defined(NIMCP_PLATFORM_WINDOWS)
    /* Windows: SRWLOCK doesn't require explicit cleanup */
    /* No-op for Windows - SRWLOCK cleanup is implicit */
    return 0;

#else
    #error "Unsupported platform"
#endif
}

int nimcp_platform_rwlock_rdlock(nimcp_platform_rwlock_t* rwlock)
{
    if (!rwlock) {
        return EINVAL;
    }

#if defined(NIMCP_PLATFORM_POSIX)
    /* POSIX: Acquire read lock (shared) */
    return pthread_rwlock_rdlock(rwlock);

#elif defined(NIMCP_PLATFORM_WINDOWS)
    /* Windows: Acquire shared lock */
    AcquireSRWLockShared(rwlock);
    return 0;

#else
    #error "Unsupported platform"
#endif
}

int nimcp_platform_rwlock_wrlock(nimcp_platform_rwlock_t* rwlock)
{
    if (!rwlock) {
        return EINVAL;
    }

#if defined(NIMCP_PLATFORM_POSIX)
    /* POSIX: Acquire write lock (exclusive) */
    return pthread_rwlock_wrlock(rwlock);

#elif defined(NIMCP_PLATFORM_WINDOWS)
    /* Windows: Acquire exclusive lock */
    AcquireSRWLockExclusive(rwlock);
    return 0;

#else
    #error "Unsupported platform"
#endif
}

int nimcp_platform_rwlock_tryrdlock(nimcp_platform_rwlock_t* rwlock)
{
    if (!rwlock) {
        return EINVAL;
    }

#if defined(NIMCP_PLATFORM_POSIX)
    /* POSIX: Try to acquire read lock (non-blocking) */
    return pthread_rwlock_tryrdlock(rwlock);

#elif defined(NIMCP_PLATFORM_WINDOWS)
    /* Windows: Try to acquire shared lock (requires Windows 7+) */
    /* TryAcquireSRWLockShared returns TRUE if acquired, FALSE otherwise */
    BOOLEAN result = TryAcquireSRWLockShared(rwlock);
    return result ? 0 : EBUSY;

#else
    #error "Unsupported platform"
#endif
}

int nimcp_platform_rwlock_trywrlock(nimcp_platform_rwlock_t* rwlock)
{
    if (!rwlock) {
        return EINVAL;
    }

#if defined(NIMCP_PLATFORM_POSIX)
    /* POSIX: Try to acquire write lock (non-blocking) */
    return pthread_rwlock_trywrlock(rwlock);

#elif defined(NIMCP_PLATFORM_WINDOWS)
    /* Windows: Try to acquire exclusive lock (requires Windows 7+) */
    /* TryAcquireSRWLockExclusive returns TRUE if acquired, FALSE otherwise */
    BOOLEAN result = TryAcquireSRWLockExclusive(rwlock);
    return result ? 0 : EBUSY;

#else
    #error "Unsupported platform"
#endif
}

int nimcp_platform_rwlock_rdunlock(nimcp_platform_rwlock_t* rwlock)
{
    if (!rwlock) {
        return EINVAL;
    }

#if defined(NIMCP_PLATFORM_POSIX)
    /* POSIX: Release lock (same function for read/write) */
    return pthread_rwlock_unlock(rwlock);

#elif defined(NIMCP_PLATFORM_WINDOWS)
    /* Windows: Release shared lock (must use correct release function) */
    ReleaseSRWLockShared(rwlock);
    return 0;

#else
    #error "Unsupported platform"
#endif
}

int nimcp_platform_rwlock_wrunlock(nimcp_platform_rwlock_t* rwlock)
{
    if (!rwlock) {
        return EINVAL;
    }

#if defined(NIMCP_PLATFORM_POSIX)
    /* POSIX: Release lock (same function for read/write) */
    return pthread_rwlock_unlock(rwlock);

#elif defined(NIMCP_PLATFORM_WINDOWS)
    /* Windows: Release exclusive lock (must use correct release function) */
    ReleaseSRWLockExclusive(rwlock);
    return 0;

#else
    #error "Unsupported platform"
#endif
}
