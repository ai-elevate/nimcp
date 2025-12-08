//=============================================================================
// nimcp_thread_mutex.c - Mutex and Spinlock Operations
//=============================================================================

#include "utils/thread/nimcp_thread.h"
#include "security/nimcp_security.h"
#include "security/nimcp_blood_brain_barrier.h"
#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_messages.h"
#include "utils/error/nimcp_error_codes.h"
#include "utils/logging/nimcp_logging.h"

#include <errno.h>
#include <string.h>

#define LOG_MODULE "thread_mutex"

// External declarations for error handling (defined in nimcp_thread.c)
extern void set_thread_error(int error_code, const char* format, ...);

//=============================================================================
// Mutex Operations
//=============================================================================

/**
 * @brief Initialize mutex with attributes (Factory + Adapter pattern)
 *
 * WHY FACTORY PATTERN:
 * - Creates different mutex types based on attributes
 * - Encapsulates pthread attribute complexity
 * - Single creation point for all mutex types
 *
 * MUTEX TYPES:
 * - NORMAL (default): Fast, no recursion checking, UB on relock
 * - RECURSIVE: Can be locked multiple times by same thread
 * - ERRORCHECK: Returns error on recursive lock (debugging)
 *
 * COMPLEXITY: O(1)
 * THREAD SAFETY: Fully safe (creates new independent mutex)
 *
 * @param mutex Mutex to initialize
 * @param attr Mutex attributes (NULL for default NORMAL type)
 * @return NIMCP_SUCCESS or NIMCP_ERROR_INVALID_PARAM or NIMCP_ERROR_SYSTEM
 */
nimcp_result_t nimcp_mutex_init(nimcp_mutex_t* mutex, const mutex_attr_t* attr)
{
    if (!mutex) {
        set_thread_error(NIMCP_ERROR_INVALID_PARAM, "Invalid mutex pointer");
        return NIMCP_ERROR_INVALID_PARAM;
    }

    // Translate NIMCP mutex types to platform layer
    bool recursive = false;
    if (attr) {
        // Platform layer currently only supports normal and recursive mutexes
        // ERRORCHECK type falls back to normal
        switch (attr->type) {
            case MUTEX_TYPE_RECURSIVE:
                recursive = true;
                break;
            case MUTEX_TYPE_ERRORCHECK:
                // Fall back to normal for now (platform layer doesn't support errorcheck yet)
                recursive = false;
                break;
            default:
                recursive = false;
        }
    }

    // Initialize mutex using platform abstraction
    int result = nimcp_platform_mutex_init(mutex, recursive);

    if (result != 0) {
        set_thread_error(NIMCP_ERROR_SYSTEM, "Mutex initialization failed: %s", strerror(result));
        return NIMCP_ERROR_SYSTEM;
    }

    return NIMCP_SUCCESS;
}

/**
 * @brief Destroy mutex (Adapter for pthread_mutex_destroy)
 *
 * WHY DESTROY:
 * - Free kernel resources (futex on Linux)
 * - Clean shutdown
 * - Detect programming errors (destroying locked mutex fails)
 *
 * COMPLEXITY: O(1)
 * THREAD SAFETY: Caller must ensure no concurrent use
 *
 * @param mutex Mutex to destroy
 * @return NIMCP_SUCCESS or NIMCP_ERROR_INVALID_PARAM or NIMCP_ERROR_SYSTEM
 */
nimcp_result_t nimcp_mutex_destroy(nimcp_mutex_t* mutex)
{
    if (!mutex) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    int result = nimcp_platform_mutex_destroy(mutex);
    if (result != 0) {
        set_thread_error(NIMCP_ERROR_SYSTEM, "Mutex destruction failed: %s", strerror(result));
        return NIMCP_ERROR_SYSTEM;
    }

    return NIMCP_SUCCESS;
}

/**
 * @brief Lock mutex (Adapter for pthread_mutex_lock)
 *
 * WHY LOCK:
 * - Protect critical section (shared data access)
 * - Ensure mutual exclusion (only one thread in section)
 * - Serialize access to non-thread-safe operations
 *
 * COMPLEXITY: O(1) uncontended, O(n) if queued (n = number of waiters)
 * THREAD SAFETY: Fully safe (that's the point!)
 *
 * @param mutex Mutex to lock
 * @return NIMCP_SUCCESS or NIMCP_ERROR_INVALID_PARAM or NIMCP_ERROR_SYSTEM
 */
nimcp_result_t nimcp_mutex_lock(nimcp_mutex_t* mutex)
{
    if (!mutex) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    int result = nimcp_platform_mutex_lock(mutex);
    if (result != 0) {
        set_thread_error(NIMCP_ERROR_SYSTEM, "Mutex lock failed: %s", strerror(result));
        return NIMCP_ERROR_SYSTEM;
    }

    return NIMCP_SUCCESS;
}

/**
 * @brief Try to lock mutex without blocking (Adapter for pthread_mutex_trylock)
 *
 * WHY TRYLOCK:
 * - Non-blocking alternative to lock
 * - Avoid deadlock (can back off and retry)
 * - Polling pattern (try lock, do other work, retry)
 *
 * COMPLEXITY: O(1) (always non-blocking)
 * THREAD SAFETY: Fully safe
 *
 * @param mutex Mutex to try locking
 * @return NIMCP_SUCCESS if locked, NIMCP_BUSY if already locked, NIMCP_ERROR_* on error
 */
nimcp_result_t nimcp_mutex_trylock(nimcp_mutex_t* mutex)
{
    if (!mutex) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    int result = nimcp_platform_mutex_trylock(mutex);

    // EBUSY means mutex is locked (expected, not error)
    if (result == EBUSY) {
        return NIMCP_BUSY;
    } else if (result != 0) {
        set_thread_error(NIMCP_ERROR_SYSTEM, "Mutex trylock failed: %s", strerror(result));
        return NIMCP_ERROR_SYSTEM;
    }

    return NIMCP_SUCCESS;
}

/**
 * @brief Unlock mutex (Adapter for pthread_mutex_unlock)
 *
 * WHY UNLOCK:
 * - Release critical section
 * - Allow other threads to acquire lock
 * - Required pairing with lock (must unlock what you locked)
 *
 * COMPLEXITY: O(1)
 * THREAD SAFETY: Fully safe (must be called by locking thread)
 *
 * @param mutex Mutex to unlock
 * @return NIMCP_SUCCESS or NIMCP_ERROR_INVALID_PARAM or NIMCP_ERROR_SYSTEM
 */
nimcp_result_t nimcp_mutex_unlock(nimcp_mutex_t* mutex)
{
    if (!mutex) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    int result = nimcp_platform_mutex_unlock(mutex);
    if (result != 0) {
        set_thread_error(NIMCP_ERROR_SYSTEM, "Mutex unlock failed: %s", strerror(result));
        return NIMCP_ERROR_SYSTEM;
    }

    return NIMCP_SUCCESS;
}

//=============================================================================
// Spinlock Operations (implemented as mutexes for portability)
//=============================================================================

/**
 * @brief Initialize spinlock (Adapter for pthread_mutex_init)
 *
 * WHY MUTEX NOT SPINLOCK:
 * - pthread_spin_lock not universally available (optional POSIX feature)
 * - Mutex provides better portability across platforms
 * - For short critical sections, mutex performance is comparable
 * - Futex-based mutexes (Linux) already spin briefly before blocking
 *
 * COMPLEXITY: O(1)
 * THREAD SAFETY: Fully safe (creates new independent lock)
 *
 * @param lock Spinlock to initialize
 * @return NIMCP_SUCCESS or NIMCP_ERROR_INVALID_PARAM or NIMCP_ERROR_SYSTEM
 */
nimcp_result_t nimcp_spinlock_init(nimcp_spinlock_t* lock)
{
    if (!lock) {
        set_thread_error(NIMCP_ERROR_INVALID_PARAM, "Invalid spinlock pointer");
        return NIMCP_ERROR_INVALID_PARAM;
    }

    int result = nimcp_platform_mutex_init(lock, false);
    if (result != 0) {
        set_thread_error(NIMCP_ERROR_SYSTEM, "Spinlock initialization failed: %s", strerror(result));
        return NIMCP_ERROR_SYSTEM;
    }

    return NIMCP_SUCCESS;
}

/**
 * @brief Destroy spinlock (Adapter for pthread_mutex_destroy)
 *
 * WHY DESTROY:
 * - Free kernel resources (futex on Linux)
 * - Clean shutdown
 * - Detect programming errors (destroying locked spinlock fails)
 *
 * COMPLEXITY: O(1)
 * THREAD SAFETY: Caller must ensure no concurrent use
 *
 * @param lock Spinlock to destroy
 * @return NIMCP_SUCCESS or NIMCP_ERROR_INVALID_PARAM or NIMCP_ERROR_SYSTEM
 */
nimcp_result_t nimcp_spinlock_destroy(nimcp_spinlock_t* lock)
{
    if (!lock) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    int result = nimcp_platform_mutex_destroy(lock);
    if (result != 0) {
        set_thread_error(NIMCP_ERROR_SYSTEM, "Spinlock destruction failed: %s", strerror(result));
        return NIMCP_ERROR_SYSTEM;
    }

    return NIMCP_SUCCESS;
}

/**
 * @brief Lock spinlock (Adapter for pthread_mutex_lock)
 *
 * WHY LOCK:
 * - Protect critical section in glial cells (short duration)
 * - Ensure mutual exclusion for activity score updates
 * - Serialize access to monitored synapse arrays
 *
 * COMPLEXITY: O(1) uncontended
 * THREAD SAFETY: Fully safe
 *
 * @param lock Spinlock to lock
 * @return NIMCP_SUCCESS or NIMCP_ERROR_INVALID_PARAM or NIMCP_ERROR_SYSTEM
 */
nimcp_result_t nimcp_spinlock_lock(nimcp_spinlock_t* lock)
{
    if (!lock) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    int result = nimcp_platform_mutex_lock(lock);
    if (result != 0) {
        set_thread_error(NIMCP_ERROR_SYSTEM, "Spinlock lock failed: %s", strerror(result));
        return NIMCP_ERROR_SYSTEM;
    }

    return NIMCP_SUCCESS;
}

/**
 * @brief Unlock spinlock (Adapter for pthread_mutex_unlock)
 *
 * WHY UNLOCK:
 * - Release critical section
 * - Allow other threads to acquire lock
 * - Required pairing with lock
 *
 * COMPLEXITY: O(1)
 * THREAD SAFETY: Fully safe (must be called by locking thread)
 *
 * @param lock Spinlock to unlock
 * @return NIMCP_SUCCESS or NIMCP_ERROR_INVALID_PARAM or NIMCP_ERROR_SYSTEM
 */
nimcp_result_t nimcp_spinlock_unlock(nimcp_spinlock_t* lock)
{
    if (!lock) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    int result = nimcp_platform_mutex_unlock(lock);
    if (result != 0) {
        set_thread_error(NIMCP_ERROR_SYSTEM, "Spinlock unlock failed: %s", strerror(result));
        return NIMCP_ERROR_SYSTEM;
    }

    return NIMCP_SUCCESS;
}
