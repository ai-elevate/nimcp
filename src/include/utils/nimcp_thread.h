//=============================================================================
// nimcp_thread.h - NIMCP Thread Utilities
//=============================================================================

#ifndef NIMCP_THREAD_H
#define NIMCP_THREAD_H

#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <time.h>

/**
 * @file nimcp_thread.h
 * @brief Professional pthread wrapper with error handling and resource management
 *
 * Features:
 * - Consistent error handling with thread-local error messages
 * - Named resource locks (string-based mutex acquisition)
 * - Thread attributes (stack size, priority, detached)
 * - Mutex types (normal, recursive, errorcheck)
 * - Condition variables with timeout
 * - Reference-counted resource locks
 */

//=============================================================================
// Result Codes
//=============================================================================

typedef int32_t nimcp_result_t;

#define NIMCP_SUCCESS                0    // Operation successful
#define NIMCP_ERROR_INVALID_PARAM   -1    // Invalid parameter
#define NIMCP_ERROR_SYSTEM          -2    // System error (check errno)
#define NIMCP_ERROR_MEMORY          -3    // Memory allocation failed
#define NIMCP_ERROR_NOT_FOUND       -4    // Resource not found
#define NIMCP_BUSY                  -5    // Resource busy (trylock)

//=============================================================================
// Type Definitions
//=============================================================================

typedef pthread_t nimcp_thread_t;
typedef pthread_mutex_t nimcp_mutex_t;
typedef pthread_cond_t nimcp_cond_t;
typedef pthread_once_t nimcp_once_t;

// Thread error information
typedef struct {
    int error_code;
    char error_message[256];
} nimcp_thread_error_t;

// Thread attributes
typedef struct {
    size_t stack_size;
    int priority;
    bool detached;
} thread_attr_t;

// Mutex attributes
typedef enum {
    MUTEX_TYPE_NORMAL,
    MUTEX_TYPE_RECURSIVE,
    MUTEX_TYPE_ERRORCHECK
} mutex_type_t;

typedef struct {
    mutex_type_t type;
} mutex_attr_t;

// Resource lock entry (for named locks)
typedef struct resource_entry {
    char* resource_id;
    nimcp_mutex_t* lock;
    size_t ref_count;
    struct resource_entry* next;
} resource_entry_t;

// Resource lock bucket
typedef struct {
    nimcp_mutex_t bucket_mutex;
    resource_entry_t* entries;
} resource_bucket_t;

//=============================================================================
// Constants
//=============================================================================

#define NIMCP_ONCE_INIT PTHREAD_ONCE_INIT
#define NIMCP_THREAD_DEFAULT_STACK_SIZE (2 * 1024 * 1024)
#define RESOURCE_LOCK_BUCKETS 256

// Resource lock table
typedef struct {
    nimcp_mutex_t global_mutex;
    resource_bucket_t buckets[RESOURCE_LOCK_BUCKETS];
    bool initialized;
} resource_lock_table_t;

//=============================================================================
// Thread Management
//=============================================================================

/**
 * @brief Initialize thread subsystem
 * Call once before using thread utilities
 */
nimcp_result_t nimcp_thread_init(void);

/**
 * @brief Create a new thread
 * @param thread Output thread handle
 * @param start_routine Thread function
 * @param arg Argument to thread function
 * @param attr Thread attributes (NULL for defaults)
 */
nimcp_result_t nimcp_thread_create(nimcp_thread_t* thread,
                                   void* (*start_routine)(void*),
                                   void* arg,
                                   const thread_attr_t* attr);

/**
 * @brief Wait for thread to finish
 */
nimcp_result_t nimcp_thread_join(nimcp_thread_t thread, void** retval);

/**
 * @brief Detach thread (auto-cleanup on exit)
 */
nimcp_result_t nimcp_thread_detach(nimcp_thread_t thread);

/**
 * @brief Exit current thread
 */
void nimcp_thread_exit(void* retval) __attribute__((noreturn));

/**
 * @brief Get current thread ID
 */
nimcp_thread_t nimcp_thread_self(void);

/**
 * @brief Compare thread IDs
 */
bool nimcp_thread_equal(nimcp_thread_t t1, nimcp_thread_t t2);

/**
 * @brief Execute function exactly once (thread-safe initialization)
 */
nimcp_result_t nimcp_once(nimcp_once_t* once_control, void (*init_routine)(void));

//=============================================================================
// Mutex Operations
//=============================================================================

/**
 * @brief Initialize mutex
 */
nimcp_result_t nimcp_mutex_init(nimcp_mutex_t* mutex, const mutex_attr_t* attr);

/**
 * @brief Destroy mutex
 */
nimcp_result_t nimcp_mutex_destroy(nimcp_mutex_t* mutex);

/**
 * @brief Lock mutex (blocking)
 */
nimcp_result_t nimcp_mutex_lock(nimcp_mutex_t* mutex);

/**
 * @brief Try to lock mutex (non-blocking)
 * @return NIMCP_SUCCESS if locked, NIMCP_BUSY if already locked
 */
nimcp_result_t nimcp_mutex_trylock(nimcp_mutex_t* mutex);

/**
 * @brief Unlock mutex
 */
nimcp_result_t nimcp_mutex_unlock(nimcp_mutex_t* mutex);

//=============================================================================
// Condition Variables
//=============================================================================

/**
 * @brief Initialize condition variable
 */
nimcp_result_t nimcp_cond_init(nimcp_cond_t* cond);

/**
 * @brief Destroy condition variable
 */
nimcp_result_t nimcp_cond_destroy(nimcp_cond_t* cond);

/**
 * @brief Wait on condition variable (releases mutex)
 */
nimcp_result_t nimcp_cond_wait(nimcp_cond_t* cond, nimcp_mutex_t* mutex);

/**
 * @brief Wait on condition with timeout (milliseconds)
 */
nimcp_result_t nimcp_cond_timedwait(nimcp_cond_t* cond, nimcp_mutex_t* mutex,
                                     uint32_t timeout_ms);

/**
 * @brief Signal one waiting thread
 */
nimcp_result_t nimcp_cond_signal(nimcp_cond_t* cond);

/**
 * @brief Signal all waiting threads
 */
nimcp_result_t nimcp_cond_broadcast(nimcp_cond_t* cond);

//=============================================================================
// Named Resource Locks
//=============================================================================

/**
 * @brief Get or create a mutex by string ID
 *
 * Example: nimcp_get_resource_lock("brain_network", &mutex);
 *
 * Reference counted - automatically freed when all holders release
 */
nimcp_result_t nimcp_get_resource_lock(const char* resource_id,
                                       nimcp_mutex_t** mutex);

/**
 * @brief Release a resource lock (decrement refcount)
 */
nimcp_result_t nimcp_release_resource_lock(const char* resource_id);

//=============================================================================
// Error Handling
//=============================================================================

/**
 * @brief Get last error message for this thread
 */
const char* nimcp_thread_get_error(void);

/**
 * @brief Clear error state
 */
void nimcp_thread_clear_error(void);

//=============================================================================
// Cleanup
//=============================================================================

/**
 * @brief Cleanup thread subsystem
 * Call on shutdown
 */
void nimcp_thread_cleanup(void);

#endif // NIMCP_THREAD_H
