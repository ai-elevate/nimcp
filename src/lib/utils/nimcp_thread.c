//=============================================================================
// nimcp_thread.c - NIMCP Thread Utilities Implementation
//=============================================================================

#include "../../include/utils/nimcp_thread.h"
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <stdarg.h>

//=============================================================================
// Thread-Local Error Storage
//=============================================================================

static __thread nimcp_thread_error_t thread_error = {0};

//=============================================================================
// Resource Lock Table
//=============================================================================

static resource_lock_table_t resource_table = {0};
static nimcp_once_t init_once = NIMCP_ONCE_INIT;

//=============================================================================
// Error Handling
//=============================================================================

static void set_thread_error(int error_code, const char* format, ...) {
    va_list args;
    va_start(args, format);
    thread_error.error_code = error_code;
    vsnprintf(thread_error.error_message, sizeof(thread_error.error_message),
              format, args);
    va_end(args);
}

const char* nimcp_thread_get_error(void) {
    return thread_error.error_message;
}

void nimcp_thread_clear_error(void) {
    thread_error.error_code = 0;
    thread_error.error_message[0] = '\0';
}

//=============================================================================
// Initialization
//=============================================================================

static void thread_init_routine(void) {
    pthread_mutex_init(&resource_table.global_mutex, NULL);

    for (int i = 0; i < RESOURCE_LOCK_BUCKETS; i++) {
        pthread_mutex_init(&resource_table.buckets[i].bucket_mutex, NULL);
        resource_table.buckets[i].entries = NULL;
    }

    resource_table.initialized = true;
}

nimcp_result_t nimcp_thread_init(void) {
    return pthread_once(&init_once, thread_init_routine) == 0 ?
           NIMCP_SUCCESS : NIMCP_ERROR_SYSTEM;
}

//=============================================================================
// Thread Management
//=============================================================================

nimcp_result_t nimcp_thread_create(nimcp_thread_t* thread,
                                   void* (*start_routine)(void*),
                                   void* arg,
                                   const thread_attr_t* attr) {
    if (!thread || !start_routine) {
        set_thread_error(NIMCP_ERROR_INVALID_PARAM, "Invalid thread parameters");
        return NIMCP_ERROR_INVALID_PARAM;
    }

    pthread_attr_t pthread_attr;
    pthread_attr_init(&pthread_attr);

    if (attr) {
        if (attr->stack_size > 0) {
            pthread_attr_setstacksize(&pthread_attr, attr->stack_size);
        }

        if (attr->detached) {
            pthread_attr_setdetachstate(&pthread_attr, PTHREAD_CREATE_DETACHED);
        }

        #ifdef _POSIX_PRIORITY_SCHEDULING
        if (attr->priority > 0) {
            struct sched_param param;
            param.sched_priority = attr->priority;
            pthread_attr_setschedparam(&pthread_attr, &param);
        }
        #endif
    }

    int result = pthread_create(thread, &pthread_attr, start_routine, arg);
    pthread_attr_destroy(&pthread_attr);

    if (result != 0) {
        set_thread_error(NIMCP_ERROR_SYSTEM, "Thread creation failed: %s",
                        strerror(result));
        return NIMCP_ERROR_SYSTEM;
    }

    return NIMCP_SUCCESS;
}

nimcp_result_t nimcp_thread_join(nimcp_thread_t thread, void** retval) {
    int result = pthread_join(thread, retval);
    if (result != 0) {
        set_thread_error(NIMCP_ERROR_SYSTEM, "Thread join failed: %s",
                        strerror(result));
        return NIMCP_ERROR_SYSTEM;
    }
    return NIMCP_SUCCESS;
}

nimcp_result_t nimcp_thread_detach(nimcp_thread_t thread) {
    int result = pthread_detach(thread);
    if (result != 0) {
        set_thread_error(NIMCP_ERROR_SYSTEM, "Thread detach failed: %s",
                        strerror(result));
        return NIMCP_ERROR_SYSTEM;
    }
    return NIMCP_SUCCESS;
}

void nimcp_thread_exit(void* retval) {
    pthread_exit(retval);
}

nimcp_thread_t nimcp_thread_self(void) {
    return pthread_self();
}

bool nimcp_thread_equal(nimcp_thread_t t1, nimcp_thread_t t2) {
    return pthread_equal(t1, t2);
}

nimcp_result_t nimcp_once(nimcp_once_t* once_control, void (*init_routine)(void)) {
    if (!once_control || !init_routine) {
        set_thread_error(NIMCP_ERROR_INVALID_PARAM, "Invalid once parameters");
        return NIMCP_ERROR_INVALID_PARAM;
    }

    int result = pthread_once(once_control, init_routine);
    if (result != 0) {
        set_thread_error(NIMCP_ERROR_SYSTEM, "pthread_once failed: %s",
                        strerror(result));
        return NIMCP_ERROR_SYSTEM;
    }

    return NIMCP_SUCCESS;
}

//=============================================================================
// Mutex Operations
//=============================================================================

nimcp_result_t nimcp_mutex_init(nimcp_mutex_t* mutex, const mutex_attr_t* attr) {
    if (!mutex) {
        set_thread_error(NIMCP_ERROR_INVALID_PARAM, "Invalid mutex pointer");
        return NIMCP_ERROR_INVALID_PARAM;
    }

    pthread_mutexattr_t mutex_attr;
    pthread_mutexattr_init(&mutex_attr);

    if (attr) {
        int type;
        switch (attr->type) {
            case MUTEX_TYPE_RECURSIVE:
                type = PTHREAD_MUTEX_RECURSIVE;
                break;
            case MUTEX_TYPE_ERRORCHECK:
                type = PTHREAD_MUTEX_ERRORCHECK;
                break;
            default:
                type = PTHREAD_MUTEX_NORMAL;
        }
        pthread_mutexattr_settype(&mutex_attr, type);
    }

    int result = pthread_mutex_init(mutex, &mutex_attr);
    pthread_mutexattr_destroy(&mutex_attr);

    if (result != 0) {
        set_thread_error(NIMCP_ERROR_SYSTEM, "Mutex initialization failed: %s",
                        strerror(result));
        return NIMCP_ERROR_SYSTEM;
    }

    return NIMCP_SUCCESS;
}

nimcp_result_t nimcp_mutex_destroy(nimcp_mutex_t* mutex) {
    if (!mutex) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    int result = pthread_mutex_destroy(mutex);
    if (result != 0) {
        set_thread_error(NIMCP_ERROR_SYSTEM, "Mutex destruction failed: %s",
                        strerror(result));
        return NIMCP_ERROR_SYSTEM;
    }

    return NIMCP_SUCCESS;
}

nimcp_result_t nimcp_mutex_lock(nimcp_mutex_t* mutex) {
    if (!mutex) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    int result = pthread_mutex_lock(mutex);
    if (result != 0) {
        set_thread_error(NIMCP_ERROR_SYSTEM, "Mutex lock failed: %s",
                        strerror(result));
        return NIMCP_ERROR_SYSTEM;
    }

    return NIMCP_SUCCESS;
}

nimcp_result_t nimcp_mutex_trylock(nimcp_mutex_t* mutex) {
    if (!mutex) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    int result = pthread_mutex_trylock(mutex);
    if (result == EBUSY) {
        return NIMCP_BUSY;
    } else if (result != 0) {
        set_thread_error(NIMCP_ERROR_SYSTEM, "Mutex trylock failed: %s",
                        strerror(result));
        return NIMCP_ERROR_SYSTEM;
    }

    return NIMCP_SUCCESS;
}

nimcp_result_t nimcp_mutex_unlock(nimcp_mutex_t* mutex) {
    if (!mutex) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    int result = pthread_mutex_unlock(mutex);
    if (result != 0) {
        set_thread_error(NIMCP_ERROR_SYSTEM, "Mutex unlock failed: %s",
                        strerror(result));
        return NIMCP_ERROR_SYSTEM;
    }

    return NIMCP_SUCCESS;
}

//=============================================================================
// Condition Variables
//=============================================================================

nimcp_result_t nimcp_cond_init(nimcp_cond_t* cond) {
    if (!cond) {
        set_thread_error(NIMCP_ERROR_INVALID_PARAM, "Invalid cond pointer");
        return NIMCP_ERROR_INVALID_PARAM;
    }

    int result = pthread_cond_init(cond, NULL);
    if (result != 0) {
        set_thread_error(NIMCP_ERROR_SYSTEM, "Cond initialization failed: %s",
                        strerror(result));
        return NIMCP_ERROR_SYSTEM;
    }

    return NIMCP_SUCCESS;
}

nimcp_result_t nimcp_cond_destroy(nimcp_cond_t* cond) {
    if (!cond) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    int result = pthread_cond_destroy(cond);
    if (result != 0) {
        set_thread_error(NIMCP_ERROR_SYSTEM, "Cond destruction failed: %s",
                        strerror(result));
        return NIMCP_ERROR_SYSTEM;
    }

    return NIMCP_SUCCESS;
}

nimcp_result_t nimcp_cond_wait(nimcp_cond_t* cond, nimcp_mutex_t* mutex) {
    if (!cond || !mutex) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    int result = pthread_cond_wait(cond, mutex);
    if (result != 0) {
        set_thread_error(NIMCP_ERROR_SYSTEM, "Cond wait failed: %s",
                        strerror(result));
        return NIMCP_ERROR_SYSTEM;
    }

    return NIMCP_SUCCESS;
}

nimcp_result_t nimcp_cond_timedwait(nimcp_cond_t* cond, nimcp_mutex_t* mutex,
                                     uint32_t timeout_ms) {
    if (!cond || !mutex) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);

    /* Add timeout */
    ts.tv_sec += timeout_ms / 1000;
    ts.tv_nsec += (timeout_ms % 1000) * 1000000;

    /* Handle nanosecond overflow */
    if (ts.tv_nsec >= 1000000000) {
        ts.tv_sec += 1;
        ts.tv_nsec -= 1000000000;
    }

    int result = pthread_cond_timedwait(cond, mutex, &ts);
    if (result == ETIMEDOUT) {
        return NIMCP_BUSY;  /* Timeout = resource busy */
    } else if (result != 0) {
        set_thread_error(NIMCP_ERROR_SYSTEM, "Cond timedwait failed: %s",
                        strerror(result));
        return NIMCP_ERROR_SYSTEM;
    }

    return NIMCP_SUCCESS;
}

nimcp_result_t nimcp_cond_signal(nimcp_cond_t* cond) {
    if (!cond) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    int result = pthread_cond_signal(cond);
    if (result != 0) {
        set_thread_error(NIMCP_ERROR_SYSTEM, "Cond signal failed: %s",
                        strerror(result));
        return NIMCP_ERROR_SYSTEM;
    }

    return NIMCP_SUCCESS;
}

nimcp_result_t nimcp_cond_broadcast(nimcp_cond_t* cond) {
    if (!cond) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    int result = pthread_cond_broadcast(cond);
    if (result != 0) {
        set_thread_error(NIMCP_ERROR_SYSTEM, "Cond broadcast failed: %s",
                        strerror(result));
        return NIMCP_ERROR_SYSTEM;
    }

    return NIMCP_SUCCESS;
}

//=============================================================================
// Named Resource Locks
//=============================================================================

static unsigned int hash_string(const char* str) {
    unsigned int hash = 5381;
    int c;
    while ((c = *str++)) {
        hash = ((hash << 5) + hash) + c;
    }
    return hash % RESOURCE_LOCK_BUCKETS;
}

nimcp_result_t nimcp_get_resource_lock(const char* resource_id,
                                       nimcp_mutex_t** mutex) {
    if (!resource_id || !mutex) {
        set_thread_error(NIMCP_ERROR_INVALID_PARAM, "Invalid resource lock parameters");
        return NIMCP_ERROR_INVALID_PARAM;
    }

    if (!resource_table.initialized) {
        nimcp_thread_init();
    }

    unsigned int bucket = hash_string(resource_id);
    pthread_mutex_lock(&resource_table.buckets[bucket].bucket_mutex);

    resource_entry_t* entry = resource_table.buckets[bucket].entries;
    while (entry) {
        if (strcmp(entry->resource_id, resource_id) == 0) {
            entry->ref_count++;
            *mutex = entry->lock;
            pthread_mutex_unlock(&resource_table.buckets[bucket].bucket_mutex);
            return NIMCP_SUCCESS;
        }
        entry = entry->next;
    }

    /* WHAT: Resource doesn't exist, create new entry */
    entry = malloc(sizeof(resource_entry_t));
    if (!entry) {
        pthread_mutex_unlock(&resource_table.buckets[bucket].bucket_mutex);
        set_thread_error(NIMCP_ERROR_MEMORY, "Failed to allocate resource entry");
        return NIMCP_ERROR_MEMORY;
    }

    entry->resource_id = strdup(resource_id);
    entry->lock = malloc(sizeof(nimcp_mutex_t));
    if (!entry->resource_id || !entry->lock) {
        free(entry->resource_id);
        free(entry->lock);
        free(entry);
        pthread_mutex_unlock(&resource_table.buckets[bucket].bucket_mutex);
        set_thread_error(NIMCP_ERROR_MEMORY, "Failed to allocate resource lock");
        return NIMCP_ERROR_MEMORY;
    }

    if (pthread_mutex_init(entry->lock, NULL) != 0) {
        free(entry->resource_id);
        free(entry->lock);
        free(entry);
        pthread_mutex_unlock(&resource_table.buckets[bucket].bucket_mutex);
        set_thread_error(NIMCP_ERROR_SYSTEM, "Failed to initialize resource mutex");
        return NIMCP_ERROR_SYSTEM;
    }

    entry->ref_count = 1;
    entry->next = resource_table.buckets[bucket].entries;
    resource_table.buckets[bucket].entries = entry;
    *mutex = entry->lock;

    pthread_mutex_unlock(&resource_table.buckets[bucket].bucket_mutex);
    return NIMCP_SUCCESS;
}

nimcp_result_t nimcp_release_resource_lock(const char* resource_id) {
    if (!resource_id) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    unsigned int bucket = hash_string(resource_id);
    pthread_mutex_lock(&resource_table.buckets[bucket].bucket_mutex);

    resource_entry_t* entry = resource_table.buckets[bucket].entries;
    resource_entry_t* prev = NULL;

    while (entry) {
        if (strcmp(entry->resource_id, resource_id) == 0) {
            entry->ref_count--;
            if (entry->ref_count == 0) {
                /* WHAT: Last reference, free the resource */
                if (prev) {
                    prev->next = entry->next;
                } else {
                    resource_table.buckets[bucket].entries = entry->next;
                }
                pthread_mutex_destroy(entry->lock);
                free(entry->lock);
                free(entry->resource_id);
                free(entry);
            }
            pthread_mutex_unlock(&resource_table.buckets[bucket].bucket_mutex);
            return NIMCP_SUCCESS;
        }
        prev = entry;
        entry = entry->next;
    }

    pthread_mutex_unlock(&resource_table.buckets[bucket].bucket_mutex);
    return NIMCP_ERROR_NOT_FOUND;
}

//=============================================================================
// Cleanup
//=============================================================================

void nimcp_thread_cleanup(void) {
    if (!resource_table.initialized) {
        return;
    }

    pthread_mutex_lock(&resource_table.global_mutex);

    for (int i = 0; i < RESOURCE_LOCK_BUCKETS; i++) {
        pthread_mutex_lock(&resource_table.buckets[i].bucket_mutex);

        resource_entry_t* entry = resource_table.buckets[i].entries;
        while (entry) {
            resource_entry_t* next = entry->next;
            pthread_mutex_destroy(entry->lock);
            free(entry->lock);
            free(entry->resource_id);
            free(entry);
            entry = next;
        }

        pthread_mutex_unlock(&resource_table.buckets[i].bucket_mutex);
        pthread_mutex_destroy(&resource_table.buckets[i].bucket_mutex);
    }

    pthread_mutex_unlock(&resource_table.global_mutex);
    pthread_mutex_destroy(&resource_table.global_mutex);
    resource_table.initialized = false;
}
