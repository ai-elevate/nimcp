//=============================================================================
// nimcp_thread_pool.c - Simple Thread Pool for Batch Processing
//=============================================================================

#include "utils/nimcp_thread_pool.h"
#include "utils/nimcp_memory.h"
#include <stdlib.h>
#include <string.h>

//=============================================================================
// Internal Structures
//=============================================================================

/**
 * @brief Task in the queue
 */
typedef struct {
    nimcp_task_fn func;
    void* arg;
} pool_task_t;

/**
 * @brief Thread pool implementation
 */
struct nimcp_thread_pool {
    /* WHAT: Worker threads */
    nimcp_thread_t threads[NIMCP_POOL_MAX_THREADS];
    size_t num_threads;

    /* WHAT: Circular task queue */
    pool_task_t queue[NIMCP_POOL_MAX_QUEUE];
    size_t queue_head;  /* Next task to dequeue */
    size_t queue_tail;  /* Next slot to enqueue */
    size_t queue_count; /* Number of tasks in queue */

    /* WHAT: Synchronization */
    nimcp_mutex_t lock;
    nimcp_cond_t task_available;  /* Signal when task added */
    nimcp_cond_t task_complete;   /* Signal when task done */

    /* WHAT: State tracking */
    size_t active_threads;  /* Threads currently executing tasks */
    bool shutdown;          /* Pool is shutting down */
};

//=============================================================================
// Worker Thread
//=============================================================================

/**
 * @brief Worker thread main loop
 */
static void* worker_thread(void* arg) {
    nimcp_thread_pool_t* pool = (nimcp_thread_pool_t*)arg;

    while (1) {
        nimcp_mutex_lock(&pool->lock);

        /* WHAT: Wait for task or shutdown */
        while (pool->queue_count == 0 && !pool->shutdown) {
            nimcp_cond_wait(&pool->task_available, &pool->lock);
        }

        /* WHAT: Check for shutdown */
        if (pool->shutdown && pool->queue_count == 0) {
            nimcp_mutex_unlock(&pool->lock);
            break;
        }

        /* WHAT: Dequeue task */
        pool_task_t task = pool->queue[pool->queue_head];
        pool->queue_head = (pool->queue_head + 1) % NIMCP_POOL_MAX_QUEUE;
        pool->queue_count--;
        pool->active_threads++;

        nimcp_mutex_unlock(&pool->lock);

        /* WHAT: Execute task outside lock */
        if (task.func) {
            task.func(task.arg);
        }

        /* WHAT: Mark completion */
        nimcp_mutex_lock(&pool->lock);
        pool->active_threads--;
        nimcp_cond_signal(&pool->task_complete);
        nimcp_mutex_unlock(&pool->lock);
    }

    return NULL;
}

//=============================================================================
// Public API
//=============================================================================

nimcp_thread_pool_t* nimcp_pool_create(size_t num_threads) {
    /* WHAT: Validate parameters */
    if (num_threads == 0 || num_threads > NIMCP_POOL_MAX_THREADS) {
        return NULL;
    }

    /* WHAT: Allocate pool */
    nimcp_thread_pool_t* pool = (nimcp_thread_pool_t*)nimcp_calloc(
        1, sizeof(nimcp_thread_pool_t));
    if (!pool) {
        return NULL;
    }

    pool->num_threads = num_threads;
    pool->queue_head = 0;
    pool->queue_tail = 0;
    pool->queue_count = 0;
    pool->active_threads = 0;
    pool->shutdown = false;

    /* WHAT: Initialize synchronization */
    if (nimcp_mutex_init(&pool->lock, NULL) != NIMCP_SUCCESS) {
        nimcp_free(pool);
        return NULL;
    }

    if (nimcp_cond_init(&pool->task_available) != NIMCP_SUCCESS) {
        nimcp_mutex_destroy(&pool->lock);
        nimcp_free(pool);
        return NULL;
    }

    if (nimcp_cond_init(&pool->task_complete) != NIMCP_SUCCESS) {
        nimcp_cond_destroy(&pool->task_available);
        nimcp_mutex_destroy(&pool->lock);
        nimcp_free(pool);
        return NULL;
    }

    /* WHAT: Create worker threads */
    for (size_t i = 0; i < num_threads; i++) {
        if (nimcp_thread_create(&pool->threads[i], worker_thread, pool, NULL)
            != NIMCP_SUCCESS) {
            /* WHAT: Cleanup on failure */
            pool->shutdown = true;
            nimcp_cond_broadcast(&pool->task_available);

            /* WHAT: Join already created threads */
            for (size_t j = 0; j < i; j++) {
                nimcp_thread_join(pool->threads[j], NULL);
            }

            nimcp_cond_destroy(&pool->task_complete);
            nimcp_cond_destroy(&pool->task_available);
            nimcp_mutex_destroy(&pool->lock);
            nimcp_free(pool);
            return NULL;
        }
    }

    return pool;
}

void nimcp_pool_destroy(nimcp_thread_pool_t* pool) {
    if (!pool) {
        return;
    }

    /* WHAT: Signal shutdown */
    nimcp_mutex_lock(&pool->lock);
    pool->shutdown = true;
    nimcp_cond_broadcast(&pool->task_available);
    nimcp_mutex_unlock(&pool->lock);

    /* WHAT: Wait for all threads to finish */
    for (size_t i = 0; i < pool->num_threads; i++) {
        nimcp_thread_join(pool->threads[i], NULL);
    }

    /* WHAT: Cleanup synchronization */
    nimcp_cond_destroy(&pool->task_complete);
    nimcp_cond_destroy(&pool->task_available);
    nimcp_mutex_destroy(&pool->lock);

    /* WHAT: Free pool */
    nimcp_free(pool);
}

nimcp_result_t nimcp_pool_submit(nimcp_thread_pool_t* pool,
                                  nimcp_task_fn task,
                                  void* arg) {
    if (!pool || !task) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    nimcp_mutex_lock(&pool->lock);

    /* WHAT: Check for shutdown */
    if (pool->shutdown) {
        nimcp_mutex_unlock(&pool->lock);
        return NIMCP_ERROR_SYSTEM;
    }

    /* WHAT: Wait if queue is full */
    while (pool->queue_count >= NIMCP_POOL_MAX_QUEUE && !pool->shutdown) {
        nimcp_cond_wait(&pool->task_complete, &pool->lock);
    }

    if (pool->shutdown) {
        nimcp_mutex_unlock(&pool->lock);
        return NIMCP_ERROR_SYSTEM;
    }

    /* WHAT: Enqueue task */
    pool->queue[pool->queue_tail].func = task;
    pool->queue[pool->queue_tail].arg = arg;
    pool->queue_tail = (pool->queue_tail + 1) % NIMCP_POOL_MAX_QUEUE;
    pool->queue_count++;

    /* WHAT: Signal worker threads */
    nimcp_cond_signal(&pool->task_available);
    nimcp_mutex_unlock(&pool->lock);

    return NIMCP_SUCCESS;
}

nimcp_result_t nimcp_pool_wait(nimcp_thread_pool_t* pool) {
    if (!pool) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    nimcp_mutex_lock(&pool->lock);

    /* WHAT: Wait until queue is empty and no active threads */
    while ((pool->queue_count > 0 || pool->active_threads > 0)
           && !pool->shutdown) {
        nimcp_cond_wait(&pool->task_complete, &pool->lock);
    }

    nimcp_mutex_unlock(&pool->lock);

    return pool->shutdown ? NIMCP_ERROR_SYSTEM : NIMCP_SUCCESS;
}

size_t nimcp_pool_pending(nimcp_thread_pool_t* pool) {
    if (!pool) {
        return 0;
    }

    nimcp_mutex_lock(&pool->lock);
    size_t pending = pool->queue_count;
    nimcp_mutex_unlock(&pool->lock);

    return pending;
}

size_t nimcp_pool_active(nimcp_thread_pool_t* pool) {
    if (!pool) {
        return 0;
    }

    nimcp_mutex_lock(&pool->lock);
    size_t active = pool->active_threads;
    nimcp_mutex_unlock(&pool->lock);

    return active;
}
