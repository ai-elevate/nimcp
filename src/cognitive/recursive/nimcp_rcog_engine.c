/**
 * @file nimcp_rcog_engine.c
 * @brief Recursive Cognition Engine - Main Processing Coordinator Implementation
 * @version 1.0.0
 * @date 2026-01-03
 */

#include "cognitive/recursive/nimcp_rcog_engine.h"
#include "cognitive/recursive/nimcp_rcog_context_store.h"
#include "cognitive/recursive/nimcp_rcog_orchestrator.h"
#include "cognitive/recursive/nimcp_rcog_delegation_pool.h"
#include "cognitive/recursive/nimcp_rcog_tool_router.h"
#include "cognitive/recursive/nimcp_rcog_answer.h"
#include "cognitive/recursive/nimcp_rcog_bio_async_bridge.h"
#include "cognitive/recursive/nimcp_rcog_immune_bridge.h"
#include "cognitive/recursive/nimcp_rcog_imagination_bridge.h"
#include "cognitive/recursive/nimcp_rcog_collective_bridge.h"
#include "cognitive/recursive/nimcp_rcog_brain_kg_bridge.h"
#include "cognitive/recursive/nimcp_rcog_snn_bridge.h"
#include "cognitive/recursive/nimcp_rcog_plasticity_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/time/nimcp_time.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "utils/exception/nimcp_exception_macros.h"

#include <string.h>
#include <stdlib.h>
#include <stdio.h>

/*=============================================================================
 * INTERNAL STRUCTURES
 *===========================================================================*/

/**
 * @brief Active request tracking
 */
typedef struct {
    rcog_request_handle_t* handle;
    rcog_process_request_t request;
    rcog_decomposition_t decomp;
    rcog_batch_handle_t* batch;
    rcog_answer_state_t* answer;
    uint64_t started_ms;
    bool active;
} rcog_active_request_t;

/**
 * @brief Engine internal state
 */
struct rcog_engine {
    /* Configuration */
    rcog_engine_config_t config;

    /* State */
    rcog_engine_state_t state;
    nimcp_mutex_t* mutex;

    /* Subsystems */
    rcog_context_store_t* context_store;
    rcog_orchestrator_t* orchestrator;
    rcog_delegation_pool_t* delegation_pool;
    rcog_tool_router_t* tool_router;
    rcog_answer_refiner_t* answer_refiner;

    /* Bridges */
    struct rcog_bio_async_bridge* bio_async_bridge;
    struct rcog_immune_bridge* immune_bridge;
    struct rcog_imagination_bridge* imagination_bridge;
    struct rcog_collective_bridge* collective_bridge;
    struct rcog_brain_kg_bridge* brain_kg_bridge;
    rcog_snn_bridge_t* snn_bridge;
    rcog_plasticity_bridge_t* plasticity_bridge;
    bool bridges_enabled;

    /* Active requests */
    rcog_active_request_t requests[RCOG_ENGINE_MAX_CONCURRENT_GOALS];
    uint32_t active_count;
    uint64_t next_request_id;
    nimcp_cond_t* request_cond;

    /* Immune modulation */
    rcog_immune_modulation_t current_modulation;
    bool in_degraded_mode;

    /* Statistics */
    rcog_engine_stats_t stats;
    uint64_t stats_start_ms;

    /* Flags */
    bool subsystems_created;
    bool subsystems_connected;
};

/*=============================================================================
 * HELPER FUNCTIONS
 *===========================================================================*/

static uint64_t engine_now_ms(void) {
    return nimcp_time_monotonic_ms();
}

static rcog_active_request_t* find_request_by_id(rcog_engine_t* engine, uint64_t request_id) {
    for (uint32_t i = 0; i < RCOG_ENGINE_MAX_CONCURRENT_GOALS; i++) {
        if (engine->requests[i].active &&
            engine->requests[i].handle &&
            engine->requests[i].handle->request_id == request_id) {
            return &engine->requests[i];
        }
    }
    return NULL;
}

static rcog_active_request_t* allocate_request_slot(rcog_engine_t* engine) {
    for (uint32_t i = 0; i < RCOG_ENGINE_MAX_CONCURRENT_GOALS; i++) {
        if (!engine->requests[i].active) {
            memset(&engine->requests[i], 0, sizeof(rcog_active_request_t));
            engine->requests[i].active = true;
            engine->active_count++;
            return &engine->requests[i];
        }
    }
    return NULL;
}

static void release_request_slot(rcog_engine_t* engine, rcog_active_request_t* req) {
    if (req && req->active) {
        req->active = false;
        if (engine->active_count > 0) {
            engine->active_count--;
        }
    }
}

static void apply_modulation_to_config(rcog_engine_t* engine) {
    /* Apply immune modulation to effective limits */
    rcog_immune_modulation_t* mod = &engine->current_modulation;

    if (engine->orchestrator) {
        rcog_orchestrator_apply_immune_modulation(engine->orchestrator, mod);
    }
    if (engine->delegation_pool) {
        rcog_delegation_pool_apply_immune_modulation(engine->delegation_pool, mod);
    }
}

/*=============================================================================
 * LIFECYCLE
 *===========================================================================*/

rcog_engine_config_t rcog_engine_default_config(void) {
    rcog_engine_config_t config;
    memset(&config, 0, sizeof(config));

    /* Processing limits */
    config.max_recursion_depth = RCOG_DEFAULT_MAX_DEPTH;
    config.max_parallel_subtasks = RCOG_DEFAULT_MAX_PARALLEL_SUBTASKS;
    config.max_concurrent_goals = RCOG_ENGINE_MAX_CONCURRENT_GOALS;
    config.default_timeout_ms = RCOG_ENGINE_DEFAULT_TIMEOUT_MS;

    /* Answer thresholds */
    config.confidence_threshold = RCOG_ENGINE_DEFAULT_CONFIDENCE_THRESHOLD;
    config.max_refinement_steps = RCOG_DEFAULT_MAX_REFINEMENT_STEPS;
    config.enable_early_termination = true;

    /* Decomposition */
    config.default_strategy = RCOG_DECOMP_ADAPTIVE;
    config.enable_adaptive_strategy = true;

    /* Integration - disabled by default */
    config.enable_bio_async = false;
    config.enable_immune_modulation = false;
    config.enable_imagination = false;
    config.enable_collective = false;
    config.enable_brain_kg = false;

    /* Builtin tools - enabled by default */
    config.register_l1_builtins = true;
    config.register_l2_builtins = true;
    config.register_l3_builtins = true;

    /* Debug */
    config.enable_tracing = false;
    config.verbose_logging = false;

    return config;
}

rcog_engine_t* rcog_engine_create(const rcog_engine_config_t* config) {
    rcog_engine_t* engine = nimcp_calloc(1, sizeof(rcog_engine_t));
    if (!engine) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "engine is NULL");

        return NULL;
    }

    /* Copy configuration */
    if (config) {
        engine->config = *config;
    } else {
        engine->config = rcog_engine_default_config();
    }

    /* Initialize mutex */
    engine->mutex = nimcp_mutex_create(NULL);
    if (!engine->mutex) {
        nimcp_free(engine);
        return NULL;
    }

    /* Initialize condition variable for request completion */
    engine->request_cond = nimcp_calloc(1, sizeof(nimcp_cond_t));
    if (!engine->request_cond) {
        nimcp_mutex_free(engine->mutex);
        nimcp_free(engine);
        return NULL;
    }
    nimcp_cond_init(engine->request_cond);

    /* Set initial state */
    engine->state = RCOG_ENGINE_UNINITIALIZED;
    engine->next_request_id = 1;
    engine->stats_start_ms = engine_now_ms();

    /* Initialize default modulation (no effect) */
    engine->current_modulation.capacity_multiplier = 1.0f;
    engine->current_modulation.max_depth_multiplier = 1.0f;
    engine->current_modulation.parallelism_multiplier = 1.0f;
    engine->current_modulation.timeout_multiplier = 1.0f;
    engine->current_modulation.enable_degraded_mode = false;

    return engine;
}

rcog_engine_t* rcog_engine_create_default(void) {
    return rcog_engine_create(NULL);
}

int rcog_engine_init(rcog_engine_t* engine) {
    if (!engine) {
        return RCOG_ERROR_NULL_POINTER;
    }

    nimcp_mutex_lock(engine->mutex);

    if (engine->state != RCOG_ENGINE_UNINITIALIZED) {
        nimcp_mutex_unlock(engine->mutex);
        return RCOG_ERROR_ALREADY_INITIALIZED;
    }

    engine->state = RCOG_ENGINE_INITIALIZING;

    /* Create context store */
    engine->context_store = rcog_context_store_create_default();
    if (!engine->context_store) {
        engine->state = RCOG_ENGINE_UNINITIALIZED;
        nimcp_mutex_unlock(engine->mutex);
        return RCOG_ERROR_OUT_OF_MEMORY;
    }

    /* Create orchestrator */
    rcog_orchestrator_config_t orch_config = rcog_orchestrator_default_config();
    orch_config.max_recursion_depth = engine->config.max_recursion_depth;
    orch_config.max_parallel_subtasks = engine->config.max_parallel_subtasks;
    orch_config.ready_threshold = engine->config.confidence_threshold;
    orch_config.max_refinement_steps = engine->config.max_refinement_steps;
    orch_config.enable_early_termination = engine->config.enable_early_termination;
    orch_config.default_strategy = engine->config.default_strategy;
    orch_config.enable_adaptive_strategy = engine->config.enable_adaptive_strategy;
    orch_config.enable_trace = engine->config.enable_tracing;
    orch_config.verbose_logging = engine->config.verbose_logging;

    engine->orchestrator = rcog_orchestrator_create(&orch_config);
    if (!engine->orchestrator) {
        rcog_context_store_destroy(engine->context_store);
        engine->context_store = NULL;
        engine->state = RCOG_ENGINE_UNINITIALIZED;
        nimcp_mutex_unlock(engine->mutex);
        return RCOG_ERROR_OUT_OF_MEMORY;
    }

    /* Create delegation pool */
    engine->delegation_pool = rcog_delegation_pool_create_default();
    if (!engine->delegation_pool) {
        rcog_orchestrator_destroy(engine->orchestrator);
        engine->orchestrator = NULL;
        rcog_context_store_destroy(engine->context_store);
        engine->context_store = NULL;
        engine->state = RCOG_ENGINE_UNINITIALIZED;
        nimcp_mutex_unlock(engine->mutex);
        return RCOG_ERROR_OUT_OF_MEMORY;
    }

    /* Create tool router */
    engine->tool_router = rcog_tool_router_create_default();
    if (!engine->tool_router) {
        rcog_delegation_pool_destroy(engine->delegation_pool);
        engine->delegation_pool = NULL;
        rcog_orchestrator_destroy(engine->orchestrator);
        engine->orchestrator = NULL;
        rcog_context_store_destroy(engine->context_store);
        engine->context_store = NULL;
        engine->state = RCOG_ENGINE_UNINITIALIZED;
        nimcp_mutex_unlock(engine->mutex);
        return RCOG_ERROR_OUT_OF_MEMORY;
    }

    /* Create answer refiner */
    rcog_answer_config_t answer_config = rcog_answer_default_config();
    answer_config.ready_threshold = engine->config.confidence_threshold;
    answer_config.max_steps = engine->config.max_refinement_steps;
    answer_config.enable_early_stopping = engine->config.enable_early_termination;

    engine->answer_refiner = rcog_answer_refiner_create(&answer_config);
    if (!engine->answer_refiner) {
        rcog_tool_router_destroy(engine->tool_router);
        engine->tool_router = NULL;
        rcog_delegation_pool_destroy(engine->delegation_pool);
        engine->delegation_pool = NULL;
        rcog_orchestrator_destroy(engine->orchestrator);
        engine->orchestrator = NULL;
        rcog_context_store_destroy(engine->context_store);
        engine->context_store = NULL;
        engine->state = RCOG_ENGINE_UNINITIALIZED;
        nimcp_mutex_unlock(engine->mutex);
        return RCOG_ERROR_OUT_OF_MEMORY;
    }

    engine->subsystems_created = true;

    /* Connect subsystems to each other */
    rcog_orchestrator_connect_context_store(engine->orchestrator, engine->context_store);
    rcog_orchestrator_connect_answer_refiner(engine->orchestrator, engine->answer_refiner);
    rcog_orchestrator_connect_delegation_pool(engine->orchestrator, engine->delegation_pool);
    rcog_orchestrator_connect_engine(engine->orchestrator, engine);

    rcog_delegation_pool_connect_tool_router(engine->delegation_pool, engine->tool_router);
    rcog_delegation_pool_connect_context_store(engine->delegation_pool, engine->context_store);

    rcog_tool_router_connect_context_store(engine->tool_router, engine->context_store);

    engine->subsystems_connected = true;

    /* Create SNN and plasticity bridges with default configs */
    rcog_snn_config_t snn_config = rcog_snn_config_default();
    engine->snn_bridge = rcog_snn_create(&snn_config);

    rcog_plasticity_config_t plasticity_config = rcog_plasticity_config_default();
    engine->plasticity_bridge = rcog_plasticity_create(&plasticity_config);

    /* Bridges are optional - don't fail init if they can't be created */
    engine->bridges_enabled = (engine->snn_bridge != NULL && engine->plasticity_bridge != NULL);

    /* Register builtin tools if configured */
    if (engine->config.register_l1_builtins) {
        rcog_tool_router_register_l1_builtins(engine->tool_router);
    }
    if (engine->config.register_l2_builtins) {
        rcog_tool_router_register_l2_builtins(engine->tool_router);
    }
    if (engine->config.register_l3_builtins) {
        rcog_tool_router_register_l3_builtins(engine->tool_router);
    }

    engine->state = RCOG_ENGINE_READY;
    nimcp_mutex_unlock(engine->mutex);

    return 0;
}

int rcog_engine_start(rcog_engine_t* engine) {
    if (!engine) {
        return RCOG_ERROR_NULL_POINTER;
    }

    nimcp_mutex_lock(engine->mutex);

    if (engine->state != RCOG_ENGINE_READY && engine->state != RCOG_ENGINE_PAUSED) {
        nimcp_mutex_unlock(engine->mutex);
        return RCOG_ERROR_NOT_INITIALIZED;
    }

    /* Start delegation pool */
    int result = rcog_delegation_pool_start(engine->delegation_pool);
    if (result != 0) {
        nimcp_mutex_unlock(engine->mutex);
        return result;
    }

    engine->state = RCOG_ENGINE_READY;
    nimcp_mutex_unlock(engine->mutex);

    return 0;
}

int rcog_engine_stop(rcog_engine_t* engine, uint32_t timeout_ms) {
    if (!engine) {
        return RCOG_ERROR_NULL_POINTER;
    }

    nimcp_mutex_lock(engine->mutex);

    if (engine->state == RCOG_ENGINE_STOPPED ||
        engine->state == RCOG_ENGINE_UNINITIALIZED) {
        nimcp_mutex_unlock(engine->mutex);
        return 0;
    }

    engine->state = RCOG_ENGINE_SHUTTING_DOWN;

    /* Cancel all active requests */
    for (uint32_t i = 0; i < RCOG_ENGINE_MAX_CONCURRENT_GOALS; i++) {
        if (engine->requests[i].active && engine->requests[i].handle) {
            engine->requests[i].handle->cancelled = true;
            engine->stats.goals_cancelled++;
        }
    }

    nimcp_mutex_unlock(engine->mutex);

    /* Stop delegation pool */
    if (engine->delegation_pool) {
        rcog_delegation_pool_stop(engine->delegation_pool, timeout_ms);
    }

    nimcp_mutex_lock(engine->mutex);
    engine->state = RCOG_ENGINE_STOPPED;

    /* Signal any waiting threads */
    nimcp_cond_broadcast(engine->request_cond);

    nimcp_mutex_unlock(engine->mutex);

    return 0;
}

void rcog_engine_destroy(rcog_engine_t* engine) {
    if (!engine) {
        return;
    }

    /* Stop if not already stopped */
    if (engine->state != RCOG_ENGINE_STOPPED &&
        engine->state != RCOG_ENGINE_UNINITIALIZED) {
        rcog_engine_stop(engine, 5000);
    }

    /* Destroy subsystems in reverse order */
    if (engine->answer_refiner) {
        rcog_answer_refiner_destroy(engine->answer_refiner);
        engine->answer_refiner = NULL;
    }

    if (engine->tool_router) {
        rcog_tool_router_destroy(engine->tool_router);
        engine->tool_router = NULL;
    }

    if (engine->delegation_pool) {
        rcog_delegation_pool_destroy(engine->delegation_pool);
        engine->delegation_pool = NULL;
    }

    if (engine->orchestrator) {
        rcog_orchestrator_destroy(engine->orchestrator);
        engine->orchestrator = NULL;
    }

    if (engine->context_store) {
        rcog_context_store_destroy(engine->context_store);
        engine->context_store = NULL;
    }

    /* Destroy SNN and plasticity bridges */
    if (engine->snn_bridge) {
        rcog_snn_destroy(engine->snn_bridge);
        engine->snn_bridge = NULL;
    }

    if (engine->plasticity_bridge) {
        rcog_plasticity_destroy(engine->plasticity_bridge);
        engine->plasticity_bridge = NULL;
    }

    engine->bridges_enabled = false;

    /* Clean up active request handles */
    for (uint32_t i = 0; i < RCOG_ENGINE_MAX_CONCURRENT_GOALS; i++) {
        if (engine->requests[i].handle) {
            nimcp_free(engine->requests[i].handle);
            engine->requests[i].handle = NULL;
        }
        if (engine->requests[i].answer) {
            rcog_answer_state_destroy(engine->requests[i].answer);
            engine->requests[i].answer = NULL;
        }
    }

    /* Destroy synchronization primitives */
    if (engine->request_cond) {
        nimcp_cond_destroy(engine->request_cond);
        nimcp_free(engine->request_cond);
        engine->request_cond = NULL;
    }

    if (engine->mutex) {
        nimcp_mutex_free(engine->mutex);
        engine->mutex = NULL;
    }

    nimcp_free(engine);
}

rcog_engine_state_t rcog_engine_get_state(const rcog_engine_t* engine) {
    if (!engine) {
        return RCOG_ENGINE_UNINITIALIZED;
    }
    return engine->state;
}

/*=============================================================================
 * SUBSYSTEM ACCESS
 *===========================================================================*/

struct rcog_context_store* rcog_engine_get_context_store(rcog_engine_t* engine) {
    if (!engine) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "engine is NULL");

        return NULL;
    }
    return engine->context_store;
}

struct rcog_orchestrator* rcog_engine_get_orchestrator(rcog_engine_t* engine) {
    if (!engine) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "engine is NULL");

        return NULL;
    }
    return engine->orchestrator;
}

struct rcog_delegation_pool* rcog_engine_get_delegation_pool(rcog_engine_t* engine) {
    if (!engine) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "engine is NULL");

        return NULL;
    }
    return engine->delegation_pool;
}

struct rcog_tool_router* rcog_engine_get_tool_router(rcog_engine_t* engine) {
    if (!engine) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "engine is NULL");

        return NULL;
    }
    return engine->tool_router;
}

struct rcog_answer_refiner* rcog_engine_get_answer_refiner(rcog_engine_t* engine) {
    if (!engine) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "engine is NULL");

        return NULL;
    }
    return engine->answer_refiner;
}

/*=============================================================================
 * BRIDGE CONNECTION
 *===========================================================================*/

int rcog_engine_connect_bio_async(
    rcog_engine_t* engine,
    struct rcog_bio_async_bridge* bridge
) {
    if (!engine) {
        return RCOG_ERROR_NULL_POINTER;
    }

    nimcp_mutex_lock(engine->mutex);
    engine->bio_async_bridge = bridge;

    /* Connect bridge to subsystems if enabled */
    if (bridge && engine->config.enable_bio_async) {
        if (engine->delegation_pool) {
            rcog_delegation_pool_connect_bio_async(engine->delegation_pool, bridge);
        }
        if (engine->tool_router) {
            rcog_tool_router_connect_bio_async(engine->tool_router, bridge);
        }
    }

    nimcp_mutex_unlock(engine->mutex);
    return 0;
}

int rcog_engine_connect_immune(
    rcog_engine_t* engine,
    struct rcog_immune_bridge* bridge
) {
    if (!engine) {
        return RCOG_ERROR_NULL_POINTER;
    }

    nimcp_mutex_lock(engine->mutex);
    engine->immune_bridge = bridge;

    /* Connect bridge to subsystems if enabled */
    if (bridge && engine->config.enable_immune_modulation) {
        if (engine->orchestrator) {
            rcog_orchestrator_connect_immune(engine->orchestrator, bridge);
        }
        if (engine->delegation_pool) {
            rcog_delegation_pool_connect_immune(engine->delegation_pool, bridge);
        }
        if (engine->tool_router) {
            rcog_tool_router_connect_immune(engine->tool_router, bridge);
        }
    }

    nimcp_mutex_unlock(engine->mutex);
    return 0;
}

int rcog_engine_connect_imagination(
    rcog_engine_t* engine,
    struct rcog_imagination_bridge* bridge
) {
    if (!engine) {
        return RCOG_ERROR_NULL_POINTER;
    }

    nimcp_mutex_lock(engine->mutex);
    engine->imagination_bridge = bridge;

    /* Connect bridge to orchestrator for planning */
    if (bridge && engine->config.enable_imagination && engine->orchestrator) {
        rcog_orchestrator_connect_imagination(engine->orchestrator, bridge);
    }

    nimcp_mutex_unlock(engine->mutex);
    return 0;
}

int rcog_engine_connect_collective(
    rcog_engine_t* engine,
    struct rcog_collective_bridge* bridge
) {
    if (!engine) {
        return RCOG_ERROR_NULL_POINTER;
    }

    nimcp_mutex_lock(engine->mutex);
    engine->collective_bridge = bridge;

    /* Connect bridge to delegation pool for swarm distribution */
    if (bridge && engine->config.enable_collective && engine->delegation_pool) {
        rcog_delegation_pool_connect_collective(engine->delegation_pool, bridge);
    }

    nimcp_mutex_unlock(engine->mutex);
    return 0;
}

int rcog_engine_connect_brain_kg(
    rcog_engine_t* engine,
    struct rcog_brain_kg_bridge* bridge
) {
    if (!engine) {
        return RCOG_ERROR_NULL_POINTER;
    }

    nimcp_mutex_lock(engine->mutex);
    engine->brain_kg_bridge = bridge;
    nimcp_mutex_unlock(engine->mutex);
    return 0;
}

/*=============================================================================
 * GOAL PROCESSING
 *===========================================================================*/

int rcog_engine_process(
    rcog_engine_t* engine,
    const rcog_goal_t* goal,
    rcog_process_result_t* result
) {
    if (!engine || !goal || !result) {
        return RCOG_ERROR_NULL_POINTER;
    }

    /* Create sync request */
    rcog_process_request_t request = rcog_engine_default_request(goal);
    request.mode = RCOG_MODE_SYNC;

    return rcog_engine_process_ex(engine, &request, result, NULL);
}

int rcog_engine_process_ex(
    rcog_engine_t* engine,
    const rcog_process_request_t* request,
    rcog_process_result_t* result,
    rcog_request_handle_t** handle
) {
    if (!engine || !request) {
        return RCOG_ERROR_NULL_POINTER;
    }

    nimcp_mutex_lock(engine->mutex);

    /* Check engine state */
    if (engine->state != RCOG_ENGINE_READY &&
        engine->state != RCOG_ENGINE_PROCESSING &&
        engine->state != RCOG_ENGINE_DEGRADED) {
        nimcp_mutex_unlock(engine->mutex);
        return RCOG_ERROR_NOT_INITIALIZED;
    }

    /* Check capacity */
    if (engine->active_count >= engine->config.max_concurrent_goals) {
        nimcp_mutex_unlock(engine->mutex);
        return RCOG_ERROR_WORKER_POOL_EXHAUSTED;
    }

    /* Allocate request slot */
    rcog_active_request_t* active_req = allocate_request_slot(engine);
    if (!active_req) {
        nimcp_mutex_unlock(engine->mutex);
        return RCOG_ERROR_WORKER_POOL_EXHAUSTED;
    }

    /* Create request handle */
    active_req->handle = nimcp_calloc(1, sizeof(rcog_request_handle_t));
    if (!active_req->handle) {
        release_request_slot(engine, active_req);
        nimcp_mutex_unlock(engine->mutex);
        return RCOG_ERROR_OUT_OF_MEMORY;
    }

    active_req->handle->request_id = engine->next_request_id++;
    active_req->handle->goal = request->goal;
    active_req->handle->mode = request->mode;
    active_req->handle->state = RCOG_ENGINE_PROCESSING;
    active_req->handle->completed = false;
    active_req->handle->cancelled = false;
    active_req->request = *request;
    active_req->started_ms = engine_now_ms();

    engine->state = RCOG_ENGINE_PROCESSING;
    engine->stats.goals_submitted++;

    nimcp_mutex_unlock(engine->mutex);

    /* Create answer state for this request */
    active_req->answer = rcog_answer_state_create(engine->answer_refiner, &request->goal);
    if (!active_req->answer) {
        nimcp_mutex_lock(engine->mutex);
        nimcp_free(active_req->handle);
        active_req->handle = NULL;
        release_request_slot(engine, active_req);
        nimcp_mutex_unlock(engine->mutex);
        return RCOG_ERROR_OUT_OF_MEMORY;
    }

    int proc_result = 0;

    /* Decompose goal */
    if (!request->skip_decomposition) {
        rcog_decomposition_strategy_t strategy = request->strategy_override;
        if (strategy == 0) {
            strategy = engine->config.default_strategy;
        }

        proc_result = rcog_orchestrator_decompose_with_strategy(
            engine->orchestrator,
            &request->goal,
            engine->context_store,
            strategy,
            &active_req->decomp
        );

        if (proc_result != 0) {
            goto cleanup;
        }

        /* Dispatch decomposition to pool */
        if (active_req->decomp.num_subtasks > 0) {
            proc_result = rcog_orchestrator_dispatch(
                engine->orchestrator,
                &active_req->decomp,
                &active_req->batch
            );

            if (proc_result != 0) {
                goto cleanup;
            }

            /* Update stats */
            nimcp_mutex_lock(engine->mutex);
            engine->stats.subtasks_created += active_req->decomp.num_subtasks;
            if (active_req->decomp.metadata.depth > engine->stats.max_depth_reached) {
                engine->stats.max_depth_reached = active_req->decomp.metadata.depth;
            }
            nimcp_mutex_unlock(engine->mutex);
        }
    }

    /* Handle by mode */
    if (request->mode == RCOG_MODE_SYNC) {
        /* Synchronous: wait for completion */
        uint32_t timeout = request->timeout_ms;
        if (timeout == 0) {
            timeout = engine->config.default_timeout_ms;
        }

        if (active_req->batch) {
            proc_result = rcog_delegation_pool_await_batch(
                engine->delegation_pool,
                active_req->batch,
                timeout
            );

            if (proc_result == RCOG_ERROR_TIMEOUT) {
                nimcp_mutex_lock(engine->mutex);
                engine->stats.goals_timeout++;
                nimcp_mutex_unlock(engine->mutex);
                goto cleanup;
            }

            /* Get batch results and aggregate */
            rcog_subtask_result_t* batch_results = NULL;
            size_t num_results = 0;

            if (active_req->decomp.num_subtasks > 0) {
                batch_results = nimcp_calloc(active_req->decomp.num_subtasks,
                                             sizeof(rcog_subtask_result_t));
                if (batch_results) {
                    rcog_delegation_pool_get_batch_results(
                        engine->delegation_pool,
                        active_req->batch,
                        batch_results,
                        active_req->decomp.num_subtasks,
                        &num_results
                    );

                    /* Aggregate results into answer */
                    rcog_orchestrator_aggregate(
                        engine->orchestrator,
                        active_req->batch,
                        batch_results,
                        num_results,
                        active_req->answer
                    );

                    /* Update subtask stats */
                    nimcp_mutex_lock(engine->mutex);
                    for (size_t i = 0; i < num_results; i++) {
                        if (batch_results[i].success) {
                            engine->stats.subtasks_completed++;
                        } else {
                            engine->stats.subtasks_failed++;
                        }
                    }
                    nimcp_mutex_unlock(engine->mutex);

                    nimcp_free(batch_results);
                }
            }
        }

        /* Refine until ready */
        while (!rcog_answer_is_ready(engine->answer_refiner, active_req->answer)) {
            proc_result = rcog_orchestrator_refine(
                engine->orchestrator,
                active_req->answer,
                engine->context_store
            );

            if (proc_result != 0) {
                break;
            }

            /* Check for stall */
            if (rcog_answer_is_stalled(engine->answer_refiner, active_req->answer, 3)) {
                break;
            }

            /* Check max steps */
            if (active_req->answer->refinement_step >= engine->config.max_refinement_steps) {
                break;
            }
        }

        /* Fill result */
        if (result) {
            memset(result, 0, sizeof(rcog_process_result_t));
            result->request_id = active_req->handle->request_id;
            result->goal = request->goal;
            result->answer = *active_req->answer;
            result->success = (active_req->answer->status == RCOG_ANSWER_READY);
            result->subtasks_created = active_req->decomp.num_subtasks;
            result->max_depth_used = active_req->decomp.metadata.depth;
            result->refinement_steps = active_req->answer->refinement_step;
            result->processing_time_ms = engine_now_ms() - active_req->started_ms;

            if (result->success) {
                nimcp_mutex_lock(engine->mutex);
                engine->stats.goals_completed++;
                engine->stats.total_processing_time_ms += result->processing_time_ms;
                engine->stats.avg_confidence =
                    (engine->stats.avg_confidence * (engine->stats.goals_completed - 1) +
                     active_req->answer->confidence) / engine->stats.goals_completed;
                nimcp_mutex_unlock(engine->mutex);
            } else {
                result->error = RCOG_ERROR_ANSWER_NOT_READY;
                result->error_message = "Answer did not reach confidence threshold";
                nimcp_mutex_lock(engine->mutex);
                engine->stats.goals_failed++;
                nimcp_mutex_unlock(engine->mutex);
            }
        }

        active_req->handle->completed = true;

    } else if (request->mode == RCOG_MODE_ASYNC) {
        /* Asynchronous: return handle immediately */
        if (handle) {
            *handle = active_req->handle;
        }
        /* Don't cleanup - caller will use rcog_engine_await */
        return 0;

    } else if (request->mode == RCOG_MODE_STREAMING) {
        /* Streaming: call progress callback */
        if (request->progress_callback) {
            rcog_progress_t progress;
            memset(&progress, 0, sizeof(progress));
            progress.total_subtasks = active_req->decomp.num_subtasks;
            progress.current_confidence = active_req->answer->confidence;
            progress.elapsed_ms = engine_now_ms() - active_req->started_ms;
            request->progress_callback(&progress, request->progress_user_data);
        }

        if (handle) {
            *handle = active_req->handle;
        }
        return 0;
    }

cleanup:
    nimcp_mutex_lock(engine->mutex);

    /* Update engine state if no more active requests */
    if (engine->active_count <= 1) {
        engine->state = engine->in_degraded_mode ? RCOG_ENGINE_DEGRADED : RCOG_ENGINE_READY;
    }

    /* Free request resources for sync mode */
    if (request->mode == RCOG_MODE_SYNC) {
        if (active_req->batch) {
            rcog_delegation_pool_free_batch_handle(active_req->batch);
            active_req->batch = NULL;
        }
        rcog_orchestrator_free_decomposition(&active_req->decomp);
        /* Answer state transferred to result, don't free */
        active_req->answer = NULL;
        nimcp_free(active_req->handle);
        active_req->handle = NULL;
        release_request_slot(engine, active_req);
    }

    nimcp_mutex_unlock(engine->mutex);

    return proc_result;
}

int rcog_engine_process_async(
    rcog_engine_t* engine,
    const rcog_goal_t* goal,
    rcog_progress_callback_t callback,
    void* user_data,
    rcog_request_handle_t** handle
) {
    if (!engine || !goal || !handle) {
        return RCOG_ERROR_NULL_POINTER;
    }

    rcog_process_request_t request = rcog_engine_default_request(goal);
    request.mode = RCOG_MODE_ASYNC;
    request.progress_callback = callback;
    request.progress_user_data = user_data;

    return rcog_engine_process_ex(engine, &request, NULL, handle);
}

int rcog_engine_await(
    rcog_engine_t* engine,
    rcog_request_handle_t* handle,
    uint32_t timeout_ms,
    rcog_process_result_t* result
) {
    if (!engine || !handle) {
        return RCOG_ERROR_NULL_POINTER;
    }

    uint64_t start_ms = engine_now_ms();
    uint64_t deadline = timeout_ms > 0 ? start_ms + timeout_ms : 0;

    nimcp_mutex_lock(engine->mutex);

    rcog_active_request_t* active_req = find_request_by_id(engine, handle->request_id);
    if (!active_req) {
        nimcp_mutex_unlock(engine->mutex);
        return RCOG_ERROR_CONTEXT_NOT_FOUND;
    }

    /* Wait for completion */
    while (!handle->completed && !handle->cancelled) {
        if (deadline > 0) {
            uint64_t now = engine_now_ms();
            if (now >= deadline) {
                nimcp_mutex_unlock(engine->mutex);
                return RCOG_ERROR_TIMEOUT;
            }
            nimcp_cond_timedwait(engine->request_cond, engine->mutex,
                                 (uint32_t)(deadline - now));
        } else {
            nimcp_cond_wait(engine->request_cond, engine->mutex);
        }
    }

    /* Fill result */
    if (result && active_req->answer) {
        memset(result, 0, sizeof(rcog_process_result_t));
        result->request_id = handle->request_id;
        result->goal = active_req->request.goal;
        result->answer = *active_req->answer;
        result->success = (active_req->answer->status == RCOG_ANSWER_READY);
        result->subtasks_created = active_req->decomp.num_subtasks;
        result->max_depth_used = active_req->decomp.metadata.depth;
        result->refinement_steps = active_req->answer->refinement_step;
        result->processing_time_ms = engine_now_ms() - active_req->started_ms;

        if (handle->cancelled) {
            result->error = RCOG_ERROR_TIMEOUT;
            result->error_message = "Request cancelled";
        }
    }

    nimcp_mutex_unlock(engine->mutex);

    return 0;
}

int rcog_engine_cancel(
    rcog_engine_t* engine,
    rcog_request_handle_t* handle
) {
    if (!engine || !handle) {
        return RCOG_ERROR_NULL_POINTER;
    }

    nimcp_mutex_lock(engine->mutex);

    rcog_active_request_t* active_req = find_request_by_id(engine, handle->request_id);
    if (!active_req) {
        nimcp_mutex_unlock(engine->mutex);
        return RCOG_ERROR_CONTEXT_NOT_FOUND;
    }

    handle->cancelled = true;
    engine->stats.goals_cancelled++;

    /* Cancel batch if exists */
    if (active_req->batch) {
        rcog_delegation_pool_cancel_batch(engine->delegation_pool, active_req->batch);
    }

    /* Signal waiters */
    nimcp_cond_broadcast(engine->request_cond);

    nimcp_mutex_unlock(engine->mutex);

    return 0;
}

void rcog_engine_free_handle(rcog_request_handle_t* handle) {
    if (handle) {
        nimcp_free(handle);
    }
}

void rcog_engine_free_result(rcog_process_result_t* result) {
    if (result) {
        if (result->answer.content) {
            nimcp_free(result->answer.content);
            result->answer.content = NULL;
        }
        if (result->answer.latent) {
            nimcp_free(result->answer.latent);
            result->answer.latent = NULL;
        }
    }
}

/*=============================================================================
 * CONTEXT MANAGEMENT
 *===========================================================================*/

int rcog_engine_set_context(
    rcog_engine_t* engine,
    const char* name,
    const void* data,
    size_t size,
    rcog_data_type_t dtype
) {
    if (!engine || !name || !data) {
        return RCOG_ERROR_NULL_POINTER;
    }

    if (!engine->context_store) {
        return RCOG_ERROR_NOT_INITIALIZED;
    }

    return rcog_context_store_set(engine->context_store, name, data, size, dtype);
}

int rcog_engine_get_context(
    rcog_engine_t* engine,
    const char* name,
    rcog_query_result_t* result
) {
    if (!engine || !name || !result) {
        return RCOG_ERROR_NULL_POINTER;
    }

    if (!engine->context_store) {
        return RCOG_ERROR_NOT_INITIALIZED;
    }

    return rcog_context_store_query(engine->context_store, name,
                                     RCOG_ACCESS_FULL, NULL, result);
}

int rcog_engine_query_context(
    rcog_engine_t* engine,
    const char* name,
    rcog_access_pattern_t pattern,
    const rcog_query_params_t* params,
    rcog_query_result_t* result
) {
    if (!engine || !name || !result) {
        return RCOG_ERROR_NULL_POINTER;
    }

    if (!engine->context_store) {
        return RCOG_ERROR_NOT_INITIALIZED;
    }

    return rcog_context_store_query(engine->context_store, name, pattern, params, result);
}

int rcog_engine_clear_context(
    rcog_engine_t* engine,
    const char* name
) {
    if (!engine || !name) {
        return RCOG_ERROR_NULL_POINTER;
    }

    if (!engine->context_store) {
        return RCOG_ERROR_NOT_INITIALIZED;
    }

    return rcog_context_store_remove(engine->context_store, name);
}

int rcog_engine_clear_all_context(rcog_engine_t* engine) {
    if (!engine) {
        return RCOG_ERROR_NULL_POINTER;
    }

    if (!engine->context_store) {
        return RCOG_ERROR_NOT_INITIALIZED;
    }

    return rcog_context_store_clear(engine->context_store);
}

/*=============================================================================
 * TOOL MANAGEMENT
 *===========================================================================*/

int rcog_engine_register_tool(
    rcog_engine_t* engine,
    const char* name,
    rcog_tool_fn handler,
    rcog_capability_tier_t tier,
    void* context
) {
    if (!engine || !name || !handler) {
        return RCOG_ERROR_NULL_POINTER;
    }

    if (!engine->tool_router) {
        return RCOG_ERROR_NOT_INITIALIZED;
    }

    rcog_tool_def_t def = rcog_tool_def_create(name, handler, tier);
    def.context = context;

    return rcog_tool_router_register(engine->tool_router, &def);
}

int rcog_engine_unregister_tool(
    rcog_engine_t* engine,
    const char* name
) {
    if (!engine || !name) {
        return RCOG_ERROR_NULL_POINTER;
    }

    if (!engine->tool_router) {
        return RCOG_ERROR_NOT_INITIALIZED;
    }

    return rcog_tool_router_unregister(engine->tool_router, name);
}

int rcog_engine_list_tools(
    rcog_engine_t* engine,
    rcog_capability_tier_t tier,
    char (*tools)[64],
    size_t max_tools,
    size_t* num_tools
) {
    if (!engine || !tools || !num_tools) {
        return RCOG_ERROR_NULL_POINTER;
    }

    if (!engine->tool_router) {
        return RCOG_ERROR_NOT_INITIALIZED;
    }

    return rcog_tool_router_get_accessible_tools(
        engine->tool_router, tier, tools, max_tools, num_tools);
}

/*=============================================================================
 * IMMUNE MODULATION
 *===========================================================================*/

int rcog_engine_apply_immune_modulation(
    rcog_engine_t* engine,
    const rcog_immune_modulation_t* modulation
) {
    if (!engine || !modulation) {
        return RCOG_ERROR_NULL_POINTER;
    }

    nimcp_mutex_lock(engine->mutex);

    engine->current_modulation = *modulation;
    engine->stats.immune_modulations++;

    /* Apply to subsystems */
    apply_modulation_to_config(engine);

    /* Check for degraded mode */
    if (modulation->enable_degraded_mode && !engine->in_degraded_mode) {
        engine->in_degraded_mode = true;
        if (engine->state == RCOG_ENGINE_READY || engine->state == RCOG_ENGINE_PROCESSING) {
            engine->state = RCOG_ENGINE_DEGRADED;
        }
    }

    nimcp_mutex_unlock(engine->mutex);

    return 0;
}

int rcog_engine_get_immune_modulation(
    const rcog_engine_t* engine,
    rcog_immune_modulation_t* modulation
) {
    if (!engine || !modulation) {
        return RCOG_ERROR_NULL_POINTER;
    }

    *modulation = engine->current_modulation;
    return 0;
}

int rcog_engine_enter_degraded_mode(rcog_engine_t* engine) {
    if (!engine) {
        return RCOG_ERROR_NULL_POINTER;
    }

    nimcp_mutex_lock(engine->mutex);

    engine->in_degraded_mode = true;
    engine->current_modulation.enable_degraded_mode = true;

    if (engine->state == RCOG_ENGINE_READY || engine->state == RCOG_ENGINE_PROCESSING) {
        engine->state = RCOG_ENGINE_DEGRADED;
    }

    /* Apply reduced limits */
    engine->current_modulation.capacity_multiplier = 0.5f;
    engine->current_modulation.max_depth_multiplier = 0.5f;
    engine->current_modulation.parallelism_multiplier = 0.5f;
    apply_modulation_to_config(engine);

    nimcp_mutex_unlock(engine->mutex);

    return 0;
}

int rcog_engine_exit_degraded_mode(rcog_engine_t* engine) {
    if (!engine) {
        return RCOG_ERROR_NULL_POINTER;
    }

    nimcp_mutex_lock(engine->mutex);

    engine->in_degraded_mode = false;
    engine->current_modulation.enable_degraded_mode = false;

    if (engine->state == RCOG_ENGINE_DEGRADED) {
        engine->state = engine->active_count > 0 ? RCOG_ENGINE_PROCESSING : RCOG_ENGINE_READY;
    }

    /* Restore full limits */
    engine->current_modulation.capacity_multiplier = 1.0f;
    engine->current_modulation.max_depth_multiplier = 1.0f;
    engine->current_modulation.parallelism_multiplier = 1.0f;
    apply_modulation_to_config(engine);

    nimcp_mutex_unlock(engine->mutex);

    return 0;
}

/*=============================================================================
 * STATISTICS
 *===========================================================================*/

int rcog_engine_get_stats(
    const rcog_engine_t* engine,
    rcog_engine_stats_t* stats
) {
    if (!engine || !stats) {
        return RCOG_ERROR_NULL_POINTER;
    }

    *stats = engine->stats;
    stats->active_goals = engine->active_count;
    stats->state = engine->state;

    /* Calculate averages */
    if (stats->goals_completed > 0) {
        stats->avg_processing_time_ms =
            (float)stats->total_processing_time_ms / stats->goals_completed;
    }

    return 0;
}

void rcog_engine_reset_stats(rcog_engine_t* engine) {
    if (!engine) {
        return;
    }

    nimcp_mutex_lock(engine->mutex);

    memset(&engine->stats, 0, sizeof(engine->stats));
    engine->stats_start_ms = engine_now_ms();

    nimcp_mutex_unlock(engine->mutex);
}

int rcog_engine_get_progress(
    const rcog_engine_t* engine,
    const rcog_request_handle_t* handle,
    rcog_progress_t* progress
) {
    if (!engine || !progress) {
        return RCOG_ERROR_NULL_POINTER;
    }

    memset(progress, 0, sizeof(rcog_progress_t));

    if (handle) {
        /* Get progress for specific request */
        nimcp_mutex_lock(((rcog_engine_t*)engine)->mutex);

        rcog_active_request_t* active_req = find_request_by_id(
            (rcog_engine_t*)engine, handle->request_id);

        if (active_req && active_req->answer) {
            progress->total_subtasks = active_req->decomp.num_subtasks;
            progress->current_confidence = active_req->answer->confidence;
            progress->refinement_step = active_req->answer->refinement_step;
            progress->max_depth_reached = active_req->decomp.metadata.depth;
            progress->elapsed_ms = engine_now_ms() - active_req->started_ms;

            /* Count completed subtasks from batch */
            if (active_req->batch) {
                size_t completed = 0;
                size_t total = 0;
                rcog_delegation_pool_poll_batch(
                    engine->delegation_pool,
                    active_req->batch,
                    &completed,
                    &total
                );
                progress->completed_subtasks = completed;
                progress->active_subtasks = total - completed;
            }
        }

        nimcp_mutex_unlock(((rcog_engine_t*)engine)->mutex);
    } else {
        /* Get overall progress */
        progress->completed_subtasks = engine->stats.subtasks_completed;
        progress->max_depth_reached = engine->stats.max_depth_reached;
        progress->current_confidence = engine->stats.avg_confidence;
    }

    return 0;
}

/*=============================================================================
 * UTILITY
 *===========================================================================*/

rcog_process_request_t rcog_engine_default_request(const rcog_goal_t* goal) {
    rcog_process_request_t request;
    memset(&request, 0, sizeof(request));

    if (goal) {
        request.goal = *goal;
    }

    request.mode = RCOG_MODE_SYNC;
    request.timeout_ms = 0;  /* Use engine default */
    request.skip_decomposition = false;
    request.force_local = false;
    request.strategy_override = 0;  /* Use engine default */

    return request;
}

rcog_goal_t rcog_engine_create_goal(
    const char* query,
    rcog_goal_type_t type
) {
    rcog_goal_t goal;
    memset(&goal, 0, sizeof(goal));

    goal.type = type;
    goal.query = query;
    goal.priority = 0.5f;
    goal.timeout_ms = 0;  /* Use default */
    goal.max_depth = 0;   /* Use default */

    return goal;
}

const char* rcog_engine_state_name(rcog_engine_state_t state) {
    switch (state) {
        case RCOG_ENGINE_UNINITIALIZED: return "uninitialized";
        case RCOG_ENGINE_INITIALIZING:  return "initializing";
        case RCOG_ENGINE_READY:         return "ready";
        case RCOG_ENGINE_PROCESSING:    return "processing";
        case RCOG_ENGINE_PAUSED:        return "paused";
        case RCOG_ENGINE_DEGRADED:      return "degraded";
        case RCOG_ENGINE_SHUTTING_DOWN: return "shutting_down";
        case RCOG_ENGINE_STOPPED:       return "stopped";
        default:                        return "unknown";
    }
}

bool rcog_engine_is_ready(const rcog_engine_t* engine) {
    if (!engine) {
        return false;
    }

    return engine->state == RCOG_ENGINE_READY ||
           engine->state == RCOG_ENGINE_PROCESSING ||
           engine->state == RCOG_ENGINE_DEGRADED;
}

bool rcog_engine_has_capacity(const rcog_engine_t* engine) {
    if (!engine) {
        return false;
    }

    if (!rcog_engine_is_ready(engine)) {
        return false;
    }

    /* Apply modulation to max concurrent */
    uint32_t effective_max = (uint32_t)(
        engine->config.max_concurrent_goals *
        engine->current_modulation.capacity_multiplier
    );

    if (effective_max == 0) {
        effective_max = 1;  /* Always allow at least one */
    }

    return engine->active_count < effective_max;
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

int rcog_engine_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;
    const kg_entity_t* self = kg_reader_get_entity(kg, "Recursive_Cognition_Engine_Module");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            /* Log self-knowledge observations */
        }
    }
    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Recursive_Cognition_Engine_Module");
    if (connections) { kg_relation_list_destroy(connections); }
    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Recursive_Cognition_Engine_Module");
    if (incoming) { kg_relation_list_destroy(incoming); }
    return self ? 1 : 0;
}
