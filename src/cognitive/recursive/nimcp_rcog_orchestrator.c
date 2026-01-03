/**
 * @file nimcp_rcog_orchestrator.c
 * @brief Recursive Cognition Orchestrator Implementation
 * @version 1.0.0
 * @date 2026-01-03
 */

#include "cognitive/recursive/nimcp_rcog_orchestrator.h"
#include "cognitive/recursive/nimcp_rcog_context_store.h"
#include "cognitive/recursive/nimcp_rcog_answer.h"
#include "cognitive/recursive/nimcp_rcog_imagination_bridge.h"
#include "cognitive/recursive/nimcp_rcog_immune_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/platform/nimcp_platform_time.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"

#include <string.h>
#include <stdio.h>

/*=============================================================================
 * INTERNAL STRUCTURES
 *===========================================================================*/

/**
 * @brief Active decomposition tracking
 */
typedef struct {
    uint64_t decomp_id;
    rcog_decomposition_t* decomp;
    bool in_use;
} rcog_active_decomp_t;

/**
 * @brief Batch handle for tracking dispatched subtasks
 */
struct rcog_batch_handle {
    uint64_t batch_id;               /**< Unique batch identifier */
    uint64_t* task_ids;              /**< Array of dispatched task IDs */
    size_t num_tasks;                /**< Number of tasks in batch */
    size_t capacity;                 /**< Capacity of task_ids array */
    rcog_subtask_status_t* statuses; /**< Status of each task */
    size_t completed_count;          /**< Number of completed tasks */
    bool all_done;                   /**< True when all tasks complete */
    uint64_t created_ms;             /**< Creation timestamp */
};

/**
 * @brief Orchestrator internal structure
 */
struct rcog_orchestrator {
    /* Configuration */
    rcog_orchestrator_config_t config;

    /* Connections */
    struct rcog_context_store* context_store;
    struct rcog_answer_refiner* answer_refiner;
    struct rcog_delegation_pool* delegation_pool;
    struct rcog_imagination_bridge* imagination;
    struct rcog_immune_bridge* immune;
    struct rcog_engine* engine;

    /* Depth tracking */
    uint32_t current_depth;
    uint32_t effective_max_depth;
    uint32_t effective_max_parallel;

    /* Immune modulation */
    rcog_immune_modulation_t current_modulation;
    bool modulation_applied;

    /* Active decompositions */
    rcog_active_decomp_t active_decomps[RCOG_ORCH_MAX_ACTIVE_DECOMPOSITIONS];
    size_t num_active_decomps;
    uint64_t next_decomp_id;
    uint64_t next_task_id;

    /* Execution trace */
    rcog_trace_t* trace;
    bool trace_enabled;

    /* Statistics */
    rcog_orchestrator_stats_t stats;

    /* Thread safety */
    nimcp_mutex_t* mutex;
};

/*=============================================================================
 * INTERNAL HELPERS
 *===========================================================================*/

/**
 * @brief Add trace entry
 */
static void add_trace_entry(
    rcog_orchestrator_t* orch,
    uint64_t task_id,
    const char* event,
    float confidence
) {
    if (!orch->trace_enabled || !orch->trace) {
        return;
    }

    /* Grow trace if needed */
    if (orch->trace->num_entries >= orch->trace->capacity) {
        size_t new_cap = orch->trace->capacity * 2;
        if (new_cap == 0) new_cap = 64;
        rcog_trace_entry_t* new_entries = nimcp_realloc(
            orch->trace->entries,
            new_cap * sizeof(rcog_trace_entry_t)
        );
        if (!new_entries) return;
        orch->trace->entries = new_entries;
        orch->trace->capacity = new_cap;
    }

    rcog_trace_entry_t* entry = &orch->trace->entries[orch->trace->num_entries++];
    entry->timestamp_ms = nimcp_platform_time_monotonic_ms();
    entry->task_id = task_id;
    entry->event = event;
    entry->depth = orch->current_depth;
    entry->confidence = confidence;
}

/**
 * @brief Generate unique task ID
 */
static uint64_t generate_task_id(rcog_orchestrator_t* orch) {
    return orch->next_task_id++;
}

/**
 * @brief Generate unique decomposition ID
 */
static uint64_t generate_decomp_id(rcog_orchestrator_t* orch) {
    return orch->next_decomp_id++;
}

/**
 * @brief Check if task has dependencies satisfied
 */
static bool task_deps_satisfied(
    const rcog_decomposition_t* decomp,
    uint64_t task_id,
    const uint64_t* completed,
    size_t num_completed
) {
    if (!decomp->deps) {
        return true;  /* No dependencies */
    }

    for (size_t i = 0; i < decomp->deps->num_edges; i++) {
        if (decomp->deps->edges[i].to_task_id == task_id) {
            uint64_t dep_id = decomp->deps->edges[i].from_task_id;
            bool found = false;
            for (size_t j = 0; j < num_completed; j++) {
                if (completed[j] == dep_id) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                return false;
            }
        }
    }
    return true;
}

/**
 * @brief Detect cycles in dependency graph using DFS
 */
static bool detect_cycle_dfs(
    const rcog_dependency_graph_t* deps,
    const rcog_subtask_t* subtasks,
    size_t num_subtasks,
    uint64_t task_id,
    bool* visited,
    bool* in_stack
) {
    /* Find task index */
    size_t idx = SIZE_MAX;
    for (size_t i = 0; i < num_subtasks; i++) {
        if (subtasks[i].task_id == task_id) {
            idx = i;
            break;
        }
    }
    if (idx == SIZE_MAX) return false;

    if (in_stack[idx]) return true;  /* Cycle detected */
    if (visited[idx]) return false;  /* Already processed */

    visited[idx] = true;
    in_stack[idx] = true;

    /* Visit all tasks that depend on this one */
    for (size_t i = 0; i < deps->num_edges; i++) {
        if (deps->edges[i].from_task_id == task_id) {
            if (detect_cycle_dfs(deps, subtasks, num_subtasks,
                                 deps->edges[i].to_task_id, visited, in_stack)) {
                return true;
            }
        }
    }

    in_stack[idx] = false;
    return false;
}

/**
 * @brief Apply modulation to limits
 */
static void apply_modulation_to_limits(rcog_orchestrator_t* orch) {
    if (!orch->modulation_applied) {
        orch->effective_max_depth = orch->config.max_recursion_depth;
        orch->effective_max_parallel = orch->config.max_parallel_subtasks;
        return;
    }

    orch->effective_max_depth = (uint32_t)(
        (float)orch->config.max_recursion_depth *
        orch->current_modulation.max_depth_multiplier
    );
    if (orch->effective_max_depth < 1) {
        orch->effective_max_depth = 1;
    }

    orch->effective_max_parallel = (uint32_t)(
        (float)orch->config.max_parallel_subtasks *
        orch->current_modulation.parallelism_multiplier
    );
    if (orch->effective_max_parallel < 1) {
        orch->effective_max_parallel = 1;
    }
}

/*=============================================================================
 * LIFECYCLE
 *===========================================================================*/

rcog_orchestrator_config_t rcog_orchestrator_default_config(void) {
    rcog_orchestrator_config_t config = {0};

    config.max_recursion_depth = RCOG_DEFAULT_MAX_DEPTH;
    config.max_parallel_subtasks = RCOG_DEFAULT_MAX_PARALLEL_SUBTASKS;

    config.ready_threshold = RCOG_DEFAULT_READY_THRESHOLD;
    config.max_refinement_steps = RCOG_DEFAULT_MAX_REFINEMENT_STEPS;
    config.enable_early_termination = true;

    config.scheduling_policy = RCOG_SCHED_ADAPTIVE;
    config.decomposition_timeout_ms = RCOG_ORCH_DEFAULT_DECOMP_TIMEOUT_MS;

    config.default_strategy = RCOG_DECOMP_ADAPTIVE;
    config.enable_adaptive_strategy = true;

    config.enable_imagination_planning = true;
    config.enable_immune_modulation = true;

    config.enable_trace = false;
    config.verbose_logging = false;

    return config;
}

rcog_orchestrator_t* rcog_orchestrator_create(
    const rcog_orchestrator_config_t* config
) {
    rcog_orchestrator_t* orch = nimcp_calloc(1, sizeof(rcog_orchestrator_t));
    if (!orch) {
        return NULL;
    }

    if (config) {
        orch->config = *config;
    } else {
        orch->config = rcog_orchestrator_default_config();
    }

    mutex_attr_t attr = {0};
    attr.type = MUTEX_TYPE_NORMAL;
    orch->mutex = nimcp_mutex_create(&attr);
    if (!orch->mutex) {
        nimcp_free(orch);
        return NULL;
    }

    /* Initialize effective limits */
    orch->effective_max_depth = orch->config.max_recursion_depth;
    orch->effective_max_parallel = orch->config.max_parallel_subtasks;

    /* Initialize modulation to neutral */
    orch->current_modulation.capacity_multiplier = 1.0f;
    orch->current_modulation.max_depth_multiplier = 1.0f;
    orch->current_modulation.parallelism_multiplier = 1.0f;
    orch->current_modulation.timeout_multiplier = 1.0f;
    orch->current_modulation.enable_degraded_mode = false;

    /* Initialize trace if enabled */
    if (orch->config.enable_trace) {
        orch->trace = nimcp_calloc(1, sizeof(rcog_trace_t));
        orch->trace_enabled = true;
    }

    /* Initialize ID generators */
    orch->next_decomp_id = 1;
    orch->next_task_id = 1;

    return orch;
}

rcog_orchestrator_t* rcog_orchestrator_create_default(void) {
    return rcog_orchestrator_create(NULL);
}

void rcog_orchestrator_destroy(rcog_orchestrator_t* orch) {
    if (!orch) {
        return;
    }

    /* Free active decompositions */
    for (size_t i = 0; i < RCOG_ORCH_MAX_ACTIVE_DECOMPOSITIONS; i++) {
        if (orch->active_decomps[i].in_use && orch->active_decomps[i].decomp) {
            rcog_orchestrator_free_decomposition(orch->active_decomps[i].decomp);
        }
    }

    /* Free trace */
    if (orch->trace) {
        rcog_orchestrator_free_trace(orch->trace);
    }

    if (orch->mutex) {
        nimcp_mutex_destroy(orch->mutex);
    }

    nimcp_free(orch);
}

/*=============================================================================
 * CONNECTION
 *===========================================================================*/

int rcog_orchestrator_connect_context_store(
    rcog_orchestrator_t* orch,
    struct rcog_context_store* store
) {
    if (!orch || !store) {
        return RCOG_ERROR_NULL_POINTER;
    }

    nimcp_mutex_lock(orch->mutex);
    orch->context_store = store;
    nimcp_mutex_unlock(orch->mutex);

    return RCOG_OK;
}

int rcog_orchestrator_connect_answer_refiner(
    rcog_orchestrator_t* orch,
    struct rcog_answer_refiner* refiner
) {
    if (!orch || !refiner) {
        return RCOG_ERROR_NULL_POINTER;
    }

    nimcp_mutex_lock(orch->mutex);
    orch->answer_refiner = refiner;
    nimcp_mutex_unlock(orch->mutex);

    return RCOG_OK;
}

int rcog_orchestrator_connect_delegation_pool(
    rcog_orchestrator_t* orch,
    struct rcog_delegation_pool* pool
) {
    if (!orch || !pool) {
        return RCOG_ERROR_NULL_POINTER;
    }

    nimcp_mutex_lock(orch->mutex);
    orch->delegation_pool = pool;
    nimcp_mutex_unlock(orch->mutex);

    return RCOG_OK;
}

int rcog_orchestrator_connect_imagination(
    rcog_orchestrator_t* orch,
    struct rcog_imagination_bridge* imagination
) {
    if (!orch || !imagination) {
        return RCOG_ERROR_NULL_POINTER;
    }

    nimcp_mutex_lock(orch->mutex);
    orch->imagination = imagination;
    nimcp_mutex_unlock(orch->mutex);

    return RCOG_OK;
}

int rcog_orchestrator_connect_immune(
    rcog_orchestrator_t* orch,
    struct rcog_immune_bridge* immune
) {
    if (!orch || !immune) {
        return RCOG_ERROR_NULL_POINTER;
    }

    nimcp_mutex_lock(orch->mutex);
    orch->immune = immune;
    nimcp_mutex_unlock(orch->mutex);

    return RCOG_OK;
}

int rcog_orchestrator_connect_engine(
    rcog_orchestrator_t* orch,
    struct rcog_engine* engine
) {
    if (!orch || !engine) {
        return RCOG_ERROR_NULL_POINTER;
    }

    nimcp_mutex_lock(orch->mutex);
    orch->engine = engine;
    nimcp_mutex_unlock(orch->mutex);

    return RCOG_OK;
}

/*=============================================================================
 * DECOMPOSITION - SEQUENTIAL STRATEGY
 *===========================================================================*/

/**
 * @brief Create sequential decomposition
 *
 * Splits goal into ordered steps that execute one after another.
 */
static int decompose_sequential(
    rcog_orchestrator_t* orch,
    const rcog_goal_t* goal,
    const struct rcog_context_store* context,
    rcog_decomposition_t* result
) {
    (void)context;  /* May use in full implementation */

    /* Estimate number of subtasks based on goal type */
    size_t num_subtasks;
    switch (goal->type) {
        case RCOG_GOAL_QUESTION_ANSWERING:
            num_subtasks = 2;  /* Parse, Answer */
            break;
        case RCOG_GOAL_REASONING:
            num_subtasks = 3;  /* Analyze, Reason, Conclude */
            break;
        case RCOG_GOAL_PLANNING:
            num_subtasks = 4;  /* Analyze, Plan, Validate, Refine */
            break;
        default:
            num_subtasks = 2;
    }

    if (num_subtasks > RCOG_ORCH_MAX_SUBTASKS_PER_DECOMP) {
        num_subtasks = RCOG_ORCH_MAX_SUBTASKS_PER_DECOMP;
    }

    /* Allocate subtasks */
    result->subtasks = nimcp_calloc(num_subtasks, sizeof(rcog_subtask_t));
    if (!result->subtasks) {
        return RCOG_ERROR_OUT_OF_MEMORY;
    }
    result->num_subtasks = num_subtasks;

    /* Create subtasks */
    for (size_t i = 0; i < num_subtasks; i++) {
        rcog_subtask_t* subtask = &result->subtasks[i];
        subtask->task_id = generate_task_id(orch);
        subtask->goal = *goal;  /* Copy goal */
        subtask->tier = RCOG_TIER_L1_REASONING;
        subtask->status = RCOG_SUBTASK_PENDING;
        subtask->priority = goal->priority * (1.0f - 0.1f * (float)i);
        subtask->timeout_ms = goal->timeout_ms > 0 ? goal->timeout_ms / (uint32_t)num_subtasks :
                              orch->config.decomposition_timeout_ms;
        subtask->depth = orch->current_depth + 1;
    }

    /* Create sequential dependencies */
    if (num_subtasks > 1) {
        result->deps = nimcp_calloc(1, sizeof(rcog_dependency_graph_t));
        if (!result->deps) {
            nimcp_free(result->subtasks);
            result->subtasks = NULL;
            return RCOG_ERROR_OUT_OF_MEMORY;
        }

        for (size_t i = 0; i < num_subtasks - 1; i++) {
            result->deps->edges[i].from_task_id = result->subtasks[i].task_id;
            result->deps->edges[i].to_task_id = result->subtasks[i + 1].task_id;
            result->deps->edges[i].weight = 1.0f;
        }
        result->deps->num_edges = num_subtasks - 1;
        result->deps->has_cycle = false;
    }

    result->ready_for_dispatch = true;
    orch->stats.subtasks_created += num_subtasks;

    return RCOG_OK;
}

/*=============================================================================
 * DECOMPOSITION - PARALLEL STRATEGY
 *===========================================================================*/

/**
 * @brief Create parallel decomposition
 *
 * Splits goal into independent subtasks that can execute simultaneously.
 */
static int decompose_parallel(
    rcog_orchestrator_t* orch,
    const rcog_goal_t* goal,
    const struct rcog_context_store* context,
    rcog_decomposition_t* result
) {
    (void)context;

    /* Number of parallel subtasks based on context refs */
    size_t num_subtasks = goal->num_context_refs > 0 ?
                          goal->num_context_refs :
                          orch->effective_max_parallel;

    if (num_subtasks > orch->effective_max_parallel) {
        num_subtasks = orch->effective_max_parallel;
    }
    if (num_subtasks > RCOG_ORCH_MAX_SUBTASKS_PER_DECOMP) {
        num_subtasks = RCOG_ORCH_MAX_SUBTASKS_PER_DECOMP;
    }
    if (num_subtasks == 0) {
        num_subtasks = 1;
    }

    /* Allocate subtasks */
    result->subtasks = nimcp_calloc(num_subtasks, sizeof(rcog_subtask_t));
    if (!result->subtasks) {
        return RCOG_ERROR_OUT_OF_MEMORY;
    }
    result->num_subtasks = num_subtasks;

    /* Create independent subtasks */
    for (size_t i = 0; i < num_subtasks; i++) {
        rcog_subtask_t* subtask = &result->subtasks[i];
        subtask->task_id = generate_task_id(orch);
        subtask->goal = *goal;
        subtask->tier = RCOG_TIER_L2_PERCEPTION;
        subtask->status = RCOG_SUBTASK_PENDING;
        subtask->priority = goal->priority;
        subtask->timeout_ms = goal->timeout_ms > 0 ? goal->timeout_ms :
                              orch->config.decomposition_timeout_ms;
        subtask->depth = orch->current_depth + 1;
    }

    /* No dependencies for parallel execution */
    result->deps = NULL;
    result->ready_for_dispatch = true;
    orch->stats.subtasks_created += num_subtasks;

    return RCOG_OK;
}

/*=============================================================================
 * DECOMPOSITION - HIERARCHICAL STRATEGY
 *===========================================================================*/

/**
 * @brief Create hierarchical decomposition
 *
 * Creates a tree structure with coordinator and worker subtasks.
 */
static int decompose_hierarchical(
    rcog_orchestrator_t* orch,
    const rcog_goal_t* goal,
    const struct rcog_context_store* context,
    rcog_decomposition_t* result
) {
    (void)context;

    /* One coordinator + multiple workers */
    size_t num_workers = orch->effective_max_parallel / 2;
    if (num_workers < 1) num_workers = 1;
    if (num_workers > 4) num_workers = 4;

    size_t num_subtasks = 1 + num_workers;  /* Coordinator + workers */

    /* Allocate subtasks */
    result->subtasks = nimcp_calloc(num_subtasks, sizeof(rcog_subtask_t));
    if (!result->subtasks) {
        return RCOG_ERROR_OUT_OF_MEMORY;
    }
    result->num_subtasks = num_subtasks;

    /* Create coordinator (first subtask) */
    rcog_subtask_t* coordinator = &result->subtasks[0];
    coordinator->task_id = generate_task_id(orch);
    coordinator->goal = *goal;
    coordinator->tier = RCOG_TIER_ROOT;  /* Coordinator only */
    coordinator->status = RCOG_SUBTASK_PENDING;
    coordinator->priority = goal->priority;
    coordinator->timeout_ms = orch->config.decomposition_timeout_ms;
    coordinator->depth = orch->current_depth + 1;

    /* Create workers */
    for (size_t i = 0; i < num_workers; i++) {
        rcog_subtask_t* worker = &result->subtasks[1 + i];
        worker->task_id = generate_task_id(orch);
        worker->goal = *goal;
        worker->tier = RCOG_TIER_L1_REASONING;
        worker->status = RCOG_SUBTASK_PENDING;
        worker->priority = goal->priority * 0.9f;
        worker->timeout_ms = orch->config.decomposition_timeout_ms;
        worker->depth = orch->current_depth + 2;  /* Deeper than coordinator */
    }

    /* Dependencies: coordinator must complete setup before workers */
    /* Then workers run in parallel, then coordinator aggregates */
    result->deps = nimcp_calloc(1, sizeof(rcog_dependency_graph_t));
    if (!result->deps) {
        nimcp_free(result->subtasks);
        result->subtasks = NULL;
        return RCOG_ERROR_OUT_OF_MEMORY;
    }

    /* Coordinator spawns workers (implicit dependency through dispatch) */
    result->deps->num_edges = 0;  /* Workers are dispatched by coordinator */
    result->deps->has_cycle = false;

    result->ready_for_dispatch = true;
    orch->stats.subtasks_created += num_subtasks;

    return RCOG_OK;
}

/*=============================================================================
 * DECOMPOSITION - ADAPTIVE STRATEGY
 *===========================================================================*/

/**
 * @brief Create adaptive decomposition
 *
 * Selects strategy based on goal characteristics.
 */
static int decompose_adaptive(
    rcog_orchestrator_t* orch,
    const rcog_goal_t* goal,
    const struct rcog_context_store* context,
    rcog_decomposition_t* result
) {
    rcog_decomposition_strategy_t selected = RCOG_DECOMP_SEQUENTIAL;

    /* Heuristic selection */
    switch (goal->type) {
        case RCOG_GOAL_QUESTION_ANSWERING:
        case RCOG_GOAL_EXTRACTION:
            /* Simple goals: sequential */
            selected = RCOG_DECOMP_SEQUENTIAL;
            break;

        case RCOG_GOAL_ANALYSIS:
        case RCOG_GOAL_SUMMARIZATION:
            /* Multi-source goals: parallel if multiple context refs */
            if (goal->num_context_refs > 1) {
                selected = RCOG_DECOMP_PARALLEL;
            } else {
                selected = RCOG_DECOMP_SEQUENTIAL;
            }
            break;

        case RCOG_GOAL_REASONING:
        case RCOG_GOAL_PLANNING:
            /* Complex goals: hierarchical */
            if (orch->current_depth < orch->effective_max_depth - 2) {
                selected = RCOG_DECOMP_HIERARCHICAL;
            } else {
                selected = RCOG_DECOMP_SEQUENTIAL;
            }
            break;

        case RCOG_GOAL_GENERATION:
        case RCOG_GOAL_TRANSLATION:
            /* Creative goals: parallel for variety */
            selected = RCOG_DECOMP_PARALLEL;
            break;

        default:
            selected = RCOG_DECOMP_SEQUENTIAL;
    }

    /* Respect depth limits */
    if (orch->current_depth >= orch->effective_max_depth - 1) {
        selected = RCOG_DECOMP_SEQUENTIAL;
    }

    /* Apply degraded mode if immune modulation active */
    if (orch->current_modulation.enable_degraded_mode) {
        selected = RCOG_DECOMP_SEQUENTIAL;
    }

    orch->stats.strategy_adaptations++;

    /* Delegate to selected strategy */
    switch (selected) {
        case RCOG_DECOMP_PARALLEL:
            return decompose_parallel(orch, goal, context, result);
        case RCOG_DECOMP_HIERARCHICAL:
            return decompose_hierarchical(orch, goal, context, result);
        default:
            return decompose_sequential(orch, goal, context, result);
    }
}

/*=============================================================================
 * DECOMPOSITION - MAIN FUNCTIONS
 *===========================================================================*/

int rcog_orchestrator_decompose(
    rcog_orchestrator_t* orch,
    const rcog_goal_t* goal,
    const struct rcog_context_store* context,
    rcog_decomposition_t* result
) {
    if (!orch || !goal || !result) {
        return RCOG_ERROR_NULL_POINTER;
    }

    nimcp_mutex_lock(orch->mutex);

    /* Check depth limit */
    if (orch->current_depth >= orch->effective_max_depth) {
        orch->stats.depth_limit_hits++;
        nimcp_mutex_unlock(orch->mutex);
        return RCOG_ERROR_MAX_DEPTH_EXCEEDED;
    }

    /* Initialize result */
    memset(result, 0, sizeof(rcog_decomposition_t));
    result->metadata.decomp_id = generate_decomp_id(orch);
    result->metadata.depth = orch->current_depth;
    result->metadata.created_ms = nimcp_platform_time_monotonic_ms();

    rcog_decomposition_strategy_t strategy = orch->config.default_strategy;
    if (orch->config.enable_adaptive_strategy) {
        strategy = RCOG_DECOMP_ADAPTIVE;
    }
    result->metadata.strategy = strategy;

    add_trace_entry(orch, 0, "decompose_start", 0.0f);

    int err;
    switch (strategy) {
        case RCOG_DECOMP_SEQUENTIAL:
            err = decompose_sequential(orch, goal, context, result);
            break;
        case RCOG_DECOMP_PARALLEL:
            err = decompose_parallel(orch, goal, context, result);
            break;
        case RCOG_DECOMP_HIERARCHICAL:
            err = decompose_hierarchical(orch, goal, context, result);
            break;
        case RCOG_DECOMP_ADAPTIVE:
        default:
            err = decompose_adaptive(orch, goal, context, result);
            break;
    }

    if (err == RCOG_OK) {
        orch->stats.goals_decomposed++;
        result->metadata.estimated_complexity =
            (float)result->num_subtasks / (float)RCOG_ORCH_MAX_SUBTASKS_PER_DECOMP;
        add_trace_entry(orch, result->metadata.decomp_id, "decompose_complete",
                       result->metadata.estimated_complexity);
    }

    nimcp_mutex_unlock(orch->mutex);

    return err;
}

int rcog_orchestrator_decompose_with_strategy(
    rcog_orchestrator_t* orch,
    const rcog_goal_t* goal,
    const struct rcog_context_store* context,
    rcog_decomposition_strategy_t strategy,
    rcog_decomposition_t* result
) {
    if (!orch || !goal || !result) {
        return RCOG_ERROR_NULL_POINTER;
    }

    nimcp_mutex_lock(orch->mutex);

    if (orch->current_depth >= orch->effective_max_depth) {
        orch->stats.depth_limit_hits++;
        nimcp_mutex_unlock(orch->mutex);
        return RCOG_ERROR_MAX_DEPTH_EXCEEDED;
    }

    memset(result, 0, sizeof(rcog_decomposition_t));
    result->metadata.decomp_id = generate_decomp_id(orch);
    result->metadata.depth = orch->current_depth;
    result->metadata.strategy = strategy;
    result->metadata.created_ms = nimcp_platform_time_monotonic_ms();

    int err;
    switch (strategy) {
        case RCOG_DECOMP_SEQUENTIAL:
            err = decompose_sequential(orch, goal, context, result);
            break;
        case RCOG_DECOMP_PARALLEL:
            err = decompose_parallel(orch, goal, context, result);
            break;
        case RCOG_DECOMP_HIERARCHICAL:
            err = decompose_hierarchical(orch, goal, context, result);
            break;
        case RCOG_DECOMP_ADAPTIVE:
        default:
            err = decompose_adaptive(orch, goal, context, result);
            break;
    }

    if (err == RCOG_OK) {
        orch->stats.goals_decomposed++;
    }

    nimcp_mutex_unlock(orch->mutex);

    return err;
}

int rcog_orchestrator_decompose_with_hints(
    rcog_orchestrator_t* orch,
    const rcog_goal_t* goal,
    const struct rcog_context_store* context,
    const rcog_strategy_hints_t* hints,
    rcog_decomposition_t* result
) {
    if (!orch || !goal || !hints || !result) {
        return RCOG_ERROR_NULL_POINTER;
    }

    /* Select strategy based on hints */
    rcog_decomposition_strategy_t strategy;
    if (hints->prefer_parallel) {
        strategy = RCOG_DECOMP_PARALLEL;
    } else if (hints->prefer_shallow) {
        strategy = RCOG_DECOMP_SEQUENTIAL;
    } else {
        strategy = RCOG_DECOMP_ADAPTIVE;
    }

    return rcog_orchestrator_decompose_with_strategy(orch, goal, context, strategy, result);
}

void rcog_orchestrator_free_decomposition(rcog_decomposition_t* decomp) {
    if (!decomp) {
        return;
    }

    if (decomp->subtasks) {
        nimcp_free(decomp->subtasks);
        decomp->subtasks = NULL;
    }

    if (decomp->deps) {
        nimcp_free(decomp->deps);
        decomp->deps = NULL;
    }

    decomp->num_subtasks = 0;
}

/*=============================================================================
 * DISPATCH
 *===========================================================================*/

int rcog_orchestrator_dispatch(
    rcog_orchestrator_t* orch,
    const rcog_decomposition_t* decomp,
    rcog_batch_handle_t** handle
) {
    if (!orch || !decomp || !handle) {
        return RCOG_ERROR_NULL_POINTER;
    }

    nimcp_mutex_lock(orch->mutex);

    if (!decomp->ready_for_dispatch) {
        nimcp_mutex_unlock(orch->mutex);
        return RCOG_ERROR_NOT_INITIALIZED;
    }

    /* In full implementation, would submit to delegation pool */
    /* For now, mark as dispatched */
    orch->stats.subtasks_dispatched += decomp->num_subtasks;

    add_trace_entry(orch, decomp->metadata.decomp_id, "dispatch", 0.0f);

    /* Create placeholder handle */
    *handle = nimcp_calloc(1, sizeof(rcog_batch_handle_t));

    nimcp_mutex_unlock(orch->mutex);

    return RCOG_OK;
}

int rcog_orchestrator_dispatch_subtask(
    rcog_orchestrator_t* orch,
    rcog_subtask_t* subtask
) {
    if (!orch || !subtask) {
        return RCOG_ERROR_NULL_POINTER;
    }

    nimcp_mutex_lock(orch->mutex);

    subtask->status = RCOG_SUBTASK_QUEUED;
    subtask->started_ms = nimcp_platform_time_monotonic_ms();
    orch->stats.subtasks_dispatched++;

    add_trace_entry(orch, subtask->task_id, "dispatch_subtask", subtask->priority);

    nimcp_mutex_unlock(orch->mutex);

    return RCOG_OK;
}

int rcog_orchestrator_await_batch(
    rcog_orchestrator_t* orch,
    rcog_batch_handle_t* handle,
    uint32_t timeout_ms
) {
    if (!orch || !handle) {
        return RCOG_ERROR_NULL_POINTER;
    }

    /* In full implementation, would wait for delegation pool */
    (void)timeout_ms;

    return RCOG_OK;
}

/*=============================================================================
 * AGGREGATION
 *===========================================================================*/

int rcog_orchestrator_aggregate(
    rcog_orchestrator_t* orch,
    rcog_batch_handle_t* handle,
    const rcog_subtask_result_t* results,
    size_t num_results,
    rcog_answer_state_t* answer
) {
    if (!orch || !handle || !results || !answer) {
        return RCOG_ERROR_NULL_POINTER;
    }

    nimcp_mutex_lock(orch->mutex);

    /* Aggregate confidence from subtask results */
    float total_confidence = 0.0f;
    size_t successful = 0;

    for (size_t i = 0; i < num_results; i++) {
        if (results[i].success) {
            total_confidence += results[i].confidence;
            successful++;
            orch->stats.subtasks_completed++;
        } else {
            orch->stats.subtasks_failed++;
        }
    }

    if (successful > 0) {
        answer->confidence = total_confidence / (float)successful;
    }
    answer->refinement_step++;
    answer->last_updated_ms = nimcp_platform_time_monotonic_ms();

    /* Check if ready */
    if (answer->confidence >= orch->config.ready_threshold) {
        answer->ready = true;
        answer->status = RCOG_ANSWER_READY;
        if (orch->config.enable_early_termination) {
            orch->stats.early_terminations++;
        }
    } else {
        answer->status = RCOG_ANSWER_REFINING;
    }

    add_trace_entry(orch, 0, "aggregate", answer->confidence);

    nimcp_mutex_unlock(orch->mutex);

    return RCOG_OK;
}

int rcog_orchestrator_refine(
    rcog_orchestrator_t* orch,
    rcog_answer_state_t* answer,
    const struct rcog_context_store* context
) {
    if (!orch || !answer) {
        return RCOG_ERROR_NULL_POINTER;
    }

    (void)context;  /* Would use for additional queries in full implementation */

    nimcp_mutex_lock(orch->mutex);

    if (answer->ready) {
        nimcp_mutex_unlock(orch->mutex);
        return RCOG_OK;
    }

    if (answer->refinement_step >= orch->config.max_refinement_steps) {
        answer->status = RCOG_ANSWER_STALLED;
        nimcp_mutex_unlock(orch->mutex);
        return RCOG_ERROR_TIMEOUT;
    }

    /* Simulate refinement step */
    float old_confidence = answer->confidence;
    answer->confidence += 0.05f * (1.0f - answer->confidence);  /* Diminishing returns */
    answer->delta = answer->confidence - old_confidence;
    answer->refinement_step++;
    answer->last_updated_ms = nimcp_platform_time_monotonic_ms();

    if (answer->confidence >= orch->config.ready_threshold) {
        answer->ready = true;
        answer->status = RCOG_ANSWER_READY;
    } else if (answer->delta < RCOG_DEFAULT_CONVERGENCE_EPSILON) {
        answer->status = RCOG_ANSWER_CONVERGING;
    }

    add_trace_entry(orch, 0, "refine", answer->confidence);

    nimcp_mutex_unlock(orch->mutex);

    return RCOG_OK;
}

bool rcog_orchestrator_is_answer_ready(
    const rcog_orchestrator_t* orch,
    const rcog_answer_state_t* answer
) {
    if (!orch || !answer) {
        return false;
    }
    return answer->ready || answer->confidence >= orch->config.ready_threshold;
}

/*=============================================================================
 * DEPTH CONTROL
 *===========================================================================*/

uint32_t rcog_orchestrator_get_current_depth(const rcog_orchestrator_t* orch) {
    if (!orch) return 0;
    return orch->current_depth;
}

uint32_t rcog_orchestrator_get_max_depth(const rcog_orchestrator_t* orch) {
    if (!orch) return 0;
    return orch->effective_max_depth;
}

bool rcog_orchestrator_would_exceed_depth(
    const rcog_orchestrator_t* orch,
    uint32_t additional_depth
) {
    if (!orch) return true;
    return (orch->current_depth + additional_depth) > orch->effective_max_depth;
}

int rcog_orchestrator_push_depth(rcog_orchestrator_t* orch) {
    if (!orch) {
        return RCOG_ERROR_NULL_POINTER;
    }

    nimcp_mutex_lock(orch->mutex);

    if (orch->current_depth >= orch->effective_max_depth) {
        orch->stats.depth_limit_hits++;
        nimcp_mutex_unlock(orch->mutex);
        return RCOG_ERROR_MAX_DEPTH_EXCEEDED;
    }

    orch->current_depth++;
    if (orch->current_depth > orch->stats.max_depth_reached) {
        orch->stats.max_depth_reached = orch->current_depth;
    }

    nimcp_mutex_unlock(orch->mutex);

    return RCOG_OK;
}

void rcog_orchestrator_pop_depth(rcog_orchestrator_t* orch) {
    if (!orch) return;

    nimcp_mutex_lock(orch->mutex);
    if (orch->current_depth > 0) {
        orch->current_depth--;
    }
    nimcp_mutex_unlock(orch->mutex);
}

/*=============================================================================
 * IMMUNE MODULATION
 *===========================================================================*/

int rcog_orchestrator_apply_immune_modulation(
    rcog_orchestrator_t* orch,
    const rcog_immune_modulation_t* modulation
) {
    if (!orch || !modulation) {
        return RCOG_ERROR_NULL_POINTER;
    }

    nimcp_mutex_lock(orch->mutex);

    orch->current_modulation = *modulation;
    orch->modulation_applied = true;
    apply_modulation_to_limits(orch);

    add_trace_entry(orch, 0, "immune_modulation", modulation->capacity_multiplier);

    nimcp_mutex_unlock(orch->mutex);

    return RCOG_OK;
}

int rcog_orchestrator_get_effective_limits(
    const rcog_orchestrator_t* orch,
    uint32_t* max_depth,
    uint32_t* max_parallel
) {
    if (!orch || !max_depth || !max_parallel) {
        return RCOG_ERROR_NULL_POINTER;
    }

    nimcp_mutex_lock(((rcog_orchestrator_t*)orch)->mutex);
    *max_depth = orch->effective_max_depth;
    *max_parallel = orch->effective_max_parallel;
    nimcp_mutex_unlock(((rcog_orchestrator_t*)orch)->mutex);

    return RCOG_OK;
}

/*=============================================================================
 * STRATEGY SELECTION
 *===========================================================================*/

int rcog_orchestrator_select_strategy(
    rcog_orchestrator_t* orch,
    const rcog_goal_t* goal,
    const struct rcog_context_store* context,
    rcog_decomposition_strategy_t* strategy
) {
    if (!orch || !goal || !strategy) {
        return RCOG_ERROR_NULL_POINTER;
    }

    (void)context;

    nimcp_mutex_lock(orch->mutex);

    /* Use adaptive selection logic */
    switch (goal->type) {
        case RCOG_GOAL_REASONING:
        case RCOG_GOAL_PLANNING:
            *strategy = RCOG_DECOMP_HIERARCHICAL;
            break;
        case RCOG_GOAL_ANALYSIS:
        case RCOG_GOAL_SUMMARIZATION:
            *strategy = goal->num_context_refs > 1 ?
                        RCOG_DECOMP_PARALLEL : RCOG_DECOMP_SEQUENTIAL;
            break;
        default:
            *strategy = RCOG_DECOMP_SEQUENTIAL;
    }

    /* Respect modulation */
    if (orch->current_modulation.enable_degraded_mode) {
        *strategy = RCOG_DECOMP_SEQUENTIAL;
    }

    nimcp_mutex_unlock(orch->mutex);

    return RCOG_OK;
}

int rcog_orchestrator_recommend_strategies(
    rcog_orchestrator_t* orch,
    const rcog_goal_t* goal,
    rcog_decomposition_strategy_t* strategies,
    float* confidences,
    size_t max_strategies,
    size_t* num_strategies
) {
    if (!orch || !goal || !strategies || !confidences || !num_strategies) {
        return RCOG_ERROR_NULL_POINTER;
    }

    nimcp_mutex_lock(orch->mutex);

    size_t n = 0;

    /* Always recommend adaptive first */
    if (n < max_strategies) {
        strategies[n] = RCOG_DECOMP_ADAPTIVE;
        confidences[n] = 0.9f;
        n++;
    }

    /* Recommend based on goal type */
    if (goal->type == RCOG_GOAL_REASONING || goal->type == RCOG_GOAL_PLANNING) {
        if (n < max_strategies) {
            strategies[n] = RCOG_DECOMP_HIERARCHICAL;
            confidences[n] = 0.8f;
            n++;
        }
    }

    if (goal->num_context_refs > 1 && n < max_strategies) {
        strategies[n] = RCOG_DECOMP_PARALLEL;
        confidences[n] = 0.7f;
        n++;
    }

    if (n < max_strategies) {
        strategies[n] = RCOG_DECOMP_SEQUENTIAL;
        confidences[n] = 0.6f;
        n++;
    }

    *num_strategies = n;

    nimcp_mutex_unlock(orch->mutex);

    return RCOG_OK;
}

/*=============================================================================
 * SUB-ORCHESTRATORS
 *===========================================================================*/

int rcog_orchestrator_spawn_sub(
    rcog_orchestrator_t* orch,
    const rcog_subtask_t* parent_task,
    rcog_orchestrator_t** sub_orch
) {
    if (!orch || !parent_task || !sub_orch) {
        return RCOG_ERROR_NULL_POINTER;
    }

    nimcp_mutex_lock(orch->mutex);

    /* Check depth limit */
    if (orch->current_depth >= orch->effective_max_depth) {
        nimcp_mutex_unlock(orch->mutex);
        return RCOG_ERROR_MAX_DEPTH_EXCEEDED;
    }

    /* Create sub-orchestrator with reduced limits */
    rcog_orchestrator_config_t sub_config = orch->config;
    sub_config.max_recursion_depth = orch->effective_max_depth - orch->current_depth - 1;
    sub_config.max_parallel_subtasks = orch->effective_max_parallel / 2;
    if (sub_config.max_parallel_subtasks < 1) {
        sub_config.max_parallel_subtasks = 1;
    }

    *sub_orch = rcog_orchestrator_create(&sub_config);
    if (!*sub_orch) {
        nimcp_mutex_unlock(orch->mutex);
        return RCOG_ERROR_OUT_OF_MEMORY;
    }

    /* Inherit connections */
    (*sub_orch)->context_store = orch->context_store;
    (*sub_orch)->answer_refiner = orch->answer_refiner;
    (*sub_orch)->delegation_pool = orch->delegation_pool;
    (*sub_orch)->imagination = orch->imagination;
    (*sub_orch)->immune = orch->immune;
    (*sub_orch)->engine = orch->engine;

    /* Set initial depth */
    (*sub_orch)->current_depth = parent_task->depth;

    nimcp_mutex_unlock(orch->mutex);

    return RCOG_OK;
}

void rcog_orchestrator_destroy_sub(
    rcog_orchestrator_t* orch,
    rcog_orchestrator_t* sub_orch
) {
    (void)orch;  /* Parent reference not needed for cleanup */
    rcog_orchestrator_destroy(sub_orch);
}

/*=============================================================================
 * STATISTICS AND TRACE
 *===========================================================================*/

int rcog_orchestrator_get_stats(
    const rcog_orchestrator_t* orch,
    rcog_orchestrator_stats_t* stats
) {
    if (!orch || !stats) {
        return RCOG_ERROR_NULL_POINTER;
    }

    nimcp_mutex_lock(((rcog_orchestrator_t*)orch)->mutex);
    *stats = orch->stats;

    /* Compute averages */
    if (stats->goals_decomposed > 0) {
        stats->avg_subtasks_per_goal =
            (float)stats->subtasks_created / (float)stats->goals_decomposed;
    }

    nimcp_mutex_unlock(((rcog_orchestrator_t*)orch)->mutex);

    return RCOG_OK;
}

void rcog_orchestrator_reset_stats(rcog_orchestrator_t* orch) {
    if (!orch) return;

    nimcp_mutex_lock(orch->mutex);
    memset(&orch->stats, 0, sizeof(rcog_orchestrator_stats_t));
    nimcp_mutex_unlock(orch->mutex);
}

int rcog_orchestrator_get_trace(
    const rcog_orchestrator_t* orch,
    rcog_trace_t** trace
) {
    if (!orch || !trace) {
        return RCOG_ERROR_NULL_POINTER;
    }

    nimcp_mutex_lock(((rcog_orchestrator_t*)orch)->mutex);

    if (!orch->trace) {
        nimcp_mutex_unlock(((rcog_orchestrator_t*)orch)->mutex);
        *trace = NULL;
        return RCOG_OK;
    }

    /* Copy trace */
    rcog_trace_t* copy = nimcp_calloc(1, sizeof(rcog_trace_t));
    if (!copy) {
        nimcp_mutex_unlock(((rcog_orchestrator_t*)orch)->mutex);
        return RCOG_ERROR_OUT_OF_MEMORY;
    }

    copy->num_entries = orch->trace->num_entries;
    copy->capacity = orch->trace->num_entries;
    if (copy->num_entries > 0) {
        copy->entries = nimcp_calloc(copy->num_entries, sizeof(rcog_trace_entry_t));
        if (!copy->entries) {
            nimcp_free(copy);
            nimcp_mutex_unlock(((rcog_orchestrator_t*)orch)->mutex);
            return RCOG_ERROR_OUT_OF_MEMORY;
        }
        memcpy(copy->entries, orch->trace->entries,
               copy->num_entries * sizeof(rcog_trace_entry_t));
    }

    *trace = copy;

    nimcp_mutex_unlock(((rcog_orchestrator_t*)orch)->mutex);

    return RCOG_OK;
}

void rcog_orchestrator_free_trace(rcog_trace_t* trace) {
    if (!trace) return;

    if (trace->entries) {
        nimcp_free(trace->entries);
    }
    nimcp_free(trace);
}

void rcog_orchestrator_clear_trace(rcog_orchestrator_t* orch) {
    if (!orch) return;

    nimcp_mutex_lock(orch->mutex);
    if (orch->trace) {
        orch->trace->num_entries = 0;
    }
    nimcp_mutex_unlock(orch->mutex);
}

/*=============================================================================
 * UTILITY FUNCTIONS
 *===========================================================================*/

int rcog_orchestrator_validate_decomposition(const rcog_decomposition_t* decomp) {
    if (!decomp) {
        return RCOG_ERROR_NULL_POINTER;
    }

    if (!decomp->subtasks || decomp->num_subtasks == 0) {
        return RCOG_ERROR_INVALID_CONFIG;
    }

    /* Check for cycles if dependencies exist */
    if (decomp->deps && decomp->deps->num_edges > 0) {
        bool* visited = nimcp_calloc(decomp->num_subtasks, sizeof(bool));
        bool* in_stack = nimcp_calloc(decomp->num_subtasks, sizeof(bool));

        if (!visited || !in_stack) {
            if (visited) nimcp_free(visited);
            if (in_stack) nimcp_free(in_stack);
            return RCOG_ERROR_OUT_OF_MEMORY;
        }

        bool has_cycle = false;
        for (size_t i = 0; i < decomp->num_subtasks && !has_cycle; i++) {
            if (!visited[i]) {
                has_cycle = detect_cycle_dfs(decomp->deps, decomp->subtasks,
                                             decomp->num_subtasks,
                                             decomp->subtasks[i].task_id,
                                             visited, in_stack);
            }
        }

        nimcp_free(visited);
        nimcp_free(in_stack);

        if (has_cycle) {
            return RCOG_ERROR_INVALID_CONFIG;
        }
    }

    return RCOG_OK;
}

int rcog_orchestrator_get_topological_order(
    const rcog_decomposition_t* decomp,
    uint64_t* order,
    size_t max_order,
    size_t* num_order
) {
    if (!decomp || !order || !num_order) {
        return RCOG_ERROR_NULL_POINTER;
    }

    if (decomp->num_subtasks > max_order) {
        return RCOG_ERROR_CONTEXT_FULL;
    }

    /* Simple case: no dependencies */
    if (!decomp->deps || decomp->deps->num_edges == 0) {
        for (size_t i = 0; i < decomp->num_subtasks; i++) {
            order[i] = decomp->subtasks[i].task_id;
        }
        *num_order = decomp->num_subtasks;
        return RCOG_OK;
    }

    /* Kahn's algorithm for topological sort */
    int* in_degree = nimcp_calloc(decomp->num_subtasks, sizeof(int));
    if (!in_degree) {
        return RCOG_ERROR_OUT_OF_MEMORY;
    }

    /* Calculate in-degrees */
    for (size_t i = 0; i < decomp->deps->num_edges; i++) {
        uint64_t to_id = decomp->deps->edges[i].to_task_id;
        for (size_t j = 0; j < decomp->num_subtasks; j++) {
            if (decomp->subtasks[j].task_id == to_id) {
                in_degree[j]++;
                break;
            }
        }
    }

    /* Process nodes with zero in-degree */
    size_t output_idx = 0;
    bool progress = true;
    while (progress && output_idx < decomp->num_subtasks) {
        progress = false;
        for (size_t i = 0; i < decomp->num_subtasks; i++) {
            if (in_degree[i] == 0) {
                order[output_idx++] = decomp->subtasks[i].task_id;
                in_degree[i] = -1;  /* Mark as processed */
                progress = true;

                /* Reduce in-degree of dependent tasks */
                for (size_t j = 0; j < decomp->deps->num_edges; j++) {
                    if (decomp->deps->edges[j].from_task_id == decomp->subtasks[i].task_id) {
                        uint64_t to_id = decomp->deps->edges[j].to_task_id;
                        for (size_t k = 0; k < decomp->num_subtasks; k++) {
                            if (decomp->subtasks[k].task_id == to_id && in_degree[k] > 0) {
                                in_degree[k]--;
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    nimcp_free(in_degree);
    *num_order = output_idx;

    return RCOG_OK;
}

int rcog_orchestrator_get_ready_subtasks(
    const rcog_decomposition_t* decomp,
    const uint64_t* completed,
    size_t num_completed,
    uint64_t* ready,
    size_t max_ready,
    size_t* num_ready
) {
    if (!decomp || !ready || !num_ready) {
        return RCOG_ERROR_NULL_POINTER;
    }

    size_t n = 0;
    for (size_t i = 0; i < decomp->num_subtasks && n < max_ready; i++) {
        /* Skip already completed */
        bool is_completed = false;
        for (size_t j = 0; j < num_completed; j++) {
            if (completed[j] == decomp->subtasks[i].task_id) {
                is_completed = true;
                break;
            }
        }
        if (is_completed) continue;

        /* Check if dependencies satisfied */
        if (task_deps_satisfied(decomp, decomp->subtasks[i].task_id,
                                completed, num_completed)) {
            ready[n++] = decomp->subtasks[i].task_id;
        }
    }

    *num_ready = n;
    return RCOG_OK;
}

int rcog_orchestrator_estimate_complexity(
    rcog_orchestrator_t* orch,
    const rcog_goal_t* goal,
    const struct rcog_context_store* context,
    float* complexity
) {
    if (!orch || !goal || !complexity) {
        return RCOG_ERROR_NULL_POINTER;
    }

    (void)context;

    /* Simple heuristic based on goal type and context */
    float base = 0.3f;  /* Base complexity */

    switch (goal->type) {
        case RCOG_GOAL_QUESTION_ANSWERING:
        case RCOG_GOAL_EXTRACTION:
            base = 0.2f;
            break;
        case RCOG_GOAL_SUMMARIZATION:
        case RCOG_GOAL_TRANSLATION:
            base = 0.4f;
            break;
        case RCOG_GOAL_REASONING:
        case RCOG_GOAL_ANALYSIS:
            base = 0.6f;
            break;
        case RCOG_GOAL_PLANNING:
        case RCOG_GOAL_GENERATION:
            base = 0.8f;
            break;
        default:
            base = 0.5f;
    }

    /* Adjust for context refs */
    if (goal->num_context_refs > 1) {
        base += 0.1f * (float)(goal->num_context_refs - 1);
    }

    if (base > 1.0f) base = 1.0f;

    *complexity = base;
    return RCOG_OK;
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

int rcog_orchestrator_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;
    const kg_entity_t* self = kg_reader_get_entity(kg, "Recursive_Cognition_Orchestrator_Module");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            /* Log self-knowledge observations */
        }
    }
    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Recursive_Cognition_Orchestrator_Module");
    if (connections) { kg_relation_list_destroy(connections); }
    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Recursive_Cognition_Orchestrator_Module");
    if (incoming) { kg_relation_list_destroy(incoming); }
    return self ? 1 : 0;
}
