/**
 * @file nimcp_executive.h
 * @brief Executive functions module - task switching, planning, inhibition
 *
 * WHAT: Dorsolateral prefrontal cortex (DLPFC) executive control
 * WHY:  Enable goal-directed behavior, multi-tasking, and impulse control
 * HOW:  Task queue, attention allocation, inhibitory control
 *
 * BIOLOGICAL BASIS:
 * - DLPFC coordinates task switching (switch cost ~100-500ms)
 * - Inhibitory control prevents prepotent responses
 * - Planning decomposes goals into action sequences
 * - Working memory capacity limits parallel tasks
 *
 * PHASE: 10.3 (Executive Functions)
 * DEPENDENCIES: Working Memory (Phase 10.1)
 * TRAINING_IMPACT: None (inference-only, task management)
 *
 * @author Claude Code
 * @date 2025-11
 */

#ifndef NIMCP_EXECUTIVE_H
#define NIMCP_EXECUTIVE_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// Task Types
//=============================================================================

/**
 * @brief Task types that executive controller can manage
 */
typedef enum {
    TASK_TYPE_CLASSIFICATION,    /**< Multi-class classification task */
    TASK_TYPE_REGRESSION,        /**< Continuous value prediction */
    TASK_TYPE_SEQUENCE,          /**< Temporal sequence processing */
    TASK_TYPE_PLANNING,          /**< Multi-step goal planning */
    TASK_TYPE_REASONING,         /**< Logical reasoning */
    TASK_TYPE_MEMORY_RETRIEVAL,  /**< Recall from knowledge base */
    TASK_TYPE_CUSTOM             /**< User-defined task */
} task_type_t;

/**
 * @brief Task priority levels
 */
typedef enum {
    PRIORITY_LOW = 0,     /**< Background tasks */
    PRIORITY_NORMAL = 1,  /**< Default priority */
    PRIORITY_HIGH = 2,    /**< Important tasks */
    PRIORITY_URGENT = 3,  /**< Immediate attention required */
    PRIORITY_CRITICAL = 4 /**< Safety-critical tasks */
} task_priority_t;

/**
 * @brief Task status
 */
typedef enum {
    TASK_STATUS_PENDING,   /**< Queued, not started */
    TASK_STATUS_ACTIVE,    /**< Currently executing */
    TASK_STATUS_SUSPENDED, /**< Paused mid-execution */
    TASK_STATUS_COMPLETED, /**< Successfully finished */
    TASK_STATUS_FAILED,    /**< Execution failed */
    TASK_STATUS_ABORTED    /**< Manually cancelled */
} task_status_t;

/**
 * @brief Task descriptor
 */
typedef struct {
    uint32_t task_id;            /**< Unique task identifier */
    task_type_t type;            /**< Task type */
    task_priority_t priority;    /**< Execution priority */
    task_status_t status;        /**< Current status */

    char name[64];               /**< Human-readable name */
    void* context;               /**< Task-specific context */

    uint64_t created_ms;         /**< When task was created */
    uint64_t started_ms;         /**< When execution started (0 if not started) */
    uint64_t completed_ms;       /**< When execution finished (0 if not finished) */
    uint64_t deadline_ms;        /**< Deadline for completion (0 = no deadline) */

    uint32_t steps_total;        /**< Total steps in plan (0 if not a plan) */
    uint32_t steps_completed;    /**< Steps completed so far */
} task_descriptor_t;

//=============================================================================
// Executive Controller
//=============================================================================

/**
 * @brief Opaque executive controller handle
 */
typedef struct executive_controller executive_controller_t;

/**
 * @brief Executive controller configuration
 */
typedef struct {
    uint32_t max_tasks;               /**< Maximum queued tasks (default: 16) */
    float task_switch_cost_ms;        /**< Time penalty for switching (default: 200ms) */
    float inhibition_threshold;       /**< Threshold for inhibitory control (default: 0.7) */
    uint32_t max_plan_depth;          /**< Maximum planning depth (default: 10) */
    bool enable_task_prioritization;  /**< Enable priority-based scheduling */
    bool enable_deadline_checking;    /**< Enable deadline violations */
} executive_config_t;

/**
 * @brief Create executive controller with default configuration
 *
 * WHAT: Initialize executive control center
 * WHY:  Enable multi-tasking and goal-directed behavior
 * HOW:  Allocate task queue and control structures
 *
 * @return Executive controller handle or NULL on error
 *
 * COMPLEXITY: O(1)
 * THREAD-SAFE: Yes (initialization)
 * MALLOC: Yes (controller structure + task queue)
 */
executive_controller_t* executive_create(void);

/**
 * @brief Create executive controller with custom configuration
 *
 * @param config Custom configuration
 * @return Executive controller handle or NULL on error
 *
 * COMPLEXITY: O(1)
 * THREAD-SAFE: Yes (initialization)
 */
executive_controller_t* executive_create_custom(const executive_config_t* config);

/**
 * @brief Destroy executive controller
 *
 * @param exec Executive controller to destroy
 *
 * COMPLEXITY: O(n) where n = number of active tasks
 * THREAD-SAFE: No (caller must ensure exclusive access)
 */
void executive_destroy(executive_controller_t* exec);

//=============================================================================
// Task Management
//=============================================================================

/**
 * @brief Add task to queue
 *
 * WHAT: Register new task for execution
 * WHY:  Enable multi-task coordination
 * HOW:  Insert into priority queue based on priority and deadline
 *
 * @param exec Executive controller
 * @param task Task descriptor
 * @return Task ID (>0) on success, 0 on error
 *
 * COMPLEXITY: O(log n) for priority queue insertion
 * THREAD-SAFE: No
 */
uint32_t executive_add_task(executive_controller_t* exec, const task_descriptor_t* task);

/**
 * @brief Switch to different task
 *
 * WHAT: Change active task
 * WHY:  Support multi-tasking and interruption handling
 * HOW:  Suspend current task, activate new task, record switch cost
 *
 * @param exec Executive controller
 * @param task_id ID of task to switch to
 * @param current_time_ms Current time (for switch cost tracking)
 * @return true on success, false on error
 *
 * COMPLEXITY: O(log n)
 * THREAD-SAFE: No
 * SIDE_EFFECT: Increments switch count, records latency
 */
bool executive_switch_task(executive_controller_t* exec, uint32_t task_id, uint64_t current_time_ms);

/**
 * @brief Get currently active task
 *
 * @param exec Executive controller
 * @return Active task descriptor or NULL if no active task
 *
 * COMPLEXITY: O(1)
 * THREAD-SAFE: Yes (read-only)
 */
const task_descriptor_t* executive_get_active_task(executive_controller_t* exec);

/**
 * @brief Complete current task
 *
 * @param exec Executive controller
 * @param success true if successful, false if failed
 * @param current_time_ms Current time
 * @return true on success, false on error
 *
 * COMPLEXITY: O(1)
 * THREAD-SAFE: No
 */
bool executive_complete_task(executive_controller_t* exec, bool success, uint64_t current_time_ms);

//=============================================================================
// Inhibitory Control
//=============================================================================

/**
 * @brief Inhibit a response
 *
 * WHAT: Suppress prepotent/automatic responses
 * WHY:  Enable impulse control and ethical behavior
 * HOW:  Check salience against inhibition threshold
 *
 * @param exec Executive controller
 * @param response_salience How strong is the prepotent response [0,1]
 * @param reason Optional reason for inhibition (can be NULL)
 * @return true if response should be inhibited, false if allowed
 *
 * COMPLEXITY: O(1)
 * THREAD-SAFE: Yes (stateless)
 * EXAMPLE:
 *   if (executive_should_inhibit(exec, 0.9f, "unethical")) {
 *       return; // Suppress unethical response
 *   }
 */
bool executive_should_inhibit(executive_controller_t* exec, float response_salience, const char* reason);

//=============================================================================
// Planning
//=============================================================================

/**
 * @brief Plan type
 */
typedef enum {
    PLAN_TYPE_SEQUENTIAL,   /**< Linear sequence of steps */
    PLAN_TYPE_BRANCHING,    /**< Conditional branching */
    PLAN_TYPE_HIERARCHICAL  /**< Nested sub-goals */
} plan_type_t;

/**
 * @brief Plan step
 */
typedef struct {
    char description[128];   /**< What to do */
    void* action_data;       /**< Action-specific data */
    uint32_t estimated_cost; /**< Estimated time/resources */
    bool is_critical;        /**< Must succeed for plan to work */
} plan_step_t;

/**
 * @brief Plan structure
 */
typedef struct {
    plan_type_t type;
    uint32_t num_steps;
    plan_step_t* steps;
    char goal[128];          /**< Overall goal description */
} plan_t;

/**
 * @brief Create plan to achieve goal
 *
 * WHAT: Decompose goal into action sequence
 * WHY:  Enable goal-directed behavior
 * HOW:  Forward search with cost estimation
 *
 * @param exec Executive controller
 * @param goal Goal description
 * @param max_steps Maximum plan length
 * @return Plan structure or NULL on error
 *
 * COMPLEXITY: O(b^d) where b=branching factor, d=max_steps
 * THREAD-SAFE: No
 * MALLOC: Yes (plan structure + steps)
 */
plan_t* executive_create_plan(executive_controller_t* exec, const char* goal, uint32_t max_steps);

/**
 * @brief Destroy plan
 *
 * @param plan Plan to destroy
 *
 * COMPLEXITY: O(n) where n = number of steps
 * THREAD-SAFE: Yes
 */
void executive_destroy_plan(plan_t* plan);

//=============================================================================
// Statistics
//=============================================================================

/**
 * @brief Executive function statistics
 */
typedef struct {
    uint32_t total_tasks;           /**< Total tasks processed */
    uint32_t completed_tasks;       /**< Successfully completed */
    uint32_t failed_tasks;          /**< Failed tasks */
    uint32_t aborted_tasks;         /**< Aborted tasks */

    uint32_t total_switches;        /**< Total task switches */
    float avg_switch_cost_ms;       /**< Average switching latency */

    uint32_t inhibitions;           /**< Total inhibited responses */
    float inhibition_rate;          /**< Inhibitions / total decisions */

    uint32_t plans_created;         /**< Total plans generated */
    float avg_plan_length;          /**< Average steps per plan */
} executive_stats_t;

/**
 * @brief Get executive function statistics
 *
 * @param exec Executive controller
 * @param stats Output statistics structure
 * @return true on success, false on error
 *
 * COMPLEXITY: O(1)
 * THREAD-SAFE: Yes (read-only)
 */
bool executive_get_stats(executive_controller_t* exec, executive_stats_t* stats);

/**
 * @brief Reset statistics
 *
 * @param exec Executive controller
 *
 * COMPLEXITY: O(1)
 * THREAD-SAFE: No
 */
void executive_reset_stats(executive_controller_t* exec);

//=============================================================================
// Error Handling
//=============================================================================

/**
 * @brief Get last error message
 *
 * @return Error message string (valid until next API call)
 *
 * COMPLEXITY: O(1)
 * THREAD-SAFE: No (thread-local storage)
 */
const char* executive_get_last_error(void);

//=============================================================================
// Bidirectional Feedback Functions (Phase 10.11.3)
//=============================================================================

/**
 * @brief Get cognitive load (utilization)
 *
 * WHAT: Query current cognitive load on executive system
 * WHY:  Other modules can adapt behavior based on load
 * HOW:  Return task count / capacity ratio
 *
 * BIOLOGY: Prefrontal cortex has limited capacity
 *
 * @param exec Executive controller
 * @return Cognitive load [0, 1] (0=idle, 1=saturated)
 */
float executive_get_cognitive_load(executive_controller_t* exec);

/**
 * @brief Boost task priority based on external signal
 *
 * WHAT: Allow modules to boost task priority
 * WHY:  Curiosity-driven tasks should be prioritized when informative
 * HOW:  Increase priority by boost_amount
 *
 * @param exec Executive controller
 * @param task_name Task name to boost
 * @param boost_amount Priority boost [0, 1]
 * @return true if task found and boosted
 */
bool executive_boost_task_priority(executive_controller_t* exec,
                                    const char* task_name,
                                    float boost_amount);

#ifdef __cplusplus
}
#endif

#endif /* NIMCP_EXECUTIVE_H */
