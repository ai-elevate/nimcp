/**
 * @file nimcp_executive.c
 * @brief Executive functions implementation - task switching, planning, inhibition
 *
 * WHAT: Dorsolateral prefrontal cortex (DLPFC) executive control
 * WHY:  Enable goal-directed behavior, multi-tasking, and impulse control
 * HOW:  Task queue with priority scheduling, switch cost tracking, planning
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

#include "cognitive/nimcp_executive.h"
#include "utils/memory/nimcp_memory.h"  // nimcp_malloc/nimcp_free
#include "utils/time/nimcp_time.h"       // get_current_time_ms
#include "plasticity/neuromodulators/nimcp_neuromodulators.h"  // Neuromodulator integration
#include "core/brain/nimcp_brain.h"      // Brain reference
#include <string.h>
#include <stdio.h>
#include <stdarg.h>

//=============================================================================
// Constants
//=============================================================================

#define DEFAULT_MAX_TASKS 16
#define DEFAULT_SWITCH_COST_MS 200.0f
#define DEFAULT_INHIBITION_THRESHOLD 0.7f
#define DEFAULT_MAX_PLAN_DEPTH 10

//=============================================================================
// Internal Structures
//=============================================================================

/**
 * @brief Executive controller internal structure
 */
struct executive_controller {
    // Task management
    task_descriptor_t** task_queue;  /**< Array of task pointers */
    uint32_t max_tasks;              /**< Maximum queue size */
    uint32_t num_tasks;              /**< Current tasks in queue */
    task_descriptor_t* active_task;  /**< Currently executing task */

    // Configuration
    executive_config_t config;

    // Statistics
    executive_stats_t stats;

    // Switch tracking
    uint64_t last_switch_time_ms;
    uint32_t next_task_id;

    // Inhibition tracking
    uint32_t total_decisions;

    // Neuromodulation (Phase 10.x - Chemical modulation integration)
    brain_t brain;  /**< Brain reference for reading neurotransmitters */
};

//=============================================================================
// Error Handling
//=============================================================================

static __thread char last_error[256] = {0};

static void set_error(const char* fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    vsnprintf(last_error, sizeof(last_error), fmt, args);
    va_end(args);
}

const char* executive_get_last_error(void)
{
    return last_error;
}

//=============================================================================
// Helper Functions
//=============================================================================

/**
 * @brief Get current time in milliseconds (wrapper for platform API)
 *
 * WHAT: Get monotonic time suitable for task switching measurement
 * WHY:  Track task switch times and compute latency
 * HOW:  Delegate to NIMCP time utilities
 *
 * COMPLEXITY: O(1)
 * THREAD-SAFE: Yes
 *
 * @return Current time in milliseconds since boot
 */
static uint64_t get_current_time_ms(void)
{
    return nimcp_time_monotonic_ms();
}

/**
 * @brief Find task descriptor by unique task ID
 *
 * WHAT: Locate task in queue or active slot by ID
 * WHY:  Enable task lookup for switching and status queries
 * HOW:  Linear search through active task and queue
 *
 * COMPLEXITY: O(n) where n = number of queued tasks
 * THREAD-SAFE: No (caller must ensure exclusive access)
 *
 * @param exec Executive controller instance (non-NULL)
 * @param task_id Unique task identifier
 * @return Task descriptor pointer or NULL if not found
 *
 * @note Checks active task first (common case optimization)
 */
static task_descriptor_t* find_task_by_id(executive_controller_t* exec, uint32_t task_id)
{
    if (!exec) return NULL;

    // Check active task first
    if (exec->active_task && exec->active_task->task_id == task_id) {
        return exec->active_task;
    }

    // Search queue
    for (uint32_t i = 0; i < exec->num_tasks; i++) {
        if (exec->task_queue[i] && exec->task_queue[i]->task_id == task_id) {
            return exec->task_queue[i];
        }
    }

    return NULL;
}

/**
 * @brief Select highest priority pending task from queue
 *
 * WHAT: Find next task to execute based on priority and deadline
 * WHY:  Support priority-based task scheduling
 * HOW:  Linear scan for max priority, deadline as tiebreaker
 *
 * ALGORITHM:
 * 1. Scan all tasks in queue
 * 2. Select highest priority (PENDING status only)
 * 3. If tied, choose earliest deadline
 * 4. Return selected task (or NULL if queue empty)
 *
 * COMPLEXITY: O(n) where n = number of queued tasks
 * THREAD-SAFE: No
 *
 * @param exec Executive controller instance
 * @return Highest priority task or NULL if no pending tasks
 *
 * @note Only considers tasks with TASK_STATUS_PENDING
 */
static task_descriptor_t* get_highest_priority_task(executive_controller_t* exec)
{
    if (!exec || exec->num_tasks == 0) return NULL;

    task_descriptor_t* best = NULL;
    uint32_t best_idx = 0;

    for (uint32_t i = 0; i < exec->num_tasks; i++) {
        task_descriptor_t* task = exec->task_queue[i];
        if (!task || task->status != TASK_STATUS_PENDING) continue;

        if (!best || task->priority > best->priority) {
            best = task;
            best_idx = i;
        } else if (task->priority == best->priority) {
            // Same priority - check deadline
            if (task->deadline_ms > 0 && (best->deadline_ms == 0 || task->deadline_ms < best->deadline_ms)) {
                best = task;
                best_idx = i;
            }
        }
    }

    (void)best_idx; // Used for future optimization
    return best;
}

/**
 * @brief Compute dopamine-modulated task switch cost
 *
 * WHAT: Adjust switch cost based on dopamine level
 * WHY:  Dopamine affects cognitive flexibility and switch cost
 * HOW:  Read DA, modulate base cost, clamp to reasonable range
 *
 * BIOLOGY: High DA → easier switching (lower cost)
 *          Low DA → harder switching (higher cost, perseveration)
 *
 * COMPLEXITY: O(1)
 *
 * @param exec Executive controller
 * @param base_cost Base switch cost (ms)
 * @return Modulated switch cost (ms)
 */
static float compute_modulated_switch_cost(executive_controller_t* exec,
                                           float base_cost)
{
    // Guard: Early return if no brain
    if (!exec || !exec->brain) {
        return base_cost;
    }

    neuromodulator_system_t neuromod = brain_get_neuromodulator_system(exec->brain);
    if (!neuromod) {
        return base_cost;
    }

    // Read dopamine level
    float da = neuromodulator_get_level(neuromod, NEUROMOD_DOPAMINE);

    // DA range [0.3, 0.7], map to cost multiplier [1.4, 0.6]
    // High DA (0.7) → 0.6× cost (flexible, easy switching)
    // Low DA (0.3) → 1.4× cost (rigid, perseverative)
    float multiplier = 1.4f - (da - 0.3f) * 2.0f;

    return base_cost * multiplier;
}

//=============================================================================
// Creation & Destruction
//=============================================================================

executive_controller_t* executive_create(void)
{
    executive_config_t default_config = {
        .max_tasks = DEFAULT_MAX_TASKS,
        .task_switch_cost_ms = DEFAULT_SWITCH_COST_MS,
        .inhibition_threshold = DEFAULT_INHIBITION_THRESHOLD,
        .max_plan_depth = DEFAULT_MAX_PLAN_DEPTH,
        .enable_task_prioritization = true,
        .enable_deadline_checking = true
    };

    return executive_create_custom(&default_config);
}

/**
 * @brief Create executive controller with custom configuration
 *
 * WHAT: Initialize executive control center with specified parameters
 * WHY:  Enable goal-directed behavior and multi-tasking
 * HOW:  Allocate controller struct, task queue, initialize state
 *
 * ALGORITHM:
 * 1. Validate configuration parameters
 * 2. Allocate executive controller structure
 * 3. Allocate task queue array
 * 4. Initialize statistics and timing
 * 5. Return initialized controller
 *
 * COMPLEXITY: O(1) - constant allocations
 * MEMORY: sizeof(executive_controller_t) + max_tasks * sizeof(task_descriptor_t*)
 *
 * @param config Custom configuration (non-NULL)
 * @return Executive controller handle or NULL on error
 *
 * @note Caller must call executive_destroy() to free resources
 * @note Use executive_create() for default configuration
 */
executive_controller_t* executive_create_custom(const executive_config_t* config)
{
    // =========================================================================
    // GUARD: Validate configuration
    // =========================================================================

    // Guard: NULL config check
    if (!config) {
        set_error("NULL config provided to executive_create_custom");
        return NULL;
    }

    // Guard: Task queue size validation
    if (config->max_tasks == 0 || config->max_tasks > 1024) {
        set_error("Invalid max_tasks: %u (must be 1-1024)", config->max_tasks);
        return NULL;
    }

    // =========================================================================
    // ALLOCATION: Create controller structure
    // =========================================================================

    executive_controller_t* exec = nimcp_calloc(1, sizeof(executive_controller_t));
    if (!exec) {
        set_error("Failed to allocate executive controller (%zu bytes)",
                  sizeof(executive_controller_t));
        return NULL;
    }

    // =========================================================================
    // ALLOCATION: Create task queue
    // =========================================================================

    exec->task_queue = nimcp_calloc(config->max_tasks, sizeof(task_descriptor_t*));
    if (!exec->task_queue) {
        set_error("Failed to allocate task queue (%zu bytes)",
                  config->max_tasks * sizeof(task_descriptor_t*));
        nimcp_free(exec);  // Cleanup before return
        return NULL;
    }

    // =========================================================================
    // INITIALIZATION: Set up state
    // =========================================================================

    exec->config = *config;
    exec->max_tasks = config->max_tasks;
    exec->num_tasks = 0;
    exec->active_task = NULL;
    exec->next_task_id = 1;
    exec->last_switch_time_ms = get_current_time_ms();
    exec->total_decisions = 0;

    memset(&exec->stats, 0, sizeof(executive_stats_t));
    exec->brain = NULL;  // Initialize to NULL

    return exec;
}

/**
 * @brief Set brain reference for neuromodulator integration
 *
 * WHAT: Associate executive controller with brain for chemical modulation
 * WHY:  Enable neurotransmitter-based modulation of executive functions
 * HOW:  Store brain reference in controller structure
 *
 * USAGE: Call after creation to enable neuromodulation
 *
 * @param exec Executive controller
 * @param brain Brain handle (can be NULL to disable neuromodulation)
 */
void executive_set_brain(executive_controller_t* exec, brain_t brain)
{
    if (!exec) {
        return;
    }

    exec->brain = brain;
}

/**
 * @brief Destroy executive controller and free all resources
 *
 * WHAT: Clean up executive controller, tasks, and queue
 * WHY:  Prevent memory leaks
 * HOW:  Free all queued tasks, active task, queue array, controller
 *
 * COMPLEXITY: O(n) where n = number of tasks in queue
 * THREAD-SAFE: No (caller must ensure exclusive access)
 *
 * @param exec Executive controller to destroy (can be NULL)
 *
 * @note Safe to call with NULL pointer (no-op)
 * @note Frees all queued and active tasks
 */
void executive_destroy(executive_controller_t* exec)
{
    if (!exec) return;

    // WHAT: Free all tasks in queue
    // WHY:  Prevent memory leaks from dynamically allocated tasks
    // HOW:  Iterate and free each task descriptor
    for (uint32_t i = 0; i < exec->num_tasks; i++) {
        nimcp_free(exec->task_queue[i]);
    }

    // Free task queue array
    nimcp_free(exec->task_queue);

    // Free active task if exists
    if (exec->active_task) {
        nimcp_free(exec->active_task);
    }

    // Free controller structure
    nimcp_free(exec);
}

//=============================================================================
// Task Management
//=============================================================================

uint32_t executive_add_task(executive_controller_t* exec, const task_descriptor_t* task)
{
    if (!exec || !task) {
        set_error("NULL parameter");
        return 0;
    }

    if (exec->num_tasks >= exec->max_tasks) {
        set_error("Task queue full (%u/%u)", exec->num_tasks, exec->max_tasks);
        return 0;
    }

    // Allocate new task
    task_descriptor_t* new_task = nimcp_malloc(sizeof(task_descriptor_t));
    if (!new_task) {
        set_error("Failed to allocate task (%zu bytes)", sizeof(task_descriptor_t));
        return 0;
    }

    // Copy task data
    *new_task = *task;
    new_task->task_id = exec->next_task_id++;
    new_task->status = TASK_STATUS_PENDING;
    new_task->created_ms = get_current_time_ms();
    new_task->started_ms = 0;
    new_task->completed_ms = 0;
    new_task->steps_completed = 0;

    // Add to queue
    exec->task_queue[exec->num_tasks++] = new_task;
    exec->stats.total_tasks++;

    return new_task->task_id;
}

bool executive_switch_task(executive_controller_t* exec, uint32_t task_id, uint64_t current_time_ms)
{
    if (!exec) {
        set_error("NULL executive controller");
        return false;
    }

    // Find target task in queue
    task_descriptor_t* target = NULL;
    uint32_t target_index = 0;
    bool found_in_queue = false;

    // Check if it's the current active task
    if (exec->active_task && exec->active_task->task_id == task_id) {
        // Already active, no-op
        return true;
    }

    // Search in queue
    for (uint32_t i = 0; i < exec->num_tasks; i++) {
        if (exec->task_queue[i] && exec->task_queue[i]->task_id == task_id) {
            target = exec->task_queue[i];
            target_index = i;
            found_in_queue = true;
            break;
        }
    }

    if (!target) {
        set_error("Task %u not found", task_id);
        return false;
    }

    if (target->status != TASK_STATUS_PENDING && target->status != TASK_STATUS_SUSPENDED) {
        set_error("Cannot switch to task in status %d", target->status);
        return false;
    }

    // Suspend current active task and PUT IT BACK in the queue
    if (exec->active_task) {
        exec->active_task->status = TASK_STATUS_SUSPENDED;
        // If there's room, add it back to the queue
        if (exec->num_tasks < exec->max_tasks) {
            exec->task_queue[exec->num_tasks++] = exec->active_task;
        } else {
            // Queue full - just free the suspended task (task switching penalty)
            nimcp_free(exec->active_task);
        }
    }

    // Calculate switch cost (modulated by dopamine if brain available)
    float switch_cost = compute_modulated_switch_cost(exec, exec->config.task_switch_cost_ms);

    // Update statistics
    exec->stats.total_switches++;
    float old_avg = exec->stats.avg_switch_cost_ms;
    float n = (float)exec->stats.total_switches;
    exec->stats.avg_switch_cost_ms = (old_avg * (n - 1.0f) + switch_cost) / n;

    // REMOVE target from queue (to prevent double-free)
    if (found_in_queue) {
        // Shift remaining tasks down
        for (uint32_t i = target_index; i < exec->num_tasks - 1; i++) {
            exec->task_queue[i] = exec->task_queue[i + 1];
        }
        exec->task_queue[exec->num_tasks - 1] = NULL;
        exec->num_tasks--;
    }

    // Activate new task
    exec->active_task = target;
    exec->active_task->status = TASK_STATUS_ACTIVE;
    if (exec->active_task->started_ms == 0) {
        exec->active_task->started_ms = current_time_ms;
    }
    exec->last_switch_time_ms = current_time_ms;

    return true;
}

const task_descriptor_t* executive_get_active_task(executive_controller_t* exec)
{
    if (!exec) return NULL;
    return exec->active_task;
}

bool executive_complete_task(executive_controller_t* exec, bool success, uint64_t current_time_ms)
{
    if (!exec) {
        set_error("NULL executive controller");
        return false;
    }

    if (!exec->active_task) {
        set_error("No active task to complete");
        return false;
    }

    // Mark task as completed or failed
    exec->active_task->status = success ? TASK_STATUS_COMPLETED : TASK_STATUS_FAILED;
    exec->active_task->completed_ms = current_time_ms;

    // Update statistics
    if (success) {
        exec->stats.completed_tasks++;
    } else {
        exec->stats.failed_tasks++;
    }

    // Free the completed task (it's no longer in the queue)
    nimcp_free(exec->active_task);

    // Remove from active
    exec->active_task = NULL;

    // Try to activate next highest priority task
    if (exec->config.enable_task_prioritization) {
        task_descriptor_t* next = get_highest_priority_task(exec);
        if (next) {
            executive_switch_task(exec, next->task_id, current_time_ms);
        }
    }

    return true;
}

//=============================================================================
// Inhibitory Control
//=============================================================================

bool executive_should_inhibit(executive_controller_t* exec, float response_salience, const char* reason)
{
    if (!exec) return false;

    exec->total_decisions++;

    // Inhibit if salience exceeds threshold
    bool inhibit = response_salience >= exec->config.inhibition_threshold;

    if (inhibit) {
        exec->stats.inhibitions++;
        exec->stats.inhibition_rate = (float)exec->stats.inhibitions / (float)exec->total_decisions;
    }

    (void)reason; // Log reason in future version

    return inhibit;
}

//=============================================================================
// Planning
//=============================================================================

plan_t* executive_create_plan(executive_controller_t* exec, const char* goal, uint32_t max_steps)
{
    if (!exec || !goal) {
        set_error("NULL parameter");
        return NULL;
    }

    if (max_steps == 0 || max_steps > exec->config.max_plan_depth) {
        set_error("Invalid max_steps: %u (max: %u)", max_steps, exec->config.max_plan_depth);
        return NULL;
    }

    // Allocate plan
    plan_t* plan = nimcp_calloc(1, sizeof(plan_t));
    if (!plan) {
        set_error("Failed to allocate plan (%zu bytes)", sizeof(plan_t));
        return NULL;
    }

    // Allocate steps
    plan->steps = nimcp_calloc(max_steps, sizeof(plan_step_t));
    if (!plan->steps) {
        set_error("Failed to allocate plan steps (%zu bytes)",
                  max_steps * sizeof(plan_step_t));
        nimcp_free(plan);
        return NULL;
    }

    plan->type = PLAN_TYPE_SEQUENTIAL;
    snprintf(plan->goal, sizeof(plan->goal), "%s", goal);

    // Simple decomposition (in real implementation, use search/planning algorithm)
    // For now, create placeholder steps
    plan->num_steps = (max_steps > 3) ? 3 : max_steps;

    snprintf(plan->steps[0].description, sizeof(plan->steps[0].description), "Analyze: %s", goal);
    plan->steps[0].estimated_cost = 100;
    plan->steps[0].is_critical = true;

    if (plan->num_steps > 1) {
        snprintf(plan->steps[1].description, sizeof(plan->steps[1].description), "Execute: %s", goal);
        plan->steps[1].estimated_cost = 500;
        plan->steps[1].is_critical = true;
    }

    if (plan->num_steps > 2) {
        snprintf(plan->steps[2].description, sizeof(plan->steps[2].description), "Verify: %s", goal);
        plan->steps[2].estimated_cost = 200;
        plan->steps[2].is_critical = false;
    }

    // Update statistics
    exec->stats.plans_created++;
    float old_avg = exec->stats.avg_plan_length;
    float n = (float)exec->stats.plans_created;
    exec->stats.avg_plan_length = (old_avg * (n - 1.0f) + (float)plan->num_steps) / n;

    return plan;
}

void executive_destroy_plan(plan_t* plan)
{
    if (!plan) return;

    if (plan->steps) {
        // Free any action_data in steps
        for (uint32_t i = 0; i < plan->num_steps; i++) {
            // action_data ownership is external, don't free here
        }
        nimcp_free(plan->steps);
    }

    nimcp_free(plan);
}

//=============================================================================
// Statistics
//=============================================================================

bool executive_get_stats(executive_controller_t* exec, executive_stats_t* stats)
{
    if (!exec || !stats) {
        set_error("NULL parameter");
        return false;
    }

    *stats = exec->stats;
    return true;
}

void executive_reset_stats(executive_controller_t* exec)
{
    if (!exec) return;

    memset(&exec->stats, 0, sizeof(executive_stats_t));
    exec->total_decisions = 0;
}

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
 * BIOLOGY: Prefrontal cortex has limited capacity (~4 chunks in working memory)
 *          High load → poor multitasking, reduced exploration
 *
 * COMPLEXITY: O(1)
 *
 * @param exec Executive controller
 * @return Cognitive load [0, 1] (0=idle, 1=saturated)
 */
float executive_get_cognitive_load(executive_controller_t* exec)
{
    // Guard: NULL controller
    if (!exec) {
        return 0.0f;
    }

    // WHAT: Compute load as ratio of active tasks to capacity
    // WHY:  Simple proxy for cognitive resource usage
    // HOW:  (num_tasks + active_task) / max_tasks
    uint32_t total_tasks = exec->num_tasks;
    if (exec->active_task) {
        total_tasks++;  // Count active task
    }

    if (exec->max_tasks == 0) {
        return 0.0f;
    }

    float load = (float)total_tasks / (float)exec->max_tasks;

    // Clamp to [0, 1]
    return fminf(fmaxf(load, 0.0f), 1.0f);
}

/**
 * @brief Boost task priority based on external signal
 *
 * WHAT: Allow modules to boost task priority
 * WHY:  Curiosity-driven tasks should be prioritized when informative
 * HOW:  Find task by name, increase priority
 *
 * COMPLEXITY: O(n) where n = number of tasks
 *
 * @param exec Executive controller
 * @param task_name Task name to boost
 * @param boost_amount Priority boost [0, 1]
 * @return true if task found and boosted
 */
bool executive_boost_task_priority(executive_controller_t* exec,
                                    const char* task_name,
                                    float boost_amount)
{
    // Guard: Validate inputs
    if (!exec || !task_name) {
        set_error("NULL parameter in boost_task_priority");
        return false;
    }

    // Clamp boost amount
    boost_amount = fminf(fmaxf(boost_amount, 0.0f), 1.0f);

    // WHAT: Search for task by name
    // WHY:  Need to find task in queue or active slot
    // HOW:  Linear search through task descriptors
    bool found = false;

    // Check active task
    if (exec->active_task && strcmp(exec->active_task->name, task_name) == 0) {
        exec->active_task->priority += boost_amount;
        found = true;
    }

    // Check queued tasks
    for (uint32_t i = 0; i < exec->num_tasks; i++) {
        task_descriptor_t* task = exec->task_queue[i];
        if (task && strcmp(task->name, task_name) == 0) {
            task->priority += boost_amount;
            found = true;
        }
    }

    if (!found) {
        set_error("Task '%s' not found", task_name);
    }

    return found;
}
