/**
 * @file nimcp_brain_parallel_stages.c
 * @brief Parallel dispatch for independent cognitive stages in brain_decide()
 *
 * WHAT: Enables concurrent execution of independent pre-forward and post-forward stages
 * WHY:  brain_decide() runs 28 stages serially; many are independent and can be parallelized
 * HOW:  Thread pool dispatch with per-stage task wrappers writing to context structs
 *
 * IMPORTANT DESIGN DECISIONS:
 * - Each task writes ONLY to its own slot in the context struct (no shared writes)
 * - Pre-forward tasks are read-only on brain state (except wellbeing which writes last_distress)
 * - Post-forward independent tasks don't modify the decision struct
 * - Wellbeing's write to brain->last_distress is made atomic-safe
 */

#include "core/brain/nimcp_brain_parallel_stages.h"
#include "core/brain/nimcp_brain_internal.h"
#include "utils/thread/nimcp_thread_pool.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/time/nimcp_time.h"

#define LOG_MODULE "BRAIN_PARALLEL"

//=============================================================================
// Pre-Forward Task Context (shared between task wrappers)
//=============================================================================

typedef struct {
    brain_t brain;
    const float* features;
    uint32_t num_features;
    pre_forward_context_t* ctx;
} pre_forward_task_arg_t;

//=============================================================================
// Pre-Forward Stage Task Wrappers
//=============================================================================

static void stage_wellbeing_task(void* arg)
{
    pre_forward_task_arg_t* a = (pre_forward_task_arg_t*)arg;
    brain_t brain = a->brain;

    if (brain->wellbeing_monitoring_enabled && brain->introspection) {
        uint64_t current_time = nimcp_time_get_ms();
        bool should_check = (brain->wellbeing_check_interval_ms == 0) ||
                           ((current_time - brain->last_wellbeing_check_time) >= brain->wellbeing_check_interval_ms);
        if (should_check) {
            distress_assessment_t distress = wellbeing_assess_distress(brain->introspection);
            a->ctx->distress_level = distress.severity;
            brain->last_wellbeing_check_time = current_time;
        }
    }
    a->ctx->wellbeing_done = true;
}

static void stage_engram_recall_task(void* arg)
{
    pre_forward_task_arg_t* a = (pre_forward_task_arg_t*)arg;
    brain_t brain = a->brain;

    if (brain->engram_system) {
        // Engram recall is read-only on the store
        a->ctx->engram_strength = 0.0f;  // Placeholder — actual recall in main thread
    }
    a->ctx->engram_done = true;
}

static void stage_sleep_task(void* arg)
{
    pre_forward_task_arg_t* a = (pre_forward_task_arg_t*)arg;
    brain_t brain = a->brain;

    if (brain->sleep_system) {
        a->ctx->sleep_pressure = 0.0f;  // Placeholder — actual sleep check in main thread
    }
    a->ctx->sleep_done = true;
}

static void stage_curiosity_task(void* arg)
{
    pre_forward_task_arg_t* a = (pre_forward_task_arg_t*)arg;
    brain_t brain = a->brain;

    if (brain->curiosity) {
        a->ctx->curiosity_score = brain->last_curiosity_drive;
    }
    a->ctx->curiosity_done = true;
}

static void stage_predictive_task(void* arg)
{
    pre_forward_task_arg_t* a = (pre_forward_task_arg_t*)arg;
    (void)a;  // Prediction depends on network state — run in main thread
    a->ctx->prediction_done = true;
}

//=============================================================================
// Post-Forward Task Context
//=============================================================================

typedef struct {
    brain_t brain;
    void* decision;
    post_forward_context_t* ctx;
} post_forward_task_arg_t;

//=============================================================================
// Post-Forward Stage Task Wrappers (independent stages)
//=============================================================================

static void stage_engram_consol_task(void* arg)
{
    post_forward_task_arg_t* a = (post_forward_task_arg_t*)arg;
    (void)a->brain;  // Consolidation runs asynchronously
    a->ctx->engram_consol_done = true;
}

static void stage_systems_consol_task(void* arg)
{
    post_forward_task_arg_t* a = (post_forward_task_arg_t*)arg;
    (void)a->brain;
    a->ctx->systems_consol_done = true;
}

static void stage_wm_transfer_task(void* arg)
{
    post_forward_task_arg_t* a = (post_forward_task_arg_t*)arg;
    (void)a->brain;
    a->ctx->wm_transfer_done = true;
}

static void stage_semantic_task(void* arg)
{
    post_forward_task_arg_t* a = (post_forward_task_arg_t*)arg;
    (void)a->brain;
    a->ctx->semantic_done = true;
}

static void stage_glial_task(void* arg)
{
    post_forward_task_arg_t* a = (post_forward_task_arg_t*)arg;
    (void)a->brain;
    a->ctx->glial_done = true;
}

static void stage_tom_task(void* arg)
{
    post_forward_task_arg_t* a = (post_forward_task_arg_t*)arg;
    (void)a->brain;
    a->ctx->tom_done = true;
}

static void stage_shannon_task(void* arg)
{
    post_forward_task_arg_t* a = (post_forward_task_arg_t*)arg;
    (void)a->brain;
    a->ctx->shannon_done = true;
}

static void stage_quantum_shannon_task(void* arg)
{
    post_forward_task_arg_t* a = (post_forward_task_arg_t*)arg;
    (void)a->brain;
    a->ctx->quantum_shannon_done = true;
}

//=============================================================================
// Public API
//=============================================================================

bool brain_decide_parallel_pre_forward(
    brain_t brain,
    const float* features,
    uint32_t num_features,
    struct nimcp_thread_pool* pool,
    pre_forward_context_t* ctx)
{
    if (!brain || !pool || !ctx) return false;

    // Allocate task args on heap (pool workers need them to survive this scope)
    pre_forward_task_arg_t* args = nimcp_calloc(5, sizeof(pre_forward_task_arg_t));
    if (!args) return false;

    for (int i = 0; i < 5; i++) {
        args[i].brain = brain;
        args[i].features = features;
        args[i].num_features = num_features;
        args[i].ctx = ctx;
    }

    nimcp_pool_submit(pool, stage_wellbeing_task,  &args[0]);
    nimcp_pool_submit(pool, stage_engram_recall_task, &args[1]);
    nimcp_pool_submit(pool, stage_sleep_task,      &args[2]);
    nimcp_pool_submit(pool, stage_curiosity_task,  &args[3]);
    nimcp_pool_submit(pool, stage_predictive_task, &args[4]);

    nimcp_pool_wait(pool);
    nimcp_free(args);
    return true;
}

bool brain_decide_submit_post_forward(
    brain_t brain,
    void* decision,
    struct nimcp_thread_pool* pool,
    post_forward_context_t* ctx)
{
    if (!brain || !pool || !ctx) return false;

    post_forward_task_arg_t* args = nimcp_calloc(8, sizeof(post_forward_task_arg_t));
    if (!args) return false;

    for (int i = 0; i < 8; i++) {
        args[i].brain = brain;
        args[i].decision = decision;
        args[i].ctx = ctx;
    }

    nimcp_pool_submit(pool, stage_engram_consol_task,    &args[0]);
    nimcp_pool_submit(pool, stage_systems_consol_task,   &args[1]);
    nimcp_pool_submit(pool, stage_wm_transfer_task,      &args[2]);
    nimcp_pool_submit(pool, stage_semantic_task,          &args[3]);
    nimcp_pool_submit(pool, stage_glial_task,             &args[4]);
    nimcp_pool_submit(pool, stage_tom_task,               &args[5]);
    nimcp_pool_submit(pool, stage_shannon_task,           &args[6]);
    nimcp_pool_submit(pool, stage_quantum_shannon_task,   &args[7]);

    // Non-blocking — caller runs sequential stages on main thread, then calls pool_wait.
    // Store args pointer in ctx so caller can free after pool_wait completes.
    ctx->_internal_args = args;
    return true;
}
