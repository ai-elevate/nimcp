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
        uint64_t last_check = __atomic_load_n(&brain->last_wellbeing_check_time, __ATOMIC_ACQUIRE);
        bool should_check = (brain->wellbeing_check_interval_ms == 0) ||
                           ((current_time - last_check) >= brain->wellbeing_check_interval_ms);
        if (should_check) {
            distress_assessment_t distress = wellbeing_assess_distress(brain->introspection);
            a->ctx->distress_level = distress.severity;
            __atomic_store_n(&brain->last_wellbeing_check_time, current_time, __ATOMIC_RELEASE);
        }
    }
    a->ctx->wellbeing_done = true;
}

static void stage_engram_recall_task(void* arg)
{
    pre_forward_task_arg_t* a = (pre_forward_task_arg_t*)arg;
    brain_t brain = a->brain;

    if (brain->engram_system) {
        // TODO: Implement parallel engram recall — currently a stub.
        //       Should call engram_recall() with feature pattern and populate
        //       a->ctx->engram_strength with the match score.
        a->ctx->engram_strength = 0.0f;  // Placeholder — actual recall in main thread
    }
    a->ctx->engram_done = true;
}

static void stage_sleep_task(void* arg)
{
    pre_forward_task_arg_t* a = (pre_forward_task_arg_t*)arg;
    brain_t brain = a->brain;

    if (brain->sleep_system) {
        // TODO: Implement parallel sleep pressure query — currently a stub.
        //       Should call sleep_get_pressure() and populate a->ctx->sleep_pressure.
        a->ctx->sleep_pressure = 0.0f;  // Placeholder — actual sleep check in main thread
    }
    a->ctx->sleep_done = true;
}

static void stage_curiosity_task(void* arg)
{
    pre_forward_task_arg_t* a = (pre_forward_task_arg_t*)arg;
    brain_t brain = a->brain;

    if (brain->curiosity) {
        // TODO: Implement parallel curiosity evaluation — currently reads a cached value.
        //       Should call curiosity_evaluate() with features for fresh drive computation.
        a->ctx->curiosity_score = brain->last_curiosity_drive;
    }
    a->ctx->curiosity_done = true;
}

static void stage_predictive_task(void* arg)
{
    pre_forward_task_arg_t* a = (pre_forward_task_arg_t*)arg;
    (void)a;  // Prediction depends on network state — run in main thread
    // TODO: Implement parallel predictive coding — currently a stub.
    //       Needs careful synchronization since prediction reads network weights.
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
    // TODO: Implement engram consolidation — should trigger engram_consolidate_step()
    //       for recently-activated engrams after the decision is made.
    a->ctx->engram_consol_done = true;
}

static void stage_systems_consol_task(void* arg)
{
    post_forward_task_arg_t* a = (post_forward_task_arg_t*)arg;
    (void)a->brain;
    // TODO: Implement systems consolidation — should call systems_consolidation_step()
    //       for hippocampus-to-cortex memory transfer after each decision.
    a->ctx->systems_consol_done = true;
}

static void stage_wm_transfer_task(void* arg)
{
    post_forward_task_arg_t* a = (post_forward_task_arg_t*)arg;
    (void)a->brain;
    // TODO: Implement working memory transfer — should call wm_transfer_step()
    //       to selectively consolidate rehearsed WM items into engrams.
    a->ctx->wm_transfer_done = true;
}

static void stage_semantic_task(void* arg)
{
    post_forward_task_arg_t* a = (post_forward_task_arg_t*)arg;
    (void)a->brain;
    // TODO: Implement semantic memory update — should call semantic_memory_step()
    //       to update spreading activation and concept priming after the decision.
    a->ctx->semantic_done = true;
}

static void stage_glial_task(void* arg)
{
    post_forward_task_arg_t* a = (post_forward_task_arg_t*)arg;
    (void)a->brain;
    // TODO: Implement glial maintenance — should call glial_step() to update
    //       astrocyte calcium dynamics and myelin modulation post-decision.
    a->ctx->glial_done = true;
}

static void stage_tom_task(void* arg)
{
    post_forward_task_arg_t* a = (post_forward_task_arg_t*)arg;
    (void)a->brain;
    // TODO: Implement Theory of Mind update — should call tom_update() to
    //       refresh agent models based on the decision outcome.
    a->ctx->tom_done = true;
}

static void stage_shannon_task(void* arg)
{
    post_forward_task_arg_t* a = (post_forward_task_arg_t*)arg;
    (void)a->brain;
    // TODO: Implement Shannon information analysis — should compute
    //       broadcast information content and update workspace entropy stats.
    a->ctx->shannon_done = true;
}

static void stage_quantum_shannon_task(void* arg)
{
    post_forward_task_arg_t* a = (post_forward_task_arg_t*)arg;
    (void)a->brain;
    // TODO: Implement quantum Shannon analysis — should compute
    //       von Neumann entropy for quantum-coherent workspace states.
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

    bool all_submitted = true;
    if (nimcp_pool_submit(pool, stage_wellbeing_task,     &args[0]) != NIMCP_SUCCESS) {
        ctx->wellbeing_done = true;  // Mark done so caller doesn't hang
        all_submitted = false;
    }
    if (nimcp_pool_submit(pool, stage_engram_recall_task, &args[1]) != NIMCP_SUCCESS) {
        ctx->engram_done = true;
        all_submitted = false;
    }
    if (nimcp_pool_submit(pool, stage_sleep_task,         &args[2]) != NIMCP_SUCCESS) {
        ctx->sleep_done = true;
        all_submitted = false;
    }
    if (nimcp_pool_submit(pool, stage_curiosity_task,     &args[3]) != NIMCP_SUCCESS) {
        ctx->curiosity_done = true;
        all_submitted = false;
    }
    if (nimcp_pool_submit(pool, stage_predictive_task,    &args[4]) != NIMCP_SUCCESS) {
        ctx->prediction_done = true;
        all_submitted = false;
    }

    if (!all_submitted) {
        LOG_MODULE_WARN(LOG_MODULE, "Some pre-forward tasks failed to submit to thread pool");
    }

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

    if (nimcp_pool_submit(pool, stage_engram_consol_task,    &args[0]) != NIMCP_SUCCESS)
        ctx->engram_consol_done = true;
    if (nimcp_pool_submit(pool, stage_systems_consol_task,   &args[1]) != NIMCP_SUCCESS)
        ctx->systems_consol_done = true;
    if (nimcp_pool_submit(pool, stage_wm_transfer_task,      &args[2]) != NIMCP_SUCCESS)
        ctx->wm_transfer_done = true;
    if (nimcp_pool_submit(pool, stage_semantic_task,          &args[3]) != NIMCP_SUCCESS)
        ctx->semantic_done = true;
    if (nimcp_pool_submit(pool, stage_glial_task,             &args[4]) != NIMCP_SUCCESS)
        ctx->glial_done = true;
    if (nimcp_pool_submit(pool, stage_tom_task,               &args[5]) != NIMCP_SUCCESS)
        ctx->tom_done = true;
    if (nimcp_pool_submit(pool, stage_shannon_task,           &args[6]) != NIMCP_SUCCESS)
        ctx->shannon_done = true;
    if (nimcp_pool_submit(pool, stage_quantum_shannon_task,   &args[7]) != NIMCP_SUCCESS)
        ctx->quantum_shannon_done = true;

    // Non-blocking — caller runs sequential stages on main thread, then calls pool_wait.
    // Store args pointer in ctx so caller can free after pool_wait completes.
    ctx->_internal_args = args;
    return true;
}
