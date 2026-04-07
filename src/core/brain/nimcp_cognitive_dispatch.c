/**
 * @file nimcp_cognitive_dispatch.c
 * @brief Parallel cognitive module dispatch — actor pattern implementation
 *
 * Each cognitive training module runs as an independent actor:
 * - Receives input via task argument struct (immutable during dispatch)
 * - Trains its own weights (exclusive ownership, no shared writes)
 * - Returns result via promise (loss, timing, success)
 *
 * The coordinator (main thread) collects all results after a barrier
 * and updates cognitive_stats single-threaded. Zero mutex contention.
 */

#include "core/brain/nimcp_cognitive_dispatch.h"
#include "core/brain/nimcp_brain_internal.h"
#include "core/brain/nimcp_brain_lazy_init.h"  /* BRAIN_ENSURE_FEP_ORCHESTRATOR */
#include "utils/thread/nimcp_thread_pool.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/time/nimcp_time.h"
#include "utils/logging/nimcp_logging.h"
#include "async/nimcp_future.h"

/* Cognitive module headers */
#include "language/nimcp_grounded_language.h"
#include "cognitive/knowledge/nimcp_knowledge.h"
#include "cognitive/vae/nimcp_vae.h"
#include "cognitive/vae/bridges/nimcp_vae_training_bridge.h"
#include "cognitive/parietal/nimcp_parietal.h"
#include "cognitive/predictive/nimcp_predictive_hierarchy.h"
#include "cognitive/jepa/nimcp_jepa_predictor.h"
#include "cognitive/jepa/nimcp_jepa_latent.h"
#include "cognitive/immune/nimcp_self_heal.h"
#include "cognitive/free_energy/nimcp_free_energy.h"
#include "snn/nimcp_snn_fno.h"

#include <math.h>
#include <string.h>

#define LOG_MODULE "COG_DISPATCH"

/* Forward declarations for subsystem APIs */
struct creative_training_bridge;
typedef struct creative_training_bridge creative_training_bridge_t;
extern int creative_training_submit_feedback(creative_training_bridge_t* bridge,
                                              const void* content,
                                              int modality,
                                              uint8_t rating,
                                              const char* feedback);

/*=============================================================================
 * Task Argument — passed to each actor via thread pool
 *=============================================================================*/

typedef struct {
    brain_t              brain;
    const float*         features;
    uint32_t             num_features;
    const float*         target;
    uint32_t             target_size;
    const char*          label;
    float                loss;
    nimcp_promise_t      promise;
    cognitive_module_id_t module_id;
} cog_task_arg_t;

/*=============================================================================
 * Helper: create a "not executed" result and complete promise
 *=============================================================================*/

static void complete_skip(nimcp_promise_t promise, cognitive_module_id_t id)
{
    cognitive_task_result_t r = {
        .module_id = id, .executed = false, .success = true, .loss = NAN
    };
    nimcp_promise_complete(promise, &r);
}

/*=============================================================================
 * 12 Task Wrapper Functions — one per cognitive module
 *=============================================================================*/

/* 1. Grounded Language */
static void task_grounded_lang(void* arg)
{
    cog_task_arg_t* a = (cog_task_arg_t*)arg;
    cognitive_task_result_t r = { .module_id = COG_MODULE_GROUNDED_LANG, .loss = NAN };
    uint64_t t0 = nimcp_time_get_us();

    if (a->brain->grounded_lang && a->label && a->label[0]) {
        grounded_language_learn_from_text(a->brain->grounded_lang, a->label);
        grounded_language_learn_syntax(a->brain->grounded_lang, a->label);
        r.executed = true;
        r.success = true;
    }
    r.elapsed_us = nimcp_time_get_us() - t0;
    nimcp_promise_complete(a->promise, &r);
}

/* 2. Knowledge System */
static void task_knowledge(void* arg)
{
    cog_task_arg_t* a = (cog_task_arg_t*)arg;
    cognitive_task_result_t r = { .module_id = COG_MODULE_KNOWLEDGE, .loss = NAN };
    uint64_t t0 = nimcp_time_get_us();

    if (a->brain->knowledge && a->label && a->label[0]) {
        knowledge_learn_from_text(a->brain->knowledge, a->label,
                                  KNOWLEDGE_DOMAIN_GENERAL);
        r.executed = true;
        r.success = true;
    }
    r.elapsed_us = nimcp_time_get_us() - t0;
    nimcp_promise_complete(a->promise, &r);
}

/* 3. VAE */
static void task_vae(void* arg)
{
    cog_task_arg_t* a = (cog_task_arg_t*)arg;
    cognitive_task_result_t r = { .module_id = COG_MODULE_VAE, .loss = NAN };
    uint64_t t0 = nimcp_time_get_us();

    if (a->brain->vae_training_bridge && a->brain->vae_enabled) {
        vae_training_bridge_t* vae = (vae_training_bridge_t*)a->brain->vae_training_bridge;
        vae_training_step_result_t vae_result = {0};
        vae_training_step(vae, a->features, a->num_features,
                          a->features, a->num_features, &vae_result);
        r.executed = true;
        r.success = true;
        r.loss = vae_result.loss.total_loss;
        r.aux_value = vae_result.loss.total_loss;
    }
    r.elapsed_us = nimcp_time_get_us() - t0;
    nimcp_promise_complete(a->promise, &r);
}

/* 4. FEP-Parietal */
static void task_fep_parietal(void* arg)
{
    cog_task_arg_t* a = (cog_task_arg_t*)arg;
    cognitive_task_result_t r = { .module_id = COG_MODULE_FEP_PARIETAL, .loss = NAN };
    uint64_t t0 = nimcp_time_get_us();

    if (a->brain->parietal && a->features && a->target) {
        fep_parietal_bridge_t* fep_bridge = parietal_get_fep_bridge(a->brain->parietal);
        if (fep_bridge) {
            const float* obs_ptr = a->features;
            const float* tgt_ptr = a->target;
            fep_parietal_train_model(fep_bridge, &obs_ptr, &tgt_ptr, 1);
            r.executed = true;
            r.success = true;
        }
    }
    r.elapsed_us = nimcp_time_get_us() - t0;
    nimcp_promise_complete(a->promise, &r);
}

/* 5. Physics NN */
static void task_physics_nn(void* arg)
{
    cog_task_arg_t* a = (cog_task_arg_t*)arg;
    cognitive_task_result_t r = { .module_id = COG_MODULE_PHYSICS_NN, .loss = NAN };
    uint64_t t0 = nimcp_time_get_us();

    if (a->brain->parietal && a->features && a->target &&
        a->num_features >= 4 && a->target_size >= 4) {
        uint32_t phys_dim = (a->num_features < 32) ? a->num_features : 32;
        float phys_state[32] = {0};
        float phys_deriv[32] = {0};
        for (uint32_t i = 0; i < phys_dim; i++) {
            phys_state[i] = a->features[i];
            phys_deriv[i] = a->target[i];
        }
        const float* state_ptr = phys_state;
        const float* deriv_ptr = phys_deriv;
        parietal_train_physics_nn(a->brain->parietal,
                                  &state_ptr, &deriv_ptr, 1, 1);
        r.executed = true;
        r.success = true;
    }
    r.elapsed_us = nimcp_time_get_us() - t0;
    nimcp_promise_complete(a->promise, &r);
}

/* 6. Predictive Hierarchy */
static void task_pred_hierarchy(void* arg)
{
    cog_task_arg_t* a = (cog_task_arg_t*)arg;
    cognitive_task_result_t r = { .module_id = COG_MODULE_PRED_HIERARCHY, .loss = NAN };
    uint64_t t0 = nimcp_time_get_us();

    if (a->brain->pred_hierarchy && a->brain->pred_hierarchy_enabled) {
        predictive_hierarchy_t* ph = (predictive_hierarchy_t*)a->brain->pred_hierarchy;
        float pred_loss = 0.0f;
        if (ph->bottom && a->num_features >= ph->bottom->dim) {
            pred_hier_learn_step(ph, a->features, &pred_loss);
        }
        r.executed = true;
        r.success = true;
        r.loss = pred_loss;
    }
    r.elapsed_us = nimcp_time_get_us() - t0;
    nimcp_promise_complete(a->promise, &r);
}

/* 7. JEPA Predictor */
static void task_jepa(void* arg)
{
    cog_task_arg_t* a = (cog_task_arg_t*)arg;
    cognitive_task_result_t r = { .module_id = COG_MODULE_JEPA, .loss = NAN };
    uint64_t t0 = nimcp_time_get_us();

    if (a->brain->jepa_predictor && a->brain->jepa_predictor_enabled &&
        a->features && a->target) {
        uint32_t latent_dim = (a->num_features < 256) ? a->num_features : 256;
        jepa_latent_t* context = jepa_latent_create_dim(latent_dim);
        jepa_latent_t* target_latent = jepa_latent_create_dim(latent_dim);

        if (context && target_latent) {
            jepa_latent_set_embedding(context, a->features,
                (a->num_features < latent_dim) ? a->num_features : latent_dim);
            jepa_latent_set_embedding(target_latent, a->target,
                (a->target_size < latent_dim) ? a->target_size : latent_dim);

            float jepa_loss = 0.0f;
            jepa_predictor_train_step(
                (jepa_predictor_t*)a->brain->jepa_predictor,
                context, target_latent, &jepa_loss);
            r.executed = true;
            r.success = true;
            r.loss = jepa_loss;
        }
        if (context) jepa_latent_destroy(context);
        if (target_latent) jepa_latent_destroy(target_latent);
    }
    r.elapsed_us = nimcp_time_get_us() - t0;
    nimcp_promise_complete(a->promise, &r);
}

/* 8. Creative Training */
static void task_creative(void* arg)
{
    cog_task_arg_t* a = (cog_task_arg_t*)arg;
    cognitive_task_result_t r = { .module_id = COG_MODULE_CREATIVE, .loss = NAN };
    uint64_t t0 = nimcp_time_get_us();

    if (a->brain->creative_training_bridge && a->brain->creative_enabled) {
        uint8_t rating = (a->loss < 0.1f) ? 5 :
                         (a->loss < 0.3f) ? 4 :
                         (a->loss < 0.5f) ? 3 :
                         (a->loss < 0.7f) ? 2 : 1;
        creative_training_submit_feedback(
            a->brain->creative_training_bridge,
            a->target, 0, rating, a->label);
        r.executed = true;
        r.success = true;
    }
    r.elapsed_us = nimcp_time_get_us() - t0;
    nimcp_promise_complete(a->promise, &r);
}

/* 9. Self-Heal Engine */
static void task_self_heal(void* arg)
{
    cog_task_arg_t* a = (cog_task_arg_t*)arg;
    cognitive_task_result_t r = { .module_id = COG_MODULE_SELF_HEAL, .loss = NAN };
    uint64_t t0 = nimcp_time_get_us();

    if (a->brain->self_heal_engine && a->brain->self_heal_enabled && a->features) {
        crash_features_t cf = {0};
        uint32_t cf_dim = (a->num_features < SELF_HEAL_FEATURE_DIM) ?
                           a->num_features : SELF_HEAL_FEATURE_DIM;
        cf.n_features = cf_dim;
        memcpy(cf.features, a->features, cf_dim * sizeof(float));
        float success = 1.0f - fminf(a->loss, 1.0f);
        self_heal_train_online(
            (self_heal_engine_t*)a->brain->self_heal_engine,
            &cf, FIX_PATTERN_UNKNOWN, success);
        r.executed = true;
        r.success = true;
    }
    r.elapsed_us = nimcp_time_get_us() - t0;
    nimcp_promise_complete(a->promise, &r);
}

/* 10. Intuition System */
static void task_intuition(void* arg)
{
    cog_task_arg_t* a = (cog_task_arg_t*)arg;
    cognitive_task_result_t r = { .module_id = COG_MODULE_INTUITION, .loss = NAN };
    uint64_t t0 = nimcp_time_get_us();

    if (a->brain->intuition_system && a->brain->intuition_system_enabled) {
        intuition_experience_t exp = {
            .id = a->brain->cognitive_train_counter,
            .hunch = NULL,
            .predicted_outcome = 0.0f,
            .actual_outcome = a->loss,
            .timestamp = (float)nimcp_time_get_us(),
            .was_successful = (a->loss < 0.3f)
        };
        const intuition_experience_t* exp_ptr = &exp;
        intuition_train_from_experience(a->brain->intuition_system,
                                         &exp_ptr, 1);
        r.executed = true;
        r.success = true;
    }
    r.elapsed_us = nimcp_time_get_us() - t0;
    nimcp_promise_complete(a->promise, &r);
}

/* 11. FEP Orchestrator — returns metrics via result, coordinator writes them */
static void task_fep_orchestrator(void* arg)
{
    cog_task_arg_t* a = (cog_task_arg_t*)arg;
    cognitive_task_result_t r = { .module_id = COG_MODULE_FEP_ORCHESTRATOR, .loss = NAN };
    uint64_t t0 = nimcp_time_get_us();

    /* FEP orchestrator metrics are returned via result struct.
     * The coordinator writes them to brain->fep_orchestrator single-threaded.
     * loss = free_energy, aux_value = surprise */
    if (a->brain->fep_orchestrator && a->brain->fep_orchestrator_enabled) {
        r.executed = true;
        r.success = true;
        r.loss = a->loss;  /* free_energy = training loss */
        r.aux_value = -logf(fmaxf(1.0f - a->loss, 1e-7f));  /* surprise */
    }
    r.elapsed_us = nimcp_time_get_us() - t0;
    nimcp_promise_complete(a->promise, &r);
}

/* 12. SNN FNO Populations */
static void task_snn_fno(void* arg)
{
    cog_task_arg_t* a = (cog_task_arg_t*)arg;
    cognitive_task_result_t r = { .module_id = COG_MODULE_SNN_FNO, .loss = NAN };
    uint64_t t0 = nimcp_time_get_us();

    if (a->brain->snn_fno_populations && a->brain->snn_fno_count > 0) {
        for (uint32_t fi = 0; fi < a->brain->snn_fno_count; fi++) {
            snn_fno_population_t* fno_pop =
                (snn_fno_population_t*)a->brain->snn_fno_populations[fi];
            if (fno_pop) {
                snn_fno_train(fno_pop, 1);
            }
        }
        r.executed = true;
        r.success = true;
    }
    r.elapsed_us = nimcp_time_get_us() - t0;
    nimcp_promise_complete(a->promise, &r);
}

/*=============================================================================
 * Task function table — indexed by cognitive_module_id_t
 *=============================================================================*/

static const nimcp_task_fn TASK_FNS[COG_MODULE_COUNT] = {
    task_grounded_lang,     /* 0 */
    task_knowledge,         /* 1 */
    task_vae,               /* 2 */
    task_fep_parietal,      /* 3 */
    task_physics_nn,        /* 4 */
    task_pred_hierarchy,    /* 5 */
    task_jepa,              /* 6 */
    task_creative,          /* 7 */
    task_self_heal,         /* 8 */
    task_intuition,         /* 9 */
    task_fep_orchestrator,  /* 10 */
    task_snn_fno            /* 11 */
};

/*=============================================================================
 * Coordinator: update cognitive_stats from batch results (single-threaded)
 *=============================================================================*/

static void dispatch_update_stats(brain_t brain, const cognitive_batch_result_t* batch)
{
    for (uint32_t i = 0; i < COG_MODULE_COUNT; i++) {
        const cognitive_task_result_t* r = &batch->results[i];
        if (!r->executed) continue;

        switch (r->module_id) {
        case COG_MODULE_GROUNDED_LANG:
            brain->cognitive_stats.grounded_lang_steps++;
            break;
        case COG_MODULE_KNOWLEDGE:
            brain->cognitive_stats.knowledge_steps++;
            break;
        case COG_MODULE_VAE:
            brain->cognitive_stats.vae_steps++;
            brain->cognitive_stats.vae_last_loss = r->loss;
            brain->last_vae_free_energy = r->aux_value;
            break;
        case COG_MODULE_FEP_PARIETAL:
            brain->cognitive_stats.fep_parietal_steps++;
            break;
        case COG_MODULE_PHYSICS_NN:
            brain->cognitive_stats.physics_nn_steps++;
            break;
        case COG_MODULE_PRED_HIERARCHY:
            brain->cognitive_stats.pred_hierarchy_steps++;
            brain->cognitive_stats.pred_hierarchy_last_loss = r->loss;
            break;
        case COG_MODULE_JEPA:
            brain->cognitive_stats.jepa_steps++;
            brain->cognitive_stats.jepa_last_loss = r->loss;
            break;
        case COG_MODULE_CREATIVE:
            brain->cognitive_stats.creative_steps++;
            break;
        case COG_MODULE_SELF_HEAL:
            brain->cognitive_stats.self_heal_steps++;
            break;
        case COG_MODULE_INTUITION:
            brain->cognitive_stats.intuition_steps++;
            break;
        case COG_MODULE_FEP_ORCHESTRATOR:
            brain->cognitive_stats.fep_orchestrator_steps++;
            if (brain->fep_orchestrator) {
                brain->fep_orchestrator->fep_metrics.free_energy = r->loss;
                brain->fep_orchestrator->fep_metrics.prediction_error = r->loss;
                brain->fep_orchestrator->fep_metrics.surprise = r->aux_value;
            }
            break;
        case COG_MODULE_SNN_FNO:
            /* No per-step counter in cognitive_stats for FNO */
            break;
        default:
            break;
        }
    }
}

/*=============================================================================
 * Main Parallel Dispatch Function
 *=============================================================================*/

cognitive_batch_result_t brain_train_cognitive_parallel(
    brain_t brain,
    const float* features,
    uint32_t num_features,
    const float* target,
    uint32_t target_size,
    const char* label,
    float loss)
{
    cognitive_batch_result_t batch = {0};
    if (!brain) return batch;

    /* Interval gating: expensive subsystem training runs every N steps.
     * NOTE: This is the ONLY place interval gating happens. The sequential
     * fallback does NOT gate (it was already gated here). The wrapper in
     * brain_learning.c calls us directly without gating. */
    uint32_t interval = brain->cognitive_train_interval;
    if (interval == 0) interval = 5;
    brain->cognitive_train_counter++;
    if ((brain->cognitive_train_counter % interval) != 0) return batch;

    uint64_t t0 = nimcp_time_get_us();

    /* Fallback to sequential if no thread pool.
     * Interval gating already passed above — sequential runs unconditionally. */
    if (!brain->inference_pool) {
        brain_train_cognitive_subsystems_sequential(
            brain, features, num_features, target, target_size, label, loss);
        batch.total_elapsed_us = nimcp_time_get_us() - t0;
        brain->dispatch_metrics.total_sequential++;
        brain->dispatch_metrics.last_dispatch_us = batch.total_elapsed_us;
        return batch;
    }

    /* Lazy-init FEP orchestrator on main thread (NOT thread-safe, must not
     * be called from workers). This ensures it's valid before dispatch. */
    BRAIN_ENSURE_FEP_ORCHESTRATOR(brain);

    /* Allocate task arguments on heap (freed after barrier) */
    cog_task_arg_t* args = (cog_task_arg_t*)nimcp_calloc(
        COG_MODULE_COUNT, sizeof(cog_task_arg_t));
    if (!args) {
        brain_train_cognitive_subsystems_sequential(
            brain, features, num_features, target, target_size, label, loss);
        batch.total_elapsed_us = nimcp_time_get_us() - t0;
        return batch;
    }

    nimcp_promise_t promises[COG_MODULE_COUNT] = {0};
    nimcp_future_t  futures[COG_MODULE_COUNT]  = {0};
    uint32_t submitted = 0;

    /* Create promises and submit tasks */
    for (uint32_t i = 0; i < COG_MODULE_COUNT; i++) {
        promises[i] = nimcp_promise_create(sizeof(cognitive_task_result_t));
        if (!promises[i]) {
            /* Promise creation failed — skip this module entirely.
             * batch.results[i] is already zero-initialized (executed=false). */
            continue;
        }
        futures[i] = nimcp_promise_get_future(promises[i]);
        if (!futures[i]) {
            nimcp_promise_destroy(promises[i]);
            promises[i] = NULL;
            continue;
        }

        /* Populate task argument — immutable input for this actor */
        args[i].brain = brain;
        args[i].features = features;
        args[i].num_features = num_features;
        args[i].target = target;
        args[i].target_size = target_size;
        args[i].label = label;
        args[i].loss = loss;
        args[i].promise = promises[i];
        args[i].module_id = (cognitive_module_id_t)i;

        if (nimcp_pool_submit(brain->inference_pool,
                              TASK_FNS[i], &args[i]) == NIMCP_SUCCESS) {
            submitted++;
        } else {
            /* Pool full or error — complete promise as skipped */
            complete_skip(promises[i], (cognitive_module_id_t)i);
        }
    }

    /* Barrier: wait for all modules to complete.
     * nimcp_future_all does NOT handle NULL entries — filter them out. */
    nimcp_future_t valid_futures[COG_MODULE_COUNT];
    uint32_t valid_count = 0;
    for (uint32_t i = 0; i < COG_MODULE_COUNT; i++) {
        if (futures[i]) valid_futures[valid_count++] = futures[i];
    }

    if (valid_count > 0) {
        nimcp_future_t all_future = nimcp_future_all(valid_futures, valid_count);
        if (all_future) {
            /* Timeout prevents deadlock if a worker thread crashes.
             * 30s is generous — modules normally complete in <5s total. */
            if (!nimcp_future_wait_timeout(all_future, 30000)) {
                LOG_WARN("Cognitive parallel dispatch timed out after 30s "
                         "(%u/%u modules may be stuck)", valid_count - submitted, valid_count);
            }
            nimcp_future_destroy(all_future);
        } else {
            /* Fallback: wait individually with timeout */
            for (uint32_t i = 0; i < valid_count; i++) {
                nimcp_future_wait_timeout(valid_futures[i], 30000);
            }
        }
    }

    /* Coordinator: collect results single-threaded */
    for (uint32_t i = 0; i < COG_MODULE_COUNT; i++) {
        if (futures[i]) {
            /* Only get result if future completed (avoids THROW_TO_IMMUNE on timeout) */
            if (nimcp_future_is_ready(futures[i])) {
                nimcp_future_get(futures[i], &batch.results[i]);
                if (batch.results[i].executed) batch.num_executed++;
            }
            /* else: batch.results[i] stays zero-initialized (executed=false) */
            nimcp_future_destroy(futures[i]);
        }
        if (promises[i]) nimcp_promise_destroy(promises[i]);
    }

    nimcp_free(args);

    /* Update brain stats from collected results — single-threaded, no contention */
    dispatch_update_stats(brain, &batch);

    batch.total_elapsed_us = nimcp_time_get_us() - t0;

    /* Update dispatch metrics on brain for probe monitoring */
    brain->dispatch_metrics.last_dispatch_us = batch.total_elapsed_us;
    brain->dispatch_metrics.last_modules_executed = batch.num_executed;
    brain->dispatch_metrics.last_modules_submitted = submitted;
    brain->dispatch_metrics.total_dispatches++;
    brain->dispatch_metrics.cumulative_dispatch_us += batch.total_elapsed_us;
    brain->dispatch_metrics.pool_thread_count =
        brain->inference_pool ? (uint32_t)nimcp_pool_active(brain->inference_pool) : 0;

    /* Compute slowest and average module time */
    float slowest = 0.0f;
    float sum = 0.0f;
    uint32_t timed = 0;
    for (uint32_t i = 0; i < COG_MODULE_COUNT; i++) {
        if (batch.results[i].executed) {
            float us = (float)batch.results[i].elapsed_us;
            if (us > slowest) slowest = us;
            sum += us;
            timed++;
        }
    }
    brain->dispatch_metrics.slowest_module_us = slowest;
    brain->dispatch_metrics.avg_module_us = timed > 0 ? sum / timed : 0.0f;

    if (submitted > 0) {
        LOG_DEBUG("Cognitive parallel: %u/%d modules in %lu us (submitted=%u)",
                  batch.num_executed, COG_MODULE_COUNT,
                  (unsigned long)batch.total_elapsed_us, submitted);
    }

    return batch;
}
