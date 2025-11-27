/**
 * @file nimcp_brain_training_integration.c
 * @brief Brain-Training Integration Module Implementation
 *
 * Phase TM-3: Integrates training modules (Loss Functions, Optimizers) with
 * the brain system and security framework.
 *
 * @version 1.0.0
 * @author NIMCP Development Team
 */

#include "middleware/training/nimcp_brain_training_integration.h"
#include "middleware/training/nimcp_training_plasticity_bridge.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/time/nimcp_time.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>

/* ============================================================================
 * Internal Structures
 * ============================================================================ */

/**
 * @brief Loss context slot
 */
typedef struct {
    nimcp_loss_context_t* ctx;
    uint32_t id;
    bool active;
} loss_slot_t;

/**
 * @brief Optimizer context slot
 */
typedef struct {
    nimcp_optimizer_context_t* ctx;
    uint32_t id;
    bool active;
} optimizer_slot_t;

/**
 * @brief LR Scheduler context slot
 */
typedef struct {
    nimcp_lr_scheduler_ctx_t* ctx;
    uint32_t id;
    bool active;
} scheduler_slot_t;

/**
 * @brief Gradient manager context slot
 */
typedef struct {
    nimcp_gradient_manager_ctx_t* ctx;
    uint32_t id;
    bool active;
} gradmgr_slot_t;

/**
 * @brief Brain-Training integration context
 */
struct nimcp_brain_training_ctx {
    /* Configuration */
    nimcp_brain_training_config_t config;

    /* Security integration */
    nimcp_sec_integration_t* security_ctx;
    uint32_t loss_module_id;
    uint32_t optimizer_module_id;
    uint32_t scheduler_module_id;
    uint32_t gradmgr_module_id;
    bool security_registered;

    /* Memory manager */
    unified_mem_manager_t memory_mgr;

    /* Loss function contexts */
    loss_slot_t loss_slots[NIMCP_TRAINING_MAX_LOSS_CONTEXTS];
    uint32_t loss_count;
    uint32_t next_loss_id;

    /* Optimizer contexts */
    optimizer_slot_t optimizer_slots[NIMCP_TRAINING_MAX_OPTIMIZER_CONTEXTS];
    uint32_t optimizer_count;
    uint32_t next_optimizer_id;

    /* LR Scheduler contexts */
    scheduler_slot_t scheduler_slots[NIMCP_TRAINING_MAX_SCHEDULER_CONTEXTS];
    uint32_t scheduler_count;
    uint32_t next_scheduler_id;

    /* Gradient manager contexts */
    gradmgr_slot_t gradmgr_slots[NIMCP_TRAINING_MAX_GRADMGR_CONTEXTS];
    uint32_t gradmgr_count;
    uint32_t next_gradmgr_id;

    /* Training state */
    nimcp_training_mode_t mode;
    uint64_t current_epoch;
    uint64_t current_batch;

    /* Statistics */
    nimcp_training_session_stats_t stats;

    /* Event callback */
    nimcp_training_event_callback_t event_callback;
    void* callback_user_data;

    /* Convergence/divergence tracking */
    float last_loss;
    uint32_t stable_loss_count;
    bool converged;
    bool diverged;

    /* Early stopping state */
    float best_loss;
    uint32_t early_stop_wait;
    bool early_stopped;

    /* Random state for dropout */
    uint32_t dropout_seed;

    /* Plasticity Bridge integration (Phase TPB-1) */
    tpb_context_t* plasticity_bridge;
    float biological_modulation_strength;
    float prev_loss;          /* For RPE computation */
    float cumulative_da;      /* For avg dopamine tracking */
    float cumulative_lr_mod;  /* For avg LR modulation tracking */
    uint64_t bio_update_count;
};

/* ============================================================================
 * Event Type and Mode Names
 * ============================================================================ */

static const char* s_training_event_names[] = {
    "NONE",
    "EPOCH_START",
    "EPOCH_END",
    "BATCH_START",
    "BATCH_END",
    "LOSS_COMPUTED",
    "GRADIENTS_READY",
    "WEIGHTS_UPDATED",
    "LR_CHANGED",
    "CONVERGENCE",
    "DIVERGENCE",
    "GRAD_CLIPPED",
    "GRAD_ACCUMULATED",
    "REGULARIZED",
    "EARLY_STOP"
};

static const char* s_training_mode_names[] = {
    "TRAIN",
    "EVAL",
    "INFERENCE"
};

const char* nimcp_training_event_type_name(nimcp_training_event_type_t type)
{
    if (type >= 0 && type < NIMCP_TRAINING_EVENT_COUNT) {
        return s_training_event_names[type];
    }
    return "UNKNOWN";
}

const char* nimcp_training_mode_name(nimcp_training_mode_t mode)
{
    if (mode >= 0 && mode <= NIMCP_TRAINING_MODE_INFERENCE) {
        return s_training_mode_names[mode];
    }
    return "UNKNOWN";
}

/* ============================================================================
 * Default Configuration
 * ============================================================================ */

nimcp_brain_training_config_t nimcp_brain_training_default_config(void)
{
    nimcp_brain_training_config_t config = {0};

    /* Loss function defaults */
    config.default_loss_type = NIMCP_LOSS_MSE;
    config.default_reduction = NIMCP_LOSS_REDUCE_MEAN;

    /* Optimizer defaults */
    config.default_optimizer = NIMCP_OPTIMIZER_ADAM;
    config.default_learning_rate = 0.001f;

    /* Learning rate scheduler defaults */
    config.default_lr_scheduler = NIMCP_LR_STEP;
    config.enable_lr_scheduler = false;

    /* Regularization defaults */
    config.default_reg_type = NIMCP_REG_L2;
    config.default_reg_lambda = 0.0001f;
    config.l1_lambda = 0.0f;             /* No L1 by default */
    config.l2_lambda = 0.0001f;          /* Small L2 weight decay */
    config.dropout_rate = 0.0f;          /* No dropout by default */
    config.default_clip_mode = NIMCP_CLIP_BY_NORM;
    config.default_clip_value = 1.0f;
    config.gradient_clip_norm = 0.0f;    /* No norm clipping by default */
    config.enable_regularization = false;
    config.enable_gradient_clipping = false;

    /* Gradient management defaults */
    config.gradient_accum_steps = 1;          /* No accumulation by default */
    config.gradient_accumulation_steps = 1;   /* Alias */
    config.gradient_clip_value = 0.0f;        /* No value clipping by default */
    config.enable_gradient_management = false;
    config.enable_gradient_scaling = false;
    config.enable_gradient_health_check = true;

    /* Early stopping */
    config.enable_early_stopping = false;
    config.early_stop_patience = 10;
    config.early_stop_min_delta = 1e-4f;

    /* Training parameters */
    config.max_epochs = 1000;
    config.batch_size = 32;
    config.convergence_threshold = 1e-6f;
    config.divergence_threshold = 1e6f;

    /* Security integration */
    config.enable_security = true;
    config.register_with_bbb = true;

    /* Event callbacks */
    config.event_callback = NULL;
    config.callback_user_data = NULL;

    /* Memory management */
    config.use_memory_pool = true;
    config.cow_strategy = UNIFIED_STRATEGY_AUTO;

    /* Plasticity Bridge defaults (Phase TPB-1) */
    config.enable_plasticity_bridge = false;  /* Disabled by default */
    config.rpe_to_da_gain = 0.5f;
    config.biological_lr_modulation = 0.5f;   /* 50% biological, 50% computational */

    return config;
}

/* ============================================================================
 * Configuration Validation
 * ============================================================================ */

nimcp_result_t nimcp_brain_training_validate_config(
    const nimcp_brain_training_config_t* config)
{
    if (!config) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    if (config->default_loss_type < 0 ||
        config->default_loss_type >= NIMCP_LOSS_TYPE_COUNT) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    if (config->default_optimizer < 0 ||
        config->default_optimizer >= NIMCP_OPTIMIZER_TYPE_COUNT) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    if (config->default_learning_rate <= 0.0f ||
        config->default_learning_rate > 10.0f) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    if (config->convergence_threshold <= 0.0f) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    if (config->divergence_threshold <= 0.0f) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Lifecycle Functions
 * ============================================================================ */

nimcp_brain_training_ctx_t* nimcp_brain_training_create(
    const nimcp_brain_training_config_t* config)
{
    nimcp_brain_training_config_t local_config;

    if (config) {
        if (nimcp_brain_training_validate_config(config) != NIMCP_SUCCESS) {
            LOG_ERROR("Invalid brain-training configuration");
            return NULL;
        }
        local_config = *config;
    } else {
        local_config = nimcp_brain_training_default_config();
    }

    nimcp_brain_training_ctx_t* ctx = calloc(1, sizeof(nimcp_brain_training_ctx_t));
    if (!ctx) {
        LOG_ERROR("Failed to allocate brain-training context");
        return NULL;
    }

    ctx->config = local_config;
    ctx->mode = NIMCP_TRAINING_MODE_TRAIN;
    ctx->next_loss_id = 1;
    ctx->next_optimizer_id = 1;
    ctx->next_scheduler_id = 1;
    ctx->next_gradmgr_id = 1;
    ctx->event_callback = local_config.event_callback;
    ctx->callback_user_data = local_config.callback_user_data;

    /* Initialize statistics */
    ctx->stats.min_loss = DBL_MAX;
    ctx->stats.max_loss = -DBL_MAX;
    ctx->stats.initial_lr = local_config.default_learning_rate;
    ctx->stats.final_lr = local_config.default_learning_rate;
    ctx->stats.lr_min = local_config.default_learning_rate;
    ctx->stats.lr_max = local_config.default_learning_rate;
    ctx->stats.current_grad_scale = 1.0f;

    ctx->last_loss = FLT_MAX;
    ctx->best_loss = FLT_MAX;
    ctx->dropout_seed = (uint32_t)time(NULL);

    /* Initialize plasticity bridge fields (Phase TPB-1) */
    ctx->plasticity_bridge = NULL;
    ctx->biological_modulation_strength = local_config.biological_lr_modulation;
    ctx->prev_loss = FLT_MAX;
    ctx->cumulative_da = 0.0f;
    ctx->cumulative_lr_mod = 0.0f;
    ctx->bio_update_count = 0;

    LOG_INFO("Brain-training context created (Phase TM-3)");
    return ctx;
}

nimcp_result_t nimcp_brain_training_init(
    nimcp_brain_training_ctx_t* ctx,
    nimcp_sec_integration_t* security_ctx,
    unified_mem_manager_t memory_mgr)
{
    if (!ctx) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    ctx->security_ctx = security_ctx;
    ctx->memory_mgr = memory_mgr;

    /* Register with security system if enabled */
    if (ctx->config.enable_security && security_ctx) {
        nimcp_result_t res = nimcp_brain_training_register_security(ctx, security_ctx);
        if (res != NIMCP_SUCCESS) {
            LOG_WARNING("Failed to register training modules with security: %d", res);
            /* Continue anyway - security is optional */
        }
    }

    LOG_INFO("Brain-training integration initialized");
    return NIMCP_SUCCESS;
}

void nimcp_brain_training_destroy(nimcp_brain_training_ctx_t* ctx)
{
    if (!ctx) {
        return;
    }

    /* Unregister from security */
    if (ctx->security_registered) {
        nimcp_brain_training_unregister_security(ctx);
    }

    /* Destroy all loss contexts */
    for (uint32_t i = 0; i < NIMCP_TRAINING_MAX_LOSS_CONTEXTS; i++) {
        if (ctx->loss_slots[i].active && ctx->loss_slots[i].ctx) {
            nimcp_loss_destroy(ctx->loss_slots[i].ctx);
        }
    }

    /* Destroy all optimizer contexts */
    for (uint32_t i = 0; i < NIMCP_TRAINING_MAX_OPTIMIZER_CONTEXTS; i++) {
        if (ctx->optimizer_slots[i].active && ctx->optimizer_slots[i].ctx) {
            nimcp_optimizer_destroy(ctx->optimizer_slots[i].ctx);
        }
    }

    /* Destroy all scheduler contexts */
    for (uint32_t i = 0; i < NIMCP_TRAINING_MAX_SCHEDULER_CONTEXTS; i++) {
        if (ctx->scheduler_slots[i].active && ctx->scheduler_slots[i].ctx) {
            nimcp_lr_scheduler_destroy(ctx->scheduler_slots[i].ctx);
        }
    }

    /* Destroy all gradient manager contexts */
    for (uint32_t i = 0; i < NIMCP_TRAINING_MAX_GRADMGR_CONTEXTS; i++) {
        if (ctx->gradmgr_slots[i].active && ctx->gradmgr_slots[i].ctx) {
            nimcp_gradient_manager_destroy(ctx->gradmgr_slots[i].ctx);
        }
    }

    free(ctx);
    LOG_INFO("Brain-training context destroyed");
}

/* ============================================================================
 * Security Registration
 * ============================================================================ */

nimcp_result_t nimcp_brain_training_register_security(
    nimcp_brain_training_ctx_t* ctx,
    nimcp_sec_integration_t* security_ctx)
{
    if (!ctx || !security_ctx) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    if (ctx->security_registered) {
        return NIMCP_SUCCESS; /* Already registered */
    }

    nimcp_result_t res;

    /* Register loss functions module */
    res = nimcp_sec_register_module(
        security_ctx,
        "training_loss_functions",
        NIMCP_SEC_CAT_PLASTICITY,
        &ctx->loss_module_id
    );
    if (res != NIMCP_SUCCESS) {
        LOG_WARNING("Failed to register loss functions with security: %d", res);
        return res;
    }

    /* Register optimizers module */
    res = nimcp_sec_register_module(
        security_ctx,
        "training_optimizers",
        NIMCP_SEC_CAT_PLASTICITY,
        &ctx->optimizer_module_id
    );
    if (res != NIMCP_SUCCESS) {
        LOG_WARNING("Failed to register optimizers with security: %d", res);
        /* Unregister loss module */
        nimcp_sec_unregister_module(security_ctx, ctx->loss_module_id);
        return res;
    }

    ctx->security_ctx = security_ctx;
    ctx->security_registered = true;

    LOG_INFO("Training modules registered with security (loss_id=%u, opt_id=%u)",
             ctx->loss_module_id, ctx->optimizer_module_id);

    return NIMCP_SUCCESS;
}

nimcp_result_t nimcp_brain_training_unregister_security(
    nimcp_brain_training_ctx_t* ctx)
{
    if (!ctx || !ctx->security_registered || !ctx->security_ctx) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    nimcp_sec_unregister_module(ctx->security_ctx, ctx->loss_module_id);
    nimcp_sec_unregister_module(ctx->security_ctx, ctx->optimizer_module_id);

    ctx->security_registered = false;
    LOG_INFO("Training modules unregistered from security");

    return NIMCP_SUCCESS;
}

bool nimcp_brain_training_is_security_registered(
    const nimcp_brain_training_ctx_t* ctx)
{
    return ctx && ctx->security_registered;
}

nimcp_result_t nimcp_brain_training_get_security_ids(
    const nimcp_brain_training_ctx_t* ctx,
    uint32_t* loss_module_id,
    uint32_t* optimizer_module_id)
{
    if (!ctx) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    if (!ctx->security_registered) {
        return NIMCP_NOT_FOUND;
    }

    if (loss_module_id) {
        *loss_module_id = ctx->loss_module_id;
    }
    if (optimizer_module_id) {
        *optimizer_module_id = ctx->optimizer_module_id;
    }

    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Loss Function Management
 * ============================================================================ */

static int find_free_loss_slot(nimcp_brain_training_ctx_t* ctx)
{
    for (uint32_t i = 0; i < NIMCP_TRAINING_MAX_LOSS_CONTEXTS; i++) {
        if (!ctx->loss_slots[i].active) {
            return (int)i;
        }
    }
    return -1;
}

static int find_loss_slot_by_id(nimcp_brain_training_ctx_t* ctx, uint32_t loss_id)
{
    for (uint32_t i = 0; i < NIMCP_TRAINING_MAX_LOSS_CONTEXTS; i++) {
        if (ctx->loss_slots[i].active && ctx->loss_slots[i].id == loss_id) {
            return (int)i;
        }
    }
    return -1;
}

nimcp_result_t nimcp_brain_training_create_loss(
    nimcp_brain_training_ctx_t* ctx,
    const nimcp_loss_config_t* config,
    uint32_t* loss_id)
{
    if (!ctx || !config || !loss_id) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    int slot = find_free_loss_slot(ctx);
    if (slot < 0) {
        LOG_ERROR("No free loss context slots available");
        return NIMCP_ERROR_MEMORY;
    }

    nimcp_loss_context_t* loss_ctx = nimcp_loss_create(
        config,
        ctx->security_ctx,
        ctx->memory_mgr
    );
    if (!loss_ctx) {
        return NIMCP_ERROR_MEMORY;
    }

    nimcp_result_t res = nimcp_loss_init(loss_ctx);
    if (res != NIMCP_SUCCESS) {
        nimcp_loss_destroy(loss_ctx);
        return res;
    }

    ctx->loss_slots[slot].ctx = loss_ctx;
    ctx->loss_slots[slot].id = ctx->next_loss_id++;
    ctx->loss_slots[slot].active = true;
    ctx->loss_count++;

    *loss_id = ctx->loss_slots[slot].id;

    /* Record successful interaction with security */
    if (ctx->security_registered) {
        NIMCP_SEC_SUCCESS(ctx->security_ctx, ctx->loss_module_id);
    }

    LOG_DEBUG("Created loss context (id=%u, type=%s)",
              *loss_id, nimcp_loss_type_name(config->type));

    return NIMCP_SUCCESS;
}

nimcp_loss_context_t* nimcp_brain_training_get_loss(
    nimcp_brain_training_ctx_t* ctx,
    uint32_t loss_id)
{
    if (!ctx) {
        return NULL;
    }

    int slot = find_loss_slot_by_id(ctx, loss_id);
    if (slot < 0) {
        return NULL;
    }

    return ctx->loss_slots[slot].ctx;
}

nimcp_result_t nimcp_brain_training_destroy_loss(
    nimcp_brain_training_ctx_t* ctx,
    uint32_t loss_id)
{
    if (!ctx) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    int slot = find_loss_slot_by_id(ctx, loss_id);
    if (slot < 0) {
        return NIMCP_NOT_FOUND;
    }

    nimcp_loss_destroy(ctx->loss_slots[slot].ctx);
    ctx->loss_slots[slot].ctx = NULL;
    ctx->loss_slots[slot].active = false;
    ctx->loss_count--;

    LOG_DEBUG("Destroyed loss context (id=%u)", loss_id);
    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Optimizer Management
 * ============================================================================ */

static int find_free_optimizer_slot(nimcp_brain_training_ctx_t* ctx)
{
    for (uint32_t i = 0; i < NIMCP_TRAINING_MAX_OPTIMIZER_CONTEXTS; i++) {
        if (!ctx->optimizer_slots[i].active) {
            return (int)i;
        }
    }
    return -1;
}

static int find_optimizer_slot_by_id(nimcp_brain_training_ctx_t* ctx, uint32_t optimizer_id)
{
    for (uint32_t i = 0; i < NIMCP_TRAINING_MAX_OPTIMIZER_CONTEXTS; i++) {
        if (ctx->optimizer_slots[i].active && ctx->optimizer_slots[i].id == optimizer_id) {
            return (int)i;
        }
    }
    return -1;
}

nimcp_result_t nimcp_brain_training_create_optimizer(
    nimcp_brain_training_ctx_t* ctx,
    const nimcp_optimizer_config_t* config,
    uint32_t* optimizer_id)
{
    if (!ctx || !config || !optimizer_id) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    int slot = find_free_optimizer_slot(ctx);
    if (slot < 0) {
        LOG_ERROR("No free optimizer context slots available");
        return NIMCP_ERROR_MEMORY;
    }

    nimcp_optimizer_context_t* opt_ctx = nimcp_optimizer_create(
        config,
        ctx->security_ctx,
        ctx->memory_mgr
    );
    if (!opt_ctx) {
        return NIMCP_ERROR_MEMORY;
    }

    ctx->optimizer_slots[slot].ctx = opt_ctx;
    ctx->optimizer_slots[slot].id = ctx->next_optimizer_id++;
    ctx->optimizer_slots[slot].active = true;
    ctx->optimizer_count++;

    *optimizer_id = ctx->optimizer_slots[slot].id;

    /* Record successful interaction with security */
    if (ctx->security_registered) {
        NIMCP_SEC_SUCCESS(ctx->security_ctx, ctx->optimizer_module_id);
    }

    LOG_DEBUG("Created optimizer context (id=%u, type=%s)",
              *optimizer_id, nimcp_optimizer_type_name(config->type));

    return NIMCP_SUCCESS;
}

nimcp_optimizer_context_t* nimcp_brain_training_get_optimizer(
    nimcp_brain_training_ctx_t* ctx,
    uint32_t optimizer_id)
{
    if (!ctx) {
        return NULL;
    }

    int slot = find_optimizer_slot_by_id(ctx, optimizer_id);
    if (slot < 0) {
        return NULL;
    }

    return ctx->optimizer_slots[slot].ctx;
}

nimcp_result_t nimcp_brain_training_destroy_optimizer(
    nimcp_brain_training_ctx_t* ctx,
    uint32_t optimizer_id)
{
    if (!ctx) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    int slot = find_optimizer_slot_by_id(ctx, optimizer_id);
    if (slot < 0) {
        return NIMCP_NOT_FOUND;
    }

    nimcp_optimizer_destroy(ctx->optimizer_slots[slot].ctx);
    ctx->optimizer_slots[slot].ctx = NULL;
    ctx->optimizer_slots[slot].active = false;
    ctx->optimizer_count--;

    LOG_DEBUG("Destroyed optimizer context (id=%u)", optimizer_id);
    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Training Operations
 * ============================================================================ */

nimcp_result_t nimcp_brain_training_set_mode(
    nimcp_brain_training_ctx_t* ctx,
    nimcp_training_mode_t mode)
{
    if (!ctx) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    if (mode < NIMCP_TRAINING_MODE_TRAIN || mode > NIMCP_TRAINING_MODE_INFERENCE) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    ctx->mode = mode;
    LOG_DEBUG("Training mode set to %s", nimcp_training_mode_name(mode));
    return NIMCP_SUCCESS;
}

nimcp_training_mode_t nimcp_brain_training_get_mode(
    const nimcp_brain_training_ctx_t* ctx)
{
    if (!ctx) {
        return NIMCP_TRAINING_MODE_INFERENCE;
    }
    return ctx->mode;
}

nimcp_result_t nimcp_brain_training_compute_loss(
    nimcp_brain_training_ctx_t* ctx,
    uint32_t loss_id,
    const float* predictions,
    const float* targets,
    size_t batch_size,
    size_t output_size,
    float* loss_value,
    float* gradients)
{
    if (!ctx || !predictions || !targets || !loss_value) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    nimcp_loss_context_t* loss_ctx = nimcp_brain_training_get_loss(ctx, loss_id);
    if (!loss_ctx) {
        return NIMCP_NOT_FOUND;
    }

    uint64_t start_time = (nimcp_time_get_us() * 1000);

    nimcp_result_t res;

    if (gradients && ctx->mode == NIMCP_TRAINING_MODE_TRAIN) {
        /* Forward + backward pass */
        nimcp_loss_result_t result = {0};
        res = nimcp_loss_forward_backward(
            loss_ctx, predictions, targets,
            batch_size, output_size, &result
        );
        if (res == NIMCP_SUCCESS) {
            *loss_value = result.loss_value;
            if (result.gradients && result.gradient_count > 0) {
                memcpy(gradients, result.gradients,
                       result.gradient_count * sizeof(float));
            }
            nimcp_loss_result_free(&result);
        }
    } else {
        /* Forward pass only */
        nimcp_loss_result_t result = {0};
        res = nimcp_loss_forward(
            loss_ctx, predictions, targets,
            batch_size, output_size, &result
        );
        if (res == NIMCP_SUCCESS) {
            *loss_value = result.loss_value;
            nimcp_loss_result_free(&result);
        }
    }

    if (res != NIMCP_SUCCESS) {
        if (ctx->security_registered) {
            NIMCP_SEC_FAILURE(ctx->security_ctx, ctx->loss_module_id);
        }
        return res;
    }

    /* Update statistics */
    ctx->stats.total_loss += *loss_value;
    if (*loss_value < ctx->stats.min_loss) {
        ctx->stats.min_loss = *loss_value;
    }
    if (*loss_value > ctx->stats.max_loss) {
        ctx->stats.max_loss = *loss_value;
    }
    ctx->stats.total_samples += batch_size;
    ctx->stats.training_time_ns += ((nimcp_time_get_us() * 1000) - start_time);

    /* Check for divergence */
    if (*loss_value > ctx->config.divergence_threshold) {
        ctx->diverged = true;

        nimcp_training_event_t event = {0};
        event.type = NIMCP_TRAINING_EVENT_DIVERGENCE;
        event.timestamp = (nimcp_time_get_us() * 1000);
        event.loss_value = *loss_value;
        nimcp_brain_training_emit_event(ctx, &event);

        LOG_WARNING("Training diverged: loss = %f", *loss_value);
    }

    /* Emit loss computed event */
    nimcp_training_event_t event = {0};
    event.type = NIMCP_TRAINING_EVENT_LOSS_COMPUTED;
    event.timestamp = (nimcp_time_get_us() * 1000);
    event.epoch = ctx->current_epoch;
    event.batch = ctx->current_batch;
    event.loss_value = *loss_value;
    nimcp_brain_training_emit_event(ctx, &event);

    if (ctx->security_registered) {
        NIMCP_SEC_SUCCESS(ctx->security_ctx, ctx->loss_module_id);
    }

    return NIMCP_SUCCESS;
}

nimcp_result_t nimcp_brain_training_optimize(
    nimcp_brain_training_ctx_t* ctx,
    uint32_t optimizer_id,
    float* params,
    const float* gradients,
    size_t count)
{
    if (!ctx || !params || !gradients || count == 0) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    if (ctx->mode != NIMCP_TRAINING_MODE_TRAIN) {
        return NIMCP_SUCCESS; /* Skip optimization in eval/inference mode */
    }

    nimcp_optimizer_context_t* opt_ctx = nimcp_brain_training_get_optimizer(ctx, optimizer_id);
    if (!opt_ctx) {
        return NIMCP_NOT_FOUND;
    }

    uint64_t start_time = (nimcp_time_get_us() * 1000);

    nimcp_result_t res = nimcp_optimizer_step(opt_ctx, params, gradients, count);

    if (res != NIMCP_SUCCESS) {
        if (ctx->security_registered) {
            NIMCP_SEC_FAILURE(ctx->security_ctx, ctx->optimizer_module_id);
        }
        return res;
    }

    /* Update statistics */
    ctx->stats.weight_updates++;
    ctx->stats.final_lr = nimcp_optimizer_get_lr(opt_ctx);
    ctx->stats.training_time_ns += ((nimcp_time_get_us() * 1000) - start_time);

    /* Emit weights updated event */
    nimcp_training_event_t event = {0};
    event.type = NIMCP_TRAINING_EVENT_WEIGHTS_UPDATED;
    event.timestamp = (nimcp_time_get_us() * 1000);
    event.epoch = ctx->current_epoch;
    event.batch = ctx->current_batch;
    event.learning_rate = ctx->stats.final_lr;
    event.gradient_norm = nimcp_optimizer_gradient_norm(gradients, count);
    nimcp_brain_training_emit_event(ctx, &event);

    if (ctx->security_registered) {
        NIMCP_SEC_SUCCESS(ctx->security_ctx, ctx->optimizer_module_id);
    }

    return NIMCP_SUCCESS;
}

nimcp_result_t nimcp_brain_training_step(
    nimcp_brain_training_ctx_t* ctx,
    uint32_t loss_id,
    uint32_t optimizer_id,
    float* params,
    const float* predictions,
    const float* targets,
    size_t batch_size,
    size_t output_size,
    size_t param_count,
    float* loss_value)
{
    if (!ctx || !params || !predictions || !targets || !loss_value) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    if (batch_size == 0 || output_size == 0 || param_count == 0) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    /* Phase TPB-1: If plasticity bridge is connected and biological modulation > 0,
     * automatically route through biological training step */
    if (ctx->plasticity_bridge && ctx->biological_modulation_strength > 0.0f) {
        return nimcp_brain_training_step_biological(
            ctx, loss_id, optimizer_id, params,
            predictions, targets, batch_size, output_size,
            param_count, 0 /* default region */, loss_value
        );
    }

    /* Calculate gradient buffer size (batch_size * output_size for loss gradients) */
    size_t gradient_count = batch_size * output_size;
    size_t gradient_size = gradient_count * sizeof(float);

    /* Allocate temporary gradient buffer using unified memory if available */
    float* gradients = NULL;
    unified_mem_handle_t grad_handle = NULL;

    if (ctx->memory_mgr) {
        /* Use unified memory manager */
        unified_mem_request_t req = {
            .size = gradient_size,
            .initial_data = NULL,
            .strategy = UNIFIED_STRATEGY_POOL_DIRECT,
            .enable_cow = false,
            .alignment = 0
        };
        grad_handle = unified_mem_alloc(ctx->memory_mgr, &req);
        if (grad_handle) {
            gradients = (float*)unified_mem_write(grad_handle);
        }
    }

    /* Fallback to malloc if unified memory not available or failed */
    if (!gradients) {
        gradients = (float*)malloc(gradient_size);
        if (!gradients) {
            return NIMCP_ERROR_MEMORY;
        }
    }

    /* Zero the gradient buffer */
    memset(gradients, 0, gradient_size);

    /* Compute loss and gradients */
    nimcp_result_t res = nimcp_brain_training_compute_loss(
        ctx, loss_id, predictions, targets,
        batch_size, output_size, loss_value, gradients
    );

    if (res != NIMCP_SUCCESS) {
        if (grad_handle) {
            unified_mem_free(grad_handle);
        } else {
            free(gradients);
        }
        return res;
    }

    /* Update weights - use gradients for first param_count elements */
    /* Note: In practice, param_count often equals batch_size * output_size for simple cases */
    size_t update_count = (param_count < gradient_count) ? param_count : gradient_count;
    res = nimcp_brain_training_optimize(
        ctx, optimizer_id, params, gradients, update_count
    );

    /* Free gradient buffer */
    if (grad_handle) {
        unified_mem_free(grad_handle);
    } else {
        free(gradients);
    }

    /* Check for convergence */
    float loss_change = fabsf(*loss_value - ctx->last_loss);
    if (loss_change < ctx->config.convergence_threshold) {
        ctx->stable_loss_count++;
        if (ctx->stable_loss_count >= 10) {
            ctx->converged = true;

            nimcp_training_event_t event = {0};
            event.type = NIMCP_TRAINING_EVENT_CONVERGENCE;
            event.timestamp = (nimcp_time_get_us() * 1000);
            event.loss_value = *loss_value;
            nimcp_brain_training_emit_event(ctx, &event);

            LOG_INFO("Training converged at loss = %f", *loss_value);
        }
    } else {
        ctx->stable_loss_count = 0;
    }
    ctx->last_loss = *loss_value;

    ctx->stats.total_batches++;

    return res;
}

/* ============================================================================
 * Event Bus Integration
 * ============================================================================ */

nimcp_result_t nimcp_brain_training_emit_event(
    nimcp_brain_training_ctx_t* ctx,
    const nimcp_training_event_t* event)
{
    if (!ctx || !event) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    /* Call registered callback */
    if (ctx->event_callback) {
        ctx->event_callback(event, ctx->callback_user_data);
    }

    return NIMCP_SUCCESS;
}

nimcp_result_t nimcp_brain_training_register_callback(
    nimcp_brain_training_ctx_t* ctx,
    nimcp_training_event_callback_t callback,
    void* user_data)
{
    if (!ctx) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    ctx->event_callback = callback;
    ctx->callback_user_data = user_data;

    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Statistics and Monitoring
 * ============================================================================ */

nimcp_result_t nimcp_brain_training_get_stats(
    const nimcp_brain_training_ctx_t* ctx,
    nimcp_training_session_stats_t* stats)
{
    if (!ctx || !stats) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    *stats = ctx->stats;

    /* Compute average loss */
    if (ctx->stats.total_batches > 0) {
        stats->avg_loss = ctx->stats.total_loss / (double)ctx->stats.total_batches;
    }

    stats->converged = ctx->converged;
    stats->diverged = ctx->diverged;

    return NIMCP_SUCCESS;
}

void nimcp_brain_training_reset_stats(nimcp_brain_training_ctx_t* ctx)
{
    if (!ctx) {
        return;
    }

    memset(&ctx->stats, 0, sizeof(nimcp_training_session_stats_t));
    ctx->stats.min_loss = DBL_MAX;
    ctx->stats.max_loss = -DBL_MAX;
    ctx->stats.initial_lr = ctx->config.default_learning_rate;
    ctx->stats.final_lr = ctx->config.default_learning_rate;

    ctx->last_loss = FLT_MAX;
    ctx->stable_loss_count = 0;
    ctx->converged = false;
    ctx->diverged = false;
    ctx->current_epoch = 0;
    ctx->current_batch = 0;
}

bool nimcp_brain_training_is_converged(const nimcp_brain_training_ctx_t* ctx)
{
    return ctx && ctx->converged;
}

bool nimcp_brain_training_is_diverged(const nimcp_brain_training_ctx_t* ctx)
{
    return ctx && ctx->diverged;
}

/* ============================================================================
 * LR Scheduler Management
 * ============================================================================ */

static int find_free_scheduler_slot(nimcp_brain_training_ctx_t* ctx)
{
    for (uint32_t i = 0; i < NIMCP_TRAINING_MAX_SCHEDULER_CONTEXTS; i++) {
        if (!ctx->scheduler_slots[i].active) {
            return (int)i;
        }
    }
    return -1;
}

static int find_scheduler_slot_by_id(nimcp_brain_training_ctx_t* ctx, uint32_t scheduler_id)
{
    for (uint32_t i = 0; i < NIMCP_TRAINING_MAX_SCHEDULER_CONTEXTS; i++) {
        if (ctx->scheduler_slots[i].active && ctx->scheduler_slots[i].id == scheduler_id) {
            return (int)i;
        }
    }
    return -1;
}

nimcp_result_t nimcp_brain_training_create_scheduler(
    nimcp_brain_training_ctx_t* ctx,
    const nimcp_lr_scheduler_config_t* config,
    uint32_t* scheduler_id)
{
    if (!ctx || !config || !scheduler_id) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    int slot = find_free_scheduler_slot(ctx);
    if (slot < 0) {
        LOG_ERROR("No free scheduler slots available");
        return NIMCP_ERROR_MEMORY;
    }

    nimcp_lr_scheduler_ctx_t* sched_ctx = nimcp_lr_scheduler_create(config);
    if (!sched_ctx) {
        return NIMCP_ERROR_MEMORY;
    }

    ctx->scheduler_slots[slot].ctx = sched_ctx;
    ctx->scheduler_slots[slot].id = ctx->next_scheduler_id++;
    ctx->scheduler_slots[slot].active = true;
    ctx->scheduler_count++;

    *scheduler_id = ctx->scheduler_slots[slot].id;

    LOG_DEBUG("Created LR scheduler context (id=%u, type=%s)",
              *scheduler_id, nimcp_lr_scheduler_type_name(config->type));

    return NIMCP_SUCCESS;
}

nimcp_lr_scheduler_ctx_t* nimcp_brain_training_get_scheduler(
    nimcp_brain_training_ctx_t* ctx,
    uint32_t scheduler_id)
{
    if (!ctx) {
        return NULL;
    }

    int slot = find_scheduler_slot_by_id(ctx, scheduler_id);
    if (slot < 0) {
        return NULL;
    }

    return ctx->scheduler_slots[slot].ctx;
}

nimcp_result_t nimcp_brain_training_destroy_scheduler(
    nimcp_brain_training_ctx_t* ctx,
    uint32_t scheduler_id)
{
    if (!ctx) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    int slot = find_scheduler_slot_by_id(ctx, scheduler_id);
    if (slot < 0) {
        return NIMCP_NOT_FOUND;
    }

    nimcp_lr_scheduler_destroy(ctx->scheduler_slots[slot].ctx);
    ctx->scheduler_slots[slot].ctx = NULL;
    ctx->scheduler_slots[slot].active = false;
    ctx->scheduler_count--;

    LOG_DEBUG("Destroyed scheduler context (id=%u)", scheduler_id);
    return NIMCP_SUCCESS;
}

float nimcp_brain_training_step_scheduler(
    nimcp_brain_training_ctx_t* ctx,
    uint32_t scheduler_id,
    uint32_t optimizer_id)
{
    if (!ctx) {
        return 0.0f;
    }

    nimcp_lr_scheduler_ctx_t* sched = nimcp_brain_training_get_scheduler(ctx, scheduler_id);
    if (!sched) {
        return 0.0f;
    }

    float new_lr = nimcp_lr_scheduler_step(sched);

    /* Update optimizer learning rate */
    nimcp_optimizer_context_t* opt = nimcp_brain_training_get_optimizer(ctx, optimizer_id);
    if (opt) {
        nimcp_optimizer_set_lr(opt, new_lr);
    }

    /* Update stats */
    ctx->stats.lr_steps++;
    ctx->stats.final_lr = new_lr;
    if (new_lr < ctx->stats.lr_min) ctx->stats.lr_min = new_lr;
    if (new_lr > ctx->stats.lr_max) ctx->stats.lr_max = new_lr;

    /* Emit LR changed event */
    nimcp_training_event_t event = {0};
    event.type = NIMCP_TRAINING_EVENT_LR_CHANGED;
    event.timestamp = (nimcp_time_get_us() * 1000);
    event.learning_rate = new_lr;
    nimcp_brain_training_emit_event(ctx, &event);

    return new_lr;
}

float nimcp_brain_training_step_scheduler_metric(
    nimcp_brain_training_ctx_t* ctx,
    uint32_t scheduler_id,
    uint32_t optimizer_id,
    float metric)
{
    if (!ctx) {
        return 0.0f;
    }

    nimcp_lr_scheduler_ctx_t* sched = nimcp_brain_training_get_scheduler(ctx, scheduler_id);
    if (!sched) {
        return 0.0f;
    }

    float new_lr = nimcp_lr_scheduler_step_metric(sched, metric);

    /* Update optimizer learning rate */
    nimcp_optimizer_context_t* opt = nimcp_brain_training_get_optimizer(ctx, optimizer_id);
    if (opt) {
        nimcp_optimizer_set_lr(opt, new_lr);
    }

    /* Update stats */
    ctx->stats.lr_steps++;
    ctx->stats.final_lr = new_lr;
    if (new_lr < ctx->stats.lr_min) ctx->stats.lr_min = new_lr;
    if (new_lr > ctx->stats.lr_max) ctx->stats.lr_max = new_lr;

    return new_lr;
}

/* ============================================================================
 * Gradient Management
 * ============================================================================ */

static int find_free_gradmgr_slot(nimcp_brain_training_ctx_t* ctx)
{
    for (uint32_t i = 0; i < NIMCP_TRAINING_MAX_GRADMGR_CONTEXTS; i++) {
        if (!ctx->gradmgr_slots[i].active) {
            return (int)i;
        }
    }
    return -1;
}

static int find_gradmgr_slot_by_id(nimcp_brain_training_ctx_t* ctx, uint32_t gradmgr_id)
{
    for (uint32_t i = 0; i < NIMCP_TRAINING_MAX_GRADMGR_CONTEXTS; i++) {
        if (ctx->gradmgr_slots[i].active && ctx->gradmgr_slots[i].id == gradmgr_id) {
            return (int)i;
        }
    }
    return -1;
}

nimcp_result_t nimcp_brain_training_create_gradient_manager(
    nimcp_brain_training_ctx_t* ctx,
    const nimcp_gradient_manager_config_t* config,
    uint32_t* gradmgr_id)
{
    if (!ctx || !config || !gradmgr_id) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    int slot = find_free_gradmgr_slot(ctx);
    if (slot < 0) {
        LOG_ERROR("No free gradient manager slots available");
        return NIMCP_ERROR_MEMORY;
    }

    nimcp_gradient_manager_ctx_t* gm_ctx = nimcp_gradient_manager_create(config);
    if (!gm_ctx) {
        return NIMCP_ERROR_MEMORY;
    }

    ctx->gradmgr_slots[slot].ctx = gm_ctx;
    ctx->gradmgr_slots[slot].id = ctx->next_gradmgr_id++;
    ctx->gradmgr_slots[slot].active = true;
    ctx->gradmgr_count++;

    *gradmgr_id = ctx->gradmgr_slots[slot].id;

    LOG_DEBUG("Created gradient manager context (id=%u)", *gradmgr_id);

    return NIMCP_SUCCESS;
}

nimcp_gradient_manager_ctx_t* nimcp_brain_training_get_gradient_manager(
    nimcp_brain_training_ctx_t* ctx,
    uint32_t gradmgr_id)
{
    if (!ctx) {
        return NULL;
    }

    int slot = find_gradmgr_slot_by_id(ctx, gradmgr_id);
    if (slot < 0) {
        return NULL;
    }

    return ctx->gradmgr_slots[slot].ctx;
}

nimcp_result_t nimcp_brain_training_destroy_gradient_manager(
    nimcp_brain_training_ctx_t* ctx,
    uint32_t gradmgr_id)
{
    if (!ctx) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    int slot = find_gradmgr_slot_by_id(ctx, gradmgr_id);
    if (slot < 0) {
        return NIMCP_NOT_FOUND;
    }

    nimcp_gradient_manager_destroy(ctx->gradmgr_slots[slot].ctx);
    ctx->gradmgr_slots[slot].ctx = NULL;
    ctx->gradmgr_slots[slot].active = false;
    ctx->gradmgr_count--;

    LOG_DEBUG("Destroyed gradient manager context (id=%u)", gradmgr_id);
    return NIMCP_SUCCESS;
}

nimcp_result_t nimcp_brain_training_accumulate_gradients(
    nimcp_brain_training_ctx_t* ctx,
    uint32_t gradmgr_id,
    const float* gradients,
    size_t count)
{
    if (!ctx || !gradients || count == 0) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    nimcp_gradient_manager_ctx_t* gm = nimcp_brain_training_get_gradient_manager(ctx, gradmgr_id);
    if (!gm) {
        return NIMCP_NOT_FOUND;
    }

    nimcp_result_t res = nimcp_gradient_accumulate(gm, gradients, count);
    if (res == NIMCP_SUCCESS) {
        ctx->stats.grad_accum_steps++;

        /* Emit accumulation event */
        nimcp_training_event_t event = {0};
        event.type = NIMCP_TRAINING_EVENT_GRAD_ACCUMULATED;
        event.timestamp = (nimcp_time_get_us() * 1000);
        event.accum_step = nimcp_gradient_get_accum_step(gm);
        nimcp_brain_training_emit_event(ctx, &event);
    }

    return res;
}

bool nimcp_brain_training_gradients_ready(
    const nimcp_brain_training_ctx_t* ctx,
    uint32_t gradmgr_id)
{
    if (!ctx) {
        return false;
    }

    /* Find slot without modifying ctx */
    for (uint32_t i = 0; i < NIMCP_TRAINING_MAX_GRADMGR_CONTEXTS; i++) {
        if (ctx->gradmgr_slots[i].active && ctx->gradmgr_slots[i].id == gradmgr_id) {
            return nimcp_gradient_accum_ready(ctx->gradmgr_slots[i].ctx);
        }
    }
    return false;
}

nimcp_result_t nimcp_brain_training_get_accumulated_gradients(
    nimcp_brain_training_ctx_t* ctx,
    uint32_t gradmgr_id,
    float* output,
    size_t count)
{
    if (!ctx || !output || count == 0) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    nimcp_gradient_manager_ctx_t* gm = nimcp_brain_training_get_gradient_manager(ctx, gradmgr_id);
    if (!gm) {
        return NIMCP_NOT_FOUND;
    }

    nimcp_result_t res = nimcp_gradient_get_accumulated(gm, output, count);
    if (res == NIMCP_SUCCESS) {
        nimcp_gradient_reset_accum(gm);
    }

    return res;
}

/* ============================================================================
 * Regularization Operations
 * ============================================================================ */

float nimcp_brain_training_apply_regularization(
    nimcp_brain_training_ctx_t* ctx,
    const float* weights,
    float* gradients,
    size_t count,
    nimcp_reg_type_t reg_type,
    float lambda)
{
    if (!ctx || !weights || !gradients || count == 0) {
        return 0.0f;
    }

    float reg_loss = 0.0f;

    switch (reg_type) {
        case NIMCP_REG_L1:
            reg_loss = nimcp_l1_loss(weights, count, lambda);
            nimcp_l1_gradient(weights, gradients, count, lambda);
            break;
        case NIMCP_REG_L2:
            reg_loss = nimcp_l2_loss(weights, count, lambda);
            nimcp_l2_gradient(weights, gradients, count, lambda);
            break;
        case NIMCP_REG_ELASTIC_NET: {
            /* Elastic net = alpha*L1 + (1-alpha)*L2, use alpha=0.5 */
            float alpha = 0.5f;
            reg_loss = nimcp_elastic_net_loss(weights, count, lambda, alpha);
            /* Apply both L1 and L2 gradients with appropriate weights */
            nimcp_l1_gradient(weights, gradients, count, lambda * alpha);
            nimcp_l2_gradient(weights, gradients, count, lambda * (1.0f - alpha));
            break;
        }
        default:
            break;
    }

    ctx->stats.total_reg_loss += reg_loss;

    /* Emit regularization event */
    nimcp_training_event_t event = {0};
    event.type = NIMCP_TRAINING_EVENT_REGULARIZED;
    event.timestamp = (nimcp_time_get_us() * 1000);
    event.regularization_loss = reg_loss;
    nimcp_brain_training_emit_event(ctx, &event);

    return reg_loss;
}

float nimcp_brain_training_clip_gradients(
    nimcp_brain_training_ctx_t* ctx,
    float* gradients,
    size_t count,
    nimcp_clip_mode_t mode,
    float threshold)
{
    if (!ctx || !gradients || count == 0) {
        return 1.0f;
    }

    float original_norm = nimcp_gradient_norm(gradients, count);
    float clip_ratio = 1.0f;

    switch (mode) {
        case NIMCP_CLIP_BY_VALUE:
            nimcp_gradient_clip_by_value(gradients, count, threshold);
            break;
        case NIMCP_CLIP_BY_NORM:
        case NIMCP_CLIP_BY_GLOBAL_NORM:
            /* Both norm-based clipping use the same function */
            clip_ratio = nimcp_gradient_clip_by_norm(gradients, count, threshold);
            break;
        default:
            return 1.0f;
    }

    /* For value clipping, compute ratio from norms */
    if (mode == NIMCP_CLIP_BY_VALUE) {
        float clipped_norm = nimcp_gradient_norm(gradients, count);
        if (clipped_norm > 0.0f) {
            clip_ratio = original_norm / clipped_norm;
        }
    }

    if (clip_ratio > 1.0f) {
        ctx->stats.gradient_clips++;

        /* Emit clip event */
        nimcp_training_event_t event = {0};
        event.type = NIMCP_TRAINING_EVENT_GRAD_CLIPPED;
        event.timestamp = (nimcp_time_get_us() * 1000);
        event.clip_ratio = clip_ratio;
        event.gradient_norm = original_norm;
        nimcp_brain_training_emit_event(ctx, &event);
    }

    return clip_ratio;
}

uint64_t nimcp_brain_training_apply_dropout(
    nimcp_brain_training_ctx_t* ctx,
    const float* input,
    float* output,
    bool* mask,
    size_t count,
    float dropout_rate)
{
    if (!ctx || !input || !output || count == 0) {
        return 0;
    }

    /* Only apply dropout in training mode */
    if (ctx->mode != NIMCP_TRAINING_MODE_TRAIN) {
        if (input != output) {
            memcpy(output, input, count * sizeof(float));
        }
        return 0;
    }

    /* Create temporary dropout context */
    nimcp_dropout_config_t cfg = nimcp_dropout_default_config(dropout_rate);
    nimcp_dropout_ctx_t* dropout_ctx = nimcp_dropout_create(&cfg);
    if (!dropout_ctx) {
        if (input != output) {
            memcpy(output, input, count * sizeof(float));
        }
        return 0;
    }

    /* Copy input to output first (dropout modifies in place) */
    if (input != output) {
        memcpy(output, input, count * sizeof(float));
    }

    /* Apply dropout */
    uint8_t* dropout_mask = mask ? (uint8_t*)malloc(count) : NULL;
    nimcp_result_t res = nimcp_dropout_forward(dropout_ctx, output, count, dropout_mask);

    /* Count dropped elements */
    uint64_t dropped = 0;
    if (res == NIMCP_SUCCESS && dropout_mask) {
        for (size_t i = 0; i < count; i++) {
            if (dropout_mask[i] == 0) dropped++;
        }
        /* Convert mask to bool array if requested */
        if (mask) {
            for (size_t i = 0; i < count; i++) {
                mask[i] = (dropout_mask[i] != 0);
            }
        }
    }

    if (dropout_mask) {
        free(dropout_mask);
    }
    nimcp_dropout_destroy(dropout_ctx);

    ctx->stats.dropout_masks++;
    return dropped;
}

bool nimcp_brain_training_check_early_stop(
    nimcp_brain_training_ctx_t* ctx,
    float current_loss)
{
    if (!ctx || !ctx->config.enable_early_stopping) {
        return false;
    }

    if (ctx->early_stopped) {
        return true;  /* Already stopped */
    }

    /* Check for improvement */
    float improvement = ctx->best_loss - current_loss;

    if (improvement > ctx->config.early_stop_min_delta) {
        /* Improvement found */
        ctx->best_loss = current_loss;
        ctx->early_stop_wait = 0;
    } else {
        /* No improvement */
        ctx->early_stop_wait++;
    }

    ctx->stats.early_stop_wait_count = ctx->early_stop_wait;
    ctx->stats.best_loss = ctx->best_loss;

    if (ctx->early_stop_wait >= ctx->config.early_stop_patience) {
        ctx->early_stopped = true;
        ctx->stats.early_stopped = true;

        /* Emit early stop event */
        nimcp_training_event_t event = {0};
        event.type = NIMCP_TRAINING_EVENT_EARLY_STOP;
        event.timestamp = (nimcp_time_get_us() * 1000);
        event.loss_value = current_loss;
        nimcp_brain_training_emit_event(ctx, &event);

        LOG_INFO("Early stopping triggered at loss = %f", current_loss);
        return true;
    }

    return false;
}

void nimcp_brain_training_reset_early_stop(nimcp_brain_training_ctx_t* ctx)
{
    if (!ctx) {
        return;
    }

    ctx->best_loss = FLT_MAX;
    ctx->early_stop_wait = 0;
    ctx->early_stopped = false;
    ctx->stats.early_stop_wait_count = 0;
    ctx->stats.best_loss = FLT_MAX;
    ctx->stats.early_stopped = false;
}

/* ============================================================================
 * Enhanced Training Operations
 * ============================================================================ */

nimcp_result_t nimcp_brain_training_step_full(
    nimcp_brain_training_ctx_t* ctx,
    uint32_t loss_id,
    uint32_t optimizer_id,
    uint32_t scheduler_id,
    uint32_t gradmgr_id,
    float* params,
    const float* predictions,
    const float* targets,
    size_t batch_size,
    size_t output_size,
    size_t param_count,
    float* loss_value)
{
    if (!ctx || !params || !predictions || !targets || !loss_value) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    if (batch_size == 0 || output_size == 0 || param_count == 0) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    /* Allocate gradient buffer */
    size_t gradient_count = batch_size * output_size;
    float* gradients = (float*)malloc(gradient_count * sizeof(float));
    if (!gradients) {
        return NIMCP_ERROR_MEMORY;
    }
    memset(gradients, 0, gradient_count * sizeof(float));

    /* Step 1: Compute loss and gradients */
    nimcp_result_t res = nimcp_brain_training_compute_loss(
        ctx, loss_id, predictions, targets,
        batch_size, output_size, loss_value, gradients
    );

    if (res != NIMCP_SUCCESS) {
        free(gradients);
        return res;
    }

    /* Phase TPB-1: Report loss to plasticity bridge for RPE computation */
    if (ctx->plasticity_bridge && ctx->biological_modulation_strength > 0.0f) {
        float rpe = 0.0f;
        if (tpb_report_loss(ctx->plasticity_bridge, *loss_value, &rpe) == NIMCP_SUCCESS) {
            ctx->stats.rpe_computations++;

            /* Get current dopamine level for tracking */
            float da_level = 0.5f, ach_level, ht5_level, ne_level;
            if (tpb_get_neuromod_levels(ctx->plasticity_bridge, &da_level, &ach_level,
                                         &ht5_level, &ne_level) == NIMCP_SUCCESS) {
                ctx->cumulative_da += da_level;
            }
        }
    }

    /* Step 2: Check gradient health */
    if (ctx->config.enable_gradient_health_check) {
        nimcp_grad_health_t health = nimcp_gradient_check_health(gradients, gradient_count);
        if (health == NIMCP_GRAD_HAS_NAN) {
            ctx->stats.grad_nan_count++;
            free(gradients);
            return NIMCP_TRAINING_ERROR_GRAD_NAN;  /* Skip this step */
        }
        if (health == NIMCP_GRAD_HAS_INF) {
            ctx->stats.grad_inf_count++;
            /* Sanitize infinity values */
            nimcp_gradient_sanitize(gradients, gradient_count, 0.0f);
        }
    }

    /* Step 3: Gradient accumulation (if using gradient manager) */
    if (gradmgr_id > 0) {
        nimcp_gradient_manager_ctx_t* gm = nimcp_brain_training_get_gradient_manager(ctx, gradmgr_id);
        if (gm) {
            res = nimcp_gradient_accumulate(gm, gradients, gradient_count);
            ctx->stats.grad_accum_steps++;

            if (!nimcp_gradient_accum_ready(gm)) {
                /* Not ready to apply yet, skip optimization */
                free(gradients);
                return NIMCP_SUCCESS;
            }

            /* Get accumulated gradients */
            res = nimcp_gradient_get_accumulated(gm, gradients, gradient_count);
            nimcp_gradient_reset_accum(gm);
        }
    }

    /* Step 4: Gradient clipping (if enabled) */
    if (ctx->config.enable_gradient_clipping) {
        nimcp_brain_training_clip_gradients(
            ctx, gradients, gradient_count,
            ctx->config.default_clip_mode,
            ctx->config.default_clip_value
        );
    }

    /* Step 5: Add regularization gradients (if enabled) */
    if (ctx->config.enable_regularization) {
        float reg_loss = nimcp_brain_training_apply_regularization(
            ctx, params, gradients,
            (param_count < gradient_count) ? param_count : gradient_count,
            ctx->config.default_reg_type,
            ctx->config.default_reg_lambda
        );
        *loss_value += reg_loss;
    }

    /* Phase TPB-1: Apply biological learning rate modulation before optimization */
    float original_lr = 0.0f;
    nimcp_optimizer_context_t* opt = NULL;
    if (ctx->plasticity_bridge && ctx->biological_modulation_strength > 0.0f) {
        opt = nimcp_brain_training_get_optimizer(ctx, optimizer_id);
        if (opt) {
            original_lr = nimcp_optimizer_get_lr(opt);
            float bio_lr = original_lr;
            if (tpb_get_modulated_lr(ctx->plasticity_bridge, 0, original_lr, &bio_lr) == NIMCP_SUCCESS) {
                float blend = ctx->biological_modulation_strength;
                float effective_lr = (1.0f - blend) * original_lr + blend * bio_lr;
                nimcp_optimizer_set_lr(opt, effective_lr);
                ctx->cumulative_lr_mod += (effective_lr / original_lr);
                ctx->bio_update_count++;
            }
        }
    }

    /* Step 6: Optimization step */
    size_t update_count = (param_count < gradient_count) ? param_count : gradient_count;
    res = nimcp_brain_training_optimize(ctx, optimizer_id, params, gradients, update_count);

    /* Restore original learning rate if modified */
    if (opt && original_lr > 0.0f) {
        nimcp_optimizer_set_lr(opt, original_lr);
    }

    free(gradients);

    if (res != NIMCP_SUCCESS) {
        return res;
    }

    /* Step 7: Learning rate scheduling (if enabled) */
    if (scheduler_id > 0 && ctx->config.enable_lr_scheduler) {
        nimcp_brain_training_step_scheduler(ctx, scheduler_id, optimizer_id);
    }

    /* Step 8: Check early stopping */
    if (ctx->config.enable_early_stopping) {
        if (nimcp_brain_training_check_early_stop(ctx, *loss_value)) {
            return NIMCP_NOT_FOUND;  /* Use as NIMCP_EARLY_STOP indicator */
        }
    }

    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Plasticity Bridge Integration (Phase TPB-1)
 * ============================================================================ */

nimcp_result_t nimcp_brain_training_connect_plasticity_bridge(
    nimcp_brain_training_ctx_t* ctx,
    tpb_context_t* bridge)
{
    if (!ctx) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    ctx->plasticity_bridge = bridge;

    if (bridge) {
        LOG_INFO("Plasticity bridge connected to training integration");
    } else {
        LOG_INFO("Plasticity bridge disconnected from training integration");
    }

    return NIMCP_SUCCESS;
}

tpb_context_t* nimcp_brain_training_get_plasticity_bridge(
    const nimcp_brain_training_ctx_t* ctx)
{
    if (!ctx) {
        return NULL;
    }
    return ctx->plasticity_bridge;
}

nimcp_result_t nimcp_brain_training_set_biological_modulation(
    nimcp_brain_training_ctx_t* ctx,
    float strength)
{
    if (!ctx) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    if (strength < 0.0f) strength = 0.0f;
    if (strength > 1.0f) strength = 1.0f;

    ctx->biological_modulation_strength = strength;
    LOG_DEBUG("Biological modulation strength set to %.2f", strength);

    return NIMCP_SUCCESS;
}

float nimcp_brain_training_get_biological_modulation(
    const nimcp_brain_training_ctx_t* ctx)
{
    if (!ctx) {
        return 0.0f;
    }
    return ctx->biological_modulation_strength;
}

nimcp_result_t nimcp_brain_training_step_biological(
    nimcp_brain_training_ctx_t* ctx,
    uint32_t loss_id,
    uint32_t optimizer_id,
    float* params,
    const float* predictions,
    const float* targets,
    size_t batch_size,
    size_t output_size,
    size_t param_count,
    uint32_t region_id,
    float* loss_value)
{
    if (!ctx || !params || !predictions || !targets || !loss_value) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    if (batch_size == 0 || output_size == 0 || param_count == 0) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    /* If no plasticity bridge or biological modulation is zero, fallback to regular step */
    if (!ctx->plasticity_bridge || ctx->biological_modulation_strength <= 0.0f) {
        return nimcp_brain_training_step(
            ctx, loss_id, optimizer_id, params,
            predictions, targets, batch_size, output_size,
            param_count, loss_value
        );
    }

    /* Allocate gradient buffer */
    size_t gradient_count = batch_size * output_size;
    float* gradients = (float*)malloc(gradient_count * sizeof(float));
    if (!gradients) {
        return NIMCP_ERROR_MEMORY;
    }
    memset(gradients, 0, gradient_count * sizeof(float));

    /* Step 1: Compute loss and gradients */
    nimcp_result_t res = nimcp_brain_training_compute_loss(
        ctx, loss_id, predictions, targets,
        batch_size, output_size, loss_value, gradients
    );

    if (res != NIMCP_SUCCESS) {
        free(gradients);
        return res;
    }

    /* Step 2: Report loss to plasticity bridge and get RPE */
    float rpe = 0.0f;
    res = tpb_report_loss(ctx->plasticity_bridge, *loss_value, &rpe);
    if (res == NIMCP_SUCCESS) {
        ctx->stats.rpe_computations++;

        /* Get current dopamine level from neuromodulator system */
        float da_level = 0.5f, ach_level, ht5_level, ne_level;
        if (tpb_get_neuromod_levels(ctx->plasticity_bridge, &da_level, &ach_level,
                                     &ht5_level, &ne_level) == NIMCP_SUCCESS) {
            ctx->cumulative_da += da_level;
        }
    }

    /* Step 3: Get biologically-modulated learning rate */
    nimcp_optimizer_context_t* opt = nimcp_brain_training_get_optimizer(ctx, optimizer_id);
    float base_lr = opt ? nimcp_optimizer_get_lr(opt) : ctx->config.default_learning_rate;

    float bio_lr = base_lr;
    tpb_get_modulated_lr(ctx->plasticity_bridge, region_id, base_lr, &bio_lr);
    float blend = ctx->biological_modulation_strength;
    float effective_lr = (1.0f - blend) * base_lr + blend * bio_lr;

    ctx->cumulative_lr_mod += (effective_lr / base_lr);

    /* Temporarily set modulated learning rate */
    if (opt) {
        nimcp_optimizer_set_lr(opt, effective_lr);
    }

    /* Step 4: Route weight updates through plasticity bridge */
    size_t update_count = (param_count < gradient_count) ? param_count : gradient_count;

    for (size_t i = 0; i < update_count; i++) {
        /* For biological routing, we compute weight delta through the bridge */
        float pre_activity = fabsf(gradients[i]);  /* Use gradient magnitude as proxy */
        float post_activity = fabsf(params[i]);     /* Use weight magnitude as proxy */
        float spike_delta = (gradients[i] > 0) ? 0.01f : -0.01f;  /* Sign indicates timing */

        float weight_delta = 0.0f;
        nimcp_result_t route_res = tpb_route_weight_update(
            ctx->plasticity_bridge,
            (uint32_t)i,    /* neuron_id - use index */
            pre_activity,
            post_activity,
            spike_delta,
            &weight_delta
        );

        /* Blend biological update with gradient-based update */
        float grad_delta = -effective_lr * gradients[i];
        float final_delta;
        if (route_res == NIMCP_SUCCESS) {
            final_delta = (1.0f - blend) * grad_delta + blend * weight_delta;
        } else {
            final_delta = grad_delta;  /* Fallback to pure gradient if routing fails */
        }
        params[i] += final_delta;
    }

    ctx->stats.biological_updates += update_count;
    ctx->bio_update_count++;

    /* Restore base learning rate */
    if (opt) {
        nimcp_optimizer_set_lr(opt, base_lr);
    }

    /* Update tracking */
    ctx->prev_loss = *loss_value;

    /* Update average stats */
    if (ctx->bio_update_count > 0) {
        ctx->stats.avg_dopamine_level = ctx->cumulative_da / (float)ctx->stats.rpe_computations;
        ctx->stats.avg_lr_modulation = ctx->cumulative_lr_mod / (float)ctx->bio_update_count;
    }

    free(gradients);

    /* Check convergence/divergence */
    float loss_change = fabsf(*loss_value - ctx->last_loss);
    if (loss_change < ctx->config.convergence_threshold) {
        ctx->stable_loss_count++;
        if (ctx->stable_loss_count >= 10) {
            ctx->converged = true;
            LOG_INFO("Biological training converged at loss = %f", *loss_value);
        }
    } else {
        ctx->stable_loss_count = 0;
    }
    ctx->last_loss = *loss_value;
    ctx->stats.total_batches++;

    return NIMCP_SUCCESS;
}
