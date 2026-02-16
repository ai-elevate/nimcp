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
#include "security/nimcp_security.h"
#include "security/nimcp_blood_brain_barrier.h"

#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"

#include "middleware/training/nimcp_training_plasticity_bridge.h"
#include "plasticity/nimcp_second_messengers.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/time/nimcp_time.h"
#include "utils/platform/nimcp_platform_tier.h"
#include "portia/nimcp_portia.h"
#include "api/nimcp_api_exception.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <time.h>

#include <stddef.h>  /* for NULL */
#include "utils/memory/nimcp_memory.h"
#include "utils/fault_tolerance/nimcp_health_agent_macros.h"
#include "constants/nimcp_dimension_constants.h"

NIMCP_DECLARE_HEALTH_AGENT_ATOMIC(brain_training_integration)

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

    /* Training Callbacks integration (Phase TCB-1) */
    tcb_context_t* callbacks;

    /* Bio-async integration */
    bio_module_context_t bio_ctx;
    bool bio_async_enabled;

    /* Second Messenger Cascade integration */
    second_messenger_system_t* second_messengers;
    bool second_messengers_enabled;

    /* Portia Resource Management integration */
    void* portia_context;              /* portia_context_t* - opaque to avoid circular dep */
    platform_tier_t current_tier;
    bool resource_aware_training;
    float tier_batch_size_multiplier;
    float tier_lr_multiplier;
    bool training_paused;
    uint32_t degradation_level;
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
    config.default_learning_rate = 0.001F;

    /* Learning rate scheduler defaults */
    config.default_lr_scheduler = NIMCP_LR_STEP;
    config.enable_lr_scheduler = false;

    /* Regularization defaults */
    config.default_reg_type = NIMCP_REG_L2;
    config.default_reg_lambda = 0.0001F;
    config.l1_lambda = 0.0F;             /* No L1 by default */
    config.l2_lambda = 0.0001F;          /* Small L2 weight decay */
    config.dropout_rate = 0.0F;          /* No dropout by default */
    config.default_clip_mode = NIMCP_CLIP_BY_NORM;
    config.default_clip_value = 1.0F;
    config.gradient_clip_norm = 0.0F;    /* No norm clipping by default */
    config.enable_regularization = false;
    config.enable_gradient_clipping = false;

    /* Gradient management defaults */
    config.gradient_accum_steps = 1;          /* No accumulation by default */
    config.gradient_accumulation_steps = 1;   /* Alias */
    config.gradient_clip_value = 0.0F;        /* No value clipping by default */
    config.enable_gradient_management = false;
    config.enable_gradient_scaling = false;
    config.enable_gradient_health_check = true;

    /* Early stopping */
    config.enable_early_stopping = false;
    config.early_stop_patience = 10;
    config.early_stop_min_delta = 1e-4F;

    /* Training parameters */
    config.max_epochs = 1000;
    config.batch_size = NIMCP_DEFAULT_BATCH_SIZE;
    config.convergence_threshold = 1e-6F;
    config.divergence_threshold = 1e6F;

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
    config.rpe_to_da_gain = 0.5F;
    config.biological_lr_modulation = 0.5F;   /* 50% biological, 50% computational */

    /* Second Messenger Cascade defaults */
    config.enable_second_messengers = true;   /* Enable by default - biological fidelity */
    config.num_neurons = 1000;                /* Default neuron count */

    /* Portia Resource Management defaults */
    config.enable_portia_integration = false; /* Disabled by default - opt-in */
    config.min_batch_size_ratio = 0.25F;      /* Minimum 25% of base batch size */
    config.allow_training_pause = true;       /* Allow pause in EMERGENCY mode */
    config.adapt_to_tier_changes = true;      /* Automatically adapt to tier changes */

    return config;
}

/* ============================================================================
 * Configuration Validation
 * ============================================================================ */

nimcp_result_t nimcp_brain_training_validate_config(
    const nimcp_brain_training_config_t* config)
{
    NIMCP_CHECK_THROW(config, NIMCP_ERROR_INVALID_PARAM, "config is NULL");
    NIMCP_CHECK_THROW(config->default_loss_type >= 0 &&
                      config->default_loss_type < NIMCP_LOSS_TYPE_COUNT,
                      NIMCP_ERROR_INVALID_PARAM, "invalid default_loss_type");
    NIMCP_CHECK_THROW(config->default_optimizer >= 0 &&
                      config->default_optimizer < NIMCP_OPTIMIZER_TYPE_COUNT,
                      NIMCP_ERROR_INVALID_PARAM, "invalid default_optimizer");
    NIMCP_CHECK_THROW(config->default_learning_rate > 0.0F &&
                      config->default_learning_rate <= 10.0F,
                      NIMCP_ERROR_INVALID_PARAM, "invalid default_learning_rate");
    NIMCP_CHECK_THROW(config->convergence_threshold > 0.0F,
                      NIMCP_ERROR_INVALID_PARAM, "invalid convergence_threshold");
    NIMCP_CHECK_THROW(config->divergence_threshold > 0.0F,
                      NIMCP_ERROR_INVALID_PARAM, "invalid divergence_threshold");

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
            NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "nimcp_brain_training_create: validation failed");
            return NULL;
        }
        local_config = *config;
    } else {
        local_config = nimcp_brain_training_default_config();
    }

    nimcp_brain_training_ctx_t* ctx = nimcp_calloc(1, sizeof(nimcp_brain_training_ctx_t));
    if (!ctx) {
        LOG_ERROR("Failed to allocate brain-training context");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "nimcp_brain_training_create: ctx is NULL");
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
    ctx->stats.current_grad_scale = 1.0F;

    ctx->last_loss = FLT_MAX;
    ctx->best_loss = FLT_MAX;
    ctx->dropout_seed = (uint32_t)time(NULL);

    /* Initialize plasticity bridge fields (Phase TPB-1) */
    ctx->plasticity_bridge = NULL;
    ctx->biological_modulation_strength = local_config.biological_lr_modulation;
    ctx->prev_loss = FLT_MAX;
    ctx->cumulative_da = 0.0F;
    ctx->cumulative_lr_mod = 0.0F;
    ctx->bio_update_count = 0;

    /* Initialize second messenger cascade system */
    ctx->second_messengers = NULL;
    ctx->second_messengers_enabled = false;

    if (local_config.enable_second_messengers && local_config.num_neurons > 0) {
        second_messenger_config_t sm_config = second_messenger_default_config();
        sm_config.enable_bio_async = true;
        sm_config.enable_security = local_config.enable_security;

        ctx->second_messengers = second_messenger_create(
            local_config.num_neurons,
            &sm_config
        );

        if (ctx->second_messengers) {
            ctx->second_messengers_enabled = true;
            LOG_INFO("Second messenger cascade system created (%u neurons)",
                     local_config.num_neurons);
        } else {
            LOG_WARNING("Failed to create second messenger cascade system");
        }
    }

    /* Initialize Portia Resource Management fields */
    ctx->portia_context = NULL;
    ctx->current_tier = PLATFORM_TIER_FULL;  /* Assume best case initially */
    ctx->resource_aware_training = local_config.enable_portia_integration;
    ctx->tier_batch_size_multiplier = 1.0F;
    ctx->tier_lr_multiplier = 1.0F;
    ctx->training_paused = false;
    ctx->degradation_level = 0;  /* DEGRADATION_LEVEL_NONE */

    LOG_INFO("Brain-training context created (Phase TM-3)");
    return ctx;
}

nimcp_result_t nimcp_brain_training_init(
    nimcp_brain_training_ctx_t* ctx,
    nimcp_sec_integration_t* security_ctx,
    unified_mem_manager_t memory_mgr)
{
    NIMCP_CHECK_THROW(ctx, NIMCP_ERROR_INVALID_PARAM, "ctx is NULL");

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

    /* Register with bio-async router */
    ctx->bio_ctx = NULL;
    ctx->bio_async_enabled = false;
    if (bio_router_is_initialized()) {
        bio_module_info_t bio_info = {
            .module_id = BIO_MODULE_TRAINING_INTEGRATION,
            .module_name = "brain_training_integration",
            .inbox_capacity = 32,
            .user_data = ctx
        };
        ctx->bio_ctx = bio_router_register_module(&bio_info);
        if (ctx->bio_ctx) {
            ctx->bio_async_enabled = true;
            LOG_INFO("Registered brain_training_integration with bio-async router");
        }

        /* Register second messengers with bio-async if enabled */
        if (ctx->second_messengers && ctx->second_messengers_enabled) {
            bio_router_t router = bio_router_get_global();
            if (router) {
                nimcp_result_t sm_res = second_messenger_register_bioasync(
                    ctx->second_messengers,
                    router
                );
                if (sm_res == NIMCP_SUCCESS) {
                    LOG_INFO("Second messengers registered with bio-async router");
                } else {
                    LOG_WARNING("Failed to register second messengers with bio-async: %d",
                                sm_res);
                }
            }
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

    /* Destroy second messenger cascade system */
    if (ctx->second_messengers) {
        second_messenger_destroy(ctx->second_messengers);
        ctx->second_messengers = NULL;
        ctx->second_messengers_enabled = false;
        LOG_INFO("Second messenger cascade system destroyed");
    }

    /* Unregister from bio-async router */
    if (ctx->bio_async_enabled && ctx->bio_ctx) {
        bio_router_unregister_module(ctx->bio_ctx);
        ctx->bio_ctx = NULL;
        ctx->bio_async_enabled = false;
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

    nimcp_free(ctx);
    LOG_INFO("Brain-training context destroyed");
}

/* ============================================================================
 * Security Registration
 * ============================================================================ */

nimcp_result_t nimcp_brain_training_register_security(
    nimcp_brain_training_ctx_t* ctx,
    nimcp_sec_integration_t* security_ctx)
{
    NIMCP_CHECK_THROW(ctx, NIMCP_ERROR_INVALID_PARAM, "ctx is NULL");
    NIMCP_CHECK_THROW(security_ctx, NIMCP_ERROR_INVALID_PARAM, "security_ctx is NULL");

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
    NIMCP_CHECK_THROW(ctx, NIMCP_ERROR_INVALID_PARAM, "ctx is NULL");
    NIMCP_CHECK_THROW(ctx->security_registered, NIMCP_ERROR_INVALID_PARAM, "not registered");
    NIMCP_CHECK_THROW(ctx->security_ctx, NIMCP_ERROR_INVALID_PARAM, "security_ctx is NULL");

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
    NIMCP_CHECK_THROW(ctx, NIMCP_ERROR_INVALID_PARAM, "ctx is NULL");

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
    /* All slots occupied - not an error, just no free slot */
    return -1;
}

static int find_loss_slot_by_id(nimcp_brain_training_ctx_t* ctx, uint32_t loss_id)
{
    for (uint32_t i = 0; i < NIMCP_TRAINING_MAX_LOSS_CONTEXTS; i++) {
        if (ctx->loss_slots[i].active && ctx->loss_slots[i].id == loss_id) {
            return (int)i;
        }
    }
    NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "find_loss_slot_by_id: validation failed");
    return -1;
}

nimcp_result_t nimcp_brain_training_create_loss(
    nimcp_brain_training_ctx_t* ctx,
    const nimcp_loss_config_t* config,
    uint32_t* loss_id)
{
    NIMCP_CHECK_THROW(ctx, NIMCP_ERROR_INVALID_PARAM, "ctx is NULL");
    NIMCP_CHECK_THROW(config, NIMCP_ERROR_INVALID_PARAM, "config is NULL");
    NIMCP_CHECK_THROW(loss_id, NIMCP_ERROR_INVALID_PARAM, "loss_id is NULL");

    int slot = find_free_loss_slot(ctx);
    NIMCP_CHECK_THROW(slot >= 0, NIMCP_ERROR_MEMORY, "No free loss context slots available");

    nimcp_loss_context_t* loss_ctx = nimcp_loss_create(
        config,
        ctx->security_ctx,
        ctx->memory_mgr
    );
    NIMCP_CHECK_THROW(loss_ctx, NIMCP_ERROR_MEMORY, "Failed to create loss context");

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
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "ctx is NULL");

        return NULL;
    }

    int slot = find_loss_slot_by_id(ctx, loss_id);
    if (slot < 0) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "nimcp_brain_training_get_loss: validation failed");
        return NULL;
    }

    return ctx->loss_slots[slot].ctx;
}

nimcp_result_t nimcp_brain_training_destroy_loss(
    nimcp_brain_training_ctx_t* ctx,
    uint32_t loss_id)
{
    NIMCP_CHECK_THROW(ctx, NIMCP_ERROR_INVALID_PARAM, "ctx is NULL");

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
    NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "find_free_optimizer_slot: ctx->optimizer_slots is NULL");
    return -1;
}

static int find_optimizer_slot_by_id(nimcp_brain_training_ctx_t* ctx, uint32_t optimizer_id)
{
    for (uint32_t i = 0; i < NIMCP_TRAINING_MAX_OPTIMIZER_CONTEXTS; i++) {
        if (ctx->optimizer_slots[i].active && ctx->optimizer_slots[i].id == optimizer_id) {
            return (int)i;
        }
    }
    NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "find_optimizer_slot_by_id: validation failed");
    return -1;
}

nimcp_result_t nimcp_brain_training_create_optimizer(
    nimcp_brain_training_ctx_t* ctx,
    const nimcp_optimizer_config_t* config,
    uint32_t* optimizer_id)
{
    NIMCP_CHECK_THROW(ctx, NIMCP_ERROR_INVALID_PARAM, "ctx is NULL");
    NIMCP_CHECK_THROW(config, NIMCP_ERROR_INVALID_PARAM, "config is NULL");
    NIMCP_CHECK_THROW(optimizer_id, NIMCP_ERROR_INVALID_PARAM, "optimizer_id is NULL");

    int slot = find_free_optimizer_slot(ctx);
    NIMCP_CHECK_THROW(slot >= 0, NIMCP_ERROR_MEMORY, "No free optimizer context slots available");

    nimcp_optimizer_context_t* opt_ctx = nimcp_optimizer_create(
        config,
        ctx->security_ctx,
        ctx->memory_mgr
    );
    NIMCP_CHECK_THROW(opt_ctx, NIMCP_ERROR_MEMORY, "Failed to create optimizer context");

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
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "ctx is NULL");

        return NULL;
    }

    int slot = find_optimizer_slot_by_id(ctx, optimizer_id);
    if (slot < 0) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "nimcp_brain_training_get_optimizer: validation failed");
        return NULL;
    }

    return ctx->optimizer_slots[slot].ctx;
}

nimcp_result_t nimcp_brain_training_destroy_optimizer(
    nimcp_brain_training_ctx_t* ctx,
    uint32_t optimizer_id)
{
    NIMCP_CHECK_THROW(ctx, NIMCP_ERROR_INVALID_PARAM, "ctx is NULL");

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
    NIMCP_CHECK_THROW(ctx, NIMCP_ERROR_INVALID_PARAM, "ctx is NULL");
    NIMCP_CHECK_THROW(mode >= NIMCP_TRAINING_MODE_TRAIN && mode <= NIMCP_TRAINING_MODE_INFERENCE,
        NIMCP_ERROR_INVALID_PARAM, "Invalid training mode");

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
    NIMCP_CHECK_THROW(ctx, NIMCP_ERROR_INVALID_PARAM, "ctx is NULL");
    NIMCP_CHECK_THROW(predictions, NIMCP_ERROR_INVALID_PARAM, "predictions is NULL");
    NIMCP_CHECK_THROW(targets, NIMCP_ERROR_INVALID_PARAM, "targets is NULL");
    NIMCP_CHECK_THROW(loss_value, NIMCP_ERROR_INVALID_PARAM, "loss_value is NULL");

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
    NIMCP_CHECK_THROW(ctx, NIMCP_ERROR_INVALID_PARAM, "ctx is NULL");
    NIMCP_CHECK_THROW(params, NIMCP_ERROR_INVALID_PARAM, "params is NULL");
    NIMCP_CHECK_THROW(gradients, NIMCP_ERROR_INVALID_PARAM, "gradients is NULL");
    NIMCP_CHECK_THROW(count > 0, NIMCP_ERROR_INVALID_PARAM, "count is 0");

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
    NIMCP_CHECK_THROW(ctx, NIMCP_ERROR_INVALID_PARAM, "ctx is NULL");
    NIMCP_CHECK_THROW(params, NIMCP_ERROR_INVALID_PARAM, "params is NULL");
    NIMCP_CHECK_THROW(predictions, NIMCP_ERROR_INVALID_PARAM, "predictions is NULL");
    NIMCP_CHECK_THROW(targets, NIMCP_ERROR_INVALID_PARAM, "targets is NULL");
    NIMCP_CHECK_THROW(loss_value, NIMCP_ERROR_INVALID_PARAM, "loss_value is NULL");
    NIMCP_CHECK_THROW(batch_size > 0, NIMCP_ERROR_INVALID_PARAM, "batch_size is 0");
    NIMCP_CHECK_THROW(output_size > 0, NIMCP_ERROR_INVALID_PARAM, "output_size is 0");
    NIMCP_CHECK_THROW(param_count > 0, NIMCP_ERROR_INVALID_PARAM, "param_count is 0");

    /* Phase TPB-1: If plasticity bridge is connected and biological modulation > 0,
     * automatically route through biological training step */
    if (ctx->plasticity_bridge && ctx->biological_modulation_strength > 0.0F) {
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
        gradients = (float*)nimcp_malloc(gradient_size);
        NIMCP_CHECK_THROW(gradients, NIMCP_ERROR_MEMORY, "Failed to allocate gradient buffer");
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
            nimcp_free(gradients);
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
        nimcp_free(gradients);
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
    NIMCP_CHECK_THROW(ctx, NIMCP_ERROR_INVALID_PARAM, "ctx is NULL");
    NIMCP_CHECK_THROW(event, NIMCP_ERROR_INVALID_PARAM, "event is NULL");

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
    NIMCP_CHECK_THROW(ctx, NIMCP_ERROR_INVALID_PARAM, "ctx is NULL");

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
    NIMCP_CHECK_THROW(ctx, NIMCP_ERROR_INVALID_PARAM, "ctx is NULL");
    NIMCP_CHECK_THROW(stats, NIMCP_ERROR_INVALID_PARAM, "stats is NULL");

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

void nimcp_brain_training_update_stats(
    nimcp_brain_training_ctx_t* ctx,
    uint32_t samples_processed,
    float loss_value)
{
    if (!ctx) {
        return;
    }

    ctx->stats.total_samples += samples_processed;
    ctx->stats.total_batches += 1;
    ctx->stats.total_loss += (double)loss_value;

    /* Update min/max loss */
    if (loss_value < ctx->stats.min_loss) {
        ctx->stats.min_loss = loss_value;
    }
    if (loss_value > ctx->stats.max_loss) {
        ctx->stats.max_loss = loss_value;
    }
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
    NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "find_free_scheduler_slot: ctx->scheduler_slots is NULL");
    return -1;
}

static int find_scheduler_slot_by_id(nimcp_brain_training_ctx_t* ctx, uint32_t scheduler_id)
{
    for (uint32_t i = 0; i < NIMCP_TRAINING_MAX_SCHEDULER_CONTEXTS; i++) {
        if (ctx->scheduler_slots[i].active && ctx->scheduler_slots[i].id == scheduler_id) {
            return (int)i;
        }
    }
    NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "find_scheduler_slot_by_id: validation failed");
    return -1;
}

nimcp_result_t nimcp_brain_training_create_scheduler(
    nimcp_brain_training_ctx_t* ctx,
    const nimcp_lr_scheduler_config_t* config,
    uint32_t* scheduler_id)
{
    NIMCP_CHECK_THROW(ctx, NIMCP_ERROR_INVALID_PARAM, "ctx is NULL");
    NIMCP_CHECK_THROW(config, NIMCP_ERROR_INVALID_PARAM, "config is NULL");
    NIMCP_CHECK_THROW(scheduler_id, NIMCP_ERROR_INVALID_PARAM, "scheduler_id is NULL");

    int slot = find_free_scheduler_slot(ctx);
    NIMCP_CHECK_THROW(slot >= 0, NIMCP_ERROR_MEMORY, "No free scheduler slots available");

    nimcp_lr_scheduler_ctx_t* sched_ctx = nimcp_lr_scheduler_create(config);
    NIMCP_CHECK_THROW(sched_ctx, NIMCP_ERROR_MEMORY, "Failed to create scheduler context");

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
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "ctx is NULL");

        return NULL;
    }

    int slot = find_scheduler_slot_by_id(ctx, scheduler_id);
    if (slot < 0) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "nimcp_brain_training_get_scheduler: validation failed");
        return NULL;
    }

    return ctx->scheduler_slots[slot].ctx;
}

nimcp_result_t nimcp_brain_training_destroy_scheduler(
    nimcp_brain_training_ctx_t* ctx,
    uint32_t scheduler_id)
{
    NIMCP_CHECK_THROW(ctx, NIMCP_ERROR_INVALID_PARAM, "ctx is NULL");

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
        return 0.0F;
    }

    nimcp_lr_scheduler_ctx_t* sched = nimcp_brain_training_get_scheduler(ctx, scheduler_id);
    if (!sched) {
        return 0.0F;
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
        return 0.0F;
    }

    nimcp_lr_scheduler_ctx_t* sched = nimcp_brain_training_get_scheduler(ctx, scheduler_id);
    if (!sched) {
        return 0.0F;
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
    NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "find_free_gradmgr_slot: ctx->gradmgr_slots is NULL");
    return -1;
}

static int find_gradmgr_slot_by_id(nimcp_brain_training_ctx_t* ctx, uint32_t gradmgr_id)
{
    for (uint32_t i = 0; i < NIMCP_TRAINING_MAX_GRADMGR_CONTEXTS; i++) {
        if (ctx->gradmgr_slots[i].active && ctx->gradmgr_slots[i].id == gradmgr_id) {
            return (int)i;
        }
    }
    NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "find_gradmgr_slot_by_id: validation failed");
    return -1;
}

nimcp_result_t nimcp_brain_training_create_gradient_manager(
    nimcp_brain_training_ctx_t* ctx,
    const nimcp_gradient_manager_config_t* config,
    uint32_t* gradmgr_id)
{
    NIMCP_CHECK_THROW(ctx, NIMCP_ERROR_INVALID_PARAM, "ctx is NULL");
    NIMCP_CHECK_THROW(config, NIMCP_ERROR_INVALID_PARAM, "config is NULL");
    NIMCP_CHECK_THROW(gradmgr_id, NIMCP_ERROR_INVALID_PARAM, "gradmgr_id is NULL");

    int slot = find_free_gradmgr_slot(ctx);
    NIMCP_CHECK_THROW(slot >= 0, NIMCP_ERROR_MEMORY, "No free gradient manager slots available");

    nimcp_gradient_manager_ctx_t* gm_ctx = nimcp_gradient_manager_create(config);
    NIMCP_CHECK_THROW(gm_ctx, NIMCP_ERROR_MEMORY, "Failed to create gradient manager context");

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
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "ctx is NULL");

        return NULL;
    }

    int slot = find_gradmgr_slot_by_id(ctx, gradmgr_id);
    if (slot < 0) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "nimcp_brain_training_get_gradient_manager: validation failed");
        return NULL;
    }

    return ctx->gradmgr_slots[slot].ctx;
}

nimcp_result_t nimcp_brain_training_destroy_gradient_manager(
    nimcp_brain_training_ctx_t* ctx,
    uint32_t gradmgr_id)
{
    NIMCP_CHECK_THROW(ctx, NIMCP_ERROR_INVALID_PARAM, "ctx is NULL");

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
    NIMCP_CHECK_THROW(ctx && gradients && count > 0, NIMCP_ERROR_INVALID_PARAM,
                      "nimcp_brain_training_accumulate_gradients: invalid arguments");

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
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "nimcp_brain_training_gradients_ready: ctx is NULL");
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
    NIMCP_CHECK_THROW(ctx && output && count > 0, NIMCP_ERROR_INVALID_PARAM,
                      "nimcp_brain_training_get_and_reset_gradients: invalid arguments");

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
        return 0.0F;
    }

    float reg_loss = 0.0F;

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
            float alpha = 0.5F;
            reg_loss = nimcp_elastic_net_loss(weights, count, lambda, alpha);
            /* Apply both L1 and L2 gradients with appropriate weights */
            nimcp_l1_gradient(weights, gradients, count, lambda * alpha);
            nimcp_l2_gradient(weights, gradients, count, lambda * (1.0F - alpha));
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
        return 1.0F;
    }

    float original_norm = nimcp_gradient_norm(gradients, count);
    float clip_ratio = 1.0F;

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
            return 1.0F;
    }

    /* For value clipping, compute ratio from norms */
    if (mode == NIMCP_CLIP_BY_VALUE) {
        float clipped_norm = nimcp_gradient_norm(gradients, count);
        if (clipped_norm > 0.0F) {
            clip_ratio = original_norm / clipped_norm;
        }
    }

    if (clip_ratio > 1.0F) {
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
    uint8_t* dropout_mask = mask ? (uint8_t*)nimcp_malloc(count) : NULL;
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
        nimcp_free(dropout_mask);
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
    NIMCP_CHECK_THROW(ctx && params && predictions && targets && loss_value,
                      NIMCP_ERROR_INVALID_PARAM,
                      "nimcp_brain_training_step_full: NULL argument");
    NIMCP_CHECK_THROW(batch_size > 0 && output_size > 0 && param_count > 0,
                      NIMCP_ERROR_INVALID_PARAM,
                      "nimcp_brain_training_step_full: size is 0");

    /* Allocate gradient buffer */
    size_t gradient_count = batch_size * output_size;
    float* gradients = (float*)nimcp_malloc(gradient_count * sizeof(float));
    NIMCP_CHECK_THROW(gradients, NIMCP_ERROR_MEMORY,
                      "nimcp_brain_training_step_full: failed to allocate gradients");
    memset(gradients, 0, gradient_count * sizeof(float));

    /* Step 1: Compute loss and gradients */
    nimcp_result_t res = nimcp_brain_training_compute_loss(
        ctx, loss_id, predictions, targets,
        batch_size, output_size, loss_value, gradients
    );

    if (res != NIMCP_SUCCESS) {
        nimcp_free(gradients);
        return res;
    }

    /* Phase TPB-1: Report loss to plasticity bridge for RPE computation */
    if (ctx->plasticity_bridge && ctx->biological_modulation_strength > 0.0F) {
        float rpe = 0.0F;
        if (tpb_report_loss(ctx->plasticity_bridge, *loss_value, &rpe) == NIMCP_SUCCESS) {
            ctx->stats.rpe_computations++;

            /* Get current dopamine level for tracking */
            float da_level = 0.5F, ach_level, ht5_level, ne_level;
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
            nimcp_free(gradients);
            return NIMCP_TRAINING_ERROR_GRAD_NAN;  /* Skip this step */
        }
        if (health == NIMCP_GRAD_HAS_INF) {
            ctx->stats.grad_inf_count++;
            /* Sanitize infinity values */
            nimcp_gradient_sanitize(gradients, gradient_count, 0.0F);
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
                nimcp_free(gradients);
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
    float original_lr = 0.0F;
    nimcp_optimizer_context_t* opt = NULL;
    if (ctx->plasticity_bridge && ctx->biological_modulation_strength > 0.0F) {
        opt = nimcp_brain_training_get_optimizer(ctx, optimizer_id);
        if (opt) {
            original_lr = nimcp_optimizer_get_lr(opt);
            float bio_lr = original_lr;
            if (tpb_get_modulated_lr(ctx->plasticity_bridge, 0, original_lr, &bio_lr) == NIMCP_SUCCESS) {
                float blend = ctx->biological_modulation_strength;
                float effective_lr = (1.0F - blend) * original_lr + blend * bio_lr;
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
    if (opt && original_lr > 0.0F) {
        nimcp_optimizer_set_lr(opt, original_lr);
    }

    nimcp_free(gradients);

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
    NIMCP_CHECK_THROW(ctx, NIMCP_ERROR_INVALID_PARAM,
                      "nimcp_brain_training_connect_plasticity_bridge: ctx is NULL");

    ctx->plasticity_bridge = bridge;

    if (bridge) {
        LOG_INFO("Plasticity bridge connected to training integration");

        /* If callbacks exist, wire them to plasticity bridge */
        if (ctx->callbacks) {
            tpb_connect_callbacks(bridge, ctx->callbacks);
            tpb_register_plasticity_callbacks(bridge);
            LOG_INFO("Auto-wired callbacks to plasticity bridge");
        }
    } else {
        LOG_INFO("Plasticity bridge disconnected from training integration");
    }

    return NIMCP_SUCCESS;
}

tpb_context_t* nimcp_brain_training_get_plasticity_bridge(
    const nimcp_brain_training_ctx_t* ctx)
{
    if (!ctx) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "ctx is NULL");

        return NULL;
    }
    return ctx->plasticity_bridge;
}

nimcp_result_t nimcp_brain_training_set_biological_modulation(
    nimcp_brain_training_ctx_t* ctx,
    float strength)
{
    NIMCP_CHECK_THROW(ctx, NIMCP_ERROR_INVALID_PARAM,
                      "nimcp_brain_training_set_biological_modulation: ctx is NULL");

    if (strength < 0.0F) strength = 0.0F;
    if (strength > 1.0F) strength = 1.0F;

    ctx->biological_modulation_strength = strength;
    LOG_DEBUG("Biological modulation strength set to %.2f", strength);

    return NIMCP_SUCCESS;
}

float nimcp_brain_training_get_biological_modulation(
    const nimcp_brain_training_ctx_t* ctx)
{
    if (!ctx) {
        return 0.0F;
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
    NIMCP_CHECK_THROW(ctx && params && predictions && targets && loss_value,
                      NIMCP_ERROR_INVALID_PARAM,
                      "nimcp_brain_training_step_biological: NULL argument");
    NIMCP_CHECK_THROW(batch_size > 0 && output_size > 0 && param_count > 0,
                      NIMCP_ERROR_INVALID_PARAM,
                      "nimcp_brain_training_step_biological: size is 0");

    /* If no plasticity bridge or biological modulation is zero, fallback to regular step */
    if (!ctx->plasticity_bridge || ctx->biological_modulation_strength <= 0.0F) {
        return nimcp_brain_training_step(
            ctx, loss_id, optimizer_id, params,
            predictions, targets, batch_size, output_size,
            param_count, loss_value
        );
    }

    /* Allocate gradient buffer */
    size_t gradient_count = batch_size * output_size;
    float* gradients = (float*)nimcp_malloc(gradient_count * sizeof(float));
    NIMCP_CHECK_THROW(gradients, NIMCP_ERROR_MEMORY,
                      "nimcp_brain_training_step_biological: failed to allocate gradients");
    memset(gradients, 0, gradient_count * sizeof(float));

    /* Step 1: Compute loss and gradients */
    nimcp_result_t res = nimcp_brain_training_compute_loss(
        ctx, loss_id, predictions, targets,
        batch_size, output_size, loss_value, gradients
    );

    if (res != NIMCP_SUCCESS) {
        nimcp_free(gradients);
        return res;
    }

    /* Step 2: Report loss to plasticity bridge and get RPE */
    float rpe = 0.0F;
    res = tpb_report_loss(ctx->plasticity_bridge, *loss_value, &rpe);
    if (res == NIMCP_SUCCESS) {
        ctx->stats.rpe_computations++;

        /* Get current dopamine level from neuromodulator system */
        float da_level = 0.5F, ach_level, ht5_level, ne_level;
        if (tpb_get_neuromod_levels(ctx->plasticity_bridge, &da_level, &ach_level,
                                     &ht5_level, &ne_level) == NIMCP_SUCCESS) {
            ctx->cumulative_da += da_level;

            /* Activate second messenger cascades based on reward signal
             * Positive RPE/dopamine activates Gs-coupled D1 receptors -> cAMP -> PKA
             * This enhances learning for reward-related updates */
            if (ctx->second_messengers_enabled && ctx->second_messengers && rpe > 0.0F) {
                uint64_t timestamp_ms = nimcp_time_get_us() / 1000;
                /* Activate Gs pathway for first neuron (representative)
                 * occupancy scales with dopamine level [0, 1] */
                float occupancy = da_level;
                if (occupancy > 1.0F) occupancy = 1.0F;

                second_messenger_activate_gs(
                    ctx->second_messengers,
                    0,  /* neuron_id */
                    occupancy,
                    timestamp_ms
                );

                LOG_DEBUG("Activated Gs cascade for reward (RPE=%.3f, DA=%.3f)",
                          rpe, da_level);
            }
        }
    }

    /* Step 3: Get biologically-modulated learning rate */
    nimcp_optimizer_context_t* opt = nimcp_brain_training_get_optimizer(ctx, optimizer_id);
    float base_lr = opt ? nimcp_optimizer_get_lr(opt) : ctx->config.default_learning_rate;

    float bio_lr = base_lr;
    tpb_get_modulated_lr(ctx->plasticity_bridge, region_id, base_lr, &bio_lr);
    float blend = ctx->biological_modulation_strength;
    float effective_lr = (1.0F - blend) * base_lr + blend * bio_lr;

    /* Apply second messenger cascade modulation if enabled
     * PKA/CaMKII activity boosts learning rates for enhanced plasticity
     * Use first neuron as representative for batch (neuron_id = 0) */
    if (ctx->second_messengers_enabled && ctx->second_messengers) {
        float cascade_mod = nimcp_brain_training_get_cascade_modulation(ctx, 0);
        effective_lr *= cascade_mod;
        LOG_DEBUG("Cascade modulation applied: %.3f (effective LR: %.6f)",
                  cascade_mod, effective_lr);

        /* Update second messenger dynamics for this timestep
         * This advances the cascade state based on current activity */
        uint64_t timestamp_ms = nimcp_time_get_us() / 1000;
        second_messenger_update(ctx->second_messengers, 1.0F, timestamp_ms);
    }

    ctx->cumulative_lr_mod += (effective_lr / base_lr);

    /* Temporarily set modulated learning rate */
    if (opt) {
        nimcp_optimizer_set_lr(opt, effective_lr);
    }

    /* Step 4: Route weight updates through plasticity bridge */
    size_t update_count = (param_count < gradient_count) ? param_count : gradient_count;

    /* Get CaMKII activity for STDP enhancement if cascades enabled */
    second_messenger_state_t sm_state = {0};
    bool have_cascade_state = false;
    if (ctx->second_messengers_enabled && ctx->second_messengers) {
        if (second_messenger_get_state(ctx->second_messengers, 0, &sm_state) == NIMCP_SUCCESS) {
            have_cascade_state = true;
        }
    }

    for (size_t i = 0; i < update_count; i++) {
        /* For biological routing, we compute weight delta through the bridge */
        float pre_activity = fabsf(gradients[i]);  /* Use gradient magnitude as proxy */
        float post_activity = fabsf(params[i]);     /* Use weight magnitude as proxy */
        float spike_delta = (gradients[i] > 0) ? 0.01F : -0.01F;  /* Sign indicates timing */

        float weight_delta = 0.0F;
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
            final_delta = (1.0F - blend) * grad_delta + blend * weight_delta;

            /* Enhance STDP-like weight changes with CaMKII activity
             * CaMKII is activated by coincident pre/post activity (calcium influx)
             * and enhances LTP magnitude */
            if (have_cascade_state && sm_state.calcium.camkii_activity > 0.5F) {
                float camkii_boost = 1.0F + (sm_state.calcium.camkii_activity - 0.5F);
                final_delta *= camkii_boost;
            }
        } else {
            final_delta = grad_delta;  /* Fallback to pure gradient if routing fails */
        }
        params[i] += final_delta;
    }

    ctx->stats.biological_updates += update_count;
    ctx->bio_update_count++;

    /* Check for CREB phosphorylation signaling consolidation
     * High CREB phosphorylation indicates readiness for long-term plasticity
     * This could trigger additional consolidation mechanisms */
    if (have_cascade_state && sm_state.gene_expr.creb_phosphorylation > 0.7F) {
        LOG_DEBUG("CREB phosphorylation high (%.3f) - consolidation signaled",
                  sm_state.gene_expr.creb_phosphorylation);

        /* Emit training event for consolidation marker */
        nimcp_training_event_t event = {0};
        event.type = NIMCP_TRAINING_EVENT_WEIGHTS_UPDATED;
        event.timestamp = nimcp_time_get_us() * 1000;
        event.epoch = ctx->current_epoch;
        event.batch = ctx->current_batch;
        event.loss_value = *loss_value;
        event.learning_rate = effective_lr;
        nimcp_brain_training_emit_event(ctx, &event);
    }

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

    nimcp_free(gradients);

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

//=============================================================================
// Training Callbacks Integration (Phase TCB-1)
//=============================================================================

nimcp_result_t nimcp_brain_training_create_callbacks(nimcp_brain_training_ctx_t* ctx)
{
    NIMCP_CHECK_THROW(ctx, NIMCP_ERROR_INVALID_PARAM,
                      "nimcp_brain_training_create_callbacks: ctx is NULL");

    if (ctx->callbacks) {
        LOG_WARNING("Callbacks already created");
        return NIMCP_SUCCESS;
    }

    /* Create callbacks context */
    ctx->callbacks = tcb_create(NULL);
    NIMCP_CHECK_THROW(ctx->callbacks, NIMCP_ERROR_MEMORY,
                      "nimcp_brain_training_create_callbacks: failed to create callbacks");

    /* If plasticity bridge is connected, wire it up */
    if (ctx->plasticity_bridge) {
        tpb_connect_callbacks(ctx->plasticity_bridge, ctx->callbacks);
        tpb_register_plasticity_callbacks(ctx->plasticity_bridge);
    }

    LOG_INFO("Training callbacks created");
    return NIMCP_SUCCESS;
}

tcb_context_t* nimcp_brain_training_get_callbacks(const nimcp_brain_training_ctx_t* ctx)
{
    if (!ctx) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "ctx is NULL");

        return NULL;
    }
    return ctx->callbacks;
}

/* ============================================================================
 * Second Messenger Cascade Integration
 * ============================================================================ */

nimcp_result_t nimcp_brain_training_connect_second_messengers(
    nimcp_brain_training_ctx_t* ctx,
    void* second_messengers)
{
    NIMCP_CHECK_THROW(ctx, NIMCP_ERROR_INVALID_PARAM,
                      "nimcp_brain_training_connect_second_messengers: ctx is NULL");

    /* Guard: NULL is allowed for disconnection */
    ctx->second_messengers = (second_messenger_system_t*)second_messengers;

    if (second_messengers) {
        ctx->second_messengers_enabled = true;
        LOG_INFO("Second messenger cascade system connected to training");
    } else {
        ctx->second_messengers_enabled = false;
        LOG_INFO("Second messenger cascade system disconnected from training");
    }

    return NIMCP_SUCCESS;
}

float nimcp_brain_training_get_cascade_modulation(
    const nimcp_brain_training_ctx_t* ctx,
    uint32_t neuron_id)
{
    /* Guard: Return baseline modulation if context invalid */
    if (!ctx) {
        return 1.0F;
    }

    /* Guard: Return baseline if second messengers not enabled */
    if (!ctx->second_messengers_enabled || !ctx->second_messengers) {
        return 1.0F;
    }

    /* Query plasticity modulation from second messenger system
     * Returns factor in range [0.5, 2.0] where 1.0 = baseline */
    float modulation = second_messenger_get_plasticity_modulation(
        ctx->second_messengers,
        neuron_id
    );

    /* Guard: Clamp to safe range in case of error */
    if (modulation < 0.5F) modulation = 0.5F;
    if (modulation > 2.0F) modulation = 2.0F;

    return modulation;
}

/* ============================================================================
 * Portia Resource Management Integration
 * ============================================================================ */

/**
 * @brief Helper: Calculate tier multipliers based on platform tier
 *
 * WHAT: Map platform tier to batch size and LR multipliers
 * WHY:  Reduce compute when resources constrained
 * HOW:  Use predefined scaling factors per tier
 */
static void calculate_tier_multipliers(
    platform_tier_t tier,
    float* batch_multiplier,
    float* lr_multiplier)
{
    switch (tier) {
        case PLATFORM_TIER_FULL:
            *batch_multiplier = 1.0F;   /* Full batch */
            *lr_multiplier = 1.0F;      /* Normal LR */
            break;

        case PLATFORM_TIER_MEDIUM:
            *batch_multiplier = 0.75F;  /* 75% batch */
            *lr_multiplier = 0.9F;      /* 90% LR */
            break;

        case PLATFORM_TIER_CONSTRAINED:
            *batch_multiplier = 0.5F;   /* 50% batch */
            *lr_multiplier = 0.75F;     /* 75% LR */
            break;

        case PLATFORM_TIER_MINIMAL:
            *batch_multiplier = 0.25F;  /* 25% batch */
            *lr_multiplier = 0.5F;      /* 50% LR */
            break;

        default:
            *batch_multiplier = 1.0F;
            *lr_multiplier = 1.0F;
            break;
    }
}

nimcp_result_t nimcp_brain_training_connect_portia(
    nimcp_brain_training_ctx_t* ctx,
    void* portia_ctx)
{
    NIMCP_CHECK_THROW(ctx, NIMCP_ERROR_INVALID_PARAM,
                      "nimcp_brain_training_connect_portia: ctx is NULL");

    /* Allow NULL to disconnect */
    ctx->portia_context = portia_ctx;

    if (portia_ctx) {
        /* Query current tier from Portia */
        ctx->current_tier = portia_get_current_tier();

        /* Calculate initial multipliers */
        calculate_tier_multipliers(
            ctx->current_tier,
            &ctx->tier_batch_size_multiplier,
            &ctx->tier_lr_multiplier
        );

        ctx->resource_aware_training = true;
        LOG_INFO("Portia connected to training system (tier=%s)",
                 platform_tier_get_name(ctx->current_tier));
    } else {
        /* Reset to defaults when disconnecting */
        ctx->resource_aware_training = false;
        ctx->tier_batch_size_multiplier = 1.0F;
        ctx->tier_lr_multiplier = 1.0F;
        LOG_INFO("Portia disconnected from training system");
    }

    return NIMCP_SUCCESS;
}

nimcp_result_t nimcp_brain_training_on_tier_change(
    nimcp_brain_training_ctx_t* ctx,
    platform_tier_t new_tier)
{
    NIMCP_CHECK_THROW(ctx, NIMCP_ERROR_INVALID_PARAM,
                      "nimcp_brain_training_on_tier_change: ctx is NULL");

    /* Guard: Check if resource-aware training enabled */
    if (!ctx->resource_aware_training) {
        return NIMCP_SUCCESS;  /* Silently ignore if disabled */
    }

    platform_tier_t old_tier = ctx->current_tier;
    ctx->current_tier = new_tier;

    /* Calculate new multipliers */
    calculate_tier_multipliers(
        new_tier,
        &ctx->tier_batch_size_multiplier,
        &ctx->tier_lr_multiplier
    );

    /* Log tier change */
    LOG_INFO("Training tier change: %s -> %s (batch=%.0f%%, lr=%.0f%%)",
             platform_tier_get_name(old_tier),
             platform_tier_get_name(new_tier),
             ctx->tier_batch_size_multiplier * 100.0F,
             ctx->tier_lr_multiplier * 100.0F);

    /* Handle EMERGENCY tier: pause training if allowed */
    if (new_tier == PLATFORM_TIER_MINIMAL && !ctx->training_paused) {
        ctx->training_paused = true;
        LOG_WARNING("Training paused due to EMERGENCY tier");

        /* TODO: Save checkpoint here in production */
    }

    /* Resume training if returning from MINIMAL tier */
    if (old_tier == PLATFORM_TIER_MINIMAL && new_tier != PLATFORM_TIER_MINIMAL) {
        if (ctx->training_paused) {
            ctx->training_paused = false;
            LOG_INFO("Training resumed after tier upgrade");
        }
    }

    return NIMCP_SUCCESS;
}

nimcp_result_t nimcp_brain_training_on_degradation_event(
    nimcp_brain_training_ctx_t* ctx,
    uint32_t degradation_level)
{
    NIMCP_CHECK_THROW(ctx, NIMCP_ERROR_INVALID_PARAM,
                      "nimcp_brain_training_on_degradation_change: ctx is NULL");

    /* Guard: Check if resource-aware training enabled */
    if (!ctx->resource_aware_training) {
        return NIMCP_SUCCESS;
    }

    uint32_t old_level = ctx->degradation_level;
    ctx->degradation_level = degradation_level;

    LOG_INFO("Training degradation level change: %u -> %u",
             old_level, degradation_level);

    /* Handle critical degradation levels */
    if (degradation_level >= 4) {  /* DEGRADATION_LEVEL_CRITICAL */
        if (!ctx->training_paused) {
            ctx->training_paused = true;
            LOG_WARNING("Training paused due to CRITICAL degradation");
        }
    } else if (degradation_level >= 3) {  /* DEGRADATION_LEVEL_SEVERE */
        /* Further reduce batch size beyond tier multiplier */
        ctx->tier_batch_size_multiplier *= 0.5F;
        LOG_INFO("Batch size reduced further due to SEVERE degradation");
    }

    /* Resume if degradation improves */
    if (old_level >= 4 && degradation_level < 4 && ctx->training_paused) {
        ctx->training_paused = false;
        /* Restore tier-based multiplier */
        calculate_tier_multipliers(
            ctx->current_tier,
            &ctx->tier_batch_size_multiplier,
            &ctx->tier_lr_multiplier
        );
        LOG_INFO("Training resumed after degradation improvement");
    }

    return NIMCP_SUCCESS;
}

bool nimcp_brain_training_is_paused(
    const nimcp_brain_training_ctx_t* ctx)
{
    if (!ctx) {
        return false;
    }

    return ctx->training_paused;
}

nimcp_result_t nimcp_brain_training_resume(
    nimcp_brain_training_ctx_t* ctx)
{
    NIMCP_CHECK_THROW(ctx, NIMCP_ERROR_INVALID_PARAM,
                      "nimcp_brain_training_resume: ctx is NULL");

    if (ctx->training_paused) {
        ctx->training_paused = false;
        LOG_INFO("Training manually resumed");
    }

    return NIMCP_SUCCESS;
}

size_t nimcp_brain_training_get_adjusted_batch_size(
    const nimcp_brain_training_ctx_t* ctx,
    size_t base_batch_size)
{
    /* Guard: Return base if context invalid */
    if (!ctx || !ctx->resource_aware_training) {
        return base_batch_size;
    }

    /* Guard: Return 0 if training paused */
    if (ctx->training_paused) {
        return 0;
    }

    /* Apply tier multiplier */
    size_t adjusted = (size_t)(base_batch_size * ctx->tier_batch_size_multiplier);

    /* Ensure minimum batch size (at least 1) */
    if (adjusted == 0 && base_batch_size > 0) {
        adjusted = 1;
    }

    return adjusted;
}

float nimcp_brain_training_get_adjusted_lr(
    const nimcp_brain_training_ctx_t* ctx,
    float base_lr)
{
    /* Guard: Return base if context invalid */
    if (!ctx || !ctx->resource_aware_training) {
        return base_lr;
    }

    /* Guard: Return 0 if training paused */
    if (ctx->training_paused) {
        return 0.0F;
    }

    /* Apply tier multiplier */
    float adjusted = base_lr * ctx->tier_lr_multiplier;

    return adjusted;
}

nimcp_result_t nimcp_brain_training_request_resources(
    nimcp_brain_training_ctx_t* ctx,
    size_t batch_size,
    size_t param_count)
{
    NIMCP_CHECK_THROW(ctx, NIMCP_ERROR_INVALID_PARAM,
                      "nimcp_brain_training_request_resources: ctx is NULL");

    /* Guard: Check if bio-async enabled */
    if (!ctx->bio_async_enabled) {
        return NIMCP_SUCCESS;  /* Silently skip if bio-async disabled */
    }

    /* Guard: Check if Portia connected */
    if (!ctx->portia_context || !ctx->resource_aware_training) {
        return NIMCP_SUCCESS;  /* Silently skip if Portia not connected */
    }

    /* Guard: Check if bio context available */
    if (!ctx->bio_ctx) {
        return NIMCP_SUCCESS;  /* Silently skip if no bio context */
    }

    /* Create resource request message */
    struct {
        bio_message_header_t header;
        uint64_t payload[4];
    } msg = {
        .header = {
            .type = BIO_MSG_TRAINING_RESOURCE_REQUEST,
            .sequence_id = 0,  /* Will be set by router */
            .source_module = BIO_MODULE_TRAINING,
            .target_module = BIO_MODULE_PORTIA,
            .timestamp_us = 0,  /* Will be set by router */
            .channel = BIO_CHANNEL_SEROTONIN,
            .payload_size = sizeof(uint64_t) * 4,
            .flags = 0
        },
        .payload = {
            (uint64_t)batch_size,
            (uint64_t)param_count,
            (uint64_t)ctx->current_tier,
            0  /* Reserved */
        }
    };

    /* Send via bio-router */
    nimcp_error_t result = bio_router_send(
        ctx->bio_ctx,
        &msg,
        sizeof(msg),
        100  /* 100ms timeout */
    );

    if (result != NIMCP_SUCCESS) {
        LOG_WARNING("Failed to send training resource request to Portia");
        return (nimcp_result_t)result;
    }

    return NIMCP_SUCCESS;
}
