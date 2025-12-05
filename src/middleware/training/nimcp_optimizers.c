/**
 * @file nimcp_optimizers.c
 * @brief Implementation of Optimizers Module for NIMCP Training Pipeline
 *
 * Phase TM-2: Training Pipeline Infrastructure
 * Phase BIO-1: Bio-async integration
 * Phase LOG-1: Logging integration
 * Phase SEC-1: Security validation
 */

#include "middleware/training/nimcp_optimizers.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "security/nimcp_blood_brain_barrier.h"
#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* Module name for security registration and logging */
#define OPTIMIZER_MODULE_NAME "nimcp_optimizer"
#define LOG_MODULE "optimizers"

/* ============================================================================
 * Internal Structures
 * ============================================================================ */

/**
 * @brief Internal optimizer state for momentum-based optimizers
 */
typedef struct optimizer_momentum_state {
    float* velocity;              /**< Velocity buffer */
    size_t count;                 /**< Number of parameters */
} optimizer_momentum_state_t;

/**
 * @brief Internal optimizer state for Adam-style optimizers
 */
typedef struct optimizer_adam_state {
    float* m;                     /**< First moment estimate */
    float* v;                     /**< Second moment estimate */
    float* v_max;                 /**< Max second moment (AMSGrad) */
    size_t count;                 /**< Number of parameters */
    uint64_t t;                   /**< Timestep */
} optimizer_adam_state_t;

/**
 * @brief Internal optimizer state for RMSprop
 */
typedef struct optimizer_rmsprop_state {
    float* square_avg;            /**< Running average of squared gradients */
    float* momentum_buffer;       /**< Momentum buffer */
    float* grad_avg;              /**< Running average of gradients (centered) */
    size_t count;                 /**< Number of parameters */
} optimizer_rmsprop_state_t;

/**
 * @brief Internal optimizer state for AdaGrad
 */
typedef struct optimizer_adagrad_state {
    float* sum;                   /**< Sum of squared gradients */
    size_t count;                 /**< Number of parameters */
    uint64_t step;                /**< Step count for LR decay */
} optimizer_adagrad_state_t;

/**
 * @brief Internal optimizer context
 */
struct nimcp_optimizer_context {
    nimcp_optimizer_config_t config;
    nimcp_sec_integration_t* security_ctx;
    unified_mem_manager_t memory_mgr;
    nimcp_optimizer_stats_t stats;

    bool initialized;
    bool security_registered;
    uint32_t module_id;

    /* Current learning rate (may differ from config due to scheduling) */
    float current_lr;
    uint64_t step_count;

    /* Optimizer-specific state */
    union {
        optimizer_momentum_state_t momentum;
        optimizer_adam_state_t adam;
        optimizer_rmsprop_state_t rmsprop;
        optimizer_adagrad_state_t adagrad;
        void* custom;
    } state;

    size_t state_size;

    /* Bio-async integration (Phase BIO-1) */
    bio_module_context_t bio_ctx;      /**< Bio-async module context */
    bool bio_async_enabled;            /**< Bio-async availability flag */
};

/* ============================================================================
 * Helper Functions
 * ============================================================================ */

static float* alloc_buffer(nimcp_optimizer_context_t* ctx, size_t count) {
    (void)ctx;  /* Reserved for future unified memory integration */
    size_t bytes = count * sizeof(float);
    return (float*)nimcp_malloc(bytes);
}

static void free_buffer(nimcp_optimizer_context_t* ctx, float* buffer) {
    (void)ctx;
    if (buffer) {
        nimcp_free(buffer);
    }
}

static uint64_t get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static void update_gradient_stats(nimcp_optimizer_context_t* ctx, float norm) {
    ctx->stats.total_gradient_norm += norm;
    if (norm < ctx->stats.min_gradient_norm) {
        ctx->stats.min_gradient_norm = norm;
    }
    if (norm > ctx->stats.max_gradient_norm) {
        ctx->stats.max_gradient_norm = norm;
    }
    ctx->stats.avg_gradient_norm = ctx->stats.total_gradient_norm / ctx->stats.step_count;
}

/* ============================================================================
 * Bio-Async Message Handlers (Phase BIO-1)
 * ============================================================================ */

/**
 * @brief Handle optimizer step request from bio-async
 */
static nimcp_error_t handle_optimizer_step_request(
    const void* msg, size_t msg_size,
    nimcp_bio_promise_t response_promise, void* user_data)
{
    nimcp_optimizer_context_t* ctx = (nimcp_optimizer_context_t*)user_data;
    if (!ctx || !msg || msg_size < sizeof(bio_msg_training_step_t)) {
        nimcp_log(LOG_LEVEL_ERROR, "[%s] Invalid optimizer step request", LOG_MODULE);
        return NIMCP_BIO_ERROR_INVALID_CHANNEL;
    }

    const bio_msg_training_step_t* req = (const bio_msg_training_step_t*)msg;
    nimcp_log(LOG_LEVEL_DEBUG, "[%s] Optimizer step request: batch=%u, lr=%f",
              LOG_MODULE, req->batch_id, ctx->current_lr);

    /* Prepare response with current optimizer state */
    bio_msg_optimizer_step_t response = {0};
    bio_msg_init_header(&response.header, BIO_MSG_OPTIMIZER_STEP,
                       BIO_MODULE_TRAINING_OPTIMIZER, req->header.source_module,
                       sizeof(bio_msg_optimizer_step_t) - sizeof(bio_message_header_t));
    response.step_number = ctx->step_count;
    response.learning_rate = ctx->current_lr;
    response.gradient_norm = (float)ctx->stats.avg_gradient_norm;
    response.weight_updates = ctx->stats.step_count;

    /* Send response via bio-async */
    if (response_promise) {
        nimcp_bio_promise_complete(response_promise, &response);
    }

    return NIMCP_SUCCESS;
}

/**
 * @brief Handle gradient computed message
 */
static nimcp_error_t handle_gradient_computed(
    const void* msg, size_t msg_size,
    nimcp_bio_promise_t response_promise, void* user_data)
{
    nimcp_optimizer_context_t* ctx = (nimcp_optimizer_context_t*)user_data;
    if (!ctx || !msg || msg_size < sizeof(bio_msg_gradient_computed_t)) {
        nimcp_log(LOG_LEVEL_ERROR, "[%s] Invalid gradient computed message", LOG_MODULE);
        return NIMCP_BIO_ERROR_INVALID_CHANNEL;
    }

    const bio_msg_gradient_computed_t* gradient_msg = (const bio_msg_gradient_computed_t*)msg;
    nimcp_log(LOG_LEVEL_DEBUG, "[%s] Gradient computed: layer=%u, norm=%f",
              LOG_MODULE, gradient_msg->layer_id, gradient_msg->gradient_norm);

    /* Update statistics with new gradient information */
    update_gradient_stats(ctx, gradient_msg->gradient_norm);

    /* Broadcast gradient norm on DOPAMINE channel if improvement */
    if (ctx->bio_async_enabled && gradient_msg->gradient_norm < ctx->stats.avg_gradient_norm) {
        bio_msg_training_metric_t metric_msg = {0};
        bio_msg_init_header(&metric_msg.header, BIO_MSG_TRAINING_METRIC,
                           BIO_MODULE_TRAINING_OPTIMIZER, BIO_MODULE_ALL,
                           sizeof(bio_msg_training_metric_t) - sizeof(bio_message_header_t));
        metric_msg.metric_type = 2; // Gradient improvement
        metric_msg.metric_value = ctx->stats.avg_gradient_norm;
        bio_router_broadcast(ctx->bio_ctx, &metric_msg, sizeof(metric_msg));
    }

    (void)response_promise; // No response needed
    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Default Configuration Constructors
 * ============================================================================ */

nimcp_sgd_config_t nimcp_optimizer_sgd_default(float learning_rate) {
    nimcp_sgd_config_t config = {
        .learning_rate = learning_rate,
        .momentum = 0.0f,
        .nesterov = false,
        .dampening = 0.0f,
        .weight_decay = 0.0f
    };
    return config;
}

nimcp_adam_config_t nimcp_optimizer_adam_default(float learning_rate) {
    nimcp_adam_config_t config = {
        .learning_rate = learning_rate,
        .beta1 = 0.9f,
        .beta2 = 0.999f,
        .epsilon = 1e-8f,
        .weight_decay = 0.0f,
        .amsgrad = false
    };
    return config;
}

nimcp_rmsprop_config_t nimcp_optimizer_rmsprop_default(float learning_rate) {
    nimcp_rmsprop_config_t config = {
        .learning_rate = learning_rate,
        .alpha = 0.99f,
        .epsilon = 1e-8f,
        .weight_decay = 0.0f,
        .momentum = 0.0f,
        .centered = false
    };
    return config;
}

nimcp_adagrad_config_t nimcp_optimizer_adagrad_default(float learning_rate) {
    nimcp_adagrad_config_t config = {
        .learning_rate = learning_rate,
        .lr_decay = 0.0f,
        .weight_decay = 0.0f,
        .initial_accumulator = 0.0f,
        .epsilon = 1e-10f
    };
    return config;
}

nimcp_optimizer_config_t nimcp_optimizer_default_config(nimcp_optimizer_type_t type) {
    nimcp_optimizer_config_t config;
    memset(&config, 0, sizeof(config));
    config.type = type;
    config.clip_gradients = false;
    config.gradient_clip_value = 1.0f;
    config.gradient_clip_norm = 0.0f;
    config.use_memory_pool = true;
    config.cow_strategy = UNIFIED_STRATEGY_POOL_DIRECT;

    switch (type) {
        case NIMCP_OPTIMIZER_SGD:
            config.params.sgd = nimcp_optimizer_sgd_default(0.01f);
            break;
        case NIMCP_OPTIMIZER_SGD_MOMENTUM:
            config.params.sgd = nimcp_optimizer_sgd_default(0.01f);
            config.params.sgd.momentum = 0.9f;
            break;
        case NIMCP_OPTIMIZER_NESTEROV:
            config.params.sgd = nimcp_optimizer_sgd_default(0.01f);
            config.params.sgd.momentum = 0.9f;
            config.params.sgd.nesterov = true;
            break;
        case NIMCP_OPTIMIZER_ADAM:
            config.params.adam = nimcp_optimizer_adam_default(0.001f);
            break;
        case NIMCP_OPTIMIZER_ADAMW:
            config.params.adamw.learning_rate = 0.001f;
            config.params.adamw.beta1 = 0.9f;
            config.params.adamw.beta2 = 0.999f;
            config.params.adamw.epsilon = 1e-8f;
            config.params.adamw.weight_decay = 0.01f;
            config.params.adamw.amsgrad = false;
            break;
        case NIMCP_OPTIMIZER_NADAM:
            config.params.adam = nimcp_optimizer_adam_default(0.001f);
            break;
        case NIMCP_OPTIMIZER_RMSPROP:
            config.params.rmsprop = nimcp_optimizer_rmsprop_default(0.01f);
            break;
        case NIMCP_OPTIMIZER_ADAGRAD:
            config.params.adagrad = nimcp_optimizer_adagrad_default(0.01f);
            break;
        default:
            break;
    }

    return config;
}

/* ============================================================================
 * Optimizer Step Implementations
 * ============================================================================ */

static void step_sgd(
    nimcp_optimizer_context_t* ctx,
    float* params,
    const float* gradients,
    size_t count)
{
    float lr = ctx->current_lr;
    float momentum = ctx->config.params.sgd.momentum;
    float dampening = ctx->config.params.sgd.dampening;
    float weight_decay = ctx->config.params.sgd.weight_decay;
    bool nesterov = ctx->config.params.sgd.nesterov;

    if (momentum != 0.0f && ctx->state.momentum.velocity == NULL) {
        ctx->state.momentum.velocity = alloc_buffer(ctx, count);
        ctx->state.momentum.count = count;
        if (ctx->state.momentum.velocity) {
            memset(ctx->state.momentum.velocity, 0, count * sizeof(float));
        }
    }

    float* velocity = ctx->state.momentum.velocity;

    for (size_t i = 0; i < count; i++) {
        float grad = gradients[i];

        /* L2 regularization */
        if (weight_decay != 0.0f) {
            grad += weight_decay * params[i];
        }

        if (momentum != 0.0f && velocity != NULL) {
            if (ctx->step_count == 0) {
                velocity[i] = grad;
            } else {
                velocity[i] = momentum * velocity[i] + (1.0f - dampening) * grad;
            }

            if (nesterov) {
                grad = grad + momentum * velocity[i];
            } else {
                grad = velocity[i];
            }
        }

        params[i] -= lr * grad;
    }
}

static void step_adam(
    nimcp_optimizer_context_t* ctx,
    float* params,
    const float* gradients,
    size_t count)
{
    float lr = ctx->current_lr;
    float beta1 = ctx->config.params.adam.beta1;
    float beta2 = ctx->config.params.adam.beta2;
    float epsilon = ctx->config.params.adam.epsilon;
    float weight_decay = ctx->config.params.adam.weight_decay;
    bool amsgrad = ctx->config.params.adam.amsgrad;

    optimizer_adam_state_t* state = &ctx->state.adam;

    if (state->m == NULL) {
        state->m = alloc_buffer(ctx, count);
        state->v = alloc_buffer(ctx, count);
        if (amsgrad) {
            state->v_max = alloc_buffer(ctx, count);
        }
        state->count = count;
        state->t = 0;

        if (state->m) memset(state->m, 0, count * sizeof(float));
        if (state->v) memset(state->v, 0, count * sizeof(float));
        if (state->v_max) memset(state->v_max, 0, count * sizeof(float));
    }

    state->t++;
    float bias_correction1 = 1.0f - powf(beta1, (float)state->t);
    float bias_correction2 = 1.0f - powf(beta2, (float)state->t);

    for (size_t i = 0; i < count; i++) {
        float grad = gradients[i];

        /* L2 regularization */
        if (weight_decay != 0.0f) {
            grad += weight_decay * params[i];
        }

        /* Update biased first moment estimate */
        state->m[i] = beta1 * state->m[i] + (1.0f - beta1) * grad;

        /* Update biased second raw moment estimate */
        state->v[i] = beta2 * state->v[i] + (1.0f - beta2) * grad * grad;

        float m_hat = state->m[i] / bias_correction1;
        float v_hat;

        if (amsgrad && state->v_max != NULL) {
            state->v_max[i] = fmaxf(state->v_max[i], state->v[i]);
            v_hat = state->v_max[i] / bias_correction2;
        } else {
            v_hat = state->v[i] / bias_correction2;
        }

        params[i] -= lr * m_hat / (sqrtf(v_hat) + epsilon);
    }
}

static void step_adamw(
    nimcp_optimizer_context_t* ctx,
    float* params,
    const float* gradients,
    size_t count)
{
    float lr = ctx->current_lr;
    float beta1 = ctx->config.params.adamw.beta1;
    float beta2 = ctx->config.params.adamw.beta2;
    float epsilon = ctx->config.params.adamw.epsilon;
    float weight_decay = ctx->config.params.adamw.weight_decay;
    bool amsgrad = ctx->config.params.adamw.amsgrad;

    optimizer_adam_state_t* state = &ctx->state.adam;

    if (state->m == NULL) {
        state->m = alloc_buffer(ctx, count);
        state->v = alloc_buffer(ctx, count);
        if (amsgrad) {
            state->v_max = alloc_buffer(ctx, count);
        }
        state->count = count;
        state->t = 0;

        if (state->m) memset(state->m, 0, count * sizeof(float));
        if (state->v) memset(state->v, 0, count * sizeof(float));
        if (state->v_max) memset(state->v_max, 0, count * sizeof(float));
    }

    state->t++;
    float bias_correction1 = 1.0f - powf(beta1, (float)state->t);
    float bias_correction2 = 1.0f - powf(beta2, (float)state->t);

    for (size_t i = 0; i < count; i++) {
        /* Decoupled weight decay */
        if (weight_decay != 0.0f) {
            params[i] -= lr * weight_decay * params[i];
        }

        float grad = gradients[i];

        /* Update biased first moment estimate */
        state->m[i] = beta1 * state->m[i] + (1.0f - beta1) * grad;

        /* Update biased second raw moment estimate */
        state->v[i] = beta2 * state->v[i] + (1.0f - beta2) * grad * grad;

        float m_hat = state->m[i] / bias_correction1;
        float v_hat;

        if (amsgrad && state->v_max != NULL) {
            state->v_max[i] = fmaxf(state->v_max[i], state->v[i]);
            v_hat = state->v_max[i] / bias_correction2;
        } else {
            v_hat = state->v[i] / bias_correction2;
        }

        params[i] -= lr * m_hat / (sqrtf(v_hat) + epsilon);
    }
}

static void step_nadam(
    nimcp_optimizer_context_t* ctx,
    float* params,
    const float* gradients,
    size_t count)
{
    float lr = ctx->current_lr;
    float beta1 = ctx->config.params.adam.beta1;
    float beta2 = ctx->config.params.adam.beta2;
    float epsilon = ctx->config.params.adam.epsilon;
    float weight_decay = ctx->config.params.adam.weight_decay;

    optimizer_adam_state_t* state = &ctx->state.adam;

    if (state->m == NULL) {
        state->m = alloc_buffer(ctx, count);
        state->v = alloc_buffer(ctx, count);
        state->count = count;
        state->t = 0;

        if (state->m) memset(state->m, 0, count * sizeof(float));
        if (state->v) memset(state->v, 0, count * sizeof(float));
    }

    state->t++;
    float bias_correction1 = 1.0f - powf(beta1, (float)state->t);
    float bias_correction2 = 1.0f - powf(beta2, (float)state->t);
    float bias_correction1_next = 1.0f - powf(beta1, (float)(state->t + 1));

    for (size_t i = 0; i < count; i++) {
        float grad = gradients[i];

        if (weight_decay != 0.0f) {
            grad += weight_decay * params[i];
        }

        state->m[i] = beta1 * state->m[i] + (1.0f - beta1) * grad;
        state->v[i] = beta2 * state->v[i] + (1.0f - beta2) * grad * grad;

        float m_hat = state->m[i] / bias_correction1;
        float v_hat = state->v[i] / bias_correction2;

        /* Nesterov momentum */
        float m_bar = (beta1 * m_hat) + ((1.0f - beta1) * grad / bias_correction1_next);

        params[i] -= lr * m_bar / (sqrtf(v_hat) + epsilon);
    }
}

static void step_rmsprop(
    nimcp_optimizer_context_t* ctx,
    float* params,
    const float* gradients,
    size_t count)
{
    float lr = ctx->current_lr;
    float alpha = ctx->config.params.rmsprop.alpha;
    float epsilon = ctx->config.params.rmsprop.epsilon;
    float weight_decay = ctx->config.params.rmsprop.weight_decay;
    float momentum = ctx->config.params.rmsprop.momentum;
    bool centered = ctx->config.params.rmsprop.centered;

    optimizer_rmsprop_state_t* state = &ctx->state.rmsprop;

    if (state->square_avg == NULL) {
        state->square_avg = alloc_buffer(ctx, count);
        if (momentum != 0.0f) {
            state->momentum_buffer = alloc_buffer(ctx, count);
        }
        if (centered) {
            state->grad_avg = alloc_buffer(ctx, count);
        }
        state->count = count;

        if (state->square_avg) memset(state->square_avg, 0, count * sizeof(float));
        if (state->momentum_buffer) memset(state->momentum_buffer, 0, count * sizeof(float));
        if (state->grad_avg) memset(state->grad_avg, 0, count * sizeof(float));
    }

    for (size_t i = 0; i < count; i++) {
        float grad = gradients[i];

        if (weight_decay != 0.0f) {
            grad += weight_decay * params[i];
        }

        state->square_avg[i] = alpha * state->square_avg[i] + (1.0f - alpha) * grad * grad;

        float avg;
        if (centered && state->grad_avg != NULL) {
            state->grad_avg[i] = alpha * state->grad_avg[i] + (1.0f - alpha) * grad;
            avg = state->square_avg[i] - state->grad_avg[i] * state->grad_avg[i];
        } else {
            avg = state->square_avg[i];
        }

        float update;
        if (momentum != 0.0f && state->momentum_buffer != NULL) {
            state->momentum_buffer[i] = momentum * state->momentum_buffer[i] +
                                        grad / (sqrtf(avg) + epsilon);
            update = state->momentum_buffer[i];
        } else {
            update = grad / (sqrtf(avg) + epsilon);
        }

        params[i] -= lr * update;
    }
}

static void step_adagrad(
    nimcp_optimizer_context_t* ctx,
    float* params,
    const float* gradients,
    size_t count)
{
    float base_lr = ctx->config.params.adagrad.learning_rate;
    float lr_decay = ctx->config.params.adagrad.lr_decay;
    float weight_decay = ctx->config.params.adagrad.weight_decay;
    float initial_acc = ctx->config.params.adagrad.initial_accumulator;
    float epsilon = ctx->config.params.adagrad.epsilon;

    optimizer_adagrad_state_t* state = &ctx->state.adagrad;

    if (state->sum == NULL) {
        state->sum = alloc_buffer(ctx, count);
        state->count = count;
        state->step = 0;

        if (state->sum) {
            for (size_t i = 0; i < count; i++) {
                state->sum[i] = initial_acc;
            }
        }
    }

    state->step++;
    float clr = base_lr / (1.0f + (float)(state->step - 1) * lr_decay);

    for (size_t i = 0; i < count; i++) {
        float grad = gradients[i];

        if (weight_decay != 0.0f) {
            grad += weight_decay * params[i];
        }

        state->sum[i] += grad * grad;
        params[i] -= clr * grad / (sqrtf(state->sum[i]) + epsilon);
    }

    ctx->current_lr = clr;
}

/* ============================================================================
 * Core Optimizer Operations
 * ============================================================================ */

nimcp_optimizer_context_t* nimcp_optimizer_create(
    const nimcp_optimizer_config_t* config,
    nimcp_sec_integration_t* security_ctx,
    unified_mem_manager_t memory_mgr)
{
    if (!config) {
        return NULL;
    }

    nimcp_optimizer_context_t* ctx = (nimcp_optimizer_context_t*)nimcp_calloc(1, sizeof(*ctx));
    if (!ctx) {
        return NULL;
    }

    memcpy(&ctx->config, config, sizeof(nimcp_optimizer_config_t));
    ctx->security_ctx = security_ctx;
    ctx->memory_mgr = memory_mgr;

    /* Set initial learning rate based on optimizer type */
    switch (config->type) {
        case NIMCP_OPTIMIZER_SGD:
        case NIMCP_OPTIMIZER_SGD_MOMENTUM:
        case NIMCP_OPTIMIZER_NESTEROV:
            ctx->current_lr = config->params.sgd.learning_rate;
            break;
        case NIMCP_OPTIMIZER_ADAM:
        case NIMCP_OPTIMIZER_NADAM:
            ctx->current_lr = config->params.adam.learning_rate;
            break;
        case NIMCP_OPTIMIZER_ADAMW:
            ctx->current_lr = config->params.adamw.learning_rate;
            break;
        case NIMCP_OPTIMIZER_RMSPROP:
            ctx->current_lr = config->params.rmsprop.learning_rate;
            break;
        case NIMCP_OPTIMIZER_ADAGRAD:
            ctx->current_lr = config->params.adagrad.learning_rate;
            break;
        default:
            ctx->current_lr = 0.01f;
            break;
    }

    /* Initialize stats */
    ctx->stats.min_gradient_norm = (double)HUGE_VALF;
    ctx->stats.current_lr = ctx->current_lr;

    /* Security registration */
    if (security_ctx) {
        nimcp_result_t err = nimcp_sec_register_module(
            security_ctx,
            OPTIMIZER_MODULE_NAME,
            NIMCP_SEC_CAT_PLASTICITY,
            &ctx->module_id
        );
        ctx->security_registered = (err == NIMCP_SUCCESS);
        nimcp_log(LOG_LEVEL_DEBUG, "[%s] Security registration: %s (id=%u)",
                  LOG_MODULE, ctx->security_registered ? "success" : "failed", ctx->module_id);
    }

    /* Bio-async integration (Phase BIO-1) */
    ctx->bio_async_enabled = false;
    if (bio_router_is_initialized()) {
        bio_module_info_t bio_info = {
            .module_id = BIO_MODULE_TRAINING_OPTIMIZER,
            .module_name = LOG_MODULE,
            .inbox_capacity = 64,
            .user_data = ctx
        };
        ctx->bio_ctx = bio_router_register_module(&bio_info);
        if (ctx->bio_ctx) {
            ctx->bio_async_enabled = true;
            /* Register message handlers */
            bio_router_register_handler(ctx->bio_ctx,
                                       BIO_MSG_TRAINING_STEP_REQUEST,
                                       handle_optimizer_step_request);
            bio_router_register_handler(ctx->bio_ctx,
                                       BIO_MSG_GRADIENT_COMPUTED,
                                       handle_gradient_computed);
            nimcp_log(LOG_LEVEL_INFO, "[%s] Bio-async integration enabled for optimizer type=%s",
                     LOG_MODULE, nimcp_optimizer_type_name(config->type));
        } else {
            nimcp_log(LOG_LEVEL_WARN, "[%s] Bio-async integration failed, continuing without it",
                     LOG_MODULE);
        }
    }

    ctx->initialized = true;
    nimcp_log(LOG_LEVEL_INFO, "[%s] Optimizer created: type=%s, lr=%f",
              LOG_MODULE, nimcp_optimizer_type_name(config->type), ctx->current_lr);
    return ctx;
}

void nimcp_optimizer_destroy(nimcp_optimizer_context_t* ctx) {
    if (!ctx) return;

    /* Bio-async cleanup (Phase BIO-1) */
    if (ctx->bio_async_enabled && ctx->bio_ctx) {
        bio_router_unregister_module(ctx->bio_ctx);
        ctx->bio_ctx = NULL;
        ctx->bio_async_enabled = false;
        nimcp_log(LOG_LEVEL_DEBUG, "[%s] Bio-async integration cleaned up", LOG_MODULE);
    }

    /* Free optimizer state */
    switch (ctx->config.type) {
        case NIMCP_OPTIMIZER_SGD:
        case NIMCP_OPTIMIZER_SGD_MOMENTUM:
        case NIMCP_OPTIMIZER_NESTEROV:
            free_buffer(ctx, ctx->state.momentum.velocity);
            break;
        case NIMCP_OPTIMIZER_ADAM:
        case NIMCP_OPTIMIZER_ADAMW:
        case NIMCP_OPTIMIZER_NADAM:
            free_buffer(ctx, ctx->state.adam.m);
            free_buffer(ctx, ctx->state.adam.v);
            free_buffer(ctx, ctx->state.adam.v_max);
            break;
        case NIMCP_OPTIMIZER_RMSPROP:
            free_buffer(ctx, ctx->state.rmsprop.square_avg);
            free_buffer(ctx, ctx->state.rmsprop.momentum_buffer);
            free_buffer(ctx, ctx->state.rmsprop.grad_avg);
            break;
        case NIMCP_OPTIMIZER_ADAGRAD:
            free_buffer(ctx, ctx->state.adagrad.sum);
            break;
        default:
            break;
    }

    nimcp_log(LOG_LEVEL_INFO, "[%s] Optimizer destroyed: type=%s",
              LOG_MODULE, nimcp_optimizer_type_name(ctx->config.type));
    nimcp_free(ctx);
}

nimcp_result_t nimcp_optimizer_init_params(
    nimcp_optimizer_context_t* ctx,
    size_t num_params)
{
    if (!ctx || num_params == 0) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    /* State will be lazily initialized on first step */
    return NIMCP_SUCCESS;
}

nimcp_result_t nimcp_optimizer_step(
    nimcp_optimizer_context_t* ctx,
    float* params,
    const float* gradients,
    size_t count)
{
    if (!ctx || !params || !gradients || count == 0) {
        nimcp_log(LOG_LEVEL_ERROR, "[%s] Invalid parameters for optimizer step", LOG_MODULE);
        return NIMCP_ERROR_INVALID_PARAM;
    }

    uint64_t start_time = get_time_ns();

    /* Compute gradient norm */
    float grad_norm = nimcp_optimizer_gradient_norm(gradients, count);

    /* Check for gradient explosion */
    if (!isfinite(grad_norm) || grad_norm > 1e6f) {
        nimcp_log(LOG_LEVEL_WARN, "[%s] Gradient explosion detected: norm=%f",
                  LOG_MODULE, grad_norm);
    }

    nimcp_log(LOG_LEVEL_TRACE, "[%s] Optimizer step: lr=%f, grad_norm=%f, step=%lu",
              LOG_MODULE, ctx->current_lr, grad_norm, ctx->step_count);

    /* Apply gradient clipping if enabled */
    float* clipped_grads = NULL;
    const float* grads_to_use = gradients;

    if (ctx->config.clip_gradients) {
        clipped_grads = alloc_buffer(ctx, count);
        if (clipped_grads) {
            memcpy(clipped_grads, gradients, count * sizeof(float));

            if (ctx->config.gradient_clip_norm > 0.0f) {
                nimcp_optimizer_clip_by_norm(clipped_grads, count, ctx->config.gradient_clip_norm);
            }
            if (ctx->config.gradient_clip_value > 0.0f) {
                size_t clips = nimcp_optimizer_clip_by_value(
                    clipped_grads, count, ctx->config.gradient_clip_value);
                ctx->stats.gradient_clips += clips;
                if (clips > 0) {
                    nimcp_log(LOG_LEVEL_DEBUG, "[%s] Clipped %zu gradients", LOG_MODULE, clips);
                }
            }
            grads_to_use = clipped_grads;
        }
    }

    /* Perform optimizer step */
    switch (ctx->config.type) {
        case NIMCP_OPTIMIZER_SGD:
        case NIMCP_OPTIMIZER_SGD_MOMENTUM:
        case NIMCP_OPTIMIZER_NESTEROV:
            step_sgd(ctx, params, grads_to_use, count);
            break;
        case NIMCP_OPTIMIZER_ADAM:
            step_adam(ctx, params, grads_to_use, count);
            break;
        case NIMCP_OPTIMIZER_ADAMW:
            step_adamw(ctx, params, grads_to_use, count);
            break;
        case NIMCP_OPTIMIZER_NADAM:
            step_nadam(ctx, params, grads_to_use, count);
            break;
        case NIMCP_OPTIMIZER_RMSPROP:
            step_rmsprop(ctx, params, grads_to_use, count);
            break;
        case NIMCP_OPTIMIZER_ADAGRAD:
            step_adagrad(ctx, params, grads_to_use, count);
            break;
        case NIMCP_OPTIMIZER_CUSTOM:
            if (ctx->config.params.custom.step_fn) {
                ctx->config.params.custom.step_fn(
                    params, grads_to_use, ctx->state.custom, count,
                    ctx->config.params.custom.user_data);
            }
            break;
        default:
            free_buffer(ctx, clipped_grads);
            return NIMCP_ERROR_NOT_IMPLEMENTED;
    }

    free_buffer(ctx, clipped_grads);

    /* Update statistics */
    ctx->step_count++;
    ctx->stats.step_count = ctx->step_count;
    update_gradient_stats(ctx, grad_norm);
    ctx->stats.current_lr = ctx->current_lr;

    uint64_t end_time = get_time_ns();
    ctx->stats.total_compute_time_ns += (end_time - start_time);

    /* Phase BIO-1: Broadcast optimizer step completion */
    if (ctx->bio_async_enabled) {
        bio_msg_optimizer_step_t step_msg = {0};
        bio_msg_init_header(&step_msg.header, BIO_MSG_OPTIMIZER_STEP,
                           BIO_MODULE_TRAINING_OPTIMIZER, BIO_MODULE_ALL,
                           sizeof(bio_msg_optimizer_step_t) - sizeof(bio_message_header_t));
        step_msg.step_number = ctx->step_count;
        step_msg.learning_rate = ctx->current_lr;
        step_msg.gradient_norm = grad_norm;
        step_msg.weight_updates = count;

        bio_router_broadcast(ctx->bio_ctx, &step_msg, sizeof(step_msg));
    }

    /* Phase LOG-1: Log significant events */
    if (ctx->step_count % 100 == 0) {
        nimcp_log(LOG_LEVEL_INFO, "[%s] Optimizer step %lu: lr=%f, grad_norm=%f, avg_grad=%f",
                  LOG_MODULE, ctx->step_count, ctx->current_lr, grad_norm,
                  ctx->stats.avg_gradient_norm);
    }

    return NIMCP_SUCCESS;
}

nimcp_result_t nimcp_optimizer_step_group(
    nimcp_optimizer_context_t* ctx,
    nimcp_param_group_t* group)
{
    if (!ctx || !group || !group->params || !group->gradients) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    /* Temporarily adjust learning rate for this group */
    float saved_lr = ctx->current_lr;
    if (group->learning_rate > 0.0f) {
        ctx->current_lr = group->learning_rate;
    }

    nimcp_result_t result = nimcp_optimizer_step(
        ctx, group->params, group->gradients, group->count);

    ctx->current_lr = saved_lr;
    return result;
}

void nimcp_optimizer_zero_grad(nimcp_param_group_t* group) {
    if (!group || !group->gradients) return;
    memset(group->gradients, 0, group->count * sizeof(float));
}

float nimcp_optimizer_get_lr(const nimcp_optimizer_context_t* ctx) {
    if (!ctx) return 0.0f;
    return ctx->current_lr;
}

void nimcp_optimizer_set_lr(nimcp_optimizer_context_t* ctx, float lr) {
    if (!ctx || lr <= 0.0f) return;

    float old_lr = ctx->current_lr;
    ctx->current_lr = lr;
    ctx->stats.current_lr = lr;

    /* Phase LOG-1: Log learning rate changes */
    nimcp_log(LOG_LEVEL_INFO, "[%s] Learning rate changed: %f -> %f (step=%lu)",
              LOG_MODULE, old_lr, lr, ctx->step_count);

    /* Phase BIO-1: Broadcast LR change on DOPAMINE if improvement (increase) */
    if (ctx->bio_async_enabled && lr > old_lr) {
        bio_msg_training_metric_t metric_msg = {0};
        bio_msg_init_header(&metric_msg.header, BIO_MSG_TRAINING_METRIC,
                           BIO_MODULE_TRAINING_OPTIMIZER, BIO_MODULE_ALL,
                           sizeof(bio_msg_training_metric_t) - sizeof(bio_message_header_t));
        metric_msg.metric_type = 3; // Learning rate change
        metric_msg.metric_value = lr;
        bio_router_broadcast(ctx->bio_ctx, &metric_msg, sizeof(metric_msg));
    }
}

uint64_t nimcp_optimizer_get_step(const nimcp_optimizer_context_t* ctx) {
    if (!ctx) return 0;
    return ctx->step_count;
}

void nimcp_optimizer_reset_state(nimcp_optimizer_context_t* ctx) {
    if (!ctx) return;

    switch (ctx->config.type) {
        case NIMCP_OPTIMIZER_SGD:
        case NIMCP_OPTIMIZER_SGD_MOMENTUM:
        case NIMCP_OPTIMIZER_NESTEROV:
            if (ctx->state.momentum.velocity) {
                memset(ctx->state.momentum.velocity, 0,
                       ctx->state.momentum.count * sizeof(float));
            }
            break;
        case NIMCP_OPTIMIZER_ADAM:
        case NIMCP_OPTIMIZER_ADAMW:
        case NIMCP_OPTIMIZER_NADAM:
            if (ctx->state.adam.m) {
                memset(ctx->state.adam.m, 0, ctx->state.adam.count * sizeof(float));
            }
            if (ctx->state.adam.v) {
                memset(ctx->state.adam.v, 0, ctx->state.adam.count * sizeof(float));
            }
            if (ctx->state.adam.v_max) {
                memset(ctx->state.adam.v_max, 0, ctx->state.adam.count * sizeof(float));
            }
            ctx->state.adam.t = 0;
            break;
        case NIMCP_OPTIMIZER_RMSPROP:
            if (ctx->state.rmsprop.square_avg) {
                memset(ctx->state.rmsprop.square_avg, 0,
                       ctx->state.rmsprop.count * sizeof(float));
            }
            if (ctx->state.rmsprop.momentum_buffer) {
                memset(ctx->state.rmsprop.momentum_buffer, 0,
                       ctx->state.rmsprop.count * sizeof(float));
            }
            if (ctx->state.rmsprop.grad_avg) {
                memset(ctx->state.rmsprop.grad_avg, 0,
                       ctx->state.rmsprop.count * sizeof(float));
            }
            break;
        case NIMCP_OPTIMIZER_ADAGRAD:
            if (ctx->state.adagrad.sum) {
                float initial = ctx->config.params.adagrad.initial_accumulator;
                for (size_t i = 0; i < ctx->state.adagrad.count; i++) {
                    ctx->state.adagrad.sum[i] = initial;
                }
            }
            ctx->state.adagrad.step = 0;
            break;
        default:
            break;
    }

    ctx->step_count = 0;
}

/* ============================================================================
 * Statistics and Diagnostics
 * ============================================================================ */

nimcp_result_t nimcp_optimizer_get_stats(
    const nimcp_optimizer_context_t* ctx,
    nimcp_optimizer_stats_t* stats)
{
    if (!ctx || !stats) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    memcpy(stats, &ctx->stats, sizeof(nimcp_optimizer_stats_t));
    return NIMCP_SUCCESS;
}

void nimcp_optimizer_reset_stats(nimcp_optimizer_context_t* ctx) {
    if (!ctx) return;
    memset(&ctx->stats, 0, sizeof(nimcp_optimizer_stats_t));
    ctx->stats.min_gradient_norm = (double)HUGE_VALF;
    ctx->stats.current_lr = ctx->current_lr;
}

const char* nimcp_optimizer_type_name(nimcp_optimizer_type_t type) {
    switch (type) {
        case NIMCP_OPTIMIZER_SGD:          return "SGD";
        case NIMCP_OPTIMIZER_SGD_MOMENTUM: return "SGD+Momentum";
        case NIMCP_OPTIMIZER_NESTEROV:     return "Nesterov";
        case NIMCP_OPTIMIZER_ADAGRAD:      return "AdaGrad";
        case NIMCP_OPTIMIZER_RMSPROP:      return "RMSprop";
        case NIMCP_OPTIMIZER_ADAM:         return "Adam";
        case NIMCP_OPTIMIZER_ADAMW:        return "AdamW";
        case NIMCP_OPTIMIZER_NADAM:        return "NAdam";
        case NIMCP_OPTIMIZER_CUSTOM:       return "Custom";
        default:                           return "Unknown";
    }
}

nimcp_result_t nimcp_optimizer_validate_config(const nimcp_optimizer_config_t* config) {
    if (!config) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    if (config->type >= NIMCP_OPTIMIZER_TYPE_COUNT) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    /* Validate type-specific parameters */
    switch (config->type) {
        case NIMCP_OPTIMIZER_SGD:
        case NIMCP_OPTIMIZER_SGD_MOMENTUM:
        case NIMCP_OPTIMIZER_NESTEROV:
            if (config->params.sgd.learning_rate <= 0.0f) {
                return NIMCP_ERROR_INVALID_PARAM;
            }
            if (config->params.sgd.momentum < 0.0f || config->params.sgd.momentum > 1.0f) {
                return NIMCP_ERROR_INVALID_PARAM;
            }
            break;
        case NIMCP_OPTIMIZER_ADAM:
        case NIMCP_OPTIMIZER_NADAM:
            if (config->params.adam.learning_rate <= 0.0f) {
                return NIMCP_ERROR_INVALID_PARAM;
            }
            if (config->params.adam.beta1 < 0.0f || config->params.adam.beta1 >= 1.0f) {
                return NIMCP_ERROR_INVALID_PARAM;
            }
            if (config->params.adam.beta2 < 0.0f || config->params.adam.beta2 >= 1.0f) {
                return NIMCP_ERROR_INVALID_PARAM;
            }
            break;
        case NIMCP_OPTIMIZER_RMSPROP:
            if (config->params.rmsprop.learning_rate <= 0.0f) {
                return NIMCP_ERROR_INVALID_PARAM;
            }
            break;
        case NIMCP_OPTIMIZER_ADAGRAD:
            if (config->params.adagrad.learning_rate <= 0.0f) {
                return NIMCP_ERROR_INVALID_PARAM;
            }
            break;
        default:
            break;
    }

    return NIMCP_SUCCESS;
}

bool nimcp_optimizer_is_registered(const nimcp_optimizer_context_t* ctx) {
    if (!ctx) return false;
    return ctx->security_registered;
}

/* ============================================================================
 * Utility Functions
 * ============================================================================ */

float nimcp_optimizer_gradient_norm(const float* gradients, size_t count) {
    if (!gradients || count == 0) return 0.0f;

    float sum_sq = 0.0f;
    for (size_t i = 0; i < count; i++) {
        sum_sq += gradients[i] * gradients[i];
    }
    return sqrtf(sum_sq);
}

size_t nimcp_optimizer_clip_by_value(
    float* gradients,
    size_t count,
    float max_value)
{
    if (!gradients || count == 0 || max_value <= 0.0f) return 0;

    size_t clips = 0;
    for (size_t i = 0; i < count; i++) {
        if (gradients[i] > max_value) {
            gradients[i] = max_value;
            clips++;
        } else if (gradients[i] < -max_value) {
            gradients[i] = -max_value;
            clips++;
        }
    }
    return clips;
}

float nimcp_optimizer_clip_by_norm(
    float* gradients,
    size_t count,
    float max_norm)
{
    if (!gradients || count == 0 || max_norm <= 0.0f) return 0.0f;

    float norm = nimcp_optimizer_gradient_norm(gradients, count);
    if (norm > max_norm) {
        float scale = max_norm / norm;
        for (size_t i = 0; i < count; i++) {
            gradients[i] *= scale;
        }
    }
    return norm;
}
