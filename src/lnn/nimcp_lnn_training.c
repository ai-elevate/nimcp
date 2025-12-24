/**
 * @file nimcp_lnn_training.c
 * @brief Training integration implementation for Liquid Neural Networks
 *
 * WHAT: Implements LNN training pipeline with full NIMCP integration
 * WHY:  Enable end-to-end training of continuous-time neural networks
 * HOW:  Wraps optimizer, gradient manager, loss, and training bridges
 */

#include "lnn/nimcp_lnn_training.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "lnn/nimcp_lnn.h"
#include "lnn/nimcp_lnn_gradient.h"
#include "lnn/nimcp_lnn_network.h"
#include "nimcp.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/memory/nimcp_unified_memory.h"
#include "middleware/training/nimcp_cognitive_training_bridge.h"
#include "middleware/training/nimcp_training_logic_bridge.h"
#include "middleware/immune/nimcp_training_immune.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

/*=============================================================================
 * Constants
 *===========================================================================*/

#define LNN_TRAINING_DEFAULT_LR           0.001f
#define LNN_TRAINING_DEFAULT_WEIGHT_DECAY 0.0f
#define LNN_TRAINING_DEFAULT_BETA1        0.9f
#define LNN_TRAINING_DEFAULT_BETA2        0.999f
#define LNN_TRAINING_DEFAULT_EPSILON      1e-8f
#define LNN_TRAINING_MAX_SCHEDULE_PARAMS  8

/*=============================================================================
 * Helper Functions
 *===========================================================================*/

/**
 * @brief Compute new LR for step schedule
 */
static float compute_step_schedule_lr(
    float base_lr,
    uint64_t step,
    uint32_t step_size,
    float gamma
) {
    uint64_t num_decays = step / step_size;
    return base_lr * powf(gamma, (float)num_decays);
}

/**
 * @brief Compute new LR for exponential schedule
 */
static float compute_exponential_schedule_lr(
    float base_lr,
    uint64_t step,
    float gamma
) {
    return base_lr * powf(gamma, (float)step);
}

/**
 * @brief Compute new LR for cosine schedule
 */
static float compute_cosine_schedule_lr(
    float base_lr,
    uint64_t step,
    uint64_t T_max
) {
    if (T_max == 0) return base_lr;
    float progress = (float)step / (float)T_max;
    if (progress > 1.0f) progress = 1.0f;
    return base_lr * 0.5f * (1.0f + cosf(3.14159265f * progress));
}

/**
 * @brief Compute new LR for warmup+cosine schedule
 */
static float compute_warmup_cosine_lr(
    float base_lr,
    uint64_t step,
    uint64_t warmup_steps,
    uint64_t T_max
) {
    if (step < warmup_steps) {
        // Linear warmup
        return base_lr * ((float)step / (float)warmup_steps);
    } else {
        // Cosine decay after warmup
        uint64_t cosine_step = step - warmup_steps;
        uint64_t cosine_T_max = (T_max > warmup_steps) ? (T_max - warmup_steps) : 1;
        return compute_cosine_schedule_lr(base_lr, cosine_step, cosine_T_max);
    }
}

/*=============================================================================
 * Lifecycle Functions
 *===========================================================================*/

int lnn_training_config_default(lnn_training_config_t* config) {
    if (!config) {
        NIMCP_LOGGING_ERROR("NULL config pointer");
        return LNN_ERROR_NULL_POINTER;
    }

    memset(config, 0, sizeof(lnn_training_config_t));

    /* Optimizer defaults (Adam) */
    config->optimizer_type = NIMCP_OPTIMIZER_ADAM;
    config->learning_rate = LNN_TRAINING_DEFAULT_LR;
    config->weight_decay = LNN_TRAINING_DEFAULT_WEIGHT_DECAY;
    config->beta1 = LNN_TRAINING_DEFAULT_BETA1;
    config->beta2 = LNN_TRAINING_DEFAULT_BETA2;
    config->epsilon = LNN_TRAINING_DEFAULT_EPSILON;

    /* Gradient manager defaults */
    config->gradient_clip_norm = 1.0f;  /* Enable gradient clipping by default */
    config->use_gradient_scaling = false;
    config->accum_mode = NIMCP_GRAD_ACCUM_MEAN;
    config->accumulation_steps = 1;

    /* Loss defaults (MSE) */
    config->loss_type = NIMCP_LOSS_MSE;
    config->reduction = NIMCP_LOSS_REDUCE_MEAN;

    /* LNN-specific defaults (adjoint method) */
    config->lnn_train_mode = LNN_TRAIN_ADJOINT;
    config->bptt_truncation = 100;
    config->use_adjoint_checkpointing = true;

    /* LR schedule defaults (constant) */
    config->lr_schedule = LNN_LR_SCHEDULE_CONSTANT;
    config->n_schedule_params = 0;

    /* Integration defaults (all disabled) */
    config->enable_cognitive_integration = false;
    config->enable_logic_integration = false;
    config->enable_immune_integration = false;
    config->enable_plasticity_integration = false;
    config->enable_bio_async = false;

    /* Training behavior defaults */
    config->use_mixed_precision = false;
    config->validate_gradients = true;
    config->track_statistics = true;

    return LNN_SUCCESS;
}

lnn_training_ctx_t* lnn_training_create(
    lnn_network_t* network,
    const lnn_training_config_t* config
) {
    if (!network) {
        NIMCP_LOGGING_ERROR("NULL network pointer");
        return NULL;
    }

    /* Use defaults if no config provided */
    lnn_training_config_t default_config;
    if (!config) {
        lnn_training_config_default(&default_config);
        config = &default_config;
    }

    /* Validate config */
    if (lnn_training_validate_config(config) != LNN_SUCCESS) {
        NIMCP_LOGGING_ERROR("Invalid training configuration");
        return NULL;
    }

    /* Allocate context */
    lnn_training_ctx_t* ctx = (lnn_training_ctx_t*)malloc(sizeof(lnn_training_ctx_t));
    if (!ctx) {
        NIMCP_LOGGING_ERROR("Failed to allocate training context");
        return NULL;
    }
    memset(ctx, 0, sizeof(lnn_training_ctx_t));

    /* Store network reference */
    ctx->network = network;

    /* Copy config */
    memcpy(&ctx->config, config, sizeof(lnn_training_config_t));

    /* Initialize state */
    ctx->step_count = 0;
    ctx->epoch_count = 0;
    ctx->current_lr = config->learning_rate;
    ctx->current_loss = 0.0f;

    /* Create optimizer */
    nimcp_optimizer_config_t opt_config = nimcp_optimizer_default_config(config->optimizer_type);
    if (config->optimizer_type == NIMCP_OPTIMIZER_ADAM) {
        opt_config.params.adam.learning_rate = config->learning_rate;
        opt_config.params.adam.beta1 = config->beta1;
        opt_config.params.adam.beta2 = config->beta2;
        opt_config.params.adam.epsilon = config->epsilon;
        opt_config.params.adam.weight_decay = config->weight_decay;
    } else if (config->optimizer_type == NIMCP_OPTIMIZER_SGD) {
        opt_config.params.sgd.learning_rate = config->learning_rate;
        opt_config.params.sgd.weight_decay = config->weight_decay;
    }
    opt_config.clip_gradients = (config->gradient_clip_norm > 0);
    opt_config.gradient_clip_norm = config->gradient_clip_norm;

    ctx->optimizer = nimcp_optimizer_create(&opt_config, NULL, NULL);
    if (!ctx->optimizer) {
        NIMCP_LOGGING_ERROR("Failed to create optimizer");
        lnn_training_destroy(ctx);
        return NULL;
    }

    /* Create gradient manager */
    nimcp_gradient_manager_config_t grad_config = nimcp_gradient_manager_default_config();
    grad_config.use_accumulation = (config->accumulation_steps > 1);
    grad_config.accumulation.accumulation_steps = config->accumulation_steps;
    grad_config.accumulation.mode = config->accum_mode;
    grad_config.use_scaling = config->use_gradient_scaling;
    grad_config.check_nan_inf = config->validate_gradients;
    grad_config.track_statistics = config->track_statistics;

    ctx->gradient_manager = nimcp_gradient_manager_create(&grad_config);
    if (!ctx->gradient_manager) {
        NIMCP_LOGGING_ERROR("Failed to create gradient manager");
        lnn_training_destroy(ctx);
        return NULL;
    }

    /* Create loss context */
    nimcp_loss_config_t loss_config = nimcp_loss_default_config(config->loss_type);
    if (config->loss_type == NIMCP_LOSS_MSE) {
        loss_config.params.mse.reduction = config->reduction;
        loss_config.params.mse.compute_gradient = true;
    }
    loss_config.clip_gradients = (config->gradient_clip_norm > 0);
    loss_config.gradient_clip_value = config->gradient_clip_norm;

    ctx->loss_context = nimcp_loss_create(&loss_config, NULL, NULL);
    if (!ctx->loss_context) {
        NIMCP_LOGGING_ERROR("Failed to create loss context");
        lnn_training_destroy(ctx);
        return NULL;
    }

    /* Create gradient context for LNN adjoint computation */
    uint32_t max_steps = config->bptt_truncation > 0 ? config->bptt_truncation : 100;
    ctx->gradient_ctx = lnn_gradient_ctx_create(
        network,
        max_steps,
        config->use_adjoint_checkpointing,
        10  /* checkpoint_interval */
    );
    if (!ctx->gradient_ctx) {
        NIMCP_LOGGING_WARN("Failed to create gradient context, training may fail");
        /* Non-fatal: some use cases may not need gradients */
    }

    /* Create output and gradient buffers */
    uint32_t output_size = network->n_outputs;
    if (output_size > 0) {
        uint32_t dims[1] = { output_size };
        ctx->output_buffer = nimcp_tensor_create(dims, 1, NIMCP_DTYPE_F32);
        ctx->loss_gradient = nimcp_tensor_create(dims, 1, NIMCP_DTYPE_F32);
    } else {
        ctx->output_buffer = NULL;
        ctx->loss_gradient = NULL;
    }

    /* Initialize LR schedule */
    ctx->lr_schedule = config->lr_schedule;
    if (config->n_schedule_params > 0) {
        ctx->n_schedule_params = config->n_schedule_params;
        ctx->lr_schedule_params = (float*)malloc(config->n_schedule_params * sizeof(float));
        if (ctx->lr_schedule_params) {
            memcpy(ctx->lr_schedule_params, config->lr_schedule_params,
                   config->n_schedule_params * sizeof(float));
        }
    }
    ctx->last_lr_update_step = 0;
    ctx->plateau_best_metric = FLT_MAX;
    ctx->plateau_patience_count = 0;

    /* Initialize gradient accumulation */
    ctx->accumulate_gradients = (config->accumulation_steps > 1);
    ctx->accumulation_steps = config->accumulation_steps;
    ctx->current_accumulation = 0;

    /* Initialize bridges to NULL (connected later) */
    ctx->cognitive_training_bridge = NULL;
    ctx->training_logic_bridge = NULL;
    ctx->training_immune_system = NULL;
    ctx->training_plasticity_bridge = NULL;

    /* Initialize bio-async */
    ctx->bio_async_enabled = false;
    ctx->bio_ctx = NULL;

    /* Initialize callbacks */
    ctx->on_step_complete = NULL;
    ctx->on_epoch_complete = NULL;
    ctx->on_lr_change = NULL;
    ctx->callback_user_data = NULL;

    /* Initialize statistics */
    memset(&ctx->stats, 0, sizeof(lnn_training_stats_t));
    ctx->stats.min_loss = FLT_MAX;
    ctx->stats.max_loss = -FLT_MAX;
    ctx->stats.min_gradient_norm = FLT_MAX;
    ctx->stats.max_gradient_norm = -FLT_MAX;
    ctx->stats.current_lr = config->learning_rate;
    ctx->stats.min_lr = config->learning_rate;
    ctx->stats.max_lr = config->learning_rate;
    ctx->stats.avg_tau_network = 0.0f;
    ctx->stats.min_tau_network = FLT_MAX;
    ctx->stats.max_tau_network = -FLT_MAX;

    /* Set network to training mode */
    lnn_set_training(network, true);

    NIMCP_LOGGING_INFO("Created LNN training context with %s optimizer, %s loss",
                       nimcp_optimizer_type_name(config->optimizer_type),
                       nimcp_loss_type_name(config->loss_type));

    return ctx;
}

void lnn_training_destroy(lnn_training_ctx_t* ctx) {
    if (!ctx) return;

    /* Destroy optimizer */
    if (ctx->optimizer) {
        nimcp_optimizer_destroy(ctx->optimizer);
    }

    /* Destroy gradient manager */
    if (ctx->gradient_manager) {
        nimcp_gradient_manager_destroy(ctx->gradient_manager);
    }

    /* Destroy loss context */
    if (ctx->loss_context) {
        nimcp_loss_destroy(ctx->loss_context);
    }

    /* Destroy gradient context */
    if (ctx->gradient_ctx) {
        lnn_gradient_ctx_destroy(ctx->gradient_ctx);
    }

    /* Destroy buffers */
    if (ctx->output_buffer) {
        nimcp_tensor_destroy(ctx->output_buffer);
    }
    if (ctx->loss_gradient) {
        nimcp_tensor_destroy(ctx->loss_gradient);
    }

    /* Free LR schedule params */
    if (ctx->lr_schedule_params) {
        free(ctx->lr_schedule_params);
    }

    /* Note: We don't destroy bridges - they're owned externally */

    /* Free context */
    free(ctx);

    NIMCP_LOGGING_INFO("Destroyed LNN training context");
}

/*=============================================================================
 * Training Step Functions
 *===========================================================================*/

int lnn_training_step(
    lnn_training_ctx_t* ctx,
    const nimcp_tensor_t* inputs,
    const nimcp_tensor_t* targets,
    float* loss_out
) {
    if (!ctx || !inputs || !targets) {
        NIMCP_LOGGING_ERROR("NULL pointer in lnn_training_step");
        return LNN_ERROR_NULL_POINTER;
    }

    uint64_t step_start_ms = nimcp_time_get_ms();
    float loss = 0.0f;
    float grad_norm = 0.0f;

    /* 1. Forward pass through network */
    float dt = ctx->network->config ? ctx->network->config->default_dt : 1.0f;
    int result = lnn_network_forward_step(ctx->network, inputs, ctx->output_buffer, dt);
    if (result != 0) {
        NIMCP_LOGGING_ERROR("Forward pass failed: %d", result);
        return result;
    }

    /* 2. Compute loss using NIMCP loss functions */
    if (ctx->loss_context && ctx->output_buffer) {
        const float* predictions = (const float*)nimcp_tensor_data(ctx->output_buffer);
        const float* target_data = (const float*)nimcp_tensor_data_const(targets);
        size_t n_outputs = nimcp_tensor_numel(ctx->output_buffer);

        nimcp_loss_result_t loss_result_data = {0};
        nimcp_result_t loss_ret = nimcp_loss_forward(
            ctx->loss_context,
            predictions,
            target_data,
            1,  /* batch_size */
            n_outputs,
            &loss_result_data
        );

        if (loss_ret == NIMCP_OK) {
            loss = loss_result_data.loss_value;
        } else {
            NIMCP_LOGGING_WARN("Loss computation failed: %d", loss_ret);
        }

        /* Compute loss gradient (dL/dx) for backward pass */
        if (ctx->loss_gradient) {
            float* loss_grad_data = (float*)nimcp_tensor_data(ctx->loss_gradient);
            nimcp_loss_backward(ctx->loss_context, predictions, target_data,
                               1, n_outputs, loss_grad_data);
        }
    }

    /* 3. Backward pass using adjoint method for LNN */
    if (ctx->gradient_ctx && ctx->loss_gradient) {
        result = lnn_gradient_compute_adjoint(ctx->gradient_ctx, ctx->network, ctx->loss_gradient);
        if (result != 0) {
            NIMCP_LOGGING_WARN("Gradient computation failed: %d", result);
        }

        /* Get gradient norm for statistics */
        grad_norm = lnn_gradient_norm(ctx->gradient_ctx);
    }

    /* 4. Accumulate gradients if needed */
    ctx->current_accumulation++;

    bool should_update = !ctx->accumulate_gradients ||
                         (ctx->current_accumulation >= ctx->accumulation_steps);

    if (should_update) {
        /* 5. Apply gradients via optimizer */
        if (ctx->optimizer && ctx->network) {
            /* Get parameter count and allocate buffers */
            size_t n_params = lnn_network_param_count(ctx->network);
            if (n_params > 0) {
                float* params = (float*)malloc(n_params * sizeof(float));
                float* grads = (float*)malloc(n_params * sizeof(float));

                if (params && grads) {
                    size_t actual_params = 0;
                    size_t actual_grads = 0;

                    /* Get current parameters and gradients */
                    if (lnn_network_get_params(ctx->network, params, &actual_params) == 0 &&
                        lnn_network_get_gradients(ctx->network, grads, &actual_grads) == 0 &&
                        actual_params == actual_grads) {

                        /* Apply optimizer step: params -= lr * grads (or Adam/etc) */
                        nimcp_optimizer_step(ctx->optimizer, params, grads, actual_params);

                        /* Write updated parameters back to network */
                        lnn_network_set_params(ctx->network, params, actual_params);
                    }

                    /* Zero gradients for next iteration */
                    lnn_network_zero_gradients(ctx->network);
                }

                if (params) free(params);
                if (grads) free(grads);
            }

            /* Clear gradient context accumulated state */
            if (ctx->gradient_ctx) {
                lnn_gradient_reset(ctx->gradient_ctx);
            }
        }

        /* Reset accumulation counter */
        ctx->current_accumulation = 0;

        /* 6. Update LR schedule */
        lnn_training_update_lr(ctx);
    }

    /* Update statistics */
    ctx->step_count++;
    ctx->current_loss = loss;
    ctx->stats.step_count = ctx->step_count;
    ctx->stats.current_loss = loss;
    ctx->stats.total_loss += (double)loss;
    ctx->stats.total_gradient_norm += (double)grad_norm;

    if (loss < ctx->stats.min_loss) ctx->stats.min_loss = loss;
    if (loss > ctx->stats.max_loss) ctx->stats.max_loss = loss;
    if (grad_norm < ctx->stats.min_gradient_norm) ctx->stats.min_gradient_norm = grad_norm;
    if (grad_norm > ctx->stats.max_gradient_norm) ctx->stats.max_gradient_norm = grad_norm;

    /* Update average loss and gradient norm */
    ctx->stats.avg_loss = (float)(ctx->stats.total_loss / (double)ctx->stats.step_count);
    ctx->stats.avg_gradient_norm = (float)(ctx->stats.total_gradient_norm / (double)ctx->stats.step_count);

    /* Track timing */
    uint64_t step_end_ms = nimcp_time_get_ms();
    (void)step_start_ms; /* Avoid unused warning if timing not tracked */
    (void)step_end_ms;

    /* 7. Call step callback */
    if (ctx->on_step_complete) {
        ctx->on_step_complete(ctx->callback_user_data, ctx->step_count, loss);
    }

    /* 8. Update training bridges */
    if (ctx->cognitive_training_bridge) {
        cognitive_training_update_metrics(ctx->cognitive_training_bridge,
                                         loss, grad_norm, ctx->current_lr, ctx->step_count);
    }
    if (ctx->training_logic_bridge) {
        training_logic_update_metrics(ctx->training_logic_bridge,
                                     loss, grad_norm, ctx->current_lr, ctx->step_count);
    }
    if (ctx->training_immune_system) {
        training_immune_update_metrics(ctx->training_immune_system,
                                      loss, grad_norm, ctx->current_lr);
    }

    /* Output loss if requested */
    if (loss_out) {
        *loss_out = loss;
    }

    return LNN_SUCCESS;
}

int lnn_training_step_batch(
    lnn_training_ctx_t* ctx,
    const nimcp_tensor_t* inputs,
    const nimcp_tensor_t* targets,
    uint32_t batch_size,
    float* loss_out
) {
    if (!ctx || !inputs || !targets) {
        NIMCP_LOGGING_ERROR("NULL pointer in lnn_training_step_batch");
        return LNN_ERROR_NULL_POINTER;
    }

    if (batch_size == 0) {
        NIMCP_LOGGING_ERROR("Invalid batch size: 0");
        return LNN_ERROR_INVALID_DIMENSION;
    }

    /* Process batch */
    float total_loss = 0.0f;

    /* TODO: Implement parallel batch processing */
    /* For now, sequential processing */
    for (uint32_t i = 0; i < batch_size; i++) {
        float sample_loss = 0.0f;
        /* TODO: Extract sample from batch, call lnn_training_step() */
        total_loss += sample_loss;
    }

    float avg_loss = total_loss / (float)batch_size;

    if (loss_out) {
        *loss_out = avg_loss;
    }

    return LNN_SUCCESS;
}

int lnn_training_epoch(
    lnn_training_ctx_t* ctx,
    const nimcp_tensor_t* dataset_inputs,
    const nimcp_tensor_t* dataset_targets,
    uint32_t n_samples,
    uint32_t batch_size,
    float* avg_loss_out
) {
    if (!ctx || !dataset_inputs || !dataset_targets) {
        NIMCP_LOGGING_ERROR("NULL pointer in lnn_training_epoch");
        return LNN_ERROR_NULL_POINTER;
    }

    if (n_samples == 0 || batch_size == 0) {
        NIMCP_LOGGING_ERROR("Invalid n_samples or batch_size");
        return LNN_ERROR_INVALID_DIMENSION;
    }

    float epoch_loss = 0.0f;
    uint32_t num_batches = (n_samples + batch_size - 1) / batch_size;

    /* Process each batch */
    for (uint32_t batch_idx = 0; batch_idx < num_batches; batch_idx++) {
        uint32_t batch_start = batch_idx * batch_size;
        uint32_t batch_end = batch_start + batch_size;
        if (batch_end > n_samples) batch_end = n_samples;
        uint32_t actual_batch_size = batch_end - batch_start;

        float batch_loss = 0.0f;
        /* TODO: Extract batch, call lnn_training_step_batch() */

        epoch_loss += batch_loss;
    }

    float avg_loss = epoch_loss / (float)num_batches;

    /* Update epoch statistics */
    ctx->epoch_count++;
    ctx->stats.epoch_count = ctx->epoch_count;

    /* Call epoch callback */
    if (ctx->on_epoch_complete) {
        ctx->on_epoch_complete(ctx->callback_user_data, ctx->epoch_count, avg_loss);
    }

    if (avg_loss_out) {
        *avg_loss_out = avg_loss;
    }

    return LNN_SUCCESS;
}

/*=============================================================================
 * Integration Connection Functions
 *===========================================================================*/

int lnn_training_connect_cognitive(
    lnn_training_ctx_t* ctx,
    cognitive_training_bridge_t* cognitive_bridge
) {
    if (!ctx) {
        NIMCP_LOGGING_ERROR("NULL context in lnn_training_connect_cognitive");
        return LNN_ERROR_NULL_POINTER;
    }

    ctx->cognitive_training_bridge = cognitive_bridge;
    ctx->stats.cognitive_bridge_connected = (cognitive_bridge != NULL);

    if (cognitive_bridge) {
        NIMCP_LOGGING_INFO("Connected LNN training to cognitive bridge");
    } else {
        NIMCP_LOGGING_INFO("Disconnected LNN training from cognitive bridge");
    }

    return LNN_SUCCESS;
}

int lnn_training_connect_logic(
    lnn_training_ctx_t* ctx,
    training_logic_bridge_t* logic_bridge
) {
    if (!ctx) {
        NIMCP_LOGGING_ERROR("NULL context in lnn_training_connect_logic");
        return LNN_ERROR_NULL_POINTER;
    }

    ctx->training_logic_bridge = logic_bridge;
    ctx->stats.logic_bridge_connected = (logic_bridge != NULL);

    if (logic_bridge) {
        NIMCP_LOGGING_INFO("Connected LNN training to logic bridge");
    } else {
        NIMCP_LOGGING_INFO("Disconnected LNN training from logic bridge");
    }

    return LNN_SUCCESS;
}

int lnn_training_connect_immune(
    lnn_training_ctx_t* ctx,
    training_immune_system_t* immune_system
) {
    if (!ctx) {
        NIMCP_LOGGING_ERROR("NULL context in lnn_training_connect_immune");
        return LNN_ERROR_NULL_POINTER;
    }

    ctx->training_immune_system = immune_system;
    ctx->stats.immune_system_connected = (immune_system != NULL);

    if (immune_system) {
        NIMCP_LOGGING_INFO("Connected LNN training to immune system");
    } else {
        NIMCP_LOGGING_INFO("Disconnected LNN training from immune system");
    }

    return LNN_SUCCESS;
}

int lnn_training_connect_plasticity(
    lnn_training_ctx_t* ctx,
    training_plasticity_bridge_t* plasticity_bridge
) {
    if (!ctx) {
        NIMCP_LOGGING_ERROR("NULL context in lnn_training_connect_plasticity");
        return LNN_ERROR_NULL_POINTER;
    }

    ctx->training_plasticity_bridge = plasticity_bridge;
    ctx->stats.plasticity_bridge_connected = (plasticity_bridge != NULL);

    if (plasticity_bridge) {
        NIMCP_LOGGING_INFO("Connected LNN training to plasticity bridge");
    } else {
        NIMCP_LOGGING_INFO("Disconnected LNN training from plasticity bridge");
    }

    return LNN_SUCCESS;
}

/*=============================================================================
 * Learning Rate Scheduling Functions
 *===========================================================================*/

int lnn_training_set_lr_schedule(
    lnn_training_ctx_t* ctx,
    lnn_lr_schedule_t schedule,
    float* params,
    uint32_t n_params
) {
    if (!ctx) {
        NIMCP_LOGGING_ERROR("NULL context in lnn_training_set_lr_schedule");
        return LNN_ERROR_NULL_POINTER;
    }

    if (n_params > LNN_TRAINING_MAX_SCHEDULE_PARAMS) {
        NIMCP_LOGGING_ERROR("Too many schedule params: %u > %u",
                           n_params, LNN_TRAINING_MAX_SCHEDULE_PARAMS);
        return LNN_ERROR_INVALID_CONFIG;
    }

    /* Free old params */
    if (ctx->lr_schedule_params) {
        free(ctx->lr_schedule_params);
        ctx->lr_schedule_params = NULL;
    }

    /* Set new schedule */
    ctx->lr_schedule = schedule;
    ctx->n_schedule_params = n_params;

    /* Copy params if provided */
    if (n_params > 0 && params) {
        ctx->lr_schedule_params = (float*)malloc(n_params * sizeof(float));
        if (!ctx->lr_schedule_params) {
            NIMCP_LOGGING_ERROR("Failed to allocate schedule params");
            return LNN_ERROR_OUT_OF_MEMORY;
        }
        memcpy(ctx->lr_schedule_params, params, n_params * sizeof(float));
    }

    /* Reset schedule state */
    ctx->last_lr_update_step = ctx->step_count;
    ctx->plateau_best_metric = FLT_MAX;
    ctx->plateau_patience_count = 0;

    NIMCP_LOGGING_INFO("Set LR schedule to %s", lnn_lr_schedule_name(schedule));

    return LNN_SUCCESS;
}

int lnn_training_update_lr(lnn_training_ctx_t* ctx) {
    if (!ctx) {
        NIMCP_LOGGING_ERROR("NULL context in lnn_training_update_lr");
        return LNN_ERROR_NULL_POINTER;
    }

    float old_lr = ctx->current_lr;
    float new_lr = old_lr;

    /* Compute new LR based on schedule */
    switch (ctx->lr_schedule) {
    case LNN_LR_SCHEDULE_CONSTANT:
        /* No change */
        break;

    case LNN_LR_SCHEDULE_STEP:
        if (ctx->n_schedule_params >= 2) {
            uint32_t step_size = (uint32_t)ctx->lr_schedule_params[0];
            float gamma = ctx->lr_schedule_params[1];
            new_lr = compute_step_schedule_lr(ctx->config.learning_rate,
                                               ctx->step_count, step_size, gamma);
        }
        break;

    case LNN_LR_SCHEDULE_EXPONENTIAL:
        if (ctx->n_schedule_params >= 1) {
            float gamma = ctx->lr_schedule_params[0];
            new_lr = compute_exponential_schedule_lr(ctx->config.learning_rate,
                                                      ctx->step_count, gamma);
        }
        break;

    case LNN_LR_SCHEDULE_COSINE:
        if (ctx->n_schedule_params >= 1) {
            uint64_t T_max = (uint64_t)ctx->lr_schedule_params[0];
            new_lr = compute_cosine_schedule_lr(ctx->config.learning_rate,
                                                 ctx->step_count, T_max);
        }
        break;

    case LNN_LR_SCHEDULE_WARMUP_COSINE:
        if (ctx->n_schedule_params >= 2) {
            uint64_t warmup_steps = (uint64_t)ctx->lr_schedule_params[0];
            uint64_t T_max = (uint64_t)ctx->lr_schedule_params[1];
            new_lr = compute_warmup_cosine_lr(ctx->config.learning_rate,
                                               ctx->step_count, warmup_steps, T_max);
        }
        break;

    case LNN_LR_SCHEDULE_REDUCE_ON_PLATEAU:
        if (ctx->n_schedule_params >= 3) {
            float factor = ctx->lr_schedule_params[0];
            uint32_t patience = (uint32_t)ctx->lr_schedule_params[1];
            float threshold = ctx->lr_schedule_params[2];

            /* Check if current loss is better than best by threshold */
            if (ctx->current_loss < ctx->plateau_best_metric - threshold) {
                ctx->plateau_best_metric = ctx->current_loss;
                ctx->plateau_patience_count = 0;
            } else {
                ctx->plateau_patience_count++;

                /* Reduce LR if plateau detected */
                if (ctx->plateau_patience_count >= patience) {
                    new_lr = ctx->current_lr * factor;
                    ctx->plateau_patience_count = 0;
                    NIMCP_LOGGING_INFO("Plateau detected, reducing LR by factor %.2f", factor);
                }
            }
        }
        break;

    default:
        NIMCP_LOGGING_WARN("Unknown LR schedule type: %d", ctx->lr_schedule);
        break;
    }

    /* Update LR if changed */
    if (fabsf(new_lr - old_lr) > 1e-9f) {
        ctx->current_lr = new_lr;
        ctx->stats.current_lr = new_lr;
        ctx->stats.lr_updates++;

        /* Update min/max */
        if (new_lr < ctx->stats.min_lr) ctx->stats.min_lr = new_lr;
        if (new_lr > ctx->stats.max_lr) ctx->stats.max_lr = new_lr;

        /* Update optimizer LR */
        nimcp_optimizer_set_lr(ctx->optimizer, new_lr);

        /* Call LR change callback */
        if (ctx->on_lr_change) {
            ctx->on_lr_change(ctx->callback_user_data, old_lr, new_lr);
        }

        ctx->last_lr_update_step = ctx->step_count;
    }

    return LNN_SUCCESS;
}

float lnn_training_get_lr(const lnn_training_ctx_t* ctx) {
    if (!ctx) return 0.0f;
    return ctx->current_lr;
}

void lnn_training_set_lr(lnn_training_ctx_t* ctx, float lr) {
    if (!ctx) return;

    float old_lr = ctx->current_lr;
    ctx->current_lr = lr;
    ctx->stats.current_lr = lr;

    /* Update optimizer */
    nimcp_optimizer_set_lr(ctx->optimizer, lr);

    /* Call callback */
    if (ctx->on_lr_change) {
        ctx->on_lr_change(ctx->callback_user_data, old_lr, lr);
    }
}

/*=============================================================================
 * Callback Functions
 *===========================================================================*/

void lnn_training_set_step_callback(
    lnn_training_ctx_t* ctx,
    void (*callback)(void* user_data, uint64_t step, float loss),
    void* user_data
) {
    if (!ctx) return;
    ctx->on_step_complete = callback;
    ctx->callback_user_data = user_data;
}

void lnn_training_set_epoch_callback(
    lnn_training_ctx_t* ctx,
    void (*callback)(void* user_data, uint64_t epoch, float avg_loss),
    void* user_data
) {
    if (!ctx) return;
    ctx->on_epoch_complete = callback;
    ctx->callback_user_data = user_data;
}

void lnn_training_set_lr_callback(
    lnn_training_ctx_t* ctx,
    void (*callback)(void* user_data, float old_lr, float new_lr),
    void* user_data
) {
    if (!ctx) return;
    ctx->on_lr_change = callback;
    ctx->callback_user_data = user_data;
}

/*=============================================================================
 * Statistics Functions
 *===========================================================================*/

uint64_t lnn_training_get_step_count(const lnn_training_ctx_t* ctx) {
    if (!ctx) return 0;
    return ctx->step_count;
}

uint64_t lnn_training_get_epoch_count(const lnn_training_ctx_t* ctx) {
    if (!ctx) return 0;
    return ctx->epoch_count;
}

float lnn_training_get_current_loss(const lnn_training_ctx_t* ctx) {
    if (!ctx) return 0.0f;
    return ctx->current_loss;
}

int lnn_training_get_stats(
    const lnn_training_ctx_t* ctx,
    lnn_training_stats_t* stats
) {
    if (!ctx || !stats) {
        NIMCP_LOGGING_ERROR("NULL pointer in lnn_training_get_stats");
        return LNN_ERROR_NULL_POINTER;
    }

    memcpy(stats, &ctx->stats, sizeof(lnn_training_stats_t));
    return LNN_SUCCESS;
}

int lnn_training_reset_stats(lnn_training_ctx_t* ctx) {
    if (!ctx) {
        NIMCP_LOGGING_ERROR("NULL context in lnn_training_reset_stats");
        return LNN_ERROR_NULL_POINTER;
    }

    /* Preserve connection status */
    bool cognitive_connected = ctx->stats.cognitive_bridge_connected;
    bool logic_connected = ctx->stats.logic_bridge_connected;
    bool immune_connected = ctx->stats.immune_system_connected;
    bool plasticity_connected = ctx->stats.plasticity_bridge_connected;

    /* Reset all stats */
    memset(&ctx->stats, 0, sizeof(lnn_training_stats_t));

    /* Restore connection status */
    ctx->stats.cognitive_bridge_connected = cognitive_connected;
    ctx->stats.logic_bridge_connected = logic_connected;
    ctx->stats.immune_system_connected = immune_connected;
    ctx->stats.plasticity_bridge_connected = plasticity_connected;

    /* Reset min/max values */
    ctx->stats.min_loss = FLT_MAX;
    ctx->stats.max_loss = -FLT_MAX;
    ctx->stats.min_gradient_norm = FLT_MAX;
    ctx->stats.max_gradient_norm = -FLT_MAX;
    ctx->stats.min_tau_network = FLT_MAX;
    ctx->stats.max_tau_network = -FLT_MAX;
    ctx->stats.current_lr = ctx->current_lr;
    ctx->stats.min_lr = ctx->current_lr;
    ctx->stats.max_lr = ctx->current_lr;

    NIMCP_LOGGING_INFO("Reset LNN training statistics");

    return LNN_SUCCESS;
}

/*=============================================================================
 * Utility Functions
 *===========================================================================*/

const char* lnn_lr_schedule_name(lnn_lr_schedule_t schedule) {
    switch (schedule) {
    case LNN_LR_SCHEDULE_CONSTANT:        return "CONSTANT";
    case LNN_LR_SCHEDULE_STEP:            return "STEP";
    case LNN_LR_SCHEDULE_EXPONENTIAL:     return "EXPONENTIAL";
    case LNN_LR_SCHEDULE_COSINE:          return "COSINE";
    case LNN_LR_SCHEDULE_WARMUP_COSINE:   return "WARMUP_COSINE";
    case LNN_LR_SCHEDULE_REDUCE_ON_PLATEAU: return "REDUCE_ON_PLATEAU";
    default:                              return "UNKNOWN";
    }
}

int lnn_training_validate_config(const lnn_training_config_t* config) {
    if (!config) {
        NIMCP_LOGGING_ERROR("NULL config pointer");
        return LNN_ERROR_NULL_POINTER;
    }

    /* Validate learning rate */
    if (config->learning_rate <= 0.0f || config->learning_rate > 1.0f) {
        NIMCP_LOGGING_ERROR("Invalid learning rate: %f", config->learning_rate);
        return LNN_ERROR_INVALID_CONFIG;
    }

    /* Validate weight decay */
    if (config->weight_decay < 0.0f || config->weight_decay > 1.0f) {
        NIMCP_LOGGING_ERROR("Invalid weight decay: %f", config->weight_decay);
        return LNN_ERROR_INVALID_CONFIG;
    }

    /* Validate Adam parameters */
    if (config->optimizer_type == NIMCP_OPTIMIZER_ADAM) {
        if (config->beta1 < 0.0f || config->beta1 >= 1.0f) {
            NIMCP_LOGGING_ERROR("Invalid beta1: %f", config->beta1);
            return LNN_ERROR_INVALID_CONFIG;
        }
        if (config->beta2 < 0.0f || config->beta2 >= 1.0f) {
            NIMCP_LOGGING_ERROR("Invalid beta2: %f", config->beta2);
            return LNN_ERROR_INVALID_CONFIG;
        }
        if (config->epsilon <= 0.0f) {
            NIMCP_LOGGING_ERROR("Invalid epsilon: %f", config->epsilon);
            return LNN_ERROR_INVALID_CONFIG;
        }
    }

    /* Validate gradient clipping */
    if (config->gradient_clip_norm < 0.0f) {
        NIMCP_LOGGING_ERROR("Invalid gradient clip norm: %f", config->gradient_clip_norm);
        return LNN_ERROR_INVALID_CONFIG;
    }

    /* Validate accumulation steps */
    if (config->accumulation_steps == 0) {
        NIMCP_LOGGING_ERROR("Invalid accumulation steps: 0");
        return LNN_ERROR_INVALID_CONFIG;
    }

    /* Validate BPTT truncation */
    if (config->bptt_truncation == 0) {
        NIMCP_LOGGING_ERROR("Invalid BPTT truncation: 0");
        return LNN_ERROR_INVALID_CONFIG;
    }

    /* Validate schedule params count */
    if (config->n_schedule_params > LNN_TRAINING_MAX_SCHEDULE_PARAMS) {
        NIMCP_LOGGING_ERROR("Too many schedule params: %u", config->n_schedule_params);
        return LNN_ERROR_INVALID_CONFIG;
    }

    return LNN_SUCCESS;
}
