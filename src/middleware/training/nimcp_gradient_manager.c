/**
 * @file nimcp_gradient_manager.c
 * @brief Gradient Management Module Implementation
 *
 * Implements:
 * - Gradient accumulation for large batch training
 * - Gradient scaling for mixed precision
 * - Health checking (NaN/Inf detection)
 * - Gradient statistics and monitoring
 *
 * @note Part of Phase TM-6: Gradient Management
 * @version 1.0.0
 */

#include "middleware/training/nimcp_gradient_manager.h"
#include "security/nimcp_security_integration.h"
#include "utils/memory/nimcp_memory.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include "utils/logging/nimcp_logging.h"
#include "security/nimcp_blood_brain_barrier.h"
#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"

#define GRADMGR_MODULE_NAME "GradientManager"
#define LOG_MODULE "gradient_manager"

/* ============================================================================
 * Internal Context Structure
 * ============================================================================ */

struct nimcp_gradient_manager_ctx {
    nimcp_gradient_manager_config_t config;

    /* Accumulation state */
    float* accum_buffer;          /**< Accumulated gradients */
    size_t accum_buffer_size;     /**< Buffer size */
    uint32_t accum_step;          /**< Current accumulation step */
    bool accum_initialized;       /**< Buffer initialized */

    /* Scaling state */
    float current_scale;          /**< Current scale factor */
    uint32_t steps_since_growth;  /**< Steps since last scale growth */
    bool found_inf;               /**< Infinity found in current step */

    /* Statistics */
    nimcp_grad_stats_t stats;

    bool initialized;

    /* Security registration */
    nimcp_sec_integration_t* security_ctx;
    bool security_registered;
    uint32_t security_module_id;
};

/* ============================================================================
 * Default Configurations
 * ============================================================================ */

nimcp_grad_accum_config_t nimcp_grad_accum_default_config(uint32_t accumulation_steps) {
    nimcp_grad_accum_config_t config = {
        .accumulation_steps = (accumulation_steps > 0 && accumulation_steps <= NIMCP_GRAD_MAX_ACCUM_STEPS)
                               ? accumulation_steps : 1,
        .mode = NIMCP_GRAD_ACCUM_SUM,
        .sync_on_step = false
    };
    return config;
}

nimcp_grad_scale_config_t nimcp_grad_scale_default_config(float initial_scale) {
    nimcp_grad_scale_config_t config = {
        .strategy = NIMCP_GRAD_SCALE_DYNAMIC,
        .initial_scale = initial_scale > 0.0f ? initial_scale : 65536.0f,
        .min_scale = NIMCP_GRAD_MIN_SCALE,
        .max_scale = NIMCP_GRAD_MAX_SCALE,
        .backoff_factor = NIMCP_GRAD_BACKOFF_FACTOR,
        .growth_factor = NIMCP_GRAD_GROWTH_FACTOR,
        .growth_interval = NIMCP_GRAD_GROWTH_INTERVAL
    };
    return config;
}

nimcp_gradient_manager_config_t nimcp_gradient_manager_default_config(void) {
    nimcp_gradient_manager_config_t config;
    memset(&config, 0, sizeof(config));

    config.accumulation = nimcp_grad_accum_default_config(1);
    config.use_accumulation = false;

    config.scaling = nimcp_grad_scale_default_config(65536.0f);
    config.use_scaling = false;

    config.check_nan_inf = true;
    config.skip_nan_gradients = true;
    config.replace_inf = true;
    config.max_grad_value = 1e6f;

    config.preallocate_buffers = false;
    config.max_buffer_size = 0;

    config.verbose = false;
    config.track_statistics = true;

    return config;
}

/* ============================================================================
 * Lifecycle Functions
 * ============================================================================ */

nimcp_gradient_manager_ctx_t* nimcp_gradient_manager_create(
    const nimcp_gradient_manager_config_t* config
) {
    if (!config) {
        return NULL;
    }

    if (nimcp_gradient_manager_validate_config(config) != NIMCP_SUCCESS) {
        return NULL;
    }

    nimcp_gradient_manager_ctx_t* ctx = (nimcp_gradient_manager_ctx_t*)nimcp_calloc(
        1, sizeof(nimcp_gradient_manager_ctx_t)
    );
    if (!ctx) {
        return NULL;
    }

    memcpy(&ctx->config, config, sizeof(nimcp_gradient_manager_config_t));

    /* Initialize scaling */
    ctx->current_scale = config->use_scaling ? config->scaling.initial_scale : 1.0f;
    ctx->steps_since_growth = 0;
    ctx->found_inf = false;

    /* Initialize accumulation buffer */
    ctx->accum_buffer = NULL;
    ctx->accum_buffer_size = 0;
    ctx->accum_step = 0;
    ctx->accum_initialized = false;

    if (config->preallocate_buffers && config->max_buffer_size > 0) {
        ctx->accum_buffer = (float*)nimcp_calloc(config->max_buffer_size, sizeof(float));
        if (ctx->accum_buffer) {
            ctx->accum_buffer_size = config->max_buffer_size;
            ctx->accum_initialized = true;
        }
    }

    /* Initialize statistics */
    memset(&ctx->stats, 0, sizeof(nimcp_grad_stats_t));
    ctx->stats.min_grad_norm = FLT_MAX;
    ctx->stats.max_grad_norm = 0.0f;
    ctx->stats.current_scale = ctx->current_scale;

    ctx->initialized = true;

    /* Register with security module if available */
    ctx->security_ctx = config->security_ctx;
    ctx->security_registered = false;
    if (ctx->security_ctx) {
        nimcp_result_t err = nimcp_sec_register_module(
            ctx->security_ctx,
            GRADMGR_MODULE_NAME,
            NIMCP_SEC_CAT_MIDDLEWARE,
            &ctx->security_module_id
        );
        if (err == NIMCP_SUCCESS) {
            ctx->security_registered = true;
        }
    }

    return ctx;
}

void nimcp_gradient_manager_destroy(nimcp_gradient_manager_ctx_t* ctx) {
    if (!ctx) {
        return;
    }

    /* Unregister from security module */
    if (ctx->security_registered && ctx->security_ctx) {
        nimcp_sec_unregister_module(ctx->security_ctx, ctx->security_module_id);
        ctx->security_registered = false;
    }

    if (ctx->accum_buffer) {
        nimcp_free(ctx->accum_buffer);
    }

    nimcp_free(ctx);
}

/* ============================================================================
 * Gradient Accumulation
 * ============================================================================ */

nimcp_result_t nimcp_gradient_accumulate(
    nimcp_gradient_manager_ctx_t* ctx,
    const float* gradients,
    size_t num_gradients
) {
    if (!ctx || !gradients || num_gradients == 0) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    if (!ctx->config.use_accumulation) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    /* Allocate or resize buffer if needed */
    if (!ctx->accum_initialized || ctx->accum_buffer_size < num_gradients) {
        if (ctx->accum_buffer) {
            nimcp_free(ctx->accum_buffer);
        }
        ctx->accum_buffer = (float*)nimcp_calloc(num_gradients, sizeof(float));
        if (!ctx->accum_buffer) {
            return NIMCP_ERROR_MEMORY;
        }
        ctx->accum_buffer_size = num_gradients;
        ctx->accum_initialized = true;
        ctx->accum_step = 0;
    }

    /* Reset buffer at start of accumulation */
    if (ctx->accum_step == 0) {
        memset(ctx->accum_buffer, 0, num_gradients * sizeof(float));
    }

    /* Check gradient health */
    if (ctx->config.check_nan_inf) {
        nimcp_grad_health_t health = nimcp_gradient_check_health(gradients, num_gradients);
        if (health == NIMCP_GRAD_HAS_NAN || health == NIMCP_GRAD_HAS_INF) {
            if (ctx->config.skip_nan_gradients) {
                ctx->stats.skipped_steps++;
                if (health == NIMCP_GRAD_HAS_NAN) ctx->stats.nan_count++;
                if (health == NIMCP_GRAD_HAS_INF) ctx->stats.inf_count++;
                return NIMCP_SUCCESS;  /* Skip this accumulation step */
            }
        }
    }

    /* Accumulate */
    for (size_t i = 0; i < num_gradients; i++) {
        ctx->accum_buffer[i] += gradients[i];
    }

    ctx->accum_step++;
    ctx->stats.total_accum_steps++;

    return NIMCP_SUCCESS;
}

bool nimcp_gradient_accum_ready(const nimcp_gradient_manager_ctx_t* ctx) {
    if (!ctx) {
        return false;  /* Invalid context = not ready */
    }
    if (!ctx->config.use_accumulation) {
        return true;  /* No accumulation = always ready */
    }
    return ctx->accum_step >= ctx->config.accumulation.accumulation_steps;
}

nimcp_result_t nimcp_gradient_get_accumulated(
    nimcp_gradient_manager_ctx_t* ctx,
    float* output,
    size_t num_gradients
) {
    if (!ctx || !output || num_gradients == 0) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    if (!ctx->config.use_accumulation) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    if (!ctx->accum_initialized || ctx->accum_buffer_size < num_gradients) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    /* Copy accumulated gradients */
    memcpy(output, ctx->accum_buffer, num_gradients * sizeof(float));

    /* Apply averaging if mode is MEAN */
    if (ctx->config.accumulation.mode == NIMCP_GRAD_ACCUM_MEAN && ctx->accum_step > 0) {
        float divisor = (float)ctx->accum_step;
        for (size_t i = 0; i < num_gradients; i++) {
            output[i] /= divisor;
        }
    }

    ctx->stats.total_steps++;

    return NIMCP_SUCCESS;
}

void nimcp_gradient_reset_accum(nimcp_gradient_manager_ctx_t* ctx) {
    if (!ctx) {
        return;
    }

    ctx->accum_step = 0;
    if (ctx->accum_buffer && ctx->accum_buffer_size > 0) {
        memset(ctx->accum_buffer, 0, ctx->accum_buffer_size * sizeof(float));
    }
}

uint32_t nimcp_gradient_get_accum_step(const nimcp_gradient_manager_ctx_t* ctx) {
    return ctx ? ctx->accum_step : 0;
}

/* ============================================================================
 * Gradient Scaling
 * ============================================================================ */

float nimcp_gradient_scale(
    nimcp_gradient_manager_ctx_t* ctx,
    float* gradients,
    size_t num_gradients
) {
    if (!ctx || !gradients || num_gradients == 0) {
        return 1.0f;
    }

    if (!ctx->config.use_scaling || ctx->config.scaling.strategy == NIMCP_GRAD_SCALE_NONE) {
        return 1.0f;
    }

    float scale = ctx->current_scale;

    for (size_t i = 0; i < num_gradients; i++) {
        gradients[i] *= scale;
    }

    return scale;
}

float nimcp_gradient_unscale(
    nimcp_gradient_manager_ctx_t* ctx,
    float* gradients,
    size_t num_gradients
) {
    if (!ctx || !gradients || num_gradients == 0) {
        return 1.0f;
    }

    if (!ctx->config.use_scaling || ctx->config.scaling.strategy == NIMCP_GRAD_SCALE_NONE) {
        return 1.0f;
    }

    float inv_scale = 1.0f / ctx->current_scale;

    for (size_t i = 0; i < num_gradients; i++) {
        gradients[i] *= inv_scale;
    }

    /* Check for overflow after unscaling */
    nimcp_grad_health_t health = nimcp_gradient_check_health(gradients, num_gradients);
    if (health == NIMCP_GRAD_HAS_INF || health == NIMCP_GRAD_HAS_NAN) {
        ctx->found_inf = true;
    }

    return inv_scale;
}

void nimcp_gradient_update_scale(
    nimcp_gradient_manager_ctx_t* ctx,
    nimcp_grad_health_t health
) {
    if (!ctx || !ctx->config.use_scaling) {
        return;
    }

    if (ctx->config.scaling.strategy != NIMCP_GRAD_SCALE_DYNAMIC) {
        return;
    }

    const nimcp_grad_scale_config_t* cfg = &ctx->config.scaling;

    if (health == NIMCP_GRAD_HAS_INF || health == NIMCP_GRAD_HAS_NAN || ctx->found_inf) {
        /* Reduce scale on overflow */
        ctx->current_scale *= cfg->backoff_factor;
        if (ctx->current_scale < cfg->min_scale) {
            ctx->current_scale = cfg->min_scale;
        }
        ctx->steps_since_growth = 0;
        ctx->found_inf = false;
        ctx->stats.scale_decreases++;
        ctx->stats.overflow_count++;
    } else {
        /* Try to grow scale */
        ctx->steps_since_growth++;
        if (ctx->steps_since_growth >= cfg->growth_interval) {
            ctx->current_scale *= cfg->growth_factor;
            if (ctx->current_scale > cfg->max_scale) {
                ctx->current_scale = cfg->max_scale;
            }
            ctx->steps_since_growth = 0;
            ctx->stats.scale_increases++;
        }
    }

    ctx->stats.current_scale = ctx->current_scale;
}

float nimcp_gradient_get_scale(const nimcp_gradient_manager_ctx_t* ctx) {
    return ctx ? ctx->current_scale : 1.0f;
}

nimcp_result_t nimcp_gradient_set_scale(
    nimcp_gradient_manager_ctx_t* ctx,
    float scale
) {
    if (!ctx) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    if (scale < NIMCP_GRAD_MIN_SCALE || scale > NIMCP_GRAD_MAX_SCALE) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    ctx->current_scale = scale;
    ctx->stats.current_scale = scale;

    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Health Checking
 * ============================================================================ */

nimcp_grad_health_t nimcp_gradient_check_health(
    const float* gradients,
    size_t num_gradients
) {
    if (!gradients || num_gradients == 0) {
        return NIMCP_GRAD_HEALTHY;
    }

    bool all_zero = true;

    for (size_t i = 0; i < num_gradients; i++) {
        if (isnan(gradients[i])) {
            return NIMCP_GRAD_HAS_NAN;
        }
        if (isinf(gradients[i])) {
            return NIMCP_GRAD_HAS_INF;
        }
        if (gradients[i] != 0.0f) {
            all_zero = false;
        }
    }

    if (all_zero) {
        return NIMCP_GRAD_HAS_ZERO;
    }

    return NIMCP_GRAD_HEALTHY;
}

nimcp_grad_health_t nimcp_gradient_check_health_ctx(
    nimcp_gradient_manager_ctx_t* ctx,
    const float* gradients,
    size_t num_gradients
) {
    nimcp_grad_health_t health = nimcp_gradient_check_health(gradients, num_gradients);

    if (ctx && ctx->config.track_statistics) {
        if (health == NIMCP_GRAD_HAS_NAN) ctx->stats.nan_count++;
        if (health == NIMCP_GRAD_HAS_INF) ctx->stats.inf_count++;
    }

    return health;
}

uint64_t nimcp_gradient_sanitize(
    float* gradients,
    size_t num_gradients,
    float replace_value
) {
    if (!gradients || num_gradients == 0) {
        return 0;
    }

    uint64_t replaced = 0;

    for (size_t i = 0; i < num_gradients; i++) {
        if (isnan(gradients[i]) || isinf(gradients[i])) {
            gradients[i] = replace_value;
            replaced++;
        }
    }

    return replaced;
}

/* ============================================================================
 * Gradient Statistics
 * ============================================================================ */

float nimcp_gradient_l2_norm(const float* gradients, size_t num_gradients) {
    if (!gradients || num_gradients == 0) {
        return 0.0f;
    }

    float sum_sq = 0.0f;
    for (size_t i = 0; i < num_gradients; i++) {
        sum_sq += gradients[i] * gradients[i];
    }

    return sqrtf(sum_sq);
}

float nimcp_gradient_l1_norm(const float* gradients, size_t num_gradients) {
    if (!gradients || num_gradients == 0) {
        return 0.0f;
    }

    float sum = 0.0f;
    for (size_t i = 0; i < num_gradients; i++) {
        sum += fabsf(gradients[i]);
    }

    return sum;
}

float nimcp_gradient_max_norm(const float* gradients, size_t num_gradients) {
    if (!gradients || num_gradients == 0) {
        return 0.0f;
    }

    float max_abs = 0.0f;
    for (size_t i = 0; i < num_gradients; i++) {
        float abs_val = fabsf(gradients[i]);
        if (abs_val > max_abs) {
            max_abs = abs_val;
        }
    }

    return max_abs;
}

nimcp_result_t nimcp_gradient_manager_get_stats(
    const nimcp_gradient_manager_ctx_t* ctx,
    nimcp_grad_stats_t* stats
) {
    if (!ctx || !stats) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    memcpy(stats, &ctx->stats, sizeof(nimcp_grad_stats_t));

    /* Compute average */
    if (ctx->stats.total_steps > 0) {
        stats->avg_grad_norm = (float)(ctx->stats.sum_grad_norm / (double)ctx->stats.total_steps);
    }

    return NIMCP_SUCCESS;
}

void nimcp_gradient_manager_reset_stats(nimcp_gradient_manager_ctx_t* ctx) {
    if (!ctx) {
        return;
    }

    memset(&ctx->stats, 0, sizeof(nimcp_grad_stats_t));
    ctx->stats.min_grad_norm = FLT_MAX;
    ctx->stats.current_scale = ctx->current_scale;
}

/* ============================================================================
 * Utility Functions
 * ============================================================================ */

const char* nimcp_grad_accum_mode_name(nimcp_grad_accum_mode_t mode) {
    static const char* names[] = {
        "Sum",
        "Mean"
    };

    if (mode >= NIMCP_GRAD_ACCUM_MODE_COUNT) {
        return "Unknown";
    }
    return names[mode];
}

const char* nimcp_grad_scale_strategy_name(nimcp_grad_scale_strategy_t strategy) {
    static const char* names[] = {
        "None",
        "Fixed",
        "Dynamic"
    };

    if (strategy >= NIMCP_GRAD_SCALE_STRATEGY_COUNT) {
        return "Unknown";
    }
    return names[strategy];
}

const char* nimcp_grad_health_name(nimcp_grad_health_t health) {
    static const char* names[] = {
        "Healthy",
        "Has NaN",
        "Has Inf",
        "All Zero",
        "Overflow",
        "Underflow"
    };

    if (health > NIMCP_GRAD_UNDERFLOW) {
        return "Unknown";
    }
    return names[health];
}

nimcp_result_t nimcp_gradient_manager_validate_config(
    const nimcp_gradient_manager_config_t* config
) {
    if (!config) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    /* Validate accumulation config */
    if (config->use_accumulation) {
        if (config->accumulation.accumulation_steps == 0 ||
            config->accumulation.accumulation_steps > NIMCP_GRAD_MAX_ACCUM_STEPS) {
            return NIMCP_ERROR_INVALID_PARAM;
        }
        if (config->accumulation.mode >= NIMCP_GRAD_ACCUM_MODE_COUNT) {
            return NIMCP_ERROR_INVALID_PARAM;
        }
    }

    /* Validate scaling config */
    if (config->use_scaling) {
        if (config->scaling.strategy >= NIMCP_GRAD_SCALE_STRATEGY_COUNT) {
            return NIMCP_ERROR_INVALID_PARAM;
        }
        if (config->scaling.strategy != NIMCP_GRAD_SCALE_NONE) {
            if (config->scaling.initial_scale <= 0.0f) {
                return NIMCP_ERROR_INVALID_PARAM;
            }
            if (config->scaling.min_scale <= 0.0f ||
                config->scaling.max_scale <= config->scaling.min_scale) {
                return NIMCP_ERROR_INVALID_PARAM;
            }
            if (config->scaling.backoff_factor <= 0.0f ||
                config->scaling.backoff_factor >= 1.0f) {
                return NIMCP_ERROR_INVALID_PARAM;
            }
            if (config->scaling.growth_factor <= 1.0f) {
                return NIMCP_ERROR_INVALID_PARAM;
            }
        }
    }

    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Distributed Training Support
 * ============================================================================ */

void nimcp_gradient_prepare_allreduce(
    float* gradients,
    size_t num_gradients,
    uint32_t num_workers
) {
    /* For all-reduce averaging, no pre-processing needed */
    /* This is a placeholder for potential future optimizations */
    (void)gradients;
    (void)num_gradients;
    (void)num_workers;
}

void nimcp_gradient_finalize_allreduce(
    float* gradients,
    size_t num_gradients,
    uint32_t num_workers
) {
    if (!gradients || num_gradients == 0 || num_workers == 0) {
        return;
    }

    /* After all-reduce sum, divide by number of workers for average */
    float divisor = (float)num_workers;
    for (size_t i = 0; i < num_gradients; i++) {
        gradients[i] /= divisor;
    }
}
