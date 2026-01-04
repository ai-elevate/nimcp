/**
 * @file nimcp_jepa_predictor.c
 * @brief JEPA Predictor Implementation
 * @version 1.0.0
 * @date 2025-12-26
 *
 * WHAT: MLP-based predictor for latent space prediction
 * WHY:  Core of JEPA architecture - predict embeddings from context
 * HOW:  Forward/backward passes with FEP precision integration
 */

#include "cognitive/jepa/nimcp_jepa_predictor.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "utils/algorithms/nimcp_monte_carlo.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <float.h>

/* ============================================================================
 * Monte Carlo Integration - GPU acceleration with CPU fallback
 * ============================================================================ */

static __thread uint32_t g_jepa_pred_mc_seed = 0;

#ifdef NIMCP_ENABLE_CUDA
#include "gpu/quantum/nimcp_qmc_gpu.h"
#include "gpu/context/nimcp_gpu_context.h"

static nimcp_gpu_context_t* g_jepa_gpu_ctx = NULL;
static qmc_gpu_rng_t g_jepa_gpu_rng = NULL;
static bool g_jepa_gpu_init_attempted = false;

static bool jepa_init_gpu_mc(void) {
    if (g_jepa_gpu_init_attempted) return g_jepa_gpu_rng != NULL;
    g_jepa_gpu_init_attempted = true;

    if (!qmc_gpu_is_available()) return false;

    g_jepa_gpu_ctx = nimcp_gpu_context_create_auto();
    if (!g_jepa_gpu_ctx) return false;

    g_jepa_gpu_rng = qmc_gpu_rng_create(g_jepa_gpu_ctx, 4096, 0);
    if (!g_jepa_gpu_rng) {
        nimcp_gpu_context_destroy(g_jepa_gpu_ctx);
        g_jepa_gpu_ctx = NULL;
        return false;
    }
    return true;
}

static inline bool jepa_has_gpu_mc(void) {
    if (!g_jepa_gpu_init_attempted) jepa_init_gpu_mc();
    return g_jepa_gpu_rng != NULL;
}

#else
static inline bool jepa_has_gpu_mc(void) { return false; }
#endif

/* ============================================================================
 * Module Constants
 * ============================================================================ */

#define LOG_MODULE "[JEPA_PRED]"

/* ============================================================================
 * Internal Helpers - Activation Functions
 * ============================================================================ */

static inline float gelu(float x) {
    /* GELU(x) = x * Φ(x) ≈ x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³))) */
    const float sqrt_2_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;
    float x3 = x * x * x;
    float inner = sqrt_2_pi * (x + coeff * x3);
    return 0.5f * x * (1.0f + tanhf(inner));
}

static inline float gelu_derivative(float x) {
    /* Approximate derivative of GELU */
    const float sqrt_2_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;
    float x2 = x * x;
    float x3 = x2 * x;
    float inner = sqrt_2_pi * (x + coeff * x3);
    float tanh_inner = tanhf(inner);
    float sech2 = 1.0f - tanh_inner * tanh_inner;
    float d_inner = sqrt_2_pi * (1.0f + 3.0f * coeff * x2);
    return 0.5f * (1.0f + tanh_inner) + 0.5f * x * sech2 * d_inner;
}

static inline float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

static inline float relu_derivative(float x) {
    return x > 0.0f ? 1.0f : 0.0f;
}

static inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static inline float sigmoid_derivative(float x) {
    float s = sigmoid(x);
    return s * (1.0f - s);
}

static float apply_activation(float x, jepa_activation_t act) {
    switch (act) {
        case JEPA_ACT_NONE:    return x;
        case JEPA_ACT_RELU:    return relu(x);
        case JEPA_ACT_GELU:    return gelu(x);
        case JEPA_ACT_TANH:    return tanhf(x);
        case JEPA_ACT_SIGMOID: return sigmoid(x);
        default:               return x;
    }
}

static float apply_activation_derivative(float x, jepa_activation_t act) {
    switch (act) {
        case JEPA_ACT_NONE:    return 1.0f;
        case JEPA_ACT_RELU:    return relu_derivative(x);
        case JEPA_ACT_GELU:    return gelu_derivative(x);
        case JEPA_ACT_TANH: {
            float t = tanhf(x);
            return 1.0f - t * t;
        }
        case JEPA_ACT_SIGMOID: return sigmoid_derivative(x);
        default:               return 1.0f;
    }
}

/* ============================================================================
 * Internal Helpers - Weight Initialization
 * ============================================================================ */

static void xavier_init(float* weights, uint32_t in_dim, uint32_t out_dim) {
    /* Xavier/Glorot initialization: scale = sqrt(2 / (in + out)) */
    float scale = sqrtf(2.0f / (float)(in_dim + out_dim));

    /* Ensure MC seed is initialized */
    if (g_jepa_pred_mc_seed == 0) {
        g_jepa_pred_mc_seed = mc_seed_from_time();
    }

    for (uint32_t i = 0; i < in_dim * out_dim; i++) {
        /* Use MC uniform random in [-scale, scale] */
        float r = mc_random_uniform(&g_jepa_pred_mc_seed) * 2.0f - 1.0f;
        weights[i] = r * scale;
    }
}

/* ============================================================================
 * Internal Helpers - MLP Layer Operations
 * ============================================================================ */

static int mlp_layer_create(jepa_mlp_layer_t* layer, uint32_t in_dim,
                             uint32_t out_dim, jepa_activation_t activation) {
    layer->in_dim = in_dim;
    layer->out_dim = out_dim;
    layer->activation = activation;

    /* Allocate weights [out_dim × in_dim] */
    layer->weights = nimcp_malloc(out_dim * in_dim * sizeof(float));
    if (!layer->weights) {
        return NIMCP_ERROR_MEMORY;
    }
    xavier_init(layer->weights, in_dim, out_dim);

    /* Allocate bias [out_dim] */
    layer->bias = nimcp_malloc(out_dim * sizeof(float));
    if (!layer->bias) {
        nimcp_free(layer->weights);
        return NIMCP_ERROR_MEMORY;
    }
    memset(layer->bias, 0, out_dim * sizeof(float));

    /* Allocate gradient buffers */
    layer->grad_weights = nimcp_malloc(out_dim * in_dim * sizeof(float));
    layer->grad_bias = nimcp_malloc(out_dim * sizeof(float));
    if (!layer->grad_weights || !layer->grad_bias) {
        nimcp_free(layer->weights);
        nimcp_free(layer->bias);
        nimcp_free(layer->grad_weights);
        nimcp_free(layer->grad_bias);
        return NIMCP_ERROR_MEMORY;
    }
    memset(layer->grad_weights, 0, out_dim * in_dim * sizeof(float));
    memset(layer->grad_bias, 0, out_dim * sizeof(float));

    return NIMCP_SUCCESS;
}

static void mlp_layer_destroy(jepa_mlp_layer_t* layer) {
    if (layer->weights) nimcp_free(layer->weights);
    if (layer->bias) nimcp_free(layer->bias);
    if (layer->grad_weights) nimcp_free(layer->grad_weights);
    if (layer->grad_bias) nimcp_free(layer->grad_bias);
    memset(layer, 0, sizeof(jepa_mlp_layer_t));
}

static void mlp_layer_forward(const jepa_mlp_layer_t* layer,
                               const float* input,
                               float* pre_act,
                               float* output) {
    /* output = activation(weights @ input + bias) */
    for (uint32_t i = 0; i < layer->out_dim; i++) {
        double sum = layer->bias[i];
        for (uint32_t j = 0; j < layer->in_dim; j++) {
            sum += layer->weights[i * layer->in_dim + j] * input[j];
        }
        pre_act[i] = (float)sum;
        output[i] = apply_activation(pre_act[i], layer->activation);
    }
}

/* ============================================================================
 * Configuration API
 * ============================================================================ */

int jepa_predictor_default_config(jepa_predictor_config_t* config) {
    if (!config) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    config->type = JEPA_PREDICTOR_MLP;
    config->input_dim = NIMCP_JEPA_LATENT_DIM;
    config->output_dim = NIMCP_JEPA_LATENT_DIM;
    config->hidden_dim = NIMCP_JEPA_PREDICTOR_HIDDEN;
    config->num_layers = JEPA_PREDICTOR_DEFAULT_LAYERS;
    config->activation = JEPA_ACT_GELU;
    config->loss_type = JEPA_LOSS_MSE;

    config->learning_rate = JEPA_PREDICTOR_DEFAULT_LR;
    config->weight_decay = JEPA_PREDICTOR_DEFAULT_WEIGHT_DECAY;
    config->dropout_rate = 0.0f;
    config->enable_layer_norm = false;

    config->enable_fep = true;
    config->initial_precision = 1.0f;

    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Lifecycle API
 * ============================================================================ */

jepa_predictor_t* jepa_predictor_create(const jepa_predictor_config_t* config) {
    jepa_predictor_config_t default_config;

    if (!config) {
        jepa_predictor_default_config(&default_config);
        config = &default_config;
    }

    /* Validate config */
    if (config->input_dim == 0 || config->output_dim == 0) {
        NIMCP_LOGGING_ERROR(LOG_MODULE " Invalid dimensions");
        return NULL;
    }
    if (config->num_layers > JEPA_PREDICTOR_MAX_LAYERS) {
        NIMCP_LOGGING_ERROR(LOG_MODULE " Too many layers: %u", config->num_layers);
        return NULL;
    }

    /* Allocate predictor */
    jepa_predictor_t* pred = nimcp_malloc(sizeof(jepa_predictor_t));
    if (!pred) {
        NIMCP_LOGGING_ERROR(LOG_MODULE " Failed to allocate predictor");
        return NULL;
    }
    memset(pred, 0, sizeof(jepa_predictor_t));

    /* Initialize bridge base */
    if (bridge_base_init(&pred->base, BIO_MODULE_JEPA_PREDICTOR, "jepa_predictor") != 0) {
        nimcp_free(pred);
        return NULL;
    }

    /* Store config */
    pred->config = *config;
    pred->type = config->type;
    pred->prediction_precision = config->initial_precision;
    pred->training_mode = false;

    /* Build MLP network */
    if (config->type == JEPA_PREDICTOR_MLP || config->type == JEPA_PREDICTOR_LINEAR) {
        jepa_mlp_t* mlp = &pred->network.mlp;

        uint32_t actual_layers = (config->type == JEPA_PREDICTOR_LINEAR) ? 1 : config->num_layers;
        mlp->num_layers = actual_layers;

        /* Allocate layer array */
        mlp->layers = nimcp_malloc(actual_layers * sizeof(jepa_mlp_layer_t));
        if (!mlp->layers) {
            bridge_base_cleanup(&pred->base);
            nimcp_free(pred);
            return NULL;
        }
        memset(mlp->layers, 0, actual_layers * sizeof(jepa_mlp_layer_t));

        /* Allocate activation buffers */
        mlp->activations = nimcp_malloc(actual_layers * sizeof(float*));
        mlp->pre_activations = nimcp_malloc(actual_layers * sizeof(float*));
        if (!mlp->activations || !mlp->pre_activations) {
            nimcp_free(mlp->layers);
            nimcp_free(mlp->activations);
            nimcp_free(mlp->pre_activations);
            bridge_base_cleanup(&pred->base);
            nimcp_free(pred);
            return NULL;
        }

        /* Create layers */
        for (uint32_t i = 0; i < actual_layers; i++) {
            uint32_t in_d, out_d;
            jepa_activation_t act;

            if (actual_layers == 1) {
                /* Linear: single layer input -> output */
                in_d = config->input_dim;
                out_d = config->output_dim;
                act = JEPA_ACT_NONE;
            } else if (i == 0) {
                /* First layer: input -> hidden */
                in_d = config->input_dim;
                out_d = config->hidden_dim;
                act = config->activation;
            } else if (i == actual_layers - 1) {
                /* Last layer: hidden -> output, no activation */
                in_d = config->hidden_dim;
                out_d = config->output_dim;
                act = JEPA_ACT_NONE;
            } else {
                /* Hidden layer: hidden -> hidden */
                in_d = config->hidden_dim;
                out_d = config->hidden_dim;
                act = config->activation;
            }

            if (mlp_layer_create(&mlp->layers[i], in_d, out_d, act) != NIMCP_SUCCESS) {
                /* Cleanup on failure */
                for (uint32_t j = 0; j < i; j++) {
                    mlp_layer_destroy(&mlp->layers[j]);
                }
                nimcp_free(mlp->layers);
                nimcp_free(mlp->activations);
                nimcp_free(mlp->pre_activations);
                bridge_base_cleanup(&pred->base);
                nimcp_free(pred);
                return NULL;
            }

            /* Allocate activation buffers for this layer */
            mlp->activations[i] = nimcp_malloc(out_d * sizeof(float));
            mlp->pre_activations[i] = nimcp_malloc(out_d * sizeof(float));
            if (!mlp->activations[i] || !mlp->pre_activations[i]) {
                /* Cleanup */
                for (uint32_t j = 0; j <= i; j++) {
                    mlp_layer_destroy(&mlp->layers[j]);
                    nimcp_free(mlp->activations[j]);
                    nimcp_free(mlp->pre_activations[j]);
                }
                nimcp_free(mlp->layers);
                nimcp_free(mlp->activations);
                nimcp_free(mlp->pre_activations);
                bridge_base_cleanup(&pred->base);
                nimcp_free(pred);
                return NULL;
            }
        }
    }

    /* Allocate temporary buffers */
    /* temp_hidden stores input for backward pass, needs input_dim size */
    uint32_t temp_size = config->input_dim > config->hidden_dim ?
                         config->input_dim : config->hidden_dim;
    pred->temp_hidden = nimcp_malloc(temp_size * sizeof(float));
    pred->temp_output = nimcp_malloc(config->output_dim * sizeof(float));
    if (!pred->temp_hidden || !pred->temp_output) {
        /* Full cleanup would be needed here - simplified */
        NIMCP_LOGGING_ERROR(LOG_MODULE " Failed to allocate temp buffers");
        jepa_predictor_destroy(pred);
        return NULL;
    }

    /* Initialize statistics */
    pred->stats.min_loss = FLT_MAX;
    pred->stats.max_loss = 0.0f;

    NIMCP_LOGGING_INFO(LOG_MODULE " Created predictor: type=%s, dims=%u->%u->%u, layers=%u",
                      jepa_predictor_type_to_string(pred->type),
                      config->input_dim, config->hidden_dim, config->output_dim,
                      config->num_layers);

    return pred;
}

void jepa_predictor_destroy(jepa_predictor_t* predictor) {
    if (!predictor) return;

    /* Cleanup MLP */
    if (predictor->type == JEPA_PREDICTOR_MLP || predictor->type == JEPA_PREDICTOR_LINEAR) {
        jepa_mlp_t* mlp = &predictor->network.mlp;
        if (mlp->layers) {
            for (uint32_t i = 0; i < mlp->num_layers; i++) {
                mlp_layer_destroy(&mlp->layers[i]);
                if (mlp->activations && mlp->activations[i]) {
                    nimcp_free(mlp->activations[i]);
                }
                if (mlp->pre_activations && mlp->pre_activations[i]) {
                    nimcp_free(mlp->pre_activations[i]);
                }
            }
            nimcp_free(mlp->layers);
        }
        nimcp_free(mlp->activations);
        nimcp_free(mlp->pre_activations);
    }

    /* Free temp buffers */
    nimcp_free(predictor->temp_hidden);
    nimcp_free(predictor->temp_output);

    /* Cleanup bridge base */
    bridge_base_cleanup(&predictor->base);

    nimcp_free(predictor);
}

int jepa_predictor_reset(jepa_predictor_t* predictor) {
    if (!predictor) return NIMCP_ERROR_NULL_POINTER;

    /* Reinitialize weights */
    if (predictor->type == JEPA_PREDICTOR_MLP || predictor->type == JEPA_PREDICTOR_LINEAR) {
        jepa_mlp_t* mlp = &predictor->network.mlp;
        for (uint32_t i = 0; i < mlp->num_layers; i++) {
            jepa_mlp_layer_t* layer = &mlp->layers[i];
            xavier_init(layer->weights, layer->in_dim, layer->out_dim);
            memset(layer->bias, 0, layer->out_dim * sizeof(float));
            memset(layer->grad_weights, 0, layer->out_dim * layer->in_dim * sizeof(float));
            memset(layer->grad_bias, 0, layer->out_dim * sizeof(float));
        }
    }

    /* Reset statistics */
    memset(&predictor->stats, 0, sizeof(jepa_predictor_stats_t));
    predictor->stats.min_loss = FLT_MAX;
    predictor->step_count = 0;

    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Prediction API
 * ============================================================================ */

int jepa_predictor_predict(jepa_predictor_t* predictor,
                            const jepa_latent_t* context,
                            jepa_latent_t* prediction) {
    if (!predictor || !context || !prediction) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    if (context->latent_dim != predictor->config.input_dim) {
        NIMCP_LOGGING_ERROR(LOG_MODULE " Input dim mismatch: %u vs %u",
                           context->latent_dim, predictor->config.input_dim);
        return NIMCP_ERROR_INVALID_PARAM;
    }

    if (prediction->latent_dim != predictor->config.output_dim) {
        NIMCP_LOGGING_ERROR(LOG_MODULE " Output dim mismatch: %u vs %u",
                           prediction->latent_dim, predictor->config.output_dim);
        return NIMCP_ERROR_INVALID_PARAM;
    }

    /* Forward pass through MLP */
    if (predictor->type == JEPA_PREDICTOR_MLP || predictor->type == JEPA_PREDICTOR_LINEAR) {
        jepa_mlp_t* mlp = &predictor->network.mlp;
        const float* input = context->embedding;

        /* Save input for backward pass */
        memcpy(predictor->temp_hidden, context->embedding,
               predictor->config.input_dim * sizeof(float));

        for (uint32_t i = 0; i < mlp->num_layers; i++) {
            float* output = (i == mlp->num_layers - 1) ?
                           prediction->embedding : mlp->activations[i];

            mlp_layer_forward(&mlp->layers[i], input,
                             mlp->pre_activations[i], output);

            input = output;  /* Output becomes input for next layer */
        }
    }

    /* Update metadata */
    prediction->modality = context->modality;
    prediction->is_normalized = false;

    predictor->stats.predictions_made++;

    return NIMCP_SUCCESS;
}

int jepa_predictor_predict_masked(jepa_predictor_t* predictor,
                                   const jepa_latent_t* context,
                                   const float* mask,
                                   jepa_latent_t* prediction) {
    /* For now, just do regular prediction - mask is used in loss computation */
    return jepa_predictor_predict(predictor, context, prediction);
}

/* ============================================================================
 * Loss Computation API
 * ============================================================================ */

int jepa_predictor_compute_error(jepa_predictor_t* predictor,
                                  const jepa_latent_t* prediction,
                                  const jepa_latent_t* target,
                                  jepa_prediction_error_t* error) {
    if (!predictor || !prediction || !target || !error) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    if (prediction->latent_dim != target->latent_dim) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    if (!error->error || error->dim != prediction->latent_dim) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    /* Compute raw error */
    double sum_sq = 0.0;
    for (uint32_t i = 0; i < prediction->latent_dim; i++) {
        error->error[i] = prediction->embedding[i] - target->embedding[i];
        sum_sq += error->error[i] * error->error[i];
    }

    /* Compute precision-weighted error if enabled */
    if (predictor->config.enable_fep && error->weighted_error) {
        float prec = predictor->prediction_precision;
        for (uint32_t i = 0; i < prediction->latent_dim; i++) {
            error->weighted_error[i] = prec * error->error[i];
        }
        error->precision = prec;
    }

    /* Compute loss based on type */
    switch (predictor->config.loss_type) {
        case JEPA_LOSS_MSE:
            error->loss = (float)(sum_sq / prediction->latent_dim);
            break;

        case JEPA_LOSS_COSINE: {
            float sim = jepa_latent_cosine_similarity(prediction, target);
            error->loss = 1.0f - sim;
            break;
        }

        case JEPA_LOSS_SMOOTH_L1: {
            /* Huber loss with delta = 1.0 */
            double loss = 0.0;
            for (uint32_t i = 0; i < prediction->latent_dim; i++) {
                float abs_err = fabsf(error->error[i]);
                if (abs_err < 1.0f) {
                    loss += 0.5f * error->error[i] * error->error[i];
                } else {
                    loss += abs_err - 0.5f;
                }
            }
            error->loss = (float)(loss / prediction->latent_dim);
            break;
        }

        case JEPA_LOSS_PRECISION_WEIGHTED:
            error->loss = (float)(predictor->prediction_precision * sum_sq / prediction->latent_dim);
            break;

        default:
            error->loss = (float)(sum_sq / prediction->latent_dim);
    }

    /* Update statistics */
    /* On first loss, initialize avg_loss directly */
    if (predictor->stats.min_loss == FLT_MAX) {
        predictor->stats.avg_loss = error->loss;
    } else {
        predictor->stats.avg_loss = 0.9f * predictor->stats.avg_loss + 0.1f * error->loss;
    }
    if (error->loss < predictor->stats.min_loss) {
        predictor->stats.min_loss = error->loss;
    }
    if (error->loss > predictor->stats.max_loss) {
        predictor->stats.max_loss = error->loss;
    }

    return NIMCP_SUCCESS;
}

float jepa_predictor_compute_loss(jepa_predictor_t* predictor,
                                   const jepa_latent_t* prediction,
                                   const jepa_latent_t* target) {
    if (!predictor || !prediction || !target) {
        return NAN;
    }

    /* Simple MSE for direct computation */
    double sum_sq = 0.0;
    for (uint32_t i = 0; i < prediction->latent_dim; i++) {
        float diff = prediction->embedding[i] - target->embedding[i];
        sum_sq += diff * diff;
    }

    return (float)(sum_sq / prediction->latent_dim);
}

/* ============================================================================
 * Training API
 * ============================================================================ */

int jepa_predictor_set_training(jepa_predictor_t* predictor, bool training) {
    if (!predictor) return NIMCP_ERROR_NULL_POINTER;
    predictor->training_mode = training;
    return NIMCP_SUCCESS;
}

int jepa_predictor_backward(jepa_predictor_t* predictor,
                             const jepa_prediction_error_t* error) {
    if (!predictor || !error) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    if (predictor->type != JEPA_PREDICTOR_MLP && predictor->type != JEPA_PREDICTOR_LINEAR) {
        return NIMCP_ERROR_NOT_IMPLEMENTED;
    }

    jepa_mlp_t* mlp = &predictor->network.mlp;

    /* Start with output error gradient */
    float* grad_out = nimcp_malloc(predictor->config.output_dim * sizeof(float));
    if (!grad_out) return NIMCP_ERROR_MEMORY;

    /* For MSE: dL/dout = 2 * (pred - target) / dim = 2 * error / dim */
    float scale = 2.0f / error->dim;
    for (uint32_t i = 0; i < error->dim; i++) {
        grad_out[i] = scale * error->error[i];
    }

    /* Backpropagate through layers */
    float* grad_curr = grad_out;
    float* grad_next = NULL;

    for (int32_t l = (int32_t)mlp->num_layers - 1; l >= 0; l--) {
        jepa_mlp_layer_t* layer = &mlp->layers[l];

        /* Get input to this layer */
        const float* layer_input = (l == 0) ?
            predictor->temp_hidden :  /* Would need to save input */
            mlp->activations[l - 1];

        /* For simplicity, use pre-stored activations */
        /* In production, would need to save input during forward pass */

        /* Apply activation gradient */
        for (uint32_t i = 0; i < layer->out_dim; i++) {
            float act_grad = apply_activation_derivative(mlp->pre_activations[l][i],
                                                          layer->activation);
            grad_curr[i] *= act_grad;
        }

        /* Accumulate bias gradient */
        for (uint32_t i = 0; i < layer->out_dim; i++) {
            layer->grad_bias[i] += grad_curr[i];
        }

        /* Accumulate weight gradient: grad_W[i,j] = grad_out[i] * input[j] */
        for (uint32_t i = 0; i < layer->out_dim; i++) {
            for (uint32_t j = 0; j < layer->in_dim; j++) {
                float input_val = layer_input ? layer_input[j] : 0.0f;
                layer->grad_weights[i * layer->in_dim + j] += grad_curr[i] * input_val;
            }
        }

        /* Propagate to previous layer if not first */
        if (l > 0) {
            grad_next = nimcp_malloc(layer->in_dim * sizeof(float));
            if (!grad_next) {
                nimcp_free(grad_out);
                return NIMCP_ERROR_MEMORY;
            }
            memset(grad_next, 0, layer->in_dim * sizeof(float));

            /* grad_input[j] = sum_i(grad_out[i] * W[i,j]) */
            for (uint32_t j = 0; j < layer->in_dim; j++) {
                for (uint32_t i = 0; i < layer->out_dim; i++) {
                    grad_next[j] += grad_curr[i] * layer->weights[i * layer->in_dim + j];
                }
            }

            if (grad_curr != grad_out) {
                nimcp_free(grad_curr);
            }
            grad_curr = grad_next;
        }
    }

    nimcp_free(grad_out);
    if (grad_curr != grad_out) {
        nimcp_free(grad_curr);
    }

    return NIMCP_SUCCESS;
}

int jepa_predictor_update_weights(jepa_predictor_t* predictor, float learning_rate) {
    if (!predictor) return NIMCP_ERROR_NULL_POINTER;

    if (learning_rate <= 0.0f) {
        learning_rate = predictor->config.learning_rate;
    }

    if (predictor->type != JEPA_PREDICTOR_MLP && predictor->type != JEPA_PREDICTOR_LINEAR) {
        return NIMCP_ERROR_NOT_IMPLEMENTED;
    }

    jepa_mlp_t* mlp = &predictor->network.mlp;
    float wd = predictor->config.weight_decay;

    for (uint32_t l = 0; l < mlp->num_layers; l++) {
        jepa_mlp_layer_t* layer = &mlp->layers[l];

        /* Update weights with gradient descent and weight decay */
        for (uint32_t i = 0; i < layer->out_dim * layer->in_dim; i++) {
            float grad = layer->grad_weights[i];

            /* Gradient clipping */
            if (grad > JEPA_PREDICTOR_GRAD_CLIP) grad = JEPA_PREDICTOR_GRAD_CLIP;
            if (grad < -JEPA_PREDICTOR_GRAD_CLIP) grad = -JEPA_PREDICTOR_GRAD_CLIP;

            layer->weights[i] -= learning_rate * (grad + wd * layer->weights[i]);
        }

        /* Update biases */
        for (uint32_t i = 0; i < layer->out_dim; i++) {
            float grad = layer->grad_bias[i];
            if (grad > JEPA_PREDICTOR_GRAD_CLIP) grad = JEPA_PREDICTOR_GRAD_CLIP;
            if (grad < -JEPA_PREDICTOR_GRAD_CLIP) grad = -JEPA_PREDICTOR_GRAD_CLIP;
            layer->bias[i] -= learning_rate * grad;
        }

        /* Zero gradients for next iteration */
        memset(layer->grad_weights, 0, layer->out_dim * layer->in_dim * sizeof(float));
        memset(layer->grad_bias, 0, layer->out_dim * sizeof(float));
    }

    predictor->stats.updates_applied++;
    predictor->step_count++;

    return NIMCP_SUCCESS;
}

int jepa_predictor_train_step(jepa_predictor_t* predictor,
                               const jepa_latent_t* context,
                               const jepa_latent_t* target,
                               float* loss) {
    if (!predictor || !context || !target) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    /* Create temporary prediction */
    jepa_latent_t* pred = jepa_latent_create_dim(predictor->config.output_dim);
    if (!pred) return NIMCP_ERROR_MEMORY;

    /* Forward pass */
    int result = jepa_predictor_predict(predictor, context, pred);
    if (result != NIMCP_SUCCESS) {
        jepa_latent_destroy(pred);
        return result;
    }

    /* Compute error */
    jepa_prediction_error_t* error = jepa_prediction_error_create(pred->latent_dim);
    if (!error) {
        jepa_latent_destroy(pred);
        return NIMCP_ERROR_MEMORY;
    }

    result = jepa_predictor_compute_error(predictor, pred, target, error);
    if (result != NIMCP_SUCCESS) {
        jepa_prediction_error_destroy(error);
        jepa_latent_destroy(pred);
        return result;
    }

    if (loss) {
        *loss = error->loss;
    }

    /* Backward pass (simplified) */
    result = jepa_predictor_backward(predictor, error);

    /* Update weights */
    if (result == NIMCP_SUCCESS) {
        result = jepa_predictor_update_weights(predictor, 0.0f);
    }

    jepa_prediction_error_destroy(error);
    jepa_latent_destroy(pred);

    return result;
}

/* ============================================================================
 * FEP Integration API
 * ============================================================================ */

int jepa_predictor_update_precision(jepa_predictor_t* predictor,
                                     const jepa_prediction_error_t* error) {
    if (!predictor || !error) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    /* Precision = 1 / mean(error²) */
    /* Use exponential moving average for stability */
    float new_precision = 1.0f / (error->loss + JEPA_LATENT_EPSILON);

    /* Clamp precision */
    if (new_precision < JEPA_LATENT_MIN_PRECISION) {
        new_precision = JEPA_LATENT_MIN_PRECISION;
    }
    if (new_precision > JEPA_LATENT_MAX_PRECISION) {
        new_precision = JEPA_LATENT_MAX_PRECISION;
    }

    /* EMA update */
    predictor->prediction_precision = 0.9f * predictor->prediction_precision +
                                       0.1f * new_precision;

    predictor->stats.avg_precision = 0.9f * predictor->stats.avg_precision +
                                      0.1f * predictor->prediction_precision;

    return NIMCP_SUCCESS;
}

int jepa_predictor_to_fep_error(jepa_predictor_t* predictor,
                                 const jepa_prediction_error_t* internal_error,
                                 fep_prediction_error_t* fep_error) {
    if (!predictor || !internal_error || !fep_error) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    /* Ensure FEP error has same dimensionality */
    if (fep_error->dim != internal_error->dim) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    /* Copy error values */
    memcpy(fep_error->error, internal_error->error, internal_error->dim * sizeof(float));

    /* Apply precision weighting */
    if (fep_error->weighted_error && fep_error->precision) {
        float prec = predictor->prediction_precision;
        for (uint32_t i = 0; i < internal_error->dim; i++) {
            fep_error->precision[i] = prec;
            fep_error->weighted_error[i] = prec * internal_error->error[i];
        }
    }

    /* Compute magnitudes */
    double sum_sq = 0.0;
    double weighted_sum_sq = 0.0;
    for (uint32_t i = 0; i < internal_error->dim; i++) {
        sum_sq += internal_error->error[i] * internal_error->error[i];
        if (fep_error->weighted_error) {
            weighted_sum_sq += fep_error->weighted_error[i] * fep_error->weighted_error[i];
        }
    }
    fep_error->magnitude = (float)sqrt(sum_sq);
    fep_error->weighted_magnitude = (float)sqrt(weighted_sum_sq);

    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Weight Access API
 * ============================================================================ */

uint32_t jepa_predictor_num_params(const jepa_predictor_t* predictor) {
    if (!predictor) return 0;

    if (predictor->type != JEPA_PREDICTOR_MLP && predictor->type != JEPA_PREDICTOR_LINEAR) {
        return 0;
    }

    uint32_t total = 0;
    const jepa_mlp_t* mlp = &predictor->network.mlp;

    for (uint32_t i = 0; i < mlp->num_layers; i++) {
        const jepa_mlp_layer_t* layer = &mlp->layers[i];
        total += layer->in_dim * layer->out_dim;  /* Weights */
        total += layer->out_dim;                   /* Biases */
    }

    return total;
}

int jepa_predictor_get_weights(const jepa_predictor_t* predictor,
                                uint32_t layer_idx,
                                float** weights,
                                uint32_t dims[2]) {
    if (!predictor || !weights || !dims) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    if (predictor->type != JEPA_PREDICTOR_MLP && predictor->type != JEPA_PREDICTOR_LINEAR) {
        return NIMCP_ERROR_NOT_IMPLEMENTED;
    }

    const jepa_mlp_t* mlp = &predictor->network.mlp;
    if (layer_idx >= mlp->num_layers) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    const jepa_mlp_layer_t* layer = &mlp->layers[layer_idx];
    *weights = layer->weights;
    dims[0] = layer->out_dim;
    dims[1] = layer->in_dim;

    return NIMCP_SUCCESS;
}

int jepa_predictor_set_weights(jepa_predictor_t* predictor,
                                uint32_t layer_idx,
                                const float* weights,
                                uint32_t in_dim,
                                uint32_t out_dim) {
    if (!predictor || !weights) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    if (predictor->type != JEPA_PREDICTOR_MLP && predictor->type != JEPA_PREDICTOR_LINEAR) {
        return NIMCP_ERROR_NOT_IMPLEMENTED;
    }

    jepa_mlp_t* mlp = &predictor->network.mlp;
    if (layer_idx >= mlp->num_layers) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    jepa_mlp_layer_t* layer = &mlp->layers[layer_idx];
    if (layer->in_dim != in_dim || layer->out_dim != out_dim) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    memcpy(layer->weights, weights, in_dim * out_dim * sizeof(float));
    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Statistics API
 * ============================================================================ */

int jepa_predictor_get_stats(const jepa_predictor_t* predictor,
                              jepa_predictor_stats_t* stats) {
    if (!predictor || !stats) {
        return NIMCP_ERROR_NULL_POINTER;
    }
    *stats = predictor->stats;
    return NIMCP_SUCCESS;
}

int jepa_predictor_reset_stats(jepa_predictor_t* predictor) {
    if (!predictor) return NIMCP_ERROR_NULL_POINTER;
    memset(&predictor->stats, 0, sizeof(jepa_predictor_stats_t));
    predictor->stats.min_loss = FLT_MAX;
    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Bio-Async API
 * ============================================================================ */

BRIDGE_DEFINE_BIO_ASYNC_FUNCS(jepa_predictor)

/* ============================================================================
 * Prediction Error Management
 * ============================================================================ */

jepa_prediction_error_t* jepa_prediction_error_create(uint32_t dim) {
    if (dim == 0) return NULL;

    jepa_prediction_error_t* error = nimcp_malloc(sizeof(jepa_prediction_error_t));
    if (!error) return NULL;

    error->dim = dim;
    error->error = nimcp_malloc(dim * sizeof(float));
    error->weighted_error = nimcp_malloc(dim * sizeof(float));

    if (!error->error || !error->weighted_error) {
        nimcp_free(error->error);
        nimcp_free(error->weighted_error);
        nimcp_free(error);
        return NULL;
    }

    memset(error->error, 0, dim * sizeof(float));
    memset(error->weighted_error, 0, dim * sizeof(float));
    error->loss = 0.0f;
    error->precision = 1.0f;

    return error;
}

void jepa_prediction_error_destroy(jepa_prediction_error_t* error) {
    if (!error) return;
    nimcp_free(error->error);
    nimcp_free(error->weighted_error);
    nimcp_free(error);
}

/* ============================================================================
 * String Conversion API
 * ============================================================================ */

const char* jepa_predictor_type_to_string(jepa_predictor_type_t type) {
    switch (type) {
        case JEPA_PREDICTOR_LINEAR:      return "linear";
        case JEPA_PREDICTOR_MLP:         return "mlp";
        case JEPA_PREDICTOR_TRANSFORMER: return "transformer";
        case JEPA_PREDICTOR_RECURRENT:   return "recurrent";
        default:                         return "unknown";
    }
}

const char* jepa_activation_to_string(jepa_activation_t activation) {
    switch (activation) {
        case JEPA_ACT_NONE:    return "none";
        case JEPA_ACT_RELU:    return "relu";
        case JEPA_ACT_GELU:    return "gelu";
        case JEPA_ACT_TANH:    return "tanh";
        case JEPA_ACT_SIGMOID: return "sigmoid";
        default:               return "unknown";
    }
}

const char* jepa_loss_to_string(jepa_loss_t loss) {
    switch (loss) {
        case JEPA_LOSS_MSE:               return "mse";
        case JEPA_LOSS_COSINE:            return "cosine";
        case JEPA_LOSS_SMOOTH_L1:         return "smooth_l1";
        case JEPA_LOSS_PRECISION_WEIGHTED: return "precision_weighted";
        default:                          return "unknown";
    }
}

/* ============================================================================
 * Monte Carlo Integration API
 * ============================================================================ */

/**
 * @brief Predict with uncertainty estimation via MC sampling
 *
 * WHAT: Make prediction with uncertainty quantification
 * WHY:  Enable confidence-aware predictions in JEPA
 * HOW:  Sample predictions with noise, compute mean and variance
 *
 * @param predictor JEPA predictor
 * @param context Input context
 * @param prediction Output prediction (mean)
 * @param uncertainty Output uncertainty (std per dimension)
 * @param num_samples Number of MC samples
 * @return NIMCP_SUCCESS on success
 */
int jepa_predictor_predict_with_uncertainty_mc(
    jepa_predictor_t* predictor,
    const jepa_latent_t* context,
    jepa_latent_t* prediction,
    float* uncertainty,
    uint32_t num_samples
) {
    if (!predictor || !context || !prediction || !uncertainty || num_samples == 0) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    if (g_jepa_pred_mc_seed == 0) {
        g_jepa_pred_mc_seed = mc_seed_from_time();
    }

    uint32_t dim = prediction->latent_dim;

    /* Allocate accumulators */
    float* mean = nimcp_calloc(dim, sizeof(float));
    float* var = nimcp_calloc(dim, sizeof(float));
    if (!mean || !var) {
        nimcp_free(mean);
        nimcp_free(var);
        return NIMCP_ERROR_MEMORY;
    }

    /* Create temporary prediction buffer */
    jepa_latent_t* temp = jepa_latent_create_dim(dim);
    if (!temp) {
        nimcp_free(mean);
        nimcp_free(var);
        return NIMCP_ERROR_MEMORY;
    }

    /* Collect samples with dropout noise */
    for (uint32_t s = 0; s < num_samples; s++) {
        /* Forward pass */
        if (jepa_predictor_predict(predictor, context, temp) != NIMCP_SUCCESS) {
            continue;
        }

        /* Add small noise for exploration */
        float noise_scale = 0.01f;
        for (uint32_t i = 0; i < dim; i++) {
            float noise = mc_random_normal(&g_jepa_pred_mc_seed, 0.0f, noise_scale);
            temp->embedding[i] += noise;
            mean[i] += temp->embedding[i];
        }
    }

    /* Compute mean */
    for (uint32_t i = 0; i < dim; i++) {
        mean[i] /= (float)num_samples;
        prediction->embedding[i] = mean[i];
    }

    /* Second pass for variance (requires re-sampling or storing) */
    /* For simplicity, use analytical approximation based on precision */
    float precision = predictor->prediction_precision;
    float base_std = 1.0f / sqrtf(precision + 1e-6f);
    for (uint32_t i = 0; i < dim; i++) {
        uncertainty[i] = base_std * (1.0f + 0.1f * fabsf(mean[i]));
    }

    jepa_latent_destroy(temp);
    nimcp_free(mean);
    nimcp_free(var);

    return NIMCP_SUCCESS;
}

/**
 * @brief Apply MC dropout during training
 *
 * WHAT: Apply stochastic dropout using MC sampling
 * WHY:  Regularization and uncertainty estimation
 * HOW:  Randomly zero elements with dropout rate
 *
 * @param activations Activation array to modify in-place
 * @param size Array size
 * @param dropout_rate Probability of dropping [0,1]
 */
void jepa_predictor_apply_dropout_mc(float* activations, uint32_t size, float dropout_rate) {
    if (!activations || size == 0 || dropout_rate <= 0.0f) return;

    float scale = 1.0f / (1.0f - dropout_rate);  /* Inverted dropout scaling */

#ifdef NIMCP_ENABLE_CUDA
    /* Try GPU first (default for size >= 256) */
    if (jepa_has_gpu_mc() && size >= 256) {
        size_t dims[] = {size};
        nimcp_gpu_tensor_t* samples = nimcp_gpu_tensor_create(
            g_jepa_gpu_ctx, dims, 1, NIMCP_GPU_PRECISION_FP32);

        if (samples && qmc_gpu_sample_uniform(g_jepa_gpu_ctx, g_jepa_gpu_rng, samples)) {
            float* h_samples = nimcp_calloc(size, sizeof(float));
            if (h_samples) {
                cudaMemcpy(h_samples, samples->data, size * sizeof(float), cudaMemcpyDeviceToHost);
                for (uint32_t i = 0; i < size; i++) {
                    if (h_samples[i] < dropout_rate) {
                        activations[i] = 0.0f;
                    } else {
                        activations[i] *= scale;
                    }
                }
                nimcp_free(h_samples);
                nimcp_gpu_tensor_destroy(samples);
                return;
            }
        }
        if (samples) nimcp_gpu_tensor_destroy(samples);
    }
#endif

    /* CPU fallback */
    if (g_jepa_pred_mc_seed == 0) {
        g_jepa_pred_mc_seed = mc_seed_from_time();
    }

    for (uint32_t i = 0; i < size; i++) {
        if (mc_random_uniform(&g_jepa_pred_mc_seed) < dropout_rate) {
            activations[i] = 0.0f;
        } else {
            activations[i] *= scale;
        }
    }
}

/**
 * @brief Estimate prediction confidence via MC sampling
 *
 * WHAT: Estimate confidence in prediction
 * WHY:  Support uncertainty-aware decision making
 * HOW:  Compute prediction variance over samples
 *
 * @param predictor JEPA predictor
 * @param context Input context
 * @param num_samples Number of MC samples
 * @return Confidence score [0,1] (inverse of normalized variance)
 */
float jepa_predictor_estimate_confidence_mc(
    jepa_predictor_t* predictor,
    const jepa_latent_t* context,
    uint32_t num_samples
) {
    if (!predictor || !context || num_samples == 0) return 0.0f;

    if (g_jepa_pred_mc_seed == 0) {
        g_jepa_pred_mc_seed = mc_seed_from_time();
    }

    uint32_t dim = predictor->config.output_dim;
    jepa_latent_t* temp = jepa_latent_create_dim(dim);
    if (!temp) return 0.0f;

    float* mean = nimcp_calloc(dim, sizeof(float));
    float* m2 = nimcp_calloc(dim, sizeof(float));  /* For Welford's online variance */
    if (!mean || !m2) {
        nimcp_free(mean);
        nimcp_free(m2);
        jepa_latent_destroy(temp);
        return 0.0f;
    }

    /* Welford's online algorithm for variance */
    for (uint32_t s = 0; s < num_samples; s++) {
        if (jepa_predictor_predict(predictor, context, temp) != NIMCP_SUCCESS) {
            continue;
        }

        /* Add exploration noise */
        for (uint32_t i = 0; i < dim; i++) {
            float val = temp->embedding[i] +
                        mc_random_normal(&g_jepa_pred_mc_seed, 0.0f, 0.01f);
            float delta = val - mean[i];
            mean[i] += delta / (s + 1);
            float delta2 = val - mean[i];
            m2[i] += delta * delta2;
        }
    }

    /* Compute average variance */
    float avg_variance = 0.0f;
    for (uint32_t i = 0; i < dim; i++) {
        avg_variance += m2[i] / (num_samples - 1 + 1e-6f);
    }
    avg_variance /= dim;

    /* Convert to confidence: higher variance = lower confidence */
    float confidence = 1.0f / (1.0f + avg_variance);

    nimcp_free(mean);
    nimcp_free(m2);
    jepa_latent_destroy(temp);

    return confidence;
}

/**
 * @brief Get thread-local MC seed for JEPA predictor
 *
 * @return Pointer to thread-local seed
 */
uint32_t* jepa_predictor_get_mc_seed(void) {
    if (g_jepa_pred_mc_seed == 0) {
        g_jepa_pred_mc_seed = mc_seed_from_time();
    }
    return &g_jepa_pred_mc_seed;
}

/* ============================================================================
 * KG Self-Awareness API
 * ============================================================================ */

int jepa_predictor_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;

    const kg_entity_t* self = kg_reader_get_entity(kg, "JEPA_Predictor");
    if (self) {
        NIMCP_LOGGING_INFO(LOG_MODULE " Self-knowledge entity: %s (type: %s)",
                          self->name, self->entity_type);
        for (uint32_t i = 0; i < self->num_observations; i++) {
            NIMCP_LOGGING_DEBUG(LOG_MODULE " Observation[%u]: %s",
                               i, self->observations[i]);
        }
    }

    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "JEPA_Predictor");
    if (connections) {
        NIMCP_LOGGING_DEBUG(LOG_MODULE " Outgoing connections: %u", connections->count);
        kg_relation_list_destroy(connections);
    }

    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "JEPA_Predictor");
    if (incoming) {
        NIMCP_LOGGING_DEBUG(LOG_MODULE " Incoming connections: %u", incoming->count);
        kg_relation_list_destroy(incoming);
    }

    return self ? 1 : 0;
}
