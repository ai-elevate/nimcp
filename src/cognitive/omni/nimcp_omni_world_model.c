/**
 * @file nimcp_omni_world_model.c
 * @brief Implementation of Omnidirectional Generative World Model
 * @version 1.0.0
 * @date 2026-01-04
 *
 * IMPLEMENTATION NOTES:
 * =====================
 * This implementation draws from key research insights:
 *
 * 1. DREAMERV3 (Hafner et al., Nature 2025):
 *    - RSSM architecture with deterministic + stochastic components
 *    - Symlog transformation for reward normalization
 *    - Learning in latent space for efficiency
 *
 * 2. JEPA (LeCun, 2022-2025):
 *    - Prediction in embedding space, not pixel space
 *    - Self-supervised learning from structure
 *
 * 3. ACTIVE INFERENCE (Friston):
 *    - World model for policy evaluation
 *    - Free energy minimization through imagination
 */

#include "cognitive/omni/nimcp_omni_world_model.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/thread/nimcp_thread.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <float.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ============================================================================
 * Internal Structures
 * ============================================================================ */

/**
 * @brief Experience replay buffer (circular)
 */
typedef struct {
    omni_wm_experience_t** experiences;  /**< Experience array */
    uint32_t capacity;                    /**< Maximum size */
    uint32_t size;                        /**< Current size */
    uint32_t head;                        /**< Write position */
    float* priorities;                    /**< Priority values */
} omni_wm_replay_buffer_t;

/**
 * @brief Forward dynamics model weights (simple linear for now)
 */
typedef struct {
    float* W_h;      /**< Weights for deterministic transition */
    float* W_z;      /**< Weights for stochastic prior */
    float* W_obs;    /**< Weights for observation model */
    float* b_h;      /**< Bias for deterministic */
    float* b_z;      /**< Bias for stochastic */
    float* b_obs;    /**< Bias for observation */
    uint32_t h_dim;
    uint32_t z_dim;
    uint32_t obs_dim;
    uint32_t action_dim;
} omni_wm_dynamics_t;

/**
 * @brief World model internal structure
 */
struct omni_world_model {
    /* Configuration */
    omni_wm_config_t config;

    /* Current state */
    omni_wm_state_t* current_state;
    omni_wm_rssm_state_t* rssm_state;

    /* Dynamics models (one per direction) */
    omni_wm_dynamics_t* forward_dynamics;
    omni_wm_dynamics_t* backward_dynamics;
    omni_wm_dynamics_t* lateral_dynamics;

    /* Latent encoder/decoder weights */
    float* encoder_W;
    float* encoder_b;
    float* decoder_W;
    float* decoder_b;

    /* Experience replay */
    omni_wm_replay_buffer_t* replay_buffer;

    /* Statistics */
    omni_wm_stats_t stats;

    /* Active inference integration */
    struct omni_active_inference* ai_ctx;

    /* Thread safety */
    nimcp_mutex_t* mutex;

    /* Random state for sampling */
    unsigned int rand_seed;

    /* Initialized flag */
    bool initialized;
};

/* ============================================================================
 * Symlog Transformation (DreamerV3)
 * ============================================================================ */

float omni_wm_symlog(float x) {
    /* symlog(x) = sign(x) * ln(|x| + 1) */
    if (x >= 0.0f) {
        return logf(x + 1.0f);
    } else {
        return -logf(-x + 1.0f);
    }
}

float omni_wm_symexp(float x) {
    /* symexp(x) = sign(x) * (exp(|x|) - 1) */
    if (x >= 0.0f) {
        return expf(x) - 1.0f;
    } else {
        return -(expf(-x) - 1.0f);
    }
}

void omni_wm_symlog_array(const float* input, float* output, uint32_t size) {
    if (!input || !output) return;
    for (uint32_t i = 0; i < size; i++) {
        output[i] = omni_wm_symlog(input[i]);
    }
}

void omni_wm_symexp_array(const float* input, float* output, uint32_t size) {
    if (!input || !output) return;
    for (uint32_t i = 0; i < size; i++) {
        output[i] = omni_wm_symexp(input[i]);
    }
}

/* ============================================================================
 * Internal Helpers
 * ============================================================================ */

static float randn(unsigned int* seed) {
    /* Box-Muller transform for normal distribution */
    float u1 = (float)rand_r(seed) / RAND_MAX;
    float u2 = (float)rand_r(seed) / RAND_MAX;
    if (u1 < 1e-10f) u1 = 1e-10f;
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
}

static omni_wm_dynamics_t* dynamics_create(uint32_t h_dim, uint32_t z_dim,
                                            uint32_t obs_dim, uint32_t action_dim) {
    omni_wm_dynamics_t* dyn = nimcp_calloc(1, sizeof(omni_wm_dynamics_t));
    if (!dyn) return NULL;

    dyn->h_dim = h_dim;
    dyn->z_dim = z_dim;
    dyn->obs_dim = obs_dim;
    dyn->action_dim = action_dim;

    /* Allocate weight matrices */
    uint32_t input_dim = h_dim + z_dim + action_dim;

    dyn->W_h = nimcp_calloc(input_dim * h_dim, sizeof(float));
    dyn->W_z = nimcp_calloc(h_dim * z_dim * 2, sizeof(float)); /* mean + std */
    dyn->W_obs = nimcp_calloc((h_dim + z_dim) * obs_dim, sizeof(float));
    dyn->b_h = nimcp_calloc(h_dim, sizeof(float));
    dyn->b_z = nimcp_calloc(z_dim * 2, sizeof(float));
    dyn->b_obs = nimcp_calloc(obs_dim, sizeof(float));

    if (!dyn->W_h || !dyn->W_z || !dyn->W_obs ||
        !dyn->b_h || !dyn->b_z || !dyn->b_obs) {
        nimcp_free(dyn->W_h);
        nimcp_free(dyn->W_z);
        nimcp_free(dyn->W_obs);
        nimcp_free(dyn->b_h);
        nimcp_free(dyn->b_z);
        nimcp_free(dyn->b_obs);
        nimcp_free(dyn);
        return NULL;
    }

    /* Initialize with small random weights */
    unsigned int seed = (unsigned int)time(NULL);
    for (uint32_t i = 0; i < input_dim * h_dim; i++) {
        dyn->W_h[i] = randn(&seed) * 0.01f;
    }
    for (uint32_t i = 0; i < h_dim * z_dim * 2; i++) {
        dyn->W_z[i] = randn(&seed) * 0.01f;
    }
    for (uint32_t i = 0; i < (h_dim + z_dim) * obs_dim; i++) {
        dyn->W_obs[i] = randn(&seed) * 0.01f;
    }

    return dyn;
}

static void dynamics_destroy(omni_wm_dynamics_t* dyn) {
    if (!dyn) return;
    nimcp_free(dyn->W_h);
    nimcp_free(dyn->W_z);
    nimcp_free(dyn->W_obs);
    nimcp_free(dyn->b_h);
    nimcp_free(dyn->b_z);
    nimcp_free(dyn->b_obs);
    nimcp_free(dyn);
}

static omni_wm_replay_buffer_t* replay_buffer_create(uint32_t capacity) {
    omni_wm_replay_buffer_t* buf = nimcp_calloc(1, sizeof(omni_wm_replay_buffer_t));
    if (!buf) return NULL;

    buf->experiences = nimcp_calloc(capacity, sizeof(omni_wm_experience_t*));
    buf->priorities = nimcp_calloc(capacity, sizeof(float));
    if (!buf->experiences || !buf->priorities) {
        nimcp_free(buf->experiences);
        nimcp_free(buf->priorities);
        nimcp_free(buf);
        return NULL;
    }

    buf->capacity = capacity;
    buf->size = 0;
    buf->head = 0;

    return buf;
}

static void replay_buffer_destroy(omni_wm_replay_buffer_t* buf) {
    if (!buf) return;
    for (uint32_t i = 0; i < buf->size; i++) {
        omni_wm_experience_destroy(buf->experiences[i]);
    }
    nimcp_free(buf->experiences);
    nimcp_free(buf->priorities);
    nimcp_free(buf);
}

/* ============================================================================
 * Core API Implementation
 * ============================================================================ */

nimcp_error_t omni_wm_get_default_config(omni_wm_config_t* config) {
    if (!config) return NIMCP_ERROR_INVALID_PARAM;

    memset(config, 0, sizeof(omni_wm_config_t));

    /* Dimensionality */
    config->state_dim = 64;
    config->action_dim = 8;
    config->obs_dim = 64;
    config->hidden_dim = 128;
    config->num_levels = 3;

    /* RSSM settings */
    config->rssm_h_dim = 64;
    config->rssm_z_dim = 32;
    config->latent_dim = 64;

    /* MDN settings */
    config->mdn_components = 5;
    config->pred_type = OMNI_WM_PRED_STOCHASTIC;

    /* Learning settings */
    config->learning_rate = 0.001f;
    config->discount_factor = 0.99f;
    config->kl_weight = 1.0f;
    config->reward_scale = 1.0f;
    config->learn_mode = OMNI_WM_LEARN_REPLAY;

    /* Experience replay */
    config->replay_buffer_size = 10000;
    config->batch_size = 32;
    config->priority_exponent = 0.6f;

    /* Dreaming settings */
    config->dream_horizon = 15;
    config->dream_episodes = 10;
    config->imagination_noise = 0.1f;

    /* Feature flags */
    config->enable_lateral = true;
    config->enable_hierarchical = false;
    config->enable_dreaming = true;
    config->use_symlog_rewards = true;
    config->use_rssm = true;
    config->use_mdn = false; /* Start simple */

    return NIMCP_SUCCESS;
}

omni_world_model_t* omni_wm_create(const omni_wm_config_t* config) {
    omni_world_model_t* wm = nimcp_calloc(1, sizeof(omni_world_model_t));
    if (!wm) return NULL;

    /* Apply config or defaults */
    if (config) {
        memcpy(&wm->config, config, sizeof(omni_wm_config_t));
    } else {
        omni_wm_get_default_config(&wm->config);
    }

    /* Initialize random seed */
    wm->rand_seed = (unsigned int)time(NULL);

    /* Create dynamics models */
    wm->forward_dynamics = dynamics_create(
        wm->config.rssm_h_dim,
        wm->config.rssm_z_dim,
        wm->config.obs_dim,
        wm->config.action_dim
    );
    if (!wm->forward_dynamics) {
        nimcp_free(wm);
        return NULL;
    }

    wm->backward_dynamics = dynamics_create(
        wm->config.rssm_h_dim,
        wm->config.rssm_z_dim,
        wm->config.obs_dim,
        wm->config.action_dim
    );
    if (!wm->backward_dynamics) {
        dynamics_destroy(wm->forward_dynamics);
        nimcp_free(wm);
        return NULL;
    }

    if (wm->config.enable_lateral) {
        wm->lateral_dynamics = dynamics_create(
            wm->config.rssm_h_dim,
            wm->config.rssm_z_dim,
            wm->config.obs_dim,
            wm->config.action_dim
        );
    }

    /* Create encoder/decoder weights */
    uint32_t enc_size = wm->config.obs_dim * wm->config.latent_dim;
    wm->encoder_W = nimcp_calloc(enc_size, sizeof(float));
    wm->encoder_b = nimcp_calloc(wm->config.latent_dim, sizeof(float));
    wm->decoder_W = nimcp_calloc(enc_size, sizeof(float));
    wm->decoder_b = nimcp_calloc(wm->config.obs_dim, sizeof(float));

    if (!wm->encoder_W || !wm->encoder_b || !wm->decoder_W || !wm->decoder_b) {
        omni_wm_destroy(wm);
        return NULL;
    }

    /* Initialize encoder/decoder */
    for (uint32_t i = 0; i < enc_size; i++) {
        wm->encoder_W[i] = randn(&wm->rand_seed) * 0.01f;
        wm->decoder_W[i] = randn(&wm->rand_seed) * 0.01f;
    }

    /* Create replay buffer */
    wm->replay_buffer = replay_buffer_create(wm->config.replay_buffer_size);
    if (!wm->replay_buffer) {
        omni_wm_destroy(wm);
        return NULL;
    }

    /* Create initial RSSM state */
    wm->rssm_state = omni_wm_rssm_state_create(
        wm->config.rssm_h_dim,
        wm->config.rssm_z_dim
    );

    /* Create simple state wrapper */
    wm->current_state = omni_wm_state_create(wm->config.state_dim);

    /* Create mutex */
    wm->mutex = nimcp_mutex_create(NULL);
    wm->initialized = true;

    return wm;
}

omni_world_model_t* omni_wm_create_simple(uint32_t state_dim,
                                           uint32_t action_dim,
                                           uint32_t obs_dim) {
    omni_wm_config_t config;
    omni_wm_get_default_config(&config);
    config.state_dim = state_dim;
    config.action_dim = action_dim;
    config.obs_dim = obs_dim;
    config.rssm_h_dim = state_dim / 2 > 0 ? state_dim / 2 : 16;
    config.rssm_z_dim = state_dim / 4 > 0 ? state_dim / 4 : 8;
    config.latent_dim = state_dim;
    return omni_wm_create(&config);
}

void omni_wm_destroy(omni_world_model_t* wm) {
    if (!wm) return;

    dynamics_destroy(wm->forward_dynamics);
    dynamics_destroy(wm->backward_dynamics);
    dynamics_destroy(wm->lateral_dynamics);

    nimcp_free(wm->encoder_W);
    nimcp_free(wm->encoder_b);
    nimcp_free(wm->decoder_W);
    nimcp_free(wm->decoder_b);

    replay_buffer_destroy(wm->replay_buffer);

    omni_wm_state_destroy(wm->current_state);
    omni_wm_rssm_state_destroy(wm->rssm_state);

    if (wm->mutex) {
        nimcp_mutex_destroy(wm->mutex);
    }

    nimcp_free(wm);
}

/* ============================================================================
 * State Management
 * ============================================================================ */

omni_wm_state_t* omni_wm_state_create(uint32_t dim) {
    if (dim == 0 || dim > OMNI_WM_MAX_STATE_DIM) return NULL;

    omni_wm_state_t* state = nimcp_calloc(1, sizeof(omni_wm_state_t));
    if (!state) return NULL;

    state->values = nimcp_calloc(dim, sizeof(float));
    if (!state->values) {
        nimcp_free(state);
        return NULL;
    }

    state->dim = dim;
    state->uncertainty = 1.0f;
    state->timestamp = 0.0;
    state->level = 0;

    return state;
}

omni_wm_state_t* omni_wm_state_from_values(const float* values, uint32_t dim) {
    if (!values || dim == 0) return NULL;

    omni_wm_state_t* state = omni_wm_state_create(dim);
    if (!state) return NULL;

    memcpy(state->values, values, dim * sizeof(float));
    return state;
}

omni_wm_state_t* omni_wm_state_clone(const omni_wm_state_t* state) {
    if (!state) return NULL;

    omni_wm_state_t* clone = omni_wm_state_create(state->dim);
    if (!clone) return NULL;

    memcpy(clone->values, state->values, state->dim * sizeof(float));
    clone->uncertainty = state->uncertainty;
    clone->timestamp = state->timestamp;
    clone->level = state->level;

    return clone;
}

void omni_wm_state_destroy(omni_wm_state_t* state) {
    if (!state) return;
    nimcp_free(state->values);
    nimcp_free(state);
}

nimcp_error_t omni_wm_set_state(omni_world_model_t* wm,
                                 const omni_wm_state_t* state) {
    if (!wm || !state) return NIMCP_ERROR_INVALID_PARAM;

    if (wm->current_state) {
        omni_wm_state_destroy(wm->current_state);
    }
    wm->current_state = omni_wm_state_clone(state);

    return wm->current_state ? NIMCP_SUCCESS : NIMCP_ERROR_NO_MEMORY;
}

const omni_wm_state_t* omni_wm_get_state(const omni_world_model_t* wm) {
    return wm ? wm->current_state : NULL;
}

/* ============================================================================
 * RSSM State Management
 * ============================================================================ */

omni_wm_rssm_state_t* omni_wm_rssm_state_create(uint32_t h_dim, uint32_t z_dim) {
    if (h_dim == 0 || z_dim == 0) return NULL;

    omni_wm_rssm_state_t* state = nimcp_calloc(1, sizeof(omni_wm_rssm_state_t));
    if (!state) return NULL;

    state->h = nimcp_calloc(h_dim, sizeof(float));
    state->z = nimcp_calloc(z_dim, sizeof(float));
    state->z_mean = nimcp_calloc(z_dim, sizeof(float));
    state->z_std = nimcp_calloc(z_dim, sizeof(float));

    if (!state->h || !state->z || !state->z_mean || !state->z_std) {
        omni_wm_rssm_state_destroy(state);
        return NULL;
    }

    state->h_dim = h_dim;
    state->z_dim = z_dim;

    /* Initialize std to 1 */
    for (uint32_t i = 0; i < z_dim; i++) {
        state->z_std[i] = 1.0f;
    }

    return state;
}

omni_wm_rssm_state_t* omni_wm_rssm_state_clone(const omni_wm_rssm_state_t* state) {
    if (!state) return NULL;

    omni_wm_rssm_state_t* clone = omni_wm_rssm_state_create(state->h_dim, state->z_dim);
    if (!clone) return NULL;

    memcpy(clone->h, state->h, state->h_dim * sizeof(float));
    memcpy(clone->z, state->z, state->z_dim * sizeof(float));
    memcpy(clone->z_mean, state->z_mean, state->z_dim * sizeof(float));
    memcpy(clone->z_std, state->z_std, state->z_dim * sizeof(float));
    clone->timestamp = state->timestamp;

    return clone;
}

void omni_wm_rssm_state_destroy(omni_wm_rssm_state_t* state) {
    if (!state) return;
    nimcp_free(state->h);
    nimcp_free(state->z);
    nimcp_free(state->z_mean);
    nimcp_free(state->z_std);
    nimcp_free(state);
}

const omni_wm_rssm_state_t* omni_wm_get_rssm_state(const omni_world_model_t* wm) {
    return wm ? wm->rssm_state : NULL;
}

nimcp_error_t omni_wm_set_rssm_state(omni_world_model_t* wm,
                                      const omni_wm_rssm_state_t* state) {
    if (!wm || !state) return NIMCP_ERROR_INVALID_PARAM;

    if (wm->rssm_state) {
        omni_wm_rssm_state_destroy(wm->rssm_state);
    }
    wm->rssm_state = omni_wm_rssm_state_clone(state);

    return wm->rssm_state ? NIMCP_SUCCESS : NIMCP_ERROR_NO_MEMORY;
}

/* ============================================================================
 * RSSM Dynamics
 * ============================================================================ */

nimcp_error_t omni_wm_rssm_step(omni_world_model_t* wm,
                                 const omni_wm_rssm_state_t* state,
                                 const float* action,
                                 uint32_t action_dim,
                                 omni_wm_rssm_state_t* next_state) {
    if (!wm || !state || !action || !next_state) return NIMCP_ERROR_INVALID_PARAM;

    omni_wm_dynamics_t* dyn = wm->forward_dynamics;

    /* Concatenate input: [h, z, a] */
    uint32_t input_dim = state->h_dim + state->z_dim + action_dim;
    float* input = nimcp_calloc(input_dim, sizeof(float));
    if (!input) return NIMCP_ERROR_NO_MEMORY;

    memcpy(input, state->h, state->h_dim * sizeof(float));
    memcpy(input + state->h_dim, state->z, state->z_dim * sizeof(float));
    uint32_t copy_dim = action_dim < dyn->action_dim ? action_dim : dyn->action_dim;
    memcpy(input + state->h_dim + state->z_dim, action, copy_dim * sizeof(float));

    /* Compute next h: h' = tanh(W_h * [h, z, a] + b_h) */
    for (uint32_t i = 0; i < dyn->h_dim; i++) {
        float sum = dyn->b_h[i];
        for (uint32_t j = 0; j < input_dim && j < dyn->h_dim + dyn->z_dim + dyn->action_dim; j++) {
            sum += dyn->W_h[i * (dyn->h_dim + dyn->z_dim + dyn->action_dim) + j] * input[j];
        }
        next_state->h[i] = tanhf(sum);
    }

    /* Compute z prior: [mean, log_std] = W_z * h' + b_z */
    for (uint32_t i = 0; i < dyn->z_dim; i++) {
        float sum_mean = dyn->b_z[i];
        float sum_std = dyn->b_z[dyn->z_dim + i];
        for (uint32_t j = 0; j < dyn->h_dim; j++) {
            sum_mean += dyn->W_z[i * dyn->h_dim + j] * next_state->h[j];
            sum_std += dyn->W_z[(dyn->z_dim + i) * dyn->h_dim + j] * next_state->h[j];
        }
        next_state->z_mean[i] = sum_mean;
        next_state->z_std[i] = expf(sum_std) + 0.01f; /* Softplus-like */

        /* Sample z from N(mean, std) */
        next_state->z[i] = next_state->z_mean[i] +
                           next_state->z_std[i] * randn(&wm->rand_seed);
    }

    next_state->h_dim = dyn->h_dim;
    next_state->z_dim = dyn->z_dim;

    nimcp_free(input);

    wm->stats.forward_predictions++;
    return NIMCP_SUCCESS;
}

nimcp_error_t omni_wm_rssm_imagine(omni_world_model_t* wm,
                                    const omni_wm_rssm_state_t* initial_state,
                                    const float* const* actions,
                                    uint32_t horizon,
                                    omni_wm_rssm_state_t** trajectory) {
    if (!wm || !initial_state || !actions || !trajectory) return NIMCP_ERROR_INVALID_PARAM;
    if (horizon == 0 || horizon > OMNI_WM_MAX_HORIZON) return NIMCP_ERROR_INVALID_PARAM;

    /* First state is initial */
    trajectory[0] = omni_wm_rssm_state_clone(initial_state);
    if (!trajectory[0]) return NIMCP_ERROR_NO_MEMORY;

    /* Roll out imagination */
    for (uint32_t t = 1; t < horizon; t++) {
        trajectory[t] = omni_wm_rssm_state_create(
            wm->config.rssm_h_dim,
            wm->config.rssm_z_dim
        );
        if (!trajectory[t]) {
            for (uint32_t i = 0; i < t; i++) {
                omni_wm_rssm_state_destroy(trajectory[i]);
            }
            return NIMCP_ERROR_NO_MEMORY;
        }

        nimcp_error_t err = omni_wm_rssm_step(
            wm, trajectory[t-1], actions[t-1],
            wm->config.action_dim, trajectory[t]
        );
        if (err != NIMCP_SUCCESS) {
            for (uint32_t i = 0; i <= t; i++) {
                omni_wm_rssm_state_destroy(trajectory[i]);
            }
            return err;
        }
    }

    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Dynamics Prediction
 * ============================================================================ */

nimcp_error_t omni_wm_predict_forward(omni_world_model_t* wm,
                                       const float* action,
                                       uint32_t action_dim,
                                       omni_wm_transition_t* result) {
    if (!wm || !action || !result) return NIMCP_ERROR_INVALID_PARAM;

    /* Use RSSM if enabled */
    if (wm->config.use_rssm && wm->rssm_state) {
        omni_wm_rssm_state_t* next_rssm = omni_wm_rssm_state_create(
            wm->config.rssm_h_dim,
            wm->config.rssm_z_dim
        );
        if (!next_rssm) return NIMCP_ERROR_NO_MEMORY;

        nimcp_error_t err = omni_wm_rssm_step(wm, wm->rssm_state,
                                               action, action_dim, next_rssm);
        if (err != NIMCP_SUCCESS) {
            omni_wm_rssm_state_destroy(next_rssm);
            return err;
        }

        /* Convert RSSM state to simple state */
        result->next_state = omni_wm_state_create(wm->config.state_dim);
        if (!result->next_state) {
            omni_wm_rssm_state_destroy(next_rssm);
            return NIMCP_ERROR_NO_MEMORY;
        }

        /* Concatenate h and z into state */
        uint32_t copy_h = next_rssm->h_dim < result->next_state->dim ?
                          next_rssm->h_dim : result->next_state->dim;
        memcpy(result->next_state->values, next_rssm->h, copy_h * sizeof(float));

        if (copy_h < result->next_state->dim) {
            uint32_t copy_z = next_rssm->z_dim;
            if (copy_h + copy_z > result->next_state->dim) {
                copy_z = result->next_state->dim - copy_h;
            }
            memcpy(result->next_state->values + copy_h,
                   next_rssm->z, copy_z * sizeof(float));
        }

        /* Compute uncertainty from z_std */
        float uncertainty = 0.0f;
        for (uint32_t i = 0; i < next_rssm->z_dim; i++) {
            uncertainty += logf(next_rssm->z_std[i]);
        }
        result->next_state->uncertainty = uncertainty / next_rssm->z_dim;

        /* Update internal RSSM state */
        omni_wm_rssm_state_destroy(wm->rssm_state);
        wm->rssm_state = next_rssm;

        result->direction = OMNI_WM_DIR_FORWARD;
        result->log_prob = 0.0f; /* TODO: compute properly */
        result->prediction_error = 0.0f;

        return NIMCP_SUCCESS;
    }

    /* Simple linear prediction fallback */
    if (!wm->current_state) return NIMCP_ERROR_INVALID_PARAM;

    result->next_state = omni_wm_state_clone(wm->current_state);
    if (!result->next_state) return NIMCP_ERROR_NO_MEMORY;

    /* Simple dynamics: next = current + action_effect */
    uint32_t min_dim = action_dim < result->next_state->dim ?
                       action_dim : result->next_state->dim;
    for (uint32_t i = 0; i < min_dim; i++) {
        result->next_state->values[i] += action[i] * 0.1f;
    }

    result->direction = OMNI_WM_DIR_FORWARD;
    wm->stats.forward_predictions++;

    return NIMCP_SUCCESS;
}

nimcp_error_t omni_wm_infer_backward(omni_world_model_t* wm,
                                      const omni_wm_state_t* current_state,
                                      omni_wm_transition_t* result) {
    if (!wm || !current_state || !result) return NIMCP_ERROR_INVALID_PARAM;

    /* Create previous state estimate */
    result->next_state = omni_wm_state_clone(current_state);
    if (!result->next_state) return NIMCP_ERROR_NO_MEMORY;

    /* Backward dynamics: infer what action led here */
    /* This is an inverse problem - approximate solution */
    result->action_taken = nimcp_calloc(wm->config.action_dim, sizeof(float));
    if (!result->action_taken) {
        omni_wm_state_destroy(result->next_state);
        return NIMCP_ERROR_NO_MEMORY;
    }

    /* Simple heuristic: action proportional to state */
    for (uint32_t i = 0; i < wm->config.action_dim && i < current_state->dim; i++) {
        result->action_taken[i] = current_state->values[i] * 0.1f;
    }

    result->action_dim = wm->config.action_dim;
    result->direction = OMNI_WM_DIR_BACKWARD;

    wm->stats.backward_inferences++;
    return NIMCP_SUCCESS;
}

nimcp_error_t omni_wm_predict_lateral(omni_world_model_t* wm,
                                       const omni_wm_state_t* source_state,
                                       uint32_t target_modality,
                                       omni_wm_state_t* result) {
    if (!wm || !source_state || !result) return NIMCP_ERROR_INVALID_PARAM;
    if (!wm->config.enable_lateral) return NIMCP_ERROR_NOT_IMPLEMENTED;

    /* Cross-modal prediction using lateral dynamics */
    memset(result->values, 0, result->dim * sizeof(float));

    /* Simple linear mapping */
    uint32_t min_dim = source_state->dim < result->dim ?
                       source_state->dim : result->dim;
    for (uint32_t i = 0; i < min_dim; i++) {
        result->values[i] = source_state->values[i] * 0.9f;
    }

    result->uncertainty = source_state->uncertainty * 1.1f;
    wm->stats.lateral_predictions++;

    return NIMCP_SUCCESS;
}

nimcp_error_t omni_wm_predict_hierarchical(omni_world_model_t* wm,
                                            const omni_wm_state_t* state,
                                            uint32_t target_level,
                                            omni_wm_state_t* result) {
    if (!wm || !state || !result) return NIMCP_ERROR_INVALID_PARAM;
    if (!wm->config.enable_hierarchical) return NIMCP_ERROR_NOT_IMPLEMENTED;

    /* Hierarchical abstraction/concretization */
    uint32_t level_diff = target_level > state->level ?
                          target_level - state->level :
                          state->level - target_level;

    /* Abstract: pool/compress, Concrete: expand/detail */
    float scale = target_level > state->level ? 0.5f : 2.0f;

    memset(result->values, 0, result->dim * sizeof(float));
    uint32_t min_dim = state->dim < result->dim ? state->dim : result->dim;

    for (uint32_t i = 0; i < min_dim; i++) {
        result->values[i] = state->values[i] * scale;
    }

    result->level = target_level;
    result->uncertainty = state->uncertainty * (1.0f + 0.1f * level_diff);

    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Latent Encoding (JEPA-inspired)
 * ============================================================================ */

omni_wm_latent_t* omni_wm_latent_create(uint32_t dim) {
    if (dim == 0 || dim > OMNI_WM_MAX_LATENT_DIM) return NULL;

    omni_wm_latent_t* latent = nimcp_calloc(1, sizeof(omni_wm_latent_t));
    if (!latent) return NULL;

    latent->embedding = nimcp_calloc(dim, sizeof(float));
    if (!latent->embedding) {
        nimcp_free(latent);
        return NULL;
    }

    latent->dim = dim;
    return latent;
}

void omni_wm_latent_destroy(omni_wm_latent_t* latent) {
    if (!latent) return;
    nimcp_free(latent->embedding);
    nimcp_free(latent);
}

nimcp_error_t omni_wm_encode(omni_world_model_t* wm,
                              const float* observation,
                              uint32_t obs_dim,
                              omni_wm_latent_t* latent) {
    if (!wm || !observation || !latent) return NIMCP_ERROR_INVALID_PARAM;

    /* Linear encoding: latent = ReLU(W * obs + b) */
    for (uint32_t i = 0; i < latent->dim; i++) {
        float sum = wm->encoder_b[i];
        for (uint32_t j = 0; j < obs_dim && j < wm->config.obs_dim; j++) {
            sum += wm->encoder_W[i * wm->config.obs_dim + j] * observation[j];
        }
        latent->embedding[i] = sum > 0 ? sum : 0; /* ReLU */
    }

    /* Compute information content (approximate entropy) */
    float entropy = 0.0f;
    for (uint32_t i = 0; i < latent->dim; i++) {
        if (latent->embedding[i] > 0.01f) {
            entropy -= latent->embedding[i] * logf(latent->embedding[i]);
        }
    }
    latent->information_content = entropy;

    return NIMCP_SUCCESS;
}

nimcp_error_t omni_wm_decode(omni_world_model_t* wm,
                              const omni_wm_latent_t* latent,
                              float* observation,
                              uint32_t obs_dim) {
    if (!wm || !latent || !observation) return NIMCP_ERROR_INVALID_PARAM;

    /* Linear decoding: obs = W * latent + b */
    for (uint32_t i = 0; i < obs_dim && i < wm->config.obs_dim; i++) {
        float sum = wm->decoder_b[i];
        for (uint32_t j = 0; j < latent->dim && j < wm->config.latent_dim; j++) {
            sum += wm->decoder_W[i * wm->config.latent_dim + j] * latent->embedding[j];
        }
        observation[i] = sum;
    }

    return NIMCP_SUCCESS;
}

nimcp_error_t omni_wm_predict_latent(omni_world_model_t* wm,
                                      const omni_wm_latent_t* latent,
                                      const float* action,
                                      uint32_t action_dim,
                                      omni_wm_latent_t* predicted_latent) {
    if (!wm || !latent || !action || !predicted_latent) return NIMCP_ERROR_INVALID_PARAM;

    /* JEPA-style: predict in latent space */
    /* Simple dynamics: next_latent = f(latent, action) */
    for (uint32_t i = 0; i < predicted_latent->dim && i < latent->dim; i++) {
        float action_effect = 0.0f;
        if (i < action_dim) {
            action_effect = action[i] * 0.1f;
        }
        predicted_latent->embedding[i] = latent->embedding[i] + action_effect;
    }

    return NIMCP_SUCCESS;
}

/* ============================================================================
 * MDN (Mixture Density Network)
 * ============================================================================ */

omni_wm_mdn_prediction_t* omni_wm_mdn_create(uint32_t num_components, uint32_t dim) {
    if (num_components == 0 || dim == 0) return NULL;
    if (num_components > OMNI_WM_MAX_MDN_COMPONENTS) return NULL;

    omni_wm_mdn_prediction_t* pred = nimcp_calloc(1, sizeof(omni_wm_mdn_prediction_t));
    if (!pred) return NULL;

    pred->components = nimcp_calloc(num_components, sizeof(omni_wm_mdn_component_t));
    if (!pred->components) {
        nimcp_free(pred);
        return NULL;
    }

    for (uint32_t k = 0; k < num_components; k++) {
        pred->components[k].mean = nimcp_calloc(dim, sizeof(float));
        pred->components[k].std = nimcp_calloc(dim, sizeof(float));
        pred->components[k].dim = dim;
        pred->components[k].weight = 1.0f / num_components;

        if (!pred->components[k].mean || !pred->components[k].std) {
            omni_wm_mdn_destroy(pred);
            return NULL;
        }

        /* Initialize std to 1 */
        for (uint32_t i = 0; i < dim; i++) {
            pred->components[k].std[i] = 1.0f;
        }
    }

    pred->num_components = num_components;
    pred->dim = dim;

    return pred;
}

void omni_wm_mdn_destroy(omni_wm_mdn_prediction_t* pred) {
    if (!pred) return;
    for (uint32_t k = 0; k < pred->num_components; k++) {
        nimcp_free(pred->components[k].mean);
        nimcp_free(pred->components[k].std);
    }
    nimcp_free(pred->components);
    nimcp_free(pred);
}

nimcp_error_t omni_wm_predict_mdn(omni_world_model_t* wm,
                                   const omni_wm_state_t* state,
                                   const float* action,
                                   uint32_t action_dim,
                                   omni_wm_mdn_prediction_t* pred) {
    if (!wm || !state || !action || !pred) return NIMCP_ERROR_INVALID_PARAM;

    /* Generate mixture components based on state and action */
    for (uint32_t k = 0; k < pred->num_components; k++) {
        /* Each component gets slightly different prediction */
        float offset = (float)k * 0.1f - 0.05f * pred->num_components;

        for (uint32_t i = 0; i < pred->dim && i < state->dim; i++) {
            float action_effect = (i < action_dim) ? action[i] * 0.1f : 0.0f;
            pred->components[k].mean[i] = state->values[i] + action_effect + offset;
            pred->components[k].std[i] = 0.1f + 0.05f * k;
        }

        pred->components[k].weight = 1.0f / pred->num_components;
    }

    return NIMCP_SUCCESS;
}

nimcp_error_t omni_wm_mdn_sample(const omni_wm_mdn_prediction_t* pred,
                                  float* sample) {
    if (!pred || !sample) return NIMCP_ERROR_INVALID_PARAM;

    /* Select component based on weights */
    unsigned int seed = (unsigned int)time(NULL);
    float r = (float)rand_r(&seed) / RAND_MAX;
    float cumsum = 0.0f;
    uint32_t selected = 0;

    for (uint32_t k = 0; k < pred->num_components; k++) {
        cumsum += pred->components[k].weight;
        if (r <= cumsum) {
            selected = k;
            break;
        }
    }

    /* Sample from selected component */
    omni_wm_mdn_component_t* comp = &pred->components[selected];
    for (uint32_t i = 0; i < pred->dim; i++) {
        sample[i] = comp->mean[i] + comp->std[i] * randn(&seed);
    }

    return NIMCP_SUCCESS;
}

nimcp_error_t omni_wm_mdn_mode(const omni_wm_mdn_prediction_t* pred,
                                float* mode) {
    if (!pred || !mode) return NIMCP_ERROR_INVALID_PARAM;

    /* Find component with highest weight */
    uint32_t best = 0;
    float best_weight = pred->components[0].weight;

    for (uint32_t k = 1; k < pred->num_components; k++) {
        if (pred->components[k].weight > best_weight) {
            best_weight = pred->components[k].weight;
            best = k;
        }
    }

    /* Return mean of best component */
    memcpy(mode, pred->components[best].mean, pred->dim * sizeof(float));

    return NIMCP_SUCCESS;
}

float omni_wm_mdn_log_prob(const omni_wm_mdn_prediction_t* pred,
                            const float* value) {
    if (!pred || !value) return -FLT_MAX;

    /* Log probability under mixture: log(sum_k pi_k * N(x; mu_k, sigma_k)) */
    float max_log = -FLT_MAX;
    float* log_probs = nimcp_calloc(pred->num_components, sizeof(float));
    if (!log_probs) return -FLT_MAX;

    for (uint32_t k = 0; k < pred->num_components; k++) {
        float log_prob = logf(pred->components[k].weight);

        for (uint32_t i = 0; i < pred->dim; i++) {
            float diff = value[i] - pred->components[k].mean[i];
            float std = pred->components[k].std[i];
            log_prob -= 0.5f * (diff * diff) / (std * std);
            log_prob -= logf(std);
            log_prob -= 0.5f * logf(2.0f * M_PI);
        }

        log_probs[k] = log_prob;
        if (log_prob > max_log) max_log = log_prob;
    }

    /* Log-sum-exp for numerical stability */
    float sum = 0.0f;
    for (uint32_t k = 0; k < pred->num_components; k++) {
        sum += expf(log_probs[k] - max_log);
    }

    nimcp_free(log_probs);
    return max_log + logf(sum);
}

/* ============================================================================
 * Experience Replay
 * ============================================================================ */

omni_wm_experience_t* omni_wm_experience_create(uint32_t state_dim,
                                                  uint32_t action_dim,
                                                  uint32_t obs_dim) {
    omni_wm_experience_t* exp = nimcp_calloc(1, sizeof(omni_wm_experience_t));
    if (!exp) return NULL;

    exp->state = omni_wm_rssm_state_create(state_dim / 2, state_dim / 4);
    exp->next_state = omni_wm_rssm_state_create(state_dim / 2, state_dim / 4);
    exp->action = nimcp_calloc(action_dim, sizeof(float));
    exp->observation = nimcp_calloc(obs_dim, sizeof(float));

    if (!exp->state || !exp->next_state || !exp->action || !exp->observation) {
        omni_wm_experience_destroy(exp);
        return NULL;
    }

    exp->action_dim = action_dim;
    exp->obs_dim = obs_dim;

    return exp;
}

void omni_wm_experience_destroy(omni_wm_experience_t* exp) {
    if (!exp) return;
    omni_wm_rssm_state_destroy(exp->state);
    omni_wm_rssm_state_destroy(exp->next_state);
    nimcp_free(exp->action);
    nimcp_free(exp->observation);
    nimcp_free(exp);
}

nimcp_error_t omni_wm_add_experience(omni_world_model_t* wm,
                                      const omni_wm_experience_t* exp) {
    if (!wm || !exp) return NIMCP_ERROR_INVALID_PARAM;

    omni_wm_replay_buffer_t* buf = wm->replay_buffer;

    /* Clone experience */
    omni_wm_experience_t* clone = omni_wm_experience_create(
        wm->config.state_dim,
        exp->action_dim,
        exp->obs_dim
    );
    if (!clone) return NIMCP_ERROR_NO_MEMORY;

    memcpy(clone->action, exp->action, exp->action_dim * sizeof(float));
    clone->reward = exp->reward;
    clone->symlog_reward = wm->config.use_symlog_rewards ?
                           omni_wm_symlog(exp->reward) : exp->reward;
    clone->terminal = exp->terminal;
    clone->timestamp = exp->timestamp;

    if (exp->observation) {
        memcpy(clone->observation, exp->observation, exp->obs_dim * sizeof(float));
    }

    /* Add to circular buffer */
    if (buf->size < buf->capacity) {
        buf->experiences[buf->size] = clone;
        buf->priorities[buf->size] = 1.0f;
        buf->size++;
    } else {
        /* Overwrite oldest */
        omni_wm_experience_destroy(buf->experiences[buf->head]);
        buf->experiences[buf->head] = clone;
        buf->priorities[buf->head] = 1.0f;
    }
    buf->head = (buf->head + 1) % buf->capacity;

    return NIMCP_SUCCESS;
}

uint32_t omni_wm_sample_experiences(omni_world_model_t* wm,
                                     omni_wm_experience_t** batch,
                                     uint32_t batch_size) {
    if (!wm || !batch || batch_size == 0) return 0;

    omni_wm_replay_buffer_t* buf = wm->replay_buffer;
    uint32_t actual_size = batch_size < buf->size ? batch_size : buf->size;

    /* Simple uniform sampling */
    for (uint32_t i = 0; i < actual_size; i++) {
        uint32_t idx = rand_r(&wm->rand_seed) % buf->size;
        batch[i] = buf->experiences[idx];
    }

    return actual_size;
}

uint32_t omni_wm_get_replay_size(const omni_world_model_t* wm) {
    return wm && wm->replay_buffer ? wm->replay_buffer->size : 0;
}

nimcp_error_t omni_wm_clear_replay(omni_world_model_t* wm) {
    if (!wm || !wm->replay_buffer) return NIMCP_ERROR_INVALID_PARAM;

    for (uint32_t i = 0; i < wm->replay_buffer->size; i++) {
        omni_wm_experience_destroy(wm->replay_buffer->experiences[i]);
        wm->replay_buffer->experiences[i] = NULL;
    }
    wm->replay_buffer->size = 0;
    wm->replay_buffer->head = 0;

    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Learning
 * ============================================================================ */

nimcp_error_t omni_wm_update(omni_world_model_t* wm,
                              const omni_wm_state_t* state,
                              const float* action,
                              uint32_t action_dim,
                              const omni_wm_state_t* next_state,
                              float reward) {
    if (!wm || !state || !action || !next_state) return NIMCP_ERROR_INVALID_PARAM;

    /* Compute prediction error */
    omni_wm_transition_t pred;
    memset(&pred, 0, sizeof(pred));

    nimcp_error_t err = omni_wm_predict_forward(wm, action, action_dim, &pred);
    if (err != NIMCP_SUCCESS) return err;

    float error = 0.0f;
    uint32_t min_dim = pred.next_state->dim < next_state->dim ?
                       pred.next_state->dim : next_state->dim;

    for (uint32_t i = 0; i < min_dim; i++) {
        float diff = pred.next_state->values[i] - next_state->values[i];
        error += diff * diff;
    }
    error = sqrtf(error / min_dim);

    /* Update running average of prediction error */
    float alpha = 0.01f;
    wm->stats.mean_prediction_error = (1.0f - alpha) * wm->stats.mean_prediction_error +
                                       alpha * error;

    /* Simple gradient update on dynamics weights */
    float lr = wm->config.learning_rate;
    omni_wm_dynamics_t* dyn = wm->forward_dynamics;

    /* Update bias towards correct prediction */
    for (uint32_t i = 0; i < dyn->h_dim && i < min_dim; i++) {
        float grad = next_state->values[i] - pred.next_state->values[i];
        dyn->b_h[i] += lr * grad;
    }

    omni_wm_state_destroy(pred.next_state);
    wm->stats.model_updates++;

    return NIMCP_SUCCESS;
}

nimcp_error_t omni_wm_dream(omni_world_model_t* wm,
                             uint32_t num_episodes,
                             uint32_t episode_length) {
    if (!wm) return NIMCP_ERROR_INVALID_PARAM;
    if (!wm->config.enable_dreaming) return NIMCP_ERROR_NOT_IMPLEMENTED;

    for (uint32_t ep = 0; ep < num_episodes; ep++) {
        /* Start from random experience or current state */
        omni_wm_rssm_state_t* dream_state = NULL;

        if (wm->replay_buffer->size > 0) {
            uint32_t idx = rand_r(&wm->rand_seed) % wm->replay_buffer->size;
            dream_state = omni_wm_rssm_state_clone(
                wm->replay_buffer->experiences[idx]->state
            );
        } else if (wm->rssm_state) {
            dream_state = omni_wm_rssm_state_clone(wm->rssm_state);
        } else {
            continue;
        }

        if (!dream_state) continue;

        /* Dream rollout */
        for (uint32_t t = 0; t < episode_length; t++) {
            /* Generate random action with noise */
            float* action = nimcp_calloc(wm->config.action_dim, sizeof(float));
            if (!action) break;

            for (uint32_t i = 0; i < wm->config.action_dim; i++) {
                action[i] = randn(&wm->rand_seed) * wm->config.imagination_noise;
            }

            /* Step in dream */
            omni_wm_rssm_state_t* next_dream = omni_wm_rssm_state_create(
                wm->config.rssm_h_dim,
                wm->config.rssm_z_dim
            );
            if (!next_dream) {
                nimcp_free(action);
                break;
            }

            omni_wm_rssm_step(wm, dream_state, action, wm->config.action_dim, next_dream);

            omni_wm_rssm_state_destroy(dream_state);
            dream_state = next_dream;
            nimcp_free(action);
        }

        omni_wm_rssm_state_destroy(dream_state);
    }

    return NIMCP_SUCCESS;
}

nimcp_error_t omni_wm_set_learning_rate(omni_world_model_t* wm, float learning_rate) {
    if (!wm) return NIMCP_ERROR_INVALID_PARAM;
    wm->config.learning_rate = learning_rate;
    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Counterfactual Reasoning
 * ============================================================================ */

omni_wm_counterfactual_query_t* omni_wm_cf_query_create(
    omni_wm_counterfactual_type_t type,
    const omni_wm_state_t* initial_state,
    const float* hypothetical_action,
    uint32_t action_dim,
    uint32_t horizon) {

    if (!initial_state || !hypothetical_action) return NULL;

    omni_wm_counterfactual_query_t* query = nimcp_calloc(1, sizeof(omni_wm_counterfactual_query_t));
    if (!query) return NULL;

    query->type = type;
    query->initial_state = omni_wm_state_clone(initial_state);
    query->hypothetical_action = nimcp_calloc(action_dim, sizeof(float));

    if (!query->initial_state || !query->hypothetical_action) {
        omni_wm_cf_query_destroy(query);
        return NULL;
    }

    memcpy(query->hypothetical_action, hypothetical_action, action_dim * sizeof(float));
    query->action_dim = action_dim;
    query->horizon = horizon;

    return query;
}

void omni_wm_cf_query_destroy(omni_wm_counterfactual_query_t* query) {
    if (!query) return;
    omni_wm_state_destroy(query->initial_state);
    nimcp_free(query->hypothetical_action);
    nimcp_free(query->context);
    nimcp_free(query);
}

nimcp_error_t omni_wm_counterfactual(omni_world_model_t* wm,
                                      const omni_wm_counterfactual_query_t* query,
                                      omni_wm_counterfactual_result_t* result) {
    if (!wm || !query || !result) return NIMCP_ERROR_INVALID_PARAM;

    memset(result, 0, sizeof(omni_wm_counterfactual_result_t));

    /* Allocate trajectory */
    result->trajectory = nimcp_calloc(query->horizon, sizeof(omni_wm_state_t*));
    if (!result->trajectory) return NIMCP_ERROR_NO_MEMORY;

    result->trajectory[0] = omni_wm_state_clone(query->initial_state);
    if (!result->trajectory[0]) {
        nimcp_free(result->trajectory);
        return NIMCP_ERROR_NO_MEMORY;
    }

    /* Rollout with hypothetical action */
    float cumulative_reward = 0.0f;

    for (uint32_t t = 1; t < query->horizon; t++) {
        omni_wm_transition_t trans;
        memset(&trans, 0, sizeof(trans));

        /* Use query action for first step, then continue with zeros */
        const float* action = (t == 1) ? query->hypothetical_action : NULL;
        uint32_t adim = (t == 1) ? query->action_dim : 0;

        if (!action) {
            action = nimcp_calloc(wm->config.action_dim, sizeof(float));
            adim = wm->config.action_dim;
        }

        /* Temporarily set state for prediction */
        omni_wm_state_t* old_state = wm->current_state;
        wm->current_state = result->trajectory[t-1];

        nimcp_error_t err = omni_wm_predict_forward(wm, action, adim, &trans);

        wm->current_state = old_state;

        if (t != 1) nimcp_free((void*)action);

        if (err != NIMCP_SUCCESS || !trans.next_state) {
            /* Cleanup on error */
            for (uint32_t i = 0; i < t; i++) {
                omni_wm_state_destroy(result->trajectory[i]);
            }
            nimcp_free(result->trajectory);
            return err;
        }

        result->trajectory[t] = trans.next_state;

        /* Estimate reward (simple heuristic) */
        float state_magnitude = 0.0f;
        for (uint32_t i = 0; i < trans.next_state->dim; i++) {
            state_magnitude += trans.next_state->values[i] * trans.next_state->values[i];
        }
        cumulative_reward += sqrtf(state_magnitude) * 0.1f;
    }

    result->trajectory_len = query->horizon;
    result->expected_reward = cumulative_reward;
    result->confidence = 0.8f / (1.0f + 0.1f * query->horizon);
    result->divergence = 0.0f; /* TODO: compute vs actual */

    wm->stats.counterfactual_queries++;
    return NIMCP_SUCCESS;
}

nimcp_error_t omni_wm_what_if(omni_world_model_t* wm,
                               const float* action,
                               uint32_t action_dim,
                               uint32_t horizon,
                               omni_wm_counterfactual_result_t* result) {
    if (!wm || !action || !result) return NIMCP_ERROR_INVALID_PARAM;
    if (!wm->current_state) return NIMCP_ERROR_INVALID_PARAM;

    omni_wm_counterfactual_query_t* query = omni_wm_cf_query_create(
        OMNI_WM_CF_ACTION,
        wm->current_state,
        action,
        action_dim,
        horizon
    );
    if (!query) return NIMCP_ERROR_NO_MEMORY;

    nimcp_error_t err = omni_wm_counterfactual(wm, query, result);
    omni_wm_cf_query_destroy(query);

    return err;
}

void omni_wm_cf_result_destroy(omni_wm_counterfactual_result_t* result) {
    if (!result) return;
    for (uint32_t i = 0; i < result->trajectory_len; i++) {
        omni_wm_state_destroy(result->trajectory[i]);
    }
    nimcp_free(result->trajectory);
    nimcp_free(result->predicted_obs);
}

/* ============================================================================
 * Policy Rollouts
 * ============================================================================ */

omni_wm_rollout_t* omni_wm_rollout_create(uint32_t max_length,
                                           uint32_t state_dim,
                                           uint32_t action_dim,
                                           uint32_t obs_dim) {
    if (max_length == 0 || max_length > OMNI_WM_MAX_HORIZON) return NULL;

    omni_wm_rollout_t* rollout = nimcp_calloc(1, sizeof(omni_wm_rollout_t));
    if (!rollout) return NULL;

    rollout->states = nimcp_calloc(max_length, sizeof(omni_wm_state_t*));
    rollout->actions = nimcp_calloc(max_length, sizeof(float*));
    rollout->observations = nimcp_calloc(max_length, sizeof(float*));
    rollout->rewards = nimcp_calloc(max_length, sizeof(float));

    if (!rollout->states || !rollout->actions ||
        !rollout->observations || !rollout->rewards) {
        omni_wm_rollout_destroy(rollout);
        return NULL;
    }

    return rollout;
}

void omni_wm_rollout_destroy(omni_wm_rollout_t* rollout) {
    if (!rollout) return;

    for (uint32_t i = 0; i < rollout->length; i++) {
        omni_wm_state_destroy(rollout->states[i]);
        nimcp_free(rollout->actions[i]);
        nimcp_free(rollout->observations[i]);
    }

    nimcp_free(rollout->states);
    nimcp_free(rollout->actions);
    nimcp_free(rollout->observations);
    nimcp_free(rollout->rewards);
    nimcp_free(rollout);
}

nimcp_error_t omni_wm_rollout(omni_world_model_t* wm,
                               omni_wm_policy_fn policy,
                               uint32_t horizon,
                               omni_wm_rollout_t* rollout,
                               void* user_data) {
    if (!wm || !policy || !rollout) return NIMCP_ERROR_INVALID_PARAM;
    if (!wm->current_state) return NIMCP_ERROR_INVALID_PARAM;

    rollout->states[0] = omni_wm_state_clone(wm->current_state);
    if (!rollout->states[0]) return NIMCP_ERROR_NO_MEMORY;

    float total_reward = 0.0f;

    for (uint32_t t = 0; t < horizon - 1; t++) {
        /* Get action from policy */
        float* action = nimcp_calloc(wm->config.action_dim, sizeof(float));
        if (!action) break;

        policy(rollout->states[t], action, user_data);
        rollout->actions[t] = action;

        /* Predict next state */
        omni_wm_transition_t trans;
        memset(&trans, 0, sizeof(trans));

        omni_wm_state_t* old = wm->current_state;
        wm->current_state = rollout->states[t];

        nimcp_error_t err = omni_wm_predict_forward(wm, action,
                                                     wm->config.action_dim, &trans);
        wm->current_state = old;

        if (err != NIMCP_SUCCESS || !trans.next_state) break;

        rollout->states[t + 1] = trans.next_state;

        /* Estimate reward */
        float reward = 0.0f;
        for (uint32_t i = 0; i < trans.next_state->dim; i++) {
            reward += trans.next_state->values[i] * 0.01f;
        }
        rollout->rewards[t] = reward;
        total_reward += reward;

        rollout->length = t + 2;
    }

    rollout->total_reward = total_reward;
    wm->stats.rollouts_completed++;

    return NIMCP_SUCCESS;
}

float omni_wm_evaluate_efe(omni_world_model_t* wm,
                            const omni_wm_rollout_t* rollout,
                            const float* preferred_obs,
                            uint32_t obs_dim) {
    if (!wm || !rollout || !preferred_obs) return FLT_MAX;

    float efe = 0.0f;
    float gamma = wm->config.discount_factor;
    float discount = 1.0f;

    for (uint32_t t = 0; t < rollout->length; t++) {
        if (!rollout->states[t]) continue;

        /* Risk: KL[q(o|pi) || p(o)] - distance from preferred */
        float risk = 0.0f;
        uint32_t min_dim = rollout->states[t]->dim < obs_dim ?
                           rollout->states[t]->dim : obs_dim;

        for (uint32_t i = 0; i < min_dim; i++) {
            float diff = rollout->states[t]->values[i] - preferred_obs[i];
            risk += diff * diff;
        }
        risk = sqrtf(risk / min_dim);

        /* Ambiguity: uncertainty in state */
        float ambiguity = rollout->states[t]->uncertainty;

        efe += discount * (risk + 0.5f * ambiguity);
        discount *= gamma;
    }

    return efe;
}

/* ============================================================================
 * Observation Prediction
 * ============================================================================ */

nimcp_error_t omni_wm_predict_observations(omni_world_model_t* wm,
                                            const omni_wm_state_t* state,
                                            float* predicted_obs,
                                            uint32_t obs_dim) {
    if (!wm || !state || !predicted_obs) return NIMCP_ERROR_INVALID_PARAM;

    /* Use decoder to predict observations from state */
    omni_wm_latent_t latent;
    latent.embedding = state->values;
    latent.dim = state->dim;

    return omni_wm_decode(wm, &latent, predicted_obs, obs_dim);
}

nimcp_error_t omni_wm_infer_state(omni_world_model_t* wm,
                                   const float* observations,
                                   uint32_t obs_dim,
                                   omni_wm_state_t* inferred_state) {
    if (!wm || !observations || !inferred_state) return NIMCP_ERROR_INVALID_PARAM;

    /* Use encoder to infer state from observations */
    omni_wm_latent_t* latent = omni_wm_latent_create(inferred_state->dim);
    if (!latent) return NIMCP_ERROR_NO_MEMORY;

    nimcp_error_t err = omni_wm_encode(wm, observations, obs_dim, latent);
    if (err == NIMCP_SUCCESS) {
        memcpy(inferred_state->values, latent->embedding,
               inferred_state->dim * sizeof(float));
        inferred_state->uncertainty = 1.0f / (1.0f + latent->information_content);
    }

    omni_wm_latent_destroy(latent);
    return err;
}

/* ============================================================================
 * Active Inference Integration
 * ============================================================================ */

nimcp_error_t omni_wm_connect_active_inference(omni_world_model_t* wm,
                                                struct omni_active_inference* ai) {
    if (!wm) return NIMCP_ERROR_INVALID_PARAM;
    wm->ai_ctx = ai;
    return NIMCP_SUCCESS;
}

float omni_wm_evaluate_policy(omni_world_model_t* wm,
                               const float* policy_actions,
                               uint32_t horizon,
                               const float* preferred_obs,
                               uint32_t obs_dim) {
    if (!wm || !policy_actions || !preferred_obs) return FLT_MAX;
    if (!wm->current_state) return FLT_MAX;

    float efe = 0.0f;
    float gamma = wm->config.discount_factor;
    float discount = 1.0f;

    omni_wm_state_t* state = omni_wm_state_clone(wm->current_state);
    if (!state) return FLT_MAX;

    for (uint32_t t = 0; t < horizon; t++) {
        /* Get action for this timestep */
        const float* action = policy_actions + t * wm->config.action_dim;

        /* Predict next state */
        omni_wm_transition_t trans;
        memset(&trans, 0, sizeof(trans));

        omni_wm_state_t* old = wm->current_state;
        wm->current_state = state;

        nimcp_error_t err = omni_wm_predict_forward(wm, action,
                                                     wm->config.action_dim, &trans);
        wm->current_state = old;

        if (err != NIMCP_SUCCESS || !trans.next_state) {
            omni_wm_state_destroy(state);
            return FLT_MAX;
        }

        /* Compute EFE components */
        float risk = 0.0f;
        uint32_t min_dim = trans.next_state->dim < obs_dim ?
                           trans.next_state->dim : obs_dim;

        for (uint32_t i = 0; i < min_dim; i++) {
            float diff = trans.next_state->values[i] - preferred_obs[i];
            risk += diff * diff;
        }
        risk = sqrtf(risk / min_dim);

        float ambiguity = trans.next_state->uncertainty;

        efe += discount * (risk + 0.5f * ambiguity);
        discount *= gamma;

        omni_wm_state_destroy(state);
        state = trans.next_state;
    }

    omni_wm_state_destroy(state);
    return efe;
}

/* ============================================================================
 * Statistics
 * ============================================================================ */

nimcp_error_t omni_wm_get_stats(const omni_world_model_t* wm,
                                 omni_wm_stats_t* stats) {
    if (!wm || !stats) return NIMCP_ERROR_INVALID_PARAM;
    memcpy(stats, &wm->stats, sizeof(omni_wm_stats_t));
    return NIMCP_SUCCESS;
}

nimcp_error_t omni_wm_reset_stats(omni_world_model_t* wm) {
    if (!wm) return NIMCP_ERROR_INVALID_PARAM;
    memset(&wm->stats, 0, sizeof(omni_wm_stats_t));
    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Utility Functions
 * ============================================================================ */

const char* omni_wm_direction_to_string(omni_wm_direction_t dir) {
    switch (dir) {
        case OMNI_WM_DIR_FORWARD:     return "forward";
        case OMNI_WM_DIR_BACKWARD:    return "backward";
        case OMNI_WM_DIR_LATERAL:     return "lateral";
        case OMNI_WM_DIR_HIERARCHICAL: return "hierarchical";
        default:                       return "unknown";
    }
}

const char* omni_wm_learn_mode_to_string(omni_wm_learn_mode_t mode) {
    switch (mode) {
        case OMNI_WM_LEARN_ONLINE:   return "online";
        case OMNI_WM_LEARN_BATCH:    return "batch";
        case OMNI_WM_LEARN_REPLAY:   return "replay";
        case OMNI_WM_LEARN_DREAMING: return "dreaming";
        default:                      return "unknown";
    }
}

const char* omni_wm_cf_type_to_string(omni_wm_counterfactual_type_t type) {
    switch (type) {
        case OMNI_WM_CF_ACTION:  return "action";
        case OMNI_WM_CF_STATE:   return "state";
        case OMNI_WM_CF_CONTEXT: return "context";
        case OMNI_WM_CF_GOAL:    return "goal";
        default:                 return "unknown";
    }
}

/* ============================================================================
 * Bio-Async Integration
 * ============================================================================ */

nimcp_error_t omni_wm_connect_bio_async(omni_world_model_t* wm) {
    if (!wm) return NIMCP_ERROR_INVALID_PARAM;
    /* TODO: Register message handlers */
    return NIMCP_SUCCESS;
}

nimcp_error_t omni_wm_disconnect_bio_async(omni_world_model_t* wm) {
    if (!wm) return NIMCP_ERROR_INVALID_PARAM;
    /* TODO: Unregister message handlers */
    return NIMCP_SUCCESS;
}
