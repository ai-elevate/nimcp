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
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <float.h>
#include <time.h>
#include <stdio.h>

//=============================================================================
#include <stddef.h>  /* for NULL */
// Health Agent Integration (Phase 8: System-Wide Health Integration)
//=============================================================================
struct nimcp_health_agent;
typedef struct nimcp_health_agent nimcp_health_agent_t;
extern void nimcp_health_agent_heartbeat_ex(nimcp_health_agent_t* agent,
                                             const char* operation,
                                             float progress);

/** Global health agent for omni_world_model module */
static nimcp_health_agent_t* g_omni_world_model_health_agent = NULL;

/**
 * @brief Set health agent for omni_world_model heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
void omni_world_model_set_health_agent(nimcp_health_agent_t* agent) {
    g_omni_world_model_health_agent = agent;
}

/** @brief Send heartbeat from omni_world_model module */
static inline void omni_world_model_heartbeat(const char* operation, float progress) {
    if (g_omni_world_model_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_omni_world_model_health_agent, operation, progress);
    }
}

/** @brief Send heartbeat from omni_world_model module (instance-level) */
static inline void omni_world_model_heartbeat_instance(
    nimcp_health_agent_t* instance_agent, const char* operation, float progress)
{
    if (g_omni_world_model_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_omni_world_model_health_agent, operation, progress);
    }
    if (instance_agent && instance_agent != g_omni_world_model_health_agent) {
        nimcp_health_agent_heartbeat_ex(instance_agent, operation, progress);
    }
}

/** @brief Instance-level health agent (global fallback for non-bridge) */
static nimcp_health_agent_t* g_omni_world_model_instance_health_agent = NULL;

void omni_world_model_set_instance_health_agent(void* ctx, nimcp_health_agent_t* agent) {
    (void)ctx;
    g_omni_world_model_instance_health_agent = agent;
}

/* ============================================================================
 * Phase 8: Instance-level Training Functions
 * ============================================================================ */

int omni_world_model_training_begin(void* ctx) {
    if (!ctx) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "omni_world_model_training_begin: NULL argument");
        return -1;
    }
    omni_world_model_heartbeat_instance(g_omni_world_model_health_agent, "training_begin", 0.0f);
    (void)ctx;
    return 0;
}

int omni_world_model_training_step(void* ctx, float progress) {
    if (!ctx) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "omni_world_model_training_step: NULL argument");
        return -1;
    }
    if (progress < 0.0f) progress = 0.0f;
    if (progress > 1.0f) progress = 1.0f;
    omni_world_model_heartbeat_instance(g_omni_world_model_health_agent, "training_step", progress);
    (void)ctx;
    return 0;
}

int omni_world_model_training_end(void* ctx) {
    if (!ctx) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "omni_world_model_training_end: NULL argument");
        return -1;
    }
    omni_world_model_heartbeat_instance(g_omni_world_model_health_agent, "training_end", 1.0f);
    (void)ctx;
    return 0;
}

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ============================================================================
 * Internal Structures
 * ============================================================================ */

/** Maximum checkpoints stored in memory */
#define OMNI_WM_MAX_CHECKPOINTS 32

/**
 * @brief Checkpoint storage structure
 */
typedef struct {
    omni_wm_checkpoint_t checkpoints[OMNI_WM_MAX_CHECKPOINTS];
    uint32_t count;
    uint64_t next_id;
} omni_wm_checkpoint_store_t;

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

    /* Checkpoint storage */
    omni_wm_checkpoint_store_t* checkpoint_store;

    /* Bio-async integration */
    bio_module_context_t bio_ctx;
    bool bio_async_connected;

    /* Initialized flag */
    bool initialized;
};

/* ============================================================================
 * Symlog Transformation (DreamerV3)
 * ============================================================================ */

float omni_wm_symlog(float x) {
    /* symlog(x) = sign(x) * ln(|x| + 1) */
    /* Phase 8: Heartbeat at operation start */
    omni_world_model_heartbeat("omni_world_m_omni_wm_symlog", 0.0f);


    if (x >= 0.0f) {
        return logf(x + 1.0f);
    } else {
        return -logf(-x + 1.0f);
    }
}

float omni_wm_symexp(float x) {
    /* symexp(x) = sign(x) * (exp(|x|) - 1) */
    /* Phase 8: Heartbeat at operation start */
    omni_world_model_heartbeat("omni_world_m_omni_wm_symexp", 0.0f);


    if (x >= 0.0f) {
        return expf(x) - 1.0f;
    } else {
        return -(expf(-x) - 1.0f);
    }
}

void omni_wm_symlog_array(const float* input, float* output, uint32_t size) {
    if (!input || !output) return;
    /* Phase 8: Heartbeat at operation start */
    omni_world_model_heartbeat("omni_world_m_omni_wm_symlog_array", 0.0f);


    for (uint32_t i = 0; i < size; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && size > 256) {
            omni_world_model_heartbeat("omni_world_m_loop",
                             (float)(i + 1) / (float)size);
        }

        output[i] = omni_wm_symlog(input[i]);
    }
}

void omni_wm_symexp_array(const float* input, float* output, uint32_t size) {
    if (!input || !output) return;
    /* Phase 8: Heartbeat at operation start */
    omni_world_model_heartbeat("omni_world_m_omni_wm_symexp_array", 0.0f);


    for (uint32_t i = 0; i < size; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && size > 256) {
            omni_world_model_heartbeat("omni_world_m_loop",
                             (float)(i + 1) / (float)size);
        }

        output[i] = omni_wm_symexp(input[i]);
    }
}

/* ============================================================================
 * Internal Helpers - Forward Declarations
 * ============================================================================ */

/* Checkpoint management (defined later in file) */
static void checkpoint_store_destroy_internal(omni_wm_checkpoint_store_t* store);

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
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && buf->size > 256) {
            omni_world_model_heartbeat("omni_world_m_loop",
                             (float)(i + 1) / (float)buf->size);
        }

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
    /* Phase 8: Heartbeat at operation start */
    omni_world_model_heartbeat("omni_world_m_omni_wm_get_default_", 0.0f);


    NIMCP_CHECK_THROW(config, NIMCP_ERROR_INVALID_PARAM, "config is NULL");

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
    /* Validate config dimensions if provided */
    /* Phase 8: Heartbeat at operation start */
    omni_world_model_heartbeat("omni_world_m_omni_wm_create", 0.0f);


    if (config) {
        if (config->state_dim == 0 || config->action_dim == 0 || config->obs_dim == 0) {
            NIMCP_THROW(NIMCP_ERROR_INVALID_PARAM, "omni_wm_create: zero dimension in config");
            return NULL;
        }
        if (config->state_dim > OMNI_WM_MAX_STATE_DIM ||
            config->action_dim > OMNI_WM_MAX_ACTION_DIM ||
            config->obs_dim > OMNI_WM_MAX_OBS_DIM) {
            NIMCP_THROW(NIMCP_ERROR_INVALID_PARAM,
                "omni_wm_create: dimension exceeds maximum (state=%u, action=%u, obs=%u)",
                config->state_dim, config->action_dim, config->obs_dim);
            return NULL;
        }
    }

    omni_world_model_t* wm = nimcp_calloc(1, sizeof(omni_world_model_t));
    if (!wm) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "omni_wm_create: failed to allocate world model");
        return NULL;
    }

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
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "omni_wm_create: failed to create forward dynamics");
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
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "omni_wm_create: failed to create backward dynamics");
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
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && enc_size > 256) {
            omni_world_model_heartbeat("omni_world_m_loop",
                             (float)(i + 1) / (float)enc_size);
        }

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
    /* Phase 8: Heartbeat at operation start */
    omni_world_model_heartbeat("omni_world_m_omni_wm_create_simpl", 0.0f);


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

    /* Phase 8: Heartbeat at operation start */
    omni_world_model_heartbeat("omni_world_m_omni_wm_destroy", 0.0f);


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

    /* Clean up checkpoint store */
    checkpoint_store_destroy_internal(wm->checkpoint_store);

    if (wm->mutex) {
        nimcp_mutex_free(wm->mutex);
    }

    nimcp_free(wm);
}

/* ============================================================================
 * State Management
 * ============================================================================ */

omni_wm_state_t* omni_wm_state_create(uint32_t dim) {
    if (dim == 0 || dim > OMNI_WM_MAX_STATE_DIM) return NULL;

    /* Phase 8: Heartbeat at operation start */
    omni_world_model_heartbeat("omni_world_m_omni_wm_state_create", 0.0f);


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

    /* Phase 8: Heartbeat at operation start */
    omni_world_model_heartbeat("omni_world_m_omni_wm_state_from_v", 0.0f);


    omni_wm_state_t* state = omni_wm_state_create(dim);
    if (!state) return NULL;

    memcpy(state->values, values, dim * sizeof(float));
    return state;
}

omni_wm_state_t* omni_wm_state_clone(const omni_wm_state_t* state) {
    if (!state) return NULL;

    /* Phase 8: Heartbeat at operation start */
    omni_world_model_heartbeat("omni_world_m_omni_wm_state_clone", 0.0f);


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
    /* Phase 8: Heartbeat at operation start */
    omni_world_model_heartbeat("omni_world_m_omni_wm_state_destro", 0.0f);


    nimcp_free(state->values);
    nimcp_free(state);
}

nimcp_error_t omni_wm_set_state(omni_world_model_t* wm,
                                 const omni_wm_state_t* state) {
    /* Phase 8: Heartbeat at operation start */
    omni_world_model_heartbeat("omni_world_m_omni_wm_set_state", 0.0f);


    NIMCP_CHECK_THROW(wm, NIMCP_ERROR_INVALID_PARAM, "world model is NULL");
    NIMCP_CHECK_THROW(state, NIMCP_ERROR_INVALID_PARAM, "state is NULL");

    if (wm->current_state) {
        omni_wm_state_destroy(wm->current_state);
    }
    wm->current_state = omni_wm_state_clone(state);

    return wm->current_state ? NIMCP_SUCCESS : NIMCP_ERROR_NO_MEMORY;
}

const omni_wm_state_t* omni_wm_get_state(const omni_world_model_t* wm) {
    /* Phase 8: Heartbeat at operation start */
    omni_world_model_heartbeat("omni_world_m_omni_wm_get_state", 0.0f);


    return wm ? wm->current_state : NULL;
}

/* ============================================================================
 * RSSM State Management
 * ============================================================================ */

omni_wm_rssm_state_t* omni_wm_rssm_state_create(uint32_t h_dim, uint32_t z_dim) {
    if (h_dim == 0 || z_dim == 0) return NULL;

    /* Phase 8: Heartbeat at operation start */
    omni_world_model_heartbeat("omni_world_m_omni_wm_rssm_state_c", 0.0f);


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
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && z_dim > 256) {
            omni_world_model_heartbeat("omni_world_m_loop",
                             (float)(i + 1) / (float)z_dim);
        }

        state->z_std[i] = 1.0f;
    }

    return state;
}

omni_wm_rssm_state_t* omni_wm_rssm_state_clone(const omni_wm_rssm_state_t* state) {
    if (!state) return NULL;

    /* Phase 8: Heartbeat at operation start */
    omni_world_model_heartbeat("omni_world_m_omni_wm_rssm_state_c", 0.0f);


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
    /* Phase 8: Heartbeat at operation start */
    omni_world_model_heartbeat("omni_world_m_omni_wm_rssm_state_d", 0.0f);


    nimcp_free(state->h);
    nimcp_free(state->z);
    nimcp_free(state->z_mean);
    nimcp_free(state->z_std);
    nimcp_free(state);
}

const omni_wm_rssm_state_t* omni_wm_get_rssm_state(const omni_world_model_t* wm) {
    /* Phase 8: Heartbeat at operation start */
    omni_world_model_heartbeat("omni_world_m_omni_wm_get_rssm_sta", 0.0f);


    return wm ? wm->rssm_state : NULL;
}

nimcp_error_t omni_wm_set_rssm_state(omni_world_model_t* wm,
                                      const omni_wm_rssm_state_t* state) {
    /* Phase 8: Heartbeat at operation start */
    omni_world_model_heartbeat("omni_world_m_omni_wm_set_rssm_sta", 0.0f);


    NIMCP_CHECK_THROW(wm, NIMCP_ERROR_INVALID_PARAM, "world model is NULL");
    NIMCP_CHECK_THROW(state, NIMCP_ERROR_INVALID_PARAM, "RSSM state is NULL");

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
    /* Phase 8: Heartbeat at operation start */
    omni_world_model_heartbeat("omni_world_m_omni_wm_rssm_step", 0.0f);


    NIMCP_CHECK_THROW(wm, NIMCP_ERROR_INVALID_PARAM, "world model is NULL");
    NIMCP_CHECK_THROW(state, NIMCP_ERROR_INVALID_PARAM, "state is NULL");
    NIMCP_CHECK_THROW(action, NIMCP_ERROR_INVALID_PARAM, "action is NULL");
    NIMCP_CHECK_THROW(next_state, NIMCP_ERROR_INVALID_PARAM, "next_state is NULL");

    omni_wm_dynamics_t* dyn = wm->forward_dynamics;

    /* Concatenate input: [h, z, a] */
    uint32_t input_dim = state->h_dim + state->z_dim + action_dim;
    float* input = nimcp_calloc(input_dim, sizeof(float));
    NIMCP_CHECK_THROW(input, NIMCP_ERROR_NO_MEMORY, "failed to allocate RSSM input buffer");

    memcpy(input, state->h, state->h_dim * sizeof(float));
    memcpy(input + state->h_dim, state->z, state->z_dim * sizeof(float));
    uint32_t copy_dim = action_dim < dyn->action_dim ? action_dim : dyn->action_dim;
    memcpy(input + state->h_dim + state->z_dim, action, copy_dim * sizeof(float));

    /* Compute next h: h' = tanh(W_h * [h, z, a] + b_h) */
    for (uint32_t i = 0; i < dyn->h_dim; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && dyn->h_dim > 256) {
            omni_world_model_heartbeat("omni_world_m_loop",
                             (float)(i + 1) / (float)dyn->h_dim);
        }

        float sum = dyn->b_h[i];
        for (uint32_t j = 0; j < input_dim && j < dyn->h_dim + dyn->z_dim + dyn->action_dim; j++) {
            sum += dyn->W_h[i * (dyn->h_dim + dyn->z_dim + dyn->action_dim) + j] * input[j];
        }
        next_state->h[i] = tanhf(sum);
    }

    /* Compute z prior: [mean, log_std] = W_z * h' + b_z */
    for (uint32_t i = 0; i < dyn->z_dim; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && dyn->z_dim > 256) {
            omni_world_model_heartbeat("omni_world_m_loop",
                             (float)(i + 1) / (float)dyn->z_dim);
        }

        float sum_mean = dyn->b_z[i];
        float sum_std = dyn->b_z[dyn->z_dim + i];
        for (uint32_t j = 0; j < dyn->h_dim; j++) {
            /* Phase 8: Loop progress heartbeat */
            if ((j & 0xFF) == 0 && dyn->h_dim > 256) {
                omni_world_model_heartbeat("omni_world_m_loop",
                                 (float)(j + 1) / (float)dyn->h_dim);
            }

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
    /* Phase 8: Heartbeat at operation start */
    omni_world_model_heartbeat("omni_world_m_omni_wm_rssm_imagine", 0.0f);


    NIMCP_CHECK_THROW(wm, NIMCP_ERROR_INVALID_PARAM, "world model is NULL");
    NIMCP_CHECK_THROW(initial_state, NIMCP_ERROR_INVALID_PARAM, "initial_state is NULL");
    NIMCP_CHECK_THROW(actions, NIMCP_ERROR_INVALID_PARAM, "actions is NULL");
    NIMCP_CHECK_THROW(trajectory, NIMCP_ERROR_INVALID_PARAM, "trajectory is NULL");
    NIMCP_CHECK_THROW(horizon > 0 && horizon <= OMNI_WM_MAX_HORIZON, NIMCP_ERROR_INVALID_PARAM,
                      "horizon must be between 1 and OMNI_WM_MAX_HORIZON");

    /* First state is initial */
    trajectory[0] = omni_wm_rssm_state_clone(initial_state);
    NIMCP_CHECK_THROW(trajectory[0], NIMCP_ERROR_NO_MEMORY, "failed to clone initial state");

    /* Roll out imagination */
    for (uint32_t t = 1; t < horizon; t++) {
        trajectory[t] = omni_wm_rssm_state_create(
            wm->config.rssm_h_dim,
            wm->config.rssm_z_dim
        );
        if (!trajectory[t]) {
            for (uint32_t i = 0; i < t; i++) {
                /* Phase 8: Loop progress heartbeat */
                if ((i & 0xFF) == 0 && t > 256) {
                    omni_world_model_heartbeat("omni_world_m_loop",
                                     (float)(i + 1) / (float)t);
                }

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
    /* Phase 8: Heartbeat at operation start */
    omni_world_model_heartbeat("omni_world_m_omni_wm_predict_forw", 0.0f);


    NIMCP_CHECK_THROW(wm, NIMCP_ERROR_INVALID_PARAM, "world model is NULL");
    NIMCP_CHECK_THROW(action, NIMCP_ERROR_INVALID_PARAM, "action is NULL");
    NIMCP_CHECK_THROW(result, NIMCP_ERROR_INVALID_PARAM, "result is NULL");

    /* Use RSSM if enabled */
    if (wm->config.use_rssm && wm->rssm_state) {
        omni_wm_rssm_state_t* next_rssm = omni_wm_rssm_state_create(
            wm->config.rssm_h_dim,
            wm->config.rssm_z_dim
        );
        NIMCP_CHECK_THROW(next_rssm, NIMCP_ERROR_NO_MEMORY, "failed to create next RSSM state");

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
            /* Phase 8: Loop progress heartbeat */
            if ((i & 0xFF) == 0 && next_rssm->z_dim > 256) {
                omni_world_model_heartbeat("omni_world_m_loop",
                                 (float)(i + 1) / (float)next_rssm->z_dim);
            }

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
    NIMCP_CHECK_THROW(wm->current_state, NIMCP_ERROR_INVALID_PARAM, "current_state is NULL");

    result->next_state = omni_wm_state_clone(wm->current_state);
    NIMCP_CHECK_THROW(result->next_state, NIMCP_ERROR_NO_MEMORY, "failed to clone current state");

    /* Simple dynamics: next = current + action_effect */
    uint32_t min_dim = action_dim < result->next_state->dim ?
                       action_dim : result->next_state->dim;
    for (uint32_t i = 0; i < min_dim; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && min_dim > 256) {
            omni_world_model_heartbeat("omni_world_m_loop",
                             (float)(i + 1) / (float)min_dim);
        }

        result->next_state->values[i] += action[i] * 0.1f;
    }

    result->direction = OMNI_WM_DIR_FORWARD;
    wm->stats.forward_predictions++;

    return NIMCP_SUCCESS;
}

nimcp_error_t omni_wm_infer_backward(omni_world_model_t* wm,
                                      const omni_wm_state_t* current_state,
                                      omni_wm_transition_t* result) {
    /* Phase 8: Heartbeat at operation start */
    omni_world_model_heartbeat("omni_world_m_omni_wm_infer_backwa", 0.0f);


    NIMCP_CHECK_THROW(wm, NIMCP_ERROR_INVALID_PARAM, "world model is NULL");
    NIMCP_CHECK_THROW(current_state, NIMCP_ERROR_INVALID_PARAM, "current_state is NULL");
    NIMCP_CHECK_THROW(result, NIMCP_ERROR_INVALID_PARAM, "result is NULL");

    /* Create previous state estimate */
    result->next_state = omni_wm_state_clone(current_state);
    NIMCP_CHECK_THROW(result->next_state, NIMCP_ERROR_NO_MEMORY, "failed to clone current state");

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
    /* Phase 8: Heartbeat at operation start */
    omni_world_model_heartbeat("omni_world_m_omni_wm_predict_late", 0.0f);


    NIMCP_CHECK_THROW(wm, NIMCP_ERROR_INVALID_PARAM, "world model is NULL");
    NIMCP_CHECK_THROW(source_state, NIMCP_ERROR_INVALID_PARAM, "source_state is NULL");
    NIMCP_CHECK_THROW(result, NIMCP_ERROR_INVALID_PARAM, "result is NULL");
    NIMCP_CHECK_THROW(wm->config.enable_lateral, NIMCP_ERROR_NOT_IMPLEMENTED, "lateral prediction not enabled");

    /* Cross-modal prediction using lateral dynamics */
    memset(result->values, 0, result->dim * sizeof(float));

    /* Simple linear mapping */
    uint32_t min_dim = source_state->dim < result->dim ?
                       source_state->dim : result->dim;
    for (uint32_t i = 0; i < min_dim; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && min_dim > 256) {
            omni_world_model_heartbeat("omni_world_m_loop",
                             (float)(i + 1) / (float)min_dim);
        }

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
    /* Phase 8: Heartbeat at operation start */
    omni_world_model_heartbeat("omni_world_m_omni_wm_predict_hier", 0.0f);


    NIMCP_CHECK_THROW(wm, NIMCP_ERROR_INVALID_PARAM, "world model is NULL");
    NIMCP_CHECK_THROW(state, NIMCP_ERROR_INVALID_PARAM, "state is NULL");
    NIMCP_CHECK_THROW(result, NIMCP_ERROR_INVALID_PARAM, "result is NULL");
    NIMCP_CHECK_THROW(wm->config.enable_hierarchical, NIMCP_ERROR_NOT_IMPLEMENTED, "hierarchical prediction not enabled");

    /* Hierarchical abstraction/concretization */
    uint32_t level_diff = target_level > state->level ?
                          target_level - state->level :
                          state->level - target_level;

    /* Abstract: pool/compress, Concrete: expand/detail */
    float scale = target_level > state->level ? 0.5f : 2.0f;

    memset(result->values, 0, result->dim * sizeof(float));
    uint32_t min_dim = state->dim < result->dim ? state->dim : result->dim;

    for (uint32_t i = 0; i < min_dim; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && min_dim > 256) {
            omni_world_model_heartbeat("omni_world_m_loop",
                             (float)(i + 1) / (float)min_dim);
        }

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

    /* Phase 8: Heartbeat at operation start */
    omni_world_model_heartbeat("omni_world_m_omni_wm_latent_creat", 0.0f);


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
    /* Phase 8: Heartbeat at operation start */
    omni_world_model_heartbeat("omni_world_m_omni_wm_latent_destr", 0.0f);


    nimcp_free(latent->embedding);
    nimcp_free(latent);
}

nimcp_error_t omni_wm_encode(omni_world_model_t* wm,
                              const float* observation,
                              uint32_t obs_dim,
                              omni_wm_latent_t* latent) {
    /* Phase 8: Heartbeat at operation start */
    omni_world_model_heartbeat("omni_world_m_omni_wm_encode", 0.0f);


    NIMCP_CHECK_THROW(wm, NIMCP_ERROR_INVALID_PARAM, "world model is NULL");
    NIMCP_CHECK_THROW(observation, NIMCP_ERROR_INVALID_PARAM, "observation is NULL");
    NIMCP_CHECK_THROW(latent, NIMCP_ERROR_INVALID_PARAM, "latent is NULL");

    /* Linear encoding: latent = ReLU(W * obs + b) */
    for (uint32_t i = 0; i < latent->dim; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && latent->dim > 256) {
            omni_world_model_heartbeat("omni_world_m_loop",
                             (float)(i + 1) / (float)latent->dim);
        }

        float sum = wm->encoder_b[i];
        for (uint32_t j = 0; j < obs_dim && j < wm->config.obs_dim; j++) {
            sum += wm->encoder_W[i * wm->config.obs_dim + j] * observation[j];
        }
        latent->embedding[i] = sum > 0 ? sum : 0; /* ReLU */
    }

    /* Compute information content (approximate entropy) */
    float entropy = 0.0f;
    for (uint32_t i = 0; i < latent->dim; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && latent->dim > 256) {
            omni_world_model_heartbeat("omni_world_m_loop",
                             (float)(i + 1) / (float)latent->dim);
        }

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
    /* Phase 8: Heartbeat at operation start */
    omni_world_model_heartbeat("omni_world_m_omni_wm_decode", 0.0f);


    NIMCP_CHECK_THROW(wm, NIMCP_ERROR_INVALID_PARAM, "world model is NULL");
    NIMCP_CHECK_THROW(latent, NIMCP_ERROR_INVALID_PARAM, "latent is NULL");
    NIMCP_CHECK_THROW(observation, NIMCP_ERROR_INVALID_PARAM, "observation is NULL");

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
    /* Phase 8: Heartbeat at operation start */
    omni_world_model_heartbeat("omni_world_m_omni_wm_predict_late", 0.0f);


    NIMCP_CHECK_THROW(wm, NIMCP_ERROR_INVALID_PARAM, "world model is NULL");
    NIMCP_CHECK_THROW(latent, NIMCP_ERROR_INVALID_PARAM, "latent is NULL");
    NIMCP_CHECK_THROW(action, NIMCP_ERROR_INVALID_PARAM, "action is NULL");
    NIMCP_CHECK_THROW(predicted_latent, NIMCP_ERROR_INVALID_PARAM, "predicted_latent is NULL");

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

    /* Phase 8: Heartbeat at operation start */
    omni_world_model_heartbeat("omni_world_m_omni_wm_mdn_create", 0.0f);


    omni_wm_mdn_prediction_t* pred = nimcp_calloc(1, sizeof(omni_wm_mdn_prediction_t));
    if (!pred) return NULL;

    pred->components = nimcp_calloc(num_components, sizeof(omni_wm_mdn_component_t));
    if (!pred->components) {
        nimcp_free(pred);
        return NULL;
    }

    for (uint32_t k = 0; k < num_components; k++) {
        /* Phase 8: Loop progress heartbeat */
        if ((k & 0xFF) == 0 && num_components > 256) {
            omni_world_model_heartbeat("omni_world_m_loop",
                             (float)(k + 1) / (float)num_components);
        }

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
            /* Phase 8: Loop progress heartbeat */
            if ((i & 0xFF) == 0 && dim > 256) {
                omni_world_model_heartbeat("omni_world_m_loop",
                                 (float)(i + 1) / (float)dim);
            }

            pred->components[k].std[i] = 1.0f;
        }
    }

    pred->num_components = num_components;
    pred->dim = dim;

    return pred;
}

void omni_wm_mdn_destroy(omni_wm_mdn_prediction_t* pred) {
    if (!pred) return;
    /* Phase 8: Heartbeat at operation start */
    omni_world_model_heartbeat("omni_world_m_omni_wm_mdn_destroy", 0.0f);


    for (uint32_t k = 0; k < pred->num_components; k++) {
        /* Phase 8: Loop progress heartbeat */
        if ((k & 0xFF) == 0 && pred->num_components > 256) {
            omni_world_model_heartbeat("omni_world_m_loop",
                             (float)(k + 1) / (float)pred->num_components);
        }

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
    /* Phase 8: Heartbeat at operation start */
    omni_world_model_heartbeat("omni_world_m_omni_wm_predict_mdn", 0.0f);


    NIMCP_CHECK_THROW(wm, NIMCP_ERROR_INVALID_PARAM, "world model is NULL");
    NIMCP_CHECK_THROW(state, NIMCP_ERROR_INVALID_PARAM, "state is NULL");
    NIMCP_CHECK_THROW(action, NIMCP_ERROR_INVALID_PARAM, "action is NULL");
    NIMCP_CHECK_THROW(pred, NIMCP_ERROR_INVALID_PARAM, "MDN prediction is NULL");

    /* Generate mixture components based on state and action */
    for (uint32_t k = 0; k < pred->num_components; k++) {
        /* Phase 8: Loop progress heartbeat */
        if ((k & 0xFF) == 0 && pred->num_components > 256) {
            omni_world_model_heartbeat("omni_world_m_loop",
                             (float)(k + 1) / (float)pred->num_components);
        }

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
    /* Phase 8: Heartbeat at operation start */
    omni_world_model_heartbeat("omni_world_m_omni_wm_mdn_sample", 0.0f);


    NIMCP_CHECK_THROW(pred, NIMCP_ERROR_INVALID_PARAM, "MDN prediction is NULL");
    NIMCP_CHECK_THROW(sample, NIMCP_ERROR_INVALID_PARAM, "sample buffer is NULL");

    /* Select component based on weights */
    unsigned int seed = (unsigned int)time(NULL);
    float r = (float)rand_r(&seed) / RAND_MAX;
    float cumsum = 0.0f;
    uint32_t selected = 0;

    for (uint32_t k = 0; k < pred->num_components; k++) {
        /* Phase 8: Loop progress heartbeat */
        if ((k & 0xFF) == 0 && pred->num_components > 256) {
            omni_world_model_heartbeat("omni_world_m_loop",
                             (float)(k + 1) / (float)pred->num_components);
        }

        cumsum += pred->components[k].weight;
        if (r <= cumsum) {
            selected = k;
            break;
        }
    }

    /* Sample from selected component */
    omni_wm_mdn_component_t* comp = &pred->components[selected];
    for (uint32_t i = 0; i < pred->dim; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && pred->dim > 256) {
            omni_world_model_heartbeat("omni_world_m_loop",
                             (float)(i + 1) / (float)pred->dim);
        }

        sample[i] = comp->mean[i] + comp->std[i] * randn(&seed);
    }

    return NIMCP_SUCCESS;
}

nimcp_error_t omni_wm_mdn_mode(const omni_wm_mdn_prediction_t* pred,
                                float* mode) {
    /* Phase 8: Heartbeat at operation start */
    omni_world_model_heartbeat("omni_world_m_omni_wm_mdn_mode", 0.0f);


    NIMCP_CHECK_THROW(pred, NIMCP_ERROR_INVALID_PARAM, "MDN prediction is NULL");
    NIMCP_CHECK_THROW(mode, NIMCP_ERROR_INVALID_PARAM, "mode buffer is NULL");

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
    /* Phase 8: Heartbeat at operation start */
    omni_world_model_heartbeat("omni_world_m_omni_wm_mdn_log_prob", 0.0f);


    float max_log = -FLT_MAX;
    float* log_probs = nimcp_calloc(pred->num_components, sizeof(float));
    if (!log_probs) return -FLT_MAX;

    for (uint32_t k = 0; k < pred->num_components; k++) {
        /* Phase 8: Loop progress heartbeat */
        if ((k & 0xFF) == 0 && pred->num_components > 256) {
            omni_world_model_heartbeat("omni_world_m_loop",
                             (float)(k + 1) / (float)pred->num_components);
        }

        float log_prob = logf(pred->components[k].weight);

        for (uint32_t i = 0; i < pred->dim; i++) {
            /* Phase 8: Loop progress heartbeat */
            if ((i & 0xFF) == 0 && pred->dim > 256) {
                omni_world_model_heartbeat("omni_world_m_loop",
                                 (float)(i + 1) / (float)pred->dim);
            }

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
        /* Phase 8: Loop progress heartbeat */
        if ((k & 0xFF) == 0 && pred->num_components > 256) {
            omni_world_model_heartbeat("omni_world_m_loop",
                             (float)(k + 1) / (float)pred->num_components);
        }

        sum += expf(log_probs[k] - max_log);
    }

    nimcp_free(log_probs);
    /* Guard against logf(0) when num_components is 0 */
    return max_log + logf(sum > 0.0f ? sum : 1e-10f);
}

/* ============================================================================
 * Experience Replay
 * ============================================================================ */

omni_wm_experience_t* omni_wm_experience_create(uint32_t state_dim,
                                                  uint32_t action_dim,
                                                  uint32_t obs_dim) {
    /* Phase 8: Heartbeat at operation start */
    omni_world_model_heartbeat("omni_world_m_omni_wm_experience_c", 0.0f);


    if (state_dim == 0 || action_dim == 0 || obs_dim == 0) {
        NIMCP_THROW(NIMCP_ERROR_INVALID_PARAM, "omni_wm_experience_create: zero dimension");
        return NULL;
    }

    omni_wm_experience_t* exp = nimcp_calloc(1, sizeof(omni_wm_experience_t));
    if (!exp) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "omni_wm_experience_create: allocation failed");
        return NULL;
    }

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
    /* Phase 8: Heartbeat at operation start */
    omni_world_model_heartbeat("omni_world_m_omni_wm_experience_d", 0.0f);


    omni_wm_rssm_state_destroy(exp->state);
    omni_wm_rssm_state_destroy(exp->next_state);
    nimcp_free(exp->action);
    nimcp_free(exp->observation);
    nimcp_free(exp);
}

nimcp_error_t omni_wm_add_experience(omni_world_model_t* wm,
                                      const omni_wm_experience_t* exp) {
    /* Phase 8: Heartbeat at operation start */
    omni_world_model_heartbeat("omni_world_m_omni_wm_add_experien", 0.0f);


    NIMCP_CHECK_THROW(wm, NIMCP_ERROR_INVALID_PARAM, "world model is NULL");
    NIMCP_CHECK_THROW(exp, NIMCP_ERROR_INVALID_PARAM, "experience is NULL");

    omni_wm_replay_buffer_t* buf = wm->replay_buffer;

    /* Clone experience */
    omni_wm_experience_t* clone = omni_wm_experience_create(
        wm->config.state_dim,
        exp->action_dim,
        exp->obs_dim
    );
    NIMCP_CHECK_THROW(clone, NIMCP_ERROR_NO_MEMORY, "failed to create experience clone");

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

    /* Phase 8: Heartbeat at operation start */
    omni_world_model_heartbeat("omni_world_m_omni_wm_sample_exper", 0.0f);


    omni_wm_replay_buffer_t* buf = wm->replay_buffer;
    if (!buf || buf->size == 0) return 0;

    uint32_t actual_size = batch_size < buf->size ? batch_size : buf->size;

    /* Simple uniform sampling - return clones so caller can safely destroy */
    for (uint32_t i = 0; i < actual_size; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && actual_size > 256) {
            omni_world_model_heartbeat("omni_world_m_loop",
                             (float)(i + 1) / (float)actual_size);
        }

        uint32_t idx = rand_r(&wm->rand_seed) % buf->size;
        omni_wm_experience_t* src = buf->experiences[idx];
        if (!src) {
            batch[i] = NULL;
            continue;
        }

        /* Clone the experience */
        omni_wm_experience_t* clone = omni_wm_experience_create(
            wm->config.state_dim,
            src->action_dim,
            src->obs_dim
        );
        if (!clone) {
            /* Clean up already allocated clones */
            for (uint32_t j = 0; j < i; j++) {
                /* Phase 8: Loop progress heartbeat */
                if ((j & 0xFF) == 0 && i > 256) {
                    omni_world_model_heartbeat("omni_world_m_loop",
                                     (float)(j + 1) / (float)i);
                }

                omni_wm_experience_destroy(batch[j]);
                batch[j] = NULL;
            }
            return 0;
        }

        memcpy(clone->action, src->action, src->action_dim * sizeof(float));
        clone->reward = src->reward;
        clone->symlog_reward = src->symlog_reward;
        clone->terminal = src->terminal;
        clone->timestamp = src->timestamp;

        if (src->observation && clone->observation) {
            memcpy(clone->observation, src->observation, src->obs_dim * sizeof(float));
        }

        batch[i] = clone;
    }

    return actual_size;
}

uint32_t omni_wm_get_replay_size(const omni_world_model_t* wm) {
    /* Phase 8: Heartbeat at operation start */
    omni_world_model_heartbeat("omni_world_m_omni_wm_get_replay_s", 0.0f);


    return wm && wm->replay_buffer ? wm->replay_buffer->size : 0;
}

nimcp_error_t omni_wm_clear_replay(omni_world_model_t* wm) {
    /* Phase 8: Heartbeat at operation start */
    omni_world_model_heartbeat("omni_world_m_omni_wm_clear_replay", 0.0f);


    NIMCP_CHECK_THROW(wm, NIMCP_ERROR_INVALID_PARAM, "world model is NULL");
    NIMCP_CHECK_THROW(wm->replay_buffer, NIMCP_ERROR_INVALID_PARAM, "replay buffer is NULL");

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
    /* Phase 8: Heartbeat at operation start */
    omni_world_model_heartbeat("omni_world_m_omni_wm_update", 0.0f);


    NIMCP_CHECK_THROW(wm, NIMCP_ERROR_INVALID_PARAM, "world model is NULL");
    NIMCP_CHECK_THROW(state, NIMCP_ERROR_INVALID_PARAM, "state is NULL");
    NIMCP_CHECK_THROW(action, NIMCP_ERROR_INVALID_PARAM, "action is NULL");
    NIMCP_CHECK_THROW(next_state, NIMCP_ERROR_INVALID_PARAM, "next_state is NULL");

    /* Compute prediction error */
    omni_wm_transition_t pred;
    memset(&pred, 0, sizeof(pred));

    nimcp_error_t err = omni_wm_predict_forward(wm, action, action_dim, &pred);
    if (err != NIMCP_SUCCESS) return err;

    float error = 0.0f;
    uint32_t min_dim = pred.next_state->dim < next_state->dim ?
                       pred.next_state->dim : next_state->dim;

    for (uint32_t i = 0; i < min_dim; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && min_dim > 256) {
            omni_world_model_heartbeat("omni_world_m_loop",
                             (float)(i + 1) / (float)min_dim);
        }

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
    /* Phase 8: Heartbeat at operation start */
    omni_world_model_heartbeat("omni_world_m_omni_wm_dream", 0.0f);


    NIMCP_CHECK_THROW(wm, NIMCP_ERROR_INVALID_PARAM, "world model is NULL");
    NIMCP_CHECK_THROW(wm->config.enable_dreaming, NIMCP_ERROR_NOT_IMPLEMENTED, "dreaming not enabled");

    for (uint32_t ep = 0; ep < num_episodes; ep++) {
        /* Phase 8: Loop progress heartbeat */
        if ((ep & 0xFF) == 0 && num_episodes > 256) {
            omni_world_model_heartbeat("omni_world_m_loop",
                             (float)(ep + 1) / (float)num_episodes);
        }

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
            /* Phase 8: Loop progress heartbeat */
            if ((t & 0xFF) == 0 && episode_length > 256) {
                omni_world_model_heartbeat("omni_world_m_loop",
                                 (float)(t + 1) / (float)episode_length);
            }

            /* Generate random action with noise */
            float* action = nimcp_calloc(wm->config.action_dim, sizeof(float));
            if (!action) break;

            for (uint32_t i = 0; i < wm->config.action_dim; i++) {
                /* Phase 8: Loop progress heartbeat */
                if ((i & 0xFF) == 0 && wm->config.action_dim > 256) {
                    omni_world_model_heartbeat("omni_world_m_loop",
                                     (float)(i + 1) / (float)wm->config.action_dim);
                }

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
    /* Phase 8: Heartbeat at operation start */
    omni_world_model_heartbeat("omni_world_m_omni_wm_set_learning", 0.0f);


    NIMCP_CHECK_THROW(wm, NIMCP_ERROR_INVALID_PARAM, "world model is NULL");
    NIMCP_CHECK_THROW(learning_rate >= 0.0f && learning_rate <= 1.0f, NIMCP_ERROR_INVALID_PARAM,
                      "learning_rate must be between 0.0 and 1.0");
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
    if (horizon == 0) return NULL;

    /* Phase 8: Heartbeat at operation start */
    omni_world_model_heartbeat("omni_world_m_omni_wm_cf_query_cre", 0.0f);


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
    /* Phase 8: Heartbeat at operation start */
    omni_world_model_heartbeat("omni_world_m_omni_wm_cf_query_des", 0.0f);


    omni_wm_state_destroy(query->initial_state);
    nimcp_free(query->hypothetical_action);
    nimcp_free(query->context);
    nimcp_free(query);
}

nimcp_error_t omni_wm_counterfactual(omni_world_model_t* wm,
                                      const omni_wm_counterfactual_query_t* query,
                                      omni_wm_counterfactual_result_t* result) {
    /* Phase 8: Heartbeat at operation start */
    omni_world_model_heartbeat("omni_world_m_omni_wm_counterfactu", 0.0f);


    NIMCP_CHECK_THROW(wm, NIMCP_ERROR_INVALID_PARAM, "world model is NULL");
    NIMCP_CHECK_THROW(query, NIMCP_ERROR_INVALID_PARAM, "query is NULL");
    NIMCP_CHECK_THROW(result, NIMCP_ERROR_INVALID_PARAM, "result is NULL");

    memset(result, 0, sizeof(omni_wm_counterfactual_result_t));

    /* Allocate trajectory */
    result->trajectory = nimcp_calloc(query->horizon, sizeof(omni_wm_state_t*));
    NIMCP_CHECK_THROW(result->trajectory, NIMCP_ERROR_NO_MEMORY, "failed to allocate trajectory");

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
                /* Phase 8: Loop progress heartbeat */
                if ((i & 0xFF) == 0 && t > 256) {
                    omni_world_model_heartbeat("omni_world_m_loop",
                                     (float)(i + 1) / (float)t);
                }

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
    /* Phase 8: Heartbeat at operation start */
    omni_world_model_heartbeat("omni_world_m_omni_wm_what_if", 0.0f);


    NIMCP_CHECK_THROW(wm, NIMCP_ERROR_INVALID_PARAM, "world model is NULL");
    NIMCP_CHECK_THROW(action, NIMCP_ERROR_INVALID_PARAM, "action is NULL");
    NIMCP_CHECK_THROW(result, NIMCP_ERROR_INVALID_PARAM, "result is NULL");
    NIMCP_CHECK_THROW(wm->current_state, NIMCP_ERROR_INVALID_PARAM, "current_state is NULL");

    omni_wm_counterfactual_query_t* query = omni_wm_cf_query_create(
        OMNI_WM_CF_ACTION,
        wm->current_state,
        action,
        action_dim,
        horizon
    );
    NIMCP_CHECK_THROW(query, NIMCP_ERROR_NO_MEMORY, "failed to create counterfactual query");

    nimcp_error_t err = omni_wm_counterfactual(wm, query, result);
    omni_wm_cf_query_destroy(query);

    return err;
}

void omni_wm_cf_result_destroy(omni_wm_counterfactual_result_t* result) {
    if (!result) return;
    /* Phase 8: Heartbeat at operation start */
    omni_world_model_heartbeat("omni_world_m_omni_wm_cf_result_de", 0.0f);


    for (uint32_t i = 0; i < result->trajectory_len; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && result->trajectory_len > 256) {
            omni_world_model_heartbeat("omni_world_m_loop",
                             (float)(i + 1) / (float)result->trajectory_len);
        }

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

    /* Phase 8: Heartbeat at operation start */
    omni_world_model_heartbeat("omni_world_m_omni_wm_rollout_crea", 0.0f);


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

    /* Phase 8: Heartbeat at operation start */
    omni_world_model_heartbeat("omni_world_m_omni_wm_rollout_dest", 0.0f);


    for (uint32_t i = 0; i < rollout->length; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && rollout->length > 256) {
            omni_world_model_heartbeat("omni_world_m_loop",
                             (float)(i + 1) / (float)rollout->length);
        }

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
    /* Phase 8: Heartbeat at operation start */
    omni_world_model_heartbeat("omni_world_m_omni_wm_rollout", 0.0f);


    NIMCP_CHECK_THROW(wm, NIMCP_ERROR_INVALID_PARAM, "world model is NULL");
    NIMCP_CHECK_THROW(policy, NIMCP_ERROR_INVALID_PARAM, "policy function is NULL");
    NIMCP_CHECK_THROW(rollout, NIMCP_ERROR_INVALID_PARAM, "rollout is NULL");
    NIMCP_CHECK_THROW(wm->current_state, NIMCP_ERROR_INVALID_PARAM, "current_state is NULL");

    rollout->states[0] = omni_wm_state_clone(wm->current_state);
    NIMCP_CHECK_THROW(rollout->states[0], NIMCP_ERROR_NO_MEMORY, "failed to clone current state for rollout");

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

    /* Phase 8: Heartbeat at operation start */
    omni_world_model_heartbeat("omni_world_m_omni_wm_evaluate_efe", 0.0f);


    float efe = 0.0f;
    float gamma = wm->config.discount_factor;
    float discount = 1.0f;

    for (uint32_t t = 0; t < rollout->length; t++) {
        /* Phase 8: Loop progress heartbeat */
        if ((t & 0xFF) == 0 && rollout->length > 256) {
            omni_world_model_heartbeat("omni_world_m_loop",
                             (float)(t + 1) / (float)rollout->length);
        }

        if (!rollout->states[t]) continue;

        /* Risk: KL[q(o|pi) || p(o)] - distance from preferred */
        float risk = 0.0f;
        uint32_t min_dim = rollout->states[t]->dim < obs_dim ?
                           rollout->states[t]->dim : obs_dim;

        for (uint32_t i = 0; i < min_dim; i++) {
            /* Phase 8: Loop progress heartbeat */
            if ((i & 0xFF) == 0 && min_dim > 256) {
                omni_world_model_heartbeat("omni_world_m_loop",
                                 (float)(i + 1) / (float)min_dim);
            }

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
    /* Phase 8: Heartbeat at operation start */
    omni_world_model_heartbeat("omni_world_m_omni_wm_predict_obse", 0.0f);


    NIMCP_CHECK_THROW(wm, NIMCP_ERROR_INVALID_PARAM, "world model is NULL");
    NIMCP_CHECK_THROW(state, NIMCP_ERROR_INVALID_PARAM, "state is NULL");
    NIMCP_CHECK_THROW(predicted_obs, NIMCP_ERROR_INVALID_PARAM, "predicted_obs buffer is NULL");

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
    /* Phase 8: Heartbeat at operation start */
    omni_world_model_heartbeat("omni_world_m_omni_wm_infer_state", 0.0f);


    NIMCP_CHECK_THROW(wm, NIMCP_ERROR_INVALID_PARAM, "world model is NULL");
    NIMCP_CHECK_THROW(observations, NIMCP_ERROR_INVALID_PARAM, "observations is NULL");
    NIMCP_CHECK_THROW(inferred_state, NIMCP_ERROR_INVALID_PARAM, "inferred_state is NULL");

    /* Use encoder to infer state from observations */
    omni_wm_latent_t* latent = omni_wm_latent_create(inferred_state->dim);
    NIMCP_CHECK_THROW(latent, NIMCP_ERROR_NO_MEMORY, "failed to create latent for state inference");

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
    /* Phase 8: Heartbeat at operation start */
    omni_world_model_heartbeat("omni_world_m_omni_wm_connect_acti", 0.0f);


    NIMCP_CHECK_THROW(wm, NIMCP_ERROR_INVALID_PARAM, "world model is NULL");
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

    /* Phase 8: Heartbeat at operation start */
    omni_world_model_heartbeat("omni_world_m_omni_wm_evaluate_pol", 0.0f);


    float efe = 0.0f;
    float gamma = wm->config.discount_factor;
    float discount = 1.0f;

    omni_wm_state_t* state = omni_wm_state_clone(wm->current_state);
    if (!state) return FLT_MAX;

    for (uint32_t t = 0; t < horizon; t++) {
        /* Phase 8: Loop progress heartbeat */
        if ((t & 0xFF) == 0 && horizon > 256) {
            omni_world_model_heartbeat("omni_world_m_loop",
                             (float)(t + 1) / (float)horizon);
        }

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
            /* Phase 8: Loop progress heartbeat */
            if ((i & 0xFF) == 0 && min_dim > 256) {
                omni_world_model_heartbeat("omni_world_m_loop",
                                 (float)(i + 1) / (float)min_dim);
            }

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
    /* Phase 8: Heartbeat at operation start */
    omni_world_model_heartbeat("omni_world_m_omni_wm_get_stats", 0.0f);


    NIMCP_CHECK_THROW(wm, NIMCP_ERROR_INVALID_PARAM, "world model is NULL");
    NIMCP_CHECK_THROW(stats, NIMCP_ERROR_INVALID_PARAM, "stats is NULL");
    memcpy(stats, &wm->stats, sizeof(omni_wm_stats_t));
    return NIMCP_SUCCESS;
}

nimcp_error_t omni_wm_reset_stats(omni_world_model_t* wm) {
    /* Phase 8: Heartbeat at operation start */
    omni_world_model_heartbeat("omni_world_m_omni_wm_reset_stats", 0.0f);


    NIMCP_CHECK_THROW(wm, NIMCP_ERROR_INVALID_PARAM, "world model is NULL");
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
 * Bio-Async Integration - Message Structures
 * ============================================================================ */

/**
 * @brief Prediction request message (BIO_MSG_OMNI_WM_PREDICT - 0x6201)
 */
typedef struct {
    bio_message_header_t header;
    float state[OMNI_WM_MAX_STATE_DIM];   /**< Current state */
    uint32_t state_dim;                    /**< State dimensionality */
    float action[OMNI_WM_MAX_ACTION_DIM]; /**< Action to predict */
    uint32_t action_dim;                   /**< Action dimensionality */
    omni_wm_direction_t direction;         /**< Dynamics direction */
} bio_msg_omni_wm_predict_t;

/**
 * @brief Prediction response message
 */
typedef struct {
    bio_message_header_t header;
    float predicted_state[OMNI_WM_MAX_STATE_DIM]; /**< Predicted state */
    uint32_t state_dim;                            /**< State dimensionality */
    float prediction_error;                        /**< Prediction error estimate */
    float confidence;                              /**< Prediction confidence */
} bio_msg_omni_wm_predict_response_t;

/**
 * @brief Counterfactual query message (BIO_MSG_OMNI_WM_COUNTERFACTUAL - 0x6202)
 */
typedef struct {
    bio_message_header_t header;
    omni_wm_counterfactual_type_t cf_type;         /**< Type of counterfactual */
    float initial_state[OMNI_WM_MAX_STATE_DIM];    /**< Starting state */
    uint32_t state_dim;                             /**< State dimensionality */
    float hypothetical_action[OMNI_WM_MAX_ACTION_DIM]; /**< Hypothetical action */
    uint32_t action_dim;                            /**< Action dimensionality */
    uint32_t horizon;                               /**< Simulation horizon */
} bio_msg_omni_wm_counterfactual_t;

/**
 * @brief Counterfactual response message
 */
typedef struct {
    bio_message_header_t header;
    float expected_reward;       /**< Expected cumulative reward */
    float divergence;            /**< Divergence from actual trajectory */
    float confidence;            /**< Prediction confidence */
    uint32_t trajectory_length;  /**< Number of simulated steps */
} bio_msg_omni_wm_counterfactual_response_t;

/**
 * @brief Model update message (BIO_MSG_OMNI_WM_UPDATE - 0x6203)
 */
typedef struct {
    bio_message_header_t header;
    float state[OMNI_WM_MAX_STATE_DIM];       /**< Current state */
    uint32_t state_dim;                        /**< State dimensionality */
    float action[OMNI_WM_MAX_ACTION_DIM];     /**< Action taken */
    uint32_t action_dim;                       /**< Action dimensionality */
    float next_state[OMNI_WM_MAX_STATE_DIM];  /**< Resulting state */
    float reward;                              /**< Reward received */
} bio_msg_omni_wm_update_t;

/**
 * @brief Model update acknowledgment message
 */
typedef struct {
    bio_message_header_t header;
    nimcp_error_t status;       /**< Update status */
    float prediction_error;     /**< Error before update */
} bio_msg_omni_wm_update_ack_t;

/**
 * @brief Rollout request message (BIO_MSG_OMNI_WM_ROLLOUT - 0x6204)
 */
typedef struct {
    bio_message_header_t header;
    float initial_state[OMNI_WM_MAX_STATE_DIM]; /**< Starting state */
    uint32_t state_dim;                          /**< State dimensionality */
    float policy_actions[OMNI_WM_MAX_HORIZON * OMNI_WM_MAX_ACTION_DIM]; /**< Action sequence */
    uint32_t action_dim;                         /**< Action dimensionality */
    uint32_t horizon;                            /**< Rollout horizon */
    float preferred_obs[OMNI_WM_MAX_OBS_DIM];   /**< Preferred observations (goal) */
    uint32_t obs_dim;                            /**< Observation dimensionality */
} bio_msg_omni_wm_rollout_t;

/**
 * @brief Rollout response message
 */
typedef struct {
    bio_message_header_t header;
    float total_reward;          /**< Cumulative reward */
    float expected_free_energy;  /**< Expected free energy */
    uint32_t steps_completed;    /**< Actual steps completed */
    float final_state[OMNI_WM_MAX_STATE_DIM]; /**< Final state */
    uint32_t state_dim;                        /**< State dimensionality */
} bio_msg_omni_wm_rollout_response_t;

/* ============================================================================
 * Bio-Async Integration - Message Handlers
 * ============================================================================ */

/**
 * @brief Handle prediction request message
 *
 * WHAT: Process forward/backward/lateral dynamics prediction request
 * WHY:  Allow other modules to query world model predictions via bio-async
 * HOW:  Extract state/action, call appropriate dynamics function, send response
 */
static nimcp_error_t handle_omni_wm_predict(
    const void* msg,
    size_t msg_size,
    nimcp_bio_promise_t response_promise,
    void* user_data)
{
    NIMCP_CHECK_THROW(msg, NIMCP_ERROR_NULL_POINTER, "message is NULL");
    NIMCP_CHECK_THROW(user_data, NIMCP_ERROR_NULL_POINTER, "user_data is NULL");

    NIMCP_CHECK_THROW(msg_size >= sizeof(bio_msg_omni_wm_predict_t), NIMCP_ERROR_INVALID_PARAM,
                      "Predict request too small: expected %zu bytes", sizeof(bio_msg_omni_wm_predict_t));

    const bio_msg_omni_wm_predict_t* req = (const bio_msg_omni_wm_predict_t*)msg;
    omni_world_model_t* wm = (omni_world_model_t*)user_data;

    NIMCP_LOGGING_DEBUG("Received WM predict request: dir=%s, state_dim=%u, action_dim=%u",
                        omni_wm_direction_to_string(req->direction),
                        req->state_dim, req->action_dim);

    /* Validate dimensions */
    NIMCP_CHECK_THROW(req->state_dim > 0 && req->state_dim <= OMNI_WM_MAX_STATE_DIM &&
                      req->action_dim <= OMNI_WM_MAX_ACTION_DIM, NIMCP_ERROR_INVALID_PARAM,
                      "Invalid dimensions in predict request");

    /* Create temporary state from request */
    omni_wm_state_t* state = omni_wm_state_from_values(req->state, req->state_dim);
    NIMCP_CHECK_THROW(state, NIMCP_ERROR_NO_MEMORY, "failed to create state from request values");

    /* Prepare response */
    bio_msg_omni_wm_predict_response_t response = {0};
    bio_msg_init_header(&response.header, BIO_MSG_OMNI_WM_PREDICT,
                        BIO_MODULE_OMNI_WORLD_MODEL,
                        req->header.source_module,
                        sizeof(response));

    /* Perform prediction based on direction */
    omni_wm_transition_t transition = {0};
    nimcp_error_t result = NIMCP_SUCCESS;

    switch (req->direction) {
        case OMNI_WM_DIR_FORWARD:
            result = omni_wm_predict_forward(wm, req->action, req->action_dim, &transition);
            break;
        case OMNI_WM_DIR_BACKWARD:
            result = omni_wm_infer_backward(wm, state, &transition);
            break;
        case OMNI_WM_DIR_LATERAL:
            /* For lateral, use action_dim as target modality ID */
            result = omni_wm_predict_lateral(wm, state, req->action_dim,
                                              transition.next_state);
            break;
        case OMNI_WM_DIR_HIERARCHICAL:
            /* For hierarchical, use first action element as target level */
            result = omni_wm_predict_hierarchical(wm, state,
                                                   (uint32_t)req->action[0],
                                                   transition.next_state);
            break;
        default:
            result = NIMCP_ERROR_INVALID_PARAM;
            break;
    }

    /* Fill response */
    if (result == NIMCP_SUCCESS && transition.next_state) {
        uint32_t copy_dim = transition.next_state->dim;
        if (copy_dim > OMNI_WM_MAX_STATE_DIM) {
            copy_dim = OMNI_WM_MAX_STATE_DIM;
        }
        memcpy(response.predicted_state, transition.next_state->values,
               copy_dim * sizeof(float));
        response.state_dim = copy_dim;
        response.prediction_error = transition.prediction_error;
        response.confidence = 1.0f - transition.prediction_error;
    } else {
        response.state_dim = 0;
        response.prediction_error = 1.0f;
        response.confidence = 0.0f;
    }

    /* Cleanup */
    omni_wm_state_destroy(state);
    if (transition.next_state) {
        omni_wm_state_destroy(transition.next_state);
    }
    if (transition.action_taken) {
        nimcp_free(transition.action_taken);
    }

    /* Send response via promise */
    if (response_promise) {
        nimcp_bio_promise_complete_sized(response_promise, &response, sizeof(response));
    }

    return result;
}

/**
 * @brief Handle counterfactual query message
 *
 * WHAT: Process "what if" queries for alternative scenarios
 * WHY:  Enable other modules to evaluate hypothetical actions
 * HOW:  Execute counterfactual simulation and return expected outcome
 */
static nimcp_error_t handle_omni_wm_counterfactual(
    const void* msg,
    size_t msg_size,
    nimcp_bio_promise_t response_promise,
    void* user_data)
{
    NIMCP_CHECK_THROW(msg, NIMCP_ERROR_NULL_POINTER, "message is NULL");
    NIMCP_CHECK_THROW(user_data, NIMCP_ERROR_NULL_POINTER, "user_data is NULL");

    NIMCP_CHECK_THROW(msg_size >= sizeof(bio_msg_omni_wm_counterfactual_t), NIMCP_ERROR_INVALID_PARAM,
                      "Counterfactual request too small: expected %zu bytes", sizeof(bio_msg_omni_wm_counterfactual_t));

    const bio_msg_omni_wm_counterfactual_t* req = (const bio_msg_omni_wm_counterfactual_t*)msg;
    omni_world_model_t* wm = (omni_world_model_t*)user_data;

    NIMCP_LOGGING_DEBUG("Received WM counterfactual request: type=%s, horizon=%u",
                        omni_wm_cf_type_to_string(req->cf_type), req->horizon);

    /* Validate dimensions */
    NIMCP_CHECK_THROW(req->state_dim > 0 && req->state_dim <= OMNI_WM_MAX_STATE_DIM &&
                      req->action_dim <= OMNI_WM_MAX_ACTION_DIM &&
                      req->horizon <= OMNI_WM_MAX_HORIZON, NIMCP_ERROR_INVALID_PARAM,
                      "Invalid dimensions in counterfactual request");

    /* Create temporary state from request */
    omni_wm_state_t* state = omni_wm_state_from_values(req->initial_state, req->state_dim);
    NIMCP_CHECK_THROW(state, NIMCP_ERROR_NO_MEMORY, "failed to create state from counterfactual request");

    /* Prepare response */
    bio_msg_omni_wm_counterfactual_response_t response = {0};
    bio_msg_init_header(&response.header, BIO_MSG_OMNI_WM_COUNTERFACTUAL,
                        BIO_MODULE_OMNI_WORLD_MODEL,
                        req->header.source_module,
                        sizeof(response));

    /* Execute counterfactual query */
    omni_wm_counterfactual_result_t cf_result = {0};
    nimcp_error_t result = omni_wm_what_if(wm,
                                            req->hypothetical_action,
                                            req->action_dim,
                                            req->horizon,
                                            &cf_result);

    /* Fill response */
    if (result == NIMCP_SUCCESS) {
        response.expected_reward = cf_result.expected_reward;
        response.divergence = cf_result.divergence;
        response.confidence = cf_result.confidence;
        response.trajectory_length = cf_result.trajectory_len;

        /* Cleanup counterfactual result */
        omni_wm_cf_result_destroy(&cf_result);
    } else {
        response.expected_reward = 0.0f;
        response.divergence = 1.0f;
        response.confidence = 0.0f;
        response.trajectory_length = 0;
    }

    /* Cleanup */
    omni_wm_state_destroy(state);

    /* Send response via promise */
    if (response_promise) {
        nimcp_bio_promise_complete_sized(response_promise, &response, sizeof(response));
    }

    return result;
}

/**
 * @brief Handle model update message
 *
 * WHAT: Process experience tuple for model learning
 * WHY:  Allow other modules to contribute to world model training
 * HOW:  Call omni_wm_update with provided state transition
 */
static nimcp_error_t handle_omni_wm_update(
    const void* msg,
    size_t msg_size,
    nimcp_bio_promise_t response_promise,
    void* user_data)
{
    NIMCP_CHECK_THROW(msg, NIMCP_ERROR_NULL_POINTER, "message is NULL");
    NIMCP_CHECK_THROW(user_data, NIMCP_ERROR_NULL_POINTER, "user_data is NULL");

    NIMCP_CHECK_THROW(msg_size >= sizeof(bio_msg_omni_wm_update_t), NIMCP_ERROR_INVALID_PARAM,
                      "Update request too small: expected %zu bytes", sizeof(bio_msg_omni_wm_update_t));

    const bio_msg_omni_wm_update_t* req = (const bio_msg_omni_wm_update_t*)msg;
    omni_world_model_t* wm = (omni_world_model_t*)user_data;

    NIMCP_LOGGING_DEBUG("Received WM update request: state_dim=%u, action_dim=%u, reward=%.3f",
                        req->state_dim, req->action_dim, req->reward);

    /* Validate dimensions */
    NIMCP_CHECK_THROW(req->state_dim > 0 && req->state_dim <= OMNI_WM_MAX_STATE_DIM &&
                      req->action_dim <= OMNI_WM_MAX_ACTION_DIM, NIMCP_ERROR_INVALID_PARAM,
                      "Invalid dimensions in update request");

    /* Create temporary states from request */
    omni_wm_state_t* state = omni_wm_state_from_values(req->state, req->state_dim);
    omni_wm_state_t* next_state = omni_wm_state_from_values(req->next_state, req->state_dim);

    if (!state || !next_state) {
        if (state) omni_wm_state_destroy(state);
        if (next_state) omni_wm_state_destroy(next_state);
        return NIMCP_ERROR_NO_MEMORY;
    }

    /* Prepare response */
    bio_msg_omni_wm_update_ack_t response = {0};
    bio_msg_init_header(&response.header, BIO_MSG_OMNI_WM_UPDATE,
                        BIO_MODULE_OMNI_WORLD_MODEL,
                        req->header.source_module,
                        sizeof(response));

    /* Get prediction error before update for reporting */
    omni_wm_transition_t pred_transition = {0};
    omni_wm_set_state(wm, state);
    omni_wm_predict_forward(wm, req->action, req->action_dim, &pred_transition);

    float prediction_error = 0.0f;
    if (pred_transition.next_state) {
        /* Compute MSE between predicted and actual next state */
        for (uint32_t i = 0; i < req->state_dim && i < pred_transition.next_state->dim; i++) {
            float diff = pred_transition.next_state->values[i] - req->next_state[i];
            prediction_error += diff * diff;
        }
        prediction_error = sqrtf(prediction_error / req->state_dim);
        omni_wm_state_destroy(pred_transition.next_state);
    }
    if (pred_transition.action_taken) {
        nimcp_free(pred_transition.action_taken);
    }

    /* Perform the update */
    nimcp_error_t result = omni_wm_update(wm, state, req->action, req->action_dim,
                                           next_state, req->reward);

    /* Fill response */
    response.status = result;
    response.prediction_error = prediction_error;

    /* Cleanup */
    omni_wm_state_destroy(state);
    omni_wm_state_destroy(next_state);

    /* Send response via promise */
    if (response_promise) {
        nimcp_bio_promise_complete_sized(response_promise, &response, sizeof(response));
    }

    NIMCP_LOGGING_DEBUG("WM update completed: status=%d, pred_error=%.4f",
                        result, prediction_error);

    return result;
}

/**
 * @brief Handle rollout request message
 *
 * WHAT: Execute policy rollout for evaluation
 * WHY:  Enable policy comparison via expected free energy
 * HOW:  Simulate trajectory using provided action sequence
 */
static nimcp_error_t handle_omni_wm_rollout(
    const void* msg,
    size_t msg_size,
    nimcp_bio_promise_t response_promise,
    void* user_data)
{
    NIMCP_CHECK_THROW(msg, NIMCP_ERROR_NULL_POINTER, "message is NULL");
    NIMCP_CHECK_THROW(user_data, NIMCP_ERROR_NULL_POINTER, "user_data is NULL");

    NIMCP_CHECK_THROW(msg_size >= sizeof(bio_msg_omni_wm_rollout_t), NIMCP_ERROR_INVALID_PARAM,
                      "Rollout request too small: expected %zu bytes", sizeof(bio_msg_omni_wm_rollout_t));

    const bio_msg_omni_wm_rollout_t* req = (const bio_msg_omni_wm_rollout_t*)msg;
    omni_world_model_t* wm = (omni_world_model_t*)user_data;

    NIMCP_LOGGING_DEBUG("Received WM rollout request: horizon=%u, state_dim=%u",
                        req->horizon, req->state_dim);

    /* Validate dimensions */
    NIMCP_CHECK_THROW(req->state_dim > 0 && req->state_dim <= OMNI_WM_MAX_STATE_DIM &&
                      req->action_dim <= OMNI_WM_MAX_ACTION_DIM &&
                      req->horizon <= OMNI_WM_MAX_HORIZON &&
                      req->obs_dim <= OMNI_WM_MAX_OBS_DIM, NIMCP_ERROR_INVALID_PARAM,
                      "Invalid dimensions in rollout request");

    /* Create initial state from request */
    omni_wm_state_t* initial_state = omni_wm_state_from_values(req->initial_state,
                                                                req->state_dim);
    NIMCP_CHECK_THROW(initial_state, NIMCP_ERROR_NO_MEMORY, "failed to create initial state for rollout");

    /* Prepare response */
    bio_msg_omni_wm_rollout_response_t response = {0};
    bio_msg_init_header(&response.header, BIO_MSG_OMNI_WM_ROLLOUT,
                        BIO_MODULE_OMNI_WORLD_MODEL,
                        req->header.source_module,
                        sizeof(response));

    /* Execute rollout by manually stepping through actions */
    omni_wm_set_state(wm, initial_state);

    float total_reward = 0.0f;
    uint32_t steps_completed = 0;
    omni_wm_state_t* current_state = omni_wm_state_clone(initial_state);

    for (uint32_t t = 0; t < req->horizon && current_state; t++) {
        /* Get action for this timestep */
        const float* action = &req->policy_actions[t * req->action_dim];

        /* Predict next state */
        omni_wm_transition_t transition = {0};
        nimcp_error_t step_result = omni_wm_predict_forward(wm, action,
                                                             req->action_dim, &transition);

        if (step_result != NIMCP_SUCCESS || !transition.next_state) {
            if (transition.action_taken) nimcp_free(transition.action_taken);
            break;
        }

        /* Accumulate reward (using negative prediction error as proxy) */
        total_reward += -transition.prediction_error;
        steps_completed++;

        /* Move to next state */
        omni_wm_state_destroy(current_state);
        current_state = transition.next_state;
        omni_wm_set_state(wm, current_state);

        if (transition.action_taken) nimcp_free(transition.action_taken);
    }

    /* Compute expected free energy */
    float efe = 0.0f;
    if (req->obs_dim > 0 && current_state) {
        /* Predict observations from final state */
        float predicted_obs[OMNI_WM_MAX_OBS_DIM];
        omni_wm_predict_observations(wm, current_state, predicted_obs, req->obs_dim);

        /* EFE = divergence from preferred + uncertainty */
        for (uint32_t i = 0; i < req->obs_dim; i++) {
            /* Phase 8: Loop progress heartbeat */
            if ((i & 0xFF) == 0 && req->obs_dim > 256) {
                omni_world_model_heartbeat("omni_world_m_loop",
                                 (float)(i + 1) / (float)req->obs_dim);
            }

            float diff = predicted_obs[i] - req->preferred_obs[i];
            efe += diff * diff;
        }
        efe = sqrtf(efe / req->obs_dim);
    }

    /* Fill response */
    response.total_reward = total_reward;
    response.expected_free_energy = efe;
    response.steps_completed = steps_completed;

    if (current_state) {
        uint32_t copy_dim = current_state->dim;
        if (copy_dim > OMNI_WM_MAX_STATE_DIM) {
            copy_dim = OMNI_WM_MAX_STATE_DIM;
        }
        memcpy(response.final_state, current_state->values, copy_dim * sizeof(float));
        response.state_dim = copy_dim;
        omni_wm_state_destroy(current_state);
    }

    /* Cleanup */
    omni_wm_state_destroy(initial_state);

    /* Send response via promise */
    if (response_promise) {
        nimcp_bio_promise_complete_sized(response_promise, &response, sizeof(response));
    }

    NIMCP_LOGGING_DEBUG("WM rollout completed: steps=%u, reward=%.3f, efe=%.3f",
                        steps_completed, total_reward, efe);

    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Bio-Async Integration - Connection Management
 * ============================================================================ */

nimcp_error_t omni_wm_connect_bio_async(omni_world_model_t* wm) {
    /* Phase 8: Heartbeat at operation start */
    omni_world_model_heartbeat("omni_world_m_omni_wm_connect_bio_", 0.0f);


    NIMCP_CHECK_THROW(wm, NIMCP_ERROR_INVALID_PARAM, "world model is NULL");

    /* Already connected? */
    if (wm->bio_async_connected) {
        NIMCP_LOGGING_DEBUG("Omni world model already connected to bio-async");
        return NIMCP_SUCCESS;
    }

    /* Check if router is available */
    if (!bio_router_is_initialized()) {
        NIMCP_LOGGING_WARN("Bio-async router not initialized, skipping registration");
        return NIMCP_SUCCESS;
    }

    /* Register module with router */
    bio_module_info_t info = {
        .module_id = BIO_MODULE_OMNI_WORLD_MODEL,
        .module_name = "omni_world_model",
        .inbox_capacity = 64,
        .user_data = wm
    };

    wm->bio_ctx = bio_router_register_module(&info);
    if (!wm->bio_ctx) {
        NIMCP_LOGGING_WARN("Failed to register omni world model with bio-async router");
        return NIMCP_SUCCESS; /* Non-fatal: module can operate without bio-async */
    }

    /* Register message handlers */
    nimcp_error_t result;

    result = bio_router_register_handler(wm->bio_ctx, BIO_MSG_OMNI_WM_PREDICT,
                                          handle_omni_wm_predict);
    if (result != NIMCP_SUCCESS) {
        NIMCP_LOGGING_WARN("Failed to register PREDICT handler: %d", result);
    }

    result = bio_router_register_handler(wm->bio_ctx, BIO_MSG_OMNI_WM_COUNTERFACTUAL,
                                          handle_omni_wm_counterfactual);
    if (result != NIMCP_SUCCESS) {
        NIMCP_LOGGING_WARN("Failed to register COUNTERFACTUAL handler: %d", result);
    }

    result = bio_router_register_handler(wm->bio_ctx, BIO_MSG_OMNI_WM_UPDATE,
                                          handle_omni_wm_update);
    if (result != NIMCP_SUCCESS) {
        NIMCP_LOGGING_WARN("Failed to register UPDATE handler: %d", result);
    }

    result = bio_router_register_handler(wm->bio_ctx, BIO_MSG_OMNI_WM_ROLLOUT,
                                          handle_omni_wm_rollout);
    if (result != NIMCP_SUCCESS) {
        NIMCP_LOGGING_WARN("Failed to register ROLLOUT handler: %d", result);
    }

    wm->bio_async_connected = true;
    NIMCP_LOGGING_INFO("Omni world model connected to bio-async with 4 handlers");

    return NIMCP_SUCCESS;
}

nimcp_error_t omni_wm_disconnect_bio_async(omni_world_model_t* wm) {
    /* Phase 8: Heartbeat at operation start */
    omni_world_model_heartbeat("omni_world_m_omni_wm_disconnect_b", 0.0f);


    NIMCP_CHECK_THROW(wm, NIMCP_ERROR_INVALID_PARAM, "world model is NULL");

    /* Not connected? */
    if (!wm->bio_async_connected) {
        return NIMCP_SUCCESS;
    }

    /* Unregister handlers */
    if (wm->bio_ctx) {
        bio_router_unregister_handler(wm->bio_ctx, BIO_MSG_OMNI_WM_PREDICT);
        bio_router_unregister_handler(wm->bio_ctx, BIO_MSG_OMNI_WM_COUNTERFACTUAL);
        bio_router_unregister_handler(wm->bio_ctx, BIO_MSG_OMNI_WM_UPDATE);
        bio_router_unregister_handler(wm->bio_ctx, BIO_MSG_OMNI_WM_ROLLOUT);

        /* Unregister module */
        bio_router_unregister_module(wm->bio_ctx);
        wm->bio_ctx = NULL;
    }

    wm->bio_async_connected = false;
    NIMCP_LOGGING_INFO("Omni world model disconnected from bio-async");

    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Serialization Helpers
 * ============================================================================ */

/**
 * @brief Simple CRC32 implementation for checksum
 */
static uint32_t crc32_compute(const uint8_t* data, size_t length) {
    uint32_t crc = 0xFFFFFFFF;
    for (size_t i = 0; i < length; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && length > 256) {
            omni_world_model_heartbeat("omni_world_m_loop",
                             (float)(i + 1) / (float)length);
        }

        crc ^= data[i];
        for (int j = 0; j < 8; j++) {
            /* Phase 8: Loop progress heartbeat */
            if ((j & 0xFF) == 0 && 8 > 256) {
                omni_world_model_heartbeat("omni_world_m_loop",
                                 (float)(j + 1) / (float)8);
            }

            crc = (crc >> 1) ^ (0xEDB88320 & -(crc & 1));
        }
    }
    return ~crc;
}

/**
 * @brief Write uint8 to buffer (big-endian)
 */
static size_t write_u8(uint8_t* buf, size_t pos, uint8_t val) {
    if (buf) buf[pos] = val;
    return pos + 1;
}

/**
 * @brief Write uint32 to buffer (big-endian)
 */
static size_t write_u32(uint8_t* buf, size_t pos, uint32_t val) {
    if (buf) {
        buf[pos] = (val >> 24) & 0xFF;
        buf[pos + 1] = (val >> 16) & 0xFF;
        buf[pos + 2] = (val >> 8) & 0xFF;
        buf[pos + 3] = val & 0xFF;
    }
    return pos + 4;
}

/**
 * @brief Write uint64 to buffer (big-endian)
 */
static size_t write_u64(uint8_t* buf, size_t pos, uint64_t val) {
    if (buf) {
        buf[pos] = (val >> 56) & 0xFF;
        buf[pos + 1] = (val >> 48) & 0xFF;
        buf[pos + 2] = (val >> 40) & 0xFF;
        buf[pos + 3] = (val >> 32) & 0xFF;
        buf[pos + 4] = (val >> 24) & 0xFF;
        buf[pos + 5] = (val >> 16) & 0xFF;
        buf[pos + 6] = (val >> 8) & 0xFF;
        buf[pos + 7] = val & 0xFF;
    }
    return pos + 8;
}

/**
 * @brief Write float to buffer (as uint32, big-endian)
 */
static size_t write_float_be(uint8_t* buf, size_t pos, float val) {
    union { float f; uint32_t i; } conv;
    conv.f = val;
    return write_u32(buf, pos, conv.i);
}

/**
 * @brief Write double to buffer (as uint64, big-endian)
 */
static size_t write_double_be(uint8_t* buf, size_t pos, double val) {
    union { double d; uint64_t i; } conv;
    conv.d = val;
    return write_u64(buf, pos, conv.i);
}

/**
 * @brief Read uint8 from buffer
 */
static uint8_t read_u8(const uint8_t* buf, size_t* pos) {
    uint8_t val = buf[*pos];
    (*pos)++;
    return val;
}

/**
 * @brief Read uint32 from buffer (big-endian)
 */
static uint32_t read_u32(const uint8_t* buf, size_t* pos) {
    uint32_t val = ((uint32_t)buf[*pos] << 24) |
                   ((uint32_t)buf[*pos + 1] << 16) |
                   ((uint32_t)buf[*pos + 2] << 8) |
                   buf[*pos + 3];
    (*pos) += 4;
    return val;
}

/**
 * @brief Read uint64 from buffer (big-endian)
 */
static uint64_t read_u64(const uint8_t* buf, size_t* pos) {
    uint64_t val = ((uint64_t)buf[*pos] << 56) |
                   ((uint64_t)buf[*pos + 1] << 48) |
                   ((uint64_t)buf[*pos + 2] << 40) |
                   ((uint64_t)buf[*pos + 3] << 32) |
                   ((uint64_t)buf[*pos + 4] << 24) |
                   ((uint64_t)buf[*pos + 5] << 16) |
                   ((uint64_t)buf[*pos + 6] << 8) |
                   buf[*pos + 7];
    (*pos) += 8;
    return val;
}

/**
 * @brief Read float from buffer
 */
static float read_float_be(const uint8_t* buf, size_t* pos) {
    union { uint32_t i; float f; } conv;
    conv.i = read_u32(buf, pos);
    return conv.f;
}

/**
 * @brief Read double from buffer
 */
static double read_double_be(const uint8_t* buf, size_t* pos) {
    union { uint64_t i; double d; } conv;
    conv.i = read_u64(buf, pos);
    return conv.d;
}

/* ============================================================================
 * Serialization Implementation
 * ============================================================================ */

/**
 * @brief Serialize config section
 */
static size_t serialize_config(uint8_t* buf, size_t pos, const omni_wm_config_t* cfg) {
    /* Dimensionality settings */
    pos = write_u32(buf, pos, cfg->state_dim);
    pos = write_u32(buf, pos, cfg->action_dim);
    pos = write_u32(buf, pos, cfg->obs_dim);
    pos = write_u32(buf, pos, cfg->hidden_dim);
    pos = write_u32(buf, pos, cfg->num_levels);

    /* RSSM settings */
    pos = write_u32(buf, pos, cfg->rssm_h_dim);
    pos = write_u32(buf, pos, cfg->rssm_z_dim);
    pos = write_u32(buf, pos, cfg->latent_dim);

    /* MDN settings */
    pos = write_u32(buf, pos, cfg->mdn_components);
    pos = write_u8(buf, pos, (uint8_t)cfg->pred_type);

    /* Learning settings */
    pos = write_float_be(buf, pos, cfg->learning_rate);
    pos = write_float_be(buf, pos, cfg->discount_factor);
    pos = write_float_be(buf, pos, cfg->kl_weight);
    pos = write_float_be(buf, pos, cfg->reward_scale);
    pos = write_u8(buf, pos, (uint8_t)cfg->learn_mode);

    /* Experience replay settings */
    pos = write_u32(buf, pos, cfg->replay_buffer_size);
    pos = write_u32(buf, pos, cfg->batch_size);
    pos = write_float_be(buf, pos, cfg->priority_exponent);

    /* Dreaming settings */
    pos = write_u32(buf, pos, cfg->dream_horizon);
    pos = write_u32(buf, pos, cfg->dream_episodes);
    pos = write_float_be(buf, pos, cfg->imagination_noise);

    /* Feature flags (packed as byte) */
    uint8_t flags = 0;
    if (cfg->enable_lateral) flags |= 0x01;
    if (cfg->enable_hierarchical) flags |= 0x02;
    if (cfg->enable_dreaming) flags |= 0x04;
    if (cfg->use_symlog_rewards) flags |= 0x08;
    if (cfg->use_rssm) flags |= 0x10;
    if (cfg->use_mdn) flags |= 0x20;
    pos = write_u8(buf, pos, flags);

    return pos;
}

/**
 * @brief Deserialize config section
 */
static size_t deserialize_config(const uint8_t* buf, size_t pos, omni_wm_config_t* cfg) {
    /* Dimensionality settings */
    cfg->state_dim = read_u32(buf, &pos);
    cfg->action_dim = read_u32(buf, &pos);
    cfg->obs_dim = read_u32(buf, &pos);
    cfg->hidden_dim = read_u32(buf, &pos);
    cfg->num_levels = read_u32(buf, &pos);

    /* RSSM settings */
    cfg->rssm_h_dim = read_u32(buf, &pos);
    cfg->rssm_z_dim = read_u32(buf, &pos);
    cfg->latent_dim = read_u32(buf, &pos);

    /* MDN settings */
    cfg->mdn_components = read_u32(buf, &pos);
    cfg->pred_type = (omni_wm_prediction_type_t)read_u8(buf, &pos);

    /* Learning settings */
    cfg->learning_rate = read_float_be(buf, &pos);
    cfg->discount_factor = read_float_be(buf, &pos);
    cfg->kl_weight = read_float_be(buf, &pos);
    cfg->reward_scale = read_float_be(buf, &pos);
    cfg->learn_mode = (omni_wm_learn_mode_t)read_u8(buf, &pos);

    /* Experience replay settings */
    cfg->replay_buffer_size = read_u32(buf, &pos);
    cfg->batch_size = read_u32(buf, &pos);
    cfg->priority_exponent = read_float_be(buf, &pos);

    /* Dreaming settings */
    cfg->dream_horizon = read_u32(buf, &pos);
    cfg->dream_episodes = read_u32(buf, &pos);
    cfg->imagination_noise = read_float_be(buf, &pos);

    /* Feature flags */
    uint8_t flags = read_u8(buf, &pos);
    cfg->enable_lateral = (flags & 0x01) != 0;
    cfg->enable_hierarchical = (flags & 0x02) != 0;
    cfg->enable_dreaming = (flags & 0x04) != 0;
    cfg->use_symlog_rewards = (flags & 0x08) != 0;
    cfg->use_rssm = (flags & 0x10) != 0;
    cfg->use_mdn = (flags & 0x20) != 0;

    return pos;
}

/**
 * @brief Serialize state
 */
static size_t serialize_state(uint8_t* buf, size_t pos, const omni_wm_state_t* state) {
    if (!state) {
        pos = write_u8(buf, pos, 0); /* null marker */
        return pos;
    }

    pos = write_u8(buf, pos, 1); /* present marker */
    pos = write_u32(buf, pos, state->dim);
    pos = write_float_be(buf, pos, state->uncertainty);
    pos = write_double_be(buf, pos, state->timestamp);
    pos = write_u32(buf, pos, state->level);

    /* Write values array */
    for (uint32_t i = 0; i < state->dim; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && state->dim > 256) {
            omni_world_model_heartbeat("omni_world_m_loop",
                             (float)(i + 1) / (float)state->dim);
        }

        pos = write_float_be(buf, pos, state->values[i]);
    }

    return pos;
}

/**
 * @brief Deserialize state
 */
static omni_wm_state_t* deserialize_state_from_buf(const uint8_t* buf, size_t* pos) {
    uint8_t present = read_u8(buf, pos);
    if (!present) return NULL;

    uint32_t dim = read_u32(buf, pos);
    omni_wm_state_t* state = omni_wm_state_create(dim);
    if (!state) return NULL;

    state->uncertainty = read_float_be(buf, pos);
    state->timestamp = read_double_be(buf, pos);
    state->level = read_u32(buf, pos);

    for (uint32_t i = 0; i < dim; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && dim > 256) {
            omni_world_model_heartbeat("omni_world_m_loop",
                             (float)(i + 1) / (float)dim);
        }

        state->values[i] = read_float_be(buf, pos);
    }

    return state;
}

/**
 * @brief Serialize RSSM state
 */
static size_t serialize_rssm_state(uint8_t* buf, size_t pos, const omni_wm_rssm_state_t* state) {
    if (!state) {
        pos = write_u8(buf, pos, 0);
        return pos;
    }

    pos = write_u8(buf, pos, 1);
    pos = write_u32(buf, pos, state->h_dim);
    pos = write_u32(buf, pos, state->z_dim);
    pos = write_double_be(buf, pos, state->timestamp);

    /* Write h array */
    for (uint32_t i = 0; i < state->h_dim; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && state->h_dim > 256) {
            omni_world_model_heartbeat("omni_world_m_loop",
                             (float)(i + 1) / (float)state->h_dim);
        }

        pos = write_float_be(buf, pos, state->h[i]);
    }

    /* Write z array */
    for (uint32_t i = 0; i < state->z_dim; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && state->z_dim > 256) {
            omni_world_model_heartbeat("omni_world_m_loop",
                             (float)(i + 1) / (float)state->z_dim);
        }

        pos = write_float_be(buf, pos, state->z[i]);
    }

    /* Write z_mean array */
    for (uint32_t i = 0; i < state->z_dim; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && state->z_dim > 256) {
            omni_world_model_heartbeat("omni_world_m_loop",
                             (float)(i + 1) / (float)state->z_dim);
        }

        pos = write_float_be(buf, pos, state->z_mean[i]);
    }

    /* Write z_std array */
    for (uint32_t i = 0; i < state->z_dim; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && state->z_dim > 256) {
            omni_world_model_heartbeat("omni_world_m_loop",
                             (float)(i + 1) / (float)state->z_dim);
        }

        pos = write_float_be(buf, pos, state->z_std[i]);
    }

    return pos;
}

/**
 * @brief Deserialize RSSM state
 */
static omni_wm_rssm_state_t* deserialize_rssm_state_from_buf(const uint8_t* buf, size_t* pos) {
    uint8_t present = read_u8(buf, pos);
    if (!present) return NULL;

    uint32_t h_dim = read_u32(buf, pos);
    uint32_t z_dim = read_u32(buf, pos);

    omni_wm_rssm_state_t* state = omni_wm_rssm_state_create(h_dim, z_dim);
    if (!state) return NULL;

    state->timestamp = read_double_be(buf, pos);

    for (uint32_t i = 0; i < h_dim; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && h_dim > 256) {
            omni_world_model_heartbeat("omni_world_m_loop",
                             (float)(i + 1) / (float)h_dim);
        }

        state->h[i] = read_float_be(buf, pos);
    }

    for (uint32_t i = 0; i < z_dim; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && z_dim > 256) {
            omni_world_model_heartbeat("omni_world_m_loop",
                             (float)(i + 1) / (float)z_dim);
        }

        state->z[i] = read_float_be(buf, pos);
    }

    for (uint32_t i = 0; i < z_dim; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && z_dim > 256) {
            omni_world_model_heartbeat("omni_world_m_loop",
                             (float)(i + 1) / (float)z_dim);
        }

        state->z_mean[i] = read_float_be(buf, pos);
    }

    for (uint32_t i = 0; i < z_dim; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && z_dim > 256) {
            omni_world_model_heartbeat("omni_world_m_loop",
                             (float)(i + 1) / (float)z_dim);
        }

        state->z_std[i] = read_float_be(buf, pos);
    }

    return state;
}

/**
 * @brief Serialize dynamics weights
 */
static size_t serialize_dynamics_weights(uint8_t* buf, size_t pos, const omni_wm_dynamics_t* dyn) {
    if (!dyn) {
        pos = write_u8(buf, pos, 0);
        return pos;
    }

    pos = write_u8(buf, pos, 1);
    pos = write_u32(buf, pos, dyn->h_dim);
    pos = write_u32(buf, pos, dyn->z_dim);
    pos = write_u32(buf, pos, dyn->obs_dim);
    pos = write_u32(buf, pos, dyn->action_dim);

    uint32_t input_dim = dyn->h_dim + dyn->z_dim + dyn->action_dim;

    /* W_h: input_dim * h_dim */
    for (uint32_t i = 0; i < input_dim * dyn->h_dim; i++) {
        pos = write_float_be(buf, pos, dyn->W_h[i]);
    }

    /* W_z: h_dim * z_dim * 2 */
    for (uint32_t i = 0; i < dyn->h_dim * dyn->z_dim * 2; i++) {
        pos = write_float_be(buf, pos, dyn->W_z[i]);
    }

    /* W_obs: (h_dim + z_dim) * obs_dim */
    for (uint32_t i = 0; i < (dyn->h_dim + dyn->z_dim) * dyn->obs_dim; i++) {
        pos = write_float_be(buf, pos, dyn->W_obs[i]);
    }

    /* Biases */
    for (uint32_t i = 0; i < dyn->h_dim; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && dyn->h_dim > 256) {
            omni_world_model_heartbeat("omni_world_m_loop",
                             (float)(i + 1) / (float)dyn->h_dim);
        }

        pos = write_float_be(buf, pos, dyn->b_h[i]);
    }
    for (uint32_t i = 0; i < dyn->z_dim * 2; i++) {
        pos = write_float_be(buf, pos, dyn->b_z[i]);
    }
    for (uint32_t i = 0; i < dyn->obs_dim; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && dyn->obs_dim > 256) {
            omni_world_model_heartbeat("omni_world_m_loop",
                             (float)(i + 1) / (float)dyn->obs_dim);
        }

        pos = write_float_be(buf, pos, dyn->b_obs[i]);
    }

    return pos;
}

/**
 * @brief Deserialize dynamics weights
 */
static omni_wm_dynamics_t* deserialize_dynamics_from_buf(const uint8_t* buf, size_t* pos) {
    uint8_t present = read_u8(buf, pos);
    if (!present) return NULL;

    uint32_t h_dim = read_u32(buf, pos);
    uint32_t z_dim = read_u32(buf, pos);
    uint32_t obs_dim = read_u32(buf, pos);
    uint32_t action_dim = read_u32(buf, pos);

    omni_wm_dynamics_t* dyn = dynamics_create(h_dim, z_dim, obs_dim, action_dim);
    if (!dyn) return NULL;

    uint32_t input_dim = h_dim + z_dim + action_dim;

    for (uint32_t i = 0; i < input_dim * h_dim; i++) {
        dyn->W_h[i] = read_float_be(buf, pos);
    }

    for (uint32_t i = 0; i < h_dim * z_dim * 2; i++) {
        dyn->W_z[i] = read_float_be(buf, pos);
    }

    for (uint32_t i = 0; i < (h_dim + z_dim) * obs_dim; i++) {
        dyn->W_obs[i] = read_float_be(buf, pos);
    }

    for (uint32_t i = 0; i < h_dim; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && h_dim > 256) {
            omni_world_model_heartbeat("omni_world_m_loop",
                             (float)(i + 1) / (float)h_dim);
        }

        dyn->b_h[i] = read_float_be(buf, pos);
    }
    for (uint32_t i = 0; i < z_dim * 2; i++) {
        dyn->b_z[i] = read_float_be(buf, pos);
    }
    for (uint32_t i = 0; i < obs_dim; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && obs_dim > 256) {
            omni_world_model_heartbeat("omni_world_m_loop",
                             (float)(i + 1) / (float)obs_dim);
        }

        dyn->b_obs[i] = read_float_be(buf, pos);
    }

    return dyn;
}

/**
 * @brief Serialize statistics
 */
static size_t serialize_wm_stats(uint8_t* buf, size_t pos, const omni_wm_stats_t* stats) {
    pos = write_u64(buf, pos, stats->forward_predictions);
    pos = write_u64(buf, pos, stats->backward_inferences);
    pos = write_u64(buf, pos, stats->lateral_predictions);
    pos = write_u64(buf, pos, stats->counterfactual_queries);
    pos = write_u64(buf, pos, stats->rollouts_completed);
    pos = write_u64(buf, pos, stats->model_updates);
    pos = write_float_be(buf, pos, stats->mean_prediction_error);
    pos = write_float_be(buf, pos, stats->mean_counterfactual_divergence);
    return pos;
}

/**
 * @brief Deserialize statistics
 */
static size_t deserialize_wm_stats(const uint8_t* buf, size_t pos, omni_wm_stats_t* stats) {
    stats->forward_predictions = read_u64(buf, &pos);
    stats->backward_inferences = read_u64(buf, &pos);
    stats->lateral_predictions = read_u64(buf, &pos);
    stats->counterfactual_queries = read_u64(buf, &pos);
    stats->rollouts_completed = read_u64(buf, &pos);
    stats->model_updates = read_u64(buf, &pos);
    stats->mean_prediction_error = read_float_be(buf, &pos);
    stats->mean_counterfactual_divergence = read_float_be(buf, &pos);
    return pos;
}

/* ============================================================================
 * Public Serialization API
 * ============================================================================ */

size_t omni_wm_serialize(const omni_world_model_t* wm,
                          uint8_t* buffer,
                          size_t buffer_size) {
    if (!wm) return 0;

    /* Phase 8: Heartbeat at operation start */
    omni_world_model_heartbeat("omni_world_m_omni_wm_serialize", 0.0f);


    size_t pos = 0;

    /* Header */
    pos = write_u32(buffer, pos, OMNI_WM_SERIAL_MAGIC);
    pos = write_u8(buffer, pos, OMNI_WM_SERIAL_VERSION);

    /* Compute flags */
    uint8_t flags = OMNI_WM_SERIAL_FLAG_HAS_DYNAMICS;
    if (wm->rssm_state) flags |= OMNI_WM_SERIAL_FLAG_HAS_RSSM;
    if (wm->replay_buffer && wm->replay_buffer->size > 0) {
        flags |= OMNI_WM_SERIAL_FLAG_HAS_REPLAY;
    }
    pos = write_u8(buffer, pos, flags);

    /* Config */
    pos = serialize_config(buffer, pos, &wm->config);

    /* Current state */
    pos = serialize_state(buffer, pos, wm->current_state);

    /* RSSM state */
    if (flags & OMNI_WM_SERIAL_FLAG_HAS_RSSM) {
        pos = serialize_rssm_state(buffer, pos, wm->rssm_state);
    }

    /* Dynamics models */
    pos = serialize_dynamics_weights(buffer, pos, wm->forward_dynamics);
    pos = serialize_dynamics_weights(buffer, pos, wm->backward_dynamics);
    pos = serialize_dynamics_weights(buffer, pos, wm->lateral_dynamics);

    /* Encoder/decoder weights */
    uint32_t enc_size = wm->config.obs_dim * wm->config.latent_dim;
    pos = write_u32(buffer, pos, enc_size);
    for (uint32_t i = 0; i < enc_size; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && enc_size > 256) {
            omni_world_model_heartbeat("omni_world_m_loop",
                             (float)(i + 1) / (float)enc_size);
        }

        pos = write_float_be(buffer, pos, wm->encoder_W[i]);
    }
    pos = write_u32(buffer, pos, wm->config.latent_dim);
    for (uint32_t i = 0; i < wm->config.latent_dim; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && wm->config.latent_dim > 256) {
            omni_world_model_heartbeat("omni_world_m_loop",
                             (float)(i + 1) / (float)wm->config.latent_dim);
        }

        pos = write_float_be(buffer, pos, wm->encoder_b[i]);
    }
    for (uint32_t i = 0; i < enc_size; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && enc_size > 256) {
            omni_world_model_heartbeat("omni_world_m_loop",
                             (float)(i + 1) / (float)enc_size);
        }

        pos = write_float_be(buffer, pos, wm->decoder_W[i]);
    }
    pos = write_u32(buffer, pos, wm->config.obs_dim);
    for (uint32_t i = 0; i < wm->config.obs_dim; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && wm->config.obs_dim > 256) {
            omni_world_model_heartbeat("omni_world_m_loop",
                             (float)(i + 1) / (float)wm->config.obs_dim);
        }

        pos = write_float_be(buffer, pos, wm->decoder_b[i]);
    }

    /* Replay buffer (limited serialization - just size info for now) */
    if (flags & OMNI_WM_SERIAL_FLAG_HAS_REPLAY) {
        pos = write_u32(buffer, pos, wm->replay_buffer->size);
        /* Note: Full replay buffer serialization would be very large.
         * For checkpointing, we just store the count. Full serialization
         * could be added as an option in a future version. */
    }

    /* Statistics */
    pos = serialize_wm_stats(buffer, pos, &wm->stats);

    /* Random seed */
    pos = write_u32(buffer, pos, wm->rand_seed);

    /* Checksum (computed over all preceding data) */
    if (buffer) {
        uint32_t checksum = crc32_compute(buffer, pos);
        pos = write_u32(buffer, pos, checksum);
    } else {
        pos += 4; /* Reserve space for checksum */
    }

    /* Verify buffer size if buffer provided */
    if (buffer && pos > buffer_size) {
        return 0; /* Buffer too small */
    }

    return pos;
}

omni_world_model_t* omni_wm_deserialize(const uint8_t* buffer,
                                         size_t buffer_size) {
    if (!buffer || buffer_size < 10) return NULL;

    /* Phase 8: Heartbeat at operation start */
    omni_world_model_heartbeat("omni_world_m_omni_wm_deserialize", 0.0f);


    size_t pos = 0;

    /* Verify magic */
    uint32_t magic = read_u32(buffer, &pos);
    if (magic != OMNI_WM_SERIAL_MAGIC) return NULL;

    /* Verify version */
    uint8_t version = read_u8(buffer, &pos);
    if (version > OMNI_WM_SERIAL_VERSION) return NULL;

    /* Read flags */
    uint8_t flags = read_u8(buffer, &pos);

    /* Read config */
    omni_wm_config_t config;
    pos = deserialize_config(buffer, pos, &config);

    /* Create world model with config */
    omni_world_model_t* wm = omni_wm_create(&config);
    if (!wm) return NULL;

    /* Read current state */
    omni_wm_state_t* state = deserialize_state_from_buf(buffer, &pos);
    if (state) {
        if (wm->current_state) {
            omni_wm_state_destroy(wm->current_state);
        }
        wm->current_state = state;
    }

    /* Read RSSM state */
    if (flags & OMNI_WM_SERIAL_FLAG_HAS_RSSM) {
        omni_wm_rssm_state_t* rssm = deserialize_rssm_state_from_buf(buffer, &pos);
        if (rssm) {
            if (wm->rssm_state) {
                omni_wm_rssm_state_destroy(wm->rssm_state);
            }
            wm->rssm_state = rssm;
        }
    }

    /* Read dynamics models */
    omni_wm_dynamics_t* forward = deserialize_dynamics_from_buf(buffer, &pos);
    if (forward) {
        dynamics_destroy(wm->forward_dynamics);
        wm->forward_dynamics = forward;
    }

    omni_wm_dynamics_t* backward = deserialize_dynamics_from_buf(buffer, &pos);
    if (backward) {
        dynamics_destroy(wm->backward_dynamics);
        wm->backward_dynamics = backward;
    }

    omni_wm_dynamics_t* lateral = deserialize_dynamics_from_buf(buffer, &pos);
    if (lateral) {
        dynamics_destroy(wm->lateral_dynamics);
        wm->lateral_dynamics = lateral;
    }

    /* Read encoder/decoder weights */
    uint32_t enc_size = read_u32(buffer, &pos);
    if (enc_size == wm->config.obs_dim * wm->config.latent_dim) {
        for (uint32_t i = 0; i < enc_size; i++) {
            /* Phase 8: Loop progress heartbeat */
            if ((i & 0xFF) == 0 && enc_size > 256) {
                omni_world_model_heartbeat("omni_world_m_loop",
                                 (float)(i + 1) / (float)enc_size);
            }

            wm->encoder_W[i] = read_float_be(buffer, &pos);
        }
    }

    uint32_t enc_b_size = read_u32(buffer, &pos);
    if (enc_b_size == wm->config.latent_dim) {
        for (uint32_t i = 0; i < enc_b_size; i++) {
            /* Phase 8: Loop progress heartbeat */
            if ((i & 0xFF) == 0 && enc_b_size > 256) {
                omni_world_model_heartbeat("omni_world_m_loop",
                                 (float)(i + 1) / (float)enc_b_size);
            }

            wm->encoder_b[i] = read_float_be(buffer, &pos);
        }
    }

    for (uint32_t i = 0; i < enc_size; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && enc_size > 256) {
            omni_world_model_heartbeat("omni_world_m_loop",
                             (float)(i + 1) / (float)enc_size);
        }

        wm->decoder_W[i] = read_float_be(buffer, &pos);
    }

    uint32_t dec_b_size = read_u32(buffer, &pos);
    if (dec_b_size == wm->config.obs_dim) {
        for (uint32_t i = 0; i < dec_b_size; i++) {
            /* Phase 8: Loop progress heartbeat */
            if ((i & 0xFF) == 0 && dec_b_size > 256) {
                omni_world_model_heartbeat("omni_world_m_loop",
                                 (float)(i + 1) / (float)dec_b_size);
            }

            wm->decoder_b[i] = read_float_be(buffer, &pos);
        }
    }

    /* Read replay buffer info */
    if (flags & OMNI_WM_SERIAL_FLAG_HAS_REPLAY) {
        uint32_t replay_size = read_u32(buffer, &pos);
        (void)replay_size; /* Note: Full replay restoration not implemented */
    }

    /* Read statistics */
    pos = deserialize_wm_stats(buffer, pos, &wm->stats);

    /* Read random seed */
    wm->rand_seed = read_u32(buffer, &pos);

    /* Verify checksum */
    if (pos + 4 <= buffer_size) {
        uint32_t stored_checksum = read_u32(buffer, &pos);
        uint32_t computed_checksum = crc32_compute(buffer, pos - 4);
        if (stored_checksum != computed_checksum) {
            omni_wm_destroy(wm);
            return NULL;
        }
    }

    return wm;
}

nimcp_error_t omni_wm_save(const omni_world_model_t* wm,
                            const char* filepath) {
    /* Phase 8: Heartbeat at operation start */
    omni_world_model_heartbeat("omni_world_m_omni_wm_save", 0.0f);


    NIMCP_CHECK_THROW(wm, NIMCP_ERROR_INVALID_PARAM, "world model is NULL");
    NIMCP_CHECK_THROW(filepath, NIMCP_ERROR_INVALID_PARAM, "filepath is NULL");

    /* Get required size */
    size_t required_size = omni_wm_serialize(wm, NULL, 0);
    NIMCP_CHECK_THROW(required_size > 0, NIMCP_ERROR_SERIALIZATION, "failed to compute serialization size");

    /* Allocate buffer */
    uint8_t* buffer = nimcp_malloc(required_size);
    NIMCP_CHECK_THROW(buffer, NIMCP_ERROR_NO_MEMORY, "failed to allocate serialization buffer");

    /* Serialize */
    size_t written = omni_wm_serialize(wm, buffer, required_size);
    if (written == 0) {
        nimcp_free(buffer);
        return NIMCP_ERROR_SERIALIZATION;
    }

    /* Write to file */
    FILE* fp = fopen(filepath, "wb");
    if (!fp) {
        nimcp_free(buffer);
        return NIMCP_ERROR_FILE_OPEN;
    }

    size_t bytes_written = fwrite(buffer, 1, written, fp);
    fclose(fp);
    nimcp_free(buffer);

    if (bytes_written != written) {
        return NIMCP_ERROR_FILE_WRITE;
    }

    return NIMCP_SUCCESS;
}

omni_world_model_t* omni_wm_load(const char* filepath) {
    if (!filepath) return NULL;

    /* Open file */
    /* Phase 8: Heartbeat at operation start */
    omni_world_model_heartbeat("omni_world_m_omni_wm_load", 0.0f);


    FILE* fp = fopen(filepath, "rb");
    if (!fp) return NULL;

    /* Get file size */
    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    if (file_size <= 0) {
        fclose(fp);
        return NULL;
    }

    /* Allocate buffer */
    uint8_t* buffer = nimcp_malloc((size_t)file_size);
    if (!buffer) {
        fclose(fp);
        return NULL;
    }

    /* Read file */
    size_t bytes_read = fread(buffer, 1, (size_t)file_size, fp);
    fclose(fp);

    if (bytes_read != (size_t)file_size) {
        nimcp_free(buffer);
        return NULL;
    }

    /* Deserialize */
    omni_world_model_t* wm = omni_wm_deserialize(buffer, (size_t)file_size);
    nimcp_free(buffer);

    return wm;
}

/* ============================================================================
 * Checkpoint Management
 * ============================================================================ */

/**
 * @brief Initialize checkpoint store
 */
static omni_wm_checkpoint_store_t* checkpoint_store_create(void) {
    omni_wm_checkpoint_store_t* store = nimcp_calloc(1, sizeof(omni_wm_checkpoint_store_t));
    if (!store) return NULL;
    store->next_id = 1;
    return store;
}

/**
 * @brief Destroy checkpoint store - used in cleanup
 */
static void checkpoint_store_destroy_internal(omni_wm_checkpoint_store_t* store) {
    if (!store) return;

    for (uint32_t i = 0; i < store->count; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && store->count > 256) {
            omni_world_model_heartbeat("omni_world_m_loop",
                             (float)(i + 1) / (float)store->count);
        }

        nimcp_free(store->checkpoints[i].data);
    }
    nimcp_free(store);
}

uint64_t omni_wm_checkpoint(omni_world_model_t* wm) {
    if (!wm) return 0;

    /* Create checkpoint store if needed */
    if (!wm->checkpoint_store) {
        wm->checkpoint_store = checkpoint_store_create();
        if (!wm->checkpoint_store) return 0;
    }

    /* Check capacity */
    /* Phase 8: Heartbeat at operation start */
    omni_world_model_heartbeat("omni_world_m_omni_wm_checkpoint", 0.0f);


    if (wm->checkpoint_store->count >= OMNI_WM_MAX_CHECKPOINTS) {
        return 0; /* No room for more checkpoints */
    }

    /* Serialize current state */
    size_t required_size = omni_wm_serialize(wm, NULL, 0);
    if (required_size == 0) return 0;

    uint8_t* data = nimcp_malloc(required_size);
    if (!data) return 0;

    size_t written = omni_wm_serialize(wm, data, required_size);
    if (written == 0) {
        nimcp_free(data);
        return 0;
    }

    /* Create checkpoint */
    uint32_t idx = wm->checkpoint_store->count;
    uint64_t id = wm->checkpoint_store->next_id++;

    wm->checkpoint_store->checkpoints[idx].id = id;
    wm->checkpoint_store->checkpoints[idx].data = data;
    wm->checkpoint_store->checkpoints[idx].data_size = written;
    wm->checkpoint_store->checkpoints[idx].timestamp = (double)time(NULL);
    memset(wm->checkpoint_store->checkpoints[idx].description, 0, 64);

    wm->checkpoint_store->count++;

    return id;
}

nimcp_error_t omni_wm_restore_checkpoint(omni_world_model_t* wm,
                                          uint64_t checkpoint_id) {
    /* Phase 8: Heartbeat at operation start */
    omni_world_model_heartbeat("omni_world_m_omni_wm_restore_chec", 0.0f);


    NIMCP_CHECK_THROW(wm, NIMCP_ERROR_INVALID_PARAM, "world model is NULL");
    NIMCP_CHECK_THROW(wm->checkpoint_store, NIMCP_ERROR_INVALID_PARAM, "checkpoint store is NULL");
    NIMCP_CHECK_THROW(checkpoint_id != 0, NIMCP_ERROR_INVALID_PARAM, "checkpoint_id is zero");

    /* Find checkpoint */
    omni_wm_checkpoint_t* cp = NULL;
    for (uint32_t i = 0; i < wm->checkpoint_store->count; i++) {
        if (wm->checkpoint_store->checkpoints[i].id == checkpoint_id) {
            cp = &wm->checkpoint_store->checkpoints[i];
            break;
        }
    }

    NIMCP_CHECK_THROW(cp, NIMCP_ERROR_NOT_FOUND, "checkpoint not found");

    /* Deserialize into temporary */
    omni_world_model_t* restored = omni_wm_deserialize(cp->data, cp->data_size);
    NIMCP_CHECK_THROW(restored, NIMCP_ERROR_INVALID_STATE, "failed to deserialize checkpoint");

    /* Preserve checkpoint store and mutex from current wm */
    omni_wm_checkpoint_store_t* store = wm->checkpoint_store;
    nimcp_mutex_t* mutex = wm->mutex;

    /* Swap internals (avoiding checkpoint store and mutex) */
    /* Free current state */
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

    /* Copy restored state */
    wm->config = restored->config;
    wm->current_state = restored->current_state;
    wm->rssm_state = restored->rssm_state;
    wm->forward_dynamics = restored->forward_dynamics;
    wm->backward_dynamics = restored->backward_dynamics;
    wm->lateral_dynamics = restored->lateral_dynamics;
    wm->encoder_W = restored->encoder_W;
    wm->encoder_b = restored->encoder_b;
    wm->decoder_W = restored->decoder_W;
    wm->decoder_b = restored->decoder_b;
    wm->replay_buffer = restored->replay_buffer;
    wm->stats = restored->stats;
    wm->rand_seed = restored->rand_seed;

    /* Restore preserved items */
    wm->checkpoint_store = store;
    wm->mutex = mutex;

    /* Free the temporary wrapper (but not its contents - now owned by wm) */
    restored->current_state = NULL;
    restored->rssm_state = NULL;
    restored->forward_dynamics = NULL;
    restored->backward_dynamics = NULL;
    restored->lateral_dynamics = NULL;
    restored->encoder_W = NULL;
    restored->encoder_b = NULL;
    restored->decoder_W = NULL;
    restored->decoder_b = NULL;
    restored->replay_buffer = NULL;
    restored->checkpoint_store = NULL;
    restored->mutex = NULL;
    omni_wm_destroy(restored);

    return NIMCP_SUCCESS;
}

nimcp_error_t omni_wm_delete_checkpoint(omni_world_model_t* wm,
                                         uint64_t checkpoint_id) {
    /* Phase 8: Heartbeat at operation start */
    omni_world_model_heartbeat("omni_world_m_omni_wm_delete_check", 0.0f);


    NIMCP_CHECK_THROW(wm, NIMCP_ERROR_INVALID_PARAM, "world model is NULL");
    NIMCP_CHECK_THROW(wm->checkpoint_store, NIMCP_ERROR_INVALID_PARAM, "checkpoint store is NULL");
    NIMCP_CHECK_THROW(checkpoint_id != 0, NIMCP_ERROR_INVALID_PARAM, "checkpoint_id is zero");

    /* Find and remove checkpoint */
    for (uint32_t i = 0; i < wm->checkpoint_store->count; i++) {
        if (wm->checkpoint_store->checkpoints[i].id == checkpoint_id) {
            nimcp_free(wm->checkpoint_store->checkpoints[i].data);

            /* Shift remaining checkpoints */
            for (uint32_t j = i; j < wm->checkpoint_store->count - 1; j++) {
                wm->checkpoint_store->checkpoints[j] = wm->checkpoint_store->checkpoints[j + 1];
            }
            wm->checkpoint_store->count--;

            return NIMCP_SUCCESS;
        }
    }

    return NIMCP_ERROR_NOT_FOUND;
}

uint32_t omni_wm_get_checkpoint_count(const omni_world_model_t* wm) {
    if (!wm || !wm->checkpoint_store) return 0;
    /* Phase 8: Heartbeat at operation start */
    omni_world_model_heartbeat("omni_world_m_omni_wm_get_checkpoi", 0.0f);


    return wm->checkpoint_store->count;
}

nimcp_error_t omni_wm_clear_checkpoints(omni_world_model_t* wm) {
    /* Phase 8: Heartbeat at operation start */
    omni_world_model_heartbeat("omni_world_m_omni_wm_clear_checkp", 0.0f);


    NIMCP_CHECK_THROW(wm, NIMCP_ERROR_INVALID_PARAM, "world model is NULL");

    if (wm->checkpoint_store) {
        for (uint32_t i = 0; i < wm->checkpoint_store->count; i++) {
            nimcp_free(wm->checkpoint_store->checkpoints[i].data);
            wm->checkpoint_store->checkpoints[i].data = NULL;
        }
        wm->checkpoint_store->count = 0;
    }

    return NIMCP_SUCCESS;
}
