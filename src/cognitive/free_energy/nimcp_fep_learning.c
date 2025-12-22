/**
 * @file nimcp_fep_learning.c
 * @brief Learnable Generative Models for Free Energy Principle
 * @version 1.0.0
 * @date 2025-12-12
 *
 * WHAT: Implementation of gradient-based learning for FEP generative models
 * WHY:  Enables adaptive generative models through experience
 * HOW:  Gradient descent with L2 regularization on transition/likelihood matrices
 */

#include "cognitive/free_energy/nimcp_fep_learning.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>

/* ============================================================================
 * Helper Functions
 * ============================================================================ */

static inline float clamp_f(float value, float min, float max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
}

static float compute_loss(const float* target, const float* prediction, size_t dim) {
    if (!target || !prediction || dim == 0) return 0.0f;

    float loss = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        float diff = target[i] - prediction[i];
        loss += diff * diff;
    }
    return 0.5f * loss;
}

static float compute_gradient_norm(const nimcp_tensor_t* gradient) {
    if (!gradient) return 0.0f;
    return nimcp_tensor_norm_p(gradient, 2.0);
}

static void record_loss(fep_learning_stats_t* stats, float loss) {
    if (!stats || !stats->loss_history) return;

    if (stats->history_count < stats->history_capacity) {
        stats->loss_history[stats->history_count++] = loss;
    } else {
        /* Shift history and add new value */
        memmove(stats->loss_history, stats->loss_history + 1,
                (stats->history_capacity - 1) * sizeof(float));
        stats->loss_history[stats->history_capacity - 1] = loss;
    }

    stats->current_loss = loss;
    if (loss < stats->min_loss || stats->min_loss == 0.0f) {
        stats->min_loss = loss;
    }
}

/* ============================================================================
 * Lifecycle Implementation
 * ============================================================================ */

int fep_learning_default_config(fep_learning_config_t* config) {
    if (!config) return -1;

    config->learning_rate = FEP_LEARNING_DEFAULT_LR;
    config->learning_rate_decay = 0.99f;
    config->min_learning_rate = FEP_LEARNING_MIN_LR;

    config->l2_regularization = FEP_LEARNING_DEFAULT_REG;

    config->optimizer = FEP_OPTIMIZER_MOMENTUM;
    config->momentum = FEP_LEARNING_DEFAULT_MOMENTUM;
    config->beta1 = 0.9f;
    config->beta2 = 0.999f;
    config->epsilon = 1e-8f;

    config->batch_size = FEP_LEARNING_DEFAULT_BATCH_SIZE;
    config->use_batch_learning = false;

    config->convergence_threshold = FEP_LEARNING_CONVERGENCE_LOSS;
    config->gradient_threshold = FEP_LEARNING_CONVERGENCE_GRAD;
    config->convergence_window = 10;

    config->track_statistics = true;
    config->history_size = 100;

    return 0;
}

fep_transition_learner_t* fep_transition_learner_create(
    const fep_learning_config_t* config,
    uint32_t state_dim
) {
    if (state_dim == 0) {
        NIMCP_LOGGING_ERROR("State dimension must be > 0");
        return NULL;
    }

    fep_transition_learner_t* learner =
        (fep_transition_learner_t*)nimcp_calloc(1, sizeof(fep_transition_learner_t));
    if (!learner) return NULL;

    /* Apply configuration */
    fep_learning_config_t default_cfg;
    if (!config) {
        fep_learning_default_config(&default_cfg);
        config = &default_cfg;
    }
    learner->config = *config;
    learner->state_dim = state_dim;

    /* Create transition matrix (state_dim x state_dim) */
    uint32_t dims[2] = {state_dim, state_dim};
    learner->matrix = nimcp_tensor_create(dims, 2, NIMCP_DTYPE_F32);
    learner->gradient = nimcp_tensor_create(dims, 2, NIMCP_DTYPE_F32);
    learner->batch_gradient = nimcp_tensor_create(dims, 2, NIMCP_DTYPE_F32);

    if (!learner->matrix || !learner->gradient || !learner->batch_gradient) {
        fep_transition_learner_destroy(learner);
        return NULL;
    }

    /* Initialize matrix to small random values (identity-ish) */
    for (uint32_t i = 0; i < state_dim; i++) {
        for (uint32_t j = 0; j < state_dim; j++) {
            uint32_t idx[2] = {i, j};
            float val = (i == j) ? 0.8f : 0.2f / (float)(state_dim - 1);
            nimcp_tensor_set(learner->matrix, idx, val);
        }
    }

    /* Create momentum buffer if needed */
    if (config->optimizer == FEP_OPTIMIZER_MOMENTUM ||
        config->optimizer == FEP_OPTIMIZER_ADAM) {
        learner->momentum = nimcp_tensor_create(dims, 2, NIMCP_DTYPE_F32);
        if (!learner->momentum) {
            fep_transition_learner_destroy(learner);
            return NULL;
        }
    }

    /* Create velocity buffer for Adam */
    if (config->optimizer == FEP_OPTIMIZER_ADAM) {
        learner->velocity = nimcp_tensor_create(dims, 2, NIMCP_DTYPE_F32);
        if (!learner->velocity) {
            fep_transition_learner_destroy(learner);
            return NULL;
        }
    }

    /* Initialize statistics */
    if (config->track_statistics) {
        learner->stats.loss_history = (float*)nimcp_calloc(
            config->history_size, sizeof(float));
        learner->stats.history_capacity = config->history_size;
    }
    learner->stats.state = FEP_LEARNING_IDLE;

    /* Create mutex */
    learner->mutex = nimcp_platform_mutex_create();
    if (!learner->mutex) {
        fep_transition_learner_destroy(learner);
        return NULL;
    }

    NIMCP_LOGGING_INFO("Transition learner created: state_dim=%u", state_dim);
    return learner;
}

void fep_transition_learner_destroy(fep_transition_learner_t* learner) {
    if (!learner) return;

    if (learner->bio_async_enabled) {
        fep_transition_learner_disconnect_bio_async(learner);
    }

    if (learner->matrix) nimcp_tensor_destroy(learner->matrix);
    if (learner->gradient) nimcp_tensor_destroy(learner->gradient);
    if (learner->momentum) nimcp_tensor_destroy(learner->momentum);
    if (learner->velocity) nimcp_tensor_destroy(learner->velocity);
    if (learner->batch_gradient) nimcp_tensor_destroy(learner->batch_gradient);

    if (learner->stats.loss_history) nimcp_free(learner->stats.loss_history);

    if (learner->mutex) {
        nimcp_platform_mutex_destroy(learner->mutex);
        nimcp_free(learner->mutex);
    }

    nimcp_free(learner);
    NIMCP_LOGGING_INFO("Transition learner destroyed");
}

fep_likelihood_learner_t* fep_likelihood_learner_create(
    const fep_learning_config_t* config,
    uint32_t observation_dim,
    uint32_t state_dim
) {
    if (observation_dim == 0 || state_dim == 0) {
        NIMCP_LOGGING_ERROR("Dimensions must be > 0");
        return NULL;
    }

    fep_likelihood_learner_t* learner =
        (fep_likelihood_learner_t*)nimcp_calloc(1, sizeof(fep_likelihood_learner_t));
    if (!learner) return NULL;

    /* Apply configuration */
    fep_learning_config_t default_cfg;
    if (!config) {
        fep_learning_default_config(&default_cfg);
        config = &default_cfg;
    }
    learner->config = *config;
    learner->observation_dim = observation_dim;
    learner->state_dim = state_dim;

    /* Create likelihood matrix (obs_dim x state_dim) */
    uint32_t dims[2] = {observation_dim, state_dim};
    learner->matrix = nimcp_tensor_create(dims, 2, NIMCP_DTYPE_F32);
    learner->gradient = nimcp_tensor_create(dims, 2, NIMCP_DTYPE_F32);
    learner->batch_gradient = nimcp_tensor_create(dims, 2, NIMCP_DTYPE_F32);

    if (!learner->matrix || !learner->gradient || !learner->batch_gradient) {
        fep_likelihood_learner_destroy(learner);
        return NULL;
    }

    /* Initialize matrix to small random values */
    for (uint32_t i = 0; i < observation_dim; i++) {
        for (uint32_t j = 0; j < state_dim; j++) {
            uint32_t idx[2] = {i, j};
            float val = 1.0f / (float)state_dim;
            nimcp_tensor_set(learner->matrix, idx, val);
        }
    }

    /* Create optimizer buffers */
    if (config->optimizer == FEP_OPTIMIZER_MOMENTUM ||
        config->optimizer == FEP_OPTIMIZER_ADAM) {
        learner->momentum = nimcp_tensor_create(dims, 2, NIMCP_DTYPE_F32);
        if (!learner->momentum) {
            fep_likelihood_learner_destroy(learner);
            return NULL;
        }
    }

    if (config->optimizer == FEP_OPTIMIZER_ADAM) {
        learner->velocity = nimcp_tensor_create(dims, 2, NIMCP_DTYPE_F32);
        if (!learner->velocity) {
            fep_likelihood_learner_destroy(learner);
            return NULL;
        }
    }

    /* Initialize statistics */
    if (config->track_statistics) {
        learner->stats.loss_history = (float*)nimcp_calloc(
            config->history_size, sizeof(float));
        learner->stats.history_capacity = config->history_size;
    }
    learner->stats.state = FEP_LEARNING_IDLE;

    /* Create mutex */
    learner->mutex = nimcp_platform_mutex_create();
    if (!learner->mutex) {
        fep_likelihood_learner_destroy(learner);
        return NULL;
    }

    NIMCP_LOGGING_INFO("Likelihood learner created: obs_dim=%u, state_dim=%u",
                      observation_dim, state_dim);
    return learner;
}

void fep_likelihood_learner_destroy(fep_likelihood_learner_t* learner) {
    if (!learner) return;

    if (learner->bio_async_enabled) {
        fep_likelihood_learner_disconnect_bio_async(learner);
    }

    if (learner->matrix) nimcp_tensor_destroy(learner->matrix);
    if (learner->gradient) nimcp_tensor_destroy(learner->gradient);
    if (learner->momentum) nimcp_tensor_destroy(learner->momentum);
    if (learner->velocity) nimcp_tensor_destroy(learner->velocity);
    if (learner->batch_gradient) nimcp_tensor_destroy(learner->batch_gradient);

    if (learner->stats.loss_history) nimcp_free(learner->stats.loss_history);

    if (learner->mutex) {
        nimcp_platform_mutex_destroy(learner->mutex);
        nimcp_free(learner->mutex);
    }

    nimcp_free(learner);
    NIMCP_LOGGING_INFO("Likelihood learner destroyed");
}

/* ============================================================================
 * Learning Implementation
 * ============================================================================ */

int fep_learn_transition(
    fep_transition_learner_t* learner,
    fep_system_t* sys,
    const float* state_t,
    const float* state_t1,
    size_t dim
) {
    if (!learner || !sys || !state_t || !state_t1) return -1;
    if (dim != learner->state_dim) return -1;

    nimcp_platform_mutex_lock(learner->mutex);
    learner->stats.state = FEP_LEARNING_ACTIVE;

    /* Compute prediction: A * s_t */
    float* prediction = (float*)nimcp_calloc(dim, sizeof(float));
    if (!prediction) {
        nimcp_platform_mutex_unlock(learner->mutex);
        return -1;
    }

    for (size_t i = 0; i < dim; i++) {
        prediction[i] = 0.0f;
        for (size_t j = 0; j < dim; j++) {
            uint32_t idx[2] = {i, j};
            float a_ij = nimcp_tensor_get(learner->matrix, idx);
            prediction[i] += a_ij * state_t[j];
        }
    }

    /* Compute loss */
    float loss = compute_loss(state_t1, prediction, dim);

    /* Compute gradient: -error ⊗ state^T + λA */
    float lr = learner->config.learning_rate;
    float reg = learner->config.l2_regularization;

    for (size_t i = 0; i < dim; i++) {
        float error = state_t1[i] - prediction[i];
        for (size_t j = 0; j < dim; j++) {
            uint32_t idx[2] = {i, j};
            float current = nimcp_tensor_get(learner->matrix, idx);
            float grad = -error * state_t[j] + reg * current;

            /* Apply momentum if configured */
            if (learner->config.optimizer == FEP_OPTIMIZER_MOMENTUM && learner->momentum) {
                float m = nimcp_tensor_get(learner->momentum, idx);
                m = learner->config.momentum * m + grad;
                nimcp_tensor_set(learner->momentum, idx, m);
                grad = m;
            }

            /* Update matrix */
            float new_val = current - lr * grad;
            nimcp_tensor_set(learner->matrix, idx, new_val);
        }
    }

    nimcp_free(prediction);

    /* Update statistics */
    learner->stats.total_updates++;
    learner->stats.online_updates++;
    record_loss(&learner->stats, loss);
    learner->stats.current_grad_norm = compute_gradient_norm(learner->gradient);

    /* Check convergence */
    if (loss < learner->config.convergence_threshold) {
        learner->stats.convergence_count++;
        if (learner->stats.convergence_count >= learner->config.convergence_window) {
            learner->stats.state = FEP_LEARNING_CONVERGED;
        }
    } else {
        learner->stats.convergence_count = 0;
    }

    nimcp_platform_mutex_unlock(learner->mutex);
    return 0;
}

int fep_learn_transition_batch(
    fep_transition_learner_t* learner,
    fep_system_t* sys,
    const float* states,
    size_t n_transitions,
    size_t dim
) {
    if (!learner || !sys || !states || n_transitions == 0) return -1;
    if (dim != learner->state_dim) return -1;

    nimcp_platform_mutex_lock(learner->mutex);
    learner->stats.state = FEP_LEARNING_ACTIVE;

    /* Zero batch gradient */
    size_t dims[2] = {dim, dim};
    nimcp_tensor_t* batch_grad = learner->batch_gradient;
    for (size_t i = 0; i < dim * dim; i++) {
        uint32_t idx[2] = {i / dim, i % dim};
        nimcp_tensor_set(batch_grad, idx, 0.0f);
    }

    float total_loss = 0.0f;
    float* prediction = (float*)nimcp_calloc(dim, sizeof(float));
    if (!prediction) {
        nimcp_platform_mutex_unlock(learner->mutex);
        return -1;
    }

    /* Accumulate gradients over batch */
    for (size_t t = 0; t < n_transitions; t++) {
        const float* state_t = states + t * dim;
        const float* state_t1 = states + (t + 1) * dim;

        /* Compute prediction */
        for (size_t i = 0; i < dim; i++) {
            prediction[i] = 0.0f;
            for (size_t j = 0; j < dim; j++) {
                uint32_t idx[2] = {i, j};
                float a_ij = nimcp_tensor_get(learner->matrix, idx);
                prediction[i] += a_ij * state_t[j];
            }
        }

        total_loss += compute_loss(state_t1, prediction, dim);

        /* Accumulate gradient */
        float reg = learner->config.l2_regularization;
        for (size_t i = 0; i < dim; i++) {
            float error = state_t1[i] - prediction[i];
            for (size_t j = 0; j < dim; j++) {
                uint32_t idx[2] = {i, j};
                float current = nimcp_tensor_get(learner->matrix, idx);
                float grad = -error * state_t[j] + reg * current;
                float prev_grad = nimcp_tensor_get(batch_grad, idx);
                nimcp_tensor_set(batch_grad, idx, prev_grad + grad);
            }
        }
    }

    nimcp_free(prediction);

    /* Average and apply gradient */
    float lr = learner->config.learning_rate;
    float n_inv = 1.0f / (float)n_transitions;

    for (size_t i = 0; i < dim; i++) {
        for (size_t j = 0; j < dim; j++) {
            uint32_t idx[2] = {i, j};
            float grad = nimcp_tensor_get(batch_grad, idx) * n_inv;
            float current = nimcp_tensor_get(learner->matrix, idx);
            nimcp_tensor_set(learner->matrix, idx, current - lr * grad);
        }
    }

    /* Update statistics */
    learner->stats.total_updates++;
    learner->stats.batch_updates++;
    record_loss(&learner->stats, total_loss / (float)n_transitions);

    nimcp_platform_mutex_unlock(learner->mutex);
    return 0;
}

int fep_learn_likelihood(
    fep_likelihood_learner_t* learner,
    fep_system_t* sys,
    const float* observation,
    const float* state,
    size_t obs_dim,
    size_t state_dim
) {
    if (!learner || !sys || !observation || !state) return -1;
    if (obs_dim != learner->observation_dim || state_dim != learner->state_dim) return -1;

    nimcp_platform_mutex_lock(learner->mutex);
    learner->stats.state = FEP_LEARNING_ACTIVE;

    /* Compute prediction: B * s */
    float* prediction = (float*)nimcp_calloc(obs_dim, sizeof(float));
    if (!prediction) {
        nimcp_platform_mutex_unlock(learner->mutex);
        return -1;
    }

    for (size_t i = 0; i < obs_dim; i++) {
        prediction[i] = 0.0f;
        for (size_t j = 0; j < state_dim; j++) {
            uint32_t idx[2] = {i, j};
            float b_ij = nimcp_tensor_get(learner->matrix, idx);
            prediction[i] += b_ij * state[j];
        }
    }

    /* Compute loss */
    float loss = compute_loss(observation, prediction, obs_dim);

    /* Compute gradient and update */
    float lr = learner->config.learning_rate;
    float reg = learner->config.l2_regularization;

    for (size_t i = 0; i < obs_dim; i++) {
        float error = observation[i] - prediction[i];
        for (size_t j = 0; j < state_dim; j++) {
            uint32_t idx[2] = {i, j};
            float current = nimcp_tensor_get(learner->matrix, idx);
            float grad = -error * state[j] + reg * current;

            if (learner->config.optimizer == FEP_OPTIMIZER_MOMENTUM && learner->momentum) {
                float m = nimcp_tensor_get(learner->momentum, idx);
                m = learner->config.momentum * m + grad;
                nimcp_tensor_set(learner->momentum, idx, m);
                grad = m;
            }

            nimcp_tensor_set(learner->matrix, idx, current - lr * grad);
        }
    }

    nimcp_free(prediction);

    /* Update statistics */
    learner->stats.total_updates++;
    learner->stats.online_updates++;
    record_loss(&learner->stats, loss);

    nimcp_platform_mutex_unlock(learner->mutex);
    return 0;
}

int fep_learn_likelihood_batch(
    fep_likelihood_learner_t* learner,
    fep_system_t* sys,
    const float* observations,
    const float* states,
    size_t n_pairs,
    size_t obs_dim,
    size_t state_dim
) {
    if (!learner || !observations || !states || n_pairs == 0) return -1;
    if (obs_dim != learner->observation_dim || state_dim != learner->state_dim) return -1;

    nimcp_platform_mutex_lock(learner->mutex);
    learner->stats.state = FEP_LEARNING_ACTIVE;

    /* Zero batch gradient */
    nimcp_tensor_t* batch_grad = learner->batch_gradient;
    for (size_t i = 0; i < obs_dim * state_dim; i++) {
        uint32_t idx[2] = {i / state_dim, i % state_dim};
        nimcp_tensor_set(batch_grad, idx, 0.0f);
    }

    float total_loss = 0.0f;
    float* prediction = (float*)nimcp_calloc(obs_dim, sizeof(float));
    if (!prediction) {
        nimcp_platform_mutex_unlock(learner->mutex);
        return -1;
    }

    /* Accumulate gradients */
    for (size_t p = 0; p < n_pairs; p++) {
        const float* obs = observations + p * obs_dim;
        const float* state = states + p * state_dim;

        for (size_t i = 0; i < obs_dim; i++) {
            prediction[i] = 0.0f;
            for (size_t j = 0; j < state_dim; j++) {
                uint32_t idx[2] = {i, j};
                prediction[i] += nimcp_tensor_get(learner->matrix, idx) * state[j];
            }
        }

        total_loss += compute_loss(obs, prediction, obs_dim);

        float reg = learner->config.l2_regularization;
        for (size_t i = 0; i < obs_dim; i++) {
            float error = obs[i] - prediction[i];
            for (size_t j = 0; j < state_dim; j++) {
                uint32_t idx[2] = {i, j};
                float current = nimcp_tensor_get(learner->matrix, idx);
                float grad = -error * state[j] + reg * current;
                float prev = nimcp_tensor_get(batch_grad, idx);
                nimcp_tensor_set(batch_grad, idx, prev + grad);
            }
        }
    }

    nimcp_free(prediction);

    /* Average and apply */
    float lr = learner->config.learning_rate;
    float n_inv = 1.0f / (float)n_pairs;

    for (size_t i = 0; i < obs_dim; i++) {
        for (size_t j = 0; j < state_dim; j++) {
            uint32_t idx[2] = {i, j};
            float grad = nimcp_tensor_get(batch_grad, idx) * n_inv;
            float current = nimcp_tensor_get(learner->matrix, idx);
            nimcp_tensor_set(learner->matrix, idx, current - lr * grad);
        }
    }

    learner->stats.total_updates++;
    learner->stats.batch_updates++;
    record_loss(&learner->stats, total_loss / (float)n_pairs);

    nimcp_platform_mutex_unlock(learner->mutex);
    return 0;
}

/* ============================================================================
 * Matrix Access Implementation
 * ============================================================================ */

int fep_get_learned_transition(
    const fep_transition_learner_t* learner,
    float* matrix,
    size_t dim
) {
    if (!learner || !matrix) return -1;
    if (dim != learner->state_dim) return -1;

    for (size_t i = 0; i < dim; i++) {
        for (size_t j = 0; j < dim; j++) {
            uint32_t idx[2] = {i, j};
            matrix[i * dim + j] = nimcp_tensor_get(learner->matrix, idx);
        }
    }
    return 0;
}

int fep_get_learned_likelihood(
    const fep_likelihood_learner_t* learner,
    float* matrix,
    size_t obs_dim,
    size_t state_dim
) {
    if (!learner || !matrix) return -1;
    if (obs_dim != learner->observation_dim || state_dim != learner->state_dim) return -1;

    for (size_t i = 0; i < obs_dim; i++) {
        for (size_t j = 0; j < state_dim; j++) {
            uint32_t idx[2] = {i, j};
            matrix[i * state_dim + j] = nimcp_tensor_get(learner->matrix, idx);
        }
    }
    return 0;
}

/* ============================================================================
 * Integration Implementation
 * ============================================================================ */

int fep_apply_learned_transition(
    fep_transition_learner_t* learner,
    fep_system_t* sys
) {
    if (!learner || !sys) return -1;

    /* Apply to FEP level 0 transition matrix if available */
    if (sys->levels && sys->num_levels > 0) {
        fep_hierarchy_level_t* level = &sys->levels[0];
        size_t dim = learner->state_dim;

        if (!level->transition_matrix) {
            level->transition_matrix = (float*)nimcp_calloc(dim * dim, sizeof(float));
            if (!level->transition_matrix) return -1;
        }

        return fep_get_learned_transition(learner, level->transition_matrix, dim);
    }
    return 0;
}

int fep_apply_learned_likelihood(
    fep_likelihood_learner_t* learner,
    fep_system_t* sys
) {
    if (!learner || !sys) return -1;

    /* Apply to FEP level 0 likelihood matrix if available */
    if (sys->levels && sys->num_levels > 0) {
        fep_hierarchy_level_t* level = &sys->levels[0];
        size_t obs_dim = learner->observation_dim;
        size_t state_dim = learner->state_dim;

        if (!level->likelihood_matrix) {
            level->likelihood_matrix = (float*)nimcp_calloc(obs_dim * state_dim, sizeof(float));
            if (!level->likelihood_matrix) return -1;
        }

        return fep_get_learned_likelihood(learner, level->likelihood_matrix, obs_dim, state_dim);
    }
    return 0;
}

/* ============================================================================
 * Statistics Implementation
 * ============================================================================ */

int fep_transition_learning_get_stats(
    const fep_transition_learner_t* learner,
    fep_learning_stats_t* stats
) {
    if (!learner || !stats) return -1;
    *stats = learner->stats;
    return 0;
}

int fep_likelihood_learning_get_stats(
    const fep_likelihood_learner_t* learner,
    fep_learning_stats_t* stats
) {
    if (!learner || !stats) return -1;
    *stats = learner->stats;
    return 0;
}

int fep_learning_reset_stats(void* learner) {
    if (!learner) return -1;

    /* Cast to transition learner to access stats (same offset in both structs) */
    fep_transition_learner_t* t_learner = (fep_transition_learner_t*)learner;

    t_learner->stats.total_updates = 0;
    t_learner->stats.online_updates = 0;
    t_learner->stats.batch_updates = 0;
    t_learner->stats.current_loss = 0.0f;
    t_learner->stats.min_loss = 0.0f;
    t_learner->stats.avg_loss = 0.0f;
    t_learner->stats.history_count = 0;
    t_learner->stats.current_grad_norm = 0.0f;
    t_learner->stats.max_grad_norm = 0.0f;
    t_learner->stats.avg_grad_norm = 0.0f;
    t_learner->stats.state = FEP_LEARNING_IDLE;
    t_learner->stats.convergence_count = 0;
    t_learner->stats.divergence_count = 0;

    return 0;
}

/* ============================================================================
 * Bio-async Implementation
 * ============================================================================ */

int fep_transition_learner_connect_bio_async(fep_transition_learner_t* learner) {
    if (!learner) return -1;
    if (learner->bio_async_enabled) return 0;

    bio_module_info_t info = {
        .module_id = BIO_MODULE_FEP_LEARNING_TRANSITION,
        .module_name = "fep_transition_learner",
        .inbox_capacity = FEP_LEARNING_BIO_INBOX_SIZE,
        .user_data = learner
    };

    learner->bio_ctx = bio_router_register_module(&info);
    if (learner->bio_ctx) {
        learner->bio_async_enabled = true;
        NIMCP_LOGGING_INFO("Transition learner connected to bio-async");
    } else {
        NIMCP_LOGGING_WARN("Bio-async router not available, skipping registration");
    }
    return 0;
}

int fep_transition_learner_disconnect_bio_async(fep_transition_learner_t* learner) {
    if (!learner) return -1;
    if (!learner->bio_async_enabled) return 0;

    if (learner->bio_ctx) {
        bio_router_unregister_module(learner->bio_ctx);
        learner->bio_ctx = NULL;
    }
    learner->bio_async_enabled = false;
    return 0;
}

int fep_likelihood_learner_connect_bio_async(fep_likelihood_learner_t* learner) {
    if (!learner) return -1;
    if (learner->bio_async_enabled) return 0;

    bio_module_info_t info = {
        .module_id = BIO_MODULE_FEP_LEARNING_LIKELIHOOD,
        .module_name = "fep_likelihood_learner",
        .inbox_capacity = FEP_LEARNING_BIO_INBOX_SIZE,
        .user_data = learner
    };

    learner->bio_ctx = bio_router_register_module(&info);
    if (learner->bio_ctx) {
        learner->bio_async_enabled = true;
        NIMCP_LOGGING_INFO("Likelihood learner connected to bio-async");
    } else {
        NIMCP_LOGGING_WARN("Bio-async router not available, skipping registration");
    }
    return 0;
}

int fep_likelihood_learner_disconnect_bio_async(fep_likelihood_learner_t* learner) {
    if (!learner) return -1;
    if (!learner->bio_async_enabled) return 0;

    if (learner->bio_ctx) {
        bio_router_unregister_module(learner->bio_ctx);
        learner->bio_ctx = NULL;
    }
    learner->bio_async_enabled = false;
    return 0;
}

/* ============================================================================
 * String Conversion Implementation
 * ============================================================================ */

const char* fep_optimizer_type_to_string(fep_optimizer_type_t type) {
    switch (type) {
        case FEP_OPTIMIZER_SGD:       return "SGD";
        case FEP_OPTIMIZER_MOMENTUM:  return "MOMENTUM";
        case FEP_OPTIMIZER_ADAM:      return "ADAM";
        case FEP_OPTIMIZER_RMSPROP:   return "RMSPROP";
        default:                       return "UNKNOWN";
    }
}

const char* fep_learning_state_to_string(fep_learning_state_t state) {
    switch (state) {
        case FEP_LEARNING_IDLE:      return "IDLE";
        case FEP_LEARNING_ACTIVE:    return "ACTIVE";
        case FEP_LEARNING_CONVERGED: return "CONVERGED";
        case FEP_LEARNING_DIVERGED:  return "DIVERGED";
        default:                      return "UNKNOWN";
    }
}
