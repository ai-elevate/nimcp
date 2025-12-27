/**
 * @file nimcp_free_energy.c
 * @brief Free Energy Principle Implementation
 * @version 1.0.0
 * @date 2025-12-12
 *
 * WHAT: Implementation of variational free energy minimization
 * WHY:  Unifying framework for perception, action, and learning
 * HOW:  Hierarchical predictive coding with precision-weighted errors
 */

#include "cognitive/free_energy/nimcp_free_energy.h"
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

/**
 * @brief Compute L2 norm of vector
 */
static float vector_norm(const float* vec, uint32_t dim) {
    if (!vec || dim == 0) return 0.0f;

    float sum = 0.0f;
    for (uint32_t i = 0; i < dim; i++) {
        sum += vec[i] * vec[i];
    }
    return sqrtf(sum);
}

/**
 * @brief Compute dot product
 */
static float vector_dot(const float* a, const float* b, uint32_t dim) {
    if (!a || !b || dim == 0) return 0.0f;

    float sum = 0.0f;
    for (uint32_t i = 0; i < dim; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

/**
 * @brief Safe log (avoid log(0))
 */
static inline float safe_log(float x) {
    if (x <= 0.0f) return -100.0f;
    return logf(x);
}

/**
 * @brief Softmax over array (in-place)
 */
static void softmax(float* values, uint32_t n, float temperature) {
    if (!values || n == 0) return;

    /* Find max for numerical stability */
    float max_val = values[0];
    for (uint32_t i = 1; i < n; i++) {
        if (values[i] > max_val) max_val = values[i];
    }

    /* Compute exp and sum */
    float sum = 0.0f;
    for (uint32_t i = 0; i < n; i++) {
        values[i] = expf((values[i] - max_val) / temperature);
        sum += values[i];
    }

    /* Normalize */
    if (sum > 0.0f) {
        for (uint32_t i = 0; i < n; i++) {
            values[i] /= sum;
        }
    }
}

/**
 * @brief Initialize belief structure
 */
static int init_belief(fep_belief_t* belief, uint32_t dim, float initial_precision) {
    if (!belief || dim == 0) return -1;

    belief->dim = dim;
    belief->mean = (float*)nimcp_calloc(dim, sizeof(float));
    belief->variance = (float*)nimcp_calloc(dim, sizeof(float));
    belief->precision = (float*)nimcp_calloc(dim, sizeof(float));

    if (!belief->mean || !belief->variance || !belief->precision) {
        /* Clean up partially allocated memory on failure */
        if (belief->mean) nimcp_free(belief->mean);
        if (belief->variance) nimcp_free(belief->variance);
        if (belief->precision) nimcp_free(belief->precision);
        belief->mean = NULL;
        belief->variance = NULL;
        belief->precision = NULL;
        belief->dim = 0;
        return -1;
    }

    /* Initialize with prior */
    for (uint32_t i = 0; i < dim; i++) {
        belief->mean[i] = 0.0f;
        belief->variance[i] = 1.0f / initial_precision;
        belief->precision[i] = initial_precision;
    }

    return 0;
}

/**
 * @brief Free belief structure
 */
static void free_belief(fep_belief_t* belief) {
    if (!belief) return;
    if (belief->mean) nimcp_free(belief->mean);
    if (belief->variance) nimcp_free(belief->variance);
    if (belief->precision) nimcp_free(belief->precision);
    belief->mean = NULL;
    belief->variance = NULL;
    belief->precision = NULL;
    belief->dim = 0;
}

/**
 * @brief Initialize prediction error structure
 */
static int init_prediction_error(fep_prediction_error_t* error, uint32_t dim) {
    if (!error || dim == 0) return -1;

    error->dim = dim;
    error->error = (float*)nimcp_calloc(dim, sizeof(float));
    error->weighted_error = (float*)nimcp_calloc(dim, sizeof(float));
    error->precision = (float*)nimcp_calloc(dim, sizeof(float));

    if (!error->error || !error->weighted_error || !error->precision) {
        /* Clean up partially allocated memory on failure */
        if (error->error) nimcp_free(error->error);
        if (error->weighted_error) nimcp_free(error->weighted_error);
        if (error->precision) nimcp_free(error->precision);
        error->error = NULL;
        error->weighted_error = NULL;
        error->precision = NULL;
        error->dim = 0;
        return -1;
    }

    for (uint32_t i = 0; i < dim; i++) {
        error->precision[i] = FEP_DEFAULT_PRECISION;
    }

    return 0;
}

/**
 * @brief Free prediction error structure
 */
static void free_prediction_error(fep_prediction_error_t* error) {
    if (!error) return;
    if (error->error) nimcp_free(error->error);
    if (error->weighted_error) nimcp_free(error->weighted_error);
    if (error->precision) nimcp_free(error->precision);
    error->error = NULL;
    error->weighted_error = NULL;
    error->precision = NULL;
    error->dim = 0;
}

/**
 * @brief Initialize hierarchy level
 */
static int init_level(
    fep_hierarchy_level_t* level,
    uint32_t level_id,
    uint32_t state_dim,
    uint32_t prediction_dim,
    float initial_precision
) {
    if (!level) return -1;

    memset(level, 0, sizeof(fep_hierarchy_level_t));
    level->level_id = level_id;

    /* Initialize beliefs */
    if (init_belief(&level->beliefs, state_dim, initial_precision) != 0) {
        return -1;
    }

    /* Initialize predictions */
    level->prediction_dim = prediction_dim;
    if (prediction_dim > 0) {
        level->predictions = (float*)nimcp_calloc(prediction_dim, sizeof(float));
        if (!level->predictions) {
            /* Clean up beliefs before returning */
            free_belief(&level->beliefs);
            return -1;
        }
    }

    /* Initialize prediction errors */
    if (init_prediction_error(&level->errors, prediction_dim > 0 ? prediction_dim : state_dim) != 0) {
        /* Clean up beliefs and predictions before returning */
        free_belief(&level->beliefs);
        if (level->predictions) nimcp_free(level->predictions);
        level->predictions = NULL;
        return -1;
    }

    /* Prior initialization */
    level->prior_mean = (float*)nimcp_calloc(state_dim, sizeof(float));
    level->prior_precision = (float*)nimcp_calloc(state_dim, sizeof(float));
    if (!level->prior_mean || !level->prior_precision) {
        /* Clean up all previously allocated resources */
        free_belief(&level->beliefs);
        free_prediction_error(&level->errors);
        if (level->predictions) nimcp_free(level->predictions);
        if (level->prior_mean) nimcp_free(level->prior_mean);
        if (level->prior_precision) nimcp_free(level->prior_precision);
        level->predictions = NULL;
        level->prior_mean = NULL;
        level->prior_precision = NULL;
        return -1;
    }

    for (uint32_t i = 0; i < state_dim; i++) {
        level->prior_precision[i] = initial_precision;
    }

    return 0;
}

/**
 * @brief Free hierarchy level
 */
static void free_level(fep_hierarchy_level_t* level) {
    if (!level) return;

    free_belief(&level->beliefs);
    free_prediction_error(&level->errors);

    if (level->predictions) nimcp_free(level->predictions);
    if (level->transition_matrix) nimcp_free(level->transition_matrix);
    if (level->likelihood_matrix) nimcp_free(level->likelihood_matrix);
    if (level->prior_mean) nimcp_free(level->prior_mean);
    if (level->prior_precision) nimcp_free(level->prior_precision);
}

/* ============================================================================
 * Lifecycle Implementation
 * ============================================================================ */

int fep_default_config(fep_config_t* config) {
    if (!config) return -1;

    config->num_levels = 2;  /* Default 2-level hierarchy */
    config->level_dims = NULL;  /* Will use defaults */

    config->belief_learning_rate = FEP_DEFAULT_BELIEF_LR;
    config->precision_learning_rate = FEP_DEFAULT_PRECISION_LR;
    config->action_learning_rate = FEP_DEFAULT_ACTION_LR;

    config->update_mode = FEP_UPDATE_PREDICTIVE_CODING;
    config->action_mode = FEP_ACTION_SOFTMAX;

    config->initial_precision = FEP_DEFAULT_PRECISION;
    config->learn_precision = true;

    config->enable_active_inference = true;
    config->planning_horizon = 3;
    config->action_temperature = 1.0f;

    config->max_iterations = 10;
    config->convergence_threshold = FEP_CONVERGENCE_THRESHOLD;

    return 0;
}

fep_system_t* fep_create(
    const fep_config_t* config,
    uint32_t observation_dim,
    uint32_t action_dim
) {
    if (observation_dim == 0) {
        NIMCP_LOGGING_ERROR("Observation dimension must be > 0");
        return NULL;
    }

    fep_system_t* fep = (fep_system_t*)nimcp_calloc(1, sizeof(fep_system_t));
    if (!fep) {
        NIMCP_LOGGING_ERROR("FEP allocation failed");
        return NULL;
    }

    /* Apply configuration */
    fep_config_t default_cfg;
    if (!config) {
        fep_default_config(&default_cfg);
        config = &default_cfg;
    }
    fep->config = *config;

    /* Ensure at least 1 level */
    if (fep->config.num_levels == 0) {
        fep->config.num_levels = 1;
    }
    if (fep->config.num_levels > FEP_MAX_HIERARCHY_LEVELS) {
        fep->config.num_levels = FEP_MAX_HIERARCHY_LEVELS;
    }

    /* Store dimensions */
    fep->observation_dim = observation_dim;
    fep->action_dim = action_dim;
    fep->num_actions = action_dim > 0 ? action_dim : 1;

    /* Allocate observations */
    fep->observations = (float*)nimcp_calloc(observation_dim, sizeof(float));
    if (!fep->observations) {
        fep_destroy(fep);
        return NULL;
    }

    /* Allocate hierarchy levels */
    fep->num_levels = fep->config.num_levels;
    fep->levels = (fep_hierarchy_level_t*)nimcp_calloc(
        fep->num_levels, sizeof(fep_hierarchy_level_t)
    );
    if (!fep->levels) {
        fep_destroy(fep);
        return NULL;
    }

    /* Initialize each level */
    for (uint32_t i = 0; i < fep->num_levels; i++) {
        uint32_t state_dim;
        uint32_t pred_dim;

        if (config->level_dims && config->level_dims[i] > 0) {
            state_dim = config->level_dims[i];
        } else {
            /* Default: each level halves dimension */
            state_dim = observation_dim >> i;
            if (state_dim < 4) state_dim = 4;
        }

        /* Prediction dimension is state dim of level below (or obs dim for level 0) */
        if (i == 0) {
            pred_dim = observation_dim;
        } else {
            pred_dim = fep->levels[i-1].beliefs.dim;
        }

        if (init_level(&fep->levels[i], i, state_dim, pred_dim,
                       fep->config.initial_precision) != 0) {
            NIMCP_LOGGING_ERROR("Failed to initialize level %u", i);
            fep_destroy(fep);
            return NULL;
        }
    }

    /* Allocate policies if active inference enabled */
    if (fep->config.enable_active_inference && action_dim > 0) {
        fep->policies = (fep_policy_t*)nimcp_calloc(
            FEP_MAX_POLICIES, sizeof(fep_policy_t)
        );
        if (!fep->policies) {
            fep_destroy(fep);
            return NULL;
        }

        /* Initialize simple one-step policies */
        fep->num_policies = action_dim < FEP_MAX_POLICIES ? action_dim : FEP_MAX_POLICIES;
        for (uint32_t i = 0; i < fep->num_policies; i++) {
            fep->policies[i].policy_id = i;
            fep->policies[i].action_dim = action_dim;
            fep->policies[i].num_actions = 1;
            fep->policies[i].actions = (float*)nimcp_calloc(action_dim, sizeof(float));
            if (fep->policies[i].actions) {
                /* One-hot encoding of actions */
                if (i < action_dim) {
                    fep->policies[i].actions[i] = 1.0f;
                }
            }
        }
    }

    /* Create mutex */
    fep->mutex = nimcp_platform_mutex_create();
    if (!fep->mutex) {
        fep_destroy(fep);
        return NULL;
    }

    NIMCP_LOGGING_INFO("FEP system created: %u levels, obs_dim=%u, act_dim=%u",
                      fep->num_levels, observation_dim, action_dim);
    return fep;
}

void fep_destroy(fep_system_t* fep) {
    if (!fep) return;

    /* Free levels */
    if (fep->levels) {
        for (uint32_t i = 0; i < fep->num_levels; i++) {
            free_level(&fep->levels[i]);
        }
        nimcp_free(fep->levels);
    }

    /* Free observations */
    if (fep->observations) nimcp_free(fep->observations);

    /* Free action space */
    if (fep->action_space) nimcp_free(fep->action_space);

    /* Free policies */
    if (fep->policies) {
        for (uint32_t i = 0; i < fep->num_policies; i++) {
            if (fep->policies[i].actions) {
                nimcp_free(fep->policies[i].actions);
            }
        }
        nimcp_free(fep->policies);
    }

    /* Free mutex */
    if (fep->mutex) {
        nimcp_platform_mutex_destroy(fep->mutex);
        nimcp_free(fep->mutex);
    }

    nimcp_free(fep);
    NIMCP_LOGGING_INFO("FEP system destroyed");
}

int fep_reset(fep_system_t* fep) {
    if (!fep) return -1;

    nimcp_platform_mutex_lock(fep->mutex);

    /* Reset beliefs to priors */
    for (uint32_t l = 0; l < fep->num_levels; l++) {
        fep_hierarchy_level_t* level = &fep->levels[l];
        for (uint32_t i = 0; i < level->beliefs.dim; i++) {
            level->beliefs.mean[i] = level->prior_mean[i];
            level->beliefs.precision[i] = fep->config.initial_precision;
            level->beliefs.variance[i] = 1.0f / fep->config.initial_precision;
        }

        /* Reset errors */
        for (uint32_t i = 0; i < level->errors.dim; i++) {
            level->errors.error[i] = 0.0f;
            level->errors.weighted_error[i] = 0.0f;
        }
        level->errors.magnitude = 0.0f;
        level->errors.weighted_magnitude = 0.0f;
    }

    /* Reset free energy */
    memset(&fep->free_energy, 0, sizeof(fep_free_energy_t));
    memset(&fep->expected_free_energy, 0, sizeof(fep_efe_t));

    /* Reset stats */
    memset(&fep->stats, 0, sizeof(fep_stats_t));

    nimcp_platform_mutex_unlock(fep->mutex);
    return 0;
}

/* ============================================================================
 * Observation Processing Implementation
 * ============================================================================ */

int fep_process_observation(
    fep_system_t* fep,
    const float* observation,
    uint32_t observation_dim
) {
    if (!fep || !observation) return -1;
    if (observation_dim != fep->observation_dim) return -1;

    nimcp_platform_mutex_lock(fep->mutex);

    /* Store observation */
    memcpy(fep->observations, observation, observation_dim * sizeof(float));

    /* Compute prediction errors through hierarchy */
    fep_propagate_hierarchy(fep);

    /* Update beliefs to minimize free energy */
    for (uint32_t iter = 0; iter < fep->config.max_iterations; iter++) {
        float prev_fe = fep->free_energy.total;

        fep_update_beliefs(fep);

        if (fep->config.learn_precision) {
            fep_update_precision(fep);
        }

        fep_free_energy_t fe;
        fep_compute_free_energy(fep, &fe);

        /* Check convergence */
        if (fabsf(fe.total - prev_fe) < fep->config.convergence_threshold) {
            break;
        }
    }

    fep->stats.total_updates++;
    nimcp_platform_mutex_unlock(fep->mutex);

    return 0;
}

uint32_t fep_compute_prediction(
    const fep_system_t* fep,
    float* prediction,
    uint32_t prediction_dim
) {
    if (!fep || !prediction) return 0;

    /* Use level 0 predictions (lowest level) */
    fep_hierarchy_level_t* level = &fep->levels[0];
    uint32_t dim = level->prediction_dim;
    if (dim > prediction_dim) dim = prediction_dim;

    /* Simple linear prediction: g(μ) = μ (identity for simplicity) */
    /* In full implementation, this would apply likelihood matrix */
    memcpy(prediction, level->beliefs.mean, dim * sizeof(float));

    return dim;
}

int fep_compute_prediction_error(
    const fep_system_t* fep,
    fep_prediction_error_t* error
) {
    if (!fep || !error) return -1;

    /* Get level 0 */
    const fep_hierarchy_level_t* level = &fep->levels[0];
    uint32_t dim = level->errors.dim;

    /* Copy error structure */
    error->dim = dim;
    error->magnitude = level->errors.magnitude;
    error->weighted_magnitude = level->errors.weighted_magnitude;

    return 0;
}

/* ============================================================================
 * Belief Update Implementation
 * ============================================================================ */

int fep_update_beliefs(fep_system_t* fep) {
    if (!fep) return -1;

    float lr = fep->config.belief_learning_rate;

    /* Update each level bottom-up */
    for (uint32_t l = 0; l < fep->num_levels; l++) {
        fep_hierarchy_level_t* level = &fep->levels[l];

        /* Gradient descent: μ' = μ - lr * ∂F/∂μ */
        /* ∂F/∂μ ≈ precision-weighted prediction error + prior deviation */
        for (uint32_t i = 0; i < level->beliefs.dim && i < level->errors.dim; i++) {
            /* Error-driven update */
            float grad = level->errors.weighted_error[i];

            /* Prior regularization */
            float prior_error = level->beliefs.mean[i] - level->prior_mean[i];
            grad += level->prior_precision[i] * prior_error * 0.1f;

            /* Apply update */
            level->beliefs.mean[i] -= lr * grad;
        }
    }

    fep->stats.belief_updates++;
    return 0;
}

int fep_update_precision(fep_system_t* fep) {
    if (!fep) return -1;

    float lr = fep->config.precision_learning_rate;

    /* Update precision based on prediction error variance */
    for (uint32_t l = 0; l < fep->num_levels; l++) {
        fep_hierarchy_level_t* level = &fep->levels[l];

        for (uint32_t i = 0; i < level->errors.dim; i++) {
            /* Precision update: Π' = Π + lr * (ε² - 1/Π) */
            float error_sq = level->errors.error[i] * level->errors.error[i];
            float current_var = 1.0f / level->errors.precision[i];

            /* Move precision toward inverse error variance */
            float target_precision = 1.0f / (error_sq + 0.01f);
            float delta = target_precision - level->errors.precision[i];

            level->errors.precision[i] += lr * delta;
            level->errors.precision[i] = clamp_f(
                level->errors.precision[i],
                FEP_MIN_PRECISION,
                FEP_MAX_PRECISION
            );
        }
    }

    return 0;
}

int fep_propagate_hierarchy(fep_system_t* fep) {
    if (!fep) return -1;

    /* Bottom level: compare with observations */
    fep_hierarchy_level_t* level0 = &fep->levels[0];
    uint32_t obs_dim = fep->observation_dim;
    uint32_t pred_dim = level0->prediction_dim;
    uint32_t min_dim = obs_dim < pred_dim ? obs_dim : pred_dim;

    /* Generate predictions from beliefs */
    for (uint32_t i = 0; i < pred_dim && i < level0->beliefs.dim; i++) {
        level0->predictions[i] = level0->beliefs.mean[i];
    }

    /* Compute prediction error: ε = o - g(μ) */
    for (uint32_t i = 0; i < min_dim; i++) {
        level0->errors.error[i] = fep->observations[i] - level0->predictions[i];
        level0->errors.weighted_error[i] =
            level0->errors.precision[i] * level0->errors.error[i];
    }
    level0->errors.magnitude = vector_norm(level0->errors.error, min_dim);
    level0->errors.weighted_magnitude = vector_norm(level0->errors.weighted_error, min_dim);

    /* Propagate through hierarchy */
    for (uint32_t l = 1; l < fep->num_levels; l++) {
        fep_hierarchy_level_t* current = &fep->levels[l];
        fep_hierarchy_level_t* lower = &fep->levels[l-1];

        /* Predictions: higher level predicts lower level state */
        uint32_t dim = current->prediction_dim;
        if (dim > lower->beliefs.dim) dim = lower->beliefs.dim;

        for (uint32_t i = 0; i < dim && i < current->beliefs.dim; i++) {
            current->predictions[i] = current->beliefs.mean[i];
        }

        /* Prediction error: lower level state - higher level prediction */
        for (uint32_t i = 0; i < dim; i++) {
            current->errors.error[i] = lower->beliefs.mean[i] - current->predictions[i];
            current->errors.weighted_error[i] =
                current->errors.precision[i] * current->errors.error[i];
        }
        current->errors.magnitude = vector_norm(current->errors.error, dim);
        current->errors.weighted_magnitude = vector_norm(current->errors.weighted_error, dim);
    }

    /* Update stats */
    fep->stats.avg_prediction_error =
        (fep->stats.avg_prediction_error * fep->stats.total_updates +
         level0->errors.magnitude) /
        (fep->stats.total_updates + 1);

    return 0;
}

/* ============================================================================
 * Free Energy Computation Implementation
 * ============================================================================ */

int fep_compute_free_energy(
    const fep_system_t* fep,
    fep_free_energy_t* fe
) {
    if (!fep || !fe) return -1;

    memset(fe, 0, sizeof(fep_free_energy_t));

    /* Sum over hierarchy levels */
    for (uint32_t l = 0; l < fep->num_levels; l++) {
        const fep_hierarchy_level_t* level = &fep->levels[l];

        /* Inaccuracy: -E[ln p(o|s)] ≈ 0.5 * Σ Π * ε² */
        for (uint32_t i = 0; i < level->errors.dim; i++) {
            float err_sq = level->errors.error[i] * level->errors.error[i];
            fe->inaccuracy += 0.5f * level->errors.precision[i] * err_sq;
        }

        /* Complexity: KL[q(s)||p(s)] ≈ 0.5 * Σ (μ-μ₀)²/σ₀² - ln(σ/σ₀) */
        for (uint32_t i = 0; i < level->beliefs.dim; i++) {
            float mean_diff = level->beliefs.mean[i] - level->prior_mean[i];
            fe->complexity += 0.5f * level->prior_precision[i] * mean_diff * mean_diff;

            /* Entropy term */
            fe->entropy += 0.5f * safe_log(2.0f * 3.14159f * level->beliefs.variance[i]);
        }
    }

    /* Total free energy */
    fe->total = fe->complexity + fe->inaccuracy;

    /* Energy term */
    fe->energy = fe->inaccuracy + fe->complexity;

    /* Surprise bound */
    fe->surprise = fe->total;

    /* Update system state */
    ((fep_system_t*)fep)->free_energy = *fe;

    /* Stats update */
    fep_stats_t* stats = &((fep_system_t*)fep)->stats;
    stats->avg_free_energy = (stats->avg_free_energy * stats->total_updates + fe->total) /
                             (stats->total_updates + 1);
    if (fe->total < stats->min_free_energy || stats->min_free_energy == 0.0f) {
        stats->min_free_energy = fe->total;
    }
    if (fe->surprise > stats->max_surprise) {
        stats->max_surprise = fe->surprise;
    }

    return 0;
}

float fep_compute_component(
    const fep_system_t* fep,
    fep_component_t component
) {
    if (!fep) return 0.0f;

    switch (component) {
        case FEP_COMPONENT_COMPLEXITY:
            return fep->free_energy.complexity;
        case FEP_COMPONENT_INACCURACY:
            return fep->free_energy.inaccuracy;
        case FEP_COMPONENT_ENERGY:
            return fep->free_energy.energy;
        case FEP_COMPONENT_ENTROPY:
            return fep->free_energy.entropy;
        default:
            return 0.0f;
    }
}

float fep_compute_surprise(const fep_system_t* fep) {
    if (!fep) return 0.0f;
    return fep->free_energy.surprise;
}

/* ============================================================================
 * Active Inference Implementation
 * ============================================================================ */

/**
 * @brief Internal policy evaluation (no mutex - caller must hold lock)
 */
static int fep_evaluate_policies_unlocked(fep_system_t* fep) {
    if (!fep || !fep->policies) return -1;

    /* Compute EFE for each policy */
    for (uint32_t i = 0; i < fep->num_policies; i++) {
        fep_efe_t efe;
        fep_compute_efe(fep, &fep->policies[i], &efe);
        fep->policies[i].expected_free_energy = efe.total;
    }

    /* Compute policy probabilities via softmax over -G(π) */
    float* neg_efe = (float*)nimcp_malloc(fep->num_policies * sizeof(float));
    if (neg_efe) {
        for (uint32_t i = 0; i < fep->num_policies; i++) {
            neg_efe[i] = -fep->policies[i].expected_free_energy;
        }

        softmax(neg_efe, fep->num_policies, fep->config.action_temperature);

        for (uint32_t i = 0; i < fep->num_policies; i++) {
            fep->policies[i].probability = neg_efe[i];
        }

        nimcp_free(neg_efe);
    }

    return 0;
}

int fep_compute_efe(
    const fep_system_t* fep,
    const fep_policy_t* policy,
    fep_efe_t* efe
) {
    if (!fep || !policy || !efe) return -1;

    memset(efe, 0, sizeof(fep_efe_t));

    /* Simplified EFE computation */
    /* G(π) = Risk + Ambiguity */

    /* Risk: Expected deviation from preferred states */
    /* For simplicity: higher action entropy = higher risk */
    float action_entropy = 0.0f;
    for (uint32_t i = 0; i < policy->action_dim && policy->actions; i++) {
        if (policy->actions[i] > 0.001f) {
            action_entropy -= policy->actions[i] * safe_log(policy->actions[i]);
        }
    }
    efe->risk = action_entropy;

    /* Ambiguity: Expected uncertainty about outcomes */
    /* Use current prediction error as proxy */
    efe->ambiguity = fep->levels[0].errors.magnitude * 0.5f;

    /* Intrinsic value: Information gain (epistemic) */
    efe->intrinsic_value = efe->ambiguity;  /* High ambiguity = more to learn */

    /* Extrinsic value: Goal achievement (pragmatic) */
    efe->extrinsic_value = -efe->risk;

    /* Total EFE */
    efe->total = efe->risk + efe->ambiguity;

    return 0;
}

int fep_evaluate_policies(fep_system_t* fep) {
    if (!fep || !fep->policies) return -1;

    nimcp_platform_mutex_lock(fep->mutex);
    int ret = fep_evaluate_policies_unlocked(fep);
    nimcp_platform_mutex_unlock(fep->mutex);

    return ret;
}

int fep_select_action(
    fep_system_t* fep,
    float* action,
    uint32_t action_dim
) {
    if (!fep || !action) return -1;
    if (action_dim < fep->action_dim) return -1;

    nimcp_platform_mutex_lock(fep->mutex);

    /* Evaluate policies first (use unlocked version since we hold mutex) */
    fep_evaluate_policies_unlocked(fep);

    /* Select based on mode */
    uint32_t selected = 0;

    switch (fep->config.action_mode) {
        case FEP_ACTION_GREEDY: {
            /* Select policy with minimum EFE */
            float min_efe = fep->policies[0].expected_free_energy;
            for (uint32_t i = 1; i < fep->num_policies; i++) {
                if (fep->policies[i].expected_free_energy < min_efe) {
                    min_efe = fep->policies[i].expected_free_energy;
                    selected = i;
                }
            }
            break;
        }

        case FEP_ACTION_SOFTMAX:
        case FEP_ACTION_THOMPSON:
        default: {
            /* Sample from policy distribution */
            float r = (float)rand() / (float)RAND_MAX;
            float cumsum = 0.0f;
            for (uint32_t i = 0; i < fep->num_policies; i++) {
                cumsum += fep->policies[i].probability;
                if (r <= cumsum) {
                    selected = i;
                    break;
                }
            }
            break;
        }
    }

    fep->selected_policy = selected;

    /* Copy action */
    if (fep->policies[selected].actions) {
        uint32_t copy_dim = fep->action_dim < action_dim ? fep->action_dim : action_dim;
        memcpy(action, fep->policies[selected].actions, copy_dim * sizeof(float));
    }

    fep->stats.action_selections++;
    nimcp_platform_mutex_unlock(fep->mutex);

    return selected;
}

int fep_set_preferences(
    fep_system_t* fep,
    const float* preferred,
    float precision,
    uint32_t dim
) {
    if (!fep || !preferred) return -1;

    /* Store preferences as prior for level 0 */
    uint32_t store_dim = dim < fep->levels[0].beliefs.dim ? dim : fep->levels[0].beliefs.dim;

    for (uint32_t i = 0; i < store_dim; i++) {
        fep->levels[0].prior_mean[i] = preferred[i];
        fep->levels[0].prior_precision[i] = precision;
    }

    return 0;
}

/* ============================================================================
 * Query Implementation
 * ============================================================================ */

int fep_get_beliefs(
    const fep_system_t* fep,
    uint32_t level,
    fep_belief_t* beliefs
) {
    if (!fep || !beliefs || level >= fep->num_levels) return -1;

    *beliefs = fep->levels[level].beliefs;
    return 0;
}

float fep_get_free_energy(const fep_system_t* fep) {
    if (!fep) return 0.0f;
    return fep->free_energy.total;
}

float fep_get_prediction_error(const fep_system_t* fep, uint32_t level) {
    if (!fep || level >= fep->num_levels) return 0.0f;
    return fep->levels[level].errors.magnitude;
}

int fep_get_selected_policy(const fep_system_t* fep, fep_policy_t* policy) {
    if (!fep || !policy || !fep->policies) return -1;
    if (fep->selected_policy >= fep->num_policies) return -1;

    *policy = fep->policies[fep->selected_policy];
    return 0;
}

int fep_get_stats(const fep_system_t* fep, fep_stats_t* stats) {
    if (!fep || !stats) return -1;
    *stats = fep->stats;
    return 0;
}

/* ============================================================================
 * String Conversion Implementation
 * ============================================================================ */

const char* fep_update_mode_to_string(fep_update_mode_t mode) {
    switch (mode) {
        case FEP_UPDATE_GRADIENT_DESCENT:    return "GRADIENT_DESCENT";
        case FEP_UPDATE_PREDICTIVE_CODING:   return "PREDICTIVE_CODING";
        case FEP_UPDATE_VARIATIONAL_MESSAGE: return "VARIATIONAL_MESSAGE";
        case FEP_UPDATE_KALMAN_FILTER:       return "KALMAN_FILTER";
        default:                              return "UNKNOWN";
    }
}

const char* fep_action_mode_to_string(fep_action_mode_t mode) {
    switch (mode) {
        case FEP_ACTION_SOFTMAX:  return "SOFTMAX";
        case FEP_ACTION_GREEDY:   return "GREEDY";
        case FEP_ACTION_THOMPSON: return "THOMPSON";
        default:                   return "UNKNOWN";
    }
}

const char* fep_component_to_string(fep_component_t component) {
    switch (component) {
        case FEP_COMPONENT_COMPLEXITY:  return "COMPLEXITY";
        case FEP_COMPONENT_INACCURACY:  return "INACCURACY";
        case FEP_COMPONENT_ENERGY:      return "ENERGY";
        case FEP_COMPONENT_ENTROPY:     return "ENTROPY";
        default:                         return "UNKNOWN";
    }
}
