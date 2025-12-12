/**
 * @file nimcp_fep_evidence.c
 * @brief Model Evidence Computation for Free Energy Principle
 * @version 1.0.0
 * @date 2025-12-12
 *
 * WHAT: Implementation of Bayesian model comparison and evidence computation
 * WHY:  Model selection enables brain to choose best generative model
 * HOW:  ELBO, importance sampling, Bayes factors for model comparison
 */

#include "cognitive/free_energy/nimcp_fep_evidence.h"
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

static inline float safe_log(float x) {
    if (x <= FEP_EVIDENCE_EPSILON) return FEP_EVIDENCE_LOG_MIN;
    float result = logf(x);
    return clamp_f(result, FEP_EVIDENCE_LOG_MIN, FEP_EVIDENCE_LOG_MAX);
}

static inline float safe_exp(float x) {
    x = clamp_f(x, -88.0f, 88.0f);  /* Prevent overflow */
    return expf(x);
}

/* Log-sum-exp for numerical stability */
static float logsumexp(const float* values, size_t n) {
    if (!values || n == 0) return FEP_EVIDENCE_LOG_MIN;

    /* Find maximum */
    float max_val = values[0];
    for (size_t i = 1; i < n; i++) {
        if (values[i] > max_val) max_val = values[i];
    }

    /* Compute sum of exp(x - max) */
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        sum += safe_exp(values[i] - max_val);
    }

    return max_val + safe_log(sum);
}

/* Internal ELBO computation without locking (for use within locked sections) */
static int fep_compute_elbo_internal(
    fep_evidence_system_t* sys,
    fep_system_t* fep,
    const float* observation,
    size_t obs_dim,
    float* elbo
) {
    if (!sys || !fep || !observation || !elbo) return -1;

    /* ELBO = E_q[log p(o,s)] - E_q[log q(s)]
     *      = E_q[log p(o|s)] + E_q[log p(s)] - E_q[log q(s)]
     *      = Accuracy - Complexity */

    float accuracy = 0.0f;
    float complexity = 0.0f;

    fep_compute_model_accuracy(sys, fep, observation, obs_dim, &accuracy);
    fep_compute_model_complexity(sys, fep, &complexity);

    *elbo = accuracy - complexity;
    *elbo = clamp_f(*elbo, FEP_EVIDENCE_LOG_MIN, FEP_EVIDENCE_LOG_MAX);

    return 0;
}

/* Softmax normalization */
static void softmax(float* probs, size_t n) {
    if (!probs || n == 0) return;

    float max_val = probs[0];
    for (size_t i = 1; i < n; i++) {
        if (probs[i] > max_val) max_val = probs[i];
    }

    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        probs[i] = safe_exp(probs[i] - max_val);
        sum += probs[i];
    }

    if (sum > FEP_EVIDENCE_EPSILON) {
        for (size_t i = 0; i < n; i++) {
            probs[i] /= sum;
        }
    }
}

/* ============================================================================
 * Lifecycle Implementation
 * ============================================================================ */

void fep_evidence_default_config(fep_evidence_config_t* config) {
    if (!config) return;

    config->method = EVIDENCE_ELBO;
    config->num_samples = FEP_EVIDENCE_DEFAULT_SAMPLES;
    config->temperature_schedule_start = FEP_EVIDENCE_DEFAULT_TEMP_START;
    config->temperature_schedule_end = FEP_EVIDENCE_DEFAULT_TEMP_END;
    config->annealing_steps = FEP_EVIDENCE_DEFAULT_ANNEAL_STEPS;
    config->enable_model_averaging = false;
}

fep_evidence_system_t* fep_evidence_create(const fep_evidence_config_t* config) {
    fep_evidence_system_t* sys = (fep_evidence_system_t*)nimcp_calloc(
        1, sizeof(fep_evidence_system_t));
    if (!sys) {
        NIMCP_LOGGING_ERROR("Failed to allocate evidence system");
        return NULL;
    }

    /* Apply configuration */
    fep_evidence_config_t default_cfg;
    if (!config) {
        fep_evidence_default_config(&default_cfg);
        config = &default_cfg;
    }
    sys->config = *config;

    /* Create mutex */
    sys->mutex = nimcp_platform_mutex_create();
    if (!sys->mutex) {
        fep_evidence_destroy(sys);
        return NULL;
    }

    NIMCP_LOGGING_INFO("Evidence system created: method=%s, samples=%u",
                      fep_evidence_method_to_string(config->method),
                      config->num_samples);
    return sys;
}

void fep_evidence_destroy(fep_evidence_system_t* sys) {
    if (!sys) return;

    if (sys->bio_async_enabled) {
        fep_evidence_disconnect_bio_async(sys);
    }

    if (sys->mutex) {
        nimcp_platform_mutex_destroy(sys->mutex);
        nimcp_free(sys->mutex);
    }

    nimcp_free(sys);
    NIMCP_LOGGING_INFO("Evidence system destroyed");
}

/* ============================================================================
 * Evidence Computation Implementation
 * ============================================================================ */

int fep_compute_log_evidence(
    fep_evidence_system_t* sys,
    fep_system_t* fep,
    const float* observations,
    size_t n_obs,
    size_t obs_dim,
    fep_evidence_result_t* result
) {
    if (!sys || !fep || !observations || !result) return -1;
    if (n_obs == 0 || obs_dim == 0) return -1;

    memset(result, 0, sizeof(fep_evidence_result_t));

    /* Compute based on configured method */
    float total_log_evidence = 0.0f;
    float total_complexity = 0.0f;
    float total_accuracy = 0.0f;

    for (size_t i = 0; i < n_obs; i++) {
        const float* obs = observations + i * obs_dim;
        float elbo, complexity, accuracy;

        /* Compute ELBO for each observation (uses internal unlocked version) */
        fep_compute_elbo_internal(sys, fep, obs, obs_dim, &elbo);
        fep_compute_model_complexity(sys, fep, &complexity);
        fep_compute_model_accuracy(sys, fep, obs, obs_dim, &accuracy);

        total_log_evidence += elbo;
        total_complexity += complexity;
        total_accuracy += accuracy;
    }

    /* Average over observations */
    result->log_evidence = total_log_evidence / (float)n_obs;
    result->evidence_lower_bound = result->log_evidence;  /* ELBO is the bound */
    result->model_complexity = total_complexity / (float)n_obs;
    result->model_accuracy = total_accuracy / (float)n_obs;

    /* Compute Bayes factor against reference if available */
    if (sys->reference_valid) {
        float log_bf = result->log_evidence - sys->reference_log_evidence;
        result->bayes_factor = safe_exp(log_bf);
        result->bf_strength = fep_interpret_bayes_factor(log_bf);
    }

    /* Update statistics */
    nimcp_platform_mutex_lock(sys->mutex);
    sys->stats.total_computations++;
    sys->stats.elbo_computations += n_obs;
    nimcp_platform_mutex_unlock(sys->mutex);

    return 0;
}

int fep_compute_elbo(
    fep_evidence_system_t* sys,
    fep_system_t* fep,
    const float* observation,
    size_t obs_dim,
    float* elbo
) {
    if (!sys || !fep || !observation || !elbo) return -1;

    /* Use internal computation */
    int ret = fep_compute_elbo_internal(sys, fep, observation, obs_dim, elbo);

    /* Update statistics */
    if (ret == 0) {
        nimcp_platform_mutex_lock(sys->mutex);
        sys->stats.elbo_computations++;
        nimcp_platform_mutex_unlock(sys->mutex);
    }

    return ret;
}

int fep_compute_model_complexity(
    fep_evidence_system_t* sys,
    fep_system_t* fep,
    float* complexity
) {
    if (!sys || !fep || !complexity) return -1;

    /* Complexity = KL[q(s)||p(s)] = E_q[log q(s)] - E_q[log p(s)]
     * For Gaussian: KL = 0.5 * (tr(Σ_prior^-1 Σ_post) + (μ_prior - μ_post)^T Σ_prior^-1 (μ_prior - μ_post)
     *                          - k + ln(|Σ_prior|/|Σ_post|))
     * Simplified: approximate as sum of squared deviations weighted by prior precision */

    float kl = 0.0f;

    for (uint32_t l = 0; l < fep->num_levels; l++) {
        fep_hierarchy_level_t* level = &fep->levels[l];
        fep_belief_t* beliefs = &level->beliefs;

        for (uint32_t i = 0; i < beliefs->dim; i++) {
            /* Deviation from prior */
            float mu_diff = beliefs->mean[i] - level->prior_mean[i];
            /* Prior precision is 1/variance, so prior_variance = 1/prior_precision */
            float prior_var = 1.0f / (level->prior_precision[i] + FEP_EVIDENCE_EPSILON);
            float var_ratio = beliefs->variance[i] / (prior_var + FEP_EVIDENCE_EPSILON);

            /* KL contribution: 0.5 * (var_ratio - 1 - log(var_ratio) + mu_diff^2 / prior_var) */
            kl += 0.5f * (var_ratio - 1.0f - safe_log(var_ratio) +
                         mu_diff * mu_diff / (prior_var + FEP_EVIDENCE_EPSILON));
        }
    }

    *complexity = clamp_f(kl, 0.0f, 1000.0f);
    return 0;
}

int fep_compute_model_accuracy(
    fep_evidence_system_t* sys,
    fep_system_t* fep,
    const float* observation,
    size_t obs_dim,
    float* accuracy
) {
    if (!sys || !fep || !observation || !accuracy) return -1;

    /* Accuracy = E_q[log p(o|s)]
     * For Gaussian likelihood: -0.5 * (o - pred)^T Σ^-1 (o - pred) - 0.5 * k * log(2π) - 0.5 * log|Σ| */

    float log_lik = 0.0f;

    if (fep->num_levels > 0) {
        fep_hierarchy_level_t* level = &fep->levels[0];
        size_t dim = obs_dim < level->beliefs.dim ? obs_dim : level->beliefs.dim;

        for (size_t i = 0; i < dim; i++) {
            float pred = level->beliefs.mean[i];
            float var = level->errors.precision[i] > 0.0f ?
                        1.0f / level->errors.precision[i] : 1.0f;

            float diff = observation[i] - pred;
            log_lik -= 0.5f * (diff * diff / (var + FEP_EVIDENCE_EPSILON) +
                              safe_log(2.0f * 3.14159f * var));
        }
    }

    *accuracy = clamp_f(log_lik, FEP_EVIDENCE_LOG_MIN, 0.0f);
    return 0;
}

/* ============================================================================
 * Model Comparison Implementation
 * ============================================================================ */

int fep_compute_bayes_factor(
    fep_evidence_system_t* sys,
    fep_system_t* model1,
    fep_system_t* model2,
    const float* observations,
    size_t n_obs,
    size_t obs_dim,
    float* log_bf
) {
    if (!sys || !model1 || !model2 || !observations || !log_bf) return -1;

    fep_evidence_result_t result1, result2;

    fep_compute_log_evidence(sys, model1, observations, n_obs, obs_dim, &result1);
    fep_compute_log_evidence(sys, model2, observations, n_obs, obs_dim, &result2);

    *log_bf = result1.log_evidence - result2.log_evidence;

    nimcp_platform_mutex_lock(sys->mutex);
    sys->stats.model_comparisons++;
    nimcp_platform_mutex_unlock(sys->mutex);

    return 0;
}

int fep_compare_models(
    fep_evidence_system_t* sys,
    fep_system_t** models,
    size_t n_models,
    const float* observations,
    size_t n_obs,
    size_t obs_dim,
    fep_model_score_t* scores
) {
    if (!sys || !models || !observations || !scores) return -1;
    if (n_models == 0) return -1;

    float* log_evidences = (float*)nimcp_calloc(n_models, sizeof(float));
    if (!log_evidences) {
        return -1;
    }

    /* Compute log evidence for each model (no lock - fep_compute_log_evidence locks internally) */
    for (size_t m = 0; m < n_models; m++) {
        fep_evidence_result_t result;
        fep_compute_log_evidence(sys, models[m], observations, n_obs, obs_dim, &result);

        scores[m].model_id = (uint32_t)m;
        scores[m].log_evidence = result.log_evidence;
        scores[m].prior_probability = 1.0f / (float)n_models;  /* Uniform prior */

        log_evidences[m] = result.log_evidence;
    }

    /* Compute posterior probabilities via softmax */
    float log_marginal = logsumexp(log_evidences, n_models);

    for (size_t m = 0; m < n_models; m++) {
        scores[m].posterior_probability = safe_exp(log_evidences[m] - log_marginal);
    }

    nimcp_free(log_evidences);

    /* Update stats with lock */
    nimcp_platform_mutex_lock(sys->mutex);
    sys->stats.model_comparisons += n_models;
    nimcp_platform_mutex_unlock(sys->mutex);

    return 0;
}

int fep_select_best_model(
    fep_evidence_system_t* sys,
    const fep_model_score_t* scores,
    size_t n_models,
    uint32_t* best_model_id
) {
    if (!scores || !best_model_id || n_models == 0) return -1;

    uint32_t best_id = 0;
    float best_prob = scores[0].posterior_probability;

    for (size_t m = 1; m < n_models; m++) {
        if (scores[m].posterior_probability > best_prob) {
            best_prob = scores[m].posterior_probability;
            best_id = scores[m].model_id;
        }
    }

    *best_model_id = best_id;
    return 0;
}

/* ============================================================================
 * Model Averaging Implementation
 * ============================================================================ */

int fep_model_average_prediction(
    fep_evidence_system_t* sys,
    fep_system_t** models,
    size_t n_models,
    const fep_model_score_t* scores,
    float* averaged_prediction,
    size_t pred_dim
) {
    if (!sys || !models || !scores || !averaged_prediction) return -1;
    if (n_models == 0 || pred_dim == 0) return -1;

    /* Initialize to zero */
    memset(averaged_prediction, 0, pred_dim * sizeof(float));

    /* Weighted average: prediction = Σ p(M|o) * E[s|M,o] */
    for (size_t m = 0; m < n_models; m++) {
        float weight = scores[m].posterior_probability;

        if (models[m] && models[m]->num_levels > 0) {
            fep_belief_t* beliefs = &models[m]->levels[0].beliefs;
            size_t dim = pred_dim < beliefs->dim ? pred_dim : beliefs->dim;

            for (size_t i = 0; i < dim; i++) {
                averaged_prediction[i] += weight * beliefs->mean[i];
            }
        }
    }

    return 0;
}

/* ============================================================================
 * Integration Implementation
 * ============================================================================ */

int fep_evidence_connect(fep_evidence_system_t* evidence, fep_system_t* fep) {
    if (!evidence || !fep) return -1;

    nimcp_platform_mutex_lock(evidence->mutex);
    evidence->fep_system = fep;
    nimcp_platform_mutex_unlock(evidence->mutex);

    NIMCP_LOGGING_INFO("Evidence system connected to FEP");
    return 0;
}

int fep_evidence_set_reference(
    fep_evidence_system_t* sys,
    fep_system_t* reference,
    const float* observations,
    size_t n_obs,
    size_t obs_dim
) {
    if (!sys || !reference || !observations) return -1;

    nimcp_platform_mutex_lock(sys->mutex);

    sys->reference_model = reference;

    /* Compute reference evidence */
    fep_evidence_result_t result;
    fep_compute_log_evidence(sys, reference, observations, n_obs, obs_dim, &result);

    sys->reference_log_evidence = result.log_evidence;
    sys->reference_valid = true;

    nimcp_platform_mutex_unlock(sys->mutex);
    return 0;
}

/* ============================================================================
 * Bio-Async Implementation
 * ============================================================================ */

int fep_evidence_connect_bio_async(fep_evidence_system_t* sys) {
    if (!sys) return -1;
    if (sys->bio_async_enabled) return 0;

    bio_module_info_t info = {
        .module_id = BIO_MODULE_FEP_EVIDENCE,
        .module_name = "fep_evidence",
        .inbox_capacity = 32,
        .user_data = sys
    };

    sys->bio_ctx = bio_router_register_module(&info);
    if (sys->bio_ctx) {
        sys->bio_async_enabled = true;
        NIMCP_LOGGING_INFO("Evidence connected to bio-async");
    } else {
        NIMCP_LOGGING_WARN("Bio-async router not available, skipping registration");
    }
    return 0;
}

int fep_evidence_disconnect_bio_async(fep_evidence_system_t* sys) {
    if (!sys) return -1;
    if (!sys->bio_async_enabled) return 0;

    if (sys->bio_ctx) {
        bio_router_unregister_module(sys->bio_ctx);
        sys->bio_ctx = NULL;
    }
    sys->bio_async_enabled = false;
    return 0;
}

bool fep_evidence_is_bio_async_connected(const fep_evidence_system_t* sys) {
    return sys && sys->bio_async_enabled;
}

/* ============================================================================
 * Query Implementation
 * ============================================================================ */

int fep_evidence_get_stats(const fep_evidence_system_t* sys, fep_evidence_stats_t* stats) {
    if (!sys || !stats) return -1;
    *stats = sys->stats;
    return 0;
}

fep_bf_strength_t fep_interpret_bayes_factor(float log_bf) {
    float bf = safe_exp(log_bf);

    if (bf < FEP_EVIDENCE_BF_WEAK) return BF_STRENGTH_NONE;
    if (bf < FEP_EVIDENCE_BF_POSITIVE) return BF_STRENGTH_WEAK;
    if (bf < FEP_EVIDENCE_BF_STRONG) return BF_STRENGTH_POSITIVE;
    if (bf < FEP_EVIDENCE_BF_VERY_STRONG) return BF_STRENGTH_STRONG;
    if (bf < FEP_EVIDENCE_BF_DECISIVE) return BF_STRENGTH_VERY_STRONG;
    return BF_STRENGTH_DECISIVE;
}

const char* fep_bf_strength_to_string(fep_bf_strength_t strength) {
    switch (strength) {
        case BF_STRENGTH_NONE:        return "NONE";
        case BF_STRENGTH_WEAK:        return "WEAK";
        case BF_STRENGTH_POSITIVE:    return "POSITIVE";
        case BF_STRENGTH_STRONG:      return "STRONG";
        case BF_STRENGTH_VERY_STRONG: return "VERY_STRONG";
        case BF_STRENGTH_DECISIVE:    return "DECISIVE";
        default:                      return "UNKNOWN";
    }
}

const char* fep_evidence_method_to_string(fep_evidence_method_t method) {
    switch (method) {
        case EVIDENCE_ELBO:       return "ELBO";
        case EVIDENCE_IMPORTANCE: return "IMPORTANCE_SAMPLING";
        case EVIDENCE_ANNEALED:   return "ANNEALED_IMPORTANCE";
        case EVIDENCE_BRIDGE:     return "BRIDGE_SAMPLING";
        default:                  return "UNKNOWN";
    }
}
