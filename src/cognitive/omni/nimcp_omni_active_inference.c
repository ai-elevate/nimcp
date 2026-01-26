/**
 * @file nimcp_omni_active_inference.c
 * @brief Implementation of Omnidirectional Active Inference
 */

#include "cognitive/omni/nimcp_omni_active_inference.h"
#include "cognitive/omni/nimcp_omni_precision.h"
#include "cognitive/omni/nimcp_omni_kg_sync.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/exception/nimcp_exception_macros.h"

#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>

//=============================================================================
#include <stddef.h>  /* for NULL */
// Health Agent Integration (Phase 8: System-Wide Health Integration)
//=============================================================================
struct nimcp_health_agent;
typedef struct nimcp_health_agent nimcp_health_agent_t;
extern void nimcp_health_agent_heartbeat_ex(nimcp_health_agent_t* agent,
                                             const char* operation,
                                             float progress);

/** Global health agent for omni_active_inference module */
static nimcp_health_agent_t* g_omni_active_inference_health_agent = NULL;

/**
 * @brief Set health agent for omni_active_inference heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
static void omni_active_inference_set_health_agent(nimcp_health_agent_t* agent) {
    g_omni_active_inference_health_agent = agent;
}

/** @brief Send heartbeat from omni_active_inference module */
static inline void omni_active_inference_heartbeat(const char* operation, float progress) {
    if (g_omni_active_inference_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_omni_active_inference_health_agent, operation, progress);
    }
}


/* ============================================================================
 * Constants
 * ============================================================================ */

#define AI_INITIAL_POLICY_CAPACITY    16
#define AI_INITIAL_GOAL_CAPACITY      8
#define AI_EPSILON                    1e-8f

/* ============================================================================
 * Thread-Safe PRNG (xorshift64)
 * ============================================================================ */

static __thread uint64_t ai_prng_state = 0;

static void ai_prng_seed(void) {
    /* Seed with address XOR'd with a constant for uniqueness per thread */
    ai_prng_state = (uint64_t)(uintptr_t)&ai_prng_state ^ 0x123456789ABCDEF0ULL;
    if (ai_prng_state == 0) ai_prng_state = 1;  /* State must be non-zero */
}

static uint64_t ai_prng_next(void) {
    if (ai_prng_state == 0) ai_prng_seed();
    uint64_t x = ai_prng_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    ai_prng_state = x;
    return x;
}

/* ============================================================================
 * Static Helpers
 * ============================================================================ */

static float compute_risk(const omni_ai_goal_t* goal,
                          const float* predicted_obs,
                          uint32_t obs_dim) {
    if (!goal || !goal->active || !goal->preferred_obs || !predicted_obs) {
        return 0.0f;
    }

    /* Risk = KL divergence approximation (squared distance) */
    float risk = 0.0f;
    uint32_t dim = (obs_dim < goal->obs_dim) ? obs_dim : goal->obs_dim;

    for (uint32_t i = 0; i < dim; i++) {
        float diff = predicted_obs[i] - goal->preferred_obs[i];
        risk += diff * diff;
    }

    return goal->goal_precision * risk;
}

static float compute_ambiguity(const float* belief_variance,
                                uint32_t belief_dim) {
    if (!belief_variance) return 0.0f;

    /* Ambiguity = expected entropy (sum of log variances) */
    float ambiguity = 0.0f;
    for (uint32_t i = 0; i < belief_dim; i++) {
        float var = belief_variance[i] + AI_EPSILON;
        ambiguity += logf(var);
    }

    return 0.5f * ambiguity;
}

static float compute_intrinsic_value(const float* current_belief,
                                      const float* predicted_belief,
                                      uint32_t belief_dim) {
    if (!current_belief || !predicted_belief) return 0.0f;

    /* Intrinsic value = expected information gain (KL divergence) */
    float info_gain = 0.0f;
    for (uint32_t i = 0; i < belief_dim; i++) {
        float p = current_belief[i] + AI_EPSILON;
        float q = predicted_belief[i] + AI_EPSILON;
        info_gain += p * logf(p / q);
    }

    return info_gain;
}

static float random_uniform(void) {
    /* Thread-safe random using xorshift64 */
    return (float)(ai_prng_next() >> 11) / (float)(1ULL << 53);
}

static void compute_policy_efe(omni_active_inference_t* ai,
                                omni_ai_policy_t* policy) {
    if (!ai || !policy) return;

    /* Compute forward EFE (standard) */
    float risk_total = 0.0f;
    for (uint32_t g = 0; g < ai->num_goals; g++) {
        if (ai->goals[g].active) {
            /* Simplified: use current observation as prediction */
            risk_total += compute_risk(&ai->goals[g], ai->current_obs, ai->obs_dim);
        }
    }

    float ambiguity = compute_ambiguity(NULL, 0); /* Placeholder */

    /* Apply config weights */
    policy->efe_risk = risk_total * ai->config.risk_weight;
    policy->efe_ambiguity = ambiguity * ai->config.ambiguity_weight;
    policy->efe_intrinsic = 0.0f; /* Would compute info gain */
    policy->efe_extrinsic = risk_total * ai->config.extrinsic_weight;

    /* Forward EFE */
    policy->efe_forward = policy->efe_risk + policy->efe_ambiguity;

    /* Backward EFE (hindsight) - lower for actions that explained observation */
    policy->efe_backward = policy->efe_forward * 0.8f; /* Simplified */

    /* Lateral EFE (cross-modal coherence) */
    policy->efe_lateral = policy->efe_forward * 0.5f; /* Simplified */

    /* Total EFE with directional weights */
    policy->efe_total = ai->config.forward_weight * policy->efe_forward +
                        ai->config.backward_weight * policy->efe_backward +
                        ai->config.lateral_weight * policy->efe_lateral;

    /* Get precision from context if available */
    if (ai->precision_ctx && ai->config.use_precision_context) {
        float prec = omni_precision_get_aggregate(ai->precision_ctx,
                                                   BIO_MODULE_OMNI_ACTIVE_INFERENCE);
        policy->precision = prec;
    } else {
        policy->precision = ai->config.policy_precision;
    }
}

/* ============================================================================
 * Configuration API
 * ============================================================================ */

int omni_ai_default_config(omni_ai_config_t* config) {
    NIMCP_CHECK_THROW(config, NIMCP_ERROR_INVALID_PARAM, "config is NULL");

    memset(config, 0, sizeof(omni_ai_config_t));

    config->select_mode = OMNI_AI_SELECT_SOFTMAX;
    config->efe_mode = OMNI_AI_EFE_BALANCED;
    config->policy_precision = OMNI_AI_DEFAULT_PRECISION;
    config->exploration_rate = OMNI_AI_DEFAULT_EXPLORATION;

    config->max_horizon = 4;
    config->num_policies = 8;

    /* EFE component weights */
    config->risk_weight = 1.0f;
    config->ambiguity_weight = 1.0f;
    config->intrinsic_weight = 0.5f;
    config->extrinsic_weight = 1.0f;

    /* Directional weights */
    config->forward_weight = 1.0f;
    config->backward_weight = 0.3f;
    config->lateral_weight = 0.2f;

    config->use_precision_context = true;
    config->enable_bio_async = true;
    config->enable_logging = false;

    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Lifecycle API
 * ============================================================================ */

omni_active_inference_t* omni_ai_create(
    const omni_ai_config_t* config,
    uint32_t action_dim,
    uint32_t obs_dim)
{
    omni_active_inference_t* ai = nimcp_calloc(1, sizeof(omni_active_inference_t));
    if (!ai) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "ai is NULL");

        return NULL;

    }

    /* Apply configuration */
    if (config) {
        memcpy(&ai->config, config, sizeof(omni_ai_config_t));
    } else {
        omni_ai_default_config(&ai->config);
    }

    /* Allocate policy array */
    ai->policy_capacity = AI_INITIAL_POLICY_CAPACITY;
    ai->policies = nimcp_calloc(ai->policy_capacity, sizeof(omni_ai_policy_t));
    if (!ai->policies) {
        nimcp_free(ai);
        return NULL;
    }

    /* Allocate goal array */
    ai->goal_capacity = AI_INITIAL_GOAL_CAPACITY;
    ai->goals = nimcp_calloc(ai->goal_capacity, sizeof(omni_ai_goal_t));
    if (!ai->goals) {
        nimcp_free(ai->policies);
        nimcp_free(ai);
        return NULL;
    }

    /* Allocate observation buffer */
    ai->obs_dim = obs_dim;
    if (obs_dim > 0) {
        ai->current_obs = nimcp_calloc(obs_dim, sizeof(float));
        if (!ai->current_obs) {
            nimcp_free(ai->goals);
            nimcp_free(ai->policies);
            nimcp_free(ai);
            return NULL;
        }
    }

    /* Create mutex */
    ai->mutex = nimcp_mutex_create(NULL);
    if (!ai->mutex) {
        if (ai->current_obs) nimcp_free(ai->current_obs);
        nimcp_free(ai->goals);
        nimcp_free(ai->policies);
        nimcp_free(ai);
        return NULL;
    }

    /* Initialize stats */
    memset(&ai->stats, 0, sizeof(omni_ai_stats_t));
    ai->stats.min_efe = FLT_MAX;

    return ai;
}

void omni_ai_destroy(omni_active_inference_t* ai) {
    if (!ai) return;

    if (ai->bio_async_connected) {
        omni_ai_disconnect_bio_async(ai);
    }

    /* Free policies */
    for (uint32_t i = 0; i < ai->num_policies; i++) {
        if (ai->policies[i].actions) {
            nimcp_free(ai->policies[i].actions);
        }
    }
    nimcp_free(ai->policies);

    /* Free goals */
    for (uint32_t i = 0; i < ai->num_goals; i++) {
        if (ai->goals[i].preferred_obs) {
            nimcp_free(ai->goals[i].preferred_obs);
        }
    }
    nimcp_free(ai->goals);

    /* Free state buffers */
    if (ai->current_obs) nimcp_free(ai->current_obs);
    if (ai->current_belief) nimcp_free(ai->current_belief);

    if (ai->mutex) nimcp_mutex_free(ai->mutex);

    nimcp_free(ai);
}

int omni_ai_reset(omni_active_inference_t* ai) {
    NIMCP_CHECK_THROW(ai, NIMCP_ERROR_INVALID_PARAM, "ai is NULL");

    nimcp_mutex_lock(ai->mutex);

    /* Clear policies */
    for (uint32_t i = 0; i < ai->num_policies; i++) {
        if (ai->policies[i].actions) {
            nimcp_free(ai->policies[i].actions);
            ai->policies[i].actions = NULL;
        }
    }
    ai->num_policies = 0;

    /* Clear goals */
    for (uint32_t i = 0; i < ai->num_goals; i++) {
        ai->goals[i].active = false;
    }

    /* Reset observations */
    if (ai->current_obs) {
        memset(ai->current_obs, 0, ai->obs_dim * sizeof(float));
    }

    /* Reset stats */
    memset(&ai->stats, 0, sizeof(omni_ai_stats_t));
    ai->stats.min_efe = FLT_MAX;

    nimcp_mutex_unlock(ai->mutex);
    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Policy API
 * ============================================================================ */

int omni_ai_add_policy(omni_active_inference_t* ai,
                        const float* actions,
                        uint32_t horizon,
                        uint32_t action_dim) {
    if (!ai || !actions) return -1;
    if (horizon == 0 || action_dim == 0) return -1;

    nimcp_mutex_lock(ai->mutex);

    /* Expand if needed */
    if (ai->num_policies >= ai->policy_capacity) {
        uint32_t new_cap = ai->policy_capacity * 2;
        omni_ai_policy_t* new_policies = nimcp_realloc(
            ai->policies, new_cap * sizeof(omni_ai_policy_t));
        if (!new_policies) {
            nimcp_mutex_unlock(ai->mutex);
            return -1;
        }
        ai->policies = new_policies;
        ai->policy_capacity = new_cap;
    }

    omni_ai_policy_t* policy = &ai->policies[ai->num_policies];
    memset(policy, 0, sizeof(omni_ai_policy_t));

    policy->policy_id = ai->num_policies;
    policy->horizon = horizon;
    policy->action_dim = action_dim;

    /* Copy actions */
    size_t action_size = horizon * action_dim * sizeof(float);
    policy->actions = nimcp_malloc(action_size);
    if (!policy->actions) {
        nimcp_mutex_unlock(ai->mutex);
        return -1;
    }
    memcpy(policy->actions, actions, action_size);

    int idx = (int)ai->num_policies;
    ai->num_policies++;

    nimcp_mutex_unlock(ai->mutex);
    return idx;
}

int omni_ai_generate_random_policies(omni_active_inference_t* ai,
                                      uint32_t num_policies,
                                      uint32_t horizon) {
    if (!ai) return 0;
    if (horizon == 0 || horizon > OMNI_AI_MAX_HORIZON) return 0;

    nimcp_mutex_lock(ai->mutex);

    uint32_t action_dim = ai->obs_dim > 0 ? ai->obs_dim : 8; /* Default */
    uint32_t generated = 0;

    for (uint32_t p = 0; p < num_policies && ai->num_policies < OMNI_AI_MAX_POLICIES; p++) {
        /* Allocate random actions */
        float* actions = nimcp_malloc(horizon * action_dim * sizeof(float));
        if (!actions) continue;

        for (uint32_t i = 0; i < horizon * action_dim; i++) {
            actions[i] = random_uniform() * 2.0f - 1.0f; /* [-1, 1] */
        }

        nimcp_mutex_unlock(ai->mutex);
        int idx = omni_ai_add_policy(ai, actions, horizon, action_dim);
        nimcp_mutex_lock(ai->mutex);

        nimcp_free(actions);

        if (idx >= 0) generated++;
    }

    nimcp_mutex_unlock(ai->mutex);
    return (int)generated;
}

int omni_ai_clear_policies(omni_active_inference_t* ai) {
    NIMCP_CHECK_THROW(ai, NIMCP_ERROR_INVALID_PARAM, "ai is NULL");

    nimcp_mutex_lock(ai->mutex);

    for (uint32_t i = 0; i < ai->num_policies; i++) {
        if (ai->policies[i].actions) {
            nimcp_free(ai->policies[i].actions);
            ai->policies[i].actions = NULL;
        }
    }
    ai->num_policies = 0;

    nimcp_mutex_unlock(ai->mutex);
    return NIMCP_SUCCESS;
}

int omni_ai_get_policy(const omni_active_inference_t* ai,
                        uint32_t index,
                        omni_ai_policy_t* policy) {
    NIMCP_CHECK_THROW(ai, NIMCP_ERROR_INVALID_PARAM, "ai is NULL");
    NIMCP_CHECK_THROW(policy, NIMCP_ERROR_INVALID_PARAM, "policy is NULL");
    NIMCP_CHECK_THROW(index < ai->num_policies, NIMCP_ERROR_OUT_OF_RANGE, "policy index out of range");

    nimcp_mutex_lock(((omni_active_inference_t*)ai)->mutex);
    memcpy(policy, &ai->policies[index], sizeof(omni_ai_policy_t));
    /* Note: action pointer is shallow copied */
    nimcp_mutex_unlock(((omni_active_inference_t*)ai)->mutex);

    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Goal API
 * ============================================================================ */

int omni_ai_set_goal(omni_active_inference_t* ai,
                      const float* preferred,
                      uint32_t obs_dim,
                      float precision) {
    if (!ai) return -1;

    /* Clear existing goals first */
    omni_ai_clear_goals(ai);

    return omni_ai_add_goal(ai, preferred, obs_dim, precision);
}

int omni_ai_add_goal(omni_active_inference_t* ai,
                      const float* preferred,
                      uint32_t obs_dim,
                      float precision) {
    if (!ai || !preferred || obs_dim == 0) return -1;

    nimcp_mutex_lock(ai->mutex);

    /* Expand if needed */
    if (ai->num_goals >= ai->goal_capacity) {
        uint32_t new_cap = ai->goal_capacity * 2;
        omni_ai_goal_t* new_goals = nimcp_realloc(
            ai->goals, new_cap * sizeof(omni_ai_goal_t));
        if (!new_goals) {
            nimcp_mutex_unlock(ai->mutex);
            return -1;
        }
        ai->goals = new_goals;
        ai->goal_capacity = new_cap;
    }

    omni_ai_goal_t* goal = &ai->goals[ai->num_goals];
    memset(goal, 0, sizeof(omni_ai_goal_t));

    goal->obs_dim = obs_dim;
    goal->goal_precision = precision;
    goal->active = true;

    goal->preferred_obs = nimcp_malloc(obs_dim * sizeof(float));
    if (!goal->preferred_obs) {
        nimcp_mutex_unlock(ai->mutex);
        return -1;
    }
    memcpy(goal->preferred_obs, preferred, obs_dim * sizeof(float));

    int idx = (int)ai->num_goals;
    ai->num_goals++;

    nimcp_mutex_unlock(ai->mutex);
    return idx;
}

int omni_ai_clear_goals(omni_active_inference_t* ai) {
    NIMCP_CHECK_THROW(ai, NIMCP_ERROR_INVALID_PARAM, "ai is NULL");

    nimcp_mutex_lock(ai->mutex);

    for (uint32_t i = 0; i < ai->num_goals; i++) {
        if (ai->goals[i].preferred_obs) {
            nimcp_free(ai->goals[i].preferred_obs);
            ai->goals[i].preferred_obs = NULL;
        }
        ai->goals[i].active = false;
    }
    ai->num_goals = 0;

    nimcp_mutex_unlock(ai->mutex);
    return NIMCP_SUCCESS;
}

int omni_ai_set_goal_active(omni_active_inference_t* ai,
                             uint32_t goal_index,
                             bool active) {
    NIMCP_CHECK_THROW(ai, NIMCP_ERROR_INVALID_PARAM, "ai is NULL");
    NIMCP_CHECK_THROW(goal_index < ai->num_goals, NIMCP_ERROR_OUT_OF_RANGE, "goal_index out of range");

    nimcp_mutex_lock(ai->mutex);
    ai->goals[goal_index].active = active;
    nimcp_mutex_unlock(ai->mutex);

    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Observation API
 * ============================================================================ */

int omni_ai_update_observation(omni_active_inference_t* ai,
                                const float* obs,
                                uint32_t obs_dim) {
    NIMCP_CHECK_THROW(ai, NIMCP_ERROR_INVALID_PARAM, "ai is NULL");
    NIMCP_CHECK_THROW(obs, NIMCP_ERROR_INVALID_PARAM, "obs is NULL");

    nimcp_mutex_lock(ai->mutex);

    /* Reallocate if dimension changed */
    if (obs_dim != ai->obs_dim) {
        float* new_obs = nimcp_realloc(ai->current_obs, obs_dim * sizeof(float));
        if (!new_obs && obs_dim > 0) {
            nimcp_mutex_unlock(ai->mutex);
            return NIMCP_ERROR_NO_MEMORY;
        }
        ai->current_obs = new_obs;
        ai->obs_dim = obs_dim;
    }

    memcpy(ai->current_obs, obs, obs_dim * sizeof(float));

    nimcp_mutex_unlock(ai->mutex);
    return NIMCP_SUCCESS;
}

int omni_ai_update_belief(omni_active_inference_t* ai,
                           const float* belief,
                           uint32_t belief_dim) {
    NIMCP_CHECK_THROW(ai, NIMCP_ERROR_INVALID_PARAM, "ai is NULL");
    NIMCP_CHECK_THROW(belief, NIMCP_ERROR_INVALID_PARAM, "belief is NULL");

    nimcp_mutex_lock(ai->mutex);

    if (belief_dim != ai->belief_dim) {
        float* new_belief = nimcp_realloc(ai->current_belief, belief_dim * sizeof(float));
        if (!new_belief && belief_dim > 0) {
            nimcp_mutex_unlock(ai->mutex);
            return NIMCP_ERROR_NO_MEMORY;
        }
        ai->current_belief = new_belief;
        ai->belief_dim = belief_dim;
    }

    memcpy(ai->current_belief, belief, belief_dim * sizeof(float));

    nimcp_mutex_unlock(ai->mutex);
    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Inference API
 * ============================================================================ */

int omni_ai_evaluate_policies(omni_active_inference_t* ai) {
    NIMCP_CHECK_THROW(ai, NIMCP_ERROR_INVALID_PARAM, "ai is NULL");

    nimcp_mutex_lock(ai->mutex);

    for (uint32_t p = 0; p < ai->num_policies; p++) {
        compute_policy_efe(ai, &ai->policies[p]);
    }

    nimcp_mutex_unlock(ai->mutex);
    return NIMCP_SUCCESS;
}

int omni_ai_select_action_forward(omni_active_inference_t* ai,
                                   omni_ai_action_result_t* result) {
    NIMCP_CHECK_THROW(ai, NIMCP_ERROR_INVALID_PARAM, "ai is NULL");
    NIMCP_CHECK_THROW(result, NIMCP_ERROR_INVALID_PARAM, "result is NULL");
    NIMCP_CHECK_THROW(ai->num_policies > 0, NIMCP_ERROR_INVALID_STATE, "no policies defined");

    nimcp_mutex_lock(ai->mutex);

    /* Evaluate policies if not done */
    for (uint32_t p = 0; p < ai->num_policies; p++) {
        compute_policy_efe(ai, &ai->policies[p]);
    }

    /* Compute policy probabilities via softmax */
    float* efe_vals = nimcp_malloc(ai->num_policies * sizeof(float));
    float* probs = nimcp_malloc(ai->num_policies * sizeof(float));
    if (!efe_vals || !probs) {
        if (efe_vals) nimcp_free(efe_vals);
        if (probs) nimcp_free(probs);
        nimcp_mutex_unlock(ai->mutex);
        return NIMCP_ERROR_NO_MEMORY;
    }

    for (uint32_t p = 0; p < ai->num_policies; p++) {
        efe_vals[p] = ai->policies[p].efe_forward;
    }

    float precision = ai->config.policy_precision;
    if (ai->precision_ctx && ai->config.use_precision_context) {
        precision = omni_precision_get_aggregate(ai->precision_ctx,
                                                  BIO_MODULE_OMNI_ACTIVE_INFERENCE);
    }

    omni_ai_softmax_efe(efe_vals, probs, ai->num_policies, precision);

    /* Select policy based on mode */
    int selected = 0;
    switch (ai->config.select_mode) {
        case OMNI_AI_SELECT_SOFTMAX:
            selected = omni_ai_sample_policy(probs, ai->num_policies);
            break;
        case OMNI_AI_SELECT_GREEDY:
            selected = omni_ai_get_best_policy(ai);
            break;
        case OMNI_AI_SELECT_THOMPSON:
        case OMNI_AI_SELECT_UCB:
        default:
            selected = omni_ai_sample_policy(probs, ai->num_policies);
            break;
    }

    /* Exploration with epsilon-greedy */
    if (random_uniform() < ai->config.exploration_rate) {
        selected = (int)(random_uniform() * ai->num_policies);
        ai->stats.explorations++;
    } else {
        ai->stats.exploitations++;
    }

    if (selected < 0) selected = 0;

    /* Fill result */
    omni_ai_policy_t* policy = &ai->policies[selected];
    result->selected_policy = (uint32_t)selected;
    result->direction = OMNI_AI_DIR_FORWARD;
    result->efe = policy->efe_forward;
    result->confidence = probs[selected];

    /* Copy action - use min of allocated and policy dims to avoid overflow */
    if (result->action && policy->actions && policy->action_dim > 0) {
        uint32_t copy_dim = (result->action_dim < policy->action_dim) ?
                            result->action_dim : policy->action_dim;
        if (copy_dim > 0) {
            memcpy(result->action, policy->actions, copy_dim * sizeof(float));
        }
        result->action_dim = copy_dim;
    } else {
        result->action_dim = 0;
    }

    /* Update stats */
    ai->stats.total_inferences++;
    ai->stats.forward_selections++;
    ai->stats.avg_efe = (ai->stats.avg_efe * (ai->stats.total_inferences - 1) +
                         policy->efe_forward) / ai->stats.total_inferences;
    ai->stats.avg_confidence = (ai->stats.avg_confidence * (ai->stats.total_inferences - 1) +
                                result->confidence) / ai->stats.total_inferences;
    if (policy->efe_forward < ai->stats.min_efe) {
        ai->stats.min_efe = policy->efe_forward;
    }

    nimcp_free(efe_vals);
    nimcp_free(probs);

    nimcp_mutex_unlock(ai->mutex);
    return NIMCP_SUCCESS;
}

int omni_ai_infer_action_backward(omni_active_inference_t* ai,
                                   const float* outcome,
                                   uint32_t outcome_dim,
                                   omni_ai_action_result_t* result) {
    NIMCP_CHECK_THROW(ai, NIMCP_ERROR_INVALID_PARAM, "ai is NULL");
    NIMCP_CHECK_THROW(result, NIMCP_ERROR_INVALID_PARAM, "result is NULL");
    NIMCP_CHECK_THROW(ai->num_policies > 0, NIMCP_ERROR_INVALID_STATE, "no policies defined");

    nimcp_mutex_lock(ai->mutex);

    /* Find policy that best explains outcome (min backward EFE) */
    int best_idx = 0;
    float best_efe = FLT_MAX;

    for (uint32_t p = 0; p < ai->num_policies; p++) {
        compute_policy_efe(ai, &ai->policies[p]);

        if (ai->policies[p].efe_backward < best_efe) {
            best_efe = ai->policies[p].efe_backward;
            best_idx = (int)p;
        }
    }

    /* Fill result */
    omni_ai_policy_t* policy = &ai->policies[best_idx];
    result->selected_policy = (uint32_t)best_idx;
    result->direction = OMNI_AI_DIR_BACKWARD;
    result->efe = policy->efe_backward;
    result->confidence = 1.0f / (1.0f + policy->efe_backward);

    /* Copy action - use min of allocated and policy dims to avoid overflow */
    if (result->action && policy->actions && policy->action_dim > 0) {
        uint32_t copy_dim = (result->action_dim < policy->action_dim) ?
                            result->action_dim : policy->action_dim;
        if (copy_dim > 0) {
            memcpy(result->action, policy->actions, copy_dim * sizeof(float));
        }
        result->action_dim = copy_dim;
    } else {
        result->action_dim = 0;
    }

    ai->stats.total_inferences++;
    ai->stats.backward_inferences++;

    nimcp_mutex_unlock(ai->mutex);
    return NIMCP_SUCCESS;
}

int omni_ai_select_action_omni(omni_active_inference_t* ai,
                                omni_ai_action_result_t* result) {
    NIMCP_CHECK_THROW(ai, NIMCP_ERROR_INVALID_PARAM, "ai is NULL");
    NIMCP_CHECK_THROW(result, NIMCP_ERROR_INVALID_PARAM, "result is NULL");
    NIMCP_CHECK_THROW(ai->num_policies > 0, NIMCP_ERROR_INVALID_STATE, "no policies defined");

    nimcp_mutex_lock(ai->mutex);

    /* Evaluate with combined directional EFE */
    for (uint32_t p = 0; p < ai->num_policies; p++) {
        compute_policy_efe(ai, &ai->policies[p]);
    }

    /* Find best policy by total EFE */
    int best_idx = 0;
    float best_efe = FLT_MAX;

    for (uint32_t p = 0; p < ai->num_policies; p++) {
        if (ai->policies[p].efe_total < best_efe) {
            best_efe = ai->policies[p].efe_total;
            best_idx = (int)p;
        }
    }

    /* Fill result */
    omni_ai_policy_t* policy = &ai->policies[best_idx];
    result->selected_policy = (uint32_t)best_idx;
    result->direction = OMNI_AI_DIR_FORWARD; /* Primary direction */
    result->efe = policy->efe_total;
    result->confidence = 1.0f / (1.0f + policy->efe_total);

    /* Copy action - use min of allocated and policy dims to avoid overflow */
    if (result->action && policy->actions && policy->action_dim > 0) {
        uint32_t copy_dim = (result->action_dim < policy->action_dim) ?
                            result->action_dim : policy->action_dim;
        if (copy_dim > 0) {
            memcpy(result->action, policy->actions, copy_dim * sizeof(float));
        }
        result->action_dim = copy_dim;
    } else {
        result->action_dim = 0;
    }

    ai->stats.total_inferences++;

    nimcp_mutex_unlock(ai->mutex);
    return NIMCP_SUCCESS;
}

float omni_ai_get_policy_efe(const omni_active_inference_t* ai,
                              uint32_t policy_index,
                              omni_ai_direction_t direction) {
    if (!ai || policy_index >= ai->num_policies) return FLT_MAX;

    const omni_ai_policy_t* p = &ai->policies[policy_index];
    switch (direction) {
        case OMNI_AI_DIR_FORWARD: return p->efe_forward;
        case OMNI_AI_DIR_BACKWARD: return p->efe_backward;
        case OMNI_AI_DIR_LATERAL: return p->efe_lateral;
        case OMNI_AI_DIR_HIERARCHICAL: return p->efe_total;
        default: return p->efe_total;
    }
}

/* ============================================================================
 * Integration API
 * ============================================================================ */

int omni_ai_connect_precision(omni_active_inference_t* ai,
                               omni_precision_ctx_t* precision_ctx) {
    NIMCP_CHECK_THROW(ai, NIMCP_ERROR_INVALID_PARAM, "ai is NULL");

    nimcp_mutex_lock(ai->mutex);
    ai->precision_ctx = precision_ctx;

    /* Register with precision context */
    if (precision_ctx) {
        omni_precision_register_module(precision_ctx,
                                        BIO_MODULE_OMNI_ACTIVE_INFERENCE,
                                        "omni_active_inference",
                                        OMNI_AI_DEFAULT_PRECISION);
        omni_precision_enable_channel(precision_ctx,
                                       BIO_MODULE_OMNI_ACTIVE_INFERENCE,
                                       OMNI_PREC_CHANNEL_FORWARD,
                                       OMNI_AI_DEFAULT_PRECISION);
    }

    nimcp_mutex_unlock(ai->mutex);
    return NIMCP_SUCCESS;
}

int omni_ai_connect_fep(omni_active_inference_t* ai, void* fep_system) {
    NIMCP_CHECK_THROW(ai, NIMCP_ERROR_INVALID_PARAM, "ai is NULL");

    nimcp_mutex_lock(ai->mutex);
    ai->fep_system = fep_system;
    nimcp_mutex_unlock(ai->mutex);

    return NIMCP_SUCCESS;
}

int omni_ai_connect_kg_sync(omni_active_inference_t* ai,
                             omni_kg_sync_t* kg_sync) {
    NIMCP_CHECK_THROW(ai, NIMCP_ERROR_INVALID_PARAM, "ai is NULL");

    nimcp_mutex_lock(ai->mutex);
    ai->kg_sync = kg_sync;
    nimcp_mutex_unlock(ai->mutex);

    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Bio-Async Message Handlers
 * ============================================================================ */

static nimcp_error_t handle_policy_eval_request(
    const void* msg,
    size_t msg_size,
    nimcp_bio_promise_t response_promise,
    void* user_data)
{
    omni_active_inference_t* ai = (omni_active_inference_t*)user_data;
    NIMCP_CHECK_THROW(ai, NIMCP_ERROR_INVALID_PARAM, "ai context is NULL");

    omni_ai_evaluate_policies(ai);

    (void)msg;
    (void)msg_size;
    (void)response_promise;
    return NIMCP_SUCCESS;
}

static nimcp_error_t handle_action_select_request(
    const void* msg,
    size_t msg_size,
    nimcp_bio_promise_t response_promise,
    void* user_data)
{
    omni_active_inference_t* ai = (omni_active_inference_t*)user_data;
    NIMCP_CHECK_THROW(ai, NIMCP_ERROR_INVALID_PARAM, "ai context is NULL");

    omni_ai_action_result_t* result = omni_ai_action_result_create(
        ai->obs_dim > 0 ? ai->obs_dim : 8);
    if (result) {
        omni_ai_select_action_forward(ai, result);
        omni_ai_action_result_destroy(result);
    }

    (void)msg;
    (void)msg_size;
    (void)response_promise;
    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Bio-Async API
 * ============================================================================ */

int omni_ai_connect_bio_async(omni_active_inference_t* ai) {
    NIMCP_CHECK_THROW(ai, NIMCP_ERROR_INVALID_PARAM, "ai is NULL");
    if (ai->bio_async_connected) return NIMCP_SUCCESS;

    bio_module_info_t info = {
        .module_id = BIO_MODULE_OMNI_ACTIVE_INFERENCE,
        .module_name = "omni_active_inference",
        .inbox_capacity = 32,
        .user_data = ai
    };

    bio_module_context_t ctx = bio_router_register_module(&info);
    if (!ctx) {
        return NIMCP_ERROR_OPERATION_FAILED;
    }

    ai->bio_context = ctx;

    bio_router_register_handler(ctx, BIO_MSG_OMNI_PREDICT_REQUEST,
                                 handle_policy_eval_request);
    bio_router_register_handler(ctx, BIO_MSG_OMNI_DIRECTION_SWITCH,
                                 handle_action_select_request);

    ai->bio_async_connected = true;
    return NIMCP_SUCCESS;
}

int omni_ai_disconnect_bio_async(omni_active_inference_t* ai) {
    NIMCP_CHECK_THROW(ai, NIMCP_ERROR_INVALID_PARAM, "ai is NULL");
    if (!ai->bio_async_connected) return NIMCP_SUCCESS;

    if (ai->bio_context) {
        bio_router_unregister_module(ai->bio_context);
        ai->bio_context = NULL;
    }

    ai->bio_async_connected = false;
    return NIMCP_SUCCESS;
}

bool omni_ai_is_bio_async_connected(const omni_active_inference_t* ai) {
    if (!ai) return false;
    return ai->bio_async_connected;
}

/* ============================================================================
 * Query API
 * ============================================================================ */

int omni_ai_get_stats(const omni_active_inference_t* ai,
                       omni_ai_stats_t* stats) {
    NIMCP_CHECK_THROW(ai, NIMCP_ERROR_INVALID_PARAM, "ai is NULL");
    NIMCP_CHECK_THROW(stats, NIMCP_ERROR_INVALID_PARAM, "stats is NULL");

    nimcp_mutex_lock(((omni_active_inference_t*)ai)->mutex);
    memcpy(stats, &ai->stats, sizeof(omni_ai_stats_t));
    nimcp_mutex_unlock(((omni_active_inference_t*)ai)->mutex);

    return NIMCP_SUCCESS;
}

int omni_ai_reset_stats(omni_active_inference_t* ai) {
    NIMCP_CHECK_THROW(ai, NIMCP_ERROR_INVALID_PARAM, "ai is NULL");

    nimcp_mutex_lock(ai->mutex);
    memset(&ai->stats, 0, sizeof(omni_ai_stats_t));
    ai->stats.min_efe = FLT_MAX;
    nimcp_mutex_unlock(ai->mutex);

    return NIMCP_SUCCESS;
}

int omni_ai_get_best_policy(const omni_active_inference_t* ai) {
    if (!ai || ai->num_policies == 0) return -1;

    int best_idx = 0;
    float best_efe = FLT_MAX;

    for (uint32_t p = 0; p < ai->num_policies; p++) {
        if (ai->policies[p].efe_total < best_efe) {
            best_efe = ai->policies[p].efe_total;
            best_idx = (int)p;
        }
    }

    return best_idx;
}

int omni_ai_get_policy_probs(const omni_active_inference_t* ai,
                              float* probs,
                              uint32_t max_policies) {
    NIMCP_CHECK_THROW(ai, NIMCP_ERROR_INVALID_PARAM, "ai is NULL");
    NIMCP_CHECK_THROW(probs, NIMCP_ERROR_INVALID_PARAM, "probs is NULL");

    nimcp_mutex_lock(((omni_active_inference_t*)ai)->mutex);

    uint32_t n = (ai->num_policies < max_policies) ? ai->num_policies : max_policies;
    for (uint32_t i = 0; i < n; i++) {
        probs[i] = ai->policies[i].probability;
    }

    nimcp_mutex_unlock(((omni_active_inference_t*)ai)->mutex);
    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Utility API
 * ============================================================================ */

int omni_ai_softmax_efe(const float* efe,
                         float* probs,
                         uint32_t n,
                         float precision) {
    NIMCP_CHECK_THROW(efe, NIMCP_ERROR_INVALID_PARAM, "efe array is NULL");
    NIMCP_CHECK_THROW(probs, NIMCP_ERROR_INVALID_PARAM, "probs array is NULL");
    NIMCP_CHECK_THROW(n > 0, NIMCP_ERROR_INVALID_PARAM, "n must be greater than 0");

    /* Find max for numerical stability */
    float max_val = -FLT_MAX;
    for (uint32_t i = 0; i < n; i++) {
        float neg_efe = -precision * efe[i];
        if (neg_efe > max_val) max_val = neg_efe;
    }

    /* Compute exp and sum */
    float sum = 0.0f;
    for (uint32_t i = 0; i < n; i++) {
        probs[i] = expf(-precision * efe[i] - max_val);
        sum += probs[i];
    }

    /* Normalize */
    if (sum > AI_EPSILON) {
        for (uint32_t i = 0; i < n; i++) {
            probs[i] /= sum;
        }
    } else {
        /* Uniform if all zero */
        float uniform = 1.0f / n;
        for (uint32_t i = 0; i < n; i++) {
            probs[i] = uniform;
        }
    }

    return NIMCP_SUCCESS;
}

int omni_ai_sample_policy(const float* probs, uint32_t n) {
    if (!probs || n == 0) return 0;

    float r = random_uniform();
    float cumsum = 0.0f;

    for (uint32_t i = 0; i < n; i++) {
        cumsum += probs[i];
        if (r <= cumsum) {
            return (int)i;
        }
    }

    return (int)(n - 1);
}

omni_ai_action_result_t* omni_ai_action_result_create(uint32_t action_dim) {
    omni_ai_action_result_t* result = nimcp_calloc(1, sizeof(omni_ai_action_result_t));
    if (!result) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "result is NULL");

        return NULL;

    }

    if (action_dim > 0) {
        result->action = nimcp_calloc(action_dim, sizeof(float));
        if (!result->action) {
            nimcp_free(result);
            return NULL;
        }
    }

    result->action_dim = action_dim;
    return result;
}

void omni_ai_action_result_destroy(omni_ai_action_result_t* result) {
    if (!result) return;
    if (result->action) nimcp_free(result->action);
    nimcp_free(result);
}

/* ============================================================================
 * String Conversion API
 * ============================================================================ */

const char* omni_ai_direction_to_string(omni_ai_direction_t direction) {
    switch (direction) {
        case OMNI_AI_DIR_FORWARD: return "FORWARD";
        case OMNI_AI_DIR_BACKWARD: return "BACKWARD";
        case OMNI_AI_DIR_LATERAL: return "LATERAL";
        case OMNI_AI_DIR_HIERARCHICAL: return "HIERARCHICAL";
        default: return "UNKNOWN";
    }
}

const char* omni_ai_select_mode_to_string(omni_ai_select_mode_t mode) {
    switch (mode) {
        case OMNI_AI_SELECT_SOFTMAX: return "SOFTMAX";
        case OMNI_AI_SELECT_GREEDY: return "GREEDY";
        case OMNI_AI_SELECT_THOMPSON: return "THOMPSON";
        case OMNI_AI_SELECT_UCB: return "UCB";
        default: return "UNKNOWN";
    }
}

const char* omni_ai_efe_mode_to_string(omni_ai_efe_mode_t mode) {
    switch (mode) {
        case OMNI_AI_EFE_BALANCED: return "BALANCED";
        case OMNI_AI_EFE_RISK_SEEKING: return "RISK_SEEKING";
        case OMNI_AI_EFE_CURIOUS: return "CURIOUS";
        case OMNI_AI_EFE_ADAPTIVE: return "ADAPTIVE";
        default: return "UNKNOWN";
    }
}
