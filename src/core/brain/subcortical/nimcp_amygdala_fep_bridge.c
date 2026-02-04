/**
 * @file nimcp_amygdala_fep_bridge.c
 * @brief Amygdala-to-Free Energy Principle Integration Bridge Implementation
 *
 * Implements FEP-based threat prediction and active inference for defensive behavior.
 */

#include "utils/bridge/nimcp_bridge_base.h"
#include "core/brain/subcortical/nimcp_amygdala_fep_bridge.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

//=============================================================================
#include <stddef.h>  /* for NULL */
#include "utils/logging/nimcp_logging.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/fault_tolerance/nimcp_health_agent_macros.h"
#include "mesh/nimcp_mesh_participant.h"
#include "mesh/nimcp_mesh_adapter.h"

NIMCP_DECLARE_HEALTH_AGENT_ATOMIC(amygdala_fep_bridge)
//=============================================================================
// Mesh Participant Registration
//=============================================================================

static mesh_participant_id_t g_amygdala_fep_bridge_mesh_id = 0;
static mesh_participant_registry_t* g_amygdala_fep_bridge_mesh_registry = NULL;

nimcp_error_t amygdala_fep_bridge_mesh_register(mesh_participant_registry_t* registry) {
    if (!registry) return NIMCP_ERROR_NULL_POINTER;
    if (g_amygdala_fep_bridge_mesh_id != 0) return NIMCP_SUCCESS;
    mesh_participant_interface_t iface;
    mesh_participant_interface_init(&iface);
    strncpy(iface.module_name, "amygdala_fep_bridge", MESH_MAX_NAME_LEN - 1);
    iface.type = MESH_PARTICIPANT_MODULE;
    iface.home_channel = mesh_adapter_get_default_channel(MESH_ADAPTER_CATEGORY_SUBCORTICAL);
    mesh_participant_config_t config;
    mesh_participant_config_init(&config);
    config.module_name = "amygdala_fep_bridge";
    config.type = MESH_PARTICIPANT_MODULE;
    config.home_channel = iface.home_channel;
    nimcp_error_t err = mesh_participant_register(registry, &iface, &config, &g_amygdala_fep_bridge_mesh_id);
    if (err == NIMCP_SUCCESS) g_amygdala_fep_bridge_mesh_registry = registry;
    return err;
}

void amygdala_fep_bridge_mesh_unregister(void) {
    if (g_amygdala_fep_bridge_mesh_registry && g_amygdala_fep_bridge_mesh_id != 0) {
        mesh_participant_unregister(g_amygdala_fep_bridge_mesh_registry, g_amygdala_fep_bridge_mesh_id);
        g_amygdala_fep_bridge_mesh_id = 0;
        g_amygdala_fep_bridge_mesh_registry = NULL;
    }
}


#define LOG_MODULE "AMYGDALA_FEP_BRIDGE"


//=============================================================================
// Internal Structures
//=============================================================================

struct amyg_fep_bridge_s {
    bridge_base_t base;              /**< MUST be first: base bridge infrastructure */
    amygdala_t* amygdala;
    void* fep_system;
    amyg_fep_config_t config;

    /* Current state beliefs (threat dimensions) */
    float beliefs[AMYG_FEP_MAX_STATE_DIM];
    float precision[AMYG_FEP_MAX_STATE_DIM];
    uint32_t state_dim;

    /* Generative model */
    amyg_fep_model_t current_model;
    float model_evidence[AMYG_FEP_NUM_THREAT_MODELS];

    /* Prediction errors */
    amyg_fep_errors_t current_errors;
    float free_energy;
    float surprise;

    /* Active inference */
    amyg_fep_action_t current_action;
    float expected_free_energy_cache[5];  /* One per action type */
    amyg_fep_safety_belief_t safety_belief;

    /* Interoceptive state */
    amyg_fep_interoception_t interoception;
    bool interoception_valid;

    /* Observations */
    float last_observations[AMYG_FEP_MAX_OBS_DIM];
    uint32_t obs_dim;

    /* Modulation */
    float current_arousal;
    float current_stress;

    /* Timing */
    float current_time_ms;

    /* Statistics */
    amyg_fep_stats_t stats;
};

//=============================================================================
// Helper Functions
//=============================================================================

static float clamp(float x, float min_val, float max_val) {
    if (x < min_val) return min_val;
    if (x > max_val) return max_val;
    return x;
}

/**
 * @brief Compute threat prediction based on model
 *
 * Each model has different priors about threat level
 */
static float compute_model_threat_prior(amyg_fep_model_t model) {
    switch (model) {
        case AMYG_FEP_MODEL_SAFE:     return 0.1f;  /* Expect low threat */
        case AMYG_FEP_MODEL_VIGILANT: return 0.3f;  /* Elevated baseline */
        case AMYG_FEP_MODEL_THREAT:   return 0.6f;  /* Active threat */
        case AMYG_FEP_MODEL_DANGER:   return 0.8f;  /* High danger */
        case AMYG_FEP_MODEL_PANIC:    return 0.95f; /* Overwhelming threat */
        default: return 0.5f;
    }
}

/**
 * @brief Predict threat evolution based on model
 */
static void compute_threat_prediction(
    amyg_fep_model_t model,
    const float* state,
    float dt_ms,
    float* predicted
) {
    float dt_s = dt_ms * 0.001f;
    float threat = state[0];  /* Primary threat estimate */
    float threat_velocity = state[1];  /* Rate of change */

    switch (model) {
        case AMYG_FEP_MODEL_SAFE:
            /* Threat decays toward zero */
            predicted[0] = threat * expf(-0.5f * dt_s);
            predicted[1] = -0.2f * threat;
            break;

        case AMYG_FEP_MODEL_VIGILANT:
            /* Slow decay, quick response to increase */
            predicted[0] = threat + threat_velocity * dt_s * 0.5f;
            predicted[1] = threat_velocity * 0.9f;
            break;

        case AMYG_FEP_MODEL_THREAT:
            /* Threat persists */
            predicted[0] = threat + threat_velocity * dt_s;
            predicted[1] = threat_velocity;
            break;

        case AMYG_FEP_MODEL_DANGER:
            /* Threat tends to escalate */
            predicted[0] = threat + (threat_velocity + 0.1f) * dt_s;
            predicted[1] = threat_velocity + 0.05f * dt_s;
            break;

        case AMYG_FEP_MODEL_PANIC:
            /* Maximum threat expectation */
            predicted[0] = fminf(1.0f, threat + 0.2f * dt_s);
            predicted[1] = 0.1f;
            break;

        default:
            memcpy(predicted, state, 2 * sizeof(float));
    }

    /* Clamp predictions */
    predicted[0] = clamp(predicted[0], 0.0f, 1.0f);
    predicted[1] = clamp(predicted[1], -1.0f, 1.0f);
}

/**
 * @brief Compute action's effect on expected threat
 */
static float action_threat_reduction(amyg_fep_action_t action, float current_threat) {
    switch (action) {
        case AMYG_FEP_ACTION_OBSERVE:
            /* No direct reduction, but information gain */
            return 0.0f;

        case AMYG_FEP_ACTION_ORIENT:
            /* Better localize threat */
            return 0.05f;

        case AMYG_FEP_ACTION_FREEZE:
            /* Reduce detection, modest safety increase */
            return 0.15f * current_threat;

        case AMYG_FEP_ACTION_AVOID:
            /* Escape removes threat */
            return 0.5f * current_threat;

        case AMYG_FEP_ACTION_APPROACH:
            /* Might reduce uncertainty but risky */
            return -0.1f;  /* Could increase threat */

        default:
            return 0.0f;
    }
}

//=============================================================================
// Configuration
//=============================================================================

int amyg_fep_bridge_default_config(amyg_fep_config_t* config) {
    if (!config) return -1;

    /* Model settings */
    config->default_model = AMYG_FEP_MODEL_VIGILANT;
    config->auto_model_selection = true;
    config->model_evidence_threshold = 0.6f;

    /* Precision settings */
    config->precision_mode = AMYG_FEP_PRECISION_ADAPTIVE;
    config->sensory_precision = AMYG_FEP_DEFAULT_PRECISION;
    config->interoceptive_precision = 0.7f;
    config->contextual_precision = 0.6f;
    config->prior_precision = 0.5f;

    /* Arousal/stress modulation */
    config->arousal_precision_gain = 0.5f;
    config->stress_precision_gain = 0.3f;
    config->anxiety_prior_boost = 0.2f;

    /* Inference settings */
    config->learning_rate = 0.15f;
    config->inference_steps = 8;
    config->action_precision = 1.0f;

    /* Integration */
    config->use_interoception = true;
    config->use_context = true;
    config->prediction_horizon_ms = 500.0f;

    return 0;
}

int amyg_fep_bridge_validate_config(const amyg_fep_config_t* config) {
    if (!config) return -1;

    if (config->default_model > AMYG_FEP_MODEL_PANIC) return -1;
    if (config->precision_mode > AMYG_FEP_PRECISION_INTEROCEPTIVE) return -1;
    if (config->sensory_precision < 0.0f) return -1;
    if (config->interoceptive_precision < 0.0f) return -1;
    if (config->contextual_precision < 0.0f) return -1;
    if (config->prior_precision < 0.0f) return -1;
    if (config->learning_rate <= 0.0f || config->learning_rate > 1.0f) return -1;
    if (config->inference_steps == 0) return -1;
    if (config->prediction_horizon_ms < 0.0f) return -1;

    return 0;
}

//=============================================================================
// Lifecycle
//=============================================================================

amyg_fep_bridge_t* amyg_fep_bridge_create(
    amygdala_t* amygdala,
    void* fep_system,
    const amyg_fep_config_t* config
) {
    amyg_fep_bridge_t* bridge = nimcp_calloc(1, sizeof(amyg_fep_bridge_t));
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return NULL;

    }

    if (config) {
        if (amyg_fep_bridge_validate_config(config) != 0) {
            nimcp_free(bridge);
            return NULL;
        }
        bridge->config = *config;
    } else {
        amyg_fep_bridge_default_config(&bridge->config);
    }

    bridge->amygdala = amygdala;
    bridge->fep_system = fep_system;

    /* Initialize state beliefs
     * Dimensions: [threat, threat_velocity, uncertainty, arousal, context_safety]
     */
    bridge->state_dim = 5;
    for (uint32_t i = 0; i < bridge->state_dim; i++) {
        bridge->beliefs[i] = 0.0f;
        bridge->precision[i] = bridge->config.prior_precision;
    }

    /* Initialize model evidence with uniform prior */
    for (int i = 0; i < AMYG_FEP_NUM_THREAT_MODELS; i++) {
        bridge->model_evidence[i] = 0.2f;
    }

    bridge->current_model = bridge->config.default_model;
    bridge->current_action = AMYG_FEP_ACTION_OBSERVE;
    bridge->safety_belief = AMYG_FEP_BELIEF_UNCERTAIN;

    return bridge;
}

void amyg_fep_bridge_destroy(amyg_fep_bridge_t* bridge) {
    if (!bridge) return;
    NIMCP_LOGGING_DEBUG("Destroying %s bridge", "amygdala_fep");
    nimcp_free(bridge);
}

int amyg_fep_bridge_reset(amyg_fep_bridge_t* bridge) {
    if (!bridge) return -1;

    /* Reset beliefs */
    for (uint32_t i = 0; i < bridge->state_dim; i++) {
        bridge->beliefs[i] = 0.0f;
        bridge->precision[i] = bridge->config.prior_precision;
    }

    /* Reset model evidence */
    for (int i = 0; i < AMYG_FEP_NUM_THREAT_MODELS; i++) {
        bridge->model_evidence[i] = 0.2f;
    }
    bridge->current_model = bridge->config.default_model;

    /* Reset errors */
    memset(&bridge->current_errors, 0, sizeof(bridge->current_errors));
    bridge->free_energy = 0.0f;
    bridge->surprise = 0.0f;

    bridge->current_action = AMYG_FEP_ACTION_OBSERVE;
    bridge->safety_belief = AMYG_FEP_BELIEF_UNCERTAIN;
    bridge->current_time_ms = 0.0f;
    bridge->interoception_valid = false;
    memset(&bridge->stats, 0, sizeof(bridge->stats));

    return 0;
}

//=============================================================================
// Prediction Error
//=============================================================================

int amyg_fep_compute_errors(
    amyg_fep_bridge_t* bridge,
    const float* observations,
    uint32_t obs_dim,
    amyg_fep_errors_t* errors
) {
    if (!bridge || !observations || !errors) return -1;

    /* Store observations */
    bridge->obs_dim = (obs_dim > AMYG_FEP_MAX_OBS_DIM) ?
        AMYG_FEP_MAX_OBS_DIM : obs_dim;
    memcpy(bridge->last_observations, observations, bridge->obs_dim * sizeof(float));

    /* Get model's expected threat level */
    float expected_threat = compute_model_threat_prior(bridge->current_model);

    /* Observed threat is first observation dimension */
    float observed_threat = (obs_dim > 0) ? clamp(observations[0], 0.0f, 1.0f) : 0.0f;

    /* Safety prediction error: threat when expected safe */
    float safety_error = 0.0f;
    if (expected_threat < 0.5f && observed_threat > expected_threat) {
        safety_error = (observed_threat - expected_threat) * bridge->config.sensory_precision;
    }

    /* Threat prediction error: safe when expected threat */
    float threat_error = 0.0f;
    if (expected_threat > 0.5f && observed_threat < expected_threat) {
        threat_error = (expected_threat - observed_threat) * bridge->config.sensory_precision;
    }

    /* Interoceptive error (body state mismatch) */
    float intero_error = 0.0f;
    if (bridge->interoception_valid && bridge->config.use_interoception) {
        float expected_arousal = expected_threat * 0.8f;
        float observed_arousal = (bridge->interoception.heart_rate_deviation + 1.0f) * 0.5f;
        intero_error = fabsf(observed_arousal - expected_arousal) *
            bridge->config.interoceptive_precision;
    }

    /* Contextual error (context familiarity mismatch) */
    float context_error = 0.0f;
    if (obs_dim > 1 && bridge->config.use_context) {
        float context_safety = clamp(observations[1], 0.0f, 1.0f);
        float expected_context_safety = 1.0f - expected_threat;
        context_error = fabsf(context_safety - expected_context_safety) *
            bridge->config.contextual_precision;
    }

    /* Precision-weighted error */
    float pw_error = 0.0f;
    uint32_t compare_dim = (obs_dim < bridge->state_dim) ? obs_dim : bridge->state_dim;
    for (uint32_t i = 0; i < compare_dim; i++) {
        float diff = observations[i] - bridge->beliefs[i];
        pw_error += diff * diff * bridge->precision[i];
    }
    pw_error = sqrtf(pw_error / (float)(compare_dim > 0 ? compare_dim : 1));

    /* Total free energy (complexity + inaccuracy) */
    float complexity = 0.0f;
    for (uint32_t i = 0; i < bridge->state_dim; i++) {
        complexity += bridge->precision[i] * bridge->beliefs[i] * bridge->beliefs[i];
    }
    complexity *= 0.1f;

    float inaccuracy = safety_error + threat_error + intero_error + context_error;
    float free_energy = complexity + inaccuracy;

    /* Store results */
    errors->safety_error = safety_error;
    errors->threat_error = threat_error;
    errors->interoceptive_error = intero_error;
    errors->contextual_error = context_error;
    errors->total_free_energy = free_energy;
    errors->precision_weighted_error = pw_error;

    bridge->current_errors = *errors;
    bridge->free_energy = free_energy;
    bridge->surprise = safety_error + threat_error;

    /* Update statistics */
    bridge->stats.avg_prediction_error =
        (bridge->stats.avg_prediction_error * bridge->stats.predictions_made + pw_error) /
        (bridge->stats.predictions_made + 1);
    bridge->stats.predictions_made++;

    /* Detect threats and false alarms */
    if (observed_threat > 0.5f && expected_threat < 0.5f) {
        bridge->stats.threat_detections++;
    }
    if (observed_threat < 0.3f && expected_threat > 0.6f) {
        bridge->stats.false_alarms++;
    }

    return 0;
}

float amyg_fep_get_free_energy(const amyg_fep_bridge_t* bridge) {
    if (!bridge) return 0.0f;
    return bridge->free_energy;
}

float amyg_fep_get_surprise(const amyg_fep_bridge_t* bridge) {
    if (!bridge) return 0.0f;
    return bridge->surprise;
}

int amyg_fep_update_precision(
    amyg_fep_bridge_t* bridge,
    float arousal,
    float stress
) {
    if (!bridge) return -1;

    bridge->current_arousal = clamp(arousal, 0.0f, 1.0f);
    bridge->current_stress = clamp(stress, 0.0f, 1.0f);

    if (bridge->config.precision_mode == AMYG_FEP_PRECISION_ADAPTIVE ||
        bridge->config.precision_mode == AMYG_FEP_PRECISION_INTEROCEPTIVE) {

        /* Arousal increases sensory precision (heightened attention) */
        float arousal_boost = bridge->current_arousal * bridge->config.arousal_precision_gain;

        /* Stress can either increase or decrease precision
         * Moderate stress increases, high stress decreases (inverted U) */
        float stress_effect = bridge->current_stress * (1.0f - bridge->current_stress) *
            bridge->config.stress_precision_gain * 2.0f;

        /* Update sensory precision */
        bridge->precision[0] = bridge->config.sensory_precision *
            (1.0f + arousal_boost + stress_effect);

        /* Update threat velocity precision */
        bridge->precision[1] = bridge->config.sensory_precision *
            (1.0f + arousal_boost * 0.5f);

        /* Interoceptive precision increases with arousal */
        if (bridge->config.precision_mode == AMYG_FEP_PRECISION_INTEROCEPTIVE) {
            bridge->precision[3] = bridge->config.interoceptive_precision *
                (1.0f + arousal_boost * 0.7f);
        }
    }

    return 0;
}

int amyg_fep_set_interoception(
    amyg_fep_bridge_t* bridge,
    const amyg_fep_interoception_t* intero
) {
    if (!bridge || !intero) return -1;

    bridge->interoception = *intero;
    bridge->interoception_valid = true;

    /* Update arousal belief from body signals */
    float body_arousal = (intero->heart_rate_deviation + 1.0f) * 0.3f +
                         intero->skin_conductance * 0.3f +
                         intero->muscle_tension * 0.2f +
                         (intero->gut_feeling + 1.0f) * 0.1f;
    bridge->beliefs[3] = clamp(body_arousal, 0.0f, 1.0f);

    return 0;
}

//=============================================================================
// Active Inference
//=============================================================================

int amyg_fep_infer_state(
    amyg_fep_bridge_t* bridge,
    const float* observations,
    uint32_t obs_dim,
    amyg_fep_inference_t* inference
) {
    if (!bridge || !observations || !inference) return -1;

    uint32_t compare_dim = (obs_dim < bridge->state_dim) ? obs_dim : bridge->state_dim;

    /* Gradient descent on free energy to update beliefs */
    for (uint32_t step = 0; step < bridge->config.inference_steps; step++) {
        for (uint32_t i = 0; i < compare_dim; i++) {
            /* Prediction error gradient */
            float error = observations[i] - bridge->beliefs[i];

            /* Update belief with precision-weighted error */
            bridge->beliefs[i] += bridge->config.learning_rate * error *
                bridge->precision[i];

            /* Clamp beliefs to valid range */
            bridge->beliefs[i] = clamp(bridge->beliefs[i], -1.0f, 1.0f);
        }
        bridge->stats.inference_steps_total++;
    }

    /* Update safety belief based on threat estimate */
    float threat = bridge->beliefs[0];
    if (threat < 0.3f) {
        bridge->safety_belief = AMYG_FEP_BELIEF_SAFE;
    } else if (threat > 0.6f) {
        bridge->safety_belief = AMYG_FEP_BELIEF_UNSAFE;
    } else {
        bridge->safety_belief = AMYG_FEP_BELIEF_UNCERTAIN;
    }

    /* Copy to output */
    memcpy(inference->beliefs, bridge->beliefs, bridge->state_dim * sizeof(float));
    memcpy(inference->precision, bridge->precision, bridge->state_dim * sizeof(float));
    inference->state_dim = bridge->state_dim;
    inference->current_action = bridge->current_action;
    inference->expected_free_energy = bridge->expected_free_energy_cache[bridge->current_action];
    inference->safety_belief = bridge->safety_belief;

    return 0;
}

int amyg_fep_select_action(
    amyg_fep_bridge_t* bridge,
    amyg_fep_action_t* action,
    float* action_value
) {
    if (!bridge || !action) return -1;

    /* Compute expected free energy for each action */
    float min_efe = 1e10f;
    amyg_fep_action_t best_action = AMYG_FEP_ACTION_OBSERVE;

    for (int a = 0; a < 5; a++) {
        float efe = amyg_fep_expected_free_energy(bridge, (amyg_fep_action_t)a);
        bridge->expected_free_energy_cache[a] = efe;

        if (efe < min_efe) {
            min_efe = efe;
            best_action = (amyg_fep_action_t)a;
        }
    }

    *action = best_action;
    if (action_value) {
        *action_value = -min_efe;  /* Convert to value (higher is better) */
    }

    return 0;
}

int amyg_fep_apply_action(
    amyg_fep_bridge_t* bridge,
    amyg_fep_action_t action
) {
    if (!bridge) return -1;
    if (action > AMYG_FEP_ACTION_APPROACH) return -1;

    bridge->current_action = action;

    /* Action affects precision and model selection */
    float threat = bridge->beliefs[0];

    switch (action) {
        case AMYG_FEP_ACTION_OBSERVE:
            /* Increase sensory precision (gathering information) */
            bridge->precision[0] *= 1.1f;
            break;

        case AMYG_FEP_ACTION_ORIENT:
            /* Focus attention, high precision on threat location */
            bridge->precision[0] *= 1.2f;
            bridge->precision[1] *= 1.1f;
            break;

        case AMYG_FEP_ACTION_FREEZE:
            /* Reduce threat by being inconspicuous */
            bridge->beliefs[0] *= 0.95f;
            if (bridge->config.auto_model_selection && threat > 0.5f) {
                bridge->current_model = AMYG_FEP_MODEL_THREAT;
            }
            break;

        case AMYG_FEP_ACTION_AVOID:
            /* Escape reduces threat substantially */
            bridge->beliefs[0] *= 0.7f;
            if (bridge->config.auto_model_selection) {
                bridge->current_model = AMYG_FEP_MODEL_VIGILANT;
            }
            break;

        case AMYG_FEP_ACTION_APPROACH:
            /* Approach reduces uncertainty but risky */
            bridge->precision[0] *= 1.3f;  /* More information */
            bridge->beliefs[2] *= 0.9f;    /* Reduce uncertainty */
            break;
    }

    return 0;
}

float amyg_fep_expected_free_energy(
    const amyg_fep_bridge_t* bridge,
    amyg_fep_action_t action
) {
    if (!bridge) return 1e10f;

    float current_threat = bridge->beliefs[0];
    float current_uncertainty = bridge->beliefs[2];

    /* Expected free energy = Risk + Ambiguity */

    /* Risk: Expected divergence from preferred state (safety) */
    float threat_reduction = action_threat_reduction(action, current_threat);
    float expected_threat = fmaxf(0.0f, current_threat - threat_reduction);
    float risk = expected_threat * 2.0f;  /* Prefer safety */

    /* Ambiguity: Expected uncertainty after action */
    float ambiguity = 0.0f;
    switch (action) {
        case AMYG_FEP_ACTION_OBSERVE:
            ambiguity = current_uncertainty * 0.8f;  /* Reduces uncertainty */
            break;
        case AMYG_FEP_ACTION_ORIENT:
            ambiguity = current_uncertainty * 0.6f;  /* More info about threat */
            break;
        case AMYG_FEP_ACTION_FREEZE:
            ambiguity = current_uncertainty * 1.1f;  /* Uncertainty persists */
            break;
        case AMYG_FEP_ACTION_AVOID:
            ambiguity = current_uncertainty * 0.9f;  /* Some resolution */
            break;
        case AMYG_FEP_ACTION_APPROACH:
            ambiguity = current_uncertainty * 0.5f;  /* Most information */
            break;
        default:
            ambiguity = current_uncertainty;
    }

    /* Action cost (metabolic/risk cost) */
    float action_cost = 0.0f;
    switch (action) {
        case AMYG_FEP_ACTION_OBSERVE:   action_cost = 0.0f; break;
        case AMYG_FEP_ACTION_ORIENT:    action_cost = 0.1f; break;
        case AMYG_FEP_ACTION_FREEZE:    action_cost = 0.2f; break;
        case AMYG_FEP_ACTION_AVOID:     action_cost = 0.3f; break;
        case AMYG_FEP_ACTION_APPROACH:  action_cost = 0.4f; break;
        default: action_cost = 0.0f;
    }

    float efe = risk + ambiguity + action_cost;
    return efe;
}

//=============================================================================
// Generative Model
//=============================================================================

int amyg_fep_set_model(
    amyg_fep_bridge_t* bridge,
    amyg_fep_model_t model
) {
    if (!bridge) return -1;
    if (model > AMYG_FEP_MODEL_PANIC) return -1;

    if (bridge->current_model != model) {
        bridge->stats.model_switches++;
    }
    bridge->current_model = model;
    return 0;
}

amyg_fep_model_t amyg_fep_get_best_model(const amyg_fep_bridge_t* bridge) {
    if (!bridge) return AMYG_FEP_MODEL_SAFE;

    float best_evidence = -1e10f;
    amyg_fep_model_t best_model = AMYG_FEP_MODEL_SAFE;

    for (int i = 0; i < AMYG_FEP_NUM_THREAT_MODELS; i++) {
        if (bridge->model_evidence[i] > best_evidence) {
            best_evidence = bridge->model_evidence[i];
            best_model = (amyg_fep_model_t)i;
        }
    }

    return best_model;
}

float amyg_fep_get_model_evidence(
    const amyg_fep_bridge_t* bridge,
    amyg_fep_model_t model
) {
    if (!bridge) return 0.0f;
    if (model > AMYG_FEP_MODEL_PANIC) return 0.0f;

    return bridge->model_evidence[model];
}

int amyg_fep_predict(
    amyg_fep_bridge_t* bridge,
    float horizon_ms,
    float* predicted_state,
    uint32_t state_dim
) {
    if (!bridge || !predicted_state) return -1;

    float temp_state[AMYG_FEP_MAX_STATE_DIM];
    memcpy(temp_state, bridge->beliefs, bridge->state_dim * sizeof(float));

    /* Iteratively predict forward */
    float dt_ms = 50.0f;  /* 20 Hz steps */
    float t = 0.0f;

    while (t < horizon_ms) {
        float next_state[AMYG_FEP_MAX_STATE_DIM];
        compute_threat_prediction(bridge->current_model, temp_state, dt_ms, next_state);

        /* Copy remaining state dimensions unchanged */
        for (uint32_t i = 2; i < bridge->state_dim; i++) {
            next_state[i] = temp_state[i];
        }

        memcpy(temp_state, next_state, bridge->state_dim * sizeof(float));
        t += dt_ms;
    }

    /* Copy result */
    uint32_t copy_dim = (state_dim < bridge->state_dim) ? state_dim : bridge->state_dim;
    memcpy(predicted_state, temp_state, copy_dim * sizeof(float));

    return (int)copy_dim;
}

int amyg_fep_condition(
    amyg_fep_bridge_t* bridge,
    const float* cs_features,
    uint32_t cs_dim,
    float threat_intensity
) {
    if (!bridge || !cs_features) return -1;

    /* Conditioning shifts priors toward threat expectation */
    float intensity = clamp(threat_intensity, 0.0f, 1.0f);

    /* Increase threat prior */
    bridge->beliefs[0] = fmaxf(bridge->beliefs[0], intensity * 0.8f);

    /* Increase model evidence for threat models */
    if (intensity > 0.5f) {
        bridge->model_evidence[AMYG_FEP_MODEL_THREAT] += 0.1f * intensity;
        bridge->model_evidence[AMYG_FEP_MODEL_DANGER] += 0.05f * intensity;
    }

    /* Normalize model evidence */
    float sum = 0.0f;
    for (int i = 0; i < AMYG_FEP_NUM_THREAT_MODELS; i++) {
        sum += bridge->model_evidence[i];
    }
    for (int i = 0; i < AMYG_FEP_NUM_THREAT_MODELS; i++) {
        bridge->model_evidence[i] /= sum;
    }

    (void)cs_dim;  /* Could use for feature-specific conditioning */

    return 0;
}

int amyg_fep_extinction(
    amyg_fep_bridge_t* bridge,
    const float* cs_features,
    uint32_t cs_dim
) {
    if (!bridge || !cs_features) return -1;

    /* Extinction reduces threat expectations */
    bridge->beliefs[0] *= 0.9f;  /* Gradual reduction */

    /* Increase evidence for safe model */
    bridge->model_evidence[AMYG_FEP_MODEL_SAFE] += 0.05f;
    bridge->model_evidence[AMYG_FEP_MODEL_VIGILANT] += 0.03f;

    /* Decrease evidence for danger models */
    bridge->model_evidence[AMYG_FEP_MODEL_DANGER] *= 0.95f;
    bridge->model_evidence[AMYG_FEP_MODEL_PANIC] *= 0.9f;

    /* Normalize */
    float sum = 0.0f;
    for (int i = 0; i < AMYG_FEP_NUM_THREAT_MODELS; i++) {
        sum += bridge->model_evidence[i];
    }
    for (int i = 0; i < AMYG_FEP_NUM_THREAT_MODELS; i++) {
        bridge->model_evidence[i] /= sum;
    }

    (void)cs_dim;

    return 0;
}

//=============================================================================
// Integration
//=============================================================================

int amyg_fep_connect_amygdala(
    amyg_fep_bridge_t* bridge,
    amygdala_t* amygdala
) {
    if (!bridge) return -1;
    bridge->amygdala = amygdala;
    return 0;
}

int amyg_fep_connect_system(
    amyg_fep_bridge_t* bridge,
    void* fep_system
) {
    if (!bridge) return -1;
    bridge->fep_system = fep_system;
    return 0;
}

int amyg_fep_update(amyg_fep_bridge_t* bridge, float dt_ms) {
    if (!bridge) return -1;

    bridge->current_time_ms += dt_ms;

    /* Update beliefs with model prediction */
    float predicted[AMYG_FEP_MAX_STATE_DIM];
    compute_threat_prediction(bridge->current_model, bridge->beliefs, dt_ms, predicted);

    /* Blend prediction with current beliefs */
    float blend = 0.2f;
    for (uint32_t i = 0; i < 2; i++) {  /* Only threat dimensions */
        bridge->beliefs[i] = (1.0f - blend) * bridge->beliefs[i] + blend * predicted[i];
    }

    /* Update model evidence based on prediction accuracy */
    float error = bridge->current_errors.precision_weighted_error;
    float evidence_update = 1.0f - clamp(error, 0.0f, 1.0f);

    bridge->model_evidence[bridge->current_model] =
        0.95f * bridge->model_evidence[bridge->current_model] +
        0.05f * evidence_update;

    /* Auto model selection */
    if (bridge->config.auto_model_selection) {
        amyg_fep_model_t best = amyg_fep_get_best_model(bridge);
        if (best != bridge->current_model &&
            bridge->model_evidence[best] > bridge->config.model_evidence_threshold) {
            amyg_fep_set_model(bridge, best);
        }
    }

    /* Update stats */
    bridge->stats.avg_free_energy =
        (bridge->stats.avg_free_energy * 0.99f) + (bridge->free_energy * 0.01f);
    bridge->stats.avg_precision =
        (bridge->stats.avg_precision * 0.99f) + (bridge->precision[0] * 0.01f);

    for (int i = 0; i < AMYG_FEP_NUM_THREAT_MODELS; i++) {
        bridge->stats.model_evidence[i] = bridge->model_evidence[i];
    }

    return 0;
}

int amyg_fep_sync_with_amygdala(amyg_fep_bridge_t* bridge) {
    if (!bridge) return -1;
    /* Would sync beliefs with amygdala fear/threat state */
    return 0;
}

//=============================================================================
// Statistics
//=============================================================================

int amyg_fep_bridge_get_stats(
    const amyg_fep_bridge_t* bridge,
    amyg_fep_stats_t* stats
) {
    if (!bridge || !stats) return -1;
    *stats = bridge->stats;
    return 0;
}

int amyg_fep_bridge_reset_stats(amyg_fep_bridge_t* bridge) {
    if (!bridge) return -1;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return 0;
}

//=============================================================================
// Utility
//=============================================================================

const char* amyg_fep_model_name(amyg_fep_model_t model) {
    switch (model) {
        case AMYG_FEP_MODEL_SAFE:     return "safe";
        case AMYG_FEP_MODEL_VIGILANT: return "vigilant";
        case AMYG_FEP_MODEL_THREAT:   return "threat";
        case AMYG_FEP_MODEL_DANGER:   return "danger";
        case AMYG_FEP_MODEL_PANIC:    return "panic";
        default: return "unknown";
    }
}

const char* amyg_fep_action_name(amyg_fep_action_t action) {
    switch (action) {
        case AMYG_FEP_ACTION_OBSERVE:  return "observe";
        case AMYG_FEP_ACTION_ORIENT:   return "orient";
        case AMYG_FEP_ACTION_FREEZE:   return "freeze";
        case AMYG_FEP_ACTION_AVOID:    return "avoid";
        case AMYG_FEP_ACTION_APPROACH: return "approach";
        default: return "unknown";
    }
}

const char* amyg_fep_precision_mode_name(amyg_fep_precision_mode_t mode) {
    switch (mode) {
        case AMYG_FEP_PRECISION_FIXED:         return "fixed";
        case AMYG_FEP_PRECISION_ADAPTIVE:      return "adaptive";
        case AMYG_FEP_PRECISION_INTEROCEPTIVE: return "interoceptive";
        default: return "unknown";
    }
}

const char* amyg_fep_safety_belief_name(amyg_fep_safety_belief_t belief) {
    switch (belief) {
        case AMYG_FEP_BELIEF_SAFE:      return "safe";
        case AMYG_FEP_BELIEF_UNCERTAIN: return "uncertain";
        case AMYG_FEP_BELIEF_UNSAFE:    return "unsafe";
        default: return "unknown";
    }
}
