/**
 * @file nimcp_jepa_imagination_bridge.c
 * @brief JEPA-Imagination Bidirectional Bridge Implementation
 * @version 1.0.0
 * @date 2026-01-03
 */

#include "cognitive/imagination/nimcp_jepa_imagination_bridge.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "cognitive/imagination/nimcp_imagination_engine.h"
#include "cognitive/jepa/nimcp_jepa_predictor.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/time/nimcp_time.h"

#include <string.h>
#include <math.h>

/* Local logging macros */
#define NIMCP_LOG_INFO(...)  LOG_INFO(__VA_ARGS__)
#define NIMCP_LOG_DEBUG(...) LOG_DEBUG(__VA_ARGS__)
#define NIMCP_LOG_ERROR(...) LOG_ERROR(__VA_ARGS__)

/* ============================================================================
 * CONFIGURATION
 * ============================================================================ */

int jepa_imagination_default_config(jepa_imagination_config_t* config) {
    if (!config) return -1;

    config->confidence_threshold = JEPA_IMAG_DEFAULT_CONFIDENCE_THRESHOLD;
    config->max_prediction_horizon = 10;
    config->prediction_decay = 0.9f;

    config->learning_rate = JEPA_IMAG_DEFAULT_LEARNING_RATE;
    config->novelty_bonus = 0.1f;
    config->enable_counterfactual_training = true;

    config->enable_prediction_constraints = true;
    config->constraint_softness = 0.3f;

    config->update_interval_ms = 16.0f;  /* ~60 Hz */
    config->enable_bio_async = true;

    return 0;
}

int jepa_imagination_validate_config(const jepa_imagination_config_t* config) {
    if (!config) return -1;

    if (config->confidence_threshold < 0.0f || config->confidence_threshold > 1.0f) {
        return -1;
    }
    if (config->max_prediction_horizon == 0 || config->max_prediction_horizon > 100) {
        return -1;
    }
    if (config->prediction_decay < 0.0f || config->prediction_decay > 1.0f) {
        return -1;
    }
    if (config->learning_rate < 0.0f || config->learning_rate > 1.0f) {
        return -1;
    }
    if (config->novelty_bonus < 0.0f || config->novelty_bonus > 1.0f) {
        return -1;
    }
    if (config->constraint_softness < 0.0f || config->constraint_softness > 1.0f) {
        return -1;
    }
    if (config->update_interval_ms < 1.0f) {
        return -1;
    }

    return 0;
}

/* ============================================================================
 * LIFECYCLE
 * ============================================================================ */

jepa_imagination_bridge_t* jepa_imagination_bridge_create(
    const jepa_imagination_config_t* config)
{
    jepa_imagination_bridge_t* bridge = nimcp_calloc(
        1, sizeof(jepa_imagination_bridge_t));
    if (!bridge) {
        NIMCP_LOG_ERROR("Failed to allocate JEPA-imagination bridge");
        return NULL;
    }

    /* Initialize base */
    bridge->base.module_id = BIO_MODULE_IMAGINATION_JEPA;
    bridge->base.module_name = "jepa_imagination_bridge";
    bridge->base.system_a_connected = false;
    bridge->base.system_b_connected = false;
    bridge->base.bridge_active = false;

    /* Create mutex */
    bridge->base.mutex = nimcp_mutex_create(NULL);
    if (!bridge->base.mutex) {
        NIMCP_LOG_ERROR("Failed to create bridge mutex");
        nimcp_free(bridge);
        return NULL;
    }

    /* Apply configuration */
    if (config) {
        if (jepa_imagination_validate_config(config) != 0) {
            NIMCP_LOG_ERROR("Invalid bridge configuration");
            nimcp_mutex_free(bridge->base.mutex);
            nimcp_free(bridge);
            return NULL;
        }
        bridge->config = *config;
    } else {
        jepa_imagination_default_config(&bridge->config);
    }

    /* Initialize effects */
    memset(&bridge->jepa_to_imag, 0, sizeof(bridge->jepa_to_imag));
    memset(&bridge->imag_to_jepa, 0, sizeof(bridge->imag_to_jepa));

    /* Initialize state */
    bridge->num_active_counterfactuals = 0;
    bridge->prediction_request_pending = false;
    bridge->pending_context = NULL;

    /* Initialize statistics */
    memset(&bridge->stats, 0, sizeof(bridge->stats));

    /* Initialize timing */
    bridge->last_update_time_ms = nimcp_time_get_ms();

    NIMCP_LOG_INFO("Created JEPA-imagination bridge");
    return bridge;
}

void jepa_imagination_bridge_destroy(jepa_imagination_bridge_t* bridge) {
    if (!bridge) return;

    /* Disconnect bio-async if connected */
    if (bridge->base.bio_async_enabled) {
        jepa_imagination_disconnect_bio_async(bridge);
    }

    /* Free pending context */
    if (bridge->pending_context) {
        nimcp_tensor_destroy(bridge->pending_context);
    }

    /* Free predicted latents */
    if (bridge->jepa_to_imag.predicted_latents) {
        nimcp_tensor_destroy(bridge->jepa_to_imag.predicted_latents);
    }

    /* Free outcome embedding */
    if (bridge->jepa_to_imag.outcome_embedding) {
        nimcp_tensor_destroy(bridge->jepa_to_imag.outcome_embedding);
    }

    /* Free target/context embeddings */
    if (bridge->imag_to_jepa.target_embedding) {
        nimcp_tensor_destroy(bridge->imag_to_jepa.target_embedding);
    }
    if (bridge->imag_to_jepa.context_embedding) {
        nimcp_tensor_destroy(bridge->imag_to_jepa.context_embedding);
    }

    /* Destroy mutex */
    if (bridge->base.mutex) {
        nimcp_mutex_free(bridge->base.mutex);
    }

    nimcp_free(bridge);
    NIMCP_LOG_INFO("Destroyed JEPA-imagination bridge");
}

int jepa_imagination_reset(jepa_imagination_bridge_t* bridge) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->base.mutex);

    /* Clear effects */
    memset(&bridge->jepa_to_imag, 0, sizeof(bridge->jepa_to_imag));
    memset(&bridge->imag_to_jepa, 0, sizeof(bridge->imag_to_jepa));

    /* Clear state */
    bridge->num_active_counterfactuals = 0;
    bridge->prediction_request_pending = false;

    /* Free and clear tensors */
    if (bridge->pending_context) {
        nimcp_tensor_destroy(bridge->pending_context);
        bridge->pending_context = NULL;
    }

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

/* ============================================================================
 * CONNECTION
 * ============================================================================ */

int jepa_imagination_connect_jepa(
    jepa_imagination_bridge_t* bridge,
    struct jepa_predictor* jepa)
{
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->base.mutex);

    bridge->jepa = jepa;
    bridge->base.system_a = jepa;
    bridge->base.system_a_connected = (jepa != NULL);
    bridge->base.bridge_active = bridge->base.system_a_connected &&
                                  bridge->base.system_b_connected;

    nimcp_mutex_unlock(bridge->base.mutex);

    NIMCP_LOG_INFO("JEPA %s to bridge",
                   jepa ? "connected" : "disconnected");
    return 0;
}

int jepa_imagination_connect_imagination(
    jepa_imagination_bridge_t* bridge,
    struct imagination_engine* imagination)
{
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->base.mutex);

    bridge->imagination = imagination;
    bridge->base.system_b = imagination;
    bridge->base.system_b_connected = (imagination != NULL);
    bridge->base.bridge_active = bridge->base.system_a_connected &&
                                  bridge->base.system_b_connected;

    nimcp_mutex_unlock(bridge->base.mutex);

    NIMCP_LOG_INFO("Imagination engine %s to bridge",
                   imagination ? "connected" : "disconnected");
    return 0;
}

int jepa_imagination_disconnect_jepa(jepa_imagination_bridge_t* bridge) {
    return jepa_imagination_connect_jepa(bridge, NULL);
}

int jepa_imagination_disconnect_imagination(jepa_imagination_bridge_t* bridge) {
    return jepa_imagination_connect_imagination(bridge, NULL);
}

bool jepa_imagination_is_connected(const jepa_imagination_bridge_t* bridge) {
    if (!bridge) return false;
    return bridge->base.bridge_active;
}

/* ============================================================================
 * UPDATE
 * ============================================================================ */

int jepa_imagination_update(
    jepa_imagination_bridge_t* bridge,
    float delta_time_ms)
{
    if (!bridge) return -1;
    if (!bridge->base.bridge_active) return 0;  /* Nothing to do */

    (void)delta_time_ms;  /* Used for rate limiting below */

    nimcp_mutex_lock(bridge->base.mutex);

    /* Check update interval */
    uint64_t now = nimcp_time_get_ms();
    float elapsed = (float)(now - bridge->last_update_time_ms);
    if (elapsed < bridge->config.update_interval_ms) {
        nimcp_mutex_unlock(bridge->base.mutex);
        return 0;
    }
    bridge->last_update_time_ms = now;

    /* Compute effects in both directions */
    jepa_imagination_compute_jepa_effects(bridge);
    jepa_imagination_compute_imag_effects(bridge);

    /* Apply effects */
    jepa_imagination_apply_effects(bridge);

    /* Update statistics */
    bridge->stats.total_updates++;
    bridge->base.total_updates++;

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int jepa_imagination_compute_jepa_effects(jepa_imagination_bridge_t* bridge) {
    if (!bridge || !bridge->jepa) return -1;

    /* Query JEPA for current prediction state */
    /* In full implementation, this would call JEPA predictor APIs */

    /* Default: low-level effects when no active prediction */
    bridge->jepa_to_imag.prediction_confidence = 0.0f;
    bridge->jepa_to_imag.world_model_coherence = 0.0f;
    bridge->jepa_to_imag.counterfactual_plausibility = 0.0f;
    bridge->jepa_to_imag.num_predictions = 0;
    bridge->jepa_to_imag.action_outcome_likelihood = 0.0f;
    bridge->jepa_to_imag.constrain_to_predictions = false;

    /* If there's a pending prediction request, process it */
    if (bridge->prediction_request_pending && bridge->pending_context) {
        /* Would call jepa_predict() here */
        /* For now, mark request as processed */
        bridge->prediction_request_pending = false;
        bridge->stats.predictions_generated++;
    }

    return 0;
}

int jepa_imagination_compute_imag_effects(jepa_imagination_bridge_t* bridge) {
    if (!bridge || !bridge->imagination) return -1;

    /* Query imagination engine for current state */
    /* In full implementation, this would call imagination engine APIs */

    /* Default: no training signals */
    bridge->imag_to_jepa.learning_strength = 0.0f;
    bridge->imag_to_jepa.prediction_error = 0.0f;
    bridge->imag_to_jepa.update_world_model = false;
    bridge->imag_to_jepa.scenario_id = 0;
    bridge->imag_to_jepa.novelty_score = 0.0f;
    bridge->imag_to_jepa.exploration_value = 0.0f;
    bridge->imag_to_jepa.is_counterfactual = false;
    bridge->imag_to_jepa.counterfactual_divergence = 0.0f;

    return 0;
}

int jepa_imagination_apply_effects(jepa_imagination_bridge_t* bridge) {
    if (!bridge) return -1;

    /* Apply JEPA effects to imagination */
    if (bridge->imagination && bridge->jepa_to_imag.num_predictions > 0) {
        /* Would call imagination APIs to inject prediction constraints */
        bridge->stats.predictions_used += bridge->jepa_to_imag.num_predictions;
    }

    /* Apply imagination effects to JEPA */
    if (bridge->jepa && bridge->imag_to_jepa.learning_strength > 0.0f) {
        /* Would call JEPA APIs to provide training signal */
        if (bridge->imag_to_jepa.update_world_model) {
            bridge->stats.world_model_updates++;
        }
        bridge->stats.training_signals_sent++;
    }

    /* Track counterfactual stats */
    if (bridge->imag_to_jepa.is_counterfactual) {
        bridge->stats.counterfactuals_simulated++;
    }

    return 0;
}

/* ============================================================================
 * JEPA-IMAGINATION INTEGRATION
 * ============================================================================ */

uint32_t jepa_imagination_request_predicted_imagination(
    jepa_imagination_bridge_t* bridge,
    const nimcp_tensor_t* context,
    struct imagination_goal* goal)
{
    if (!bridge || !context || !goal) return 0;
    if (!bridge->base.bridge_active) return 0;

    nimcp_mutex_lock(bridge->base.mutex);

    /* Store the context for processing */
    if (bridge->pending_context) {
        nimcp_tensor_destroy(bridge->pending_context);
    }
    bridge->pending_context = nimcp_tensor_clone(context);
    bridge->prediction_request_pending = true;

    /* In full implementation, would:
     * 1. Get latent predictions from JEPA
     * 2. Pass predictions to imagination as constraints
     * 3. Start imagination scenario with goal
     */

    /* For now, return a placeholder scenario ID */
    uint32_t scenario_id = 1;  /* Would get from imagination_begin_scenario() */

    nimcp_mutex_unlock(bridge->base.mutex);

    NIMCP_LOG_DEBUG("Requested prediction-constrained imagination, scenario %u", scenario_id);
    return scenario_id;
}

uint32_t jepa_imagination_request_counterfactual(
    jepa_imagination_bridge_t* bridge,
    const nimcp_tensor_t* context,
    const nimcp_tensor_t* action)
{
    if (!bridge || !context || !action) return 0;
    if (!bridge->base.bridge_active) return 0;

    nimcp_mutex_lock(bridge->base.mutex);

    /* In full implementation, would:
     * 1. Get JEPA prediction for actual action
     * 2. Start imagination with alternative action
     * 3. Track divergence from predicted outcome
     */

    uint32_t scenario_id = 2;  /* Placeholder */

    if (bridge->num_active_counterfactuals < JEPA_IMAG_MAX_COUNTERFACTUALS) {
        bridge->active_counterfactuals[bridge->num_active_counterfactuals++] = scenario_id;
    }

    bridge->stats.counterfactuals_simulated++;

    nimcp_mutex_unlock(bridge->base.mutex);

    NIMCP_LOG_DEBUG("Requested counterfactual simulation, scenario %u", scenario_id);
    return scenario_id;
}

int jepa_imagination_query_world_model(
    jepa_imagination_bridge_t* bridge,
    const nimcp_tensor_t* context,
    const nimcp_tensor_t* action,
    nimcp_tensor_t** outcome)
{
    if (!bridge || !context || !action || !outcome) return -1;
    if (!bridge->jepa) return -1;

    nimcp_mutex_lock(bridge->base.mutex);

    /* In full implementation, would call JEPA predictor to get outcome */
    /* For now, return NULL outcome */
    *outcome = NULL;

    nimcp_mutex_unlock(bridge->base.mutex);

    NIMCP_LOG_DEBUG("Queried world model for action outcome");
    return 0;
}

int jepa_imagination_provide_training_signal(
    jepa_imagination_bridge_t* bridge,
    const struct imagination_scenario* scenario,
    float emotional_weight)
{
    if (!bridge || !scenario) return -1;
    if (!bridge->jepa) return -1;

    nimcp_mutex_lock(bridge->base.mutex);

    /* Set up training signal */
    bridge->imag_to_jepa.learning_strength = emotional_weight * bridge->config.learning_rate;
    bridge->imag_to_jepa.update_world_model = true;

    /* In full implementation, would:
     * 1. Extract latent state from scenario
     * 2. Call JEPA training API
     */

    bridge->stats.training_signals_sent++;

    nimcp_mutex_unlock(bridge->base.mutex);

    NIMCP_LOG_DEBUG("Provided training signal with weight %.2f", emotional_weight);
    return 0;
}

int jepa_imagination_get_jepa_effects(
    const jepa_imagination_bridge_t* bridge,
    jepa_to_imagination_effects_t* effects)
{
    if (!bridge || !effects) return -1;

    *effects = bridge->jepa_to_imag;
    return 0;
}

int jepa_imagination_get_imagination_effects(
    const jepa_imagination_bridge_t* bridge,
    imagination_to_jepa_effects_t* effects)
{
    if (!bridge || !effects) return -1;

    *effects = bridge->imag_to_jepa;
    return 0;
}

/* ============================================================================
 * QUERY
 * ============================================================================ */

int jepa_imagination_get_stats(
    const jepa_imagination_bridge_t* bridge,
    jepa_imagination_stats_t* stats)
{
    if (!bridge || !stats) return -1;

    *stats = bridge->stats;
    return 0;
}

int jepa_imagination_reset_stats(jepa_imagination_bridge_t* bridge) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->base.mutex);
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

uint32_t jepa_imagination_get_counterfactual_count(
    const jepa_imagination_bridge_t* bridge)
{
    if (!bridge) return 0;
    return bridge->num_active_counterfactuals;
}

/* ============================================================================
 * BIO-ASYNC
 * ============================================================================ */

int jepa_imagination_connect_bio_async(jepa_imagination_bridge_t* bridge) {
    if (!bridge) return -1;
    if (bridge->base.bio_async_enabled) return 0;  /* Already connected */

    /* Use bridge base helper if available */
    int result = bridge_base_connect_bio_async(&bridge->base);
    if (result == 0) {
        NIMCP_LOG_INFO("JEPA-imagination bridge connected to bio-async");
    }

    return result;
}

int jepa_imagination_disconnect_bio_async(jepa_imagination_bridge_t* bridge) {
    if (!bridge) return -1;
    if (!bridge->base.bio_async_enabled) return 0;  /* Already disconnected */

    int result = bridge_base_disconnect_bio_async(&bridge->base);
    if (result == 0) {
        NIMCP_LOG_INFO("JEPA-imagination bridge disconnected from bio-async");
    }

    return result;
}

bool jepa_imagination_is_bio_async_connected(
    const jepa_imagination_bridge_t* bridge)
{
    if (!bridge) return false;
    return bridge->base.bio_async_enabled;
}

int jepa_imagination_process_messages(jepa_imagination_bridge_t* bridge) {
    if (!bridge) return -1;
    if (!bridge->base.bio_async_enabled) return 0;

    /* Process pending bio-async messages */
    /* In full implementation, would handle:
     * - BIO_MSG_JEPA_PREDICTION_REQUEST
     * - BIO_MSG_JEPA_PREDICTION_RESPONSE
     * - BIO_MSG_IMAGINATION_TRAINING_SIGNAL
     * - BIO_MSG_COUNTERFACTUAL_REQUEST
     */

    return 0;
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

/**
 * WHAT: Query knowledge graph for JEPA Imagination Bridge self-knowledge
 * WHY:  Enable self-awareness about module's role and connections
 * HOW:  Query KG for entity observations and relations
 */
int jepa_imagination_bridge_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;
    const kg_entity_t* self = kg_reader_get_entity(kg, "JEPA_Imagination_Bridge");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            NIMCP_LOG_DEBUG("JEPA Imagination Bridge self-knowledge: %s", self->observations[i]);
        }
    }
    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "JEPA_Imagination_Bridge");
    if (connections) { kg_relation_list_destroy(connections); }
    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "JEPA_Imagination_Bridge");
    if (incoming) { kg_relation_list_destroy(incoming); }
    return self ? 1 : 0;
}
