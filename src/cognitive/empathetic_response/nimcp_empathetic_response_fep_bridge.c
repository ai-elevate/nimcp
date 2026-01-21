/**
 * @file nimcp_empathetic_response_fep_bridge.c
 * @brief Free Energy Principle - Empathetic Response Integration Bridge Implementation
 *
 * WHAT: Implements bidirectional integration between FEP and empathetic response system
 * WHY:  Empathy is simulating others' emotional states via shared generative models
 * HOW:  FEP models others' hidden states; empathy modulates social prediction precision
 */

#include "cognitive/empathetic_response/nimcp_empathetic_response_fep_bridge.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/validation/nimcp_common.h"
#include "utils/exception/nimcp_exception_macros.h"

#include <string.h>
#include <math.h>

#define LOG_MODULE "empathetic_response_fep_bridge"

/* ============================================================================
 * Default Configuration
 * ============================================================================ */

int empathetic_response_fep_default_config(empathetic_response_fep_config_t* config) {
    NIMCP_CHECK_THROW(config, NIMCP_ERROR_NULL_POINTER, "config is NULL");

    /* FEP -> Empathetic Response */
    config->pe_empathy_gain = 1.0f;
    config->precision_response_confidence = 1.0f;
    config->enable_pe_empathy_generation = true;
    config->enable_precision_confidence = true;

    /* Empathetic Response -> FEP */
    config->empathy_precision_modulation = 1.0f;
    config->response_uncertainty_gain = 1.0f;
    config->enable_empathy_precision = true;
    config->enable_response_uncertainty = true;

    /* Sensitivity */
    config->fe_sensitivity = 1.0f;
    config->emotion_sensitivity = 1.0f;

    return 0;
}

/* ============================================================================
 * Lifecycle API
 * ============================================================================ */

empathetic_response_fep_bridge_t* empathetic_response_fep_create(
    const empathetic_response_fep_config_t* config
) {
    empathetic_response_fep_bridge_t* bridge = nimcp_malloc(sizeof(empathetic_response_fep_bridge_t));
    if (!bridge) {
        NIMCP_LOGGING_ERROR("Failed to allocate empathetic response FEP bridge");
        return NULL;
    }

    memset(bridge, 0, sizeof(empathetic_response_fep_bridge_t));

    /* Set configuration */
    if (config) {
        bridge->config = *config;
    } else {
        empathetic_response_fep_default_config(&bridge->config);
    }

    /* Create mutex */
    bridge->base.mutex = nimcp_platform_mutex_create();
    if (!bridge->base.mutex) {
        NIMCP_LOGGING_ERROR("Failed to create mutex");
        nimcp_free(bridge);
        return NULL;
    }

    /* Initialize defaults */
    bridge->emotion_effects.intervention_threshold = 0.7f;

    NIMCP_LOGGING_INFO("Created empathetic response FEP bridge");
    return bridge;
}

void empathetic_response_fep_destroy(empathetic_response_fep_bridge_t* bridge) {
    if (!bridge) return;

    /* Disconnect bio-async if connected */
    if (bridge->base.bio_async_enabled) {
        empathetic_response_fep_disconnect_bio_async(bridge);
    }

    /* Destroy mutex */
    if (bridge->base.mutex) {
        nimcp_platform_mutex_destroy(bridge->base.mutex);
    }

    nimcp_free(bridge);
    NIMCP_LOGGING_INFO("Destroyed empathetic response FEP bridge");
}

/* ============================================================================
 * Connection API
 * ============================================================================ */

int empathetic_response_fep_connect_fep(
    empathetic_response_fep_bridge_t* bridge,
    fep_system_t* fep
) {
    NIMCP_CHECK_THROW(bridge && fep, NIMCP_ERROR_NULL_POINTER, "bridge or fep is NULL");

    nimcp_mutex_lock(bridge->base.mutex);
    bridge->fep_system = fep;
    nimcp_mutex_unlock(bridge->base.mutex);

    NIMCP_LOGGING_INFO("Connected FEP system to empathetic response bridge");
    return 0;
}

int empathetic_response_fep_connect_empathy(
    empathetic_response_fep_bridge_t* bridge,
    empathetic_response_engine_t empathy
) {
    NIMCP_CHECK_THROW(bridge, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

    nimcp_mutex_lock(bridge->base.mutex);
    bridge->empathetic_system = empathy;
    nimcp_mutex_unlock(bridge->base.mutex);

    NIMCP_LOGGING_INFO("Connected empathetic response system to FEP bridge");
    return 0;
}

int empathetic_response_fep_disconnect(empathetic_response_fep_bridge_t* bridge) {
    NIMCP_CHECK_THROW(bridge, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

    nimcp_mutex_lock(bridge->base.mutex);
    bridge->fep_system = NULL;
    memset(&bridge->empathetic_system, 0, sizeof(empathetic_response_engine_t));
    nimcp_mutex_unlock(bridge->base.mutex);

    NIMCP_LOGGING_INFO("Disconnected all systems from empathetic response FEP bridge");
    return 0;
}

/* ============================================================================
 * FEP -> Empathetic Response Direction
 * ============================================================================ */

int empathetic_response_fep_infer_user_state(
    empathetic_response_fep_bridge_t* bridge,
    float pe_magnitude
) {
    NIMCP_CHECK_THROW(bridge, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");
    if (!bridge->config.enable_pe_empathy_generation) return 0;

    nimcp_mutex_lock(bridge->base.mutex);

    /* Prediction error drives inference of user's emotional state
     * High PE from social signals -> infer user distress
     * Social PE = mismatch between expected and observed behavior
     */
    float distress_inference = fabsf(pe_magnitude) * bridge->config.pe_empathy_gain;

    /* Clamp to [0, 1] */
    if (distress_inference > 1.0f) distress_inference = 1.0f;

    bridge->fep_effects.inferred_user_distress = distress_inference;

    /* Confidence increases with PE magnitude (more signal = more confidence) */
    float confidence = 0.5f + (fabsf(pe_magnitude) * bridge->config.precision_response_confidence * 0.5f);
    if (confidence > 1.0f) confidence = 1.0f;
    bridge->fep_effects.response_confidence = confidence;

    /* Activate empathy when distress detected */
    bridge->fep_effects.empathy_active = (distress_inference > 0.3f);

    /* Update state */
    bridge->state.current_prediction_error = pe_magnitude;
    bridge->state.current_empathy_level = distress_inference;

    /* Check for crisis detection */
    if (distress_inference > bridge->emotion_effects.intervention_threshold) {
        bridge->state.crisis_detected = true;
        bridge->stats.crisis_detection_events++;
        NIMCP_LOGGING_WARN("Crisis detected: distress level = %f", distress_inference);
    } else {
        bridge->state.crisis_detected = false;
    }

    /* Update stats */
    bridge->stats.empathy_generation_events++;
    bridge->stats.avg_empathy_level =
        (bridge->stats.avg_empathy_level * 0.9f) + (distress_inference * 0.1f);

    nimcp_mutex_unlock(bridge->base.mutex);

    NIMCP_LOGGING_DEBUG("Inferred user distress: %f, confidence: %f",
                        distress_inference, confidence);
    return 0;
}

int empathetic_response_fep_modulate_social_precision(
    empathetic_response_fep_bridge_t* bridge
) {
    NIMCP_CHECK_THROW(bridge, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");
    if (!bridge->config.enable_empathy_precision) return 0;

    nimcp_mutex_lock(bridge->base.mutex);

    /* High empathy level increases precision on social signals */
    float empathy_level = bridge->state.current_empathy_level;
    float social_precision = 1.0f + (empathy_level * bridge->config.empathy_precision_modulation);

    bridge->emotion_effects.social_precision_weight = social_precision;

    /* Response uncertainty decreases with empathy level */
    float uncertainty = 1.0f - (empathy_level * 0.5f);
    if (uncertainty < 0.1f) uncertainty = 0.1f;
    bridge->emotion_effects.response_uncertainty = uncertainty;

    nimcp_mutex_unlock(bridge->base.mutex);

    NIMCP_LOGGING_DEBUG("Modulated social precision: %f, uncertainty: %f",
                        social_precision, uncertainty);
    return 0;
}

/* ============================================================================
 * Update API
 * ============================================================================ */

int empathetic_response_fep_update(
    empathetic_response_fep_bridge_t* bridge,
    uint64_t delta_ms
) {
    NIMCP_CHECK_THROW(bridge, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

    /* Apply social precision modulation */
    empathetic_response_fep_modulate_social_precision(bridge);

    nimcp_mutex_lock(bridge->base.mutex);

    /* Decay empathy level over time if not reinforced */
    float decay = 0.995f;
    bridge->state.current_empathy_level *= decay;
    if (bridge->state.current_empathy_level < 0.01f) {
        bridge->state.crisis_detected = false;
    }

    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

/* ============================================================================
 * Query API
 * ============================================================================ */

int empathetic_response_fep_get_state(
    const empathetic_response_fep_bridge_t* bridge,
    empathetic_response_fep_state_t* state
) {
    NIMCP_CHECK_THROW(bridge && state, NIMCP_ERROR_NULL_POINTER, "bridge or state is NULL");

    nimcp_mutex_lock(bridge->base.mutex);
    *state = bridge->state;
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

int empathetic_response_fep_get_stats(
    const empathetic_response_fep_bridge_t* bridge,
    empathetic_response_fep_stats_t* stats
) {
    NIMCP_CHECK_THROW(bridge && stats, NIMCP_ERROR_NULL_POINTER, "bridge or stats is NULL");

    nimcp_mutex_lock(bridge->base.mutex);
    *stats = bridge->stats;
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

/* ============================================================================
 * Bio-Async Integration
 * ============================================================================ */

int empathetic_response_fep_connect_bio_async(
    empathetic_response_fep_bridge_t* bridge
) {
    NIMCP_CHECK_THROW(bridge, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");
    if (bridge->base.bio_async_enabled) return 0;

    bio_module_info_t info = {
        .module_id = BIO_MODULE_FEP_EMPATHETIC_RESPONSE_BRIDGE,
        .module_name = "empathetic_response_fep_bridge",
        .inbox_capacity = 32,
        .user_data = bridge
    };

    bridge->base.bio_ctx = bio_router_register_module(&info);
    if (bridge->base.bio_ctx) {
        bridge->base.bio_async_enabled = true;
        NIMCP_LOGGING_INFO("Connected to bio-async router");
    } else {
        NIMCP_LOGGING_WARN("Bio-async router not available");
    }

    return 0;
}

int empathetic_response_fep_disconnect_bio_async(
    empathetic_response_fep_bridge_t* bridge
) {
    NIMCP_CHECK_THROW(bridge, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");
    if (!bridge->base.bio_async_enabled) return 0;

    if (bridge->base.bio_ctx) {
        bio_router_unregister_module(bridge->base.bio_ctx);
        bridge->base.bio_ctx = NULL;
    }

    bridge->base.bio_async_enabled = false;
    NIMCP_LOGGING_INFO("Disconnected from bio-async router");

    return 0;
}

bool empathetic_response_fep_is_bio_async_connected(
    const empathetic_response_fep_bridge_t* bridge
) {
    return bridge ? bridge->base.bio_async_enabled : false;
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

int empathetic_response_fep_bridge_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;

    const kg_entity_t* self = kg_reader_get_entity(kg, "Empathetic_Response_FEP_Bridge");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            (void)self->observations[i];
        }
    }

    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Empathetic_Response_FEP_Bridge");
    if (connections) {
        kg_relation_list_destroy(connections);
    }

    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Empathetic_Response_FEP_Bridge");
    if (incoming) {
        kg_relation_list_destroy(incoming);
    }

    return self ? 1 : 0;
}
