/**
 * @file nimcp_omni_amygdala_bridge.c
 * @brief Implementation of Omnidirectional Inference to Amygdala Bridge
 */

#include "core/brain/subcortical/nimcp_omni_amygdala_bridge.h"
#include "cognitive/jepa/nimcp_jepa_bidirectional.h"
#include "cognitive/predictive/nimcp_predictive_hierarchy.h"
#include "cognitive/memory/nimcp_hopfield_memory.h"
#include "utils/thread/nimcp_thread.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"
#include "utils/exception/nimcp_exception_macros.h"

#include <string.h>
#include <math.h>

/* ============================================================================
 * Static Helpers
 * ============================================================================ */

static omni_threat_level_t compute_threat_level(float pe, float sensitivity) {
    float adjusted_pe = pe * sensitivity;

    if (adjusted_pe < 0.5f) {
        return OMNI_THREAT_NONE;
    } else if (adjusted_pe < 1.5f) {
        return OMNI_THREAT_LOW;
    } else if (adjusted_pe < 3.0f) {
        return OMNI_THREAT_MODERATE;
    } else if (adjusted_pe < 6.0f) {
        return OMNI_THREAT_HIGH;
    } else {
        return OMNI_THREAT_EXTREME;
    }
}

static omni_emotional_mode_t compute_emotional_mode(omni_threat_level_t threat,
                                                     float anxiety) {
    if (threat >= OMNI_THREAT_HIGH) {
        return OMNI_EMO_FEARFUL;
    }
    if (anxiety > 0.7f) {
        return OMNI_EMO_ANXIOUS;
    }
    if (threat == OMNI_THREAT_MODERATE) {
        return OMNI_EMO_VIGILANT;
    }
    if (threat == OMNI_THREAT_NONE && anxiety < 0.2f) {
        return OMNI_EMO_SAFE;
    }
    return OMNI_EMO_NEUTRAL;
}

static float compute_forward_bias(omni_threat_level_t threat) {
    switch (threat) {
        case OMNI_THREAT_NONE: return 0.0f;
        case OMNI_THREAT_LOW: return 0.2f;
        case OMNI_THREAT_MODERATE: return 0.5f;
        case OMNI_THREAT_HIGH: return 0.8f;
        case OMNI_THREAT_EXTREME: return 1.0f;
        default: return 0.0f;
    }
}

/* ============================================================================
 * Configuration API
 * ============================================================================ */

int omni_amygdala_default_config(omni_amygdala_config_t* config) {
    if (!config) return NIMCP_ERROR_INVALID_PARAM;

    memset(config, 0, sizeof(omni_amygdala_config_t));

    config->fear_pe_threshold = OMNI_AMYG_FEAR_PE_THRESHOLD;
    config->threat_sensitivity = OMNI_AMYG_DEFAULT_SENSITIVITY;
    config->anxiety_threshold = OMNI_AMYG_ANXIETY_SUPPRESS_THRESHOLD;

    config->max_forward_bias = 0.9f;
    config->max_precision_boost = 2.0f;
    config->allow_backward_suppression = true;

    config->enable_fear_conditioning = true;
    config->conditioning_rate = 0.1f;
    config->extinction_rate = 0.05f;

    config->enable_bio_async = true;
    config->enable_logging = false;

    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Lifecycle API
 * ============================================================================ */

omni_amygdala_bridge_t* omni_amygdala_bridge_create(
    const omni_amygdala_config_t* config) {

    omni_amygdala_bridge_t* bridge = nimcp_calloc(1, sizeof(omni_amygdala_bridge_t));
    if (!bridge) return NULL;

    if (config) {
        memcpy(&bridge->config, config, sizeof(omni_amygdala_config_t));
    } else {
        omni_amygdala_default_config(&bridge->config);
    }

    bridge->mutex = nimcp_mutex_create(NULL);
    if (!bridge->mutex) {
        nimcp_free(bridge);
        return NULL;
    }

    memset(&bridge->stats, 0, sizeof(omni_amygdala_stats_t));

    /* Initialize default emotional state */
    bridge->amygdala_effects.mode = OMNI_EMO_NEUTRAL;
    bridge->amygdala_effects.fear_level = 0.0f;
    bridge->amygdala_effects.anxiety_level = 0.0f;
    bridge->amygdala_effects.forward_bias = 0.0f;

    return bridge;
}

void omni_amygdala_bridge_destroy(omni_amygdala_bridge_t* bridge) {
    if (!bridge) return;

    if (bridge->omni_effects.threat_pattern) {
        nimcp_free(bridge->omni_effects.threat_pattern);
    }
    if (bridge->amygdala_effects.threat_priors) {
        nimcp_free(bridge->amygdala_effects.threat_priors);
    }

    if (bridge->mutex) {
        nimcp_mutex_free(bridge->mutex);
    }

    nimcp_free(bridge);
}

/* ============================================================================
 * Connection API
 * ============================================================================ */

int omni_amygdala_connect_jepa(omni_amygdala_bridge_t* bridge,
                                jepa_bidirectional_t* jepa) {
    if (!bridge) return NIMCP_ERROR_INVALID_PARAM;
    nimcp_mutex_lock(bridge->mutex);
    bridge->jepa = jepa;
    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_SUCCESS;
}

int omni_amygdala_connect_pred_hier(omni_amygdala_bridge_t* bridge,
                                     predictive_hierarchy_t* pred_hier) {
    if (!bridge) return NIMCP_ERROR_INVALID_PARAM;
    nimcp_mutex_lock(bridge->mutex);
    bridge->pred_hier = pred_hier;
    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_SUCCESS;
}

int omni_amygdala_connect_hopfield(omni_amygdala_bridge_t* bridge,
                                    hopfield_memory_t* hopfield) {
    if (!bridge) return NIMCP_ERROR_INVALID_PARAM;
    nimcp_mutex_lock(bridge->mutex);
    bridge->hopfield = hopfield;
    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_SUCCESS;
}

int omni_amygdala_connect_amygdala(omni_amygdala_bridge_t* bridge,
                                    amygdala_t* amygdala) {
    if (!bridge) return NIMCP_ERROR_INVALID_PARAM;
    nimcp_mutex_lock(bridge->mutex);
    bridge->amygdala = amygdala;
    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Update API
 * ============================================================================ */

int omni_amygdala_update(omni_amygdala_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_INVALID_PARAM;

    nimcp_mutex_lock(bridge->mutex);

    /* Get prediction error from connected systems */
    float pe = 0.0f;
    if (bridge->pred_hier) {
        float fe = pred_hier_compute_free_energy(bridge->pred_hier);
        pe = isnan(fe) ? 0.0f : fe;
    }

    /* Compute omni → amygdala effects */
    bridge->omni_effects.pe_magnitude = pe;
    bridge->omni_effects.threat_level =
        compute_threat_level(pe, bridge->config.threat_sensitivity);
    bridge->omni_effects.threat_confidence = 1.0f - expf(-pe);
    bridge->omni_effects.is_novel = (pe > bridge->config.fear_pe_threshold);

    /* Compute amygdala → omni effects */
    bridge->amygdala_effects.mode =
        compute_emotional_mode(bridge->omni_effects.threat_level,
                              bridge->amygdala_effects.anxiety_level);
    bridge->amygdala_effects.forward_bias =
        fminf(compute_forward_bias(bridge->omni_effects.threat_level),
              bridge->config.max_forward_bias);

    /* Update fear/anxiety levels */
    if (bridge->omni_effects.threat_level >= OMNI_THREAT_MODERATE) {
        bridge->amygdala_effects.fear_level =
            0.9f * bridge->amygdala_effects.fear_level +
            0.1f * (float)(bridge->omni_effects.threat_level) / 4.0f;
    } else {
        bridge->amygdala_effects.fear_level *= 0.95f;
    }

    bridge->amygdala_effects.anxiety_level =
        0.95f * bridge->amygdala_effects.anxiety_level +
        0.05f * bridge->amygdala_effects.fear_level;

    /* Suppression logic */
    bridge->amygdala_effects.suppress_backward =
        bridge->config.allow_backward_suppression &&
        (bridge->omni_effects.threat_level >= OMNI_THREAT_HIGH);
    bridge->amygdala_effects.suppress_lateral =
        (bridge->omni_effects.threat_level >= OMNI_THREAT_EXTREME);

    /* Precision boost for threats */
    bridge->amygdala_effects.precision_boost = 1.0f +
        (bridge->config.max_precision_boost - 1.0f) *
        (float)(bridge->omni_effects.threat_level) / 4.0f;

    /* Update statistics */
    bridge->stats.total_updates++;
    if (bridge->omni_effects.threat_level != OMNI_THREAT_NONE) {
        bridge->stats.threat_detections++;
    }
    if (bridge->amygdala_effects.fear_level > 0.5f) {
        bridge->stats.fear_activations++;
    }
    if (bridge->amygdala_effects.anxiety_level > bridge->config.anxiety_threshold) {
        bridge->stats.anxiety_episodes++;
    }
    if (bridge->amygdala_effects.suppress_backward) {
        bridge->stats.backward_suppressions++;
    }

    float n = (float)bridge->stats.total_updates;
    bridge->stats.avg_fear_level =
        (bridge->stats.avg_fear_level * (n - 1) + bridge->amygdala_effects.fear_level) / n;
    bridge->stats.avg_anxiety_level =
        (bridge->stats.avg_anxiety_level * (n - 1) + bridge->amygdala_effects.anxiety_level) / n;
    if ((float)bridge->omni_effects.threat_level > bridge->stats.max_threat_level) {
        bridge->stats.max_threat_level = (float)bridge->omni_effects.threat_level;
    }

    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_SUCCESS;
}

int omni_amygdala_apply_to_amygdala(omni_amygdala_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_INVALID_PARAM;
    return NIMCP_SUCCESS;
}

int omni_amygdala_apply_to_omni(omni_amygdala_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_INVALID_PARAM;
    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Threat Detection API
 * ============================================================================ */

omni_threat_level_t omni_amygdala_get_threat_level(
    const omni_amygdala_bridge_t* bridge) {
    if (!bridge) return OMNI_THREAT_NONE;
    return bridge->omni_effects.threat_level;
}

omni_emotional_mode_t omni_amygdala_get_emotional_mode(
    const omni_amygdala_bridge_t* bridge) {
    if (!bridge) return OMNI_EMO_NEUTRAL;
    return bridge->amygdala_effects.mode;
}

bool omni_amygdala_should_suppress_backward(
    const omni_amygdala_bridge_t* bridge) {
    if (!bridge) return false;
    return bridge->amygdala_effects.suppress_backward;
}

float omni_amygdala_get_forward_bias(const omni_amygdala_bridge_t* bridge) {
    if (!bridge) return 0.0f;
    return bridge->amygdala_effects.forward_bias;
}

/* ============================================================================
 * Fear Conditioning API
 * ============================================================================ */

int omni_amygdala_condition_fear(omni_amygdala_bridge_t* bridge,
                                  const float* pattern,
                                  uint32_t dim,
                                  float strength) {
    if (!bridge || !pattern || dim == 0) return NIMCP_ERROR_INVALID_PARAM;

    nimcp_mutex_lock(bridge->mutex);

    /* Store fear pattern in Hopfield if connected */
    if (bridge->hopfield && bridge->config.enable_fear_conditioning) {
        hopfield_memory_store_with_meta(bridge->hopfield, pattern, strength, NULL, NULL);
    }

    bridge->amygdala_effects.fear_level =
        fminf(1.0f, bridge->amygdala_effects.fear_level +
              strength * bridge->config.conditioning_rate);

    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_SUCCESS;
}

int omni_amygdala_extinguish_fear(omni_amygdala_bridge_t* bridge,
                                   const float* pattern,
                                   uint32_t dim) {
    if (!bridge || !pattern || dim == 0) return NIMCP_ERROR_INVALID_PARAM;

    nimcp_mutex_lock(bridge->mutex);

    bridge->amygdala_effects.fear_level *=
        (1.0f - bridge->config.extinction_rate);
    bridge->amygdala_effects.anxiety_level *=
        (1.0f - bridge->config.extinction_rate * 0.5f);

    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Query API
 * ============================================================================ */

int omni_amygdala_get_omni_effects(const omni_amygdala_bridge_t* bridge,
                                    omni_to_amygdala_effects_t* effects) {
    if (!bridge || !effects) return NIMCP_ERROR_INVALID_PARAM;
    nimcp_mutex_lock(((omni_amygdala_bridge_t*)bridge)->mutex);
    memcpy(effects, &bridge->omni_effects, sizeof(omni_to_amygdala_effects_t));
    nimcp_mutex_unlock(((omni_amygdala_bridge_t*)bridge)->mutex);
    return NIMCP_SUCCESS;
}

int omni_amygdala_get_amygdala_effects(const omni_amygdala_bridge_t* bridge,
                                        amygdala_to_omni_effects_t* effects) {
    if (!bridge || !effects) return NIMCP_ERROR_INVALID_PARAM;
    nimcp_mutex_lock(((omni_amygdala_bridge_t*)bridge)->mutex);
    memcpy(effects, &bridge->amygdala_effects, sizeof(amygdala_to_omni_effects_t));
    nimcp_mutex_unlock(((omni_amygdala_bridge_t*)bridge)->mutex);
    return NIMCP_SUCCESS;
}

int omni_amygdala_get_stats(const omni_amygdala_bridge_t* bridge,
                             omni_amygdala_stats_t* stats) {
    if (!bridge || !stats) return NIMCP_ERROR_INVALID_PARAM;
    nimcp_mutex_lock(((omni_amygdala_bridge_t*)bridge)->mutex);
    memcpy(stats, &bridge->stats, sizeof(omni_amygdala_stats_t));
    nimcp_mutex_unlock(((omni_amygdala_bridge_t*)bridge)->mutex);
    return NIMCP_SUCCESS;
}

int omni_amygdala_reset_stats(omni_amygdala_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_INVALID_PARAM;
    nimcp_mutex_lock(bridge->mutex);
    memset(&bridge->stats, 0, sizeof(omni_amygdala_stats_t));
    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Bio-Async Message Handlers
 * ============================================================================ */

static nimcp_error_t handle_amygdala_free_energy_report(
    const void* msg,
    size_t msg_size,
    nimcp_bio_promise_t response_promise,
    void* user_data)
{
    omni_amygdala_bridge_t* bridge = (omni_amygdala_bridge_t*)user_data;
    if (!bridge || !msg) return NIMCP_ERROR_INVALID_PARAM;

    omni_amygdala_update(bridge);

    (void)response_promise;
    (void)msg_size;
    return NIMCP_SUCCESS;
}

static nimcp_error_t handle_amygdala_error_propagate(
    const void* msg,
    size_t msg_size,
    nimcp_bio_promise_t response_promise,
    void* user_data)
{
    omni_amygdala_bridge_t* bridge = (omni_amygdala_bridge_t*)user_data;
    if (!bridge || !msg) return NIMCP_ERROR_INVALID_PARAM;

    /* Process error for emotional modulation */
    omni_amygdala_update(bridge);

    (void)response_promise;
    (void)msg_size;
    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Bio-Async API
 * ============================================================================ */

int omni_amygdala_connect_bio_async(omni_amygdala_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_INVALID_PARAM;
    if (bridge->bio_async_connected) return NIMCP_SUCCESS;

    bio_module_info_t info = {
        .module_id = BIO_MODULE_OMNI_AMYGDALA_BRIDGE,
        .module_name = "omni_amygdala_bridge",
        .inbox_capacity = 32,
        .user_data = bridge
    };

    bio_module_context_t ctx = bio_router_register_module(&info);
    if (!ctx) {
        return NIMCP_ERROR_OPERATION_FAILED;
    }

    bridge->bio_context = ctx;

    bio_router_register_handler(ctx, BIO_MSG_OMNI_FREE_ENERGY_REPORT,
                                 handle_amygdala_free_energy_report);
    bio_router_register_handler(ctx, BIO_MSG_PRED_HIER_ERROR_PROPAGATE,
                                 handle_amygdala_error_propagate);

    bridge->bio_async_connected = true;
    return NIMCP_SUCCESS;
}

int omni_amygdala_disconnect_bio_async(omni_amygdala_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_INVALID_PARAM;
    if (!bridge->bio_async_connected) return NIMCP_SUCCESS;

    if (bridge->bio_context) {
        bio_router_unregister_module(bridge->bio_context);
        bridge->bio_context = NULL;
    }

    bridge->bio_async_connected = false;
    return NIMCP_SUCCESS;
}

bool omni_amygdala_is_bio_async_connected(const omni_amygdala_bridge_t* bridge) {
    if (!bridge) return false;
    return bridge->bio_async_connected;
}

/* ============================================================================
 * String Conversion API
 * ============================================================================ */

const char* omni_amygdala_threat_to_string(omni_threat_level_t level) {
    switch (level) {
        case OMNI_THREAT_NONE: return "NONE";
        case OMNI_THREAT_LOW: return "LOW";
        case OMNI_THREAT_MODERATE: return "MODERATE";
        case OMNI_THREAT_HIGH: return "HIGH";
        case OMNI_THREAT_EXTREME: return "EXTREME";
        default: return "UNKNOWN";
    }
}

const char* omni_amygdala_mode_to_string(omni_emotional_mode_t mode) {
    switch (mode) {
        case OMNI_EMO_NEUTRAL: return "NEUTRAL";
        case OMNI_EMO_VIGILANT: return "VIGILANT";
        case OMNI_EMO_ANXIOUS: return "ANXIOUS";
        case OMNI_EMO_FEARFUL: return "FEARFUL";
        case OMNI_EMO_SAFE: return "SAFE";
        default: return "UNKNOWN";
    }
}

const char* omni_amygdala_fear_type_to_string(omni_fear_type_t type) {
    switch (type) {
        case OMNI_FEAR_NONE: return "NONE";
        case OMNI_FEAR_CONDITIONED: return "CONDITIONED";
        case OMNI_FEAR_CONTEXTUAL: return "CONTEXTUAL";
        case OMNI_FEAR_GENERALIZED: return "GENERALIZED";
        default: return "UNKNOWN";
    }
}
