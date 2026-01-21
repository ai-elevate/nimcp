/**
 * @file nimcp_omni_immune_bridge.c
 * @brief Implementation of Omnidirectional Inference to Immune System Bridge
 */

#include "cognitive/immune/nimcp_omni_immune_bridge.h"
#include "cognitive/jepa/nimcp_jepa_bidirectional.h"
#include "cognitive/memory/nimcp_hopfield_memory.h"
#include "cognitive/predictive/nimcp_predictive_hierarchy.h"
#include "cognitive/memory/nimcp_temporal_replay.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/logging/nimcp_logging.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"
#include "api/nimcp_api_exception.h"
#include "utils/exception/nimcp_exception_macros.h"

#include <string.h>
#include <math.h>

/* ============================================================================
 * Static Helpers
 * ============================================================================ */

static float compute_aggregated_pe(omni_immune_bridge_t* bridge) {
    float total_pe = 0.0f;
    float total_weight = 0.0f;

    if (bridge->jepa) {
        jepa_bidir_stats_t stats;
        if (jepa_bidirectional_get_stats(bridge->jepa, &stats) == NIMCP_SUCCESS) {
            /* Use free energy as a proxy for prediction error */
            float fe = stats.avg_free_energy;
            total_pe += fe * bridge->config.jepa_forward_weight;
            total_pe += fe * bridge->config.jepa_backward_weight;
            total_weight += bridge->config.jepa_forward_weight + bridge->config.jepa_backward_weight;
        }
    }

    if (bridge->hopfield) {
        hopfield_stats_t stats;
        if (hopfield_memory_get_stats(bridge->hopfield, &stats) == NIMCP_SUCCESS) {
            float hopfield_error = 1.0f - stats.avg_similarity;
            total_pe += hopfield_error * bridge->config.hopfield_weight;
            total_weight += bridge->config.hopfield_weight;
        }
    }

    if (bridge->pred_hier) {
        float hier_fe = pred_hier_compute_free_energy(bridge->pred_hier);
        if (!isnan(hier_fe)) {
            total_pe += hier_fe * bridge->config.pred_hier_weight;
            total_weight += bridge->config.pred_hier_weight;
        }
    }

    if (bridge->replay) {
        replay_stats_t stats;
        if (temporal_replay_get_stats(bridge->replay, &stats) == NIMCP_SUCCESS) {
            float replay_error = stats.avg_priority;
            total_pe += replay_error * bridge->config.replay_weight;
            total_weight += bridge->config.replay_weight;
        }
    }

    return total_weight > 0.0f ? total_pe / total_weight : 0.0f;
}

static omni_immune_response_t compute_response_level(float pe, float fe,
                                                      const omni_immune_config_t* config) {
    if (pe < config->pe_threshold * 0.5f) {
        return OMNI_IMMUNE_NONE;
    } else if (pe < config->pe_threshold) {
        return OMNI_IMMUNE_LOCAL;
    } else if (fe < config->fe_threshold) {
        return OMNI_IMMUNE_REGIONAL;
    } else if (fe < config->fe_threshold * 2.0f) {
        return OMNI_IMMUNE_SYSTEMIC;
    } else {
        return OMNI_IMMUNE_MEMORY;
    }
}

static omni_immune_source_t find_dominant_source(omni_immune_bridge_t* bridge) {
    float max_error = 0.0f;
    omni_immune_source_t source = OMNI_SOURCE_AGGREGATED;

    if (bridge->jepa) {
        jepa_bidir_stats_t stats;
        if (jepa_bidirectional_get_stats(bridge->jepa, &stats) == NIMCP_SUCCESS) {
            /* Use free energy as proxy for error */
            float fe = stats.avg_free_energy;
            if (fe > max_error) {
                max_error = fe;
                source = OMNI_SOURCE_JEPA_FORWARD;
            }
        }
    }

    if (bridge->pred_hier) {
        float hier_fe = pred_hier_compute_free_energy(bridge->pred_hier);
        if (!isnan(hier_fe) && hier_fe > max_error) {
            max_error = hier_fe;
            source = OMNI_SOURCE_PRED_HIERARCHY;
        }
    }

    return source;
}

/* ============================================================================
 * Configuration API
 * ============================================================================ */

int omni_immune_default_config(omni_immune_config_t* config) {
    if (!config) return NIMCP_ERROR_INVALID_PARAM;

    memset(config, 0, sizeof(omni_immune_config_t));

    config->pe_threshold = OMNI_IMMUNE_DEFAULT_PE_THRESHOLD;
    config->fe_threshold = OMNI_IMMUNE_DEFAULT_FE_THRESHOLD;
    config->inflammation_scale = OMNI_IMMUNE_DEFAULT_INFLAMMATION_SCALE;

    config->jepa_forward_weight = 1.0f;
    config->jepa_backward_weight = 1.2f;
    config->hopfield_weight = 0.8f;
    config->pred_hier_weight = 1.5f;
    config->replay_weight = 0.6f;

    config->enable_microglia = true;
    config->enable_memory_immune = true;
    config->enable_precision_feedback = true;

    config->enable_bio_async = true;
    config->enable_logging = false;

    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Lifecycle API
 * ============================================================================ */

omni_immune_bridge_t* omni_immune_bridge_create(const omni_immune_config_t* config) {
    omni_immune_bridge_t* bridge = nimcp_calloc(1, sizeof(omni_immune_bridge_t));
    if (!bridge) return NULL;

    if (config) {
        memcpy(&bridge->config, config, sizeof(omni_immune_config_t));
    } else {
        omni_immune_default_config(&bridge->config);
    }

    bridge->mutex = nimcp_mutex_create(NULL);
    if (!bridge->mutex) {
        nimcp_free(bridge);
        return NULL;
    }

    memset(&bridge->stats, 0, sizeof(omni_immune_stats_t));

    return bridge;
}

void omni_immune_bridge_destroy(omni_immune_bridge_t* bridge) {
    if (!bridge) return;

    if (bridge->mutex) {
        nimcp_mutex_free(bridge->mutex);
    }

    nimcp_free(bridge);
}

/* ============================================================================
 * Connection API
 * ============================================================================ */

int omni_immune_connect_jepa(omni_immune_bridge_t* bridge,
                              jepa_bidirectional_t* jepa) {
    if (!bridge) return NIMCP_ERROR_INVALID_PARAM;
    nimcp_mutex_lock(bridge->mutex);
    bridge->jepa = jepa;
    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_SUCCESS;
}

int omni_immune_connect_hopfield(omni_immune_bridge_t* bridge,
                                  hopfield_memory_t* hopfield) {
    if (!bridge) return NIMCP_ERROR_INVALID_PARAM;
    nimcp_mutex_lock(bridge->mutex);
    bridge->hopfield = hopfield;
    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_SUCCESS;
}

int omni_immune_connect_pred_hier(omni_immune_bridge_t* bridge,
                                   predictive_hierarchy_t* pred_hier) {
    if (!bridge) return NIMCP_ERROR_INVALID_PARAM;
    nimcp_mutex_lock(bridge->mutex);
    bridge->pred_hier = pred_hier;
    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_SUCCESS;
}

int omni_immune_connect_replay(omni_immune_bridge_t* bridge,
                                temporal_replay_t* replay) {
    if (!bridge) return NIMCP_ERROR_INVALID_PARAM;
    nimcp_mutex_lock(bridge->mutex);
    bridge->replay = replay;
    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_SUCCESS;
}

int omni_immune_connect_immune(omni_immune_bridge_t* bridge,
                                brain_immune_system_t* immune) {
    if (!bridge) return NIMCP_ERROR_INVALID_PARAM;
    nimcp_mutex_lock(bridge->mutex);
    bridge->immune = immune;
    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Update API
 * ============================================================================ */

int omni_immune_update(omni_immune_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_INVALID_PARAM;

    nimcp_mutex_lock(bridge->mutex);

    /* Compute omni → immune effects */
    float pe_magnitude = compute_aggregated_pe(bridge);
    float free_energy = 0.0f;
    if (bridge->pred_hier) {
        free_energy = pred_hier_compute_free_energy(bridge->pred_hier);
        if (isnan(free_energy)) free_energy = 0.0f;
    }

    bridge->omni_effects.pe_magnitude = pe_magnitude;
    bridge->omni_effects.free_energy = free_energy;
    bridge->omni_effects.inflammation_signal =
        fminf(pe_magnitude * bridge->config.inflammation_scale, OMNI_IMMUNE_MAX_PE_INFLAMMATION);
    bridge->omni_effects.source = find_dominant_source(bridge);
    bridge->omni_effects.response = compute_response_level(pe_magnitude, free_energy,
                                                           &bridge->config);

    if (bridge->pred_hier) {
        bridge->omni_effects.affected_levels = pred_hier_num_levels(bridge->pred_hier);
    }

    /* Compute immune → omni effects */
    bridge->immune_effects.precision_modulation = 1.0f;
    bridge->immune_effects.pe_sensitivity = 1.0f;
    bridge->immune_effects.retrieval_threshold = 0.0f;
    bridge->immune_effects.replay_priority = 0.0f;
    bridge->immune_effects.suppress_backward = false;
    bridge->immune_effects.boost_consolidation = false;

    if (bridge->omni_effects.response >= OMNI_IMMUNE_REGIONAL) {
        bridge->immune_effects.precision_modulation = 1.5f;
        bridge->immune_effects.pe_sensitivity = 1.2f;
    }
    if (bridge->omni_effects.response >= OMNI_IMMUNE_SYSTEMIC) {
        bridge->immune_effects.precision_modulation = 2.0f;
        bridge->immune_effects.boost_consolidation = true;
    }

    /* Update statistics */
    bridge->stats.total_updates++;
    if (bridge->omni_effects.response != OMNI_IMMUNE_NONE) {
        bridge->stats.immune_activations++;
    }
    switch (bridge->omni_effects.response) {
        case OMNI_IMMUNE_LOCAL: bridge->stats.local_responses++; break;
        case OMNI_IMMUNE_REGIONAL: bridge->stats.regional_responses++; break;
        case OMNI_IMMUNE_SYSTEMIC: bridge->stats.systemic_responses++; break;
        case OMNI_IMMUNE_MEMORY: bridge->stats.memory_responses++; break;
        default: break;
    }

    float n = (float)bridge->stats.total_updates;
    bridge->stats.avg_pe_magnitude =
        (bridge->stats.avg_pe_magnitude * (n - 1) + pe_magnitude) / n;
    bridge->stats.avg_inflammation =
        (bridge->stats.avg_inflammation * (n - 1) + bridge->omni_effects.inflammation_signal) / n;
    if (bridge->omni_effects.inflammation_signal > bridge->stats.max_inflammation) {
        bridge->stats.max_inflammation = bridge->omni_effects.inflammation_signal;
    }

    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_SUCCESS;
}

int omni_immune_apply_to_immune(omni_immune_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_INVALID_PARAM;
    /* In a full implementation, this would call brain_immune_system APIs */
    return NIMCP_SUCCESS;
}

int omni_immune_apply_to_omni(omni_immune_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_INVALID_PARAM;
    nimcp_mutex_lock(bridge->mutex);
    bridge->stats.precision_feedbacks++;
    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Query API
 * ============================================================================ */

int omni_immune_get_omni_effects(const omni_immune_bridge_t* bridge,
                                  omni_to_immune_effects_t* effects) {
    if (!bridge || !effects) return NIMCP_ERROR_INVALID_PARAM;
    nimcp_mutex_lock(((omni_immune_bridge_t*)bridge)->mutex);
    memcpy(effects, &bridge->omni_effects, sizeof(omni_to_immune_effects_t));
    nimcp_mutex_unlock(((omni_immune_bridge_t*)bridge)->mutex);
    return NIMCP_SUCCESS;
}

int omni_immune_get_immune_effects(const omni_immune_bridge_t* bridge,
                                    immune_to_omni_effects_t* effects) {
    if (!bridge || !effects) return NIMCP_ERROR_INVALID_PARAM;
    nimcp_mutex_lock(((omni_immune_bridge_t*)bridge)->mutex);
    memcpy(effects, &bridge->immune_effects, sizeof(immune_to_omni_effects_t));
    nimcp_mutex_unlock(((omni_immune_bridge_t*)bridge)->mutex);
    return NIMCP_SUCCESS;
}

int omni_immune_get_stats(const omni_immune_bridge_t* bridge,
                           omni_immune_stats_t* stats) {
    if (!bridge || !stats) return NIMCP_ERROR_INVALID_PARAM;
    nimcp_mutex_lock(((omni_immune_bridge_t*)bridge)->mutex);
    memcpy(stats, &bridge->stats, sizeof(omni_immune_stats_t));
    nimcp_mutex_unlock(((omni_immune_bridge_t*)bridge)->mutex);
    return NIMCP_SUCCESS;
}

int omni_immune_reset_stats(omni_immune_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_INVALID_PARAM;
    nimcp_mutex_lock(bridge->mutex);
    memset(&bridge->stats, 0, sizeof(omni_immune_stats_t));
    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Bio-Async Message Handlers
 * ============================================================================ */

static nimcp_error_t handle_immune_free_energy_report(
    const void* msg,
    size_t msg_size,
    nimcp_bio_promise_t response_promise,
    void* user_data)
{
    omni_immune_bridge_t* bridge = (omni_immune_bridge_t*)user_data;
    if (!bridge || !msg) return NIMCP_ERROR_INVALID_PARAM;

    omni_immune_update(bridge);

    (void)response_promise;
    (void)msg_size;
    return NIMCP_SUCCESS;
}

static nimcp_error_t handle_immune_error_propagate(
    const void* msg,
    size_t msg_size,
    nimcp_bio_promise_t response_promise,
    void* user_data)
{
    omni_immune_bridge_t* bridge = (omni_immune_bridge_t*)user_data;
    if (!bridge || !msg) return NIMCP_ERROR_INVALID_PARAM;

    /* Process error propagation for immune response */
    omni_immune_update(bridge);

    (void)response_promise;
    (void)msg_size;
    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Bio-Async API
 * ============================================================================ */

int omni_immune_connect_bio_async(omni_immune_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_INVALID_PARAM;
    if (bridge->bio_async_connected) return NIMCP_SUCCESS;

    bio_module_info_t info = {
        .module_id = BIO_MODULE_OMNI_IMMUNE_BRIDGE,
        .module_name = "omni_immune_bridge",
        .inbox_capacity = 32,
        .user_data = bridge
    };

    bio_module_context_t ctx = bio_router_register_module(&info);
    if (!ctx) {
        return NIMCP_ERROR_OPERATION_FAILED;
    }

    bridge->bio_context = ctx;

    bio_router_register_handler(ctx, BIO_MSG_OMNI_FREE_ENERGY_REPORT,
                                 handle_immune_free_energy_report);
    bio_router_register_handler(ctx, BIO_MSG_PRED_HIER_ERROR_PROPAGATE,
                                 handle_immune_error_propagate);

    bridge->bio_async_connected = true;
    return NIMCP_SUCCESS;
}

int omni_immune_disconnect_bio_async(omni_immune_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_INVALID_PARAM;
    if (!bridge->bio_async_connected) return NIMCP_SUCCESS;

    if (bridge->bio_context) {
        bio_router_unregister_module(bridge->bio_context);
        bridge->bio_context = NULL;
    }

    bridge->bio_async_connected = false;
    return NIMCP_SUCCESS;
}

bool omni_immune_is_bio_async_connected(const omni_immune_bridge_t* bridge) {
    if (!bridge) return false;
    return bridge->bio_async_connected;
}

/* ============================================================================
 * String Conversion API
 * ============================================================================ */

const char* omni_immune_source_to_string(omni_immune_source_t source) {
    switch (source) {
        case OMNI_SOURCE_JEPA_FORWARD: return "JEPA_FORWARD";
        case OMNI_SOURCE_JEPA_BACKWARD: return "JEPA_BACKWARD";
        case OMNI_SOURCE_JEPA_LATERAL: return "JEPA_LATERAL";
        case OMNI_SOURCE_HOPFIELD: return "HOPFIELD";
        case OMNI_SOURCE_PRED_HIERARCHY: return "PRED_HIERARCHY";
        case OMNI_SOURCE_TEMPORAL_REPLAY: return "TEMPORAL_REPLAY";
        case OMNI_SOURCE_AGGREGATED: return "AGGREGATED";
        default: return "UNKNOWN";
    }
}

const char* omni_immune_response_to_string(omni_immune_response_t response) {
    switch (response) {
        case OMNI_IMMUNE_NONE: return "NONE";
        case OMNI_IMMUNE_LOCAL: return "LOCAL";
        case OMNI_IMMUNE_REGIONAL: return "REGIONAL";
        case OMNI_IMMUNE_SYSTEMIC: return "SYSTEMIC";
        case OMNI_IMMUNE_MEMORY: return "MEMORY";
        default: return "UNKNOWN";
    }
}
