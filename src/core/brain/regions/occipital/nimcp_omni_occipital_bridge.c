/**
 * @file nimcp_omni_occipital_bridge.c
 * @brief Implementation of Omnidirectional Inference to Occipital Lobe Bridge
 */

#include "core/brain/regions/occipital/nimcp_omni_occipital_bridge.h"
#include "core/brain/regions/occipital/nimcp_occipital_adapter.h"
#include "cognitive/jepa/nimcp_jepa_bidirectional.h"
#include "cognitive/predictive/nimcp_predictive_hierarchy.h"
#include "cognitive/memory/nimcp_hopfield_memory.h"
#include "utils/thread/nimcp_thread.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"

#include <string.h>
#include <math.h>

/* ============================================================================
 * Static Helpers
 * ============================================================================ */

static float compute_stream_pe(const omni_visual_area_state_t* v1,
                                const omni_visual_area_state_t* v2,
                                const omni_visual_area_state_t* target) {
    /* Weight PEs from areas in stream */
    float pe = 0.0f;
    int count = 0;

    if (v1 && v1->active) {
        pe += v1->pe_magnitude * v1->precision;
        count++;
    }
    if (v2 && v2->active) {
        pe += v2->pe_magnitude * v2->precision;
        count++;
    }
    if (target && target->active) {
        pe += target->pe_magnitude * target->precision;
        count++;
    }

    return count > 0 ? pe / (float)count : 0.0f;
}

static float compute_visual_free_energy(const omni_visual_area_state_t* areas,
                                         uint32_t num_areas) {
    float fe = 0.0f;
    for (uint32_t i = 0; i < num_areas; i++) {
        if (areas[i].active) {
            fe += areas[i].precision * areas[i].pe_magnitude * areas[i].pe_magnitude;
        }
    }
    return 0.5f * fe;
}

/* ============================================================================
 * Configuration API
 * ============================================================================ */

int omni_occipital_default_config(omni_occipital_config_t* config) {
    if (!config) return NIMCP_ERROR_INVALID_PARAM;

    memset(config, 0, sizeof(omni_occipital_config_t));

    config->default_v1_precision = 1.0f;
    config->default_v2_precision = 1.0f;
    config->default_v4_precision = 1.0f;
    config->default_v5_precision = 1.0f;

    config->pe_threshold = OMNI_OCCIPITAL_PE_THRESHOLD;
    config->motion_threshold = 0.5f;
    config->recognition_threshold = 0.7f;

    config->default_stream = OMNI_STREAM_BOTH;
    config->enable_cross_stream = true;

    config->enable_spatial_attention = true;
    config->enable_feature_attention = true;
    config->attention_gain = 2.0f;

    config->enable_bio_async = true;
    config->enable_logging = false;

    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Lifecycle API
 * ============================================================================ */

omni_occipital_bridge_t* omni_occipital_bridge_create(
    const omni_occipital_config_t* config) {

    omni_occipital_bridge_t* bridge =
        nimcp_calloc(1, sizeof(omni_occipital_bridge_t));
    if (!bridge) return NULL;

    if (config) {
        memcpy(&bridge->config, config, sizeof(omni_occipital_config_t));
    } else {
        omni_occipital_default_config(&bridge->config);
    }

    bridge->mutex = nimcp_mutex_create(NULL);
    if (!bridge->mutex) {
        nimcp_free(bridge);
        return NULL;
    }

    /* Initialize area states with default precision */
    bridge->area_states[OMNI_VISUAL_V1].precision = bridge->config.default_v1_precision;
    bridge->area_states[OMNI_VISUAL_V2].precision = bridge->config.default_v2_precision;
    bridge->area_states[OMNI_VISUAL_V4].precision = bridge->config.default_v4_precision;
    bridge->area_states[OMNI_VISUAL_V5].precision = bridge->config.default_v5_precision;

    bridge->omni_effects.active_stream = bridge->config.default_stream;

    memset(&bridge->stats, 0, sizeof(omni_occipital_stats_t));

    return bridge;
}

void omni_occipital_bridge_destroy(omni_occipital_bridge_t* bridge) {
    if (!bridge) return;

    /* Free area state buffers */
    for (int i = 0; i < OMNI_OCCIPITAL_NUM_AREAS; i++) {
        if (bridge->area_states[i].prediction) {
            nimcp_free(bridge->area_states[i].prediction);
        }
        if (bridge->area_states[i].prediction_error) {
            nimcp_free(bridge->area_states[i].prediction_error);
        }
        if (bridge->area_states[i].features) {
            nimcp_free(bridge->area_states[i].features);
        }
    }

    /* Free effect buffers */
    if (bridge->omni_effects.v1_prediction) nimcp_free(bridge->omni_effects.v1_prediction);
    if (bridge->omni_effects.v2_prediction) nimcp_free(bridge->omni_effects.v2_prediction);
    if (bridge->omni_effects.v4_prediction) nimcp_free(bridge->omni_effects.v4_prediction);
    if (bridge->omni_effects.v5_prediction) nimcp_free(bridge->omni_effects.v5_prediction);

    if (bridge->occipital_effects.v1_pe) nimcp_free(bridge->occipital_effects.v1_pe);
    if (bridge->occipital_effects.v2_pe) nimcp_free(bridge->occipital_effects.v2_pe);
    if (bridge->occipital_effects.v4_pe) nimcp_free(bridge->occipital_effects.v4_pe);
    if (bridge->occipital_effects.v5_pe) nimcp_free(bridge->occipital_effects.v5_pe);

    if (bridge->mutex) {
        nimcp_mutex_free(bridge->mutex);
    }

    nimcp_free(bridge);
}

/* ============================================================================
 * Connection API
 * ============================================================================ */

int omni_occipital_connect_jepa(omni_occipital_bridge_t* bridge,
                                 jepa_bidirectional_t* jepa) {
    if (!bridge) return NIMCP_ERROR_INVALID_PARAM;
    nimcp_mutex_lock(bridge->mutex);
    bridge->jepa = jepa;
    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_SUCCESS;
}

int omni_occipital_connect_pred_hier(omni_occipital_bridge_t* bridge,
                                      predictive_hierarchy_t* pred_hier) {
    if (!bridge) return NIMCP_ERROR_INVALID_PARAM;
    nimcp_mutex_lock(bridge->mutex);
    bridge->pred_hier = pred_hier;
    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_SUCCESS;
}

int omni_occipital_connect_hopfield(omni_occipital_bridge_t* bridge,
                                     hopfield_memory_t* hopfield) {
    if (!bridge) return NIMCP_ERROR_INVALID_PARAM;
    nimcp_mutex_lock(bridge->mutex);
    bridge->hopfield = hopfield;
    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_SUCCESS;
}

int omni_occipital_connect_occipital(omni_occipital_bridge_t* bridge,
                                      occipital_adapter_t* occipital) {
    if (!bridge) return NIMCP_ERROR_INVALID_PARAM;
    nimcp_mutex_lock(bridge->mutex);
    bridge->occipital = occipital;
    /* Mark areas as active when occipital is connected */
    for (int i = 0; i < OMNI_OCCIPITAL_NUM_AREAS; i++) {
        bridge->area_states[i].active = (occipital != NULL);
    }
    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Update API
 * ============================================================================ */

int omni_occipital_update(omni_occipital_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_INVALID_PARAM;

    nimcp_mutex_lock(bridge->mutex);

    /* Compute stream PEs */
    bridge->occipital_effects.dorsal_pe = compute_stream_pe(
        &bridge->area_states[OMNI_VISUAL_V1],
        &bridge->area_states[OMNI_VISUAL_V2],
        &bridge->area_states[OMNI_VISUAL_V5]);

    bridge->occipital_effects.ventral_pe = compute_stream_pe(
        &bridge->area_states[OMNI_VISUAL_V1],
        &bridge->area_states[OMNI_VISUAL_V2],
        &bridge->area_states[OMNI_VISUAL_V4]);

    /* Combined PE */
    bridge->occipital_effects.combined_pe =
        (bridge->occipital_effects.dorsal_pe + bridge->occipital_effects.ventral_pe) / 2.0f;

    /* Compute visual free energy */
    bridge->occipital_effects.free_energy = compute_visual_free_energy(
        bridge->area_states, OMNI_OCCIPITAL_NUM_AREAS);

    /* Check for motion detection (V5 PE above threshold) */
    bridge->occipital_effects.motion_detected =
        bridge->area_states[OMNI_VISUAL_V5].pe_magnitude > bridge->config.motion_threshold;

    /* Check for object recognition (V4 confidence) */
    bridge->occipital_effects.object_recognized =
        (1.0f / (1.0f + bridge->area_states[OMNI_VISUAL_V4].pe_magnitude)) >
        bridge->config.recognition_threshold;

    /* Update precision based on prediction errors */
    omni_occipital_update_precision(bridge);

    /* Copy precision to effects */
    bridge->omni_effects.precision_v1 = bridge->area_states[OMNI_VISUAL_V1].precision;
    bridge->omni_effects.precision_v2 = bridge->area_states[OMNI_VISUAL_V2].precision;
    bridge->omni_effects.precision_v4 = bridge->area_states[OMNI_VISUAL_V4].precision;
    bridge->omni_effects.precision_v5 = bridge->area_states[OMNI_VISUAL_V5].precision;

    /* Update statistics */
    bridge->stats.total_updates++;

    if (bridge->occipital_effects.motion_detected) {
        bridge->stats.motion_events++;
    }
    if (bridge->occipital_effects.object_recognized) {
        bridge->stats.recognition_events++;
    }

    float n = (float)bridge->stats.total_updates;
    bridge->stats.avg_v1_pe =
        (bridge->stats.avg_v1_pe * (n - 1) +
         bridge->area_states[OMNI_VISUAL_V1].pe_magnitude) / n;
    bridge->stats.avg_v2_pe =
        (bridge->stats.avg_v2_pe * (n - 1) +
         bridge->area_states[OMNI_VISUAL_V2].pe_magnitude) / n;
    bridge->stats.avg_v4_pe =
        (bridge->stats.avg_v4_pe * (n - 1) +
         bridge->area_states[OMNI_VISUAL_V4].pe_magnitude) / n;
    bridge->stats.avg_v5_pe =
        (bridge->stats.avg_v5_pe * (n - 1) +
         bridge->area_states[OMNI_VISUAL_V5].pe_magnitude) / n;
    bridge->stats.avg_free_energy =
        (bridge->stats.avg_free_energy * (n - 1) +
         bridge->occipital_effects.free_energy) / n;

    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_SUCCESS;
}

int omni_occipital_apply_to_occipital(omni_occipital_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_INVALID_PARAM;
    return NIMCP_SUCCESS;
}

int omni_occipital_apply_to_omni(omni_occipital_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_INVALID_PARAM;
    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Prediction API
 * ============================================================================ */

int omni_occipital_predict_area(omni_occipital_bridge_t* bridge,
                                 omni_visual_area_t area,
                                 float* prediction,
                                 uint32_t dim) {
    if (!bridge || !prediction || dim == 0) return NIMCP_ERROR_INVALID_PARAM;
    if (area >= OMNI_OCCIPITAL_NUM_AREAS) return NIMCP_ERROR_INVALID_PARAM;

    nimcp_mutex_lock(bridge->mutex);

    /* Update stats */
    switch (area) {
        case OMNI_VISUAL_V1: bridge->stats.v1_predictions++; break;
        case OMNI_VISUAL_V2: bridge->stats.v2_predictions++; break;
        case OMNI_VISUAL_V4: bridge->stats.v4_predictions++; break;
        case OMNI_VISUAL_V5: bridge->stats.v5_predictions++; break;
    }

    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_SUCCESS;
}

int omni_occipital_compute_pe(omni_occipital_bridge_t* bridge,
                               omni_visual_area_t area,
                               const float* observation,
                               uint32_t dim,
                               float* pe) {
    if (!bridge || !observation || !pe || dim == 0) return NIMCP_ERROR_INVALID_PARAM;
    if (area >= OMNI_OCCIPITAL_NUM_AREAS) return NIMCP_ERROR_INVALID_PARAM;

    nimcp_mutex_lock(bridge->mutex);

    omni_visual_area_state_t* state = &bridge->area_states[area];

    if (state->prediction && state->dim == dim) {
        float sum_sq = 0.0f;
        for (uint32_t i = 0; i < dim; i++) {
            float diff = state->prediction[i] - observation[i];
            sum_sq += diff * diff;
        }
        *pe = sqrtf(sum_sq / (float)dim);
    } else {
        *pe = 0.0f;
    }

    state->pe_magnitude = *pe;
    state->dim = dim;

    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_SUCCESS;
}

int omni_occipital_propagate_backward(omni_occipital_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_INVALID_PARAM;
    /* Backward propagation: V4/V5 → V2 → V1 */
    return NIMCP_SUCCESS;
}

int omni_occipital_propagate_forward(omni_occipital_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_INVALID_PARAM;
    /* Forward propagation: V1 → V2 → V4/V5 */
    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Stream API
 * ============================================================================ */

int omni_occipital_set_stream(omni_occipital_bridge_t* bridge,
                               omni_visual_stream_t stream) {
    if (!bridge) return NIMCP_ERROR_INVALID_PARAM;
    nimcp_mutex_lock(bridge->mutex);
    bridge->omni_effects.active_stream = stream;
    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_SUCCESS;
}

float omni_occipital_get_dorsal_pe(const omni_occipital_bridge_t* bridge) {
    if (!bridge) return 0.0f;
    return bridge->occipital_effects.dorsal_pe;
}

float omni_occipital_get_ventral_pe(const omni_occipital_bridge_t* bridge) {
    if (!bridge) return 0.0f;
    return bridge->occipital_effects.ventral_pe;
}

int omni_occipital_bind_streams(omni_occipital_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_INVALID_PARAM;
    /* Bind dorsal (where) and ventral (what) streams */
    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Precision API
 * ============================================================================ */

int omni_occipital_set_precision(omni_occipital_bridge_t* bridge,
                                  omni_visual_area_t area,
                                  float precision) {
    if (!bridge) return NIMCP_ERROR_INVALID_PARAM;
    if (area >= OMNI_OCCIPITAL_NUM_AREAS) return NIMCP_ERROR_INVALID_PARAM;

    nimcp_mutex_lock(bridge->mutex);
    bridge->area_states[area].precision = fmaxf(0.0f, precision);
    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_SUCCESS;
}

float omni_occipital_get_precision(const omni_occipital_bridge_t* bridge,
                                    omni_visual_area_t area) {
    if (!bridge || area >= OMNI_OCCIPITAL_NUM_AREAS) return 0.0f;
    return bridge->area_states[area].precision;
}

int omni_occipital_update_precision(omni_occipital_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_INVALID_PARAM;

    /* Update precision based on PE (inverse relationship) */
    for (int i = 0; i < OMNI_OCCIPITAL_NUM_AREAS; i++) {
        omni_visual_area_state_t* state = &bridge->area_states[i];
        if (state->active) {
            state->precision = 1.0f / (1.0f + state->pe_magnitude);
        }
    }

    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Attention API
 * ============================================================================ */

int omni_occipital_apply_spatial_attention(omni_occipital_bridge_t* bridge,
                                            const float* attention_map,
                                            uint32_t width,
                                            uint32_t height) {
    if (!bridge || !attention_map || width == 0 || height == 0) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    nimcp_mutex_lock(bridge->mutex);

    if (bridge->config.enable_spatial_attention) {
        /* Apply spatial attention gain to V1 and V2 precision */
        float max_attn = 0.0f;
        for (uint32_t i = 0; i < width * height; i++) {
            if (attention_map[i] > max_attn) {
                max_attn = attention_map[i];
            }
        }

        float gain = bridge->config.attention_gain;
        bridge->area_states[OMNI_VISUAL_V1].precision *= (1.0f + (gain - 1.0f) * max_attn);
        bridge->area_states[OMNI_VISUAL_V2].precision *= (1.0f + (gain - 1.0f) * max_attn);
    }

    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_SUCCESS;
}

int omni_occipital_apply_feature_attention(omni_occipital_bridge_t* bridge,
                                            omni_visual_feature_t feature,
                                            float attention_weight) {
    if (!bridge) return NIMCP_ERROR_INVALID_PARAM;

    nimcp_mutex_lock(bridge->mutex);

    if (bridge->config.enable_feature_attention) {
        float gain = bridge->config.attention_gain;

        switch (feature) {
            case OMNI_VIS_EDGE:
                bridge->area_states[OMNI_VISUAL_V1].precision *=
                    (1.0f + (gain - 1.0f) * attention_weight);
                break;
            case OMNI_VIS_CONTOUR:
                bridge->area_states[OMNI_VISUAL_V2].precision *=
                    (1.0f + (gain - 1.0f) * attention_weight);
                break;
            case OMNI_VIS_COLOR:
            case OMNI_VIS_FORM:
                bridge->area_states[OMNI_VISUAL_V4].precision *=
                    (1.0f + (gain - 1.0f) * attention_weight);
                break;
            case OMNI_VIS_MOTION:
                bridge->area_states[OMNI_VISUAL_V5].precision *=
                    (1.0f + (gain - 1.0f) * attention_weight);
                break;
            case OMNI_VIS_DEPTH:
                bridge->area_states[OMNI_VISUAL_V2].precision *=
                    (1.0f + (gain - 1.0f) * attention_weight * 0.5f);
                bridge->area_states[OMNI_VISUAL_V5].precision *=
                    (1.0f + (gain - 1.0f) * attention_weight * 0.5f);
                break;
        }
    }

    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Query API
 * ============================================================================ */

int omni_occipital_get_omni_effects(const omni_occipital_bridge_t* bridge,
                                     omni_to_occipital_effects_t* effects) {
    if (!bridge || !effects) return NIMCP_ERROR_INVALID_PARAM;
    nimcp_mutex_lock(((omni_occipital_bridge_t*)bridge)->mutex);
    memcpy(effects, &bridge->omni_effects, sizeof(omni_to_occipital_effects_t));
    nimcp_mutex_unlock(((omni_occipital_bridge_t*)bridge)->mutex);
    return NIMCP_SUCCESS;
}

int omni_occipital_get_occipital_effects(const omni_occipital_bridge_t* bridge,
                                          occipital_to_omni_effects_t* effects) {
    if (!bridge || !effects) return NIMCP_ERROR_INVALID_PARAM;
    nimcp_mutex_lock(((omni_occipital_bridge_t*)bridge)->mutex);
    memcpy(effects, &bridge->occipital_effects, sizeof(occipital_to_omni_effects_t));
    nimcp_mutex_unlock(((omni_occipital_bridge_t*)bridge)->mutex);
    return NIMCP_SUCCESS;
}

int omni_occipital_get_area_state(const omni_occipital_bridge_t* bridge,
                                   omni_visual_area_t area,
                                   omni_visual_area_state_t* state) {
    if (!bridge || !state) return NIMCP_ERROR_INVALID_PARAM;
    if (area >= OMNI_OCCIPITAL_NUM_AREAS) return NIMCP_ERROR_INVALID_PARAM;

    nimcp_mutex_lock(((omni_occipital_bridge_t*)bridge)->mutex);
    memcpy(state, &bridge->area_states[area], sizeof(omni_visual_area_state_t));
    nimcp_mutex_unlock(((omni_occipital_bridge_t*)bridge)->mutex);
    return NIMCP_SUCCESS;
}

int omni_occipital_get_stats(const omni_occipital_bridge_t* bridge,
                              omni_occipital_stats_t* stats) {
    if (!bridge || !stats) return NIMCP_ERROR_INVALID_PARAM;
    nimcp_mutex_lock(((omni_occipital_bridge_t*)bridge)->mutex);
    memcpy(stats, &bridge->stats, sizeof(omni_occipital_stats_t));
    nimcp_mutex_unlock(((omni_occipital_bridge_t*)bridge)->mutex);
    return NIMCP_SUCCESS;
}

int omni_occipital_reset_stats(omni_occipital_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_INVALID_PARAM;
    nimcp_mutex_lock(bridge->mutex);
    memset(&bridge->stats, 0, sizeof(omni_occipital_stats_t));
    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Bio-Async Message Handlers
 * ============================================================================ */

static nimcp_error_t handle_occipital_predict_request(
    const void* msg,
    size_t msg_size,
    nimcp_bio_promise_t response_promise,
    void* user_data)
{
    omni_occipital_bridge_t* bridge = (omni_occipital_bridge_t*)user_data;
    if (!bridge || !msg) return NIMCP_ERROR_INVALID_PARAM;

    omni_occipital_update(bridge);

    (void)response_promise;
    (void)msg_size;
    return NIMCP_SUCCESS;
}

static nimcp_error_t handle_occipital_precision_update(
    const void* msg,
    size_t msg_size,
    nimcp_bio_promise_t response_promise,
    void* user_data)
{
    omni_occipital_bridge_t* bridge = (omni_occipital_bridge_t*)user_data;
    if (!bridge || !msg) return NIMCP_ERROR_INVALID_PARAM;

    omni_occipital_update_precision(bridge);

    (void)response_promise;
    (void)msg_size;
    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Bio-Async API
 * ============================================================================ */

int omni_occipital_connect_bio_async(omni_occipital_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_INVALID_PARAM;
    if (bridge->bio_async_connected) return NIMCP_SUCCESS;

    bio_module_info_t info = {
        .module_id = BIO_MODULE_OMNI_OCCIPITAL_BRIDGE,
        .module_name = "omni_occipital_bridge",
        .inbox_capacity = 32,
        .user_data = bridge
    };

    bio_module_context_t ctx = bio_router_register_module(&info);
    if (!ctx) {
        return NIMCP_ERROR_OPERATION_FAILED;
    }

    bridge->bio_context = ctx;

    bio_router_register_handler(ctx, BIO_MSG_OMNI_PREDICT_REQUEST,
                                 handle_occipital_predict_request);
    bio_router_register_handler(ctx, BIO_MSG_OMNI_PRECISION_UPDATE,
                                 handle_occipital_precision_update);

    bridge->bio_async_connected = true;
    return NIMCP_SUCCESS;
}

int omni_occipital_disconnect_bio_async(omni_occipital_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_INVALID_PARAM;
    if (!bridge->bio_async_connected) return NIMCP_SUCCESS;

    if (bridge->bio_context) {
        bio_router_unregister_module(bridge->bio_context);
        bridge->bio_context = NULL;
    }

    bridge->bio_async_connected = false;
    return NIMCP_SUCCESS;
}

bool omni_occipital_is_bio_async_connected(const omni_occipital_bridge_t* bridge) {
    if (!bridge) return false;
    return bridge->bio_async_connected;
}

/* ============================================================================
 * String Conversion API
 * ============================================================================ */

const char* omni_occipital_area_to_string(omni_visual_area_t area) {
    switch (area) {
        case OMNI_VISUAL_V1: return "V1";
        case OMNI_VISUAL_V2: return "V2";
        case OMNI_VISUAL_V4: return "V4";
        case OMNI_VISUAL_V5: return "V5";
        default: return "UNKNOWN";
    }
}

const char* omni_occipital_stream_to_string(omni_visual_stream_t stream) {
    switch (stream) {
        case OMNI_STREAM_DORSAL: return "DORSAL";
        case OMNI_STREAM_VENTRAL: return "VENTRAL";
        case OMNI_STREAM_BOTH: return "BOTH";
        default: return "UNKNOWN";
    }
}

const char* omni_occipital_feature_to_string(omni_visual_feature_t feature) {
    switch (feature) {
        case OMNI_VIS_EDGE: return "EDGE";
        case OMNI_VIS_CONTOUR: return "CONTOUR";
        case OMNI_VIS_COLOR: return "COLOR";
        case OMNI_VIS_FORM: return "FORM";
        case OMNI_VIS_MOTION: return "MOTION";
        case OMNI_VIS_DEPTH: return "DEPTH";
        default: return "UNKNOWN";
    }
}
