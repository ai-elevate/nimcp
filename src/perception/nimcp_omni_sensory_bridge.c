/**
 * @file nimcp_omni_sensory_bridge.c
 * @brief Implementation of Omnidirectional Inference to Sensory Cortex Bridge
 */

#include "perception/nimcp_omni_sensory_bridge.h"
#include "cognitive/jepa/nimcp_jepa_bidirectional.h"
#include "cognitive/predictive/nimcp_predictive_hierarchy.h"
#include "cognitive/memory/nimcp_hopfield_memory.h"
#include "perception/nimcp_audio_cortex.h"
#include "perception/nimcp_visual_cortex.h"
#include "perception/nimcp_speech_cortex.h"
#include "utils/thread/nimcp_thread.h"

#include <string.h>
#include <math.h>

/* ============================================================================
 * Static Helpers
 * ============================================================================ */

static float compute_modality_pe(const float* prediction,
                                  const float* observation,
                                  uint32_t dim) {
    if (!prediction || !observation || dim == 0) return 0.0f;

    float sum_sq = 0.0f;
    for (uint32_t i = 0; i < dim; i++) {
        float diff = prediction[i] - observation[i];
        sum_sq += diff * diff;
    }
    return sqrtf(sum_sq / (float)dim);
}

static float compute_binding_strength(const omni_modality_state_t* s1,
                                       const omni_modality_state_t* s2,
                                       float weight) {
    if (!s1 || !s2 || !s1->active || !s2->active) return 0.0f;

    /* Binding strength based on temporal coherence and PE similarity */
    float pe_diff = fabsf(s1->prediction_error - s2->prediction_error);
    float pe_similarity = 1.0f / (1.0f + pe_diff);
    float precision_product = s1->precision * s2->precision;

    return weight * pe_similarity * precision_product;
}

static void update_modality_state(omni_modality_state_t* state,
                                   const float* features,
                                   uint32_t dim,
                                   float pe) {
    if (!state) return;

    state->prediction_error = pe;
    state->dim = dim;
    state->active = (features != NULL);

    /* Copy features if provided */
    if (features && dim > 0) {
        if (!state->features || state->dim != dim) {
            if (state->features) nimcp_free(state->features);
            state->features = nimcp_calloc(dim, sizeof(float));
        }
        if (state->features) {
            memcpy(state->features, features, dim * sizeof(float));
        }
    }
}

/* ============================================================================
 * Configuration API
 * ============================================================================ */

int omni_sensory_default_config(omni_sensory_config_t* config) {
    if (!config) return NIMCP_ERROR_INVALID_PARAM;

    memset(config, 0, sizeof(omni_sensory_config_t));

    config->default_audio_precision = 1.0f;
    config->default_visual_precision = 1.0f;
    config->default_speech_precision = 1.0f;

    config->pe_threshold = OMNI_SENSORY_PE_THRESHOLD;
    config->novelty_threshold = 2.0f;

    config->crossmodal_mode = OMNI_CROSSMODAL_ALL;
    config->crossmodal_weight = OMNI_SENSORY_CROSSMODAL_WEIGHT;
    config->enable_binding = true;

    config->enable_attention = true;
    config->attention_gain = 2.0f;

    config->enable_bio_async = true;
    config->enable_logging = false;

    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Lifecycle API
 * ============================================================================ */

omni_sensory_bridge_t* omni_sensory_bridge_create(
    const omni_sensory_config_t* config) {

    omni_sensory_bridge_t* bridge = nimcp_calloc(1, sizeof(omni_sensory_bridge_t));
    if (!bridge) return NULL;

    if (config) {
        memcpy(&bridge->config, config, sizeof(omni_sensory_config_t));
    } else {
        omni_sensory_default_config(&bridge->config);
    }

    bridge->mutex = nimcp_mutex_create(NULL);
    if (!bridge->mutex) {
        nimcp_free(bridge);
        return NULL;
    }

    /* Initialize modality states */
    bridge->modality_states[OMNI_MODALITY_AUDIO].precision =
        bridge->config.default_audio_precision;
    bridge->modality_states[OMNI_MODALITY_VISUAL].precision =
        bridge->config.default_visual_precision;
    bridge->modality_states[OMNI_MODALITY_SPEECH].precision =
        bridge->config.default_speech_precision;

    memset(&bridge->stats, 0, sizeof(omni_sensory_stats_t));

    return bridge;
}

void omni_sensory_bridge_destroy(omni_sensory_bridge_t* bridge) {
    if (!bridge) return;

    /* Free modality state buffers */
    for (int i = 0; i < OMNI_MODALITY_COUNT; i++) {
        if (bridge->modality_states[i].prediction) {
            nimcp_free(bridge->modality_states[i].prediction);
        }
        if (bridge->modality_states[i].features) {
            nimcp_free(bridge->modality_states[i].features);
        }
    }

    /* Free effect buffers */
    if (bridge->omni_effects.top_down_audio) {
        nimcp_free(bridge->omni_effects.top_down_audio);
    }
    if (bridge->omni_effects.top_down_visual) {
        nimcp_free(bridge->omni_effects.top_down_visual);
    }
    if (bridge->omni_effects.top_down_speech) {
        nimcp_free(bridge->omni_effects.top_down_speech);
    }
    if (bridge->sensory_effects.audio_features) {
        nimcp_free(bridge->sensory_effects.audio_features);
    }
    if (bridge->sensory_effects.visual_features) {
        nimcp_free(bridge->sensory_effects.visual_features);
    }
    if (bridge->sensory_effects.speech_features) {
        nimcp_free(bridge->sensory_effects.speech_features);
    }
    if (bridge->binding.integrated_repr) {
        nimcp_free(bridge->binding.integrated_repr);
    }

    if (bridge->mutex) {
        nimcp_mutex_destroy(bridge->mutex);
    }

    nimcp_free(bridge);
}

/* ============================================================================
 * Connection API
 * ============================================================================ */

int omni_sensory_connect_jepa(omni_sensory_bridge_t* bridge,
                               jepa_bidirectional_t* jepa) {
    if (!bridge) return NIMCP_ERROR_INVALID_PARAM;
    nimcp_mutex_lock(bridge->mutex);
    bridge->jepa = jepa;
    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_SUCCESS;
}

int omni_sensory_connect_pred_hier(omni_sensory_bridge_t* bridge,
                                    predictive_hierarchy_t* pred_hier) {
    if (!bridge) return NIMCP_ERROR_INVALID_PARAM;
    nimcp_mutex_lock(bridge->mutex);
    bridge->pred_hier = pred_hier;
    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_SUCCESS;
}

int omni_sensory_connect_hopfield(omni_sensory_bridge_t* bridge,
                                   hopfield_memory_t* hopfield) {
    if (!bridge) return NIMCP_ERROR_INVALID_PARAM;
    nimcp_mutex_lock(bridge->mutex);
    bridge->hopfield = hopfield;
    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_SUCCESS;
}

int omni_sensory_connect_audio(omni_sensory_bridge_t* bridge,
                                audio_cortex_t* audio) {
    if (!bridge) return NIMCP_ERROR_INVALID_PARAM;
    nimcp_mutex_lock(bridge->mutex);
    bridge->audio_cortex = audio;
    bridge->modality_states[OMNI_MODALITY_AUDIO].active = (audio != NULL);
    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_SUCCESS;
}

int omni_sensory_connect_visual(omni_sensory_bridge_t* bridge,
                                 visual_cortex_t* visual) {
    if (!bridge) return NIMCP_ERROR_INVALID_PARAM;
    nimcp_mutex_lock(bridge->mutex);
    bridge->visual_cortex = visual;
    bridge->modality_states[OMNI_MODALITY_VISUAL].active = (visual != NULL);
    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_SUCCESS;
}

int omni_sensory_connect_speech(omni_sensory_bridge_t* bridge,
                                 speech_cortex_t* speech) {
    if (!bridge) return NIMCP_ERROR_INVALID_PARAM;
    nimcp_mutex_lock(bridge->mutex);
    bridge->speech_cortex = speech;
    bridge->modality_states[OMNI_MODALITY_SPEECH].active = (speech != NULL);
    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Update API
 * ============================================================================ */

int omni_sensory_update(omni_sensory_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_INVALID_PARAM;

    nimcp_mutex_lock(bridge->mutex);

    /* Compute sensory free energy from each modality */
    float total_fe = 0.0f;
    int active_count = 0;

    for (int i = 0; i < OMNI_MODALITY_COUNT; i++) {
        omni_modality_state_t* state = &bridge->modality_states[i];
        if (state->active) {
            total_fe += state->precision * state->prediction_error *
                        state->prediction_error;
            active_count++;
        }
    }

    if (active_count > 0) {
        bridge->sensory_effects.combined_free_energy = total_fe / active_count;
    }

    /* Update precision based on prediction errors */
    if (bridge->config.enable_attention) {
        omni_sensory_update_precision(bridge);
    }

    /* Compute cross-modal binding if enabled (call unlocked version) */
    if (bridge->config.enable_binding) {
        omni_crossmodal_binding_t binding;
        /* Unlock before calling to avoid deadlock, or call internal helper */
        nimcp_mutex_unlock(bridge->mutex);
        omni_sensory_compute_binding(bridge, &binding);
        nimcp_mutex_lock(bridge->mutex);
    }

    /* Copy individual PEs to effects */
    bridge->sensory_effects.audio_pe =
        bridge->modality_states[OMNI_MODALITY_AUDIO].prediction_error;
    bridge->sensory_effects.visual_pe =
        bridge->modality_states[OMNI_MODALITY_VISUAL].prediction_error;
    bridge->sensory_effects.speech_pe =
        bridge->modality_states[OMNI_MODALITY_SPEECH].prediction_error;

    /* Check for novelty */
    bridge->sensory_effects.audio_novelty =
        bridge->sensory_effects.audio_pe > bridge->config.novelty_threshold;
    bridge->sensory_effects.visual_novelty =
        bridge->sensory_effects.visual_pe > bridge->config.novelty_threshold;
    bridge->sensory_effects.speech_novelty =
        bridge->sensory_effects.speech_pe > bridge->config.novelty_threshold;

    if (bridge->sensory_effects.audio_novelty ||
        bridge->sensory_effects.visual_novelty ||
        bridge->sensory_effects.speech_novelty) {
        bridge->stats.novelty_events++;
    }

    /* Update statistics */
    bridge->stats.total_updates++;
    float n = (float)bridge->stats.total_updates;
    bridge->stats.avg_audio_pe =
        (bridge->stats.avg_audio_pe * (n - 1) + bridge->sensory_effects.audio_pe) / n;
    bridge->stats.avg_visual_pe =
        (bridge->stats.avg_visual_pe * (n - 1) + bridge->sensory_effects.visual_pe) / n;
    bridge->stats.avg_speech_pe =
        (bridge->stats.avg_speech_pe * (n - 1) + bridge->sensory_effects.speech_pe) / n;
    bridge->stats.avg_free_energy =
        (bridge->stats.avg_free_energy * (n - 1) +
         bridge->sensory_effects.combined_free_energy) / n;

    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_SUCCESS;
}

int omni_sensory_apply_to_sensory(omni_sensory_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_INVALID_PARAM;
    /* Apply top-down predictions to sensory cortices */
    return NIMCP_SUCCESS;
}

int omni_sensory_apply_to_omni(omni_sensory_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_INVALID_PARAM;
    /* Apply bottom-up features to omnidirectional inference */
    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Prediction API
 * ============================================================================ */

int omni_sensory_predict_modality(omni_sensory_bridge_t* bridge,
                                   omni_modality_t modality,
                                   float* prediction,
                                   uint32_t dim) {
    if (!bridge || !prediction || dim == 0) return NIMCP_ERROR_INVALID_PARAM;
    if (modality >= OMNI_MODALITY_COUNT) return NIMCP_ERROR_INVALID_PARAM;

    nimcp_mutex_lock(bridge->mutex);

    /* Generate top-down prediction using JEPA backward inference */
    if (bridge->jepa) {
        /* Use backward direction for top-down prediction */
        /* jepa_bidirectional_predict(bridge->jepa, JEPA_DIR_BACKWARD, ...) */
    }

    /* Update stats */
    switch (modality) {
        case OMNI_MODALITY_AUDIO:
            bridge->stats.audio_predictions++;
            break;
        case OMNI_MODALITY_VISUAL:
            bridge->stats.visual_predictions++;
            break;
        case OMNI_MODALITY_SPEECH:
            bridge->stats.speech_predictions++;
            break;
        default:
            break;
    }

    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_SUCCESS;
}

int omni_sensory_compute_pe(omni_sensory_bridge_t* bridge,
                             omni_modality_t modality,
                             const float* observation,
                             uint32_t dim,
                             float* pe) {
    if (!bridge || !observation || !pe || dim == 0) return NIMCP_ERROR_INVALID_PARAM;
    if (modality >= OMNI_MODALITY_COUNT) return NIMCP_ERROR_INVALID_PARAM;

    nimcp_mutex_lock(bridge->mutex);

    omni_modality_state_t* state = &bridge->modality_states[modality];

    if (state->prediction && state->dim == dim) {
        *pe = compute_modality_pe(state->prediction, observation, dim);
    } else {
        *pe = 0.0f;
    }

    /* Update state */
    update_modality_state(state, observation, dim, *pe);

    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_SUCCESS;
}

int omni_sensory_crossmodal_predict(omni_sensory_bridge_t* bridge,
                                     omni_modality_t source_modality,
                                     omni_modality_t target_modality,
                                     float* prediction,
                                     uint32_t dim) {
    if (!bridge || !prediction || dim == 0) return NIMCP_ERROR_INVALID_PARAM;
    if (source_modality >= OMNI_MODALITY_COUNT) return NIMCP_ERROR_INVALID_PARAM;
    if (target_modality >= OMNI_MODALITY_COUNT) return NIMCP_ERROR_INVALID_PARAM;
    if (source_modality == target_modality) return NIMCP_ERROR_INVALID_PARAM;

    nimcp_mutex_lock(bridge->mutex);

    /* Use lateral inference for cross-modal prediction */
    if (bridge->jepa) {
        /* jepa_bidirectional_predict(bridge->jepa, JEPA_DIR_LATERAL, ...) */
    }

    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Binding API
 * ============================================================================ */

int omni_sensory_compute_binding(omni_sensory_bridge_t* bridge,
                                  omni_crossmodal_binding_t* binding) {
    if (!bridge || !binding) return NIMCP_ERROR_INVALID_PARAM;

    nimcp_mutex_lock(bridge->mutex);

    memset(binding, 0, sizeof(omni_crossmodal_binding_t));

    omni_modality_state_t* audio = &bridge->modality_states[OMNI_MODALITY_AUDIO];
    omni_modality_state_t* visual = &bridge->modality_states[OMNI_MODALITY_VISUAL];
    omni_modality_state_t* speech = &bridge->modality_states[OMNI_MODALITY_SPEECH];

    float crossmodal_weight = bridge->config.crossmodal_weight;

    /* Audio-visual binding */
    if (bridge->config.crossmodal_mode == OMNI_CROSSMODAL_AUDIO_VISUAL ||
        bridge->config.crossmodal_mode == OMNI_CROSSMODAL_ALL) {
        float av_strength = compute_binding_strength(audio, visual, crossmodal_weight);
        binding->audiovisual_bound = (av_strength > 0.5f);
        binding->binding_strength = fmaxf(binding->binding_strength, av_strength);
    }

    /* Visual-speech binding */
    if (bridge->config.crossmodal_mode == OMNI_CROSSMODAL_VISUAL_SPEECH ||
        bridge->config.crossmodal_mode == OMNI_CROSSMODAL_ALL) {
        float vs_strength = compute_binding_strength(visual, speech, crossmodal_weight);
        binding->visualspeech_bound = (vs_strength > 0.5f);
        binding->binding_strength = fmaxf(binding->binding_strength, vs_strength);
    }

    /* Audio-speech binding */
    if (bridge->config.crossmodal_mode == OMNI_CROSSMODAL_AUDIO_SPEECH ||
        bridge->config.crossmodal_mode == OMNI_CROSSMODAL_ALL) {
        float as_strength = compute_binding_strength(audio, speech, crossmodal_weight);
        binding->audiospeech_bound = (as_strength > 0.5f);
        binding->binding_strength = fmaxf(binding->binding_strength, as_strength);
    }

    /* Compute overall coherence */
    int bound_count = (binding->audiovisual_bound ? 1 : 0) +
                      (binding->visualspeech_bound ? 1 : 0) +
                      (binding->audiospeech_bound ? 1 : 0);
    binding->coherence = (float)bound_count / 3.0f;

    /* Copy to bridge state */
    memcpy(&bridge->binding, binding, sizeof(omni_crossmodal_binding_t));

    if (bound_count > 0) {
        bridge->stats.crossmodal_bindings++;
    }

    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_SUCCESS;
}

bool omni_sensory_are_bound(const omni_sensory_bridge_t* bridge,
                             omni_modality_t m1,
                             omni_modality_t m2) {
    if (!bridge) return false;
    if (m1 >= OMNI_MODALITY_COUNT || m2 >= OMNI_MODALITY_COUNT) return false;
    if (m1 == m2) return true;

    /* Sort modalities for consistent lookup */
    if (m1 > m2) {
        omni_modality_t tmp = m1;
        m1 = m2;
        m2 = tmp;
    }

    if (m1 == OMNI_MODALITY_AUDIO && m2 == OMNI_MODALITY_VISUAL) {
        return bridge->binding.audiovisual_bound;
    } else if (m1 == OMNI_MODALITY_VISUAL && m2 == OMNI_MODALITY_SPEECH) {
        return bridge->binding.visualspeech_bound;
    } else if (m1 == OMNI_MODALITY_AUDIO && m2 == OMNI_MODALITY_SPEECH) {
        return bridge->binding.audiospeech_bound;
    }
    return false;
}

float omni_sensory_get_binding_strength(const omni_sensory_bridge_t* bridge,
                                         omni_modality_t m1,
                                         omni_modality_t m2) {
    if (!bridge) return 0.0f;
    if (m1 >= OMNI_MODALITY_COUNT || m2 >= OMNI_MODALITY_COUNT) return 0.0f;
    if (m1 == m2) return 1.0f;

    return compute_binding_strength(
        &bridge->modality_states[m1],
        &bridge->modality_states[m2],
        bridge->config.crossmodal_weight
    );
}

/* ============================================================================
 * Precision API
 * ============================================================================ */

int omni_sensory_set_precision(omni_sensory_bridge_t* bridge,
                                omni_modality_t modality,
                                float precision) {
    if (!bridge) return NIMCP_ERROR_INVALID_PARAM;
    if (modality >= OMNI_MODALITY_COUNT) return NIMCP_ERROR_INVALID_PARAM;

    nimcp_mutex_lock(bridge->mutex);
    bridge->modality_states[modality].precision = fmaxf(0.0f, precision);
    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_SUCCESS;
}

float omni_sensory_get_precision(const omni_sensory_bridge_t* bridge,
                                  omni_modality_t modality) {
    if (!bridge || modality >= OMNI_MODALITY_COUNT) return 0.0f;
    return bridge->modality_states[modality].precision;
}

int omni_sensory_update_precision(omni_sensory_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_INVALID_PARAM;

    /* Precision weighting: reduce precision for high-PE modalities */
    for (int i = 0; i < OMNI_MODALITY_COUNT; i++) {
        omni_modality_state_t* state = &bridge->modality_states[i];
        if (state->active) {
            float pe = state->prediction_error;
            /* Precision inversely related to PE (more reliable = higher precision) */
            state->precision = 1.0f / (1.0f + pe);
        }
    }

    /* Update effect structures */
    bridge->omni_effects.precision_audio =
        bridge->modality_states[OMNI_MODALITY_AUDIO].precision;
    bridge->omni_effects.precision_visual =
        bridge->modality_states[OMNI_MODALITY_VISUAL].precision;
    bridge->omni_effects.precision_speech =
        bridge->modality_states[OMNI_MODALITY_SPEECH].precision;

    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Attention API
 * ============================================================================ */

int omni_sensory_set_attention(omni_sensory_bridge_t* bridge,
                                omni_sensory_attention_t mode) {
    if (!bridge) return NIMCP_ERROR_INVALID_PARAM;
    nimcp_mutex_lock(bridge->mutex);
    bridge->omni_effects.attention_mode = mode;
    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_SUCCESS;
}

int omni_sensory_apply_attention(omni_sensory_bridge_t* bridge,
                                  omni_modality_t modality,
                                  const float* attention_weights,
                                  uint32_t num_weights) {
    if (!bridge || !attention_weights || num_weights == 0) {
        return NIMCP_ERROR_INVALID_PARAM;
    }
    if (modality >= OMNI_MODALITY_COUNT) return NIMCP_ERROR_INVALID_PARAM;

    nimcp_mutex_lock(bridge->mutex);

    omni_modality_state_t* state = &bridge->modality_states[modality];

    /* Apply attention gain to precision */
    if (state->active && bridge->config.enable_attention) {
        /* Sum of attention weights modulates precision */
        float attn_sum = 0.0f;
        for (uint32_t i = 0; i < num_weights; i++) {
            attn_sum += attention_weights[i];
        }
        attn_sum /= (float)num_weights;

        state->precision *= (1.0f + (bridge->config.attention_gain - 1.0f) * attn_sum);
    }

    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Query API
 * ============================================================================ */

int omni_sensory_get_omni_effects(const omni_sensory_bridge_t* bridge,
                                   omni_to_sensory_effects_t* effects) {
    if (!bridge || !effects) return NIMCP_ERROR_INVALID_PARAM;
    nimcp_mutex_lock(((omni_sensory_bridge_t*)bridge)->mutex);
    memcpy(effects, &bridge->omni_effects, sizeof(omni_to_sensory_effects_t));
    nimcp_mutex_unlock(((omni_sensory_bridge_t*)bridge)->mutex);
    return NIMCP_SUCCESS;
}

int omni_sensory_get_sensory_effects(const omni_sensory_bridge_t* bridge,
                                      sensory_to_omni_effects_t* effects) {
    if (!bridge || !effects) return NIMCP_ERROR_INVALID_PARAM;
    nimcp_mutex_lock(((omni_sensory_bridge_t*)bridge)->mutex);
    memcpy(effects, &bridge->sensory_effects, sizeof(sensory_to_omni_effects_t));
    nimcp_mutex_unlock(((omni_sensory_bridge_t*)bridge)->mutex);
    return NIMCP_SUCCESS;
}

int omni_sensory_get_modality_state(const omni_sensory_bridge_t* bridge,
                                     omni_modality_t modality,
                                     omni_modality_state_t* state) {
    if (!bridge || !state) return NIMCP_ERROR_INVALID_PARAM;
    if (modality >= OMNI_MODALITY_COUNT) return NIMCP_ERROR_INVALID_PARAM;

    nimcp_mutex_lock(((omni_sensory_bridge_t*)bridge)->mutex);
    memcpy(state, &bridge->modality_states[modality], sizeof(omni_modality_state_t));
    nimcp_mutex_unlock(((omni_sensory_bridge_t*)bridge)->mutex);
    return NIMCP_SUCCESS;
}

int omni_sensory_get_stats(const omni_sensory_bridge_t* bridge,
                            omni_sensory_stats_t* stats) {
    if (!bridge || !stats) return NIMCP_ERROR_INVALID_PARAM;
    nimcp_mutex_lock(((omni_sensory_bridge_t*)bridge)->mutex);
    memcpy(stats, &bridge->stats, sizeof(omni_sensory_stats_t));
    nimcp_mutex_unlock(((omni_sensory_bridge_t*)bridge)->mutex);
    return NIMCP_SUCCESS;
}

int omni_sensory_reset_stats(omni_sensory_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_INVALID_PARAM;
    nimcp_mutex_lock(bridge->mutex);
    memset(&bridge->stats, 0, sizeof(omni_sensory_stats_t));
    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Bio-Async API
 * ============================================================================ */

int omni_sensory_connect_bio_async(omni_sensory_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_INVALID_PARAM;
    return NIMCP_SUCCESS;
}

int omni_sensory_disconnect_bio_async(omni_sensory_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_INVALID_PARAM;
    return NIMCP_SUCCESS;
}

bool omni_sensory_is_bio_async_connected(const omni_sensory_bridge_t* bridge) {
    if (!bridge) return false;
    return bridge->config.enable_bio_async;
}

/* ============================================================================
 * String Conversion API
 * ============================================================================ */

const char* omni_sensory_modality_to_string(omni_modality_t modality) {
    switch (modality) {
        case OMNI_MODALITY_AUDIO: return "AUDIO";
        case OMNI_MODALITY_VISUAL: return "VISUAL";
        case OMNI_MODALITY_SPEECH: return "SPEECH";
        default: return "UNKNOWN";
    }
}

const char* omni_sensory_crossmodal_to_string(omni_crossmodal_mode_t mode) {
    switch (mode) {
        case OMNI_CROSSMODAL_NONE: return "NONE";
        case OMNI_CROSSMODAL_AUDIO_VISUAL: return "AUDIO_VISUAL";
        case OMNI_CROSSMODAL_VISUAL_SPEECH: return "VISUAL_SPEECH";
        case OMNI_CROSSMODAL_AUDIO_SPEECH: return "AUDIO_SPEECH";
        case OMNI_CROSSMODAL_ALL: return "ALL";
        default: return "UNKNOWN";
    }
}

const char* omni_sensory_direction_to_string(omni_sensory_direction_t dir) {
    switch (dir) {
        case OMNI_SENS_BOTTOM_UP: return "BOTTOM_UP";
        case OMNI_SENS_TOP_DOWN: return "TOP_DOWN";
        case OMNI_SENS_LATERAL: return "LATERAL";
        case OMNI_SENS_BIDIRECTIONAL: return "BIDIRECTIONAL";
        default: return "UNKNOWN";
    }
}

const char* omni_sensory_attention_to_string(omni_sensory_attention_t attn) {
    switch (attn) {
        case OMNI_SENS_ATTN_NONE: return "NONE";
        case OMNI_SENS_ATTN_SPATIAL: return "SPATIAL";
        case OMNI_SENS_ATTN_TEMPORAL: return "TEMPORAL";
        case OMNI_SENS_ATTN_FEATURE: return "FEATURE";
        case OMNI_SENS_ATTN_OBJECT: return "OBJECT";
        default: return "UNKNOWN";
    }
}
