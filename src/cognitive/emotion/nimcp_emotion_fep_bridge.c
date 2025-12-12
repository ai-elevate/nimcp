/**
 * @file nimcp_emotion_fep_bridge.c
 * @brief Free Energy Principle - Emotion Integration Bridge Implementation
 */

#include "cognitive/emotion/nimcp_emotion_fep_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/validation/nimcp_common.h"
#include <string.h>

#define LOG_MODULE "emotion_fep_bridge"

int emotion_fep_bridge_default_config(emotion_fep_config_t* config) {
    if (!config) return NIMCP_ERROR_NULL_POINTER;
    config->pe_valence_scaling = 1.0f;
    config->pe_arousal_scaling = EMOTION_FEP_AROUSAL_PE_SCALING;
    config->precision_intensity_scaling = 1.0f;
    config->enable_pe_emotion_generation = true;
    config->enable_precision_intensity = true;
    config->enable_interoceptive_inference = true;
    config->emotion_precision_modulation = 1.0f;
    config->emotion_learning_modulation = 1.0f;
    config->enable_emotion_precision = true;
    config->enable_emotion_learning = true;
    config->fe_sensitivity = 1.0f;
    config->emotion_sensitivity = 1.0f;
    return 0;
}

emotion_fep_bridge_t* emotion_fep_bridge_create(const emotion_fep_config_t* config) {
    emotion_fep_bridge_t* bridge = nimcp_malloc(sizeof(emotion_fep_bridge_t));
    if (!bridge) return NULL;
    memset(bridge, 0, sizeof(emotion_fep_bridge_t));
    if (config) bridge->config = *config;
    else emotion_fep_bridge_default_config(&bridge->config);
    bridge->mutex = nimcp_platform_mutex_create();
    if (!bridge->mutex) { nimcp_free(bridge); return NULL; }
    return bridge;
}

void emotion_fep_bridge_destroy(emotion_fep_bridge_t* bridge) {
    if (!bridge) return;
    if (bridge->bio_async_enabled) emotion_fep_bridge_disconnect_bio_async(bridge);
    if (bridge->mutex) nimcp_mutex_destroy(bridge->mutex);
    nimcp_free(bridge);
}

int emotion_fep_bridge_connect_fep(emotion_fep_bridge_t* bridge, fep_system_t* fep) {
    if (!bridge || !fep) return NIMCP_ERROR_NULL_POINTER;
    nimcp_mutex_lock(bridge->mutex);
    bridge->fep_system = fep;
    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int emotion_fep_bridge_connect_emotion(emotion_fep_bridge_t* bridge, emotion_recognition_system_t* emotion) {
    if (!bridge || !emotion) return NIMCP_ERROR_NULL_POINTER;
    nimcp_mutex_lock(bridge->mutex);
    bridge->emotion_system = emotion;
    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int emotion_fep_bridge_disconnect(emotion_fep_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    nimcp_mutex_lock(bridge->mutex);
    bridge->fep_system = NULL;
    bridge->emotion_system = NULL;
    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int emotion_fep_generate_valenced_pe(emotion_fep_bridge_t* bridge, float pe_magnitude) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    if (!bridge->config.enable_pe_emotion_generation) return 0;
    nimcp_mutex_lock(bridge->mutex);
    bridge->fep_effects.prediction_error_valence = (pe_magnitude > 0) ? 1.0f : -1.0f;
    bridge->fep_effects.prediction_error_arousal = pe_magnitude * bridge->config.pe_arousal_scaling;
    bridge->fep_effects.emotion_generated = true;
    bridge->stats.emotion_generation_events++;
    bridge->state.current_prediction_error = pe_magnitude;
    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int emotion_fep_modulate_precision_by_intensity(emotion_fep_bridge_t* bridge) {
    if (!bridge || !bridge->fep_system) return NIMCP_ERROR_NULL_POINTER;
    if (!bridge->config.enable_precision_intensity) return 0;
    nimcp_mutex_lock(bridge->mutex);
    bridge->fep_effects.precision_intensity = bridge->state.current_precision * bridge->config.precision_intensity_scaling;
    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int emotion_fep_apply_emotion_precision_modulation(emotion_fep_bridge_t* bridge) {
    if (!bridge || !bridge->fep_system) return NIMCP_ERROR_NULL_POINTER;
    if (!bridge->config.enable_emotion_precision) return 0;
    nimcp_mutex_lock(bridge->mutex);
    bridge->emotion_effects.emotion_precision_modifier = bridge->config.emotion_precision_modulation;
    bridge->stats.precision_modulation_events++;
    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int emotion_fep_apply_emotion_learning_modulation(emotion_fep_bridge_t* bridge) {
    if (!bridge || !bridge->fep_system) return NIMCP_ERROR_NULL_POINTER;
    if (!bridge->config.enable_emotion_learning) return 0;
    nimcp_mutex_lock(bridge->mutex);
    bridge->emotion_effects.emotion_learning_modifier = bridge->config.emotion_learning_modulation;
    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int emotion_fep_bridge_update(emotion_fep_bridge_t* bridge, uint64_t delta_ms) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    emotion_fep_modulate_precision_by_intensity(bridge);
    emotion_fep_apply_emotion_precision_modulation(bridge);
    emotion_fep_apply_emotion_learning_modulation(bridge);
    return 0;
}

int emotion_fep_bridge_get_state(const emotion_fep_bridge_t* bridge, emotion_fep_state_t* state) {
    if (!bridge || !state) return NIMCP_ERROR_NULL_POINTER;
    nimcp_mutex_lock(bridge->mutex);
    *state = bridge->state;
    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int emotion_fep_bridge_get_stats(const emotion_fep_bridge_t* bridge, emotion_fep_stats_t* stats) {
    if (!bridge || !stats) return NIMCP_ERROR_NULL_POINTER;
    nimcp_mutex_lock(bridge->mutex);
    *stats = bridge->stats;
    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int emotion_fep_bridge_connect_bio_async(emotion_fep_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    if (bridge->bio_async_enabled) return 0;
    bio_module_info_t info = {
        .module_id = BIO_MODULE_FEP_EMOTION_BRIDGE,
        .module_name = "emotion_fep_bridge",
        .inbox_capacity = 32,
        .user_data = bridge
    };
    bridge->bio_ctx = bio_router_register_module(&info);
    if (bridge->bio_ctx) bridge->bio_async_enabled = true;
    return 0;
}

int emotion_fep_bridge_disconnect_bio_async(emotion_fep_bridge_t* bridge) {
    if (!bridge || !bridge->bio_async_enabled) return 0;
    if (bridge->bio_ctx) bio_router_unregister_module(bridge->bio_ctx);
    bridge->bio_ctx = NULL;
    bridge->bio_async_enabled = false;
    return 0;
}

bool emotion_fep_bridge_is_bio_async_connected(const emotion_fep_bridge_t* bridge) {
    return bridge ? bridge->bio_async_enabled : false;
}
