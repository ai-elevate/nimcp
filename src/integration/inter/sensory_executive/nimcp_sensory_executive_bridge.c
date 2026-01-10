/**
 * @file nimcp_sensory_executive_bridge.c
 * @brief Sensory-Executive Inter-Layer Bridge Implementation
 * @version 1.0.0
 * @date 2026-01-10
 */

#include "integration/inter/sensory_executive/nimcp_sensory_executive_bridge.h"
#include <string.h>
#include <stdlib.h>

struct nimcp_sensory_executive_bridge_struct {
    nimcp_sensory_executive_config_t config;
    nimcp_layer_registry_t registry;
    nimcp_sensory_intra_t sensory;
    nimcp_executive_intra_t executive;
    nimcp_sensory_executive_state_t state;
    nimcp_sensory_executive_stats_t stats;
    bool is_initialized;
};

nimcp_sensory_executive_config_t nimcp_sensory_executive_default_config(void) {
    nimcp_sensory_executive_config_t config = {
        .attention_capture_threshold = 0.6f,
        .decision_input_coupling = 0.75f,
        .conflict_detection_gain = 0.7f,
        .feature_enhancement_strength = 0.8f,
        .filter_coupling = 0.65f,
        .predictive_enhancement_gain = 0.7f,
        .update_interval_ms = 10,
        .enable_attention_control = true,
        .enable_logging = false,
        .enable_metrics = true
    };
    return config;
}

nimcp_sensory_executive_bridge_t nimcp_sensory_executive_create(const nimcp_sensory_executive_config_t* config) {
    nimcp_sensory_executive_bridge_t bridge = (nimcp_sensory_executive_bridge_t)calloc(1, sizeof(struct nimcp_sensory_executive_bridge_struct));
    if (!bridge) return NULL;
    bridge->config = config ? *config : nimcp_sensory_executive_default_config();
    bridge->state.bridge_coherence = 1.0f;
    bridge->state.attention_capture_level = 0.5f;
    bridge->state.filter_strength = 0.5f;
    return bridge;
}

void nimcp_sensory_executive_destroy(nimcp_sensory_executive_bridge_t bridge) {
    if (!bridge) return;
    if (bridge->is_initialized) nimcp_sensory_executive_shutdown(bridge);
    free(bridge);
}

nimcp_layer_error_t nimcp_sensory_executive_init(
    nimcp_sensory_executive_bridge_t bridge,
    nimcp_layer_registry_t registry,
    nimcp_sensory_intra_t sensory,
    nimcp_executive_intra_t executive
) {
    if (!bridge || !registry) return NIMCP_LAYER_ERR_NULL_PTR;
    if (bridge->is_initialized) return NIMCP_LAYER_ERR_ALREADY_REGISTERED;
    bridge->registry = registry;
    bridge->sensory = sensory;
    bridge->executive = executive;
    bridge->is_initialized = true;
    return NIMCP_LAYER_OK;
}

nimcp_layer_error_t nimcp_sensory_executive_shutdown(nimcp_sensory_executive_bridge_t bridge) {
    if (!bridge) return NIMCP_LAYER_ERR_NULL_PTR;
    if (!bridge->is_initialized) return NIMCP_LAYER_ERR_NOT_INITIALIZED;
    bridge->is_initialized = false;
    return NIMCP_LAYER_OK;
}

nimcp_layer_error_t nimcp_sensory_executive_update(nimcp_sensory_executive_bridge_t bridge, float dt) {
    if (!bridge) return NIMCP_LAYER_ERR_NULL_PTR;
    if (!bridge->is_initialized) return NIMCP_LAYER_ERR_NOT_INITIALIZED;

    /* Salient stimuli capture attention */
    bridge->state.attention_capture_level *= (1.0f - dt * 0.01f);

    /* Decision input strength */
    bridge->state.decision_input_strength *= (1.0f - dt * 0.02f);
    bridge->state.decision_input_strength += dt * bridge->config.decision_input_coupling * 0.02f;

    /* Conflict level decay */
    bridge->state.conflict_level *= (1.0f - dt * 0.03f);

    /* Feature enhancement from attention */
    bridge->state.feature_enhancement *= (1.0f - dt * 0.02f);
    bridge->state.feature_enhancement += dt * bridge->config.feature_enhancement_strength * 0.02f;

    /* Filter strength adjustment */
    bridge->state.filter_strength *= (1.0f - dt * 0.015f);

    /* Predictive enhancement decay */
    bridge->state.predictive_enhancement *= (1.0f - dt * 0.025f);

    /* Update averages */
    bridge->stats.avg_attention_capture = bridge->stats.avg_attention_capture * 0.99f + bridge->state.attention_capture_level * 0.01f;
    bridge->stats.avg_feature_enhancement = bridge->stats.avg_feature_enhancement * 0.99f + bridge->state.feature_enhancement * 0.01f;

    return NIMCP_LAYER_OK;
}

nimcp_layer_error_t nimcp_sensory_executive_transfer_bottom_up(nimcp_sensory_executive_bridge_t bridge, const nimcp_layer_msg_t* msg) {
    if (!bridge || !msg) return NIMCP_LAYER_ERR_NULL_PTR;
    bridge->state.bottom_up_messages++;
    bridge->stats.attention_captures++;
    return NIMCP_LAYER_OK;
}

nimcp_layer_error_t nimcp_sensory_executive_transfer_top_down(nimcp_sensory_executive_bridge_t bridge, const nimcp_layer_msg_t* msg) {
    if (!bridge || !msg) return NIMCP_LAYER_ERR_NULL_PTR;
    bridge->state.top_down_messages++;
    bridge->stats.feature_enhancements++;
    return NIMCP_LAYER_OK;
}

nimcp_layer_error_t nimcp_sensory_executive_get_state(nimcp_sensory_executive_bridge_t bridge, nimcp_sensory_executive_state_t* state_out) {
    if (!bridge || !state_out) return NIMCP_LAYER_ERR_NULL_PTR;
    *state_out = bridge->state;
    return NIMCP_LAYER_OK;
}

nimcp_layer_error_t nimcp_sensory_executive_get_stats(nimcp_sensory_executive_bridge_t bridge, nimcp_sensory_executive_stats_t* stats_out) {
    if (!bridge || !stats_out) return NIMCP_LAYER_ERR_NULL_PTR;
    *stats_out = bridge->stats;
    return NIMCP_LAYER_OK;
}

float nimcp_sensory_executive_get_coherence(nimcp_sensory_executive_bridge_t bridge) {
    return bridge ? bridge->state.bridge_coherence : -1.0f;
}

nimcp_layer_error_t nimcp_sensory_executive_reset_stats(nimcp_sensory_executive_bridge_t bridge) {
    if (!bridge) return NIMCP_LAYER_ERR_NULL_PTR;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return NIMCP_LAYER_OK;
}
