/**
 * @file nimcp_sensory_memory_bridge.c
 * @brief Sensory-Memory Inter-Layer Bridge Implementation
 * @version 1.0.0
 * @date 2026-01-10
 */

#include "integration/inter/sensory_memory/nimcp_sensory_memory_bridge.h"
#include <string.h>
#include <stdlib.h>

struct nimcp_sensory_memory_bridge_struct {
    nimcp_sensory_memory_config_t config;
    nimcp_layer_registry_t registry;
    nimcp_sensory_intra_t sensory;
    nimcp_memory_intra_t memory;
    nimcp_sensory_memory_state_t state;
    nimcp_sensory_memory_stats_t stats;
    bool is_initialized;
};

nimcp_sensory_memory_config_t nimcp_sensory_memory_default_config(void) {
    nimcp_sensory_memory_config_t config = {
        .feature_encoding_coupling = 0.8f,
        .multimodal_binding_strength = 0.75f,
        .spatial_encoding_gain = 0.7f,
        .expectation_coupling = 0.65f,
        .priming_strength = 0.6f,
        .context_bias_coupling = 0.55f,
        .update_interval_ms = 10,
        .enable_predictive_coding = true,
        .enable_logging = false,
        .enable_metrics = true
    };
    return config;
}

nimcp_sensory_memory_bridge_t nimcp_sensory_memory_create(const nimcp_sensory_memory_config_t* config) {
    nimcp_sensory_memory_bridge_t bridge = (nimcp_sensory_memory_bridge_t)calloc(1, sizeof(struct nimcp_sensory_memory_bridge_struct));
    if (!bridge) return NULL;
    bridge->config = config ? *config : nimcp_sensory_memory_default_config();
    bridge->state.bridge_coherence = 1.0f;
    bridge->state.feature_encoding_level = 0.5f;
    bridge->state.multimodal_binding = 0.5f;
    return bridge;
}

void nimcp_sensory_memory_destroy(nimcp_sensory_memory_bridge_t bridge) {
    if (!bridge) return;
    if (bridge->is_initialized) nimcp_sensory_memory_shutdown(bridge);
    free(bridge);
}

nimcp_layer_error_t nimcp_sensory_memory_init(
    nimcp_sensory_memory_bridge_t bridge,
    nimcp_layer_registry_t registry,
    nimcp_sensory_intra_t sensory,
    nimcp_memory_intra_t memory
) {
    if (!bridge || !registry) return NIMCP_LAYER_ERR_NULL_PTR;
    if (bridge->is_initialized) return NIMCP_LAYER_ERR_ALREADY_REGISTERED;
    bridge->registry = registry;
    bridge->sensory = sensory;
    bridge->memory = memory;
    bridge->is_initialized = true;
    return NIMCP_LAYER_OK;
}

nimcp_layer_error_t nimcp_sensory_memory_shutdown(nimcp_sensory_memory_bridge_t bridge) {
    if (!bridge) return NIMCP_LAYER_ERR_NULL_PTR;
    if (!bridge->is_initialized) return NIMCP_LAYER_ERR_NOT_INITIALIZED;
    bridge->is_initialized = false;
    return NIMCP_LAYER_OK;
}

nimcp_layer_error_t nimcp_sensory_memory_update(nimcp_sensory_memory_bridge_t bridge, float dt) {
    if (!bridge) return NIMCP_LAYER_ERR_NULL_PTR;
    if (!bridge->is_initialized) return NIMCP_LAYER_ERR_NOT_INITIALIZED;

    /* Feature encoding to hippocampus */
    bridge->state.feature_encoding_level *= (1.0f - dt * 0.01f);
    bridge->state.feature_encoding_level += dt * bridge->config.feature_encoding_coupling * 0.01f;

    /* Multi-modal binding for episodic context */
    bridge->state.multimodal_binding *= (1.0f - dt * 0.02f);
    bridge->state.multimodal_binding += dt * bridge->config.multimodal_binding_strength * 0.02f;

    /* Spatial encoding for place cells */
    bridge->state.spatial_encoding *= (1.0f - dt * 0.015f);

    /* Expectation strength decay */
    bridge->state.expectation_strength *= (1.0f - dt * 0.03f);

    /* Priming level decay */
    bridge->state.priming_level *= (1.0f - dt * 0.04f);

    /* Context bias decay */
    bridge->state.context_bias *= (1.0f - dt * 0.025f);

    /* Update averages */
    bridge->stats.avg_encoding_strength = bridge->stats.avg_encoding_strength * 0.99f + bridge->state.feature_encoding_level * 0.01f;
    bridge->stats.avg_expectation_match = bridge->stats.avg_expectation_match * 0.99f + bridge->state.expectation_strength * 0.01f;

    return NIMCP_LAYER_OK;
}

nimcp_layer_error_t nimcp_sensory_memory_transfer_bottom_up(nimcp_sensory_memory_bridge_t bridge, const nimcp_layer_msg_t* msg) {
    if (!bridge || !msg) return NIMCP_LAYER_ERR_NULL_PTR;
    bridge->state.bottom_up_messages++;
    bridge->stats.feature_encodings++;
    return NIMCP_LAYER_OK;
}

nimcp_layer_error_t nimcp_sensory_memory_transfer_top_down(nimcp_sensory_memory_bridge_t bridge, const nimcp_layer_msg_t* msg) {
    if (!bridge || !msg) return NIMCP_LAYER_ERR_NULL_PTR;
    bridge->state.top_down_messages++;
    bridge->stats.expectations++;
    return NIMCP_LAYER_OK;
}

nimcp_layer_error_t nimcp_sensory_memory_get_state(nimcp_sensory_memory_bridge_t bridge, nimcp_sensory_memory_state_t* state_out) {
    if (!bridge || !state_out) return NIMCP_LAYER_ERR_NULL_PTR;
    *state_out = bridge->state;
    return NIMCP_LAYER_OK;
}

nimcp_layer_error_t nimcp_sensory_memory_get_stats(nimcp_sensory_memory_bridge_t bridge, nimcp_sensory_memory_stats_t* stats_out) {
    if (!bridge || !stats_out) return NIMCP_LAYER_ERR_NULL_PTR;
    *stats_out = bridge->stats;
    return NIMCP_LAYER_OK;
}

float nimcp_sensory_memory_get_coherence(nimcp_sensory_memory_bridge_t bridge) {
    return bridge ? bridge->state.bridge_coherence : -1.0f;
}

nimcp_layer_error_t nimcp_sensory_memory_reset_stats(nimcp_sensory_memory_bridge_t bridge) {
    if (!bridge) return NIMCP_LAYER_ERR_NULL_PTR;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return NIMCP_LAYER_OK;
}
