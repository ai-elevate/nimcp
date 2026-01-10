/**
 * @file nimcp_neuromod_memory_bridge.c
 * @brief Neuromodulatory-Memory Inter-Layer Bridge Implementation
 * @version 1.0.0
 * @date 2026-01-10
 */

#include "integration/inter/neuromod_memory/nimcp_neuromod_memory_bridge.h"
#include <string.h>
#include <stdlib.h>

struct nimcp_neuromod_memory_bridge_struct {
    nimcp_neuromod_memory_config_t config;
    nimcp_layer_registry_t registry;
    nimcp_neuromod_intra_t neuromod;
    nimcp_memory_intra_t memory;
    nimcp_neuromod_memory_state_t state;
    nimcp_neuromod_memory_stats_t stats;
    bool is_initialized;
};

nimcp_neuromod_memory_config_t nimcp_neuromod_memory_default_config(void) {
    nimcp_neuromod_memory_config_t config = {
        .ne_encoding_coupling = 0.9f,
        .da_reward_tagging_gain = 0.8f,
        .serotonin_pattern_coupling = 0.6f,
        .emotional_retrieval_gain = 0.7f,
        .familiarity_suppression = 0.5f,
        .prediction_error_gain = 0.8f,
        .update_interval_ms = 10,
        .enable_emotional_memory = true,
        .enable_logging = false,
        .enable_metrics = true
    };
    return config;
}

nimcp_neuromod_memory_bridge_t nimcp_neuromod_memory_create(const nimcp_neuromod_memory_config_t* config) {
    nimcp_neuromod_memory_bridge_t bridge = (nimcp_neuromod_memory_bridge_t)calloc(1, sizeof(struct nimcp_neuromod_memory_bridge_struct));
    if (!bridge) return NULL;
    bridge->config = config ? *config : nimcp_neuromod_memory_default_config();
    bridge->state.bridge_coherence = 1.0f;
    bridge->state.encoding_enhancement = 0.5f;
    bridge->state.pattern_separation_bias = 0.5f;
    return bridge;
}

void nimcp_neuromod_memory_destroy(nimcp_neuromod_memory_bridge_t bridge) {
    if (!bridge) return;
    if (bridge->is_initialized) nimcp_neuromod_memory_shutdown(bridge);
    free(bridge);
}

nimcp_layer_error_t nimcp_neuromod_memory_init(
    nimcp_neuromod_memory_bridge_t bridge,
    nimcp_layer_registry_t registry,
    nimcp_neuromod_intra_t neuromod,
    nimcp_memory_intra_t memory
) {
    if (!bridge || !registry) return NIMCP_LAYER_ERR_NULL_PTR;
    if (bridge->is_initialized) return NIMCP_LAYER_ERR_ALREADY_REGISTERED;
    bridge->registry = registry;
    bridge->neuromod = neuromod;
    bridge->memory = memory;
    bridge->is_initialized = true;
    return NIMCP_LAYER_OK;
}

nimcp_layer_error_t nimcp_neuromod_memory_shutdown(nimcp_neuromod_memory_bridge_t bridge) {
    if (!bridge) return NIMCP_LAYER_ERR_NULL_PTR;
    if (!bridge->is_initialized) return NIMCP_LAYER_ERR_NOT_INITIALIZED;
    bridge->is_initialized = false;
    return NIMCP_LAYER_OK;
}

nimcp_layer_error_t nimcp_neuromod_memory_update(nimcp_neuromod_memory_bridge_t bridge, float dt) {
    if (!bridge) return NIMCP_LAYER_ERR_NULL_PTR;
    if (!bridge->is_initialized) return NIMCP_LAYER_ERR_NOT_INITIALIZED;

    /* NE levels enhance encoding (flashbulb memory) */
    bridge->state.encoding_enhancement *= (1.0f - dt * 0.01f);
    bridge->state.encoding_enhancement += dt * bridge->config.ne_encoding_coupling * 0.01f;

    /* DA reward tagging */
    bridge->state.reward_tag_strength *= (1.0f - dt * 0.02f);
    bridge->state.reward_tag_strength += dt * bridge->config.da_reward_tagging_gain * 0.02f;

    /* Pattern separation/completion balance via serotonin */
    bridge->state.pattern_separation_bias *= (1.0f - dt * 0.015f);

    /* Emotional activation decay */
    bridge->state.emotional_activation *= (1.0f - dt * 0.03f);

    /* Prediction error decay */
    bridge->state.prediction_error *= (1.0f - dt * 0.05f);

    /* Update averages */
    bridge->stats.avg_encoding_strength = bridge->stats.avg_encoding_strength * 0.99f + bridge->state.encoding_enhancement * 0.01f;
    bridge->stats.avg_emotional_activation = bridge->stats.avg_emotional_activation * 0.99f + bridge->state.emotional_activation * 0.01f;

    return NIMCP_LAYER_OK;
}

nimcp_layer_error_t nimcp_neuromod_memory_transfer_bottom_up(nimcp_neuromod_memory_bridge_t bridge, const nimcp_layer_msg_t* msg) {
    if (!bridge || !msg) return NIMCP_LAYER_ERR_NULL_PTR;
    bridge->state.bottom_up_messages++;
    bridge->stats.encoding_enhancements++;
    return NIMCP_LAYER_OK;
}

nimcp_layer_error_t nimcp_neuromod_memory_transfer_top_down(nimcp_neuromod_memory_bridge_t bridge, const nimcp_layer_msg_t* msg) {
    if (!bridge || !msg) return NIMCP_LAYER_ERR_NULL_PTR;
    bridge->state.top_down_messages++;
    bridge->stats.emotional_activations++;
    return NIMCP_LAYER_OK;
}

nimcp_layer_error_t nimcp_neuromod_memory_get_state(nimcp_neuromod_memory_bridge_t bridge, nimcp_neuromod_memory_state_t* state_out) {
    if (!bridge || !state_out) return NIMCP_LAYER_ERR_NULL_PTR;
    *state_out = bridge->state;
    return NIMCP_LAYER_OK;
}

nimcp_layer_error_t nimcp_neuromod_memory_get_stats(nimcp_neuromod_memory_bridge_t bridge, nimcp_neuromod_memory_stats_t* stats_out) {
    if (!bridge || !stats_out) return NIMCP_LAYER_ERR_NULL_PTR;
    *stats_out = bridge->stats;
    return NIMCP_LAYER_OK;
}

float nimcp_neuromod_memory_get_coherence(nimcp_neuromod_memory_bridge_t bridge) {
    return bridge ? bridge->state.bridge_coherence : -1.0f;
}

nimcp_layer_error_t nimcp_neuromod_memory_reset_stats(nimcp_neuromod_memory_bridge_t bridge) {
    if (!bridge) return NIMCP_LAYER_ERR_NULL_PTR;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return NIMCP_LAYER_OK;
}
