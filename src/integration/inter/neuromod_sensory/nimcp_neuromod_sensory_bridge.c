/**
 * @file nimcp_neuromod_sensory_bridge.c
 * @brief Neuromodulatory-Sensory Inter-Layer Bridge Implementation
 * @version 1.0.0
 * @date 2026-01-10
 */

#include "integration/inter/neuromod_sensory/nimcp_neuromod_sensory_bridge.h"
#include <string.h>
#include <stdlib.h>

struct nimcp_neuromod_sensory_bridge_struct {
    nimcp_neuromod_sensory_config_t config;
    nimcp_layer_registry_t registry;
    nimcp_neuromod_intra_t neuromod;
    nimcp_sensory_intra_t sensory;
    nimcp_neuromod_sensory_state_t state;
    nimcp_neuromod_sensory_stats_t stats;
    bool is_initialized;
};

nimcp_neuromod_sensory_config_t nimcp_neuromod_sensory_default_config(void) {
    nimcp_neuromod_sensory_config_t config = {
        .ne_gain_coupling = 0.8f,
        .da_salience_coupling = 0.7f,
        .arousal_threshold_coupling = 0.6f,
        .novelty_lc_gain = 0.9f,
        .reward_vta_gain = 0.8f,
        .update_interval_ms = 10,
        .enable_gain_control = true,
        .enable_logging = false,
        .enable_metrics = true
    };
    return config;
}

nimcp_neuromod_sensory_bridge_t nimcp_neuromod_sensory_create(const nimcp_neuromod_sensory_config_t* config) {
    nimcp_neuromod_sensory_bridge_t bridge = (nimcp_neuromod_sensory_bridge_t)calloc(1, sizeof(struct nimcp_neuromod_sensory_bridge_struct));
    if (!bridge) return NULL;
    bridge->config = config ? *config : nimcp_neuromod_sensory_default_config();
    bridge->state.bridge_coherence = 1.0f;
    bridge->state.sensory_gain = 1.0f;
    bridge->state.threshold_level = 0.5f;
    return bridge;
}

void nimcp_neuromod_sensory_destroy(nimcp_neuromod_sensory_bridge_t bridge) {
    if (!bridge) return;
    if (bridge->is_initialized) nimcp_neuromod_sensory_shutdown(bridge);
    free(bridge);
}

nimcp_layer_error_t nimcp_neuromod_sensory_init(
    nimcp_neuromod_sensory_bridge_t bridge,
    nimcp_layer_registry_t registry,
    nimcp_neuromod_intra_t neuromod,
    nimcp_sensory_intra_t sensory
) {
    if (!bridge || !registry) return NIMCP_LAYER_ERR_NULL_PTR;
    if (bridge->is_initialized) return NIMCP_LAYER_ERR_ALREADY_REGISTERED;
    bridge->registry = registry;
    bridge->neuromod = neuromod;
    bridge->sensory = sensory;
    bridge->is_initialized = true;
    return NIMCP_LAYER_OK;
}

nimcp_layer_error_t nimcp_neuromod_sensory_shutdown(nimcp_neuromod_sensory_bridge_t bridge) {
    if (!bridge) return NIMCP_LAYER_ERR_NULL_PTR;
    if (!bridge->is_initialized) return NIMCP_LAYER_ERR_NOT_INITIALIZED;
    bridge->is_initialized = false;
    return NIMCP_LAYER_OK;
}

nimcp_layer_error_t nimcp_neuromod_sensory_update(nimcp_neuromod_sensory_bridge_t bridge, float dt) {
    if (!bridge) return NIMCP_LAYER_ERR_NULL_PTR;
    if (!bridge->is_initialized) return NIMCP_LAYER_ERR_NOT_INITIALIZED;

    /* NE levels affect sensory gain */
    bridge->state.sensory_gain *= (1.0f - dt * 0.01f);
    bridge->state.sensory_gain += dt * bridge->config.ne_gain_coupling * 0.01f;

    /* DA levels affect salience boost */
    bridge->state.salience_boost *= (1.0f - dt * 0.02f);
    bridge->state.salience_boost += dt * bridge->config.da_salience_coupling * 0.02f;

    /* Threshold adjustment based on arousal */
    bridge->state.threshold_level *= (1.0f - dt * 0.015f);

    /* Novelty signal decay */
    bridge->state.novelty_signal *= (1.0f - dt * 0.05f);

    /* Reward signal decay */
    bridge->state.reward_signal *= (1.0f - dt * 0.03f);

    /* Update averages */
    bridge->stats.avg_sensory_gain = bridge->stats.avg_sensory_gain * 0.99f + bridge->state.sensory_gain * 0.01f;
    bridge->stats.avg_salience = bridge->stats.avg_salience * 0.99f + bridge->state.salience_boost * 0.01f;

    return NIMCP_LAYER_OK;
}

nimcp_layer_error_t nimcp_neuromod_sensory_transfer_bottom_up(nimcp_neuromod_sensory_bridge_t bridge, const nimcp_layer_msg_t* msg) {
    if (!bridge || !msg) return NIMCP_LAYER_ERR_NULL_PTR;
    bridge->state.bottom_up_messages++;
    bridge->stats.gain_modulations++;
    return NIMCP_LAYER_OK;
}

nimcp_layer_error_t nimcp_neuromod_sensory_transfer_top_down(nimcp_neuromod_sensory_bridge_t bridge, const nimcp_layer_msg_t* msg) {
    if (!bridge || !msg) return NIMCP_LAYER_ERR_NULL_PTR;
    bridge->state.top_down_messages++;
    bridge->stats.novel_stimuli++;
    return NIMCP_LAYER_OK;
}

nimcp_layer_error_t nimcp_neuromod_sensory_get_state(nimcp_neuromod_sensory_bridge_t bridge, nimcp_neuromod_sensory_state_t* state_out) {
    if (!bridge || !state_out) return NIMCP_LAYER_ERR_NULL_PTR;
    *state_out = bridge->state;
    return NIMCP_LAYER_OK;
}

nimcp_layer_error_t nimcp_neuromod_sensory_get_stats(nimcp_neuromod_sensory_bridge_t bridge, nimcp_neuromod_sensory_stats_t* stats_out) {
    if (!bridge || !stats_out) return NIMCP_LAYER_ERR_NULL_PTR;
    *stats_out = bridge->stats;
    return NIMCP_LAYER_OK;
}

float nimcp_neuromod_sensory_get_coherence(nimcp_neuromod_sensory_bridge_t bridge) {
    return bridge ? bridge->state.bridge_coherence : -1.0f;
}

nimcp_layer_error_t nimcp_neuromod_sensory_reset_stats(nimcp_neuromod_sensory_bridge_t bridge) {
    if (!bridge) return NIMCP_LAYER_ERR_NULL_PTR;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return NIMCP_LAYER_OK;
}
