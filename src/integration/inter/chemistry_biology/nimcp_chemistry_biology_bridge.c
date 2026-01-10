/**
 * @file nimcp_chemistry_biology_bridge.c
 * @brief Chemistry-Biology Inter-Layer Bridge Implementation
 * @version 1.0.0
 * @date 2026-01-10
 */

#include "integration/inter/chemistry_biology/nimcp_chemistry_biology_bridge.h"
#include <string.h>
#include <stdlib.h>

struct nimcp_chemistry_biology_bridge_struct {
    nimcp_chemistry_biology_config_t config;
    nimcp_layer_registry_t registry;
    nimcp_chemistry_intra_t chemistry;
    nimcp_biology_intra_t biology;
    nimcp_chemistry_biology_state_t state;
    nimcp_chemistry_biology_stats_t stats;
    bool is_initialized;
};

nimcp_chemistry_biology_config_t nimcp_chemistry_biology_default_config(void) {
    nimcp_chemistry_biology_config_t config = {
        .receptor_coupling_strength = 0.8f,
        .ph_sensitivity = 0.9f,
        .no_signaling_gain = 0.7f,
        .protein_synthesis_rate = 0.5f,
        .update_interval_ms = 10,
        .enable_receptor_dynamics = true,
        .enable_logging = false,
        .enable_metrics = true
    };
    return config;
}

nimcp_chemistry_biology_bridge_t nimcp_chemistry_biology_create(const nimcp_chemistry_biology_config_t* config) {
    nimcp_chemistry_biology_bridge_t bridge = (nimcp_chemistry_biology_bridge_t)calloc(1, sizeof(struct nimcp_chemistry_biology_bridge_struct));
    if (!bridge) return NULL;
    bridge->config = config ? *config : nimcp_chemistry_biology_default_config();
    bridge->state.bridge_coherence = 1.0f;
    return bridge;
}

void nimcp_chemistry_biology_destroy(nimcp_chemistry_biology_bridge_t bridge) {
    if (!bridge) return;
    if (bridge->is_initialized) nimcp_chemistry_biology_shutdown(bridge);
    free(bridge);
}

nimcp_layer_error_t nimcp_chemistry_biology_init(
    nimcp_chemistry_biology_bridge_t bridge,
    nimcp_layer_registry_t registry,
    nimcp_chemistry_intra_t chemistry,
    nimcp_biology_intra_t biology
) {
    if (!bridge || !registry) return NIMCP_LAYER_ERR_NULL_PTR;
    if (bridge->is_initialized) return NIMCP_LAYER_ERR_ALREADY_REGISTERED;
    bridge->registry = registry;
    bridge->chemistry = chemistry;
    bridge->biology = biology;
    bridge->is_initialized = true;
    return NIMCP_LAYER_OK;
}

nimcp_layer_error_t nimcp_chemistry_biology_shutdown(nimcp_chemistry_biology_bridge_t bridge) {
    if (!bridge) return NIMCP_LAYER_ERR_NULL_PTR;
    if (!bridge->is_initialized) return NIMCP_LAYER_ERR_NOT_INITIALIZED;
    bridge->is_initialized = false;
    return NIMCP_LAYER_OK;
}

nimcp_layer_error_t nimcp_chemistry_biology_update(nimcp_chemistry_biology_bridge_t bridge, float dt) {
    if (!bridge) return NIMCP_LAYER_ERR_NULL_PTR;
    if (!bridge->is_initialized) return NIMCP_LAYER_ERR_NOT_INITIALIZED;

    /* Receptor activation decays */
    bridge->state.receptor_activation_level *= (1.0f - dt * 0.1f);

    /* pH effects on protein function */
    bridge->state.ph_effect_magnitude *= (1.0f - dt * 0.05f);

    /* NO signal diffuses and decays */
    bridge->state.no_signal_strength *= (1.0f - dt * 0.2f);

    /* Update stats */
    bridge->stats.avg_receptor_activation = bridge->stats.avg_receptor_activation * 0.99f + bridge->state.receptor_activation_level * 0.01f;
    bridge->stats.avg_synthesis_rate = bridge->stats.avg_synthesis_rate * 0.99f + bridge->state.protein_synthesis_load * 0.01f;

    return NIMCP_LAYER_OK;
}

nimcp_layer_error_t nimcp_chemistry_biology_transfer_bottom_up(nimcp_chemistry_biology_bridge_t bridge, const nimcp_layer_msg_t* msg) {
    if (!bridge || !msg) return NIMCP_LAYER_ERR_NULL_PTR;
    if (!bridge->is_initialized) return NIMCP_LAYER_ERR_NOT_INITIALIZED;
    bridge->state.bottom_up_messages++;
    bridge->stats.receptor_activations++;
    return NIMCP_LAYER_OK;
}

nimcp_layer_error_t nimcp_chemistry_biology_transfer_top_down(nimcp_chemistry_biology_bridge_t bridge, const nimcp_layer_msg_t* msg) {
    if (!bridge || !msg) return NIMCP_LAYER_ERR_NULL_PTR;
    if (!bridge->is_initialized) return NIMCP_LAYER_ERR_NOT_INITIALIZED;
    bridge->state.top_down_messages++;
    bridge->stats.protein_requests++;
    return NIMCP_LAYER_OK;
}

nimcp_layer_error_t nimcp_chemistry_biology_get_state(nimcp_chemistry_biology_bridge_t bridge, nimcp_chemistry_biology_state_t* state_out) {
    if (!bridge || !state_out) return NIMCP_LAYER_ERR_NULL_PTR;
    *state_out = bridge->state;
    return NIMCP_LAYER_OK;
}

nimcp_layer_error_t nimcp_chemistry_biology_get_stats(nimcp_chemistry_biology_bridge_t bridge, nimcp_chemistry_biology_stats_t* stats_out) {
    if (!bridge || !stats_out) return NIMCP_LAYER_ERR_NULL_PTR;
    *stats_out = bridge->stats;
    return NIMCP_LAYER_OK;
}

float nimcp_chemistry_biology_get_coherence(nimcp_chemistry_biology_bridge_t bridge) {
    return bridge ? bridge->state.bridge_coherence : -1.0f;
}

nimcp_layer_error_t nimcp_chemistry_biology_reset_stats(nimcp_chemistry_biology_bridge_t bridge) {
    if (!bridge) return NIMCP_LAYER_ERR_NULL_PTR;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return NIMCP_LAYER_OK;
}
