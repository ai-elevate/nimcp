/**
 * @file nimcp_biology_neuromod_bridge.c
 * @brief Biology-Neuromodulatory Inter-Layer Bridge Implementation
 * @version 1.0.0
 * @date 2026-01-10
 */

#include "integration/inter/biology_neuromod/nimcp_biology_neuromod_bridge.h"
#include <string.h>
#include <stdlib.h>

struct nimcp_biology_neuromod_bridge_struct {
    nimcp_biology_neuromod_config_t config;
    nimcp_layer_registry_t registry;
    nimcp_biology_intra_t biology;
    nimcp_neuromod_intra_t neuromod;
    nimcp_biology_neuromod_state_t state;
    nimcp_biology_neuromod_stats_t stats;
    bool is_initialized;
};

nimcp_biology_neuromod_config_t nimcp_biology_neuromod_default_config(void) {
    nimcp_biology_neuromod_config_t config = {
        .receptor_density_coupling = 0.8f,
        .neurogenesis_integration_rate = 0.5f,
        .bdnf_expression_gain = 0.7f,
        .plasticity_gene_sensitivity = 0.6f,
        .update_interval_ms = 10,
        .enable_activity_dependent_plasticity = true,
        .enable_logging = false,
        .enable_metrics = true
    };
    return config;
}

nimcp_biology_neuromod_bridge_t nimcp_biology_neuromod_create(const nimcp_biology_neuromod_config_t* config) {
    nimcp_biology_neuromod_bridge_t bridge = (nimcp_biology_neuromod_bridge_t)calloc(1, sizeof(struct nimcp_biology_neuromod_bridge_struct));
    if (!bridge) return NULL;
    bridge->config = config ? *config : nimcp_biology_neuromod_default_config();
    bridge->state.bridge_coherence = 1.0f;
    bridge->state.receptor_density_level = 0.5f;
    bridge->state.energy_capacity = 1.0f;
    return bridge;
}

void nimcp_biology_neuromod_destroy(nimcp_biology_neuromod_bridge_t bridge) {
    if (!bridge) return;
    if (bridge->is_initialized) nimcp_biology_neuromod_shutdown(bridge);
    free(bridge);
}

nimcp_layer_error_t nimcp_biology_neuromod_init(
    nimcp_biology_neuromod_bridge_t bridge,
    nimcp_layer_registry_t registry,
    nimcp_biology_intra_t biology,
    nimcp_neuromod_intra_t neuromod
) {
    if (!bridge || !registry) return NIMCP_LAYER_ERR_NULL_PTR;
    if (bridge->is_initialized) return NIMCP_LAYER_ERR_ALREADY_REGISTERED;
    bridge->registry = registry;
    bridge->biology = biology;
    bridge->neuromod = neuromod;
    bridge->is_initialized = true;
    return NIMCP_LAYER_OK;
}

nimcp_layer_error_t nimcp_biology_neuromod_shutdown(nimcp_biology_neuromod_bridge_t bridge) {
    if (!bridge) return NIMCP_LAYER_ERR_NULL_PTR;
    if (!bridge->is_initialized) return NIMCP_LAYER_ERR_NOT_INITIALIZED;
    bridge->is_initialized = false;
    return NIMCP_LAYER_OK;
}

nimcp_layer_error_t nimcp_biology_neuromod_update(nimcp_biology_neuromod_bridge_t bridge, float dt) {
    if (!bridge) return NIMCP_LAYER_ERR_NULL_PTR;
    if (!bridge->is_initialized) return NIMCP_LAYER_ERR_NOT_INITIALIZED;

    /* Gene expression affects receptor density */
    bridge->state.receptor_density_level *= (1.0f - dt * 0.01f);
    bridge->state.receptor_density_level += dt * bridge->config.receptor_density_coupling * 0.01f;

    /* BDNF expression depends on activity */
    bridge->state.bdnf_expression_level *= (1.0f - dt * 0.02f);
    bridge->state.bdnf_expression_level += dt * bridge->config.bdnf_expression_gain * 0.02f;

    /* Plasticity gene activity */
    bridge->state.plasticity_gene_activity *= (1.0f - dt * 0.015f);

    /* Energy capacity decay */
    bridge->state.energy_capacity *= (1.0f - dt * 0.005f);

    /* Update averages */
    bridge->stats.avg_receptor_density = bridge->stats.avg_receptor_density * 0.99f + bridge->state.receptor_density_level * 0.01f;
    bridge->stats.avg_bdnf_level = bridge->stats.avg_bdnf_level * 0.99f + bridge->state.bdnf_expression_level * 0.01f;

    return NIMCP_LAYER_OK;
}

nimcp_layer_error_t nimcp_biology_neuromod_transfer_bottom_up(nimcp_biology_neuromod_bridge_t bridge, const nimcp_layer_msg_t* msg) {
    if (!bridge || !msg) return NIMCP_LAYER_ERR_NULL_PTR;
    bridge->state.bottom_up_messages++;
    bridge->stats.receptor_density_changes++;
    return NIMCP_LAYER_OK;
}

nimcp_layer_error_t nimcp_biology_neuromod_transfer_top_down(nimcp_biology_neuromod_bridge_t bridge, const nimcp_layer_msg_t* msg) {
    if (!bridge || !msg) return NIMCP_LAYER_ERR_NULL_PTR;
    bridge->state.top_down_messages++;
    bridge->stats.bdnf_expressions++;
    return NIMCP_LAYER_OK;
}

nimcp_layer_error_t nimcp_biology_neuromod_get_state(nimcp_biology_neuromod_bridge_t bridge, nimcp_biology_neuromod_state_t* state_out) {
    if (!bridge || !state_out) return NIMCP_LAYER_ERR_NULL_PTR;
    *state_out = bridge->state;
    return NIMCP_LAYER_OK;
}

nimcp_layer_error_t nimcp_biology_neuromod_get_stats(nimcp_biology_neuromod_bridge_t bridge, nimcp_biology_neuromod_stats_t* stats_out) {
    if (!bridge || !stats_out) return NIMCP_LAYER_ERR_NULL_PTR;
    *stats_out = bridge->stats;
    return NIMCP_LAYER_OK;
}

float nimcp_biology_neuromod_get_coherence(nimcp_biology_neuromod_bridge_t bridge) {
    return bridge ? bridge->state.bridge_coherence : -1.0f;
}

nimcp_layer_error_t nimcp_biology_neuromod_reset_stats(nimcp_biology_neuromod_bridge_t bridge) {
    if (!bridge) return NIMCP_LAYER_ERR_NULL_PTR;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return NIMCP_LAYER_OK;
}
