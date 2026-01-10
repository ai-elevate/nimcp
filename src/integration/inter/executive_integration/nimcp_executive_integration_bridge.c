/**
 * @file nimcp_executive_integration_bridge.c
 * @brief Executive-Integration Inter-Layer Bridge Implementation
 * @version 1.0.0
 * @date 2026-01-10
 */

#include "integration/inter/executive_integration/nimcp_executive_integration_bridge.h"
#include <string.h>
#include <stdlib.h>

struct nimcp_executive_integration_bridge_struct {
    nimcp_executive_integration_config_t config;
    nimcp_layer_registry_t registry;
    nimcp_executive_intra_t executive;
    nimcp_integration_intra_t integration;
    nimcp_executive_integration_state_t state;
    nimcp_executive_integration_stats_t stats;
    bool is_initialized;
};

nimcp_executive_integration_config_t nimcp_executive_integration_default_config(void) {
    nimcp_executive_integration_config_t config = {
        .decision_awareness_coupling = 0.8f,
        .goal_broadcast_strength = 0.75f,
        .conflict_arousal_gain = 0.7f,
        .exec_priority_coupling = 0.7f,
        .cognitive_capacity_coupling = 0.65f,
        .decision_coherence_coupling = 0.75f,
        .update_interval_ms = 10,
        .enable_conscious_control = true,
        .enable_logging = false,
        .enable_metrics = true
    };
    return config;
}

nimcp_executive_integration_bridge_t nimcp_executive_integration_create(const nimcp_executive_integration_config_t* config) {
    nimcp_executive_integration_bridge_t bridge = (nimcp_executive_integration_bridge_t)calloc(1, sizeof(struct nimcp_executive_integration_bridge_struct));
    if (!bridge) return NULL;
    bridge->config = config ? *config : nimcp_executive_integration_default_config();
    bridge->state.bridge_coherence = 1.0f;
    bridge->state.decision_awareness = 0.5f;
    bridge->state.cognitive_capacity = 0.8f;
    bridge->state.decision_coherence = 0.7f;
    return bridge;
}

void nimcp_executive_integration_destroy(nimcp_executive_integration_bridge_t bridge) {
    if (!bridge) return;
    if (bridge->is_initialized) nimcp_executive_integration_shutdown(bridge);
    free(bridge);
}

nimcp_layer_error_t nimcp_executive_integration_init(
    nimcp_executive_integration_bridge_t bridge,
    nimcp_layer_registry_t registry,
    nimcp_executive_intra_t executive,
    nimcp_integration_intra_t integration
) {
    if (!bridge || !registry) return NIMCP_LAYER_ERR_NULL_PTR;
    if (bridge->is_initialized) return NIMCP_LAYER_ERR_ALREADY_REGISTERED;
    bridge->registry = registry;
    bridge->executive = executive;
    bridge->integration = integration;
    bridge->is_initialized = true;
    return NIMCP_LAYER_OK;
}

nimcp_layer_error_t nimcp_executive_integration_shutdown(nimcp_executive_integration_bridge_t bridge) {
    if (!bridge) return NIMCP_LAYER_ERR_NULL_PTR;
    if (!bridge->is_initialized) return NIMCP_LAYER_ERR_NOT_INITIALIZED;
    bridge->is_initialized = false;
    return NIMCP_LAYER_OK;
}

nimcp_layer_error_t nimcp_executive_integration_update(nimcp_executive_integration_bridge_t bridge, float dt) {
    if (!bridge) return NIMCP_LAYER_ERR_NULL_PTR;
    if (!bridge->is_initialized) return NIMCP_LAYER_ERR_NOT_INITIALIZED;

    /* Decision awareness for consciousness */
    bridge->state.decision_awareness *= (1.0f - dt * 0.01f);
    bridge->state.decision_awareness += dt * bridge->config.decision_awareness_coupling * 0.01f;

    /* Goal broadcast to global workspace */
    bridge->state.goal_broadcast_level *= (1.0f - dt * 0.02f);
    bridge->state.goal_broadcast_level += dt * bridge->config.goal_broadcast_strength * 0.02f;

    /* Conflict arousal */
    bridge->state.conflict_arousal *= (1.0f - dt * 0.03f);

    /* Executive priority from global state */
    bridge->state.executive_priority *= (1.0f - dt * 0.015f);

    /* Cognitive capacity from arousal */
    bridge->state.cognitive_capacity *= (1.0f - dt * 0.01f);
    bridge->state.cognitive_capacity += dt * bridge->config.cognitive_capacity_coupling * 0.01f;

    /* Decision coherence from binding */
    bridge->state.decision_coherence *= (1.0f - dt * 0.02f);
    bridge->state.decision_coherence += dt * bridge->config.decision_coherence_coupling * 0.02f;

    /* Update averages */
    bridge->stats.avg_decision_awareness = bridge->stats.avg_decision_awareness * 0.99f + bridge->state.decision_awareness * 0.01f;
    bridge->stats.avg_cognitive_capacity = bridge->stats.avg_cognitive_capacity * 0.99f + bridge->state.cognitive_capacity * 0.01f;

    return NIMCP_LAYER_OK;
}

nimcp_layer_error_t nimcp_executive_integration_transfer_bottom_up(nimcp_executive_integration_bridge_t bridge, const nimcp_layer_msg_t* msg) {
    if (!bridge || !msg) return NIMCP_LAYER_ERR_NULL_PTR;
    bridge->state.bottom_up_messages++;
    bridge->stats.decision_awarenesses++;
    return NIMCP_LAYER_OK;
}

nimcp_layer_error_t nimcp_executive_integration_transfer_top_down(nimcp_executive_integration_bridge_t bridge, const nimcp_layer_msg_t* msg) {
    if (!bridge || !msg) return NIMCP_LAYER_ERR_NULL_PTR;
    bridge->state.top_down_messages++;
    bridge->stats.exec_priorities++;
    return NIMCP_LAYER_OK;
}

nimcp_layer_error_t nimcp_executive_integration_get_state(nimcp_executive_integration_bridge_t bridge, nimcp_executive_integration_state_t* state_out) {
    if (!bridge || !state_out) return NIMCP_LAYER_ERR_NULL_PTR;
    *state_out = bridge->state;
    return NIMCP_LAYER_OK;
}

nimcp_layer_error_t nimcp_executive_integration_get_stats(nimcp_executive_integration_bridge_t bridge, nimcp_executive_integration_stats_t* stats_out) {
    if (!bridge || !stats_out) return NIMCP_LAYER_ERR_NULL_PTR;
    *stats_out = bridge->stats;
    return NIMCP_LAYER_OK;
}

float nimcp_executive_integration_get_coherence(nimcp_executive_integration_bridge_t bridge) {
    return bridge ? bridge->state.bridge_coherence : -1.0f;
}

nimcp_layer_error_t nimcp_executive_integration_reset_stats(nimcp_executive_integration_bridge_t bridge) {
    if (!bridge) return NIMCP_LAYER_ERR_NULL_PTR;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return NIMCP_LAYER_OK;
}
