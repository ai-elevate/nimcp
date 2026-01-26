/**
 * @file nimcp_integration_superhuman_bridge.c
 * @brief Integration-Superhuman Inter-Layer Bridge Implementation
 * @version 1.0.0
 * @date 2026-01-10
 */

#include "utils/bridge/nimcp_bridge_base.h"
#include "integration/inter/integration_superhuman/nimcp_integration_superhuman_bridge.h"
#include "api/nimcp_api_exception.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>
#include <stdlib.h>

#include <stddef.h>  /* for NULL */
//=============================================================================
// Health Agent Integration (Phase 8: System-Wide Health Integration)
//=============================================================================
struct nimcp_health_agent;
typedef struct nimcp_health_agent nimcp_health_agent_t;
extern void nimcp_health_agent_heartbeat_ex(nimcp_health_agent_t* agent,
                                             const char* operation,
                                             float progress);

/** Global health agent for integration_superhuman_bridge module */
static nimcp_health_agent_t* g_integration_superhuman_bridge_health_agent = NULL;

/**
 * @brief Set health agent for integration_superhuman_bridge heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
static void integration_superhuman_bridge_set_health_agent(nimcp_health_agent_t* agent) {
    g_integration_superhuman_bridge_health_agent = agent;
}

/** @brief Send heartbeat from integration_superhuman_bridge module */
static inline void integration_superhuman_bridge_heartbeat(const char* operation, float progress) {
    if (g_integration_superhuman_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_integration_superhuman_bridge_health_agent, operation, progress);
    }
}


struct nimcp_integration_superhuman_bridge_struct {
    bridge_base_t base;              /**< MUST be first: base bridge infrastructure */
    nimcp_integration_superhuman_config_t config;
    nimcp_layer_registry_t registry;
    nimcp_integration_intra_t integration;
    nimcp_superhuman_intra_t superhuman;
    nimcp_integration_superhuman_state_t state;
    nimcp_integration_superhuman_stats_t stats;
    bool is_initialized;
};

nimcp_integration_superhuman_config_t nimcp_integration_superhuman_default_config(void) {
    nimcp_integration_superhuman_config_t config = {
        .enhance_focus_coupling = 0.8f,
        .capability_coordination_strength = 0.75f,
        .enhancement_activation_threshold = 0.6f,
        .gw_percept_coupling = 0.7f,
        .unified_experience_strength = 0.8f,
        .expanded_awareness_coupling = 0.7f,
        .update_interval_ms = 10,
        .enable_conscious_enhancement = true,
        .enable_logging = false,
        .enable_metrics = true
    };
    return config;
}

nimcp_integration_superhuman_bridge_t nimcp_integration_superhuman_create(const nimcp_integration_superhuman_config_t* config) {
    nimcp_integration_superhuman_bridge_t bridge = (nimcp_integration_superhuman_bridge_t)calloc(1, sizeof(struct nimcp_integration_superhuman_bridge_struct));
    NIMCP_API_CHECK_ALLOC(bridge, "Failed to allocate integration-superhuman bridge");
    bridge->config = config ? *config : nimcp_integration_superhuman_default_config();
    bridge->state.bridge_coherence = 1.0f;
    bridge->state.enhance_focus_level = 0.5f;
    bridge->state.capability_coordination = 0.5f;
    bridge->state.unified_experience = 0.5f;
    return bridge;
}

void nimcp_integration_superhuman_destroy(nimcp_integration_superhuman_bridge_t bridge) {
    if (!bridge) return;
    if (bridge->is_initialized) nimcp_integration_superhuman_shutdown(bridge);
    free(bridge);
}

nimcp_layer_error_t nimcp_integration_superhuman_init(
    nimcp_integration_superhuman_bridge_t bridge,
    nimcp_layer_registry_t registry,
    nimcp_integration_intra_t integration,
    nimcp_superhuman_intra_t superhuman
) {
    NIMCP_API_CHECK_NULL(bridge, NIMCP_LAYER_ERR_NULL_PTR, "Bridge is NULL in integration_superhuman_init");
    NIMCP_API_CHECK_NULL(registry, NIMCP_LAYER_ERR_NULL_PTR, "Registry is NULL in integration_superhuman_init");
    NIMCP_API_CHECK(!bridge->is_initialized, NIMCP_LAYER_ERR_ALREADY_REGISTERED, "Bridge already initialized in integration_superhuman_init");
    bridge->registry = registry;
    bridge->integration = integration;
    bridge->superhuman = superhuman;
    bridge->is_initialized = true;
    return NIMCP_LAYER_OK;
}

nimcp_layer_error_t nimcp_integration_superhuman_shutdown(nimcp_integration_superhuman_bridge_t bridge) {
    NIMCP_API_CHECK_NULL(bridge, NIMCP_LAYER_ERR_NULL_PTR, "Bridge is NULL in integration_superhuman_shutdown");
    NIMCP_API_CHECK(bridge->is_initialized, NIMCP_LAYER_ERR_NOT_INITIALIZED, "Bridge not initialized in integration_superhuman_shutdown");
    bridge->is_initialized = false;
    return NIMCP_LAYER_OK;
}

nimcp_layer_error_t nimcp_integration_superhuman_update(nimcp_integration_superhuman_bridge_t bridge, float dt) {
    NIMCP_API_CHECK_NULL(bridge, NIMCP_LAYER_ERR_NULL_PTR, "Bridge is NULL in integration_superhuman_update");
    NIMCP_API_CHECK(bridge->is_initialized, NIMCP_LAYER_ERR_NOT_INITIALIZED, "Bridge not initialized in integration_superhuman_update");

    /* Conscious attention enhances perception focus */
    bridge->state.enhance_focus_level *= (1.0f - dt * 0.01f);
    bridge->state.enhance_focus_level += dt * bridge->config.enhance_focus_coupling * 0.01f;

    /* Capability coordination from global binding */
    bridge->state.capability_coordination *= (1.0f - dt * 0.02f);
    bridge->state.capability_coordination += dt * bridge->config.capability_coordination_strength * 0.02f;

    /* Enhancement activation from arousal */
    bridge->state.enhancement_activation *= (1.0f - dt * 0.015f);

    /* Enhanced percepts enter global workspace */
    bridge->state.gw_percept_strength *= (1.0f - dt * 0.02f);

    /* Unified experience from capability blending */
    bridge->state.unified_experience *= (1.0f - dt * 0.01f);
    bridge->state.unified_experience += dt * bridge->config.unified_experience_strength * 0.01f;

    /* Expanded awareness from extended sensing */
    bridge->state.expanded_awareness *= (1.0f - dt * 0.025f);

    /* Update averages */
    bridge->stats.avg_enhancement_level = bridge->stats.avg_enhancement_level * 0.99f + bridge->state.enhancement_activation * 0.01f;
    bridge->stats.avg_unified_experience = bridge->stats.avg_unified_experience * 0.99f + bridge->state.unified_experience * 0.01f;

    return NIMCP_LAYER_OK;
}

nimcp_layer_error_t nimcp_integration_superhuman_transfer_bottom_up(nimcp_integration_superhuman_bridge_t bridge, const nimcp_layer_msg_t* msg) {
    NIMCP_API_CHECK_NULL(bridge, NIMCP_LAYER_ERR_NULL_PTR, "Bridge is NULL in integration_superhuman_transfer_bottom_up");
    NIMCP_API_CHECK_NULL(msg, NIMCP_LAYER_ERR_NULL_PTR, "Message is NULL in integration_superhuman_transfer_bottom_up");
    bridge->state.bottom_up_messages++;
    bridge->stats.enhance_focuses++;
    return NIMCP_LAYER_OK;
}

nimcp_layer_error_t nimcp_integration_superhuman_transfer_top_down(nimcp_integration_superhuman_bridge_t bridge, const nimcp_layer_msg_t* msg) {
    NIMCP_API_CHECK_NULL(bridge, NIMCP_LAYER_ERR_NULL_PTR, "Bridge is NULL in integration_superhuman_transfer_top_down");
    NIMCP_API_CHECK_NULL(msg, NIMCP_LAYER_ERR_NULL_PTR, "Message is NULL in integration_superhuman_transfer_top_down");
    bridge->state.top_down_messages++;
    bridge->stats.gw_percepts++;
    return NIMCP_LAYER_OK;
}

nimcp_layer_error_t nimcp_integration_superhuman_get_state(nimcp_integration_superhuman_bridge_t bridge, nimcp_integration_superhuman_state_t* state_out) {
    NIMCP_API_CHECK_NULL(bridge, NIMCP_LAYER_ERR_NULL_PTR, "Bridge is NULL in integration_superhuman_get_state");
    NIMCP_API_CHECK_NULL(state_out, NIMCP_LAYER_ERR_NULL_PTR, "state_out is NULL in integration_superhuman_get_state");
    *state_out = bridge->state;
    return NIMCP_LAYER_OK;
}

nimcp_layer_error_t nimcp_integration_superhuman_get_stats(nimcp_integration_superhuman_bridge_t bridge, nimcp_integration_superhuman_stats_t* stats_out) {
    NIMCP_API_CHECK_NULL(bridge, NIMCP_LAYER_ERR_NULL_PTR, "Bridge is NULL in integration_superhuman_get_stats");
    NIMCP_API_CHECK_NULL(stats_out, NIMCP_LAYER_ERR_NULL_PTR, "stats_out is NULL in integration_superhuman_get_stats");
    *stats_out = bridge->stats;
    return NIMCP_LAYER_OK;
}

float nimcp_integration_superhuman_get_coherence(nimcp_integration_superhuman_bridge_t bridge) {
    return bridge ? bridge->state.bridge_coherence : -1.0f;
}

nimcp_layer_error_t nimcp_integration_superhuman_reset_stats(nimcp_integration_superhuman_bridge_t bridge) {
    NIMCP_API_CHECK_NULL(bridge, NIMCP_LAYER_ERR_NULL_PTR, "Bridge is NULL in integration_superhuman_reset_stats");
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return NIMCP_LAYER_OK;
}
