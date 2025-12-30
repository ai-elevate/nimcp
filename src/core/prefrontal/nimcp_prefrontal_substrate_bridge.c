/**
 * @file nimcp_prefrontal_substrate_bridge.c
 * @brief Prefrontal Cortex-Neural Substrate Bridge Implementation
 */

#include "core/prefrontal/nimcp_prefrontal_substrate_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include <string.h>

struct prefrontal_substrate_bridge {
    void* prefrontal;
    neural_substrate_t* substrate;
    prefrontal_substrate_config_t config;
    prefrontal_substrate_effects_t effects;
    bio_router_t* router;
    bool bio_async_connected;
    uint64_t update_count;
};

static float clamp_f(float v, float min, float max) { return v < min ? min : (v > max ? max : v); }

prefrontal_substrate_config_t prefrontal_substrate_default_config(void) {
    prefrontal_substrate_config_t cfg = { .enable_atp_modulation = true, .enable_fatigue_modulation = true,
        .enable_bio_async = false, .atp_sensitivity = 1.2f, .fatigue_sensitivity = 1.0f, .min_capacity = 0.2f };
    return cfg;
}

prefrontal_substrate_bridge_t* prefrontal_substrate_bridge_create(void* prefrontal, neural_substrate_t* substrate, const prefrontal_substrate_config_t* config) {
    if (!substrate) return NULL;
    prefrontal_substrate_bridge_t* bridge = nimcp_calloc(1, sizeof(prefrontal_substrate_bridge_t));
    if (!bridge) return NULL;
    bridge->prefrontal = prefrontal;
    bridge->substrate = substrate;
    bridge->config = config ? *config : prefrontal_substrate_default_config();
    bridge->effects.executive_function = 1.0f;
    bridge->effects.working_memory = 1.0f;
    bridge->effects.inhibitory_control = 1.0f;
    bridge->effects.planning_capacity = 1.0f;
    bridge->effects.overall_capacity = 1.0f;
    return bridge;
}

void prefrontal_substrate_bridge_destroy(prefrontal_substrate_bridge_t* bridge) { if (bridge) nimcp_free(bridge); }

int prefrontal_substrate_bridge_update(prefrontal_substrate_bridge_t* bridge) {
    if (!bridge || !bridge->substrate) return -1;
    substrate_metabolic_state_t metabolic;
    if (substrate_get_metabolic_state(bridge->substrate, &metabolic) != 0) return -1;
    float atp = metabolic.atp_level, metabolic_cap = metabolic.metabolic_capacity, min_cap = bridge->config.min_capacity;
    /* PFC is most sensitive to ATP depletion */
    if (bridge->config.enable_atp_modulation) {
        bridge->effects.executive_function = clamp_f(atp * bridge->config.atp_sensitivity, min_cap, 1.0f);
        bridge->effects.working_memory = clamp_f(atp * 1.15f * bridge->config.atp_sensitivity, min_cap, 1.0f);
    }
    /* Fatigue impairs inhibition and planning first */
    if (bridge->config.enable_fatigue_modulation) {
        bridge->effects.inhibitory_control = clamp_f(metabolic_cap * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
        bridge->effects.planning_capacity = clamp_f(metabolic_cap * 0.85f * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
    }
    bridge->effects.overall_capacity = (bridge->effects.executive_function + bridge->effects.working_memory +
                                        bridge->effects.inhibitory_control + bridge->effects.planning_capacity) / 4.0f;
    bridge->update_count++;
    return 0;
}

int prefrontal_substrate_bridge_get_effects(const prefrontal_substrate_bridge_t* bridge, prefrontal_substrate_effects_t* effects) {
    if (!bridge || !effects) return -1;
    *effects = bridge->effects;
    return 0;
}

int prefrontal_substrate_bridge_apply_effects(prefrontal_substrate_bridge_t* bridge) { return bridge ? 0 : -1; }

int prefrontal_substrate_bridge_register_bio_async(prefrontal_substrate_bridge_t* bridge, bio_router_t* router) {
    if (!bridge) return -1;
    bridge->router = router;
    bridge->bio_async_connected = (router != NULL);
    return 0;
}
