/**
 * @file nimcp_meta_learning_substrate_bridge.c
 * @brief Meta-Learning-Neural Substrate Bridge Implementation
 */

#include "cognitive/meta_learning/nimcp_meta_learning_substrate_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include <string.h>

struct meta_learning_substrate_bridge {
    void* meta_learning;
    neural_substrate_t* substrate;
    meta_learning_substrate_config_t config;
    meta_learning_substrate_effects_t effects;
    bio_router_t* router;
    bool bio_async_connected;
    uint64_t update_count;
};

static float clamp_f(float v, float min, float max) { return v < min ? min : (v > max ? max : v); }

meta_learning_substrate_config_t meta_learning_substrate_default_config(void) {
    meta_learning_substrate_config_t cfg = { .enable_atp_modulation = true, .enable_fatigue_modulation = true,
        .enable_bio_async = false, .atp_sensitivity = 1.0f, .fatigue_sensitivity = 1.0f, .min_capacity = 0.2f };
    return cfg;
}

meta_learning_substrate_bridge_t* meta_learning_substrate_bridge_create(void* meta_learning, neural_substrate_t* substrate, const meta_learning_substrate_config_t* config) {
    if (!substrate) return NULL;
    meta_learning_substrate_bridge_t* bridge = nimcp_calloc(1, sizeof(meta_learning_substrate_bridge_t));
    if (!bridge) return NULL;
    bridge->meta_learning = meta_learning;
    bridge->substrate = substrate;
    bridge->config = config ? *config : meta_learning_substrate_default_config();
    bridge->effects.learning_rate_adapt = 1.0f;
    bridge->effects.strategy_flexibility = 1.0f;
    bridge->effects.transfer_capacity = 1.0f;
    bridge->effects.plasticity_level = 1.0f;
    bridge->effects.overall_capacity = 1.0f;
    return bridge;
}

void meta_learning_substrate_bridge_destroy(meta_learning_substrate_bridge_t* bridge) { if (bridge) nimcp_free(bridge); }

int meta_learning_substrate_bridge_update(meta_learning_substrate_bridge_t* bridge) {
    if (!bridge || !bridge->substrate) return -1;
    substrate_metabolic_state_t metabolic;
    if (substrate_get_metabolic_state(bridge->substrate, &metabolic) != 0) return -1;
    float atp = metabolic.atp_level, metabolic_cap = metabolic.metabolic_capacity, min_cap = bridge->config.min_capacity;
    if (bridge->config.enable_atp_modulation) {
        bridge->effects.learning_rate_adapt = clamp_f(atp * bridge->config.atp_sensitivity, min_cap, 1.0f);
        bridge->effects.plasticity_level = clamp_f(atp * 1.1f * bridge->config.atp_sensitivity, min_cap, 1.0f);
    }
    if (bridge->config.enable_fatigue_modulation) {
        bridge->effects.strategy_flexibility = clamp_f(metabolic_cap * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
        bridge->effects.transfer_capacity = clamp_f(metabolic_cap * 0.9f * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
    }
    bridge->effects.overall_capacity = (bridge->effects.learning_rate_adapt + bridge->effects.strategy_flexibility +
                                        bridge->effects.transfer_capacity + bridge->effects.plasticity_level) / 4.0f;
    bridge->update_count++;
    return 0;
}

int meta_learning_substrate_bridge_get_effects(const meta_learning_substrate_bridge_t* bridge, meta_learning_substrate_effects_t* effects) {
    if (!bridge || !effects) return -1;
    *effects = bridge->effects;
    return 0;
}

int meta_learning_substrate_bridge_apply_effects(meta_learning_substrate_bridge_t* bridge) { return bridge ? 0 : -1; }

int meta_learning_substrate_bridge_register_bio_async(meta_learning_substrate_bridge_t* bridge, bio_router_t* router) {
    if (!bridge) return -1;
    bridge->router = router;
    bridge->bio_async_connected = (router != NULL);
    return 0;
}
