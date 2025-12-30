/**
 * @file nimcp_llf_substrate_bridge.c
 * @brief Love/Loyalty/Friendship-Neural Substrate Bridge Implementation
 */

#include "cognitive/love_loyalty_friendship/nimcp_llf_substrate_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include <string.h>

struct llf_substrate_bridge {
    void* llf;
    neural_substrate_t* substrate;
    llf_substrate_config_t config;
    llf_substrate_effects_t effects;
    bio_router_t* router;
    bool bio_async_connected;
    uint64_t update_count;
};

static float clamp_f(float v, float min, float max) { return v < min ? min : (v > max ? max : v); }

llf_substrate_config_t llf_substrate_default_config(void) {
    llf_substrate_config_t cfg = { .enable_atp_modulation = true, .enable_fatigue_modulation = true,
        .enable_bio_async = false, .atp_sensitivity = 1.0f, .fatigue_sensitivity = 1.0f, .min_capacity = 0.2f };
    return cfg;
}

llf_substrate_bridge_t* llf_substrate_bridge_create(void* llf, neural_substrate_t* substrate, const llf_substrate_config_t* config) {
    if (!substrate) return NULL;
    llf_substrate_bridge_t* bridge = nimcp_calloc(1, sizeof(llf_substrate_bridge_t));
    if (!bridge) return NULL;
    bridge->llf = llf;
    bridge->substrate = substrate;
    bridge->config = config ? *config : llf_substrate_default_config();
    bridge->effects.attachment_strength = 1.0f;
    bridge->effects.trust_capacity = 1.0f;
    bridge->effects.social_investment = 1.0f;
    bridge->effects.loyalty_maintenance = 1.0f;
    bridge->effects.overall_capacity = 1.0f;
    return bridge;
}

void llf_substrate_bridge_destroy(llf_substrate_bridge_t* bridge) { if (bridge) nimcp_free(bridge); }

int llf_substrate_bridge_update(llf_substrate_bridge_t* bridge) {
    if (!bridge || !bridge->substrate) return -1;
    substrate_metabolic_state_t metabolic;
    if (substrate_get_metabolic_state(bridge->substrate, &metabolic) != 0) return -1;
    float atp = metabolic.atp_level, metabolic_cap = metabolic.metabolic_capacity, min_cap = bridge->config.min_capacity;
    /* ATP enables attachment and trust */
    if (bridge->config.enable_atp_modulation) {
        bridge->effects.attachment_strength = clamp_f(atp * bridge->config.atp_sensitivity, min_cap, 1.0f);
        bridge->effects.trust_capacity = clamp_f(atp * 1.05f * bridge->config.atp_sensitivity, min_cap, 1.0f);
    }
    /* Low fatigue enables social investment and loyalty */
    if (bridge->config.enable_fatigue_modulation) {
        bridge->effects.social_investment = clamp_f(metabolic_cap * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
        bridge->effects.loyalty_maintenance = clamp_f(metabolic_cap * 0.9f * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
    }
    bridge->effects.overall_capacity = (bridge->effects.attachment_strength + bridge->effects.trust_capacity +
                                        bridge->effects.social_investment + bridge->effects.loyalty_maintenance) / 4.0f;
    bridge->update_count++;
    return 0;
}

int llf_substrate_bridge_get_effects(const llf_substrate_bridge_t* bridge, llf_substrate_effects_t* effects) {
    if (!bridge || !effects) return -1;
    *effects = bridge->effects;
    return 0;
}

int llf_substrate_bridge_apply_effects(llf_substrate_bridge_t* bridge) { return bridge ? 0 : -1; }

int llf_substrate_bridge_register_bio_async(llf_substrate_bridge_t* bridge, bio_router_t* router) {
    if (!bridge) return -1;
    bridge->router = router;
    bridge->bio_async_connected = (router != NULL);
    return 0;
}
