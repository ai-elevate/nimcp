/**
 * @file nimcp_hypothalamus_substrate_bridge.c
 * @brief Hypothalamus-Neural Substrate Bridge Implementation
 */

#include "utils/bridge/nimcp_bridge_base.h"
#include "core/hypothalamus/nimcp_hypothalamus_substrate_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>

struct hypothalamus_substrate_bridge {
    bridge_base_t base;              /**< MUST be first: base bridge infrastructure */
    void* hypothalamus;
    neural_substrate_t* substrate;
    hypothalamus_substrate_config_t config;
    hypothalamus_substrate_effects_t effects;
    bio_router_t* router;
    bool bio_async_connected;
    uint64_t update_count;
};

static float clamp_f(float v, float min, float max) { return v < min ? min : (v > max ? max : v); }

hypothalamus_substrate_config_t hypothalamus_substrate_default_config(void) {
    hypothalamus_substrate_config_t cfg = { .enable_atp_modulation = true, .enable_fatigue_modulation = true,
        .enable_bio_async = false, .atp_sensitivity = 1.0f, .fatigue_sensitivity = 1.0f, .min_capacity = 0.3f };
    return cfg;
}

hypothalamus_substrate_bridge_t* hypothalamus_substrate_bridge_create(void* hypothalamus, neural_substrate_t* substrate, const hypothalamus_substrate_config_t* config) {
    if (!substrate) return NULL;
    hypothalamus_substrate_bridge_t* bridge = nimcp_calloc(1, sizeof(hypothalamus_substrate_bridge_t));
    if (!bridge) return NULL;
    bridge->hypothalamus = hypothalamus;
    bridge->substrate = substrate;
    bridge->config = config ? *config : hypothalamus_substrate_default_config();
    bridge->effects.homeostatic_drive = 1.0f;
    bridge->effects.hunger_signal = 0.0f;
    bridge->effects.circadian_strength = 1.0f;
    bridge->effects.stress_response = 0.0f;
    bridge->effects.overall_capacity = 1.0f;
    return bridge;
}

void hypothalamus_substrate_bridge_destroy(hypothalamus_substrate_bridge_t* bridge) { if (bridge) nimcp_free(bridge); }

int hypothalamus_substrate_bridge_update(hypothalamus_substrate_bridge_t* bridge) {
    if (!bridge || !bridge->substrate) return -1;
    substrate_metabolic_state_t metabolic;
    if (substrate_get_metabolic_state(bridge->substrate, &metabolic) != 0) return -1;
    float atp = metabolic.atp_level, fatigue = 1.0f - metabolic.metabolic_capacity, min_cap = bridge->config.min_capacity;
    /* Hypothalamus directly senses ATP - low ATP triggers hunger and stress */
    if (bridge->config.enable_atp_modulation) {
        bridge->effects.homeostatic_drive = clamp_f(atp * bridge->config.atp_sensitivity, min_cap, 1.0f);
        /* Low ATP increases hunger signal */
        bridge->effects.hunger_signal = clamp_f((1.0f - atp) * bridge->config.atp_sensitivity, 0.0f, 1.0f);
        /* Low ATP triggers stress response */
        bridge->effects.stress_response = clamp_f((1.0f - atp) * 0.8f * bridge->config.atp_sensitivity, 0.0f, 1.0f);
    }
    /* Fatigue modulates circadian strength */
    if (bridge->config.enable_fatigue_modulation) {
        bridge->effects.circadian_strength = clamp_f((1.0f - fatigue * 0.5f) * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
    }
    bridge->effects.overall_capacity = bridge->effects.homeostatic_drive;
    bridge->update_count++;
    return 0;
}

int hypothalamus_substrate_bridge_get_effects(const hypothalamus_substrate_bridge_t* bridge, hypothalamus_substrate_effects_t* effects) {
    if (!bridge || !effects) return -1;
    *effects = bridge->effects;
    return 0;
}

int hypothalamus_substrate_bridge_apply_effects(hypothalamus_substrate_bridge_t* bridge) { return bridge ? 0 : -1; }

int hypothalamus_substrate_bridge_register_bio_async(hypothalamus_substrate_bridge_t* bridge, bio_router_t* router) {
    if (!bridge) return -1;
    bridge->router = router;
    bridge->bio_async_connected = (router != NULL);
    return 0;
}
