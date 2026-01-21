/**
 * @file nimcp_insula_substrate_bridge.c
 * @brief Insula-Neural Substrate Bridge Implementation
 */

#include "core/insula/nimcp_insula_substrate_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>

struct insula_substrate_bridge {
    void* insula;
    neural_substrate_t* substrate;
    insula_substrate_config_t config;
    insula_substrate_effects_t effects;
    bio_router_t* router;
    bool bio_async_connected;
    uint64_t update_count;
};

static float clamp_f(float v, float min, float max) { return v < min ? min : (v > max ? max : v); }

insula_substrate_config_t insula_substrate_default_config(void) {
    insula_substrate_config_t cfg = { .enable_atp_modulation = true, .enable_fatigue_modulation = true,
        .enable_bio_async = false, .atp_sensitivity = 1.0f, .fatigue_sensitivity = 1.0f, .min_capacity = 0.2f };
    return cfg;
}

insula_substrate_bridge_t* insula_substrate_bridge_create(void* insula, neural_substrate_t* substrate, const insula_substrate_config_t* config) {
    if (!substrate) return NULL;
    insula_substrate_bridge_t* bridge = nimcp_calloc(1, sizeof(insula_substrate_bridge_t));
    if (!bridge) return NULL;
    bridge->insula = insula;
    bridge->substrate = substrate;
    bridge->config = config ? *config : insula_substrate_default_config();
    bridge->effects.interoceptive_accuracy = 1.0f;
    bridge->effects.emotional_awareness = 1.0f;
    bridge->effects.disgust_sensitivity = 1.0f;
    bridge->effects.empathic_response = 1.0f;
    bridge->effects.overall_capacity = 1.0f;
    return bridge;
}

void insula_substrate_bridge_destroy(insula_substrate_bridge_t* bridge) { if (bridge) nimcp_free(bridge); }

int insula_substrate_bridge_update(insula_substrate_bridge_t* bridge) {
    if (!bridge || !bridge->substrate) return -1;
    substrate_metabolic_state_t metabolic;
    if (substrate_get_metabolic_state(bridge->substrate, &metabolic) != 0) return -1;
    float atp = metabolic.atp_level, metabolic_cap = metabolic.metabolic_capacity, min_cap = bridge->config.min_capacity;
    /* ATP affects interoceptive accuracy - insula directly senses metabolic state */
    if (bridge->config.enable_atp_modulation) {
        bridge->effects.interoceptive_accuracy = clamp_f(atp * bridge->config.atp_sensitivity, min_cap, 1.0f);
        bridge->effects.disgust_sensitivity = clamp_f(atp * 1.05f * bridge->config.atp_sensitivity, min_cap, 1.0f);
    }
    /* Fatigue affects emotional awareness and empathy */
    if (bridge->config.enable_fatigue_modulation) {
        bridge->effects.emotional_awareness = clamp_f(metabolic_cap * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
        bridge->effects.empathic_response = clamp_f(metabolic_cap * 0.9f * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
    }
    bridge->effects.overall_capacity = (bridge->effects.interoceptive_accuracy + bridge->effects.emotional_awareness +
                                        bridge->effects.disgust_sensitivity + bridge->effects.empathic_response) / 4.0f;
    bridge->update_count++;
    return 0;
}

int insula_substrate_bridge_get_effects(const insula_substrate_bridge_t* bridge, insula_substrate_effects_t* effects) {
    if (!bridge || !effects) return -1;
    *effects = bridge->effects;
    return 0;
}

int insula_substrate_bridge_apply_effects(insula_substrate_bridge_t* bridge) { return bridge ? 0 : -1; }

int insula_substrate_bridge_register_bio_async(insula_substrate_bridge_t* bridge, bio_router_t* router) {
    if (!bridge) return -1;
    bridge->router = router;
    bridge->bio_async_connected = (router != NULL);
    return 0;
}
