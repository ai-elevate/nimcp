/**
 * @file nimcp_self_awareness_substrate_bridge.c
 * @brief Self-Awareness-Neural Substrate Bridge Implementation
 *
 * WHAT: Links self-awareness to metabolic state
 * WHY: Metacognition requires medial prefrontal and cingulate resources
 * HOW: Monitors ATP/fatigue; modulates self-reflection, introspection, metacognition
 */

#include "cognitive/self_awareness/nimcp_self_awareness_substrate_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include <string.h>
#include <math.h>

struct self_awareness_substrate_bridge {
    void* self_awareness;
    neural_substrate_t* substrate;
    self_awareness_substrate_config_t config;
    self_awareness_substrate_effects_t effects;
    bio_router_t* router;
    bool bio_async_connected;
    uint64_t update_count;
};

static float clamp_f(float v, float min, float max) {
    return v < min ? min : (v > max ? max : v);
}

self_awareness_substrate_config_t self_awareness_substrate_default_config(void) {
    self_awareness_substrate_config_t cfg = {
        .enable_atp_modulation = true,
        .enable_fatigue_modulation = true,
        .enable_bio_async = false,
        .atp_sensitivity = 1.0f,
        .fatigue_sensitivity = 1.0f,
        .min_capacity = 0.2f
    };
    return cfg;
}

self_awareness_substrate_bridge_t* self_awareness_substrate_bridge_create(void* self_awareness, neural_substrate_t* substrate, const self_awareness_substrate_config_t* config) {
    if (!substrate) return NULL;

    self_awareness_substrate_bridge_t* bridge = nimcp_calloc(1, sizeof(self_awareness_substrate_bridge_t));
    if (!bridge) return NULL;

    bridge->self_awareness = self_awareness;
    bridge->substrate = substrate;
    bridge->config = config ? *config : self_awareness_substrate_default_config();

    bridge->effects.introspection_depth = 1.0f;
    bridge->effects.metacognitive_accuracy = 1.0f;
    bridge->effects.self_monitoring = 1.0f;
    bridge->effects.identity_coherence = 1.0f;
    bridge->effects.overall_capacity = 1.0f;

    return bridge;
}

void self_awareness_substrate_bridge_destroy(self_awareness_substrate_bridge_t* bridge) {
    if (bridge) nimcp_free(bridge);
}

int self_awareness_substrate_bridge_update(self_awareness_substrate_bridge_t* bridge) {
    if (!bridge || !bridge->substrate) return -1;

    substrate_metabolic_state_t metabolic;
    if (substrate_get_metabolic_state(bridge->substrate, &metabolic) != 0) return -1;

    float atp = metabolic.atp_level;
    float metabolic_cap = metabolic.metabolic_capacity;
    float min_cap = bridge->config.min_capacity;

    if (bridge->config.enable_atp_modulation) {
        bridge->effects.introspection_depth = clamp_f(atp * bridge->config.atp_sensitivity, min_cap, 1.0f);
        bridge->effects.metacognitive_accuracy = clamp_f(atp * 1.05f * bridge->config.atp_sensitivity, min_cap, 1.0f);
    }

    if (bridge->config.enable_fatigue_modulation) {
        bridge->effects.self_monitoring = clamp_f(metabolic_cap * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
        bridge->effects.identity_coherence = clamp_f(metabolic_cap * 0.95f * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
    }

    bridge->effects.overall_capacity = (bridge->effects.introspection_depth +
                                        bridge->effects.metacognitive_accuracy +
                                        bridge->effects.self_monitoring +
                                        bridge->effects.identity_coherence) / 4.0f;

    bridge->update_count++;
    return 0;
}

int self_awareness_substrate_bridge_get_effects(const self_awareness_substrate_bridge_t* bridge, self_awareness_substrate_effects_t* effects) {
    if (!bridge || !effects) return -1;
    *effects = bridge->effects;
    return 0;
}

int self_awareness_substrate_bridge_apply_effects(self_awareness_substrate_bridge_t* bridge) {
    if (!bridge) return -1;
    return 0;
}

int self_awareness_substrate_bridge_register_bio_async(self_awareness_substrate_bridge_t* bridge, bio_router_t* router) {
    if (!bridge) return -1;
    bridge->router = router;
    bridge->bio_async_connected = (router != NULL);
    return 0;
}
