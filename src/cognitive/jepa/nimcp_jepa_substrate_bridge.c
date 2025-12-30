/**
 * @file nimcp_jepa_substrate_bridge.c
 * @brief JEPA (Joint Embedding Predictive Architecture)-Neural Substrate Bridge Implementation
 */

#include "cognitive/jepa/nimcp_jepa_substrate_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include <string.h>

struct jepa_substrate_bridge {
    void* jepa;
    neural_substrate_t* substrate;
    jepa_substrate_config_t config;
    jepa_substrate_effects_t effects;
    bio_router_t* router;
    bool bio_async_connected;
    uint64_t update_count;
};

static float clamp_f(float v, float min, float max) { return v < min ? min : (v > max ? max : v); }

jepa_substrate_config_t jepa_substrate_default_config(void) {
    jepa_substrate_config_t cfg = { .enable_atp_modulation = true, .enable_fatigue_modulation = true,
        .enable_bio_async = false, .atp_sensitivity = 1.0f, .fatigue_sensitivity = 1.0f, .min_capacity = 0.2f };
    return cfg;
}

jepa_substrate_bridge_t* jepa_substrate_bridge_create(void* jepa, neural_substrate_t* substrate, const jepa_substrate_config_t* config) {
    if (!substrate) return NULL;
    jepa_substrate_bridge_t* bridge = nimcp_calloc(1, sizeof(jepa_substrate_bridge_t));
    if (!bridge) return NULL;
    bridge->jepa = jepa;
    bridge->substrate = substrate;
    bridge->config = config ? *config : jepa_substrate_default_config();
    bridge->effects.prediction_horizon = 1.0f;
    bridge->effects.model_precision = 1.0f;
    bridge->effects.embedding_quality = 1.0f;
    bridge->effects.update_rate = 1.0f;
    bridge->effects.overall_capacity = 1.0f;
    return bridge;
}

void jepa_substrate_bridge_destroy(jepa_substrate_bridge_t* bridge) { if (bridge) nimcp_free(bridge); }

int jepa_substrate_bridge_update(jepa_substrate_bridge_t* bridge) {
    if (!bridge || !bridge->substrate) return -1;
    substrate_metabolic_state_t metabolic;
    if (substrate_get_metabolic_state(bridge->substrate, &metabolic) != 0) return -1;
    float atp = metabolic.atp_level, metabolic_cap = metabolic.metabolic_capacity, min_cap = bridge->config.min_capacity;
    /* ATP enables longer prediction horizons and better model precision */
    if (bridge->config.enable_atp_modulation) {
        bridge->effects.prediction_horizon = clamp_f(atp * bridge->config.atp_sensitivity, min_cap, 1.0f);
        bridge->effects.model_precision = clamp_f(atp * 1.05f * bridge->config.atp_sensitivity, min_cap, 1.0f);
    }
    /* Low fatigue enables quality embeddings and update rate */
    if (bridge->config.enable_fatigue_modulation) {
        bridge->effects.embedding_quality = clamp_f(metabolic_cap * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
        bridge->effects.update_rate = clamp_f(metabolic_cap * 0.9f * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
    }
    bridge->effects.overall_capacity = (bridge->effects.prediction_horizon + bridge->effects.model_precision +
                                        bridge->effects.embedding_quality + bridge->effects.update_rate) / 4.0f;
    bridge->update_count++;
    return 0;
}

int jepa_substrate_bridge_get_effects(const jepa_substrate_bridge_t* bridge, jepa_substrate_effects_t* effects) {
    if (!bridge || !effects) return -1;
    *effects = bridge->effects;
    return 0;
}

int jepa_substrate_bridge_apply_effects(jepa_substrate_bridge_t* bridge) { return bridge ? 0 : -1; }

int jepa_substrate_bridge_register_bio_async(jepa_substrate_bridge_t* bridge, bio_router_t* router) {
    if (!bridge) return -1;
    bridge->router = router;
    bridge->bio_async_connected = (router != NULL);
    return 0;
}
