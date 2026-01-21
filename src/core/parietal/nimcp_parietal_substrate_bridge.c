/**
 * @file nimcp_parietal_substrate_bridge.c
 * @brief Parietal Cortex-Neural Substrate Bridge Implementation
 */

#include "core/parietal/nimcp_parietal_substrate_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>

struct parietal_substrate_bridge {
    void* parietal;
    neural_substrate_t* substrate;
    parietal_substrate_config_t config;
    parietal_substrate_effects_t effects;
    bio_router_t* router;
    bool bio_async_connected;
    uint64_t update_count;
};

static float clamp_f(float v, float min, float max) { return v < min ? min : (v > max ? max : v); }

parietal_substrate_config_t parietal_substrate_default_config(void) {
    parietal_substrate_config_t cfg = { .enable_atp_modulation = true, .enable_fatigue_modulation = true,
        .enable_bio_async = false, .atp_sensitivity = 1.0f, .fatigue_sensitivity = 1.0f, .min_capacity = 0.2f };
    return cfg;
}

parietal_substrate_bridge_t* parietal_substrate_bridge_create(void* parietal, neural_substrate_t* substrate, const parietal_substrate_config_t* config) {
    if (!substrate) return NULL;
    parietal_substrate_bridge_t* bridge = nimcp_calloc(1, sizeof(parietal_substrate_bridge_t));
    if (!bridge) return NULL;
    bridge->parietal = parietal;
    bridge->substrate = substrate;
    bridge->config = config ? *config : parietal_substrate_default_config();
    bridge->effects.spatial_attention = 1.0f;
    bridge->effects.numerical_processing = 1.0f;
    bridge->effects.sensory_integration = 1.0f;
    bridge->effects.body_awareness = 1.0f;
    bridge->effects.overall_capacity = 1.0f;
    return bridge;
}

void parietal_substrate_bridge_destroy(parietal_substrate_bridge_t* bridge) { if (bridge) nimcp_free(bridge); }

int parietal_substrate_bridge_update(parietal_substrate_bridge_t* bridge) {
    if (!bridge || !bridge->substrate) return -1;
    substrate_metabolic_state_t metabolic;
    if (substrate_get_metabolic_state(bridge->substrate, &metabolic) != 0) return -1;
    float atp = metabolic.atp_level, metabolic_cap = metabolic.metabolic_capacity, min_cap = bridge->config.min_capacity;
    /* ATP affects spatial attention and numerical processing */
    if (bridge->config.enable_atp_modulation) {
        bridge->effects.spatial_attention = clamp_f(atp * bridge->config.atp_sensitivity, min_cap, 1.0f);
        bridge->effects.numerical_processing = clamp_f(atp * 1.05f * bridge->config.atp_sensitivity, min_cap, 1.0f);
    }
    /* Fatigue affects sensory integration and body awareness */
    if (bridge->config.enable_fatigue_modulation) {
        bridge->effects.sensory_integration = clamp_f(metabolic_cap * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
        bridge->effects.body_awareness = clamp_f(metabolic_cap * 0.9f * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
    }
    bridge->effects.overall_capacity = (bridge->effects.spatial_attention + bridge->effects.numerical_processing +
                                        bridge->effects.sensory_integration + bridge->effects.body_awareness) / 4.0f;
    bridge->update_count++;
    return 0;
}

int parietal_substrate_bridge_get_effects(const parietal_substrate_bridge_t* bridge, parietal_substrate_effects_t* effects) {
    if (!bridge || !effects) return -1;
    *effects = bridge->effects;
    return 0;
}

int parietal_substrate_bridge_apply_effects(parietal_substrate_bridge_t* bridge) { return bridge ? 0 : -1; }

int parietal_substrate_bridge_register_bio_async(parietal_substrate_bridge_t* bridge, bio_router_t* router) {
    if (!bridge) return -1;
    bridge->router = router;
    bridge->bio_async_connected = (router != NULL);
    return 0;
}
