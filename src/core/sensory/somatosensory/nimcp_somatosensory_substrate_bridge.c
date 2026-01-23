/**
 * @file nimcp_somatosensory_substrate_bridge.c
 * @brief Somatosensory-Neural Substrate Bridge Implementation
 */

#include "utils/bridge/nimcp_bridge_base.h"
#include "core/sensory/somatosensory/nimcp_somatosensory_substrate_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>

struct somatosensory_substrate_bridge {
    bridge_base_t base;              /**< MUST be first: base bridge infrastructure */
    void* somatosensory;
    neural_substrate_t* substrate;
    somatosensory_substrate_config_t config;
    somatosensory_substrate_effects_t effects;
    bio_router_t* router;
    bool bio_async_connected;
    uint64_t update_count;
};

static float clamp_f(float v, float min, float max) { return v < min ? min : (v > max ? max : v); }

somatosensory_substrate_config_t somatosensory_substrate_default_config(void) {
    somatosensory_substrate_config_t cfg = { .enable_atp_modulation = true, .enable_fatigue_modulation = true,
        .enable_bio_async = false, .atp_sensitivity = 1.0f, .fatigue_sensitivity = 1.0f, .min_capacity = 0.2f };
    return cfg;
}

somatosensory_substrate_bridge_t* somatosensory_substrate_bridge_create(void* somatosensory, neural_substrate_t* substrate, const somatosensory_substrate_config_t* config) {
    if (!substrate) return NULL;
    somatosensory_substrate_bridge_t* bridge = nimcp_calloc(1, sizeof(somatosensory_substrate_bridge_t));
    if (!bridge) return NULL;
    bridge->somatosensory = somatosensory;
    bridge->substrate = substrate;
    bridge->config = config ? *config : somatosensory_substrate_default_config();
    bridge->effects.tactile_acuity = 1.0f;
    bridge->effects.proprioception = 1.0f;
    bridge->effects.pain_processing = 1.0f;
    bridge->effects.temperature_sense = 1.0f;
    bridge->effects.overall_capacity = 1.0f;
    return bridge;
}

void somatosensory_substrate_bridge_destroy(somatosensory_substrate_bridge_t* bridge) { if (bridge) nimcp_free(bridge); }

int somatosensory_substrate_bridge_update(somatosensory_substrate_bridge_t* bridge) {
    if (!bridge || !bridge->substrate) return -1;
    substrate_metabolic_state_t metabolic;
    if (substrate_get_metabolic_state(bridge->substrate, &metabolic) != 0) return -1;
    float atp = metabolic.atp_level, metabolic_cap = metabolic.metabolic_capacity, min_cap = bridge->config.min_capacity;
    /* ATP affects tactile acuity and proprioception */
    if (bridge->config.enable_atp_modulation) {
        bridge->effects.tactile_acuity = clamp_f(atp * bridge->config.atp_sensitivity, min_cap, 1.0f);
        bridge->effects.proprioception = clamp_f(atp * 1.05f * bridge->config.atp_sensitivity, min_cap, 1.0f);
    }
    /* Fatigue affects pain processing and temperature sense */
    if (bridge->config.enable_fatigue_modulation) {
        bridge->effects.pain_processing = clamp_f(metabolic_cap * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
        bridge->effects.temperature_sense = clamp_f(metabolic_cap * 0.9f * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
    }
    bridge->effects.overall_capacity = (bridge->effects.tactile_acuity + bridge->effects.proprioception +
                                        bridge->effects.pain_processing + bridge->effects.temperature_sense) / 4.0f;
    bridge->update_count++;
    return 0;
}

int somatosensory_substrate_bridge_get_effects(const somatosensory_substrate_bridge_t* bridge, somatosensory_substrate_effects_t* effects) {
    if (!bridge || !effects) return -1;
    *effects = bridge->effects;
    return 0;
}

int somatosensory_substrate_bridge_apply_effects(somatosensory_substrate_bridge_t* bridge) { return bridge ? 0 : -1; }

int somatosensory_substrate_bridge_register_bio_async(somatosensory_substrate_bridge_t* bridge, bio_router_t* router) {
    if (!bridge) return -1;
    bridge->router = router;
    bridge->bio_async_connected = (router != NULL);
    return 0;
}
