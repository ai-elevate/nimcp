/**
 * @file nimcp_hippocampus_substrate_bridge.c
 * @brief Hippocampus-Neural Substrate Bridge Implementation
 */

#include "utils/bridge/nimcp_bridge_base.h"
#include "core/hippocampus/nimcp_hippocampus_substrate_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>

struct hippocampus_substrate_bridge {
    bridge_base_t base;              /**< MUST be first: base bridge infrastructure */
    void* hippocampus;
    neural_substrate_t* substrate;
    hippocampus_substrate_config_t config;
    hippocampus_substrate_effects_t effects;
    bio_router_t* router;
    bool bio_async_connected;
    uint64_t update_count;
};

static float clamp_f(float v, float min, float max) { return v < min ? min : (v > max ? max : v); }

hippocampus_substrate_config_t hippocampus_substrate_default_config(void) {
    hippocampus_substrate_config_t cfg = { .enable_atp_modulation = true, .enable_fatigue_modulation = true,
        .enable_bio_async = false, .atp_sensitivity = 1.0f, .fatigue_sensitivity = 1.0f, .min_capacity = 0.2f };
    return cfg;
}

hippocampus_substrate_bridge_t* hippocampus_substrate_bridge_create(void* hippocampus, neural_substrate_t* substrate, const hippocampus_substrate_config_t* config) {
    if (!substrate) return NULL;
    hippocampus_substrate_bridge_t* bridge = nimcp_calloc(1, sizeof(hippocampus_substrate_bridge_t));
    if (!bridge) return NULL;
    bridge->hippocampus = hippocampus;
    bridge->substrate = substrate;
    bridge->config = config ? *config : hippocampus_substrate_default_config();
    bridge->effects.encoding_capacity = 1.0f;
    bridge->effects.consolidation_rate = 1.0f;
    bridge->effects.retrieval_accuracy = 1.0f;
    bridge->effects.spatial_processing = 1.0f;
    bridge->effects.overall_capacity = 1.0f;
    return bridge;
}

void hippocampus_substrate_bridge_destroy(hippocampus_substrate_bridge_t* bridge) { if (bridge) nimcp_free(bridge); }

int hippocampus_substrate_bridge_update(hippocampus_substrate_bridge_t* bridge) {
    if (!bridge || !bridge->substrate) return -1;
    substrate_metabolic_state_t metabolic;
    if (substrate_get_metabolic_state(bridge->substrate, &metabolic) != 0) return -1;
    float atp = metabolic.atp_level, metabolic_cap = metabolic.metabolic_capacity, min_cap = bridge->config.min_capacity;
    /* ATP is critical for memory encoding and LTP */
    if (bridge->config.enable_atp_modulation) {
        bridge->effects.encoding_capacity = clamp_f(atp * bridge->config.atp_sensitivity, min_cap, 1.0f);
        bridge->effects.consolidation_rate = clamp_f(atp * 1.15f * bridge->config.atp_sensitivity, min_cap, 1.0f);
    }
    /* Fatigue affects retrieval and spatial processing */
    if (bridge->config.enable_fatigue_modulation) {
        bridge->effects.retrieval_accuracy = clamp_f(metabolic_cap * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
        bridge->effects.spatial_processing = clamp_f(metabolic_cap * 0.9f * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
    }
    bridge->effects.overall_capacity = (bridge->effects.encoding_capacity + bridge->effects.consolidation_rate +
                                        bridge->effects.retrieval_accuracy + bridge->effects.spatial_processing) / 4.0f;
    bridge->update_count++;
    return 0;
}

int hippocampus_substrate_bridge_get_effects(const hippocampus_substrate_bridge_t* bridge, hippocampus_substrate_effects_t* effects) {
    if (!bridge || !effects) return -1;
    *effects = bridge->effects;
    return 0;
}

int hippocampus_substrate_bridge_apply_effects(hippocampus_substrate_bridge_t* bridge) { return bridge ? 0 : -1; }

int hippocampus_substrate_bridge_register_bio_async(hippocampus_substrate_bridge_t* bridge, bio_router_t* router) {
    if (!bridge) return -1;
    bridge->router = router;
    bridge->bio_async_connected = (router != NULL);
    return 0;
}
