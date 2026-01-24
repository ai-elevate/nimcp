/**
 * @file nimcp_cingulate_substrate_bridge.c
 * @brief Cingulate Cortex-Neural Substrate Bridge Implementation
 */

#include "utils/bridge/nimcp_bridge_base.h"
#include "core/cingulate/nimcp_cingulate_substrate_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>

struct cingulate_substrate_bridge {
    bridge_base_t base;              /**< MUST be first: base bridge infrastructure */
    void* cingulate;
    neural_substrate_t* substrate;
    cingulate_substrate_config_t config;
    cingulate_substrate_effects_t effects;
    bio_router_t* router;
    bool bio_async_connected;
    uint64_t update_count;
};

static float clamp_f(float v, float min, float max) { return v < min ? min : (v > max ? max : v); }

cingulate_substrate_config_t cingulate_substrate_default_config(void) {
    cingulate_substrate_config_t cfg = { .enable_atp_modulation = true, .enable_fatigue_modulation = true,
        .enable_bio_async = false, .atp_sensitivity = 1.0f, .fatigue_sensitivity = 1.0f, .min_capacity = 0.2f };
    return cfg;
}

cingulate_substrate_bridge_t* cingulate_substrate_bridge_create(void* cingulate, neural_substrate_t* substrate, const cingulate_substrate_config_t* config) {
    if (!substrate) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "substrate is NULL");

        return NULL;

    }
    cingulate_substrate_bridge_t* bridge = nimcp_calloc(1, sizeof(cingulate_substrate_bridge_t));
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return NULL;

    }
    bridge->cingulate = cingulate;
    bridge->substrate = substrate;
    bridge->config = config ? *config : cingulate_substrate_default_config();
    bridge->effects.error_detection = 1.0f;
    bridge->effects.conflict_resolution = 1.0f;
    bridge->effects.emotional_regulation = 1.0f;
    bridge->effects.pain_processing = 1.0f;
    bridge->effects.overall_capacity = 1.0f;
    return bridge;
}

void cingulate_substrate_bridge_destroy(cingulate_substrate_bridge_t* bridge) { if (bridge) nimcp_free(bridge); }

int cingulate_substrate_bridge_update(cingulate_substrate_bridge_t* bridge) {
    if (!bridge || !bridge->substrate) return -1;
    substrate_metabolic_state_t metabolic;
    if (substrate_get_metabolic_state(bridge->substrate, &metabolic) != 0) return -1;
    float atp = metabolic.atp_level, metabolic_cap = metabolic.metabolic_capacity, min_cap = bridge->config.min_capacity;
    /* ATP affects error detection and emotional regulation */
    if (bridge->config.enable_atp_modulation) {
        bridge->effects.error_detection = clamp_f(atp * bridge->config.atp_sensitivity, min_cap, 1.0f);
        bridge->effects.emotional_regulation = clamp_f(atp * 1.1f * bridge->config.atp_sensitivity, min_cap, 1.0f);
    }
    /* Fatigue affects conflict resolution and pain processing */
    if (bridge->config.enable_fatigue_modulation) {
        bridge->effects.conflict_resolution = clamp_f(metabolic_cap * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
        bridge->effects.pain_processing = clamp_f(metabolic_cap * 0.9f * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
    }
    bridge->effects.overall_capacity = (bridge->effects.error_detection + bridge->effects.conflict_resolution +
                                        bridge->effects.emotional_regulation + bridge->effects.pain_processing) / 4.0f;
    bridge->update_count++;
    return 0;
}

int cingulate_substrate_bridge_get_effects(const cingulate_substrate_bridge_t* bridge, cingulate_substrate_effects_t* effects) {
    if (!bridge || !effects) return -1;
    *effects = bridge->effects;
    return 0;
}

int cingulate_substrate_bridge_apply_effects(cingulate_substrate_bridge_t* bridge) { return bridge ? 0 : -1; }

int cingulate_substrate_bridge_register_bio_async(cingulate_substrate_bridge_t* bridge, bio_router_t* router) {
    if (!bridge) return -1;
    bridge->router = router;
    bridge->bio_async_connected = (router != NULL);
    return 0;
}
