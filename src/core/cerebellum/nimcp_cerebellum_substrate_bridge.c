/**
 * @file nimcp_cerebellum_substrate_bridge.c
 * @brief Cerebellum-Neural Substrate Bridge Implementation
 */

#include "utils/bridge/nimcp_bridge_base.h"
#include "core/cerebellum/nimcp_cerebellum_substrate_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>

struct cerebellum_substrate_bridge {
    bridge_base_t base;              /**< MUST be first: base bridge infrastructure */
    void* cerebellum;
    neural_substrate_t* substrate;
    cerebellum_substrate_config_t config;
    cerebellum_substrate_effects_t effects;
    bio_router_t* router;
    bool bio_async_connected;
    uint64_t update_count;
};

static float clamp_f(float v, float min, float max) { return v < min ? min : (v > max ? max : v); }

cerebellum_substrate_config_t cerebellum_substrate_default_config(void) {
    cerebellum_substrate_config_t cfg = { .enable_atp_modulation = true, .enable_fatigue_modulation = true,
        .enable_bio_async = false, .atp_sensitivity = 1.0f, .fatigue_sensitivity = 1.0f, .min_capacity = 0.2f };
    return cfg;
}

cerebellum_substrate_bridge_t* cerebellum_substrate_bridge_create(void* cerebellum, neural_substrate_t* substrate, const cerebellum_substrate_config_t* config) {
    if (!substrate) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "substrate is NULL");

        return NULL;

    }
    cerebellum_substrate_bridge_t* bridge = nimcp_calloc(1, sizeof(cerebellum_substrate_bridge_t));
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return NULL;

    }
    bridge->cerebellum = cerebellum;
    bridge->substrate = substrate;
    bridge->config = config ? *config : cerebellum_substrate_default_config();
    bridge->effects.motor_coordination = 1.0f;
    bridge->effects.timing_precision = 1.0f;
    bridge->effects.procedural_learning = 1.0f;
    bridge->effects.error_correction = 1.0f;
    bridge->effects.overall_capacity = 1.0f;
    return bridge;
}

void cerebellum_substrate_bridge_destroy(cerebellum_substrate_bridge_t* bridge) { if (bridge) nimcp_free(bridge); }

int cerebellum_substrate_bridge_update(cerebellum_substrate_bridge_t* bridge) {
    if (!bridge || !bridge->substrate) return -1;
    substrate_metabolic_state_t metabolic;
    if (substrate_get_metabolic_state(bridge->substrate, &metabolic) != 0) return -1;
    float atp = metabolic.atp_level, metabolic_cap = metabolic.metabolic_capacity, min_cap = bridge->config.min_capacity;
    /* ATP affects coordination and procedural learning */
    if (bridge->config.enable_atp_modulation) {
        bridge->effects.motor_coordination = clamp_f(atp * bridge->config.atp_sensitivity, min_cap, 1.0f);
        bridge->effects.procedural_learning = clamp_f(atp * 1.1f * bridge->config.atp_sensitivity, min_cap, 1.0f);
    }
    /* Fatigue affects timing precision and error correction */
    if (bridge->config.enable_fatigue_modulation) {
        bridge->effects.timing_precision = clamp_f(metabolic_cap * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
        bridge->effects.error_correction = clamp_f(metabolic_cap * 0.9f * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
    }
    bridge->effects.overall_capacity = (bridge->effects.motor_coordination + bridge->effects.timing_precision +
                                        bridge->effects.procedural_learning + bridge->effects.error_correction) / 4.0f;
    bridge->update_count++;
    return 0;
}

int cerebellum_substrate_bridge_get_effects(const cerebellum_substrate_bridge_t* bridge, cerebellum_substrate_effects_t* effects) {
    if (!bridge || !effects) return -1;
    *effects = bridge->effects;
    return 0;
}

int cerebellum_substrate_bridge_apply_effects(cerebellum_substrate_bridge_t* bridge) { return bridge ? 0 : -1; }

int cerebellum_substrate_bridge_register_bio_async(cerebellum_substrate_bridge_t* bridge, bio_router_t* router) {
    if (!bridge) return -1;
    bridge->router = router;
    bridge->bio_async_connected = (router != NULL);
    return 0;
}
