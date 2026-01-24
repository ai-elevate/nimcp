/**
 * @file nimcp_auditory_substrate_bridge.c
 * @brief Auditory-Neural Substrate Bridge Implementation
 */

#include "utils/bridge/nimcp_bridge_base.h"
#include "core/sensory/auditory/nimcp_auditory_substrate_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>

struct auditory_substrate_bridge {
    bridge_base_t base;              /**< MUST be first: base bridge infrastructure */
    void* auditory;
    neural_substrate_t* substrate;
    auditory_substrate_config_t config;
    auditory_substrate_effects_t effects;
    bio_router_t* router;
    bool bio_async_connected;
    uint64_t update_count;
};

static float clamp_f(float v, float min, float max) { return v < min ? min : (v > max ? max : v); }

auditory_substrate_config_t auditory_substrate_default_config(void) {
    auditory_substrate_config_t cfg = { .enable_atp_modulation = true, .enable_fatigue_modulation = true,
        .enable_bio_async = false, .atp_sensitivity = 1.0f, .fatigue_sensitivity = 1.0f, .min_capacity = 0.2f };
    return cfg;
}

auditory_substrate_bridge_t* auditory_substrate_bridge_create(void* auditory, neural_substrate_t* substrate, const auditory_substrate_config_t* config) {
    if (!substrate) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "substrate is NULL");

        return NULL;

    }
    auditory_substrate_bridge_t* bridge = nimcp_calloc(1, sizeof(auditory_substrate_bridge_t));
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return NULL;

    }
    bridge->auditory = auditory;
    bridge->substrate = substrate;
    bridge->config = config ? *config : auditory_substrate_default_config();
    bridge->effects.frequency_resolution = 1.0f;
    bridge->effects.temporal_precision = 1.0f;
    bridge->effects.speech_processing = 1.0f;
    bridge->effects.noise_filtering = 1.0f;
    bridge->effects.overall_capacity = 1.0f;
    return bridge;
}

void auditory_substrate_bridge_destroy(auditory_substrate_bridge_t* bridge) { if (bridge) nimcp_free(bridge); }

int auditory_substrate_bridge_update(auditory_substrate_bridge_t* bridge) {
    if (!bridge || !bridge->substrate) return -1;
    substrate_metabolic_state_t metabolic;
    if (substrate_get_metabolic_state(bridge->substrate, &metabolic) != 0) return -1;
    float atp = metabolic.atp_level, metabolic_cap = metabolic.metabolic_capacity, min_cap = bridge->config.min_capacity;
    /* ATP affects frequency resolution and temporal precision */
    if (bridge->config.enable_atp_modulation) {
        bridge->effects.frequency_resolution = clamp_f(atp * bridge->config.atp_sensitivity, min_cap, 1.0f);
        bridge->effects.temporal_precision = clamp_f(atp * 1.1f * bridge->config.atp_sensitivity, min_cap, 1.0f);
    }
    /* Fatigue affects speech processing and noise filtering */
    if (bridge->config.enable_fatigue_modulation) {
        bridge->effects.speech_processing = clamp_f(metabolic_cap * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
        bridge->effects.noise_filtering = clamp_f(metabolic_cap * 0.85f * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
    }
    bridge->effects.overall_capacity = (bridge->effects.frequency_resolution + bridge->effects.temporal_precision +
                                        bridge->effects.speech_processing + bridge->effects.noise_filtering) / 4.0f;
    bridge->update_count++;
    return 0;
}

int auditory_substrate_bridge_get_effects(const auditory_substrate_bridge_t* bridge, auditory_substrate_effects_t* effects) {
    if (!bridge || !effects) return -1;
    *effects = bridge->effects;
    return 0;
}

int auditory_substrate_bridge_apply_effects(auditory_substrate_bridge_t* bridge) { return bridge ? 0 : -1; }

int auditory_substrate_bridge_register_bio_async(auditory_substrate_bridge_t* bridge, bio_router_t* router) {
    if (!bridge) return -1;
    bridge->router = router;
    bridge->bio_async_connected = (router != NULL);
    return 0;
}
