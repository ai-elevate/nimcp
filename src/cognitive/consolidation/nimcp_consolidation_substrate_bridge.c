/**
 * @file nimcp_consolidation_substrate_bridge.c
 * @brief Memory Consolidation-Neural Substrate Bridge Implementation
 *
 * WHAT: Links memory consolidation to metabolic state
 * WHY: Consolidation requires sustained energy for synaptic protein synthesis
 * HOW: Monitors ATP/fatigue; modulates rate, fidelity, and memory priority
 */

#include "cognitive/consolidation/nimcp_consolidation_substrate_bridge.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/platform/nimcp_platform_mutex.h"
#include <string.h>
#include <math.h>

struct consolidation_substrate_bridge {
    bridge_base_t base;                 /* MUST be first: base bridge infrastructure */
    void* consolidation;
    neural_substrate_t* substrate;
    consolidation_substrate_config_t config;
    consolidation_substrate_effects_t effects;
    bio_router_t* router;
    bool bio_async_connected;
    uint64_t update_count;
};

static float clamp_f(float v, float min, float max) {
    return v < min ? min : (v > max ? max : v);
}

consolidation_substrate_config_t consolidation_substrate_default_config(void) {
    consolidation_substrate_config_t cfg = {
        .enable_atp_modulation = true,
        .enable_fatigue_modulation = true,
        .enable_bio_async = false,
        .atp_sensitivity = 1.0f,
        .fatigue_sensitivity = 1.0f,
        .min_capacity = 0.2f
    };
    return cfg;
}

consolidation_substrate_bridge_t* consolidation_substrate_bridge_create(void* consolidation, neural_substrate_t* substrate, const consolidation_substrate_config_t* config) {
    if (!substrate) return NULL;

    consolidation_substrate_bridge_t* bridge = nimcp_calloc(1, sizeof(consolidation_substrate_bridge_t));
    if (!bridge) return NULL;

    bridge->consolidation = consolidation;
    bridge->substrate = substrate;
    bridge->config = config ? *config : consolidation_substrate_default_config();

    /* Initialize mutex */
    bridge->base.mutex = (nimcp_mutex_t*)nimcp_malloc(sizeof(nimcp_mutex_t));
    if (!bridge->base.mutex) {
        NIMCP_LOGGING_ERROR("Failed to allocate mutex for consolidation substrate bridge");
        nimcp_free(bridge);
        return NULL;
    }

    if (nimcp_platform_mutex_init(bridge->base.mutex, false) != 0) {
        NIMCP_LOGGING_ERROR("Failed to initialize mutex for consolidation substrate bridge");
        nimcp_free(bridge->base.mutex);
        nimcp_free(bridge);
        return NULL;
    }

    bridge->effects.consolidation_rate = 1.0f;
    bridge->effects.encoding_fidelity = 1.0f;
    bridge->effects.priority_threshold = 1.0f;
    bridge->effects.protein_synthesis = 1.0f;
    bridge->effects.overall_capacity = 1.0f;

    return bridge;
}

void consolidation_substrate_bridge_destroy(consolidation_substrate_bridge_t* bridge) {
    if (!bridge) return;

    /* Destroy mutex */
    if (bridge->base.mutex) {
        nimcp_platform_mutex_destroy(bridge->base.mutex);
        nimcp_free(bridge->base.mutex);
    }

    nimcp_free(bridge);
}

int consolidation_substrate_bridge_update(consolidation_substrate_bridge_t* bridge) {
    if (!bridge || !bridge->substrate) return -1;

    nimcp_platform_mutex_lock(bridge->base.mutex);

    substrate_metabolic_state_t metabolic;
    if (substrate_get_metabolic_state(bridge->substrate, &metabolic) != 0) {
        nimcp_platform_mutex_unlock(bridge->base.mutex);
        return -1;
    }

    float atp = metabolic.atp_level;
    float metabolic_cap = metabolic.metabolic_capacity;
    float min_cap = bridge->config.min_capacity;

    if (bridge->config.enable_atp_modulation) {
        /* Consolidation rate heavily dependent on ATP for protein synthesis */
        bridge->effects.consolidation_rate = clamp_f(atp * 1.2f * bridge->config.atp_sensitivity, min_cap, 1.0f);
        bridge->effects.protein_synthesis = clamp_f(atp * bridge->config.atp_sensitivity, min_cap, 1.0f);
    }

    if (bridge->config.enable_fatigue_modulation) {
        /* Encoding fidelity decreases with fatigue */
        bridge->effects.encoding_fidelity = clamp_f(metabolic_cap * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
        /* Priority threshold increases (more selective) under metabolic stress */
        bridge->effects.priority_threshold = clamp_f(1.0f - (metabolic_cap * 0.3f), min_cap, 1.0f);
    }

    bridge->effects.overall_capacity = (bridge->effects.consolidation_rate +
                                        bridge->effects.encoding_fidelity +
                                        bridge->effects.priority_threshold +
                                        bridge->effects.protein_synthesis) / 4.0f;

    bridge->update_count++;

    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return 0;
}

int consolidation_substrate_bridge_get_effects(const consolidation_substrate_bridge_t* bridge, consolidation_substrate_effects_t* effects) {
    if (!bridge || !effects) return -1;
    *effects = bridge->effects;
    return 0;
}

int consolidation_substrate_bridge_apply_effects(consolidation_substrate_bridge_t* bridge) {
    if (!bridge) return -1;
    return 0;
}

int consolidation_substrate_bridge_register_bio_async(consolidation_substrate_bridge_t* bridge, bio_router_t* router) {
    if (!bridge) return -1;
    bridge->router = router;
    bridge->bio_async_connected = (router != NULL);
    return 0;
}
