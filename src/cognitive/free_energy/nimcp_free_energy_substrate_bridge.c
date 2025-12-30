/**
 * @file nimcp_free_energy_substrate_bridge.c
 * @brief Free Energy Principle-Neural Substrate Bridge Implementation
 *
 * WHAT: Links FEP to metabolic state
 * WHY: Variational inference requires sustained computational resources
 * HOW: Monitors ATP/fatigue; modulates precision, depth, active inference
 */

#include "cognitive/free_energy/nimcp_free_energy_substrate_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include <string.h>
#include <math.h>

struct free_energy_substrate_bridge {
    void* free_energy;
    neural_substrate_t* substrate;
    free_energy_substrate_config_t config;
    free_energy_substrate_effects_t effects;
    bio_router_t* router;
    bool bio_async_connected;
    uint64_t update_count;
};

static float clamp_f(float v, float min, float max) {
    return v < min ? min : (v > max ? max : v);
}

free_energy_substrate_config_t free_energy_substrate_default_config(void) {
    free_energy_substrate_config_t cfg = {
        .enable_atp_modulation = true,
        .enable_fatigue_modulation = true,
        .enable_bio_async = false,
        .atp_sensitivity = 1.0f,
        .fatigue_sensitivity = 1.0f,
        .min_capacity = 0.2f
    };
    return cfg;
}

free_energy_substrate_bridge_t* free_energy_substrate_bridge_create(void* free_energy, neural_substrate_t* substrate, const free_energy_substrate_config_t* config) {
    if (!substrate) return NULL;

    free_energy_substrate_bridge_t* bridge = nimcp_calloc(1, sizeof(free_energy_substrate_bridge_t));
    if (!bridge) return NULL;

    bridge->free_energy = free_energy;
    bridge->substrate = substrate;
    bridge->config = config ? *config : free_energy_substrate_default_config();

    bridge->effects.precision_weighting = 1.0f;
    bridge->effects.prediction_depth = 1.0f;
    bridge->effects.active_inference = 1.0f;
    bridge->effects.model_complexity = 1.0f;
    bridge->effects.overall_capacity = 1.0f;

    return bridge;
}

void free_energy_substrate_bridge_destroy(free_energy_substrate_bridge_t* bridge) {
    if (bridge) nimcp_free(bridge);
}

int free_energy_substrate_bridge_update(free_energy_substrate_bridge_t* bridge) {
    if (!bridge || !bridge->substrate) return -1;

    substrate_metabolic_state_t metabolic;
    if (substrate_get_metabolic_state(bridge->substrate, &metabolic) != 0) return -1;

    float atp = metabolic.atp_level;
    float metabolic_cap = metabolic.metabolic_capacity;
    float min_cap = bridge->config.min_capacity;

    if (bridge->config.enable_atp_modulation) {
        /* Precision weighting requires stable neural resources */
        bridge->effects.precision_weighting = clamp_f(atp * bridge->config.atp_sensitivity, min_cap, 1.0f);
        /* Active inference is computationally demanding */
        bridge->effects.active_inference = clamp_f(atp * 0.95f * bridge->config.atp_sensitivity, min_cap, 1.0f);
    }

    if (bridge->config.enable_fatigue_modulation) {
        /* Prediction depth decreases with fatigue */
        bridge->effects.prediction_depth = clamp_f(metabolic_cap * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
        /* Model complexity simplifies under metabolic stress */
        bridge->effects.model_complexity = clamp_f(metabolic_cap * 0.85f * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
    }

    bridge->effects.overall_capacity = (bridge->effects.precision_weighting +
                                        bridge->effects.prediction_depth +
                                        bridge->effects.active_inference +
                                        bridge->effects.model_complexity) / 4.0f;

    bridge->update_count++;
    return 0;
}

int free_energy_substrate_bridge_get_effects(const free_energy_substrate_bridge_t* bridge, free_energy_substrate_effects_t* effects) {
    if (!bridge || !effects) return -1;
    *effects = bridge->effects;
    return 0;
}

int free_energy_substrate_bridge_apply_effects(free_energy_substrate_bridge_t* bridge) {
    if (!bridge) return -1;
    return 0;
}

int free_energy_substrate_bridge_register_bio_async(free_energy_substrate_bridge_t* bridge, bio_router_t* router) {
    if (!bridge) return -1;
    bridge->router = router;
    bridge->bio_async_connected = (router != NULL);
    return 0;
}
