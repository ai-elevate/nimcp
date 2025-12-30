/**
 * @file nimcp_symbolic_logic_substrate_bridge.c
 * @brief Symbolic Logic-Neural Substrate Bridge Implementation
 */

#include "cognitive/symbolic_logic/nimcp_symbolic_logic_substrate_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include <string.h>

struct symbolic_logic_substrate_bridge {
    void* symbolic_logic;
    neural_substrate_t* substrate;
    symbolic_logic_substrate_config_t config;
    symbolic_logic_substrate_effects_t effects;
    bio_router_t* router;
    bool bio_async_connected;
    uint64_t update_count;
};

static float clamp_f(float v, float min, float max) { return v < min ? min : (v > max ? max : v); }

symbolic_logic_substrate_config_t symbolic_logic_substrate_default_config(void) {
    symbolic_logic_substrate_config_t cfg = { .enable_atp_modulation = true, .enable_fatigue_modulation = true,
        .enable_bio_async = false, .atp_sensitivity = 1.0f, .fatigue_sensitivity = 1.0f, .min_capacity = 0.2f };
    return cfg;
}

symbolic_logic_substrate_bridge_t* symbolic_logic_substrate_bridge_create(void* symbolic_logic, neural_substrate_t* substrate, const symbolic_logic_substrate_config_t* config) {
    if (!substrate) return NULL;
    symbolic_logic_substrate_bridge_t* bridge = nimcp_calloc(1, sizeof(symbolic_logic_substrate_bridge_t));
    if (!bridge) return NULL;
    bridge->symbolic_logic = symbolic_logic;
    bridge->substrate = substrate;
    bridge->config = config ? *config : symbolic_logic_substrate_default_config();
    bridge->effects.symbol_manipulation = 1.0f;
    bridge->effects.rule_application = 1.0f;
    bridge->effects.inference_depth = 1.0f;
    bridge->effects.abstraction_level = 1.0f;
    bridge->effects.overall_capacity = 1.0f;
    return bridge;
}

void symbolic_logic_substrate_bridge_destroy(symbolic_logic_substrate_bridge_t* bridge) { if (bridge) nimcp_free(bridge); }

int symbolic_logic_substrate_bridge_update(symbolic_logic_substrate_bridge_t* bridge) {
    if (!bridge || !bridge->substrate) return -1;
    substrate_metabolic_state_t metabolic;
    if (substrate_get_metabolic_state(bridge->substrate, &metabolic) != 0) return -1;
    float atp = metabolic.atp_level, metabolic_cap = metabolic.metabolic_capacity, min_cap = bridge->config.min_capacity;
    /* ATP enables symbol manipulation and inference depth */
    if (bridge->config.enable_atp_modulation) {
        bridge->effects.symbol_manipulation = clamp_f(atp * bridge->config.atp_sensitivity, min_cap, 1.0f);
        bridge->effects.inference_depth = clamp_f(atp * 1.1f * bridge->config.atp_sensitivity, min_cap, 1.0f);
    }
    /* Low fatigue enables rule application and abstraction */
    if (bridge->config.enable_fatigue_modulation) {
        bridge->effects.rule_application = clamp_f(metabolic_cap * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
        bridge->effects.abstraction_level = clamp_f(metabolic_cap * 0.9f * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
    }
    bridge->effects.overall_capacity = (bridge->effects.symbol_manipulation + bridge->effects.rule_application +
                                        bridge->effects.inference_depth + bridge->effects.abstraction_level) / 4.0f;
    bridge->update_count++;
    return 0;
}

int symbolic_logic_substrate_bridge_get_effects(const symbolic_logic_substrate_bridge_t* bridge, symbolic_logic_substrate_effects_t* effects) {
    if (!bridge || !effects) return -1;
    *effects = bridge->effects;
    return 0;
}

int symbolic_logic_substrate_bridge_apply_effects(symbolic_logic_substrate_bridge_t* bridge) { return bridge ? 0 : -1; }

int symbolic_logic_substrate_bridge_register_bio_async(symbolic_logic_substrate_bridge_t* bridge, bio_router_t* router) {
    if (!bridge) return -1;
    bridge->router = router;
    bridge->bio_async_connected = (router != NULL);
    return 0;
}
