/**
 * @file nimcp_game_theory_substrate_bridge.c
 * @brief Game Theory-Neural Substrate Bridge Implementation
 *
 * WHAT: Links game theory to metabolic state
 * WHY: Strategic reasoning requires sustained prefrontal resources
 * HOW: Monitors ATP/fatigue; modulates strategy depth, opponent modeling
 */

#include "cognitive/game_theory/nimcp_game_theory_substrate_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include <string.h>
#include <math.h>

struct game_theory_substrate_bridge {
    void* game_theory;
    neural_substrate_t* substrate;
    game_theory_substrate_config_t config;
    game_theory_substrate_effects_t effects;
    bio_router_t* router;
    bool bio_async_connected;
    uint64_t update_count;
};

static float clamp_f(float v, float min, float max) {
    return v < min ? min : (v > max ? max : v);
}

game_theory_substrate_config_t game_theory_substrate_default_config(void) {
    game_theory_substrate_config_t cfg = {
        .enable_atp_modulation = true,
        .enable_fatigue_modulation = true,
        .enable_bio_async = false,
        .atp_sensitivity = 1.0f,
        .fatigue_sensitivity = 1.0f,
        .min_capacity = 0.2f
    };
    return cfg;
}

game_theory_substrate_bridge_t* game_theory_substrate_bridge_create(void* game_theory, neural_substrate_t* substrate, const game_theory_substrate_config_t* config) {
    if (!substrate) return NULL;

    game_theory_substrate_bridge_t* bridge = nimcp_calloc(1, sizeof(game_theory_substrate_bridge_t));
    if (!bridge) return NULL;

    bridge->game_theory = game_theory;
    bridge->substrate = substrate;
    bridge->config = config ? *config : game_theory_substrate_default_config();

    bridge->effects.strategy_depth = 1.0f;
    bridge->effects.opponent_modeling = 1.0f;
    bridge->effects.fairness_consideration = 1.0f;
    bridge->effects.equilibrium_search = 1.0f;
    bridge->effects.overall_capacity = 1.0f;

    return bridge;
}

void game_theory_substrate_bridge_destroy(game_theory_substrate_bridge_t* bridge) {
    if (bridge) nimcp_free(bridge);
}

int game_theory_substrate_bridge_update(game_theory_substrate_bridge_t* bridge) {
    if (!bridge || !bridge->substrate) return -1;

    substrate_metabolic_state_t metabolic;
    if (substrate_get_metabolic_state(bridge->substrate, &metabolic) != 0) return -1;

    float atp = metabolic.atp_level;
    float metabolic_cap = metabolic.metabolic_capacity;
    float min_cap = bridge->config.min_capacity;

    if (bridge->config.enable_atp_modulation) {
        /* Strategy depth requires sustained cognitive resources */
        bridge->effects.strategy_depth = clamp_f(atp * bridge->config.atp_sensitivity, min_cap, 1.0f);
        /* Equilibrium search is computationally intensive */
        bridge->effects.equilibrium_search = clamp_f(atp * 0.9f * bridge->config.atp_sensitivity, min_cap, 1.0f);
    }

    if (bridge->config.enable_fatigue_modulation) {
        /* Opponent modeling degrades with fatigue (ToM capacity) */
        bridge->effects.opponent_modeling = clamp_f(metabolic_cap * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
        /* Fairness consideration is vulnerable to fatigue (utilitarian bias increases) */
        bridge->effects.fairness_consideration = clamp_f(metabolic_cap * 0.85f * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
    }

    bridge->effects.overall_capacity = (bridge->effects.strategy_depth +
                                        bridge->effects.opponent_modeling +
                                        bridge->effects.fairness_consideration +
                                        bridge->effects.equilibrium_search) / 4.0f;

    bridge->update_count++;
    return 0;
}

int game_theory_substrate_bridge_get_effects(const game_theory_substrate_bridge_t* bridge, game_theory_substrate_effects_t* effects) {
    if (!bridge || !effects) return -1;
    *effects = bridge->effects;
    return 0;
}

int game_theory_substrate_bridge_apply_effects(game_theory_substrate_bridge_t* bridge) {
    if (!bridge) return -1;
    return 0;
}

int game_theory_substrate_bridge_register_bio_async(game_theory_substrate_bridge_t* bridge, bio_router_t* router) {
    if (!bridge) return -1;
    bridge->router = router;
    bridge->bio_async_connected = (router != NULL);
    return 0;
}
