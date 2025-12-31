/**
 * @file nimcp_game_theory_substrate_bridge.c
 * @brief Game Theory-Neural Substrate Bridge Implementation
 *
 * WHAT: Links game theory to metabolic state
 * WHY: Strategic reasoning requires sustained prefrontal resources
 * HOW: Monitors ATP/fatigue; modulates strategy depth, opponent modeling
 */

#include "cognitive/game_theory/nimcp_game_theory_substrate_bridge.h"
#include "cognitive/common/nimcp_metabolic_modulation.h"
#include "async/nimcp_bio_messages.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include <string.h>
#include <math.h>

struct game_theory_substrate_bridge {
    void* game_theory;
    neural_substrate_t* substrate;
    game_theory_substrate_config_t config;
    game_theory_substrate_effects_t effects;
    bio_module_context_t ctx;
    bio_router_t* router;
    bool bio_async_connected;
    uint64_t update_count;
    float prev_overall_capacity;
};

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
        bridge->effects.strategy_depth = nimcp_clamp_f(atp * bridge->config.atp_sensitivity, min_cap, 1.0f);
        /* Equilibrium search is computationally intensive */
        bridge->effects.equilibrium_search = nimcp_clamp_f(atp * 0.9f * bridge->config.atp_sensitivity, min_cap, 1.0f);
    }

    if (bridge->config.enable_fatigue_modulation) {
        /* Opponent modeling degrades with fatigue (ToM capacity) */
        bridge->effects.opponent_modeling = nimcp_clamp_f(metabolic_cap * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
        /* Fairness consideration is vulnerable to fatigue (utilitarian bias increases) */
        bridge->effects.fairness_consideration = nimcp_clamp_f(metabolic_cap * 0.85f * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
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

    if (!bridge->bio_async_connected || !bridge->ctx) {
        return 0;
    }

    substrate_metabolic_state_t metabolic;
    float atp_level = 1.0f, fatigue_level = 0.0f;
    if (bridge->substrate && substrate_get_metabolic_state(bridge->substrate, &metabolic) == 0) {
        atp_level = metabolic.atp_level;
        fatigue_level = 1.0f - metabolic.metabolic_capacity;
    }

    bio_msg_substrate_modulation_t msg;
    memset(&msg, 0, sizeof(msg));
    bio_msg_init_header(&msg.header, BIO_MSG_SUBSTRATE_MODULATION,
                        BIO_MODULE_SUBSTRATE_GAME_THEORY, 0, sizeof(msg));
    msg.header.channel = BIO_CHANNEL_DOPAMINE;  /* Game theory uses dopamine for reward processing */

    msg.bridge_module_id = BIO_MODULE_SUBSTRATE_GAME_THEORY;
    msg.processing_capacity = bridge->effects.strategy_depth;
    msg.overall_capacity = bridge->effects.overall_capacity;
    msg.effect_values[0] = bridge->effects.strategy_depth;
    msg.effect_values[1] = bridge->effects.opponent_modeling;
    msg.effect_values[2] = bridge->effects.fairness_consideration;
    msg.effect_values[3] = bridge->effects.equilibrium_search;
    msg.atp_level = atp_level;
    msg.fatigue_level = fatigue_level;
    msg.update_count = bridge->update_count;
    msg.critical_low = (atp_level < bridge->config.min_capacity);

    bio_router_broadcast(bridge->ctx, &msg, sizeof(msg));

    float delta = bridge->effects.overall_capacity - bridge->prev_overall_capacity;
    if (delta < -0.1f || delta > 0.1f) {
        bio_msg_substrate_capacity_update_t update_msg;
        memset(&update_msg, 0, sizeof(update_msg));
        bio_msg_init_header(&update_msg.header, BIO_MSG_SUBSTRATE_CAPACITY_UPDATE,
                            BIO_MODULE_SUBSTRATE_GAME_THEORY, 0, sizeof(update_msg));
        update_msg.bridge_module_id = BIO_MODULE_SUBSTRATE_GAME_THEORY;
        update_msg.old_capacity = bridge->prev_overall_capacity;
        update_msg.new_capacity = bridge->effects.overall_capacity;
        update_msg.delta = delta;
        update_msg.significant_change = true;
        bio_router_broadcast(bridge->ctx, &update_msg, sizeof(update_msg));
    }

    if (msg.critical_low) {
        bio_msg_substrate_atp_critical_t alert;
        memset(&alert, 0, sizeof(alert));
        bio_msg_init_header(&alert.header, BIO_MSG_SUBSTRATE_ATP_CRITICAL,
                            BIO_MODULE_SUBSTRATE_GAME_THEORY, 0, sizeof(alert));
        alert.header.channel = BIO_CHANNEL_NOREPINEPHRINE;
        alert.bridge_module_id = BIO_MODULE_SUBSTRATE_GAME_THEORY;
        alert.atp_level = atp_level;
        alert.threshold = bridge->config.min_capacity;
        alert.min_capacity = bridge->config.min_capacity;
        bio_router_broadcast(bridge->ctx, &alert, sizeof(alert));
    }

    bridge->prev_overall_capacity = bridge->effects.overall_capacity;
    return 0;
}

int game_theory_substrate_bridge_register_bio_async(game_theory_substrate_bridge_t* bridge, bio_router_t* router) {
    if (!bridge) return -1;

    if (bridge->bio_async_connected && bridge->ctx) {
        bio_router_unregister_module(bridge->ctx);
        bridge->ctx = NULL;
        bridge->bio_async_connected = false;
    }

    if (!router) {
        bridge->router = NULL;
        return 0;
    }

    bio_module_info_t info = {
        .module_id = BIO_MODULE_SUBSTRATE_GAME_THEORY,
        .module_name = "game_theory_substrate_bridge",
        .inbox_capacity = 16,
        .user_data = bridge
    };

    bridge->ctx = bio_router_register_module(&info);
    if (bridge->ctx) {
        bridge->bio_async_connected = true;
    }
    bridge->router = router;
    return 0;
}
