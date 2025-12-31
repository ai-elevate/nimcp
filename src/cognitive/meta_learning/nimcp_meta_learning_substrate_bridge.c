/**
 * @file nimcp_meta_learning_substrate_bridge.c
 * @brief Meta-Learning-Neural Substrate Bridge Implementation
 */

#include "cognitive/meta_learning/nimcp_meta_learning_substrate_bridge.h"
#include "cognitive/common/nimcp_metabolic_modulation.h"
#include "async/nimcp_bio_messages.h"
#include "utils/memory/nimcp_memory.h"
#include <string.h>

struct meta_learning_substrate_bridge {
    void* meta_learning;
    neural_substrate_t* substrate;
    meta_learning_substrate_config_t config;
    meta_learning_substrate_effects_t effects;
    bio_module_context_t ctx;
    bio_router_t* router;
    bool bio_async_connected;
    uint64_t update_count;
    float prev_overall_capacity;
};

meta_learning_substrate_config_t meta_learning_substrate_default_config(void) {
    meta_learning_substrate_config_t cfg = { .enable_atp_modulation = true, .enable_fatigue_modulation = true,
        .enable_bio_async = false, .atp_sensitivity = 1.0f, .fatigue_sensitivity = 1.0f, .min_capacity = 0.2f };
    return cfg;
}

meta_learning_substrate_bridge_t* meta_learning_substrate_bridge_create(void* meta_learning, neural_substrate_t* substrate, const meta_learning_substrate_config_t* config) {
    if (!substrate) return NULL;
    meta_learning_substrate_bridge_t* bridge = nimcp_calloc(1, sizeof(meta_learning_substrate_bridge_t));
    if (!bridge) return NULL;
    bridge->meta_learning = meta_learning;
    bridge->substrate = substrate;
    bridge->config = config ? *config : meta_learning_substrate_default_config();
    bridge->effects.learning_rate_adapt = 1.0f;
    bridge->effects.strategy_flexibility = 1.0f;
    bridge->effects.transfer_capacity = 1.0f;
    bridge->effects.plasticity_level = 1.0f;
    bridge->effects.overall_capacity = 1.0f;
    return bridge;
}

void meta_learning_substrate_bridge_destroy(meta_learning_substrate_bridge_t* bridge) {
    if (!bridge) return;
    if (bridge->bio_async_connected && bridge->ctx) {
        bio_router_unregister_module(bridge->ctx);
    }
    nimcp_free(bridge);
}

int meta_learning_substrate_bridge_update(meta_learning_substrate_bridge_t* bridge) {
    if (!bridge || !bridge->substrate) return -1;
    substrate_metabolic_state_t metabolic;
    if (substrate_get_metabolic_state(bridge->substrate, &metabolic) != 0) return -1;
    float atp = metabolic.atp_level, metabolic_cap = metabolic.metabolic_capacity, min_cap = bridge->config.min_capacity;
    if (bridge->config.enable_atp_modulation) {
        bridge->effects.learning_rate_adapt = nimcp_clamp_f(atp * bridge->config.atp_sensitivity, min_cap, 1.0f);
        bridge->effects.plasticity_level = nimcp_clamp_f(atp * 1.1f * bridge->config.atp_sensitivity, min_cap, 1.0f);
    }
    if (bridge->config.enable_fatigue_modulation) {
        bridge->effects.strategy_flexibility = nimcp_clamp_f(metabolic_cap * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
        bridge->effects.transfer_capacity = nimcp_clamp_f(metabolic_cap * 0.9f * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
    }
    bridge->effects.overall_capacity = (bridge->effects.learning_rate_adapt + bridge->effects.strategy_flexibility +
                                        bridge->effects.transfer_capacity + bridge->effects.plasticity_level) / 4.0f;
    bridge->update_count++;
    return 0;
}

int meta_learning_substrate_bridge_get_effects(const meta_learning_substrate_bridge_t* bridge, meta_learning_substrate_effects_t* effects) {
    if (!bridge || !effects) return -1;
    *effects = bridge->effects;
    return 0;
}

int meta_learning_substrate_bridge_apply_effects(meta_learning_substrate_bridge_t* bridge) {
    if (!bridge) return -1;
    if (!bridge->bio_async_connected || !bridge->ctx) return 0;

    substrate_metabolic_state_t metabolic;
    float atp_level = 1.0f, fatigue_level = 0.0f;
    if (bridge->substrate && substrate_get_metabolic_state(bridge->substrate, &metabolic) == 0) {
        atp_level = metabolic.atp_level;
        fatigue_level = 1.0f - metabolic.metabolic_capacity;
    }

    bio_msg_substrate_modulation_t msg;
    memset(&msg, 0, sizeof(msg));
    bio_msg_init_header(&msg.header, BIO_MSG_SUBSTRATE_MODULATION, BIO_MODULE_SUBSTRATE_META_LEARNING, 0, sizeof(msg));
    msg.header.channel = BIO_CHANNEL_DOPAMINE;
    msg.bridge_module_id = BIO_MODULE_SUBSTRATE_META_LEARNING;
    msg.processing_capacity = bridge->effects.learning_rate_adapt;
    msg.overall_capacity = bridge->effects.overall_capacity;
    msg.effect_values[0] = bridge->effects.learning_rate_adapt;
    msg.effect_values[1] = bridge->effects.strategy_flexibility;
    msg.effect_values[2] = bridge->effects.transfer_capacity;
    msg.effect_values[3] = bridge->effects.plasticity_level;
    msg.atp_level = atp_level;
    msg.fatigue_level = fatigue_level;
    msg.update_count = bridge->update_count;
    msg.critical_low = (atp_level < bridge->config.min_capacity);
    bio_router_broadcast(bridge->ctx, &msg, sizeof(msg));

    float delta = bridge->effects.overall_capacity - bridge->prev_overall_capacity;
    if (delta < -0.1f || delta > 0.1f) {
        bio_msg_substrate_capacity_update_t update_msg;
        memset(&update_msg, 0, sizeof(update_msg));
        bio_msg_init_header(&update_msg.header, BIO_MSG_SUBSTRATE_CAPACITY_UPDATE, BIO_MODULE_SUBSTRATE_META_LEARNING, 0, sizeof(update_msg));
        update_msg.bridge_module_id = BIO_MODULE_SUBSTRATE_META_LEARNING;
        update_msg.old_capacity = bridge->prev_overall_capacity;
        update_msg.new_capacity = bridge->effects.overall_capacity;
        update_msg.delta = delta;
        update_msg.significant_change = true;
        bio_router_broadcast(bridge->ctx, &update_msg, sizeof(update_msg));
    }
    bridge->prev_overall_capacity = bridge->effects.overall_capacity;
    return 0;
}

int meta_learning_substrate_bridge_register_bio_async(meta_learning_substrate_bridge_t* bridge, bio_router_t* router) {
    if (!bridge) return -1;
    if (bridge->bio_async_connected && bridge->ctx) {
        bio_router_unregister_module(bridge->ctx);
        bridge->ctx = NULL;
        bridge->bio_async_connected = false;
    }
    if (!router) return 0;
    bio_module_info_t info = { .module_id = BIO_MODULE_SUBSTRATE_META_LEARNING, .module_name = "meta_learning_substrate_bridge", .inbox_capacity = 16, .user_data = bridge };
    bridge->ctx = bio_router_register_module(&info);
    if (bridge->ctx) bridge->bio_async_connected = true;
    return 0;
}
