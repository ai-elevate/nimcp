/**
 * @file nimcp_bias_substrate_bridge.c
 * @brief Cognitive Bias-Neural Substrate Bridge Implementation
 */

#include "cognitive/bias/nimcp_bias_substrate_bridge.h"
#include "async/nimcp_bio_messages.h"
#include "utils/memory/nimcp_memory.h"
#include <string.h>

struct bias_substrate_bridge {
    void* bias;
    neural_substrate_t* substrate;
    bias_substrate_config_t config;
    bias_substrate_effects_t effects;
    bio_module_context_t ctx;
    bio_router_t* router;
    bool bio_async_connected;
    uint64_t update_count;
    float prev_overall_capacity;
};

static float clamp_f(float v, float min, float max) { return v < min ? min : (v > max ? max : v); }

bias_substrate_config_t bias_substrate_default_config(void) {
    bias_substrate_config_t cfg = { .enable_atp_modulation = true, .enable_fatigue_modulation = true,
        .enable_bio_async = false, .atp_sensitivity = 1.0f, .fatigue_sensitivity = 1.0f, .min_capacity = 0.2f };
    return cfg;
}

bias_substrate_bridge_t* bias_substrate_bridge_create(void* bias, neural_substrate_t* substrate, const bias_substrate_config_t* config) {
    if (!substrate) return NULL;
    bias_substrate_bridge_t* bridge = nimcp_calloc(1, sizeof(bias_substrate_bridge_t));
    if (!bridge) return NULL;
    bridge->bias = bias;
    bridge->substrate = substrate;
    bridge->config = config ? *config : bias_substrate_default_config();
    bridge->effects.bias_detection = 1.0f;
    bridge->effects.correction_strength = 1.0f;
    bridge->effects.metacognitive_oversight = 1.0f;
    bridge->effects.heuristic_resistance = 1.0f;
    bridge->effects.overall_capacity = 1.0f;
    return bridge;
}

void bias_substrate_bridge_destroy(bias_substrate_bridge_t* bridge) {
    if (!bridge) return;
    if (bridge->bio_async_connected && bridge->ctx) {
        bio_router_unregister_module(bridge->ctx);
    }
    nimcp_free(bridge);
}

int bias_substrate_bridge_update(bias_substrate_bridge_t* bridge) {
    if (!bridge || !bridge->substrate) return -1;
    substrate_metabolic_state_t metabolic;
    if (substrate_get_metabolic_state(bridge->substrate, &metabolic) != 0) return -1;
    float atp = metabolic.atp_level, metabolic_cap = metabolic.metabolic_capacity, min_cap = bridge->config.min_capacity;
    if (bridge->config.enable_atp_modulation) {
        bridge->effects.bias_detection = clamp_f(atp * bridge->config.atp_sensitivity, min_cap, 1.0f);
        bridge->effects.correction_strength = clamp_f(atp * 1.1f * bridge->config.atp_sensitivity, min_cap, 1.0f);
    }
    if (bridge->config.enable_fatigue_modulation) {
        bridge->effects.metacognitive_oversight = clamp_f(metabolic_cap * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
        bridge->effects.heuristic_resistance = clamp_f(metabolic_cap * 0.85f * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
    }
    bridge->effects.overall_capacity = (bridge->effects.bias_detection + bridge->effects.correction_strength +
                                        bridge->effects.metacognitive_oversight + bridge->effects.heuristic_resistance) / 4.0f;
    bridge->update_count++;
    return 0;
}

int bias_substrate_bridge_get_effects(const bias_substrate_bridge_t* bridge, bias_substrate_effects_t* effects) {
    if (!bridge || !effects) return -1;
    *effects = bridge->effects;
    return 0;
}

int bias_substrate_bridge_apply_effects(bias_substrate_bridge_t* bridge) {
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
    bio_msg_init_header(&msg.header, BIO_MSG_SUBSTRATE_MODULATION, BIO_MODULE_SUBSTRATE_BIAS, 0, sizeof(msg));
    msg.header.channel = BIO_CHANNEL_NOREPINEPHRINE;  /* Bias detection uses norepinephrine for alertness */
    msg.bridge_module_id = BIO_MODULE_SUBSTRATE_BIAS;
    msg.processing_capacity = bridge->effects.bias_detection;
    msg.overall_capacity = bridge->effects.overall_capacity;
    msg.effect_values[0] = bridge->effects.bias_detection;
    msg.effect_values[1] = bridge->effects.correction_strength;
    msg.effect_values[2] = bridge->effects.metacognitive_oversight;
    msg.effect_values[3] = bridge->effects.heuristic_resistance;
    msg.atp_level = atp_level;
    msg.fatigue_level = fatigue_level;
    msg.update_count = bridge->update_count;
    msg.critical_low = (atp_level < bridge->config.min_capacity);
    bio_router_broadcast(bridge->ctx, &msg, sizeof(msg));

    float delta = bridge->effects.overall_capacity - bridge->prev_overall_capacity;
    if (delta < -0.1f || delta > 0.1f) {
        bio_msg_substrate_capacity_update_t update_msg;
        memset(&update_msg, 0, sizeof(update_msg));
        bio_msg_init_header(&update_msg.header, BIO_MSG_SUBSTRATE_CAPACITY_UPDATE, BIO_MODULE_SUBSTRATE_BIAS, 0, sizeof(update_msg));
        update_msg.bridge_module_id = BIO_MODULE_SUBSTRATE_BIAS;
        update_msg.old_capacity = bridge->prev_overall_capacity;
        update_msg.new_capacity = bridge->effects.overall_capacity;
        update_msg.delta = delta;
        update_msg.significant_change = true;
        bio_router_broadcast(bridge->ctx, &update_msg, sizeof(update_msg));
    }
    bridge->prev_overall_capacity = bridge->effects.overall_capacity;
    return 0;
}

int bias_substrate_bridge_register_bio_async(bias_substrate_bridge_t* bridge, bio_router_t* router) {
    if (!bridge) return -1;
    if (bridge->bio_async_connected && bridge->ctx) {
        bio_router_unregister_module(bridge->ctx);
        bridge->ctx = NULL;
        bridge->bio_async_connected = false;
    }
    if (!router) return 0;
    bio_module_info_t info = { .module_id = BIO_MODULE_SUBSTRATE_BIAS, .module_name = "bias_substrate_bridge", .inbox_capacity = 16, .user_data = bridge };
    bridge->ctx = bio_router_register_module(&info);
    if (bridge->ctx) bridge->bio_async_connected = true;
    return 0;
}
