/**
 * @file nimcp_knowledge_substrate_bridge.c
 * @brief Knowledge-Neural Substrate Bridge Implementation
 *
 * WHAT: Links knowledge/semantic memory to metabolic state
 * WHY: Semantic retrieval requires temporal-parietal resources
 * HOW: Monitors ATP/fatigue; modulates retrieval fluency, accuracy, integration
 */

#include "cognitive/knowledge/nimcp_knowledge_substrate_bridge.h"
#include "async/nimcp_bio_messages.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include <string.h>
#include <math.h>

struct knowledge_substrate_bridge {
    void* knowledge;
    neural_substrate_t* substrate;
    knowledge_substrate_config_t config;
    knowledge_substrate_effects_t effects;
    bio_module_context_t ctx;
    bio_router_t* router;
    bool bio_async_connected;
    uint64_t update_count;
    float prev_overall_capacity;
};

static float clamp_f(float v, float min, float max) {
    return v < min ? min : (v > max ? max : v);
}

knowledge_substrate_config_t knowledge_substrate_default_config(void) {
    knowledge_substrate_config_t cfg = {
        .enable_atp_modulation = true,
        .enable_fatigue_modulation = true,
        .enable_bio_async = false,
        .atp_sensitivity = 1.0f,
        .fatigue_sensitivity = 1.0f,
        .min_capacity = 0.2f
    };
    return cfg;
}

knowledge_substrate_bridge_t* knowledge_substrate_bridge_create(void* knowledge, neural_substrate_t* substrate, const knowledge_substrate_config_t* config) {
    if (!substrate) return NULL;

    knowledge_substrate_bridge_t* bridge = nimcp_calloc(1, sizeof(knowledge_substrate_bridge_t));
    if (!bridge) return NULL;

    bridge->knowledge = knowledge;
    bridge->substrate = substrate;
    bridge->config = config ? *config : knowledge_substrate_default_config();

    bridge->effects.retrieval_speed = 1.0f;
    bridge->effects.retrieval_accuracy = 1.0f;
    bridge->effects.consolidation_rate = 1.0f;
    bridge->effects.association_strength = 1.0f;
    bridge->effects.overall_capacity = 1.0f;

    return bridge;
}

void knowledge_substrate_bridge_destroy(knowledge_substrate_bridge_t* bridge) {
    if (!bridge) return;
    if (bridge->bio_async_connected && bridge->ctx) {
        bio_router_unregister_module(bridge->ctx);
    }
    nimcp_free(bridge);
}

int knowledge_substrate_bridge_update(knowledge_substrate_bridge_t* bridge) {
    if (!bridge || !bridge->substrate) return -1;

    substrate_metabolic_state_t metabolic;
    if (substrate_get_metabolic_state(bridge->substrate, &metabolic) != 0) return -1;

    float atp = metabolic.atp_level;
    float metabolic_cap = metabolic.metabolic_capacity;
    float min_cap = bridge->config.min_capacity;

    if (bridge->config.enable_atp_modulation) {
        bridge->effects.retrieval_accuracy = clamp_f(atp * bridge->config.atp_sensitivity, min_cap, 1.0f);
        bridge->effects.association_strength = clamp_f(atp * 1.1f * bridge->config.atp_sensitivity, min_cap, 1.0f);
    }

    if (bridge->config.enable_fatigue_modulation) {
        bridge->effects.retrieval_speed = clamp_f(metabolic_cap * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
        bridge->effects.consolidation_rate = clamp_f(metabolic_cap * 0.95f * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
    }

    bridge->effects.overall_capacity = (bridge->effects.retrieval_speed +
                                        bridge->effects.retrieval_accuracy +
                                        bridge->effects.consolidation_rate +
                                        bridge->effects.association_strength) / 4.0f;

    bridge->update_count++;
    return 0;
}

int knowledge_substrate_bridge_get_effects(const knowledge_substrate_bridge_t* bridge, knowledge_substrate_effects_t* effects) {
    if (!bridge || !effects) return -1;
    *effects = bridge->effects;
    return 0;
}

int knowledge_substrate_bridge_apply_effects(knowledge_substrate_bridge_t* bridge) {
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
                        BIO_MODULE_SUBSTRATE_KNOWLEDGE, 0, sizeof(msg));
    msg.header.channel = BIO_CHANNEL_ACETYLCHOLINE;  /* Knowledge uses acetylcholine */

    msg.bridge_module_id = BIO_MODULE_SUBSTRATE_KNOWLEDGE;
    msg.processing_capacity = bridge->effects.retrieval_speed;
    msg.overall_capacity = bridge->effects.overall_capacity;
    msg.effect_values[0] = bridge->effects.retrieval_speed;
    msg.effect_values[1] = bridge->effects.retrieval_accuracy;
    msg.effect_values[2] = bridge->effects.consolidation_rate;
    msg.effect_values[3] = bridge->effects.association_strength;
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
                            BIO_MODULE_SUBSTRATE_KNOWLEDGE, 0, sizeof(update_msg));
        update_msg.bridge_module_id = BIO_MODULE_SUBSTRATE_KNOWLEDGE;
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
                            BIO_MODULE_SUBSTRATE_KNOWLEDGE, 0, sizeof(alert));
        alert.header.channel = BIO_CHANNEL_NOREPINEPHRINE;
        alert.bridge_module_id = BIO_MODULE_SUBSTRATE_KNOWLEDGE;
        alert.atp_level = atp_level;
        alert.threshold = bridge->config.min_capacity;
        alert.min_capacity = bridge->config.min_capacity;
        bio_router_broadcast(bridge->ctx, &alert, sizeof(alert));
    }

    bridge->prev_overall_capacity = bridge->effects.overall_capacity;
    return 0;
}

int knowledge_substrate_bridge_register_bio_async(knowledge_substrate_bridge_t* bridge, bio_router_t* router) {
    if (!bridge) return -1;

    if (bridge->bio_async_connected && bridge->ctx) {
        bio_router_unregister_module(bridge->ctx);
        bridge->ctx = NULL;
        bridge->bio_async_connected = false;
    }

    if (!router) {
        return 0;
    }

    bio_module_info_t info = {
        .module_id = BIO_MODULE_SUBSTRATE_KNOWLEDGE,
        .module_name = "knowledge_substrate_bridge",
        .inbox_capacity = 16,
        .user_data = bridge
    };

    bridge->ctx = bio_router_register_module(&info);
    if (bridge->ctx) {
        bridge->bio_async_connected = true;
    }
    return 0;
}
