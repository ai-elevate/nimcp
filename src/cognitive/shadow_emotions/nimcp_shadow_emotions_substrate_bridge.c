/**
 * @file nimcp_shadow_emotions_substrate_bridge.c
 * @brief Shadow Emotions-Neural Substrate Bridge Implementation
 */

#include "cognitive/shadow_emotions/nimcp_shadow_emotions_substrate_bridge.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "cognitive/common/nimcp_metabolic_modulation.h"
#include "async/nimcp_bio_messages.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>

struct shadow_emotions_substrate_bridge {
    void* shadow_emotions;
    neural_substrate_t* substrate;
    shadow_emotions_substrate_config_t config;
    shadow_emotions_substrate_effects_t effects;
    bio_module_context_t ctx;
    bio_router_t* router;
    bool bio_async_connected;
    uint64_t update_count;
    float prev_overall_capacity;
};

shadow_emotions_substrate_config_t shadow_emotions_substrate_default_config(void) {
    shadow_emotions_substrate_config_t cfg = { .enable_atp_modulation = true, .enable_fatigue_modulation = true,
        .enable_bio_async = false, .atp_sensitivity = 1.0f, .fatigue_sensitivity = 1.0f, .min_capacity = 0.2f };
    return cfg;
}

shadow_emotions_substrate_bridge_t* shadow_emotions_substrate_bridge_create(void* shadow_emotions, neural_substrate_t* substrate, const shadow_emotions_substrate_config_t* config) {
    if (!substrate) return NULL;
    shadow_emotions_substrate_bridge_t* bridge = nimcp_calloc(1, sizeof(shadow_emotions_substrate_bridge_t));
    if (!bridge) return NULL;
    bridge->shadow_emotions = shadow_emotions;
    bridge->substrate = substrate;
    bridge->config = config ? *config : shadow_emotions_substrate_default_config();
    bridge->effects.suppression_strength = 1.0f;
    bridge->effects.emergence_threshold = 0.5f;
    bridge->effects.regulation_capacity = 1.0f;
    bridge->effects.integration_ability = 1.0f;
    bridge->effects.overall_capacity = 1.0f;
    return bridge;
}

void shadow_emotions_substrate_bridge_destroy(shadow_emotions_substrate_bridge_t* bridge) {
    if (!bridge) return;
    if (bridge->bio_async_connected && bridge->ctx) {
        bio_router_unregister_module(bridge->ctx);
    }
    nimcp_free(bridge);
}

int shadow_emotions_substrate_bridge_update(shadow_emotions_substrate_bridge_t* bridge) {
    if (!bridge || !bridge->substrate) return -1;
    substrate_metabolic_state_t metabolic;
    if (substrate_get_metabolic_state(bridge->substrate, &metabolic) != 0) return -1;
    float atp = metabolic.atp_level, fatigue = 1.0f - metabolic.metabolic_capacity, min_cap = bridge->config.min_capacity;
    /* ATP enables suppression and regulation - low ATP allows emergence */
    if (bridge->config.enable_atp_modulation) {
        bridge->effects.suppression_strength = nimcp_clamp_f(atp * bridge->config.atp_sensitivity, min_cap, 1.0f);
        bridge->effects.regulation_capacity = nimcp_clamp_f(atp * 1.1f * bridge->config.atp_sensitivity, min_cap, 1.0f);
    }
    /* Fatigue lowers emergence threshold and reduces integration */
    if (bridge->config.enable_fatigue_modulation) {
        bridge->effects.emergence_threshold = nimcp_clamp_f(0.5f - fatigue * 0.3f * bridge->config.fatigue_sensitivity, 0.2f, 0.8f);
        bridge->effects.integration_ability = nimcp_clamp_f((1.0f - fatigue) * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
    }
    bridge->effects.overall_capacity = (bridge->effects.suppression_strength + bridge->effects.regulation_capacity +
                                        bridge->effects.integration_ability) / 3.0f;
    bridge->update_count++;
    return 0;
}

int shadow_emotions_substrate_bridge_get_effects(const shadow_emotions_substrate_bridge_t* bridge, shadow_emotions_substrate_effects_t* effects) {
    if (!bridge || !effects) return -1;
    *effects = bridge->effects;
    return 0;
}

int shadow_emotions_substrate_bridge_apply_effects(shadow_emotions_substrate_bridge_t* bridge) {
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
    bio_msg_init_header(&msg.header, BIO_MSG_SUBSTRATE_MODULATION, BIO_MODULE_SUBSTRATE_SHADOW_EMOTIONS, 0, sizeof(msg));
    msg.header.channel = BIO_CHANNEL_SEROTONIN;
    msg.bridge_module_id = BIO_MODULE_SUBSTRATE_SHADOW_EMOTIONS;
    msg.processing_capacity = bridge->effects.suppression_strength;
    msg.overall_capacity = bridge->effects.overall_capacity;
    msg.effect_values[0] = bridge->effects.suppression_strength;
    msg.effect_values[1] = bridge->effects.emergence_threshold;
    msg.effect_values[2] = bridge->effects.regulation_capacity;
    msg.effect_values[3] = bridge->effects.integration_ability;
    msg.atp_level = atp_level;
    msg.fatigue_level = fatigue_level;
    msg.update_count = bridge->update_count;
    msg.critical_low = (atp_level < bridge->config.min_capacity);
    bio_router_broadcast(bridge->ctx, &msg, sizeof(msg));

    float delta = bridge->effects.overall_capacity - bridge->prev_overall_capacity;
    if (delta < -0.1f || delta > 0.1f) {
        bio_msg_substrate_capacity_update_t update_msg;
        memset(&update_msg, 0, sizeof(update_msg));
        bio_msg_init_header(&update_msg.header, BIO_MSG_SUBSTRATE_CAPACITY_UPDATE, BIO_MODULE_SUBSTRATE_SHADOW_EMOTIONS, 0, sizeof(update_msg));
        update_msg.bridge_module_id = BIO_MODULE_SUBSTRATE_SHADOW_EMOTIONS;
        update_msg.old_capacity = bridge->prev_overall_capacity;
        update_msg.new_capacity = bridge->effects.overall_capacity;
        update_msg.delta = delta;
        update_msg.significant_change = true;
        bio_router_broadcast(bridge->ctx, &update_msg, sizeof(update_msg));
    }
    bridge->prev_overall_capacity = bridge->effects.overall_capacity;
    return 0;
}

int shadow_emotions_substrate_bridge_register_bio_async(shadow_emotions_substrate_bridge_t* bridge, bio_router_t* router) {
    if (!bridge) return -1;
    if (bridge->bio_async_connected && bridge->ctx) {
        bio_router_unregister_module(bridge->ctx);
        bridge->ctx = NULL;
        bridge->bio_async_connected = false;
    }
    if (!router) return 0;
    bio_module_info_t info = { .module_id = BIO_MODULE_SUBSTRATE_SHADOW_EMOTIONS, .module_name = "shadow_emotions_substrate_bridge", .inbox_capacity = 16, .user_data = bridge };
    bridge->ctx = bio_router_register_module(&info);
    if (bridge->ctx) bridge->bio_async_connected = true;
    return 0;
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

int shadow_emotions_substrate_bridge_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;

    const kg_entity_t* self = kg_reader_get_entity(kg, "Shadow_Emotions_Substrate_Bridge");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            (void)self->observations[i];
        }
    }

    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Shadow_Emotions_Substrate_Bridge");
    if (connections) {
        kg_relation_list_destroy(connections);
    }

    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Shadow_Emotions_Substrate_Bridge");
    if (incoming) {
        kg_relation_list_destroy(incoming);
    }

    return self ? 1 : 0;
}
