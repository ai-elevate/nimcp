/**
 * @file nimcp_grief_substrate_bridge.c
 * @brief Grief-Neural Substrate Bridge Implementation
 *
 * Uses shared metabolic modulation utilities from nimcp_metabolic_modulation.h
 */

#include "cognitive/grief/nimcp_grief_substrate_bridge.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "cognitive/common/nimcp_metabolic_modulation.h"
#include "async/nimcp_bio_messages.h"
#include "utils/memory/nimcp_memory.h"
#include <string.h>

struct grief_substrate_bridge {
    void* grief;
    neural_substrate_t* substrate;
    grief_substrate_config_t config;
    grief_substrate_effects_t effects;
    metabolic_modulation_config_t metabolic_config;  /* Shared metabolic config */
    bio_module_context_t ctx;
    bool bio_async_connected;
    uint64_t update_count;
    float prev_overall_capacity;
};

grief_substrate_config_t grief_substrate_default_config(void) {
    grief_substrate_config_t cfg = { .enable_atp_modulation = true, .enable_fatigue_modulation = true,
        .enable_bio_async = false, .atp_sensitivity = 1.0f, .fatigue_sensitivity = 1.0f, .min_capacity = 0.2f };
    return cfg;
}

grief_substrate_bridge_t* grief_substrate_bridge_create(void* grief, neural_substrate_t* substrate, const grief_substrate_config_t* config) {
    if (!substrate) return NULL;
    grief_substrate_bridge_t* bridge = nimcp_calloc(1, sizeof(grief_substrate_bridge_t));
    if (!bridge) return NULL;
    bridge->grief = grief;
    bridge->substrate = substrate;
    bridge->config = config ? *config : grief_substrate_default_config();

    /* Initialize shared metabolic config from bridge-specific config */
    bridge->metabolic_config = metabolic_config_from_fields(
        bridge->config.enable_atp_modulation,
        bridge->config.enable_fatigue_modulation,
        bridge->config.enable_bio_async,
        bridge->config.atp_sensitivity,
        bridge->config.fatigue_sensitivity,
        bridge->config.min_capacity,
        NULL  /* Use default multipliers */
    );

    /* Initialize effects to full capacity */
    bridge->effects.processing_capacity = 1.0f;
    bridge->effects.emotion_regulation = 1.0f;
    bridge->effects.adaptation_rate = 1.0f;
    bridge->effects.resilience_level = 1.0f;
    bridge->effects.overall_capacity = 1.0f;
    return bridge;
}

void grief_substrate_bridge_destroy(grief_substrate_bridge_t* bridge) {
    if (!bridge) return;
    if (bridge->bio_async_connected && bridge->ctx) {
        bio_router_unregister_module(bridge->ctx);
    }
    nimcp_free(bridge);
}

int grief_substrate_bridge_update(grief_substrate_bridge_t* bridge) {
    if (!bridge || !bridge->substrate) return -1;

    substrate_metabolic_state_t metabolic;
    if (substrate_get_metabolic_state(bridge->substrate, &metabolic) != 0) return -1;

    /* Use shared metabolic computation */
    metabolic_input_t input = {
        .atp_level = metabolic.atp_level,
        .metabolic_capacity = metabolic.metabolic_capacity
    };
    metabolic_effects_t generic_effects;
    metabolic_effects_init_full(&generic_effects);

    if (metabolic_compute_effects(&input, &bridge->metabolic_config, &generic_effects) == 0) {
        /* Map generic effects to grief-specific effect names */
        bridge->effects.processing_capacity = generic_effects.primary_atp;
        bridge->effects.resilience_level = generic_effects.secondary_atp;
        bridge->effects.emotion_regulation = generic_effects.primary_fatigue;
        bridge->effects.adaptation_rate = generic_effects.secondary_fatigue;
        bridge->effects.overall_capacity = generic_effects.overall_capacity;
    }

    bridge->update_count++;
    return 0;
}

int grief_substrate_bridge_get_effects(const grief_substrate_bridge_t* bridge, grief_substrate_effects_t* effects) {
    if (!bridge || !effects) return -1;
    *effects = bridge->effects;
    return 0;
}

int grief_substrate_bridge_apply_effects(grief_substrate_bridge_t* bridge) {
    if (!bridge) return -1;

    /* If bio-async not connected, effects are still calculated but not broadcast */
    if (!bridge->bio_async_connected || !bridge->ctx) {
        return 0;
    }

    /* Get current metabolic state for the message */
    substrate_metabolic_state_t metabolic;
    float atp_level = 1.0f, fatigue_level = 0.0f;
    if (bridge->substrate && substrate_get_metabolic_state(bridge->substrate, &metabolic) == 0) {
        atp_level = metabolic.atp_level;
        fatigue_level = 1.0f - metabolic.metabolic_capacity;
    }

    /* Prepare substrate modulation message */
    bio_msg_substrate_modulation_t msg;
    memset(&msg, 0, sizeof(msg));
    bio_msg_init_header(&msg.header, BIO_MSG_SUBSTRATE_MODULATION,
                        BIO_MODULE_SUBSTRATE_GRIEF, 0, sizeof(msg));
    msg.header.channel = BIO_CHANNEL_SEROTONIN;  /* Grief uses serotonin channel */

    msg.bridge_module_id = BIO_MODULE_SUBSTRATE_GRIEF;
    msg.processing_capacity = bridge->effects.processing_capacity;
    msg.overall_capacity = bridge->effects.overall_capacity;
    msg.effect_values[0] = bridge->effects.processing_capacity;
    msg.effect_values[1] = bridge->effects.emotion_regulation;
    msg.effect_values[2] = bridge->effects.adaptation_rate;
    msg.effect_values[3] = bridge->effects.resilience_level;
    msg.atp_level = atp_level;
    msg.fatigue_level = fatigue_level;
    msg.update_count = bridge->update_count;
    msg.critical_low = (atp_level < bridge->config.min_capacity);

    /* Broadcast effects to interested modules */
    bio_router_broadcast(bridge->ctx, &msg, sizeof(msg));

    /* Check for significant capacity change and send update notification */
    float delta = bridge->effects.overall_capacity - bridge->prev_overall_capacity;
    if (delta < -0.1f || delta > 0.1f) {
        bio_msg_substrate_capacity_update_t update_msg;
        memset(&update_msg, 0, sizeof(update_msg));
        bio_msg_init_header(&update_msg.header, BIO_MSG_SUBSTRATE_CAPACITY_UPDATE,
                            BIO_MODULE_SUBSTRATE_GRIEF, 0, sizeof(update_msg));
        update_msg.bridge_module_id = BIO_MODULE_SUBSTRATE_GRIEF;
        update_msg.old_capacity = bridge->prev_overall_capacity;
        update_msg.new_capacity = bridge->effects.overall_capacity;
        update_msg.delta = delta;
        update_msg.significant_change = true;
        bio_router_broadcast(bridge->ctx, &update_msg, sizeof(update_msg));
    }

    /* Send critical ATP alert if needed */
    if (msg.critical_low) {
        bio_msg_substrate_atp_critical_t alert;
        memset(&alert, 0, sizeof(alert));
        bio_msg_init_header(&alert.header, BIO_MSG_SUBSTRATE_ATP_CRITICAL,
                            BIO_MODULE_SUBSTRATE_GRIEF, 0, sizeof(alert));
        alert.header.channel = BIO_CHANNEL_NOREPINEPHRINE;  /* Alerts on norepinephrine */
        alert.bridge_module_id = BIO_MODULE_SUBSTRATE_GRIEF;
        alert.atp_level = atp_level;
        alert.threshold = bridge->config.min_capacity;
        alert.min_capacity = bridge->config.min_capacity;
        bio_router_broadcast(bridge->ctx, &alert, sizeof(alert));
    }

    bridge->prev_overall_capacity = bridge->effects.overall_capacity;
    return 0;
}

int grief_substrate_bridge_register_bio_async(grief_substrate_bridge_t* bridge, bio_router_t* router) {
    if (!bridge) return -1;

    /* If already connected, disconnect first */
    if (bridge->bio_async_connected && bridge->ctx) {
        bio_router_unregister_module(bridge->ctx);
        bridge->ctx = NULL;
        bridge->bio_async_connected = false;
    }

    if (!router) {
        /* Passing NULL disconnects */
        return 0;
    }

    /* Register with the bio-async router */
    bio_module_info_t info = {
        .module_id = BIO_MODULE_SUBSTRATE_GRIEF,
        .module_name = "grief_substrate_bridge",
        .inbox_capacity = 16,
        .user_data = bridge
    };

    bridge->ctx = bio_router_register_module(&info);
    if (bridge->ctx) {
        bridge->bio_async_connected = true;
        return 0;
    }

    /* Router may not be initialized, that's okay */
    return 0;
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

int grief_substrate_bridge_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;

    const kg_entity_t* self = kg_reader_get_entity(kg, "Grief_Substrate_Bridge");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            (void)self->observations[i];
        }
    }

    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Grief_Substrate_Bridge");
    if (connections) {
        kg_relation_list_destroy(connections);
    }

    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Grief_Substrate_Bridge");
    if (incoming) {
        kg_relation_list_destroy(incoming);
    }

    return self ? 1 : 0;
}
