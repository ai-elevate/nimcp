/**
 * @file nimcp_sleep_wake_substrate_bridge.c
 * @brief Sleep-Wake-Neural Substrate Bridge Implementation
 */

#include "cognitive/sleep_wake/nimcp_sleep_wake_substrate_bridge.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "cognitive/common/nimcp_metabolic_modulation.h"
#include "async/nimcp_bio_messages.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>

struct sleep_wake_substrate_bridge {
    void* sleep_wake;
    neural_substrate_t* substrate;
    sleep_wake_substrate_config_t config;
    sleep_wake_substrate_effects_t effects;
    bio_module_context_t ctx;
    bio_router_t* router;
    bool bio_async_connected;
    uint64_t update_count;
    float prev_overall_capacity;
};

sleep_wake_substrate_config_t sleep_wake_substrate_default_config(void) {
    sleep_wake_substrate_config_t cfg = { .enable_atp_modulation = true, .enable_fatigue_modulation = true,
        .enable_bio_async = false, .atp_sensitivity = 1.0f, .fatigue_sensitivity = 1.0f, .min_capacity = 0.2f };
    return cfg;
}

sleep_wake_substrate_bridge_t* sleep_wake_substrate_bridge_create(void* sleep_wake, neural_substrate_t* substrate, const sleep_wake_substrate_config_t* config) {
    if (!substrate) return NULL;
    sleep_wake_substrate_bridge_t* bridge = nimcp_calloc(1, sizeof(sleep_wake_substrate_bridge_t));
    if (!bridge) return NULL;
    bridge->sleep_wake = sleep_wake;
    bridge->substrate = substrate;
    bridge->config = config ? *config : sleep_wake_substrate_default_config();
    bridge->effects.arousal_level = 1.0f;
    bridge->effects.sleep_pressure = 0.0f;
    bridge->effects.circadian_phase = 1.0f;
    bridge->effects.recovery_rate = 1.0f;
    bridge->effects.overall_capacity = 1.0f;
    return bridge;
}

void sleep_wake_substrate_bridge_destroy(sleep_wake_substrate_bridge_t* bridge) {
    if (!bridge) return;
    if (bridge->bio_async_connected && bridge->ctx) {
        bio_router_unregister_module(bridge->ctx);
    }
    nimcp_free(bridge);
}

int sleep_wake_substrate_bridge_update(sleep_wake_substrate_bridge_t* bridge) {
    if (!bridge || !bridge->substrate) return -1;
    substrate_metabolic_state_t metabolic;
    if (substrate_get_metabolic_state(bridge->substrate, &metabolic) != 0) return -1;
    float atp = metabolic.atp_level, metabolic_cap = metabolic.metabolic_capacity, min_cap = bridge->config.min_capacity;
    /* ATP depletion increases sleep pressure */
    if (bridge->config.enable_atp_modulation) {
        bridge->effects.arousal_level = nimcp_clamp_f(atp * bridge->config.atp_sensitivity, min_cap, 1.0f);
        bridge->effects.recovery_rate = nimcp_clamp_f(atp * 1.1f * bridge->config.atp_sensitivity, min_cap, 1.0f);
    }
    /* Fatigue directly drives sleep pressure */
    if (bridge->config.enable_fatigue_modulation) {
        bridge->effects.sleep_pressure = nimcp_clamp_f((1.0f - metabolic_cap) * bridge->config.fatigue_sensitivity, 0.0f, 1.0f);
        bridge->effects.circadian_phase = nimcp_clamp_f((1.0f - (1.0f - metabolic_cap) * 0.3f), 0.5f, 1.0f);
    }
    bridge->effects.overall_capacity = bridge->effects.arousal_level;
    bridge->update_count++;
    return 0;
}

int sleep_wake_substrate_bridge_get_effects(const sleep_wake_substrate_bridge_t* bridge, sleep_wake_substrate_effects_t* effects) {
    if (!bridge || !effects) return -1;
    *effects = bridge->effects;
    return 0;
}

int sleep_wake_substrate_bridge_apply_effects(sleep_wake_substrate_bridge_t* bridge) {
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
                        BIO_MODULE_SUBSTRATE_SLEEP_WAKE, 0, sizeof(msg));
    msg.header.channel = BIO_CHANNEL_SEROTONIN;  /* Sleep/wake uses serotonin */

    msg.bridge_module_id = BIO_MODULE_SUBSTRATE_SLEEP_WAKE;
    msg.processing_capacity = bridge->effects.arousal_level;
    msg.overall_capacity = bridge->effects.overall_capacity;
    msg.effect_values[0] = bridge->effects.arousal_level;
    msg.effect_values[1] = bridge->effects.sleep_pressure;
    msg.effect_values[2] = bridge->effects.circadian_phase;
    msg.effect_values[3] = bridge->effects.recovery_rate;
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
                            BIO_MODULE_SUBSTRATE_SLEEP_WAKE, 0, sizeof(update_msg));
        update_msg.bridge_module_id = BIO_MODULE_SUBSTRATE_SLEEP_WAKE;
        update_msg.old_capacity = bridge->prev_overall_capacity;
        update_msg.new_capacity = bridge->effects.overall_capacity;
        update_msg.delta = delta;
        update_msg.significant_change = true;
        bio_router_broadcast(bridge->ctx, &update_msg, sizeof(update_msg));
    }

    /* Send fatigue alert for high sleep pressure */
    if (bridge->effects.sleep_pressure > 0.7f) {
        bio_msg_substrate_fatigue_alert_t alert;
        memset(&alert, 0, sizeof(alert));
        bio_msg_init_header(&alert.header, BIO_MSG_SUBSTRATE_FATIGUE_ALERT,
                            BIO_MODULE_SUBSTRATE_SLEEP_WAKE, 0, sizeof(alert));
        alert.header.channel = BIO_CHANNEL_SEROTONIN;
        alert.bridge_module_id = BIO_MODULE_SUBSTRATE_SLEEP_WAKE;
        alert.fatigue_level = fatigue_level;
        alert.threshold = 0.7f;
        alert.capacity_reduction = 1.0f - bridge->effects.overall_capacity;
        bio_router_broadcast(bridge->ctx, &alert, sizeof(alert));
    }

    bridge->prev_overall_capacity = bridge->effects.overall_capacity;
    return 0;
}

int sleep_wake_substrate_bridge_register_bio_async(sleep_wake_substrate_bridge_t* bridge, bio_router_t* router) {
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
        .module_id = BIO_MODULE_SUBSTRATE_SLEEP_WAKE,
        .module_name = "sleep_wake_substrate_bridge",
        .inbox_capacity = 16,
        .user_data = bridge
    };

    bridge->ctx = bio_router_register_module(&info);
    if (bridge->ctx) {
        bridge->bio_async_connected = true;
    }
    return 0;
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

int sleep_wake_substrate_bridge_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;

    const kg_entity_t* self = kg_reader_get_entity(kg, "Sleep_Wake_Substrate_Bridge");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            (void)self->observations[i];
        }
    }

    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Sleep_Wake_Substrate_Bridge");
    if (connections) {
        kg_relation_list_destroy(connections);
    }

    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Sleep_Wake_Substrate_Bridge");
    if (incoming) {
        kg_relation_list_destroy(incoming);
    }

    return self ? 1 : 0;
}
