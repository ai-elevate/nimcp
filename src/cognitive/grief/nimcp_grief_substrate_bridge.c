/**
 * @file nimcp_grief_substrate_bridge.c
 * @brief Grief-Neural Substrate Bridge Implementation
 */

#include "cognitive/grief/nimcp_grief_substrate_bridge.h"
#include "async/nimcp_bio_messages.h"
#include "utils/memory/nimcp_memory.h"
#include <string.h>

struct grief_substrate_bridge {
    void* grief;
    neural_substrate_t* substrate;
    grief_substrate_config_t config;
    grief_substrate_effects_t effects;
    bio_module_context_t ctx;
    bool bio_async_connected;
    uint64_t update_count;
    float prev_overall_capacity;
};

static float clamp_f(float v, float min, float max) { return v < min ? min : (v > max ? max : v); }

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
    float atp = metabolic.atp_level, metabolic_cap = metabolic.metabolic_capacity, min_cap = bridge->config.min_capacity;
    if (bridge->config.enable_atp_modulation) {
        bridge->effects.processing_capacity = clamp_f(atp * bridge->config.atp_sensitivity, min_cap, 1.0f);
        bridge->effects.resilience_level = clamp_f(atp * 1.1f * bridge->config.atp_sensitivity, min_cap, 1.0f);
    }
    if (bridge->config.enable_fatigue_modulation) {
        bridge->effects.emotion_regulation = clamp_f(metabolic_cap * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
        bridge->effects.adaptation_rate = clamp_f(metabolic_cap * 0.9f * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
    }
    bridge->effects.overall_capacity = (bridge->effects.processing_capacity + bridge->effects.emotion_regulation +
                                        bridge->effects.adaptation_rate + bridge->effects.resilience_level) / 4.0f;
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
