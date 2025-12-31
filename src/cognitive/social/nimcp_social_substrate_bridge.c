/**
 * @file nimcp_social_substrate_bridge.c
 * @brief Social Cognition-Neural Substrate Bridge Implementation
 *
 * WHAT: Links social cognition to metabolic state
 * WHY: Social processing requires sustained prefrontal and limbic resources
 * HOW: Monitors ATP/fatigue; modulates bonding, loyalty, trust
 */

#include "cognitive/social/nimcp_social_substrate_bridge.h"
#include "cognitive/common/nimcp_metabolic_modulation.h"
#include "async/nimcp_bio_messages.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include <string.h>
#include <math.h>

struct social_substrate_bridge {
    void* social;
    neural_substrate_t* substrate;
    social_substrate_config_t config;
    social_substrate_effects_t effects;
    bio_module_context_t ctx;
    bio_router_t* router;
    bool bio_async_connected;
    uint64_t update_count;
    float prev_overall_capacity;
};

social_substrate_config_t social_substrate_default_config(void) {
    social_substrate_config_t cfg = {
        .enable_atp_modulation = true,
        .enable_fatigue_modulation = true,
        .enable_bio_async = false,
        .atp_sensitivity = 1.0f,
        .fatigue_sensitivity = 1.0f,
        .min_capacity = 0.2f
    };
    return cfg;
}

social_substrate_bridge_t* social_substrate_bridge_create(void* social, neural_substrate_t* substrate, const social_substrate_config_t* config) {
    if (!substrate) return NULL;

    social_substrate_bridge_t* bridge = nimcp_calloc(1, sizeof(social_substrate_bridge_t));
    if (!bridge) return NULL;

    bridge->social = social;
    bridge->substrate = substrate;
    bridge->config = config ? *config : social_substrate_default_config();

    bridge->effects.bonding_capacity = 1.0f;
    bridge->effects.loyalty_strength = 1.0f;
    bridge->effects.trust_evaluation = 1.0f;
    bridge->effects.prosocial_motivation = 1.0f;
    bridge->effects.overall_capacity = 1.0f;

    return bridge;
}

void social_substrate_bridge_destroy(social_substrate_bridge_t* bridge) {
    if (bridge) nimcp_free(bridge);
}

int social_substrate_bridge_update(social_substrate_bridge_t* bridge) {
    if (!bridge || !bridge->substrate) return -1;

    substrate_metabolic_state_t metabolic;
    if (substrate_get_metabolic_state(bridge->substrate, &metabolic) != 0) return -1;

    float atp = metabolic.atp_level;
    float metabolic_cap = metabolic.metabolic_capacity;
    float min_cap = bridge->config.min_capacity;

    if (bridge->config.enable_atp_modulation) {
        /* Bonding capacity requires sustained neural resources */
        bridge->effects.bonding_capacity = nimcp_clamp_f(atp * bridge->config.atp_sensitivity, min_cap, 1.0f);
        /* Trust evaluation is cognitively demanding */
        bridge->effects.trust_evaluation = nimcp_clamp_f(atp * 0.95f * bridge->config.atp_sensitivity, min_cap, 1.0f);
    }

    if (bridge->config.enable_fatigue_modulation) {
        /* Loyalty strength maintained but may waver under fatigue */
        bridge->effects.loyalty_strength = nimcp_clamp_f(metabolic_cap * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
        /* Prosocial motivation decreases with fatigue */
        bridge->effects.prosocial_motivation = nimcp_clamp_f(metabolic_cap * 0.85f * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
    }

    bridge->effects.overall_capacity = (bridge->effects.bonding_capacity +
                                        bridge->effects.loyalty_strength +
                                        bridge->effects.trust_evaluation +
                                        bridge->effects.prosocial_motivation) / 4.0f;

    bridge->update_count++;
    return 0;
}

int social_substrate_bridge_get_effects(const social_substrate_bridge_t* bridge, social_substrate_effects_t* effects) {
    if (!bridge || !effects) return -1;
    *effects = bridge->effects;
    return 0;
}

int social_substrate_bridge_apply_effects(social_substrate_bridge_t* bridge) {
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
                        BIO_MODULE_SUBSTRATE_SOCIAL, 0, sizeof(msg));
    msg.header.channel = BIO_CHANNEL_SEROTONIN;  /* Social cognition uses serotonin */

    msg.bridge_module_id = BIO_MODULE_SUBSTRATE_SOCIAL;
    msg.processing_capacity = bridge->effects.bonding_capacity;
    msg.overall_capacity = bridge->effects.overall_capacity;
    msg.effect_values[0] = bridge->effects.bonding_capacity;
    msg.effect_values[1] = bridge->effects.loyalty_strength;
    msg.effect_values[2] = bridge->effects.trust_evaluation;
    msg.effect_values[3] = bridge->effects.prosocial_motivation;
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
                            BIO_MODULE_SUBSTRATE_SOCIAL, 0, sizeof(update_msg));
        update_msg.bridge_module_id = BIO_MODULE_SUBSTRATE_SOCIAL;
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
                            BIO_MODULE_SUBSTRATE_SOCIAL, 0, sizeof(alert));
        alert.header.channel = BIO_CHANNEL_NOREPINEPHRINE;
        alert.bridge_module_id = BIO_MODULE_SUBSTRATE_SOCIAL;
        alert.atp_level = atp_level;
        alert.threshold = bridge->config.min_capacity;
        alert.min_capacity = bridge->config.min_capacity;
        bio_router_broadcast(bridge->ctx, &alert, sizeof(alert));
    }

    bridge->prev_overall_capacity = bridge->effects.overall_capacity;
    return 0;
}

int social_substrate_bridge_register_bio_async(social_substrate_bridge_t* bridge, bio_router_t* router) {
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
        .module_id = BIO_MODULE_SUBSTRATE_SOCIAL,
        .module_name = "social_substrate_bridge",
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
