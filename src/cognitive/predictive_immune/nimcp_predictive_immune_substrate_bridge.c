/**
 * @file nimcp_predictive_immune_substrate_bridge.c
 * @brief Predictive Immune-Neural Substrate Bridge Implementation
 *
 * WHAT: Links predictive-immune integration to metabolic state
 * WHY: Immune-cognitive integration requires sustained prefrontal and insular resources
 * HOW: Monitors ATP/fatigue; modulates prediction accuracy, immune precision, cytokine sensitivity
 */

#include "cognitive/predictive_immune/nimcp_predictive_immune_substrate_bridge.h"
#include "cognitive/common/nimcp_metabolic_modulation.h"
#include "async/nimcp_bio_messages.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/platform/nimcp_platform_mutex.h"
#include <string.h>
#include <math.h>

struct predictive_immune_substrate_bridge {
    bridge_base_t base;                 /* MUST be first: base bridge infrastructure */
    void* predictive_immune;
    neural_substrate_t* substrate;
    predictive_immune_substrate_config_t config;
    predictive_immune_substrate_effects_t effects;
    bio_module_context_t ctx;
    bio_router_t* router;
    bool bio_async_connected;
    uint64_t update_count;
    float prev_overall_capacity;
};

predictive_immune_substrate_config_t predictive_immune_substrate_default_config(void) {
    predictive_immune_substrate_config_t cfg = {
        .enable_atp_modulation = true,
        .enable_fatigue_modulation = true,
        .enable_bio_async = false,
        .atp_sensitivity = 1.0f,
        .fatigue_sensitivity = 1.0f,
        .min_capacity = 0.2f
    };
    return cfg;
}

predictive_immune_substrate_bridge_t* predictive_immune_substrate_bridge_create(void* predictive_immune, neural_substrate_t* substrate, const predictive_immune_substrate_config_t* config) {
    if (!substrate) return NULL;

    predictive_immune_substrate_bridge_t* bridge = nimcp_calloc(1, sizeof(predictive_immune_substrate_bridge_t));
    if (!bridge) return NULL;

    bridge->predictive_immune = predictive_immune;
    bridge->substrate = substrate;
    bridge->config = config ? *config : predictive_immune_substrate_default_config();

    /* Initialize mutex */
    bridge->base.mutex = (nimcp_mutex_t*)nimcp_malloc(sizeof(nimcp_mutex_t));
    if (!bridge->base.mutex) {
        NIMCP_LOGGING_ERROR("Failed to allocate mutex for predictive immune substrate bridge");
        nimcp_free(bridge);
        return NULL;
    }

    if (nimcp_platform_mutex_init(bridge->base.mutex, false) != 0) {
        NIMCP_LOGGING_ERROR("Failed to initialize mutex for predictive immune substrate bridge");
        nimcp_free(bridge->base.mutex);
        nimcp_free(bridge);
        return NULL;
    }

    bridge->effects.prediction_accuracy = 1.0f;
    bridge->effects.immune_precision = 1.0f;
    bridge->effects.cytokine_sensitivity = 1.0f;
    bridge->effects.integration_capacity = 1.0f;
    bridge->effects.overall_capacity = 1.0f;

    return bridge;
}

void predictive_immune_substrate_bridge_destroy(predictive_immune_substrate_bridge_t* bridge) {
    if (!bridge) return;

    /* Destroy mutex */
    if (bridge->base.mutex) {
        nimcp_platform_mutex_destroy(bridge->base.mutex);
        nimcp_free(bridge->base.mutex);
    }

    nimcp_free(bridge);
}

int predictive_immune_substrate_bridge_update(predictive_immune_substrate_bridge_t* bridge) {
    if (!bridge || !bridge->substrate) return -1;

    nimcp_platform_mutex_lock(bridge->base.mutex);

    substrate_metabolic_state_t metabolic;
    if (substrate_get_metabolic_state(bridge->substrate, &metabolic) != 0) {
        nimcp_platform_mutex_unlock(bridge->base.mutex);
        return -1;
    }

    float atp = metabolic.atp_level;
    float metabolic_cap = metabolic.metabolic_capacity;
    float min_cap = bridge->config.min_capacity;

    if (bridge->config.enable_atp_modulation) {
        /* Prediction accuracy requires stable prefrontal resources */
        bridge->effects.prediction_accuracy = nimcp_clamp_f(atp * bridge->config.atp_sensitivity, min_cap, 1.0f);
        /* Immune precision is ATP-dependent */
        bridge->effects.immune_precision = nimcp_clamp_f(atp * 0.95f * bridge->config.atp_sensitivity, min_cap, 1.0f);
    }

    if (bridge->config.enable_fatigue_modulation) {
        /* Cytokine sensitivity decreases with fatigue */
        bridge->effects.cytokine_sensitivity = nimcp_clamp_f(metabolic_cap * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
        /* Integration capacity is vulnerable to metabolic stress */
        bridge->effects.integration_capacity = nimcp_clamp_f(metabolic_cap * 0.85f * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
    }

    bridge->effects.overall_capacity = (bridge->effects.prediction_accuracy +
                                        bridge->effects.immune_precision +
                                        bridge->effects.cytokine_sensitivity +
                                        bridge->effects.integration_capacity) / 4.0f;

    bridge->update_count++;

    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return 0;
}

int predictive_immune_substrate_bridge_get_effects(const predictive_immune_substrate_bridge_t* bridge, predictive_immune_substrate_effects_t* effects) {
    if (!bridge || !effects) return -1;
    *effects = bridge->effects;
    return 0;
}

int predictive_immune_substrate_bridge_apply_effects(predictive_immune_substrate_bridge_t* bridge) {
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
                        BIO_MODULE_SUBSTRATE_PREDICTIVE_IMMUNE, 0, sizeof(msg));
    msg.header.channel = BIO_CHANNEL_SEROTONIN;  /* Immune uses serotonin (state coordination) */

    msg.bridge_module_id = BIO_MODULE_SUBSTRATE_PREDICTIVE_IMMUNE;
    msg.processing_capacity = bridge->effects.prediction_accuracy;
    msg.overall_capacity = bridge->effects.overall_capacity;
    msg.effect_values[0] = bridge->effects.prediction_accuracy;
    msg.effect_values[1] = bridge->effects.immune_precision;
    msg.effect_values[2] = bridge->effects.cytokine_sensitivity;
    msg.effect_values[3] = bridge->effects.integration_capacity;
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
                            BIO_MODULE_SUBSTRATE_PREDICTIVE_IMMUNE, 0, sizeof(update_msg));
        update_msg.bridge_module_id = BIO_MODULE_SUBSTRATE_PREDICTIVE_IMMUNE;
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
                            BIO_MODULE_SUBSTRATE_PREDICTIVE_IMMUNE, 0, sizeof(alert));
        alert.header.channel = BIO_CHANNEL_NOREPINEPHRINE;
        alert.bridge_module_id = BIO_MODULE_SUBSTRATE_PREDICTIVE_IMMUNE;
        alert.atp_level = atp_level;
        alert.threshold = bridge->config.min_capacity;
        alert.min_capacity = bridge->config.min_capacity;
        bio_router_broadcast(bridge->ctx, &alert, sizeof(alert));
    }

    bridge->prev_overall_capacity = bridge->effects.overall_capacity;
    return 0;
}

int predictive_immune_substrate_bridge_register_bio_async(predictive_immune_substrate_bridge_t* bridge, bio_router_t* router) {
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
        .module_id = BIO_MODULE_SUBSTRATE_PREDICTIVE_IMMUNE,
        .module_name = "predictive_immune_substrate_bridge",
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
