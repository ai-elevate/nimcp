/**
 * @file nimcp_free_energy_substrate_bridge.c
 * @brief Free Energy Principle-Neural Substrate Bridge Implementation
 *
 * WHAT: Links FEP to metabolic state
 * WHY: Variational inference requires sustained computational resources
 * HOW: Monitors ATP/fatigue; modulates precision, depth, active inference
 */

#include "cognitive/free_energy/nimcp_free_energy_substrate_bridge.h"
#include "cognitive/common/nimcp_metabolic_modulation.h"
#include "async/nimcp_bio_messages.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include <string.h>
#include <math.h>

struct free_energy_substrate_bridge {
    void* free_energy;
    neural_substrate_t* substrate;
    free_energy_substrate_config_t config;
    free_energy_substrate_effects_t effects;
    bio_module_context_t ctx;
    bio_router_t* router;
    bool bio_async_connected;
    uint64_t update_count;
    float prev_overall_capacity;
};

free_energy_substrate_config_t free_energy_substrate_default_config(void) {
    free_energy_substrate_config_t cfg = {
        .enable_atp_modulation = true,
        .enable_fatigue_modulation = true,
        .enable_bio_async = false,
        .atp_sensitivity = 1.0f,
        .fatigue_sensitivity = 1.0f,
        .min_capacity = 0.2f
    };
    return cfg;
}

free_energy_substrate_bridge_t* free_energy_substrate_bridge_create(void* free_energy, neural_substrate_t* substrate, const free_energy_substrate_config_t* config) {
    if (!substrate) return NULL;

    free_energy_substrate_bridge_t* bridge = nimcp_calloc(1, sizeof(free_energy_substrate_bridge_t));
    if (!bridge) return NULL;

    bridge->free_energy = free_energy;
    bridge->substrate = substrate;
    bridge->config = config ? *config : free_energy_substrate_default_config();

    bridge->effects.precision_weighting = 1.0f;
    bridge->effects.prediction_depth = 1.0f;
    bridge->effects.active_inference = 1.0f;
    bridge->effects.model_complexity = 1.0f;
    bridge->effects.overall_capacity = 1.0f;

    return bridge;
}

void free_energy_substrate_bridge_destroy(free_energy_substrate_bridge_t* bridge) {
    if (bridge) nimcp_free(bridge);
}

int free_energy_substrate_bridge_update(free_energy_substrate_bridge_t* bridge) {
    if (!bridge || !bridge->substrate) return -1;

    substrate_metabolic_state_t metabolic;
    if (substrate_get_metabolic_state(bridge->substrate, &metabolic) != 0) return -1;

    float atp = metabolic.atp_level;
    float metabolic_cap = metabolic.metabolic_capacity;
    float min_cap = bridge->config.min_capacity;

    if (bridge->config.enable_atp_modulation) {
        /* Precision weighting requires stable neural resources */
        bridge->effects.precision_weighting = nimcp_clamp_f(atp * bridge->config.atp_sensitivity, min_cap, 1.0f);
        /* Active inference is computationally demanding */
        bridge->effects.active_inference = nimcp_clamp_f(atp * 0.95f * bridge->config.atp_sensitivity, min_cap, 1.0f);
    }

    if (bridge->config.enable_fatigue_modulation) {
        /* Prediction depth decreases with fatigue */
        bridge->effects.prediction_depth = nimcp_clamp_f(metabolic_cap * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
        /* Model complexity simplifies under metabolic stress */
        bridge->effects.model_complexity = nimcp_clamp_f(metabolic_cap * 0.85f * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
    }

    bridge->effects.overall_capacity = (bridge->effects.precision_weighting +
                                        bridge->effects.prediction_depth +
                                        bridge->effects.active_inference +
                                        bridge->effects.model_complexity) / 4.0f;

    bridge->update_count++;
    return 0;
}

int free_energy_substrate_bridge_get_effects(const free_energy_substrate_bridge_t* bridge, free_energy_substrate_effects_t* effects) {
    if (!bridge || !effects) return -1;
    *effects = bridge->effects;
    return 0;
}

int free_energy_substrate_bridge_apply_effects(free_energy_substrate_bridge_t* bridge) {
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
                        BIO_MODULE_SUBSTRATE_FREE_ENERGY, 0, sizeof(msg));
    msg.header.channel = BIO_CHANNEL_DOPAMINE;  /* FEP uses dopamine for precision weighting */

    msg.bridge_module_id = BIO_MODULE_SUBSTRATE_FREE_ENERGY;
    msg.processing_capacity = bridge->effects.precision_weighting;
    msg.overall_capacity = bridge->effects.overall_capacity;
    msg.effect_values[0] = bridge->effects.precision_weighting;
    msg.effect_values[1] = bridge->effects.prediction_depth;
    msg.effect_values[2] = bridge->effects.active_inference;
    msg.effect_values[3] = bridge->effects.model_complexity;
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
                            BIO_MODULE_SUBSTRATE_FREE_ENERGY, 0, sizeof(update_msg));
        update_msg.bridge_module_id = BIO_MODULE_SUBSTRATE_FREE_ENERGY;
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
                            BIO_MODULE_SUBSTRATE_FREE_ENERGY, 0, sizeof(alert));
        alert.header.channel = BIO_CHANNEL_NOREPINEPHRINE;
        alert.bridge_module_id = BIO_MODULE_SUBSTRATE_FREE_ENERGY;
        alert.atp_level = atp_level;
        alert.threshold = bridge->config.min_capacity;
        alert.min_capacity = bridge->config.min_capacity;
        bio_router_broadcast(bridge->ctx, &alert, sizeof(alert));
    }

    bridge->prev_overall_capacity = bridge->effects.overall_capacity;
    return 0;
}

int free_energy_substrate_bridge_register_bio_async(free_energy_substrate_bridge_t* bridge, bio_router_t* router) {
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
        .module_id = BIO_MODULE_SUBSTRATE_FREE_ENERGY,
        .module_name = "free_energy_substrate_bridge",
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
