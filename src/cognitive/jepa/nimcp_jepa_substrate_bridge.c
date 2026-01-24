/**
 * @file nimcp_jepa_substrate_bridge.c
 * @brief JEPA (Joint Embedding Predictive Architecture)-Neural Substrate Bridge Implementation
 */

#include "utils/bridge/nimcp_bridge_base.h"
#include "cognitive/jepa/nimcp_jepa_substrate_bridge.h"
#include "cognitive/common/nimcp_metabolic_modulation.h"
#include "async/nimcp_bio_messages.h"
#include "utils/memory/nimcp_memory.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>

struct jepa_substrate_bridge {
    bridge_base_t base;              /**< MUST be first: base bridge infrastructure */
    void* jepa;
    neural_substrate_t* substrate;
    jepa_substrate_config_t config;
    jepa_substrate_effects_t effects;
    bio_module_context_t ctx;
    bio_router_t* router;
    bool bio_async_connected;
    uint64_t update_count;
    float prev_overall_capacity;
};

jepa_substrate_config_t jepa_substrate_default_config(void) {
    jepa_substrate_config_t cfg = { .enable_atp_modulation = true, .enable_fatigue_modulation = true,
        .enable_bio_async = false, .atp_sensitivity = 1.0f, .fatigue_sensitivity = 1.0f, .min_capacity = 0.2f };
    return cfg;
}

jepa_substrate_bridge_t* jepa_substrate_bridge_create(void* jepa, neural_substrate_t* substrate, const jepa_substrate_config_t* config) {
    if (!substrate) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "substrate is NULL");

        return NULL;

    }
    jepa_substrate_bridge_t* bridge = nimcp_calloc(1, sizeof(jepa_substrate_bridge_t));
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return NULL;

    }
    bridge->jepa = jepa;
    bridge->substrate = substrate;
    bridge->config = config ? *config : jepa_substrate_default_config();
    bridge->effects.prediction_horizon = 1.0f;
    bridge->effects.model_precision = 1.0f;
    bridge->effects.embedding_quality = 1.0f;
    bridge->effects.update_rate = 1.0f;
    bridge->effects.overall_capacity = 1.0f;
    return bridge;
}

void jepa_substrate_bridge_destroy(jepa_substrate_bridge_t* bridge) {
    if (!bridge) return;
    if (bridge->bio_async_connected && bridge->ctx) {
        bio_router_unregister_module(bridge->ctx);
    }
    nimcp_free(bridge);
}

int jepa_substrate_bridge_update(jepa_substrate_bridge_t* bridge) {
    NIMCP_CHECK_THROW(bridge && bridge->substrate, -1, "bridge or substrate is NULL");
    substrate_metabolic_state_t metabolic;
    if (substrate_get_metabolic_state(bridge->substrate, &metabolic) != 0) return -1;
    float atp = metabolic.atp_level, metabolic_cap = metabolic.metabolic_capacity, min_cap = bridge->config.min_capacity;
    /* ATP enables longer prediction horizons and better model precision */
    if (bridge->config.enable_atp_modulation) {
        bridge->effects.prediction_horizon = nimcp_clamp_f(atp * bridge->config.atp_sensitivity, min_cap, 1.0f);
        bridge->effects.model_precision = nimcp_clamp_f(atp * 1.05f * bridge->config.atp_sensitivity, min_cap, 1.0f);
    }
    /* Low fatigue enables quality embeddings and update rate */
    if (bridge->config.enable_fatigue_modulation) {
        bridge->effects.embedding_quality = nimcp_clamp_f(metabolic_cap * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
        bridge->effects.update_rate = nimcp_clamp_f(metabolic_cap * 0.9f * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
    }
    bridge->effects.overall_capacity = (bridge->effects.prediction_horizon + bridge->effects.model_precision +
                                        bridge->effects.embedding_quality + bridge->effects.update_rate) / 4.0f;
    bridge->update_count++;
    return 0;
}

int jepa_substrate_bridge_get_effects(const jepa_substrate_bridge_t* bridge, jepa_substrate_effects_t* effects) {
    NIMCP_CHECK_THROW(bridge && effects, -1, "bridge or effects is NULL");
    *effects = bridge->effects;
    return 0;
}

int jepa_substrate_bridge_apply_effects(jepa_substrate_bridge_t* bridge) {
    NIMCP_CHECK_THROW(bridge, -1, "bridge is NULL");
    if (!bridge->bio_async_connected || !bridge->ctx) return 0;

    substrate_metabolic_state_t metabolic;
    float atp_level = 1.0f, fatigue_level = 0.0f;
    if (bridge->substrate && substrate_get_metabolic_state(bridge->substrate, &metabolic) == 0) {
        atp_level = metabolic.atp_level;
        fatigue_level = 1.0f - metabolic.metabolic_capacity;
    }

    bio_msg_substrate_modulation_t msg;
    memset(&msg, 0, sizeof(msg));
    bio_msg_init_header(&msg.header, BIO_MSG_SUBSTRATE_MODULATION, BIO_MODULE_SUBSTRATE_JEPA, 0, sizeof(msg));
    msg.header.channel = BIO_CHANNEL_DOPAMINE;  /* Predictive learning uses dopamine */
    msg.bridge_module_id = BIO_MODULE_SUBSTRATE_JEPA;
    msg.processing_capacity = bridge->effects.prediction_horizon;
    msg.overall_capacity = bridge->effects.overall_capacity;
    msg.effect_values[0] = bridge->effects.prediction_horizon;
    msg.effect_values[1] = bridge->effects.model_precision;
    msg.effect_values[2] = bridge->effects.embedding_quality;
    msg.effect_values[3] = bridge->effects.update_rate;
    msg.atp_level = atp_level;
    msg.fatigue_level = fatigue_level;
    msg.update_count = bridge->update_count;
    msg.critical_low = (atp_level < bridge->config.min_capacity);
    bio_router_broadcast(bridge->ctx, &msg, sizeof(msg));

    float delta = bridge->effects.overall_capacity - bridge->prev_overall_capacity;
    if (delta < -0.1f || delta > 0.1f) {
        bio_msg_substrate_capacity_update_t update_msg;
        memset(&update_msg, 0, sizeof(update_msg));
        bio_msg_init_header(&update_msg.header, BIO_MSG_SUBSTRATE_CAPACITY_UPDATE, BIO_MODULE_SUBSTRATE_JEPA, 0, sizeof(update_msg));
        update_msg.bridge_module_id = BIO_MODULE_SUBSTRATE_JEPA;
        update_msg.old_capacity = bridge->prev_overall_capacity;
        update_msg.new_capacity = bridge->effects.overall_capacity;
        update_msg.delta = delta;
        update_msg.significant_change = true;
        bio_router_broadcast(bridge->ctx, &update_msg, sizeof(update_msg));
    }
    bridge->prev_overall_capacity = bridge->effects.overall_capacity;
    return 0;
}

int jepa_substrate_bridge_register_bio_async(jepa_substrate_bridge_t* bridge, bio_router_t* router) {
    NIMCP_CHECK_THROW(bridge, -1, "bridge is NULL");
    if (bridge->bio_async_connected && bridge->ctx) {
        bio_router_unregister_module(bridge->ctx);
        bridge->ctx = NULL;
        bridge->bio_async_connected = false;
    }
    if (!router) return 0;
    bio_module_info_t info = { .module_id = BIO_MODULE_SUBSTRATE_JEPA, .module_name = "jepa_substrate_bridge", .inbox_capacity = 16, .user_data = bridge };
    bridge->ctx = bio_router_register_module(&info);
    if (bridge->ctx) bridge->bio_async_connected = true;
    return 0;
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

int jepa_substrate_bridge_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;

    const kg_entity_t* self = kg_reader_get_entity(kg, "JEPA_Substrate_Bridge");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            (void)self->observations[i];
        }
    }

    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "JEPA_Substrate_Bridge");
    if (connections) {
        kg_relation_list_destroy(connections);
    }

    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "JEPA_Substrate_Bridge");
    if (incoming) {
        kg_relation_list_destroy(incoming);
    }

    return self ? 1 : 0;
}
