/**
 * @file nimcp_fractal_cognitive_substrate_bridge.c
 * @brief Fractal Cognitive-Neural Substrate Bridge Implementation
 *
 * WHAT: Links fractal cognitive processing to metabolic state
 * WHY: Scale-free computations require sustained energy
 * HOW: Monitors ATP/fatigue; modulates depth, self-similarity, complexity
 */

#include "cognitive/fractal_cognitive/nimcp_fractal_cognitive_substrate_bridge.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "cognitive/common/nimcp_metabolic_modulation.h"
#include "async/nimcp_bio_messages.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/platform/nimcp_platform_mutex.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>
#include <math.h>

struct fractal_cognitive_substrate_bridge {
    bridge_base_t base;                 /* MUST be first: base bridge infrastructure */
    void* fractal_cognitive;
    neural_substrate_t* substrate;
    fractal_cognitive_substrate_config_t config;
    fractal_cognitive_substrate_effects_t effects;
    bio_module_context_t ctx;
    bio_router_t* router;
    bool bio_async_connected;
    uint64_t update_count;
    float prev_overall_capacity;
};

fractal_cognitive_substrate_config_t fractal_cognitive_substrate_default_config(void) {
    fractal_cognitive_substrate_config_t cfg = {
        .enable_atp_modulation = true,
        .enable_fatigue_modulation = true,
        .enable_bio_async = false,
        .atp_sensitivity = 1.0f,
        .fatigue_sensitivity = 1.0f,
        .min_capacity = 0.2f
    };
    return cfg;
}

fractal_cognitive_substrate_bridge_t* fractal_cognitive_substrate_bridge_create(void* fractal_cognitive, neural_substrate_t* substrate, const fractal_cognitive_substrate_config_t* config) {
    if (!substrate) return NULL;

    fractal_cognitive_substrate_bridge_t* bridge = nimcp_calloc(1, sizeof(fractal_cognitive_substrate_bridge_t));
    if (!bridge) return NULL;

    bridge->fractal_cognitive = fractal_cognitive;
    bridge->substrate = substrate;
    bridge->config = config ? *config : fractal_cognitive_substrate_default_config();

    /* Initialize mutex */
    bridge->base.mutex = (nimcp_mutex_t*)nimcp_malloc(sizeof(nimcp_mutex_t));
    if (!bridge->base.mutex) {
        NIMCP_LOGGING_ERROR("Failed to allocate mutex for fractal cognitive substrate bridge");
        nimcp_free(bridge);
        return NULL;
    }

    if (nimcp_platform_mutex_init(bridge->base.mutex, false) != 0) {
        NIMCP_LOGGING_ERROR("Failed to initialize mutex for fractal cognitive substrate bridge");
        nimcp_free(bridge);
        return NULL;
    }

    bridge->effects.fractal_depth = 1.0f;
    bridge->effects.self_similarity = 1.0f;
    bridge->effects.scale_invariance = 1.0f;
    bridge->effects.complexity_capacity = 1.0f;
    bridge->effects.overall_capacity = 1.0f;

    return bridge;
}

void fractal_cognitive_substrate_bridge_destroy(fractal_cognitive_substrate_bridge_t* bridge) {
    if (!bridge) return;

    /* Destroy mutex */
    if (bridge->base.mutex) {
        nimcp_platform_mutex_destroy(bridge->base.mutex);
    }

    nimcp_free(bridge);
}

int fractal_cognitive_substrate_bridge_update(fractal_cognitive_substrate_bridge_t* bridge) {
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
        /* Fractal depth is exponentially sensitive to ATP (each level compounds) */
        bridge->effects.fractal_depth = nimcp_clamp_f(powf(atp, 0.7f) * bridge->config.atp_sensitivity, min_cap, 1.0f);
        /* Scale invariance requires consistent ATP across scales */
        bridge->effects.scale_invariance = nimcp_clamp_f(atp * bridge->config.atp_sensitivity, min_cap, 1.0f);
    }

    if (bridge->config.enable_fatigue_modulation) {
        /* Self-similarity degrades with fatigue */
        bridge->effects.self_similarity = nimcp_clamp_f(metabolic_cap * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
        /* Complexity generation requires sustained metabolic resources */
        bridge->effects.complexity_capacity = nimcp_clamp_f(metabolic_cap * 0.9f * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
    }

    bridge->effects.overall_capacity = (bridge->effects.fractal_depth +
                                        bridge->effects.self_similarity +
                                        bridge->effects.scale_invariance +
                                        bridge->effects.complexity_capacity) / 4.0f;

    bridge->update_count++;

    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return 0;
}

int fractal_cognitive_substrate_bridge_get_effects(const fractal_cognitive_substrate_bridge_t* bridge, fractal_cognitive_substrate_effects_t* effects) {
    if (!bridge || !effects) return -1;
    *effects = bridge->effects;
    return 0;
}

int fractal_cognitive_substrate_bridge_apply_effects(fractal_cognitive_substrate_bridge_t* bridge) {
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
                        BIO_MODULE_SUBSTRATE_FRACTAL_COGNITIVE, 0, sizeof(msg));
    msg.header.channel = BIO_CHANNEL_ACETYLCHOLINE;  /* Fractal uses acetylcholine for attention */

    msg.bridge_module_id = BIO_MODULE_SUBSTRATE_FRACTAL_COGNITIVE;
    msg.processing_capacity = bridge->effects.fractal_depth;
    msg.overall_capacity = bridge->effects.overall_capacity;
    msg.effect_values[0] = bridge->effects.fractal_depth;
    msg.effect_values[1] = bridge->effects.self_similarity;
    msg.effect_values[2] = bridge->effects.scale_invariance;
    msg.effect_values[3] = bridge->effects.complexity_capacity;
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
                            BIO_MODULE_SUBSTRATE_FRACTAL_COGNITIVE, 0, sizeof(update_msg));
        update_msg.bridge_module_id = BIO_MODULE_SUBSTRATE_FRACTAL_COGNITIVE;
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
                            BIO_MODULE_SUBSTRATE_FRACTAL_COGNITIVE, 0, sizeof(alert));
        alert.header.channel = BIO_CHANNEL_NOREPINEPHRINE;
        alert.bridge_module_id = BIO_MODULE_SUBSTRATE_FRACTAL_COGNITIVE;
        alert.atp_level = atp_level;
        alert.threshold = bridge->config.min_capacity;
        alert.min_capacity = bridge->config.min_capacity;
        bio_router_broadcast(bridge->ctx, &alert, sizeof(alert));
    }

    bridge->prev_overall_capacity = bridge->effects.overall_capacity;
    return 0;
}

int fractal_cognitive_substrate_bridge_register_bio_async(fractal_cognitive_substrate_bridge_t* bridge, bio_router_t* router) {
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
        .module_id = BIO_MODULE_SUBSTRATE_FRACTAL_COGNITIVE,
        .module_name = "fractal_cognitive_substrate_bridge",
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

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

int fractal_cognitive_substrate_bridge_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;

    const kg_entity_t* self = kg_reader_get_entity(kg, "Fractal_Cognitive_Substrate_Bridge");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            (void)self->observations[i];
        }
    }

    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Fractal_Cognitive_Substrate_Bridge");
    if (connections) {
        kg_relation_list_destroy(connections);
    }

    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Fractal_Cognitive_Substrate_Bridge");
    if (incoming) {
        kg_relation_list_destroy(incoming);
    }

    return self ? 1 : 0;
}
