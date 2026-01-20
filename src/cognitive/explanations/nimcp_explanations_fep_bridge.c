/**
 * @file nimcp_explanations_fep_bridge.c
 * @brief Explanations FEP Bridge Implementation
 */

#include "cognitive/explanations/nimcp_explanations_fep_bridge.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/validation/nimcp_common.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include <string.h>

#define LOG_MODULE "explanations_fep_bridge"

int explanations_fep_bridge_default_config(explanations_fep_config_t* config) {
    if (!config) return NIMCP_ERROR_NULL_POINTER;
    config->uncertainty_threshold = EXPLANATIONS_FEP_HIGH_UNCERTAINTY_THRESHOLD;
    config->detail_boost_factor = EXPLANATIONS_FEP_DETAIL_BOOST_FACTOR;
    config->enable_uncertainty_modulation = true;
    config->enable_causal_extraction = true;
    config->enable_counterfactual_testing = true;
    config->enable_model_exposure = true;
    config->fe_sensitivity = 1.0f;
    config->explanation_sensitivity = 1.0f;
    return 0;
}

explanations_fep_bridge_t* explanations_fep_bridge_create(const explanations_fep_config_t* config) {
    explanations_fep_bridge_t* bridge = nimcp_malloc(sizeof(explanations_fep_bridge_t));
    if (!bridge) return NULL;
    memset(bridge, 0, sizeof(explanations_fep_bridge_t));
    if (config) bridge->config = *config;
    else explanations_fep_bridge_default_config(&bridge->config);
    bridge->base.mutex = nimcp_platform_mutex_create();
    if (!bridge->base.mutex) { nimcp_free(bridge); return NULL; }
    return bridge;
}

void explanations_fep_bridge_destroy(explanations_fep_bridge_t* bridge) {
    if (!bridge) return;
    if (bridge->base.bio_async_enabled) explanations_fep_bridge_disconnect_bio_async(bridge);
    if (bridge->base.mutex) {
        nimcp_platform_mutex_destroy(bridge->base.mutex);
    }
    nimcp_free(bridge);
}

int explanations_fep_bridge_connect_fep(explanations_fep_bridge_t* bridge, fep_system_t* fep) {
    if (!bridge || !fep) return NIMCP_ERROR_NULL_POINTER;
    nimcp_mutex_lock(bridge->base.mutex);
    bridge->fep_system = fep;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int explanations_fep_bridge_connect_generator(explanations_fep_bridge_t* bridge, explanation_generator_t gen) {
    if (!bridge || !gen) return NIMCP_ERROR_NULL_POINTER;
    nimcp_mutex_lock(bridge->base.mutex);
    bridge->explanation_gen = gen;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int explanations_fep_bridge_disconnect(explanations_fep_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    nimcp_mutex_lock(bridge->base.mutex);
    bridge->fep_system = NULL;
    bridge->explanation_gen = NULL;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int explanations_fep_modulate_detail(explanations_fep_bridge_t* bridge, float uncertainty) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    if (!bridge->config.enable_uncertainty_modulation) return 0;
    nimcp_mutex_lock(bridge->base.mutex);
    bridge->fep_effects.current_uncertainty = uncertainty;
    if (uncertainty > bridge->config.uncertainty_threshold) {
        bridge->fep_effects.detailed_explanation_needed = true;
        bridge->stats.explanations_generated++;
    }
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int explanations_fep_extract_causal_chain(explanations_fep_bridge_t* bridge) {
    if (!bridge || !bridge->fep_system) return NIMCP_ERROR_NULL_POINTER;
    if (!bridge->config.enable_causal_extraction) return 0;
    nimcp_mutex_lock(bridge->base.mutex);
    bridge->explanation_effects.model_structure_exposed = true;
    bridge->stats.model_exposures++;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int explanations_fep_test_counterfactual(explanations_fep_bridge_t* bridge) {
    if (!bridge || !bridge->fep_system) return NIMCP_ERROR_NULL_POINTER;
    if (!bridge->config.enable_counterfactual_testing) return 0;
    nimcp_mutex_lock(bridge->base.mutex);
    bridge->fep_effects.counterfactual_mode_active = true;
    bridge->stats.counterfactuals_tested++;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int explanations_fep_bridge_update(explanations_fep_bridge_t* bridge, uint64_t delta_ms) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    explanations_fep_extract_causal_chain(bridge);
    return 0;
}

int explanations_fep_bridge_get_state(const explanations_fep_bridge_t* bridge, explanations_fep_state_t* state) {
    if (!bridge || !state) return NIMCP_ERROR_NULL_POINTER;
    nimcp_mutex_lock(bridge->base.mutex);
    *state = bridge->state;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int explanations_fep_bridge_get_stats(const explanations_fep_bridge_t* bridge, explanations_fep_stats_t* stats) {
    if (!bridge || !stats) return NIMCP_ERROR_NULL_POINTER;
    nimcp_mutex_lock(bridge->base.mutex);
    *stats = bridge->stats;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int explanations_fep_bridge_connect_bio_async(explanations_fep_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    if (bridge->base.bio_async_enabled) return 0;
    bio_module_info_t info = {
        .module_id = BIO_MODULE_FEP_EXPLANATIONS_BRIDGE,
        .module_name = "explanations_fep_bridge",
        .inbox_capacity = 32,
        .user_data = bridge
    };
    bridge->base.bio_ctx = bio_router_register_module(&info);
    if (bridge->base.bio_ctx) {
        bridge->base.bio_async_enabled = true;
        NIMCP_LOGGING_INFO("Connected to bio-async router");
    }
    return 0;
}

int explanations_fep_bridge_disconnect_bio_async(explanations_fep_bridge_t* bridge) {
    if (!bridge || !bridge->base.bio_async_enabled) return 0;
    if (bridge->base.bio_ctx) bio_router_unregister_module(bridge->base.bio_ctx);
    bridge->base.bio_ctx = NULL;
    bridge->base.bio_async_enabled = false;
    return 0;
}

bool explanations_fep_bridge_is_bio_async_connected(const explanations_fep_bridge_t* bridge) {
    return bridge ? bridge->base.bio_async_enabled : false;
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

int explanations_fep_bridge_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;

    const kg_entity_t* self = kg_reader_get_entity(kg, "Explanations_FEP_Bridge");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            (void)self->observations[i];
        }
    }

    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Explanations_FEP_Bridge");
    if (connections) {
        kg_relation_list_destroy(connections);
    }

    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Explanations_FEP_Bridge");
    if (incoming) {
        kg_relation_list_destroy(incoming);
    }

    return self ? 1 : 0;
}
