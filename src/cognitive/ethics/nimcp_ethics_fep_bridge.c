/**
 * @file nimcp_ethics_fep_bridge.c
 * @brief Ethics FEP Bridge Implementation
 */

#include "cognitive/ethics/nimcp_ethics_fep_bridge.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/validation/nimcp_common.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include <string.h>
#include <math.h>

#define LOG_MODULE_ETHICS_FEP "[ETHICS_FEP]"

int ethics_fep_bridge_default_config(ethics_fep_config_t* config) {
    if (!config) return NIMCP_ERROR_NULL_POINTER;
    config->harm_threshold = ETHICS_FEP_HARM_THRESHOLD;
    config->value_prior_weight = ETHICS_FEP_VALUE_PRIOR_WEIGHT;
    config->deontological_penalty = ETHICS_FEP_DEONTOLOGICAL_PENALTY;
    config->enable_value_priors = true;
    config->enable_deontological_constraints = true;
    config->enable_harm_prediction = true;
    config->pe_sensitivity = 1.0f;
    return NIMCP_SUCCESS;
}

ethics_fep_bridge_t* ethics_fep_bridge_create(const ethics_fep_config_t* config) {
    ethics_fep_bridge_t* bridge = (ethics_fep_bridge_t*)nimcp_malloc(sizeof(ethics_fep_bridge_t));
    if (!bridge) return NULL;
    memset(bridge, 0, sizeof(ethics_fep_bridge_t));
    if (config) bridge->config = *config;
    else ethics_fep_bridge_default_config(&bridge->config);
    bridge->base.mutex = nimcp_platform_mutex_create();
    if (!bridge->base.mutex) { nimcp_free(bridge); return NULL; }
    bridge->state.current_value_alignment = 0.5f;
    NIMCP_LOGGING_INFO(LOG_MODULE_ETHICS_FEP " Bridge created");
    return bridge;
}

void ethics_fep_bridge_destroy(ethics_fep_bridge_t* bridge) {
    if (!bridge) return;
    if (bridge->base.bio_async_enabled) ethics_fep_bridge_disconnect_bio_async(bridge);
    if (bridge->base.mutex) {
        nimcp_platform_mutex_destroy(bridge->base.mutex);
        nimcp_free(bridge->base.mutex);
    }
    nimcp_free(bridge);
}

int ethics_fep_bridge_connect_fep(ethics_fep_bridge_t* bridge, fep_system_t* fep) {
    if (!bridge || !fep) return NIMCP_ERROR_NULL_POINTER;
    nimcp_platform_mutex_lock(bridge->base.mutex);
    bridge->fep_system = fep;
    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return NIMCP_SUCCESS;
}

int ethics_fep_bridge_connect_ethics(ethics_fep_bridge_t* bridge, void* ethics) {
    if (!bridge || !ethics) return NIMCP_ERROR_NULL_POINTER;
    nimcp_platform_mutex_lock(bridge->base.mutex);
    bridge->ethics_system = ethics;
    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return NIMCP_SUCCESS;
}

int ethics_fep_apply_value_priors(ethics_fep_bridge_t* bridge) {
    if (!bridge || !bridge->fep_system) return NIMCP_ERROR_NULL_POINTER;
    if (!bridge->config.enable_value_priors) return NIMCP_SUCCESS;
    nimcp_platform_mutex_lock(bridge->base.mutex);
    bridge->effects.value_prior = bridge->config.value_prior_weight;
    NIMCP_LOGGING_DEBUG(LOG_MODULE_ETHICS_FEP " Applied value priors to FEP");
    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return NIMCP_SUCCESS;
}

int ethics_fep_constrain_policy(ethics_fep_bridge_t* bridge, bool is_ethical) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    if (!bridge->config.enable_deontological_constraints) return NIMCP_SUCCESS;
    nimcp_platform_mutex_lock(bridge->base.mutex);
    if (!is_ethical) {
        bridge->effects.ethical_constraint_active = true;
        bridge->state.harmful_actions_blocked++;
        bridge->stats.harm_preventions_total++;
        NIMCP_LOGGING_WARN(LOG_MODULE_ETHICS_FEP " Blocked unethical action");
    } else {
        bridge->state.ethical_policies_selected++;
        bridge->stats.ethical_selections_total++;
    }
    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return NIMCP_SUCCESS;
}

int ethics_fep_predict_harm(ethics_fep_bridge_t* bridge, float* harm_score) {
    if (!bridge || !harm_score) return NIMCP_ERROR_NULL_POINTER;
    if (!bridge->config.enable_harm_prediction) return NIMCP_SUCCESS;
    nimcp_platform_mutex_lock(bridge->base.mutex);
    *harm_score = bridge->effects.harm_prediction;
    if (*harm_score > bridge->config.harm_threshold) {
        NIMCP_LOGGING_WARN(LOG_MODULE_ETHICS_FEP " High harm predicted (%.2f)", *harm_score);
    }
    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return NIMCP_SUCCESS;
}

int ethics_fep_bridge_update(ethics_fep_bridge_t* bridge, uint64_t delta_ms) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    nimcp_platform_mutex_lock(bridge->base.mutex);
    if (bridge->state.ethical_policies_selected > 0) {
        float total = bridge->state.ethical_policies_selected + bridge->state.harmful_actions_blocked;
        bridge->state.current_value_alignment = bridge->state.ethical_policies_selected / total;
    }
    bridge->stats.avg_value_alignment = (bridge->stats.avg_value_alignment * 0.99f) +
        (bridge->state.current_value_alignment * 0.01f);
    bridge->effects.ethical_constraint_active = false;
    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return NIMCP_SUCCESS;
}

int ethics_fep_bridge_get_state(const ethics_fep_bridge_t* bridge, ethics_fep_state_t* state) {
    if (!bridge || !state) return NIMCP_ERROR_NULL_POINTER;
    nimcp_platform_mutex_lock(bridge->base.mutex);
    *state = bridge->state;
    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return NIMCP_SUCCESS;
}

int ethics_fep_bridge_get_stats(const ethics_fep_bridge_t* bridge, ethics_fep_stats_t* stats) {
    if (!bridge || !stats) return NIMCP_ERROR_NULL_POINTER;
    nimcp_platform_mutex_lock(bridge->base.mutex);
    *stats = bridge->stats;
    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return NIMCP_SUCCESS;
}

int ethics_fep_bridge_connect_bio_async(ethics_fep_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    if (bridge->base.bio_async_enabled) return NIMCP_SUCCESS;
    bio_module_info_t info = {
        .module_id = BIO_MODULE_FEP_ETHICS_BRIDGE,
        .module_name = "ethics_fep_bridge",
        .inbox_capacity = 32,
        .user_data = bridge
    };
    bridge->base.bio_ctx = bio_router_register_module(&info);
    if (bridge->base.bio_ctx) {
        bridge->base.bio_async_enabled = true;
        return NIMCP_SUCCESS;
    }
    return -1;
}

int ethics_fep_bridge_disconnect_bio_async(ethics_fep_bridge_t* bridge) {
    if (!bridge || !bridge->base.bio_async_enabled) return NIMCP_SUCCESS;
    if (bridge->base.bio_ctx) bio_router_unregister_module(bridge->base.bio_ctx);
    bridge->base.bio_async_enabled = false;
    return NIMCP_SUCCESS;
}

bool ethics_fep_bridge_is_bio_async_connected(const ethics_fep_bridge_t* bridge) {
    return bridge ? bridge->base.bio_async_enabled : false;
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

/**
 * WHAT: Query knowledge graph for Ethics FEP Bridge self-knowledge
 * WHY:  Enable self-awareness about module's role and connections
 * HOW:  Query KG for entity observations and relations
 */
int ethics_fep_bridge_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;
    const kg_entity_t* self = kg_reader_get_entity(kg, "Ethics_FEP_Bridge_Module");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            NIMCP_LOGGING_DEBUG("Ethics FEP bridge self-knowledge: %s", self->observations[i]);
        }
    }
    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Ethics_FEP_Bridge_Module");
    if (connections) { kg_relation_list_destroy(connections); }
    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Ethics_FEP_Bridge_Module");
    if (incoming) { kg_relation_list_destroy(incoming); }
    return self ? 1 : 0;
}
