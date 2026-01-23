/**
 * @file nimcp_knowledge_fep_bridge.c
 * @brief Knowledge FEP Bridge Implementation
 */

#include "cognitive/knowledge/nimcp_knowledge_fep_bridge.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/validation/nimcp_common.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>
#include <math.h>

#define LOG_MODULE_KNOWLEDGE_FEP "[KNOWLEDGE_FEP]"

int knowledge_fep_bridge_default_config(knowledge_fep_config_t* config) {
    NIMCP_CHECK_THROW(config, NIMCP_ERROR_NULL_POINTER, "config is NULL");
    config->knowledge_update_threshold = KNOWLEDGE_FEP_UPDATE_THRESHOLD;
    config->semantic_prior_weight = KNOWLEDGE_FEP_SEMANTIC_PRIOR_WEIGHT;
    config->enable_knowledge_updates = true;
    config->enable_semantic_priors = true;
    config->pe_sensitivity = 1.0f;
    return NIMCP_SUCCESS;
}

knowledge_fep_bridge_t* knowledge_fep_bridge_create(const knowledge_fep_config_t* config) {
    knowledge_fep_bridge_t* bridge = (knowledge_fep_bridge_t*)nimcp_malloc(sizeof(knowledge_fep_bridge_t));
    if (!bridge) return NULL;
    memset(bridge, 0, sizeof(knowledge_fep_bridge_t));
    if (config) bridge->config = *config;
    else knowledge_fep_bridge_default_config(&bridge->config);
    if (bridge_base_init(&bridge->base, 0, "knowledge_fep") != 0) { nimcp_free(bridge); return NULL; }
    if (!bridge->base.mutex) { nimcp_free(bridge); return NULL; }
    bridge->state.current_semantic_prior = 0.5f;
    NIMCP_LOGGING_INFO(LOG_MODULE_KNOWLEDGE_FEP " Bridge created");
    return bridge;
}

void knowledge_fep_bridge_destroy(knowledge_fep_bridge_t* bridge) {
    if (!bridge) return;
    if (bridge->base.bio_async_enabled) knowledge_fep_bridge_disconnect_bio_async(bridge);
    if (bridge->base.mutex) {
        bridge_base_cleanup(&bridge->base);
    }
    nimcp_free(bridge);
}

int knowledge_fep_bridge_connect_fep(knowledge_fep_bridge_t* bridge, fep_system_t* fep) {
    NIMCP_CHECK_THROW(bridge && fep, NIMCP_ERROR_NULL_POINTER, "bridge or fep is NULL");
    nimcp_mutex_lock(bridge->base.mutex);
    bridge->fep_system = fep;
    nimcp_mutex_unlock(bridge->base.mutex);
    return NIMCP_SUCCESS;
}

int knowledge_fep_bridge_connect_knowledge(knowledge_fep_bridge_t* bridge, knowledge_system_t knowledge) {
    NIMCP_CHECK_THROW(bridge && knowledge, NIMCP_ERROR_NULL_POINTER, "bridge or knowledge is NULL");
    nimcp_mutex_lock(bridge->base.mutex);
    bridge->knowledge_system = knowledge;
    nimcp_mutex_unlock(bridge->base.mutex);
    return NIMCP_SUCCESS;
}

int knowledge_fep_update_knowledge(knowledge_fep_bridge_t* bridge, float prediction_error) {
    NIMCP_CHECK_THROW(bridge, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");
    if (!bridge->config.enable_knowledge_updates) return NIMCP_SUCCESS;
    nimcp_mutex_lock(bridge->base.mutex);
    if (fabsf(prediction_error) > bridge->config.knowledge_update_threshold) {
        bridge->state.knowledge_updates++;
        bridge->stats.updates_total++;
        bridge->effects.semantic_pe = fabsf(prediction_error);
    }
    bridge->stats.avg_semantic_pe = (bridge->stats.avg_semantic_pe * 0.9f) + (fabsf(prediction_error) * 0.1f);
    nimcp_mutex_unlock(bridge->base.mutex);
    return NIMCP_SUCCESS;
}

int knowledge_fep_apply_semantic_priors(knowledge_fep_bridge_t* bridge) {
    NIMCP_CHECK_THROW(bridge && bridge->fep_system, NIMCP_ERROR_NULL_POINTER, "bridge or fep_system is NULL");
    nimcp_mutex_lock(bridge->base.mutex);
    /* Apply knowledge as semantic priors to FEP */
    bridge->state.current_semantic_prior = bridge->config.semantic_prior_weight;
    nimcp_mutex_unlock(bridge->base.mutex);
    return NIMCP_SUCCESS;
}

int knowledge_fep_bridge_update(knowledge_fep_bridge_t* bridge, uint64_t delta_ms) {
    NIMCP_CHECK_THROW(bridge, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");
    nimcp_mutex_lock(bridge->base.mutex);
    bridge->effects.knowledge_confidence = 1.0f - (bridge->effects.semantic_pe / 10.0f);
    if (bridge->effects.knowledge_confidence < 0.0f) bridge->effects.knowledge_confidence = 0.0f;
    nimcp_mutex_unlock(bridge->base.mutex);
    return NIMCP_SUCCESS;
}

int knowledge_fep_bridge_get_state(const knowledge_fep_bridge_t* bridge, knowledge_fep_state_t* state) {
    NIMCP_CHECK_THROW(bridge && state, NIMCP_ERROR_NULL_POINTER, "bridge or state is NULL");
    nimcp_mutex_lock(bridge->base.mutex);
    *state = bridge->state;
    nimcp_mutex_unlock(bridge->base.mutex);
    return NIMCP_SUCCESS;
}

int knowledge_fep_bridge_get_stats(const knowledge_fep_bridge_t* bridge, knowledge_fep_stats_t* stats) {
    NIMCP_CHECK_THROW(bridge && stats, NIMCP_ERROR_NULL_POINTER, "bridge or stats is NULL");
    nimcp_mutex_lock(bridge->base.mutex);
    *stats = bridge->stats;
    nimcp_mutex_unlock(bridge->base.mutex);
    return NIMCP_SUCCESS;
}

int knowledge_fep_bridge_connect_bio_async(knowledge_fep_bridge_t* bridge) {
    NIMCP_CHECK_THROW(bridge, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");
    if (bridge->base.bio_async_enabled) return NIMCP_SUCCESS;
    bio_module_info_t info = {
        .module_id = BIO_MODULE_FEP_KNOWLEDGE_BRIDGE,
        .module_name = "knowledge_fep_bridge",
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

int knowledge_fep_bridge_disconnect_bio_async(knowledge_fep_bridge_t* bridge) {
    if (!bridge || !bridge->base.bio_async_enabled) return NIMCP_SUCCESS;
    if (bridge->base.bio_ctx) bio_router_unregister_module(bridge->base.bio_ctx);
    bridge->base.bio_async_enabled = false;
    return NIMCP_SUCCESS;
}

bool knowledge_fep_bridge_is_bio_async_connected(const knowledge_fep_bridge_t* bridge) {
    return bridge ? bridge->base.bio_async_enabled : false;
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

/**
 * @brief Query knowledge graph for self-knowledge about FEP knowledge bridge
 *
 * WHAT: Retrieves entity observations and relations for FEP-knowledge bridge
 * WHY: Enables self-aware introspection of module capabilities
 * HOW: Uses kg_reader to query JSONL knowledge graph
 *
 * @param kg Knowledge graph reader instance
 * @return 1 if self-knowledge found, 0 otherwise
 */
int knowledge_fep_bridge_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;

    const kg_entity_t* self = kg_reader_get_entity(kg, "Knowledge_FEP_Bridge");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            (void)self->observations[i];
        }
    }

    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Knowledge_FEP_Bridge");
    if (connections) {
        kg_relation_list_destroy(connections);
    }

    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Knowledge_FEP_Bridge");
    if (incoming) {
        kg_relation_list_destroy(incoming);
    }

    return self ? 1 : 0;
}
