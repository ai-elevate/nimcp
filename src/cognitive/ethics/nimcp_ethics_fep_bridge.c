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
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>
#include <math.h>

#define LOG_MODULE_ETHICS_FEP "[ETHICS_FEP]"

//=============================================================================
#include <stddef.h>  /* for NULL */
// Health Agent Integration (Phase 8: System-Wide Health Integration)
//=============================================================================
struct nimcp_health_agent;
typedef struct nimcp_health_agent nimcp_health_agent_t;
extern void nimcp_health_agent_heartbeat_ex(nimcp_health_agent_t* agent,
                                             const char* operation,
                                             float progress);

/** Global health agent for ethics_fep_bridge module */
static nimcp_health_agent_t* g_ethics_fep_bridge_health_agent = NULL;

/**
 * @brief Set health agent for ethics_fep_bridge heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
void ethics_fep_bridge_set_health_agent(nimcp_health_agent_t* agent) {
    g_ethics_fep_bridge_health_agent = agent;
}

/** @brief Send heartbeat from ethics_fep_bridge module */
static inline void ethics_fep_bridge_heartbeat(const char* operation, float progress) {
    if (g_ethics_fep_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_ethics_fep_bridge_health_agent, operation, progress);
    }
}


int ethics_fep_bridge_default_config(ethics_fep_config_t* config) {
    /* Phase 8: Heartbeat at operation start */
    ethics_fep_bridge_heartbeat("ethics_fep_b_default_config", 0.0f);


    NIMCP_CHECK_THROW(config, NIMCP_ERROR_NULL_POINTER, "config is NULL");
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
    /* Phase 8: Heartbeat at operation start */
    ethics_fep_bridge_heartbeat("ethics_fep_b_create", 0.0f);


    ethics_fep_bridge_t* bridge = (ethics_fep_bridge_t*)nimcp_malloc(sizeof(ethics_fep_bridge_t));
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return NULL;

    }
    memset(bridge, 0, sizeof(ethics_fep_bridge_t));
    if (config) bridge->config = *config;
    else ethics_fep_bridge_default_config(&bridge->config);
    if (bridge_base_init(&bridge->base, 0, "ethics_fep") != 0) { nimcp_free(bridge); return NULL; }
    if (!bridge->base.mutex) { nimcp_free(bridge); return NULL; }
    bridge->state.current_value_alignment = 0.5f;
    NIMCP_LOGGING_INFO(LOG_MODULE_ETHICS_FEP " Bridge created");
    return bridge;
}

void ethics_fep_bridge_destroy(ethics_fep_bridge_t* bridge) {
    if (!bridge) return;
    /* Phase 8: Heartbeat at operation start */
    ethics_fep_bridge_heartbeat("ethics_fep_b_destroy", 0.0f);


    if (bridge->base.bio_async_enabled) ethics_fep_bridge_disconnect_bio_async(bridge);
    if (bridge->base.mutex) {
        bridge_base_cleanup(&bridge->base);
    }
    nimcp_free(bridge);
}

int ethics_fep_bridge_connect_fep(ethics_fep_bridge_t* bridge, fep_system_t* fep) {
    /* Phase 8: Heartbeat at operation start */
    ethics_fep_bridge_heartbeat("ethics_fep_b_connect_fep", 0.0f);


    NIMCP_CHECK_THROW(bridge && fep, NIMCP_ERROR_NULL_POINTER, "bridge or fep is NULL");
    nimcp_platform_mutex_lock(bridge->base.mutex);
    bridge->fep_system = fep;
    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return NIMCP_SUCCESS;
}

int ethics_fep_bridge_connect_ethics(ethics_fep_bridge_t* bridge, void* ethics) {
    /* Phase 8: Heartbeat at operation start */
    ethics_fep_bridge_heartbeat("ethics_fep_b_connect_ethics", 0.0f);


    NIMCP_CHECK_THROW(bridge && ethics, NIMCP_ERROR_NULL_POINTER, "bridge or ethics is NULL");
    nimcp_platform_mutex_lock(bridge->base.mutex);
    bridge->ethics_system = ethics;
    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return NIMCP_SUCCESS;
}

int ethics_fep_apply_value_priors(ethics_fep_bridge_t* bridge) {
    /* Phase 8: Heartbeat at operation start */
    ethics_fep_bridge_heartbeat("ethics_fep_b_ethics_fep_apply_val", 0.0f);


    NIMCP_CHECK_THROW(bridge && bridge->fep_system, NIMCP_ERROR_NULL_POINTER, "bridge or fep_system is NULL");
    if (!bridge->config.enable_value_priors) return NIMCP_SUCCESS;
    nimcp_platform_mutex_lock(bridge->base.mutex);
    bridge->effects.value_prior = bridge->config.value_prior_weight;
    NIMCP_LOGGING_DEBUG(LOG_MODULE_ETHICS_FEP " Applied value priors to FEP");
    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return NIMCP_SUCCESS;
}

int ethics_fep_constrain_policy(ethics_fep_bridge_t* bridge, bool is_ethical) {
    /* Phase 8: Heartbeat at operation start */
    ethics_fep_bridge_heartbeat("ethics_fep_b_ethics_fep_constrain", 0.0f);


    NIMCP_CHECK_THROW(bridge, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");
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
    /* Phase 8: Heartbeat at operation start */
    ethics_fep_bridge_heartbeat("ethics_fep_b_ethics_fep_predict_h", 0.0f);


    NIMCP_CHECK_THROW(bridge && harm_score, NIMCP_ERROR_NULL_POINTER, "bridge or harm_score is NULL");
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
    /* Phase 8: Heartbeat at operation start */
    ethics_fep_bridge_heartbeat("ethics_fep_b_update", 0.0f);


    NIMCP_CHECK_THROW(bridge, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");
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
    /* Phase 8: Heartbeat at operation start */
    ethics_fep_bridge_heartbeat("ethics_fep_b_get_state", 0.0f);


    NIMCP_CHECK_THROW(bridge && state, NIMCP_ERROR_NULL_POINTER, "bridge or state is NULL");
    nimcp_platform_mutex_lock(bridge->base.mutex);
    *state = bridge->state;
    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return NIMCP_SUCCESS;
}

int ethics_fep_bridge_get_stats(const ethics_fep_bridge_t* bridge, ethics_fep_stats_t* stats) {
    /* Phase 8: Heartbeat at operation start */
    ethics_fep_bridge_heartbeat("ethics_fep_b_get_stats", 0.0f);


    NIMCP_CHECK_THROW(bridge && stats, NIMCP_ERROR_NULL_POINTER, "bridge or stats is NULL");
    nimcp_platform_mutex_lock(bridge->base.mutex);
    *stats = bridge->stats;
    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return NIMCP_SUCCESS;
}

int ethics_fep_bridge_connect_bio_async(ethics_fep_bridge_t* bridge) {
    /* Phase 8: Heartbeat at operation start */
    ethics_fep_bridge_heartbeat("ethics_fep_b_connect_bio_async", 0.0f);


    NIMCP_CHECK_THROW(bridge, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");
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
    /* Phase 8: Heartbeat at operation start */
    ethics_fep_bridge_heartbeat("ethics_fep_b_disconnect_bio_async", 0.0f);


    if (bridge->base.bio_ctx) bio_router_unregister_module(bridge->base.bio_ctx);
    bridge->base.bio_async_enabled = false;
    return NIMCP_SUCCESS;
}

bool ethics_fep_bridge_is_bio_async_connected(const ethics_fep_bridge_t* bridge) {
    /* Phase 8: Heartbeat at operation start */
    ethics_fep_bridge_heartbeat("ethics_fep_b_is_bio_async_connect", 0.0f);


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
    /* Phase 8: Heartbeat at operation start */
    ethics_fep_bridge_heartbeat("ethics_fep_b_query_self_knowledge", 0.0f);


    const kg_entity_t* self = kg_reader_get_entity(kg, "Ethics_FEP_Bridge_Module");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            /* Phase 8: Loop progress heartbeat */
            if ((i & 0xFF) == 0 && self->num_observations > 256) {
                ethics_fep_bridge_heartbeat("ethics_fep_b_loop",
                                 (float)(i + 1) / (float)self->num_observations);
            }

            NIMCP_LOGGING_DEBUG("Ethics FEP bridge self-knowledge: %s", self->observations[i]);
        }
    }
    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Ethics_FEP_Bridge_Module");
    if (connections) { kg_relation_list_destroy(connections); }
    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Ethics_FEP_Bridge_Module");
    if (incoming) { kg_relation_list_destroy(incoming); }
    return self ? 1 : 0;
}
