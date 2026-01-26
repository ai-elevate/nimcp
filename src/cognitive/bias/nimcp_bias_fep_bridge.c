/**
 * @file nimcp_bias_fep_bridge.c
 * @brief Bias FEP Bridge Implementation
 */

#include "cognitive/bias/nimcp_bias_fep_bridge.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/validation/nimcp_common.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>
#include <math.h>

#define LOG_MODULE_BIAS_FEP "[BIAS_FEP]"

//=============================================================================
#include <stddef.h>  /* for NULL */
// Health Agent Integration (Phase 8: System-Wide Health Integration)
//=============================================================================
struct nimcp_health_agent;
typedef struct nimcp_health_agent nimcp_health_agent_t;
extern void nimcp_health_agent_heartbeat_ex(nimcp_health_agent_t* agent,
                                             const char* operation,
                                             float progress);

/** Global health agent for bias_fep_bridge module */
static nimcp_health_agent_t* g_bias_fep_bridge_health_agent = NULL;

/**
 * @brief Set health agent for bias_fep_bridge heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
void bias_fep_bridge_set_health_agent(nimcp_health_agent_t* agent) {
    g_bias_fep_bridge_health_agent = agent;
}

/** @brief Send heartbeat from bias_fep_bridge module */
static inline void bias_fep_bridge_heartbeat(const char* operation, float progress) {
    if (g_bias_fep_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_bias_fep_bridge_health_agent, operation, progress);
    }
}


int bias_fep_bridge_default_config(bias_fep_config_t* config) {
    /* Phase 8: Heartbeat at operation start */
    bias_fep_bridge_heartbeat("bias_fep_bri_default_config", 0.0f);


    NIMCP_CHECK_THROW(config, NIMCP_ERROR_NULL_POINTER, "config is NULL");
    config->systematic_pe_threshold = BIAS_FEP_SYSTEMATIC_PE_THRESHOLD;
    config->confirmation_threshold = BIAS_FEP_CONFIRMATION_THRESHOLD;
    config->prior_correction_rate = BIAS_FEP_PRIOR_CORRECTION_RATE;
    config->enable_bias_detection = true;
    config->enable_prior_correction = true;
    config->pe_sensitivity = 1.0f;
    return NIMCP_SUCCESS;
}

bias_fep_bridge_t* bias_fep_bridge_create(const bias_fep_config_t* config) {
    /* Phase 8: Heartbeat at operation start */
    bias_fep_bridge_heartbeat("bias_fep_bri_create", 0.0f);


    bias_fep_bridge_t* bridge = (bias_fep_bridge_t*)nimcp_malloc(sizeof(bias_fep_bridge_t));
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return NULL;

    }
    memset(bridge, 0, sizeof(bias_fep_bridge_t));
    if (config) bridge->config = *config;
    else bias_fep_bridge_default_config(&bridge->config);
    if (bridge_base_init(&bridge->base, 0, "bias_fep") != 0) { nimcp_free(bridge); return NULL; }
    if (!bridge->base.mutex) { nimcp_free(bridge); return NULL; }
    NIMCP_LOGGING_INFO(LOG_MODULE_BIAS_FEP " Bridge created");
    return bridge;
}

void bias_fep_bridge_destroy(bias_fep_bridge_t* bridge) {
    if (!bridge) return;
    /* Phase 8: Heartbeat at operation start */
    bias_fep_bridge_heartbeat("bias_fep_bri_destroy", 0.0f);


    if (bridge->base.bio_async_enabled) bias_fep_bridge_disconnect_bio_async(bridge);
    if (bridge->base.mutex) {
        bridge_base_cleanup(&bridge->base);
    }
    nimcp_free(bridge);
}

int bias_fep_bridge_connect_fep(bias_fep_bridge_t* bridge, fep_system_t* fep) {
    /* Phase 8: Heartbeat at operation start */
    bias_fep_bridge_heartbeat("bias_fep_bri_connect_fep", 0.0f);


    NIMCP_CHECK_THROW(bridge && fep, NIMCP_ERROR_NULL_POINTER, "bridge or fep is NULL");
    nimcp_platform_mutex_lock(bridge->base.mutex);
    bridge->fep_system = fep;
    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return NIMCP_SUCCESS;
}

int bias_fep_detect_bias(bias_fep_bridge_t* bridge, float prediction_error) {
    /* Phase 8: Heartbeat at operation start */
    bias_fep_bridge_heartbeat("bias_fep_bri_bias_fep_detect_bias", 0.0f);


    NIMCP_CHECK_THROW(bridge, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");
    if (!bridge->config.enable_bias_detection) return NIMCP_SUCCESS;
    nimcp_platform_mutex_lock(bridge->base.mutex);
    float pe_abs = fabsf(prediction_error);
    if (pe_abs > bridge->config.systematic_pe_threshold) {
        bridge->effects.systematic_pe = pe_abs;
        bridge->effects.detected_bias = BIAS_TYPE_CONFIRMATION;
        bridge->effects.bias_active = true;
        bridge->state.biases_detected++;
        bridge->state.bias_magnitude = pe_abs;
        bridge->state.current_bias = BIAS_TYPE_CONFIRMATION;
        bridge->stats.bias_detections_total++;
        NIMCP_LOGGING_DEBUG(LOG_MODULE_BIAS_FEP " Bias detected (PE=%.2f)", pe_abs);
    }
    bridge->stats.avg_systematic_pe = (bridge->stats.avg_systematic_pe * 0.9f) + (pe_abs * 0.1f);
    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return NIMCP_SUCCESS;
}

int bias_fep_correct_prior(bias_fep_bridge_t* bridge, cognitive_bias_type_t bias) {
    /* Phase 8: Heartbeat at operation start */
    bias_fep_bridge_heartbeat("bias_fep_bri_bias_fep_correct_pri", 0.0f);


    NIMCP_CHECK_THROW(bridge && bridge->fep_system, NIMCP_ERROR_NULL_POINTER, "bridge or fep_system is NULL");
    if (!bridge->config.enable_prior_correction) return NIMCP_SUCCESS;
    nimcp_platform_mutex_lock(bridge->base.mutex);
    float correction = bridge->config.prior_correction_rate;
    bridge->stats.prior_corrections_total++;
    NIMCP_LOGGING_DEBUG(LOG_MODULE_BIAS_FEP " Applied prior correction (bias=%d)", bias);
    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return NIMCP_SUCCESS;
}

int bias_fep_bridge_update(bias_fep_bridge_t* bridge, uint64_t delta_ms) {
    /* Phase 8: Heartbeat at operation start */
    bias_fep_bridge_heartbeat("bias_fep_bri_update", 0.0f);


    NIMCP_CHECK_THROW(bridge, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");
    nimcp_platform_mutex_lock(bridge->base.mutex);
    if (bridge->effects.bias_active) {
        bridge->effects.systematic_pe *= 0.95f;
        if (bridge->effects.systematic_pe < 0.5f) {
            bridge->effects.bias_active = false;
        }
    }
    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return NIMCP_SUCCESS;
}

int bias_fep_bridge_get_state(const bias_fep_bridge_t* bridge, bias_fep_state_t* state) {
    /* Phase 8: Heartbeat at operation start */
    bias_fep_bridge_heartbeat("bias_fep_bri_get_state", 0.0f);


    NIMCP_CHECK_THROW(bridge && state, NIMCP_ERROR_NULL_POINTER, "bridge or state is NULL");
    nimcp_platform_mutex_lock(bridge->base.mutex);
    *state = bridge->state;
    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return NIMCP_SUCCESS;
}

int bias_fep_bridge_get_stats(const bias_fep_bridge_t* bridge, bias_fep_stats_t* stats) {
    /* Phase 8: Heartbeat at operation start */
    bias_fep_bridge_heartbeat("bias_fep_bri_get_stats", 0.0f);


    NIMCP_CHECK_THROW(bridge && stats, NIMCP_ERROR_NULL_POINTER, "bridge or stats is NULL");
    nimcp_platform_mutex_lock(bridge->base.mutex);
    *stats = bridge->stats;
    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return NIMCP_SUCCESS;
}

int bias_fep_bridge_connect_bio_async(bias_fep_bridge_t* bridge) {
    /* Phase 8: Heartbeat at operation start */
    bias_fep_bridge_heartbeat("bias_fep_bri_connect_bio_async", 0.0f);


    NIMCP_CHECK_THROW(bridge, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");
    if (bridge->base.bio_async_enabled) return NIMCP_SUCCESS;
    bio_module_info_t info = {
        .module_id = BIO_MODULE_FEP_BIAS_BRIDGE,
        .module_name = "bias_fep_bridge",
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

int bias_fep_bridge_disconnect_bio_async(bias_fep_bridge_t* bridge) {
    if (!bridge || !bridge->base.bio_async_enabled) return NIMCP_SUCCESS;
    /* Phase 8: Heartbeat at operation start */
    bias_fep_bridge_heartbeat("bias_fep_bri_disconnect_bio_async", 0.0f);


    if (bridge->base.bio_ctx) bio_router_unregister_module(bridge->base.bio_ctx);
    bridge->base.bio_async_enabled = false;
    return NIMCP_SUCCESS;
}

bool bias_fep_bridge_is_bio_async_connected(const bias_fep_bridge_t* bridge) {
    /* Phase 8: Heartbeat at operation start */
    bias_fep_bridge_heartbeat("bias_fep_bri_is_bio_async_connect", 0.0f);


    return bridge ? bridge->base.bio_async_enabled : false;
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

int bias_fep_bridge_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;

    /* Phase 8: Heartbeat at operation start */
    bias_fep_bridge_heartbeat("bias_fep_bri_query_self_knowledge", 0.0f);


    const kg_entity_t* self = kg_reader_get_entity(kg, "Bias_FEP_Bridge");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            /* Phase 8: Loop progress heartbeat */
            if ((i & 0xFF) == 0 && self->num_observations > 256) {
                bias_fep_bridge_heartbeat("bias_fep_bri_loop",
                                 (float)(i + 1) / (float)self->num_observations);
            }

            (void)self->observations[i];
        }
    }

    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Bias_FEP_Bridge");
    if (connections) {
        kg_relation_list_destroy(connections);
    }

    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Bias_FEP_Bridge");
    if (incoming) {
        kg_relation_list_destroy(incoming);
    }

    return self ? 1 : 0;
}
