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

int bias_fep_bridge_default_config(bias_fep_config_t* config) {
    if (!config) return NIMCP_ERROR_NULL_POINTER;
    config->systematic_pe_threshold = BIAS_FEP_SYSTEMATIC_PE_THRESHOLD;
    config->confirmation_threshold = BIAS_FEP_CONFIRMATION_THRESHOLD;
    config->prior_correction_rate = BIAS_FEP_PRIOR_CORRECTION_RATE;
    config->enable_bias_detection = true;
    config->enable_prior_correction = true;
    config->pe_sensitivity = 1.0f;
    return NIMCP_SUCCESS;
}

bias_fep_bridge_t* bias_fep_bridge_create(const bias_fep_config_t* config) {
    bias_fep_bridge_t* bridge = (bias_fep_bridge_t*)nimcp_malloc(sizeof(bias_fep_bridge_t));
    if (!bridge) return NULL;
    memset(bridge, 0, sizeof(bias_fep_bridge_t));
    if (config) bridge->config = *config;
    else bias_fep_bridge_default_config(&bridge->config);
    bridge->base.mutex = nimcp_platform_mutex_create();
    if (!bridge->base.mutex) { nimcp_free(bridge); return NULL; }
    NIMCP_LOGGING_INFO(LOG_MODULE_BIAS_FEP " Bridge created");
    return bridge;
}

void bias_fep_bridge_destroy(bias_fep_bridge_t* bridge) {
    if (!bridge) return;
    if (bridge->base.bio_async_enabled) bias_fep_bridge_disconnect_bio_async(bridge);
    if (bridge->base.mutex) {
        nimcp_platform_mutex_destroy(bridge->base.mutex);
    }
    nimcp_free(bridge);
}

int bias_fep_bridge_connect_fep(bias_fep_bridge_t* bridge, fep_system_t* fep) {
    if (!bridge || !fep) return NIMCP_ERROR_NULL_POINTER;
    nimcp_platform_mutex_lock(bridge->base.mutex);
    bridge->fep_system = fep;
    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return NIMCP_SUCCESS;
}

int bias_fep_detect_bias(bias_fep_bridge_t* bridge, float prediction_error) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
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
    if (!bridge || !bridge->fep_system) return NIMCP_ERROR_NULL_POINTER;
    if (!bridge->config.enable_prior_correction) return NIMCP_SUCCESS;
    nimcp_platform_mutex_lock(bridge->base.mutex);
    float correction = bridge->config.prior_correction_rate;
    bridge->stats.prior_corrections_total++;
    NIMCP_LOGGING_DEBUG(LOG_MODULE_BIAS_FEP " Applied prior correction (bias=%d)", bias);
    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return NIMCP_SUCCESS;
}

int bias_fep_bridge_update(bias_fep_bridge_t* bridge, uint64_t delta_ms) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
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
    if (!bridge || !state) return NIMCP_ERROR_NULL_POINTER;
    nimcp_platform_mutex_lock(bridge->base.mutex);
    *state = bridge->state;
    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return NIMCP_SUCCESS;
}

int bias_fep_bridge_get_stats(const bias_fep_bridge_t* bridge, bias_fep_stats_t* stats) {
    if (!bridge || !stats) return NIMCP_ERROR_NULL_POINTER;
    nimcp_platform_mutex_lock(bridge->base.mutex);
    *stats = bridge->stats;
    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return NIMCP_SUCCESS;
}

int bias_fep_bridge_connect_bio_async(bias_fep_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
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
    if (bridge->base.bio_ctx) bio_router_unregister_module(bridge->base.bio_ctx);
    bridge->base.bio_async_enabled = false;
    return NIMCP_SUCCESS;
}

bool bias_fep_bridge_is_bio_async_connected(const bias_fep_bridge_t* bridge) {
    return bridge ? bridge->base.bio_async_enabled : false;
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

int bias_fep_bridge_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;

    const kg_entity_t* self = kg_reader_get_entity(kg, "Bias_FEP_Bridge");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
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
