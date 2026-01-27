/**
 * @file nimcp_attention_fep_bridge.c
 * @brief Free Energy Principle - Attention Integration Bridge Implementation
 * @version 1.0.0
 * @date 2025-12-12
 */

#include "cognitive/attention/nimcp_attention_fep_bridge.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/validation/nimcp_common.h"
#include "utils/exception/nimcp_exception_macros.h"
#include "security/nimcp_bbb_helpers.h"

#include <string.h>
#include <math.h>

/* Define LOG_MODULE for this file */
#define LOG_MODULE "attention_fep_bridge"

//=============================================================================
#include <stddef.h>  /* for NULL */
// Health Agent Integration (Phase 8: System-Wide Health Integration)
//=============================================================================
struct nimcp_health_agent;
typedef struct nimcp_health_agent nimcp_health_agent_t;
extern void nimcp_health_agent_heartbeat_ex(nimcp_health_agent_t* agent,
                                             const char* operation,
                                             float progress);

/** Global health agent for attention_fep_bridge module */
static nimcp_health_agent_t* g_attention_fep_bridge_health_agent = NULL;

/**
 * @brief Set health agent for attention_fep_bridge heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
void attention_fep_bridge_set_health_agent(nimcp_health_agent_t* agent) {
    g_attention_fep_bridge_health_agent = agent;
}

/** @brief Send heartbeat from attention_fep_bridge module */
static inline void attention_fep_bridge_heartbeat(const char* operation, float progress) {
    if (g_attention_fep_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_attention_fep_bridge_health_agent, operation, progress);
    }
}

BRIDGE_DEFINE_SECURITY_SETTERS(attention_fep_bridge)

/* ============================================================================
 * Default Configuration
 * ============================================================================ */

int attention_fep_bridge_default_config(attention_fep_config_t* config) {
    if (!config) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "attention_fep_bridge_default_config: config is NULL");
        return NIMCP_ERROR_NULL_POINTER;
    }

    /* FEP → Attention */
    /* Phase 8: Heartbeat at operation start */
    attention_fep_bridge_heartbeat("attention_fe_default_config", 0.0f);


    config->precision_gain_scaling = 1.0f;
    config->pe_attention_shift_threshold = ATTENTION_FEP_SURPRISE_SHIFT_THRESHOLD;
    config->efe_info_seeking_threshold = ATTENTION_FEP_EFE_THRESHOLD;
    config->enable_precision_gain_modulation = true;
    config->enable_surprise_attention_shift = true;
    config->enable_efe_info_seeking = true;

    /* Attention → FEP */
    config->attended_precision_boost = ATTENTION_FEP_ATTENDED_PRECISION_MULT;
    config->unattended_precision_reduction = ATTENTION_FEP_UNATTENDED_PRECISION_MULT;
    config->attention_learning_rate_scaling = 1.0f;
    config->enable_attentional_gating = true;
    config->enable_attention_lr_modulation = true;
    config->enable_focus_model_narrowing = true;

    /* Sensitivity */
    config->precision_sensitivity = 1.0f;
    config->attention_sensitivity = 1.0f;

    return 0;
}

/* ============================================================================
 * Lifecycle API
 * ============================================================================ */

attention_fep_bridge_t* attention_fep_bridge_create(const attention_fep_config_t* config) {
    /* Phase 8: Heartbeat at operation start */
    attention_fep_bridge_heartbeat("attention_fe_create", 0.0f);


    attention_fep_bridge_t* bridge = nimcp_malloc(sizeof(attention_fep_bridge_t));
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "attention_fep_bridge_create: failed to allocate bridge");
        return NULL;
    }

    memset(bridge, 0, sizeof(attention_fep_bridge_t));

    /* Set configuration */
    if (config) {
        bridge->config = *config;
    } else {
        attention_fep_bridge_default_config(&bridge->config);
    }

    /* Create mutex */
    if (bridge_base_init(&bridge->base, 0, "attention_fep") != 0) { nimcp_free(bridge); return NULL; }
    if (!bridge->base.mutex) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_OPERATION_FAILED, "attention_fep_bridge_create: failed to create mutex");
        nimcp_free(bridge);
        return NULL;
    }

    NIMCP_LOGGING_INFO("Created attention FEP bridge");
    return bridge;
}

void attention_fep_bridge_destroy(attention_fep_bridge_t* bridge) {
    if (!bridge) return;

    /* Disconnect bio-async if connected */
    /* Phase 8: Heartbeat at operation start */
    attention_fep_bridge_heartbeat("attention_fe_destroy", 0.0f);


    if (bridge->base.bio_async_enabled) {
        attention_fep_bridge_disconnect_bio_async(bridge);
    }

    /* Destroy mutex */
    if (bridge->base.mutex) {
        bridge_base_cleanup(&bridge->base);
    }

    nimcp_free(bridge);
    NIMCP_LOGGING_INFO("Destroyed attention FEP bridge");
}

/* ============================================================================
 * Connection API
 * ============================================================================ */

int attention_fep_bridge_connect_fep(
    attention_fep_bridge_t* bridge,
    fep_system_t* fep
) {
    if (!bridge || !fep) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "attention_fep_bridge_connect_fep: bridge or fep is NULL");
        return NIMCP_ERROR_NULL_POINTER;
    }

    /* Phase 8: Heartbeat at operation start */
    attention_fep_bridge_heartbeat("attention_fe_connect_fep", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    bridge->fep_system = fep;
    nimcp_mutex_unlock(bridge->base.mutex);

    NIMCP_LOGGING_INFO("Connected FEP system to attention bridge");
    return 0;
}

int attention_fep_bridge_connect_attention(
    attention_fep_bridge_t* bridge,
    multihead_attention_t attention
) {
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "attention_fep_bridge_connect_attention: bridge is NULL");
        return NIMCP_ERROR_NULL_POINTER;
    }

    /* Phase 8: Heartbeat at operation start */
    attention_fep_bridge_heartbeat("attention_fe_connect_attention", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    bridge->attention = attention;
    nimcp_mutex_unlock(bridge->base.mutex);

    NIMCP_LOGGING_INFO("Connected attention system to FEP bridge");
    return 0;
}

int attention_fep_bridge_disconnect(attention_fep_bridge_t* bridge) {
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "attention_fep_bridge_disconnect: bridge is NULL");
        return NIMCP_ERROR_NULL_POINTER;
    }

    /* Phase 8: Heartbeat at operation start */
    attention_fep_bridge_heartbeat("attention_fe_disconnect", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    bridge->fep_system = NULL;
    bridge->attention = NULL;
    nimcp_mutex_unlock(bridge->base.mutex);

    NIMCP_LOGGING_INFO("Disconnected all systems from attention FEP bridge");
    return 0;
}

/* ============================================================================
 * FEP → Attention Direction
 * ============================================================================ */

int attention_fep_apply_precision_gain_modulation(attention_fep_bridge_t* bridge) {
    if (!bridge || !bridge->fep_system) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "attention_fep_apply_precision_gain_modulation: bridge or fep_system is NULL");
        return NIMCP_ERROR_NULL_POINTER;
    }

    if (!bridge->config.enable_precision_gain_modulation) {
        return 0;
    }

    /* Phase 8: Heartbeat at operation start */
    attention_fep_bridge_heartbeat("attention_fe_attention_fep_apply_", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);

    /* Get current precision from FEP system */
    float precision = bridge->state.current_precision;

    /* Compute gain modifier */
    float gain_modifier = 1.0f;
    if (precision > ATTENTION_FEP_PRECISION_THRESHOLD) {
        gain_modifier = ATTENTION_FEP_HIGH_PRECISION_GAIN;
    } else {
        gain_modifier = ATTENTION_FEP_LOW_PRECISION_GAIN;
    }

    /* Apply sensitivity scaling */
    gain_modifier = 1.0f + (gain_modifier - 1.0f) * bridge->config.precision_sensitivity;

    /* Update effects */
    bridge->fep_effects.precision_gain_modifier = gain_modifier;
    bridge->fep_effects.total_gain_modulation = gain_modifier;

    /* Update stats */
    bridge->stats.precision_gain_modulations++;
    bridge->stats.avg_gain_modulation =
        (bridge->stats.avg_gain_modulation * (bridge->stats.precision_gain_modulations - 1) +
         gain_modifier) / bridge->stats.precision_gain_modulations;

    nimcp_mutex_unlock(bridge->base.mutex);

    NIMCP_LOGGING_DEBUG("Applied precision gain modulation: %f", gain_modifier);
    return 0;
}

int attention_fep_surprise_attention_shift(
    attention_fep_bridge_t* bridge,
    float pe_magnitude
) {
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "attention_fep_surprise_attention_shift: bridge is NULL");
        return NIMCP_ERROR_NULL_POINTER;
    }

    if (!bridge->config.enable_surprise_attention_shift) {
        return 0;
    }

    /* Phase 8: Heartbeat at operation start */
    attention_fep_bridge_heartbeat("attention_fe_attention_fep_surpri", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);

    /* Check if PE exceeds threshold */
    if (pe_magnitude > bridge->config.pe_attention_shift_threshold) {
        bridge->fep_effects.surprise_shift_active = true;
        bridge->state.surprise_shift_triggered = true;

        /* Update stats */
        bridge->stats.surprise_attention_shifts++;

        NIMCP_LOGGING_DEBUG("Surprise-driven attention shift triggered (PE: %f)", pe_magnitude);
    }

    bridge->fep_effects.current_prediction_error = pe_magnitude;
    bridge->state.current_prediction_error = pe_magnitude;

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int attention_fep_efe_info_seeking(attention_fep_bridge_t* bridge) {
    if (!bridge || !bridge->fep_system) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "attention_fep_efe_info_seeking: bridge or fep_system is NULL");
        return NIMCP_ERROR_NULL_POINTER;
    }

    if (!bridge->config.enable_efe_info_seeking) {
        return 0;
    }

    /* Phase 8: Heartbeat at operation start */
    attention_fep_bridge_heartbeat("attention_fe_attention_fep_efe_in", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);

    /* Get current EFE from FEP system */
    float efe = bridge->fep_effects.current_efe;

    /* Check if EFE indicates information-seeking */
    if (efe > bridge->config.efe_info_seeking_threshold) {
        bridge->fep_effects.info_seeking_active = true;
        bridge->state.info_seeking_active = true;

        /* Update stats */
        bridge->stats.efe_info_seeking_events++;

        NIMCP_LOGGING_DEBUG("EFE-guided info-seeking activated (EFE: %f)", efe);
    } else {
        bridge->fep_effects.info_seeking_active = false;
        bridge->state.info_seeking_active = false;
    }

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

/* ============================================================================
 * Attention → FEP Direction
 * ============================================================================ */

int attention_fep_apply_attentional_gating(attention_fep_bridge_t* bridge) {
    if (!bridge || !bridge->fep_system) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "attention_fep_apply_attentional_gating: bridge or fep_system is NULL");
        return NIMCP_ERROR_NULL_POINTER;
    }

    if (!bridge->config.enable_attentional_gating) {
        return 0;
    }

    /* Phase 8: Heartbeat at operation start */
    attention_fep_bridge_heartbeat("attention_fe_attention_fep_apply_", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);

    /* Get attention focus */
    float focus = bridge->state.current_attention_focus;

    /* Compute precision gating */
    float precision_mult = 1.0f;
    if (focus > ATTENTION_FEP_FOCUS_THRESHOLD) {
        precision_mult = bridge->config.attended_precision_boost;
    } else {
        precision_mult = bridge->config.unattended_precision_reduction;
    }

    /* Apply sensitivity scaling */
    precision_mult = 1.0f + (precision_mult - 1.0f) * bridge->config.attention_sensitivity;

    /* Update effects */
    bridge->attention_effects.precision_multiplier = precision_mult;
    bridge->attention_effects.gated_precision =
        bridge->state.current_precision * precision_mult;

    /* Update state and stats */
    bridge->state.attentional_gating_active = true;
    bridge->state.precision_gating = precision_mult;
    bridge->stats.attentional_gating_events++;

    nimcp_mutex_unlock(bridge->base.mutex);

    NIMCP_LOGGING_DEBUG("Applied attentional gating: %f", precision_mult);
    return 0;
}

int attention_fep_modulate_learning_rate(attention_fep_bridge_t* bridge) {
    if (!bridge || !bridge->fep_system) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "attention_fep_modulate_learning_rate: bridge or fep_system is NULL");
        return NIMCP_ERROR_NULL_POINTER;
    }

    if (!bridge->config.enable_attention_lr_modulation) {
        return 0;
    }

    /* Phase 8: Heartbeat at operation start */
    attention_fep_bridge_heartbeat("attention_fe_attention_fep_modula", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);

    /* Get attention gain */
    float gain = bridge->state.current_attention_focus;

    /* Compute LR modifier */
    float lr_modifier = 1.0f;
    if (gain > ATTENTION_FEP_FOCUS_THRESHOLD) {
        lr_modifier = ATTENTION_FEP_HIGH_GAIN_LR_MULT;
    } else {
        lr_modifier = ATTENTION_FEP_LOW_GAIN_LR_MULT;
    }

    /* Apply scaling */
    lr_modifier = 1.0f + (lr_modifier - 1.0f) * bridge->config.attention_learning_rate_scaling;

    /* Update effects */
    bridge->attention_effects.learning_rate_modifier = lr_modifier;

    /* Update state and stats */
    bridge->state.lr_modulation = lr_modifier;
    bridge->stats.lr_modulation_events++;

    nimcp_mutex_unlock(bridge->base.mutex);

    NIMCP_LOGGING_DEBUG("Modulated learning rate by attention: %f", lr_modifier);
    return 0;
}

int attention_fep_apply_focus_model_narrowing(attention_fep_bridge_t* bridge) {
    if (!bridge || !bridge->fep_system) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "attention_fep_apply_focus_model_narrowing: bridge or fep_system is NULL");
        return NIMCP_ERROR_NULL_POINTER;
    }

    if (!bridge->config.enable_focus_model_narrowing) {
        return 0;
    }

    /* Phase 8: Heartbeat at operation start */
    attention_fep_bridge_heartbeat("attention_fe_attention_fep_apply_", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);

    /* Get attention focus */
    float focus = bridge->state.current_attention_focus;

    /* Compute model narrowing factor (high focus = narrow model) */
    float narrowing = focus;

    /* Update effects */
    bridge->attention_effects.model_narrowing_factor = narrowing;

    /* Update stats */
    bridge->stats.model_narrowing_events++;

    nimcp_mutex_unlock(bridge->base.mutex);

    NIMCP_LOGGING_DEBUG("Applied focus model narrowing: %f", narrowing);
    return 0;
}

/* ============================================================================
 * Update Cycle
 * ============================================================================ */

int attention_fep_bridge_update(
    attention_fep_bridge_t* bridge,
    uint64_t delta_ms
) {
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "attention_fep_bridge_update: bridge is NULL");
        return NIMCP_ERROR_NULL_POINTER;
    }

    /* FEP → Attention */
    /* Phase 8: Heartbeat at operation start */
    attention_fep_bridge_heartbeat("attention_fe_update", 0.0f);


    attention_fep_apply_precision_gain_modulation(bridge);
    attention_fep_efe_info_seeking(bridge);

    /* Attention → FEP */
    attention_fep_apply_attentional_gating(bridge);
    attention_fep_modulate_learning_rate(bridge);
    attention_fep_apply_focus_model_narrowing(bridge);

    /* Update average stats */
    nimcp_mutex_lock(bridge->base.mutex);

    bridge->stats.avg_precision =
        (bridge->stats.avg_precision * 0.9f) + (bridge->state.current_precision * 0.1f);

    bridge->stats.avg_attention_focus =
        (bridge->stats.avg_attention_focus * 0.9f) + (bridge->state.current_attention_focus * 0.1f);

    bridge->stats.avg_prediction_error =
        (bridge->stats.avg_prediction_error * 0.9f) + (bridge->state.current_prediction_error * 0.1f);

    nimcp_mutex_unlock(bridge->base.mutex);

    /* Notify coordinator of update cycle completion */
    bridge_base_notify_coordinator_tick(&bridge->base, 0);
    return 0;
}

/* ============================================================================
 * State/Stats API
 * ============================================================================ */

int attention_fep_bridge_get_state(
    const attention_fep_bridge_t* bridge,
    attention_fep_state_t* state
) {
    if (!bridge || !state) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "attention_fep_bridge_get_state: bridge or state is NULL");
        return NIMCP_ERROR_NULL_POINTER;
    }

    /* Phase 8: Heartbeat at operation start */
    attention_fep_bridge_heartbeat("attention_fe_get_state", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    *state = bridge->state;
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

int attention_fep_bridge_get_stats(
    const attention_fep_bridge_t* bridge,
    attention_fep_stats_t* stats
) {
    if (!bridge || !stats) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "attention_fep_bridge_get_stats: bridge or stats is NULL");
        return NIMCP_ERROR_NULL_POINTER;
    }

    /* Phase 8: Heartbeat at operation start */
    attention_fep_bridge_heartbeat("attention_fe_get_stats", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    *stats = bridge->stats;
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

/* ============================================================================
 * Bio-Async Integration
 * ============================================================================ */

int attention_fep_bridge_connect_bio_async(attention_fep_bridge_t* bridge) {
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "attention_fep_bridge_connect_bio_async: bridge is NULL");
        return NIMCP_ERROR_NULL_POINTER;
    }
    if (bridge->base.bio_async_enabled) return 0;

    /* Phase 8: Heartbeat at operation start */
    attention_fep_bridge_heartbeat("attention_fe_connect_bio_async", 0.0f);


    bio_module_info_t info = {
        .module_id = BIO_MODULE_FEP_ATTENTION_BRIDGE,
        .module_name = "attention_fep_bridge",
        .inbox_capacity = 32,
        .user_data = bridge
    };

    bridge->base.bio_ctx = bio_router_register_module(&info);
    if (bridge->base.bio_ctx) {
        bridge->base.bio_async_enabled = true;
        NIMCP_LOGGING_INFO("Connected to bio-async router");
    } else {
        NIMCP_LOGGING_WARN("Bio-async router not available");
    }

    return 0;
}

int attention_fep_bridge_disconnect_bio_async(attention_fep_bridge_t* bridge) {
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "attention_fep_bridge_disconnect_bio_async: bridge is NULL");
        return NIMCP_ERROR_NULL_POINTER;
    }
    if (!bridge->base.bio_async_enabled) return 0;

    /* Phase 8: Heartbeat at operation start */
    attention_fep_bridge_heartbeat("attention_fe_disconnect_bio_async", 0.0f);


    if (bridge->base.bio_ctx) {
        bio_router_unregister_module(bridge->base.bio_ctx);
        bridge->base.bio_ctx = NULL;
    }

    bridge->base.bio_async_enabled = false;
    NIMCP_LOGGING_INFO("Disconnected from bio-async router");

    return 0;
}

bool attention_fep_bridge_is_bio_async_connected(
    const attention_fep_bridge_t* bridge
) {
    /* Phase 8: Heartbeat at operation start */
    attention_fep_bridge_heartbeat("attention_fe_is_bio_async_connect", 0.0f);


    return bridge ? bridge->base.bio_async_enabled : false;
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

/**
 * WHAT: Query knowledge graph for Attention FEP Bridge self-knowledge
 * WHY:  Enable self-awareness about module's role and connections
 * HOW:  Query KG for entity observations and relations
 */
int attention_fep_bridge_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;
    /* Phase 8: Heartbeat at operation start */
    attention_fep_bridge_heartbeat("attention_fe_query_self_knowledge", 0.0f);


    const kg_entity_t* self = kg_reader_get_entity(kg, "Attention_FEP_Bridge");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            /* Phase 8: Loop progress heartbeat */
            if ((i & 0xFF) == 0 && self->num_observations > 256) {
                attention_fep_bridge_heartbeat("attention_fe_loop",
                                 (float)(i + 1) / (float)self->num_observations);
            }

            NIMCP_LOGGING_DEBUG("Attention FEP Bridge self-knowledge: %s", self->observations[i]);
        }
    }
    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Attention_FEP_Bridge");
    if (connections) { kg_relation_list_destroy(connections); }
    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Attention_FEP_Bridge");
    if (incoming) { kg_relation_list_destroy(incoming); }
    return self ? 1 : 0;
}
