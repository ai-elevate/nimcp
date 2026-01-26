/**
 * @file nimcp_joy_fep_bridge.c
 * @brief Free Energy Principle - Joy/Euphoria Integration Bridge Implementation
 *
 * WHAT: Implements bidirectional integration between FEP and joy/euphoria system
 * WHY:  Joy arises from positive prediction errors (reward prediction errors) -
 *       better-than-expected outcomes generate positive affect
 * HOW:  FEP positive PEs trigger joy; joy modulates dopamine-mediated learning
 */

#include "cognitive/joy/nimcp_joy_fep_bridge.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/validation/nimcp_common.h"
#include "utils/exception/nimcp_exception_macros.h"

#include <string.h>
#include <math.h>

#define LOG_MODULE "joy_fep_bridge"

//=============================================================================
#include <stddef.h>  /* for NULL */
// Health Agent Integration (Phase 8: System-Wide Health Integration)
//=============================================================================
struct nimcp_health_agent;
typedef struct nimcp_health_agent nimcp_health_agent_t;
extern void nimcp_health_agent_heartbeat_ex(nimcp_health_agent_t* agent,
                                             const char* operation,
                                             float progress);

/** Global health agent for joy_fep_bridge module */
static nimcp_health_agent_t* g_joy_fep_bridge_health_agent = NULL;

/**
 * @brief Set health agent for joy_fep_bridge heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
static void joy_fep_bridge_set_health_agent(nimcp_health_agent_t* agent) {
    g_joy_fep_bridge_health_agent = agent;
}

/** @brief Send heartbeat from joy_fep_bridge module */
static inline void joy_fep_bridge_heartbeat(const char* operation, float progress) {
    if (g_joy_fep_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_joy_fep_bridge_health_agent, operation, progress);
    }
}


/* ============================================================================
 * Default Configuration
 * ============================================================================ */

int joy_fep_default_config(joy_fep_config_t* config) {
    NIMCP_CHECK_THROW(config, NIMCP_ERROR_NULL_POINTER, "config is NULL");

    /* FEP -> Joy */
    config->positive_pe_joy_gain = 1.0f;
    config->euphoria_threshold = 0.8f;  /* High positive PE triggers euphoria */
    config->enable_positive_pe_joy = true;
    config->enable_euphoria_detection = true;

    /* Joy -> FEP */
    config->joy_learning_rate_boost = 1.5f;  /* Joy enhances learning */
    config->joy_dopamine_modulation = 1.0f;
    config->enable_joy_learning_boost = true;
    config->enable_joy_dopamine = true;

    /* Sensitivity */
    config->fe_sensitivity = 1.0f;
    config->emotion_sensitivity = 1.0f;

    return 0;
}

/* ============================================================================
 * Lifecycle API
 * ============================================================================ */

joy_fep_bridge_t* joy_fep_create(
    const joy_fep_config_t* config
) {
    joy_fep_bridge_t* bridge = nimcp_malloc(sizeof(joy_fep_bridge_t));
    if (!bridge) {
        NIMCP_LOGGING_ERROR("Failed to allocate joy FEP bridge");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return NULL;
    }

    memset(bridge, 0, sizeof(joy_fep_bridge_t));

    /* Set configuration */
    if (config) {
        bridge->config = *config;
    } else {
        joy_fep_default_config(&bridge->config);
    }

    /* Create mutex */
    if (bridge_base_init(&bridge->base, 0, "joy_fep") != 0) { nimcp_free(bridge); return NULL; }
    if (!bridge->base.mutex) {
        NIMCP_LOGGING_ERROR("Failed to create mutex");
        nimcp_free(bridge);
        return NULL;
    }

    /* Initialize defaults */
    bridge->emotion_effects.learning_rate_boost = 1.0f;
    bridge->emotion_effects.dopamine_multiplier = 1.0f;
    bridge->emotion_effects.value_update_gain = 1.0f;

    NIMCP_LOGGING_INFO("Created joy FEP bridge");
    return bridge;
}

void joy_fep_destroy(joy_fep_bridge_t* bridge) {
    if (!bridge) return;

    /* Disconnect bio-async if connected */
    if (bridge->base.bio_async_enabled) {
        joy_fep_disconnect_bio_async(bridge);
    }

    /* Destroy mutex */
    if (bridge->base.mutex) {
        bridge_base_cleanup(&bridge->base);
    }

    nimcp_free(bridge);
    NIMCP_LOGGING_INFO("Destroyed joy FEP bridge");
}

/* ============================================================================
 * Connection API
 * ============================================================================ */

int joy_fep_connect_fep(
    joy_fep_bridge_t* bridge,
    fep_system_t* fep
) {
    NIMCP_CHECK_THROW(bridge && fep, NIMCP_ERROR_NULL_POINTER, "bridge or fep is NULL");

    nimcp_mutex_lock(bridge->base.mutex);
    bridge->fep_system = fep;
    nimcp_mutex_unlock(bridge->base.mutex);

    NIMCP_LOGGING_INFO("Connected FEP system to joy bridge");
    return 0;
}

int joy_fep_connect_joy(
    joy_fep_bridge_t* bridge,
    joy_system_t* joy
) {
    NIMCP_CHECK_THROW(bridge && joy, NIMCP_ERROR_NULL_POINTER, "bridge or joy is NULL");

    nimcp_mutex_lock(bridge->base.mutex);
    bridge->joy_system = joy;
    nimcp_mutex_unlock(bridge->base.mutex);

    NIMCP_LOGGING_INFO("Connected joy system to FEP bridge");
    return 0;
}

int joy_fep_disconnect(joy_fep_bridge_t* bridge) {
    NIMCP_CHECK_THROW(bridge, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

    nimcp_mutex_lock(bridge->base.mutex);
    bridge->fep_system = NULL;
    bridge->joy_system = NULL;
    nimcp_mutex_unlock(bridge->base.mutex);

    NIMCP_LOGGING_INFO("Disconnected all systems from joy FEP bridge");
    return 0;
}

/* ============================================================================
 * FEP -> Joy Direction
 * ============================================================================ */

int joy_fep_process_positive_pe(
    joy_fep_bridge_t* bridge,
    float pe_magnitude
) {
    NIMCP_CHECK_THROW(bridge, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");
    if (!bridge->config.enable_positive_pe_joy) return 0;

    nimcp_mutex_lock(bridge->base.mutex);

    /* Positive prediction error = better than expected = joy
     * PE > 0: outcome better than predicted -> positive affect
     * The magnitude determines intensity of joy
     */
    float joy_intensity = 0.0f;
    float positive_valence = 0.0f;
    bool euphoria = false;

    if (pe_magnitude > 0.0f) {
        /* Positive PE generates joy */
        joy_intensity = pe_magnitude * bridge->config.positive_pe_joy_gain;
        if (joy_intensity > 1.0f) joy_intensity = 1.0f;

        positive_valence = joy_intensity;

        /* Check for euphoria threshold */
        if (joy_intensity > bridge->config.euphoria_threshold) {
            euphoria = true;
            bridge->stats.euphoria_events++;
            NIMCP_LOGGING_INFO("Euphoria triggered: joy_intensity=%f", joy_intensity);
        }
    }

    bridge->fep_effects.joy_intensity_from_pe = joy_intensity;
    bridge->fep_effects.positive_valence = positive_valence;
    bridge->fep_effects.euphoria_triggered = euphoria;

    /* Update state */
    bridge->state.current_positive_pe = pe_magnitude > 0.0f ? pe_magnitude : 0.0f;
    bridge->state.joy_intensity = joy_intensity;
    bridge->state.euphoria_intensity = euphoria ? joy_intensity : 0.0f;
    bridge->state.joyful = (joy_intensity > 0.3f);
    bridge->state.euphoric = euphoria;

    /* Update stats */
    if (joy_intensity > 0.1f) {
        bridge->stats.joy_events++;
    }
    bridge->stats.avg_joy_intensity =
        (bridge->stats.avg_joy_intensity * 0.9f) + (joy_intensity * 0.1f);
    bridge->stats.avg_positive_pe =
        (bridge->stats.avg_positive_pe * 0.9f) + (bridge->state.current_positive_pe * 0.1f);

    nimcp_mutex_unlock(bridge->base.mutex);

    NIMCP_LOGGING_DEBUG("Processed positive PE: joy=%f, valence=%f, euphoria=%s",
                        joy_intensity, positive_valence, euphoria ? "yes" : "no");
    return 0;
}

int joy_fep_boost_learning_rate(
    joy_fep_bridge_t* bridge
) {
    NIMCP_CHECK_THROW(bridge, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");
    if (!bridge->config.enable_joy_learning_boost) return 0;

    nimcp_mutex_lock(bridge->base.mutex);

    /* Joy enhances learning rate
     * Positive affect facilitates encoding of successful behaviors
     * Dopamine release from positive RPE strengthens synaptic connections
     */
    float joy_intensity = bridge->state.joy_intensity;
    float learning_boost = 1.0f + (joy_intensity * (bridge->config.joy_learning_rate_boost - 1.0f));

    bridge->emotion_effects.learning_rate_boost = learning_boost;

    /* Dopamine multiplier */
    float dopamine = 1.0f + (joy_intensity * bridge->config.joy_dopamine_modulation);
    bridge->emotion_effects.dopamine_multiplier = dopamine;

    /* Value update gain - joy increases value assigned to current action */
    float value_gain = 1.0f + (joy_intensity * 0.5f);
    bridge->emotion_effects.value_update_gain = value_gain;

    nimcp_mutex_unlock(bridge->base.mutex);

    NIMCP_LOGGING_DEBUG("Boosted learning: rate=%f, dopamine=%f, value_gain=%f",
                        learning_boost, dopamine, value_gain);
    return 0;
}

/* ============================================================================
 * Update API
 * ============================================================================ */

int joy_fep_update(
    joy_fep_bridge_t* bridge,
    uint64_t delta_ms
) {
    NIMCP_CHECK_THROW(bridge, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

    /* Apply learning rate boost */
    joy_fep_boost_learning_rate(bridge);

    nimcp_mutex_lock(bridge->base.mutex);

    /* Joy decays relatively quickly compared to grief
     * This models the transient nature of positive affect
     */
    float decay = 0.995f;
    bridge->state.joy_intensity *= decay;
    bridge->state.euphoria_intensity *= decay;

    if (bridge->state.joy_intensity < 0.1f) {
        bridge->state.joyful = false;
    }
    if (bridge->state.euphoria_intensity < bridge->config.euphoria_threshold) {
        bridge->state.euphoric = false;
    }

    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

/* ============================================================================
 * Query API
 * ============================================================================ */

int joy_fep_get_state(
    const joy_fep_bridge_t* bridge,
    joy_fep_state_t* state
) {
    NIMCP_CHECK_THROW(bridge && state, NIMCP_ERROR_NULL_POINTER, "bridge or state is NULL");

    nimcp_mutex_lock(bridge->base.mutex);
    *state = bridge->state;
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

int joy_fep_get_stats(
    const joy_fep_bridge_t* bridge,
    joy_fep_stats_t* stats
) {
    NIMCP_CHECK_THROW(bridge && stats, NIMCP_ERROR_NULL_POINTER, "bridge or stats is NULL");

    nimcp_mutex_lock(bridge->base.mutex);
    *stats = bridge->stats;
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

/* ============================================================================
 * Bio-Async Integration
 * ============================================================================ */

int joy_fep_connect_bio_async(
    joy_fep_bridge_t* bridge
) {
    NIMCP_CHECK_THROW(bridge, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");
    if (bridge->base.bio_async_enabled) return 0;

    bio_module_info_t info = {
        .module_id = BIO_MODULE_FEP_JOY_BRIDGE,
        .module_name = "joy_fep_bridge",
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

int joy_fep_disconnect_bio_async(
    joy_fep_bridge_t* bridge
) {
    NIMCP_CHECK_THROW(bridge, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");
    if (!bridge->base.bio_async_enabled) return 0;

    if (bridge->base.bio_ctx) {
        bio_router_unregister_module(bridge->base.bio_ctx);
        bridge->base.bio_ctx = NULL;
    }

    bridge->base.bio_async_enabled = false;
    NIMCP_LOGGING_INFO("Disconnected from bio-async router");

    return 0;
}

bool joy_fep_is_bio_async_connected(
    const joy_fep_bridge_t* bridge
) {
    return bridge ? bridge->base.bio_async_enabled : false;
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

int joy_fep_bridge_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;

    const kg_entity_t* self = kg_reader_get_entity(kg, "Joy_FEP_Bridge");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            (void)self->observations[i];
        }
    }

    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Joy_FEP_Bridge");
    if (connections) {
        kg_relation_list_destroy(connections);
    }

    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Joy_FEP_Bridge");
    if (incoming) {
        kg_relation_list_destroy(incoming);
    }

    return self ? 1 : 0;
}
