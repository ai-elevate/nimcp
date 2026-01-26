/**
 * @file nimcp_mirror_thalamic_bridge.c
 * @brief Mirror Neurons-Thalamic Bridge Implementation
 */

#include "cognitive/mirror_neurons/nimcp_mirror_thalamic_bridge.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>

//=============================================================================
#include <stddef.h>  /* for NULL */
// Health Agent Integration (Phase 8: System-Wide Health Integration)
//=============================================================================
struct nimcp_health_agent;
typedef struct nimcp_health_agent nimcp_health_agent_t;
extern void nimcp_health_agent_heartbeat_ex(nimcp_health_agent_t* agent,
                                             const char* operation,
                                             float progress);

/** Global health agent for mirror_thalamic_bridge module */
static nimcp_health_agent_t* g_mirror_thalamic_bridge_health_agent = NULL;

/**
 * @brief Set health agent for mirror_thalamic_bridge heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
void mirror_thalamic_bridge_set_health_agent(nimcp_health_agent_t* agent) {
    g_mirror_thalamic_bridge_health_agent = agent;
}

/** @brief Send heartbeat from mirror_thalamic_bridge module */
static inline void mirror_thalamic_bridge_heartbeat(const char* operation, float progress) {
    if (g_mirror_thalamic_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_mirror_thalamic_bridge_health_agent, operation, progress);
    }
}


struct mirror_thalamic_bridge {
    bridge_base_t base;
    void* mirror;
    thalamic_router_t* router;
    mirror_thalamic_config_t config;
    mirror_thalamic_stats_t stats;
    float attention_weight;
};

mirror_thalamic_config_t mirror_thalamic_default_config(void) {
    /* Phase 8: Heartbeat at operation start */
    mirror_thalamic_bridge_heartbeat("mirror_thala_mirror_thalamic_defa", 0.0f);


    mirror_thalamic_config_t cfg = {
        .enable_attention_gating = true,
        .enable_empathy_boost = true,
        .min_mirroring_strength = 0.3f,
        .empathy_threshold = 0.5f
    };
    return cfg;
}

mirror_thalamic_bridge_t* mirror_thalamic_bridge_create(void* mirror, thalamic_router_t* router, const mirror_thalamic_config_t* config) {
    /* Phase 8: Heartbeat at operation start */
    mirror_thalamic_bridge_heartbeat("mirror_thala_create", 0.0f);


    mirror_thalamic_bridge_t* bridge = nimcp_calloc(1, sizeof(mirror_thalamic_bridge_t));
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return NULL;

    }
    if (bridge_base_init(&bridge->base, 0, "mirror_thalamic") != 0) { nimcp_free(bridge); return NULL; }
    if (!bridge->base.mutex) {
        nimcp_free(bridge);
        return NULL;
    }
    bridge->mirror = mirror;
    bridge->router = router;
    bridge->config = config ? *config : mirror_thalamic_default_config();
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return bridge;
}

void mirror_thalamic_bridge_destroy(mirror_thalamic_bridge_t* bridge) {
    if (!bridge) return;
    /* Phase 8: Heartbeat at operation start */
    mirror_thalamic_bridge_heartbeat("mirror_thala_destroy", 0.0f);


    if (bridge->base.mutex) {
        bridge_base_cleanup(&bridge->base);
    }
    nimcp_free(bridge);
}

int mirror_thalamic_bridge_reset(mirror_thalamic_bridge_t* bridge) {
    if (!bridge) return -1;
    /* Phase 8: Heartbeat at operation start */
    mirror_thalamic_bridge_heartbeat("mirror_thala_reset", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int mirror_thalamic_route_action(mirror_thalamic_bridge_t* bridge, const mirror_thalamic_signal_t* signal) {
    if (!bridge || !signal) return -1;
    /* Phase 8: Heartbeat at operation start */
    mirror_thalamic_bridge_heartbeat("mirror_thala_mirror_thalamic_rout", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    if (bridge->config.enable_attention_gating && signal->mirroring_strength < bridge->config.min_mirroring_strength) {
        nimcp_mutex_unlock(bridge->base.mutex);
        return 0;
    }
    bridge->stats.actions_mirrored++;
    bridge->stats.avg_mirroring_strength = (bridge->stats.avg_mirroring_strength * (bridge->stats.actions_mirrored - 1) +
                                            signal->mirroring_strength) / bridge->stats.actions_mirrored;
    if (signal->signal_type == MIRROR_SIGNAL_IMITATION) {
        bridge->stats.imitations_triggered++;
    }
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int mirror_thalamic_route_empathy(mirror_thalamic_bridge_t* bridge, const void* emotion, float resonance) {
    if (!bridge) return -1;
    /* Phase 8: Heartbeat at operation start */
    mirror_thalamic_bridge_heartbeat("mirror_thala_mirror_thalamic_rout", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    if (resonance >= bridge->config.empathy_threshold) {
        bridge->stats.empathic_responses++;
    }
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int mirror_thalamic_set_attention(mirror_thalamic_bridge_t* bridge, float attention) {
    if (!bridge) return -1;
    /* Phase 8: Heartbeat at operation start */
    mirror_thalamic_bridge_heartbeat("mirror_thala_mirror_thalamic_set_", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    bridge->attention_weight = attention < 0.0f ? 0.0f : (attention > 1.0f ? 1.0f : attention);
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int mirror_thalamic_get_attention(const mirror_thalamic_bridge_t* bridge, float* attention) {
    if (!bridge || !attention) return -1;
    *attention = bridge->attention_weight;
    /* Phase 8: Heartbeat at operation start */
    mirror_thalamic_bridge_heartbeat("mirror_thala_mirror_thalamic_get_", 0.0f);


    return 0;
}

int mirror_thalamic_bridge_get_stats(const mirror_thalamic_bridge_t* bridge, mirror_thalamic_stats_t* stats) {
    if (!bridge || !stats) return -1;
    *stats = bridge->stats;
    /* Phase 8: Heartbeat at operation start */
    mirror_thalamic_bridge_heartbeat("mirror_thala_get_stats", 0.0f);


    return 0;
}

/* ============================================================================
 * KG Self-Awareness Integration
 * ============================================================================ */

int mirror_thalamic_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;
    /* Phase 8: Heartbeat at operation start */
    mirror_thalamic_bridge_heartbeat("mirror_thala_mirror_thalamic_quer", 0.0f);


    const kg_entity_t* self = kg_reader_get_entity(kg, "Mirror_Thalamic_Bridge");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            /* Phase 8: Loop progress heartbeat */
            if ((i & 0xFF) == 0 && self->num_observations > 256) {
                mirror_thalamic_bridge_heartbeat("mirror_thala_loop",
                                 (float)(i + 1) / (float)self->num_observations);
            }

            NIMCP_LOGGING_DEBUG("Mirror thalamic bridge self-knowledge: %s", self->observations[i]);
        }
    }
    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Mirror_Thalamic_Bridge");
    if (connections) { kg_relation_list_destroy(connections); }
    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Mirror_Thalamic_Bridge");
    if (incoming) { kg_relation_list_destroy(incoming); }
    return self ? 1 : 0;
}
