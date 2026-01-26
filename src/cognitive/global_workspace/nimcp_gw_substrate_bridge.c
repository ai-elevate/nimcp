/**
 * @file nimcp_gw_substrate_bridge.c
 * @brief Global Workspace-Neural Substrate Bridge Implementation
 *
 * WHAT: Links global workspace to metabolic state
 * WHY: Conscious access requires sustained prefrontal-parietal activation
 * HOW: Monitors ATP/fatigue; modulates broadcast capacity, competition, integration
 */

#include "utils/bridge/nimcp_bridge_base.h"
#include "cognitive/global_workspace/nimcp_gw_substrate_bridge.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "cognitive/common/nimcp_metabolic_modulation.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/error/nimcp_error_codes.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>
#include <math.h>

//=============================================================================
#include <stddef.h>  /* for NULL */
// Health Agent Integration (Phase 8: System-Wide Health Integration)
//=============================================================================
struct nimcp_health_agent;
typedef struct nimcp_health_agent nimcp_health_agent_t;
extern void nimcp_health_agent_heartbeat_ex(nimcp_health_agent_t* agent,
                                             const char* operation,
                                             float progress);

/** Global health agent for gw_substrate_bridge module */
static nimcp_health_agent_t* g_gw_substrate_bridge_health_agent = NULL;

/**
 * @brief Set health agent for gw_substrate_bridge heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
void gw_substrate_bridge_set_health_agent(nimcp_health_agent_t* agent) {
    g_gw_substrate_bridge_health_agent = agent;
}

/** @brief Send heartbeat from gw_substrate_bridge module */
static inline void gw_substrate_bridge_heartbeat(const char* operation, float progress) {
    if (g_gw_substrate_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_gw_substrate_bridge_health_agent, operation, progress);
    }
}


/* Internal structure */
struct gw_substrate_bridge {
    bridge_base_t base;              /**< MUST be first: base bridge infrastructure */
    void* gw;
    neural_substrate_t* substrate;
    gw_substrate_config_t config;
    gw_substrate_effects_t effects;
    bio_router_t* router;
    bool bio_async_connected;
    struct {
        uint64_t update_count;
        float avg_broadcast_reach;
        float avg_coherence;
    } stats;
};

gw_substrate_config_t gw_substrate_default_config(void) {
    /* Phase 8: Heartbeat at operation start */
    gw_substrate_bridge_heartbeat("gw_substrate_gw_substrate_default", 0.0f);


    gw_substrate_config_t cfg = {
        .enable_atp_modulation = true,
        .enable_fatigue_modulation = true,
        .enable_bio_async = false,
        .atp_sensitivity = 1.0f,
        .fatigue_sensitivity = 1.0f,
        .min_capacity = 0.2f
    };
    return cfg;
}

gw_substrate_bridge_t* gw_substrate_bridge_create(void* gw, neural_substrate_t* substrate, const gw_substrate_config_t* config) {
    if (!substrate) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "substrate is NULL");

        return NULL;

    }

    /* Phase 8: Heartbeat at operation start */
    gw_substrate_bridge_heartbeat("gw_substrate_create", 0.0f);


    gw_substrate_bridge_t* bridge = nimcp_calloc(1, sizeof(gw_substrate_bridge_t));
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return NULL;

    }

    bridge->gw = gw;
    bridge->substrate = substrate;
    bridge->config = config ? *config : gw_substrate_default_config();

    bridge->effects.broadcast_reach = 1.0f;
    bridge->effects.ignition_threshold = 0.5f;
    bridge->effects.coherence = 1.0f;
    bridge->effects.processing_depth = 1.0f;
    bridge->effects.overall_capacity = 1.0f;

    return bridge;
}

void gw_substrate_bridge_destroy(gw_substrate_bridge_t* bridge) {
    /* Phase 8: Heartbeat at operation start */
    gw_substrate_bridge_heartbeat("gw_substrate_destroy", 0.0f);


    if (bridge) nimcp_free(bridge);
}

int gw_substrate_bridge_update(gw_substrate_bridge_t* bridge) {
    if (!bridge || !bridge->substrate) return -1;

    /* Phase 8: Heartbeat at operation start */
    gw_substrate_bridge_heartbeat("gw_substrate_update", 0.0f);


    substrate_metabolic_state_t metabolic;
    if (substrate_get_metabolic_state(bridge->substrate, &metabolic) != 0) return -1;

    float atp = metabolic.atp_level;
    float metabolic_cap = metabolic.metabolic_capacity;
    float min_cap = bridge->config.min_capacity;

    if (bridge->config.enable_atp_modulation) {
        bridge->effects.broadcast_reach = nimcp_clamp_f(atp * bridge->config.atp_sensitivity, min_cap, 1.0f);
        bridge->effects.processing_depth = nimcp_clamp_f(atp * 1.1f * bridge->config.atp_sensitivity, min_cap, 1.0f);
    }

    if (bridge->config.enable_fatigue_modulation) {
        bridge->effects.coherence = nimcp_clamp_f(metabolic_cap * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
        bridge->effects.ignition_threshold = nimcp_clamp_f(0.3f + (1.0f - metabolic_cap) * 0.4f, 0.3f, 0.7f);
    }

    bridge->effects.overall_capacity = (bridge->effects.broadcast_reach +
                                        bridge->effects.coherence +
                                        bridge->effects.processing_depth) / 3.0f;

    bridge->stats.update_count++;
    float alpha = 0.01f;
    bridge->stats.avg_broadcast_reach = (1.0f - alpha) * bridge->stats.avg_broadcast_reach + alpha * bridge->effects.broadcast_reach;
    bridge->stats.avg_coherence = (1.0f - alpha) * bridge->stats.avg_coherence + alpha * bridge->effects.coherence;

    return 0;
}

int gw_substrate_bridge_get_effects(const gw_substrate_bridge_t* bridge, gw_substrate_effects_t* effects) {
    if (!bridge || !effects) return -1;
    *effects = bridge->effects;
    /* Phase 8: Heartbeat at operation start */
    gw_substrate_bridge_heartbeat("gw_substrate_get_effects", 0.0f);


    return 0;
}

int gw_substrate_bridge_apply_effects(gw_substrate_bridge_t* bridge) {
    if (!bridge) return -1;
    /* Effects are computed; application is module-specific */
    /* Phase 8: Heartbeat at operation start */
    gw_substrate_bridge_heartbeat("gw_substrate_apply_effects", 0.0f);


    return 0;
}

int gw_substrate_bridge_register_bio_async(gw_substrate_bridge_t* bridge, bio_router_t* router) {
    if (!bridge) return -1;
    /* Phase 8: Heartbeat at operation start */
    gw_substrate_bridge_heartbeat("gw_substrate_register_bio_async", 0.0f);


    bridge->router = router;
    bridge->bio_async_connected = (router != NULL);
    return 0;
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

int gw_substrate_bridge_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;

    /* Phase 8: Heartbeat at operation start */
    gw_substrate_bridge_heartbeat("gw_substrate_query_self_knowledge", 0.0f);


    const kg_entity_t* self = kg_reader_get_entity(kg, "Global_Workspace_Substrate_Bridge");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            /* Phase 8: Loop progress heartbeat */
            if ((i & 0xFF) == 0 && self->num_observations > 256) {
                gw_substrate_bridge_heartbeat("gw_substrate_loop",
                                 (float)(i + 1) / (float)self->num_observations);
            }

            (void)self->observations[i];
        }
    }

    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Global_Workspace_Substrate_Bridge");
    if (connections) {
        kg_relation_list_destroy(connections);
    }

    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Global_Workspace_Substrate_Bridge");
    if (incoming) {
        kg_relation_list_destroy(incoming);
    }

    return self ? 1 : 0;
}
