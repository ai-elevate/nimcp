/**
 * @file nimcp_self_awareness_extended_substrate_bridge.c
 * @brief Extended Self-Awareness-Neural Substrate Bridge Implementation
 *
 * WHAT: Links extended self-awareness to metabolic state
 * WHY: Extended self-awareness requires sustained prefrontal and parietal resources
 * HOW: Monitors ATP/fatigue; modulates metacognition, temporal self, narrative integration
 */

#include "cognitive/self_awareness_extended/nimcp_self_awareness_extended_substrate_bridge.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "cognitive/common/nimcp_metabolic_modulation.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/platform/nimcp_platform_mutex.h"
#include <string.h>
#include <math.h>

struct self_awareness_ext_substrate_bridge {
    bridge_base_t base;                 /* MUST be first: base bridge infrastructure */
    void* self_awareness_ext;
    neural_substrate_t* substrate;
    self_awareness_ext_substrate_config_t config;
    self_awareness_ext_substrate_effects_t effects;
    bio_router_t* router;
    bool bio_async_connected;
    uint64_t update_count;
};

self_awareness_ext_substrate_config_t self_awareness_ext_substrate_default_config(void) {
    self_awareness_ext_substrate_config_t cfg = {
        .enable_atp_modulation = true,
        .enable_fatigue_modulation = true,
        .enable_bio_async = false,
        .atp_sensitivity = 1.0f,
        .fatigue_sensitivity = 1.0f,
        .min_capacity = 0.2f
    };
    return cfg;
}

self_awareness_ext_substrate_bridge_t* self_awareness_ext_substrate_bridge_create(void* self_awareness_ext, neural_substrate_t* substrate, const self_awareness_ext_substrate_config_t* config) {
    if (!substrate) return NULL;

    self_awareness_ext_substrate_bridge_t* bridge = nimcp_calloc(1, sizeof(self_awareness_ext_substrate_bridge_t));
    if (!bridge) return NULL;

    bridge->self_awareness_ext = self_awareness_ext;
    bridge->substrate = substrate;
    bridge->config = config ? *config : self_awareness_ext_substrate_default_config();

    /* Initialize mutex */
    bridge->base.mutex = (nimcp_mutex_t*)nimcp_malloc(sizeof(nimcp_mutex_t));
    if (!bridge->base.mutex) {
        NIMCP_LOGGING_ERROR("Failed to allocate mutex for self awareness ext substrate bridge");
        nimcp_free(bridge);
        return NULL;
    }

    if (nimcp_platform_mutex_init(bridge->base.mutex, false) != 0) {
        NIMCP_LOGGING_ERROR("Failed to initialize mutex for self awareness ext substrate bridge");
        nimcp_free(bridge->base.mutex);
        nimcp_free(bridge);
        return NULL;
    }

    bridge->effects.metacognitive_depth = 1.0f;
    bridge->effects.temporal_self_coherence = 1.0f;
    bridge->effects.narrative_integration = 1.0f;
    bridge->effects.future_self_projection = 1.0f;
    bridge->effects.overall_capacity = 1.0f;

    return bridge;
}

void self_awareness_ext_substrate_bridge_destroy(self_awareness_ext_substrate_bridge_t* bridge) {
    if (!bridge) return;

    /* Destroy mutex */
    if (bridge->base.mutex) {
        nimcp_platform_mutex_destroy(bridge->base.mutex);
        nimcp_free(bridge->base.mutex);
    }

    nimcp_free(bridge);
}

int self_awareness_ext_substrate_bridge_update(self_awareness_ext_substrate_bridge_t* bridge) {
    if (!bridge || !bridge->substrate) return -1;

    nimcp_platform_mutex_lock(bridge->base.mutex);

    substrate_metabolic_state_t metabolic;
    if (substrate_get_metabolic_state(bridge->substrate, &metabolic) != 0) {
        nimcp_platform_mutex_unlock(bridge->base.mutex);
        return -1;
    }

    float atp = metabolic.atp_level;
    float metabolic_cap = metabolic.metabolic_capacity;
    float min_cap = bridge->config.min_capacity;

    if (bridge->config.enable_atp_modulation) {
        /* Metacognitive depth requires stable prefrontal resources */
        bridge->effects.metacognitive_depth = nimcp_clamp_f(atp * bridge->config.atp_sensitivity, min_cap, 1.0f);
        /* Future self projection is cognitively demanding */
        bridge->effects.future_self_projection = nimcp_clamp_f(atp * 0.9f * bridge->config.atp_sensitivity, min_cap, 1.0f);
    }

    if (bridge->config.enable_fatigue_modulation) {
        /* Temporal self coherence decreases with fatigue */
        bridge->effects.temporal_self_coherence = nimcp_clamp_f(metabolic_cap * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
        /* Narrative integration is vulnerable to metabolic stress */
        bridge->effects.narrative_integration = nimcp_clamp_f(metabolic_cap * 0.85f * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
    }

    bridge->effects.overall_capacity = (bridge->effects.metacognitive_depth +
                                        bridge->effects.temporal_self_coherence +
                                        bridge->effects.narrative_integration +
                                        bridge->effects.future_self_projection) / 4.0f;

    bridge->update_count++;

    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return 0;
}

int self_awareness_ext_substrate_bridge_get_effects(const self_awareness_ext_substrate_bridge_t* bridge, self_awareness_ext_substrate_effects_t* effects) {
    if (!bridge || !effects) return -1;
    *effects = bridge->effects;
    return 0;
}

int self_awareness_ext_substrate_bridge_apply_effects(self_awareness_ext_substrate_bridge_t* bridge) {
    if (!bridge) return -1;
    return 0;
}

int self_awareness_ext_substrate_bridge_register_bio_async(self_awareness_ext_substrate_bridge_t* bridge, bio_router_t* router) {
    if (!bridge) return -1;
    bridge->router = router;
    bridge->bio_async_connected = (router != NULL);
    return 0;
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

int self_awareness_ext_substrate_bridge_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;

    const kg_entity_t* self = kg_reader_get_entity(kg, "Self_Awareness_Extended_Substrate_Bridge");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            (void)self->observations[i];
        }
    }

    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Self_Awareness_Extended_Substrate_Bridge");
    if (connections) {
        kg_relation_list_destroy(connections);
    }

    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Self_Awareness_Extended_Substrate_Bridge");
    if (incoming) {
        kg_relation_list_destroy(incoming);
    }

    return self ? 1 : 0;
}
