/**
 * @file nimcp_personality_fep_bridge.c
 * @brief Free Energy Principle - Personality Integration Bridge Implementation
 */

#include "cognitive/nimcp_personality_fep_bridge.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/error/nimcp_error_codes.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/validation/nimcp_common.h"
#include <string.h>
#include <math.h>

#define LOG_MODULE "personality_fep_bridge"

int personality_fep_bridge_default_config(personality_fep_config_t* config) {
    if (!config) return NIMCP_ERROR_NULL_POINTER;
    config->enable_openness_exploration = true;
    config->enable_neuroticism_precision = true;
    config->enable_conscientiousness_planning = true;
    config->personality_sensitivity = 1.0f;
    config->fep_sensitivity = 1.0f;
    return 0;
}

personality_fep_bridge_t* personality_fep_bridge_create(const personality_fep_config_t* config) {
    personality_fep_bridge_t* bridge = nimcp_malloc(sizeof(personality_fep_bridge_t));
    if (!bridge) {
        NIMCP_LOGGING_ERROR("Failed to allocate personality FEP bridge");
        return NULL;
    }
    memset(bridge, 0, sizeof(personality_fep_bridge_t));
    if (config) {
        bridge->config = *config;
    } else {
        personality_fep_bridge_default_config(&bridge->config);
    }
    bridge->base.mutex = nimcp_platform_mutex_create();
    if (!bridge->base.mutex) {
        NIMCP_LOGGING_ERROR("Failed to create mutex");
        nimcp_free(bridge);
        return NULL;
    }
    NIMCP_LOGGING_INFO("Created personality FEP bridge");
    return bridge;
}

void personality_fep_bridge_destroy(personality_fep_bridge_t* bridge) {
    if (!bridge) return;
    if (bridge->base.bio_async_enabled) {
        personality_fep_bridge_disconnect_bio_async(bridge);
    }
    if (bridge->base.mutex) {
        nimcp_mutex_destroy(bridge->base.mutex);
    }
    nimcp_free(bridge);
    NIMCP_LOGGING_INFO("Destroyed personality FEP bridge");
}

int personality_fep_bridge_connect_fep(personality_fep_bridge_t* bridge, fep_system_t* fep) {
    if (!bridge || !fep) return NIMCP_ERROR_NULL_POINTER;
    nimcp_mutex_lock(bridge->base.mutex);
    bridge->fep_system = fep;
    nimcp_mutex_unlock(bridge->base.mutex);
    NIMCP_LOGGING_INFO("Connected FEP system to personality bridge");
    return 0;
}

int personality_fep_bridge_connect_personality(personality_fep_bridge_t* bridge,
                                                personality_profile_t* personality) {
    if (!bridge || !personality) return NIMCP_ERROR_NULL_POINTER;
    nimcp_mutex_lock(bridge->base.mutex);
    bridge->personality = personality;
    nimcp_mutex_unlock(bridge->base.mutex);
    NIMCP_LOGGING_INFO("Connected personality to FEP bridge");
    return 0;
}

int personality_fep_bridge_disconnect(personality_fep_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    nimcp_mutex_lock(bridge->base.mutex);
    bridge->fep_system = NULL;
    bridge->personality = NULL;
    nimcp_mutex_unlock(bridge->base.mutex);
    NIMCP_LOGGING_INFO("Disconnected all systems from personality FEP bridge");
    return 0;
}

int personality_fep_bridge_update(personality_fep_bridge_t* bridge) {
    if (!bridge || !bridge->fep_system || !bridge->personality) return NIMCP_ERROR_INVALID_STATE;
    nimcp_mutex_lock(bridge->base.mutex);
    bridge->state.current_free_energy = fep_get_free_energy(bridge->fep_system);
    if (bridge->config.enable_openness_exploration) {
        bridge->fep_effects.openness_exploration_boost =
            (bridge->personality->traits.openness - 0.5f);
        bridge->state.exploration_rate = 0.5f + bridge->fep_effects.openness_exploration_boost;
    }
    if (bridge->config.enable_neuroticism_precision) {
        bridge->fep_effects.neuroticism_precision_modifier =
            1.0f + (bridge->personality->traits.neuroticism - 0.5f);
        bridge->state.precision_modifier = bridge->fep_effects.neuroticism_precision_modifier;
    }
    bridge->stats.avg_free_energy =
        (bridge->stats.avg_free_energy * 0.99f) + (bridge->state.current_free_energy * 0.01f);
    bridge->stats.modulation_events++;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int personality_fep_bridge_get_state(const personality_fep_bridge_t* bridge,
                                      personality_fep_state_t* state) {
    if (!bridge || !state) return NIMCP_ERROR_NULL_POINTER;
    nimcp_mutex_lock(bridge->base.mutex);
    *state = bridge->state;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int personality_fep_bridge_get_stats(const personality_fep_bridge_t* bridge,
                                      personality_fep_stats_t* stats) {
    if (!bridge || !stats) return NIMCP_ERROR_NULL_POINTER;
    nimcp_mutex_lock(bridge->base.mutex);
    *stats = bridge->stats;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int personality_fep_bridge_connect_bio_async(personality_fep_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    if (bridge->base.bio_async_enabled) return 0;
    bio_module_info_t info = {
        .module_id = BIO_MODULE_FEP_PERSONALITY_BRIDGE,
        .module_name = "personality_fep_bridge",
        .inbox_capacity = 32,
        .user_data = bridge
    };
    bridge->base.bio_ctx = bio_router_register_module(&info);
    if (bridge->base.bio_ctx) {
        bridge->base.bio_async_enabled = true;
        NIMCP_LOGGING_INFO("Connected to bio-async router");
    }
    return 0;
}

int personality_fep_bridge_disconnect_bio_async(personality_fep_bridge_t* bridge) {
    if (!bridge || !bridge->base.bio_async_enabled) return 0;
    if (bridge->base.bio_ctx) {
        bio_router_unregister_module(bridge->base.bio_ctx);
        bridge->base.bio_ctx = NULL;
    }
    bridge->base.bio_async_enabled = false;
    NIMCP_LOGGING_INFO("Disconnected from bio-async router");
    return 0;
}

bool personality_fep_bridge_is_bio_async_connected(const personality_fep_bridge_t* bridge) {
    if (!bridge) return false;
    return bridge->base.bio_async_enabled;
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

int personality_fep_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;

    const kg_entity_t* self = kg_reader_get_entity(kg, "Personality_FEP_Bridge");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            (void)self->observations[i];
        }
    }

    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Personality_FEP_Bridge");
    if (connections) {
        kg_relation_list_destroy(connections);
    }

    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Personality_FEP_Bridge");
    if (incoming) {
        kg_relation_list_destroy(incoming);
    }

    return self ? 1 : 0;
}
