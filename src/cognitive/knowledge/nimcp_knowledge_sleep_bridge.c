/**
 * @file nimcp_knowledge_sleep_bridge.c
 * @brief Sleep-Knowledge Integration Bridge Implementation
 * @version 1.0.0
 * @date 2025-12-17
 */

#include "cognitive/knowledge/nimcp_knowledge_sleep_bridge.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/platform/nimcp_platform_mutex.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>

struct knowledge_sleep_bridge_struct {
    bridge_base_t base;               /**< MUST be first: base bridge infrastructure */

    knowledge_sleep_config_t config;
    sleep_system_t sleep_system;
    knowledge_sleep_effects_t effects;
    bool callback_registered;
};

/* Forward declarations */
static void knowledge_on_sleep_state_change(sleep_state_t new_state, void* user_data);

/**
 * WHAT: Callback invoked when sleep state changes
 * WHY:  Immediately update knowledge parameters for new sleep state
 * HOW:  Called by sleep system via observer pattern
 *
 * BIOLOGICAL BASIS:
 * - Sleep state changes affect semantic memory access immediately
 * - Neocortex shifts from retrieval (awake) to consolidation (sleep)
 * - Sleep strengthens semantic associations
 */
static void knowledge_on_sleep_state_change(sleep_state_t new_state, void* user_data)
{
    knowledge_sleep_bridge_t bridge = (knowledge_sleep_bridge_t)user_data;

    /* Guard clause: Validate bridge */
    if (!bridge) {
        NIMCP_LOGGING_ERROR("NULL bridge in sleep state callback");
        return;
    }

    NIMCP_LOGGING_DEBUG("Knowledge bridge received sleep state: %d", new_state);

    nimcp_mutex_lock(bridge->base.mutex);

    bridge->effects.current_state = new_state;

    /* Update retrieval speed */
    if (bridge->config.enable_retrieval_modulation) {
        float retrieval_base = knowledge_sleep_retrieval_for_state(new_state);
        bridge->effects.retrieval_speed_factor = retrieval_base * bridge->config.modulation_strength +
                                                 (1.0f - bridge->config.modulation_strength);
    }

    /* Update consolidation rate */
    if (bridge->config.enable_consolidation_modulation) {
        bridge->effects.consolidation_rate_factor = knowledge_sleep_consolidation_for_state(new_state);
    }

    /* Knowledge integration peaks during REM */
    bridge->effects.integration_factor = (new_state == SLEEP_STATE_REM) ? 1.0f :
                                         (new_state == SLEEP_STATE_DEEP_NREM) ? 0.6f :
                                         (new_state == SLEEP_STATE_AWAKE) ? 0.3f : 0.1f;

    /* Association strength building during sleep */
    bridge->effects.association_strength_factor = (new_state == SLEEP_STATE_DEEP_NREM) ? 1.0f :
                                                  (new_state == SLEEP_STATE_REM) ? 0.9f :
                                                  (new_state == SLEEP_STATE_LIGHT_NREM) ? 0.7f : 0.2f;

    /* Update consolidation status */
    bridge->effects.consolidation_active = (new_state == SLEEP_STATE_DEEP_NREM ||
                                            new_state == SLEEP_STATE_LIGHT_NREM ||
                                            new_state == SLEEP_STATE_REM);

    nimcp_mutex_unlock(bridge->base.mutex);

    NIMCP_LOGGING_DEBUG("Knowledge modulated: retrieval=%.2f, consolidation=%.2f, active=%d",
                        bridge->effects.retrieval_speed_factor,
                        bridge->effects.consolidation_rate_factor,
                        bridge->effects.consolidation_active);
}

int knowledge_sleep_default_config(knowledge_sleep_config_t* config) {
    if (!config) return -1;
    config->enable_retrieval_modulation = true;
    config->enable_consolidation_modulation = true;
    config->modulation_strength = 1.0f;
    return 0;
}

knowledge_sleep_bridge_t knowledge_sleep_bridge_create(
    const knowledge_sleep_config_t* config,
    sleep_system_t sleep)
{
    if (!sleep) return NULL;

    struct knowledge_sleep_bridge_struct* bridge =
        (struct knowledge_sleep_bridge_struct*)nimcp_malloc(sizeof(struct knowledge_sleep_bridge_struct));
    if (!bridge) return NULL;

    memset(bridge, 0, sizeof(struct knowledge_sleep_bridge_struct));

    if (config) bridge->config = *config;
    else knowledge_sleep_default_config(&bridge->config);

    bridge->sleep_system = sleep;
    bridge->effects.retrieval_speed_factor = 1.0f;
    bridge->effects.consolidation_rate_factor = 0.1f;
    bridge->effects.integration_factor = 0.3f;
    bridge->effects.association_strength_factor = 0.2f;
    bridge->effects.consolidation_active = false;

    if (bridge_base_init(&bridge->base, 0, "knowledge_sleep") != 0) { nimcp_free(bridge); return NULL; }
    if (!bridge->base.mutex) { nimcp_free(bridge); return NULL; }

    /* Register callback for automatic state updates */
    bridge->callback_registered = sleep_register_state_callback(
        sleep,
        knowledge_on_sleep_state_change,
        bridge);

    if (!bridge->callback_registered) {
        NIMCP_LOGGING_WARN("Failed to register sleep state callback - will use polling");
    } else {
        NIMCP_LOGGING_DEBUG("Registered sleep state callback for knowledge bridge");
    }

    /* Get initial state immediately */
    sleep_state_t initial_state = sleep_get_current_state(sleep);
    knowledge_on_sleep_state_change(initial_state, bridge);

    NIMCP_LOGGING_INFO("Knowledge-sleep bridge created");
    return bridge;
}

void knowledge_sleep_bridge_destroy(knowledge_sleep_bridge_t bridge) {
    if (!bridge) return;

    /* Unregister callback if it was registered */
    if (bridge->callback_registered && bridge->sleep_system) {
        bool unregistered = sleep_unregister_state_callback(
            bridge->sleep_system,
            knowledge_on_sleep_state_change,
            bridge);

        if (unregistered) {
            NIMCP_LOGGING_DEBUG("Unregistered sleep state callback for knowledge bridge");
        }
    }

    if (bridge->base.mutex) bridge_base_cleanup(&bridge->base);
    nimcp_free(bridge);
}

int knowledge_sleep_update(knowledge_sleep_bridge_t bridge) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->base.mutex);

    sleep_state_t state = sleep_get_current_state(bridge->sleep_system);
    float pressure = sleep_get_pressure(bridge->sleep_system);

    bridge->effects.current_state = state;
    bridge->effects.sleep_pressure = pressure;

    if (bridge->config.enable_retrieval_modulation) {
        float retrieval_base = knowledge_sleep_retrieval_for_state(state);
        bridge->effects.retrieval_speed_factor = retrieval_base * bridge->config.modulation_strength +
                                                 (1.0f - bridge->config.modulation_strength);
        /* Sleep pressure slows retrieval */
        if (state == SLEEP_STATE_AWAKE && pressure > 0.7f) {
            bridge->effects.retrieval_speed_factor *= (1.0f - (pressure - 0.7f) * 0.6f);
        }
    }

    if (bridge->config.enable_consolidation_modulation) {
        bridge->effects.consolidation_rate_factor = knowledge_sleep_consolidation_for_state(state);
    }

    /* Knowledge integration peaks during REM */
    bridge->effects.integration_factor = (state == SLEEP_STATE_REM) ? 1.0f :
                                         (state == SLEEP_STATE_DEEP_NREM) ? 0.6f :
                                         (state == SLEEP_STATE_AWAKE) ? 0.3f : 0.1f;

    /* Association strength building during sleep */
    bridge->effects.association_strength_factor = (state == SLEEP_STATE_DEEP_NREM) ? 1.0f :
                                                  (state == SLEEP_STATE_REM) ? 0.9f :
                                                  (state == SLEEP_STATE_LIGHT_NREM) ? 0.7f : 0.2f;

    bridge->effects.consolidation_active = (state == SLEEP_STATE_DEEP_NREM ||
                                            state == SLEEP_STATE_LIGHT_NREM ||
                                            state == SLEEP_STATE_REM);

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int knowledge_sleep_get_effects(const knowledge_sleep_bridge_t bridge, knowledge_sleep_effects_t* effects) {
    if (!bridge || !effects) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    *effects = bridge->effects;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

float knowledge_sleep_get_retrieval_speed(const knowledge_sleep_bridge_t bridge) {
    if (!bridge) return 1.0f;
    nimcp_mutex_lock(bridge->base.mutex);
    float result = bridge->effects.retrieval_speed_factor;
    nimcp_mutex_unlock(bridge->base.mutex);
    return result;
}

bool knowledge_sleep_is_consolidation_active(const knowledge_sleep_bridge_t bridge) {
    if (!bridge) return false;
    nimcp_mutex_lock(bridge->base.mutex);
    bool result = bridge->effects.consolidation_active;
    nimcp_mutex_unlock(bridge->base.mutex);
    return result;
}

float knowledge_sleep_retrieval_for_state(sleep_state_t state) {
    switch (state) {
        case SLEEP_STATE_AWAKE:      return KNOWLEDGE_SLEEP_RETRIEVAL_AWAKE;
        case SLEEP_STATE_DROWSY:     return KNOWLEDGE_SLEEP_RETRIEVAL_DROWSY;
        case SLEEP_STATE_LIGHT_NREM: return KNOWLEDGE_SLEEP_RETRIEVAL_LIGHT_NREM;
        case SLEEP_STATE_DEEP_NREM:  return KNOWLEDGE_SLEEP_RETRIEVAL_DEEP_NREM;
        case SLEEP_STATE_REM:        return KNOWLEDGE_SLEEP_RETRIEVAL_REM;
        default:                     return KNOWLEDGE_SLEEP_RETRIEVAL_AWAKE;
    }
}

float knowledge_sleep_consolidation_for_state(sleep_state_t state) {
    switch (state) {
        case SLEEP_STATE_AWAKE:      return KNOWLEDGE_SLEEP_CONSOLIDATION_AWAKE;
        case SLEEP_STATE_DROWSY:     return KNOWLEDGE_SLEEP_CONSOLIDATION_DROWSY;
        case SLEEP_STATE_LIGHT_NREM: return KNOWLEDGE_SLEEP_CONSOLIDATION_LIGHT_NREM;
        case SLEEP_STATE_DEEP_NREM:  return KNOWLEDGE_SLEEP_CONSOLIDATION_DEEP_NREM;
        case SLEEP_STATE_REM:        return KNOWLEDGE_SLEEP_CONSOLIDATION_REM;
        default:                     return KNOWLEDGE_SLEEP_CONSOLIDATION_AWAKE;
    }
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

/**
 * @brief Query knowledge graph for self-knowledge about sleep-knowledge bridge
 *
 * WHAT: Retrieves entity observations and relations for sleep bridge
 * WHY: Enables self-aware introspection of module capabilities
 * HOW: Uses kg_reader to query JSONL knowledge graph
 *
 * @param kg Knowledge graph reader instance
 * @return 1 if self-knowledge found, 0 otherwise
 */
int knowledge_sleep_bridge_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;

    const kg_entity_t* self = kg_reader_get_entity(kg, "Knowledge_Sleep_Bridge");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            (void)self->observations[i];
        }
    }

    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Knowledge_Sleep_Bridge");
    if (connections) {
        kg_relation_list_destroy(connections);
    }

    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Knowledge_Sleep_Bridge");
    if (incoming) {
        kg_relation_list_destroy(incoming);
    }

    return self ? 1 : 0;
}
