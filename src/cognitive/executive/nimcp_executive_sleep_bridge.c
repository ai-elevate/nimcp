/**
 * @file nimcp_executive_sleep_bridge.c
 * @brief Sleep-Executive Function Integration Bridge Implementation
 * @version 1.0.0
 * @date 2025-12-17
 */

#include "cognitive/executive/nimcp_executive_sleep_bridge.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/platform/nimcp_platform_mutex.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>

struct executive_sleep_bridge_struct {
    bridge_base_t base;               /**< MUST be first: base bridge infrastructure */

    executive_sleep_config_t config;
    sleep_system_t sleep_system;
    executive_sleep_effects_t effects;
    bool callback_registered;  /* Track if callback is registered for cleanup */
};

/* Forward declarations */
static void executive_on_sleep_state_change(sleep_state_t new_state, void* user_data);

/**
 * WHAT: Callback invoked when sleep state changes
 * WHY:  Immediately update executive function parameters for new sleep state
 * HOW:  Called by sleep system via observer pattern
 *
 * BIOLOGICAL BASIS:
 * - Prefrontal cortex is highly sensitive to sleep deprivation
 * - Inhibitory control degrades first with sleep pressure
 * - Cognitive flexibility suffers during drowsiness
 */
static void executive_on_sleep_state_change(sleep_state_t new_state, void* user_data)
{
    executive_sleep_bridge_t bridge = (executive_sleep_bridge_t)user_data;

    if (!bridge) {
        NIMCP_LOGGING_ERROR("NULL bridge in sleep state callback");
        return;
    }

    NIMCP_LOGGING_DEBUG("Executive bridge received sleep state: %d", new_state);

    nimcp_mutex_lock(bridge->base.mutex);

    bridge->effects.current_state = new_state;

    if (bridge->config.enable_inhibition_modulation) {
        bridge->effects.inhibition_factor = executive_sleep_inhibition_for_state(new_state);
    }

    if (bridge->config.enable_flexibility_modulation) {
        bridge->effects.flexibility_factor = executive_sleep_flexibility_for_state(new_state);
    }

    if (bridge->config.enable_switch_cost_modulation) {
        bridge->effects.switch_cost_factor = executive_sleep_switch_cost_for_state(new_state);
    }

    bridge->effects.executive_offline = (new_state == SLEEP_STATE_DEEP_NREM ||
                                         new_state == SLEEP_STATE_LIGHT_NREM);

    nimcp_mutex_unlock(bridge->base.mutex);

    NIMCP_LOGGING_DEBUG("Executive modulated: inhibition=%.2f, flexibility=%.2f, offline=%d",
                        bridge->effects.inhibition_factor,
                        bridge->effects.flexibility_factor,
                        bridge->effects.executive_offline);
}

int executive_sleep_default_config(executive_sleep_config_t* config) {
    if (!config) return -1;
    config->enable_inhibition_modulation = true;
    config->enable_flexibility_modulation = true;
    config->enable_switch_cost_modulation = true;
    config->modulation_strength = 1.0f;
    return 0;
}

executive_sleep_bridge_t executive_sleep_bridge_create(
    const executive_sleep_config_t* config,
    sleep_system_t sleep)
{
    if (!sleep) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "sleep is NULL");

        return NULL;

    }

    struct executive_sleep_bridge_struct* bridge =
        (struct executive_sleep_bridge_struct*)nimcp_malloc(
            sizeof(struct executive_sleep_bridge_struct));
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return NULL;

    }

    memset(bridge, 0, sizeof(struct executive_sleep_bridge_struct));

    if (config) bridge->config = *config;
    else executive_sleep_default_config(&bridge->config);

    bridge->sleep_system = sleep;
    bridge->effects.inhibition_factor = 1.0f;
    bridge->effects.flexibility_factor = 1.0f;
    bridge->effects.switch_cost_factor = 1.0f;
    bridge->effects.executive_offline = false;

    if (bridge_base_init(&bridge->base, 0, "executive_sleep") != 0) { nimcp_free(bridge); return NULL; }
    if (!bridge->base.mutex) { nimcp_free(bridge); return NULL; }

    /* Register callback for automatic state updates */
    bridge->callback_registered = sleep_register_state_callback(
        sleep,
        executive_on_sleep_state_change,
        bridge);

    if (!bridge->callback_registered) {
        NIMCP_LOGGING_WARN("Failed to register sleep state callback - will use polling");
    } else {
        NIMCP_LOGGING_DEBUG("Registered sleep state callback for executive bridge");
    }

    /* Get initial state immediately */
    sleep_state_t initial_state = sleep_get_current_state(sleep);
    executive_on_sleep_state_change(initial_state, bridge);

    NIMCP_LOGGING_INFO("Executive-sleep bridge created");
    return bridge;
}

void executive_sleep_bridge_destroy(executive_sleep_bridge_t bridge) {
    if (!bridge) return;

    /* Unregister callback if it was registered */
    if (bridge->callback_registered && bridge->sleep_system) {
        bool unregistered = sleep_unregister_state_callback(
            bridge->sleep_system,
            executive_on_sleep_state_change,
            bridge);

        if (unregistered) {
            NIMCP_LOGGING_DEBUG("Unregistered sleep state callback for executive bridge");
        }
    }

    if (bridge->base.mutex) bridge_base_cleanup(&bridge->base);
    nimcp_free(bridge);
}

int executive_sleep_update(executive_sleep_bridge_t bridge) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->base.mutex);

    sleep_state_t state = sleep_get_current_state(bridge->sleep_system);
    float pressure = sleep_get_pressure(bridge->sleep_system);

    bridge->effects.current_state = state;
    bridge->effects.sleep_pressure = pressure;

    if (bridge->config.enable_inhibition_modulation) {
        bridge->effects.inhibition_factor = executive_sleep_inhibition_for_state(state);
        /* Sleep pressure impairs inhibition even when awake */
        if (state == SLEEP_STATE_AWAKE && pressure > 0.5f) {
            bridge->effects.inhibition_factor *= (1.0f - 0.4f * (pressure - 0.5f) / 0.5f);
        }
    }

    if (bridge->config.enable_flexibility_modulation) {
        bridge->effects.flexibility_factor = executive_sleep_flexibility_for_state(state);
    }

    if (bridge->config.enable_switch_cost_modulation) {
        bridge->effects.switch_cost_factor = executive_sleep_switch_cost_for_state(state);
    }

    bridge->effects.executive_offline = (state == SLEEP_STATE_DEEP_NREM ||
                                         state == SLEEP_STATE_LIGHT_NREM);

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int executive_sleep_get_effects(
    const executive_sleep_bridge_t bridge,
    executive_sleep_effects_t* effects)
{
    if (!bridge || !effects) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    *effects = bridge->effects;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

float executive_sleep_get_inhibition(const executive_sleep_bridge_t bridge) {
    if (!bridge) return 1.0f;
    nimcp_mutex_lock(bridge->base.mutex);
    float result = bridge->effects.inhibition_factor;
    nimcp_mutex_unlock(bridge->base.mutex);
    return result;
}

bool executive_sleep_is_offline(const executive_sleep_bridge_t bridge) {
    if (!bridge) return false;
    nimcp_mutex_lock(bridge->base.mutex);
    bool result = bridge->effects.executive_offline;
    nimcp_mutex_unlock(bridge->base.mutex);
    return result;
}

float executive_sleep_inhibition_for_state(sleep_state_t state) {
    switch (state) {
        case SLEEP_STATE_AWAKE:      return EXEC_SLEEP_INHIBITION_AWAKE;
        case SLEEP_STATE_DROWSY:     return EXEC_SLEEP_INHIBITION_DROWSY;
        case SLEEP_STATE_LIGHT_NREM: return EXEC_SLEEP_INHIBITION_LIGHT_NREM;
        case SLEEP_STATE_DEEP_NREM:  return EXEC_SLEEP_INHIBITION_DEEP_NREM;
        case SLEEP_STATE_REM:        return EXEC_SLEEP_INHIBITION_REM;
        default:                     return EXEC_SLEEP_INHIBITION_AWAKE;
    }
}

float executive_sleep_flexibility_for_state(sleep_state_t state) {
    switch (state) {
        case SLEEP_STATE_AWAKE:      return EXEC_SLEEP_FLEXIBILITY_AWAKE;
        case SLEEP_STATE_DROWSY:     return EXEC_SLEEP_FLEXIBILITY_DROWSY;
        case SLEEP_STATE_LIGHT_NREM:
        case SLEEP_STATE_DEEP_NREM:  return EXEC_SLEEP_FLEXIBILITY_NREM;
        case SLEEP_STATE_REM:        return EXEC_SLEEP_FLEXIBILITY_REM;
        default:                     return EXEC_SLEEP_FLEXIBILITY_AWAKE;
    }
}

float executive_sleep_switch_cost_for_state(sleep_state_t state) {
    switch (state) {
        case SLEEP_STATE_AWAKE:      return EXEC_SLEEP_SWITCH_COST_AWAKE;
        case SLEEP_STATE_DROWSY:     return EXEC_SLEEP_SWITCH_COST_DROWSY;
        case SLEEP_STATE_LIGHT_NREM:
        case SLEEP_STATE_DEEP_NREM:  return EXEC_SLEEP_SWITCH_COST_NREM;
        case SLEEP_STATE_REM:        return EXEC_SLEEP_SWITCH_COST_REM;
        default:                     return EXEC_SLEEP_SWITCH_COST_AWAKE;
    }
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

/**
 * WHAT: Query knowledge graph for Executive Sleep Bridge self-knowledge
 * WHY:  Enable self-awareness about module's role and connections
 * HOW:  Query KG for entity observations and relations
 */
int executive_sleep_bridge_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;
    const kg_entity_t* self = kg_reader_get_entity(kg, "Executive_Sleep_Bridge");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            NIMCP_LOGGING_DEBUG("Executive Sleep Bridge self-knowledge: %s", self->observations[i]);
        }
    }
    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Executive_Sleep_Bridge");
    if (connections) { kg_relation_list_destroy(connections); }
    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Executive_Sleep_Bridge");
    if (incoming) { kg_relation_list_destroy(incoming); }
    return self ? 1 : 0;
}
