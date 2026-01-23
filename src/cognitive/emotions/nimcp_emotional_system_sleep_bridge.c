/**
 * @file nimcp_emotional_system_sleep_bridge.c
 * @brief Sleep-Emotional System Integration Bridge Implementation
 * @version 1.0.0
 * @date 2025-12-17
 */

#include "cognitive/emotions/nimcp_emotional_system_sleep_bridge.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/platform/nimcp_platform_mutex.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>

struct emotional_sleep_bridge_struct {
    bridge_base_t base;               /**< MUST be first: base bridge infrastructure */

    emotional_sleep_config_t config;
    sleep_system_t sleep_system;
    emotional_sleep_effects_t effects;
    bool callback_registered;
};

/* Forward declarations */
static void emotional_on_sleep_state_change(sleep_state_t new_state, void* user_data);

/**
 * WHAT: Callback invoked when sleep state changes
 * WHY:  Immediately update emotional system parameters for new sleep state
 * HOW:  Called by sleep system via observer pattern
 *
 * BIOLOGICAL BASIS:
 * - Sleep state changes affect emotional regulation immediately
 * - Prefrontal-amygdala connectivity depends on arousal level
 * - Sleep deprivation disinhibits amygdala (60% increase in reactivity)
 */
static void emotional_on_sleep_state_change(sleep_state_t new_state, void* user_data)
{
    emotional_sleep_bridge_t bridge = (emotional_sleep_bridge_t)user_data;

    /* Guard clause: Validate bridge */
    if (!bridge) {
        NIMCP_LOGGING_ERROR("NULL bridge in sleep state callback");
        return;
    }

    NIMCP_LOGGING_DEBUG("Emotional system bridge received sleep state: %d", new_state);

    nimcp_mutex_lock(bridge->base.mutex);

    bridge->effects.current_state = new_state;

    /* Update regulation capacity */
    if (bridge->config.enable_regulation_modulation) {
        float regulation_base = emotional_sleep_regulation_for_state(new_state);
        bridge->effects.regulation_capacity_factor = regulation_base * bridge->config.modulation_strength +
                                                     (1.0f - bridge->config.modulation_strength);
    }

    /* Update emotional reactivity */
    if (bridge->config.enable_reactivity_modulation) {
        bridge->effects.emotional_reactivity_factor = emotional_sleep_reactivity_for_state(new_state);
    }

    /* Positive affect decreases with drowsiness */
    bridge->effects.positive_affect_factor = (new_state == SLEEP_STATE_AWAKE) ? 1.0f :
                                             (new_state == SLEEP_STATE_DROWSY) ? 0.7f :
                                             (new_state == SLEEP_STATE_REM) ? 0.5f : 0.3f;

    /* Negative affect increases with drowsiness */
    bridge->effects.negative_affect_factor = (new_state == SLEEP_STATE_AWAKE) ? 1.0f :
                                             (new_state == SLEEP_STATE_DROWSY) ? 1.3f : 0.5f;

    /* Emotional processing active during sleep */
    bridge->effects.emotional_processing_active = (new_state == SLEEP_STATE_DEEP_NREM ||
                                                   new_state == SLEEP_STATE_REM);

    nimcp_mutex_unlock(bridge->base.mutex);

    NIMCP_LOGGING_DEBUG("Emotional system modulated: regulation=%.2f, reactivity=%.2f, processing=%d",
                        bridge->effects.regulation_capacity_factor,
                        bridge->effects.emotional_reactivity_factor,
                        bridge->effects.emotional_processing_active);
}

int emotional_sleep_default_config(emotional_sleep_config_t* config) {
    if (!config) return -1;
    config->enable_regulation_modulation = true;
    config->enable_reactivity_modulation = true;
    config->modulation_strength = 1.0f;
    return 0;
}

emotional_sleep_bridge_t emotional_sleep_bridge_create(
    const emotional_sleep_config_t* config,
    sleep_system_t sleep)
{
    if (!sleep) return NULL;

    struct emotional_sleep_bridge_struct* bridge =
        (struct emotional_sleep_bridge_struct*)nimcp_malloc(sizeof(struct emotional_sleep_bridge_struct));
    if (!bridge) return NULL;

    memset(bridge, 0, sizeof(struct emotional_sleep_bridge_struct));

    if (config) bridge->config = *config;
    else emotional_sleep_default_config(&bridge->config);

    bridge->sleep_system = sleep;
    bridge->effects.regulation_capacity_factor = 1.0f;
    bridge->effects.emotional_reactivity_factor = 1.0f;
    bridge->effects.positive_affect_factor = 1.0f;
    bridge->effects.negative_affect_factor = 1.0f;
    bridge->effects.emotional_processing_active = false;

    if (bridge_base_init(&bridge->base, 0, "emotional_system_sleep") != 0) { nimcp_free(bridge); return NULL; }
    if (!bridge->base.mutex) { nimcp_free(bridge); return NULL; }

    /* Register callback for automatic state updates */
    bridge->callback_registered = sleep_register_state_callback(
        sleep,
        emotional_on_sleep_state_change,
        bridge);

    if (!bridge->callback_registered) {
        NIMCP_LOGGING_WARN("Failed to register sleep state callback - will use polling");
    } else {
        NIMCP_LOGGING_DEBUG("Registered sleep state callback for emotional system bridge");
    }

    /* Get initial state immediately */
    sleep_state_t initial_state = sleep_get_current_state(sleep);
    emotional_on_sleep_state_change(initial_state, bridge);

    NIMCP_LOGGING_INFO("Emotional system-sleep bridge created");
    return bridge;
}

void emotional_sleep_bridge_destroy(emotional_sleep_bridge_t bridge) {
    if (!bridge) return;

    /* Unregister callback if it was registered */
    if (bridge->callback_registered && bridge->sleep_system) {
        bool unregistered = sleep_unregister_state_callback(
            bridge->sleep_system,
            emotional_on_sleep_state_change,
            bridge);

        if (unregistered) {
            NIMCP_LOGGING_DEBUG("Unregistered sleep state callback for emotional system bridge");
        }
    }

    if (bridge->base.mutex) bridge_base_cleanup(&bridge->base);
    nimcp_free(bridge);
}

int emotional_sleep_update(emotional_sleep_bridge_t bridge) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->base.mutex);

    sleep_state_t state = sleep_get_current_state(bridge->sleep_system);
    float pressure = sleep_get_pressure(bridge->sleep_system);

    bridge->effects.current_state = state;
    bridge->effects.sleep_pressure = pressure;

    if (bridge->config.enable_regulation_modulation) {
        float regulation_base = emotional_sleep_regulation_for_state(state);
        bridge->effects.regulation_capacity_factor = regulation_base * bridge->config.modulation_strength +
                                                     (1.0f - bridge->config.modulation_strength);
        /* Sleep pressure severely impairs regulation */
        if (state == SLEEP_STATE_AWAKE && pressure > 0.7f) {
            bridge->effects.regulation_capacity_factor *= (1.0f - (pressure - 0.7f) * 0.8f);
        }
    }

    if (bridge->config.enable_reactivity_modulation) {
        float reactivity_base = emotional_sleep_reactivity_for_state(state);
        bridge->effects.emotional_reactivity_factor = reactivity_base;
        /* Sleep pressure dramatically increases reactivity */
        if (state == SLEEP_STATE_AWAKE && pressure > 0.7f) {
            bridge->effects.emotional_reactivity_factor *= (1.0f + (pressure - 0.7f) * 0.6f);
        }
    }

    /* Positive affect decreases with drowsiness */
    bridge->effects.positive_affect_factor = (state == SLEEP_STATE_AWAKE) ? 1.0f :
                                             (state == SLEEP_STATE_DROWSY) ? 0.7f :
                                             (state == SLEEP_STATE_REM) ? 0.5f : 0.3f;
    if (state == SLEEP_STATE_AWAKE && pressure > 0.7f) {
        bridge->effects.positive_affect_factor *= (1.0f - (pressure - 0.7f) * 0.5f);
    }

    /* Negative affect increases with drowsiness */
    bridge->effects.negative_affect_factor = (state == SLEEP_STATE_AWAKE) ? 1.0f :
                                             (state == SLEEP_STATE_DROWSY) ? 1.3f : 0.5f;
    if (state == SLEEP_STATE_AWAKE && pressure > 0.7f) {
        bridge->effects.negative_affect_factor *= (1.0f + (pressure - 0.7f) * 0.6f);
    }

    bridge->effects.emotional_processing_active = (state == SLEEP_STATE_DEEP_NREM ||
                                                   state == SLEEP_STATE_REM);

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int emotional_sleep_get_effects(const emotional_sleep_bridge_t bridge, emotional_sleep_effects_t* effects) {
    if (!bridge || !effects) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    *effects = bridge->effects;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

float emotional_sleep_get_regulation_capacity(const emotional_sleep_bridge_t bridge) {
    if (!bridge) return 1.0f;
    nimcp_mutex_lock(bridge->base.mutex);
    float result = bridge->effects.regulation_capacity_factor;
    nimcp_mutex_unlock(bridge->base.mutex);
    return result;
}

bool emotional_sleep_is_processing_active(const emotional_sleep_bridge_t bridge) {
    if (!bridge) return false;
    nimcp_mutex_lock(bridge->base.mutex);
    bool result = bridge->effects.emotional_processing_active;
    nimcp_mutex_unlock(bridge->base.mutex);
    return result;
}

float emotional_sleep_regulation_for_state(sleep_state_t state) {
    switch (state) {
        case SLEEP_STATE_AWAKE:      return EMOTIONAL_SLEEP_REGULATION_AWAKE;
        case SLEEP_STATE_DROWSY:     return EMOTIONAL_SLEEP_REGULATION_DROWSY;
        case SLEEP_STATE_LIGHT_NREM: return EMOTIONAL_SLEEP_REGULATION_LIGHT_NREM;
        case SLEEP_STATE_DEEP_NREM:  return EMOTIONAL_SLEEP_REGULATION_DEEP_NREM;
        case SLEEP_STATE_REM:        return EMOTIONAL_SLEEP_REGULATION_REM;
        default:                     return EMOTIONAL_SLEEP_REGULATION_AWAKE;
    }
}

float emotional_sleep_reactivity_for_state(sleep_state_t state) {
    switch (state) {
        case SLEEP_STATE_AWAKE:      return EMOTIONAL_SLEEP_REACTIVITY_AWAKE;
        case SLEEP_STATE_DROWSY:     return EMOTIONAL_SLEEP_REACTIVITY_DROWSY;
        case SLEEP_STATE_LIGHT_NREM:
        case SLEEP_STATE_DEEP_NREM:  return EMOTIONAL_SLEEP_REACTIVITY_NREM;
        case SLEEP_STATE_REM:        return EMOTIONAL_SLEEP_REACTIVITY_REM;
        default:                     return EMOTIONAL_SLEEP_REACTIVITY_AWAKE;
    }
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

int emotional_system_sleep_bridge_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;

    const kg_entity_t* self = kg_reader_get_entity(kg, "Emotional_System_Sleep_Bridge");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            (void)self->observations[i];
        }
    }

    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Emotional_System_Sleep_Bridge");
    if (connections) {
        kg_relation_list_destroy(connections);
    }

    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Emotional_System_Sleep_Bridge");
    if (incoming) {
        kg_relation_list_destroy(incoming);
    }

    return self ? 1 : 0;
}
