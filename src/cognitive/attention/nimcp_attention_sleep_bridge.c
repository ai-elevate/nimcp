/**
 * @file nimcp_attention_sleep_bridge.c
 * @brief Sleep-Attention Integration Bridge Implementation
 * @version 1.0.0
 * @date 2025-12-17
 */

#include "cognitive/attention/nimcp_attention_sleep_bridge.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/platform/nimcp_platform_mutex.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>

struct attention_sleep_bridge_struct {
    bridge_base_t base;               /**< MUST be first: base bridge infrastructure */

    attention_sleep_config_t config;
    sleep_system_t sleep_system;
    attention_sleep_effects_t effects;
    bool callback_registered;  /* Track if callback is registered for cleanup */
};

/* Forward declarations */
static void attention_on_sleep_state_change(sleep_state_t new_state, void* user_data);

/**
 * WHAT: Callback invoked when sleep state changes
 * WHY:  Immediately update attention parameters for new sleep state
 * HOW:  Called by sleep system via observer pattern
 *
 * BIOLOGICAL BASIS:
 * - Sleep state changes affect attention capacity immediately
 * - Neurotransmitter levels shift with sleep state transitions
 * - Immediate modulation reflects neural reality
 */
static void attention_on_sleep_state_change(sleep_state_t new_state, void* user_data)
{
    attention_sleep_bridge_t bridge = (attention_sleep_bridge_t)user_data;

    /* Guard clause: Validate bridge */
    if (!bridge) {
        NIMCP_LOGGING_ERROR("NULL bridge in sleep state callback");
        return;
    }

    NIMCP_LOGGING_DEBUG("Attention bridge received sleep state: %d", new_state);

    nimcp_mutex_lock(bridge->base.mutex);

    bridge->effects.current_state = new_state;

    /* Update capacity factor */
    if (bridge->config.enable_capacity_modulation) {
        float cap_base = attention_sleep_capacity_for_state(new_state);
        bridge->effects.capacity_factor = cap_base * bridge->config.modulation_strength +
                                          (1.0f - bridge->config.modulation_strength);
    }

    /* Update vigilance factor */
    if (bridge->config.enable_vigilance_modulation) {
        bridge->effects.vigilance_factor = attention_sleep_vigilance_for_state(new_state);
    }

    /* Spotlight narrows with drowsiness, expands in REM */
    bridge->effects.spotlight_size_factor = (new_state == SLEEP_STATE_DROWSY) ? 0.7f :
                                            (new_state == SLEEP_STATE_REM) ? 1.3f : 1.0f;

    /* Update offline status */
    bridge->effects.attention_offline = (new_state == SLEEP_STATE_DEEP_NREM ||
                                         new_state == SLEEP_STATE_LIGHT_NREM);

    nimcp_mutex_unlock(bridge->base.mutex);

    NIMCP_LOGGING_DEBUG("Attention modulated: capacity=%.2f, vigilance=%.2f, offline=%d",
                        bridge->effects.capacity_factor,
                        bridge->effects.vigilance_factor,
                        bridge->effects.attention_offline);
}

int attention_sleep_default_config(attention_sleep_config_t* config) {
    if (!config) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "attention_sleep_default_config: config is NULL");
        return -1;
    }
    config->enable_capacity_modulation = true;
    config->enable_vigilance_modulation = true;
    config->modulation_strength = 1.0f;
    return 0;
}

attention_sleep_bridge_t attention_sleep_bridge_create(
    const attention_sleep_config_t* config,
    sleep_system_t sleep)
{
    if (!sleep) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "attention_sleep_bridge_create: sleep is NULL");
        return NULL;
    }

    struct attention_sleep_bridge_struct* bridge =
        (struct attention_sleep_bridge_struct*)nimcp_malloc(sizeof(struct attention_sleep_bridge_struct));
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "attention_sleep_bridge_create: failed to allocate bridge");
        return NULL;
    }

    memset(bridge, 0, sizeof(struct attention_sleep_bridge_struct));

    if (config) bridge->config = *config;
    else attention_sleep_default_config(&bridge->config);

    bridge->sleep_system = sleep;
    bridge->effects.capacity_factor = 1.0f;
    bridge->effects.vigilance_factor = 1.0f;
    bridge->effects.spotlight_size_factor = 1.0f;
    bridge->effects.attention_offline = false;

    if (bridge_base_init(&bridge->base, 0, "attention_sleep") != 0) { nimcp_free(bridge); return NULL; }
    if (!bridge->base.mutex) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_OPERATION_FAILED, "attention_sleep_bridge_create: failed to create mutex");
        nimcp_free(bridge);
        return NULL;
    }

    /* Register callback for automatic state updates */
    bridge->callback_registered = sleep_register_state_callback(
        sleep,
        attention_on_sleep_state_change,
        bridge);

    if (!bridge->callback_registered) {
        NIMCP_LOGGING_WARN("Failed to register sleep state callback - will use polling");
    } else {
        NIMCP_LOGGING_DEBUG("Registered sleep state callback for attention bridge");
    }

    /* Get initial state immediately */
    sleep_state_t initial_state = sleep_get_current_state(sleep);
    attention_on_sleep_state_change(initial_state, bridge);

    NIMCP_LOGGING_INFO("Attention-sleep bridge created");
    return bridge;
}

void attention_sleep_bridge_destroy(attention_sleep_bridge_t bridge) {
    if (!bridge) return;

    /* Unregister callback if it was registered */
    if (bridge->callback_registered && bridge->sleep_system) {
        bool unregistered = sleep_unregister_state_callback(
            bridge->sleep_system,
            attention_on_sleep_state_change,
            bridge);

        if (unregistered) {
            NIMCP_LOGGING_DEBUG("Unregistered sleep state callback for attention bridge");
        }
    }

    if (bridge->base.mutex) bridge_base_cleanup(&bridge->base);
    nimcp_free(bridge);
}

int attention_sleep_update(attention_sleep_bridge_t bridge) {
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "attention_sleep_update: bridge is NULL");
        return -1;
    }

    nimcp_mutex_lock(bridge->base.mutex);

    sleep_state_t state = sleep_get_current_state(bridge->sleep_system);
    float pressure = sleep_get_pressure(bridge->sleep_system);

    bridge->effects.current_state = state;
    bridge->effects.sleep_pressure = pressure;

    if (bridge->config.enable_capacity_modulation) {
        float cap_base = attention_sleep_capacity_for_state(state);
        bridge->effects.capacity_factor = cap_base * bridge->config.modulation_strength +
                                          (1.0f - bridge->config.modulation_strength);
        /* Sleep pressure further reduces capacity */
        if (state == SLEEP_STATE_AWAKE && pressure > 0.7f) {
            bridge->effects.capacity_factor *= (1.0f - (pressure - 0.7f));
        }
    }

    if (bridge->config.enable_vigilance_modulation) {
        bridge->effects.vigilance_factor = attention_sleep_vigilance_for_state(state);
    }

    /* Spotlight narrows with drowsiness, expands in REM */
    bridge->effects.spotlight_size_factor = (state == SLEEP_STATE_DROWSY) ? 0.7f :
                                            (state == SLEEP_STATE_REM) ? 1.3f : 1.0f;

    bridge->effects.attention_offline = (state == SLEEP_STATE_DEEP_NREM ||
                                         state == SLEEP_STATE_LIGHT_NREM);

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int attention_sleep_get_effects(const attention_sleep_bridge_t bridge, attention_sleep_effects_t* effects) {
    if (!bridge || !effects) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "attention_sleep_get_effects: bridge or effects is NULL");
        return -1;
    }
    nimcp_mutex_lock(bridge->base.mutex);
    *effects = bridge->effects;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

float attention_sleep_get_capacity(const attention_sleep_bridge_t bridge) {
    if (!bridge) return 1.0f;
    nimcp_mutex_lock(bridge->base.mutex);
    float result = bridge->effects.capacity_factor;
    nimcp_mutex_unlock(bridge->base.mutex);
    return result;
}

bool attention_sleep_is_offline(const attention_sleep_bridge_t bridge) {
    if (!bridge) return false;
    nimcp_mutex_lock(bridge->base.mutex);
    bool result = bridge->effects.attention_offline;
    nimcp_mutex_unlock(bridge->base.mutex);
    return result;
}

float attention_sleep_capacity_for_state(sleep_state_t state) {
    switch (state) {
        case SLEEP_STATE_AWAKE:      return ATTN_SLEEP_CAPACITY_AWAKE;
        case SLEEP_STATE_DROWSY:     return ATTN_SLEEP_CAPACITY_DROWSY;
        case SLEEP_STATE_LIGHT_NREM: return ATTN_SLEEP_CAPACITY_LIGHT_NREM;
        case SLEEP_STATE_DEEP_NREM:  return ATTN_SLEEP_CAPACITY_DEEP_NREM;
        case SLEEP_STATE_REM:        return ATTN_SLEEP_CAPACITY_REM;
        default:                     return ATTN_SLEEP_CAPACITY_AWAKE;
    }
}

float attention_sleep_vigilance_for_state(sleep_state_t state) {
    switch (state) {
        case SLEEP_STATE_AWAKE:      return ATTN_SLEEP_VIGILANCE_AWAKE;
        case SLEEP_STATE_DROWSY:     return ATTN_SLEEP_VIGILANCE_DROWSY;
        case SLEEP_STATE_LIGHT_NREM:
        case SLEEP_STATE_DEEP_NREM:  return ATTN_SLEEP_VIGILANCE_NREM;
        case SLEEP_STATE_REM:        return ATTN_SLEEP_VIGILANCE_REM;
        default:                     return ATTN_SLEEP_VIGILANCE_AWAKE;
    }
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

/**
 * WHAT: Query knowledge graph for Attention Sleep Bridge self-knowledge
 * WHY:  Enable self-awareness about module's role and connections
 * HOW:  Query KG for entity observations and relations
 */
int attention_sleep_bridge_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;
    const kg_entity_t* self = kg_reader_get_entity(kg, "Attention_Sleep_Bridge");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            NIMCP_LOGGING_DEBUG("Attention Sleep Bridge self-knowledge: %s", self->observations[i]);
        }
    }
    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Attention_Sleep_Bridge");
    if (connections) { kg_relation_list_destroy(connections); }
    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Attention_Sleep_Bridge");
    if (incoming) { kg_relation_list_destroy(incoming); }
    return self ? 1 : 0;
}
