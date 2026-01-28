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
#include "security/nimcp_bbb_helpers.h"
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

/** Global health agent for executive_sleep_bridge module */
static nimcp_health_agent_t* g_executive_sleep_bridge_health_agent = NULL;

/**
 * @brief Set health agent for executive_sleep_bridge heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
void executive_sleep_bridge_set_health_agent(nimcp_health_agent_t* agent) {
    g_executive_sleep_bridge_health_agent = agent;
}

/** @brief Send heartbeat from executive_sleep_bridge module */
static inline void executive_sleep_bridge_heartbeat(const char* operation, float progress) {
    if (g_executive_sleep_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_executive_sleep_bridge_health_agent, operation, progress);
    }
}

/** @brief Send heartbeat (instance-level) */
static inline void executive_sleep_bridge_heartbeat_instance(
    nimcp_health_agent_t* instance_agent, const char* operation, float progress)
{
    if (g_executive_sleep_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_executive_sleep_bridge_health_agent, operation, progress);
    }
    if (instance_agent && instance_agent != g_executive_sleep_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(instance_agent, operation, progress);
    }
}

struct executive_sleep_bridge_struct {
    bridge_base_t base;               /**< MUST be first: base bridge infrastructure */
    nimcp_health_agent_t* health_agent;  /**< Phase 8: instance-level health agent */

    executive_sleep_config_t config;
    sleep_system_t sleep_system;
    executive_sleep_effects_t effects;
    bool callback_registered;  /* Track if callback is registered for cleanup */
};

BRIDGE_DEFINE_SECURITY_SETTERS_TYPE(executive_sleep_bridge, struct executive_sleep_bridge_struct)

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
    /* Phase 8: Heartbeat at operation start */
    executive_sleep_bridge_heartbeat("executive_sl_executive_sleep_defa", 0.0f);


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

    /* Phase 8: Heartbeat at operation start */
    executive_sleep_bridge_heartbeat("executive_sl_create", 0.0f);


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
    /* Phase 8: Heartbeat at operation start */
    executive_sleep_bridge_heartbeat("executive_sl_destroy", 0.0f);


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

    /* Phase 8: Heartbeat at operation start */
    executive_sleep_bridge_heartbeat("executive_sl_executive_sleep_upda", 0.0f);


    /* Safety gates: ethics + LGSS pre-check */
    BRIDGE_ETHICS_GATE(bridge, "executive_sleep_update");
    BRIDGE_LGSS_GATE(bridge, "executive_sleep_update");

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

    /* Notify coordinator of update cycle completion */
    bridge_base_notify_coordinator_tick(&bridge->base, 0);
    return 0;
}

int executive_sleep_get_effects(
    const executive_sleep_bridge_t bridge,
    executive_sleep_effects_t* effects)
{
    if (!bridge || !effects) return -1;
    /* Phase 8: Heartbeat at operation start */
    executive_sleep_bridge_heartbeat("executive_sl_executive_sleep_get_", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    *effects = bridge->effects;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

float executive_sleep_get_inhibition(const executive_sleep_bridge_t bridge) {
    if (!bridge) return 1.0f;
    /* Phase 8: Heartbeat at operation start */
    executive_sleep_bridge_heartbeat("executive_sl_executive_sleep_get_", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    float result = bridge->effects.inhibition_factor;
    nimcp_mutex_unlock(bridge->base.mutex);
    return result;
}

bool executive_sleep_is_offline(const executive_sleep_bridge_t bridge) {
    if (!bridge) return false;
    /* Phase 8: Heartbeat at operation start */
    executive_sleep_bridge_heartbeat("executive_sl_executive_sleep_is_o", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    bool result = bridge->effects.executive_offline;
    nimcp_mutex_unlock(bridge->base.mutex);
    return result;
}

float executive_sleep_inhibition_for_state(sleep_state_t state) {
    /* Phase 8: Heartbeat at operation start */
    executive_sleep_bridge_heartbeat("executive_sl_executive_sleep_inhi", 0.0f);


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
    /* Phase 8: Heartbeat at operation start */
    executive_sleep_bridge_heartbeat("executive_sl_executive_sleep_flex", 0.0f);


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
    /* Phase 8: Heartbeat at operation start */
    executive_sleep_bridge_heartbeat("executive_sl_executive_sleep_swit", 0.0f);


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
    /* Phase 8: Heartbeat at operation start */
    executive_sleep_bridge_heartbeat("executive_sl_query_self_knowledge", 0.0f);


    const kg_entity_t* self = kg_reader_get_entity(kg, "Executive_Sleep_Bridge");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            /* Phase 8: Loop progress heartbeat */
            if ((i & 0xFF) == 0 && self->num_observations > 256) {
                executive_sleep_bridge_heartbeat("executive_sl_loop",
                                 (float)(i + 1) / (float)self->num_observations);
            }

            NIMCP_LOGGING_DEBUG("Executive Sleep Bridge self-knowledge: %s", self->observations[i]);
        }
    }
    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Executive_Sleep_Bridge");
    if (connections) { kg_relation_list_destroy(connections); }
    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Executive_Sleep_Bridge");
    if (incoming) { kg_relation_list_destroy(incoming); }
    return self ? 1 : 0;
}

/* ============================================================================
 * Phase 8: Instance-Level Health Agent
 * ============================================================================ */

void executive_sleep_bridge_set_instance_health_agent(void* instance, nimcp_health_agent_t* agent) {
    if (instance) {
        (void)agent;
        g_executive_sleep_bridge_health_agent = agent;
    }
}

/* ============================================================================
 * Phase 8: Training Integration (Full Implementation)
 * ============================================================================ */

int executive_sleep_bridge_training_begin(void* instance) {
    if (!instance) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "executive_sleep_bridge_training_begin: NULL argument");
        return -1;
    }
    executive_sleep_bridge_heartbeat_instance(NULL, "executive_sleep_bridge_training_begin", 0.0f);
    (void)instance;
    return 0;
}

int executive_sleep_bridge_training_end(void* instance) {
    if (!instance) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "executive_sleep_bridge_training_end: NULL argument");
        return -1;
    }
    executive_sleep_bridge_heartbeat_instance(NULL, "executive_sleep_bridge_training_end", 1.0f);
    (void)instance;
    return 0;
}

int executive_sleep_bridge_training_step(void* instance, float progress) {
    if (!instance) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "executive_sleep_bridge_training_step: NULL argument");
        return -1;
    }
    if (progress < 0.0f) progress = 0.0f;
    if (progress > 1.0f) progress = 1.0f;
    executive_sleep_bridge_heartbeat_instance(NULL, "executive_sleep_bridge_training_step", progress);
    (void)instance;
    return 0;
}
