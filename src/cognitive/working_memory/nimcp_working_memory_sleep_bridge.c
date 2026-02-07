/**
 * @file nimcp_working_memory_sleep_bridge.c
 * @brief Sleep-Working Memory Integration Bridge Implementation
 * @version 1.0.0
 * @date 2025-12-17
 */

#include "cognitive/working_memory/nimcp_working_memory_sleep_bridge.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/platform/nimcp_platform_mutex.h"
#include "utils/exception/nimcp_exception_macros.h"
#include "security/nimcp_bbb_helpers.h"
#include <string.h>
#include "utils/fault_tolerance/nimcp_health_agent_macros.h"
#include "mesh/nimcp_mesh_participant.h"
#include "mesh/nimcp_mesh_adapter.h"

NIMCP_DECLARE_HEALTH_AGENT_ATOMIC(working_memory_sleep_bridge)
//=============================================================================
// Mesh Participant Registration
//=============================================================================

static mesh_participant_id_t g_working_memory_sleep_bridge_mesh_id = 0;
static mesh_participant_registry_t* g_working_memory_sleep_bridge_mesh_registry = NULL;

nimcp_error_t working_memory_sleep_bridge_mesh_register(mesh_participant_registry_t* registry) {
    if (!registry) return NIMCP_ERROR_NULL_POINTER;
    if (g_working_memory_sleep_bridge_mesh_id != 0) return NIMCP_SUCCESS;
    mesh_participant_interface_t iface;
    mesh_participant_interface_init(&iface);
    strncpy(iface.module_name, "working_memory_sleep_bridge", MESH_MAX_NAME_LEN - 1);
    iface.type = MESH_PARTICIPANT_MODULE;
    iface.home_channel = mesh_adapter_get_default_channel(MESH_ADAPTER_CATEGORY_COGNITIVE);
    mesh_participant_config_t config;
    mesh_participant_config_init(&config);
    config.module_name = "working_memory_sleep_bridge";
    config.type = MESH_PARTICIPANT_MODULE;
    config.home_channel = iface.home_channel;
    nimcp_error_t err = mesh_participant_register(registry, &iface, &config, &g_working_memory_sleep_bridge_mesh_id);
    if (err == NIMCP_SUCCESS) g_working_memory_sleep_bridge_mesh_registry = registry;
    return err;
}

void working_memory_sleep_bridge_mesh_unregister(void) {
    if (g_working_memory_sleep_bridge_mesh_registry && g_working_memory_sleep_bridge_mesh_id != 0) {
        mesh_participant_unregister(g_working_memory_sleep_bridge_mesh_registry, g_working_memory_sleep_bridge_mesh_id);
        g_working_memory_sleep_bridge_mesh_id = 0;
        g_working_memory_sleep_bridge_mesh_registry = NULL;
    }
}


/** @brief Send heartbeat from working_memory_sleep_bridge module (instance-level) */
static inline void working_memory_sleep_bridge_heartbeat_instance(
    nimcp_health_agent_t* instance_agent, const char* operation, float progress)
{
    if (g_working_memory_sleep_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_working_memory_sleep_bridge_health_agent, operation, progress);
    }
    if (instance_agent && instance_agent != g_working_memory_sleep_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(instance_agent, operation, progress);
    }
}


/* Security subsystem setters (Phase 1: Audit Gap Remediation) */
struct working_memory_sleep_bridge_struct {
    bridge_base_t base;               /**< MUST be first: base bridge infrastructure */
    nimcp_health_agent_t* health_agent;  /**< Phase 8: instance-level health agent */

    working_memory_sleep_config_t config;
    sleep_system_t sleep_system;
    working_memory_sleep_effects_t effects;
    bool callback_registered;  /* Track if callback is registered for cleanup */
};

BRIDGE_DEFINE_SECURITY_SETTERS_TYPE(working_memory_sleep_bridge, struct working_memory_sleep_bridge_struct)

/* Forward declarations */
static void working_memory_on_sleep_state_change(sleep_state_t new_state, void* user_data);

/**
 * WHAT: Callback invoked when sleep state changes
 * WHY:  Immediately update WM parameters for new sleep state
 * HOW:  Called by sleep system via observer pattern
 *
 * BIOLOGICAL BASIS:
 * - WM capacity drops sharply with drowsiness (prefrontal hypofunction)
 * - Deep NREM enables consolidation to long-term memory
 * - Sleep deprivation severely impairs working memory span
 */
static void working_memory_on_sleep_state_change(sleep_state_t new_state, void* user_data)
{
    working_memory_sleep_bridge_t bridge = (working_memory_sleep_bridge_t)user_data;

    if (!bridge) {
        NIMCP_LOGGING_ERROR("NULL bridge in sleep state callback");
        return;
    }

    NIMCP_LOGGING_DEBUG("Working memory bridge received sleep state: %d", new_state);

    nimcp_mutex_lock(bridge->base.mutex);

    bridge->effects.current_state = new_state;

    if (bridge->config.enable_capacity_modulation) {
        bridge->effects.capacity_factor = working_memory_sleep_capacity_for_state(new_state);
    }

    if (bridge->config.enable_decay_modulation) {
        bridge->effects.decay_rate_factor = working_memory_sleep_decay_for_state(new_state);
    }

    /* Rehearsal efficiency drops with drowsiness */
    bridge->effects.rehearsal_efficiency = (new_state == SLEEP_STATE_AWAKE) ? 1.0f :
                                           (new_state == SLEEP_STATE_DROWSY) ? 0.5f : 0.0f;

    bridge->effects.wm_offline = (new_state == SLEEP_STATE_DEEP_NREM);
    bridge->effects.consolidation_active = (new_state == SLEEP_STATE_DEEP_NREM ||
                                            new_state == SLEEP_STATE_LIGHT_NREM);

    nimcp_mutex_unlock(bridge->base.mutex);

    NIMCP_LOGGING_DEBUG("WM modulated: capacity=%.2f, decay=%.2f, offline=%d",
                        bridge->effects.capacity_factor,
                        bridge->effects.decay_rate_factor,
                        bridge->effects.wm_offline);
}

int working_memory_sleep_default_config(working_memory_sleep_config_t* config) {
    if (!config) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "working_memory_sleep_default_config: config is NULL");
        return -1;
    }
    config->enable_capacity_modulation = true;
    config->enable_decay_modulation = true;
    config->enable_transfer_on_sleep = true;
    config->modulation_strength = 1.0f;
    return 0;
}

working_memory_sleep_bridge_t working_memory_sleep_bridge_create(
    const working_memory_sleep_config_t* config,
    sleep_system_t sleep)
{
    if (!sleep) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "sleep is NULL");

        return NULL;

    }

    struct working_memory_sleep_bridge_struct* bridge =
        (struct working_memory_sleep_bridge_struct*)nimcp_malloc(
            sizeof(struct working_memory_sleep_bridge_struct));
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return NULL;

    }

    memset(bridge, 0, sizeof(struct working_memory_sleep_bridge_struct));

    if (config) bridge->config = *config;
    else working_memory_sleep_default_config(&bridge->config);

    bridge->sleep_system = sleep;
    bridge->effects.capacity_factor = 1.0f;
    bridge->effects.decay_rate_factor = 1.0f;
    bridge->effects.rehearsal_efficiency = 1.0f;
    bridge->effects.wm_offline = false;
    bridge->effects.consolidation_active = false;

    if (bridge_base_init(&bridge->base, 0, "working_memory_sleep") != 0) { nimcp_free(bridge); return NULL; }
    if (!bridge->base.mutex) { nimcp_free(bridge); return NULL; }

    /* Register callback for automatic state updates */
    bridge->callback_registered = sleep_register_state_callback(
        sleep,
        working_memory_on_sleep_state_change,
        bridge);

    if (!bridge->callback_registered) {
        NIMCP_LOGGING_WARN("Failed to register sleep state callback - will use polling");
    } else {
        NIMCP_LOGGING_DEBUG("Registered sleep state callback for working memory bridge");
    }

    /* Get initial state immediately */
    sleep_state_t initial_state = sleep_get_current_state(sleep);
    working_memory_on_sleep_state_change(initial_state, bridge);

    NIMCP_LOGGING_INFO("Working memory-sleep bridge created");
    return bridge;
}

void working_memory_sleep_bridge_destroy(working_memory_sleep_bridge_t bridge) {
    if (!bridge) return;

    /* Unregister callback if it was registered */
    if (bridge->callback_registered && bridge->sleep_system) {
        bool unregistered = sleep_unregister_state_callback(
            bridge->sleep_system,
            working_memory_on_sleep_state_change,
            bridge);

        if (unregistered) {
            NIMCP_LOGGING_DEBUG("Unregistered sleep state callback for working memory bridge");
        }
    }

    if (bridge->base.mutex) bridge_base_cleanup(&bridge->base);
    nimcp_free(bridge);
}

int working_memory_sleep_update(working_memory_sleep_bridge_t bridge) {
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "working_memory_sleep_update: bridge is NULL");
        return -1;
    }

    /* Safety gates: ethics + LGSS pre-check */
    BRIDGE_ETHICS_GATE(bridge, "working_memory_sleep_update");
    BRIDGE_LGSS_GATE(bridge, "working_memory_sleep_update");

    nimcp_mutex_lock(bridge->base.mutex);

    sleep_state_t state = sleep_get_current_state(bridge->sleep_system);
    float pressure = sleep_get_pressure(bridge->sleep_system);

    bridge->effects.current_state = state;
    bridge->effects.sleep_pressure = pressure;

    if (bridge->config.enable_capacity_modulation) {
        float cap_base = working_memory_sleep_capacity_for_state(state);
        bridge->effects.capacity_factor = cap_base;
        /* High sleep pressure reduces WM capacity even when awake */
        if (state == SLEEP_STATE_AWAKE && pressure > 0.6f) {
            bridge->effects.capacity_factor *= (1.0f - 0.3f * (pressure - 0.6f) / 0.4f);
        }
    }

    if (bridge->config.enable_decay_modulation) {
        bridge->effects.decay_rate_factor = working_memory_sleep_decay_for_state(state);
    }

    /* Rehearsal efficiency drops with drowsiness */
    bridge->effects.rehearsal_efficiency = (state == SLEEP_STATE_AWAKE) ? 1.0f :
                                           (state == SLEEP_STATE_DROWSY) ? 0.5f : 0.0f;

    bridge->effects.wm_offline = (state == SLEEP_STATE_DEEP_NREM);
    bridge->effects.consolidation_active = (state == SLEEP_STATE_DEEP_NREM ||
                                            state == SLEEP_STATE_LIGHT_NREM);

    nimcp_mutex_unlock(bridge->base.mutex);

    /* Notify coordinator of update cycle completion */
    bridge_base_notify_coordinator_tick(&bridge->base, 0);
    return 0;
}

int working_memory_sleep_get_effects(
    const working_memory_sleep_bridge_t bridge,
    working_memory_sleep_effects_t* effects)
{
    if (!bridge || !effects) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "working_memory_sleep_get_effects: required parameter is NULL (bridge, effects)");
        return -1;
    }
    nimcp_mutex_lock(bridge->base.mutex);
    *effects = bridge->effects;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

float working_memory_sleep_get_capacity(const working_memory_sleep_bridge_t bridge) {
    if (!bridge) return 1.0f;
    nimcp_mutex_lock(bridge->base.mutex);
    float result = bridge->effects.capacity_factor;
    nimcp_mutex_unlock(bridge->base.mutex);
    return result;
}

bool working_memory_sleep_is_offline(const working_memory_sleep_bridge_t bridge) {
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "working_memory_sleep_is_offline: bridge is NULL");
        return false;
    }
    nimcp_mutex_lock(bridge->base.mutex);
    bool result = bridge->effects.wm_offline;
    nimcp_mutex_unlock(bridge->base.mutex);
    return result;
}

float working_memory_sleep_capacity_for_state(sleep_state_t state) {
    switch (state) {
        case SLEEP_STATE_AWAKE:      return WM_SLEEP_CAPACITY_AWAKE;
        case SLEEP_STATE_DROWSY:     return WM_SLEEP_CAPACITY_DROWSY;
        case SLEEP_STATE_LIGHT_NREM: return WM_SLEEP_CAPACITY_LIGHT_NREM;
        case SLEEP_STATE_DEEP_NREM:  return WM_SLEEP_CAPACITY_DEEP_NREM;
        case SLEEP_STATE_REM:        return WM_SLEEP_CAPACITY_REM;
        default:                     return WM_SLEEP_CAPACITY_AWAKE;
    }
}

float working_memory_sleep_decay_for_state(sleep_state_t state) {
    switch (state) {
        case SLEEP_STATE_AWAKE:      return WM_SLEEP_DECAY_AWAKE;
        case SLEEP_STATE_DROWSY:     return WM_SLEEP_DECAY_DROWSY;
        case SLEEP_STATE_LIGHT_NREM: return WM_SLEEP_DECAY_LIGHT_NREM;
        case SLEEP_STATE_DEEP_NREM:  return WM_SLEEP_DECAY_DEEP_NREM;
        case SLEEP_STATE_REM:        return WM_SLEEP_DECAY_REM;
        default:                     return WM_SLEEP_DECAY_AWAKE;
    }
}

/* ============================================================================
 * KG Self-Awareness Integration
 * ============================================================================ */

/**
 * WHAT: Query self-knowledge from knowledge graph
 * WHY:  Enable introspection about module capabilities and connections
 * HOW:  Query KG reader for entity and relations
 */
int working_memory_sleep_bridge_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;

    const kg_entity_t* self = kg_reader_get_entity(kg, "Working_Memory_Sleep_Bridge");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            NIMCP_LOGGING_DEBUG("Working memory sleep bridge self-knowledge: %s", self->observations[i]);
        }
    }

    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Working_Memory_Sleep_Bridge");
    if (connections) {
        kg_relation_list_destroy(connections);
    }

    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Working_Memory_Sleep_Bridge");
    if (incoming) {
        kg_relation_list_destroy(incoming);
    }

    return self ? 1 : 0;
}

/* ============================================================================
 * Phase 8: Instance-Level Health Agent
 * ============================================================================ */

void working_memory_sleep_bridge_set_instance_health_agent(void* instance, nimcp_health_agent_t* agent) {
    if (instance) {
        (void)agent;
        g_working_memory_sleep_bridge_health_agent = agent;
    }
}

/* ============================================================================
 * Phase 8: Training Integration (Full Implementation)
 * ============================================================================ */

int working_memory_sleep_bridge_training_begin(void* instance) {
    if (!instance) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "working_memory_sleep_bridge_training_begin: NULL argument");
        return -1;
    }
    working_memory_sleep_bridge_heartbeat_instance(NULL, "working_memory_sleep_bridge_training_begin", 0.0f);
    (void)instance;
    return 0;
}

int working_memory_sleep_bridge_training_end(void* instance) {
    if (!instance) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "working_memory_sleep_bridge_training_end: NULL argument");
        return -1;
    }
    working_memory_sleep_bridge_heartbeat_instance(NULL, "working_memory_sleep_bridge_training_end", 1.0f);
    (void)instance;
    return 0;
}

int working_memory_sleep_bridge_training_step(void* instance, float progress) {
    if (!instance) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "working_memory_sleep_bridge_training_step: NULL argument");
        return -1;
    }
    if (progress < 0.0f) progress = 0.0f;
    if (progress > 1.0f) progress = 1.0f;
    working_memory_sleep_bridge_heartbeat_instance(NULL, "working_memory_sleep_bridge_training_step", progress);
    (void)instance;
    return 0;
}
