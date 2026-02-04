/**
 * @file nimcp_symbolic_logic_attachment.c
 * @brief MODULE 1: Symbolic Logic Attachment implementation
 *
 * SINGLE RESPONSIBILITY: Attach/detach symbolic logic engines to/from brains
 *
 * @author NIMCP Development Team - SRP Refactoring
 * @date 2025-11-20
 * @version 3.0.0
 */

#include "cognitive/reasoning/nimcp_symbolic_logic_attachment.h"
#include "security/nimcp_security.h"
#include "security/nimcp_blood_brain_barrier.h"

#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_router.h"

#include "core/brain/nimcp_brain_internal.h"
#include "utils/validation/nimcp_validate.h"
#include "utils/logging/nimcp_logging.h"
#include "core/events/nimcp_event_bus.h"
#include "api/nimcp_api_exception.h"
#include "utils/exception/nimcp_exception_macros.h"

#include <string.h>
#include <stdio.h>
#include <stdarg.h>
#include "utils/fault_tolerance/nimcp_health_agent_macros.h"
#include "mesh/nimcp_mesh_participant.h"
#include "mesh/nimcp_mesh_adapter.h"

NIMCP_DECLARE_HEALTH_AGENT_ATOMIC(symbolic_logic_attachment)
//=============================================================================
// Mesh Participant Registration
//=============================================================================

static mesh_participant_id_t g_symbolic_logic_attachment_mesh_id = 0;
static mesh_participant_registry_t* g_symbolic_logic_attachment_mesh_registry = NULL;

nimcp_error_t symbolic_logic_attachment_mesh_register(mesh_participant_registry_t* registry) {
    if (!registry) return NIMCP_ERROR_NULL_POINTER;
    if (g_symbolic_logic_attachment_mesh_id != 0) return NIMCP_SUCCESS;
    mesh_participant_interface_t iface;
    mesh_participant_interface_init(&iface);
    strncpy(iface.module_name, "symbolic_logic_attachment", MESH_MAX_NAME_LEN - 1);
    iface.type = MESH_PARTICIPANT_MODULE;
    iface.home_channel = mesh_adapter_get_default_channel(MESH_ADAPTER_CATEGORY_COGNITIVE);
    mesh_participant_config_t config;
    mesh_participant_config_init(&config);
    config.module_name = "symbolic_logic_attachment";
    config.type = MESH_PARTICIPANT_MODULE;
    config.home_channel = iface.home_channel;
    nimcp_error_t err = mesh_participant_register(registry, &iface, &config, &g_symbolic_logic_attachment_mesh_id);
    if (err == NIMCP_SUCCESS) g_symbolic_logic_attachment_mesh_registry = registry;
    return err;
}

void symbolic_logic_attachment_mesh_unregister(void) {
    if (g_symbolic_logic_attachment_mesh_registry && g_symbolic_logic_attachment_mesh_id != 0) {
        mesh_participant_unregister(g_symbolic_logic_attachment_mesh_registry, g_symbolic_logic_attachment_mesh_id);
        g_symbolic_logic_attachment_mesh_id = 0;
        g_symbolic_logic_attachment_mesh_registry = NULL;
    }
}


/** @brief Send heartbeat from symbolic_logic_attachment module (instance-level) */
static inline void symbolic_logic_attachment_heartbeat_instance(
    nimcp_health_agent_t* instance_agent, const char* operation, float progress)
{
    if (g_symbolic_logic_attachment_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_symbolic_logic_attachment_health_agent, operation, progress);
    }
    if (instance_agent && instance_agent != g_symbolic_logic_attachment_health_agent) {
        nimcp_health_agent_heartbeat_ex(instance_agent, operation, progress);
    }
}



//=============================================================================
// Event Types
//=============================================================================

#define EVENT_LOGIC_ENGINE_ATTACHED 0x0940
#define EVENT_LOGIC_ENGINE_DETACHED 0x0941

//=============================================================================
// Error Handling
//=============================================================================

static __thread char last_error[256] = {0};

static void set_error(const char* fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    vsnprintf(last_error, sizeof(last_error), fmt, args);
    va_end(args);
}

const char* brain_logic_attachment_get_last_error(void)
{
    return last_error;
}

//=============================================================================
// Attachment API Implementation
//=============================================================================

bool brain_attach_symbolic_logic(
    brain_t brain,
    symbolic_logic_t* logic_engine)
{
    // Validate inputs
    if (!nimcp_validate_pointer(brain, "brain")) {
        set_error("Brain is NULL");
        NIMCP_LOGGING_ERROR("brain_attach_symbolic_logic: brain is NULL");
        return false;
    }

    if (!nimcp_validate_pointer(logic_engine, "logic_engine")) {
        set_error("Logic engine is NULL");
        NIMCP_LOGGING_ERROR("brain_attach_symbolic_logic: logic_engine is NULL");
        return false;
    }

    // Check if already attached
    if (brain->symbolic_logic != NULL) {
        set_error("Symbolic logic engine already attached to this brain");
        NIMCP_LOGGING_WARN("brain_attach_symbolic_logic: brain %p already has logic engine",
                              (void*)brain);
        return false;
    }

    // Attach the engine
    /* Phase 8: Heartbeat at operation start */
    symbolic_logic_attachment_heartbeat("symbolic_log_brain_attach_symboli", 0.0f);


    brain->symbolic_logic = logic_engine;

    // Publish attachment event
    // event_data_t event = {
    //     .event_type = EVENT_LOGIC_ENGINE_ATTACHED,
    //     .timestamp = 0, // Event bus will set this
    //     .priority = EVENT_PRIORITY_NORMAL,
    //     .data_size = sizeof(void*),
    //     .data = (void*)logic_engine
    // };
    // if (brain->event_bus) { // Event bus always exists now
    //     event_bus_publish(brain->event_bus, &event); // Event API changed
    // }

    NIMCP_LOGGING_INFO("Symbolic logic engine attached to brain %p", (void*)brain);
    return true;
}

symbolic_logic_t* brain_detach_symbolic_logic(brain_t brain)
{
    // Validate input
    if (!nimcp_validate_pointer(brain, "brain")) {
        set_error("Brain is NULL");
        NIMCP_LOGGING_ERROR("brain_detach_symbolic_logic: brain is NULL");
        return NULL;
    }

    // Get current engine
    /* Phase 8: Heartbeat at operation start */
    symbolic_logic_attachment_heartbeat("symbolic_log_brain_detach_symboli", 0.0f);


    symbolic_logic_t* engine = brain->symbolic_logic;

    // Check if anything attached
    if (engine == NULL) {
        NIMCP_LOGGING_DEBUG("brain_detach_symbolic_logic: no engine attached to brain %p",
                           (void*)brain);
        return NULL;
    }

    // Detach the engine
    brain->symbolic_logic = NULL;

    // Publish detachment event
    // event_data_t event = {
    //     .event_type = EVENT_LOGIC_ENGINE_DETACHED,
    //     .timestamp = 0,
    //     .priority = EVENT_PRIORITY_NORMAL,
    //     .data_size = sizeof(void*),
    //     .data = (void*)engine
    // };
    // if (brain->event_bus) { // Event bus always exists now
    //     event_bus_publish(brain->event_bus, &event); // Event API changed
    // }

    NIMCP_LOGGING_INFO("Symbolic logic engine detached from brain %p", (void*)brain);
    return engine;
}

// brain_get_symbolic_logic is defined in src/core/brain/accessors/nimcp_brain_accessors.c
// symbolic_logic_t* brain_get_symbolic_logic(brain_t brain)
// {
//     // Validate input
//     if (!nimcp_validate_pointer(brain, "brain")) {
//         set_error("Brain is NULL");
//         return NULL;
//     }
//     return brain->symbolic_logic;
// }

bool brain_has_symbolic_logic(brain_t brain)
{
    // Validate input
    if (!nimcp_validate_pointer(brain, "brain")) {
        return false;
    }

    /* Phase 8: Heartbeat at operation start */
    symbolic_logic_attachment_heartbeat("symbolic_log_brain_has_symbolic_l", 0.0f);


    return brain->symbolic_logic != NULL;
}

//=============================================================================
// KG Self-Awareness Integration
//=============================================================================

int symbolic_logic_attachment_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;
    /* Phase 8: Heartbeat at operation start */
    symbolic_logic_attachment_heartbeat("symbolic_log_query_self_knowledge", 0.0f);


    const kg_entity_t* self = kg_reader_get_entity(kg, "Symbolic_Logic_Attachment");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            /* Phase 8: Loop progress heartbeat */
            if ((i & 0xFF) == 0 && self->num_observations > 256) {
                symbolic_logic_attachment_heartbeat("symbolic_log_loop",
                                 (float)(i + 1) / (float)self->num_observations);
            }

            NIMCP_LOGGING_DEBUG("Symbolic_Logic_Attachment self-knowledge: %s", self->observations[i]);
        }
    }
    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Symbolic_Logic_Attachment");
    if (connections) { kg_relation_list_destroy(connections); }
    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Symbolic_Logic_Attachment");
    if (incoming) { kg_relation_list_destroy(incoming); }
    return self ? 1 : 0;
}

/* ============================================================================
 * Phase 8: Instance-Level Health Agent
 * ============================================================================ */

void symbolic_logic_attachment_set_instance_health_agent(void* instance, nimcp_health_agent_t* agent) {
    if (instance) {
        (void)agent;
        g_symbolic_logic_attachment_health_agent = agent;
    }
}

/* ============================================================================
 * Phase 8: Training Integration (Full Implementation)
 * ============================================================================ */

int symbolic_logic_attachment_training_begin(void* instance) {
    if (!instance) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "symbolic_logic_attachment_training_begin: NULL argument");
        return -1;
    }
    symbolic_logic_attachment_heartbeat_instance(NULL, "symbolic_logic_attachment_training_begin", 0.0f);
    return 0;
}

int symbolic_logic_attachment_training_end(void* instance) {
    if (!instance) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "symbolic_logic_attachment_training_end: NULL argument");
        return -1;
    }
    symbolic_logic_attachment_heartbeat_instance(NULL, "symbolic_logic_attachment_training_end", 1.0f);
    return 0;
}

int symbolic_logic_attachment_training_step(void* instance, float progress) {
    if (!instance) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "symbolic_logic_attachment_training_step: NULL argument");
        return -1;
    }
    if (progress < 0.0f) progress = 0.0f;
    if (progress > 1.0f) progress = 1.0f;
    symbolic_logic_attachment_heartbeat_instance(NULL, "symbolic_logic_attachment_training_step", progress);
    return 0;
}
