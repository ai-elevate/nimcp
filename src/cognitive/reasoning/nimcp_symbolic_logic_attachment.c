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

//=============================================================================
#include <stddef.h>  /* for NULL */
// Health Agent Integration (Phase 8: System-Wide Health Integration)
//=============================================================================
struct nimcp_health_agent;
typedef struct nimcp_health_agent nimcp_health_agent_t;
extern void nimcp_health_agent_heartbeat_ex(nimcp_health_agent_t* agent,
                                             const char* operation,
                                             float progress);

/** Global health agent for symbolic_logic_attachment module */
static nimcp_health_agent_t* g_symbolic_logic_attachment_health_agent = NULL;

/**
 * @brief Set health agent for symbolic_logic_attachment heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
void symbolic_logic_attachment_set_health_agent(nimcp_health_agent_t* agent) {
    g_symbolic_logic_attachment_health_agent = agent;
}

/** @brief Send heartbeat from symbolic_logic_attachment module */
static inline void symbolic_logic_attachment_heartbeat(const char* operation, float progress) {
    if (g_symbolic_logic_attachment_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_symbolic_logic_attachment_health_agent, operation, progress);
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
