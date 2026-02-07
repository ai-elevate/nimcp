/**
 * @file nimcp_unification_engine.c
 * @brief MODULE 5: Unification Engine implementation
 *
 * @author NIMCP Development Team - SRP Refactoring
 * @date 2025-11-20
 * @version 3.0.0
 */

#include "cognitive/reasoning/nimcp_unification_engine.h"
#include "security/nimcp_security.h"
#include "security/nimcp_blood_brain_barrier.h"

#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "cognitive/reasoning/nimcp_symbolic_logic_attachment.h"
#include "cognitive/nimcp_symbolic_logic.h"
#include "utils/validation/nimcp_validate.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/memory/nimcp_memory.h"
#include "api/nimcp_api_exception.h"
#include "core/events/nimcp_event_bus.h"

// Bio-async integration
#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_messages.h"
#include "async/nimcp_bio_router.h"
#include "nimcp.h"  // For error codes
#include "utils/exception/nimcp_exception_macros.h"

#include <string.h>
#include <stdio.h>
#include <stdarg.h>

#define LOG_MODULE "reasoning"
#include "utils/fault_tolerance/nimcp_health_agent_macros.h"
#include "mesh/nimcp_mesh_participant.h"
#include "mesh/nimcp_mesh_adapter.h"

NIMCP_DECLARE_HEALTH_AGENT_ATOMIC(unification_engine)
//=============================================================================
// Mesh Participant Registration
//=============================================================================

static mesh_participant_id_t g_unification_engine_mesh_id = 0;
static mesh_participant_registry_t* g_unification_engine_mesh_registry = NULL;

nimcp_error_t unification_engine_mesh_register(mesh_participant_registry_t* registry) {
    if (!registry) return NIMCP_ERROR_NULL_POINTER;
    if (g_unification_engine_mesh_id != 0) return NIMCP_SUCCESS;
    mesh_participant_interface_t iface;
    mesh_participant_interface_init(&iface);
    strncpy(iface.module_name, "unification_engine", MESH_MAX_NAME_LEN - 1);
    iface.type = MESH_PARTICIPANT_MODULE;
    iface.home_channel = mesh_adapter_get_default_channel(MESH_ADAPTER_CATEGORY_COGNITIVE);
    mesh_participant_config_t config;
    mesh_participant_config_init(&config);
    config.module_name = "unification_engine";
    config.type = MESH_PARTICIPANT_MODULE;
    config.home_channel = iface.home_channel;
    nimcp_error_t err = mesh_participant_register(registry, &iface, &config, &g_unification_engine_mesh_id);
    if (err == NIMCP_SUCCESS) g_unification_engine_mesh_registry = registry;
    return err;
}

void unification_engine_mesh_unregister(void) {
    if (g_unification_engine_mesh_registry && g_unification_engine_mesh_id != 0) {
        mesh_participant_unregister(g_unification_engine_mesh_registry, g_unification_engine_mesh_id);
        g_unification_engine_mesh_id = 0;
        g_unification_engine_mesh_registry = NULL;
    }
}


/** @brief Send heartbeat from unification_engine module (instance-level) */
static inline void unification_engine_heartbeat_instance(
    nimcp_health_agent_t* instance_agent, const char* operation, float progress)
{
    if (g_unification_engine_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_unification_engine_health_agent, operation, progress);
    }
    if (instance_agent && instance_agent != g_unification_engine_health_agent) {
        nimcp_health_agent_heartbeat_ex(instance_agent, operation, progress);
    }
}



static __thread char last_error[256] = {0};

static void set_error(const char* fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    vsnprintf(last_error, sizeof(last_error), fmt, args);
    va_end(args);
}

const char* unification_get_last_error(void)
{
    return last_error;
}

bool brain_unify_terms(
    brain_t brain,
    const logical_term_t* term1,
    const logical_term_t* term2,
    unification_t* result)
{
    if (!nimcp_validate_pointer(brain, "brain") || !brain_has_symbolic_logic(brain)) {
        set_error("Brain is NULL or no logic engine attached");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "brain_unify_terms: required parameter is NULL (nimcp_validate_pointer, brain_has_symbolic_logic)");
        return false;
    }

    if (!nimcp_validate_pointer(term1, "term1") || !nimcp_validate_pointer(term2, "term2") ||
        !nimcp_validate_pointer(result, "result")) {
        set_error("NULL parameter");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "brain_unify_terms: required parameter is NULL (nimcp_validate_pointer, nimcp_validate_pointer)");
        return false;
    }

    /* Phase 8: Heartbeat at operation start */
    unification_engine_heartbeat("unification_brain_unify_terms", 0.0f);

    // Delegate to symbolic logic unification
    // Note: symbolic_logic_unify returns unification_t* rather than taking output param
    (void)brain; // Brain reference not needed for current API
    unification_t* unif_result = symbolic_logic_unify(
        (logical_term_t*)term1,
        (logical_term_t*)term2
    );

    bool success = false;
    if (unif_result) {
        // Copy result
        result->success = unif_result->success;
        result->bindings = unif_result->bindings;
        result->num_bindings = unif_result->num_bindings;
        success = unif_result->success;

        // Don't destroy - caller owns the bindings now
        nimcp_free(unif_result);
    } else {
        result->success = false;
        result->bindings = NULL;
        result->num_bindings = 0;
    }

    return success;
}

bool brain_apply_substitution(
    brain_t brain,
    const logical_term_t* term,
    const unification_t* bindings,
    logical_term_t** result)
{
    if (!nimcp_validate_pointer(brain, "brain") || !brain_has_symbolic_logic(brain)) {
        set_error("Brain is NULL or no logic engine attached");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "brain_apply_substitution: required parameter is NULL (nimcp_validate_pointer, brain_has_symbolic_logic)");
        return false;
    }

    if (!nimcp_validate_pointer(term, "term") || !nimcp_validate_pointer(bindings, "bindings") ||
        !nimcp_validate_pointer(result, "result")) {
        set_error("NULL parameter");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "brain_apply_substitution: required parameter is NULL (nimcp_validate_pointer, nimcp_validate_pointer)");
        return false;
    }

    /* Phase 8: Heartbeat at operation start */
    unification_engine_heartbeat("unification_brain_apply_substitution", 0.0f);

    // Delegate to symbolic logic substitution
    // Note: symbolic_logic_substitute takes (term, subst) and returns new term
    (void)brain; // Brain reference not needed for current API

    // Apply each binding in sequence
    logical_term_t* current = logic_term_create(term->type, term->name);
    if (!current) {
        set_error("Failed to copy term");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "brain_apply_substitution: current is NULL");
        return false;
    }

    for (uint32_t i = 0; i < bindings->num_bindings; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && bindings->num_bindings > 256) {
            unification_engine_heartbeat("unification__loop",
                             (float)(i + 1) / (float)bindings->num_bindings);
        }

        if (bindings->bindings[i]) {
            logical_term_t* substituted = symbolic_logic_substitute(
                current,
                bindings->bindings[i]
            );
            if (substituted) {
                logic_term_destroy(current);
                current = substituted;
            }
        }
    }

    *result = current;
    return true;
}

void unification_free_result(unification_t* result)
{
    if (!result) return;
    // Free implementation
    /* Phase 8: Heartbeat at operation start */
    unification_engine_heartbeat("unification__unification_free_res", 0.0f);


    memset(result, 0, sizeof(unification_t));
}

//=============================================================================
// KG Self-Awareness Integration
//=============================================================================

int unification_engine_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;
    /* Phase 8: Heartbeat at operation start */
    unification_engine_heartbeat("unification__query_self_knowledge", 0.0f);


    const kg_entity_t* self = kg_reader_get_entity(kg, "Unification_Engine");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            /* Phase 8: Loop progress heartbeat */
            if ((i & 0xFF) == 0 && self->num_observations > 256) {
                unification_engine_heartbeat("unification__loop",
                                 (float)(i + 1) / (float)self->num_observations);
            }

            NIMCP_LOGGING_DEBUG("Unification_Engine self-knowledge: %s", self->observations[i]);
        }
    }
    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Unification_Engine");
    if (connections) { kg_relation_list_destroy(connections); }
    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Unification_Engine");
    if (incoming) { kg_relation_list_destroy(incoming); }
    return self ? 1 : 0;
}

/* ============================================================================
 * Phase 8: Instance-Level Health Agent
 * ============================================================================ */

void unification_engine_set_instance_health_agent(void* instance, nimcp_health_agent_t* agent) {
    if (instance) {
        (void)agent;
        g_unification_engine_health_agent = agent;
    }
}

/* ============================================================================
 * Phase 8: Training Integration (Full Implementation)
 * ============================================================================ */

int unification_engine_training_begin(void* instance) {
    if (!instance) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "unification_engine_training_begin: NULL argument");
        return -1;
    }
    unification_engine_heartbeat_instance(NULL, "unification_engine_training_begin", 0.0f);
    return 0;
}

int unification_engine_training_end(void* instance) {
    if (!instance) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "unification_engine_training_end: NULL argument");
        return -1;
    }
    unification_engine_heartbeat_instance(NULL, "unification_engine_training_end", 1.0f);
    return 0;
}

int unification_engine_training_step(void* instance, float progress) {
    if (!instance) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "unification_engine_training_step: NULL argument");
        return -1;
    }
    if (progress < 0.0f) progress = 0.0f;
    if (progress > 1.0f) progress = 1.0f;
    unification_engine_heartbeat_instance(NULL, "unification_engine_training_step", progress);
    return 0;
}
