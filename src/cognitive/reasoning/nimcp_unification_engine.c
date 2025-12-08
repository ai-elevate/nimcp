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

#include "cognitive/reasoning/nimcp_symbolic_logic_attachment.h"
#include "cognitive/nimcp_symbolic_logic.h"
#include "utils/validation/nimcp_validate.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/memory/nimcp_memory.h"
#include "core/events/nimcp_event_bus.h"

// Bio-async integration
#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_messages.h"
#include "async/nimcp_bio_router.h"
#include "nimcp.h"  // For error codes

#include <string.h>
#include <stdio.h>
#include <stdarg.h>

#define LOG_MODULE "reasoning"

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
        return false;
    }

    if (!nimcp_validate_pointer(term1, "term1") || !nimcp_validate_pointer(term2, "term2") ||
        !nimcp_validate_pointer(result, "result")) {
        set_error("NULL parameter");
        return false;
    }

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
        return false;
    }

    if (!nimcp_validate_pointer(term, "term") || !nimcp_validate_pointer(bindings, "bindings") ||
        !nimcp_validate_pointer(result, "result")) {
        set_error("NULL parameter");
        return false;
    }

    // Delegate to symbolic logic substitution
    // Note: symbolic_logic_substitute takes (term, subst) and returns new term
    (void)brain; // Brain reference not needed for current API

    // Apply each binding in sequence
    logical_term_t* current = logic_term_create(term->type, term->name);
    if (!current) {
        set_error("Failed to copy term");
        return false;
    }

    for (uint32_t i = 0; i < bindings->num_bindings; i++) {
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
    memset(result, 0, sizeof(unification_t));
}
