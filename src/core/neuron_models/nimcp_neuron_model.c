//=============================================================================
// nimcp_neuron_model.c - Neuron Model Plugin Interface Implementation
//=============================================================================
/**
 * @file nimcp_neuron_model.c
 * @brief Implementation of generic neuron model interface
 *
 * ARCHITECTURAL OVERVIEW:
 * This file implements the plugin framework for neuron models using:
 * - Strategy Pattern: Dispatch via virtual function tables
 * - Handle/Body: Opaque state structure with vtable pointer
 * - Factory Pattern: Model-agnostic creation interface
 *
 * DESIGN PATTERNS:
 * - Strategy: Each model implements same interface differently
 * - Factory: neuron_model_create() dispatches to model constructors
 * - Handle/Body: State struct hidden behind opaque pointer
 *
 * COMPLEXITY:
 * - All operations: O(1) - simple function pointer dispatch
 * - Memory: sizeof(neuron_model_state_struct) + model-specific state
 *
 * DESIGN PRINCIPLES:
 * - No nested ifs: Guard clauses only
 * - Single responsibility: Each function does one thing
 * - Functions <50 lines: Keep interface layer thin
 *
 * @author NIMCP Development Team
 * @date 2025-11-06
 */

#include "nimcp_neuron_model.h"
#include "nimcp_neuron_model_internal.h"
#include <stdlib.h>
#include <string.h>
#include "utils/memory/nimcp_memory.h"

//=============================================================================
// Factory Functions
//=============================================================================

/**
 * @brief Create neuron model state with vtable
 *
 * WHAT: Allocates state structure and initializes model
 * WHY: Single allocation improves cache locality
 * HOW: Allocates [vtable_ptr + model_state] in one block
 *
 * PATTERN: Factory Method
 * COMPLEXITY: O(1) + model init complexity
 * MEMORY: sizeof(struct) + vtable->state_size
 *
 * @param vtable Virtual function table
 * @param params Model-specific parameters (NULL = use defaults)
 * @return Initialized state or NULL on failure
 */
neuron_model_state_t neuron_model_create(const neuron_model_vtable_t* vtable, const void* params) {
    // Guard: Validate vtable
    if (!vtable || !vtable->init || vtable->state_size == 0) {
        return NULL;
    }

    // Allocate contiguous block for state + model data
    const size_t total_size = sizeof(struct neuron_model_state_struct) + vtable->state_size;
    neuron_model_state_t state = (neuron_model_state_t)nimcp_calloc(1, total_size);

    // Guard: Check allocation
    if (!state) {
        return NULL;
    }

    // Initialize vtable pointer
    state->vtable = vtable;

    // Call model-specific initialization
    vtable->init(state, params);

    return state;
}

/**
 * @brief Destroy neuron model state
 *
 * WHAT: Calls model destructor and frees memory
 * WHY: Proper cleanup prevents memory leaks
 * HOW: Calls vtable->destroy() if present, then nimcp_free()
 *
 * COMPLEXITY: O(1) + model destroy complexity
 *
 * @param state State to destroy (NULL-safe)
 */
void neuron_model_destroy(neuron_model_state_t state) {
    // Guard: NULL check
    if (!state) {
        return;
    }

    // Call model-specific cleanup if provided
    if (state->vtable->destroy) {
        state->vtable->destroy(state);
    }

    // Free the state block
    nimcp_free(state);
}

//=============================================================================
// Dynamics Functions - Dispatch to Model Implementation
//=============================================================================

/**
 * @brief Update neuron model dynamics
 *
 * PATTERN: Strategy dispatch
 * COMPLEXITY: O(1) dispatch + model update complexity
 */
void neuron_model_update(neuron_model_state_t state, float dt, float input_current) {
    // Guard: Validate state and vtable
    if (!state || !state->vtable || !state->vtable->update) {
        return;
    }

    state->vtable->update(state, dt, input_current);
}

/**
 * @brief Check if neuron spiked
 *
 * PATTERN: Strategy dispatch
 * COMPLEXITY: O(1)
 */
bool neuron_model_check_spike(const neuron_model_state_t state) {
    // Guard: Validate state and vtable
    if (!state || !state->vtable || !state->vtable->check_spike) {
        return false;
    }

    return state->vtable->check_spike(state);
}

/**
 * @brief Execute post-spike reset
 *
 * PATTERN: Strategy dispatch
 * COMPLEXITY: O(1)
 */
void neuron_model_post_spike(neuron_model_state_t state) {
    // Guard: Validate state and vtable
    if (!state || !state->vtable || !state->vtable->post_spike) {
        return;
    }

    state->vtable->post_spike(state);
}

//=============================================================================
// State Access Functions
//=============================================================================

/**
 * @brief Get membrane voltage
 *
 * COMPLEXITY: O(1)
 */
float neuron_model_get_voltage(const neuron_model_state_t state) {
    // Guard: Validate state and vtable
    if (!state || !state->vtable || !state->vtable->get_voltage) {
        return 0.0f;
    }

    return state->vtable->get_voltage(state);
}

/**
 * @brief Set membrane voltage
 *
 * COMPLEXITY: O(1)
 */
void neuron_model_set_voltage(neuron_model_state_t state, float voltage) {
    // Guard: Validate state and vtable
    if (!state || !state->vtable || !state->vtable->set_voltage) {
        return;
    }

    state->vtable->set_voltage(state, voltage);
}

/**
 * @brief Reset neuron to resting state
 *
 * COMPLEXITY: O(1)
 */
void neuron_model_reset(neuron_model_state_t state) {
    // Guard: Validate state and vtable
    if (!state || !state->vtable || !state->vtable->reset) {
        return;
    }

    state->vtable->reset(state);
}

//=============================================================================
// Introspection Functions
//=============================================================================

/**
 * @brief Get model name
 *
 * COMPLEXITY: O(1)
 */
const char* neuron_model_get_name(const neuron_model_state_t state) {
    // Guard: Validate state and vtable
    if (!state || !state->vtable || !state->vtable->name) {
        return "unknown";
    }

    return state->vtable->name;
}

/**
 * @brief Get model type
 *
 * COMPLEXITY: O(1)
 */
neuron_model_type_t neuron_model_get_type(const neuron_model_state_t state) {
    // Guard: Validate state and vtable
    if (!state || !state->vtable) {
        return NEURON_MODEL_LIF;  // Default fallback
    }

    return state->vtable->type;
}
