//=============================================================================
// nimcp_neuron_model.h - Plugin Architecture for Neuron Models
//=============================================================================
/**
 * @file nimcp_neuron_model.h
 * @brief Generic neuron model interface using Strategy Pattern
 *
 * ARCHITECTURAL OVERVIEW:
 * This module provides a plugin-based architecture for multiple neuron models:
 * - Strategy Pattern: Each model implements the same interface
 * - Plugin Pattern: Models can be registered and selected at runtime
 * - Factory Pattern: Models are created via constructor functions
 *
 * SUPPORTED MODELS:
 * - Leaky Integrate-and-Fire (LIF) - Legacy model
 * - Izhikevich - Simple, biologically realistic, 20+ firing patterns
 * - Adaptive Exponential IF (AdEx) - Future extension
 * - Hodgkin-Huxley - Future extension for detailed biophysics
 *
 * DESIGN PATTERNS:
 * - Strategy Pattern: Neuron model polymorphism via function pointers
 * - Handle/Body: Opaque neuron_model_state_t hides implementation
 * - Factory Pattern: Model-specific constructors
 *
 * PERFORMANCE:
 * - Model selection: O(1) via function pointer dispatch
 * - Update step: Model-dependent (typically O(1))
 * - Memory: Model-dependent state size
 *
 * @author NIMCP Development Team
 * @date 2025-11-06
 */

#ifndef NIMCP_NEURON_MODEL_H
#define NIMCP_NEURON_MODEL_H

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// Forward Declarations
//=============================================================================

typedef struct neuron_model_state_struct* neuron_model_state_t;
typedef struct neuron_model_vtable neuron_model_vtable_t;

//=============================================================================
// Model Type Enumeration
//=============================================================================

/**
 * @brief Neuron model types
 *
 * WHAT: Enumeration of supported neuron models
 * WHY: Allows runtime selection and switching between models
 * HOW: Used as discriminator in neuron structures
 */
typedef enum {
    NEURON_MODEL_LIF,           /**< Leaky Integrate-and-Fire */
    NEURON_MODEL_IZHIKEVICH,    /**< Izhikevich simple model */
    NEURON_MODEL_ADEX,          /**< Adaptive Exponential IF */
    NEURON_MODEL_HODGKIN_HUXLEY /**< Hodgkin-Huxley biophysical */
} neuron_model_type_t;

//=============================================================================
// Virtual Table - Strategy Pattern Implementation
//=============================================================================

/**
 * @brief Neuron model virtual function table
 *
 * DESIGN PATTERN: Strategy Pattern
 * WHAT: Interface for all neuron model implementations
 * WHY: Enables polymorphic behavior without inheritance
 * HOW: Each model provides its own implementation of these functions
 *
 * INVARIANTS:
 * - All function pointers must be non-NULL
 * - state_size must be > 0
 * - Model name must be valid C string
 */
struct neuron_model_vtable {
    /** Model identification */
    const char* name;                /**< Human-readable model name */
    neuron_model_type_t type;        /**< Model type discriminator */
    size_t state_size;               /**< Size of model state in bytes */

    /** Lifecycle functions */
    void (*init)(neuron_model_state_t state, const void* params);
    void (*destroy)(neuron_model_state_t state);

    /** Dynamics functions */
    void (*update)(neuron_model_state_t state, float dt, float input_current);
    bool (*check_spike)(const neuron_model_state_t state);
    void (*post_spike)(neuron_model_state_t state);

    /** State access */
    float (*get_voltage)(const neuron_model_state_t state);
    void (*set_voltage)(neuron_model_state_t state, float voltage);

    /** Utility functions */
    void (*reset)(neuron_model_state_t state);
    void (*copy)(neuron_model_state_t dst, const neuron_model_state_t src);
};

//=============================================================================
// Public API - Factory and Management
//=============================================================================

/**
 * @brief Create a neuron model state
 *
 * WHAT: Allocates and initializes a neuron model state
 * WHY: Factory function for model-agnostic creation
 * HOW: Dispatches to model-specific constructor via vtable
 *
 * COMPLEXITY: O(1)
 * MEMORY: Allocates vtable->state_size bytes
 *
 * @param vtable Virtual function table for desired model
 * @param params Model-specific initialization parameters (can be NULL)
 * @return Opaque pointer to neuron model state, or NULL on failure
 */
neuron_model_state_t neuron_model_create(const neuron_model_vtable_t* vtable, const void* params);

/**
 * @brief Destroy a neuron model state
 *
 * WHAT: Frees neuron model state and calls model-specific cleanup
 * WHY: Proper resource cleanup and memory management
 * HOW: Calls destroy() from vtable if provided, then frees state
 *
 * COMPLEXITY: O(1)
 *
 * @param state Neuron model state to destroy (can be NULL - no-op)
 */
void neuron_model_destroy(neuron_model_state_t state);

/**
 * @brief Update neuron dynamics for one time step
 *
 * WHAT: Advances neuron model state by dt milliseconds
 * WHY: Core simulation loop - updates membrane potential and state variables
 * HOW: Dispatches to model-specific update function
 *
 * COMPLEXITY: Model-dependent (typically O(1))
 *
 * @param state Neuron model state
 * @param dt Time step in milliseconds
 * @param input_current Input current in arbitrary units
 */
void neuron_model_update(neuron_model_state_t state, float dt, float input_current);

/**
 * @brief Check if neuron has spiked
 *
 * WHAT: Tests whether neuron crossed spike threshold
 * WHY: Spike detection for event-driven processing
 * HOW: Calls model-specific spike detection function
 *
 * COMPLEXITY: O(1)
 *
 * @param state Neuron model state
 * @return true if neuron spiked this timestep, false otherwise
 */
bool neuron_model_check_spike(const neuron_model_state_t state);

/**
 * @brief Perform post-spike reset
 *
 * WHAT: Resets neuron state variables after spike
 * WHY: Implements refractory period and spike aftermath
 * HOW: Calls model-specific post-spike handler
 *
 * COMPLEXITY: O(1)
 *
 * @param state Neuron model state
 */
void neuron_model_post_spike(neuron_model_state_t state);

/**
 * @brief Get current membrane voltage
 *
 * WHAT: Retrieves neuron membrane potential
 * WHY: Monitoring and visualization of neuron state
 * HOW: Calls model-specific voltage getter
 *
 * COMPLEXITY: O(1)
 *
 * @param state Neuron model state
 * @return Membrane voltage in millivolts
 */
float neuron_model_get_voltage(const neuron_model_state_t state);

/**
 * @brief Set membrane voltage
 *
 * WHAT: Manually sets neuron membrane potential
 * WHY: External stimulation and initialization
 * HOW: Calls model-specific voltage setter
 *
 * COMPLEXITY: O(1)
 *
 * @param state Neuron model state
 * @param voltage Desired membrane voltage in millivolts
 */
void neuron_model_set_voltage(neuron_model_state_t state, float voltage);

/**
 * @brief Reset neuron to resting state
 *
 * WHAT: Returns neuron to initial conditions
 * WHY: Network reset and testing
 * HOW: Calls model-specific reset function
 *
 * COMPLEXITY: O(1)
 *
 * @param state Neuron model state
 */
void neuron_model_reset(neuron_model_state_t state);

/**
 * @brief Get model name string
 *
 * WHAT: Retrieves human-readable model name
 * WHY: Debugging, logging, and introspection
 * HOW: Returns name field from vtable
 *
 * COMPLEXITY: O(1)
 *
 * @param state Neuron model state
 * @return Model name string (never NULL if state is valid)
 */
const char* neuron_model_get_name(const neuron_model_state_t state);

/**
 * @brief Get model type
 *
 * WHAT: Retrieves model type discriminator
 * WHY: Runtime type checking and model-specific operations
 * HOW: Returns type field from vtable
 *
 * COMPLEXITY: O(1)
 *
 * @param state Neuron model state
 * @return Model type enumeration value
 */
neuron_model_type_t neuron_model_get_type(const neuron_model_state_t state);

//=============================================================================
// Model Registration - Available Models
//=============================================================================

/**
 * @brief Get Izhikevich model vtable
 *
 * WHAT: Returns virtual function table for Izhikevich neuron model
 * WHY: Factory accessor for Izhikevich model creation
 * HOW: Returns pointer to static vtable structure
 *
 * @return Pointer to Izhikevich vtable (never NULL)
 */
const neuron_model_vtable_t* neuron_model_get_izhikevich_vtable(void);

/**
 * @brief Get LIF model vtable
 *
 * WHAT: Returns virtual function table for Leaky Integrate-and-Fire model
 * WHY: Factory accessor for legacy LIF model
 * HOW: Returns pointer to static vtable structure
 *
 * @return Pointer to LIF vtable (never NULL)
 */
const neuron_model_vtable_t* neuron_model_get_lif_vtable(void);

#ifdef __cplusplus
}
#endif

#endif  // NIMCP_NEURON_MODEL_H
