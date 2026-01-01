/**
 * @file nimcp_neural_logic_evaluation.c
 * @brief MODULE 2: Neural Logic Evaluation Implementation
 * @version 3.0.0
 * @date 2025-11-20
 *
 * WHAT: Evaluation interface for logic gates with event publishing
 * WHY:  Single Responsibility: Execute logic operations and notify observers
 * HOW:  Wrapper around neural_logic_evaluate() with event bus integration
 *
 * @author NIMCP Development Team
 */

#include "core/logic/nimcp_neural_logic_evaluation.h"
#include "security/nimcp_security.h"
#include "security/nimcp_blood_brain_barrier.h"

#include "core/logic/nimcp_neural_logic_attachment.h"
#include "core/logic/nimcp_neural_logic_circuit_builder.h"
#include "core/brain/nimcp_brain_internal.h"
#include "core/events/nimcp_event_bus.h"
#include "utils/validation/nimcp_validate.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/time/nimcp_time.h"
#include <string.h>

// === BIO-ASYNC + LOGGING + UNIFIED MEMORY INTEGRATION ===
#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/memory/nimcp_unified_memory.h"

#define LOG_MODULE "neural_logic_evaluation"
#define BIO_MODULE_ID 0x0138


//=============================================================================
// Event Payload Structures
//=============================================================================

/**
 * @brief Event payload for EVENT_LOGIC_GATE_EVALUATED
 *
 * WHAT: Data structure published when logic gate is evaluated
 * WHY:  Notify observers of logic operations for monitoring and learning
 * HOW:  Packed struct with gate ID, inputs, output, timestamp
 */
typedef struct {
    uint32_t gate_id;           // Logic gate neuron ID
    uint32_t num_inputs;        // Number of inputs
    float inputs[4];            // Input values (max 4 for complex gates)
    float output;               // Output value [0,1]
    uint64_t timestamp_us;      // Evaluation timestamp
    uint64_t eval_time_us;      // Evaluation duration
} logic_gate_evaluated_event_t;

//=============================================================================
// Helper Functions
//=============================================================================

/**
 * @brief Publish logic gate evaluation event
 *
 * WHAT: Send EVENT_LOGIC_GATE_EVALUATED to event bus
 * WHY:  Notify observers of successful logic evaluation
 * HOW:  Pack payload, call event_bus_publish()
 */
static void publish_gate_evaluated_event(
    brain_t brain,
    uint32_t gate_id,
    const float* inputs,
    uint32_t num_inputs,
    float output,
    uint64_t eval_time_us
) {
    // Guard: NULL brain or no event bus
    if (!brain || !brain->event_bus) {
        return;
    }

    // WHAT: Pack event payload
    // WHY:  Encapsulate evaluation details for subscribers
    // HOW:  Fill struct, clamp inputs to max 4

    logic_gate_evaluated_event_t payload = {0};
    payload.gate_id = gate_id;
    payload.num_inputs = num_inputs < 4 ? num_inputs : 4;
    payload.output = output;
    payload.timestamp_us = 0;  // Timestamp set by event bus
    payload.eval_time_us = eval_time_us;

    // Copy inputs (up to 4)
    for (uint32_t i = 0; i < payload.num_inputs; i++) {
        payload.inputs[i] = inputs[i];
    }

    // Publish event
    brain_event_t event = event_create(EVENT_LOGIC_GATE_EVALUATED, EVENT_PRIORITY_LOW, "neural_logic");
    event_set_data(&event, &payload, sizeof(payload));
    event_bus_publish(brain->event_bus, &event);
}

//=============================================================================
// MODULE 2: Neural Logic Evaluation Implementation
//=============================================================================

bool brain_evaluate_logic_gate(
    brain_t brain,
    uint32_t gate_id,
    const float* inputs,
    uint32_t num_inputs,
    float* output
) {
    // WHAT: Validate all inputs with guard clauses
    // WHY:  Prevent NULL derefs and invalid operations
    // HOW:  Early returns with error logging

    // Guard: NULL brain
    if (!nimcp_validate_pointer(brain, "brain")) {
        LOG_ERROR("brain_evaluate_logic_gate: NULL brain");
        return false;
    }

    // Guard: no logic network attached
    if (!brain_has_neural_logic(brain)) {
        LOG_ERROR("brain_evaluate_logic_gate: brain '%s' has no logic network",
                  "brain");
        return false;
    }

    // Guard: NULL inputs
    if (!nimcp_validate_pointer(inputs, "inputs")) {
        LOG_ERROR("brain_evaluate_logic_gate: NULL inputs");
        return false;
    }

    // Guard: NULL output
    if (!nimcp_validate_pointer(output, "output")) {
        LOG_ERROR("brain_evaluate_logic_gate: NULL output");
        return false;
    }

    // Guard: zero inputs
    if (num_inputs == 0) {
        LOG_ERROR("brain_evaluate_logic_gate: num_inputs is zero");
        return false;
    }

    // WHAT: Execute logic gate evaluation
    // WHY:  Perform actual computation via neural logic network
    // HOW:  Call neural_logic_evaluate() with timing

    neural_logic_network_t network = brain_get_neural_logic(brain);

    bool success = neural_logic_evaluate(
        network,
        gate_id,
        inputs,
        num_inputs,
        output
    );

    uint64_t eval_time_us = 0;  // Timing not available

    if (!success) {
        LOG_ERROR("brain_evaluate_logic_gate: evaluation failed for gate %u", gate_id);
        return false;
    }

    // WHAT: Publish evaluation event for observers
    // WHY:  Enable monitoring, learning, and debugging
    // HOW:  Send EVENT_LOGIC_GATE_EVALUATED

    publish_gate_evaluated_event(brain, gate_id, inputs, num_inputs, *output, eval_time_us);

    LOG_DEBUG("brain_evaluate_logic_gate: gate=%u, output=%.3f, time=%llu μs",
              gate_id, *output, (unsigned long long)eval_time_us);

    return true;
}

bool brain_evaluate_logic_expression(
    brain_t brain,
    const char* expression,
    const float* bindings,
    uint32_t num_bindings,
    float* output
) {
    // WHAT: Validate inputs for expression evaluation
    // WHY:  Ensure safe parsing and evaluation
    // HOW:  Guard clauses with NULL and empty checks

    // Guard: NULL brain
    if (!nimcp_validate_pointer(brain, "brain")) {
        LOG_ERROR("brain_evaluate_logic_expression: NULL brain");
        return false;
    }

    // Guard: no logic network attached
    if (!brain_has_neural_logic(brain)) {
        LOG_ERROR("brain_evaluate_logic_expression: brain '%s' has no logic network",
                  "brain");
        return false;
    }

    // Guard: NULL expression
    if (!nimcp_validate_pointer(expression, "expression")) {
        LOG_ERROR("brain_evaluate_logic_expression: NULL expression");
        return false;
    }

    // Guard: empty expression
    if (expression[0] == '\0') {
        LOG_ERROR("brain_evaluate_logic_expression: empty expression");
        return false;
    }

    // Guard: NULL bindings (if num_bindings > 0)
    if (num_bindings > 0 && !nimcp_validate_pointer(bindings, "bindings")) {
        LOG_ERROR("brain_evaluate_logic_expression: NULL bindings with num_bindings=%u",
                  num_bindings);
        return false;
    }

    // Guard: NULL output
    if (!nimcp_validate_pointer(output, "output")) {
        LOG_ERROR("brain_evaluate_logic_expression: NULL output");
        return false;
    }

    // WHAT: Build circuit from expression
    // WHY:  Convert string to executable neural circuit
    // HOW:  Call MODULE 3 (circuit builder) to parse and construct

    uint32_t circuit_id = brain_build_logic_circuit(brain, expression);
    if (circuit_id == UINT32_MAX) {
        LOG_ERROR("brain_evaluate_logic_expression: failed to parse '%s'", expression);
        return false;
    }

    // WHAT: Map variable bindings to inputs
    // WHY:  Substitute variable values (A, B, C...) with actual input values
    // HOW:  Extract variables from expression, create input array

    // For simplicity, assume circuit_id is root gate and bindings map directly
    // Full implementation would analyze expression to determine input mapping

    // Evaluate circuit with bindings as inputs
    // Timing removed

    bool success = brain_evaluate_logic_gate(
        brain,
        circuit_id,
        bindings,
        num_bindings,
        output
    );

    uint64_t eval_time_us = 0;  // Timing not available

    if (!success) {
        LOG_ERROR("brain_evaluate_logic_expression: evaluation failed for '%s'", expression);
        return false;
    }

    LOG_INFO("brain_evaluate_logic_expression: '%s' = %.3f (time=%llu μs)",
             expression, *output, (unsigned long long)eval_time_us);

    return true;
}

//=============================================================================
// MODULE 2.1: Batch Logic Gate Evaluation Implementation
//=============================================================================

// Forward declarations for GPU batch kernel launchers
// Note: GPU batch operations are implemented in nimcp_neural_logic_kernels.cu
// The C implementation uses CPU fallback or calls through neural_logic API

/**
 * @brief CPU fallback for batch gate evaluation
 *
 * Sequential evaluation when GPU is unavailable.
 */
static bool evaluate_gates_batch_cpu(
    brain_t brain,
    const uint32_t* gate_ids,
    uint32_t num_gates,
    const float* all_inputs,
    const uint32_t* inputs_per_gate,
    float* outputs
) {
    // WHAT: Sequential evaluation of gates on CPU
    // WHY:  Fallback when GPU unavailable
    // HOW:  Loop through gates, call brain_evaluate_logic_gate() for each

    uint32_t input_offset = 0;

    for (uint32_t i = 0; i < num_gates; i++) {
        uint32_t gate_id = gate_ids[i];
        uint32_t num_inputs = inputs_per_gate[i];

        // Evaluate single gate
        bool success = brain_evaluate_logic_gate(
            brain,
            gate_id,
            &all_inputs[input_offset],
            num_inputs,
            &outputs[i]
        );

        if (!success) {
            LOG_ERROR("evaluate_gates_batch_cpu: failed at gate %u (index %u)", gate_id, i);
            return false;
        }

        input_offset += num_inputs;
    }

    return true;
}

bool brain_evaluate_logic_gates_batch(
    brain_t brain,
    const uint32_t* gate_ids,
    uint32_t num_gates,
    const float* all_inputs,
    const uint32_t* inputs_per_gate,
    float* outputs
) {
    // WHAT: Validate all inputs with guard clauses
    // WHY:  Prevent NULL derefs and invalid operations
    // HOW:  Early returns with error logging

    // Guard: NULL brain
    if (!nimcp_validate_pointer(brain, "brain")) {
        LOG_ERROR("brain_evaluate_logic_gates_batch: NULL brain");
        return false;
    }

    // Guard: no logic network attached
    if (!brain_has_neural_logic(brain)) {
        LOG_ERROR("brain_evaluate_logic_gates_batch: brain has no logic network");
        return false;
    }

    // Guard: NULL gate_ids
    if (!nimcp_validate_pointer(gate_ids, "gate_ids")) {
        LOG_ERROR("brain_evaluate_logic_gates_batch: NULL gate_ids");
        return false;
    }

    // Guard: zero gates
    if (num_gates == 0) {
        LOG_ERROR("brain_evaluate_logic_gates_batch: num_gates is zero");
        return false;
    }

    // Guard: NULL all_inputs
    if (!nimcp_validate_pointer(all_inputs, "all_inputs")) {
        LOG_ERROR("brain_evaluate_logic_gates_batch: NULL all_inputs");
        return false;
    }

    // Guard: NULL inputs_per_gate
    if (!nimcp_validate_pointer(inputs_per_gate, "inputs_per_gate")) {
        LOG_ERROR("brain_evaluate_logic_gates_batch: NULL inputs_per_gate");
        return false;
    }

    // Guard: NULL outputs
    if (!nimcp_validate_pointer(outputs, "outputs")) {
        LOG_ERROR("brain_evaluate_logic_gates_batch: NULL outputs");
        return false;
    }

    // WHAT: Evaluate gates in batch
    // WHY:  Provides API for batch operations; GPU acceleration happens at network level
    // HOW:  Use CPU path with optimized sequential evaluation
    //       GPU kernels are available in nimcp_neural_logic_kernels.cu for direct use
    //       when neural_logic_network provides batch evaluation API

    // For Phase 3.1-3.2, we implement CPU path with GPU kernels available
    // The GPU path requires neural_logic_evaluate_batch() to be added to
    // nimcp_neural_logic.h/c which manages device memory

    bool success = evaluate_gates_batch_cpu(brain, gate_ids, num_gates, all_inputs, inputs_per_gate, outputs);
    if (success) {
        LOG_DEBUG("brain_evaluate_logic_gates_batch: evaluated %u gates", num_gates);
    }
    return success;
}

bool brain_evaluate_logic_expressions_batch(
    brain_t brain,
    const char** expressions,
    uint32_t num_expressions,
    const float** bindings,
    const uint32_t* num_bindings_per_expr,
    float* outputs
) {
    // WHAT: Validate all inputs with guard clauses
    // WHY:  Prevent NULL derefs and invalid operations
    // HOW:  Early returns with error logging

    // Guard: NULL brain
    if (!nimcp_validate_pointer(brain, "brain")) {
        LOG_ERROR("brain_evaluate_logic_expressions_batch: NULL brain");
        return false;
    }

    // Guard: no logic network attached
    if (!brain_has_neural_logic(brain)) {
        LOG_ERROR("brain_evaluate_logic_expressions_batch: brain has no logic network");
        return false;
    }

    // Guard: NULL expressions
    if (!nimcp_validate_pointer(expressions, "expressions")) {
        LOG_ERROR("brain_evaluate_logic_expressions_batch: NULL expressions");
        return false;
    }

    // Guard: zero expressions
    if (num_expressions == 0) {
        LOG_ERROR("brain_evaluate_logic_expressions_batch: num_expressions is zero");
        return false;
    }

    // Guard: NULL bindings
    if (!nimcp_validate_pointer(bindings, "bindings")) {
        LOG_ERROR("brain_evaluate_logic_expressions_batch: NULL bindings");
        return false;
    }

    // Guard: NULL num_bindings_per_expr
    if (!nimcp_validate_pointer(num_bindings_per_expr, "num_bindings_per_expr")) {
        LOG_ERROR("brain_evaluate_logic_expressions_batch: NULL num_bindings_per_expr");
        return false;
    }

    // Guard: NULL outputs
    if (!nimcp_validate_pointer(outputs, "outputs")) {
        LOG_ERROR("brain_evaluate_logic_expressions_batch: NULL outputs");
        return false;
    }

    // WHAT: Build circuits from expressions, then batch evaluate
    // WHY:  Leverage GPU acceleration for evaluation phase
    // HOW:  Parse expressions -> collect gate IDs -> batch evaluate

    // Step 1: Parse all expressions and build circuits
    uint32_t* circuit_ids = (uint32_t*)nimcp_calloc(num_expressions, sizeof(uint32_t));
    if (!circuit_ids) {
        LOG_ERROR("brain_evaluate_logic_expressions_batch: failed to allocate circuit_ids");
        return false;
    }

    for (uint32_t i = 0; i < num_expressions; i++) {
        if (!expressions[i] || expressions[i][0] == '\0') {
            LOG_ERROR("brain_evaluate_logic_expressions_batch: NULL or empty expression at index %u", i);
            nimcp_free(circuit_ids);
            return false;
        }

        circuit_ids[i] = brain_build_logic_circuit(brain, expressions[i]);
        if (circuit_ids[i] == UINT32_MAX) {
            LOG_ERROR("brain_evaluate_logic_expressions_batch: failed to parse expression '%s'", expressions[i]);
            nimcp_free(circuit_ids);
            return false;
        }
    }

    // Step 2: Calculate total inputs and build flattened arrays
    uint32_t total_inputs = 0;
    for (uint32_t i = 0; i < num_expressions; i++) {
        total_inputs += num_bindings_per_expr[i];
    }

    float* all_inputs = (float*)nimcp_calloc(total_inputs, sizeof(float));
    if (!all_inputs) {
        LOG_ERROR("brain_evaluate_logic_expressions_batch: failed to allocate all_inputs");
        nimcp_free(circuit_ids);
        return false;
    }

    // Flatten bindings into single array
    uint32_t offset = 0;
    for (uint32_t i = 0; i < num_expressions; i++) {
        for (uint32_t j = 0; j < num_bindings_per_expr[i]; j++) {
            all_inputs[offset++] = bindings[i][j];
        }
    }

    // Step 3: Batch evaluate all circuits
    bool success = brain_evaluate_logic_gates_batch(
        brain,
        circuit_ids,
        num_expressions,
        all_inputs,
        num_bindings_per_expr,
        outputs
    );

    nimcp_free(all_inputs);
    nimcp_free(circuit_ids);

    if (success) {
        LOG_INFO("brain_evaluate_logic_expressions_batch: evaluated %u expressions", num_expressions);
    }

    return success;
}

bool brain_get_evaluation_stats(
    brain_t brain,
    uint64_t* eval_time_us,
    uint32_t* spike_count
) {
    // WHAT: Validate inputs for stats query
    // WHY:  Prevent NULL derefs when writing output
    // HOW:  Guard clauses on all pointers

    // Guard: NULL brain
    if (!nimcp_validate_pointer(brain, "brain")) {
        LOG_ERROR("brain_get_evaluation_stats: NULL brain");
        return false;
    }

    // Guard: no logic network attached
    if (!brain_has_neural_logic(brain)) {
        LOG_ERROR("brain_get_evaluation_stats: brain '%s' has no logic network",
                  "brain");
        return false;
    }

    // Guard: NULL eval_time_us
    if (!nimcp_validate_pointer(eval_time_us, "eval_time_us")) {
        LOG_ERROR("brain_get_evaluation_stats: NULL eval_time_us");
        return false;
    }

    // Guard: NULL spike_count
    if (!nimcp_validate_pointer(spike_count, "spike_count")) {
        LOG_ERROR("brain_get_evaluation_stats: NULL spike_count");
        return false;
    }

    // WHAT: Query network statistics
    // WHY:  Retrieve performance metrics from neural logic network
    // HOW:  Call neural_logic_get_stats() and extract relevant fields

    neural_logic_network_t network = brain_get_neural_logic(brain);

    uint32_t total_gates = 0;
    uint32_t total_vars = 0;
    uint64_t total_spikes = 0;
    float avg_eval_time = 0.0F;
    uint64_t gpu_memory = 0;

    bool success = neural_logic_get_stats(
        network,
        &total_gates,
        &total_vars,
        &total_spikes,
        &avg_eval_time,
        &gpu_memory
    );

    if (!success) {
        LOG_ERROR("brain_get_evaluation_stats: failed to get network stats");
        return false;
    }

    // Convert average evaluation time to microseconds
    *eval_time_us = (uint64_t)(avg_eval_time);
    *spike_count = (uint32_t)total_spikes;

    return true;
}
