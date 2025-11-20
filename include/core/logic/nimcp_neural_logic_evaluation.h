/**
 * @file nimcp_neural_logic_evaluation.h
 * @brief MODULE 2: Neural Logic Evaluation - Evaluate Logic Gates Through Brain
 * @version 3.0.0
 * @date 2025-11-20
 *
 * WHAT: Evaluation interface for logic gates with event publishing
 * WHY:  Single Responsibility: Execute logic operations and notify observers
 * HOW:  Wrapper around neural_logic_evaluate() with event bus integration
 *
 * SINGLE RESPONSIBILITY PRINCIPLE (SRP):
 * - SOLE RESPONSIBILITY: Evaluate logic gates and publish evaluation events
 * - DOES: Evaluate single gates, evaluate expressions, publish EVENT_LOGIC_GATE_EVALUATED
 * - DOES NOT: Attach networks (MODULE 1), build circuits (MODULE 3), modulate (MODULE 4)
 *
 * @author NIMCP Development Team
 */

#ifndef NIMCP_NEURAL_LOGIC_EVALUATION_H
#define NIMCP_NEURAL_LOGIC_EVALUATION_H

#include <stdint.h>
#include <stdbool.h>
#include "core/neuron_types/nimcp_neural_logic.h"
#include "core/brain/nimcp_brain.h"
#include "common/nimcp_export.h"

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// MODULE 2: Neural Logic Evaluation API
//=============================================================================

/**
 * @brief Evaluate logic gate through brain interface
 *
 * WHAT: Execute logic gate evaluation with inputs and capture output
 * WHY:  Provide brain-level wrapper for neural logic evaluation
 * HOW:  Validate inputs → call neural_logic_evaluate() → publish event → return result
 *
 * @param brain Brain instance with attached logic network
 * @param gate_id Logic gate neuron ID
 * @param inputs Input values (boolean encoded as float 0/1), array of size num_inputs
 * @param num_inputs Number of inputs (typically 1-2 for binary gates)
 * @param output Output value [0,1] (OUT parameter)
 * @return true on success, false on failure
 *
 * GUARD CLAUSES:
 * - NULL brain → false + error log
 * - brain->logic == NULL → false + error log
 * - NULL inputs → false + error log
 * - NULL output → false + error log
 * - num_inputs == 0 → false + error log
 *
 * BEHAVIOR:
 * - Validates all inputs with guard clauses
 * - Calls neural_logic_evaluate(brain->logic, gate_id, inputs, num_inputs, output)
 * - On success: publishes EVENT_LOGIC_GATE_EVALUATED event
 * - On failure: logs error and returns false
 *
 * EVENTS:
 * - EVENT_LOGIC_GATE_EVALUATED: Fires after successful evaluation
 *   Payload: gate_id, inputs, num_inputs, output, timestamp
 *
 * COMPLEXITY: O(t * n) where t = simulation time, n = neurons in gate circuit
 * THREAD SAFETY: Not thread-safe
 *
 * EXAMPLE:
 * ```c
 * float inputs[2] = {1.0f, 1.0f};  // A=true, B=true
 * float output;
 *
 * if (brain_evaluate_logic_gate(brain, and_gate_id, inputs, 2, &output)) {
 *     printf("AND(1, 1) = %.1f\n", output);  // Expected: 1.0
 * }
 * ```
 */
NIMCP_EXPORT bool brain_evaluate_logic_gate(
    brain_t brain,
    uint32_t gate_id,
    const float* inputs,
    uint32_t num_inputs,
    float* output
);

/**
 * @brief Evaluate logic expression with variable bindings
 *
 * WHAT: Evaluate string expression (e.g., "A AND B") with variable substitution
 * WHY:  Provide high-level interface for complex logic without manual gate management
 * HOW:  Parse expression → bind variables → evaluate circuit → publish event
 *
 * @param brain Brain instance with attached logic network
 * @param expression Logical expression string (e.g., "A AND B", "(A OR B) AND C")
 * @param bindings Variable bindings (A→inputs[0], B→inputs[1], ..., Z→inputs[25])
 * @param num_bindings Number of variable bindings (typically matches unique vars in expr)
 * @param output Evaluation result [0,1] (OUT parameter)
 * @return true on success, false on failure
 *
 * GUARD CLAUSES:
 * - NULL brain → false + error log
 * - brain->logic == NULL → false + error log
 * - NULL expression → false + error log
 * - Empty expression → false + error log
 * - NULL bindings → false + error log (if num_bindings > 0)
 * - NULL output → false + error log
 *
 * BEHAVIOR:
 * - Parses expression into circuit (uses MODULE 3 internally)
 * - Maps variable bindings to gate inputs (A→bindings[0], B→bindings[1], ...)
 * - Evaluates circuit with neural_logic_evaluate()
 * - Publishes EVENT_LOGIC_GATE_EVALUATED event
 * - Cleans up temporary circuit
 *
 * VARIABLE BINDING:
 * - Variables are single uppercase letters (A-Z)
 * - bindings[0] = value for 'A', bindings[1] = value for 'B', etc.
 * - Expression can use subset of variables (e.g., "A AND C" uses bindings[0], bindings[2])
 *
 * SUPPORTED OPERATORS:
 * - AND: "A AND B", "A & B", "A ∧ B"
 * - OR:  "A OR B", "A | B", "A ∨ B"
 * - NOT: "NOT A", "!A", "¬A"
 * - XOR: "A XOR B", "A ⊕ B"
 * - IMPLIES: "A -> B", "A → B"
 * - Parentheses: "(A AND B) OR C"
 *
 * COMPLEXITY: O(m + t * n) where m = expression length, t = sim time, n = neurons
 * THREAD SAFETY: Not thread-safe
 *
 * EXAMPLE:
 * ```c
 * // Evaluate: (A AND B) OR C
 * const char* expr = "(A AND B) OR C";
 * float bindings[3] = {1.0f, 1.0f, 0.0f};  // A=T, B=T, C=F
 * float output;
 *
 * if (brain_evaluate_logic_expression(brain, expr, bindings, 3, &output)) {
 *     printf("Result: %.1f\n", output);  // Expected: 1.0 (true OR false = true)
 * }
 * ```
 */
NIMCP_EXPORT bool brain_evaluate_logic_expression(
    brain_t brain,
    const char* expression,
    const float* bindings,
    uint32_t num_bindings,
    float* output
);

/**
 * @brief Get last evaluation statistics
 *
 * WHAT: Retrieve timing and performance metrics from last evaluation
 * WHY:  Enable performance monitoring and optimization
 * HOW:  Query neural logic network stats for last gate evaluation
 *
 * @param brain Brain instance with attached logic network
 * @param eval_time_us Output: evaluation time in microseconds (OUT parameter)
 * @param spike_count Output: number of spikes during evaluation (OUT parameter)
 * @return true on success, false on failure
 *
 * GUARD CLAUSES:
 * - NULL brain → false + error log
 * - brain->logic == NULL → false + error log
 * - NULL eval_time_us → false + error log
 * - NULL spike_count → false + error log
 *
 * COMPLEXITY: O(1)
 * THREAD SAFETY: Thread-safe for read-only access
 *
 * EXAMPLE:
 * ```c
 * float inputs[2] = {1.0f, 0.0f};
 * float output;
 * brain_evaluate_logic_gate(brain, gate_id, inputs, 2, &output);
 *
 * uint64_t eval_time;
 * uint32_t spikes;
 * brain_get_evaluation_stats(brain, &eval_time, &spikes);
 * printf("Evaluation: %llu μs, %u spikes\n", eval_time, spikes);
 * ```
 */
NIMCP_EXPORT bool brain_get_evaluation_stats(
    brain_t brain,
    uint64_t* eval_time_us,
    uint32_t* spike_count
);

#ifdef __cplusplus
}
#endif

#endif // NIMCP_NEURAL_LOGIC_EVALUATION_H
