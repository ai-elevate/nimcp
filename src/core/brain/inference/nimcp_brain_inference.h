//=============================================================================
// nimcp_brain_inference.h - Brain Inference Module
//=============================================================================
/**
 * @file nimcp_brain_inference.h
 * @brief Brain inference and prediction API
 *
 * WHAT: Inference engine for brain predictions and decisions
 * WHY:  Separates inference logic from core brain module for modularity
 * HOW:  Forward pass, decision caching, mirror neuron integration
 *
 * EXTRACTED FROM: nimcp_brain.c (lines 5343-7087)
 *
 * ARCHITECTURE:
 * - Primary inference via brain_decide()
 * - Batch processing via brain_decide_batch()
 * - Mirror neuron integration via brain_observe_action()
 * - Decision caching for repeated inputs (thread-safe)
 *
 * PERFORMANCE:
 * - Inference: O(s*n) where s=sparsity, n=active_neurons
 * - Caching: O(1) cache hit, O(s*n) cache miss
 * - Batch: O(m*s*n) where m=batch_size
 *
 * THREAD SAFETY: Functions use brain's internal mutex for cache protection
 */

#ifndef NIMCP_BRAIN_INFERENCE_H
#define NIMCP_BRAIN_INFERENCE_H

#include "core/brain/nimcp_brain.h"
#include "core/brain/nimcp_brain_internal.h"

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// Inference API - Public Functions
//=============================================================================

/**
 * @brief Make decision for input
 *
 * WHY: Primary inference interface
 * Performs forward pass and returns structured decision
 *
 * COMPLEXITY: O(s*n) where s = sparsity, n = active_neurons
 * PERFORMANCE: <1ms for small, ~5ms for medium, ~50ms for large
 * OPTIMIZATION: Caches results for repeated identical inputs
 *
 * @param brain Brain handle
 * @param features Input features
 * @param num_features Feature count
 * @return Decision result (caller must free with brain_free_decision)
 */
brain_decision_t* brain_decide(brain_t brain, const float* features, uint32_t num_features);

/**
 * @brief Free decision result
 *
 * WHY: Proper memory management for decision results
 * Handles all allocated sub-structures
 *
 * COMPLEXITY: O(1)
 *
 * @param decision Decision to free
 */
void brain_free_decision(brain_decision_t* decision);

/**
 * @brief Batch inference
 *
 * WHY: More efficient than individual calls for large batches
 * Enables parallel processing opportunities
 *
 * COMPLEXITY: O(m*s*n) where m = num_inputs
 *
 * @param brain Brain handle
 * @param inputs Array of input vectors
 * @param num_inputs Number of inputs
 * @param features_per_input Features per input
 * @param decisions Output decisions array (allocated by caller)
 * @return true on success
 */
bool brain_decide_batch(brain_t brain, const float** inputs, uint32_t num_inputs,
                        uint32_t features_per_input, brain_decision_t* decisions);

/**
 * @brief Observe action performed by another agent (Phase 10.11)
 *
 * WHAT: Record observed action in mirror neuron system for observational learning
 * WHY:  Enable learning from watching others (imitation, social cognition)
 * HOW:  Convert input features to observed action and send to mirror neurons
 *
 * This is the OBSERVATION PATHWAY for mirror neurons. When the brain observes
 * another agent performing an action, this function records it for learning.
 *
 * USE CASES:
 * - Robot watching human demonstration
 * - Agent observing another agent's behavior
 * - Learning from video/sensor data of actions
 * - Social learning and imitation
 *
 * COMPLEXITY: O(n) where n = num_features
 * THREAD-SAFE: No (requires external synchronization)
 *
 * @param brain Brain handle
 * @param features Observed action features (sensor data, visual features, etc.)
 * @param num_features Number of features
 * @param agent_id ID of agent being observed (must be > 0, as 0 = self)
 * @return true on success, false on error
 */
bool brain_observe_action(brain_t brain, const float* features, uint32_t num_features,
                          uint32_t agent_id);

#ifdef __cplusplus
}
#endif

#endif // NIMCP_BRAIN_INFERENCE_H
