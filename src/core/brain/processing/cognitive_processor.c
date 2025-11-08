//=============================================================================
// cognitive_processor.c - Cognitive Assessment Implementation
//=============================================================================
/**
 * @file cognitive_processor.c
 * @brief Single Responsibility: Apply cognitive assessments to neural output
 *
 * REFACTORING NOTE:
 * Extracted from nimcp_brain.c brain_process_multimodal() (394 lines → ~80 lines)
 * Reason: Apply Single Responsibility Principle - separate cognitive assessment
 *
 * DESIGN:
 * - Each cognitive module assessed independently
 * - Fallback computations if modules not initialized
 * - Pure function: no side effects except annotation writes
 */

#include "core/brain/processing/cognitive_processor.h"
#include "core/brain/nimcp_brain.h"
#include "cognitive/introspection/nimcp_introspection.h"
#include "cognitive/ethics/nimcp_ethics.h"
#include "cognitive/salience/nimcp_salience.h"
#include "cognitive/curiosity/nimcp_curiosity.h"
#include "core/neuron_types/nimcp_neural_logic.h"
#include <string.h>
#include <stdio.h>
#include <math.h>

//=============================================================================
// Internal Brain Structure Access
//=============================================================================

// Forward declaration of brain structure (opaque type)
struct brain_struct {
    // Cognitive modules
    introspection_engine_t introspection;
    ethics_engine_t ethics;
    salience_evaluator_t salience;
    curiosity_engine_t curiosity;
    neural_logic_network_t logic;

    // Feature buffer
    float* integrated_feature_buffer;

    // Configuration
    struct {
        uint32_t num_inputs;
    } config;

    // ... other fields not needed by this module
};

//=============================================================================
// Helper Functions
//=============================================================================

/**
 * @brief Compute fallback confidence from output statistics
 */
static float compute_fallback_confidence(
    const float* output_vector,
    uint32_t output_size,
    uint32_t spikes_generated,
    uint32_t num_inputs)
{
    // Compute output variance
    float output_variance = 0.0f;
    float output_mean = 0.0f;

    for (uint32_t i = 0; i < output_size; i++) {
        output_mean += output_vector[i];
    }
    output_mean /= output_size;

    for (uint32_t i = 0; i < output_size; i++) {
        float diff = output_vector[i] - output_mean;
        output_variance += diff * diff;
    }
    output_variance /= output_size;

    // Higher activity, lower variance → higher confidence
    float confidence = fminf(1.0f, (float)spikes_generated / (num_inputs * 2.0f));
    confidence *= (1.0f - fminf(1.0f, output_variance));

    return confidence;
}

/**
 * @brief Compute fallback salience from max output activation
 */
static float compute_fallback_salience(
    const float* output_vector,
    uint32_t output_size)
{
    float max_activation = 0.0f;
    for (uint32_t i = 0; i < output_size; i++) {
        if (output_vector[i] > max_activation) {
            max_activation = output_vector[i];
        }
    }
    return fminf(1.0f, max_activation);
}

/**
 * @brief Compute fallback novelty from spike deviation
 */
static float compute_fallback_novelty(
    uint32_t spikes_generated,
    uint32_t num_inputs)
{
    float expected_spikes = num_inputs * 0.5f;
    float spike_diff = fabsf((float)spikes_generated - expected_spikes);
    return fminf(1.0f, spike_diff / expected_spikes);
}

/**
 * @brief Check for ethical violations (NaN, inf, extreme values)
 */
static bool check_ethical_output(
    const float* output_vector,
    uint32_t output_size)
{
    for (uint32_t i = 0; i < output_size; i++) {
        if (isnan(output_vector[i]) || isinf(output_vector[i]) ||
            fabsf(output_vector[i]) > 1000.0f) {
            return false;  // Ethical violation
        }
    }
    return true;  // Ethically acceptable
}

//=============================================================================
// API Implementation
//=============================================================================

void cognitive_annotations_init(cognitive_annotations_t* annotations)
{
    if (!annotations) {
        return;
    }

    memset(annotations, 0, sizeof(cognitive_annotations_t));

    annotations->confidence = 0.5f;  // Neutral confidence
    annotations->uncertainty = 0.5f;
    annotations->ethical_approved = true;  // Assume ethical unless proven otherwise
    annotations->salience_score = 0.5f;
    annotations->novelty_score = 0.5f;
    annotations->urgency_score = 0.0f;
    annotations->exploration_bonus = 0.0f;
    annotations->information_gain = 0.0f;
    annotations->logic_valid = true;
}

bool cognitive_process_output(
    const brain_t brain,
    const network_output_t* net_output,
    const float* integrated_features,
    uint32_t integrated_dim,
    uint64_t timestamp_ms,
    cognitive_annotations_t* annotations)
{
    // =========================================================================
    // VALIDATION
    // =========================================================================

    if (!brain || !net_output || !annotations) {
        fprintf(stderr, "cognitive_processor: Invalid parameters\n");
        return false;
    }

    // Initialize output
    cognitive_annotations_init(annotations);

    // =========================================================================
    // INTROSPECTION: Confidence and Uncertainty
    // =========================================================================

    if (brain->introspection) {
        // Use introspection module for sophisticated uncertainty estimation
        brain_uncertainty_t uncertainty = brain_get_uncertainty(
            brain->introspection,
            integrated_features,
            integrated_dim
        );

        annotations->uncertainty = uncertainty.total;
        annotations->confidence = 1.0f - annotations->uncertainty;
    } else {
        // Fallback: Compute confidence from output statistics
        annotations->confidence = compute_fallback_confidence(
            net_output->output_vector,
            net_output->output_size,
            net_output->spikes_generated,
            brain->config.num_inputs
        );
        annotations->uncertainty = 1.0f - annotations->confidence;
    }

    // =========================================================================
    // ETHICS: Validate Output
    // =========================================================================

    if (brain->ethics) {
        // Use ethics module for sophisticated ethical evaluation
        // For now, just check for NaN/inf/extreme values
        annotations->ethical_approved = check_ethical_output(
            net_output->output_vector,
            net_output->output_size
        );
    } else {
        // Fallback: Basic validity check
        annotations->ethical_approved = check_ethical_output(
            net_output->output_vector,
            net_output->output_size
        );
    }

    // =========================================================================
    // SALIENCE: Input Importance (Novelty, Surprise, Urgency)
    // =========================================================================

    if (brain->salience) {
        // Use salience module for sophisticated importance evaluation
        brain_salience_t salience = brain_evaluate_salience_temporal(
            brain->salience,
            integrated_features,
            integrated_dim,
            timestamp_ms
        );

        annotations->salience_score = salience.salience;
        annotations->novelty_score = salience.novelty;
        annotations->urgency_score = salience.surprise;  // Map surprise to urgency
    } else {
        // Fallback: Max output activation as salience
        annotations->salience_score = compute_fallback_salience(
            net_output->output_vector,
            net_output->output_size
        );

        // Fallback novelty from spike deviation
        annotations->novelty_score = compute_fallback_novelty(
            net_output->spikes_generated,
            brain->config.num_inputs
        );

        annotations->urgency_score = 0.0f;  // No urgency without salience module
    }

    // =========================================================================
    // CURIOSITY: Exploration Value
    // =========================================================================

    if (brain->curiosity) {
        // Curiosity engine can compute exploration bonus and information gain
        // Based on novelty and uncertainty
        annotations->exploration_bonus = annotations->novelty_score * annotations->uncertainty;
        annotations->information_gain = annotations->novelty_score;
    } else {
        // Fallback: No exploration bonus without curiosity module
        annotations->exploration_bonus = 0.0f;
        annotations->information_gain = 0.0f;
    }

    // =========================================================================
    // NEURAL LOGIC: Logical Reasoning (Phase 9.0)
    // =========================================================================

    if (brain->logic) {
        // Neural logic gates available for constraint checking / logical inference
        // For now, assume logic is valid (no constraints violated)
        annotations->logic_valid = true;
        // TODO: Add specific logic gate circuits for cognitive constraint checking
    } else {
        // No logic module, assume valid
        annotations->logic_valid = true;
    }

    return true;
}
