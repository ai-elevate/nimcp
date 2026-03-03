//=============================================================================
// nimcp_brain_learning.c - Brain Learning Module Implementation
//=============================================================================
/**
 * @file nimcp_brain_learning.c
 * @brief Learning, training, and reward-based learning implementation
 *
 * WHAT: High-level supervised and reinforcement learning for brain networks
 * WHY:  Separates learning logic from core brain management
 * HOW:  Integrates adaptive network learning with biological enhancements
 *
 * ARCHITECTURE:
 * - Forward pass → loss computation → backpropagation → weight update
 * - Epistemic filtering: Skepticism filter for training data quality
 * - Curiosity-driven LR: Boost learning rate for novel patterns
 * - Biological security: Excitotoxicity monitoring and emergency inhibition
 * - Emotional integration: Joy, remorse, shadow emotions
 * - Memory encoding: Engram formation, consolidation pipeline
 * - Quantum optimization: Periodic weight escape from local minima
 *
 * PERFORMANCE CHARACTERISTICS:
 * - Single example: ~0.1-1ms (small), ~10ms (large networks)
 * - Batch learning: ~m × single_example_time
 * - Quantum annealing: ~50-500ms (run every N steps)
 * - Memory overhead: O(1) per learning call
 *
 * @author NIMCP Development Team
 * @version 1.0
 */

#include "core/brain/learning/nimcp_brain_learning.h"
#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_messages.h"
#include <math.h>
#include <string.h>
#include <ctype.h>
#include <stdbool.h>
#include "plasticity/adaptive/nimcp_adaptive.h"
#include "core/neuralnet/nimcp_neuralnet.h"
#include "core/neuralnet/nimcp_neuron_synapse_access.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/time/nimcp_time.h"
#include "security/nimcp_security.h"
#include "security/nimcp_blood_brain_barrier.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/thread/nimcp_thread.h"
#include "async/nimcp_future.h"
#include "utils/exception/nimcp_exception_macros.h"
#include "core/brain/nimcp_brain_internal.h"
#include "utils/containers/nimcp_hash_table.h"
#include "cognitive/memory/core/nimcp_prime_signature.h"
#include "core/brain/regions/mammillary/nimcp_mammillary.h"

/* Loss history circular buffer size — must match brain_internal.h loss_history[10] */
#define LOSS_HISTORY_SIZE 10

#define LOG_MODULE "core_brain_learning"
#include "utils/fault_tolerance/nimcp_health_agent_macros.h"
#include "utils/bridge/nimcp_bridge_boilerplate.h"
#include "mesh/nimcp_mesh_participant.h"
#include "mesh/nimcp_mesh_adapter.h"
#include "constants/nimcp_buffer_constants.h"

// Multi-network training includes (LNN + CNN + dispatch)
#include "training/nimcp_training_dispatch.h"
#include "lnn/nimcp_lnn.h"
#include "lnn/nimcp_lnn_network.h"
#include "lnn/nimcp_lnn_training.h"
#include "training/nimcp_cnn_training.h"
#include "cognitive/vae/nimcp_vae.h"
#include "cognitive/vae/bridges/nimcp_vae_training_bridge.h"
#include "cognitive/attention/nimcp_attention_plasticity_bridge.h"

// Perceptual cortex + cortical column + predictive coding training integration
#include "perception/nimcp_visual_cortex.h"
#include "perception/nimcp_audio_cortex.h"
#include "core/cortical_columns/nimcp_cortical_column.h"
#include "plasticity/predictive/nimcp_predictive_coding.h"
#include "core/brain_regions/nimcp_brain_regions.h"

BRIDGE_BOILERPLATE_MESH_ONLY(brain_learning, MESH_ADAPTER_CATEGORY_COGNITIVE)


// Import needed from brain internal structure
// Note: These functions access brain internal state, defined in nimcp_brain.c
extern void set_error(const char* format, ...);
extern void brain_clear_error(void);
extern bool ensure_writable_network(brain_t brain);
extern void clear_cache(brain_t brain);

// Import BBB global system (defined in nimcp_brain_init.c)
extern bbb_system_t nimcp_bbb_get_global_system(void);

//=============================================================================
// Internal Helper Functions
//=============================================================================

/**
 * @brief Find or create output label index
 *
 * WHY: Maps string labels to numeric indices
 * Enables human-readable classification
 *
 * COMPLEXITY: O(k) where k = num_existing_labels
 * OPTIMIZATION: Linear search sufficient for small label sets
 *
 * @param brain Brain handle
 * @param label Label string
 * @return Label index
 */
uint32_t nimcp_brain_learning_get_or_create_label_index(brain_t brain, const char* label)
{
    uint32_t result = 0;

    // THREAD SAFETY FIX: Protect label creation with mutex
    // Without this, concurrent threads could:
    // 1. Both see label doesn't exist
    // 2. Both create the same label at same index
    // 3. Result in buffer overflow past output_labels capacity → heap corruption
    nimcp_platform_mutex_lock(&brain->cache_mutex);

    // O(1) hash table lookup (replaces O(k) linear strcmp scan)
    if (brain->label_index) {
        void* found = hash_table_lookup_string(brain->label_index, label);
        if (found) {
            result = *(uint32_t*)found;
            goto done;
        }
    }

    // Guard: Check capacity — labels exceed output neurons
    if (brain->num_output_labels >= brain->config.num_outputs) {
        // Hash overflow labels across existing outputs instead of always index 0
        // (M2): Use djb2 consistent hash so labels get deterministic, well-distributed
        // assignments even when beyond capacity. Avoids the pathological case where
        // simple modulo of a monotonic counter clusters labels into low indices.
        uint32_t h = 5381;
        for (const char* p = label; *p; p++) {
            h = ((h << 5) + h) + (unsigned char)*p;
        }
        result = h % brain->config.num_outputs;
        // Still insert into hash table for O(1) future lookups of overflow labels
        if (brain->label_index)
            hash_table_insert_string(brain->label_index, label, &result, sizeof(uint32_t));
        LOG_WARN("Label overflow: '%s' hashed to output %u (num_labels=%u >= num_outputs=%u)",
                 label, result, brain->num_output_labels, brain->config.num_outputs);
        goto done;
    }

    // (M2): Early warning when approaching label capacity — gives callers a
    // chance to notice before overflow causes hash collisions.
    if (brain->num_output_labels >= (brain->config.num_outputs * 9u / 10u) &&
        brain->num_output_labels == (brain->config.num_outputs * 9u / 10u)) {
        LOG_WARN("Label capacity at 90%%: %u of %u output slots used. "
                 "New labels beyond capacity will be hash-mapped to existing outputs.",
                 brain->num_output_labels, brain->config.num_outputs);
    }

    // Create new label (use nimcp_malloc to match nimcp_free in brain_destroy)
    size_t label_len = strlen(label);
    brain->output_labels[brain->num_output_labels] = nimcp_malloc(label_len + 1);
    if (!brain->output_labels[brain->num_output_labels]) {
        result = 0;
        goto done;
    }
    memcpy(brain->output_labels[brain->num_output_labels], label, label_len + 1);
    result = brain->num_output_labels;

    // Insert into hash table for O(1) future lookups
    if (brain->label_index)
        hash_table_insert_string(brain->label_index, label, &result, sizeof(uint32_t));

    brain->num_output_labels++;

done:
    nimcp_platform_mutex_unlock(&brain->cache_mutex);
    return result;
}

/**
 * @brief Convert label to one-hot encoded output vector
 *
 * WHY: Transforms string labels to neural network targets
 * One-hot encoding standard for classification
 *
 * COMPLEXITY: O(n) where n = num_outputs
 *
 * @param brain Brain handle
 * @param label Label string
 * @param output Output buffer
 * @param confidence Confidence value for label
 */
void nimcp_brain_learning_label_to_output(brain_t brain, const char* label,
                                          float* output, float confidence)
{
    uint32_t label_idx = nimcp_brain_learning_get_or_create_label_index(brain, label);

    memset(output, 0, brain->config.num_outputs * sizeof(float));
    output[label_idx] = 1.0f;  /* Always one-hot; confidence scales LR, not target */
}

/**
 * WHAT: Adapt learning rate based on loss trend (Phase 11: Simple Meta-Learning)
 * WHY:  Accelerate when loss decreasing, slow down when loss increasing
 * HOW:  Track loss in rolling window, compute trend, adjust LR
 *
 * COMPLEXITY: O(1)
 *
 * BIOLOGICAL BASIS:
 * - Meta-learning: "learning to learn"
 * - Homeostatic regulation of synaptic plasticity
 */
void nimcp_brain_learning_adapt_learning_rate(brain_t brain, float current_loss)
{
    // Guard: NULL check
    if (!brain) {
        return;
    }

    // Guard: Initialize base_learning_rate on first call
    if (brain->base_learning_rate == 0.0F) {
        brain->base_learning_rate = brain->config.learning_rate;
    }

    // Store current loss in circular buffer
    brain->loss_history[brain->loss_history_index] = current_loss;
    brain->loss_history_index = (brain->loss_history_index + 1) % LOSS_HISTORY_SIZE;
    if (brain->loss_history_count < LOSS_HISTORY_SIZE) {
        brain->loss_history_count++;
    }

    // Need at least 3 samples to compute trend
    if (brain->loss_history_count < 3) {
        return;
    }

    // Compute loss trend: recent avg vs older avg
    // Use modular arithmetic to correctly index the circular buffer.
    // Oldest entry is at loss_history_index (the next write position),
    // so we walk forward from there using (head + offset) % capacity.
    float recent_avg = 0.0F;
    float older_avg = 0.0F;
    uint32_t half = brain->loss_history_count / 2;
    uint32_t head = (brain->loss_history_index + LOSS_HISTORY_SIZE - brain->loss_history_count) % LOSS_HISTORY_SIZE;

    // Older half (oldest entries first)
    for (uint32_t i = 0; i < half; i++) {
        older_avg += brain->loss_history[(head + i) % LOSS_HISTORY_SIZE];
    }
    older_avg /= half;

    // Recent half (newest entries)
    for (uint32_t i = half; i < brain->loss_history_count; i++) {
        recent_avg += brain->loss_history[(head + i) % LOSS_HISTORY_SIZE];
    }
    recent_avg /= (brain->loss_history_count - half);

    // Compute trend
    float trend = recent_avg - older_avg;

    // Adapt learning rate
    if (trend < -0.01F) {
        brain->config.learning_rate *= 1.05F;  // Accelerate
    } else if (trend > 0.01F) {
        brain->config.learning_rate *= 0.9F;   // Slow down
    }

    // Bounds: [0.1x, 10x] of base rate
    float min_lr = brain->base_learning_rate * 0.1F;
    float max_lr = brain->base_learning_rate * 10.0F;
    if (brain->config.learning_rate < min_lr) {
        brain->config.learning_rate = min_lr;
    }
    if (brain->config.learning_rate > max_lr) {
        brain->config.learning_rate = max_lr;
    }
}

/**
 * @brief Energy function for quantum annealing weight optimization
 *
 * WHAT: Compute L2 regularization energy for given weights
 * WHY:  Simple proxy energy function for weight optimization
 * HOW:  Sum of squared weights, normalized by dimension
 *
 * NOTE: Full implementation would use validation loss
 *
 * @param weights Weight vector
 * @param dim Vector dimension
 * @param user_data Unused
 * @return Energy (lower is better)
 */
float nimcp_brain_learning_quantum_weight_energy(const float* weights, uint32_t dim,
                                                  void* user_data)
{
    (void)user_data;  // Unused
    float energy = 0.0F;
    for (uint32_t i = 0; i < dim; i++) {
        energy += weights[i] * weights[i];
    }
    return energy / (float)dim;  // Normalized
}

//=============================================================================
// Public Learning API
//=============================================================================

/**
 * @brief Learn from a dense target vector (distillation / generative training)
 *
 * WHY: Enables teacher-model distillation and semantic embedding training
 * HOW: Builds training_example_t with dense target, uses LEARN_MODE_DISTILLATION
 *
 * @param brain Brain handle
 * @param features Input features
 * @param num_features Feature count
 * @param target Dense target output vector
 * @param target_size Target vector size
 * @param label Optional semantic label (can be NULL)
 * @param confidence Training weight
 * @return Loss value or -1 on error
 */
float brain_learn_vector(brain_t brain, const float* features, uint32_t num_features,
                          const float* target, uint32_t target_size,
                          const char* label, float confidence)
{
    if (!brain || !features || !target) {
        set_error("Invalid parameters to brain_learn_vector");
        return -1.0f;
    }

    if (num_features != brain->config.num_inputs) {
        set_error("Feature count mismatch: expected %u, got %u",
                  brain->config.num_inputs, num_features);
        return -1.0f;
    }

    if (target_size != brain->config.num_outputs) {
        set_error("Target size mismatch: expected %u, got %u",
                  brain->config.num_outputs, target_size);
        return -1.0f;
    }

    /* Validate features for NaN/Inf */
    for (uint32_t i = 0; i < num_features; i++) {
        if (isnan(features[i]) || isinf(features[i])) {
            set_error("Invalid feature value at index %u: NaN or Inf", i);
            return -1.0f;
        }
    }

    /* Validate target for NaN/Inf */
    for (uint32_t i = 0; i < target_size; i++) {
        if (isnan(target[i]) || isinf(target[i])) {
            set_error("Invalid target value at index %u: NaN or Inf", i);
            return -1.0f;
        }
    }

    if (!ensure_writable_network(brain)) {
        return -1.0f;
    }

    /* Build training example with dense target directly — no one-hot conversion */
    training_example_t example = {
        .input = (float*)features,
        .input_size = num_features,
        .target = (float*)target,
        .target_size = target_size,
        .confidence = confidence
    };
    if (label) {
        strncpy(example.label, label, sizeof(example.label) - 1);
    } else {
        example.label[0] = '\0';
    }

    /* Use fast training path (distillation mode) */
    float loss = adaptive_network_learn(brain->network, &example,
                                         LEARN_MODE_DISTILLATION,
                                         brain->config.learning_rate);
    nimcp_brain_learning_adapt_learning_rate(brain, loss);
    __atomic_fetch_add(&brain->stats.total_learning_steps, 1, __ATOMIC_RELAXED);

    /* Accumulate sleep pressure */
    if (brain->sleep_system && brain->config.enable_sleep_wake_cycle) {
        sleep_accumulate_pressure(brain->sleep_system, 1);
    }

    clear_cache(brain);
    return loss;
}


/**
 * @brief Learn from single labeled example
 *
 * WHY: Primary learning interface - supervised learning
 * Updates network weights to match label
 *
 * COMPLEXITY: O(s*n) where s = sparsity, n = active_neurons
 * PERFORMANCE: ~0.1-1ms for small networks, ~10ms for large
 *
 * @param brain Brain handle
 * @param features Input features
 * @param num_features Feature count
 * @param label Target label
 * @param confidence Training weight
 * @return Loss value or -1 on error
 */
float brain_learn_example(brain_t brain, const float* features, uint32_t num_features,
                          const char* label, float confidence)
{
    // Guard: Validate parameters
    if (!brain || !features || !label) {
        set_error("Invalid parameters to brain_learn_example");
        return -1.0F;
    }

    // Guard: Check feature dimension
    if (num_features != brain->config.num_inputs) {
        set_error("Feature count mismatch: expected %u, got %u", brain->config.num_inputs,
                  num_features);
        return -1.0F;
    }

    // ========================================================================
    // SECURITY VALIDATION: Check for NaN/Inf in features
    // ========================================================================
    // WHAT: Validate all feature values for NaN/Inf before processing
    // WHY:  NaN/Inf can indicate adversarial attacks or corrupted data
    // HOW:  Check each feature value and reject if invalid
    for (uint32_t i = 0; i < num_features; i++) {
        if (isnan(features[i]) || isinf(features[i])) {
            set_error("Invalid feature value at index %u: NaN or Inf detected", i);
            LOG_WARN("Security: Rejected learning example with NaN/Inf feature[%u]", i);
            return -1.0F;
        }
    }

    // ========================================================================
    // SECURITY VALIDATION: Validate label for malicious patterns
    // ========================================================================
    // WHAT: Check label string for SQL injection, format strings, code injection
    // WHY:  Labels are used in logging and may be stored - must be sanitized
    // HOW:  Use Blood-Brain Barrier (BBB) validation to detect malicious patterns
    //
    // NOTE: We use BBB validation here because it checks for SQL injection,
    //       format strings, XSS, and other code injection attacks. The
    //       nimcp_security_validate_input is designed for LLM prompt injection.
    //
    // FALLBACK: If BBB is not initialized, perform basic pattern matching
    bbb_system_t bbb = nimcp_bbb_get_global_system();
    if (bbb && bbb_system_is_enabled(bbb)) {
        bbb_validation_result_t bbb_result;
        if (!bbb_validate_string(bbb, label, &bbb_result)) {
            set_error("Invalid label: %s (threat: %s, severity: %s)",
                      bbb_result.reason,
                      bbb_threat_type_name(bbb_result.threat),
                      bbb_severity_name(bbb_result.severity));
            LOG_WARN("Security: Rejected learning example with malicious label: %s", label);
            return -1.0F;
        }
    } else {
        // Fallback validation when BBB is not available
        // Check for common malicious patterns
        const char* dangerous_patterns[] = {
            "'; drop ", "'; delete ", "'; insert ", "'; update ",
            "; drop ", "; delete ", "; insert ", "; update ",
            "%n", "%s%s%s", "%x%x%x", "%p%p%p",
            "<script", "javascript:", "onerror=",
            NULL
        };

        // Convert label to lowercase for case-insensitive matching
        size_t label_len = strlen(label);
        char* label_lower = nimcp_malloc(label_len + 1);
        if (!label_lower) {
            set_error("Memory allocation failed for security validation");
            return -1.0F;
        }

        for (size_t i = 0; i < label_len; i++) {
            label_lower[i] = tolower((unsigned char)label[i]);
        }
        label_lower[label_len] = '\0';

        bool found_threat = false;
        for (int i = 0; dangerous_patterns[i] != NULL && !found_threat; i++) {
            if (strstr(label_lower, dangerous_patterns[i])) {
                found_threat = true;
            }
        }

        nimcp_free(label_lower);

        if (found_threat) {
            set_error("Invalid label: malicious pattern detected");
            LOG_WARN("Security: Rejected learning example with malicious label: %s", label);
            return -1.0F;
        }
    }

    // Phase 2: Ensure network is writable (trigger COW if needed)
    if (!ensure_writable_network(brain)) {
        return -1.0F;  // Error already set by ensure_writable_network
    }

    // Convert label to target output (use pre-allocated scratch if available)
    bool target_from_scratch = false;
    float* target = NULL;
    if (brain->learn_scratch.target && brain->learn_scratch.target_cap >= brain->config.num_outputs) {
        target = brain->learn_scratch.target;
        target_from_scratch = true;
    } else {
        target = nimcp_malloc(brain->config.num_outputs * sizeof(float));
    }
    if (!target) {
        set_error("Failed to allocate target vector (%u outputs)", brain->config.num_outputs);
        NIMCP_THROW(NIMCP_ERROR_NO_MEMORY,
                    "failed to allocate target vector for brain learning");
        return -1.0F;
    }
    nimcp_brain_learning_label_to_output(brain, label, target, confidence);

    // Create training example
    training_example_t example = {.input = (float*) features,
                                  .input_size = num_features,
                                  .target = target,
                                  .target_size = brain->config.num_outputs,
                                  .confidence = confidence};
    strncpy(example.label, label, sizeof(example.label) - 1);

    // ========================================================================
    // FAST TRAINING MODE: Skip all biological subsystems for bulk throughput
    // ========================================================================
    // When fast_training_mode is enabled, we only do:
    //   1. Input validation (already done above)
    //   2. adaptive_network_learn() (GPU forward + parallel backprop)
    //   3. Post-learning MSE computation
    //   4. Adaptive LR + statistics
    // This gives 5-10x speedup by skipping ~25 biological subsystems.
    //
    // NOTE (M1): Fast training mode feeds raw float features directly to the
    // adaptive network, skipping Poisson spike encoding. This is intentional —
    // the adaptive network handles continuous floats natively. The spike
    // encoding path is only relevant for the biological SNN pipeline which
    // fast mode bypasses entirely. Both train and eval use the same float
    // path under fast mode, so there is no train/eval encoding mismatch.
    if (brain->config.fast_training_mode) {
        float fast_loss = adaptive_network_learn(brain->network, &example,
                                                  LEARN_MODE_SUPERVISED,
                                                  brain->config.learning_rate);

        // Use loss directly from adaptive_network_learn — skip second forward pass.
        // The learn() call already computes loss internally; a second forward pass
        // through 1.5M neurons would double the per-example time for minimal benefit.
        nimcp_brain_learning_adapt_learning_rate(brain, fast_loss);
        __atomic_fetch_add(&brain->stats.total_learning_steps, 1, __ATOMIC_RELAXED);

        // Track label-match accuracy by reading output neuron states directly.
        // After adaptive_network_learn (GPU path), sync_activations wrote the
        // forward pass outputs to neuron->state. Backprop only modifies weights
        // and biases (not states), so reading states post-learn is safe.
        {
            neural_network_t nn = adaptive_network_get_base_network(brain->network);
            uint32_t total_n = nn ? neural_network_get_num_neurons(nn) : 0;
            uint32_t num_out = brain->config.num_outputs;
            if (nn && total_n > num_out) {
                uint32_t out_start = total_n - num_out;

                // Argmax of prediction — scan only trained labels
                uint32_t label_range = brain->num_output_labels > 0
                    ? brain->num_output_labels : num_out;
                if (label_range > num_out) label_range = num_out;

                uint32_t pred_argmax = 0;
                float pred_max = -1e30f;
                for (uint32_t i = 0; i < label_range; i++) {
                    neuron_t* on = neural_network_get_neuron(nn, out_start + i);
                    if (on && on->state > pred_max) {
                        pred_max = on->state;
                        pred_argmax = i;
                    }
                }
                // Argmax of target (one-hot)
                uint32_t tgt_argmax = 0;
                for (uint32_t i = 0; i < num_out; i++) {
                    if (target[i] > 0.5f) { tgt_argmax = i; break; }
                }
                float match_val = (pred_argmax == tgt_argmax) ? 1.0f : 0.0f;
                brain->stats.running_accuracy =
                    brain->stats.running_accuracy * 0.99f + match_val * 0.01f;
            }
        }

        // Accumulate sleep pressure even in fast mode — needed for sleep/wake cycle
        if (brain->sleep_system && brain->config.enable_sleep_wake_cycle) {
            sleep_accumulate_pressure(brain->sleep_system, 1);
        }

        clear_cache(brain);

        if (!target_from_scratch) nimcp_free(target);
        return fast_loss;
    }

    // ========================================================================
    // Phase 11: EPISTEMIC FILTERING - Apply Skepticism to Training Data
    // ========================================================================
    // WHAT: Evaluate training label for epistemic quality before learning
    // WHY:  Prevent learning from low-quality, biased, or false claims
    // HOW:  Use epistemic filter to assess claim, reduce confidence if suspicious
    //
    // BIOLOGICAL BASIS:
    // - Critical thinking and skepticism (prefrontal cortex)
    // - Source monitoring (distinguishing fact from opinion)
    // - Metacognitive filtering (evaluating information quality)
    //
    // COGNITIVE BENEFITS:
    // - Prevents conspiracy-theory-like reasoning
    // - Detects and mitigates cognitive biases
    // - Distinguishes facts from opinions
    // - Applies "extraordinary claims require extraordinary evidence"
    //
    // COMPLEXITY: O(1)
    float epistemic_confidence_multiplier = 1.0F;  // Default: no adjustment
    if (brain->epistemic && confidence < 1.0F) {
        // Initialize evidence structure (assume moderate quality by default)
        claim_evidence_t evidence;
        epistemic_evidence_init(&evidence);

        // Default assumptions for training data (conservative)
        evidence.evidence_quality = EVIDENCE_MODERATE;  // Training data assumed moderate quality
        evidence.plausibility = PLAUSIBLE_NEUTRAL;      // Don't assume plausibility
        evidence.num_sources = 1;                       // Single source (training dataset)
        evidence.is_falsifiable = true;                 // Assume falsifiable
        evidence.expert_consensus = 0.5F;               // Unknown consensus

        // Assess the claim
        epistemic_assessment_t assessment;
        epistemic_assessment_init(&assessment);

        // Prior probability based on input variance (novel = less certain)
        float prior_prob = 1.0F - brain->last_novelty_score;  // High novelty = lower prior

        if (epistemic_assess_claim(brain->epistemic, label, prior_prob, &evidence, &assessment)) {
            // Apply epistemic quality to learning confidence
            // Low epistemic quality → reduce learning strength
            epistemic_confidence_multiplier = assessment.epistemic_quality;

            // If claim is highly suspicious (conspiracy-like, biased), reduce further
            if (assessment.num_biases_detected > 0) {
                // Each detected bias reduces confidence by 10%
                float bias_penalty = assessment.num_biases_detected * 0.1F;
                epistemic_confidence_multiplier *= fmaxf(0.1F, 1.0F - bias_penalty);
            }

            // Check conspiracy pattern (additional safety)
            float conspiracy_score = epistemic_check_conspiracy_pattern(brain->epistemic, label, &evidence);
            if (conspiracy_score > 0.7F) {
                // High conspiracy score → severely reduce confidence
                epistemic_confidence_multiplier *= 0.2F;  // Only learn 20% strength
            }

            // Update example confidence with epistemic multiplier
            example.confidence *= epistemic_confidence_multiplier;

            // Ensure minimum confidence (don't completely ignore, but learn weakly)
            if (example.confidence < 0.01F) {
                example.confidence = 0.01F;
            }
        }
    }

    // ========================================================================
    // Phase 11: CURIOSITY-DRIVEN LEARNING RATE BOOST (40% faster on novelty)
    // ========================================================================
    // WHAT: Boost learning rate for novel inputs based on curiosity drive
    // WHY:  Novel patterns should be learned faster (exploration bonus)
    // HOW:  Multiply learning rate by (1.0 + curiosity_drive * novelty * 0.4)
    //
    // BIOLOGICAL BASIS:
    // - Dopamine modulates synaptic plasticity (Schultz, 1998)
    // - Novelty enhances memory consolidation (Lisman & Grace, 2005)
    // - Exploration bonus in reinforcement learning
    //
    // COGNITIVE BENEFITS: 40% faster learning on novel patterns (from audit)
    //
    // COMPLEXITY: O(1)
    float effective_learning_rate = brain->config.learning_rate;
    if (brain->curiosity && brain->config.enable_curiosity) {
        // Compute learning rate boost: base + (curiosity × novelty × 40%)
        // Example: LR=0.01, curiosity=0.8, novelty=0.7 → boost=0.224
        //          effective_LR = 0.01 × (1 + 0.224) = 0.01224 (+22.4%)
        float curiosity_boost = brain->last_curiosity_drive * brain->last_novelty_score * 0.4F;
        effective_learning_rate *= (1.0F + curiosity_boost);

        // Cap boost at 2x to prevent instability
        float max_lr = brain->config.learning_rate * 2.0F;
        if (effective_learning_rate > max_lr) {
            effective_learning_rate = max_lr;
        }
    }

    // ========================================================================
    // PHASE 3.5: PERCEPTUAL CORTEX FEATURE EXTRACTION (pre-learning)
    // ========================================================================
    // WHAT: Enable training mode on sensory cortices, extract perceptual state
    // WHY:  Perceptual confidence modulates how aggressively we learn —
    //       high-confidence sensory input = trust the gradient more.
    //       Training mode caches activations for gradient feedback in Phase 8.
    // HOW:  Set training mode, query training state, modulate LR by confidence.
    //
    // BIOLOGICAL BASIS:
    // - V1/A1 training mode ≈ neuromodulatory "attend and encode" state
    // - Perceptual confidence gates cortical learning rate (Feldman 2010)
    // - Top-down attention increases V1 gain → stronger LTP
    float perceptual_confidence = 1.0f;

    // Visual cortex: enable training mode and extract confidence
    if (brain->visual_cortex) {
        visual_cortex_set_training_mode(brain->visual_cortex, true);
        visual_training_state_t vstate;
        memset(&vstate, 0, sizeof(vstate));
        if (visual_cortex_get_training_state(brain->visual_cortex, &vstate) == 0
            && vstate.valid) {
            // Blend visual confidence into perceptual confidence
            perceptual_confidence *= (0.5f + 0.5f * vstate.confidence);
        }
    }

    // Audio cortex: enable training mode and extract quality
    if (brain->audio_cortex) {
        audio_cortex_set_training_mode(brain->audio_cortex, true);
        audio_training_state_t astate;
        memset(&astate, 0, sizeof(astate));
        if (audio_cortex_get_training_state(brain->audio_cortex, &astate) == 0
            && astate.valid) {
            // Blend audio quality into perceptual confidence
            perceptual_confidence *= (0.5f + 0.5f * astate.quality);
        }
    }

    // Modulate effective learning rate by perceptual confidence
    // Low confidence → dampen LR to avoid learning from noisy input
    effective_learning_rate *= perceptual_confidence;

    // ========================================================================
    // PHASE 4.5: ATTENTION-GUIDED FEATURE PROCESSING
    // ========================================================================
    // WHAT: Run input through multihead attention for selective feature focus
    // WHY:  Attention determines which features are most relevant for learning.
    //       Attended features receive amplified learning (biased competition).
    //       Thalamic gate provides top-down executive control over learning focus.
    // HOW:  Set gate from confidence/novelty, forward pass through attention,
    //       blend attended output with original features, modulate LR.
    //
    // BIOLOGICAL BASIS:
    // - Desimone & Duncan (1995): Biased competition model of attention
    // - Attended stimuli induce stronger LTP (Noudoost et al., 2010)
    // - Thalamic gating of sensory input (Sherman & Guillery, 2002)
    // - Top-down attention from PFC modulates V1-V4 learning rates
    float* attended_features = NULL;
    bool attended_from_scratch = false;
    if (brain->multihead_attention && brain->attention_training_enabled) {
        // Set thalamic gate: confidence gates executive attention,
        // novelty opens the gate wider for exploration
        float gate_signal = fminf(1.0f, confidence * 0.5f + brain->last_novelty_score * 0.5f);
        multihead_attention_set_gate(brain->multihead_attention, gate_signal);

        // Run attention forward pass (single token: [1 × input_dim])
        // Even with seq_len=1, this applies Q/K/V projections + thalamic gate
        if (brain->learn_scratch.attended_features && brain->learn_scratch.features_cap >= num_features) {
            attended_features = brain->learn_scratch.attended_features;
            attended_from_scratch = true;
        } else {
            attended_features = nimcp_malloc(num_features * sizeof(float));
        }
        if (attended_features) {
            bool attn_ok = multihead_attention_forward(
                brain->multihead_attention,
                features,            // [1 × num_features]
                1,                   // sequence_length = 1
                NULL,                // no external salience
                attended_features    // [1 × num_features]
            );

            if (attn_ok) {
                // Get overall attention strength for LR modulation
                float attn_strength = multihead_attention_get_strength(
                    brain->multihead_attention);
                brain->last_attention_strength = attn_strength;

                // Blend attended with original: preserve raw signal integrity
                // while incorporating attention's selective emphasis
                float blend = attn_strength * 0.4f; // Up to 40% attention influence
                for (uint32_t i = 0; i < num_features; i++) {
                    attended_features[i] = (1.0f - blend) * features[i]
                                         + blend * attended_features[i];
                }

                // Use attended features for the learning forward/backward pass
                // (downstream phases using 'features' directly still get raw input
                //  for memory encoding — biologically correct separation)
                example.input = attended_features;

                // Attention boost: highly attended inputs learn up to 20% faster
                effective_learning_rate *= (1.0f + attn_strength * 0.2f);
            } else {
                if (!attended_from_scratch) nimcp_free(attended_features);
                attended_features = NULL;
            }
        }
    }

    // ========================================================================
    // PHASE 5.0: VAE ANOMALY DETECTION & LATENT PRIMING
    // ========================================================================
    // WHAT: Encode input through VAE to get anomaly score and latent representation
    // WHY:  Anomaly score identifies out-of-distribution inputs (hard examples).
    //       Latent representation provides compressed features for downstream use.
    //       Free energy from ELBO directly implements the Free Energy Principle.
    // HOW:  Forward pass through VAE encoder → mu, sigma → anomaly score.
    //       Store anomaly score and free energy on brain struct for later use.
    //
    // BIOLOGICAL BASIS:
    // - Predictive coding in visual cortex (Rao & Ballard 1999)
    // - Surprise/prediction error drives attention (Friston 2010)
    // - Generative models in hippocampus for novelty detection
    if (brain->vae_system && brain->vae_enabled) {
        vae_system_t* vae = (vae_system_t*)brain->vae_system;

        // Create input tensor from features (rank-2: [1, num_features] for batch_size=1)
        // VAE expects dims[0]=batch_size; rank-1 would misinterpret num_features as batch
        uint32_t vae_dims[2] = { 1, num_features };
        nimcp_tensor_t* vae_input = nimcp_tensor_create(vae_dims, 2, NIMCP_DTYPE_F32);
        if (vae_input) {
            float* vae_data = (float*)nimcp_tensor_data(vae_input);
            if (vae_data) {
                memcpy(vae_data, features, num_features * sizeof(float));
            }

            // Compute anomaly score (high = out-of-distribution)
            float anomaly_score = 0.0f;
            if (vae_compute_anomaly_score(vae, vae_input, &anomaly_score) == 0) {
                brain->last_vae_anomaly_score = anomaly_score;

                // Boost learning rate for anomalous inputs (harder = learn more)
                if (anomaly_score > 2.0f) {
                    effective_learning_rate *= fminf(1.5f, 1.0f + anomaly_score * 0.1f);
                }
            }

            // Get free energy (negative ELBO) for FEP integration
            float fe = vae_get_free_energy(vae);
            if (!isnan(fe)) {
                brain->last_vae_free_energy = fe;
            }

            nimcp_tensor_destroy(vae_input);
        }
    }

    // ========================================================================
    // PHASE 5.1: ENGRAM RECALL — PRIME NETWORK WITH SIMILAR MEMORIES
    // ========================================================================
    // WHAT: Recall similar memories before learning to prime hidden neurons
    // WHY:  Biological brains recall related knowledge before encoding new info
    //       This creates contextual priming that improves learning
    // HOW:  Query engram system with input features as cue, inject recalled
    //       activations into hidden layer to give forward pass a "head start"
    //
    // BIOLOGICAL BASIS:
    // - Hippocampal pattern completion (Marr 1971)
    // - Contextual priming in associative memory (Anderson 1983)
    // - Prior knowledge facilitates new learning (Tse et al. 2007)
    if (brain->engram_system) {
        bool cue_from_scratch = false;
        uint32_t* cue_neurons;
        if (brain->learn_scratch.cue_neurons && brain->learn_scratch.features_cap >= num_features) {
            cue_neurons = brain->learn_scratch.cue_neurons;
            cue_from_scratch = true;
        } else {
            cue_neurons = nimcp_malloc(num_features * sizeof(uint32_t));
        }
        if (cue_neurons) {
            // Map features to cue neuron IDs
            for (uint32_t i = 0; i < num_features; i++) {
                cue_neurons[i] = i;
            }

            #define RECALL_MAX_NEURONS 100
            uint32_t recalled_neurons[RECALL_MAX_NEURONS];
            float recalled_activations[RECALL_MAX_NEURONS];
            memset(recalled_neurons, 0, sizeof(recalled_neurons));
            memset(recalled_activations, 0, sizeof(recalled_activations));
            float recall_confidence = 0.0f;

            uint64_t recalled_id = engram_recall(
                brain->engram_system,
                cue_neurons, num_features,
                recalled_neurons, recalled_activations,
                RECALL_MAX_NEURONS, &recall_confidence
            );

            // If recall succeeded with decent confidence, prime network neurons
            if (recalled_id != 0 && recall_confidence > 0.3f) {
                neural_network_t base_net = adaptive_network_get_base_network(brain->network);
                if (base_net) {
                    // Inject recalled activations as priming (blend with current state)
                    float prime_strength = recall_confidence * 0.3f; // 30% of recall confidence
                    for (uint32_t r = 0; r < RECALL_MAX_NEURONS; r++) {
                        uint32_t nid = recalled_neurons[r];
                        if (nid < neural_network_get_num_neurons(base_net)) {
                            neuron_t* neuron = neural_network_get_neuron(base_net, nid);
                            if (neuron) {
                                neuron->state += prime_strength * recalled_activations[r];
                            }
                        }
                    }
                }
                // Trigger reconsolidation on recalled engram (biologically realistic)
                engram_trigger_reconsolidation(brain->engram_system, recalled_id);
            }

            if (!cue_from_scratch) nimcp_free(cue_neurons);
        }
    }

    // ========================================================================
    // PHASE 5.2: NEUROMODULATOR PRE-COMPUTATION
    // ========================================================================
    // WHAT: Set neuromodulator levels based on learning context
    // WHY:  Dopamine gates STDP/eligibility, ACh enhances encoding, NE modulates attention
    // HOW:  Compute levels from curiosity, confidence, and recent loss trend
    //
    // BIOLOGICAL BASIS:
    // - DA: Reward prediction error (Schultz 1997)
    // - ACh: Encoding mode in hippocampus (Hasselmo 2006)
    // - NE: Arousal and attention (Aston-Jones & Cohen 2005)
    if (brain->neuromodulator_system) {
        // Dopamine: based on expected reward (inverse of recent loss)
        float da_level = 0.5f; // baseline
        if (brain->loss_history_count >= LOSS_HISTORY_SIZE) {
            // Use loss trend: compare oldest vs newest entry in circular buffer.
            // Oldest entry is at loss_history_index (next write position).
            // Newest entry is at (loss_history_index + LOSS_HISTORY_SIZE - 1) % LOSS_HISTORY_SIZE.
            uint32_t oldest_idx = brain->loss_history_index;
            uint32_t newest_idx = (brain->loss_history_index + LOSS_HISTORY_SIZE - 1) % LOSS_HISTORY_SIZE;
            float loss_trend = brain->loss_history[oldest_idx] - brain->loss_history[newest_idx];
            if (loss_trend > 0.0f) {
                da_level = fminf(1.0f, 0.5f + loss_trend * 2.0f); // Boost DA when improving
            } else {
                da_level = fmaxf(0.1f, 0.5f + loss_trend * 2.0f); // Suppress DA when worsening
            }
        }

        // Acetylcholine: high during novel inputs (encoding mode)
        float ach_level = 0.5f;
        if (brain->last_novelty_score > 0.0f) {
            ach_level = fminf(1.0f, 0.5f + brain->last_novelty_score * 0.5f);
        }

        // Norepinephrine: arousal/difficulty signal
        float ne_level = fminf(1.0f, confidence); // Higher confidence = more engaged

        // Set levels (these will be read by STDP/eligibility during hybrid learning)
        neuromodulator_set_level(brain->neuromodulator_system, NEUROMOD_DOPAMINE, da_level);
        neuromodulator_set_level(brain->neuromodulator_system, NEUROMOD_ACETYLCHOLINE, ach_level);
        neuromodulator_set_level(brain->neuromodulator_system, NEUROMOD_NOREPINEPHRINE, ne_level);
    }

    // ========================================================================
    // PHASE 5.3: OSCILLATION THETA GATING
    // ========================================================================
    // WHAT: Check theta oscillation phase to optimize memory encoding timing
    // WHY:  Memory encoding is enhanced during theta peak (Hasselmo et al. 2002)
    // HOW:  Read theta phase from oscillations, modulate learning rate
    //
    // BIOLOGICAL BASIS:
    // - Theta-phase coupling in hippocampus (Buzsaki 2002)
    // - Encoding at theta peak, retrieval at theta trough (Hasselmo 2005)
    if (brain->oscillations) {
        // Theta modulation: boost learning during encoding-favorable phase
        // Simplified theta gate: sinusoidal modulation based on step count
        // Full implementation would use actual phase from oscillation analyzer
        float theta_freq = 6.0f; // 6 Hz theta
        float phase = fmodf((float)brain->stats.total_learning_steps * 0.01f * theta_freq, 1.0f);
        float theta_modulation = 0.8f + 0.2f * sinf(phase * 2.0f * 3.14159265f);
        effective_learning_rate *= theta_modulation;
    }

    // Learn using adaptive network with curiosity-modulated learning rate
    // HYBRID mode: Supervised backprop + biological plasticity (STDP/BCM/eligibility)
    // This enables neuromodulator-gated plasticity alongside gradient descent
    uint64_t _t_learn_start = nimcp_time_get_us();
    float network_loss = adaptive_network_learn(brain->network, &example, LEARN_MODE_HYBRID,
                                                effective_learning_rate);
    uint64_t _t_learn_us = nimcp_time_get_us() - _t_learn_start;

    // ========================================================================
    // BIOLOGICAL SECURITY: Monitor for attacks after learning (Phase 11)
    // ========================================================================
    if (brain->config.enable_bio_security) {
        nimcp_activity_stats_t activity_stats;
        nimcp_bio_attack_type_t attack = nimcp_security_monitor_excitotoxicity(
            adaptive_network_get_base_network(brain->network),
            &activity_stats
        );

        if (attack == NIMCP_BIO_ATTACK_EXCITOTOXICITY) {
            // CRITICAL: Excitotoxicity detected
            set_error("SECURITY: Excitotoxicity attack detected (%.1f%% activity)",
                     activity_stats.activity_ratio * 100.0F);

            if (brain->config.emergency_inhibit_on_attack) {
                // Emergency response
                nimcp_security_emergency_inhibit(adaptive_network_get_base_network(brain->network));
            }

            if (!attended_from_scratch) nimcp_free(attended_features);
            if (!target_from_scratch) nimcp_free(target);
            return -1.0F;  // Abort learning
        } else if (activity_stats.activity_ratio > brain->config.activity_warning_threshold) {
            // WARNING: Elevated activity - apply graduated inhibition
            nimcp_security_increase_inhibition(
                adaptive_network_get_base_network(brain->network),
                1.2F  // Increase inhibition by 20%
            );
        }
    }

    // ========================================================================
    // PHASE 6.5: MAMMILLARY MEMORY RELAY (Papez Circuit)
    // ========================================================================
    // WHAT: Relay training pattern through mammillary bodies for memory consolidation
    // WHY:  Papez circuit (hippocampus -> mammillary -> thalamus -> cingulate)
    //       is essential for episodic memory formation
    // HOW:  Send feature vector as memory trace, process Papez cycle
    //
    // BIOLOGICAL BASIS:
    // - Mammillary bodies as critical relay in Papez circuit (Papez 1937)
    // - Damage causes anterograde amnesia (Korsakoff syndrome)
    // - Head direction and spatial context binding
    if (brain->mammillary) {
        nimcp_mammillary_t* mb = (nimcp_mammillary_t*)brain->mammillary;
        // Create memory trace from training input
        uint32_t trace_id = 0;
        float emotional_valence = confidence > 0.5f ? 0.5f : -0.3f; // Positive for high-confidence

        int mb_rc = mammillary_receive_hippocampal_input(
            mb,
            features, num_features,
            MEMORY_TRACE_EPISODIC,
            emotional_valence,
            &trace_id
        );

        // Process Papez circuit cycle (relay -> thalamus)
        if (mb_rc == 0 && trace_id > 0) {
            mammillary_relay_to_thalamus(mb, trace_id);
            mammillary_process_papez_cycle(mb);

            // Strengthen trace based on learning success (low loss = stronger memory)
            if (network_loss < 0.3f) {
                mammillary_strengthen_trace(mb, trace_id,
                                            1.0f - network_loss);
            }
        }

        // Update mammillary state
        mammillary_update(mb, 0.01f);
    }

    // ========================================================================
    // PHASE 6.6: PRIME RESONANCE SIGNATURE GENERATION
    // ========================================================================
    // WHAT: Generate prime signature from training features for content-addressable memory
    // WHY:  Prime signatures enable efficient similarity-based memory retrieval
    // HOW:  Convert features to prime signature, store for later resonance queries
    //
    // BIOLOGICAL BASIS:
    // - Content-addressable memory in hippocampal CA3 (Marr 1971)
    // - Pattern separation via prime factorization analog
    if (brain->pr_memory_enabled) {
        prime_signature_t* sig = prime_sig_from_floats(features, num_features);
        if (sig) {
            // Store signature with the engram for future resonance-based retrieval
            // The signature enables finding similar training examples via Jaccard similarity
            // For now, generate and immediately free - the engram encoding in Phase M1
            // will use this for content addressing when PR memory is fully integrated
            prime_sig_destroy(sig);
        }
    }

    // Post-learning evaluation: MSE between updated network output and target
    uint64_t _t_post_start = nimcp_time_get_us();
    bool prediction_from_scratch = false;
    float* prediction;
    if (brain->learn_scratch.prediction && brain->learn_scratch.target_cap >= brain->config.num_outputs) {
        prediction = brain->learn_scratch.prediction;
        prediction_from_scratch = true;
    } else {
        prediction = nimcp_malloc(brain->config.num_outputs * sizeof(float));
    }
    if (prediction) {
        // Forward pass AFTER weight update to measure post-learning accuracy
        adaptive_network_forward(brain->network, features, num_features, prediction,
                                brain->config.num_outputs, 0);

        // MSE loss: meaningful metric that tracks actual output-target convergence
        float post_mse = 0.0F;
        uint32_t pred_argmax = 0;
        float pred_max = prediction[0];
        for (uint32_t i = 0; i < brain->config.num_outputs; i++) {
            float error = prediction[i] - target[i];
            post_mse += error * error;
            if (prediction[i] > pred_max) {
                pred_max = prediction[i];
                pred_argmax = i;
            }
        }
        post_mse /= brain->config.num_outputs;
        network_loss = post_mse;

        // Track label-match accuracy: does argmax(output) match argmax(target)?
        uint32_t target_argmax = 0;
        float target_max = target[0];
        for (uint32_t i = 1; i < brain->config.num_outputs; i++) {
            if (target[i] > target_max) {
                target_max = target[i];
                target_argmax = i;
            }
        }
        bool label_match = (pred_argmax == target_argmax);

        // Update running accuracy (exponential moving average, alpha=0.01)
        float match_val = label_match ? 1.0F : 0.0F;
        brain->stats.running_accuracy =
            brain->stats.running_accuracy * 0.99F + match_val * 0.01F;

        if (!prediction_from_scratch) nimcp_free(prediction);
    }

    // ========================================================================
    // PHASE 7: MULTI-NETWORK ENSEMBLE TRAINING (LNN + CNN)
    // ========================================================================
    // WHAT: Train auxiliary networks alongside the adaptive SNN
    // WHY:  Ensemble learning: different architectures capture different patterns.
    //       LNN captures temporal dynamics, CNN captures spatial features,
    //       Adaptive SNN captures spike-timing correlations.
    // HOW:  Use training_dispatch_step() in HYBRID mode which trains ALL
    //       available networks (LNN, CNN, SNN) and picks lowest loss.
    //
    // BIOLOGICAL BASIS:
    // - Parallel cortical pathways: dorsal (spatial) + ventral (object) streams
    // - Multiple memory systems: procedural (SNN), semantic (CNN), episodic (LNN)
    // - Population coding: diverse representations improve robustness
    if (brain->active_network_type == NIMCP_NETWORK_HYBRID) {
        training_dispatch_result_t dispatch_result = {0};

        // Use the pre-built one-hot target vector
        // training_dispatch_step in HYBRID mode trains all available networks
        training_dispatch_step(brain, features, num_features, target,
                              brain->config.num_outputs, &dispatch_result);
    }

    // ========================================================================
    // PHASE 7.5: VAE TRAINING STEP
    // ========================================================================
    if (brain->vae_system && brain->vae_enabled) {
        vae_system_t* vae = (vae_system_t*)brain->vae_system;

        // Create input tensor from features (rank-2: [1, num_features] for batch_size=1)
        uint32_t vae_dims[2] = { 1, num_features };
        nimcp_tensor_t* vae_input = nimcp_tensor_create(vae_dims, 2, NIMCP_DTYPE_F32);
        if (vae_input) {
            float* vae_data = (float*)nimcp_tensor_data(vae_input);
            if (vae_data) {
                memcpy(vae_data, features, num_features * sizeof(float));
            }

            // Train VAE: forward + loss + backward + update
            vae_loss_t vae_loss = {0};
            int vae_rc = vae_train_step(vae, vae_input, &vae_loss);
            if (vae_rc == 0) {
                // Update free energy on brain (negative ELBO)
                brain->last_vae_free_energy = vae_loss.free_energy;
            }

            // If VAE training bridge exists, do joint VAE+SNN training
            if (brain->vae_training_bridge) {
                vae_training_bridge_t* vtb = (vae_training_bridge_t*)brain->vae_training_bridge;
                vae_training_step_result_t vae_step_result = {0};
                vae_training_step(vtb, features, num_features, target,
                                  brain->config.num_outputs, &vae_step_result);
            }

            nimcp_tensor_destroy(vae_input);
        }
    }

    // ========================================================================
    // PHASE 7.8: ATTENTION-PLASTICITY BRIDGE UPDATE
    // ========================================================================
    // WHAT: Feed learning outcomes back to attention-plasticity bridge
    // WHY:  Attention-modulated STDP: successful learning (low loss) reinforces
    //       the attention patterns that led to it. Reward signal modulates
    //       eligibility traces, BCM thresholds, and novelty detection.
    // HOW:  Record reward (inverse loss), focus event, and update plasticity.
    //
    // BIOLOGICAL BASIS:
    // - Roelfsema & van Ooyen (2005): Attention-gated reinforcement learning
    // - Reward-modulated STDP: DA gates trace-to-weight conversion
    // - Attended synapses undergo stronger LTP (Noudoost et al., 2010)
    if (brain->attention_plasticity && brain->attention_training_enabled) {
        attention_plasticity_bridge_t* apb =
            (attention_plasticity_bridge_t*)brain->attention_plasticity;
        uint64_t now_us = nimcp_time_get_us();

        // Reward signal: low loss = high reward, clamped to [0, 1]
        float reward = fmaxf(0.0f, 1.0f - network_loss * 2.0f);
        attention_plasticity_reward(apb, reward, now_us);

        // Record focus event with current attention strength
        attention_plasticity_focus(apb, 0, brain->last_attention_strength, now_us);

        // Set modulation level so bridge knows current attention state
        attention_plasticity_set_attention_modulation(apb, brain->last_attention_strength);

        // Update all plasticity mechanisms (STDP, BCM, eligibility, habituation)
        attention_plasticity_update(apb, 1.0f);  // 1ms learning step
    }

    // ========================================================================
    // PHASE 8: PERCEPTUAL CORTEX GRADIENT FEEDBACK (post-learning)
    // ========================================================================
    // WHAT: Feed learning loss back to sensory cortices as gradient signal
    // WHY:  Allows V1/A1 to adapt feature extraction based on downstream
    //       learning outcomes — features that cause high loss get modified.
    // HOW:  Use network_loss as a uniform gradient signal, scaled by LR.
    //
    // BIOLOGICAL BASIS:
    // - Feedback projections from PFC/IT to V1 modulate plasticity (Gilbert 2013)
    // - Top-down error signals refine sensory representations over time
    // - Gradient ≈ neuromodulatory feedback (DA/ACh) gating V1/A1 STDP
    if (brain->visual_cortex) {
        uint32_t vdim = visual_cortex_get_feature_dim(brain->visual_cortex);
        if (vdim > 0) {
            // Uniform gradient: broadcast loss as feedback to all visual features
            float* vgrad = nimcp_malloc(vdim * sizeof(float));
            if (vgrad) {
                for (uint32_t i = 0; i < vdim; i++) {
                    vgrad[i] = network_loss;
                }
                float scale = fminf(effective_learning_rate, 1.0f);
                visual_cortex_apply_gradient_feedback(brain->visual_cortex,
                                                     vgrad, vdim, scale);
                nimcp_free(vgrad);
            }
        }
        // Disable training mode (caching) until next example
        visual_cortex_set_training_mode(brain->visual_cortex, false);
    }

    if (brain->audio_cortex) {
        uint32_t adim = audio_cortex_get_feature_dim(brain->audio_cortex);
        if (adim > 0) {
            float* agrad = nimcp_malloc(adim * sizeof(float));
            if (agrad) {
                for (uint32_t i = 0; i < adim; i++) {
                    agrad[i] = network_loss;
                }
                float scale = fminf(effective_learning_rate, 1.0f);
                audio_cortex_apply_gradient_feedback(brain->audio_cortex,
                                                    agrad, adim, scale);
                nimcp_free(agrad);
            }
        }
        audio_cortex_set_training_mode(brain->audio_cortex, false);
    }

    // ========================================================================
    // PHASE 8.5: CORTICAL COLUMN LEARNING (post-learning)
    // ========================================================================
    // WHAT: Step brain regions so cortical columns process the training input
    // WHY:  Cortical columns encode features via competitive dynamics and
    //       lateral inhibition — stepping after learning lets columns adapt
    //       their spatial representations based on the training example.
    // HOW:  Feed training input to brain regions, then step forward.
    //
    // BIOLOGICAL BASIS:
    // - Cortical minicolumns are the basic functional units of cortex
    // - Competitive learning sharpens feature selectivity (Mountcastle 1997)
    // - Activity-dependent refinement of columnar representations
    if (brain->brain_regions) {
        // Step brain regions with a 1ms time delta to trigger
        // intra-region plasticity and inter-region propagation
        brain_module_step(brain->brain_regions, 1000);  // 1000 µs = 1 ms
    }

    // ========================================================================
    // PHASE 8.8: PREDICTIVE CODING UPDATE (post-learning)
    // ========================================================================
    // WHAT: Feed prediction error (from learning loss) into predictive hierarchy
    // WHY:  Predictive coding minimizes free energy — the learning loss IS the
    //       prediction error. Feeding it back updates internal generative model.
    // HOW:  Set input, run inference step with learning enabled, update weights.
    //
    // BIOLOGICAL BASIS:
    // - Free Energy Principle (Friston 2010): brain minimizes surprise
    // - Prediction errors in superficial layers drive learning in deep layers
    // - Loss ↔ surprise: network_loss is the empirical prediction error
    if (brain->predictive_coding) {
        // Present the training features as sensory input to the hierarchy
        pc_hierarchy_set_input(brain->predictive_coding, features);

        // Run one inference + learning step with the prediction error
        // dt=1.0ms, learn=true to update prediction weights
        pc_hierarchy_inference_step(brain->predictive_coding, 1.0f, true);
    }

    // Free attended features buffer (allocated in Phase 4.5)
    if (!attended_from_scratch) nimcp_free(attended_features);  // NULL-safe

    if (!target_from_scratch) nimcp_free(target);

    // Update statistics (atomic for thread safety)
    __atomic_fetch_add(&brain->stats.total_learning_steps, 1, __ATOMIC_RELAXED);

    // Invalidate cache after learning
    clear_cache(brain);

    // ========================================================================
    // PHASE E: HIGHER-ORDER COGNITIVE & SOCIAL LEARNING INTEGRATION
    // ========================================================================
    // WHAT: Update shadow emotions and bias detection during training
    // WHY:  Monitor for maladaptive patterns and biases in training data/behavior
    // HOW:  Analyze labels for bias markers, update self-monitoring systems
    //
    // BIOLOGICAL BASIS:
    // - Metacognitive monitoring during learning (prefrontal cortex)
    // - Social cognition processes during information encoding
    // - Emotional regulation during knowledge acquisition
    //
    // COMPLEXITY: O(1) - Simple pattern matching and updates

    uint64_t current_time = (uint64_t)(brain->stats.total_learning_steps * 1000);  // Simulated time in ms

    // Phase E5: Shadow Emotions - Monitor for maladaptive learning patterns
    if (brain->shadow_emotions) {
        // Decay dynamics
        shadow_update(brain->shadow_emotions, 0.001F, current_time);  // Small dt for incremental updates

        // Auto-intervene if shadow emotions detected
        shadow_auto_intervene(brain->shadow_emotions, current_time);
    }

    // Phase E6: Bias Detection - Analyze training labels for bias markers
    if (brain->bias_detection) {
        // Update system dynamics
        bias_update(brain->bias_detection, 0.001F, current_time);

        // Simple language analysis of training label for bias markers
        // This allows the system to detect if training data contains biased language
        social_group_t general_group;
        general_group.group_id = 0;
        general_group.bias_type = BIAS_RACIAL;  // Default, would be determined by context
        strncpy(general_group.group_name, "General", sizeof(general_group.group_name) - 1);
        general_group.is_marginalized = false;
        general_group.is_stigmatized = false;

        language_pattern_t pattern = bias_analyze_language(brain->bias_detection, label, &general_group, current_time);

        // If bias detected in training data, reduce future learning confidence
        if (pattern.contains_slur || pattern.objectification || pattern.victim_blaming ||
            pattern.hostile_sexism || pattern.incel_ideology) {
            // Apply automatic debiasing intervention
            if (bias_is_detected(brain->bias_detection, general_group.bias_type)) {
                bias_auto_debias(brain->bias_detection, current_time);
            }

            // Record detection in stats (could add brain->stats.bias_detections_in_training)
        }
    }

    // === PHASE E: FULL EMOTIONAL INTELLIGENCE UPDATES ===
    // Update all emotion systems during learning for holistic emotional processing

    // Phase E1: Grief and Loss - Monitor for themes of loss in training data
    if (brain->grief_system) {
        grief_update(brain->grief_system, 0.001F, current_time);
        // Could detect loss/grief themes in training labels for empathetic understanding
    }

    // Phase E2: Joy and Euphoria - Reward for successful learning steps
    if (brain->joy_system) {
        joy_update(brain->joy_system, 0.001F, current_time);

        // If learning was successful (low loss), register as value-aligned success
        if (network_loss >= 0 && network_loss < 0.1F) {
            // Low loss = successful learning = joy trigger
            // This creates positive reinforcement for good learning
            uint32_t aligned_values[] = {VALUE_CATEGORY_LEARNING, VALUE_CATEGORY_ACCURACY};
            joy_process_success(brain->joy_system,
                SUCCESS_TYPE_LEARNED_SKILL,
                aligned_values,
                2,  // num_values
                0.5F,  // Difficulty (moderate)
                brain->last_novelty_score,  // Use actual novelty from curiosity system
                current_time);
        }
    }

    // Phase E3: Remorse and Regret - Monitor for poor decisions or errors
    if (brain->remorse_system) {
        remorse_update(brain->remorse_system, 0.001F, current_time);

        // If training resulted in high loss, register as regrettable event
        if (network_loss > 0.5F) {
            // High loss = learning failure = potential for improvement
            // Creates motivation to avoid similar failures
            uint32_t violated_values[] = {VALUE_CATEGORY_ACCURACY};
            remorse_process_event(brain->remorse_system,
                EVENT_POOR_DECISION,
                violated_values,
                1,  // num_values
                0.3F,  // Harm caused (moderate - not severe)
                0.8F,  // Controllability (we have control over learning)
                true,  // Reversible (can learn better next time)
                current_time);
        }
    }

    // Phase E4: Love, Loyalty, Friendship - Build positive associations with learning
    if (brain->social_bond_system) {
        social_update(brain->social_bond_system, 0.001F, current_time);
        // Could strengthen bonds with training data sources that provide good examples
    }

    // ========================================================================
    // Phase 11 Enhancement C1.1: QUANTUM ANNEALING WEIGHT OPTIMIZATION
    // ========================================================================
    // WHAT: Periodically run quantum annealing to escape local minima
    // WHY:  Gradient descent can get stuck; quantum tunneling explores better solutions
    // HOW:  Every N learning steps, optimize network weights using simulated quantum annealing
    //
    // BIOLOGICAL BASIS:
    // - Sleep-based memory consolidation reorganizes synaptic weights
    // - Exploration vs exploitation balance in learning
    // - Stochastic resonance helps escape suboptimal configurations
    //
    // COMPLEXITY: O(annealing_steps * num_weights) - expensive, run infrequently

    if (brain->config.enable_quantum_annealing &&
        brain->quantum_annealer &&
        brain->config.quantum_annealing_frequency > 0 &&
        (brain->stats.total_learning_steps % brain->config.quantum_annealing_frequency) == 0) {

        brain->stats.quantum_annealing_runs++;

        // Quantum annealing weight optimization - simplified implementation
        // WHAT: Optimize a small subset of critical weights to avoid excessive computation
        // WHY:  Full network optimization is expensive; focus on high-impact weights
        // HOW:  Sample top-K synapses by activity, optimize their weights, apply back
        //
        // PERFORMANCE: Limit to 100 weights to keep optimization tractable
        // Full network optimization would be O(N*M) where N=neurons, M=synapses_per_neuron

        neural_network_t base_net = adaptive_network_get_base_network(brain->network);
        if (base_net) {
            uint32_t num_neurons = neural_network_get_num_neurons(base_net);

            // Sample a small subset for optimization (e.g., first 10 neurons, ~100 weights)
            uint32_t num_neurons_to_optimize = (num_neurons < 10) ? num_neurons : 10;

            // Count total weights to optimize
            uint32_t total_weights = 0;
            for (uint32_t n = 0; n < num_neurons_to_optimize; n++) {
                neuron_t* neuron = neural_network_get_neuron(base_net, n);
                if (neuron) {
                    total_weights += NEURON_OUT_COUNT(neuron);
                }
            }

            if (total_weights > 0 && total_weights <= 1000) {  // Reasonable limit
                // Allocate weight vectors
                float* current_weights = nimcp_malloc(total_weights * sizeof(float));
                float* optimized_weights = nimcp_malloc(total_weights * sizeof(float));

                if (current_weights && optimized_weights) {
                    // Extract current weights
                    uint32_t weight_idx = 0;
                    for (uint32_t n = 0; n < num_neurons_to_optimize; n++) {
                        neuron_t* neuron = neural_network_get_neuron(base_net, n);
                        if (neuron) {
                            uint32_t nsyn = NEURON_OUT_COUNT(neuron);
                            for (uint32_t s = 0; s < nsyn; s++) {
                                synapse_handle_t* h = NEURON_OUT_HANDLE(neuron, s);
                                current_weights[weight_idx++] = h ? h->weight : 0.0F;
                            }
                        }
                    }

                    // Run quantum annealing optimization
                    // Uses quantum_weight_energy (L2 regularization) as objective
                    quantum_anneal(brain->quantum_annealer, nimcp_brain_learning_quantum_weight_energy,
                                 current_weights, optimized_weights, total_weights, NULL);

                    // Apply optimized weights back to network
                    weight_idx = 0;
                    for (uint32_t n = 0; n < num_neurons_to_optimize; n++) {
                        neuron_t* neuron = neural_network_get_neuron(base_net, n);
                        if (neuron) {
                            uint32_t nsyn = NEURON_OUT_COUNT(neuron);
                            for (uint32_t s = 0; s < nsyn; s++) {
                                synapse_handle_t* h = NEURON_OUT_HANDLE(neuron, s);
                                if (!h) { weight_idx++; continue; }
                                // Apply with damping to avoid disrupting learning too much
                                float alpha = 0.1F;  // Mix 10% optimized, 90% current
                                h->weight =
                                    alpha * optimized_weights[weight_idx] +
                                    (1.0F - alpha) * h->weight;
                                weight_idx++;
                            }
                        }
                    }
                }

                nimcp_free(current_weights);
                nimcp_free(optimized_weights);
            }
        }
    }

    // ========================================================================
    // Phase 11: ADAPTIVE LEARNING RATE (Simple Online Meta-Learning)
    // ========================================================================
    // WHAT: Adjust learning rate based on loss trend
    // WHY:  Faster convergence with adaptive step sizes
    // HOW:  Track loss history, accelerate if decreasing, slow if increasing
    //
    // BIOLOGICAL BASIS:
    // - Meta-learning: brain adapts its own learning process
    // - Synaptic homeostasis: plasticity rates self-regulate
    //
    // COMPLEXITY: O(1)
    nimcp_brain_learning_adapt_learning_rate(brain, network_loss);

    // ========================================================================
    // FEEDBACK LOOP: Learning → Sleep Pressure Accumulation
    // ========================================================================
    // WHAT: Accumulate sleep pressure after learning
    // WHY:  Learning is metabolically expensive, builds adenosine (sleep pressure)
    // HOW:  Increment sleep pressure counter each time we learn
    if (brain->sleep_system && brain->config.enable_sleep_wake_cycle) {
        // Each learning step increases sleep pressure
        // Biological basis: synaptic activity produces adenosine
        sleep_accumulate_pressure(brain->sleep_system, 1);  // 1 learning step
    }

    // ========================================================================
    // PHASE M1: MEMORY ENGRAM ENCODING
    // ========================================================================
    // WHAT: Encode learning experience as distributed memory trace
    // WHY:  Enable pattern completion recall and biological memory consolidation
    // HOW:  Map input features to engram neurons, tag with emotional state
    //
    // BIOLOGICAL BASIS:
    // - Engram cells (Tonegawa et al., 2015): neurons active during encoding
    // - IEG expression (c-fos/Arc) tags active neurons for consolidation
    // - Emotional arousal enhances encoding (amygdala modulation)
    // - Memory traces stored as distributed synaptic patterns
    //
    // COMPLEXITY: O(n) where n = num_features
    if (brain->engram_system) {
        // Create neuron ID array from feature indices (simplified mapping)
        // In full implementation, would map to actual active neurons in network
        uint32_t* neuron_ids = nimcp_malloc(num_features * sizeof(uint32_t));
        float* activations = nimcp_malloc(num_features * sizeof(float));

        if (neuron_ids && activations) {
            // Map features to engram neurons
            for (uint32_t i = 0; i < num_features; i++) {
                neuron_ids[i] = i;  // Simplified: feature index = neuron ID
                activations[i] = features[i];  // Feature value = activation
            }

            // Get emotional state for tagging
            // Note: Use confidence as proxy for emotional arousal
            // Higher confidence learning → higher arousal encoding
            emotional_tag_t emotion = {
                .valence = (confidence > 0.7F) ? 0.5F : 0.0F,  // Positive valence for high confidence
                .arousal = confidence,                          // Confidence as arousal proxy
                .timestamp_ms = current_time,
                .category = EMOTION_CAT_NEUTRAL,
                .intensity = confidence
            };

            // Encode engram (episodic memory of this learning event)
            uint64_t engram_id = engram_encode(
                brain->engram_system,
                neuron_ids,
                activations,
                num_features,
                MEMORY_TYPE_EPISODIC,  // Learning experiences are episodic
                emotion
            );

            (void)engram_id;  // Suppress unused warning (could log for debugging)
        }
        // Free unconditionally — nimcp_free(NULL) is a safe no-op,
        // so partial allocation failure (one NULL, one non-NULL) is handled.
        nimcp_free(neuron_ids);
        nimcp_free(activations);
    }


    // ========================================================================
    // PHASE M3: WORKING MEMORY TRANSFER INTEGRATION (LEARNING PIPELINE)
    // ========================================================================
    // WHAT: Add learned pattern to working memory with high attention
    // WHY:  Newly learned information should be available for consolidation
    // HOW:  Update attention (high for new learning), evaluate transfer criteria
    //
    // BIOLOGICAL BASIS:
    // - Attended learning enters working memory (Baddeley & Hitch, 1974)
    // - Rehearsal and attention enhance transfer to LTM (Atkinson & Shiffrin, 1968)
    // - Confidence/arousal boosts encoding strength (McGaugh, 2000)
    //
    // COMPLEXITY: O(1) - Constant time for transfer evaluation
    if (brain->wm_transfer_system && brain->working_memory) {
        // Note: Working memory updates would happen here in full implementation
        // For now, we demonstrate the transfer evaluation mechanism

        // Evaluate transfer criteria (time delta: typical learning cycle ~0.1s)
        const float TIME_DELTA_SECONDS = 0.1F;
        wm_transfer_evaluate(brain->wm_transfer_system, TIME_DELTA_SECONDS);
    }

    // ========================================================================
    // PHASE M4: SEMANTIC MEMORY INTEGRATION (LEARNING PIPELINE)
    // ========================================================================
    // WHAT: Extract semantic concepts from consolidated memories
    // WHY:  Build abstract knowledge network from learned experiences
    // HOW:  Query Phase M2 for semantic memories, create concepts and relations
    //
    // BIOLOGICAL BASIS:
    // - Semantic abstraction from episodic experiences (Tulving, 1972)
    // - Concept formation through repeated exposure (Rosch, 1975)
    // - Semantic networks built from cortical representations (Collins & Quillian, 1969)
    //
    // COMPLEXITY: O(n) where n = number of new semantic memories in M2
    if (brain->semantic_memory && brain->systems_consolidation) {
        // Extract concepts from consolidated semantic memories
        // This builds the semantic network over time as learning progresses
        uint32_t concepts_extracted = semantic_memory_extract_from_consolidation(
            brain->semantic_memory
        );
        (void)concepts_extracted;  // Suppress unused warning (could log for debugging)
    }

    // ========================================================================
    // PHASE C4: SHANNON INFORMATION FLOW ANALYSIS (LEARNING PIPELINE)
    // ========================================================================
    // WHAT: Analyze information flow and detect bottlenecks after learning
    // WHY:  Monitor channel capacity, detect underutilized synapses
    // HOW:  Sample synapses, compute Shannon metrics, detect bottlenecks
    //
    // BIOLOGICAL BASIS:
    // - Neural efficiency: Information-theoretic brain function (Laughlin & Sejnowski, 2003)
    // - Sparse coding: Maximize information transfer with minimal energy (Olshausen & Field, 1996)
    // - Capacity limits: Channel capacity constraints in neural circuits (Koch et al., 2006)
    //
    // COMPLEXITY: O(1) - Monitoring enabled, detailed metrics computed via separate API
    //
    // NOTE: Full synapse-level Shannon analysis will be available through a dedicated
    // API once internal neuron/synapse structures are exposed via proper accessors.
    // For now, this marks monitoring as requested and initializes metrics structure.
    if (brain->enable_shannon_monitoring) {
        // Initialize/update basic network-level metrics
        // Detailed synapse sampling will be added in future enhancement
        brain->last_shannon_metrics.num_synapses = 0;  // To be computed
        brain->last_shannon_metrics.num_neurons = 0;   // To be computed
        // Full implementation pending internal accessor APIs
    }

    // ========================================================================
    // PHASE T1: BIOLOGICAL PLASTICITY FRAMEWORK (TRAINING PIPELINE)
    // ========================================================================
    // WHAT: Apply biological plasticity mechanisms during learning
    // WHY:  Provides homeostatic regulation, dendritic computation, and predictive coding
    // HOW:  Three integrated systems working together for stable, efficient learning
    //
    // BIOLOGICAL BASIS:
    // - Homeostatic plasticity: Maintains neural activity within healthy ranges
    // - Dendritic nonlinearities: Local computation in dendritic branches
    // - Predictive coding: Hierarchical error minimization (Free Energy Principle)
    //
    // COMPLEXITY: O(n) for homeostatic, O(branches) for dendritic, O(levels) for predictive

    // Phase T1.1: Homeostatic Plasticity - Maintain activity levels
    if (brain->homeostatic && brain->config.enable_homeostatic_plasticity) {
        // Get current network activity (use loss as proxy for activity deviation)
        // Low loss = network performing well = moderate activity
        // High loss = network struggling = potentially abnormal activity
        neural_network_t base_net = adaptive_network_get_base_network(brain->network);
        if (base_net) {
            uint32_t num_neurons = neural_network_get_num_neurons(base_net);
            uint32_t homeo_cap = num_neurons < 1000 ? num_neurons : 1000;

            // Compute firing rates array from network state (capped at 1000 neurons)
            float* firing_rates = nimcp_malloc(homeo_cap * sizeof(float));
            if (firing_rates) {
                // Estimate firing rates from neuron outputs
                for (uint32_t i = 0; i < num_neurons && i < 1000; i++) {  // Cap at 1000 neurons
                    neuron_t* neuron = neural_network_get_neuron(base_net, i);
                    if (neuron) {
                        // Use neuron state as proxy for firing rate
                        // Higher state = higher instantaneous rate estimate
                        firing_rates[i] = neuron->state > 0 ? neuron->state * 10.0F : 1.0F;
                    }
                }

                // Update homeostatic controller (rate-based only, no weight access)
                // WHY:  Extracting a flat weight array from sparse synapse storage
                //        for 1.5M neurons is impractical. The controller iterates
                //        controller->num_neurons (all neurons) but weights was only
                //        allocated for ~100 neurons — causing massive buffer overrun.
                //        Pass NULL for weights; controller skips synaptic scaling
                //        weight modification but still tracks rate stability,
                //        intrinsic plasticity, and metaplasticity.
                const float DT_MS = 1.0F;
                homeostatic_controller_update(brain->homeostatic, firing_rates, NULL,
                                             0, DT_MS);

                nimcp_free(firing_rates);
            }
        }
    }


    // Phase T1.2: Dendritic Computation - Process through dendritic tree
    if (brain->dendritic && brain->config.enable_dendritic_computation) {
        // Dendritic trees process input signals through compartmental modeling
        // During learning, we update the dendritic state based on input features
        //
        // This simulates dendritic integration where:
        // - Each branch receives subset of inputs
        // - NMDA dynamics modulate signal propagation
        // - Local dendritic spikes enhance signal detection

        // Inject a sample input into first branch as demonstration
        // Full implementation would distribute inputs across branches based on connectivity
        const float DT_MS = 1.0F;
        dendritic_tree_update(brain->dendritic, DT_MS);
    }

    // Phase T1.3: Predictive Coding - Hierarchical error minimization
    if (brain->predictive_coding && brain->config.enable_biological_predictive) {
        // Predictive coding implements the Free Energy Principle:
        // - Bottom-up: Prediction errors propagate upward
        // - Top-down: Predictions propagate downward
        // - Learning minimizes prediction error at each level
        //
        // The loss from supervised learning drives prediction error updates

        // Set sensory input as observation for predictive coding
        // The hierarchy will update predictions based on bottom-up errors
        // Note: Input size is stored in the hierarchy config
        pc_hierarchy_set_input(brain->predictive_coding, features);

        // Run inference step with learning enabled
        // dt_ms = 1.0 simulates one millisecond of neural dynamics
        pc_hierarchy_inference_step(brain->predictive_coding, 1.0F, true);
    }

    // ========================================================================
    // PHASE C4.1: QUANTUM-SHANNON DIFFUSION (LEARNING PHASE)
    // ========================================================================
    // WHAT: Evolve quantum-Shannon diffusion after learning step
    // WHY:  Monitor information flow during learning, detect bottlenecks
    // HOW:  Evolve quantum walker, update Shannon metrics
    //
    // COMPLEXITY: O(E + N) where E = edges, N = neurons
    if (brain->enable_quantum_shannon_diffusion && brain->quantum_shannon_diffusion) {
        quantum_shannon_diffusion_t* qsd = (quantum_shannon_diffusion_t*)brain->quantum_shannon_diffusion;

        // Evolve with configured steps
        if (quantum_shannon_evolve(qsd, brain->quantum_shannon_evolution_steps)) {
            // Update metrics
            quantum_shannon_get_metrics(qsd, &brain->last_quantum_shannon_metrics);

            // Log if bottlenecks detected (useful for debugging/optimization)
            if (brain->last_quantum_shannon_metrics.num_bottlenecks > 0) {
                // Bottlenecks detected - could trigger adaptive plasticity in future
                // For now, just track in metrics
            }
        }
    }

    // ========================================================================
    // PHASE 23.5: GLIAL MODULATION UPDATE
    // ========================================================================
    // WHAT: Update astrocyte/oligodendrocyte state based on learning activity
    // WHY:  Glial cells modulate synaptic strength and myelinate active pathways
    // HOW:  Feed learning signal to glial integration system
    //
    // BIOLOGICAL BASIS:
    // - Astrocyte calcium waves modulate synaptic transmission (Perea et al. 2009)
    // - Activity-dependent myelination (Fields 2015)
    // - Astrocytic glutamate release enhances LTP (Jourdain et al. 2007)
    if (brain->glial) {
        // Use simulated time in microseconds for timestamp-based step
        uint64_t glial_timestamp = (uint64_t)(brain->stats.total_learning_steps * 1000);
        glial_integration_step(brain->glial, glial_timestamp);
    }

    uint64_t _t_post_us = nimcp_time_get_us() - _t_post_start;
    uint64_t _t_total_us = _t_learn_us + _t_post_us;

    // Log timing every 10 examples for profiling
    if ((brain->stats.total_learning_steps % 10) == 0) {
        LOG_INFO("LEARN TIMING: total=%llu ms  core_learn=%llu ms  post_phases=%llu ms",
                 (unsigned long long)(_t_total_us / 1000),
                 (unsigned long long)(_t_learn_us / 1000),
                 (unsigned long long)(_t_post_us / 1000));
    }

    brain_clear_error();
    return network_loss;
}

/**
 * @brief Enable multi-network training (LNN + CNN alongside adaptive)
 *
 * WHAT: Creates auxiliary LNN and CNN networks matched to brain dimensions
 * WHY:  Ensemble learning: different architectures capture different patterns.
 *       LNN captures temporal dynamics, CNN captures spatial features,
 *       Adaptive SNN captures spike-timing correlations.
 * HOW:  Creates NCP-architecture LNN, dense CNN, initializes dispatch as HYBRID
 *
 * @param brain Brain to augment
 * @return 0 on success, -1 on error
 */
int brain_enable_multi_network_training(brain_t brain)
{
    // Guard: NULL check
    if (!brain) {
        set_error("brain_enable_multi_network_training: brain is NULL");
        return -1;
    }

    // Guard: Already in HYBRID mode
    if (brain->active_network_type == NIMCP_NETWORK_HYBRID) {
        return 0;  // Already enabled, idempotent
    }

    uint32_t num_inputs = brain->config.num_inputs;
    uint32_t num_outputs = brain->config.num_outputs;

    // ========================================================================
    // STEP 1: Create LNN network (NCP architecture)
    // ========================================================================
    // WHAT: Create a Liquid Neural Network matched to brain dimensions
    // WHY:  LNN captures continuous-time temporal dynamics via ODE neurons
    // HOW:  Use NCP (Neural Circuit Policy) 4-layer architecture:
    //       Input -> Sensory -> Inter -> Command -> Motor -> Output
    // LNN requires minimum dimensions for stable tensor operations.
    // Skip LNN for very small networks (< 8 inputs/outputs) to avoid
    // tensor dimension mismatch crashes in the ODE forward pass.
    if (!brain->lnn_network && num_inputs >= 8 && num_outputs >= 8) {
        // Create NCP (Neural Circuit Policy) LNN with ALL layers capped to 256.
        // LNN papers use 19-256 neurons; beyond 256 the O(n^2) adjoint Jacobian
        // dominates (1024^2 = 1M ops/step vs 256^2 = 65K). When brain dims exceed
        // the cap, lnn_train_step() average-pools the input and truncates targets
        // to match the LNN's smaller dimensions.
        uint32_t lnn_cap = 256;
        uint32_t lnn_in = (num_inputs > lnn_cap) ? lnn_cap : num_inputs;
        uint32_t lnn_out = (num_outputs > lnn_cap) ? lnn_cap : num_outputs;
        uint32_t n_inter = (lnn_in + lnn_out) / 2;
        if (n_inter < 8) n_inter = 8;
        if (n_inter > lnn_cap) n_inter = lnn_cap;
        uint32_t n_command = lnn_out * 2;
        if (n_command < 8) n_command = 8;
        if (n_command > lnn_cap) n_command = lnn_cap;

        NIMCP_LOGGING_INFO("LNN NCP: brain dims %u→%u, LNN dims %u→%u→%u→%u→%u",
                           num_inputs, num_outputs, lnn_in, n_inter, n_command, lnn_out);

        lnn_network_t* lnn = lnn_network_create_ncp(
            lnn_in, n_inter, n_command, lnn_out
        );
        if (!lnn) {
            NIMCP_LOGGING_WARN("Failed to create LNN network for multi-network training");
            // Non-fatal: continue without LNN
        } else {
            lnn_network_init_weights(lnn, 0);  // Random seed
            lnn_network_set_training(lnn, true);
            brain->lnn_network = lnn;
            brain->owns_specialized_network = true;
        }
    } else if (!brain->lnn_network) {
        NIMCP_LOGGING_INFO("LNN skipped: requires num_inputs >= 8 and num_outputs >= 8 "
                          "(brain has %u inputs, %u outputs)", num_inputs, num_outputs);
    }

    // ========================================================================
    // STEP 2: Create LNN training context
    // ========================================================================
    // WHAT: Initialize adjoint-method training for the LNN
    // WHY:  LNN requires ODE-aware gradient computation
    // HOW:  Create training context with slow learning rate (0.01)
    if (brain->lnn_network && !brain->lnn_training_ctx) {
        lnn_training_config_t lnn_cfg;
        lnn_training_config_default(&lnn_cfg);
        lnn_cfg.learning_rate = 0.01f;  // LNN learns slower than backprop
        lnn_cfg.gradient_clip_norm = 1.0f;

        lnn_training_ctx_t* lnn_ctx = lnn_training_create(brain->lnn_network, &lnn_cfg);
        if (!lnn_ctx) {
            NIMCP_LOGGING_WARN("Failed to create LNN training context");
        } else {
            brain->lnn_training_ctx = lnn_ctx;
        }
    }

    // ========================================================================
    // STEP 3: Create CNN trainer with dense layers
    // ========================================================================
    // WHAT: Create a CNN trainer with two dense layers for classification
    // WHY:  CNN dense layers capture spatial feature relationships
    // HOW:  Dense(num_inputs -> hidden) -> ReLU -> Dense(hidden -> num_outputs)
    if (!brain->cnn_trainer) {
        uint32_t hidden_size = num_outputs * 4;
        if (hidden_size < 32) hidden_size = 32;
        if (hidden_size > 512) hidden_size = 512;

        cnn_trainer_config_t cnn_cfg;
        cnn_trainer_default_config(&cnn_cfg);
        cnn_cfg.learning_rate = 0.001f;

        cnn_trainer_t* trainer = cnn_trainer_create(&cnn_cfg);
        if (!trainer) {
            NIMCP_LOGGING_WARN("Failed to create CNN trainer for multi-network training");
        } else {
            // Add dense layer: num_inputs -> hidden_size
            cnn_dense_config_t dense1 = {
                .in_features = num_inputs,
                .out_features = hidden_size,
                .activation = CNN_ACTIVATION_NONE,
                .use_bias = true,
                .weight_init_std = 0.01f
            };
            cnn_trainer_add_dense_layer(trainer, &dense1);

            // Add ReLU activation
            cnn_trainer_add_activation_layer(trainer, CNN_ACTIVATION_RELU);

            // Add dense layer: hidden_size -> num_outputs
            cnn_dense_config_t dense2 = {
                .in_features = hidden_size,
                .out_features = num_outputs,
                .activation = CNN_ACTIVATION_NONE,
                .use_bias = true,
                .weight_init_std = 0.01f
            };
            cnn_trainer_add_dense_layer(trainer, &dense2);

            brain->cnn_trainer = trainer;
        }
    }

    // ========================================================================
    // STEP 3.5: Create VAE (Variational Autoencoder)
    // ========================================================================
    // WHAT: Create a VAE matched to brain input dimensions
    // WHY:  VAE learns compressed latent representations, enables:
    //       - Anomaly detection (flag out-of-distribution inputs)
    //       - Generative replay (combat catastrophic forgetting)
    //       - Free Energy Principle integration (ELBO = variational free energy)
    //       - Better engram encoding via latent space
    // HOW:  Encoder: input_dim → hidden → latent_dim (mu, log_var)
    //       Decoder: latent_dim → hidden → input_dim (reconstruction)
    if (!brain->vae_system) {
        vae_config_t vae_cfg;
        vae_default_config(&vae_cfg);

        // Configure encoder: 2 hidden layers
        uint32_t latent_dim = num_inputs / 4;
        if (latent_dim < 16) latent_dim = 16;
        if (latent_dim > 256) latent_dim = 256;

        uint32_t vae_hidden = num_inputs / 2;
        if (vae_hidden < 32) vae_hidden = 32;
        if (vae_hidden > 512) vae_hidden = 512;

        vae_cfg.encoder.input_dim = num_inputs;
        vae_cfg.encoder.latent_dim = latent_dim;
        vae_cfg.encoder.num_layers = 2;
        vae_cfg.encoder.layers[0] = (vae_layer_config_t){
            .units = vae_hidden, .activation = VAE_ACTIVATION_RELU,
            .dropout_rate = 0.0f, .batch_norm = false, .use_bias = true
        };
        vae_cfg.encoder.layers[1] = (vae_layer_config_t){
            .units = vae_hidden / 2, .activation = VAE_ACTIVATION_RELU,
            .dropout_rate = 0.0f, .batch_norm = false, .use_bias = true
        };
        vae_cfg.encoder.mu_activation = VAE_ACTIVATION_LINEAR;
        vae_cfg.encoder.var_activation = VAE_ACTIVATION_SOFTPLUS;

        // Configure decoder: mirror of encoder
        vae_cfg.decoder.latent_dim = latent_dim;
        vae_cfg.decoder.output_dim = num_inputs;
        vae_cfg.decoder.num_layers = 2;
        vae_cfg.decoder.layers[0] = (vae_layer_config_t){
            .units = vae_hidden / 2, .activation = VAE_ACTIVATION_RELU,
            .dropout_rate = 0.0f, .batch_norm = false, .use_bias = true
        };
        vae_cfg.decoder.layers[1] = (vae_layer_config_t){
            .units = vae_hidden, .activation = VAE_ACTIVATION_RELU,
            .dropout_rate = 0.0f, .batch_norm = false, .use_bias = true
        };
        vae_cfg.decoder.final_activation = VAE_ACTIVATION_SIGMOID;
        vae_cfg.decoder.output_variance = false;

        // Training config
        vae_cfg.training.learning_rate = 0.001f;
        vae_cfg.training.beta = 1.0f;      // Standard VAE (beta=1)
        vae_cfg.training.beta_warmup_steps = 1000;
        vae_cfg.training.loss_type = VAE_LOSS_MSE;
        vae_cfg.training.gradient_clip = 5.0f;
        vae_cfg.training.batch_size = 1;    // Online learning

        vae_cfg.variant = VAE_VARIANT_BETA;
        vae_cfg.prior_type = VAE_PRIOR_STANDARD_NORMAL;
        vae_cfg.anomaly_threshold = 3.0f;

        vae_system_t* vae = vae_create(&vae_cfg);
        if (!vae) {
            NIMCP_LOGGING_WARN("Failed to create VAE for multi-network training");
        } else {
            vae_set_training(vae, true);
            brain->vae_system = vae;
            brain->vae_enabled = true;

            // Create VAE training bridge for joint VAE+SNN training
            vae_training_bridge_config_t vtb_cfg;
            vae_training_bridge_default_config(&vtb_cfg);
            vtb_cfg.algorithm = VAE_TRAIN_JOINT;
            vtb_cfg.loss_combination = VAE_LOSS_WEIGHTED;
            vtb_cfg.vae_loss_weight = 0.3f;   // 30% VAE loss
            vtb_cfg.snn_loss_weight = 0.7f;   // 70% SNN loss
            vtb_cfg.optimizer.learning_rate = 0.001f;
            vtb_cfg.optimizer.use_gradient_clipping = true;
            vtb_cfg.optimizer.gradient_clip_norm = 5.0f;

            vae_training_bridge_t* vtb = vae_training_bridge_create(&vtb_cfg);
            if (vtb) {
                vae_training_bridge_connect_vae(vtb, vae);
                brain->vae_training_bridge = vtb;
            }

            NIMCP_LOGGING_INFO("VAE created: input_dim=%u, latent_dim=%u, hidden=%u",
                              num_inputs, latent_dim, vae_hidden);
        }
    }

    // ========================================================================
    // STEP 4.5: Initialize attention-plasticity bridge for training
    // ========================================================================
    // WHAT: Create attention-plasticity bridge for attention-modulated learning
    // WHY:  Enables attention to gate STDP, BCM, and eligibility trace plasticity.
    //       Attended inputs undergo stronger LTP; unattended undergo LTD.
    //       Reward signals reinforce successful attention patterns.
    // HOW:  Create bridge with attention modulation + eligibility + novelty detection
    if (brain->multihead_attention && !brain->attention_plasticity) {
        attention_plasticity_config_t apb_cfg = attention_plasticity_config_default();
        apb_cfg.enable_attention_modulation = true;
        apb_cfg.attention_learning_gain = 0.3f;   // 30% attention influence on LR
        apb_cfg.focus_learning_boost = 1.5f;       // 50% LTP boost for focused items
        apb_cfg.unfocused_ltd_boost = 0.3f;        // Weak LTD for unfocused
        apb_cfg.enable_eligibility = true;
        apb_cfg.eligibility_decay = 0.95f;         // Slow trace decay
        apb_cfg.reward_modulation_gain = 1.0f;     // Full reward modulation
        apb_cfg.enable_novelty_detection = true;
        apb_cfg.novelty_boost = 0.3f;              // 30% boost for novel inputs
        apb_cfg.enable_bcm = true;                 // BCM metaplasticity
        apb_cfg.bcm_threshold_tau = 0.01f;         // Slow threshold adaptation

        attention_plasticity_bridge_t* apb = attention_plasticity_create(&apb_cfg);
        if (apb) {
            brain->attention_plasticity = apb;
            brain->attention_training_enabled = true;
            NIMCP_LOGGING_INFO("Attention-plasticity bridge created for training");
        } else {
            NIMCP_LOGGING_WARN("Failed to create attention-plasticity bridge");
        }
    }

    // ========================================================================
    // STEP 4: Set HYBRID mode and initialize dispatch
    // ========================================================================
    // WHAT: Switch brain to HYBRID training mode
    // WHY:  Enables training_dispatch_step to train all available networks
    // HOW:  Set active_network_type, call training_dispatch_init
    brain->active_network_type = NIMCP_NETWORK_HYBRID;

    nimcp_training_config_t dispatch_cfg = nimcp_training_config_default();
    dispatch_cfg.network_type = NIMCP_NETWORK_HYBRID;
    dispatch_cfg.learning_rate = brain->config.learning_rate;
    training_dispatch_init(brain, &dispatch_cfg);

    NIMCP_LOGGING_INFO("Multi-network training enabled: Adaptive + %s + %s + %s + %s",
        brain->lnn_network ? "LNN" : "(no LNN)",
        brain->cnn_trainer ? "CNN" : "(no CNN)",
        brain->vae_system  ? "VAE" : "(no VAE)",
        brain->attention_training_enabled ? "Attention" : "(no Attention)");

    return 0;
}

/**
 * @brief Learn from batch of examples
 *
 * WHY: More efficient than individual calls
 * Enables mini-batch gradient descent
 *
 * COMPLEXITY: O(m*s*n) where m = num_examples
 *
 * @param brain Brain handle
 * @param examples Array of examples
 * @param num_examples Example count
 * @return Average loss or -1 on error
 */
float brain_learn_batch(brain_t brain, const brain_example_t* examples, uint32_t num_examples)
{
    // Guard: Validate parameters
    if (!brain || !examples || num_examples == 0) {
        set_error("Invalid parameters to brain_learn_batch");
        return -1.0F;
    }

    // Health monitoring: signal start of batch learning
    brain_heartbeat(brain, "brain_learn_batch:start", 0.0f);

    float total_loss = 0.0F;

    // Calculate heartbeat interval (every 10% or every 100 examples, whichever is larger)
    uint32_t heartbeat_interval = (num_examples / 10 > 100) ? num_examples / 10 : 100;
    if (heartbeat_interval == 0) heartbeat_interval = 1;

    for (uint32_t i = 0; i < num_examples; i++) {
        float loss = brain_learn_example(brain, examples[i].features, examples[i].num_features,
                                         examples[i].label, examples[i].confidence);

        if (loss < 0.0F) {
            return -1.0F;
        }

        total_loss += loss;

        // Health monitoring: periodic progress update
        if ((i + 1) % heartbeat_interval == 0) {
            float progress = (float)(i + 1) / (float)num_examples;
            brain_heartbeat(brain, "brain_learn_batch:progress", progress);
        }
    }

    // Health monitoring: batch learning complete
    brain_heartbeat(brain, "brain_learn_batch:complete", 1.0f);

    brain_clear_error();
    return total_loss / num_examples;
}

/**
 * @brief Learn batch with per-example loss output
 *
 * WHY: Python binding needs individual loss values for training diagnostics.
 *      The original brain_learn_batch only returns the average.
 *
 * @param brain Brain handle
 * @param examples Array of examples
 * @param num_examples Example count
 * @param losses_out Caller-allocated float[num_examples] to receive per-example losses.
 *                   Can be NULL to skip per-example tracking.
 * @return Average loss or -1 on error
 */
float brain_learn_batch_detailed(brain_t brain, const brain_example_t* examples,
                                 uint32_t num_examples, float* losses_out)
{
    if (!brain || !examples || num_examples == 0) {
        set_error("Invalid parameters to brain_learn_batch_detailed");
        return -1.0F;
    }

    brain_heartbeat(brain, "brain_learn_batch:start", 0.0f);

    float total_loss = 0.0F;
    uint32_t heartbeat_interval = (num_examples / 10 > 100) ? num_examples / 10 : 100;
    if (heartbeat_interval == 0) heartbeat_interval = 1;

    for (uint32_t i = 0; i < num_examples; i++) {
        float loss = brain_learn_example(brain, examples[i].features, examples[i].num_features,
                                         examples[i].label, examples[i].confidence);

        if (loss < 0.0F) {
            /* Fill remaining losses_out with -1 to signal error */
            if (losses_out) {
                for (uint32_t j = i; j < num_examples; j++)
                    losses_out[j] = -1.0F;
            }
            return -1.0F;
        }

        if (losses_out)
            losses_out[i] = loss;
        total_loss += loss;

        if ((i + 1) % heartbeat_interval == 0) {
            float progress = (float)(i + 1) / (float)num_examples;
            brain_heartbeat(brain, "brain_learn_batch:progress", progress);
        }
    }

    brain_heartbeat(brain, "brain_learn_batch:complete", 1.0f);
    brain_clear_error();
    return total_loss / num_examples;
}

/**
 * @brief Apply reward-based reinforcement learning
 *
 * WHAT: Apply eligibility-trace-based learning with reward signal
 * WHY:  Enable temporal credit assignment for RL tasks
 * HOW:  Call neural_network_apply_reward_learning() with reward and dopamine
 *
 * BIOLOGY: Three-factor learning rule (Hebbian + Reward + Dopamine)
 * - Eligibility traces mark recently active synapses ("synaptic tags")
 * - Dopamine bursts trigger consolidation ("capture")
 * - Reward signal modulates weight changes (Frey & Morris 1997)
 *
 * COMPLEXITY: O(n × s) where n=neurons, s=synapses_per_neuron
 * USE CASE: Reinforcement learning, temporal credit assignment
 *
 * @param brain Brain handle
 * @param reward Reward signal (0-1 positive, -1-0 negative)
 * @return Number of synapses modified
 */
uint32_t brain_apply_reward_learning(brain_t brain, float reward)
{
    // Guard: Validate brain
    if (!brain) {
        set_error("NULL brain handle");
        return 0;
    }

    // Guard: Validate reward range
    if (reward < -1.0F || reward > 1.0F) {
        set_error("Reward must be in range [-1.0, 1.0], got %.2f", reward);
        return 0;
    }

    // Phase 2: Ensure network is writable
    if (!ensure_writable_network(brain)) {
        return 0;  // Error already set
    }

    // Get base neural network from adaptive network
    neural_network_t base_network = adaptive_network_get_base_network(brain->network);
    if (!base_network) {
        set_error("Failed to get base network");
        return 0;
    }

    // Get current network time
    uint64_t current_time = nimcp_time_get_us();

    // Apply reward-modulated learning with eligibility traces
    // Uses neural_network_apply_reward_learning() from neuralnet.c (lines 1472-1600)
    uint32_t num_modified = neural_network_apply_reward_learning(
        base_network,
        reward,
        brain->config.learning_rate,
        current_time
    );

    // Update brain stats (atomic for thread safety)
    __atomic_fetch_add(&brain->stats.total_learning_steps, 1, __ATOMIC_RELAXED);

    brain_clear_error();
    return num_modified;
}

/**
 * @brief Learn by querying an LLM teacher
 *
 * WHY: Enables distillation from larger models
 * Brain learns to mimic LLM decisions efficiently
 *
 * COMPLEXITY: O(s*n) + LLM query time
 * USE CASE: Compress LLM knowledge into fast neural network
 *
 * @param brain Brain handle
 * @param input Input features
 * @param num_features Feature count
 * @param llm_fn Teacher function
 * @param llm_context Context for teacher
 * @return Loss value or -1 on error
 */
float brain_learn_from_llm(brain_t brain, const float* input, uint32_t num_features,
                           llm_teacher_fn_t llm_fn, void* llm_context)
{
    // Guard: Validate parameters
    if (!brain || !input || !llm_fn) {
        set_error("Invalid parameters to brain_learn_from_llm");
        return -1.0F;
    }

    // Guard: Check dimensions
    if (num_features != brain->config.num_inputs) {
        set_error("Feature count mismatch: expected %u, got %u", brain->config.num_inputs,
                  num_features);
        return -1.0F;
    }

    // Query LLM teacher
    char label[NIMCP_ID_BUFFER_SIZE] = {0};
    float confidence = llm_fn(input, num_features, llm_context, label, sizeof(label));

    // Guard: Validate LLM response
    if (confidence <= 0.0F) {
        set_error("LLM teacher returned invalid confidence: %.2f", confidence);
        return -1.0F;
    }

    // Learn from LLM's decision
    float loss = brain_learn_example(brain, input, num_features, label, confidence);

    brain_clear_error();
    return loss;
}


//=============================================================================
// Async Learning Implementation
//=============================================================================

/**
 * @brief Context for async learning worker thread
 */
typedef struct {
    brain_t brain;
    float* features;
    uint32_t num_features;
    char* label;
    float confidence;
    nimcp_promise_t promise;
} async_learn_context_t;

/**
 * @brief Background thread function for async learning
 *
 * WHAT: Performs learning in background thread and completes promise
 * WHY:  Enable non-blocking learning without blocking caller
 * HOW:  Call brain_learn_example, set result or error on promise
 *
 * THREAD SAFETY: Each thread has its own context, brain must support concurrent writes
 *                (or caller must ensure serialization)
 *
 * @param arg async_learn_context_t pointer
 * @return NULL (unused)
 */
static void* async_learn_thread(void* arg)
{
    async_learn_context_t* ctx = (async_learn_context_t*)arg;

    if (!ctx) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "ctx is NULL");


        return NULL;
    }

    LOG_MODULE_DEBUG("core_brain_learning", "Async learning started: %u features, label='%s'",
                    ctx->num_features, ctx->label);

    // Perform synchronous learning
    float loss = brain_learn_example(
        ctx->brain,
        ctx->features,
        ctx->num_features,
        ctx->label,
        ctx->confidence
    );

    if (loss < 0.0f) {
        // Learning failed
        nimcp_error_t error = NIMCP_ERROR_OPERATION_FAILED;
        nimcp_promise_fail(ctx->promise, error);
        LOG_MODULE_ERROR("core_brain_learning", "Async learning failed for label '%s'", ctx->label);
    } else {
        // Learning succeeded - pass loss value
        nimcp_promise_complete(ctx->promise, &loss);
        LOG_MODULE_DEBUG("core_brain_learning", "Async learning completed: loss=%.4f", loss);
    }

    // Cleanup
    nimcp_free(ctx->features);
    nimcp_free(ctx->label);
    nimcp_promise_destroy(ctx->promise);
    nimcp_free(ctx);

    /* BUG FIX: Previously unconditionally threw NIMCP_THROW_TO_IMMUNE here,
     * which fired even on successful learning. Thread functions should just
     * return; errors are already propagated via the promise/future. */
    return NULL;
}

/**
 * @brief Asynchronous learning
 *
 * WHAT: Non-blocking version of brain_learn_example
 * WHY:  Enable concurrent learning without blocking caller
 * HOW:  Copy inputs, create promise/future, spawn thread
 *
 * IMPLEMENTATION NOTES:
 * - Features and label are deep-copied to ensure thread safety
 * - Promise is created with sizeof(float) for loss result
 * - Thread is detached for auto-cleanup
 * - Context is freed by worker thread
 *
 * ERROR HANDLING:
 * - Returns NULL if allocation fails or thread creation fails
 * - Learning errors propagated through future
 *
 * MEMORY MANAGEMENT:
 * - Context allocated on heap, freed by worker thread
 * - Features and label copied to heap, freed by worker thread
 * - Promise destroyed by worker thread after completion
 * - Future must be destroyed by caller
 *
 * @param brain Brain handle
 * @param features Input features (will be copied)
 * @param num_features Feature count
 * @param label Target label (will be copied)
 * @param confidence Training weight
 * @return Future handle or NULL on error
 */
nimcp_future_t nimcp_brain_learn_async(brain_t brain, const float* features,
                                        uint32_t num_features, const char* label,
                                        float confidence)
{
    // Validate parameters
    if (!brain || !features || !label) {
        LOG_MODULE_ERROR("core_brain_learning", "Invalid parameters to nimcp_brain_learn_async");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "nimcp_brain_learn_async: required parameter is NULL (brain, features, label)");
        return NULL;
    }

    if (num_features == 0) {
        LOG_MODULE_ERROR("core_brain_learning", "Invalid num_features=0");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "nimcp_brain_learn_async: num_features is zero");
        return NULL;
    }

    // Validate confidence range
    if (confidence < 0.0f || confidence > 1.0f) {
        LOG_MODULE_ERROR("core_brain_learning", "Invalid confidence=%.2f (must be 0.0-1.0)", confidence);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "nimcp_brain_learn_async: validation failed");
        return NULL;
    }

    // Allocate context for worker thread
    async_learn_context_t* ctx = nimcp_malloc(sizeof(async_learn_context_t));
    if (!ctx) {
        LOG_MODULE_ERROR("core_brain_learning", "Failed to allocate async learning context");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "nimcp_brain_learn_async: ctx is NULL");
        return NULL;
    }

    // Copy features to heap (worker thread will free)
    ctx->features = nimcp_malloc(num_features * sizeof(float));
    if (!ctx->features) {
        LOG_MODULE_ERROR("core_brain_learning", "Failed to allocate features array (%u floats)", num_features);
        nimcp_free(ctx);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "nimcp_brain_learn_async: ctx->features is NULL");
        return NULL;
    }
    memcpy(ctx->features, features, num_features * sizeof(float));

    // Copy label to heap (worker thread will free)
    size_t label_len = strlen(label);
    ctx->label = nimcp_malloc(label_len + 1);
    if (!ctx->label) {
        LOG_MODULE_ERROR("core_brain_learning", "Failed to allocate label string");
        nimcp_free(ctx->features);
        nimcp_free(ctx);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "nimcp_brain_learn_async: ctx->label is NULL");
        return NULL;
    }
    memcpy(ctx->label, label, label_len + 1);

    // Set context fields
    ctx->brain = brain;
    ctx->num_features = num_features;
    ctx->confidence = confidence;

    // Create promise for result (loss is a float)
    ctx->promise = nimcp_promise_create(sizeof(float));
    if (!ctx->promise) {
        LOG_MODULE_ERROR("core_brain_learning", "Failed to create promise for async learning");
        nimcp_free(ctx->label);
        nimcp_free(ctx->features);
        nimcp_free(ctx);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "nimcp_brain_learn_async: ctx->promise is NULL");
        return NULL;
    }

    // Get future before starting thread
    nimcp_future_t future = nimcp_promise_get_future(ctx->promise);
    if (!future) {
        LOG_MODULE_ERROR("core_brain_learning", "Failed to get future from promise");
        nimcp_promise_destroy(ctx->promise);
        nimcp_free(ctx->label);
        nimcp_free(ctx->features);
        nimcp_free(ctx);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "nimcp_brain_learn_async: future is NULL");
        return NULL;
    }

    // Create worker thread
    nimcp_thread_t thread;
    thread_attr_t attr = {
        .stack_size = NIMCP_THREAD_DEFAULT_STACK_SIZE,
        .priority = 0,
        .detached = true  // Auto-cleanup on exit
    };

    nimcp_result_t result = nimcp_thread_create(&thread, async_learn_thread, ctx, &attr);
    if (result != NIMCP_SUCCESS) {
        LOG_MODULE_ERROR("core_brain_learning", "Failed to create async learning thread: %d", result);
        nimcp_future_destroy(future);
        nimcp_promise_destroy(ctx->promise);
        nimcp_free(ctx->label);
        nimcp_free(ctx->features);
        nimcp_free(ctx);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "nimcp_brain_learn_async: validation failed");
        return NULL;
    }

    LOG_MODULE_INFO("core_brain_learning", "Async learning thread created successfully for label='%s'", label);

    // Return future to caller (caller must destroy)
    return future;
}
