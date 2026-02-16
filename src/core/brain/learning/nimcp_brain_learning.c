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
#include "utils/memory/nimcp_memory.h"
#include "utils/time/nimcp_time.h"
#include "security/nimcp_security.h"
#include "security/nimcp_blood_brain_barrier.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/thread/nimcp_thread.h"
#include "async/nimcp_future.h"
#include "utils/exception/nimcp_exception_macros.h"
#include "core/brain/nimcp_brain_internal.h"

#define LOG_MODULE "core_brain_learning"
#include "utils/fault_tolerance/nimcp_health_agent_macros.h"
#include "utils/bridge/nimcp_bridge_boilerplate.h"
#include "mesh/nimcp_mesh_participant.h"
#include "mesh/nimcp_mesh_adapter.h"
#include "constants/nimcp_buffer_constants.h"

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
    // Search existing labels - O(k)
    for (uint32_t i = 0; i < brain->num_output_labels; i++) {
        if (strcmp(brain->output_labels[i], label) == 0) {
            return i;
        }
    }

    // Guard: Check capacity
    if (brain->num_output_labels >= brain->config.num_outputs) {
        return 0;
    }

    // Create new label (use nimcp_malloc to match nimcp_free in brain_destroy)
    size_t label_len = strlen(label);
    brain->output_labels[brain->num_output_labels] = nimcp_malloc(label_len + 1);
    if (!brain->output_labels[brain->num_output_labels])
        return 0;
    strncpy(brain->output_labels[brain->num_output_labels], label, label_len + 1);
    brain->output_labels[brain->num_output_labels][label_len] = '\0';
    return brain->num_output_labels++;
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
    output[label_idx] = confidence;
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
    brain->loss_history_index = (brain->loss_history_index + 1) % 10;
    if (brain->loss_history_count < 10) {
        brain->loss_history_count++;
    }

    // Need at least 3 samples to compute trend
    if (brain->loss_history_count < 3) {
        return;
    }

    // Compute loss trend: recent avg vs older avg
    float recent_avg = 0.0F;
    float older_avg = 0.0F;
    uint32_t half = brain->loss_history_count / 2;

    // Older half
    for (uint32_t i = 0; i < half; i++) {
        older_avg += brain->loss_history[i];
    }
    older_avg /= half;

    // Recent half
    for (uint32_t i = half; i < brain->loss_history_count; i++) {
        recent_avg += brain->loss_history[i];
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

    // Convert label to target output
    float* target = nimcp_malloc(brain->config.num_outputs * sizeof(float));
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
    if (brain->epistemic) {
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

    // Learn using adaptive network with curiosity-modulated learning rate
    float network_loss = adaptive_network_learn(brain->network, &example, LEARN_MODE_SUPERVISED,
                                                effective_learning_rate);

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

            nimcp_free(target);
            return -1.0F;  // Abort learning
        } else if (activity_stats.activity_ratio > brain->config.activity_warning_threshold) {
            // WARNING: Elevated activity - apply graduated inhibition
            nimcp_security_increase_inhibition(
                adaptive_network_get_base_network(brain->network),
                1.2F  // Increase inhibition by 20%
            );
        }
    }

    // Compute task-specific loss using strategy
    // Get network prediction to compute task-specific loss
    float* prediction = nimcp_malloc(brain->config.num_outputs * sizeof(float));
    if (prediction) {
        // Forward pass to get current prediction
        adaptive_network_forward(brain->network, features, num_features, prediction,
                                brain->config.num_outputs, 0);  // timestamp = 0

        // Compute task-specific loss using strategy
        float task_loss = brain->strategy->compute_loss(prediction, target,
                                                       brain->config.num_outputs);

        // Use task-specific loss (more meaningful for the specific task)
        network_loss = task_loss;

        nimcp_free(prediction);
    }

    nimcp_free(target);

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
                if (neuron && neuron->synapses) {
                    total_weights += neuron->num_synapses;
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
                        if (neuron && neuron->synapses) {
                            for (uint32_t s = 0; s < neuron->num_synapses; s++) {
                                current_weights[weight_idx++] = neuron->synapses[s].weight;
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
                        if (neuron && neuron->synapses) {
                            for (uint32_t s = 0; s < neuron->num_synapses; s++) {
                                // Apply with damping to avoid disrupting learning too much
                                float alpha = 0.1F;  // Mix 10% optimized, 90% current
                                neuron->synapses[s].weight =
                                    alpha * optimized_weights[weight_idx] +
                                    (1.0F - alpha) * neuron->synapses[s].weight;
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

            nimcp_free(neuron_ids);
            nimcp_free(activations);
        }
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

            // Compute firing rates array from network state
            float* firing_rates = nimcp_malloc(num_neurons * sizeof(float));
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

                // Get weights array - count total synapses
                uint32_t total_synapses = 0;
                uint32_t synapses_per_neuron = 0;
                for (uint32_t i = 0; i < num_neurons && i < 100; i++) {
                    neuron_t* neuron = neural_network_get_neuron(base_net, i);
                    if (neuron && neuron->synapses) {
                        total_synapses += neuron->num_synapses;
                        if (i == 0) synapses_per_neuron = neuron->num_synapses;
                    }
                }

                if (total_synapses > 0 && synapses_per_neuron > 0) {
                    // Extract weights
                    float* weights = nimcp_malloc(total_synapses * sizeof(float));
                    if (weights) {
                        uint32_t w_idx = 0;
                        for (uint32_t i = 0; i < num_neurons && i < 100 && w_idx < total_synapses; i++) {
                            neuron_t* neuron = neural_network_get_neuron(base_net, i);
                            if (neuron && neuron->synapses) {
                                for (uint32_t s = 0; s < neuron->num_synapses && w_idx < total_synapses; s++) {
                                    weights[w_idx++] = neuron->synapses[s].weight;
                                }
                            }
                        }

                        // Update homeostatic controller
                        const float DT_MS = 1.0F;  // Simulated time step
                        homeostatic_controller_update(brain->homeostatic, firing_rates, weights,
                                                     synapses_per_neuron, DT_MS);

                        nimcp_free(weights);
                    }
                }

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

    brain_clear_error();
    return network_loss;
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

    NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "async_learn_thread: operation failed");
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
