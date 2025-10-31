/**
 * @file nimcp_salience.h
 * @brief Salience and attention evaluation for active consciousness
 *
 * WHAT: Fast evaluation of input "interestingness" without full decision
 * WHY: Active consciousness needs to know what to pay attention to
 * HOW: Partial network activation + novelty detection + surprise measurement
 *
 * DESIGN RATIONALE:
 * Full brain decisions are expensive (~1ms). Salience evaluation is ~10x faster
 * (~0.1ms) by using:
 * - Early network layers only (not full propagation)
 * - Recent input history for novelty detection
 * - Prediction error for surprise measurement
 * - Heuristic urgency scoring
 *
 * DESIGN PATTERNS:
 * - Strategy Pattern: Different salience computation strategies
 * - Memento Pattern: Recent input history for comparison
 * - Observer Pattern: Attention threshold callbacks
 * - Factory Pattern: Salience evaluator creation
 *
 * BIOLOGICAL INSPIRATION:
 * Human attention is selective - we notice:
 * 1. Novel stimuli (never seen before)
 * 2. Surprising stimuli (violated expectations)
 * 3. Salient stimuli (important/intense)
 * 4. Urgent stimuli (requires immediate response)
 *
 * This API models those four attention mechanisms.
 *
 * EXAMPLE:
 * @code
 *   // Create salience evaluator
 *   salience_evaluator_t eval = salience_evaluator_create(brain, &config);
 *
 *   // Fast salience check (~0.1ms vs 1ms for full decision)
 *   brain_salience_t salience = brain_evaluate_salience(eval, features, 13);
 *
 *   if (salience.urgency > 0.9) {
 *       // Immediate LLM reasoning needed
 *       response = llm_deliberate(features);
 *   } else if (salience.novelty > 0.8) {
 *       // Novel input - learn from it
 *       brain_learn_example(brain, features, label, 1.0);
 *   } else if (salience.surprise > 0.7) {
 *       // Unexpected - update predictions
 *       brain_update_predictions(brain, features);
 *   } else {
 *       // Normal - fast neural path
 *       decision = brain_decide(brain, features, num_features);
 *   }
 *
 *   salience_evaluator_destroy(eval);
 * @endcode
 */

#ifndef NIMCP_SALIENCE_H
#define NIMCP_SALIENCE_H

#include "nimcp_brain.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// Forward Declarations and Opaque Types
//=============================================================================

/**
 * WHAT: Opaque salience evaluator handle
 * WHY: Hides implementation details, maintains encapsulation
 * PATTERN: Opaque pointer (Pimpl idiom)
 */
typedef struct salience_evaluator_struct* salience_evaluator_t;

//=============================================================================
// Salience Metrics and Structures
//=============================================================================

/**
 * WHAT: Comprehensive salience scores for an input
 * WHY: Multiple dimensions of "interestingness" for different use cases
 * HOW: Each metric computed independently, combined for overall salience
 *
 * INTERPRETATION:
 * - All values are 0.0-1.0 (normalized)
 * - Higher values = more attention-worthy
 * - Use individual metrics or combined salience depending on needs
 */
typedef struct {
    /**
     * WHAT: Overall salience (combined metric)
     * WHY: Single score for general attention allocation
     * HOW: Weighted combination of novelty, surprise, urgency
     * RANGE: 0.0 (ignorable) to 1.0 (critical attention)
     */
    float salience;

    /**
     * WHAT: Novelty score (how different from recent inputs)
     * WHY: Novel stimuli deserve attention (learning opportunity)
     * HOW: Compare to recent input history, measure distance
     * RANGE: 0.0 (very familiar) to 1.0 (completely novel)
     *
     * BIOLOGICAL: Human attention drawn to novel stimuli
     */
    float novelty;

    /**
     * WHAT: Surprise score (prediction error magnitude)
     * WHY: Surprising events indicate model needs updating
     * HOW: |predicted - actual| for features
     * RANGE: 0.0 (expected) to 1.0 (totally unexpected)
     *
     * BIOLOGICAL: Prediction errors drive learning
     */
    float surprise;

    /**
     * WHAT: Urgency score (needs immediate attention)
     * WHY: Some situations require fast response
     * HOW: Confidence variance + temporal dynamics + learned urgency patterns
     * RANGE: 0.0 (can wait) to 1.0 (immediate action required)
     *
     * BIOLOGICAL: Threat detection, reflexive responses
     */
    float urgency;

    /**
     * WHAT: Confidence in salience evaluation
     * WHY: Sometimes salience itself is uncertain
     * HOW: Based on input quality, history depth, model certainty
     * RANGE: 0.0 (uncertain salience) to 1.0 (confident salience)
     */
    float confidence;

    /**
     * WHAT: Computational cost of full decision
     * WHY: Cost-benefit analysis for attention allocation
     * HOW: Estimated from network size, complexity
     * RANGE: 0.0 (cheap) to 1.0 (expensive)
     *
     * USE CASE: If salience is low but cost is high, skip full decision
     */
    float estimated_cost;

} brain_salience_t;

/**
 * WHAT: Salience computation strategy
 * WHY: Different applications need different salience metrics
 * HOW: Strategy pattern - behavior changes based on strategy
 */
typedef enum {
    /**
     * WHAT: Fast heuristic salience
     * WHY: Maximum speed, acceptable accuracy
     * WHEN: High-frequency input, latency critical
     * ACCURACY: ~80%
     * SPEED: ~0.05ms
     */
    SALIENCE_STRATEGY_FAST,

    /**
     * WHAT: Balanced salience evaluation
     * WHY: Good speed-accuracy tradeoff
     * WHEN: Normal operation
     * ACCURACY: ~90%
     * SPEED: ~0.1ms
     */
    SALIENCE_STRATEGY_BALANCED,

    /**
     * WHAT: Accurate deep salience
     * WHY: Maximum accuracy for critical decisions
     * WHEN: Low-frequency input, accuracy critical
     * ACCURACY: ~95%
     * SPEED: ~0.5ms
     */
    SALIENCE_STRATEGY_ACCURATE

} salience_strategy_t;

/**
 * WHAT: Salience evaluator configuration
 * WHY: Flexible configuration for different use cases
 * PATTERN: Builder pattern - step-by-step configuration
 */
typedef struct {
    // Strategy selection
    salience_strategy_t strategy;      /**< Computation strategy */

    // History for novelty detection
    uint32_t history_size;             /**< How many recent inputs to remember */
    bool enable_novelty;               /**< Compute novelty scores? */

    // Prediction for surprise detection
    bool enable_surprise;              /**< Compute surprise scores? */
    bool enable_prediction;            /**< Maintain predictive model? */

    // Urgency detection
    bool enable_urgency;               /**< Compute urgency scores? */
    float urgency_baseline;            /**< Baseline urgency (0-1) */

    // Weighting for combined salience
    float novelty_weight;              /**< Weight for novelty (default: 0.3) */
    float surprise_weight;             /**< Weight for surprise (default: 0.4) */
    float urgency_weight;              /**< Weight for urgency (default: 0.3) */

    // Attention thresholds (for callbacks)
    float high_salience_threshold;     /**< Trigger callback above this */
    float high_novelty_threshold;      /**< Trigger callback for novel */
    float high_surprise_threshold;     /**< Trigger callback for surprising */
    float high_urgency_threshold;      /**< Trigger callback for urgent */

    // Performance tuning
    bool enable_caching;               /**< Cache recent evaluations? */
    uint32_t cache_size;               /**< Cache size if enabled */

} salience_config_t;

/**
 * WHAT: Salience event types for callbacks
 * WHY: Notify application when attention thresholds exceeded
 * PATTERN: Observer pattern
 */
typedef enum {
    SALIENCE_EVENT_HIGH_SALIENCE,      /**< Overall salience high */
    SALIENCE_EVENT_HIGH_NOVELTY,       /**< Novel input detected */
    SALIENCE_EVENT_HIGH_SURPRISE,      /**< Surprising input detected */
    SALIENCE_EVENT_HIGH_URGENCY        /**< Urgent input detected */
} salience_event_type_t;

/**
 * WHAT: Salience event data for callbacks
 * WHY: Provide context about attention event
 */
typedef struct {
    salience_event_type_t type;        /**< Event type */
    brain_salience_t salience;         /**< Salience scores */
    const float* features;             /**< Input features (read-only) */
    uint32_t num_features;             /**< Feature count */
    uint64_t timestamp;                /**< When event occurred */
    const char* message;               /**< Human-readable description */
} salience_event_t;

/**
 * WHAT: Salience event callback function type
 * WHY: Allow application to react to attention events
 * PATTERN: Observer pattern - callback is the observer
 *
 * @param event Event that occurred
 * @param context User-provided context pointer
 */
typedef void (*salience_event_callback_fn)(const salience_event_t* event, void* context);

//=============================================================================
// Salience Evaluator Lifecycle (Factory Pattern)
//=============================================================================

/**
 * WHAT: Create salience evaluator
 * WHY: Factory function - single creation point with validation
 * HOW: Validates config, allocates resources, initializes history
 *
 * COMPLEXITY: O(h) where h = history_size
 * THREAD-SAFE: Yes
 * PATTERN: Factory pattern
 *
 * @param brain Brain to evaluate salience for
 * @param config Salience configuration (copied internally)
 * @return Evaluator handle or NULL on error
 *
 * ERROR CONDITIONS:
 * - NULL brain: Returns NULL, sets error "Invalid brain"
 * - NULL config: Returns NULL, sets error "Invalid config"
 * - Invalid history_size: Returns NULL, sets error "History size too large"
 */
salience_evaluator_t salience_evaluator_create(
    brain_t brain,
    const salience_config_t* config
);

/**
 * WHAT: Destroy salience evaluator and free resources
 * WHY: Clean shutdown and memory cleanup
 * HOW: Frees history, cache, and structure
 *
 * COMPLEXITY: O(h + c) where h = history_size, c = cache_size
 * THREAD-SAFE: Yes (but evaluator should not be used after this call)
 *
 * @param evaluator Evaluator to destroy (NULL is safe)
 */
void salience_evaluator_destroy(salience_evaluator_t evaluator);

//=============================================================================
// Salience Evaluation Functions
//=============================================================================

/**
 * WHAT: Evaluate salience of single input (fast)
 * WHY: Quick check of "interestingness" before expensive full decision
 * HOW: Partial network activation + history comparison + prediction error
 *
 * COMPLEXITY: O(1) - constant time heuristics
 * THREAD-SAFE: Yes (uses internal mutex)
 * PERFORMANCE: ~0.1ms (10x faster than brain_decide)
 *
 * @param evaluator Salience evaluator handle
 * @param features Input feature vector
 * @param num_features Size of feature vector
 * @return Salience scores, or all zeros on error
 *
 * BEHAVIOR:
 * - Computes novelty by comparing to recent history
 * - Computes surprise by comparing to prediction
 * - Computes urgency using learned patterns
 * - Combines into overall salience score
 * - Updates history for next evaluation
 * - Triggers callbacks if thresholds exceeded
 *
 * USE CASE:
 * @code
 *   brain_salience_t s = brain_evaluate_salience(eval, features, 13);
 *   if (s.urgency > 0.9) {
 *       // Immediate attention needed
 *       handle_urgent(features);
 *   } else if (s.novelty > 0.8) {
 *       // Novel - worth learning from
 *       brain_learn_example(brain, features, label, 1.0);
 *   } else if (s.salience < 0.3) {
 *       // Not interesting - skip expensive decision
 *       return;
 *   } else {
 *       // Normal - proceed with decision
 *       decision = brain_decide(brain, features, num_features);
 *   }
 * @endcode
 */
brain_salience_t brain_evaluate_salience(
    salience_evaluator_t evaluator,
    const float* features,
    uint32_t num_features
);

/**
 * WHAT: Evaluate salience for batch of inputs (efficient)
 * WHY: Batch processing is much faster than individual evaluations
 * HOW: Vectorized operations, shared history comparisons
 *
 * COMPLEXITY: O(n) where n = num_samples (but with better constant factor)
 * THREAD-SAFE: Yes
 * PERFORMANCE: ~10-100x faster than N individual calls
 *
 * @param evaluator Salience evaluator handle
 * @param features Array of feature vectors
 * @param num_samples Number of samples
 * @param num_features Features per sample
 * @param salience_scores Output array (pre-allocated, size = num_samples)
 * @return Number of samples successfully evaluated
 *
 * USE CASE:
 * @code
 *   // Evaluate 100 inputs
 *   brain_salience_t scores[100];
 *   brain_evaluate_salience_batch(eval, features, 100, 13, scores);
 *
 *   // Process only high-salience inputs
 *   for (int i = 0; i < 100; i++) {
 *       if (scores[i].salience > 0.7) {
 *           process_input(features[i]);
 *       }
 *   }
 * @endcode
 */
uint32_t brain_evaluate_salience_batch(
    salience_evaluator_t evaluator,
    const float** features,
    uint32_t num_samples,
    uint32_t num_features,
    brain_salience_t* salience_scores
);

/**
 * WHAT: Evaluate salience with explicit timestamp
 * WHY: Temporal dynamics affect salience (recency, rate of change)
 * HOW: Uses timestamp for temporal weighting and decay
 *
 * @param evaluator Salience evaluator handle
 * @param features Input features
 * @param num_features Feature count
 * @param timestamp Input timestamp (milliseconds)
 * @return Salience scores
 */
brain_salience_t brain_evaluate_salience_temporal(
    salience_evaluator_t evaluator,
    const float* features,
    uint32_t num_features,
    uint64_t timestamp
);

//=============================================================================
// Salience Configuration and Control
//=============================================================================

/**
 * WHAT: Update salience weights dynamically
 * WHY: Adapt attention mechanism to changing conditions
 * HOW: Updates internal weights without rebuilding evaluator
 *
 * @param evaluator Evaluator handle
 * @param novelty_weight Weight for novelty (0-1)
 * @param surprise_weight Weight for surprise (0-1)
 * @param urgency_weight Weight for urgency (0-1)
 * @return true on success
 *
 * NOTE: Weights will be normalized to sum to 1.0
 */
bool salience_set_weights(
    salience_evaluator_t evaluator,
    float novelty_weight,
    float surprise_weight,
    float urgency_weight
);

/**
 * WHAT: Set attention thresholds for callbacks
 * WHY: Dynamically adjust attention sensitivity
 * HOW: Updates thresholds in configuration
 *
 * @param evaluator Evaluator handle
 * @param high_salience_threshold Overall salience threshold (0-1)
 * @param high_novelty_threshold Novelty threshold (0-1)
 * @param high_surprise_threshold Surprise threshold (0-1)
 * @param high_urgency_threshold Urgency threshold (0-1)
 * @return true on success
 */
bool salience_set_thresholds(
    salience_evaluator_t evaluator,
    float high_salience_threshold,
    float high_novelty_threshold,
    float high_surprise_threshold,
    float high_urgency_threshold
);

/**
 * WHAT: Register callback for attention events
 * WHY: Observer pattern - be notified when interesting inputs arrive
 * HOW: Stores callback function pointer and context
 *
 * @param evaluator Evaluator handle
 * @param callback Callback function
 * @param context User context passed to callback
 * @return true on success
 */
bool salience_register_callback(
    salience_evaluator_t evaluator,
    salience_event_callback_fn callback,
    void* context
);

//=============================================================================
// Salience History and State Management
//=============================================================================

/**
 * WHAT: Clear evaluation history
 * WHY: Reset novelty detection after context change
 * HOW: Empties recent input buffer
 *
 * @param evaluator Evaluator handle
 * @return true on success
 *
 * USE CASE: When switching to new conversation or domain
 */
bool salience_clear_history(salience_evaluator_t evaluator);

/**
 * WHAT: Get evaluation statistics
 * WHY: Monitor salience evaluator performance
 * HOW: Returns internal counters and metrics
 *
 * @param evaluator Evaluator handle
 * @param stats Output parameter for statistics
 * @return true on success
 */
typedef struct {
    uint64_t evaluations_performed;      /**< Total evaluations */
    uint64_t high_salience_count;        /**< High salience detections */
    uint64_t high_novelty_count;         /**< High novelty detections */
    uint64_t high_surprise_count;        /**< High surprise detections */
    uint64_t high_urgency_count;         /**< High urgency detections */

    float avg_salience;                  /**< Average overall salience */
    float avg_novelty;                   /**< Average novelty */
    float avg_surprise;                  /**< Average surprise */
    float avg_urgency;                   /**< Average urgency */

    float avg_evaluation_time_us;        /**< Average time per evaluation (microseconds) */

    uint32_t history_size;               /**< Current history depth */
    uint32_t cache_hit_rate;             /**< Cache hits (if enabled) */
} salience_stats_t;

bool salience_get_stats(salience_evaluator_t evaluator, salience_stats_t* stats);

/**
 * WHAT: Reset salience statistics
 * WHY: Allow clearing counters for fresh measurements
 * HOW: Resets all statistics counters to zero
 *
 * @param evaluator Evaluator handle
 * @return true on success, false on error
 */
bool salience_reset_stats(salience_evaluator_t evaluator);

//=============================================================================
// Convenience Functions
//=============================================================================

/**
 * WHAT: Create default salience configuration
 * WHY: Sensible defaults for common use case
 * HOW: Returns pre-filled config struct
 *
 * @return Default configuration
 */
salience_config_t salience_default_config(void);

/**
 * WHAT: Quick salience check (convenience wrapper)
 * WHY: One-shot evaluation without creating evaluator
 * HOW: Creates temporary evaluator, evaluates, destroys
 *
 * PERFORMANCE: Slower than reusing evaluator (evaluator creation overhead)
 * USE CASE: One-off evaluations, prototyping
 *
 * @param brain Brain to evaluate with
 * @param features Input features
 * @param num_features Feature count
 * @return Salience scores
 */
brain_salience_t salience_quick_evaluate(
    brain_t brain,
    const float* features,
    uint32_t num_features
);

//=============================================================================
// Error Handling
//=============================================================================

/**
 * WHAT: Get last salience error message
 * WHY: Thread-safe error reporting
 * HOW: Thread-local storage for error strings
 *
 * @return Error message string (valid until next salience call)
 */
const char* salience_get_last_error(void);

#ifdef __cplusplus
}
#endif

#endif // NIMCP_SALIENCE_H
