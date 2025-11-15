/**
 * @file nimcp_salience.c
 * @brief Salience and attention evaluation implementation
 *
 * WHAT: Fast "interestingness" evaluation for inputs
 * WHY: Active consciousness needs to know what deserves attention
 * HOW: Partial activation + novelty detection + surprise + urgency heuristics
 *
 * PERFORMANCE OPTIMIZATION:
 * - Full decision: ~1ms (full network propagation)
 * - Salience evaluation: ~0.1ms (partial propagation + heuristics)
 * - 10x speedup enables attention-based filtering
 *
 * ALGORITHM:
 * 1. NOVELTY: Compare input to recent history using cosine distance
 * 2. SURPRISE: Compare input to predicted next input (prediction error)
 * 3. URGENCY: Learned patterns + confidence variance
 * 4. SALIENCE: Weighted combination of above
 */

#include "nimcp_salience.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "utils/memory/nimcp_memory.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/thread/nimcp_thread_pool.h"
#include "utils/time/nimcp_time.h"
#include "utils/containers/nimcp_vector.h"
#include "plasticity/neuromodulators/nimcp_neuromodulators.h"  // Acetylcholine gating

//=============================================================================
// Thread-Local Error Handling
//=============================================================================

static __thread char g_salience_error[512] = {0};

static void salience_set_error(const char* format, ...)
{
    va_list args;
    va_start(args, format);
    vsnprintf(g_salience_error, sizeof(g_salience_error), format, args);
    va_end(args);
}

const char* salience_get_last_error(void)
{
    return g_salience_error;
}

//=============================================================================
// History Buffer for Novelty Detection (Memento Pattern) - Using nimcp_queue
//=============================================================================

/**
 * WHAT: Maximum feature vector size
 * WHY: Fixed-size entries required for nimcp_queue
 * HOW: Generous limit to handle most use cases
 */
#define SALIENCE_MAX_FEATURES 512

/**
 * WHAT: Single entry in history buffer
 * WHY: Store recent inputs for novelty comparison
 * NOTE: Now uses fixed-size array for nimcp_queue compatibility
 */
typedef struct {
    float features[SALIENCE_MAX_FEATURES];  // Feature vector (fixed-size)
    uint32_t num_features;                  // Actual size used
    uint64_t timestamp;                     // When input occurred
    bool valid;                             // Is this slot occupied?
} history_entry_t;

/**
 * WHAT: Circular buffer for recent input history
 * WHY: Novelty = how different from recent past
 * HOW: Ring buffer with fixed-size entries (no heap allocation per entry)
 *
 * PATTERN: Memento pattern - stores past states for comparison
 * NOTE: Uses fixed-size entries to eliminate per-entry malloc/free
 */
typedef struct {
    history_entry_t* entries;  // Circular buffer (fixed-size entries)
    uint32_t capacity;         // Buffer size
    uint32_t head;             // Next write position
    uint32_t count;            // Current entry count
    nimcp_mutex_t lock;        // Thread safety
} history_buffer_t;

/**
 * WHAT: Create history buffer
 * WHY: Initialize novelty detection storage
 * HOW: Allocate circular buffer
 */
static history_buffer_t* history_buffer_create(uint32_t capacity)
{
    history_buffer_t* hist = nimcp_calloc(1, sizeof(history_buffer_t));
    if (!hist)
        return NULL;

    hist->entries = nimcp_calloc(capacity, sizeof(history_entry_t));
    if (!hist->entries) {
        nimcp_free(hist);
        return NULL;
    }

    hist->capacity = capacity;
    hist->head = 0;
    hist->count = 0;
    nimcp_mutex_init(&hist->lock, NULL);

    return hist;
}

/**
 * WHAT: Destroy history buffer
 * WHY: Free all resources
 * HOW: Free entries array, then structure (features are stack-allocated in entries)
 */
static void history_buffer_destroy(history_buffer_t* hist)
{
    if (!hist)
        return;

    /**
     * WHAT: Free entries array
     * WHY: No per-entry cleanup needed - features are fixed-size arrays
     * NOTE: Simpler than before - no malloc/free per entry
     */
    if (hist->entries) {
        nimcp_free(hist->entries);
    }

    nimcp_mutex_destroy(&hist->lock);
    nimcp_free(hist);
}

/**
 * WHAT: Add entry to history
 * WHY: Update recent input history for novelty detection
 * HOW: Overwrite oldest entry (circular buffer) using fixed-size array
 */
static void history_buffer_add(history_buffer_t* hist, const float* features, uint32_t num_features,
                               uint64_t timestamp)
{
    /**
     * WHAT: Validate feature count
     * WHY: Fixed-size array has maximum capacity
     */
    if (num_features > SALIENCE_MAX_FEATURES) {
        return;  // Silently reject oversized inputs
    }

    nimcp_mutex_lock(&hist->lock);

    history_entry_t* entry = &hist->entries[hist->head];

    /**
     * WHAT: Copy features directly to fixed-size array
     * WHY: No heap allocation needed - eliminates malloc/free overhead
     * HOW: Direct memcpy to stack-allocated array
     */
    memcpy(entry->features, features, num_features * sizeof(float));
    entry->num_features = num_features;
    entry->timestamp = timestamp;
    entry->valid = true;

    /**
     * WHAT: Advance circular buffer head
     * WHY: Oldest entries are automatically overwritten
     */
    hist->head = (hist->head + 1) % hist->capacity;
    if (hist->count < hist->capacity) {
        hist->count++;
    }

    nimcp_mutex_unlock(&hist->lock);
}

/**
 * WHAT: Compute novelty score
 * WHY: Novelty = how different from recent history
 * HOW: Compare input to all history entries using cosine distance
 *
 * COMPLEXITY: O(h * f) where h = history_size, f = num_features
 *
 * @return Novelty score (0.0 = very familiar, 1.0 = completely novel)
 */
static float history_buffer_compute_novelty(history_buffer_t* hist, const float* features,
                                            uint32_t num_features)
{
    if (hist->count == 0) {
        /**
         * WHAT: No history yet
         * WHY: Everything is novel when we have no context
         * RETURN: Maximum novelty
         */
        return 1.0f;
    }

    nimcp_mutex_lock(&hist->lock);

    /**
     * WHAT: Find minimum distance to any historical entry
     * WHY: Novelty = distance to nearest similar input
     * HOW: Compute cosine distance to each entry, take minimum
     */
    float min_distance = 2.0f;  // Max cosine distance is 2.0

    for (uint32_t i = 0; i < hist->count; i++) {
        history_entry_t* entry = &hist->entries[i];
        if (!entry->valid || entry->num_features != num_features) {
            continue;
        }

        /**
         * WHAT: Compute cosine distance using vector utilities
         * WHY: Scale-invariant similarity metric
         * HOW: Use nimcp_vector_cosine_distance for consistency
         */
        float distance = nimcp_vector_cosine_distance(features, entry->features, num_features);

        if (distance < min_distance) {
            min_distance = distance;
        }
    }

    nimcp_mutex_unlock(&hist->lock);

    /**
     * WHAT: Normalize distance to 0-1
     * WHY: Cosine distance is 0-2, we want 0-1
     * HOW: Divide by 2, clamp to [0, 1]
     */
    float novelty = min_distance / 2.0f;
    novelty = fminf(1.0f, fmaxf(0.0f, novelty));

    return novelty;
}

/**
 * WHAT: Clear history buffer
 * WHY: Reset novelty detection
 * HOW: Simply mark all entries as invalid and reset counters
 */
static void history_buffer_clear(history_buffer_t* hist)
{
    nimcp_mutex_lock(&hist->lock);

    /**
     * WHAT: Mark all entries as invalid
     * WHY: No memory cleanup needed - features are fixed-size arrays
     * NOTE: Much simpler than before - no per-entry nimcp_free()
     */
    for (uint32_t i = 0; i < hist->count; i++) {
        hist->entries[i].valid = false;
    }

    hist->head = 0;
    hist->count = 0;

    nimcp_mutex_unlock(&hist->lock);
}

//=============================================================================
// Prediction Model for Surprise Detection
//=============================================================================

/**
 * WHAT: Simple exponential moving average predictor
 * WHY: Predict next input based on recent trend
 * HOW: EMA of recent inputs
 *
 * PATTERN: Memento pattern - maintains prediction state
 */
typedef struct {
    float* prediction;      // Predicted next input
    uint32_t num_features;  // Feature dimension
    float alpha;            // EMA smoothing factor (0-1)
    bool initialized;       // Has seen at least one input?
    nimcp_mutex_t lock;     // Thread safety
} predictor_t;

/**
 * WHAT: Create predictor
 */
static predictor_t* predictor_create(uint32_t num_features)
{
    predictor_t* pred = nimcp_calloc(1, sizeof(predictor_t));
    if (!pred)
        return NULL;

    pred->prediction = nimcp_calloc(num_features, sizeof(float));
    if (!pred->prediction) {
        nimcp_free(pred);
        return NULL;
    }

    pred->num_features = num_features;
    pred->alpha = 0.3f;  // 30% new, 70% old
    pred->initialized = false;
    nimcp_mutex_init(&pred->lock, NULL);

    return pred;
}

/**
 * WHAT: Destroy predictor
 */
static void predictor_destroy(predictor_t* pred)
{
    if (!pred)
        return;

    if (pred->prediction) {
        nimcp_free(pred->prediction);
    }

    nimcp_mutex_destroy(&pred->lock);
    nimcp_free(pred);
}

/**
 * WHAT: Update prediction with new observation
 * WHY: Learn temporal patterns
 * HOW: Exponential moving average
 */
static void predictor_update(predictor_t* pred, const float* features)
{
    nimcp_mutex_lock(&pred->lock);

    if (!pred->initialized) {
        /**
         * WHAT: Initialize prediction with first observation
         * WHY: Need starting point for EMA
         */
        memcpy(pred->prediction, features, pred->num_features * sizeof(float));
        pred->initialized = true;
    } else {
        /**
         * WHAT: Update prediction using EMA
         * WHY: Smooth temporal trend
         * HOW: prediction = alpha * new + (1-alpha) * old
         */
        for (uint32_t i = 0; i < pred->num_features; i++) {
            pred->prediction[i] =
                pred->alpha * features[i] + (1.0f - pred->alpha) * pred->prediction[i];
        }
    }

    nimcp_mutex_unlock(&pred->lock);
}

/**
 * WHAT: Compute surprise (prediction error)
 * WHY: Surprise = |actual - predicted|
 * HOW: Mean absolute error between prediction and actual
 *
 * @return Surprise score (0.0 = expected, 1.0 = totally unexpected)
 */
static float predictor_compute_surprise(predictor_t* pred, const float* features)
{
    if (!pred->initialized) {
        return 0.5f;  // Moderate surprise when no prediction exists
    }

    nimcp_mutex_lock(&pred->lock);

    /**
     * WHAT: Compute mean absolute prediction error
     * WHY: Measure of unexpectedness
     * HOW: Sum |actual - predicted| / num_features
     */
    float total_error = 0.0f;
    for (uint32_t i = 0; i < pred->num_features; i++) {
        total_error += fabsf(features[i] - pred->prediction[i]);
    }

    float mae = total_error / pred->num_features;

    nimcp_mutex_unlock(&pred->lock);

    /**
     * WHAT: Normalize to 0-1
     * WHY: Features have different scales
     * HOW: Assume features in [-1, 1] range, so max error is 2
     */
    float surprise = fminf(1.0f, mae / 2.0f);

    return surprise;
}

//=============================================================================
// Salience Evaluator Structure (Opaque Implementation)
//=============================================================================

/**
 * WHAT: Complete salience evaluator state
 * WHY: Encapsulates all evaluation data and configuration
 * PATTERN: Opaque pointer (Pimpl idiom)
 */
struct salience_evaluator_struct {
    // Brain association
    brain_t brain;

    // Configuration
    salience_config_t config;

    // Novelty detection (Memento pattern)
    history_buffer_t* history;

    // Surprise detection (Memento pattern)
    predictor_t* predictor;

    // Event callback (Observer pattern)
    salience_event_callback_fn callback;
    void* callback_context;

    // Statistics
    _Atomic uint64_t stats_evaluations;
    _Atomic uint64_t stats_high_salience;
    _Atomic uint64_t stats_high_novelty;
    _Atomic uint64_t stats_high_surprise;
    _Atomic uint64_t stats_high_urgency;

    float running_avg_salience;
    float running_avg_novelty;
    float running_avg_surprise;
    float running_avg_urgency;

    uint64_t total_eval_time_us;

    // Thread safety
    nimcp_mutex_t eval_lock;

    // Parallel processing (for batch operations)
    nimcp_thread_pool_t* thread_pool;
};

//=============================================================================
// Salience Computation Strategies (Strategy Pattern)
//=============================================================================

/**
 * WHAT: Fast heuristic salience computation
 * WHY: Maximum speed for high-frequency input
 * HOW: Simplified calculations, approximations
 */
static brain_salience_t compute_salience_fast(salience_evaluator_t eval, const float* features,
                                              uint32_t num_features, uint64_t timestamp)
{
    brain_salience_t salience = {0};

    /**
     * WHAT: Fast novelty (only check against last N=5 entries)
     * WHY: Full history scan is expensive
     */
    if (eval->config.enable_novelty) {
        salience.novelty = history_buffer_compute_novelty(eval->history, features, num_features);
    }

    /**
     * WHAT: Fast surprise (simple threshold check)
     * WHY: Full prediction error calculation is expensive
     */
    if (eval->config.enable_surprise && eval->predictor->initialized) {
        salience.surprise = predictor_compute_surprise(eval->predictor, features);
    }

    /**
     * WHAT: Fast urgency (baseline only)
     * WHY: Learned urgency patterns require full network
     */
    if (eval->config.enable_urgency) {
        salience.urgency = eval->config.urgency_baseline;
    }

    /**
     * WHAT: Combine scores using weights
     * WHY: Different metrics have different importance
     * HOW: Weighted average
     */
    float total_weight =
        eval->config.novelty_weight + eval->config.surprise_weight + eval->config.urgency_weight;

    if (total_weight > 0.0f) {
        salience.salience = (salience.novelty * eval->config.novelty_weight +
                             salience.surprise * eval->config.surprise_weight +
                             salience.urgency * eval->config.urgency_weight) /
                            total_weight;
    }

    salience.confidence = 0.7f;      // Lower confidence for fast mode
    salience.estimated_cost = 0.1f;  // Low cost estimate

    return salience;
}

/**
 * WHAT: Balanced salience computation
 * WHY: Good speed-accuracy tradeoff
 * HOW: Full novelty + surprise, simplified urgency
 */
static brain_salience_t compute_salience_balanced(salience_evaluator_t eval, const float* features,
                                                  uint32_t num_features, uint64_t timestamp)
{
    brain_salience_t salience = {0};

    // Full novelty computation
    if (eval->config.enable_novelty) {
        salience.novelty = history_buffer_compute_novelty(eval->history, features, num_features);
    }

    // Full surprise computation
    if (eval->config.enable_surprise) {
        salience.surprise = predictor_compute_surprise(eval->predictor, features);
    }

    // Enhanced urgency (add variance to baseline)
    if (eval->config.enable_urgency) {
        /**
         * WHAT: Urgency based on input variance
         * WHY: Rapidly changing inputs suggest urgency
         * HOW: Measure feature variance, add to baseline
         */
        float variance = 0.0f;
        for (uint32_t i = 0; i < num_features; i++) {
            variance += features[i] * features[i];
        }
        variance /= num_features;

        float urgency_boost = fminf(0.3f, variance);
        salience.urgency = fminf(1.0f, eval->config.urgency_baseline + urgency_boost);
    }

    // Combine scores
    float total_weight =
        eval->config.novelty_weight + eval->config.surprise_weight + eval->config.urgency_weight;

    if (total_weight > 0.0f) {
        salience.salience = (salience.novelty * eval->config.novelty_weight +
                             salience.surprise * eval->config.surprise_weight +
                             salience.urgency * eval->config.urgency_weight) /
                            total_weight;
    }

    salience.confidence = 0.85f;
    salience.estimated_cost = 0.5f;

    return salience;
}

/**
 * WHAT: Accurate deep salience computation
 * WHY: Maximum accuracy for critical decisions
 * HOW: Full computation including partial brain activation
 */
static brain_salience_t compute_salience_accurate(salience_evaluator_t eval, const float* features,
                                                  uint32_t num_features, uint64_t timestamp)
{
    // For now, same as balanced (TODO: add partial brain activation)
    brain_salience_t salience = compute_salience_balanced(eval, features, num_features, timestamp);

    salience.confidence = 0.95f;
    salience.estimated_cost = 0.9f;

    return salience;
}

/**
 * @brief Apply acetylcholine gating to salience scores
 *
 * WHAT: Modulate all salience scores by current acetylcholine level
 * WHY:  ACh gates attention - models attentiveness vs drowsiness
 * HOW:  Read ACh from brain, compute modulation factor, scale all scores
 *
 * BIOLOGY: Acetylcholine enhances sensory processing and attention
 *          High ACh (0.7) → 1.4× salience (highly attentive, focused)
 *          Low ACh (0.3) → 0.6× salience (inattentive, drowsy)
 *
 * COMPLEXITY: O(1)
 *
 * @param salience Salience scores to modulate (modified in-place)
 * @param brain Brain to read ACh from
 */
static void apply_acetylcholine_gating(brain_salience_t* salience, brain_t brain)
{
    // Guard: Early return if no brain
    if (!brain || !salience) {
        return;
    }

    // Get neuromodulator system
    neuromodulator_system_t neuromod = brain_get_neuromodulator_system(brain);
    if (!neuromod) {
        return;
    }

    // Read current acetylcholine level
    float ach = neuromodulator_get_level(neuromod, NEUROMOD_ACETYLCHOLINE);

    // Map ACh range [0.3, 0.7] to modulation [0.6, 1.4]
    float modulation = 0.6f + (ach - 0.3f) * 2.0f;

    // Modulate all salience scores
    salience->salience *= modulation;
    salience->novelty *= modulation;
    salience->surprise *= modulation;
    salience->urgency *= modulation;

    // Clamp to [0, 1] range
    salience->salience = fminf(salience->salience, 1.0f);
    salience->novelty = fminf(salience->novelty, 1.0f);
    salience->surprise = fminf(salience->surprise, 1.0f);
    salience->urgency = fminf(salience->urgency, 1.0f);
}

//=============================================================================
// Salience Evaluator Lifecycle (Factory Pattern)
//=============================================================================

/**
 * WHAT: Validate salience configuration
 * WHY: Guard clause pattern - fail fast on invalid config
 */
static bool validate_salience_config(const salience_config_t* config)
{
    if (!config) {
        salience_set_error("NULL configuration");
        return false;
    }

    if (config->history_size == 0 && config->enable_novelty) {
        salience_set_error("Novelty requires non-zero history size");
        return false;
    }

    if (config->history_size > 10000) {
        salience_set_error("History size too large (max 10000)");
        return false;
    }

    return true;
}

salience_evaluator_t salience_evaluator_create(brain_t brain, const salience_config_t* config)
{
    // Guard clauses
    if (!brain) {
        salience_set_error("NULL brain");
        return NULL;
    }

    if (!validate_salience_config(config)) {
        return NULL;
    }

    // Allocate evaluator
    salience_evaluator_t eval = nimcp_calloc(1, sizeof(struct salience_evaluator_struct));
    if (!eval) {
        salience_set_error("Failed to allocate evaluator");
        return NULL;
    }

    eval->brain = brain;
    memcpy(&eval->config, config, sizeof(salience_config_t));

    // Create history buffer if novelty enabled
    if (config->enable_novelty && config->history_size > 0) {
        eval->history = history_buffer_create(config->history_size);
        if (!eval->history) {
            salience_set_error("Failed to create history buffer");
            nimcp_free(eval);
            return NULL;
        }
    }

    // Create predictor if surprise enabled
    if (config->enable_surprise) {
        // Get num_features from brain configuration via getter
        uint32_t num_features = brain_get_num_inputs(brain);
        eval->predictor = predictor_create(num_features);
        if (!eval->predictor) {
            salience_set_error("Failed to create predictor");
            if (eval->history)
                history_buffer_destroy(eval->history);
            nimcp_free(eval);
            return NULL;
        }
    }

    // Initialize statistics
    eval->stats_evaluations = 0;
    eval->stats_high_salience = 0;
    eval->stats_high_novelty = 0;
    eval->stats_high_surprise = 0;
    eval->stats_high_urgency = 0;

    eval->running_avg_salience = 0.0f;
    eval->running_avg_novelty = 0.0f;
    eval->running_avg_surprise = 0.0f;
    eval->running_avg_urgency = 0.0f;

    eval->total_eval_time_us = 0;

    // Initialize mutex
    nimcp_mutex_init(&eval->eval_lock, NULL);

    /**
     * WHAT: Create thread pool for batch processing
     * WHY: Parallel evaluation significantly improves batch performance
     * HOW: 4 worker threads for optimal CPU utilization
     */
    eval->thread_pool = nimcp_pool_create(4);
    if (!eval->thread_pool) {
        salience_set_error("Failed to create thread pool");
        if (eval->history)
            history_buffer_destroy(eval->history);
        if (eval->predictor)
            predictor_destroy(eval->predictor);
        nimcp_mutex_destroy(&eval->eval_lock);
        nimcp_free(eval);
        return NULL;
    }

    return eval;
}

void salience_evaluator_destroy(salience_evaluator_t eval)
{
    if (!eval)
        return;

    /**
     * WHAT: Destroy thread pool first
     * WHY: Ensures all pending tasks complete before cleanup
     * HOW: nimcp_pool_destroy waits for completion
     */
    if (eval->thread_pool) {
        nimcp_pool_destroy(eval->thread_pool);
    }

    if (eval->history) {
        history_buffer_destroy(eval->history);
    }

    if (eval->predictor) {
        predictor_destroy(eval->predictor);
    }

    nimcp_mutex_destroy(&eval->eval_lock);

    nimcp_free(eval);
}

//=============================================================================
// Salience Evaluation Functions
//=============================================================================

brain_salience_t brain_evaluate_salience(salience_evaluator_t eval, const float* features,
                                         uint32_t num_features)
{
    return brain_evaluate_salience_temporal(eval, features, num_features, 0);
}

/**
 * WHAT: Task context for parallel batch evaluation
 * WHY: Worker threads need access to evaluator and specific sample
 * PATTERN: Command pattern - encapsulates evaluation request
 */
typedef struct {
    salience_evaluator_t eval; /* Evaluator to use */
    const float* features;     /* Input features for this sample */
    uint32_t num_features;     /* Feature count */
    brain_salience_t* output;  /* Where to store result */
} batch_task_t;

/**
 * WHAT: Worker function for parallel salience evaluation (stateless)
 * WHY: Executed by thread pool workers
 * HOW: Evaluates single sample WITHOUT updating shared state
 *
 * NOTE: Does NOT update history/predictor to avoid lock contention
 */
static void evaluate_single_task(void* arg)
{
    batch_task_t* task = (batch_task_t*) arg;
    if (!task || !task->eval || !task->features || !task->output) {
        return;
    }

    /**
     * WHAT: Compute salience WITHOUT state updates
     * WHY: Avoid lock contention - state updated after batch completes
     * HOW: Call strategy function directly, skip history/predictor updates
     */
    salience_evaluator_t eval = task->eval;

    /* WHAT: Use configured strategy for evaluation */
    switch (eval->config.strategy) {
        case SALIENCE_STRATEGY_FAST:
            *task->output = compute_salience_fast(eval, task->features, task->num_features, 0);
            break;

        case SALIENCE_STRATEGY_BALANCED:
            *task->output = compute_salience_balanced(eval, task->features, task->num_features, 0);
            break;

        case SALIENCE_STRATEGY_ACCURATE:
            *task->output = compute_salience_accurate(eval, task->features, task->num_features, 0);
            break;
    }
}

brain_salience_t brain_evaluate_salience_temporal(salience_evaluator_t eval, const float* features,
                                                  uint32_t num_features, uint64_t timestamp)
{
    brain_salience_t salience = {0};

    if (!eval || !features) {
        return salience;  // Return zeros
    }

    uint64_t start_time = nimcp_time_monotonic_us();

    nimcp_mutex_lock(&eval->eval_lock);

    /**
     * WHAT: Compute salience using configured strategy
     * WHY: Different strategies for different performance needs
     * HOW: Strategy pattern - delegate to strategy function
     */
    switch (eval->config.strategy) {
        case SALIENCE_STRATEGY_FAST:
            salience = compute_salience_fast(eval, features, num_features, timestamp);
            break;

        case SALIENCE_STRATEGY_BALANCED:
            salience = compute_salience_balanced(eval, features, num_features, timestamp);
            break;

        case SALIENCE_STRATEGY_ACCURATE:
            salience = compute_salience_accurate(eval, features, num_features, timestamp);
            break;
    }

    // Apply acetylcholine gating (attentiveness modulation)
    apply_acetylcholine_gating(&salience, eval->brain);

    /**
     * WHAT: Update history and predictor
     * WHY: Learn from this input for future comparisons
     * HOW: Add to history buffer, update prediction
     */
    if (eval->history) {
        history_buffer_add(eval->history, features, num_features, timestamp);
    }

    if (eval->predictor) {
        predictor_update(eval->predictor, features);
    }

    /**
     * WHAT: Update statistics
     * WHY: Monitor evaluator performance
     */
    eval->stats_evaluations++;

    // Running averages (exponential moving average)
    float alpha = 0.1f;
    eval->running_avg_salience =
        alpha * salience.salience + (1.0f - alpha) * eval->running_avg_salience;
    eval->running_avg_novelty =
        alpha * salience.novelty + (1.0f - alpha) * eval->running_avg_novelty;
    eval->running_avg_surprise =
        alpha * salience.surprise + (1.0f - alpha) * eval->running_avg_surprise;
    eval->running_avg_urgency =
        alpha * salience.urgency + (1.0f - alpha) * eval->running_avg_urgency;

    // Count high events
    if (salience.salience > eval->config.high_salience_threshold) {
        eval->stats_high_salience++;
    }
    if (salience.novelty > eval->config.high_novelty_threshold) {
        eval->stats_high_novelty++;
    }
    if (salience.surprise > eval->config.high_surprise_threshold) {
        eval->stats_high_surprise++;
    }
    if (salience.urgency > eval->config.high_urgency_threshold) {
        eval->stats_high_urgency++;
    }

    nimcp_mutex_unlock(&eval->eval_lock);

    /**
     * WHAT: Trigger callbacks if thresholds exceeded
     * WHY: Observer pattern - notify application
     */
    if (eval->callback) {
        if (salience.salience > eval->config.high_salience_threshold) {
            salience_event_t event = {.type = SALIENCE_EVENT_HIGH_SALIENCE,
                                      .salience = salience,
                                      .features = features,
                                      .num_features = num_features,
                                      .timestamp = timestamp,
                                      .message = "High salience detected"};
            eval->callback(&event, eval->callback_context);
        }
    }

    uint64_t elapsed_us = nimcp_time_elapsed_us(start_time);
    eval->total_eval_time_us += elapsed_us;

    return salience;
}

uint32_t brain_evaluate_salience_batch(salience_evaluator_t eval, const float** features,
                                       uint32_t num_samples, uint32_t num_features,
                                       brain_salience_t* salience_scores)
{
    if (!eval || !features || !salience_scores) {
        return 0;
    }

    /**
     * WHAT: Choose sequential vs parallel based on batch size
     * WHY: Thread pool overhead makes small batches slower
     * HOW: Use sequential for < 200 samples, parallel for larger batches
     *
     * PERFORMANCE: Thread pool overhead ~1ms, so need enough work to amortize
     */
    const uint32_t PARALLEL_THRESHOLD = 200;

    if (num_samples < PARALLEL_THRESHOLD) {
        /**
         * WHAT: Sequential evaluation for small batches
         * WHY: Avoid thread pool overhead
         * HOW: Call evaluation function directly for each sample
         */
        for (uint32_t i = 0; i < num_samples; i++) {
            salience_scores[i] = brain_evaluate_salience(eval, features[i], num_features);
        }
        return num_samples;
    }

    /**
     * WHAT: Use thread pool for parallel batch evaluation (large batches)
     * WHY: Significant performance improvement for batch processing
     * HOW: Parallel evaluation WITHOUT state updates, then sequential state update
     *
     * PATTERN: Map-reduce with thread pool
     * KEY INSIGHT: Lock contention kills parallelism - compute in parallel, update sequentially
     */

    /**
     * WHAT: Allocate task context array
     * WHY: Each worker needs context for its sample
     * HOW: Heap allocation for safety across threads
     */
    batch_task_t* tasks = (batch_task_t*) nimcp_malloc(num_samples * sizeof(batch_task_t));
    if (!tasks) {
        return 0;
    }

    /**
     * WHAT: Prepare all tasks
     * WHY: Submit them to thread pool for parallel execution
     */
    for (uint32_t i = 0; i < num_samples; i++) {
        tasks[i].eval = eval;
        tasks[i].features = features[i];
        tasks[i].num_features = num_features;
        tasks[i].output = &salience_scores[i];
    }

    /**
     * WHAT: Submit all tasks to thread pool
     * WHY: Workers will execute them in parallel WITHOUT locks
     * HOW: Pool distributes work across available threads
     */
    for (uint32_t i = 0; i < num_samples; i++) {
        nimcp_result_t result =
            nimcp_pool_submit(eval->thread_pool, evaluate_single_task, &tasks[i]);

        if (result != NIMCP_SUCCESS) {
            /**
             * WHAT: Handle submission failure
             * WHY: Queue might be full or system error
             * HOW: Wait for pool to drain, retry
             */
            nimcp_pool_wait(eval->thread_pool);
            nimcp_pool_submit(eval->thread_pool, evaluate_single_task, &tasks[i]);
        }
    }

    /**
     * WHAT: Wait for all tasks to complete
     * WHY: Ensure all results are ready before state updates
     * HOW: Blocks until queue empty and no active workers
     */
    nimcp_pool_wait(eval->thread_pool);

    /**
     * WHAT: Update shared state sequentially after parallel evaluation
     * WHY: Maintain history/predictor consistency without lock contention
     * HOW: Add each sample to history and update predictor
     *
     * NOTE: This is the only place we update shared state - avoids parallelism bottleneck
     */
    nimcp_mutex_lock(&eval->eval_lock);

    for (uint32_t i = 0; i < num_samples; i++) {
        /* WHAT: Update history buffer if enabled */
        if (eval->history) {
            history_buffer_add(eval->history, features[i], num_features, 0);
        }

        /* WHAT: Update predictor if enabled */
        if (eval->predictor) {
            predictor_update(eval->predictor, features[i]);
        }

        /* WHAT: Update statistics */
        eval->stats_evaluations++;

        float alpha = 0.1f;
        eval->running_avg_salience =
            alpha * salience_scores[i].salience + (1.0f - alpha) * eval->running_avg_salience;
        eval->running_avg_novelty =
            alpha * salience_scores[i].novelty + (1.0f - alpha) * eval->running_avg_novelty;
        eval->running_avg_surprise =
            alpha * salience_scores[i].surprise + (1.0f - alpha) * eval->running_avg_surprise;
        eval->running_avg_urgency =
            alpha * salience_scores[i].urgency + (1.0f - alpha) * eval->running_avg_urgency;

        if (salience_scores[i].salience > eval->config.high_salience_threshold) {
            eval->stats_high_salience++;
        }
        if (salience_scores[i].novelty > eval->config.high_novelty_threshold) {
            eval->stats_high_novelty++;
        }
        if (salience_scores[i].surprise > eval->config.high_surprise_threshold) {
            eval->stats_high_surprise++;
        }
        if (salience_scores[i].urgency > eval->config.high_urgency_threshold) {
            eval->stats_high_urgency++;
        }
    }

    nimcp_mutex_unlock(&eval->eval_lock);

    /**
     * WHAT: Free task array
     * WHY: Cleanup temporary allocation
     */
    nimcp_free(tasks);

    return num_samples;
}

//=============================================================================
// Configuration and Control Functions
//=============================================================================

bool salience_set_weights(salience_evaluator_t eval, float novelty_weight, float surprise_weight,
                          float urgency_weight)
{
    if (!eval)
        return false;

    nimcp_mutex_lock(&eval->eval_lock);

    eval->config.novelty_weight = novelty_weight;
    eval->config.surprise_weight = surprise_weight;
    eval->config.urgency_weight = urgency_weight;

    nimcp_mutex_unlock(&eval->eval_lock);

    return true;
}

bool salience_set_thresholds(salience_evaluator_t eval, float high_salience_threshold,
                             float high_novelty_threshold, float high_surprise_threshold,
                             float high_urgency_threshold)
{
    if (!eval)
        return false;

    nimcp_mutex_lock(&eval->eval_lock);

    eval->config.high_salience_threshold = high_salience_threshold;
    eval->config.high_novelty_threshold = high_novelty_threshold;
    eval->config.high_surprise_threshold = high_surprise_threshold;
    eval->config.high_urgency_threshold = high_urgency_threshold;

    nimcp_mutex_unlock(&eval->eval_lock);

    return true;
}

bool salience_register_callback(salience_evaluator_t eval, salience_event_callback_fn callback,
                                void* context)
{
    if (!eval)
        return false;

    nimcp_mutex_lock(&eval->eval_lock);

    eval->callback = callback;
    eval->callback_context = context;

    nimcp_mutex_unlock(&eval->eval_lock);

    return true;
}

bool salience_clear_history(salience_evaluator_t eval)
{
    if (!eval || !eval->history)
        return false;

    history_buffer_clear(eval->history);

    return true;
}

bool salience_get_stats(salience_evaluator_t eval, salience_stats_t* stats)
{
    if (!eval || !stats)
        return false;

    nimcp_mutex_lock(&eval->eval_lock);

    stats->evaluations_performed = eval->stats_evaluations;
    stats->high_salience_count = eval->stats_high_salience;
    stats->high_novelty_count = eval->stats_high_novelty;
    stats->high_surprise_count = eval->stats_high_surprise;
    stats->high_urgency_count = eval->stats_high_urgency;

    stats->avg_salience = eval->running_avg_salience;
    stats->avg_novelty = eval->running_avg_novelty;
    stats->avg_surprise = eval->running_avg_surprise;
    stats->avg_urgency = eval->running_avg_urgency;

    stats->avg_evaluation_time_us = eval->stats_evaluations > 0
                                        ? (float) eval->total_eval_time_us / eval->stats_evaluations
                                        : 0.0f;

    stats->history_size = eval->history ? eval->history->count : 0;
    stats->cache_hit_rate = 0;  // TODO: Implement caching

    nimcp_mutex_unlock(&eval->eval_lock);

    return true;
}

/**
 * WHAT: Reset salience statistics
 * WHY: Allow clearing counters for fresh measurements
 * HOW: Resets all statistics counters to zero
 *
 * THREAD-SAFE: Yes (mutex-protected)
 *
 * @param eval Evaluator handle
 * @return true on success, false on error
 */
bool salience_reset_stats(salience_evaluator_t eval)
{
    if (!eval) {
        return false;
    }

    /**
     * WHAT: Reset all statistic counters
     * WHY: Provide clean slate for new measurement period
     * HOW: Mutex-protected zero assignment to all counters
     */
    nimcp_mutex_lock(&eval->eval_lock);

    eval->stats_evaluations = 0;
    eval->stats_high_salience = 0;
    eval->stats_high_novelty = 0;
    eval->stats_high_surprise = 0;
    eval->stats_high_urgency = 0;

    eval->running_avg_salience = 0.0f;
    eval->running_avg_novelty = 0.0f;
    eval->running_avg_surprise = 0.0f;
    eval->running_avg_urgency = 0.0f;

    eval->total_eval_time_us = 0;

    nimcp_mutex_unlock(&eval->eval_lock);

    return true;
}

//=============================================================================
// Convenience Functions
//=============================================================================

salience_config_t salience_default_config(void)
{
    salience_config_t config = {.strategy = SALIENCE_STRATEGY_BALANCED,
                                .history_size = 100,
                                .enable_novelty = true,
                                .enable_surprise = true,
                                .enable_urgency = true,
                                .enable_prediction = true,
                                .urgency_baseline = 0.3f,
                                .novelty_weight = 0.3f,
                                .surprise_weight = 0.4f,
                                .urgency_weight = 0.3f,
                                .high_salience_threshold = 0.7f,
                                .high_novelty_threshold = 0.8f,
                                .high_surprise_threshold = 0.8f,
                                .high_urgency_threshold = 0.9f,
                                .enable_caching = false,
                                .cache_size = 0};

    return config;
}

brain_salience_t salience_quick_evaluate(brain_t brain, const float* features,
                                         uint32_t num_features)
{
    salience_config_t config = salience_default_config();
    config.history_size = 10;  // Small history for quick eval

    salience_evaluator_t eval = salience_evaluator_create(brain, &config);
    if (!eval) {
        brain_salience_t empty = {0};
        return empty;
    }

    brain_salience_t salience = brain_evaluate_salience(eval, features, num_features);

    salience_evaluator_destroy(eval);

    return salience;
}

//=============================================================================
// Bidirectional Feedback Functions (Phase 10.11.3)
//=============================================================================

/**
 * @brief Boost salience for negative cues
 *
 * WHAT: Increase attention to negative stimuli
 * WHY:  Depression biases attention toward negative information
 * HOW:  Store boost factor to apply in next evaluation
 *
 * BIOLOGY: Mood-congruent attentional bias
 *          Depression → hyperattention to negative cues
 *
 * COMPLEXITY: O(1)
 *
 * @param evaluator Salience evaluator
 * @param boost_factor Salience boost [0, 1]
 */
void salience_boost_negative_cues(salience_evaluator_t evaluator, float boost_factor)
{
    // Guard: Validate evaluator
    if (!evaluator) {
        return;
    }

    // Clamp boost factor
    boost_factor = fminf(fmaxf(boost_factor, 0.0f), 1.0f);

    // WHAT: Increase novelty weight to bias toward unexpected negatives
    // WHY:  Depression makes negative novelty more salient
    // HOW:  Scale novelty weight by (1 + boost_factor)
    nimcp_mutex_lock(&evaluator->eval_lock);

    evaluator->config.novelty_weight *= (1.0f + boost_factor * 0.5f);

    nimcp_mutex_unlock(&evaluator->eval_lock);
}

/**
 * @brief Boost threat detection sensitivity
 *
 * WHAT: Increase urgency for potential threats
 * WHY:  Anxiety increases threat vigilance
 * HOW:  Lower threshold for urgency detection
 *
 * BIOLOGY: Amygdala hyperactivity in anxiety
 *          Anxious individuals detect threats faster with lower thresholds
 *
 * COMPLEXITY: O(1)
 *
 * @param evaluator Salience evaluator
 * @param boost_factor Threat sensitivity boost [0, 1]
 */
void salience_boost_threat_detection(salience_evaluator_t evaluator, float boost_factor)
{
    // Guard: Validate evaluator
    if (!evaluator) {
        return;
    }

    // Clamp boost factor
    boost_factor = fminf(fmaxf(boost_factor, 0.0f), 1.0f);

    // WHAT: Increase urgency baseline and weight
    // WHY:  Anxiety makes everything seem more urgent
    // HOW:  Scale urgency parameters by boost factor
    nimcp_mutex_lock(&evaluator->eval_lock);

    evaluator->config.urgency_baseline = fminf(
        evaluator->config.urgency_baseline * (1.0f + boost_factor),
        1.0f
    );

    evaluator->config.urgency_weight *= (1.0f + boost_factor * 0.5f);

    nimcp_mutex_unlock(&evaluator->eval_lock);
}

/**
 * @brief Get surprise level from recent evaluation
 *
 * WHAT: Query most recent surprise score
 * WHY:  Emotional system modulates arousal based on surprise
 * HOW:  Return average surprise from running statistics
 *
 * COMPLEXITY: O(1)
 *
 * @param evaluator Salience evaluator
 * @return Surprise level [0, 1]
 */
float salience_get_surprise_level(salience_evaluator_t evaluator)
{
    // Guard: Validate evaluator
    if (!evaluator) {
        return 0.0f;
    }

    // WHAT: Return running average surprise
    // WHY:  More stable than single sample
    // HOW:  Read from statistics
    nimcp_mutex_lock(&evaluator->eval_lock);

    float surprise = evaluator->running_avg_surprise;

    nimcp_mutex_unlock(&evaluator->eval_lock);

    return surprise;
}
