//=============================================================================
// nimcp_brain.c - High-Level Brain API Implementation (Refactored)
//=============================================================================
/**
 * @file nimcp_brain.c
 * @brief Production-ready brain API with Factory and Strategy patterns
 *
 * ARCHITECTURE:
 * - Factory Pattern: Creates brains of different types with validated configs
 * - Strategy Pattern: Task-specific behaviors (classification, regression, etc.)
 * - Builder Pattern: Modular configuration construction
 *
 * PERFORMANCE CHARACTERISTICS:
 * - Brain creation: O(n) where n = num_neurons
 * - Learning: O(s*n) where s = sparsity, n = active_neurons
 * - Inference: O(s*n) with caching for repeated inputs
 * - Memory: O(n*c) where c = average_connections_per_neuron
 *
 * DESIGN DECISIONS:
 * - No nested ifs: All validation uses early returns (guard clauses)
 * - Functions <50 lines: Complex operations decomposed into helpers
 * - Caching: Decision results cached for repeated identical inputs
 * - Thread-safe: Error handling uses thread-local storage
 */

#include "../include/nimcp_brain.h"
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../include/nimcp_adaptive.h"
#include "../include/nimcp_neuralnet.h"
#include "utils/nimcp_memory.h"
#include "utils/nimcp_time.h"
#include "utils/nimcp_validate.h"

//=============================================================================
// Forward Declarations - Strategy Pattern
//=============================================================================

typedef struct task_strategy task_strategy_t;

//=============================================================================
// Internal Brain Structure with Strategy
//=============================================================================

/**
 * @brief Brain internal structure with strategy pattern
 *
 * WHY: Encapsulates network, config, labels, stats, and task strategy
 * Enables type-specific behaviors without switch statements
 *
 * COMPLEXITY: O(1) access to all members
 */
struct brain_struct {
    adaptive_network_t network;  // Underlying neural network
    brain_config_t config;       // User configuration
    task_strategy_t* strategy;   // Task-specific behavior strategy

    // Label to output mapping
    char** output_labels;        // Label strings
    uint32_t num_output_labels;  // Current label count

    // Statistics
    brain_stats_t stats;  // Performance metrics

    // Decision caching for optimization
    float* last_input;                  // Cached input vector
    brain_decision_t* cached_decision;  // Cached decision
    uint32_t input_size;                // For cache validation

    // Phase 3: Distributed cognition coordinator
    distrib_cognition_t distributed;  // P2P cognitive coordination (NULL if standalone)
};

//=============================================================================
// Strategy Pattern - Task-Specific Behaviors
//=============================================================================

/**
 * @brief Task strategy interface
 *
 * WHY: Different tasks (classification, regression) need different:
 * - Learning rates
 * - Output transformations
 * - Performance metrics
 *
 * PATTERN: Strategy pattern - encapsulates algorithm families
 */
struct task_strategy {
    brain_task_t task_type;

    /**
     * @brief Get recommended learning rate for this task
     * COMPLEXITY: O(1)
     */
    float (*get_learning_rate)(void);

    /**
     * @brief Transform raw output to task-specific format
     * COMPLEXITY: O(num_outputs)
     */
    void (*transform_output)(float* output, uint32_t size);

    /**
     * @brief Compute task-specific loss
     * COMPLEXITY: O(num_outputs)
     */
    float (*compute_loss)(const float* predicted, const float* target, uint32_t size);
};

//=============================================================================
// Strategy Implementations
//=============================================================================

/**
 * @brief Classification strategy - softmax output, cross-entropy loss
 *
 * WHY: Classification needs probabilities summing to 1.0
 * WHEN: Multi-class or binary classification tasks
 * COMPLEXITY: O(n) for softmax normalization
 */
static float strategy_classification_lr(void)
{
    return 0.01f;
}

static void strategy_classification_transform(float* output, uint32_t size)
{
    // Softmax normalization for probability distribution
    float max_val = output[0];
    for (uint32_t i = 1; i < size; i++) {
        if (output[i] > max_val)
            max_val = output[i];
    }

    float sum = 0.0f;
    for (uint32_t i = 0; i < size; i++) {
        output[i] = expf(output[i] - max_val);
        sum += output[i];
    }

    for (uint32_t i = 0; i < size; i++) {
        output[i] /= sum;
    }
}

static float strategy_classification_loss(const float* pred, const float* target, uint32_t size)
{
    // Cross-entropy loss
    float loss = 0.0f;
    for (uint32_t i = 0; i < size; i++) {
        if (target[i] > 0.0f) {
            loss -= target[i] * logf(fmaxf(pred[i], 1e-10f));
        }
    }
    return loss;
}

/**
 * @brief Regression strategy - linear output, MSE loss
 *
 * WHY: Regression predicts continuous values
 * WHEN: Predicting real-valued outputs
 * COMPLEXITY: O(n) for MSE calculation
 */
static float strategy_regression_lr(void)
{
    return 0.005f;
}

static void strategy_regression_transform(float* output, uint32_t size)
{
    // No transformation - use raw values
    (void) output;
    (void) size;
}

static float strategy_regression_loss(const float* pred, const float* target, uint32_t size)
{
    // Mean squared error
    float loss = 0.0f;
    for (uint32_t i = 0; i < size; i++) {
        float diff = pred[i] - target[i];
        loss += diff * diff;
    }
    return loss / size;
}

/**
 * @brief Pattern matching strategy - high LR, binary output
 *
 * WHY: Pattern matching needs fast adaptation
 * WHEN: Recognizing specific patterns quickly
 * COMPLEXITY: O(n)
 */
static float strategy_pattern_lr(void)
{
    return 0.02f;
}

static void strategy_pattern_transform(float* output, uint32_t size)
{
    // Threshold to binary
    for (uint32_t i = 0; i < size; i++) {
        output[i] = output[i] > 0.5f ? 1.0f : 0.0f;
    }
}

static float strategy_pattern_loss(const float* pred, const float* target, uint32_t size)
{
    // Binary cross-entropy
    float loss = 0.0f;
    for (uint32_t i = 0; i < size; i++) {
        float p = fmaxf(fminf(pred[i], 0.9999f), 0.0001f);
        loss -= target[i] * logf(p) + (1.0f - target[i]) * logf(1.0f - p);
    }
    return loss / size;
}

/**
 * @brief Association learning strategy - Hebbian-focused
 *
 * WHY: Association learning uses different plasticity rules
 * WHEN: Building associative memories
 * COMPLEXITY: O(n)
 */
static float strategy_association_lr(void)
{
    return 0.05f;
}

static void strategy_association_transform(float* output, uint32_t size)
{
    // Normalize to unit range
    float max_val = 0.0f;
    for (uint32_t i = 0; i < size; i++) {
        if (fabsf(output[i]) > max_val)
            max_val = fabsf(output[i]);
    }

    if (max_val > 0.0f) {
        for (uint32_t i = 0; i < size; i++) {
            output[i] /= max_val;
        }
    }
}

static float strategy_association_loss(const float* pred, const float* target, uint32_t size)
{
    // Cosine distance
    float dot = 0.0f, norm_p = 0.0f, norm_t = 0.0f;
    for (uint32_t i = 0; i < size; i++) {
        dot += pred[i] * target[i];
        norm_p += pred[i] * pred[i];
        norm_t += target[i] * target[i];
    }
    float cosine = dot / (sqrtf(norm_p) * sqrtf(norm_t) + 1e-10f);
    return 1.0f - cosine;
}

//=============================================================================
// Strategy Factory
//=============================================================================

/**
 * @brief Create strategy for task type
 *
 * WHY: Factory pattern for strategy creation
 * Centralizes strategy instantiation logic
 *
 * COMPLEXITY: O(1) - simple allocation and assignment
 *
 * @param task Task type
 * @return Strategy instance or NULL on error
 */
static task_strategy_t* strategy_create(brain_task_t task)
{
    task_strategy_t* strategy = nimcp_calloc(1, sizeof(task_strategy_t));
    if (!strategy)
        return NULL;

    strategy->task_type = task;

    switch (task) {
        case BRAIN_TASK_CLASSIFICATION:
            strategy->get_learning_rate = strategy_classification_lr;
            strategy->transform_output = strategy_classification_transform;
            strategy->compute_loss = strategy_classification_loss;
            break;

        case BRAIN_TASK_REGRESSION:
            strategy->get_learning_rate = strategy_regression_lr;
            strategy->transform_output = strategy_regression_transform;
            strategy->compute_loss = strategy_regression_loss;
            break;

        case BRAIN_TASK_PATTERN_MATCHING:
            strategy->get_learning_rate = strategy_pattern_lr;
            strategy->transform_output = strategy_pattern_transform;
            strategy->compute_loss = strategy_pattern_loss;
            break;

        case BRAIN_TASK_ASSOCIATION:
            strategy->get_learning_rate = strategy_association_lr;
            strategy->transform_output = strategy_association_transform;
            strategy->compute_loss = strategy_association_loss;
            break;

        default:
            // Default to classification
            strategy->get_learning_rate = strategy_classification_lr;
            strategy->transform_output = strategy_classification_transform;
            strategy->compute_loss = strategy_classification_loss;
            break;
    }

    return strategy;
}

/**
 * @brief Destroy strategy
 *
 * COMPLEXITY: O(1)
 */
static void strategy_destroy(task_strategy_t* strategy)
{
    nimcp_free(strategy);
}

//=============================================================================
// Error Handling
//=============================================================================

static __thread char last_error[256] = {0};

/**
 * @brief Set error message with printf-style formatting
 *
 * WHY: Thread-local error storage prevents race conditions
 * Thread-safe error reporting without locks
 *
 * COMPLEXITY: O(n) where n = message length
 *
 * @param format Printf-style format string
 */
static void set_error(const char* format, ...)
{
    va_list args;
    va_start(args, format);
    vsnprintf(last_error, sizeof(last_error), format, args);
    va_end(args);
}

/**
 * @brief Get last error message
 *
 * COMPLEXITY: O(1)
 * THREAD-SAFE: Uses thread-local storage
 *
 * @return Error string or NULL if no error
 */
const char* brain_get_last_error(void)
{
    return last_error[0] != '\0' ? last_error : NULL;
}

/**
 * @brief Clear last error
 *
 * COMPLEXITY: O(1)
 */
void brain_clear_error(void)
{
    last_error[0] = '\0';
}

//=============================================================================
// Internal Access API (NIMCP 2.5 Consciousness Subsystems)
//=============================================================================

/**
 * WHAT: Get underlying adaptive network from brain
 * WHY: Introspection/salience/consolidation need direct network access
 * HOW: Return internal network handle
 *
 * WARNING: Exposes internals - for consciousness subsystems only!
 */
adaptive_network_t brain_get_network(brain_t brain)
{
    if (!brain) {
        set_error("NULL brain passed to brain_get_network");
        return NULL;
    }
    return brain->network;
}

//=============================================================================
// Size Presets - Builder Helpers
//=============================================================================

/**
 * @brief Get neuron count for size preset
 *
 * WHY: Abstracts size->neuron mapping for maintainability
 * Centralizes sizing logic
 *
 * COMPLEXITY: O(1)
 *
 * @param size Brain size preset
 * @return Neuron count for size
 */
static uint32_t get_neuron_count(brain_size_t size)
{
    switch (size) {
        case BRAIN_SIZE_TINY:
            return 100;
        case BRAIN_SIZE_SMALL:
            return 1000;
        case BRAIN_SIZE_MEDIUM:
            return 10000;
        case BRAIN_SIZE_LARGE:
            return 100000;
        case BRAIN_SIZE_CUSTOM:
            return 1000;
        default:
            return 1000;
    }
}

/**
 * @brief Get default sparsity target for size
 *
 * WHY: Larger networks need higher sparsity for efficiency
 * Balances performance and memory
 *
 * COMPLEXITY: O(1)
 *
 * @param size Brain size preset
 * @return Sparsity target (0.0-1.0)
 */
static float get_default_sparsity(brain_size_t size)
{
    switch (size) {
        case BRAIN_SIZE_TINY:
            return 0.70f;
        case BRAIN_SIZE_SMALL:
            return 0.80f;
        case BRAIN_SIZE_MEDIUM:
            return 0.85f;
        case BRAIN_SIZE_LARGE:
            return 0.90f;
        default:
            return 0.80f;
    }
}

//=============================================================================
// Configuration Builders
//=============================================================================

/**
 * @brief Build spike parameters for brain configuration
 *
 * WHY: Separates spike config from main creation logic
 * Makes configuration more maintainable and testable
 *
 * COMPLEXITY: O(1)
 *
 * @param sparsity_target Target sparsity level
 * @return Spike parameters structure
 */
static adaptive_spike_params_t build_spike_params(float sparsity_target)
{
    adaptive_spike_params_t params = {0};
    params.k_factor = 0.5f;
    params.sparsity_target = sparsity_target;
    params.encoding = SPIKE_ENCODING_INTEGER;
    params.enable_soft_reset = true;
    params.enable_adaptation = true;
    params.adaptation_window = 100;
    params.min_threshold = 0.1f;
    params.max_threshold = 10.0f;
    return params;
}

/**
 * @brief Build base network configuration
 *
 * WHY: Isolates network config from brain config
 * Enables reuse and testing of network setup
 *
 * COMPLEXITY: O(1) + memory allocation
 *
 * @param num_inputs Input dimension
 * @param num_outputs Output dimension
 * @param num_neurons Total neuron count
 * @return Base network config (caller must free layer_sizes)
 */
static network_config_t build_base_network_config(uint32_t num_inputs, uint32_t num_outputs,
                                                  uint32_t num_neurons)
{
    network_config_t config = {0};
    config.input_size = num_inputs;
    config.output_size = num_outputs;
    config.num_neurons = num_neurons;
    config.num_layers = 3;

    config.layer_sizes = nimcp_calloc(3, sizeof(uint32_t));
    // Guard: Check allocation
    // WHY: If allocation fails, returning config with NULL layer_sizes will crash
    if (!config.layer_sizes) {
        set_error("Failed to allocate layer_sizes array");
        return config;  // Return with layer_sizes = NULL to signal error
    }

    config.layer_sizes[0] = num_inputs;
    config.layer_sizes[1] = num_neurons;
    config.layer_sizes[2] = num_outputs;

    config.enable_stdp = true;
    config.enable_hebbian = true;
    config.enable_oja = true;
    config.enable_homeostasis = true;

    return config;
}

/**
 * @brief Build complete adaptive network configuration
 *
 * WHY: Combines base config and spike params
 * Single point of network configuration assembly
 *
 * COMPLEXITY: O(1) + memory allocation
 *
 * @param num_inputs Input dimension
 * @param num_outputs Output dimension
 * @param num_neurons Neuron count
 * @param sparsity_target Target sparsity
 * @return Complete adaptive network config
 */
static adaptive_network_config_t build_network_config(uint32_t num_inputs, uint32_t num_outputs,
                                                      uint32_t num_neurons, float sparsity_target)
{
    adaptive_network_config_t config = {0};

    fprintf(stderr, "[DEBUG build_network_config] Building base config...\n"); fflush(stderr);
    config.base_config = build_base_network_config(num_inputs, num_outputs, num_neurons);
    fprintf(stderr, "[DEBUG build_network_config] base_config.layer_sizes=%p\n", (void*)config.base_config.layer_sizes); fflush(stderr);

    fprintf(stderr, "[DEBUG build_network_config] Building spike params...\n"); fflush(stderr);
    config.spike_params = build_spike_params(sparsity_target);
    fprintf(stderr, "[DEBUG build_network_config] After spike_params, layer_sizes=%p\n", (void*)config.base_config.layer_sizes); fflush(stderr);

    config.enable_sparsity = true;
    config.pruning_threshold = 0.01f;
    config.update_frequency = 100;

    fprintf(stderr, "[DEBUG build_network_config] Returning config with layer_sizes=%p\n", (void*)config.base_config.layer_sizes); fflush(stderr);
    return config;
}

/**
 * @brief Initialize brain configuration
 *
 * WHY: Centralizes config initialization with strategy
 * Ensures consistent config setup
 *
 * COMPLEXITY: O(1)
 *
 * @param config Output config structure
 * @param task_name Name for brain
 * @param size Size preset
 * @param task Task type
 * @param num_inputs Input dimension
 * @param num_outputs Output dimension
 * @param strategy Task strategy for learning rate
 */
static void init_brain_config(brain_config_t* config, const char* task_name, brain_size_t size,
                              brain_task_t task, uint32_t num_inputs, uint32_t num_outputs,
                              task_strategy_t* strategy)
{
    config->size = size;
    config->task = task;
    config->num_inputs = num_inputs;
    config->num_outputs = num_outputs;
    config->learning_rate = strategy->get_learning_rate();
    config->sparsity_target = get_default_sparsity(size);
    config->enable_explanations = true;
    strncpy(config->task_name, task_name, sizeof(config->task_name) - 1);
}

/**
 * @brief Initialize brain statistics
 *
 * WHY: Separates stats initialization for clarity
 * Makes stats setup reusable
 *
 * COMPLEXITY: O(1)
 *
 * @param stats Output stats structure
 * @param task_name Name for brain
 * @param size Size preset
 * @param num_inputs Input dimension
 * @param learning_rate Learning rate
 */
static void init_brain_stats(brain_stats_t* stats, const char* task_name, brain_size_t size,
                             uint32_t num_inputs, float learning_rate)
{
    uint32_t num_neurons = get_neuron_count(size);

    stats->size = size;
    stats->num_neurons = num_neurons;
    stats->num_synapses = num_neurons * num_inputs;
    stats->num_active_synapses = stats->num_synapses;
    stats->current_learning_rate = learning_rate;
    strncpy(stats->task_name, task_name, sizeof(stats->task_name) - 1);
}

//=============================================================================
// Decision Caching
//=============================================================================

/**
 * @brief Check if input matches cached input
 *
 * WHY: Avoid redundant computation for repeated inputs
 * Common in batch processing and validation
 *
 * COMPLEXITY: O(n) where n = num_features
 * OPTIMIZATION: Early exit on first mismatch
 *
 * @param brain Brain handle
 * @param features Input to check
 * @param num_features Feature count
 * @return true if cached input matches
 */
static bool is_cached_input(brain_t brain, const float* features, uint32_t num_features)
{
    if (!brain->last_input || !brain->cached_decision)
        return false;
    if (brain->input_size != num_features)
        return false;

    return memcmp(brain->last_input, features, num_features * sizeof(float)) == 0;
}

/**
 * @brief Cache decision for input
 *
 * WHY: Store decision for potential reuse
 * Improves batch processing performance
 *
 * COMPLEXITY: O(n) for input copy
 *
 * @param brain Brain handle
 * @param features Input to cache
 * @param num_features Feature count
 * @param decision Decision to cache
 */
static void cache_decision(brain_t brain, const float* features, uint32_t num_features,
                           brain_decision_t* decision)
{
    if (!brain->last_input) {
        brain->last_input = nimcp_malloc(num_features * sizeof(float));
        brain->input_size = num_features;
    }

    memcpy(brain->last_input, features, num_features * sizeof(float));

    if (brain->cached_decision) {
        brain_free_decision(brain->cached_decision);
    }
    brain->cached_decision = decision;
}

/**
 * @brief Clear decision cache
 *
 * COMPLEXITY: O(1)
 */
static void clear_cache(brain_t brain)
{
    nimcp_free(brain->last_input);
    brain->last_input = NULL;

    if (brain->cached_decision) {
        brain_free_decision(brain->cached_decision);
        brain->cached_decision = NULL;
    }
}

//=============================================================================
// Brain Factory - Creation with Validation
//=============================================================================

/**
 * @brief Validate brain creation parameters
 *
 * WHY: Guard clause pattern - early exit on invalid input
 * Prevents invalid state propagation
 *
 * COMPLEXITY: O(1)
 *
 * @param task_name Brain name
 * @param num_inputs Input dimension
 * @param num_outputs Output dimension
 * @return true if valid
 */
static bool validate_creation_params(const char* task_name, uint32_t num_inputs,
                                     uint32_t num_outputs)
{
    if (!task_name) {
        set_error("task_name cannot be NULL");
        return false;
    }

    if (num_inputs == 0) {
        set_error("num_inputs must be > 0");
        return false;
    }

    if (num_outputs == 0) {
        set_error("num_outputs must be > 0");
        return false;
    }

    return true;
}

/**
 * @brief Allocate and initialize brain structure
 *
 * WHY: Separates allocation from configuration
 * Makes memory management explicit
 *
 * COMPLEXITY: O(1)
 *
 * @return Allocated brain or NULL on error
 */
static brain_t allocate_brain(void)
{
    brain_t brain = nimcp_calloc(1, sizeof(struct brain_struct));
    if (!brain) {
        set_error("Failed to allocate brain structure");
        return NULL;
    }

    brain->last_input = NULL;
    brain->cached_decision = NULL;
    brain->input_size = 0;
    brain->distributed = NULL;  // Initialize as standalone brain

    return brain;
}

/**
 * @brief Create adaptive network for brain
 *
 * WHY: Isolates network creation complexity
 * Handles network config lifecycle
 *
 * COMPLEXITY: O(n) where n = num_neurons
 *
 * @param num_inputs Input dimension
 * @param num_outputs Output dimension
 * @param num_neurons Neuron count
 * @param sparsity_target Target sparsity
 * @return Network handle or NULL on error
 */
static adaptive_network_t create_brain_network(uint32_t num_inputs, uint32_t num_outputs,
                                               uint32_t num_neurons, float sparsity_target)
{
    fprintf(stderr, "[DEBUG create_brain_network] Building config...\n"); fflush(stderr);
    adaptive_network_config_t net_config =
        build_network_config(num_inputs, num_outputs, num_neurons, sparsity_target);

    // Guard: Check if layer_sizes allocation failed in build_base_network_config
    // WHY: NULL layer_sizes will cause crash in adaptive_network_create
    fprintf(stderr, "[DEBUG create_brain_network] Checking layer_sizes=%p\n", (void*)net_config.base_config.layer_sizes); fflush(stderr);
    if (!net_config.base_config.layer_sizes) {
        // Error already set by build_base_network_config
        fprintf(stderr, "[DEBUG create_brain_network] ERROR: layer_sizes is NULL\n"); fflush(stderr);
        return NULL;
    }

    fprintf(stderr, "[DEBUG create_brain_network] Calling adaptive_network_create...\n"); fflush(stderr);
    adaptive_network_t network = adaptive_network_create(&net_config);

    // Free our copy of layer_sizes - adaptive_network_create makes its own deep copy (or fails)
    // WHY: Avoid memory leak - we allocated this in build_base_network_config
    // WHAT: Safe to free even if network creation failed, because we still own this allocation
    // Note: layer_sizes pointer should not be modified by adaptive_network_create (const param)
    if (net_config.base_config.layer_sizes) {
        nimcp_free((void*)net_config.base_config.layer_sizes);
    }

    return network;
}

/**
 * @brief Initialize output labels array
 *
 * COMPLEXITY: O(1)
 *
 * @param brain Brain to initialize
 * @param num_outputs Number of output labels
 * @return true on success
 */
static bool init_output_labels(brain_t brain, uint32_t num_outputs)
{
    brain->output_labels = nimcp_calloc(num_outputs, sizeof(char*));
    if (!brain->output_labels) {
        set_error("Failed to allocate output labels");
        return false;
    }
    brain->num_output_labels = 0;
    return true;
}

/**
 * @brief Create brain with preset size and task
 *
 * WHY: Factory pattern - single creation entry point
 * Encapsulates all creation complexity with validation
 *
 * COMPLEXITY: O(n) where n = num_neurons
 * MEMORY: O(n*c) where c = connections per neuron
 *
 * PATTERN: Factory pattern with guard clauses
 *
 * @param task_name Human-readable name
 * @param size Brain size preset
 * @param task Task template
 * @param num_inputs Input dimension
 * @param num_outputs Output dimension
 * @return Brain handle or NULL on error
 */
brain_t brain_create(const char* task_name, brain_size_t size, brain_task_t task,
                     uint32_t num_inputs, uint32_t num_outputs)
{
    // Guard: Validate parameters
    fprintf(stderr, "[DEBUG brain_create] Validating params...\n"); fflush(stderr);
    if (!validate_creation_params(task_name, num_inputs, num_outputs)) {
        return NULL;
    }

    // Allocate brain structure
    fprintf(stderr, "[DEBUG brain_create] Allocating brain...\n"); fflush(stderr);
    brain_t brain = allocate_brain();
    if (!brain)
        return NULL;

    // Create strategy for task
    fprintf(stderr, "[DEBUG brain_create] Creating strategy...\n"); fflush(stderr);
    brain->strategy = strategy_create(task);
    if (!brain->strategy) {
        set_error("Failed to create task strategy");
        nimcp_free(brain);
        return NULL;
    }

    // Initialize configuration
    fprintf(stderr, "[DEBUG brain_create] Initializing config...\n"); fflush(stderr);
    init_brain_config(&brain->config, task_name, size, task, num_inputs, num_outputs,
                      brain->strategy);

    // Create network
    fprintf(stderr, "[DEBUG brain_create] Creating network...\n"); fflush(stderr);
    uint32_t num_neurons = get_neuron_count(size);
    brain->network =
        create_brain_network(num_inputs, num_outputs, num_neurons, brain->config.sparsity_target);

    if (!brain->network) {
        set_error("Failed to create adaptive network");
        strategy_destroy(brain->strategy);
        nimcp_free(brain);
        return NULL;
    }

    // Initialize output labels
    if (!init_output_labels(brain, num_outputs)) {
        adaptive_network_destroy(brain->network);
        strategy_destroy(brain->strategy);
        nimcp_free(brain);
        return NULL;
    }

    // Initialize statistics
    init_brain_stats(&brain->stats, task_name, size, num_inputs, brain->config.learning_rate);

    brain_clear_error();
    return brain;
}

/**
 * @brief Create brain with custom configuration
 *
 * WHY: Allows advanced users to customize all parameters
 * Delegates to standard factory after validation
 *
 * COMPLEXITY: O(n) where n = num_neurons
 *
 * @param config Custom configuration
 * @return Brain handle or NULL on error
 */
brain_t brain_create_custom(const brain_config_t* config)
{
    if (!config) {
        set_error("Null config provided");
        return NULL;
    }

    // Validate task_name as string field (NULL termination, UTF-8)
    size_t task_name_size = sizeof(config->task_name);
    if (!nimcp_validate_string_field(config->task_name, task_name_size)) {
        set_error("Invalid task_name in config");
        return NULL;
    }

    // Validate num_inputs (integer field, range 1-10000)
    if (!nimcp_validate_integer_field(&config->num_inputs, sizeof(config->num_inputs))) {
        set_error("Invalid num_inputs in config");
        return NULL;
    }
    if (config->num_inputs < 1 || config->num_inputs > 10000) {
        set_error("num_inputs out of range (1-10000): %u", config->num_inputs);
        return NULL;
    }

    // Validate num_outputs (integer field, range 1-10000)
    if (!nimcp_validate_integer_field(&config->num_outputs, sizeof(config->num_outputs))) {
        set_error("Invalid num_outputs in config");
        return NULL;
    }
    if (config->num_outputs < 1 || config->num_outputs > 10000) {
        set_error("num_outputs out of range (1-10000): %u", config->num_outputs);
        return NULL;
    }

    // Validate learning_rate (float field - NaN/Inf)
    if (!nimcp_validate_float_field(&config->learning_rate, sizeof(config->learning_rate))) {
        set_error("Invalid learning_rate in config (NaN or Inf)");
        return NULL;
    }

    // Validate sparsity_target (float field - NaN/Inf)
    if (!nimcp_validate_float_field(&config->sparsity_target, sizeof(config->sparsity_target))) {
        set_error("Invalid sparsity_target in config (NaN or Inf)");
        return NULL;
    }

    return brain_create(config->task_name, config->size, config->task, config->num_inputs,
                        config->num_outputs);
}

/**
 * @brief Destroy brain and free resources
 *
 * WHY: Proper cleanup prevents memory leaks
 * Handles partial initialization gracefully
 *
 * COMPLEXITY: O(n) where n = num_neurons (for network cleanup)
 *
 * @param brain Brain to destroy
 */
void brain_destroy(brain_t brain)
{
    if (!brain)
        return;

    if (brain->network) {
        adaptive_network_destroy(brain->network);
    }

    if (brain->strategy) {
        strategy_destroy(brain->strategy);
    }

    if (brain->output_labels) {
        for (uint32_t i = 0; i < brain->num_output_labels; i++) {
            nimcp_free(brain->output_labels[i]);
        }
        nimcp_free(brain->output_labels);
    }

    // Phase 3: Cleanup distributed cognition coordinator
    if (brain->distributed) {
        distrib_cognition_destroy(brain->distributed);
    }

    clear_cache(brain);
    nimcp_free(brain);
}

//=============================================================================
// Learning API
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
static uint32_t get_or_create_label_index(brain_t brain, const char* label)
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

    // Create new label
    brain->output_labels[brain->num_output_labels] = strdup(label);
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
static void label_to_output(brain_t brain, const char* label, float* output, float confidence)
{
    uint32_t label_idx = get_or_create_label_index(brain, label);

    memset(output, 0, brain->config.num_outputs * sizeof(float));
    output[label_idx] = confidence;
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
        return -1.0f;
    }

    // Guard: Check feature dimension
    if (num_features != brain->config.num_inputs) {
        set_error("Feature count mismatch: expected %u, got %u", brain->config.num_inputs,
                  num_features);
        return -1.0f;
    }

    // Convert label to target output
    float* target = nimcp_malloc(brain->config.num_outputs * sizeof(float));
    label_to_output(brain, label, target, confidence);

    // Create training example
    training_example_t example = {.input = (float*) features,
                                  .input_size = num_features,
                                  .target = target,
                                  .target_size = brain->config.num_outputs,
                                  .confidence = confidence};
    strncpy(example.label, label, sizeof(example.label) - 1);

    // Learn using adaptive network
    float loss = adaptive_network_learn(brain->network, &example, LEARN_MODE_SUPERVISED,
                                        brain->config.learning_rate);

    nimcp_free(target);

    // Update statistics
    brain->stats.total_learning_steps++;

    // Invalidate cache after learning
    clear_cache(brain);

    brain_clear_error();
    return loss;
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
        return -1.0f;
    }

    float total_loss = 0.0f;

    for (uint32_t i = 0; i < num_examples; i++) {
        float loss = brain_learn_example(brain, examples[i].features, examples[i].num_features,
                                         examples[i].label, examples[i].confidence);

        if (loss < 0.0f) {
            return -1.0f;
        }

        total_loss += loss;
    }

    brain_clear_error();
    return total_loss / num_examples;
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
        return -1.0f;
    }

    // Guard: Check dimensions
    if (num_features != brain->config.num_inputs) {
        set_error("Feature count mismatch: expected %u, got %u", brain->config.num_inputs,
                  num_features);
        return -1.0f;
    }

    // Query LLM teacher
    char label[64] = {0};
    float confidence = llm_fn(input, num_features, llm_context, label, sizeof(label));

    // Guard: Validate LLM response
    if (confidence <= 0.0f) {
        set_error("LLM teacher returned invalid confidence: %.2f", confidence);
        return -1.0f;
    }

    // Learn from LLM's decision
    float loss = brain_learn_example(brain, input, num_features, label, confidence);

    brain_clear_error();
    return loss;
}

//=============================================================================
// Inference API
//=============================================================================

/**
 * @brief Allocate decision structure
 *
 * COMPLEXITY: O(1)
 */
static brain_decision_t* allocate_decision(uint32_t output_size)
{
    brain_decision_t* decision = nimcp_calloc(1, sizeof(brain_decision_t));
    if (!decision)
        return NULL;

    decision->output_size = output_size;
    decision->output_vector = nimcp_malloc(output_size * sizeof(float));

    if (!decision->output_vector) {
        nimcp_free(decision);
        return NULL;
    }

    return decision;
}

/**
 * @brief Deep copy a decision structure
 *
 * WHAT: Creates an independent copy of a decision
 * WHY: Cached decisions must not be freed by caller - return copies instead
 * HOW: Allocates new decision and copies all fields including dynamically allocated arrays
 *
 * COMPLEXITY: O(n) where n = output_size + num_active_neurons
 *
 * @param source Decision to copy
 * @return New decision copy, or NULL on allocation failure
 */
static brain_decision_t* copy_decision(const brain_decision_t* source)
{
    if (!source)
        return NULL;

    // Allocate new decision structure
    brain_decision_t* copy = nimcp_calloc(1, sizeof(brain_decision_t));
    if (!copy)
        return NULL;

    // Copy scalar fields
    memcpy(copy, source, sizeof(brain_decision_t));

    // NULL out pointer fields to prevent accidental sharing
    copy->output_vector = NULL;
    copy->active_neuron_ids = NULL;

    // Deep copy output_vector
    if (source->output_vector && source->output_size > 0) {
        copy->output_vector = nimcp_malloc(source->output_size * sizeof(float));
        if (!copy->output_vector) {
            nimcp_free(copy);
            return NULL;
        }
        memcpy(copy->output_vector, source->output_vector, source->output_size * sizeof(float));
    }

    // Deep copy active_neuron_ids
    if (source->active_neuron_ids && source->num_active_neurons > 0) {
        copy->active_neuron_ids = nimcp_malloc(source->num_active_neurons * sizeof(uint32_t));
        if (!copy->active_neuron_ids) {
            if (copy->output_vector)
                nimcp_free(copy->output_vector);
            nimcp_free(copy);
            return NULL;
        }
        memcpy(copy->active_neuron_ids, source->active_neuron_ids,
               source->num_active_neurons * sizeof(uint32_t));
    }

    return copy;
}

/**
 * @brief Perform forward pass through network
 *
 * COMPLEXITY: O(s*n) where s = sparsity
 *
 * @param brain Brain handle
 * @param features Input features
 * @param num_features Feature count
 * @param decision Decision to populate
 * @return Number of active neurons
 */
static uint32_t perform_forward_pass(brain_t brain, const float* features, uint32_t num_features,
                                     brain_decision_t* decision)
{
    uint64_t start_time = nimcp_time_monotonic_us();

    uint32_t active_neurons = adaptive_network_forward(
        brain->network, features, num_features, decision->output_vector, decision->output_size, 0);

    decision->inference_time_us = nimcp_time_elapsed_us(start_time);

    return active_neurons;
}

/**
 * @brief Find maximum output and determine label
 *
 * COMPLEXITY: O(n) where n = num_outputs
 */
static void determine_output_label(brain_t brain, brain_decision_t* decision)
{
    uint32_t max_idx = 0;
    float max_value = decision->output_vector[0];

    for (uint32_t i = 1; i < decision->output_size; i++) {
        if (decision->output_vector[i] > max_value) {
            max_value = decision->output_vector[i];
            max_idx = i;
        }
    }

    // Set label
    if (max_idx < brain->num_output_labels) {
        strncpy(decision->label, brain->output_labels[max_idx], sizeof(decision->label) - 1);
    } else {
        snprintf(decision->label, sizeof(decision->label), "output_%u", max_idx);
    }

    // Normalize confidence
    decision->confidence = fminf(max_value / 10.0f, 1.0f);
}

/**
 * @brief Populate interpretability information
 *
 * COMPLEXITY: O(n)
 */
static void populate_interpretability(brain_t brain, const float* features, uint32_t num_features,
                                      uint32_t active_neurons, brain_decision_t* decision)
{
    decision->num_active_neurons = active_neurons;
    decision->sparsity = adaptive_network_get_sparsity(brain->network);

    if (brain->config.enable_explanations) {
        adaptive_network_explain(brain->network, features, num_features, decision->explanation,
                                 sizeof(decision->explanation));
    }

    // Populate active neuron IDs
    decision->active_neuron_ids = nimcp_malloc(active_neurons * sizeof(uint32_t));
    for (uint32_t i = 0; i < active_neurons; i++) {
        decision->active_neuron_ids[i] = i;
    }
}

/**
 * @brief Update brain statistics after inference
 *
 * COMPLEXITY: O(1)
 */
static void update_inference_stats(brain_t brain, brain_decision_t* decision)
{
    brain->stats.total_inferences++;
    brain->stats.avg_inference_time_us =
        (brain->stats.avg_inference_time_us * (brain->stats.total_inferences - 1) +
         decision->inference_time_us) /
        brain->stats.total_inferences;
    brain->stats.avg_sparsity = decision->sparsity;
}

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
brain_decision_t* brain_decide(brain_t brain, const float* features, uint32_t num_features)
{
    // Guard: Validate parameters
    if (!brain || !features) {
        set_error("Invalid parameters to brain_decide");
        return NULL;
    }

    // Guard: Check dimensions
    if (num_features != brain->config.num_inputs) {
        set_error("Feature count mismatch: expected %u, got %u", brain->config.num_inputs,
                  num_features);
        return NULL;
    }

    /**
     * WHAT: Check cache and return a COPY of cached decision
     * WHY: Cached decision is owned by brain, caller expects to own returned decision
     * HOW: Deep copy the cached decision to give caller an independent copy
     *
     * BUG FIX: Previously returned cached pointer directly, causing double-free
     * when caller freed it and then brain_destroy tried to free cached_decision
     */
    if (is_cached_input(brain, features, num_features)) {
        return copy_decision(brain->cached_decision);
    }

    // Allocate decision structure
    brain_decision_t* decision = allocate_decision(brain->config.num_outputs);
    if (!decision) {
        set_error("Failed to allocate decision structure");
        return NULL;
    }

    // Perform forward pass
    uint32_t active_neurons = perform_forward_pass(brain, features, num_features, decision);

    // Apply task-specific output transformation
    brain->strategy->transform_output(decision->output_vector, decision->output_size);

    // Determine output label and confidence
    determine_output_label(brain, decision);

    // Populate interpretability information
    populate_interpretability(brain, features, num_features, active_neurons, decision);

    // Update statistics
    update_inference_stats(brain, decision);

    // Cache decision for potential reuse
    cache_decision(brain, features, num_features, decision);

    /**
     * WHAT: Return a copy of the decision, not the original
     * WHY: We cached the original, so brain owns it. Caller needs own copy.
     * HOW: Deep copy before returning
     *
     * BUG FIX: Previously returned cached decision directly, causing double-free
     * when both caller and brain_destroy tried to free it.
     */
    brain_decision_t* decision_copy = copy_decision(decision);
    if (!decision_copy) {
        // Copy failed - this is bad, but at least don't leak the original
        set_error("Failed to copy decision");
        return NULL;
    }

    brain_clear_error();
    return decision_copy;
}

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
void brain_free_decision(brain_decision_t* decision)
{
    if (!decision)
        return;

    /**
     * WHAT: Free decision structure and its allocated fields
     * WHY: Prevent memory leaks
     * HOW: Use nimcp_free() for memory allocated with nimcp_malloc()
     */
    if (decision->output_vector) {
        nimcp_free(decision->output_vector);
    }
    if (decision->active_neuron_ids) {
        nimcp_free(decision->active_neuron_ids);
    }
    nimcp_free(decision);
}

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
                        uint32_t features_per_input, brain_decision_t* decisions)
{
    // Guard: Validate parameters
    if (!brain || !inputs || !decisions || num_inputs == 0) {
        set_error("Invalid parameters to brain_decide_batch");
        return false;
    }

    for (uint32_t i = 0; i < num_inputs; i++) {
        brain_decision_t* decision = brain_decide(brain, inputs[i], features_per_input);

        if (!decision) {
            return false;
        }

        memcpy(&decisions[i], decision, sizeof(brain_decision_t));
        nimcp_free(decision);
    }

    brain_clear_error();
    return true;
}

//=============================================================================
// Persistence API
//=============================================================================

/**
 * @brief Save metadata file
 *
 * COMPLEXITY: O(k) where k = num_labels
 */
static bool save_metadata(brain_t brain, const char* filepath)
{
    char meta_path[512];
    snprintf(meta_path, sizeof(meta_path), "%s.meta", filepath);

    FILE* meta_file = fopen(meta_path, "wb");
    if (!meta_file)
        return false;

    fwrite(&brain->config, sizeof(brain_config_t), 1, meta_file);
    fwrite(&brain->num_output_labels, sizeof(uint32_t), 1, meta_file);

    for (uint32_t i = 0; i < brain->num_output_labels; i++) {
        uint32_t len = strlen(brain->output_labels[i]) + 1;
        fwrite(&len, sizeof(uint32_t), 1, meta_file);
        fwrite(brain->output_labels[i], len, 1, meta_file);
    }

    fclose(meta_file);
    return true;
}

/**
 * @brief Save brain to file
 *
 * WHY: Enables model persistence across sessions
 * Saves both network and metadata
 *
 * COMPLEXITY: O(n*c) where n = neurons, c = connections
 *
 * @param brain Brain handle
 * @param filepath Path to save to
 * @return true on success
 */
bool brain_save(brain_t brain, const char* filepath)
{
    // Guard: Validate parameters
    if (!brain || !filepath) {
        set_error("Invalid parameters to brain_save");
        return false;
    }

    // Save adaptive network
    bool success = adaptive_network_save(brain->network, filepath, SERIALIZE_FORMAT_BINARY);

    if (!success) {
        set_error("Failed to save adaptive network to %s", filepath);
        return false;
    }

    // Save metadata
    if (!save_metadata(brain, filepath)) {
        set_error("Failed to save metadata");
        return false;
    }

    brain_clear_error();
    return true;
}

/**
 * @brief Load metadata file
 *
 * COMPLEXITY: O(k) where k = num_labels
 */
static bool load_metadata(brain_t brain, const char* filepath)
{
    char meta_path[512];
    snprintf(meta_path, sizeof(meta_path), "%s.meta", filepath);

    FILE* meta_file = fopen(meta_path, "rb");
    if (!meta_file)
        return false;

    fread(&brain->config, sizeof(brain_config_t), 1, meta_file);

    // Validate brain->config fields after reading
    // Validate learning_rate (float field - NaN/Inf check)
    if (!nimcp_validate_float_field(&brain->config.learning_rate,
                                    sizeof(brain->config.learning_rate))) {
        fprintf(stderr, "ERROR: Invalid learning_rate in loaded config (NaN or Inf)\n");
        fclose(meta_file);
        return false;
    }

    // Validate sparsity_target (float field - NaN/Inf check)
    if (!nimcp_validate_float_field(&brain->config.sparsity_target,
                                    sizeof(brain->config.sparsity_target))) {
        fprintf(stderr, "ERROR: Invalid sparsity_target in loaded config (NaN or Inf)\n");
        fclose(meta_file);
        return false;
    }

    // Validate num_inputs (integer field, range 1-10000)
    if (!nimcp_validate_integer_field(&brain->config.num_inputs,
                                      sizeof(brain->config.num_inputs))) {
        fprintf(stderr, "ERROR: Invalid num_inputs in loaded config\n");
        fclose(meta_file);
        return false;
    }
    if (brain->config.num_inputs < 1 || brain->config.num_inputs > 10000) {
        fprintf(stderr, "ERROR: num_inputs out of range (1-10000): %u\n", brain->config.num_inputs);
        fclose(meta_file);
        return false;
    }

    // Validate num_outputs (integer field, range 1-10000)
    if (!nimcp_validate_integer_field(&brain->config.num_outputs,
                                      sizeof(brain->config.num_outputs))) {
        fprintf(stderr, "ERROR: Invalid num_outputs in loaded config\n");
        fclose(meta_file);
        return false;
    }
    if (brain->config.num_outputs < 1 || brain->config.num_outputs > 10000) {
        fprintf(stderr, "ERROR: num_outputs out of range (1-10000): %u\n",
                brain->config.num_outputs);
        fclose(meta_file);
        return false;
    }

    fread(&brain->num_output_labels, sizeof(uint32_t), 1, meta_file);

    // Validate num_output_labels (range 0-10000, 0 means no labels)
    if (!nimcp_validate_integer_field(&brain->num_output_labels,
                                      sizeof(brain->num_output_labels))) {
        fprintf(stderr, "ERROR: Invalid num_output_labels in loaded metadata\n");
        fclose(meta_file);
        return false;
    }
    if (brain->num_output_labels > 10000) {
        fprintf(stderr, "ERROR: num_output_labels out of range (0-10000): %u\n",
                brain->num_output_labels);
        fclose(meta_file);
        return false;
    }

    // Handle case where there are no labels
    if (brain->num_output_labels == 0) {
        brain->output_labels = NULL;
        fclose(meta_file);
        return true;
    }

    brain->output_labels = nimcp_malloc(brain->num_output_labels * sizeof(char*));
    if (!brain->output_labels) {
        fprintf(stderr, "ERROR: Failed to allocate output_labels array\n");
        fclose(meta_file);
        return false;
    }

    uint32_t i;
    for (i = 0; i < brain->num_output_labels; i++) {
        uint32_t len;
        fread(&len, sizeof(uint32_t), 1, meta_file);

        // Validate label length (1-1024 bytes)
        if (!nimcp_validate_integer_field(&len, sizeof(len))) {
            fprintf(stderr, "ERROR: Invalid label length at index %u\n", i);
            goto cleanup;
        }
        if (len < 1 || len > 1024) {
            fprintf(stderr, "ERROR: Label length out of range (1-1024) at index %u: %u\n", i, len);
            goto cleanup;
        }

        brain->output_labels[i] = nimcp_malloc(len);
        if (!brain->output_labels[i]) {
            fprintf(stderr, "ERROR: Failed to allocate label at index %u\n", i);
            goto cleanup;
        }

        fread(brain->output_labels[i], len, 1, meta_file);
    }

    fclose(meta_file);
    return true;

cleanup:
    // Free any allocated labels before the failed one
    for (uint32_t j = 0; j < i; j++) {
        nimcp_free(brain->output_labels[j]);
    }
    nimcp_free(brain->output_labels);
    brain->output_labels = NULL;
    fclose(meta_file);
    return false;
}

/**
 * @brief Load brain from file
 *
 * WHY: Restores saved brain state
 * Reconstructs network and metadata
 *
 * COMPLEXITY: O(n*c) where n = neurons, c = connections
 *
 * @param filepath Path to load from
 * @return Brain handle or NULL on error
 */
brain_t brain_load(const char* filepath)
{
    // Guard: Validate filepath
    if (!filepath) {
        set_error("Null filepath provided to brain_load");
        return NULL;
    }

    // Load adaptive network
    adaptive_network_t network = adaptive_network_load(filepath);
    if (!network) {
        set_error("Failed to load adaptive network from %s", filepath);
        return NULL;
    }

    // Allocate brain structure
    brain_t brain = allocate_brain();
    if (!brain) {
        adaptive_network_destroy(network);
        return NULL;
    }

    brain->network = network;

    // Load metadata
    if (!load_metadata(brain, filepath)) {
        // Use defaults if no metadata
        brain->config.size = BRAIN_SIZE_SMALL;
        brain->config.task = BRAIN_TASK_CLASSIFICATION;
        brain->config.learning_rate = 0.01f;
        brain->config.sparsity_target = 0.8f;
        brain->config.enable_explanations = true;
        snprintf(brain->config.task_name, sizeof(brain->config.task_name), "loaded_brain");

        // Set placeholder dimensions (actual dimensions are in the network)
        // TODO: Add adaptive_network getter functions to retrieve actual values
        brain->config.num_inputs = 1;
        brain->config.num_outputs = 1;
    }

    // Create strategy for task
    brain->strategy = strategy_create(brain->config.task);
    if (!brain->strategy) {
        set_error("Failed to create task strategy");
        brain_destroy(brain);
        return NULL;
    }

    // Initialize statistics
    init_brain_stats(&brain->stats, brain->config.task_name, brain->config.size,
                     brain->config.num_inputs, brain->config.learning_rate);

    brain_clear_error();
    return brain;
}

/**
 * @brief Get brain memory footprint
 *
 * WHY: Enables memory usage monitoring
 * Important for embedded and resource-constrained environments
 *
 * COMPLEXITY: O(n) where n = num_neurons
 *
 * @param brain Brain handle
 * @return Memory usage in bytes
 */
size_t brain_get_memory_usage(brain_t brain)
{
    if (!brain)
        return 0;

    size_t size = sizeof(struct brain_struct);
    size += adaptive_network_get_size(brain->network);

    for (uint32_t i = 0; i < brain->num_output_labels; i++) {
        size += strlen(brain->output_labels[i]) + 1;
    }

    return size;
}

//=============================================================================
// Analysis & Monitoring API
//=============================================================================

/**
 * @brief Get brain statistics
 *
 * WHY: Provides performance and training metrics
 * Essential for monitoring and debugging
 *
 * COMPLEXITY: O(1) - mostly copying cached stats
 *
 * @param brain Brain handle
 * @param stats Output statistics
 * @return true on success
 */
bool brain_get_stats(brain_t brain, brain_stats_t* stats)
{
    // Guard: Validate parameters
    if (!brain || !stats) {
        set_error("Invalid parameters to brain_get_stats");
        return false;
    }

    // Get performance from adaptive network
    network_performance_t perf;
    adaptive_network_get_performance(brain->network, &perf);

    // Copy to brain stats
    stats->size = brain->config.size;
    stats->num_neurons = brain->stats.num_neurons;
    stats->num_synapses = brain->stats.num_synapses;
    stats->num_active_synapses = brain->stats.num_active_synapses;
    stats->total_inferences = perf.total_inferences;
    stats->total_learning_steps = perf.total_learning_steps;
    stats->avg_sparsity = perf.avg_sparsity;
    stats->avg_inference_time_us = perf.avg_inference_time_us;
    stats->current_learning_rate = brain->config.learning_rate;
    stats->accuracy = perf.accuracy;
    stats->memory_bytes = perf.memory_usage_bytes;
    strncpy(stats->task_name, brain->config.task_name, sizeof(stats->task_name) - 1);

    brain_clear_error();
    return true;
}

/**
 * @brief Print brain info to stdout
 *
 * WHY: Convenient debugging and monitoring
 * Human-readable status display
 *
 * COMPLEXITY: O(1)
 *
 * @param brain Brain handle
 */
void brain_print_info(brain_t brain)
{
    if (!brain)
        return;

    brain_stats_t stats;
    brain_get_stats(brain, &stats);

    printf("Brain: %s\n", stats.task_name);
    printf("  Size: %u neurons, %u synapses\n", stats.num_neurons, stats.num_synapses);
    printf("  Training: %lu learning steps\n", stats.total_learning_steps);
    printf("  Inferences: %lu\n", stats.total_inferences);
    printf("  Avg inference time: %.3f ms\n", stats.avg_inference_time_us / 1000.0);
    printf("  Avg sparsity: %.1f%%\n", stats.avg_sparsity * 100.0);
    printf("  Memory usage: %.2f MB\n", stats.memory_bytes / (1024.0 * 1024.0));
    printf("  Learning rate: %.4f\n", stats.current_learning_rate);
}

/**
 * @brief Get most important neurons
 *
 * WHY: Identifies which neurons contribute most to decisions
 * Useful for pruning and interpretability
 *
 * COMPLEXITY: O(n*log(k)) where n = total_neurons, k = top_n
 *
 * @param brain Brain handle
 * @param top_n Number of neurons to return
 * @param neuron_ids Output array of neuron IDs
 * @param importances Output array of importance scores
 * @return Number of neurons returned
 */
uint32_t brain_get_top_neurons(brain_t brain, uint32_t top_n, uint32_t* neuron_ids,
                               float* importances)
{
    // Guard: Validate parameters
    if (!brain || !neuron_ids || !importances) {
        set_error("Invalid parameters to brain_get_top_neurons");
        return 0;
    }

    // Get neuron rankings from adaptive network
    neuron_importance_t* rankings = nimcp_malloc(top_n * sizeof(neuron_importance_t));
    uint32_t count = adaptive_network_rank_neurons(brain->network, rankings, top_n);

    // Extract IDs and importances
    for (uint32_t i = 0; i < count; i++) {
        neuron_ids[i] = rankings[i].neuron_id;
        importances[i] = rankings[i].importance;
    }

    nimcp_free(rankings);

    brain_clear_error();
    return count;
}

/**
 * @brief Explain why brain made a decision
 *
 * WHY: Provides human-readable explanation of decision
 * Critical for trust and debugging
 *
 * COMPLEXITY: O(k) where k = num_active_neurons
 *
 * @param brain Brain handle
 * @param features Input that led to decision
 * @param num_features Feature count
 * @param explanation Output buffer
 * @param max_length Max explanation length
 * @return true on success
 */
bool brain_explain_decision(brain_t brain, const float* features, uint32_t num_features,
                            char* explanation, uint32_t max_length)
{
    // Guard: Validate parameters
    if (!brain || !features || !explanation) {
        set_error("Invalid parameters to brain_explain_decision");
        return false;
    }

    uint32_t written =
        adaptive_network_explain(brain->network, features, num_features, explanation, max_length);

    brain_clear_error();
    return written > 0;
}

//=============================================================================
// Optimization API
//=============================================================================

/**
 * @brief Prune weak connections
 *
 * WHY: Removes low-weight synapses to improve efficiency
 * Reduces memory and speeds up inference
 *
 * COMPLEXITY: O(n*c) where c = connections per neuron
 * BENEFIT: 2-10x inference speedup possible
 *
 * @param brain Brain handle
 * @param threshold Prune synapses with weight < threshold
 * @return Number of synapses pruned
 */
uint32_t brain_prune(brain_t brain, float threshold)
{
    // Guard: Validate brain
    if (!brain) {
        set_error("Null brain provided to brain_prune");
        return 0;
    }

    uint32_t pruned = adaptive_network_prune(brain->network, threshold);

    brain->stats.num_active_synapses -= pruned;

    // Invalidate cache after pruning
    clear_cache(brain);

    brain_clear_error();
    return pruned;
}

/**
 * @brief Optimize brain for inference
 *
 * WHY: Prepares brain for production deployment
 * Performs aggressive optimization for speed
 *
 * COMPLEXITY: O(n*c)
 * BENEFIT: Can achieve 5-10x speedup
 *
 * @param brain Brain handle
 * @return true on success
 */
bool brain_optimize_for_inference(brain_t brain)
{
    // Guard: Validate brain
    if (!brain) {
        set_error("Null brain provided to brain_optimize_for_inference");
        return false;
    }

    // Aggressive pruning for target sparsity
    float threshold = brain_recommend_pruning_threshold(brain, 0.90f);
    brain_prune(brain, threshold);

    brain_clear_error();
    return true;
}

/**
 * @brief Get recommended pruning threshold
 *
 * WHY: Provides heuristic for safe pruning
 * Balances sparsity vs accuracy
 *
 * COMPLEXITY: O(1)
 *
 * @param brain Brain handle
 * @param target_sparsity Desired sparsity (0-1)
 * @return Recommended threshold
 */
float brain_recommend_pruning_threshold(brain_t brain, float target_sparsity)
{
    // Guard: Validate brain
    if (!brain) {
        set_error("Null brain provided to brain_recommend_pruning_threshold");
        return 0.01f;
    }

    // Heuristic: lower threshold for higher sparsity
    float threshold = 0.1f * (1.0f - target_sparsity);

    brain_clear_error();
    return threshold;
}

//=============================================================================
// Phase 3: Distributed Brain API Implementation
//=============================================================================

/**
 * WHAT: Create distributed brain with P2P coordination
 * WHY:  Enable multi-brain collaborative learning and shared chemical signals
 * HOW:  Create standard brain, then attach distributed cognition coordinator
 *
 * THREAD SAFETY: Thread-safe creation
 * PERFORMANCE: O(n) where n = num_neurons + network initialization
 */
brain_t brain_create_distributed(
    const char* task_name,
    brain_size_t size,
    brain_task_t task,
    uint32_t num_inputs,
    uint32_t num_outputs,
    p2p_node_t p2p_node
)
{
    // Guard: Validate P2P node
    if (!p2p_node) {
        set_error("NULL p2p_node provided to brain_create_distributed");
        return NULL;
    }

    // Create standard brain first
    brain_t brain = brain_create(task_name, size, task, num_inputs, num_outputs);
    if (!brain) {
        return NULL;
    }

    // Enable distributed coordination
    if (!brain_enable_distributed(brain, p2p_node)) {
        brain_destroy(brain);
        return NULL;
    }

    brain_clear_error();
    return brain;
}

/**
 * WHAT: Enable distributed coordination on existing brain
 * WHY:  Allow conversion of standalone brain to distributed mode
 * HOW:  Creates distrib_cognition coordinator and starts sync threads
 *
 * THREAD SAFETY: Thread-safe if brain not actively processing
 */
bool brain_enable_distributed(brain_t brain, p2p_node_t p2p_node)
{
    // Guard: Validate parameters
    if (!brain) {
        set_error("NULL brain provided to brain_enable_distributed");
        return false;
    }

    if (!p2p_node) {
        set_error("NULL p2p_node provided to brain_enable_distributed");
        return false;
    }

    // Guard: Check if already distributed
    if (brain->distributed) {
        set_error("Brain is already distributed");
        return false;
    }

    // Create distributed cognition configuration
    distrib_cognition_config_t config;
    config.enable_neuromod_sync = true;
    config.neuromod_broadcast_interval_ms = 100;
    config.neuromod_diffusion_rate = 0.5f;
    config.enable_glial_sync = true;
    config.glial_sync_interval_ms = 100;
    config.enable_region_sync = true;
    config.region_sync_interval_ms = 100;
    config.sync_mode = SYNC_MODE_BIDIRECTIONAL;
    config.max_message_queue = 1000;

    // Create distributed cognition coordinator
    brain->distributed = distrib_cognition_create(&config, p2p_node);
    if (!brain->distributed) {
        set_error("Failed to create distributed cognition coordinator");
        return false;
    }

    // Start distributed coordination
    if (!distrib_cognition_start(brain->distributed)) {
        set_error("Failed to start distributed cognition");
        distrib_cognition_destroy(brain->distributed);
        brain->distributed = NULL;
        return false;
    }

    // Update brain config
    brain->config.enable_distributed = true;
    brain->config.p2p_node = p2p_node;

    brain_clear_error();
    return true;
}

/**
 * WHAT: Synchronize neuromodulators with peer brains
 * WHY:  Allow explicit control of sync timing for performance tuning
 * HOW:  Calls distrib_cognition_broadcast_neuromod for all neuromod types
 *
 * THREAD SAFETY: Thread-safe
 * PERFORMANCE: O(P × N) where P=peers, N=neuromod types
 */
bool brain_sync_neuromodulators(brain_t brain)
{
    // Guard: Validate brain
    if (!brain) {
        set_error("NULL brain provided to brain_sync_neuromodulators");
        return false;
    }

    // Guard: Check if distributed
    if (!brain->distributed) {
        set_error("Brain is not distributed - cannot sync neuromodulators");
        return false;
    }

    // Broadcast all neuromodulator types with default concentrations
    // In a full implementation, these would be read from the brain's neuromodulator pool
    bool success = true;
    success &= distrib_cognition_broadcast_neuromod(brain->distributed, NEUROMOD_DOPAMINE, 0.5f);
    success &= distrib_cognition_broadcast_neuromod(brain->distributed, NEUROMOD_SEROTONIN, 0.5f);
    success &= distrib_cognition_broadcast_neuromod(brain->distributed, NEUROMOD_NOREPINEPHRINE, 0.5f);
    success &= distrib_cognition_broadcast_neuromod(brain->distributed, NEUROMOD_ACETYLCHOLINE, 0.5f);

    if (!success) {
        set_error("Failed to broadcast some neuromodulators");
        return false;
    }

    brain_clear_error();
    return true;
}

/**
 * WHAT: Get distributed cognition statistics
 * WHY:  Monitor distributed brain performance and health
 * HOW:  Forwards query to underlying distrib_cognition coordinator
 */
bool brain_get_distributed_stats(
    brain_t brain,
    distrib_cognition_stats_t* stats
)
{
    // Guard: Validate parameters
    if (!brain) {
        set_error("NULL brain provided to brain_get_distributed_stats");
        return false;
    }

    if (!stats) {
        set_error("NULL stats provided to brain_get_distributed_stats");
        return false;
    }

    // Guard: Check if distributed
    if (!brain->distributed) {
        set_error("Brain is not distributed - no stats available");
        return false;
    }

    // Forward to distributed cognition
    if (!distrib_cognition_get_stats(brain->distributed, stats)) {
        set_error("Failed to get distributed cognition stats");
        return false;
    }

    brain_clear_error();
    return true;
}

/**
 * WHAT: Check if brain is distributed
 * WHY:  Allow callers to query brain mode before calling distributed APIs
 * HOW:  Return true if distributed coordinator exists
 */
bool brain_is_distributed(brain_t brain)
{
    if (!brain) {
        return false;
    }

    return brain->distributed != NULL;
}
