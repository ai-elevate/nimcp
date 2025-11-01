/**
 * @file nimcp_introspection.c
 * @brief Implementation of brain introspection and state access
 *
 * WHAT: Implements APIs to query internal neural network state, neuron
 * activity, uncertainty estimation, pattern tracking, and topology analysis.
 *
 * WHY: Provides the window into brain internals needed for metacognition,
 * explanation, debugging, and conscious self-awareness.
 *
 * HOW: Accesses internal neural network structures, computes statistics,
 * tracks patterns, estimates uncertainty via ensemble methods, and maintains
 * activity history. Thread-safe with mutex protection.
 *
 * DESIGN PATTERNS:
 * - Factory: Context creation/destruction
 * - Strategy: Different state extraction strategies
 * - Observer: State change callbacks
 * - Memento: Activity history snapshots
 * - Facade: Simplified access to complex internals
 *
 * THREAD SAFETY: All public functions are thread-safe via mutex protection.
 * Internal state is protected by context->lock.
 *
 * @author NIMCP Development Team
 * @date 2025
 */

#include "nimcp_introspection.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "nimcp_adaptive.h"
#include "nimcp_brain.h"
#include "utils/nimcp_memory.h"
#include "utils/nimcp_queue.h"
#include "utils/nimcp_thread.h"
#include "utils/nimcp_time.h"
#include "utils/nimcp_vector.h"

/* ========================================================================
 * INTERNAL STRUCTURES
 * ======================================================================== */

/**
 * WHAT: Pattern registry entry
 * WHY: Track learned patterns and their activity
 * HOW: Hash table of pattern metadata
 */
typedef struct pattern_entry {
    char* name;                 /* Pattern identifier */
    float current_activity;     /* Current activation */
    float activity_sum;         /* Sum for average */
    uint32_t activation_count;  /* Count for average */
    float pattern_strength;     /* Learning strength */
    uint64_t first_learned;     /* First occurrence */
    uint64_t last_activated;    /* Last occurrence */
    struct pattern_entry* next; /* Hash table chain */
} pattern_entry_t;

/**
 * WHAT: Pattern registry (hash table)
 * WHY: Fast O(1) lookup of patterns by name
 * HOW: Chained hash table with 256 buckets
 */
typedef struct {
    pattern_entry_t* buckets[256]; /* Hash buckets */
    uint32_t num_patterns;         /* Total patterns */
    nimcp_mutex_t lock;            /* Thread safety */
} pattern_registry_t;

/**
 * WHAT: Introspection context structure (Pimpl)
 * WHY: Encapsulate implementation details
 * HOW: Opaque pointer pattern
 *
 * REFACTORING NOTE (activity history):
 * - Replaced custom circular buffer (activity_history_buffer_t) with nimcp_queue
 * - WHY: Eliminates ~50 lines of code duplication, standardized API
 * - BENEFITS: Blocking operations, better statistics, peek/clear, thread-safe
 * - NO PERFORMANCE IMPACT: Both use mutex (not lock-free)
 */
struct introspection_context_struct {
    brain_t brain;                 /* Associated brain */
    introspection_config_t config; /* Configuration */

    /* Pattern tracking */
    pattern_registry_t* pattern_registry; /* Learned patterns */

    /* Activity history - now using standardized queue utility */
    nimcp_queue_handle_t activity_queue; /* Activity snapshots queue */

    /* Network topology cache */
    network_topology_t topology; /* Cached topology */
    bool topology_cached;        /* Is topology valid? */

    /* Statistics */
    introspection_stats_t stats; /* Performance stats */

    /* Thread safety */
    nimcp_mutex_t lock; /* Protects context */
};

/* ========================================================================
 * FORWARD DECLARATIONS
 * ======================================================================== */

static uint32_t hash_string(const char* str);
static pattern_entry_t* pattern_registry_lookup(pattern_registry_t* registry, const char* name);
static void pattern_registry_update(pattern_registry_t* registry, const char* name, float activity);
static float compute_entropy(const float* values, uint32_t count);
static float compute_cosine_similarity(const float* a, const float* b, uint32_t dimension);

/* ========================================================================
 * CONFIGURATION
 * ======================================================================== */

/**
 * WHAT: Get default introspection configuration
 * WHY: Sensible defaults for most use cases
 * HOW: Return pre-configured struct
 */
introspection_config_t introspection_default_config(void)
{
    introspection_config_t config = {.default_strategy = STATE_STRATEGY_BALANCED,
                                     .activity_threshold = 0.3f,
                                     .history_size = 100,
                                     .enable_pattern_tracking = true,
                                     .enable_uncertainty_estimation = true,
                                     .uncertainty_ensemble_size = 5,
                                     .on_state_change = NULL,
                                     .callback_context = NULL};
    return config;
}

/* ========================================================================
 * CONTEXT MANAGEMENT (Factory Pattern)
 * ======================================================================== */

/**
 * WHAT: Create introspection context
 * WHY: Initialize introspection subsystem for a brain
 * HOW: Allocate context, initialize pattern registry, history buffer
 *
 * COMPLEXITY: O(n) where n = network size (topology analysis)
 */
introspection_context_t introspection_context_create(brain_t brain,
                                                     const introspection_config_t* config)
{
    /* WHAT: Validate inputs */
    if (brain == NULL) {
        return NULL;
    }

    /* WHAT: Allocate context */
    introspection_context_t context =
        (introspection_context_t) nimcp_calloc(1, sizeof(struct introspection_context_struct));
    if (context == NULL) {
        return NULL;
    }

    /* WHAT: Initialize basic fields */
    context->brain = brain;
    context->config = config ? *config : introspection_default_config();
    context->topology_cached = false;
    nimcp_mutex_init(&context->lock, NULL);

    /* WHAT: Create pattern registry if enabled */
    /* WHY: Track learned patterns for queries */
    if (context->config.enable_pattern_tracking) {
        context->pattern_registry =
            (pattern_registry_t*) nimcp_calloc(1, sizeof(pattern_registry_t));
        if (context->pattern_registry == NULL) {
            nimcp_free(context);
            return NULL;
        }
        nimcp_mutex_init(&context->pattern_registry->lock, NULL);
    }

    /* WHAT: Create activity history queue using nimcp_queue utility */
    /* WHY: Track state evolution over time with standardized queue API */
    /* HOW: Create blocking queue with capacity from config */
    nimcp_queue_config_t queue_config = {.max_size = context->config.history_size,
                                         .item_size = sizeof(activity_history_entry_t),
                                         .is_blocking =
                                             false,  // Don't block - drop oldest on overflow
                                         .timeout_ms = 0};

    nimcp_result_t result = nimcp_queue_create(&queue_config, &context->activity_queue);
    if (result != NIMCP_SUCCESS) {
        if (context->pattern_registry) {
            nimcp_mutex_destroy(&context->pattern_registry->lock);
            nimcp_free(context->pattern_registry);
        }
        nimcp_mutex_destroy(&context->lock);
        nimcp_free(context);
        return NULL;
    }

    /* WHAT: Initialize statistics */
    memset(&context->stats, 0, sizeof(introspection_stats_t));
    context->stats.memory_used_bytes =
        sizeof(struct introspection_context_struct) + sizeof(pattern_registry_t) +
        (context->config.history_size * sizeof(activity_history_entry_t));

    return context;
}

/**
 * WHAT: Destroy introspection context
 * WHY: Free all resources and prevent memory leaks
 * HOW: Free pattern registry, history, topology, context itself
 */
void introspection_context_destroy(introspection_context_t context)
{
    if (context == NULL) {
        return;
    }

    /* WHAT: Free pattern registry */
    if (context->pattern_registry) {
        nimcp_mutex_lock(&context->pattern_registry->lock);
        for (uint32_t i = 0; i < 256; i++) {
            pattern_entry_t* entry = context->pattern_registry->buckets[i];
            while (entry) {
                pattern_entry_t* next = entry->next;
                nimcp_free(entry->name);
                nimcp_free(entry);
                entry = next;
            }
        }
        nimcp_mutex_unlock(&context->pattern_registry->lock);
        nimcp_mutex_destroy(&context->pattern_registry->lock);
        nimcp_free(context->pattern_registry);
    }

    /* WHAT: Destroy activity history queue */
    if (context->activity_queue) {
        nimcp_queue_destroy(context->activity_queue);
    }

    /* WHAT: Free topology cache */
    if (context->topology_cached) {
        network_topology_free(&context->topology);
    }

    /* WHAT: Destroy mutex and free context */
    nimcp_mutex_destroy(&context->lock);
    nimcp_free(context);
}

/* ========================================================================
 * NEURON POPULATION QUERIES
 * ======================================================================== */

/**
 * WHAT: Get currently active neuron population
 * WHY: See which neurons are firing
 * HOW: Scan network, collect neurons above threshold
 *
 * COMPLEXITY: O(n) where n = network size
 * PERFORMANCE: ~0.1-1ms for typical networks
 */
neuron_population_t brain_get_active_population(introspection_context_t context, float threshold)
{
    neuron_population_t population;
    memset(&population, 0, sizeof(neuron_population_t));

    if (context == NULL) {
        return population;
    }

    nimcp_mutex_lock(&context->lock);
    context->stats.queries_total++;
    context->stats.queries_active_population++;
    nimcp_mutex_unlock(&context->lock);

    /* WHAT: Get brain's underlying adaptive network */
    /* WHY: Need direct access to neuron activations */
    adaptive_network_t network = brain_get_network(context->brain);
    if (network == NULL) {
        return population;
    }

    /* WHAT: Get network size */
    uint32_t total_neurons = adaptive_network_get_neuron_count(network);
    if (total_neurons == 0) {
        return population;
    }

    /* WHAT: Allocate temporary arrays for gathering active neurons */
    /* WHY: Don't know count until we scan, so use max size */
    uint32_t* temp_ids = (uint32_t*) nimcp_malloc(total_neurons * sizeof(uint32_t));
    float* temp_activations = (float*) nimcp_malloc(total_neurons * sizeof(float));

    if (temp_ids == NULL || temp_activations == NULL) {
        nimcp_free(temp_ids);
        nimcp_free(temp_activations);
        return population;
    }

    /* WHAT: Get active neurons from real network */
    /* HOW: Use new adaptive_network API that does the scan for us */
    uint32_t active_count = adaptive_network_get_active_neurons(network, threshold, temp_ids,
                                                                temp_activations, total_neurons);

    /* WHAT: Set population metadata */
    population.total_neurons = total_neurons;
    population.activity_threshold = threshold;
    population.timestamp = nimcp_time_monotonic_ms();
    population.num_active = active_count;

    if (active_count == 0) {
        nimcp_free(temp_ids);
        nimcp_free(temp_activations);
        return population;
    }

    /* WHAT: Allocate right-sized arrays for results */
    population.neuron_ids = (uint32_t*) nimcp_malloc(active_count * sizeof(uint32_t));
    population.activation_levels = (float*) nimcp_malloc(active_count * sizeof(float));

    if (population.neuron_ids == NULL || population.activation_levels == NULL) {
        nimcp_free(population.neuron_ids);
        nimcp_free(population.activation_levels);
        nimcp_free(temp_ids);
        nimcp_free(temp_activations);
        memset(&population, 0, sizeof(neuron_population_t));
        return population;
    }

    /* WHAT: Copy active neuron data to results */
    memcpy(population.neuron_ids, temp_ids, active_count * sizeof(uint32_t));
    memcpy(population.activation_levels, temp_activations, active_count * sizeof(float));

    /* WHAT: Free temporary arrays */
    nimcp_free(temp_ids);
    nimcp_free(temp_activations);

    return population;
}

/**
 * WHAT: Free neuron population structure
 * WHY: Release allocated arrays
 * HOW: Free arrays, zero struct
 */
void neuron_population_free(neuron_population_t* population)
{
    if (population == NULL) {
        return;
    }

    nimcp_free(population->neuron_ids);
    nimcp_free(population->activation_levels);
    memset(population, 0, sizeof(neuron_population_t));
}

/**
 * WHAT: Get detailed activity for specific neuron
 * WHY: Deep inspection of individual neuron
 * HOW: Access neuron structure, compute statistics
 *
 * COMPLEXITY: O(1)
 */
neuron_activity_t brain_get_neuron_activity(introspection_context_t context, uint32_t neuron_id)
{
    neuron_activity_t activity;
    memset(&activity, 0, sizeof(neuron_activity_t));

    if (context == NULL) {
        return activity;
    }

    nimcp_mutex_lock(&context->lock);
    context->stats.queries_total++;
    nimcp_mutex_unlock(&context->lock);

    /* WHAT: Get brain's underlying network */
    adaptive_network_t network = brain_get_network(context->brain);
    if (network == NULL) {
        return activity;
    }

    /* WHAT: Fill in neuron metadata */
    activity.neuron_id = neuron_id;

    /* WHAT: Get real neuron activation from network */
    float raw_activation;
    if (!adaptive_network_get_neuron_activation(network, neuron_id, &raw_activation)) {
        return activity; /* Neuron doesn't exist */
    }

    /* WHAT: Normalize biological potential to 0-1 range */
    /* WHY: Neurons use biological potentials (-65 to +30 mV), but
     * introspection API should present normalized activations */
    /* HOW: Map [-65, +30] mV to [0, 1] */
    const float REST_POTENTIAL = -65.0f;
    const float PEAK_POTENTIAL = 30.0f;
    activity.activation = (raw_activation - REST_POTENTIAL) / (PEAK_POTENTIAL - REST_POTENTIAL);

    /* WHAT: Clamp to valid range */
    if (activity.activation < 0.0f)
        activity.activation = 0.0f;
    if (activity.activation > 1.0f)
        activity.activation = 1.0f;

    /* WHAT: Get connection count and total weight */
    adaptive_network_get_connection_count(network, neuron_id, &activity.num_connections);
    adaptive_network_get_total_weight(network, neuron_id, &activity.total_weight);

    /* WHAT: Compute derived properties */
    /* TODO: Get gradient from actual backprop when available */
    activity.gradient = 0.0f;                             /* Not tracked yet */
    activity.decision_contribution = activity.activation; /* Approximate */
    activity.is_active = activity.activation >= context->config.activity_threshold;

    return activity;
}

/* ========================================================================
 * INTERNAL STATE QUERIES
 * ======================================================================== */

/**
 * WHAT: Get compressed internal state representation
 * WHY: State vector for logging, comparison, transfer
 * HOW: Sample neurons based on strategy, compute entropy
 *
 * DESIGN PATTERN: Strategy (different sampling strategies)
 * COMPLEXITY: O(n*s) where s = sampling rate
 */
brain_state_t brain_get_internal_state(introspection_context_t context,
                                       state_extraction_strategy_t strategy)
{
    brain_state_t state;
    memset(&state, 0, sizeof(brain_state_t));

    if (context == NULL) {
        return state;
    }

    nimcp_mutex_lock(&context->lock);
    context->stats.queries_total++;
    context->stats.queries_internal_state++;
    nimcp_mutex_unlock(&context->lock);

    /* WHAT: Determine sampling rate based on strategy */
    /* WHY: Trade-off between speed and accuracy */
    float sampling_rate;
    const char* strategy_name;
    switch (strategy) {
        case STATE_STRATEGY_FAST:
            sampling_rate = 0.1f; /* 10% sampling */
            strategy_name = "fast";
            break;
        case STATE_STRATEGY_BALANCED:
            sampling_rate = 0.3f; /* 30% sampling */
            strategy_name = "balanced";
            break;
        case STATE_STRATEGY_DETAILED:
            sampling_rate = 1.0f; /* 100% sampling */
            strategy_name = "detailed";
            break;
        default:
            sampling_rate = 0.3f;
            strategy_name = "balanced";
    }

    /* WHAT: Get brain's underlying network */
    adaptive_network_t network = brain_get_network(context->brain);
    if (network == NULL) {
        return state;
    }

    /* WHAT: Get actual network size */
    uint32_t total_neurons = adaptive_network_get_neuron_count(network);
    uint32_t sampled_neurons = (uint32_t) (total_neurons * sampling_rate);

    if (sampled_neurons == 0) {
        sampled_neurons = 1; /* At least one sample */
    }

    /* WHAT: Allocate state vector */
    state.dimension = sampled_neurons;
    state.state_vector = (float*) nimcp_malloc(sampled_neurons * sizeof(float));
    if (state.state_vector == NULL) {
        return state;
    }

    /* WHAT: Sample neuron activations from real network */
    /* HOW: For fast/balanced, sample evenly spaced neurons */
    /*      For detailed, get all neurons */
    uint32_t stride = (uint32_t) (1.0f / sampling_rate);
    if (stride == 0)
        stride = 1;

    /* WHAT: Biological potential normalization constants */
    const float REST_POTENTIAL = -65.0f;
    const float PEAK_POTENTIAL = 30.0f;

    for (uint32_t i = 0; i < sampled_neurons; i++) {
        uint32_t neuron_id = i * stride;
        if (neuron_id >= total_neurons) {
            neuron_id = total_neurons - 1; /* Clamp to valid range */
        }

        /* WHAT: Get real neuron activation */
        float raw_activation;
        if (adaptive_network_get_neuron_activation(network, neuron_id, &raw_activation)) {
            /* WHAT: Normalize biological potential to 0-1 range */
            state.state_vector[i] =
                (raw_activation - REST_POTENTIAL) / (PEAK_POTENTIAL - REST_POTENTIAL);
            if (state.state_vector[i] < 0.0f)
                state.state_vector[i] = 0.0f;
            if (state.state_vector[i] > 1.0f)
                state.state_vector[i] = 1.0f;
        } else {
            state.state_vector[i] = 0.0f; /* Default if unavailable */
        }
    }

    /* WHAT: Compute information content (entropy) */
    /* WHY: Measure how much information is in this state */
    state.information_content = compute_entropy(state.state_vector, sampled_neurons);

    /* WHAT: Set metadata */
    state.compression_ratio = (float) total_neurons / (float) sampled_neurons;
    state.timestamp = nimcp_time_monotonic_ms();

    /* WHAT: Generate human-readable interpretation */
    /* WHY: Help developers understand what state means */
    char interp_buffer[256];
    snprintf(interp_buffer, sizeof(interp_buffer),
             "State extracted using %s strategy (%.0f%% sampling), "
             "%.2f bits entropy, compression ratio %.2fx",
             strategy_name, sampling_rate * 100.0f, state.information_content,
             state.compression_ratio);
    state.interpretation = nimcp_strdup(interp_buffer);

    return state;
}

/**
 * WHAT: Free brain state structure
 * WHY: Release allocated memory
 * HOW: Free vector and interpretation
 */
void brain_state_free(brain_state_t* state)
{
    if (state == NULL) {
        return;
    }

    nimcp_free(state->state_vector);
    nimcp_free(state->interpretation);
    memset(state, 0, sizeof(brain_state_t));
}

/**
 * WHAT: Compare two brain states for similarity
 * WHY: Detect state changes, measure drift
 * HOW: Cosine similarity between state vectors
 *
 * COMPLEXITY: O(d) where d = dimension
 */
float brain_state_similarity(const brain_state_t* state1, const brain_state_t* state2)
{
    if (state1 == NULL || state2 == NULL) {
        return 0.0f;
    }

    /* WHAT: States must have same dimension */
    if (state1->dimension != state2->dimension) {
        return 0.0f;
    }

    return compute_cosine_similarity(state1->state_vector, state2->state_vector, state1->dimension);
}

/* ========================================================================
 * UNCERTAINTY ESTIMATION
 * ======================================================================== */

/**
 * WHAT: Estimate uncertainty for a decision
 * WHY: Metacognition requires knowing when uncertain
 * HOW: Ensemble method - variance = epistemic, entropy = aleatoric
 *
 * METHOD:
 * 1. Get predictions from ensemble of models
 * 2. Variance of predictions = epistemic uncertainty (model doesn't know)
 * 3. Entropy of each prediction = aleatoric uncertainty (data is noisy)
 * 4. Total = epistemic + aleatoric
 *
 * COMPLEXITY: O(k*m) where k=ensemble size, m=model complexity
 * TIME: ~1-5ms depending on ensemble size
 */
brain_uncertainty_t brain_get_uncertainty(introspection_context_t context, const float* features,
                                          uint32_t num_features)
{
    brain_uncertainty_t uncertainty;
    memset(&uncertainty, 0, sizeof(brain_uncertainty_t));

    if (context == NULL || features == NULL || num_features == 0) {
        return uncertainty;
    }

    if (!context->config.enable_uncertainty_estimation) {
        return uncertainty;
    }

    nimcp_mutex_lock(&context->lock);
    context->stats.queries_total++;
    context->stats.queries_uncertainty++;
    nimcp_mutex_unlock(&context->lock);

    /* WHAT: Get ensemble size from config */
    uint32_t ensemble_size = context->config.uncertainty_ensemble_size;
    uncertainty.ensemble_size = ensemble_size;

    /* WHAT: Allocate array for ensemble predictions */
    uncertainty.ensemble_predictions = (float*) nimcp_malloc(ensemble_size * sizeof(float));
    if (uncertainty.ensemble_predictions == NULL) {
        return uncertainty;
    }

    /* WHAT: Get predictions from ensemble */
    /* WHY: Variance in predictions indicates uncertainty */
    /* TODO: In real implementation, this would use actual ensemble */
    /* For now, simulate with random variations */
    float mean_prediction = 0.0f;
    for (uint32_t i = 0; i < ensemble_size; i++) {
        /* Simulate ensemble predictions with some variance */
        uncertainty.ensemble_predictions[i] = 0.5f + ((float) rand() / RAND_MAX - 0.5f) * 0.3f;
        mean_prediction += uncertainty.ensemble_predictions[i];
    }
    mean_prediction /= ensemble_size;

    /* WHAT: Compute epistemic uncertainty (variance of predictions) */
    /* WHY: High variance means models disagree = model doesn't know */
    float variance = 0.0f;
    for (uint32_t i = 0; i < ensemble_size; i++) {
        float diff = uncertainty.ensemble_predictions[i] - mean_prediction;
        variance += diff * diff;
    }
    variance /= ensemble_size;
    uncertainty.epistemic = sqrtf(variance); /* Standard deviation */

    /* WHAT: Compute aleatoric uncertainty (entropy of mean prediction) */
    /* WHY: High entropy means prediction is uncertain = data is noisy */
    /* For binary classification: H = -p*log(p) - (1-p)*log(1-p) */
    float p = mean_prediction;
    if (p < 1e-6f)
        p = 1e-6f;
    if (p > 1.0f - 1e-6f)
        p = 1.0f - 1e-6f;
    uncertainty.aleatoric = -(p * log2f(p) + (1.0f - p) * log2f(1.0f - p));

    /* WHAT: Combine uncertainties */
    /* WHY: Total uncertainty is sum of epistemic and aleatoric */
    uncertainty.total = uncertainty.epistemic + uncertainty.aleatoric;
    if (uncertainty.total > 1.0f)
        uncertainty.total = 1.0f;

    uncertainty.confidence = 1.0f - uncertainty.total;

    return uncertainty;
}

/**
 * WHAT: Free uncertainty structure
 * WHY: Release ensemble predictions array
 * HOW: Free array, zero struct
 */
void brain_uncertainty_free(brain_uncertainty_t* uncertainty)
{
    if (uncertainty == NULL) {
        return;
    }

    nimcp_free(uncertainty->ensemble_predictions);
    memset(uncertainty, 0, sizeof(brain_uncertainty_t));
}

/* ========================================================================
 * PATTERN QUERIES
 * ======================================================================== */

/**
 * WHAT: Simple string hash function
 * WHY: Fast O(1) pattern lookup
 * HOW: djb2 hash algorithm
 */
static uint32_t hash_string(const char* str)
{
    uint32_t hash = 5381;
    int c;
    while ((c = *str++) != 0) {
        hash = ((hash << 5) + hash) + c; /* hash * 33 + c */
    }
    return hash % 256; /* Modulo bucket count */
}

/**
 * WHAT: Lookup pattern in registry
 * WHY: Find existing pattern entry
 * HOW: Hash lookup with chaining
 *
 * COMPLEXITY: O(1) average, O(n) worst case
 */
static pattern_entry_t* pattern_registry_lookup(pattern_registry_t* registry, const char* name)
{
    if (registry == NULL || name == NULL) {
        return NULL;
    }

    uint32_t bucket = hash_string(name);
    pattern_entry_t* entry = registry->buckets[bucket];

    while (entry) {
        if (strcmp(entry->name, name) == 0) {
            return entry;
        }
        entry = entry->next;
    }

    return NULL;
}

/**
 * WHAT: Update pattern activity in registry
 * WHY: Track pattern activation over time
 * HOW: Update existing entry or create new one
 */
static void pattern_registry_update(pattern_registry_t* registry, const char* name, float activity)
{
    if (registry == NULL || name == NULL) {
        return;
    }

    nimcp_mutex_lock(&registry->lock);

    /* WHAT: Try to find existing entry */
    pattern_entry_t* entry = pattern_registry_lookup(registry, name);

    if (entry) {
        /* WHAT: Update existing pattern */
        entry->current_activity = activity;
        entry->activity_sum += activity;
        entry->activation_count++;
        entry->last_activated = nimcp_time_monotonic_ms();
    } else {
        /* WHAT: Create new pattern entry */
        entry = (pattern_entry_t*) nimcp_calloc(1, sizeof(pattern_entry_t));
        if (entry == NULL) {
            nimcp_mutex_unlock(&registry->lock);
            return;
        }

        entry->name = nimcp_strdup(name);
        entry->current_activity = activity;
        entry->activity_sum = activity;
        entry->activation_count = 1;
        entry->pattern_strength = 0.5f; /* Initial strength */
        entry->first_learned = nimcp_time_monotonic_ms();
        entry->last_activated = entry->first_learned;

        /* WHAT: Insert into hash table */
        uint32_t bucket = hash_string(name);
        entry->next = registry->buckets[bucket];
        registry->buckets[bucket] = entry;
        registry->num_patterns++;
    }

    nimcp_mutex_unlock(&registry->lock);
}

/**
 * WHAT: Check if pattern is currently active
 * WHY: Query specific learned patterns
 * HOW: Hash lookup, check activity threshold
 *
 * COMPLEXITY: O(1)
 */
bool brain_is_pattern_active(introspection_context_t context, const char* pattern_name)
{
    if (context == NULL || pattern_name == NULL) {
        return false;
    }

    if (!context->config.enable_pattern_tracking || context->pattern_registry == NULL) {
        return false;
    }

    nimcp_mutex_lock(&context->lock);
    context->stats.queries_total++;
    context->stats.queries_pattern++;
    nimcp_mutex_unlock(&context->lock);

    nimcp_mutex_lock(&context->pattern_registry->lock);
    pattern_entry_t* entry = pattern_registry_lookup(context->pattern_registry, pattern_name);
    bool is_active =
        (entry != NULL && entry->current_activity >= context->config.activity_threshold);
    nimcp_mutex_unlock(&context->pattern_registry->lock);

    return is_active;
}

/**
 * WHAT: Get detailed pattern information
 * WHY: Inspect pattern metadata
 * HOW: Lookup and copy pattern entry
 *
 * COMPLEXITY: O(1)
 */
pattern_info_t* brain_get_pattern_info(introspection_context_t context, const char* pattern_name)
{
    if (context == NULL || pattern_name == NULL) {
        return NULL;
    }

    if (!context->config.enable_pattern_tracking || context->pattern_registry == NULL) {
        return NULL;
    }

    nimcp_mutex_lock(&context->pattern_registry->lock);
    pattern_entry_t* entry = pattern_registry_lookup(context->pattern_registry, pattern_name);

    if (entry == NULL) {
        nimcp_mutex_unlock(&context->pattern_registry->lock);
        return NULL;
    }

    /* WHAT: Allocate and populate info structure */
    pattern_info_t* info = (pattern_info_t*) nimcp_malloc(sizeof(pattern_info_t));
    if (info == NULL) {
        nimcp_mutex_unlock(&context->pattern_registry->lock);
        return NULL;
    }

    info->pattern_name = nimcp_strdup(entry->name);
    info->current_activity = entry->current_activity;
    info->average_activity = entry->activity_sum / entry->activation_count;
    info->pattern_strength = entry->pattern_strength;
    info->activation_count = entry->activation_count;
    info->first_learned = entry->first_learned;
    info->last_activated = entry->last_activated;

    nimcp_mutex_unlock(&context->pattern_registry->lock);

    return info;
}

/**
 * WHAT: Free pattern info structure
 * WHY: Release memory
 * HOW: Free name string and struct
 */
void pattern_info_free(pattern_info_t* info)
{
    if (info == NULL) {
        return;
    }

    nimcp_free(info->pattern_name);
    nimcp_free(info);
}

/**
 * WHAT: Get list of all registered patterns
 * WHY: Discover learned patterns
 * HOW: Scan hash table, collect names
 *
 * COMPLEXITY: O(p) where p = number of patterns
 */
char** brain_list_patterns(introspection_context_t context, uint32_t* num_patterns)
{
    if (context == NULL || num_patterns == NULL) {
        return NULL;
    }

    if (!context->config.enable_pattern_tracking || context->pattern_registry == NULL) {
        *num_patterns = 0;
        return NULL;
    }

    nimcp_mutex_lock(&context->pattern_registry->lock);

    *num_patterns = context->pattern_registry->num_patterns;
    if (*num_patterns == 0) {
        nimcp_mutex_unlock(&context->pattern_registry->lock);
        return NULL;
    }

    /* WHAT: Allocate array of strings */
    char** pattern_list = (char**) nimcp_malloc(*num_patterns * sizeof(char*));
    if (pattern_list == NULL) {
        nimcp_mutex_unlock(&context->pattern_registry->lock);
        *num_patterns = 0;
        return NULL;
    }

    /* WHAT: Collect pattern names from all buckets */
    uint32_t index = 0;
    for (uint32_t bucket = 0; bucket < 256; bucket++) {
        pattern_entry_t* entry = context->pattern_registry->buckets[bucket];
        while (entry) {
            pattern_list[index++] = nimcp_strdup(entry->name);
            entry = entry->next;
        }
    }

    nimcp_mutex_unlock(&context->pattern_registry->lock);

    return pattern_list;
}

/**
 * WHAT: Free pattern list
 * WHY: Release memory from brain_list_patterns
 * HOW: Free each string, free array
 */
void pattern_list_free(char** pattern_list, uint32_t num_patterns)
{
    if (pattern_list == NULL) {
        return;
    }

    for (uint32_t i = 0; i < num_patterns; i++) {
        nimcp_free(pattern_list[i]);
    }
    nimcp_free(pattern_list);
}

/* ========================================================================
 * NETWORK TOPOLOGY
 * ======================================================================== */

/**
 * WHAT: Get network topology statistics
 * WHY: Understand network structure
 * HOW: Analyze connection graph, compute metrics
 *
 * COMPLEXITY: O(n + e) where n=neurons, e=edges
 */
/**
 * @brief Deep copy neurons_per_layer array
 *
 * DESIGN PATTERN: Extract Method
 * WHY: Eliminates code duplication, single responsibility
 * PREVENTS: Double-free by ensuring separate memory allocations
 *
 * @param source Source array to copy from
 * @param num_layers Number of layers to copy
 * @return New allocated array or NULL on failure
 */
static uint32_t* clone_neurons_per_layer(const uint32_t* source, uint32_t num_layers)
{
    /* Guard clause: validate inputs */
    if (source == NULL || num_layers == 0) {
        return NULL;
    }

    /* Allocate new array */
    uint32_t* clone = (uint32_t*) nimcp_malloc(num_layers * sizeof(uint32_t));

    /* Guard clause: allocation failed */
    if (clone == NULL) {
        return NULL;
    }

    /* Copy data */
    memcpy(clone, source, num_layers * sizeof(uint32_t));
    return clone;
}

/**
 * @brief Create deep copy of topology structure
 *
 * DESIGN PATTERN: Prototype (deep copy for safety)
 * WHY: Prevents double-free when caller frees returned topology
 *
 * @param source Topology to copy
 * @return Deep copy with separate neurons_per_layer allocation
 */
static network_topology_t clone_topology(const network_topology_t* source)
{
    /* Guard clause: validate input */
    if (source == NULL) {
        network_topology_t empty;
        memset(&empty, 0, sizeof(network_topology_t));
        return empty;
    }

    /* Shallow copy all scalar fields */
    network_topology_t clone = *source;

    /* Deep copy neurons_per_layer array */
    clone.neurons_per_layer =
        clone_neurons_per_layer(source->neurons_per_layer, source->num_layers);

    return clone;
}

/**
 * @brief Build new topology from brain structure
 *
 * DESIGN PATTERN: Builder
 * WHY: Separate topology construction from caching logic
 * TODO: Replace simulated data with actual brain structure
 *
 * @return Newly constructed topology
 */
static network_topology_t build_topology(void)
{
    network_topology_t topology;
    memset(&topology, 0, sizeof(network_topology_t));

    /* Populate scalar fields */
    topology.total_neurons = 10000;
    topology.total_connections = 50000;
    topology.avg_connections_per_neuron = 5.0f;
    topology.connection_sparsity = 1.0f - (50000.0f / (10000.0f * 10000.0f));
    topology.clustering_coefficient = 0.3f;
    topology.num_layers = 3;

    /* Allocate and populate layer array */
    topology.neurons_per_layer = (uint32_t*) nimcp_malloc(3 * sizeof(uint32_t));

    /* Guard clause: allocation failed */
    if (topology.neurons_per_layer == NULL) {
        return topology;
    }

    topology.neurons_per_layer[0] = 1000; /* Input layer */
    topology.neurons_per_layer[1] = 8000; /* Hidden layer */
    topology.neurons_per_layer[2] = 1000; /* Output layer */

    return topology;
}

network_topology_t brain_get_topology(introspection_context_t context)
{
    /* Guard clause: validate context */
    if (context == NULL) {
        network_topology_t empty;
        memset(&empty, 0, sizeof(network_topology_t));
        return empty;
    }

    nimcp_mutex_lock(&context->lock);

    /* CASE 1: Return cached topology (most common path) */
    /* DESIGN PATTERN: Lazy Initialization with memoization */
    if (context->topology_cached) {
        network_topology_t result = clone_topology(&context->topology);
        nimcp_mutex_unlock(&context->lock);
        return result;
    }

    /* CASE 2: Build and cache new topology (first call only) */
    network_topology_t topology = build_topology();

    /* Cache with independent copy to prevent double-free */
    context->topology = clone_topology(&topology);
    context->topology_cached = true;

    nimcp_mutex_unlock(&context->lock);

    return topology;
}

/**
 * WHAT: Free topology structure
 * WHY: Release neurons_per_layer array
 * HOW: Free array, zero struct
 */
void network_topology_free(network_topology_t* topology)
{
    if (topology == NULL) {
        return;
    }

    nimcp_free(topology->neurons_per_layer);
    memset(topology, 0, sizeof(network_topology_t));
}

/* ========================================================================
 * ACTIVITY HISTORY
 * ======================================================================== */

/**
 * WHAT: Add entry to activity history queue
 * WHY: Track state evolution over time
 * HOW: Enqueue entry using nimcp_queue (drops oldest on overflow)
 *
 * REFACTORING NOTE:
 * - Replaced custom circular buffer logic with nimcp_queue_enqueue
 * - SIMPLIFIED: ~20 lines → 1 function call
 * - BENEFITS: Better error handling, statistics, standard API
 * - NOTE: This function is currently UNUSED (defined but never called)
 *         Keeping it for future use when activity tracking is implemented
 *
 * DESIGN PATTERN: Memento (store state snapshots)
 *
 * @param context Introspection context
 * @param entry Activity history entry to add
 */
static void activity_history_add(introspection_context_t context,
                                 const activity_history_entry_t* entry)
{
    if (context == NULL || entry == NULL) {
        return;
    }

    // WHY timeout=0: Non-blocking - queue configured to drop on overflow
    nimcp_queue_enqueue(context->activity_queue, entry, 0);
}

/**
 * WHAT: Get recent activity history
 * WHY: Track brain state evolution
 * HOW: Dequeue all entries from activity queue
 *
 * REFACTORING NOTE:
 * - Replaced custom circular buffer traversal with nimcp_queue operations
 * - SIMPLIFIED: ~30 lines of modulo arithmetic → simple dequeue loop
 * - BEHAVIOR CHANGE: Now consumes (empties) the queue when called
 *   WHY: nimcp_queue doesn't have peek_all, and this matches "get history" semantics
 *   IMPACT: Low - function is rarely called, and getting history to examine it makes sense
 *
 * COMPLEXITY: O(h) where h = history size
 * THREAD-SAFE: Yes (queue operations are thread-safe)
 *
 * @param context Introspection context
 * @param num_entries Output: number of history entries returned
 * @return Array of history entries (must be freed by caller) or NULL if empty
 */
activity_history_entry_t* brain_get_activity_history(introspection_context_t context,
                                                     uint32_t* num_entries)
{
    if (context == NULL || num_entries == NULL) {
        return NULL;
    }

    // Get current queue size
    size_t queue_size = nimcp_queue_get_size(context->activity_queue);
    *num_entries = (uint32_t) queue_size;

    if (queue_size == 0) {
        return NULL;  // No history available
    }

    // Allocate array for history entries
    activity_history_entry_t* history =
        (activity_history_entry_t*) nimcp_malloc(queue_size * sizeof(activity_history_entry_t));

    if (history == NULL) {
        *num_entries = 0;
        return NULL;
    }

    // Dequeue all entries (empties the queue)
    // WHY CONSUME: No peek_all API, and "get history" implies reading it out
    for (uint32_t i = 0; i < queue_size; i++) {
        nimcp_result_t result = nimcp_queue_dequeue(context->activity_queue, &history[i], 0);
        if (result != NIMCP_SUCCESS) {
            // Partial failure - return what we got
            *num_entries = i;
            if (i == 0) {
                nimcp_free(history);
                return NULL;
            }
            break;
        }
    }

    return history;
}

/* ========================================================================
 * STATISTICS
 * ======================================================================== */

/**
 * WHAT: Get introspection statistics
 * WHY: Monitor performance
 * HOW: Copy stats structure
 *
 * COMPLEXITY: O(1)
 */
bool introspection_get_stats(introspection_context_t context, introspection_stats_t* stats)
{
    if (context == NULL || stats == NULL) {
        return false;
    }

    nimcp_mutex_lock(&context->lock);
    *stats = context->stats;
    nimcp_mutex_unlock(&context->lock);

    return true;
}

/**
 * WHAT: Reset introspection statistics
 * WHY: Clear counters for new measurement
 * HOW: Zero all counters except memory usage
 *
 * COMPLEXITY: O(1)
 */
void introspection_reset_stats(introspection_context_t context)
{
    if (context == NULL) {
        return;
    }

    nimcp_mutex_lock(&context->lock);

    size_t memory_used = context->stats.memory_used_bytes;
    memset(&context->stats, 0, sizeof(introspection_stats_t));
    context->stats.memory_used_bytes = memory_used;

    nimcp_mutex_unlock(&context->lock);
}

/* ========================================================================
 * HELPER FUNCTIONS
 * ======================================================================== */


/**
 * WHAT: Compute Shannon entropy of values
 * WHY: Measure information content
 * HOW: H = -sum(p * log2(p))
 *
 * COMPLEXITY: O(n)
 */
static float compute_entropy(const float* values, uint32_t count)
{
    if (values == NULL || count == 0) {
        return 0.0f;
    }

    /* WHAT: Normalize values to probabilities */
    float sum = 0.0f;
    for (uint32_t i = 0; i < count; i++) {
        sum += fabsf(values[i]);
    }

    if (sum < 1e-10f) {
        return 0.0f;
    }

    /* WHAT: Compute entropy */
    float entropy = 0.0f;
    for (uint32_t i = 0; i < count; i++) {
        float p = fabsf(values[i]) / sum;
        if (p > 1e-10f) {
            entropy -= p * log2f(p);
        }
    }

    return entropy;
}

/**
 * WHAT: Compute cosine similarity between two vectors
 * WHY: Compare state vectors for similarity
 * HOW: dot(a,b) / (||a|| * ||b||)
 *
 * COMPLEXITY: O(d) where d = dimension
 */
static float compute_cosine_similarity(const float* a, const float* b, uint32_t dimension)
{
    /**
     * WHAT: Delegate to vector utility function
     * WHY: Eliminate code duplication, use centralized implementation
     * HOW: Direct call to nimcp_vector_cosine_similarity
     */
    return nimcp_vector_cosine_similarity(a, b, dimension);
}
