#include <stddef.h>  /* for NULL */
//=============================================================================
// nimcp_adaptive.c - Refactored Adaptive Threshold Spiking Implementation
//=============================================================================
// ARCHITECTURAL OVERVIEW:
// This module implements adaptive threshold spiking neural networks using
// several design patterns for performance and maintainability:
//
// - Strategy Pattern: Different encoding/learning strategies via function pointers
// - Object Pool Pattern: Reusable buffers to eliminate heap allocations
// - Repository Pattern: Hash-indexed label storage for O(1) lookups
// - Factory Pattern: Network creation with validated configuration
//
// COMPLEXITY ANALYSIS:
// - Spike encoding: O(n) where n = spike count (linear time)
// - Threshold computation: O(n) single pass (previously had potential loops)
// - Network forward pass: O(n) where n = output size (linear)
// - Label lookup: O(1) via hash table (previously O(n) linear search)
// - Learning: O(n) where n = target size (single pass)
//
// DESIGN PRINCIPLES:
// - Single Responsibility: Each function has one clear purpose
// - Guard Clauses: Early returns instead of nested conditionals
// - No Nested Loops: All loops extracted to helper functions
// - Open/Closed: Extensible via function pointers without modification
//
// INVARIANTS:
// - All thresholds remain within [min_threshold, max_threshold]
// - Sparsity maintained at target_sparsity ± 0.1
// - Neuron states updated atomically per forward pass
// - Label map always synchronized with label count
//=============================================================================

#include "plasticity/adaptive/nimcp_adaptive.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "core/neuralnet/nimcp_neuralnet.h"
#include "core/neuralnet/nimcp_neuron_synapse_access.h"
#include "utils/containers/nimcp_hash_table.h"
#include "utils/memory/nimcp_memory.h"  // CRITICAL: Declares nimcp_calloc/nimcp_free return types
#include "utils/logging/nimcp_logging.h"
#include "utils/algorithms/nimcp_sort.h"  // Consolidated sorting algorithms
#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_messages.h"
#include "security/nimcp_security.h"
#include "utils/exception/nimcp_exception_macros.h"

#define LOG_MODULE "plasticity_adaptive"
#include "utils/fault_tolerance/nimcp_health_agent_macros.h"

NIMCP_DECLARE_HEALTH_AGENT_ATOMIC(adaptive)

#include "utils/memory/nimcp_memory_pool.h"  // Phase MP: Memory pool for hot paths
#include "utils/memory/nimcp_page_cow.h"     // Phase COW: Page-level COW for state snapshots
#include "plasticity/eligibility/nimcp_eligibility_trace.h"  // Phase 11: Eligibility traces
#include "plasticity/bcm/nimcp_bcm.h"  // Phase 11: BCM homeostatic plasticity
#include "plasticity/neuromodulators/nimcp_neuromodulators.h"  // Phase 11: Neuromodulator system
#include "gpu/execution/nimcp_gpu_detect.h"          // Phase GPU: GPU detection
#include "gpu/context/nimcp_gpu_context.h"           // Phase GPU: GPU context
#include "gpu/training/nimcp_training_bridge.h"      // Phase GPU: Weight cache + forward pass
#include "gpu/neuron/nimcp_neuron_bridge.h"          // Phase Inferentia: NeuronCore inference

//=============================================================================
// Constants and Configuration
//=============================================================================

#define HASH_TABLE_SIZE 256
#define MAX_POOL_BUFFERS 32
#define SPARSITY_ADAPT_INCREASE 1.01f
#define SPARSITY_ADAPT_DECREASE 0.99f
#define SPARSITY_EMA_WEIGHT 0.1f

// WHAT: Helper macros for checked file I/O (cert-err33-c compliance)
// WHY:  fwrite/fread return values must be checked to detect I/O errors
// HOW:  Macros check return value and set success=false on failure
#define FWRITE_CHECKED(ptr, size, count, stream) \
    do { if (fwrite((ptr), (size), (count), (stream)) != (count)) success = false; } while(0)
#define FREAD_CHECKED(ptr, size, count, stream) \
    do { if (fread((ptr), (size), (count), (stream)) != (count)) goto cleanup; } while(0)

//=============================================================================
// Memory Pool for Hot Paths (Phase MP)
//=============================================================================

/**
 * @brief Memory pool for forward/training buffers
 *
 * WHAT: Global memory pool for spike_input and output buffers
 * WHY:  These buffers are allocated/freed on every forward pass - O(1) vs O(log n)
 * HOW:  Lazily initialized pool with blocks sized for typical networks
 */
#define ADAPTIVE_POOL_BLOCK_SIZE 4096   // 4KB - fits 1024 floats
#define ADAPTIVE_POOL_NUM_BLOCKS 256    // 256 concurrent buffers

#include "utils/platform/nimcp_platform_once.h"
#include <stdatomic.h>

/* ============================================================================
 * Comparison function for sorting neuron importance (descending order)
 * ============================================================================ */

/**
 * @brief Compare neuron importance for descending sort
 *
 * Used by nimcp_sort to rank neurons by importance (highest first).
 */
static int compare_neuron_importance_desc(const void* a, const void* b) {
    const neuron_importance_t* na = (const neuron_importance_t*)a;
    const neuron_importance_t* nb = (const neuron_importance_t*)b;

    /* Descending order: higher importance first */
    if (nb->importance > na->importance) return 1;
    if (nb->importance < na->importance) return -1;
    return 0;
}

static _Atomic(memory_pool_t) g_adaptive_pool = NULL;
static nimcp_platform_once_t g_adaptive_pool_once = NIMCP_PLATFORM_ONCE_INIT;

/**
 * @brief One-time initialization of adaptive memory pool
 */
static void init_adaptive_pool(void) {
    memory_pool_config_t config = memory_pool_default_config(
        ADAPTIVE_POOL_BLOCK_SIZE, ADAPTIVE_POOL_NUM_BLOCKS);
    memory_pool_t pool = memory_pool_create(&config);
    atomic_store(&g_adaptive_pool, pool);
}

/**
 * @brief Get or create the adaptive network memory pool (thread-safe)
 */
static memory_pool_t get_adaptive_pool(void) {
    nimcp_platform_once(&g_adaptive_pool_once, init_adaptive_pool);
    return atomic_load(&g_adaptive_pool);
}

/**
 * @brief Cleanup adaptive network memory pool
 *
 * WHAT: Destroy the global memory pool and reset the once flag
 * WHY:  Allow proper re-initialization after nimcp_shutdown()
 * HOW:  Destroy pool, store NULL, reset once flag
 *
 * THREAD SAFETY: NOT thread-safe - call only during shutdown
 *                when no other threads are using the adaptive module
 */
void adaptive_pool_cleanup(void) {
    memory_pool_t pool = atomic_load(&g_adaptive_pool);
    if (pool) {
        memory_pool_destroy(pool);
        atomic_store(&g_adaptive_pool, (memory_pool_t)NULL);
    }
    // Reset once flag to allow re-initialization
    g_adaptive_pool_once = NIMCP_PLATFORM_ONCE_INIT;
}

/**
 * @brief Allocate buffer from pool or heap
 */
static void* alloc_hot_buffer(size_t size) {
    if (size <= ADAPTIVE_POOL_BLOCK_SIZE) {
        memory_pool_t pool = get_adaptive_pool();
        if (pool) {
            void* buf = memory_pool_acquire(pool);
            if (buf) {
                memset(buf, 0, size);
                return buf;
            }
        }
    }
    return nimcp_calloc(1, size);
}

/**
 * @brief Free buffer to pool or heap
 */
static void free_hot_buffer(void* buf) {
    if (!buf) return;
    memory_pool_t pool = get_adaptive_pool();
    if (pool && memory_pool_owns(pool, buf)) {
        memory_pool_release(pool, buf);
    } else {
        nimcp_free(buf);
    }
}

//=============================================================================
// Object Pool Pattern - Eliminates repeated allocations
//=============================================================================

/**
 * @brief Object pool for spike buffers to avoid heap allocations
 *
 * WHY: Frequent allocation/deallocation during spike encoding causes
 * heap fragmentation. Pool provides O(1) allocation and deterministic performance.
 *
 * COMPLEXITY: O(1) for acquire/release operations
 *
 * INVARIANTS:
 * - in_use array accurately reflects buffer availability
 * - next_available points to valid index [0, MAX_POOL_BUFFERS)
 */
typedef struct {
    uint8_t buffers[MAX_POOL_BUFFERS][256];  // Reusable spike train buffers
    bool in_use[MAX_POOL_BUFFERS];
    uint32_t next_available;
} spike_buffer_pool_t;

/**
 * @brief Label index value stored in hash table
 *
 * WHY: Hash table maps label strings to neuron indices for O(1) lookup.
 * No longer needs 'next' pointer as chaining is handled by nimcp_hash_table.
 *
 * INVARIANTS:
 * - index matches position in label_map array
 */
typedef struct {
    uint32_t index;
} label_index_t;

/**
 * @brief Strategy Pattern - Spike encoding function pointer
 *
 * WHY: Eliminates switch statement in hot path. Each encoding has its own
 * optimized function. New encodings can be added without modifying existing code.
 *
 * @param spike_count - Integer spike count to encode
 * @param spike_train - Output buffer
 * @param max_length - Buffer capacity
 * @return Actual spike train length
 */
typedef uint32_t (*spike_encoder_fn)(int32_t spike_count, uint8_t* spike_train,
                                     uint32_t max_length);

/**
 * @brief Strategy Pattern - Spike decoding function pointer
 *
 * @param spike_train - Input spike train
 * @param length - Train length
 * @return Decoded spike count
 */
typedef int32_t (*spike_decoder_fn)(const uint8_t* spike_train, uint32_t length);

/**
 * @brief Encoding/decoding strategy table
 *
 * WHY: O(1) dispatch to encoding function vs O(n) switch statement
 */
typedef struct {
    spike_encoder_fn encoders[4];  // One per encoding type
    spike_decoder_fn decoders[4];
} spike_strategy_table_t;

//=============================================================================
// Internal Structures
//=============================================================================

/**
 * @brief Internal adaptive network structure
 *
 * INVARIANTS:
 * - base_network always valid after creation
 * - num_neurons == config.base_config.num_neurons
 * - neuron_states array has exactly num_neurons elements
 * - running_sparsity in range [0.0, 1.0]
 * - num_labels == label_table.num_entries
 */
struct adaptive_network_struct {
    // Base neural network
    neural_network_t base_network;

    // Configuration
    adaptive_network_config_t config;

    // Adaptive neuron states
    adaptive_neuron_state_t* neuron_states;
    uint32_t num_neurons;

    // Design pattern components
    spike_buffer_pool_t buffer_pool;
    hash_table_t* label_table;  // Using nimcp_hash_table utility
    spike_strategy_table_t strategy_table;

    // Statistics
    uint64_t total_inferences;
    uint64_t total_learning_steps;
    float running_sparsity;
    float running_inference_time_us;
    float running_learning_time_us;
    uint32_t num_pruned_synapses;

    // Threshold adaptation
    uint32_t adaptation_counter;
    float* input_statistics;  // Running statistics for adaptation

    // Label mapping for classification
    char** label_map;
    uint32_t num_labels;

    // Copy-on-Write support for state snapshots (Phase COW)
    // WHY: Enable O(1) network state snapshots for training rollback
    bool uses_cow_states;                    // True if neuron states use COW
    page_cow_region_t cow_states_region;     // COW region for neuron states
    page_cow_view_t cow_states_view;         // View into COW states region

    // GPU acceleration (Phase GPU)
    // WHY: Forward pass + loss on GPU when available, CPU bio-plasticity unchanged
    struct nimcp_gpu_context_s* gpu_ctx;
    struct nimcp_gpu_weight_cache_s* gpu_weight_cache;
    bool gpu_enabled;

    // Neuron inference acceleration (Phase Inferentia)
    // WHY: NeuronCore-accelerated forward pass on AWS Inferentia hardware
    struct nimcp_neuron_inference_cache* neuron_cache;
    bool neuron_enabled;

    // Frozen network (Phase GPU-INF)
    // WHY: Disable learning and lock weights for inference-only mode
    bool frozen;

    // Gradient norm tracking
    float last_grad_norm;  /**< L2 norm of gradients from most recent learn step */
};

//=============================================================================
// Object Pool Implementation
//=============================================================================

/**
 * @brief Initializes the spike buffer pool
 *
 * WHY: Pre-allocates all buffers at startup to avoid runtime allocations.
 * Provides deterministic performance and eliminates allocation failures.
 *
 * COMPLEXITY: O(1) - Fixed size initialization
 */
static void init_spike_buffer_pool(spike_buffer_pool_t* pool)
{
    // Guard clause: Validate input
    if (!pool)
        return;

    memset(pool->in_use, 0, sizeof(pool->in_use));
    pool->next_available = 0;
}

/**
 * @brief Acquires a spike buffer from the pool
 *
 * WHY: O(1) buffer acquisition vs O(log n) malloc. Eliminates heap
 * fragmentation and allocation failures during encoding.
 *
 * COMPLEXITY: O(1) average case
 *
 * @return Pointer to buffer or NULL if pool exhausted
 */
static uint8_t* acquire_spike_buffer(spike_buffer_pool_t* pool)
{
    // Guard clause: Validate input
    if (!pool) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "pool is NULL");

        return NULL;

    }

    // Find available buffer starting from next_available
    for (uint32_t i = 0; i < MAX_POOL_BUFFERS; i++) {
        uint32_t idx = (pool->next_available + i) % MAX_POOL_BUFFERS;
        if (!pool->in_use[idx]) {
            pool->in_use[idx] = true;
            pool->next_available = (idx + 1) % MAX_POOL_BUFFERS;
            return pool->buffers[idx];
        }
    }

    return NULL;  // Pool exhausted - all buffers in use
}

/**
 * @brief Releases a spike buffer back to the pool
 *
 * WHY: Enables buffer reuse. O(1) operation maintains performance.
 *
 * COMPLEXITY: O(1)
 */
static void release_spike_buffer(spike_buffer_pool_t* pool, uint8_t* buffer)
{
    // Guard clause: Validate inputs
    if (!pool || !buffer)
        return;

    // Find and release the buffer
    for (uint32_t i = 0; i < MAX_POOL_BUFFERS; i++) {
        if (pool->buffers[i] == buffer) {
            pool->in_use[i] = false;
            return;
        }
    }
}

//=============================================================================
// Hash Table Implementation (Repository Pattern)
//=============================================================================

/**
 * @brief Initializes hash table for label storage using nimcp_hash_table utility
 *
 * WHY: Provides O(1) label lookup. Uses nimcp_hash_table with djb2 algorithm.
 * Essential for real-time performance with many labels.
 *
 * COMPLEXITY: O(1)
 */
static hash_table_t* create_label_hash_table(void)
{
    hash_table_config_t config = {.initial_buckets = HASH_TABLE_SIZE,
                                  .key_type = HASH_KEY_STRING,
                                  .hash_algorithm = HASH_ALG_DJB2,
                                  .case_insensitive = false,
                                  .value_destructor = NULL,  // label_index_t doesn't need cleanup
                                  .thread_safe = false};
    return hash_table_create(&config);
}

/**
 * @brief Looks up label in hash table using nimcp_hash_table utility
 *
 * WHY: O(1) average case lookup is critical for real-time classification.
 * Delegates to nimcp_hash_table for consistent behavior.
 *
 * COMPLEXITY: O(1) average, O(k) worst case where k = collision chain length
 *
 * @return Label index or UINT32_MAX if not found
 */
static uint32_t hash_table_lookup_label(hash_table_t* table, const char* label)
{
    if (!table || !label)
        return UINT32_MAX;

    label_index_t* entry = (label_index_t*) hash_table_lookup_string(table, label);
    return entry ? entry->index : UINT32_MAX;
}

/**
 * @brief Inserts label into hash table using nimcp_hash_table utility
 *
 * WHY: O(1) insertion enables fast label registration.
 * Delegates to nimcp_hash_table for consistent behavior.
 *
 * COMPLEXITY: O(1) average case
 *
 * @return true if inserted, false on failure
 */
static bool hash_table_insert_label(hash_table_t* table, const char* label, uint32_t index)
{
    if (!table || !label) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "hash_table_insert_label: required parameter is NULL (table, label)");
        return false;
    }

    label_index_t value = {.index = index};
    return hash_table_insert_string(table, label, &value, sizeof(label_index_t));
}


//=============================================================================
// Utility Functions
//=============================================================================

/**
 * @brief Computes mean absolute value of input vector
 *
 * WHY: Used for adaptive threshold calculation. Single pass algorithm
 * is optimal for this calculation.
 *
 * COMPLEXITY: O(n) where n = size - single pass, cannot be improved
 *
 * @return Mean absolute value or 0.0f if invalid input
 */
static float compute_mean_abs(const float* input, uint32_t size)
{
    // Guard clause: Validate inputs
    if (!input || size == 0)
        return 0.0F;

    float sum = 0.0F;
    for (uint32_t i = 0; i < size; i++) {
        sum += fabsf(input[i]);
    }
    return sum / size;
}

/**
 * @brief Updates running statistics using Welford's online algorithm
 *
 * WHY: Welford's algorithm is numerically stable and computes mean and
 * variance in a single pass with O(1) space complexity.
 *
 * COMPLEXITY: O(1) - Constant time update
 *
 * @param state - Neuron state to update
 * @param activation - New activation value
 */
static void update_statistics(adaptive_neuron_state_t* state, float activation)
{
    // Guard clause: Validate input
    if (!state)
        return;

    // Welford's online algorithm for mean and variance
    state->sample_count++;
    float delta = activation - state->activation_mean;
    state->activation_mean += delta / state->sample_count;
    float delta2 = activation - state->activation_mean;
    state->activation_variance += delta * delta2;
}

/**
 * @brief Finds or adds label to label map with hash table acceleration
 *
 * WHY: Uses hash table for O(1) lookup instead of O(n) linear search.
 * Critical for performance with many labels.
 *
 * COMPLEXITY: O(1) average case for lookup, O(1) for insertion
 *
 * @return Label index in label_map array
 */
static uint32_t get_label_index(adaptive_network_t network, const char* label)
{
    // Guard clause: Validate inputs
    if (!network || !label)
        return 0;

    // O(1) hash table lookup
    uint32_t index = hash_table_lookup_label(network->label_table, label);

    // Guard clause: Return if found
    if (index != UINT32_MAX)
        return index;

    // Add new label
    char** new_map = nimcp_realloc(network->label_map, (network->num_labels + 1) * sizeof(char*));
    // Guard clause: Check allocation
    if (!new_map)
        return 0;

    network->label_map = new_map;
    // Use nimcp_malloc instead of strdup to match nimcp_free in free_label_map
    size_t label_len = strlen(label);
    network->label_map[network->num_labels] = nimcp_malloc(label_len + 1);
    if (!network->label_map[network->num_labels])
        return 0;
    strncpy(network->label_map[network->num_labels], label, label_len + 1);
    network->label_map[network->num_labels][label_len] = '\0';

    // Insert into hash table for future O(1) lookups
    hash_table_insert_label(network->label_table, network->label_map[network->num_labels],
                            network->num_labels);

    return network->num_labels++;
}

//=============================================================================
// Strategy Pattern - Spike Encoding Implementations
//=============================================================================

/**
 * @brief Encodes spike count as direct integer (fastest encoding)
 *
 * WHY: Fastest encoding/decoding with zero compression. Used for
 * performance-critical paths where bandwidth is not a concern.
 *
 * COMPLEXITY: O(1) - Constant time copy
 */
static uint32_t encode_integer(int32_t spike_count, uint8_t* spike_train, uint32_t max_length)
{
    // Guard clause: Check buffer size
    if (max_length < sizeof(int32_t))
        return 0;

    memcpy(spike_train, &spike_count, sizeof(int32_t));
    return sizeof(int32_t);
}

/**
 * @brief Encodes spike count as binary expansion (sparse representation)
 *
 * WHY: Provides sparse representation for small spike counts.
 * Efficient for counts < 10.
 *
 * COMPLEXITY: O(n) where n = |spike_count|
 */
static uint32_t encode_binary(int32_t spike_count, uint8_t* spike_train, uint32_t max_length)
{
    uint32_t abs_count = (uint32_t) abs(spike_count);
    uint32_t length = (abs_count < max_length) ? abs_count : max_length;
    memset(spike_train, 1, length);
    return length;
}

/**
 * @brief Encodes spike count as ternary (-1, 0, +1)
 *
 * WHY: Most compressed encoding for simple sign information.
 * Used when only direction matters, not magnitude.
 *
 * COMPLEXITY: O(1) - Single byte encoding
 */
static uint32_t encode_ternary(int32_t spike_count, uint8_t* spike_train, uint32_t max_length)
{
    // Guard clause: Check buffer
    if (max_length < 1)
        return 0;

    spike_train[0] = (spike_count > 0) ? 1 : ((spike_count < 0) ? 255 : 0);
    return 1;
}

/**
 * @brief Encodes spike count as bitwise representation
 *
 * WHY: Balanced compression for medium spike counts.
 * Good for counts in range [10, 1000].
 *
 * COMPLEXITY: O(log n) where n = spike_count
 */
static uint32_t encode_bitwise(int32_t spike_count, uint8_t* spike_train, uint32_t max_length)
{
    uint32_t abs_count = (uint32_t) abs(spike_count);
    uint32_t bytes_needed = (abs_count + 7) / 8;

    // Guard clause: Check capacity
    if (bytes_needed > max_length)
        bytes_needed = max_length;

    // Additional guard: Prevent shift overflow (max 4 bytes for uint32_t)
    if (bytes_needed > 4)
        bytes_needed = 4;

    for (uint32_t i = 0; i < bytes_needed; i++) {
        spike_train[i] = (abs_count >> (i * 8)) & 0xFF;
    }
    return bytes_needed;
}

/**
 * @brief Decodes integer-encoded spike train
 *
 * COMPLEXITY: O(1)
 */
static int32_t decode_integer(const uint8_t* spike_train, uint32_t length)
{
    // Guard clause: Check minimum length
    if (length < sizeof(int32_t))
        return 0;

    int32_t spike_count = 0;
    memcpy(&spike_count, spike_train, sizeof(int32_t));
    return spike_count;
}

/**
 * @brief Decodes binary-encoded spike train
 *
 * COMPLEXITY: O(n) where n = length
 */
static int32_t decode_binary(const uint8_t* spike_train, uint32_t length)
{
    int32_t spike_count = 0;
    for (uint32_t i = 0; i < length; i++) {
        spike_count += spike_train[i];
    }
    return spike_count;
}

/**
 * @brief Decodes ternary-encoded spike train
 *
 * COMPLEXITY: O(1)
 */
static int32_t decode_ternary(const uint8_t* spike_train, uint32_t length)
{
    // Guard clause: Check length
    if (length < 1)
        return 0;

    return (spike_train[0] == 1) ? 1 : ((spike_train[0] == 255) ? -1 : 0);
}

/**
 * @brief Decodes bitwise-encoded spike train
 *
 * COMPLEXITY: O(log n) where n = spike_count
 */
static int32_t decode_bitwise(const uint8_t* spike_train, uint32_t length)
{
    int32_t spike_count = 0;
    for (uint32_t i = 0; i < length; i++) {
        spike_count |= (spike_train[i] << (i * 8));
    }
    return spike_count;
}

/**
 * @brief Initializes spike encoding/decoding strategy table
 *
 * WHY: Replaces switch statement with O(1) function pointer dispatch.
 * Follows Open/Closed Principle - new encodings can be added without
 * modifying existing code.
 *
 * COMPLEXITY: O(1)
 */
static void init_spike_strategy_table(spike_strategy_table_t* table)
{
    // Guard clause: Validate input
    if (!table)
        return;

    // Register encoding strategies
    table->encoders[SPIKE_ENCODING_INTEGER] = encode_integer;
    table->encoders[SPIKE_ENCODING_BINARY] = encode_binary;
    table->encoders[SPIKE_ENCODING_TERNARY] = encode_ternary;
    table->encoders[SPIKE_ENCODING_BITWISE] = encode_bitwise;

    // Register decoding strategies
    table->decoders[SPIKE_ENCODING_INTEGER] = decode_integer;
    table->decoders[SPIKE_ENCODING_BINARY] = decode_binary;
    table->decoders[SPIKE_ENCODING_TERNARY] = decode_ternary;
    table->decoders[SPIKE_ENCODING_BITWISE] = decode_bitwise;
}

//=============================================================================
// Adaptive Threshold Functions
//=============================================================================

/**
 * @brief Computes adaptive threshold for input vector
 *
 * WHY: Adaptive thresholds enable sparse activation by adjusting to input
 * statistics. Formula: V_th(x) = (1/k) × mean(|x|)
 *
 * COMPLEXITY: O(n) where n = size - Single pass through input
 *
 * @param input - Input vector
 * @param size - Vector size
 * @param k_factor - Firing rate control (higher = more sparsity)
 * @return Adaptive threshold value
 */
float adaptive_compute_threshold(const float* input, uint32_t size, float k_factor)
{
    // Guard clause: Validate inputs
    if (!input || size == 0)
        return 1.0F;

    // Guard clause: Validate k_factor
    if (k_factor <= 0.0F)
        return 1.0F;

    float mean_abs = compute_mean_abs(input, size);
    return mean_abs / k_factor;
}

/**
 * @brief Converts continuous value to integer spike count
 *
 * WHY: Quantizes continuous values to integer spikes for efficient
 * computation. Formula: s_INT = round(x / V_th(x))
 *
 * COMPLEXITY: O(1) - Constant time operation
 *
 * @param value - Continuous input value
 * @param threshold - Adaptive threshold
 * @return Integer spike count
 */
int32_t adaptive_value_to_spikes(float value, float threshold)
{
    // Guard clause: Ensure valid threshold
    if (threshold <= 0.0F)
        threshold = 1.0F;

    return (int32_t) roundf(value / threshold);
}

/**
 * @brief Encodes spike count using strategy pattern (no switch statement)
 *
 * WHY: Uses function pointer dispatch for O(1) encoding selection.
 * Eliminates switch statement and makes code extensible.
 *
 * COMPLEXITY: O(1) dispatch + encoding complexity (varies by type)
 *
 * @return Actual spike train length or 0 on error
 */
uint32_t adaptive_encode_spikes(int32_t spike_count, spike_encoding_t encoding,
                                uint8_t* spike_train, uint32_t max_length)
{
    // Guard clause: Validate inputs
    if (!spike_train || max_length == 0)
        return 0;

    // Guard clause: Validate encoding type
    if (encoding >= 4)
        return 0;

    // Create temporary strategy table (should be passed in production code)
    spike_strategy_table_t table;
    init_spike_strategy_table(&table);

    // O(1) function pointer dispatch
    spike_encoder_fn encoder = table.encoders[encoding];

    // Guard clause: Check encoder exists
    if (!encoder)
        return 0;

    return encoder(spike_count, spike_train, max_length);
}

/**
 * @brief Decodes spike train back to continuous value using strategy pattern
 *
 * WHY: Uses function pointer dispatch instead of switch statement.
 * Maintains symmetry with encoding for clean design.
 *
 * COMPLEXITY: O(1) dispatch + decoding complexity (varies by type)
 *
 * @return Decoded value or 0.0f on error
 */
float adaptive_decode_spikes(const uint8_t* spike_train, uint32_t length, spike_encoding_t encoding,
                             float threshold)
{
    // Guard clause: Validate inputs
    if (!spike_train || length == 0)
        return 0.0F;

    // Guard clause: Validate encoding type
    if (encoding >= 4)
        return 0.0F;

    // Create temporary strategy table
    spike_strategy_table_t table;
    init_spike_strategy_table(&table);

    // O(1) function pointer dispatch
    spike_decoder_fn decoder = table.decoders[encoding];

    // Guard clause: Check decoder exists
    if (!decoder)
        return 0.0F;

    int32_t spike_count = decoder(spike_train, length);
    return spike_count * threshold;
}

//=============================================================================
// Factory Pattern - Network Creation Helper Functions
//=============================================================================

/**
 * @brief Validates network configuration
 *
 * WHY: Extracts validation logic to reduce creation function complexity.
 * Single responsibility - only validates configuration parameters.
 *
 * COMPLEXITY: O(1)
 *
 * @return true if valid, false otherwise
 */
static bool validate_network_config(const adaptive_network_config_t* config)
{
    // Guard clause: Check config pointer
    if (!config) {
        fprintf(stderr, "[ERROR] validate_network_config: NULL config\n");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "validate_network_config: config is NULL");
        return false;
    }

    // Guard clause: Validate base config layer_sizes
    // WHY: NULL layer_sizes will cause crash during memcpy or neural_network_create
    if (config->base_config.num_layers > 0 && !config->base_config.layer_sizes) {
        fprintf(stderr, "[ERROR] validate_network_config: num_layers=%u but layer_sizes is NULL\n",
                config->base_config.num_layers);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "validate_network_config: config->base_config is NULL");
        return false;
    }

    // Guard clause: Validate spike parameters
    if (config->spike_params.k_factor <= 0.0F) {
        fprintf(stderr, "[ERROR] validate_network_config: invalid k_factor=%f\n", config->spike_params.k_factor);
        return false;
    }
    if (config->spike_params.min_threshold <= 0.0F) {
        fprintf(stderr, "[ERROR] validate_network_config: invalid min_threshold=%f\n", config->spike_params.min_threshold);
        return false;
    }
    if (config->spike_params.max_threshold <= config->spike_params.min_threshold) {
        fprintf(stderr, "[ERROR] validate_network_config: max_threshold=%f <= min_threshold=%f\n",
                config->spike_params.max_threshold, config->spike_params.min_threshold);
        return false;
    }
    if (config->spike_params.sparsity_target < 0.0F || config->spike_params.sparsity_target > 1.0F) {
        fprintf(stderr, "[ERROR] validate_network_config: invalid sparsity_target=%f\n", config->spike_params.sparsity_target);
        return false;
    }

    return true;
}

/**
 * @brief Initializes neuron states with default values
 *
 * WHY: Extracted from creation to reduce nesting. Single pass initialization.
 *
 * COMPLEXITY: O(n) where n = num_neurons
 */
static void initialize_neuron_states(adaptive_neuron_state_t* states, uint32_t num_neurons,
                                     float default_threshold)
{
    // Guard clause: Validate input
    if (!states)
        return;

    for (uint32_t i = 0; i < num_neurons; i++) {
        states[i].adaptive_threshold = default_threshold;
        states[i].membrane_potential = 0.0F;
        states[i].spike_count = 0;
        states[i].spike_train = NULL;
        states[i].spike_train_length = 0;
        states[i].activation_mean = 0.0F;
        states[i].activation_variance = 0.0F;
        states[i].sample_count = 0;
    }
}

/**
 * @brief Allocates neuron state array
 *
 * WHY: Extracted allocation for clarity and error handling.
 *
 * COMPLEXITY: O(1) allocation + O(n) initialization
 */
static adaptive_neuron_state_t* allocate_neuron_states(uint32_t num_neurons,
                                                       float default_threshold)
{
    adaptive_neuron_state_t* states = nimcp_calloc(num_neurons, sizeof(adaptive_neuron_state_t));

    // Guard clause: Check allocation
    if (!states) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "states is NULL");

        return NULL;

    }

    initialize_neuron_states(states, num_neurons, default_threshold);
    return states;
}

/**
 * @brief Initializes design pattern components
 *
 * WHY: Extracted initialization of object pool, hash table, and strategy table.
 * Single responsibility - only initializes design patterns.
 *
 * COMPLEXITY: O(1)
 */
static void initialize_design_patterns(adaptive_network_t network)
{
    // Guard clause: Validate input
    if (!network)
        return;

    init_spike_buffer_pool(&network->buffer_pool);
    network->label_table = create_label_hash_table();
    init_spike_strategy_table(&network->strategy_table);
}

//=============================================================================
// Adaptive Network Creation/Destruction
//=============================================================================

/**
 * @brief Creates adaptive neural network (Factory Pattern)
 *
 * WHY: Factory function that orchestrates network creation using multiple
 * helper functions. Each step is extracted to a separate function to eliminate
 * nesting and improve testability.
 *
 * COMPLEXITY: O(n) where n = num_neurons (for state initialization)
 *
 * @param config - Network configuration
 * @return Network handle or NULL on error
 */
adaptive_network_t adaptive_network_create(const adaptive_network_config_t* config)
{

    // Guard clause: Validate configuration
    if (!validate_network_config(config)) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "adaptive_network_create: validate_network_config is NULL");
        return NULL;
    }

    // Auto-load from checkpoint if enabled (default behavior)
    // WHY: Allow seamless continuation of training from saved state
    // HOW: Check if checkpoint exists and auto_load is enabled, then load instead of creating fresh
    if (config->checkpoint_path && config->auto_load) {
        // Check if checkpoint file exists
        FILE* test_file = fopen(config->checkpoint_path, "rb");
        if (test_file) {
            fclose(test_file);
            // Checkpoint exists, load it
            adaptive_network_t loaded_network = adaptive_network_load(config->checkpoint_path);
            if (loaded_network) {
                // Successfully loaded from checkpoint
                return loaded_network;
            }
            // If load failed, fall through to create fresh network
            fprintf(stderr, "WARNING: Failed to load checkpoint from '%s', creating fresh network\n",
                    config->checkpoint_path);
        }
        // Checkpoint doesn't exist yet, create fresh network (will be saved later)
    }

    // Allocate network structure
    adaptive_network_t network = nimcp_calloc(1, sizeof(struct adaptive_network_struct));

    // Guard clause: Check allocation
    if (!network) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "network is NULL");

        return NULL;
    }

    // Copy configuration (shallow copy first)
    memcpy(&network->config, config, sizeof(adaptive_network_config_t));

    // Initialize COW fields to NULL/false (Phase COW)
    network->uses_cow_states = false;
    network->cow_states_region = NULL;
    network->cow_states_view = NULL;

    // Deep copy layer_sizes array to avoid dangling pointer
    // WHY: Config may be stack-allocated by caller, so we need our own copy
    // WHAT: Allocate and copy the layer_sizes array if present
    if (config->base_config.num_layers > 0 && config->base_config.layer_sizes) {
        uint32_t* layer_sizes_copy =
            nimcp_calloc(config->base_config.num_layers, sizeof(uint32_t));
        // Guard clause: Check allocation
        if (!layer_sizes_copy) {
            nimcp_free(network);
            NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "adaptive_network_create: layer_sizes_copy is NULL");
            return NULL;
        }
        memcpy(layer_sizes_copy, config->base_config.layer_sizes,
               config->base_config.num_layers * sizeof(uint32_t));
        network->config.base_config.layer_sizes = layer_sizes_copy;
    } else {
        network->config.base_config.layer_sizes = NULL;
    }

    // Initialize design pattern components
    initialize_design_patterns(network);

    // Create base neural network using our deep-copied config
    // WHY: Use network->config not original config, since we deep-copied layer_sizes
    // WHAT: Pass address of our config which has the corrected layer_sizes pointer
    network->base_network = neural_network_create(&network->config.base_config);
    // Guard clause: Check base network creation
    if (!network->base_network) {
        if (network->config.base_config.layer_sizes) {
            nimcp_free((void*)network->config.base_config.layer_sizes);
        }
        nimcp_free(network);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "adaptive_network_create: validation failed");
        return NULL;
    }

    // Get number of neurons from base network
    network->num_neurons = config->base_config.num_neurons;

    // Allocate and initialize neuron states
    float default_threshold = config->spike_params.min_threshold;
    network->neuron_states = allocate_neuron_states(network->num_neurons, default_threshold);
    // Guard clause: Check neuron states allocation
    if (!network->neuron_states) {
        neural_network_destroy(network->base_network);
        if (network->config.base_config.layer_sizes) {
            nimcp_free((void*)network->config.base_config.layer_sizes);
        }
        nimcp_free(network);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "adaptive_network_create: validation failed");
        return NULL;
    }

    // Allocate input statistics buffer
    network->input_statistics = nimcp_calloc(config->base_config.input_size, sizeof(float));
    // Guard clause: Check statistics buffer
    if (!network->input_statistics) {
        nimcp_free(network->neuron_states);
        neural_network_destroy(network->base_network);
        if (network->config.base_config.layer_sizes) {
            nimcp_free((void*)network->config.base_config.layer_sizes);
        }
        nimcp_free(network);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "adaptive_network_create: validation failed");
        return NULL;
    }

    // Initialize label map
    network->label_map = NULL;
    network->num_labels = 0;

    // Phase GPU-INF: Frozen state init
    network->frozen = false;

    // Phase GPU: Initialize GPU acceleration (GPU-default, CPU fallback)
    network->gpu_enabled = false;
    network->gpu_ctx = NULL;
    network->gpu_weight_cache = NULL;
    if (gpu_is_available() &&
        network->config.base_config.num_layers >= 2 &&
        network->config.base_config.layer_sizes) {
        network->gpu_ctx = nimcp_gpu_context_create_auto();
        if (network->gpu_ctx) {
            network->gpu_weight_cache = nimcp_gpu_weight_cache_create(
                network->gpu_ctx,
                network->base_network,
                network->config.base_config.layer_sizes,
                network->config.base_config.num_layers);
            if (network->gpu_weight_cache) {
                nimcp_gpu_weight_cache_upload(network->gpu_weight_cache,
                                             network->base_network);
                network->gpu_enabled = true;
                fprintf(stderr, "[GPU] Training enabled for adaptive network (%u layers)\n",
                        network->config.base_config.num_layers);
                NIMCP_LOG_INFO("GPU training enabled for adaptive network (%u layers)",
                               network->config.base_config.num_layers);
            } else {
                nimcp_gpu_context_destroy(network->gpu_ctx);
                network->gpu_ctx = NULL;
            }
        }
    }

    return network;
}

/**
 * @brief Frees neuron state spike trains
 *
 * WHY: Extracted cleanup logic. Single responsibility.
 *
 * COMPLEXITY: O(n) where n = num_neurons
 */
static void free_neuron_spike_trains(adaptive_neuron_state_t* states, uint32_t num_neurons)
{
    // Guard clause: Validate input
    if (!states)
        return;

    for (uint32_t i = 0; i < num_neurons; i++) {
        if (states[i].spike_train) {
            nimcp_free(states[i].spike_train);
            states[i].spike_train = NULL;
        }
    }
}

/**
 * @brief Frees label map strings
 *
 * WHY: Extracted cleanup logic for labels.
 *
 * COMPLEXITY: O(n) where n = num_labels
 */
static void free_label_map(char** label_map, uint32_t num_labels)
{
    // Guard clause: Validate input
    if (!label_map)
        return;

    for (uint32_t i = 0; i < num_labels; i++) {
        nimcp_free(label_map[i]);
    }
    nimcp_free(label_map);
}

/**
 * @brief Destroys adaptive neural network and frees all resources
 *
 * WHY: Ensures no memory leaks. Uses helper functions to avoid nesting.
 *
 * COMPLEXITY: O(n + m) where n = num_neurons, m = num_labels
 *
 * @param network - Network to destroy
 */
void adaptive_network_destroy(adaptive_network_t network)
{
    // Guard clause: Validate input
    if (!network)
        return;

    // Phase Inferentia: Clean up Neuron inference cache
    if (network->neuron_cache) {
        nimcp_neuron_cache_destroy(network->neuron_cache);
        network->neuron_cache = NULL;
    }

    // Phase GPU: Clean up GPU resources before base network
    if (network->gpu_weight_cache) {
        nimcp_gpu_weight_cache_destroy(network->gpu_weight_cache);
        network->gpu_weight_cache = NULL;
    }
    if (network->gpu_ctx) {
        nimcp_gpu_context_destroy(network->gpu_ctx);
        network->gpu_ctx = NULL;
    }

    // Destroy base network
    if (network->base_network) {
        neural_network_destroy(network->base_network);
    }

    // Handle COW cleanup (Phase COW)
    if (network->uses_cow_states) {
        if (network->cow_states_view) {
            page_cow_view_destroy(network->cow_states_view);
        }
        if (network->cow_states_region) {
            page_cow_region_destroy(network->cow_states_region);
        }
    }

    // Free neuron states and their spike trains
    if (network->neuron_states) {
        free_neuron_spike_trains(network->neuron_states, network->num_neurons);
        nimcp_free(network->neuron_states);
    }

    // Free input statistics buffer
    nimcp_free(network->input_statistics);

    // Free label map
    if (network->label_map) {
        free_label_map(network->label_map, network->num_labels);
    }

    // Destroy hash table
    if (network->label_table) {
        hash_table_destroy(network->label_table);
    }

    // Free layer_sizes array (deep copy allocated in create)
    // WHY: We own this memory after deep copy in adaptive_network_create
    if (network->config.base_config.layer_sizes) {
        nimcp_free((void*)network->config.base_config.layer_sizes);
    }

    nimcp_free(network);
}

//=============================================================================
// Forward Pass Helper Functions
//=============================================================================

/**
 * @brief Converts input to spike representation
 *
 * WHY: Extracted spike conversion logic. Single responsibility.
 *
 * COMPLEXITY: O(n) where n = input_size
 *
 * @return Spike-encoded input (caller must free)
 */
static float* convert_input_to_spikes(const float* input, uint32_t input_size, float threshold,
                                      spike_encoding_t encoding)
{
    // Phase MP: Allocate from pool for hot path performance
    float* spike_input = (float*)alloc_hot_buffer(input_size * sizeof(float));
    // Guard clause: Check allocation
    if (!spike_input) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "spike_input is NULL");

        return NULL;

    }

    // Guard clause: Use integer encoding only
    if (encoding == SPIKE_ENCODING_INTEGER) {
        for (uint32_t i = 0; i < input_size; i++) {
            spike_input[i] = adaptive_value_to_spikes(input[i], threshold);
        }
        return spike_input;
    }

    // Otherwise, copy directly
    memcpy(spike_input, input, input_size * sizeof(float));
    return spike_input;
}

/**
 * @brief Clamps threshold within valid range
 *
 * WHY: Extracted clamping logic. Ensures invariant: threshold in [min, max]
 *
 * COMPLEXITY: O(1)
 *
 * @return Clamped threshold value
 */
static float clamp_threshold(float threshold, float min_threshold, float max_threshold)
{
    if (threshold < min_threshold)
        return min_threshold;
    if (threshold > max_threshold)
        return max_threshold;
    return threshold;
}

/**
 * @brief Adapts neuron threshold based on sparsity target
 *
 * WHY: Extracted adaptation logic. Single responsibility - only adjusts threshold.
 *
 * COMPLEXITY: O(1)
 *
 * @param state - Neuron state to adapt
 * @param current_sparsity - Network's current sparsity
 * @param target_sparsity - Desired sparsity level
 * @param min_threshold - Minimum allowed threshold
 * @param max_threshold - Maximum allowed threshold
 */
static void adapt_neuron_threshold(adaptive_neuron_state_t* state, float current_sparsity,
                                   float target_sparsity, float min_threshold, float max_threshold)
{
    // Guard clause: Validate state
    if (!state)
        return;

    // Adjust threshold based on sparsity
    if (current_sparsity < target_sparsity) {
        // Too many neurons firing, increase threshold
        state->adaptive_threshold *= SPARSITY_ADAPT_INCREASE;
    } else {
        // Too few neurons firing, decrease threshold
        state->adaptive_threshold *= SPARSITY_ADAPT_DECREASE;
    }

    // Clamp to valid range
    state->adaptive_threshold =
        clamp_threshold(state->adaptive_threshold, min_threshold, max_threshold);
}

/**
 * @brief Processes single neuron output with thresholding
 *
 * WHY: Extracted neuron processing logic. Eliminates nested if statements.
 * Single responsibility - processes one neuron.
 *
 * COMPLEXITY: O(1)
 *
 * @param state - Neuron state
 * @param output_value - Neuron output value (modified in place)
 * @param enable_sparsity - Whether to enforce sparsity
 * @return true if neuron is active, false otherwise
 */
static bool process_neuron_output(adaptive_neuron_state_t* state, float* output_value,
                                  bool enable_sparsity)
{
    // Guard clause: Validate inputs
    if (!state || !output_value) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "process_neuron_output: required parameter is NULL (state, output_value)");
        return false;
    }

    bool is_active = (fabsf(*output_value) > state->adaptive_threshold);

    // Apply sparsity if enabled
    if (enable_sparsity && !is_active) {
        *output_value = 0.0F;
        return false;
    }

    // Update spike count
    state->spike_count = adaptive_value_to_spikes(*output_value, state->adaptive_threshold);

    return is_active;
}

/**
 * @brief Processes all neuron outputs and applies adaptive thresholding
 *
 * WHY: Extracted output processing loop. No nested conditions - uses helper functions.
 * Single pass through neurons maintains O(n) complexity.
 *
 * COMPLEXITY: O(n) where n = output_size
 *
 * @return Number of active neurons
 */
static uint32_t process_network_outputs(adaptive_network_t network, float* output,
                                        uint32_t output_size)
{
    // Guard clause: Validate inputs
    if (!network || !output)
        return 0;

    uint32_t active_count = 0;
    const adaptive_spike_params_t* params = &network->config.spike_params;

    // Single pass through neurons - O(n)
    uint32_t num_to_process =
        (output_size < network->num_neurons) ? output_size : network->num_neurons;

    for (uint32_t i = 0; i < num_to_process; i++) {
        adaptive_neuron_state_t* state = &network->neuron_states[i];

        // Update statistics (O(1))
        update_statistics(state, output[i]);

        // Adapt threshold if enabled (O(1))
        if (params->enable_adaptation) {
            adapt_neuron_threshold(state, network->running_sparsity, params->sparsity_target,
                                   params->min_threshold, params->max_threshold);
        }

        // Process neuron output (O(1))
        bool is_active = process_neuron_output(state, &output[i], network->config.enable_sparsity);

        if (is_active) {
            active_count++;
        }
    }

    return active_count;
}

/**
 * @brief Updates network's running sparsity estimate
 *
 * WHY: Extracted sparsity update. Uses exponential moving average for stability.
 *
 * COMPLEXITY: O(1)
 */
static void update_running_sparsity(adaptive_network_t network, uint32_t active_count,
                                    uint32_t output_size)
{
    // Guard clause: Validate inputs
    if (!network || output_size == 0)
        return;

    float current_sparsity = 1.0F - ((float) active_count / output_size);
    network->running_sparsity = (1.0F - SPARSITY_EMA_WEIGHT) * network->running_sparsity +
                                SPARSITY_EMA_WEIGHT * current_sparsity;
}

//=============================================================================
// Forward Pass with Adaptive Spiking
//=============================================================================

/**
 * @brief Processes input through adaptive network (main forward pass)
 *
 * WHY: Refactored from 90+ line function with nested ifs to clean orchestration.
 * Each step is a single function call with clear purpose. No nested loops or
 * nested ifs - only guard clauses.
 *
 * ALGORITHM:
 * 1. Validate inputs (O(1))
 * 2. Compute adaptive threshold for input (O(n))
 * 3. Convert input to spikes if needed (O(n))
 * 4. Forward pass through base network (O(n))
 * 5. Process outputs with adaptive thresholding (O(n))
 * 6. Update running sparsity statistics (O(1))
 *
 * COMPLEXITY: O(n) where n = max(input_size, output_size)
 *             Single pass through data - cannot be improved
 *
 * @return Number of active neurons (for sparsity tracking)
 */
uint32_t adaptive_network_forward(adaptive_network_t network, const float* input,
                                  uint32_t input_size, float* output, uint32_t output_size,
                                  uint64_t timestamp)
{
    // Guard clause: Validate inputs
    if (!network || !input || !output)
        return 0;

    // Phase Inferentia: NeuronCore-accelerated forward pass (highest priority for inference)
    if (network->neuron_enabled && network->neuron_cache &&
        nimcp_neuron_is_ready(network->neuron_cache)) {
        int result = nimcp_neuron_forward_pass(network->neuron_cache,
                                                input, input_size,
                                                output, output_size);
        if (result == 0) {
            // NeuronCore succeeded — adaptive thresholding + sparsity on CPU
            uint32_t active_count = process_network_outputs(network, output, output_size);
            update_running_sparsity(network, active_count, output_size);
            network->total_inferences++;
            return active_count;
        }
        // NeuronCore failed — fall through to GPU or CPU path
    }

    // Phase GPU: GPU-accelerated forward pass
    if (network->gpu_enabled && network->gpu_weight_cache) {
        // Re-upload weights if CPU biological learning modified them
        if (network->gpu_weight_cache->weights_dirty_on_cpu) {
            nimcp_gpu_weight_cache_upload(network->gpu_weight_cache,
                                         network->base_network);
        }

        // Spike encoding with FIXED threshold to preserve magnitude discrimination.
        // The adaptive threshold (mean_abs / k_factor) normalizes per-input, making
        // uniform inputs like [0.9]*N and [0.1]*N produce identical spike patterns.
        // A fixed threshold preserves magnitude: 0.9/0.1=9 spikes vs 0.1/0.1=1 spike.
        float fixed_threshold = network->config.spike_params.min_threshold;
        if (fixed_threshold <= 0.0f) fixed_threshold = 0.1f;  // safe default

        float* spike_input = convert_input_to_spikes(input, input_size, fixed_threshold,
                                                     network->config.spike_params.encoding);
        if (!spike_input) return 0;

        nimcp_gpu_forward_pass(network->gpu_weight_cache,
                               spike_input, input_size, output, output_size);

        free_hot_buffer(spike_input);

        // Adaptive thresholding + sparsity tracking on CPU
        uint32_t active_count = process_network_outputs(network, output, output_size);
        update_running_sparsity(network, active_count, output_size);
        network->total_inferences++;
        return active_count;
    }

    // CPU fallback path (original code)

    // Step 1: Compute adaptive threshold for input
    float input_threshold =
        adaptive_compute_threshold(input, input_size, network->config.spike_params.k_factor);

    // Step 2: Convert input to spikes if needed
    float* spike_input = convert_input_to_spikes(input, input_size, input_threshold,
                                                 network->config.spike_params.encoding);

    // Guard clause: Check spike input allocation
    if (!spike_input)
        return 0;

    // Step 3: Forward pass through base network
    neural_network_forward(network->base_network, spike_input, input_size, output, output_size);

    free_hot_buffer(spike_input);  // Phase MP: Return to pool

    // Step 4: Process outputs with adaptive thresholding
    uint32_t active_count = process_network_outputs(network, output, output_size);

    // Step 5: Update running sparsity
    update_running_sparsity(network, active_count, output_size);

    // Update statistics
    network->total_inferences++;

    return active_count;
}

/**
 * @brief Read-only forward pass (Phase 3: COW-safe inference)
 *
 * WHAT: Identical to adaptive_network_forward() but without statistics updates
 * WHY:  Enables multiple brains to share network indefinitely during inference
 * HOW:  Skips update_running_sparsity() and total_inferences++
 *
 * PERFORMANCE: Same as forward() - O(s*n) where s=sparsity, n=neurons
 * THREAD SAFETY: Read-only - safe for concurrent access
 *
 * @param network Adaptive network (const - not modified)
 * @param input Input vector
 * @param input_size Input dimension
 * @param output Output buffer
 * @param output_size Output dimension
 * @param timestamp Current time (unused in read-only mode)
 * @return Number of active neurons
 */
uint32_t adaptive_network_forward_readonly(const adaptive_network_t network, const float* input,
                                           uint32_t input_size, float* output, uint32_t output_size,
                                           uint64_t timestamp)
{
    // Guard clause: Validate inputs
    if (!network || !input || !output)
        return 0;

    // Phase Inferentia: NeuronCore-accelerated forward pass (read-only — no statistics update)
    if (network->neuron_enabled && network->neuron_cache &&
        nimcp_neuron_is_ready(network->neuron_cache)) {
        int result = nimcp_neuron_forward_pass(network->neuron_cache,
                                                input, input_size,
                                                output, output_size);
        if (result == 0) {
            uint32_t active_count = process_network_outputs((adaptive_network_t)network, output, output_size);
            return active_count;
        }
        // NeuronCore failed — fall through to GPU or CPU
    }

    // Phase GPU: GPU-accelerated forward pass (read-only — no statistics update)
    if (network->gpu_enabled && network->gpu_weight_cache) {
        if (network->gpu_weight_cache->weights_dirty_on_cpu) {
            nimcp_gpu_weight_cache_upload(network->gpu_weight_cache,
                                         network->base_network);
        }
        nimcp_gpu_forward_pass(network->gpu_weight_cache,
                               input, input_size, output, output_size);
        uint32_t active_count = process_network_outputs((adaptive_network_t)network, output, output_size);
        return active_count;
    }

    // CPU fallback path

    // Step 1: Compute adaptive threshold for input
    float input_threshold =
        adaptive_compute_threshold(input, input_size, network->config.spike_params.k_factor);

    // Step 2: Convert input to spikes if needed
    float* spike_input = convert_input_to_spikes(input, input_size, input_threshold,
                                                 network->config.spike_params.encoding);

    // Guard clause: Check spike input allocation
    if (!spike_input)
        return 0;

    // Step 3: Forward pass through base network
    // Cast away const - neural_network_forward doesn't actually modify network
    // (it only reads weights/structure, writes to output buffer)
    neural_network_forward(network->base_network, spike_input, input_size, output, output_size);

    free_hot_buffer(spike_input);  // Phase MP: Return to pool

    // Step 4: Process outputs with adaptive thresholding
    // Cast away const - process_network_outputs only reads network config
    uint32_t active_count = process_network_outputs((adaptive_network_t)network, output, output_size);

    // Phase 3: SKIP statistics updates for read-only inference
    // REMOVED: update_running_sparsity(network, active_count, output_size);
    // REMOVED: network->total_inferences++;

    return active_count;
}

/**
 * @brief Raw forward pass — spike encoding + network forward, NO output thresholding
 *
 * WHAT: Same as adaptive_network_forward() but skips process_network_outputs()
 * WHY:  predict_fast needs raw network output for accurate argmax classification.
 *       The adaptive thresholding zeros out below-threshold outputs which collapses
 *       all predictions to the same class.
 * HOW:  Steps 1-3 of adaptive_network_forward(), skip step 4-5
 */
uint32_t adaptive_network_forward_raw(adaptive_network_t network, const float* input,
                                      uint32_t input_size, float* output, uint32_t output_size)
{
    if (!network || !input || !output)
        return 0;

    // Spike encoding (must match learning forward pass)
    float input_threshold =
        adaptive_compute_threshold(input, input_size, network->config.spike_params.k_factor);
    float* spike_input = convert_input_to_spikes(input, input_size, input_threshold,
                                                 network->config.spike_params.encoding);
    if (!spike_input)
        return 0;

    // Forward pass through base network
    neural_network_forward(network->base_network, spike_input, input_size, output, output_size);
    free_hot_buffer(spike_input);

    // Count non-zero outputs (informational, no thresholding applied)
    uint32_t active = 0;
    for (uint32_t i = 0; i < output_size; i++) {
        if (fabsf(output[i]) > 1e-6f) active++;
    }
    return active;
}

//=============================================================================
// Sparsity and Pruning
//=============================================================================

float adaptive_network_get_sparsity(adaptive_network_t network)
{
    if (!network)
        return 0.0F;
    return network->running_sparsity;
}

uint32_t adaptive_network_prune(adaptive_network_t network, float threshold)
{
    if (!network)
        return 0;

    // Prune weak connections in base network
    // This would require exposing weight pruning in neural_network API
    // For now, return 0 and mark for future implementation
    network->num_pruned_synapses = 0;

    return network->num_pruned_synapses;
}

//=============================================================================
// Pattern Learning
//=============================================================================

float adaptive_network_learn(adaptive_network_t network, const training_example_t* example,
                             learning_mode_t mode, float learning_rate)
{
    if (!network || !example)
        return -1.0F;

    // Frozen network rejects learning
    if (network->frozen)
        return 0.0F;

    // Phase GPU: GPU-accelerated learning path
    if (network->gpu_enabled && network->gpu_weight_cache &&
        (mode == LEARN_MODE_SUPERVISED || mode == LEARN_MODE_DISTILLATION || mode == LEARN_MODE_HYBRID)) {
        // 1. Sync weights to GPU if CPU biological learning modified them
        if (network->gpu_weight_cache->weights_dirty_on_cpu) {
            nimcp_gpu_weight_cache_upload(network->gpu_weight_cache,
                                         network->base_network);
        }

        // 2. Spike-encode input with fixed threshold (preserves magnitude discrimination)
        float fixed_threshold = network->config.spike_params.min_threshold;
        if (fixed_threshold <= 0.0f) fixed_threshold = 0.1f;

        float* spike_input = convert_input_to_spikes(example->input, example->input_size,
                                                     fixed_threshold,
                                                     network->config.spike_params.encoding);
        if (!spike_input) return -1.0F;

        // 3. GPU forward pass with spike-encoded input
        float* output = (float*)alloc_hot_buffer(example->target_size * sizeof(float));
        if (!output) { free_hot_buffer(spike_input); return -1.0F; }

        nimcp_gpu_forward_pass(network->gpu_weight_cache,
                               spike_input, example->input_size,
                               output, example->target_size);

        free_hot_buffer(spike_input);

        // 3. GPU MSE loss computation
        float loss = nimcp_gpu_compute_loss(network->gpu_weight_cache,
                                            output, example->target,
                                            example->target_size);
        if (loss < 0.0F) loss = 0.0F;  // Fallback on GPU error

        // 4. Download activations so CPU bio-plasticity can read neuron states
        nimcp_gpu_weight_cache_sync_activations(network->gpu_weight_cache,
                                                network->base_network);

        // 4.5 Delta rule for output layer + hidden layer backprop
        {
            uint32_t num_layers = network->config.base_config.num_layers;
            uint32_t* layer_sizes = network->config.base_config.layer_sizes;
            if (num_layers >= 2 && layer_sizes) {
                // Compute output layer offset
                uint32_t output_offset = 0;
                for (uint32_t l = 0; l < num_layers - 1; l++)
                    output_offset += layer_sizes[l];
                uint32_t output_size = layer_sizes[num_layers - 1];

                // Output layer learning rate (boosted for classification)
                float output_lr = learning_rate * 10.0f;

                for (uint32_t j = 0; j < output_size && j < example->target_size; j++) {
                    float error_j = example->target[j] - output[j];
                    float sig_deriv = output[j] * (1.0f - output[j]);
                    if (sig_deriv < 0.01f) sig_deriv = 0.01f;

                    neuron_t* out_n = neural_network_get_neuron(
                        network->base_network, output_offset + j);
                    if (!out_n) continue;

                    for (uint32_t k = 0; k < NEURON_IN_COUNT(out_n); k++) {
                        synapse_handle_t* in_syn = NEURON_IN_HANDLE(out_n, k);
                        if (!in_syn) continue;
                        neuron_t* src = neural_network_get_neuron(
                            network->base_network, in_syn->target_neuron_id);
                        if (!src) continue;

                        float delta = output_lr * error_j * src->state * sig_deriv;
                        if (delta > 0.1f) delta = 0.1f;
                        if (delta < -0.1f) delta = -0.1f;
                        in_syn->weight += delta;

                        // Update outgoing synapse copy
                        for (uint32_t s = 0; s < NEURON_OUT_COUNT(src); s++) {
                            synapse_handle_t* out_syn = NEURON_OUT_HANDLE(src, s);
                            if (out_syn && out_syn->target_neuron_id == output_offset + j) {
                                out_syn->weight += delta;
                                break;
                            }
                        }
                    }

                    out_n->bias += output_lr * error_j * sig_deriv;
                }
            }
        }

        // 6. Mark weights dirty (learning modified synapse weights on CPU)
        network->gpu_weight_cache->weights_dirty_on_cpu = true;

        // 7. For HYBRID mode: apply biological plasticity after GPU backprop
        if (mode == LEARN_MODE_HYBRID && network->base_network) {
            float reward = fmaxf(0.0f, 1.0f - loss);
            neural_network_apply_reward_learning(
                network->base_network, reward, learning_rate * 0.1f,
                network->total_learning_steps);
        }

        free_hot_buffer(output);
        network->total_learning_steps++;
        return loss;
    }

    // CPU fallback path (original code)

    // Forward pass to get current output (Phase MP: use pool)
    float* output = (float*)alloc_hot_buffer(example->target_size * sizeof(float));
    if (!output) {
        return -1.0F;
    }
    adaptive_network_forward(network, example->input, example->input_size, output,
                             example->target_size, 0);

    // Compute loss based on mode
    float loss = 0.0F;

    switch (mode) {
        case LEARN_MODE_SUPERVISED:
        case LEARN_MODE_DISTILLATION:
            // Mean squared error
            for (uint32_t i = 0; i < example->target_size; i++) {
                float error = output[i] - example->target[i];
                loss += error * error;
            }
            loss /= example->target_size;

            // =================================================================
            // BIOLOGICAL PLASTICITY: Apply STDP/BCM to synapses
            // =================================================================
            // WHAT: Update synaptic weights using biological learning rules
            // WHY:  Enable Hebbian learning instead of gradient descent
            // HOW:  Compute reward signal → Call neural_network_apply_reward_learning
            //
            // BIOLOGICAL BASIS:
            // - STDP: Hebbian learning ("neurons that fire together wire together")
            // - BCM: Homeostatic plasticity (sliding threshold prevents saturation)
            // - Eligibility traces: Temporal credit assignment (bridge reward delay)
            //
            // COMPLEXITY: O(N×S) where N = neurons, S = synapses per neuron
            // =================================================================

            // Full backpropagation through all layers (sparse-efficient)
            // WHY: Output-layer-only delta rule can't learn non-linear features.
            //      Backprop through hidden layers enables feature learning.
            // HOW: Standard backprop adapted for sparse connectivity:
            //      - Only neurons with active synapses get updated
            //      - O(synapses) per layer, not O(neurons²)
            {
                float grad_norm_sq = 0.0f;  // Accumulate squared gradients for L2 norm
                uint32_t num_layers = network->config.base_config.num_layers;
                uint32_t* layer_sizes = network->config.base_config.layer_sizes;
                if (num_layers >= 2 && layer_sizes) {
                    // Compute layer offsets
                    uint32_t layer_offsets[num_layers];
                    layer_offsets[0] = 0;
                    for (uint32_t l = 1; l < num_layers; l++)
                        layer_offsets[l] = layer_offsets[l-1] + layer_sizes[l-1];

                    uint32_t output_size = layer_sizes[num_layers - 1];
                    uint32_t output_offset = layer_offsets[num_layers - 1];

                    // Allocate delta buffers for current and next layer
                    // (reuse hot buffer pool to avoid malloc per learn call)
                    uint32_t max_layer = 0;
                    for (uint32_t l = 0; l < num_layers; l++)
                        if (layer_sizes[l] > max_layer) max_layer = layer_sizes[l];

                    float* delta_cur = (float*)alloc_hot_buffer(max_layer * sizeof(float));
                    float* delta_prev = (float*)alloc_hot_buffer(max_layer * sizeof(float));
                    if (!delta_cur || !delta_prev) {
                        if (delta_cur) free_hot_buffer(delta_cur);
                        if (delta_prev) free_hot_buffer(delta_prev);
                        break;
                    }

                    // --- Output layer deltas ---
                    for (uint32_t j = 0; j < output_size && j < example->target_size; j++) {
                        float error_j = example->target[j] - output[j];
                        float sig_deriv = output[j] * (1.0f - output[j]);
                        if (sig_deriv < 0.01f) sig_deriv = 0.01f;
                        delta_cur[j] = error_j * sig_deriv;
                    }

                    // --- Backpropagate layer by layer (output → first hidden) ---
                    for (int32_t layer = (int32_t)num_layers - 1; layer >= 1; layer--) {
                        uint32_t cur_offset = layer_offsets[layer];
                        uint32_t cur_size = layer_sizes[layer];
                        uint32_t prev_size = layer_sizes[layer - 1];

                        // Scale learning rate by 1/sqrt(fan_in) to prevent saturation
                        // with high-dimensional spike-encoded inputs
                        float fan_in = (float)prev_size;
                        float layer_lr = learning_rate / sqrtf(fmaxf(fan_in, 1.0f));

                        // Zero prev-layer deltas (will accumulate from backprop)
                        memset(delta_prev, 0, prev_size * sizeof(float));

                        // Update weights and propagate deltas
                        for (uint32_t j = 0; j < cur_size; j++) {
                            if (layer == (int32_t)num_layers - 1 && j >= example->target_size)
                                break;

                            neuron_t* cur_n = neural_network_get_neuron(
                                network->base_network, cur_offset + j);
                            if (!cur_n) continue;

                            // Gradient clipping: prevent exploding gradients
                            float dj = delta_cur[j];
                            if (dj > 1.0f) dj = 1.0f;
                            if (dj < -1.0f) dj = -1.0f;

                            // Update incoming synapses + propagate delta to prev layer
                            for (uint32_t k = 0; k < NEURON_IN_COUNT(cur_n); k++) {
                                synapse_handle_t* in_syn = NEURON_IN_HANDLE(cur_n, k);
                                if (!in_syn) continue;

                                uint32_t src_id = in_syn->target_neuron_id;
                                neuron_t* src = neural_network_get_neuron(
                                    network->base_network, src_id);
                                if (!src) continue;

                                // Weight update: Δw = lr * delta_j * activation_i
                                float weight_delta = layer_lr * dj * src->state;

                                // Clip weight delta to prevent weight explosion
                                if (weight_delta > 0.1f) weight_delta = 0.1f;
                                if (weight_delta < -0.1f) weight_delta = -0.1f;

                                // Accumulate squared gradient for L2 norm
                                grad_norm_sq += weight_delta * weight_delta;

                                in_syn->weight += weight_delta;

                                // Update outgoing synapse copy
                                for (uint32_t s = 0; s < NEURON_OUT_COUNT(src); s++) {
                                    synapse_handle_t* out_syn = NEURON_OUT_HANDLE(src, s);
                                    if (out_syn && out_syn->target_neuron_id == cur_offset + j) {
                                        out_syn->weight += weight_delta;
                                        break;
                                    }
                                }

                                // Accumulate delta for prev layer neuron
                                uint32_t prev_offset = layer_offsets[layer - 1];
                                if (src_id >= prev_offset && src_id < prev_offset + prev_size) {
                                    uint32_t prev_idx = src_id - prev_offset;
                                    // Backprop: delta_i += weight * delta_j
                                    delta_prev[prev_idx] += in_syn->weight * dj;
                                }
                            }

                            // Bias update (also accumulate for gradient norm)
                            float bias_delta = layer_lr * dj;
                            grad_norm_sq += bias_delta * bias_delta;
                            cur_n->bias += bias_delta;
                        }

                        // Apply activation derivative to prev-layer deltas
                        if (layer > 1) {  // Don't backprop into input layer
                            uint32_t prev_offset = layer_offsets[layer - 1];
                            for (uint32_t i = 0; i < prev_size; i++) {
                                neuron_t* prev_n = neural_network_get_neuron(
                                    network->base_network, prev_offset + i);
                                if (!prev_n) continue;
                                float s = prev_n->state;
                                // Activation derivative must match the neuron's actual function
                                float act_deriv;
                                switch (prev_n->activation_type) {
                                    case ACTIVATION_RELU:
                                        act_deriv = (s > 0.0f) ? 1.0f : 0.0f;
                                        break;
                                    case ACTIVATION_LEAKY_RELU:
                                        act_deriv = (s > 0.0f) ? 1.0f : 0.01f;
                                        break;
                                    case ACTIVATION_TANH:
                                        act_deriv = 1.0f - s * s;
                                        break;
                                    case ACTIVATION_SIGMOID:
                                        act_deriv = s * (1.0f - s);
                                        if (act_deriv < 0.01f) act_deriv = 0.01f;
                                        break;
                                    default:  // ADAPTIVE and others
                                        act_deriv = (s > 0.0f) ? 1.0f : 0.01f;
                                        break;
                                }
                                delta_prev[i] *= act_deriv;
                            }

                            // Swap buffers for next iteration
                            float* tmp = delta_cur;
                            delta_cur = delta_prev;
                            delta_prev = tmp;
                        }
                    }

                    free_hot_buffer(delta_cur);
                    free_hot_buffer(delta_prev);
                }
                // Store gradient L2 norm on the network
                network->last_grad_norm = sqrtf(grad_norm_sq);
            }
            break;

        case LEARN_MODE_UNSUPERVISED:
            // Hebbian learning - no explicit target
            // Already handled by base network's plasticity rules during forward pass
            loss = 0.0F;
            break;

        case LEARN_MODE_REINFORCEMENT:
            // Use confidence as reward signal
            loss = 1.0F - example->confidence;

            // Apply reinforcement learning with eligibility traces
            // Reward comes directly from confidence
            {
            neural_network_t base_rl = network->base_network;
            uint64_t time_rl = network->total_learning_steps;
            float rl_reward = example->confidence;  // Direct reward signal

            neural_network_apply_reward_learning(base_rl, rl_reward, learning_rate, time_rl);
            }
            break;

        case LEARN_MODE_HYBRID:
            // =================================================================
            // HYBRID LEARNING: Backpropagation + Biological Plasticity
            // =================================================================
            // WHAT: Supervised backprop followed by biological learning rules
            // WHY:  Backprop provides fast gradient-based learning, while
            //       biological plasticity (STDP/BCM/eligibility) adds
            //       homeostatic regulation and temporal credit assignment
            // HOW:  Phase 1 = MSE loss + full backprop (same as supervised)
            //       Phase 2 = reward-modulated biological learning (STDP/BCM)
            //
            // BIOLOGICAL BASIS:
            // - Cortical circuits use both error-driven and Hebbian learning
            // - Dopamine-modulated STDP bridges reward with spike timing
            // - BCM prevents weight saturation via sliding threshold
            // =================================================================

            // Phase 1: MSE loss computation (same as supervised)
            for (uint32_t i = 0; i < example->target_size; i++) {
                float error = output[i] - example->target[i];
                loss += error * error;
            }
            loss /= example->target_size;

            // Phase 2: Full backpropagation through all layers (same as supervised)
            {
                float grad_norm_sq = 0.0f;  // Accumulate squared gradients for L2 norm
                uint32_t num_layers = network->config.base_config.num_layers;
                uint32_t* layer_sizes = network->config.base_config.layer_sizes;
                if (num_layers >= 2 && layer_sizes) {
                    // Compute layer offsets
                    uint32_t layer_offsets[num_layers];
                    layer_offsets[0] = 0;
                    for (uint32_t l = 1; l < num_layers; l++)
                        layer_offsets[l] = layer_offsets[l-1] + layer_sizes[l-1];

                    uint32_t output_size = layer_sizes[num_layers - 1];
                    uint32_t output_offset = layer_offsets[num_layers - 1];

                    // Allocate delta buffers
                    uint32_t max_layer = 0;
                    for (uint32_t l = 0; l < num_layers; l++)
                        if (layer_sizes[l] > max_layer) max_layer = layer_sizes[l];

                    float* delta_cur = (float*)alloc_hot_buffer(max_layer * sizeof(float));
                    float* delta_prev = (float*)alloc_hot_buffer(max_layer * sizeof(float));
                    if (!delta_cur || !delta_prev) {
                        if (delta_cur) free_hot_buffer(delta_cur);
                        if (delta_prev) free_hot_buffer(delta_prev);
                        break;
                    }

                    // --- Output layer deltas ---
                    for (uint32_t j = 0; j < output_size && j < example->target_size; j++) {
                        float error_j = example->target[j] - output[j];
                        float sig_deriv = output[j] * (1.0f - output[j]);
                        if (sig_deriv < 0.01f) sig_deriv = 0.01f;
                        delta_cur[j] = error_j * sig_deriv;
                    }

                    // --- Backpropagate layer by layer (output -> first hidden) ---
                    for (int32_t layer = (int32_t)num_layers - 1; layer >= 1; layer--) {
                        uint32_t cur_offset = layer_offsets[layer];
                        uint32_t cur_size = layer_sizes[layer];
                        uint32_t prev_size = layer_sizes[layer - 1];

                        float fan_in = (float)prev_size;
                        float layer_lr = learning_rate / sqrtf(fmaxf(fan_in, 1.0f));

                        memset(delta_prev, 0, prev_size * sizeof(float));

                        for (uint32_t j = 0; j < cur_size; j++) {
                            if (layer == (int32_t)num_layers - 1 && j >= example->target_size)
                                break;

                            neuron_t* cur_n = neural_network_get_neuron(
                                network->base_network, cur_offset + j);
                            if (!cur_n) continue;

                            float dj = delta_cur[j];
                            if (dj > 1.0f) dj = 1.0f;
                            if (dj < -1.0f) dj = -1.0f;

                            for (uint32_t k = 0; k < NEURON_IN_COUNT(cur_n); k++) {
                                synapse_handle_t* in_syn = NEURON_IN_HANDLE(cur_n, k);
                                if (!in_syn) continue;

                                uint32_t src_id = in_syn->target_neuron_id;
                                neuron_t* src = neural_network_get_neuron(
                                    network->base_network, src_id);
                                if (!src) continue;

                                float weight_delta = layer_lr * dj * src->state;
                                if (weight_delta > 0.1f) weight_delta = 0.1f;
                                if (weight_delta < -0.1f) weight_delta = -0.1f;

                                // Accumulate squared gradient for L2 norm
                                grad_norm_sq += weight_delta * weight_delta;

                                in_syn->weight += weight_delta;

                                for (uint32_t s = 0; s < NEURON_OUT_COUNT(src); s++) {
                                    synapse_handle_t* out_syn = NEURON_OUT_HANDLE(src, s);
                                    if (out_syn && out_syn->target_neuron_id == cur_offset + j) {
                                        out_syn->weight += weight_delta;
                                        break;
                                    }
                                }

                                uint32_t prev_offset = layer_offsets[layer - 1];
                                if (src_id >= prev_offset && src_id < prev_offset + prev_size) {
                                    uint32_t prev_idx = src_id - prev_offset;
                                    delta_prev[prev_idx] += in_syn->weight * dj;
                                }
                            }

                            // Bias update (also accumulate for gradient norm)
                            float bias_delta = layer_lr * dj;
                            grad_norm_sq += bias_delta * bias_delta;
                            cur_n->bias += bias_delta;
                        }

                        if (layer > 1) {
                            uint32_t prev_offset = layer_offsets[layer - 1];
                            for (uint32_t i = 0; i < prev_size; i++) {
                                neuron_t* prev_n = neural_network_get_neuron(
                                    network->base_network, prev_offset + i);
                                if (!prev_n) continue;
                                float s = prev_n->state;
                                float act_deriv;
                                switch (prev_n->activation_type) {
                                    case ACTIVATION_RELU:
                                        act_deriv = (s > 0.0f) ? 1.0f : 0.0f;
                                        break;
                                    case ACTIVATION_LEAKY_RELU:
                                        act_deriv = (s > 0.0f) ? 1.0f : 0.01f;
                                        break;
                                    case ACTIVATION_TANH:
                                        act_deriv = 1.0f - s * s;
                                        break;
                                    case ACTIVATION_SIGMOID:
                                        act_deriv = s * (1.0f - s);
                                        if (act_deriv < 0.01f) act_deriv = 0.01f;
                                        break;
                                    default:
                                        act_deriv = (s > 0.0f) ? 1.0f : 0.01f;
                                        break;
                                }
                                delta_prev[i] *= act_deriv;
                            }

                            float* tmp = delta_cur;
                            delta_cur = delta_prev;
                            delta_prev = tmp;
                        }
                    }

                    free_hot_buffer(delta_cur);
                    free_hot_buffer(delta_prev);

                    // Phase 3: Biological plasticity AFTER backprop
                    // Convert loss to reward: low loss = high reward = strong LTP
                    float reward = fmaxf(0.0f, 1.0f - loss);
                    neural_network_t base_hybrid = network->base_network;
                    if (base_hybrid) {
                        // Use 10% of learning rate for biological rules to prevent
                        // bio-plasticity from overwhelming backprop gradients
                        neural_network_apply_reward_learning(
                            base_hybrid, reward, learning_rate * 0.1f,
                            network->total_learning_steps);
                    }

                    // Phase 4: Lateral inhibition on output layer for sharper classification
                    if (network->base_network) {
                        neural_network_apply_lateral_inhibition(
                            network->base_network,
                            output_offset, output_size,
                            0.3f);  // Moderate inhibition strength
                    }
                }
                // Store gradient L2 norm on the network
                network->last_grad_norm = sqrtf(grad_norm_sq);
            }
            break;
    }

    free_hot_buffer(output);  // Phase MP: Return to pool

    network->total_learning_steps++;

    return loss;
}

float adaptive_network_learn_batch(adaptive_network_t network, const training_example_t* examples,
                                   uint32_t num_examples, learning_mode_t mode, float learning_rate)
{
    if (!network || !examples || num_examples == 0)
        return -1.0F;

    float total_loss = 0.0F;

    for (uint32_t i = 0; i < num_examples; i++) {
        float loss = adaptive_network_learn(network, &examples[i], mode, learning_rate);
        total_loss += loss;
    }

    return total_loss / num_examples;
}

float adaptive_network_distill(adaptive_network_t network, const float* input, uint32_t input_size,
                               teacher_function_t teacher_fn, void* teacher_context,
                               float learning_rate)
{
    if (!network || !input || !teacher_fn)
        return -1.0F;

    // Query teacher for target
    float* teacher_output = teacher_fn(input, input_size, teacher_context);
    if (!teacher_output)
        return -1.0F;

    // Create training example from teacher's output
    training_example_t example = {.input = (float*) input,
                                  .input_size = input_size,
                                  .target = teacher_output,
                                  .target_size = network->config.base_config.output_size,
                                  .confidence = 1.0F,  // Assume high confidence in teacher
                                  .label = {0}};

    float loss = adaptive_network_learn(network, &example, LEARN_MODE_DISTILLATION, learning_rate);

    // Free teacher output (caller's responsibility to allocate, our responsibility to free)
    nimcp_free(teacher_output);

    return loss;
}

//=============================================================================
// Model Persistence
//=============================================================================

bool adaptive_network_save(adaptive_network_t network, const char* filepath,
                           serialize_format_t format)
{
    if (!network || !filepath) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "adaptive_network_save: required parameter is NULL (network, filepath)");
        return false;
    }

    // Open file for writing
    FILE* file = fopen(filepath, "wb");
    if (!file) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "adaptive_network_save: file is NULL");
        return false;
    }

    bool success = true;

    switch (format) {
        case SERIALIZE_FORMAT_BINARY:
            // Write magic header
            uint32_t magic = 0x4E494D43;    // "NIMC"
            uint32_t version = 0x00020500;  // v2.5.0

            FWRITE_CHECKED(&magic, sizeof(uint32_t), 1, file);
            FWRITE_CHECKED(&version, sizeof(uint32_t), 1, file);

            // WHAT: Write configuration (excluding pointer fields)
            // WHY:  Can't serialize pointers directly - they're invalid when loaded
            // HOW:  Write num_layers, then layer_sizes array data separately

            // Write base_config fields (safe ones)
            FWRITE_CHECKED(&network->config.base_config.num_neurons, sizeof(uint32_t), 1, file);
            FWRITE_CHECKED(&network->config.base_config.ei_ratio, sizeof(float), 1, file);
            FWRITE_CHECKED(&network->config.base_config.learning_rate, sizeof(float), 1, file);
            FWRITE_CHECKED(&network->config.base_config.hebbian_rate, sizeof(float), 1, file);
            FWRITE_CHECKED(&network->config.base_config.stdp_window, sizeof(float), 1, file);
            FWRITE_CHECKED(&network->config.base_config.homeostatic_rate, sizeof(float), 1, file);
            FWRITE_CHECKED(&network->config.base_config.target_activity, sizeof(float), 1, file);
            FWRITE_CHECKED(&network->config.base_config.adaptation_rate, sizeof(float), 1, file);
            FWRITE_CHECKED(&network->config.base_config.refractory_period, sizeof(float), 1, file);
            FWRITE_CHECKED(&network->config.base_config.min_weight, sizeof(float), 1, file);
            FWRITE_CHECKED(&network->config.base_config.max_weight, sizeof(float), 1, file);
            FWRITE_CHECKED(&network->config.base_config.update_interval, sizeof(uint32_t), 1, file);
            FWRITE_CHECKED(&network->config.base_config.input_size, sizeof(uint32_t), 1, file);
            FWRITE_CHECKED(&network->config.base_config.output_size, sizeof(uint32_t), 1, file);
            FWRITE_CHECKED(&network->config.base_config.num_layers, sizeof(uint32_t), 1, file);

            // Write layer_sizes array separately
            if (network->config.base_config.num_layers > 0 && network->config.base_config.layer_sizes) {
                FWRITE_CHECKED(network->config.base_config.layer_sizes,
                       sizeof(uint32_t),
                       network->config.base_config.num_layers,
                       file);
            }

            // Continue with base_config booleans and enums
            FWRITE_CHECKED(&network->config.base_config.enable_stdp, sizeof(bool), 1, file);
            FWRITE_CHECKED(&network->config.base_config.enable_hebbian, sizeof(bool), 1, file);
            FWRITE_CHECKED(&network->config.base_config.enable_oja, sizeof(bool), 1, file);
            FWRITE_CHECKED(&network->config.base_config.enable_homeostasis, sizeof(bool), 1, file);
            FWRITE_CHECKED(&network->config.base_config.neuron_model, sizeof(neuron_model_type_t), 1, file);
            // Skip model_params pointer (NULL in most cases, complex to serialize)
            FWRITE_CHECKED(&network->config.base_config.integration_method, sizeof(ode_integration_method_t), 1, file);
            FWRITE_CHECKED(&network->config.base_config.enable_bcm, sizeof(bool), 1, file);
            FWRITE_CHECKED(&network->config.base_config.enable_eligibility, sizeof(bool), 1, file);

            // Write spike_params struct
            FWRITE_CHECKED(&network->config.spike_params, sizeof(adaptive_spike_params_t), 1, file);

            // Write remaining adaptive config fields
            FWRITE_CHECKED(&network->config.enable_sparsity, sizeof(bool), 1, file);
            FWRITE_CHECKED(&network->config.pruning_threshold, sizeof(float), 1, file);
            FWRITE_CHECKED(&network->config.update_frequency, sizeof(uint32_t), 1, file);

            // Skip checkpoint_path pointer (don't need to save this)
            FWRITE_CHECKED(&network->config.auto_load, sizeof(bool), 1, file);
            FWRITE_CHECKED(&network->config.auto_save, sizeof(bool), 1, file);
            FWRITE_CHECKED(&network->config.auto_save_interval, sizeof(uint32_t), 1, file);

            // Write neuron count
            FWRITE_CHECKED(&network->num_neurons, sizeof(uint32_t), 1, file);

            // Write neuron states
            for (uint32_t i = 0; i < network->num_neurons; i++) {
                FWRITE_CHECKED(&network->neuron_states[i], sizeof(adaptive_neuron_state_t), 1, file);
            }

            // Write label map
            FWRITE_CHECKED(&network->num_labels, sizeof(uint32_t), 1, file);
            for (uint32_t i = 0; i < network->num_labels; i++) {
                uint32_t label_len = strlen(network->label_map[i]) + 1;
                FWRITE_CHECKED(&label_len, sizeof(uint32_t), 1, file);
                FWRITE_CHECKED(network->label_map[i], label_len, 1, file);
            }

            // Write statistics
            FWRITE_CHECKED(&network->total_inferences, sizeof(uint64_t), 1, file);
            FWRITE_CHECKED(&network->total_learning_steps, sizeof(uint64_t), 1, file);
            FWRITE_CHECKED(&network->running_sparsity, sizeof(float), 1, file);

            // Write synaptic weights from base_network (CRITICAL: enables weight persistence)
            if (network->base_network) {
                synapse_metadata_pool_t save_m_pool = neural_network_get_synapse_metadata_pool(network->base_network);
                uint32_t base_num_neurons = neural_network_get_num_neurons(network->base_network);
                FWRITE_CHECKED(&base_num_neurons, sizeof(uint32_t), 1, file);

                for (uint32_t i = 0; i < base_num_neurons; i++) {
                    neuron_t* neuron = neural_network_get_neuron(network->base_network, i);
                    if (!neuron) continue;

                    // Write number of synapses for this neuron
                    uint32_t num_syn = NEURON_OUT_COUNT(neuron);
                    FWRITE_CHECKED(&num_syn, sizeof(uint32_t), 1, file);

                    // Write each synapse (weight and key plasticity data)
                    for (uint32_t j = 0; j < num_syn; j++) {
                        synapse_handle_t* h = NEURON_OUT_HANDLE(neuron, j);
                        if (!h) continue;
                        synapse_t* syn = sparse_synapse_get_metadata(save_m_pool, h);
                        FWRITE_CHECKED(&h->target_neuron_id, sizeof(uint32_t), 1, file);
                        FWRITE_CHECKED(&h->weight, sizeof(float), 1, file);
                        float plasticity = syn ? syn->plasticity : 0.0f;
                        float trace = syn ? syn->trace : 0.0f;
                        float strength = h->strength;
                        float meta_plasticity = syn ? syn->meta_plasticity : 0.0f;
                        float last_change = syn ? syn->last_change : 0.0f;
                        uint64_t last_active = syn ? syn->last_active : 0;
                        bool enable_stp = syn ? syn->enable_stp : false;
                        FWRITE_CHECKED(&plasticity, sizeof(float), 1, file);
                        FWRITE_CHECKED(&trace, sizeof(float), 1, file);
                        FWRITE_CHECKED(&strength, sizeof(float), 1, file);
                        FWRITE_CHECKED(&meta_plasticity, sizeof(float), 1, file);
                        FWRITE_CHECKED(&last_change, sizeof(float), 1, file);
                        FWRITE_CHECKED(&last_active, sizeof(uint64_t), 1, file);
                        FWRITE_CHECKED(&enable_stp, sizeof(bool), 1, file);
                        if (enable_stp && syn) {
                            FWRITE_CHECKED(&syn->stp, sizeof(stp_state_t), 1, file);
                        }
                    }

                    // Write full neuron state (CRITICAL: enables exact training resume)
                    FWRITE_CHECKED(&neuron->state, sizeof(float), 1, file);
                    FWRITE_CHECKED(&neuron->bias, sizeof(float), 1, file);
                    FWRITE_CHECKED(&neuron->threshold, sizeof(float), 1, file);
                    FWRITE_CHECKED(&neuron->adaptation, sizeof(float), 1, file);
                    FWRITE_CHECKED(&neuron->calcium_concentration, sizeof(float), 1, file);
                    FWRITE_CHECKED(&neuron->plasticity_rate, sizeof(float), 1, file);
                    FWRITE_CHECKED(&neuron->homeostatic_factor, sizeof(float), 1, file);
                    FWRITE_CHECKED(&neuron->avg_activity, sizeof(float), 1, file);
                    FWRITE_CHECKED(&neuron->weight_norm, sizeof(float), 1, file);
                    FWRITE_CHECKED(&neuron->learning_rule, sizeof(learning_rule_t), 1, file);
                    FWRITE_CHECKED(&neuron->activation_type, sizeof(activation_type_t), 1, file);
                    FWRITE_CHECKED(&neuron->oja_params, sizeof(oja_params_t), 1, file);
                    FWRITE_CHECKED(&neuron->stdp_params, sizeof(stdp_params_t), 1, file);
                    FWRITE_CHECKED(&neuron->homeostatic, sizeof(homeostatic_params_t), 1, file);
                    FWRITE_CHECKED(&neuron->last_spike, sizeof(uint64_t), 1, file);
                    FWRITE_CHECKED(&neuron->last_update, sizeof(uint64_t), 1, file);
                    FWRITE_CHECKED(&neuron->model_type, sizeof(neuron_model_type_t), 1, file);
                    // Note: neuron.model (neuron_model_state_t) is a pointer to opaque state
                    // TODO: If model is not NULL, need to serialize model-specific state
                }
            } else {
                // No base network - write 0 neurons
                uint32_t zero = 0;
                FWRITE_CHECKED(&zero, sizeof(uint32_t), 1, file);
            }

            break;

        case SERIALIZE_FORMAT_JSON:
        case SERIALIZE_FORMAT_SAFETENSORS:
            // TODO: Implement JSON and SafeTensors formats
            success = false;
            break;
    }

    fclose(file);
    return success;
}

adaptive_network_t adaptive_network_load(const char* filepath)
{
    if (!filepath) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "filepath is NULL");

        return NULL;

    }

    FILE* file = fopen(filepath, "rb");
    if (!file) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "file is NULL");

        return NULL;

    }

    // Read magic header
    uint32_t magic = 0;
    uint32_t version = 0;
    if (fread(&magic, sizeof(uint32_t), 1, file) != 1 || magic != 0x4E494D43) {
        fclose(file);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "adaptive_network_load: validation failed");
        return NULL;
    }

    if (fread(&version, sizeof(uint32_t), 1, file) != 1) {
        fclose(file);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "adaptive_network_load: validation failed");
        return NULL;
    }

    // WHAT: Read configuration (excluding pointer fields)
    // WHY:  Pointers were serialized separately
    // HOW:  Read individual fields, then reconstruct layer_sizes array

    adaptive_network_config_t config;
    memset(&config, 0, sizeof(adaptive_network_config_t));

    // Read base_config fields
    // Note: Using (void) cast intentionally ignores return value for non-critical config fields
    // that have safe defaults. Critical fields use explicit error checking.
    if (fread(&config.base_config.num_neurons, sizeof(uint32_t), 1, file) != 1) {
        fclose(file);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "adaptive_network_load: validation failed");
        return NULL;
    }
    // Validate num_neurons: max 10M neurons to prevent corrupt data from causing crashes
    if (config.base_config.num_neurons > 10000000) {
        fclose(file);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM,
            "adaptive_network_load: num_neurons exceeds maximum (10M)");
        return NULL;
    }
    (void)fread(&config.base_config.ei_ratio, sizeof(float), 1, file);
    (void)fread(&config.base_config.learning_rate, sizeof(float), 1, file);
    (void)fread(&config.base_config.hebbian_rate, sizeof(float), 1, file);
    (void)fread(&config.base_config.stdp_window, sizeof(float), 1, file);
    (void)fread(&config.base_config.homeostatic_rate, sizeof(float), 1, file);
    (void)fread(&config.base_config.target_activity, sizeof(float), 1, file);
    (void)fread(&config.base_config.adaptation_rate, sizeof(float), 1, file);
    (void)fread(&config.base_config.refractory_period, sizeof(float), 1, file);
    (void)fread(&config.base_config.min_weight, sizeof(float), 1, file);
    (void)fread(&config.base_config.max_weight, sizeof(float), 1, file);
    (void)fread(&config.base_config.update_interval, sizeof(uint32_t), 1, file);
    (void)fread(&config.base_config.input_size, sizeof(uint32_t), 1, file);
    (void)fread(&config.base_config.output_size, sizeof(uint32_t), 1, file);
    (void)fread(&config.base_config.num_layers, sizeof(uint32_t), 1, file);

    // Validate dimensions: prevent corrupt data from causing crashes
    if (config.base_config.input_size > 1000000 ||
        config.base_config.output_size > 1000000) {
        fclose(file);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM,
            "adaptive_network_load: input/output size exceeds maximum (1M)");
        return NULL;
    }

    // Read layer_sizes array separately
    uint32_t* layer_sizes = NULL;
    // Validate num_layers to prevent corrupt data from causing massive allocations
    if (config.base_config.num_layers > 1024) {
        fclose(file);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM,
            "adaptive_network_load: num_layers exceeds maximum (1024)");
        return NULL;
    }
    if (config.base_config.num_layers > 0) {
        layer_sizes = nimcp_malloc(config.base_config.num_layers * sizeof(uint32_t));
        if (!layer_sizes) {
            fclose(file);
            NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "adaptive_network_load: layer_sizes is NULL");
            return NULL;
        }
        if (fread(layer_sizes, sizeof(uint32_t), config.base_config.num_layers, file) != config.base_config.num_layers) {
            nimcp_free(layer_sizes);
            fclose(file);
            NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "adaptive_network_load: validation failed");
            return NULL;
        }
        config.base_config.layer_sizes = layer_sizes;
    }

    // Read remaining base_config fields (non-critical, use (void) cast)
    (void)fread(&config.base_config.enable_stdp, sizeof(bool), 1, file);
    (void)fread(&config.base_config.enable_hebbian, sizeof(bool), 1, file);
    (void)fread(&config.base_config.enable_oja, sizeof(bool), 1, file);
    (void)fread(&config.base_config.enable_homeostasis, sizeof(bool), 1, file);
    (void)fread(&config.base_config.neuron_model, sizeof(neuron_model_type_t), 1, file);
    config.base_config.model_params = NULL;  // Not serialized
    (void)fread(&config.base_config.integration_method, sizeof(ode_integration_method_t), 1, file);
    (void)fread(&config.base_config.enable_bcm, sizeof(bool), 1, file);
    (void)fread(&config.base_config.enable_eligibility, sizeof(bool), 1, file);

    // Read spike_params (non-critical)
    (void)fread(&config.spike_params, sizeof(adaptive_spike_params_t), 1, file);

    // Read remaining adaptive config fields (non-critical)
    (void)fread(&config.enable_sparsity, sizeof(bool), 1, file);
    (void)fread(&config.pruning_threshold, sizeof(float), 1, file);
    (void)fread(&config.update_frequency, sizeof(uint32_t), 1, file);

    config.checkpoint_path = NULL;  // Not serialized, will be NULL
    (void)fread(&config.auto_load, sizeof(bool), 1, file);
    (void)fread(&config.auto_save, sizeof(bool), 1, file);
    (void)fread(&config.auto_save_interval, sizeof(uint32_t), 1, file);

    // Create network with reconstructed config
    adaptive_network_t network = adaptive_network_create(&config);

    // Free temporary layer_sizes array (create function makes its own copy)
    if (layer_sizes) {
        nimcp_free(layer_sizes);
    }

    if (!network) {
        fclose(file);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "adaptive_network_load: network is NULL");
        return NULL;
    }

    // Read neuron count
    uint32_t num_neurons = 0;
    if (fread(&num_neurons, sizeof(uint32_t), 1, file) != 1) {
        adaptive_network_destroy(network);
        fclose(file);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "adaptive_network_load: validation failed");
        return NULL;
    }

    // Validate neuron count against network capacity
    if (num_neurons > 10000000) {
        adaptive_network_destroy(network);
        fclose(file);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM,
            "adaptive_network_load: saved neuron count exceeds maximum (10M)");
        return NULL;
    }

    // Read neuron states
    for (uint32_t i = 0; i < num_neurons; i++) {
        if (fread(&network->neuron_states[i], sizeof(adaptive_neuron_state_t), 1, file) != 1) {
            adaptive_network_destroy(network);
            fclose(file);
            NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "adaptive_network_load: validation failed");
            return NULL;
        }
    }

    // Read label map
    uint32_t num_labels = 0;
    if (fread(&num_labels, sizeof(uint32_t), 1, file) != 1) {
        adaptive_network_destroy(network);
        fclose(file);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "adaptive_network_load: validation failed");
        return NULL;
    }

    // Validate num_labels to prevent corrupt data from causing crashes
    if (num_labels > 100000) {
        adaptive_network_destroy(network);
        fclose(file);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM,
            "adaptive_network_load: num_labels exceeds maximum (100K)");
        return NULL;
    }

    network->label_map = nimcp_malloc(num_labels * sizeof(char*));
    network->num_labels = num_labels;

    for (uint32_t i = 0; i < num_labels; i++) {
        uint32_t label_len = 0;
        if (fread(&label_len, sizeof(uint32_t), 1, file) != 1) {
            adaptive_network_destroy(network);
            fclose(file);
            NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "adaptive_network_load: validation failed");
            return NULL;
        }

        // Validate label length
        if (label_len > 65536) {
            adaptive_network_destroy(network);
            fclose(file);
            NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM,
                "adaptive_network_load: label_len exceeds maximum (64K)");
            return NULL;
        }

        network->label_map[i] = nimcp_malloc(label_len);
        if (fread(network->label_map[i], label_len, 1, file) != 1) {
            adaptive_network_destroy(network);
            fclose(file);
            NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "adaptive_network_load: validation failed");
            return NULL;
        }
    }

    // Read statistics (non-critical, use (void) cast)
    (void)fread(&network->total_inferences, sizeof(uint64_t), 1, file);
    (void)fread(&network->total_learning_steps, sizeof(uint64_t), 1, file);
    (void)fread(&network->running_sparsity, sizeof(float), 1, file);

    // Read synaptic weights from base_network (CRITICAL: restores learned weights)
    uint32_t base_num_neurons = 0;
    if (fread(&base_num_neurons, sizeof(uint32_t), 1, file) == 1 && base_num_neurons > 0) {
        if (network->base_network) {
            // Validate neuron count matches
            uint32_t current_num_neurons = neural_network_get_num_neurons(network->base_network);
            if (base_num_neurons != current_num_neurons) {
                fprintf(stderr, "WARNING: Saved neuron count (%u) != current (%u), skipping weight load\n",
                        base_num_neurons, current_num_neurons);
                fclose(file);
                return network;
            }

            for (uint32_t i = 0; i < base_num_neurons; i++) {
                neuron_t* neuron = neural_network_get_neuron(network->base_network, i);
                if (!neuron) {
                    fprintf(stderr, "WARNING: Failed to get neuron %u\n", i);
                    break;
                }

                // Read number of synapses
                uint32_t num_synapses = 0;
                if (fread(&num_synapses, sizeof(uint32_t), 1, file) != 1) {
                    fprintf(stderr, "WARNING: Failed to read synapse count for neuron %u\n", i);
                    break;
                }

                // Clear existing synapses and load from file
                sparse_synapse_pool_t h_pool = neural_network_get_synapse_handle_pool(network->base_network);
                synapse_metadata_pool_t m_pool = neural_network_get_synapse_metadata_pool(network->base_network);

                // Actually clear existing outgoing synapses before loading saved ones
                sparse_synapse_storage_cleanup(h_pool, &neuron->outgoing);
                sparse_synapse_storage_init(&neuron->outgoing);

                for (uint32_t j = 0; j < num_synapses; j++) {

                    // Read synapse data into temporaries
                    uint32_t target_id; float weight, plasticity, trace, strength;
                    float meta_plasticity, last_change; uint64_t last_active;
                    bool enable_stp;
                    if (fread(&target_id, sizeof(uint32_t), 1, file) != 1) break;
                    if (fread(&weight, sizeof(float), 1, file) != 1) break;
                    if (fread(&plasticity, sizeof(float), 1, file) != 1) break;
                    if (fread(&trace, sizeof(float), 1, file) != 1) break;
                    if (fread(&strength, sizeof(float), 1, file) != 1) break;
                    if (fread(&meta_plasticity, sizeof(float), 1, file) != 1) break;
                    if (fread(&last_change, sizeof(float), 1, file) != 1) break;
                    if (fread(&last_active, sizeof(uint64_t), 1, file) != 1) break;
                    if (fread(&enable_stp, sizeof(bool), 1, file) != 1) break;
                    stp_state_t stp_data = {0};
                    if (enable_stp) {
                        if (fread(&stp_data, sizeof(stp_state_t), 1, file) != 1) break;
                    }

                    // Add synapse with metadata to sparse storage
                    if (sparse_synapse_add_with_metadata(h_pool, m_pool,
                            &neuron->outgoing, target_id, weight, 0) == 0) {
                        uint32_t idx = NEURON_OUT_COUNT(neuron) - 1;
                        synapse_handle_t* h = NEURON_OUT_HANDLE(neuron, idx);
                        if (h) h->strength = strength;
                        synapse_t* syn = h ? sparse_synapse_get_metadata(m_pool, h) : NULL;
                        if (syn) {
                            syn->plasticity = plasticity;
                            syn->trace = trace;
                            syn->strength = strength;
                            syn->meta_plasticity = meta_plasticity;
                            syn->last_change = last_change;
                            syn->last_active = last_active;
                            syn->enable_stp = enable_stp;
                            if (enable_stp) syn->stp = stp_data;
                        }
                    }
                }

                // Restore full neuron state (CRITICAL: enables exact training resume)
                if (fread(&neuron->state, sizeof(float), 1, file) != 1) break;
                if (fread(&neuron->bias, sizeof(float), 1, file) != 1) break;
                if (fread(&neuron->threshold, sizeof(float), 1, file) != 1) break;
                if (fread(&neuron->adaptation, sizeof(float), 1, file) != 1) break;
                if (fread(&neuron->calcium_concentration, sizeof(float), 1, file) != 1) break;
                if (fread(&neuron->plasticity_rate, sizeof(float), 1, file) != 1) break;
                if (fread(&neuron->homeostatic_factor, sizeof(float), 1, file) != 1) break;
                if (fread(&neuron->avg_activity, sizeof(float), 1, file) != 1) break;
                if (fread(&neuron->weight_norm, sizeof(float), 1, file) != 1) break;
                if (fread(&neuron->learning_rule, sizeof(learning_rule_t), 1, file) != 1) break;
                if (fread(&neuron->activation_type, sizeof(activation_type_t), 1, file) != 1) break;
                if (fread(&neuron->oja_params, sizeof(oja_params_t), 1, file) != 1) break;
                if (fread(&neuron->stdp_params, sizeof(stdp_params_t), 1, file) != 1) break;
                if (fread(&neuron->homeostatic, sizeof(homeostatic_params_t), 1, file) != 1) break;
                if (fread(&neuron->last_spike, sizeof(uint64_t), 1, file) != 1) break;
                if (fread(&neuron->last_update, sizeof(uint64_t), 1, file) != 1) break;
                if (fread(&neuron->model_type, sizeof(neuron_model_type_t), 1, file) != 1) break;
                // Note: neuron.model (neuron_model_state_t) is opaque pointer, not restored here
                // TODO: If model_type != NEURON_MODEL_NONE, deserialize model-specific state
            }
        }
    }

    fclose(file);

    // Rebuild incoming synapses from outgoing for forward pass
    if (network->base_network) {
        neural_network_rebuild_incoming(network->base_network);
    }

    // Phase GPU: Initialize GPU acceleration for loaded network
    network->gpu_enabled = false;
    network->gpu_ctx = NULL;
    network->gpu_weight_cache = NULL;
    if (gpu_is_available() &&
        network->config.base_config.num_layers >= 2 &&
        network->config.base_config.layer_sizes) {
        network->gpu_ctx = nimcp_gpu_context_create_auto();
        if (network->gpu_ctx) {
            network->gpu_weight_cache = nimcp_gpu_weight_cache_create(
                network->gpu_ctx,
                network->base_network,
                network->config.base_config.layer_sizes,
                network->config.base_config.num_layers);
            if (network->gpu_weight_cache) {
                nimcp_gpu_weight_cache_upload(network->gpu_weight_cache,
                                             network->base_network);
                network->gpu_enabled = true;
            } else {
                nimcp_gpu_context_destroy(network->gpu_ctx);
                network->gpu_ctx = NULL;
            }
        }
    }

    return network;
}

size_t adaptive_network_get_size(adaptive_network_t network)
{
    if (!network)
        return 0;

    size_t size = sizeof(struct adaptive_network_struct);
    size += network->num_neurons * sizeof(adaptive_neuron_state_t);
    size += network->config.base_config.input_size * sizeof(float);

    // Add label map size
    for (uint32_t i = 0; i < network->num_labels; i++) {
        size += strlen(network->label_map[i]) + 1;
    }

    // Add base network size (approximate)
    size += (size_t)network->num_neurons * network->config.base_config.input_size * sizeof(float);

    return size;
}

//=============================================================================
// Interpretability & Analysis
//=============================================================================

bool adaptive_network_analyze_activation(adaptive_network_t network, const float* input,
                                         uint32_t input_size, activation_analysis_t* analysis)
{
    if (!network || !input || !analysis) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "adaptive_network_analyze_activation: required parameter is NULL (network, input, analysis)");
        return false;
    }

    // Forward pass (Phase MP: use pool)
    float* output = (float*)alloc_hot_buffer(network->config.base_config.output_size * sizeof(float));
    if (!output) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "adaptive_network_analyze_activation: output is NULL");
        return false;
    }
    uint32_t active_count = adaptive_network_forward(network, input, input_size, output,
                                                     network->config.base_config.output_size, 0);

    analysis->num_active_neurons = active_count;
    analysis->sparsity = network->running_sparsity;

    // Allocate arrays for active neurons
    analysis->active_neuron_ids = nimcp_malloc(active_count * sizeof(uint32_t));
    analysis->activation_strengths = nimcp_malloc(active_count * sizeof(float));

    if (!analysis->active_neuron_ids || !analysis->activation_strengths) {
        nimcp_free(analysis->active_neuron_ids);
        nimcp_free(analysis->activation_strengths);
        free_hot_buffer(output);  // Phase MP: Return to pool
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "adaptive_network_analyze_activation: required parameter is NULL (analysis->active_neuron_ids, analysis->activation_strengths)");
        return false;
    }

    // Collect active neurons
    uint32_t idx = 0;
    for (uint32_t i = 0; i < network->config.base_config.output_size && idx < active_count; i++) {
        if (fabsf(output[i]) > network->neuron_states[i].adaptive_threshold) {
            analysis->active_neuron_ids[idx] = i;
            analysis->activation_strengths[idx] = output[i];
            idx++;
        }
    }

    // Compute confidence (based on max activation)
    float max_activation = 0.0F;
    for (uint32_t i = 0; i < active_count; i++) {
        if (fabsf(analysis->activation_strengths[i]) > max_activation) {
            max_activation = fabsf(analysis->activation_strengths[i]);
        }
    }
    analysis->confidence = fminf(max_activation / 10.0F, 1.0F);  // Normalize to [0,1]

    free_hot_buffer(output);  // Phase MP: Return to pool
    return true;
}

uint32_t adaptive_network_rank_neurons(adaptive_network_t network, neuron_importance_t* rankings,
                                       uint32_t max_rankings)
{
    if (!network || !rankings)
        return 0;

    uint32_t num_to_rank =
        (max_rankings < network->num_neurons) ? max_rankings : network->num_neurons;

    // Compute importance based on activation statistics
    for (uint32_t i = 0; i < num_to_rank; i++) {
        rankings[i].neuron_id = i;
        rankings[i].avg_activation = network->neuron_states[i].activation_mean;
        rankings[i].activation_count = network->neuron_states[i].sample_count;

        // Importance = activation_mean * sqrt(activation_count) / variance
        float variance = (network->neuron_states[i].sample_count > 1)
                             ? network->neuron_states[i].activation_variance /
                                   (network->neuron_states[i].sample_count - 1)
                             : 1.0F;

        float act_count = (float) rankings[i].activation_count;
        float importance = rankings[i].avg_activation *
                           sqrtf(act_count > 0.0f ? act_count : 0.0f) / (variance + 1e-6F);
        rankings[i].importance = isfinite(importance) ? importance : 0.0f;

        rankings[i].most_active_for = NULL;  // TODO: Track pattern associations
    }

    // Sort by importance (descending) using consolidated nimcp_sort
    nimcp_sort(rankings, num_to_rank, sizeof(neuron_importance_t),
               compare_neuron_importance_desc);

    return num_to_rank;
}

uint32_t adaptive_network_explain(adaptive_network_t network, const float* input,
                                  uint32_t input_size, char* explanation, uint32_t max_length)
{
    if (!network || !input || !explanation || max_length == 0)
        return 0;

    // Analyze activation
    activation_analysis_t analysis;
    if (!adaptive_network_analyze_activation(network, input, input_size, &analysis)) {
        return 0;
    }

    // Generate explanation
    int written = snprintf(explanation, max_length,
                           "Activated %u/%u neurons (%.1f%% sparse). "
                           "Confidence: %.2f. "
                           "Top contributors: ",
                           analysis.num_active_neurons, network->config.base_config.output_size,
                           analysis.sparsity * 100.0F, analysis.confidence);

    // Add top 3 neurons
    uint32_t top_count = (analysis.num_active_neurons < 3) ? analysis.num_active_neurons : 3;
    for (uint32_t i = 0; i < top_count; i++) {
        written += snprintf(explanation + written, max_length - written, "N%u(%.2f) ",
                            analysis.active_neuron_ids[i], analysis.activation_strengths[i]);
    }

    // Free analysis arrays
    nimcp_free(analysis.active_neuron_ids);
    nimcp_free(analysis.activation_strengths);

    return written;
}

//=============================================================================
// Performance Statistics
//=============================================================================

bool adaptive_network_get_performance(adaptive_network_t network, network_performance_t* stats)
{
    if (!network || !stats) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "adaptive_network_get_performance: required parameter is NULL (network, stats)");
        return false;
    }

    stats->total_inferences = network->total_inferences;
    stats->total_learning_steps = network->total_learning_steps;
    stats->avg_sparsity = network->running_sparsity;
    stats->avg_inference_time_us = network->running_inference_time_us;
    stats->avg_learning_time_us = network->running_learning_time_us;
    stats->memory_usage_bytes = adaptive_network_get_size(network);
    stats->accuracy = 0.0F;  // TODO: Track validation accuracy
    stats->num_pruned_synapses = network->num_pruned_synapses;

    return true;
}

void adaptive_network_reset_stats(adaptive_network_t network)
{
    if (!network)
        return;

    network->total_inferences = 0;
    network->total_learning_steps = 0;
    network->running_sparsity = 0.0F;
    network->running_inference_time_us = 0.0F;
    network->running_learning_time_us = 0.0F;
}

//=============================================================================
// Introspection & Network State Access (NIMCP 2.5 Consciousness APIs)
//=============================================================================

/**
 * WHAT: Get total number of neurons
 * WHY: Introspection needs to know network size
 * HOW: Return cached neuron count
 */
uint32_t adaptive_network_get_neuron_count(adaptive_network_t network)
{
    if (!network)
        return 0;
    return network->num_neurons;
}

/**
 * WHAT: Get activation level of specific neuron
 * WHY: Introspection needs to query individual neuron states
 * HOW: Access base network neuron state
 */
bool adaptive_network_get_neuron_activation(adaptive_network_t network, uint32_t neuron_id,
                                            float* activation)
{
    if (!network || !activation || neuron_id >= network->num_neurons) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "adaptive_network_get_neuron_activation: required parameter is NULL (network, activation)");
        return false;
    }

    /* WHAT: Get neuron state from base network */
    return neural_network_get_neuron_state(network->base_network, neuron_id, activation);
}

/**
 * WHAT: Get list of active neurons above threshold
 * WHY: Introspection brain_get_active_population() needs this
 * HOW: Scan all neurons, collect those above threshold
 *
 * COMPLEXITY: O(n) where n = num_neurons
 */
uint32_t adaptive_network_get_active_neurons(adaptive_network_t network, float threshold,
                                             uint32_t* neuron_ids, float* activations,
                                             uint32_t max_neurons)
{
    if (!network || !neuron_ids || !activations || max_neurons == 0) {
        return 0;
    }

    /* WHAT: Scan neurons and collect active ones */
    /* NOTE: Biological potentials need normalization */
    const float REST_POTENTIAL = -65.0F;
    const float PEAK_POTENTIAL = 30.0F;

    uint32_t count = 0;
    for (uint32_t i = 0; i < network->num_neurons && count < max_neurons; i++) {
        float raw_activation = 0.0F;
        if (neural_network_get_neuron_state(network->base_network, i, &raw_activation)) {
            /* WHAT: Normalize to 0-1 range */
            float normalized =
                (raw_activation - REST_POTENTIAL) / (PEAK_POTENTIAL - REST_POTENTIAL);
            if (normalized < 0.0F)
                normalized = 0.0F;
            if (normalized > 1.0F)
                normalized = 1.0F;

            if (normalized >= threshold) {
                neuron_ids[count] = i;
                activations[count] = normalized;
                count++;
            }
        }
    }

    return count;
}

/**
 * WHAT: Get connection count for a neuron
 * WHY: Introspection needs neuron topology info
 * HOW: Access base network neuron synapse count
 */
bool adaptive_network_get_connection_count(adaptive_network_t network, uint32_t neuron_id,
                                           uint32_t* num_connections)
{
    if (!network || !num_connections || neuron_id >= network->num_neurons) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "adaptive_network_get_connection_count: required parameter is NULL (network, num_connections)");
        return false;
    }

    /* WHAT: Access base network to get neuron */
    /* NOTE: This requires adding a new API to neural_network or
     * making neurons publicly accessible. For now, estimate from
     * network configuration. */

    /* TODO: Add neural_network_get_neuron_info() API */
    /* For now, return average connections based on sparsity */
    uint32_t avg_connections =
        (uint32_t) (network->num_neurons * (1.0F - network->config.spike_params.sparsity_target));
    *num_connections = avg_connections;

    return true;
}

/**
 * WHAT: Get total weight for a neuron
 * WHY: Introspection needs neuron importance metrics
 * HOW: Sum absolute values of connection weights
 */
bool adaptive_network_get_total_weight(adaptive_network_t network, uint32_t neuron_id,
                                       float* total_weight)
{
    if (!network || !total_weight || neuron_id >= network->num_neurons) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "adaptive_network_get_total_weight: required parameter is NULL (network, total_weight)");
        return false;
    }

    /* TODO: Add neural_network_get_neuron_weights() API */
    /* For now, estimate based on neuron activation and network stats */
    float activation = 0.0F;
    if (neural_network_get_neuron_state(network->base_network, neuron_id, &activation)) {
        /* Estimate: active neurons have higher total weights */
        /* Use absolute value since total weight should be non-negative */
        *total_weight = fabsf(activation) * 10.0F; /* Rough estimate */
        return true;
    }

    NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "adaptive_network_get_total_weight: validation failed");
    return false;
}

/**
 * WHAT: Get base network handle
 * WHY: Allow consciousness APIs direct access when needed
 * HOW: Return internal base_network pointer
 *
 * WARNING: This exposes internals - use carefully!
 */
neural_network_t adaptive_network_get_base_network(adaptive_network_t network)
{
    if (!network) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "network is NULL");

        return NULL;

    }
    return network->base_network;
}

//=============================================================================
// GPU Inference Accessors
//=============================================================================

void adaptive_network_set_gpu_context(adaptive_network_t network, struct nimcp_gpu_context_s* ctx)
{
    if (!network) return;
    // Don't destroy existing — brain owns the context, not us
    network->gpu_ctx = ctx;
}

void adaptive_network_set_gpu_weight_cache(adaptive_network_t network, struct nimcp_gpu_weight_cache_s* cache)
{
    if (!network) return;
    // Destroy existing cache if we owned one
    if (network->gpu_weight_cache && network->gpu_weight_cache != cache) {
        nimcp_gpu_weight_cache_destroy(network->gpu_weight_cache);
    }
    network->gpu_weight_cache = cache;
}

void adaptive_network_set_gpu_enabled(adaptive_network_t network, bool enabled)
{
    if (!network) return;
    network->gpu_enabled = enabled;
}

bool adaptive_network_is_gpu_enabled(adaptive_network_t network)
{
    if (!network) return false;
    return network->gpu_enabled;
}

//=============================================================================
// Neuron (Inferentia) Inference Accessors
//=============================================================================

void adaptive_network_set_neuron_cache(adaptive_network_t network, struct nimcp_neuron_inference_cache* cache)
{
    if (!network) return;
    // Destroy existing cache if we owned one
    if (network->neuron_cache && network->neuron_cache != cache) {
        nimcp_neuron_cache_destroy(network->neuron_cache);
    }
    network->neuron_cache = cache;
}

void adaptive_network_set_neuron_enabled(adaptive_network_t network, bool enabled)
{
    if (!network) return;
    network->neuron_enabled = enabled;
}

bool adaptive_network_is_neuron_enabled(adaptive_network_t network)
{
    if (!network) return false;
    return network->neuron_enabled;
}

//=============================================================================
// Frozen Network Support
//=============================================================================

void adaptive_network_freeze(adaptive_network_t network)
{
    if (!network) return;

    // Mark frozen
    network->frozen = true;

    // Clear dirty flag — weights are final
    if (network->gpu_weight_cache) {
        network->gpu_weight_cache->weights_dirty_on_cpu = false;
    }
}

bool adaptive_network_is_frozen(adaptive_network_t network)
{
    if (!network) return false;
    return network->frozen;
}

float adaptive_network_get_last_grad_norm(adaptive_network_t network)
{
    if (!network) return 0.0f;
    return network->last_grad_norm;
}

/**
 * WHAT: Get network configuration (read-only)
 * WHY: Enable brain resize to read current config for cloning
 * HOW: Return pointer to internal config structure
 *
 * Phase 2.8: Brain Dynamic Resizing Support
 * COMPLEXITY: O(1) - direct pointer access
 *
 * @param network Adaptive network
 * @return Pointer to config (read-only), or NULL on error
 */
const adaptive_network_config_t* adaptive_network_get_config(adaptive_network_t network)
{
    /* Guard: Validate network handle */
    if (!network) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "network is NULL");

        return NULL;
    }

    /* WHAT: Return pointer to internal config */
    /* NOTE: Returned as const to prevent modification */
    /* WHY: Config should only be changed during creation or explicit reconfigure */
    return &network->config;
}

/**
 * WHAT: Get number of neurons in the adaptive network
 * WHY: Enable topology analysis and community detection
 * HOW: Return neuron count from base network
 *
 * COMPLEXITY: O(1) - direct field access
 *
 * @param network Adaptive network
 * @return Number of neurons, or 0 on error
 */
uint32_t adaptive_network_get_num_neurons(adaptive_network_t network)
{
    /* Guard: Validate network handle */
    if (!network) {
        return 0;
    }

    /* Guard: Validate base network */
    if (!network->base_network) {
        return 0;
    }

    /* Return neuron count from base network */
    return neural_network_get_num_neurons(network->base_network);
}

/**
 * WHAT: Get synapse weight between two neurons
 * WHY: Enable community detection to analyze connectivity patterns
 * HOW: Access neuron synapses to find connection weight
 *
 * COMPLEXITY: O(s) where s = synapses per neuron (typically < 100)
 *
 * @param network Adaptive network
 * @param from_neuron Source neuron ID
 * @param to_neuron Target neuron ID
 * @return Synapse weight, or 0.0f if no connection exists
 */
float adaptive_network_get_synapse_weight(adaptive_network_t network, uint32_t from_neuron, uint32_t to_neuron)
{
    /* Guard: Validate network */
    if (!network || !network->base_network) {
        return 0.0F;
    }

    /* Guard: Validate neuron IDs */
    uint32_t num_neurons = neural_network_get_num_neurons(network->base_network);
    if (from_neuron >= num_neurons || to_neuron >= num_neurons) {
        return 0.0F;
    }

    /* Get source neuron */
    neuron_t* from = neural_network_get_neuron(network->base_network, from_neuron);
    if (!from) {
        return 0.0F;
    }

    /* Search for synapse to target neuron */
    for (uint32_t i = 0; i < NEURON_OUT_COUNT(from); i++) {
        synapse_handle_t* h = NEURON_OUT_HANDLE(from, i);
        if (h && h->target_neuron_id == to_neuron) {
            return h->weight;
        }
    }

    /* No connection found */
    return 0.0F;
}

//=============================================================================
// Phase COW: Copy-on-Write Support for Training Snapshots
//=============================================================================

/**
 * @brief Enable COW-backed state storage for the adaptive network
 *
 * WHAT: Convert neuron states to COW-backed storage
 * WHY:  Enable O(1) snapshots for training rollback and checkpointing
 * HOW:  Create COW region from existing states, redirect pointer
 *
 * @param network Adaptive network
 * @return true on success, false if already using COW or on failure
 *
 * COMPLEXITY: O(n) where n = neuron_states size (one-time copy)
 */
bool adaptive_network_enable_cow_states(adaptive_network_t network)
{
    if (!network || !network->neuron_states) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "adaptive_network_enable_cow_states: required parameter is NULL (network, network->neuron_states)");
        return false;
    }

    if (network->uses_cow_states) {
        return true;  // Already enabled
    }

    // Calculate states size
    size_t states_size = network->num_neurons * sizeof(adaptive_neuron_state_t);
    size_t aligned_size = page_cow_align_size(states_size);

    // Create COW region with existing states
    page_cow_config_t config = page_cow_default_config(aligned_size);
    config.enable_tracking = true;

    network->cow_states_region = page_cow_region_create(&config, network->neuron_states);
    if (!network->cow_states_region) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "adaptive_network_enable_cow_states: network->cow_states_region is NULL");
        return false;
    }

    // Create view
    network->cow_states_view = page_cow_view_create(network->cow_states_region);
    if (!network->cow_states_view) {
        page_cow_region_destroy(network->cow_states_region);
        network->cow_states_region = NULL;
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "adaptive_network_enable_cow_states: network->cow_states_view is NULL");
        return false;
    }

    // Free original states and point to COW view
    nimcp_free(network->neuron_states);
    network->neuron_states = (adaptive_neuron_state_t*)page_cow_view_write(network->cow_states_view);
    network->uses_cow_states = true;

    return network->neuron_states != NULL;
}

/**
 * @brief Create instant snapshot of network state for rollback
 *
 * WHAT: O(1) snapshot of neuron states
 * WHY:  Enable rollback during training (e.g., failed batch)
 * HOW:  Page-level COW snapshot, copies only on subsequent writes
 *
 * @param network Adaptive network (must have COW enabled)
 * @return Snapshot handle, or NULL on failure
 *
 * COMPLEXITY: O(1) - No data copying
 */
page_cow_snapshot_t adaptive_network_snapshot_states(adaptive_network_t network)
{
    if (!network || !network->uses_cow_states || !network->cow_states_view) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "adaptive_network_snapshot_states: required parameter is NULL (network, network->uses_cow_states, network->cow_states_view)");
        return NULL;
    }

    return page_cow_snapshot_create(network->cow_states_view);
}

/**
 * @brief Restore network state from snapshot
 *
 * WHAT: Rollback neuron states to snapshot state
 * WHY:  Recover from failed training iterations
 * HOW:  Page-level restore, discards private pages
 *
 * @param network Adaptive network
 * @param snapshot Snapshot to restore from
 * @return true on success
 *
 * COMPLEXITY: O(num_modified_pages)
 */
bool adaptive_network_restore_states(adaptive_network_t network,
                                     page_cow_snapshot_t snapshot)
{
    if (!network || !network->uses_cow_states || !network->cow_states_view || !snapshot) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "adaptive_network_restore_states: required parameter is NULL (network, network->uses_cow_states, network->cow_states_view, snapshot)");
        return false;
    }

    bool success = page_cow_snapshot_restore(network->cow_states_view, snapshot);
    if (success) {
        // Re-acquire writable pointer after restore
        network->neuron_states = (adaptive_neuron_state_t*)page_cow_view_write(network->cow_states_view);
    }

    return success;
}

/**
 * @brief Get memory savings from COW state sharing
 *
 * @param network Adaptive network
 * @return Bytes saved by sharing, or 0 if not using COW
 */
size_t adaptive_network_get_cow_savings(adaptive_network_t network)
{
    if (!network || !network->uses_cow_states || !network->cow_states_view) {
        return 0;
    }

    return page_cow_view_get_memory_saved(network->cow_states_view);
}
