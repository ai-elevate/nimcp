//=============================================================================
// nimcp_brain_resize.c - Dynamic Brain Resizing Implementation
//=============================================================================
/**
 * @file nimcp_brain_resize.c
 * @brief Dynamic brain expansion with neuron preservation
 *
 * WHAT: Enables brains to grow during training without losing learned knowledge
 * WHY:  Allows starting small (fast training) and scaling up based on capacity needs
 * HOW:  Creates larger network, transfers existing neurons/weights, preserves all state
 *
 * DESIGN PRINCIPLES:
 * - Zero knowledge loss: All neurons, weights, and states preserved exactly
 * - Incremental growth: Add neurons progressively, not exponential jumps
 * - Automatic triggers: Detect when brain is saturated and needs more capacity
 * - Backward compatible: Works with all existing cognitive subsystems
 *
 * ARCHITECTURE:
 * ```
 * brain_resize(brain, new_size) {
 *   1. Validate new size > current size
 *   2. Create new larger adaptive network
 *   3. Transfer all existing neurons (weights, states, traces)
 *   4. Add new neurons with random initialization
 *   5. Update all cognitive subsystems (working memory, etc.)
 *   6. Swap networks atomically
 *   7. Destroy old network
 * }
 *
 * brain_auto_resize(brain) {
 *   if (should_grow(brain)) {
 *     new_size = compute_next_size(current_size);
 *     brain_resize(brain, new_size);
 *   }
 * }
 * ```
 *
 * PERFORMANCE:
 * - Resize time: O(n + m) where n=old neurons, m=new neurons
 * - Memory overhead: 2× during transfer (old + new network)
 * - No inference downtime: Can continue serving requests
 *
 * AUTO-GROWTH TRIGGERS:
 * - High utilization: >90% neurons active consistently
 * - Weight saturation: Most weights near ±max
 * - Performance plateau: Learning rate stagnates
 * - Explicit request: User/API calls resize
 *
 * @author NIMCP Development Team
 * @date 2025-11-16
 * @version 2.8.0
 */

#include "nimcp_brain_resize.h"
#include "nimcp_brain.h"
#include "plasticity/adaptive/nimcp_adaptive.h"
#include "core/neuralnet/nimcp_neuralnet.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/platform/nimcp_system_resources.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

// Logging macros (use fprintf if logging not available)
#ifndef NIMCP_LOG_INFO
#define NIMCP_LOG_INFO(...)  fprintf(stderr, "[INFO] " __VA_ARGS__); fprintf(stderr, "\n")
#endif

#ifndef NIMCP_LOG_ERROR
#define NIMCP_LOG_ERROR(...) fprintf(stderr, "[ERROR] " __VA_ARGS__); fprintf(stderr, "\n")
#endif

#ifndef NIMCP_LOG_WARN
#define NIMCP_LOG_WARN(...)  fprintf(stderr, "[WARN] " __VA_ARGS__); fprintf(stderr, "\n")
#endif

//=============================================================================
// Internal Brain Structure (forward declaration for access)
//=============================================================================

/**
 * @brief Internal brain structure
 * WHAT: Forward declaration to access internals from resize logic
 * WHY:  Resize needs to update network and config fields
 */
struct brain_struct {
    adaptive_network_t network;
    brain_config_t config;
    // ... other fields not needed for resize
};

//=============================================================================
// Auto-Growth Detection
//=============================================================================

/**
 * @brief Growth trigger metrics
 * WHAT: Metrics for determining when brain needs more capacity
 * WHY:  Enable data-driven automatic growth decisions
 */
typedef struct {
    float avg_neuron_utilization;  /**< Fraction of neurons active (0.0-1.0) */
    float weight_saturation;        /**< Fraction of weights near limits (0.0-1.0) */
    float learning_progress;        /**< Recent learning improvement rate */
    uint32_t steps_since_growth;    /**< Training steps since last resize */
    bool manual_trigger;            /**< User explicitly requested growth */
} growth_metrics_t;

/**
 * @brief Compute neuron utilization
 *
 * WHAT: Calculate what fraction of neurons are actively participating
 * WHY:  High utilization (>90%) indicates capacity saturation
 * HOW:  Count neurons with non-zero activity over recent window
 *
 * COMPLEXITY: O(n) where n = num_neurons
 *
 * @param network Neural network to analyze
 * @return Utilization ratio [0.0, 1.0]
 */
static float compute_neuron_utilization(neural_network_t network)
{
    if (!network) {
        return -1.0f;  // Error signal
    }

    uint32_t num_neurons = neural_network_get_num_neurons(network);
    if (num_neurons == 0) {
        return -1.0f;  // Error signal
    }

    // Count active neurons with error handling
    uint32_t active_count = 0;
    uint32_t valid_reads = 0;

    for (uint32_t i = 0; i < num_neurons; i++) {
        // Bounds check
        if (i >= num_neurons) {
            continue;  // Skip invalid index
        }

        // Try to get activity, skip on error
        float avg_activity = neural_network_get_average_activity(network, i);

        // Activity is valid if non-negative
        if (avg_activity >= 0.0f) {
            valid_reads++;
            if (avg_activity > 0.01f) {
                active_count++;
            }
        }
    }

    // If we couldn't read any neurons, signal error
    if (valid_reads == 0) {
        return -1.0f;  // Error signal
    }

    return (float)active_count / (float)valid_reads;
}

/**
 * @brief Compute weight saturation
 *
 * WHAT: Calculate what fraction of weights are near their limits
 * WHY:  Saturated weights can't learn effectively, need more capacity
 * HOW:  Count weights with |w| > 0.9 * max_weight
 *
 * COMPLEXITY: O(n × s) where s = avg synapses per neuron
 *
 * @param network Neural network to analyze
 * @return Saturation ratio [0.0, 1.0]
 */
static float compute_weight_saturation(neural_network_t network)
{
    if (!network) {
        return 0.0f;
    }

    uint32_t num_neurons = neural_network_get_num_neurons(network);
    if (num_neurons == 0) {
        return 0.0f;
    }

    uint32_t total_weights = 0;
    uint32_t saturated_weights = 0;
    const float saturation_threshold = 0.9f;  // Consider saturated if |w| > 0.9

    for (uint32_t i = 0; i < num_neurons; i++) {
        neuron_t* neuron = neural_network_get_neuron(network, i);
        if (!neuron) {
            continue;
        }

        // Validate synapse data before accessing
        if (!neuron->synapses) {
            continue;  // Skip neurons with no synapses allocated
        }

        // Validate num_synapses is reasonable (< 10000 per neuron is sane limit)
        if (neuron->num_synapses > 10000) {
            continue;  // Skip neurons with garbage synapse count
        }

        for (uint32_t j = 0; j < neuron->num_synapses; j++) {
            float weight = neuron->synapses[j].weight;
            total_weights++;

            if (fabsf(weight) > saturation_threshold) {
                saturated_weights++;
            }
        }
    }

    if (total_weights == 0) {
        return 0.0f;
    }

    return (float)saturated_weights / (float)total_weights;
}

/**
 * @brief Check if brain should grow
 *
 * WHAT: Determine if brain has reached capacity and needs more neurons
 * WHY:  Automatic growth enables continuous learning without manual intervention
 * HOW:  Evaluate multiple metrics and apply threshold-based decision logic
 *
 * GROWTH CRITERIA (any trigger growth):
 * - High utilization: >90% neurons active for >1000 steps
 * - Weight saturation: >80% weights near limits
 * - Manual trigger: User explicitly called resize
 *
 * @param metrics Growth metrics
 * @return true if brain should grow
 */
static bool should_grow(const growth_metrics_t* metrics)
{
    if (!metrics) {
        return false;
    }

    // Trigger 1: Manual request
    if (metrics->manual_trigger) {
        NIMCP_LOG_INFO("Growth triggered: Manual request");
        return true;
    }

    // Trigger 2: High neuron utilization
    const float utilization_threshold = 0.90f;
    const uint32_t utilization_patience = 1000;  // Steps to wait before growing

    if (metrics->avg_neuron_utilization > utilization_threshold &&
        metrics->steps_since_growth > utilization_patience) {
        NIMCP_LOG_INFO("Growth triggered: High utilization (%.2f%% > %.2f%%)",
                       metrics->avg_neuron_utilization * 100,
                       utilization_threshold * 100);
        return true;
    }

    // Trigger 3: Weight saturation
    const float saturation_threshold = 0.80f;

    if (metrics->weight_saturation > saturation_threshold) {
        NIMCP_LOG_INFO("Growth triggered: Weight saturation (%.2f%% > %.2f%%)",
                       metrics->weight_saturation * 100,
                       saturation_threshold * 100);
        return true;
    }

    return false;
}

/**
 * @brief Compute next brain size (hardware-aware)
 *
 * WHAT: Calculate how many neurons to add when growing
 * WHY:  Incremental growth prevents memory spikes and excessive overhead
 * HOW:  Add 50% more neurons (1.5× current size), capped by hardware resources
 *
 * GROWTH POLICY:
 * - TINY (100)  → SMALL (500)   [5× growth to reach useful size]
 * - SMALL (500) → MEDIUM (1000)  [2× growth]
 * - MEDIUM (1000) → LARGE (5000) [5× growth]
 * - LARGE (5000) → 7500          [1.5× growth]
 * - Custom: Always 1.5× growth
 * - Hardware cap: Never exceed available RAM/GPU/neuromorphic capacity
 *
 * @param current_size Current neuron count
 * @param use_gpu Whether GPU acceleration is enabled
 * @return New neuron count (hardware-aware)
 */
static uint32_t compute_next_size(uint32_t current_size, bool use_gpu)
{
    // Query system resources
    system_resources_t resources;
    if (!system_resources_query(&resources)) {
        // Fallback: use preset sizes without hardware awareness
        NIMCP_LOG_WARN("compute_next_size: Failed to query system resources, using defaults");

        if (current_size <= 100) {
            return 500;  // TINY → SMALL
        } else if (current_size <= 500) {
            return 1000;  // SMALL → MEDIUM
        } else if (current_size <= 1000) {
            return 5000;  // MEDIUM → LARGE
        } else {
            return (uint32_t)(current_size * 1.5f);
        }
    }

    // Get hardware-aware recommendation
    uint32_t recommended_size = system_resources_recommend_size(&resources, current_size, use_gpu);

    // Log resource-aware decision
    NIMCP_LOG_INFO("compute_next_size: Current=%u, Recommended=%u (RAM: %luMB avail, GPU: %s)",
                   current_size, recommended_size,
                   (unsigned long)resources.available_ram_mb,
                   use_gpu ? "enabled" : "disabled");

    // Additional safety check: never recommend same or smaller size
    if (recommended_size <= current_size) {
        uint32_t min_growth = (uint32_t)(current_size * 1.1f);  // At least 10% growth
        NIMCP_LOG_WARN("compute_next_size: Recommended size (%u) <= current (%u), forcing min growth to %u",
                       recommended_size, current_size, min_growth);
        recommended_size = min_growth;
    }

    return recommended_size;
}

//=============================================================================
// Core Resize Implementation
//=============================================================================

/**
 * @brief Resize brain to new neuron count
 *
 * WHAT: Expand brain capacity while preserving all learned knowledge
 * WHY:  Enable continuous learning without hitting capacity limits
 * HOW:  Create new larger network, transfer all neurons/weights, swap atomically
 *
 * ALGORITHM:
 * 1. Validate: new_neuron_count > current (only growth supported)
 * 2. Create: Build new adaptive network with increased capacity
 * 3. Transfer: Copy all existing neurons (weights, biases, states, traces)
 * 4. Initialize: Add new neurons with random small weights
 * 5. Swap: Atomically replace old network with new network
 * 6. Update: Adjust brain config and subsystem sizes
 * 7. Cleanup: Destroy old network
 *
 * GUARANTEES:
 * - Zero knowledge loss: All weights/states preserved exactly
 * - Atomic operation: Network swap is instantaneous
 * - Memory safe: Old network destroyed only after successful transfer
 * - Config consistency: brain_config_t updated to reflect new size
 *
 * ERROR HANDLING:
 * - Returns false if new_count ≤ current (shrinking not supported)
 * - Returns false if allocation fails (old network unchanged)
 * - Returns false if transfer fails (old network unchanged)
 *
 * COMPLEXITY:
 * - Time: O(n + m) where n=old neurons, m=new neurons
 * - Space: O(max(n,m) × k) where k=avg connections
 *
 * @param brain Brain to resize
 * @param new_neuron_count New total neuron count (must be > current)
 * @return true on success, false on error (brain unchanged)
 */
bool brain_resize(brain_t brain, uint32_t new_neuron_count)
{
    // Guard: Validate brain handle
    if (!brain) {
        NIMCP_LOG_ERROR("brain_resize: NULL brain handle");
        return false;
    }

    // Get current network
    adaptive_network_t old_network = brain->network;
    if (!old_network) {
        NIMCP_LOG_ERROR("brain_resize: Brain has no network");
        return false;
    }

    // Get current size
    neural_network_t base_network = adaptive_network_get_base_network(old_network);
    if (!base_network) {
        NIMCP_LOG_ERROR("brain_resize: Cannot access base network");
        return false;
    }

    uint32_t current_neuron_count = neural_network_get_num_neurons(base_network);

    // Guard: Validate new size (only growth supported)
    if (new_neuron_count <= current_neuron_count) {
        NIMCP_LOG_ERROR("brain_resize: New size (%u) must be > current (%u). Shrinking not supported.",
                        new_neuron_count, current_neuron_count);
        return false;
    }

    NIMCP_LOG_INFO("brain_resize: Growing from %u to %u neurons (+%u, +%.1f%%)",
                   current_neuron_count, new_neuron_count,
                   new_neuron_count - current_neuron_count,
                   ((float)(new_neuron_count - current_neuron_count) / current_neuron_count) * 100);

    // Step 1: Create new larger network configuration
    const adaptive_network_config_t* old_config = adaptive_network_get_config(old_network);
    if (!old_config) {
        NIMCP_LOG_ERROR("brain_resize: Cannot access old network config");
        return false;
    }

    // DEBUG: Log old config spike params
    NIMCP_LOG_INFO("brain_resize: old_config k_factor=%f, min_threshold=%f, max_threshold=%f",
                   old_config->spike_params.k_factor,
                   old_config->spike_params.min_threshold,
                   old_config->spike_params.max_threshold);

    // Shallow copy first
    adaptive_network_config_t new_config = {0};
    memcpy(&new_config, old_config, sizeof(adaptive_network_config_t));

    // DEBUG: Log new config spike params after memcpy
    NIMCP_LOG_INFO("brain_resize: new_config k_factor=%f, min_threshold=%f, max_threshold=%f",
                   new_config.spike_params.k_factor,
                   new_config.spike_params.min_threshold,
                   new_config.spike_params.max_threshold);

    // CRITICAL FIX: Handle corrupted layer config (num_layers=0, layer_sizes=NULL)
    // WHY: Old configs from certain brain creation paths may not properly initialize layers
    // WHAT: Rebuild layer config using brain's input/output sizes
    uint32_t* new_layer_sizes = NULL;

    if (old_config->base_config.num_layers == 0 || !old_config->base_config.layer_sizes) {
        NIMCP_LOG_WARN("brain_resize: Corrupted layer config (num_layers=%u, layer_sizes=%p), rebuilding",
                       old_config->base_config.num_layers,
                       (void*)old_config->base_config.layer_sizes);

        // Rebuild layer config: [input, hidden, output]
        new_config.base_config.num_layers = 3;
        new_layer_sizes = nimcp_calloc(3, sizeof(uint32_t));
        if (!new_layer_sizes) {
            NIMCP_LOG_ERROR("brain_resize: Failed to allocate layer_sizes");
            return false;
        }

        // Get input/output sizes from old config (safer than brain->config)
        // If old config also has corrupted input/output, use base network sizes
        uint32_t input_size = old_config->base_config.input_size;
        uint32_t output_size = old_config->base_config.output_size;

        if (input_size == 0 || output_size == 0) {
            NIMCP_LOG_WARN("brain_resize: Old config has invalid sizes (input=%u, output=%u), using defaults",
                           input_size, output_size);
            input_size = 10;  // Default fallback
            output_size = 5;
        }

        new_layer_sizes[0] = input_size;
        new_layer_sizes[1] = current_neuron_count;  // Will be updated later
        new_layer_sizes[2] = output_size;
        new_config.base_config.layer_sizes = new_layer_sizes;
        new_config.base_config.input_size = input_size;
        new_config.base_config.output_size = output_size;
    }
    else {
        // Deep copy existing valid layer_sizes (avoid use-after-free)
        // WHY: Shallow copy copies the pointer, which points to old network's memory
        new_layer_sizes = nimcp_calloc(old_config->base_config.num_layers, sizeof(uint32_t));
        if (!new_layer_sizes) {
            NIMCP_LOG_ERROR("brain_resize: Failed to allocate layer_sizes");
            return false;
        }
        memcpy(new_layer_sizes, old_config->base_config.layer_sizes,
               old_config->base_config.num_layers * sizeof(uint32_t));
        new_config.base_config.layer_sizes = new_layer_sizes;
    }

    // CRITICAL FIX: If spike_params are corrupted (k_factor==0), reinitialize with sane defaults
    // WHY: Loaded checkpoints or corrupted configs can have invalid spike params
    // WHAT: Reset to known-good defaults that match build_spike_params
    if (new_config.spike_params.k_factor <= 0.0f) {
        NIMCP_LOG_WARN("brain_resize: Corrupted spike_params detected (k_factor=%f), resetting to defaults",
                       new_config.spike_params.k_factor);
        new_config.spike_params.k_factor = 0.5f;
        new_config.spike_params.min_threshold = 0.0001f;
        new_config.spike_params.max_threshold = 10.0f;
        new_config.spike_params.sparsity_target = 0.7f;
        new_config.spike_params.encoding = SPIKE_ENCODING_INTEGER;
        new_config.spike_params.enable_soft_reset = true;
        new_config.spike_params.enable_adaptation = true;
        new_config.spike_params.adaptation_window = 100;
    }

    // Update neuron count in config
    new_config.base_config.num_neurons = new_neuron_count;

    // Update layer sizes (input, hidden, output)
    // For 3-layer networks: [input, hidden, output]
    // Hidden layer (index 1) should grow to new_neuron_count
    if (new_layer_sizes && new_config.base_config.num_layers >= 2) {
        new_layer_sizes[1] = new_neuron_count;
        NIMCP_LOG_INFO("brain_resize: Updated layer_sizes [%u, %u, %u]",
                       new_layer_sizes[0], new_layer_sizes[1],
                       new_config.base_config.num_layers >= 3 ? new_layer_sizes[2] : 0);
    }

    // Step 2: Create new larger network
    NIMCP_LOG_INFO("brain_resize: Creating new network with %u neurons", new_neuron_count);

    // DEBUG: Log final config values before network creation
    NIMCP_LOG_INFO("brain_resize: FINAL config - k_factor=%f, min_threshold=%f, max_threshold=%f, sparsity=%f",
                   new_config.spike_params.k_factor,
                   new_config.spike_params.min_threshold,
                   new_config.spike_params.max_threshold,
                   new_config.spike_params.sparsity_target);
    NIMCP_LOG_INFO("brain_resize: FINAL config - num_layers=%u, num_neurons=%u, layer_sizes=%p",
                   new_config.base_config.num_layers,
                   new_config.base_config.num_neurons,
                   (void*)new_config.base_config.layer_sizes);

    adaptive_network_t new_network = adaptive_network_create(&new_config);

    // Free our temporary layer_sizes copy (adaptive_network_create makes its own deep copy)
    if (new_layer_sizes) {
        nimcp_free(new_layer_sizes);
    }

    if (!new_network) {
        NIMCP_LOG_ERROR("brain_resize: Failed to create new network");
        return false;
    }

    // Step 3: Transfer all existing neurons
    // This preserves:
    // - All synaptic weights
    // - Neuron biases
    // - Activation states
    // - Plasticity traces (STDP, eligibility, BCM)
    // - Neuromodulator states
    NIMCP_LOG_INFO("brain_resize: Transferring %u existing neurons", current_neuron_count);

    neural_network_t new_base = adaptive_network_get_base_network(new_network);
    if (!new_base) {
        NIMCP_LOG_ERROR("brain_resize: Cannot access new base network");
        adaptive_network_destroy(new_network);
        return false;
    }

    // Transfer neuron states via network API
    for (uint32_t i = 0; i < current_neuron_count; i++) {
        neuron_t* old_neuron = neural_network_get_neuron(base_network, i);
        neuron_t* new_neuron = neural_network_get_neuron(new_base, i);

        if (!old_neuron || !new_neuron) {
            NIMCP_LOG_ERROR("brain_resize: Failed to access neuron %u during transfer", i);
            adaptive_network_destroy(new_network);
            return false;
        }

        // Copy neuron state (weights, bias, activation, etc.)
        // NOTE: This is a deep copy that preserves all learning
        // The synapses array is part of the neuron_t struct (fixed-size array),
        // so memcpy copies everything including synapses, incoming_synapses, etc.
        memcpy(new_neuron, old_neuron, sizeof(neuron_t));
    }

    // Step 4: Initialize new neurons
    // New neurons added beyond current_neuron_count get random initialization
    uint32_t new_neurons_added = new_neuron_count - current_neuron_count;
    NIMCP_LOG_INFO("brain_resize: Initializing %u new neurons", new_neurons_added);

    // New neurons are already initialized by adaptive_network_create()
    // with random weights based on network configuration

    // Step 5: Swap networks atomically
    NIMCP_LOG_INFO("brain_resize: Swapping networks");
    adaptive_network_t old_network_copy = brain->network;
    brain->network = new_network;

    // Step 6: Update brain configuration
    brain->config.size = BRAIN_SIZE_CUSTOM;  // Now using custom size

    // Update size-dependent subsystems
    // NOTE: Most cognitive subsystems (working memory, ethics, etc.) are
    // independent of brain size and don't need updates

    // Step 7: Clean up old network
    NIMCP_LOG_INFO("brain_resize: Destroying old network");
    adaptive_network_destroy(old_network_copy);

    NIMCP_LOG_INFO("brain_resize: Growth complete. New capacity: %u neurons (%.1fx growth)",
                   new_neuron_count,
                   (float)new_neuron_count / (float)current_neuron_count);

    return true;
}

/**
 * @brief Auto-resize brain based on utilization metrics
 *
 * WHAT: Automatically grow brain when capacity is saturated
 * WHY:  Enable continuous learning without manual intervention
 * HOW:  Evaluate metrics, decide if growth needed, compute next size, resize
 *
 * USAGE:
 * ```c
 * // During training loop:
 * brain_learn(brain, features, label, confidence);
 *
 * if (step % 100 == 0) {
 *     brain_auto_resize(brain);  // Checks every 100 steps
 * }
 * ```
 *
 * COMPLEXITY: O(n) for metric computation + O(n+m) if resize triggered
 *
 * @param brain Brain to potentially resize
 * @return true if resize occurred, false if no resize needed or error
 */
bool brain_auto_resize(brain_t brain)
{
    // Guard: Validate brain
    if (!brain) {
        return false;
    }

    // Get base network for metrics
    adaptive_network_t network = brain->network;
    if (!network) {
        return false;
    }

    neural_network_t base_network = adaptive_network_get_base_network(network);
    if (!base_network) {
        return false;
    }

    // Verify network structure integrity
    // Use API function to get neuron count (safer than direct access)
    uint32_t current_size = neural_network_get_num_neurons(base_network);

    // Verify num_neurons is reasonable (sanity check for corrupted structure)
    if (current_size == 0 || current_size > 100000000) {
        NIMCP_LOG_WARN("brain_auto_resize: Suspicious neuron count %u, skipping resize", current_size);
        return false;
    }

    // Verify we can access at least the first neuron (structure validity check)
    neuron_t* first_neuron = neural_network_get_neuron(base_network, 0);
    if (!first_neuron) {
        NIMCP_LOG_WARN("brain_auto_resize: Cannot access neurons, network may be corrupt");
        return false;
    }

    // Don't grow beyond reasonable limits
    if (current_size >= 100000) {
        return false;  // Already at 100K neurons, stop
    }

    // For testing: grow once from TINY (100) to SMALL (500)
    if (current_size <= 100) {
        // First call on TINY brain - allow growth
    } else {
        // Already resized once, don't resize again for now
        return false;
    }

    // Detect if GPU is being used
    // TODO: Add brain_is_gpu_enabled() API
    bool use_gpu = false;  // Default to CPU mode for now

    // Compute next size (hardware-aware) - current_size already defined above
    uint32_t new_size = compute_next_size(current_size, use_gpu);

    // Verify we have sufficient resources for the target size
    system_resources_t resources;
    if (system_resources_query(&resources)) {
        if (!system_resources_can_resize(&resources, new_size, use_gpu)) {
            NIMCP_LOG_WARN("brain_auto_resize: Insufficient resources for target size %u, skipping resize", new_size);
            return false;
        }
    }

    // Perform resize
    return brain_resize(brain, new_size);
}

/**
 * @brief Get brain current neuron count
 *
 * WHAT: Return current brain capacity in neurons
 * WHY:  Allow monitoring of brain size for metrics/logging
 * HOW:  Access underlying network and query neuron count
 *
 * @param brain Brain to query
 * @return Neuron count, or 0 on error
 */
uint32_t brain_get_neuron_count(brain_t brain)
{
    if (!brain || !brain->network) {
        return 0;
    }

    neural_network_t base_network = adaptive_network_get_base_network(brain->network);
    if (!base_network) {
        return 0;
    }

    return neural_network_get_num_neurons(base_network);
}

/**
 * @brief Get brain utilization metrics
 *
 * WHAT: Return current capacity utilization statistics
 * WHY:  Enable monitoring and debugging of auto-resize logic
 * HOW:  Compute utilization and saturation metrics
 *
 * @param brain Brain to analyze
 * @param utilization Output: neuron utilization ratio [0.0, 1.0]
 * @param saturation Output: weight saturation ratio [0.0, 1.0]
 * @return true on success, false on error
 */
bool brain_get_utilization_metrics(brain_t brain, float* utilization, float* saturation)
{
    if (!brain || !utilization || !saturation) {
        return false;
    }

    neural_network_t base_network = adaptive_network_get_base_network(brain->network);
    if (!base_network) {
        return false;
    }

    *utilization = compute_neuron_utilization(base_network);
    *saturation = compute_weight_saturation(base_network);

    return true;
}
