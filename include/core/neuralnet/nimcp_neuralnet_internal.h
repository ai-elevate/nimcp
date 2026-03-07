/**
 * @file nimcp_neuralnet_internal.h
 * @brief SINGLE authoritative definition of neural_network_struct
 *
 * WARNING: This is the ONLY place neural_network_struct should be defined.
 * All files needing access to the internal struct MUST include this header.
 * DO NOT duplicate this struct definition in any other file.
 *
 * After changing neuron_t or neural_network_struct, run: make -j4 (full rebuild)
 *
 * HISTORY:
 * - Consolidated from 7+ duplicate definitions (2026-02-26)
 * - Several duplicates were stale (missing sparse synapse pools, bulk allocation,
 *   wrong activation_strategy_table_t sizes, bogus immune_system field)
 */
#ifndef NIMCP_NEURALNET_INTERNAL_H
#define NIMCP_NEURALNET_INTERNAL_H

#include "core/neuralnet/nimcp_neuralnet.h"       /* neuron_t, network_config_t, neural_network_t */
#include "core/neuralnet/nimcp_sparse_synapse.h"   /* sparse_synapse_pool_t, synapse_metadata_pool_t */
#include "async/nimcp_bio_async.h"                 /* bio_module_context_t */

/**
 * @brief Function pointer type for activation strategies
 *
 * WHY: Eliminates switch statements, enables O(1) dispatch
 * Follows Strategy pattern - each activation type is its own algorithm
 *
 * @param input Raw input value
 * @param threshold Adaptive threshold for spike-based activations
 * @return Activated output value
 */
typedef float (*activation_fn_t)(float input, float threshold);

/**
 * @brief Activation function strategy table
 *
 * INVARIANT: All entries must be non-NULL after initialization
 * PATTERN: Strategy pattern - function pointer dispatch
 */
typedef struct {
    activation_fn_t functions[8];  /* Max activation types */
} activation_strategy_table_t;

/**
 * @brief Internal neural network structure
 *
 * INVARIANTS:
 * - num_neurons <= MAX_NEURONS
 * - current_time <= network_time
 * - 0.0 <= global_activity <= 1.0
 * - 0.0 <= network_stability <= 1.0
 *
 * PATTERN: Handle/Body idiom - opaque pointer hides implementation
 * COMPLEXITY: O(1) access to all fields
 *
 * WARNING: Field order matters! Adding fields MUST be done at the end,
 * before the closing brace. Reordering fields will break all compiled
 * modules until a full rebuild.
 */
struct neural_network_struct {
    neuron_t* neurons;     /**< Dynamically allocated neurons array */
    uint32_t num_neurons;  /**< Current number of active neurons */
    uint32_t capacity;     /**< Allocated capacity (for add_neuron) */
    uint64_t current_time;
    network_config_t config;
    uint64_t network_time;
    float global_activity;
    float network_stability;
    float learning_momentum;
    float last_avg_weight;
    uint64_t last_maintenance;
    activation_strategy_table_t activation_strategies;

    /* NIMCP 2.7: NLP Integration - Subsystems for synapse compute context */
    void* neuromodulator_system;  /**< Neuromodulator system (opaque pointer) */
    float* global_state;          /**< Global state for synapse computation (attention output, etc) */
    uint32_t global_state_size;   /**< Size of global state buffer */

    /* NIMCP Phase 6: Glial Integration - Neuro-glial signaling */
    void* glial_integration;      /**< Glial integration system (opaque pointer) */

    /* Axon Integration - Signal propagation with realistic conduction delays */
    void* axon_network;           /**< Axon network for spike propagation (axon_network_t*, NULL = no axons) */

    /* Bio-async integration */
    bio_module_context_t bio_ctx;
    bool bio_async_enabled;

    /* Sparse synapse pools (NIMCP 2.11: Memory-efficient synapse storage) */
    sparse_synapse_pool_t synapse_handle_pool;
    synapse_metadata_pool_t synapse_metadata_pool;

    /* Bulk allocation for large networks - reduces 2M tracked allocs to 2 */
    spike_record_t* spike_history_bulk;   /**< Single contiguous spike history allocation */
    float* activity_history_bulk;         /**< Single contiguous activity history allocation */
    uint32_t bulk_neuron_count;           /**< Neurons using bulk arrays (0 = individual allocs) */

    /* Phase: Active Neuron Set Tracking (40-watt brain optimization)
     * Only 1-5% of neurons fire at any time. Track active set to skip dormant neurons.
     * active_neuron_ids: pool of neuron IDs that fired or received input last step
     * num_active_neurons: count of active neurons from last compute_step
     * active_set_valid: whether the active set is up to date */
    uint32_t* active_neuron_ids;      /**< Pool of active neuron IDs */
    uint32_t num_active_neurons;      /**< Count of active neurons last step */
    uint32_t active_neuron_capacity;  /**< Allocated capacity for active set */
    bool active_set_valid;            /**< Whether active set is current */
};

#endif /* NIMCP_NEURALNET_INTERNAL_H */
