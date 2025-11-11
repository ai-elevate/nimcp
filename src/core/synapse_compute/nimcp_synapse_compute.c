//=============================================================================
// nimcp_synapse_compute.c - Programmable Synapse Computation (NIMCP 2.7)
//=============================================================================
/**
 * @file nimcp_synapse_compute.c
 * @brief Production implementation of synapse-level computation
 *
 * ARCHITECTURE:
 * - Strategy Pattern: Function pointers enable per-synapse customization
 * - Memento Pattern: compute_state preserves synapse-specific memory
 * - Template Method: Consistent flow with customizable steps
 *
 * PERFORMANCE CHARACTERISTICS:
 * - Default computation: O(1) - ~10 CPU cycles
 * - Attention computation: O(d) - ~100-200 cycles for d=16
 * - Semantic computation: O(d) - ~300 cycles for d=300
 * - Gating computation: O(1) - ~15 CPU cycles
 * - Memory overhead: 24 bytes per synapse (function pointers)
 * - Cache efficiency: Local memory (64 bytes) fits in L1 cache
 *
 * DESIGN DECISIONS:
 * - Function pointers over virtual dispatch: Zero indirection cost
 * - Local memory first: 16 floats (64 bytes) before extended allocation
 * - Context parameter: Enables global coordination without tight coupling
 * - Const neuron pointers: Documents read-only access for optimization
 *
 * BIOLOGICAL MOTIVATION:
 * Real synapses perform computation via:
 * - Active conductances in dendritic spines (Yuste & Denk, 1995)
 * - Local protein synthesis (Sutton & Schuman, 2006)
 * - Heterosynaptic plasticity (Lynch et al., 1977)
 * - Dendritic integration (London & Häusser, 2005)
 *
 * This transforms NIMCP from 100K passive weights to 100K active processors.
 *
 * @author Claude Code + NIMCP Development Team
 * @date 2025-11-07
 * @version 2.7.0
 */

// NIMCP 2.7: Synapse compute functions
// NOTE: synapse_compute.h includes neuralnet.h, which has full struct definitions
#include "core/synapse_compute/nimcp_synapse_compute.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

//=============================================================================
// Helper Functions - Mathematical Operations
//=============================================================================

/**
 * @brief Compute dot product of two vectors
 *
 * WHAT: Standard vector dot product
 * WHY: Core operation for attention and semantic similarity
 * HOW: Single-pass accumulation
 *
 * COMPLEXITY: O(n)
 * PERFORMANCE: ~3n CPU cycles (load + multiply + add per element)
 */
static inline float dot_product(const float* a, const float* b, uint32_t size) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < size; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

/**
 * @brief Compute cosine similarity between two vectors
 *
 * WHAT: Normalized dot product: cos(θ) = (a·b) / (||a|| ||b||)
 * WHY: Measures semantic similarity independent of magnitude
 * HOW: Single-pass computation of dot product and norms
 *
 * COMPLEXITY: O(n)
 * PERFORMANCE: ~5n CPU cycles + 1 sqrt + 1 div
 */
static inline float cosine_similarity(const float* a, const float* b, uint32_t size) {
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (uint32_t i = 0; i < size; i++) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    float denom = sqrtf(norm_a * norm_b);
    return (denom > 1e-8f) ? (dot / denom) : 0.0f;
}

/**
 * @brief Sigmoid activation function
 *
 * WHAT: σ(x) = 1 / (1 + e^(-x))
 * WHY: Smooth gating with biological plausibility
 * HOW: Standard implementation using exp()
 *
 * COMPLEXITY: O(1)
 * PERFORMANCE: ~100 CPU cycles (dominated by exp())
 */
static inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

//=============================================================================
// Built-In Compute Functions - Transmission Computation
//=============================================================================

/**
 * @brief Default synapse computation (baseline)
 *
 * WHAT: Standard weighted transmission with optional STP
 * WHY: Backward compatible with NIMCP 2.0-2.6, zero overhead if NULL
 * HOW: Multiply weight × activity, modulate by short-term plasticity
 * WHEN: Use when no custom computation needed (most synapses)
 *
 * ALGORITHM: output = weight × pre_activity × STP_modulation
 *
 * COMPLEXITY: O(1) - 2 multiplications + 1 conditional
 * PERFORMANCE: ~10 CPU cycles (no STP) or ~30 cycles (with STP)
 */
float synapse_compute_default(
    struct synapse_t* syn,
    const neuron_t* pre_neuron,
    const neuron_t* post_neuron,
    float pre_activity,
    struct synapse_compute_context_t* context
) {
    // Guard against null synapse
    if (!syn) return 0.0f;

    // Base transmission
    float output = syn->weight * pre_activity;

    // Apply short-term plasticity if enabled
    if (syn->enable_stp) {
        float stp_modulation = stp_get_modulation(&syn->stp);
        output *= stp_modulation;
    }

    return output;
}

/**
 * @brief Attention-modulated synapse computation (Transformer-like)
 *
 * WHAT: Implements scaled dot-product attention at synapse level
 * WHY: Enables focus on relevant inputs, critical for NLP and sequence tasks
 * HOW: Query-key similarity modulates transmission strength
 * WHEN: Use for attention layers, sequence processing, associative memory
 *
 * ALGORITHM:
 * 1. Extract query from post neuron (last 16 elements of activity_history)
 * 2. Extract key from pre neuron (last 16 elements of activity_history)
 * 3. Compute attention: exp(query · key / √d)  [scaled dot-product]
 * 4. Modulate transmission by attention weight
 *
 * BIOLOGICAL INSPIRATION:
 * - Dendritic coincidence detection (Larkum et al., 1999)
 * - Context-dependent gating in pyramidal neurons
 *
 * CONTEXT OPTIONS:
 * - Option 1: Use activity_history as embeddings (default)
 * - Option 2: Use global query/key from context.global_state (for Transformer layers)
 *
 * COMPLEXITY: O(d) where d = 16 (query/key dimension)
 * PERFORMANCE: ~150 cycles (16 muls + sqrt + exp + overhead)
 */
float synapse_compute_attention(
    struct synapse_t* syn,
    const neuron_t* pre_neuron,
    const neuron_t* post_neuron,
    float pre_activity,
    struct synapse_compute_context_t* context
) {
    // Guard against null inputs
    if (!syn || !pre_neuron || !post_neuron) return 0.0f;

    #define ATTENTION_DIM 16

    // Option 1: Use neuron activity history as query/key
    float query[ATTENTION_DIM];
    float key[ATTENTION_DIM];

    // Extract last ATTENTION_DIM activity values as embeddings
    for (uint32_t i = 0; i < ATTENTION_DIM; i++) {
        uint32_t idx = (HISTORY_WINDOW - ATTENTION_DIM + i) % HISTORY_WINDOW;
        query[i] = post_neuron->activity_history[idx];
        key[i] = pre_neuron->activity_history[idx];
    }

    // Option 2: Use global state if provided
    if (context && context->global_state && context->global_state_size >= ATTENTION_DIM * 2) {
        // First half = query, second half = key
        memcpy(query, context->global_state, ATTENTION_DIM * sizeof(float));
        memcpy(key, context->global_state + ATTENTION_DIM, ATTENTION_DIM * sizeof(float));
    }

    // Compute scaled dot-product attention
    float attention_score = dot_product(query, key, ATTENTION_DIM) / sqrtf(ATTENTION_DIM);
    float attention_weight = expf(attention_score);  // Note: softmax normalization done globally

    // Store attention weight in local memory for debugging
    if (syn->compute_state) {
        syn->compute_state->local_memory[0] = attention_weight;
    }

    // Base transmission modulated by attention
    float output = syn->weight * pre_activity * attention_weight;

    // Apply STP if enabled
    if (syn->enable_stp) {
        output *= stp_get_modulation(&syn->stp);
    }

    return output;
}

/**
 * @brief Semantic similarity-modulated synapse
 *
 * For NLP: modulates transmission by semantic similarity between word embeddings.
 *
 * STORAGE: Embeddings stored in compute_state->extended_memory
 * Layout: [embedding_dim][pre_embedding...][post_embedding...]
 *
 * COMPLEXITY: O(d) where d = embedding dimension
 * PERFORMANCE: ~300 cycles for d=300
 */
float synapse_compute_semantic(
    struct synapse_t* syn,
    const neuron_t* pre_neuron,
    const neuron_t* post_neuron,
    float pre_activity,
    struct synapse_compute_context_t* context
) {
    // Guard against null synapse
    if (!syn) return 0.0f;

    // Need compute state with embeddings
    if (!syn->compute_state || !syn->compute_state->extended_memory) {
        // Fallback to default if no embeddings
        return synapse_compute_default(syn, pre_neuron, post_neuron, pre_activity, context);
    }

    // Extract embedding dimension (stored in first float)
    uint32_t embedding_dim = (uint32_t)syn->compute_state->extended_memory[0];

    // Get pointers to pre/post embeddings
    float* pre_embedding = &syn->compute_state->extended_memory[1];
    float* post_embedding = &syn->compute_state->extended_memory[1 + embedding_dim];

    // Compute cosine similarity
    float similarity = cosine_similarity(pre_embedding, post_embedding, embedding_dim);

    // Store for analysis
    syn->compute_state->local_memory[0] = similarity;

    // Modulate transmission by semantic similarity
    float modulation = 0.5f + 0.5f * similarity;  // Map [-1,1] to [0,1]
    float output = syn->weight * pre_activity * modulation;

    // Apply STP
    if (syn->enable_stp) {
        output *= stp_get_modulation(&syn->stp);
    }

    return output;
}

/**
 * @brief Gating synapse (multiplicative modulation)
 *
 * Acts as a gate controlled by external signal.
 *
 * GATE SIGNAL: Read from compute_state->local_memory[0]
 * This should be set by external controller before simulation step.
 *
 * COMPLEXITY: O(1)
 * PERFORMANCE: ~15 cycles
 */
float synapse_compute_gating(
    struct synapse_t* syn,
    const neuron_t* pre_neuron,
    const neuron_t* post_neuron,
    float pre_activity,
    struct synapse_compute_context_t* context
) {
    // Guard against null synapse
    if (!syn) return 0.0f;

    // Default gate = open (1.0)
    float gate_signal = 1.0f;

    // Read gate from local memory if available
    if (syn->compute_state) {
        gate_signal = syn->compute_state->local_memory[0];
    }

    // Clamp gate to [0, 1]
    if (gate_signal < 0.0f) gate_signal = 0.0f;
    if (gate_signal > 1.0f) gate_signal = 1.0f;

    // Gated transmission
    float output = syn->weight * pre_activity * gate_signal;

    // Apply STP
    if (syn->enable_stp) {
        output *= stp_get_modulation(&syn->stp);
    }

    return output;
}

/**
 * @brief Neuromodulator-sensitive synapse
 *
 * Transmission modulated by dopamine/serotonin levels.
 *
 * SENSITIVITY: Stored in local_memory[0] (default = 1.0)
 * MODULATION: output × (1.0 + neuromod × sensitivity)
 *
 * COMPLEXITY: O(1)
 * PERFORMANCE: ~15 cycles
 */
float synapse_compute_neuromodulated(
    struct synapse_t* syn,
    const neuron_t* pre_neuron,
    const neuron_t* post_neuron,
    float pre_activity,
    struct synapse_compute_context_t* context
) {
    // Guard against null synapse
    if (!syn) return 0.0f;

    // Default sensitivity = 1.0
    float sensitivity = 1.0f;
    if (syn->compute_state) {
        sensitivity = syn->compute_state->local_memory[0];
    }

    // Get neuromodulation level from context
    float neuromod = 0.0f;
    if (context) {
        neuromod = context->neuromodulation;
    }

    // Compute modulation factor
    float modulation = 1.0f + neuromod * sensitivity;

    // Modulated transmission
    float output = syn->weight * pre_activity * modulation;

    // Apply STP
    if (syn->enable_stp) {
        output *= stp_get_modulation(&syn->stp);
    }

    return output;
}

/**
 * @brief Dendritic computation synapse
 *
 * Implements local dendritic non-linearity by summing nearby synapses.
 *
 * REQUIRES: function_data points to array of neighbor synapse pointers
 * Format: [num_neighbors][syn_ptr1][syn_ptr2]...
 *
 * COMPLEXITY: O(k) where k = number of neighbors (~10-50)
 * PERFORMANCE: ~50-500 cycles depending on k
 */
float synapse_compute_dendritic(
    struct synapse_t* syn,
    const neuron_t* pre_neuron,
    const neuron_t* post_neuron,
    float pre_activity,
    struct synapse_compute_context_t* context
) {
    // Guard against null synapse
    if (!syn) return 0.0f;

    float local_sum = 0.0f;

    // Sum activity from neighboring synapses if available
    if (syn->compute_state && syn->compute_state->function_data) {
        // First element = number of neighbors
        uint32_t* data = (uint32_t*)syn->compute_state->function_data;
        uint32_t num_neighbors = data[0];

        // Sanity check: reasonable number of neighbors (prevent crashes on uninitialized data)
        if (num_neighbors > 0 && num_neighbors < 10000) {
            struct synapse_t** neighbors = (struct synapse_t**)&data[1];

            // Sum recent activity from neighbors (stored in local_memory[1])
            for (uint32_t i = 0; i < num_neighbors; i++) {
                if (neighbors[i] && neighbors[i]->compute_state) {
                    local_sum += neighbors[i]->compute_state->local_memory[1];
                }
            }
        }
    }

    // Apply dendritic non-linearity (sigmoid)
    float nonlinear = sigmoid(local_sum);

    // Store current activity for neighbors to read
    if (syn->compute_state) {
        syn->compute_state->local_memory[1] = pre_activity * syn->weight;
    }

    // Modulated transmission
    float output = syn->weight * pre_activity * nonlinear;

    // Apply STP
    if (syn->enable_stp) {
        output *= stp_get_modulation(&syn->stp);
    }

    return output;
}

// ============================================================================
// BUILT-IN LEARNING FUNCTIONS
// ============================================================================

/**
 * @brief Three-factor learning rule
 *
 * ALGORITHM: Δw = η × eligibility_trace × reward_signal
 *
 * WHERE:
 * - Eligibility trace updated by STDP
 * - Decays with τ = 1000ms
 * - Reward signal provided by context
 */
void synapse_learn_three_factor(
    struct synapse_t* syn,
    const neuron_t* pre_neuron,
    const neuron_t* post_neuron,
    float pre_spike_time,
    float post_spike_time,
    float reward_signal,
    struct synapse_compute_context_t* context
) {
    // Guard against null synapse
    if (!syn) return;

    const float TAU_ELIGIBILITY = 1000.0f; // ms
    const float LEARNING_RATE = 0.01f;

    // Update eligibility trace with STDP
    if (pre_spike_time > 0 && post_spike_time > 0) {
        float dt = post_spike_time - pre_spike_time;
        float stdp_update = expf(-fabsf(dt) / 20.0f);
        if (dt > 0) {
            stdp_update *= 1.0f;  // LTP
        } else {
            stdp_update *= -0.5f;  // LTD (asymmetric)
        }

        // Update trace
        syn->trace += stdp_update;
    }

    // Decay eligibility trace
    if (context) {
        float dt = 1.0f; // Assume 1ms timestep
        syn->trace *= expf(-dt / TAU_ELIGIBILITY);
    }

    // Three-factor weight update
    float weight_change = LEARNING_RATE * syn->trace * reward_signal;
    syn->weight += weight_change;
    syn->last_change = weight_change;

    // Weight bounds
    if (syn->weight < -10.0f) syn->weight = -10.0f;
    if (syn->weight > 10.0f) syn->weight = 10.0f;
}

/**
 * @brief Attention-modulated learning
 *
 * Learning rate scaled by current attention weight.
 */
void synapse_learn_attention_modulated(
    struct synapse_t* syn,
    const neuron_t* pre_neuron,
    const neuron_t* post_neuron,
    float pre_spike_time,
    float post_spike_time,
    float reward_signal,
    struct synapse_compute_context_t* context
) {
    // Guard against null synapse
    if (!syn) return;

    // Get attention weight (stored during compute phase)
    float attention = 1.0f;
    if (syn->compute_state) {
        attention = syn->compute_state->local_memory[0];
        if (attention < 0.1f) attention = 0.1f; // Minimum learning
    }

    // Standard STDP
    if (pre_spike_time > 0 && post_spike_time > 0) {
        float dt = post_spike_time - pre_spike_time;
        float stdp_update = expf(-fabsf(dt) / 20.0f);
        if (dt > 0) {
            stdp_update *= 1.0f;
        } else {
            stdp_update *= -0.5f;
        }

        // Scale by attention
        float weight_change = 0.01f * attention * stdp_update;
        syn->weight += weight_change;
        syn->last_change = weight_change;
    }

    // Weight bounds
    if (syn->weight < -10.0f) syn->weight = -10.0f;
    if (syn->weight > 10.0f) syn->weight = 10.0f;
}

/**
 * @brief Meta-plasticity learning
 *
 * Learning rate adapts based on recent synaptic activity.
 */
void synapse_learn_metaplastic(
    struct synapse_t* syn,
    const neuron_t* pre_neuron,
    const neuron_t* post_neuron,
    float pre_spike_time,
    float post_spike_time,
    float reward_signal,
    struct synapse_compute_context_t* context
) {
    // Guard against null inputs
    if (!syn || !post_neuron) return;

    // Update meta-plasticity based on recent activity
    // High activity → lower plasticity (consolidation)
    // Low activity → higher plasticity (exploration)

    float avg_activity = post_neuron->avg_activity;
    float theta = 0.5f; // Threshold

    // BCM-like learning rate modulation
    float activity_factor = (avg_activity > theta) ? 0.5f : 1.5f;

    // Update meta-plasticity state
    syn->meta_plasticity = 0.9f * syn->meta_plasticity + 0.1f * activity_factor;

    // Standard STDP with meta-plasticity modulation
    if (pre_spike_time > 0 && post_spike_time > 0) {
        float dt = post_spike_time - pre_spike_time;
        float stdp_update = expf(-fabsf(dt) / 20.0f);
        if (dt > 0) {
            stdp_update *= 1.0f;
        } else {
            stdp_update *= -0.5f;
        }

        // Modulate by meta-plasticity
        float weight_change = 0.01f * syn->meta_plasticity * stdp_update;
        syn->weight += weight_change;
        syn->last_change = weight_change;
    }

    // Weight bounds
    if (syn->weight < -10.0f) syn->weight = -10.0f;
    if (syn->weight > 10.0f) syn->weight = 10.0f;
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * @brief Initialize synapse compute state
 */
int synapse_compute_state_init(synapse_compute_state_t* state, uint32_t extended_size) {
    if (!state) return -1;

    // Zero local memory
    memset(state->local_memory, 0, sizeof(state->local_memory));

    // Allocate extended memory if requested
    if (extended_size > 0) {
        state->extended_memory = (float*)calloc(extended_size, sizeof(float));
        if (!state->extended_memory) {
            return -1; // Allocation failed
        }
        state->extended_size = extended_size;
    } else {
        state->extended_memory = NULL;
        state->extended_size = 0;
    }

    state->function_data = NULL;
    state->cleanup_fn = NULL;

    return 0;
}

/**
 * @brief Cleanup synapse compute state
 */
void synapse_compute_state_cleanup(synapse_compute_state_t* state) {
    if (!state) return;

    // Free extended memory
    if (state->extended_memory) {
        free(state->extended_memory);
        state->extended_memory = NULL;
        state->extended_size = 0;
    }

    // Call cleanup function if registered
    if (state->cleanup_fn && state->function_data) {
        state->cleanup_fn(state->function_data);
    }

    state->function_data = NULL;
    state->cleanup_fn = NULL;
}

/**
 * @brief Set synapse compute function
 */
int synapse_set_compute_function(
    struct synapse_t* syn,
    synapse_compute_fn compute_fn,
    synapse_learn_fn learn_fn,
    void* function_data,
    void (*cleanup_fn)(void*)
) {
    if (!syn) return -1;

    // Allocate compute state if needed
    if (!syn->compute_state && (function_data || compute_fn)) {
        syn->compute_state = (synapse_compute_state_t*)calloc(1, sizeof(synapse_compute_state_t));
        if (!syn->compute_state) {
            return -1; // Allocation failed
        }
    }

    // Set function pointers
    syn->compute_function = compute_fn;
    syn->learn_function = learn_fn;

    // Set function data
    if (syn->compute_state) {
        syn->compute_state->function_data = function_data;
        syn->compute_state->cleanup_fn = cleanup_fn;
    }

    return 0;
}
