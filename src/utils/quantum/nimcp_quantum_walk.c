//=============================================================================
// nimcp_quantum_walk.c - Quantum Walk Implementation
//=============================================================================

#include "utils/quantum/nimcp_quantum_walk.h"
#include "core/neuralnet/nimcp_neuralnet.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/time/nimcp_time.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

// Mathematical constants
#define SQRT2 1.41421356237f
#define INV_SQRT2 0.70710678118f
#define PI 3.14159265359f
#define PROB_TOLERANCE 1e-6f      // Probability conservation tolerance
#define AMPLITUDE_THRESHOLD 1e-8f // Ignore amplitudes below this

//=============================================================================
// Complex Number Utilities
//=============================================================================

void quantum_normalize(quantum_amplitude_t* amplitudes, uint32_t size) {
    // Guard: NULL checks
    if (!amplitudes || size == 0) return;

    // STEP 1: Compute norm squared
    // WHAT: Calculate Σ|αᵢ|²
    // WHY: Need total probability for normalization
    // HOW: Sum real² + imag² for all amplitudes
    float norm_squared = 0.0f;
    for (uint32_t i = 0; i < size; i++) {
        float re = crealf(amplitudes[i]);
        float im = cimagf(amplitudes[i]);
        norm_squared += re * re + im * im;
    }

    // Guard: Avoid division by zero
    if (norm_squared < 1e-12f) {
        // State is essentially zero → uniform distribution
        quantum_amplitude_t uniform = 1.0f / sqrtf((float)size);
        for (uint32_t i = 0; i < size; i++) {
            amplitudes[i] = uniform;
        }
        return;
    }

    // STEP 2: Normalize
    // WHAT: Scale all amplitudes by 1/√(Σ|αᵢ|²)
    // WHY: Ensure Σ|αᵢ|² = 1 (probability conservation)
    // HOW: Multiply each amplitude by normalization factor
    float norm = sqrtf(norm_squared);
    float inv_norm = 1.0f / norm;

    for (uint32_t i = 0; i < size; i++) {
        amplitudes[i] *= inv_norm;
    }
}

//=============================================================================
// Configuration Functions
//=============================================================================

quantum_walk_config_t quantum_walk_default_config(void) {
    quantum_walk_config_t config = {
        .coin_type = COIN_HADAMARD,        // Balanced superposition
        .num_steps = 100,                  // 100 evolution steps
        .hybrid_mixing = 0.0f,             // Pure quantum (no classical mixing)
        .normalize_each_step = true,       // Maintain probability conservation
        .decoherence_rate = 0.01f,         // Minimal decoherence (1% per step)
        .measurement_interval = 0,         // Only measure at end
        .enable_boundary_conditions = true // Reflect at boundaries
    };
    return config;
}

quantum_walk_config_t quantum_walk_fast_config(void) {
    quantum_walk_config_t config = quantum_walk_default_config();
    config.coin_type = COIN_GROVER;    // Faster spreading
    config.num_steps = 50;             // Fewer steps
    config.decoherence_rate = 0.0f;    // No decoherence (pure quantum)
    config.normalize_each_step = false; // Skip normalization for speed
    return config;
}

quantum_walk_config_t quantum_walk_hybrid_config(void) {
    quantum_walk_config_t config = quantum_walk_default_config();
    config.hybrid_mixing = 0.5f;       // 50% quantum + 50% classical
    config.decoherence_rate = 0.05f;   // Higher decoherence (5% per step)
    return config;
}

//=============================================================================
// Helper Functions
//=============================================================================

/**
 * @brief Build adjacency list from neural network
 *
 * WHAT: Extract graph structure from network synapses
 * WHY: Fast neighbor lookup during quantum walk
 * HOW: Iterate synapses, build adjacency lists
 *
 * COMPLEXITY: O(N + E)
 */
static bool build_adjacency_lists(
    quantum_walker_t* walker,
    neural_network_t network
) {
    // Get number of neurons
    uint32_t num_neurons = neural_network_get_num_neurons(network);
    if (num_neurons == 0) return false;

    walker->num_nodes = num_neurons;

    // Allocate degree array
    walker->node_degrees = (uint32_t*)nimcp_calloc(num_neurons, sizeof(uint32_t));
    if (!walker->node_degrees) return false;

    // PASS 1: Count degrees
    // WHAT: Count outgoing edges for each neuron
    // WHY: Need to allocate adjacency lists
    // HOW: Iterate all synapses, increment source neuron degree
    for (uint32_t neuron_id = 0; neuron_id < num_neurons; neuron_id++) {
        // Get neuron and count its outgoing synapses
        neuron_t* neuron = neural_network_get_neuron(network, neuron_id);
        uint32_t num_synapses = neuron ? neuron->num_synapses : 0;
        walker->node_degrees[neuron_id] = num_synapses;
    }

    // PASS 2: Allocate and fill adjacency lists
    walker->adjacency_list = (uint32_t**)nimcp_malloc(num_neurons * sizeof(uint32_t*));
    if (!walker->adjacency_list) {
        nimcp_free(walker->node_degrees);
        return false;
    }

    for (uint32_t i = 0; i < num_neurons; i++) {
        uint32_t degree = walker->node_degrees[i];
        if (degree > 0) {
            walker->adjacency_list[i] = (uint32_t*)nimcp_malloc(degree * sizeof(uint32_t));
            if (!walker->adjacency_list[i]) {
                // Cleanup on failure
                for (uint32_t j = 0; j < i; j++) {
                    if (walker->adjacency_list[j]) nimcp_free(walker->adjacency_list[j]);
                }
                nimcp_free(walker->adjacency_list);
                nimcp_free(walker->node_degrees);
                return false;
            }

            // Fill adjacency list with target neuron IDs
            // TODO: Need API to get synapse targets
            // For now, assume fully connected small-world network
            uint32_t idx = 0;
            for (uint32_t j = 0; j < num_neurons && idx < degree; j++) {
                if (i != j) { // No self-loops
                    walker->adjacency_list[i][idx++] = j;
                }
            }
        } else {
            walker->adjacency_list[i] = NULL;
        }
    }

    return true;
}

/**
 * @brief Apply Hadamard coin operator
 *
 * WHAT: H = (1/√2) [1  1]
 *                  [1 -1]
 * WHY: Create balanced superposition
 * HOW: α' = (α + β)/√2, β' = (α - β)/√2
 */
static inline void apply_hadamard_coin(
    quantum_amplitude_t* amplitudes,
    uint32_t node_id,
    uint32_t degree
) {
    // For single walker: α' = H×α = α/√2 (simplified)
    // Full implementation would need edge-specific amplitudes
    amplitudes[node_id] *= INV_SQRT2;
}

/**
 * @brief Apply Grover coin operator
 *
 * WHAT: G = (2/N)J - I (J = all-ones matrix)
 * WHY: Bias toward uniform distribution
 * HOW: α' = (2/N)Σαᵢ - α
 */
static void apply_grover_coin(
    quantum_amplitude_t* amplitudes,
    uint32_t num_nodes
) {
    // Compute average amplitude
    quantum_amplitude_t avg = 0.0f;
    for (uint32_t i = 0; i < num_nodes; i++) {
        avg += amplitudes[i];
    }
    avg /= (float)num_nodes;

    // Apply Grover diffusion: α' = 2×avg - α
    for (uint32_t i = 0; i < num_nodes; i++) {
        amplitudes[i] = 2.0f * avg - amplitudes[i];
    }
}

/**
 * @brief Apply Fourier coin operator
 *
 * WHAT: DFT_N (Discrete Fourier Transform)
 * WHY: Phase-dependent mixing
 * HOW: α'ⱼ = (1/√N)Σₖ exp(2πijk/N)×αₖ
 */
static void apply_fourier_coin(
    quantum_amplitude_t* amplitudes,
    uint32_t num_nodes
) {
    if (num_nodes == 0) return;

    // Allocate temporary buffer
    quantum_amplitude_t* temp = (quantum_amplitude_t*)nimcp_calloc(
        num_nodes, sizeof(quantum_amplitude_t)
    );
    if (!temp) return;

    // Compute DFT
    float inv_sqrt_n = 1.0f / sqrtf((float)num_nodes);
    for (uint32_t j = 0; j < num_nodes; j++) {
        quantum_amplitude_t sum = 0.0f;
        for (uint32_t k = 0; k < num_nodes; k++) {
            float phase = 2.0f * PI * (float)(j * k) / (float)num_nodes;
            quantum_amplitude_t twiddle = cosf(phase) + I * sinf(phase);
            sum += amplitudes[k] * twiddle;
        }
        temp[j] = sum * inv_sqrt_n;
    }

    // Copy back
    memcpy(amplitudes, temp, num_nodes * sizeof(quantum_amplitude_t));
    nimcp_free(temp);
}

//=============================================================================
// Lifecycle Functions
//=============================================================================

quantum_walker_t* quantum_walk_create(
    neural_network_t network,
    const quantum_walk_config_t* config
) {
    // Guard: NULL checks
    if (!network || !config) return NULL;

    // Allocate walker structure
    quantum_walker_t* walker = (quantum_walker_t*)nimcp_malloc(sizeof(quantum_walker_t));
    if (!walker) return NULL;

    // Initialize fields
    memset(walker, 0, sizeof(quantum_walker_t));
    walker->network = network;
    walker->config = *config;

    // Build adjacency lists from network
    if (!build_adjacency_lists(walker, network)) {
        nimcp_free(walker);
        return NULL;
    }

    // Allocate amplitude arrays
    walker->amplitudes = (quantum_amplitude_t*)nimcp_calloc(
        walker->num_nodes, sizeof(quantum_amplitude_t)
    );
    if (!walker->amplitudes) {
        quantum_walk_destroy(walker);
        return NULL;
    }

    walker->probabilities = (float*)nimcp_calloc(
        walker->num_nodes, sizeof(float)
    );
    if (!walker->probabilities) {
        quantum_walk_destroy(walker);
        return NULL;
    }

    walker->temp_amplitudes = (quantum_amplitude_t*)nimcp_calloc(
        walker->num_nodes, sizeof(quantum_amplitude_t)
    );
    if (!walker->temp_amplitudes) {
        quantum_walk_destroy(walker);
        return NULL;
    }

    // Initialize statistics
    walker->stats.total_probability = 1.0f;
    walker->stats.speedup_vs_classical = sqrtf((float)walker->num_nodes);

    return walker;
}

void quantum_walk_destroy(quantum_walker_t* walker) {
    if (!walker) return;

    // Free adjacency lists
    if (walker->adjacency_list) {
        for (uint32_t i = 0; i < walker->num_nodes; i++) {
            if (walker->adjacency_list[i]) {
                nimcp_free(walker->adjacency_list[i]);
            }
        }
        nimcp_free(walker->adjacency_list);
    }

    if (walker->node_degrees) nimcp_free(walker->node_degrees);
    if (walker->amplitudes) nimcp_free(walker->amplitudes);
    if (walker->probabilities) nimcp_free(walker->probabilities);
    if (walker->temp_amplitudes) nimcp_free(walker->temp_amplitudes);

    nimcp_free(walker);
}

quantum_walker_t* quantum_walk_clone(const quantum_walker_t* walker) {
    if (!walker) return NULL;

    // Create new walker with same configuration
    quantum_walker_t* clone = quantum_walk_create(walker->network, &walker->config);
    if (!clone) return NULL;

    // Copy quantum state
    memcpy(clone->amplitudes, walker->amplitudes,
           walker->num_nodes * sizeof(quantum_amplitude_t));
    memcpy(clone->probabilities, walker->probabilities,
           walker->num_nodes * sizeof(float));

    clone->current_step = walker->current_step;
    clone->initial_node = walker->initial_node;
    clone->stats = walker->stats;

    return clone;
}

//=============================================================================
// Quantum Walk Operations
//=============================================================================

bool quantum_walk_initialize(quantum_walker_t* walker, uint32_t node_id) {
    // Guard: NULL check and bounds
    if (!walker) return false;
    if (node_id >= walker->num_nodes) return false;

    // WHAT: Set |ψ⟩ = |node_id⟩
    // WHY: Localized initial state (e.g., reward source)
    // HOW: amplitude[node_id] = 1.0, rest = 0.0

    // Zero all amplitudes
    memset(walker->amplitudes, 0, walker->num_nodes * sizeof(quantum_amplitude_t));

    // Set single node to 1.0 (real amplitude)
    walker->amplitudes[node_id] = 1.0f + 0.0f * I;

    // Update probabilities
    memset(walker->probabilities, 0, walker->num_nodes * sizeof(float));
    walker->probabilities[node_id] = 1.0f;

    // Reset state
    walker->current_step = 0;
    walker->initial_node = node_id;

    return true;
}

bool quantum_walk_initialize_superposition(
    quantum_walker_t* walker,
    const quantum_amplitude_t* initial_amplitudes
) {
    // Guard: NULL check
    if (!walker) return false;

    if (initial_amplitudes == NULL) {
        // Uniform superposition: |ψ⟩ = (1/√N)Σ|i⟩
        quantum_amplitude_t uniform = 1.0f / sqrtf((float)walker->num_nodes);
        for (uint32_t i = 0; i < walker->num_nodes; i++) {
            walker->amplitudes[i] = uniform;
        }
    } else {
        // Copy provided amplitudes
        memcpy(walker->amplitudes, initial_amplitudes,
               walker->num_nodes * sizeof(quantum_amplitude_t));
    }

    // Normalize
    quantum_normalize(walker->amplitudes, walker->num_nodes);

    // Update probabilities
    for (uint32_t i = 0; i < walker->num_nodes; i++) {
        walker->probabilities[i] = quantum_probability(walker->amplitudes[i]);
    }

    walker->current_step = 0;

    return true;
}

bool quantum_walk_step(quantum_walker_t* walker) {
    // Guard: NULL check
    if (!walker) return false;

    uint64_t start_time = nimcp_time_get_us();

    // ALGORITHM: Quantum walk = Coin + Shift operators
    //
    // U = S × C
    // C = Coin operator (create superposition)
    // S = Shift operator (move along edges)

    // STEP 1: Apply coin operator
    // WHAT: Create quantum superposition at each node
    // WHY: Enable quantum interference
    // HOW: Apply unitary transformation based on coin_type
    switch (walker->config.coin_type) {
        case COIN_HADAMARD:
            for (uint32_t i = 0; i < walker->num_nodes; i++) {
                if (walker->node_degrees[i] > 0) {
                    apply_hadamard_coin(walker->amplitudes, i, walker->node_degrees[i]);
                }
            }
            break;

        case COIN_GROVER:
            apply_grover_coin(walker->amplitudes, walker->num_nodes);
            break;

        case COIN_FOURIER:
            apply_fourier_coin(walker->amplitudes, walker->num_nodes);
            break;

        case COIN_IDENTITY:
            // No coin operation (classical limit)
            break;
    }

    // STEP 2: Apply shift operator
    // WHAT: Move amplitude along graph edges
    // WHY: Propagate quantum state through network
    // HOW: For each node j, accumulate amplitudes from neighbors i
    //
    // α'ⱼ = Σᵢ∈neighbors(j) αᵢ / √degree(i)

    // Zero temp buffer
    memset(walker->temp_amplitudes, 0, walker->num_nodes * sizeof(quantum_amplitude_t));

    // Propagate amplitudes
    for (uint32_t i = 0; i < walker->num_nodes; i++) {
        if (walker->node_degrees[i] == 0) continue;

        quantum_amplitude_t amplitude = walker->amplitudes[i];
        float weight = 1.0f / sqrtf((float)walker->node_degrees[i]);

        // Distribute amplitude to neighbors
        for (uint32_t n = 0; n < walker->node_degrees[i]; n++) {
            uint32_t neighbor_id = walker->adjacency_list[i][n];
            walker->temp_amplitudes[neighbor_id] += amplitude * weight;
        }
    }

    // Copy temp to main
    memcpy(walker->amplitudes, walker->temp_amplitudes,
           walker->num_nodes * sizeof(quantum_amplitude_t));

    // STEP 3: Apply decoherence (if enabled)
    // WHAT: Add noise to quantum state
    // WHY: Model environmental decoherence
    // HOW: Mix with random phases
    if (walker->config.decoherence_rate > 0.0f) {
        quantum_walk_apply_decoherence(walker, walker->config.decoherence_rate);
    }

    // STEP 4: Normalize (if enabled)
    // WHAT: Ensure Σ|αᵢ|² = 1
    // WHY: Probability conservation
    // HOW: Scale all amplitudes
    if (walker->config.normalize_each_step) {
        quantum_normalize(walker->amplitudes, walker->num_nodes);
    }

    // STEP 5: Update probabilities
    // WHAT: Compute P(i) = |αᵢ|²
    // WHY: Cache for fast access
    // HOW: Square magnitude
    for (uint32_t i = 0; i < walker->num_nodes; i++) {
        walker->probabilities[i] = quantum_probability(walker->amplitudes[i]);
    }

    // Update statistics
    walker->current_step++;
    uint64_t end_time = nimcp_time_get_us();
    walker->stats.evolution_time_us += (end_time - start_time);

    return true;
}

bool quantum_walk_evolve(quantum_walker_t* walker, uint32_t num_steps) {
    // Guard: NULL check
    if (!walker) return false;

    for (uint32_t step = 0; step < num_steps; step++) {
        if (!quantum_walk_step(walker)) {
            return false;
        }
    }

    return true;
}

bool quantum_walk_reset(quantum_walker_t* walker) {
    // Guard: NULL check
    if (!walker) return false;

    // Re-initialize at initial node
    return quantum_walk_initialize(walker, walker->initial_node);
}

//=============================================================================
// Measurement and Output
//=============================================================================

bool quantum_walk_get_distribution(
    const quantum_walker_t* walker,
    float* probabilities
) {
    // Guard: NULL checks
    if (!walker || !probabilities) return false;

    // Copy cached probabilities
    memcpy(probabilities, walker->probabilities,
           walker->num_nodes * sizeof(float));

    return true;
}

bool quantum_walk_get_amplitudes(
    const quantum_walker_t* walker,
    quantum_amplitude_t* amplitudes
) {
    // Guard: NULL checks
    if (!walker || !amplitudes) return false;

    // Copy quantum amplitudes
    memcpy(amplitudes, walker->amplitudes,
           walker->num_nodes * sizeof(quantum_amplitude_t));

    return true;
}

uint32_t quantum_walk_measure(quantum_walker_t* walker) {
    // Guard: NULL check
    if (!walker) return 0;

    // WHAT: Sample from probability distribution
    // WHY: Collapse quantum state to classical position
    // HOW: Weighted random sampling

    // Generate random number [0, 1]
    float r = (float)rand() / (float)RAND_MAX;

    // Find node via cumulative probability
    float cumulative = 0.0f;
    for (uint32_t i = 0; i < walker->num_nodes; i++) {
        cumulative += walker->probabilities[i];
        if (r <= cumulative) {
            // Collapse to measured position
            quantum_walk_initialize(walker, i);
            return i;
        }
    }

    // Fallback: last node
    return walker->num_nodes - 1;
}

//=============================================================================
// Statistics and Diagnostics
//=============================================================================

bool quantum_walk_compute_stats(
    quantum_walker_t* walker,
    quantum_walk_stats_t* stats
) {
    // Guard: NULL checks
    if (!walker || !stats) return false;

    // Compute total probability
    float total_prob = 0.0f;
    float max_amp = 0.0f;
    uint32_t nonzero_count = 0;

    for (uint32_t i = 0; i < walker->num_nodes; i++) {
        float prob = walker->probabilities[i];
        total_prob += prob;

        float amp_mag = sqrtf(prob);
        if (amp_mag > max_amp) max_amp = amp_mag;

        if (amp_mag > AMPLITUDE_THRESHOLD) nonzero_count++;
    }

    // Compute Shannon entropy: H = -Σ P(i)×log(P(i))
    float entropy = 0.0f;
    for (uint32_t i = 0; i < walker->num_nodes; i++) {
        float p = walker->probabilities[i];
        if (p > AMPLITUDE_THRESHOLD) {
            entropy -= p * logf(p);
        }
    }

    // Compute spreading distance (simplified)
    float spreading = sqrtf((float)nonzero_count);

    // Fill statistics
    stats->total_probability = total_prob;
    stats->max_amplitude = max_amp;
    stats->entropy = entropy;
    stats->spreading_distance = spreading;
    stats->num_nonzero_amplitudes = nonzero_count;
    stats->speedup_vs_classical = sqrtf((float)walker->num_nodes);
    stats->evolution_time_us = walker->stats.evolution_time_us;

    walker->stats = *stats;

    return true;
}

void quantum_walk_print_stats(const quantum_walker_t* walker) {
    if (!walker) {
        printf("Quantum Walker: NULL\n");
        return;
    }

    printf("\n=== Quantum Walk Statistics ===\n");
    printf("Nodes:                  %u\n", walker->num_nodes);
    printf("Current step:           %u\n", walker->current_step);
    printf("Initial node:           %u\n", walker->initial_node);
    printf("Total probability:      %.6f\n", walker->stats.total_probability);
    printf("Max amplitude:          %.6f\n", walker->stats.max_amplitude);
    printf("Entropy:                %.6f\n", walker->stats.entropy);
    printf("Spreading distance:     %.2f nodes\n", walker->stats.spreading_distance);
    printf("Nonzero amplitudes:     %u (%.1f%%)\n",
           walker->stats.num_nonzero_amplitudes,
           100.0f * walker->stats.num_nonzero_amplitudes / walker->num_nodes);
    printf("Speedup vs classical:   %.2fx\n", walker->stats.speedup_vs_classical);
    printf("Evolution time:         %lu μs\n", walker->stats.evolution_time_us);
    printf("===============================\n\n");
}

bool quantum_walk_verify(const quantum_walker_t* walker) {
    if (!walker) return false;

    // WHAT: Check probability conservation
    // WHY: Detect numerical errors
    // HOW: Verify Σ|αᵢ|² ≈ 1.0

    float total_prob = 0.0f;
    for (uint32_t i = 0; i < walker->num_nodes; i++) {
        total_prob += walker->probabilities[i];
    }

    float error = fabsf(total_prob - 1.0f);
    if (error > PROB_TOLERANCE) {
        printf("⚠️ Quantum walk verification FAILED: Probability = %.6f (error = %.6f)\n",
               total_prob, error);
        return false;
    }

    return true;
}

//=============================================================================
// Advanced Operations
//=============================================================================

bool quantum_walk_apply_custom_coin(
    quantum_walker_t* walker,
    const quantum_amplitude_t** coin_matrix
) {
    // Guard: NULL checks
    if (!walker || !coin_matrix) return false;

    // TODO: Implement custom coin operator
    // Requires matrix multiplication at each node

    return false; // Not yet implemented
}

bool quantum_walk_hybrid_step(
    quantum_walker_t* walker,
    const float* classical_weights,
    float mixing_ratio
) {
    // Guard: NULL checks
    if (!walker || !classical_weights) return false;
    if (mixing_ratio < 0.0f || mixing_ratio > 1.0f) return false;

    // WHAT: Mix quantum + classical diffusion
    // WHY: Balance speedup and biological realism
    // HOW: α_new = (1-λ)×α_quantum + λ×α_classical

    // Perform quantum step
    quantum_walk_step(walker);

    // Mix with classical weights
    for (uint32_t i = 0; i < walker->num_nodes; i++) {
        quantum_amplitude_t quantum_amp = walker->amplitudes[i];
        quantum_amplitude_t classical_amp = classical_weights[i] + 0.0f * I;

        walker->amplitudes[i] = (1.0f - mixing_ratio) * quantum_amp
                              + mixing_ratio * classical_amp;
    }

    // Renormalize
    quantum_normalize(walker->amplitudes, walker->num_nodes);

    return true;
}

bool quantum_walk_apply_decoherence(
    quantum_walker_t* walker,
    float decoherence_strength
) {
    // Guard: NULL check
    if (!walker) return false;
    if (decoherence_strength < 0.0f || decoherence_strength > 1.0f) return false;

    // WHAT: Add noise to quantum amplitudes
    // WHY: Model environmental decoherence
    // HOW: Add random phase, reduce off-diagonal coherence

    for (uint32_t i = 0; i < walker->num_nodes; i++) {
        // Add random phase
        float random_phase = 2.0f * PI * ((float)rand() / (float)RAND_MAX);
        quantum_amplitude_t phase_factor = cosf(random_phase) + I * sinf(random_phase);

        // Mix original amplitude with dephased version
        walker->amplitudes[i] = (1.0f - decoherence_strength) * walker->amplitudes[i]
                              + decoherence_strength * phase_factor * walker->amplitudes[i];
    }

    return true;
}
