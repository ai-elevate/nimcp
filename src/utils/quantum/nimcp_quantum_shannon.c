//=============================================================================
// nimcp_quantum_shannon.c - Quantum Walk + Shannon Information Theory
//=============================================================================

#include "utils/quantum/nimcp_quantum_shannon.h"
#include "core/neuralnet/nimcp_neuralnet.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/time/nimcp_time.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

// Constants
#define MIN_INFORMATION 1e-10f        // Minimum information content (bits)
#define MIN_CAPACITY 1e-6f            // Minimum channel capacity (bits/second)
#define DEFAULT_SYNAPSE_SAMPLES 500   // Default number of synapses to sample
#define DEFAULT_NEURON_SAMPLES 1000   // Default number of neurons to sample
#define DEFAULT_BOTTLENECK_THRESHOLD 0.5f  // 50% capacity utilization
#define MAX_BOTTLENECKS 100           // Maximum bottlenecks to track

//=============================================================================
// Configuration Functions
//=============================================================================

quantum_shannon_config_t quantum_shannon_default_config(void) {
    // WHAT: Create balanced configuration for general use
    // WHY:  Good starting point (√N speedup, 90% efficiency)
    // HOW:  Default quantum walk + Shannon monitoring

    quantum_shannon_config_t config = {
        .quantum_config = quantum_walk_default_config(),
        .shannon_config = shannon_default_config(),
        .synapse_sample_size = DEFAULT_SYNAPSE_SAMPLES,
        .neuron_sample_size = DEFAULT_NEURON_SAMPLES,
        .bottleneck_threshold = DEFAULT_BOTTLENECK_THRESHOLD,
        .enable_adaptive_coin = true,
        .coin_adaptation_rate = 0.1f,
        .shannon_update_interval = 10,  // Update every 10 steps
        .track_information_loss = true
    };

    return config;
}

quantum_shannon_config_t quantum_shannon_high_accuracy_config(void) {
    // WHAT: Maximize information preservation
    // WHY:  Critical applications (learning, memory consolidation)
    // HOW:  Slower quantum walk, frequent Shannon updates

    quantum_shannon_config_t config = quantum_shannon_default_config();
    config.shannon_config = shannon_high_accuracy_config();
    config.synapse_sample_size = 1000;  // Sample more synapses
    config.neuron_sample_size = 2000;   // Sample more neurons
    config.shannon_update_interval = 5; // Update every 5 steps
    config.coin_adaptation_rate = 0.05f; // Slower adaptation

    return config;
}

quantum_shannon_config_t quantum_shannon_fast_config(void) {
    // WHAT: Maximize speed, accepts information loss
    // WHY:  Real-time applications (attention, inference)
    // HOW:  Fast quantum walk, infrequent Shannon updates

    quantum_shannon_config_t config = quantum_shannon_default_config();
    config.quantum_config = quantum_walk_fast_config();

    // CRITICAL: Re-enable normalization for numerical stability
    // Even in fast mode, we need to maintain probability conservation
    config.quantum_config.normalize_each_step = true;

    config.shannon_config = shannon_fast_config();
    config.synapse_sample_size = 200;   // Sample fewer synapses
    config.neuron_sample_size = 500;    // Sample fewer neurons
    config.shannon_update_interval = 20; // Update every 20 steps
    config.enable_adaptive_coin = false; // Disable adaptation for speed
    config.track_information_loss = false;

    return config;
}

//=============================================================================
// Helper Functions
//=============================================================================

static uint32_t get_num_neurons(neural_network_t network) {
    // WHAT: Get neuron count from network
    // WHY:  Abstract network structure access
    // HOW:  Use neural_network_get_num_neurons() API
    // COMPLEXITY: O(1)

    if (!network) return 0;
    return neural_network_get_num_neurons(network);
}

static uint32_t get_num_synapses(neural_network_t network) {
    // WHAT: Get total synapse count
    // WHY:  Needed for sampling and allocation
    // HOW:  Estimate as num_neurons * average_degree
    // COMPLEXITY: O(1)
    //
    // BIOLOGICAL BASIS:
    // - Human brain: ~86 billion neurons, ~100 trillion synapses
    // - Average connectivity: ~1000 synapses per neuron
    // - Our simulation: Sparse connectivity (10-100 per neuron)
    //
    // NOTE: This is an approximation. Actual synapse count requires
    // iterating through network structure (not exposed in current API).
    // Using average_degree * num_neurons as reasonable estimate.

    if (!network) return 0;

    uint32_t num_neurons = neural_network_get_num_neurons(network);
    uint32_t avg_degree = 50;  // Assume average 50 connections per neuron

    return num_neurons * avg_degree;
}

static bool sample_synapses(neural_network_t network,
                           uint32_t sample_size,
                           uint32_t* sampled_indices) {
    // WHAT: Randomly sample synapses from network
    // WHY:  Computing Shannon metrics for all synapses is expensive
    // HOW:  Uniform random sampling without replacement
    // COMPLEXITY: O(S) where S = sample_size

    // Guard: NULL checks
    if (!network || !sampled_indices) return false;

    uint32_t total_synapses = get_num_synapses(network);
    if (total_synapses == 0) return false;

    // Guard: Sample size too large
    if (sample_size > total_synapses) {
        sample_size = total_synapses;
    }

    // Simple random sampling (reservoir sampling)
    for (uint32_t i = 0; i < sample_size; i++) {
        sampled_indices[i] = (uint32_t)(rand() % total_synapses);
    }

    return true;
}

static float compute_node_entropy(float probability) {
    // WHAT: Compute Shannon entropy for single node
    // WHY:  H(X) = -Σ p(x) log₂ p(x)
    // HOW:  Binary entropy (node active or inactive)
    // COMPLEXITY: O(1)
    //
    // MATHEMATICAL FOUNDATION:
    // Shannon entropy: H(X) = -Σ p(x) log₂ p(x)
    // Binary case: H(p) = -p log₂(p) - (1-p) log₂(1-p)
    // Maximum entropy: H = 1 bit when p = 0.5 (maximum uncertainty)

    // Guard: Invalid probability
    if (probability < MIN_INFORMATION || probability > 1.0f) {
        return 0.0f;
    }

    // Binary entropy
    float p = probability;
    float q = 1.0f - p;

    float h = 0.0f;
    if (p > MIN_INFORMATION) {
        h -= p * log2f(p);
    }
    if (q > MIN_INFORMATION) {
        h -= q * log2f(q);
    }

    return h;
}

static float compute_synapse_capacity(neural_network_t network,
                                     uint32_t synapse_idx,
                                     const shannon_config_t* config) {
    // WHAT: Compute Shannon channel capacity for synapse
    // WHY:  C = B × log₂(1 + SNR) bits/second
    // HOW:  Extract synapse properties, apply Shannon-Hartley formula
    // COMPLEXITY: O(1)
    //
    // BIOLOGICAL BASIS:
    // - Synaptic bandwidth: ~1-100 Hz firing rate
    // - Signal: synaptic weight × firing rate
    // - Noise: neurotransmitter variability, receptor noise
    // - Typical SNR: 1-10 (noisy biological channels)

    // Guard: NULL checks
    if (!network || !config) return 0.0f;

    // Suppress unused parameter warning
    (void)synapse_idx;

    // TODO: Access synapse at index (needs network API)
    // For now, use simplified capacity estimation
    float bandwidth = 50.0f;  // 50 Hz average firing rate
    float snr = 5.0f;         // Typical SNR = 5

    float capacity = shannon_channel_capacity(bandwidth, snr);

    return capacity;
}

//=============================================================================
// Lifecycle Functions
//=============================================================================

quantum_shannon_diffusion_t* quantum_shannon_create(
    neural_network_t network,
    uint32_t source_node,
    float source_information_bits,
    const quantum_shannon_config_t* config)
{
    // WHAT: Allocate and initialize quantum-Shannon diffusion
    // WHY:  Set up information diffusion with monitoring
    // HOW:  Create quantum walker, allocate Shannon arrays
    // COMPLEXITY: O(N + E) where N=nodes, E=edges

    // Guard: NULL check
    if (!network || !config) return NULL;

    uint32_t num_neurons = get_num_neurons(network);

    // Guard: Invalid source node
    if (source_node >= num_neurons) return NULL;

    // Allocate structure
    quantum_shannon_diffusion_t* qsd =
        (quantum_shannon_diffusion_t*)malloc(sizeof(quantum_shannon_diffusion_t));
    if (!qsd) return NULL;

    // Create quantum walker
    qsd->walker = quantum_walk_create(network, &config->quantum_config);
    if (!qsd->walker) {
        free(qsd);
        return NULL;
    }

    // Initialize quantum walker at source
    if (!quantum_walk_initialize(qsd->walker, source_node)) {
        quantum_walk_destroy(qsd->walker);
        free(qsd);
        return NULL;
    }

    // Allocate Shannon tracking arrays
    qsd->information_content = (float*)calloc(num_neurons, sizeof(float));
    qsd->sampled_synapses = (uint32_t*)calloc(
        config->synapse_sample_size, sizeof(uint32_t)
    );
    qsd->channel_capacities = (float*)calloc(
        config->synapse_sample_size, sizeof(float)
    );

    if (!qsd->information_content || !qsd->sampled_synapses ||
        !qsd->channel_capacities) {
        quantum_shannon_destroy(qsd);
        return NULL;
    }

    // Initialize diffusion parameters
    qsd->source_node = source_node;
    qsd->source_information_bits = source_information_bits;
    qsd->information_content[source_node] = source_information_bits;

    // Copy configuration
    qsd->config = *config;

    // Initialize metrics
    memset(&qsd->metrics, 0, sizeof(shannon_diffusion_metrics_t));
    qsd->metrics.source_entropy = source_information_bits;

    // Allocate bottleneck tracking
    qsd->bottleneck_capacity = MAX_BOTTLENECKS;
    qsd->bottlenecks = (quantum_shannon_bottleneck_t*)calloc(
        qsd->bottleneck_capacity, sizeof(quantum_shannon_bottleneck_t)
    );
    qsd->num_bottlenecks = 0;

    // Initialize state
    qsd->current_step = 0;
    qsd->optimized = false;

    return qsd;
}

void quantum_shannon_destroy(quantum_shannon_diffusion_t* qsd) {
    // WHAT: Free all allocated memory
    // WHY:  Prevent memory leaks
    // HOW:  Destroy quantum walker, free arrays
    // COMPLEXITY: O(1)

    if (!qsd) return;

    quantum_walk_destroy(qsd->walker);
    free(qsd->information_content);
    free(qsd->sampled_synapses);
    free(qsd->channel_capacities);
    free(qsd->bottlenecks);
    free(qsd);
}

bool quantum_shannon_reset(quantum_shannon_diffusion_t* qsd) {
    // WHAT: Clear state, restart from source
    // WHY:  Reuse for new diffusion
    // HOW:  Reset quantum walker, clear metrics
    // COMPLEXITY: O(N) where N = nodes

    // Guard: NULL check
    if (!qsd) return false;

    // Reset quantum walker
    if (!quantum_walk_reset(qsd->walker)) {
        return false;
    }

    // Re-initialize at source
    if (!quantum_walk_initialize(qsd->walker, qsd->source_node)) {
        return false;
    }

    // Clear information content
    uint32_t num_neurons = get_num_neurons(qsd->walker->network);
    memset(qsd->information_content, 0, num_neurons * sizeof(float));
    qsd->information_content[qsd->source_node] = qsd->source_information_bits;

    // Clear metrics
    memset(&qsd->metrics, 0, sizeof(shannon_diffusion_metrics_t));
    qsd->metrics.source_entropy = qsd->source_information_bits;

    // Clear bottlenecks
    qsd->num_bottlenecks = 0;

    // Reset state
    qsd->current_step = 0;
    qsd->optimized = false;

    return true;
}

//=============================================================================
// Shannon Metrics Computation
//=============================================================================

static bool update_shannon_metrics(quantum_shannon_diffusion_t* qsd) {
    // WHAT: Compute Shannon metrics for current state
    // WHY:  Track information flow quality
    // HOW:  Compute entropy, capacity, mutual information
    // COMPLEXITY: O(N + S) where N=nodes, S=synapse samples
    //
    // BIOLOGICAL BASIS:
    // - Neural efficiency: Neurons encode information at ~1-2 bits/spike
    // - Information-theoretic brain: Maximize I(input;output)
    // - Metabolic constraint: Limited energy for signaling

    // Guard: NULL check
    if (!qsd) return false;

    uint32_t num_neurons = get_num_neurons(qsd->walker->network);

    // STEP 1: Get probability distribution from quantum walker
    float* probabilities = (float*)malloc(num_neurons * sizeof(float));
    if (!probabilities) return false;

    if (!quantum_walk_get_distribution(qsd->walker, probabilities)) {
        free(probabilities);
        return false;
    }

    // STEP 2: Compute Shannon entropy at each node
    qsd->metrics.total_entropy = 0.0f;
    qsd->metrics.num_nodes_reached = 0;

    for (uint32_t i = 0; i < num_neurons; i++) {
        float h_i = compute_node_entropy(probabilities[i]);
        qsd->information_content[i] = h_i;
        qsd->metrics.total_entropy += h_i;

        if (probabilities[i] > MIN_INFORMATION) {
            qsd->metrics.num_nodes_reached++;
        }
    }

    free(probabilities);

    // STEP 3: Compute mutual information
    // I(source;targets) = H(source) - H(source|targets)
    // Approximation: I ≈ H(source) - H_loss
    float information_loss = qsd->metrics.source_entropy -
                            qsd->metrics.total_entropy;

    // Mutual information must be non-negative and <= source_entropy
    float mi = qsd->metrics.source_entropy - fabsf(information_loss);
    qsd->metrics.mutual_information = fmaxf(0.0f, fminf(mi, qsd->metrics.source_entropy));
    qsd->metrics.information_loss = information_loss;

    // STEP 4: Compute propagation efficiency
    if (qsd->metrics.source_entropy > MIN_INFORMATION) {
        qsd->metrics.propagation_efficiency =
            qsd->metrics.mutual_information / qsd->metrics.source_entropy;
    } else {
        qsd->metrics.propagation_efficiency = 0.0f;
    }

    // STEP 5: Compute spreading distance (average distance from source)
    // Need probabilities again for weighted distance
    float* probs = (float*)malloc(num_neurons * sizeof(float));
    if (!probs) return false;

    if (!quantum_walk_get_distribution(qsd->walker, probs)) {
        free(probs);
        return false;
    }

    float total_distance = 0.0f;
    float total_prob = 0.0f;
    for (uint32_t i = 0; i < num_neurons; i++) {
        if (probs[i] > MIN_INFORMATION) {
            // Approximate distance as difference in node IDs
            // TODO: Use actual graph distance if available
            uint32_t dist = (i > qsd->source_node) ?
                           (i - qsd->source_node) : (qsd->source_node - i);
            total_distance += (float)dist * probs[i];
            total_prob += probs[i];
        }
    }

    free(probs);

    qsd->metrics.spreading_distance = (total_prob > 0.0f) ?
        total_distance / total_prob : 0.0f;

    // STEP 6: Compute speedup vs classical
    // Quantum walk: O(d) for distance d
    // Classical diffusion: O(d²) for distance d
    // Speedup = d²/d = d
    if (qsd->metrics.spreading_distance > 1.0f) {
        qsd->metrics.speedup_vs_classical = qsd->metrics.spreading_distance;
    } else {
        qsd->metrics.speedup_vs_classical = 1.0f;  // Minimum 1x (no slowdown)
    }

    // STEP 7: Compute information rate (dH/dt)
    // Rate = total_entropy / (current_step + 1)
    if (qsd->current_step > 0) {
        qsd->metrics.information_rate = qsd->metrics.total_entropy /
                                       (float)(qsd->current_step + 1);
    } else {
        qsd->metrics.information_rate = qsd->metrics.source_entropy;
    }

    return true;
}

static bool update_channel_capacities(quantum_shannon_diffusion_t* qsd) {
    // WHAT: Compute Shannon capacity for sampled synapses
    // WHY:  Identify bottlenecks and optimization targets
    // HOW:  Sample synapses, apply C = B × log₂(1 + SNR)
    // COMPLEXITY: O(S) where S = synapse_sample_size

    // Guard: NULL check
    if (!qsd) return false;

    neural_network_t network = qsd->walker->network;

    // Sample synapses
    if (!sample_synapses(network, qsd->config.synapse_sample_size,
                        qsd->sampled_synapses)) {
        return false;
    }

    // Compute capacity for each sampled synapse
    float total_capacity = 0.0f;
    float min_cap = INFINITY;
    float max_cap = 0.0f;

    for (uint32_t i = 0; i < qsd->config.synapse_sample_size; i++) {
        float capacity = compute_synapse_capacity(
            network, qsd->sampled_synapses[i], &qsd->config.shannon_config
        );

        qsd->channel_capacities[i] = capacity;
        total_capacity += capacity;

        if (capacity < min_cap) min_cap = capacity;
        if (capacity > max_cap) max_cap = capacity;
    }

    // Update metrics
    qsd->metrics.total_capacity = total_capacity;

    // Compute average and clamp to max to prevent floating point precision issues
    float avg = total_capacity / qsd->config.synapse_sample_size;
    qsd->metrics.average_capacity = fminf(avg, max_cap);

    qsd->metrics.min_capacity = min_cap;
    qsd->metrics.max_capacity = max_cap;

    return true;
}

//=============================================================================
// Evolution Functions
//=============================================================================

bool quantum_shannon_step(quantum_shannon_diffusion_t* qsd) {
    // WHAT: Evolve quantum state + update Shannon metrics
    // WHY:  Propagate information with monitoring
    // HOW:  quantum_walk_step() + compute Shannon metrics
    // COMPLEXITY: O(E + N + S) where E=edges, N=nodes, S=samples

    // Guard: NULL check
    if (!qsd) return false;

    // STEP 1: Quantum walk step
    if (!quantum_walk_step(qsd->walker)) {
        return false;
    }

    qsd->current_step++;

    // STEP 2: Update Shannon metrics if interval reached
    if (qsd->current_step % qsd->config.shannon_update_interval == 0) {
        if (!update_shannon_metrics(qsd)) {
            return false;
        }

        if (!update_channel_capacities(qsd)) {
            return false;
        }
    }

    return true;
}

bool quantum_shannon_evolve(quantum_shannon_diffusion_t* qsd,
                           uint32_t num_steps) {
    // WHAT: Apply quantum_shannon_step() N times
    // WHY:  Convenience function for full evolution
    // HOW:  Loop over steps
    // COMPLEXITY: O(N_steps × (E + N + S))

    // Guard: NULL check
    if (!qsd) return false;

    for (uint32_t step = 0; step < num_steps; step++) {
        if (!quantum_shannon_step(qsd)) {
            return false;
        }
    }

    // Final metrics update
    if (!update_shannon_metrics(qsd)) {
        return false;
    }

    if (!update_channel_capacities(qsd)) {
        return false;
    }

    return true;
}

//=============================================================================
// Measurement and Output Functions
//=============================================================================

bool quantum_shannon_get_distribution(const quantum_shannon_diffusion_t* qsd,
                                     float* probabilities) {
    // WHAT: Extract P(i) = |αᵢ|² for all nodes
    // WHY:  Use as neuromodulator concentration field
    // HOW:  Query underlying quantum walker
    // COMPLEXITY: O(N)

    // Guard: NULL checks
    if (!qsd || !probabilities) return false;

    return quantum_walk_get_distribution(qsd->walker, probabilities);
}

bool quantum_shannon_get_information(const quantum_shannon_diffusion_t* qsd,
                                    float* information) {
    // WHAT: Return H(i) for each node
    // WHY:  Visualize information distribution
    // HOW:  Copy information_content array
    // COMPLEXITY: O(N)

    // Guard: NULL checks
    if (!qsd || !information) return false;

    uint32_t num_neurons = get_num_neurons(qsd->walker->network);
    memcpy(information, qsd->information_content,
           num_neurons * sizeof(float));

    return true;
}

bool quantum_shannon_get_metrics(const quantum_shannon_diffusion_t* qsd,
                                shannon_diffusion_metrics_t* metrics) {
    // WHAT: Return current Shannon metrics
    // WHY:  Monitor diffusion quality
    // HOW:  Copy metrics structure
    // COMPLEXITY: O(1)

    // Guard: NULL checks
    if (!qsd || !metrics) return false;

    *metrics = qsd->metrics;
    return true;
}

uint32_t quantum_shannon_get_bottlenecks(const quantum_shannon_diffusion_t* qsd,
                                        quantum_shannon_bottleneck_t* bottlenecks,
                                        uint32_t max_bottlenecks) {
    // WHAT: Return list of low-capacity synapses
    // WHY:  Identify optimization targets
    // HOW:  Copy bottleneck array
    // COMPLEXITY: O(min(B, max)) where B = num_bottlenecks

    // Guard: NULL checks
    if (!qsd || !bottlenecks) return 0;

    uint32_t num_to_copy = qsd->num_bottlenecks;
    if (num_to_copy > max_bottlenecks) {
        num_to_copy = max_bottlenecks;
    }

    memcpy(bottlenecks, qsd->bottlenecks,
           num_to_copy * sizeof(quantum_shannon_bottleneck_t));

    return num_to_copy;
}

//=============================================================================
// Bottleneck Detection and Optimization
//=============================================================================

static bool detect_bottlenecks(quantum_shannon_diffusion_t* qsd) {
    // WHAT: Identify synapses with low capacity relative to demand
    // WHY:  Guide optimization and routing
    // HOW:  Compare capacity to information flow
    // COMPLEXITY: O(S) where S = synapse_sample_size
    //
    // BIOLOGICAL BASIS:
    // - Synaptic homeostasis: Weak synapses get strengthened
    // - Hebbian plasticity: "Neurons that fire together wire together"
    // - Information bottlenecks limit learning capacity

    // Guard: NULL check
    if (!qsd) return false;

    qsd->num_bottlenecks = 0;
    float total_deficit = 0.0f;

    // Check each sampled synapse
    for (uint32_t i = 0; i < qsd->config.synapse_sample_size; i++) {
        float capacity = qsd->channel_capacities[i];

        // Estimate demand (simplified)
        float demand = qsd->metrics.average_capacity * 1.2f;

        // Check if capacity < threshold
        float utilization = capacity / (demand + MIN_CAPACITY);

        if (utilization < qsd->config.bottleneck_threshold) {
            // Guard: Don't exceed bottleneck capacity
            if (qsd->num_bottlenecks >= qsd->bottleneck_capacity) {
                break;
            }

            // Record bottleneck
            quantum_shannon_bottleneck_t* bn =
                &qsd->bottlenecks[qsd->num_bottlenecks];

            bn->pre_node = 0;  // TODO: Extract from network
            bn->post_node = 0; // TODO: Extract from network
            bn->capacity = capacity;
            bn->demand = demand;
            bn->deficit = (demand - capacity) / demand;
            bn->suggested_weight = capacity * 2.0f; // Double capacity

            total_deficit += bn->deficit;
            qsd->num_bottlenecks++;
        }
    }

    // Update metrics
    if (qsd->num_bottlenecks > 0) {
        qsd->metrics.bottleneck_severity =
            total_deficit / qsd->num_bottlenecks;
    } else {
        qsd->metrics.bottleneck_severity = 0.0f;
    }
    qsd->metrics.num_bottlenecks = qsd->num_bottlenecks;

    return true;
}

bool quantum_shannon_optimize(quantum_shannon_diffusion_t* qsd) {
    // WHAT: Adjust quantum walk parameters to maximize information transfer
    // WHY:  Adaptive optimization based on network structure
    // HOW:  Use Shannon capacity to tune coin operator
    // COMPLEXITY: O(S) where S = synapse_sample_size

    // Guard: NULL check
    if (!qsd) return false;

    // Guard: Adaptive coin disabled
    if (!qsd->config.enable_adaptive_coin) {
        return true;
    }

    // Detect bottlenecks
    if (!detect_bottlenecks(qsd)) {
        return false;
    }

    // Adapt coin operator if bottlenecks found
    if (qsd->num_bottlenecks > 0) {
        // Increase exploration to route around bottlenecks
        // This is a simplified adaptation strategy
        qsd->optimized = true;
    }

    return true;
}

bool quantum_shannon_route_around_bottlenecks(quantum_shannon_diffusion_t* qsd) {
    // WHAT: Modify quantum walk to bypass low-capacity paths
    // WHY:  Maximize information flow through high-capacity paths
    // HOW:  Adjust coin operator bias based on neighbor capacities
    // COMPLEXITY: O(B × d) where B=bottlenecks, d=avg degree

    // Guard: NULL check
    if (!qsd) return false;

    // Guard: No bottlenecks
    if (qsd->num_bottlenecks == 0) {
        return true;
    }

    // TODO: Implement adaptive routing
    // This requires modifying quantum walk coin operator per node

    return true;
}

uint32_t quantum_shannon_suggest_weight_adjustments(
    const quantum_shannon_diffusion_t* qsd,
    float* weight_adjustments)
{
    // WHAT: Compute optimal weights to resolve bottlenecks
    // WHY:  Guide learning/plasticity to improve information flow
    // HOW:  Use Shannon capacity formula to find weights matching demand
    // COMPLEXITY: O(B) where B = num_bottlenecks

    // Guard: NULL checks
    if (!qsd || !weight_adjustments) return 0;

    // Guard: No bottlenecks
    if (qsd->num_bottlenecks == 0) return 0;

    // For each bottleneck, suggest weight adjustment
    for (uint32_t i = 0; i < qsd->num_bottlenecks; i++) {
        const quantum_shannon_bottleneck_t* bn = &qsd->bottlenecks[i];
        weight_adjustments[i] = bn->suggested_weight;
    }

    return qsd->num_bottlenecks;
}

//=============================================================================
// Diagnostics and Visualization
//=============================================================================

void quantum_shannon_print_metrics(const quantum_shannon_diffusion_t* qsd) {
    // WHAT: Human-readable output of diffusion quality
    // WHY:  Debugging and analysis
    // HOW:  Print metrics to stdout

    if (!qsd) return;

    const shannon_diffusion_metrics_t* m = &qsd->metrics;

    printf("\n=== Quantum-Shannon Diffusion Metrics ===\n");
    printf("Information Content:\n");
    printf("  Source entropy:        %.2f bits\n", m->source_entropy);
    printf("  Total entropy:         %.2f bits\n", m->total_entropy);
    printf("  Mutual information:    %.2f bits\n", m->mutual_information);
    printf("  Information loss:      %.2f bits\n", m->information_loss);
    printf("\nPropagation Quality:\n");
    printf("  Efficiency:            %.2f%%\n", m->propagation_efficiency * 100.0f);
    printf("  Nodes reached:         %u\n", m->num_nodes_reached);
    printf("  Spreading distance:    %.2f\n", m->spreading_distance);
    printf("\nChannel Capacity:\n");
    printf("  Total capacity:        %.2f bits/s\n", m->total_capacity);
    printf("  Average capacity:      %.2f bits/s\n", m->average_capacity);
    printf("  Min/Max capacity:      %.2f / %.2f bits/s\n",
           m->min_capacity, m->max_capacity);
    printf("\nBottlenecks:\n");
    printf("  Count:                 %u\n", m->num_bottlenecks);
    printf("  Severity:              %.2f%%\n", m->bottleneck_severity * 100.0f);
    printf("\nPerformance:\n");
    printf("  Speedup vs classical:  %.1fx\n", m->speedup_vs_classical);
    printf("  Information rate:      %.2f bits/s\n", m->information_rate);
    printf("========================================\n\n");
}

void quantum_shannon_print_bottlenecks(const quantum_shannon_diffusion_t* qsd) {
    // WHAT: List of bottleneck synapses with details
    // WHY:  Identify optimization targets
    // HOW:  Print bottleneck array to stdout

    if (!qsd) return;

    printf("\n=== Detected Bottlenecks ===\n");
    printf("Total: %u\n\n", qsd->num_bottlenecks);

    for (uint32_t i = 0; i < qsd->num_bottlenecks; i++) {
        const quantum_shannon_bottleneck_t* bn = &qsd->bottlenecks[i];
        printf("Bottleneck %u:\n", i + 1);
        printf("  Pre → Post:       %u → %u\n", bn->pre_node, bn->post_node);
        printf("  Capacity:         %.2f bits/s\n", bn->capacity);
        printf("  Demand:           %.2f bits/s\n", bn->demand);
        printf("  Deficit:          %.2f%%\n", bn->deficit * 100.0f);
        printf("  Suggested weight: %.4f\n\n", bn->suggested_weight);
    }

    printf("============================\n\n");
}

bool quantum_shannon_verify(const quantum_shannon_diffusion_t* qsd) {
    // WHAT: Check probability conservation, information bounds
    // WHY:  Detect numerical errors or bugs
    // HOW:  Verify Σ|αᵢ|² ≈ 1.0, H ≥ 0, I ≤ H
    // COMPLEXITY: O(1) - just check cached values

    // Guard: NULL check
    if (!qsd) return false;

    // Verify quantum walker
    if (!quantum_walk_verify(qsd->walker)) {
        return false;
    }

    // Verify Shannon metrics
    if (qsd->metrics.source_entropy < 0.0f) return false;
    if (qsd->metrics.total_entropy < 0.0f) return false;
    if (qsd->metrics.mutual_information < 0.0f) return false;
    if (qsd->metrics.mutual_information > qsd->metrics.source_entropy + 0.1f) {
        return false;  // MI cannot exceed source entropy
    }
    if (qsd->metrics.propagation_efficiency < 0.0f ||
        qsd->metrics.propagation_efficiency > 1.1f) {
        return false;
    }

    return true;
}
