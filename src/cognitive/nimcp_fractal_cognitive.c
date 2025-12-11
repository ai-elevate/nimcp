//=============================================================================
// nimcp_fractal_cognitive.c - Fractal Topology Cognitive Integration
//=============================================================================

#include "cognitive/nimcp_fractal_cognitive.h"
#include "security/nimcp_security.h"
#include "security/nimcp_blood_brain_barrier.h"

#include "utils/logging/nimcp_logging.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_messages.h"
#include "utils/memory/nimcp_unified_memory.h"
#include "core/topology/nimcp_fractal_topology.h"
#include "utils/memory/nimcp_memory.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

//=============================================================================
// Initialization and Caching
//=============================================================================

bool fractal_cognitive_init(neural_network_t network, fractal_cognitive_cache_t *cache) {
    if (!network || !cache) {
        return false;
    }

    memset(cache, 0, sizeof(fractal_cognitive_cache_t));

    // Compute topology statistics
    if (!topology_compute_stats(network, &cache->stats)) {
        return false;
    }

    uint32_t N = cache->stats.num_neurons;
    if (N == 0) {
        return false;
    }

    // Identify hub neurons (top 10%)
    if (!topology_identify_hubs(network, 0.9F, &cache->hub_indices, &cache->num_hubs)) {
        return false;
    }

    // Allocate centrality scores
    cache->centrality_scores = (float*)nimcp_malloc(N * sizeof(float));
    if (!cache->centrality_scores) {
        fractal_cognitive_free(cache);
        return false;
    }

    // Compute betweenness centrality
    if (!topology_compute_betweenness(network, cache->centrality_scores)) {
        fractal_cognitive_free(cache);
        return false;
    }

    // Allocate and compute normalized degrees
    cache->degree_normalized = (float*)nimcp_malloc(N * sizeof(float));
    if (!cache->degree_normalized) {
        fractal_cognitive_free(cache);
        return false;
    }

    // Normalize degrees by maximum degree
    float max_degree = 0.0F;
    for (uint32_t i = 0; i < N; i++) {
        // Get degree from network (would need network API - use placeholder)
        float degree = 1.0F;  // TODO: Get actual degree from network
        if (degree > max_degree) {
            max_degree = degree;
        }
    }

    if (max_degree > 0.0F) {
        for (uint32_t i = 0; i < N; i++) {
            float degree = 1.0F;  // TODO: Get actual degree
            cache->degree_normalized[i] = degree / max_degree;
        }
    } else {
        // All degrees are 0, set uniform
        for (uint32_t i = 0; i < N; i++) {
            cache->degree_normalized[i] = 0.0F;
        }
    }

    cache->valid = true;
    return true;
}

void fractal_cognitive_free(fractal_cognitive_cache_t *cache) {
    if (!cache) {
        return;
    }

    if (cache->hub_indices) {
        nimcp_free(cache->hub_indices);
        cache->hub_indices = NULL;
    }

    if (cache->centrality_scores) {
        nimcp_free(cache->centrality_scores);
        cache->centrality_scores = NULL;
    }

    if (cache->degree_normalized) {
        nimcp_free(cache->degree_normalized);
        cache->degree_normalized = NULL;
    }

    cache->num_hubs = 0;
    cache->valid = false;
}

bool fractal_cognitive_refresh(neural_network_t network, fractal_cognitive_cache_t *cache) {
    if (!cache) {
        return false;
    }

    // Free old cache
    fractal_cognitive_free(cache);

    // Reinitialize
    return fractal_cognitive_init(network, cache);
}

//=============================================================================
// Hub Neuron Queries
//=============================================================================

bool fractal_is_hub_neuron(const fractal_cognitive_cache_t *cache, uint32_t neuron_index) {
    if (!cache || !cache->valid || !cache->hub_indices) {
        return false;
    }

    // Binary search (assuming hub_indices is sorted)
    uint32_t left = 0;
    uint32_t right = cache->num_hubs;

    while (left < right) {
        uint32_t mid = left + (right - left) / 2;
        if (cache->hub_indices[mid] == neuron_index) {
            return true;
        } else if (cache->hub_indices[mid] < neuron_index) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    return false;
}

uint32_t fractal_nearest_hub(neural_network_t network,
                              const fractal_cognitive_cache_t *cache,
                              uint32_t neuron_index,
                              uint32_t *distance_out) {
    if (!network || !cache || !cache->valid) {
        if (distance_out) *distance_out = UINT32_MAX;
        return UINT32_MAX;
    }

    // Simplified: Return first hub with lowest index difference
    // TODO: Implement proper BFS for graph distance
    uint32_t nearest = UINT32_MAX;
    uint32_t min_dist = UINT32_MAX;

    for (uint32_t i = 0; i < cache->num_hubs; i++) {
        uint32_t hub = cache->hub_indices[i];
        uint32_t dist = (hub > neuron_index) ? (hub - neuron_index) : (neuron_index - hub);

        if (dist < min_dist) {
            min_dist = dist;
            nearest = hub;
        }
    }

    if (distance_out) {
        *distance_out = min_dist;
    }

    return nearest;
}

uint32_t fractal_get_central_neighbors(neural_network_t network,
                                        const fractal_cognitive_cache_t *cache,
                                        uint32_t neuron_index,
                                        uint32_t radius,
                                        uint32_t k,
                                        uint32_t *central_out) {
    if (!network || !cache || !cache->valid || !central_out || k == 0) {
        return 0;
    }

    // Simplified: Return k most central neurons from entire network
    // TODO: Implement radius-constrained search

    typedef struct {
        uint32_t index;
        float centrality;
    } scored_neuron_t;

    uint32_t N = cache->stats.num_neurons;
    scored_neuron_t *scored = (scored_neuron_t*)nimcp_malloc(N * sizeof(scored_neuron_t));
    if (!scored) {
        return 0;
    }

    // Score all neurons
    for (uint32_t i = 0; i < N; i++) {
        scored[i].index = i;
        scored[i].centrality = cache->centrality_scores[i];
    }

    // Sort by centrality (bubble sort for simplicity)
    for (uint32_t i = 0; i < N - 1; i++) {
        for (uint32_t j = 0; j < N - i - 1; j++) {
            if (scored[j].centrality < scored[j + 1].centrality) {
                scored_neuron_t temp = scored[j];
                scored[j] = scored[j + 1];
                scored[j + 1] = temp;
            }
        }
    }

    // Copy top k to output
    uint32_t count = (k < N) ? k : N;
    for (uint32_t i = 0; i < count; i++) {
        central_out[i] = scored[i].index;
    }

    nimcp_free(scored);
    return count;
}

//=============================================================================
// Centrality Queries
//=============================================================================

float fractal_get_centrality(const fractal_cognitive_cache_t *cache, uint32_t neuron_index) {
    if (!cache || !cache->valid || !cache->centrality_scores) {
        return 0.0F;
    }

    if (neuron_index >= cache->stats.num_neurons) {
        return 0.0F;
    }

    return cache->centrality_scores[neuron_index];
}

float fractal_get_degree_normalized(const fractal_cognitive_cache_t *cache, uint32_t neuron_index) {
    if (!cache || !cache->valid || !cache->degree_normalized) {
        return 0.0F;
    }

    if (neuron_index >= cache->stats.num_neurons) {
        return 0.0F;
    }

    return cache->degree_normalized[neuron_index];
}

//=============================================================================
// Hierarchical Structure Queries
//=============================================================================

float fractal_get_hierarchical_level(const fractal_cognitive_cache_t *cache, uint32_t neuron_index) {
    if (!cache || !cache->valid) {
        return 0.5F;  // Mid-level default
    }

    float centrality = fractal_get_centrality(cache, neuron_index);
    float degree = fractal_get_degree_normalized(cache, neuron_index);

    // High centrality + high degree → near root (level ≈ 0)
    // Low centrality + low degree → near leaf (level ≈ 1)
    float level = 1.0F - sqrtf(centrality * degree);

    return level;
}

bool fractal_get_neurons_at_level(const fractal_cognitive_cache_t *cache,
                                   float level,
                                   float tolerance,
                                   uint32_t **neurons_out,
                                   uint32_t *count_out) {
    if (!cache || !cache->valid || !neurons_out || !count_out) {
        return false;
    }

    uint32_t N = cache->stats.num_neurons;

    // Count matching neurons
    uint32_t count = 0;
    for (uint32_t i = 0; i < N; i++) {
        float neuron_level = fractal_get_hierarchical_level(cache, i);
        if (fabsf(neuron_level - level) <= tolerance) {
            count++;
        }
    }

    if (count == 0) {
        *neurons_out = NULL;
        *count_out = 0;
        return true;
    }

    // Allocate output array
    *neurons_out = (uint32_t*)nimcp_malloc(count * sizeof(uint32_t));
    if (!*neurons_out) {
        *count_out = 0;
        return false;
    }

    // Fill output array
    uint32_t idx = 0;
    for (uint32_t i = 0; i < N; i++) {
        float neuron_level = fractal_get_hierarchical_level(cache, i);
        if (fabsf(neuron_level - level) <= tolerance) {
            (*neurons_out)[idx++] = i;
        }
    }

    *count_out = count;
    return true;
}

//=============================================================================
// Debug/Visualization
//=============================================================================

void fractal_cognitive_print_summary(const fractal_cognitive_cache_t *cache) {
    if (!cache || !cache->valid) {
        printf("Fractal cognitive cache: INVALID\n");
        return;
    }

    printf("=== Fractal Cognitive Properties ===\n");
    printf("Total neurons: %u\n", cache->stats.num_neurons);
    printf("Total synapses: %u\n", cache->stats.num_synapses);
    printf("Avg degree: %.2f\n", cache->stats.avg_degree);
    printf("Hub neurons: %u (%.1f%%)\n", cache->num_hubs,
           100.0F * cache->num_hubs / cache->stats.num_neurons);
    printf("Clustering coeff: %.3f\n", cache->stats.clustering_coefficient);
    printf("Char. path length: %.2f\n", cache->stats.characteristic_path);
    printf("Power-law fit R²: %.3f\n", cache->stats.power_law_fit);
    printf("Small-world σ: %.2f\n", cache->stats.small_world_sigma);

    // Print top 5 most central neurons
    printf("\nTop 5 Central Neurons:\n");
    uint32_t top_neurons[5];
    uint32_t num_found = fractal_get_central_neighbors(NULL, cache, 0, UINT32_MAX, 5, top_neurons);
    for (uint32_t i = 0; i < num_found; i++) {
        printf("  %u: centrality=%.4f, level=%.3f\n",
               top_neurons[i],
               fractal_get_centrality(cache, top_neurons[i]),
               fractal_get_hierarchical_level(cache, top_neurons[i]));
    }
    printf("====================================\n");
}
