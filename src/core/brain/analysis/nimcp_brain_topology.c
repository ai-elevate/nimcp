//=============================================================================
// nimcp_brain_topology.c - Brain Topology/Graph Analysis Implementation
//=============================================================================
/**
 * @file nimcp_brain_topology.c
 * @brief Network topology analysis functions for brain module
 *
 * EXTRACTED FROM: nimcp_brain.c (lines 7752-8150)
 *
 * ARCHITECTURE:
 * - Static helper: brain_build_topology_graph (converts brain → graph)
 * - Public API: Community detection, hub analysis, metrics, validation
 * - Integration: Works with network_analyzer module for real-time analysis
 */

#include "core/brain/analysis/nimcp_brain_topology.h"
#include "core/brain/nimcp_brain_internal.h"
#include "core/brain/strategy/nimcp_brain_strategy.h"
#include <math.h>
#include <string.h>
#include "plasticity/adaptive/nimcp_adaptive.h"
#include "core/neuralnet/nimcp_neuralnet.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "core/topology/nimcp_community_detection.h"
#include "utils/algorithms/nimcp_graph_metrics.h"
#include "utils/containers/nimcp_graph.h"
#include "cognitive/analysis/nimcp_network_analysis.h"

//=============================================================================
// Static Helper: Build Graph from Brain Topology
//=============================================================================

/**
 * @brief Build graph representation from brain network
 *
 * WHAT: Convert brain network to graph data structure
 * WHY:  Community detection and graph algorithms need graph format
 * HOW:  Extract neurons and synapses, build adjacency list graph
 *
 * ALGORITHM:
 * 1. Create empty graph
 * 2. Add vertex for each neuron
 * 3. Add edge for each synapse (with absolute weight)
 * 4. Return graph
 *
 * COMPLEXITY: O(N + E) where N = neurons, E = synapses
 *
 * @param brain Brain instance
 * @return Graph or NULL on error
 */
static NimcpGraph* brain_build_topology_graph(brain_t brain) {
    if (!brain || !brain->network) {
        NIMCP_LOGGING_ERROR("brain_build_topology_graph: NULL brain or network");
        return NULL;
    }

    // Create graph
    NimcpGraph* graph = nimcp_graph_create();
    if (!graph) {
        NIMCP_LOGGING_ERROR("brain_build_topology_graph: failed to create graph");
        return NULL;
    }

    adaptive_network_t net = brain->network;
    uint32_t num_neurons = adaptive_network_get_num_neurons(net);

    // Add vertices (neurons)
    for (uint32_t i = 0; i < num_neurons; i++) {
        uint32_t vertex_idx = nimcp_graph_add_vertex(graph, i, 0.0f, 0.0f, 0.0f, 0);
        if (vertex_idx == NIMCP_INVALID_VERTEX) {
            NIMCP_LOGGING_WARN("Failed to add vertex %u to graph", i);
        }
    }

    // Add edges (synapses)
    // Get synapse connectivity from network
    for (uint32_t i = 0; i < num_neurons; i++) {
        for (uint32_t j = 0; j < num_neurons; j++) {
            // Check if synapse exists from i to j
            float weight = adaptive_network_get_synapse_weight(net, i, j);
            if (weight != 0.0f) {
                // Add edge with absolute weight (direction doesn't matter for community detection)
                nimcp_weight_t edge_weight = fabsf(weight);
                if (!nimcp_graph_add_edge(graph, i, j, edge_weight)) {
                    NIMCP_LOGGING_WARN("Failed to add edge %u -> %u", i, j);
                }
            }
        }
    }

    NIMCP_LOGGING_INFO("Built topology graph: %u neurons, %u synapses",
                   graph->vertex_count, graph->edge_count);

    return graph;
}

//=============================================================================
// Community Detection
//=============================================================================

NIMCP_EXPORT bool brain_detect_communities(brain_t brain) {
    // Guard: NULL check
    if (!brain) {
        NIMCP_LOGGING_ERROR("brain_detect_communities: NULL brain");
        return false;
    }

    brain_clear_error();

    // 1. Build graph from brain topology
    NimcpGraph* graph = brain_build_topology_graph(brain);
    if (!graph) {
        NIMCP_LOGGING_ERROR("brain_detect_communities: failed to build graph");
        return false;
    }

    // 2. Run Louvain algorithm
    adaptive_network_t network = brain_get_network(brain);
    if (!network) {
        nimcp_graph_destroy(graph);
        NIMCP_LOGGING_ERROR("brain_detect_communities: failed to get adaptive network");
        return false;
    }
    neural_network_t base_network = adaptive_network_get_base_network(network);
    if (!base_network) {
        nimcp_graph_destroy(graph);
        NIMCP_LOGGING_ERROR("brain_detect_communities: failed to get base network");
        return false;
    }
    community_structure_t* communities = community_detect(base_network, NULL);
    if (!communities) {
        nimcp_graph_destroy(graph);
        NIMCP_LOGGING_ERROR("brain_detect_communities: Louvain algorithm failed");
        return false;
    }

    // 3. Store results (replace old if exists)
    if (brain->functional_modules) {
        topology_community_structure_free(brain->functional_modules);
    }
    brain->functional_modules = communities;

    // 4. Log results
    NIMCP_LOGGING_INFO("Detected %u functional modules (Q=%.3f)",
                   communities->num_communities,
                   communities->modularity);

    // Log community sizes
    if (communities->community_sizes) {
        for (uint32_t i = 0; i < communities->num_communities && i < 10; i++) {
            NIMCP_LOGGING_INFO("  Community %u: %u neurons",
                          i, communities->community_sizes[i]);
        }
        if (communities->num_communities > 10) {
            NIMCP_LOGGING_INFO("  ... (%u more communities)", communities->num_communities - 10);
        }
    }

    nimcp_graph_destroy(graph);
    return true;
}

NIMCP_EXPORT uint32_t brain_get_neuron_community(brain_t brain, uint32_t neuron_id) {
    // Guard: NULL check
    if (!brain) {
        return UINT32_MAX;
    }

    // Check if communities detected
    if (!brain->functional_modules) {
        NIMCP_LOGGING_ERROR("brain_get_neuron_community: no communities detected (call brain_detect_communities first)");
        return UINT32_MAX;
    }

    // Get community assignment
    return community_get_neuron_community(brain->functional_modules, neuron_id);
}

//=============================================================================
// Hub Detection
//=============================================================================

NIMCP_EXPORT bool brain_detect_hubs(brain_t brain, float threshold) {
    // Guard: NULL check
    if (!brain) {
        NIMCP_LOGGING_ERROR("brain_detect_hubs: NULL brain");
        return false;
    }

    brain_clear_error();

    // 1. Build graph from brain topology
    NimcpGraph* graph = brain_build_topology_graph(brain);
    if (!graph) {
        NIMCP_LOGGING_ERROR("brain_detect_hubs: failed to build graph");
        return false;
    }

    // 2. Run hub detection algorithm
    adaptive_network_t network = brain_get_network(brain);
    if (!network) {
        nimcp_graph_destroy(graph);
        NIMCP_LOGGING_ERROR("brain_detect_hubs: failed to get adaptive network");
        return false;
    }
    neural_network_t base_network = adaptive_network_get_base_network(network);
    if (!base_network) {
        nimcp_graph_destroy(graph);
        NIMCP_LOGGING_ERROR("brain_detect_hubs: failed to get base network");
        return false;
    }
    hub_structure_t* hubs = community_detect_hubs(base_network, threshold);
    if (!hubs) {
        nimcp_graph_destroy(graph);
        NIMCP_LOGGING_ERROR("brain_detect_hubs: hub detection failed");
        return false;
    }

    // 3. Store results (replace old if exists)
    if (brain->network_hubs) {
        hub_structure_free(brain->network_hubs);
    }
    brain->network_hubs = hubs;

    // 4. Log results
    NIMCP_LOGGING_INFO("Detected %u hub neurons (threshold=%.1f std above mean)",
                   hubs->num_hubs, threshold);

    // Log top hubs
    uint32_t num_to_log = hubs->num_hubs < 10 ? hubs->num_hubs : 10;
    for (uint32_t i = 0; i < num_to_log; i++) {
        NIMCP_LOGGING_INFO("  Hub %u: neuron %u (score=%.1f)",
                      i, hubs->hub_indices[i], hubs->degree_centrality[i]);
    }
    if (hubs->num_hubs > 10) {
        NIMCP_LOGGING_INFO("  ... (%u more hubs)", hubs->num_hubs - 10);
    }

    nimcp_graph_destroy(graph);
    return true;
}

NIMCP_EXPORT bool brain_is_hub_neuron(brain_t brain, uint32_t neuron_id) {
    // Guard: NULL check
    if (!brain) {
        return false;
    }

    // Check if hubs detected
    if (!brain->network_hubs) {
        NIMCP_LOGGING_ERROR("brain_is_hub_neuron: no hubs detected (call brain_detect_hubs first)");
        return false;
    }

    // Fast lookup - check if neuron_id is in hub_indices array
    hub_structure_t* hubs = brain->network_hubs;
    for (uint32_t i = 0; i < hubs->num_hubs; i++) {
        if (hubs->hub_indices[i] == neuron_id) {
            return true;
        }
    }
    return false;
}

//=============================================================================
// Topology Metrics
//=============================================================================

NIMCP_EXPORT bool brain_compute_topology_metrics(brain_t brain) {
    // Guard: NULL check
    if (!brain) {
        NIMCP_LOGGING_ERROR("brain_compute_topology_metrics: NULL brain");
        return false;
    }

    brain_clear_error();

    // 1. Build graph from brain topology
    NimcpGraph* graph = brain_build_topology_graph(brain);
    if (!graph) {
        NIMCP_LOGGING_ERROR("brain_compute_topology_metrics: failed to build graph");
        return false;
    }

    // 2. Compute topology metrics using validate function
    // Convert NimcpGraph to neural_network_t
    adaptive_network_t network = brain_get_network(brain);
    if (!network) {
        nimcp_graph_destroy(graph);
        NIMCP_LOGGING_ERROR("brain_compute_topology_metrics: failed to get network");
        return false;
    }
    neural_network_t base_network = adaptive_network_get_base_network(network);
    if (!base_network) {
        nimcp_graph_destroy(graph);
        NIMCP_LOGGING_ERROR("brain_compute_topology_metrics: failed to get base network");
        return false;
    }

    topology_validation_t validation = community_validate_topology(base_network, 0.0f);

    // 3. Store results (replace old if exists)
    if (brain->topology_metrics) {
        nimcp_free(brain->topology_metrics);
    }
    brain->topology_metrics = nimcp_malloc(sizeof(topology_validation_t));
    if (!brain->topology_metrics) {
        nimcp_graph_destroy(graph);
        return false;
    }
    *brain->topology_metrics = validation;

    // 4. Log results
    NIMCP_LOGGING_INFO("Topology metrics:");
    NIMCP_LOGGING_INFO("  Modularity Q: %.3f", brain->topology_metrics->modularity);
    NIMCP_LOGGING_INFO("  Clustering C: %.3f", brain->topology_metrics->clustering_coefficient);
    NIMCP_LOGGING_INFO("  Avg path length L: %.2f", brain->topology_metrics->characteristic_path);
    NIMCP_LOGGING_INFO("  Small-world σ: %.2f", brain->topology_metrics->small_world_sigma);
    NIMCP_LOGGING_INFO("  Communities: %u", brain->topology_metrics->num_communities);
    NIMCP_LOGGING_INFO("  Hubs: %u", brain->topology_metrics->num_hubs);

    // Interpret results
    if (brain->topology_metrics->modularity > 0.5f) {
        NIMCP_LOGGING_INFO("  → Excellent community structure");
    } else if (brain->topology_metrics->modularity > 0.3f) {
        NIMCP_LOGGING_INFO("  → Good community structure");
    } else {
        NIMCP_LOGGING_WARN("  → Weak community structure (Q < 0.3)");
    }

    if (brain->topology_metrics->small_world_sigma > 1.0f) {
        NIMCP_LOGGING_INFO("  → Small-world network (efficient)");
    } else {
        NIMCP_LOGGING_WARN("  → Not small-world (σ < 1.0)");
    }

    nimcp_graph_destroy(graph);
    return true;
}

NIMCP_EXPORT bool brain_validate_topology(brain_t brain) {
    // Guard: NULL check
    if (!brain) {
        NIMCP_LOGGING_ERROR("brain_validate_topology: NULL brain");
        return false;
    }

    brain_clear_error();

    // 1. Build graph from brain topology
    NimcpGraph* graph = brain_build_topology_graph(brain);
    if (!graph) {
        NIMCP_LOGGING_ERROR("brain_validate_topology: failed to build graph");
        return false;
    }

    // 2. Run validation
    adaptive_network_t network = brain_get_network(brain);
    if (!network) {
        nimcp_graph_destroy(graph);
        NIMCP_LOGGING_ERROR("brain_validate_topology: failed to get network");
        return false;
    }
    neural_network_t base_network = adaptive_network_get_base_network(network);
    if (!base_network) {
        nimcp_graph_destroy(graph);
        NIMCP_LOGGING_ERROR("brain_validate_topology: failed to get base network");
        return false;
    }

    topology_validation_t validation = community_validate_topology(base_network, 0.25f);
    bool is_valid = validation.is_valid;

    if (!is_valid) {
        NIMCP_LOGGING_ERROR("Brain topology validation failed: %s", validation.error_message);
        NIMCP_LOGGING_ERROR("Topology validation failed: %s", validation.error_message);
    } else {
        NIMCP_LOGGING_INFO("Topology validation passed: network is healthy");
    }

    nimcp_graph_destroy(graph);
    return is_valid;
}

//=============================================================================
// Network Analyzer API
//=============================================================================

NIMCP_EXPORT void* brain_get_network_analyzer(brain_t brain) {
    // WHAT: Get or create network analyzer for real-time topology analysis
    // WHY:  Enables continuous monitoring of network topology during inference
    // HOW:  Lazy initialization - create analyzer on first access, cache for reuse
    //
    // ALGORITHM:
    // 1. Validate brain handle and network existence
    // 2. Check if analyzer already exists (cached)
    // 3. If not cached, create new analyzer with network_analyzer_create()
    // 4. Configure analyzer for real-time monitoring:
    //    - Auto-analyze enabled with 10-iteration interval
    //    - Hub detection threshold at 0.7 (top 30% centrality)
    // 5. Run initial analysis to populate topology metrics
    // 6. Cache analyzer in brain structure for reuse
    // 7. Return analyzer pointer
    //
    // INTEGRATION:
    // - Used by consolidation (memory optimization based on communities)
    // - Used by curiosity (novelty from community emergence)
    // - Used by meta-learning (architecture search using topology)
    // - Used by quantum-Shannon (adaptive routing based on hubs)
    //
    // PERFORMANCE:
    // - O(1) after first creation (cached lookup)
    // - Initial creation: O(N + E) where N=neurons, E=synapses
    // - Initial analysis: O(N log N) for Louvain algorithm
    // - Typical overhead: 50-200ms for first call, <1μs for subsequent calls
    //
    // COMPLEXITY: O(1) cached, O(N + E) first call

    // Guard: NULL check
    if (!brain) {
        NIMCP_LOGGING_ERROR("brain_get_network_analyzer: NULL brain");
        return NULL;
    }

    struct brain_struct* b = (struct brain_struct*)brain;

    // Guard: Check if network is valid
    if (!b->network) {
        NIMCP_LOGGING_ERROR("brain_get_network_analyzer: brain has no network");
        return NULL;
    }

    // Lazy initialization: Create analyzer on first access
    if (!b->network_analyzer) {
        // Create network analyzer for this brain
        network_analyzer_t* analyzer = network_analyzer_create(brain);

        // Guard: Creation failure
        if (!analyzer) {
            NIMCP_LOGGING_ERROR("brain_get_network_analyzer: failed to create analyzer");
            return NULL;
        }

        // Configure analyzer for real-time analysis during inference
        network_analyzer_set_auto_analyze(analyzer, true, 10);  // Auto-run every 10 iterations
        network_analyzer_set_hub_threshold(analyzer, 0.7f);      // Detect high-centrality hubs

        // Cache analyzer in brain structure
        b->network_analyzer = (void*)analyzer;

        // Initial analysis to populate topology metrics
        if (!network_analyzer_run(analyzer)) {
            NIMCP_LOGGING_WARN("brain_get_network_analyzer: initial analysis failed, will retry");
        }

        NIMCP_LOGGING_INFO("brain_get_network_analyzer: created and initialized analyzer");
    }

    // Return cached analyzer
    return b->network_analyzer;
}
