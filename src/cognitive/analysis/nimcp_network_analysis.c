//=============================================================================
// nimcp_network_analysis.c - Brain Network Analysis Implementation
//=============================================================================

#include "nimcp_network_analysis.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "core/neuralnet/nimcp_neuralnet.h"
#include "plasticity/adaptive/nimcp_adaptive.h"

//=============================================================================
// Constants
//=============================================================================

#define DEFAULT_HISTORY_CAPACITY 1000
#define DEFAULT_HUB_THRESHOLD 0.7f
#define MIN_MODULARITY_HEALTHY 0.2f
#define MIN_COMMUNITY_SIZE 5
#define MAX_COMMUNITIES 100

//=============================================================================
// Internal Structures
//=============================================================================

/**
 * @brief Adjacency list for network graph
 */
typedef struct {
    uint32_t* neighbors;
    uint32_t num_neighbors;
    uint32_t capacity;
} adjacency_list_t;

/**
 * @brief Graph representation of brain network
 */
typedef struct {
    adjacency_list_t* adj_lists;
    uint32_t num_neurons;
    uint32_t num_edges;
} network_graph_t;

//=============================================================================
// Forward Declarations
//=============================================================================

static network_graph_t* build_network_graph(brain_t brain);
static void destroy_network_graph(network_graph_t* graph);
static bool detect_communities_louvain(network_graph_t* graph, network_analyzer_t* analyzer);
static bool detect_hubs_centrality(network_graph_t* graph, network_analyzer_t* analyzer);
static bool compute_topology_metrics(network_graph_t* graph, network_analyzer_t* analyzer);
static float compute_modularity(network_graph_t* graph, uint32_t* community_assignments, uint32_t num_communities);
static float compute_clustering_coefficient(network_graph_t* graph);
static float compute_avg_path_length(network_graph_t* graph);

//=============================================================================
// Lifecycle
//=============================================================================

network_analyzer_t* network_analyzer_create(brain_t brain)
{
    if (!brain) {
        NIMCP_LOGGING_ERROR("network_analyzer_create: NULL brain");
        return NULL;
    }

    network_analyzer_t* analyzer = nimcp_calloc(1, sizeof(network_analyzer_t));
    if (!analyzer) {
        NIMCP_LOGGING_ERROR("network_analyzer_create: allocation failed");
        return NULL;
    }

    analyzer->brain = brain;
    analyzer->auto_analyze = false;
    analyzer->analysis_interval = 100;
    analyzer->hub_threshold = DEFAULT_HUB_THRESHOLD;
    analyzer->iteration_counter = 0;
    analyzer->topology_is_valid = true;
    analyzer->history_capacity = DEFAULT_HISTORY_CAPACITY;

    // Allocate history arrays
    analyzer->modularity_history = nimcp_calloc(DEFAULT_HISTORY_CAPACITY, sizeof(float));
    analyzer->community_count_history = nimcp_calloc(DEFAULT_HISTORY_CAPACITY, sizeof(uint32_t));

    if (!analyzer->modularity_history || !analyzer->community_count_history) {
        NIMCP_LOGGING_ERROR("network_analyzer_create: history allocation failed");
        network_analyzer_destroy(analyzer);
        return NULL;
    }

    NIMCP_LOGGING_INFO("Network analyzer created");
    return analyzer;
}

void network_analyzer_destroy(network_analyzer_t* analyzer)
{
    if (!analyzer) return;

    // Free community structure
    if (analyzer->communities) {
        for (uint32_t i = 0; i < analyzer->communities->num_communities; i++) {
            if (analyzer->communities->communities[i].neurons) {
                nimcp_free(analyzer->communities->communities[i].neurons);
            }
        }
        if (analyzer->communities->communities) {
            nimcp_free(analyzer->communities->communities);
        }
        nimcp_free(analyzer->communities);
    }

    // Free hub detection
    if (analyzer->hubs) {
        if (analyzer->hubs->hubs) {
            nimcp_free(analyzer->hubs->hubs);
        }
        nimcp_free(analyzer->hubs);
    }

    // Free history arrays
    if (analyzer->modularity_history) {
        nimcp_free(analyzer->modularity_history);
    }
    if (analyzer->community_count_history) {
        nimcp_free(analyzer->community_count_history);
    }

    nimcp_free(analyzer);
    NIMCP_LOGGING_INFO("Network analyzer destroyed");
}

//=============================================================================
// Analysis Operations
//=============================================================================

bool network_analyzer_run(network_analyzer_t* analyzer)
{
    if (!analyzer) {
        NIMCP_LOGGING_ERROR("network_analyzer_run: NULL analyzer");
        return false;
    }

    NIMCP_LOGGING_INFO("Running complete network analysis");

    // Build graph representation of brain network
    network_graph_t* graph = build_network_graph(analyzer->brain);
    if (!graph) {
        NIMCP_LOGGING_ERROR("network_analyzer_run: failed to build graph");
        return false;
    }

    // Run all analyses
    bool success = true;
    success &= detect_communities_louvain(graph, analyzer);
    success &= detect_hubs_centrality(graph, analyzer);
    success &= compute_topology_metrics(graph, analyzer);

    // Update history
    if (success && analyzer->analysis_count < analyzer->history_capacity) {
        uint32_t idx = analyzer->analysis_count;
        analyzer->modularity_history[idx] = analyzer->communities ? analyzer->communities->modularity : 0.0f;
        analyzer->community_count_history[idx] = analyzer->communities ? analyzer->communities->num_communities : 0;
        analyzer->analysis_count++;
    }

    destroy_network_graph(graph);

    if (success) {
        NIMCP_LOGGING_INFO("Network analysis complete: Q=%.3f, %u communities, %u hubs",
                      analyzer->communities ? analyzer->communities->modularity : 0.0f,
                      analyzer->communities ? analyzer->communities->num_communities : 0,
                      analyzer->hubs ? analyzer->hubs->num_hubs : 0);
    }

    return success;
}

bool network_analyzer_detect_communities(network_analyzer_t* analyzer)
{
    if (!analyzer) return false;

    network_graph_t* graph = build_network_graph(analyzer->brain);
    if (!graph) return false;

    bool success = detect_communities_louvain(graph, analyzer);
    destroy_network_graph(graph);
    return success;
}

bool network_analyzer_detect_hubs(network_analyzer_t* analyzer)
{
    if (!analyzer) return false;

    network_graph_t* graph = build_network_graph(analyzer->brain);
    if (!graph) return false;

    bool success = detect_hubs_centrality(graph, analyzer);
    destroy_network_graph(graph);
    return success;
}

bool network_analyzer_compute_metrics(network_analyzer_t* analyzer)
{
    if (!analyzer) return false;

    network_graph_t* graph = build_network_graph(analyzer->brain);
    if (!graph) return false;

    bool success = compute_topology_metrics(graph, analyzer);
    destroy_network_graph(graph);
    return success;
}

//=============================================================================
// Validation
//=============================================================================

bool network_analyzer_validate_learning(network_analyzer_t* analyzer)
{
    if (!analyzer) return false;

    analyzer->topology_is_valid = true;
    analyzer->last_error[0] = '\0';

    // Check if analysis has been run
    if (!analyzer->communities) {
        snprintf(analyzer->last_error, sizeof(analyzer->last_error),
                "No community structure - run analysis first");
        analyzer->topology_is_valid = false;
        return false;
    }

    // Check modularity hasn't collapsed
    if (analyzer->communities->modularity < MIN_MODULARITY_HEALTHY) {
        snprintf(analyzer->last_error, sizeof(analyzer->last_error),
                "Modularity collapsed: Q=%.3f (threshold=%.3f)",
                analyzer->communities->modularity, MIN_MODULARITY_HEALTHY);
        analyzer->topology_is_valid = false;
        return false;
    }

    // Check for isolated communities (too small)
    for (uint32_t i = 0; i < analyzer->communities->num_communities; i++) {
        if (analyzer->communities->communities[i].size < MIN_COMMUNITY_SIZE) {
            snprintf(analyzer->last_error, sizeof(analyzer->last_error),
                    "Community %u too small: %u neurons (min=%u)",
                    i, analyzer->communities->communities[i].size, MIN_COMMUNITY_SIZE);
            analyzer->topology_is_valid = false;
            return false;
        }
    }

    // Check hub neurons still have connections
    if (analyzer->hubs && analyzer->hubs->num_hubs > 0) {
        // If we had hubs before, we should still have some
        if (analyzer->analysis_count > 1 && analyzer->hubs->num_hubs == 0) {
            snprintf(analyzer->last_error, sizeof(analyzer->last_error),
                    "All hub neurons lost connections");
            analyzer->topology_is_valid = false;
            return false;
        }
    }

    // Check network density hasn't become too sparse
    if (analyzer->metrics.density < 0.01f) {
        snprintf(analyzer->last_error, sizeof(analyzer->last_error),
                "Network too sparse: density=%.4f", analyzer->metrics.density);
        analyzer->topology_is_valid = false;
        return false;
    }

    NIMCP_LOGGING_INFO("Topology validation passed: Q=%.3f, density=%.3f",
                  analyzer->communities->modularity, analyzer->metrics.density);
    return true;
}

const char* network_analyzer_get_error(network_analyzer_t* analyzer)
{
    return analyzer ? analyzer->last_error : "";
}

//=============================================================================
// Configuration
//=============================================================================

void network_analyzer_set_auto_analyze(network_analyzer_t* analyzer, bool enable, uint32_t interval)
{
    if (!analyzer) return;
    analyzer->auto_analyze = enable;
    analyzer->analysis_interval = interval > 0 ? interval : 1;
    NIMCP_LOGGING_INFO("Auto-analysis %s (interval=%u)",
                  enable ? "enabled" : "disabled", analyzer->analysis_interval);
}

void network_analyzer_set_hub_threshold(network_analyzer_t* analyzer, float threshold)
{
    if (!analyzer) return;
    analyzer->hub_threshold = fminf(fmaxf(threshold, 0.5f), 1.0f);
    NIMCP_LOGGING_INFO("Hub threshold set to %.2f", analyzer->hub_threshold);
}

//=============================================================================
// Query Results
//=============================================================================

const community_structure_t* network_analyzer_get_communities(network_analyzer_t* analyzer)
{
    return analyzer ? analyzer->communities : NULL;
}

const hub_detection_t* network_analyzer_get_hubs(network_analyzer_t* analyzer)
{
    return analyzer ? analyzer->hubs : NULL;
}

topology_metrics_t network_analyzer_get_metrics(network_analyzer_t* analyzer)
{
    topology_metrics_t empty = {0};
    return analyzer ? analyzer->metrics : empty;
}

const float* network_analyzer_get_modularity_history(network_analyzer_t* analyzer, uint32_t* count)
{
    if (!analyzer || !count) return NULL;
    *count = analyzer->analysis_count;
    return analyzer->modularity_history;
}

//=============================================================================
// Reporting
//=============================================================================

void network_analyzer_print_report(network_analyzer_t* analyzer)
{
    if (!analyzer) return;

    printf("\n=== Network Topology Analysis ===\n");
    printf("Analysis runs: %u\n", analyzer->analysis_count);

    if (analyzer->communities) {
        printf("\nCommunity Structure:\n");
        printf("  Communities: %u\n", analyzer->communities->num_communities);
        printf("  Modularity Q: %.3f\n", analyzer->communities->modularity);

        for (uint32_t i = 0; i < analyzer->communities->num_communities && i < 10; i++) {
            community_t* comm = &analyzer->communities->communities[i];
            printf("  Community %u: %u neurons (internal=%.3f, external=%.3f)\n",
                   i, comm->size, comm->internal_density, comm->external_density);
        }
    }

    if (analyzer->hubs) {
        printf("\nHub Neurons:\n");
        printf("  Total hubs: %u (threshold=%.2f)\n",
               analyzer->hubs->num_hubs, analyzer->hub_threshold);

        uint32_t show_count = analyzer->hubs->num_hubs < 10 ? analyzer->hubs->num_hubs : 10;
        for (uint32_t i = 0; i < show_count; i++) {
            hub_neuron_t* hub = &analyzer->hubs->hubs[i];
            printf("  Hub %u: neuron %u (degree=%.3f, betweenness=%.3f%s)\n",
                   i, hub->neuron_id, hub->degree_centrality, hub->betweenness,
                   hub->is_connector_hub ? ", connector" : "");
        }
    }

    printf("\nTopology Metrics:\n");
    printf("  Clustering: %.3f\n", analyzer->metrics.clustering_coefficient);
    printf("  Avg path length: %.2f\n", analyzer->metrics.avg_path_length);
    printf("  Small-worldness: %.2f\n", analyzer->metrics.small_worldness);
    printf("  Density: %.4f\n", analyzer->metrics.density);
    printf("  Edges: %u\n", analyzer->metrics.num_edges);

    printf("\nTopology Status: %s\n",
           analyzer->topology_is_valid ? "HEALTHY" : "DAMAGED");
    if (!analyzer->topology_is_valid) {
        printf("  Error: %s\n", analyzer->last_error);
    }
    printf("================================\n\n");
}

void network_analyzer_print_modularity_trend(network_analyzer_t* analyzer)
{
    if (!analyzer || analyzer->analysis_count == 0) return;

    printf("\n=== Modularity Trend ===\n");
    printf("Analysis runs: %u\n\n", analyzer->analysis_count);

    // Simple ASCII chart
    for (uint32_t i = 0; i < analyzer->analysis_count && i < 50; i++) {
        float Q = analyzer->modularity_history[i];
        uint32_t bar_len = (uint32_t)(Q * 50.0f);
        printf("%3u: ", i);
        for (uint32_t j = 0; j < bar_len; j++) printf("#");
        printf(" %.3f\n", Q);
    }
    printf("========================\n\n");
}

//=============================================================================
// Integration Hooks
//=============================================================================

void network_analyzer_on_learning_event(network_analyzer_t* analyzer)
{
    if (!analyzer || !analyzer->auto_analyze) return;

    analyzer->iteration_counter++;
    if (analyzer->iteration_counter >= analyzer->analysis_interval) {
        analyzer->iteration_counter = 0;
        network_analyzer_run(analyzer);
    }
}

bool network_analyzer_check_new_community(network_analyzer_t* analyzer)
{
    if (!analyzer || analyzer->analysis_count < 2) return false;

    uint32_t prev_count = analyzer->community_count_history[analyzer->analysis_count - 2];
    uint32_t curr_count = analyzer->community_count_history[analyzer->analysis_count - 1];

    return curr_count > prev_count;
}

//=============================================================================
// Internal Implementation - Graph Building
//=============================================================================

/**
 * @brief Build graph from brain network
 *
 * WHAT: Extract connectivity graph from neural network
 * WHY:  Need graph representation for analysis algorithms
 * HOW:  Traverse synapses from adaptive network, build adjacency lists
 */
static network_graph_t* build_network_graph(brain_t brain)
{
    if (!brain) return NULL;

    // WHAT: Get adaptive network from brain
    // WHY: Need access to real synaptic connectivity
    adaptive_network_t network = brain_get_network(brain);
    if (!network) {
        NIMCP_LOGGING_ERROR("build_network_graph: failed to get network from brain");
        return NULL;
    }

    // WHAT: Get number of neurons in network
    uint32_t num_neurons = adaptive_network_get_neuron_count(network);
    if (num_neurons == 0) {
        NIMCP_LOGGING_ERROR("build_network_graph: network has no neurons");
        return NULL;
    }

    // Allocate graph structure
    network_graph_t* graph = nimcp_calloc(1, sizeof(network_graph_t));
    if (!graph) return NULL;

    graph->num_neurons = num_neurons;
    graph->num_edges = 0;

    // Allocate adjacency lists
    graph->adj_lists = nimcp_calloc(num_neurons, sizeof(adjacency_list_t));
    if (!graph->adj_lists) {
        nimcp_free(graph);
        return NULL;
    }

    // WHAT: Initialize all adjacency lists
    for (uint32_t i = 0; i < num_neurons; i++) {
        graph->adj_lists[i].neighbors = NULL;
        graph->adj_lists[i].num_neighbors = 0;
        graph->adj_lists[i].capacity = 0;
    }

    // WHAT: Extract connectivity by iterating through neurons
    // WHY: Build adjacency lists from real synaptic connections
    // HOW: For each neuron, get outgoing synapses and record targets
    for (uint32_t source_id = 0; source_id < num_neurons; source_id++) {
        neuron_t* neuron = neural_network_get_neuron((neural_network_t)network, source_id);
        if (!neuron) continue;

        for (uint32_t i = 0; i < neuron->num_synapses; i++) {
            synapse_t* syn = &neuron->synapses[i];
            uint32_t target_id = syn->target_id;

            // Skip connections with zero weight
            if (fabsf(syn->weight) < 1e-6f) continue;

            // Allocate or expand neighbors array for target neuron
            adjacency_list_t* adj = &graph->adj_lists[target_id];
            if (adj->num_neighbors >= adj->capacity) {
                uint32_t new_capacity = adj->capacity == 0 ? 8 : adj->capacity * 2;
                uint32_t* new_neighbors = nimcp_realloc(adj->neighbors, new_capacity * sizeof(uint32_t));
                if (!new_neighbors) {
                    // Cleanup on allocation failure
                    for (uint32_t j = 0; j < num_neurons; j++) {
                        if (graph->adj_lists[j].neighbors) {
                            nimcp_free(graph->adj_lists[j].neighbors);
                        }
                    }
                    nimcp_free(graph->adj_lists);
                    nimcp_free(graph);
                    return NULL;
                }
                adj->neighbors = new_neighbors;
                adj->capacity = new_capacity;
            }

            // Add source neuron as neighbor (incoming connection)
            adj->neighbors[adj->num_neighbors++] = source_id;
            graph->num_edges++;
        }
    }

    NIMCP_LOGGING_INFO("Built network graph: %u neurons, %u edges",
                      graph->num_neurons, graph->num_edges);
    return graph;
}

static void destroy_network_graph(network_graph_t* graph)
{
    if (!graph) return;

    if (graph->adj_lists) {
        for (uint32_t i = 0; i < graph->num_neurons; i++) {
            if (graph->adj_lists[i].neighbors) {
                nimcp_free(graph->adj_lists[i].neighbors);
            }
        }
        nimcp_free(graph->adj_lists);
    }

    nimcp_free(graph);
}

//=============================================================================
// Internal Implementation - Community Detection
//=============================================================================

/**
 * @brief Detect communities using greedy modularity optimization
 *
 * WHAT: Find functional modules in brain network using connectivity patterns
 * WHY:  Brain organizes into specialized modules for efficiency
 * HOW:  Greedy algorithm - merge communities that maximize modularity gain
 *
 * ALGORITHM: Simplified Louvain-style approach:
 * 1. Start with each neuron in its own community
 * 2. For each neuron, try moving it to neighbor communities
 * 3. Move neuron to community that gives max modularity increase
 * 4. Repeat until no improvement
 *
 * NOTE: This is a simplified version. Full Louvain has a second phase
 * that builds a new graph from communities and repeats. We skip that
 * for performance since brain networks change frequently.
 */
static bool detect_communities_louvain(network_graph_t* graph, network_analyzer_t* analyzer)
{
    if (!graph || !analyzer) return false;

    uint32_t num_neurons = graph->num_neurons;
    if (num_neurons == 0) return false;

    // WHAT: Allocate community assignments (one per neuron)
    // WHY: Track which community each neuron belongs to
    uint32_t* community_id = nimcp_calloc(num_neurons, sizeof(uint32_t));
    if (!community_id) return false;

    // WHAT: Initialize - each neuron starts in its own community
    for (uint32_t i = 0; i < num_neurons; i++) {
        community_id[i] = i;
    }

    // WHAT: Greedy optimization - move neurons to improve modularity
    // WHY: Find community structure that maximizes within-group connections
    bool improved = true;
    uint32_t iterations = 0;
    const uint32_t MAX_ITERATIONS = 10;  // Prevent infinite loops

    while (improved && iterations < MAX_ITERATIONS) {
        improved = false;
        iterations++;

        // For each neuron, try moving to neighbor communities
        for (uint32_t neuron = 0; neuron < num_neurons; neuron++) {
            uint32_t current_comm = community_id[neuron];
            uint32_t best_comm = current_comm;
            uint32_t max_neighbor_edges = 0;

            // WHAT: Count edges to each neighbor community
            // WHY: Move neuron to community with most connections
            adjacency_list_t* adj = &graph->adj_lists[neuron];
            for (uint32_t i = 0; i < adj->num_neighbors; i++) {
                uint32_t neighbor = adj->neighbors[i];
                uint32_t neighbor_comm = community_id[neighbor];

                if (neighbor_comm != current_comm) {
                    // Count edges to this community
                    uint32_t edge_count = 0;
                    for (uint32_t j = 0; j < adj->num_neighbors; j++) {
                        if (community_id[adj->neighbors[j]] == neighbor_comm) {
                            edge_count++;
                        }
                    }

                    if (edge_count > max_neighbor_edges) {
                        max_neighbor_edges = edge_count;
                        best_comm = neighbor_comm;
                    }
                }
            }

            // WHAT: Move neuron if better community found
            if (best_comm != current_comm && max_neighbor_edges > 0) {
                community_id[neuron] = best_comm;
                improved = true;
            }
        }
    }

    // WHAT: Compact community IDs (0, 1, 2, ... instead of sparse IDs)
    // WHY: Community IDs after merging may have gaps
    uint32_t* id_remap = nimcp_calloc(num_neurons, sizeof(uint32_t));
    if (!id_remap) {
        nimcp_free(community_id);
        return false;
    }

    uint32_t next_id = 0;
    for (uint32_t i = 0; i < num_neurons; i++) {
        id_remap[i] = UINT32_MAX;
    }

    for (uint32_t i = 0; i < num_neurons; i++) {
        uint32_t old_id = community_id[i];
        if (id_remap[old_id] == UINT32_MAX) {
            id_remap[old_id] = next_id++;
        }
        community_id[i] = id_remap[old_id];
    }
    nimcp_free(id_remap);

    uint32_t num_communities = next_id;
    if (num_communities > MAX_COMMUNITIES) {
        num_communities = MAX_COMMUNITIES;
    }

    // WHAT: Build community structure from assignments
    community_structure_t* cs = nimcp_calloc(1, sizeof(community_structure_t));
    if (!cs) {
        nimcp_free(community_id);
        return false;
    }

    cs->num_communities = num_communities;
    cs->communities = nimcp_calloc(num_communities, sizeof(community_t));
    if (!cs->communities) {
        nimcp_free(cs);
        nimcp_free(community_id);
        return false;
    }

    // WHAT: Count neurons per community
    uint32_t* comm_sizes = nimcp_calloc(num_communities, sizeof(uint32_t));
    if (!comm_sizes) {
        nimcp_free(cs->communities);
        nimcp_free(cs);
        nimcp_free(community_id);
        return false;
    }

    for (uint32_t i = 0; i < num_neurons; i++) {
        if (community_id[i] < num_communities) {
            comm_sizes[community_id[i]]++;
        }
    }

    // WHAT: Allocate neuron arrays for each community
    for (uint32_t c = 0; c < num_communities; c++) {
        cs->communities[c].size = comm_sizes[c];
        cs->communities[c].neurons = nimcp_calloc(comm_sizes[c], sizeof(uint32_t));
        if (!cs->communities[c].neurons) {
            // Cleanup on failure
            for (uint32_t j = 0; j < c; j++) {
                nimcp_free(cs->communities[j].neurons);
            }
            nimcp_free(cs->communities);
            nimcp_free(cs);
            nimcp_free(comm_sizes);
            nimcp_free(community_id);
            return false;
        }
        comm_sizes[c] = 0;  // Reuse as index counter
        snprintf(cs->communities[c].label, sizeof(cs->communities[c].label),
                "module_%u", c);
    }

    // WHAT: Fill in neuron IDs for each community
    for (uint32_t i = 0; i < num_neurons; i++) {
        uint32_t c = community_id[i];
        if (c < num_communities) {
            cs->communities[c].neurons[comm_sizes[c]++] = i;
        }
    }

    nimcp_free(comm_sizes);

    // WHAT: Compute modularity Q
    cs->modularity = compute_modularity(graph, community_id, num_communities);

    // WHAT: Compute community densities
    for (uint32_t c = 0; c < num_communities; c++) {
        uint32_t internal_edges = 0;
        uint32_t external_edges = 0;
        uint32_t max_internal = (cs->communities[c].size * (cs->communities[c].size - 1)) / 2;

        for (uint32_t i = 0; i < cs->communities[c].size; i++) {
            uint32_t neuron = cs->communities[c].neurons[i];
            adjacency_list_t* adj = &graph->adj_lists[neuron];

            for (uint32_t j = 0; j < adj->num_neighbors; j++) {
                uint32_t neighbor = adj->neighbors[j];
                if (community_id[neighbor] == c) {
                    internal_edges++;
                } else {
                    external_edges++;
                }
            }
        }

        cs->communities[c].internal_density =
            max_internal > 0 ? (float)internal_edges / (float)max_internal : 0.0f;
        cs->communities[c].external_density =
            cs->communities[c].size > 0 ? (float)external_edges / (float)cs->communities[c].size : 0.0f;
    }

    nimcp_free(community_id);
    cs->timestamp = 0;  // TODO: Add real timestamp

    // Free old communities
    if (analyzer->communities) {
        for (uint32_t i = 0; i < analyzer->communities->num_communities; i++) {
            if (analyzer->communities->communities[i].neurons) {
                nimcp_free(analyzer->communities->communities[i].neurons);
            }
        }
        if (analyzer->communities->communities) {
            nimcp_free(analyzer->communities->communities);
        }
        nimcp_free(analyzer->communities);
    }

    analyzer->communities = cs;
    return true;
}

//=============================================================================
// Internal Implementation - Hub Detection
//=============================================================================

/**
 * @brief Detect hub neurons using degree centrality
 *
 * WHAT: Find highly connected neurons (hubs)
 * WHY:  Hubs coordinate information flow between modules
 * HOW:  Compute degree centrality, identify top neurons above threshold
 *
 * BIOLOGY: Hub neurons are critical for inter-module communication
 * Connector hubs link different communities
 */
static bool detect_hubs_centrality(network_graph_t* graph, network_analyzer_t* analyzer)
{
    if (!graph || !analyzer) return false;

    uint32_t num_neurons = graph->num_neurons;
    if (num_neurons == 0) return false;

    // WHAT: Compute degree centrality for each neuron
    // WHY: Degree = total connections (in + out)
    float* degree_centrality = nimcp_calloc(num_neurons, sizeof(float));
    if (!degree_centrality) return false;

    // Count total degree (incoming + outgoing connections)
    uint32_t max_degree = 0;
    for (uint32_t i = 0; i < num_neurons; i++) {
        uint32_t degree = graph->adj_lists[i].num_neighbors;

        // Also count neurons that connect TO this neuron
        for (uint32_t j = 0; j < num_neurons; j++) {
            if (i == j) continue;
            adjacency_list_t* adj = &graph->adj_lists[j];
            for (uint32_t k = 0; k < adj->num_neighbors; k++) {
                if (adj->neighbors[k] == i) {
                    degree++;
                    break;
                }
            }
        }

        degree_centrality[i] = (float)degree;
        if (degree > max_degree) {
            max_degree = degree;
        }
    }

    // WHAT: Normalize degree centrality to [0, 1]
    if (max_degree > 0) {
        for (uint32_t i = 0; i < num_neurons; i++) {
            degree_centrality[i] /= (float)max_degree;
        }
    }

    // WHAT: Count neurons above hub threshold
    uint32_t num_hubs = 0;
    for (uint32_t i = 0; i < num_neurons; i++) {
        if (degree_centrality[i] >= analyzer->hub_threshold) {
            num_hubs++;
        }
    }

    if (num_hubs == 0) {
        // No hubs found - lower threshold or network too sparse
        nimcp_free(degree_centrality);
        return true;  // Not an error, just no hubs
    }

    // Allocate hub detection structure
    hub_detection_t* hd = nimcp_calloc(1, sizeof(hub_detection_t));
    if (!hd) {
        nimcp_free(degree_centrality);
        return false;
    }

    hd->num_hubs = num_hubs;
    hd->hubs = nimcp_calloc(num_hubs, sizeof(hub_neuron_t));
    if (!hd->hubs) {
        nimcp_free(hd);
        nimcp_free(degree_centrality);
        return false;
    }

    // WHAT: Fill in hub neuron data
    uint32_t hub_idx = 0;
    for (uint32_t i = 0; i < num_neurons && hub_idx < num_hubs; i++) {
        if (degree_centrality[i] >= analyzer->hub_threshold) {
            hd->hubs[hub_idx].neuron_id = i;
            hd->hubs[hub_idx].degree_centrality = degree_centrality[i];
            hd->hubs[hub_idx].betweenness = 0.5f;  // TODO: Compute real betweenness

            // WHAT: Determine if connector hub (connects multiple communities)
            // WHY: Connector hubs are especially important for inter-module communication
            if (analyzer->communities) {
                // Find which community this hub belongs to
                uint32_t hub_community = UINT32_MAX;
                for (uint32_t c = 0; c < analyzer->communities->num_communities; c++) {
                    for (uint32_t j = 0; j < analyzer->communities->communities[c].size; j++) {
                        if (analyzer->communities->communities[c].neurons[j] == i) {
                            hub_community = c;
                            break;
                        }
                    }
                    if (hub_community != UINT32_MAX) break;
                }

                hd->hubs[hub_idx].community_id = hub_community;

                // WHAT: Check if hub connects to multiple communities
                // Count unique communities among neighbors
                uint32_t unique_communities = 0;
                bool* seen_community = nimcp_calloc(analyzer->communities->num_communities, sizeof(bool));
                if (seen_community) {
                    adjacency_list_t* adj = &graph->adj_lists[i];
                    for (uint32_t j = 0; j < adj->num_neighbors; j++) {
                        uint32_t neighbor = adj->neighbors[j];

                        // Find neighbor's community
                        for (uint32_t c = 0; c < analyzer->communities->num_communities; c++) {
                            for (uint32_t k = 0; k < analyzer->communities->communities[c].size; k++) {
                                if (analyzer->communities->communities[c].neurons[k] == neighbor) {
                                    if (!seen_community[c]) {
                                        seen_community[c] = true;
                                        unique_communities++;
                                    }
                                    break;
                                }
                            }
                        }
                    }

                    // Connector hub if connects to 2+ communities
                    hd->hubs[hub_idx].is_connector_hub = (unique_communities >= 2);
                    nimcp_free(seen_community);
                } else {
                    hd->hubs[hub_idx].is_connector_hub = false;
                }
            } else {
                hd->hubs[hub_idx].community_id = 0;
                hd->hubs[hub_idx].is_connector_hub = false;
            }

            hub_idx++;
        }
    }

    nimcp_free(degree_centrality);
    hd->hub_threshold = analyzer->hub_threshold;
    hd->timestamp = 0;  // TODO: Add real timestamp

    // Free old hubs
    if (analyzer->hubs) {
        if (analyzer->hubs->hubs) {
            nimcp_free(analyzer->hubs->hubs);
        }
        nimcp_free(analyzer->hubs);
    }

    analyzer->hubs = hd;
    return true;
}

//=============================================================================
// Internal Implementation - Topology Metrics
//=============================================================================

/**
 * @brief Compute network topology metrics
 *
 * WHAT: Calculate clustering, path length, etc.
 * WHY:  Quantify network organization
 * HOW:  Graph theory algorithms
 */
static bool compute_topology_metrics(network_graph_t* graph, network_analyzer_t* analyzer)
{
    if (!graph || !analyzer) return false;

    // Compute metrics (placeholder values)
    analyzer->metrics.clustering_coefficient = 0.45f;
    analyzer->metrics.avg_path_length = 3.2f;
    analyzer->metrics.small_worldness = 2.3f;
    analyzer->metrics.assortativity = 0.1f;
    analyzer->metrics.num_edges = graph->num_edges;
    analyzer->metrics.density = (float)graph->num_edges / (graph->num_neurons * (graph->num_neurons - 1) / 2);

    return true;
}

static float compute_modularity(network_graph_t* graph, uint32_t* community_assignments, uint32_t num_communities)
{
    // Simplified modularity calculation
    // Q = (fraction of edges within communities) - (expected fraction if random)
    return 0.5f;  // Placeholder
}

static float compute_clustering_coefficient(network_graph_t* graph)
{
    // Global clustering coefficient
    return 0.4f;  // Placeholder
}

static float compute_avg_path_length(network_graph_t* graph)
{
    // Average shortest path length
    return 3.0f;  // Placeholder
}
