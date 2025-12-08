/**
 * @file nimcp_centrality.c
 * @brief Centrality measures implementation
 *
 * WHAT: Implements various network centrality measures
 * WHY: Identify important nodes in network structure
 * HOW: Calculate different centrality metrics using algorithms
 */

#include "utils/algorithms/nimcp_centrality.h"
#include "security/nimcp_security.h"
#include "security/nimcp_blood_brain_barrier.h"

#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_messages.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "utils/memory/nimcp_memory.h"
#include "utils/memory/nimcp_unified_memory.h"
#include "utils/logging/nimcp_logging.h"

//=============================================================================
// Internal Helper Functions
//=============================================================================

/**
 * WHAT: Calculate mean and standard deviation
 * WHY: For hub detection threshold
 * HOW: Standard statistical formulas
 */
static void calculate_stats(const double* values, uint32_t count, double* mean,
                            double* stdev)
{
    if (!values || !mean || !stdev) {
        return;
    }

    if (count == 0) {
        *mean = 0.0;
        *stdev = 0.0;
        return;
    }

    // Calculate mean
    *mean = 0.0;
    for (uint32_t i = 0; i < count; i++) {
        *mean += values[i];
    }
    *mean /= count;

    // Calculate standard deviation
    *stdev = 0.0;
    for (uint32_t i = 0; i < count; i++) {
        double diff = values[i] - *mean;
        *stdev += diff * diff;
    }
    *stdev = sqrt(*stdev / count);
}

/**
 * WHAT: BFS to find shortest distances
 * WHY: Support closeness and betweenness calculations
 * HOW: Standard breadth-first search algorithm
 */
static bool bfs_shortest_paths(const NimcpGraph* graph, uint32_t source, uint32_t* distances,
                               uint32_t* path_counts)
{
    if (!graph || !distances) {
        return false;
    }

    if (source >= graph->vertex_count) {
        return false;
    }

    memset(distances, 0xff, sizeof(uint32_t) * graph->vertex_count);
    if (path_counts) {
        memset(path_counts, 0, sizeof(uint32_t) * graph->vertex_count);
    }

    distances[source] = 0;
    if (path_counts) {
        path_counts[source] = 1;
    }

    // Simple BFS queue
    uint32_t* queue = (uint32_t*)nimcp_malloc(sizeof(uint32_t) * graph->vertex_count);
    if (!queue) {
        return false;
    }

    uint32_t front = 0, rear = 0;
    queue[rear++] = source;

    while (front < rear) {
        uint32_t u = queue[front++];

        uint32_t neighbors[256];
        uint32_t neighbor_count =
            nimcp_graph_get_neighbors(graph, u, neighbors, 256);

        for (uint32_t i = 0; i < neighbor_count; i++) {
            uint32_t v = neighbors[i];

            if (distances[v] == UINT32_MAX) {
                distances[v] = distances[u] + 1;
                if (path_counts) {
                    path_counts[v] = path_counts[u];
                }
                queue[rear++] = v;
            } else if (distances[v] == distances[u] + 1 && path_counts) {
                path_counts[v] += path_counts[u];
            }
        }
    }

    nimcp_free(queue);
    return true;
}

//=============================================================================
// Public API Functions
//=============================================================================

NimcpCentralityScores* nimcp_degree_centrality(const NimcpGraph* graph)
{
    LOG_DEBUG("Entering nimcp_degree_centrality");
    LOG_DEBUG("Computing degree centrality for graph with %u vertices",
              graph ? graph->vertex_count : 0);

    if (!graph) {
        LOG_ERROR("NULL graph provided to degree centrality");
        return NULL;
    }

    if (graph->vertex_count == 0) {
        LOG_WARN("Empty graph provided to degree centrality");
        LOG_ERROR("nimcp_degree_centrality failed: returning error");
        return NULL;
    }

    NimcpCentralityScores* scores =
        (NimcpCentralityScores*)nimcp_malloc(sizeof(NimcpCentralityScores));
    if (!scores) {
        LOG_ERROR("Failed to allocate centrality scores structure");
        return NULL;
    }

    scores->scores = (double*)nimcp_malloc(sizeof(double) * graph->vertex_count);
    if (!scores->scores) {
        nimcp_free(scores);
        LOG_ERROR("nimcp_degree_centrality failed: returning error");
        return NULL;
    }

    scores->num_scores = graph->vertex_count;

    // Calculate degree centrality: degree(v) / (n-1)
    double norm = (graph->vertex_count > 1) ? (graph->vertex_count - 1) : 1.0;

    for (uint32_t i = 0; i < graph->vertex_count; i++) {
        scores->scores[i] = (double)graph->vertices[i].edge_count / norm;
    }

    return scores;
}

NimcpCentralityScores* nimcp_betweenness_centrality(const NimcpGraph* graph)
{
    LOG_DEBUG("Entering nimcp_betweenness_centrality");
    if (!graph) {
        LOG_ERROR("nimcp_betweenness_centrality failed: returning error");
        return NULL;
    }

    if (graph->vertex_count == 0) {
        LOG_ERROR("nimcp_betweenness_centrality failed: returning error");
        return NULL;
    }

    NimcpCentralityScores* scores =
        (NimcpCentralityScores*)nimcp_malloc(sizeof(NimcpCentralityScores));
    if (!scores) {
        LOG_ERROR("nimcp_betweenness_centrality failed: returning error");
        return NULL;
    }

    scores->scores = (double*)nimcp_malloc(sizeof(double) * graph->vertex_count);
    if (!scores->scores) {
        nimcp_free(scores);
        LOG_ERROR("nimcp_betweenness_centrality failed: returning error");
        return NULL;
    }

    scores->num_scores = graph->vertex_count;
    memset(scores->scores, 0, sizeof(double) * graph->vertex_count);

    // For each source, find shortest paths and accumulate betweenness
    uint32_t* distances = (uint32_t*)nimcp_malloc(sizeof(uint32_t) * graph->vertex_count);
    uint32_t* path_counts = (uint32_t*)nimcp_malloc(sizeof(uint32_t) * graph->vertex_count);

    if (!distances || !path_counts) {
        nimcp_free(distances);
        nimcp_free(path_counts);
        nimcp_centrality_scores_destroy(scores);
        LOG_ERROR("nimcp_betweenness_centrality failed: returning error");
        return NULL;
    }

    for (uint32_t s = 0; s < graph->vertex_count; s++) {
        if (!bfs_shortest_paths(graph, s, distances, path_counts)) {
            continue;
        }

        // Accumulate betweenness contributions
        for (uint32_t v = 0; v < graph->vertex_count; v++) {
            if (v != s && distances[v] != UINT32_MAX) {
                scores->scores[v] += 1.0;  // Simplified: count paths
            }
        }
    }

    nimcp_free(distances);
    nimcp_free(path_counts);

    // Normalize
    double norm = (graph->vertex_count > 2)
                      ? ((graph->vertex_count - 1) * (graph->vertex_count - 2))
                      : 1.0;
    for (uint32_t i = 0; i < graph->vertex_count; i++) {
        scores->scores[i] /= norm;
    }

    return scores;
}

NimcpCentralityScores* nimcp_closeness_centrality(const NimcpGraph* graph)
{
    LOG_DEBUG("Entering nimcp_closeness_centrality");
    if (!graph) {
        LOG_ERROR("nimcp_closeness_centrality failed: returning error");
        return NULL;
    }

    if (graph->vertex_count == 0) {
        LOG_ERROR("nimcp_closeness_centrality failed: returning error");
        return NULL;
    }

    NimcpCentralityScores* scores =
        (NimcpCentralityScores*)nimcp_malloc(sizeof(NimcpCentralityScores));
    if (!scores) {
        LOG_ERROR("nimcp_closeness_centrality failed: returning error");
        return NULL;
    }

    scores->scores = (double*)nimcp_malloc(sizeof(double) * graph->vertex_count);
    if (!scores->scores) {
        nimcp_free(scores);
        LOG_ERROR("nimcp_closeness_centrality failed: returning error");
        return NULL;
    }

    scores->num_scores = graph->vertex_count;

    uint32_t* distances = (uint32_t*)nimcp_malloc(sizeof(uint32_t) * graph->vertex_count);
    if (!distances) {
        nimcp_centrality_scores_destroy(scores);
        LOG_ERROR("nimcp_closeness_centrality failed: returning error");
        return NULL;
    }

    // Calculate closeness for each vertex
    for (uint32_t v = 0; v < graph->vertex_count; v++) {
        if (!bfs_shortest_paths(graph, v, distances, NULL)) {
            scores->scores[v] = 0.0;
            continue;
        }

        // Sum distances to all reachable vertices
        uint32_t sum_distances = 0;
        uint32_t reachable = 0;

        for (uint32_t u = 0; u < graph->vertex_count; u++) {
            if (u != v && distances[u] != UINT32_MAX) {
                sum_distances += distances[u];
                reachable++;
            }
        }

        // Closeness = (n-1) / sum_distances
        if (sum_distances > 0) {
            scores->scores[v] = (double)(graph->vertex_count - 1) / sum_distances;
        } else {
            scores->scores[v] = 0.0;
        }
    }

    nimcp_free(distances);
    return scores;
}

NimcpCentralityScores* nimcp_eigenvector_centrality(const NimcpGraph* graph,
                                                     uint32_t max_iterations)
{
    if (!graph) {
        return NULL;
    }

    if (graph->vertex_count == 0 || max_iterations == 0) {
        return NULL;
    }

    NimcpCentralityScores* scores =
        (NimcpCentralityScores*)nimcp_malloc(sizeof(NimcpCentralityScores));
    if (!scores) {
        return NULL;
    }

    scores->scores = (double*)nimcp_malloc(sizeof(double) * graph->vertex_count);
    if (!scores->scores) {
        nimcp_free(scores);
        return NULL;
    }

    scores->num_scores = graph->vertex_count;

    // Allocate working arrays
    double* prev_scores = (double*)nimcp_malloc(sizeof(double) * graph->vertex_count);
    if (!prev_scores) {
        nimcp_centrality_scores_destroy(scores);
        return NULL;
    }

    // Initialize with uniform values
    for (uint32_t i = 0; i < graph->vertex_count; i++) {
        scores->scores[i] = 1.0 / sqrt(graph->vertex_count);
    }

    // Power method iteration
    for (uint32_t iter = 0; iter < max_iterations; iter++) {
        memcpy(prev_scores, scores->scores, sizeof(double) * graph->vertex_count);
        memset(scores->scores, 0, sizeof(double) * graph->vertex_count);

        // Multiply by adjacency matrix
        for (uint32_t u = 0; u < graph->vertex_count; u++) {
            const NimcpVertex* vertex = &graph->vertices[u];
            for (NimcpEdgeNode* edge = vertex->edges; edge; edge = edge->next) {
                uint32_t v = edge->dest;
                if (v < graph->vertex_count) {
                    scores->scores[v] += prev_scores[u];
                }
            }
        }

        // Normalize
        double norm = 0.0;
        for (uint32_t i = 0; i < graph->vertex_count; i++) {
            norm += scores->scores[i] * scores->scores[i];
        }

        if (norm > 0.0) {
            norm = sqrt(norm);
            for (uint32_t i = 0; i < graph->vertex_count; i++) {
                scores->scores[i] /= norm;
            }
        }

        // Check convergence
        double diff = 0.0;
        for (uint32_t i = 0; i < graph->vertex_count; i++) {
            double delta = scores->scores[i] - prev_scores[i];
            diff += delta * delta;
        }

        if (sqrt(diff) < 1e-6) {
            break;
        }
    }

    nimcp_free(prev_scores);
    return scores;
}

uint32_t nimcp_detect_hubs(const NimcpCentralityScores* scores, double threshold,
                            uint32_t* hubs, uint32_t max_hubs)
{
    if (!scores || !hubs) {
        return 0;
    }

    if (scores->num_scores == 0) {
        return 0;
    }

    // Calculate mean and standard deviation
    double mean, stdev;
    calculate_stats(scores->scores, scores->num_scores, &mean, &stdev);

    // Find nodes above threshold
    double hub_threshold = mean + threshold * stdev;

    uint32_t count = 0;
    for (uint32_t i = 0; i < scores->num_scores && count < max_hubs; i++) {
        if (scores->scores[i] > hub_threshold) {
            hubs[count++] = i;
        }
    }

    return count;
}

void nimcp_centrality_scores_destroy(NimcpCentralityScores* scores)
{
    LOG_DEBUG("Entering nimcp_centrality_scores_destroy");
    if (!scores) {
        return;
    }

    nimcp_free(scores->scores);
    nimcp_free(scores);
}

double nimcp_get_centrality_score(const NimcpCentralityScores* scores, uint32_t vertex_idx)
{
    LOG_DEBUG("Entering nimcp_get_centrality_score");
    if (!scores || !scores->scores || vertex_idx >= scores->num_scores) {
        LOG_ERROR("nimcp_get_centrality_score failed: returning error");
        return -1.0;
    }

    return scores->scores[vertex_idx];
}
