/**
 * @file nimcp_graph.c
 * @brief Implementation of network topology graph structure
 */

#include "utils/nimcp_graph.h"
#include <stdlib.h>
#include <string.h>
#include <float.h>

/* Internal helper functions */

/**
 * @brief Allocate and initialize a new edge node
 */
static NimcpEdgeNode* create_edge_node(uint32_t dest, nimcp_weight_t weight) {
    NimcpEdgeNode* node = (NimcpEdgeNode*)malloc(sizeof(NimcpEdgeNode));
    if (node) {
        node->dest = dest;
        node->weight = weight;
        node->flags = 0;
        node->last_updated = 0;
        node->next = NULL;
    }
    return node;
}

/**
 * @brief Free an entire edge list
 */
static void free_edge_list(NimcpEdgeNode* head) {
    while (head) {
        NimcpEdgeNode* next = head->next;
        free(head);
        head = next;
    }
}

/**
 * @brief Depth-first search for component labeling
 */
static void dfs_component(
    const NimcpGraph* graph,
    uint32_t vertex,
    uint32_t component,
    bool* visited) {
    
    visited[vertex] = true;
    graph->components[vertex] = component;

    NimcpEdgeNode* edge = graph->vertices[vertex].edges;
    while (edge) {
        if (!visited[edge->dest]) {
            dfs_component(graph, edge->dest, component, visited);
        }
        edge = edge->next;
    }
}

/**
 * @brief Find minimum distance vertex for Dijkstra's algorithm
 */
static uint32_t find_min_distance(
    nimcp_weight_t* distances,
    bool* visited,
    uint32_t vertex_count) {
    
    nimcp_weight_t min = FLT_MAX;
    uint32_t min_vertex = NIMCP_INVALID_VERTEX;

    for (uint32_t v = 0; v < vertex_count; v++) {
        if (!visited[v] && distances[v] < min) {
            min = distances[v];
            min_vertex = v;
        }
    }

    return min_vertex;
}

/* Public function implementations */

NimcpGraph* nimcp_graph_create(void) {
    NimcpGraph* graph = (NimcpGraph*)malloc(sizeof(NimcpGraph));
    if (!graph) return NULL;

    // Initialize graph structure
    graph->vertex_count = 0;
    graph->edge_count = 0;
    
    // Allocate vertex array
    graph->vertices = (NimcpVertex*)calloc(NIMCP_MAX_VERTICES, sizeof(NimcpVertex));
    if (!graph->vertices) {
        free(graph);
        return NULL;
    }

    // Allocate component tracking array
    graph->components = (uint32_t*)malloc(NIMCP_MAX_VERTICES * sizeof(uint32_t));
    if (!graph->components) {
        free(graph->vertices);
        free(graph);
        return NULL;
    }

    graph->component_count = 0;
    return graph;
}

void nimcp_graph_destroy(NimcpGraph* graph) {
    if (!graph) return;

    // Free all edge lists
    for (uint32_t i = 0; i < graph->vertex_count; i++) {
        free_edge_list(graph->vertices[i].edges);
    }

    // Free arrays
    free(graph->components);
    free(graph->vertices);
    free(graph);
}

uint32_t nimcp_graph_add_vertex(
    NimcpGraph* graph,
    uint64_t peer_id,
    float x,
    float y,
    float z,
    uint32_t capabilities) {
    
    if (!graph || graph->vertex_count >= NIMCP_MAX_VERTICES) {
        return NIMCP_INVALID_VERTEX;
    }

    // Check for existing peer_id
    for (uint32_t i = 0; i < graph->vertex_count; i++) {
        if (graph->vertices[i].peer_id == peer_id) {
            return NIMCP_INVALID_VERTEX;
        }
    }

    uint32_t idx = graph->vertex_count++;
    NimcpVertex* vertex = &graph->vertices[idx];

    // Initialize vertex
    vertex->peer_id = peer_id;
    vertex->x = x;
    vertex->y = y;
    vertex->z = z;
    vertex->capabilities = capabilities;
    vertex->state = 0;
    vertex->last_updated = 0;
    vertex->edges = NULL;
    vertex->edge_count = 0;

    return idx;
}

bool nimcp_graph_remove_vertex(NimcpGraph* graph, uint32_t vertex_idx) {
    if (!graph || vertex_idx >= graph->vertex_count) {
        return false;
    }

    // Free vertex's edge list
    free_edge_list(graph->vertices[vertex_idx].edges);

    // Remove incoming edges from other vertices
    for (uint32_t i = 0; i < graph->vertex_count; i++) {
        if (i == vertex_idx) continue;

        NimcpEdgeNode** edge = &graph->vertices[i].edges;
        while (*edge) {
            if ((*edge)->dest == vertex_idx) {
                NimcpEdgeNode* temp = *edge;
                *edge = (*edge)->next;
                free(temp);
                graph->vertices[i].edge_count--;
                graph->edge_count--;
            } else {
                edge = &(*edge)->next;
            }
        }
    }

    // Shift remaining vertices
    if (vertex_idx < graph->vertex_count - 1) {
        memmove(&graph->vertices[vertex_idx],
                &graph->vertices[vertex_idx + 1],
                (graph->vertex_count - vertex_idx - 1) * sizeof(NimcpVertex));

        // Update destination indices in remaining edges
        for (uint32_t i = 0; i < graph->vertex_count - 1; i++) {
            NimcpEdgeNode* edge = graph->vertices[i].edges;
            while (edge) {
                if (edge->dest > vertex_idx) {
                    edge->dest--;
                }
                edge = edge->next;
            }
        }
    }

    graph->vertex_count--;
    return true;
}

bool nimcp_graph_add_edge(
    NimcpGraph* graph,
    uint32_t from,
    uint32_t to,
    nimcp_weight_t weight) {
    
    if (!graph || from >= graph->vertex_count || to >= graph->vertex_count ||
        from == to || graph->edge_count >= NIMCP_MAX_EDGES) {
        return false;
    }

    // Check if edge already exists
    NimcpEdgeNode* edge = graph->vertices[from].edges;
    while (edge) {
        if (edge->dest == to) {
            edge->weight = weight;  // Update weight if edge exists
            return true;
        }
        edge = edge->next;
    }

    // Create new edge node
    NimcpEdgeNode* new_edge = create_edge_node(to, weight);
    if (!new_edge) return false;

    // Add to adjacency list
    new_edge->next = graph->vertices[from].edges;
    graph->vertices[from].edges = new_edge;
    graph->vertices[from].edge_count++;
    graph->edge_count++;

    return true;
}

bool nimcp_graph_remove_edge(
    NimcpGraph* graph,
    uint32_t from,
    uint32_t to) {
    
    if (!graph || from >= graph->vertex_count || to >= graph->vertex_count) {
        return false;
    }

    NimcpEdgeNode** edge = &graph->vertices[from].edges;
    while (*edge) {
        if ((*edge)->dest == to) {
            NimcpEdgeNode* temp = *edge;
            *edge = (*edge)->next;
            free(temp);
            graph->vertices[from].edge_count--;
            graph->edge_count--;
            return true;
        }
        edge = &(*edge)->next;
    }

    return false;
}

NimcpPath* nimcp_graph_shortest_path(
    const NimcpGraph* graph,
    uint32_t from,
    uint32_t to) {
    
    if (!graph || from >= graph->vertex_count || to >= graph->vertex_count) {
        return NULL;
    }

    // Initialize Dijkstra's algorithm
    nimcp_weight_t* distances = (nimcp_weight_t*)malloc(
        graph->vertex_count * sizeof(nimcp_weight_t));
    uint32_t* previous = (uint32_t*)malloc(
        graph->vertex_count * sizeof(uint32_t));
    bool* visited = (bool*)calloc(graph->vertex_count, sizeof(bool));

    if (!distances || !previous || !visited) {
        free(distances);
        free(previous);
        free(visited);
        return NULL;
    }

    // Initialize distances
    for (uint32_t i = 0; i < graph->vertex_count; i++) {
        distances[i] = FLT_MAX;
        previous[i] = NIMCP_INVALID_VERTEX;
    }
    distances[from] = 0;

    // Find shortest path
    for (uint32_t count = 0; count < graph->vertex_count - 1; count++) {
        uint32_t u = find_min_distance(distances, visited, graph->vertex_count);
        if (u == NIMCP_INVALID_VERTEX) break;

        visited[u] = true;

        NimcpEdgeNode* edge = graph->vertices[u].edges;
        while (edge) {
            uint32_t v = edge->dest;
            if (!visited[v] && 
                distances[u] != FLT_MAX &&
                distances[u] + edge->weight < distances[v]) {
                distances[v] = distances[u] + edge->weight;
                previous[v] = u;
            }
            edge = edge->next;
        }
    }

    // Build path if destination was reached
    NimcpPath* path = NULL;
    if (distances[to] != FLT_MAX) {
        path = (NimcpPath*)malloc(sizeof(NimcpPath));
        if (path) {
            // Count path length
            uint32_t length = 0;
            uint32_t curr = to;
            while (curr != NIMCP_INVALID_VERTEX) {
                length++;
                curr = previous[curr];
            }

            path->vertices = (uint32_t*)malloc(length * sizeof(uint32_t));
            if (path->vertices) {
                path->length = length;
                path->total_weight = distances[to];

                // Fill path array in reverse
                curr = to;
                for (int i = length - 1; i >= 0; i--) {
                    path->vertices[i] = curr;
                    curr = previous[curr];
                }
            } else {
                free(path);
                path = NULL;
            }
        }
    }

    free(distances);
    free(previous);
    free(visited);
    return path;
}

bool nimcp_graph_update_coordinates(
    NimcpGraph* graph,
    uint32_t vertex_idx,
    float x,
    float y,
    float z) {
    
    if (!graph || vertex_idx >= graph->vertex_count) {
        return false;
    }

    graph->vertices[vertex_idx].x = x;
    graph->vertices[vertex_idx].y = y;
    graph->vertices[vertex_idx].z = z;
    graph->vertices[vertex_idx].last_updated = 0; // TODO: Add timestamp

    return true;
}

uint32_t nimcp_graph_find_vertex(
    const NimcpGraph* graph,
    uint64_t peer_id) {
    
    if (!graph) return NIMCP_INVALID_VERTEX;

    for (uint32_t i = 0; i < graph->vertex_count; i++) {
        if (graph->vertices[i].peer_id == peer_id) {
            return i;
        }
    }

    return NIMCP_INVALID_VERTEX;
}

uint32_t nimcp_graph_update_components(NimcpGraph* graph) {
    if (!graph) return 0;

    bool* visited = (bool*)calloc(graph->vertex_count, sizeof(bool));
    if (!visited) return 0;

    uint32_t component = 0;
    for (uint32_t v = 0; v < graph->vertex_count; v++) {
        if (!visited[v]) {
            dfs_component(graph, v, component, visited);
            component++;
        }
    }

    free(visited);
    graph->component_count = component;
    return component;
}

uint32_t nimcp_graph_get_neighbors(
    const NimcpGraph* graph,
    uint32_t vertex_idx,
    uint32_t* neighbors,
    uint32_t max_neighbors) {
    
    if (!graph || !neighbors || vertex_idx >= graph->vertex_count) {
        return 0;
    }

    uint32_t count = 0;
    NimcpEdgeNode* edge = graph->vertices[vertex_idx].edges;
    
    while (edge && count < max_neighbors) {
        neighbors[count++] = edge->dest;
        edge = edge->next;
    }

    return count;
}

bool nimcp_graph_get_edge_weight(
    const NimcpGraph* graph,
    uint32_t from,
    uint32_t to,
    nimcp_weight_t* weight) {
    
    if (!graph || !weight || from >= graph->vertex_count || to >= graph->vertex_count) {
        return false;
    }

    NimcpEdgeNode* edge = graph->vertices[from].edges;
    while (edge) {
        if (edge->dest == to) {
            *weight = edge->weight;
            return true;
        }
        edge = edge->next;
    }

    return false;
}
