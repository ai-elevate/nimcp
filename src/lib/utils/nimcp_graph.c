/**
 * @file nimcp_graph.c
 * @brief Implementation of network topology graph structure
 *
 * REFACTORED: Added memory tracking, thread safety, and validation
 * WHY: Prevent memory leaks, ensure thread-safe topology updates, validate inputs
 */

#include "utils/nimcp_graph.h"
#include "utils/nimcp_memory.h"
#include "utils/nimcp_validate.h"
#include <stdlib.h>
#include <string.h>
#include <float.h>

/* Internal helper functions */

/**
 * @brief Allocate and initialize a new edge node
 *
 * WHAT: Creates edge node in adjacency list
 * WHY: Use nimcp_malloc for memory tracking
 * PATTERN: Factory Method - centralized edge creation
 */
static NimcpEdgeNode* create_edge_node(uint32_t dest, nimcp_weight_t weight) {
    // Validate edge weight
    // WHY: Ensure weight is valid float (no NaN/Inf, within range)
    if (!nimcp_validate_float_field(&weight, sizeof(nimcp_weight_t))) {
        return NULL;
    }

    NimcpEdgeNode* node = (NimcpEdgeNode*)nimcp_malloc(sizeof(NimcpEdgeNode));
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
 *
 * WHAT: Recursively frees adjacency list
 * WHY: Use nimcp_free for memory tracking
 * PATTERN: Chain of Responsibility - traverse and free linked list
 */
static void free_edge_list(NimcpEdgeNode* head) {
    while (head) {
        NimcpEdgeNode* next = head->next;
        nimcp_free(head);
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

/**
 * WHAT: Creates new graph for network topology
 * WHY: Use nimcp_* memory functions for leak detection
 * PATTERN: Builder - step-by-step construction with error handling
 */
NimcpGraph* nimcp_graph_create(void) {
    NimcpGraph* graph = (NimcpGraph*)nimcp_malloc(sizeof(NimcpGraph));
    if (!graph) return NULL;

    // Initialize graph structure
    graph->vertex_count = 0;
    graph->edge_count = 0;
    graph->component_count = 0;

    // Initialize mutex for thread safety (Monitor Pattern)
    if (nimcp_mutex_init(&graph->lock, NULL) != NIMCP_SUCCESS) {
        nimcp_free(graph);
        return NULL;
    }

    // Allocate vertex array (zero-initialized for clean state)
    graph->vertices = (NimcpVertex*)nimcp_calloc(NIMCP_MAX_VERTICES, sizeof(NimcpVertex));
    if (!graph->vertices) {
        nimcp_mutex_destroy(&graph->lock);
        nimcp_free(graph);
        return NULL;
    }

    // Allocate component tracking array
    graph->components = (uint32_t*)nimcp_malloc(NIMCP_MAX_VERTICES * sizeof(uint32_t));
    if (!graph->components) {
        nimcp_free(graph->vertices);
        nimcp_mutex_destroy(&graph->lock);
        nimcp_free(graph);
        return NULL;
    }

    return graph;
}

/**
 * WHAT: Destroys graph and frees all resources
 * WHY: Proper cleanup with nimcp_free for memory tracking
 * PATTERN: Destructor - systematic resource release
 */
void nimcp_graph_destroy(NimcpGraph* graph) {
    if (!graph) return;

    // Free all edge lists (uses nimcp_free internally)
    for (uint32_t i = 0; i < graph->vertex_count; i++) {
        free_edge_list(graph->vertices[i].edges);
    }

    // Free arrays
    nimcp_free(graph->components);
    nimcp_free(graph->vertices);

    // Destroy mutex
    nimcp_mutex_destroy(&graph->lock);

    nimcp_free(graph);
}

/**
 * WHAT: Adds peer as vertex to topology graph
 * WHY: Track new peers joining the network
 * PATTERN: Monitor - thread-safe with guard clauses
 */
uint32_t nimcp_graph_add_vertex(
    NimcpGraph* graph,
    uint64_t peer_id,
    float x,
    float y,
    float z,
    uint32_t capabilities) {

    // Guard clause: validate parameters before locking
    if (!graph) return NIMCP_INVALID_VERTEX;

    nimcp_mutex_lock(&graph->lock);

    // Guard clause: check capacity
    if (graph->vertex_count >= NIMCP_MAX_VERTICES) {
        nimcp_mutex_unlock(&graph->lock);
        return NIMCP_INVALID_VERTEX;
    }

    // Check for duplicate peer_id
    for (uint32_t i = 0; i < graph->vertex_count; i++) {
        if (graph->vertices[i].peer_id == peer_id) {
            nimcp_mutex_unlock(&graph->lock);
            return NIMCP_INVALID_VERTEX;
        }
    }

    uint32_t idx = graph->vertex_count++;
    NimcpVertex* vertex = &graph->vertices[idx];

    // Initialize vertex with peer properties
    vertex->peer_id = peer_id;
    vertex->x = x;
    vertex->y = y;
    vertex->z = z;
    vertex->capabilities = capabilities;
    vertex->state = 0;
    vertex->last_updated = 0;
    vertex->edges = NULL;
    vertex->edge_count = 0;

    nimcp_mutex_unlock(&graph->lock);
    return idx;
}

/**
 * WHAT: Removes vertex and all connected edges
 * WHY: Maintains graph consistency when peer disconnects
 * PATTERN: Cascade Delete - removes vertex and dependent edges
 */
bool nimcp_graph_remove_vertex(NimcpGraph* graph, uint32_t vertex_idx) {
    if (!graph) return false;

    nimcp_mutex_lock(&graph->lock);

    // Guard clause: validate vertex index
    if (vertex_idx >= graph->vertex_count) {
        nimcp_mutex_unlock(&graph->lock);
        return false;
    }

    // Free vertex's outgoing edge list
    free_edge_list(graph->vertices[vertex_idx].edges);

    // Remove incoming edges from other vertices
    // WHY: Maintain graph integrity - no dangling edge references
    for (uint32_t i = 0; i < graph->vertex_count; i++) {
        if (i == vertex_idx) continue;

        NimcpEdgeNode** edge = &graph->vertices[i].edges;
        while (*edge) {
            if ((*edge)->dest == vertex_idx) {
                NimcpEdgeNode* temp = *edge;
                *edge = (*edge)->next;
                nimcp_free(temp);
                graph->vertices[i].edge_count--;
                graph->edge_count--;
            } else {
                edge = &(*edge)->next;
            }
        }
    }

    // Shift remaining vertices to compact array
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
    nimcp_mutex_unlock(&graph->lock);
    return true;
}

/**
 * WHAT: Adds or updates edge between peers
 * WHY: Track connection establishment with quality metric
 * PATTERN: Monitor with early validation
 */
bool nimcp_graph_add_edge(
    NimcpGraph* graph,
    uint32_t from,
    uint32_t to,
    nimcp_weight_t weight) {

    if (!graph) return false;

    nimcp_mutex_lock(&graph->lock);

    // Guard clauses: validate parameters
    if (from >= graph->vertex_count || to >= graph->vertex_count ||
        from == to || graph->edge_count >= NIMCP_MAX_EDGES) {
        nimcp_mutex_unlock(&graph->lock);
        return false;
    }

    // Check if edge already exists
    NimcpEdgeNode* edge = graph->vertices[from].edges;
    while (edge) {
        if (edge->dest == to) {
            edge->weight = weight;  // Update weight if edge exists
            nimcp_mutex_unlock(&graph->lock);
            return true;
        }
        edge = edge->next;
    }

    // Create new edge node
    NimcpEdgeNode* new_edge = create_edge_node(to, weight);
    if (!new_edge) {
        nimcp_mutex_unlock(&graph->lock);
        return false;
    }

    // Prepend to adjacency list (O(1) insertion)
    new_edge->next = graph->vertices[from].edges;
    graph->vertices[from].edges = new_edge;
    graph->vertices[from].edge_count++;
    graph->edge_count++;

    nimcp_mutex_unlock(&graph->lock);
    return true;
}

/**
 * WHAT: Removes edge between two vertices
 * WHY: Update topology when peer connection is lost
 * PATTERN: Linked List Remove - find and unlink node
 */
bool nimcp_graph_remove_edge(
    NimcpGraph* graph,
    uint32_t from,
    uint32_t to) {

    if (!graph) return false;

    nimcp_mutex_lock(&graph->lock);

    // Guard clauses: validate indices
    if (from >= graph->vertex_count || to >= graph->vertex_count) {
        nimcp_mutex_unlock(&graph->lock);
        return false;
    }

    // Traverse adjacency list to find and remove edge
    NimcpEdgeNode** edge = &graph->vertices[from].edges;
    while (*edge) {
        if ((*edge)->dest == to) {
            NimcpEdgeNode* temp = *edge;
            *edge = (*edge)->next;
            nimcp_free(temp);
            graph->vertices[from].edge_count--;
            graph->edge_count--;
            nimcp_mutex_unlock(&graph->lock);
            return true;
        }
        edge = &(*edge)->next;
    }

    nimcp_mutex_unlock(&graph->lock);
    return false;
}

/**
 * WHAT: Finds shortest path between two peers using Dijkstra's algorithm
 * WHY: Enable routing decisions and topology analysis
 * PATTERN: Algorithm Strategy - pluggable pathfinding
 * NOTE: Read-only but needs lock for consistent snapshot
 */
NimcpPath* nimcp_graph_shortest_path(
    const NimcpGraph* graph,
    uint32_t from,
    uint32_t to) {

    if (!graph) return NULL;

    // Cast away const to lock (graph isn't modified, just protected)
    NimcpGraph* g = (NimcpGraph*)graph;
    nimcp_mutex_lock(&g->lock);

    // Guard clauses: validate indices
    if (from >= graph->vertex_count || to >= graph->vertex_count) {
        nimcp_mutex_unlock(&g->lock);
        return NULL;
    }

    // Allocate Dijkstra's algorithm working arrays
    nimcp_weight_t* distances = (nimcp_weight_t*)nimcp_malloc(
        graph->vertex_count * sizeof(nimcp_weight_t));
    uint32_t* previous = (uint32_t*)nimcp_malloc(
        graph->vertex_count * sizeof(uint32_t));
    bool* visited = (bool*)nimcp_calloc(graph->vertex_count, sizeof(bool));

    if (!distances || !previous || !visited) {
        nimcp_free(distances);
        nimcp_free(previous);
        nimcp_free(visited);
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
        path = (NimcpPath*)nimcp_malloc(sizeof(NimcpPath));
        if (path) {
            // Count path length by traversing previous array
            uint32_t length = 0;
            uint32_t curr = to;
            while (curr != NIMCP_INVALID_VERTEX) {
                length++;
                curr = previous[curr];
            }

            path->vertices = (uint32_t*)nimcp_malloc(length * sizeof(uint32_t));
            if (path->vertices) {
                path->length = length;
                path->total_weight = distances[to];

                // Fill path array in reverse order (from dest to source)
                curr = to;
                for (int i = length - 1; i >= 0; i--) {
                    path->vertices[i] = curr;
                    curr = previous[curr];
                }
            } else {
                nimcp_free(path);
                path = NULL;
            }
        }
    }

    // Free working arrays
    nimcp_free(distances);
    nimcp_free(previous);
    nimcp_free(visited);

    nimcp_mutex_unlock(&g->lock);
    return path;
}

/**
 * WHAT: Updates peer's logical network coordinates
 * WHY: Track topology changes for optimization
 * PATTERN: Monitor with simple update
 */
bool nimcp_graph_update_coordinates(
    NimcpGraph* graph,
    uint32_t vertex_idx,
    float x,
    float y,
    float z) {

    if (!graph) return false;

    // Validate coordinates before locking
    // WHY: Ensure coordinates are valid floats (no NaN/Inf, within range)
    if (!nimcp_validate_float_field(&x, sizeof(float)) ||
        !nimcp_validate_float_field(&y, sizeof(float)) ||
        !nimcp_validate_float_field(&z, sizeof(float))) {
        return false;
    }

    nimcp_mutex_lock(&graph->lock);

    // Guard clause: validate vertex index
    if (vertex_idx >= graph->vertex_count) {
        nimcp_mutex_unlock(&graph->lock);
        return false;
    }

    graph->vertices[vertex_idx].x = x;
    graph->vertices[vertex_idx].y = y;
    graph->vertices[vertex_idx].z = z;
    graph->vertices[vertex_idx].last_updated = 0; // TODO: Add timestamp

    nimcp_mutex_unlock(&graph->lock);
    return true;
}

/**
 * WHAT: Finds vertex index by peer ID
 * WHY: Translate peer ID to graph index for operations
 * PATTERN: Linear search (could optimize with hash map)
 */
uint32_t nimcp_graph_find_vertex(
    const NimcpGraph* graph,
    uint64_t peer_id) {

    if (!graph) return NIMCP_INVALID_VERTEX;

    // Cast away const to lock (read-only operation)
    NimcpGraph* g = (NimcpGraph*)graph;
    nimcp_mutex_lock(&g->lock);

    for (uint32_t i = 0; i < graph->vertex_count; i++) {
        if (graph->vertices[i].peer_id == peer_id) {
            nimcp_mutex_unlock(&g->lock);
            return i;
        }
    }

    nimcp_mutex_unlock(&g->lock);
    return NIMCP_INVALID_VERTEX;
}

/**
 * WHAT: Identifies connected components in network topology
 * WHY: Detect network partitions and isolated subgraphs
 * PATTERN: Depth-First Search - component discovery algorithm
 */
uint32_t nimcp_graph_update_components(NimcpGraph* graph) {
    if (!graph) return 0;

    nimcp_mutex_lock(&graph->lock);

    bool* visited = (bool*)nimcp_calloc(graph->vertex_count, sizeof(bool));
    if (!visited) {
        nimcp_mutex_unlock(&graph->lock);
        return 0;
    }

    // Label each connected component with unique ID
    uint32_t component = 0;
    for (uint32_t v = 0; v < graph->vertex_count; v++) {
        if (!visited[v]) {
            dfs_component(graph, v, component, visited);
            component++;
        }
    }

    nimcp_free(visited);
    graph->component_count = component;

    nimcp_mutex_unlock(&graph->lock);
    return component;
}

/**
 * WHAT: Retrieves all neighbors of a vertex
 * WHY: Get list of directly connected peers
 * PATTERN: Iterator - traverse adjacency list
 */
uint32_t nimcp_graph_get_neighbors(
    const NimcpGraph* graph,
    uint32_t vertex_idx,
    uint32_t* neighbors,
    uint32_t max_neighbors) {

    if (!graph || !neighbors) return 0;

    // Cast away const to lock (read-only operation)
    NimcpGraph* g = (NimcpGraph*)graph;
    nimcp_mutex_lock(&g->lock);

    // Guard clause: validate vertex index
    if (vertex_idx >= graph->vertex_count) {
        nimcp_mutex_unlock(&g->lock);
        return 0;
    }

    uint32_t count = 0;
    NimcpEdgeNode* edge = graph->vertices[vertex_idx].edges;

    // Traverse adjacency list up to max_neighbors limit
    while (edge && count < max_neighbors) {
        neighbors[count++] = edge->dest;
        edge = edge->next;
    }

    nimcp_mutex_unlock(&g->lock);
    return count;
}

/**
 * WHAT: Retrieves edge weight between two vertices
 * WHY: Get connection quality metric for routing decisions
 * PATTERN: Search - traverse adjacency list for specific edge
 */
bool nimcp_graph_get_edge_weight(
    const NimcpGraph* graph,
    uint32_t from,
    uint32_t to,
    nimcp_weight_t* weight) {

    if (!graph || !weight) return false;

    // Cast away const to lock (read-only operation)
    NimcpGraph* g = (NimcpGraph*)graph;
    nimcp_mutex_lock(&g->lock);

    // Guard clauses: validate indices
    if (from >= graph->vertex_count || to >= graph->vertex_count) {
        nimcp_mutex_unlock(&g->lock);
        return false;
    }

    NimcpEdgeNode* edge = graph->vertices[from].edges;
    while (edge) {
        if (edge->dest == to) {
            *weight = edge->weight;
            nimcp_mutex_unlock(&g->lock);
            return true;
        }
        edge = edge->next;
    }

    nimcp_mutex_unlock(&g->lock);
    return false;
}
