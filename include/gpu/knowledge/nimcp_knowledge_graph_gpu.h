/**
 * @file nimcp_knowledge_graph_gpu.h
 * @brief GPU-accelerated Knowledge Graph Operations
 *
 * WHAT: CUDA kernels for knowledge graph operations
 * WHY:  GPU acceleration for large-scale graph traversal and similarity
 * HOW:  Custom kernels for BFS/DFS, embeddings, and hyperbolic operations
 *
 * ARCHITECTURE:
 * - Parallel graph traversal (BFS/DFS across frontiers)
 * - Semantic similarity via cosine similarity on embeddings
 * - Hyperbolic space operations for hierarchical knowledge
 * - Node/edge feature aggregation with shared memory
 * - Subgraph matching with parallel pattern detection
 *
 * PARALLELIZATION STRATEGIES:
 * - BFS: Parallelize across nodes in current frontier
 * - Similarity: Parallelize across node pairs
 * - Aggregation: Warp-level reductions with shared memory
 * - Subgraph matching: Parallelize across candidate mappings
 *
 * @version 1.0
 * @author NIMCP Development Team
 * @date 2025
 */

#ifndef NIMCP_KNOWLEDGE_GRAPH_GPU_H
#define NIMCP_KNOWLEDGE_GRAPH_GPU_H

#include "gpu/context/nimcp_gpu_context.h"

#ifdef __cplusplus
extern "C" {
#endif

#include "common/nimcp_export.h"
#include "gpu/tensor/nimcp_tensor_gpu.h"
#include <stdbool.h>
#include <stdint.h>

//=============================================================================
// Graph Representation (CSR Format)
//=============================================================================

/**
 * @brief GPU knowledge graph in Compressed Sparse Row (CSR) format
 *
 * CSR is memory-efficient and enables coalesced memory access for graph traversal.
 * row_offsets[i] to row_offsets[i+1] gives edge indices for node i.
 */
typedef struct {
    nimcp_gpu_tensor_t* row_offsets;      /**< CSR row offsets [num_nodes + 1] */
    nimcp_gpu_tensor_t* col_indices;      /**< CSR column indices [num_edges] */
    nimcp_gpu_tensor_t* edge_weights;     /**< Edge weights [num_edges] (optional) */
    nimcp_gpu_tensor_t* node_embeddings;  /**< Node embeddings [num_nodes x embed_dim] */
    nimcp_gpu_tensor_t* edge_embeddings;  /**< Edge embeddings [num_edges x embed_dim] (optional) */
    uint32_t num_nodes;                   /**< Number of nodes */
    uint32_t num_edges;                   /**< Number of edges */
    uint32_t embed_dim;                   /**< Embedding dimension */
    bool is_hyperbolic;                   /**< Whether embeddings are in hyperbolic space */
    nimcp_gpu_context_t* ctx;             /**< GPU context */
} nimcp_gpu_knowledge_graph_t;

/**
 * @brief BFS/DFS traversal result
 */
typedef struct {
    nimcp_gpu_tensor_t* distances;        /**< Distance from source [num_nodes] */
    nimcp_gpu_tensor_t* parents;          /**< Parent node for path reconstruction [num_nodes] */
    nimcp_gpu_tensor_t* visited;          /**< Visited flags [num_nodes] */
    uint32_t num_visited;                 /**< Number of nodes visited */
    uint32_t max_depth;                   /**< Maximum depth reached */
} nimcp_graph_traversal_result_t;

/**
 * @brief Similarity search result
 */
typedef struct {
    nimcp_gpu_tensor_t* indices;          /**< Indices of similar nodes [k] */
    nimcp_gpu_tensor_t* scores;           /**< Similarity scores [k] */
    uint32_t k;                           /**< Number of results */
} nimcp_similarity_result_t;

/**
 * @brief Subgraph matching result
 */
typedef struct {
    nimcp_gpu_tensor_t* mappings;         /**< Node mappings [num_matches x pattern_size] */
    nimcp_gpu_tensor_t* scores;           /**< Match confidence scores [num_matches] */
    uint32_t num_matches;                 /**< Number of matches found */
} nimcp_subgraph_match_result_t;

/**
 * @brief Feature aggregation mode
 */
typedef enum {
    NIMCP_AGGREGATE_SUM = 0,              /**< Sum neighbor features */
    NIMCP_AGGREGATE_MEAN = 1,             /**< Mean of neighbor features */
    NIMCP_AGGREGATE_MAX = 2,              /**< Max pooling over neighbors */
    NIMCP_AGGREGATE_ATTENTION = 3         /**< Attention-weighted aggregation */
} nimcp_aggregate_mode_t;

//=============================================================================
// Graph Creation and Destruction
//=============================================================================

/**
 * @brief Create GPU knowledge graph from CSR arrays
 *
 * @param ctx GPU context
 * @param row_offsets Host array of row offsets [num_nodes + 1]
 * @param col_indices Host array of column indices [num_edges]
 * @param edge_weights Host array of edge weights [num_edges] (can be NULL)
 * @param num_nodes Number of nodes
 * @param num_edges Number of edges
 * @param embed_dim Embedding dimension
 * @return GPU knowledge graph or NULL on failure
 */
NIMCP_EXPORT nimcp_gpu_knowledge_graph_t* nimcp_gpu_knowledge_graph_create(
    nimcp_gpu_context_t* ctx,
    const uint32_t* row_offsets,
    const uint32_t* col_indices,
    const float* edge_weights,
    uint32_t num_nodes,
    uint32_t num_edges,
    uint32_t embed_dim
);

/**
 * @brief Destroy GPU knowledge graph
 *
 * @param graph Graph to destroy
 */
NIMCP_EXPORT void nimcp_gpu_knowledge_graph_destroy(nimcp_gpu_knowledge_graph_t* graph);

/**
 * @brief Set node embeddings from host data
 *
 * @param graph GPU knowledge graph
 * @param embeddings Host embeddings [num_nodes x embed_dim]
 * @return true on success
 */
NIMCP_EXPORT bool nimcp_gpu_knowledge_graph_set_embeddings(
    nimcp_gpu_knowledge_graph_t* graph,
    const float* embeddings
);

/**
 * @brief Set hyperbolic embeddings (Poincare ball model)
 *
 * @param graph GPU knowledge graph
 * @param embeddings Host embeddings in Poincare ball [num_nodes x embed_dim]
 * @return true on success
 */
NIMCP_EXPORT bool nimcp_gpu_knowledge_graph_set_hyperbolic_embeddings(
    nimcp_gpu_knowledge_graph_t* graph,
    const float* embeddings
);

//=============================================================================
// Graph Traversal (Parallel BFS/DFS)
//=============================================================================

/**
 * @brief Parallel Breadth-First Search
 *
 * Uses frontier-based parallelization where all nodes at current depth
 * are processed in parallel.
 *
 * COMPLEXITY: O(V + E) work, O(D) depth where D is diameter
 * PARALLELISM: O(frontier_size) per level
 *
 * @param graph GPU knowledge graph
 * @param source_node Starting node
 * @param max_depth Maximum traversal depth (-1 for unlimited)
 * @param result Output traversal result
 * @return true on success
 */
NIMCP_EXPORT bool nimcp_gpu_bfs(
    nimcp_gpu_knowledge_graph_t* graph,
    uint32_t source_node,
    int32_t max_depth,
    nimcp_graph_traversal_result_t* result
);

/**
 * @brief Parallel multi-source BFS
 *
 * BFS from multiple source nodes simultaneously.
 *
 * @param graph GPU knowledge graph
 * @param source_nodes Array of source node indices
 * @param num_sources Number of source nodes
 * @param max_depth Maximum traversal depth
 * @param result Output traversal result (distances are min across sources)
 * @return true on success
 */
NIMCP_EXPORT bool nimcp_gpu_bfs_multi_source(
    nimcp_gpu_knowledge_graph_t* graph,
    const uint32_t* source_nodes,
    uint32_t num_sources,
    int32_t max_depth,
    nimcp_graph_traversal_result_t* result
);

/**
 * @brief Parallel Depth-First Search (iterative)
 *
 * Uses work-stealing approach for parallel DFS exploration.
 *
 * @param graph GPU knowledge graph
 * @param source_node Starting node
 * @param max_depth Maximum traversal depth
 * @param result Output traversal result
 * @return true on success
 */
NIMCP_EXPORT bool nimcp_gpu_dfs(
    nimcp_gpu_knowledge_graph_t* graph,
    uint32_t source_node,
    int32_t max_depth,
    nimcp_graph_traversal_result_t* result
);

/**
 * @brief Find shortest path between two nodes
 *
 * Uses bidirectional BFS for efficiency.
 *
 * @param graph GPU knowledge graph
 * @param source Starting node
 * @param target Target node
 * @param path_out Output path indices (caller allocated, max size num_nodes)
 * @param path_length Output path length
 * @return true if path found
 */
NIMCP_EXPORT bool nimcp_gpu_shortest_path(
    nimcp_gpu_knowledge_graph_t* graph,
    uint32_t source,
    uint32_t target,
    uint32_t* path_out,
    uint32_t* path_length
);

/**
 * @brief Destroy traversal result
 */
NIMCP_EXPORT void nimcp_graph_traversal_result_destroy(nimcp_graph_traversal_result_t* result);

//=============================================================================
// Semantic Similarity (Cosine Similarity)
//=============================================================================

/**
 * @brief Compute cosine similarity between two nodes
 *
 * @param graph GPU knowledge graph
 * @param node_a First node index
 * @param node_b Second node index
 * @param similarity Output similarity score
 * @return true on success
 */
NIMCP_EXPORT bool nimcp_gpu_node_similarity(
    nimcp_gpu_knowledge_graph_t* graph,
    uint32_t node_a,
    uint32_t node_b,
    float* similarity
);

/**
 * @brief Compute pairwise similarity matrix
 *
 * Computes cosine similarity for all pairs of specified nodes.
 *
 * @param graph GPU knowledge graph
 * @param node_indices Indices of nodes to compare
 * @param num_nodes Number of nodes
 * @param similarity_matrix Output similarity matrix [num_nodes x num_nodes]
 * @return true on success
 */
NIMCP_EXPORT bool nimcp_gpu_pairwise_similarity(
    nimcp_gpu_knowledge_graph_t* graph,
    const uint32_t* node_indices,
    uint32_t num_nodes,
    nimcp_gpu_tensor_t* similarity_matrix
);

/**
 * @brief Find k most similar nodes to query
 *
 * Uses optimized top-k selection with parallel reduction.
 *
 * @param graph GPU knowledge graph
 * @param query_node Query node index
 * @param k Number of similar nodes to find
 * @param result Output similarity result
 * @return true on success
 */
NIMCP_EXPORT bool nimcp_gpu_knn_similarity(
    nimcp_gpu_knowledge_graph_t* graph,
    uint32_t query_node,
    uint32_t k,
    nimcp_similarity_result_t* result
);

/**
 * @brief Find k most similar nodes to query embedding
 *
 * @param graph GPU knowledge graph
 * @param query_embedding Query embedding [embed_dim]
 * @param k Number of similar nodes to find
 * @param result Output similarity result
 * @return true on success
 */
NIMCP_EXPORT bool nimcp_gpu_knn_similarity_embedding(
    nimcp_gpu_knowledge_graph_t* graph,
    const float* query_embedding,
    uint32_t k,
    nimcp_similarity_result_t* result
);

/**
 * @brief Destroy similarity result
 */
NIMCP_EXPORT void nimcp_similarity_result_destroy(nimcp_similarity_result_t* result);

//=============================================================================
// Knowledge Embedding Operations
//=============================================================================

/**
 * @brief Update embeddings via gradient descent
 *
 * @param graph GPU knowledge graph
 * @param gradients Gradients [num_nodes x embed_dim]
 * @param learning_rate Learning rate
 * @return true on success
 */
NIMCP_EXPORT bool nimcp_gpu_embedding_update(
    nimcp_gpu_knowledge_graph_t* graph,
    const nimcp_gpu_tensor_t* gradients,
    float learning_rate
);

/**
 * @brief Compute embedding loss (triplet loss for knowledge graph embeddings)
 *
 * loss = max(0, margin + d(anchor, positive) - d(anchor, negative))
 *
 * @param graph GPU knowledge graph
 * @param anchors Anchor node indices [batch_size]
 * @param positives Positive node indices [batch_size]
 * @param negatives Negative node indices [batch_size]
 * @param batch_size Number of triplets
 * @param margin Margin for triplet loss
 * @param loss_out Output total loss
 * @return true on success
 */
NIMCP_EXPORT bool nimcp_gpu_triplet_loss(
    nimcp_gpu_knowledge_graph_t* graph,
    const uint32_t* anchors,
    const uint32_t* positives,
    const uint32_t* negatives,
    uint32_t batch_size,
    float margin,
    float* loss_out
);

/**
 * @brief Normalize embeddings (L2 normalization)
 *
 * @param graph GPU knowledge graph
 * @return true on success
 */
NIMCP_EXPORT bool nimcp_gpu_normalize_embeddings(
    nimcp_gpu_knowledge_graph_t* graph
);

//=============================================================================
// Node/Edge Feature Aggregation
//=============================================================================

/**
 * @brief Aggregate neighbor features for each node
 *
 * Implements Graph Neural Network style message passing.
 *
 * @param graph GPU knowledge graph
 * @param mode Aggregation mode (sum, mean, max, attention)
 * @param output Output aggregated features [num_nodes x embed_dim]
 * @return true on success
 */
NIMCP_EXPORT bool nimcp_gpu_aggregate_neighbors(
    nimcp_gpu_knowledge_graph_t* graph,
    nimcp_aggregate_mode_t mode,
    nimcp_gpu_tensor_t* output
);

/**
 * @brief Aggregate with attention weights
 *
 * Uses attention mechanism to weight neighbor contributions.
 *
 * @param graph GPU knowledge graph
 * @param query_weights Query weight matrix [embed_dim x attention_dim]
 * @param key_weights Key weight matrix [embed_dim x attention_dim]
 * @param output Output aggregated features [num_nodes x embed_dim]
 * @return true on success
 */
NIMCP_EXPORT bool nimcp_gpu_aggregate_attention(
    nimcp_gpu_knowledge_graph_t* graph,
    const nimcp_gpu_tensor_t* query_weights,
    const nimcp_gpu_tensor_t* key_weights,
    nimcp_gpu_tensor_t* output
);

/**
 * @brief Multi-hop feature aggregation
 *
 * Aggregates features from k-hop neighborhood.
 *
 * @param graph GPU knowledge graph
 * @param num_hops Number of hops (layers)
 * @param mode Aggregation mode per hop
 * @param output Output features [num_nodes x embed_dim]
 * @return true on success
 */
NIMCP_EXPORT bool nimcp_gpu_multi_hop_aggregate(
    nimcp_gpu_knowledge_graph_t* graph,
    uint32_t num_hops,
    nimcp_aggregate_mode_t mode,
    nimcp_gpu_tensor_t* output
);

//=============================================================================
// Subgraph Matching
//=============================================================================

/**
 * @brief Find subgraph matches (isomorphism)
 *
 * Uses parallel candidate filtering and verification.
 *
 * @param graph Target GPU knowledge graph
 * @param pattern Pattern graph to match
 * @param max_matches Maximum number of matches to find
 * @param result Output match result
 * @return true on success
 */
NIMCP_EXPORT bool nimcp_gpu_subgraph_match(
    nimcp_gpu_knowledge_graph_t* graph,
    nimcp_gpu_knowledge_graph_t* pattern,
    uint32_t max_matches,
    nimcp_subgraph_match_result_t* result
);

/**
 * @brief Find approximate subgraph matches
 *
 * Uses embedding-based similarity for approximate matching.
 *
 * @param graph Target GPU knowledge graph
 * @param pattern Pattern graph
 * @param similarity_threshold Minimum similarity score
 * @param max_matches Maximum matches
 * @param result Output match result
 * @return true on success
 */
NIMCP_EXPORT bool nimcp_gpu_subgraph_match_approximate(
    nimcp_gpu_knowledge_graph_t* graph,
    nimcp_gpu_knowledge_graph_t* pattern,
    float similarity_threshold,
    uint32_t max_matches,
    nimcp_subgraph_match_result_t* result
);

/**
 * @brief Destroy subgraph match result
 */
NIMCP_EXPORT void nimcp_subgraph_match_result_destroy(nimcp_subgraph_match_result_t* result);

//=============================================================================
// Hyperbolic Space Operations
//=============================================================================

/**
 * @brief Compute hyperbolic distance (Poincare ball model)
 *
 * d(u,v) = arcosh(1 + 2 * ||u-v||^2 / ((1-||u||^2)(1-||v||^2)))
 *
 * @param graph GPU knowledge graph (must be hyperbolic)
 * @param node_a First node
 * @param node_b Second node
 * @param distance Output hyperbolic distance
 * @return true on success
 */
NIMCP_EXPORT bool nimcp_gpu_hyperbolic_distance(
    nimcp_gpu_knowledge_graph_t* graph,
    uint32_t node_a,
    uint32_t node_b,
    float* distance
);

/**
 * @brief Compute pairwise hyperbolic distances
 *
 * @param graph GPU knowledge graph
 * @param node_indices Node indices
 * @param num_nodes Number of nodes
 * @param distance_matrix Output distance matrix [num_nodes x num_nodes]
 * @return true on success
 */
NIMCP_EXPORT bool nimcp_gpu_hyperbolic_pairwise_distance(
    nimcp_gpu_knowledge_graph_t* graph,
    const uint32_t* node_indices,
    uint32_t num_nodes,
    nimcp_gpu_tensor_t* distance_matrix
);

/**
 * @brief Riemannian SGD step for hyperbolic embeddings
 *
 * Projects Euclidean gradient to tangent space and updates via
 * exponential map.
 *
 * @param graph GPU knowledge graph
 * @param euclidean_gradients Euclidean gradients [num_nodes x embed_dim]
 * @param learning_rate Learning rate
 * @return true on success
 */
NIMCP_EXPORT bool nimcp_gpu_hyperbolic_sgd_step(
    nimcp_gpu_knowledge_graph_t* graph,
    const nimcp_gpu_tensor_t* euclidean_gradients,
    float learning_rate
);

/**
 * @brief Convert Euclidean embeddings to Poincare ball
 *
 * Uses exponential map from origin.
 *
 * @param graph GPU knowledge graph
 * @return true on success
 */
NIMCP_EXPORT bool nimcp_gpu_euclidean_to_hyperbolic(
    nimcp_gpu_knowledge_graph_t* graph
);

/**
 * @brief k-NN search in hyperbolic space
 *
 * @param graph GPU knowledge graph
 * @param query_node Query node
 * @param k Number of neighbors
 * @param result Output similarity result (with hyperbolic distances)
 * @return true on success
 */
NIMCP_EXPORT bool nimcp_gpu_hyperbolic_knn(
    nimcp_gpu_knowledge_graph_t* graph,
    uint32_t query_node,
    uint32_t k,
    nimcp_similarity_result_t* result
);

//=============================================================================
// Utility Functions
//=============================================================================

/**
 * @brief Get graph statistics
 *
 * @param graph GPU knowledge graph
 * @param avg_degree Output average node degree
 * @param max_degree Output maximum node degree
 * @param embedding_norm Output average embedding norm
 * @return true on success
 */
NIMCP_EXPORT bool nimcp_gpu_knowledge_graph_stats(
    nimcp_gpu_knowledge_graph_t* graph,
    float* avg_degree,
    uint32_t* max_degree,
    float* embedding_norm
);

/**
 * @brief Check if graph is valid and properly initialized
 *
 * @param graph GPU knowledge graph
 * @return true if valid
 */
NIMCP_EXPORT bool nimcp_gpu_knowledge_graph_is_valid(
    const nimcp_gpu_knowledge_graph_t* graph
);

#ifdef __cplusplus
}
#endif

#endif // NIMCP_KNOWLEDGE_GRAPH_GPU_H
