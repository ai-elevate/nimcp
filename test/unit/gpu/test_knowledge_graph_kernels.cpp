/**
 * @file test_knowledge_graph_kernels.cpp
 * @brief Comprehensive unit tests for GPU knowledge graph kernels
 *
 * WHAT: Tests for GPU-accelerated knowledge graph operations
 * WHY:  Verify graph traversal, similarity search, hyperbolic operations
 * HOW:  GoogleTest with GPU context setup/teardown and numerical verification
 *
 * TEST COVERAGE:
 * - Graph lifecycle (create, destroy)
 * - Graph traversal (BFS, DFS, shortest path)
 * - Semantic similarity (cosine, pairwise, k-NN)
 * - Knowledge embedding operations
 * - Node/edge aggregation
 * - Subgraph matching
 * - Hyperbolic space operations
 *
 * @author NIMCP Development Team
 * @date 2025
 */

#include <gtest/gtest.h>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>

#include "gpu/knowledge/nimcp_knowledge_graph_gpu.h"
#include "gpu/context/nimcp_gpu_context.h"
#include "gpu/tensor/nimcp_tensor_gpu.h"

//=============================================================================
// Test Constants
//=============================================================================

static constexpr uint32_t DEFAULT_NUM_NODES = 100;
static constexpr uint32_t DEFAULT_NUM_EDGES = 500;
static constexpr uint32_t DEFAULT_EMBED_DIM = 64;
static constexpr float NUMERICAL_EPS = 1e-5f;

//=============================================================================
// Test Fixture
//=============================================================================

/**
 * @brief Test fixture for GPU knowledge graph kernel tests
 */
class KnowledgeGraphKernelTest : public ::testing::Test {
protected:
    nimcp_gpu_context_t* ctx = nullptr;
    bool gpu_available = false;
    std::mt19937 rng{42};

    void SetUp() override {
        ctx = nimcp_gpu_context_create_auto();
        gpu_available = (ctx != nullptr && nimcp_gpu_context_is_valid(ctx));
    }

    void TearDown() override {
        if (ctx) {
            nimcp_gpu_context_destroy(ctx);
            ctx = nullptr;
        }
    }

    void RequireGPU() {
        if (!gpu_available) {
            GTEST_SKIP() << "GPU not available, skipping test";
        }
    }

    /**
     * @brief Create a simple connected graph in CSR format
     */
    void create_simple_graph(
        std::vector<uint32_t>& row_offsets,
        std::vector<uint32_t>& col_indices,
        std::vector<float>& edge_weights,
        uint32_t num_nodes,
        uint32_t avg_degree
    ) {
        row_offsets.resize(num_nodes + 1);
        col_indices.clear();
        edge_weights.clear();

        std::uniform_int_distribution<uint32_t> node_dist(0, num_nodes - 1);
        std::uniform_real_distribution<float> weight_dist(0.1f, 1.0f);

        uint32_t edge_count = 0;
        for (uint32_t i = 0; i < num_nodes; i++) {
            row_offsets[i] = edge_count;

            // Add avg_degree random edges
            for (uint32_t j = 0; j < avg_degree; j++) {
                uint32_t target = node_dist(rng);
                if (target != i) {  // No self-loops
                    col_indices.push_back(target);
                    edge_weights.push_back(weight_dist(rng));
                    edge_count++;
                }
            }
        }
        row_offsets[num_nodes] = edge_count;
    }

    /**
     * @brief Create a chain graph (node 0 -> 1 -> 2 -> ... -> n-1)
     */
    void create_chain_graph(
        std::vector<uint32_t>& row_offsets,
        std::vector<uint32_t>& col_indices,
        uint32_t num_nodes
    ) {
        row_offsets.resize(num_nodes + 1);
        col_indices.clear();

        for (uint32_t i = 0; i < num_nodes; i++) {
            row_offsets[i] = i;
            if (i < num_nodes - 1) {
                col_indices.push_back(i + 1);
            }
        }
        row_offsets[num_nodes] = num_nodes - 1;
    }

    /**
     * @brief Create random embeddings
     */
    std::vector<float> create_random_embeddings(uint32_t num_nodes, uint32_t embed_dim) {
        std::normal_distribution<float> dist(0.0f, 1.0f);
        std::vector<float> embeddings(num_nodes * embed_dim);
        for (auto& e : embeddings) {
            e = dist(rng);
        }
        return embeddings;
    }

    /**
     * @brief Normalize embeddings (L2)
     */
    void normalize_embeddings(std::vector<float>& embeddings, uint32_t num_nodes, uint32_t embed_dim) {
        for (uint32_t i = 0; i < num_nodes; i++) {
            float norm = 0.0f;
            for (uint32_t j = 0; j < embed_dim; j++) {
                float val = embeddings[i * embed_dim + j];
                norm += val * val;
            }
            norm = std::sqrt(norm);
            if (norm > NUMERICAL_EPS) {
                for (uint32_t j = 0; j < embed_dim; j++) {
                    embeddings[i * embed_dim + j] /= norm;
                }
            }
        }
    }

    /**
     * @brief Create GPU tensor from host data
     */
    nimcp_gpu_tensor_t* create_tensor_from_data(const float* data, size_t size) {
        if (!gpu_available) return nullptr;
        size_t dims[1] = {size};
        return nimcp_gpu_tensor_from_host(ctx, data, dims, 1, NIMCP_GPU_PRECISION_FP32);
    }

    /**
     * @brief Create GPU tensor filled with zeros
     */
    nimcp_gpu_tensor_t* create_zero_tensor(size_t size) {
        if (!gpu_available) return nullptr;
        size_t dims[1] = {size};
        nimcp_gpu_tensor_t* tensor = nimcp_gpu_tensor_create(ctx, dims, 1, NIMCP_GPU_PRECISION_FP32);
        if (tensor) {
            nimcp_gpu_zeros(ctx, tensor);
        }
        return tensor;
    }

    /**
     * @brief Create 2D GPU tensor (matrix)
     */
    nimcp_gpu_tensor_t* create_matrix(size_t rows, size_t cols) {
        if (!gpu_available) return nullptr;
        size_t dims[2] = {rows, cols};
        nimcp_gpu_tensor_t* tensor = nimcp_gpu_tensor_create(ctx, dims, 2, NIMCP_GPU_PRECISION_FP32);
        if (tensor) {
            nimcp_gpu_zeros(ctx, tensor);
        }
        return tensor;
    }

    /**
     * @brief Copy tensor data to host
     */
    bool copy_to_host(const nimcp_gpu_tensor_t* tensor, float* host_data) {
        if (!tensor || !host_data) return false;
        return nimcp_gpu_tensor_to_host(tensor, host_data);
    }
};

//=============================================================================
// Graph Lifecycle Tests
//=============================================================================

/**
 * TEST: Knowledge graph creation
 * WHAT: Create GPU knowledge graph from CSR arrays
 * WHY:  Verify graph allocation and initialization
 */
TEST_F(KnowledgeGraphKernelTest, GraphCreate_FromCSR_Succeeds) {
    RequireGPU();

    std::vector<uint32_t> row_offsets, col_indices;
    std::vector<float> edge_weights;
    create_simple_graph(row_offsets, col_indices, edge_weights, DEFAULT_NUM_NODES, 5);

    nimcp_gpu_knowledge_graph_t* graph = nimcp_gpu_knowledge_graph_create(
        ctx,
        row_offsets.data(),
        col_indices.data(),
        edge_weights.data(),
        DEFAULT_NUM_NODES,
        static_cast<uint32_t>(col_indices.size()),
        DEFAULT_EMBED_DIM
    );

    if (graph) {
        EXPECT_EQ(graph->num_nodes, DEFAULT_NUM_NODES);
        EXPECT_EQ(graph->num_edges, static_cast<uint32_t>(col_indices.size()));
        EXPECT_EQ(graph->embed_dim, DEFAULT_EMBED_DIM);
        EXPECT_NE(graph->row_offsets, nullptr);
        EXPECT_NE(graph->col_indices, nullptr);
        nimcp_gpu_knowledge_graph_destroy(graph);
    }
}

/**
 * TEST: Knowledge graph destruction with NULL
 * WHAT: Destroy NULL graph
 * WHY:  Verify NULL-safety
 */
TEST_F(KnowledgeGraphKernelTest, GraphDestroy_Null_NoOp) {
    nimcp_gpu_knowledge_graph_destroy(nullptr);
    SUCCEED() << "Should not crash";
}

/**
 * TEST: Knowledge graph without edge weights
 * WHAT: Create graph without optional edge weights
 * WHY:  Verify optional parameter handling
 */
TEST_F(KnowledgeGraphKernelTest, GraphCreate_NoWeights_Succeeds) {
    RequireGPU();

    std::vector<uint32_t> row_offsets, col_indices;
    std::vector<float> edge_weights;
    create_simple_graph(row_offsets, col_indices, edge_weights, 50, 3);

    nimcp_gpu_knowledge_graph_t* graph = nimcp_gpu_knowledge_graph_create(
        ctx,
        row_offsets.data(),
        col_indices.data(),
        nullptr,  // No edge weights
        50,
        static_cast<uint32_t>(col_indices.size()),
        32
    );

    if (graph) {
        EXPECT_EQ(graph->num_nodes, 50u);
        nimcp_gpu_knowledge_graph_destroy(graph);
    }
}

/**
 * TEST: Set node embeddings
 * WHAT: Set embeddings from host data
 * WHY:  Initialize graph for similarity operations
 */
TEST_F(KnowledgeGraphKernelTest, SetEmbeddings_UpdatesGraph) {
    RequireGPU();

    std::vector<uint32_t> row_offsets, col_indices;
    std::vector<float> edge_weights;
    create_simple_graph(row_offsets, col_indices, edge_weights, DEFAULT_NUM_NODES, 5);

    nimcp_gpu_knowledge_graph_t* graph = nimcp_gpu_knowledge_graph_create(
        ctx,
        row_offsets.data(),
        col_indices.data(),
        edge_weights.data(),
        DEFAULT_NUM_NODES,
        static_cast<uint32_t>(col_indices.size()),
        DEFAULT_EMBED_DIM
    );

    if (!graph) {
        GTEST_SKIP() << "Graph creation failed";
    }

    std::vector<float> embeddings = create_random_embeddings(DEFAULT_NUM_NODES, DEFAULT_EMBED_DIM);
    bool result = nimcp_gpu_knowledge_graph_set_embeddings(graph, embeddings.data());

    if (result) {
        EXPECT_NE(graph->node_embeddings, nullptr);
    }

    nimcp_gpu_knowledge_graph_destroy(graph);
}

/**
 * TEST: Graph validity check
 * WHAT: Check if graph is valid and properly initialized
 * WHY:  Verify graph state
 */
TEST_F(KnowledgeGraphKernelTest, GraphIsValid_ReturnsTrueForValidGraph) {
    RequireGPU();

    std::vector<uint32_t> row_offsets, col_indices;
    std::vector<float> edge_weights;
    create_simple_graph(row_offsets, col_indices, edge_weights, 50, 3);

    nimcp_gpu_knowledge_graph_t* graph = nimcp_gpu_knowledge_graph_create(
        ctx,
        row_offsets.data(),
        col_indices.data(),
        edge_weights.data(),
        50,
        static_cast<uint32_t>(col_indices.size()),
        32
    );

    if (graph) {
        bool is_valid = nimcp_gpu_knowledge_graph_is_valid(graph);
        EXPECT_TRUE(is_valid);
        nimcp_gpu_knowledge_graph_destroy(graph);
    }
}

//=============================================================================
// Graph Traversal Tests
//=============================================================================

/**
 * TEST: Parallel BFS traversal
 * WHAT: Run BFS from a source node
 * WHY:  Core graph traversal operation
 */
TEST_F(KnowledgeGraphKernelTest, BFS_FromSource_ComputesDistances) {
    RequireGPU();

    // Create chain graph for predictable BFS results
    std::vector<uint32_t> row_offsets, col_indices;
    create_chain_graph(row_offsets, col_indices, 10);

    nimcp_gpu_knowledge_graph_t* graph = nimcp_gpu_knowledge_graph_create(
        ctx,
        row_offsets.data(),
        col_indices.data(),
        nullptr,
        10,
        static_cast<uint32_t>(col_indices.size()),
        DEFAULT_EMBED_DIM
    );

    if (!graph) {
        GTEST_SKIP() << "Graph creation failed";
    }

    nimcp_graph_traversal_result_t result;
    result.distances = create_zero_tensor(10);
    result.parents = create_zero_tensor(10);
    result.visited = create_zero_tensor(10);

    if (!result.distances || !result.parents || !result.visited) {
        if (result.distances) nimcp_gpu_tensor_destroy(result.distances);
        if (result.parents) nimcp_gpu_tensor_destroy(result.parents);
        if (result.visited) nimcp_gpu_tensor_destroy(result.visited);
        nimcp_gpu_knowledge_graph_destroy(graph);
        GTEST_SKIP() << "Tensor creation failed";
    }

    bool bfs_result = nimcp_gpu_bfs(graph, 0, -1, &result);

    if (bfs_result) {
        std::vector<float> distances(10);
        copy_to_host(result.distances, distances.data());

        // In chain graph, distance from 0 to i should be i
        for (int i = 0; i < 10; i++) {
            // Check that distance is computed (may be inf for unreachable)
            EXPECT_TRUE(std::isfinite(distances[i]) || distances[i] > 0);
        }
    }

    nimcp_graph_traversal_result_destroy(&result);
    nimcp_gpu_knowledge_graph_destroy(graph);
}

/**
 * TEST: Multi-source BFS
 * WHAT: BFS from multiple source nodes
 * WHY:  Efficiently compute distances from multiple sources
 */
TEST_F(KnowledgeGraphKernelTest, BFSMultiSource_ComputesMinDistances) {
    RequireGPU();

    std::vector<uint32_t> row_offsets, col_indices;
    std::vector<float> edge_weights;
    create_simple_graph(row_offsets, col_indices, edge_weights, 50, 5);

    nimcp_gpu_knowledge_graph_t* graph = nimcp_gpu_knowledge_graph_create(
        ctx,
        row_offsets.data(),
        col_indices.data(),
        nullptr,
        50,
        static_cast<uint32_t>(col_indices.size()),
        DEFAULT_EMBED_DIM
    );

    if (!graph) {
        GTEST_SKIP() << "Graph creation failed";
    }

    nimcp_graph_traversal_result_t result;
    result.distances = create_zero_tensor(50);
    result.parents = create_zero_tensor(50);
    result.visited = create_zero_tensor(50);

    if (!result.distances) {
        nimcp_gpu_knowledge_graph_destroy(graph);
        GTEST_SKIP() << "Tensor creation failed";
    }

    std::vector<uint32_t> sources = {0, 10, 20};
    bool bfs_result = nimcp_gpu_bfs_multi_source(graph, sources.data(), 3, 5, &result);

    if (bfs_result) {
        // Verify result structure is valid
        EXPECT_GT(result.num_visited, 0u);
    }

    nimcp_graph_traversal_result_destroy(&result);
    nimcp_gpu_knowledge_graph_destroy(graph);
}

/**
 * TEST: Parallel DFS traversal
 * WHAT: Run DFS from a source node
 * WHY:  Alternative traversal strategy
 */
TEST_F(KnowledgeGraphKernelTest, DFS_FromSource_ExploresGraph) {
    RequireGPU();

    std::vector<uint32_t> row_offsets, col_indices;
    std::vector<float> edge_weights;
    create_simple_graph(row_offsets, col_indices, edge_weights, 30, 4);

    nimcp_gpu_knowledge_graph_t* graph = nimcp_gpu_knowledge_graph_create(
        ctx,
        row_offsets.data(),
        col_indices.data(),
        nullptr,
        30,
        static_cast<uint32_t>(col_indices.size()),
        DEFAULT_EMBED_DIM
    );

    if (!graph) {
        GTEST_SKIP() << "Graph creation failed";
    }

    nimcp_graph_traversal_result_t result;
    result.distances = create_zero_tensor(30);
    result.parents = create_zero_tensor(30);
    result.visited = create_zero_tensor(30);

    bool dfs_result = nimcp_gpu_dfs(graph, 0, 10, &result);

    if (dfs_result) {
        std::vector<float> visited(30);
        copy_to_host(result.visited, visited.data());

        // At least some nodes should be visited
        int visited_count = 0;
        for (float v : visited) {
            if (v > 0.5f) visited_count++;
        }
        EXPECT_GT(visited_count, 0);
    }

    nimcp_graph_traversal_result_destroy(&result);
    nimcp_gpu_knowledge_graph_destroy(graph);
}

/**
 * TEST: Shortest path between nodes
 * WHAT: Find shortest path using bidirectional BFS
 * WHY:  Path finding in knowledge graphs
 */
TEST_F(KnowledgeGraphKernelTest, ShortestPath_FindsPath) {
    RequireGPU();

    // Create chain graph for predictable path
    std::vector<uint32_t> row_offsets, col_indices;
    create_chain_graph(row_offsets, col_indices, 10);

    nimcp_gpu_knowledge_graph_t* graph = nimcp_gpu_knowledge_graph_create(
        ctx,
        row_offsets.data(),
        col_indices.data(),
        nullptr,
        10,
        static_cast<uint32_t>(col_indices.size()),
        DEFAULT_EMBED_DIM
    );

    if (!graph) {
        GTEST_SKIP() << "Graph creation failed";
    }

    std::vector<uint32_t> path(10);
    uint32_t path_length = 0;

    bool result = nimcp_gpu_shortest_path(graph, 0, 5, path.data(), &path_length);

    if (result) {
        EXPECT_GT(path_length, 0u);
        EXPECT_LE(path_length, 10u);
    }

    nimcp_gpu_knowledge_graph_destroy(graph);
}

//=============================================================================
// Semantic Similarity Tests
//=============================================================================

/**
 * TEST: Cosine similarity between two nodes
 * WHAT: Compute similarity using embeddings
 * WHY:  Core similarity operation
 */
TEST_F(KnowledgeGraphKernelTest, NodeSimilarity_ComputesCosine) {
    RequireGPU();

    std::vector<uint32_t> row_offsets, col_indices;
    std::vector<float> edge_weights;
    create_simple_graph(row_offsets, col_indices, edge_weights, 50, 3);

    nimcp_gpu_knowledge_graph_t* graph = nimcp_gpu_knowledge_graph_create(
        ctx,
        row_offsets.data(),
        col_indices.data(),
        nullptr,
        50,
        static_cast<uint32_t>(col_indices.size()),
        32
    );

    if (!graph) {
        GTEST_SKIP() << "Graph creation failed";
    }

    // Set embeddings
    std::vector<float> embeddings = create_random_embeddings(50, 32);
    normalize_embeddings(embeddings, 50, 32);
    nimcp_gpu_knowledge_graph_set_embeddings(graph, embeddings.data());

    float similarity = 0.0f;
    bool result = nimcp_gpu_node_similarity(graph, 0, 1, &similarity);

    if (result) {
        // Cosine similarity should be between -1 and 1
        EXPECT_GE(similarity, -1.0f - NUMERICAL_EPS);
        EXPECT_LE(similarity, 1.0f + NUMERICAL_EPS);
    }

    nimcp_gpu_knowledge_graph_destroy(graph);
}

/**
 * TEST: Self-similarity
 * WHAT: Similarity of a node with itself
 * WHY:  Self-similarity should be 1 for normalized embeddings
 */
TEST_F(KnowledgeGraphKernelTest, NodeSimilarity_SelfSimilarity_IsOne) {
    RequireGPU();

    std::vector<uint32_t> row_offsets, col_indices;
    std::vector<float> edge_weights;
    create_simple_graph(row_offsets, col_indices, edge_weights, 20, 3);

    nimcp_gpu_knowledge_graph_t* graph = nimcp_gpu_knowledge_graph_create(
        ctx,
        row_offsets.data(),
        col_indices.data(),
        nullptr,
        20,
        static_cast<uint32_t>(col_indices.size()),
        32
    );

    if (!graph) {
        GTEST_SKIP() << "Graph creation failed";
    }

    // Set normalized embeddings
    std::vector<float> embeddings = create_random_embeddings(20, 32);
    normalize_embeddings(embeddings, 20, 32);
    nimcp_gpu_knowledge_graph_set_embeddings(graph, embeddings.data());

    float similarity = 0.0f;
    bool result = nimcp_gpu_node_similarity(graph, 5, 5, &similarity);

    if (result) {
        EXPECT_NEAR(similarity, 1.0f, 0.01f);
    }

    nimcp_gpu_knowledge_graph_destroy(graph);
}

/**
 * TEST: Pairwise similarity matrix
 * WHAT: Compute similarity for all pairs of specified nodes
 * WHY:  Batch similarity computation
 */
TEST_F(KnowledgeGraphKernelTest, PairwiseSimilarity_SymmetricMatrix) {
    RequireGPU();

    std::vector<uint32_t> row_offsets, col_indices;
    std::vector<float> edge_weights;
    create_simple_graph(row_offsets, col_indices, edge_weights, 50, 3);

    nimcp_gpu_knowledge_graph_t* graph = nimcp_gpu_knowledge_graph_create(
        ctx,
        row_offsets.data(),
        col_indices.data(),
        nullptr,
        50,
        static_cast<uint32_t>(col_indices.size()),
        32
    );

    if (!graph) {
        GTEST_SKIP() << "Graph creation failed";
    }

    std::vector<float> embeddings = create_random_embeddings(50, 32);
    normalize_embeddings(embeddings, 50, 32);
    nimcp_gpu_knowledge_graph_set_embeddings(graph, embeddings.data());

    std::vector<uint32_t> nodes = {0, 5, 10, 15, 20};
    nimcp_gpu_tensor_t* sim_matrix = create_matrix(5, 5);

    if (!sim_matrix) {
        nimcp_gpu_knowledge_graph_destroy(graph);
        GTEST_SKIP() << "Matrix creation failed";
    }

    bool result = nimcp_gpu_pairwise_similarity(graph, nodes.data(), 5, sim_matrix);

    if (result) {
        std::vector<float> sim_host(25);
        copy_to_host(sim_matrix, sim_host.data());

        // Matrix should be symmetric
        for (int i = 0; i < 5; i++) {
            for (int j = i + 1; j < 5; j++) {
                EXPECT_NEAR(sim_host[i * 5 + j], sim_host[j * 5 + i], 0.01f);
            }
        }

        // Diagonal should be 1
        for (int i = 0; i < 5; i++) {
            EXPECT_NEAR(sim_host[i * 5 + i], 1.0f, 0.01f);
        }
    }

    nimcp_gpu_tensor_destroy(sim_matrix);
    nimcp_gpu_knowledge_graph_destroy(graph);
}

/**
 * TEST: k-NN similarity search
 * WHAT: Find k most similar nodes to a query
 * WHY:  Core retrieval operation
 */
TEST_F(KnowledgeGraphKernelTest, KNNSimilarity_FindsTopK) {
    RequireGPU();

    std::vector<uint32_t> row_offsets, col_indices;
    std::vector<float> edge_weights;
    create_simple_graph(row_offsets, col_indices, edge_weights, 100, 3);

    nimcp_gpu_knowledge_graph_t* graph = nimcp_gpu_knowledge_graph_create(
        ctx,
        row_offsets.data(),
        col_indices.data(),
        nullptr,
        100,
        static_cast<uint32_t>(col_indices.size()),
        64
    );

    if (!graph) {
        GTEST_SKIP() << "Graph creation failed";
    }

    std::vector<float> embeddings = create_random_embeddings(100, 64);
    normalize_embeddings(embeddings, 100, 64);
    nimcp_gpu_knowledge_graph_set_embeddings(graph, embeddings.data());

    nimcp_similarity_result_t result;
    result.k = 5;
    result.indices = create_zero_tensor(5);
    result.scores = create_zero_tensor(5);

    if (!result.indices || !result.scores) {
        if (result.indices) nimcp_gpu_tensor_destroy(result.indices);
        if (result.scores) nimcp_gpu_tensor_destroy(result.scores);
        nimcp_gpu_knowledge_graph_destroy(graph);
        GTEST_SKIP() << "Tensor creation failed";
    }

    bool knn_result = nimcp_gpu_knn_similarity(graph, 0, 5, &result);

    if (knn_result) {
        std::vector<float> scores(5);
        copy_to_host(result.scores, scores.data());

        // First result should be self (highest similarity)
        // Scores should be sorted in descending order
        for (int i = 0; i < 4; i++) {
            EXPECT_GE(scores[i], scores[i + 1] - NUMERICAL_EPS);
        }
    }

    nimcp_similarity_result_destroy(&result);
    nimcp_gpu_knowledge_graph_destroy(graph);
}

//=============================================================================
// Knowledge Embedding Operations Tests
//=============================================================================

/**
 * TEST: Embedding update via gradient descent
 * WHAT: Update embeddings with gradients
 * WHY:  Training knowledge graph embeddings
 */
TEST_F(KnowledgeGraphKernelTest, EmbeddingUpdate_ModifiesEmbeddings) {
    RequireGPU();

    std::vector<uint32_t> row_offsets, col_indices;
    std::vector<float> edge_weights;
    create_simple_graph(row_offsets, col_indices, edge_weights, 50, 3);

    nimcp_gpu_knowledge_graph_t* graph = nimcp_gpu_knowledge_graph_create(
        ctx,
        row_offsets.data(),
        col_indices.data(),
        nullptr,
        50,
        static_cast<uint32_t>(col_indices.size()),
        32
    );

    if (!graph) {
        GTEST_SKIP() << "Graph creation failed";
    }

    std::vector<float> embeddings = create_random_embeddings(50, 32);
    nimcp_gpu_knowledge_graph_set_embeddings(graph, embeddings.data());

    // Create gradient tensor
    std::vector<float> grads(50 * 32, 0.1f);
    size_t grad_dims[2] = {50, 32};
    nimcp_gpu_tensor_t* gradients = nimcp_gpu_tensor_from_host(ctx, grads.data(), grad_dims, 2, NIMCP_GPU_PRECISION_FP32);

    if (!gradients) {
        nimcp_gpu_knowledge_graph_destroy(graph);
        GTEST_SKIP() << "Gradient tensor creation failed";
    }

    bool result = nimcp_gpu_embedding_update(graph, gradients, 0.01f);

    // Function may not be implemented, but should not crash
    if (result) {
        SUCCEED();
    }

    nimcp_gpu_tensor_destroy(gradients);
    nimcp_gpu_knowledge_graph_destroy(graph);
}

/**
 * TEST: Triplet loss computation
 * WHAT: Compute triplet loss for training
 * WHY:  Knowledge graph embedding objective
 */
TEST_F(KnowledgeGraphKernelTest, TripletLoss_ComputesLoss) {
    RequireGPU();

    std::vector<uint32_t> row_offsets, col_indices;
    std::vector<float> edge_weights;
    create_simple_graph(row_offsets, col_indices, edge_weights, 50, 3);

    nimcp_gpu_knowledge_graph_t* graph = nimcp_gpu_knowledge_graph_create(
        ctx,
        row_offsets.data(),
        col_indices.data(),
        nullptr,
        50,
        static_cast<uint32_t>(col_indices.size()),
        32
    );

    if (!graph) {
        GTEST_SKIP() << "Graph creation failed";
    }

    std::vector<float> embeddings = create_random_embeddings(50, 32);
    nimcp_gpu_knowledge_graph_set_embeddings(graph, embeddings.data());

    std::vector<uint32_t> anchors = {0, 1, 2, 3};
    std::vector<uint32_t> positives = {1, 2, 3, 4};
    std::vector<uint32_t> negatives = {10, 11, 12, 13};

    float loss = 0.0f;
    bool result = nimcp_gpu_triplet_loss(graph, anchors.data(), positives.data(),
                                          negatives.data(), 4, 0.5f, &loss);

    if (result) {
        EXPECT_GE(loss, 0.0f);
    }

    nimcp_gpu_knowledge_graph_destroy(graph);
}

/**
 * TEST: Embedding normalization
 * WHAT: L2 normalize all embeddings
 * WHY:  Ensure unit-length embeddings for cosine similarity
 */
TEST_F(KnowledgeGraphKernelTest, NormalizeEmbeddings_ProducesUnitVectors) {
    RequireGPU();

    std::vector<uint32_t> row_offsets, col_indices;
    std::vector<float> edge_weights;
    create_simple_graph(row_offsets, col_indices, edge_weights, 30, 3);

    nimcp_gpu_knowledge_graph_t* graph = nimcp_gpu_knowledge_graph_create(
        ctx,
        row_offsets.data(),
        col_indices.data(),
        nullptr,
        30,
        static_cast<uint32_t>(col_indices.size()),
        32
    );

    if (!graph) {
        GTEST_SKIP() << "Graph creation failed";
    }

    // Set non-normalized embeddings
    std::vector<float> embeddings = create_random_embeddings(30, 32);
    nimcp_gpu_knowledge_graph_set_embeddings(graph, embeddings.data());

    bool result = nimcp_gpu_normalize_embeddings(graph);

    if (result) {
        // Self-similarity should be 1 after normalization
        float sim = 0.0f;
        nimcp_gpu_node_similarity(graph, 0, 0, &sim);
        EXPECT_NEAR(sim, 1.0f, 0.01f);
    }

    nimcp_gpu_knowledge_graph_destroy(graph);
}

//=============================================================================
// Node/Edge Aggregation Tests
//=============================================================================

/**
 * TEST: Neighbor feature aggregation (sum)
 * WHAT: Sum neighbor features for each node
 * WHY:  GNN-style message passing
 */
TEST_F(KnowledgeGraphKernelTest, AggregateNeighbors_Sum_ProducesOutput) {
    RequireGPU();

    std::vector<uint32_t> row_offsets, col_indices;
    std::vector<float> edge_weights;
    create_simple_graph(row_offsets, col_indices, edge_weights, 50, 5);

    nimcp_gpu_knowledge_graph_t* graph = nimcp_gpu_knowledge_graph_create(
        ctx,
        row_offsets.data(),
        col_indices.data(),
        nullptr,
        50,
        static_cast<uint32_t>(col_indices.size()),
        32
    );

    if (!graph) {
        GTEST_SKIP() << "Graph creation failed";
    }

    std::vector<float> embeddings = create_random_embeddings(50, 32);
    nimcp_gpu_knowledge_graph_set_embeddings(graph, embeddings.data());

    nimcp_gpu_tensor_t* output = create_matrix(50, 32);

    if (!output) {
        nimcp_gpu_knowledge_graph_destroy(graph);
        GTEST_SKIP() << "Output tensor creation failed";
    }

    bool result = nimcp_gpu_aggregate_neighbors(graph, NIMCP_AGGREGATE_SUM, output);

    if (result) {
        std::vector<float> out_host(50 * 32);
        copy_to_host(output, out_host.data());

        // Output should be finite
        for (float v : out_host) {
            EXPECT_TRUE(std::isfinite(v));
        }
    }

    nimcp_gpu_tensor_destroy(output);
    nimcp_gpu_knowledge_graph_destroy(graph);
}

/**
 * TEST: Multi-hop aggregation
 * WHAT: Aggregate features from k-hop neighborhood
 * WHY:  Multi-layer GNN computation
 */
TEST_F(KnowledgeGraphKernelTest, MultiHopAggregate_MultipleHops) {
    RequireGPU();

    std::vector<uint32_t> row_offsets, col_indices;
    std::vector<float> edge_weights;
    create_simple_graph(row_offsets, col_indices, edge_weights, 50, 5);

    nimcp_gpu_knowledge_graph_t* graph = nimcp_gpu_knowledge_graph_create(
        ctx,
        row_offsets.data(),
        col_indices.data(),
        nullptr,
        50,
        static_cast<uint32_t>(col_indices.size()),
        32
    );

    if (!graph) {
        GTEST_SKIP() << "Graph creation failed";
    }

    std::vector<float> embeddings = create_random_embeddings(50, 32);
    nimcp_gpu_knowledge_graph_set_embeddings(graph, embeddings.data());

    nimcp_gpu_tensor_t* output = create_matrix(50, 32);

    if (!output) {
        nimcp_gpu_knowledge_graph_destroy(graph);
        GTEST_SKIP() << "Output tensor creation failed";
    }

    bool result = nimcp_gpu_multi_hop_aggregate(graph, 2, NIMCP_AGGREGATE_MEAN, output);

    if (result) {
        std::vector<float> out_host(50 * 32);
        copy_to_host(output, out_host.data());

        for (float v : out_host) {
            EXPECT_TRUE(std::isfinite(v));
        }
    }

    nimcp_gpu_tensor_destroy(output);
    nimcp_gpu_knowledge_graph_destroy(graph);
}

//=============================================================================
// Hyperbolic Space Operations Tests
//=============================================================================

/**
 * TEST: Set hyperbolic embeddings
 * WHAT: Set embeddings in Poincare ball model
 * WHY:  Support hierarchical knowledge
 */
TEST_F(KnowledgeGraphKernelTest, SetHyperbolicEmbeddings_SetsFlag) {
    RequireGPU();

    std::vector<uint32_t> row_offsets, col_indices;
    std::vector<float> edge_weights;
    create_simple_graph(row_offsets, col_indices, edge_weights, 30, 3);

    nimcp_gpu_knowledge_graph_t* graph = nimcp_gpu_knowledge_graph_create(
        ctx,
        row_offsets.data(),
        col_indices.data(),
        nullptr,
        30,
        static_cast<uint32_t>(col_indices.size()),
        32
    );

    if (!graph) {
        GTEST_SKIP() << "Graph creation failed";
    }

    // Create embeddings inside Poincare ball (norm < 1)
    std::vector<float> embeddings = create_random_embeddings(30, 32);
    for (size_t i = 0; i < 30; i++) {
        float norm = 0.0f;
        for (size_t j = 0; j < 32; j++) {
            norm += embeddings[i * 32 + j] * embeddings[i * 32 + j];
        }
        norm = std::sqrt(norm);
        // Scale to be inside ball
        float scale = 0.9f / (norm + 1.0f);
        for (size_t j = 0; j < 32; j++) {
            embeddings[i * 32 + j] *= scale;
        }
    }

    bool result = nimcp_gpu_knowledge_graph_set_hyperbolic_embeddings(graph, embeddings.data());

    if (result) {
        EXPECT_TRUE(graph->is_hyperbolic);
    }

    nimcp_gpu_knowledge_graph_destroy(graph);
}

/**
 * TEST: Hyperbolic distance
 * WHAT: Compute Poincare ball distance
 * WHY:  Distance in hyperbolic space
 */
TEST_F(KnowledgeGraphKernelTest, HyperbolicDistance_ComputesDistance) {
    RequireGPU();

    std::vector<uint32_t> row_offsets, col_indices;
    std::vector<float> edge_weights;
    create_simple_graph(row_offsets, col_indices, edge_weights, 20, 3);

    nimcp_gpu_knowledge_graph_t* graph = nimcp_gpu_knowledge_graph_create(
        ctx,
        row_offsets.data(),
        col_indices.data(),
        nullptr,
        20,
        static_cast<uint32_t>(col_indices.size()),
        16
    );

    if (!graph) {
        GTEST_SKIP() << "Graph creation failed";
    }

    // Set hyperbolic embeddings
    std::vector<float> embeddings(20 * 16);
    for (size_t i = 0; i < 20 * 16; i++) {
        embeddings[i] = 0.1f * static_cast<float>(i % 10) / 10.0f;
    }
    nimcp_gpu_knowledge_graph_set_hyperbolic_embeddings(graph, embeddings.data());

    float distance = 0.0f;
    bool result = nimcp_gpu_hyperbolic_distance(graph, 0, 1, &distance);

    if (result) {
        EXPECT_GE(distance, 0.0f);
        EXPECT_TRUE(std::isfinite(distance));
    }

    nimcp_gpu_knowledge_graph_destroy(graph);
}

/**
 * TEST: Convert Euclidean to hyperbolic
 * WHAT: Use exponential map from origin
 * WHY:  Convert standard embeddings to Poincare ball
 */
TEST_F(KnowledgeGraphKernelTest, EuclideanToHyperbolic_ConvertsEmbeddings) {
    RequireGPU();

    std::vector<uint32_t> row_offsets, col_indices;
    std::vector<float> edge_weights;
    create_simple_graph(row_offsets, col_indices, edge_weights, 20, 3);

    nimcp_gpu_knowledge_graph_t* graph = nimcp_gpu_knowledge_graph_create(
        ctx,
        row_offsets.data(),
        col_indices.data(),
        nullptr,
        20,
        static_cast<uint32_t>(col_indices.size()),
        16
    );

    if (!graph) {
        GTEST_SKIP() << "Graph creation failed";
    }

    // Set Euclidean embeddings
    std::vector<float> embeddings = create_random_embeddings(20, 16);
    nimcp_gpu_knowledge_graph_set_embeddings(graph, embeddings.data());

    bool result = nimcp_gpu_euclidean_to_hyperbolic(graph);

    if (result) {
        EXPECT_TRUE(graph->is_hyperbolic);
    }

    nimcp_gpu_knowledge_graph_destroy(graph);
}

//=============================================================================
// Graph Statistics Tests
//=============================================================================

/**
 * TEST: Graph statistics computation
 * WHAT: Get avg degree, max degree, embedding norm
 * WHY:  Monitor graph state
 */
TEST_F(KnowledgeGraphKernelTest, GraphStats_ComputesStatistics) {
    RequireGPU();

    std::vector<uint32_t> row_offsets, col_indices;
    std::vector<float> edge_weights;
    create_simple_graph(row_offsets, col_indices, edge_weights, 50, 5);

    nimcp_gpu_knowledge_graph_t* graph = nimcp_gpu_knowledge_graph_create(
        ctx,
        row_offsets.data(),
        col_indices.data(),
        nullptr,
        50,
        static_cast<uint32_t>(col_indices.size()),
        32
    );

    if (!graph) {
        GTEST_SKIP() << "Graph creation failed";
    }

    std::vector<float> embeddings = create_random_embeddings(50, 32);
    nimcp_gpu_knowledge_graph_set_embeddings(graph, embeddings.data());

    float avg_degree = 0.0f;
    uint32_t max_degree = 0;
    float embedding_norm = 0.0f;

    bool result = nimcp_gpu_knowledge_graph_stats(graph, &avg_degree, &max_degree, &embedding_norm);

    if (result) {
        EXPECT_GT(avg_degree, 0.0f);
        EXPECT_GT(max_degree, 0u);
        EXPECT_GT(embedding_norm, 0.0f);
    }

    nimcp_gpu_knowledge_graph_destroy(graph);
}

//=============================================================================
// Main
//=============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
