/**
 * @file test_performance_optimizations.cpp
 * @brief TDD tests for algorithmic complexity optimizations
 *
 * This test suite follows Test-Driven Development (TDD) principles:
 * 1. Tests are written BEFORE implementation
 * 2. Tests define the expected behavior and interface
 * 3. Implementation follows to make tests pass
 *
 * Design Patterns Used:
 * - Iterator Pattern: For traversing incoming/outgoing synapses
 * - Strategy Pattern: For different pathfinding algorithms
 * - Builder Pattern: For constructing optimized network structures
 * - Abstract Data Type: For min-heap with clean interface
 */

#include <gtest/gtest.h>
#include <chrono>
#include <vector>
#include "../include/nimcp_neuralnet.h"
#include "../include/utils/nimcp_graph.h"
#include "../include/utils/nimcp_min_heap.h"

//==============================================================================
// Test Fixture
//==============================================================================

class PerformanceOptimizationTest : public ::testing::Test {
   protected:
    void SetUp() override
    {
        // Setup common test infrastructure
    }

    void TearDown() override
    {
        // Cleanup
    }

    // Helper to measure execution time in microseconds
    template <typename Func>
    long long measure_time_us(Func&& func)
    {
        auto start = std::chrono::high_resolution_clock::now();
        func();
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }
};

//==============================================================================
// PART 1: Bidirectional Synapse Tracking Tests (TDD)
//==============================================================================

/**
 * @test Verify bidirectional synapse API exists
 *
 * Design: Iterator Pattern
 * - Forward iterator for outgoing synapses (existing)
 * - Reverse iterator for incoming synapses (new)
 */
TEST_F(PerformanceOptimizationTest, DISABLED_BidirectionalSynapse_APIExists)
{
    // TDD: This test defines the API we want to exist
    network_config_t config = {0};
    config.num_neurons = 10;
    config.ei_ratio = 0.8f;
    config.learning_rate = 0.01f;
    config.hebbian_rate = 0.001f;
    config.min_weight = 0.0f;
    config.max_weight = 1.0f;

    neural_network_t network = neural_network_create(&config);
    ASSERT_NE(network, nullptr);

    // Add connections: 0 -> 5, 1 -> 5, 2 -> 5
    ASSERT_TRUE(neural_network_add_connection(network, 0, 5, 0.5f));
    ASSERT_TRUE(neural_network_add_connection(network, 1, 5, 0.6f));
    ASSERT_TRUE(neural_network_add_connection(network, 2, 5, 0.7f));

    // NEW API: Get incoming synapse count (should be 3 for neuron 5)
    uint32_t incoming_count = neural_network_get_incoming_synapse_count(network, 5);
    EXPECT_EQ(incoming_count, 3);

    // NEW API: Iterate through incoming synapses
    const synapse_t* incoming_synapses = nullptr;
    uint32_t count = neural_network_get_incoming_synapses(network, 5, &incoming_synapses);
    EXPECT_EQ(count, 3);
    ASSERT_NE(incoming_synapses, nullptr);

    // Verify incoming synapse sources
    std::vector<uint32_t> sources;
    for (uint32_t i = 0; i < count; i++) {
        sources.push_back(incoming_synapses[i].target_id);  // In reverse edge, target_id holds source
    }

    EXPECT_TRUE(std::find(sources.begin(), sources.end(), 0) != sources.end());
    EXPECT_TRUE(std::find(sources.begin(), sources.end(), 1) != sources.end());
    EXPECT_TRUE(std::find(sources.begin(), sources.end(), 2) != sources.end());

    neural_network_destroy(network);
}

/**
 * @test Verify O(S) complexity for input summation (vs old O(N×S))
 *
 * This test validates the performance improvement from bidirectional tracking
 */
TEST_F(PerformanceOptimizationTest, DISABLED_BidirectionalSynapse_PerformanceImprovement)
{
    // Create large network to measure performance
    network_config_t config = {0};
    config.num_neurons = 1000;  // Large network
    config.ei_ratio = 0.8f;
    config.learning_rate = 0.01f;
    config.hebbian_rate = 0.001f;
    config.min_weight = 0.0f;
    config.max_weight = 1.0f;

    neural_network_t network = neural_network_create(&config);
    ASSERT_NE(network, nullptr);

    // Create fully connected layer: 500 neurons -> neuron 999
    for (uint32_t i = 0; i < 500; i++) {
        ASSERT_TRUE(neural_network_add_connection(network, i, 999, 0.5f));
    }

    // Measure time to get all incoming synapses
    // With bidirectional tracking: Should be O(S) = O(500)
    // Without: Would be O(N×S) = O(1000×500) = O(500,000)

    long long time_us = measure_time_us([&]() {
        const synapse_t* incoming = nullptr;
        neural_network_get_incoming_synapses(network, 999, &incoming);
    });

    // Should complete in microseconds, not milliseconds
    EXPECT_LT(time_us, 1000) << "Incoming synapse lookup too slow: " << time_us << " us";

    neural_network_destroy(network);
}

/**
 * @test Verify bidirectional consistency
 *
 * When adding/removing connections, both forward and reverse edges must stay synchronized
 */
TEST_F(PerformanceOptimizationTest, DISABLED_BidirectionalSynapse_ConsistencyOnModification)
{
    network_config_t config = {0};
    config.num_neurons = 10;
    config.ei_ratio = 0.8f;
    config.learning_rate = 0.01f;
    config.min_weight = 0.0f;
    config.max_weight = 1.0f;

    neural_network_t network = neural_network_create(&config);
    ASSERT_NE(network, nullptr);

    // Add connection: 0 -> 5
    ASSERT_TRUE(neural_network_add_connection(network, 0, 5, 0.5f));

    // Verify forward edge exists
    // (Existing API would check this)

    // Verify reverse edge exists
    uint32_t incoming_count = neural_network_get_incoming_synapse_count(network, 5);
    EXPECT_EQ(incoming_count, 1);

    // Add another connection: 3 -> 5
    ASSERT_TRUE(neural_network_add_connection(network, 3, 5, 0.7f));
    incoming_count = neural_network_get_incoming_synapse_count(network, 5);
    EXPECT_EQ(incoming_count, 2);

    // TODO: When synapse removal API exists, test that both edges are removed

    neural_network_destroy(network);
}

//==============================================================================
// PART 2: Min-Heap Data Structure Tests (TDD)
//==============================================================================

/**
 * @test Min-heap creation and destruction
 *
 * Design Pattern: Abstract Data Type
 * - Clean interface with create/destroy
 * - No implementation details leaked
 */
TEST_F(PerformanceOptimizationTest, DISABLED_MinHeap_CreateDestroy)
{
    nimcp_min_heap_t* heap = nimcp_min_heap_create(100);  // capacity = 100
    ASSERT_NE(heap, nullptr);

    EXPECT_EQ(nimcp_min_heap_size(heap), 0);
    EXPECT_TRUE(nimcp_min_heap_is_empty(heap));
    EXPECT_FALSE(nimcp_min_heap_is_full(heap));

    nimcp_min_heap_destroy(heap);
}

/**
 * @test Min-heap insert and extract operations
 *
 * Verify O(log n) insert and extract-min operations
 */
TEST_F(PerformanceOptimizationTest, DISABLED_MinHeap_InsertExtract)
{
    nimcp_min_heap_t* heap = nimcp_min_heap_create(10);
    ASSERT_NE(heap, nullptr);

    // Insert elements with different priorities
    nimcp_heap_element_t elem1 = {1, 5.0f};
    nimcp_heap_element_t elem2 = {2, 2.0f};
    nimcp_heap_element_t elem3 = {3, 8.0f};
    nimcp_heap_element_t elem4 = {4, 1.0f};

    EXPECT_TRUE(nimcp_min_heap_insert(heap, &elem1));
    EXPECT_TRUE(nimcp_min_heap_insert(heap, &elem2));
    EXPECT_TRUE(nimcp_min_heap_insert(heap, &elem3));
    EXPECT_TRUE(nimcp_min_heap_insert(heap, &elem4));

    EXPECT_EQ(nimcp_min_heap_size(heap), 4);

    // Extract minimum (should be elem4 with priority 1.0)
    nimcp_heap_element_t min_elem;
    EXPECT_TRUE(nimcp_min_heap_extract_min(heap, &min_elem));
    EXPECT_EQ(min_elem.vertex_id, 4);
    EXPECT_FLOAT_EQ(min_elem.priority, 1.0f);

    // Next minimum (should be elem2 with priority 2.0)
    EXPECT_TRUE(nimcp_min_heap_extract_min(heap, &min_elem));
    EXPECT_EQ(min_elem.vertex_id, 2);
    EXPECT_FLOAT_EQ(min_elem.priority, 2.0f);

    EXPECT_EQ(nimcp_min_heap_size(heap), 2);

    nimcp_min_heap_destroy(heap);
}

/**
 * @test Min-heap decrease key operation
 *
 * Required for Dijkstra's algorithm optimization
 */
TEST_F(PerformanceOptimizationTest, DISABLED_MinHeap_DecreaseKey)
{
    nimcp_min_heap_t* heap = nimcp_min_heap_create(10);
    ASSERT_NE(heap, nullptr);

    nimcp_heap_element_t elem1 = {1, 10.0f};
    nimcp_heap_element_t elem2 = {2, 5.0f};

    nimcp_min_heap_insert(heap, &elem1);
    nimcp_min_heap_insert(heap, &elem2);

    // Decrease priority of vertex 1 from 10.0 to 3.0
    EXPECT_TRUE(nimcp_min_heap_decrease_key(heap, 1, 3.0f));

    // Now minimum should be vertex 1
    nimcp_heap_element_t min_elem;
    EXPECT_TRUE(nimcp_min_heap_extract_min(heap, &min_elem));
    EXPECT_EQ(min_elem.vertex_id, 1);
    EXPECT_FLOAT_EQ(min_elem.priority, 3.0f);

    nimcp_min_heap_destroy(heap);
}

/**
 * @test Min-heap performance: O(log n) operations
 *
 * Verify that operations scale logarithmically
 */
TEST_F(PerformanceOptimizationTest, DISABLED_MinHeap_LogarithmicPerformance)
{
    const int N = 10000;
    nimcp_min_heap_t* heap = nimcp_min_heap_create(N);
    ASSERT_NE(heap, nullptr);

    // Insert N elements - should be O(N log N) total
    long long insert_time = measure_time_us([&]() {
        for (int i = 0; i < N; i++) {
            nimcp_heap_element_t elem = {(uint32_t)i, (float)(N - i)};
            nimcp_min_heap_insert(heap, &elem);
        }
    });

    // Extract all elements - should be O(N log N) total
    long long extract_time = measure_time_us([&]() {
        nimcp_heap_element_t elem;
        for (int i = 0; i < N; i++) {
            nimcp_min_heap_extract_min(heap, &elem);
        }
    });

    // Both operations should complete in milliseconds, not seconds
    EXPECT_LT(insert_time, 100000) << "Heap insertion too slow: " << insert_time << " us";
    EXPECT_LT(extract_time, 100000) << "Heap extraction too slow: " << extract_time << " us";

    nimcp_min_heap_destroy(heap);
}

//==============================================================================
// PART 3: Optimized Dijkstra Tests (TDD)
//==============================================================================

/**
 * @test Dijkstra correctness with heap optimization
 *
 * Design Pattern: Strategy Pattern
 * - Same interface as original Dijkstra
 * - Different implementation (heap-based vs linear search)
 */
TEST_F(PerformanceOptimizationTest, DISABLED_Dijkstra_HeapBased_Correctness)
{
    NimcpGraph* graph = nimcp_graph_create();
    ASSERT_NE(graph, nullptr);

    // Create simple graph: 0 --5-- 1 --2-- 2
    //                            \      /
    //                             --8--
    uint32_t v0 = nimcp_graph_add_vertex(graph, 0, 0.0f, 0.0f, 0.0f, 0);
    uint32_t v1 = nimcp_graph_add_vertex(graph, 1, 1.0f, 0.0f, 0.0f, 0);
    uint32_t v2 = nimcp_graph_add_vertex(graph, 2, 2.0f, 0.0f, 0.0f, 0);

    ASSERT_TRUE(nimcp_graph_add_edge(graph, v0, v1, 5.0f));
    ASSERT_TRUE(nimcp_graph_add_edge(graph, v1, v2, 2.0f));
    ASSERT_TRUE(nimcp_graph_add_edge(graph, v0, v2, 8.0f));

    // Find shortest path from 0 to 2
    NimcpPath* path = nimcp_graph_shortest_path(graph, v0, v2);
    ASSERT_NE(path, nullptr);

    // Should be: 0 -> 1 -> 2 (total weight: 7) NOT 0 -> 2 (weight: 8)
    EXPECT_EQ(path->length, 3);
    EXPECT_FLOAT_EQ(path->total_weight, 7.0f);

    EXPECT_EQ(path->vertices[0], v0);
    EXPECT_EQ(path->vertices[1], v1);
    EXPECT_EQ(path->vertices[2], v2);

    free(path->vertices);
    free(path);
    nimcp_graph_destroy(graph);
}

/**
 * @test Dijkstra performance improvement O(V²) -> O((V+E) log V)
 *
 * Compare performance on larger graphs
 */
TEST_F(PerformanceOptimizationTest, DISABLED_Dijkstra_HeapBased_Performance)
{
    NimcpGraph* graph = nimcp_graph_create();
    ASSERT_NE(graph, nullptr);

    // Create a graph with 200 vertices (approaching NIMCP_MAX_VERTICES)
    const int V = 200;
    std::vector<uint32_t> vertices;

    for (int i = 0; i < V; i++) {
        uint32_t v = nimcp_graph_add_vertex(graph, i, (float)i, 0.0f, 0.0f, 0);
        vertices.push_back(v);
    }

    // Create edges: each vertex connects to next 5 vertices
    for (int i = 0; i < V - 1; i++) {
        for (int j = 1; j <= 5 && i + j < V; j++) {
            nimcp_graph_add_edge(graph, vertices[i], vertices[i + j], (float)j);
        }
    }

    // Measure shortest path computation
    long long time_us = measure_time_us([&]() {
        NimcpPath* path = nimcp_graph_shortest_path(graph, vertices[0], vertices[V - 1]);
        if (path) {
            free(path->vertices);
            free(path);
        }
    });

    // With heap optimization O((V+E) log V):
    //   V = 200, E ≈ 1000, log V ≈ 8
    //   Operations: ~8,000
    // Without heap O(V²):
    //   Operations: ~40,000

    // Should complete in < 10ms with heap optimization
    EXPECT_LT(time_us, 10000) << "Dijkstra too slow: " << time_us << " us";

    nimcp_graph_destroy(graph);
}

/**
 * @test Verify heap-based Dijkstra handles disconnected graphs
 */
TEST_F(PerformanceOptimizationTest, DISABLED_Dijkstra_HeapBased_DisconnectedGraph)
{
    NimcpGraph* graph = nimcp_graph_create();
    ASSERT_NE(graph, nullptr);

    // Create two disconnected components
    uint32_t v0 = nimcp_graph_add_vertex(graph, 0, 0.0f, 0.0f, 0.0f, 0);
    uint32_t v1 = nimcp_graph_add_vertex(graph, 1, 1.0f, 0.0f, 0.0f, 0);
    uint32_t v2 = nimcp_graph_add_vertex(graph, 2, 2.0f, 0.0f, 0.0f, 0);
    uint32_t v3 = nimcp_graph_add_vertex(graph, 3, 3.0f, 0.0f, 0.0f, 0);

    // Component 1: 0 -- 1
    ASSERT_TRUE(nimcp_graph_add_edge(graph, v0, v1, 1.0f));

    // Component 2: 2 -- 3
    ASSERT_TRUE(nimcp_graph_add_edge(graph, v2, v3, 1.0f));

    // No path from 0 to 3 (disconnected)
    NimcpPath* path = nimcp_graph_shortest_path(graph, v0, v3);
    EXPECT_EQ(path, nullptr);

    nimcp_graph_destroy(graph);
}

//==============================================================================
// PART 4: Integration and Benchmark Tests
//==============================================================================

/**
 * @test End-to-end performance comparison
 *
 * Measure real-world performance improvement on typical workloads
 */
TEST_F(PerformanceOptimizationTest, DISABLED_Integration_RealWorldPerformance)
{
    // This test would measure performance on realistic neural network
    // and graph operations to quantify the actual speedup achieved

    GTEST_SKIP() << "Integration test - run separately for benchmarking";
}
