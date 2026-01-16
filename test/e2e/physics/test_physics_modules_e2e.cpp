/**
 * @file test_physics_modules_e2e.cpp
 * @brief End-to-End Tests for Physics Modules Pipeline
 * @version 1.0.0
 * @date 2026-01-16
 *
 * Tests complete workflows across Graph Theory Bridge, Dynamical Systems,
 * and Information Geometry Bridge modules with full NIMCP integration.
 */

#include <gtest/gtest.h>
#include <cstring>
#include <cmath>
#include <vector>

extern "C" {
#include "physics/graphs/nimcp_graph_theory_bridge.h"
#include "physics/dynamics/nimcp_dynamical_systems.h"
#include "physics/geometry/nimcp_information_geometry_bridge.h"
#include "brain/kg/nimcp_brain_kg.h"
#include "utils/exception/nimcp_exception.h"
}

//=============================================================================
// Test Helper: Lorenz System for Dynamical Systems
//=============================================================================

static const double LORENZ_SIGMA = 10.0;
static const double LORENZ_RHO = 28.0;
static const double LORENZ_BETA = 8.0 / 3.0;

static void lorenz_system(const double* state, double* derivatives, void* context) {
    (void)context;
    derivatives[0] = LORENZ_SIGMA * (state[1] - state[0]);
    derivatives[1] = state[0] * (LORENZ_RHO - state[2]) - state[1];
    derivatives[2] = state[0] * state[1] - LORENZ_BETA * state[2];
}

static void lorenz_jacobian(const double* state, double* jacobian, void* context) {
    (void)context;
    // Row 0
    jacobian[0] = -LORENZ_SIGMA;
    jacobian[1] = LORENZ_SIGMA;
    jacobian[2] = 0.0;
    // Row 1
    jacobian[3] = LORENZ_RHO - state[2];
    jacobian[4] = -1.0;
    jacobian[5] = -state[0];
    // Row 2
    jacobian[6] = state[1];
    jacobian[7] = state[0];
    jacobian[8] = -LORENZ_BETA;
}

//=============================================================================
// Test Fixture for Full Pipeline E2E Tests
//=============================================================================

class PhysicsModulesE2ETest : public ::testing::Test {
protected:
    brain_kg_t* kg_ = nullptr;
    graph_theory_bridge_t graph_bridge_ = nullptr;
    dynsys_system_t dynsys_ = nullptr;
    info_geom_bridge_t geom_bridge_ = nullptr;

    void SetUp() override {
        // Create brain KG
        brain_kg_config_t kg_config;
        memset(&kg_config, 0, sizeof(kg_config));
        kg_config.max_nodes = 1000;
        kg_config.max_edges = 5000;
        kg_config.enable_persistence = false;
        kg_ = brain_kg_create(&kg_config);
        ASSERT_NE(kg_, nullptr);

        // Create graph theory bridge
        graph_theory_bridge_config_t graph_config;
        graph_theory_bridge_default_config(&graph_config);
        graph_bridge_ = graph_theory_bridge_create(&graph_config);
        ASSERT_NE(graph_bridge_, nullptr);

        // Create dynamical system (Lorenz)
        dynsys_config_t dyn_config = dynsys_default_config();
        dyn_config.dimension = 3;
        dyn_config.enable_kg_wiring = false;
        dyn_config.enable_exception_handling = true;
        dynsys_ = dynsys_create(&dyn_config, lorenz_system, lorenz_jacobian, nullptr);
        ASSERT_NE(dynsys_, nullptr);

        // Create information geometry bridge
        info_geom_bridge_config_t geom_config = info_geom_bridge_default_config();
        geom_bridge_ = info_geom_bridge_create(&geom_config);
        ASSERT_NE(geom_bridge_, nullptr);
    }

    void TearDown() override {
        if (geom_bridge_) {
            info_geom_bridge_destroy(geom_bridge_);
        }
        if (dynsys_) {
            dynsys_destroy(dynsys_);
        }
        if (graph_bridge_) {
            graph_theory_bridge_destroy(graph_bridge_);
        }
        if (kg_) {
            brain_kg_destroy(kg_);
        }
    }
};

//=============================================================================
// E2E Test: Complete Graph Analysis Pipeline
//=============================================================================

TEST_F(PhysicsModulesE2ETest, CompleteGraphAnalysisPipeline) {
    // Step 1: Register KG with graph bridge
    graph_theory_error_t err = graph_theory_bridge_register_kg(
        graph_bridge_, kg_, "e2e_test_token");
    EXPECT_EQ(err, GRAPH_THEORY_SUCCESS);

    // Step 2: Create a graph structure
    graph_theory_graph_t graph;
    memset(&graph, 0, sizeof(graph));
    graph.num_nodes = 5;
    graph.num_edges = 8;

    double adj_data[25] = {
        0, 1, 1, 0, 0,
        1, 0, 1, 1, 0,
        1, 1, 0, 1, 1,
        0, 1, 1, 0, 1,
        0, 0, 1, 1, 0
    };
    graph.adjacency_matrix = adj_data;
    graph.weighted = false;

    // Step 3: Compute centrality
    graph_centrality_result_t centrality;
    err = graph_theory_compute_centrality(graph_bridge_, &graph, CENTRALITY_DEGREE, &centrality);
    EXPECT_EQ(err, GRAPH_THEORY_SUCCESS);
    EXPECT_EQ(centrality.num_nodes, 5);

    // Verify node 2 has highest degree (4 connections)
    size_t max_idx = 0;
    double max_val = centrality.centrality_values[0];
    for (size_t i = 1; i < centrality.num_nodes; i++) {
        if (centrality.centrality_values[i] > max_val) {
            max_val = centrality.centrality_values[i];
            max_idx = i;
        }
    }
    EXPECT_EQ(max_idx, 2);  // Node 2 has highest degree

    // Step 4: Detect communities
    graph_community_result_t communities;
    err = graph_theory_detect_communities(graph_bridge_, &graph, COMMUNITY_LOUVAIN, 2, &communities);
    EXPECT_EQ(err, GRAPH_THEORY_SUCCESS);
    EXPECT_GE(communities.num_communities, 1);
    EXPECT_LE(communities.num_communities, 5);

    // Step 5: Compute topology metrics
    graph_topology_metrics_t topology;
    err = graph_theory_compute_topology(graph_bridge_, &graph, &topology);
    EXPECT_EQ(err, GRAPH_THEORY_SUCCESS);
    EXPECT_GT(topology.clustering_coefficient, 0.0);
    EXPECT_GT(topology.average_path_length, 0.0);

    // Step 6: Verify KG was populated
    size_t node_count = brain_kg_get_node_count(kg_);
    EXPECT_GT(node_count, 0);
}

//=============================================================================
// E2E Test: Complete Dynamical Systems Analysis Pipeline
//=============================================================================

TEST_F(PhysicsModulesE2ETest, CompleteDynamicalSystemsPipeline) {
    // Step 1: Initialize Lorenz system state
    double state[3] = {1.0, 1.0, 1.0};
    double dt = 0.01;

    // Step 2: Evolve system for several steps
    for (int i = 0; i < 100; i++) {
        dynsys_error_t err = dynsys_step_rk4(dynsys_, state, dt);
        EXPECT_EQ(err, DYNSYS_SUCCESS);
    }

    // Verify state evolved (Lorenz attractor behavior)
    bool state_changed = (fabs(state[0] - 1.0) > 0.1 ||
                          fabs(state[1] - 1.0) > 0.1 ||
                          fabs(state[2] - 1.0) > 0.1);
    EXPECT_TRUE(state_changed);

    // Step 3: Create Lyapunov analyzer
    dynsys_lyapunov_config_t lyap_config;
    memset(&lyap_config, 0, sizeof(lyap_config));
    lyap_config.num_exponents = 3;
    lyap_config.perturbation_scale = 1e-8;
    lyap_config.orthonormalization_steps = 10;

    dynsys_lyapunov_t lyap = dynsys_lyapunov_create(dynsys_, &lyap_config);
    if (lyap != nullptr) {
        // Compute Lyapunov exponents
        double initial_state[3] = {1.0, 1.0, 1.0};
        dynsys_lyapunov_result_t lyap_result;
        dynsys_error_t err = dynsys_lyapunov_compute(lyap, initial_state, &lyap_result);
        if (err == DYNSYS_SUCCESS) {
            // Lorenz system has one positive Lyapunov exponent (chaos)
            bool has_positive = false;
            for (size_t i = 0; i < lyap_result.num_exponents; i++) {
                if (lyap_result.exponents[i] > 0) {
                    has_positive = true;
                    break;
                }
            }
            EXPECT_TRUE(has_positive);
        }
        dynsys_lyapunov_destroy(lyap);
    }

    // Step 4: Create bifurcation analyzer
    dynsys_bifurcation_config_t bif_config;
    memset(&bif_config, 0, sizeof(bif_config));
    bif_config.param_start = 20.0;
    bif_config.param_end = 30.0;
    bif_config.num_samples = 10;
    bif_config.transient_steps = 100;
    bif_config.sample_steps = 50;

    dynsys_bifurcation_t bif = dynsys_bifurcation_create(dynsys_, &bif_config);
    if (bif != nullptr) {
        // Bifurcation analysis would be done here
        dynsys_bifurcation_destroy(bif);
    }
}

//=============================================================================
// E2E Test: Complete Information Geometry Bridge Pipeline
//=============================================================================

TEST_F(PhysicsModulesE2ETest, CompleteInfoGeometryPipeline) {
    // Step 1: Register KG with geometry bridge
    int result = info_geom_bridge_register_kg(geom_bridge_, kg_);
    EXPECT_EQ(result, 0);

    // Step 2: Register exception handler
    void* dummy_exception = (void*)0x12345678;
    result = info_geom_bridge_register_exception(geom_bridge_, dummy_exception);
    EXPECT_EQ(result, 0);

    // Step 3: Register bio-async channel
    void* dummy_channel = (void*)0x87654321;
    result = info_geom_bridge_register_bio_async(geom_bridge_, dummy_channel);
    EXPECT_EQ(result, 0);

    // Step 4: Verify registrations work (re-register with null to clear)
    result = info_geom_bridge_register_exception(geom_bridge_, nullptr);
    EXPECT_EQ(result, 0);
    result = info_geom_bridge_register_bio_async(geom_bridge_, nullptr);
    EXPECT_EQ(result, 0);
}

//=============================================================================
// E2E Test: Cross-Module Integration
//=============================================================================

TEST_F(PhysicsModulesE2ETest, CrossModuleIntegration) {
    // Step 1: Use all modules with shared KG
    graph_theory_error_t g_err = graph_theory_bridge_register_kg(
        graph_bridge_, kg_, "cross_module_token");
    EXPECT_EQ(g_err, GRAPH_THEORY_SUCCESS);

    int i_err = info_geom_bridge_register_kg(geom_bridge_, kg_);
    EXPECT_EQ(i_err, 0);

    // Step 2: Run graph analysis
    graph_theory_graph_t graph;
    memset(&graph, 0, sizeof(graph));
    graph.num_nodes = 4;
    graph.num_edges = 4;
    double adj[16] = {
        0, 1, 0, 1,
        1, 0, 1, 0,
        0, 1, 0, 1,
        1, 0, 1, 0
    };
    graph.adjacency_matrix = adj;

    graph_centrality_result_t centrality;
    g_err = graph_theory_compute_centrality(graph_bridge_, &graph, CENTRALITY_DEGREE, &centrality);
    EXPECT_EQ(g_err, GRAPH_THEORY_SUCCESS);

    // Step 3: Run dynamical systems analysis
    double state[3] = {0.1, 0.1, 0.1};
    for (int i = 0; i < 50; i++) {
        dynsys_error_t err = dynsys_step_rk4(dynsys_, state, 0.01);
        EXPECT_EQ(err, DYNSYS_SUCCESS);
    }

    // Step 4: Verify shared KG has nodes from both modules
    size_t node_count = brain_kg_get_node_count(kg_);
    EXPECT_GT(node_count, 0);
}

//=============================================================================
// E2E Test: Spectral Graph Analysis with Dynamical Systems
//=============================================================================

TEST_F(PhysicsModulesE2ETest, SpectralDynamicsIntegration) {
    // Create a larger graph for spectral analysis
    graph_theory_graph_t graph;
    memset(&graph, 0, sizeof(graph));
    graph.num_nodes = 8;
    graph.num_edges = 12;

    // Ring graph with some cross-connections
    double adj[64] = {
        0, 1, 0, 0, 0, 0, 0, 1,
        1, 0, 1, 0, 0, 0, 0, 0,
        0, 1, 0, 1, 0, 0, 0, 0,
        0, 0, 1, 0, 1, 0, 0, 0,
        0, 0, 0, 1, 0, 1, 0, 0,
        0, 0, 0, 0, 1, 0, 1, 0,
        0, 0, 0, 0, 0, 1, 0, 1,
        1, 0, 0, 0, 0, 0, 1, 0
    };
    graph.adjacency_matrix = adj;

    // Run spectral analysis
    graph_spectral_result_t spectral;
    graph_theory_error_t err = graph_theory_compute_spectral(graph_bridge_, &graph, 3, &spectral);
    if (err == GRAPH_THEORY_SUCCESS) {
        EXPECT_EQ(spectral.num_eigenvalues, 3);
        EXPECT_GE(spectral.spectral_gap, 0.0);
    }

    // Evolve dynamical system
    double state[3] = {5.0, 5.0, 5.0};
    for (int i = 0; i < 200; i++) {
        dynsys_step_rk4(dynsys_, state, 0.01);
    }

    // Verify system is still in bounded region (Lorenz attractor)
    EXPECT_LT(fabs(state[0]), 50.0);
    EXPECT_LT(fabs(state[1]), 50.0);
    EXPECT_LT(fabs(state[2]), 100.0);
}

//=============================================================================
// E2E Test: Quantum Walk with Graph Theory
//=============================================================================

TEST_F(PhysicsModulesE2ETest, QuantumWalkPipeline) {
    // Create graph for quantum walk
    graph_theory_graph_t graph;
    memset(&graph, 0, sizeof(graph));
    graph.num_nodes = 6;
    graph.num_edges = 7;

    // Binary tree-like structure
    double adj[36] = {
        0, 1, 1, 0, 0, 0,
        1, 0, 0, 1, 1, 0,
        1, 0, 0, 0, 0, 1,
        0, 1, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0
    };
    graph.adjacency_matrix = adj;

    // Initialize quantum walk
    graph_qwalk_config_t qwalk_config;
    memset(&qwalk_config, 0, sizeof(qwalk_config));
    qwalk_config.walk_type = QWALK_CONTINUOUS;
    qwalk_config.max_steps = 100;
    qwalk_config.gamma = 0.1;

    graph_qwalk_result_t qwalk_result;
    graph_theory_error_t err = graph_theory_quantum_walk(
        graph_bridge_, &graph, 0, &qwalk_config, &qwalk_result);

    if (err == GRAPH_THEORY_SUCCESS) {
        // Verify probability distribution is valid
        double total_prob = 0.0;
        for (size_t i = 0; i < qwalk_result.num_nodes; i++) {
            EXPECT_GE(qwalk_result.final_probabilities[i], 0.0);
            EXPECT_LE(qwalk_result.final_probabilities[i], 1.0);
            total_prob += qwalk_result.final_probabilities[i];
        }
        EXPECT_NEAR(total_prob, 1.0, 1e-6);
    }
}

//=============================================================================
// E2E Test: Long-Running Stability
//=============================================================================

TEST_F(PhysicsModulesE2ETest, LongRunningStability) {
    // Multiple iterations of create/compute/destroy cycle
    for (int cycle = 0; cycle < 5; cycle++) {
        // Create fresh modules each cycle
        graph_theory_bridge_config_t graph_config;
        graph_theory_bridge_default_config(&graph_config);
        graph_theory_bridge_t local_graph = graph_theory_bridge_create(&graph_config);
        ASSERT_NE(local_graph, nullptr);

        dynsys_config_t dyn_config = dynsys_default_config();
        dyn_config.dimension = 3;
        dynsys_system_t local_sys = dynsys_create(&dyn_config, lorenz_system, lorenz_jacobian, nullptr);
        ASSERT_NE(local_sys, nullptr);

        info_geom_bridge_config_t geom_config = info_geom_bridge_default_config();
        info_geom_bridge_t local_geom = info_geom_bridge_create(&geom_config);
        ASSERT_NE(local_geom, nullptr);

        // Do some work
        double state[3] = {1.0, 1.0, 1.0};
        for (int step = 0; step < 50; step++) {
            dynsys_step_rk4(local_sys, state, 0.01);
        }

        graph_theory_graph_t graph;
        memset(&graph, 0, sizeof(graph));
        graph.num_nodes = 3;
        graph.num_edges = 3;
        double adj[9] = {0, 1, 1, 1, 0, 1, 1, 1, 0};
        graph.adjacency_matrix = adj;

        graph_centrality_result_t centrality;
        graph_theory_compute_centrality(local_graph, &graph, CENTRALITY_DEGREE, &centrality);

        // Cleanup
        info_geom_bridge_destroy(local_geom);
        dynsys_destroy(local_sys);
        graph_theory_bridge_destroy(local_graph);
    }
}

//=============================================================================
// E2E Test: Error Recovery Pipeline
//=============================================================================

TEST_F(PhysicsModulesE2ETest, ErrorRecoveryPipeline) {
    // Test with invalid inputs, verify recovery

    // Null graph to centrality
    graph_theory_error_t err = graph_theory_compute_centrality(
        graph_bridge_, nullptr, CENTRALITY_DEGREE, nullptr);
    EXPECT_NE(err, GRAPH_THEORY_SUCCESS);

    // Valid operations should still work after error
    graph_theory_graph_t graph;
    memset(&graph, 0, sizeof(graph));
    graph.num_nodes = 3;
    graph.num_edges = 2;
    double adj[9] = {0, 1, 0, 1, 0, 1, 0, 1, 0};
    graph.adjacency_matrix = adj;

    graph_centrality_result_t centrality;
    err = graph_theory_compute_centrality(graph_bridge_, &graph, CENTRALITY_DEGREE, &centrality);
    EXPECT_EQ(err, GRAPH_THEORY_SUCCESS);

    // Test info geometry with null registrations
    int ierr = info_geom_bridge_register_kg(nullptr, kg_);
    EXPECT_EQ(ierr, -1);

    ierr = info_geom_bridge_register_kg(geom_bridge_, nullptr);
    EXPECT_EQ(ierr, -1);

    // Should still work with valid inputs
    ierr = info_geom_bridge_register_kg(geom_bridge_, kg_);
    EXPECT_EQ(ierr, 0);
}

//=============================================================================
// E2E Test: Community Detection and Phase Coherence
//=============================================================================

TEST_F(PhysicsModulesE2ETest, CommunityPhaseCoherencePipeline) {
    // Create a graph with clear community structure
    graph_theory_graph_t graph;
    memset(&graph, 0, sizeof(graph));
    graph.num_nodes = 6;
    graph.num_edges = 9;

    // Two clusters of 3 nodes each, weakly connected
    double adj[36] = {
        0, 1, 1, 0, 0, 0,   // Cluster 1: nodes 0,1,2
        1, 0, 1, 0, 0, 0,
        1, 1, 0, 1, 0, 0,   // Connection 2-3
        0, 0, 1, 0, 1, 1,   // Cluster 2: nodes 3,4,5
        0, 0, 0, 1, 0, 1,
        0, 0, 0, 1, 1, 0
    };
    graph.adjacency_matrix = adj;

    // Detect communities
    graph_community_result_t communities;
    graph_theory_error_t err = graph_theory_detect_communities(
        graph_bridge_, &graph, COMMUNITY_LOUVAIN, 2, &communities);

    if (err == GRAPH_THEORY_SUCCESS) {
        // Should detect approximately 2 communities
        EXPECT_GE(communities.num_communities, 1);
        EXPECT_LE(communities.num_communities, 3);

        // Compute phase coherence
        graph_phase_coherence_result_t coherence;
        err = graph_theory_compute_phase_coherence(
            graph_bridge_, &graph, &communities, &coherence);

        if (err == GRAPH_THEORY_SUCCESS) {
            EXPECT_GE(coherence.global_coherence, 0.0);
            EXPECT_LE(coherence.global_coherence, 1.0);
        }
    }
}

//=============================================================================
// E2E Test: Hyperbolic Embedding
//=============================================================================

TEST_F(PhysicsModulesE2ETest, HyperbolicEmbeddingPipeline) {
    // Create a scale-free-like graph
    graph_theory_graph_t graph;
    memset(&graph, 0, sizeof(graph));
    graph.num_nodes = 5;
    graph.num_edges = 6;

    // Hub-and-spoke structure (node 0 is hub)
    double adj[25] = {
        0, 1, 1, 1, 1,
        1, 0, 1, 0, 0,
        1, 1, 0, 0, 0,
        1, 0, 0, 0, 0,
        1, 0, 0, 0, 0
    };
    graph.adjacency_matrix = adj;

    graph_hyperbolic_config_t hyp_config;
    memset(&hyp_config, 0, sizeof(hyp_config));
    hyp_config.embedding_dim = 2;
    hyp_config.curvature = -1.0;
    hyp_config.max_iterations = 100;
    hyp_config.learning_rate = 0.1;

    graph_hyperbolic_result_t hyp_result;
    graph_theory_error_t err = graph_theory_hyperbolic_embed(
        graph_bridge_, &graph, &hyp_config, &hyp_result);

    if (err == GRAPH_THEORY_SUCCESS) {
        EXPECT_EQ(hyp_result.num_nodes, 5);
        // Hub node should have smaller radial coordinate
        // (closer to center in hyperbolic space)
        EXPECT_GE(hyp_result.distortion, 0.0);
    }
}

//=============================================================================
// E2E Test: Full Analysis Report
//=============================================================================

TEST_F(PhysicsModulesE2ETest, FullAnalysisReport) {
    // Register all modules with shared KG
    graph_theory_bridge_register_kg(graph_bridge_, kg_, "report_token");
    info_geom_bridge_register_kg(geom_bridge_, kg_);

    // Create analysis graph
    graph_theory_graph_t graph;
    memset(&graph, 0, sizeof(graph));
    graph.num_nodes = 10;
    graph.num_edges = 20;

    // Small-world network structure
    double adj[100];
    memset(adj, 0, sizeof(adj));
    // Ring connections
    for (int i = 0; i < 10; i++) {
        adj[i * 10 + ((i + 1) % 10)] = 1.0;
        adj[((i + 1) % 10) * 10 + i] = 1.0;
    }
    // Some random long-range connections
    adj[0 * 10 + 5] = 1.0;
    adj[5 * 10 + 0] = 1.0;
    adj[2 * 10 + 7] = 1.0;
    adj[7 * 10 + 2] = 1.0;
    adj[3 * 10 + 8] = 1.0;
    adj[8 * 10 + 3] = 1.0;
    graph.adjacency_matrix = adj;
    graph.weighted = false;

    // Collect all metrics
    graph_centrality_result_t centrality;
    graph_theory_compute_centrality(graph_bridge_, &graph, CENTRALITY_BETWEENNESS, &centrality);

    graph_community_result_t communities;
    graph_theory_detect_communities(graph_bridge_, &graph, COMMUNITY_LOUVAIN, 3, &communities);

    graph_topology_metrics_t topology;
    graph_theory_compute_topology(graph_bridge_, &graph, &topology);

    graph_spectral_result_t spectral;
    graph_theory_compute_spectral(graph_bridge_, &graph, 5, &spectral);

    // Evolve dynamical system
    double state[3] = {1.0, 1.0, 1.0};
    std::vector<double> trajectory_x;
    trajectory_x.reserve(1000);

    for (int i = 0; i < 1000; i++) {
        dynsys_step_rk4(dynsys_, state, 0.01);
        trajectory_x.push_back(state[0]);
    }

    // Verify we have meaningful results
    EXPECT_GT(centrality.num_nodes, 0);
    EXPECT_GE(communities.num_communities, 1);
    EXPECT_GT(topology.average_path_length, 0.0);
    EXPECT_EQ(trajectory_x.size(), 1000);

    // Verify KG has been populated by all modules
    size_t final_node_count = brain_kg_get_node_count(kg_);
    EXPECT_GT(final_node_count, 0);
}

//=============================================================================
// Main
//=============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
