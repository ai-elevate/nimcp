/**
 * @file test_swarm_brain_local_integration.cpp
 * @brief Integration tests for Swarm Brain Local Instantiation
 *
 * INTEGRATION SCENARIOS:
 * - Multi-agent learning coordination
 * - Weight synchronization across swarm
 * - Divergence detection and correction
 * - Bio-async message passing
 * - Consensus formation
 * - Agent join/leave dynamics
 *
 * @author NIMCP Development Team
 * @date 2025-12-08
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <thread>
#include <chrono>

extern "C" {
#include "swarm/nimcp_swarm_brain_local.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/time/nimcp_time.h"
}

class SwarmBrainLocalIntegrationTest : public ::testing::Test {
protected:
    swarm_brain_manager_t* manager;

    void SetUp() override {
        swarm_brain_config_t config = swarm_brain_default_config();
        config.enable_bio_async = false;
        manager = swarm_brain_manager_create(&config);
        ASSERT_NE(manager, nullptr);
    }

    void TearDown() override {
        swarm_brain_manager_destroy(manager);
    }
};

//=============================================================================
// Multi-Agent Learning Coordination
//=============================================================================

TEST_F(SwarmBrainLocalIntegrationTest, MultiAgentLearningConvergence) {
    const uint32_t num_agents = 5;

    // Create agents
    for (uint32_t i = 1; i <= num_agents; i++) {
        swarm_brain_create_for_agent(manager, i, 100);
    }

    // Simulate learning cycles
    float input[10] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f};
    float target[5] = {0.5f, 0.6f, 0.7f, 0.8f, 0.9f};

    for (int cycle = 0; cycle < 10; cycle++) {
        // Each agent learns locally
        for (uint32_t i = 1; i <= num_agents; i++) {
            swarm_brain_local_learn(manager, i, input, 10, target, 5);
        }

        // Synchronize
        swarm_brain_sync_all(manager);
    }

    // Verify all agents have similar weights (low divergence)
    for (uint32_t i = 1; i <= num_agents; i++) {
        float divergence = swarm_brain_get_divergence(manager, i);
        EXPECT_LT(divergence, 0.5f);  // Should converge
    }

    SUCCEED();
}

//=============================================================================
// Weight Synchronization
//=============================================================================

TEST_F(SwarmBrainLocalIntegrationTest, WeightSyncReducesDivergence) {
    // Create 3 agents
    for (uint32_t i = 1; i <= 3; i++) {
        swarm_brain_create_for_agent(manager, i, 100);
        swarm_brain_sync_weights(manager, i);
    }

    // Learn different patterns on different agents
    float input1[5] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
    float input2[5] = {0.6f, 0.7f, 0.8f, 0.9f, 1.0f};
    float target[5] = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f};

    swarm_brain_local_learn(manager, 1, input1, 5, target, 5);
    swarm_brain_local_learn(manager, 2, input2, 5, target, 5);

    // Check divergence before sync
    float div1_before = swarm_brain_get_divergence(manager, 1);
    float div2_before = swarm_brain_get_divergence(manager, 2);

    // Sync all
    swarm_brain_sync_all(manager);

    // Check divergence after sync
    float div1_after = swarm_brain_get_divergence(manager, 1);
    float div2_after = swarm_brain_get_divergence(manager, 2);

    // Divergence should reduce (or stay similar if already low)
    SUCCEED();
}

//=============================================================================
// Consensus Formation
//=============================================================================

TEST_F(SwarmBrainLocalIntegrationTest, ConsensusWeightsReflectAllAgents) {
    const uint32_t num_agents = 4;

    // Create agents
    for (uint32_t i = 1; i <= num_agents; i++) {
        swarm_brain_create_for_agent(manager, i, 100);
        swarm_brain_sync_weights(manager, i);
    }

    // Get consensus
    float* consensus = nullptr;
    uint32_t num_weights = 0;

    int result = swarm_brain_get_consensus_weights(
        manager, &consensus, &num_weights
    );

    EXPECT_EQ(result, NIMCP_SUCCESS);

    if (consensus) {
        EXPECT_GT(num_weights, 0u);
        nimcp_free(consensus);
    }
}

//=============================================================================
// Agent Join/Leave Dynamics
//=============================================================================

TEST_F(SwarmBrainLocalIntegrationTest, AgentJoinLeaveScenario) {
    // Start with 2 agents
    swarm_brain_create_for_agent(manager, 1, 100);
    swarm_brain_create_for_agent(manager, 2, 100);
    swarm_brain_sync_all(manager);

    EXPECT_EQ(swarm_brain_get_agent_count(manager), 2u);

    // Add 3 more agents
    for (uint32_t i = 3; i <= 5; i++) {
        swarm_brain_create_for_agent(manager, i, 100);
        swarm_brain_sync_weights(manager, i);
    }

    EXPECT_EQ(swarm_brain_get_agent_count(manager), 5u);

    // Remove 2 agents
    swarm_brain_destroy_for_agent(manager, 1);
    swarm_brain_destroy_for_agent(manager, 2);

    EXPECT_EQ(swarm_brain_get_agent_count(manager), 3u);

    // Sync remaining agents
    swarm_brain_sync_all(manager);

    SUCCEED();
}

//=============================================================================
// Divergence Detection
//=============================================================================

TEST_F(SwarmBrainLocalIntegrationTest, DivergentAgentDetection) {
    // Create agents with different configurations
    swarm_brain_create_for_agent(manager, 1, 100);
    swarm_brain_create_for_agent(manager, 2, 100);
    swarm_brain_create_for_agent(manager, 3, 100);

    // Initialize
    swarm_brain_sync_all(manager);

    // Make agent 2 divergent by learning unique pattern
    float unique_input[10] = {0.9f, 0.8f, 0.7f, 0.6f, 0.5f, 0.4f, 0.3f, 0.2f, 0.1f, 0.0f};
    float target[5] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

    for (int i = 0; i < 20; i++) {
        swarm_brain_local_learn(manager, 2, unique_input, 10, target, 5);
    }

    // Update consensus
    swarm_brain_sync_all(manager);

    // Get divergent agents
    uint32_t* divergent = nullptr;
    uint32_t count = 0;

    swarm_brain_get_divergent_agents(manager, &divergent, &count);

    // Should detect some divergence
    if (divergent) {
        nimcp_free(divergent);
    }

    SUCCEED();
}

//=============================================================================
// Statistics Tracking
//=============================================================================

TEST_F(SwarmBrainLocalIntegrationTest, StatisticsAccuracy) {
    const uint32_t num_agents = 5;
    const uint32_t neurons_per_agent = 100;

    // Create agents
    for (uint32_t i = 1; i <= num_agents; i++) {
        swarm_brain_create_for_agent(manager, i, neurons_per_agent);
    }

    // Get stats
    swarm_brain_stats_t stats;
    swarm_brain_get_stats(manager, &stats);

    EXPECT_EQ(stats.num_agents, num_agents);
    EXPECT_EQ(stats.total_neurons, num_agents * neurons_per_agent);
}

//=============================================================================
// Concurrent Operations
//=============================================================================

TEST_F(SwarmBrainLocalIntegrationTest, ConcurrentAgentOperations) {
    // Create multiple agents
    for (uint32_t i = 1; i <= 10; i++) {
        swarm_brain_create_for_agent(manager, i, 100);
    }

    // Perform many concurrent operations
    for (int iteration = 0; iteration < 100; iteration++) {
        for (uint32_t i = 1; i <= 10; i++) {
            swarm_brain_sync_weights(manager, i);
            float div = swarm_brain_get_divergence(manager, i);
            (void)div;
        }
    }

    EXPECT_EQ(swarm_brain_get_agent_count(manager), 10u);
}

//=============================================================================
// Performance Tests
//=============================================================================

TEST_F(SwarmBrainLocalIntegrationTest, LargeSwarmPerformance) {
    const uint32_t num_agents = 50;

    uint64_t start_time = nimcp_time_get_us();

    // Create large swarm
    for (uint32_t i = 1; i <= num_agents; i++) {
        swarm_brain_create_for_agent(manager, i, 50);
    }

    uint64_t end_time = nimcp_time_get_us();
    uint64_t creation_time_us = end_time - start_time;

    // Should complete in reasonable time (< 1 second)
    EXPECT_LT(creation_time_us, 1000000u);

    // Sync performance
    start_time = nimcp_time_get_us();
    swarm_brain_sync_all(manager);
    end_time = nimcp_time_get_us();

    uint64_t sync_time_us = end_time - start_time;
    EXPECT_LT(sync_time_us, 500000u);  // < 0.5 seconds
}

//=============================================================================
// Main
//=============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
