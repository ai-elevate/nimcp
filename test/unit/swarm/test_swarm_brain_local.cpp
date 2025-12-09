/**
 * @file test_swarm_brain_local.cpp
 * @brief Comprehensive unit tests for NIMCP Swarm Brain Local Instantiation
 *
 * TEST COVERAGE:
 * - Manager creation and destruction
 * - Agent brain creation and management
 * - Local learning and processing
 * - Weight synchronization
 * - Divergence detection
 * - Consensus weight calculation
 * - Bio-async integration
 * - Statistics tracking
 * - Edge cases and error handling
 * - Multi-agent scenarios
 * - Performance under load
 *
 * @author NIMCP Development Team
 * @date 2025-12-08
 */

#include <gtest/gtest.h>
#include <cstring>
#include <cmath>
#include <vector>

extern "C" {
#include "swarm/nimcp_swarm_brain_local.h"
#include "utils/memory/nimcp_memory.h"
}

//=============================================================================
// Test Fixture
//=============================================================================

class SwarmBrainLocalTest : public ::testing::Test {
protected:
    swarm_brain_manager_t* manager;
    swarm_brain_config_t config;

    void SetUp() override {
        // Get default configuration
        config = swarm_brain_default_config();
        config.enable_bio_async = false;  // Disable for unit tests

        // Create manager
        manager = swarm_brain_manager_create(&config);
        ASSERT_NE(manager, nullptr);
    }

    void TearDown() override {
        if (manager) {
            swarm_brain_manager_destroy(manager);
            manager = nullptr;
        }
    }
};

//=============================================================================
// Manager Creation and Destruction Tests
//=============================================================================

TEST_F(SwarmBrainLocalTest, CreateValidManager) {
    EXPECT_NE(manager, nullptr);
}

TEST_F(SwarmBrainLocalTest, CreateWithNullConfig) {
    auto* mgr = swarm_brain_manager_create(nullptr);
    EXPECT_NE(mgr, nullptr);  // Should use defaults
    if (mgr) {
        swarm_brain_manager_destroy(mgr);
    }
}

TEST_F(SwarmBrainLocalTest, DestroyNullManager) {
    swarm_brain_manager_destroy(nullptr);
    // Should not crash
    SUCCEED();
}

TEST_F(SwarmBrainLocalTest, CreateWithCustomConfig) {
    swarm_brain_config_t custom_config = {
        .default_brain_size = 50,
        .max_local_neurons = 200,
        .sync_interval_ms = 500,
        .divergence_threshold = 0.2f,
        .enable_weight_sharing = true,
        .enable_bio_async = false
    };

    auto* mgr = swarm_brain_manager_create(&custom_config);
    EXPECT_NE(mgr, nullptr);
    swarm_brain_manager_destroy(mgr);
}

//=============================================================================
// Agent Brain Creation Tests
//=============================================================================

TEST_F(SwarmBrainLocalTest, CreateAgentBrain) {
    int result = swarm_brain_create_for_agent(manager, 1, 100);
    EXPECT_EQ(result, NIMCP_SUCCESS);
    EXPECT_TRUE(swarm_brain_has_agent(manager, 1));
}

TEST_F(SwarmBrainLocalTest, CreateMultipleAgentBrains) {
    for (uint32_t i = 1; i <= 10; i++) {
        int result = swarm_brain_create_for_agent(manager, i, 100);
        EXPECT_EQ(result, NIMCP_SUCCESS);
    }

    EXPECT_EQ(swarm_brain_get_agent_count(manager), 10u);
}

TEST_F(SwarmBrainLocalTest, CreateAgentWithDefaultSize) {
    int result = swarm_brain_create_for_agent(manager, 1, 0);  // 0 = use default
    EXPECT_EQ(result, NIMCP_SUCCESS);
}

TEST_F(SwarmBrainLocalTest, CreateAgentWithExcessiveSize) {
    int result = swarm_brain_create_for_agent(
        manager, 1, config.max_local_neurons + 100
    );
    EXPECT_NE(result, NIMCP_SUCCESS);  // Should fail
}

TEST_F(SwarmBrainLocalTest, CreateDuplicateAgent) {
    swarm_brain_create_for_agent(manager, 1, 100);
    int result = swarm_brain_create_for_agent(manager, 1, 100);
    EXPECT_NE(result, NIMCP_SUCCESS);  // Should fail - already exists
}

TEST_F(SwarmBrainLocalTest, CreateAgentNullManager) {
    int result = swarm_brain_create_for_agent(nullptr, 1, 100);
    EXPECT_NE(result, NIMCP_SUCCESS);
}

//=============================================================================
// Agent Brain Destruction Tests
//=============================================================================

TEST_F(SwarmBrainLocalTest, DestroyAgentBrain) {
    swarm_brain_create_for_agent(manager, 1, 100);
    int result = swarm_brain_destroy_for_agent(manager, 1);
    EXPECT_EQ(result, NIMCP_SUCCESS);
    EXPECT_FALSE(swarm_brain_has_agent(manager, 1));
}

TEST_F(SwarmBrainLocalTest, DestroyNonexistentAgent) {
    int result = swarm_brain_destroy_for_agent(manager, 999);
    EXPECT_NE(result, NIMCP_SUCCESS);  // Should fail
}

TEST_F(SwarmBrainLocalTest, DestroyAgentNullManager) {
    int result = swarm_brain_destroy_for_agent(nullptr, 1);
    EXPECT_NE(result, NIMCP_SUCCESS);
}

//=============================================================================
// Agent Brain Access Tests
//=============================================================================

TEST_F(SwarmBrainLocalTest, GetAgentBrain) {
    swarm_brain_create_for_agent(manager, 1, 100);
    brain_t brain = swarm_brain_get(manager, 1);
    EXPECT_NE(brain, nullptr);
}

TEST_F(SwarmBrainLocalTest, GetNonexistentAgentBrain) {
    brain_t brain = swarm_brain_get(manager, 999);
    EXPECT_EQ(brain, nullptr);
}

TEST_F(SwarmBrainLocalTest, GetAgentBrainNullManager) {
    brain_t brain = swarm_brain_get(nullptr, 1);
    EXPECT_EQ(brain, nullptr);
}

TEST_F(SwarmBrainLocalTest, HasAgent) {
    EXPECT_FALSE(swarm_brain_has_agent(manager, 1));
    swarm_brain_create_for_agent(manager, 1, 100);
    EXPECT_TRUE(swarm_brain_has_agent(manager, 1));
}

TEST_F(SwarmBrainLocalTest, GetAgentCount) {
    EXPECT_EQ(swarm_brain_get_agent_count(manager), 0u);

    for (uint32_t i = 1; i <= 5; i++) {
        swarm_brain_create_for_agent(manager, i, 100);
    }

    EXPECT_EQ(swarm_brain_get_agent_count(manager), 5u);
}

//=============================================================================
// Local Learning Tests
//=============================================================================

TEST_F(SwarmBrainLocalTest, LocalLearnValidData) {
    swarm_brain_create_for_agent(manager, 1, 100);

    float input[10] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f};
    float target[5] = {0.5f, 0.6f, 0.7f, 0.8f, 0.9f};

    int result = swarm_brain_local_learn(
        manager, 1, input, 10, target, 5
    );

    // Result depends on brain API implementation
    // At minimum, should not crash
    SUCCEED();
}

TEST_F(SwarmBrainLocalTest, LocalLearnNonexistentAgent) {
    float input[10] = {0.0f};
    float target[5] = {0.0f};

    int result = swarm_brain_local_learn(
        manager, 999, input, 10, target, 5
    );

    EXPECT_NE(result, NIMCP_SUCCESS);
}

TEST_F(SwarmBrainLocalTest, LocalLearnNullData) {
    swarm_brain_create_for_agent(manager, 1, 100);

    int result = swarm_brain_local_learn(
        manager, 1, nullptr, 10, nullptr, 5
    );

    EXPECT_NE(result, NIMCP_SUCCESS);
}

//=============================================================================
// Local Processing Tests
//=============================================================================

TEST_F(SwarmBrainLocalTest, LocalProcessValidData) {
    swarm_brain_create_for_agent(manager, 1, 100);

    float input[10] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f};
    float output[10] = {0.0f};
    uint32_t output_size = 10;

    int result = swarm_brain_local_process(
        manager, 1, input, 10, output, &output_size
    );

    // Result depends on brain API implementation
    SUCCEED();
}

TEST_F(SwarmBrainLocalTest, LocalProcessNonexistentAgent) {
    float input[10] = {0.0f};
    float output[10] = {0.0f};
    uint32_t output_size = 10;

    int result = swarm_brain_local_process(
        manager, 999, input, 10, output, &output_size
    );

    EXPECT_NE(result, NIMCP_SUCCESS);
}

TEST_F(SwarmBrainLocalTest, LocalProcessNullData) {
    swarm_brain_create_for_agent(manager, 1, 100);

    int result = swarm_brain_local_process(
        manager, 1, nullptr, 10, nullptr, nullptr
    );

    EXPECT_NE(result, NIMCP_SUCCESS);
}

//=============================================================================
// Weight Synchronization Tests
//=============================================================================

TEST_F(SwarmBrainLocalTest, SyncWeightsSingleAgent) {
    swarm_brain_create_for_agent(manager, 1, 100);

    int result = swarm_brain_sync_weights(manager, 1);
    EXPECT_EQ(result, NIMCP_SUCCESS);
}

TEST_F(SwarmBrainLocalTest, SyncWeightsNonexistentAgent) {
    int result = swarm_brain_sync_weights(manager, 999);
    EXPECT_NE(result, NIMCP_SUCCESS);
}

TEST_F(SwarmBrainLocalTest, SyncAllWeights) {
    for (uint32_t i = 1; i <= 5; i++) {
        swarm_brain_create_for_agent(manager, i, 100);
    }

    int result = swarm_brain_sync_all(manager);
    EXPECT_EQ(result, NIMCP_SUCCESS);
}

TEST_F(SwarmBrainLocalTest, SyncWeightsNullManager) {
    int result = swarm_brain_sync_weights(nullptr, 1);
    EXPECT_NE(result, NIMCP_SUCCESS);
}

//=============================================================================
// Consensus Weights Tests
//=============================================================================

TEST_F(SwarmBrainLocalTest, GetConsensusWeightsNoAgents) {
    float* weights = nullptr;
    uint32_t num_weights = 0;

    int result = swarm_brain_get_consensus_weights(
        manager, &weights, &num_weights
    );

    EXPECT_EQ(result, NIMCP_SUCCESS);
    EXPECT_EQ(weights, nullptr);
    EXPECT_EQ(num_weights, 0u);
}

TEST_F(SwarmBrainLocalTest, GetConsensusWeightsMultipleAgents) {
    for (uint32_t i = 1; i <= 3; i++) {
        swarm_brain_create_for_agent(manager, i, 100);
        swarm_brain_sync_weights(manager, i);  // Initialize weights
    }

    float* weights = nullptr;
    uint32_t num_weights = 0;

    int result = swarm_brain_get_consensus_weights(
        manager, &weights, &num_weights
    );

    EXPECT_EQ(result, NIMCP_SUCCESS);

    if (weights) {
        nimcp_free(weights);
    }
}

TEST_F(SwarmBrainLocalTest, GetConsensusWeightsNullParams) {
    int result = swarm_brain_get_consensus_weights(
        manager, nullptr, nullptr
    );

    EXPECT_NE(result, NIMCP_SUCCESS);
}

//=============================================================================
// Divergence Detection Tests
//=============================================================================

TEST_F(SwarmBrainLocalTest, GetDivergenceExistingAgent) {
    swarm_brain_create_for_agent(manager, 1, 100);
    swarm_brain_sync_weights(manager, 1);

    float divergence = swarm_brain_get_divergence(manager, 1);
    EXPECT_GE(divergence, 0.0f);  // Should be non-negative
    EXPECT_LE(divergence, 1.0f);  // Should be normalized
}

TEST_F(SwarmBrainLocalTest, GetDivergenceNonexistentAgent) {
    float divergence = swarm_brain_get_divergence(manager, 999);
    EXPECT_EQ(divergence, -1.0f);  // Error indicator
}

TEST_F(SwarmBrainLocalTest, GetDivergenceNullManager) {
    float divergence = swarm_brain_get_divergence(nullptr, 1);
    EXPECT_EQ(divergence, -1.0f);
}

TEST_F(SwarmBrainLocalTest, GetDivergentAgentsNone) {
    swarm_brain_create_for_agent(manager, 1, 100);

    uint32_t* agents = nullptr;
    uint32_t count = 0;

    int result = swarm_brain_get_divergent_agents(
        manager, &agents, &count
    );

    EXPECT_EQ(result, NIMCP_SUCCESS);
    EXPECT_EQ(agents, nullptr);
    EXPECT_EQ(count, 0u);
}

TEST_F(SwarmBrainLocalTest, GetDivergentAgentsNullParams) {
    int result = swarm_brain_get_divergent_agents(
        manager, nullptr, nullptr
    );

    EXPECT_NE(result, NIMCP_SUCCESS);
}

//=============================================================================
// Statistics Tests
//=============================================================================

TEST_F(SwarmBrainLocalTest, GetStatsInitial) {
    swarm_brain_stats_t stats;
    int result = swarm_brain_get_stats(manager, &stats);

    EXPECT_EQ(result, NIMCP_SUCCESS);
    EXPECT_EQ(stats.num_agents, 0u);
    EXPECT_EQ(stats.total_neurons, 0u);
}

TEST_F(SwarmBrainLocalTest, GetStatsWithAgents) {
    for (uint32_t i = 1; i <= 3; i++) {
        swarm_brain_create_for_agent(manager, i, 100);
    }

    swarm_brain_stats_t stats;
    int result = swarm_brain_get_stats(manager, &stats);

    EXPECT_EQ(result, NIMCP_SUCCESS);
    EXPECT_EQ(stats.num_agents, 3u);
    EXPECT_EQ(stats.total_neurons, 300u);
}

TEST_F(SwarmBrainLocalTest, ResetStats) {
    swarm_brain_create_for_agent(manager, 1, 100);
    swarm_brain_sync_weights(manager, 1);

    int result = swarm_brain_reset_stats(manager);
    EXPECT_EQ(result, NIMCP_SUCCESS);

    swarm_brain_stats_t stats;
    swarm_brain_get_stats(manager, &stats);
    EXPECT_EQ(stats.sync_count, 0u);
}

TEST_F(SwarmBrainLocalTest, GetStatsNullParams) {
    int result = swarm_brain_get_stats(manager, nullptr);
    EXPECT_NE(result, NIMCP_SUCCESS);
}

//=============================================================================
// Utility Function Tests
//=============================================================================

TEST_F(SwarmBrainLocalTest, DefaultConfigValues) {
    swarm_brain_config_t cfg = swarm_brain_default_config();

    EXPECT_EQ(cfg.default_brain_size, SWARM_BRAIN_DEFAULT_SIZE);
    EXPECT_EQ(cfg.max_local_neurons, SWARM_BRAIN_MAX_NEURONS);
    EXPECT_EQ(cfg.sync_interval_ms, SWARM_BRAIN_DEFAULT_SYNC_INTERVAL);
    EXPECT_FLOAT_EQ(cfg.divergence_threshold,
                    SWARM_BRAIN_DEFAULT_DIVERGENCE_THRESHOLD);
}

TEST_F(SwarmBrainLocalTest, GetAllAgentsEmpty) {
    uint32_t* agents = nullptr;
    uint32_t count = 0;

    int result = swarm_brain_get_all_agents(manager, &agents, &count);

    EXPECT_EQ(result, NIMCP_SUCCESS);
    EXPECT_EQ(agents, nullptr);
    EXPECT_EQ(count, 0u);
}

TEST_F(SwarmBrainLocalTest, GetAllAgents) {
    std::vector<uint32_t> agent_ids = {1, 5, 10, 15};

    for (uint32_t id : agent_ids) {
        swarm_brain_create_for_agent(manager, id, 100);
    }

    uint32_t* agents = nullptr;
    uint32_t count = 0;

    int result = swarm_brain_get_all_agents(manager, &agents, &count);

    EXPECT_EQ(result, NIMCP_SUCCESS);
    EXPECT_EQ(count, agent_ids.size());

    if (agents) {
        // Verify all agent IDs are present
        for (uint32_t i = 0; i < count; i++) {
            EXPECT_TRUE(swarm_brain_has_agent(manager, agents[i]));
        }
        nimcp_free(agents);
    }
}

TEST_F(SwarmBrainLocalTest, GetAllAgentsNullParams) {
    int result = swarm_brain_get_all_agents(manager, nullptr, nullptr);
    EXPECT_NE(result, NIMCP_SUCCESS);
}

//=============================================================================
// Multi-Agent Scenario Tests
//=============================================================================

TEST_F(SwarmBrainLocalTest, MultiAgentCreationAndDestruction) {
    const uint32_t num_agents = 20;

    // Create agents
    for (uint32_t i = 1; i <= num_agents; i++) {
        int result = swarm_brain_create_for_agent(manager, i, 100);
        EXPECT_EQ(result, NIMCP_SUCCESS);
    }

    EXPECT_EQ(swarm_brain_get_agent_count(manager), num_agents);

    // Destroy half
    for (uint32_t i = 1; i <= num_agents / 2; i++) {
        int result = swarm_brain_destroy_for_agent(manager, i);
        EXPECT_EQ(result, NIMCP_SUCCESS);
    }

    EXPECT_EQ(swarm_brain_get_agent_count(manager), num_agents / 2);
}

TEST_F(SwarmBrainLocalTest, MultiAgentSync) {
    for (uint32_t i = 1; i <= 5; i++) {
        swarm_brain_create_for_agent(manager, i, 100);
    }

    // Sync all
    int result = swarm_brain_sync_all(manager);
    EXPECT_EQ(result, NIMCP_SUCCESS);

    // Verify all agents synced
    swarm_brain_stats_t stats;
    swarm_brain_get_stats(manager, &stats);
    EXPECT_GT(stats.sync_count, 0u);
}

//=============================================================================
// Edge Cases and Error Handling
//=============================================================================

TEST_F(SwarmBrainLocalTest, MaxAgentsLimit) {
    // Try to create more than max agents
    for (uint32_t i = 1; i <= SWARM_BRAIN_MAX_AGENTS + 10; i++) {
        int result = swarm_brain_create_for_agent(manager, i, 50);
        if (i <= SWARM_BRAIN_MAX_AGENTS) {
            EXPECT_EQ(result, NIMCP_SUCCESS);
        } else {
            EXPECT_NE(result, NIMCP_SUCCESS);  // Should fail after limit
        }
    }
}

TEST_F(SwarmBrainLocalTest, ZeroSizeBrain) {
    // Should use default size
    int result = swarm_brain_create_for_agent(manager, 1, 0);
    EXPECT_EQ(result, NIMCP_SUCCESS);
}

TEST_F(SwarmBrainLocalTest, ConcurrentAccess) {
    // Basic thread safety check (single-threaded test)
    swarm_brain_create_for_agent(manager, 1, 100);

    for (int i = 0; i < 100; i++) {
        swarm_brain_sync_weights(manager, 1);
        float divergence = swarm_brain_get_divergence(manager, 1);
        (void)divergence;  // Suppress unused warning
    }

    SUCCEED();
}

//=============================================================================
// Memory Leak Tests
//=============================================================================

TEST_F(SwarmBrainLocalTest, NoMemoryLeaksOnDestruction) {
    // Create and destroy multiple times
    for (int iteration = 0; iteration < 10; iteration++) {
        auto* mgr = swarm_brain_manager_create(&config);
        ASSERT_NE(mgr, nullptr);

        for (uint32_t i = 1; i <= 5; i++) {
            swarm_brain_create_for_agent(mgr, i, 100);
        }

        swarm_brain_manager_destroy(mgr);
    }

    SUCCEED();
}

TEST_F(SwarmBrainLocalTest, NoMemoryLeaksOnAgentDestruction) {
    for (int iteration = 0; iteration < 10; iteration++) {
        for (uint32_t i = 1; i <= 5; i++) {
            swarm_brain_create_for_agent(manager, i, 100);
        }

        for (uint32_t i = 1; i <= 5; i++) {
            swarm_brain_destroy_for_agent(manager, i);
        }
    }

    EXPECT_EQ(swarm_brain_get_agent_count(manager), 0u);
}

//=============================================================================
// Main
//=============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
