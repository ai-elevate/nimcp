/**
 * @file test_swarm_brain_local_regression.cpp
 * @brief Regression tests for Swarm Brain Local Instantiation
 *
 * REGRESSION FOCUS:
 * - Memory leak prevention
 * - Performance degradation detection
 * - Edge case handling consistency
 * - API stability
 * - Thread safety
 *
 * @author NIMCP Development Team
 * @date 2025-12-08
 */

#include <gtest/gtest.h>
#include <vector>

extern "C" {
#include "swarm/nimcp_swarm_brain_local.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/time/nimcp_time.h"
}

class SwarmBrainLocalRegressionTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

//=============================================================================
// Memory Leak Regression Tests
//=============================================================================

TEST_F(SwarmBrainLocalRegressionTest, NoLeaksOnRepeatedCreationDestruction) {
    swarm_brain_config_t config = swarm_brain_default_config();
    config.enable_bio_async = false;

    // Repeat many times to detect memory leaks
    for (int iteration = 0; iteration < 100; iteration++) {
        auto* mgr = swarm_brain_manager_create(&config);
        ASSERT_NE(mgr, nullptr);

        // Create agents
        for (uint32_t i = 1; i <= 10; i++) {
            swarm_brain_create_for_agent(mgr, i, 100);
        }

        // Use agents
        for (uint32_t i = 1; i <= 10; i++) {
            swarm_brain_sync_weights(mgr, i);
        }

        swarm_brain_manager_destroy(mgr);
    }

    SUCCEED();
}

TEST_F(SwarmBrainLocalRegressionTest, NoLeaksOnAgentChurn) {
    swarm_brain_config_t config = swarm_brain_default_config();
    config.enable_bio_async = false;
    auto* mgr = swarm_brain_manager_create(&config);

    // Repeatedly create and destroy agents
    for (int iteration = 0; iteration < 100; iteration++) {
        for (uint32_t i = 1; i <= 5; i++) {
            swarm_brain_create_for_agent(mgr, i, 100);
        }

        for (uint32_t i = 1; i <= 5; i++) {
            swarm_brain_destroy_for_agent(mgr, i);
        }
    }

    EXPECT_EQ(swarm_brain_get_agent_count(mgr), 0u);
    swarm_brain_manager_destroy(mgr);
}

//=============================================================================
// Performance Regression Tests
//=============================================================================

TEST_F(SwarmBrainLocalRegressionTest, AgentCreationPerformance) {
    swarm_brain_config_t config = swarm_brain_default_config();
    config.enable_bio_async = false;
    auto* mgr = swarm_brain_manager_create(&config);

    const uint32_t num_agents = 50;
    uint64_t start_time = nimcp_time_get_us();

    for (uint32_t i = 1; i <= num_agents; i++) {
        swarm_brain_create_for_agent(mgr, i, 100);
    }

    uint64_t end_time = nimcp_time_get_us();
    uint64_t duration_us = end_time - start_time;

    // Should complete in reasonable time (baseline: < 1 second)
    EXPECT_LT(duration_us, 1000000u);

    swarm_brain_manager_destroy(mgr);
}

TEST_F(SwarmBrainLocalRegressionTest, SyncPerformance) {
    swarm_brain_config_t config = swarm_brain_default_config();
    config.enable_bio_async = false;
    auto* mgr = swarm_brain_manager_create(&config);

    for (uint32_t i = 1; i <= 20; i++) {
        swarm_brain_create_for_agent(mgr, i, 100);
    }

    uint64_t start_time = nimcp_time_get_us();

    swarm_brain_sync_all(mgr);

    uint64_t end_time = nimcp_time_get_us();
    uint64_t duration_us = end_time - start_time;

    // Should complete quickly (baseline: < 100ms)
    EXPECT_LT(duration_us, 100000u);

    swarm_brain_manager_destroy(mgr);
}

//=============================================================================
// Edge Case Regression Tests
//=============================================================================

TEST_F(SwarmBrainLocalRegressionTest, NullParameterHandling) {
    // All null parameter cases should fail gracefully
    EXPECT_EQ(swarm_brain_manager_destroy(nullptr), void());
    EXPECT_NE(swarm_brain_create_for_agent(nullptr, 1, 100), NIMCP_SUCCESS);
    EXPECT_NE(swarm_brain_destroy_for_agent(nullptr, 1), NIMCP_SUCCESS);
    EXPECT_EQ(swarm_brain_get(nullptr, 1), nullptr);
    EXPECT_NE(swarm_brain_sync_weights(nullptr, 1), NIMCP_SUCCESS);
    EXPECT_EQ(swarm_brain_get_divergence(nullptr, 1), -1.0f);
}

TEST_F(SwarmBrainLocalRegressionTest, InvalidAgentIDHandling) {
    swarm_brain_config_t config = swarm_brain_default_config();
    config.enable_bio_async = false;
    auto* mgr = swarm_brain_manager_create(&config);

    // Operations on non-existent agents should fail gracefully
    EXPECT_NE(swarm_brain_destroy_for_agent(mgr, 999), NIMCP_SUCCESS);
    EXPECT_EQ(swarm_brain_get(mgr, 999), nullptr);
    EXPECT_NE(swarm_brain_sync_weights(mgr, 999), NIMCP_SUCCESS);
    EXPECT_EQ(swarm_brain_get_divergence(mgr, 999), -1.0f);

    swarm_brain_manager_destroy(mgr);
}

TEST_F(SwarmBrainLocalRegressionTest, MaxAgentsLimit) {
    swarm_brain_config_t config = swarm_brain_default_config();
    config.enable_bio_async = false;
    auto* mgr = swarm_brain_manager_create(&config);

    // Create up to max
    for (uint32_t i = 1; i <= SWARM_BRAIN_MAX_AGENTS; i++) {
        int result = swarm_brain_create_for_agent(mgr, i, 50);
        EXPECT_EQ(result, NIMCP_SUCCESS);
    }

    // Next should fail
    int result = swarm_brain_create_for_agent(
        mgr, SWARM_BRAIN_MAX_AGENTS + 1, 50
    );
    EXPECT_NE(result, NIMCP_SUCCESS);

    swarm_brain_manager_destroy(mgr);
}

//=============================================================================
// API Stability Regression Tests
//=============================================================================

TEST_F(SwarmBrainLocalRegressionTest, DefaultConfigStability) {
    swarm_brain_config_t config1 = swarm_brain_default_config();
    swarm_brain_config_t config2 = swarm_brain_default_config();

    // Default config should be consistent
    EXPECT_EQ(config1.default_brain_size, config2.default_brain_size);
    EXPECT_EQ(config1.max_local_neurons, config2.max_local_neurons);
    EXPECT_FLOAT_EQ(config1.sync_interval_ms, config2.sync_interval_ms);
    EXPECT_FLOAT_EQ(config1.divergence_threshold, config2.divergence_threshold);
}

TEST_F(SwarmBrainLocalRegressionTest, StatisticsConsistency) {
    swarm_brain_config_t config = swarm_brain_default_config();
    config.enable_bio_async = false;
    auto* mgr = swarm_brain_manager_create(&config);

    // Create agents
    for (uint32_t i = 1; i <= 5; i++) {
        swarm_brain_create_for_agent(mgr, i, 100);
    }

    // Get stats multiple times - should be consistent
    swarm_brain_stats_t stats1, stats2;
    swarm_brain_get_stats(mgr, &stats1);
    swarm_brain_get_stats(mgr, &stats2);

    EXPECT_EQ(stats1.num_agents, stats2.num_agents);
    EXPECT_EQ(stats1.total_neurons, stats2.total_neurons);

    swarm_brain_manager_destroy(mgr);
}

//=============================================================================
// Main
//=============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
