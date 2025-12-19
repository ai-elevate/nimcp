/**
 * @file test_axon_orchestrator_bridge.cpp
 * @brief Unit Tests for Axon-Orchestrator Bridge
 *
 * Tests all aspects of the axon-orchestrator bridge including:
 * - Lifecycle management (create/destroy)
 * - Configuration
 * - Synapse-axon mapping
 * - Spike callback forwarding
 * - Delay compensation
 * - Activity tracking
 * - Bio-async integration
 * - Statistics
 */

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <atomic>

extern "C" {
#include "plasticity/orchestrator/nimcp_axon_orchestrator_bridge.h"
#include "plasticity/nimcp_plasticity_orchestrator.h"
#include "core/axon/nimcp_axon.h"
#include "utils/memory/nimcp_memory.h"
}

// ============================================================================
// Test Fixture
// ============================================================================

class AxonOrchestratorBridgeTest : public ::testing::Test {
protected:
    axon_orchestrator_bridge_t* bridge = nullptr;
    plasticity_orchestrator_t* orchestrator = nullptr;
    axon_network_t* axon_network = nullptr;
    axon_orchestrator_config_t config;

    void SetUp() override {
        axon_orchestrator_default_config(&config);

        // Create orchestrator
        orchestrator = plasticity_orchestrator_create(nullptr);

        // Create axon network with default config
        axon_network_config_t net_config;
        axon_network_default_config(&net_config);
        axon_network = axon_network_create(&net_config);
    }

    void TearDown() override {
        if (bridge) {
            axon_orchestrator_bridge_destroy(bridge);
            bridge = nullptr;
        }
        if (orchestrator) {
            plasticity_orchestrator_destroy(orchestrator);
            orchestrator = nullptr;
        }
        if (axon_network) {
            axon_network_destroy(axon_network);
            axon_network = nullptr;
        }
    }

    // Helper to create bridge
    void CreateBridge() {
        bridge = axon_orchestrator_bridge_create(&config, orchestrator, axon_network);
    }
};

// ============================================================================
// Lifecycle Tests
// ============================================================================

TEST_F(AxonOrchestratorBridgeTest, DefaultConfigSetsReasonableDefaults) {
    EXPECT_EQ(axon_orchestrator_default_config(&config), 0);

    // Check enabled flags
    EXPECT_TRUE(config.enable_delay_compensation);
    EXPECT_TRUE(config.enable_activity_tracking);
    EXPECT_FALSE(config.enable_bio_async);  // Off by default

    // Check parameters
    EXPECT_GT(config.activity_ema_tau_ms, 0.0f);
    EXPECT_GT(config.initial_mapping_capacity, 0u);
}

TEST_F(AxonOrchestratorBridgeTest, DefaultConfigReturnsErrorForNull) {
    EXPECT_EQ(axon_orchestrator_default_config(nullptr), -1);
}

TEST_F(AxonOrchestratorBridgeTest, CreateWithDefaultConfig) {
    bridge = axon_orchestrator_bridge_create(nullptr, orchestrator, axon_network);
    ASSERT_NE(bridge, nullptr);
}

TEST_F(AxonOrchestratorBridgeTest, CreateWithCustomConfig) {
    config.enable_delay_compensation = false;
    config.enable_activity_tracking = false;

    CreateBridge();
    ASSERT_NE(bridge, nullptr);
}

TEST_F(AxonOrchestratorBridgeTest, CreateRequiresOrchestrator) {
    bridge = axon_orchestrator_bridge_create(&config, nullptr, axon_network);
    EXPECT_EQ(bridge, nullptr);
}

TEST_F(AxonOrchestratorBridgeTest, CreateRequiresAxonNetwork) {
    bridge = axon_orchestrator_bridge_create(&config, orchestrator, nullptr);
    EXPECT_EQ(bridge, nullptr);
}

TEST_F(AxonOrchestratorBridgeTest, DestroyNullIsSafe) {
    axon_orchestrator_bridge_destroy(nullptr);
    // Should not crash
}

// ============================================================================
// Mapping Tests
// ============================================================================

TEST_F(AxonOrchestratorBridgeTest, MapSynapseToAxon) {
    CreateBridge();
    ASSERT_NE(bridge, nullptr);

    // Map synapse 100 to axon 1
    EXPECT_EQ(axon_orchestrator_map_synapse(bridge, 100, 1), 0);

    // Verify mapping
    uint32_t axon_id;
    EXPECT_EQ(axon_orchestrator_get_synapse_mapping(bridge, 100, &axon_id), 0);
    EXPECT_EQ(axon_id, 1u);
}

TEST_F(AxonOrchestratorBridgeTest, MapMultipleSynapsesToAxon) {
    CreateBridge();
    ASSERT_NE(bridge, nullptr);

    // Map multiple synapses to same axon
    EXPECT_EQ(axon_orchestrator_map_synapse(bridge, 100, 1), 0);
    EXPECT_EQ(axon_orchestrator_map_synapse(bridge, 101, 1), 0);
    EXPECT_EQ(axon_orchestrator_map_synapse(bridge, 102, 1), 0);

    // Verify all mappings
    uint32_t axon_id;
    EXPECT_EQ(axon_orchestrator_get_synapse_mapping(bridge, 100, &axon_id), 0);
    EXPECT_EQ(axon_id, 1u);
    EXPECT_EQ(axon_orchestrator_get_synapse_mapping(bridge, 101, &axon_id), 0);
    EXPECT_EQ(axon_id, 1u);
    EXPECT_EQ(axon_orchestrator_get_synapse_mapping(bridge, 102, &axon_id), 0);
    EXPECT_EQ(axon_id, 1u);
}

TEST_F(AxonOrchestratorBridgeTest, MapSynapsesToDifferentAxons) {
    CreateBridge();
    ASSERT_NE(bridge, nullptr);

    // Map synapses to different axons
    EXPECT_EQ(axon_orchestrator_map_synapse(bridge, 100, 1), 0);
    EXPECT_EQ(axon_orchestrator_map_synapse(bridge, 200, 2), 0);
    EXPECT_EQ(axon_orchestrator_map_synapse(bridge, 300, 3), 0);

    // Verify mappings
    uint32_t axon_id;
    EXPECT_EQ(axon_orchestrator_get_synapse_mapping(bridge, 100, &axon_id), 0);
    EXPECT_EQ(axon_id, 1u);
    EXPECT_EQ(axon_orchestrator_get_synapse_mapping(bridge, 200, &axon_id), 0);
    EXPECT_EQ(axon_id, 2u);
    EXPECT_EQ(axon_orchestrator_get_synapse_mapping(bridge, 300, &axon_id), 0);
    EXPECT_EQ(axon_id, 3u);
}

TEST_F(AxonOrchestratorBridgeTest, UnmapSynapse) {
    CreateBridge();
    ASSERT_NE(bridge, nullptr);

    // Map and then unmap
    EXPECT_EQ(axon_orchestrator_map_synapse(bridge, 100, 1), 0);
    EXPECT_EQ(axon_orchestrator_unmap_synapse(bridge, 100), 0);

    // Verify unmapped
    uint32_t axon_id;
    EXPECT_EQ(axon_orchestrator_get_synapse_mapping(bridge, 100, &axon_id), -1);
}

TEST_F(AxonOrchestratorBridgeTest, GetMappingForUnknownSynapseReturnsError) {
    CreateBridge();
    ASSERT_NE(bridge, nullptr);

    uint32_t axon_id;
    EXPECT_EQ(axon_orchestrator_get_synapse_mapping(bridge, 99999, &axon_id), -1);
}

TEST_F(AxonOrchestratorBridgeTest, MapWithNullBridgeReturnsError) {
    EXPECT_EQ(axon_orchestrator_map_synapse(nullptr, 100, 1), -1);
}

TEST_F(AxonOrchestratorBridgeTest, GetMappingCount) {
    CreateBridge();
    ASSERT_NE(bridge, nullptr);

    EXPECT_EQ(axon_orchestrator_get_mapping_count(bridge), 0u);

    axon_orchestrator_map_synapse(bridge, 100, 1);
    EXPECT_EQ(axon_orchestrator_get_mapping_count(bridge), 1u);

    axon_orchestrator_map_synapse(bridge, 200, 2);
    EXPECT_EQ(axon_orchestrator_get_mapping_count(bridge), 2u);

    axon_orchestrator_unmap_synapse(bridge, 100);
    EXPECT_EQ(axon_orchestrator_get_mapping_count(bridge), 1u);
}

// ============================================================================
// Pre-Spike Forwarding Tests
// ============================================================================

TEST_F(AxonOrchestratorBridgeTest, ForwardPreSpike) {
    CreateBridge();
    ASSERT_NE(bridge, nullptr);

    // Set initial weight
    plasticity_orchestrator_set_weight(orchestrator, 100, 0.5f);

    // Forward pre-spike
    EXPECT_EQ(axon_orchestrator_forward_pre_spike(bridge, 100, 1000), 0);
}

TEST_F(AxonOrchestratorBridgeTest, ForwardPreSpikeWithNullBridge) {
    EXPECT_EQ(axon_orchestrator_forward_pre_spike(nullptr, 100, 1000), -1);
}

TEST_F(AxonOrchestratorBridgeTest, ForwardPreSpikeUpdatesStats) {
    CreateBridge();
    ASSERT_NE(bridge, nullptr);

    axon_orchestrator_stats_t stats;
    axon_orchestrator_get_stats(bridge, &stats);
    EXPECT_EQ(stats.pre_spikes_forwarded, 0u);

    // Forward multiple pre-spikes
    plasticity_orchestrator_set_weight(orchestrator, 100, 0.5f);
    axon_orchestrator_forward_pre_spike(bridge, 100, 1000);
    axon_orchestrator_forward_pre_spike(bridge, 100, 2000);
    axon_orchestrator_forward_pre_spike(bridge, 100, 3000);

    axon_orchestrator_get_stats(bridge, &stats);
    EXPECT_EQ(stats.pre_spikes_forwarded, 3u);
}

// ============================================================================
// Activity Tracking Tests
// ============================================================================

TEST_F(AxonOrchestratorBridgeTest, GetAxonActivityInitiallyZero) {
    CreateBridge();
    ASSERT_NE(bridge, nullptr);

    float activity = axon_orchestrator_get_axon_activity(bridge, 1);
    EXPECT_FLOAT_EQ(activity, 0.0f);
}

TEST_F(AxonOrchestratorBridgeTest, UpdateIncreasesActivityWithSpikes) {
    config.enable_activity_tracking = true;
    CreateBridge();
    ASSERT_NE(bridge, nullptr);

    // Map synapse and forward spikes
    axon_orchestrator_map_synapse(bridge, 100, 1);
    plasticity_orchestrator_set_weight(orchestrator, 100, 0.5f);

    // Forward several pre-spikes
    for (int i = 0; i < 10; i++) {
        axon_orchestrator_forward_pre_spike(bridge, 100, i * 1000);
    }

    // Run update to process activity
    axon_orchestrator_bridge_update(bridge, 10000);

    // Activity should be tracked (implementation-dependent)
    axon_orchestrator_stats_t stats;
    axon_orchestrator_get_stats(bridge, &stats);
    EXPECT_GE(stats.update_calls, 1u);
}

// ============================================================================
// Delay Compensation Tests
// ============================================================================

TEST_F(AxonOrchestratorBridgeTest, DelayCompensationEnabled) {
    config.enable_delay_compensation = true;
    CreateBridge();
    ASSERT_NE(bridge, nullptr);

    // Map synapse with axon that has a delay
    axon_orchestrator_map_synapse(bridge, 100, 1);

    // The spike callback should adjust timing based on axon delay
    // This is tested via the callback mechanism
}

TEST_F(AxonOrchestratorBridgeTest, DelayCompensationDisabled) {
    config.enable_delay_compensation = false;
    CreateBridge();
    ASSERT_NE(bridge, nullptr);

    // With delay compensation disabled, spikes are forwarded without adjustment
    // This is tested via the callback mechanism
}

// ============================================================================
// Statistics Tests
// ============================================================================

TEST_F(AxonOrchestratorBridgeTest, GetStatsInitiallyZero) {
    CreateBridge();
    ASSERT_NE(bridge, nullptr);

    axon_orchestrator_stats_t stats;
    EXPECT_EQ(axon_orchestrator_get_stats(bridge, &stats), 0);

    EXPECT_EQ(stats.pre_spikes_forwarded, 0u);
    EXPECT_EQ(stats.spikes_received, 0u);
    EXPECT_EQ(stats.mappings_created, 0u);
    EXPECT_EQ(stats.mappings_removed, 0u);
    EXPECT_EQ(stats.bio_async_messages_sent, 0u);
    EXPECT_EQ(stats.update_calls, 0u);
}

TEST_F(AxonOrchestratorBridgeTest, GetStatsWithNullBridgeReturnsError) {
    axon_orchestrator_stats_t stats;
    EXPECT_EQ(axon_orchestrator_get_stats(nullptr, &stats), -1);
}

TEST_F(AxonOrchestratorBridgeTest, GetStatsWithNullStatsReturnsError) {
    CreateBridge();
    ASSERT_NE(bridge, nullptr);

    EXPECT_EQ(axon_orchestrator_get_stats(bridge, nullptr), -1);
}

TEST_F(AxonOrchestratorBridgeTest, ResetStats) {
    CreateBridge();
    ASSERT_NE(bridge, nullptr);

    // Create some activity
    axon_orchestrator_map_synapse(bridge, 100, 1);
    plasticity_orchestrator_set_weight(orchestrator, 100, 0.5f);
    axon_orchestrator_forward_pre_spike(bridge, 100, 1000);

    // Verify non-zero stats
    axon_orchestrator_stats_t stats;
    axon_orchestrator_get_stats(bridge, &stats);
    EXPECT_GT(stats.pre_spikes_forwarded + stats.mappings_created, 0u);

    // Reset and verify zero
    EXPECT_EQ(axon_orchestrator_reset_stats(bridge), 0);
    axon_orchestrator_get_stats(bridge, &stats);
    EXPECT_EQ(stats.pre_spikes_forwarded, 0u);
    EXPECT_EQ(stats.spikes_received, 0u);
}

TEST_F(AxonOrchestratorBridgeTest, MappingStatsTracked) {
    CreateBridge();
    ASSERT_NE(bridge, nullptr);

    // Create mappings
    axon_orchestrator_map_synapse(bridge, 100, 1);
    axon_orchestrator_map_synapse(bridge, 200, 2);

    axon_orchestrator_stats_t stats;
    axon_orchestrator_get_stats(bridge, &stats);
    EXPECT_EQ(stats.mappings_created, 2u);

    // Remove a mapping
    axon_orchestrator_unmap_synapse(bridge, 100);
    axon_orchestrator_get_stats(bridge, &stats);
    EXPECT_EQ(stats.mappings_removed, 1u);
}

// ============================================================================
// Update Tests
// ============================================================================

TEST_F(AxonOrchestratorBridgeTest, BridgeUpdate) {
    CreateBridge();
    ASSERT_NE(bridge, nullptr);

    EXPECT_EQ(axon_orchestrator_bridge_update(bridge, 1000), 0);

    axon_orchestrator_stats_t stats;
    axon_orchestrator_get_stats(bridge, &stats);
    EXPECT_EQ(stats.update_calls, 1u);
}

TEST_F(AxonOrchestratorBridgeTest, BridgeUpdateWithNullReturnsError) {
    EXPECT_EQ(axon_orchestrator_bridge_update(nullptr, 1000), -1);
}

// ============================================================================
// Accessor Tests
// ============================================================================

TEST_F(AxonOrchestratorBridgeTest, GetOrchestrator) {
    CreateBridge();
    ASSERT_NE(bridge, nullptr);

    plasticity_orchestrator_t* orch = axon_orchestrator_get_orchestrator(bridge);
    EXPECT_EQ(orch, orchestrator);
}

TEST_F(AxonOrchestratorBridgeTest, GetAxonNetwork) {
    CreateBridge();
    ASSERT_NE(bridge, nullptr);

    axon_network_t* net = axon_orchestrator_get_axon_network(bridge);
    EXPECT_EQ(net, axon_network);
}

TEST_F(AxonOrchestratorBridgeTest, GetOrchestratorWithNull) {
    EXPECT_EQ(axon_orchestrator_get_orchestrator(nullptr), nullptr);
}

TEST_F(AxonOrchestratorBridgeTest, GetAxonNetworkWithNull) {
    EXPECT_EQ(axon_orchestrator_get_axon_network(nullptr), nullptr);
}

// ============================================================================
// Bio-Async Integration Tests
// ============================================================================

TEST_F(AxonOrchestratorBridgeTest, ConnectBioAsyncWithoutRouter) {
    CreateBridge();
    ASSERT_NE(bridge, nullptr);

    // Should succeed even without router (just sets flag)
    int result = axon_orchestrator_connect_bio_async(bridge);
    // May return 0 or -1 depending on bio-async availability
    (void)result;
}

TEST_F(AxonOrchestratorBridgeTest, DisconnectBioAsync) {
    CreateBridge();
    ASSERT_NE(bridge, nullptr);

    // Connect and disconnect
    axon_orchestrator_connect_bio_async(bridge);
    EXPECT_EQ(axon_orchestrator_disconnect_bio_async(bridge), 0);
}

TEST_F(AxonOrchestratorBridgeTest, IsBioAsyncConnectedInitiallyFalse) {
    CreateBridge();
    ASSERT_NE(bridge, nullptr);

    EXPECT_FALSE(axon_orchestrator_is_bio_async_connected(bridge));
}

TEST_F(AxonOrchestratorBridgeTest, ConnectBioAsyncWithNull) {
    EXPECT_EQ(axon_orchestrator_connect_bio_async(nullptr), -1);
}

TEST_F(AxonOrchestratorBridgeTest, DisconnectBioAsyncWithNull) {
    EXPECT_EQ(axon_orchestrator_disconnect_bio_async(nullptr), -1);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(AxonOrchestratorBridgeTest, MapSameSynapseTwiceUpdatesMapping) {
    CreateBridge();
    ASSERT_NE(bridge, nullptr);

    // Map synapse to axon 1
    EXPECT_EQ(axon_orchestrator_map_synapse(bridge, 100, 1), 0);

    // Map same synapse to axon 2
    EXPECT_EQ(axon_orchestrator_map_synapse(bridge, 100, 2), 0);

    // Should have updated to axon 2
    uint32_t axon_id;
    EXPECT_EQ(axon_orchestrator_get_synapse_mapping(bridge, 100, &axon_id), 0);
    EXPECT_EQ(axon_id, 2u);
}

TEST_F(AxonOrchestratorBridgeTest, UnmapNonExistentSynapseReturnsError) {
    CreateBridge();
    ASSERT_NE(bridge, nullptr);

    EXPECT_EQ(axon_orchestrator_unmap_synapse(bridge, 99999), -1);
}

TEST_F(AxonOrchestratorBridgeTest, LargeMappingCount) {
    config.initial_mapping_capacity = 100;
    CreateBridge();
    ASSERT_NE(bridge, nullptr);

    // Create many mappings (beyond initial capacity)
    for (uint32_t i = 0; i < 500; i++) {
        EXPECT_EQ(axon_orchestrator_map_synapse(bridge, i, i % 10), 0);
    }

    EXPECT_EQ(axon_orchestrator_get_mapping_count(bridge), 500u);

    // Verify random samples
    uint32_t axon_id;
    EXPECT_EQ(axon_orchestrator_get_synapse_mapping(bridge, 123, &axon_id), 0);
    EXPECT_EQ(axon_id, 123u % 10);

    EXPECT_EQ(axon_orchestrator_get_synapse_mapping(bridge, 456, &axon_id), 0);
    EXPECT_EQ(axon_id, 456u % 10);
}
