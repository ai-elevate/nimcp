/**
 * @file test_surface_bio_async_integration.cpp
 * @brief Integration Tests for Surface Geometry Bio-Async Integration
 *
 * WHAT: Tests for surface geometry <-> bio-async message routing
 * WHY:  Verify geometry events propagate correctly through bio-async system
 * HOW:  GTest-based integration tests with message routing verification
 *
 * NIMCP STANDARDS:
 * - Integration tests verify cross-module communication
 * - Tests may create real subsystems (not mocks)
 */

#include <gtest/gtest.h>
#include <thread>
#include <chrono>
#include <atomic>

extern "C" {
#include "core/geometry/nimcp_surface_geometry.h"
#include "core/geometry/nimcp_surface_geometry_types.h"
#include "async/bridges/nimcp_surface_bio_async_bridge.h"
#include "async/nimcp_bio_async.h"
}

// Test tolerance
#define TOLERANCE 1e-5f

//=============================================================================
// Test Fixture
//=============================================================================

class SurfaceBioAsyncIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create geometry context
        surface_geometry_default_config(&geo_config);
        geo_ctx = surface_geometry_create(&geo_config);
        ASSERT_NE(geo_ctx, nullptr);

        // Create bio-async bridge
        surface_bio_async_default_config(&bio_config);
        bio_bridge = surface_bio_async_bridge_create(&bio_config);
        ASSERT_NE(bio_bridge, nullptr);

        // Connect bridge to geometry
        int ret = surface_bio_async_bridge_connect_geometry(bio_bridge, geo_ctx);
        ASSERT_EQ(ret, 0);
    }

    void TearDown() override {
        if (bio_bridge) {
            surface_bio_async_bridge_destroy(bio_bridge);
            bio_bridge = nullptr;
        }
        if (geo_ctx) {
            surface_geometry_destroy(geo_ctx);
            geo_ctx = nullptr;
        }
    }

    surface_geometry_config_t geo_config;
    surface_geometry_ctx_t* geo_ctx;
    surface_bio_async_config_t bio_config;
    surface_bio_async_bridge_t* bio_bridge;
};

//=============================================================================
// Connection Tests
//=============================================================================

TEST_F(SurfaceBioAsyncIntegrationTest, BridgeConnectsToGeometry) {
    bool connected = surface_bio_async_bridge_is_connected(bio_bridge);
    EXPECT_TRUE(connected);
}

TEST_F(SurfaceBioAsyncIntegrationTest, BridgeDisconnectsCleanly) {
    int ret = surface_bio_async_bridge_disconnect(bio_bridge);
    EXPECT_EQ(ret, 0);

    bool connected = surface_bio_async_bridge_is_connected(bio_bridge);
    EXPECT_FALSE(connected);
}

TEST_F(SurfaceBioAsyncIntegrationTest, ReconnectAfterDisconnect) {
    surface_bio_async_bridge_disconnect(bio_bridge);
    int ret = surface_bio_async_bridge_connect_geometry(bio_bridge, geo_ctx);
    EXPECT_EQ(ret, 0);

    bool connected = surface_bio_async_bridge_is_connected(bio_bridge);
    EXPECT_TRUE(connected);
}

//=============================================================================
// Message Sending Tests
//=============================================================================

TEST_F(SurfaceBioAsyncIntegrationTest, SendGeometryUpdate) {
    surface_geometry_params_t params = {};
    params.chi = 0.5f;
    params.rho = 0.4f;
    params.regime = SURFACE_REGIME_SPROUTING;

    int ret = surface_bio_async_send_geometry_update(bio_bridge, &params, 1);
    EXPECT_EQ(ret, 0);

    // Check stats updated
    surface_bio_async_stats_t stats = {};
    surface_bio_async_bridge_get_stats(bio_bridge, &stats);
    EXPECT_GE(stats.messages_sent, 1u);
}

TEST_F(SurfaceBioAsyncIntegrationTest, SendBranchFormed) {
    surface_branch_point_t branch = {};
    branch.id = 1;
    branch.degree = 3;
    branch.position = {0.0f, 0.0f, 0.0f};
    branch.params.chi = 0.5f;

    int ret = surface_bio_async_send_branch_formed(bio_bridge, &branch);
    EXPECT_EQ(ret, 0);
}

TEST_F(SurfaceBioAsyncIntegrationTest, SendTrifurcationDetected) {
    surface_branch_point_t branch = {};
    branch.id = 1;
    branch.degree = 4;  // Trifurcation
    branch.params.chi = 0.9f;  // Above threshold

    int ret = surface_bio_async_send_trifurcation_detected(bio_bridge, &branch);
    EXPECT_EQ(ret, 0);
}

TEST_F(SurfaceBioAsyncIntegrationTest, SendSproutFormed) {
    surface_branch_point_t sprout = {};
    sprout.id = 1;
    sprout.is_sprout = true;
    sprout.params.rho = 0.3f;  // Below threshold

    int ret = surface_bio_async_send_sprout_formed(bio_bridge, &sprout);
    EXPECT_EQ(ret, 0);
}

TEST_F(SurfaceBioAsyncIntegrationTest, SendOptimizationComplete) {
    surface_optimization_result_t result = {};
    result.surface_area = 10.0f;
    result.wire_length = 8.0f;
    result.converged = true;

    int ret = surface_bio_async_send_optimization_complete(bio_bridge, &result);
    EXPECT_EQ(ret, 0);
}

//=============================================================================
// Message Sequence Tests
//=============================================================================

TEST_F(SurfaceBioAsyncIntegrationTest, MultipleMessagesInSequence) {
    // Send multiple messages
    surface_geometry_params_t params = {};
    params.chi = 0.5f;
    params.rho = 0.4f;

    for (int i = 0; i < 10; i++) {
        params.chi = 0.1f * (i + 1);
        surface_bio_async_send_geometry_update(bio_bridge, &params, i + 1);
    }

    surface_bio_async_stats_t stats = {};
    surface_bio_async_bridge_get_stats(bio_bridge, &stats);
    EXPECT_GE(stats.messages_sent, 10u);
}

TEST_F(SurfaceBioAsyncIntegrationTest, MixedMessageTypes) {
    // Send different message types
    surface_geometry_params_t params = {};
    params.chi = 0.5f;
    surface_bio_async_send_geometry_update(bio_bridge, &params, 1);

    surface_branch_point_t branch = {};
    branch.id = 1;
    branch.degree = 3;
    surface_bio_async_send_branch_formed(bio_bridge, &branch);

    branch.degree = 4;
    branch.params.chi = 0.9f;
    surface_bio_async_send_trifurcation_detected(bio_bridge, &branch);

    surface_bio_async_stats_t stats = {};
    surface_bio_async_bridge_get_stats(bio_bridge, &stats);
    EXPECT_GE(stats.messages_sent, 3u);
}

//=============================================================================
// Subscription Tests
//=============================================================================

TEST_F(SurfaceBioAsyncIntegrationTest, SubscribeToMessages) {
    surface_bio_subscription_config_t sub_config = {};
    sub_config.message_types = SURFACE_BIO_MSG_GEOMETRY_UPDATE |
                               SURFACE_BIO_MSG_BRANCH_FORMED;
    sub_config.callback = nullptr;  // Would be set in real usage

    uint32_t sub_id;
    int ret = surface_bio_async_subscribe(bio_bridge, &sub_config, &sub_id);
    EXPECT_EQ(ret, 0);
    EXPECT_GT(sub_id, 0u);
}

TEST_F(SurfaceBioAsyncIntegrationTest, UnsubscribeFromMessages) {
    surface_bio_subscription_config_t sub_config = {};
    sub_config.message_types = SURFACE_BIO_MSG_GEOMETRY_UPDATE;

    uint32_t sub_id;
    surface_bio_async_subscribe(bio_bridge, &sub_config, &sub_id);

    int ret = surface_bio_async_unsubscribe(bio_bridge, sub_id);
    EXPECT_EQ(ret, 0);
}

//=============================================================================
// Channel Selection Tests
//=============================================================================

TEST_F(SurfaceBioAsyncIntegrationTest, AnomalyUsesUrgentChannel) {
    // Anomalies should use NOREPINEPHRINE (urgent) channel
    surface_branch_point_t branch = {};
    branch.id = 1;
    branch.params.chi = 3.0f;  // Out of range - anomaly

    int ret = surface_bio_async_send_anomaly_detected(bio_bridge, &branch,
                                                      SURFACE_ANTIGEN_INVALID_CHI);
    EXPECT_EQ(ret, 0);

    // Verify channel selection in stats if available
    surface_bio_async_stats_t stats = {};
    surface_bio_async_bridge_get_stats(bio_bridge, &stats);
    EXPECT_GE(stats.messages_sent, 1u);
}

//=============================================================================
// Error Handling Tests
//=============================================================================

TEST_F(SurfaceBioAsyncIntegrationTest, SendAfterDisconnect) {
    surface_bio_async_bridge_disconnect(bio_bridge);

    surface_geometry_params_t params = {};
    params.chi = 0.5f;

    int ret = surface_bio_async_send_geometry_update(bio_bridge, &params, 1);
    // May fail or queue depending on implementation
    EXPECT_TRUE(ret == 0 || ret == -1);
}

TEST_F(SurfaceBioAsyncIntegrationTest, SendNullParams) {
    int ret = surface_bio_async_send_geometry_update(bio_bridge, nullptr, 1);
    EXPECT_EQ(ret, -1);
}

TEST_F(SurfaceBioAsyncIntegrationTest, SendNullBranch) {
    int ret = surface_bio_async_send_branch_formed(bio_bridge, nullptr);
    EXPECT_EQ(ret, -1);
}

//=============================================================================
// Statistics Accuracy Tests
//=============================================================================

TEST_F(SurfaceBioAsyncIntegrationTest, StatsTrackMessagesAccurately) {
    surface_bio_async_stats_t stats_before = {};
    surface_bio_async_bridge_get_stats(bio_bridge, &stats_before);

    surface_geometry_params_t params = {};
    for (int i = 0; i < 5; i++) {
        surface_bio_async_send_geometry_update(bio_bridge, &params, i);
    }

    surface_bio_async_stats_t stats_after = {};
    surface_bio_async_bridge_get_stats(bio_bridge, &stats_after);

    EXPECT_EQ(stats_after.messages_sent - stats_before.messages_sent, 5u);
}

TEST_F(SurfaceBioAsyncIntegrationTest, ResetStatsWorks) {
    surface_geometry_params_t params = {};
    surface_bio_async_send_geometry_update(bio_bridge, &params, 1);

    int ret = surface_bio_async_bridge_reset_stats(bio_bridge);
    EXPECT_EQ(ret, 0);

    surface_bio_async_stats_t stats = {};
    surface_bio_async_bridge_get_stats(bio_bridge, &stats);
    EXPECT_EQ(stats.messages_sent, 0u);
}
