/**
 * @file test_curiosity_fep_bridge.cpp
 * @brief Unit tests for Curiosity FEP Bridge module
 *
 * WHAT: Comprehensive tests for FEP-Curiosity bidirectional integration
 * WHY:  Ensure epistemic value computation and exploration triggering work correctly
 * HOW:  Test lifecycle, connections, epistemic value, knowledge gaps, exploration, and bio-async
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstring>
#include "cognitive/curiosity/nimcp_curiosity_fep_bridge.h"
#include "cognitive/free_energy/nimcp_free_energy.h"

class CuriosityFepBridgeTest : public ::testing::Test {
protected:
    curiosity_fep_bridge_t* bridge = nullptr;

    void SetUp() override {
        curiosity_fep_config_t config;
        curiosity_fep_bridge_default_config(&config);
        bridge = curiosity_fep_bridge_create(&config);
    }

    void TearDown() override {
        if (bridge) {
            curiosity_fep_bridge_destroy(bridge);
            bridge = nullptr;
        }
    }
};

/* ============================================================================
 * Lifecycle Tests
 * ============================================================================ */

TEST_F(CuriosityFepBridgeTest, CreateDestroy) {
    ASSERT_NE(bridge, nullptr);
}

TEST_F(CuriosityFepBridgeTest, CreateWithNullConfig) {
    curiosity_fep_bridge_t* br = curiosity_fep_bridge_create(nullptr);
    ASSERT_NE(br, nullptr);
    curiosity_fep_bridge_destroy(br);
}

TEST_F(CuriosityFepBridgeTest, DestroyNull) {
    curiosity_fep_bridge_destroy(nullptr);
}

TEST_F(CuriosityFepBridgeTest, DefaultConfig) {
    curiosity_fep_config_t config;
    int ret = curiosity_fep_bridge_default_config(&config);

    EXPECT_EQ(ret, 0);
    EXPECT_GT(config.epistemic_value_weight, 0.0f);
    EXPECT_GT(config.uncertainty_sensitivity, 0.0f);
    EXPECT_TRUE(config.enable_epistemic_curiosity);
    EXPECT_TRUE(config.enable_knowledge_gap_detection);
}

TEST_F(CuriosityFepBridgeTest, DefaultConfigNullPtr) {
    int ret = curiosity_fep_bridge_default_config(nullptr);
    EXPECT_EQ(ret, -1);
}

/* ============================================================================
 * Connection Tests
 * ============================================================================ */

TEST_F(CuriosityFepBridgeTest, ConnectFep) {
    fep_config_t fep_config;
    fep_default_config(&fep_config);
    fep_system_t* fep = fep_create(&fep_config, 8, 4);
    ASSERT_NE(fep, nullptr);

    int ret = curiosity_fep_bridge_connect_fep(bridge, fep);
    EXPECT_EQ(ret, 0);

    fep_destroy(fep);
}

TEST_F(CuriosityFepBridgeTest, ConnectFepNull) {
    EXPECT_NE(curiosity_fep_bridge_connect_fep(nullptr, nullptr), 0);
}

TEST_F(CuriosityFepBridgeTest, ConnectCuriosity) {
    curiosity_engine_t curiosity = 0;
    int ret = curiosity_fep_bridge_connect_curiosity(bridge, curiosity);
    EXPECT_EQ(ret, 0);
}

TEST_F(CuriosityFepBridgeTest, ConnectCuriosityNull) {
    curiosity_engine_t curiosity = 0;
    EXPECT_NE(curiosity_fep_bridge_connect_curiosity(nullptr, curiosity), 0);
}

/* ============================================================================
 * FEP → Curiosity Tests
 * ============================================================================ */

TEST_F(CuriosityFepBridgeTest, ComputeEpistemicValue) {
    int ret = curiosity_fep_compute_epistemic_value(bridge);
    EXPECT_EQ(ret, 0);
}

TEST_F(CuriosityFepBridgeTest, ComputeEpistemicValueNull) {
    int ret = curiosity_fep_compute_epistemic_value(nullptr);
    EXPECT_NE(ret, 0);
}

TEST_F(CuriosityFepBridgeTest, DetectKnowledgeGaps) {
    int ret = curiosity_fep_detect_knowledge_gaps(bridge);
    EXPECT_EQ(ret, 0);
}

TEST_F(CuriosityFepBridgeTest, DetectKnowledgeGapsNull) {
    int ret = curiosity_fep_detect_knowledge_gaps(nullptr);
    EXPECT_NE(ret, 0);
}

TEST_F(CuriosityFepBridgeTest, TriggerExploration) {
    int ret = curiosity_fep_trigger_exploration(bridge);
    EXPECT_EQ(ret, 0);
}

TEST_F(CuriosityFepBridgeTest, TriggerExplorationNull) {
    int ret = curiosity_fep_trigger_exploration(nullptr);
    EXPECT_NE(ret, 0);
}

/* ============================================================================
 * Curiosity → FEP Tests
 * ============================================================================ */

TEST_F(CuriosityFepBridgeTest, UpdateModelFromLearning) {
    int ret = curiosity_fep_update_model_from_learning(bridge);
    EXPECT_EQ(ret, 0);
}

TEST_F(CuriosityFepBridgeTest, UpdateModelFromLearningNull) {
    int ret = curiosity_fep_update_model_from_learning(nullptr);
    EXPECT_NE(ret, 0);
}

/* ============================================================================
 * Update & State Tests
 * ============================================================================ */

TEST_F(CuriosityFepBridgeTest, Update) {
    int ret = curiosity_fep_bridge_update(bridge, 100);
    EXPECT_EQ(ret, 0);
}

TEST_F(CuriosityFepBridgeTest, UpdateNull) {
    int ret = curiosity_fep_bridge_update(nullptr, 100);
    EXPECT_NE(ret, 0);
}

TEST_F(CuriosityFepBridgeTest, GetState) {
    curiosity_fep_state_t state;
    int ret = curiosity_fep_bridge_get_state(bridge, &state);
    EXPECT_EQ(ret, 0);
}

TEST_F(CuriosityFepBridgeTest, GetStateNull) {
    curiosity_fep_state_t state;
    EXPECT_NE(curiosity_fep_bridge_get_state(nullptr, &state), 0);
    EXPECT_NE(curiosity_fep_bridge_get_state(bridge, nullptr), 0);
}

TEST_F(CuriosityFepBridgeTest, GetStats) {
    curiosity_fep_stats_t stats;
    int ret = curiosity_fep_bridge_get_stats(bridge, &stats);
    EXPECT_EQ(ret, 0);
}

TEST_F(CuriosityFepBridgeTest, GetStatsNull) {
    curiosity_fep_stats_t stats;
    EXPECT_NE(curiosity_fep_bridge_get_stats(nullptr, &stats), 0);
    EXPECT_NE(curiosity_fep_bridge_get_stats(bridge, nullptr), 0);
}

/* ============================================================================
 * Bio-Async Tests
 * ============================================================================ */

TEST_F(CuriosityFepBridgeTest, ConnectBioAsync) {
    int ret = curiosity_fep_bridge_connect_bio_async(bridge);
    (void)ret;
}

TEST_F(CuriosityFepBridgeTest, ConnectBioAsyncNull) {
    int ret = curiosity_fep_bridge_connect_bio_async(nullptr);
    EXPECT_NE(ret, 0);
}

TEST_F(CuriosityFepBridgeTest, DisconnectBioAsync) {
    curiosity_fep_bridge_connect_bio_async(bridge);
    int ret = curiosity_fep_bridge_disconnect_bio_async(bridge);
    EXPECT_EQ(ret, 0);
}

TEST_F(CuriosityFepBridgeTest, DisconnectBioAsyncNull) {
    int ret = curiosity_fep_bridge_disconnect_bio_async(nullptr);
    EXPECT_NE(ret, 0);
}

TEST_F(CuriosityFepBridgeTest, IsBioAsyncConnected) {
    bool connected = curiosity_fep_bridge_is_bio_async_connected(bridge);
    (void)connected;
}

TEST_F(CuriosityFepBridgeTest, IsBioAsyncConnectedNull) {
    bool connected = curiosity_fep_bridge_is_bio_async_connected(nullptr);
    EXPECT_FALSE(connected);
}
