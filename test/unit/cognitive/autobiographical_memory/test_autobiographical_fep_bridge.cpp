/**
 * @file test_autobiographical_fep_bridge.cpp
 * @brief Unit tests for Autobiographical Memory FEP Bridge module
 *
 * WHAT: Comprehensive tests for FEP-Autobiographical Memory bidirectional integration
 * WHY:  Ensure surprise encoding, memory replay, and prior updates work correctly
 * HOW:  Test lifecycle, connections, surprise encoding, replay, prior updates, and bio-async
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstring>
#include "cognitive/autobiographical_memory/nimcp_autobiographical_fep_bridge.h"
#include "cognitive/free_energy/nimcp_free_energy.h"

class AutobiographicalFepBridgeTest : public ::testing::Test {
protected:
    autobiographical_fep_bridge_t* bridge = nullptr;

    void SetUp() override {
        autobiographical_fep_config_t config;
        autobiographical_fep_bridge_default_config(&config);
        bridge = autobiographical_fep_bridge_create(&config);
    }

    void TearDown() override {
        if (bridge) {
            autobiographical_fep_bridge_destroy(bridge);
            bridge = nullptr;
        }
    }
};

/* ============================================================================
 * Lifecycle Tests
 * ============================================================================ */

TEST_F(AutobiographicalFepBridgeTest, CreateDestroy) {
    ASSERT_NE(bridge, nullptr);
}

TEST_F(AutobiographicalFepBridgeTest, CreateWithNullConfig) {
    autobiographical_fep_bridge_t* br = autobiographical_fep_bridge_create(nullptr);
    ASSERT_NE(br, nullptr);
    autobiographical_fep_bridge_destroy(br);
}

TEST_F(AutobiographicalFepBridgeTest, DestroyNull) {
    autobiographical_fep_bridge_destroy(nullptr);
}

TEST_F(AutobiographicalFepBridgeTest, DefaultConfig) {
    autobiographical_fep_config_t config;
    int ret = autobiographical_fep_bridge_default_config(&config);

    EXPECT_EQ(ret, 0);
    EXPECT_GT(config.surprise_memory_threshold, 0.0f);
    EXPECT_GT(config.model_update_rate, 0.0f);
    EXPECT_TRUE(config.enable_surprise_encoding);
    EXPECT_TRUE(config.enable_memory_replay);
}

TEST_F(AutobiographicalFepBridgeTest, DefaultConfigNullPtr) {
    int ret = autobiographical_fep_bridge_default_config(nullptr);
    EXPECT_EQ(ret, -1);
}

/* ============================================================================
 * Connection Tests
 * ============================================================================ */

TEST_F(AutobiographicalFepBridgeTest, ConnectFep) {
    fep_config_t fep_config;
    fep_default_config(&fep_config);
    fep_system_t* fep = fep_create(&fep_config, 8, 4);
    ASSERT_NE(fep, nullptr);

    int ret = autobiographical_fep_bridge_connect_fep(bridge, fep);
    EXPECT_EQ(ret, 0);

    fep_destroy(fep);
}

TEST_F(AutobiographicalFepBridgeTest, ConnectFepNull) {
    EXPECT_NE(autobiographical_fep_bridge_connect_fep(nullptr, nullptr), 0);
}

TEST_F(AutobiographicalFepBridgeTest, ConnectAutobiographical) {
    autobiographical_memory_t autobio = 0;
    int ret = autobiographical_fep_bridge_connect_autobiographical(bridge, autobio);
    EXPECT_EQ(ret, 0);
}

TEST_F(AutobiographicalFepBridgeTest, ConnectAutobiographicalNull) {
    autobiographical_memory_t autobio = 0;
    EXPECT_NE(autobiographical_fep_bridge_connect_autobiographical(nullptr, autobio), 0);
}

/* ============================================================================
 * FEP → Autobiographical Tests
 * ============================================================================ */

TEST_F(AutobiographicalFepBridgeTest, EncodeSurprisingEpisode) {
    int ret = autobiographical_fep_encode_surprising_episode(bridge);
    EXPECT_EQ(ret, 0);
}

TEST_F(AutobiographicalFepBridgeTest, EncodeSurprisingEpisodeNull) {
    int ret = autobiographical_fep_encode_surprising_episode(nullptr);
    EXPECT_NE(ret, 0);
}

TEST_F(AutobiographicalFepBridgeTest, ReplayMemories) {
    int ret = autobiographical_fep_replay_memories(bridge);
    EXPECT_EQ(ret, 0);
}

TEST_F(AutobiographicalFepBridgeTest, ReplayMemoriesNull) {
    int ret = autobiographical_fep_replay_memories(nullptr);
    EXPECT_NE(ret, 0);
}

/* ============================================================================
 * Autobiographical → FEP Tests
 * ============================================================================ */

TEST_F(AutobiographicalFepBridgeTest, UpdatePriorsFromMemory) {
    int ret = autobiographical_fep_update_priors_from_memory(bridge);
    EXPECT_EQ(ret, 0);
}

TEST_F(AutobiographicalFepBridgeTest, UpdatePriorsFromMemoryNull) {
    int ret = autobiographical_fep_update_priors_from_memory(nullptr);
    EXPECT_NE(ret, 0);
}

/* ============================================================================
 * Update & State Tests
 * ============================================================================ */

TEST_F(AutobiographicalFepBridgeTest, Update) {
    int ret = autobiographical_fep_bridge_update(bridge, 100);
    EXPECT_EQ(ret, 0);
}

TEST_F(AutobiographicalFepBridgeTest, UpdateNull) {
    int ret = autobiographical_fep_bridge_update(nullptr, 100);
    EXPECT_NE(ret, 0);
}

TEST_F(AutobiographicalFepBridgeTest, GetState) {
    autobiographical_fep_state_t state;
    int ret = autobiographical_fep_bridge_get_state(bridge, &state);
    EXPECT_EQ(ret, 0);
}

TEST_F(AutobiographicalFepBridgeTest, GetStateNull) {
    autobiographical_fep_state_t state;
    EXPECT_NE(autobiographical_fep_bridge_get_state(nullptr, &state), 0);
    EXPECT_NE(autobiographical_fep_bridge_get_state(bridge, nullptr), 0);
}

TEST_F(AutobiographicalFepBridgeTest, GetStats) {
    autobiographical_fep_stats_t stats;
    int ret = autobiographical_fep_bridge_get_stats(bridge, &stats);
    EXPECT_EQ(ret, 0);
}

TEST_F(AutobiographicalFepBridgeTest, GetStatsNull) {
    autobiographical_fep_stats_t stats;
    EXPECT_NE(autobiographical_fep_bridge_get_stats(nullptr, &stats), 0);
    EXPECT_NE(autobiographical_fep_bridge_get_stats(bridge, nullptr), 0);
}

/* ============================================================================
 * Bio-Async Tests
 * ============================================================================ */

TEST_F(AutobiographicalFepBridgeTest, ConnectBioAsync) {
    int ret = autobiographical_fep_bridge_connect_bio_async(bridge);
    (void)ret;
}

TEST_F(AutobiographicalFepBridgeTest, ConnectBioAsyncNull) {
    int ret = autobiographical_fep_bridge_connect_bio_async(nullptr);
    EXPECT_NE(ret, 0);
}

TEST_F(AutobiographicalFepBridgeTest, DisconnectBioAsync) {
    autobiographical_fep_bridge_connect_bio_async(bridge);
    int ret = autobiographical_fep_bridge_disconnect_bio_async(bridge);
    EXPECT_EQ(ret, 0);
}

TEST_F(AutobiographicalFepBridgeTest, DisconnectBioAsyncNull) {
    int ret = autobiographical_fep_bridge_disconnect_bio_async(nullptr);
    EXPECT_NE(ret, 0);
}

TEST_F(AutobiographicalFepBridgeTest, IsBioAsyncConnected) {
    bool connected = autobiographical_fep_bridge_is_bio_async_connected(bridge);
    (void)connected;
}

TEST_F(AutobiographicalFepBridgeTest, IsBioAsyncConnectedNull) {
    bool connected = autobiographical_fep_bridge_is_bio_async_connected(nullptr);
    EXPECT_FALSE(connected);
}
