/**
 * @file test_swarm_consciousness_fep_bridge.cpp
 * @brief Unit tests for Swarm Consciousness-FEP Bridge module
 *
 * WHAT: Tests for FEP-Swarm Consciousness bidirectional integration
 * WHY:  Ensure collective consciousness metrics and FEP beliefs align
 * HOW:  Test lifecycle, phi computation, integration effects, and updates
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstring>

extern "C" {
#include "swarm/nimcp_swarm_consciousness_fep_bridge.h"
#include "cognitive/free_energy/nimcp_free_energy.h"
}

class SwarmConsciousnessFepBridgeTest : public ::testing::Test {
protected:
    swarm_consciousness_fep_bridge_t* bridge = nullptr;
    fep_system_t* fep = nullptr;

    void SetUp() override {
        /* Create FEP system */
        fep_config_t fep_config;
        fep_default_config(&fep_config);
        fep = fep_create(&fep_config, 8, 4);
        ASSERT_NE(fep, nullptr);

        /* Create bridge */
        swarm_consciousness_fep_config_t config;
        swarm_consciousness_fep_default_config(&config);
        bridge = swarm_consciousness_fep_create(&config, fep);
    }

    void TearDown() override {
        if (bridge) {
            swarm_consciousness_fep_destroy(bridge);
            bridge = nullptr;
        }
        if (fep) {
            fep_destroy(fep);
            fep = nullptr;
        }
    }
};

/* ============================================================================
 * Lifecycle Tests
 * ============================================================================ */

TEST_F(SwarmConsciousnessFepBridgeTest, CreateDestroy) {
    ASSERT_NE(bridge, nullptr);
}

TEST_F(SwarmConsciousnessFepBridgeTest, CreateWithNullConfig) {
    swarm_consciousness_fep_bridge_t* br = swarm_consciousness_fep_create(nullptr, fep);
    EXPECT_EQ(br, nullptr);
}

TEST_F(SwarmConsciousnessFepBridgeTest, CreateWithNullFep) {
    swarm_consciousness_fep_config_t config;
    swarm_consciousness_fep_default_config(&config);
    swarm_consciousness_fep_bridge_t* br = swarm_consciousness_fep_create(&config, nullptr);
    EXPECT_EQ(br, nullptr);
}

TEST_F(SwarmConsciousnessFepBridgeTest, DestroyNull) {
    swarm_consciousness_fep_destroy(nullptr);
}

TEST_F(SwarmConsciousnessFepBridgeTest, DefaultConfig) {
    swarm_consciousness_fep_config_t config;
    int ret = swarm_consciousness_fep_default_config(&config);
    EXPECT_EQ(ret, 0);
    EXPECT_GT(config.phi_precision_coupling, 0.0f);
    EXPECT_TRUE(config.enable_collective_phi);
}

TEST_F(SwarmConsciousnessFepBridgeTest, DefaultConfigNull) {
    EXPECT_NE(swarm_consciousness_fep_default_config(nullptr), 0);
}

/* ============================================================================
 * Update Tests
 * ============================================================================ */

TEST_F(SwarmConsciousnessFepBridgeTest, Update) {
    int ret = swarm_consciousness_fep_update(bridge);
    EXPECT_EQ(ret, 0);
}

TEST_F(SwarmConsciousnessFepBridgeTest, UpdateNull) {
    EXPECT_NE(swarm_consciousness_fep_update(nullptr), 0);
}

/* ============================================================================
 * Effects Tests
 * ============================================================================ */

TEST_F(SwarmConsciousnessFepBridgeTest, GetEffects) {
    swarm_consciousness_fep_update(bridge);

    swarm_consciousness_fep_effects_t effects;
    int ret = swarm_consciousness_fep_get_effects(bridge, &effects);

    EXPECT_EQ(ret, 0);
    EXPECT_GE(effects.integration_modulation, 0.0f);
}

TEST_F(SwarmConsciousnessFepBridgeTest, GetEffectsNull) {
    swarm_consciousness_fep_effects_t effects;
    EXPECT_NE(swarm_consciousness_fep_get_effects(nullptr, &effects), 0);
    EXPECT_NE(swarm_consciousness_fep_get_effects(bridge, nullptr), 0);
}

TEST_F(SwarmConsciousnessFepBridgeTest, GetConsciousnessEffects) {
    swarm_consciousness_fep_update(bridge);

    fep_swarm_consciousness_effects_t effects;
    int ret = swarm_consciousness_fep_get_consciousness_effects(bridge, &effects);

    EXPECT_EQ(ret, 0);
    EXPECT_GE(effects.collective_phi, 0.0f);
}

TEST_F(SwarmConsciousnessFepBridgeTest, GetConsciousnessEffectsNull) {
    fep_swarm_consciousness_effects_t effects;
    EXPECT_NE(swarm_consciousness_fep_get_consciousness_effects(nullptr, &effects), 0);
    EXPECT_NE(swarm_consciousness_fep_get_consciousness_effects(bridge, nullptr), 0);
}

/* ============================================================================
 * Statistics Tests
 * ============================================================================ */

TEST_F(SwarmConsciousnessFepBridgeTest, GetStats) {
    swarm_consciousness_fep_update(bridge);

    swarm_consciousness_fep_stats_t stats;
    int ret = swarm_consciousness_fep_get_stats(bridge, &stats);

    EXPECT_EQ(ret, 0);
}

TEST_F(SwarmConsciousnessFepBridgeTest, GetStatsNull) {
    swarm_consciousness_fep_stats_t stats;
    EXPECT_NE(swarm_consciousness_fep_get_stats(nullptr, &stats), 0);
    EXPECT_NE(swarm_consciousness_fep_get_stats(bridge, nullptr), 0);
}

TEST_F(SwarmConsciousnessFepBridgeTest, ResetStats) {
    swarm_consciousness_fep_update(bridge);
    int ret = swarm_consciousness_fep_reset_stats(bridge);
    EXPECT_EQ(ret, 0);
}

TEST_F(SwarmConsciousnessFepBridgeTest, ResetStatsNull) {
    EXPECT_NE(swarm_consciousness_fep_reset_stats(nullptr), 0);
}

/* ============================================================================
 * Bio-Async Tests
 * ============================================================================ */

TEST_F(SwarmConsciousnessFepBridgeTest, InitiallyNotConnected) {
    EXPECT_FALSE(swarm_consciousness_fep_is_bio_async_connected(bridge));
}

TEST_F(SwarmConsciousnessFepBridgeTest, ConnectDisconnect) {
    int ret = swarm_consciousness_fep_connect_bio_async(bridge);
    EXPECT_EQ(ret, 0);
    EXPECT_TRUE(swarm_consciousness_fep_is_bio_async_connected(bridge));

    ret = swarm_consciousness_fep_disconnect_bio_async(bridge);
    EXPECT_EQ(ret, 0);
    EXPECT_FALSE(swarm_consciousness_fep_is_bio_async_connected(bridge));
}
