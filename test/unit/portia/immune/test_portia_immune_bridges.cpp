/**
 * @file test_portia_immune_bridges.cpp
 * @brief Unit tests for Portia-Immune Bridge modules
 * @version 1.0.0
 * @date 2025-12-12
 *
 * Tests for:
 * - Portia Attention-Immune Bridge (resource allocation)
 * - Portia Learning-Immune Bridge (adaptive learning)
 * - Portia Sensor Fusion-Immune Bridge (multimodal sensing)
 */

#include <gtest/gtest.h>
#include <cstring>

extern "C" {
#include "portia/immune/nimcp_portia_attention_immune_bridge.h"
#include "portia/immune/nimcp_portia_learning_immune_bridge.h"
#include "portia/immune/nimcp_portia_sensor_fusion_immune_bridge.h"
#include "cognitive/immune/nimcp_brain_immune.h"
#include "utils/memory/nimcp_memory.h"
}

/* ============================================================================
 * Portia Attention-Immune Bridge Test Fixture
 * ============================================================================ */

class PortiaAttentionImmuneBridgeTest : public ::testing::Test {
protected:
    brain_immune_system_t* immune_system = nullptr;
    portia_attention_t mock_attention;
    portia_attention_immune_bridge_t* bridge = nullptr;

    void SetUp() override {
        /* Create brain immune system */
        brain_immune_config_t config;
        brain_immune_default_config(&config);
        immune_system = brain_immune_create(&config);
        ASSERT_NE(immune_system, nullptr);
        brain_immune_start(immune_system);

        /* Initialize mock attention system */
        memset(&mock_attention, 0, sizeof(mock_attention));

        /* Create bridge */
        portia_attention_immune_config_t bridge_config;
        portia_attention_immune_default_config(&bridge_config);
        bridge = portia_attention_immune_bridge_create(
            &bridge_config,
            immune_system,
            &mock_attention
        );
        ASSERT_NE(bridge, nullptr);
    }

    void TearDown() override {
        if (bridge) {
            portia_attention_immune_bridge_destroy(bridge);
            bridge = nullptr;
        }
        if (immune_system) {
            brain_immune_stop(immune_system);
            brain_immune_destroy(immune_system);
            immune_system = nullptr;
        }
    }
};

/* ============================================================================
 * Portia Attention-Immune Bridge Lifecycle Tests
 * ============================================================================ */

TEST_F(PortiaAttentionImmuneBridgeTest, DefaultConfigIsValid) {
    portia_attention_immune_config_t config;
    int result = portia_attention_immune_default_config(&config);
    EXPECT_EQ(result, 0);
    EXPECT_TRUE(config.enable_resource_budget_reduction);
    EXPECT_TRUE(config.enable_allocation_efficiency_loss);
}

TEST_F(PortiaAttentionImmuneBridgeTest, DefaultConfigNullFails) {
    int result = portia_attention_immune_default_config(nullptr);
    EXPECT_EQ(result, -1);
}

TEST_F(PortiaAttentionImmuneBridgeTest, CreateWithNullImmuneFails) {
    portia_attention_immune_bridge_t* b = portia_attention_immune_bridge_create(
        nullptr, nullptr, &mock_attention
    );
    EXPECT_EQ(b, nullptr);
}

TEST_F(PortiaAttentionImmuneBridgeTest, CreateWithNullAttentionFails) {
    portia_attention_immune_bridge_t* b = portia_attention_immune_bridge_create(
        nullptr, immune_system, nullptr
    );
    EXPECT_EQ(b, nullptr);
}

TEST_F(PortiaAttentionImmuneBridgeTest, CreateWithDefaultConfig) {
    portia_attention_immune_bridge_t* b = portia_attention_immune_bridge_create(
        nullptr, immune_system, &mock_attention
    );
    ASSERT_NE(b, nullptr);
    portia_attention_immune_bridge_destroy(b);
}

TEST_F(PortiaAttentionImmuneBridgeTest, DestroyNull) {
    portia_attention_immune_bridge_destroy(nullptr);
}

/* ============================================================================
 * Portia Attention-Immune Bridge Update Tests
 * ============================================================================ */

TEST_F(PortiaAttentionImmuneBridgeTest, BridgeUpdate) {
    int result = portia_attention_immune_bridge_update(bridge, 100);
    EXPECT_EQ(result, NIMCP_SUCCESS);
    EXPECT_GT(bridge->total_updates, 0u);
}

TEST_F(PortiaAttentionImmuneBridgeTest, BridgeUpdateNull) {
    int result = portia_attention_immune_bridge_update(nullptr, 100);
    EXPECT_EQ(result, NIMCP_ERROR_NULL_POINTER);
}

TEST_F(PortiaAttentionImmuneBridgeTest, MultipleUpdates) {
    for (int i = 0; i < 10; i++) {
        int result = portia_attention_immune_bridge_update(bridge, 10);
        EXPECT_EQ(result, NIMCP_SUCCESS);
    }
    EXPECT_EQ(bridge->total_updates, 10u);
}

/* ============================================================================
 * Portia Attention-Immune Bridge Query Tests
 * ============================================================================ */

TEST_F(PortiaAttentionImmuneBridgeTest, GetCytokineEffects) {
    cytokine_portia_attention_effects_t effects;
    int result = portia_attention_immune_get_cytokine_effects(bridge, &effects);
    EXPECT_EQ(result, NIMCP_SUCCESS);
}

TEST_F(PortiaAttentionImmuneBridgeTest, GetInflammationState) {
    inflammation_portia_attention_state_t state;
    int result = portia_attention_immune_get_inflammation_state(bridge, &state);
    EXPECT_EQ(result, NIMCP_SUCCESS);
}

TEST_F(PortiaAttentionImmuneBridgeTest, GetResourceBudgetFactor) {
    float factor = portia_attention_immune_get_resource_budget_factor(bridge);
    EXPECT_GE(factor, 0.0f);
    EXPECT_LE(factor, 1.0f);
}

TEST_F(PortiaAttentionImmuneBridgeTest, GetResourceBudgetFactorNull) {
    float factor = portia_attention_immune_get_resource_budget_factor(nullptr);
    EXPECT_FLOAT_EQ(factor, 1.0f); /* Default full budget */
}

/* ============================================================================
 * Portia Attention-Immune Bridge Bio-Async Tests
 * ============================================================================ */

TEST_F(PortiaAttentionImmuneBridgeTest, ConnectBioAsync) {
    int result = portia_attention_immune_connect_bio_async(bridge);
    EXPECT_EQ(result, NIMCP_SUCCESS);
    EXPECT_TRUE(portia_attention_immune_is_bio_async_connected(bridge));
}

TEST_F(PortiaAttentionImmuneBridgeTest, DisconnectBioAsync) {
    portia_attention_immune_connect_bio_async(bridge);
    int result = portia_attention_immune_disconnect_bio_async(bridge);
    EXPECT_EQ(result, NIMCP_SUCCESS);
    EXPECT_FALSE(portia_attention_immune_is_bio_async_connected(bridge));
}

TEST_F(PortiaAttentionImmuneBridgeTest, IsBioAsyncConnectedNull) {
    bool connected = portia_attention_immune_is_bio_async_connected(nullptr);
    EXPECT_FALSE(connected);
}

/* ============================================================================
 * Portia Learning-Immune Bridge Test Fixture
 * ============================================================================ */

class PortiaLearningImmuneBridgeTest : public ::testing::Test {
protected:
    brain_immune_system_t* immune_system = nullptr;
    portia_learning_t mock_learning;
    portia_learning_immune_bridge_t* bridge = nullptr;

    void SetUp() override {
        /* Create brain immune system */
        brain_immune_config_t config;
        brain_immune_default_config(&config);
        immune_system = brain_immune_create(&config);
        ASSERT_NE(immune_system, nullptr);
        brain_immune_start(immune_system);

        /* Initialize mock learning system */
        memset(&mock_learning, 0, sizeof(mock_learning));

        /* Create bridge */
        portia_learning_immune_config_t bridge_config;
        portia_learning_immune_default_config(&bridge_config);
        bridge = portia_learning_immune_bridge_create(
            &bridge_config,
            immune_system,
            &mock_learning
        );
        ASSERT_NE(bridge, nullptr);
    }

    void TearDown() override {
        if (bridge) {
            portia_learning_immune_bridge_destroy(bridge);
            bridge = nullptr;
        }
        if (immune_system) {
            brain_immune_stop(immune_system);
            brain_immune_destroy(immune_system);
            immune_system = nullptr;
        }
    }
};

/* ============================================================================
 * Portia Learning-Immune Bridge Lifecycle Tests
 * ============================================================================ */

TEST_F(PortiaLearningImmuneBridgeTest, DefaultConfigIsValid) {
    portia_learning_immune_config_t config;
    int result = portia_learning_immune_default_config(&config);
    EXPECT_EQ(result, 0);
    EXPECT_TRUE(config.enable_learning_rate_modulation);
}

TEST_F(PortiaLearningImmuneBridgeTest, DefaultConfigNullFails) {
    int result = portia_learning_immune_default_config(nullptr);
    EXPECT_EQ(result, -1);
}

TEST_F(PortiaLearningImmuneBridgeTest, CreateWithDefaultConfig) {
    portia_learning_immune_bridge_t* b = portia_learning_immune_bridge_create(
        nullptr, immune_system, &mock_learning
    );
    ASSERT_NE(b, nullptr);
    portia_learning_immune_bridge_destroy(b);
}

TEST_F(PortiaLearningImmuneBridgeTest, DestroyNull) {
    portia_learning_immune_bridge_destroy(nullptr);
}

/* ============================================================================
 * Portia Learning-Immune Bridge Update Tests
 * ============================================================================ */

TEST_F(PortiaLearningImmuneBridgeTest, BridgeUpdate) {
    int result = portia_learning_immune_bridge_update(bridge, 100);
    EXPECT_EQ(result, NIMCP_SUCCESS);
    EXPECT_GT(bridge->total_updates, 0u);
}

TEST_F(PortiaLearningImmuneBridgeTest, BridgeUpdateNull) {
    int result = portia_learning_immune_bridge_update(nullptr, 100);
    EXPECT_EQ(result, NIMCP_ERROR_NULL_POINTER);
}

/* ============================================================================
 * Portia Learning-Immune Bridge Bio-Async Tests
 * ============================================================================ */

TEST_F(PortiaLearningImmuneBridgeTest, ConnectBioAsync) {
    int result = portia_learning_immune_connect_bio_async(bridge);
    EXPECT_EQ(result, NIMCP_SUCCESS);
    EXPECT_TRUE(portia_learning_immune_is_bio_async_connected(bridge));
}

TEST_F(PortiaLearningImmuneBridgeTest, DisconnectBioAsync) {
    portia_learning_immune_connect_bio_async(bridge);
    int result = portia_learning_immune_disconnect_bio_async(bridge);
    EXPECT_EQ(result, NIMCP_SUCCESS);
    EXPECT_FALSE(portia_learning_immune_is_bio_async_connected(bridge));
}

/* ============================================================================
 * Portia Sensor Fusion-Immune Bridge Test Fixture
 * ============================================================================ */

class PortiaSensorFusionImmuneBridgeTest : public ::testing::Test {
protected:
    brain_immune_system_t* immune_system = nullptr;
    portia_sensor_fusion_t mock_sensor_fusion;
    portia_sensor_fusion_immune_bridge_t* bridge = nullptr;

    void SetUp() override {
        /* Create brain immune system */
        brain_immune_config_t config;
        brain_immune_default_config(&config);
        immune_system = brain_immune_create(&config);
        ASSERT_NE(immune_system, nullptr);
        brain_immune_start(immune_system);

        /* Initialize mock sensor fusion system */
        memset(&mock_sensor_fusion, 0, sizeof(mock_sensor_fusion));

        /* Create bridge */
        portia_sensor_fusion_immune_config_t bridge_config;
        portia_sensor_fusion_immune_default_config(&bridge_config);
        bridge = portia_sensor_fusion_immune_bridge_create(
            &bridge_config,
            immune_system,
            &mock_sensor_fusion
        );
        ASSERT_NE(bridge, nullptr);
    }

    void TearDown() override {
        if (bridge) {
            portia_sensor_fusion_immune_bridge_destroy(bridge);
            bridge = nullptr;
        }
        if (immune_system) {
            brain_immune_stop(immune_system);
            brain_immune_destroy(immune_system);
            immune_system = nullptr;
        }
    }
};

/* ============================================================================
 * Portia Sensor Fusion-Immune Bridge Lifecycle Tests
 * ============================================================================ */

TEST_F(PortiaSensorFusionImmuneBridgeTest, DefaultConfigIsValid) {
    portia_sensor_fusion_immune_config_t config;
    int result = portia_sensor_fusion_immune_default_config(&config);
    EXPECT_EQ(result, 0);
    EXPECT_TRUE(config.enable_sensor_sensitivity_modulation);
}

TEST_F(PortiaSensorFusionImmuneBridgeTest, DefaultConfigNullFails) {
    int result = portia_sensor_fusion_immune_default_config(nullptr);
    EXPECT_EQ(result, -1);
}

TEST_F(PortiaSensorFusionImmuneBridgeTest, CreateWithDefaultConfig) {
    portia_sensor_fusion_immune_bridge_t* b = portia_sensor_fusion_immune_bridge_create(
        nullptr, immune_system, &mock_sensor_fusion
    );
    ASSERT_NE(b, nullptr);
    portia_sensor_fusion_immune_bridge_destroy(b);
}

TEST_F(PortiaSensorFusionImmuneBridgeTest, DestroyNull) {
    portia_sensor_fusion_immune_bridge_destroy(nullptr);
}

/* ============================================================================
 * Portia Sensor Fusion-Immune Bridge Update Tests
 * ============================================================================ */

TEST_F(PortiaSensorFusionImmuneBridgeTest, BridgeUpdate) {
    int result = portia_sensor_fusion_immune_bridge_update(bridge, 100);
    EXPECT_EQ(result, NIMCP_SUCCESS);
    EXPECT_GT(bridge->total_updates, 0u);
}

TEST_F(PortiaSensorFusionImmuneBridgeTest, BridgeUpdateNull) {
    int result = portia_sensor_fusion_immune_bridge_update(nullptr, 100);
    EXPECT_EQ(result, NIMCP_ERROR_NULL_POINTER);
}

/* ============================================================================
 * Portia Sensor Fusion-Immune Bridge Bio-Async Tests
 * ============================================================================ */

TEST_F(PortiaSensorFusionImmuneBridgeTest, ConnectBioAsync) {
    int result = portia_sensor_fusion_immune_connect_bio_async(bridge);
    EXPECT_EQ(result, NIMCP_SUCCESS);
    EXPECT_TRUE(portia_sensor_fusion_immune_is_bio_async_connected(bridge));
}

TEST_F(PortiaSensorFusionImmuneBridgeTest, DisconnectBioAsync) {
    portia_sensor_fusion_immune_connect_bio_async(bridge);
    int result = portia_sensor_fusion_immune_disconnect_bio_async(bridge);
    EXPECT_EQ(result, NIMCP_SUCCESS);
    EXPECT_FALSE(portia_sensor_fusion_immune_is_bio_async_connected(bridge));
}

TEST_F(PortiaSensorFusionImmuneBridgeTest, IsBioAsyncConnectedNull) {
    bool connected = portia_sensor_fusion_immune_is_bio_async_connected(nullptr);
    EXPECT_FALSE(connected);
}

/* ============================================================================
 * Integration Tests
 * ============================================================================ */

TEST_F(PortiaAttentionImmuneBridgeTest, FullUpdateCycle) {
    /* Connect bio-async */
    portia_attention_immune_connect_bio_async(bridge);

    /* Run update cycles */
    for (int i = 0; i < 5; i++) {
        int result = portia_attention_immune_bridge_update(bridge, 100);
        EXPECT_EQ(result, NIMCP_SUCCESS);
    }

    /* Query final state */
    cytokine_portia_attention_effects_t effects;
    portia_attention_immune_get_cytokine_effects(bridge, &effects);

    inflammation_portia_attention_state_t state;
    portia_attention_immune_get_inflammation_state(bridge, &state);

    /* State should be valid */
    float budget = portia_attention_immune_get_resource_budget_factor(bridge);
    EXPECT_GE(budget, 0.0f);
    EXPECT_LE(budget, 1.0f);
}
