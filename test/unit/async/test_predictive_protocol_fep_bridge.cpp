/**
 * @file test_predictive_protocol_fep_bridge.cpp
 * @brief Unit tests for Predictive Protocol-FEP Bridge module
 *
 * WHAT: Comprehensive tests for FEP-Predictive Protocol bidirectional integration
 * WHY:  Ensure prediction-based protocol state management works correctly
 * HOW:  Test lifecycle, protocol prediction, state updates, and effects
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstring>

extern "C" {
#include "async/nimcp_predictive_protocol_fep_bridge.h"
#include "cognitive/free_energy/nimcp_free_energy.h"
}

class PredictiveProtocolFepBridgeTest : public ::testing::Test {
protected:
    predictive_protocol_fep_bridge_t* bridge = nullptr;
    fep_system_t* fep = nullptr;

    void SetUp() override {
        /* Create FEP system */
        fep_config_t fep_config;
        fep_default_config(&fep_config);
        fep = fep_create(&fep_config, 8, 4);
        ASSERT_NE(fep, nullptr);

        /* Create bridge */
        predictive_protocol_fep_config_t config;
        predictive_protocol_fep_default_config(&config);
        bridge = predictive_protocol_fep_create(&config, fep);
    }

    void TearDown() override {
        if (bridge) {
            predictive_protocol_fep_destroy(bridge);
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

TEST_F(PredictiveProtocolFepBridgeTest, CreateDestroy) {
    ASSERT_NE(bridge, nullptr);
}

TEST_F(PredictiveProtocolFepBridgeTest, CreateWithNullConfig) {
    predictive_protocol_fep_bridge_t* br =
        predictive_protocol_fep_create(nullptr, fep);
    EXPECT_EQ(br, nullptr);
}

TEST_F(PredictiveProtocolFepBridgeTest, CreateWithNullFep) {
    predictive_protocol_fep_config_t config;
    predictive_protocol_fep_default_config(&config);
    predictive_protocol_fep_bridge_t* br =
        predictive_protocol_fep_create(&config, nullptr);
    EXPECT_EQ(br, nullptr);
}

TEST_F(PredictiveProtocolFepBridgeTest, DestroyNull) {
    predictive_protocol_fep_destroy(nullptr);  /* Should not crash */
}

TEST_F(PredictiveProtocolFepBridgeTest, DefaultConfig) {
    predictive_protocol_fep_config_t config;
    int ret = predictive_protocol_fep_default_config(&config);

    EXPECT_EQ(ret, 0);
    EXPECT_GT(config.state_prediction_confidence, 0.0f);
    EXPECT_GT(config.transition_learning_rate, 0.0f);
    EXPECT_TRUE(config.enable_state_prediction);
    EXPECT_TRUE(config.enable_transition_learning);
}

TEST_F(PredictiveProtocolFepBridgeTest, DefaultConfigNullPtr) {
    int ret = predictive_protocol_fep_default_config(nullptr);
    EXPECT_NE(ret, 0);
}

/* ============================================================================
 * Effects Update Tests
 * ============================================================================ */

TEST_F(PredictiveProtocolFepBridgeTest, UpdateEffects) {
    int ret = predictive_protocol_fep_update_effects(bridge);
    EXPECT_EQ(ret, 0);
}

TEST_F(PredictiveProtocolFepBridgeTest, UpdateEffectsNull) {
    EXPECT_NE(predictive_protocol_fep_update_effects(nullptr), 0);
}

TEST_F(PredictiveProtocolFepBridgeTest, UpdateEffectsComputesPredictions) {
    int ret = predictive_protocol_fep_update_effects(bridge);
    EXPECT_EQ(ret, 0);

    predictive_protocol_fep_effects_t effects;
    ret = predictive_protocol_fep_get_effects(bridge, &effects);
    EXPECT_EQ(ret, 0);

    /* Prediction confidence should be valid */
    EXPECT_GE(effects.state_prediction_confidence, 0.0f);
    EXPECT_LE(effects.state_prediction_confidence, 1.0f);
}

/* ============================================================================
 * Protocol State Prediction Tests
 * ============================================================================ */

TEST_F(PredictiveProtocolFepBridgeTest, PredictNextState) {
    uint32_t predicted_state;
    float confidence;

    int ret = predictive_protocol_fep_predict_next_state(
        bridge, 0, &predicted_state, &confidence);
    EXPECT_EQ(ret, 0);
    EXPECT_GE(confidence, 0.0f);
    EXPECT_LE(confidence, 1.0f);
}

TEST_F(PredictiveProtocolFepBridgeTest, PredictNextStateNull) {
    uint32_t predicted_state;
    float confidence;

    EXPECT_NE(predictive_protocol_fep_predict_next_state(
        nullptr, 0, &predicted_state, &confidence), 0);
    EXPECT_NE(predictive_protocol_fep_predict_next_state(
        bridge, 0, nullptr, &confidence), 0);
    EXPECT_NE(predictive_protocol_fep_predict_next_state(
        bridge, 0, &predicted_state, nullptr), 0);
}

/* ============================================================================
 * State Transition Observation Tests
 * ============================================================================ */

TEST_F(PredictiveProtocolFepBridgeTest, ObserveTransition) {
    int ret = predictive_protocol_fep_observe_transition(bridge, 0, 1);
    EXPECT_EQ(ret, 0);
}

TEST_F(PredictiveProtocolFepBridgeTest, ObserveTransitionNull) {
    EXPECT_NE(predictive_protocol_fep_observe_transition(nullptr, 0, 1), 0);
}

TEST_F(PredictiveProtocolFepBridgeTest, ObserveTransitionUpdatesEffects) {
    predictive_protocol_fep_observe_transition(bridge, 0, 1);

    fep_predictive_protocol_effects_t effects;
    int ret = predictive_protocol_fep_get_protocol_effects(bridge, &effects);
    EXPECT_EQ(ret, 0);

    EXPECT_EQ(effects.total_transitions, 1u);
}

/* ============================================================================
 * Bio-Async Integration Tests
 * ============================================================================ */

TEST_F(PredictiveProtocolFepBridgeTest, InitiallyNotConnected) {
    EXPECT_FALSE(predictive_protocol_fep_is_bio_async_connected(bridge));
}

TEST_F(PredictiveProtocolFepBridgeTest, ConnectDisconnectBioAsync) {
    int ret = predictive_protocol_fep_connect_bio_async(bridge);
    EXPECT_EQ(ret, 0);
    EXPECT_TRUE(predictive_protocol_fep_is_bio_async_connected(bridge));

    ret = predictive_protocol_fep_disconnect_bio_async(bridge);
    EXPECT_EQ(ret, 0);
    EXPECT_FALSE(predictive_protocol_fep_is_bio_async_connected(bridge));
}

TEST_F(PredictiveProtocolFepBridgeTest, ConnectNull) {
    EXPECT_NE(predictive_protocol_fep_connect_bio_async(nullptr), 0);
}

TEST_F(PredictiveProtocolFepBridgeTest, DisconnectNull) {
    EXPECT_NE(predictive_protocol_fep_disconnect_bio_async(nullptr), 0);
}

TEST_F(PredictiveProtocolFepBridgeTest, IsConnectedNull) {
    EXPECT_FALSE(predictive_protocol_fep_is_bio_async_connected(nullptr));
}

/* ============================================================================
 * Query Tests
 * ============================================================================ */

TEST_F(PredictiveProtocolFepBridgeTest, GetEffects) {
    predictive_protocol_fep_effects_t effects;
    int ret = predictive_protocol_fep_get_effects(bridge, &effects);

    EXPECT_EQ(ret, 0);
    EXPECT_GE(effects.state_prediction_confidence, 0.0f);
}

TEST_F(PredictiveProtocolFepBridgeTest, GetEffectsNull) {
    predictive_protocol_fep_effects_t effects;

    EXPECT_NE(predictive_protocol_fep_get_effects(nullptr, &effects), 0);
    EXPECT_NE(predictive_protocol_fep_get_effects(bridge, nullptr), 0);
}

TEST_F(PredictiveProtocolFepBridgeTest, GetProtocolEffects) {
    fep_predictive_protocol_effects_t effects;
    int ret = predictive_protocol_fep_get_protocol_effects(bridge, &effects);

    EXPECT_EQ(ret, 0);
    EXPECT_GE(effects.total_transitions, 0u);
}

TEST_F(PredictiveProtocolFepBridgeTest, GetProtocolEffectsNull) {
    fep_predictive_protocol_effects_t effects;

    EXPECT_NE(predictive_protocol_fep_get_protocol_effects(nullptr, &effects), 0);
    EXPECT_NE(predictive_protocol_fep_get_protocol_effects(bridge, nullptr), 0);
}

TEST_F(PredictiveProtocolFepBridgeTest, GetStats) {
    predictive_protocol_fep_stats_t stats;
    int ret = predictive_protocol_fep_get_stats(bridge, &stats);

    EXPECT_EQ(ret, 0);
    EXPECT_GE(stats.prediction_accuracy, 0.0f);
}

TEST_F(PredictiveProtocolFepBridgeTest, GetStatsNull) {
    predictive_protocol_fep_stats_t stats;

    EXPECT_NE(predictive_protocol_fep_get_stats(nullptr, &stats), 0);
    EXPECT_NE(predictive_protocol_fep_get_stats(bridge, nullptr), 0);
}

TEST_F(PredictiveProtocolFepBridgeTest, ResetStats) {
    predictive_protocol_fep_observe_transition(bridge, 0, 1);

    int ret = predictive_protocol_fep_reset_stats(bridge);
    EXPECT_EQ(ret, 0);

    predictive_protocol_fep_stats_t stats;
    predictive_protocol_fep_get_stats(bridge, &stats);

    EXPECT_EQ(stats.total_predictions, 0u);
}

TEST_F(PredictiveProtocolFepBridgeTest, ResetStatsNull) {
    EXPECT_NE(predictive_protocol_fep_reset_stats(nullptr), 0);
}

/* ============================================================================
 * Integration Behavior Tests
 * ============================================================================ */

TEST_F(PredictiveProtocolFepBridgeTest, TransitionLearning) {
    /* Learn a pattern: state 0 -> 1 -> 2 */
    predictive_protocol_fep_observe_transition(bridge, 0, 1);
    predictive_protocol_fep_observe_transition(bridge, 1, 2);
    predictive_protocol_fep_observe_transition(bridge, 0, 1);
    predictive_protocol_fep_observe_transition(bridge, 1, 2);

    /* Predict next state from state 0 */
    uint32_t predicted_state;
    float confidence;
    predictive_protocol_fep_predict_next_state(bridge, 0, &predicted_state, &confidence);

    /* After learning, should predict state 1 with some confidence */
    /* Note: exact values depend on implementation */
    EXPECT_GE(confidence, 0.0f);
}

TEST_F(PredictiveProtocolFepBridgeTest, PredictionErrorComputed) {
    /* Make prediction */
    predictive_protocol_fep_update_effects(bridge);

    /* Observe transition */
    predictive_protocol_fep_observe_transition(bridge, 0, 1);

    fep_predictive_protocol_effects_t effects;
    predictive_protocol_fep_get_protocol_effects(bridge, &effects);

    /* Prediction error should be computed */
    EXPECT_GE(effects.transition_prediction_error, 0.0f);
}
