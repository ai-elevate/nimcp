/**
 * @file test_semantic_compression_fep_bridge.cpp
 * @brief Unit tests for Semantic Compression-FEP Bridge module
 *
 * WHAT: Comprehensive tests for FEP-Semantic Compression bidirectional integration
 * WHY:  Ensure prediction-based compression and semantic preservation work correctly
 * HOW:  Test lifecycle, compression prediction, primitive learning, and effects updates
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstring>

extern "C" {
#include "async/nimcp_semantic_compression_fep_bridge.h"
#include "cognitive/free_energy/nimcp_free_energy.h"
}

class SemanticCompressionFepBridgeTest : public ::testing::Test {
protected:
    semantic_compression_fep_bridge_t* bridge = nullptr;
    fep_system_t* fep = nullptr;
    nimcp_semantic_compressor_t* compressor = nullptr;

    void SetUp() override {
        /* Create FEP system */
        fep_config_t fep_config;
        fep_default_config(&fep_config);
        fep = fep_create(&fep_config, 8, 4);
        ASSERT_NE(fep, nullptr);

        /* Create semantic compressor */
        nimcp_compression_config_t comp_config = nimcp_semantic_compressor_default_config();
        compressor = nimcp_semantic_compressor_create(&comp_config);
        ASSERT_NE(compressor, nullptr);

        /* Create bridge */
        semantic_compression_fep_config_t config;
        semantic_compression_fep_default_config(&config);
        bridge = semantic_compression_fep_create(&config, fep, compressor);
    }

    void TearDown() override {
        if (bridge) {
            semantic_compression_fep_destroy(bridge);
            bridge = nullptr;
        }
        if (compressor) {
            nimcp_semantic_compressor_destroy(compressor);
            compressor = nullptr;
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

TEST_F(SemanticCompressionFepBridgeTest, CreateDestroy) {
    ASSERT_NE(bridge, nullptr);
}

TEST_F(SemanticCompressionFepBridgeTest, CreateWithNullConfig) {
    semantic_compression_fep_bridge_t* br =
        semantic_compression_fep_create(nullptr, fep, compressor);
    EXPECT_EQ(br, nullptr);
}

TEST_F(SemanticCompressionFepBridgeTest, CreateWithNullFep) {
    semantic_compression_fep_config_t config;
    semantic_compression_fep_default_config(&config);
    semantic_compression_fep_bridge_t* br =
        semantic_compression_fep_create(&config, nullptr, compressor);
    EXPECT_EQ(br, nullptr);
}

TEST_F(SemanticCompressionFepBridgeTest, CreateWithNullCompressor) {
    semantic_compression_fep_config_t config;
    semantic_compression_fep_default_config(&config);
    semantic_compression_fep_bridge_t* br =
        semantic_compression_fep_create(&config, fep, nullptr);
    EXPECT_EQ(br, nullptr);
}

TEST_F(SemanticCompressionFepBridgeTest, DestroyNull) {
    semantic_compression_fep_destroy(nullptr);  /* Should not crash */
}

TEST_F(SemanticCompressionFepBridgeTest, DefaultConfig) {
    semantic_compression_fep_config_t config;
    int ret = semantic_compression_fep_default_config(&config);

    EXPECT_EQ(ret, 0);
    EXPECT_GT(config.prediction_threshold, 0.0f);
    EXPECT_GT(config.semantic_surprise_threshold, 0.0f);
    EXPECT_TRUE(config.enable_predictive_compression);
    EXPECT_TRUE(config.enable_semantic_preservation);
    EXPECT_GT(config.max_primitives, 0u);
}

TEST_F(SemanticCompressionFepBridgeTest, DefaultConfigNullPtr) {
    int ret = semantic_compression_fep_default_config(nullptr);
    EXPECT_NE(ret, 0);
}

/* ============================================================================
 * Effects Update Tests
 * ============================================================================ */

TEST_F(SemanticCompressionFepBridgeTest, UpdateEffects) {
    int ret = semantic_compression_fep_update_effects(bridge);
    EXPECT_EQ(ret, 0);
}

TEST_F(SemanticCompressionFepBridgeTest, UpdateEffectsNull) {
    EXPECT_NE(semantic_compression_fep_update_effects(nullptr), 0);
}

TEST_F(SemanticCompressionFepBridgeTest, UpdateEffectsComputesPredictedCompressibility) {
    int ret = semantic_compression_fep_update_effects(bridge);
    EXPECT_EQ(ret, 0);

    semantic_compression_fep_effects_t effects;
    ret = semantic_compression_fep_get_effects(bridge, &effects);
    EXPECT_EQ(ret, 0);

    /* Predicted compressibility should be in valid range [0, 1] */
    EXPECT_GE(effects.predicted_compressibility, 0.0f);
    EXPECT_LE(effects.predicted_compressibility, 1.0f);
}

TEST_F(SemanticCompressionFepBridgeTest, UpdateEffectsSetsQualityModulation) {
    int ret = semantic_compression_fep_update_effects(bridge);
    EXPECT_EQ(ret, 0);

    semantic_compression_fep_effects_t effects;
    ret = semantic_compression_fep_get_effects(bridge, &effects);
    EXPECT_EQ(ret, 0);

    /* Quality modulation should be in range [0.5, 1.0] */
    EXPECT_GE(effects.quality_modulation, 0.5f);
    EXPECT_LE(effects.quality_modulation, 1.0f);
}

/* ============================================================================
 * Compression Observation Tests
 * ============================================================================ */

TEST_F(SemanticCompressionFepBridgeTest, ObserveCompression) {
    float ratio = 5.0f;
    float semantic_loss = 0.1f;

    int ret = semantic_compression_fep_observe_compression(bridge, ratio, semantic_loss);
    EXPECT_EQ(ret, 0);
}

TEST_F(SemanticCompressionFepBridgeTest, ObserveCompressionNull) {
    EXPECT_NE(semantic_compression_fep_observe_compression(nullptr, 5.0f, 0.1f), 0);
}

TEST_F(SemanticCompressionFepBridgeTest, ObserveCompressionUpdatesEffects) {
    float ratio = 5.0f;
    float semantic_loss = 0.1f;

    semantic_compression_fep_observe_compression(bridge, ratio, semantic_loss);

    fep_semantic_compression_effects_t effects;
    int ret = semantic_compression_fep_get_compression_effects(bridge, &effects);
    EXPECT_EQ(ret, 0);

    EXPECT_FLOAT_EQ(effects.achieved_compression_ratio, ratio);
    EXPECT_FLOAT_EQ(effects.semantic_loss_measured, semantic_loss);
}

TEST_F(SemanticCompressionFepBridgeTest, ObserveCompressionComputesPredictionError) {
    /* First update effects to set predictions */
    semantic_compression_fep_update_effects(bridge);

    /* Then observe actual compression */
    float ratio = 5.0f;
    float semantic_loss = 0.1f;
    semantic_compression_fep_observe_compression(bridge, ratio, semantic_loss);

    fep_semantic_compression_effects_t effects;
    semantic_compression_fep_get_compression_effects(bridge, &effects);

    /* Prediction error should be computed */
    /* The error can be positive or negative */
    EXPECT_NE(effects.compression_prediction_error, 0.0f);
}

TEST_F(SemanticCompressionFepBridgeTest, ObserveCompressionDetectsHighSemanticLoss) {
    /* Observe compression with high semantic loss */
    float ratio = 10.0f;
    float semantic_loss = 5.0f;  /* Above default threshold of 2.0 */

    semantic_compression_fep_observe_compression(bridge, ratio, semantic_loss);

    fep_semantic_compression_effects_t effects;
    semantic_compression_fep_get_compression_effects(bridge, &effects);

    EXPECT_TRUE(effects.high_semantic_loss_event);
}

/* ============================================================================
 * Compressibility Prediction Tests
 * ============================================================================ */

TEST_F(SemanticCompressionFepBridgeTest, PredictCompressibility) {
    float signal[] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
    float predicted_ratio = 0.0f;
    float confidence = 0.0f;

    int ret = semantic_compression_fep_predict_compressibility(
        bridge, signal, 5, &predicted_ratio, &confidence);
    EXPECT_EQ(ret, 0);

    EXPECT_GE(predicted_ratio, 1.0f);
    EXPECT_LE(predicted_ratio, 20.0f);
    EXPECT_GE(confidence, 0.0f);
    EXPECT_LE(confidence, 1.0f);
}

TEST_F(SemanticCompressionFepBridgeTest, PredictCompressibilityNull) {
    float signal[] = {0.1f, 0.2f, 0.3f};
    float predicted_ratio = 0.0f;
    float confidence = 0.0f;

    EXPECT_NE(semantic_compression_fep_predict_compressibility(
        nullptr, signal, 3, &predicted_ratio, &confidence), 0);
    EXPECT_NE(semantic_compression_fep_predict_compressibility(
        bridge, nullptr, 3, &predicted_ratio, &confidence), 0);
    EXPECT_NE(semantic_compression_fep_predict_compressibility(
        bridge, signal, 3, nullptr, &confidence), 0);
    EXPECT_NE(semantic_compression_fep_predict_compressibility(
        bridge, signal, 3, &predicted_ratio, nullptr), 0);
}

/* ============================================================================
 * Primitive Learning Tests
 * ============================================================================ */

TEST_F(SemanticCompressionFepBridgeTest, LearnPrimitive) {
    uint32_t primitive_id = 0;
    int ret = semantic_compression_fep_learn_primitive(bridge, 0, &primitive_id);
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(primitive_id, 0u);  /* First primitive ID should be 0 */
}

TEST_F(SemanticCompressionFepBridgeTest, LearnPrimitiveNull) {
    uint32_t primitive_id = 0;
    EXPECT_NE(semantic_compression_fep_learn_primitive(nullptr, 0, &primitive_id), 0);
    EXPECT_NE(semantic_compression_fep_learn_primitive(bridge, 0, nullptr), 0);
}

TEST_F(SemanticCompressionFepBridgeTest, LearnMultiplePrimitives) {
    uint32_t primitive_id1 = 0, primitive_id2 = 0;

    semantic_compression_fep_learn_primitive(bridge, 0, &primitive_id1);
    semantic_compression_fep_learn_primitive(bridge, 1, &primitive_id2);

    /* Primitive IDs should be different */
    EXPECT_NE(primitive_id1, primitive_id2);
}

/* ============================================================================
 * Bio-Async Integration Tests
 * ============================================================================ */

TEST_F(SemanticCompressionFepBridgeTest, InitiallyNotConnected) {
    EXPECT_FALSE(semantic_compression_fep_is_bio_async_connected(bridge));
}

TEST_F(SemanticCompressionFepBridgeTest, ConnectDisconnectBioAsync) {
    int ret = semantic_compression_fep_connect_bio_async(bridge);
    EXPECT_EQ(ret, 0);
    EXPECT_TRUE(semantic_compression_fep_is_bio_async_connected(bridge));

    ret = semantic_compression_fep_disconnect_bio_async(bridge);
    EXPECT_EQ(ret, 0);
    EXPECT_FALSE(semantic_compression_fep_is_bio_async_connected(bridge));
}

TEST_F(SemanticCompressionFepBridgeTest, ConnectNull) {
    EXPECT_NE(semantic_compression_fep_connect_bio_async(nullptr), 0);
}

TEST_F(SemanticCompressionFepBridgeTest, DisconnectNull) {
    EXPECT_NE(semantic_compression_fep_disconnect_bio_async(nullptr), 0);
}

TEST_F(SemanticCompressionFepBridgeTest, IsConnectedNull) {
    EXPECT_FALSE(semantic_compression_fep_is_bio_async_connected(nullptr));
}

/* ============================================================================
 * Query Tests
 * ============================================================================ */

TEST_F(SemanticCompressionFepBridgeTest, GetEffects) {
    semantic_compression_fep_effects_t effects;
    int ret = semantic_compression_fep_get_effects(bridge, &effects);

    EXPECT_EQ(ret, 0);
    EXPECT_GE(effects.compression_confidence, 0.0f);
}

TEST_F(SemanticCompressionFepBridgeTest, GetEffectsNull) {
    semantic_compression_fep_effects_t effects;

    EXPECT_NE(semantic_compression_fep_get_effects(nullptr, &effects), 0);
    EXPECT_NE(semantic_compression_fep_get_effects(bridge, nullptr), 0);
}

TEST_F(SemanticCompressionFepBridgeTest, GetCompressionEffects) {
    fep_semantic_compression_effects_t effects;
    int ret = semantic_compression_fep_get_compression_effects(bridge, &effects);

    EXPECT_EQ(ret, 0);
    EXPECT_GE(effects.achieved_compression_ratio, 0.0f);
}

TEST_F(SemanticCompressionFepBridgeTest, GetCompressionEffectsNull) {
    fep_semantic_compression_effects_t effects;

    EXPECT_NE(semantic_compression_fep_get_compression_effects(nullptr, &effects), 0);
    EXPECT_NE(semantic_compression_fep_get_compression_effects(bridge, nullptr), 0);
}

TEST_F(SemanticCompressionFepBridgeTest, GetStats) {
    semantic_compression_fep_stats_t stats;
    int ret = semantic_compression_fep_get_stats(bridge, &stats);

    EXPECT_EQ(ret, 0);
    EXPECT_GE(stats.avg_compression_ratio, 0.0f);
}

TEST_F(SemanticCompressionFepBridgeTest, GetStatsNull) {
    semantic_compression_fep_stats_t stats;

    EXPECT_NE(semantic_compression_fep_get_stats(nullptr, &stats), 0);
    EXPECT_NE(semantic_compression_fep_get_stats(bridge, nullptr), 0);
}

TEST_F(SemanticCompressionFepBridgeTest, ResetStats) {
    /* Observe some compressions */
    semantic_compression_fep_observe_compression(bridge, 5.0f, 0.1f);
    semantic_compression_fep_observe_compression(bridge, 10.0f, 0.2f);

    /* Reset stats */
    int ret = semantic_compression_fep_reset_stats(bridge);
    EXPECT_EQ(ret, 0);

    /* Verify stats are reset */
    semantic_compression_fep_stats_t stats;
    semantic_compression_fep_get_stats(bridge, &stats);

    EXPECT_FLOAT_EQ(stats.avg_compression_ratio, 0.0f);
    EXPECT_FLOAT_EQ(stats.avg_semantic_loss, 0.0f);
}

TEST_F(SemanticCompressionFepBridgeTest, ResetStatsNull) {
    EXPECT_NE(semantic_compression_fep_reset_stats(nullptr), 0);
}

/* ============================================================================
 * Integration Behavior Tests
 * ============================================================================ */

TEST_F(SemanticCompressionFepBridgeTest, SuccessfulCompressionsTracked) {
    /* Observe successful compression (ratio > 1.0, loss < 0.5) */
    semantic_compression_fep_observe_compression(bridge, 5.0f, 0.1f);
    semantic_compression_fep_observe_compression(bridge, 8.0f, 0.2f);

    /* Stats should show successful compressions */
    semantic_compression_fep_stats_t stats;
    semantic_compression_fep_get_stats(bridge, &stats);

    /* Average compression ratio should be computed */
    EXPECT_GT(stats.avg_compression_ratio, 0.0f);
}

TEST_F(SemanticCompressionFepBridgeTest, CompressionSurpriseComputed) {
    /* First update effects to establish predictions */
    semantic_compression_fep_update_effects(bridge);

    /* Observe compression */
    semantic_compression_fep_observe_compression(bridge, 5.0f, 0.1f);

    fep_semantic_compression_effects_t effects;
    semantic_compression_fep_get_compression_effects(bridge, &effects);

    /* Compression surprise should be computed based on prediction error */
    EXPECT_GE(effects.compression_surprise, 0.0f);
}
