/**
 * @file test_sensory_processing_regression.cpp
 * @brief Regression tests for Phase 6 Sensory Processing modules
 *
 * TEST PHILOSOPHY:
 * - Test API stability and backward compatibility
 * - Test consistent behavior across module updates
 * - Test serialization format stability
 * - Test performance characteristics remain stable
 * - Test error handling consistency
 *
 * @author NIMCP Development Team
 * @date 2026-01-12
 * @version 1.0.0 Phase 6 Sensory Processing Regression
 */

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <chrono>
#include <cstring>

/* Somatosensory module */
extern "C" {
#include "core/brain/regions/somatosensory/nimcp_somatosensory.h"
}

/* Olfactory module */
extern "C" {
#include "core/brain/regions/olfactory/nimcp_olfactory.h"
}

/* Gustatory module */
extern "C" {
#include "core/brain/regions/gustatory/nimcp_gustatory.h"
}

//=============================================================================
// Somatosensory API Stability Tests
//=============================================================================

class SomatosensoryAPIStabilityTest : public ::testing::Test {
protected:
    nimcp_somatosensory_t* soma = nullptr;

    void SetUp() override {
        soma = soma_create(nullptr);
    }

    void TearDown() override {
        if (soma) soma_destroy(soma);
    }
};

TEST_F(SomatosensoryAPIStabilityTest, DefaultConfig_ValuesStable) {
    soma_config_t config = soma_default_config();

    /* These values should remain stable across versions */
    EXPECT_EQ(config.num_area_3a_neurons, SOMA_DEFAULT_AREA_3A_NEURONS);
    EXPECT_EQ(config.num_area_3b_neurons, SOMA_DEFAULT_AREA_3B_NEURONS);
    EXPECT_EQ(config.num_area_1_neurons, SOMA_DEFAULT_AREA_1_NEURONS);
    EXPECT_EQ(config.num_area_2_neurons, SOMA_DEFAULT_AREA_2_NEURONS);
    EXPECT_EQ(config.num_s2_neurons, SOMA_DEFAULT_S2_NEURONS);
    EXPECT_FLOAT_EQ(config.pain_threshold, SOMA_PAIN_THRESHOLD);
}

TEST_F(SomatosensoryAPIStabilityTest, CreateDestroy_NoMemoryLeak) {
    /* Create and destroy multiple times */
    for (int i = 0; i < 100; i++) {
        nimcp_somatosensory_t* temp = soma_create(nullptr);
        ASSERT_NE(temp, nullptr);
        soma_destroy(temp);
    }
}

TEST_F(SomatosensoryAPIStabilityTest, StatusValues_StableEnumeration) {
    ASSERT_NE(soma, nullptr);

    /* Verify status enum values are stable */
    EXPECT_EQ((int)SOMA_STATUS_IDLE, 0);
    EXPECT_EQ((int)SOMA_STATUS_READY, 1);
    EXPECT_EQ((int)SOMA_STATUS_PROCESSING_TOUCH, 2);
    EXPECT_EQ((int)SOMA_STATUS_PROCESSING_PAIN, 3);
    EXPECT_EQ((int)SOMA_STATUS_ERROR, 8);
}

TEST_F(SomatosensoryAPIStabilityTest, ErrorCodes_StableEnumeration) {
    /* Verify error enum values are stable */
    EXPECT_EQ((int)SOMA_ERROR_NONE, 0);
    EXPECT_EQ((int)SOMA_ERROR_INVALID_INPUT, 1);
    EXPECT_EQ((int)SOMA_ERROR_INTERNAL, 8);
}

TEST_F(SomatosensoryAPIStabilityTest, BodySegments_StableEnumeration) {
    /* Critical body segment enum values */
    EXPECT_EQ((int)BODY_SEG_HEAD, 0);
    EXPECT_EQ((int)BODY_SEG_HAND_R, 16);
    EXPECT_EQ((int)BODY_SEG_INDEX_R, 20);
    EXPECT_EQ((int)BODY_SEG_COUNT, 45);
}

TEST_F(SomatosensoryAPIStabilityTest, TouchProcessing_ConsistentBehavior) {
    ASSERT_NE(soma, nullptr);

    float position[3] = {0.5f, 0.5f, 0.0f};
    uint32_t event_id;

    /* Touch processing should always succeed with valid input */
    EXPECT_EQ(soma_process_touch(soma, BODY_SEG_INDEX_R, position, 0.5f, TOUCH_LIGHT, &event_id), 0);

    /* Event ID should be assigned */
    EXPECT_GE(event_id, 0u);

    /* Status should change to ready or processing */
    soma_status_t status = soma_get_status(soma);
    EXPECT_TRUE(status == SOMA_STATUS_READY || status == SOMA_STATUS_PROCESSING_TOUCH);
}

TEST_F(SomatosensoryAPIStabilityTest, PainProcessing_ConsistentBehavior) {
    ASSERT_NE(soma, nullptr);

    uint32_t event_id;

    /* Pain processing should always succeed with valid input */
    EXPECT_EQ(soma_process_pain(soma, BODY_SEG_HAND_L, PAIN_SHARP, 0.7f, &event_id), 0);

    /* Pain level should be trackable */
    float pain = soma_get_pain_level(soma, BODY_SEG_HAND_L);
    EXPECT_GE(pain, 0.0f);
}

TEST_F(SomatosensoryAPIStabilityTest, ProprioceptionUpdate_ConsistentBehavior) {
    ASSERT_NE(soma, nullptr);

    float pos[3] = {0.0f, 1.0f, 0.0f};
    float vel[3] = {0.1f, 0.0f, 0.0f};

    EXPECT_EQ(soma_update_proprioception(soma, BODY_SEG_UPPER_ARM_R, pos, vel, 0.5f, 0.6f), 0);

    soma_proprio_state_t state;
    EXPECT_EQ(soma_get_proprioception(soma, BODY_SEG_UPPER_ARM_R, &state), 0);
    EXPECT_GT(state.confidence, 0.0f);
}

TEST_F(SomatosensoryAPIStabilityTest, Reset_ClearsState) {
    ASSERT_NE(soma, nullptr);

    /* Add some events */
    float pos[3] = {0.5f, 0.5f, 0.0f};
    uint32_t event_id;
    soma_process_touch(soma, BODY_SEG_INDEX_R, pos, 0.7f, TOUCH_LIGHT, &event_id);
    soma_process_pain(soma, BODY_SEG_HAND_L, PAIN_SHARP, 0.5f, &event_id);
    soma_update(soma, 0.01f);

    /* Reset */
    EXPECT_EQ(soma_reset(soma), 0);

    /* Counters should be cleared */
    EXPECT_EQ(soma->updates_processed, 0u);
    EXPECT_EQ(soma->touch_events_total, 0u);
    EXPECT_EQ(soma->pain_events_total, 0u);
}

//=============================================================================
// Olfactory API Stability Tests
//=============================================================================

class OlfactoryAPIStabilityTest : public ::testing::Test {
protected:
    nimcp_olfactory_t* olfact = nullptr;

    void SetUp() override {
        olfact = olfact_create(nullptr);
    }

    void TearDown() override {
        if (olfact) olfact_destroy(olfact);
    }
};

TEST_F(OlfactoryAPIStabilityTest, DefaultConfig_ValuesStable) {
    olfact_config_t config = olfact_default_config();

    EXPECT_EQ(config.num_mitral_cells, OLFACT_DEFAULT_MITRAL_CELLS);
    EXPECT_EQ(config.num_piriform_neurons, OLFACT_DEFAULT_PIRIFORM);
    EXPECT_GT(config.max_stored_odors, 0u);
}

TEST_F(OlfactoryAPIStabilityTest, CreateDestroy_NoMemoryLeak) {
    for (int i = 0; i < 100; i++) {
        nimcp_olfactory_t* temp = olfact_create(nullptr);
        ASSERT_NE(temp, nullptr);
        olfact_destroy(temp);
    }
}

TEST_F(OlfactoryAPIStabilityTest, StatusValues_StableEnumeration) {
    EXPECT_EQ((int)OLFACT_STATUS_IDLE, 0);
    EXPECT_EQ((int)OLFACT_STATUS_READY, 1);
    EXPECT_EQ((int)OLFACT_STATUS_SNIFFING, 2);
    EXPECT_EQ((int)OLFACT_STATUS_ERROR, 6);
}

TEST_F(OlfactoryAPIStabilityTest, OdorCategories_StableEnumeration) {
    EXPECT_EQ((int)ODOR_CAT_UNKNOWN, 0);
    EXPECT_EQ((int)ODOR_CAT_FLORAL, 1);
    EXPECT_EQ((int)ODOR_CAT_DECAYED, 11);
    EXPECT_EQ((int)ODOR_CAT_COUNT, 13);
}

TEST_F(OlfactoryAPIStabilityTest, HedonicValence_StableEnumeration) {
    EXPECT_EQ((int)HEDONIC_VERY_UNPLEASANT, 0);
    EXPECT_EQ((int)HEDONIC_NEUTRAL, 3);
    EXPECT_EQ((int)HEDONIC_VERY_PLEASANT, 6);
}

TEST_F(OlfactoryAPIStabilityTest, OdorProcessing_ConsistentBehavior) {
    ASSERT_NE(olfact, nullptr);

    float pattern[OLFACT_MAX_RECEPTORS];
    for (uint32_t i = 0; i < OLFACT_MAX_RECEPTORS; i++) {
        pattern[i] = (float)(i % 50) / 50.0f;
    }

    EXPECT_EQ(olfact_process_odor(olfact, pattern, OLFACT_MAX_RECEPTORS, 0.6f), 0);

    float intensity = olfact_get_intensity(olfact);
    EXPECT_GE(intensity, 0.0f);
    EXPECT_LE(intensity, 1.0f);
}

TEST_F(OlfactoryAPIStabilityTest, OdorComparison_SymmetricAndBounded) {
    ASSERT_NE(olfact, nullptr);

    float pattern1[256], pattern2[256];
    for (uint32_t i = 0; i < 256; i++) {
        pattern1[i] = (float)i / 256.0f;
        pattern2[i] = 1.0f - (float)i / 256.0f;
    }

    /* Compare pattern1 vs pattern2 */
    float sim12 = olfact_compare_odors(olfact, pattern1, pattern2, 256);

    /* Compare pattern2 vs pattern1 (should be symmetric) */
    float sim21 = olfact_compare_odors(olfact, pattern2, pattern1, 256);

    /* Similarity should be bounded [0, 1] */
    EXPECT_GE(sim12, 0.0f);
    EXPECT_LE(sim12, 1.0f);

    /* Symmetric property */
    EXPECT_FLOAT_EQ(sim12, sim21);

    /* Self-similarity should be 1.0 */
    float self_sim = olfact_compare_odors(olfact, pattern1, pattern1, 256);
    EXPECT_FLOAT_EQ(self_sim, 1.0f);
}

TEST_F(OlfactoryAPIStabilityTest, Reset_ClearsState) {
    ASSERT_NE(olfact, nullptr);

    float pattern[OLFACT_MAX_RECEPTORS];
    for (uint32_t i = 0; i < OLFACT_MAX_RECEPTORS; i++) pattern[i] = 0.5f;
    olfact_process_odor(olfact, pattern, OLFACT_MAX_RECEPTORS, 0.6f);
    olfact_update(olfact, 0.01f);

    EXPECT_EQ(olfact_reset(olfact), 0);

    /* Verify state cleared */
    EXPECT_EQ(olfact->updates_processed, 0u);
    EXPECT_EQ(olfact->sniff_state.cycle_count, 0u);
}

//=============================================================================
// Gustatory API Stability Tests
//=============================================================================

class GustatoryAPIStabilityTest : public ::testing::Test {
protected:
    nimcp_gustatory_t* gust = nullptr;

    void SetUp() override {
        gust = gust_create(nullptr);
    }

    void TearDown() override {
        if (gust) gust_destroy(gust);
    }
};

TEST_F(GustatoryAPIStabilityTest, DefaultConfig_ValuesStable) {
    gust_config_t config = gust_default_config();

    EXPECT_EQ(config.num_insula_neurons, GUST_DEFAULT_INSULA_NEURONS);
    EXPECT_EQ(config.num_ofc_neurons, GUST_DEFAULT_OFC_NEURONS);
}

TEST_F(GustatoryAPIStabilityTest, CreateDestroy_NoMemoryLeak) {
    for (int i = 0; i < 100; i++) {
        nimcp_gustatory_t* temp = gust_create(nullptr);
        ASSERT_NE(temp, nullptr);
        gust_destroy(temp);
    }
}

TEST_F(GustatoryAPIStabilityTest, TasteTypes_StableEnumeration) {
    EXPECT_EQ((int)TASTE_SWEET, 0);
    EXPECT_EQ((int)TASTE_SALTY, 1);
    EXPECT_EQ((int)TASTE_SOUR, 2);
    EXPECT_EQ((int)TASTE_BITTER, 3);
    EXPECT_EQ((int)TASTE_UMAMI, 4);
    EXPECT_EQ((int)TASTE_COUNT, 5);
}

TEST_F(GustatoryAPIStabilityTest, DisgustLevels_StableEnumeration) {
    EXPECT_EQ((int)DISGUST_NONE, 0);
    EXPECT_EQ((int)DISGUST_MILD, 1);
    EXPECT_EQ((int)DISGUST_EXTREME, 4);
}

TEST_F(GustatoryAPIStabilityTest, StatusValues_StableEnumeration) {
    EXPECT_EQ((int)GUST_STATUS_IDLE, 0);
    EXPECT_EQ((int)GUST_STATUS_READY, 1);
    EXPECT_EQ((int)GUST_STATUS_TASTING, 2);
    EXPECT_EQ((int)GUST_STATUS_ERROR, 5);
}

TEST_F(GustatoryAPIStabilityTest, TasteProcessing_ConsistentBehavior) {
    ASSERT_NE(gust, nullptr);

    taste_stimulus_t stimulus = {};
    stimulus.sweet = 0.7f;
    stimulus.salty = 0.2f;
    stimulus.temperature = 25.0f;

    EXPECT_EQ(gust_process_taste(gust, &stimulus), 0);

    float palatability = gust_get_palatability(gust);
    EXPECT_GE(palatability, 0.0f);
    EXPECT_LE(palatability, 1.0f);
}

TEST_F(GustatoryAPIStabilityTest, LearnedPreference_BoundedChange) {
    ASSERT_NE(gust, nullptr);

    /* Learn preference */
    EXPECT_EQ(gust_learn_preference(gust, TASTE_SWEET, 0.5f), 0);

    /* Preference should be bounded */
    float pref = gust->learned_preferences[TASTE_SWEET];
    EXPECT_GE(pref, -1.0f);
    EXPECT_LE(pref, 1.0f);
}

TEST_F(GustatoryAPIStabilityTest, Reset_ClearsState) {
    ASSERT_NE(gust, nullptr);

    taste_stimulus_t stimulus = {};
    stimulus.sweet = 0.7f;
    gust_process_taste(gust, &stimulus);
    gust_update(gust, 0.01f);

    EXPECT_EQ(gust_reset(gust), 0);

    EXPECT_EQ(gust->updates_processed, 0u);
}

//=============================================================================
// Cross-Module Backward Compatibility Tests
//=============================================================================

class SensoryBackwardCompatTest : public ::testing::Test {
protected:
    nimcp_somatosensory_t* soma = nullptr;
    nimcp_olfactory_t* olfact = nullptr;
    nimcp_gustatory_t* gust = nullptr;

    void SetUp() override {
        soma = soma_create(nullptr);
        olfact = olfact_create(nullptr);
        gust = gust_create(nullptr);
    }

    void TearDown() override {
        if (soma) soma_destroy(soma);
        if (olfact) olfact_destroy(olfact);
        if (gust) gust_destroy(gust);
    }
};

TEST_F(SensoryBackwardCompatTest, NullConfig_CreatesWithDefaults) {
    /* All modules should accept NULL config */
    EXPECT_NE(soma, nullptr);
    EXPECT_NE(olfact, nullptr);
    EXPECT_NE(gust, nullptr);

    /* And be in ready state */
    EXPECT_EQ(soma_get_status(soma), SOMA_STATUS_READY);
    EXPECT_EQ(olfact_get_status(olfact), OLFACT_STATUS_READY);
    EXPECT_EQ(gust_get_status(gust), GUST_STATUS_READY);
}

TEST_F(SensoryBackwardCompatTest, BridgeInit_NullContext_Succeeds) {
    /* All bridges should accept NULL context */
    EXPECT_EQ(soma_init_thalamus_bridge(soma, nullptr), 0);
    EXPECT_EQ(soma_init_motor_bridge(soma, nullptr), 0);
    EXPECT_EQ(soma_init_parietal_bridge(soma, nullptr), 0);

    EXPECT_EQ(olfact_init_amygdala_bridge(olfact, nullptr), 0);
    EXPECT_EQ(olfact_init_entorhinal_bridge(olfact, nullptr), 0);

    EXPECT_EQ(gust_init_hypothalamus_bridge(gust, nullptr), 0);
    EXPECT_EQ(gust_init_amygdala_bridge(gust, nullptr), 0);
}

TEST_F(SensoryBackwardCompatTest, SyncFunctions_NonInitialized_NoOp) {
    /* Sync functions should return 0 (no-op) when bridges not initialized */
    EXPECT_EQ(soma_sync_thalamus(soma), 0);
    EXPECT_EQ(soma_sync_motor_cortex(soma), 0);
    EXPECT_EQ(soma_sync_parietal(soma), 0);

    EXPECT_EQ(olfact_sync_amygdala(olfact), 0);
    EXPECT_EQ(olfact_sync_entorhinal(olfact), 0);

    EXPECT_EQ(gust_sync_hypothalamus(gust), 0);
    EXPECT_EQ(gust_sync_olfactory(gust), 0);
}

TEST_F(SensoryBackwardCompatTest, ErrorStrings_NotNull) {
    /* Error string functions should never return NULL */
    EXPECT_NE(soma_error_string(SOMA_ERROR_NONE), nullptr);
    EXPECT_NE(soma_error_string(SOMA_ERROR_INVALID_INPUT), nullptr);
    EXPECT_NE(soma_status_string(SOMA_STATUS_READY), nullptr);

    EXPECT_NE(olfact_error_string(OLFACT_ERROR_NONE), nullptr);
    EXPECT_NE(olfact_status_string(OLFACT_STATUS_READY), nullptr);

    EXPECT_NE(gust_error_string(GUST_ERROR_NONE), nullptr);
    EXPECT_NE(gust_status_string(GUST_STATUS_READY), nullptr);
}

TEST_F(SensoryBackwardCompatTest, UtilityStrings_NotNull) {
    /* Utility name functions should never return NULL */
    EXPECT_NE(soma_segment_name(BODY_SEG_HAND_R), nullptr);
    EXPECT_NE(soma_receptor_type_name(RECEPTOR_MEISSNER), nullptr);
    EXPECT_NE(soma_area_name(SOMA_AREA_3A), nullptr);
    EXPECT_NE(soma_pain_type_name(PAIN_SHARP), nullptr);

    EXPECT_NE(olfact_category_name(ODOR_CAT_FLORAL), nullptr);
    EXPECT_NE(olfact_valence_name(HEDONIC_PLEASANT), nullptr);
    EXPECT_NE(olfact_sniff_phase_name(SNIFF_PHASE_INSPIRATION), nullptr);

    EXPECT_NE(gust_taste_name(TASTE_SWEET), nullptr);
    EXPECT_NE(gust_food_category_name(FOOD_CAT_FRUIT), nullptr);
    EXPECT_NE(gust_disgust_name(DISGUST_MILD), nullptr);
}

//=============================================================================
// Serialization Format Stability Tests
//=============================================================================

TEST_F(SensoryBackwardCompatTest, Somatosensory_SerializationSize_Deterministic) {
    /* Same config should produce same serialization size */
    nimcp_somatosensory_t* soma2 = soma_create(nullptr);

    size_t size1 = soma_get_serialization_size(soma);
    size_t size2 = soma_get_serialization_size(soma2);

    EXPECT_EQ(size1, size2);

    soma_destroy(soma2);
}

TEST_F(SensoryBackwardCompatTest, Olfactory_SerializationSize_Deterministic) {
    nimcp_olfactory_t* olfact2 = olfact_create(nullptr);

    size_t size1 = olfact_get_serialization_size(olfact);
    size_t size2 = olfact_get_serialization_size(olfact2);

    EXPECT_EQ(size1, size2);

    olfact_destroy(olfact2);
}

TEST_F(SensoryBackwardCompatTest, Gustatory_SerializationSize_Deterministic) {
    nimcp_gustatory_t* gust2 = gust_create(nullptr);

    size_t size1 = gust_get_serialization_size(gust);
    size_t size2 = gust_get_serialization_size(gust2);

    EXPECT_EQ(size1, size2);

    gust_destroy(gust2);
}

TEST_F(SensoryBackwardCompatTest, Serialization_RoundTrip_Stable) {
    /* Serialize somatosensory */
    size_t soma_size = soma_get_serialization_size(soma);
    std::vector<uint8_t> soma_buf(soma_size);
    size_t soma_written;
    EXPECT_EQ(soma_serialize(soma, soma_buf.data(), soma_size, &soma_written), 0);
    EXPECT_GT(soma_written, 0u);

    size_t soma_read;
    nimcp_somatosensory_t* soma_restored = soma_deserialize(soma_buf.data(), soma_written, &soma_read);
    ASSERT_NE(soma_restored, nullptr);
    /* Verify restored object is functional */
    EXPECT_EQ(soma_reset(soma_restored), 0);
    soma_destroy(soma_restored);

    /* Serialize olfactory */
    size_t olfact_size = olfact_get_serialization_size(olfact);
    std::vector<uint8_t> olfact_buf(olfact_size);
    size_t olfact_written;
    EXPECT_EQ(olfact_serialize(olfact, olfact_buf.data(), olfact_size, &olfact_written), 0);
    EXPECT_GT(olfact_written, 0u);

    size_t olfact_read;
    nimcp_olfactory_t* olfact_restored = olfact_deserialize(olfact_buf.data(), olfact_written, &olfact_read);
    ASSERT_NE(olfact_restored, nullptr);
    /* Verify restored object is functional */
    EXPECT_EQ(olfact_reset(olfact_restored), 0);
    olfact_destroy(olfact_restored);

    /* Serialize gustatory */
    size_t gust_size = gust_get_serialization_size(gust);
    std::vector<uint8_t> gust_buf(gust_size);
    size_t gust_written;
    EXPECT_EQ(gust_serialize(gust, gust_buf.data(), gust_size, &gust_written), 0);
    EXPECT_GT(gust_written, 0u);

    size_t gust_read;
    nimcp_gustatory_t* gust_restored = gust_deserialize(gust_buf.data(), gust_written, &gust_read);
    ASSERT_NE(gust_restored, nullptr);
    /* Verify restored object is functional */
    EXPECT_EQ(gust_reset(gust_restored), 0);
    gust_destroy(gust_restored);
}

//=============================================================================
// Performance Regression Tests
//=============================================================================

TEST_F(SomatosensoryAPIStabilityTest, TouchProcessing_PerformanceBaseline) {
    ASSERT_NE(soma, nullptr);

    float position[3] = {0.5f, 0.5f, 0.0f};

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 1000; i++) {
        uint32_t event_id;
        soma_process_touch(soma, BODY_SEG_INDEX_R, position, 0.5f, TOUCH_LIGHT, &event_id);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    /* 1000 touch events should complete in under 100ms */
    EXPECT_LT(duration.count(), 100000);
}

TEST_F(OlfactoryAPIStabilityTest, OdorProcessing_PerformanceBaseline) {
    ASSERT_NE(olfact, nullptr);

    float pattern[OLFACT_MAX_RECEPTORS];
    for (uint32_t i = 0; i < OLFACT_MAX_RECEPTORS; i++) pattern[i] = 0.5f;

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 1000; i++) {
        olfact_process_odor(olfact, pattern, OLFACT_MAX_RECEPTORS, 0.5f);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    /* 1000 odor processing calls should complete in under 100ms */
    EXPECT_LT(duration.count(), 100000);
}

TEST_F(GustatoryAPIStabilityTest, TasteProcessing_PerformanceBaseline) {
    ASSERT_NE(gust, nullptr);

    taste_stimulus_t stimulus = {};
    stimulus.sweet = 0.5f;
    stimulus.salty = 0.3f;

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 1000; i++) {
        gust_process_taste(gust, &stimulus);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    /* 1000 taste processing calls should complete in under 100ms */
    EXPECT_LT(duration.count(), 100000);
}

//=============================================================================
// Error Handling Consistency Tests
//=============================================================================

TEST_F(SomatosensoryAPIStabilityTest, NullInput_ConsistentErrorHandling) {
    /* Functions should handle NULL input consistently */
    soma_destroy(nullptr);  /* Should not crash */
    EXPECT_LT(soma_reset(nullptr), 0);
    EXPECT_LT(soma_update(nullptr, 0.01f), 0);

    soma_status_t status = soma_get_status(nullptr);
    EXPECT_EQ(status, SOMA_STATUS_ERROR);

    float pain = soma_get_pain_level(nullptr, BODY_SEG_HAND_R);
    EXPECT_EQ(pain, 0.0f);
}

TEST_F(OlfactoryAPIStabilityTest, NullInput_ConsistentErrorHandling) {
    olfact_destroy(nullptr);  /* Should not crash */
    EXPECT_LT(olfact_reset(nullptr), 0);
    EXPECT_LT(olfact_update(nullptr, 0.01f), 0);

    olfact_status_t status = olfact_get_status(nullptr);
    EXPECT_EQ(status, OLFACT_STATUS_ERROR);

    float intensity = olfact_get_intensity(nullptr);
    EXPECT_EQ(intensity, 0.0f);

    float modulation = olfact_get_sniff_modulation(nullptr);
    EXPECT_EQ(modulation, 0.0f);
}

TEST_F(GustatoryAPIStabilityTest, NullInput_ConsistentErrorHandling) {
    gust_destroy(nullptr);  /* Should not crash */
    EXPECT_LT(gust_reset(nullptr), 0);
    EXPECT_LT(gust_update(nullptr, 0.01f), 0);

    gust_status_t status = gust_get_status(nullptr);
    EXPECT_EQ(status, GUST_STATUS_ERROR);

    float palatability = gust_get_palatability(nullptr);
    EXPECT_EQ(palatability, 0.0f);
}
