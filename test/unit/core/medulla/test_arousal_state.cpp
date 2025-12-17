/**
 * @file test_arousal_state.cpp
 * @brief Unit tests for the arousal state module
 *
 * WHAT: Tests for arousal state management with hysteresis
 * WHY:  Ensure stable arousal transitions and state management
 * HOW:  Use GoogleTest framework with state transition validation
 */

#include <gtest/gtest.h>
#include <cstring>

extern "C" {
#include "core/medulla/nimcp_arousal_state.h"
#include "utils/error/nimcp_error_codes.h"
}

//=============================================================================
// Test Fixture
//=============================================================================

class ArousalStateTest : public ::testing::Test {
protected:
    arousal_state_t* arousal = nullptr;

    void SetUp() override {
        arousal_state_config_t config;
        arousal_state_default_config(&config);
        arousal = arousal_state_create(&config);
        ASSERT_NE(arousal, nullptr);
    }

    void TearDown() override {
        if (arousal) {
            arousal_state_destroy(arousal);
            arousal = nullptr;
        }
    }
};

//=============================================================================
// Configuration Tests
//=============================================================================

TEST_F(ArousalStateTest, DefaultConfig) {
    arousal_state_config_t config;
    int result = arousal_state_default_config(&config);
    EXPECT_EQ(result, NIMCP_SUCCESS);

    // Verify default values are set
    EXPECT_GT(config.hysteresis_margin, 0.0f);
    EXPECT_GT(config.min_dwell_time_ms, 0.0f);
    EXPECT_GT(config.max_rate_per_sec, 0.0f);
}

TEST_F(ArousalStateTest, CreateWithNullConfig) {
    // Should use defaults when config is NULL
    arousal_state_t* a = arousal_state_create(nullptr);
    EXPECT_NE(a, nullptr);
    if (a) arousal_state_destroy(a);
}

TEST_F(ArousalStateTest, DestroyNull) {
    // Should not crash with NULL
    arousal_state_destroy(nullptr);
}

//=============================================================================
// State Query Tests
//=============================================================================

TEST_F(ArousalStateTest, GetState) {
    arousal_state_enum_t state;
    int result = arousal_state_get_state(arousal, &state);
    EXPECT_EQ(result, NIMCP_SUCCESS);
    EXPECT_GE((int)state, 0);
    EXPECT_LT((int)state, (int)AROUSAL_STATE_COUNT);
}

TEST_F(ArousalStateTest, GetLevel) {
    float level;
    int result = arousal_state_get_level(arousal, &level);
    EXPECT_EQ(result, NIMCP_SUCCESS);
    EXPECT_GE(level, 0.0f);
    EXPECT_LE(level, 1.0f);
}

TEST_F(ArousalStateTest, GetStateNullOutput) {
    int result = arousal_state_get_state(arousal, nullptr);
    EXPECT_EQ(result, NIMCP_ERROR_NULL_POINTER);
}

TEST_F(ArousalStateTest, GetLevelNullOutput) {
    int result = arousal_state_get_level(arousal, nullptr);
    EXPECT_EQ(result, NIMCP_ERROR_NULL_POINTER);
}

//=============================================================================
// Stimulus Tests
//=============================================================================

TEST_F(ArousalStateTest, ApplyStimulus) {
    float initial;
    arousal_state_get_level(arousal, &initial);

    // Apply positive stimulus
    int result = arousal_state_apply_stimulus(arousal, 0.2f);
    EXPECT_EQ(result, NIMCP_SUCCESS);
}

TEST_F(ArousalStateTest, ApplyStimulusNullState) {
    int result = arousal_state_apply_stimulus(nullptr, 0.2f);
    EXPECT_EQ(result, NIMCP_ERROR_NULL_POINTER);
}

//=============================================================================
// Target Level Tests
//=============================================================================

TEST_F(ArousalStateTest, SetTarget) {
    int result = arousal_state_set_target(arousal, 0.7f);
    EXPECT_EQ(result, NIMCP_SUCCESS);
}

TEST_F(ArousalStateTest, SetTargetClamped) {
    // Set out of range target - function rejects invalid values
    int result = arousal_state_set_target(arousal, 1.5f);
    EXPECT_EQ(result, NIMCP_ERROR_INVALID_PARAMETER);  // Rejected, not clamped
}

TEST_F(ArousalStateTest, SetTargetNullState) {
    int result = arousal_state_set_target(nullptr, 0.5f);
    EXPECT_EQ(result, NIMCP_ERROR_NULL_POINTER);
}

//=============================================================================
// Update Tests
//=============================================================================

TEST_F(ArousalStateTest, Update) {
    int result = arousal_state_update(arousal, 100.0f);  // 100ms delta
    EXPECT_EQ(result, NIMCP_SUCCESS);
}

TEST_F(ArousalStateTest, UpdateNullState) {
    int result = arousal_state_update(nullptr, 100.0f);
    EXPECT_EQ(result, NIMCP_ERROR_NULL_POINTER);
}

//=============================================================================
// State Name Tests
//=============================================================================

TEST_F(ArousalStateTest, StateNames) {
    const char* name = arousal_state_get_state_name(AROUSAL_STATE_COMA);
    EXPECT_NE(name, nullptr);
    EXPECT_STRNE(name, "");

    name = arousal_state_get_state_name(AROUSAL_STATE_PANIC);
    EXPECT_NE(name, nullptr);
    EXPECT_STRNE(name, "");

    name = arousal_state_get_state_name(AROUSAL_STATE_RELAXED);
    EXPECT_NE(name, nullptr);
    EXPECT_STRNE(name, "");
}

//=============================================================================
// Bio-async Tests
//=============================================================================

TEST_F(ArousalStateTest, BioAsyncConnection) {
    // Initially not connected
    bool connected = arousal_state_is_bio_async_connected(arousal);
    EXPECT_FALSE(connected);

    // Try to connect (may fail if router not available)
    int result = arousal_state_connect_bio_async(arousal);
    // Result depends on whether bio-async router is initialized

    // Disconnect
    arousal_state_disconnect_bio_async(arousal);
    connected = arousal_state_is_bio_async_connected(arousal);
    EXPECT_FALSE(connected);
}

TEST_F(ArousalStateTest, BioAsyncNullState) {
    bool connected = arousal_state_is_bio_async_connected(nullptr);
    EXPECT_FALSE(connected);
}

//=============================================================================
// Main
//=============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
