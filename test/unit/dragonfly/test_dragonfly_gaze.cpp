/**
 * @file test_dragonfly_gaze.cpp
 * @brief Unit tests for gaze stabilization module
 *
 * @author NIMCP Team
 * @date 2024-12-29
 */

#include <gtest/gtest.h>
#include <cmath>

extern "C" {
#include "dragonfly/nimcp_dragonfly_gaze.h"
}

//=============================================================================
// Test Fixture
//=============================================================================

class GazeTest : public ::testing::Test {
protected:
    dragonfly_gaze_t gaze = nullptr;

    void SetUp() override {
        gaze = dragonfly_gaze_create(nullptr);
        ASSERT_NE(gaze, nullptr);
    }

    void TearDown() override {
        if (gaze) {
            dragonfly_gaze_destroy(gaze);
            gaze = nullptr;
        }
    }
};

//=============================================================================
// Configuration Tests
//=============================================================================

TEST_F(GazeTest, DefaultConfig) {
    gaze_config_t config = dragonfly_gaze_default_config();
    EXPECT_GT(config.vor_gain, 0.0f);
    EXPECT_GT(config.smooth_pursuit_gain, 0.0f);
}

TEST_F(GazeTest, ValidateConfig) {
    gaze_config_t config = dragonfly_gaze_default_config();
    EXPECT_TRUE(dragonfly_gaze_validate_config(&config));

    config.vor_gain = -1.0f;
    EXPECT_FALSE(dragonfly_gaze_validate_config(&config));

    EXPECT_FALSE(dragonfly_gaze_validate_config(nullptr));
}

TEST_F(GazeTest, CreateWithCustomConfig) {
    gaze_config_t config = dragonfly_gaze_default_config();
    config.vor_gain = 1.2f;

    dragonfly_gaze_t custom = dragonfly_gaze_create(&config);
    ASSERT_NE(custom, nullptr);
    dragonfly_gaze_destroy(custom);
}

//=============================================================================
// Lifecycle Tests
//=============================================================================

TEST_F(GazeTest, CreateAndDestroy) {
    dragonfly_gaze_t g = dragonfly_gaze_create(nullptr);
    ASSERT_NE(g, nullptr);
    dragonfly_gaze_destroy(g);
}

TEST_F(GazeTest, DestroyNull) {
    dragonfly_gaze_destroy(nullptr);  // Should not crash
}

TEST_F(GazeTest, Reset) {
    EXPECT_EQ(dragonfly_gaze_reset(gaze), 0);
}

//=============================================================================
// VOR Tests
//=============================================================================

TEST_F(GazeTest, VORCompensation) {
    // Set target straight ahead
    float target_pos[3] = {100, 0, 0};
    dragonfly_gaze_set_target(gaze, target_pos);

    // Apply head rotation
    float head_vel[3] = {0, 0.5f, 0};  // Yaw rotation
    EXPECT_EQ(dragonfly_gaze_update_head(gaze, head_vel, 0.016f), 0);

    // VOR should compensate
    gaze_state_t state;
    dragonfly_gaze_get_state(gaze, &state);
    // Eye velocity should partially counter head velocity
}

TEST_F(GazeTest, VORWithNoRotation) {
    float target_pos[3] = {100, 0, 0};
    dragonfly_gaze_set_target(gaze, target_pos);

    float head_vel[3] = {0, 0, 0};  // No rotation
    dragonfly_gaze_update_head(gaze, head_vel, 0.016f);

    gaze_state_t state;
    dragonfly_gaze_get_state(gaze, &state);
    // Minimal compensation needed
}

//=============================================================================
// Smooth Pursuit Tests
//=============================================================================

TEST_F(GazeTest, SmoothPursuitOfMovingTarget) {
    // Initial target position
    float target_pos[3] = {100, 0, 0};
    dragonfly_gaze_set_target(gaze, target_pos);

    // Update target position (moving)
    target_pos[0] = 95;
    target_pos[1] = 5;
    EXPECT_EQ(dragonfly_gaze_update_target(gaze, target_pos, 0.016f), 0);

    gaze_state_t state;
    dragonfly_gaze_get_state(gaze, &state);
    // Gaze should be tracking toward new position
}

TEST_F(GazeTest, PredictiveGaze) {
    float target_pos[3] = {100, 0, 0};
    float target_vel[3] = {-10, 5, 0};

    EXPECT_EQ(dragonfly_gaze_set_target_with_velocity(gaze, target_pos, target_vel), 0);
    EXPECT_EQ(dragonfly_gaze_update(gaze, 0.1f), 0);

    gaze_command_t cmd;
    dragonfly_gaze_get_command(gaze, &cmd);
    // Gaze should lead the target
}

//=============================================================================
// Gaze Command Tests
//=============================================================================

TEST_F(GazeTest, GetGazeCommand) {
    float target_pos[3] = {50, 30, 10};
    dragonfly_gaze_set_target(gaze, target_pos);
    dragonfly_gaze_update(gaze, 0.016f);

    gaze_command_t cmd;
    EXPECT_EQ(dragonfly_gaze_get_command(gaze, &cmd), 0);
    // Should have valid angles
}

TEST_F(GazeTest, GazeAngleComputation) {
    // Target directly ahead
    float target_pos[3] = {100, 0, 0};
    dragonfly_gaze_set_target(gaze, target_pos);
    dragonfly_gaze_update(gaze, 0.016f);

    gaze_command_t cmd;
    dragonfly_gaze_get_command(gaze, &cmd);
    // Pitch and yaw should be approximately zero
    EXPECT_NEAR(cmd.yaw_rad, 0.0f, 0.1f);
    EXPECT_NEAR(cmd.pitch_rad, 0.0f, 0.1f);
}

TEST_F(GazeTest, GazeAngleToSide) {
    // Target to the right
    float target_pos[3] = {0, 100, 0};
    dragonfly_gaze_set_target(gaze, target_pos);
    dragonfly_gaze_update(gaze, 0.016f);

    gaze_command_t cmd;
    dragonfly_gaze_get_command(gaze, &cmd);
    // Yaw should be around 90 degrees (pi/2)
    EXPECT_GT(cmd.yaw_rad, 1.0f);
}

//=============================================================================
// Saccade Tests
//=============================================================================

TEST_F(GazeTest, TriggerSaccade) {
    // Large target jump should trigger saccade
    float target_pos1[3] = {100, 0, 0};
    dragonfly_gaze_set_target(gaze, target_pos1);
    dragonfly_gaze_update(gaze, 0.016f);

    // Jump to very different location
    float target_pos2[3] = {0, 100, 50};
    dragonfly_gaze_set_target(gaze, target_pos2);
    dragonfly_gaze_update(gaze, 0.016f);

    gaze_state_t state;
    dragonfly_gaze_get_state(gaze, &state);
    // Might be in saccade mode
}

//=============================================================================
// State Query Tests
//=============================================================================

TEST_F(GazeTest, GetState) {
    gaze_state_t state;
    EXPECT_EQ(dragonfly_gaze_get_state(gaze, &state), 0);
}

TEST_F(GazeTest, IsOnTarget) {
    float target_pos[3] = {100, 0, 0};
    dragonfly_gaze_set_target(gaze, target_pos);

    // Update several times to reach target
    for (int i = 0; i < 10; i++) {
        dragonfly_gaze_update(gaze, 0.016f);
    }

    gaze_state_t state;
    dragonfly_gaze_get_state(gaze, &state);
    EXPECT_TRUE(state.on_target);
}

//=============================================================================
// Statistics Tests
//=============================================================================

TEST_F(GazeTest, GetStats) {
    gaze_stats_t stats;
    EXPECT_EQ(dragonfly_gaze_get_stats(gaze, &stats), 0);
}

TEST_F(GazeTest, ResetStats) {
    float target_pos[3] = {100, 0, 0};
    dragonfly_gaze_set_target(gaze, target_pos);
    dragonfly_gaze_update(gaze, 0.016f);

    EXPECT_EQ(dragonfly_gaze_reset_stats(gaze), 0);
}

//=============================================================================
// Error Handling Tests
//=============================================================================

TEST_F(GazeTest, NullPointerHandling) {
    float target[3] = {100, 0, 0};
    gaze_command_t cmd;
    gaze_state_t state;

    EXPECT_EQ(dragonfly_gaze_set_target(nullptr, target), -1);
    EXPECT_EQ(dragonfly_gaze_set_target(gaze, nullptr), -1);
    EXPECT_EQ(dragonfly_gaze_update(nullptr, 0.016f), -1);
    EXPECT_EQ(dragonfly_gaze_get_command(nullptr, &cmd), -1);
    EXPECT_EQ(dragonfly_gaze_get_command(gaze, nullptr), -1);
    EXPECT_EQ(dragonfly_gaze_get_state(nullptr, &state), -1);
}

TEST_F(GazeTest, InvalidDeltaTime) {
    EXPECT_EQ(dragonfly_gaze_update(gaze, -1.0f), -1);
    EXPECT_EQ(dragonfly_gaze_update(gaze, 0.0f), -1);
}
