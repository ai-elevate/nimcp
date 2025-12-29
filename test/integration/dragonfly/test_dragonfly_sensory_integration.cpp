//=============================================================================
// test_dragonfly_sensory_integration.cpp - Sensory Integration Tests
//=============================================================================
/**
 * @file test_dragonfly_sensory_integration.cpp
 * @brief Integration tests for dragonfly visual and audio sensory fusion
 *
 * WHAT: Tests multi-sensory integration for target detection
 * WHY:  Dragonflies use visual AND audio cues for hunting
 * HOW:  Combine visual/audio bridges, verify fusion improves tracking
 *
 * BIOLOGICAL BASIS:
 * - Dragonflies use compound eyes for wide-field motion detection
 * - Directional hearing aids in prey localization
 * - Sensor fusion reduces false positives and improves accuracy
 */

#include <gtest/gtest.h>
#include <cmath>

extern "C" {
#include "dragonfly/nimcp_dragonfly.h"
#include "dragonfly/nimcp_dragonfly_visual_bridge.h"
#include "dragonfly/nimcp_dragonfly_audio_bridge.h"
#include "dragonfly/nimcp_dragonfly_tracking.h"
}

//=============================================================================
// Test Fixture
//=============================================================================

class DragonflySensoryIntegrationTest : public ::testing::Test {
protected:
    dragonfly_system_t* system = nullptr;
    dragonfly_visual_bridge_t* visual_bridge = nullptr;
    dragonfly_audio_bridge_t* audio_bridge = nullptr;
    dragonfly_tracker_t* tracker = nullptr;

    void SetUp() override {
        // Create system
        dragonfly_config_t config = dragonfly_default_config();
        system = dragonfly_system_create(&config);
        ASSERT_NE(system, nullptr);

        // Create sensory bridges
        visual_bridge = dragonfly_visual_bridge_create(
            dragonfly_visual_bridge_default_config());
        audio_bridge = dragonfly_audio_bridge_create(
            dragonfly_audio_bridge_default_config());
        tracker = dragonfly_tracker_create(tracking_default_config());

        ASSERT_NE(visual_bridge, nullptr);
        ASSERT_NE(audio_bridge, nullptr);
        ASSERT_NE(tracker, nullptr);
    }

    void TearDown() override {
        if (tracker) dragonfly_tracker_destroy(tracker);
        if (audio_bridge) dragonfly_audio_bridge_destroy(audio_bridge);
        if (visual_bridge) dragonfly_visual_bridge_destroy(visual_bridge);
        if (system) dragonfly_system_destroy(system);
    }

    // Convert position to azimuth angle
    float position_to_azimuth(float x, float y) {
        return atan2f(y, x);
    }
};

//=============================================================================
// Visual Detection Tests
//=============================================================================

TEST_F(DragonflySensoryIntegrationTest, VisualDetectionToTracker) {
    // Visual detection
    dragonfly_visual_bridge_input_t visual_input = {
        .position = {15.0f, 8.0f, 0.0f},
        .size = 0.05f,
        .contrast = 0.85f,
        .motion_direction = 0.4f,
        .motion_speed = 3.0f
    };
    EXPECT_EQ(dragonfly_visual_bridge_process(visual_bridge, &visual_input), 0);

    dragonfly_visual_bridge_output_t visual_out;
    EXPECT_EQ(dragonfly_visual_bridge_get_output(visual_bridge, &visual_out), 0);

    // Feed to tracker
    dragonfly_detection_t detection = {
        .position = {visual_input.position[0],
                    visual_input.position[1],
                    visual_input.position[2]},
        .size = visual_input.size,
        .contrast = visual_input.contrast,
        .motion_direction_rad = visual_input.motion_direction,
        .motion_speed = visual_input.motion_speed,
        .timestamp_us = 0,
        .id = 1
    };
    EXPECT_EQ(dragonfly_tracker_process_detection(tracker, &detection), 0);

    // Tracker should have the target
    uint32_t count;
    EXPECT_EQ(dragonfly_tracker_get_target_count(tracker, &count), 0);
    EXPECT_GE(count, 1);
}

TEST_F(DragonflySensoryIntegrationTest, AudioCueToDirection) {
    // Audio detection from specific direction
    float target_azimuth = 0.5f;  // radians
    dragonfly_audio_bridge_input_t audio_input = {
        .azimuth = target_azimuth,
        .elevation = 0.1f,
        .confidence = 0.8f,
        .frequency_hz = 400.0f
    };
    EXPECT_EQ(dragonfly_audio_bridge_process(audio_bridge, &audio_input), 0);

    dragonfly_audio_bridge_output_t audio_out;
    EXPECT_EQ(dragonfly_audio_bridge_get_output(audio_bridge, &audio_out), 0);

    // Direction should match input
    EXPECT_NEAR(audio_out.direction_rad, target_azimuth, 0.1f);
    EXPECT_GE(audio_out.confidence, 0.5f);
}

//=============================================================================
// Sensor Fusion Tests
//=============================================================================

TEST_F(DragonflySensoryIntegrationTest, VisualAudioAgreement) {
    // Target at position (10, 5) -> azimuth ~0.46 rad
    float target_x = 10.0f;
    float target_y = 5.0f;
    float expected_azimuth = position_to_azimuth(target_x, target_y);

    // Visual detection
    dragonfly_visual_bridge_input_t visual_input = {
        .position = {target_x, target_y, 0.0f},
        .size = 0.05f,
        .contrast = 0.8f,
        .motion_direction = expected_azimuth,
        .motion_speed = 2.0f
    };
    EXPECT_EQ(dragonfly_visual_bridge_process(visual_bridge, &visual_input), 0);

    // Audio detection from same direction
    dragonfly_audio_bridge_input_t audio_input = {
        .azimuth = expected_azimuth,
        .elevation = 0.0f,
        .confidence = 0.75f,
        .frequency_hz = 350.0f
    };
    EXPECT_EQ(dragonfly_audio_bridge_process(audio_bridge, &audio_input), 0);

    // Get outputs
    dragonfly_visual_bridge_output_t visual_out;
    dragonfly_audio_bridge_output_t audio_out;
    EXPECT_EQ(dragonfly_visual_bridge_get_output(visual_bridge, &visual_out), 0);
    EXPECT_EQ(dragonfly_audio_bridge_get_output(audio_bridge, &audio_out), 0);

    // Both should indicate similar direction
    float direction_diff = fabs(visual_out.direction_rad - audio_out.direction_rad);
    EXPECT_LT(direction_diff, 0.3f) << "Visual and audio should agree on direction";
}

TEST_F(DragonflySensoryIntegrationTest, VisualAudioConflict) {
    // Visual says target is at azimuth 0.5
    dragonfly_visual_bridge_input_t visual_input = {
        .position = {10.0f, 5.0f, 0.0f},
        .size = 0.05f,
        .contrast = 0.8f,
        .motion_direction = 0.5f,
        .motion_speed = 2.0f
    };
    EXPECT_EQ(dragonfly_visual_bridge_process(visual_bridge, &visual_input), 0);

    // Audio says target is at azimuth 2.0 (opposite direction)
    dragonfly_audio_bridge_input_t audio_input = {
        .azimuth = 2.0f,
        .elevation = 0.0f,
        .confidence = 0.6f,
        .frequency_hz = 350.0f
    };
    EXPECT_EQ(dragonfly_audio_bridge_process(audio_bridge, &audio_input), 0);

    // Get outputs
    dragonfly_visual_bridge_output_t visual_out;
    dragonfly_audio_bridge_output_t audio_out;
    EXPECT_EQ(dragonfly_visual_bridge_get_output(visual_bridge, &visual_out), 0);
    EXPECT_EQ(dragonfly_audio_bridge_get_output(audio_bridge, &audio_out), 0);

    // Directions should differ significantly
    float direction_diff = fabs(visual_out.direction_rad - audio_out.direction_rad);
    EXPECT_GT(direction_diff, 1.0f) << "Conflicting sensors should show different directions";
}

//=============================================================================
// Temporal Correlation Tests
//=============================================================================

TEST_F(DragonflySensoryIntegrationTest, SequentialVisualDetections) {
    // Target moves across field of view
    float start_x = 5.0f;
    float start_y = 10.0f;
    float velocity_x = 2.0f;
    float velocity_y = -0.5f;

    for (int frame = 0; frame < 30; frame++) {
        float t = frame * 0.016f;  // 60 FPS
        float x = start_x + velocity_x * t;
        float y = start_y + velocity_y * t;

        dragonfly_visual_bridge_input_t visual_input = {
            .position = {x, y, 0.0f},
            .size = 0.05f,
            .contrast = 0.8f,
            .motion_direction = atan2f(velocity_y, velocity_x),
            .motion_speed = sqrtf(velocity_x*velocity_x + velocity_y*velocity_y)
        };
        EXPECT_EQ(dragonfly_visual_bridge_process(visual_bridge, &visual_input), 0);

        dragonfly_visual_bridge_output_t visual_out;
        EXPECT_EQ(dragonfly_visual_bridge_get_output(visual_bridge, &visual_out), 0);

        // Motion direction should be consistent
        float expected_direction = atan2f(velocity_y, velocity_x);
        EXPECT_NEAR(visual_out.direction_rad, expected_direction, 0.5f);
    }
}

TEST_F(DragonflySensoryIntegrationTest, CorrelatedVisualAudioSequence) {
    // Target makes sound while moving
    for (int frame = 0; frame < 20; frame++) {
        float t = frame * 0.033f;  // 30 FPS
        float x = 8.0f + t * 3.0f;
        float y = 6.0f + sin(t) * 2.0f;
        float azimuth = position_to_azimuth(x, y);

        // Visual detection
        dragonfly_visual_bridge_input_t visual_input = {
            .position = {x, y, 0.0f},
            .size = 0.05f,
            .contrast = 0.75f,
            .motion_direction = azimuth,
            .motion_speed = 3.0f
        };
        EXPECT_EQ(dragonfly_visual_bridge_process(visual_bridge, &visual_input), 0);

        // Audio detection (with slight noise)
        dragonfly_audio_bridge_input_t audio_input = {
            .azimuth = azimuth + 0.05f * sin(frame),  // Small variation
            .elevation = 0.0f,
            .confidence = 0.7f,
            .frequency_hz = 400.0f + 50.0f * sin(frame * 0.5f)
        };
        EXPECT_EQ(dragonfly_audio_bridge_process(audio_bridge, &audio_input), 0);
    }

    SUCCEED();
}

//=============================================================================
// Size Selectivity Tests
//=============================================================================

TEST_F(DragonflySensoryIntegrationTest, FilterSmallTargets) {
    // Very small target - below size threshold
    dragonfly_visual_bridge_input_t visual_input = {
        .position = {10.0f, 5.0f, 0.0f},
        .size = 0.001f,  // Very small
        .contrast = 0.8f,
        .motion_direction = 0.5f,
        .motion_speed = 2.0f
    };
    EXPECT_EQ(dragonfly_visual_bridge_process(visual_bridge, &visual_input), 0);

    dragonfly_visual_bridge_output_t visual_out;
    EXPECT_EQ(dragonfly_visual_bridge_get_output(visual_bridge, &visual_out), 0);

    // Salience should be low for tiny target
    EXPECT_LT(visual_out.salience, 0.3f);
}

TEST_F(DragonflySensoryIntegrationTest, FilterLargeTargets) {
    // Very large target - above size threshold (not prey-like)
    dragonfly_visual_bridge_input_t visual_input = {
        .position = {10.0f, 5.0f, 0.0f},
        .size = 0.5f,  // Very large
        .contrast = 0.8f,
        .motion_direction = 0.5f,
        .motion_speed = 2.0f
    };
    EXPECT_EQ(dragonfly_visual_bridge_process(visual_bridge, &visual_input), 0);

    dragonfly_visual_bridge_output_t visual_out;
    EXPECT_EQ(dragonfly_visual_bridge_get_output(visual_bridge, &visual_out), 0);

    // Salience should be reduced for large target
    EXPECT_LT(visual_out.salience, 0.5f);
}

TEST_F(DragonflySensoryIntegrationTest, OptimalTargetSize) {
    // Optimal target size for dragonfly prey
    dragonfly_visual_bridge_input_t visual_input = {
        .position = {10.0f, 5.0f, 0.0f},
        .size = 0.05f,  // Optimal prey size
        .contrast = 0.85f,
        .motion_direction = 0.5f,
        .motion_speed = 2.5f
    };
    EXPECT_EQ(dragonfly_visual_bridge_process(visual_bridge, &visual_input), 0);

    dragonfly_visual_bridge_output_t visual_out;
    EXPECT_EQ(dragonfly_visual_bridge_get_output(visual_bridge, &visual_out), 0);

    // Salience should be high for optimal size
    EXPECT_GT(visual_out.salience, 0.6f);
}

//=============================================================================
// Contrast Sensitivity Tests
//=============================================================================

TEST_F(DragonflySensoryIntegrationTest, HighContrastTarget) {
    dragonfly_visual_bridge_input_t visual_input = {
        .position = {10.0f, 5.0f, 0.0f},
        .size = 0.05f,
        .contrast = 0.95f,  // High contrast
        .motion_direction = 0.5f,
        .motion_speed = 2.0f
    };
    EXPECT_EQ(dragonfly_visual_bridge_process(visual_bridge, &visual_input), 0);

    dragonfly_visual_bridge_output_t visual_out;
    EXPECT_EQ(dragonfly_visual_bridge_get_output(visual_bridge, &visual_out), 0);

    EXPECT_GT(visual_out.salience, 0.7f);
}

TEST_F(DragonflySensoryIntegrationTest, LowContrastTarget) {
    dragonfly_visual_bridge_input_t visual_input = {
        .position = {10.0f, 5.0f, 0.0f},
        .size = 0.05f,
        .contrast = 0.2f,  // Low contrast
        .motion_direction = 0.5f,
        .motion_speed = 2.0f
    };
    EXPECT_EQ(dragonfly_visual_bridge_process(visual_bridge, &visual_input), 0);

    dragonfly_visual_bridge_output_t visual_out;
    EXPECT_EQ(dragonfly_visual_bridge_get_output(visual_bridge, &visual_out), 0);

    EXPECT_LT(visual_out.salience, 0.5f);
}

//=============================================================================
// Multi-Target Discrimination Tests
//=============================================================================

TEST_F(DragonflySensoryIntegrationTest, TwoTargetDiscrimination) {
    // First target
    dragonfly_visual_bridge_input_t visual1 = {
        .position = {10.0f, 5.0f, 0.0f},
        .size = 0.05f,
        .contrast = 0.8f,
        .motion_direction = 0.3f,
        .motion_speed = 2.0f
    };
    EXPECT_EQ(dragonfly_visual_bridge_process(visual_bridge, &visual1), 0);

    // Second target (different location)
    dragonfly_visual_bridge_input_t visual2 = {
        .position = {-8.0f, 12.0f, 0.0f},
        .size = 0.04f,
        .contrast = 0.75f,
        .motion_direction = -0.5f,
        .motion_speed = 1.8f
    };
    EXPECT_EQ(dragonfly_visual_bridge_process(visual_bridge, &visual2), 0);

    // Bridge should process both (latest detection in output)
    dragonfly_visual_bridge_output_t visual_out;
    EXPECT_EQ(dragonfly_visual_bridge_get_output(visual_bridge, &visual_out), 0);

    // Stats should show 2 detections
    dragonfly_visual_bridge_stats_t stats;
    EXPECT_EQ(dragonfly_visual_bridge_get_stats(visual_bridge, &stats), 0);
    EXPECT_GE(stats.inputs_processed, 2);
}

TEST_F(DragonflySensoryIntegrationTest, DistractorSuppression) {
    // Primary target - high contrast, optimal size
    dragonfly_visual_bridge_input_t primary = {
        .position = {10.0f, 5.0f, 0.0f},
        .size = 0.05f,
        .contrast = 0.9f,
        .motion_direction = 0.5f,
        .motion_speed = 3.0f
    };
    EXPECT_EQ(dragonfly_visual_bridge_process(visual_bridge, &primary), 0);

    dragonfly_visual_bridge_output_t primary_out;
    EXPECT_EQ(dragonfly_visual_bridge_get_output(visual_bridge, &primary_out), 0);
    float primary_salience = primary_out.salience;

    // Distractor - low contrast, suboptimal size
    dragonfly_visual_bridge_input_t distractor = {
        .position = {-5.0f, 8.0f, 0.0f},
        .size = 0.02f,
        .contrast = 0.4f,
        .motion_direction = -0.3f,
        .motion_speed = 1.0f
    };
    EXPECT_EQ(dragonfly_visual_bridge_process(visual_bridge, &distractor), 0);

    dragonfly_visual_bridge_output_t distractor_out;
    EXPECT_EQ(dragonfly_visual_bridge_get_output(visual_bridge, &distractor_out), 0);
    float distractor_salience = distractor_out.salience;

    // Primary should have higher salience
    EXPECT_GT(primary_salience, distractor_salience);
}
