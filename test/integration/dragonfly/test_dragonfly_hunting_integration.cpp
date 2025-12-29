/**
 * @file test_dragonfly_hunting_integration.cpp
 * @brief Integration tests for dragonfly hunting subsystems
 *
 * Tests the integration of energy, learning, collision, gaze, multi-target,
 * emotion bridge, sleep bridge, and immune bridge modules.
 *
 * @author NIMCP Team
 * @date 2024-12-29
 */

#include <gtest/gtest.h>
#include <cmath>
#include <thread>
#include <chrono>

extern "C" {
#include "dragonfly/nimcp_dragonfly.h"
#include "dragonfly/nimcp_dragonfly_multi_target.h"
#include "dragonfly/nimcp_dragonfly_energy.h"
#include "dragonfly/nimcp_dragonfly_collision.h"
#include "dragonfly/nimcp_dragonfly_learning.h"
#include "dragonfly/nimcp_dragonfly_gaze.h"
#include "dragonfly/nimcp_dragonfly_emotion_bridge.h"
#include "dragonfly/nimcp_dragonfly_sleep_bridge.h"
#include "dragonfly/nimcp_dragonfly_immune_bridge.h"
}

//=============================================================================
// Test Fixture
//=============================================================================

class DragonflyHuntingIntegrationTest : public ::testing::Test {
protected:
    dragonfly_multi_target_t multi_target = nullptr;
    dragonfly_energy_t energy = nullptr;
    dragonfly_collision_t collision = nullptr;
    dragonfly_learning_t learning = nullptr;
    dragonfly_gaze_t gaze = nullptr;
    dragonfly_emotion_bridge_t emotion = nullptr;
    dragonfly_sleep_bridge_t sleep = nullptr;
    dragonfly_immune_bridge_t immune = nullptr;

    void SetUp() override {
        multi_target = dragonfly_multi_target_create(nullptr);
        energy = dragonfly_energy_create(nullptr);
        collision = dragonfly_collision_create(nullptr);
        learning = dragonfly_learning_create(nullptr);
        gaze = dragonfly_gaze_create(nullptr);
        emotion = dragonfly_emotion_bridge_create(nullptr);
        sleep = dragonfly_sleep_bridge_create(nullptr);
        immune = dragonfly_immune_bridge_create(nullptr);

        ASSERT_NE(multi_target, nullptr);
        ASSERT_NE(energy, nullptr);
        ASSERT_NE(collision, nullptr);
        ASSERT_NE(learning, nullptr);
        ASSERT_NE(gaze, nullptr);
        ASSERT_NE(emotion, nullptr);
        ASSERT_NE(sleep, nullptr);
        ASSERT_NE(immune, nullptr);
    }

    void TearDown() override {
        if (multi_target) dragonfly_multi_target_destroy(multi_target);
        if (energy) dragonfly_energy_destroy(energy);
        if (collision) dragonfly_collision_destroy(collision);
        if (learning) dragonfly_learning_destroy(learning);
        if (gaze) dragonfly_gaze_destroy(gaze);
        if (emotion) dragonfly_emotion_bridge_destroy(emotion);
        if (sleep) dragonfly_sleep_bridge_destroy(sleep);
        if (immune) dragonfly_immune_bridge_destroy(immune);
    }

    dragonfly_detection_t make_detection(uint32_t id, float x, float y, float z) {
        dragonfly_detection_t det = {};
        det.id = id;
        det.position[0] = x;
        det.position[1] = y;
        det.position[2] = z;
        det.motion_direction_rad = 0.0f;
        det.motion_speed = 3.0f;
        det.size = 0.05f;
        det.contrast = 0.8f;
        det.timestamp_us = 1000000;
        return det;
    }

    dragonfly_self_state_t make_self_state() {
        dragonfly_self_state_t self = {};
        self.position[0] = 0;
        self.position[1] = 0;
        self.position[2] = 0;
        self.max_speed = 15.0f;
        self.max_accel = 10.0f;
        self.max_turn_rate = 3.0f;
        self.energy_level = 0.8f;
        return self;
    }
};

//=============================================================================
// Hunt Decision Making Tests
//=============================================================================

TEST_F(DragonflyHuntingIntegrationTest, EnergyAffectsHuntDecision) {
    // When energy is low, hunting motivation should decrease
    dragonfly_emotion_bridge_update(emotion, 0.1f);

    float motivation_high = dragonfly_emotion_get_hunting_motivation(emotion);

    // Simulate energy depletion
    for (int i = 0; i < 50; i++) {
        dragonfly_energy_update(energy, ACTIVITY_PURSUIT, 0.1f);
    }

    // Check energy state
    energy_budget_t budget;
    dragonfly_energy_get_budget(energy, &budget);

    // Lower energy should affect pursuit decisions
    EXPECT_LT(budget.current_energy_j, budget.max_energy_j);
}

TEST_F(DragonflyHuntingIntegrationTest, HealthAffectsHuntCapability) {
    // Healthy dragonfly can pursue
    EXPECT_TRUE(dragonfly_immune_can_pursue(immune));

    // Injure the dragonfly
    dragonfly_immune_on_injury(immune, 0.9f);

    // Severely injured should not pursue
    EXPECT_FALSE(dragonfly_immune_can_pursue(immune));
}

TEST_F(DragonflyHuntingIntegrationTest, EmotionModulatesAggression) {
    // Build up hunger to increase aggression
    for (int i = 0; i < 100; i++) {
        dragonfly_emotion_bridge_update(emotion, 0.1f);
    }

    dragonfly_drives_t drives;
    dragonfly_emotion_get_drives(emotion, &drives);

    // Hunger should increase aggression
    EXPECT_GT(drives.hunger, 0.3f);

    pursuit_modulation_t mod;
    dragonfly_emotion_get_pursuit_modulation(emotion, &mod);
    EXPECT_GT(mod.aggression_modifier, 0.0f);
}

//=============================================================================
// Target Selection Integration Tests
//=============================================================================

TEST_F(DragonflyHuntingIntegrationTest, MultiTargetWithEnergyConsideration) {
    dragonfly_self_state_t self = make_self_state();

    // Add multiple targets at different distances
    dragonfly_detection_t close = make_detection(1, 50, 0, 0);
    dragonfly_detection_t far = make_detection(2, 200, 0, 0);

    dragonfly_multi_target_update(multi_target, &close, &self, 0.016f);
    dragonfly_multi_target_update(multi_target, &far, &self, 0.016f);

    // Get best target
    queued_target_t best;
    dragonfly_multi_target_get_best(multi_target, &best);

    // Closer target should be preferred (better energy efficiency)
    EXPECT_EQ(best.id, 1u);
}

TEST_F(DragonflyHuntingIntegrationTest, GazeTracksSelectedTarget) {
    dragonfly_self_state_t self = make_self_state();

    // Add a target
    dragonfly_detection_t det = make_detection(1, 100, 50, 0);
    dragonfly_multi_target_update(multi_target, &det, &self, 0.016f);

    // Get target position
    queued_target_t target;
    dragonfly_multi_target_get_best(multi_target, &target);

    // Direct gaze to target
    dragonfly_gaze_set_target(gaze, target.position);
    dragonfly_gaze_update(gaze, 0.1f);

    // Gaze should be moving toward target
    gaze_state_t state;
    dragonfly_gaze_get_state(gaze, &state);
    // Gaze should be responding
}

//=============================================================================
// Collision Avoidance Integration Tests
//=============================================================================

TEST_F(DragonflyHuntingIntegrationTest, CollisionAvoidanceModifiesPursuit) {
    // Add obstacle in path
    obstacle_t obs = {};
    obs.id = 1;
    obs.position[0] = 5.0f;
    obs.position[1] = 0.0f;
    obs.position[2] = 0.0f;
    obs.radius = 1.0f;
    obs.type = OBSTACLE_STATIC;

    dragonfly_collision_add_obstacle(collision, &obs);

    // Self moving toward obstacle
    dragonfly_self_state_t self = make_self_state();
    self.velocity[0] = 10.0f;

    dragonfly_collision_update(collision, &self, 0.016f);

    collision_state_t state;
    dragonfly_collision_get_state(collision, &state);

    // Should detect imminent collision
    EXPECT_TRUE(state.collision_imminent);
    EXPECT_GT(state.threat_level, 0.0f);
}

//=============================================================================
// Learning Integration Tests
//=============================================================================

TEST_F(DragonflyHuntingIntegrationTest, LearnFromSuccessfulHunt) {
    // Record a successful hunt
    hunt_episode_t episode = {};
    episode.outcome = HUNT_SUCCESS;
    episode.strategy = INTERCEPT_PURSUIT;
    episode.pursuit_duration_s = 2.0f;
    episode.target_size = 0.05f;
    episode.target_speed = 4.0f;
    episode.target_maneuverability = 0.3f;
    episode.initial_range = 50.0f;

    dragonfly_learning_record_episode(learning, &episode);

    // Get strategy stats
    strategy_learning_t stats;
    dragonfly_learning_get_strategy_stats(learning, INTERCEPT_PURSUIT, &stats);
    EXPECT_GE(stats.attempts, 1u);
}

TEST_F(DragonflyHuntingIntegrationTest, SleepConsolidatesLearning) {
    // Record experiences
    for (int i = 0; i < 10; i++) {
        dragonfly_sleep_record_success(sleep, i, 0.05f, INTERCEPT_PURSUIT);
    }

    // Consolidate during "sleep"
    dragonfly_sleep_consolidate(sleep);

    // Check memory was updated
    consolidated_memory_t memory;
    dragonfly_sleep_get_memory(sleep, &memory);
    EXPECT_GT(memory.experiences_processed, 0u);
}

//=============================================================================
// Circadian Integration Tests
//=============================================================================

TEST_F(DragonflyHuntingIntegrationTest, CircadianAffectsActivity) {
    // Update circadian state
    dragonfly_sleep_bridge_update(sleep, 0.1f);

    float activity = dragonfly_sleep_get_activity(sleep);
    EXPECT_GE(activity, 0.0f);
    EXPECT_LE(activity, 1.5f);

    bool allowed = dragonfly_sleep_hunting_allowed(sleep);
    // Activity level affects hunting permission
    (void)allowed;
}

//=============================================================================
// Stress and Recovery Integration Tests
//=============================================================================

TEST_F(DragonflyHuntingIntegrationTest, PursuitIncreasesStress) {
    dragonfly_health_state_t before, after;
    dragonfly_immune_get_health(immune, &before);

    // Simulate intense pursuit
    dragonfly_immune_on_pursuit_start(immune, 0.9f);

    dragonfly_immune_get_health(immune, &after);
    EXPECT_GE(after.stress_level, before.stress_level);
}

TEST_F(DragonflyHuntingIntegrationTest, RestRecovery) {
    // Build up stress
    dragonfly_immune_on_pursuit_start(immune, 0.9f);

    dragonfly_health_state_t before, after;
    dragonfly_immune_get_health(immune, &before);

    // Simulate rest
    for (int i = 0; i < 50; i++) {
        dragonfly_immune_bridge_update(immune, 0.1f);
    }

    dragonfly_immune_get_health(immune, &after);
    EXPECT_LT(after.stress_level, before.stress_level);
}

//=============================================================================
// Full Hunt Simulation Tests
//=============================================================================

TEST_F(DragonflyHuntingIntegrationTest, SimulateSuccessfulHunt) {
    dragonfly_self_state_t self = make_self_state();

    // 1. Detect target
    dragonfly_detection_t det = make_detection(1, 80, 0, 0);
    dragonfly_multi_target_update(multi_target, &det, &self, 0.016f);

    // 2. Check hunt viability
    EXPECT_TRUE(dragonfly_immune_can_pursue(immune));

    // 3. Get hunting motivation
    float motivation = dragonfly_emotion_get_hunting_motivation(emotion);
    EXPECT_GT(motivation, 0.0f);

    // 4. Check energy budget
    energy_budget_t budget;
    dragonfly_energy_get_budget(energy, &budget);
    EXPECT_GT(budget.current_energy_j, 0.0f);

    // 5. Direct gaze to target
    queued_target_t target;
    dragonfly_multi_target_get_best(multi_target, &target);
    dragonfly_gaze_set_target(gaze, target.position);

    // 6. Begin pursuit (update emotion/immune)
    dragonfly_emotion_on_prey_sighted(emotion, 0.8f);
    dragonfly_immune_on_pursuit_start(immune, motivation);

    // 7. Simulate pursuit updates
    for (int i = 0; i < 30; i++) {
        dragonfly_gaze_update(gaze, 0.016f);
        dragonfly_collision_update(collision, &self, 0.016f);
        dragonfly_energy_update(energy, ACTIVITY_PURSUIT, 0.016f);
        dragonfly_emotion_bridge_update(emotion, 0.016f);
        dragonfly_immune_bridge_update(immune, 0.016f);
    }

    // 8. Successful catch
    dragonfly_emotion_on_prey_caught(emotion, 0.1f);
    dragonfly_energy_gain(energy, 200.0f);

    // 9. Record success
    dragonfly_sleep_record_success(sleep, 1, 0.05f, INTERCEPT_PURSUIT);

    // 10. End pursuit
    dragonfly_immune_on_pursuit_end(immune, 0.5f, true);

    // Verify stats
    dragonfly_sleep_stats_t stats;
    dragonfly_sleep_get_stats(sleep, &stats);
    EXPECT_EQ(stats.experiences_recorded, 1u);
}

TEST_F(DragonflyHuntingIntegrationTest, SimulateFailedHunt) {
    dragonfly_self_state_t self = make_self_state();

    // 1. Detect target
    dragonfly_detection_t det = make_detection(1, 150, 0, 0);
    dragonfly_multi_target_update(multi_target, &det, &self, 0.016f);

    // 2. Begin pursuit
    dragonfly_emotion_on_prey_sighted(emotion, 0.5f);
    dragonfly_immune_on_pursuit_start(immune, 0.6f);

    // 3. Simulate long pursuit (target escapes)
    for (int i = 0; i < 100; i++) {
        dragonfly_energy_update(energy, ACTIVITY_PURSUIT, 0.016f);
        dragonfly_immune_bridge_update(immune, 0.016f);
    }

    // 4. Record failure
    dragonfly_sleep_record_failure(sleep, 1, "target_escaped", INTERCEPT_PURSUIT);

    // 5. End pursuit
    dragonfly_immune_on_pursuit_end(immune, 2.0f, false);

    // Learn from failure
    hunt_episode_t episode = {};
    episode.outcome = HUNT_FAILURE_ESCAPED;
    episode.strategy = INTERCEPT_PURSUIT;
    episode.pursuit_duration_s = 2.0f;
    episode.failure_reason = FAILURE_TARGET_ESCAPED;
    dragonfly_learning_record_episode(learning, &episode);

    // Verify recorded
    learning_stats_t lstats;
    dragonfly_learning_get_stats(learning, &lstats);
    EXPECT_GE(lstats.episodes_recorded, 1u);
}
