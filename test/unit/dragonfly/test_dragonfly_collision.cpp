/**
 * @file test_dragonfly_collision.cpp
 * @brief Unit tests for collision avoidance module
 *
 * @author NIMCP Team
 * @date 2024-12-29
 */

#include <gtest/gtest.h>
#include <cmath>

extern "C" {
#include "dragonfly/nimcp_dragonfly_collision.h"
}

//=============================================================================
// Test Fixture
//=============================================================================

class CollisionTest : public ::testing::Test {
protected:
    dragonfly_collision_t collision = nullptr;

    void SetUp() override {
        collision = dragonfly_collision_create(nullptr);
        ASSERT_NE(collision, nullptr);
    }

    void TearDown() override {
        if (collision) {
            dragonfly_collision_destroy(collision);
            collision = nullptr;
        }
    }

    obstacle_t make_obstacle(uint32_t id, float x, float y, float z, float radius) {
        obstacle_t obs = {};
        obs.id = id;
        obs.position[0] = x;
        obs.position[1] = y;
        obs.position[2] = z;
        obs.radius = radius;
        obs.type = OBSTACLE_STATIC;
        return obs;
    }

    dragonfly_self_state_t make_self_state(float x, float y, float z,
                                            float vx, float vy, float vz) {
        dragonfly_self_state_t self = {};
        self.position[0] = x;
        self.position[1] = y;
        self.position[2] = z;
        self.velocity[0] = vx;
        self.velocity[1] = vy;
        self.velocity[2] = vz;
        self.max_speed = 15.0f;
        self.max_accel = 10.0f;
        self.max_turn_rate = 3.0f;
        return self;
    }
};

//=============================================================================
// Configuration Tests
//=============================================================================

TEST_F(CollisionTest, DefaultConfig) {
    collision_config_t config = dragonfly_collision_default_config();
    EXPECT_GT(config.detection_range, 0.0f);
    EXPECT_GT(config.safety_margin, 0.0f);
    EXPECT_GT(config.critical_ttc_s, 0.0f);
}

TEST_F(CollisionTest, ValidateConfig) {
    collision_config_t config = dragonfly_collision_default_config();
    EXPECT_TRUE(dragonfly_collision_validate_config(&config));

    config.detection_range = 0.0f;
    EXPECT_FALSE(dragonfly_collision_validate_config(&config));

    config = dragonfly_collision_default_config();
    config.critical_ttc_s = -1.0f;
    EXPECT_FALSE(dragonfly_collision_validate_config(&config));

    EXPECT_FALSE(dragonfly_collision_validate_config(nullptr));
}

TEST_F(CollisionTest, CreateWithCustomConfig) {
    collision_config_t config = dragonfly_collision_default_config();
    config.detection_range = 20.0f;
    config.safety_margin = 0.5f;

    dragonfly_collision_t custom = dragonfly_collision_create(&config);
    ASSERT_NE(custom, nullptr);
    dragonfly_collision_destroy(custom);
}

//=============================================================================
// Lifecycle Tests
//=============================================================================

TEST_F(CollisionTest, CreateAndDestroy) {
    dragonfly_collision_t c = dragonfly_collision_create(nullptr);
    ASSERT_NE(c, nullptr);
    dragonfly_collision_destroy(c);
}

TEST_F(CollisionTest, DestroyNull) {
    dragonfly_collision_destroy(nullptr);  // Should not crash
}

TEST_F(CollisionTest, Reset) {
    EXPECT_EQ(dragonfly_collision_reset(collision), 0);
}

//=============================================================================
// Obstacle Management Tests
//=============================================================================

TEST_F(CollisionTest, AddObstacle) {
    obstacle_t obs = make_obstacle(1, 10, 0, 0, 1.0f);
    EXPECT_EQ(dragonfly_collision_add_obstacle(collision, &obs), 0);
}

TEST_F(CollisionTest, RemoveObstacle) {
    obstacle_t obs = make_obstacle(1, 10, 0, 0, 1.0f);
    dragonfly_collision_add_obstacle(collision, &obs);
    EXPECT_EQ(dragonfly_collision_remove_obstacle(collision, 1), 0);
}

TEST_F(CollisionTest, ClearObstacles) {
    obstacle_t obs1 = make_obstacle(1, 10, 0, 0, 1.0f);
    obstacle_t obs2 = make_obstacle(2, 20, 0, 0, 1.5f);
    dragonfly_collision_add_obstacle(collision, &obs1);
    dragonfly_collision_add_obstacle(collision, &obs2);

    EXPECT_EQ(dragonfly_collision_clear_obstacles(collision), 0);
}

//=============================================================================
// Collision Detection Tests
//=============================================================================

TEST_F(CollisionTest, NoCollisionWhenClear) {
    dragonfly_self_state_t self = make_self_state(0, 0, 0, 10, 0, 0);
    // No obstacles added

    collision_state_t state;
    EXPECT_EQ(dragonfly_collision_update(collision, &self, 0.016f), 0);
    EXPECT_EQ(dragonfly_collision_get_state(collision, &state), 0);
    EXPECT_FALSE(state.collision_imminent);
}

TEST_F(CollisionTest, DetectDirectCollision) {
    // Add obstacle directly ahead
    obstacle_t obs = make_obstacle(1, 5, 0, 0, 1.0f);
    dragonfly_collision_add_obstacle(collision, &obs);

    // Moving towards obstacle
    dragonfly_self_state_t self = make_self_state(0, 0, 0, 15, 0, 0);

    collision_state_t state;
    dragonfly_collision_update(collision, &self, 0.016f);
    dragonfly_collision_get_state(collision, &state);

    EXPECT_TRUE(state.collision_imminent);
    EXPECT_GT(state.threat_level, 0.0f);
}

TEST_F(CollisionTest, TTCCalculation) {
    // Obstacle at 10m, moving at 10m/s = 1 second TTC
    obstacle_t obs = make_obstacle(1, 10, 0, 0, 1.0f);
    dragonfly_collision_add_obstacle(collision, &obs);

    dragonfly_self_state_t self = make_self_state(0, 0, 0, 10, 0, 0);

    dragonfly_collision_update(collision, &self, 0.016f);

    collision_state_t state;
    dragonfly_collision_get_state(collision, &state);

    EXPECT_GT(state.min_ttc_s, 0.0f);
    EXPECT_LT(state.min_ttc_s, 2.0f);  // Should be around 1 second
}

TEST_F(CollisionTest, NoCollisionWhenMovingAway) {
    // Obstacle behind
    obstacle_t obs = make_obstacle(1, -10, 0, 0, 1.0f);
    dragonfly_collision_add_obstacle(collision, &obs);

    // Moving forward (away from obstacle)
    dragonfly_self_state_t self = make_self_state(0, 0, 0, 10, 0, 0);

    dragonfly_collision_update(collision, &self, 0.016f);

    collision_state_t state;
    dragonfly_collision_get_state(collision, &state);

    EXPECT_FALSE(state.collision_imminent);
}

//=============================================================================
// Avoidance Planning Tests
//=============================================================================

TEST_F(CollisionTest, ComputeAvoidanceManeuver) {
    obstacle_t obs = make_obstacle(1, 5, 0, 0, 1.0f);
    dragonfly_collision_add_obstacle(collision, &obs);

    dragonfly_self_state_t self = make_self_state(0, 0, 0, 10, 0, 0);
    dragonfly_collision_update(collision, &self, 0.016f);

    avoidance_maneuver_t maneuver;
    EXPECT_EQ(dragonfly_collision_get_avoidance(collision, &maneuver), 0);

    if (maneuver.avoidance_needed) {
        // Should have non-zero adjustment
        float adjustment_mag = sqrtf(
            maneuver.velocity_adjustment[0] * maneuver.velocity_adjustment[0] +
            maneuver.velocity_adjustment[1] * maneuver.velocity_adjustment[1] +
            maneuver.velocity_adjustment[2] * maneuver.velocity_adjustment[2]);
        EXPECT_GT(adjustment_mag, 0.0f);
    }
}

TEST_F(CollisionTest, FindSafeDirection) {
    obstacle_t obs = make_obstacle(1, 5, 0, 0, 1.0f);
    dragonfly_collision_add_obstacle(collision, &obs);

    dragonfly_self_state_t self = make_self_state(0, 0, 0, 10, 0, 0);

    float safe_dir[3];
    EXPECT_EQ(dragonfly_collision_find_safe_direction(collision, &self, safe_dir), 0);

    // Safe direction should not point at obstacle
    // (dot product of safe direction and obstacle direction should be low)
}

//=============================================================================
// Emergency Stop Tests
//=============================================================================

TEST_F(CollisionTest, EmergencyStopRequired) {
    // Very close obstacle
    obstacle_t obs = make_obstacle(1, 1, 0, 0, 0.5f);
    dragonfly_collision_add_obstacle(collision, &obs);

    dragonfly_self_state_t self = make_self_state(0, 0, 0, 15, 0, 0);
    dragonfly_collision_update(collision, &self, 0.016f);

    collision_state_t state;
    dragonfly_collision_get_state(collision, &state);

    // Should trigger emergency response
    EXPECT_GT(state.threat_level, 0.7f);
}

//=============================================================================
// Statistics Tests
//=============================================================================

TEST_F(CollisionTest, GetStats) {
    collision_stats_t stats;
    EXPECT_EQ(dragonfly_collision_get_stats(collision, &stats), 0);
}

TEST_F(CollisionTest, ResetStats) {
    dragonfly_self_state_t self = make_self_state(0, 0, 0, 10, 0, 0);
    dragonfly_collision_update(collision, &self, 0.016f);

    EXPECT_EQ(dragonfly_collision_reset_stats(collision), 0);

    collision_stats_t stats;
    dragonfly_collision_get_stats(collision, &stats);
    EXPECT_EQ(stats.updates_processed, 0u);
}

//=============================================================================
// Error Handling Tests
//=============================================================================

TEST_F(CollisionTest, NullPointerHandling) {
    dragonfly_self_state_t self = make_self_state(0, 0, 0, 10, 0, 0);
    obstacle_t obs = make_obstacle(1, 10, 0, 0, 1.0f);

    EXPECT_EQ(dragonfly_collision_update(nullptr, &self, 0.016f), -1);
    EXPECT_EQ(dragonfly_collision_update(collision, nullptr, 0.016f), -1);
    EXPECT_EQ(dragonfly_collision_add_obstacle(nullptr, &obs), -1);
    EXPECT_EQ(dragonfly_collision_add_obstacle(collision, nullptr), -1);
    EXPECT_EQ(dragonfly_collision_get_state(nullptr, nullptr), -1);
}
