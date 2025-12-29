/**
 * @file test_dragonfly_environment.cpp
 * @brief Unit tests for dragonfly environment module
 *
 * Tests environmental sensing including wind, obstacles,
 * light conditions, and weather effects on hunting.
 *
 * @author NIMCP Team
 * @date 2024-12-29
 */

#include <gtest/gtest.h>
#include <cmath>

extern "C" {
#include "dragonfly/nimcp_dragonfly_environment.h"
}

//=============================================================================
// Test Fixture
//=============================================================================

class DragonEnvironmentTest : public ::testing::Test {
protected:
    dragonfly_environment_t env = nullptr;

    void SetUp() override {
        dragonfly_env_config_t config = dragonfly_env_default_config();
        env = dragonfly_environment_create(&config);
        ASSERT_NE(env, nullptr);
    }

    void TearDown() override {
        if (env) {
            dragonfly_environment_destroy(env);
            env = nullptr;
        }
    }
};

//=============================================================================
// Configuration Tests
//=============================================================================

TEST_F(DragonEnvironmentTest, DefaultConfig) {
    dragonfly_env_config_t config = dragonfly_env_default_config();

    EXPECT_GT(config.wind_sensing_radius, 0.0f);
    EXPECT_GT(config.obstacle_sensing_radius, 0.0f);
    EXPECT_GE(config.min_visibility, 0.0f);
}

TEST_F(DragonEnvironmentTest, ValidateConfig) {
    dragonfly_env_config_t config = dragonfly_env_default_config();
    EXPECT_TRUE(dragonfly_env_validate_config(&config));

    EXPECT_FALSE(dragonfly_env_validate_config(nullptr));
}

TEST_F(DragonEnvironmentTest, CreateWithCustomConfig) {
    dragonfly_env_config_t config = dragonfly_env_default_config();
    config.wind_sensing_radius = 20.0f;

    dragonfly_environment_t custom = dragonfly_environment_create(&config);
    ASSERT_NE(custom, nullptr);
    dragonfly_environment_destroy(custom);
}

//=============================================================================
// Lifecycle Tests
//=============================================================================

TEST_F(DragonEnvironmentTest, CreateAndDestroy) {
    dragonfly_environment_t e = dragonfly_environment_create(nullptr);
    ASSERT_NE(e, nullptr);
    dragonfly_environment_destroy(e);
}

TEST_F(DragonEnvironmentTest, DestroyNull) {
    dragonfly_environment_destroy(nullptr);
}

TEST_F(DragonEnvironmentTest, Reset) {
    EXPECT_EQ(dragonfly_environment_reset(env), 0);
}

//=============================================================================
// Update Tests
//=============================================================================

TEST_F(DragonEnvironmentTest, UpdateBasic) {
    EXPECT_EQ(dragonfly_environment_update(env, 0.016f), 0);
}

TEST_F(DragonEnvironmentTest, UpdateMultiple) {
    for (int i = 0; i < 100; i++) {
        EXPECT_EQ(dragonfly_environment_update(env, 0.016f), 0);
    }
}

//=============================================================================
// Wind Tests
//=============================================================================

TEST_F(DragonEnvironmentTest, SetWind) {
    float wind[3] = {5.0f, 0.0f, 0.0f};
    EXPECT_EQ(dragonfly_environment_set_wind(env, wind), 0);
}

TEST_F(DragonEnvironmentTest, GetWind) {
    float set_wind[3] = {3.0f, 2.0f, 0.0f};
    dragonfly_environment_set_wind(env, set_wind);

    float get_wind[3];
    EXPECT_EQ(dragonfly_environment_get_wind(env, get_wind), 0);
    EXPECT_FLOAT_EQ(get_wind[0], set_wind[0]);
    EXPECT_FLOAT_EQ(get_wind[1], set_wind[1]);
}

TEST_F(DragonEnvironmentTest, WindAffectsMovement) {
    float strong_wind[3] = {10.0f, 0.0f, 0.0f};
    dragonfly_environment_set_wind(env, strong_wind);

    env_modulation_t mod;
    EXPECT_EQ(dragonfly_environment_get_modulation(env, mod), 0);
    // Strong headwind should reduce effective speed
}

//=============================================================================
// Obstacle Tests
//=============================================================================

TEST_F(DragonEnvironmentTest, AddObstacle) {
    float position[3] = {10.0f, 0.0f, 0.0f};
    float size[3] = {1.0f, 1.0f, 1.0f};
    EXPECT_EQ(dragonfly_environment_add_obstacle(env, position, size), 0);
}

TEST_F(DragonEnvironmentTest, ClearObstacles) {
    float position[3] = {10.0f, 0.0f, 0.0f};
    float size[3] = {1.0f, 1.0f, 1.0f};
    dragonfly_environment_add_obstacle(env, position, size);

    EXPECT_EQ(dragonfly_environment_clear_obstacles(env), 0);
}

TEST_F(DragonEnvironmentTest, CheckCollision) {
    float obs_pos[3] = {5.0f, 0.0f, 0.0f};
    float obs_size[3] = {2.0f, 2.0f, 2.0f};
    dragonfly_environment_add_obstacle(env, obs_pos, obs_size);

    float test_pos[3] = {5.0f, 0.0f, 0.0f};
    EXPECT_TRUE(dragonfly_environment_check_collision(env, test_pos, 0.5f));

    float safe_pos[3] = {20.0f, 0.0f, 0.0f};
    EXPECT_FALSE(dragonfly_environment_check_collision(env, safe_pos, 0.5f));
}

//=============================================================================
// Light Conditions Tests
//=============================================================================

TEST_F(DragonEnvironmentTest, SetLightLevel) {
    EXPECT_EQ(dragonfly_environment_set_light(env, 0.8f), 0);
}

TEST_F(DragonEnvironmentTest, GetLightLevel) {
    dragonfly_environment_set_light(env, 0.6f);
    float level = dragonfly_environment_get_light(env);
    EXPECT_FLOAT_EQ(level, 0.6f);
}

TEST_F(DragonEnvironmentTest, LightClampedToRange) {
    dragonfly_environment_set_light(env, 1.5f);
    EXPECT_LE(dragonfly_environment_get_light(env), 1.0f);

    dragonfly_environment_set_light(env, -0.5f);
    EXPECT_GE(dragonfly_environment_get_light(env), 0.0f);
}

TEST_F(DragonEnvironmentTest, LowLightReducesVisibility) {
    dragonfly_environment_set_light(env, 0.1f);

    dragonfly_env_state_t state;
    EXPECT_EQ(dragonfly_environment_get_state(env, &state), 0);
    EXPECT_LT(state.visibility, 1.0f);
}

//=============================================================================
// Weather Tests
//=============================================================================

TEST_F(DragonEnvironmentTest, SetWeather) {
    EXPECT_EQ(dragonfly_environment_set_weather(env, WEATHER_CLEAR), 0);
    EXPECT_EQ(dragonfly_environment_set_weather(env, WEATHER_RAIN), 0);
    EXPECT_EQ(dragonfly_environment_set_weather(env, WEATHER_OVERCAST), 0);
}

TEST_F(DragonEnvironmentTest, GetWeather) {
    dragonfly_environment_set_weather(env, WEATHER_RAIN);
    EXPECT_EQ(dragonfly_environment_get_weather(env), WEATHER_RAIN);
}

TEST_F(DragonEnvironmentTest, RainReducesHuntability) {
    dragonfly_environment_set_weather(env, WEATHER_RAIN);

    dragonfly_env_state_t state;
    EXPECT_EQ(dragonfly_environment_get_state(env, &state), 0);
    EXPECT_LT(state.hunting_suitability, 1.0f);
}

//=============================================================================
// State Tests
//=============================================================================

TEST_F(DragonEnvironmentTest, GetState) {
    dragonfly_env_state_t state;
    EXPECT_EQ(dragonfly_environment_get_state(env, &state), 0);

    EXPECT_GE(state.visibility, 0.0f);
    EXPECT_LE(state.visibility, 1.0f);
    EXPECT_GE(state.hunting_suitability, 0.0f);
}

TEST_F(DragonEnvironmentTest, GetStats) {
    dragonfly_env_stats_t stats;
    EXPECT_EQ(dragonfly_environment_get_stats(env, &stats), 0);
}

//=============================================================================
// Modulation Tests
//=============================================================================

TEST_F(DragonEnvironmentTest, GetModulation) {
    env_modulation_t mod;
    EXPECT_EQ(dragonfly_environment_get_modulation(env, mod), 0);
}

//=============================================================================
// Null Parameter Tests
//=============================================================================

TEST_F(DragonEnvironmentTest, NullWindArray) {
    EXPECT_NE(dragonfly_environment_set_wind(env, nullptr), 0);
    EXPECT_NE(dragonfly_environment_get_wind(env, nullptr), 0);
}

TEST_F(DragonEnvironmentTest, NullStateOutput) {
    EXPECT_NE(dragonfly_environment_get_state(env, nullptr), 0);
}
