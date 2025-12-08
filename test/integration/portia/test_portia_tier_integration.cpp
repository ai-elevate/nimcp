/**
 * @file test_portia_tier_integration.cpp
 * @brief Integration tests for Portia platform tier system
 *
 * WHAT: Tests tier switching with real platform metrics
 * WHY:  Validate tier system adapts correctly to resource constraints
 * HOW:  Simulate resource changes, verify tier switching and subsystem coordination
 */

#include <gtest/gtest.h>
#include <thread>
#include <chrono>

extern "C" {
#include "utils/platform/nimcp_platform_tier.h"
#include "portia/nimcp_portia_degradation.h"
#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_messages.h"
#include "utils/platform/nimcp_system_resources.h"
#include "utils/validation/nimcp_common.h"
}

class PortiaTierIntegrationTest : public ::testing::Test {
protected:
    degradation_state_t* degrade_state = nullptr;
    nimcp_bio_async_ctx_t* bio_ctx = nullptr;

    void SetUp() override {
        // Initialize bio-async system
        nimcp_bio_async_config_t bio_config = nimcp_bio_async_default_config();
        nimcp_bio_async_init(&bio_config);
        bio_ctx = nimcp_bio_async_get_context();

        // Initialize degradation system
        portia_degradation_config_t config = {
            .level_thresholds = {0.0f, 60.0f, 75.0f, 85.0f, 95.0f},
            .hysteresis_ms = 1000,
            .enable_auto_degrade = true,
            .enable_auto_restore = true,
            .restore_threshold = 10.0f
        };

        degrade_state = portia_degradation_init(&config, bio_ctx);
        ASSERT_NE(degrade_state, nullptr);

        // Register test features
        degradation_feature_t features[] = {
            {FEATURE_PLASTICITY, "plasticity", DEGRADATION_LEVEL_MINOR, 0.3f, false, true},
            {FEATURE_LEARNING, "learning", DEGRADATION_LEVEL_MODERATE, 0.4f, false, true},
            {FEATURE_EMOTIONS, "emotions", DEGRADATION_LEVEL_MODERATE, 0.2f, false, true},
            {FEATURE_PLANNING, "planning", DEGRADATION_LEVEL_SEVERE, 0.5f, false, true},
        };

        for (size_t i = 0; i < sizeof(features)/sizeof(features[0]); i++) {
            ASSERT_EQ(portia_degradation_register_feature(degrade_state, &features[i]), NIMCP_OK);
        }
    }

    void TearDown() override {
        if (degrade_state) {
            portia_degradation_cleanup(degrade_state);
            degrade_state = nullptr;
        }
        nimcp_bio_async_shutdown();
    }
};

//=============================================================================
// TEST SUITE 1: Tier Detection and Configuration
//=============================================================================

TEST_F(PortiaTierIntegrationTest, TierDetection_DetectsCurrentPlatform) {
    // Detect platform tier
    platform_tier_t tier = platform_tier_detect();

    // Should be a valid tier
    EXPECT_GE(tier, PLATFORM_TIER_FULL);
    EXPECT_LT(tier, PLATFORM_TIER_COUNT);

    // Should return a consistent name
    const char* name = platform_tier_get_name(tier);
    EXPECT_NE(name, nullptr);
    EXPECT_GT(strlen(name), 0);
}

TEST_F(PortiaTierIntegrationTest, TierConfiguration_ProvidesValidConfig) {
    platform_tier_t tier = platform_tier_detect();
    platform_tier_config_t config = platform_tier_get_config(tier);

    // Basic sanity checks
    EXPECT_GT(config.max_neurons, 0u);
    EXPECT_GT(config.initial_neurons, 0u);
    EXPECT_LE(config.initial_neurons, config.max_neurons);
    EXPECT_GT(config.memory_budget_mb, 0u);
    EXPECT_GT(config.max_threads, 0u);
}

TEST_F(PortiaTierIntegrationTest, TierConfiguration_ModuleEnablingMatchesTier) {
    // Full tier should enable most modules
    EXPECT_TRUE(platform_tier_can_enable_module(PLATFORM_TIER_FULL,
                                                  COGNITIVE_MODULE_CURIOSITY));
    EXPECT_TRUE(platform_tier_can_enable_module(PLATFORM_TIER_FULL,
                                                  COGNITIVE_MODULE_META_LEARNING));

    // Minimal tier should have limited modules
    EXPECT_TRUE(platform_tier_can_enable_module(PLATFORM_TIER_MINIMAL,
                                                 COGNITIVE_MODULE_ATTENTION));
    EXPECT_FALSE(platform_tier_can_enable_module(PLATFORM_TIER_MINIMAL,
                                                   COGNITIVE_MODULE_CURIOSITY));
}

//=============================================================================
// TEST SUITE 2: Tier Switch Triggers Degradation
//=============================================================================

TEST_F(PortiaTierIntegrationTest, TierSwitch_HighResourceUsageTriggersMinorDegradation) {
    // Initially at normal level
    degradation_level_t level;
    uint32_t active_features;
    float resource_usage;

    ASSERT_EQ(portia_degradation_get_state(degrade_state, &level,
                                            &active_features, &resource_usage), NIMCP_OK);
    EXPECT_EQ(level, DEGRADATION_LEVEL_NONE);

    // Simulate high resource usage (triggers MINOR degradation at 60%)
    ASSERT_EQ(portia_degradation_evaluate(degrade_state, 65.0f, bio_ctx), NIMCP_OK);

    // Should now be at MINOR degradation
    ASSERT_EQ(portia_degradation_get_state(degrade_state, &level,
                                            &active_features, &resource_usage), NIMCP_OK);
    EXPECT_EQ(level, DEGRADATION_LEVEL_MINOR);

    // Plasticity should be disabled at MINOR
    bool enabled;
    ASSERT_EQ(portia_degradation_is_feature_enabled(degrade_state,
                                                      FEATURE_PLASTICITY, &enabled), NIMCP_OK);
    EXPECT_FALSE(enabled);
}

TEST_F(PortiaTierIntegrationTest, TierSwitch_ProgressiveDegradation) {
    // Progress through degradation levels
    degradation_level_t level;
    uint32_t active_features;
    float resource_usage;

    // Start normal
    ASSERT_EQ(portia_degradation_get_state(degrade_state, &level,
                                            &active_features, &resource_usage), NIMCP_OK);
    uint32_t initial_features = active_features;

    // Trigger MINOR (60%)
    ASSERT_EQ(portia_degradation_evaluate(degrade_state, 65.0f, bio_ctx), NIMCP_OK);
    ASSERT_EQ(portia_degradation_get_state(degrade_state, &level,
                                            &active_features, &resource_usage), NIMCP_OK);
    EXPECT_EQ(level, DEGRADATION_LEVEL_MINOR);
    EXPECT_LT(active_features, initial_features);

    // Trigger MODERATE (75%)
    ASSERT_EQ(portia_degradation_evaluate(degrade_state, 80.0f, bio_ctx), NIMCP_OK);
    ASSERT_EQ(portia_degradation_get_state(degrade_state, &level,
                                            &active_features, &resource_usage), NIMCP_OK);
    EXPECT_EQ(level, DEGRADATION_LEVEL_MODERATE);

    // Learning and emotions should be disabled at MODERATE
    bool learning_enabled, emotions_enabled;
    ASSERT_EQ(portia_degradation_is_feature_enabled(degrade_state,
                                                      FEATURE_LEARNING, &learning_enabled), NIMCP_OK);
    ASSERT_EQ(portia_degradation_is_feature_enabled(degrade_state,
                                                      FEATURE_EMOTIONS, &emotions_enabled), NIMCP_OK);
    EXPECT_FALSE(learning_enabled);
    EXPECT_FALSE(emotions_enabled);

    // Trigger SEVERE (85%)
    ASSERT_EQ(portia_degradation_evaluate(degrade_state, 90.0f, bio_ctx), NIMCP_OK);
    ASSERT_EQ(portia_degradation_get_state(degrade_state, &level,
                                            &active_features, &resource_usage), NIMCP_OK);
    EXPECT_EQ(level, DEGRADATION_LEVEL_SEVERE);

    // Planning should be disabled at SEVERE
    bool planning_enabled;
    ASSERT_EQ(portia_degradation_is_feature_enabled(degrade_state,
                                                      FEATURE_PLANNING, &planning_enabled), NIMCP_OK);
    EXPECT_FALSE(planning_enabled);
}

//=============================================================================
// TEST SUITE 3: Tier Switch Updates Power Profile
//=============================================================================

TEST_F(PortiaTierIntegrationTest, TierSwitch_PowerProfileUpdates) {
    // Get configs for different tiers
    platform_tier_config_t full_config = platform_tier_get_config(PLATFORM_TIER_FULL);
    platform_tier_config_t minimal_config = platform_tier_get_config(PLATFORM_TIER_MINIMAL);

    // Minimal should have lower resource budgets
    EXPECT_LT(minimal_config.max_neurons, full_config.max_neurons);
    EXPECT_LT(minimal_config.memory_budget_mb, full_config.memory_budget_mb);
    EXPECT_LE(minimal_config.max_threads, full_config.max_threads);

    // Minimal should have fewer enabled modules
    EXPECT_LT(__builtin_popcount(minimal_config.cognitive_modules_enabled),
              __builtin_popcount(full_config.cognitive_modules_enabled));
}

//=============================================================================
// TEST SUITE 4: Multiple Subsystems Coordinate During Switch
//=============================================================================

TEST_F(PortiaTierIntegrationTest, TierSwitch_SubsystemCoordination) {
    // Register multiple features across different subsystems
    degradation_feature_t subsystem_features[] = {
        {FEATURE_MEMORY_LONG, "long_term_memory", DEGRADATION_LEVEL_MODERATE, 0.6f, false, true},
        {FEATURE_MEMORY_WORKING, "working_memory", DEGRADATION_LEVEL_SEVERE, 0.3f, false, true},
        {FEATURE_SENSORS_FULL, "full_sensors", DEGRADATION_LEVEL_MINOR, 0.4f, false, true},
        {FEATURE_COMMUNICATION, "swarm_comm", DEGRADATION_LEVEL_MODERATE, 0.5f, false, true},
    };

    for (size_t i = 0; i < sizeof(subsystem_features)/sizeof(subsystem_features[0]); i++) {
        ASSERT_EQ(portia_degradation_register_feature(degrade_state,
                                                       &subsystem_features[i]), NIMCP_OK);
    }

    // Trigger MODERATE degradation
    ASSERT_EQ(portia_degradation_evaluate(degrade_state, 80.0f, bio_ctx), NIMCP_OK);

    // Check which subsystems are affected
    bool sensors_enabled, comm_enabled, ltm_enabled, wm_enabled;

    ASSERT_EQ(portia_degradation_is_feature_enabled(degrade_state,
                                                      FEATURE_SENSORS_FULL, &sensors_enabled), NIMCP_OK);
    ASSERT_EQ(portia_degradation_is_feature_enabled(degrade_state,
                                                      FEATURE_COMMUNICATION, &comm_enabled), NIMCP_OK);
    ASSERT_EQ(portia_degradation_is_feature_enabled(degrade_state,
                                                      FEATURE_MEMORY_LONG, &ltm_enabled), NIMCP_OK);
    ASSERT_EQ(portia_degradation_is_feature_enabled(degrade_state,
                                                      FEATURE_MEMORY_WORKING, &wm_enabled), NIMCP_OK);

    // At MODERATE: sensors and comm should be off, working memory still on
    EXPECT_FALSE(sensors_enabled);
    EXPECT_FALSE(comm_enabled);
    EXPECT_FALSE(ltm_enabled);
    EXPECT_TRUE(wm_enabled);  // Only disabled at SEVERE
}

TEST_F(PortiaTierIntegrationTest, TierSwitch_AutoRestore) {
    // Trigger degradation
    ASSERT_EQ(portia_degradation_evaluate(degrade_state, 80.0f, bio_ctx), NIMCP_OK);

    degradation_level_t level;
    uint32_t active_features;
    float resource_usage;

    ASSERT_EQ(portia_degradation_get_state(degrade_state, &level,
                                            &active_features, &resource_usage), NIMCP_OK);
    EXPECT_EQ(level, DEGRADATION_LEVEL_MODERATE);

    // Wait for hysteresis period
    std::this_thread::sleep_for(std::chrono::milliseconds(1100));

    // Resource usage drops significantly (below threshold - restore_threshold)
    ASSERT_EQ(portia_degradation_evaluate(degrade_state, 50.0f, bio_ctx), NIMCP_OK);

    // Should restore to MINOR or NONE
    ASSERT_EQ(portia_degradation_get_state(degrade_state, &level,
                                            &active_features, &resource_usage), NIMCP_OK);
    EXPECT_LT(level, DEGRADATION_LEVEL_MODERATE);
}

TEST_F(PortiaTierIntegrationTest, TierSwitch_HysteresisPreventsThrashing) {
    // Trigger MINOR degradation
    ASSERT_EQ(portia_degradation_evaluate(degrade_state, 65.0f, bio_ctx), NIMCP_OK);

    degradation_level_t level1;
    uint32_t active_features1;
    float resource_usage1;
    ASSERT_EQ(portia_degradation_get_state(degrade_state, &level1,
                                            &active_features1, &resource_usage1), NIMCP_OK);

    // Immediately try to restore (within hysteresis period)
    ASSERT_EQ(portia_degradation_evaluate(degrade_state, 55.0f, bio_ctx), NIMCP_OK);

    degradation_level_t level2;
    uint32_t active_features2;
    float resource_usage2;
    ASSERT_EQ(portia_degradation_get_state(degrade_state, &level2,
                                            &active_features2, &resource_usage2), NIMCP_OK);

    // Should remain at same level due to hysteresis
    EXPECT_EQ(level1, level2);

    // Wait for hysteresis period to expire
    std::this_thread::sleep_for(std::chrono::milliseconds(1100));

    // Now should be able to restore
    ASSERT_EQ(portia_degradation_evaluate(degrade_state, 45.0f, bio_ctx), NIMCP_OK);

    degradation_level_t level3;
    uint32_t active_features3;
    float resource_usage3;
    ASSERT_EQ(portia_degradation_get_state(degrade_state, &level3,
                                            &active_features3, &resource_usage3), NIMCP_OK);

    // Should restore to NONE
    EXPECT_LT(level3, level1);
}

//=============================================================================
// TEST SUITE 5: Neuron Count Recommendations
//=============================================================================

TEST_F(PortiaTierIntegrationTest, NeuronRecommendation_ScalesWithTier) {
    system_resources_t resources;
    system_resources_query(&resources);

    uint32_t full_neurons = platform_tier_recommend_neuron_count(PLATFORM_TIER_FULL, &resources);
    uint32_t medium_neurons = platform_tier_recommend_neuron_count(PLATFORM_TIER_MEDIUM, &resources);
    uint32_t constrained_neurons = platform_tier_recommend_neuron_count(PLATFORM_TIER_CONSTRAINED, &resources);
    uint32_t minimal_neurons = platform_tier_recommend_neuron_count(PLATFORM_TIER_MINIMAL, &resources);

    // Should decrease with tier
    EXPECT_GT(full_neurons, medium_neurons);
    EXPECT_GT(medium_neurons, constrained_neurons);
    EXPECT_GT(constrained_neurons, minimal_neurons);

    // All should be reasonable values
    EXPECT_GT(minimal_neurons, 0u);
    EXPECT_LT(full_neurons, 100000000u);  // Sanity check
}
