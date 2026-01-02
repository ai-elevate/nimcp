/**
 * @file regression_core_medulla.cpp
 * @brief Regression tests for the Medulla Oblongata module
 *
 * WHAT: Regression tests ensuring stability and backward compatibility
 * WHY:  Prevent regressions in arousal, protection, circadian behavior
 * HOW:  Test known edge cases, boundary conditions, and historical bugs
 */

#include <gtest/gtest.h>
#include <cstring>
#include <cmath>

// Headers have their own extern "C" guards
#include "core/medulla/nimcp_medulla.h"
#include "utils/error/nimcp_error_codes.h"

//=============================================================================
// Test Fixture
//=============================================================================

class MedullaRegressionTest : public ::testing::Test {
protected:
    medulla_t medulla = nullptr;

    void SetUp() override {
        medulla_config_t config = medulla_default_config();
        medulla = medulla_create(&config);
        ASSERT_NE(medulla, nullptr);
    }

    void TearDown() override {
        if (medulla) {
            medulla_stop(medulla);
            medulla_destroy(medulla);
            medulla = nullptr;
        }
    }
};

//=============================================================================
// Boundary Condition Tests
//=============================================================================

TEST_F(MedullaRegressionTest, ZeroDeltaTimeUpdate) {
    EXPECT_EQ(medulla_start(medulla), NIMCP_SUCCESS);

    // Zero delta time should not crash
    int result = medulla_update(medulla, 0.0f);
    EXPECT_EQ(result, NIMCP_SUCCESS);
}

TEST_F(MedullaRegressionTest, VerySmallDeltaTime) {
    EXPECT_EQ(medulla_start(medulla), NIMCP_SUCCESS);

    // Very small delta should work
    for (int i = 0; i < 100; i++) {
        int result = medulla_update(medulla, 0.0001f);
        EXPECT_EQ(result, NIMCP_SUCCESS);
    }
}

TEST_F(MedullaRegressionTest, LargeDeltaTime) {
    EXPECT_EQ(medulla_start(medulla), NIMCP_SUCCESS);

    // Large delta time should work without overflow
    int result = medulla_update(medulla, 10.0f);
    EXPECT_EQ(result, NIMCP_SUCCESS);
}

//=============================================================================
// State Consistency Tests
//=============================================================================

TEST_F(MedullaRegressionTest, StatsConsistencyAfterUpdates) {
    EXPECT_EQ(medulla_start(medulla), NIMCP_SUCCESS);

    medulla_stats_t stats_before;
    medulla_get_stats(medulla, &stats_before);

    // Run updates
    for (int i = 0; i < 100; i++) {
        medulla_update(medulla, 0.016f);
    }

    medulla_stats_t stats_after;
    medulla_get_stats(medulla, &stats_after);

    // Stats should be monotonically increasing
    EXPECT_GE(stats_after.total_updates, stats_before.total_updates + 100);
    EXPECT_GE(stats_after.uptime_ms, stats_before.uptime_ms);
}

TEST_F(MedullaRegressionTest, ProtectionLevelNeverNegative) {
    EXPECT_EQ(medulla_start(medulla), NIMCP_SUCCESS);

    for (int i = 0; i < 50; i++) {
        medulla_update(medulla, 0.1f);
        protection_level_t level = medulla_get_protection_level(medulla);
        EXPECT_GE((int)level, 0);
        EXPECT_LE((int)level, (int)PROTECTION_LEVEL_SHUTDOWN);
    }
}

TEST_F(MedullaRegressionTest, CircadianPhaseAlwaysValid) {
    EXPECT_EQ(medulla_start(medulla), NIMCP_SUCCESS);

    for (int i = 0; i < 100; i++) {
        medulla_update(medulla, 0.5f);  // Larger steps to advance circadian
        circadian_phase_t phase = medulla_get_circadian_phase(medulla);
        EXPECT_GE((int)phase, 0);
        EXPECT_LT((int)phase, 8);
    }
}

//=============================================================================
// State Transition Regression Tests
//=============================================================================

TEST_F(MedullaRegressionTest, NoStateCorruptionOnEmergency) {
    EXPECT_EQ(medulla_start(medulla), NIMCP_SUCCESS);

    // Get pre-emergency state
    medulla_stats_t stats_before;
    medulla_get_stats(medulla, &stats_before);

    // Trigger emergency
    medulla_emergency_shutdown(medulla, "regression test");

    // Get post-emergency state
    medulla_stats_t stats_after;
    medulla_get_stats(medulla, &stats_after);

    // State should still be valid
    EXPECT_GE((int)stats_after.state, 0);
    EXPECT_LE((int)stats_after.state, (int)MEDULLA_STATE_STOPPING);

    // Emergency count should increment
    EXPECT_GE(stats_after.emergency_shutdowns, 1u);
}

TEST_F(MedullaRegressionTest, RecoveryFromDegradedState) {
    EXPECT_EQ(medulla_start(medulla), NIMCP_SUCCESS);

    // Go to degraded
    EXPECT_EQ(medulla_request_state_change(medulla, MEDULLA_STATE_DEGRADED), NIMCP_SUCCESS);

    medulla_stats_t stats;
    medulla_get_stats(medulla, &stats);
    EXPECT_EQ(stats.state, MEDULLA_STATE_DEGRADED);

    // Recover to running
    EXPECT_EQ(medulla_request_state_change(medulla, MEDULLA_STATE_RUNNING), NIMCP_SUCCESS);

    medulla_get_stats(medulla, &stats);
    EXPECT_EQ(stats.state, MEDULLA_STATE_RUNNING);
}

//=============================================================================
// Memory Safety Tests
//=============================================================================

TEST_F(MedullaRegressionTest, MultipleCreateDestroy) {
    // Multiple create/destroy cycles shouldn't leak memory
    for (int i = 0; i < 10; i++) {
        medulla_config_t config = medulla_default_config();
        medulla_t m = medulla_create(&config);
        ASSERT_NE(m, nullptr);

        medulla_start(m);
        medulla_update(m, 0.016f);
        medulla_stop(m);
        medulla_destroy(m);
    }
}

TEST_F(MedullaRegressionTest, NullPointerSafety) {
    // All functions should handle null gracefully
    EXPECT_LT(medulla_start(nullptr), 0);
    EXPECT_LT(medulla_stop(nullptr), 0);
    EXPECT_LT(medulla_update(nullptr, 0.016f), 0);
    EXPECT_LT(medulla_emergency_shutdown(nullptr, "test"), 0);
    EXPECT_LT(medulla_request_state_change(nullptr, MEDULLA_STATE_RUNNING), 0);

    medulla_stats_t stats;
    EXPECT_LT(medulla_get_stats(nullptr, &stats), 0);

    // These should return safe defaults
    EXPECT_FALSE(medulla_is_bio_async_connected(nullptr));

    medulla_destroy(nullptr);  // Should not crash
}

//=============================================================================
// Arousal State Regression Tests
//=============================================================================

TEST_F(MedullaRegressionTest, ArousalBoundedZeroToOne) {
    EXPECT_EQ(medulla_start(medulla), NIMCP_SUCCESS);

    for (int i = 0; i < 100; i++) {
        medulla_update(medulla, 0.1f);

        medulla_stats_t stats;
        medulla_get_stats(medulla, &stats);

        EXPECT_GE(stats.current_arousal, 0.0f);
        EXPECT_LE(stats.current_arousal, 1.0f);
    }
}

//=============================================================================
// Stress Test Regressions
//=============================================================================

TEST_F(MedullaRegressionTest, RapidStartStopCycles) {
    for (int i = 0; i < 20; i++) {
        EXPECT_EQ(medulla_start(medulla), NIMCP_SUCCESS);
        EXPECT_EQ(medulla_stop(medulla), NIMCP_SUCCESS);
    }
}

TEST_F(MedullaRegressionTest, HighFrequencyUpdates) {
    EXPECT_EQ(medulla_start(medulla), NIMCP_SUCCESS);

    // Simulate 60fps for 10 "seconds"
    for (int i = 0; i < 600; i++) {
        int result = medulla_update(medulla, 0.016667f);
        EXPECT_EQ(result, NIMCP_SUCCESS);
    }

    medulla_stats_t stats;
    medulla_get_stats(medulla, &stats);
    EXPECT_GE(stats.total_updates, 600u);
}

//=============================================================================
// Configuration Regression Tests
//=============================================================================

TEST_F(MedullaRegressionTest, DefaultConfigStability) {
    medulla_config_t config1 = medulla_default_config();
    medulla_config_t config2 = medulla_default_config();

    // Default configs should be consistent
    EXPECT_EQ(config1.update_interval_ms, config2.update_interval_ms);
    EXPECT_FLOAT_EQ(config1.arousal.baseline_arousal, config2.arousal.baseline_arousal);
    EXPECT_FLOAT_EQ(config1.protection.health_threshold_critical,
                    config2.protection.health_threshold_critical);
}

TEST_F(MedullaRegressionTest, CreateWithNullConfigDefaults) {
    medulla_t m = medulla_create(nullptr);
    ASSERT_NE(m, nullptr);

    // Should start successfully with default config
    EXPECT_EQ(medulla_start(m), NIMCP_SUCCESS);
    EXPECT_EQ(medulla_stop(m), NIMCP_SUCCESS);

    medulla_destroy(m);
}

//=============================================================================
// Circadian Time Regression Tests
//=============================================================================

TEST_F(MedullaRegressionTest, CircadianTimeProgression) {
    EXPECT_EQ(medulla_start(medulla), NIMCP_SUCCESS);

    medulla_stats_t stats_before;
    medulla_get_stats(medulla, &stats_before);

    // Run many updates to advance circadian time
    for (int i = 0; i < 1000; i++) {
        medulla_update(medulla, 0.1f);
    }

    medulla_stats_t stats_after;
    medulla_get_stats(medulla, &stats_after);

    // Circadian time should be valid (0-24 hours)
    EXPECT_GE(stats_after.circadian_time_hours, 0.0f);
    EXPECT_LT(stats_after.circadian_time_hours, 24.0f);
}

//=============================================================================
// Main
//=============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
