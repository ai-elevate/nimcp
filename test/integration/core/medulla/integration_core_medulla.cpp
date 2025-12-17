/**
 * @file integration_core_medulla.cpp
 * @brief Integration tests for the Medulla Oblongata module
 *
 * WHAT: Integration tests verifying cross-module behavior
 * WHY:  Ensure arousal, protection, circadian, and coupling work together
 * HOW:  Test realistic scenarios requiring multi-module coordination
 */

#include <gtest/gtest.h>
#include <cstring>
#include <thread>
#include <chrono>

extern "C" {
#include "core/medulla/nimcp_medulla.h"
#include "utils/error/nimcp_error_codes.h"
}

//=============================================================================
// Test Fixture
//=============================================================================

class MedullaIntegrationTest : public ::testing::Test {
protected:
    medulla_t medulla = nullptr;

    void SetUp() override {
        medulla_config_t config = medulla_default_config();
        config.enable_bio_async = false;  // Disable for unit testing
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
// Arousal-Protection Integration Tests
//=============================================================================

TEST_F(MedullaIntegrationTest, ProtectionAffectsArousal) {
    // Start medulla
    EXPECT_EQ(medulla_start(medulla), NIMCP_SUCCESS);

    // Get initial protection level
    protection_level_t initial_level = medulla_get_protection_level(medulla);
    EXPECT_EQ(initial_level, PROTECTION_LEVEL_NORMAL);

    // Trigger emergency shutdown
    EXPECT_EQ(medulla_emergency_shutdown(medulla, "test threat"), NIMCP_SUCCESS);

    // Protection should be elevated
    protection_level_t after_level = medulla_get_protection_level(medulla);
    EXPECT_GT((int)after_level, (int)PROTECTION_LEVEL_NORMAL);
}

TEST_F(MedullaIntegrationTest, UpdateCycleIntegration) {
    EXPECT_EQ(medulla_start(medulla), NIMCP_SUCCESS);

    medulla_stats_t stats_before;
    medulla_get_stats(medulla, &stats_before);

    // Run multiple update cycles
    for (int i = 0; i < 50; i++) {
        EXPECT_EQ(medulla_update(medulla, 0.02f), NIMCP_SUCCESS);
    }

    medulla_stats_t stats_after;
    medulla_get_stats(medulla, &stats_after);

    // Verify updates were counted
    EXPECT_GT(stats_after.total_updates, stats_before.total_updates);
    EXPECT_GE(stats_after.total_updates, 50u);
}

//=============================================================================
// Circadian-Arousal Integration Tests
//=============================================================================

TEST_F(MedullaIntegrationTest, CircadianPhaseTracking) {
    EXPECT_EQ(medulla_start(medulla), NIMCP_SUCCESS);

    // Get initial phase
    circadian_phase_t phase = medulla_get_circadian_phase(medulla);
    EXPECT_GE((int)phase, 0);
    EXPECT_LT((int)phase, 8);

    // Run updates
    for (int i = 0; i < 100; i++) {
        medulla_update(medulla, 0.1f);
    }

    // Phase should still be valid
    phase = medulla_get_circadian_phase(medulla);
    EXPECT_GE((int)phase, 0);
    EXPECT_LT((int)phase, 8);
}

//=============================================================================
// State Transition Integration Tests
//=============================================================================

TEST_F(MedullaIntegrationTest, StateTransitions) {
    EXPECT_EQ(medulla_start(medulla), NIMCP_SUCCESS);

    medulla_stats_t stats;
    medulla_get_stats(medulla, &stats);
    EXPECT_EQ(stats.state, MEDULLA_STATE_RUNNING);

    // Request degraded state
    EXPECT_EQ(medulla_request_state_change(medulla, MEDULLA_STATE_DEGRADED), NIMCP_SUCCESS);

    medulla_get_stats(medulla, &stats);
    EXPECT_EQ(stats.state, MEDULLA_STATE_DEGRADED);

    // Request back to running
    EXPECT_EQ(medulla_request_state_change(medulla, MEDULLA_STATE_RUNNING), NIMCP_SUCCESS);

    medulla_get_stats(medulla, &stats);
    EXPECT_EQ(stats.state, MEDULLA_STATE_RUNNING);
}

TEST_F(MedullaIntegrationTest, EmergencyStateTransition) {
    EXPECT_EQ(medulla_start(medulla), NIMCP_SUCCESS);

    // Trigger emergency
    EXPECT_EQ(medulla_emergency_shutdown(medulla, "critical failure"), NIMCP_SUCCESS);

    medulla_stats_t stats;
    medulla_get_stats(medulla, &stats);

    // Should be in emergency or stopped state
    EXPECT_TRUE(stats.state == MEDULLA_STATE_EMERGENCY ||
                stats.protection_level >= PROTECTION_LEVEL_CRITICAL);
}

//=============================================================================
// Lifecycle Integration Tests
//=============================================================================

TEST_F(MedullaIntegrationTest, StartStopCycles) {
    // Multiple start/stop cycles should work
    for (int cycle = 0; cycle < 3; cycle++) {
        EXPECT_EQ(medulla_start(medulla), NIMCP_SUCCESS);

        // Run some updates
        for (int i = 0; i < 10; i++) {
            EXPECT_EQ(medulla_update(medulla, 0.016f), NIMCP_SUCCESS);
        }

        EXPECT_EQ(medulla_stop(medulla), NIMCP_SUCCESS);
    }
}

TEST_F(MedullaIntegrationTest, StatsAccumulation) {
    EXPECT_EQ(medulla_start(medulla), NIMCP_SUCCESS);

    // Run updates
    for (int i = 0; i < 100; i++) {
        medulla_update(medulla, 0.01f);
    }

    medulla_stats_t stats;
    medulla_get_stats(medulla, &stats);

    // Verify statistics accumulated
    EXPECT_GE(stats.total_updates, 100u);
    EXPECT_GE(stats.arousal_updates, 0u);
}

//=============================================================================
// Protection Level Progression Tests
//=============================================================================

TEST_F(MedullaIntegrationTest, ProtectionLevelProgression) {
    EXPECT_EQ(medulla_start(medulla), NIMCP_SUCCESS);

    // Initial level should be normal
    protection_level_t level = medulla_get_protection_level(medulla);
    EXPECT_EQ(level, PROTECTION_LEVEL_NORMAL);

    // After emergency, should be elevated
    medulla_emergency_shutdown(medulla, "test");
    level = medulla_get_protection_level(medulla);
    EXPECT_GE((int)level, (int)PROTECTION_LEVEL_CRITICAL);
}

//=============================================================================
// Arousal State Modulation Tests
//=============================================================================

TEST_F(MedullaIntegrationTest, ArousalLevelRange) {
    EXPECT_EQ(medulla_start(medulla), NIMCP_SUCCESS);

    medulla_stats_t stats;
    medulla_get_stats(medulla, &stats);

    // Arousal should be within valid range
    EXPECT_GE(stats.current_arousal, 0.0f);
    EXPECT_LE(stats.current_arousal, 1.0f);
}

//=============================================================================
// Concurrent Update Safety Tests
//=============================================================================

TEST_F(MedullaIntegrationTest, RapidUpdateSequence) {
    EXPECT_EQ(medulla_start(medulla), NIMCP_SUCCESS);

    // Rapid updates shouldn't cause issues
    for (int i = 0; i < 1000; i++) {
        int result = medulla_update(medulla, 0.001f);
        EXPECT_EQ(result, NIMCP_SUCCESS);
    }

    medulla_stats_t stats;
    medulla_get_stats(medulla, &stats);
    EXPECT_GE(stats.total_updates, 1000u);
}

//=============================================================================
// Main
//=============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
