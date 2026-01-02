/**
 * @file test_medulla.cpp
 * @brief Unit tests for the medulla orchestrator module
 *
 * WHAT: Tests for the main medulla coordination system
 * WHY:  Ensure proper orchestration of medulla subsystems
 * HOW:  Use GoogleTest framework with lifecycle and state validation
 */

#include <gtest/gtest.h>
#include <cstring>

// Headers have their own extern "C" guards
#include "core/medulla/nimcp_medulla.h"
#include "utils/error/nimcp_error_codes.h"

//=============================================================================
// Test Fixture
//=============================================================================

class MedullaTest : public ::testing::Test {
protected:
    medulla_t medulla = nullptr;

    void SetUp() override {
        medulla_config_t config = medulla_default_config();
        medulla = medulla_create(&config);
        ASSERT_NE(medulla, nullptr);
    }

    void TearDown() override {
        if (medulla) {
            medulla_destroy(medulla);
            medulla = nullptr;
        }
    }
};

//=============================================================================
// Configuration Tests
//=============================================================================

TEST_F(MedullaTest, DefaultConfig) {
    medulla_config_t config = medulla_default_config();

    // Verify arousal defaults
    EXPECT_GE(config.arousal.baseline_arousal, 0.0f);
    EXPECT_LE(config.arousal.baseline_arousal, 1.0f);

    // Verify update interval
    EXPECT_GT(config.update_interval_ms, 0u);
}

TEST_F(MedullaTest, CreateWithNullConfig) {
    medulla_t m = medulla_create(nullptr);
    EXPECT_NE(m, nullptr);
    if (m) medulla_destroy(m);
}

TEST_F(MedullaTest, DestroyNull) {
    medulla_destroy(nullptr);
}

//=============================================================================
// Lifecycle Tests
//=============================================================================

TEST_F(MedullaTest, StartStop) {
    int result = medulla_start(medulla);
    EXPECT_EQ(result, NIMCP_SUCCESS);

    result = medulla_stop(medulla);
    EXPECT_EQ(result, NIMCP_SUCCESS);
}

TEST_F(MedullaTest, StartNull) {
    int result = medulla_start(nullptr);
    EXPECT_LT(result, 0);  // Any negative error code
}

TEST_F(MedullaTest, StopNull) {
    int result = medulla_stop(nullptr);
    EXPECT_LT(result, 0);  // Any negative error code
}

TEST_F(MedullaTest, MultipleStartStop) {
    // First start/stop cycle
    EXPECT_EQ(medulla_start(medulla), NIMCP_SUCCESS);
    EXPECT_EQ(medulla_stop(medulla), NIMCP_SUCCESS);

    // Second start/stop cycle
    EXPECT_EQ(medulla_start(medulla), NIMCP_SUCCESS);
    EXPECT_EQ(medulla_stop(medulla), NIMCP_SUCCESS);
}

//=============================================================================
// Update Tests
//=============================================================================

TEST_F(MedullaTest, Update) {
    medulla_start(medulla);

    int result = medulla_update(medulla, 0.1f);  // 100ms
    EXPECT_EQ(result, NIMCP_SUCCESS);

    medulla_stop(medulla);
}

TEST_F(MedullaTest, UpdateNull) {
    int result = medulla_update(nullptr, 0.1f);
    EXPECT_LT(result, 0);  // Any negative error code
}

TEST_F(MedullaTest, MultipleUpdates) {
    medulla_start(medulla);

    for (int i = 0; i < 100; i++) {
        int result = medulla_update(medulla, 0.016f);  // ~60 fps
        EXPECT_EQ(result, NIMCP_SUCCESS);
    }

    medulla_stop(medulla);
}

//=============================================================================
// Emergency Shutdown Tests
//=============================================================================

TEST_F(MedullaTest, EmergencyShutdown) {
    medulla_start(medulla);

    int result = medulla_emergency_shutdown(medulla, "test shutdown");
    EXPECT_EQ(result, NIMCP_SUCCESS);

    // Verify protection level is at max
    protection_level_t level = medulla_get_protection_level(medulla);
    EXPECT_GE((int)level, (int)PROTECTION_LEVEL_CRITICAL);

    medulla_stop(medulla);
}

TEST_F(MedullaTest, EmergencyShutdownNull) {
    int result = medulla_emergency_shutdown(nullptr, "test");
    EXPECT_LT(result, 0);  // Any negative error code
}

//=============================================================================
// State Query Tests
//=============================================================================

TEST_F(MedullaTest, GetProtectionLevel) {
    protection_level_t level = medulla_get_protection_level(medulla);
    EXPECT_GE((int)level, 0);
    EXPECT_LE((int)level, (int)PROTECTION_LEVEL_SHUTDOWN);
}

TEST_F(MedullaTest, GetCircadianPhase) {
    circadian_phase_t phase = medulla_get_circadian_phase(medulla);
    EXPECT_GE((int)phase, 0);
    EXPECT_LT((int)phase, 8);  // 8 phases
}

//=============================================================================
// Statistics Tests
//=============================================================================

TEST_F(MedullaTest, GetStats) {
    medulla_stats_t stats;
    int result = medulla_get_stats(medulla, &stats);
    EXPECT_EQ(result, NIMCP_SUCCESS);

    // Verify state is valid
    EXPECT_GE((int)stats.state, 0);
    EXPECT_LE((int)stats.state, (int)MEDULLA_STATE_STOPPING);
}

TEST_F(MedullaTest, GetStatsNull) {
    medulla_stats_t stats;
    int result = medulla_get_stats(nullptr, &stats);
    EXPECT_LT(result, 0);  // Any negative error code

    result = medulla_get_stats(medulla, nullptr);
    EXPECT_LT(result, 0);  // Any negative error code
}

TEST_F(MedullaTest, StatsAfterUpdates) {
    medulla_start(medulla);

    // Run some updates
    for (int i = 0; i < 10; i++) {
        medulla_update(medulla, 0.1f);
    }

    medulla_stats_t stats;
    medulla_get_stats(medulla, &stats);

    // Verify update count increased
    EXPECT_GE(stats.total_updates, 10u);

    medulla_stop(medulla);
}

//=============================================================================
// State Change Tests
//=============================================================================

TEST_F(MedullaTest, RequestStateChange) {
    medulla_start(medulla);

    int result = medulla_request_state_change(medulla, MEDULLA_STATE_DEGRADED);
    EXPECT_EQ(result, NIMCP_SUCCESS);

    medulla_stop(medulla);
}

TEST_F(MedullaTest, RequestStateChangeNull) {
    int result = medulla_request_state_change(nullptr, MEDULLA_STATE_RUNNING);
    EXPECT_LT(result, 0);  // Any negative error code
}

//=============================================================================
// Bio-async Tests
//=============================================================================

TEST_F(MedullaTest, BioAsyncConnection) {
    bool connected = medulla_is_bio_async_connected(medulla);
    EXPECT_FALSE(connected);

    int result = medulla_connect_bio_async(medulla);
    // Result depends on router availability

    medulla_disconnect_bio_async(medulla);
    connected = medulla_is_bio_async_connected(medulla);
    EXPECT_FALSE(connected);
}

TEST_F(MedullaTest, BioAsyncNullState) {
    bool connected = medulla_is_bio_async_connected(nullptr);
    EXPECT_FALSE(connected);
}

//=============================================================================
// Integration Connection Tests
//=============================================================================

TEST_F(MedullaTest, ConnectHealthMonitorNull) {
    int result = medulla_connect_health_monitor(medulla, nullptr);
    // May return success with null if function ignores null (just logging)
    EXPECT_TRUE(result == NIMCP_SUCCESS || result < 0);
}

TEST_F(MedullaTest, ConnectRecoveryNull) {
    int result = medulla_connect_recovery_system(medulla, nullptr);
    EXPECT_TRUE(result == NIMCP_SUCCESS || result < 0);
}

TEST_F(MedullaTest, ConnectSleepWakeNull) {
    int result = medulla_connect_sleep_wake(medulla, nullptr);
    EXPECT_TRUE(result == NIMCP_SUCCESS || result < 0);
}

TEST_F(MedullaTest, ConnectNeuromodulatorsNull) {
    int result = medulla_connect_neuromodulators(medulla, nullptr);
    EXPECT_TRUE(result == NIMCP_SUCCESS || result < 0);
}

//=============================================================================
// Main
//=============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
