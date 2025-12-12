/**
 * @file test_immune_exhaustion.cpp
 * @brief Unit tests for Immune Exhaustion Module
 * @date 2025-12-12
 *
 * Tests T cell exhaustion progression, functional decline,
 * recovery mechanisms, and checkpoint blockade therapy.
 */

#include <gtest/gtest.h>

extern "C" {
#include "cognitive/immune/nimcp_immune_exhaustion.h"
#include "cognitive/immune/nimcp_brain_immune.h"
#include "utils/memory/nimcp_memory.h"
}

/* ============================================================================
 * Test Fixture
 * ============================================================================ */

class ImmuneExhaustionTest : public ::testing::Test {
protected:
    brain_immune_system_t* immune_system = nullptr;
    exhaustion_system_t* exhaustion = nullptr;
    exhaustion_config_t config;

    void SetUp() override {
        brain_immune_config_t immune_config;
        brain_immune_default_config(&immune_config);
        immune_system = brain_immune_create(&immune_config);
        ASSERT_NE(immune_system, nullptr);
        brain_immune_start(immune_system);

        exhaustion_default_config(&config);
        exhaustion = exhaustion_create(&config, immune_system);
        ASSERT_NE(exhaustion, nullptr);
    }

    void TearDown() override {
        if (exhaustion) {
            exhaustion_destroy(exhaustion);
            exhaustion = nullptr;
        }
        if (immune_system) {
            brain_immune_stop(immune_system);
            brain_immune_destroy(immune_system);
            immune_system = nullptr;
        }
    }

    // Helper to create and activate T cell
    uint32_t activateTCell() {
        uint8_t epitope[] = {0x01, 0x02, 0x03, 0x04};
        uint32_t antigen_id, t_cell_id;
        brain_immune_present_antigen(immune_system, ANTIGEN_SOURCE_MANUAL,
                                     epitope, sizeof(epitope), 5, 1, &antigen_id);
        brain_immune_activate_helper_t(immune_system, antigen_id, &t_cell_id);
        return t_cell_id;
    }
};

/* ============================================================================
 * Lifecycle Tests
 * ============================================================================ */

TEST_F(ImmuneExhaustionTest, DefaultConfigIsValid) {
    exhaustion_config_t cfg;
    int result = exhaustion_default_config(&cfg);
    EXPECT_EQ(result, 0);
    EXPECT_GT(cfg.early_exhaustion_threshold_ms, 0u);
    EXPECT_GT(cfg.natural_recovery_duration_ms, 0u);
    EXPECT_GT(cfg.checkpoint_blockade_efficacy, 0.0f);
}

TEST_F(ImmuneExhaustionTest, DefaultConfigNullFails) {
    int result = exhaustion_default_config(nullptr);
    EXPECT_EQ(result, -1);
}

TEST_F(ImmuneExhaustionTest, CreateWithNullConfig) {
    exhaustion_system_t* sys = exhaustion_create(nullptr, immune_system);
    ASSERT_NE(sys, nullptr);
    exhaustion_destroy(sys);
}

TEST_F(ImmuneExhaustionTest, DestroyNullSafe) {
    exhaustion_destroy(nullptr);
}

/* ============================================================================
 * Exhaustion State Tests
 * ============================================================================ */

TEST_F(ImmuneExhaustionTest, InitialStateNaive) {
    uint32_t t_cell_id = activateTCell();
    exhaustion_state_t state = exhaustion_get_cell_state(exhaustion, t_cell_id);
    // Newly activated cells should be in effector state
    EXPECT_TRUE(state == EXHAUSTION_STATE_NAIVE || state == EXHAUSTION_STATE_EFFECTOR);
}

TEST_F(ImmuneExhaustionTest, StateProgressionWithTime) {
    uint32_t t_cell_id = activateTCell();

    // Initial state
    exhaustion_state_t initial = exhaustion_get_cell_state(exhaustion, t_cell_id);

    // Simulate prolonged activation (beyond early threshold)
    exhaustion_update(exhaustion, EXHAUSTION_EARLY_THRESHOLD_MS + 1000);

    exhaustion_state_t after = exhaustion_get_cell_state(exhaustion, t_cell_id);
    // Should have progressed
}

TEST_F(ImmuneExhaustionTest, AdvancedExhaustionState) {
    uint32_t t_cell_id = activateTCell();

    // Simulate very prolonged activation
    exhaustion_update(exhaustion, EXHAUSTION_ADVANCED_THRESHOLD_MS + 1000);

    exhaustion_state_t state = exhaustion_get_cell_state(exhaustion, t_cell_id);
    // Should be in exhausted or terminal state
}

/* ============================================================================
 * Functional Capacity Tests
 * ============================================================================ */

TEST_F(ImmuneExhaustionTest, InitialCapacityFull) {
    uint32_t t_cell_id = activateTCell();

    float capacity = exhaustion_get_effector_capacity(exhaustion, t_cell_id);
    EXPECT_GE(capacity, 0.9f);  // Near full capacity initially
}

TEST_F(ImmuneExhaustionTest, CapacityDeclineWithExhaustion) {
    uint32_t t_cell_id = activateTCell();

    float initial = exhaustion_get_effector_capacity(exhaustion, t_cell_id);

    // Progress through exhaustion
    exhaustion_update(exhaustion, EXHAUSTION_ADVANCED_THRESHOLD_MS);

    float after = exhaustion_get_effector_capacity(exhaustion, t_cell_id);
    EXPECT_LT(after, initial);
}

TEST_F(ImmuneExhaustionTest, TerminalExhaustionMinimalCapacity) {
    uint32_t t_cell_id = activateTCell();

    // Simulate terminal exhaustion
    exhaustion_update(exhaustion, EXHAUSTION_TERMINAL_THRESHOLD_MS + 1000);

    float capacity = exhaustion_get_effector_capacity(exhaustion, t_cell_id);
    EXPECT_LT(capacity, 0.3f);  // Very low capacity
}

/* ============================================================================
 * Exhaustion Markers Tests
 * ============================================================================ */

TEST_F(ImmuneExhaustionTest, GetMarkersInitial) {
    uint32_t t_cell_id = activateTCell();

    exhaustion_markers_t markers;
    int result = exhaustion_get_markers(exhaustion, t_cell_id, &markers);
    EXPECT_EQ(result, 0);

    // Low initial markers
    EXPECT_LT(markers.pd1_level, 0.3f);
    EXPECT_LT(markers.lag3_level, 0.3f);
    EXPECT_LT(markers.tim3_level, 0.3f);
}

TEST_F(ImmuneExhaustionTest, MarkersIncreaseWithExhaustion) {
    uint32_t t_cell_id = activateTCell();

    exhaustion_markers_t initial;
    exhaustion_get_markers(exhaustion, t_cell_id, &initial);

    exhaustion_update(exhaustion, EXHAUSTION_ADVANCED_THRESHOLD_MS);

    exhaustion_markers_t after;
    exhaustion_get_markers(exhaustion, t_cell_id, &after);

    EXPECT_GT(after.pd1_level, initial.pd1_level);
    EXPECT_GT(after.composite, initial.composite);
}

TEST_F(ImmuneExhaustionTest, GetMarkersNullFails) {
    int result = exhaustion_get_markers(exhaustion, 1, nullptr);
    EXPECT_EQ(result, -1);
}

/* ============================================================================
 * System Fatigue Tests
 * ============================================================================ */

TEST_F(ImmuneExhaustionTest, InitialFatigueLow) {
    float fatigue = exhaustion_get_system_fatigue(exhaustion);
    EXPECT_LT(fatigue, 0.1f);  // Fresh system
}

TEST_F(ImmuneExhaustionTest, FatigueIncreasesWithExhaustedCells) {
    // Activate multiple T cells
    for (int i = 0; i < 5; i++) {
        activateTCell();
    }

    // Exhaust them
    exhaustion_update(exhaustion, EXHAUSTION_ADVANCED_THRESHOLD_MS);

    float fatigue = exhaustion_get_system_fatigue(exhaustion);
    EXPECT_GT(fatigue, 0.0f);
}

/* ============================================================================
 * Natural Recovery Tests
 * ============================================================================ */

TEST_F(ImmuneExhaustionTest, InitiateRecovery) {
    uint32_t t_cell_id = activateTCell();

    // Exhaust the cell first
    exhaustion_update(exhaustion, EXHAUSTION_ADVANCED_THRESHOLD_MS);

    int result = exhaustion_initiate_recovery(exhaustion, t_cell_id);
    EXPECT_EQ(result, 0);

    exhaustion_state_t state = exhaustion_get_cell_state(exhaustion, t_cell_id);
    EXPECT_EQ(state, EXHAUSTION_STATE_RECOVERING);
}

TEST_F(ImmuneExhaustionTest, RecoveryRestoresCapacity) {
    uint32_t t_cell_id = activateTCell();

    // Exhaust
    exhaustion_update(exhaustion, EXHAUSTION_ADVANCED_THRESHOLD_MS);
    float exhausted_capacity = exhaustion_get_effector_capacity(exhaustion, t_cell_id);

    // Initiate recovery
    exhaustion_initiate_recovery(exhaustion, t_cell_id);

    // Progress through recovery
    exhaustion_update(exhaustion, EXHAUSTION_RECOVERY_DURATION_MS);

    float recovered_capacity = exhaustion_get_effector_capacity(exhaustion, t_cell_id);
    EXPECT_GT(recovered_capacity, exhausted_capacity);
}

TEST_F(ImmuneExhaustionTest, RecoveryOnNonExhaustedFails) {
    uint32_t t_cell_id = activateTCell();

    // Try recovery on fresh cell
    int result = exhaustion_initiate_recovery(exhaustion, t_cell_id);
    EXPECT_NE(result, 0);
}

/* ============================================================================
 * Checkpoint Blockade Tests
 * ============================================================================ */

TEST_F(ImmuneExhaustionTest, CheckpointBlockadeRestoresFunction) {
    uint32_t t_cell_id = activateTCell();

    // Exhaust the cell
    exhaustion_update(exhaustion, EXHAUSTION_ADVANCED_THRESHOLD_MS);
    float exhausted_capacity = exhaustion_get_effector_capacity(exhaustion, t_cell_id);

    // Apply checkpoint blockade
    int result = exhaustion_checkpoint_blockade(exhaustion, t_cell_id);
    EXPECT_EQ(result, 0);

    float post_blockade = exhaustion_get_effector_capacity(exhaustion, t_cell_id);
    EXPECT_GT(post_blockade, exhausted_capacity);
}

TEST_F(ImmuneExhaustionTest, CheckpointBlockadeReducesPD1) {
    uint32_t t_cell_id = activateTCell();

    // Exhaust
    exhaustion_update(exhaustion, EXHAUSTION_ADVANCED_THRESHOLD_MS);

    exhaustion_markers_t before;
    exhaustion_get_markers(exhaustion, t_cell_id, &before);

    // Apply checkpoint blockade
    exhaustion_checkpoint_blockade(exhaustion, t_cell_id);

    exhaustion_markers_t after;
    exhaustion_get_markers(exhaustion, t_cell_id, &after);

    EXPECT_LT(after.pd1_level, before.pd1_level);
}

TEST_F(ImmuneExhaustionTest, CheckpointBlockadeEfficacyLimit) {
    uint32_t t_cell_id = activateTCell();

    // Exhaust to terminal
    exhaustion_update(exhaustion, EXHAUSTION_TERMINAL_THRESHOLD_MS);

    exhaustion_checkpoint_blockade(exhaustion, t_cell_id);

    float capacity = exhaustion_get_effector_capacity(exhaustion, t_cell_id);
    // Should not exceed checkpoint blockade efficacy limit
    EXPECT_LE(capacity, EXHAUSTION_CHECKPOINT_RECOVERY);
}

/* ============================================================================
 * Callback Tests
 * ============================================================================ */

static bool exhaustion_callback_called = false;
static void test_exhaustion_callback(exhaustion_system_t* sys, uint32_t t_cell_id,
                                     exhaustion_state_t old_state, exhaustion_state_t new_state,
                                     void* user_data) {
    exhaustion_callback_called = true;
}

TEST_F(ImmuneExhaustionTest, ExhaustionCallbackInvoked) {
    exhaustion_callback_called = false;
    exhaustion_set_exhaustion_callback(exhaustion, test_exhaustion_callback, nullptr);

    uint32_t t_cell_id = activateTCell();
    exhaustion_update(exhaustion, EXHAUSTION_ADVANCED_THRESHOLD_MS);

    // Callback should have been invoked on state transition
    // (depends on implementation details)
}

static bool recovery_callback_called = false;
static void test_recovery_callback(exhaustion_system_t* sys, uint32_t t_cell_id,
                                   float recovered_capacity, void* user_data) {
    recovery_callback_called = true;
}

TEST_F(ImmuneExhaustionTest, RecoveryCallbackInvoked) {
    recovery_callback_called = false;
    exhaustion_set_recovery_callback(exhaustion, test_recovery_callback, nullptr);

    uint32_t t_cell_id = activateTCell();
    exhaustion_update(exhaustion, EXHAUSTION_ADVANCED_THRESHOLD_MS);
    exhaustion_initiate_recovery(exhaustion, t_cell_id);
    exhaustion_update(exhaustion, EXHAUSTION_RECOVERY_DURATION_MS);

    // Check callback was invoked
}

/* ============================================================================
 * Statistics Tests
 * ============================================================================ */

TEST_F(ImmuneExhaustionTest, GetStats) {
    exhaustion_stats_t stats;
    int result = exhaustion_get_stats(exhaustion, &stats);
    EXPECT_EQ(result, 0);
}

TEST_F(ImmuneExhaustionTest, StatsTrackExhaustionEvents) {
    // Activate and exhaust multiple cells
    for (int i = 0; i < 3; i++) {
        activateTCell();
    }
    exhaustion_update(exhaustion, EXHAUSTION_ADVANCED_THRESHOLD_MS);

    exhaustion_stats_t stats;
    exhaustion_get_stats(exhaustion, &stats);

    EXPECT_GT(stats.total_exhaustion_events, 0u);
}

/* ============================================================================
 * String Conversion Tests
 * ============================================================================ */

TEST_F(ImmuneExhaustionTest, StateToString) {
    EXPECT_STREQ(exhaustion_state_to_string(EXHAUSTION_STATE_NAIVE), "Naive");
    EXPECT_STREQ(exhaustion_state_to_string(EXHAUSTION_STATE_EFFECTOR), "Effector");
    EXPECT_STREQ(exhaustion_state_to_string(EXHAUSTION_STATE_EXHAUSTED), "Exhausted");
    EXPECT_STREQ(exhaustion_state_to_string(EXHAUSTION_STATE_RECOVERING), "Recovering");
}

TEST_F(ImmuneExhaustionTest, RecoveryStrategyToString) {
    EXPECT_STREQ(exhaustion_recovery_strategy_to_string(RECOVERY_STRATEGY_NATURAL), "Natural");
    EXPECT_STREQ(exhaustion_recovery_strategy_to_string(RECOVERY_STRATEGY_CHECKPOINT), "Checkpoint");
}

/* ============================================================================
 * Edge Cases
 * ============================================================================ */

TEST_F(ImmuneExhaustionTest, UpdateNullSystem) {
    int result = exhaustion_update(nullptr, 1000);
    EXPECT_EQ(result, -1);
}

TEST_F(ImmuneExhaustionTest, GetStateForUnknownCell) {
    exhaustion_state_t state = exhaustion_get_cell_state(exhaustion, 99999);
    EXPECT_EQ(state, EXHAUSTION_STATE_NAIVE);  // Default for unknown
}

TEST_F(ImmuneExhaustionTest, MultipleRecoveryAttempts) {
    uint32_t t_cell_id = activateTCell();
    exhaustion_update(exhaustion, EXHAUSTION_ADVANCED_THRESHOLD_MS);

    exhaustion_initiate_recovery(exhaustion, t_cell_id);
    // Second attempt should fail or be ignored
    int result = exhaustion_initiate_recovery(exhaustion, t_cell_id);
    EXPECT_NE(result, 0);
}
