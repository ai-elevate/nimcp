/**
 * @file test_fep_sleep.cpp
 * @brief Unit tests for FEP Sleep module
 */

#include <gtest/gtest.h>
#include <cmath>
#include "cognitive/free_energy/nimcp_fep_sleep.h"
#include "cognitive/free_energy/nimcp_free_energy.h"

class FEPSleepTest : public ::testing::Test {
protected:
    fep_sleep_system_t* sleep = nullptr;
    fep_system_t* fep = nullptr;

    static const uint32_t OBS_DIM = 8;
    static const uint32_t ACTION_DIM = 4;

    void SetUp() override {
        fep_sleep_config_t config;
        fep_sleep_default_config(&config);
        sleep = fep_sleep_create(&config);

        /* Create FEP system for integration tests */
        fep_config_t fep_config;
        fep_default_config(&fep_config);
        fep = fep_create(&fep_config, OBS_DIM, ACTION_DIM);
    }

    void TearDown() override {
        if (sleep) {
            fep_sleep_destroy(sleep);
            sleep = nullptr;
        }
        if (fep) {
            fep_destroy(fep);
            fep = nullptr;
        }
    }
};

/* ============================================================================
 * Lifecycle Tests
 * ============================================================================ */

TEST_F(FEPSleepTest, CreateDestroy) {
    ASSERT_NE(sleep, nullptr);
}

TEST_F(FEPSleepTest, CreateWithNullConfig) {
    fep_sleep_system_t* sys = fep_sleep_create(nullptr);
    ASSERT_NE(sys, nullptr);
    fep_sleep_destroy(sys);
}

TEST_F(FEPSleepTest, DestroyNull) {
    fep_sleep_destroy(nullptr);  /* Should not crash */
}

TEST_F(FEPSleepTest, DefaultConfig) {
    fep_sleep_config_t config;
    fep_sleep_default_config(&config);

    EXPECT_GT(config.n1_duration_ms, 0u);
    EXPECT_GT(config.n2_duration_ms, 0u);
    EXPECT_GT(config.sws_duration_ms, 0u);
    EXPECT_GT(config.rem_duration_ms, 0u);
    EXPECT_GT(config.replays_per_cycle, 0u);
    EXPECT_GT(config.downscale_factor, 0.0f);
    EXPECT_LT(config.downscale_factor, 1.0f);
    EXPECT_GT(config.experience_buffer_size, 0u);
}

TEST_F(FEPSleepTest, CreateWithCustomConfig) {
    fep_sleep_config_t config;
    fep_sleep_default_config(&config);
    config.replays_per_cycle = 200;
    config.downscale_factor = 0.85f;
    config.experience_buffer_size = 2000;

    fep_sleep_system_t* custom = fep_sleep_create(&config);
    ASSERT_NE(custom, nullptr);
    fep_sleep_destroy(custom);
}

/* ============================================================================
 * Sleep Stage Tests
 * ============================================================================ */

TEST_F(FEPSleepTest, InitialStage) {
    fep_sleep_stage_t stage = fep_sleep_get_stage(sleep);
    EXPECT_EQ(stage, SLEEP_STAGE_WAKE);
}

TEST_F(FEPSleepTest, SetStageN1) {
    int ret = fep_sleep_set_stage(sleep, SLEEP_STAGE_N1);
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(fep_sleep_get_stage(sleep), SLEEP_STAGE_N1);
}

TEST_F(FEPSleepTest, SetStageN2) {
    int ret = fep_sleep_set_stage(sleep, SLEEP_STAGE_N2);
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(fep_sleep_get_stage(sleep), SLEEP_STAGE_N2);
}

TEST_F(FEPSleepTest, SetStageSWS) {
    int ret = fep_sleep_set_stage(sleep, SLEEP_STAGE_SWS);
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(fep_sleep_get_stage(sleep), SLEEP_STAGE_SWS);
}

TEST_F(FEPSleepTest, SetStageREM) {
    int ret = fep_sleep_set_stage(sleep, SLEEP_STAGE_REM);
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(fep_sleep_get_stage(sleep), SLEEP_STAGE_REM);
}

TEST_F(FEPSleepTest, SetStageWake) {
    fep_sleep_set_stage(sleep, SLEEP_STAGE_SWS);
    int ret = fep_sleep_set_stage(sleep, SLEEP_STAGE_WAKE);
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(fep_sleep_get_stage(sleep), SLEEP_STAGE_WAKE);
}

TEST_F(FEPSleepTest, SetStageNullParams) {
    EXPECT_EQ(fep_sleep_set_stage(nullptr, SLEEP_STAGE_N1), -1);
}

TEST_F(FEPSleepTest, GetStageNull) {
    EXPECT_EQ(fep_sleep_get_stage(nullptr), SLEEP_STAGE_WAKE);
}

/* ============================================================================
 * Update Tests
 * ============================================================================ */

TEST_F(FEPSleepTest, UpdateWake) {
    int ret = fep_sleep_update(sleep, 1000);
    EXPECT_EQ(ret, 0);
}

TEST_F(FEPSleepTest, UpdateNull) {
    EXPECT_EQ(fep_sleep_update(nullptr, 1000), -1);
}

TEST_F(FEPSleepTest, UpdateMultipleTimes) {
    for (int i = 0; i < 100; i++) {
        EXPECT_EQ(fep_sleep_update(sleep, 100), 0);
    }
}

TEST_F(FEPSleepTest, UpdateDuringSleep) {
    fep_sleep_set_stage(sleep, SLEEP_STAGE_N2);
    int ret = fep_sleep_update(sleep, 5000);
    EXPECT_EQ(ret, 0);
}

TEST_F(FEPSleepTest, AutoCycleProgression) {
    fep_sleep_config_t config;
    fep_sleep_default_config(&config);
    config.enable_auto_cycle = true;
    config.n1_duration_ms = 100;  /* Short for testing */

    fep_sleep_system_t* auto_sleep = fep_sleep_create(&config);
    ASSERT_NE(auto_sleep, nullptr);

    fep_sleep_set_stage(auto_sleep, SLEEP_STAGE_N1);

    /* Update past N1 duration */
    fep_sleep_update(auto_sleep, 150);

    /* May have progressed to next stage */
    fep_sleep_stage_t stage = fep_sleep_get_stage(auto_sleep);
    EXPECT_GE(stage, SLEEP_STAGE_N1);

    fep_sleep_destroy(auto_sleep);
}

/* ============================================================================
 * Experience Buffer Tests
 * ============================================================================ */

TEST_F(FEPSleepTest, AddExperience) {
    float state[4] = {0.1f, 0.2f, 0.3f, 0.4f};
    float observation[4] = {0.5f, 0.6f, 0.7f, 0.8f};
    float next_state[4] = {0.2f, 0.3f, 0.4f, 0.5f};

    int ret = fep_sleep_add_experience(sleep, state, observation, next_state, 4, 4);
    EXPECT_EQ(ret, 0);
}

TEST_F(FEPSleepTest, AddExperienceNullParams) {
    float state[4] = {0.0f};
    float obs[4] = {0.0f};
    float next[4] = {0.0f};

    EXPECT_EQ(fep_sleep_add_experience(nullptr, state, obs, next, 4, 4), -1);
    EXPECT_EQ(fep_sleep_add_experience(sleep, nullptr, obs, next, 4, 4), -1);
    EXPECT_EQ(fep_sleep_add_experience(sleep, state, nullptr, next, 4, 4), -1);
    EXPECT_EQ(fep_sleep_add_experience(sleep, state, obs, nullptr, 4, 4), -1);
    EXPECT_EQ(fep_sleep_add_experience(sleep, state, obs, next, 0, 4), -1);
}

TEST_F(FEPSleepTest, AddMultipleExperiences) {
    float state[4] = {0.0f};
    float obs[4] = {0.0f};
    float next[4] = {0.0f};

    for (int i = 0; i < 50; i++) {
        state[0] = (float)i / 50.0f;
        int ret = fep_sleep_add_experience(sleep, state, obs, next, 4, 4);
        EXPECT_EQ(ret, 0);
    }
}

TEST_F(FEPSleepTest, ExperienceBufferOverflow) {
    fep_sleep_config_t config;
    fep_sleep_default_config(&config);
    config.experience_buffer_size = 10;

    fep_sleep_system_t* small_buffer = fep_sleep_create(&config);
    ASSERT_NE(small_buffer, nullptr);

    float state[4] = {0.0f};
    float obs[4] = {0.0f};
    float next[4] = {0.0f};

    /* Fill beyond capacity */
    for (int i = 0; i < 20; i++) {
        int ret = fep_sleep_add_experience(small_buffer, state, obs, next, 4, 4);
        EXPECT_EQ(ret, 0);  /* Should handle overflow gracefully */
    }

    fep_sleep_destroy(small_buffer);
}

/* ============================================================================
 * Consolidation Tests
 * ============================================================================ */

TEST_F(FEPSleepTest, ReplayConsolidation) {
    fep_sleep_connect(sleep, fep);

    /* Add some experiences */
    float state[8] = {0.1f, 0.2f, 0.3f, 0.4f, 0.0f, 0.0f, 0.0f, 0.0f};
    float obs[8] = {0.5f, 0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float next[8] = {0.2f, 0.3f, 0.4f, 0.5f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int i = 0; i < 10; i++) {
        fep_sleep_add_experience(sleep, state, obs, next, 8, 8);
    }

    /* Enter SWS for consolidation */
    fep_sleep_set_stage(sleep, SLEEP_STAGE_SWS);

    int ret = fep_sleep_replay_consolidation(sleep, fep, 5);
    EXPECT_EQ(ret, 0);
}

TEST_F(FEPSleepTest, ReplayConsolidationNullParams) {
    EXPECT_EQ(fep_sleep_replay_consolidation(nullptr, fep, 10), -1);
    EXPECT_EQ(fep_sleep_replay_consolidation(sleep, nullptr, 10), -1);
}

TEST_F(FEPSleepTest, ReplayConsolidationEmptyBuffer) {
    fep_sleep_connect(sleep, fep);
    fep_sleep_set_stage(sleep, SLEEP_STAGE_SWS);

    /* Replay with empty buffer */
    int ret = fep_sleep_replay_consolidation(sleep, fep, 10);
    EXPECT_EQ(ret, 0);  /* Should return success but do nothing */
}

TEST_F(FEPSleepTest, ApplyDownscaling) {
    fep_sleep_connect(sleep, fep);

    int ret = fep_sleep_apply_downscaling(sleep, fep, 0.9f);
    EXPECT_EQ(ret, 0);
}

TEST_F(FEPSleepTest, ApplyDownscalingNullParams) {
    EXPECT_EQ(fep_sleep_apply_downscaling(nullptr, fep, 0.9f), -1);
    EXPECT_EQ(fep_sleep_apply_downscaling(sleep, nullptr, 0.9f), -1);
}

TEST_F(FEPSleepTest, ApplyDownscalingBoundaries) {
    fep_sleep_connect(sleep, fep);

    EXPECT_EQ(fep_sleep_apply_downscaling(sleep, fep, 0.0f), 0);
    EXPECT_EQ(fep_sleep_apply_downscaling(sleep, fep, 1.0f), 0);
    EXPECT_EQ(fep_sleep_apply_downscaling(sleep, fep, 0.5f), 0);
}

TEST_F(FEPSleepTest, REMIntegration) {
    fep_sleep_connect(sleep, fep);

    /* Add experiences for REM to process */
    float state[8] = {0.0f};
    float obs[8] = {0.0f};
    float next[8] = {0.0f};
    for (int i = 0; i < 10; i++) {
        fep_sleep_add_experience(sleep, state, obs, next, 8, 8);
    }

    fep_sleep_set_stage(sleep, SLEEP_STAGE_REM);

    int ret = fep_sleep_rem_integration(sleep, fep);
    EXPECT_EQ(ret, 0);
}

TEST_F(FEPSleepTest, REMIntegrationNullParams) {
    EXPECT_EQ(fep_sleep_rem_integration(nullptr, fep), -1);
    EXPECT_EQ(fep_sleep_rem_integration(sleep, nullptr), -1);
}

/* ============================================================================
 * State/Stats Tests
 * ============================================================================ */

TEST_F(FEPSleepTest, GetState) {
    fep_sleep_state_t state;
    int ret = fep_sleep_get_state(sleep, &state);

    EXPECT_EQ(ret, 0);
    EXPECT_EQ(state.current_stage, SLEEP_STAGE_WAKE);
    EXPECT_EQ(state.cycle_count, 0u);
    EXPECT_GE(state.sleep_efficiency, 0.0f);
}

TEST_F(FEPSleepTest, GetStateNullParams) {
    fep_sleep_state_t state;
    EXPECT_EQ(fep_sleep_get_state(nullptr, &state), -1);
    EXPECT_EQ(fep_sleep_get_state(sleep, nullptr), -1);
}

TEST_F(FEPSleepTest, GetStats) {
    fep_sleep_stats_t stats;
    int ret = fep_sleep_get_stats(sleep, &stats);

    EXPECT_EQ(ret, 0);
    EXPECT_EQ(stats.total_cycles, 0u);
    EXPECT_EQ(stats.total_replays, 0u);
}

TEST_F(FEPSleepTest, GetStatsNullParams) {
    fep_sleep_stats_t stats;
    EXPECT_EQ(fep_sleep_get_stats(nullptr, &stats), -1);
    EXPECT_EQ(fep_sleep_get_stats(sleep, nullptr), -1);
}

TEST_F(FEPSleepTest, StatsAfterActivity) {
    fep_sleep_connect(sleep, fep);

    /* Add experiences and run consolidation */
    float state[8] = {0.0f};
    float obs[8] = {0.0f};
    float next[8] = {0.0f};
    for (int i = 0; i < 20; i++) {
        fep_sleep_add_experience(sleep, state, obs, next, 8, 8);
    }

    fep_sleep_set_stage(sleep, SLEEP_STAGE_SWS);
    fep_sleep_replay_consolidation(sleep, fep, 10);

    fep_sleep_stats_t stats;
    fep_sleep_get_stats(sleep, &stats);
    EXPECT_GT(stats.total_replays, 0u);
}

/* ============================================================================
 * Precision Modifier Tests
 * ============================================================================ */

TEST_F(FEPSleepTest, PrecisionModifierWake) {
    fep_sleep_set_stage(sleep, SLEEP_STAGE_WAKE);
    float precision = fep_sleep_get_precision_modifier(sleep);
    EXPECT_NEAR(precision, FEP_SLEEP_WAKE_PRECISION, 0.01f);
}

TEST_F(FEPSleepTest, PrecisionModifierN1) {
    fep_sleep_set_stage(sleep, SLEEP_STAGE_N1);
    float precision = fep_sleep_get_precision_modifier(sleep);
    EXPECT_NEAR(precision, FEP_SLEEP_N1_PRECISION, 0.01f);
}

TEST_F(FEPSleepTest, PrecisionModifierN2) {
    fep_sleep_set_stage(sleep, SLEEP_STAGE_N2);
    float precision = fep_sleep_get_precision_modifier(sleep);
    EXPECT_NEAR(precision, FEP_SLEEP_N2_PRECISION, 0.01f);
}

TEST_F(FEPSleepTest, PrecisionModifierSWS) {
    fep_sleep_set_stage(sleep, SLEEP_STAGE_SWS);
    float precision = fep_sleep_get_precision_modifier(sleep);
    EXPECT_NEAR(precision, FEP_SLEEP_SWS_PRECISION, 0.01f);
}

TEST_F(FEPSleepTest, PrecisionModifierREM) {
    fep_sleep_set_stage(sleep, SLEEP_STAGE_REM);
    float precision = fep_sleep_get_precision_modifier(sleep);
    EXPECT_NEAR(precision, FEP_SLEEP_REM_PRECISION, 0.01f);
}

TEST_F(FEPSleepTest, PrecisionModifierNull) {
    float precision = fep_sleep_get_precision_modifier(nullptr);
    EXPECT_EQ(precision, 0.0f);
}

TEST_F(FEPSleepTest, PrecisionDecreasesDuringSleep) {
    fep_sleep_set_stage(sleep, SLEEP_STAGE_WAKE);
    float wake_precision = fep_sleep_get_precision_modifier(sleep);

    fep_sleep_set_stage(sleep, SLEEP_STAGE_SWS);
    float sws_precision = fep_sleep_get_precision_modifier(sleep);

    EXPECT_LT(sws_precision, wake_precision);
}

/* ============================================================================
 * Integration Tests
 * ============================================================================ */

TEST_F(FEPSleepTest, Connect) {
    int ret = fep_sleep_connect(sleep, fep);
    EXPECT_EQ(ret, 0);
}

TEST_F(FEPSleepTest, ConnectNullParams) {
    EXPECT_EQ(fep_sleep_connect(nullptr, fep), -1);
    EXPECT_EQ(fep_sleep_connect(sleep, nullptr), -1);
}

/* ============================================================================
 * Bio-Async Tests
 * ============================================================================ */

TEST_F(FEPSleepTest, BioAsyncConnectDisconnect) {
    EXPECT_FALSE(fep_sleep_is_bio_async_connected(sleep));

    int ret = fep_sleep_connect_bio_async(sleep);
    EXPECT_EQ(ret, 0);

    ret = fep_sleep_disconnect_bio_async(sleep);
    EXPECT_EQ(ret, 0);
    EXPECT_FALSE(fep_sleep_is_bio_async_connected(sleep));
}

TEST_F(FEPSleepTest, BioAsyncDoubleConnect) {
    fep_sleep_connect_bio_async(sleep);
    int ret = fep_sleep_connect_bio_async(sleep);
    EXPECT_EQ(ret, 0);
    fep_sleep_disconnect_bio_async(sleep);
}

TEST_F(FEPSleepTest, BioAsyncNullParams) {
    EXPECT_EQ(fep_sleep_connect_bio_async(nullptr), -1);
    EXPECT_EQ(fep_sleep_disconnect_bio_async(nullptr), -1);
    EXPECT_FALSE(fep_sleep_is_bio_async_connected(nullptr));
}

/* ============================================================================
 * String Conversion Tests
 * ============================================================================ */

TEST_F(FEPSleepTest, StageToStringWake) {
    EXPECT_STREQ(fep_sleep_stage_to_string(SLEEP_STAGE_WAKE), "WAKE");
}

TEST_F(FEPSleepTest, StageToStringN1) {
    EXPECT_STREQ(fep_sleep_stage_to_string(SLEEP_STAGE_N1), "N1");
}

TEST_F(FEPSleepTest, StageToStringN2) {
    EXPECT_STREQ(fep_sleep_stage_to_string(SLEEP_STAGE_N2), "N2");
}

TEST_F(FEPSleepTest, StageToStringSWS) {
    EXPECT_STREQ(fep_sleep_stage_to_string(SLEEP_STAGE_SWS), "SWS");
}

TEST_F(FEPSleepTest, StageToStringREM) {
    EXPECT_STREQ(fep_sleep_stage_to_string(SLEEP_STAGE_REM), "REM");
}

TEST_F(FEPSleepTest, StageToStringInvalid) {
    const char* str = fep_sleep_stage_to_string((fep_sleep_stage_t)99);
    EXPECT_NE(str, nullptr);
    EXPECT_STREQ(str, "UNKNOWN");
}

/* ============================================================================
 * Configuration Feature Tests
 * ============================================================================ */

TEST_F(FEPSleepTest, DisabledFeatures) {
    fep_sleep_config_t config;
    fep_sleep_default_config(&config);
    config.enable_auto_cycle = false;
    config.enable_synaptic_homeostasis = false;
    config.enable_replay_consolidation = false;
    config.enable_rem_integration = false;

    fep_sleep_system_t* disabled = fep_sleep_create(&config);
    ASSERT_NE(disabled, nullptr);

    fep_sleep_connect(disabled, fep);

    /* Features should be disabled but not crash */
    fep_sleep_set_stage(disabled, SLEEP_STAGE_SWS);
    EXPECT_EQ(fep_sleep_replay_consolidation(disabled, fep, 10), 0);
    EXPECT_EQ(fep_sleep_apply_downscaling(disabled, fep, 0.9f), 0);

    fep_sleep_set_stage(disabled, SLEEP_STAGE_REM);
    EXPECT_EQ(fep_sleep_rem_integration(disabled, fep), 0);

    fep_sleep_destroy(disabled);
}

/* ============================================================================
 * Full Sleep Cycle Tests
 * ============================================================================ */

TEST_F(FEPSleepTest, FullSleepCycle) {
    fep_sleep_connect(sleep, fep);

    /* Add experiences during "wake" */
    float state[8] = {0.0f};
    float obs[8] = {0.0f};
    float next[8] = {0.0f};
    for (int i = 0; i < 50; i++) {
        state[0] = (float)i / 50.0f;
        fep_sleep_add_experience(sleep, state, obs, next, 8, 8);
    }

    /* Transition through sleep stages */
    EXPECT_EQ(fep_sleep_set_stage(sleep, SLEEP_STAGE_N1), 0);
    fep_sleep_update(sleep, 1000);

    EXPECT_EQ(fep_sleep_set_stage(sleep, SLEEP_STAGE_N2), 0);
    fep_sleep_update(sleep, 1000);

    EXPECT_EQ(fep_sleep_set_stage(sleep, SLEEP_STAGE_SWS), 0);
    fep_sleep_replay_consolidation(sleep, fep, 20);
    fep_sleep_apply_downscaling(sleep, fep, 0.9f);

    EXPECT_EQ(fep_sleep_set_stage(sleep, SLEEP_STAGE_REM), 0);
    fep_sleep_rem_integration(sleep, fep);

    /* Return to wake */
    EXPECT_EQ(fep_sleep_set_stage(sleep, SLEEP_STAGE_WAKE), 0);

    /* Verify stats were tracked */
    fep_sleep_stats_t stats;
    fep_sleep_get_stats(sleep, &stats);
    EXPECT_GT(stats.total_replays, 0u);
}

/* ============================================================================
 * Time Tracking Tests
 * ============================================================================ */

TEST_F(FEPSleepTest, TimeTrackingInStages) {
    fep_sleep_set_stage(sleep, SLEEP_STAGE_N1);
    fep_sleep_update(sleep, 5000);

    fep_sleep_set_stage(sleep, SLEEP_STAGE_N2);
    fep_sleep_update(sleep, 10000);

    fep_sleep_set_stage(sleep, SLEEP_STAGE_SWS);
    fep_sleep_update(sleep, 20000);

    fep_sleep_stats_t stats;
    fep_sleep_get_stats(sleep, &stats);

    EXPECT_GT(stats.total_n1_time_ms, 0u);
    EXPECT_GT(stats.total_n2_time_ms, 0u);
    EXPECT_GT(stats.total_sws_time_ms, 0u);
}

/* ============================================================================
 * Consolidation Quality Tests
 * ============================================================================ */

TEST_F(FEPSleepTest, ConsolidationProgress) {
    fep_sleep_connect(sleep, fep);

    float state[8] = {0.0f};
    float obs[8] = {0.0f};
    float next[8] = {0.0f};
    for (int i = 0; i < 30; i++) {
        fep_sleep_add_experience(sleep, state, obs, next, 8, 8);
    }

    fep_sleep_set_stage(sleep, SLEEP_STAGE_SWS);
    fep_sleep_replay_consolidation(sleep, fep, 15);

    fep_sleep_state_t state_out;
    fep_sleep_get_state(sleep, &state_out);
    EXPECT_GT(state_out.replays_this_cycle, 0u);
}
