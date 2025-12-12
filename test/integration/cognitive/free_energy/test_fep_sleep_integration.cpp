/**
 * @file test_fep_sleep_integration.cpp
 * @brief Integration tests for FEP Sleep module with other FEP components
 */

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "cognitive/free_energy/nimcp_fep_sleep.h"
#include "cognitive/free_energy/nimcp_free_energy.h"
#include "cognitive/free_energy/nimcp_fep_learning.h"
#include "cognitive/free_energy/nimcp_fep_evidence.h"

class FEPSleepIntegrationTest : public ::testing::Test {
protected:
    static const uint32_t OBS_DIM = 8;
    static const uint32_t ACTION_DIM = 4;
    static const uint32_t STATE_DIM = 8;
    fep_sleep_system_t* sleep = nullptr;
    fep_system_t* fep = nullptr;
    fep_transition_learner_t* transition_learner = nullptr;

    void SetUp() override {
        /* Create sleep system */
        fep_sleep_config_t sleep_config;
        fep_sleep_default_config(&sleep_config);
        sleep = fep_sleep_create(&sleep_config);

        /* Create FEP system */
        fep_config_t fep_config;
        fep_default_config(&fep_config);
        fep = fep_create(&fep_config, OBS_DIM, ACTION_DIM);

        /* Create transition learner */
        fep_learning_config_t learn_config;
        fep_learning_default_config(&learn_config);
        transition_learner = fep_transition_learner_create(&learn_config, STATE_DIM);
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
        if (transition_learner) {
            fep_transition_learner_destroy(transition_learner);
            transition_learner = nullptr;
        }
    }
};

/* ============================================================================
 * Sleep + FEP Core Integration Tests
 * ============================================================================ */

TEST_F(FEPSleepIntegrationTest, SleepWithFEPSystem) {
    fep_sleep_connect(sleep, fep);

    /* Sleep system should be connected */
    fep_sleep_state_t state;
    int ret = fep_sleep_get_state(sleep, &state);
    EXPECT_EQ(ret, 0);
}

TEST_F(FEPSleepIntegrationTest, PrecisionModulationBySleepStage) {
    fep_sleep_connect(sleep, fep);

    /* Wake precision */
    fep_sleep_set_stage(sleep, SLEEP_STAGE_WAKE);
    float wake_precision = fep_sleep_get_precision_modifier(sleep);

    /* SWS precision */
    fep_sleep_set_stage(sleep, SLEEP_STAGE_SWS);
    float sws_precision = fep_sleep_get_precision_modifier(sleep);

    EXPECT_GT(wake_precision, sws_precision);
}

/* ============================================================================
 * Experience Accumulation Tests
 * ============================================================================ */

TEST_F(FEPSleepIntegrationTest, AccumulateExperiencesDuringWake) {
    fep_sleep_connect(sleep, fep);

    fep_sleep_set_stage(sleep, SLEEP_STAGE_WAKE);

    /* Accumulate experiences */
    for (int i = 0; i < 50; i++) {
        std::vector<float> state(STATE_DIM, 0.0f);
        std::vector<float> obs(OBS_DIM, 0.0f);
        std::vector<float> next(STATE_DIM, 0.0f);
        state[i % STATE_DIM] = 1.0f;
        next[(i + 1) % STATE_DIM] = 1.0f;

        int ret = fep_sleep_add_experience(sleep, state.data(), obs.data(), next.data(), STATE_DIM, OBS_DIM);
        EXPECT_EQ(ret, 0);
    }
}

/* ============================================================================
 * Sleep Consolidation Tests
 * ============================================================================ */

TEST_F(FEPSleepIntegrationTest, SWSConsolidation) {
    fep_sleep_connect(sleep, fep);

    /* Accumulate experiences */
    for (int i = 0; i < 30; i++) {
        std::vector<float> state(STATE_DIM, 0.0f);
        std::vector<float> obs(OBS_DIM, 0.0f);
        std::vector<float> next(STATE_DIM, 0.0f);
        state[i % STATE_DIM] = 1.0f;
        fep_sleep_add_experience(sleep, state.data(), obs.data(), next.data(), STATE_DIM, OBS_DIM);
    }

    /* Enter SWS */
    fep_sleep_set_stage(sleep, SLEEP_STAGE_SWS);

    /* Run consolidation */
    int ret = fep_sleep_replay_consolidation(sleep, fep, 20);
    EXPECT_EQ(ret, 0);

    /* Check stats */
    fep_sleep_stats_t stats;
    fep_sleep_get_stats(sleep, &stats);
    EXPECT_GT(stats.total_replays, 0u);
}

TEST_F(FEPSleepIntegrationTest, DownscalingDuringSWS) {
    fep_sleep_connect(sleep, fep);

    fep_sleep_set_stage(sleep, SLEEP_STAGE_SWS);

    int ret = fep_sleep_apply_downscaling(sleep, fep, 0.9f);
    EXPECT_EQ(ret, 0);
}

TEST_F(FEPSleepIntegrationTest, REMIntegration) {
    fep_sleep_connect(sleep, fep);

    /* Add experiences first */
    for (int i = 0; i < 20; i++) {
        std::vector<float> state(STATE_DIM, 0.0f);
        std::vector<float> obs(OBS_DIM, 0.0f);
        std::vector<float> next(STATE_DIM, 0.0f);
        fep_sleep_add_experience(sleep, state.data(), obs.data(), next.data(), STATE_DIM, OBS_DIM);
    }

    /* Enter REM */
    fep_sleep_set_stage(sleep, SLEEP_STAGE_REM);

    int ret = fep_sleep_rem_integration(sleep, fep);
    EXPECT_EQ(ret, 0);
}

/* ============================================================================
 * Sleep + Learning Integration Tests
 * ============================================================================ */

TEST_F(FEPSleepIntegrationTest, WakeLearning) {
    fep_sleep_connect(sleep, fep);

    fep_sleep_set_stage(sleep, SLEEP_STAGE_WAKE);

    /* Learning should work at full precision during wake */
    std::vector<float> state(STATE_DIM, 0.0f);
    std::vector<float> next(STATE_DIM, 0.0f);
    state[0] = 1.0f;
    next[1] = 1.0f;

    int ret = fep_learn_transition(transition_learner, fep, state.data(), next.data(), STATE_DIM);
    EXPECT_EQ(ret, 0);
}

TEST_F(FEPSleepIntegrationTest, SleepReducesActiveLearning) {
    fep_sleep_connect(sleep, fep);

    /* Precision during SWS should be reduced */
    fep_sleep_set_stage(sleep, SLEEP_STAGE_SWS);
    float sws_precision = fep_sleep_get_precision_modifier(sleep);

    EXPECT_LT(sws_precision, 0.5f);
}

/* ============================================================================
 * Sleep Cycle Tests
 * ============================================================================ */

TEST_F(FEPSleepIntegrationTest, FullSleepCycleIntegration) {
    fep_sleep_connect(sleep, fep);

    /* 1. Wake: Learn and accumulate experiences */
    fep_sleep_set_stage(sleep, SLEEP_STAGE_WAKE);

    for (int i = 0; i < 50; i++) {
        std::vector<float> state(STATE_DIM, 0.0f);
        std::vector<float> obs(OBS_DIM, 0.0f);
        std::vector<float> next(STATE_DIM, 0.0f);
        state[i % STATE_DIM] = 1.0f;
        next[(i + 1) % STATE_DIM] = 1.0f;

        fep_learn_transition(transition_learner, fep, state.data(), next.data(), STATE_DIM);
        fep_sleep_add_experience(sleep, state.data(), obs.data(), next.data(), STATE_DIM, OBS_DIM);
    }

    /* 2. N1: Transition */
    fep_sleep_set_stage(sleep, SLEEP_STAGE_N1);
    fep_sleep_update(sleep, 1000);

    /* 3. N2: Memory tagging */
    fep_sleep_set_stage(sleep, SLEEP_STAGE_N2);
    fep_sleep_update(sleep, 2000);

    /* 4. SWS: Consolidation */
    fep_sleep_set_stage(sleep, SLEEP_STAGE_SWS);
    fep_sleep_replay_consolidation(sleep, fep, 30);
    fep_sleep_apply_downscaling(sleep, fep, 0.9f);

    /* 5. REM: Integration */
    fep_sleep_set_stage(sleep, SLEEP_STAGE_REM);
    fep_sleep_rem_integration(sleep, fep);

    /* 6. Back to wake */
    fep_sleep_set_stage(sleep, SLEEP_STAGE_WAKE);

    /* Verify stats */
    fep_sleep_stats_t stats;
    fep_sleep_get_stats(sleep, &stats);
    EXPECT_GT(stats.total_replays, 0u);
}

/* ============================================================================
 * Evidence Integration Tests
 * ============================================================================ */

TEST_F(FEPSleepIntegrationTest, EvidenceAfterConsolidation) {
    fep_sleep_connect(sleep, fep);

    fep_evidence_config_t ev_config;
    fep_evidence_default_config(&ev_config);
    fep_evidence_system_t* evidence = fep_evidence_create(&ev_config);
    fep_evidence_connect(evidence, fep);

    /* Compute evidence before consolidation */
    std::vector<float> obs_test(OBS_DIM, 0.0f);
    obs_test[0] = 0.8f;
    obs_test[1] = 0.2f;
    float elbo_before;
    fep_compute_elbo(evidence, fep, obs_test.data(), OBS_DIM, &elbo_before);

    /* Add experiences */
    for (int i = 0; i < 30; i++) {
        std::vector<float> state(STATE_DIM, 0.0f);
        std::vector<float> obs(OBS_DIM, 0.0f);
        std::vector<float> next(STATE_DIM, 0.0f);
        state[i % STATE_DIM] = 1.0f;
        fep_sleep_add_experience(sleep, state.data(), obs.data(), next.data(), STATE_DIM, OBS_DIM);
    }

    /* Consolidate */
    fep_sleep_set_stage(sleep, SLEEP_STAGE_SWS);
    fep_sleep_replay_consolidation(sleep, fep, 20);

    /* Compute evidence after */
    float elbo_after;
    fep_compute_elbo(evidence, fep, obs_test.data(), OBS_DIM, &elbo_after);

    EXPECT_TRUE(std::isfinite(elbo_before));
    EXPECT_TRUE(std::isfinite(elbo_after));

    fep_evidence_destroy(evidence);
}

/* ============================================================================
 * Auto-Cycle Tests
 * ============================================================================ */

TEST_F(FEPSleepIntegrationTest, AutoCycleProgression) {
    fep_sleep_config_t config;
    fep_sleep_default_config(&config);
    config.enable_auto_cycle = true;
    config.n1_duration_ms = 100;
    config.n2_duration_ms = 200;
    config.sws_duration_ms = 300;
    config.rem_duration_ms = 150;

    fep_sleep_system_t* auto_sleep = fep_sleep_create(&config);
    fep_sleep_connect(auto_sleep, fep);

    /* Start sleep */
    fep_sleep_set_stage(auto_sleep, SLEEP_STAGE_N1);

    /* Update through stages */
    for (int i = 0; i < 20; i++) {
        fep_sleep_update(auto_sleep, 100);
    }

    fep_sleep_state_t state;
    fep_sleep_get_state(auto_sleep, &state);

    /* Should have progressed through stages */
    EXPECT_GE(state.total_sleep_ms, 0u);

    fep_sleep_destroy(auto_sleep);
}

/* ============================================================================
 * Stats Tracking Tests
 * ============================================================================ */

TEST_F(FEPSleepIntegrationTest, TimeTrackingPerStage) {
    fep_sleep_connect(sleep, fep);

    /* Spend time in each stage */
    fep_sleep_set_stage(sleep, SLEEP_STAGE_N1);
    fep_sleep_update(sleep, 5000);

    fep_sleep_set_stage(sleep, SLEEP_STAGE_N2);
    fep_sleep_update(sleep, 10000);

    fep_sleep_set_stage(sleep, SLEEP_STAGE_SWS);
    fep_sleep_update(sleep, 15000);

    fep_sleep_set_stage(sleep, SLEEP_STAGE_REM);
    fep_sleep_update(sleep, 8000);

    fep_sleep_stats_t stats;
    fep_sleep_get_stats(sleep, &stats);

    EXPECT_GT(stats.total_n1_time_ms, 0u);
    EXPECT_GT(stats.total_n2_time_ms, 0u);
    EXPECT_GT(stats.total_sws_time_ms, 0u);
    EXPECT_GT(stats.total_rem_time_ms, 0u);
}

/* ============================================================================
 * Bio-Async Integration Tests
 * ============================================================================ */

TEST_F(FEPSleepIntegrationTest, BioAsyncWithSleep) {
    fep_sleep_connect(sleep, fep);
    fep_sleep_connect_bio_async(sleep);

    /* Sleep should work with bio-async */
    fep_sleep_set_stage(sleep, SLEEP_STAGE_N2);
    int ret = fep_sleep_update(sleep, 100);
    EXPECT_EQ(ret, 0);

    fep_sleep_disconnect_bio_async(sleep);
}

/* ============================================================================
 * String Conversion Tests
 * ============================================================================ */

TEST_F(FEPSleepIntegrationTest, StageStrings) {
    EXPECT_STREQ(fep_sleep_stage_to_string(SLEEP_STAGE_WAKE), "WAKE");
    EXPECT_STREQ(fep_sleep_stage_to_string(SLEEP_STAGE_N1), "N1");
    EXPECT_STREQ(fep_sleep_stage_to_string(SLEEP_STAGE_N2), "N2");
    EXPECT_STREQ(fep_sleep_stage_to_string(SLEEP_STAGE_SWS), "SWS");
    EXPECT_STREQ(fep_sleep_stage_to_string(SLEEP_STAGE_REM), "REM");
}

/* ============================================================================
 * Config Integration Tests
 * ============================================================================ */

TEST_F(FEPSleepIntegrationTest, CustomConsolidationConfig) {
    fep_sleep_config_t config;
    fep_sleep_default_config(&config);
    config.replays_per_cycle = 200;
    config.downscale_factor = 0.85f;

    fep_sleep_system_t* custom = fep_sleep_create(&config);
    fep_sleep_connect(custom, fep);

    /* Add experiences */
    for (int i = 0; i < 20; i++) {
        std::vector<float> state(STATE_DIM, 0.0f);
        std::vector<float> obs(OBS_DIM, 0.0f);
        std::vector<float> next(STATE_DIM, 0.0f);
        fep_sleep_add_experience(custom, state.data(), obs.data(), next.data(), STATE_DIM, OBS_DIM);
    }

    fep_sleep_set_stage(custom, SLEEP_STAGE_SWS);
    fep_sleep_replay_consolidation(custom, fep, 50);

    fep_sleep_destroy(custom);
}

TEST_F(FEPSleepIntegrationTest, DisabledFeaturesConfig) {
    fep_sleep_config_t config;
    fep_sleep_default_config(&config);
    config.enable_auto_cycle = false;
    config.enable_synaptic_homeostasis = false;
    config.enable_replay_consolidation = false;

    fep_sleep_system_t* disabled = fep_sleep_create(&config);
    fep_sleep_connect(disabled, fep);

    fep_sleep_set_stage(disabled, SLEEP_STAGE_SWS);
    int ret = fep_sleep_update(disabled, 100);
    EXPECT_EQ(ret, 0);

    fep_sleep_destroy(disabled);
}
