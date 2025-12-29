/**
 * @file test_dragonfly_fep_bridge.cpp
 * @brief Unit tests for Dragonfly-to-FEP Integration Bridge
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstring>

extern "C" {
#include "dragonfly/nimcp_dragonfly_fep_bridge.h"
}

//=============================================================================
// Test Fixtures
//=============================================================================

class DragonflyFEPBridgeTest : public ::testing::Test {
protected:
    dragonfly_fep_bridge_t* bridge = nullptr;
    dragonfly_fep_config_t config;

    void SetUp() override {
        ASSERT_EQ(0, dragonfly_fep_bridge_default_config(&config));
    }

    void TearDown() override {
        if (bridge) {
            dragonfly_fep_bridge_destroy(bridge);
            bridge = nullptr;
        }
    }

    void CreateBridge() {
        bridge = dragonfly_fep_bridge_create(nullptr, nullptr, &config);
        ASSERT_NE(nullptr, bridge);
    }
};

//=============================================================================
// Configuration Tests
//=============================================================================

TEST_F(DragonflyFEPBridgeTest, DefaultConfigValid) {
    dragonfly_fep_config_t cfg;
    EXPECT_EQ(0, dragonfly_fep_bridge_default_config(&cfg));

    EXPECT_EQ(FEP_MODEL_CONSTANT_VELOCITY, cfg.default_model);
    EXPECT_TRUE(cfg.auto_model_selection);
    EXPECT_EQ(FEP_PRECISION_ADAPTIVE, cfg.precision_mode);
    EXPECT_GT(cfg.sensory_precision, 0.0f);
    EXPECT_GT(cfg.learning_rate, 0.0f);
    EXPECT_LE(cfg.learning_rate, 1.0f);
    EXPECT_GT(cfg.inference_steps, 0u);
}

TEST_F(DragonflyFEPBridgeTest, DefaultConfigNullReturnsError) {
    EXPECT_EQ(-1, dragonfly_fep_bridge_default_config(nullptr));
}

TEST_F(DragonflyFEPBridgeTest, ValidateConfigSuccess) {
    EXPECT_EQ(0, dragonfly_fep_bridge_validate_config(&config));
}

TEST_F(DragonflyFEPBridgeTest, ValidateConfigNullReturnsError) {
    EXPECT_EQ(-1, dragonfly_fep_bridge_validate_config(nullptr));
}

TEST_F(DragonflyFEPBridgeTest, ValidateConfigInvalidModel) {
    config.default_model = (dragonfly_fep_model_t)99;
    EXPECT_EQ(-1, dragonfly_fep_bridge_validate_config(&config));
}

TEST_F(DragonflyFEPBridgeTest, ValidateConfigInvalidPrecisionMode) {
    config.precision_mode = (dragonfly_fep_precision_mode_t)99;
    EXPECT_EQ(-1, dragonfly_fep_bridge_validate_config(&config));
}

TEST_F(DragonflyFEPBridgeTest, ValidateConfigNegativePrecision) {
    config.sensory_precision = -1.0f;
    EXPECT_EQ(-1, dragonfly_fep_bridge_validate_config(&config));
}

TEST_F(DragonflyFEPBridgeTest, ValidateConfigInvalidLearningRate) {
    config.learning_rate = 0.0f;
    EXPECT_EQ(-1, dragonfly_fep_bridge_validate_config(&config));

    config.learning_rate = 1.5f;
    EXPECT_EQ(-1, dragonfly_fep_bridge_validate_config(&config));
}

TEST_F(DragonflyFEPBridgeTest, ValidateConfigZeroInferenceSteps) {
    config.inference_steps = 0;
    EXPECT_EQ(-1, dragonfly_fep_bridge_validate_config(&config));
}

//=============================================================================
// Lifecycle Tests
//=============================================================================

TEST_F(DragonflyFEPBridgeTest, CreateWithDefaultConfig) {
    bridge = dragonfly_fep_bridge_create(nullptr, nullptr, &config);
    EXPECT_NE(nullptr, bridge);
}

TEST_F(DragonflyFEPBridgeTest, CreateWithNullConfigUsesDefaults) {
    bridge = dragonfly_fep_bridge_create(nullptr, nullptr, nullptr);
    EXPECT_NE(nullptr, bridge);
}

TEST_F(DragonflyFEPBridgeTest, CreateWithInvalidConfigReturnsNull) {
    config.learning_rate = 0.0f;
    bridge = dragonfly_fep_bridge_create(nullptr, nullptr, &config);
    EXPECT_EQ(nullptr, bridge);
}

TEST_F(DragonflyFEPBridgeTest, DestroyNullSafe) {
    dragonfly_fep_bridge_destroy(nullptr);
    /* No crash = success */
}

TEST_F(DragonflyFEPBridgeTest, ResetSuccess) {
    CreateBridge();
    EXPECT_EQ(0, dragonfly_fep_bridge_reset(bridge));
}

TEST_F(DragonflyFEPBridgeTest, ResetNullReturnsError) {
    EXPECT_EQ(-1, dragonfly_fep_bridge_reset(nullptr));
}

//=============================================================================
// Prediction Error Tests
//=============================================================================

TEST_F(DragonflyFEPBridgeTest, ComputeErrorsSuccess) {
    CreateBridge();

    float observations[6] = {1.0f, 2.0f, 3.0f, 0.1f, 0.2f, 0.3f};
    dragonfly_fep_errors_t errors;
    EXPECT_EQ(0, dragonfly_fep_compute_errors(bridge, observations, 6, &errors));

    EXPECT_GE(errors.sensory_error, 0.0f);
    EXPECT_GE(errors.total_free_energy, 0.0f);
}

TEST_F(DragonflyFEPBridgeTest, ComputeErrorsNullReturnsError) {
    CreateBridge();
    float observations[6] = {1.0f, 2.0f, 3.0f, 0.1f, 0.2f, 0.3f};
    dragonfly_fep_errors_t errors;
    EXPECT_EQ(-1, dragonfly_fep_compute_errors(nullptr, observations, 6, &errors));
    EXPECT_EQ(-1, dragonfly_fep_compute_errors(bridge, nullptr, 6, &errors));
    EXPECT_EQ(-1, dragonfly_fep_compute_errors(bridge, observations, 6, nullptr));
}

TEST_F(DragonflyFEPBridgeTest, GetFreeEnergyValid) {
    CreateBridge();

    /* Compute errors first to set free energy */
    float observations[6] = {1.0f, 2.0f, 3.0f, 0.1f, 0.2f, 0.3f};
    dragonfly_fep_errors_t errors;
    dragonfly_fep_compute_errors(bridge, observations, 6, &errors);

    float fe = dragonfly_fep_get_free_energy(bridge);
    EXPECT_GE(fe, 0.0f);
}

TEST_F(DragonflyFEPBridgeTest, GetFreeEnergyNullReturnsZero) {
    EXPECT_EQ(0.0f, dragonfly_fep_get_free_energy(nullptr));
}

TEST_F(DragonflyFEPBridgeTest, GetSurpriseValid) {
    CreateBridge();

    float observations[6] = {10.0f, 20.0f, 30.0f, 0.1f, 0.2f, 0.3f};
    dragonfly_fep_errors_t errors;
    dragonfly_fep_compute_errors(bridge, observations, 6, &errors);

    float surprise = dragonfly_fep_get_surprise(bridge);
    EXPECT_GE(surprise, 0.0f);
}

TEST_F(DragonflyFEPBridgeTest, GetSurpriseNullReturnsZero) {
    EXPECT_EQ(0.0f, dragonfly_fep_get_surprise(nullptr));
}

TEST_F(DragonflyFEPBridgeTest, UpdatePrecisionSuccess) {
    CreateBridge();
    EXPECT_EQ(0, dragonfly_fep_update_precision(bridge, 0.9f, 0.8f));
}

TEST_F(DragonflyFEPBridgeTest, UpdatePrecisionNullReturnsError) {
    EXPECT_EQ(-1, dragonfly_fep_update_precision(nullptr, 0.9f, 0.8f));
}

//=============================================================================
// Active Inference Tests
//=============================================================================

TEST_F(DragonflyFEPBridgeTest, InferStateSuccess) {
    CreateBridge();

    float observations[6] = {1.0f, 2.0f, 3.0f, 0.1f, 0.2f, 0.3f};
    dragonfly_fep_inference_t inference;
    EXPECT_EQ(0, dragonfly_fep_infer_state(bridge, observations, 6, &inference));

    EXPECT_GT(inference.state_dim, 0u);
}

TEST_F(DragonflyFEPBridgeTest, InferStateNullReturnsError) {
    CreateBridge();
    float observations[6] = {1.0f, 2.0f, 3.0f, 0.1f, 0.2f, 0.3f};
    dragonfly_fep_inference_t inference;
    EXPECT_EQ(-1, dragonfly_fep_infer_state(nullptr, observations, 6, &inference));
    EXPECT_EQ(-1, dragonfly_fep_infer_state(bridge, nullptr, 6, &inference));
    EXPECT_EQ(-1, dragonfly_fep_infer_state(bridge, observations, 6, nullptr));
}

TEST_F(DragonflyFEPBridgeTest, InferStateUpdatesBeliefsTowardObservations) {
    CreateBridge();

    /* First observation */
    float obs1[6] = {10.0f, 20.0f, 30.0f, 1.0f, 2.0f, 3.0f};
    dragonfly_fep_inference_t inference1;
    dragonfly_fep_infer_state(bridge, obs1, 6, &inference1);

    /* Beliefs should move toward observations */
    /* With zeros initial beliefs and positive observations, beliefs should become positive */
    bool moved_toward = false;
    for (uint32_t i = 0; i < inference1.state_dim && i < 6; i++) {
        if (inference1.beliefs[i] > 0.0f) {
            moved_toward = true;
            break;
        }
    }
    EXPECT_TRUE(moved_toward);
}

TEST_F(DragonflyFEPBridgeTest, SelectActionSuccess) {
    CreateBridge();

    dragonfly_fep_action_t action;
    float value;
    EXPECT_EQ(0, dragonfly_fep_select_action(bridge, &action, &value));

    EXPECT_LE(action, FEP_ACTION_PREDICT);
}

TEST_F(DragonflyFEPBridgeTest, SelectActionNullReturnsError) {
    CreateBridge();
    dragonfly_fep_action_t action;
    EXPECT_EQ(-1, dragonfly_fep_select_action(nullptr, &action, nullptr));
    EXPECT_EQ(-1, dragonfly_fep_select_action(bridge, nullptr, nullptr));
}

TEST_F(DragonflyFEPBridgeTest, ApplyActionSuccess) {
    CreateBridge();
    EXPECT_EQ(0, dragonfly_fep_apply_action(bridge, FEP_ACTION_OBSERVE));
    EXPECT_EQ(0, dragonfly_fep_apply_action(bridge, FEP_ACTION_PURSUIT));
    EXPECT_EQ(0, dragonfly_fep_apply_action(bridge, FEP_ACTION_INTERCEPT));
    EXPECT_EQ(0, dragonfly_fep_apply_action(bridge, FEP_ACTION_PREDICT));
}

TEST_F(DragonflyFEPBridgeTest, ApplyActionNullReturnsError) {
    EXPECT_EQ(-1, dragonfly_fep_apply_action(nullptr, FEP_ACTION_OBSERVE));
}

TEST_F(DragonflyFEPBridgeTest, ApplyActionInvalidReturnsError) {
    CreateBridge();
    EXPECT_EQ(-1, dragonfly_fep_apply_action(bridge, (dragonfly_fep_action_t)99));
}

TEST_F(DragonflyFEPBridgeTest, ExpectedFreeEnergyValid) {
    CreateBridge();

    float efe_observe = dragonfly_fep_expected_free_energy(bridge, FEP_ACTION_OBSERVE);
    float efe_intercept = dragonfly_fep_expected_free_energy(bridge, FEP_ACTION_INTERCEPT);

    /* Intercept should have lower EFE (higher value) in pursuit scenario */
    EXPECT_NE(efe_observe, efe_intercept);
}

TEST_F(DragonflyFEPBridgeTest, ExpectedFreeEnergyNullReturnsLarge) {
    float efe = dragonfly_fep_expected_free_energy(nullptr, FEP_ACTION_OBSERVE);
    EXPECT_GT(efe, 1e9f);  /* Very large value indicates error */
}

//=============================================================================
// Generative Model Tests
//=============================================================================

TEST_F(DragonflyFEPBridgeTest, SetModelSuccess) {
    CreateBridge();
    EXPECT_EQ(0, dragonfly_fep_set_model(bridge, FEP_MODEL_LINEAR));
    EXPECT_EQ(0, dragonfly_fep_set_model(bridge, FEP_MODEL_CONSTANT_VELOCITY));
    EXPECT_EQ(0, dragonfly_fep_set_model(bridge, FEP_MODEL_CONSTANT_ACCELERATION));
    EXPECT_EQ(0, dragonfly_fep_set_model(bridge, FEP_MODEL_MANEUVERING));
    EXPECT_EQ(0, dragonfly_fep_set_model(bridge, FEP_MODEL_EVASIVE));
}

TEST_F(DragonflyFEPBridgeTest, SetModelNullReturnsError) {
    EXPECT_EQ(-1, dragonfly_fep_set_model(nullptr, FEP_MODEL_LINEAR));
}

TEST_F(DragonflyFEPBridgeTest, SetModelInvalidReturnsError) {
    CreateBridge();
    EXPECT_EQ(-1, dragonfly_fep_set_model(bridge, (dragonfly_fep_model_t)99));
}

TEST_F(DragonflyFEPBridgeTest, GetBestModelValid) {
    CreateBridge();
    dragonfly_fep_model_t best = dragonfly_fep_get_best_model(bridge);
    EXPECT_LE(best, FEP_MODEL_EVASIVE);
}

TEST_F(DragonflyFEPBridgeTest, GetBestModelNullReturnsLinear) {
    EXPECT_EQ(FEP_MODEL_LINEAR, dragonfly_fep_get_best_model(nullptr));
}

TEST_F(DragonflyFEPBridgeTest, GetModelEvidenceValid) {
    CreateBridge();
    for (int m = 0; m <= FEP_MODEL_EVASIVE; m++) {
        float evidence = dragonfly_fep_get_model_evidence(bridge, (dragonfly_fep_model_t)m);
        EXPECT_GE(evidence, 0.0f);
        EXPECT_LE(evidence, 1.0f);
    }
}

TEST_F(DragonflyFEPBridgeTest, GetModelEvidenceNullReturnsZero) {
    EXPECT_EQ(0.0f, dragonfly_fep_get_model_evidence(nullptr, FEP_MODEL_LINEAR));
}

TEST_F(DragonflyFEPBridgeTest, GetModelEvidenceInvalidReturnsZero) {
    CreateBridge();
    EXPECT_EQ(0.0f, dragonfly_fep_get_model_evidence(bridge, (dragonfly_fep_model_t)99));
}

TEST_F(DragonflyFEPBridgeTest, PredictSuccess) {
    CreateBridge();

    /* First set some beliefs */
    float observations[6] = {1.0f, 2.0f, 3.0f, 0.1f, 0.2f, 0.3f};
    dragonfly_fep_inference_t inference;
    dragonfly_fep_infer_state(bridge, observations, 6, &inference);

    float predicted[6];
    int dim = dragonfly_fep_predict(bridge, 100.0f, predicted, 6);
    EXPECT_GT(dim, 0);
}

TEST_F(DragonflyFEPBridgeTest, PredictNullReturnsError) {
    CreateBridge();
    float predicted[6];
    EXPECT_EQ(-1, dragonfly_fep_predict(nullptr, 100.0f, predicted, 6));
    EXPECT_EQ(-1, dragonfly_fep_predict(bridge, 100.0f, nullptr, 6));
}

//=============================================================================
// Integration Tests
//=============================================================================

TEST_F(DragonflyFEPBridgeTest, ConnectDragonflySuccess) {
    CreateBridge();
    int dummy;
    EXPECT_EQ(0, dragonfly_fep_connect_dragonfly(bridge, (dragonfly_system_t*)&dummy));
}

TEST_F(DragonflyFEPBridgeTest, ConnectDragonflyNullBridgeReturnsError) {
    int dummy;
    EXPECT_EQ(-1, dragonfly_fep_connect_dragonfly(nullptr, (dragonfly_system_t*)&dummy));
}

TEST_F(DragonflyFEPBridgeTest, ConnectSystemSuccess) {
    CreateBridge();
    int dummy;
    EXPECT_EQ(0, dragonfly_fep_connect_system(bridge, &dummy));
}

TEST_F(DragonflyFEPBridgeTest, ConnectSystemNullBridgeReturnsError) {
    int dummy;
    EXPECT_EQ(-1, dragonfly_fep_connect_system(nullptr, &dummy));
}

TEST_F(DragonflyFEPBridgeTest, UpdateSuccess) {
    CreateBridge();
    EXPECT_EQ(0, dragonfly_fep_update(bridge, 16.0f));
}

TEST_F(DragonflyFEPBridgeTest, UpdateNullReturnsError) {
    EXPECT_EQ(-1, dragonfly_fep_update(nullptr, 16.0f));
}

TEST_F(DragonflyFEPBridgeTest, SyncWithTrackerSuccess) {
    CreateBridge();
    EXPECT_EQ(0, dragonfly_fep_sync_with_tracker(bridge));
}

TEST_F(DragonflyFEPBridgeTest, SyncWithTrackerNullReturnsError) {
    EXPECT_EQ(-1, dragonfly_fep_sync_with_tracker(nullptr));
}

TEST_F(DragonflyFEPBridgeTest, SyncWithTSDNSuccess) {
    CreateBridge();
    EXPECT_EQ(0, dragonfly_fep_sync_with_tsdn(bridge));
}

TEST_F(DragonflyFEPBridgeTest, SyncWithTSDNNullReturnsError) {
    EXPECT_EQ(-1, dragonfly_fep_sync_with_tsdn(nullptr));
}

//=============================================================================
// Statistics Tests
//=============================================================================

TEST_F(DragonflyFEPBridgeTest, GetStatsSuccess) {
    CreateBridge();
    dragonfly_fep_stats_t stats;
    EXPECT_EQ(0, dragonfly_fep_bridge_get_stats(bridge, &stats));
}

TEST_F(DragonflyFEPBridgeTest, GetStatsNullReturnsError) {
    CreateBridge();
    dragonfly_fep_stats_t stats;
    EXPECT_EQ(-1, dragonfly_fep_bridge_get_stats(nullptr, &stats));
    EXPECT_EQ(-1, dragonfly_fep_bridge_get_stats(bridge, nullptr));
}

TEST_F(DragonflyFEPBridgeTest, ResetStatsSuccess) {
    CreateBridge();
    EXPECT_EQ(0, dragonfly_fep_bridge_reset_stats(bridge));
}

TEST_F(DragonflyFEPBridgeTest, ResetStatsNullReturnsError) {
    EXPECT_EQ(-1, dragonfly_fep_bridge_reset_stats(nullptr));
}

TEST_F(DragonflyFEPBridgeTest, StatsTrackInferenceSteps) {
    CreateBridge();

    /* Perform some inference */
    float observations[6] = {1.0f, 2.0f, 3.0f, 0.1f, 0.2f, 0.3f};
    dragonfly_fep_inference_t inference;
    for (int i = 0; i < 5; i++) {
        dragonfly_fep_infer_state(bridge, observations, 6, &inference);
    }

    dragonfly_fep_stats_t stats;
    dragonfly_fep_bridge_get_stats(bridge, &stats);
    EXPECT_GT(stats.inference_steps_total, 0u);
}

TEST_F(DragonflyFEPBridgeTest, StatsTrackModelSwitches) {
    CreateBridge();

    /* Switch models */
    dragonfly_fep_set_model(bridge, FEP_MODEL_LINEAR);
    dragonfly_fep_set_model(bridge, FEP_MODEL_EVASIVE);
    dragonfly_fep_set_model(bridge, FEP_MODEL_MANEUVERING);

    dragonfly_fep_stats_t stats;
    dragonfly_fep_bridge_get_stats(bridge, &stats);
    EXPECT_GE(stats.model_switches, 2u);  /* At least 2 switches */
}

//=============================================================================
// Utility Tests
//=============================================================================

TEST_F(DragonflyFEPBridgeTest, ModelNameValid) {
    EXPECT_STREQ("linear", dragonfly_fep_model_name(FEP_MODEL_LINEAR));
    EXPECT_STREQ("constant_velocity", dragonfly_fep_model_name(FEP_MODEL_CONSTANT_VELOCITY));
    EXPECT_STREQ("constant_acceleration", dragonfly_fep_model_name(FEP_MODEL_CONSTANT_ACCELERATION));
    EXPECT_STREQ("maneuvering", dragonfly_fep_model_name(FEP_MODEL_MANEUVERING));
    EXPECT_STREQ("evasive", dragonfly_fep_model_name(FEP_MODEL_EVASIVE));
}

TEST_F(DragonflyFEPBridgeTest, ModelNameInvalid) {
    EXPECT_STREQ("unknown", dragonfly_fep_model_name((dragonfly_fep_model_t)99));
}

TEST_F(DragonflyFEPBridgeTest, ActionNameValid) {
    EXPECT_STREQ("observe", dragonfly_fep_action_name(FEP_ACTION_OBSERVE));
    EXPECT_STREQ("pursuit", dragonfly_fep_action_name(FEP_ACTION_PURSUIT));
    EXPECT_STREQ("intercept", dragonfly_fep_action_name(FEP_ACTION_INTERCEPT));
    EXPECT_STREQ("predict", dragonfly_fep_action_name(FEP_ACTION_PREDICT));
}

TEST_F(DragonflyFEPBridgeTest, ActionNameInvalid) {
    EXPECT_STREQ("unknown", dragonfly_fep_action_name((dragonfly_fep_action_t)99));
}

TEST_F(DragonflyFEPBridgeTest, PrecisionModeNameValid) {
    EXPECT_STREQ("fixed", dragonfly_fep_precision_mode_name(FEP_PRECISION_FIXED));
    EXPECT_STREQ("adaptive", dragonfly_fep_precision_mode_name(FEP_PRECISION_ADAPTIVE));
    EXPECT_STREQ("hierarchical", dragonfly_fep_precision_mode_name(FEP_PRECISION_HIERARCHICAL));
}

TEST_F(DragonflyFEPBridgeTest, PrecisionModeNameInvalid) {
    EXPECT_STREQ("unknown", dragonfly_fep_precision_mode_name((dragonfly_fep_precision_mode_t)99));
}

//=============================================================================
// Active Inference Cycle Test
//=============================================================================

TEST_F(DragonflyFEPBridgeTest, FullActiveInferenceCycle) {
    CreateBridge();

    /* Simulate a target tracking scenario */
    float target_x = 0.0f, target_y = 0.0f, target_z = 1.0f;
    float target_vx = 0.5f, target_vy = 0.3f, target_vz = 0.0f;

    dragonfly_fep_errors_t errors;
    dragonfly_fep_inference_t inference;
    dragonfly_fep_action_t action;

    for (int step = 0; step < 20; step++) {
        /* Update target position */
        target_x += target_vx * 0.016f;
        target_y += target_vy * 0.016f;
        target_z += target_vz * 0.016f;

        /* Create observation */
        float obs[6] = {target_x, target_y, target_z, target_vx, target_vy, target_vz};

        /* Compute prediction errors */
        dragonfly_fep_compute_errors(bridge, obs, 6, &errors);

        /* Update beliefs */
        dragonfly_fep_infer_state(bridge, obs, 6, &inference);

        /* Select action */
        dragonfly_fep_select_action(bridge, &action, nullptr);

        /* Apply action */
        dragonfly_fep_apply_action(bridge, action);

        /* Update bridge */
        dragonfly_fep_update(bridge, 16.0f);
    }

    /* Verify stats accumulated */
    dragonfly_fep_stats_t stats;
    dragonfly_fep_bridge_get_stats(bridge, &stats);
    EXPECT_EQ(20u, stats.predictions_made);
    EXPECT_GT(stats.inference_steps_total, 0u);
}
