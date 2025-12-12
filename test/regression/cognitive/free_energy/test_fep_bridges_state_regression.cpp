/**
 * @file test_fep_bridges_state_regression.cpp
 * @brief State consistency regression tests for FEP bridge modules
 *
 * WHAT: Tests state consistency across update cycles, range validation, trend verification
 * WHY:  Prevent state corruption, ensure values stay in valid ranges, verify expected behaviors
 * HOW:  Monitor state over many updates, validate ranges, check monotonicity/convergence
 */

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include "cognitive/free_energy/nimcp_free_energy.h"
#include "cognitive/free_energy/nimcp_fep_immune_bridge.h"
#include "cognitive/free_energy/nimcp_fep_consciousness.h"
#include "cognitive/free_energy/nimcp_fep_context.h"
#include "cognitive/free_energy/nimcp_fep_curiosity.h"
#include "cognitive/free_energy/nimcp_fep_learning.h"
#include "cognitive/free_energy/nimcp_fep_neuromod.h"
#include "cognitive/free_energy/nimcp_fep_planning.h"
#include "cognitive/free_energy/nimcp_fep_sleep.h"
#include "cognitive/free_energy/nimcp_fep_evidence.h"
#include "cognitive/immune/nimcp_brain_immune.h"
#include "cognitive/introspection/nimcp_introspection.h"

class FEPStateRegressionTest : public ::testing::Test {
protected:
    static const int ITERATIONS = 1000;
    static const uint64_t TIMESTEP_MS = 16;

    fep_system_t* fep = nullptr;
    brain_immune_system_t* immune = nullptr;

    static const uint32_t OBS_DIM = 16;
    static const uint32_t ACTION_DIM = 8;

    void SetUp() override {
        fep_config_t fep_config;
        fep_default_config(&fep_config);
        fep_config.num_levels = 2;
        fep = fep_create(&fep_config, OBS_DIM, ACTION_DIM);
        ASSERT_NE(fep, nullptr);

        brain_immune_config_t immune_config;
        brain_immune_default_config(&immune_config);
        immune = brain_immune_create(&immune_config);
        ASSERT_NE(immune, nullptr);
    }

    void TearDown() override {
        if (fep) {
            fep_destroy(fep);
            fep = nullptr;
        }
        if (immune) {
            brain_immune_destroy(immune);
            immune = nullptr;
        }
    }

    // Helper to check if value is in valid range
    bool in_range(float value, float min, float max) {
        return value >= min && value <= max && !std::isnan(value) && !std::isinf(value);
    }
};

/* ============================================================================
 * Immune Bridge State Consistency Tests
 * ============================================================================ */

TEST_F(FEPStateRegressionTest, ImmuneBridgePrecisionRangeRegression) {
    fep_immune_bridge_t* bridge = fep_immune_bridge_create(nullptr);
    ASSERT_NE(bridge, nullptr);

    fep_immune_bridge_connect_fep(bridge, fep);
    fep_immune_bridge_connect_immune(bridge, immune);

    // Track precision reduction over many updates
    for (int i = 0; i < ITERATIONS; i++) {
        fep_immune_bridge_update(bridge, TIMESTEP_MS);

        fep_immune_state_t state;
        int ret = fep_immune_bridge_get_state(bridge, &state);
        ASSERT_EQ(ret, 0);

        // Precision reduction must be in [0, 1]
        EXPECT_TRUE(in_range(state.precision_reduction, 0.0f, 1.0f))
            << "Iteration " << i << ": precision_reduction=" << state.precision_reduction;

        // Learning impairment must be in [0, 1]
        EXPECT_TRUE(in_range(state.learning_impairment, 0.0f, 1.0f))
            << "Iteration " << i << ": learning_impairment=" << state.learning_impairment;

        // Current inflammation must be in [0, 1]
        EXPECT_TRUE(in_range(state.current_inflammation, 0.0f, 1.0f))
            << "Iteration " << i << ": current_inflammation=" << state.current_inflammation;

        // Sickness intensity must be in [0, 1]
        EXPECT_TRUE(in_range(state.sickness_intensity, 0.0f, 1.0f))
            << "Iteration " << i << ": sickness_intensity=" << state.sickness_intensity;

        // Recovery progress must be in [0, 1]
        EXPECT_TRUE(in_range(state.recovery_progress, 0.0f, 1.0f))
            << "Iteration " << i << ": recovery_progress=" << state.recovery_progress;
    }

    fep_immune_bridge_destroy(bridge);
}

TEST_F(FEPStateRegressionTest, ImmuneBridgeStatisticsMonotonicity) {
    fep_immune_bridge_t* bridge = fep_immune_bridge_create(nullptr);
    ASSERT_NE(bridge, nullptr);

    fep_immune_bridge_connect_fep(bridge, fep);
    fep_immune_bridge_connect_immune(bridge, immune);

    fep_immune_stats_t prev_stats;
    int ret = fep_immune_bridge_get_stats(bridge, &prev_stats);
    ASSERT_EQ(ret, 0);

    // Statistics should only increase (counters)
    for (int i = 0; i < 100; i++) {
        fep_immune_bridge_update(bridge, TIMESTEP_MS);

        if (i % 10 == 0) {
            float magnitude = 5.0f + (i % 5) * 2.0f;
            fep_immune_report_prediction_failure(bridge, magnitude);
        }

        fep_immune_stats_t curr_stats;
        ret = fep_immune_bridge_get_stats(bridge, &curr_stats);
        ASSERT_EQ(ret, 0);

        // Counters should be monotonic
        EXPECT_GE(curr_stats.prediction_failures, prev_stats.prediction_failures);
        EXPECT_GE(curr_stats.immune_activations, prev_stats.immune_activations);

        prev_stats = curr_stats;
    }

    fep_immune_bridge_destroy(bridge);
}

TEST_F(FEPStateRegressionTest, ImmuneBridgeInflammationLevelConsistency) {
    fep_immune_bridge_t* bridge = fep_immune_bridge_create(nullptr);
    ASSERT_NE(bridge, nullptr);

    fep_immune_bridge_connect_fep(bridge, fep);
    fep_immune_bridge_connect_immune(bridge, immune);

    // Verify inflammation level matches current_inflammation value
    for (int i = 0; i < 100; i++) {
        fep_immune_bridge_update(bridge, TIMESTEP_MS);

        fep_immune_state_t state;
        int ret = fep_immune_bridge_get_state(bridge, &state);
        ASSERT_EQ(ret, 0);

        brain_inflammation_level_t level = fep_immune_get_inflammation_level(bridge);

        // Check consistency between level enum and float value
        switch (level) {
            case INFLAMMATION_NONE:
                EXPECT_LT(state.current_inflammation, 0.1f);
                break;
            case INFLAMMATION_LOCAL:
                EXPECT_GE(state.current_inflammation, 0.1f);
                EXPECT_LT(state.current_inflammation, 0.3f);
                break;
            case INFLAMMATION_REGIONAL:
                EXPECT_GE(state.current_inflammation, 0.3f);
                EXPECT_LT(state.current_inflammation, 0.6f);
                break;
            case INFLAMMATION_SYSTEMIC:
                EXPECT_GE(state.current_inflammation, 0.6f);
                EXPECT_LT(state.current_inflammation, 0.9f);
                break;
            case INFLAMMATION_STORM:
                EXPECT_GE(state.current_inflammation, 0.9f);
                break;
        }
    }

    fep_immune_bridge_destroy(bridge);
}

/* ============================================================================
 * Learning Bridge State Consistency Tests
 * ============================================================================ */

TEST_F(FEPStateRegressionTest, LearningBridgeLossMonotonicity) {
    fep_learning_config_t config;
    fep_learning_default_config(&config);
    config.transition_learning_rate = 0.01f;
    config.likelihood_learning_rate = 0.01f;

    fep_learning_system_t* learning = fep_learning_create(&config);
    ASSERT_NE(learning, nullptr);

    fep_learning_connect_fep(learning, fep);

    float observations[OBS_DIM];
    float prev_states[8] = {0.5f, 0.5f, 0.3f, 0.3f, 0.2f, 0.2f, 0.1f, 0.1f};
    float curr_states[8] = {0.6f, 0.6f, 0.4f, 0.4f, 0.3f, 0.3f, 0.2f, 0.2f};

    for (int i = 0; i < OBS_DIM; i++) {
        observations[i] = 0.5f + 0.1f * (i % 3);
    }

    std::vector<float> loss_history;

    // Train and track loss
    for (int i = 0; i < 500; i++) {
        fep_learning_update_transition(learning, curr_states, 8, prev_states, 8);
        fep_learning_update_likelihood(learning, observations, OBS_DIM, curr_states, 8);

        float loss;
        int ret = fep_learning_get_loss(learning, &loss);
        ASSERT_EQ(ret, 0);

        EXPECT_TRUE(in_range(loss, 0.0f, 1000.0f)) << "Iteration " << i << ": loss=" << loss;
        loss_history.push_back(loss);
    }

    // Loss should show general decreasing trend
    // Compare first 50 vs last 50
    float avg_early = 0.0f, avg_late = 0.0f;
    for (int i = 0; i < 50; i++) {
        avg_early += loss_history[i];
        avg_late += loss_history[loss_history.size() - 50 + i];
    }
    avg_early /= 50.0f;
    avg_late /= 50.0f;

    EXPECT_LT(avg_late, avg_early) << "Loss should decrease with training";

    fep_learning_destroy(learning);
}

TEST_F(FEPStateRegressionTest, LearningBridgeGradientNormRegression) {
    fep_learning_system_t* learning = fep_learning_create(nullptr);
    ASSERT_NE(learning, nullptr);

    fep_learning_connect_fep(learning, fep);

    float observations[OBS_DIM];
    float states[8];
    for (int i = 0; i < OBS_DIM; i++) observations[i] = 0.5f;
    for (int i = 0; i < 8; i++) states[i] = 0.5f;

    // Gradient norm should remain finite
    for (int i = 0; i < ITERATIONS; i++) {
        fep_learning_update_likelihood(learning, observations, OBS_DIM, states, 8);

        float grad_norm;
        int ret = fep_learning_get_gradient_norm(learning, &grad_norm);
        ASSERT_EQ(ret, 0);

        EXPECT_FALSE(std::isnan(grad_norm)) << "Iteration " << i;
        EXPECT_FALSE(std::isinf(grad_norm)) << "Iteration " << i;
        EXPECT_GE(grad_norm, 0.0f) << "Iteration " << i;
        EXPECT_LT(grad_norm, 1000.0f) << "Iteration " << i; // Reasonable upper bound
    }

    fep_learning_destroy(learning);
}

TEST_F(FEPStateRegressionTest, LearningBridgeConvergenceDetection) {
    fep_learning_config_t config;
    fep_learning_default_config(&config);
    config.convergence_threshold = 0.01f;

    fep_learning_system_t* learning = fep_learning_create(&config);
    ASSERT_NE(learning, nullptr);

    fep_learning_connect_fep(learning, fep);

    float observations[OBS_DIM];
    float states[8];
    for (int i = 0; i < OBS_DIM; i++) observations[i] = 0.5f;
    for (int i = 0; i < 8; i++) states[i] = 0.5f;

    bool converged = false;
    int convergence_iter = -1;

    for (int i = 0; i < 1000; i++) {
        fep_learning_update_likelihood(learning, observations, OBS_DIM, states, 8);

        bool is_converged = fep_learning_has_converged(learning);
        if (is_converged && !converged) {
            converged = true;
            convergence_iter = i;
        }

        // Once converged, should stay converged (or very close)
        if (converged && i > convergence_iter + 50) {
            EXPECT_TRUE(is_converged || i < convergence_iter + 100)
                << "Should maintain convergence after iteration " << convergence_iter;
        }
    }

    fep_learning_destroy(learning);
}

/* ============================================================================
 * Context Bridge State Consistency Tests
 * ============================================================================ */

TEST_F(FEPStateRegressionTest, ContextBridgeActivationRangeRegression) {
    fep_context_system_t* context = fep_context_create(nullptr);
    ASSERT_NE(context, nullptr);

    fep_context_connect_fep(context, fep);

    uint32_t ctx_ids[5];
    for (int i = 0; i < 5; i++) {
        char name[32];
        snprintf(name, sizeof(name), "context_%d", i);
        int ret = fep_context_add_context(context, name, &ctx_ids[i]);
        ASSERT_EQ(ret, 0);
    }

    float observations[OBS_DIM];

    // Verify activation levels stay in [0, 1]
    for (int i = 0; i < ITERATIONS; i++) {
        for (int j = 0; j < OBS_DIM; j++) {
            observations[j] = 0.5f + 0.3f * sinf(i * 0.01f + j);
        }

        uint32_t inferred_ctx;
        float confidence;
        int ret = fep_context_infer(context, observations, OBS_DIM, &inferred_ctx, &confidence);
        ASSERT_EQ(ret, 0);

        EXPECT_TRUE(in_range(confidence, 0.0f, 1.0f))
            << "Iteration " << i << ": confidence=" << confidence;

        // Get activation for all contexts
        for (int j = 0; j < 5; j++) {
            float activation;
            ret = fep_context_get_activation(context, ctx_ids[j], &activation);
            if (ret == 0) {
                EXPECT_TRUE(in_range(activation, 0.0f, 1.0f))
                    << "Iteration " << i << ", context " << j << ": activation=" << activation;
            }
        }
    }

    fep_context_destroy(context);
}

TEST_F(FEPStateRegressionTest, ContextBridgeSwitchConsistency) {
    fep_context_system_t* context = fep_context_create(nullptr);
    ASSERT_NE(context, nullptr);

    fep_context_connect_fep(context, fep);

    uint32_t ctx1, ctx2;
    fep_context_add_context(context, "context_1", &ctx1);
    fep_context_add_context(context, "context_2", &ctx2);

    // After explicit switch, active context should match
    for (int i = 0; i < 100; i++) {
        uint32_t target = (i % 2 == 0) ? ctx1 : ctx2;
        int ret = fep_context_switch_to(context, target);
        ASSERT_EQ(ret, 0);

        uint32_t active;
        ret = fep_context_get_active(context, &active);
        ASSERT_EQ(ret, 0);

        EXPECT_EQ(active, target) << "Iteration " << i;
    }

    fep_context_destroy(context);
}

/* ============================================================================
 * Neuromodulation Bridge State Consistency Tests
 * ============================================================================ */

TEST_F(FEPStateRegressionTest, NeuromodBridgeLevelRangeRegression) {
    fep_neuromod_system_t* neuromod = fep_neuromod_create(nullptr);
    ASSERT_NE(neuromod, nullptr);

    fep_neuromod_connect_fep(neuromod, fep);

    // Verify all neuromodulator levels stay in [0, 1]
    for (int i = 0; i < ITERATIONS; i++) {
        fep_neuromod_update(neuromod, TIMESTEP_MS);

        fep_neuromod_state_t state;
        int ret = fep_neuromod_get_state(neuromod, &state);
        ASSERT_EQ(ret, 0);

        EXPECT_TRUE(in_range(state.dopamine_level, 0.0f, 1.0f))
            << "Iteration " << i << ": dopamine=" << state.dopamine_level;

        EXPECT_TRUE(in_range(state.serotonin_level, 0.0f, 1.0f))
            << "Iteration " << i << ": serotonin=" << state.serotonin_level;

        EXPECT_TRUE(in_range(state.norepinephrine_level, 0.0f, 1.0f))
            << "Iteration " << i << ": norepinephrine=" << state.norepinephrine_level;

        EXPECT_TRUE(in_range(state.acetylcholine_level, 0.0f, 1.0f))
            << "Iteration " << i << ": acetylcholine=" << state.acetylcholine_level;
    }

    fep_neuromod_destroy(neuromod);
}

TEST_F(FEPStateRegressionTest, NeuromodBridgeSetLevelConsistency) {
    fep_neuromod_system_t* neuromod = fep_neuromod_create(nullptr);
    ASSERT_NE(neuromod, nullptr);

    fep_neuromod_connect_fep(neuromod, fep);

    const float test_levels[] = {0.0f, 0.25f, 0.5f, 0.75f, 1.0f};

    for (float level : test_levels) {
        // Set dopamine
        int ret1 = fep_neuromod_set_dopamine(neuromod, level);
        ASSERT_EQ(ret1, 0);

        // Verify it was set correctly
        fep_neuromod_state_t state;
        int ret2 = fep_neuromod_get_state(neuromod, &state);
        ASSERT_EQ(ret2, 0);

        EXPECT_NEAR(state.dopamine_level, level, 0.01f)
            << "Set level=" << level << " but got " << state.dopamine_level;
    }

    fep_neuromod_destroy(neuromod);
}

/* ============================================================================
 * Sleep Bridge State Consistency Tests
 * ============================================================================ */

TEST_F(FEPStateRegressionTest, SleepBridgeStateTransitionConsistency) {
    fep_sleep_system_t* sleep = fep_sleep_create(nullptr);
    ASSERT_NE(sleep, nullptr);

    fep_sleep_connect_fep(sleep, fep);

    // Test state transitions are consistent
    for (int cycle = 0; cycle < 10; cycle++) {
        // Enter sleep
        int ret1 = fep_sleep_enter_sleep(sleep);
        ASSERT_EQ(ret1, 0);

        fep_sleep_state_t state1;
        fep_sleep_get_state(sleep, &state1);
        EXPECT_TRUE(state1.is_asleep);

        // Update in sleep
        for (int i = 0; i < 50; i++) {
            fep_sleep_update(sleep, TIMESTEP_MS);

            fep_sleep_state_t state;
            fep_sleep_get_state(sleep, &state);
            EXPECT_TRUE(state.is_asleep) << "Cycle " << cycle << ", update " << i;
        }

        // Wake up
        int ret2 = fep_sleep_wake_up(sleep);
        ASSERT_EQ(ret2, 0);

        fep_sleep_state_t state2;
        fep_sleep_get_state(sleep, &state2);
        EXPECT_FALSE(state2.is_asleep);

        // Update while awake
        for (int i = 0; i < 50; i++) {
            fep_sleep_update(sleep, TIMESTEP_MS);

            fep_sleep_state_t state;
            fep_sleep_get_state(sleep, &state);
            EXPECT_FALSE(state.is_asleep) << "Cycle " << cycle << ", update " << i;
        }
    }

    fep_sleep_destroy(sleep);
}

TEST_F(FEPStateRegressionTest, SleepBridgeTimeAccumulation) {
    fep_sleep_system_t* sleep = fep_sleep_create(nullptr);
    ASSERT_NE(sleep, nullptr);

    fep_sleep_connect_fep(sleep, fep);

    fep_sleep_enter_sleep(sleep);

    uint64_t expected_time = 0;

    // Verify time accumulates correctly
    for (int i = 0; i < 100; i++) {
        fep_sleep_update(sleep, TIMESTEP_MS);
        expected_time += TIMESTEP_MS;

        fep_sleep_state_t state;
        fep_sleep_get_state(sleep, &state);

        // Allow small tolerance for floating point
        EXPECT_NEAR(state.time_asleep_ms, expected_time, TIMESTEP_MS)
            << "Iteration " << i;
    }

    fep_sleep_destroy(sleep);
}

/* ============================================================================
 * Curiosity Bridge State Consistency Tests
 * ============================================================================ */

TEST_F(FEPStateRegressionTest, CuriosityBridgeNoveltyRangeRegression) {
    fep_curiosity_config_t config;
    fep_curiosity_default_config(&config);

    fep_curiosity_system_t* curiosity = fep_curiosity_create(&config);
    ASSERT_NE(curiosity, nullptr);

    fep_curiosity_connect_fep(curiosity, fep);

    float observations[OBS_DIM];

    // Novelty should stay in [0, 1] range
    for (int i = 0; i < ITERATIONS; i++) {
        for (int j = 0; j < OBS_DIM; j++) {
            observations[j] = 0.5f + 0.3f * sinf(i * 0.01f + j);
        }

        float novelty;
        int ret = fep_curiosity_compute_novelty(curiosity, observations, OBS_DIM, &novelty);
        ASSERT_EQ(ret, 0);

        EXPECT_TRUE(in_range(novelty, 0.0f, 1.0f))
            << "Iteration " << i << ": novelty=" << novelty;
    }

    fep_curiosity_destroy(curiosity);
}

TEST_F(FEPStateRegressionTest, CuriosityBridgeIntrinsicMotivationRegression) {
    fep_curiosity_config_t config;
    fep_curiosity_default_config(&config);

    fep_curiosity_system_t* curiosity = fep_curiosity_create(&config);
    ASSERT_NE(curiosity, nullptr);

    fep_curiosity_connect_fep(curiosity, fep);

    // Intrinsic motivation should stay in valid range
    for (int i = 0; i < ITERATIONS; i++) {
        fep_curiosity_update(curiosity, TIMESTEP_MS);

        float motivation;
        int ret = fep_curiosity_get_intrinsic_motivation(curiosity, &motivation);
        ASSERT_EQ(ret, 0);

        EXPECT_TRUE(in_range(motivation, 0.0f, 10.0f))
            << "Iteration " << i << ": motivation=" << motivation;
    }

    fep_curiosity_destroy(curiosity);
}

/* ============================================================================
 * Evidence Bridge State Consistency Tests
 * ============================================================================ */

TEST_F(FEPStateRegressionTest, EvidenceBridgeValueRangeRegression) {
    fep_evidence_config_t config;
    fep_evidence_default_config(&config);

    fep_evidence_system_t* evidence = fep_evidence_create(&config);
    ASSERT_NE(evidence, nullptr);

    fep_evidence_connect_fep(evidence, fep);

    // Evidence values should be finite
    for (int i = 0; i < ITERATIONS; i++) {
        fep_evidence_update(evidence, TIMESTEP_MS);

        float evidence_val;
        int ret = fep_evidence_compute(evidence, &evidence_val);
        ASSERT_EQ(ret, 0);

        EXPECT_FALSE(std::isnan(evidence_val)) << "Iteration " << i;
        EXPECT_FALSE(std::isinf(evidence_val)) << "Iteration " << i;
        EXPECT_GT(evidence_val, -1000.0f) << "Iteration " << i; // Reasonable lower bound
    }

    fep_evidence_destroy(evidence);
}
