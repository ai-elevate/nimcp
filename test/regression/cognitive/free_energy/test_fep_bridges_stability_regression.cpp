/**
 * @file test_fep_bridges_stability_regression.cpp
 * @brief Stability regression tests for FEP bridge modules
 *
 * WHAT: Tests bridge stability under repeated operations, stress conditions
 * WHY:  Prevent regressions in long-running stability, memory safety
 * HOW:  Repeated update cycles, stress testing, memory leak detection
 */

#include <gtest/gtest.h>
#include <chrono>
#include <vector>
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

class FEPStabilityRegressionTest : public ::testing::Test {
protected:
    static const int ITERATIONS = 1000;
    static const int STRESS_ITERATIONS = 5000;
    static const uint64_t TIMESTEP_MS = 16; // 60Hz update rate

    fep_system_t* fep = nullptr;
    brain_immune_system_t* immune = nullptr;
    introspection_context_t introspection = nullptr;

    static const uint32_t OBS_DIM = 16;
    static const uint32_t ACTION_DIM = 8;

    void SetUp() override {
        // Create FEP system
        fep_config_t fep_config;
        fep_default_config(&fep_config);
        fep_config.num_levels = 2;
        fep = fep_create(&fep_config, OBS_DIM, ACTION_DIM);
        ASSERT_NE(fep, nullptr);

        // Create brain immune system
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
};

/* ============================================================================
 * Immune Bridge Stability Tests
 * ============================================================================ */

TEST_F(FEPStabilityRegressionTest, ImmuneBridgeRepeatedUpdateStability) {
    fep_immune_config_t config;
    fep_immune_bridge_default_config(&config);
    fep_immune_bridge_t* bridge = fep_immune_bridge_create(&config);
    ASSERT_NE(bridge, nullptr);

    fep_immune_bridge_connect_fep(bridge, fep);
    fep_immune_bridge_connect_immune(bridge, immune);

    // Repeated updates should not crash or cause memory issues
    for (int i = 0; i < ITERATIONS; i++) {
        int ret = fep_immune_bridge_update(bridge, TIMESTEP_MS);
        EXPECT_EQ(ret, 0);
    }

    // Verify state remains valid
    fep_immune_state_t state;
    int ret = fep_immune_bridge_get_state(bridge, &state);
    EXPECT_EQ(ret, 0);
    EXPECT_GE(state.precision_reduction, 0.0f);
    EXPECT_LE(state.precision_reduction, 1.0f);

    fep_immune_bridge_destroy(bridge);
}

TEST_F(FEPStabilityRegressionTest, ImmuneBridgeStressTest) {
    fep_immune_config_t config;
    fep_immune_bridge_default_config(&config);
    fep_immune_bridge_t* bridge = fep_immune_bridge_create(&config);
    ASSERT_NE(bridge, nullptr);

    fep_immune_bridge_connect_fep(bridge, fep);
    fep_immune_bridge_connect_immune(bridge, immune);

    // High-frequency stress test
    for (int i = 0; i < STRESS_ITERATIONS; i++) {
        fep_immune_bridge_update(bridge, TIMESTEP_MS);

        // Periodically report prediction failures
        if (i % 100 == 0) {
            float magnitude = 5.0f + (i % 10) * 2.0f;
            fep_immune_report_prediction_failure(bridge, magnitude);
        }
    }

    // Check statistics are reasonable
    fep_immune_stats_t stats;
    int ret = fep_immune_bridge_get_stats(bridge, &stats);
    EXPECT_EQ(ret, 0);
    EXPECT_GT(stats.total_updates, 0);

    fep_immune_bridge_destroy(bridge);
}

TEST_F(FEPStabilityRegressionTest, ImmuneBridgeRapidConnectionCycles) {
    fep_immune_config_t config;
    fep_immune_bridge_default_config(&config);
    fep_immune_bridge_t* bridge = fep_immune_bridge_create(&config);
    ASSERT_NE(bridge, nullptr);

    // Rapid connect/disconnect cycles
    for (int i = 0; i < 100; i++) {
        int ret1 = fep_immune_bridge_connect_fep(bridge, fep);
        EXPECT_EQ(ret1, 0);

        int ret2 = fep_immune_bridge_connect_immune(bridge, immune);
        EXPECT_EQ(ret2, 0);

        int ret3 = fep_immune_bridge_disconnect(bridge);
        EXPECT_EQ(ret3, 0);
    }

    fep_immune_bridge_destroy(bridge);
}

/* ============================================================================
 * Learning Bridge Stability Tests
 * ============================================================================ */

TEST_F(FEPStabilityRegressionTest, LearningBridgeConvergenceStability) {
    fep_learning_config_t config;
    fep_learning_default_config(&config);
    config.transition_learning_rate = 0.01f;
    config.likelihood_learning_rate = 0.01f;

    fep_learning_system_t* learning = fep_learning_create(&config);
    ASSERT_NE(learning, nullptr);

    fep_learning_connect_fep(learning, fep);

    // Train over many iterations
    float observations[OBS_DIM];
    float prev_states[8] = {0.5f, 0.5f, 0.3f, 0.3f, 0.2f, 0.2f, 0.1f, 0.1f};
    float curr_states[8] = {0.6f, 0.6f, 0.4f, 0.4f, 0.3f, 0.3f, 0.2f, 0.2f};

    for (int i = 0; i < OBS_DIM; i++) {
        observations[i] = 0.5f + 0.1f * (i % 3);
    }

    std::vector<float> loss_history;
    for (int i = 0; i < ITERATIONS; i++) {
        // Update transition matrix
        fep_learning_update_transition(learning, curr_states, 8, prev_states, 8);

        // Update likelihood matrix
        fep_learning_update_likelihood(learning, observations, OBS_DIM, curr_states, 8);

        // Track loss
        float loss;
        int ret = fep_learning_get_loss(learning, &loss);
        EXPECT_EQ(ret, 0);
        loss_history.push_back(loss);
    }

    // Loss should generally decrease (may have some noise)
    EXPECT_LT(loss_history.back(), loss_history[10]);

    fep_learning_destroy(learning);
}

TEST_F(FEPStabilityRegressionTest, LearningBridgeBatchUpdateStability) {
    fep_learning_config_t config;
    fep_learning_default_config(&config);

    fep_learning_system_t* learning = fep_learning_create(&config);
    ASSERT_NE(learning, nullptr);

    fep_learning_connect_fep(learning, fep);

    // Batch updates with various batch sizes
    const int batch_sizes[] = {1, 5, 10, 50, 100};
    for (int batch_size : batch_sizes) {
        for (int iter = 0; iter < 10; iter++) {
            // Simulate batch update
            int ret = fep_learning_step_batch(learning, batch_size);
            EXPECT_EQ(ret, 0);
        }
    }

    fep_learning_destroy(learning);
}

/* ============================================================================
 * Context Bridge Stability Tests
 * ============================================================================ */

TEST_F(FEPStabilityRegressionTest, ContextBridgeRapidSwitchingStability) {
    fep_context_config_t config;
    fep_context_default_config(&config);

    fep_context_system_t* context = fep_context_create(&config);
    ASSERT_NE(context, nullptr);

    fep_context_connect_fep(context, fep);

    // Add multiple contexts
    uint32_t ctx_ids[5];
    for (int i = 0; i < 5; i++) {
        char name[32];
        snprintf(name, sizeof(name), "context_%d", i);
        int ret = fep_context_add_context(context, name, &ctx_ids[i]);
        EXPECT_EQ(ret, 0);
    }

    // Rapid context switching
    for (int i = 0; i < ITERATIONS; i++) {
        uint32_t target_ctx = ctx_ids[i % 5];
        int ret = fep_context_switch_to(context, target_ctx);
        EXPECT_EQ(ret, 0);
    }

    // Verify system is still responsive
    uint32_t active_ctx;
    int ret = fep_context_get_active(context, &active_ctx);
    EXPECT_EQ(ret, 0);

    fep_context_destroy(context);
}

TEST_F(FEPStabilityRegressionTest, ContextBridgeInferenceStability) {
    fep_context_config_t config;
    fep_context_default_config(&config);

    fep_context_system_t* context = fep_context_create(&config);
    ASSERT_NE(context, nullptr);

    fep_context_connect_fep(context, fep);

    // Add contexts
    uint32_t ctx1, ctx2;
    fep_context_add_context(context, "context_1", &ctx1);
    fep_context_add_context(context, "context_2", &ctx2);

    // Repeated context inference
    float observations[OBS_DIM];
    for (int iter = 0; iter < ITERATIONS; iter++) {
        // Vary observations
        for (int i = 0; i < OBS_DIM; i++) {
            observations[i] = 0.5f + 0.3f * sinf(iter * 0.01f + i);
        }

        uint32_t inferred_ctx;
        float confidence;
        int ret = fep_context_infer(context, observations, OBS_DIM, &inferred_ctx, &confidence);
        EXPECT_EQ(ret, 0);
        EXPECT_GE(confidence, 0.0f);
        EXPECT_LE(confidence, 1.0f);
    }

    fep_context_destroy(context);
}

/* ============================================================================
 * Neuromodulation Bridge Stability Tests
 * ============================================================================ */

TEST_F(FEPStabilityRegressionTest, NeuromodBridgeLongRunningStability) {
    fep_neuromod_config_t config;
    fep_neuromod_default_config(&config);

    fep_neuromod_system_t* neuromod = fep_neuromod_create(&config);
    ASSERT_NE(neuromod, nullptr);

    fep_neuromod_connect_fep(neuromod, fep);

    // Long-running modulation simulation
    for (int i = 0; i < ITERATIONS; i++) {
        // Update neuromodulation state
        int ret = fep_neuromod_update(neuromod, TIMESTEP_MS);
        EXPECT_EQ(ret, 0);

        // Periodically apply modulation
        if (i % 10 == 0) {
            fep_neuromod_apply_modulation(neuromod);
        }
    }

    // Check modulation levels are in valid ranges
    fep_neuromod_state_t state;
    int ret = fep_neuromod_get_state(neuromod, &state);
    EXPECT_EQ(ret, 0);
    EXPECT_GE(state.dopamine_level, 0.0f);
    EXPECT_LE(state.dopamine_level, 1.0f);

    fep_neuromod_destroy(neuromod);
}

TEST_F(FEPStabilityRegressionTest, NeuromodBridgeExtremeModulationStability) {
    fep_neuromod_config_t config;
    fep_neuromod_default_config(&config);

    fep_neuromod_system_t* neuromod = fep_neuromod_create(&config);
    ASSERT_NE(neuromod, nullptr);

    fep_neuromod_connect_fep(neuromod, fep);

    // Test extreme modulation levels
    const float extreme_levels[] = {0.0f, 0.1f, 0.5f, 0.9f, 1.0f};

    for (float level : extreme_levels) {
        // Set dopamine
        int ret1 = fep_neuromod_set_dopamine(neuromod, level);
        EXPECT_EQ(ret1, 0);

        // Set serotonin
        int ret2 = fep_neuromod_set_serotonin(neuromod, 1.0f - level);
        EXPECT_EQ(ret2, 0);

        // Apply modulation
        int ret3 = fep_neuromod_apply_modulation(neuromod);
        EXPECT_EQ(ret3, 0);

        // Update
        fep_neuromod_update(neuromod, TIMESTEP_MS);
    }

    fep_neuromod_destroy(neuromod);
}

/* ============================================================================
 * Sleep Bridge Stability Tests
 * ============================================================================ */

TEST_F(FEPStabilityRegressionTest, SleepBridgeCycleStability) {
    fep_sleep_config_t config;
    fep_sleep_default_config(&config);

    fep_sleep_system_t* sleep = fep_sleep_create(&config);
    ASSERT_NE(sleep, nullptr);

    fep_sleep_connect_fep(sleep, fep);

    // Simulate multiple sleep/wake cycles
    for (int cycle = 0; cycle < 10; cycle++) {
        // Enter sleep
        int ret1 = fep_sleep_enter_sleep(sleep);
        EXPECT_EQ(ret1, 0);

        // Sleep for "some time"
        for (int i = 0; i < 100; i++) {
            fep_sleep_update(sleep, TIMESTEP_MS);
        }

        // Wake up
        int ret2 = fep_sleep_wake_up(sleep);
        EXPECT_EQ(ret2, 0);

        // Awake for "some time"
        for (int i = 0; i < 100; i++) {
            fep_sleep_update(sleep, TIMESTEP_MS);
        }
    }

    fep_sleep_destroy(sleep);
}

TEST_F(FEPStabilityRegressionTest, SleepBridgeConsolidationStability) {
    fep_sleep_config_t config;
    fep_sleep_default_config(&config);
    config.enable_memory_consolidation = true;

    fep_sleep_system_t* sleep = fep_sleep_create(&config);
    ASSERT_NE(sleep, nullptr);

    fep_sleep_connect_fep(sleep, fep);

    // Repeated consolidation cycles
    fep_sleep_enter_sleep(sleep);

    for (int i = 0; i < ITERATIONS; i++) {
        int ret = fep_sleep_consolidate_memories(sleep);
        EXPECT_EQ(ret, 0);

        fep_sleep_update(sleep, TIMESTEP_MS);
    }

    fep_sleep_wake_up(sleep);
    fep_sleep_destroy(sleep);
}

/* ============================================================================
 * Multi-Bridge Stability Tests
 * ============================================================================ */

TEST_F(FEPStabilityRegressionTest, MultiBridgeConcurrentOperations) {
    // Create multiple bridges
    fep_immune_bridge_t* immune_bridge = fep_immune_bridge_create(nullptr);
    fep_learning_system_t* learning = fep_learning_create(nullptr);
    fep_neuromod_system_t* neuromod = fep_neuromod_create(nullptr);

    ASSERT_NE(immune_bridge, nullptr);
    ASSERT_NE(learning, nullptr);
    ASSERT_NE(neuromod, nullptr);

    // Connect all
    fep_immune_bridge_connect_fep(immune_bridge, fep);
    fep_immune_bridge_connect_immune(immune_bridge, immune);
    fep_learning_connect_fep(learning, fep);
    fep_neuromod_connect_fep(neuromod, fep);

    // Concurrent updates
    for (int i = 0; i < 500; i++) {
        fep_immune_bridge_update(immune_bridge, TIMESTEP_MS);
        fep_neuromod_update(neuromod, TIMESTEP_MS);

        if (i % 10 == 0) {
            fep_neuromod_apply_modulation(neuromod);
        }
    }

    // All should still be operational
    fep_immune_state_t immune_state;
    EXPECT_EQ(fep_immune_bridge_get_state(immune_bridge, &immune_state), 0);

    fep_neuromod_state_t neuromod_state;
    EXPECT_EQ(fep_neuromod_get_state(neuromod, &neuromod_state), 0);

    // Cleanup
    fep_immune_bridge_destroy(immune_bridge);
    fep_learning_destroy(learning);
    fep_neuromod_destroy(neuromod);
}

TEST_F(FEPStabilityRegressionTest, MultiBridgeMemorySafety) {
    // Create and destroy bridges repeatedly
    for (int i = 0; i < 100; i++) {
        fep_immune_bridge_t* immune_bridge = fep_immune_bridge_create(nullptr);
        fep_learning_system_t* learning = fep_learning_create(nullptr);
        fep_context_system_t* context = fep_context_create(nullptr);

        ASSERT_NE(immune_bridge, nullptr);
        ASSERT_NE(learning, nullptr);
        ASSERT_NE(context, nullptr);

        // Do some operations
        fep_immune_bridge_connect_fep(immune_bridge, fep);
        fep_learning_connect_fep(learning, fep);
        fep_context_connect_fep(context, fep);

        fep_immune_bridge_update(immune_bridge, TIMESTEP_MS);

        // Cleanup
        fep_immune_bridge_destroy(immune_bridge);
        fep_learning_destroy(learning);
        fep_context_destroy(context);
    }
    // No memory leaks expected
}
