/**
 * @file test_learning_stability.cpp
 * @brief Regression tests for learning stability with bio-async integration
 *
 * These tests ensure that the integration of bio-async messaging doesn't
 * destabilize learning, and that plasticity rules continue to converge
 * properly.
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>

extern "C" {
#include "plasticity/stdp/nimcp_stdp.h"
#include "plasticity/homeostatic/nimcp_homeostatic.h"
#include "plasticity/bcm/nimcp_bcm.h"
#include "plasticity/predictive/nimcp_predictive_coding.h"
#include "plasticity/adaptive/nimcp_adaptive.h"
#include "plasticity/neuromodulators/nimcp_neuromodulators.h"
#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_router.h"
#include "utils/memory/nimcp_unified_memory.h"
}

class LearningStabilityTest : public ::testing::Test {
protected:
    void SetUp() override {
        bio_async_init();
        bio_router_init();
        nimcp_unified_memory_init();
    }

    void TearDown() override {
        bio_router_shutdown();
        bio_async_shutdown();
        nimcp_unified_memory_shutdown();
    }

    float ComputeVariance(const std::vector<float>& values) {
        if (values.empty()) return 0.0f;

        float mean = 0.0f;
        for (float v : values) mean += v;
        mean /= values.size();

        float variance = 0.0f;
        for (float v : values) {
            float diff = v - mean;
            variance += diff * diff;
        }
        return variance / values.size();
    }

    bool IsStable(const std::vector<float>& trajectory, float tolerance = 0.01f) {
        if (trajectory.size() < 100) return false;

        // Check last 20% of trajectory for stability
        size_t start = trajectory.size() * 4 / 5;
        std::vector<float> tail(trajectory.begin() + start, trajectory.end());

        float variance = ComputeVariance(tail);
        return variance < tolerance;
    }
};

TEST_F(LearningStabilityTest, STDPConvergence) {
    // Test that STDP weights converge to stable values
    stdp_config_t config = stdp_default_config();
    stdp_state_t* state = stdp_create(&config);
    ASSERT_NE(state, nullptr);

    std::vector<float> weight_trajectory;
    float initial_weight = 0.5f;
    float weight = initial_weight;

    // Simulate 1000 learning steps
    for (int i = 0; i < 1000; i++) {
        // Simulate correlated pre-post spike pairs
        float pre_time = i * 1.0f;
        float post_time = pre_time + 5.0f; // 5ms delay

        float delta_w = stdp_compute_weight_change(
            state, pre_time, post_time, weight, 1.0f);

        weight += delta_w;
        weight_trajectory.push_back(weight);

        // Process bio-async messages periodically
        if (i % 10 == 0) {
            bio_router_process_messages();
        }
    }

    stdp_destroy(state);

    // Check that weights converged
    EXPECT_TRUE(IsStable(weight_trajectory, 0.01f))
        << "STDP weights did not converge to stable values";

    // Check weights stayed in valid range
    float final_weight = weight_trajectory.back();
    EXPECT_GE(final_weight, 0.0f);
    EXPECT_LE(final_weight, 1.0f);
}

TEST_F(LearningStabilityTest, HomeostaticRegulation) {
    // Test that homeostatic plasticity maintains stable firing rates
    homeostatic_config_t config = homeostatic_default_config();
    homeostatic_state_t* state = homeostatic_create(&config);
    ASSERT_NE(state, nullptr);

    std::vector<float> rate_trajectory;
    float target_rate = 5.0f; // Target 5 Hz

    // Simulate 1000 updates
    for (int i = 0; i < 1000; i++) {
        // Simulate variable input rate
        float input_rate = 10.0f + 5.0f * sinf(i * 0.01f);

        homeostatic_update(state, input_rate, 0.01f);

        float current_rate = homeostatic_get_firing_rate(state);
        rate_trajectory.push_back(current_rate);

        if (i % 10 == 0) {
            bio_router_process_messages();
        }
    }

    homeostatic_destroy(state);

    // Check that firing rate stabilized near target
    EXPECT_TRUE(IsStable(rate_trajectory, 1.0f))
        << "Firing rate did not stabilize";

    float final_rate = rate_trajectory.back();
    EXPECT_NEAR(final_rate, target_rate, 2.0f)
        << "Final firing rate far from target";
}

TEST_F(LearningStabilityTest, BCMThresholdAdaptation) {
    // Test that BCM threshold adapts appropriately
    bcm_config_t config = bcm_default_config();
    bcm_state_t* state = bcm_create(&config);
    ASSERT_NE(state, nullptr);

    std::vector<float> threshold_trajectory;

    // Simulate learning with varying activity levels
    for (int i = 0; i < 1000; i++) {
        float pre_activity = 0.5f;
        float post_activity = 0.3f + 0.2f * sinf(i * 0.02f);
        float weight = 0.5f;

        bcm_update(state, pre_activity, post_activity, &weight, 0.01f);

        float threshold = bcm_get_threshold(state);
        threshold_trajectory.push_back(threshold);

        if (i % 10 == 0) {
            bio_router_process_messages();
        }
    }

    bcm_destroy(state);

    // Check threshold converged
    EXPECT_TRUE(IsStable(threshold_trajectory, 0.01f))
        << "BCM threshold did not converge";
}

TEST_F(LearningStabilityTest, PredictiveCodingConvergence) {
    // Test that predictive coding hierarchy converges to low free energy
    uint32_t units[] = {10, 5, 2};
    pc_hierarchy_config_t config = pc_hierarchy_config_default(3, units);
    pc_hierarchy_t hierarchy = pc_hierarchy_create(&config);
    ASSERT_NE(hierarchy, nullptr);

    std::vector<float> free_energy_trajectory;

    // Fixed input pattern
    float input[] = {0.5f, 0.3f, 0.8f, 0.1f, 0.6f, 0.4f, 0.9f, 0.2f, 0.7f, 0.0f};

    // Run inference for 500 steps
    for (int i = 0; i < 500; i++) {
        pc_hierarchy_set_input(hierarchy, input);
        pc_hierarchy_inference_step(hierarchy, 1.0f, true);

        float free_energy = pc_hierarchy_get_free_energy(hierarchy);
        free_energy_trajectory.push_back(free_energy);

        if (i % 10 == 0) {
            bio_router_process_messages();
        }
    }

    pc_hierarchy_destroy(hierarchy);

    // Check free energy decreased and stabilized
    float initial_fe = free_energy_trajectory[0];
    float final_fe = free_energy_trajectory.back();

    EXPECT_LT(final_fe, initial_fe)
        << "Free energy did not decrease during inference";

    EXPECT_TRUE(IsStable(free_energy_trajectory, 1.0f))
        << "Free energy did not converge";
}

TEST_F(LearningStabilityTest, NeuromodulatorDynamics) {
    // Test that neuromodulator concentrations remain stable
    neuromodulator_config_t config = neuromodulator_default_config();
    neuromodulator_state_t* state = neuromodulator_create(&config);
    ASSERT_NE(state, nullptr);

    std::vector<float> dopamine_trajectory;

    // Simulate reward/error signals
    for (int i = 0; i < 1000; i++) {
        float reward = (i % 100 < 50) ? 1.0f : 0.0f; // Periodic reward

        neuromodulator_update(state, reward, 0.0f, 0.0f, 0.01f);

        float dopamine = neuromodulator_get_concentration(state, NEUROMODULATOR_DOPAMINE);
        dopamine_trajectory.push_back(dopamine);

        if (i % 10 == 0) {
            bio_router_process_messages();
        }
    }

    neuromodulator_destroy(state);

    // Check dopamine stayed in physiological range
    for (float conc : dopamine_trajectory) {
        EXPECT_GE(conc, 0.0f);
        EXPECT_LE(conc, 1.0f);
    }

    // Check for oscillatory stability (not runaway)
    float variance = ComputeVariance(dopamine_trajectory);
    EXPECT_LT(variance, 0.5f)
        << "Dopamine concentration variance too high";
}

TEST_F(LearningStabilityTest, WeightBoundsRespected) {
    // Regression test: ensure weights stay in [0, 1] during learning
    stdp_config_t config = stdp_default_config();
    stdp_state_t* state = stdp_create(&config);
    ASSERT_NE(state, nullptr);

    float weight = 0.5f;

    // Extreme learning scenario
    for (int i = 0; i < 10000; i++) {
        // Very strong potentiation signal
        float delta_w = stdp_compute_weight_change(state, 0.0f, 1.0f, weight, 10.0f);
        weight += delta_w;

        // Weights should never go out of bounds
        ASSERT_GE(weight, 0.0f) << "Weight went below 0 at iteration " << i;
        ASSERT_LE(weight, 1.0f) << "Weight exceeded 1 at iteration " << i;

        if (i % 10 == 0) {
            bio_router_process_messages();
        }
    }

    stdp_destroy(state);
}

TEST_F(LearningStabilityTest, NoMemoryLeaks) {
    // Regression test: repeated create/destroy doesn't leak memory
    for (int i = 0; i < 100; i++) {
        stdp_config_t config = stdp_default_config();
        stdp_state_t* state = stdp_create(&config);
        ASSERT_NE(state, nullptr);

        // Do some work
        for (int j = 0; j < 10; j++) {
            stdp_compute_weight_change(state, j * 1.0f, j * 1.0f + 5.0f, 0.5f, 1.0f);
        }

        stdp_destroy(state);

        if (i % 10 == 0) {
            bio_router_process_messages();
        }
    }

    // If we get here without crashing, no obvious memory corruption
    SUCCEED();
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
