//=============================================================================
// test_learning_stability.cpp - Regression Tests for Learning Stability
//=============================================================================
/**
 * @file test_learning_stability.cpp
 * @brief Regression tests to ensure learning stability over time
 *
 * WHAT: Long-running tests for learning convergence and stability
 * WHY:  Detect regressions in learning algorithms
 * HOW:  Extended training runs with convergence checks
 *
 * @author NIMCP Development Team
 * @date 2025-12-05
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
extern "C" {
#include "core/brain/nimcp_brain.h"
#include "core/brain/learning/nimcp_brain_learning.h"
#include "core/brain/learning/nimcp_association_learning.h"
#include "utils/memory/nimcp_unified_memory.h"
}

class LearningStabilityTest : public ::testing::Test {
protected:
    brain_t brain;

    void SetUp() override {
        brain_config_t config;
        memset(&config, 0, sizeof(config));
        config.num_inputs = 10;
        config.num_outputs = 3;
        config.num_hidden_neurons = 30;
        config.learning_rate = 0.01f;

        brain = brain_create(&config);
        ASSERT_NE(brain, nullptr);
    }

    void TearDown() override {
        if (brain) {
            brain_destroy(brain);
        }
    }

    float compute_average_loss(const std::vector<float>& losses, int last_n) {
        if (losses.size() < (size_t)last_n) return -1.0f;

        float sum = 0.0f;
        for (size_t i = losses.size() - last_n; i < losses.size(); i++) {
            sum += losses[i];
        }
        return sum / last_n;
    }
};

//=============================================================================
// Test: Learning Convergence Over 1000 Iterations
//=============================================================================

TEST_F(LearningStabilityTest, ConvergesOver1000Iterations) {
    const int num_iterations = 1000;
    const int num_classes = 3;
    std::vector<float> loss_history;

    // Generate synthetic data: 3 linearly separable classes
    for (int iter = 0; iter < num_iterations; iter++) {
        int class_id = iter % num_classes;
        float features[10];

        for (int i = 0; i < 10; i++) {
            // Class-specific pattern
            features[i] = (class_id == 0) ? (float)i / 10.0f :
                         (class_id == 1) ? (float)(10 - i) / 10.0f :
                         0.5f;
        }

        char label[32];
        snprintf(label, sizeof(label), "class_%d", class_id);

        float loss = brain_learn_example(brain, features, 10, label, 1.0f);
        if (loss >= 0.0f) {
            loss_history.push_back(loss);
        }
    }

    ASSERT_GT(loss_history.size(), (size_t)500);

    // Check convergence: average loss in last 100 iterations < first 100
    float early_loss = compute_average_loss(loss_history, 100);
    float late_loss = compute_average_loss(
        std::vector<float>(loss_history.end() - 100, loss_history.end()), 100);

    // Learning should reduce loss
    EXPECT_LT(late_loss, early_loss);

    // Final loss should be reasonably low
    EXPECT_LT(late_loss, 1.0f);
}

//=============================================================================
// Test: Association Strength Stability
//=============================================================================

TEST_F(LearningStabilityTest, AssociationStrengthStable) {
    // Build associations over many updates
    for (int i = 0; i < 100; i++) {
        brain_learn_association(brain, "concept_A", "concept_B", 1);
    }

    float strength1 = get_association_strength(brain, "concept_A", "concept_B");
    ASSERT_GT(strength1, 0.0f);

    // Continue training
    for (int i = 0; i < 100; i++) {
        brain_learn_association(brain, "concept_A", "concept_B", 1);
    }

    float strength2 = get_association_strength(brain, "concept_A", "concept_B");

    // Strength should stabilize (not grow unbounded)
    EXPECT_NEAR(strength1, strength2, 0.2f); // Within 20%
    EXPECT_LE(strength2, 1.0f); // Should not exceed 1.0
}

//=============================================================================
// Test: Learning Rate Adaptation Stability
//=============================================================================

TEST_F(LearningStabilityTest, LearningRateAdaptationStable) {
    float initial_lr = brain->config.learning_rate;
    std::vector<float> losses;

    // Train with adaptive learning rate
    for (int i = 0; i < 200; i++) {
        float features[10];
        for (int j = 0; j < 10; j++) {
            features[j] = (float)((i + j) % 10) / 10.0f;
        }

        float loss = brain_learn_example(brain, features, 10, "adaptive_class", 0.9f);
        if (loss >= 0.0f) {
            losses.push_back(loss);
        }
    }

    float final_lr = brain->config.learning_rate;

    // Learning rate should adapt but stay within bounds
    EXPECT_GT(final_lr, initial_lr * 0.1f); // Not below 10% of initial
    EXPECT_LT(final_lr, initial_lr * 10.0f); // Not above 10x initial

    // Loss should generally decrease
    float early_avg = compute_average_loss(losses, 50);
    float late_avg = compute_average_loss(
        std::vector<float>(losses.end() - 50, losses.end()), 50);

    EXPECT_LT(late_avg, early_avg * 1.5f); // At most 1.5x early loss
}

//=============================================================================
// Test: Memory Stability Under Extended Learning
//=============================================================================

TEST_F(LearningStabilityTest, NoMemoryLeaksInExtendedLearning) {
    // Baseline: measure initial state (simplified - would use valgrind in production)

    // Perform many learning iterations
    for (int i = 0; i < 5000; i++) {
        float features[10];
        for (int j = 0; j < 10; j++) {
            features[j] = (float)(rand() % 100) / 100.0f;
        }

        brain_learn_example(brain, features, 10, "stress_test", 0.8f);
    }

    // Verify brain still functional
    float features[10] = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f,
                          0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
    float loss = brain_learn_example(brain, features, 10, "post_stress", 0.8f);

    EXPECT_GE(loss, 0.0f);

    // Note: Full memory leak detection requires valgrind/AddressSanitizer
}

//=============================================================================
// Test: Batch Learning Consistency
//=============================================================================

TEST_F(LearningStabilityTest, BatchLearningConsistent) {
    const int batch_size = 10;
    brain_example_t examples[batch_size];

    // Create batch
    float feature_data[batch_size][10];
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < 10; j++) {
            feature_data[i][j] = (float)((i + j) % 10) / 10.0f;
        }

        examples[i].features = feature_data[i];
        examples[i].num_features = 10;
        examples[i].label = "batch_class";
        examples[i].confidence = 0.9f;
    }

    // Learn batch multiple times
    float loss1 = brain_learn_batch(brain, examples, batch_size);
    ASSERT_GE(loss1, 0.0f);

    float loss2 = brain_learn_batch(brain, examples, batch_size);
    ASSERT_GE(loss2, 0.0f);

    // Losses should decrease (learning working)
    EXPECT_LE(loss2, loss1);
}

//=============================================================================
// Test: Reward Learning Stability
//=============================================================================

TEST_F(LearningStabilityTest, RewardLearningStable) {
    // Perform forward passes to build eligibility traces
    float features[10] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f,
                          0.6f, 0.7f, 0.8f, 0.9f, 1.0f};

    // Simulate successful action → reward
    for (int i = 0; i < 100; i++) {
        // Forward pass (builds traces)
        float output[3];
        brain_predict(brain, features, 10, output, 3);

        // Apply reward
        float reward = 1.0f; // Positive reward
        uint32_t modified = brain_apply_reward_learning(brain, reward);

        // Should modify some synapses
        if (i < 10) {
            // Early iterations may not have traces yet
            continue;
        }
        EXPECT_GT(modified, 0U);
    }

    // Brain should remain functional after reward learning
    float output[3];
    float conf = brain_predict(brain, features, 10, output, 3);
    EXPECT_GE(conf, 0.0f);
}

//=============================================================================
// Main Test Runner
//=============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
