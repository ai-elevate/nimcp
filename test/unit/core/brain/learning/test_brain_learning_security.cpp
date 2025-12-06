//=============================================================================
// test_brain_learning_security.cpp - Security Tests for Brain Learning
//=============================================================================
/**
 * @file test_brain_learning_security.cpp
 * @brief Security validation tests for brain learning module
 *
 * WHAT: Tests adversarial input detection and injection prevention
 * WHY:  Ensure learning system is robust against attacks
 * HOW:  Inject malicious inputs and verify rejection
 *
 * @author NIMCP Development Team
 * @date 2025-12-05
 */

#include <gtest/gtest.h>
#include <cmath>
extern "C" {
#include "core/brain/nimcp_brain.h"
#include "core/brain/learning/nimcp_brain_learning.h"
#include "utils/memory/nimcp_unified_memory.h"
}

class BrainLearningSecurityTest : public ::testing::Test {
protected:
    brain_t brain;

    void SetUp() override {
        brain_config_t config;
        memset(&config, 0, sizeof(config));
        config.num_inputs = 10;
        config.num_outputs = 5;
        config.num_hidden_neurons = 20;
        config.learning_rate = 0.01f;

        brain = brain_create(&config);
        ASSERT_NE(brain, nullptr);
    }

    void TearDown() override {
        if (brain) {
            brain_destroy(brain);
        }
    }
};

//=============================================================================
// Test: NaN/Inf Detection
//=============================================================================

TEST_F(BrainLearningSecurityTest, RejectsNaNFeatures) {
    float features[10] = {0.1f, NAN, 0.3f, 0.4f, 0.5f,
                          0.6f, 0.7f, 0.8f, 0.9f, 1.0f};

    float loss = brain_learn_example(brain, features, 10, "valid_label", 0.8f);

    // Should reject input with NaN
    EXPECT_LT(loss, 0.0f);
}

TEST_F(BrainLearningSecurityTest, RejectsInfFeatures) {
    float features[10] = {0.1f, 0.2f, INFINITY, 0.4f, 0.5f,
                          0.6f, 0.7f, 0.8f, 0.9f, 1.0f};

    float loss = brain_learn_example(brain, features, 10, "valid_label", 0.8f);

    // Should reject input with Inf
    EXPECT_LT(loss, 0.0f);
}

//=============================================================================
// Test: Format String Attack Prevention
//=============================================================================

TEST_F(BrainLearningSecurityTest, RejectsFormatStringInLabel) {
    float features[10] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f,
                          0.6f, 0.7f, 0.8f, 0.9f, 1.0f};

    // Attempt format string attack via label
    float loss = brain_learn_example(brain, features, 10, "%s%n%x", 0.8f);

    // Should reject malicious label
    EXPECT_LT(loss, 0.0f);
}

//=============================================================================
// Test: SQL Injection Prevention
//=============================================================================

TEST_F(BrainLearningSecurityTest, RejectsSQLInjectionInLabel) {
    float features[10] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f,
                          0.6f, 0.7f, 0.8f, 0.9f, 1.0f};

    // Attempt SQL injection via label
    float loss = brain_learn_example(brain, features, 10, "'; DROP TABLE users--", 0.8f);

    // Should reject malicious label
    EXPECT_LT(loss, 0.0f);
}

//=============================================================================
// Test: Confidence Range Validation
//=============================================================================

TEST_F(BrainLearningSecurityTest, RejectsInvalidConfidence) {
    float features[10] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f,
                          0.6f, 0.7f, 0.8f, 0.9f, 1.0f};

    // Test confidence > 1.0
    float loss1 = brain_learn_example(brain, features, 10, "valid_label", 1.5f);
    EXPECT_LT(loss1, 0.0f);

    // Test confidence < 0.0
    float loss2 = brain_learn_example(brain, features, 10, "valid_label", -0.5f);
    EXPECT_LT(loss2, 0.0f);
}

//=============================================================================
// Test: Extreme Feature Values
//=============================================================================

TEST_F(BrainLearningSecurityTest, HandlesExtremeFeatureValues) {
    float features[10] = {1e10f, -1e10f, 0.3f, 0.4f, 0.5f,
                          0.6f, 0.7f, 0.8f, 0.9f, 1.0f};

    // Should accept but log warning for extreme values
    float loss = brain_learn_example(brain, features, 10, "valid_label", 0.8f);

    // Implementation logs warning but doesn't reject
    // This is debatable - could reject if too extreme
}

//=============================================================================
// Test: Label Length Validation
//=============================================================================

TEST_F(BrainLearningSecurityTest, RejectsTooLongLabel) {
    float features[10] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f,
                          0.6f, 0.7f, 0.8f, 0.9f, 1.0f};

    // Create label exceeding 256 characters
    char long_label[300];
    memset(long_label, 'A', 299);
    long_label[299] = '\0';

    float loss = brain_learn_example(brain, features, 10, long_label, 0.8f);

    // Should reject overly long label
    EXPECT_LT(loss, 0.0f);
}

TEST_F(BrainLearningSecurityTest, RejectsEmptyLabel) {
    float features[10] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f,
                          0.6f, 0.7f, 0.8f, 0.9f, 1.0f};

    float loss = brain_learn_example(brain, features, 10, "", 0.8f);

    // Should reject empty label
    EXPECT_LT(loss, 0.0f);
}

//=============================================================================
// Test: Valid Input Acceptance
//=============================================================================

TEST_F(BrainLearningSecurityTest, AcceptsValidInput) {
    float features[10] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f,
                          0.6f, 0.7f, 0.8f, 0.9f, 1.0f};

    float loss = brain_learn_example(brain, features, 10, "normal_label", 0.8f);

    // Should accept valid input
    EXPECT_GE(loss, 0.0f);
}

//=============================================================================
// Main Test Runner
//=============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
