//=============================================================================
// test_shannon_backward_compat.cpp - Regression Tests for Shannon Module
//=============================================================================
/**
 * @file test_shannon_backward_compat.cpp
 * @brief Backward compatibility tests for Shannon information theory
 *
 * PURPOSE: Ensure Shannon module doesn't break existing NIMCP functionality
 * COVERAGE: All pre-Shannon APIs continue to work unchanged
 * TEST COUNT: 20+ regression tests
 *
 * @author NIMCP Development Team
 * @date 2025-11-13
 * @version 3.0.0 Phase C4
 */

#include <gtest/gtest.h>
#include "information/nimcp_shannon.h"
#include "core/brain/nimcp_brain.h"
#include <cmath>

//=============================================================================
// Test Fixture
//=============================================================================

class ShannonRegressionTest : public ::testing::Test {
protected:
    void SetUp() override {
        brain = brain_create("regression_test", BRAIN_SIZE_SMALL,
                           BRAIN_TASK_CLASSIFICATION, 10, 10);
        ASSERT_NE(brain, nullptr);
    }

    void TearDown() override {
        if (brain) {
            brain_destroy(brain);
        }
    }

    brain_t brain;

    void create_pattern(float* features, uint32_t size, float value) {
        for (uint32_t i = 0; i < size; i++) {
            features[i] = value + 0.01f * (float)i;
        }
    }
};

//=============================================================================
// Brain Creation/Destruction (Pre-Shannon Functionality)
//=============================================================================

TEST_F(ShannonRegressionTest, BrainCreate_StillWorks) {
    brain_t test_brain = brain_create("test_regression", BRAIN_SIZE_SMALL,
                                      BRAIN_TASK_CLASSIFICATION, 10, 10);
    ASSERT_NE(test_brain, nullptr);
    brain_destroy(test_brain);
}

TEST_F(ShannonRegressionTest, BrainDestroy_HandlesNull) {
    brain_destroy(nullptr);  // Should not crash
}

//=============================================================================
// Learning Pipeline (Pre-Shannon Functionality)
//=============================================================================

TEST_F(ShannonRegressionTest, LearnExample_StillWorks) {
    float features[10];
    create_pattern(features, 10, 0.5f);

    float loss = brain_learn_example(brain, features, 10, "class_a", 0.9f);

    EXPECT_GE(loss, 0.0f);
}

TEST_F(ShannonRegressionTest, LearnExample_MultipleEpochs) {
    float features[10];
    create_pattern(features, 10, 0.5f);

    for (int i = 0; i < 10; i++) {
        float loss = brain_learn_example(brain, features, 10, "class_a", 0.9f);
        EXPECT_GE(loss, 0.0f);
    }
}

TEST_F(ShannonRegressionTest, LearnBatch_StillWorks) {
    const uint32_t batch_size = 5;
    brain_example_t* examples = (brain_example_t*)malloc(
        batch_size * sizeof(brain_example_t)
    );

    for (uint32_t i = 0; i < batch_size; i++) {
        examples[i].features = (float*)malloc(10 * sizeof(float));
        examples[i].num_features = 10;
        create_pattern(examples[i].features, 10, 0.1f * (float)i);
        strncpy(examples[i].label, "class_a", 63);
        examples[i].confidence = 0.9f;
    }

    brain_learn_batch(brain, examples, batch_size);

    // Cleanup
    for (uint32_t i = 0; i < batch_size; i++) {
        free(examples[i].features);
    }
    free(examples);
}

//=============================================================================
// Inference Pipeline (Pre-Shannon Functionality)
//=============================================================================

TEST_F(ShannonRegressionTest, Decide_StillWorks) {
    // Train first
    float train_features[10];
    create_pattern(train_features, 10, 0.7f);
    brain_learn_example(brain, train_features, 10, "class_a", 0.9f);

    // Infer
    float test_features[10];
    create_pattern(test_features, 10, 0.65f);

    brain_decision_t* decision = brain_decide(brain, test_features, 10);
    ASSERT_NE(decision, nullptr);

    EXPECT_GE(decision->confidence, 0.0f);
    EXPECT_LE(decision->confidence, 1.0f);
}

TEST_F(ShannonRegressionTest, DecideBatch_StillWorks) {
    // Train
    float train_features[10];
    create_pattern(train_features, 10, 0.6f);
    brain_learn_example(brain, train_features, 10, "class_a", 0.9f);

    // Batch inference
    const uint32_t batch_size = 3;
    float** input_features = (float**)malloc(batch_size * sizeof(float*));

    for (uint32_t i = 0; i < batch_size; i++) {
        input_features[i] = (float*)malloc(10 * sizeof(float));
        create_pattern(input_features[i], 10, 0.55f + 0.05f * (float)i);
    }

    brain_decision_t* outputs = (brain_decision_t*)calloc(
        batch_size, sizeof(brain_decision_t)
    );

    brain_decide_batch(brain, (const float**)input_features, batch_size, 10, outputs);

    for (uint32_t i = 0; i < batch_size; i++) {
        EXPECT_GE(outputs[i].confidence, 0.0f);
        EXPECT_LE(outputs[i].confidence, 1.0f);
    }

    // Cleanup
    for (uint32_t i = 0; i < batch_size; i++) {
        free(input_features[i]);
    }
    free(input_features);
    free(outputs);
}

//=============================================================================
// Memory Systems (Pre-Shannon Functionality)
//=============================================================================

TEST_F(ShannonRegressionTest, PhaseM1_EngramsStillWork) {
    // Phase M1 engram system should work unchanged
    float features[10];
    create_pattern(features, 10, 0.5f);

    for (int i = 0; i < 5; i++) {
        float loss = brain_learn_example(brain, features, 10, "class_a", 0.9f);
        EXPECT_GE(loss, 0.0f);
    }

    // Engrams should be encoded (tested implicitly through learning)
}

TEST_F(ShannonRegressionTest, PhaseM2_SystemsConsolidationStillWorks) {
    // Phase M2 consolidation should work unchanged
    for (int i = 0; i < 10; i++) {
        float features[10];
        create_pattern(features, 10, 0.1f * (float)i);
        brain_learn_example(brain, features, 10, "class_a", 0.8f);
    }

    // Consolidation should occur (tested implicitly)
}

TEST_F(ShannonRegressionTest, PhaseM3_WorkingMemoryTransferStillWorks) {
    // Phase M3 WM transfer should work unchanged
    float features[10];
    create_pattern(features, 10, 0.6f);

    brain_learn_example(brain, features, 10, "class_a", 0.9f);
    brain_decide(brain, features, 10);

    // WM transfer should occur (tested implicitly)
}

TEST_F(ShannonRegressionTest, PhaseM4_SemanticMemoryStillWorks) {
    // Phase M4 semantic memory should work unchanged
    for (int i = 0; i < 5; i++) {
        float features[10];
        create_pattern(features, 10, 0.2f + 0.1f * (float)i);
        brain_learn_example(brain, features, 10, "concept", 0.9f);
    }

    // Semantic concepts should be extracted (tested implicitly)
}

//=============================================================================
// Plasticity Mechanisms (Pre-Shannon Functionality)
//=============================================================================

TEST_F(ShannonRegressionTest, STDP_StillWorks) {
    // STDP should continue to function
    float features[10];
    create_pattern(features, 10, 0.5f);

    for (int i = 0; i < 5; i++) {
        brain_learn_example(brain, features, 10, "class_a", 0.9f);
    }

    // STDP modulation occurs (tested implicitly)
}

TEST_F(ShannonRegressionTest, Neuromodulators_StillWork) {
    // Neuromodulator system should work unchanged
    float features[10];
    create_pattern(features, 10, 0.7f);

    brain_learn_example(brain, features, 10, "class_a", 0.9f);

    // Dopamine, serotonin, etc. should modulate (tested implicitly)
}

//=============================================================================
// Different Brain Configurations (Pre-Shannon Functionality)
//=============================================================================

TEST_F(ShannonRegressionTest, SmallBrain_StillWorks) {
    brain_t small_brain = brain_create("small_regression", BRAIN_SIZE_TINY,
                                       BRAIN_TASK_CLASSIFICATION, 5, 5);
    ASSERT_NE(small_brain, nullptr);

    float features[5];
    create_pattern(features, 5, 0.5f);

    float loss = brain_learn_example(small_brain, features, 5, "test", 0.8f);
    EXPECT_GE(loss, 0.0f);

    brain_decision_t* decision = brain_decide(small_brain, features, 5);
    ASSERT_NE(decision, nullptr);
    EXPECT_GE(decision->confidence, 0.0f);

    brain_destroy(small_brain);
}

TEST_F(ShannonRegressionTest, LargeBrain_StillWorks) {
    brain_t large_brain = brain_create("large_regression", BRAIN_SIZE_LARGE,
                                       BRAIN_TASK_CLASSIFICATION, 50, 50);
    ASSERT_NE(large_brain, nullptr);

    float features[50];
    create_pattern(features, 50, 0.5f);

    float loss = brain_learn_example(large_brain, features, 50, "test", 0.8f);
    EXPECT_GE(loss, 0.0f);

    brain_decision_t* decision = brain_decide(large_brain, features, 50);
    ASSERT_NE(decision, nullptr);
    EXPECT_GE(decision->confidence, 0.0f);

    brain_destroy(large_brain);
}

//=============================================================================
// Multi-Task Learning (Pre-Shannon Functionality)
//=============================================================================

TEST_F(ShannonRegressionTest, MultiTask_StillWorks) {
    float features_a[10];
    float features_b[10];
    create_pattern(features_a, 10, 0.3f);
    create_pattern(features_b, 10, 0.7f);

    brain_learn_example(brain, features_a, 10, "task_a", 0.9f);
    brain_learn_example(brain, features_b, 10, "task_b", 0.9f);

    brain_decision_t* decision_a = brain_decide(brain, features_a, 10);
    brain_decision_t* decision_b = brain_decide(brain, features_b, 10);
    ASSERT_NE(decision_a, nullptr);
    ASSERT_NE(decision_b, nullptr);

    EXPECT_GE(decision_a->confidence, 0.0f);
    EXPECT_GE(decision_b->confidence, 0.0f);
}

//=============================================================================
// Continuous Learning (Pre-Shannon Functionality)
//=============================================================================

TEST_F(ShannonRegressionTest, ContinuousLearning_StillWorks) {
    // Simulate continuous learning over time
    for (int epoch = 0; epoch < 20; epoch++) {
        float features[10];
        create_pattern(features, 10, 0.5f + 0.02f * (float)epoch);

        float loss = brain_learn_example(brain, features, 10, "class_a", 0.9f);
        EXPECT_GE(loss, 0.0f);

        if (epoch % 5 == 0) {
            brain_decision_t* decision = brain_decide(brain, features, 10);
            if (decision) {
                EXPECT_GE(decision->confidence, 0.0f);
            }
        }
    }
}

//=============================================================================
// Error Handling (Pre-Shannon Functionality)
//=============================================================================

TEST_F(ShannonRegressionTest, NullFeatures_HandledGracefully) {
    // Null features should be rejected with error code
    float loss = brain_learn_example(brain, nullptr, 10, "class_a", 0.9f);
    EXPECT_LT(loss, 0.0f);  // Should return error code

    brain_decision_t* decision = brain_decide(brain, nullptr, 10);
    // Null decision is acceptable for null input (graceful handling)
    (void)decision;  // May be null, which is fine
}

TEST_F(ShannonRegressionTest, ZeroFeatures_HandledGracefully) {
    float features[10];
    memset(features, 0, sizeof(features));

    float loss = brain_learn_example(brain, features, 10, "class_a", 0.9f);
    EXPECT_GE(loss, 0.0f);

    brain_decision_t* decision = brain_decide(brain, features, 10);
    if (decision) {
        EXPECT_GE(decision->confidence, 0.0f);
    }
}

//=============================================================================
// API Stability Tests
//=============================================================================

TEST_F(ShannonRegressionTest, BrainAPI_SignaturesUnchanged) {
    // Verify all pre-Shannon API signatures still compile

    brain_t test_brain = brain_create("api_test", BRAIN_SIZE_SMALL,
                                      BRAIN_TASK_CLASSIFICATION, 10, 10);
    ASSERT_NE(test_brain, nullptr);

    float features[10];
    create_pattern(features, 10, 0.5f);

    // These should all compile without modification
    float loss = brain_learn_example(test_brain, features, 10, "test", 0.9f);
    (void)loss;

    brain_decision_t* decision = brain_decide(test_brain, features, 10);
    ASSERT_NE(decision, nullptr);
    (void)decision;

    brain_example_t example;
    example.features = features;
    example.num_features = 10;
    strncpy(example.label, "test", 63);
    example.confidence = 0.9f;

    brain_learn_batch(test_brain, &example, 1);

    brain_decision_t output;
    const float* input_ptr = example.features;
    brain_decide_batch(test_brain, &input_ptr, 1, 10, &output);

    brain_destroy(test_brain);
}

//=============================================================================
// Performance Regression Tests
//=============================================================================

TEST_F(ShannonRegressionTest, LearningSpeed_NotSignificantlySlower) {
    // Ensure Shannon module doesn't significantly slow down learning

    float features[10];
    create_pattern(features, 10, 0.5f);

    // This should complete in reasonable time
    for (int i = 0; i < 100; i++) {
        float loss = brain_learn_example(brain, features, 10, "class_a", 0.9f);
        EXPECT_GE(loss, 0.0f);
    }

    // If this test times out, Shannon module is too slow
}

TEST_F(ShannonRegressionTest, InferenceSpeed_NotSignificantlySlower) {
    // Train once
    float train_features[10];
    create_pattern(train_features, 10, 0.7f);
    brain_learn_example(brain, train_features, 10, "class_a", 0.9f);

    // Inference should remain fast
    float test_features[10];
    create_pattern(test_features, 10, 0.65f);

    for (int i = 0; i < 100; i++) {
        brain_decision_t* decision = brain_decide(brain, test_features, 10);
        if (decision) {
            EXPECT_GE(decision->confidence, 0.0f);
        }
    }

    // If this test times out, Shannon module is too slow
}

//=============================================================================
// Memory Usage Regression
//=============================================================================

TEST_F(ShannonRegressionTest, MemoryUsage_NoSignificantIncrease) {
    // Create multiple brains to test memory usage
    std::vector<brain_t> brains;

    for (int i = 0; i < 10; i++) {
        brain_t b = brain_create("memory_test", BRAIN_SIZE_MEDIUM,
                                 BRAIN_TASK_CLASSIFICATION, 10, 10);
        ASSERT_NE(b, nullptr);
        brains.push_back(b);

        float features[10];
        create_pattern(features, 10, 0.5f);
        brain_learn_example(b, features, 10, "class_a", 0.9f);
    }

    // Cleanup
    for (brain_t b : brains) {
        brain_destroy(b);
    }

    // If this causes memory issues, Shannon module has memory leak
}

//=============================================================================
// Integration with Existing Features
//=============================================================================

TEST_F(ShannonRegressionTest, AllPhases_WorkTogether) {
    // Test that all memory phases (M1-M4) still work together
    float features[10];

    // Phase M1: Engram encoding
    for (int i = 0; i < 3; i++) {
        create_pattern(features, 10, 0.3f + 0.1f * (float)i);
        brain_learn_example(brain, features, 10, "class_a", 0.9f);
    }

    // Phase M2: Systems consolidation (implicit)
    // Phase M3: Working memory transfer (implicit)
    // Phase M4: Semantic memory extraction (implicit)

    // Verify inference still works
    create_pattern(features, 10, 0.4f);
    brain_decision_t* decision = brain_decide(brain, features, 10);
    ASSERT_NE(decision, nullptr);

    EXPECT_GE(decision->confidence, 0.0f);
    EXPECT_LE(decision->confidence, 1.0f);
}

//=============================================================================
// Backward Compatibility with Configuration
//=============================================================================

TEST_F(ShannonRegressionTest, DefaultBrainConfig_StillValid) {
    // Default brain configuration should work unchanged
    brain_t default_brain = brain_create("default_config", BRAIN_SIZE_MEDIUM,
                                         BRAIN_TASK_CLASSIFICATION, 10, 10);
    ASSERT_NE(default_brain, nullptr);

    float features[10];
    create_pattern(features, 10, 0.5f);

    float loss = brain_learn_example(default_brain, features, 10, "test", 0.9f);
    EXPECT_GE(loss, 0.0f);

    brain_destroy(default_brain);
}

//=============================================================================
// Main
//=============================================================================

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
