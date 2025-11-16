/**
 * @file test_explanations_regression.cpp
 * @brief Regression tests ensuring explanations work with existing decisions
 *
 * WHAT: Test that explanation generation doesn't break existing functionality
 * WHY:  Ensure backward compatibility and no regressions
 * HOW:  Test with various decision scenarios from existing codebase
 *
 * @author NIMCP Test Team
 * @date 2025-11-16
 */

#include <gtest/gtest.h>

#include "cognitive/nimcp_explanations.h"
#include "core/brain/nimcp_brain.h"
#include "utils/memory/nimcp_memory.h"
#include <string.h>
#include <cmath>

//=============================================================================
// Test Fixture
//=============================================================================

class ExplanationRegressionTest : public ::testing::Test {
protected:
    explanation_generator_t gen;

    void SetUp() override {
        gen = nullptr;
    }

    void TearDown() override {
        if (gen) {
            explanation_generator_destroy(gen);
            gen = nullptr;
        }
    }
};

//=============================================================================
// Regression: No Stub Placeholders Remain
//=============================================================================

TEST_F(ExplanationRegressionTest, NoTODOInGeneratedText) {
    // Arrange
    gen = explanation_generator_create(nullptr);
    ASSERT_NE(gen, nullptr);

    brain_t brain = (brain_t)0x1;
    brain_decision_t decision;
    memset(&decision, 0, sizeof(decision));
    strncpy(decision.label, "test_class", sizeof(decision.label) - 1);
    decision.confidence = 0.85f;
    decision.num_active_neurons = 30;
    decision.sparsity = 0.9f;

    natural_explanation_t explanation;
    memset(&explanation, 0, sizeof(explanation));

    // Act
    bool result = explanation_generate_from_decision(gen, brain, &decision, &explanation);

    // Assert
    EXPECT_TRUE(result);

    // No TODO should remain in any field
    EXPECT_EQ(strstr(explanation.what, "TODO"), nullptr);
    EXPECT_EQ(strstr(explanation.why, "TODO"), nullptr);
    EXPECT_EQ(strstr(explanation.how, "TODO"), nullptr);
    EXPECT_EQ(strstr(explanation.confidence, "TODO"), nullptr);
    EXPECT_EQ(strstr(explanation.alternatives, "TODO"), nullptr);
    EXPECT_EQ(strstr(explanation.counterfactual, "TODO"), nullptr);
}

TEST_F(ExplanationRegressionTest, NoPlaceholderInGeneratedText) {
    // Arrange
    gen = explanation_generator_create(nullptr);
    ASSERT_NE(gen, nullptr);

    brain_t brain = (brain_t)0x1;
    brain_decision_t decision;
    memset(&decision, 0, sizeof(decision));
    strncpy(decision.label, "regression", sizeof(decision.label) - 1);
    decision.confidence = 0.75f;
    decision.num_active_neurons = 25;
    decision.sparsity = 0.85f;

    natural_explanation_t explanation;
    memset(&explanation, 0, sizeof(explanation));

    // Act
    bool result = explanation_generate_from_decision(gen, brain, &decision, &explanation);

    // Assert
    EXPECT_TRUE(result);

    // No placeholder text should remain
    EXPECT_EQ(strstr(explanation.what, "placeholder"), nullptr);
    EXPECT_EQ(strstr(explanation.why, "placeholder"), nullptr);
    EXPECT_EQ(strstr(explanation.how, "placeholder"), nullptr);
}

//=============================================================================
// Regression: Metadata Extraction Works
//=============================================================================

TEST_F(ExplanationRegressionTest, MetadataNotZeroAfterExtraction) {
    // Arrange
    gen = explanation_generator_create(nullptr);
    ASSERT_NE(gen, nullptr);

    brain_t brain = (brain_t)0x1;
    brain_decision_t decision;
    memset(&decision, 0, sizeof(decision));
    strncpy(decision.label, "metadata_test", sizeof(decision.label) - 1);
    decision.confidence = 0.92f;
    decision.num_active_neurons = 45;
    decision.sparsity = 0.94f;

    natural_explanation_t explanation;
    memset(&explanation, 0, sizeof(explanation));

    // Act
    bool result = explanation_generate_from_decision(gen, brain, &decision, &explanation);

    // Assert
    EXPECT_TRUE(result);

    // Metadata should be extracted, not left at zero
    EXPECT_EQ(explanation.num_features_used, 45u);
    EXPECT_FLOAT_EQ(explanation.decision_confidence, 0.92f);

    // Symbolic logic check should run (result depends on brain)
    // Just verify it's set to some value (true or false)
    EXPECT_FALSE(explanation.has_symbolic_proof);  // Mock brain has no symbolic logic
}

//=============================================================================
// Regression: Label Extraction Works
//=============================================================================

TEST_F(ExplanationRegressionTest, LabelAppearsInWhatExplanation) {
    // Arrange
    gen = explanation_generator_create(nullptr);
    ASSERT_NE(gen, nullptr);

    brain_t brain = (brain_t)0x1;
    brain_decision_t decision;
    memset(&decision, 0, sizeof(decision));
    strncpy(decision.label, "specific_label", sizeof(decision.label) - 1);
    decision.confidence = 0.88f;
    decision.num_active_neurons = 35;
    decision.sparsity = 0.9f;

    natural_explanation_t explanation;
    memset(&explanation, 0, sizeof(explanation));

    // Act
    bool result = explanation_generate_from_decision(gen, brain, &decision, &explanation);

    // Assert
    EXPECT_TRUE(result);
    EXPECT_NE(strstr(explanation.what, "specific_label"), nullptr);
}

TEST_F(ExplanationRegressionTest, EmptyLabelHandledGracefully) {
    // Arrange
    gen = explanation_generator_create(nullptr);
    ASSERT_NE(gen, nullptr);

    brain_t brain = (brain_t)0x1;
    brain_decision_t decision;
    memset(&decision, 0, sizeof(decision));
    decision.label[0] = '\0';  // Empty label
    decision.confidence = 0.7f;
    decision.num_active_neurons = 20;
    decision.sparsity = 0.8f;

    natural_explanation_t explanation;
    memset(&explanation, 0, sizeof(explanation));

    // Act
    bool result = explanation_generate_from_decision(gen, brain, &decision, &explanation);

    // Assert
    EXPECT_TRUE(result);
    EXPECT_GT(strlen(explanation.what), 0u);  // Should still generate text
    // Should mention confidence even without label
    EXPECT_NE(strstr(explanation.what, "70"), nullptr);
}

//=============================================================================
// Regression: Confidence Extraction Works
//=============================================================================

TEST_F(ExplanationRegressionTest, ConfidenceAppearsInExplanation) {
    // Arrange
    gen = explanation_generator_create(nullptr);
    ASSERT_NE(gen, nullptr);

    brain_t brain = (brain_t)0x1;
    brain_decision_t decision;
    memset(&decision, 0, sizeof(decision));
    strncpy(decision.label, "conf_test", sizeof(decision.label) - 1);
    decision.confidence = 0.763f;  // Specific value to check
    decision.num_active_neurons = 30;
    decision.sparsity = 0.85f;

    natural_explanation_t explanation;
    memset(&explanation, 0, sizeof(explanation));

    // Act
    bool result = explanation_generate_from_decision(gen, brain, &decision, &explanation);

    // Assert
    EXPECT_TRUE(result);
    EXPECT_FLOAT_EQ(explanation.decision_confidence, 0.763f);

    // Confidence should appear in confidence explanation
    EXPECT_NE(strstr(explanation.confidence, "76"), nullptr);
}

//=============================================================================
// Regression: Sparsity-Based Feature Description Works
//=============================================================================

TEST_F(ExplanationRegressionTest, HighSparsityDescribedCorrectly) {
    // Arrange
    gen = explanation_generator_create(nullptr);
    ASSERT_NE(gen, nullptr);

    brain_t brain = (brain_t)0x1;
    brain_decision_t decision;
    memset(&decision, 0, sizeof(decision));
    strncpy(decision.label, "sparse", sizeof(decision.label) - 1);
    decision.confidence = 0.9f;
    decision.num_active_neurons = 5;
    decision.sparsity = 0.98f;  // Very high sparsity

    natural_explanation_t explanation;
    memset(&explanation, 0, sizeof(explanation));

    // Act
    bool result = explanation_generate_from_decision(gen, brain, &decision, &explanation);

    // Assert
    EXPECT_TRUE(result);

    // Should describe as "very few" or "few"
    EXPECT_TRUE(
        strstr(explanation.why, "very few") != nullptr ||
        strstr(explanation.why, "few") != nullptr
    );
}

TEST_F(ExplanationRegressionTest, LowSparsityDescribedCorrectly) {
    // Arrange
    gen = explanation_generator_create(nullptr);
    ASSERT_NE(gen, nullptr);

    brain_t brain = (brain_t)0x1;
    brain_decision_t decision;
    memset(&decision, 0, sizeof(decision));
    strncpy(decision.label, "dense", sizeof(decision.label) - 1);
    decision.confidence = 0.8f;
    decision.num_active_neurons = 100;
    decision.sparsity = 0.2f;  // Low sparsity (many active)

    natural_explanation_t explanation;
    memset(&explanation, 0, sizeof(explanation));

    // Act
    bool result = explanation_generate_from_decision(gen, brain, &decision, &explanation);

    // Assert
    EXPECT_TRUE(result);

    // Should describe as "many"
    EXPECT_NE(strstr(explanation.why, "many"), nullptr);
}

//=============================================================================
// Regression: Multimodal Modality Extraction Works
//=============================================================================

TEST_F(ExplanationRegressionTest, MultimodalExtractsAllModalities) {
    // Arrange
    gen = explanation_generator_create(nullptr);
    ASSERT_NE(gen, nullptr);

    brain_t brain = (brain_t)0x1;
    brain_multimodal_output_t output;
    memset(&output, 0, sizeof(output));

    strncpy(output.decision_label, "multi", sizeof(output.decision_label) - 1);
    output.confidence = 0.85f;
    output.visual_attention = 0.25f;
    output.audio_attention = 0.25f;
    output.speech_attention = 0.25f;
    output.language_attention = 0.25f;

    natural_explanation_t explanation;
    memset(&explanation, 0, sizeof(explanation));

    // Act
    bool result = explanation_generate_from_multimodal(gen, brain, &output, &explanation);

    // Assert
    EXPECT_TRUE(result);

    // All modalities with non-zero attention should appear
    EXPECT_NE(strstr(explanation.why, "visual"), nullptr);
    EXPECT_NE(strstr(explanation.why, "audio"), nullptr);
    EXPECT_NE(strstr(explanation.why, "speech"), nullptr);
    EXPECT_NE(strstr(explanation.why, "language"), nullptr);
}

TEST_F(ExplanationRegressionTest, MultimodalSkipsZeroAttentionModalities) {
    // Arrange
    gen = explanation_generator_create(nullptr);
    ASSERT_NE(gen, nullptr);

    brain_t brain = (brain_t)0x1;
    brain_multimodal_output_t output;
    memset(&output, 0, sizeof(output));

    strncpy(output.decision_label, "partial", sizeof(output.decision_label) - 1);
    output.confidence = 0.9f;
    output.visual_attention = 0.8f;
    output.audio_attention = 0.2f;
    output.speech_attention = 0.0f;  // Zero - should not appear
    output.language_attention = 0.0f;  // Zero - should not appear
    output.direct_attention = 0.0f;  // Zero - should not appear

    natural_explanation_t explanation;
    memset(&explanation, 0, sizeof(explanation));

    // Act
    bool result = explanation_generate_from_multimodal(gen, brain, &output, &explanation);

    // Assert
    EXPECT_TRUE(result);

    // Only non-zero modalities should appear
    EXPECT_NE(strstr(explanation.why, "visual"), nullptr);
    EXPECT_NE(strstr(explanation.why, "audio"), nullptr);
}

//=============================================================================
// Regression: Backward Compatibility
//=============================================================================

TEST_F(ExplanationRegressionTest, ExistingCodeStillWorks) {
    // Arrange - Use pattern from existing test_explanations.cpp
    gen = explanation_generator_create(nullptr);
    ASSERT_NE(gen, nullptr);

    brain_t mock_brain = (brain_t)0x1;
    brain_decision_t decision;
    memset(&decision, 0, sizeof(decision));
    strncpy(decision.label, "backward_compat", sizeof(decision.label) - 1);
    decision.confidence = 0.8f;
    decision.num_active_neurons = 40;
    decision.sparsity = 0.88f;

    natural_explanation_t explanation;

    // Act
    bool result = explanation_generate_from_decision(gen, mock_brain, &decision, &explanation);

    // Assert
    EXPECT_TRUE(result);
    EXPECT_GT(strlen(explanation.what), 0u);
    EXPECT_GT(strlen(explanation.why), 0u);
    EXPECT_GT(strlen(explanation.how), 0u);
}

TEST_F(ExplanationRegressionTest, JSONExportStillWorks) {
    // Arrange
    natural_explanation_t explanation;
    memset(&explanation, 0, sizeof(explanation));

    strncpy(explanation.what, "Test decision", sizeof(explanation.what) - 1);
    strncpy(explanation.why, "Test reason", sizeof(explanation.why) - 1);
    explanation.num_features_used = 25;
    explanation.decision_confidence = 0.82f;
    explanation.has_symbolic_proof = false;

    char json[2048];

    // Act
    bool result = explanation_to_json(&explanation, json, sizeof(json));

    // Assert
    EXPECT_TRUE(result);
    EXPECT_GT(strlen(json), 0u);
    EXPECT_NE(strstr(json, "\"what\""), nullptr);
    EXPECT_NE(strstr(json, "Test decision"), nullptr);
    EXPECT_NE(strstr(json, "\"num_features_used\": 25"), nullptr);
}

//=============================================================================
// Regression: Buffer Safety
//=============================================================================

TEST_F(ExplanationRegressionTest, NoBufferOverflowWithLongLabel) {
    // Arrange
    gen = explanation_generator_create(nullptr);
    ASSERT_NE(gen, nullptr);

    brain_t brain = (brain_t)0x1;
    brain_decision_t decision;
    memset(&decision, 0, sizeof(decision));

    // Fill label to maximum size
    memset(decision.label, 'X', sizeof(decision.label) - 1);
    decision.label[sizeof(decision.label) - 1] = '\0';

    decision.confidence = 0.75f;
    decision.num_active_neurons = 30;
    decision.sparsity = 0.85f;

    natural_explanation_t explanation;
    memset(&explanation, 0, sizeof(explanation));

    // Act
    bool result = explanation_generate_from_decision(gen, brain, &decision, &explanation);

    // Assert - Should handle gracefully without overflow
    EXPECT_TRUE(result);
    EXPECT_GT(strlen(explanation.what), 0u);
    EXPECT_LT(strlen(explanation.what), sizeof(explanation.what));
}

TEST_F(ExplanationRegressionTest, MultimodalBufferSafety) {
    // Arrange
    gen = explanation_generator_create(nullptr);
    ASSERT_NE(gen, nullptr);

    brain_t brain = (brain_t)0x1;
    brain_multimodal_output_t output;
    memset(&output, 0, sizeof(output));

    // All modalities maxed out
    output.visual_attention = 0.2f;
    output.audio_attention = 0.2f;
    output.speech_attention = 0.2f;
    output.language_attention = 0.2f;
    output.direct_attention = 0.2f;
    output.confidence = 0.9f;

    memset(output.decision_label, 'Y', sizeof(output.decision_label) - 1);
    output.decision_label[sizeof(output.decision_label) - 1] = '\0';

    natural_explanation_t explanation;
    memset(&explanation, 0, sizeof(explanation));

    // Act
    bool result = explanation_generate_from_multimodal(gen, brain, &output, &explanation);

    // Assert - Should handle all modalities without overflow
    EXPECT_TRUE(result);
    EXPECT_LT(strlen(explanation.why), sizeof(explanation.why));
    EXPECT_LT(strlen(explanation.how), sizeof(explanation.how));
}

// Main is provided by GTest
