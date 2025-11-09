/**
 * @file test_explanations.cpp
 * @brief Unit tests for Natural Language Explanations (Phase 10.7)
 *
 * WHAT: Comprehensive tests for explanation generation
 * WHY:  Ensure explanations are correct, complete, and useful
 * HOW:  GTest framework with AAA pattern
 *
 * @author NIMCP Phase 10 Test Team
 * @date 2025-11-09
 */

#include <gtest/gtest.h>

extern "C" {
#include "cognitive/nimcp_explanations.h"
#include "core/brain/nimcp_brain.h"
#include "utils/memory/nimcp_memory.h"
#include <string.h>
}

//=============================================================================
// Test Fixture
//=============================================================================

class ExplanationsTest : public ::testing::Test {
protected:
    explanation_generator_t gen;
    brain_t mock_brain;
    brain_decision_t* mock_decision;

    void SetUp() override {
        gen = nullptr;
        mock_brain = nullptr;
        mock_decision = nullptr;
    }

    void TearDown() override {
        if (gen) {
            explanation_generator_destroy(gen);
            gen = nullptr;
        }
        // Note: mock_brain and mock_decision are not dynamically allocated in these tests
    }

    // Helper: Create mock decision (placeholder for real decision struct)
    brain_decision_t* create_mock_decision() {
        // In real implementation, would use brain_decide()
        // For now, return non-NULL placeholder pointer
        return (brain_decision_t*)0x1;
    }
};

//=============================================================================
// Creation & Destruction Tests
//=============================================================================

TEST_F(ExplanationsTest, CreateWithDefaults) {
    // Arrange & Act
    gen = explanation_generator_create(nullptr);

    // Assert
    ASSERT_NE(gen, nullptr);
}

TEST_F(ExplanationsTest, CreateWithCustomConfig) {
    // Arrange
    explanation_config_t config = explanation_default_config();
    config.generate_what = false;  // Disable "what" explanations
    config.max_alternatives = 5;

    // Act
    gen = explanation_generator_create(&config);

    // Assert
    ASSERT_NE(gen, nullptr);
}

TEST_F(ExplanationsTest, DestroyNull) {
    // Arrange & Act & Assert
    explanation_generator_destroy(nullptr);  // Should not crash
}

//=============================================================================
// Default Configuration Tests
//=============================================================================

TEST_F(ExplanationsTest, DefaultConfigAllEnabled) {
    // Arrange & Act
    explanation_config_t config = explanation_default_config();

    // Assert
    EXPECT_TRUE(config.generate_what);
    EXPECT_TRUE(config.generate_why);
    EXPECT_TRUE(config.generate_how);
    EXPECT_TRUE(config.generate_confidence);
    EXPECT_TRUE(config.generate_alternatives);
    EXPECT_TRUE(config.generate_counterfactuals);
    EXPECT_EQ(config.max_alternatives, 3u);
    EXPECT_FLOAT_EQ(config.min_alternative_prob, 0.05f);
}

//=============================================================================
// Explanation Generation Tests
//=============================================================================

TEST_F(ExplanationsTest, GenerateBasicExplanation) {
    // Arrange
    gen = explanation_generator_create(nullptr);
    ASSERT_NE(gen, nullptr);

    mock_brain = (brain_t)0x1;  // Placeholder
    mock_decision = create_mock_decision();

    natural_explanation_t explanation;

    // Act
    bool result = explanation_generate_from_decision(gen, mock_brain, mock_decision, &explanation);

    // Assert
    EXPECT_TRUE(result);
    // Explanation fields should be populated
    EXPECT_GT(strlen(explanation.what), 0u);
    EXPECT_GT(strlen(explanation.why), 0u);
    EXPECT_GT(strlen(explanation.how), 0u);
}

TEST_F(ExplanationsTest, ExplainDecisionWithNullGenerator) {
    // Arrange
    mock_brain = (brain_t)0x1;
    mock_decision = create_mock_decision();
    natural_explanation_t explanation;

    // Act
    bool result = explanation_generate_from_decision(nullptr, mock_brain, mock_decision, &explanation);

    // Assert
    EXPECT_FALSE(result);
}

TEST_F(ExplanationsTest, ExplainDecisionWithNullBrain) {
    // Arrange
    gen = explanation_generator_create(nullptr);
    mock_decision = create_mock_decision();
    natural_explanation_t explanation;

    // Act
    bool result = explanation_generate_from_decision(gen, nullptr, mock_decision, &explanation);

    // Assert
    EXPECT_FALSE(result);
}

TEST_F(ExplanationsTest, ExplainDecisionWithNullDecision) {
    // Arrange
    gen = explanation_generator_create(nullptr);
    mock_brain = (brain_t)0x1;
    natural_explanation_t explanation;

    // Act
    bool result = explanation_generate_from_decision(gen, mock_brain, nullptr, &explanation);

    // Assert
    EXPECT_FALSE(result);
}

TEST_F(ExplanationsTest, ExplainDecisionWithNullOutput) {
    // Arrange
    gen = explanation_generator_create(nullptr);
    mock_brain = (brain_t)0x1;
    mock_decision = create_mock_decision();

    // Act
    bool result = explanation_generate_from_decision(gen, mock_brain, mock_decision, nullptr);

    // Assert
    EXPECT_FALSE(result);
}

//=============================================================================
// Configuration-Driven Generation Tests
//=============================================================================

TEST_F(ExplanationsTest, GenerateOnlyWhat) {
    // Arrange
    explanation_config_t config = explanation_default_config();
    config.generate_what = true;
    config.generate_why = false;
    config.generate_how = false;
    config.generate_confidence = false;
    config.generate_alternatives = false;
    config.generate_counterfactuals = false;

    gen = explanation_generator_create(&config);
    ASSERT_NE(gen, nullptr);

    mock_brain = (brain_t)0x1;
    mock_decision = create_mock_decision();
    natural_explanation_t explanation;

    // Act
    bool result = explanation_generate_from_decision(gen, mock_brain, mock_decision, &explanation);

    // Assert
    EXPECT_TRUE(result);
    EXPECT_GT(strlen(explanation.what), 0u);
    EXPECT_EQ(strlen(explanation.why), 0u);  // Should be empty
    EXPECT_EQ(strlen(explanation.how), 0u);  // Should be empty
}

TEST_F(ExplanationsTest, GenerateOnlyWhy) {
    // Arrange
    explanation_config_t config = explanation_default_config();
    config.generate_what = false;
    config.generate_why = true;
    config.generate_how = false;
    config.generate_confidence = false;
    config.generate_alternatives = false;
    config.generate_counterfactuals = false;

    gen = explanation_generator_create(&config);
    mock_brain = (brain_t)0x1;
    mock_decision = create_mock_decision();
    natural_explanation_t explanation;

    // Act
    bool result = explanation_generate_from_decision(gen, mock_brain, mock_decision, &explanation);

    // Assert
    EXPECT_TRUE(result);
    EXPECT_EQ(strlen(explanation.what), 0u);  // Should be empty
    EXPECT_GT(strlen(explanation.why), 0u);
}

TEST_F(ExplanationsTest, GenerateOnlyHow) {
    // Arrange
    explanation_config_t config = explanation_default_config();
    config.generate_what = false;
    config.generate_why = false;
    config.generate_how = true;
    config.generate_confidence = false;
    config.generate_alternatives = false;
    config.generate_counterfactuals = false;

    gen = explanation_generator_create(&config);
    mock_brain = (brain_t)0x1;
    mock_decision = create_mock_decision();
    natural_explanation_t explanation;

    // Act
    bool result = explanation_generate_from_decision(gen, mock_brain, mock_decision, &explanation);

    // Assert
    EXPECT_TRUE(result);
    EXPECT_EQ(strlen(explanation.what), 0u);
    EXPECT_EQ(strlen(explanation.why), 0u);
    EXPECT_GT(strlen(explanation.how), 0u);
}

//=============================================================================
// Multimodal Explanation Tests
//=============================================================================

TEST_F(ExplanationsTest, ExplainMultimodalDecision) {
    // Arrange
    gen = explanation_generator_create(nullptr);
    ASSERT_NE(gen, nullptr);

    mock_brain = (brain_t)0x1;
    brain_multimodal_output_t* mock_output = (brain_multimodal_output_t*)0x1;
    natural_explanation_t explanation;

    // Act
    bool result = explanation_generate_from_multimodal(gen, mock_brain, mock_output, &explanation);

    // Assert
    EXPECT_TRUE(result);
    EXPECT_GT(strlen(explanation.what), 0u);
}

TEST_F(ExplanationsTest, ExplainMultimodalWithNullGenerator) {
    // Arrange
    mock_brain = (brain_t)0x1;
    brain_multimodal_output_t* mock_output = (brain_multimodal_output_t*)0x1;
    natural_explanation_t explanation;

    // Act
    bool result = explanation_generate_from_multimodal(nullptr, mock_brain, mock_output, &explanation);

    // Assert
    EXPECT_FALSE(result);
}

//=============================================================================
// Symbolic Logic Tests
//=============================================================================

TEST_F(ExplanationsTest, GenerateSymbolicProof) {
    // Arrange
    gen = explanation_generator_create(nullptr);
    mock_brain = (brain_t)0x1;
    mock_decision = create_mock_decision();
    char proof[512];

    // Act
    bool result = explain_with_symbolic_logic(gen, mock_brain, mock_decision, proof, sizeof(proof));

    // Assert
    EXPECT_TRUE(result);
    EXPECT_GT(strlen(proof), 0u);
}

TEST_F(ExplanationsTest, SymbolicProofWithNullBuffer) {
    // Arrange
    gen = explanation_generator_create(nullptr);
    mock_brain = (brain_t)0x1;
    mock_decision = create_mock_decision();

    // Act
    bool result = explain_with_symbolic_logic(gen, mock_brain, mock_decision, nullptr, 512);

    // Assert
    EXPECT_FALSE(result);
}

//=============================================================================
// Causal Chain Tests
//=============================================================================

TEST_F(ExplanationsTest, GenerateCausalChain) {
    // Arrange
    gen = explanation_generator_create(nullptr);
    mock_brain = (brain_t)0x1;
    const char* input_desc = "gray cat photo";
    const char* output_label = "cat";
    char causal_chain[512];

    // Act
    bool result = generate_causal_chain(gen, mock_brain, input_desc, output_label,
                                       causal_chain, sizeof(causal_chain));

    // Assert
    EXPECT_TRUE(result);
    EXPECT_GT(strlen(causal_chain), 0u);
    // Should contain input description and output label
    EXPECT_NE(strstr(causal_chain, input_desc), nullptr);
    EXPECT_NE(strstr(causal_chain, output_label), nullptr);
}

TEST_F(ExplanationsTest, CausalChainWithNullInput) {
    // Arrange
    gen = explanation_generator_create(nullptr);
    mock_brain = (brain_t)0x1;
    char causal_chain[512];

    // Act
    bool result = generate_causal_chain(gen, mock_brain, nullptr, "cat",
                                       causal_chain, sizeof(causal_chain));

    // Assert
    EXPECT_FALSE(result);
}

//=============================================================================
// Counterfactual Tests
//=============================================================================

TEST_F(ExplanationsTest, GenerateCounterfactual) {
    // Arrange
    gen = explanation_generator_create(nullptr);
    mock_brain = (brain_t)0x1;
    mock_decision = create_mock_decision();
    const char* feature = "ears";
    const char* modification = "floppy instead of pointy";
    char counterfactual[256];

    // Act
    bool result = generate_counterfactual(gen, mock_brain, mock_decision,
                                         feature, modification,
                                         counterfactual, sizeof(counterfactual));

    // Assert
    EXPECT_TRUE(result);
    EXPECT_GT(strlen(counterfactual), 0u);
    // Should mention the feature
    EXPECT_NE(strstr(counterfactual, feature), nullptr);
}

TEST_F(ExplanationsTest, CounterfactualWithNullFeature) {
    // Arrange
    gen = explanation_generator_create(nullptr);
    mock_brain = (brain_t)0x1;
    mock_decision = create_mock_decision();
    char counterfactual[256];

    // Act
    bool result = generate_counterfactual(gen, mock_brain, mock_decision,
                                         nullptr, "different",
                                         counterfactual, sizeof(counterfactual));

    // Assert
    EXPECT_FALSE(result);
}

//=============================================================================
// Utility Function Tests
//=============================================================================

TEST_F(ExplanationsTest, PrintExplanation) {
    // Arrange
    natural_explanation_t explanation;
    strncpy(explanation.what, "Decision: cat", sizeof(explanation.what));
    strncpy(explanation.why, "Because whiskers detected", sizeof(explanation.why));
    strncpy(explanation.how, "V1 → IT → PFC", sizeof(explanation.how));

    // Act & Assert (just verify it doesn't crash)
    explanation_print(&explanation);
}

TEST_F(ExplanationsTest, PrintNullExplanation) {
    // Act & Assert (should not crash)
    explanation_print(nullptr);
}

TEST_F(ExplanationsTest, ExplanationToJSON) {
    // Arrange
    natural_explanation_t explanation;
    memset(&explanation, 0, sizeof(explanation));
    strncpy(explanation.what, "Decision: cat", sizeof(explanation.what));
    strncpy(explanation.why, "Because whiskers", sizeof(explanation.why));
    explanation.decision_confidence = 0.87f;
    explanation.has_symbolic_proof = true;

    char json[2048];

    // Act
    bool result = explanation_to_json(&explanation, json, sizeof(json));

    // Assert
    EXPECT_TRUE(result);
    EXPECT_GT(strlen(json), 0u);
    // Should be valid JSON with key fields
    EXPECT_NE(strstr(json, "\"what\""), nullptr);
    EXPECT_NE(strstr(json, "\"why\""), nullptr);
    EXPECT_NE(strstr(json, "Decision: cat"), nullptr);
}

TEST_F(ExplanationsTest, JSONWithNullBuffer) {
    // Arrange
    natural_explanation_t explanation;
    memset(&explanation, 0, sizeof(explanation));

    // Act
    bool result = explanation_to_json(&explanation, nullptr, 2048);

    // Assert
    EXPECT_FALSE(result);
}

//=============================================================================
// Integration Tests
//=============================================================================

TEST_F(ExplanationsTest, CompleteWorkflow) {
    // Arrange
    gen = explanation_generator_create(nullptr);
    ASSERT_NE(gen, nullptr);

    mock_brain = (brain_t)0x1;
    mock_decision = create_mock_decision();
    natural_explanation_t explanation;

    // Act
    bool result = explanation_generate_from_decision(gen, mock_brain, mock_decision, &explanation);

    // Assert - Full explanation generated
    EXPECT_TRUE(result);
    EXPECT_GT(strlen(explanation.what), 0u);
    EXPECT_GT(strlen(explanation.why), 0u);
    EXPECT_GT(strlen(explanation.how), 0u);
    EXPECT_GT(strlen(explanation.confidence), 0u);

    // Can convert to JSON
    char json[2048];
    EXPECT_TRUE(explanation_to_json(&explanation, json, sizeof(json)));
    EXPECT_GT(strlen(json), 0u);
}

TEST_F(ExplanationsTest, MultipleExplanations) {
    // Arrange
    gen = explanation_generator_create(nullptr);
    mock_brain = (brain_t)0x1;
    mock_decision = create_mock_decision();

    // Act - Generate multiple explanations
    for (int i = 0; i < 10; i++) {
        natural_explanation_t explanation;
        bool result = explanation_generate_from_decision(gen, mock_brain, mock_decision, &explanation);
        EXPECT_TRUE(result);
    }

    // Assert - No memory leaks or crashes
    // Test passes if we get here
}

// Main is provided by GTest::Main library
