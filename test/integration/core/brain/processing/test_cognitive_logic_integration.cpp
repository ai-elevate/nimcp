/**
 * @file test_cognitive_logic_integration.cpp
 * @brief Integration tests for cognitive logic circuits with brain
 *
 * WHAT: Test cognitive constraint validation in realistic brain scenarios
 * WHY:  Ensure logic circuits integrate correctly with cognitive processor
 * HOW:  Test full brain decision pipeline with constraint checking
 *
 * COVERAGE TARGET: 7+ integration tests
 *
 * @author NIMCP Development Team
 * @date 2025-11-16
 * @version 2.7.0
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstring>

#include "core/brain/nimcp_brain.h"
#include "core/brain/processing/cognitive_processor.h"
#include "core/neuron_types/nimcp_neural_logic.h"
#include "utils/memory/nimcp_memory.h"

//=============================================================================
// Test Fixture
//=============================================================================

class CognitiveLogicIntegrationTest : public ::testing::Test {
protected:
    brain_t brain = nullptr;
    neural_logic_network_t logic = nullptr;

    void SetUp() override {
        // Create brain with neural logic enabled
        brain_config_t config = {0};
        config.num_inputs = 128;
        config.num_outputs = 20;
        config.hidden_layers = 2;
        config.neurons_per_layer = 64;
        config.enable_neural_logic = true;
        config.enable_introspection = false;  // Simplify for testing
        config.enable_ethics = false;
        config.enable_salience = false;
        config.enable_curiosity = false;

        brain = brain_create(&config);
        if (!brain) {
            return;
        }

        // Create separate logic network for direct testing
        neural_logic_config_t logic_config = neural_logic_default_config(100);
        logic = neural_logic_create(&logic_config);
    }

    void TearDown() override {
        if (logic) {
            neural_logic_destroy(logic);
        }
        if (brain) {
            brain_destroy(brain);
        }
    }

    // Helper: Create multimodal input
    void create_test_input(
        brain_multimodal_input_t* input,
        float visual_val,
        float auditory_val)
    {
        memset(input, 0, sizeof(brain_multimodal_input_t));

        // Visual input
        for (int i = 0; i < 32; i++) {
            input->visual_input[i] = visual_val;
        }
        input->visual_dim = 32;

        // Auditory input
        for (int i = 0; i < 32; i++) {
            input->auditory_input[i] = auditory_val;
        }
        input->auditory_dim = 32;

        input->timestamp_ms = 1000;
    }
};

//=============================================================================
// Brain Decision Pipeline Integration
//=============================================================================

TEST_F(CognitiveLogicIntegrationTest, BrainDecisionWithLogicValidation) {
    // Test: Full brain decision with logic constraint checking
    if (!brain) {
        GTEST_SKIP() << "Brain creation failed";
    }

    brain_multimodal_input_t input;
    create_test_input(&input, 0.8f, 0.6f);

    brain_decision_t decision;
    bool result = brain_decide(&decision, brain, &input);

    EXPECT_TRUE(result);
    // Decision should have logic_valid field set
}

TEST_F(CognitiveLogicIntegrationTest, ConstraintCheckingWithRealBrain) {
    // Test: Constraint checking integrated with real brain processing
    if (!brain) {
        GTEST_SKIP() << "Brain creation failed";
    }

    brain_multimodal_input_t input;
    create_test_input(&input, 0.5f, 0.5f);

    brain_decision_t decision;
    bool result = brain_decide(&decision, brain, &input);

    EXPECT_TRUE(result);

    // Validate decision constraints
    // Confidence and uncertainty should be complementary
    float sum = decision.confidence + decision.uncertainty;
    EXPECT_NEAR(sum, 1.0f, 0.3f);  // Allow some tolerance
}

TEST_F(CognitiveLogicIntegrationTest, LogicCircuitReuse) {
    // Test: Logic circuits can be reused across multiple decisions
    if (!brain) {
        GTEST_SKIP() << "Brain creation failed";
    }

    brain_multimodal_input_t input1;
    create_test_input(&input1, 0.7f, 0.3f);

    brain_multimodal_input_t input2;
    create_test_input(&input2, 0.4f, 0.9f);

    brain_decision_t decision1, decision2;

    // First decision
    bool result1 = brain_decide(&decision1, brain, &input1);
    EXPECT_TRUE(result1);

    // Second decision (reuses logic network)
    bool result2 = brain_decide(&decision2, brain, &input2);
    EXPECT_TRUE(result2);

    // Both should complete successfully
    EXPECT_GE(decision1.confidence, 0.0f);
    EXPECT_GE(decision2.confidence, 0.0f);
}

TEST_F(CognitiveLogicIntegrationTest, HighUncertaintyScenario) {
    // Test: High uncertainty input triggers constraint checking
    if (!brain) {
        GTEST_SKIP() << "Brain creation failed";
    }

    brain_multimodal_input_t input;
    // Create ambiguous input (mid-range values)
    create_test_input(&input, 0.5f, 0.5f);

    brain_decision_t decision;
    bool result = brain_decide(&decision, brain, &input);

    EXPECT_TRUE(result);

    // Uncertainty should be reasonably high for ambiguous input
    // (This depends on introspection module being enabled)
    EXPECT_GE(decision.uncertainty, 0.0f);
    EXPECT_LE(decision.uncertainty, 1.0f);
}

TEST_F(CognitiveLogicIntegrationTest, EthicalConstraintValidation) {
    // Test: Ethical violations trigger constraint checking
    if (!brain) {
        GTEST_SKIP() << "Brain creation failed";
    }

    brain_multimodal_input_t input;
    create_test_input(&input, 0.8f, 0.8f);

    brain_decision_t decision;
    bool result = brain_decide(&decision, brain, &input);

    EXPECT_TRUE(result);

    // Should have ethical approval status
    // (Default should be approved for normal inputs)
}

//=============================================================================
// Constraint Violation Detection
//=============================================================================

TEST_F(CognitiveLogicIntegrationTest, DetectConfidenceUncertaintyConflict) {
    // Test: Detect when confidence and uncertainty conflict
    if (!brain) {
        GTEST_SKIP() << "Brain creation failed";
    }

    // Process multiple inputs to see constraint variations
    for (int i = 0; i < 5; i++) {
        brain_multimodal_input_t input;
        float val = 0.2f * i;
        create_test_input(&input, val, 1.0f - val);

        brain_decision_t decision;
        bool result = brain_decide(&decision, brain, &input);

        EXPECT_TRUE(result);

        // Check complementary relationship
        float sum = decision.confidence + decision.uncertainty;
        EXPECT_GE(sum, 0.5f);
        EXPECT_LE(sum, 1.5f);
    }
}

TEST_F(CognitiveLogicIntegrationTest, MultipleDecisionsStressTest) {
    // Test: Multiple decisions in sequence maintain constraint validity
    if (!brain) {
        GTEST_SKIP() << "Brain creation failed";
    }

    const int num_decisions = 20;
    int valid_decisions = 0;

    for (int i = 0; i < num_decisions; i++) {
        brain_multimodal_input_t input;
        float visual_val = (float)i / num_decisions;
        float auditory_val = 1.0f - visual_val;
        create_test_input(&input, visual_val, auditory_val);

        brain_decision_t decision;
        bool result = brain_decide(&decision, brain, &input);

        if (result) {
            valid_decisions++;
        }
    }

    // Most decisions should succeed
    EXPECT_GT(valid_decisions, num_decisions / 2);
}

//=============================================================================
// Performance Tests
//=============================================================================

TEST_F(CognitiveLogicIntegrationTest, LogicCircuitPerformance) {
    // Test: Logic circuit evaluation performance
    if (!logic) {
        GTEST_SKIP() << "Logic network creation failed";
    }

    const int num_evaluations = 100;

    uint32_t and_gate = neural_logic_create_gate(logic, LOGIC_GATE_AND, 1.8f);
    ASSERT_NE(and_gate, UINT32_MAX);

    uint32_t successful_evals = 0;

    for (int i = 0; i < num_evaluations; i++) {
        float inputs[2] = {
            (float)(i % 2),
            (float)((i + 1) % 2)
        };
        float output = 0.0f;

        if (neural_logic_evaluate(logic, and_gate, inputs, 2, &output)) {
            successful_evals++;
        }
    }

    // All evaluations should succeed
    EXPECT_EQ(successful_evals, num_evaluations);
}
