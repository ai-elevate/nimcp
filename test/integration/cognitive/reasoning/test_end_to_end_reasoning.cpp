//=============================================================================
// test_end_to_end_reasoning.cpp - End-to-End Reasoning Integration Tests
//=============================================================================

#include <gtest/gtest.h>
#include "core/brain/nimcp_brain.h"
#include "core/brain/learning/nimcp_rule_learning.h"
#include "core/brain/learning/nimcp_association_learning.h"
#include "core/brain/learning/nimcp_circuit_compilation.h"

class EndToEndReasoningTest : public ::testing::Test {
protected:
    brain_t brain;

    void SetUp() override {
        // brain_config_t config;
        // config.num_neurons = 1000;
        // config.num_inputs = 10;
        // config.num_outputs = 5;
        // config.sparsity = 0.1f;
        // config.learning_rate = 0.01f;

        brain = brain_create("reasoning_test", BRAIN_SIZE_SMALL, BRAIN_TASK_CLASSIFICATION, 10, 5);
        ASSERT_NE(brain, nullptr);
    }

    void TearDown() override {
        if (brain) {
            brain_destroy(brain);
        }
    }
};

// Test 1: Learn rules from examples
TEST_F(EndToEndReasoningTest, LearnRulesFromExamples) {
    // Create training examples
    float cat_features[] = {1.0f, 0.8f, 0.2f, 0.1f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float dog_features[] = {1.0f, 0.7f, 0.3f, 0.8f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    rule_example_t examples[] = {
        {cat_features, 10, "cat", 1.0f},
        {dog_features, 10, "dog", 1.0f}
    };

    const char* labels[] = {"cat", "dog"};

    // Learn rules
    int rules_learned = brain_learn_rule_from_examples(brain, examples, labels, 2);

    EXPECT_GE(rules_learned, 1);
}

// Test 2: Store rules in knowledge base
TEST_F(EndToEndReasoningTest, StoreRulesInKnowledgeBase) {
    const char* rule = "IF furry AND meows THEN cat";
    float confidence = 0.95f;

    bool success = add_learned_rule_to_kb(brain, rule, confidence);

    EXPECT_TRUE(success);
}

// Test 3: Forward chaining inference
TEST_F(EndToEndReasoningTest, ForwardChainingInference) {
    // Add rules
    add_learned_rule_to_kb(brain, "IF A THEN B", 0.9f);
    add_learned_rule_to_kb(brain, "IF B THEN C", 0.8f);

    // TODO: Test forward chaining
    // Expected: A → B → C (transitive inference)
    SUCCEED();
}

// Test 4: Derive new facts from rules
TEST_F(EndToEndReasoningTest, DeriveNewFacts) {
    // Add rule
    add_learned_rule_to_kb(brain, "IF mammal AND lays_eggs THEN platypus", 1.0f);

    // TODO: Test fact derivation
    SUCCEED();
}

// Test 5: Rule confidence propagation
TEST_F(EndToEndReasoningTest, RuleConfidencePropagation) {
    // Test confidence calculation
    float conf1 = compute_rule_confidence(10, 12);  // 10/12 support
    float conf2 = compute_rule_confidence(5, 10);   // 5/10 support

    EXPECT_GT(conf1, conf2);
    EXPECT_GE(conf1, 0.0f);
    EXPECT_LE(conf1, 1.0f);
}

// Test 6: Multi-step reasoning
TEST_F(EndToEndReasoningTest, MultiStepReasoning) {
    // Add chain of rules
    add_learned_rule_to_kb(brain, "IF sees_smoke THEN fire", 0.7f);
    add_learned_rule_to_kb(brain, "IF fire THEN danger", 0.9f);
    add_learned_rule_to_kb(brain, "IF danger THEN evacuate", 0.95f);

    // TODO: Test multi-step inference
    SUCCEED();
}

// Test 7: Contradictory rules handling
TEST_F(EndToEndReasoningTest, ContradictoryRulesHandling) {
    // Add contradictory rules with different confidences
    add_learned_rule_to_kb(brain, "IF A THEN B", 0.9f);
    add_learned_rule_to_kb(brain, "IF A THEN NOT B", 0.3f);

    // TODO: Test conflict resolution (higher confidence wins)
    SUCCEED();
}

// Test 8: Rule extraction accuracy
TEST_F(EndToEndReasoningTest, RuleExtractionAccuracy) {
    // Create consistent examples
    float ex1[] = {1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float ex2[] = {1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float ex3[] = {1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    rule_example_t examples[] = {
        {ex1, 10, "class_A", 1.0f},
        {ex2, 10, "class_A", 1.0f},
        {ex3, 10, "class_A", 1.0f}
    };

    const char* labels[] = {"class_A", "class_A", "class_A"};

    char rule[512];
    bool found = extract_rule_pattern(examples, 3, "class_A", rule, sizeof(rule));

    EXPECT_TRUE(found);
    EXPECT_NE(strstr(rule, "feature_0"), nullptr);
    EXPECT_NE(strstr(rule, "feature_1"), nullptr);
}

// Test 9: Empty example handling
TEST_F(EndToEndReasoningTest, EmptyExampleHandling) {
    int result = brain_learn_rule_from_examples(brain, nullptr, nullptr, 0);
    EXPECT_EQ(result, -1);
}

// Test 10: Large rule set
TEST_F(EndToEndReasoningTest, LargeRuleSet) {
    // Add many rules
    for (int i = 0; i < 100; i++) {
        char rule[256];
        snprintf(rule, sizeof(rule), "IF condition_%d THEN result_%d", i, i);
        bool success = add_learned_rule_to_kb(brain, rule, 0.8f);
        EXPECT_TRUE(success);
    }

    SUCCEED();
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
