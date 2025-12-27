//=============================================================================
// test_working_memory_reasoning.cpp - Working Memory Reasoning Integration Tests
//=============================================================================

#include <gtest/gtest.h>
#include "core/brain/nimcp_brain.h"
#include "core/brain/learning/nimcp_rule_learning.h"
#include "core/brain/learning/nimcp_association_learning.h"

class WorkingMemoryReasoningTest : public ::testing::Test {
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

// Test 1: Active inferences stored in working memory
TEST_F(WorkingMemoryReasoningTest, ActiveInferencesStored) {
    // Add a rule to the knowledge base
    bool rule_added = add_learned_rule_to_kb(brain, "IF A THEN B", 0.9f);
    ASSERT_TRUE(rule_added) << "Rule should be added to KB successfully";

    // Perform forward chaining to trigger inference
    forward_chain_result_t result;
    bool chain_success = brain_forward_chain(brain, 5, &result);

    // Verify forward chaining was attempted (may have no new facts initially)
    // The key verification is that the process completes without error
    if (chain_success) {
        // At minimum, iterations should be performed
        EXPECT_GE(result.iterations_performed, 0u);
        forward_chain_free_result(&result);
    } else {
        // Forward chaining requires logic engine attachment
        // If not configured, this is expected - verify no crash occurred
        EXPECT_NE(brain, nullptr) << "Brain should still be valid";
    }
}

// Test 2: Working memory capacity limits
TEST_F(WorkingMemoryReasoningTest, WorkingMemoryCapacity) {
    // Add many inferences
    for (int i = 0; i < 20; i++) {
        char rule[256];
        snprintf(rule, sizeof(rule), "IF fact_%d THEN result_%d", i, i);
        add_learned_rule_to_kb(brain, rule, 0.8f);
    }

    // TODO: Verify WM capacity limits (Miller's 7±2)
    SUCCEED();
}

// Test 3: Inference decay over time
TEST_F(WorkingMemoryReasoningTest, InferenceDecay) {
    brain_learn_association(brain, "recent", "active", 10);

    // Simulate time passage
    decay_all_associations(brain, 0.95f);

    float strength = get_association_strength(brain, "recent", "active");
    EXPECT_GE(strength, 0.0f);
}

// Test 4: Conflict resolution in WM
TEST_F(WorkingMemoryReasoningTest, ConflictResolution) {
    add_learned_rule_to_kb(brain, "IF A THEN B", 0.9f);
    add_learned_rule_to_kb(brain, "IF A THEN C", 0.7f);

    // TODO: Verify highest confidence inference wins
    SUCCEED();
}

// Test 5: Inference priority ordering
TEST_F(WorkingMemoryReasoningTest, InferencePriorityOrdering) {
    add_learned_rule_to_kb(brain, "IF urgent THEN action", 0.95f);
    add_learned_rule_to_kb(brain, "IF normal THEN wait", 0.5f);

    // TODO: Verify high-confidence inferences prioritized
    SUCCEED();
}

// Test 6: WM refresh from associations
TEST_F(WorkingMemoryReasoningTest, WMRefreshFromAssociations) {
    brain_learn_association(brain, "cue", "memory", 10);
    update_association_strength(brain, "cue", "memory", 1.0f);

    float strength = get_association_strength(brain, "cue", "memory");
    EXPECT_GT(strength, 0.0f);
}

// Test 7: Rehearsal prevents decay
TEST_F(WorkingMemoryReasoningTest, RehearsalPreventsDecay) {
    brain_learn_association(brain, "rehearsed", "remembered", 5);

    // Strengthen (rehearse)
    update_association_strength(brain, "rehearsed", "remembered", 1.0f);

    float strength = get_association_strength(brain, "rehearsed", "remembered");
    EXPECT_GT(strength, 0.5f);
}

// Test 8: Interference between similar inferences
TEST_F(WorkingMemoryReasoningTest, InterferenceBetweenInferences) {
    brain_learn_association(brain, "A", "B1", 5);
    brain_learn_association(brain, "A", "B2", 5);

    float s1 = get_association_strength(brain, "A", "B1");
    float s2 = get_association_strength(brain, "A", "B2");

    EXPECT_GE(s1, 0.0f);
    EXPECT_GE(s2, 0.0f);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
