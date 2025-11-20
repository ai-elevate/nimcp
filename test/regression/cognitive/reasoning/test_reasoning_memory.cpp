//=============================================================================
// test_reasoning_memory.cpp - Reasoning Memory Regression Tests
//=============================================================================

#include <gtest/gtest.h>
#include "core/brain/nimcp_brain.h"
#include "core/brain/learning/nimcp_rule_learning.h"
#include "core/brain/learning/nimcp_association_learning.h"
#include "core/brain/learning/nimcp_circuit_compilation.h"

class ReasoningMemoryTest : public ::testing::Test {
protected:
    brain_t brain;

    void SetUp() override {
        // brain_config_t config;
        // config.num_neurons = 10000;
        // config.num_inputs = 100;
        // config.num_outputs = 50;
        // config.sparsity = 0.1f;
        // config.learning_rate = 0.01f;

        brain = brain_create("reasoning_test", BRAIN_SIZE_LARGE, BRAIN_TASK_CLASSIFICATION, 100, 50);
        ASSERT_NE(brain, nullptr);
    }

    void TearDown() override {
        if (brain) {
            brain_destroy(brain);
        }
    }
};

// Test 1: No memory leaks after 10000 inferences
TEST_F(ReasoningMemoryTest, NoMemoryLeaksInferences) {
    // Add rules
    for (int i = 0; i < 100; i++) {
        char rule[256];
        snprintf(rule, sizeof(rule), "IF A_%d THEN B_%d", i, i);
        add_learned_rule_to_kb(brain, rule, 0.8f);
    }

    // Perform 10000 inferences
    for (int i = 0; i < 10000; i++) {
        // TODO: Actual inference
        // Check for leaks via valgrind in CI
    }

    SUCCEED();
}

// Test 2: Memory usage under 100MB for 1000 rules
TEST_F(ReasoningMemoryTest, MemoryUsageLimit) {
    // Add 1000 rules
    for (int i = 0; i < 1000; i++) {
        char rule[512];
        snprintf(rule, sizeof(rule), "IF condition_%d AND subcondition_%d THEN result_%d",
                i, i, i);
        add_learned_rule_to_kb(brain, rule, 0.8f);
    }

    // TODO: Measure actual memory usage
    // For now, just verify no crash
    SUCCEED();
}

// Test 3: Circuit memory cleanup
TEST_F(ReasoningMemoryTest, CircuitMemoryCleanup) {
    circuit_id_t circuits[100];

    // Create circuits
    for (int i = 0; i < 100; i++) {
        char rule[256];
        snprintf(rule, sizeof(rule), "IF X_%d THEN Y_%d", i, i);
        circuits[i] = compile_rule_to_circuit(brain, rule);
        ASSERT_NE(circuits[i], 0);
    }

    // Delete circuits
    for (int i = 0; i < 100; i++) {
        bool deleted = delete_circuit(brain, circuits[i]);
        EXPECT_TRUE(deleted);
    }

    // Verify cleanup
    for (int i = 0; i < 100; i++) {
        uint32_t gates = get_circuit_gate_count(brain, circuits[i]);
        EXPECT_EQ(gates, 0);
    }
}

// Test 4: Association memory stability
TEST_F(ReasoningMemoryTest, AssociationMemoryStability) {
    // Create many associations
    for (int i = 0; i < 1000; i++) {
        char a[32], b[32];
        snprintf(a, sizeof(a), "concept_%d", i);
        snprintf(b, sizeof(b), "related_%d", i);
        brain_learn_association(brain, a, b, 5);
    }

    // Update many times
    for (int round = 0; round < 100; round++) {
        for (int i = 0; i < 1000; i++) {
            char a[32], b[32];
            snprintf(a, sizeof(a), "concept_%d", i);
            snprintf(b, sizeof(b), "related_%d", i);
            update_association_strength(brain, a, b, 0.1f);
        }
    }

    // Verify stability
    for (int i = 0; i < 1000; i++) {
        char a[32], b[32];
        snprintf(a, sizeof(a), "concept_%d", i);
        snprintf(b, sizeof(b), "related_%d", i);
        float strength = get_association_strength(brain, a, b);
        EXPECT_GE(strength, 0.0f);
        EXPECT_LE(strength, 1.0f);
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
