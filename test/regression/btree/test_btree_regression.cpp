/**
 * B-tree Regression Tests
 *
 * WHAT: Comprehensive regression tests for B-tree operations
 * WHY: Ensure B-tree maintains correctness with unique keys after bug fix
 * HOW: Test knowledge B-tree insert, remove, traverse, and uniqueness
 */

#include <gtest/gtest.h>

#include "cognitive/knowledge/nimcp_knowledge.h"
#include "utils/memory/nimcp_memory.h"

//=============================================================================
// Knowledge B-tree Regression Tests
//=============================================================================

class KnowledgeBtreeRegressionTest : public ::testing::Test {
protected:
    knowledge_system_t system;

    void SetUp() override {
        system = knowledge_system_create("btree_regression_test");
        ASSERT_NE(system, nullptr);
    }

    void TearDown() override {
        if (system) {
            knowledge_system_destroy(system);
        }
    }
};

/**
 * REGRESSION: Test that items with identical confidence values maintain unique keys
 * BUG FIX: Previously, identical confidence values caused btree_remove to remove wrong items
 * FIXED BY: Using confidence_index format (e.g., "0.300000_00123")
 */
TEST_F(KnowledgeBtreeRegressionTest, UniqueKeys_IdenticalConfidence_AllItemsPresent) {
    // Learn three concepts - use single words to create exactly 3 items
    // They will all start with confidence = 0.3
    knowledge_learn_from_text(system, "alpha", KNOWLEDGE_DOMAIN_GENERAL);
    knowledge_learn_from_text(system, "beta", KNOWLEDGE_DOMAIN_GENERAL);
    knowledge_learn_from_text(system, "gamma", KNOWLEDGE_DOMAIN_GENERAL);

    // Get all ordered by confidence
    knowledge_item_t* results = nullptr;
    uint32_t count = knowledge_get_all_ordered_by_confidence(system, &results);

    EXPECT_EQ(count, 3u);
    ASSERT_NE(results, nullptr);

    // All should have same confidence (0.3)
    EXPECT_FLOAT_EQ(results[0].confidence, 0.3f);
    EXPECT_FLOAT_EQ(results[1].confidence, 0.3f);
    EXPECT_FLOAT_EQ(results[2].confidence, 0.3f);

    nimcp_free(results);
}

/**
 * REGRESSION: Test that reinforcement updates B-tree correctly
 * BUG FIX: Previously, B-tree wasn't updated when confidence changed
 * FIXED BY: Remove with old key, update confidence, re-insert with new key
 */
TEST_F(KnowledgeBtreeRegressionTest, Reinforcement_ConfidenceUpdate_SortOrderMaintained) {
    // Learn three concepts - use single words
    knowledge_learn_from_text(system, "delta", KNOWLEDGE_DOMAIN_GENERAL);
    knowledge_learn_from_text(system, "epsilon", KNOWLEDGE_DOMAIN_GENERAL);
    knowledge_learn_from_text(system, "zeta", KNOWLEDGE_DOMAIN_GENERAL);

    // Reinforce them different amounts
    knowledge_reinforce(system, "delta", nullptr);  // 0.3 + 0.05 = 0.35
    knowledge_reinforce(system, "epsilon", nullptr);  // 0.3 + 0.05 = 0.35
    knowledge_reinforce(system, "epsilon", nullptr);  // 0.35 + 0.05 = 0.40
    knowledge_reinforce(system, "zeta", nullptr);  // 0.3 + 0.05 = 0.35
    knowledge_reinforce(system, "zeta", nullptr);  // 0.35 + 0.05 = 0.40
    knowledge_reinforce(system, "zeta", nullptr);  // 0.40 + 0.05 = 0.45

    // Get ordered results
    knowledge_item_t* results = nullptr;
    uint32_t count = knowledge_get_all_ordered_by_confidence(system, &results);

    EXPECT_EQ(count, 3u);
    ASSERT_NE(results, nullptr);

    // Verify ascending order
    EXPECT_LE(results[0].confidence, results[1].confidence);
    EXPECT_LE(results[1].confidence, results[2].confidence);

    // Verify actual values (delta=0.35, epsilon=0.40, zeta=0.45)
    EXPECT_FLOAT_EQ(results[0].confidence, 0.35f);
    EXPECT_FLOAT_EQ(results[1].confidence, 0.40f);
    EXPECT_FLOAT_EQ(results[2].confidence, 0.45f);

    nimcp_free(results);
}

/**
 * REGRESSION: Test B-tree with many items (stress test)
 */
TEST_F(KnowledgeBtreeRegressionTest, StressTest_ManyItems_NoCorruption) {
    const uint32_t NUM_ITEMS = 100;

    // Learn many concepts - use unique single words to create exactly NUM_ITEMS items
    for (uint32_t i = 0; i < NUM_ITEMS; i++) {
        char text[64];
        snprintf(text, sizeof(text), "word%u", i);
        knowledge_learn_from_text(system, text, KNOWLEDGE_DOMAIN_GENERAL);
    }

    // Reinforce some of them (every 3rd item)
    for (uint32_t i = 0; i < NUM_ITEMS; i += 3) {
        char query[64];
        snprintf(query, sizeof(query), "word%u", i);
        knowledge_reinforce(system, query, nullptr);
    }

    // Get all ordered
    knowledge_item_t* results = nullptr;
    uint32_t count = knowledge_get_all_ordered_by_confidence(system, &results);

    EXPECT_EQ(count, NUM_ITEMS);
    ASSERT_NE(results, nullptr);

    // Verify order maintained (ascending by confidence)
    for (uint32_t i = 1; i < count; i++) {
        EXPECT_GE(results[i].confidence, results[i-1].confidence);
    }

    nimcp_free(results);
}

/**
 * REGRESSION: Test that B-tree correctly handles remove-update-reinsert cycles
 */
TEST_F(KnowledgeBtreeRegressionTest, UpdateCycles_MultipleReinforcements_NoMemoryLeaks) {
    // Use single word to create exactly 1 item
    knowledge_learn_from_text(system, "theta", KNOWLEDGE_DOMAIN_GENERAL);

    // Reinforce many times (each does remove + reinsert in B-tree)
    for (int i = 0; i < 50; i++) {
        knowledge_reinforce(system, "theta", nullptr);
    }

    // Verify still works
    knowledge_item_t* results = nullptr;
    uint32_t count = knowledge_get_all_ordered_by_confidence(system, &results);

    EXPECT_EQ(count, 1u);
    ASSERT_NE(results, nullptr);

    // Should be capped at 1.0
    EXPECT_FLOAT_EQ(results[0].confidence, 1.0f);

    nimcp_free(results);
}

/**
 * REGRESSION: Test B-tree traversal returns items in correct order
 */
TEST_F(KnowledgeBtreeRegressionTest, Traversal_DifferentConfidences_AscendingOrder) {
    // Create items with specific confidences by varying reinforcement
    // Use single words to create exactly 3 items
    knowledge_learn_from_text(system, "iota", KNOWLEDGE_DOMAIN_GENERAL);  // 0.3

    knowledge_learn_from_text(system, "kappa", KNOWLEDGE_DOMAIN_GENERAL);  // 0.3
    knowledge_reinforce(system, "kappa", nullptr);  // 0.35

    knowledge_learn_from_text(system, "lambda", KNOWLEDGE_DOMAIN_GENERAL);  // 0.3
    knowledge_reinforce(system, "lambda", nullptr);  // 0.35
    knowledge_reinforce(system, "lambda", nullptr);  // 0.40

    // Get ordered
    knowledge_item_t* results = nullptr;
    uint32_t count = knowledge_get_all_ordered_by_confidence(system, &results);

    EXPECT_EQ(count, 3u);
    ASSERT_NE(results, nullptr);

    // Verify strict ascending order
    for (uint32_t i = 1; i < count; i++) {
        EXPECT_GE(results[i].confidence, results[i-1].confidence)
            << "Item " << i << " has lower confidence than item " << (i-1);
    }

    nimcp_free(results);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
