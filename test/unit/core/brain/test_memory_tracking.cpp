//=============================================================================
// test_memory_tracking.cpp - Memory Tracking Unit Tests
//=============================================================================
/**
 * @file test_memory_tracking.cpp
 * @brief Comprehensive unit tests for pretrained model memory tracking
 *
 * WHAT: Tests for accurate memory usage reporting for brain instances
 * WHY:  Verify 100% correctness of memory tracking implementation
 * HOW:  GoogleTest framework with detailed memory assertions
 *
 * TEST COVERAGE:
 * 1. Basic memory tracking accuracy
 * 2. Component-wise memory breakdown
 * 3. Memory tracking for different brain sizes
 * 4. COW clone memory tracking
 * 5. Cognitive subsystem memory accounting
 * 6. Error handling and edge cases
 */

#include <gtest/gtest.h>
#include "nimcp.h"
#include "core/brain/nimcp_brain.h"
#include <cmath>

class MemoryTrackingTest : public ::testing::Test {
protected:
    void SetUp() override {
        nimcp_init();
    }

    void TearDown() override {
        nimcp_shutdown();
    }

    // Helper: Create brain with known size
    nimcp_brain_t create_brain_with_size(nimcp_brain_size_t size) {
        nimcp_brain_config_t config = nimcp_brain_config_defaults(size);
        return nimcp_brain_create(&config);
    }

    // Helper: Calculate expected minimum memory
    size_t calculate_expected_min_memory(nimcp_brain_t brain) {
        nimcp_brain_stats_t stats;
        if (nimcp_brain_get_stats(brain, &stats) != NIMCP_OK) {
            return 0;
        }

        // Minimum: brain struct + neurons + synapses
        size_t min_memory = sizeof(void*) * 100;  // Rough brain struct estimate
        min_memory += (size_t)stats.num_neurons * 120;
        min_memory += (size_t)stats.num_synapses * 24;

        return min_memory;
    }
};

//=============================================================================
// Test 1: Basic Memory Tracking Non-Zero
//=============================================================================

TEST_F(MemoryTrackingTest, BasicMemoryTrackingNonZero) {
    // WHAT: Verify memory tracking returns non-zero for valid brain
    // WHY:  Ensure basic functionality works
    // HOW:  Create brain and check memory > 0

    nimcp_brain_t brain = create_brain_with_size(NIMCP_BRAIN_SMALL);
    ASSERT_NE(brain, nullptr);

    size_t memory = brain_get_memory_usage(brain->internal_brain);

    // VERIFY: Memory usage is non-zero
    EXPECT_GT(memory, 0);

    // VERIFY: Memory is reasonable (not absurdly large)
    EXPECT_LT(memory, 1024ULL * 1024ULL * 1024ULL);  // < 1GB

    nimcp_brain_destroy(brain);
}

//=============================================================================
// Test 2: Memory Tracking Accuracy - Small Brain
//=============================================================================

TEST_F(MemoryTrackingTest, MemoryTrackingAccuracySmall) {
    // WHAT: Verify memory tracking is accurate for small brain
    // WHY:  Ensure calculations match expected values
    // HOW:  Compare reported memory with calculated minimum

    nimcp_brain_t brain = create_brain_with_size(NIMCP_BRAIN_SMALL);
    ASSERT_NE(brain, nullptr);

    size_t reported_memory = brain_get_memory_usage(brain->internal_brain);
    size_t expected_min = calculate_expected_min_memory(brain);

    // VERIFY: Reported memory >= expected minimum
    EXPECT_GE(reported_memory, expected_min);

    // VERIFY: Reported memory is within reasonable bounds (not 10x expected)
    EXPECT_LE(reported_memory, expected_min * 10);

    nimcp_brain_destroy(brain);
}

//=============================================================================
// Test 3: Memory Tracking Accuracy - Medium Brain
//=============================================================================

TEST_F(MemoryTrackingTest, MemoryTrackingAccuracyMedium) {
    // WHAT: Verify memory tracking is accurate for medium brain
    // WHY:  Test with larger network
    // HOW:  Compare reported memory with calculated minimum

    nimcp_brain_t brain = create_brain_with_size(NIMCP_BRAIN_MEDIUM);
    ASSERT_NE(brain, nullptr);

    size_t reported_memory = brain_get_memory_usage(brain->internal_brain);
    size_t expected_min = calculate_expected_min_memory(brain);

    // VERIFY: Reported memory >= expected minimum
    EXPECT_GE(reported_memory, expected_min);

    // VERIFY: Reported memory is within reasonable bounds
    EXPECT_LE(reported_memory, expected_min * 10);

    nimcp_brain_destroy(brain);
}

//=============================================================================
// Test 4: Memory Scaling with Brain Size
//=============================================================================

TEST_F(MemoryTrackingTest, MemoryScalingWithBrainSize) {
    // WHAT: Verify memory usage scales appropriately with brain size
    // WHY:  Ensure tracking reflects actual memory consumption
    // HOW:  Compare tiny, small, and medium brains

    nimcp_brain_t tiny = create_brain_with_size(NIMCP_BRAIN_TINY);
    nimcp_brain_t small = create_brain_with_size(NIMCP_BRAIN_SMALL);
    nimcp_brain_t medium = create_brain_with_size(NIMCP_BRAIN_MEDIUM);

    ASSERT_NE(tiny, nullptr);
    ASSERT_NE(small, nullptr);
    ASSERT_NE(medium, nullptr);

    size_t mem_tiny = brain_get_memory_usage(tiny->internal_brain);
    size_t mem_small = brain_get_memory_usage(small->internal_brain);
    size_t mem_medium = brain_get_memory_usage(medium->internal_brain);

    // VERIFY: Memory increases with brain size
    EXPECT_LT(mem_tiny, mem_small);
    EXPECT_LT(mem_small, mem_medium);

    // VERIFY: Memory scales reasonably (roughly proportional to neuron count)
    // Medium should be ~10x Small (10K vs 1K neurons)
    double small_to_medium_ratio = (double)mem_medium / (double)mem_small;
    EXPECT_GT(small_to_medium_ratio, 5.0);   // At least 5x
    EXPECT_LT(small_to_medium_ratio, 20.0);  // At most 20x

    nimcp_brain_destroy(tiny);
    nimcp_brain_destroy(small);
    nimcp_brain_destroy(medium);
}

//=============================================================================
// Test 5: Memory Tracking After Training
//=============================================================================

TEST_F(MemoryTrackingTest, MemoryTrackingAfterTraining) {
    // WHAT: Verify memory tracking updates after training
    // WHY:  Ensure dynamic memory changes are captured
    // HOW:  Train brain and check if memory increases

    nimcp_brain_t brain = create_brain_with_size(NIMCP_BRAIN_SMALL);
    ASSERT_NE(brain, nullptr);

    size_t memory_before = brain_get_memory_usage(brain->internal_brain);

    // Train brain extensively
    float inputs[10] = {1.0f, 0.5f, 0.3f, 0.8f, 0.2f, 0.6f, 0.4f, 0.9f, 0.1f, 0.7f};
    for (int i = 0; i < 100; i++) {
        nimcp_brain_teach(brain, inputs, "test_class");
    }

    size_t memory_after = brain_get_memory_usage(brain->internal_brain);

    // VERIFY: Memory may increase or stay same (depends on implementation)
    // At minimum, it should not decrease significantly
    EXPECT_GE(memory_after, memory_before * 0.9);  // Allow 10% variance

    nimcp_brain_destroy(brain);
}

//=============================================================================
// Test 6: COW Clone Memory Tracking
//=============================================================================

TEST_F(MemoryTrackingTest, COWCloneMemoryTracking) {
    // WHAT: Verify memory tracking for COW clones
    // WHY:  Ensure COW sharing is reflected in memory reports
    // HOW:  Create clone and compare memory

    nimcp_brain_t original = create_brain_with_size(NIMCP_BRAIN_SMALL);
    ASSERT_NE(original, nullptr);

    size_t original_memory = brain_get_memory_usage(original->internal_brain);

    // Create COW clone
    nimcp_brain_t clone = nimcp_brain_clone_cow(original);
    ASSERT_NE(clone, nullptr);

    size_t clone_memory = brain_get_memory_usage(clone->internal_brain);

    // VERIFY: Clone memory is comparable to original
    // (May be slightly different due to metadata, but should be similar)
    double ratio = (double)clone_memory / (double)original_memory;
    EXPECT_GT(ratio, 0.8);  // At least 80% of original
    EXPECT_LT(ratio, 1.2);  // At most 120% of original

    nimcp_brain_destroy(original);
    nimcp_brain_destroy(clone);
}

//=============================================================================
// Test 7: Memory Components - Output Labels
//=============================================================================

TEST_F(MemoryTrackingTest, MemoryComponentsOutputLabels) {
    // WHAT: Verify output labels are included in memory tracking
    // WHY:  Ensure all components are accounted for
    // HOW:  Add labels and check memory increases

    nimcp_brain_t brain = create_brain_with_size(NIMCP_BRAIN_SMALL);
    ASSERT_NE(brain, nullptr);

    size_t memory_before = brain_get_memory_usage(brain->internal_brain);

    // Add many labels
    float inputs[10] = {1.0f, 0.5f, 0.3f, 0.8f, 0.2f, 0.6f, 0.4f, 0.9f, 0.1f, 0.7f};
    for (int i = 0; i < 50; i++) {
        char label[32];
        snprintf(label, sizeof(label), "label_%d", i);
        nimcp_brain_teach(brain, inputs, label);
    }

    size_t memory_after = brain_get_memory_usage(brain->internal_brain);

    // VERIFY: Memory increased (labels added)
    EXPECT_GT(memory_after, memory_before);

    nimcp_brain_destroy(brain);
}

//=============================================================================
// Test 8: Memory Tracking - Decision Caching
//=============================================================================

TEST_F(MemoryTrackingTest, MemoryTrackingDecisionCaching) {
    // WHAT: Verify decision cache is included in memory tracking
    // WHY:  Ensure all allocated memory is tracked
    // HOW:  Make decisions and check memory

    nimcp_brain_t brain = create_brain_with_size(NIMCP_BRAIN_SMALL);
    ASSERT_NE(brain, nullptr);

    size_t memory_before = brain_get_memory_usage(brain->internal_brain);

    // Make decisions to populate cache
    float inputs[10] = {1.0f, 0.5f, 0.3f, 0.8f, 0.2f, 0.6f, 0.4f, 0.9f, 0.1f, 0.7f};
    nimcp_brain_decision_t decision;

    for (int i = 0; i < 10; i++) {
        nimcp_brain_decide(brain, inputs, &decision);
        nimcp_brain_decision_destroy(&decision);
    }

    size_t memory_after = brain_get_memory_usage(brain->internal_brain);

    // VERIFY: Memory may increase (cache allocated) or stay same
    EXPECT_GE(memory_after, memory_before * 0.9);  // Allow 10% variance

    nimcp_brain_destroy(brain);
}

//=============================================================================
// Test 9: Error Handling - NULL Brain
//=============================================================================

TEST_F(MemoryTrackingTest, ErrorHandlingNullBrain) {
    // WHAT: Verify proper error handling for NULL brain
    // WHY:  Ensure robustness against invalid inputs
    // HOW:  Pass NULL and check for 0 return

    size_t memory = brain_get_memory_usage(nullptr);

    EXPECT_EQ(memory, 0);
}

//=============================================================================
// Test 10: Memory Consistency Across Calls
//=============================================================================

TEST_F(MemoryTrackingTest, MemoryConsistencyAcrossCalls) {
    // WHAT: Verify memory tracking is consistent across multiple calls
    // WHY:  Ensure tracking is deterministic
    // HOW:  Call multiple times and compare results

    nimcp_brain_t brain = create_brain_with_size(NIMCP_BRAIN_SMALL);
    ASSERT_NE(brain, nullptr);

    size_t memory1 = brain_get_memory_usage(brain->internal_brain);
    size_t memory2 = brain_get_memory_usage(brain->internal_brain);
    size_t memory3 = brain_get_memory_usage(brain->internal_brain);

    // VERIFY: All calls return same value (brain unchanged)
    EXPECT_EQ(memory1, memory2);
    EXPECT_EQ(memory2, memory3);

    nimcp_brain_destroy(brain);
}

//=============================================================================
// Test 11: Memory Breakdown Reasonableness
//=============================================================================

TEST_F(MemoryTrackingTest, MemoryBreakdownReasonableness) {
    // WHAT: Verify memory breakdown components are reasonable
    // WHY:  Ensure no component dominates unreasonably
    // HOW:  Calculate expected component sizes and compare

    nimcp_brain_t brain = create_brain_with_size(NIMCP_BRAIN_SMALL);
    ASSERT_NE(brain, nullptr);

    size_t total_memory = brain_get_memory_usage(brain->internal_brain);

    // Get brain stats
    nimcp_brain_stats_t stats;
    ASSERT_EQ(nimcp_brain_get_stats(brain, &stats), NIMCP_OK);

    // Calculate network memory (should be dominant component)
    size_t network_memory = (size_t)(stats.num_neurons * 120 + stats.num_synapses * 24);

    // VERIFY: Network memory is significant portion of total
    double network_ratio = (double)network_memory / (double)total_memory;
    EXPECT_GT(network_ratio, 0.5);  // Network should be > 50% of total

    nimcp_brain_destroy(brain);
}

//=============================================================================
// Test 12: Memory Tracking Performance
//=============================================================================

TEST_F(MemoryTrackingTest, MemoryTrackingPerformance) {
    // WHAT: Verify memory tracking is fast (<1ms)
    // WHY:  Ensure tracking doesn't add significant overhead
    // HOW:  Time multiple calls and verify speed

    nimcp_brain_t brain = create_brain_with_size(NIMCP_BRAIN_SMALL);
    ASSERT_NE(brain, nullptr);

    // Warm up
    brain_get_memory_usage(brain->internal_brain);

    // Time 1000 calls
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; i++) {
        brain_get_memory_usage(brain->internal_brain);
    }
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double avg_us = duration.count() / 1000.0;

    // VERIFY: Average call is fast (< 100 microseconds)
    EXPECT_LT(avg_us, 100.0);

    nimcp_brain_destroy(brain);
}

//=============================================================================
// Main
//=============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
