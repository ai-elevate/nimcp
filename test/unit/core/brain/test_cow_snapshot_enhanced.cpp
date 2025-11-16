//=============================================================================
// test_cow_snapshot_enhanced.cpp - Enhanced COW Snapshot Unit Tests
//=============================================================================
/**
 * @file test_cow_snapshot_enhanced.cpp
 * @brief Comprehensive unit tests for enhanced COW snapshot functionality
 *
 * WHAT: Tests for advanced cache reference tracking in COW snapshots
 * WHY:  Verify 100% correctness of snapshot isolation and memory tracking
 * HOW:  GoogleTest framework with detailed assertions
 *
 * TEST COVERAGE:
 * 1. Snapshot creation with reference tracking
 * 2. Memory size calculation accuracy
 * 3. Snapshot isolation guarantees
 * 4. Multiple snapshot scenarios
 * 5. Cache statistics integration
 * 6. Error handling and edge cases
 */

#include <gtest/gtest.h>
#include "nimcp.h"
#include "utils/cache/nimcp_cache.h"
#include "core/brain/nimcp_brain.h"
#include <cstring>

class COWSnapshotEnhancedTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize NIMCP system
        nimcp_init();

        // Clear cache statistics
        nimcp_cache_clear_stats();

        // Create test brain
        nimcp_brain_config_t config = nimcp_brain_config_defaults(NIMCP_BRAIN_SMALL);
        config.num_inputs = 10;
        config.num_outputs = 5;
        brain = nimcp_brain_create(&config);
        ASSERT_NE(brain, nullptr);
    }

    void TearDown() override {
        if (brain) {
            nimcp_brain_destroy(brain);
        }
        nimcp_shutdown();
    }

    nimcp_brain_t brain = nullptr;
};

//=============================================================================
// Test 1: Snapshot Creation with Reference Tracking
//=============================================================================

TEST_F(COWSnapshotEnhancedTest, SnapshotCreationWithReferenceTracking) {
    // WHAT: Verify snapshot creation tracks references correctly
    // WHY:  Ensure cache statistics are updated properly
    // HOW:  Create snapshot and check cache stats

    // Get initial cache stats
    nimcp_cache_stats_t stats_before;
    ASSERT_TRUE(nimcp_cache_get_stats(&stats_before));
    uint64_t refs_before = stats_before.references_created;

    // Create COW snapshot
    nimcp_brain_snapshot_t snapshot = nimcp_brain_snapshot_cow(brain);
    ASSERT_NE(snapshot, nullptr);

    // Get cache stats after snapshot
    nimcp_cache_stats_t stats_after;
    ASSERT_TRUE(nimcp_cache_get_stats(&stats_after));

    // VERIFY: Reference count increased
    EXPECT_GT(stats_after.references_created, refs_before);

    // VERIFY: Memory saved increased (COW benefits)
    EXPECT_GT(stats_after.memory_saved, stats_before.memory_saved);

    // Cleanup
    nimcp_brain_snapshot_destroy(snapshot);
}

//=============================================================================
// Test 2: Memory Size Calculation Accuracy
//=============================================================================

TEST_F(COWSnapshotEnhancedTest, MemorySizeCalculationAccuracy) {
    // WHAT: Verify shared memory size calculation is accurate
    // WHY:  Ensure memory tracking reports correct values
    // HOW:  Compare calculated size with brain stats

    // Create snapshot
    nimcp_brain_snapshot_t snapshot = nimcp_brain_snapshot_cow(brain);
    ASSERT_NE(snapshot, nullptr);

    // Get brain stats to calculate expected size
    nimcp_brain_stats_t stats;
    ASSERT_EQ(nimcp_brain_get_stats(brain, &stats), NIMCP_OK);

    // Expected size: neurons * 100 + synapses * 20
    size_t expected_min_size = (size_t)(stats.num_neurons * 100 + stats.num_synapses * 20);

    // Get cache stats
    nimcp_cache_stats_t cache_stats;
    ASSERT_TRUE(nimcp_cache_get_stats(&cache_stats));

    // VERIFY: Memory saved is at least the expected network size
    EXPECT_GE(cache_stats.memory_saved, expected_min_size);

    // Cleanup
    nimcp_brain_snapshot_destroy(snapshot);
}

//=============================================================================
// Test 3: Snapshot Isolation Guarantees
//=============================================================================

TEST_F(COWSnapshotEnhancedTest, SnapshotIsolationGuarantees) {
    // WHAT: Verify snapshot is isolated from original brain modifications
    // WHY:  Ensure COW semantics work correctly
    // HOW:  Modify original after snapshot and verify snapshot unchanged

    // Train original brain
    float inputs[10] = {1.0f, 0.5f, 0.3f, 0.8f, 0.2f, 0.6f, 0.4f, 0.9f, 0.1f, 0.7f};
    nimcp_brain_teach(brain, inputs, "class_A");

    // Create snapshot
    nimcp_brain_snapshot_t snapshot = nimcp_brain_snapshot_cow(brain);
    ASSERT_NE(snapshot, nullptr);

    // Get decision from snapshot
    nimcp_brain_decision_t snapshot_decision;
    ASSERT_EQ(nimcp_brain_snapshot_decide(snapshot, inputs, &snapshot_decision), NIMCP_OK);
    const char* snapshot_output = snapshot_decision.output_label;

    // Train original brain on different data
    float new_inputs[10] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f};
    for (int i = 0; i < 100; i++) {
        nimcp_brain_teach(brain, new_inputs, "class_B");
    }

    // Get decision from snapshot again
    nimcp_brain_decision_t snapshot_decision2;
    ASSERT_EQ(nimcp_brain_snapshot_decide(snapshot, inputs, &snapshot_decision2), NIMCP_OK);

    // VERIFY: Snapshot output unchanged despite original brain training
    EXPECT_STREQ(snapshot_output, snapshot_decision2.output_label);

    // Cleanup
    nimcp_brain_decision_destroy(&snapshot_decision);
    nimcp_brain_decision_destroy(&snapshot_decision2);
    nimcp_brain_snapshot_destroy(snapshot);
}

//=============================================================================
// Test 4: Multiple Snapshots Share Data
//=============================================================================

TEST_F(COWSnapshotEnhancedTest, MultipleSnapshotsShareData) {
    // WHAT: Verify multiple snapshots share underlying network data
    // WHY:  Ensure memory efficiency with multiple snapshots
    // HOW:  Create multiple snapshots and check cache stats

    // Get initial memory usage
    nimcp_cache_stats_t stats_before;
    ASSERT_TRUE(nimcp_cache_get_stats(&stats_before));
    size_t memory_before = stats_before.memory_saved;

    // Create first snapshot
    nimcp_brain_snapshot_t snapshot1 = nimcp_brain_snapshot_cow(brain);
    ASSERT_NE(snapshot1, nullptr);

    // Get memory after first snapshot
    nimcp_cache_stats_t stats_after1;
    ASSERT_TRUE(nimcp_cache_get_stats(&stats_after1));
    size_t memory_increase1 = stats_after1.memory_saved - memory_before;

    // Create second snapshot
    nimcp_brain_snapshot_t snapshot2 = nimcp_brain_snapshot_cow(brain);
    ASSERT_NE(snapshot2, nullptr);

    // Get memory after second snapshot
    nimcp_cache_stats_t stats_after2;
    ASSERT_TRUE(nimcp_cache_get_stats(&stats_after2));
    size_t memory_increase2 = stats_after2.memory_saved - stats_after1.memory_saved;

    // VERIFY: Second snapshot adds same amount (reference counted, not copied)
    EXPECT_EQ(memory_increase1, memory_increase2);

    // Create third snapshot
    nimcp_brain_snapshot_t snapshot3 = nimcp_brain_snapshot_cow(brain);
    ASSERT_NE(snapshot3, nullptr);

    // Get memory after third snapshot
    nimcp_cache_stats_t stats_after3;
    ASSERT_TRUE(nimcp_cache_get_stats(&stats_after3));
    size_t memory_increase3 = stats_after3.memory_saved - stats_after2.memory_saved;

    // VERIFY: Third snapshot also adds same amount
    EXPECT_EQ(memory_increase1, memory_increase3);

    // Cleanup
    nimcp_brain_snapshot_destroy(snapshot1);
    nimcp_brain_snapshot_destroy(snapshot2);
    nimcp_brain_snapshot_destroy(snapshot3);
}

//=============================================================================
// Test 5: Snapshot Restoration Preserves State
//=============================================================================

TEST_F(COWSnapshotEnhancedTest, SnapshotRestorationPreservesState) {
    // WHAT: Verify snapshot restoration accurately restores brain state
    // WHY:  Ensure rollback functionality works correctly
    // HOW:  Create snapshot, modify brain, restore, and verify

    // Train original brain
    float inputs[10] = {1.0f, 0.5f, 0.3f, 0.8f, 0.2f, 0.6f, 0.4f, 0.9f, 0.1f, 0.7f};
    nimcp_brain_teach(brain, inputs, "original_state");

    // Get decision before snapshot
    nimcp_brain_decision_t decision_before;
    ASSERT_EQ(nimcp_brain_decide(brain, inputs, &decision_before), NIMCP_OK);

    // Create snapshot
    nimcp_brain_snapshot_t snapshot = nimcp_brain_snapshot_cow(brain);
    ASSERT_NE(snapshot, nullptr);

    // Modify brain significantly
    float new_inputs[10] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f};
    for (int i = 0; i < 200; i++) {
        nimcp_brain_teach(brain, new_inputs, "modified_state");
    }

    // Get decision after modification
    nimcp_brain_decision_t decision_after_mod;
    ASSERT_EQ(nimcp_brain_decide(brain, inputs, &decision_after_mod), NIMCP_OK);

    // VERIFY: Brain state changed
    EXPECT_STRNE(decision_before.output_label, decision_after_mod.output_label);

    // Restore from snapshot
    ASSERT_EQ(nimcp_brain_restore_cow(brain, snapshot), NIMCP_OK);

    // Get decision after restoration
    nimcp_brain_decision_t decision_restored;
    ASSERT_EQ(nimcp_brain_decide(brain, inputs, &decision_restored), NIMCP_OK);

    // VERIFY: Restored state matches original snapshot state
    EXPECT_STREQ(decision_before.output_label, decision_restored.output_label);

    // Cleanup
    nimcp_brain_decision_destroy(&decision_before);
    nimcp_brain_decision_destroy(&decision_after_mod);
    nimcp_brain_decision_destroy(&decision_restored);
    nimcp_brain_snapshot_destroy(snapshot);
}

//=============================================================================
// Test 6: Error Handling - NULL Brain
//=============================================================================

TEST_F(COWSnapshotEnhancedTest, ErrorHandlingNullBrain) {
    // WHAT: Verify proper error handling for NULL brain
    // WHY:  Ensure robustness against invalid inputs
    // HOW:  Pass NULL and check for NULL return

    nimcp_brain_snapshot_t snapshot = nimcp_brain_snapshot_cow(nullptr);
    EXPECT_EQ(snapshot, nullptr);

    const char* error = nimcp_get_error();
    EXPECT_NE(error, nullptr);
    EXPECT_NE(strstr(error, "NULL"), nullptr);
}

//=============================================================================
// Test 7: Error Handling - NULL Snapshot
//=============================================================================

TEST_F(COWSnapshotEnhancedTest, ErrorHandlingNullSnapshot) {
    // WHAT: Verify proper error handling for NULL snapshot in restore
    // WHY:  Ensure robustness against invalid inputs
    // HOW:  Pass NULL snapshot and check for error

    nimcp_status_t status = nimcp_brain_restore_cow(brain, nullptr);
    EXPECT_EQ(status, NIMCP_ERROR_NULL_ARG);

    const char* error = nimcp_get_error();
    EXPECT_NE(error, nullptr);
    EXPECT_NE(strstr(error, "NULL"), nullptr);
}

//=============================================================================
// Test 8: Cache Statistics Consistency
//=============================================================================

TEST_F(COWSnapshotEnhancedTest, CacheStatisticsConsistency) {
    // WHAT: Verify cache statistics remain consistent across operations
    // WHY:  Ensure accurate memory tracking
    // HOW:  Create/destroy snapshots and verify stats update correctly

    // Get initial stats
    nimcp_cache_stats_t stats_initial;
    ASSERT_TRUE(nimcp_cache_get_stats(&stats_initial));

    // Create snapshot
    nimcp_brain_snapshot_t snapshot = nimcp_brain_snapshot_cow(brain);
    ASSERT_NE(snapshot, nullptr);

    // Get stats after creation
    nimcp_cache_stats_t stats_created;
    ASSERT_TRUE(nimcp_cache_get_stats(&stats_created));

    // VERIFY: References increased
    EXPECT_GT(stats_created.references_created, stats_initial.references_created);

    // Destroy snapshot
    nimcp_brain_snapshot_destroy(snapshot);

    // Get stats after destruction
    nimcp_cache_stats_t stats_destroyed;
    ASSERT_TRUE(nimcp_cache_get_stats(&stats_destroyed));

    // VERIFY: Active allocations decreased (if tracked)
    // Note: This depends on implementation details
    EXPECT_GE(stats_created.active_allocations, stats_destroyed.active_allocations);
}

//=============================================================================
// Main
//=============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
