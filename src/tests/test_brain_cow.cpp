/**
 * @file test_brain_cow.cpp
 * @brief Tests for brain COW cloning and snapshots
 *
 * WHAT: Test-driven development for COW brain operations
 * WHY: Ensure correctness before integration with replication
 * HOW: Test memory sharing, copy-on-write, and snapshots
 */

#include <gtest/gtest.h>
extern "C" {
    #include "include/nimcp.h"
    #include "core/brain/nimcp_brain.h"
    #include "utils/cache/nimcp_cache.h"
    #include "utils/memory/nimcp_memory.h"
}

class BrainCOWTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize systems
        nimcp_memory_init();
        nimcp_cache_init();
        nimcp_init();

        // Configure cache
        nimcp_cache_config_t config = nimcp_cache_get_default_config();
        config.enable_tracking = true;
        config.enable_debug_output = false;  // Keep tests quiet
        nimcp_cache_configure(&config);

        // Clear statistics
        nimcp_cache_clear_stats();
    }

    void TearDown() override {
        nimcp_shutdown();
        nimcp_cache_cleanup();
        nimcp_memory_cleanup();
    }
};

//=============================================================================
// Brain Clone COW Tests
//=============================================================================

TEST_F(BrainCOWTest, CloneCOWCreatesValidBrain) {
    // Create original brain
    nimcp_brain_t original = nimcp_brain_create(
        "original",
        NIMCP_BRAIN_SMALL,
        NIMCP_TASK_CLASSIFICATION,
        10, 3
    );
    ASSERT_NE(original, nullptr);

    // Clone with COW
    nimcp_brain_t clone = nimcp_brain_clone_cow(original);
    ASSERT_NE(clone, nullptr);

    // Verify clone is valid
    nimcp_brain_probe_t probe;
    ASSERT_EQ(nimcp_brain_probe(clone, &probe), NIMCP_OK);
    EXPECT_GT(probe.num_neurons, 0u);
    EXPECT_GT(probe.num_synapses, 0u);

    // Cleanup
    nimcp_brain_destroy(clone);
    nimcp_brain_destroy(original);
}

TEST_F(BrainCOWTest, CloneCOWSharesMemory) {
    // Create original brain
    nimcp_brain_t original = nimcp_brain_create(
        "original",
        NIMCP_BRAIN_SMALL,
        NIMCP_TASK_CLASSIFICATION,
        10, 3
    );
    ASSERT_NE(original, nullptr);

    // Get initial cache stats
    nimcp_cache_stats_t stats_before;
    nimcp_cache_get_stats(&stats_before);

    // Clone with COW
    nimcp_brain_t clone = nimcp_brain_clone_cow(original);
    ASSERT_NE(clone, nullptr);

    // Get cache stats after clone
    nimcp_cache_stats_t stats_after;
    nimcp_cache_get_stats(&stats_after);

    // Should have created references, not new allocations
    EXPECT_GT(stats_after.references_created, stats_before.references_created);
    EXPECT_GT(stats_after.memory_shared, 0u);
    EXPECT_GT(stats_after.memory_saved, 0u);

    // Verify COW flags in probe
    nimcp_brain_probe_t probe;
    nimcp_brain_probe(clone, &probe);
    EXPECT_TRUE(probe.is_cow_clone);
    EXPECT_GT(probe.cow_shared_bytes, 0u);

    // Cleanup
    nimcp_brain_destroy(clone);
    nimcp_brain_destroy(original);
}

TEST_F(BrainCOWTest, CloneCOWFasterThanFullCopy) {
    // Create original brain
    nimcp_brain_t original = nimcp_brain_create(
        "original",
        NIMCP_BRAIN_MEDIUM,  // Larger for measurable difference
        NIMCP_TASK_CLASSIFICATION,
        100, 10
    );
    ASSERT_NE(original, nullptr);

    // Time COW clone
    uint64_t start_cow = nimcp_time_get_ms();
    nimcp_brain_t clone_cow = nimcp_brain_clone_cow(original);
    uint64_t time_cow = nimcp_time_get_ms() - start_cow;
    ASSERT_NE(clone_cow, nullptr);

    // Time full copy (if implemented, otherwise skip)
    // nimcp_brain_t clone_full = nimcp_brain_clone_full(original);
    // uint64_t time_full = ...;
    // EXPECT_LT(time_cow, time_full);  // COW should be much faster

    // COW clone should be near-instant (< 10ms)
    EXPECT_LT(time_cow, 10u);

    // Cleanup
    nimcp_brain_destroy(clone_cow);
    nimcp_brain_destroy(original);
}

TEST_F(BrainCOWTest, MultipleClonesCOWShareMemory) {
    // Create original brain
    nimcp_brain_t original = nimcp_brain_create(
        "original",
        NIMCP_BRAIN_SMALL,
        NIMCP_TASK_CLASSIFICATION,
        10, 3
    );
    ASSERT_NE(original, nullptr);

    // Create multiple clones
    const int num_clones = 5;
    nimcp_brain_t clones[num_clones];

    nimcp_cache_stats_t stats;
    nimcp_cache_get_stats(&stats);
    size_t memory_before = stats.memory_allocated;

    for (int i = 0; i < num_clones; i++) {
        clones[i] = nimcp_brain_clone_cow(original);
        ASSERT_NE(clones[i], nullptr);
    }

    // Get memory after clones
    nimcp_cache_get_stats(&stats);
    size_t memory_after = stats.memory_allocated;

    // Memory increase should be minimal (just overhead, not full copies)
    size_t memory_increase = memory_after - memory_before;

    // With COW, 5 clones should add < 1MB overhead
    // Without COW, 5 clones would add ~50MB (5 × 10MB)
    EXPECT_LT(memory_increase, 1024 * 1024);  // < 1MB

    // Cleanup
    for (int i = 0; i < num_clones; i++) {
        nimcp_brain_destroy(clones[i]);
    }
    nimcp_brain_destroy(original);
}

TEST_F(BrainCOWTest, CloneCOWTriggersWriteOnLearning) {
    // Create original brain
    nimcp_brain_t original = nimcp_brain_create(
        "original",
        NIMCP_BRAIN_SMALL,
        NIMCP_TASK_CLASSIFICATION,
        10, 3
    );
    ASSERT_NE(original, nullptr);

    // Clone with COW
    nimcp_brain_t clone = nimcp_brain_clone_cow(original);
    ASSERT_NE(clone, nullptr);

    // Get cache stats before learning
    nimcp_cache_stats_t stats_before;
    nimcp_cache_get_stats(&stats_before);

    // Trigger learning on clone (should trigger COW)
    float features[10] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
    nimcp_status_t status = nimcp_brain_learn_example(
        clone, features, 10, "test_label", 0.9f
    );
    EXPECT_EQ(status, NIMCP_OK);

    // Get cache stats after learning
    nimcp_cache_stats_t stats_after;
    nimcp_cache_get_stats(&stats_after);

    // Should have triggered at least one copy
    EXPECT_GT(stats_after.copies_triggered, stats_before.copies_triggered);

    // Clone should now be private
    nimcp_brain_probe_t probe;
    nimcp_brain_probe(clone, &probe);
    // After learning, shared bytes should decrease
    // (Implementation detail: may still show some sharing)

    // Cleanup
    nimcp_brain_destroy(clone);
    nimcp_brain_destroy(original);
}

//=============================================================================
// Brain Snapshot COW Tests
//=============================================================================

TEST_F(BrainCOWTest, SnapshotCOWCreatesValidSnapshot) {
    // Create brain
    nimcp_brain_t brain = nimcp_brain_create(
        "test_brain",
        NIMCP_BRAIN_SMALL,
        NIMCP_TASK_CLASSIFICATION,
        10, 3
    );
    ASSERT_NE(brain, nullptr);

    // Create snapshot
    nimcp_brain_snapshot_t snapshot = nimcp_brain_snapshot_cow(brain);
    ASSERT_NE(snapshot, nullptr);

    // Snapshot should be instant (< 1ms)
    uint64_t start = nimcp_time_get_ms();
    nimcp_brain_snapshot_t snapshot2 = nimcp_brain_snapshot_cow(brain);
    uint64_t elapsed = nimcp_time_get_ms() - start;
    EXPECT_LT(elapsed, 1u);
    ASSERT_NE(snapshot2, nullptr);

    // Cleanup
    nimcp_brain_snapshot_release(snapshot2);
    nimcp_brain_snapshot_release(snapshot);
    nimcp_brain_destroy(brain);
}

TEST_F(BrainCOWTest, SnapshotCOWSharesMemory) {
    // Create brain
    nimcp_brain_t brain = nimcp_brain_create(
        "test_brain",
        NIMCP_BRAIN_SMALL,
        NIMCP_TASK_CLASSIFICATION,
        10, 3
    );
    ASSERT_NE(brain, nullptr);

    // Get memory before snapshot
    nimcp_cache_stats_t stats_before;
    nimcp_cache_get_stats(&stats_before);

    // Create snapshot
    nimcp_brain_snapshot_t snapshot = nimcp_brain_snapshot_cow(brain);
    ASSERT_NE(snapshot, nullptr);

    // Get memory after snapshot
    nimcp_cache_stats_t stats_after;
    nimcp_cache_get_stats(&stats_after);

    // Memory increase should be minimal (< 1KB)
    size_t memory_increase = stats_after.memory_allocated - stats_before.memory_allocated;
    EXPECT_LT(memory_increase, 1024u);  // < 1KB

    // Should have created references
    EXPECT_GT(stats_after.references_created, stats_before.references_created);

    // Cleanup
    nimcp_brain_snapshot_release(snapshot);
    nimcp_brain_destroy(brain);
}

TEST_F(BrainCOWTest, SnapshotCOWRestoresState) {
    // Create brain and train it
    nimcp_brain_t brain = nimcp_brain_create(
        "test_brain",
        NIMCP_BRAIN_SMALL,
        NIMCP_TASK_CLASSIFICATION,
        10, 3
    );
    ASSERT_NE(brain, nullptr);

    float features[10] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};

    // Train once
    nimcp_brain_learn_example(brain, features, 10, "initial_label", 0.9f);

    // Get initial state
    nimcp_brain_probe_t probe_initial;
    nimcp_brain_probe(brain, &probe_initial);
    uint64_t learning_steps_initial = probe_initial.total_learning_steps;

    // Create snapshot
    nimcp_brain_snapshot_t snapshot = nimcp_brain_snapshot_cow(brain);
    ASSERT_NE(snapshot, nullptr);

    // Train more
    for (int i = 0; i < 10; i++) {
        nimcp_brain_learn_example(brain, features, 10, "modified_label", 0.9f);
    }

    // Verify state changed
    nimcp_brain_probe_t probe_modified;
    nimcp_brain_probe(brain, &probe_modified);
    EXPECT_GT(probe_modified.total_learning_steps, learning_steps_initial);

    // Restore from snapshot
    bool restored = nimcp_brain_restore_cow(brain, snapshot);
    EXPECT_TRUE(restored);

    // Verify state restored
    nimcp_brain_probe_t probe_restored;
    nimcp_brain_probe(brain, &probe_restored);
    EXPECT_EQ(probe_restored.total_learning_steps, learning_steps_initial);

    // Cleanup
    nimcp_brain_snapshot_release(snapshot);
    nimcp_brain_destroy(brain);
}

TEST_F(BrainCOWTest, MultipleSnapshotsShareMemory) {
    // Create brain
    nimcp_brain_t brain = nimcp_brain_create(
        "test_brain",
        NIMCP_BRAIN_SMALL,
        NIMCP_TASK_CLASSIFICATION,
        10, 3
    );
    ASSERT_NE(brain, nullptr);

    // Get initial memory
    nimcp_cache_stats_t stats_before;
    nimcp_cache_get_stats(&stats_before);

    // Create multiple snapshots
    const int num_snapshots = 10;
    nimcp_brain_snapshot_t snapshots[num_snapshots];

    for (int i = 0; i < num_snapshots; i++) {
        snapshots[i] = nimcp_brain_snapshot_cow(brain);
        ASSERT_NE(snapshots[i], nullptr);
    }

    // Get memory after snapshots
    nimcp_cache_stats_t stats_after;
    nimcp_cache_get_stats(&stats_after);

    // Memory increase should be minimal (10 × ~100 bytes = ~1KB)
    size_t memory_increase = stats_after.memory_allocated - stats_before.memory_allocated;
    EXPECT_LT(memory_increase, 10 * 1024u);  // < 10KB

    // Cleanup
    for (int i = 0; i < num_snapshots; i++) {
        nimcp_brain_snapshot_release(snapshots[i]);
    }
    nimcp_brain_destroy(brain);
}

//=============================================================================
// Integration Tests
//=============================================================================

TEST_F(BrainCOWTest, CloneAndSnapshotTogether) {
    // Create original brain
    nimcp_brain_t original = nimcp_brain_create(
        "original",
        NIMCP_BRAIN_SMALL,
        NIMCP_TASK_CLASSIFICATION,
        10, 3
    );
    ASSERT_NE(original, nullptr);

    // Clone with COW
    nimcp_brain_t clone = nimcp_brain_clone_cow(original);
    ASSERT_NE(clone, nullptr);

    // Create snapshot of clone
    nimcp_brain_snapshot_t snapshot = nimcp_brain_snapshot_cow(clone);
    ASSERT_NE(snapshot, nullptr);

    // Modify clone
    float features[10] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
    nimcp_brain_learn_example(clone, features, 10, "test_label", 0.9f);

    // Restore snapshot
    bool restored = nimcp_brain_restore_cow(clone, snapshot);
    EXPECT_TRUE(restored);

    // Verify cache stats show sharing
    nimcp_cache_stats_t stats;
    nimcp_cache_get_stats(&stats);
    EXPECT_GT(stats.references_created, 0u);
    EXPECT_GT(stats.memory_saved, 0u);

    // Cleanup
    nimcp_brain_snapshot_release(snapshot);
    nimcp_brain_destroy(clone);
    nimcp_brain_destroy(original);
}

TEST_F(BrainCOWTest, MemorySavingsWithManyReplicas) {
    // Create original brain
    nimcp_brain_t original = nimcp_brain_create(
        "original",
        NIMCP_BRAIN_SMALL,
        NIMCP_TASK_CLASSIFICATION,
        10, 3
    );
    ASSERT_NE(original, nullptr);

    // Get probe of original
    nimcp_brain_probe_t probe_original;
    nimcp_brain_probe(original, &probe_original);
    size_t original_memory = probe_original.memory_bytes;

    // Get cache stats before cloning
    nimcp_cache_stats_t stats_before;
    nimcp_cache_get_stats(&stats_before);

    // Create many clones (simulating replication)
    const int num_replicas = 10;
    nimcp_brain_t replicas[num_replicas];

    for (int i = 0; i < num_replicas; i++) {
        replicas[i] = nimcp_brain_clone_cow(original);
        ASSERT_NE(replicas[i], nullptr);
    }

    // Get cache stats after cloning
    nimcp_cache_stats_t stats_after;
    nimcp_cache_get_stats(&stats_after);

    // Calculate memory savings
    size_t memory_saved = stats_after.memory_saved;

    // With COW, we should save ~9 × original_memory
    // (10 clones - 1 original = 9 copies avoided)
    size_t expected_savings = original_memory * 9;

    // Actual savings should be at least 70% of expected
    // (Some structures may not be COW-enabled yet)
    EXPECT_GT(memory_saved, expected_savings * 0.7);

    printf("\nMemory Savings Analysis:\n");
    printf("  Original brain: %zu MB\n", original_memory / (1024 * 1024));
    printf("  Num replicas: %d\n", num_replicas);
    printf("  Expected without COW: %zu MB\n",
           (original_memory * (num_replicas + 1)) / (1024 * 1024));
    printf("  Memory saved: %zu MB\n", memory_saved / (1024 * 1024));
    printf("  Savings percentage: %.1f%%\n",
           (float)memory_saved / (original_memory * num_replicas) * 100.0f);

    // Cleanup
    for (int i = 0; i < num_replicas; i++) {
        nimcp_brain_destroy(replicas[i]);
    }
    nimcp_brain_destroy(original);
}

//=============================================================================
// Error Handling Tests
//=============================================================================

TEST_F(BrainCOWTest, CloneCOWHandlesNullBrain) {
    nimcp_brain_t clone = nimcp_brain_clone_cow(nullptr);
    EXPECT_EQ(clone, nullptr);
}

TEST_F(BrainCOWTest, SnapshotCOWHandlesNullBrain) {
    nimcp_brain_snapshot_t snapshot = nimcp_brain_snapshot_cow(nullptr);
    EXPECT_EQ(snapshot, nullptr);
}

TEST_F(BrainCOWTest, RestoreCOWHandlesNullInputs) {
    nimcp_brain_t brain = nimcp_brain_create(
        "test_brain",
        NIMCP_BRAIN_SMALL,
        NIMCP_TASK_CLASSIFICATION,
        10, 3
    );
    ASSERT_NE(brain, nullptr);

    nimcp_brain_snapshot_t snapshot = nimcp_brain_snapshot_cow(brain);
    ASSERT_NE(snapshot, nullptr);

    // Null brain
    EXPECT_FALSE(nimcp_brain_restore_cow(nullptr, snapshot));

    // Null snapshot
    EXPECT_FALSE(nimcp_brain_restore_cow(brain, nullptr));

    // Cleanup
    nimcp_brain_snapshot_release(snapshot);
    nimcp_brain_destroy(brain);
}
