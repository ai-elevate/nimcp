/**
 * @file test_api_brain_snapshots.cpp
 * @brief Unit tests for NIMCP API brain snapshot functionality
 *
 * Tests named snapshot API including:
 * - nimcp_brain_snapshot_save()
 * - nimcp_brain_snapshot_restore()
 * - nimcp_brain_snapshot_list()
 * - nimcp_brain_snapshot_delete()
 * - Multiple snapshot management
 * - Snapshot metadata validation
 */

#include <gtest/gtest.h>
#include "nimcp.h"
#include <cstring>
#include <unistd.h>

class BrainSnapshotsTest : public ::testing::Test {
protected:
    nimcp_brain_t brain;

    void SetUp() override {
        // Initialize NIMCP
        nimcp_init();

        // Create test brain
        brain = nimcp_brain_create("snapshot_test_brain", NIMCP_BRAIN_TINY,
                                   NIMCP_TASK_CLASSIFICATION, 10, 5);
        ASSERT_NE(brain, nullptr);

        // Train with some data to have non-trivial state
        float features[10] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f,
                              0.6f, 0.7f, 0.8f, 0.9f, 1.0f};
        nimcp_brain_learn_example(brain, features, 10, "class_a", 0.9f);
    }

    void TearDown() override {
        // Cleanup brain
        if (brain) {
            nimcp_brain_destroy(brain);
            brain = nullptr;
        }

        // Shutdown NIMCP
        nimcp_shutdown();
    }
};

//=============================================================================
// Snapshot Save Tests
//=============================================================================

TEST_F(BrainSnapshotsTest, SnapshotSaveSucceeds) {
    nimcp_status_t status = nimcp_brain_snapshot_save(brain, "test_snapshot", "Test description");

    EXPECT_EQ(status, NIMCP_OK);
}

TEST_F(BrainSnapshotsTest, SnapshotSaveWithNullBrainFails) {
    nimcp_status_t status = nimcp_brain_snapshot_save(nullptr, "test_snapshot", "Test description");

    EXPECT_EQ(status, NIMCP_ERROR_NULL_ARG);
    EXPECT_STREQ(nimcp_get_error(), "Brain handle is NULL");
}

TEST_F(BrainSnapshotsTest, SnapshotSaveWithNullNameFails) {
    nimcp_status_t status = nimcp_brain_snapshot_save(brain, nullptr, "Test description");

    EXPECT_EQ(status, NIMCP_ERROR_NULL_ARG);
    EXPECT_STREQ(nimcp_get_error(), "Snapshot name is NULL");
}

TEST_F(BrainSnapshotsTest, SnapshotSaveWithNullDescriptionSucceeds) {
    nimcp_status_t status = nimcp_brain_snapshot_save(brain, "test_snapshot", nullptr);

    // NULL description should be allowed (optional parameter)
    EXPECT_EQ(status, NIMCP_OK);
}

TEST_F(BrainSnapshotsTest, SnapshotSaveWithEmptyDescriptionSucceeds) {
    nimcp_status_t status = nimcp_brain_snapshot_save(brain, "test_snapshot", "");

    EXPECT_EQ(status, NIMCP_OK);
}

TEST_F(BrainSnapshotsTest, SnapshotSaveWithLongDescription) {
    char long_desc[600];
    memset(long_desc, 'A', 599);
    long_desc[599] = '\0';

    nimcp_status_t status = nimcp_brain_snapshot_save(brain, "test_snapshot", long_desc);

    // Should succeed even with long description (may be truncated)
    EXPECT_EQ(status, NIMCP_OK);
}

TEST_F(BrainSnapshotsTest, SnapshotSaveMultipleTimes) {
    // Save same name multiple times (should overwrite or create new)
    nimcp_status_t status1 = nimcp_brain_snapshot_save(brain, "snapshot1", "First save");
    EXPECT_EQ(status1, NIMCP_OK);

    nimcp_status_t status2 = nimcp_brain_snapshot_save(brain, "snapshot1", "Second save");
    EXPECT_EQ(status2, NIMCP_OK);
}

//=============================================================================
// Snapshot Restore Tests
//=============================================================================

TEST_F(BrainSnapshotsTest, SnapshotRestoreSucceeds) {
    // Save snapshot
    ASSERT_EQ(nimcp_brain_snapshot_save(brain, "restore_test", "Test"), NIMCP_OK);

    // Restore
    nimcp_brain_t restored = nimcp_brain_snapshot_restore(brain, "restore_test");

    EXPECT_NE(restored, nullptr);
    nimcp_brain_destroy(restored);
}

TEST_F(BrainSnapshotsTest, SnapshotRestoreWithNullNameFails) {
    nimcp_brain_t restored = nimcp_brain_snapshot_restore(brain, nullptr);

    EXPECT_EQ(restored, nullptr);
    EXPECT_STREQ(nimcp_get_error(), "Snapshot name is NULL");
}

TEST_F(BrainSnapshotsTest, SnapshotRestoreNonexistentFails) {
    nimcp_brain_t restored = nimcp_brain_snapshot_restore(brain, "nonexistent_snapshot_12345");

    EXPECT_EQ(restored, nullptr);
}

TEST_F(BrainSnapshotsTest, SnapshotRestoreReturnsValidBrain) {
    // Save snapshot
    ASSERT_EQ(nimcp_brain_snapshot_save(brain, "valid_test", "Test"), NIMCP_OK);

    // Restore
    nimcp_brain_t restored = nimcp_brain_snapshot_restore(brain, "valid_test");
    ASSERT_NE(restored, nullptr);

    // Test that restored brain is valid
    nimcp_brain_probe_t probe;
    nimcp_status_t status = nimcp_brain_probe(restored, &probe);
    EXPECT_EQ(status, NIMCP_OK);

    nimcp_brain_destroy(restored);
}

TEST_F(BrainSnapshotsTest, SnapshotRestoreWithNullBrainSucceeds) {
    // Save snapshot first
    ASSERT_EQ(nimcp_brain_snapshot_save(brain, "null_brain_test", "Test"), NIMCP_OK);

    // Restore with NULL brain (should create new brain from snapshot)
    nimcp_brain_t restored = nimcp_brain_snapshot_restore(nullptr, "null_brain_test");

    EXPECT_NE(restored, nullptr);
    nimcp_brain_destroy(restored);
}

TEST_F(BrainSnapshotsTest, SnapshotRestorePreservesState) {
    // Get original state
    nimcp_brain_probe_t original_probe;
    ASSERT_EQ(nimcp_brain_probe(brain, &original_probe), NIMCP_OK);

    // Save snapshot
    ASSERT_EQ(nimcp_brain_snapshot_save(brain, "state_test", "Test"), NIMCP_OK);

    // Modify brain
    float features[10] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                          1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    nimcp_brain_learn_example(brain, features, 10, "class_b", 0.95f);

    // Restore
    nimcp_brain_t restored = nimcp_brain_snapshot_restore(brain, "state_test");
    ASSERT_NE(restored, nullptr);

    // Get restored state
    nimcp_brain_probe_t restored_probe;
    ASSERT_EQ(nimcp_brain_probe(restored, &restored_probe), NIMCP_OK);

    // Verify state matches original
    EXPECT_EQ(original_probe.num_neurons, restored_probe.num_neurons);
    EXPECT_EQ(original_probe.size, restored_probe.size);
    EXPECT_EQ(original_probe.task, restored_probe.task);

    nimcp_brain_destroy(restored);
}

//=============================================================================
// Snapshot List Tests
//=============================================================================

TEST_F(BrainSnapshotsTest, SnapshotListSucceeds) {
    // Create snapshots
    nimcp_brain_snapshot_save(brain, "snapshot1", "First");
    nimcp_brain_snapshot_save(brain, "snapshot2", "Second");

    // List snapshots
    nimcp_brain_snapshot_info_t infos[10];
    uint32_t count = 0;
    nimcp_status_t status = nimcp_brain_snapshot_list(brain, infos, 10, &count);

    EXPECT_EQ(status, NIMCP_OK);
    EXPECT_GT(count, 0);
}

TEST_F(BrainSnapshotsTest, SnapshotListWithNullBrainFails) {
    nimcp_brain_snapshot_info_t infos[10];
    uint32_t count = 0;
    nimcp_status_t status = nimcp_brain_snapshot_list(nullptr, infos, 10, &count);

    EXPECT_EQ(status, NIMCP_ERROR_NULL_ARG);
    EXPECT_STREQ(nimcp_get_error(), "Brain handle is NULL");
}

TEST_F(BrainSnapshotsTest, SnapshotListWithNullInfosFails) {
    uint32_t count = 0;
    nimcp_status_t status = nimcp_brain_snapshot_list(brain, nullptr, 10, &count);

    EXPECT_EQ(status, NIMCP_ERROR_NULL_ARG);
    EXPECT_STREQ(nimcp_get_error(), "Infos array is NULL");
}

TEST_F(BrainSnapshotsTest, SnapshotListWithNullCountSucceeds) {
    nimcp_brain_snapshot_info_t infos[10];
    nimcp_status_t status = nimcp_brain_snapshot_list(brain, infos, 10, nullptr);

    // NULL count should be allowed (optional parameter)
    EXPECT_EQ(status, NIMCP_OK);
}

TEST_F(BrainSnapshotsTest, SnapshotListReturnsCorrectCount) {
    // Create known number of snapshots
    nimcp_brain_snapshot_save(brain, "snap_a", "First");
    nimcp_brain_snapshot_save(brain, "snap_b", "Second");
    nimcp_brain_snapshot_save(brain, "snap_c", "Third");

    nimcp_brain_snapshot_info_t infos[10];
    uint32_t count = 0;
    nimcp_status_t status = nimcp_brain_snapshot_list(brain, infos, 10, &count);

    EXPECT_EQ(status, NIMCP_OK);
    EXPECT_GE(count, 3); // At least our 3 snapshots
}

TEST_F(BrainSnapshotsTest, SnapshotListWithSmallBufferLimitsResults) {
    // Create several snapshots
    nimcp_brain_snapshot_save(brain, "snap1", "One");
    nimcp_brain_snapshot_save(brain, "snap2", "Two");
    nimcp_brain_snapshot_save(brain, "snap3", "Three");

    // Request only 2 snapshots
    nimcp_brain_snapshot_info_t infos[2];
    uint32_t count = 0;
    nimcp_status_t status = nimcp_brain_snapshot_list(brain, infos, 2, &count);

    EXPECT_EQ(status, NIMCP_OK);
    EXPECT_LE(count, 2); // Should not exceed buffer size
}

//=============================================================================
// Snapshot Delete Tests
//=============================================================================

TEST_F(BrainSnapshotsTest, SnapshotDeleteSucceeds) {
    // Create snapshot
    ASSERT_EQ(nimcp_brain_snapshot_save(brain, "delete_test", "Test"), NIMCP_OK);

    // Delete it
    nimcp_status_t status = nimcp_brain_snapshot_delete(brain, "delete_test");

    EXPECT_EQ(status, NIMCP_OK);
}

TEST_F(BrainSnapshotsTest, SnapshotDeleteWithNullBrainFails) {
    nimcp_status_t status = nimcp_brain_snapshot_delete(nullptr, "delete_test");

    EXPECT_EQ(status, NIMCP_ERROR_NULL_ARG);
    EXPECT_STREQ(nimcp_get_error(), "Brain handle is NULL");
}

TEST_F(BrainSnapshotsTest, SnapshotDeleteWithNullNameFails) {
    nimcp_status_t status = nimcp_brain_snapshot_delete(brain, nullptr);

    EXPECT_EQ(status, NIMCP_ERROR_NULL_ARG);
    EXPECT_STREQ(nimcp_get_error(), "Snapshot name is NULL");
}

TEST_F(BrainSnapshotsTest, SnapshotDeleteNonexistentFails) {
    nimcp_status_t status = nimcp_brain_snapshot_delete(brain, "nonexistent_12345");

    EXPECT_EQ(status, NIMCP_ERROR);
}

TEST_F(BrainSnapshotsTest, SnapshotDeleteRemovesSnapshot) {
    // Create and delete snapshot
    ASSERT_EQ(nimcp_brain_snapshot_save(brain, "temp_snapshot", "Temp"), NIMCP_OK);
    ASSERT_EQ(nimcp_brain_snapshot_delete(brain, "temp_snapshot"), NIMCP_OK);

    // Try to restore deleted snapshot (should fail)
    nimcp_brain_t restored = nimcp_brain_snapshot_restore(brain, "temp_snapshot");
    EXPECT_EQ(restored, nullptr);
}

//=============================================================================
// Multiple Snapshots Tests
//=============================================================================

TEST_F(BrainSnapshotsTest, MultipleSnapshotsWork) {
    // Create multiple snapshots at different states
    ASSERT_EQ(nimcp_brain_snapshot_save(brain, "state1", "Initial state"), NIMCP_OK);

    // Modify brain
    float features1[10] = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f,
                           0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
    nimcp_brain_learn_example(brain, features1, 10, "class_b", 0.8f);
    ASSERT_EQ(nimcp_brain_snapshot_save(brain, "state2", "After learning"), NIMCP_OK);

    // Modify again
    float features2[10] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                           1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    nimcp_brain_learn_example(brain, features2, 10, "class_c", 0.9f);
    ASSERT_EQ(nimcp_brain_snapshot_save(brain, "state3", "After more learning"), NIMCP_OK);

    // Verify all snapshots can be listed
    nimcp_brain_snapshot_info_t infos[10];
    uint32_t count = 0;
    ASSERT_EQ(nimcp_brain_snapshot_list(brain, infos, 10, &count), NIMCP_OK);
    EXPECT_GE(count, 3);
}

TEST_F(BrainSnapshotsTest, RestoreFromMultipleSnapshots) {
    // Create snapshots
    ASSERT_EQ(nimcp_brain_snapshot_save(brain, "snap_alpha", "Alpha"), NIMCP_OK);
    ASSERT_EQ(nimcp_brain_snapshot_save(brain, "snap_beta", "Beta"), NIMCP_OK);

    // Restore from first
    nimcp_brain_t restored1 = nimcp_brain_snapshot_restore(brain, "snap_alpha");
    ASSERT_NE(restored1, nullptr);

    // Restore from second
    nimcp_brain_t restored2 = nimcp_brain_snapshot_restore(brain, "snap_beta");
    ASSERT_NE(restored2, nullptr);

    nimcp_brain_destroy(restored1);
    nimcp_brain_destroy(restored2);
}

TEST_F(BrainSnapshotsTest, DeleteOneOfMultipleSnapshots) {
    // Create multiple snapshots
    ASSERT_EQ(nimcp_brain_snapshot_save(brain, "keep1", "Keep this"), NIMCP_OK);
    ASSERT_EQ(nimcp_brain_snapshot_save(brain, "delete_me", "Delete this"), NIMCP_OK);
    ASSERT_EQ(nimcp_brain_snapshot_save(brain, "keep2", "Keep this too"), NIMCP_OK);

    // Delete one
    ASSERT_EQ(nimcp_brain_snapshot_delete(brain, "delete_me"), NIMCP_OK);

    // Verify others still exist
    nimcp_brain_t restored1 = nimcp_brain_snapshot_restore(brain, "keep1");
    EXPECT_NE(restored1, nullptr);

    nimcp_brain_t restored2 = nimcp_brain_snapshot_restore(brain, "keep2");
    EXPECT_NE(restored2, nullptr);

    // Verify deleted one doesn't exist
    nimcp_brain_t restored3 = nimcp_brain_snapshot_restore(brain, "delete_me");
    EXPECT_EQ(restored3, nullptr);

    nimcp_brain_destroy(restored1);
    nimcp_brain_destroy(restored2);
}

//=============================================================================
// Snapshot Metadata Tests
//=============================================================================

TEST_F(BrainSnapshotsTest, SnapshotMetadataContainsName) {
    // Create snapshot
    ASSERT_EQ(nimcp_brain_snapshot_save(brain, "metadata_test", "Test"), NIMCP_OK);

    // List and check metadata
    nimcp_brain_snapshot_info_t infos[10];
    uint32_t count = 0;
    ASSERT_EQ(nimcp_brain_snapshot_list(brain, infos, 10, &count), NIMCP_OK);

    // Find our snapshot
    bool found = false;
    for (uint32_t i = 0; i < count; i++) {
        if (strstr(infos[i].name, "metadata_test") != nullptr) {
            found = true;
            EXPECT_GT(strlen(infos[i].name), 0);
            break;
        }
    }
    EXPECT_TRUE(found);
}

TEST_F(BrainSnapshotsTest, SnapshotMetadataContainsDescription) {
    const char* test_desc = "This is a test description";
    ASSERT_EQ(nimcp_brain_snapshot_save(brain, "desc_test", test_desc), NIMCP_OK);

    nimcp_brain_snapshot_info_t infos[10];
    uint32_t count = 0;
    ASSERT_EQ(nimcp_brain_snapshot_list(brain, infos, 10, &count), NIMCP_OK);

    // Find our snapshot and check description
    bool found = false;
    for (uint32_t i = 0; i < count; i++) {
        if (strstr(infos[i].name, "desc_test") != nullptr) {
            found = true;
            // Description should be set (may be truncated)
            EXPECT_GT(strlen(infos[i].description), 0);
            break;
        }
    }
    EXPECT_TRUE(found);
}

TEST_F(BrainSnapshotsTest, SnapshotMetadataContainsTimestamp) {
    ASSERT_EQ(nimcp_brain_snapshot_save(brain, "timestamp_test", "Test"), NIMCP_OK);

    nimcp_brain_snapshot_info_t infos[10];
    uint32_t count = 0;
    ASSERT_EQ(nimcp_brain_snapshot_list(brain, infos, 10, &count), NIMCP_OK);

    // Find our snapshot and check timestamp
    bool found = false;
    for (uint32_t i = 0; i < count; i++) {
        if (strstr(infos[i].name, "timestamp_test") != nullptr) {
            found = true;
            // Timestamp should be non-zero
            EXPECT_GT(infos[i].timestamp, 0);
            break;
        }
    }
    EXPECT_TRUE(found);
}

TEST_F(BrainSnapshotsTest, SnapshotMetadataTimestampsAreOrdered) {
    // Create snapshots with delays
    ASSERT_EQ(nimcp_brain_snapshot_save(brain, "time1", "First"), NIMCP_OK);
    usleep(10000); // 10ms delay
    ASSERT_EQ(nimcp_brain_snapshot_save(brain, "time2", "Second"), NIMCP_OK);

    nimcp_brain_snapshot_info_t infos[10];
    uint32_t count = 0;
    ASSERT_EQ(nimcp_brain_snapshot_list(brain, infos, 10, &count), NIMCP_OK);

    // Find both snapshots
    uint64_t time1 = 0, time2 = 0;
    for (uint32_t i = 0; i < count; i++) {
        if (strstr(infos[i].name, "time1") != nullptr) {
            time1 = infos[i].timestamp;
        }
        if (strstr(infos[i].name, "time2") != nullptr) {
            time2 = infos[i].timestamp;
        }
    }

    // Second snapshot should have later timestamp
    if (time1 > 0 && time2 > 0) {
        EXPECT_LT(time1, time2);
    }
}
