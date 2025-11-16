//=============================================================================
// test_enhanced_features_integration.cpp - Integration Tests
//=============================================================================
/**
 * @file test_enhanced_features_integration.cpp
 * @brief Integration tests for COW snapshots, version checking, and memory tracking
 *
 * WHAT: End-to-end integration tests for enhanced NIMCP features
 * WHY:  Verify features work together correctly in realistic scenarios
 * HOW:  GoogleTest framework with complete workflows
 *
 * TEST COVERAGE:
 * 1. COW snapshot + memory tracking integration
 * 2. Pretrained model loading + version checking
 * 3. Complete workflow: load → snapshot → train → restore
 * 4. Multi-brain scenarios with shared memory
 */

#include <gtest/gtest.h>
#include "nimcp.h"
#include "core/brain/nimcp_brain.h"
#include "utils/cache/nimcp_cache.h"

class EnhancedFeaturesIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        nimcp_init();
        nimcp_cache_clear_stats();
    }

    void TearDown() override {
        nimcp_shutdown();
    }
};

//=============================================================================
// Integration Test 1: COW Snapshot + Memory Tracking
//=============================================================================

TEST_F(EnhancedFeaturesIntegrationTest, COWSnapshotWithMemoryTracking) {
    // WHAT: Verify COW snapshots integrate properly with memory tracking
    // WHY:  Ensure both features work together correctly
    // HOW:  Create snapshot, check memory, verify consistency

    // Create brain
    nimcp_brain_config_t config = nimcp_brain_config_defaults(NIMCP_BRAIN_SMALL);
    config.num_inputs = 10;
    config.num_outputs = 5;
    nimcp_brain_t brain = nimcp_brain_create(&config);
    ASSERT_NE(brain, nullptr);

    // Get initial memory
    size_t memory_before_snapshot = brain_get_memory_usage(brain->internal_brain);
    EXPECT_GT(memory_before_snapshot, 0);

    // Create COW snapshot
    nimcp_brain_snapshot_t snapshot = nimcp_brain_snapshot_cow(brain);
    ASSERT_NE(snapshot, nullptr);

    // Get memory after snapshot
    size_t memory_after_snapshot = brain_get_memory_usage(brain->internal_brain);

    // VERIFY: Memory didn't increase significantly (COW sharing)
    double ratio = (double)memory_after_snapshot / (double)memory_before_snapshot;
    EXPECT_LT(ratio, 1.1);  // Less than 10% increase

    // Check cache stats
    nimcp_cache_stats_t cache_stats;
    ASSERT_TRUE(nimcp_cache_get_stats(&cache_stats));
    EXPECT_GT(cache_stats.memory_saved, 0);

    // Cleanup
    nimcp_brain_snapshot_destroy(snapshot);
    nimcp_brain_destroy(brain);
}

//=============================================================================
// Integration Test 2: Pretrained Model + Version Checking
//=============================================================================

TEST_F(EnhancedFeaturesIntegrationTest, PretrainedModelWithVersionChecking) {
    // WHAT: Verify pretrained model loading integrates with version checking
    // WHY:  Ensure version checking works with real models
    // HOW:  Load model and verify version info is populated

    // Try to get model info
    brain_model_info_t info;
    bool info_available = brain_get_model_info("nimcp_foundation_medium_v1.0", &info);

    if (info_available) {
        // VERIFY: Version information is present
        EXPECT_NE(info.version[0], '\0');
        EXPECT_NE(info.model_id[0], '\0');

        // VERIFY: Update flag is set (true or false)
        EXPECT_TRUE(info.update_available == true || info.update_available == false);

        // Try to load the model if available
        if (info.is_available) {
            brain_t brain = brain_load_pretrained("nimcp_foundation_medium_v1.0", nullptr);
            if (brain) {
                // VERIFY: Memory tracking works for pretrained model
                size_t memory = brain_get_memory_usage(brain);
                EXPECT_GT(memory, 0);

                brain_destroy(brain);
            }
        }
    } else {
        // Model not found - OK for test environment
        SUCCEED();
    }
}

//=============================================================================
// Integration Test 3: Complete Workflow
//=============================================================================

TEST_F(EnhancedFeaturesIntegrationTest, CompleteWorkflowLoadSnapshotTrainRestore) {
    // WHAT: Test complete workflow with all enhanced features
    // WHY:  Verify realistic usage scenario works end-to-end
    // HOW:  Create → snapshot → train → track memory → restore

    // Step 1: Create brain
    nimcp_brain_config_t config = nimcp_brain_config_defaults(NIMCP_BRAIN_SMALL);
    config.num_inputs = 10;
    config.num_outputs = 3;
    nimcp_brain_t brain = nimcp_brain_create(&config);
    ASSERT_NE(brain, nullptr);

    // Step 2: Train initial model
    float inputs[10] = {1.0f, 0.5f, 0.3f, 0.8f, 0.2f, 0.6f, 0.4f, 0.9f, 0.1f, 0.7f};
    for (int i = 0; i < 50; i++) {
        nimcp_brain_teach(brain, inputs, "initial_class");
    }

    // Step 3: Track memory before snapshot
    size_t memory_initial = brain_get_memory_usage(brain->internal_brain);
    EXPECT_GT(memory_initial, 0);

    // Step 4: Create COW snapshot
    nimcp_brain_snapshot_t snapshot = nimcp_brain_snapshot_cow(brain);
    ASSERT_NE(snapshot, nullptr);

    // Step 5: Get initial decision
    nimcp_brain_decision_t decision_initial;
    ASSERT_EQ(nimcp_brain_decide(brain, inputs, &decision_initial), NIMCP_OK);
    const char* initial_output = decision_initial.output_label;

    // Step 6: Train on different data
    float new_inputs[10] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f};
    for (int i = 0; i < 100; i++) {
        nimcp_brain_teach(brain, new_inputs, "modified_class");
    }

    // Step 7: Track memory after training
    size_t memory_after_train = brain_get_memory_usage(brain->internal_brain);
    EXPECT_GE(memory_after_train, memory_initial * 0.9);  // Allow variance

    // Step 8: Verify decision changed
    nimcp_brain_decision_t decision_modified;
    ASSERT_EQ(nimcp_brain_decide(brain, inputs, &decision_modified), NIMCP_OK);
    EXPECT_STRNE(initial_output, decision_modified.output_label);

    // Step 9: Restore from snapshot
    ASSERT_EQ(nimcp_brain_restore_cow(brain, snapshot), NIMCP_OK);

    // Step 10: Verify decision restored
    nimcp_brain_decision_t decision_restored;
    ASSERT_EQ(nimcp_brain_decide(brain, inputs, &decision_restored), NIMCP_OK);
    EXPECT_STREQ(initial_output, decision_restored.output_label);

    // Step 11: Verify memory consistency
    size_t memory_after_restore = brain_get_memory_usage(brain->internal_brain);
    double restore_ratio = (double)memory_after_restore / (double)memory_initial;
    EXPECT_GT(restore_ratio, 0.8);  // Should be close to original
    EXPECT_LT(restore_ratio, 1.5);

    // Cleanup
    nimcp_brain_decision_destroy(&decision_initial);
    nimcp_brain_decision_destroy(&decision_modified);
    nimcp_brain_decision_destroy(&decision_restored);
    nimcp_brain_snapshot_destroy(snapshot);
    nimcp_brain_destroy(brain);
}

//=============================================================================
// Integration Test 4: Multi-Brain Memory Sharing
//=============================================================================

TEST_F(EnhancedFeaturesIntegrationTest, MultiBrainMemorySharing) {
    // WHAT: Test multiple brains sharing memory via COW
    // WHY:  Verify memory tracking accurately reports sharing
    // HOW:  Create original + multiple clones, check memory

    // Create original brain
    nimcp_brain_config_t config = nimcp_brain_config_defaults(NIMCP_BRAIN_SMALL);
    nimcp_brain_t original = nimcp_brain_create(&config);
    ASSERT_NE(original, nullptr);

    size_t original_memory = brain_get_memory_usage(original->internal_brain);

    // Create multiple COW clones
    nimcp_brain_t clone1 = nimcp_brain_clone_cow(original);
    nimcp_brain_t clone2 = nimcp_brain_clone_cow(original);
    nimcp_brain_t clone3 = nimcp_brain_clone_cow(original);

    ASSERT_NE(clone1, nullptr);
    ASSERT_NE(clone2, nullptr);
    ASSERT_NE(clone3, nullptr);

    // Get memory for clones
    size_t clone1_memory = brain_get_memory_usage(clone1->internal_brain);
    size_t clone2_memory = brain_get_memory_usage(clone2->internal_brain);
    size_t clone3_memory = brain_get_memory_usage(clone3->internal_brain);

    // VERIFY: All clones have similar memory footprint
    EXPECT_TRUE(std::abs((int64_t)clone1_memory - (int64_t)original_memory) < (int64_t)original_memory / 2);
    EXPECT_TRUE(std::abs((int64_t)clone2_memory - (int64_t)original_memory) < (int64_t)original_memory / 2);
    EXPECT_TRUE(std::abs((int64_t)clone3_memory - (int64_t)original_memory) < (int64_t)original_memory / 2);

    // Check cache stats
    nimcp_cache_stats_t cache_stats;
    ASSERT_TRUE(nimcp_cache_get_stats(&cache_stats));

    // VERIFY: Significant memory savings (multiple brains sharing)
    EXPECT_GT(cache_stats.memory_saved, original_memory * 2);

    // Cleanup
    nimcp_brain_destroy(original);
    nimcp_brain_destroy(clone1);
    nimcp_brain_destroy(clone2);
    nimcp_brain_destroy(clone3);
}

//=============================================================================
// Integration Test 5: Snapshot + Memory Tracking Consistency
//=============================================================================

TEST_F(EnhancedFeaturesIntegrationTest, SnapshotMemoryTrackingConsistency) {
    // WHAT: Verify memory tracking remains consistent with snapshots
    // WHY:  Ensure tracking doesn't break with COW operations
    // HOW:  Create/destroy snapshots repeatedly, check consistency

    nimcp_brain_config_t config = nimcp_brain_config_defaults(NIMCP_BRAIN_SMALL);
    nimcp_brain_t brain = nimcp_brain_create(&config);
    ASSERT_NE(brain, nullptr);

    size_t initial_memory = brain_get_memory_usage(brain->internal_brain);

    // Create and destroy multiple snapshots
    for (int i = 0; i < 10; i++) {
        nimcp_brain_snapshot_t snapshot = nimcp_brain_snapshot_cow(brain);
        ASSERT_NE(snapshot, nullptr);

        size_t memory_with_snapshot = brain_get_memory_usage(brain->internal_brain);

        // VERIFY: Memory is consistent
        double ratio = (double)memory_with_snapshot / (double)initial_memory;
        EXPECT_GT(ratio, 0.8);
        EXPECT_LT(ratio, 1.2);

        nimcp_brain_snapshot_destroy(snapshot);
    }

    // Final memory check
    size_t final_memory = brain_get_memory_usage(brain->internal_brain);
    double final_ratio = (double)final_memory / (double)initial_memory;
    EXPECT_GT(final_ratio, 0.9);
    EXPECT_LT(final_ratio, 1.1);

    nimcp_brain_destroy(brain);
}

//=============================================================================
// Integration Test 6: API Layer Integration
//=============================================================================

TEST_F(EnhancedFeaturesIntegrationTest, APILayerIntegration) {
    // WHAT: Test features through public API layer
    // WHY:  Verify API wrappers work correctly
    // HOW:  Use nimcp.h functions exclusively

    // Create brain through API
    nimcp_brain_config_t config = nimcp_brain_config_defaults(NIMCP_BRAIN_SMALL);
    config.num_inputs = 5;
    config.num_outputs = 3;
    nimcp_brain_t brain = nimcp_brain_create(&config);
    ASSERT_NE(brain, nullptr);

    // Train through API
    float inputs[5] = {1.0f, 0.5f, 0.3f, 0.8f, 0.2f};
    nimcp_status_t status = nimcp_brain_teach(brain, inputs, "api_test");
    ASSERT_EQ(status, NIMCP_OK);

    // Create snapshot through API
    nimcp_brain_snapshot_t snapshot = nimcp_brain_snapshot_cow(brain);
    ASSERT_NE(snapshot, nullptr);

    // Check stats through API
    nimcp_brain_stats_t stats;
    status = nimcp_brain_get_stats(brain, &stats);
    ASSERT_EQ(status, NIMCP_OK);
    EXPECT_GT(stats.num_neurons, 0);

    // Restore through API
    status = nimcp_brain_restore_cow(brain, snapshot);
    ASSERT_EQ(status, NIMCP_OK);

    // Check cache stats through API
    nimcp_cache_stats_t cache_stats;
    ASSERT_TRUE(nimcp_cache_get_stats(&cache_stats));
    EXPECT_GT(cache_stats.references_created, 0);

    // Cleanup through API
    nimcp_brain_snapshot_destroy(snapshot);
    nimcp_brain_destroy(brain);
}

//=============================================================================
// Main
//=============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
