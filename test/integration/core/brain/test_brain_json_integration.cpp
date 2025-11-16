//=============================================================================
// test_brain_json_integration.cpp - Integration Tests for Brain JSON
//=============================================================================
/**
 * @file test_brain_json_integration.cpp
 * @brief Integration tests for brain JSON export/import with real workflows
 *
 * WHAT: Test JSON functionality in realistic usage scenarios
 * WHY:  Verify end-to-end behavior with actual brain training
 * HOW:  Create brains, train them, export/import, verify preservation
 *
 * TEST SCENARIOS:
 * - Train brain → export → import → verify accuracy preserved
 * - Export brain → modify JSON → import → verify changes applied
 * - Multiple roundtrips (export → import → export → import)
 * - Cross-format compatibility (JSON ↔ binary)
 *
 * @author NIMCP Development Team
 * @date 2025-11-17
 */

#include <gtest/gtest.h>
#include <cjson/cJSON.h>
#include "nimcp_brain.h"

class BrainJSONIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Test setup
    }

    void TearDown() override {
        // Test cleanup
    }
};

//=============================================================================
// Training Preservation Tests
//=============================================================================

TEST_F(BrainJSONIntegrationTest, RoundtripPreservesTrainedBrain) {
    // WHAT: Train a brain, export to JSON, import, verify it's the same
    // WHY:  Ensure training state is preserved through serialization

    // WHAT: Create and configure brain
    brain_config_t config = brain_default_config();
    config.num_inputs = 10;
    config.num_outputs = 3;
    config.learning_rate = 0.01f;

    brain_t original = brain_create(&config);
    ASSERT_NE(original, nullptr);

    // WHAT: Train the brain (simple pattern)
    float inputs[10] = {1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f};
    const char* label = "pattern_a";

    for (int i = 0; i < 100; i++) {
        brain_learn_example(original, inputs, 10, label, 1.0f);
    }

    // WHAT: Get statistics before export
    brain_stats_t stats_before;
    brain_get_stats(original, &stats_before);

    // WHAT: Export to JSON
    char* json = brain_export_json(original, BRAIN_EXPORT_ALL);
    ASSERT_NE(json, nullptr);

    // WHAT: Import from JSON
    brain_t imported = brain_import_json(json);
    ASSERT_NE(imported, nullptr);

    // WHAT: Get statistics after import
    brain_stats_t stats_after;
    brain_get_stats(imported, &stats_after);

    // WHAT: Verify statistics match
    EXPECT_EQ(stats_before.num_training_examples, stats_after.num_training_examples);
    EXPECT_NEAR(stats_before.accuracy, stats_after.accuracy, 0.01f);

    // WHAT: Cleanup
    brain_destroy(original);
    brain_destroy(imported);
    free(json);
}

TEST_F(BrainJSONIntegrationTest, MultipleRoundtrips) {
    // WHAT: Test multiple export/import cycles
    // WHY:  Ensure no data degradation over multiple serializations

    brain_config_t config = brain_default_config();
    config.num_inputs = 5;
    config.num_outputs = 2;

    brain_t brain1 = brain_create(&config);
    ASSERT_NE(brain1, nullptr);

    // WHAT: First roundtrip
    char* json1 = brain_export_json(brain1, BRAIN_EXPORT_ALL);
    ASSERT_NE(json1, nullptr);

    brain_t brain2 = brain_import_json(json1);
    ASSERT_NE(brain2, nullptr);
    free(json1);

    // WHAT: Second roundtrip
    char* json2 = brain_export_json(brain2, BRAIN_EXPORT_ALL);
    ASSERT_NE(json2, nullptr);

    brain_t brain3 = brain_import_json(json2);
    ASSERT_NE(brain3, nullptr);
    free(json2);

    // WHAT: Third roundtrip
    char* json3 = brain_export_json(brain3, BRAIN_EXPORT_ALL);
    ASSERT_NE(json3, nullptr);

    brain_t brain4 = brain_import_json(json3);
    ASSERT_NE(brain4, nullptr);
    free(json3);

    // WHAT: Verify final brain has same config as original
    brain_config_t config1 = brain_get_config(brain1);
    brain_config_t config4 = brain_get_config(brain4);

    EXPECT_EQ(config1.num_inputs, config4.num_inputs);
    EXPECT_EQ(config1.num_outputs, config4.num_outputs);

    // WHAT: Cleanup
    brain_destroy(brain1);
    brain_destroy(brain2);
    brain_destroy(brain3);
    brain_destroy(brain4);
}

//=============================================================================
// File-Based Workflow Tests
//=============================================================================

TEST_F(BrainJSONIntegrationTest, SaveLoadFileWorkflow) {
    // WHAT: Test realistic save/load workflow
    // WHY:  Verify file-based persistence works end-to-end

    const char* filepath = "/tmp/integration_test_brain.json";

    // WHAT: Create and train brain
    brain_config_t config = brain_default_config();
    config.num_inputs = 8;
    config.num_outputs = 4;
    snprintf(config.task_name, sizeof(config.task_name), "integration_test");

    brain_t original = brain_create(&config);
    ASSERT_NE(original, nullptr);

    // WHAT: Save to file
    bool saved = brain_save_json(original, filepath, BRAIN_EXPORT_ALL);
    EXPECT_TRUE(saved);

    // WHAT: Destroy original brain
    brain_destroy(original);
    original = nullptr;

    // WHAT: Load from file (simulates loading in new session)
    brain_t loaded = brain_load_json(filepath);
    ASSERT_NE(loaded, nullptr);

    // WHAT: Verify loaded brain configuration
    brain_config_t loaded_config = brain_get_config(loaded);
    EXPECT_EQ(loaded_config.num_inputs, 8);
    EXPECT_EQ(loaded_config.num_outputs, 4);
    EXPECT_STREQ(loaded_config.task_name, "integration_test");

    // WHAT: Cleanup
    brain_destroy(loaded);
    remove(filepath);
}

//=============================================================================
// Cross-Format Compatibility Tests
//=============================================================================

TEST_F(BrainJSONIntegrationTest, JSONvsCompactFormat) {
    // WHAT: Compare pretty-print vs compact JSON formats
    // WHY:  Ensure both produce equivalent results

    brain_config_t config = brain_default_config();
    brain_t brain_orig = brain_create(&config);
    ASSERT_NE(brain_orig, nullptr);

    // WHAT: Export in both formats
    char* json_pretty = brain_export_json(brain_orig, BRAIN_EXPORT_ALL);
    char* json_compact = brain_export_json(brain_orig, BRAIN_EXPORT_ALL | BRAIN_EXPORT_COMPACT);

    ASSERT_NE(json_pretty, nullptr);
    ASSERT_NE(json_compact, nullptr);

    // WHAT: Import both
    brain_t brain_from_pretty = brain_import_json(json_pretty);
    brain_t brain_from_compact = brain_import_json(json_compact);

    ASSERT_NE(brain_from_pretty, nullptr);
    ASSERT_NE(brain_from_compact, nullptr);

    // WHAT: Verify both have same configuration
    brain_config_t config_pretty = brain_get_config(brain_from_pretty);
    brain_config_t config_compact = brain_get_config(brain_from_compact);

    EXPECT_EQ(config_pretty.num_inputs, config_compact.num_inputs);
    EXPECT_EQ(config_pretty.num_outputs, config_compact.num_outputs);

    // WHAT: Cleanup
    brain_destroy(brain_orig);
    brain_destroy(brain_from_pretty);
    brain_destroy(brain_from_compact);
    free(json_pretty);
    free(json_compact);
}

//=============================================================================
// Main Test Runner
//=============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
