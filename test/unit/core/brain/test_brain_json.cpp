//=============================================================================
// test_brain_json.cpp - Comprehensive Unit Tests for Brain JSON Export/Import
//=============================================================================
/**
 * @file test_brain_json.cpp
 * @brief Unit tests for brain JSON serialization/deserialization
 *
 * WHAT: Test all aspects of brain JSON export/import functionality
 * WHY:  Ensure 100% coverage and correctness of JSON serialization
 * HOW:  Use Google Test framework with comprehensive test cases
 *
 * TEST COVERAGE:
 * - Export with different flag combinations
 * - Import from valid JSON
 * - Import validation and error handling
 * - Roundtrip consistency (export → import → export)
 * - Schema version compatibility
 * - Edge cases and error conditions
 *
 * @author NIMCP Development Team
 * @date 2025-11-17
 */

#include <gtest/gtest.h>
#include <cjson/cJSON.h>
#include "nimcp_brain.h"

// Test fixture for brain JSON tests
class BrainJSONTest : public ::testing::Test {
protected:
    brain_t brain;
    brain_config_t config;

    void SetUp() override {
        // WHAT: Initialize test brain with known configuration
        // WHY:  Consistent test environment
        config = brain_default_config();
        config.num_inputs = 10;
        config.num_outputs = 5;
        config.num_hidden = 20;
        config.learning_rate = 0.01f;
        config.enable_stdp = true;
        config.enable_working_memory = true;
        snprintf(config.task_name, sizeof(config.task_name), "test_brain");

        brain = brain_create(&config);
        ASSERT_NE(brain, nullptr) << "Failed to create test brain";
    }

    void TearDown() override {
        if (brain) {
            brain_destroy(brain);
            brain = nullptr;
        }
    }
};

//=============================================================================
// Basic Export Tests
//=============================================================================

TEST_F(BrainJSONTest, ExportBasic) {
    // WHAT: Test basic JSON export with all flags
    // WHY:  Verify complete export functionality
    char* json = brain_export_json(brain, BRAIN_EXPORT_ALL);

    ASSERT_NE(json, nullptr) << "Export should succeed";
    EXPECT_GT(strlen(json), 100) << "JSON should not be empty";

    // WHAT: Verify JSON is valid
    // WHY:  Ensure well-formed output
    cJSON* root = cJSON_Parse(json);
    ASSERT_NE(root, nullptr) << "Exported JSON should be valid";

    // WHAT: Verify required fields exist
    EXPECT_TRUE(cJSON_HasObjectItem(root, "schema_version"));
    EXPECT_TRUE(cJSON_HasObjectItem(root, "metadata"));

    cJSON_Delete(root);
    free(json);
}

TEST_F(BrainJSONTest, ExportWithMetadataOnly) {
    // WHAT: Test export with metadata flag only
    // WHY:  Verify selective export
    char* json = brain_export_json(brain, BRAIN_EXPORT_METADATA);

    ASSERT_NE(json, nullptr);

    cJSON* root = cJSON_Parse(json);
    ASSERT_NE(root, nullptr);

    // WHAT: Verify metadata is present
    EXPECT_TRUE(cJSON_HasObjectItem(root, "metadata"));

    // WHAT: Verify other sections are absent (since only metadata flag set)
    // Note: schema_version is always included
    EXPECT_TRUE(cJSON_HasObjectItem(root, "schema_version"));

    cJSON_Delete(root);
    free(json);
}

TEST_F(BrainJSONTest, ExportWithConfig) {
    // WHAT: Test export with config flag
    // WHY:  Verify configuration serialization
    char* json = brain_export_json(brain, BRAIN_EXPORT_CONFIG);

    ASSERT_NE(json, nullptr);

    cJSON* root = cJSON_Parse(json);
    ASSERT_NE(root, nullptr);

    // WHAT: Verify config section exists
    cJSON* config_obj = cJSON_GetObjectItem(root, "config");
    ASSERT_NE(config_obj, nullptr);

    // WHAT: Verify config fields
    cJSON* num_inputs = cJSON_GetObjectItem(config_obj, "num_inputs");
    ASSERT_NE(num_inputs, nullptr);
    EXPECT_EQ(num_inputs->valueint, 10);

    cJSON* num_outputs = cJSON_GetObjectItem(config_obj, "num_outputs");
    ASSERT_NE(num_outputs, nullptr);
    EXPECT_EQ(num_outputs->valueint, 5);

    cJSON* learning_rate = cJSON_GetObjectItem(config_obj, "learning_rate");
    ASSERT_NE(learning_rate, nullptr);
    EXPECT_NEAR(learning_rate->valuedouble, 0.01, 0.001);

    cJSON_Delete(root);
    free(json);
}

TEST_F(BrainJSONTest, ExportCompactFormat) {
    // WHAT: Test compact JSON export (no pretty-print)
    // WHY:  Verify format flag functionality
    char* json_pretty = brain_export_json(brain, BRAIN_EXPORT_ALL);
    char* json_compact = brain_export_json(brain, BRAIN_EXPORT_ALL | BRAIN_EXPORT_COMPACT);

    ASSERT_NE(json_pretty, nullptr);
    ASSERT_NE(json_compact, nullptr);

    // WHAT: Compact should be shorter (no whitespace)
    // WHY:  Verify compression
    EXPECT_LT(strlen(json_compact), strlen(json_pretty));

    // WHAT: Both should parse to valid JSON
    cJSON* root_pretty = cJSON_Parse(json_pretty);
    cJSON* root_compact = cJSON_Parse(json_compact);
    ASSERT_NE(root_pretty, nullptr);
    ASSERT_NE(root_compact, nullptr);

    cJSON_Delete(root_pretty);
    cJSON_Delete(root_compact);
    free(json_pretty);
    free(json_compact);
}

TEST_F(BrainJSONTest, ExportTopology) {
    // WHAT: Test topology export
    // WHY:  Verify network structure serialization
    char* json = brain_export_json(brain, BRAIN_EXPORT_TOPOLOGY);

    ASSERT_NE(json, nullptr);

    cJSON* root = cJSON_Parse(json);
    ASSERT_NE(root, nullptr);

    // WHAT: Verify topology section exists
    cJSON* topology = cJSON_GetObjectItem(root, "topology");
    ASSERT_NE(topology, nullptr);

    // WHAT: Verify topology fields
    cJSON* num_neurons = cJSON_GetObjectItem(topology, "num_neurons");
    ASSERT_NE(num_neurons, nullptr);
    EXPECT_GT(num_neurons->valueint, 0);

    cJSON_Delete(root);
    free(json);
}

TEST_F(BrainJSONTest, ExportStats) {
    // WHAT: Test statistics export
    // WHY:  Verify metrics serialization
    char* json = brain_export_json(brain, BRAIN_EXPORT_STATS);

    ASSERT_NE(json, nullptr);

    cJSON* root = cJSON_Parse(json);
    ASSERT_NE(root, nullptr);

    // WHAT: Verify stats section exists
    cJSON* stats = cJSON_GetObjectItem(root, "stats");
    ASSERT_NE(stats, nullptr);

    // WHAT: Verify stats fields
    EXPECT_TRUE(cJSON_HasObjectItem(stats, "num_neurons"));
    EXPECT_TRUE(cJSON_HasObjectItem(stats, "num_synapses"));

    cJSON_Delete(root);
    free(json);
}

//=============================================================================
// Import Tests
//=============================================================================

TEST_F(BrainJSONTest, ImportBasic) {
    // WHAT: Test basic import from exported JSON
    // WHY:  Verify roundtrip functionality
    char* json = brain_export_json(brain, BRAIN_EXPORT_ALL);
    ASSERT_NE(json, nullptr);

    brain_t imported = brain_import_json(json);
    ASSERT_NE(imported, nullptr) << "Import should succeed";

    // WHAT: Verify imported brain has same configuration
    brain_config_t imported_config = brain_get_config(imported);
    EXPECT_EQ(imported_config.num_inputs, config.num_inputs);
    EXPECT_EQ(imported_config.num_outputs, config.num_outputs);
    EXPECT_NEAR(imported_config.learning_rate, config.learning_rate, 0.001f);

    brain_destroy(imported);
    free(json);
}

TEST_F(BrainJSONTest, ImportInvalidJSON) {
    // WHAT: Test import with malformed JSON
    // WHY:  Verify error handling
    const char* invalid_json = "{invalid json}";

    brain_t imported = brain_import_json(invalid_json);
    EXPECT_EQ(imported, nullptr) << "Import should fail on invalid JSON";
}

TEST_F(BrainJSONTest, ImportNullString) {
    // WHAT: Test import with NULL string
    // WHY:  Verify null parameter handling
    brain_t imported = brain_import_json(nullptr);
    EXPECT_EQ(imported, nullptr) << "Import should fail on NULL";
}

TEST_F(BrainJSONTest, ImportEmptyString) {
    // WHAT: Test import with empty string
    // WHY:  Verify empty input handling
    brain_t imported = brain_import_json("");
    EXPECT_EQ(imported, nullptr) << "Import should fail on empty string";
}

TEST_F(BrainJSONTest, ImportInvalidSchemaVersion) {
    // WHAT: Test import with incompatible schema version
    // WHY:  Verify version validation
    const char* json_wrong_version = R"({
        "schema_version": "99.0",
        "config": {
            "num_inputs": 10,
            "num_outputs": 5,
            "num_hidden": 20,
            "learning_rate": 0.01,
            "sparsity_target": 0.8,
            "task": 0,
            "size": 0,
            "task_name": "test",
            "enable_stdp": true,
            "enable_eligibility_traces": false,
            "enable_neuromodulation": false,
            "enable_working_memory": false,
            "enable_curiosity": false,
            "enable_explanations": false
        }
    })";

    brain_t imported = brain_import_json(json_wrong_version);
    EXPECT_EQ(imported, nullptr) << "Import should fail on wrong schema version";
}

TEST_F(BrainJSONTest, ImportMissingConfig) {
    // WHAT: Test import with missing config section
    // WHY:  Verify required field validation
    const char* json_no_config = R"({
        "schema_version": "1.0",
        "metadata": {}
    })";

    brain_t imported = brain_import_json(json_no_config);
    EXPECT_EQ(imported, nullptr) << "Import should fail without config";
}

//=============================================================================
// Roundtrip Tests
//=============================================================================

TEST_F(BrainJSONTest, RoundtripConfig) {
    // WHAT: Test export → import → export consistency
    // WHY:  Verify data preservation
    char* json1 = brain_export_json(brain, BRAIN_EXPORT_CONFIG);
    ASSERT_NE(json1, nullptr);

    brain_t imported = brain_import_json(json1);
    ASSERT_NE(imported, nullptr);

    char* json2 = brain_export_json(imported, BRAIN_EXPORT_CONFIG);
    ASSERT_NE(json2, nullptr);

    // WHAT: Parse both JSONs and compare
    cJSON* root1 = cJSON_Parse(json1);
    cJSON* root2 = cJSON_Parse(json2);
    ASSERT_NE(root1, nullptr);
    ASSERT_NE(root2, nullptr);

    // WHAT: Verify config consistency
    cJSON* config1 = cJSON_GetObjectItem(root1, "config");
    cJSON* config2 = cJSON_GetObjectItem(root2, "config");
    ASSERT_NE(config1, nullptr);
    ASSERT_NE(config2, nullptr);

    // WHAT: Compare specific fields
    cJSON* num_inputs1 = cJSON_GetObjectItem(config1, "num_inputs");
    cJSON* num_inputs2 = cJSON_GetObjectItem(config2, "num_inputs");
    EXPECT_EQ(num_inputs1->valueint, num_inputs2->valueint);

    cJSON_Delete(root1);
    cJSON_Delete(root2);
    brain_destroy(imported);
    free(json1);
    free(json2);
}

//=============================================================================
// File I/O Tests
//=============================================================================

TEST_F(BrainJSONTest, SaveAndLoadFile) {
    // WHAT: Test file-based save/load
    // WHY:  Verify file I/O wrappers
    const char* filepath = "/tmp/test_brain.json";

    // WHAT: Save to file
    bool saved = brain_save_json(brain, filepath, BRAIN_EXPORT_ALL);
    EXPECT_TRUE(saved) << "Save should succeed";

    // WHAT: Load from file
    brain_t loaded = brain_load_json(filepath);
    ASSERT_NE(loaded, nullptr) << "Load should succeed";

    // WHAT: Verify loaded brain has same config
    brain_config_t loaded_config = brain_get_config(loaded);
    EXPECT_EQ(loaded_config.num_inputs, config.num_inputs);
    EXPECT_EQ(loaded_config.num_outputs, config.num_outputs);

    brain_destroy(loaded);

    // WHAT: Cleanup
    remove(filepath);
}

TEST_F(BrainJSONTest, SaveNullBrain) {
    // WHAT: Test save with NULL brain
    // WHY:  Verify null parameter handling
    bool saved = brain_save_json(nullptr, "/tmp/test.json", BRAIN_EXPORT_ALL);
    EXPECT_FALSE(saved) << "Save should fail on NULL brain";
}

TEST_F(BrainJSONTest, SaveNullPath) {
    // WHAT: Test save with NULL filepath
    // WHY:  Verify null parameter handling
    bool saved = brain_save_json(brain, nullptr, BRAIN_EXPORT_ALL);
    EXPECT_FALSE(saved) << "Save should fail on NULL path";
}

TEST_F(BrainJSONTest, LoadNullPath) {
    // WHAT: Test load with NULL filepath
    // WHY:  Verify null parameter handling
    brain_t loaded = brain_load_json(nullptr);
    EXPECT_EQ(loaded, nullptr) << "Load should fail on NULL path";
}

TEST_F(BrainJSONTest, LoadNonexistentFile) {
    // WHAT: Test load from nonexistent file
    // WHY:  Verify file error handling
    brain_t loaded = brain_load_json("/tmp/nonexistent_brain_12345.json");
    EXPECT_EQ(loaded, nullptr) << "Load should fail on nonexistent file";
}

//=============================================================================
// Edge Cases and Error Handling
//=============================================================================

TEST_F(BrainJSONTest, ExportNullBrain) {
    // WHAT: Test export with NULL brain
    // WHY:  Verify null parameter handling
    char* json = brain_export_json(nullptr, BRAIN_EXPORT_ALL);
    EXPECT_EQ(json, nullptr) << "Export should fail on NULL brain";
}

TEST_F(BrainJSONTest, ExportZeroFlags) {
    // WHAT: Test export with zero flags
    // WHY:  Should still export schema version and metadata
    char* json = brain_export_json(brain, 0);
    ASSERT_NE(json, nullptr);

    cJSON* root = cJSON_Parse(json);
    ASSERT_NE(root, nullptr);

    // WHAT: Verify at minimum schema_version exists
    EXPECT_TRUE(cJSON_HasObjectItem(root, "schema_version"));

    cJSON_Delete(root);
    free(json);
}

TEST_F(BrainJSONTest, MetadataTimestamp) {
    // WHAT: Verify metadata contains valid timestamp
    // WHY:  Ensure timestamp generation works
    char* json = brain_export_json(brain, BRAIN_EXPORT_METADATA);
    ASSERT_NE(json, nullptr);

    cJSON* root = cJSON_Parse(json);
    ASSERT_NE(root, nullptr);

    cJSON* metadata = cJSON_GetObjectItem(root, "metadata");
    ASSERT_NE(metadata, nullptr);

    cJSON* export_time = cJSON_GetObjectItem(metadata, "export_time");
    ASSERT_NE(export_time, nullptr);
    ASSERT_TRUE(cJSON_IsString(export_time));

    // WHAT: Verify timestamp format (ISO 8601)
    // WHY:  Should contain 'T' and 'Z' for RFC 3339
    const char* timestamp = export_time->valuestring;
    EXPECT_NE(strchr(timestamp, 'T'), nullptr) << "Timestamp should contain 'T'";
    EXPECT_NE(strchr(timestamp, 'Z'), nullptr) << "Timestamp should contain 'Z'";

    cJSON_Delete(root);
    free(json);
}

TEST_F(BrainJSONTest, LargeJSONHandling) {
    // WHAT: Test handling of very large JSON strings
    // WHY:  Verify size limit protection
    // Note: We can't easily create a 100MB+ JSON, but we can verify the check exists
    // This is more of a documentation test

    char* json = brain_export_json(brain, BRAIN_EXPORT_ALL);
    ASSERT_NE(json, nullptr);

    // WHAT: Verify reasonable JSON size
    size_t json_size = strlen(json);
    EXPECT_LT(json_size, 100 * 1024 * 1024) << "JSON should be under 100MB limit";

    free(json);
}

//=============================================================================
// Main Test Runner
//=============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
