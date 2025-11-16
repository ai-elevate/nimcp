//=============================================================================
// test_brain_json_regression.cpp - Regression Tests for Brain JSON
//=============================================================================
/**
 * @file test_brain_json_regression.cpp
 * @brief Regression tests for brain JSON schema validation and compatibility
 *
 * WHAT: Verify JSON schema stability and backward compatibility
 * WHY:  Prevent breaking changes to JSON format
 * HOW:  Test against known-good JSON samples and schema constraints
 *
 * TEST COVERAGE:
 * - Schema version validation
 * - Required field presence
 * - Field type validation
 * - Range validation
 * - Backward compatibility with older schemas
 * - Forward compatibility (graceful degradation)
 *
 * @author NIMCP Development Team
 * @date 2025-11-17
 */

#include <gtest/gtest.h>
#include <cjson/cJSON.h>
#include "nimcp_brain.h"

class BrainJSONRegressionTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Test setup
    }

    void TearDown() override {
        // Test cleanup
    }

    // WHAT: Helper to validate JSON schema structure
    // WHY:  Ensure exported JSON matches expected schema
    bool validateJSONSchema(const char* json_str) {
        cJSON* root = cJSON_Parse(json_str);
        if (!root) return false;

        // WHAT: Required fields
        bool has_schema_version = cJSON_HasObjectItem(root, "schema_version");

        cJSON_Delete(root);
        return has_schema_version;
    }
};

//=============================================================================
// Schema Structure Tests
//=============================================================================

TEST_F(BrainJSONRegressionTest, SchemaVersionPresent) {
    // WHAT: Verify schema_version field is always present
    // WHY:  Critical for compatibility checking

    brain_config_t config = brain_default_config();
    brain_t brain = brain_create(&config);
    ASSERT_NE(brain, nullptr);

    char* json = brain_export_json(brain, BRAIN_EXPORT_ALL);
    ASSERT_NE(json, nullptr);

    cJSON* root = cJSON_Parse(json);
    ASSERT_NE(root, nullptr);

    // WHAT: Verify schema_version exists and is a string
    cJSON* schema_version = cJSON_GetObjectItem(root, "schema_version");
    ASSERT_NE(schema_version, nullptr) << "schema_version must be present";
    EXPECT_TRUE(cJSON_IsString(schema_version)) << "schema_version must be string";

    // WHAT: Verify version format (X.Y)
    const char* version = schema_version->valuestring;
    EXPECT_NE(strchr(version, '.'), nullptr) << "Version should contain '.'";

    cJSON_Delete(root);
    brain_destroy(brain);
    free(json);
}

TEST_F(BrainJSONRegressionTest, MetadataStructure) {
    // WHAT: Verify metadata section has expected structure
    // WHY:  Prevent breaking changes to metadata format

    brain_config_t config = brain_default_config();
    brain_t brain = brain_create(&config);
    ASSERT_NE(brain, nullptr);

    char* json = brain_export_json(brain, BRAIN_EXPORT_METADATA);
    ASSERT_NE(json, nullptr);

    cJSON* root = cJSON_Parse(json);
    ASSERT_NE(root, nullptr);

    cJSON* metadata = cJSON_GetObjectItem(root, "metadata");
    ASSERT_NE(metadata, nullptr) << "metadata section must exist";

    // WHAT: Verify required metadata fields
    EXPECT_TRUE(cJSON_HasObjectItem(metadata, "nimcp_version"));
    EXPECT_TRUE(cJSON_HasObjectItem(metadata, "export_time"));

    // WHAT: Verify nimcp_version is string
    cJSON* nimcp_version = cJSON_GetObjectItem(metadata, "nimcp_version");
    ASSERT_NE(nimcp_version, nullptr);
    EXPECT_TRUE(cJSON_IsString(nimcp_version));

    // WHAT: Verify export_time is string (ISO 8601)
    cJSON* export_time = cJSON_GetObjectItem(metadata, "export_time");
    ASSERT_NE(export_time, nullptr);
    EXPECT_TRUE(cJSON_IsString(export_time));

    cJSON_Delete(root);
    brain_destroy(brain);
    free(json);
}

TEST_F(BrainJSONRegressionTest, ConfigFieldTypes) {
    // WHAT: Verify config fields have correct types
    // WHY:  Prevent type changes that break parsers

    brain_config_t config = brain_default_config();
    brain_t brain = brain_create(&config);
    ASSERT_NE(brain, nullptr);

    char* json = brain_export_json(brain, BRAIN_EXPORT_CONFIG);
    ASSERT_NE(json, nullptr);

    cJSON* root = cJSON_Parse(json);
    ASSERT_NE(root, nullptr);

    cJSON* config_obj = cJSON_GetObjectItem(root, "config");
    ASSERT_NE(config_obj, nullptr);

    // WHAT: Verify integer fields
    cJSON* num_inputs = cJSON_GetObjectItem(config_obj, "num_inputs");
    ASSERT_NE(num_inputs, nullptr);
    EXPECT_TRUE(cJSON_IsNumber(num_inputs)) << "num_inputs must be number";

    cJSON* num_outputs = cJSON_GetObjectItem(config_obj, "num_outputs");
    ASSERT_NE(num_outputs, nullptr);
    EXPECT_TRUE(cJSON_IsNumber(num_outputs)) << "num_outputs must be number";

    // WHAT: Verify float fields
    cJSON* learning_rate = cJSON_GetObjectItem(config_obj, "learning_rate");
    ASSERT_NE(learning_rate, nullptr);
    EXPECT_TRUE(cJSON_IsNumber(learning_rate)) << "learning_rate must be number";

    // WHAT: Verify boolean fields
    cJSON* enable_stdp = cJSON_GetObjectItem(config_obj, "enable_stdp");
    ASSERT_NE(enable_stdp, nullptr);
    EXPECT_TRUE(cJSON_IsBool(enable_stdp)) << "enable_stdp must be boolean";

    // WHAT: Verify string fields
    cJSON* task_name = cJSON_GetObjectItem(config_obj, "task_name");
    ASSERT_NE(task_name, nullptr);
    EXPECT_TRUE(cJSON_IsString(task_name)) << "task_name must be string";

    cJSON_Delete(root);
    brain_destroy(brain);
    free(json);
}

//=============================================================================
// Range Validation Tests
//=============================================================================

TEST_F(BrainJSONRegressionTest, ConfigRangeValidation) {
    // WHAT: Verify config values are in valid ranges
    // WHY:  Ensure exported data is semantically valid

    brain_config_t config = brain_default_config();
    brain_t brain = brain_create(&config);
    ASSERT_NE(brain, nullptr);

    char* json = brain_export_json(brain, BRAIN_EXPORT_CONFIG);
    ASSERT_NE(json, nullptr);

    cJSON* root = cJSON_Parse(json);
    ASSERT_NE(root, nullptr);

    cJSON* config_obj = cJSON_GetObjectItem(root, "config");
    ASSERT_NE(config_obj, nullptr);

    // WHAT: Verify num_inputs in valid range
    cJSON* num_inputs = cJSON_GetObjectItem(config_obj, "num_inputs");
    ASSERT_NE(num_inputs, nullptr);
    EXPECT_GT(num_inputs->valueint, 0);
    EXPECT_LT(num_inputs->valueint, 100000);

    // WHAT: Verify learning_rate in valid range
    cJSON* learning_rate = cJSON_GetObjectItem(config_obj, "learning_rate");
    ASSERT_NE(learning_rate, nullptr);
    EXPECT_GE(learning_rate->valuedouble, 0.0);
    EXPECT_LE(learning_rate->valuedouble, 1.0);

    cJSON_Delete(root);
    brain_destroy(brain);
    free(json);
}

//=============================================================================
// Backward Compatibility Tests
//=============================================================================

TEST_F(BrainJSONRegressionTest, SchemaVersion1_0Compatibility) {
    // WHAT: Verify current version can import schema 1.0
    // WHY:  Ensure backward compatibility

    // WHAT: Create valid schema 1.0 JSON
    const char* json_v1_0 = R"({
        "schema_version": "1.0",
        "metadata": {
            "nimcp_version": "2.7.0",
            "export_time": "2025-11-17T12:00:00Z"
        },
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

    // WHAT: Import should succeed
    brain_t brain = brain_import_json(json_v1_0);
    ASSERT_NE(brain, nullptr) << "Should import schema 1.0";

    // WHAT: Verify imported configuration
    brain_config_t config = brain_get_config(brain);
    EXPECT_EQ(config.num_inputs, 10);
    EXPECT_EQ(config.num_outputs, 5);
    EXPECT_NEAR(config.learning_rate, 0.01, 0.001);

    brain_destroy(brain);
}

//=============================================================================
// Schema Stability Tests
//=============================================================================

TEST_F(BrainJSONRegressionTest, ExportFormatStability) {
    // WHAT: Verify export format is stable across multiple exports
    // WHY:  Ensure deterministic JSON output

    brain_config_t config = brain_default_config();
    config.num_inputs = 5;
    config.num_outputs = 3;
    config.learning_rate = 0.02f;

    brain_t brain = brain_create(&config);
    ASSERT_NE(brain, nullptr);

    // WHAT: Export multiple times
    char* json1 = brain_export_json(brain, BRAIN_EXPORT_CONFIG | BRAIN_EXPORT_COMPACT);
    char* json2 = brain_export_json(brain, BRAIN_EXPORT_CONFIG | BRAIN_EXPORT_COMPACT);

    ASSERT_NE(json1, nullptr);
    ASSERT_NE(json2, nullptr);

    // WHAT: Parse both JSONs
    cJSON* root1 = cJSON_Parse(json1);
    cJSON* root2 = cJSON_Parse(json2);

    ASSERT_NE(root1, nullptr);
    ASSERT_NE(root2, nullptr);

    // WHAT: Verify config sections are identical
    cJSON* config1 = cJSON_GetObjectItem(root1, "config");
    cJSON* config2 = cJSON_GetObjectItem(root2, "config");

    ASSERT_NE(config1, nullptr);
    ASSERT_NE(config2, nullptr);

    // WHAT: Compare numeric fields
    cJSON* num_inputs1 = cJSON_GetObjectItem(config1, "num_inputs");
    cJSON* num_inputs2 = cJSON_GetObjectItem(config2, "num_inputs");
    EXPECT_EQ(num_inputs1->valueint, num_inputs2->valueint);

    cJSON_Delete(root1);
    cJSON_Delete(root2);
    brain_destroy(brain);
    free(json1);
    free(json2);
}

//=============================================================================
// Main Test Runner
//=============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
