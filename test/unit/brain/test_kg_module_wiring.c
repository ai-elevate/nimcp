/**
 * @file test_kg_module_wiring.c
 * @brief Unit tests for KG Module Wiring Descriptor
 *
 * WHAT: Comprehensive test suite for kg_module_wiring API
 * WHY:  Verify correct behavior of module wiring descriptors
 * HOW:  Unit tests using Check framework covering all API functions
 *
 * @author NIMCP Development Team
 * @date 2026-01-16
 */

#include <check.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "core/brain/nimcp_kg_module_wiring.h"

/*=============================================================================
 * Test Fixtures
 *=============================================================================*/

static kg_module_wiring_t* wiring = NULL;

static void setup(void)
{
    wiring = kg_module_wiring_create("test_module", "TEST");
    ck_assert_ptr_nonnull(wiring);
}

static void teardown(void)
{
    if (wiring) {
        kg_module_wiring_destroy(wiring);
        wiring = NULL;
    }
}

/*=============================================================================
 * Lifecycle Tests
 *=============================================================================*/

START_TEST(test_create_valid)
{
    kg_module_wiring_t* w = kg_module_wiring_create("my_module", "SNN");
    ck_assert_ptr_nonnull(w);
    ck_assert_str_eq(w->module_name, "my_module");
    ck_assert_str_eq(w->module_type, "SNN");
    ck_assert_uint_eq(w->input_count, 0);
    ck_assert_uint_eq(w->output_count, 0);
    ck_assert_uint_eq(w->handler_count, 0);
    ck_assert_uint_eq(w->network_type, KG_WEIGHT_NONE);
    ck_assert_ptr_null(w->initial_weights);
    ck_assert_uint_eq(w->weights_size, 0);
    ck_assert_uint_gt(w->creation_timestamp, 0);
    ck_assert_uint_eq(w->version, 1);
    kg_module_wiring_destroy(w);
}
END_TEST

START_TEST(test_create_null_name)
{
    kg_module_wiring_t* w = kg_module_wiring_create(NULL, "SNN");
    ck_assert_ptr_null(w);
}
END_TEST

START_TEST(test_create_null_type)
{
    kg_module_wiring_t* w = kg_module_wiring_create("my_module", NULL);
    ck_assert_ptr_null(w);
}
END_TEST

START_TEST(test_create_empty_name)
{
    kg_module_wiring_t* w = kg_module_wiring_create("", "SNN");
    ck_assert_ptr_null(w);
}
END_TEST

START_TEST(test_create_empty_type)
{
    kg_module_wiring_t* w = kg_module_wiring_create("my_module", "");
    ck_assert_ptr_null(w);
}
END_TEST

START_TEST(test_create_name_too_long)
{
    /* Name must be < KG_WIRING_MAX_NAME_LEN (64) */
    char long_name[128];
    memset(long_name, 'a', 64);
    long_name[64] = '\0';
    kg_module_wiring_t* w = kg_module_wiring_create(long_name, "SNN");
    ck_assert_ptr_null(w);
}
END_TEST

START_TEST(test_create_type_too_long)
{
    /* Type must be < KG_WIRING_MAX_TYPE_LEN (32) */
    char long_type[64];
    memset(long_type, 'a', 32);
    long_type[32] = '\0';
    kg_module_wiring_t* w = kg_module_wiring_create("my_module", long_type);
    ck_assert_ptr_null(w);
}
END_TEST

START_TEST(test_create_max_valid_name)
{
    /* Name of exactly 63 chars should work */
    char max_name[64];
    memset(max_name, 'a', 63);
    max_name[63] = '\0';
    kg_module_wiring_t* w = kg_module_wiring_create(max_name, "SNN");
    ck_assert_ptr_nonnull(w);
    ck_assert_str_eq(w->module_name, max_name);
    kg_module_wiring_destroy(w);
}
END_TEST

START_TEST(test_destroy_null)
{
    /* Should not crash */
    kg_module_wiring_destroy(NULL);
}
END_TEST

/*=============================================================================
 * Input Registration Tests
 *=============================================================================*/

START_TEST(test_add_input_valid)
{
    int ret = kg_module_wiring_add_input(wiring, "source_mod", "MSG_TYPE", true);
    ck_assert_int_eq(ret, 0);
    ck_assert_uint_eq(wiring->input_count, 1);
    ck_assert_str_eq(wiring->inputs[0].source_module, "source_mod");
    ck_assert_str_eq(wiring->inputs[0].message_type, "MSG_TYPE");
    ck_assert(wiring->inputs[0].required == true);
}
END_TEST

START_TEST(test_add_input_optional)
{
    int ret = kg_module_wiring_add_input(wiring, "source_mod", "MSG_TYPE", false);
    ck_assert_int_eq(ret, 0);
    ck_assert(wiring->inputs[0].required == false);
}
END_TEST

START_TEST(test_add_input_null_wiring)
{
    int ret = kg_module_wiring_add_input(NULL, "source", "type", true);
    ck_assert_int_eq(ret, -1);
}
END_TEST

START_TEST(test_add_input_null_source)
{
    int ret = kg_module_wiring_add_input(wiring, NULL, "type", true);
    ck_assert_int_eq(ret, -1);
}
END_TEST

START_TEST(test_add_input_null_msg_type)
{
    int ret = kg_module_wiring_add_input(wiring, "source", NULL, true);
    ck_assert_int_eq(ret, -1);
}
END_TEST

START_TEST(test_add_input_multiple)
{
    int ret;
    ret = kg_module_wiring_add_input(wiring, "mod1", "TYPE_A", true);
    ck_assert_int_eq(ret, 0);
    ret = kg_module_wiring_add_input(wiring, "mod2", "TYPE_B", false);
    ck_assert_int_eq(ret, 0);
    ret = kg_module_wiring_add_input(wiring, "mod3", "TYPE_C", true);
    ck_assert_int_eq(ret, 0);
    ck_assert_uint_eq(wiring->input_count, 3);
}
END_TEST

START_TEST(test_add_input_duplicate_updates)
{
    /* Adding duplicate source+msg_type should update required flag */
    int ret;
    ret = kg_module_wiring_add_input(wiring, "source", "TYPE", true);
    ck_assert_int_eq(ret, 0);
    ck_assert_uint_eq(wiring->input_count, 1);
    ck_assert(wiring->inputs[0].required == true);

    ret = kg_module_wiring_add_input(wiring, "source", "TYPE", false);
    ck_assert_int_eq(ret, 0);
    ck_assert_uint_eq(wiring->input_count, 1);  /* Still 1, not 2 */
    ck_assert(wiring->inputs[0].required == false);  /* Updated */
}
END_TEST

START_TEST(test_add_input_same_source_diff_type)
{
    /* Same source but different msg_type should be separate entries */
    int ret;
    ret = kg_module_wiring_add_input(wiring, "source", "TYPE_A", true);
    ck_assert_int_eq(ret, 0);
    ret = kg_module_wiring_add_input(wiring, "source", "TYPE_B", true);
    ck_assert_int_eq(ret, 0);
    ck_assert_uint_eq(wiring->input_count, 2);
}
END_TEST

START_TEST(test_add_input_max_capacity)
{
    /* Fill up to max inputs (32), then fail on next */
    int ret;
    char source[32];
    for (uint32_t i = 0; i < KG_WIRING_MAX_INPUTS; i++) {
        snprintf(source, sizeof(source), "source_%u", i);
        ret = kg_module_wiring_add_input(wiring, source, "TYPE", true);
        ck_assert_int_eq(ret, 0);
    }
    ck_assert_uint_eq(wiring->input_count, KG_WIRING_MAX_INPUTS);

    /* Next should fail */
    ret = kg_module_wiring_add_input(wiring, "overflow_source", "TYPE", true);
    ck_assert_int_eq(ret, -1);
    ck_assert_uint_eq(wiring->input_count, KG_WIRING_MAX_INPUTS);
}
END_TEST

/*=============================================================================
 * Output Registration Tests
 *=============================================================================*/

START_TEST(test_add_output_valid)
{
    int ret = kg_module_wiring_add_output(wiring, "OUTPUT_MSG", "Output description");
    ck_assert_int_eq(ret, 0);
    ck_assert_uint_eq(wiring->output_count, 1);
    ck_assert_str_eq(wiring->outputs[0].message_type, "OUTPUT_MSG");
    ck_assert_str_eq(wiring->outputs[0].description, "Output description");
}
END_TEST

START_TEST(test_add_output_null_description)
{
    /* NULL description is allowed */
    int ret = kg_module_wiring_add_output(wiring, "OUTPUT_MSG", NULL);
    ck_assert_int_eq(ret, 0);
    ck_assert_uint_eq(wiring->output_count, 1);
    ck_assert_str_eq(wiring->outputs[0].description, "");
}
END_TEST

START_TEST(test_add_output_null_wiring)
{
    int ret = kg_module_wiring_add_output(NULL, "OUTPUT_MSG", "desc");
    ck_assert_int_eq(ret, -1);
}
END_TEST

START_TEST(test_add_output_null_msg_type)
{
    int ret = kg_module_wiring_add_output(wiring, NULL, "desc");
    ck_assert_int_eq(ret, -1);
}
END_TEST

START_TEST(test_add_output_duplicate_updates)
{
    int ret;
    ret = kg_module_wiring_add_output(wiring, "OUTPUT_MSG", "Description 1");
    ck_assert_int_eq(ret, 0);
    ck_assert_uint_eq(wiring->output_count, 1);

    ret = kg_module_wiring_add_output(wiring, "OUTPUT_MSG", "Description 2");
    ck_assert_int_eq(ret, 0);
    ck_assert_uint_eq(wiring->output_count, 1);  /* Still 1 */
    ck_assert_str_eq(wiring->outputs[0].description, "Description 2");  /* Updated */
}
END_TEST

START_TEST(test_add_output_max_capacity)
{
    int ret;
    char msg_type[64];
    for (uint32_t i = 0; i < KG_WIRING_MAX_OUTPUTS; i++) {
        snprintf(msg_type, sizeof(msg_type), "OUTPUT_%u", i);
        ret = kg_module_wiring_add_output(wiring, msg_type, "desc");
        ck_assert_int_eq(ret, 0);
    }
    ck_assert_uint_eq(wiring->output_count, KG_WIRING_MAX_OUTPUTS);

    ret = kg_module_wiring_add_output(wiring, "OVERFLOW", "desc");
    ck_assert_int_eq(ret, -1);
}
END_TEST

/*=============================================================================
 * Handler Registration Tests
 *=============================================================================*/

START_TEST(test_add_handler_valid)
{
    int ret = kg_module_wiring_add_handler(wiring, "HANDLER_MSG", 100);
    ck_assert_int_eq(ret, 0);
    ck_assert_uint_eq(wiring->handler_count, 1);
    ck_assert_str_eq(wiring->handlers[0].message_type, "HANDLER_MSG");
    ck_assert_uint_eq(wiring->handlers[0].priority, 100);
}
END_TEST

START_TEST(test_add_handler_null_wiring)
{
    int ret = kg_module_wiring_add_handler(NULL, "MSG", 100);
    ck_assert_int_eq(ret, -1);
}
END_TEST

START_TEST(test_add_handler_null_msg_type)
{
    int ret = kg_module_wiring_add_handler(wiring, NULL, 100);
    ck_assert_int_eq(ret, -1);
}
END_TEST

START_TEST(test_add_handler_duplicate_updates_priority)
{
    int ret;
    ret = kg_module_wiring_add_handler(wiring, "MSG", 100);
    ck_assert_int_eq(ret, 0);
    ck_assert_uint_eq(wiring->handlers[0].priority, 100);

    ret = kg_module_wiring_add_handler(wiring, "MSG", 200);
    ck_assert_int_eq(ret, 0);
    ck_assert_uint_eq(wiring->handler_count, 1);  /* Still 1 */
    ck_assert_uint_eq(wiring->handlers[0].priority, 200);  /* Updated */
}
END_TEST

START_TEST(test_add_handler_max_capacity)
{
    int ret;
    char msg_type[64];
    for (uint32_t i = 0; i < KG_WIRING_MAX_HANDLERS; i++) {
        snprintf(msg_type, sizeof(msg_type), "HANDLER_%u", i);
        ret = kg_module_wiring_add_handler(wiring, msg_type, i);
        ck_assert_int_eq(ret, 0);
    }
    ck_assert_uint_eq(wiring->handler_count, KG_WIRING_MAX_HANDLERS);

    ret = kg_module_wiring_add_handler(wiring, "OVERFLOW", 999);
    ck_assert_int_eq(ret, -1);
}
END_TEST

/*=============================================================================
 * Weight State Tests
 *=============================================================================*/

START_TEST(test_set_weights_valid)
{
    uint8_t weights[] = {1, 2, 3, 4, 5};
    int ret = kg_module_wiring_set_weights(wiring, KG_WEIGHT_SNN, weights, sizeof(weights));
    ck_assert_int_eq(ret, 0);
    ck_assert_uint_eq(wiring->network_type, KG_WEIGHT_SNN);
    ck_assert_ptr_nonnull(wiring->initial_weights);
    ck_assert_uint_eq(wiring->weights_size, sizeof(weights));
    ck_assert_mem_eq(wiring->initial_weights, weights, sizeof(weights));
}
END_TEST

START_TEST(test_set_weights_null_wiring)
{
    uint8_t weights[] = {1, 2, 3};
    int ret = kg_module_wiring_set_weights(NULL, KG_WEIGHT_SNN, weights, sizeof(weights));
    ck_assert_int_eq(ret, -1);
}
END_TEST

START_TEST(test_set_weights_null_data)
{
    /* NULL weights with size 0 should just set the type */
    int ret = kg_module_wiring_set_weights(wiring, KG_WEIGHT_CNN, NULL, 0);
    ck_assert_int_eq(ret, 0);
    ck_assert_uint_eq(wiring->network_type, KG_WEIGHT_CNN);
    ck_assert_ptr_null(wiring->initial_weights);
    ck_assert_uint_eq(wiring->weights_size, 0);
}
END_TEST

START_TEST(test_set_weights_replace_existing)
{
    uint8_t weights1[] = {1, 2, 3};
    uint8_t weights2[] = {4, 5, 6, 7, 8};

    int ret = kg_module_wiring_set_weights(wiring, KG_WEIGHT_SNN, weights1, sizeof(weights1));
    ck_assert_int_eq(ret, 0);

    ret = kg_module_wiring_set_weights(wiring, KG_WEIGHT_LNN, weights2, sizeof(weights2));
    ck_assert_int_eq(ret, 0);
    ck_assert_uint_eq(wiring->network_type, KG_WEIGHT_LNN);
    ck_assert_uint_eq(wiring->weights_size, sizeof(weights2));
    ck_assert_mem_eq(wiring->initial_weights, weights2, sizeof(weights2));
}
END_TEST

START_TEST(test_set_weights_clear_existing)
{
    uint8_t weights[] = {1, 2, 3};
    int ret = kg_module_wiring_set_weights(wiring, KG_WEIGHT_SNN, weights, sizeof(weights));
    ck_assert_int_eq(ret, 0);
    ck_assert_ptr_nonnull(wiring->initial_weights);

    /* Clear by setting NULL */
    ret = kg_module_wiring_set_weights(wiring, KG_WEIGHT_NONE, NULL, 0);
    ck_assert_int_eq(ret, 0);
    ck_assert_uint_eq(wiring->network_type, KG_WEIGHT_NONE);
    ck_assert_ptr_null(wiring->initial_weights);
    ck_assert_uint_eq(wiring->weights_size, 0);
}
END_TEST

/*=============================================================================
 * Metadata Tests
 *=============================================================================*/

START_TEST(test_set_metadata_valid)
{
    int ret = kg_module_wiring_set_metadata(wiring, "Author Name", "Category", "Description text");
    ck_assert_int_eq(ret, 0);
    ck_assert_str_eq(wiring->metadata.author, "Author Name");
    ck_assert_str_eq(wiring->metadata.category, "Category");
    ck_assert_str_eq(wiring->metadata.description, "Description text");
}
END_TEST

START_TEST(test_set_metadata_null_fields)
{
    /* NULL for any field should skip that field */
    int ret = kg_module_wiring_set_metadata(wiring, "Author", NULL, NULL);
    ck_assert_int_eq(ret, 0);
    ck_assert_str_eq(wiring->metadata.author, "Author");
    /* category and description should remain empty/unchanged */
}
END_TEST

START_TEST(test_set_metadata_null_wiring)
{
    int ret = kg_module_wiring_set_metadata(NULL, "Author", "Cat", "Desc");
    ck_assert_int_eq(ret, -1);
}
END_TEST

START_TEST(test_add_metadata_entry_valid)
{
    int ret = kg_module_wiring_add_metadata_entry(wiring, "key1", "value1");
    ck_assert_int_eq(ret, 0);
    ck_assert_uint_eq(wiring->metadata.entry_count, 1);
    ck_assert_str_eq(wiring->metadata.entries[0].key, "key1");
    ck_assert_str_eq(wiring->metadata.entries[0].value, "value1");
}
END_TEST

START_TEST(test_add_metadata_entry_null_params)
{
    int ret;
    ret = kg_module_wiring_add_metadata_entry(NULL, "key", "value");
    ck_assert_int_eq(ret, -1);
    ret = kg_module_wiring_add_metadata_entry(wiring, NULL, "value");
    ck_assert_int_eq(ret, -1);
    ret = kg_module_wiring_add_metadata_entry(wiring, "key", NULL);
    ck_assert_int_eq(ret, -1);
}
END_TEST

START_TEST(test_add_metadata_entry_duplicate_updates)
{
    int ret;
    ret = kg_module_wiring_add_metadata_entry(wiring, "key", "value1");
    ck_assert_int_eq(ret, 0);

    ret = kg_module_wiring_add_metadata_entry(wiring, "key", "value2");
    ck_assert_int_eq(ret, 0);
    ck_assert_uint_eq(wiring->metadata.entry_count, 1);  /* Still 1 */
    ck_assert_str_eq(wiring->metadata.entries[0].value, "value2");  /* Updated */
}
END_TEST

START_TEST(test_add_metadata_entry_max_capacity)
{
    int ret;
    char key[32];
    for (uint32_t i = 0; i < KG_WIRING_MAX_METADATA; i++) {
        snprintf(key, sizeof(key), "key_%u", i);
        ret = kg_module_wiring_add_metadata_entry(wiring, key, "value");
        ck_assert_int_eq(ret, 0);
    }
    ck_assert_uint_eq(wiring->metadata.entry_count, KG_WIRING_MAX_METADATA);

    ret = kg_module_wiring_add_metadata_entry(wiring, "overflow_key", "value");
    ck_assert_int_eq(ret, -1);
}
END_TEST

START_TEST(test_set_version_valid)
{
    int ret = kg_module_wiring_set_version(wiring, 2, 5, 10);
    ck_assert_int_eq(ret, 0);
    ck_assert_uint_eq(wiring->metadata.version_major, 2);
    ck_assert_uint_eq(wiring->metadata.version_minor, 5);
    ck_assert_uint_eq(wiring->metadata.version_patch, 10);

    /* Check encoded version: (2 << 32) | (5 << 16) | 10 */
    uint64_t expected = ((uint64_t)2 << 32) | ((uint64_t)5 << 16) | 10;
    ck_assert_uint_eq(wiring->version, expected);
}
END_TEST

START_TEST(test_set_version_null_wiring)
{
    int ret = kg_module_wiring_set_version(NULL, 1, 0, 0);
    ck_assert_int_eq(ret, -1);
}
END_TEST

/*=============================================================================
 * Query API Tests
 *=============================================================================*/

START_TEST(test_has_input_found)
{
    kg_module_wiring_add_input(wiring, "source", "MSG_TYPE", true);
    ck_assert(kg_module_wiring_has_input(wiring, "source", "MSG_TYPE") == true);
}
END_TEST

START_TEST(test_has_input_not_found)
{
    kg_module_wiring_add_input(wiring, "source", "MSG_TYPE", true);
    ck_assert(kg_module_wiring_has_input(wiring, "other", "MSG_TYPE") == false);
    ck_assert(kg_module_wiring_has_input(wiring, "source", "OTHER") == false);
}
END_TEST

START_TEST(test_has_input_null_msg_type_matches_any)
{
    kg_module_wiring_add_input(wiring, "source", "MSG_TYPE", true);
    /* NULL msg_type should match any message type from that source */
    ck_assert(kg_module_wiring_has_input(wiring, "source", NULL) == true);
}
END_TEST

START_TEST(test_has_input_null_params)
{
    ck_assert(kg_module_wiring_has_input(NULL, "source", "type") == false);
    ck_assert(kg_module_wiring_has_input(wiring, NULL, "type") == false);
}
END_TEST

START_TEST(test_has_output_found)
{
    kg_module_wiring_add_output(wiring, "OUTPUT_MSG", "desc");
    ck_assert(kg_module_wiring_has_output(wiring, "OUTPUT_MSG") == true);
}
END_TEST

START_TEST(test_has_output_not_found)
{
    kg_module_wiring_add_output(wiring, "OUTPUT_MSG", "desc");
    ck_assert(kg_module_wiring_has_output(wiring, "OTHER") == false);
}
END_TEST

START_TEST(test_has_output_null_params)
{
    ck_assert(kg_module_wiring_has_output(NULL, "type") == false);
    ck_assert(kg_module_wiring_has_output(wiring, NULL) == false);
}
END_TEST

START_TEST(test_has_handler_found)
{
    kg_module_wiring_add_handler(wiring, "HANDLER_MSG", 100);
    ck_assert(kg_module_wiring_has_handler(wiring, "HANDLER_MSG") == true);
}
END_TEST

START_TEST(test_has_handler_not_found)
{
    kg_module_wiring_add_handler(wiring, "HANDLER_MSG", 100);
    ck_assert(kg_module_wiring_has_handler(wiring, "OTHER") == false);
}
END_TEST

START_TEST(test_has_handler_null_params)
{
    ck_assert(kg_module_wiring_has_handler(NULL, "type") == false);
    ck_assert(kg_module_wiring_has_handler(wiring, NULL) == false);
}
END_TEST

START_TEST(test_get_handler_priority_found)
{
    kg_module_wiring_add_handler(wiring, "HANDLER_MSG", 150);
    uint32_t priority = kg_module_wiring_get_handler_priority(wiring, "HANDLER_MSG");
    ck_assert_uint_eq(priority, 150);
}
END_TEST

START_TEST(test_get_handler_priority_not_found)
{
    kg_module_wiring_add_handler(wiring, "HANDLER_MSG", 150);
    uint32_t priority = kg_module_wiring_get_handler_priority(wiring, "OTHER");
    ck_assert_uint_eq(priority, 0);
}
END_TEST

START_TEST(test_get_handler_priority_null_params)
{
    ck_assert_uint_eq(kg_module_wiring_get_handler_priority(NULL, "type"), 0);
    ck_assert_uint_eq(kg_module_wiring_get_handler_priority(wiring, NULL), 0);
}
END_TEST

/*=============================================================================
 * Validation Tests
 *=============================================================================*/

START_TEST(test_validate_valid)
{
    char error_buf[256] = {0};
    int ret = kg_module_wiring_validate(wiring, error_buf, sizeof(error_buf));
    ck_assert_int_eq(ret, 0);
}
END_TEST

START_TEST(test_validate_null_wiring)
{
    char error_buf[256] = {0};
    int ret = kg_module_wiring_validate(NULL, error_buf, sizeof(error_buf));
    ck_assert_int_eq(ret, -1);
    ck_assert_str_eq(error_buf, "Wiring descriptor is NULL");
}
END_TEST

START_TEST(test_validate_null_error_buf)
{
    /* Should work even without error buffer */
    int ret = kg_module_wiring_validate(wiring, NULL, 0);
    ck_assert_int_eq(ret, 0);
}
END_TEST

START_TEST(test_validate_with_inputs_outputs_handlers)
{
    kg_module_wiring_add_input(wiring, "source", "MSG", true);
    kg_module_wiring_add_output(wiring, "OUT", "desc");
    kg_module_wiring_add_handler(wiring, "HANDLER", 100);

    char error_buf[256] = {0};
    int ret = kg_module_wiring_validate(wiring, error_buf, sizeof(error_buf));
    ck_assert_int_eq(ret, 0);
}
END_TEST

START_TEST(test_validate_weight_consistency_pointer_no_size)
{
    /* Set weights with pointer but then manually corrupt size */
    uint8_t weights[] = {1, 2, 3};
    kg_module_wiring_set_weights(wiring, KG_WEIGHT_SNN, weights, sizeof(weights));
    wiring->weights_size = 0;  /* Corrupt */

    char error_buf[256] = {0};
    int ret = kg_module_wiring_validate(wiring, error_buf, sizeof(error_buf));
    ck_assert_int_eq(ret, -1);
    ck_assert(strstr(error_buf, "Weights pointer set but size is zero") != NULL);
}
END_TEST

START_TEST(test_validate_weight_consistency_size_no_pointer)
{
    /* Set weights then manually corrupt */
    uint8_t weights[] = {1, 2, 3};
    kg_module_wiring_set_weights(wiring, KG_WEIGHT_SNN, weights, sizeof(weights));
    void* saved = wiring->initial_weights;
    wiring->initial_weights = NULL;  /* Corrupt */

    char error_buf[256] = {0};
    int ret = kg_module_wiring_validate(wiring, error_buf, sizeof(error_buf));
    ck_assert_int_eq(ret, -1);
    ck_assert(strstr(error_buf, "Weights size set but pointer is NULL") != NULL);

    /* Restore for proper cleanup */
    wiring->initial_weights = saved;
}
END_TEST

/*=============================================================================
 * String Conversion Tests
 *=============================================================================*/

START_TEST(test_weight_type_to_string)
{
    ck_assert_str_eq(kg_weight_type_to_string(KG_WEIGHT_NONE), "NONE");
    ck_assert_str_eq(kg_weight_type_to_string(KG_WEIGHT_SNN), "SNN");
    ck_assert_str_eq(kg_weight_type_to_string(KG_WEIGHT_LNN), "LNN");
    ck_assert_str_eq(kg_weight_type_to_string(KG_WEIGHT_CNN), "CNN");
    ck_assert_str_eq(kg_weight_type_to_string(KG_WEIGHT_RNN), "RNN");
    ck_assert_str_eq(kg_weight_type_to_string(KG_WEIGHT_TRANSFORMER), "TRANSFORMER");
    ck_assert_str_eq(kg_weight_type_to_string(KG_WEIGHT_GNN), "GNN");
    ck_assert_str_eq(kg_weight_type_to_string(KG_WEIGHT_HYBRID), "HYBRID");
    ck_assert_str_eq(kg_weight_type_to_string(KG_WEIGHT_CUSTOM), "CUSTOM");
}
END_TEST

START_TEST(test_weight_type_to_string_unknown)
{
    ck_assert_str_eq(kg_weight_type_to_string((kg_weight_type_t)999), "UNKNOWN");
    ck_assert_str_eq(kg_weight_type_to_string((kg_weight_type_t)-1), "UNKNOWN");
}
END_TEST

START_TEST(test_weight_type_from_string)
{
    ck_assert_uint_eq(kg_weight_type_from_string("NONE"), KG_WEIGHT_NONE);
    ck_assert_uint_eq(kg_weight_type_from_string("SNN"), KG_WEIGHT_SNN);
    ck_assert_uint_eq(kg_weight_type_from_string("LNN"), KG_WEIGHT_LNN);
    ck_assert_uint_eq(kg_weight_type_from_string("CNN"), KG_WEIGHT_CNN);
    ck_assert_uint_eq(kg_weight_type_from_string("RNN"), KG_WEIGHT_RNN);
    ck_assert_uint_eq(kg_weight_type_from_string("TRANSFORMER"), KG_WEIGHT_TRANSFORMER);
    ck_assert_uint_eq(kg_weight_type_from_string("GNN"), KG_WEIGHT_GNN);
    ck_assert_uint_eq(kg_weight_type_from_string("HYBRID"), KG_WEIGHT_HYBRID);
    ck_assert_uint_eq(kg_weight_type_from_string("CUSTOM"), KG_WEIGHT_CUSTOM);
}
END_TEST

START_TEST(test_weight_type_from_string_case_insensitive)
{
    ck_assert_uint_eq(kg_weight_type_from_string("snn"), KG_WEIGHT_SNN);
    ck_assert_uint_eq(kg_weight_type_from_string("Snn"), KG_WEIGHT_SNN);
    ck_assert_uint_eq(kg_weight_type_from_string("transformer"), KG_WEIGHT_TRANSFORMER);
}
END_TEST

START_TEST(test_weight_type_from_string_null)
{
    ck_assert_uint_eq(kg_weight_type_from_string(NULL), KG_WEIGHT_NONE);
}
END_TEST

START_TEST(test_weight_type_from_string_unknown)
{
    ck_assert_uint_eq(kg_weight_type_from_string("INVALID"), KG_WEIGHT_NONE);
    ck_assert_uint_eq(kg_weight_type_from_string(""), KG_WEIGHT_NONE);
}
END_TEST

/*=============================================================================
 * Integration Tests
 *=============================================================================*/

START_TEST(test_full_wiring_workflow)
{
    /* Complete workflow test */
    kg_module_wiring_t* w = kg_module_wiring_create("prefrontal_cortex", "COGNITIVE");
    ck_assert_ptr_nonnull(w);

    /* Set metadata */
    int ret = kg_module_wiring_set_metadata(w, "NIMCP Team", "Brain", "Prefrontal cortex module");
    ck_assert_int_eq(ret, 0);

    ret = kg_module_wiring_set_version(w, 1, 2, 3);
    ck_assert_int_eq(ret, 0);

    ret = kg_module_wiring_add_metadata_entry(w, "region", "frontal");
    ck_assert_int_eq(ret, 0);

    /* Add inputs */
    ret = kg_module_wiring_add_input(w, "hippocampus", "MEMORY_QUERY", true);
    ck_assert_int_eq(ret, 0);
    ret = kg_module_wiring_add_input(w, "basal_ganglia", "REWARD_SIGNAL", false);
    ck_assert_int_eq(ret, 0);

    /* Add outputs */
    ret = kg_module_wiring_add_output(w, "DECISION_OUT", "Executive decisions");
    ck_assert_int_eq(ret, 0);
    ret = kg_module_wiring_add_output(w, "WORKING_MEM", "Working memory updates");
    ck_assert_int_eq(ret, 0);

    /* Add handlers */
    ret = kg_module_wiring_add_handler(w, "MEMORY_QUERY", 100);
    ck_assert_int_eq(ret, 0);
    ret = kg_module_wiring_add_handler(w, "DECISION_REQUEST", 200);
    ck_assert_int_eq(ret, 0);

    /* Set weights */
    float weights[] = {0.1f, 0.2f, 0.3f, 0.4f};
    ret = kg_module_wiring_set_weights(w, KG_WEIGHT_SNN, weights, sizeof(weights));
    ck_assert_int_eq(ret, 0);

    /* Validate */
    char error_buf[256] = {0};
    ret = kg_module_wiring_validate(w, error_buf, sizeof(error_buf));
    ck_assert_int_eq(ret, 0);

    /* Query checks */
    ck_assert(kg_module_wiring_has_input(w, "hippocampus", "MEMORY_QUERY"));
    ck_assert(kg_module_wiring_has_input(w, "basal_ganglia", NULL));  /* Any from source */
    ck_assert(!kg_module_wiring_has_input(w, "amygdala", NULL));

    ck_assert(kg_module_wiring_has_output(w, "DECISION_OUT"));
    ck_assert(!kg_module_wiring_has_output(w, "OTHER"));

    ck_assert(kg_module_wiring_has_handler(w, "MEMORY_QUERY"));
    ck_assert_uint_eq(kg_module_wiring_get_handler_priority(w, "DECISION_REQUEST"), 200);

    /* Verify counts */
    ck_assert_uint_eq(w->input_count, 2);
    ck_assert_uint_eq(w->output_count, 2);
    ck_assert_uint_eq(w->handler_count, 2);

    kg_module_wiring_destroy(w);
}
END_TEST

/*=============================================================================
 * Test Suite Creation
 *=============================================================================*/

Suite* kg_module_wiring_suite(void)
{
    Suite* s = suite_create("KG Module Wiring");

    /* Lifecycle tests */
    TCase* tc_lifecycle = tcase_create("Lifecycle");
    tcase_add_test(tc_lifecycle, test_create_valid);
    tcase_add_test(tc_lifecycle, test_create_null_name);
    tcase_add_test(tc_lifecycle, test_create_null_type);
    tcase_add_test(tc_lifecycle, test_create_empty_name);
    tcase_add_test(tc_lifecycle, test_create_empty_type);
    tcase_add_test(tc_lifecycle, test_create_name_too_long);
    tcase_add_test(tc_lifecycle, test_create_type_too_long);
    tcase_add_test(tc_lifecycle, test_create_max_valid_name);
    tcase_add_test(tc_lifecycle, test_destroy_null);
    suite_add_tcase(s, tc_lifecycle);

    /* Input registration tests */
    TCase* tc_input = tcase_create("Input Registration");
    tcase_add_checked_fixture(tc_input, setup, teardown);
    tcase_add_test(tc_input, test_add_input_valid);
    tcase_add_test(tc_input, test_add_input_optional);
    tcase_add_test(tc_input, test_add_input_null_wiring);
    tcase_add_test(tc_input, test_add_input_null_source);
    tcase_add_test(tc_input, test_add_input_null_msg_type);
    tcase_add_test(tc_input, test_add_input_multiple);
    tcase_add_test(tc_input, test_add_input_duplicate_updates);
    tcase_add_test(tc_input, test_add_input_same_source_diff_type);
    tcase_add_test(tc_input, test_add_input_max_capacity);
    suite_add_tcase(s, tc_input);

    /* Output registration tests */
    TCase* tc_output = tcase_create("Output Registration");
    tcase_add_checked_fixture(tc_output, setup, teardown);
    tcase_add_test(tc_output, test_add_output_valid);
    tcase_add_test(tc_output, test_add_output_null_description);
    tcase_add_test(tc_output, test_add_output_null_wiring);
    tcase_add_test(tc_output, test_add_output_null_msg_type);
    tcase_add_test(tc_output, test_add_output_duplicate_updates);
    tcase_add_test(tc_output, test_add_output_max_capacity);
    suite_add_tcase(s, tc_output);

    /* Handler registration tests */
    TCase* tc_handler = tcase_create("Handler Registration");
    tcase_add_checked_fixture(tc_handler, setup, teardown);
    tcase_add_test(tc_handler, test_add_handler_valid);
    tcase_add_test(tc_handler, test_add_handler_null_wiring);
    tcase_add_test(tc_handler, test_add_handler_null_msg_type);
    tcase_add_test(tc_handler, test_add_handler_duplicate_updates_priority);
    tcase_add_test(tc_handler, test_add_handler_max_capacity);
    suite_add_tcase(s, tc_handler);

    /* Weight state tests */
    TCase* tc_weights = tcase_create("Weight State");
    tcase_add_checked_fixture(tc_weights, setup, teardown);
    tcase_add_test(tc_weights, test_set_weights_valid);
    tcase_add_test(tc_weights, test_set_weights_null_wiring);
    tcase_add_test(tc_weights, test_set_weights_null_data);
    tcase_add_test(tc_weights, test_set_weights_replace_existing);
    tcase_add_test(tc_weights, test_set_weights_clear_existing);
    suite_add_tcase(s, tc_weights);

    /* Metadata tests */
    TCase* tc_metadata = tcase_create("Metadata");
    tcase_add_checked_fixture(tc_metadata, setup, teardown);
    tcase_add_test(tc_metadata, test_set_metadata_valid);
    tcase_add_test(tc_metadata, test_set_metadata_null_fields);
    tcase_add_test(tc_metadata, test_set_metadata_null_wiring);
    tcase_add_test(tc_metadata, test_add_metadata_entry_valid);
    tcase_add_test(tc_metadata, test_add_metadata_entry_null_params);
    tcase_add_test(tc_metadata, test_add_metadata_entry_duplicate_updates);
    tcase_add_test(tc_metadata, test_add_metadata_entry_max_capacity);
    tcase_add_test(tc_metadata, test_set_version_valid);
    tcase_add_test(tc_metadata, test_set_version_null_wiring);
    suite_add_tcase(s, tc_metadata);

    /* Query API tests */
    TCase* tc_query = tcase_create("Query API");
    tcase_add_checked_fixture(tc_query, setup, teardown);
    tcase_add_test(tc_query, test_has_input_found);
    tcase_add_test(tc_query, test_has_input_not_found);
    tcase_add_test(tc_query, test_has_input_null_msg_type_matches_any);
    tcase_add_test(tc_query, test_has_input_null_params);
    tcase_add_test(tc_query, test_has_output_found);
    tcase_add_test(tc_query, test_has_output_not_found);
    tcase_add_test(tc_query, test_has_output_null_params);
    tcase_add_test(tc_query, test_has_handler_found);
    tcase_add_test(tc_query, test_has_handler_not_found);
    tcase_add_test(tc_query, test_has_handler_null_params);
    tcase_add_test(tc_query, test_get_handler_priority_found);
    tcase_add_test(tc_query, test_get_handler_priority_not_found);
    tcase_add_test(tc_query, test_get_handler_priority_null_params);
    suite_add_tcase(s, tc_query);

    /* Validation tests */
    TCase* tc_validation = tcase_create("Validation");
    tcase_add_checked_fixture(tc_validation, setup, teardown);
    tcase_add_test(tc_validation, test_validate_valid);
    tcase_add_test(tc_validation, test_validate_null_wiring);
    tcase_add_test(tc_validation, test_validate_null_error_buf);
    tcase_add_test(tc_validation, test_validate_with_inputs_outputs_handlers);
    tcase_add_test(tc_validation, test_validate_weight_consistency_pointer_no_size);
    tcase_add_test(tc_validation, test_validate_weight_consistency_size_no_pointer);
    suite_add_tcase(s, tc_validation);

    /* String conversion tests */
    TCase* tc_strings = tcase_create("String Conversions");
    tcase_add_test(tc_strings, test_weight_type_to_string);
    tcase_add_test(tc_strings, test_weight_type_to_string_unknown);
    tcase_add_test(tc_strings, test_weight_type_from_string);
    tcase_add_test(tc_strings, test_weight_type_from_string_case_insensitive);
    tcase_add_test(tc_strings, test_weight_type_from_string_null);
    tcase_add_test(tc_strings, test_weight_type_from_string_unknown);
    suite_add_tcase(s, tc_strings);

    /* Integration tests */
    TCase* tc_integration = tcase_create("Integration");
    tcase_add_test(tc_integration, test_full_wiring_workflow);
    suite_add_tcase(s, tc_integration);

    return s;
}

int main(void)
{
    int number_failed;
    Suite* s = kg_module_wiring_suite();
    SRunner* sr = srunner_create(s);

    /* Run all tests */
    srunner_run_all(sr, CK_NORMAL);
    number_failed = srunner_ntests_failed(sr);
    srunner_free(sr);

    return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
