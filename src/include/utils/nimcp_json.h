/**
 * @file nimcp_json.h
 * @brief Thread-safe JSON wrapper for NIMCP using Jansson
 */

#ifndef NIMCP_JSON_H
#define NIMCP_JSON_H

#include <jansson.h>
#include <stdbool.h>
#include <stdint.h>
#include "logging/nimcp_logging.h"
#include "nimcp_common.h"
#include "nimcp_memory.h"
#include "nimcp_thread.h"

// JSON operation result codes
typedef enum {
    JSON_SUCCESS = 0,
    JSON_ERROR_INVALID_PARAM = -1,
    JSON_ERROR_MEMORY = -2,
    JSON_ERROR_PARSE = -3,
    JSON_ERROR_FILE = -4,
    JSON_ERROR_TYPE = -5,
    JSON_ERROR_NOT_FOUND = -6,
    JSON_ERROR_LOCK = -7,
    JSON_ERROR_NULL_VALUE = -8
} JsonResult;

// Thread-safe JSON context
typedef struct JsonContext JsonContext;

#ifdef NIMCP_INTERNAL
// Internal context structure
struct JsonContext {
    nimcp_mutex_t mutex;
    json_t* root;
};
#endif  // NIMCP_INTERNAL

// Context management
JsonResult nimcp_json_create_context(JsonContext** ctx);
void nimcp_json_destroy_context(JsonContext* ctx);

// Thread-safe Jansson wrapper functions
JsonResult nimcp_json_load_file(JsonContext* ctx, const char* filename, size_t flags);
JsonResult nimcp_json_load_string(JsonContext* ctx, const char* input, size_t flags);
JsonResult nimcp_json_dump_file(JsonContext* ctx, const char* filename, size_t flags);
JsonResult nimcp_json_dump_string(JsonContext* ctx, char** output, size_t flags);

// Thread-safe accessors
JsonResult nimcp_json_get_value(JsonContext* ctx, const char* path, json_t** value);
JsonResult nimcp_json_set_value(JsonContext* ctx, const char* path, json_t* value);
JsonResult nimcp_json_delete_value(JsonContext* ctx, const char* path);

// Convenience functions for common operations
JsonResult nimcp_json_get_string_value(JsonContext* ctx, const char* path, char* buffer,
                                       size_t size);
JsonResult nimcp_json_get_integer_value(JsonContext* ctx, const char* path, int64_t* value);
JsonResult nimcp_json_get_boolean_value(JsonContext* ctx, const char* path, bool* value);
JsonResult nimcp_json_get_number_value(JsonContext* ctx, const char* path, double* value);

JsonResult nimcp_json_set_string_value(JsonContext* ctx, const char* path, const char* value);
JsonResult nimcp_json_set_integer_value(JsonContext* ctx, const char* path, int64_t value);
JsonResult nimcp_json_set_boolean_value(JsonContext* ctx, const char* path, bool value);
JsonResult nimcp_json_set_number_value(JsonContext* ctx, const char* path, double value);

// Null value functions
JsonResult nimcp_json_is_null_value(JsonContext* ctx, const char* path, bool* is_null);
JsonResult nimcp_json_set_null_value(JsonContext* ctx, const char* path);

// Error handling
const char* nimcp_json_get_error(JsonResult result);

#endif  // NIMCP_JSON_H
