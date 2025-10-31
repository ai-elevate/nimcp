/**
 * @file nimcp_json.c
 * @brief Implementation of thread-safe JSON wrapper using Jansson
 */

#include "utils/nimcp_json.h"
#include <string.h>
#include <stdio.h>

// Static error messages
static const char* const ERROR_MESSAGES[] = {
    [JSON_SUCCESS] = "Success",
    [JSON_ERROR_INVALID_PARAM] = "Invalid parameter",
    [JSON_ERROR_MEMORY] = "Memory allocation failed",
    [JSON_ERROR_PARSE] = "JSON parsing failed",
    [JSON_ERROR_FILE] = "File operation failed",
    [JSON_ERROR_TYPE] = "Type mismatch",
    [JSON_ERROR_NOT_FOUND] = "Path not found",
    [JSON_ERROR_LOCK] = "Lock acquisition failed"
    [JSON_ERROR_NULL_VALUE] = "Null value encountered"
};

// Custom memory functions for Jansson
static void* json_malloc_wrapper(size_t size) {
    return memory_pool_alloc(NULL, size); // Global pool will be used
}

static void json_free_wrapper(void* ptr) {
    memory_pool_free(NULL, ptr);
}

// Internal helper functions
static JsonResult acquire_lock(JsonContext* ctx) {
    if (!ctx) return JSON_ERROR_INVALID_PARAM;
    if (thread_mutex_lock(&ctx->mutex) != 0) return JSON_ERROR_LOCK;
    return JSON_SUCCESS;
}

static void release_lock(JsonContext* ctx) {
    if (ctx) thread_mutex_unlock(&ctx->mutex);
}

static void log_json_error(JsonContext* ctx, const char* message, JsonResult result) {
    if (ctx && ctx->logger) {
        log_error(ctx->logger, "JSON error: %s (%s)", message, json_get_error(result));
    }
}

static json_t* resolve_json_path(json_t* root, const char* path, json_t** parent, char** last_key) {
    if (!root || !path) return NULL;
    
    json_t* current = root;
    json_t* prev = NULL;
    char* path_copy = strdup(path);
    char* saveptr = NULL;
    char* token = strtok_r(path_copy, "/", &saveptr);
    char* last_token = NULL;
    
    while (token) {
        last_token = token;
        prev = current;
        
        if (json_is_object(current)) {
            current = json_object_get(current, token);
        } else if (json_is_array(current)) {
            char* endptr;
            long index = strtol(token, &endptr, 10);
            if (*endptr != '\0' || index < 0) {
                current = NULL;
                break;
            }
            current = json_array_get(current, index);
        } else {
            current = NULL;
            break;
        }
        
        token = strtok_r(NULL, "/", &saveptr);
    }
    
    if (parent) *parent = prev;
    if (last_key && last_token) *last_key = strdup(last_token);
    free(path_copy);
    
    return current;
}

// Context management implementation
JsonResult json_create_context(JsonContext** ctx, memory_pool_t* pool, log_context_t* logger) {
    if (!ctx) return JSON_ERROR_INVALID_PARAM;
    
    JsonContext* new_ctx = NULL;
    if (pool) {
        new_ctx = memory_pool_alloc(pool, sizeof(JsonContext));
    } else {
        new_ctx = malloc(sizeof(JsonContext));
    }
    
    if (!new_ctx) return JSON_ERROR_MEMORY;
    
    memset(new_ctx, 0, sizeof(JsonContext));
    new_ctx->memory_pool = pool;
    new_ctx->logger = logger;
    new_ctx->owns_memory_pool = (pool == NULL);
    new_ctx->root = NULL;
    
    if (thread_mutex_init(&new_ctx->mutex) != 0) {
        if (pool) {
            memory_pool_free(pool, new_ctx);
        } else {
            free(new_ctx);
        }
        return JSON_ERROR_LOCK;
    }
    
    // Set up custom memory functions if using memory pool
    if (pool) {
        json_set_alloc_funcs(json_malloc_wrapper, json_free_wrapper);
    }
    
    *ctx = new_ctx;
    return JSON_SUCCESS;
}

void json_destroy_context(JsonContext* ctx) {
    if (!ctx) return;
    
    thread_mutex_lock(&ctx->mutex);
    if (ctx->root) {
        json_decref(ctx->root);
        ctx->root = NULL;
    }
    thread_mutex_unlock(&ctx->mutex);
    
    thread_mutex_destroy(&ctx->mutex);
    
    if (ctx->owns_memory_pool) {
        free(ctx);
    } else if (ctx->memory_pool) {
        memory_pool_free(ctx->memory_pool, ctx);
    }
}

// File operations implementation
JsonResult json_load_file(JsonContext* ctx, const char* filename, size_t flags) {
    if (!ctx || !filename) return JSON_ERROR_INVALID_PARAM;
    
    JsonResult result = acquire_lock(ctx);
    if (result != JSON_SUCCESS) return result;
    
    json_error_t error;
    json_t* new_root = json_load_file(filename, flags, &error);
    
    if (!new_root) {
        log_json_error(ctx, error.text, JSON_ERROR_PARSE);
        release_lock(ctx);
        return JSON_ERROR_PARSE;
    }
    
    if (ctx->root) {
        json_decref(ctx->root);
    }
    ctx->root = new_root;
    
    release_lock(ctx);
    return JSON_SUCCESS;
}

JsonResult json_dump_file(JsonContext* ctx, const char* filename, size_t flags) {
    if (!ctx || !filename || !ctx->root) return JSON_ERROR_INVALID_PARAM;
    
    JsonResult result = acquire_lock(ctx);
    if (result != JSON_SUCCESS) return result;
    
    if (json_dump_file(ctx->root, filename, flags) != 0) {
        release_lock(ctx);
        return JSON_ERROR_FILE;
    }
    
    release_lock(ctx);
    return JSON_SUCCESS;
}

// Value access implementation
JsonResult json_get_value(JsonContext* ctx, const char* path, json_t** value) {
    if (!ctx || !path || !value) return JSON_ERROR_INVALID_PARAM;
    
    JsonResult result = acquire_lock(ctx);
    if (result != JSON_SUCCESS) return result;
    
    *value = resolve_json_path(ctx->root, path, NULL, NULL);
    if (!*value) {
        release_lock(ctx);
        return JSON_ERROR_NOT_FOUND;
    }
    
    json_incref(*value);
    release_lock(ctx);
    return JSON_SUCCESS;
}

JsonResult json_set_value(JsonContext* ctx, const char* path, json_t* value) {
    if (!ctx || !path || !value) return JSON_ERROR_INVALID_PARAM;
    
    JsonResult result = acquire_lock(ctx);
    if (result != JSON_SUCCESS) return result;
    
    json_t* parent = NULL;
    char* last_key = NULL;
    
    if (!ctx->root) {
        ctx->root = json_object();
        if (!ctx->root) {
            release_lock(ctx);
            return JSON_ERROR_MEMORY;
        }
    }
    
    resolve_json_path(ctx->root, path, &parent, &last_key);
    
    if (!parent || !last_key) {
        release_lock(ctx);
        free(last_key);
        return JSON_ERROR_NOT_FOUND;
    }
    
    if (json_is_object(parent)) {
        json_object_set(parent, last_key, value);
    } else if (json_is_array(parent)) {
        char* endptr;
        long index = strtol(last_key, &endptr, 10);
        if (*endptr != '\0' || index < 0) {
            free(last_key);
            release_lock(ctx);
            return JSON_ERROR_INVALID_PARAM;
        }
        json_array_set(parent, index, value);
    } else {
        free(last_key);
        release_lock(ctx);
        return JSON_ERROR_TYPE;
    }
    
    free(last_key);
    release_lock(ctx);
    return JSON_SUCCESS;
}

// Convenience functions implementation

JsonResult json_get_string_value(JsonContext* ctx, const char* path, char* buffer, size_t size) {
    if (!buffer || size == 0) return JSON_ERROR_INVALID_PARAM;
    
    json_t* value = NULL;
    JsonResult result = json_get_value(ctx, path, &value);
    
    if (result != JSON_SUCCESS) return result;
    
    if (json_is_null(value)) {
        json_decref(value);
        return JSON_ERROR_NULL_VALUE;
    }
    
    if (!json_is_string(value)) {
        json_decref(value);
        return JSON_ERROR_TYPE;
    }
    
    const char* str = json_string_value(value);
    strncpy(buffer, str, size - 1);
    buffer[size - 1] = '\0';
    
    json_decref(value);
    return JSON_SUCCESS;
}

JsonResult json_get_integer_value(JsonContext* ctx, const char* path, int64_t* value) {
    if (!value) return JSON_ERROR_INVALID_PARAM;
    
    json_t* json_value = NULL;
    JsonResult result = json_get_value(ctx, path, &json_value);
    
    if (result != JSON_SUCCESS) return result;
    
    if (json_is_null(json_value)) {
        json_decref(json_value);
        return JSON_ERROR_NULL_VALUE;
    }
    
    if (!json_is_integer(json_value)) {
        json_decref(json_value);
        return JSON_ERROR_TYPE;
    }
    
    *value = json_integer_value(json_value);
    json_decref(json_value);
    return JSON_SUCCESS;
}

JsonResult json_get_boolean_value(JsonContext* ctx, const char* path, bool* value) {
    if (!value) return JSON_ERROR_INVALID_PARAM;
    
    json_t* json_value = NULL;
    JsonResult result = json_get_value(ctx, path, &json_value);
    
    if (result != JSON_SUCCESS) return result;
    
    if (json_is_null(json_value)) {
        json_decref(json_value);
        return JSON_ERROR_NULL_VALUE;
    }
    
    if (!json_is_boolean(json_value)) {
        json_decref(json_value);
        return JSON_ERROR_TYPE;
    }
    
    *value = json_boolean_value(json_value);
    json_decref(json_value);
    return JSON_SUCCESS;
}

JsonResult json_get_number_value(JsonContext* ctx, const char* path, double* value) {
    if (!value) return JSON_ERROR_INVALID_PARAM;
    
    json_t* json_value = NULL;
    JsonResult result = json_get_value(ctx, path, &json_value);
    
    if (result != JSON_SUCCESS) return result;
    
    if (json_is_null(json_value)) {
        json_decref(json_value);
        return JSON_ERROR_NULL_VALUE;
    }
    
    if (!json_is_number(json_value)) {
        json_decref(json_value);
        return JSON_ERROR_TYPE;
    }
    
    *value = json_number_value(json_value);
    json_decref(json_value);
    return JSON_SUCCESS;
}


// Error handling implementation
const char* json_get_error(JsonResult result) {
    if (result >= 0 || result < -7) return "Unknown error";
    return ERROR_MESSAGES[-result];
}

JsonResult json_is_null_value(JsonContext* ctx, const char* path, bool* is_null) {
    if (!ctx || !path || !is_null) return JSON_ERROR_INVALID_PARAM;
    
    JsonResult result = acquire_lock(ctx);
    if (result != JSON_SUCCESS) return result;
    
    json_t* value = resolve_json_path(ctx->root, path, NULL, NULL);
    if (!value) {
        release_lock(ctx);
        return JSON_ERROR_NOT_FOUND;
    }
    
    *is_null = json_is_null(value);
    release_lock(ctx);
    return JSON_SUCCESS;
}

JsonResult json_set_null_value(JsonContext* ctx, const char* path) {
    if (!ctx || !path) return JSON_ERROR_INVALID_PARAM;
    
    JsonResult result = acquire_lock(ctx);
    if (result != JSON_SUCCESS) return result;
    
    json_t* parent = NULL;
    char* last_key = NULL;
    
    if (!ctx->root) {
        ctx->root = json_object();
        if (!ctx->root) {
            release_lock(ctx);
            return JSON_ERROR_MEMORY;
        }
    }
    
    resolve_json_path(ctx->root, path, &parent, &last_key);
    
    if (!parent || !last_key) {
        free(last_key);
        release_lock(ctx);
        return JSON_ERROR_NOT_FOUND;
    }
    
    json_t* null_value = json_null();
    
    if (json_is_object(parent)) {
        json_object_set(parent, last_key, null_value);
    } else if (json_is_array(parent)) {
        char* endptr;
        long index = strtol(last_key, &endptr, 10);
        if (*endptr != '\0' || index < 0) {
            json_decref(null_value);
            free(last_key);
            release_lock(ctx);
            return JSON_ERROR_INVALID_PARAM;
        }
        json_array_set(parent, index, null_value);
    } else {
        json_decref(null_value);
        free(last_key);
        release_lock(ctx);
        return JSON_ERROR_TYPE;
    }
    
    json_decref(null_value);
    free(last_key);
    release_lock(ctx);
    return JSON_SUCCESS;
}
