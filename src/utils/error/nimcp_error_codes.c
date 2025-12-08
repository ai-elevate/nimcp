/**
 * @file nimcp_error_codes.c
 * @brief Error code system implementation
 *
 * @author NIMCP Team
 * @date 2025-11-09
 */

#include "utils/error/nimcp_error_codes.h"
#include "security/nimcp_security.h"
#include "security/nimcp_blood_brain_barrier.h"

#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_messages.h"
#include "utils/memory/nimcp_memory.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include "utils/memory/nimcp_unified_memory.h"
#include "utils/logging/nimcp_logging.h"

//=============================================================================
// Thread-Local Error Storage
//=============================================================================

static pthread_key_t g_error_key;
static pthread_once_t g_error_key_once = PTHREAD_ONCE_INIT;

static void error_destructor(void* value)
{
    if (value) {
        nimcp_free(value);
    }
}

static void create_error_key(void)
{
    pthread_key_create(&g_error_key, error_destructor);
}

static nimcp_error_info_t* get_thread_error(void)
{
    pthread_once(&g_error_key_once, create_error_key);

    nimcp_error_info_t* info = (nimcp_error_info_t*)pthread_getspecific(g_error_key);
    if (!info) {
        info = (nimcp_error_info_t*)nimcp_calloc(1, sizeof(nimcp_error_info_t));
        pthread_setspecific(g_error_key, info);
    }
    return info;
}

//=============================================================================
// Error Message Lookup
//=============================================================================

const char* nimcp_error_to_string(nimcp_error_t code)
{
    LOG_DEBUG("Entering nimcp_error_to_string");
    switch (code) {
        // Success
        case NIMCP_SUCCESS: return "Success";
        case NIMCP_SUCCESS_WITH_WARNINGS: return "Success with warnings";
        case NIMCP_SUCCESS_PARTIAL: return "Partial success";

        // Generic errors
        case NIMCP_ERROR_UNKNOWN: return "Unknown error";
        case NIMCP_ERROR_NOT_IMPLEMENTED: return "Feature not implemented";
        case NIMCP_ERROR_INVALID_PARAMETER: return "Invalid parameter";
        case NIMCP_ERROR_NULL_POINTER: return "Null pointer";
        case NIMCP_ERROR_OUT_OF_RANGE: return "Value out of range";
        case NIMCP_ERROR_INVALID_STATE: return "Invalid state";
        case NIMCP_ERROR_OPERATION_FAILED: return "Operation failed";
        case NIMCP_ERROR_NOT_INITIALIZED: return "Not initialized";
        case NIMCP_ERROR_ALREADY_EXISTS: return "Already exists";
        case NIMCP_ERROR_NOT_FOUND: return "Not found";
        case NIMCP_ERROR_TIMEOUT: return "Timeout";
        case NIMCP_ERROR_CANCELLED: return "Cancelled";
        case NIMCP_ERROR_PERMISSION_DENIED: return "Permission denied";

        // Memory errors
        case NIMCP_ERROR_NO_MEMORY: return "Out of memory";
        case NIMCP_ERROR_BUFFER_TOO_SMALL: return "Buffer too small";
        case NIMCP_ERROR_BUFFER_OVERFLOW: return "Buffer overflow";
        case NIMCP_ERROR_MEMORY_CORRUPTION: return "Memory corruption detected";
        case NIMCP_ERROR_INVALID_ADDRESS: return "Invalid memory address";
        case NIMCP_ERROR_MEMORY_LEAK: return "Memory leak detected";
        case NIMCP_ERROR_DOUBLE_FREE: return "Double free detected";

        // Brain/Network errors
        case NIMCP_ERROR_BRAIN_CREATION: return "Brain creation failed";
        case NIMCP_ERROR_BRAIN_INVALID: return "Invalid brain";
        case NIMCP_ERROR_NETWORK_CREATION: return "Network creation failed";
        case NIMCP_ERROR_NETWORK_INVALID: return "Invalid network";
        case NIMCP_ERROR_DIMENSION_MISMATCH: return "Dimension mismatch";
        case NIMCP_ERROR_WEIGHT_INIT: return "Weight initialization failed";
        case NIMCP_ERROR_FORWARD_PASS: return "Forward pass failed";
        case NIMCP_ERROR_BACKWARD_PASS: return "Backward pass failed";
        case NIMCP_ERROR_LEARNING_FAILED: return "Learning failed";
        case NIMCP_ERROR_INFERENCE_FAILED: return "Inference failed";
        case NIMCP_ERROR_COW_FAILED: return "Copy-on-write failed";
        case NIMCP_ERROR_CLONE_FAILED: return "Clone failed";

        // I/O errors
        case NIMCP_ERROR_FILE_NOT_FOUND: return "File not found";
        case NIMCP_ERROR_FILE_READ: return "File read error";
        case NIMCP_ERROR_FILE_WRITE: return "File write error";
        case NIMCP_ERROR_FILE_OPEN: return "File open error";
        case NIMCP_ERROR_FILE_CLOSE: return "File close error";
        case NIMCP_ERROR_FILE_CORRUPT: return "File corrupted";
        case NIMCP_ERROR_SERIALIZATION: return "Serialization failed";
        case NIMCP_ERROR_DESERIALIZATION: return "Deserialization failed";
        case NIMCP_ERROR_NETWORK_IO: return "Network I/O error";
        case NIMCP_ERROR_SOCKET_ERROR: return "Socket error";

        // Config errors
        case NIMCP_ERROR_CONFIG_INVALID: return "Invalid configuration";
        case NIMCP_ERROR_CONFIG_PARSE: return "Config parse error";
        case NIMCP_ERROR_CONFIG_MISSING: return "Required config missing";
        case NIMCP_ERROR_CONFIG_TYPE: return "Config type mismatch";
        case NIMCP_ERROR_CONFIG_RANGE: return "Config value out of range";
        case NIMCP_ERROR_CONFIG_RELOAD: return "Config reload failed";

        // Threading errors
        case NIMCP_ERROR_THREAD_CREATE: return "Thread creation failed";
        case NIMCP_ERROR_THREAD_JOIN: return "Thread join failed";
        case NIMCP_ERROR_MUTEX_LOCK: return "Mutex lock failed";
        case NIMCP_ERROR_MUTEX_UNLOCK: return "Mutex unlock failed";
        case NIMCP_ERROR_MUTEX_INIT: return "Mutex init failed";
        case NIMCP_ERROR_DEADLOCK: return "Deadlock detected";
        case NIMCP_ERROR_RACE_CONDITION: return "Race condition detected";
        case NIMCP_ERROR_THREAD_SYNC: return "Thread synchronization failed";

        // Signal/Crash errors
        case NIMCP_ERROR_SIGNAL_RECEIVED: return "Signal received";
        case NIMCP_ERROR_SIGSEGV: return "Segmentation fault";
        case NIMCP_ERROR_SIGABRT: return "Abort signal";
        case NIMCP_ERROR_SIGFPE: return "Floating point exception";
        case NIMCP_ERROR_SIGBUS: return "Bus error";
        case NIMCP_ERROR_SIGILL: return "Illegal instruction";
        case NIMCP_ERROR_CRASH_RECOVERY: return "Crash recovery failed";
        case NIMCP_ERROR_CHECKPOINT_SAVE: return "Checkpoint save failed";
        case NIMCP_ERROR_CHECKPOINT_LOAD: return "Checkpoint load failed";

        // Phase 10 errors
        case NIMCP_ERROR_WORKING_MEMORY: return "Working memory error";
        case NIMCP_ERROR_EMOTIONAL_TAGGING: return "Emotional tagging error";
        case NIMCP_ERROR_EXECUTIVE_CONTROL: return "Executive control error";
        case NIMCP_ERROR_SLEEP_WAKE: return "Sleep/wake cycle error";
        case NIMCP_ERROR_MENTAL_HEALTH: return "Mental health monitor error";
        case NIMCP_ERROR_THEORY_OF_MIND: return "Theory of mind error";
        case NIMCP_ERROR_EXPLANATIONS: return "Natural explanations error";
        case NIMCP_ERROR_META_LEARNING: return "Meta-learning error";
        case NIMCP_ERROR_PREDICTIVE: return "Predictive processing error";

        default: return "Unknown error code";
    }
}

const char* nimcp_error_get_category_name(nimcp_error_t code)
{
    LOG_DEBUG("Entering nimcp_error_get_category_name");
    int category = nimcp_error_get_category(code);

    switch (category) {
        case 0: return "Success";
        case 1: return "Generic Error";
        case 2: return "Memory Error";
        case 3: return "Brain/Network Error";
        case 4: return "I/O Error";
        case 5: return "Configuration Error";
        case 6: return "Threading Error";
        case 7: return "Signal/Crash Error";
        case 8: return "Cognitive Error";
        default: return "Unknown Category";
    }
}

void nimcp_error_set(nimcp_error_t code, const char* file, int line,
                     const char* function, const char* message)
{
    nimcp_error_info_t* info = get_thread_error();
    if (!info) return;

    info->code = code;
    info->file = file;
    info->line = line;
    info->function = function;
    info->message = message ? message : nimcp_error_to_string(code);
    info->context = NULL;
}

const nimcp_error_info_t* nimcp_error_get_last(void)
{
    LOG_DEBUG("Entering nimcp_error_get_last");
    return get_thread_error();
}

void nimcp_error_clear(void)
{
    LOG_DEBUG("Entering nimcp_error_clear");
    nimcp_error_info_t* info = get_thread_error();
    if (info) {
        memset(info, 0, sizeof(nimcp_error_info_t));
    }
}

void nimcp_error_print(nimcp_error_t code)
{
    LOG_DEBUG("Entering nimcp_error_print");
    fprintf(stderr, "[ERROR %d] %s: %s\n",
            code,
            nimcp_error_get_category_name(code),
            nimcp_error_to_string(code));
}

void nimcp_error_print_detailed(const nimcp_error_info_t* info)
{
    LOG_DEBUG("Entering nimcp_error_print_detailed");
    if (!info) return;

    fprintf(stderr, "\n=== ERROR DETAILS ===\n");
    fprintf(stderr, "Code: %d\n", info->code);
    fprintf(stderr, "Category: %s\n", nimcp_error_get_category_name(info->code));
    fprintf(stderr, "Message: %s\n", info->message ? info->message : "None");
    if (info->file) {
        fprintf(stderr, "File: %s:%d\n", info->file, info->line);
    }
    if (info->function) {
        fprintf(stderr, "Function: %s\n", info->function);
    }
    fprintf(stderr, "====================\n");
}
