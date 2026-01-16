/**
 * @file nimcp_exception.c
 * @brief Exception handling system implementation
 * @version 1.0.0
 * @date 2025-01-16
 *
 * WHAT: Core exception creation, lifecycle, and stack trace capture
 * WHY:  Provide rich exception information for debugging and immune integration
 * HOW:  Allocate exception structs, capture stack traces, manage ref counting
 *
 * @author NIMCP Development Team
 */

#include "utils/exception/nimcp_exception.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/platform/nimcp_platform_mutex.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <time.h>

#ifdef __linux__
#include <execinfo.h>
#endif

/* ============================================================================
 * Module State
 * ============================================================================ */

static bool g_exception_system_initialized = false;
static nimcp_platform_mutex_t* g_exception_mutex = NULL;

/* Thread-local current exception */
static _Thread_local nimcp_exception_t* tl_current_exception = NULL;

/* ============================================================================
 * Internal Helpers
 * ============================================================================ */

/**
 * @brief Get current timestamp in microseconds
 */
static uint64_t get_timestamp_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ULL + (uint64_t)ts.tv_nsec / 1000ULL;
}

/**
 * @brief Get category from error code
 */
nimcp_exception_category_t nimcp_exception_get_category_from_code(nimcp_error_t code) {
    if (code < 1000) return EXCEPTION_CATEGORY_GENERIC;

    /* GPU errors are in 1100-1199 range */
    if (code >= 1100 && code < 1200) return EXCEPTION_CATEGORY_GPU;

    int category = code / 1000;
    switch (category) {
        case 1:  return EXCEPTION_CATEGORY_GENERIC;
        case 2:  return EXCEPTION_CATEGORY_MEMORY;
        case 3:  return EXCEPTION_CATEGORY_BRAIN;
        case 4:  return EXCEPTION_CATEGORY_IO;
        case 5:  return EXCEPTION_CATEGORY_CONFIG;
        case 6:  return EXCEPTION_CATEGORY_THREADING;
        case 7:  return EXCEPTION_CATEGORY_SIGNAL;
        case 8:  return EXCEPTION_CATEGORY_COGNITIVE;
        default:
            if (code >= 10000 && code < 20000) {
                return EXCEPTION_CATEGORY_BRAIN_REGION;
            }
            return EXCEPTION_CATEGORY_GENERIC;
    }
}

/**
 * @brief Get severity from error code
 */
nimcp_exception_severity_t nimcp_exception_get_severity_from_code(nimcp_error_t code) {
    /* Fatal signals */
    if (code >= 7001 && code <= 7005) return EXCEPTION_SEVERITY_FATAL;

    /* Critical errors */
    if (code == NIMCP_ERROR_MEMORY_CORRUPTION ||
        code == NIMCP_ERROR_DEADLOCK ||
        code == NIMCP_ERROR_CRASH_RECOVERY) {
        return EXCEPTION_SEVERITY_CRITICAL;
    }

    /* Severe errors */
    if (code == NIMCP_ERROR_NO_MEMORY ||
        code == NIMCP_ERROR_BUFFER_OVERFLOW ||
        code == NIMCP_ERROR_DOUBLE_FREE ||
        code == NIMCP_ERROR_SIGSEGV) {
        return EXCEPTION_SEVERITY_SEVERE;
    }

    /* General errors */
    int category = code / 1000;
    switch (category) {
        case 0:  return EXCEPTION_SEVERITY_DEBUG;    /* Success codes */
        case 1:  return EXCEPTION_SEVERITY_ERROR;    /* Generic */
        case 2:  return EXCEPTION_SEVERITY_SEVERE;   /* Memory */
        case 3:  return EXCEPTION_SEVERITY_ERROR;    /* Brain */
        case 4:  return EXCEPTION_SEVERITY_ERROR;    /* I/O */
        case 5:  return EXCEPTION_SEVERITY_WARNING;  /* Config */
        case 6:  return EXCEPTION_SEVERITY_SEVERE;   /* Threading */
        case 7:  return EXCEPTION_SEVERITY_CRITICAL; /* Signal */
        case 8:  return EXCEPTION_SEVERITY_ERROR;    /* Cognitive */
        default:
            if (code >= 10000 && code < 20000) {
                return EXCEPTION_SEVERITY_ERROR;     /* Brain region */
            }
            return EXCEPTION_SEVERITY_ERROR;
    }
}

/**
 * @brief Get suggested recovery action
 */
nimcp_recovery_action_t nimcp_exception_get_suggested_recovery(nimcp_exception_t* ex) {
    if (!ex) return RECOVERY_ACTION_NONE;

    switch (ex->category) {
        case EXCEPTION_CATEGORY_MEMORY:
            return RECOVERY_ACTION_GC;

        case EXCEPTION_CATEGORY_BRAIN:
        case EXCEPTION_CATEGORY_BRAIN_REGION:
            if (ex->code == NIMCP_ERROR_FORWARD_PASS ||
                ex->code == NIMCP_ERROR_BACKWARD_PASS) {
                return RECOVERY_ACTION_ROLLBACK;
            }
            return RECOVERY_ACTION_REDUCE_LOAD;

        case EXCEPTION_CATEGORY_IO:
            return RECOVERY_ACTION_RETRY;

        case EXCEPTION_CATEGORY_THREADING:
            if (ex->code == NIMCP_ERROR_DEADLOCK) {
                return RECOVERY_ACTION_RESTART_THREAD;
            }
            return RECOVERY_ACTION_RETRY;

        case EXCEPTION_CATEGORY_SIGNAL:
            return RECOVERY_ACTION_EMERGENCY_SAVE;

        case EXCEPTION_CATEGORY_GPU:
            return RECOVERY_ACTION_CLEAR_CACHE;

        case EXCEPTION_CATEGORY_SECURITY:
            return RECOVERY_ACTION_QUARANTINE;

        default:
            return RECOVERY_ACTION_NONE;
    }
}

/* ============================================================================
 * Stack Trace
 * ============================================================================ */

size_t nimcp_exception_capture_stack_trace(nimcp_stack_trace_t* trace, int skip_frames) {
    if (!trace) return 0;

    memset(trace, 0, sizeof(nimcp_stack_trace_t));

#ifdef __linux__
    void* addresses[NIMCP_EXCEPTION_MAX_STACK_DEPTH + 8];
    int count = backtrace(addresses, NIMCP_EXCEPTION_MAX_STACK_DEPTH + skip_frames + 1);

    /* Skip requested frames plus our own frame */
    int start = skip_frames + 1;
    if (start >= count) return 0;

    char** symbols = backtrace_symbols(addresses + start, count - start);

    for (int i = 0; i < count - start && trace->depth < NIMCP_EXCEPTION_MAX_STACK_DEPTH; i++) {
        trace->frames[trace->depth].address = addresses[start + i];
        trace->frames[trace->depth].function = symbols ? symbols[i] : NULL;
        trace->frames[trace->depth].file = NULL;
        trace->frames[trace->depth].line = 0;
        trace->depth++;
    }

    /* Note: symbols memory is intentionally not freed here as we store
     * the pointers. In practice, this is OK for exception handling since
     * exceptions are rare. For production, consider copying strings. */
#endif

    return trace->depth;
}

/* ============================================================================
 * Epitope Generation
 * ============================================================================ */

/**
 * @brief Simple hash function for epitope generation
 */
static uint32_t hash_string(const char* str, size_t len) {
    uint32_t hash = 5381;
    for (size_t i = 0; i < len && str[i]; i++) {
        hash = ((hash << 5) + hash) + (uint8_t)str[i];
    }
    return hash;
}

size_t nimcp_exception_generate_epitope(nimcp_exception_t* ex) {
    if (!ex) return 0;

    memset(ex->epitope, 0, NIMCP_EXCEPTION_EPITOPE_SIZE);
    size_t offset = 0;

    /* Error code (4 bytes) */
    memcpy(ex->epitope + offset, &ex->code, sizeof(ex->code));
    offset += sizeof(ex->code);

    /* Category (4 bytes) */
    uint32_t cat = (uint32_t)ex->category;
    memcpy(ex->epitope + offset, &cat, sizeof(cat));
    offset += sizeof(cat);

    /* Severity (4 bytes) */
    uint32_t sev = (uint32_t)ex->severity;
    memcpy(ex->epitope + offset, &sev, sizeof(sev));
    offset += sizeof(sev);

    /* File hash (4 bytes) */
    uint32_t file_hash = ex->file ? hash_string(ex->file, 256) : 0;
    memcpy(ex->epitope + offset, &file_hash, sizeof(file_hash));
    offset += sizeof(file_hash);

    /* Function hash (4 bytes) */
    uint32_t func_hash = ex->function ? hash_string(ex->function, 128) : 0;
    memcpy(ex->epitope + offset, &func_hash, sizeof(func_hash));
    offset += sizeof(func_hash);

    /* Line number (4 bytes) */
    uint32_t line = (uint32_t)ex->line;
    memcpy(ex->epitope + offset, &line, sizeof(line));
    offset += sizeof(line);

    /* Stack trace hash (32 bytes) - use first few addresses */
    for (size_t i = 0; i < 8 && i < ex->stack_trace.depth && offset < NIMCP_EXCEPTION_EPITOPE_SIZE - 4; i++) {
        uint32_t addr = (uint32_t)(uintptr_t)ex->stack_trace.frames[i].address;
        memcpy(ex->epitope + offset, &addr, sizeof(addr));
        offset += sizeof(addr);
    }

    ex->epitope_len = offset;
    return offset;
}

/* ============================================================================
 * Exception Creation
 * ============================================================================ */

nimcp_exception_t* nimcp_exception_create(
    nimcp_error_t code,
    nimcp_exception_severity_t severity,
    const char* file,
    int line,
    const char* func,
    const char* format,
    ...
) {
    nimcp_exception_t* ex = nimcp_calloc(1, sizeof(nimcp_exception_t));
    if (!ex) return NULL;

    ex->type = EXCEPTION_TYPE_BASE;
    ex->category = nimcp_exception_get_category_from_code(code);
    ex->code = code;
    ex->severity = severity;
    ex->file = file;
    ex->line = line;
    ex->function = func;
    ex->timestamp_us = get_timestamp_us();
    ex->ref_count = 1;

    /* Format message */
    if (format) {
        va_list args;
        va_start(args, format);
        vsnprintf(ex->message, NIMCP_EXCEPTION_MAX_MESSAGE, format, args);
        va_end(args);
    }

    /* Capture stack trace (skip 2 frames: this function + caller) */
    nimcp_exception_capture_stack_trace(&ex->stack_trace, 2);

    /* Generate immune epitope */
    nimcp_exception_generate_epitope(ex);

    /* Determine suggested recovery */
    ex->suggested_action = nimcp_exception_get_suggested_recovery(ex);

    return ex;
}

nimcp_memory_exception_t* nimcp_memory_exception_create(
    nimcp_error_t code,
    nimcp_exception_severity_t severity,
    const char* file,
    int line,
    const char* func,
    size_t requested_size,
    const char* format,
    ...
) {
    nimcp_memory_exception_t* ex = nimcp_calloc(1, sizeof(nimcp_memory_exception_t));
    if (!ex) return NULL;

    ex->base.type = EXCEPTION_TYPE_MEMORY;
    ex->base.category = EXCEPTION_CATEGORY_MEMORY;
    ex->base.code = code;
    ex->base.severity = severity;
    ex->base.file = file;
    ex->base.line = line;
    ex->base.function = func;
    ex->base.timestamp_us = get_timestamp_us();
    ex->base.ref_count = 1;
    ex->base.suggested_action = RECOVERY_ACTION_GC;

    ex->requested_size = requested_size;
    ex->is_heap = true;

    /* Format message */
    if (format) {
        va_list args;
        va_start(args, format);
        vsnprintf(ex->base.message, NIMCP_EXCEPTION_MAX_MESSAGE, format, args);
        va_end(args);
    }

    nimcp_exception_capture_stack_trace(&ex->base.stack_trace, 2);
    nimcp_exception_generate_epitope(&ex->base);

    return ex;
}

nimcp_brain_exception_t* nimcp_brain_exception_create(
    nimcp_error_t code,
    nimcp_exception_severity_t severity,
    const char* file,
    int line,
    const char* func,
    uint32_t brain_id,
    const char* region_name,
    const char* format,
    ...
) {
    nimcp_brain_exception_t* ex = nimcp_calloc(1, sizeof(nimcp_brain_exception_t));
    if (!ex) return NULL;

    ex->base.type = EXCEPTION_TYPE_BRAIN;
    ex->base.category = EXCEPTION_CATEGORY_BRAIN;
    ex->base.code = code;
    ex->base.severity = severity;
    ex->base.file = file;
    ex->base.line = line;
    ex->base.function = func;
    ex->base.timestamp_us = get_timestamp_us();
    ex->base.ref_count = 1;
    ex->base.suggested_action = RECOVERY_ACTION_ROLLBACK;

    ex->brain_id = brain_id;
    ex->region_name = region_name;

    if (format) {
        va_list args;
        va_start(args, format);
        vsnprintf(ex->base.message, NIMCP_EXCEPTION_MAX_MESSAGE, format, args);
        va_end(args);
    }

    nimcp_exception_capture_stack_trace(&ex->base.stack_trace, 2);
    nimcp_exception_generate_epitope(&ex->base);

    return ex;
}

nimcp_io_exception_t* nimcp_io_exception_create(
    nimcp_error_t code,
    nimcp_exception_severity_t severity,
    const char* file,
    int line,
    const char* func,
    const char* path,
    const char* format,
    ...
) {
    nimcp_io_exception_t* ex = nimcp_calloc(1, sizeof(nimcp_io_exception_t));
    if (!ex) return NULL;

    ex->base.type = EXCEPTION_TYPE_IO;
    ex->base.category = EXCEPTION_CATEGORY_IO;
    ex->base.code = code;
    ex->base.severity = severity;
    ex->base.file = file;
    ex->base.line = line;
    ex->base.function = func;
    ex->base.timestamp_us = get_timestamp_us();
    ex->base.ref_count = 1;
    ex->base.suggested_action = RECOVERY_ACTION_RETRY;

    ex->path = path;

    if (format) {
        va_list args;
        va_start(args, format);
        vsnprintf(ex->base.message, NIMCP_EXCEPTION_MAX_MESSAGE, format, args);
        va_end(args);
    }

    nimcp_exception_capture_stack_trace(&ex->base.stack_trace, 2);
    nimcp_exception_generate_epitope(&ex->base);

    return ex;
}

nimcp_threading_exception_t* nimcp_threading_exception_create(
    nimcp_error_t code,
    nimcp_exception_severity_t severity,
    const char* file,
    int line,
    const char* func,
    uint64_t thread_id,
    const char* format,
    ...
) {
    nimcp_threading_exception_t* ex = nimcp_calloc(1, sizeof(nimcp_threading_exception_t));
    if (!ex) return NULL;

    ex->base.type = EXCEPTION_TYPE_THREADING;
    ex->base.category = EXCEPTION_CATEGORY_THREADING;
    ex->base.code = code;
    ex->base.severity = severity;
    ex->base.file = file;
    ex->base.line = line;
    ex->base.function = func;
    ex->base.timestamp_us = get_timestamp_us();
    ex->base.ref_count = 1;
    ex->base.suggested_action = RECOVERY_ACTION_RESTART_THREAD;

    ex->thread_id = thread_id;
    ex->is_deadlock = (code == NIMCP_ERROR_DEADLOCK);

    if (format) {
        va_list args;
        va_start(args, format);
        vsnprintf(ex->base.message, NIMCP_EXCEPTION_MAX_MESSAGE, format, args);
        va_end(args);
    }

    nimcp_exception_capture_stack_trace(&ex->base.stack_trace, 2);
    nimcp_exception_generate_epitope(&ex->base);

    return ex;
}

nimcp_security_exception_t* nimcp_security_exception_create(
    nimcp_error_t code,
    nimcp_exception_severity_t severity,
    const char* file,
    int line,
    const char* func,
    uint32_t threat_type,
    const char* format,
    ...
) {
    nimcp_security_exception_t* ex = nimcp_calloc(1, sizeof(nimcp_security_exception_t));
    if (!ex) return NULL;

    ex->base.type = EXCEPTION_TYPE_SECURITY;
    ex->base.category = EXCEPTION_CATEGORY_SECURITY;
    ex->base.code = code;
    ex->base.severity = severity;
    ex->base.file = file;
    ex->base.line = line;
    ex->base.function = func;
    ex->base.timestamp_us = get_timestamp_us();
    ex->base.ref_count = 1;
    ex->base.suggested_action = RECOVERY_ACTION_QUARANTINE;

    ex->threat_type = threat_type;
    ex->quarantine_required = (severity >= EXCEPTION_SEVERITY_SEVERE);

    if (format) {
        va_list args;
        va_start(args, format);
        vsnprintf(ex->base.message, NIMCP_EXCEPTION_MAX_MESSAGE, format, args);
        va_end(args);
    }

    nimcp_exception_capture_stack_trace(&ex->base.stack_trace, 2);
    nimcp_exception_generate_epitope(&ex->base);

    return ex;
}

nimcp_gpu_exception_t* nimcp_gpu_exception_create(
    nimcp_error_t code,
    nimcp_exception_severity_t severity,
    const char* file,
    int line,
    const char* func,
    int device_id,
    int cuda_error,
    const char* format,
    ...
) {
    nimcp_gpu_exception_t* ex = nimcp_calloc(1, sizeof(nimcp_gpu_exception_t));
    if (!ex) return NULL;

    ex->base.type = EXCEPTION_TYPE_GPU;
    ex->base.category = EXCEPTION_CATEGORY_GPU;
    ex->base.code = code;
    ex->base.severity = severity;
    ex->base.file = file;
    ex->base.line = line;
    ex->base.function = func;
    ex->base.timestamp_us = get_timestamp_us();
    ex->base.ref_count = 1;
    ex->base.suggested_action = RECOVERY_ACTION_CLEAR_CACHE;

    ex->device_id = device_id;
    ex->cuda_error = cuda_error;

    if (format) {
        va_list args;
        va_start(args, format);
        vsnprintf(ex->base.message, NIMCP_EXCEPTION_MAX_MESSAGE, format, args);
        va_end(args);
    }

    nimcp_exception_capture_stack_trace(&ex->base.stack_trace, 2);
    nimcp_exception_generate_epitope(&ex->base);

    return ex;
}

/* ============================================================================
 * Signal Exception API
 * ============================================================================ */

#include <signal.h>
#include "utils/signal/nimcp_signal_handler.h"

nimcp_error_t nimcp_signal_to_error_code(int signal_number) {
    switch (signal_number) {
        case SIGSEGV: return NIMCP_ERROR_SIGSEGV;
        case SIGABRT: return NIMCP_ERROR_SIGABRT;
        case SIGFPE:  return NIMCP_ERROR_SIGFPE;
        case SIGBUS:  return NIMCP_ERROR_SIGBUS;
        case SIGILL:  return NIMCP_ERROR_SIGILL;
        default:      return NIMCP_ERROR_SIGNAL_RECEIVED;
    }
}

const char* nimcp_signal_name(int signal_number) {
    switch (signal_number) {
        case SIGSEGV: return "SIGSEGV";
        case SIGABRT: return "SIGABRT";
        case SIGFPE:  return "SIGFPE";
        case SIGBUS:  return "SIGBUS";
        case SIGILL:  return "SIGILL";
        case SIGTERM: return "SIGTERM";
        case SIGINT:  return "SIGINT";
        case SIGHUP:  return "SIGHUP";
        default:      return "UNKNOWN";
    }
}

nimcp_signal_exception_t* nimcp_signal_exception_create(
    int signal_number,
    void* fault_address,
    const char* file,
    int line,
    const char* func,
    const char* format,
    ...
) {
    nimcp_signal_exception_t* ex = nimcp_calloc(1, sizeof(nimcp_signal_exception_t));
    if (!ex) return NULL;

    nimcp_error_t code = nimcp_signal_to_error_code(signal_number);

    ex->base.type = EXCEPTION_TYPE_SIGNAL;
    ex->base.category = EXCEPTION_CATEGORY_SIGNAL;
    ex->base.code = code;
    ex->base.severity = EXCEPTION_SEVERITY_FATAL;
    ex->base.file = file;
    ex->base.line = line;
    ex->base.function = func;
    ex->base.timestamp_us = get_timestamp_us();
    ex->base.ref_count = 1;
    ex->base.suggested_action = RECOVERY_ACTION_EMERGENCY_SAVE;

    ex->signal_number = signal_number;
    ex->fault_address = fault_address;
    ex->recovery_attempted = false;
    ex->siglongjmp_executed = false;
    ex->retry_count = 0;

    /* Format message */
    if (format) {
        va_list args;
        va_start(args, format);
        vsnprintf(ex->base.message, NIMCP_EXCEPTION_MAX_MESSAGE, format, args);
        va_end(args);
    } else {
        snprintf(ex->base.message, NIMCP_EXCEPTION_MAX_MESSAGE,
                 "Signal %d (%s) at address %p",
                 signal_number, nimcp_signal_name(signal_number), fault_address);
    }

    nimcp_exception_capture_stack_trace(&ex->base.stack_trace, 2);
    nimcp_exception_generate_epitope(&ex->base);

    return ex;
}

nimcp_signal_exception_t* nimcp_signal_exception_create_from_context(
    const struct signal_crash_context* ctx
) {
    if (!ctx) return NULL;

    nimcp_signal_exception_t* ex = nimcp_calloc(1, sizeof(nimcp_signal_exception_t));
    if (!ex) return NULL;

    nimcp_error_t code = nimcp_signal_to_error_code(ctx->signal);

    ex->base.type = EXCEPTION_TYPE_SIGNAL;
    ex->base.category = EXCEPTION_CATEGORY_SIGNAL;
    ex->base.code = code;
    ex->base.severity = EXCEPTION_SEVERITY_FATAL;
    ex->base.file = NULL;  /* Not available from crash context */
    ex->base.line = 0;
    ex->base.function = NULL;
    ex->base.timestamp_us = get_timestamp_us();
    ex->base.ref_count = 1;
    ex->base.suggested_action = RECOVERY_ACTION_EMERGENCY_SAVE;

    /* Copy signal-specific fields from crash context */
    ex->signal_number = ctx->signal;
    ex->fault_address = ctx->fault_address;
    ex->instruction_pointer = ctx->instruction_pointer;
    ex->stack_pointer = ctx->stack_pointer;
    ex->base_pointer = ctx->base_pointer;
    ex->recovery_attempted = false;
    ex->siglongjmp_executed = false;
    ex->retry_count = 0;

    /* Copy memory region info */
    if (ctx->memory_region[0] != '\0') {
        strncpy(ex->memory_region, ctx->memory_region,
                NIMCP_SIGNAL_EXCEPTION_MEMORY_REGION_SIZE - 1);
        ex->memory_region[NIMCP_SIGNAL_EXCEPTION_MEMORY_REGION_SIZE - 1] = '\0';
    }

    /* Format message */
    snprintf(ex->base.message, NIMCP_EXCEPTION_MAX_MESSAGE,
             "Signal %d (%s) at fault address %p, IP=%p",
             ctx->signal, nimcp_signal_name(ctx->signal),
             ctx->fault_address, ctx->instruction_pointer);

    /* Copy backtrace from context if available */
    if (ctx->backtrace_depth > 0) {
        ex->base.stack_trace.depth = (size_t)ctx->backtrace_depth;
        if (ex->base.stack_trace.depth > NIMCP_EXCEPTION_MAX_STACK_DEPTH) {
            ex->base.stack_trace.depth = NIMCP_EXCEPTION_MAX_STACK_DEPTH;
        }
        for (size_t i = 0; i < ex->base.stack_trace.depth; i++) {
            ex->base.stack_trace.frames[i].address = ctx->backtrace[i];
            ex->base.stack_trace.frames[i].function = NULL;
            ex->base.stack_trace.frames[i].file = NULL;
            ex->base.stack_trace.frames[i].line = 0;
        }
    }

    nimcp_exception_generate_epitope(&ex->base);

    return ex;
}

/* ============================================================================
 * Aggregate Exception API
 * ============================================================================ */

nimcp_aggregate_exception_t* nimcp_aggregate_exception_create(
    nimcp_error_t code,
    nimcp_exception_severity_t severity,
    const char* file,
    int line,
    const char* func,
    const char* format,
    ...
) {
    nimcp_aggregate_exception_t* ex = nimcp_calloc(1, sizeof(nimcp_aggregate_exception_t));
    if (!ex) return NULL;

    ex->base.type = EXCEPTION_TYPE_AGGREGATE;
    ex->base.category = nimcp_exception_get_category_from_code(code);
    ex->base.code = code;
    ex->base.severity = severity;
    ex->base.file = file;
    ex->base.line = line;
    ex->base.function = func;
    ex->base.timestamp_us = get_timestamp_us();
    ex->base.ref_count = 1;
    ex->base.suggested_action = RECOVERY_ACTION_NONE;

    ex->child_count = 0;

    /* Format message */
    if (format) {
        va_list args;
        va_start(args, format);
        vsnprintf(ex->base.message, NIMCP_EXCEPTION_MAX_MESSAGE, format, args);
        va_end(args);
    }

    nimcp_exception_capture_stack_trace(&ex->base.stack_trace, 2);
    nimcp_exception_generate_epitope(&ex->base);

    return ex;
}

int nimcp_aggregate_exception_add(nimcp_aggregate_exception_t* agg, nimcp_exception_t* child) {
    if (!agg || !child) return -1;

    if (agg->child_count >= NIMCP_EXCEPTION_MAX_CHILDREN) {
        return -1; /* Aggregate is full */
    }

    /* Take a reference to the child */
    agg->children[agg->child_count] = nimcp_exception_ref(child);
    agg->child_count++;

    /* Update aggregate severity to max of children */
    if (child->severity > agg->base.severity) {
        agg->base.severity = child->severity;
    }

    /* Update suggested action if child has higher priority */
    if (child->suggested_action > agg->base.suggested_action) {
        agg->base.suggested_action = child->suggested_action;
    }

    return 0;
}

size_t nimcp_aggregate_exception_count(const nimcp_aggregate_exception_t* agg) {
    return agg ? agg->child_count : 0;
}

nimcp_exception_t* nimcp_aggregate_exception_get(const nimcp_aggregate_exception_t* agg, size_t index) {
    if (!agg || index >= agg->child_count) {
        return NULL;
    }
    return agg->children[index];
}

/* ============================================================================
 * Exception Context API
 * ============================================================================ */

int nimcp_exception_set_context(nimcp_exception_t* ex, const char* key, const char* value) {
    if (!ex || !key || !value) return -1;

    /* First, check if key already exists and update it */
    for (size_t i = 0; i < ex->context_count; i++) {
        if (strncmp(ex->context[i].key, key, NIMCP_EXCEPTION_MAX_CONTEXT_KEY) == 0) {
            /* Update existing entry */
            strncpy(ex->context[i].value, value, NIMCP_EXCEPTION_MAX_CONTEXT_VALUE - 1);
            ex->context[i].value[NIMCP_EXCEPTION_MAX_CONTEXT_VALUE - 1] = '\0';
            return 0;
        }
    }

    /* Key not found, add new entry if space available */
    if (ex->context_count >= NIMCP_EXCEPTION_MAX_CONTEXT_ENTRIES) {
        return -1; /* Context is full */
    }

    strncpy(ex->context[ex->context_count].key, key, NIMCP_EXCEPTION_MAX_CONTEXT_KEY - 1);
    ex->context[ex->context_count].key[NIMCP_EXCEPTION_MAX_CONTEXT_KEY - 1] = '\0';
    strncpy(ex->context[ex->context_count].value, value, NIMCP_EXCEPTION_MAX_CONTEXT_VALUE - 1);
    ex->context[ex->context_count].value[NIMCP_EXCEPTION_MAX_CONTEXT_VALUE - 1] = '\0';
    ex->context_count++;

    return 0;
}

const char* nimcp_exception_get_context(const nimcp_exception_t* ex, const char* key) {
    if (!ex || !key) return NULL;

    for (size_t i = 0; i < ex->context_count; i++) {
        if (strncmp(ex->context[i].key, key, NIMCP_EXCEPTION_MAX_CONTEXT_KEY) == 0) {
            return ex->context[i].value;
        }
    }

    return NULL;
}

int nimcp_exception_remove_context(nimcp_exception_t* ex, const char* key) {
    if (!ex || !key) return -1;

    for (size_t i = 0; i < ex->context_count; i++) {
        if (strncmp(ex->context[i].key, key, NIMCP_EXCEPTION_MAX_CONTEXT_KEY) == 0) {
            /* Shift remaining entries down */
            for (size_t j = i; j < ex->context_count - 1; j++) {
                memcpy(&ex->context[j], &ex->context[j + 1], sizeof(nimcp_exception_context_entry_t));
            }
            ex->context_count--;
            return 0;
        }
    }

    return -1; /* Key not found */
}

size_t nimcp_exception_context_count(const nimcp_exception_t* ex) {
    return ex ? ex->context_count : 0;
}

/* ============================================================================
 * Lifecycle Management
 * ============================================================================ */

nimcp_exception_t* nimcp_exception_ref(nimcp_exception_t* ex) {
    if (!ex) return NULL;
    __atomic_add_fetch(&ex->ref_count, 1, __ATOMIC_SEQ_CST);
    return ex;
}

void nimcp_exception_unref(nimcp_exception_t* ex) {
    if (!ex) return;

    int32_t old = __atomic_fetch_sub(&ex->ref_count, 1, __ATOMIC_SEQ_CST);
    if (old <= 1) {
        /* Free cause chain */
        if (ex->cause) {
            nimcp_exception_unref(ex->cause);
        }

        /* Free aggregate children */
        if (ex->type == EXCEPTION_TYPE_AGGREGATE) {
            nimcp_aggregate_exception_t* agg = (nimcp_aggregate_exception_t*)ex;
            for (size_t i = 0; i < agg->child_count; i++) {
                if (agg->children[i]) {
                    nimcp_exception_unref(agg->children[i]);
                }
            }
        }

        nimcp_free(ex);
    }
}

void nimcp_exception_set_cause(nimcp_exception_t* ex, nimcp_exception_t* cause) {
    if (!ex) return;

    /* Release old cause */
    if (ex->cause) {
        nimcp_exception_unref(ex->cause);
    }

    /* Set new cause (takes ownership of reference) */
    ex->cause = cause;
}

nimcp_exception_t* nimcp_exception_get_cause(nimcp_exception_t* ex) {
    return ex ? ex->cause : NULL;
}

/* ============================================================================
 * Logging/Printing
 * ============================================================================ */

const char* nimcp_exception_severity_to_string(nimcp_exception_severity_t severity) {
    switch (severity) {
        case EXCEPTION_SEVERITY_DEBUG:    return "DEBUG";
        case EXCEPTION_SEVERITY_INFO:     return "INFO";
        case EXCEPTION_SEVERITY_WARNING:  return "WARNING";
        case EXCEPTION_SEVERITY_ERROR:    return "ERROR";
        case EXCEPTION_SEVERITY_SEVERE:   return "SEVERE";
        case EXCEPTION_SEVERITY_CRITICAL: return "CRITICAL";
        case EXCEPTION_SEVERITY_FATAL:    return "FATAL";
        default: return "UNKNOWN";
    }
}

const char* nimcp_exception_category_to_string(nimcp_exception_category_t category) {
    switch (category) {
        case EXCEPTION_CATEGORY_GENERIC:    return "GENERIC";
        case EXCEPTION_CATEGORY_MEMORY:     return "MEMORY";
        case EXCEPTION_CATEGORY_BRAIN:      return "BRAIN";
        case EXCEPTION_CATEGORY_IO:         return "IO";
        case EXCEPTION_CATEGORY_CONFIG:     return "CONFIG";
        case EXCEPTION_CATEGORY_THREADING:  return "THREADING";
        case EXCEPTION_CATEGORY_SIGNAL:     return "SIGNAL";
        case EXCEPTION_CATEGORY_COGNITIVE:  return "COGNITIVE";
        case EXCEPTION_CATEGORY_GPU:        return "GPU";
        case EXCEPTION_CATEGORY_BRAIN_REGION: return "BRAIN_REGION";
        case EXCEPTION_CATEGORY_SECURITY:   return "SECURITY";
        default: return "UNKNOWN";
    }
}

const char* nimcp_exception_type_to_string(nimcp_exception_type_t type) {
    switch (type) {
        case EXCEPTION_TYPE_BASE:      return "BASE";
        case EXCEPTION_TYPE_MEMORY:    return "MEMORY";
        case EXCEPTION_TYPE_BRAIN:     return "BRAIN";
        case EXCEPTION_TYPE_IO:        return "IO";
        case EXCEPTION_TYPE_THREADING: return "THREADING";
        case EXCEPTION_TYPE_SECURITY:  return "SECURITY";
        case EXCEPTION_TYPE_COGNITIVE: return "COGNITIVE";
        case EXCEPTION_TYPE_GPU:       return "GPU";
        case EXCEPTION_TYPE_AGGREGATE: return "AGGREGATE";
        case EXCEPTION_TYPE_SIGNAL:    return "SIGNAL";
        default: return "UNKNOWN";
    }
}

const char* nimcp_recovery_action_to_string(nimcp_recovery_action_t action) {
    switch (action) {
        case RECOVERY_ACTION_NONE:             return "NONE";
        case RECOVERY_ACTION_RETRY:            return "RETRY";
        case RECOVERY_ACTION_GC:               return "GC";
        case RECOVERY_ACTION_COMPACT:          return "COMPACT";
        case RECOVERY_ACTION_ROLLBACK:         return "ROLLBACK";
        case RECOVERY_ACTION_RESTART_THREAD:   return "RESTART_THREAD";
        case RECOVERY_ACTION_RESTART_COMPONENT: return "RESTART_COMPONENT";
        case RECOVERY_ACTION_QUARANTINE:       return "QUARANTINE";
        case RECOVERY_ACTION_REDUCE_LOAD:      return "REDUCE_LOAD";
        case RECOVERY_ACTION_CLEAR_CACHE:      return "CLEAR_CACHE";
        case RECOVERY_ACTION_EMERGENCY_SAVE:   return "EMERGENCY_SAVE";
        case RECOVERY_ACTION_GRACEFUL_SHUTDOWN: return "GRACEFUL_SHUTDOWN";
        default: return "UNKNOWN";
    }
}

void nimcp_exception_log(const nimcp_exception_t* ex) {
    if (!ex) return;

    LOG_ERROR("[Exception] %s (%d) at %s:%d in %s(): %s",
              nimcp_exception_severity_to_string(ex->severity),
              ex->code,
              ex->file ? ex->file : "unknown",
              ex->line,
              ex->function ? ex->function : "unknown",
              ex->message);
}

void nimcp_exception_print(const nimcp_exception_t* ex) {
    if (!ex) return;

    char buffer[2048];
    nimcp_exception_to_string(ex, buffer, sizeof(buffer));
    fprintf(stderr, "%s\n", buffer);
}

size_t nimcp_exception_to_string(
    const nimcp_exception_t* ex,
    char* buffer,
    size_t buffer_size
) {
    if (!ex || !buffer || buffer_size == 0) return 0;

    int written = snprintf(buffer, buffer_size,
        "Exception: %s\n"
        "  Code: %d\n"
        "  Category: %s\n"
        "  Severity: %s\n"
        "  Message: %s\n"
        "  Location: %s:%d in %s()\n"
        "  Suggested Recovery: %s\n",
        nimcp_exception_type_to_string(ex->type),
        ex->code,
        nimcp_exception_category_to_string(ex->category),
        nimcp_exception_severity_to_string(ex->severity),
        ex->message,
        ex->file ? ex->file : "unknown",
        ex->line,
        ex->function ? ex->function : "unknown",
        nimcp_recovery_action_to_string(ex->suggested_action)
    );

    return (size_t)(written > 0 ? written : 0);
}

size_t nimcp_stack_trace_to_string(
    const nimcp_stack_trace_t* trace,
    char* buffer,
    size_t buffer_size
) {
    if (!trace || !buffer || buffer_size == 0) return 0;

    size_t offset = 0;
    offset += snprintf(buffer + offset, buffer_size - offset, "Stack trace (%zu frames):\n", trace->depth);

    for (size_t i = 0; i < trace->depth && offset < buffer_size - 1; i++) {
        const nimcp_stack_frame_t* frame = &trace->frames[i];
        offset += snprintf(buffer + offset, buffer_size - offset,
            "  #%zu %p %s\n",
            i,
            frame->address,
            frame->function ? frame->function : "(unknown)"
        );
    }

    return offset;
}

/* ============================================================================
 * Thread-Local Exception Context
 * ============================================================================ */

void nimcp_exception_set_current(nimcp_exception_t* ex) {
    if (tl_current_exception) {
        nimcp_exception_unref(tl_current_exception);
    }
    tl_current_exception = ex ? nimcp_exception_ref(ex) : NULL;
}

nimcp_exception_t* nimcp_exception_get_current(void) {
    return tl_current_exception;
}

void nimcp_exception_clear_current(void) {
    if (tl_current_exception) {
        nimcp_exception_unref(tl_current_exception);
        tl_current_exception = NULL;
    }
}

/* ============================================================================
 * System Initialization
 * ============================================================================ */

int nimcp_exception_system_init(void) {
    if (g_exception_system_initialized) return 0;

    g_exception_mutex = nimcp_platform_mutex_create();
    if (!g_exception_mutex) return -1;

    g_exception_system_initialized = true;
    return 0;
}

void nimcp_exception_system_shutdown(void) {
    if (!g_exception_system_initialized) return;

    nimcp_exception_clear_current();

    if (g_exception_mutex) {
        nimcp_platform_mutex_destroy(g_exception_mutex);
        nimcp_free(g_exception_mutex);
        g_exception_mutex = NULL;
    }

    g_exception_system_initialized = false;
}

bool nimcp_exception_system_is_initialized(void) {
    return g_exception_system_initialized;
}
