#include <stddef.h>  /* for NULL */
//=============================================================================
// nimcp_logging.c - Enhanced Logging System Implementation
//=============================================================================
/**
 * @file nimcp_logging.c
 * @brief High-performance async logging with rotation, filtering, and metrics
 *
 * WHAT: Complete logging implementation with all enhanced features
 * WHY:  Production-ready logging for NIMCP brain modules
 * HOW:  Lock-free ring buffer, background writer, pluggable destinations
 *
 * ARCHITECTURE:
 * - Lock-free SPMC ring buffer for async logging
 * - Background writer thread for file/console/syslog output
 * - Pluggable formatters (text, JSON, syslog)
 * - Automatic rotation with compression
 * - Rate limiting with token bucket algorithm
 * - Thread-local context for correlation IDs
 *
 * @author NIMCP Development Team
 * @date 2025-11-28
 * @version 2.0.0
 */

#include "utils/logging/nimcp_logging.h"
#include "security/nimcp_security.h"
#include "security/nimcp_blood_brain_barrier.h"

#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_messages.h"
#include "utils/platform/nimcp_platform_mutex.h"
#include "utils/platform/nimcp_platform_cond.h"
#include "utils/platform/nimcp_platform_thread.h"
#include "utils/platform/nimcp_platform_time.h"
#include "utils/platform/nimcp_platform_once.h"
#include "utils/thread/nimcp_atomic.h"
#include "utils/exception/nimcp_exception_macros.h"

#include <errno.h>
#include <libgen.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>  // For strcasecmp
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#ifdef __linux__
#include <sys/syscall.h>
#include <syslog.h>
#include "utils/memory/nimcp_memory.h"
#endif

//=============================================================================
// Forward Declarations for Security and Memory Integration
//=============================================================================

/**
 * WHAT: Forward declarations for security integration
 * WHY:  Avoid header conflicts, use void* for opaque types
 */
typedef struct nimcp_sec_integration nimcp_sec_integration_t;
typedef struct unified_mem_manager_struct* unified_mem_manager_t;
typedef struct unified_mem_handle_struct* unified_mem_handle_t;

#define NIMCP_SEC_CAT_UTILITY 7

extern int nimcp_sec_register_module(
    nimcp_sec_integration_t* ctx,
    const char* name,
    int category,
    uint32_t* module_id
);

extern int nimcp_sec_unregister_module(
    nimcp_sec_integration_t* ctx,
    uint32_t module_id
);

extern int nimcp_sec_record_interaction(
    nimcp_sec_integration_t* ctx,
    uint32_t module_id,
    bool success,
    double weight
);

// Unified memory forward declarations
typedef struct {
    size_t size;
    const void* initial_data;
    int strategy;
    bool enable_cow;
    size_t alignment;
} unified_mem_request_t;

extern unified_mem_handle_t unified_mem_alloc(
    unified_mem_manager_t manager,
    const unified_mem_request_t* request
);

extern void unified_mem_free(unified_mem_handle_t handle);
extern void* unified_mem_write(unified_mem_handle_t handle);

//=============================================================================
// Constants
//=============================================================================

#define LOG_MODULE_NAME "logging"
#include "utils/fault_tolerance/nimcp_health_agent_macros.h"

NIMCP_DECLARE_HEALTH_AGENT_ATOMIC(logging)

#define DEFAULT_LOG_PATH "/var/log/nimcp/nimcp.log"
#define MAX_MODULE_FILTERS 64
#define CONTEXT_KEY_SIZE 32

/** @brief ANSI color codes for console output */
static const char* ANSI_COLORS[] = {
    "\033[37m",      // TRACE - white
    "\033[36m",      // DEBUG - cyan
    "\033[32m",      // INFO - green
    "\033[33m",      // WARN - yellow
    "\033[31m",      // ERROR - red
    "\033[35;1m",    // FATAL - bold magenta
    ""               // OFF
};
static const char* ANSI_RESET = "\033[0m";

/** @brief Level name strings */
static const char* LEVEL_NAMES[] = {
    "TRACE", "DEBUG", "INFO", "WARN", "ERROR", "FATAL", "OFF"
};

/** @brief Short level names for compact format */
static const char* LEVEL_NAMES_SHORT[] = {
    "T", "D", "I", "W", "E", "F", "-"
};

//=============================================================================
// Internal Structures
//=============================================================================

/**
 * @brief Module filter entry
 */
typedef struct {
    char module[NIMCP_LOG_MAX_MODULE_LEN];
    log_level_t level;
    bool enabled;
} log_module_filter_t;

/**
 * @brief Rate limiter state (token bucket)
 */
typedef struct {
    nimcp_atomic_uint64_t tokens;
    nimcp_atomic_uint64_t last_refill_time_ns;
    uint32_t max_tokens;
    uint32_t refill_rate;  // tokens per second
    bool enabled;
} log_rate_limiter_t;

/**
 * @brief Ring buffer for async logging
 */
typedef struct {
    nimcp_log_entry_t* entries;
    size_t capacity;
    nimcp_atomic_uint64_t write_pos;
    nimcp_atomic_uint64_t read_pos;
    nimcp_atomic_uint64_t drop_count;
} log_ring_buffer_t;

/**
 * @brief Allocation header for unified memory tracking
 */
typedef struct {
    unified_mem_handle_t handle;
    size_t size;
} log_alloc_header_t;

/**
 * @brief Logger instance structure
 */
struct nimcp_logger_struct {
    uint32_t magic;                     /**< Magic for validation */

    // Configuration
    log_level_t level;
    uint32_t destinations;
    nimcp_log_format_t format;
    nimcp_log_async_mode_t async_mode;
    nimcp_log_color_mode_t color_mode;
    bool include_source_location;

    // File output
    FILE* file;
    char file_path[PATH_MAX];
    nimcp_platform_mutex_t file_mutex;

    // Rotation
    nimcp_log_rotation_config_t rotation;
    size_t current_file_size;
    time_t last_rotation_time;

    // Rate limiting
    log_rate_limiter_t rate_limiter;

    // Module filtering
    log_module_filter_t module_filters[MAX_MODULE_FILTERS];
    size_t module_filter_count;
    nimcp_platform_mutex_t filter_mutex;

    // Callbacks
    nimcp_log_output_callback_t callback;
    void* callback_user_data;
    nimcp_log_filter_t filter;
    void* filter_user_data;

    // Async components
    log_ring_buffer_t ring_buffer;
    nimcp_platform_thread_t writer_thread;
    nimcp_platform_mutex_t async_mutex;
    nimcp_platform_cond_t async_cond;
    nimcp_atomic_bool_t async_running;
    uint32_t flush_interval_ms;

    // Statistics
    nimcp_atomic_uint64_t messages_logged;
    nimcp_atomic_uint64_t messages_dropped;
    nimcp_atomic_uint64_t messages_filtered;
    nimcp_atomic_uint64_t level_counts[LOG_LEVEL_COUNT];
    nimcp_atomic_uint64_t async_writes;
    nimcp_atomic_uint64_t sync_writes;
    nimcp_atomic_uint64_t buffer_overflows;
    nimcp_atomic_uint64_t flush_operations;
    nimcp_atomic_uint64_t rotations_performed;
    nimcp_atomic_uint64_t bytes_written;
    nimcp_atomic_uint64_t rate_limit_hits;
    nimcp_atomic_uint64_t total_log_time_ns;
    nimcp_atomic_uint64_t max_log_time_ns;
    nimcp_atomic_uint32_t sequence;
    uint64_t start_time_ns;

    // Integration
    unified_mem_manager_t memory_mgr;
    nimcp_sec_integration_t* security_ctx;
    uint32_t security_module_id;

    // Context (thread-local simulation via mutex)
    nimcp_platform_mutex_t context_mutex;
    char current_context[NIMCP_LOG_MAX_CONTEXT_LEN];

    // Syslog
    bool syslog_opened;
};

/**
 * @brief Log context structure
 */
struct nimcp_log_context_struct {
    nimcp_logger_t logger;
    char context_id[NIMCP_LOG_MAX_CONTEXT_LEN];
    char previous_context[NIMCP_LOG_MAX_CONTEXT_LEN];
};

//=============================================================================
// Global State
//=============================================================================

static nimcp_logger_t g_global_logger = NULL;
static nimcp_platform_mutex_t g_global_mutex;
static nimcp_platform_once_t g_global_once = NIMCP_PLATFORM_ONCE_INIT;
static bool g_global_initialized = false;

/**
 * WHAT: One-time global mutex initialization
 * WHY:  Ensure thread-safe initialization of global logger mutex
 */
static void init_global_mutex(void) {
    nimcp_platform_mutex_init(&g_global_mutex, false);
}

//=============================================================================
// Memory Allocation Helpers
//=============================================================================

/**
 * WHAT: Allocate memory with unified memory support
 * WHY:  Use memory pool when available, fall back to malloc
 */
static void* log_alloc(nimcp_logger_t logger, size_t size) {
    if (logger && logger->memory_mgr) {
        size_t total = sizeof(log_alloc_header_t) + size;
        unified_mem_request_t req = {
            .size = total,
            .initial_data = NULL,
            .strategy = 3,  // UNIFIED_STRATEGY_POOL_DIRECT
            .enable_cow = false,
            .alignment = 0
        };
        unified_mem_handle_t handle = unified_mem_alloc(logger->memory_mgr, &req);
        if (handle) {
            void* base = unified_mem_write(handle);
            if (base) {
                log_alloc_header_t* header = (log_alloc_header_t*)base;
                header->handle = handle;
                header->size = size;
                return (char*)base + sizeof(log_alloc_header_t);
            }
            unified_mem_free(handle);
        }
    }

    // Fallback to malloc
    size_t total = sizeof(log_alloc_header_t) + size;
    void* base = nimcp_malloc(total);
    if (base) {
        log_alloc_header_t* header = (log_alloc_header_t*)base;
        header->handle = NULL;
        header->size = size;
        return (char*)base + sizeof(log_alloc_header_t);
    }
    NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "log_alloc: validation failed");
    return NULL;
}

/**
 * WHAT: Free memory allocated with log_alloc
 */
static void log_free(void* ptr) {
    if (!ptr) return;
    log_alloc_header_t* header = (log_alloc_header_t*)((char*)ptr - sizeof(log_alloc_header_t));
    if (header->handle) {
        unified_mem_free(header->handle);
    } else {
        nimcp_free(header);
    }
}

//=============================================================================
// Utility Functions
//=============================================================================

/**
 * WHAT: Get current thread ID
 */
static uint32_t get_thread_id(void) {
#ifdef __linux__
    return (uint32_t)syscall(SYS_gettid);
#else
    return (uint32_t)(uintptr_t)pthread_self();
#endif
}

/**
 * WHAT: Get monotonic time in nanoseconds
 */
static uint64_t get_time_ns(void) {
    return nimcp_platform_time_monotonic_ms() * 1000000ULL;
}

/**
 * WHAT: Get wall clock time in nanoseconds
 */
static uint64_t get_wall_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

/**
 * WHAT: Create directory and all parent directories
 * WHY:  mkdir() only creates one level, we need recursive creation
 */
static int mkdir_p(const char* path) {
    char tmp[PATH_MAX];
    char* p = NULL;
    size_t len;

    snprintf(tmp, sizeof(tmp), "%s", path);
    len = strlen(tmp);
    if (len > 0 && tmp[len - 1] == '/') {
        tmp[len - 1] = '\0';
    }

    for (p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = '\0';
            if (mkdir(tmp, 0755) != 0 && errno != EEXIST) {
                NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "mkdir_p: validation failed");
                return -1;
            }
            *p = '/';
        }
    }

    if (mkdir(tmp, 0755) != 0 && errno != EEXIST) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "mkdir_p: validation failed");
        return -1;
    }
    return 0;
}

/**
 * WHAT: Extract directory from file path
 */
static void get_directory(const char* path, char* dir, size_t dir_size) {
    char tmp[PATH_MAX];
    snprintf(tmp, sizeof(tmp), "%s", path);
    char* d = dirname(tmp);
    snprintf(dir, dir_size, "%s", d);
}

/**
 * WHAT: Get basename from file path
 */
static const char* get_basename(const char* path) {
    const char* base = strrchr(path, '/');
    return base ? base + 1 : path;
}

//=============================================================================
// Ring Buffer Implementation
//=============================================================================

/**
 * WHAT: Initialize ring buffer
 */
static bool ring_buffer_init(log_ring_buffer_t* rb, size_t capacity, nimcp_logger_t logger) {
    rb->entries = (nimcp_log_entry_t*)log_alloc(logger, capacity * sizeof(nimcp_log_entry_t));
    if (!rb->entries) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "ring_buffer_init: rb->entries is NULL");
        return false;
    }

    rb->capacity = capacity;
    nimcp_atomic_init_u64(&rb->write_pos, 0);
    nimcp_atomic_init_u64(&rb->read_pos, 0);
    nimcp_atomic_init_u64(&rb->drop_count, 0);

    memset(rb->entries, 0, capacity * sizeof(nimcp_log_entry_t));
    return true;
}

/**
 * WHAT: Destroy ring buffer
 */
static void ring_buffer_destroy(log_ring_buffer_t* rb) {
    if (rb->entries) {
        log_free(rb->entries);
        rb->entries = NULL;
    }
}

/**
 * WHAT: Push entry to ring buffer (producer)
 * WHY:  Lock-free push for async logging
 * HOW:  CAS on write position
 */
static bool ring_buffer_push(log_ring_buffer_t* rb, const nimcp_log_entry_t* entry) {
    uint64_t write_pos, read_pos, next_pos;

    // Try to reserve a slot
    for (int retry = 0; retry < 3; retry++) {
        write_pos = nimcp_atomic_load_u64(&rb->write_pos, NIMCP_MEMORY_ORDER_ACQUIRE);
        read_pos = nimcp_atomic_load_u64(&rb->read_pos, NIMCP_MEMORY_ORDER_ACQUIRE);
        next_pos = write_pos + 1;

        // Check if buffer is full
        if (next_pos - read_pos >= rb->capacity) {
            nimcp_atomic_fetch_add_u64(&rb->drop_count, 1, NIMCP_MEMORY_ORDER_RELAXED);
            NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_BUFFER_OVERFLOW, "ring_buffer_push: capacity exceeded");
            return false;  // Buffer full
        }

        // Try to claim the slot
        if (nimcp_atomic_compare_exchange_u64(&rb->write_pos, &write_pos, next_pos,
                                               NIMCP_MEMORY_ORDER_ACQ_REL)) {
            // Successfully claimed slot, copy entry
            size_t index = write_pos % rb->capacity;
            memcpy(&rb->entries[index], entry, sizeof(nimcp_log_entry_t));
            return true;
        }
        // CAS failed, retry
    }

    nimcp_atomic_fetch_add_u64(&rb->drop_count, 1, NIMCP_MEMORY_ORDER_RELAXED);
    NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "ring_buffer_push: operation failed");
    return false;
}

/**
 * WHAT: Pop entry from ring buffer (consumer)
 * WHY:  Single consumer, no CAS needed
 */
static bool ring_buffer_pop(log_ring_buffer_t* rb, nimcp_log_entry_t* entry) {
    uint64_t read_pos = nimcp_atomic_load_u64(&rb->read_pos, NIMCP_MEMORY_ORDER_ACQUIRE);
    uint64_t write_pos = nimcp_atomic_load_u64(&rb->write_pos, NIMCP_MEMORY_ORDER_ACQUIRE);

    if (read_pos >= write_pos) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "ring_buffer_pop: capacity exceeded");
        return false;  // Buffer empty
    }

    size_t index = read_pos % rb->capacity;
    memcpy(entry, &rb->entries[index], sizeof(nimcp_log_entry_t));

    nimcp_atomic_store_u64(&rb->read_pos, read_pos + 1, NIMCP_MEMORY_ORDER_RELEASE);
    return true;
}

/**
 * WHAT: Get number of entries in ring buffer
 */
static size_t ring_buffer_size(log_ring_buffer_t* rb) {
    uint64_t write_pos = nimcp_atomic_load_u64(&rb->write_pos, NIMCP_MEMORY_ORDER_ACQUIRE);
    uint64_t read_pos = nimcp_atomic_load_u64(&rb->read_pos, NIMCP_MEMORY_ORDER_ACQUIRE);
    return (size_t)(write_pos - read_pos);
}

//=============================================================================
// Rate Limiter Implementation
//=============================================================================

/**
 * WHAT: Initialize rate limiter
 */
static void rate_limiter_init(log_rate_limiter_t* rl, const nimcp_log_rate_limit_config_t* config) {
    rl->enabled = config->enabled;
    rl->max_tokens = config->burst_size > 0 ? config->burst_size : config->max_per_second;
    rl->refill_rate = config->max_per_second;
    nimcp_atomic_init_u64(&rl->tokens, rl->max_tokens);
    nimcp_atomic_init_u64(&rl->last_refill_time_ns, get_time_ns());
}

/**
 * WHAT: Try to consume a token
 * WHY:  Token bucket rate limiting
 * HOW:  Refill tokens based on elapsed time, then try to consume
 */
static bool rate_limiter_try_acquire(log_rate_limiter_t* rl) {
    if (!rl->enabled) {
        return true;
    }

    uint64_t now = get_time_ns();
    uint64_t last_refill = nimcp_atomic_load_u64(&rl->last_refill_time_ns, NIMCP_MEMORY_ORDER_ACQUIRE);
    uint64_t elapsed_ns = now - last_refill;

    // Refill tokens (1 token per 1/rate seconds)
    if (elapsed_ns > 0 && rl->refill_rate > 0) {
        uint64_t tokens_to_add = (elapsed_ns * rl->refill_rate) / 1000000000ULL;
        if (tokens_to_add > 0) {
            uint64_t expected = nimcp_atomic_load_u64(&rl->tokens, NIMCP_MEMORY_ORDER_ACQUIRE);
            uint64_t desired;
            do {
                desired = expected + tokens_to_add;
                if (desired > rl->max_tokens) {
                    desired = rl->max_tokens;
                }
            } while (!nimcp_atomic_compare_exchange_u64(&rl->tokens, &expected, desired, NIMCP_MEMORY_ORDER_ACQ_REL));

            nimcp_atomic_store_u64(&rl->last_refill_time_ns, now, NIMCP_MEMORY_ORDER_RELEASE);
        }
    }

    // Try to consume a token
    uint64_t tokens = nimcp_atomic_load_u64(&rl->tokens, NIMCP_MEMORY_ORDER_ACQUIRE);
    if (tokens > 0) {
        nimcp_atomic_fetch_sub_u64(&rl->tokens, 1, NIMCP_MEMORY_ORDER_ACQ_REL);
        return true;
    }

    NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "rate_limiter_try_acquire: validation failed");
    return false;
}

//=============================================================================
// Formatter Implementation
//=============================================================================

/**
 * WHAT: Format timestamp
 */
static void format_timestamp(uint64_t timestamp_ns, char* buf, size_t buf_size, bool include_ms) {
    time_t seconds = (time_t)(timestamp_ns / 1000000000ULL);
    struct tm tm_info;
    localtime_r(&seconds, &tm_info);

    if (include_ms) {
        uint32_t ms = (uint32_t)((timestamp_ns % 1000000000ULL) / 1000000ULL);
        snprintf(buf, buf_size, "%04d-%02d-%02d %02d:%02d:%02d.%03u",
                 tm_info.tm_year + 1900, tm_info.tm_mon + 1, tm_info.tm_mday,
                 tm_info.tm_hour, tm_info.tm_min, tm_info.tm_sec, ms);
    } else {
        strftime(buf, buf_size, "%Y-%m-%d %H:%M:%S", &tm_info);
    }
}

/**
 * WHAT: Format log entry as text
 */
static size_t format_text(const nimcp_log_entry_t* entry, char* buf, size_t buf_size,
                          bool include_source, bool color, bool compact) {
    char timestamp[32];
    const char* color_start = "";
    const char* color_end = "";

    if (color && entry->level < LOG_LEVEL_COUNT) {
        color_start = ANSI_COLORS[entry->level];
        color_end = ANSI_RESET;
    }

    if (compact) {
        // Compact format: [L] message
        return snprintf(buf, buf_size, "%s[%s]%s %s\n",
                        color_start, LEVEL_NAMES_SHORT[entry->level], color_end,
                        entry->message);
    }

    format_timestamp(entry->timestamp_ns, timestamp, sizeof(timestamp), true);

    size_t len = 0;

    // Base format: [timestamp] [LEVEL] message
    if (entry->module[0] != '\0') {
        if (include_source && entry->file[0] != '\0') {
            len = snprintf(buf, buf_size, "[%s] %s[%-5s]%s [%s] [%s:%d] %s\n",
                           timestamp, color_start, LEVEL_NAMES[entry->level], color_end,
                           entry->module, get_basename(entry->file), entry->line,
                           entry->message);
        } else {
            len = snprintf(buf, buf_size, "[%s] %s[%-5s]%s [%s] %s\n",
                           timestamp, color_start, LEVEL_NAMES[entry->level], color_end,
                           entry->module, entry->message);
        }
    } else {
        if (include_source && entry->file[0] != '\0') {
            len = snprintf(buf, buf_size, "[%s] %s[%-5s]%s [%s:%d] %s\n",
                           timestamp, color_start, LEVEL_NAMES[entry->level], color_end,
                           get_basename(entry->file), entry->line, entry->message);
        } else {
            len = snprintf(buf, buf_size, "[%s] %s[%-5s]%s %s\n",
                           timestamp, color_start, LEVEL_NAMES[entry->level], color_end,
                           entry->message);
        }
    }

    return len;
}

/**
 * WHAT: Format log entry as JSON
 */
static size_t format_json(const nimcp_log_entry_t* entry, char* buf, size_t buf_size) {
    char timestamp[32];
    format_timestamp(entry->timestamp_ns, timestamp, sizeof(timestamp), true);

    // Escape message for JSON
    char escaped_msg[NIMCP_LOG_MAX_MESSAGE_LEN * 2];
    const char* src = entry->message;
    char* dst = escaped_msg;
    char* end = escaped_msg + sizeof(escaped_msg) - 1;

    while (*src && dst < end) {
        switch (*src) {
            case '"': if (dst + 2 <= end) { *dst++ = '\\'; *dst++ = '"'; } break;
            case '\\': if (dst + 2 <= end) { *dst++ = '\\'; *dst++ = '\\'; } break;
            case '\n': if (dst + 2 <= end) { *dst++ = '\\'; *dst++ = 'n'; } break;
            case '\r': if (dst + 2 <= end) { *dst++ = '\\'; *dst++ = 'r'; } break;
            case '\t': if (dst + 2 <= end) { *dst++ = '\\'; *dst++ = 't'; } break;
            default: *dst++ = *src; break;
        }
        src++;
    }
    *dst = '\0';

    return snprintf(buf, buf_size,
        "{\"timestamp\":\"%s\",\"level\":\"%s\",\"thread\":%u,"
        "\"module\":\"%s\",\"file\":\"%s\",\"line\":%d,"
        "\"context\":\"%s\",\"seq\":%u,\"message\":\"%s\"}\n",
        timestamp, LEVEL_NAMES[entry->level], entry->thread_id,
        entry->module, entry->file[0] ? get_basename(entry->file) : "",
        entry->line, entry->context_id, entry->sequence, escaped_msg);
}

/**
 * WHAT: Format log entry for syslog
 */
static size_t format_syslog(const nimcp_log_entry_t* entry, char* buf, size_t buf_size) {
    // Syslog format: <priority>timestamp hostname app[pid]: message
    return snprintf(buf, buf_size, "%s[%u]: %s\n",
                    entry->module[0] ? entry->module : "nimcp",
                    entry->thread_id, entry->message);
}

//=============================================================================
// Output Destination Implementation
//=============================================================================

/**
 * WHAT: Write entry to file
 */
static void write_to_file(nimcp_logger_t logger, const nimcp_log_entry_t* entry,
                          const char* formatted, size_t len) {
    if (!logger->file) return;

    nimcp_platform_mutex_lock(&logger->file_mutex);

    size_t written = fwrite(formatted, 1, len, logger->file);
    fflush(logger->file);

    if (written > 0) {
        logger->current_file_size += written;
        nimcp_atomic_fetch_add_u64(&logger->bytes_written, written, NIMCP_MEMORY_ORDER_RELAXED);
    }

    // Check for rotation
    if (logger->rotation.mode != NIMCP_LOG_ROTATE_NONE) {
        bool should_rotate = false;

        if (logger->rotation.mode == NIMCP_LOG_ROTATE_SIZE ||
            logger->rotation.mode == NIMCP_LOG_ROTATE_BOTH) {
            if (logger->current_file_size >= logger->rotation.max_file_size) {
                should_rotate = true;
            }
        }

        if (logger->rotation.mode == NIMCP_LOG_ROTATE_TIME ||
            logger->rotation.mode == NIMCP_LOG_ROTATE_BOTH) {
            time_t now = time(NULL);
            time_t interval = 0;
            switch (logger->rotation.interval) {
                case NIMCP_LOG_ROTATE_HOURLY: interval = 3600; break;
                case NIMCP_LOG_ROTATE_DAILY: interval = 86400; break;
                case NIMCP_LOG_ROTATE_WEEKLY: interval = 604800; break;
                case NIMCP_LOG_ROTATE_MONTHLY: interval = 2592000; break;
            }
            if (interval > 0 && (now - logger->last_rotation_time) >= interval) {
                should_rotate = true;
            }
        }

        if (should_rotate) {
            nimcp_platform_mutex_unlock(&logger->file_mutex);
            nimcp_log_rotate(logger);
            return;
        }
    }

    nimcp_platform_mutex_unlock(&logger->file_mutex);
}

/**
 * WHAT: Write entry to console
 */
static void write_to_console(nimcp_logger_t logger, const nimcp_log_entry_t* entry,
                             bool use_color) {
    char buf[NIMCP_LOG_MAX_MESSAGE_LEN + 256];
    size_t len = format_text(entry, buf, sizeof(buf),
                             logger->include_source_location, use_color, false);

    // Use stderr for WARN and above, stdout for others
    FILE* out = (entry->level >= LOG_LEVEL_WARN) ? stderr : stdout;
    fwrite(buf, 1, len, out);
    fflush(out);
}

/**
 * WHAT: Write entry to syslog
 */
static void write_to_syslog(nimcp_logger_t logger, const nimcp_log_entry_t* entry) {
#ifdef __linux__
    if (!logger->syslog_opened) {
        openlog("nimcp", LOG_PID | LOG_NDELAY, LOG_USER);
        logger->syslog_opened = true;
    }

    int priority;
    switch (entry->level) {
        case LOG_LEVEL_TRACE:
        case LOG_LEVEL_DEBUG: priority = LOG_DEBUG; break;
        case LOG_LEVEL_INFO: priority = LOG_INFO; break;
        case LOG_LEVEL_WARN: priority = LOG_WARNING; break;
        case LOG_LEVEL_ERROR: priority = LOG_ERR; break;
        case LOG_LEVEL_FATAL: priority = LOG_CRIT; break;
        default: priority = LOG_INFO; break;
    }

    syslog(priority, "[%s] %s", entry->module[0] ? entry->module : "nimcp", entry->message);
#else
    (void)logger;
    (void)entry;
#endif
}

/**
 * WHAT: Write entry to callback
 */
static void write_to_callback(nimcp_logger_t logger, const nimcp_log_entry_t* entry) {
    if (logger->callback) {
        logger->callback(
            entry->level,
            entry->timestamp_ns / 1000000000ULL,
            entry->module[0] ? entry->module : NULL,
            entry->file[0] ? entry->file : NULL,
            entry->line,
            entry->message,
            logger->callback_user_data
        );
    }
}

/**
 * WHAT: Write entry to all configured destinations
 */
static void write_entry(nimcp_logger_t logger, const nimcp_log_entry_t* entry) {
    char formatted[NIMCP_LOG_MAX_MESSAGE_LEN + 512];
    size_t len = 0;

    // Determine color support
    bool use_color = false;
    if (logger->color_mode == NIMCP_LOG_COLOR_ON) {
        use_color = true;
    } else if (logger->color_mode == NIMCP_LOG_COLOR_AUTO) {
        use_color = nimcp_log_is_tty();
    }

    // Format message based on format type
    switch (logger->format) {
        case NIMCP_LOG_FORMAT_JSON:
            len = format_json(entry, formatted, sizeof(formatted));
            break;
        case NIMCP_LOG_FORMAT_COMPACT:
            len = format_text(entry, formatted, sizeof(formatted),
                              false, use_color, true);
            break;
        case NIMCP_LOG_FORMAT_SYSLOG:
            len = format_syslog(entry, formatted, sizeof(formatted));
            break;
        case NIMCP_LOG_FORMAT_TEXT:
        default:
            len = format_text(entry, formatted, sizeof(formatted),
                              logger->include_source_location, false, false);
            break;
    }

    // Write to file
    if (logger->destinations & NIMCP_LOG_DEST_FILE) {
        write_to_file(logger, entry, formatted, len);
    }

    // Write to console
    if (logger->destinations & NIMCP_LOG_DEST_CONSOLE) {
        write_to_console(logger, entry, use_color);
    }

    // Write to syslog
    if (logger->destinations & NIMCP_LOG_DEST_SYSLOG) {
        write_to_syslog(logger, entry);
    }

    // Write to callback
    if (logger->destinations & NIMCP_LOG_DEST_CALLBACK) {
        write_to_callback(logger, entry);
    }
}

//=============================================================================
// Async Writer Thread
//=============================================================================

/**
 * WHAT: Background writer thread function
 * WHY:  Process async log messages without blocking producers
 */
static void* async_writer_thread(void* arg) {
    nimcp_logger_t logger = (nimcp_logger_t)arg;
    nimcp_log_entry_t entry;

    while (nimcp_atomic_load_bool(&logger->async_running, NIMCP_MEMORY_ORDER_ACQUIRE)) {
        bool wrote = false;

        // Process all available entries
        while (ring_buffer_pop(&logger->ring_buffer, &entry)) {
            write_entry(logger, &entry);
            nimcp_atomic_fetch_add_u64(&logger->async_writes, 1, NIMCP_MEMORY_ORDER_RELAXED);
            wrote = true;
        }

        if (wrote) {
            nimcp_atomic_fetch_add_u64(&logger->flush_operations, 1, NIMCP_MEMORY_ORDER_RELAXED);
        }

        // Wait for more entries or timeout
        nimcp_platform_mutex_lock(&logger->async_mutex);
        if (nimcp_atomic_load_bool(&logger->async_running, NIMCP_MEMORY_ORDER_ACQUIRE)) {
            nimcp_platform_cond_timedwait(&logger->async_cond, &logger->async_mutex,
                                          logger->flush_interval_ms);
        }
        nimcp_platform_mutex_unlock(&logger->async_mutex);
    }

    // Flush remaining entries on shutdown
    while (ring_buffer_pop(&logger->ring_buffer, &entry)) {
        write_entry(logger, &entry);
    }

    NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "async_writer_thread: operation failed");
    return NULL;
}

//=============================================================================
// Core Logging Functions
//=============================================================================

/**
 * WHAT: Check if message should be logged (level + module filter)
 */
static bool should_log(nimcp_logger_t logger, log_level_t level, const char* module) {
    // Check OFF level (never log)
    if (level >= LOG_LEVEL_OFF) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "should_log: capacity exceeded");
        return false;
    }

    // Check module-specific filter first (takes precedence over global level)
    if (module && module[0] != '\0' && logger->module_filter_count > 0) {
        nimcp_platform_mutex_lock(&logger->filter_mutex);
        for (size_t i = 0; i < logger->module_filter_count; i++) {
            if (strcmp(logger->module_filters[i].module, module) == 0) {
                bool result = logger->module_filters[i].enabled &&
                              level >= logger->module_filters[i].level;
                nimcp_platform_mutex_unlock(&logger->filter_mutex);
                return result;
            }
        }
        nimcp_platform_mutex_unlock(&logger->filter_mutex);
    }

    // Check global level
    if (level < logger->level) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "should_log: validation failed");
        return false;
    }

    return true;
}

/**
 * WHAT: Process a log entry
 */
static void process_entry(nimcp_logger_t logger, nimcp_log_entry_t* entry) {
    // Apply custom filter
    if (logger->filter) {
        if (!logger->filter(entry->level, entry->module, entry->message, logger->filter_user_data)) {
            nimcp_atomic_fetch_add_u64(&logger->messages_filtered, 1, NIMCP_MEMORY_ORDER_RELAXED);
            return;
        }
    }

    // Apply rate limiting (bypass for ERROR and FATAL - critical messages must always be logged)
    if (entry->level < LOG_LEVEL_ERROR && !rate_limiter_try_acquire(&logger->rate_limiter)) {
        nimcp_atomic_fetch_add_u64(&logger->rate_limit_hits, 1, NIMCP_MEMORY_ORDER_RELAXED);
        nimcp_atomic_fetch_add_u64(&logger->messages_dropped, 1, NIMCP_MEMORY_ORDER_RELAXED);
        return;
    }

    // Add sequence number
    entry->sequence = nimcp_atomic_fetch_add_u32(&logger->sequence, 1, NIMCP_MEMORY_ORDER_RELAXED);

    // Update statistics
    nimcp_atomic_fetch_add_u64(&logger->messages_logged, 1, NIMCP_MEMORY_ORDER_RELAXED);
    if (entry->level < LOG_LEVEL_COUNT) {
        nimcp_atomic_fetch_add_u64(&logger->level_counts[entry->level], 1, NIMCP_MEMORY_ORDER_RELAXED);
    }

    // Route to async or sync path
    bool use_async = (logger->async_mode == NIMCP_LOG_ASYNC_ON) ||
                     (logger->async_mode == NIMCP_LOG_ASYNC_HYBRID &&
                      entry->level < LOG_LEVEL_ERROR);

    if (use_async && logger->ring_buffer.entries) {
        if (!ring_buffer_push(&logger->ring_buffer, entry)) {
            // Buffer full, fall back to sync
            nimcp_atomic_fetch_add_u64(&logger->buffer_overflows, 1, NIMCP_MEMORY_ORDER_RELAXED);
            write_entry(logger, entry);
            nimcp_atomic_fetch_add_u64(&logger->sync_writes, 1, NIMCP_MEMORY_ORDER_RELAXED);
        } else {
            // Signal writer thread
            nimcp_platform_mutex_lock(&logger->async_mutex);
            nimcp_platform_cond_signal(&logger->async_cond);
            nimcp_platform_mutex_unlock(&logger->async_mutex);
        }
    } else {
        write_entry(logger, entry);
        nimcp_atomic_fetch_add_u64(&logger->sync_writes, 1, NIMCP_MEMORY_ORDER_RELAXED);
    }

    // Record security interaction
    if (logger->security_ctx && logger->security_module_id != 0) {
        nimcp_sec_record_interaction(logger->security_ctx, logger->security_module_id, true, 0.1);
    }
}

//=============================================================================
// Public API Implementation
//=============================================================================

nimcp_log_config_t nimcp_log_default_config(void) {
    nimcp_log_config_t config = {
        .level = LOG_LEVEL_INFO,
        .destinations = NIMCP_LOG_DEST_FILE | NIMCP_LOG_DEST_CONSOLE,
        .format = NIMCP_LOG_FORMAT_TEXT,
        .file_path = NULL,
        .append_mode = true,
        .async_mode = NIMCP_LOG_ASYNC_ON,
        .buffer_size = NIMCP_LOG_DEFAULT_BUFFER_SIZE,
        .flush_interval_ms = NIMCP_LOG_DEFAULT_FLUSH_INTERVAL_MS,
        .rotation = {
            .mode = NIMCP_LOG_ROTATE_SIZE,
            .max_file_size = NIMCP_LOG_DEFAULT_MAX_FILE_SIZE,
            .interval = NIMCP_LOG_ROTATE_DAILY,
            .max_rotated_files = NIMCP_LOG_DEFAULT_MAX_ROTATED_FILES,
            .compress_rotated = false
        },
        .rate_limit = {
            .enabled = true,
            .max_per_second = NIMCP_LOG_DEFAULT_RATE_LIMIT,
            .burst_size = NIMCP_LOG_DEFAULT_RATE_LIMIT * 2,
            .per_module = false
        },
        .color_mode = NIMCP_LOG_COLOR_AUTO,
        .callback = NULL,
        .callback_user_data = NULL,
        .filter = NULL,
        .filter_user_data = NULL,
        .include_source_location = true,
        .memory_manager = NULL,
        .security_context = NULL
    };
    return config;
}

nimcp_logger_t nimcp_log_create(const nimcp_log_config_t* config) {
    nimcp_log_config_t cfg = config ? *config : nimcp_log_default_config();

    // Allocate logger (can't use unified memory yet as it's not initialized)
    nimcp_logger_t logger = (nimcp_logger_t)nimcp_malloc(sizeof(struct nimcp_logger_struct));
    if (!logger) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "logger is NULL");

        return NULL;
    }

    memset(logger, 0, sizeof(struct nimcp_logger_struct));
    logger->magic = NIMCP_LOG_MAGIC;

    // Store configuration
    logger->level = cfg.level;
    logger->destinations = cfg.destinations;
    logger->format = cfg.format;
    logger->async_mode = cfg.async_mode;
    logger->color_mode = cfg.color_mode;
    logger->include_source_location = cfg.include_source_location;
    logger->flush_interval_ms = cfg.flush_interval_ms > 0 ? cfg.flush_interval_ms : NIMCP_LOG_DEFAULT_FLUSH_INTERVAL_MS;

    // Copy rotation config
    logger->rotation = cfg.rotation;
    logger->last_rotation_time = time(NULL);

    // Initialize rate limiter
    rate_limiter_init(&logger->rate_limiter, &cfg.rate_limit);

    // Initialize mutexes
    if (nimcp_platform_mutex_init(&logger->file_mutex, false) != 0 ||
        nimcp_platform_mutex_init(&logger->filter_mutex, false) != 0 ||
        nimcp_platform_mutex_init(&logger->async_mutex, false) != 0 ||
        nimcp_platform_mutex_init(&logger->context_mutex, false) != 0) {
        nimcp_free(logger);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NOT_INITIALIZED, "nimcp_log_create: validation failed");
        return NULL;
    }

    if (nimcp_platform_cond_init(&logger->async_cond) != 0) {
        nimcp_platform_mutex_destroy(&logger->file_mutex);
        nimcp_platform_mutex_destroy(&logger->filter_mutex);
        nimcp_platform_mutex_destroy(&logger->async_mutex);
        nimcp_platform_mutex_destroy(&logger->context_mutex);
        nimcp_free(logger);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NOT_INITIALIZED, "nimcp_log_create: validation failed");
        return NULL;
    }

    // Store integrations
    logger->memory_mgr = (unified_mem_manager_t)cfg.memory_manager;
    logger->security_ctx = (nimcp_sec_integration_t*)cfg.security_context;

    // Callbacks
    logger->callback = cfg.callback;
    logger->callback_user_data = cfg.callback_user_data;
    logger->filter = cfg.filter;
    logger->filter_user_data = cfg.filter_user_data;

    // Initialize atomic counters
    nimcp_atomic_init_u64(&logger->messages_logged, 0);
    nimcp_atomic_init_u64(&logger->messages_dropped, 0);
    nimcp_atomic_init_u64(&logger->messages_filtered, 0);
    for (int i = 0; i < LOG_LEVEL_COUNT; i++) {
        nimcp_atomic_init_u64(&logger->level_counts[i], 0);
    }
    nimcp_atomic_init_u64(&logger->async_writes, 0);
    nimcp_atomic_init_u64(&logger->sync_writes, 0);
    nimcp_atomic_init_u64(&logger->buffer_overflows, 0);
    nimcp_atomic_init_u64(&logger->flush_operations, 0);
    nimcp_atomic_init_u64(&logger->rotations_performed, 0);
    nimcp_atomic_init_u64(&logger->bytes_written, 0);
    nimcp_atomic_init_u64(&logger->rate_limit_hits, 0);
    nimcp_atomic_init_u64(&logger->total_log_time_ns, 0);
    nimcp_atomic_init_u64(&logger->max_log_time_ns, 0);
    nimcp_atomic_init_u32(&logger->sequence, 0);
    nimcp_atomic_init_bool(&logger->async_running, false);

    logger->start_time_ns = get_time_ns();

    // Open log file if file destination enabled
    if (logger->destinations & NIMCP_LOG_DEST_FILE) {
        const char* path = cfg.file_path ? cfg.file_path : DEFAULT_LOG_PATH;
        snprintf(logger->file_path, sizeof(logger->file_path), "%s", path);

        // Create directory
        char dir[PATH_MAX];
        get_directory(path, dir, sizeof(dir));
        mkdir_p(dir);

        logger->file = fopen(path, cfg.append_mode ? "a" : "w");
        if (!logger->file) {
            // Fallback to /tmp
            snprintf(logger->file_path, sizeof(logger->file_path), "/tmp/nimcp.log");
            logger->file = fopen(logger->file_path, cfg.append_mode ? "a" : "w");
        }

        if (logger->file) {
            // Get current file size
            fseek(logger->file, 0, SEEK_END);
            logger->current_file_size = (size_t)ftell(logger->file);
        }
    }

    // Initialize async components
    if (logger->async_mode != NIMCP_LOG_ASYNC_OFF) {
        size_t buffer_size = cfg.buffer_size > 0 ? cfg.buffer_size : NIMCP_LOG_DEFAULT_BUFFER_SIZE;
        if (!ring_buffer_init(&logger->ring_buffer, buffer_size, logger)) {
            // Fall back to sync mode
            logger->async_mode = NIMCP_LOG_ASYNC_OFF;
        } else {
            // Start writer thread
            nimcp_atomic_store_bool(&logger->async_running, true, NIMCP_MEMORY_ORDER_RELEASE);
            if (nimcp_platform_thread_create(&logger->writer_thread, async_writer_thread, logger) != 0) {
                ring_buffer_destroy(&logger->ring_buffer);
                logger->async_mode = NIMCP_LOG_ASYNC_OFF;
                nimcp_atomic_store_bool(&logger->async_running, false, NIMCP_MEMORY_ORDER_RELEASE);
            }
        }
    }

    // Register with security module
    if (logger->security_ctx) {
        nimcp_sec_register_module(logger->security_ctx, LOG_MODULE_NAME,
                                  NIMCP_SEC_CAT_UTILITY, &logger->security_module_id);
    }

    return logger;
}

void nimcp_log_destroy(nimcp_logger_t logger) {
    if (!logger || logger->magic != NIMCP_LOG_MAGIC) {
        return;
    }

    // Stop async writer
    if (nimcp_atomic_load_bool(&logger->async_running, NIMCP_MEMORY_ORDER_ACQUIRE)) {
        nimcp_atomic_store_bool(&logger->async_running, false, NIMCP_MEMORY_ORDER_RELEASE);

        // Wake up writer thread
        nimcp_platform_mutex_lock(&logger->async_mutex);
        nimcp_platform_cond_signal(&logger->async_cond);
        nimcp_platform_mutex_unlock(&logger->async_mutex);

        // Wait for thread to exit
        nimcp_platform_thread_join(logger->writer_thread, NULL);
    }

    // Destroy ring buffer
    ring_buffer_destroy(&logger->ring_buffer);

    // Unregister from security
    if (logger->security_ctx && logger->security_module_id != 0) {
        nimcp_sec_unregister_module(logger->security_ctx, logger->security_module_id);
    }

    // Close syslog
    if (logger->syslog_opened) {
#ifdef __linux__
        closelog();
#endif
    }

    // Close file
    if (logger->file) {
        fflush(logger->file);
        fclose(logger->file);
    }

    // Destroy mutexes and condition
    nimcp_platform_cond_destroy(&logger->async_cond);
    nimcp_platform_mutex_destroy(&logger->async_mutex);
    nimcp_platform_mutex_destroy(&logger->filter_mutex);
    nimcp_platform_mutex_destroy(&logger->file_mutex);
    nimcp_platform_mutex_destroy(&logger->context_mutex);

    // Invalidate and free
    logger->magic = 0;
    nimcp_free(logger);
}

void nimcp_log_flush(nimcp_logger_t logger) {
    if (!logger) logger = g_global_logger;
    if (!logger || logger->magic != NIMCP_LOG_MAGIC) return;

    // Signal async writer to flush
    if (nimcp_atomic_load_bool(&logger->async_running, NIMCP_MEMORY_ORDER_ACQUIRE)) {
        nimcp_platform_mutex_lock(&logger->async_mutex);
        nimcp_platform_cond_signal(&logger->async_cond);
        nimcp_platform_mutex_unlock(&logger->async_mutex);

        // Wait for buffer to drain with proper sleep
        int max_wait_cycles = 1000;  // Max 1 second wait
        while (ring_buffer_size(&logger->ring_buffer) > 0 && max_wait_cycles > 0) {
            struct timespec ts = {0, 1000000};  // 1ms
            nanosleep(&ts, NULL);
            max_wait_cycles--;
        }

        // Give async writer time to finish writing current batch
        struct timespec ts = {0, 5000000};  // 5ms
        nanosleep(&ts, NULL);
    }

    // Flush file
    if (logger->file) {
        nimcp_platform_mutex_lock(&logger->file_mutex);
        fflush(logger->file);
        nimcp_platform_mutex_unlock(&logger->file_mutex);
    }

    nimcp_atomic_fetch_add_u64(&logger->flush_operations, 1, NIMCP_MEMORY_ORDER_RELAXED);
}

bool nimcp_log_is_initialized(nimcp_logger_t logger) {
    if (!logger) logger = g_global_logger;
    return logger && logger->magic == NIMCP_LOG_MAGIC;
}

//=============================================================================
// Global Logger Functions
//=============================================================================

int nimcp_log_init(const nimcp_log_config_t* config) {
    nimcp_platform_once(&g_global_once, init_global_mutex);
    nimcp_platform_mutex_lock(&g_global_mutex);

    if (g_global_initialized && g_global_logger) {
        nimcp_platform_mutex_unlock(&g_global_mutex);
        return 0;  // Already initialized
    }

    g_global_logger = nimcp_log_create(config);
    g_global_initialized = (g_global_logger != NULL);

    nimcp_platform_mutex_unlock(&g_global_mutex);
    return g_global_initialized ? 0 : -1;
}

void nimcp_log_shutdown(void) {
    nimcp_platform_once(&g_global_once, init_global_mutex);
    nimcp_platform_mutex_lock(&g_global_mutex);

    if (g_global_logger) {
        nimcp_log_destroy(g_global_logger);
        g_global_logger = NULL;
    }
    g_global_initialized = false;

    nimcp_platform_mutex_unlock(&g_global_mutex);
}

nimcp_logger_t nimcp_log_get_global(void) {
    return g_global_logger;
}

void nimcp_log_set_global(nimcp_logger_t logger) {
    nimcp_platform_once(&g_global_once, init_global_mutex);
    nimcp_platform_mutex_lock(&g_global_mutex);
    g_global_logger = logger;
    g_global_initialized = (logger != NULL);
    nimcp_platform_mutex_unlock(&g_global_mutex);
}

//=============================================================================
// Legacy API Implementation
//=============================================================================

void log_init(const char* log_file) {
    nimcp_log_config_t config = nimcp_log_default_config();
    config.file_path = log_file;
    config.async_mode = NIMCP_LOG_ASYNC_OFF;  // Legacy was sync
    config.destinations = NIMCP_LOG_DEST_FILE;
    nimcp_log_init(&config);
}

void log_message(log_level_t level, const char* format, ...) {
    va_list args;
    va_start(args, format);
    nimcp_log_writev(NULL, level, NULL, NULL, 0, format, args);
    va_end(args);
}

void log_close(void) {
    nimcp_log_shutdown();
}

void nimcp_log(log_level_t level, const char* format, ...) {
    va_list args;
    va_start(args, format);
    nimcp_log_writev(NULL, level, NULL, NULL, 0, format, args);
    va_end(args);
}

//=============================================================================
// Core Logging API Implementation
//=============================================================================

void nimcp_log_write(
    nimcp_logger_t logger,
    log_level_t level,
    const char* module,
    const char* file,
    int line,
    const char* format,
    ...
) {
    va_list args;
    va_start(args, format);
    nimcp_log_writev(logger, level, module, file, line, format, args);
    va_end(args);
}

void nimcp_log_writev(
    nimcp_logger_t logger,
    log_level_t level,
    const char* module,
    const char* file,
    int line,
    const char* format,
    va_list args
) {
    if (!logger) logger = g_global_logger;
    if (!logger || logger->magic != NIMCP_LOG_MAGIC) {
        // Fallback to stderr if logger not initialized
        if (level >= LOG_LEVEL_WARN) {
            vfprintf(stderr, format, args);
            fprintf(stderr, "\n");
        }
        return;
    }

    // Check if should log
    if (!should_log(logger, level, module)) {
        return;
    }

    uint64_t start_time = get_time_ns();

    // Build log entry
    nimcp_log_entry_t entry;
    memset(&entry, 0, sizeof(entry));

    entry.level = level;
    entry.timestamp_ns = get_wall_time_ns();
    entry.thread_id = get_thread_id();
    entry.line = line;

    if (module) {
        snprintf(entry.module, sizeof(entry.module), "%s", module);
    }

    if (file) {
        snprintf(entry.file, sizeof(entry.file), "%s", file);
    }

    // Get context ID
    nimcp_platform_mutex_lock(&logger->context_mutex);
    if (logger->current_context[0] != '\0') {
        snprintf(entry.context_id, sizeof(entry.context_id), "%s", logger->current_context);
    }
    nimcp_platform_mutex_unlock(&logger->context_mutex);

    // Format message
    vsnprintf(entry.message, sizeof(entry.message), format, args);

    // Process entry
    process_entry(logger, &entry);

    // Update timing stats
    uint64_t elapsed = get_time_ns() - start_time;
    nimcp_atomic_fetch_add_u64(&logger->total_log_time_ns, elapsed, NIMCP_MEMORY_ORDER_RELAXED);

    uint64_t current_max = nimcp_atomic_load_u64(&logger->max_log_time_ns, NIMCP_MEMORY_ORDER_RELAXED);
    while (elapsed > current_max) {
        if (nimcp_atomic_compare_exchange_u64(&logger->max_log_time_ns, &current_max, elapsed,
                                               NIMCP_MEMORY_ORDER_ACQ_REL)) {
            break;
        }
    }
}

void nimcp_log_entry(nimcp_logger_t logger, const nimcp_log_entry_t* entry) {
    if (!logger) logger = g_global_logger;
    if (!logger || logger->magic != NIMCP_LOG_MAGIC || !entry) {
        return;
    }

    nimcp_log_entry_t entry_copy = *entry;
    process_entry(logger, &entry_copy);
}

//=============================================================================
// Configuration API Implementation
//=============================================================================

void nimcp_log_set_level(nimcp_logger_t logger, log_level_t level) {
    if (!logger) logger = g_global_logger;
    if (!logger || logger->magic != NIMCP_LOG_MAGIC) return;
    logger->level = level;
}

log_level_t nimcp_log_get_level(nimcp_logger_t logger) {
    if (!logger) logger = g_global_logger;
    if (!logger || logger->magic != NIMCP_LOG_MAGIC) return LOG_LEVEL_INFO;
    return logger->level;
}

bool nimcp_log_is_level_enabled(nimcp_logger_t logger, log_level_t level) {
    if (!logger) logger = g_global_logger;
    if (!logger || logger->magic != NIMCP_LOG_MAGIC) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "nimcp_log_is_level_enabled: logger is NULL");
        return false;
    }
    return level >= logger->level && level < LOG_LEVEL_OFF;
}

void nimcp_log_set_destinations(nimcp_logger_t logger, uint32_t destinations) {
    if (!logger) logger = g_global_logger;
    if (!logger || logger->magic != NIMCP_LOG_MAGIC) return;
    logger->destinations = destinations;
}

void nimcp_log_set_format(nimcp_logger_t logger, nimcp_log_format_t format) {
    if (!logger) logger = g_global_logger;
    if (!logger || logger->magic != NIMCP_LOG_MAGIC) return;
    logger->format = format;
}

void nimcp_log_set_color_mode(nimcp_logger_t logger, nimcp_log_color_mode_t mode) {
    if (!logger) logger = g_global_logger;
    if (!logger || logger->magic != NIMCP_LOG_MAGIC) return;
    logger->color_mode = mode;
}

void nimcp_log_set_callback(
    nimcp_logger_t logger,
    nimcp_log_output_callback_t callback,
    void* user_data
) {
    if (!logger) logger = g_global_logger;
    if (!logger || logger->magic != NIMCP_LOG_MAGIC) return;
    logger->callback = callback;
    logger->callback_user_data = user_data;
}

void nimcp_log_set_filter(
    nimcp_logger_t logger,
    nimcp_log_filter_t filter,
    void* user_data
) {
    if (!logger) logger = g_global_logger;
    if (!logger || logger->magic != NIMCP_LOG_MAGIC) return;
    logger->filter = filter;
    logger->filter_user_data = user_data;
}

//=============================================================================
// Module Filtering API Implementation
//=============================================================================

void nimcp_log_enable_module(nimcp_logger_t logger, const char* module, log_level_t level) {
    if (!logger) logger = g_global_logger;
    if (!logger || logger->magic != NIMCP_LOG_MAGIC || !module) return;

    nimcp_platform_mutex_lock(&logger->filter_mutex);

    // Check if already exists
    for (size_t i = 0; i < logger->module_filter_count; i++) {
        if (strcmp(logger->module_filters[i].module, module) == 0) {
            logger->module_filters[i].level = level;
            logger->module_filters[i].enabled = true;
            nimcp_platform_mutex_unlock(&logger->filter_mutex);
            return;
        }
    }

    // Add new filter
    if (logger->module_filter_count < MAX_MODULE_FILTERS) {
        snprintf(logger->module_filters[logger->module_filter_count].module,
                 sizeof(logger->module_filters[0].module), "%s", module);
        logger->module_filters[logger->module_filter_count].level = level;
        logger->module_filters[logger->module_filter_count].enabled = true;
        logger->module_filter_count++;
    }

    nimcp_platform_mutex_unlock(&logger->filter_mutex);
}

void nimcp_log_disable_module(nimcp_logger_t logger, const char* module) {
    if (!logger) logger = g_global_logger;
    if (!logger || logger->magic != NIMCP_LOG_MAGIC || !module) return;

    nimcp_platform_mutex_lock(&logger->filter_mutex);

    for (size_t i = 0; i < logger->module_filter_count; i++) {
        if (strcmp(logger->module_filters[i].module, module) == 0) {
            logger->module_filters[i].enabled = false;
            break;
        }
    }

    nimcp_platform_mutex_unlock(&logger->filter_mutex);
}

void nimcp_log_clear_module_filters(nimcp_logger_t logger) {
    if (!logger) logger = g_global_logger;
    if (!logger || logger->magic != NIMCP_LOG_MAGIC) return;

    nimcp_platform_mutex_lock(&logger->filter_mutex);
    logger->module_filter_count = 0;
    nimcp_platform_mutex_unlock(&logger->filter_mutex);
}

//=============================================================================
// Context API Implementation
//=============================================================================

nimcp_log_context_t nimcp_log_context_create(nimcp_logger_t logger, const char* context_id) {
    if (!logger) logger = g_global_logger;
    if (!logger || logger->magic != NIMCP_LOG_MAGIC) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "nimcp_log_context_create: logger is NULL");
        return NULL;
    }

    nimcp_log_context_t ctx = (nimcp_log_context_t)log_alloc(logger,
                                                              sizeof(struct nimcp_log_context_struct));
    if (!ctx) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "ctx is NULL");

        return NULL;

    }

    ctx->logger = logger;

    // Save previous context
    nimcp_platform_mutex_lock(&logger->context_mutex);
    snprintf(ctx->previous_context, sizeof(ctx->previous_context), "%s", logger->current_context);

    // Set new context
    if (context_id) {
        snprintf(ctx->context_id, sizeof(ctx->context_id), "%s", context_id);
    } else {
        // Auto-generate context ID
        snprintf(ctx->context_id, sizeof(ctx->context_id), "ctx-%u-%lu",
                 get_thread_id(), (unsigned long)get_time_ns() % 1000000);
    }
    snprintf(logger->current_context, sizeof(logger->current_context), "%s", ctx->context_id);
    nimcp_platform_mutex_unlock(&logger->context_mutex);

    return ctx;
}

void nimcp_log_context_destroy(nimcp_log_context_t context) {
    if (!context) return;

    nimcp_logger_t logger = context->logger;
    if (logger && logger->magic == NIMCP_LOG_MAGIC) {
        // Restore previous context
        nimcp_platform_mutex_lock(&logger->context_mutex);
        snprintf(logger->current_context, sizeof(logger->current_context),
                 "%s", context->previous_context);
        nimcp_platform_mutex_unlock(&logger->context_mutex);
    }

    log_free(context);
}

const char* nimcp_log_get_context_id(nimcp_logger_t logger) {
    if (!logger) logger = g_global_logger;
    if (!logger || logger->magic != NIMCP_LOG_MAGIC) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "nimcp_log_get_context_id: logger is NULL");
        return NULL;
    }

    return logger->current_context[0] != '\0' ? logger->current_context : NULL;
}

void nimcp_log_set_context_id(nimcp_logger_t logger, const char* context_id) {
    if (!logger) logger = g_global_logger;
    if (!logger || logger->magic != NIMCP_LOG_MAGIC) return;

    nimcp_platform_mutex_lock(&logger->context_mutex);
    if (context_id) {
        snprintf(logger->current_context, sizeof(logger->current_context), "%s", context_id);
    } else {
        logger->current_context[0] = '\0';
    }
    nimcp_platform_mutex_unlock(&logger->context_mutex);
}

//=============================================================================
// Rotation API Implementation
//=============================================================================

int nimcp_log_rotate(nimcp_logger_t logger) {
    if (!logger) logger = g_global_logger;
    if (!logger || logger->magic != NIMCP_LOG_MAGIC) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "nimcp_log_rotate: logger is NULL");
        return -1;
    }
    if (!logger->file) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "nimcp_log_rotate: logger->file is NULL");
        return -1;
    }

    nimcp_platform_mutex_lock(&logger->file_mutex);

    // Close current file
    fflush(logger->file);
    fclose(logger->file);
    logger->file = NULL;

    // Rotate files (rename current to .1, .1 to .2, etc.)
    char rotated_path[PATH_MAX + 16];
    char prev_path[PATH_MAX + 16];

    // Remove oldest
    snprintf(rotated_path, sizeof(rotated_path), "%s.%u",
             logger->file_path, logger->rotation.max_rotated_files);
    unlink(rotated_path);

    // Shift existing rotated files
    for (uint32_t i = logger->rotation.max_rotated_files; i > 1; i--) {
        snprintf(prev_path, sizeof(prev_path), "%s.%u", logger->file_path, i - 1);
        snprintf(rotated_path, sizeof(rotated_path), "%s.%u", logger->file_path, i);
        rename(prev_path, rotated_path);
    }

    // Rename current to .1
    snprintf(rotated_path, sizeof(rotated_path), "%s.1", logger->file_path);
    rename(logger->file_path, rotated_path);

    // Open new file
    logger->file = fopen(logger->file_path, "w");
    logger->current_file_size = 0;
    logger->last_rotation_time = time(NULL);

    nimcp_atomic_fetch_add_u64(&logger->rotations_performed, 1, NIMCP_MEMORY_ORDER_RELAXED);

    nimcp_platform_mutex_unlock(&logger->file_mutex);

    return logger->file ? 0 : -1;
}

void nimcp_log_set_rotation(nimcp_logger_t logger, const nimcp_log_rotation_config_t* config) {
    if (!logger) logger = g_global_logger;
    if (!logger || logger->magic != NIMCP_LOG_MAGIC || !config) return;
    logger->rotation = *config;
}

const char* nimcp_log_get_file_path(nimcp_logger_t logger) {
    if (!logger) logger = g_global_logger;
    if (!logger || logger->magic != NIMCP_LOG_MAGIC) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "nimcp_log_get_file_path: logger is NULL");
        return NULL;
    }
    return logger->file_path[0] ? logger->file_path : NULL;
}

int nimcp_log_set_file_path(nimcp_logger_t logger, const char* path) {
    if (!logger) logger = g_global_logger;
    if (!logger || logger->magic != NIMCP_LOG_MAGIC || !path) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "nimcp_log_set_file_path: required parameter is NULL (logger, path)");
        return -1;
    }

    nimcp_platform_mutex_lock(&logger->file_mutex);

    // Close current file
    if (logger->file) {
        fflush(logger->file);
        fclose(logger->file);
    }

    // Update path
    snprintf(logger->file_path, sizeof(logger->file_path), "%s", path);

    // Create directory
    char dir[PATH_MAX];
    get_directory(path, dir, sizeof(dir));
    mkdir_p(dir);

    // Open new file
    logger->file = fopen(path, "a");
    if (logger->file) {
        fseek(logger->file, 0, SEEK_END);
        logger->current_file_size = (size_t)ftell(logger->file);
    }

    nimcp_platform_mutex_unlock(&logger->file_mutex);

    return logger->file ? 0 : -1;
}

//=============================================================================
// Rate Limiting API Implementation
//=============================================================================

void nimcp_log_set_rate_limit(nimcp_logger_t logger, const nimcp_log_rate_limit_config_t* config) {
    if (!logger) logger = g_global_logger;
    if (!logger || logger->magic != NIMCP_LOG_MAGIC || !config) return;
    rate_limiter_init(&logger->rate_limiter, config);
}

void nimcp_log_reset_rate_limit(nimcp_logger_t logger) {
    if (!logger) logger = g_global_logger;
    if (!logger || logger->magic != NIMCP_LOG_MAGIC) return;
    nimcp_atomic_store_u64(&logger->rate_limiter.tokens, logger->rate_limiter.max_tokens,
                           NIMCP_MEMORY_ORDER_RELEASE);
}

//=============================================================================
// Statistics API Implementation
//=============================================================================

int nimcp_log_get_stats(nimcp_logger_t logger, nimcp_log_stats_t* stats) {
    if (!logger) logger = g_global_logger;
    if (!logger || logger->magic != NIMCP_LOG_MAGIC || !stats) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "nimcp_log_get_stats: required parameter is NULL (logger, stats)");
        return -1;
    }

    memset(stats, 0, sizeof(*stats));

    stats->messages_logged = nimcp_atomic_load_u64(&logger->messages_logged, NIMCP_MEMORY_ORDER_RELAXED);
    stats->messages_dropped = nimcp_atomic_load_u64(&logger->messages_dropped, NIMCP_MEMORY_ORDER_RELAXED);
    stats->messages_filtered = nimcp_atomic_load_u64(&logger->messages_filtered, NIMCP_MEMORY_ORDER_RELAXED);

    for (int i = 0; i < LOG_LEVEL_COUNT; i++) {
        stats->level_counts[i] = nimcp_atomic_load_u64(&logger->level_counts[i], NIMCP_MEMORY_ORDER_RELAXED);
    }

    stats->async_writes = nimcp_atomic_load_u64(&logger->async_writes, NIMCP_MEMORY_ORDER_RELAXED);
    stats->sync_writes = nimcp_atomic_load_u64(&logger->sync_writes, NIMCP_MEMORY_ORDER_RELAXED);
    stats->buffer_overflows = nimcp_atomic_load_u64(&logger->buffer_overflows, NIMCP_MEMORY_ORDER_RELAXED);
    stats->flush_operations = nimcp_atomic_load_u64(&logger->flush_operations, NIMCP_MEMORY_ORDER_RELAXED);

    stats->rotations_performed = nimcp_atomic_load_u64(&logger->rotations_performed, NIMCP_MEMORY_ORDER_RELAXED);
    stats->bytes_written = nimcp_atomic_load_u64(&logger->bytes_written, NIMCP_MEMORY_ORDER_RELAXED);
    stats->current_file_size = logger->current_file_size;

    stats->rate_limit_hits = nimcp_atomic_load_u64(&logger->rate_limit_hits, NIMCP_MEMORY_ORDER_RELAXED);

    stats->total_log_time_ns = nimcp_atomic_load_u64(&logger->total_log_time_ns, NIMCP_MEMORY_ORDER_RELAXED);
    stats->max_log_time_ns = nimcp_atomic_load_u64(&logger->max_log_time_ns, NIMCP_MEMORY_ORDER_RELAXED);
    if (stats->messages_logged > 0) {
        stats->avg_log_time_ns = stats->total_log_time_ns / stats->messages_logged;
    }

    // Buffer utilization
    if (logger->ring_buffer.entries) {
        size_t used = ring_buffer_size(&logger->ring_buffer);
        stats->buffer_utilization = (used * 100) / logger->ring_buffer.capacity;
        stats->memory_used = logger->ring_buffer.capacity * sizeof(nimcp_log_entry_t);
    }

    stats->uptime_ns = get_time_ns() - logger->start_time_ns;

    return 0;
}

void nimcp_log_reset_stats(nimcp_logger_t logger) {
    if (!logger) logger = g_global_logger;
    if (!logger || logger->magic != NIMCP_LOG_MAGIC) return;

    nimcp_atomic_store_u64(&logger->messages_logged, 0, NIMCP_MEMORY_ORDER_RELAXED);
    nimcp_atomic_store_u64(&logger->messages_dropped, 0, NIMCP_MEMORY_ORDER_RELAXED);
    nimcp_atomic_store_u64(&logger->messages_filtered, 0, NIMCP_MEMORY_ORDER_RELAXED);

    for (int i = 0; i < LOG_LEVEL_COUNT; i++) {
        nimcp_atomic_store_u64(&logger->level_counts[i], 0, NIMCP_MEMORY_ORDER_RELAXED);
    }

    nimcp_atomic_store_u64(&logger->async_writes, 0, NIMCP_MEMORY_ORDER_RELAXED);
    nimcp_atomic_store_u64(&logger->sync_writes, 0, NIMCP_MEMORY_ORDER_RELAXED);
    nimcp_atomic_store_u64(&logger->buffer_overflows, 0, NIMCP_MEMORY_ORDER_RELAXED);
    nimcp_atomic_store_u64(&logger->flush_operations, 0, NIMCP_MEMORY_ORDER_RELAXED);
    nimcp_atomic_store_u64(&logger->rate_limit_hits, 0, NIMCP_MEMORY_ORDER_RELAXED);
    nimcp_atomic_store_u64(&logger->total_log_time_ns, 0, NIMCP_MEMORY_ORDER_RELAXED);
    nimcp_atomic_store_u64(&logger->max_log_time_ns, 0, NIMCP_MEMORY_ORDER_RELAXED);

    logger->start_time_ns = get_time_ns();
}

uint32_t nimcp_log_get_security_id(nimcp_logger_t logger) {
    if (!logger) logger = g_global_logger;
    if (!logger || logger->magic != NIMCP_LOG_MAGIC) return 0;
    return logger->security_module_id;
}

//=============================================================================
// Utility Functions Implementation
//=============================================================================

const char* nimcp_log_level_name(log_level_t level) {
    if (level >= 0 && level < LOG_LEVEL_COUNT) {
        return LEVEL_NAMES[level];
    }
    return "UNKNOWN";
}

log_level_t nimcp_log_level_from_string(const char* name) {
    if (!name) return LOG_LEVEL_INFO;

    // Case-insensitive comparison
    for (int i = 0; i < LOG_LEVEL_COUNT; i++) {
        if (strcasecmp(name, LEVEL_NAMES[i]) == 0) {
            return (log_level_t)i;
        }
    }

    // Handle common aliases
    if (strcasecmp(name, "WARNING") == 0) {
        return LOG_LEVEL_WARN;
    }

    return LOG_LEVEL_INFO;
}

const char* nimcp_log_format_name(nimcp_log_format_t format) {
    switch (format) {
        case NIMCP_LOG_FORMAT_TEXT: return "text";
        case NIMCP_LOG_FORMAT_JSON: return "json";
        case NIMCP_LOG_FORMAT_COMPACT: return "compact";
        case NIMCP_LOG_FORMAT_SYSLOG: return "syslog";
        default: return "unknown";
    }
}

bool nimcp_log_is_tty(void) {
    return isatty(STDERR_FILENO) != 0;
}
