
#ifndef LOGGING_H
#define LOGGING_H

#include <stdio.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef PATH_MAX
#define PATH_MAX 256  // Fallback if system doesn't define PATH_MAX
#endif
typedef enum {
    LOG_LEVEL_DEBUG,
    LOG_LEVEL_INFO,
    LOG_LEVEL_WARN,
    LOG_LEVEL_ERROR,
    LOG_LEVEL_FATAL
} log_level_t;

// Function to initialize the logging system
void log_init(const char* log_file);

// Function to log messages
void log_message(log_level_t level, const char* format, ...);

// Function to close the logging system
void log_close();

// Legacy wrapper function (for backward compatibility)
void nimcp_log(log_level_t level, const char* format, ...);

// Convenience macros for logging
// Note: Using NIMCP_LOGGING_ prefix to avoid conflict with log level enums in nimcp_common.h
#define NIMCP_LOGGING_DEBUG(...) log_message(LOG_LEVEL_DEBUG, __VA_ARGS__)
#define NIMCP_LOGGING_INFO(...) log_message(LOG_LEVEL_INFO, __VA_ARGS__)
#define NIMCP_LOGGING_WARN(...) log_message(LOG_LEVEL_WARN, __VA_ARGS__)
#define NIMCP_LOGGING_ERROR(...) log_message(LOG_LEVEL_ERROR, __VA_ARGS__)
#define NIMCP_LOGGING_FATAL(...) log_message(LOG_LEVEL_FATAL, __VA_ARGS__)

// Short-form macros for convenience (used by fault tolerance modules)
#define LOG_DEBUG(...) log_message(LOG_LEVEL_DEBUG, __VA_ARGS__)
#define LOG_INFO(...) log_message(LOG_LEVEL_INFO, __VA_ARGS__)
#define LOG_WARNING(...) log_message(LOG_LEVEL_WARN, __VA_ARGS__)
#define LOG_ERROR(...) log_message(LOG_LEVEL_ERROR, __VA_ARGS__)
#define LOG_FATAL(...) log_message(LOG_LEVEL_FATAL, __VA_ARGS__)

#ifdef __cplusplus
}
#endif

#endif  // LOGGING_H
