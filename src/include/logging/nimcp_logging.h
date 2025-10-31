
#ifndef LOGGING_H
#define LOGGING_H

#include <stdio.h>
#include <time.h>

#define PATH_MAX 256
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

// Convenience macros for logging
// Note: Using NIMCP_LOGGING_ prefix to avoid conflict with log level enums in nimcp_common.h
#define NIMCP_LOGGING_DEBUG(...) log_message(LOG_LEVEL_DEBUG, __VA_ARGS__)
#define NIMCP_LOGGING_INFO(...) log_message(LOG_LEVEL_INFO, __VA_ARGS__)
#define NIMCP_LOGGING_WARN(...) log_message(LOG_LEVEL_WARN, __VA_ARGS__)
#define NIMCP_LOGGING_ERROR(...) log_message(LOG_LEVEL_ERROR, __VA_ARGS__)
#define NIMCP_LOGGING_FATAL(...) log_message(LOG_LEVEL_FATAL, __VA_ARGS__)

#endif // LOGGING_H
