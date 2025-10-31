
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

#endif // LOGGING_H
