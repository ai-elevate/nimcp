#include "logging/nimcp_logging.h"
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static FILE* log_file = NULL;
static const char* DEFAULT_LOG_DIR = "/var/log/nimcp";
static const char* LOG_FILE_NAME = "nimcp.log";

void log_init(const char* log_file_path)
{
    // Create the log directory if it does not exist
    struct stat st;
    if (stat(DEFAULT_LOG_DIR, &st) != 0) {
        // Directory does not exist, create it
        if (mkdir(DEFAULT_LOG_DIR, 0755) != 0) {
            perror("Failed to create log directory");
            exit(EXIT_FAILURE);
        }
    }

    // Construct the full log file path
    char full_log_path[PATH_MAX];
    snprintf(full_log_path, sizeof(full_log_path), "%s/%s", DEFAULT_LOG_DIR, LOG_FILE_NAME);

    if (log_file) {
        fclose(log_file);  // Close any existing log file
    }
    log_file = fopen(full_log_path, "a");
    if (!log_file) {
        perror("Failed to open log file");
        exit(EXIT_FAILURE);
    }
}

void log_message(log_level_t level, const char* format, ...)
{
    if (!log_file) {
        return;  // Log file not initialized
    }

    const char* level_strings[] = {"DEBUG", "INFO", "WARN", "ERROR", "FATAL"};
    time_t now = time(NULL);
    struct tm* tm_info = localtime(&now);

    char time_buffer[26];
    strftime(time_buffer, 26, "%Y-%m-%d %H:%M:%S", tm_info);

    fprintf(log_file, "[%s] [%s] ", time_buffer, level_strings[level]);

    va_list args;
    va_start(args, format);
    vfprintf(log_file, format, args);
    va_end(args);

    fprintf(log_file, "\n");
    fflush(log_file);  // Ensure the message is written immediately
}

void log_close()
{
    if (log_file) {
        fclose(log_file);
        log_file = NULL;
    }
}
