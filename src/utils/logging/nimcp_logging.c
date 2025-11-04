#include "utils/logging/nimcp_logging.h"
#include <errno.h>
#include <libgen.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static FILE* log_file = NULL;
static const char* DEFAULT_LOG_FILE = "/var/log/nimcp/nimcp.log";

/**
 * WHAT: Create directory and all parent directories
 * WHY:  mkdir() only creates one level, we need recursive creation
 * HOW:  Similar to `mkdir -p` - creates each directory in the path
 */
static int mkdir_p(const char* path)
{
    char tmp[PATH_MAX];
    char* p = NULL;
    size_t len;

    snprintf(tmp, sizeof(tmp), "%s", path);
    len = strlen(tmp);
    if (tmp[len - 1] == '/')
        tmp[len - 1] = 0;

    for (p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = 0;
            if (mkdir(tmp, 0755) != 0) {
                if (errno != EEXIST) {
                    return -1;
                }
            }
            *p = '/';
        }
    }

    if (mkdir(tmp, 0755) != 0) {
        if (errno != EEXIST) {
            return -1;
        }
    }

    return 0;
}

void log_init(const char* log_file_path)
{
    /* WHAT: Determine which log file path to use
     * WHY:  NULL means use default /var/log/nimcp, otherwise use provided path
     */
    const char* target_path = log_file_path ? log_file_path : DEFAULT_LOG_FILE;

    /* WHAT: Extract directory from file path and create it
     * WHY:  Parent directory must exist before we can create the log file
     * NOTE: dirname() may modify input, so use a copy
     */
    char path_copy[PATH_MAX];
    snprintf(path_copy, sizeof(path_copy), "%s", target_path);
    char* dir_path = dirname(path_copy);

    /* WHAT: Create directory (and parents) if it doesn't exist
     * WHY:  Log file creation will fail if directory doesn't exist
     */
    struct stat st;
    if (stat(dir_path, &st) != 0) {
        if (mkdir_p(dir_path) != 0) {
            perror("Failed to create log directory");
            exit(EXIT_FAILURE);
        }
    }

    /* WHAT: Open log file for appending
     * WHY:  We want to append to existing logs, not overwrite them
     */
    if (log_file) {
        fclose(log_file);  // Close any existing log file
    }

    log_file = fopen(target_path, "a");
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
