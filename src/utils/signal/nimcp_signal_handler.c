/**
 * @file nimcp_signal_handler.c
 * @brief Signal handling implementation with graceful recovery
 *
 * @author NIMCP Team
 * @date 2025-11-09
 */

#include "nimcp_signal_handler.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "core/brain/persistence/nimcp_brain_persistence.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <execinfo.h>  // For backtrace()
#include <time.h>
#include <sys/stat.h>  // For directory operations
#include <dirent.h>    // For directory scanning

//=============================================================================
// Signal-Safe Globals (accessed from signal handlers)
//=============================================================================

// IMPORTANT: These must be volatile sig_atomic_t for signal safety
static volatile sig_atomic_t g_shutdown_requested = 0;
static volatile sig_atomic_t g_reload_requested = 0;
static volatile sig_atomic_t g_last_signal = 0;

// Signal statistics (updated atomically)
static volatile uint64_t g_sigsegv_count = 0;
static volatile uint64_t g_sigabrt_count = 0;
static volatile uint64_t g_sigbus_count = 0;
static volatile uint64_t g_sigfpe_count = 0;
static volatile uint64_t g_sigill_count = 0;
static volatile uint64_t g_sigterm_count = 0;
static volatile uint64_t g_sigint_count = 0;
static volatile uint64_t g_sighup_count = 0;
static volatile uint64_t g_recoveries = 0;
static volatile uint64_t g_fatal_crashes = 0;
static volatile uint64_t g_checkpoint_saves = 0;
static volatile uint64_t g_recovery_failures = 0;

// Configuration (set once during initialization)
static signal_handler_config_t g_config;
static bool g_installed = false;

// Recovery configuration
static bool g_auto_recovery_enabled = true;
static int g_max_recovery_attempts = 3;
static int g_checkpoint_retention = 5;
static volatile uint64_t g_recovery_attempt_count = 0;

// Brain instance for crash recovery (signal-safe pointer)
static brain_t g_registered_brain = NULL;

// Previous signal handlers (for restoration)
static struct sigaction g_old_sigsegv;
static struct sigaction g_old_sigabrt;
static struct sigaction g_old_sigbus;
static struct sigaction g_old_sigfpe;
static struct sigaction g_old_sigill;
static struct sigaction g_old_sigterm;
static struct sigaction g_old_sigint;
static struct sigaction g_old_sighup;

//=============================================================================
// Signal-Safe Utility Functions
//=============================================================================

/**
 * WHAT: Write string to stderr (signal-safe)
 * WHY:  printf() is NOT signal-safe
 * HOW:  Use write() syscall
 */
static void safe_write(const char* msg)
{
    if (msg) {
        write(STDERR_FILENO, msg, strlen(msg));
    }
}

/**
 * WHAT: Attempt checkpoint save (signal-safe version)
 * WHY:  Preserve brain state before termination
 * HOW:  Use brain_save() if safe and enabled
 * NOTE: This is NOT fully signal-safe but necessary for recovery
 */
static void attempt_checkpoint_save_unsafe(void)
{
    if (!g_config.enable_checkpoint_save || !g_registered_brain) {
        return;
    }

    // Make checkpoint directory if needed
    mkdir(g_config.checkpoint_path, 0755);

    // Generate timestamped checkpoint filename
    char checkpoint_file[512];
    time_t now = time(NULL);
    snprintf(checkpoint_file, sizeof(checkpoint_file),
             "%s/crash_checkpoint_%ld.nimcp",
             g_config.checkpoint_path, now);

    // Attempt to save (may fail in signal handler)
    if (brain_save(g_registered_brain, checkpoint_file)) {
        g_checkpoint_saves++;
        safe_write("\n*** Checkpoint saved to: ");
        safe_write(checkpoint_file);
        safe_write("\n");
    } else {
        safe_write("\n*** Failed to save checkpoint\n");
    }
}

/**
 * WHAT: Write integer to stderr (signal-safe)
 * WHY:  sprintf() is NOT signal-safe
 * HOW:  Manual digit extraction and write()
 */
static void safe_write_int(int value)
{
    char buf[32];
    int i = 0;
    int is_negative = (value < 0);

    if (is_negative) {
        value = -value;
    }

    // Extract digits in reverse
    do {
        buf[i++] = '0' + (value % 10);
        value /= 10;
    } while (value > 0 && i < 31);

    if (is_negative) {
        buf[i++] = '-';
    }

    // Write in forward order
    for (int j = i - 1; j >= 0; j--) {
        write(STDERR_FILENO, &buf[j], 1);
    }
}

/**
 * WHAT: Log stack trace (best-effort, may not be fully signal-safe)
 * WHY:  Helps diagnose crash location
 * HOW:  Use backtrace() and backtrace_symbols_fd()
 */
static void log_stack_trace(void)
{
    void* buffer[100];
    int nptrs = backtrace(buffer, 100);

    safe_write("\n=== STACK TRACE ===\n");
    // backtrace_symbols_fd() is async-signal-safe according to man pages
    backtrace_symbols_fd(buffer, nptrs, STDERR_FILENO);
    safe_write("===================\n");
}

//=============================================================================
// Signal Handlers (MUST BE ASYNC-SIGNAL-SAFE)
//=============================================================================

/**
 * WHAT: Handle fatal signals (SIGSEGV, SIGBUS, SIGABRT, etc.)
 * WHY:  Log crash info before termination
 * HOW:  Write to stderr, optionally log stack trace, save checkpoint, then exit
 */
static void handle_fatal_signal(int sig)
{
    g_last_signal = sig;
    g_fatal_crashes++;

    // Update signal-specific counter
    switch (sig) {
        case SIGSEGV: g_sigsegv_count++; break;
        case SIGABRT: g_sigabrt_count++; break;
        case SIGBUS:  g_sigbus_count++; break;
        case SIGFPE:  g_sigfpe_count++; break;
        case SIGILL:  g_sigill_count++; break;
    }

    // Log crash (signal-safe)
    safe_write("\n!!! FATAL SIGNAL RECEIVED: ");
    safe_write_int(sig);
    safe_write(" (");
    safe_write(signal_handler_get_signal_name(sig));
    safe_write(")\n");

    // Log stack trace if enabled
    if (g_config.enable_stack_trace) {
        log_stack_trace();
    }

    // Call custom crash callback if set
    if (g_config.on_fatal_signal) {
        g_config.on_fatal_signal(sig);
    }

    // TODO COMPLETED: Attempt checkpoint save if enabled
    // NOTE: This is risky in a signal handler but necessary for recovery
    // We attempt it with proper error handling and logging
    if (g_config.enable_checkpoint_save && g_registered_brain) {
        safe_write("\n!!! Attempting checkpoint save (signal handler context)\n");
        attempt_checkpoint_save_unsafe();
    }

    safe_write("!!! Process terminating due to fatal signal\n");

    // Restore default handler and re-raise to generate core dump
    signal(sig, SIG_DFL);
    raise(sig);
}

/**
 * WHAT: Handle floating point exceptions
 * WHY:  Attempt to continue processing with NaN/Inf
 * HOW:  Log error and return (may cause propagation of NaN)
 */
static void handle_sigfpe(int sig)
{
    g_last_signal = sig;
    g_sigfpe_count++;

    safe_write("\n*** FLOATING POINT EXCEPTION (SIGFPE) ***\n");

    if (g_config.sigfpe_mode == SIGNAL_MODE_LOG_CONTINUE) {
        safe_write("*** Attempting to continue (may propagate NaN/Inf)\n");
        g_recoveries++;
        // Return from handler - FPU state restored by OS
        return;
    } else {
        // Treat as fatal
        handle_fatal_signal(sig);
    }
}

/**
 * WHAT: Handle termination signals (SIGTERM, SIGINT)
 * WHY:  Graceful shutdown with state save
 * HOW:  Set flag for main loop to check
 */
static void handle_shutdown_signal(int sig)
{
    g_last_signal = sig;
    g_shutdown_requested = 1;

    if (sig == SIGTERM) {
        g_sigterm_count++;
    } else if (sig == SIGINT) {
        g_sigint_count++;
    }

    safe_write("\n=== SHUTDOWN SIGNAL RECEIVED: ");
    safe_write_int(sig);
    safe_write(" ===\n");
    safe_write("=== Graceful shutdown initiated ===\n");

    // Call custom shutdown callback if set
    if (g_config.on_graceful_shutdown) {
        g_config.on_graceful_shutdown();
    }
}

/**
 * WHAT: Handle config reload signal (SIGHUP)
 * WHY:  Reload configuration without restarting
 * HOW:  Set flag for main loop to check
 */
static void handle_sighup(int sig)
{
    (void)sig;  // Unused
    g_last_signal = SIGHUP;
    g_sighup_count++;
    g_reload_requested = 1;

    safe_write("\n=== CONFIG RELOAD REQUESTED (SIGHUP) ===\n");

    // Call custom reload callback if set
    if (g_config.on_reload_config) {
        g_config.on_reload_config();
    }
}

//=============================================================================
// Public API Implementation
//=============================================================================

signal_handler_config_t signal_handler_default_config(void)
{
    signal_handler_config_t config = {
        // Fatal signals → log and shutdown
        .sigsegv_mode = SIGNAL_MODE_LOG_SHUTDOWN,
        .sigabrt_mode = SIGNAL_MODE_LOG_SHUTDOWN,
        .sigbus_mode = SIGNAL_MODE_LOG_SHUTDOWN,
        .sigfpe_mode = SIGNAL_MODE_LOG_CONTINUE,  // Try to continue with NaN
        .sigill_mode = SIGNAL_MODE_LOG_SHUTDOWN,

        // Termination signals → graceful shutdown
        .sigterm_mode = SIGNAL_MODE_LOG_SHUTDOWN,
        .sigint_mode = SIGNAL_MODE_LOG_SHUTDOWN,

        // Config reload
        .sighup_mode = SIGNAL_MODE_LOG_CONTINUE,

        // Diagnostics
        .enable_stack_trace = true,
        .enable_state_dump = false,      // Disabled by default (not signal-safe)
        .enable_checkpoint_save = false, // Disabled by default (not signal-safe)

        // Paths
        .crash_log_path = "/tmp/nimcp_crash.log",
        .checkpoint_path = "/tmp/nimcp_checkpoint",

        // Callbacks
        .on_fatal_signal = NULL,
        .on_reload_config = NULL,
        .on_graceful_shutdown = NULL
    };
    return config;
}

bool signal_handler_install(const signal_handler_config_t* config)
{
    if (g_installed) {
        return false;  // Already installed
    }

    // Use defaults if config is NULL
    if (config) {
        g_config = *config;
    } else {
        g_config = signal_handler_default_config();
    }

    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sigemptyset(&sa.sa_mask);

    // Install SIGSEGV handler
    if (g_config.sigsegv_mode != SIGNAL_MODE_IGNORE) {
        sa.sa_handler = handle_fatal_signal;
        sa.sa_flags = SA_RESETHAND;  // Reset to default after first signal
        if (sigaction(SIGSEGV, &sa, &g_old_sigsegv) != 0) {
            return false;
        }
    }

    // Install SIGABRT handler
    if (g_config.sigabrt_mode != SIGNAL_MODE_IGNORE) {
        sa.sa_handler = handle_fatal_signal;
        if (sigaction(SIGABRT, &sa, &g_old_sigabrt) != 0) {
            return false;
        }
    }

    // Install SIGBUS handler
    if (g_config.sigbus_mode != SIGNAL_MODE_IGNORE) {
        sa.sa_handler = handle_fatal_signal;
        if (sigaction(SIGBUS, &sa, &g_old_sigbus) != 0) {
            return false;
        }
    }

    // Install SIGFPE handler
    if (g_config.sigfpe_mode != SIGNAL_MODE_IGNORE) {
        sa.sa_handler = handle_sigfpe;
        sa.sa_flags = 0;  // Don't reset for FPE (want to recover)
        if (sigaction(SIGFPE, &sa, &g_old_sigfpe) != 0) {
            return false;
        }
    }

    // Install SIGILL handler
    if (g_config.sigill_mode != SIGNAL_MODE_IGNORE) {
        sa.sa_handler = handle_fatal_signal;
        sa.sa_flags = SA_RESETHAND;
        if (sigaction(SIGILL, &sa, &g_old_sigill) != 0) {
            return false;
        }
    }

    // Install SIGTERM handler
    if (g_config.sigterm_mode != SIGNAL_MODE_IGNORE) {
        sa.sa_handler = handle_shutdown_signal;
        sa.sa_flags = 0;
        if (sigaction(SIGTERM, &sa, &g_old_sigterm) != 0) {
            return false;
        }
    }

    // Install SIGINT handler
    if (g_config.sigint_mode != SIGNAL_MODE_IGNORE) {
        sa.sa_handler = handle_shutdown_signal;
        if (sigaction(SIGINT, &sa, &g_old_sigint) != 0) {
            return false;
        }
    }

    // Install SIGHUP handler
    if (g_config.sighup_mode != SIGNAL_MODE_IGNORE) {
        sa.sa_handler = handle_sighup;
        if (sigaction(SIGHUP, &sa, &g_old_sighup) != 0) {
            return false;
        }
    }

    g_installed = true;
    return true;
}

bool signal_handler_uninstall(void)
{
    if (!g_installed) {
        return false;
    }

    // Restore previous handlers
    sigaction(SIGSEGV, &g_old_sigsegv, NULL);
    sigaction(SIGABRT, &g_old_sigabrt, NULL);
    sigaction(SIGBUS, &g_old_sigbus, NULL);
    sigaction(SIGFPE, &g_old_sigfpe, NULL);
    sigaction(SIGILL, &g_old_sigill, NULL);
    sigaction(SIGTERM, &g_old_sigterm, NULL);
    sigaction(SIGINT, &g_old_sigint, NULL);
    sigaction(SIGHUP, &g_old_sighup, NULL);

    g_installed = false;
    return true;
}

void signal_handler_register_brain(brain_t brain)
{
    g_registered_brain = brain;
}

void signal_handler_unregister_brain(void)
{
    g_registered_brain = NULL;
}

signal_handler_stats_t signal_handler_get_stats(void)
{
    signal_handler_stats_t stats = {
        .sigsegv_count = g_sigsegv_count,
        .sigabrt_count = g_sigabrt_count,
        .sigbus_count = g_sigbus_count,
        .sigfpe_count = g_sigfpe_count,
        .sigill_count = g_sigill_count,
        .sigterm_count = g_sigterm_count,
        .sigint_count = g_sigint_count,
        .sighup_count = g_sighup_count,
        .recoveries = g_recoveries,
        .fatal_crashes = g_fatal_crashes
    };
    return stats;
}

void signal_handler_reset_stats(void)
{
    g_sigsegv_count = 0;
    g_sigabrt_count = 0;
    g_sigbus_count = 0;
    g_sigfpe_count = 0;
    g_sigill_count = 0;
    g_sigterm_count = 0;
    g_sigint_count = 0;
    g_sighup_count = 0;
    g_recoveries = 0;
    g_fatal_crashes = 0;
}

bool signal_handler_shutdown_requested(void)
{
    return (g_shutdown_requested != 0);
}

bool signal_handler_reload_requested(void)
{
    bool requested = (g_reload_requested != 0);
    if (requested) {
        g_reload_requested = 0;  // Clear flag
    }
    return requested;
}

void signal_handler_set_crash_callback(void (*callback)(int sig))
{
    g_config.on_fatal_signal = callback;
}

void signal_handler_set_reload_callback(void (*callback)(void))
{
    g_config.on_reload_config = callback;
}

bool signal_handler_force_checkpoint(void)
{
    // Deprecated: use signal_handler_checkpoint_save() instead
    return signal_handler_checkpoint_save(NULL);
}

int signal_handler_get_last_signal(void)
{
    return (int)g_last_signal;
}

const char* signal_handler_get_signal_name(int sig)
{
    switch (sig) {
        case SIGSEGV: return "SIGSEGV";
        case SIGABRT: return "SIGABRT";
        case SIGBUS:  return "SIGBUS";
        case SIGFPE:  return "SIGFPE";
        case SIGILL:  return "SIGILL";
        case SIGTERM: return "SIGTERM";
        case SIGINT:  return "SIGINT";
        case SIGHUP:  return "SIGHUP";
        default:      return "UNKNOWN";
    }
}

//=============================================================================
// Enhanced Recovery & Diagnostics Implementation
//=============================================================================

/**
 * WHAT: Count checkpoints in retention directory
 * WHY:  Monitor checkpoint storage and manage retention
 * HOW:  Scan directory and count .nimcp files
 */
static int count_checkpoints_in_dir(const char* dir_path)
{
    if (!dir_path) {
        return -1;
    }

    DIR* dir = opendir(dir_path);
    if (!dir) {
        return -1;
    }

    int count = 0;
    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL) {
        // Count .nimcp checkpoint files (excluding backup files)
        if (entry->d_type == DT_REG && strstr(entry->d_name, ".nimcp")) {
            count++;
        }
    }

    closedir(dir);
    return count;
}

/**
 * WHAT: Cleanup old checkpoints to maintain retention limit
 * WHY:  Prevent disk space exhaustion from checkpoint accumulation
 * HOW:  Delete oldest files when limit exceeded
 */
static void cleanup_old_checkpoints(const char* dir_path, int max_count)
{
    if (!dir_path || max_count <= 0) {
        return;
    }

    DIR* dir = opendir(dir_path);
    if (!dir) {
        return;
    }

    // Collect checkpoint files with timestamps
    typedef struct {
        char* filename;
        time_t mtime;
    } checkpoint_entry_t;

    checkpoint_entry_t* entries = NULL;
    int entry_count = 0;
    int capacity = 16;

    entries = nimcp_malloc(sizeof(checkpoint_entry_t) * capacity);
    if (!entries) {
        closedir(dir);
        return;
    }

    // Scan directory
    struct dirent* entry;
    char fullpath[512];
    struct stat st;

    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_type == DT_REG && strstr(entry->d_name, ".nimcp")) {
            // Get file modification time
            snprintf(fullpath, sizeof(fullpath), "%s/%s", dir_path, entry->d_name);
            if (stat(fullpath, &st) == 0) {
                // Resize if needed
                if (entry_count >= capacity) {
                    capacity *= 2;
                    checkpoint_entry_t* new_entries = nimcp_realloc(entries,
                        sizeof(checkpoint_entry_t) * capacity);
                    if (!new_entries) {
                        break;
                    }
                    entries = new_entries;
                }

                entries[entry_count].filename = nimcp_malloc(strlen(entry->d_name) + 1);
                if (entries[entry_count].filename) {
                    strcpy(entries[entry_count].filename, entry->d_name);
                    entries[entry_count].mtime = st.st_mtime;
                    entry_count++;
                }
            }
        }
    }

    closedir(dir);

    // Sort by modification time (oldest first)
    for (int i = 0; i < entry_count - 1; i++) {
        for (int j = i + 1; j < entry_count; j++) {
            if (entries[j].mtime < entries[i].mtime) {
                // Swap
                checkpoint_entry_t temp = entries[i];
                entries[i] = entries[j];
                entries[j] = temp;
            }
        }
    }

    // Delete oldest files beyond retention limit
    for (int i = 0; i < entry_count && i < (entry_count - max_count); i++) {
        snprintf(fullpath, sizeof(fullpath), "%s/%s", dir_path, entries[i].filename);
        remove(fullpath);
    }

    // Cleanup
    for (int i = 0; i < entry_count; i++) {
        nimcp_free(entries[i].filename);
    }
    nimcp_free(entries);
}

signal_health_info_t signal_handler_get_health_status(void)
{
    signal_health_info_t health = {0};

    // Calculate totals
    uint64_t total_signals = g_sigsegv_count + g_sigabrt_count + g_sigbus_count +
                            g_sigfpe_count + g_sigill_count + g_sigterm_count +
                            g_sigint_count + g_sighup_count;

    health.total_signals = total_signals;
    health.fatal_crashes = g_fatal_crashes;
    health.successful_recoveries = g_recoveries;
    health.failed_recoveries = g_recovery_failures;
    health.checkpoint_saves = g_checkpoint_saves;
    health.last_signal = g_last_signal;
    health.last_signal_name = signal_handler_get_signal_name(g_last_signal);
    health.is_in_recovery = (g_recovery_attempt_count > 0);

    // Calculate recovery success rate
    uint64_t total_attempts = g_recoveries + g_recovery_failures;
    health.recovery_success_rate = (total_attempts > 0) ?
        (100.0f * g_recoveries / total_attempts) : 0.0f;

    // Determine health status based on metrics
    if (g_fatal_crashes == 0 && g_recovery_failures == 0 && g_sigsegv_count == 0) {
        health.status = SIGNAL_HEALTH_HEALTHY;
    } else if (g_recovery_failures > 0 || g_sigsegv_count > 5) {
        health.status = SIGNAL_HEALTH_CRITICAL;
    } else if (g_sigsegv_count > 3 || health.recovery_success_rate < 50.0f) {
        health.status = SIGNAL_HEALTH_COMPROMISED;
    } else if (g_fatal_crashes > 0 || g_recovery_failures > 0) {
        health.status = SIGNAL_HEALTH_DEGRADED;
    } else {
        health.status = SIGNAL_HEALTH_HEALTHY;
    }

    return health;
}

bool signal_handler_checkpoint_save(const char* checkpoint_path)
{
    // Use provided path or fall back to config default
    const char* path = checkpoint_path ? checkpoint_path : g_config.checkpoint_path;

    if (!g_registered_brain) {
        return false;
    }

    if (!path) {
        return false;
    }

    // Create checkpoint directory if it doesn't exist
    mkdir(path, 0755);

    // Generate timestamped checkpoint filename
    char checkpoint_file[512];
    time_t now = time(NULL);
    snprintf(checkpoint_file, sizeof(checkpoint_file),
             "%s/checkpoint_%ld.nimcp",
             path, now);

    // Attempt save
    if (brain_save(g_registered_brain, checkpoint_file)) {
        g_checkpoint_saves++;

        // Cleanup old checkpoints if retention limit is set
        if (g_checkpoint_retention > 0) {
            cleanup_old_checkpoints(path, g_checkpoint_retention);
        }

        return true;
    }

    return false;
}

void signal_handler_set_checkpoint_retention(int max_checkpoints)
{
    g_checkpoint_retention = max_checkpoints;
}

int signal_handler_get_checkpoint_count(void)
{
    return count_checkpoints_in_dir(g_config.checkpoint_path);
}

void signal_handler_set_auto_recovery(bool enable)
{
    g_auto_recovery_enabled = enable;
}

bool signal_handler_is_auto_recovery_enabled(void)
{
    return g_auto_recovery_enabled;
}

void signal_handler_set_max_recovery_attempts(int max_attempts)
{
    g_max_recovery_attempts = max_attempts;
}
