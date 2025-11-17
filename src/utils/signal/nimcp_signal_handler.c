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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <execinfo.h>  // For backtrace()
#include <time.h>

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

// Configuration (set once during initialization)
static signal_handler_config_t g_config;
static bool g_installed = false;

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
 * HOW:  Write to stderr, optionally log stack trace, then exit
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

    // TODO: If enable_checkpoint_save, attempt to save brain state
    // NOTE: This is risky in a signal handler - only attempt if in safe state

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
    // TODO: Implement brain checkpoint saving
    // For now, just return false (not implemented)
    (void)g_registered_brain;
    return false;
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
