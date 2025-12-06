/**
 * @file nimcp_signal_handler.h
 * @brief Signal handling and crash recovery system
 *
 * WHAT: Graceful handling of fatal signals to prevent brain instance crashes
 * WHY:  Production systems need recovery from SIGSEGV, SIGABRT, etc.
 * HOW:  Install signal handlers, log state, attempt recovery or graceful shutdown
 *
 * SIGNALS HANDLED:
 * - SIGSEGV: Segmentation fault (NULL pointer, invalid memory access)
 * - SIGABRT: Abort signal (assertion failure, explicit abort)
 * - SIGBUS: Bus error (misaligned memory access)
 * - SIGFPE: Floating point exception (division by zero, overflow)
 * - SIGILL: Illegal instruction
 * - SIGTERM: Termination request (graceful shutdown)
 * - SIGINT: Interrupt (Ctrl+C)
 * - SIGHUP: Hang up (reload config)
 *
 * RECOVERY STRATEGY:
 * 1. Fatal signals (SIGSEGV, SIGBUS): Log state, save checkpoint, attempt cleanup
 * 2. Numeric errors (SIGFPE): Return NaN/Inf, continue processing
 * 3. Termination (SIGTERM, SIGINT): Graceful shutdown with state save
 * 4. Config reload (SIGHUP): Reload hyperparameters from config file
 *
 * @author NIMCP Team
 * @date 2025-11-09
 */

#ifndef NIMCP_SIGNAL_HANDLER_H
#define NIMCP_SIGNAL_HANDLER_H

#include <stdint.h>
#include <stdbool.h>
#include <signal.h>

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// Forward Declarations
//=============================================================================

typedef struct brain_struct* brain_t;

//=============================================================================
// Signal Handler Configuration
//=============================================================================

/**
 * @brief Signal handler behavior modes
 */
typedef enum {
    SIGNAL_MODE_IGNORE,      /**< Ignore signal (not recommended for fatal signals) */
    SIGNAL_MODE_LOG_ONLY,    /**< Log signal but don't handle */
    SIGNAL_MODE_LOG_CONTINUE,/**< Log signal and attempt to continue */
    SIGNAL_MODE_LOG_SHUTDOWN,/**< Log signal and graceful shutdown */
    SIGNAL_MODE_LOG_ABORT    /**< Log signal and abort immediately */
} signal_handler_mode_t;

/**
 * @brief Configuration for signal handler
 */
typedef struct {
    signal_handler_mode_t sigsegv_mode;  /**< SIGSEGV handling mode */
    signal_handler_mode_t sigabrt_mode;  /**< SIGABRT handling mode */
    signal_handler_mode_t sigbus_mode;   /**< SIGBUS handling mode */
    signal_handler_mode_t sigfpe_mode;   /**< SIGFPE handling mode */
    signal_handler_mode_t sigill_mode;   /**< SIGILL handling mode */
    signal_handler_mode_t sigterm_mode;  /**< SIGTERM handling mode */
    signal_handler_mode_t sigint_mode;   /**< SIGINT handling mode */
    signal_handler_mode_t sighup_mode;   /**< SIGHUP handling mode */

    bool enable_stack_trace;             /**< Log stack trace on crash */
    bool enable_state_dump;              /**< Dump brain state on crash */
    bool enable_checkpoint_save;         /**< Save checkpoint on crash */
    const char* crash_log_path;          /**< Path to crash log file */
    const char* checkpoint_path;         /**< Path to checkpoint directory */

    // Recovery callbacks
    void (*on_fatal_signal)(int sig);    /**< Called on fatal signal */
    void (*on_reload_config)(void);      /**< Called on SIGHUP */
    void (*on_graceful_shutdown)(void);  /**< Called on SIGTERM/SIGINT */
} signal_handler_config_t;

/**
 * @brief Signal statistics
 */
typedef struct {
    uint64_t sigsegv_count;     /**< Number of SIGSEGV signals received */
    uint64_t sigabrt_count;     /**< Number of SIGABRT signals received */
    uint64_t sigbus_count;      /**< Number of SIGBUS signals received */
    uint64_t sigfpe_count;      /**< Number of SIGFPE signals received */
    uint64_t sigill_count;      /**< Number of SIGILL signals received */
    uint64_t sigterm_count;     /**< Number of SIGTERM signals received */
    uint64_t sigint_count;      /**< Number of SIGINT signals received */
    uint64_t sighup_count;      /**< Number of SIGHUP signals received */
    uint64_t recoveries;        /**< Number of successful recoveries */
    uint64_t fatal_crashes;     /**< Number of unrecoverable crashes */
} signal_handler_stats_t;

//=============================================================================
// Public API
//=============================================================================

/**
 * @brief Get default signal handler configuration
 *
 * WHAT: Return safe defaults for signal handling
 * WHY:  Simplify initialization
 * HOW:  Fatal signals → LOG_SHUTDOWN, SIGHUP → reload config
 *
 * @return Default configuration
 */
signal_handler_config_t signal_handler_default_config(void);

/**
 * @brief Install signal handlers
 *
 * WHAT: Register signal handlers for all configured signals
 * WHY:  Enable graceful error handling
 * HOW:  Use sigaction() to install handlers
 *
 * @param config Signal handler configuration (NULL for defaults)
 * @return true on success, false on failure
 */
bool signal_handler_install(const signal_handler_config_t* config);

/**
 * @brief Uninstall signal handlers
 *
 * WHAT: Restore default signal handlers
 * WHY:  Clean shutdown or testing
 * HOW:  Restore SIG_DFL for all signals
 *
 * @return true on success, false on failure
 */
bool signal_handler_uninstall(void);

/**
 * @brief Register brain instance for crash recovery
 *
 * WHAT: Associate brain instance with signal handler
 * WHY:  Enable state dump and checkpoint save on crash
 * HOW:  Store brain pointer in signal-safe global
 *
 * @param brain Brain instance to register (NULL to unregister)
 */
void signal_handler_register_brain(brain_t brain);

/**
 * @brief Unregister brain instance (test cleanup)
 *
 * WHAT: Clear registered brain instance from signal handler
 * WHY:  Prevent test contamination from global state
 * HOW:  Set g_registered_brain to NULL
 *
 * NOTE: Should be called in test TearDown() to ensure clean state
 */
void signal_handler_unregister_brain(void);

/**
 * @brief Get signal handler statistics
 *
 * WHAT: Return signal counts and recovery stats
 * WHY:  Monitor system health and crash frequency
 * HOW:  Return copy of internal counters
 *
 * @return Signal statistics
 */
signal_handler_stats_t signal_handler_get_stats(void);

/**
 * @brief Reset signal handler statistics
 *
 * WHAT: Zero out all signal counters
 * WHY:  Fresh start for monitoring
 * HOW:  Memset stats to zero
 */
void signal_handler_reset_stats(void);

/**
 * @brief Check if graceful shutdown was requested
 *
 * WHAT: Check if SIGTERM/SIGINT was received
 * WHY:  Application loop can check and shutdown cleanly
 * HOW:  Return flag set by signal handler
 *
 * @return true if shutdown requested, false otherwise
 */
bool signal_handler_shutdown_requested(void);

/**
 * @brief Check if config reload was requested
 *
 * WHAT: Check if SIGHUP was received
 * WHY:  Application can reload config on SIGHUP
 * HOW:  Return and clear flag set by signal handler
 *
 * @return true if reload requested, false otherwise
 */
bool signal_handler_reload_requested(void);

/**
 * @brief Set custom crash callback
 *
 * WHAT: Register callback to be called before crash handling
 * WHY:  Application-specific cleanup or logging
 * HOW:  Store function pointer, call from signal handler
 *
 * @param callback Function to call on fatal signal (NULL to unset)
 */
void signal_handler_set_crash_callback(void (*callback)(int sig));

/**
 * @brief Set custom config reload callback
 *
 * WHAT: Register callback to be called on SIGHUP
 * WHY:  Application needs to reload config file
 * HOW:  Store function pointer, call from SIGHUP handler
 *
 * @param callback Function to call on SIGHUP (NULL to unset)
 */
void signal_handler_set_reload_callback(void (*callback)(void));

/**
 * @brief Force a checkpoint save
 *
 * WHAT: Manually trigger checkpoint save of registered brain
 * WHY:  Periodic checkpointing or before risky operations
 * HOW:  Call brain_save_checkpoint() if brain is registered
 *
 * @return true on success, false on failure
 */
bool signal_handler_force_checkpoint(void);

/**
 * @brief Get last signal received
 *
 * WHAT: Return signal number of last signal
 * WHY:  Debugging and diagnostics
 * HOW:  Return cached signal number
 *
 * @return Signal number or 0 if none
 */
int signal_handler_get_last_signal(void);

/**
 * @brief Get signal name as string
 *
 * WHAT: Convert signal number to name string
 * WHY:  Human-readable logging
 * HOW:  Map signal numbers to names
 *
 * @param sig Signal number
 * @return Signal name string (e.g., "SIGSEGV")
 */
const char* signal_handler_get_signal_name(int sig);

//=============================================================================
// Enhanced Recovery & Diagnostics API
//=============================================================================

/**
 * @brief Health status of the system
 */
typedef enum {
    SIGNAL_HEALTH_HEALTHY,      /**< All systems operational */
    SIGNAL_HEALTH_DEGRADED,     /**< Some issues detected, recovery attempted */
    SIGNAL_HEALTH_COMPROMISED,  /**< Multiple issues, stability at risk */
    SIGNAL_HEALTH_CRITICAL,     /**< Critical issues, immediate attention needed */
    SIGNAL_HEALTH_UNKNOWN        /**< Status cannot be determined */
} signal_health_status_t;

/**
 * @brief Detailed health information
 */
typedef struct {
    signal_health_status_t status;           /**< Overall health status */
    uint64_t total_signals;                  /**< Total signals received */
    uint64_t fatal_crashes;                  /**< Unrecoverable crashes */
    uint64_t successful_recoveries;          /**< Recovery attempts that succeeded */
    uint64_t failed_recoveries;              /**< Recovery attempts that failed */
    float recovery_success_rate;             /**< % of recoveries that succeeded */
    int last_signal;                         /**< Last signal received */
    const char* last_signal_name;            /**< Name of last signal */
    uint64_t checkpoint_saves;               /**< Checkpoints successfully saved */
    bool is_in_recovery;                     /**< Currently attempting recovery */
} signal_health_info_t;

/**
 * @brief Get current health status
 *
 * WHAT: Return comprehensive health information
 * WHY:  Monitor system stability and recovery effectiveness
 * HOW:  Analyze signal stats, recovery history, and system state
 *
 * @return Health information structure
 */
signal_health_info_t signal_handler_get_health_status(void);

/**
 * @brief Attempt to manually trigger checkpoint save
 *
 * WHAT: Synchronously save registered brain state
 * WHY:  Guarantee consistent checkpoint before risky operations
 * HOW:  Call brain_save() if safe, handle errors gracefully
 *
 * @param checkpoint_path Path to save checkpoint to (NULL for config default)
 * @return true on success, false on failure or no brain registered
 */
bool signal_handler_checkpoint_save(const char* checkpoint_path);

/**
 * @brief Set the number of checkpoints to keep
 *
 * WHAT: Configure checkpoint retention policy
 * WHY:  Balance disk usage vs. recovery history
 * HOW:  Manage rotating checkpoint files
 *
 * @param max_checkpoints Maximum number to retain (0 = unlimited)
 */
void signal_handler_set_checkpoint_retention(int max_checkpoints);

/**
 * @brief Get checkpoint statistics
 *
 * WHAT: Return information about saved checkpoints
 * WHY:  Monitor checkpoint health and disk usage
 * HOW:  Scan checkpoint directory and return stats
 *
 * @return Count of saved checkpoints (-1 on error)
 */
int signal_handler_get_checkpoint_count(void);

/**
 * @brief Enable or disable automatic recovery attempts
 *
 * WHAT: Control whether signal handler tries to recover
 * WHY:  Allow explicit control over recovery behavior
 * HOW:  Set internal flag checked during signal handling
 *
 * @param enable true to enable automatic recovery, false to disable
 */
void signal_handler_set_auto_recovery(bool enable);

/**
 * @brief Get current recovery configuration status
 *
 * WHAT: Return whether automatic recovery is enabled
 * WHY:  Query current recovery settings
 * HOW:  Return flag state
 *
 * @return true if auto-recovery enabled, false otherwise
 */
bool signal_handler_is_auto_recovery_enabled(void);

/**
 * @brief Set maximum recovery attempts before giving up
 *
 * WHAT: Configure how many times we attempt recovery for same issue
 * WHY:  Prevent infinite recovery loops
 * HOW:  Store limit and track attempts per signal
 *
 * @param max_attempts Maximum recovery attempts (0 = unlimited)
 */
void signal_handler_set_max_recovery_attempts(int max_attempts);

#ifdef __cplusplus
}
#endif

#endif // NIMCP_SIGNAL_HANDLER_H
