/**
 * @file nimcp_blood_brain_barrier.c
 * @brief Blood-Brain Barrier - Perimeter Defense Layer Implementation
 *
 * WHAT: Implementation of the unified BBB API for perimeter security
 * WHY:  Prevent code injection, memory corruption, and unauthorized access
 * HOW:  Four-layer defense with thread-safe operations and threat tracking
 *
 * BIOLOGICAL MODEL:
 *   Endothelial cells (tight junctions) -> Input validation gates
 *   Basement membrane                   -> Code signing verification layer
 *   Astrocyte end-feet                  -> Memory boundary monitors
 *   Pericytes                           -> Access control enforcers
 *
 * NIMCP STANDARDS:
 *   - All functions < 50 lines
 *   - Guard clauses (early returns)
 *   - WHAT-WHY-HOW documentation
 *   - Thread-safe operations
 *
 * @version 1.0.0
 * @date 2025-11-24
 * @author NIMCP Development Team
 */

//=============================================================================
// Includes
//=============================================================================

#include "security/nimcp_blood_brain_barrier.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/thread/nimcp_thread.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>

//=============================================================================
// Constants
//=============================================================================

/**
 * WHAT: Maximum number of threat reports stored in circular buffer
 * WHY:  Limits memory usage while maintaining recent threat history
 */
#define BBB_MAX_THREAT_REPORTS 100

/**
 * WHAT: Maximum number of quarantined memory regions
 * WHY:  Limits tracking overhead for isolated threat regions
 */
#define BBB_MAX_QUARANTINE_REGIONS 64

//=============================================================================
// Internal Structures
//=============================================================================

/**
 * WHAT: Quarantined memory region tracking
 * WHY:  Track isolated regions containing detected threats
 */
typedef struct {
    void* address;      /**< Start address of quarantined region */
    size_t size;        /**< Size of quarantined region */
    uint64_t timestamp; /**< When region was quarantined */
    bool active;        /**< Whether region is still quarantined */
} bbb_quarantine_entry_t;

/**
 * WHAT: Internal BBB system structure
 * WHY:  Encapsulates all BBB state with thread-safe access
 */
struct bbb_system_struct {
    /* Configuration */
    bbb_config_t config;

    /* System state */
    bool enabled;
    uint64_t creation_time;

    /* Thread safety */
    nimcp_mutex_t mutex;
    bool mutex_initialized;

    /* Statistics */
    bbb_statistics_t stats;

    /* Threat report circular buffer */
    bbb_threat_report_t threat_reports[BBB_MAX_THREAT_REPORTS];
    size_t threat_report_head;
    size_t threat_report_count;

    /* Quarantine list */
    bbb_quarantine_entry_t quarantine[BBB_MAX_QUARANTINE_REGIONS];
    size_t quarantine_count;
};

//=============================================================================
// Note: bbb_calculate_hash is implemented in nimcp_bbb_code_signing.c
// using proper SHA256 algorithm for cryptographic security
//=============================================================================

//=============================================================================
// Utility Functions - Name Conversions
//=============================================================================

/**
 * @brief Get human-readable threat type name
 *
 * WHAT: Convert threat type enum to string
 * WHY:  Human-readable logging and reporting
 * HOW:  Static lookup table indexed by enum
 */
NIMCP_EXPORT const char* bbb_threat_type_name(bbb_threat_type_t type)
{
    static const char* names[] = {
        "NONE",
        "BUFFER_OVERFLOW",
        "FORMAT_STRING",
        "INTEGER_OVERFLOW",
        "SQL_INJECTION",
        "CODE_INJECTION",
        "SHELLCODE",
        "ROP_CHAIN",
        "INVALID_SIGNATURE",
        "MEMORY_VIOLATION",
        "UNAUTHORIZED_ACCESS",
        "DATA_TAMPERING",
        "UNKNOWN"
    };

    if (type < 0 || type > BBB_THREAT_UNKNOWN) {
        return "INVALID";
    }

    return names[type];
}

/**
 * @brief Get human-readable severity name
 *
 * WHAT: Convert severity enum to string
 * WHY:  Human-readable logging and reporting
 * HOW:  Static lookup table indexed by enum
 */
NIMCP_EXPORT const char* bbb_severity_name(bbb_severity_t severity)
{
    static const char* names[] = {
        "NONE",
        "LOW",
        "MEDIUM",
        "HIGH",
        "CRITICAL"
    };

    if (severity < 0 || severity > BBB_SEVERITY_CRITICAL) {
        return "INVALID";
    }

    return names[severity];
}

/**
 * @brief Get human-readable action name
 *
 * WHAT: Convert action enum to string
 * WHY:  Human-readable logging and reporting
 * HOW:  Static lookup table indexed by enum
 */
NIMCP_EXPORT const char* bbb_action_name(bbb_action_t action)
{
    static const char* names[] = {
        "ALLOW",
        "LOG",
        "BLOCK",
        "QUARANTINE",
        "TERMINATE",
        "LOCKDOWN"
    };

    if (action < 0 || action > BBB_ACTION_LOCKDOWN) {
        return "INVALID";
    }

    return names[action];
}

//=============================================================================
// Configuration
//=============================================================================

/**
 * @brief Get default BBB configuration with conservative security settings
 *
 * WHAT: Return default configuration struct
 * WHY:  Provide sensible defaults for immediate use
 * HOW:  Static initialization of all config fields
 */
NIMCP_EXPORT bbb_config_t bbb_default_config(void)
{
    bbb_config_t config;
    memset(&config, 0, sizeof(config));

    /* Input gate defaults - conservative validation */
    config.input.validate_strings = true;
    config.input.validate_integers = true;
    config.input.validate_pointers = true;
    config.input.sanitize_html = true;
    config.input.sanitize_sql = true;
    config.input.max_string_length = 65536;     /* 64KB max string */
    config.input.max_array_size = 1048576;      /* 1M elements max */
    config.input.min_integer = INT64_MIN / 2;   /* Half range to detect overflow */
    config.input.max_integer = INT64_MAX / 2;

    /* Code signing defaults */
    config.signing.require_signatures = false;  /* Off by default for ease of use */
    config.signing.verify_on_load = true;
    config.signing.verify_on_execute = false;   /* Performance-sensitive */
    config.signing.trusted_keys_path = NULL;
    config.signing.hash_algorithm = 0;          /* SHA256 */

    /* Memory boundary defaults */
    config.memory.enable_stack_canaries = true;
    config.memory.enable_heap_guards = true;
    config.memory.enable_aslr = true;
    config.memory.enable_wx_protection = true;
    config.memory.enable_mprotect = true;
    config.memory.guard_page_size = 4096;
    config.memory.stack_canary_size = 8;

    /* Access control defaults */
    config.access.enable_rbac = true;
    config.access.enable_mac = false;           /* Mandatory AC off by default */
    config.access.enable_capability = true;
    config.access.log_access_attempts = true;
    config.access.max_privilege_level = 16;   /* Support up to 16 privilege levels */
    config.access.policy_file = NULL;

    /* System defaults */
    config.strict_mode = false;
    config.default_action = BBB_ACTION_BLOCK;
    config.alert_callback = NULL;

    return config;
}

//=============================================================================
// Internal Statistics Update Functions (for sub-modules)
//=============================================================================

/**
 * @brief Increment validation counter (internal use)
 *
 * WHAT: Thread-safe increment of total_validations stat
 * WHY:  Allow sub-modules to update stats without exposing struct
 */
void bbb_system_inc_validations(bbb_system_t system)
{
    if (!system) return;
    nimcp_mutex_lock(&system->mutex);
    system->stats.total_validations++;
    nimcp_mutex_unlock(&system->mutex);
}

/**
 * @brief Increment threat detection counter (internal use)
 *
 * WHAT: Thread-safe increment of threats_detected stat
 * WHY:  Allow sub-modules to update stats without exposing struct
 */
void bbb_system_inc_threats(bbb_system_t system)
{
    if (!system) return;
    nimcp_mutex_lock(&system->mutex);
    system->stats.threats_detected++;
    nimcp_mutex_unlock(&system->mutex);
}

//=============================================================================
// System Lifecycle
//=============================================================================

/**
 * @brief Create BBB system with configuration
 *
 * WHAT: Allocate and initialize BBB system struct
 * WHY:  Create new security perimeter instance
 * HOW:  Malloc, initialize mutex, copy config, set defaults
 */
NIMCP_EXPORT bbb_system_t bbb_system_create(const bbb_config_t* config)
{
    /* Allocate system structure */
    bbb_system_t system = (bbb_system_t)nimcp_calloc(1, sizeof(struct bbb_system_struct));
    if (!system) {
        return NULL;
    }

    /* Initialize mutex */
    if (nimcp_mutex_init(&system->mutex, NULL) != NIMCP_SUCCESS) {
        nimcp_free(system);
        return NULL;
    }
    system->mutex_initialized = true;

    /* Copy configuration or use defaults */
    if (config) {
        system->config = *config;
    } else {
        system->config = bbb_default_config();
    }

    /* Initialize state */
    system->enabled = true;
    system->creation_time = (uint64_t)time(NULL);
    system->threat_report_head = 0;
    system->threat_report_count = 0;
    system->quarantine_count = 0;

    /* Initialize statistics */
    memset(&system->stats, 0, sizeof(system->stats));

    return system;
}

/**
 * @brief Destroy BBB system and free all resources
 *
 * WHAT: Clean up and deallocate BBB system
 * WHY:  Prevent memory leaks and resource exhaustion
 * HOW:  Destroy mutex, zero sensitive data, free memory
 */
NIMCP_EXPORT void bbb_system_destroy(bbb_system_t system)
{
    /* Guard: Null system */
    if (!system) {
        return;
    }

    /* Destroy mutex if initialized */
    if (system->mutex_initialized) {
        nimcp_mutex_destroy(&system->mutex);
    }

    /* Zero sensitive data before freeing */
    memset(system->threat_reports, 0, sizeof(system->threat_reports));
    memset(&system->stats, 0, sizeof(system->stats));

    /* Free system */
    nimcp_free(system);
}

//=============================================================================
// Enable/Disable Control
//=============================================================================

/**
 * @brief Enable or disable the BBB system
 *
 * WHAT: Toggle BBB protection on/off
 * WHY:  Allow runtime control of security layer
 * HOW:  Thread-safe flag modification
 */
NIMCP_EXPORT bool bbb_system_set_enabled(bbb_system_t system, bool enabled)
{
    /* Guard: Null system */
    if (!system) {
        return false;
    }

    nimcp_mutex_lock(&system->mutex);
    system->enabled = enabled;
    nimcp_mutex_unlock(&system->mutex);

    return true;
}

/**
 * @brief Check if BBB system is enabled
 *
 * WHAT: Query current enabled state
 * WHY:  Check protection status before operations
 * HOW:  Thread-safe flag read
 */
NIMCP_EXPORT bool bbb_system_is_enabled(bbb_system_t system)
{
    /* Guard: Null system */
    if (!system) {
        return false;
    }

    nimcp_mutex_lock(&system->mutex);
    bool enabled = system->enabled;
    nimcp_mutex_unlock(&system->mutex);

    return enabled;
}

//=============================================================================
// Statistics
//=============================================================================

/**
 * @brief Get BBB system statistics
 *
 * WHAT: Retrieve current security statistics
 * WHY:  Monitoring, reporting, and analysis
 * HOW:  Thread-safe copy of stats struct
 */
NIMCP_EXPORT bool bbb_system_get_statistics(bbb_system_t system, bbb_statistics_t* stats)
{
    /* Guard: Null parameters */
    if (!system || !stats) {
        return false;
    }

    nimcp_mutex_lock(&system->mutex);

    /* Copy statistics */
    *stats = system->stats;

    /* Calculate uptime */
    stats->uptime_seconds = (uint64_t)time(NULL) - system->creation_time;

    nimcp_mutex_unlock(&system->mutex);

    return true;
}

/**
 * @brief Reset BBB system statistics
 *
 * WHAT: Clear all accumulated statistics
 * WHY:  Start fresh measurement period
 * HOW:  Thread-safe zero of stats struct
 */
NIMCP_EXPORT void bbb_system_reset_statistics(bbb_system_t system)
{
    /* Guard: Null system */
    if (!system) {
        return;
    }

    nimcp_mutex_lock(&system->mutex);
    memset(&system->stats, 0, sizeof(system->stats));
    nimcp_mutex_unlock(&system->mutex);
}

//=============================================================================
// Threat Reporting
//=============================================================================

/**
 * @brief Determine action based on severity and configuration
 *
 * WHAT: Map severity to appropriate response action
 * WHY:  Automated threat response based on severity
 * HOW:  Severity threshold checks
 */
static bbb_action_t determine_action(bbb_system_t system, bbb_severity_t severity)
{
    if (system->config.strict_mode) {
        /* Strict mode: aggressive response */
        if (severity >= BBB_SEVERITY_MEDIUM) {
            return BBB_ACTION_QUARANTINE;
        }
        return BBB_ACTION_BLOCK;
    }

    /* Normal mode: graduated response */
    switch (severity) {
        case BBB_SEVERITY_NONE:
            return BBB_ACTION_ALLOW;
        case BBB_SEVERITY_LOW:
            return BBB_ACTION_LOG;
        case BBB_SEVERITY_MEDIUM:
            return BBB_ACTION_BLOCK;
        case BBB_SEVERITY_HIGH:
            return BBB_ACTION_QUARANTINE;
        case BBB_SEVERITY_CRITICAL:
            return BBB_ACTION_TERMINATE;
        default:
            return system->config.default_action;
    }
}

/**
 * @brief Report a detected threat to the BBB system
 *
 * WHAT: Create threat report with timestamp and hash
 * WHY:  Record threats for analysis and response
 * HOW:  Populate report struct, add to circular buffer
 */
NIMCP_EXPORT bbb_threat_report_t bbb_report_threat(bbb_system_t system,
                                                    bbb_threat_type_t type,
                                                    bbb_severity_t severity,
                                                    const char* description,
                                                    const void* source_address,
                                                    const void* threat_data,
                                                    size_t threat_size)
{
    bbb_threat_report_t report;
    memset(&report, 0, sizeof(report));

    /* Guard: Null system - return empty report */
    if (!system) {
        return report;
    }

    /* Populate report */
    report.type = type;
    report.severity = severity;
    report.timestamp = (uint64_t)time(NULL);
    report.source_address = source_address;
    report.threat_size = threat_size;
    report.quarantined = false;

    /* Copy description */
    if (description) {
        strncpy(report.description, description, sizeof(report.description) - 1);
        report.description[sizeof(report.description) - 1] = '\0';
    }

    /* Calculate threat hash */
    if (threat_data && threat_size > 0) {
        bbb_calculate_hash(threat_data, threat_size, report.threat_hash);
    }

    /* Determine action */
    report.action_taken = determine_action(system, severity);

    /* Thread-safe addition to circular buffer */
    nimcp_mutex_lock(&system->mutex);

    /* Update statistics */
    system->stats.threats_detected++;
    if (report.action_taken == BBB_ACTION_BLOCK ||
        report.action_taken == BBB_ACTION_QUARANTINE) {
        /* Both BLOCK and QUARANTINE prevent the threat */
        system->stats.threats_blocked++;
    }
    if (report.action_taken == BBB_ACTION_QUARANTINE) {
        system->stats.threats_quarantined++;
        report.quarantined = true;
    }

    /* Add to circular buffer */
    system->threat_reports[system->threat_report_head] = report;
    system->threat_report_head = (system->threat_report_head + 1) % BBB_MAX_THREAT_REPORTS;
    if (system->threat_report_count < BBB_MAX_THREAT_REPORTS) {
        system->threat_report_count++;
    }

    nimcp_mutex_unlock(&system->mutex);

    /* Invoke alert callback if configured */
    if (system->config.alert_callback && severity >= BBB_SEVERITY_MEDIUM) {
        system->config.alert_callback(type, severity, description);
    }

    return report;
}

/**
 * @brief Get recent threat reports from circular buffer
 *
 * WHAT: Retrieve stored threat reports
 * WHY:  Analysis and forensics of detected threats
 * HOW:  Copy from circular buffer to output array
 */
NIMCP_EXPORT size_t bbb_get_threat_reports(bbb_system_t system,
                                           bbb_threat_report_t* reports,
                                           size_t max_reports)
{
    /* Guard: Null parameters */
    if (!system || !reports || max_reports == 0) {
        return 0;
    }

    nimcp_mutex_lock(&system->mutex);

    size_t count = (system->threat_report_count < max_reports)
                   ? system->threat_report_count : max_reports;

    /* Copy reports from circular buffer (newest first) */
    for (size_t i = 0; i < count; i++) {
        size_t idx = (system->threat_report_head + BBB_MAX_THREAT_REPORTS - 1 - i)
                     % BBB_MAX_THREAT_REPORTS;
        reports[i] = system->threat_reports[idx];
    }

    nimcp_mutex_unlock(&system->mutex);

    return count;
}

/**
 * @brief Clear all threat reports
 *
 * WHAT: Empty the threat report buffer
 * WHY:  Reset after handling or periodic cleanup
 * HOW:  Zero buffer and reset indices
 */
NIMCP_EXPORT void bbb_clear_threat_reports(bbb_system_t system)
{
    /* Guard: Null system */
    if (!system) {
        return;
    }

    nimcp_mutex_lock(&system->mutex);
    memset(system->threat_reports, 0, sizeof(system->threat_reports));
    system->threat_report_head = 0;
    system->threat_report_count = 0;
    nimcp_mutex_unlock(&system->mutex);
}

//=============================================================================
// Quarantine Management
//=============================================================================

/**
 * @brief Check if an address is in a quarantined region
 *
 * WHAT: Test if address overlaps any quarantined region
 * WHY:  Prevent access to quarantined memory
 * HOW:  Scan quarantine list for overlap
 */
NIMCP_EXPORT bool bbb_is_quarantined(bbb_system_t system, const void* address, size_t size)
{
    if (!system || !address || size == 0) {
        return false;
    }

    nimcp_mutex_lock(&system->mutex);

    uintptr_t check_start = (uintptr_t)address;
    uintptr_t check_end = check_start + size;

    for (size_t i = 0; i < BBB_MAX_QUARANTINE_REGIONS; i++) {
        if (!system->quarantine[i].active) continue;

        uintptr_t q_start = (uintptr_t)system->quarantine[i].address;
        uintptr_t q_end = q_start + system->quarantine[i].size;

        /* Check for overlap */
        if (check_start < q_end && check_end > q_start) {
            nimcp_mutex_unlock(&system->mutex);
            return true;
        }
    }

    nimcp_mutex_unlock(&system->mutex);
    return false;
}

/**
 * @brief Quarantine a memory region containing a threat
 *
 * WHAT: Isolate suspicious memory region
 * WHY:  Contain threat while preserving evidence
 * HOW:  Add to quarantine tracking list
 */
NIMCP_EXPORT bool bbb_quarantine_region(bbb_system_t system, void* address, size_t size)
{
    /* Guard: Null parameters */
    if (!system || !address || size == 0) {
        return false;
    }

    nimcp_mutex_lock(&system->mutex);

    /* Check if we have room */
    if (system->quarantine_count >= BBB_MAX_QUARANTINE_REGIONS) {
        nimcp_mutex_unlock(&system->mutex);
        return false;
    }

    /* Find free slot (may have gaps from releases) */
    size_t slot = 0;
    bool found = false;
    for (size_t i = 0; i < BBB_MAX_QUARANTINE_REGIONS; i++) {
        if (!system->quarantine[i].active) {
            slot = i;
            found = true;
            break;
        }
    }

    if (!found) {
        nimcp_mutex_unlock(&system->mutex);
        return false;
    }

    /* Add to quarantine */
    system->quarantine[slot].address = address;
    system->quarantine[slot].size = size;
    system->quarantine[slot].timestamp = (uint64_t)time(NULL);
    system->quarantine[slot].active = true;
    system->quarantine_count++;

    /* Update statistics */
    system->stats.threats_quarantined++;

    nimcp_mutex_unlock(&system->mutex);

    return true;
}

/**
 * @brief Release a quarantined memory region
 *
 * WHAT: Remove region from quarantine tracking
 * WHY:  Allow cleanup after threat handling
 * HOW:  Find and deactivate quarantine entry
 */
NIMCP_EXPORT bool bbb_release_quarantine(bbb_system_t system, void* address)
{
    /* Guard: Null parameters */
    if (!system || !address) {
        return false;
    }

    nimcp_mutex_lock(&system->mutex);

    /* Find quarantine entry */
    bool found = false;
    for (size_t i = 0; i < BBB_MAX_QUARANTINE_REGIONS; i++) {
        if (system->quarantine[i].active &&
            system->quarantine[i].address == address) {
            system->quarantine[i].active = false;
            system->quarantine_count--;
            found = true;
            break;
        }
    }

    nimcp_mutex_unlock(&system->mutex);

    return found;
}

//=============================================================================
// Print Functions
//=============================================================================

/**
 * @brief Print BBB statistics to stdout
 *
 * WHAT: Display formatted statistics summary
 * WHY:  Human-readable status reporting
 * HOW:  Printf formatted output
 */
NIMCP_EXPORT void bbb_print_statistics(const bbb_statistics_t* stats)
{
    /* Guard: Null stats */
    if (!stats) {
        printf("BBB Statistics: (null)\n");
        return;
    }

    printf("\n");
    printf("=== Blood-Brain Barrier Statistics ===\n");
    printf("Uptime:                %lu seconds\n", (unsigned long)stats->uptime_seconds);
    printf("Total Validations:     %lu\n", (unsigned long)stats->total_validations);
    printf("Threats Detected:      %lu\n", (unsigned long)stats->threats_detected);
    printf("Threats Blocked:       %lu\n", (unsigned long)stats->threats_blocked);
    printf("Threats Quarantined:   %lu\n", (unsigned long)stats->threats_quarantined);
    printf("False Positives:       %lu\n", (unsigned long)stats->false_positives);
    printf("Signatures Verified:   %lu\n", (unsigned long)stats->signatures_verified);
    printf("Signature Failures:    %lu\n", (unsigned long)stats->signatures_failed);
    printf("Memory Violations:     %lu\n", (unsigned long)stats->memory_violations);
    printf("Access Violations:     %lu\n", (unsigned long)stats->access_violations);
    printf("======================================\n");
    printf("\n");
}

/**
 * @brief Print threat report to stdout
 *
 * WHAT: Display formatted threat report
 * WHY:  Human-readable threat details
 * HOW:  Printf formatted output with hash display
 */
NIMCP_EXPORT void bbb_print_threat_report(const bbb_threat_report_t* report)
{
    /* Guard: Null report */
    if (!report) {
        printf("Threat Report: (null)\n");
        return;
    }

    printf("\n");
    printf("=== Threat Report ===\n");
    printf("Type:        %s\n", bbb_threat_type_name(report->type));
    printf("Severity:    %s\n", bbb_severity_name(report->severity));
    printf("Action:      %s\n", bbb_action_name(report->action_taken));
    printf("Timestamp:   %lu\n", (unsigned long)report->timestamp);
    printf("Source:      %p\n", report->source_address);
    printf("Size:        %zu bytes\n", report->threat_size);
    printf("Description: %s\n", report->description);
    printf("Quarantined: %s\n", report->quarantined ? "YES" : "NO");

    /* Print hash in hex */
    printf("Hash:        ");
    for (int i = 0; i < 32; i++) {
        printf("%02x", report->threat_hash[i]);
    }
    printf("\n");
    printf("=====================\n");
    printf("\n");
}

//=============================================================================
// Note: Full implementations of validation, signing, memory, and access control
// APIs are in their respective sub-modules:
// - nimcp_bbb_input_gate.c: Input validation (bbb_validate_*, bbb_sanitize_*)
// - nimcp_bbb_code_signing.c: Code signing (bbb_sign_*, bbb_verify_*, bbb_*_trusted_key)
// - nimcp_bbb_memory_boundary.c: Memory protection (bbb_*_memory_*, bbb_*_stack_canary)
// - nimcp_bbb_access_control.c: Access control (bbb_check_access, bbb_register_*, bbb_*_capability)
//=============================================================================

//=============================================================================
// Test Support - Unified Reset Function
//=============================================================================

/* Forward declarations for internal reset functions in sub-modules */
extern void bbb_access_control_reset_internal(void);
extern void bbb_memory_boundary_reset_internal(void);

/**
 * @brief Reset all BBB subsystem state for testing
 *
 * WHAT: Clear all registered subjects, objects, memory regions, etc.
 * WHY:  Enable test isolation by resetting between test cases
 * HOW:  Call internal reset functions for each subsystem
 *
 * NOTE: This is a unified reset to avoid DRY violations
 */
void bbb_reset_test_state(void)
{
    bbb_access_control_reset_internal();
    bbb_memory_boundary_reset_internal();
}
