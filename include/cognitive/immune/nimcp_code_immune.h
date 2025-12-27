/**
 * @file nimcp_code_immune.h
 * @brief Code Immune System - Runtime Crash Defense and Hot-Patching
 * @version 1.0.0
 * @date 2025-12-27
 *
 * WHAT: Immune system that detects code crashes, learns crash signatures,
 *       and generates runtime patches to prevent recurrence.
 * WHY:  Enable self-healing software that can survive and adapt to bugs.
 * HOW:  Integrate with signal handler for crash detection, pattern-match
 *       crash signatures (epitopes), produce code antibodies (patches).
 *
 * BIOLOGICAL MODEL:
 * ```
 * BIOLOGICAL CONCEPT              CODE IMMUNE IMPLEMENTATION
 * ───────────────────────────────────────────────────────────────────
 * Antigen presentation         -> Crash signal -> code_antigen_t
 * B cells (antibody production)-> Pattern matching -> fix generation
 * Memory B cells               -> Learned crash patterns for fast response
 * Antibodies                   -> Hot patches / .so injection
 * Apoptosis                    -> Old code cleanup / memory reclaim
 * ```
 *
 * ARCHITECTURE:
 * ```
 * ╔═══════════════════════════════════════════════════════════════════════════╗
 * ║                    CODE IMMUNE SYSTEM                                      ║
 * ╠═══════════════════════════════════════════════════════════════════════════╣
 * ║                                                                            ║
 * ║   ┌────────────────────────────────────────────────────────────────────┐  ║
 * ║   │                 SIGNAL HANDLER INTEGRATION                          │  ║
 * ║   │   ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐    │  ║
 * ║   │   │   SIGSEGV    │  │   SIGBUS     │  │     SIGABRT          │    │  ║
 * ║   │   └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘    │  ║
 * ║   │          └─────────────────┼──────────────────────┘                │  ║
 * ║   │                            ▼                                       │  ║
 * ║   │                 [code_immune_present_crash]                        │  ║
 * ║   │                    (Crash as Antigen)                              │  ║
 * ║   └────────────────────────────┬───────────────────────────────────────┘  ║
 * ║                                ▼                                          ║
 * ║   ┌────────────────────────────────────────────────────────────────────┐  ║
 * ║   │                    PATTERN MATCHING                                 │  ║
 * ║   │                                                                     │  ║
 * ║   │   ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐   │  ║
 * ║   │   │  B CELLS    │    │   MEMORY    │    │    FIX TEMPLATES    │   │  ║
 * ║   │   │ ──────────  │    │  B CELLS    │    │ ─────────────────   │   │  ║
 * ║   │   │ Crash       │    │  Learned    │    │ Null check, bounds  │   │  ║
 * ║   │   │ Signatures  │    │  Patterns   │    │ Alignment, div/0    │   │  ║
 * ║   │   └─────────────┘    └─────────────┘    └─────────────────────┘   │  ║
 * ║   └────────────────────────────────────────────────────────────────────┘  ║
 * ║                                │                                          ║
 * ║                                ▼                                          ║
 * ║   ┌────────────────────────────────────────────────────────────────────┐  ║
 * ║   │                    ANTIBODY PRODUCTION                              │  ║
 * ║   │                                                                     │  ║
 * ║   │   ┌─────────────────────────────────────────────────────────────┐  │  ║
 * ║   │   │                     code_antibody_t                          │  │  ║
 * ║   │   │    Hot patch generation (.so) / Function redirection         │  │  ║
 * ║   │   │    (IGM = quick/untested, IGG = verified, IGE = emergency)   │  │  ║
 * ║   │   └─────────────────────────────────────────────────────────────┘  │  ║
 * ║   └────────────────────────────────────────────────────────────────────┘  ║
 * ║                                                                            ║
 * ╚═══════════════════════════════════════════════════════════════════════════╝
 * ```
 *
 * NIMCP STANDARDS:
 * - All functions < 50 lines
 * - Guard clauses (early returns)
 * - WHAT-WHY-HOW documentation
 * - Thread-safe via mutex
 * - nimcp_malloc/nimcp_free memory management
 *
 * @author NIMCP Development Team
 */

#ifndef NIMCP_CODE_IMMUNE_H
#define NIMCP_CODE_IMMUNE_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <signal.h>

/* Integration with parent immune system */
#include "cognitive/immune/nimcp_brain_immune.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Constants
 * ============================================================================ */

#define CODE_IMMUNE_MAX_ANTIGENS       128   /**< Max pending crash antigens */
#define CODE_IMMUNE_MAX_B_CELLS        256   /**< Max code B cells */
#define CODE_IMMUNE_MAX_ANTIBODIES     512   /**< Max active code antibodies */
#define CODE_IMMUNE_EPITOPE_SIZE       64    /**< Crash signature hash size */
#define CODE_IMMUNE_BACKTRACE_DEPTH    32    /**< Max backtrace frames */
#define CODE_IMMUNE_MODULE_NAME        "code_immune"

/* ============================================================================
 * Forward Declarations
 * ============================================================================ */

typedef struct code_immune_system code_immune_system_t;

/* ============================================================================
 * Enumerations
 * ============================================================================ */

/**
 * @brief Crash signal types (bitmask for pattern matching)
 *
 * BIOLOGICAL BASIS:
 * Different crash types are like different pathogen classes.
 * B cells can specialize in handling specific crash types.
 */
typedef enum {
    CODE_CRASH_NONE     = 0,
    CODE_CRASH_SIGSEGV  = 1,       /**< Segmentation fault - NULL/invalid ptr */
    CODE_CRASH_SIGBUS   = 2,       /**< Bus error - misaligned access */
    CODE_CRASH_SIGILL   = 4,       /**< Illegal instruction */
    CODE_CRASH_SIGFPE   = 8,       /**< Floating point exception */
    CODE_CRASH_SIGABRT  = 16,      /**< Abort - assertion/explicit abort */
    CODE_CRASH_ALL      = 31       /**< All crash types */
} code_crash_type_t;

/**
 * @brief Code B cell states (parallel to brain_b_cell_state_t)
 *
 * BIOLOGICAL BASIS:
 * B cells progress through activation states as they learn
 * to recognize and respond to crash patterns.
 */
typedef enum {
    CODE_B_CELL_NAIVE = 0,         /**< Unactivated, no bound crash */
    CODE_B_CELL_ACTIVATED,         /**< Recognized crash pattern */
    CODE_B_CELL_PLASMA,            /**< Actively producing antibodies */
    CODE_B_CELL_MEMORY,            /**< Long-term pattern memory */
    CODE_B_CELL_APOPTOTIC          /**< Marked for cleanup */
} code_b_cell_state_t;

/**
 * @brief Code antibody classes (response intensity)
 *
 * BIOLOGICAL BASIS:
 * IgM = first response (quick but unverified)
 * IgG = mature/tested response
 * IgE = emergency override
 */
typedef enum {
    CODE_ANTIBODY_IGM = 0,         /**< First response - untested patch */
    CODE_ANTIBODY_IGG = 1,         /**< Mature - validated patch */
    CODE_ANTIBODY_IGE = 2          /**< Emergency - bypass validation */
} code_antibody_class_t;

/* ============================================================================
 * Core Structures
 * ============================================================================ */

/**
 * @brief Code antigen - crash event presented for immune processing
 *
 * Represents a crash detected by signal handler, normalized
 * for code immune processing. The epitope is a hash of the
 * crash signature for pattern matching.
 */
typedef struct {
    uint64_t id;                              /**< Unique antigen ID */
    char epitope[CODE_IMMUNE_EPITOPE_SIZE];   /**< Hash of crash signature */

    /* Crash details */
    int signal;                               /**< Signal number */
    void* fault_address;                      /**< Address that caused fault */
    void* instruction_pointer;                /**< IP at time of crash */

    /* Source location (if available via debug info) */
    char source_file[256];                    /**< Source file path */
    uint32_t line_number;                     /**< Line number */
    char function_name[128];                  /**< Function that crashed */

    /* Backtrace */
    void* backtrace[CODE_IMMUNE_BACKTRACE_DEPTH]; /**< Stack trace */
    int backtrace_depth;                      /**< Number of frames */

    /* Severity and confidence */
    float severity;                           /**< 0-1 severity score */
    float confidence;                         /**< 0-1 pattern match confidence */
    float danger_signal;                      /**< 0-1 cumulative danger */

    /* State */
    bool processed;                           /**< Processed by B cells */
    bool neutralized;                         /**< Fix applied successfully */
    uint64_t timestamp;                       /**< Detection time */
    uint32_t recurrence_count;                /**< How many times seen */
} code_antigen_t;

/**
 * @brief Code B cell - crash pattern recognition and fix generation
 *
 * B cells learn to recognize crash patterns (via receptor)
 * and can generate fixes (antibodies) when activated.
 */
typedef struct {
    uint64_t id;                              /**< Unique B cell ID */
    char receptor[CODE_IMMUNE_EPITOPE_SIZE];  /**< Pattern to match */
    code_crash_type_t crash_types;            /**< Crash types this cell handles */

    /* Pattern specifics */
    char source_pattern[256];                 /**< Source file/function pattern */
    char fix_template[1024];                  /**< Template for generating fix */

    /* State */
    code_b_cell_state_t state;                /**< Current state */
    float affinity;                           /**< Pattern match strength 0-1 */
    uint64_t bound_antigen_id;                /**< Currently bound antigen */

    /* Success tracking */
    uint32_t successful_fixes;                /**< Fixes that worked */
    uint32_t failed_fixes;                    /**< Fixes that failed */

    /* Timing */
    uint64_t creation_time;                   /**< When created */
    uint64_t last_activation;                 /**< Last activation time */
} code_b_cell_t;

/**
 * @brief Code antibody - runtime patch for crash prevention
 *
 * Antibodies are actual fixes that can be injected at runtime
 * to prevent crash recurrence. They can be hot-loaded .so files
 * or function pointer redirections.
 */
typedef struct {
    uint64_t id;                              /**< Unique antibody ID */
    uint64_t target_antigen_id;               /**< Antigen this fixes */
    uint64_t producer_b_cell_id;              /**< B cell that produced this */

    /* Code patch content */
    char original_code[4096];                 /**< Original problematic code */
    char fixed_code[4096];                    /**< Patched code */

    /* Location */
    char source_file[256];                    /**< File containing bug */
    uint32_t start_line;                      /**< Start line of patch */
    uint32_t end_line;                        /**< End line of patch */

    /* Hot-patch info */
    char patch_so_path[256];                  /**< Path to compiled .so patch */
    void* patch_handle;                       /**< dlopen handle */
    void* new_function;                       /**< Patched function pointer */
    void* old_function;                       /**< Original function pointer */
    char fn_name[128];                        /**< Function being patched */

    /* Classification */
    code_antibody_class_t ab_class;           /**< Antibody class */
    float effectiveness;                      /**< 0-1 how well it works */

    /* State */
    bool validated;                           /**< Passed validation tests */
    bool injected;                            /**< Currently active in runtime */

    /* Timing */
    uint64_t creation_time;                   /**< When created */
    uint64_t injection_time;                  /**< When injected */
    uint32_t prevention_count;                /**< Crashes prevented */
} code_antibody_t;

/* ============================================================================
 * Configuration
 * ============================================================================ */

/**
 * @brief Code immune system configuration
 */
typedef struct {
    /* Population limits */
    size_t max_antigens;                      /**< Max pending crash antigens */
    size_t max_b_cells;                       /**< Max B cells */
    size_t max_antibodies;                    /**< Max antibodies */

    /* Timing (milliseconds) */
    uint64_t activation_delay_ms;             /**< Delay before response */
    uint64_t memory_formation_delay_ms;       /**< Delay for memory formation */
    uint64_t antibody_half_life_ms;           /**< Antibody decay time */

    /* Thresholds */
    float recognition_threshold;              /**< Pattern match threshold 0-1 */
    float activation_threshold;               /**< Activation threshold 0-1 */
    float memory_formation_threshold;         /**< Threshold for memory cell 0-1 */

    /* Patching options */
    bool enable_hot_patching;                 /**< Allow runtime .so injection */
    bool enable_function_redirect;            /**< Allow function ptr redirect */
    bool require_validation;                  /**< Require patch validation */
    const char* patch_build_dir;              /**< Directory for compiled patches */

    /* Integration */
    bool enable_logging;                      /**< Enable security logging */
    bool sync_with_brain_immune;              /**< Sync with parent immune */
} code_immune_config_t;

/**
 * @brief Code immune system statistics
 */
typedef struct {
    /* Cell counts */
    uint32_t active_b_cells;
    uint32_t memory_b_cells;
    uint32_t active_antibodies;

    /* Activity */
    uint64_t crashes_detected;
    uint64_t crashes_neutralized;
    uint64_t patches_generated;
    uint64_t patches_applied;
    uint64_t patches_validated;
    uint64_t patches_failed;

    /* Effectiveness */
    float avg_response_time_ms;
    float prevention_rate;                    /**< % of recurrences prevented */
    float system_health;                      /**< Overall health 0-1 */

    /* By crash type */
    uint64_t sigsegv_count;
    uint64_t sigbus_count;
    uint64_t sigill_count;
    uint64_t sigfpe_count;
    uint64_t sigabrt_count;
} code_immune_stats_t;

/* ============================================================================
 * Callback Types
 * ============================================================================ */

/**
 * @brief Callback for crash antigen detection
 */
typedef void (*code_immune_crash_cb_t)(
    code_immune_system_t* system,
    const code_antigen_t* antigen,
    void* user_data
);

/**
 * @brief Callback for patch application
 */
typedef void (*code_immune_patch_cb_t)(
    code_immune_system_t* system,
    const code_antibody_t* antibody,
    bool success,
    void* user_data
);

/**
 * @brief Callback for memory cell formation
 */
typedef void (*code_immune_memory_cb_t)(
    code_immune_system_t* system,
    const code_b_cell_t* memory_cell,
    void* user_data
);

/* ============================================================================
 * Main System Structure
 * ============================================================================ */

/**
 * @brief Code immune system state
 */
struct code_immune_system {
    code_immune_config_t config;              /**< Configuration */

    /* Parent immune system integration */
    brain_immune_system_t* parent_immune;     /**< Brain immune system */

    /* Antigen pool */
    code_antigen_t* antigens;
    size_t antigen_count;
    size_t antigen_capacity;
    uint64_t next_antigen_id;

    /* B cells */
    code_b_cell_t* b_cells;
    size_t b_cell_count;
    size_t b_cell_capacity;
    uint64_t next_b_cell_id;

    /* Antibodies */
    code_antibody_t* antibodies;
    size_t antibody_count;
    size_t antibody_capacity;
    uint64_t next_antibody_id;

    /* Callbacks */
    code_immune_crash_cb_t on_crash;
    code_immune_patch_cb_t on_patch;
    code_immune_memory_cb_t on_memory;
    void* callback_user_data;

    /* Statistics */
    code_immune_stats_t stats;

    /* Thread safety */
    void* mutex;                              /**< Platform mutex */

    /* State */
    bool running;
    bool signal_handler_connected;
    uint64_t start_time;
};

/* ============================================================================
 * Lifecycle API
 * ============================================================================ */

/**
 * @brief Get default configuration
 *
 * WHAT: Provide sensible default configuration
 * WHY:  Easy initialization with good defaults
 * HOW:  Return struct with balanced parameters
 *
 * @param config Output configuration
 * @return 0 on success, -1 on error
 */
int code_immune_default_config(code_immune_config_t* config);

/**
 * @brief Create code immune system
 *
 * WHAT: Initialize code immune coordination layer
 * WHY:  Set up crash defense infrastructure
 * HOW:  Allocate pools, optionally link to brain immune
 *
 * @param parent_immune Parent brain immune system (can be NULL)
 * @return New code immune system or NULL on failure
 */
code_immune_system_t* code_immune_create(brain_immune_system_t* parent_immune);

/**
 * @brief Create code immune system with config
 *
 * WHAT: Initialize with custom configuration
 * WHY:  Allow fine-grained control
 * HOW:  Use provided config instead of defaults
 *
 * @param parent_immune Parent brain immune system (can be NULL)
 * @param config Custom configuration
 * @return New code immune system or NULL on failure
 */
code_immune_system_t* code_immune_create_with_config(
    brain_immune_system_t* parent_immune,
    const code_immune_config_t* config
);

/**
 * @brief Destroy code immune system
 *
 * WHAT: Clean up code immune system resources
 * WHY:  Proper resource deallocation
 * HOW:  Unload patches, free pools, unregister from signal handler
 *
 * @param system System to destroy
 */
void code_immune_destroy(code_immune_system_t* system);

/**
 * @brief Start code immune system
 *
 * WHAT: Activate crash monitoring
 * WHY:  Begin crash detection and response
 * HOW:  Register with signal handler
 *
 * @param system Code immune system
 * @return 0 on success
 */
int code_immune_start(code_immune_system_t* system);

/**
 * @brief Stop code immune system
 *
 * WHAT: Deactivate code immune system
 * WHY:  Graceful shutdown
 * HOW:  Unregister from signal handler, keep learned patterns
 *
 * @param system Code immune system
 * @return 0 on success
 */
int code_immune_stop(code_immune_system_t* system);

/* ============================================================================
 * Signal Handler Integration API
 * ============================================================================ */

/**
 * @brief Connect to signal handler
 *
 * WHAT: Register with signal handler for crash notification
 * WHY:  Receive crashes as antigens for processing
 * HOW:  Set callback in signal handler
 *
 * @param system Code immune system
 * @return 0 on success
 */
int code_immune_connect_signal_handler(code_immune_system_t* system);

/**
 * @brief Disconnect from signal handler
 *
 * WHAT: Unregister from signal handler
 * WHY:  Stop receiving crash notifications
 * HOW:  Clear callback in signal handler
 *
 * @param system Code immune system
 * @return 0 on success
 */
int code_immune_disconnect_signal_handler(code_immune_system_t* system);

/**
 * @brief Present crash as antigen
 *
 * WHAT: Process crash signal as immune antigen
 * WHY:  Unified crash handling through immune system
 * HOW:  Create antigen from signal context
 *
 * Note: This is called from signal handler context - must be signal-safe
 *
 * @param system Code immune system
 * @param signal Signal number (SIGSEGV, etc)
 * @param ucontext Signal ucontext (can be NULL)
 * @param fault_addr Faulting address (can be NULL)
 * @return 0 on success (antigen queued), -1 on error
 */
int code_immune_present_crash(
    code_immune_system_t* system,
    int signal,
    void* ucontext,
    void* fault_addr
);

/**
 * @brief Present crash with full details
 *
 * WHAT: Present crash with complete diagnostic info
 * WHY:  Better pattern matching with full context
 * HOW:  Create detailed antigen record
 *
 * Note: NOT signal-safe - call from deferred context
 *
 * @param system Code immune system
 * @param signal Signal number
 * @param fault_addr Faulting address
 * @param ip Instruction pointer
 * @param source_file Source file (can be NULL)
 * @param line Line number
 * @param function Function name (can be NULL)
 * @param backtrace Backtrace array
 * @param backtrace_depth Number of frames
 * @param antigen_id Output: assigned antigen ID
 * @return 0 on success
 */
int code_immune_present_crash_detailed(
    code_immune_system_t* system,
    int signal,
    void* fault_addr,
    void* ip,
    const char* source_file,
    uint32_t line,
    const char* function,
    void** backtrace,
    int backtrace_depth,
    uint64_t* antigen_id
);

/* ============================================================================
 * B Cell API
 * ============================================================================ */

/**
 * @brief Find matching B cell for antigen
 *
 * WHAT: Search for B cell that recognizes crash pattern
 * WHY:  Reuse existing pattern knowledge
 * HOW:  Compare epitope against B cell receptors
 *
 * @param system Code immune system
 * @param antigen_id Antigen to match
 * @param b_cell_id Output: matching B cell ID (if found)
 * @return 0 if found, -1 if no match
 */
int code_immune_find_matching_b_cell(
    code_immune_system_t* system,
    uint64_t antigen_id,
    uint64_t* b_cell_id
);

/**
 * @brief Activate B cell for antigen
 *
 * WHAT: Activate B cell that recognizes antigen
 * WHY:  Begin antibody production process
 * HOW:  Set B cell state, bind to antigen
 *
 * @param system Code immune system
 * @param b_cell_id B cell to activate
 * @param antigen_id Triggering antigen
 * @return 0 on success
 */
int code_immune_activate_b_cell(
    code_immune_system_t* system,
    uint64_t b_cell_id,
    uint64_t antigen_id
);

/**
 * @brief Create new B cell for novel crash pattern
 *
 * WHAT: Create B cell for unrecognized crash
 * WHY:  Learn new crash patterns
 * HOW:  Create B cell with receptor matching antigen epitope
 *
 * @param system Code immune system
 * @param antigen_id Antigen to create B cell for
 * @param b_cell_id Output: new B cell ID
 * @return 0 on success
 */
int code_immune_create_b_cell(
    code_immune_system_t* system,
    uint64_t antigen_id,
    uint64_t* b_cell_id
);

/**
 * @brief Form memory B cell
 *
 * WHAT: Convert activated B cell to memory cell
 * WHY:  Persist learned crash patterns for fast future response
 * HOW:  Update B cell state, persist pattern
 *
 * @param system Code immune system
 * @param b_cell_id B cell to convert
 * @return 0 on success
 */
int code_immune_form_memory(
    code_immune_system_t* system,
    uint64_t b_cell_id
);

/**
 * @brief Set fix template for B cell
 *
 * WHAT: Associate fix template with B cell
 * WHY:  Enable antibody generation
 * HOW:  Store template for use in antibody production
 *
 * @param system Code immune system
 * @param b_cell_id B cell ID
 * @param fix_template Fix code template
 * @return 0 on success
 */
int code_immune_set_fix_template(
    code_immune_system_t* system,
    uint64_t b_cell_id,
    const char* fix_template
);

/* ============================================================================
 * Antibody API
 * ============================================================================ */

/**
 * @brief Produce antibody from B cell
 *
 * WHAT: Generate code patch from activated B cell
 * WHY:  Create fix for crash
 * HOW:  Apply fix template to generate patch
 *
 * @param system Code immune system
 * @param b_cell_id Producing B cell
 * @param ab_class Antibody class (IGM=quick, IGG=verified, IGE=emergency)
 * @param antibody_id Output: new antibody ID
 * @return 0 on success
 */
int code_immune_produce_antibody(
    code_immune_system_t* system,
    uint64_t b_cell_id,
    code_antibody_class_t ab_class,
    uint64_t* antibody_id
);

/**
 * @brief Validate antibody
 *
 * WHAT: Test that patch works correctly
 * WHY:  Ensure fix doesn't cause other problems
 * HOW:  Run validation tests on patch
 *
 * @param system Code immune system
 * @param antibody_id Antibody to validate
 * @return 0 if valid, -1 if invalid
 */
int code_immune_validate_antibody(
    code_immune_system_t* system,
    uint64_t antibody_id
);

/**
 * @brief Apply antibody
 *
 * WHAT: Inject patch into running system
 * WHY:  Prevent crash recurrence
 * HOW:  Load .so or redirect function pointer
 *
 * @param system Code immune system
 * @param antibody_id Antibody to apply
 * @return 0 on success
 */
int code_immune_apply_antibody(
    code_immune_system_t* system,
    uint64_t antibody_id
);

/**
 * @brief Remove antibody (apoptosis)
 *
 * WHAT: Remove patch from running system
 * WHY:  Clean up old/superseded patches
 * HOW:  Unload .so, restore original function
 *
 * @param system Code immune system
 * @param antibody_id Antibody to remove
 * @return 0 on success
 */
int code_immune_apoptosis(
    code_immune_system_t* system,
    uint64_t antibody_id
);

/**
 * @brief Upgrade antibody class
 *
 * WHAT: Promote IGM to IGG after validation
 * WHY:  Track antibody maturation
 * HOW:  Update antibody class field
 *
 * @param system Code immune system
 * @param antibody_id Antibody to upgrade
 * @param new_class New antibody class
 * @return 0 on success
 */
int code_immune_upgrade_antibody(
    code_immune_system_t* system,
    uint64_t antibody_id,
    code_antibody_class_t new_class
);

/* ============================================================================
 * Brain Immune Integration API
 * ============================================================================ */

/**
 * @brief Sync crash to brain immune as antigen
 *
 * WHAT: Present code crash to parent brain immune system
 * WHY:  Unified immune response across all subsystems
 * HOW:  Create brain antigen from code crash
 *
 * @param system Code immune system
 * @param antigen_id Code antigen to sync
 * @return 0 on success
 */
int code_immune_sync_to_brain(
    code_immune_system_t* system,
    uint64_t antigen_id
);

/**
 * @brief Request cytokine release from brain immune
 *
 * WHAT: Trigger brain immune cytokine for code crash
 * WHY:  Coordinate system-wide response
 * HOW:  Call brain immune cytokine API
 *
 * @param system Code immune system
 * @param cytokine_type Cytokine type to release
 * @param concentration Signal strength 0-1
 * @return 0 on success
 */
int code_immune_request_cytokine(
    code_immune_system_t* system,
    brain_cytokine_type_t cytokine_type,
    float concentration
);

/* ============================================================================
 * Callback Registration
 * ============================================================================ */

/**
 * @brief Set crash detection callback
 */
int code_immune_set_crash_callback(
    code_immune_system_t* system,
    code_immune_crash_cb_t callback,
    void* user_data
);

/**
 * @brief Set patch application callback
 */
int code_immune_set_patch_callback(
    code_immune_system_t* system,
    code_immune_patch_cb_t callback,
    void* user_data
);

/**
 * @brief Set memory formation callback
 */
int code_immune_set_memory_callback(
    code_immune_system_t* system,
    code_immune_memory_cb_t callback,
    void* user_data
);

/* ============================================================================
 * Update and Query API
 * ============================================================================ */

/**
 * @brief Update code immune system state
 *
 * WHAT: Process pending crashes and decay antibodies
 * WHY:  Advance immune state machine
 * HOW:  Process antigen queue, update statistics
 *
 * @param system Code immune system
 * @param delta_ms Time since last update
 * @return 0 on success
 */
int code_immune_update(
    code_immune_system_t* system,
    uint64_t delta_ms
);

/**
 * @brief Get code immune system statistics
 *
 * @param system Code immune system
 * @param stats Output statistics
 * @return 0 on success
 */
int code_immune_get_stats(
    code_immune_system_t* system,
    code_immune_stats_t* stats
);

/**
 * @brief Get antigen by ID
 *
 * @param system Code immune system
 * @param antigen_id Antigen ID
 * @return Antigen or NULL if not found
 */
const code_antigen_t* code_immune_get_antigen(
    code_immune_system_t* system,
    uint64_t antigen_id
);

/**
 * @brief Get B cell by ID
 *
 * @param system Code immune system
 * @param b_cell_id B cell ID
 * @return B cell or NULL if not found
 */
const code_b_cell_t* code_immune_get_b_cell(
    code_immune_system_t* system,
    uint64_t b_cell_id
);

/**
 * @brief Get antibody by ID
 *
 * @param system Code immune system
 * @param antibody_id Antibody ID
 * @return Antibody or NULL if not found
 */
const code_antibody_t* code_immune_get_antibody(
    code_immune_system_t* system,
    uint64_t antibody_id
);

/**
 * @brief Check if crash type is handled
 *
 * @param system Code immune system
 * @param crash_type Crash type to check
 * @return true if memory exists for this crash type
 */
bool code_immune_has_memory_for(
    code_immune_system_t* system,
    code_crash_type_t crash_type
);

/* ============================================================================
 * String Conversion Utilities
 * ============================================================================ */

const char* code_immune_crash_type_to_string(code_crash_type_t type);
const char* code_immune_b_cell_state_to_string(code_b_cell_state_t state);
const char* code_immune_antibody_class_to_string(code_antibody_class_t ab_class);

/* ============================================================================
 * Epitope/Pattern Matching Utilities
 * ============================================================================ */

/**
 * @brief Compute crash epitope (hash signature)
 *
 * WHAT: Generate unique signature for crash
 * WHY:  Enable pattern matching across crashes
 * HOW:  Hash crash details into epitope string
 *
 * @param signal Signal number
 * @param fault_addr Faulting address
 * @param ip Instruction pointer
 * @param backtrace Backtrace frames
 * @param backtrace_depth Number of frames
 * @param epitope Output buffer (CODE_IMMUNE_EPITOPE_SIZE)
 * @return 0 on success
 */
int code_immune_compute_epitope(
    int signal,
    void* fault_addr,
    void* ip,
    void** backtrace,
    int backtrace_depth,
    char* epitope
);

/**
 * @brief Compute pattern affinity
 *
 * WHAT: Calculate similarity between epitopes
 * WHY:  Determine if B cell matches antigen
 * HOW:  Fuzzy string matching
 *
 * @param pattern1 First pattern
 * @param pattern2 Second pattern
 * @return Affinity score 0-1
 */
float code_immune_compute_affinity(
    const char* pattern1,
    const char* pattern2
);

#ifdef __cplusplus
}
#endif

#endif /* NIMCP_CODE_IMMUNE_H */
