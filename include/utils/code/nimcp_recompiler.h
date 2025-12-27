/**
 * @file nimcp_recompiler.h
 * @brief Dynamic code recompilation pipeline for self-healing modules
 *
 * WHAT: Compile source code into loadable shared objects at runtime
 * WHY: Enable hot-patching and self-healing of faulty code
 * HOW: Invoke GCC, capture output, verify symbols, sandbox test execution
 *
 * ARCHITECTURE:
 *
 *   Source Code                 Recompiler                   Runtime
 *   ┌─────────────┐            ┌──────────────────┐         ┌─────────────┐
 *   │ Patch       │────────────>│ 1. Write temp.c  │         │ dlopen()    │
 *   │ Source      │            │ 2. gcc compile   │────────>│ Load .so    │
 *   └─────────────┘            │ 3. Verify output │         │             │
 *                              │ 4. nm symbols    │         └─────────────┘
 *   ┌─────────────┐            │ 5. dlopen test   │
 *   │ Source      │────────────>│ 6. Sandbox test  │
 *   │ File        │            └──────────────────┘
 *   └─────────────┘                    │
 *                                      ▼
 *                              ┌──────────────────┐
 *                              │ Compiler Output  │
 *                              │ - Errors         │
 *                              │ - Warnings       │
 *                              │ - Exit code      │
 *                              └──────────────────┘
 *
 * FEATURES:
 * - Compile source files to shared objects
 * - Compile inline patch code strings
 * - Capture compiler errors and warnings
 * - Verify output .so exists and is valid
 * - Symbol verification via nm
 * - Test loading via dlopen
 * - Sandboxed test execution
 * - Timeout protection for compilation
 * - Incremental compilation support
 * - Auto-detect NIMCP include paths and CFLAGS
 *
 * SECURITY CONSIDERATIONS:
 * - Sandbox all test executions
 * - Validate source before compilation
 * - Timeout protection prevents hangs
 * - Temporary files in controlled directory
 *
 * USAGE:
 *   recompiler_config_t config = recompiler_default_config();
 *   config.temp_dir = "/tmp/nimcp_patches";
 *   recompiler_t recompiler = recompiler_create(&config);
 *
 *   // Compile a source file
 *   recompile_request_t req = {0};
 *   strncpy(req.source_file, "patched_function.c", sizeof(req.source_file));
 *   strncpy(req.output_so, "patch.so", sizeof(req.output_so));
 *   req.debug_symbols = true;
 *   req.position_independent = true;
 *
 *   recompile_result_t result = {0};
 *   if (recompiler_compile(recompiler, &req, &result)) {
 *       printf("Compiled successfully: %s\n", result.output_path);
 *       if (recompiler_verify_symbol(result.output_path, "patched_func")) {
 *           printf("Symbol verified\n");
 *       }
 *   } else {
 *       printf("Compile failed: %s\n", result.error_msg);
 *   }
 *
 *   recompiler_destroy(recompiler);
 *
 * @author NIMCP Development Team
 * @date 2025
 */

#ifndef NIMCP_RECOMPILER_H
#define NIMCP_RECOMPILER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * EXPORT MACRO
 * ============================================================================ */

#ifndef NIMCP_EXPORT
    #define NIMCP_EXPORT
#endif

/* ============================================================================
 * CONSTANTS
 * ============================================================================ */

/** Maximum source file path length */
#define NIMCP_RECOMPILER_MAX_PATH 512

/** Maximum temp directory path length */
#define NIMCP_RECOMPILER_MAX_TEMP_PATH 256

/** Maximum compiler output buffer size */
#define NIMCP_RECOMPILER_MAX_OUTPUT 4096

/** Maximum error message length */
#define NIMCP_RECOMPILER_MAX_ERROR 1024

/** Maximum number of include paths */
#define NIMCP_RECOMPILER_MAX_INCLUDES 32

/** Maximum number of library paths */
#define NIMCP_RECOMPILER_MAX_LIB_PATHS 16

/** Maximum number of libraries to link */
#define NIMCP_RECOMPILER_MAX_LIBS 32

/** Maximum extra CFLAGS length */
#define NIMCP_RECOMPILER_MAX_CFLAGS 512

/** Maximum symbol name length */
#define NIMCP_RECOMPILER_MAX_SYMBOL 256

/** Default compilation timeout (ms) */
#define NIMCP_RECOMPILER_DEFAULT_TIMEOUT_MS 30000

/** Magic value for recompiler validation */
#define NIMCP_RECOMPILER_MAGIC 0x52434D50  /* 'RCMP' */

/* ============================================================================
 * TYPE DEFINITIONS
 * ============================================================================ */

/**
 * @brief Compile request parameters
 *
 * WHAT: Specification for a single compilation job
 * WHY: Encapsulate all compilation options in one structure
 * HOW: Pass to recompiler_compile() or recompiler_compile_patch()
 */
typedef struct {
    char source_file[NIMCP_RECOMPILER_MAX_PATH];    /**< Source file to compile */
    char output_so[NIMCP_RECOMPILER_MAX_PATH];      /**< Output .so path */
    char temp_dir[NIMCP_RECOMPILER_MAX_TEMP_PATH];  /**< Temp directory for intermediates */

    const char** include_paths;     /**< Array of include directories */
    uint32_t include_count;         /**< Number of include paths */

    const char** library_paths;     /**< Array of library directories */
    uint32_t library_count;         /**< Number of library paths */

    const char** libraries;         /**< Array of libraries to link */
    uint32_t lib_count;             /**< Number of libraries */

    const char* extra_cflags;       /**< Additional CFLAGS string */

    bool debug_symbols;             /**< Include debug symbols (-g) */
    bool position_independent;      /**< Position independent code (-fPIC) */
    bool optimize;                  /**< Enable optimization (-O2) */
    bool warnings_as_errors;        /**< Treat warnings as errors (-Werror) */
    bool verbose;                   /**< Verbose output (-v) */

    uint32_t timeout_ms;            /**< Compilation timeout (0 = default) */
} recompile_request_t;

/**
 * @brief Compile result
 *
 * WHAT: Result of a compilation attempt
 * WHY: Provide feedback on success/failure and diagnostics
 */
typedef struct {
    bool success;                   /**< Compilation succeeded */
    int exit_code;                  /**< Compiler exit code */
    char output[NIMCP_RECOMPILER_MAX_OUTPUT];  /**< stdout/stderr combined */
    char error_msg[NIMCP_RECOMPILER_MAX_ERROR]; /**< Human-readable error */
    char output_path[NIMCP_RECOMPILER_MAX_PATH]; /**< Path to output .so */
    uint64_t compile_time_ms;       /**< Time spent compiling */
    uint32_t warning_count;         /**< Number of warnings detected */
    uint32_t error_count;           /**< Number of errors detected */
} recompile_result_t;

/**
 * @brief Recompiler configuration
 *
 * WHAT: Global configuration for recompiler instance
 * WHY: Set up paths, default options
 */
typedef struct {
    char temp_dir[NIMCP_RECOMPILER_MAX_TEMP_PATH]; /**< Default temp directory */
    char compiler_path[NIMCP_RECOMPILER_MAX_PATH]; /**< Path to gcc (NULL = auto) */
    char nm_path[NIMCP_RECOMPILER_MAX_PATH];       /**< Path to nm (NULL = auto) */
    uint32_t default_timeout_ms;    /**< Default compilation timeout */
    bool auto_cleanup;              /**< Auto-cleanup temp files on destroy */
    bool preserve_intermediates;    /**< Keep intermediate files for debugging */
    bool use_ccache;                /**< Use ccache if available */
} recompiler_config_t;

/**
 * @brief Symbol verification result
 */
typedef struct {
    bool found;                     /**< Symbol was found */
    char symbol_type;               /**< Symbol type (T, U, D, etc.) */
    uint64_t address;               /**< Symbol address (if available) */
    bool is_defined;                /**< Symbol is defined (not undefined) */
    bool is_global;                 /**< Symbol has global binding */
} symbol_info_t;

/**
 * @brief Sandbox test function signature
 *
 * @param so_path Path to loaded shared object
 * @param user_data User-provided context
 * @return 0 on success, non-zero on failure
 */
typedef int (*sandbox_test_fn_t)(const char* so_path, void* user_data);

/**
 * @brief Recompiler opaque handle
 */
typedef struct nimcp_recompiler* recompiler_t;

/* ============================================================================
 * LIFECYCLE FUNCTIONS
 * ============================================================================ */

/**
 * @brief Create recompiler instance
 *
 * WHAT: Initialize recompiler with configuration
 * WHY: Set up compilation infrastructure
 * HOW: Validate paths, create temp directories
 *
 * @param config Configuration (NULL for defaults)
 * @return Recompiler handle or NULL on failure
 *
 * COMPLEXITY: O(1)
 * THREAD-SAFE: Yes (after creation)
 */
NIMCP_EXPORT recompiler_t recompiler_create(const recompiler_config_t* config);

/**
 * @brief Destroy recompiler instance
 *
 * WHAT: Free all recompiler resources
 * WHY: Clean shutdown
 * HOW: Cleanup temp files (if configured), free memory
 *
 * @param recompiler Recompiler handle
 *
 * COMPLEXITY: O(n) where n = temp files
 * THREAD-SAFE: No (must not have active compilations)
 */
NIMCP_EXPORT void recompiler_destroy(recompiler_t recompiler);

/**
 * @brief Get default configuration
 *
 * @return Default configuration with sensible values
 */
NIMCP_EXPORT recompiler_config_t recompiler_default_config(void);

/* ============================================================================
 * COMPILATION FUNCTIONS
 * ============================================================================ */

/**
 * @brief Compile source file to shared object
 *
 * WHAT: Compile a source file to loadable .so
 * WHY: Create hot-patchable code
 * HOW: Invoke gcc with specified options
 *
 * @param recompiler Recompiler handle
 * @param request Compilation request
 * @param result Output result (must not be NULL)
 * @return true on success
 *
 * COMPLEXITY: O(source_size) + compile time
 * THREAD-SAFE: Yes (can run concurrent compilations)
 *
 * EXAMPLE:
 * ```c
 * recompile_request_t req = {0};
 * strncpy(req.source_file, "patch.c", sizeof(req.source_file));
 * strncpy(req.output_so, "/tmp/patch.so", sizeof(req.output_so));
 * req.position_independent = true;
 * req.debug_symbols = true;
 *
 * recompile_result_t result = {0};
 * if (recompiler_compile(recompiler, &req, &result)) {
 *     printf("Success: %s\n", result.output_path);
 * }
 * ```
 */
NIMCP_EXPORT bool recompiler_compile(
    recompiler_t recompiler,
    const recompile_request_t* request,
    recompile_result_t* result
);

/**
 * @brief Compile patch source code string to shared object
 *
 * WHAT: Compile inline source code to loadable .so
 * WHY: Allow patching without separate source file
 * HOW: Write to temp file, compile, return result
 *
 * @param recompiler Recompiler handle
 * @param source_code Source code string
 * @param fn_name Primary function name (for output naming)
 * @param result Output result
 * @return true on success
 *
 * COMPLEXITY: O(source_len) + compile time
 * THREAD-SAFE: Yes
 *
 * EXAMPLE:
 * ```c
 * const char* patch = "int patched_func(int x) { return x * 2; }";
 * recompile_result_t result = {0};
 * if (recompiler_compile_patch(recompiler, patch, "patched_func", &result)) {
 *     // Load result.output_path via dlopen
 * }
 * ```
 */
NIMCP_EXPORT bool recompiler_compile_patch(
    recompiler_t recompiler,
    const char* source_code,
    const char* fn_name,
    recompile_result_t* result
);

/**
 * @brief Compile with incremental compilation
 *
 * WHAT: Compile only if source newer than output
 * WHY: Skip unnecessary recompilation
 * HOW: Check mtime, skip if output is newer
 *
 * @param recompiler Recompiler handle
 * @param request Compilation request
 * @param result Output result
 * @param was_compiled Set to true if compilation was performed
 * @return true on success (or if skipped)
 *
 * COMPLEXITY: O(1) stat + optional compile
 * THREAD-SAFE: Yes
 */
NIMCP_EXPORT bool recompiler_compile_incremental(
    recompiler_t recompiler,
    const recompile_request_t* request,
    recompile_result_t* result,
    bool* was_compiled
);

/* ============================================================================
 * VERIFICATION FUNCTIONS
 * ============================================================================ */

/**
 * @brief Verify symbol exists in shared object
 *
 * WHAT: Check if symbol is defined in .so file
 * WHY: Ensure patched function is present
 * HOW: Run nm -D on .so file, parse output
 *
 * @param so_path Path to shared object
 * @param symbol_name Symbol to look for
 * @return true if symbol exists
 *
 * COMPLEXITY: O(1) + nm execution
 * THREAD-SAFE: Yes
 */
NIMCP_EXPORT bool recompiler_verify_symbol(
    const char* so_path,
    const char* symbol_name
);

/**
 * @brief Get detailed symbol information
 *
 * WHAT: Get full details about a symbol
 * WHY: Determine symbol type, binding, address
 * HOW: Run nm with verbose output, parse
 *
 * @param so_path Path to shared object
 * @param symbol_name Symbol to look for
 * @param info Output symbol information
 * @return true if symbol found
 */
NIMCP_EXPORT bool recompiler_get_symbol_info(
    const char* so_path,
    const char* symbol_name,
    symbol_info_t* info
);

/**
 * @brief Test load shared object with dlopen
 *
 * WHAT: Verify .so can be loaded
 * WHY: Catch link errors before production use
 * HOW: dlopen with RTLD_NOW, close immediately
 *
 * @param so_path Path to shared object
 * @return true if loadable
 *
 * COMPLEXITY: O(1) + dlopen time
 * THREAD-SAFE: Yes
 */
NIMCP_EXPORT bool recompiler_test_load(const char* so_path);

/**
 * @brief Get load error message
 *
 * WHAT: Get last dlopen/dlerror message
 * WHY: Diagnose load failures
 * HOW: Return dlerror() result
 *
 * @return Error message or NULL
 *
 * @note Thread-local, not safe across threads
 */
NIMCP_EXPORT const char* recompiler_get_load_error(void);

/* ============================================================================
 * SANDBOX TEST FUNCTIONS
 * ============================================================================ */

/**
 * @brief Run test in sandbox before production use
 *
 * WHAT: Execute test function in isolated context
 * WHY: Validate patch works correctly before activation
 * HOW: Fork child process, run test, collect result
 *
 * @param so_path Path to shared object
 * @param test_fn Test function to run
 * @param user_data User context for test function
 * @param timeout_ms Test timeout (0 = default)
 * @return Test function return value, or -1 on sandbox failure
 *
 * COMPLEXITY: O(1) + test execution time
 * THREAD-SAFE: Yes
 *
 * SANDBOX FEATURES:
 * - Runs in child process (fork)
 * - Timeout protection
 * - Signal isolation
 * - Exit code collection
 *
 * EXAMPLE:
 * ```c
 * int my_test(const char* so_path, void* user_data) {
 *     void* handle = dlopen(so_path, RTLD_NOW);
 *     if (!handle) return -1;
 *     int (*func)(int) = dlsym(handle, "patched_func");
 *     int result = func(5);
 *     dlclose(handle);
 *     return (result == 10) ? 0 : -1;  // Expect 5*2 = 10
 * }
 *
 * int status = recompiler_sandbox_test(
 *     result.output_path,
 *     my_test,
 *     NULL,
 *     5000  // 5 second timeout
 * );
 * ```
 */
NIMCP_EXPORT int recompiler_sandbox_test(
    const char* so_path,
    sandbox_test_fn_t test_fn,
    void* user_data,
    uint32_t timeout_ms
);

/* ============================================================================
 * NIMCP INTEGRATION FUNCTIONS
 * ============================================================================ */

/**
 * @brief Get CFLAGS for NIMCP compilation
 *
 * WHAT: Get compiler flags used to build NIMCP
 * WHY: Ensure patches use compatible flags
 * HOW: Return compiled-in flags or detect from CMake
 *
 * @return CFLAGS string (static, do not free)
 */
NIMCP_EXPORT const char* recompiler_get_nimcp_cflags(void);

/**
 * @brief Get NIMCP include paths
 *
 * WHAT: Get include directories for NIMCP headers
 * WHY: Patches need access to NIMCP types/functions
 * HOW: Return array of include paths
 *
 * @param count Output number of paths
 * @return Array of path strings (static, do not free)
 */
NIMCP_EXPORT const char** recompiler_get_nimcp_includes(uint32_t* count);

/**
 * @brief Get NIMCP library paths
 *
 * WHAT: Get library directories for linking
 * WHY: Patches may need to link against NIMCP
 * HOW: Return array of library paths
 *
 * @param count Output number of paths
 * @return Array of path strings (static, do not free)
 */
NIMCP_EXPORT const char** recompiler_get_nimcp_lib_paths(uint32_t* count);

/**
 * @brief Populate request with NIMCP defaults
 *
 * WHAT: Fill request with NIMCP-compatible options
 * WHY: Convenience for common use case
 * HOW: Set include paths, cflags, etc.
 *
 * @param request Request to populate
 *
 * @note Does not overwrite source_file, output_so
 */
NIMCP_EXPORT void recompiler_set_nimcp_defaults(recompile_request_t* request);

/* ============================================================================
 * CLEANUP FUNCTIONS
 * ============================================================================ */

/**
 * @brief Clean up temporary files
 *
 * WHAT: Remove temp files created during compilation
 * WHY: Free disk space, security
 * HOW: Delete files in temp directory
 *
 * @param recompiler Recompiler handle
 * @return Number of files removed
 *
 * COMPLEXITY: O(n) where n = temp files
 * THREAD-SAFE: No (should not have active compilations)
 */
NIMCP_EXPORT uint32_t recompiler_cleanup_temp(recompiler_t recompiler);

/**
 * @brief Remove specific output file
 *
 * WHAT: Delete a compiled .so file
 * WHY: Cleanup after patch is no longer needed
 * HOW: unlink file
 *
 * @param so_path Path to shared object
 * @return true if removed (or didn't exist)
 */
NIMCP_EXPORT bool recompiler_remove_output(const char* so_path);

/* ============================================================================
 * STATISTICS
 * ============================================================================ */

/**
 * @brief Recompiler statistics
 */
typedef struct {
    uint64_t compilations_total;        /**< Total compilation attempts */
    uint64_t compilations_success;      /**< Successful compilations */
    uint64_t compilations_failed;       /**< Failed compilations */
    uint64_t total_compile_time_ms;     /**< Total compile time */
    uint64_t avg_compile_time_ms;       /**< Average compile time */
    uint64_t symbols_verified;          /**< Symbol verifications */
    uint64_t load_tests;                /**< Load tests performed */
    uint64_t sandbox_tests;             /**< Sandbox tests performed */
    uint64_t temp_files_created;        /**< Temp files created */
    uint64_t temp_files_cleaned;        /**< Temp files cleaned up */
} recompiler_stats_t;

/**
 * @brief Get recompiler statistics
 *
 * @param recompiler Recompiler handle
 * @param stats Output statistics structure
 * @return true on success
 */
NIMCP_EXPORT bool recompiler_get_stats(
    recompiler_t recompiler,
    recompiler_stats_t* stats
);

/**
 * @brief Reset statistics
 *
 * @param recompiler Recompiler handle
 */
NIMCP_EXPORT void recompiler_reset_stats(recompiler_t recompiler);

#ifdef __cplusplus
}
#endif

#endif /* NIMCP_RECOMPILER_H */
