/**
 * @file nimcp_recompiler.c
 * @brief Dynamic code recompilation pipeline implementation
 *
 * WHAT: Compile source code into loadable shared objects at runtime
 * WHY: Enable hot-patching and self-healing of faulty code
 * HOW: Invoke GCC, capture output, verify symbols, sandbox test execution
 *
 * IMPLEMENTATION NOTES:
 * - Uses fork/exec for gcc invocation
 * - Pipes capture stdout/stderr
 * - Timeout via alarm/SIGALRM
 * - nm for symbol verification
 * - dlopen for load testing
 * - Fork-based sandbox for test isolation
 *
 * @author NIMCP Development Team
 * @date 2025
 */

#include "utils/code/nimcp_recompiler.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/time/nimcp_time.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/platform/nimcp_platform_mutex.h"

#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

/* ============================================================================
 * INTERNAL STRUCTURES
 * ============================================================================ */

/**
 * @brief Internal recompiler state
 */
struct nimcp_recompiler {
    uint32_t magic;                 /**< Validation magic */
    recompiler_config_t config;     /**< Configuration */
    nimcp_platform_mutex_t mutex;   /**< Thread safety */
    recompiler_stats_t stats;       /**< Statistics */
    uint32_t temp_counter;          /**< Unique temp file counter */
    char last_error[NIMCP_RECOMPILER_MAX_ERROR]; /**< Last error message */
};

/* ============================================================================
 * STATIC DATA
 * ============================================================================ */

/**
 * @brief Default NIMCP include paths
 *
 * WHAT: Include directories for NIMCP headers
 * WHY: Patches need to find nimcp headers
 * HOW: Compiled-in paths based on build configuration
 */
static const char* s_nimcp_includes[] = {
    "/home/bbrelin/nimcp/include",
    NULL
};

/**
 * @brief Default NIMCP library paths
 */
static const char* s_nimcp_lib_paths[] = {
    "/home/bbrelin/nimcp/build/src/lib",
    NULL
};

/**
 * @brief Default NIMCP CFLAGS
 *
 * WHAT: Compiler flags for NIMCP compatibility
 * WHY: Ensure patches compile with same settings
 */
static const char* s_nimcp_cflags =
    "-std=c11 -Wall -Wextra -D_GNU_SOURCE -D_POSIX_C_SOURCE=200809L";

/* ============================================================================
 * FORWARD DECLARATIONS
 * ============================================================================ */

static bool validate_recompiler(recompiler_t recompiler);
static bool ensure_temp_dir(const char* path);
static char* generate_temp_path(recompiler_t recompiler, const char* suffix);
static bool write_temp_source(const char* path, const char* source);
static bool build_gcc_command(
    recompiler_t recompiler,
    const recompile_request_t* request,
    char* cmd_buffer,
    size_t buffer_size
);
static bool execute_compiler(
    const char* command,
    uint32_t timeout_ms,
    char* output,
    size_t output_size,
    int* exit_code
);
static bool verify_output_exists(const char* path);
static void parse_compiler_output(
    const char* output,
    uint32_t* warning_count,
    uint32_t* error_count
);
static bool run_nm_check(const char* so_path, const char* symbol_name);

/* ============================================================================
 * LIFECYCLE FUNCTIONS
 * ============================================================================ */

recompiler_config_t recompiler_default_config(void)
{
    LOG_DEBUG("Entering recompiler_default_config");

    recompiler_config_t config = {0};

    /**
     * WHAT: Set default temp directory
     * WHY: Need a place for intermediate files
     * HOW: Use /tmp with nimcp prefix
     */
    strncpy(config.temp_dir, "/tmp/nimcp_recompiler",
            sizeof(config.temp_dir) - 1);

    /**
     * WHAT: Set default compiler paths
     * WHY: Allow auto-detection
     * HOW: NULL means use PATH search
     */
    config.compiler_path[0] = '\0';  /* Auto-detect gcc */
    config.nm_path[0] = '\0';        /* Auto-detect nm */

    config.default_timeout_ms = NIMCP_RECOMPILER_DEFAULT_TIMEOUT_MS;
    config.auto_cleanup = true;
    config.preserve_intermediates = false;
    config.use_ccache = false;

    return config;
}

recompiler_t recompiler_create(const recompiler_config_t* config)
{
    LOG_DEBUG("Entering recompiler_create");

    /**
     * WHAT: Allocate recompiler structure
     * WHY: Need persistent state
     * HOW: nimcp_calloc for zero-initialization
     */
    struct nimcp_recompiler* recompiler = nimcp_calloc(1, sizeof(*recompiler));
    if (!recompiler) {
        LOG_ERROR("Failed to allocate recompiler");
        return NULL;
    }

    /**
     * WHAT: Initialize with config or defaults
     * WHY: Allow customization
     */
    if (config) {
        recompiler->config = *config;
    } else {
        recompiler->config = recompiler_default_config();
    }

    /**
     * WHAT: Ensure temp directory exists
     * WHY: Need place for temp files
     * HOW: mkdir -p equivalent
     */
    if (!ensure_temp_dir(recompiler->config.temp_dir)) {
        LOG_ERROR("Failed to create temp directory: %s", recompiler->config.temp_dir);
        nimcp_free(recompiler);
        return NULL;
    }

    /**
     * WHAT: Initialize mutex for thread safety
     * WHY: Support concurrent compilations
     */
    if (nimcp_platform_mutex_init(&recompiler->mutex, false) != 0) {
        LOG_ERROR("Failed to initialize mutex");
        nimcp_free(recompiler);
        return NULL;
    }

    /**
     * WHAT: Set magic value
     * WHY: Validate handle in subsequent calls
     */
    recompiler->magic = NIMCP_RECOMPILER_MAGIC;

    LOG_INFO("Recompiler created with temp_dir: %s", recompiler->config.temp_dir);
    return recompiler;
}

void recompiler_destroy(recompiler_t recompiler)
{
    LOG_DEBUG("Entering recompiler_destroy");

    if (!validate_recompiler(recompiler)) {
        return;
    }

    /**
     * WHAT: Cleanup temp files if configured
     * WHY: Free disk space
     */
    if (recompiler->config.auto_cleanup) {
        recompiler_cleanup_temp(recompiler);
    }

    /**
     * WHAT: Destroy mutex
     * WHY: Clean resource release
     */
    nimcp_platform_mutex_destroy(&recompiler->mutex);

    /**
     * WHAT: Invalidate magic
     * WHY: Catch use-after-free
     */
    recompiler->magic = 0;

    nimcp_free(recompiler);
    LOG_DEBUG("Recompiler destroyed");
}

/* ============================================================================
 * COMPILATION FUNCTIONS
 * ============================================================================ */

bool recompiler_compile(
    recompiler_t recompiler,
    const recompile_request_t* request,
    recompile_result_t* result)
{
    LOG_DEBUG("Entering recompiler_compile");

    /**
     * WHAT: Validate inputs
     * WHY: Guard against NULL pointers
     */
    if (!validate_recompiler(recompiler)) {
        if (result) {
            result->success = false;
            strncpy(result->error_msg, "Invalid recompiler handle",
                    sizeof(result->error_msg) - 1);
        }
        return false;
    }

    if (!request || !result) {
        LOG_ERROR("NULL request or result");
        return false;
    }

    /**
     * WHAT: Clear result structure
     * WHY: Start fresh
     */
    memset(result, 0, sizeof(*result));

    /**
     * WHAT: Validate source file exists
     * WHY: Early fail before invoking compiler
     */
    struct stat st;
    if (stat(request->source_file, &st) != 0) {
        snprintf(result->error_msg, sizeof(result->error_msg),
                 "Source file not found: %s", request->source_file);
        LOG_ERROR("%s", result->error_msg);
        return false;
    }

    /**
     * WHAT: Build gcc command line
     * WHY: Construct full compilation command
     */
    char cmd[4096];
    if (!build_gcc_command(recompiler, request, cmd, sizeof(cmd))) {
        strncpy(result->error_msg, "Failed to build gcc command",
                sizeof(result->error_msg) - 1);
        LOG_ERROR("%s", result->error_msg);
        return false;
    }

    LOG_DEBUG("Compiler command: %s", cmd);

    /**
     * WHAT: Execute compiler with timeout
     * WHY: Prevent hanging on infinite loops
     */
    uint64_t start_time = nimcp_time_monotonic_ms();

    uint32_t timeout = request->timeout_ms > 0 ?
                       request->timeout_ms :
                       recompiler->config.default_timeout_ms;

    if (!execute_compiler(cmd, timeout, result->output,
                          sizeof(result->output), &result->exit_code)) {
        strncpy(result->error_msg, "Compiler execution failed",
                sizeof(result->error_msg) - 1);
        LOG_ERROR("%s", result->error_msg);

        nimcp_platform_mutex_lock(&recompiler->mutex);
        recompiler->stats.compilations_failed++;
        recompiler->stats.compilations_total++;
        nimcp_platform_mutex_unlock(&recompiler->mutex);

        return false;
    }

    result->compile_time_ms = nimcp_time_monotonic_ms() - start_time;

    /**
     * WHAT: Parse compiler output for warnings/errors
     * WHY: Provide structured diagnostics
     */
    parse_compiler_output(result->output, &result->warning_count,
                          &result->error_count);

    /**
     * WHAT: Check compiler exit code
     * WHY: Determine success
     */
    if (result->exit_code != 0) {
        snprintf(result->error_msg, sizeof(result->error_msg),
                 "Compiler exited with code %d", result->exit_code);
        LOG_ERROR("%s: %s", result->error_msg, result->output);

        nimcp_platform_mutex_lock(&recompiler->mutex);
        recompiler->stats.compilations_failed++;
        recompiler->stats.compilations_total++;
        recompiler->stats.total_compile_time_ms += result->compile_time_ms;
        nimcp_platform_mutex_unlock(&recompiler->mutex);

        return false;
    }

    /**
     * WHAT: Verify output file was created
     * WHY: Compiler might exit 0 but not produce output
     */
    if (!verify_output_exists(request->output_so)) {
        snprintf(result->error_msg, sizeof(result->error_msg),
                 "Output file not created: %s", request->output_so);
        LOG_ERROR("%s", result->error_msg);

        nimcp_platform_mutex_lock(&recompiler->mutex);
        recompiler->stats.compilations_failed++;
        recompiler->stats.compilations_total++;
        nimcp_platform_mutex_unlock(&recompiler->mutex);

        return false;
    }

    /**
     * WHAT: Copy output path to result
     * WHY: Caller needs to know where output is
     */
    strncpy(result->output_path, request->output_so,
            sizeof(result->output_path) - 1);

    result->success = true;

    /**
     * WHAT: Update statistics
     * WHY: Track compilation performance
     */
    nimcp_platform_mutex_lock(&recompiler->mutex);
    recompiler->stats.compilations_success++;
    recompiler->stats.compilations_total++;
    recompiler->stats.total_compile_time_ms += result->compile_time_ms;
    if (recompiler->stats.compilations_success > 0) {
        recompiler->stats.avg_compile_time_ms =
            recompiler->stats.total_compile_time_ms /
            recompiler->stats.compilations_success;
    }
    nimcp_platform_mutex_unlock(&recompiler->mutex);

    LOG_INFO("Compiled successfully: %s (%lu ms)",
             result->output_path, (unsigned long)result->compile_time_ms);

    return true;
}

bool recompiler_compile_patch(
    recompiler_t recompiler,
    const char* source_code,
    const char* fn_name,
    recompile_result_t* result)
{
    LOG_DEBUG("Entering recompiler_compile_patch");

    if (!validate_recompiler(recompiler)) {
        if (result) {
            result->success = false;
            strncpy(result->error_msg, "Invalid recompiler handle",
                    sizeof(result->error_msg) - 1);
        }
        return false;
    }

    if (!source_code || !fn_name || !result) {
        LOG_ERROR("NULL source_code, fn_name, or result");
        return false;
    }

    memset(result, 0, sizeof(*result));

    /**
     * WHAT: Generate temp source file path
     * WHY: Need to write source to file for gcc
     */
    char* temp_source = generate_temp_path(recompiler, ".c");
    if (!temp_source) {
        strncpy(result->error_msg, "Failed to generate temp path",
                sizeof(result->error_msg) - 1);
        return false;
    }

    /**
     * WHAT: Write source code to temp file
     * WHY: gcc needs file input
     */
    if (!write_temp_source(temp_source, source_code)) {
        snprintf(result->error_msg, sizeof(result->error_msg),
                 "Failed to write temp source: %s", temp_source);
        nimcp_free(temp_source);
        return false;
    }

    nimcp_platform_mutex_lock(&recompiler->mutex);
    recompiler->stats.temp_files_created++;
    nimcp_platform_mutex_unlock(&recompiler->mutex);

    /**
     * WHAT: Generate output .so path
     * WHY: Need a destination for compiled code
     */
    char output_so[NIMCP_RECOMPILER_MAX_PATH];
    snprintf(output_so, sizeof(output_so), "%s/%s_%u.so",
             recompiler->config.temp_dir, fn_name, recompiler->temp_counter);

    /**
     * WHAT: Build and execute compilation request
     * WHY: Use main compile function
     */
    recompile_request_t request = {0};
    strncpy(request.source_file, temp_source, sizeof(request.source_file) - 1);
    strncpy(request.output_so, output_so, sizeof(request.output_so) - 1);
    strncpy(request.temp_dir, recompiler->config.temp_dir,
            sizeof(request.temp_dir) - 1);

    /* Set defaults for patch compilation */
    request.position_independent = true;
    request.debug_symbols = true;
    request.optimize = false;  /* Don't optimize patches for easier debugging */

    recompiler_set_nimcp_defaults(&request);

    bool success = recompiler_compile(recompiler, &request, result);

    /**
     * WHAT: Cleanup temp source if configured
     * WHY: Don't leave source lying around
     */
    if (!recompiler->config.preserve_intermediates) {
        unlink(temp_source);
        nimcp_platform_mutex_lock(&recompiler->mutex);
        recompiler->stats.temp_files_cleaned++;
        nimcp_platform_mutex_unlock(&recompiler->mutex);
    }

    nimcp_free(temp_source);
    return success;
}

bool recompiler_compile_incremental(
    recompiler_t recompiler,
    const recompile_request_t* request,
    recompile_result_t* result,
    bool* was_compiled)
{
    LOG_DEBUG("Entering recompiler_compile_incremental");

    if (!validate_recompiler(recompiler) || !request || !result) {
        return false;
    }

    if (was_compiled) {
        *was_compiled = false;
    }

    /**
     * WHAT: Check if output exists and is newer than source
     * WHY: Skip compilation if not needed
     */
    struct stat src_stat, out_stat;

    if (stat(request->source_file, &src_stat) != 0) {
        snprintf(result->error_msg, sizeof(result->error_msg),
                 "Source file not found: %s", request->source_file);
        return false;
    }

    if (stat(request->output_so, &out_stat) == 0) {
        /* Output exists - check if it's newer */
        if (out_stat.st_mtime >= src_stat.st_mtime) {
            /* Output is up to date */
            LOG_DEBUG("Output is up to date, skipping compilation");
            result->success = true;
            strncpy(result->output_path, request->output_so,
                    sizeof(result->output_path) - 1);
            return true;
        }
    }

    /**
     * WHAT: Compile because output is stale or missing
     * WHY: Need fresh output
     */
    bool success = recompiler_compile(recompiler, request, result);
    if (was_compiled && success) {
        *was_compiled = true;
    }

    return success;
}

/* ============================================================================
 * VERIFICATION FUNCTIONS
 * ============================================================================ */

bool recompiler_verify_symbol(const char* so_path, const char* symbol_name)
{
    LOG_DEBUG("Entering recompiler_verify_symbol: %s in %s", symbol_name, so_path);

    if (!so_path || !symbol_name) {
        return false;
    }

    return run_nm_check(so_path, symbol_name);
}

bool recompiler_get_symbol_info(
    const char* so_path,
    const char* symbol_name,
    symbol_info_t* info)
{
    LOG_DEBUG("Entering recompiler_get_symbol_info");

    if (!so_path || !symbol_name || !info) {
        return false;
    }

    memset(info, 0, sizeof(*info));

    /**
     * WHAT: Run nm -D with output parsing
     * WHY: Get detailed symbol information
     */
    char cmd[1024];
    snprintf(cmd, sizeof(cmd), "nm -D '%s' 2>/dev/null | grep ' %s$'",
             so_path, symbol_name);

    FILE* fp = popen(cmd, "r");
    if (!fp) {
        return false;
    }

    char line[256];
    if (fgets(line, sizeof(line), fp)) {
        info->found = true;

        /**
         * WHAT: Parse nm output format: "address type symbol"
         * WHY: Extract symbol metadata
         */
        uint64_t addr;
        char type;
        char name[256];

        if (sscanf(line, "%lx %c %255s", &addr, &type, name) == 3) {
            info->address = addr;
            info->symbol_type = type;
            info->is_defined = (type != 'U');
            info->is_global = (type == 'T' || type == 'D' || type == 'B' ||
                               type == 'R' || type == 'G');
        } else if (sscanf(line, " %c %255s", &type, name) == 2) {
            /* No address (undefined symbol) */
            info->symbol_type = type;
            info->is_defined = false;
            info->is_global = false;
        }
    }

    pclose(fp);
    return info->found;
}

bool recompiler_test_load(const char* so_path)
{
    LOG_DEBUG("Entering recompiler_test_load: %s", so_path);

    if (!so_path) {
        return false;
    }

    /**
     * WHAT: Attempt to load with RTLD_NOW
     * WHY: Force resolution of all symbols
     * HOW: dlopen then immediately dlclose
     */
    void* handle = dlopen(so_path, RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        LOG_ERROR("dlopen failed: %s", dlerror());
        return false;
    }

    dlclose(handle);
    LOG_DEBUG("Test load successful: %s", so_path);
    return true;
}

const char* recompiler_get_load_error(void)
{
    return dlerror();
}

/* ============================================================================
 * SANDBOX TEST FUNCTIONS
 * ============================================================================ */

/**
 * @brief Volatile flag for timeout handling
 */
static volatile sig_atomic_t s_sandbox_timeout = 0;

/**
 * @brief SIGALRM handler for sandbox timeout
 */
static void sandbox_alarm_handler(int sig)
{
    (void)sig;
    s_sandbox_timeout = 1;
}

int recompiler_sandbox_test(
    const char* so_path,
    sandbox_test_fn_t test_fn,
    void* user_data,
    uint32_t timeout_ms)
{
    LOG_DEBUG("Entering recompiler_sandbox_test");

    if (!so_path || !test_fn) {
        return -1;
    }

    /**
     * WHAT: Fork child process for isolation
     * WHY: Test execution should not affect parent
     */
    pid_t pid = fork();
    if (pid < 0) {
        LOG_ERROR("fork failed: %s", strerror(errno));
        return -1;
    }

    if (pid == 0) {
        /**
         * WHAT: Child process - run test
         * WHY: Isolated execution environment
         */

        /* Set up timeout if specified */
        if (timeout_ms > 0) {
            s_sandbox_timeout = 0;
            signal(SIGALRM, sandbox_alarm_handler);
            alarm((timeout_ms + 999) / 1000);  /* Convert to seconds, round up */
        }

        /* Execute test function */
        int result = test_fn(so_path, user_data);

        /* Cancel alarm */
        alarm(0);

        /* Exit with test result */
        _exit(result);
    }

    /**
     * WHAT: Parent process - wait for child
     * WHY: Collect result
     */
    int status;
    pid_t waited = waitpid(pid, &status, 0);

    if (waited < 0) {
        LOG_ERROR("waitpid failed: %s", strerror(errno));
        return -1;
    }

    /**
     * WHAT: Interpret child status
     * WHY: Return appropriate result
     */
    if (WIFEXITED(status)) {
        int exit_code = WEXITSTATUS(status);
        LOG_DEBUG("Sandbox test exited with code: %d", exit_code);
        return exit_code;
    }

    if (WIFSIGNALED(status)) {
        int sig = WTERMSIG(status);
        LOG_ERROR("Sandbox test killed by signal: %d", sig);
        return -1;
    }

    return -1;
}

/* ============================================================================
 * NIMCP INTEGRATION FUNCTIONS
 * ============================================================================ */

const char* recompiler_get_nimcp_cflags(void)
{
    return s_nimcp_cflags;
}

const char** recompiler_get_nimcp_includes(uint32_t* count)
{
    if (count) {
        /* Count non-NULL entries */
        uint32_t n = 0;
        while (s_nimcp_includes[n]) {
            n++;
        }
        *count = n;
    }
    return s_nimcp_includes;
}

const char** recompiler_get_nimcp_lib_paths(uint32_t* count)
{
    if (count) {
        uint32_t n = 0;
        while (s_nimcp_lib_paths[n]) {
            n++;
        }
        *count = n;
    }
    return s_nimcp_lib_paths;
}

void recompiler_set_nimcp_defaults(recompile_request_t* request)
{
    if (!request) {
        return;
    }

    /**
     * WHAT: Set include paths to NIMCP defaults
     * WHY: Patches need NIMCP headers
     */
    uint32_t include_count;
    request->include_paths = recompiler_get_nimcp_includes(&include_count);
    request->include_count = include_count;

    /**
     * WHAT: Set library paths
     * WHY: May need to link against NIMCP
     */
    uint32_t lib_path_count;
    request->library_paths = recompiler_get_nimcp_lib_paths(&lib_path_count);
    request->library_count = lib_path_count;

    /**
     * WHAT: Set extra CFLAGS
     * WHY: Match NIMCP compile settings
     */
    request->extra_cflags = recompiler_get_nimcp_cflags();

    /**
     * WHAT: Enable required flags
     * WHY: Shared objects need -fPIC
     */
    request->position_independent = true;
}

/* ============================================================================
 * CLEANUP FUNCTIONS
 * ============================================================================ */

uint32_t recompiler_cleanup_temp(recompiler_t recompiler)
{
    LOG_DEBUG("Entering recompiler_cleanup_temp");

    if (!validate_recompiler(recompiler)) {
        return 0;
    }

    /**
     * WHAT: Clean temp directory
     * WHY: Remove compiled artifacts and intermediates
     * HOW: Use shell for glob pattern matching
     */
    char cmd[1024];
    snprintf(cmd, sizeof(cmd), "rm -f '%s'/*.c '%s'/*.so '%s'/*.o 2>/dev/null",
             recompiler->config.temp_dir,
             recompiler->config.temp_dir,
             recompiler->config.temp_dir);

    int ret = system(cmd);
    (void)ret;  /* Ignore return value - files may not exist */

    uint32_t cleaned = 0;  /* Can't easily count deleted files */

    nimcp_platform_mutex_lock(&recompiler->mutex);
    recompiler->stats.temp_files_cleaned += cleaned;
    nimcp_platform_mutex_unlock(&recompiler->mutex);

    LOG_DEBUG("Cleaned temp directory: %s", recompiler->config.temp_dir);
    return cleaned;
}

bool recompiler_remove_output(const char* so_path)
{
    if (!so_path) {
        return false;
    }

    if (unlink(so_path) == 0) {
        return true;
    }

    /* File didn't exist is not an error */
    if (errno == ENOENT) {
        return true;
    }

    LOG_ERROR("Failed to remove %s: %s", so_path, strerror(errno));
    return false;
}

/* ============================================================================
 * STATISTICS FUNCTIONS
 * ============================================================================ */

bool recompiler_get_stats(recompiler_t recompiler, recompiler_stats_t* stats)
{
    if (!validate_recompiler(recompiler) || !stats) {
        return false;
    }

    nimcp_platform_mutex_lock(&recompiler->mutex);
    *stats = recompiler->stats;
    nimcp_platform_mutex_unlock(&recompiler->mutex);

    return true;
}

void recompiler_reset_stats(recompiler_t recompiler)
{
    if (!validate_recompiler(recompiler)) {
        return;
    }

    nimcp_platform_mutex_lock(&recompiler->mutex);
    memset(&recompiler->stats, 0, sizeof(recompiler->stats));
    nimcp_platform_mutex_unlock(&recompiler->mutex);
}

/* ============================================================================
 * INTERNAL HELPER FUNCTIONS
 * ============================================================================ */

/**
 * @brief Validate recompiler handle
 *
 * WHAT: Check if handle is valid
 * WHY: Prevent use of invalid/freed handles
 */
static bool validate_recompiler(recompiler_t recompiler)
{
    if (!recompiler) {
        LOG_ERROR("NULL recompiler handle");
        return false;
    }

    if (recompiler->magic != NIMCP_RECOMPILER_MAGIC) {
        LOG_ERROR("Invalid recompiler magic: 0x%08X", recompiler->magic);
        return false;
    }

    return true;
}

/**
 * @brief Ensure directory exists
 *
 * WHAT: Create directory if it doesn't exist
 * WHY: Need temp directory for files
 */
static bool ensure_temp_dir(const char* path)
{
    if (!path) {
        return false;
    }

    struct stat st;
    if (stat(path, &st) == 0) {
        return S_ISDIR(st.st_mode);
    }

    /* Create directory with 0700 permissions */
    if (mkdir(path, 0700) != 0) {
        if (errno != EEXIST) {
            LOG_ERROR("mkdir failed for %s: %s", path, strerror(errno));
            return false;
        }
    }

    return true;
}

/**
 * @brief Generate unique temp file path
 *
 * WHAT: Create unique filename in temp directory
 * WHY: Avoid conflicts between compilations
 */
static char* generate_temp_path(recompiler_t recompiler, const char* suffix)
{
    nimcp_platform_mutex_lock(&recompiler->mutex);
    uint32_t counter = recompiler->temp_counter++;
    nimcp_platform_mutex_unlock(&recompiler->mutex);

    size_t len = strlen(recompiler->config.temp_dir) + 32 + strlen(suffix);
    char* path = nimcp_malloc(len);
    if (!path) {
        return NULL;
    }

    snprintf(path, len, "%s/patch_%u_%lu%s",
             recompiler->config.temp_dir,
             counter,
             (unsigned long)nimcp_time_get_us(),
             suffix);

    return path;
}

/**
 * @brief Write source code to temp file
 *
 * WHAT: Write string to file
 * WHY: gcc needs file input
 */
static bool write_temp_source(const char* path, const char* source)
{
    FILE* fp = fopen(path, "w");
    if (!fp) {
        LOG_ERROR("fopen failed for %s: %s", path, strerror(errno));
        return false;
    }

    size_t len = strlen(source);
    size_t written = fwrite(source, 1, len, fp);
    fclose(fp);

    if (written != len) {
        LOG_ERROR("fwrite incomplete: %zu of %zu", written, len);
        return false;
    }

    return true;
}

/**
 * @brief Build gcc command line
 *
 * WHAT: Construct full compilation command
 * WHY: gcc needs many flags
 */
static bool build_gcc_command(
    recompiler_t recompiler,
    const recompile_request_t* request,
    char* cmd_buffer,
    size_t buffer_size)
{
    char* p = cmd_buffer;
    char* end = cmd_buffer + buffer_size;

    /**
     * WHAT: Start with compiler executable
     * WHY: Need gcc or alternative
     */
    const char* compiler = recompiler->config.compiler_path[0] ?
                           recompiler->config.compiler_path : "gcc";

    if (recompiler->config.use_ccache) {
        p += snprintf(p, end - p, "ccache ");
    }

    p += snprintf(p, end - p, "%s -shared", compiler);

    /**
     * WHAT: Add standard flags
     * WHY: Required for shared library
     */
    if (request->position_independent) {
        p += snprintf(p, end - p, " -fPIC");
    }

    if (request->debug_symbols) {
        p += snprintf(p, end - p, " -g");
    }

    if (request->optimize) {
        p += snprintf(p, end - p, " -O2");
    } else {
        p += snprintf(p, end - p, " -O0");
    }

    if (request->warnings_as_errors) {
        p += snprintf(p, end - p, " -Werror");
    }

    if (request->verbose) {
        p += snprintf(p, end - p, " -v");
    }

    /**
     * WHAT: Add include paths
     * WHY: Headers need to be found
     */
    for (uint32_t i = 0; i < request->include_count && request->include_paths; i++) {
        if (request->include_paths[i]) {
            p += snprintf(p, end - p, " -I'%s'", request->include_paths[i]);
        }
    }

    /**
     * WHAT: Add library paths
     * WHY: Libraries need to be found
     */
    for (uint32_t i = 0; i < request->library_count && request->library_paths; i++) {
        if (request->library_paths[i]) {
            p += snprintf(p, end - p, " -L'%s'", request->library_paths[i]);
        }
    }

    /**
     * WHAT: Add extra CFLAGS
     * WHY: Allow custom flags
     */
    if (request->extra_cflags && request->extra_cflags[0]) {
        p += snprintf(p, end - p, " %s", request->extra_cflags);
    }

    /**
     * WHAT: Add source file
     * WHY: Input to compile
     */
    p += snprintf(p, end - p, " '%s'", request->source_file);

    /**
     * WHAT: Add libraries to link
     * WHY: May need external libraries
     */
    for (uint32_t i = 0; i < request->lib_count && request->libraries; i++) {
        if (request->libraries[i]) {
            p += snprintf(p, end - p, " -l%s", request->libraries[i]);
        }
    }

    /**
     * WHAT: Add output path
     * WHY: Specify destination
     */
    p += snprintf(p, end - p, " -o '%s'", request->output_so);

    /**
     * WHAT: Redirect stderr to stdout
     * WHY: Capture all output
     */
    p += snprintf(p, end - p, " 2>&1");

    if (p >= end) {
        LOG_ERROR("Command buffer overflow");
        return false;
    }

    return true;
}

/**
 * @brief Execute compiler command with timeout
 *
 * WHAT: Run gcc and capture output
 * WHY: Need compiler result
 */
static bool execute_compiler(
    const char* command,
    uint32_t timeout_ms,
    char* output,
    size_t output_size,
    int* exit_code)
{
    /**
     * WHAT: Create pipe for output capture
     * WHY: Need to read stdout/stderr
     */
    int pipefd[2];
    if (pipe(pipefd) != 0) {
        LOG_ERROR("pipe failed: %s", strerror(errno));
        return false;
    }

    /**
     * WHAT: Fork child process
     * WHY: Execute compiler in isolation
     */
    pid_t pid = fork();
    if (pid < 0) {
        LOG_ERROR("fork failed: %s", strerror(errno));
        close(pipefd[0]);
        close(pipefd[1]);
        return false;
    }

    if (pid == 0) {
        /* Child process */

        /* Redirect stdout and stderr to pipe */
        close(pipefd[0]);
        dup2(pipefd[1], STDOUT_FILENO);
        dup2(pipefd[1], STDERR_FILENO);
        close(pipefd[1]);

        /* Execute via shell */
        execl("/bin/sh", "sh", "-c", command, (char*)NULL);

        /* If exec failed */
        _exit(127);
    }

    /* Parent process */
    close(pipefd[1]);

    /**
     * WHAT: Set up non-blocking read with timeout
     * WHY: Prevent hanging on slow compilation
     */
    int flags = fcntl(pipefd[0], F_GETFL, 0);
    fcntl(pipefd[0], F_SETFL, flags | O_NONBLOCK);

    /**
     * WHAT: Read output with timeout
     * WHY: Capture compiler output
     */
    size_t total_read = 0;
    uint64_t start_time = nimcp_time_monotonic_ms();
    bool timed_out = false;

    while (total_read < output_size - 1) {
        /* Check timeout */
        uint64_t elapsed = nimcp_time_monotonic_ms() - start_time;
        if (timeout_ms > 0 && elapsed > timeout_ms) {
            timed_out = true;
            kill(pid, SIGKILL);
            break;
        }

        /* Try to read */
        ssize_t n = read(pipefd[0], output + total_read,
                         output_size - 1 - total_read);

        if (n > 0) {
            total_read += n;
        } else if (n == 0) {
            /* EOF */
            break;
        } else if (errno == EAGAIN || errno == EWOULDBLOCK) {
            /* No data available, check if child is done */
            int status;
            pid_t result = waitpid(pid, &status, WNOHANG);
            if (result == pid) {
                /* Child exited, read any remaining data */
                while ((n = read(pipefd[0], output + total_read,
                                 output_size - 1 - total_read)) > 0) {
                    total_read += n;
                }
                break;
            }
            /* Small sleep before retry */
            nimcp_time_sleep_ms(10);
        } else {
            /* Read error */
            break;
        }
    }

    output[total_read] = '\0';
    close(pipefd[0]);

    /**
     * WHAT: Wait for child to finish
     * WHY: Collect exit status
     */
    int status;
    waitpid(pid, &status, 0);

    if (timed_out) {
        *exit_code = -1;
        snprintf(output + total_read, output_size - total_read,
                 "\n[TIMEOUT after %u ms]", timeout_ms);
        return false;
    }

    if (WIFEXITED(status)) {
        *exit_code = WEXITSTATUS(status);
    } else if (WIFSIGNALED(status)) {
        *exit_code = -WTERMSIG(status);
    } else {
        *exit_code = -1;
    }

    return true;
}

/**
 * @brief Verify output file exists
 *
 * WHAT: Check that compiled .so was created
 * WHY: Compiler might not produce output
 */
static bool verify_output_exists(const char* path)
{
    struct stat st;
    if (stat(path, &st) != 0) {
        return false;
    }

    return S_ISREG(st.st_mode) && st.st_size > 0;
}

/**
 * @brief Parse compiler output for warnings/errors
 *
 * WHAT: Count warning and error messages
 * WHY: Provide structured diagnostics
 */
static void parse_compiler_output(
    const char* output,
    uint32_t* warning_count,
    uint32_t* error_count)
{
    *warning_count = 0;
    *error_count = 0;

    if (!output) {
        return;
    }

    /**
     * WHAT: Count occurrences of warning/error patterns
     * WHY: Quick diagnostic summary
     */
    const char* p = output;
    while (*p) {
        if (strncmp(p, "warning:", 8) == 0) {
            (*warning_count)++;
        } else if (strncmp(p, "error:", 6) == 0) {
            (*error_count)++;
        }
        p++;
    }
}

/**
 * @brief Run nm to check for symbol
 *
 * WHAT: Use nm to verify symbol exists
 * WHY: Ensure function is exported
 */
static bool run_nm_check(const char* so_path, const char* symbol_name)
{
    /**
     * WHAT: Build nm command
     * WHY: Query symbol table
     */
    char cmd[1024];
    snprintf(cmd, sizeof(cmd),
             "nm -D '%s' 2>/dev/null | grep -q ' %s$'",
             so_path, symbol_name);

    int ret = system(cmd);
    return (ret == 0);
}
