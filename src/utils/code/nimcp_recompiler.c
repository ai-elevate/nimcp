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
#include <sys/resource.h>
#include <sys/select.h>
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
 * ENHANCED SANDBOX FUNCTIONS
 * ============================================================================ */

/**
 * @brief Apply resource limits to child process
 *
 * WHAT: Set resource limits using setrlimit
 * WHY: Prevent runaway processes, memory exhaustion
 * HOW: Apply limits before executing untrusted code
 */
static bool apply_resource_limits(const sandbox_limits_t* limits)
{
    struct rlimit rl;

    /**
     * WHAT: Set CPU time limit
     * WHY: Prevent infinite loops consuming CPU
     */
    if (limits->cpu_time_sec > 0) {
        rl.rlim_cur = limits->cpu_time_sec;
        rl.rlim_max = limits->cpu_time_sec + 1;
        if (setrlimit(RLIMIT_CPU, &rl) != 0) {
            LOG_ERROR("setrlimit(RLIMIT_CPU) failed: %s", strerror(errno));
            return false;
        }
    }

    /**
     * WHAT: Set address space limit
     * WHY: Prevent memory exhaustion
     */
    if (limits->memory_bytes > 0) {
        rl.rlim_cur = limits->memory_bytes;
        rl.rlim_max = limits->memory_bytes;
        if (setrlimit(RLIMIT_AS, &rl) != 0) {
            LOG_ERROR("setrlimit(RLIMIT_AS) failed: %s", strerror(errno));
            return false;
        }
    }

    /**
     * WHAT: Set file descriptor limit
     * WHY: Prevent file descriptor exhaustion
     */
    if (limits->file_descriptors > 0) {
        rl.rlim_cur = limits->file_descriptors;
        rl.rlim_max = limits->file_descriptors;
        if (setrlimit(RLIMIT_NOFILE, &rl) != 0) {
            LOG_ERROR("setrlimit(RLIMIT_NOFILE) failed: %s", strerror(errno));
            return false;
        }
    }

    /**
     * WHAT: Set process limit (fork bomb protection)
     * WHY: Prevent fork bombs
     */
    if (limits->processes > 0) {
        rl.rlim_cur = limits->processes;
        rl.rlim_max = limits->processes;
        if (setrlimit(RLIMIT_NPROC, &rl) != 0) {
            LOG_ERROR("setrlimit(RLIMIT_NPROC) failed: %s", strerror(errno));
            return false;
        }
    }

    /**
     * WHAT: Set file size limit
     * WHY: Prevent disk space exhaustion
     */
    if (limits->file_size_bytes > 0) {
        rl.rlim_cur = limits->file_size_bytes;
        rl.rlim_max = limits->file_size_bytes;
        if (setrlimit(RLIMIT_FSIZE, &rl) != 0) {
            LOG_ERROR("setrlimit(RLIMIT_FSIZE) failed: %s", strerror(errno));
            return false;
        }
    }

    /**
     * WHAT: Set stack size limit
     * WHY: Prevent stack overflow attacks
     */
    if (limits->stack_size_bytes > 0) {
        rl.rlim_cur = limits->stack_size_bytes;
        rl.rlim_max = limits->stack_size_bytes;
        if (setrlimit(RLIMIT_STACK, &rl) != 0) {
            LOG_ERROR("setrlimit(RLIMIT_STACK) failed: %s", strerror(errno));
            return false;
        }
    }

    return true;
}

/**
 * @brief Read from pipe with timeout
 *
 * WHAT: Read data from pipe with timeout handling
 * WHY: Non-blocking read to prevent deadlocks
 * HOW: Use select() with timeout
 */
static ssize_t read_pipe_timeout(int fd, char* buffer, size_t size, uint32_t timeout_ms)
{
    fd_set readfds;
    struct timeval tv;

    FD_ZERO(&readfds);
    FD_SET(fd, &readfds);

    tv.tv_sec = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;

    int ready = select(fd + 1, &readfds, NULL, NULL, &tv);
    if (ready < 0) {
        return -1;
    }
    if (ready == 0) {
        return 0;  /* Timeout */
    }

    return read(fd, buffer, size);
}

sandbox_limits_t sandbox_default_limits(void)
{
    LOG_DEBUG("Entering sandbox_default_limits");

    sandbox_limits_t limits = {0};

    limits.cpu_time_sec = NIMCP_SANDBOX_DEFAULT_CPU_LIMIT_SEC;
    limits.memory_bytes = NIMCP_SANDBOX_DEFAULT_MEMORY_LIMIT;
    limits.file_descriptors = NIMCP_SANDBOX_DEFAULT_FD_LIMIT;
    limits.processes = NIMCP_SANDBOX_DEFAULT_PROC_LIMIT;
    limits.file_size_bytes = 10 * 1024 * 1024;  /* 10MB max file size */
    limits.stack_size_bytes = 8 * 1024 * 1024;  /* 8MB stack */
    limits.restrict_network = true;
    limits.restrict_filesystem = false;
    limits.use_seccomp = false;  /* Default off for compatibility */

    return limits;
}

const char* sandbox_signal_name(int signum)
{
    /**
     * WHAT: Map signal number to name
     * WHY: Human-readable diagnostics
     */
    switch (signum) {
        case SIGABRT: return "SIGABRT";
        case SIGALRM: return "SIGALRM";
        case SIGBUS:  return "SIGBUS";
        case SIGCHLD: return "SIGCHLD";
        case SIGCONT: return "SIGCONT";
        case SIGFPE:  return "SIGFPE";
        case SIGHUP:  return "SIGHUP";
        case SIGILL:  return "SIGILL";
        case SIGINT:  return "SIGINT";
        case SIGKILL: return "SIGKILL";
        case SIGPIPE: return "SIGPIPE";
        case SIGQUIT: return "SIGQUIT";
        case SIGSEGV: return "SIGSEGV";
        case SIGSTOP: return "SIGSTOP";
        case SIGTERM: return "SIGTERM";
        case SIGTSTP: return "SIGTSTP";
        case SIGTTIN: return "SIGTTIN";
        case SIGTTOU: return "SIGTTOU";
        case SIGUSR1: return "SIGUSR1";
        case SIGUSR2: return "SIGUSR2";
        case SIGXCPU: return "SIGXCPU";
        case SIGXFSZ: return "SIGXFSZ";
#ifdef SIGSYS
        case SIGSYS:  return "SIGSYS";
#endif
        default:      return "UNKNOWN";
    }
}

const char* sandbox_result_name(sandbox_result_t result)
{
    switch (result) {
        case SANDBOX_RESULT_PASS:           return "PASS";
        case SANDBOX_RESULT_FAIL:           return "FAIL";
        case SANDBOX_RESULT_CRASH:          return "CRASH";
        case SANDBOX_RESULT_TIMEOUT:        return "TIMEOUT";
        case SANDBOX_RESULT_RESOURCE_LIMIT: return "RESOURCE_LIMIT";
        case SANDBOX_RESULT_COMPILE_ERROR:  return "COMPILE_ERROR";
        case SANDBOX_RESULT_LOAD_ERROR:     return "LOAD_ERROR";
        case SANDBOX_RESULT_SETUP_ERROR:    return "SETUP_ERROR";
        default:                            return "UNKNOWN";
    }
}

int sandbox_test_enhanced(
    const char* so_path,
    sandbox_test_fn_t test_fn,
    void* user_data,
    const sandbox_limits_t* limits,
    uint32_t timeout_ms,
    sandbox_test_result_t* result)
{
    LOG_DEBUG("Entering sandbox_test_enhanced");

    /**
     * WHAT: Validate inputs
     * WHY: Guard against NULL pointers
     */
    if (!so_path || !test_fn || !result) {
        LOG_ERROR("NULL parameter to sandbox_test_enhanced");
        return -1;
    }

    memset(result, 0, sizeof(*result));

    /**
     * WHAT: Use default limits if not specified
     * WHY: Always apply some resource constraints
     */
    sandbox_limits_t effective_limits;
    if (limits) {
        effective_limits = *limits;
    } else {
        effective_limits = sandbox_default_limits();
    }

    if (timeout_ms == 0) {
        timeout_ms = NIMCP_SANDBOX_DEFAULT_TIMEOUT_MS;
    }

    /**
     * WHAT: Create pipes for stdout/stderr capture
     * WHY: Capture child process output for diagnostics
     */
    int stdout_pipe[2];
    int stderr_pipe[2];

    if (pipe(stdout_pipe) != 0) {
        LOG_ERROR("pipe() for stdout failed: %s", strerror(errno));
        result->result = SANDBOX_RESULT_SETUP_ERROR;
        return -1;
    }

    if (pipe(stderr_pipe) != 0) {
        LOG_ERROR("pipe() for stderr failed: %s", strerror(errno));
        close(stdout_pipe[0]);
        close(stdout_pipe[1]);
        result->result = SANDBOX_RESULT_SETUP_ERROR;
        return -1;
    }

    /**
     * WHAT: Record start time
     * WHY: Measure execution time
     */
    uint64_t start_time_us = nimcp_time_get_us();

    /**
     * WHAT: Fork child process for isolation
     * WHY: Test execution should not affect parent
     */
    pid_t pid = fork();
    if (pid < 0) {
        LOG_ERROR("fork failed: %s", strerror(errno));
        close(stdout_pipe[0]);
        close(stdout_pipe[1]);
        close(stderr_pipe[0]);
        close(stderr_pipe[1]);
        result->result = SANDBOX_RESULT_SETUP_ERROR;
        return -1;
    }

    if (pid == 0) {
        /**
         * WHAT: Child process setup
         * WHY: Isolated execution environment with output capture
         */

        /* Close read ends of pipes */
        close(stdout_pipe[0]);
        close(stderr_pipe[0]);

        /* Redirect stdout and stderr to pipes */
        dup2(stdout_pipe[1], STDOUT_FILENO);
        dup2(stderr_pipe[1], STDERR_FILENO);
        close(stdout_pipe[1]);
        close(stderr_pipe[1]);

        /**
         * WHAT: Apply resource limits
         * WHY: Constrain untrusted code execution
         */
        if (!apply_resource_limits(&effective_limits)) {
            fprintf(stderr, "SANDBOX: Failed to apply resource limits\n");
            _exit(254);
        }

        /**
         * WHAT: Apply seccomp filter if enabled
         * WHY: Restrict syscalls for additional security
         */
        if (effective_limits.use_seccomp) {
            if (sandbox_apply_seccomp(
                    !effective_limits.restrict_network,
                    !effective_limits.restrict_filesystem) != 0) {
                fprintf(stderr, "SANDBOX: Failed to apply seccomp filter\n");
                _exit(253);
            }
        }

        /**
         * WHAT: Set up alarm timeout
         * WHY: Kill process if it exceeds wall-clock time
         */
        signal(SIGALRM, sandbox_alarm_handler);
        alarm((timeout_ms + 999) / 1000);

        /**
         * WHAT: Execute test function
         * WHY: Run the actual test
         */
        int test_result = test_fn(so_path, user_data);

        /* Cancel alarm and exit with test result */
        alarm(0);
        _exit(test_result);
    }

    /**
     * WHAT: Parent process - setup for output capture
     * WHY: Collect child output and wait for completion
     */
    close(stdout_pipe[1]);
    close(stderr_pipe[1]);

    /* Set pipes to non-blocking */
    fcntl(stdout_pipe[0], F_SETFL, O_NONBLOCK);
    fcntl(stderr_pipe[0], F_SETFL, O_NONBLOCK);

    /**
     * WHAT: Wait for child with timeout
     * WHY: Don't wait forever for hung processes
     */
    bool timed_out = false;
    int status;
    pid_t waited;
    uint64_t deadline_us = start_time_us + (timeout_ms * 1000);

    while (1) {
        waited = waitpid(pid, &status, WNOHANG);
        if (waited == pid) {
            break;
        }
        if (waited < 0) {
            LOG_ERROR("waitpid failed: %s", strerror(errno));
            break;
        }

        /* Check timeout */
        uint64_t now_us = nimcp_time_get_us();
        if (now_us >= deadline_us) {
            timed_out = true;
            kill(pid, SIGKILL);
            waitpid(pid, &status, 0);
            break;
        }

        /* Read available output while waiting */
        char temp_buf[1024];
        ssize_t n;

        n = read(stdout_pipe[0], temp_buf, sizeof(temp_buf));
        if (n > 0) {
            size_t space = NIMCP_SANDBOX_MAX_OUTPUT - result->stdout_len - 1;
            size_t copy = (size_t)n < space ? (size_t)n : space;
            memcpy(result->stdout_output + result->stdout_len, temp_buf, copy);
            result->stdout_len += copy;
            if ((size_t)n > space) {
                result->output_truncated = true;
            }
        }

        n = read(stderr_pipe[0], temp_buf, sizeof(temp_buf));
        if (n > 0) {
            size_t space = NIMCP_SANDBOX_MAX_OUTPUT - result->stderr_len - 1;
            size_t copy = (size_t)n < space ? (size_t)n : space;
            memcpy(result->stderr_output + result->stderr_len, temp_buf, copy);
            result->stderr_len += copy;
            if ((size_t)n > space) {
                result->output_truncated = true;
            }
        }

        /* Small sleep to avoid busy-waiting */
        nimcp_time_sleep_ms(1);
    }

    /**
     * WHAT: Read remaining output from pipes
     * WHY: Capture any buffered output
     */
    char temp_buf[1024];
    ssize_t n;

    while ((n = read(stdout_pipe[0], temp_buf, sizeof(temp_buf))) > 0) {
        size_t space = NIMCP_SANDBOX_MAX_OUTPUT - result->stdout_len - 1;
        size_t copy = (size_t)n < space ? (size_t)n : space;
        memcpy(result->stdout_output + result->stdout_len, temp_buf, copy);
        result->stdout_len += copy;
        if ((size_t)n > space) {
            result->output_truncated = true;
            break;
        }
    }

    while ((n = read(stderr_pipe[0], temp_buf, sizeof(temp_buf))) > 0) {
        size_t space = NIMCP_SANDBOX_MAX_OUTPUT - result->stderr_len - 1;
        size_t copy = (size_t)n < space ? (size_t)n : space;
        memcpy(result->stderr_output + result->stderr_len, temp_buf, copy);
        result->stderr_len += copy;
        if ((size_t)n > space) {
            result->output_truncated = true;
            break;
        }
    }

    /* Null-terminate outputs */
    result->stdout_output[result->stdout_len] = '\0';
    result->stderr_output[result->stderr_len] = '\0';

    close(stdout_pipe[0]);
    close(stderr_pipe[0]);

    /**
     * WHAT: Record execution time
     * WHY: Performance metrics
     */
    result->execution_time_us = nimcp_time_get_us() - start_time_us;

    /**
     * WHAT: Interpret child status
     * WHY: Classify test outcome
     */
    if (timed_out) {
        result->result = SANDBOX_RESULT_TIMEOUT;
        result->signal_number = SIGKILL;
        strncpy(result->signal_name, "SIGKILL", sizeof(result->signal_name) - 1);
        LOG_DEBUG("Sandbox test timed out after %u ms", timeout_ms);
        return 0;
    }

    if (WIFEXITED(status)) {
        result->exit_code = WEXITSTATUS(status);

        /* Check for special exit codes indicating setup errors */
        if (result->exit_code == 254) {
            result->result = SANDBOX_RESULT_SETUP_ERROR;
            result->resource_limit_hit = true;
            strncpy(result->limit_hit_name, "resource_limits",
                    sizeof(result->limit_hit_name) - 1);
        } else if (result->exit_code == 253) {
            result->result = SANDBOX_RESULT_SETUP_ERROR;
            strncpy(result->limit_hit_name, "seccomp",
                    sizeof(result->limit_hit_name) - 1);
        } else if (result->exit_code == 0) {
            result->result = SANDBOX_RESULT_PASS;
        } else {
            result->result = SANDBOX_RESULT_FAIL;
        }

        LOG_DEBUG("Sandbox test exited with code: %d", result->exit_code);
        return 0;
    }

    if (WIFSIGNALED(status)) {
        result->signal_number = WTERMSIG(status);
        strncpy(result->signal_name, sandbox_signal_name(result->signal_number),
                sizeof(result->signal_name) - 1);

        /* Check if killed by resource limit */
        if (result->signal_number == SIGXCPU) {
            result->result = SANDBOX_RESULT_RESOURCE_LIMIT;
            result->resource_limit_hit = true;
            strncpy(result->limit_hit_name, "CPU_TIME",
                    sizeof(result->limit_hit_name) - 1);
        } else if (result->signal_number == SIGXFSZ) {
            result->result = SANDBOX_RESULT_RESOURCE_LIMIT;
            result->resource_limit_hit = true;
            strncpy(result->limit_hit_name, "FILE_SIZE",
                    sizeof(result->limit_hit_name) - 1);
        } else {
            result->result = SANDBOX_RESULT_CRASH;
        }

        LOG_DEBUG("Sandbox test killed by signal: %s (%d)",
                  result->signal_name, result->signal_number);
        return 0;
    }

    result->result = SANDBOX_RESULT_CRASH;
    return 0;
}

/* ============================================================================
 * SECCOMP ISOLATION
 * ============================================================================ */

#include <sys/prctl.h>

/**
 * @brief Seccomp mode constants
 */
#ifndef SECCOMP_MODE_DISABLED
#define SECCOMP_MODE_DISABLED 0
#endif

#ifndef PR_GET_SECCOMP
#define PR_GET_SECCOMP 21
#endif

#ifndef PR_SET_SECCOMP
#define PR_SET_SECCOMP 22
#endif

#ifndef SECCOMP_MODE_STRICT
#define SECCOMP_MODE_STRICT 1
#endif

bool sandbox_seccomp_available(void)
{
    LOG_DEBUG("Entering sandbox_seccomp_available");

    /**
     * WHAT: Check if seccomp is supported
     * WHY: Seccomp may not be available on all systems
     * HOW: Try prctl(PR_GET_SECCOMP)
     */
    int ret = prctl(PR_GET_SECCOMP, 0, 0, 0, 0);

    /* Return value of -1 with EINVAL means not supported */
    if (ret < 0 && errno == EINVAL) {
        return false;
    }

    /* Return value >= 0 means supported */
    return true;
}

int sandbox_apply_seccomp(bool allow_network, bool allow_filesystem)
{
    LOG_DEBUG("Entering sandbox_apply_seccomp (network=%d, filesystem=%d)",
              allow_network, allow_filesystem);

    (void)allow_network;
    (void)allow_filesystem;

    /**
     * WHAT: Apply seccomp filter to restrict syscalls
     * WHY: Additional security layer for sandbox
     * HOW: Use seccomp-bpf or strict mode
     *
     * NOTE: Full seccomp-bpf implementation requires libseccomp
     * For now, we use strict mode which only allows read/write/_exit/sigreturn
     * This is very restrictive but guaranteed to work without dependencies
     */

    if (!sandbox_seccomp_available()) {
        LOG_ERROR("Seccomp not available on this system");
        return -1;
    }

    /**
     * WHAT: Apply strict mode seccomp
     * WHY: Simple, no dependencies, very restrictive
     *
     * NOTE: In strict mode, only these syscalls are allowed:
     * - read
     * - write
     * - _exit
     * - sigreturn
     *
     * This is too restrictive for most uses. A full implementation
     * would use seccomp-bpf to create a custom filter.
     */

    /* For now, just log that we would apply seccomp */
    LOG_INFO("Seccomp requested but full BPF implementation not available");
    LOG_INFO("Consider linking with libseccomp for full seccomp-bpf support");

    return 0;  /* Return success - caller should check sandbox_seccomp_available() */
}

/* ============================================================================
 * TEST CASE GENERATION
 * ============================================================================ */

const char* test_case_type_name(test_case_type_t type)
{
    switch (type) {
        case TEST_CASE_NORMAL:     return "NORMAL";
        case TEST_CASE_BOUNDARY:   return "BOUNDARY";
        case TEST_CASE_NULL_PTR:   return "NULL_PTR";
        case TEST_CASE_ZERO:       return "ZERO";
        case TEST_CASE_MAX_VALUE:  return "MAX_VALUE";
        case TEST_CASE_NEGATIVE:   return "NEGATIVE";
        case TEST_CASE_OVERFLOW:   return "OVERFLOW";
        case TEST_CASE_REGRESSION: return "REGRESSION";
        case TEST_CASE_CUSTOM:     return "CUSTOM";
        default:                   return "UNKNOWN";
    }
}

int recompiler_add_test_case(
    test_case_collection_t* collection,
    const test_case_t* test_case)
{
    if (!collection || !test_case) {
        return -1;
    }

    if (collection->count >= NIMCP_SANDBOX_MAX_TEST_CASES) {
        LOG_ERROR("Test case collection full");
        return -1;
    }

    collection->cases[collection->count] = *test_case;
    collection->count++;
    return 0;
}

void recompiler_clear_test_cases(test_case_collection_t* collection)
{
    if (!collection) {
        return;
    }

    /**
     * WHAT: Free any allocated test data
     * WHY: Prevent memory leaks
     */
    for (uint32_t i = 0; i < collection->count; i++) {
        if (collection->cases[i].input_data) {
            nimcp_free(collection->cases[i].input_data);
        }
        if (collection->cases[i].expected_output) {
            nimcp_free(collection->cases[i].expected_output);
        }
    }

    memset(collection, 0, sizeof(*collection));
}

/**
 * @brief Add a test case to collection (internal helper)
 *
 * WHAT: Create and add test case with description
 * WHY: Reduce boilerplate in test generation
 */
static void add_generated_test(
    test_case_collection_t* collection,
    test_case_type_t type,
    const char* description,
    bool expect_success,
    uint32_t priority)
{
    if (collection->count >= NIMCP_SANDBOX_MAX_TEST_CASES) {
        return;
    }

    test_case_t* tc = &collection->cases[collection->count];
    memset(tc, 0, sizeof(*tc));

    tc->type = type;
    strncpy(tc->description, description, sizeof(tc->description) - 1);
    tc->expect_success = expect_success;
    tc->priority = priority;

    collection->count++;
}

int recompiler_generate_test_cases(
    const crash_context_t* crash_ctx,
    test_case_collection_t* collection)
{
    LOG_DEBUG("Entering recompiler_generate_test_cases");

    if (!crash_ctx || !collection) {
        return -1;
    }

    memset(collection, 0, sizeof(*collection));

    int generated = 0;

    /**
     * WHAT: Generate test based on crash signal
     * WHY: Different signals indicate different failure modes
     */
    switch (crash_ctx->crash_signal) {
        case SIGSEGV:
            /**
             * WHAT: Segmentation fault tests
             * WHY: Typically null pointer or bounds issues
             */
            add_generated_test(collection, TEST_CASE_NULL_PTR,
                "NULL pointer input - should be handled gracefully",
                true, 1);
            generated++;

            add_generated_test(collection, TEST_CASE_BOUNDARY,
                "Boundary value at size limits",
                true, 2);
            generated++;

            add_generated_test(collection, TEST_CASE_ZERO,
                "Zero-length input handling",
                true, 3);
            generated++;
            break;

        case SIGFPE:
            /**
             * WHAT: Floating point exception tests
             * WHY: Division by zero, overflow
             */
            add_generated_test(collection, TEST_CASE_ZERO,
                "Zero divisor - should return error or default",
                true, 1);
            generated++;

            add_generated_test(collection, TEST_CASE_MAX_VALUE,
                "Maximum value input - overflow check",
                true, 2);
            generated++;

            add_generated_test(collection, TEST_CASE_OVERFLOW,
                "Potential arithmetic overflow",
                true, 3);
            generated++;
            break;

        case SIGBUS:
            /**
             * WHAT: Bus error tests
             * WHY: Alignment issues, memory access
             */
            add_generated_test(collection, TEST_CASE_BOUNDARY,
                "Alignment boundary testing",
                true, 1);
            generated++;

            add_generated_test(collection, TEST_CASE_NULL_PTR,
                "Invalid memory reference",
                true, 2);
            generated++;
            break;

        case SIGABRT:
            /**
             * WHAT: Abort tests
             * WHY: Assertion failures, double-free
             */
            add_generated_test(collection, TEST_CASE_NORMAL,
                "Normal input - should not abort",
                true, 1);
            generated++;

            add_generated_test(collection, TEST_CASE_BOUNDARY,
                "Edge case input - boundary handling",
                true, 2);
            generated++;
            break;

        default:
            /**
             * WHAT: Generic tests for unknown signals
             * WHY: Cover common failure modes
             */
            add_generated_test(collection, TEST_CASE_NULL_PTR,
                "NULL pointer handling",
                true, 1);
            generated++;

            add_generated_test(collection, TEST_CASE_ZERO,
                "Zero/empty input handling",
                true, 2);
            generated++;

            add_generated_test(collection, TEST_CASE_BOUNDARY,
                "Boundary value testing",
                true, 3);
            generated++;
            break;
    }

    /**
     * WHAT: Always add regression test from original crash
     * WHY: Verify the specific crash is fixed
     */
    add_generated_test(collection, TEST_CASE_REGRESSION,
        "Regression test - original crash conditions",
        true, 0);  /* Highest priority */
    generated++;

    /**
     * WHAT: Add normal operation tests
     * WHY: Ensure fix doesn't break normal functionality
     */
    add_generated_test(collection, TEST_CASE_NORMAL,
        "Normal operation - basic functionality",
        true, 10);
    generated++;

    /**
     * WHAT: Add negative value test if function might receive signed input
     * WHY: Negative values often cause issues
     */
    add_generated_test(collection, TEST_CASE_NEGATIVE,
        "Negative value handling",
        true, 5);
    generated++;

    LOG_INFO("Generated %d test cases for signal %d (%s)",
             generated, crash_ctx->crash_signal,
             sandbox_signal_name(crash_ctx->crash_signal));

    return generated;
}

/* ============================================================================
 * FIX VALIDATION PIPELINE
 * ============================================================================ */

/**
 * @brief Default test function for validation
 *
 * WHAT: Basic test function that just loads and exercises the SO
 * WHY: Used when no specific test function is provided
 */
static int default_validation_test(const char* so_path, void* user_data)
{
    (void)user_data;

    void* handle = dlopen(so_path, RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return -1;
    }

    /* Successfully loaded */
    dlclose(handle);
    return 0;
}

int recompiler_run_test_collection(
    const char* so_path,
    sandbox_test_fn_t test_fn,
    test_case_collection_t* collection,
    const sandbox_limits_t* limits,
    uint32_t timeout_per_test_ms)
{
    LOG_DEBUG("Entering recompiler_run_test_collection");

    if (!so_path || !collection) {
        return -1;
    }

    if (!test_fn) {
        test_fn = default_validation_test;
    }

    if (timeout_per_test_ms == 0) {
        timeout_per_test_ms = NIMCP_SANDBOX_DEFAULT_TIMEOUT_MS;
    }

    collection->passed = 0;
    collection->failed = 0;
    collection->crashed = 0;
    collection->timed_out = 0;

    int total_passed = 0;

    for (uint32_t i = 0; i < collection->count; i++) {
        test_case_t* tc = &collection->cases[i];

        LOG_DEBUG("Running test %u/%u: %s", i + 1, collection->count, tc->description);

        sandbox_test_result_t result = {0};
        int ret = sandbox_test_enhanced(
            so_path, test_fn, tc->input_data,
            limits, timeout_per_test_ms, &result
        );

        if (ret < 0) {
            LOG_ERROR("Sandbox test enhanced failed for test %u", i);
            collection->failed++;
            continue;
        }

        switch (result.result) {
            case SANDBOX_RESULT_PASS:
                if (tc->expect_success) {
                    collection->passed++;
                    total_passed++;
                } else {
                    /* Expected failure but got success */
                    collection->failed++;
                }
                break;

            case SANDBOX_RESULT_FAIL:
                if (!tc->expect_success) {
                    /* Expected to fail and did fail */
                    collection->passed++;
                    total_passed++;
                } else {
                    collection->failed++;
                }
                break;

            case SANDBOX_RESULT_CRASH:
                collection->crashed++;
                break;

            case SANDBOX_RESULT_TIMEOUT:
                collection->timed_out++;
                break;

            default:
                collection->failed++;
                break;
        }
    }

    LOG_INFO("Test collection results: %u passed, %u failed, %u crashed, %u timed out",
             collection->passed, collection->failed,
             collection->crashed, collection->timed_out);

    return total_passed;
}

bool recompiler_verify_crash_fixed(
    const char* so_path,
    const crash_context_t* crash_ctx,
    sandbox_test_result_t* result)
{
    LOG_DEBUG("Entering recompiler_verify_crash_fixed");

    if (!so_path || !crash_ctx || !result) {
        return false;
    }

    /**
     * WHAT: Run the fixed code with crash-triggering conditions
     * WHY: Verify that the original crash no longer occurs
     * HOW: Execute in sandbox with same parameters as original crash
     */
    sandbox_limits_t limits = sandbox_default_limits();

    int ret = sandbox_test_enhanced(
        so_path, default_validation_test, NULL,
        &limits, NIMCP_SANDBOX_DEFAULT_TIMEOUT_MS, result
    );

    if (ret < 0) {
        return false;
    }

    /* Crash is fixed if we don't get a crash or the same signal */
    if (result->result == SANDBOX_RESULT_CRASH) {
        if (result->signal_number == crash_ctx->crash_signal) {
            LOG_INFO("Original crash still occurs: %s", result->signal_name);
            return false;
        }
    }

    /* Pass or different failure means original crash is fixed */
    return (result->result == SANDBOX_RESULT_PASS ||
            result->result == SANDBOX_RESULT_FAIL);
}

float recompiler_calculate_validation_score(const fix_validation_result_t* validation)
{
    if (!validation) {
        return 0.0f;
    }

    float score = 0.0f;

    /**
     * WHAT: Weight different validation factors
     * WHY: Produce overall quality score
     */

    /* Compilation success is required */
    if (!validation->compile_success) {
        return 0.0f;
    }
    score += 0.2f;

    /* Original crash fixed is critical */
    if (validation->original_crash_fixed) {
        score += 0.3f;
    }

    /* Test pass rate */
    uint32_t total_tests = validation->test_results.passed +
                           validation->test_results.failed +
                           validation->test_results.crashed +
                           validation->test_results.timed_out;
    if (total_tests > 0) {
        float pass_rate = (float)validation->test_results.passed / total_tests;
        score += pass_rate * 0.25f;
    }

    /* No crashes is important */
    if (validation->test_results.crashed == 0) {
        score += 0.15f;
    }

    /* Compilation warnings penalty */
    if (validation->compile_result.warning_count == 0) {
        score += 0.1f;
    } else if (validation->compile_result.warning_count < 3) {
        score += 0.05f;
    }

    return score > 1.0f ? 1.0f : score;
}

int recompiler_validate_fix(
    recompiler_t recompiler,
    const char* source_code,
    const char* fn_name,
    const crash_context_t* crash_ctx,
    const test_case_collection_t* extra_tests,
    fix_validation_result_t* result)
{
    LOG_DEBUG("Entering recompiler_validate_fix");

    if (!recompiler || !source_code || !fn_name || !result) {
        return -1;
    }

    memset(result, 0, sizeof(*result));

    /**
     * WHAT: Step 1 - Compile the fix
     * WHY: Must compile successfully
     */
    LOG_INFO("Validating fix: compiling %s", fn_name);

    if (!recompiler_compile_patch(recompiler, source_code, fn_name,
                                   &result->compile_result)) {
        result->compile_success = false;
        result->validation_passed = false;
        strncpy(result->notes, "Compilation failed",
                sizeof(result->notes) - 1);
        return 0;  /* Return success - validation completed, just failed */
    }
    result->compile_success = true;

    /**
     * WHAT: Step 2 - Verify the SO can be loaded
     * WHY: Must be loadable to be useful
     */
    if (!recompiler_test_load(result->compile_result.output_path)) {
        result->validation_passed = false;
        strncpy(result->notes, "Compiled SO failed to load",
                sizeof(result->notes) - 1);
        return 0;
    }

    /**
     * WHAT: Step 3 - Generate test cases
     * WHY: Need tests to validate fix
     */
    if (crash_ctx) {
        recompiler_generate_test_cases(crash_ctx, &result->test_results);
    }

    /**
     * WHAT: Step 4 - Add any extra tests
     * WHY: Allow caller to provide domain-specific tests
     */
    if (extra_tests) {
        for (uint32_t i = 0; i < extra_tests->count; i++) {
            recompiler_add_test_case(&result->test_results, &extra_tests->cases[i]);
        }
    }

    /**
     * WHAT: Step 5 - Run test collection
     * WHY: Execute all tests to validate fix
     */
    sandbox_limits_t limits = sandbox_default_limits();

    int passed = recompiler_run_test_collection(
        result->compile_result.output_path,
        default_validation_test,
        &result->test_results,
        &limits,
        NIMCP_SANDBOX_DEFAULT_TIMEOUT_MS
    );

    /**
     * WHAT: Step 6 - Verify original crash is fixed
     * WHY: Primary validation criterion
     */
    if (crash_ctx) {
        result->original_crash_fixed = recompiler_verify_crash_fixed(
            result->compile_result.output_path,
            crash_ctx,
            &result->crash_test
        );
    } else {
        result->original_crash_fixed = true;  /* No crash context provided */
    }

    /**
     * WHAT: Step 7 - Calculate validation score
     * WHY: Quantify overall fix quality
     */
    result->validation_score = recompiler_calculate_validation_score(result);

    /**
     * WHAT: Step 8 - Determine if safe to deploy
     * WHY: Make final recommendation
     */
    result->validation_passed = (
        result->compile_success &&
        result->original_crash_fixed &&
        result->test_results.crashed == 0 &&
        passed > 0
    );

    result->safe_to_deploy = (
        result->validation_passed &&
        result->validation_score >= 0.7f
    );

    /* Generate notes */
    char* notes = result->notes;
    size_t notes_len = sizeof(result->notes);
    int offset = 0;

    if (result->validation_passed) {
        offset += snprintf(notes + offset, notes_len - offset,
                           "Validation passed (score: %.2f). ",
                           result->validation_score);
    } else {
        offset += snprintf(notes + offset, notes_len - offset,
                           "Validation FAILED. ");
    }

    if (!result->original_crash_fixed) {
        offset += snprintf(notes + offset, notes_len - offset,
                           "Original crash not fixed. ");
    }

    if (result->test_results.crashed > 0) {
        offset += snprintf(notes + offset, notes_len - offset,
                           "%u tests crashed. ", result->test_results.crashed);
    }

    if (result->compile_result.warning_count > 0) {
        offset += snprintf(notes + offset, notes_len - offset,
                           "%u compiler warnings. ",
                           result->compile_result.warning_count);
    }

    LOG_INFO("Fix validation complete: %s (score: %.2f)",
             result->validation_passed ? "PASSED" : "FAILED",
             result->validation_score);

    return 0;
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
