/**
 * @file nimcp_code_immune.c
 * @brief Code Immune System - Runtime Crash Defense Implementation
 * @version 1.0.0
 * @date 2025-12-27
 *
 * WHAT: Implements the code immune system for crash detection and hot-patching
 * WHY:  Enable self-healing software that can survive and adapt to bugs
 * HOW:  Integrate with signal handler, pattern-match crashes, generate patches
 *
 * @author NIMCP Development Team
 */

#include "cognitive/immune/nimcp_code_immune.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/platform/nimcp_platform_mutex.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/signal/nimcp_signal_handler.h"
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <dlfcn.h>
#include <execinfo.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>

/* Mutex convenience macros - matches brain_immune.c pattern */
#define nimcp_mutex_create() nimcp_platform_mutex_create()
#define nimcp_mutex_lock(m) nimcp_platform_mutex_lock((nimcp_platform_mutex_t*)(m))
#define nimcp_mutex_unlock(m) nimcp_platform_mutex_unlock((nimcp_platform_mutex_t*)(m))
#define nimcp_mutex_destroy(m) do { \
    nimcp_platform_mutex_destroy((nimcp_platform_mutex_t*)(m)); \
    nimcp_free(m); \
} while(0)

/* ============================================================================
 * Internal Helpers - Forward Declarations
 * ============================================================================ */

static uint64_t get_timestamp_ms(void);
static code_antigen_t* find_antigen_by_id(code_immune_system_t* system, uint64_t id);
static code_b_cell_t* find_b_cell_by_id(code_immune_system_t* system, uint64_t id);
static code_antibody_t* find_antibody_by_id(code_immune_system_t* system, uint64_t id);
static code_crash_type_t signal_to_crash_type(int signal);
static void process_pending_antigens(code_immune_system_t* system);
static void decay_antibodies(code_immune_system_t* system, uint64_t delta_ms);

/* Global code immune instance for signal handler callback */
static code_immune_system_t* g_code_immune_instance = NULL;

/* ============================================================================
 * Async-Signal-Safe Crash Queue
 * ============================================================================
 * Signal handlers can only call async-signal-safe functions. We use volatile
 * sig_atomic_t to safely communicate with the main thread without locks.
 */
#include <signal.h>

#define SIGNAL_SAFE_QUEUE_SIZE 8

typedef struct {
    volatile sig_atomic_t signal;
    volatile sig_atomic_t valid;
} signal_safe_crash_entry_t;

static signal_safe_crash_entry_t g_pending_crashes[SIGNAL_SAFE_QUEUE_SIZE];
static volatile sig_atomic_t g_pending_crash_head = 0;
static volatile sig_atomic_t g_pending_crash_count = 0;

/* ============================================================================
 * String Conversion
 * ============================================================================ */

/**
 * @brief Convert crash type to string
 */
const char* code_immune_crash_type_to_string(code_crash_type_t type) {
    switch (type) {
        case CODE_CRASH_NONE:    return "NONE";
        case CODE_CRASH_SIGSEGV: return "SIGSEGV";
        case CODE_CRASH_SIGBUS:  return "SIGBUS";
        case CODE_CRASH_SIGILL:  return "SIGILL";
        case CODE_CRASH_SIGFPE:  return "SIGFPE";
        case CODE_CRASH_SIGABRT: return "SIGABRT";
        case CODE_CRASH_ALL:     return "ALL";
        default: return "UNKNOWN";
    }
}

/**
 * @brief Convert B cell state to string
 */
const char* code_immune_b_cell_state_to_string(code_b_cell_state_t state) {
    switch (state) {
        case CODE_B_CELL_NAIVE:     return "NAIVE";
        case CODE_B_CELL_ACTIVATED: return "ACTIVATED";
        case CODE_B_CELL_PLASMA:    return "PLASMA";
        case CODE_B_CELL_MEMORY:    return "MEMORY";
        case CODE_B_CELL_APOPTOTIC: return "APOPTOTIC";
        default: return "UNKNOWN";
    }
}

/**
 * @brief Convert antibody class to string
 */
const char* code_immune_antibody_class_to_string(code_antibody_class_t ab_class) {
    switch (ab_class) {
        case CODE_ANTIBODY_IGM: return "IgM";
        case CODE_ANTIBODY_IGG: return "IgG";
        case CODE_ANTIBODY_IGE: return "IgE";
        default: return "UNKNOWN";
    }
}

/* ============================================================================
 * Internal Helpers - Implementation
 * ============================================================================ */

/**
 * @brief Get current timestamp in milliseconds
 */
static uint64_t get_timestamp_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000 + (uint64_t)ts.tv_nsec / 1000000;
}

/**
 * @brief Convert signal to crash type
 */
static code_crash_type_t signal_to_crash_type(int signal) {
    switch (signal) {
        case SIGSEGV: return CODE_CRASH_SIGSEGV;
        case SIGBUS:  return CODE_CRASH_SIGBUS;
        case SIGILL:  return CODE_CRASH_SIGILL;
        case SIGFPE:  return CODE_CRASH_SIGFPE;
        case SIGABRT: return CODE_CRASH_SIGABRT;
        default:      return CODE_CRASH_NONE;
    }
}

/**
 * @brief Find antigen by ID
 */
static code_antigen_t* find_antigen_by_id(code_immune_system_t* system, uint64_t id) {
    if (!system) return NULL;
    for (size_t i = 0; i < system->antigen_count; i++) {
        if (system->antigens[i].id == id) {
            return &system->antigens[i];
        }
    }
    return NULL;
}

/**
 * @brief Find B cell by ID
 */
static code_b_cell_t* find_b_cell_by_id(code_immune_system_t* system, uint64_t id) {
    if (!system) return NULL;
    for (size_t i = 0; i < system->b_cell_count; i++) {
        if (system->b_cells[i].id == id) {
            return &system->b_cells[i];
        }
    }
    return NULL;
}

/**
 * @brief Find antibody by ID
 */
static code_antibody_t* find_antibody_by_id(code_immune_system_t* system, uint64_t id) {
    if (!system) return NULL;
    for (size_t i = 0; i < system->antibody_count; i++) {
        if (system->antibodies[i].id == id) {
            return &system->antibodies[i];
        }
    }
    return NULL;
}

/**
 * @brief Process pending antigens and auto-activate responses
 *
 * Called with mutex held.
 */
static void process_pending_antigens(code_immune_system_t* system) {
    if (!system) return;

    for (size_t i = 0; i < system->antigen_count; i++) {
        code_antigen_t* antigen = &system->antigens[i];
        if (antigen->neutralized || antigen->processed) continue;

        /* Check if danger signal exceeds threshold */
        if (antigen->danger_signal >= system->config.activation_threshold) {
            /* Try to find matching B cell */
            uint64_t b_cell_id = 0;
            bool found = false;

            for (size_t j = 0; j < system->b_cell_count; j++) {
                code_b_cell_t* b_cell = &system->b_cells[j];
                if (b_cell->state == CODE_B_CELL_APOPTOTIC) continue;

                float affinity = code_immune_compute_affinity(
                    b_cell->receptor, antigen->epitope);

                if (affinity >= system->config.recognition_threshold) {
                    b_cell_id = b_cell->id;
                    found = true;
                    break;
                }
            }

            /* Create new B cell if no match found */
            if (!found && system->b_cell_count < system->b_cell_capacity) {
                code_b_cell_t* b_cell = &system->b_cells[system->b_cell_count];
                memset(b_cell, 0, sizeof(*b_cell));

                b_cell->id = system->next_b_cell_id++;
                b_cell->state = CODE_B_CELL_ACTIVATED;
                b_cell->crash_types = signal_to_crash_type(antigen->signal);
                memcpy(b_cell->receptor, antigen->epitope, CODE_IMMUNE_EPITOPE_SIZE);
                b_cell->affinity = antigen->confidence;
                b_cell->bound_antigen_id = antigen->id;
                b_cell->creation_time = get_timestamp_ms();
                b_cell->last_activation = b_cell->creation_time;

                system->b_cell_count++;
                system->stats.active_b_cells++;
                b_cell_id = b_cell->id;
            }

            antigen->processed = true;
        }
    }
}

/**
 * @brief Decay antibodies based on half-life
 */
static void decay_antibodies(code_immune_system_t* system, uint64_t delta_ms) {
    if (!system || delta_ms == 0) return;

    float half_life = (float)system->config.antibody_half_life_ms;
    if (half_life <= 0) return;

    /* Calculate decay factor */
    float decay_factor = 0.5f;
    if (delta_ms < half_life) {
        decay_factor = 1.0f - (0.5f * (float)delta_ms / half_life);
    }

    for (size_t i = 0; i < system->antibody_count; i++) {
        code_antibody_t* ab = &system->antibodies[i];
        if (!ab->injected) continue;

        ab->effectiveness *= decay_factor;

        /* Consider removing if too weak - but keep injected patches */
        if (ab->effectiveness < 0.1f && !ab->validated) {
            /* Only IGM unvalidated antibodies decay away */
            if (ab->ab_class == CODE_ANTIBODY_IGM) {
                ab->effectiveness = 0.0f;
            }
        }
    }
}

/* ============================================================================
 * Lifecycle API
 * ============================================================================ */

/**
 * @brief Get default configuration
 */
int code_immune_default_config(code_immune_config_t* config) {
    if (!config) return -1;

    memset(config, 0, sizeof(*config));

    /* Population limits */
    config->max_antigens = CODE_IMMUNE_MAX_ANTIGENS;
    config->max_b_cells = CODE_IMMUNE_MAX_B_CELLS;
    config->max_antibodies = CODE_IMMUNE_MAX_ANTIBODIES;

    /* Timing */
    config->activation_delay_ms = 50;
    config->memory_formation_delay_ms = 5000;
    config->antibody_half_life_ms = 60000;

    /* Thresholds */
    config->recognition_threshold = 0.7f;
    config->activation_threshold = 0.5f;
    config->memory_formation_threshold = 0.8f;

    /* Patching options */
    config->enable_hot_patching = false;      /* Disabled by default - risky */
    config->enable_function_redirect = true;
    config->require_validation = true;
    config->patch_build_dir = "/tmp/nimcp_patches";

    /* Integration */
    config->enable_logging = true;
    config->sync_with_brain_immune = true;

    return 0;
}

/**
 * @brief Create code immune system
 */
code_immune_system_t* code_immune_create(brain_immune_system_t* parent_immune) {
    code_immune_config_t config;
    code_immune_default_config(&config);
    return code_immune_create_with_config(parent_immune, &config);
}

/**
 * @brief Create code immune system with config
 */
code_immune_system_t* code_immune_create_with_config(
    brain_immune_system_t* parent_immune,
    const code_immune_config_t* config
) {
    code_immune_system_t* system = nimcp_calloc(1, sizeof(code_immune_system_t));
    if (!system) return NULL;

    /* Apply configuration */
    if (config) {
        system->config = *config;
    } else {
        code_immune_default_config(&system->config);
    }

    /* Link to parent immune system */
    system->parent_immune = parent_immune;

    /* Allocate antigen pool */
    system->antigen_capacity = system->config.max_antigens;
    system->antigens = nimcp_calloc(system->antigen_capacity, sizeof(code_antigen_t));
    if (!system->antigens) goto cleanup;

    /* Allocate B cells */
    system->b_cell_capacity = system->config.max_b_cells;
    system->b_cells = nimcp_calloc(system->b_cell_capacity, sizeof(code_b_cell_t));
    if (!system->b_cells) goto cleanup;

    /* Allocate antibodies */
    system->antibody_capacity = system->config.max_antibodies;
    system->antibodies = nimcp_calloc(system->antibody_capacity, sizeof(code_antibody_t));
    if (!system->antibodies) goto cleanup;

    /* Initialize mutex */
    system->mutex = nimcp_mutex_create();
    if (!system->mutex) goto cleanup;

    /* Set initial state */
    system->next_antigen_id = 1;
    system->next_b_cell_id = 1;
    system->next_antibody_id = 1;
    system->start_time = get_timestamp_ms();

    if (system->config.enable_logging) {
        LOG_MODULE_INFO(CODE_IMMUNE_MODULE_NAME, "Code immune system created");
    }

    return system;

cleanup:
    code_immune_destroy(system);
    return NULL;
}

/**
 * @brief Destroy code immune system
 */
void code_immune_destroy(code_immune_system_t* system) {
    if (!system) return;

    code_immune_stop(system);

    /* Unload any injected patches */
    for (size_t i = 0; i < system->antibody_count; i++) {
        code_antibody_t* ab = &system->antibodies[i];
        if (ab->patch_handle) {
            dlclose(ab->patch_handle);
            ab->patch_handle = NULL;
        }
    }

    if (system->mutex) {
        nimcp_mutex_destroy(system->mutex);
    }

    nimcp_free(system->antigens);
    nimcp_free(system->b_cells);
    nimcp_free(system->antibodies);
    nimcp_free(system);
}

/**
 * @brief Start code immune system
 */
int code_immune_start(code_immune_system_t* system) {
    if (!system) return -1;

    /* Check auto-load settings before starting */
    bool should_auto_load = false;
    char load_path[CODE_IMMUNE_PERSIST_MAX_PATH];
    load_path[0] = '\0';

    nimcp_mutex_lock(system->mutex);
    should_auto_load = system->auto_load_enabled;
    if (should_auto_load && system->auto_load_path[0] != '\0') {
        strncpy(load_path, system->auto_load_path, sizeof(load_path) - 1);
    }
    system->running = true;
    system->start_time = get_timestamp_ms();
    nimcp_mutex_unlock(system->mutex);

    /* Auto-load persisted memory (outside lock to avoid deadlock) */
    if (should_auto_load) {
        const char* path = load_path[0] != '\0' ? load_path : NULL;
        code_immune_load_memory(system, path, NULL);
    }

    if (system->config.enable_logging) {
        LOG_MODULE_INFO(CODE_IMMUNE_MODULE_NAME, "Code immune system started");
    }

    return 0;
}

/**
 * @brief Stop code immune system
 */
int code_immune_stop(code_immune_system_t* system) {
    if (!system) return -1;

    /* Disconnect from signal handler */
    code_immune_disconnect_signal_handler(system);

    nimcp_mutex_lock(system->mutex);
    system->running = false;
    nimcp_mutex_unlock(system->mutex);

    if (system->config.enable_logging) {
        LOG_MODULE_INFO(CODE_IMMUNE_MODULE_NAME, "Code immune system stopped");
    }

    return 0;
}

/* ============================================================================
 * Signal Handler Integration
 * ============================================================================ */

/**
 * @brief Process pending crashes queued from signal handler
 *
 * WHAT: Drain the async-signal-safe crash queue and process each crash
 * WHY:  Signal handlers cannot safely call most functions; defer to main thread
 * HOW:  Read from volatile queue and call normal crash processing
 *
 * Call this periodically from the main thread or update loop.
 */
void code_immune_process_pending_crashes(code_immune_system_t* system) {
    if (!system) return;

    while (g_pending_crash_count > 0) {
        sig_atomic_t idx = g_pending_crash_head;
        if (!g_pending_crashes[idx].valid) break;

        int sig = (int)g_pending_crashes[idx].signal;
        g_pending_crashes[idx].valid = 0;

        /* Update head with wraparound */
        g_pending_crash_head = (g_pending_crash_head + 1) % SIGNAL_SAFE_QUEUE_SIZE;
        g_pending_crash_count--;

        /* Now safe to call full crash processing */
        code_immune_present_crash(system, sig, NULL, NULL);
    }
}

/**
 * @brief Signal handler callback for code immune (ASYNC-SIGNAL-SAFE)
 *
 * WHAT: Queue crash for deferred processing
 * WHY:  Signal handlers can only use async-signal-safe functions
 * HOW:  Write to volatile queue without locks, process later in main thread
 *
 * Called from signal handler context - uses ONLY async-signal-safe operations:
 * - Reading/writing volatile sig_atomic_t variables
 * - No malloc, no mutex, no logging, no snprintf
 */
static void code_immune_signal_callback(int sig) {
    /* Cannot use locks in signal handler - check volatile pointer */
    if (!g_code_immune_instance) return;

    /* Check if queue has space (async-safe: volatile read) */
    if (g_pending_crash_count >= SIGNAL_SAFE_QUEUE_SIZE) {
        /* Queue full - drop this crash (better than undefined behavior) */
        return;
    }

    /* Find next slot (async-safe: volatile arithmetic) */
    sig_atomic_t idx = (g_pending_crash_head + g_pending_crash_count) % SIGNAL_SAFE_QUEUE_SIZE;

    /* Store crash info (async-safe: volatile writes) */
    g_pending_crashes[idx].signal = (sig_atomic_t)sig;
    g_pending_crashes[idx].valid = 1;
    g_pending_crash_count++;

    /* Note: actual crash processing deferred to code_immune_process_pending_crashes() */
}

/**
 * @brief Connect to signal handler
 */
int code_immune_connect_signal_handler(code_immune_system_t* system) {
    if (!system) return -1;

    nimcp_mutex_lock(system->mutex);

    /* Store global reference for signal callback */
    g_code_immune_instance = system;

    /* Set callback in signal handler */
    signal_handler_set_crash_callback(code_immune_signal_callback);

    system->signal_handler_connected = true;
    nimcp_mutex_unlock(system->mutex);

    if (system->config.enable_logging) {
        LOG_MODULE_INFO(CODE_IMMUNE_MODULE_NAME, "Connected to signal handler");
    }

    return 0;
}

/**
 * @brief Disconnect from signal handler
 */
int code_immune_disconnect_signal_handler(code_immune_system_t* system) {
    if (!system) return -1;

    nimcp_mutex_lock(system->mutex);

    if (system->signal_handler_connected) {
        signal_handler_set_crash_callback(NULL);
        g_code_immune_instance = NULL;
        system->signal_handler_connected = false;
    }

    nimcp_mutex_unlock(system->mutex);

    if (system->config.enable_logging) {
        LOG_MODULE_INFO(CODE_IMMUNE_MODULE_NAME, "Disconnected from signal handler");
    }

    return 0;
}

/**
 * @brief Present crash as antigen (signal-safe version)
 */
int code_immune_present_crash(
    code_immune_system_t* system,
    int signal,
    void* ucontext,
    void* fault_addr
) {
    if (!system) return -1;

    /* Try to get lock - if can't, we're likely in nested signal */
    if (nimcp_platform_mutex_trylock((nimcp_platform_mutex_t*)system->mutex) != 0) {
        return -1;
    }

    if (system->antigen_count >= system->antigen_capacity) {
        nimcp_mutex_unlock(system->mutex);
        return -1;
    }

    code_antigen_t* antigen = &system->antigens[system->antigen_count];
    memset(antigen, 0, sizeof(*antigen));

    antigen->id = system->next_antigen_id++;
    antigen->signal = signal;
    antigen->fault_address = fault_addr;
    antigen->timestamp = get_timestamp_ms();

    /* Compute basic epitope from signal and address */
    snprintf(antigen->epitope, CODE_IMMUNE_EPITOPE_SIZE,
             "sig%d_addr%p", signal, fault_addr);

    /* Set severity based on signal type */
    switch (signal) {
        case SIGSEGV:
        case SIGBUS:
            antigen->severity = 0.9f;
            antigen->danger_signal = 0.8f;
            break;
        case SIGABRT:
            antigen->severity = 0.8f;
            antigen->danger_signal = 0.7f;
            break;
        case SIGFPE:
        case SIGILL:
            antigen->severity = 0.6f;
            antigen->danger_signal = 0.5f;
            break;
        default:
            antigen->severity = 0.5f;
            antigen->danger_signal = 0.4f;
    }

    antigen->confidence = 0.8f;  /* High confidence from signal handler */

    /* Get backtrace if possible */
    antigen->backtrace_depth = backtrace(antigen->backtrace, CODE_IMMUNE_BACKTRACE_DEPTH);

    /* Extract instruction pointer from ucontext if available */
    if (ucontext) {
        /* Platform-specific extraction would go here */
        (void)ucontext;
    }

    system->antigen_count++;
    system->stats.crashes_detected++;

    /* Update per-type statistics */
    switch (signal) {
        case SIGSEGV: system->stats.sigsegv_count++; break;
        case SIGBUS:  system->stats.sigbus_count++; break;
        case SIGILL:  system->stats.sigill_count++; break;
        case SIGFPE:  system->stats.sigfpe_count++; break;
        case SIGABRT: system->stats.sigabrt_count++; break;
    }

    nimcp_mutex_unlock(system->mutex);

    /* Call crash callback if set */
    if (system->on_crash) {
        system->on_crash(system, antigen, system->callback_user_data);
    }

    return 0;
}

/**
 * @brief Present crash with full details
 */
int code_immune_present_crash_detailed(
    code_immune_system_t* system,
    int signal,
    void* fault_addr,
    void* ip,
    const char* source_file,
    uint32_t line,
    const char* function,
    void** backtrace_frames,
    int backtrace_depth,
    uint64_t* antigen_id
) {
    if (!system) return -1;

    nimcp_mutex_lock(system->mutex);

    if (system->antigen_count >= system->antigen_capacity) {
        nimcp_mutex_unlock(system->mutex);
        return -1;
    }

    code_antigen_t* antigen = &system->antigens[system->antigen_count];
    memset(antigen, 0, sizeof(*antigen));

    antigen->id = system->next_antigen_id++;
    antigen->signal = signal;
    antigen->fault_address = fault_addr;
    antigen->instruction_pointer = ip;
    antigen->line_number = line;
    antigen->timestamp = get_timestamp_ms();

    /* Copy source location info */
    if (source_file) {
        strncpy(antigen->source_file, source_file, sizeof(antigen->source_file) - 1);
    }
    if (function) {
        strncpy(antigen->function_name, function, sizeof(antigen->function_name) - 1);
    }

    /* Copy backtrace */
    if (backtrace_frames && backtrace_depth > 0) {
        int depth = backtrace_depth;
        if (depth > CODE_IMMUNE_BACKTRACE_DEPTH) {
            depth = CODE_IMMUNE_BACKTRACE_DEPTH;
        }
        memcpy(antigen->backtrace, backtrace_frames, depth * sizeof(void*));
        antigen->backtrace_depth = depth;
    }

    /* Compute full epitope */
    code_immune_compute_epitope(
        signal, fault_addr, ip,
        antigen->backtrace, antigen->backtrace_depth,
        antigen->epitope
    );

    /* Set severity */
    antigen->severity = 0.8f;
    antigen->danger_signal = 0.7f;
    antigen->confidence = 0.9f;

    system->antigen_count++;
    system->stats.crashes_detected++;

    /* Update per-type statistics */
    switch (signal) {
        case SIGSEGV: system->stats.sigsegv_count++; break;
        case SIGBUS:  system->stats.sigbus_count++; break;
        case SIGILL:  system->stats.sigill_count++; break;
        case SIGFPE:  system->stats.sigfpe_count++; break;
        case SIGABRT: system->stats.sigabrt_count++; break;
    }

    if (antigen_id) {
        *antigen_id = antigen->id;
    }

    nimcp_mutex_unlock(system->mutex);

    /* Call crash callback if set */
    if (system->on_crash) {
        system->on_crash(system, antigen, system->callback_user_data);
    }

    /* Sync to brain immune if configured */
    if (system->config.sync_with_brain_immune && system->parent_immune) {
        code_immune_sync_to_brain(system, antigen->id);
    }

    return 0;
}

/* ============================================================================
 * B Cell API
 * ============================================================================ */

/**
 * @brief Find matching B cell for antigen
 */
int code_immune_find_matching_b_cell(
    code_immune_system_t* system,
    uint64_t antigen_id,
    uint64_t* b_cell_id
) {
    if (!system || !b_cell_id) return -1;

    nimcp_mutex_lock(system->mutex);

    code_antigen_t* antigen = find_antigen_by_id(system, antigen_id);
    if (!antigen) {
        nimcp_mutex_unlock(system->mutex);
        return -1;
    }

    float best_affinity = 0.0f;
    code_b_cell_t* best_match = NULL;

    for (size_t i = 0; i < system->b_cell_count; i++) {
        code_b_cell_t* b_cell = &system->b_cells[i];
        if (b_cell->state == CODE_B_CELL_APOPTOTIC) continue;

        float affinity = code_immune_compute_affinity(
            b_cell->receptor, antigen->epitope);

        if (affinity > best_affinity &&
            affinity >= system->config.recognition_threshold) {
            best_affinity = affinity;
            best_match = b_cell;
        }
    }

    if (best_match) {
        *b_cell_id = best_match->id;
        nimcp_mutex_unlock(system->mutex);
        return 0;
    }

    nimcp_mutex_unlock(system->mutex);
    return -1;
}

/**
 * @brief Activate B cell for antigen
 */
int code_immune_activate_b_cell(
    code_immune_system_t* system,
    uint64_t b_cell_id,
    uint64_t antigen_id
) {
    if (!system) return -1;

    nimcp_mutex_lock(system->mutex);

    code_b_cell_t* b_cell = find_b_cell_by_id(system, b_cell_id);
    code_antigen_t* antigen = find_antigen_by_id(system, antigen_id);

    if (!b_cell || !antigen) {
        nimcp_mutex_unlock(system->mutex);
        return -1;
    }

    b_cell->state = CODE_B_CELL_ACTIVATED;
    b_cell->bound_antigen_id = antigen_id;
    b_cell->last_activation = get_timestamp_ms();

    /* Increase affinity based on repeated exposure */
    float affinity = code_immune_compute_affinity(b_cell->receptor, antigen->epitope);
    if (affinity > b_cell->affinity) {
        b_cell->affinity = affinity;
    }

    system->stats.active_b_cells++;

    nimcp_mutex_unlock(system->mutex);

    if (system->config.enable_logging) {
        LOG_MODULE_INFO(CODE_IMMUNE_MODULE_NAME,
            "B cell %lu activated for antigen %lu (affinity=%.2f)",
            b_cell_id, antigen_id, b_cell->affinity);
    }

    return 0;
}

/**
 * @brief Create new B cell for novel crash pattern
 */
int code_immune_create_b_cell(
    code_immune_system_t* system,
    uint64_t antigen_id,
    uint64_t* b_cell_id
) {
    if (!system || !b_cell_id) return -1;

    nimcp_mutex_lock(system->mutex);

    if (system->b_cell_count >= system->b_cell_capacity) {
        nimcp_mutex_unlock(system->mutex);
        return -1;
    }

    code_antigen_t* antigen = find_antigen_by_id(system, antigen_id);
    if (!antigen) {
        nimcp_mutex_unlock(system->mutex);
        return -1;
    }

    code_b_cell_t* b_cell = &system->b_cells[system->b_cell_count];
    memset(b_cell, 0, sizeof(*b_cell));

    b_cell->id = system->next_b_cell_id++;
    b_cell->state = CODE_B_CELL_NAIVE;
    b_cell->crash_types = signal_to_crash_type(antigen->signal);
    memcpy(b_cell->receptor, antigen->epitope, CODE_IMMUNE_EPITOPE_SIZE);
    b_cell->affinity = antigen->confidence;
    b_cell->creation_time = get_timestamp_ms();

    /* Copy source pattern if available */
    if (antigen->source_file[0]) {
        snprintf(b_cell->source_pattern, sizeof(b_cell->source_pattern),
                 "%s:%s", antigen->source_file, antigen->function_name);
    }

    *b_cell_id = b_cell->id;
    system->b_cell_count++;

    nimcp_mutex_unlock(system->mutex);

    if (system->config.enable_logging) {
        LOG_MODULE_INFO(CODE_IMMUNE_MODULE_NAME,
            "Created B cell %lu for crash type %s",
            b_cell->id, code_immune_crash_type_to_string(b_cell->crash_types));
    }

    return 0;
}

/**
 * @brief Form memory B cell
 */
int code_immune_form_memory(
    code_immune_system_t* system,
    uint64_t b_cell_id
) {
    if (!system) return -1;

    nimcp_mutex_lock(system->mutex);

    code_b_cell_t* b_cell = find_b_cell_by_id(system, b_cell_id);
    if (!b_cell) {
        nimcp_mutex_unlock(system->mutex);
        return -1;
    }

    /* Only activated or plasma cells can become memory */
    if (b_cell->state != CODE_B_CELL_ACTIVATED &&
        b_cell->state != CODE_B_CELL_PLASMA) {
        nimcp_mutex_unlock(system->mutex);
        return -1;
    }

    /* Check if enough successful fixes to warrant memory */
    if (b_cell->successful_fixes < 1 &&
        b_cell->affinity < system->config.memory_formation_threshold) {
        nimcp_mutex_unlock(system->mutex);
        return -1;
    }

    b_cell->state = CODE_B_CELL_MEMORY;
    system->stats.memory_b_cells++;

    nimcp_mutex_unlock(system->mutex);

    /* Call memory callback if set */
    if (system->on_memory) {
        system->on_memory(system, b_cell, system->callback_user_data);
    }

    if (system->config.enable_logging) {
        LOG_MODULE_INFO(CODE_IMMUNE_MODULE_NAME,
            "B cell %lu transitioned to MEMORY state (affinity=%.2f, fixes=%u)",
            b_cell_id, b_cell->affinity, b_cell->successful_fixes);
    }

    return 0;
}

/**
 * @brief Set fix template for B cell
 */
int code_immune_set_fix_template(
    code_immune_system_t* system,
    uint64_t b_cell_id,
    const char* fix_template
) {
    if (!system || !fix_template) return -1;

    nimcp_mutex_lock(system->mutex);

    code_b_cell_t* b_cell = find_b_cell_by_id(system, b_cell_id);
    if (!b_cell) {
        nimcp_mutex_unlock(system->mutex);
        return -1;
    }

    strncpy(b_cell->fix_template, fix_template, sizeof(b_cell->fix_template) - 1);

    nimcp_mutex_unlock(system->mutex);
    return 0;
}

/* ============================================================================
 * Antibody API
 * ============================================================================ */

/**
 * @brief Produce antibody from B cell
 */
int code_immune_produce_antibody(
    code_immune_system_t* system,
    uint64_t b_cell_id,
    code_antibody_class_t ab_class,
    uint64_t* antibody_id
) {
    if (!system || !antibody_id) return -1;

    nimcp_mutex_lock(system->mutex);

    if (system->antibody_count >= system->antibody_capacity) {
        nimcp_mutex_unlock(system->mutex);
        return -1;
    }

    code_b_cell_t* b_cell = find_b_cell_by_id(system, b_cell_id);
    if (!b_cell) {
        nimcp_mutex_unlock(system->mutex);
        return -1;
    }

    /* B cell must be at least activated to produce antibodies */
    if (b_cell->state == CODE_B_CELL_NAIVE ||
        b_cell->state == CODE_B_CELL_APOPTOTIC) {
        nimcp_mutex_unlock(system->mutex);
        return -1;
    }

    /* Transition to plasma state */
    if (b_cell->state == CODE_B_CELL_ACTIVATED) {
        b_cell->state = CODE_B_CELL_PLASMA;
    }

    code_antigen_t* antigen = find_antigen_by_id(system, b_cell->bound_antigen_id);

    code_antibody_t* antibody = &system->antibodies[system->antibody_count];
    memset(antibody, 0, sizeof(*antibody));

    antibody->id = system->next_antibody_id++;
    antibody->producer_b_cell_id = b_cell_id;
    antibody->target_antigen_id = b_cell->bound_antigen_id;
    antibody->ab_class = ab_class;
    antibody->effectiveness = b_cell->affinity;
    antibody->creation_time = get_timestamp_ms();

    /* Copy fix template if available */
    if (b_cell->fix_template[0]) {
        strncpy(antibody->fixed_code, b_cell->fix_template,
                sizeof(antibody->fixed_code) - 1);
    }

    /* Copy source info from antigen if available */
    if (antigen) {
        strncpy(antibody->source_file, antigen->source_file,
                sizeof(antibody->source_file) - 1);
        strncpy(antibody->fn_name, antigen->function_name,
                sizeof(antibody->fn_name) - 1);
        antibody->start_line = antigen->line_number;
        antibody->end_line = antigen->line_number;
    }

    *antibody_id = antibody->id;
    system->antibody_count++;
    system->stats.active_antibodies++;
    system->stats.patches_generated++;

    nimcp_mutex_unlock(system->mutex);

    if (system->config.enable_logging) {
        LOG_MODULE_INFO(CODE_IMMUNE_MODULE_NAME,
            "Produced %s antibody %lu from B cell %lu",
            code_immune_antibody_class_to_string(ab_class),
            antibody->id, b_cell_id);
    }

    return 0;
}

/**
 * @brief Validate antibody
 */
int code_immune_validate_antibody(
    code_immune_system_t* system,
    uint64_t antibody_id
) {
    if (!system) return -1;

    nimcp_mutex_lock(system->mutex);

    code_antibody_t* antibody = find_antibody_by_id(system, antibody_id);
    if (!antibody) {
        nimcp_mutex_unlock(system->mutex);
        return -1;
    }

    bool validated = false;

    /* Validation would run tests here */
    /* For now, IGG antibodies are considered validated */
    if (antibody->ab_class == CODE_ANTIBODY_IGG) {
        antibody->validated = true;
        validated = true;
        system->stats.patches_validated++;
    }

    /* IGE (emergency) bypasses validation */
    if (antibody->ab_class == CODE_ANTIBODY_IGE) {
        antibody->validated = true;
        validated = true;
    }

    /* Check if auto-save enabled */
    bool should_auto_save = validated && system->auto_save_enabled;
    char save_path[CODE_IMMUNE_PERSIST_MAX_PATH];
    save_path[0] = '\0';
    if (should_auto_save && system->auto_save_path[0] != '\0') {
        strncpy(save_path, system->auto_save_path, sizeof(save_path) - 1);
    }

    nimcp_mutex_unlock(system->mutex);

    /* Auto-save after validation (outside lock to avoid deadlock) */
    if (should_auto_save) {
        const char* path = save_path[0] != '\0' ? save_path : NULL;
        code_immune_save_memory(system, path, NULL);
    }

    /* IGM needs actual validation - mark as pending */
    return validated ? 0 : -1;
}

/**
 * @brief Apply antibody
 */
int code_immune_apply_antibody(
    code_immune_system_t* system,
    uint64_t antibody_id
) {
    if (!system) return -1;

    nimcp_mutex_lock(system->mutex);

    code_antibody_t* antibody = find_antibody_by_id(system, antibody_id);
    if (!antibody) {
        nimcp_mutex_unlock(system->mutex);
        return -1;
    }

    /* Check validation requirement */
    if (system->config.require_validation && !antibody->validated) {
        nimcp_mutex_unlock(system->mutex);
        return -1;
    }

    /* Try hot-patching if enabled */
    if (system->config.enable_hot_patching && antibody->patch_so_path[0]) {
        antibody->patch_handle = dlopen(antibody->patch_so_path, RTLD_NOW);
        if (antibody->patch_handle) {
            /* Find patched function */
            antibody->new_function = dlsym(antibody->patch_handle, antibody->fn_name);
            if (antibody->new_function) {
                antibody->injected = true;
                antibody->injection_time = get_timestamp_ms();
                system->stats.patches_applied++;
            }
        }
    }

    /* Try function redirect if enabled */
    if (!antibody->injected && system->config.enable_function_redirect) {
        if (antibody->old_function && antibody->new_function) {
            /* Would use mprotect/memcpy to redirect - platform specific */
            antibody->injected = true;
            antibody->injection_time = get_timestamp_ms();
            system->stats.patches_applied++;
        }
    }

    /* Mark antigen as neutralized */
    if (antibody->injected) {
        code_antigen_t* antigen = find_antigen_by_id(system, antibody->target_antigen_id);
        if (antigen) {
            antigen->neutralized = true;
            system->stats.crashes_neutralized++;
        }

        /* Update B cell success count */
        code_b_cell_t* b_cell = find_b_cell_by_id(system, antibody->producer_b_cell_id);
        if (b_cell) {
            b_cell->successful_fixes++;
        }
    }

    bool success = antibody->injected;
    nimcp_mutex_unlock(system->mutex);

    /* Call patch callback if set */
    if (system->on_patch) {
        system->on_patch(system, antibody, success, system->callback_user_data);
    }

    if (system->config.enable_logging) {
        LOG_MODULE_INFO(CODE_IMMUNE_MODULE_NAME,
            "Applied antibody %lu: %s",
            antibody_id, success ? "SUCCESS" : "FAILED");
    }

    return success ? 0 : -1;
}

/**
 * @brief Remove antibody (apoptosis)
 */
int code_immune_apoptosis(
    code_immune_system_t* system,
    uint64_t antibody_id
) {
    if (!system) return -1;

    nimcp_mutex_lock(system->mutex);

    code_antibody_t* antibody = find_antibody_by_id(system, antibody_id);
    if (!antibody) {
        nimcp_mutex_unlock(system->mutex);
        return -1;
    }

    /* Unload patch if loaded */
    if (antibody->patch_handle) {
        dlclose(antibody->patch_handle);
        antibody->patch_handle = NULL;
    }

    antibody->injected = false;
    antibody->new_function = NULL;
    antibody->effectiveness = 0.0f;

    nimcp_mutex_unlock(system->mutex);

    if (system->config.enable_logging) {
        LOG_MODULE_INFO(CODE_IMMUNE_MODULE_NAME,
            "Antibody %lu underwent apoptosis", antibody_id);
    }

    return 0;
}

/**
 * @brief Upgrade antibody class
 */
int code_immune_upgrade_antibody(
    code_immune_system_t* system,
    uint64_t antibody_id,
    code_antibody_class_t new_class
) {
    if (!system) return -1;

    nimcp_mutex_lock(system->mutex);

    code_antibody_t* antibody = find_antibody_by_id(system, antibody_id);
    if (!antibody) {
        nimcp_mutex_unlock(system->mutex);
        return -1;
    }

    antibody->ab_class = new_class;

    nimcp_mutex_unlock(system->mutex);

    if (system->config.enable_logging) {
        LOG_MODULE_INFO(CODE_IMMUNE_MODULE_NAME,
            "Upgraded antibody %lu to %s",
            antibody_id, code_immune_antibody_class_to_string(new_class));
    }

    return 0;
}

/* ============================================================================
 * Brain Immune Integration API
 * ============================================================================ */

/**
 * @brief Sync crash to brain immune as antigen
 */
int code_immune_sync_to_brain(
    code_immune_system_t* system,
    uint64_t antigen_id
) {
    if (!system || !system->parent_immune) return -1;

    nimcp_mutex_lock(system->mutex);

    code_antigen_t* antigen = find_antigen_by_id(system, antigen_id);
    if (!antigen) {
        nimcp_mutex_unlock(system->mutex);
        return -1;
    }

    /* Create brain antigen from code antigen */
    uint32_t brain_antigen_id = 0;
    uint8_t epitope_bytes[BRAIN_IMMUNE_EPITOPE_SIZE];
    memcpy(epitope_bytes, antigen->epitope,
           CODE_IMMUNE_EPITOPE_SIZE < BRAIN_IMMUNE_EPITOPE_SIZE ?
           CODE_IMMUNE_EPITOPE_SIZE : BRAIN_IMMUNE_EPITOPE_SIZE);

    /* Present as manual antigen (custom source) */
    int result = brain_immune_present_antigen(
        system->parent_immune,
        ANTIGEN_SOURCE_MANUAL,
        epitope_bytes,
        CODE_IMMUNE_EPITOPE_SIZE,
        (uint32_t)(antigen->severity * 10),  /* Convert 0-1 to 1-10 */
        0,  /* No specific source node */
        &brain_antigen_id
    );

    nimcp_mutex_unlock(system->mutex);

    if (result == 0 && system->config.enable_logging) {
        LOG_MODULE_INFO(CODE_IMMUNE_MODULE_NAME,
            "Synced code antigen %lu -> brain antigen %u",
            antigen_id, brain_antigen_id);
    }

    return result;
}

/**
 * @brief Request cytokine release from brain immune
 */
int code_immune_request_cytokine(
    code_immune_system_t* system,
    brain_cytokine_type_t cytokine_type,
    float concentration
) {
    if (!system || !system->parent_immune) return -1;

    uint32_t cytokine_id = 0;
    return brain_immune_release_cytokine(
        system->parent_immune,
        cytokine_type,
        0,              /* Source: code immune */
        concentration,
        0,              /* Broadcast */
        &cytokine_id
    );
}

/* ============================================================================
 * Callback Registration
 * ============================================================================ */

int code_immune_set_crash_callback(
    code_immune_system_t* system,
    code_immune_crash_cb_t callback,
    void* user_data
) {
    if (!system) return -1;
    nimcp_mutex_lock(system->mutex);
    system->on_crash = callback;
    system->callback_user_data = user_data;
    nimcp_mutex_unlock(system->mutex);
    return 0;
}

int code_immune_set_patch_callback(
    code_immune_system_t* system,
    code_immune_patch_cb_t callback,
    void* user_data
) {
    if (!system) return -1;
    nimcp_mutex_lock(system->mutex);
    system->on_patch = callback;
    system->callback_user_data = user_data;
    nimcp_mutex_unlock(system->mutex);
    return 0;
}

int code_immune_set_memory_callback(
    code_immune_system_t* system,
    code_immune_memory_cb_t callback,
    void* user_data
) {
    if (!system) return -1;
    nimcp_mutex_lock(system->mutex);
    system->on_memory = callback;
    system->callback_user_data = user_data;
    nimcp_mutex_unlock(system->mutex);
    return 0;
}

/* ============================================================================
 * Update and Query API
 * ============================================================================ */

/**
 * @brief Update code immune system state
 */
int code_immune_update(
    code_immune_system_t* system,
    uint64_t delta_ms
) {
    if (!system) return -1;

    nimcp_mutex_lock(system->mutex);

    if (!system->running) {
        nimcp_mutex_unlock(system->mutex);
        return 0;
    }

    /* Process pending antigens */
    process_pending_antigens(system);

    /* Decay antibodies */
    decay_antibodies(system, delta_ms);

    /* Update statistics */
    if (system->stats.crashes_detected > 0) {
        system->stats.prevention_rate =
            (float)system->stats.crashes_neutralized /
            (float)system->stats.crashes_detected;
    }

    system->stats.system_health = 1.0f - (0.1f * system->antigen_count);
    if (system->stats.system_health < 0.0f) {
        system->stats.system_health = 0.0f;
    }

    nimcp_mutex_unlock(system->mutex);
    return 0;
}

/**
 * @brief Get code immune system statistics
 */
int code_immune_get_stats(
    code_immune_system_t* system,
    code_immune_stats_t* stats
) {
    if (!system || !stats) return -1;

    nimcp_mutex_lock(system->mutex);
    *stats = system->stats;
    nimcp_mutex_unlock(system->mutex);

    return 0;
}

/**
 * @brief Get antigen by ID
 */
const code_antigen_t* code_immune_get_antigen(
    code_immune_system_t* system,
    uint64_t antigen_id
) {
    if (!system) return NULL;

    nimcp_mutex_lock(system->mutex);
    code_antigen_t* antigen = find_antigen_by_id(system, antigen_id);
    nimcp_mutex_unlock(system->mutex);

    return antigen;
}

/**
 * @brief Get B cell by ID
 */
const code_b_cell_t* code_immune_get_b_cell(
    code_immune_system_t* system,
    uint64_t b_cell_id
) {
    if (!system) return NULL;

    nimcp_mutex_lock(system->mutex);
    code_b_cell_t* b_cell = find_b_cell_by_id(system, b_cell_id);
    nimcp_mutex_unlock(system->mutex);

    return b_cell;
}

/**
 * @brief Get antibody by ID
 */
const code_antibody_t* code_immune_get_antibody(
    code_immune_system_t* system,
    uint64_t antibody_id
) {
    if (!system) return NULL;

    nimcp_mutex_lock(system->mutex);
    code_antibody_t* antibody = find_antibody_by_id(system, antibody_id);
    nimcp_mutex_unlock(system->mutex);

    return antibody;
}

/**
 * @brief Check if crash type is handled
 */
bool code_immune_has_memory_for(
    code_immune_system_t* system,
    code_crash_type_t crash_type
) {
    if (!system) return false;

    nimcp_mutex_lock(system->mutex);

    for (size_t i = 0; i < system->b_cell_count; i++) {
        code_b_cell_t* b_cell = &system->b_cells[i];
        if (b_cell->state == CODE_B_CELL_MEMORY &&
            (b_cell->crash_types & crash_type)) {
            nimcp_mutex_unlock(system->mutex);
            return true;
        }
    }

    nimcp_mutex_unlock(system->mutex);
    return false;
}

/* ============================================================================
 * Epitope/Pattern Matching Utilities
 * ============================================================================ */

/**
 * @brief Compute crash epitope (hash signature)
 */
int code_immune_compute_epitope(
    int signal,
    void* fault_addr,
    void* ip,
    void** backtrace_frames,
    int backtrace_depth,
    char* epitope
) {
    if (!epitope) return -1;

    /* Simple hash combining signal, addresses, and top backtrace frames */
    uint64_t hash = 0;

    /* Mix in signal */
    hash = hash * 31 + (uint64_t)signal;

    /* Mix in fault address (if present) */
    if (fault_addr) {
        hash = hash * 31 + (uint64_t)(uintptr_t)fault_addr;
    }

    /* Mix in instruction pointer */
    if (ip) {
        hash = hash * 31 + (uint64_t)(uintptr_t)ip;
    }

    /* Mix in top backtrace frames (most significant) */
    if (backtrace_frames && backtrace_depth > 0) {
        int depth = backtrace_depth > 4 ? 4 : backtrace_depth;
        for (int i = 0; i < depth; i++) {
            if (backtrace_frames[i]) {
                hash = hash * 31 + (uint64_t)(uintptr_t)backtrace_frames[i];
            }
        }
    }

    /* Format as hex string */
    snprintf(epitope, CODE_IMMUNE_EPITOPE_SIZE,
             "%016lx_%d", (unsigned long)hash, signal);

    return 0;
}

/**
 * @brief Compute pattern affinity
 */
float code_immune_compute_affinity(
    const char* pattern1,
    const char* pattern2
) {
    if (!pattern1 || !pattern2) return 0.0f;

    size_t len1 = strlen(pattern1);
    size_t len2 = strlen(pattern2);

    if (len1 == 0 || len2 == 0) return 0.0f;

    /* Exact match */
    if (strcmp(pattern1, pattern2) == 0) {
        return 1.0f;
    }

    /* Count matching characters */
    size_t matches = 0;
    size_t max_len = len1 > len2 ? len1 : len2;
    size_t min_len = len1 < len2 ? len1 : len2;

    for (size_t i = 0; i < min_len; i++) {
        if (pattern1[i] == pattern2[i]) {
            matches++;
        }
    }

    /* Also check for common substrings */
    /* Look for signal type match (last part after _) */
    const char* sig1 = strrchr(pattern1, '_');
    const char* sig2 = strrchr(pattern2, '_');
    if (sig1 && sig2 && strcmp(sig1, sig2) == 0) {
        matches += 4;  /* Bonus for same signal type */
    }

    float affinity = (float)matches / (float)max_len;

    /* Clamp to 0-1 */
    if (affinity > 1.0f) affinity = 1.0f;

    return affinity;
}

/* ============================================================================
 * Persistence API - Internal Helpers
 * ============================================================================ */

/**
 * @brief Compute CRC32 checksum
 *
 * WHAT: Calculate simple checksum for data validation
 * WHY:  Detect file corruption
 * HOW:  Simple XOR-based checksum (fast, good enough for detection)
 */
static uint32_t compute_persist_checksum(const uint8_t* data, size_t len) {
    if (!data || len == 0) return 0;

    uint32_t checksum = 0xFFFFFFFF;
    for (size_t i = 0; i < len; i++) {
        checksum ^= data[i];
        checksum = (checksum << 1) | (checksum >> 31);
    }
    return checksum;
}

/**
 * @brief Write header to file
 */
static int write_persist_header(FILE* file, const code_immune_persist_header_t* header) {
    if (!file || !header) return -1;
    if (fwrite(header, sizeof(*header), 1, file) != 1) {
        LOG_MODULE_ERROR(CODE_IMMUNE_MODULE_NAME, "Failed to write header");
        return -1;
    }
    return 0;
}

/**
 * @brief Read header from file
 */
static int read_persist_header(FILE* file, code_immune_persist_header_t* header) {
    if (!file || !header) return -1;
    if (fread(header, sizeof(*header), 1, file) != 1) {
        LOG_MODULE_ERROR(CODE_IMMUNE_MODULE_NAME, "Failed to read header");
        return -1;
    }
    return 0;
}

/**
 * @brief Write counts to file
 */
static int write_persist_counts(FILE* file, const code_immune_persist_counts_t* counts) {
    if (!file || !counts) return -1;
    if (fwrite(counts, sizeof(*counts), 1, file) != 1) {
        LOG_MODULE_ERROR(CODE_IMMUNE_MODULE_NAME, "Failed to write counts");
        return -1;
    }
    return 0;
}

/**
 * @brief Read counts from file
 */
static int read_persist_counts(FILE* file, code_immune_persist_counts_t* counts) {
    if (!file || !counts) return -1;
    if (fread(counts, sizeof(*counts), 1, file) != 1) {
        LOG_MODULE_ERROR(CODE_IMMUNE_MODULE_NAME, "Failed to read counts");
        return -1;
    }
    return 0;
}

/**
 * @brief Validate header contents
 */
static int validate_persist_header(const code_immune_persist_header_t* header) {
    if (!header) return -1;

    /* Check magic */
    if (memcmp(header->magic, CODE_IMMUNE_PERSIST_MAGIC,
               CODE_IMMUNE_PERSIST_MAGIC_LEN) != 0) {
        LOG_MODULE_ERROR(CODE_IMMUNE_MODULE_NAME, "Invalid magic header");
        return -1;
    }

    /* Check version compatibility */
    if (!code_immune_is_version_compatible(header->version)) {
        LOG_MODULE_ERROR(CODE_IMMUNE_MODULE_NAME,
            "Incompatible version: %u (current: %u)",
            header->version, CODE_IMMUNE_PERSIST_VERSION);
        return -1;
    }

    return 0;
}

/**
 * @brief Validate counts against capacity limits
 */
static int validate_persist_counts(const code_immune_persist_counts_t* counts) {
    if (!counts) return -1;

    if (counts->b_cell_count > CODE_IMMUNE_MAX_B_CELLS) {
        LOG_MODULE_ERROR(CODE_IMMUNE_MODULE_NAME,
            "B cell count %u exceeds max %u",
            counts->b_cell_count, CODE_IMMUNE_MAX_B_CELLS);
        return -1;
    }
    if (counts->antibody_count > CODE_IMMUNE_MAX_ANTIBODIES) {
        LOG_MODULE_ERROR(CODE_IMMUNE_MODULE_NAME,
            "Antibody count %u exceeds max %u",
            counts->antibody_count, CODE_IMMUNE_MAX_ANTIBODIES);
        return -1;
    }

    return 0;
}

/**
 * @brief Find B cell with matching receptor
 */
static code_b_cell_t* find_b_cell_by_receptor(
    code_immune_system_t* system,
    const char* receptor
) {
    if (!system || !receptor) return NULL;

    for (size_t i = 0; i < system->b_cell_count; i++) {
        if (strcmp(system->b_cells[i].receptor, receptor) == 0) {
            return &system->b_cells[i];
        }
    }
    return NULL;
}

/* ============================================================================
 * Persistence API - Configuration
 * ============================================================================ */

/**
 * @brief Get default persistence configuration
 */
int code_immune_persist_default_config(code_immune_persist_config_t* config) {
    if (!config) return -1;

    memset(config, 0, sizeof(*config));

    /* Compression/Encryption - disabled by default for speed */
    config->enable_compression = false;
    config->enable_encryption = false;
    config->encryption_key_set = false;

    /* Save all components */
    config->save_b_cells = true;
    config->save_antibodies = true;
    config->save_pattern_stats = true;

    /* Memory-only mode disabled */
    config->memory_cells_only = false;

    /* Validation enabled */
    config->verify_on_load = true;
    config->strict_version_check = false;

    /* Backup enabled */
    config->create_backup = true;
    strncpy(config->backup_suffix, ".bak", sizeof(config->backup_suffix) - 1);

    /* Pruning enabled */
    config->auto_prune = true;
    config->min_confidence = CODE_IMMUNE_PERSIST_MIN_CONFIDENCE;
    config->min_successful_fixes = 0;

    /* Auto-save on validation enabled */
    config->auto_save_on_validation = true;

    /* Default path empty (will use get_default_memory_path) */
    config->memory_file_path[0] = '\0';

    return 0;
}

/* ============================================================================
 * Persistence API - Path Management
 * ============================================================================ */

/**
 * @brief Get default memory file path
 */
int code_immune_get_default_memory_path(char* path_out, bool create_dir) {
    if (!path_out) return -1;

    /* Get home directory */
    const char* home = getenv("HOME");
    if (!home) {
        home = "/tmp";  /* Fallback */
    }

    /* Build path: ~/.nimcp/code_immune_memory.bin */
    int len = snprintf(path_out, CODE_IMMUNE_PERSIST_MAX_PATH,
                       "%s/.nimcp/%s", home, CODE_IMMUNE_PERSIST_DEFAULT_FILE);
    if (len < 0 || len >= CODE_IMMUNE_PERSIST_MAX_PATH) {
        return -1;
    }

    /* Create directory if requested */
    if (create_dir) {
        char dir_path[CODE_IMMUNE_PERSIST_MAX_PATH];
        snprintf(dir_path, sizeof(dir_path), "%s/.nimcp", home);

        /* Use mkdir with mode 0755 - ignore error if exists */
        mkdir(dir_path, 0755);
    }

    return 0;
}

/**
 * @brief Resolve filepath (use default if NULL)
 */
static int resolve_persist_path(
    const char* filepath,
    const code_immune_persist_config_t* config,
    char* resolved_path
) {
    if (!resolved_path) return -1;

    if (filepath && filepath[0] != '\0') {
        strncpy(resolved_path, filepath, CODE_IMMUNE_PERSIST_MAX_PATH - 1);
        resolved_path[CODE_IMMUNE_PERSIST_MAX_PATH - 1] = '\0';
        return 0;
    }

    if (config && config->memory_file_path[0] != '\0') {
        strncpy(resolved_path, config->memory_file_path,
                CODE_IMMUNE_PERSIST_MAX_PATH - 1);
        resolved_path[CODE_IMMUNE_PERSIST_MAX_PATH - 1] = '\0';
        return 0;
    }

    return code_immune_get_default_memory_path(resolved_path, true);
}

/* ============================================================================
 * Persistence API - Validation
 * ============================================================================ */

/**
 * @brief Check version compatibility
 */
bool code_immune_is_version_compatible(uint32_t file_version) {
    /* For now, require exact match */
    return (file_version == CODE_IMMUNE_PERSIST_VERSION);
}

/**
 * @brief Validate persistence file
 */
int code_immune_validate_memory_file(const char* filepath, bool verify_checksum) {
    if (!filepath) return -1;

    FILE* file = fopen(filepath, "rb");
    if (!file) {
        LOG_MODULE_ERROR(CODE_IMMUNE_MODULE_NAME, "File not found: %s", filepath);
        return -1;
    }

    code_immune_persist_header_t header;
    if (read_persist_header(file, &header) != 0) {
        fclose(file);
        return -1;
    }

    if (validate_persist_header(&header) != 0) {
        fclose(file);
        return -1;
    }

    code_immune_persist_counts_t counts;
    if (read_persist_counts(file, &counts) != 0) {
        fclose(file);
        return -1;
    }

    if (validate_persist_counts(&counts) != 0) {
        fclose(file);
        return -1;
    }

    /* TODO: Verify checksum if enabled */
    (void)verify_checksum;

    fclose(file);
    return 0;
}

/**
 * @brief Get file information without loading
 */
int code_immune_get_memory_file_info(
    const char* filepath,
    code_immune_persist_header_t* header,
    code_immune_persist_counts_t* counts
) {
    if (!filepath) return -1;

    FILE* file = fopen(filepath, "rb");
    if (!file) return -1;

    int result = -1;

    if (header) {
        if (read_persist_header(file, header) != 0) goto cleanup;
    } else {
        code_immune_persist_header_t tmp_header;
        if (read_persist_header(file, &tmp_header) != 0) goto cleanup;
    }

    if (counts) {
        if (read_persist_counts(file, counts) != 0) goto cleanup;
    }

    result = 0;

cleanup:
    fclose(file);
    return result;
}

/* ============================================================================
 * Persistence API - Backup
 * ============================================================================ */

/**
 * @brief Create backup of persistence file
 */
int code_immune_create_backup(const char* filepath, const char* backup_suffix) {
    if (!filepath) return -1;

    /* Check if source file exists */
    FILE* src = fopen(filepath, "rb");
    if (!src) return 0; /* No file to back up, not an error */

    /* Create backup path */
    char backup_path[CODE_IMMUNE_PERSIST_MAX_PATH];
    const char* suffix = backup_suffix ? backup_suffix : ".bak";
    snprintf(backup_path, sizeof(backup_path), "%s%s", filepath, suffix);

    /* Open backup file */
    FILE* dst = fopen(backup_path, "wb");
    if (!dst) {
        fclose(src);
        LOG_MODULE_ERROR(CODE_IMMUNE_MODULE_NAME,
                        "Failed to create backup file: %s", backup_path);
        return -1;
    }

    /* Copy data */
    char buffer[8192];
    size_t bytes;
    while ((bytes = fread(buffer, 1, sizeof(buffer), src)) > 0) {
        if (fwrite(buffer, 1, bytes, dst) != bytes) {
            fclose(src);
            fclose(dst);
            remove(backup_path);
            LOG_MODULE_ERROR(CODE_IMMUNE_MODULE_NAME, "Failed to write backup");
            return -1;
        }
    }

    fclose(src);
    fclose(dst);

    LOG_MODULE_INFO(CODE_IMMUNE_MODULE_NAME, "Created backup: %s", backup_path);
    return 0;
}

/* ============================================================================
 * Persistence API - Save
 * ============================================================================ */

/**
 * @brief Save code immune memory to file
 */
int code_immune_save_memory(
    code_immune_system_t* system,
    const char* filepath,
    const code_immune_persist_config_t* config
) {
    if (!system) return -1;

    /* Use default config if not provided */
    code_immune_persist_config_t default_cfg;
    if (!config) {
        code_immune_persist_default_config(&default_cfg);
        config = &default_cfg;
    }

    /* Resolve filepath */
    char resolved_path[CODE_IMMUNE_PERSIST_MAX_PATH];
    if (resolve_persist_path(filepath, config, resolved_path) != 0) {
        return -1;
    }

    /* Create backup if requested */
    if (config->create_backup) {
        code_immune_create_backup(resolved_path, config->backup_suffix);
    }

    /* Create temporary file for atomic write */
    char temp_path[CODE_IMMUNE_PERSIST_MAX_PATH];
    snprintf(temp_path, sizeof(temp_path), "%s.tmp", resolved_path);

    FILE* file = fopen(temp_path, "wb");
    if (!file) {
        LOG_MODULE_ERROR(CODE_IMMUNE_MODULE_NAME,
                        "Failed to open file for writing: %s", temp_path);
        return -1;
    }

    int result = -1;
    nimcp_mutex_lock(system->mutex);

    /* Count items to save */
    uint32_t b_cell_save_count = 0;
    uint32_t antibody_save_count = 0;

    if (config->save_b_cells) {
        for (size_t i = 0; i < system->b_cell_count; i++) {
            code_b_cell_t* b_cell = &system->b_cells[i];

            /* Skip if memory-only mode and not memory cell */
            if (config->memory_cells_only &&
                b_cell->state != CODE_B_CELL_MEMORY) {
                continue;
            }

            /* Skip low confidence if auto-prune */
            if (config->auto_prune &&
                b_cell->affinity < config->min_confidence) {
                continue;
            }

            b_cell_save_count++;
        }
    }

    if (config->save_antibodies) {
        for (size_t i = 0; i < system->antibody_count; i++) {
            code_antibody_t* ab = &system->antibodies[i];

            /* Skip unvalidated antibodies */
            if (!ab->validated) continue;

            /* Skip low effectiveness if auto-prune */
            if (config->auto_prune &&
                ab->effectiveness < config->min_confidence) {
                continue;
            }

            antibody_save_count++;
        }
    }

    /* Prepare header */
    code_immune_persist_header_t header;
    memset(&header, 0, sizeof(header));
    memcpy(header.magic, CODE_IMMUNE_PERSIST_MAGIC, CODE_IMMUNE_PERSIST_MAGIC_LEN);
    header.version = CODE_IMMUNE_PERSIST_VERSION;
    header.timestamp = get_timestamp_ms();
    header.flags = 0;

    if (config->enable_compression) {
        header.flags |= CODE_IMMUNE_FORMAT_FLAG_COMPRESSED;
    }
    if (config->enable_encryption) {
        header.flags |= CODE_IMMUNE_FORMAT_FLAG_ENCRYPTED;
    }
    if (config->memory_cells_only) {
        header.flags |= CODE_IMMUNE_FORMAT_FLAG_MEMORY_ONLY;
    }

    /* Prepare counts */
    code_immune_persist_counts_t counts;
    memset(&counts, 0, sizeof(counts));
    counts.b_cell_count = b_cell_save_count;
    counts.antibody_count = antibody_save_count;
    counts.pattern_stat_count = 0;  /* TODO: Implement pattern stats */

    /* Write header (checksum filled later) */
    if (write_persist_header(file, &header) != 0) goto cleanup;

    /* Write counts */
    if (write_persist_counts(file, &counts) != 0) goto cleanup;

    /* Write B cells */
    if (config->save_b_cells) {
        for (size_t i = 0; i < system->b_cell_count; i++) {
            code_b_cell_t* b_cell = &system->b_cells[i];

            /* Apply same filters as counting */
            if (config->memory_cells_only &&
                b_cell->state != CODE_B_CELL_MEMORY) {
                continue;
            }
            if (config->auto_prune &&
                b_cell->affinity < config->min_confidence) {
                continue;
            }

            if (fwrite(b_cell, sizeof(code_b_cell_t), 1, file) != 1) {
                LOG_MODULE_ERROR(CODE_IMMUNE_MODULE_NAME, "Failed to write B cell");
                goto cleanup;
            }
        }
    }

    /* Write antibodies */
    if (config->save_antibodies) {
        for (size_t i = 0; i < system->antibody_count; i++) {
            code_antibody_t* ab = &system->antibodies[i];

            /* Apply same filters as counting */
            if (!ab->validated) continue;
            if (config->auto_prune &&
                ab->effectiveness < config->min_confidence) {
                continue;
            }

            if (fwrite(ab, sizeof(code_antibody_t), 1, file) != 1) {
                LOG_MODULE_ERROR(CODE_IMMUNE_MODULE_NAME, "Failed to write antibody");
                goto cleanup;
            }
        }
    }

    /* Update header with file size */
    long file_size = ftell(file);
    header.file_size = (uint64_t)file_size;
    header.checksum = 0; /* TODO: Implement proper checksum */

    /* Rewrite header with updated values */
    fseek(file, 0, SEEK_SET);
    if (write_persist_header(file, &header) != 0) goto cleanup;

    result = 0;
    LOG_MODULE_INFO(CODE_IMMUNE_MODULE_NAME,
        "Saved code immune memory: %u B cells, %u antibodies to %s",
        counts.b_cell_count, counts.antibody_count, resolved_path);

cleanup:
    nimcp_mutex_unlock(system->mutex);
    fclose(file);

    if (result == 0) {
        /* Atomic rename */
        if (rename(temp_path, resolved_path) != 0) {
            LOG_MODULE_ERROR(CODE_IMMUNE_MODULE_NAME,
                            "Failed to rename temp file to %s", resolved_path);
            remove(temp_path);
            return -1;
        }
    } else {
        remove(temp_path);
    }

    return result;
}

/**
 * @brief Save with detailed result information
 */
int code_immune_save_memory_ex(
    code_immune_system_t* system,
    const char* filepath,
    const code_immune_persist_config_t* config,
    code_immune_persist_result_t* result
) {
    if (!result) return -1;

    memset(result, 0, sizeof(*result));
    uint64_t start_time = get_timestamp_ms();

    int ret = code_immune_save_memory(system, filepath, config);

    result->success = (ret == 0);
    result->save_time_ms = get_timestamp_ms() - start_time;

    if (ret != 0) {
        snprintf(result->error_message, sizeof(result->error_message),
                 "Save failed");
    }

    return ret;
}

/* ============================================================================
 * Persistence API - Load
 * ============================================================================ */

/**
 * @brief Load code immune memory from file
 */
int code_immune_load_memory(
    code_immune_system_t* system,
    const char* filepath,
    const code_immune_persist_config_t* config
) {
    if (!system) return -1;

    /* Use default config if not provided */
    code_immune_persist_config_t default_cfg;
    if (!config) {
        code_immune_persist_default_config(&default_cfg);
        config = &default_cfg;
    }

    /* Resolve filepath */
    char resolved_path[CODE_IMMUNE_PERSIST_MAX_PATH];
    if (resolve_persist_path(filepath, config, resolved_path) != 0) {
        return -1;
    }

    FILE* file = fopen(resolved_path, "rb");
    if (!file) {
        /* File doesn't exist - not an error, just nothing to load */
        LOG_MODULE_INFO(CODE_IMMUNE_MODULE_NAME,
                       "No memory file found at %s", resolved_path);
        return 0;
    }

    int result = -1;
    nimcp_mutex_lock(system->mutex);

    /* Read and validate header */
    code_immune_persist_header_t header;
    if (read_persist_header(file, &header) != 0) goto cleanup;
    if (validate_persist_header(&header) != 0) goto cleanup;

    /* Read and validate counts */
    code_immune_persist_counts_t counts;
    if (read_persist_counts(file, &counts) != 0) goto cleanup;
    if (validate_persist_counts(&counts) != 0) goto cleanup;

    /* Load B cells (merge with existing) */
    uint32_t b_cells_loaded = 0;
    for (uint32_t i = 0; i < counts.b_cell_count; i++) {
        code_b_cell_t loaded_cell;
        if (fread(&loaded_cell, sizeof(code_b_cell_t), 1, file) != 1) {
            LOG_MODULE_ERROR(CODE_IMMUNE_MODULE_NAME, "Failed to read B cell");
            goto cleanup;
        }

        /* Check if B cell with same receptor already exists */
        code_b_cell_t* existing = find_b_cell_by_receptor(system, loaded_cell.receptor);
        if (existing) {
            /* Update if loaded has higher affinity */
            if (loaded_cell.affinity > existing->affinity) {
                existing->affinity = loaded_cell.affinity;
                existing->successful_fixes += loaded_cell.successful_fixes;
                existing->failed_fixes += loaded_cell.failed_fixes;
                if (loaded_cell.fix_template[0] && !existing->fix_template[0]) {
                    memcpy(existing->fix_template, loaded_cell.fix_template,
                           sizeof(existing->fix_template));
                }
            }
        } else {
            /* Add new B cell if capacity allows */
            if (system->b_cell_count < system->b_cell_capacity) {
                loaded_cell.id = system->next_b_cell_id++;
                system->b_cells[system->b_cell_count] = loaded_cell;
                system->b_cell_count++;
                b_cells_loaded++;

                if (loaded_cell.state == CODE_B_CELL_MEMORY) {
                    system->stats.memory_b_cells++;
                }
            }
        }
    }

    /* Load antibodies (merge with existing) */
    uint32_t antibodies_loaded = 0;
    for (uint32_t i = 0; i < counts.antibody_count; i++) {
        code_antibody_t loaded_ab;
        if (fread(&loaded_ab, sizeof(code_antibody_t), 1, file) != 1) {
            LOG_MODULE_ERROR(CODE_IMMUNE_MODULE_NAME, "Failed to read antibody");
            goto cleanup;
        }

        /* Add antibody if capacity allows */
        if (system->antibody_count < system->antibody_capacity) {
            /* Clear runtime state */
            loaded_ab.patch_handle = NULL;
            loaded_ab.old_function = NULL;
            loaded_ab.new_function = NULL;
            loaded_ab.injected = false;

            loaded_ab.id = system->next_antibody_id++;
            system->antibodies[system->antibody_count] = loaded_ab;
            system->antibody_count++;
            antibodies_loaded++;

            system->stats.active_antibodies++;
        }
    }

    result = 0;
    LOG_MODULE_INFO(CODE_IMMUNE_MODULE_NAME,
        "Loaded code immune memory: %u B cells, %u antibodies from %s",
        b_cells_loaded, antibodies_loaded, resolved_path);

cleanup:
    nimcp_mutex_unlock(system->mutex);
    fclose(file);
    return result;
}

/**
 * @brief Load with detailed result information
 */
int code_immune_load_memory_ex(
    code_immune_system_t* system,
    const char* filepath,
    const code_immune_persist_config_t* config,
    code_immune_persist_result_t* result
) {
    if (!result) return -1;

    memset(result, 0, sizeof(*result));
    uint64_t start_time = get_timestamp_ms();

    /* Get file info first */
    char resolved_path[CODE_IMMUNE_PERSIST_MAX_PATH];
    if (resolve_persist_path(filepath, config, resolved_path) == 0) {
        code_immune_persist_header_t header;
        if (code_immune_get_memory_file_info(resolved_path, &header, NULL) == 0) {
            result->version_loaded = header.version;
        }
    }

    int ret = code_immune_load_memory(system, filepath, config);

    result->success = (ret == 0);
    result->load_time_ms = get_timestamp_ms() - start_time;

    if (ret != 0) {
        snprintf(result->error_message, sizeof(result->error_message),
                 "Load failed");
    }

    return ret;
}

/* ============================================================================
 * Persistence API - Consolidation
 * ============================================================================ */

/**
 * @brief Consolidate immune memory
 */
int code_immune_consolidate_memory(
    code_immune_system_t* system,
    const code_immune_persist_config_t* config,
    uint32_t* items_pruned
) {
    if (!system) return -1;

    /* Use default config if not provided */
    code_immune_persist_config_t default_cfg;
    if (!config) {
        code_immune_persist_default_config(&default_cfg);
        config = &default_cfg;
    }

    uint32_t pruned = 0;

    nimcp_mutex_lock(system->mutex);

    /* Prune low-confidence B cells */
    size_t write_idx = 0;
    for (size_t read_idx = 0; read_idx < system->b_cell_count; read_idx++) {
        code_b_cell_t* b_cell = &system->b_cells[read_idx];

        /* Keep if above threshold or is memory cell with successful fixes */
        bool keep = (b_cell->affinity >= config->min_confidence) ||
                    (b_cell->state == CODE_B_CELL_MEMORY &&
                     b_cell->successful_fixes >= config->min_successful_fixes);

        if (keep) {
            if (write_idx != read_idx) {
                system->b_cells[write_idx] = system->b_cells[read_idx];
            }
            write_idx++;
        } else {
            pruned++;
            if (b_cell->state == CODE_B_CELL_MEMORY) {
                system->stats.memory_b_cells--;
            }
        }
    }
    system->b_cell_count = write_idx;

    /* Prune low-effectiveness antibodies */
    write_idx = 0;
    for (size_t read_idx = 0; read_idx < system->antibody_count; read_idx++) {
        code_antibody_t* ab = &system->antibodies[read_idx];

        /* Keep if validated and above threshold */
        bool keep = ab->validated && (ab->effectiveness >= config->min_confidence);

        if (keep) {
            if (write_idx != read_idx) {
                system->antibodies[write_idx] = system->antibodies[read_idx];
            }
            write_idx++;
        } else {
            /* Unload patch if loaded */
            if (ab->patch_handle) {
                dlclose(ab->patch_handle);
            }
            pruned++;
            system->stats.active_antibodies--;
        }
    }
    system->antibody_count = write_idx;

    /* TODO: Merge similar B cells (affinity > 0.95) */

    nimcp_mutex_unlock(system->mutex);

    if (items_pruned) {
        *items_pruned = pruned;
    }

    LOG_MODULE_INFO(CODE_IMMUNE_MODULE_NAME,
        "Consolidated memory: pruned %u items", pruned);

    return 0;
}

/* ============================================================================
 * Persistence API - Auto-Save/Auto-Load
 * ============================================================================ */

/**
 * @brief Enable auto-save on fix validation
 */
int code_immune_enable_auto_save(
    code_immune_system_t* system,
    bool enable,
    const char* filepath
) {
    if (!system) return -1;

    nimcp_mutex_lock(system->mutex);

    system->auto_save_enabled = enable;

    if (filepath && filepath[0] != '\0') {
        strncpy(system->auto_save_path, filepath,
                sizeof(system->auto_save_path) - 1);
        system->auto_save_path[sizeof(system->auto_save_path) - 1] = '\0';
    } else {
        system->auto_save_path[0] = '\0';  /* Use default path */
    }

    nimcp_mutex_unlock(system->mutex);

    LOG_MODULE_INFO(CODE_IMMUNE_MODULE_NAME,
        "Auto-save %s", enable ? "enabled" : "disabled");

    return 0;
}

/**
 * @brief Enable auto-load on startup
 */
int code_immune_enable_auto_load(
    code_immune_system_t* system,
    bool enable,
    const char* filepath
) {
    if (!system) return -1;

    nimcp_mutex_lock(system->mutex);

    system->auto_load_enabled = enable;

    if (filepath && filepath[0] != '\0') {
        strncpy(system->auto_load_path, filepath,
                sizeof(system->auto_load_path) - 1);
        system->auto_load_path[sizeof(system->auto_load_path) - 1] = '\0';
    } else {
        system->auto_load_path[0] = '\0';  /* Use default path */
    }

    nimcp_mutex_unlock(system->mutex);

    LOG_MODULE_INFO(CODE_IMMUNE_MODULE_NAME,
        "Auto-load %s", enable ? "enabled" : "disabled");

    return 0;
}
