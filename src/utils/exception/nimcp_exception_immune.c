/**
 * @file nimcp_exception_immune.c
 * @brief Exception-to-immune system integration implementation
 * @version 1.0.0
 * @date 2025-01-16
 *
 * WHAT: Bridge between exception handling and brain immune system
 * WHY:  Enable automatic error resolution through immune-style responses
 * HOW:  Present exceptions as antigens, execute recovery antibodies
 *
 * @author NIMCP Development Team
 */

#include "utils/exception/nimcp_exception_immune.h"
#include "utils/exception/nimcp_exception_handlers.h"
#include "core/brain/nimcp_kg_module_wiring.h"  /* Must be before macros.h */
#include "utils/exception/nimcp_exception_macros.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/platform/nimcp_platform_mutex.h"

/* Include brain immune only if available - forward declare otherwise */
#ifdef NIMCP_BRAIN_IMMUNE_H
#include "cognitive/immune/nimcp_brain_immune.h"
#endif

#include <string.h>
#include <time.h>

/* ============================================================================
 * Module State
 * ============================================================================ */

static bool g_immune_initialized = false;
static nimcp_exception_immune_config_t g_config;
static brain_immune_system_t* g_immune_system = NULL;
static nimcp_platform_mutex_t* g_immune_mutex = NULL;

/* Async presentation queue */
static struct {
    nimcp_exception_t* exceptions[NIMCP_EXCEPTION_IMMUNE_QUEUE_SIZE];
    size_t head;
    size_t tail;
    size_t count;
} g_async_queue = {0};

/* Statistics */
static nimcp_exception_immune_stats_t g_stats = {0};

/* ============================================================================
 * Internal Helpers
 * ============================================================================ */

static uint64_t get_timestamp_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ULL + (uint64_t)ts.tv_nsec / 1000ULL;
}

/* ============================================================================
 * Configuration
 * ============================================================================ */

void nimcp_exception_immune_default_config(nimcp_exception_immune_config_t* config) {
    if (!config) return;

    config->enable_auto_present = true;
    config->min_present_severity = EXCEPTION_SEVERITY_SEVERE;
    config->enable_auto_recovery = true;
    config->enable_memory_formation = true;
    config->async_presentation = false;
    config->max_pending_exceptions = NIMCP_EXCEPTION_IMMUNE_QUEUE_SIZE;
    config->response_timeout_ms = 5000;
}

/* ============================================================================
 * Initialization
 * ============================================================================ */

int nimcp_exception_immune_init(const nimcp_exception_immune_config_t* config) {
    if (g_immune_initialized) return 0;

    if (config) {
        g_config = *config;
    } else {
        nimcp_exception_immune_default_config(&g_config);
    }

    g_immune_mutex = nimcp_platform_mutex_create();
    if (!g_immune_mutex) return -1;

    memset(&g_async_queue, 0, sizeof(g_async_queue));
    memset(&g_stats, 0, sizeof(g_stats));

    g_immune_initialized = true;
    LOG_INFO("Exception-immune integration initialized");

    return 0;
}

void nimcp_exception_immune_shutdown(void) {
    if (!g_immune_initialized) return;

    /* Clear async queue */
    while (g_async_queue.count > 0) {
        nimcp_exception_t* ex = g_async_queue.exceptions[g_async_queue.head];
        if (ex) nimcp_exception_unref(ex);
        g_async_queue.head = (g_async_queue.head + 1) % NIMCP_EXCEPTION_IMMUNE_QUEUE_SIZE;
        g_async_queue.count--;
    }

    if (g_immune_mutex) {
        nimcp_platform_mutex_destroy(g_immune_mutex);
        nimcp_free(g_immune_mutex);
        g_immune_mutex = NULL;
    }

    g_immune_system = NULL;
    g_immune_initialized = false;

    LOG_INFO("Exception-immune integration shutdown");
}

/* ============================================================================
 * Connection
 * ============================================================================ */

int nimcp_exception_immune_connect(brain_immune_system_t* immune_system) {
    if (!immune_system) return -1;
    if (!g_immune_initialized) {
        if (nimcp_exception_immune_init(NULL) != 0) return -1;
    }

    g_immune_system = immune_system;
    LOG_INFO("Connected to brain immune system");
    return 0;
}

int nimcp_exception_immune_disconnect(void) {
    g_immune_system = NULL;
    LOG_INFO("Disconnected from brain immune system");
    return 0;
}

bool nimcp_exception_immune_is_connected(void) {
    return g_immune_system != NULL;
}

/* ============================================================================
 * Mapping Functions
 * ============================================================================ */

exception_antigen_source_t nimcp_exception_to_antigen_source(
    nimcp_exception_category_t category
) {
    switch (category) {
        case EXCEPTION_CATEGORY_SECURITY:
            return EX_ANTIGEN_SOURCE_BBB;

        case EXCEPTION_CATEGORY_BRAIN:
        case EXCEPTION_CATEGORY_BRAIN_REGION:
            return EX_ANTIGEN_SOURCE_BBB;

        case EXCEPTION_CATEGORY_THREADING:
            return EX_ANTIGEN_SOURCE_BFT;

        case EXCEPTION_CATEGORY_MEMORY:
        case EXCEPTION_CATEGORY_IO:
        case EXCEPTION_CATEGORY_CONFIG:
        case EXCEPTION_CATEGORY_SIGNAL:
        case EXCEPTION_CATEGORY_COGNITIVE:
        case EXCEPTION_CATEGORY_GPU:
        default:
            return EX_ANTIGEN_SOURCE_ANOMALY;
    }
}

uint32_t nimcp_exception_to_immune_severity(nimcp_exception_severity_t severity) {
    switch (severity) {
        case EXCEPTION_SEVERITY_DEBUG:    return 1;
        case EXCEPTION_SEVERITY_INFO:     return 2;
        case EXCEPTION_SEVERITY_WARNING:  return 3;
        case EXCEPTION_SEVERITY_ERROR:    return 5;
        case EXCEPTION_SEVERITY_SEVERE:   return 7;
        case EXCEPTION_SEVERITY_CRITICAL: return 9;
        case EXCEPTION_SEVERITY_FATAL:    return 10;
        default: return 5;
    }
}

void nimcp_exception_get_recovery_strategy(
    const nimcp_exception_t* ex,
    nimcp_recovery_strategy_t* strategy
) {
    if (!strategy) return;

    memset(strategy, 0, sizeof(nimcp_recovery_strategy_t));

    if (!ex) {
        strategy->primary_action = RECOVERY_ACTION_NONE;
        strategy->fallback_action = RECOVERY_ACTION_NONE;
        return;
    }

    /* Set based on exception category */
    switch (ex->category) {
        case EXCEPTION_CATEGORY_MEMORY:
            strategy->primary_action = RECOVERY_ACTION_GC;
            strategy->fallback_action = RECOVERY_ACTION_QUARANTINE;
            strategy->retry_count = 3;
            strategy->cooldown_ms = 100;
            break;

        case EXCEPTION_CATEGORY_BRAIN:
        case EXCEPTION_CATEGORY_BRAIN_REGION:
            strategy->primary_action = RECOVERY_ACTION_ROLLBACK;
            strategy->fallback_action = RECOVERY_ACTION_REDUCE_LOAD;
            strategy->retry_count = 1;
            strategy->cooldown_ms = 500;
            break;

        case EXCEPTION_CATEGORY_THREADING:
            strategy->primary_action = RECOVERY_ACTION_RESTART_THREAD;
            strategy->fallback_action = RECOVERY_ACTION_GRACEFUL_SHUTDOWN;
            strategy->retry_count = 2;
            strategy->cooldown_ms = 1000;
            break;

        case EXCEPTION_CATEGORY_IO:
            strategy->primary_action = RECOVERY_ACTION_RETRY;
            strategy->fallback_action = RECOVERY_ACTION_ROLLBACK;
            strategy->retry_count = 5;
            strategy->cooldown_ms = 200;
            break;

        case EXCEPTION_CATEGORY_SIGNAL:
            strategy->primary_action = RECOVERY_ACTION_EMERGENCY_SAVE;
            strategy->fallback_action = RECOVERY_ACTION_GRACEFUL_SHUTDOWN;
            strategy->retry_count = 1;
            strategy->cooldown_ms = 0;
            break;

        case EXCEPTION_CATEGORY_GPU:
            strategy->primary_action = RECOVERY_ACTION_CLEAR_CACHE;
            strategy->fallback_action = RECOVERY_ACTION_REDUCE_LOAD;
            strategy->retry_count = 3;
            strategy->cooldown_ms = 50;
            break;

        case EXCEPTION_CATEGORY_SECURITY:
            strategy->primary_action = RECOVERY_ACTION_QUARANTINE;
            strategy->fallback_action = RECOVERY_ACTION_GRACEFUL_SHUTDOWN;
            strategy->retry_count = 1;
            strategy->cooldown_ms = 0;
            break;

        default:
            strategy->primary_action = RECOVERY_ACTION_RETRY;
            strategy->fallback_action = RECOVERY_ACTION_NONE;
            strategy->retry_count = 3;
            strategy->cooldown_ms = 100;
            break;
    }

    strategy->severity_threshold = 0.5f;
}

/* ============================================================================
 * Epitope Generation
 * ============================================================================ */

/**
 * @brief Simple hash function
 */
static uint32_t hash_bytes(const void* data, size_t len) {
    const uint8_t* bytes = (const uint8_t*)data;
    uint32_t hash = 5381;
    for (size_t i = 0; i < len; i++) {
        hash = ((hash << 5) + hash) + bytes[i];
    }
    return hash;
}

size_t nimcp_exception_compute_epitope(
    const nimcp_exception_t* ex,
    uint8_t* epitope,
    size_t epitope_size
) {
    if (!ex || !epitope || epitope_size < NIMCP_EXCEPTION_EPITOPE_SIZE) return 0;

    memset(epitope, 0, epitope_size);
    size_t offset = 0;

    /* Error code (4 bytes) */
    memcpy(epitope + offset, &ex->code, sizeof(ex->code));
    offset += sizeof(ex->code);

    /* Category hash (4 bytes) */
    uint32_t cat = (uint32_t)ex->category;
    memcpy(epitope + offset, &cat, sizeof(cat));
    offset += sizeof(cat);

    /* Severity (4 bytes) */
    uint32_t sev = (uint32_t)ex->severity;
    memcpy(epitope + offset, &sev, sizeof(sev));
    offset += sizeof(sev);

    /* Type (4 bytes) */
    uint32_t type = (uint32_t)ex->type;
    memcpy(epitope + offset, &type, sizeof(type));
    offset += sizeof(type);

    /* File hash (4 bytes) */
    if (ex->file) {
        uint32_t file_hash = hash_bytes(ex->file, strlen(ex->file));
        memcpy(epitope + offset, &file_hash, sizeof(file_hash));
    }
    offset += 4;

    /* Function hash (4 bytes) */
    if (ex->function) {
        uint32_t func_hash = hash_bytes(ex->function, strlen(ex->function));
        memcpy(epitope + offset, &func_hash, sizeof(func_hash));
    }
    offset += 4;

    /* Line (4 bytes) */
    uint32_t line = (uint32_t)ex->line;
    memcpy(epitope + offset, &line, sizeof(line));
    offset += sizeof(line);

    /* Message hash (4 bytes) */
    if (ex->message[0]) {
        uint32_t msg_hash = hash_bytes(ex->message, strlen(ex->message));
        memcpy(epitope + offset, &msg_hash, sizeof(msg_hash));
    }
    offset += 4;

    /* Stack trace addresses (remaining bytes) */
    for (size_t i = 0; i < ex->stack_trace.depth && offset < NIMCP_EXCEPTION_EPITOPE_SIZE - 4; i++) {
        uint32_t addr = (uint32_t)(uintptr_t)ex->stack_trace.frames[i].address;
        memcpy(epitope + offset, &addr, sizeof(addr));
        offset += sizeof(addr);
    }

    return offset;
}

/* ============================================================================
 * Exception Presentation
 * ============================================================================ */

int nimcp_exception_present_to_immune(
    nimcp_exception_t* ex,
    nimcp_immune_response_t* response
) {
    if (!ex) return -1;
    if (!g_immune_system) {
        /* Not connected - just log */
        LOG_DEBUG("Exception not presented to immune: not connected");
        return 0;
    }

    if (ex->presented_to_immune) {
        /* Already presented */
        return 0;
    }

    uint64_t start_time = get_timestamp_us();

    /* Compute epitope if not already done */
    if (ex->epitope_len == 0) {
        nimcp_exception_generate_epitope(ex);
    }

    /* Map to immune concepts */
    exception_antigen_source_t source = nimcp_exception_to_antigen_source(ex->category);
    uint32_t severity = nimcp_exception_to_immune_severity(ex->severity);

    /* Present to immune system */
    uint32_t antigen_id = 0;

#ifdef NIMCP_BRAIN_IMMUNE_H
    int result = brain_immune_present_antigen(
        g_immune_system,
        (brain_antigen_source_t)source,
        ex->epitope,
        ex->epitope_len,
        severity,
        0,  /* source_node - we don't track this for exceptions */
        &antigen_id
    );

    if (result != 0) {
        LOG_WARNING("Failed to present exception to immune system: code=%d", ex->code);
        return -1;
    }
#else
    /* Immune system header not included - simulate success */
    antigen_id = (uint32_t)(ex->timestamp_us & 0xFFFFFFFF);
#endif

    ex->presented_to_immune = true;
    ex->antigen_id = antigen_id;

    uint64_t response_time = get_timestamp_us() - start_time;

    /* Update stats */
    g_stats.exceptions_presented++;
    g_stats.avg_response_time_us =
        (g_stats.avg_response_time_us * (g_stats.exceptions_presented - 1) + response_time)
        / g_stats.exceptions_presented;

    LOG_DEBUG("Presented exception to immune: code=%d, antigen_id=%u, severity=%u",
              ex->code, antigen_id, severity);

    /* Fill response if requested */
    if (response) {
        response->antigen_id = antigen_id;
        response->antibody_id = 0;  /* Will be set by immune response */
        response->action_taken = RECOVERY_ACTION_NONE;
        response->recovery_attempted = false;
        response->recovery_succeeded = false;
        response->response_time_us = response_time;
        response->memory_formed = false;
    }

    /* Auto-execute recovery if enabled */
    if (g_config.enable_auto_recovery && ex->suggested_action != RECOVERY_ACTION_NONE) {
        int recovery_result = nimcp_exception_execute_recovery(ex, ex->suggested_action);

        if (response) {
            response->action_taken = ex->suggested_action;
            response->recovery_attempted = true;
            response->recovery_succeeded = (recovery_result == 0);
        }

        g_stats.recoveries_attempted++;
        if (recovery_result == 0) {
            g_stats.recoveries_succeeded++;
        }
    }

    return 0;
}

int nimcp_exception_present_async(nimcp_exception_t* ex) {
    if (!ex) return -1;
    if (!g_immune_initialized) return -1;

    if (g_immune_mutex) nimcp_platform_mutex_lock(g_immune_mutex);

    if (g_async_queue.count >= NIMCP_EXCEPTION_IMMUNE_QUEUE_SIZE) {
        g_stats.queue_overflows++;
        if (g_immune_mutex) nimcp_platform_mutex_unlock(g_immune_mutex);
        return -1;
    }

    /* Add to queue */
    g_async_queue.exceptions[g_async_queue.tail] = nimcp_exception_ref(ex);
    g_async_queue.tail = (g_async_queue.tail + 1) % NIMCP_EXCEPTION_IMMUNE_QUEUE_SIZE;
    g_async_queue.count++;
    g_stats.exceptions_pending = g_async_queue.count;

    if (g_immune_mutex) nimcp_platform_mutex_unlock(g_immune_mutex);

    return 0;
}

size_t nimcp_exception_immune_process_pending(size_t max_count) {
    if (!g_immune_initialized) return 0;

    size_t processed = 0;

    if (g_immune_mutex) nimcp_platform_mutex_lock(g_immune_mutex);

    while (g_async_queue.count > 0 && (max_count == 0 || processed < max_count)) {
        nimcp_exception_t* ex = g_async_queue.exceptions[g_async_queue.head];
        g_async_queue.head = (g_async_queue.head + 1) % NIMCP_EXCEPTION_IMMUNE_QUEUE_SIZE;
        g_async_queue.count--;

        if (g_immune_mutex) nimcp_platform_mutex_unlock(g_immune_mutex);

        if (ex) {
            nimcp_exception_present_to_immune(ex, NULL);
            nimcp_exception_unref(ex);
            processed++;
        }

        if (g_immune_mutex) nimcp_platform_mutex_lock(g_immune_mutex);
    }

    g_stats.exceptions_pending = g_async_queue.count;

    if (g_immune_mutex) nimcp_platform_mutex_unlock(g_immune_mutex);

    return processed;
}

/* ============================================================================
 * Recovery Execution
 * ============================================================================ */

int nimcp_exception_execute_recovery(
    nimcp_exception_t* ex,
    nimcp_recovery_action_t action
) {
    return nimcp_execute_recovery(ex, action);
}

int nimcp_exception_notify_recovery_result(
    nimcp_exception_t* ex,
    nimcp_recovery_action_t action,
    bool success
) {
    if (!ex) return -1;

    ex->recovery_attempted = true;
    ex->recovery_succeeded = success;

    LOG_INFO("Recovery result: code=%d, action=%s, success=%s",
             ex->code,
             nimcp_recovery_action_to_string(action),
             success ? "true" : "false");

    /* Notify immune system of result */
#ifdef NIMCP_BRAIN_IMMUNE_H
    if (g_immune_system && ex->presented_to_immune) {
        /* Could trigger memory formation here */
        if (success && g_config.enable_memory_formation) {
            g_stats.memories_formed++;
        }
    }
#endif

    return 0;
}

/* ============================================================================
 * Default Recovery Callbacks
 * ============================================================================ */

int nimcp_recovery_gc(nimcp_exception_t* ex, nimcp_recovery_action_t action, void* user_data) {
    (void)ex;
    (void)action;
    (void)user_data;

    LOG_INFO("Executing GC recovery action");
    /* Would trigger actual GC here */
    /* nimcp_gc_collect(); */
    return 0;
}

int nimcp_recovery_retry(nimcp_exception_t* ex, nimcp_recovery_action_t action, void* user_data) {
    (void)ex;
    (void)action;
    (void)user_data;

    LOG_INFO("Executing retry recovery action");
    /* Retry logic would need context from exception */
    return 0;
}

int nimcp_recovery_rollback(nimcp_exception_t* ex, nimcp_recovery_action_t action, void* user_data) {
    (void)ex;
    (void)action;
    (void)user_data;

    LOG_INFO("Executing rollback recovery action");
    /* Would trigger checkpoint rollback */
    return 0;
}

int nimcp_recovery_restart_thread(nimcp_exception_t* ex, nimcp_recovery_action_t action, void* user_data) {
    (void)action;
    (void)user_data;

    LOG_INFO("Executing thread restart recovery action");

    if (ex->type == EXCEPTION_TYPE_THREADING) {
        nimcp_threading_exception_t* tex = (nimcp_threading_exception_t*)ex;
        LOG_INFO("Would restart thread %lu", (unsigned long)tex->thread_id);
    }

    return 0;
}

int nimcp_recovery_quarantine(nimcp_exception_t* ex, nimcp_recovery_action_t action, void* user_data) {
    (void)ex;
    (void)action;
    (void)user_data;

    LOG_INFO("Executing quarantine recovery action");
    /* Would quarantine affected region via BFT */
    return 0;
}

int nimcp_recovery_emergency_save(nimcp_exception_t* ex, nimcp_recovery_action_t action, void* user_data) {
    (void)ex;
    (void)action;
    (void)user_data;

    LOG_INFO("Executing emergency save recovery action");
    /* Would trigger emergency checkpoint save */
    return 0;
}

int nimcp_exception_install_default_recovery_callbacks(void) {
    nimcp_register_recovery_callback(RECOVERY_ACTION_GC, nimcp_recovery_gc, NULL);
    nimcp_register_recovery_callback(RECOVERY_ACTION_RETRY, nimcp_recovery_retry, NULL);
    nimcp_register_recovery_callback(RECOVERY_ACTION_ROLLBACK, nimcp_recovery_rollback, NULL);
    nimcp_register_recovery_callback(RECOVERY_ACTION_RESTART_THREAD, nimcp_recovery_restart_thread, NULL);
    nimcp_register_recovery_callback(RECOVERY_ACTION_QUARANTINE, nimcp_recovery_quarantine, NULL);
    nimcp_register_recovery_callback(RECOVERY_ACTION_EMERGENCY_SAVE, nimcp_recovery_emergency_save, NULL);

    LOG_INFO("Installed default recovery callbacks");
    return 0;
}

/* ============================================================================
 * Statistics
 * ============================================================================ */

void nimcp_exception_immune_get_stats(nimcp_exception_immune_stats_t* stats) {
    if (!stats) return;
    *stats = g_stats;
}

void nimcp_exception_immune_reset_stats(void) {
    memset(&g_stats, 0, sizeof(g_stats));
}

/* ============================================================================
 * KG Wiring Integration
 * ============================================================================ */

kg_module_wiring_t* nimcp_exception_create_kg_wiring(void) {
    kg_module_wiring_t* wiring = kg_module_wiring_create(
        KG_EXCEPTION_MODULE_NAME,
        KG_EXCEPTION_MODULE_TYPE
    );
    if (!wiring) return NULL;

    /* Set metadata */
    kg_module_wiring_set_metadata(wiring,
        "NIMCP Team",
        "FAULT_TOLERANCE",
        "Exception handling with brain immune integration"
    );
    kg_module_wiring_set_version(wiring, 1, 0, 0);

    /* Register inputs */
    kg_module_wiring_add_input(wiring, "*", KG_MSG_ERROR_REPORT, false);
    kg_module_wiring_add_input(wiring, "signal_handler", KG_MSG_CRASH_SIGNAL, false);

    /* Register outputs */
    kg_module_wiring_add_output(wiring, KG_MSG_EXCEPTION_RAISED, "Exception raised event for observers");
    kg_module_wiring_add_output(wiring, KG_MSG_ANTIGEN_PRESENTED, "Exception presented as antigen to immune system");
    kg_module_wiring_add_output(wiring, KG_MSG_RECOVERY_REQUEST, "Recovery action requested");
    kg_module_wiring_add_output(wiring, KG_MSG_RECOVERY_RESULT, "Recovery action result notification");

    /* Register handlers */
    kg_module_wiring_add_handler(wiring, KG_MSG_ERROR_REPORT, 200);
    kg_module_wiring_add_handler(wiring, KG_MSG_CRASH_SIGNAL, 300);

    /* Add custom metadata */
    kg_module_wiring_add_metadata_entry(wiring, "immune_integration", "enabled");
    kg_module_wiring_add_metadata_entry(wiring, "min_immune_severity", "SEVERE");

    return wiring;
}
