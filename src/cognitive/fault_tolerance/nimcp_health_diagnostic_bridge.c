/**
 * @file nimcp_health_diagnostic_bridge.c
 * @brief Implementation of Health Monitoring to Diagnostics Format Bridge
 * @version 1.0.0
 * @date 2025-01-20
 */

#include "cognitive/fault_tolerance/nimcp_health_diagnostic_bridge.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/time/nimcp_time.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <execinfo.h>

//=============================================================================
#include <stddef.h>  /* for NULL */
// Health Agent Integration (Phase 8: System-Wide Health Integration)
//=============================================================================
struct nimcp_health_agent;
typedef struct nimcp_health_agent nimcp_health_agent_t;
extern void nimcp_health_agent_heartbeat_ex(nimcp_health_agent_t* agent,
                                             const char* operation,
                                             float progress);

/** Global health agent for health_diagnostic_bridge module */
static nimcp_health_agent_t* g_health_diagnostic_bridge_health_agent = NULL;

/**
 * @brief Set health agent for health_diagnostic_bridge heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
void health_diagnostic_bridge_set_health_agent(nimcp_health_agent_t* agent) {
    g_health_diagnostic_bridge_health_agent = agent;
}

/** @brief Send heartbeat from health_diagnostic_bridge module */
static inline void health_diagnostic_bridge_heartbeat(const char* operation, float progress) {
    if (g_health_diagnostic_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_health_diagnostic_bridge_health_agent, operation, progress);
    }
}


/* ============================================================================
 * Internal Structures
 * ============================================================================ */

/**
 * @brief Health diagnostic bridge internal state
 */
struct health_diag_bridge {
    bridge_base_t base;                     /**< MUST be first: base bridge infrastructure */
    uint32_t magic;                         /**< Magic number for validation */
    health_diag_bridge_config_t config;     /**< Configuration */

    /* Custom mappings */
    anomaly_error_mapping_t* custom_anomaly_mappings;
    uint32_t custom_anomaly_mapping_count;
    agent_error_mapping_t* custom_agent_mappings;
    uint32_t custom_agent_mapping_count;

    /* Statistics */
    health_diag_bridge_stats_t stats;

    /* Timing for stats */
    uint64_t total_conversion_time_us;
    uint64_t conversion_count;

    /* State */
    bool initialized;
};

/* ============================================================================
 * Default Mapping Tables
 * ============================================================================ */

/**
 * @brief Default anomaly to error type mappings
 */
static const anomaly_error_mapping_t default_anomaly_mappings[] = {
    {ANOMALY_MEMORY_LEAK, ERROR_TYPE_MEMORY_LEAK, DIAG_SEVERITY_WARNING,
     0.8f, "Memory leak detected: %s"},
    {ANOMALY_PERFORMANCE_DEGRADATION, ERROR_TYPE_INVALID_STATE, DIAG_SEVERITY_WARNING,
     0.7f, "Performance degradation: %s"},
    {ANOMALY_ERROR_SPIKE, ERROR_TYPE_UNKNOWN, DIAG_SEVERITY_ERROR,
     0.8f, "Error spike detected: %s"},
    {ANOMALY_THROUGHPUT_DROP, ERROR_TYPE_INVALID_STATE, DIAG_SEVERITY_WARNING,
     0.6f, "Throughput drop: %s"},
    {ANOMALY_CACHE_THRASHING, ERROR_TYPE_INVALID_STATE, DIAG_SEVERITY_WARNING,
     0.7f, "Cache thrashing detected: %s"},
    {ANOMALY_RESOURCE_EXHAUSTION, ERROR_TYPE_OUT_OF_MEMORY, DIAG_SEVERITY_CRITICAL,
     0.9f, "Resource exhaustion: %s"},
    {ANOMALY_NUMERICAL_INSTABILITY, ERROR_TYPE_NAN_DETECTED, DIAG_SEVERITY_ERROR,
     0.85f, "Numerical instability: %s"},
    {ANOMALY_THREAD_CONTENTION, ERROR_TYPE_DEADLOCK, DIAG_SEVERITY_WARNING,
     0.7f, "Thread contention: %s"},
    {ANOMALY_UNKNOWN, ERROR_TYPE_UNKNOWN, DIAG_SEVERITY_INFO,
     0.5f, "Unknown anomaly: %s"},
    {ANOMALY_NONE, ERROR_TYPE_NONE, DIAG_SEVERITY_INFO,
     1.0f, "No anomaly"}
};

static const size_t default_anomaly_mapping_count =
    sizeof(default_anomaly_mappings) / sizeof(default_anomaly_mappings[0]);

/**
 * @brief Default agent message to error type mappings
 */
static const agent_error_mapping_t default_agent_mappings[] = {
    {HEALTH_MSG_ANOMALY_DETECTED, ERROR_TYPE_UNKNOWN, DIAG_SEVERITY_WARNING,
     0.7f, "Health anomaly detected: %s"},
    {HEALTH_MSG_CYTOKINE_SIGNAL, ERROR_TYPE_UNKNOWN, DIAG_SEVERITY_INFO,
     0.6f, "Cytokine signal: %s"},
    {HEALTH_MSG_EMERGENCY, ERROR_TYPE_INVALID_STATE, DIAG_SEVERITY_CRITICAL,
     0.9f, "Emergency condition: %s"},
    {HEALTH_MSG_RECOVERY_REQUEST, ERROR_TYPE_UNKNOWN, DIAG_SEVERITY_WARNING,
     0.7f, "Recovery requested: %s"},
    {HEALTH_MSG_STATE_CORRUPTION, ERROR_TYPE_HEAP_CORRUPTION, DIAG_SEVERITY_CRITICAL,
     0.9f, "State corruption detected: %s"},
    {HEALTH_MSG_HEARTBEAT_TIMEOUT, ERROR_TYPE_DEADLOCK, DIAG_SEVERITY_CRITICAL,
     0.85f, "Heartbeat timeout: %s"},
    {HEALTH_MSG_DEADLOCK_DETECTED, ERROR_TYPE_DEADLOCK, DIAG_SEVERITY_CRITICAL,
     0.95f, "Deadlock detected: %s"},
    {HEALTH_MSG_NAN_DETECTED, ERROR_TYPE_NAN_DETECTED, DIAG_SEVERITY_ERROR,
     0.9f, "NaN detected: %s"},
    {HEALTH_MSG_MEMORY_CORRUPTION, ERROR_TYPE_HEAP_CORRUPTION, DIAG_SEVERITY_CRITICAL,
     0.95f, "Memory corruption: %s"},
    {HEALTH_MSG_RESOURCE_EXHAUSTION, ERROR_TYPE_OUT_OF_MEMORY, DIAG_SEVERITY_CRITICAL,
     0.9f, "Resource exhaustion: %s"},
    {HEALTH_MSG_STATUS_UPDATE, ERROR_TYPE_NONE, DIAG_SEVERITY_INFO,
     1.0f, "Status update: %s"}
};

static const size_t default_agent_mapping_count =
    sizeof(default_agent_mappings) / sizeof(default_agent_mappings[0]);

/**
 * @brief Anomaly severity to diagnostic severity mapping
 */
static const diag_severity_t anomaly_severity_map[] = {
    [ANOMALY_SEVERITY_INFO] = DIAG_SEVERITY_INFO,
    [ANOMALY_SEVERITY_WARNING] = DIAG_SEVERITY_WARNING,
    [ANOMALY_SEVERITY_ERROR] = DIAG_SEVERITY_ERROR,
    [ANOMALY_SEVERITY_CRITICAL] = DIAG_SEVERITY_CRITICAL
};

/**
 * @brief Agent severity to diagnostic severity mapping
 */
static const diag_severity_t agent_severity_map[] = {
    [HEALTH_SEVERITY_INFO] = DIAG_SEVERITY_INFO,
    [HEALTH_SEVERITY_WARNING] = DIAG_SEVERITY_WARNING,
    [HEALTH_SEVERITY_ERROR] = DIAG_SEVERITY_ERROR,
    [HEALTH_SEVERITY_CRITICAL] = DIAG_SEVERITY_CRITICAL,
    [HEALTH_SEVERITY_FATAL] = DIAG_SEVERITY_FATAL
};

/* ============================================================================
 * Internal Helper Functions
 * ============================================================================ */

/* Forward declarations */
static int analyze_patterns_unlocked(
    health_diag_bridge_t* bridge,
    diagnostic_result_t* result
);

/**
 * @brief Find anomaly mapping (custom first, then default)
 */
static const anomaly_error_mapping_t* find_anomaly_mapping(
    const health_diag_bridge_t* bridge,
    anomaly_type_t type
) {
    /* Search custom mappings first */
    for (uint32_t i = 0; i < bridge->custom_anomaly_mapping_count; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && bridge->custom_anomaly_mapping_count > 256) {
            health_diagnostic_bridge_heartbeat("health_diagn_loop",
                             (float)(i + 1) / (float)bridge->custom_anomaly_mapping_count);
        }

        if (bridge->custom_anomaly_mappings[i].anomaly_type == type) {
            return &bridge->custom_anomaly_mappings[i];
        }
    }

    /* Search default mappings */
    for (size_t i = 0; i < default_anomaly_mapping_count; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && default_anomaly_mapping_count > 256) {
            health_diagnostic_bridge_heartbeat("health_diagn_loop",
                             (float)(i + 1) / (float)default_anomaly_mapping_count);
        }

        if (default_anomaly_mappings[i].anomaly_type == type) {
            return &default_anomaly_mappings[i];
        }
    }

    /* Return unknown mapping as fallback */
    for (size_t i = 0; i < default_anomaly_mapping_count; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && default_anomaly_mapping_count > 256) {
            health_diagnostic_bridge_heartbeat("health_diagn_loop",
                             (float)(i + 1) / (float)default_anomaly_mapping_count);
        }

        if (default_anomaly_mappings[i].anomaly_type == ANOMALY_UNKNOWN) {
            return &default_anomaly_mappings[i];
        }
    }

    return NULL;
}

/**
 * @brief Find agent message mapping (custom first, then default)
 */
static const agent_error_mapping_t* find_agent_mapping(
    const health_diag_bridge_t* bridge,
    health_agent_msg_type_t type
) {
    /* Search custom mappings first */
    for (uint32_t i = 0; i < bridge->custom_agent_mapping_count; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && bridge->custom_agent_mapping_count > 256) {
            health_diagnostic_bridge_heartbeat("health_diagn_loop",
                             (float)(i + 1) / (float)bridge->custom_agent_mapping_count);
        }

        if (bridge->custom_agent_mappings[i].msg_type == type) {
            return &bridge->custom_agent_mappings[i];
        }
    }

    /* Search default mappings */
    for (size_t i = 0; i < default_agent_mapping_count; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && default_agent_mapping_count > 256) {
            health_diagnostic_bridge_heartbeat("health_diagn_loop",
                             (float)(i + 1) / (float)default_agent_mapping_count);
        }

        if (default_agent_mappings[i].msg_type == type) {
            return &default_agent_mappings[i];
        }
    }

    return NULL;
}

/**
 * @brief Allocate and initialize diagnostic result
 */
static diagnostic_result_t* allocate_diagnostic_result(void) {
    diagnostic_result_t* result = nimcp_calloc(1, sizeof(diagnostic_result_t));
    if (!result) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "result is NULL");

        return NULL;
    }

    /* Set version */
    snprintf(result->diagnostic_version, sizeof(result->diagnostic_version),
             "%s", HEALTH_DIAG_BRIDGE_VERSION);

    /* Initialize timestamp */
    result->timestamp = time(NULL);

    /* Generate unique ID */
    result->error_id = (uint64_t)nimcp_time_get_us();

    return result;
}

/**
 * @brief Capture current stack trace into diagnostic result
 */
static int capture_stack_trace(diagnostic_result_t* result) {
    if (!result) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "result is NULL");

        return -1;
    }

    void* buffer[MAX_STACK_DEPTH];
    int depth = backtrace(buffer, MAX_STACK_DEPTH);
    if (depth <= 0) {
        return -1;
    }

    result->stack_depth = (uint32_t)depth;

    /* Get symbols for better debugging */
    char** symbols = backtrace_symbols(buffer, depth);

    for (int i = 0; i < depth && i < MAX_STACK_DEPTH; i++) {
        result->stack_trace[i].address = buffer[i];
        result->stack_trace[i].is_symbolicated = (symbols != NULL);

        if (symbols && symbols[i]) {
            snprintf(result->stack_trace[i].function_name,
                     sizeof(result->stack_trace[i].function_name),
                     "%s", symbols[i]);
        }
    }

    if (symbols) {
        free(symbols);
    }

    return 0;
}

/**
 * @brief Capture memory snapshot into diagnostic result
 */
static int capture_memory_snapshot(diagnostic_result_t* result) {
    if (!result) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "result is NULL");

        return -1;
    }

    /* Get memory stats from nimcp_memory */
    nimcp_memory_stats_t stats;
    if (nimcp_memory_get_stats(&stats)) {
        result->memory_state.total_allocated_bytes = stats.total_allocated;
        result->memory_state.peak_allocated_bytes = stats.peak_allocated;
        result->memory_state.allocation_count = (uint32_t)stats.allocation_count;
        result->memory_state.deallocation_count = (uint32_t)stats.free_count;

        /* Calculate potential leaks */
        if (stats.allocation_count > stats.free_count) {
            result->memory_state.leaked_blocks =
                (uint32_t)(stats.allocation_count - stats.free_count);
        }
    }

    return 0;
}

/* ============================================================================
 * Lifecycle API Implementation
 * ============================================================================ */

int health_diag_bridge_default_config(health_diag_bridge_config_t* config) {
    if (!config) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "config is NULL");

        return -1;
    }

    /* Phase 8: Heartbeat at operation start */
    health_diagnostic_bridge_heartbeat("health_diagn_health_diag_bridge_d", 0.0f);


    memset(config, 0, sizeof(*config));

    config->capture_stack_trace = true;
    config->capture_memory_snapshot = true;
    config->default_confidence = 0.7f;
    config->enable_pattern_analysis = true;
    config->escalation_threshold = 3;
    config->min_severity = ANOMALY_SEVERITY_WARNING;
    config->min_agent_severity = HEALTH_SEVERITY_WARNING;
    config->enable_bio_async = true;
    config->verbose_logging = false;

    return 0;
}

health_diag_bridge_t* health_diag_bridge_create(
    const health_diag_bridge_config_t* config
) {
    /* Phase 8: Heartbeat at operation start */
    health_diagnostic_bridge_heartbeat("health_diagn_health_diag_bridge_c", 0.0f);


    health_diag_bridge_t* bridge = nimcp_calloc(1, sizeof(health_diag_bridge_t));
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return NULL;
    }

    bridge->magic = HEALTH_DIAG_BRIDGE_MAGIC;

    /* Apply config or defaults */
    if (config) {
        bridge->config = *config;
    } else {
        health_diag_bridge_default_config(&bridge->config);
    }

    /* Allocate custom mapping arrays */
    bridge->custom_anomaly_mappings = nimcp_calloc(
        HEALTH_DIAG_BRIDGE_MAX_MAPPINGS, sizeof(anomaly_error_mapping_t));
    bridge->custom_agent_mappings = nimcp_calloc(
        HEALTH_DIAG_BRIDGE_MAX_MAPPINGS, sizeof(agent_error_mapping_t));

    if (!bridge->custom_anomaly_mappings || !bridge->custom_agent_mappings) {
        health_diag_bridge_destroy(bridge);
        return NULL;
    }

    /* Create mutex */
    /* Initialize base bridge */
    if (bridge_base_init(&bridge->base, 0, "health_diagnostic") != 0) {
        health_diag_bridge_destroy(bridge);
        return NULL;
    }

    bridge->initialized = true;
    return bridge;
}

void health_diag_bridge_destroy(health_diag_bridge_t* bridge) {
    if (!bridge) {
        return;
    }

    /* Phase 8: Heartbeat at operation start */
    health_diagnostic_bridge_heartbeat("health_diagn_health_diag_bridge_d", 0.0f);


    if (bridge->magic != HEALTH_DIAG_BRIDGE_MAGIC) {
        return;
    }

    bridge->magic = 0;
    bridge->initialized = false;

    if (bridge->base.mutex) {
        bridge_base_cleanup(&bridge->base);
        bridge->base.mutex = NULL;
    }

    if (bridge->custom_anomaly_mappings) {
        nimcp_free(bridge->custom_anomaly_mappings);
    }

    if (bridge->custom_agent_mappings) {
        nimcp_free(bridge->custom_agent_mappings);
    }

    nimcp_free(bridge);
}

/* ============================================================================
 * Conversion API - Anomaly to Diagnostic
 * ============================================================================ */

int health_diag_bridge_convert_anomaly(
    health_diag_bridge_t* bridge,
    const anomaly_t* anomaly,
    diagnostic_result_t** result
) {
    if (!bridge || !anomaly || !result) {
        return -1;
    }

    if (!bridge->initialized) {
        return -1;
    }

    /* Check minimum severity filter */
    /* Phase 8: Heartbeat at operation start */
    health_diagnostic_bridge_heartbeat("health_diagn_health_diag_bridge_c", 0.0f);


    if (anomaly->severity < bridge->config.min_severity) {
        return -1;
    }

    uint64_t start_time = nimcp_time_get_us();

    nimcp_mutex_lock(bridge->base.mutex);

    /* Allocate result */
    diagnostic_result_t* diag = allocate_diagnostic_result();
    if (!diag) {
        nimcp_mutex_unlock(bridge->base.mutex);
        return -1;
    }

    /* Find mapping for this anomaly type */
    const anomaly_error_mapping_t* mapping = find_anomaly_mapping(bridge, anomaly->type);
    if (!mapping) {
        nimcp_free(diag);
        nimcp_mutex_unlock(bridge->base.mutex);
        bridge->stats.conversions_failed++;
        return -1;
    }

    /* Fill in diagnostic result */
    diag->error_type = mapping->error_type;
    diag->severity = health_diag_bridge_translate_anomaly_severity(anomaly->severity);

    /* Set confidence with clamping to [0.0, 1.0] */
    float conf = anomaly->confidence > 0 ? anomaly->confidence : mapping->default_confidence;
    if (conf > 1.0f) conf = 1.0f;
    if (conf < 0.0f) conf = 0.0f;
    diag->confidence = conf;

    /* Copy root cause from anomaly description */
    snprintf(diag->root_cause, sizeof(diag->root_cause), "%s", anomaly->description);

    /* Fill symptoms */
    snprintf(diag->symptoms, sizeof(diag->symptoms),
             "Metric value: %.4f (expected: %.4f, deviation: %.4f)",
             anomaly->metric_value, anomaly->expected_value, anomaly->deviation);

    /* Copy affected component */
    snprintf(diag->likely_faulty_function, sizeof(diag->likely_faulty_function),
             "%s", anomaly->affected_component);

    /* Capture stack trace if configured */
    if (bridge->config.capture_stack_trace) {
        capture_stack_trace(diag);
        bridge->stats.stack_traces_captured++;
    }

    /* Capture memory snapshot if configured */
    if (bridge->config.capture_memory_snapshot) {
        capture_memory_snapshot(diag);
        bridge->stats.memory_snapshots_captured++;
    }

    /* Analyze patterns if configured */
    if (bridge->config.enable_pattern_analysis) {
        analyze_patterns_unlocked(bridge, diag);
    }

    /* Suggest recovery actions */
    diagnostics_suggest_recovery(diag);

    /* Update statistics */
    bridge->stats.anomalies_converted++;
    if (anomaly->type <= ANOMALY_UNKNOWN) {
        bridge->stats.by_anomaly_type[anomaly->type]++;
    }
    if (diag->severity <= DIAG_SEVERITY_FATAL) {
        bridge->stats.by_severity[diag->severity]++;
    }

    /* Update timing stats */
    uint64_t elapsed = nimcp_time_get_us() - start_time;
    bridge->total_conversion_time_us += elapsed;
    bridge->conversion_count++;
    bridge->stats.avg_conversion_time_us =
        (float)bridge->total_conversion_time_us / (float)bridge->conversion_count;

    nimcp_mutex_unlock(bridge->base.mutex);

    *result = diag;
    return 0;
}

int health_diag_bridge_convert_anomalies(
    health_diag_bridge_t* bridge,
    const anomaly_t* anomalies,
    uint32_t anomaly_count,
    diagnostic_result_t** results,
    uint32_t* converted_count
) {
    if (!bridge || !anomalies || !results || !converted_count) {
        return -1;
    }

    *converted_count = 0;

    /* Phase 8: Heartbeat at operation start */
    health_diagnostic_bridge_heartbeat("health_diagn_health_diag_bridge_c", 0.0f);


    for (uint32_t i = 0; i < anomaly_count; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && anomaly_count > 256) {
            health_diagnostic_bridge_heartbeat("health_diagn_loop",
                             (float)(i + 1) / (float)anomaly_count);
        }

        if (health_diag_bridge_convert_anomaly(bridge, &anomalies[i], &results[i]) == 0) {
            (*converted_count)++;
        } else {
            results[i] = NULL;
        }
    }

    return 0;
}

/* ============================================================================
 * Conversion API - Health Agent Message to Diagnostic
 * ============================================================================ */

int health_diag_bridge_convert_agent_message(
    health_diag_bridge_t* bridge,
    const health_agent_message_t* message,
    diagnostic_result_t** result
) {
    if (!bridge || !message || !result) {
        return -1;
    }

    if (!bridge->initialized) {
        return -1;
    }

    /* Check minimum severity filter */
    /* Phase 8: Heartbeat at operation start */
    health_diagnostic_bridge_heartbeat("health_diagn_health_diag_bridge_c", 0.0f);


    if (message->severity < bridge->config.min_agent_severity) {
        return -1;
    }

    uint64_t start_time = nimcp_time_get_us();

    nimcp_mutex_lock(bridge->base.mutex);

    /* Allocate result */
    diagnostic_result_t* diag = allocate_diagnostic_result();
    if (!diag) {
        nimcp_mutex_unlock(bridge->base.mutex);
        return -1;
    }

    /* Find mapping for this message type */
    const agent_error_mapping_t* mapping = find_agent_mapping(bridge, message->type);
    if (mapping) {
        diag->error_type = mapping->error_type;
        diag->confidence = mapping->default_confidence;
    } else {
        diag->error_type = ERROR_TYPE_UNKNOWN;
        diag->confidence = bridge->config.default_confidence;
    }

    /* Translate severity */
    diag->severity = health_diag_bridge_translate_agent_severity(message->severity);

    /* Copy description as root cause */
    snprintf(diag->root_cause, sizeof(diag->root_cause), "%s", message->description);

    /* Extract type-specific details into symptoms */
    switch (message->type) {
        case HEALTH_MSG_MEMORY_CORRUPTION:
            snprintf(diag->symptoms, sizeof(diag->symptoms),
                     "Memory corruption at %p, size=%zu, expected_canary=0x%llX, actual=0x%llX",
                     message->data.memory.address,
                     message->data.memory.size,
                     (unsigned long long)message->data.memory.expected_canary,
                     (unsigned long long)message->data.memory.actual_canary);
            diag->memory_corruption_detected = true;
            diag->fault_address = message->data.memory.address;
            break;

        case HEALTH_MSG_DEADLOCK_DETECTED:
            snprintf(diag->symptoms, sizeof(diag->symptoms),
                     "Deadlock between threads %llu and %llu on mutexes %p and %p",
                     (unsigned long long)message->data.deadlock.thread_id_1,
                     (unsigned long long)message->data.deadlock.thread_id_2,
                     message->data.deadlock.mutex_1,
                     message->data.deadlock.mutex_2);
            break;

        case HEALTH_MSG_NAN_DETECTED:
            snprintf(diag->symptoms, sizeof(diag->symptoms),
                     "NaN in neuron %u layer %u, last valid=%.6f, nan_count=%u",
                     message->data.nan.neuron_id,
                     message->data.nan.layer_id,
                     message->data.nan.last_valid_value,
                     message->data.nan.nan_count);
            break;

        case HEALTH_MSG_HEARTBEAT_TIMEOUT:
            snprintf(diag->symptoms, sizeof(diag->symptoms),
                     "Heartbeat timeout: last=%llu us, threshold=%llu us, missed=%u",
                     (unsigned long long)message->data.heartbeat.last_heartbeat_us,
                     (unsigned long long)message->data.heartbeat.timeout_threshold_us,
                     message->data.heartbeat.missed_beats);
            break;

        case HEALTH_MSG_RESOURCE_EXHAUSTION:
            snprintf(diag->symptoms, sizeof(diag->symptoms),
                     "Resource exhaustion: used=%llu, limit=%llu, util=%.1f%%, TTX=%u ms",
                     (unsigned long long)message->data.resource.memory_used,
                     (unsigned long long)message->data.resource.memory_limit,
                     message->data.resource.utilization_pct,
                     message->data.resource.time_to_exhaust_ms);
            break;

        default:
            snprintf(diag->symptoms, sizeof(diag->symptoms), "%s", message->description);
            break;
    }

    /* Capture stack trace if configured */
    if (bridge->config.capture_stack_trace) {
        capture_stack_trace(diag);
        bridge->stats.stack_traces_captured++;
    }

    /* Capture memory snapshot if configured */
    if (bridge->config.capture_memory_snapshot) {
        capture_memory_snapshot(diag);
        bridge->stats.memory_snapshots_captured++;
    }

    /* Analyze patterns if configured */
    if (bridge->config.enable_pattern_analysis) {
        analyze_patterns_unlocked(bridge, diag);
    }

    /* Suggest recovery actions */
    diagnostics_suggest_recovery(diag);

    /* Update statistics */
    bridge->stats.agent_messages_converted++;
    if (message->type < HEALTH_MSG_COUNT) {
        bridge->stats.by_agent_msg_type[message->type]++;
    }
    if (diag->severity <= DIAG_SEVERITY_FATAL) {
        bridge->stats.by_severity[diag->severity]++;
    }

    /* Update timing stats */
    uint64_t elapsed = nimcp_time_get_us() - start_time;
    bridge->total_conversion_time_us += elapsed;
    bridge->conversion_count++;
    bridge->stats.avg_conversion_time_us =
        (float)bridge->total_conversion_time_us / (float)bridge->conversion_count;

    nimcp_mutex_unlock(bridge->base.mutex);

    *result = diag;
    return 0;
}

/* ============================================================================
 * Enrichment API
 * ============================================================================ */

int health_diag_bridge_enrich_stack_trace(
    health_diag_bridge_t* bridge,
    diagnostic_result_t* result
) {
    if (!bridge || !result) {
        return -1;
    }

    /* Phase 8: Heartbeat at operation start */
    health_diagnostic_bridge_heartbeat("health_diagn_health_diag_bridge_e", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    int ret = capture_stack_trace(result);
    if (ret == 0) {
        bridge->stats.stack_traces_captured++;
    }
    nimcp_mutex_unlock(bridge->base.mutex);

    return ret;
}

int health_diag_bridge_enrich_memory_snapshot(
    health_diag_bridge_t* bridge,
    diagnostic_result_t* result
) {
    if (!bridge || !result) {
        return -1;
    }

    /* Phase 8: Heartbeat at operation start */
    health_diagnostic_bridge_heartbeat("health_diagn_health_diag_bridge_e", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    int ret = capture_memory_snapshot(result);
    if (ret == 0) {
        bridge->stats.memory_snapshots_captured++;
    }
    nimcp_mutex_unlock(bridge->base.mutex);

    return ret;
}

/**
 * @brief Analyze patterns without locking (internal helper)
 */
static int analyze_patterns_unlocked(
    health_diag_bridge_t* bridge,
    diagnostic_result_t* result
) {
    /* Pattern analysis is delegated to diagnostics system */
    /* This function marks the result for later pattern matching */
    result->is_recurring = false;
    result->occurrence_count = 1;
    result->first_occurrence = result->timestamp;
    result->last_occurrence = result->timestamp;

    /* Actual pattern detection happens when diagnostic is added to history */
    /* Here we just increment the counter for tracking */
    bridge->stats.patterns_detected++;

    return 0;
}

int health_diag_bridge_analyze_patterns(
    health_diag_bridge_t* bridge,
    diagnostic_result_t* result
) {
    if (!bridge || !result) {
        return -1;
    }

    /* Phase 8: Heartbeat at operation start */
    health_diagnostic_bridge_heartbeat("health_diagn_health_diag_bridge_a", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    int ret = analyze_patterns_unlocked(bridge, result);
    nimcp_mutex_unlock(bridge->base.mutex);

    return ret;
}

/* ============================================================================
 * Mapping Configuration API
 * ============================================================================ */

int health_diag_bridge_add_anomaly_mapping(
    health_diag_bridge_t* bridge,
    const anomaly_error_mapping_t* mapping
) {
    if (!bridge || !mapping) {
        return -1;
    }

    /* Phase 8: Heartbeat at operation start */
    health_diagnostic_bridge_heartbeat("health_diagn_health_diag_bridge_a", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);

    if (bridge->custom_anomaly_mapping_count >= HEALTH_DIAG_BRIDGE_MAX_MAPPINGS) {
        nimcp_mutex_unlock(bridge->base.mutex);
        return -1;
    }

    bridge->custom_anomaly_mappings[bridge->custom_anomaly_mapping_count++] = *mapping;

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int health_diag_bridge_add_agent_mapping(
    health_diag_bridge_t* bridge,
    const agent_error_mapping_t* mapping
) {
    if (!bridge || !mapping) {
        return -1;
    }

    /* Phase 8: Heartbeat at operation start */
    health_diagnostic_bridge_heartbeat("health_diagn_health_diag_bridge_a", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);

    if (bridge->custom_agent_mapping_count >= HEALTH_DIAG_BRIDGE_MAX_MAPPINGS) {
        nimcp_mutex_unlock(bridge->base.mutex);
        return -1;
    }

    bridge->custom_agent_mappings[bridge->custom_agent_mapping_count++] = *mapping;

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

const anomaly_error_mapping_t* health_diag_bridge_get_anomaly_mapping(
    const health_diag_bridge_t* bridge,
    anomaly_type_t anomaly_type
) {
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return NULL;
    }

    /* Phase 8: Heartbeat at operation start */
    health_diagnostic_bridge_heartbeat("health_diagn_health_diag_bridge_g", 0.0f);


    return find_anomaly_mapping(bridge, anomaly_type);
}

/* ============================================================================
 * Severity Translation API
 * ============================================================================ */

diag_severity_t health_diag_bridge_translate_anomaly_severity(
    anomaly_severity_t anomaly_severity
) {
    /* Phase 8: Heartbeat at operation start */
    health_diagnostic_bridge_heartbeat("health_diagn_health_diag_bridge_t", 0.0f);


    if (anomaly_severity <= ANOMALY_SEVERITY_CRITICAL) {
        return anomaly_severity_map[anomaly_severity];
    }
    return DIAG_SEVERITY_INFO;
}

diag_severity_t health_diag_bridge_translate_agent_severity(
    health_agent_severity_t agent_severity
) {
    /* Phase 8: Heartbeat at operation start */
    health_diagnostic_bridge_heartbeat("health_diagn_health_diag_bridge_t", 0.0f);


    if (agent_severity <= HEALTH_SEVERITY_FATAL) {
        return agent_severity_map[agent_severity];
    }
    return DIAG_SEVERITY_INFO;
}

/* ============================================================================
 * Statistics and Query API
 * ============================================================================ */

int health_diag_bridge_get_stats(
    const health_diag_bridge_t* bridge,
    health_diag_bridge_stats_t* stats
) {
    if (!bridge || !stats) {
        return -1;
    }

    /* Phase 8: Heartbeat at operation start */
    health_diagnostic_bridge_heartbeat("health_diagn_health_diag_bridge_g", 0.0f);


    nimcp_mutex_lock((nimcp_mutex_t*)bridge->base.mutex);
    *stats = bridge->stats;
    nimcp_mutex_unlock((nimcp_mutex_t*)bridge->base.mutex);

    return 0;
}

void health_diag_bridge_reset_stats(health_diag_bridge_t* bridge) {
    if (!bridge) {
        return;
    }

    /* Phase 8: Heartbeat at operation start */
    health_diagnostic_bridge_heartbeat("health_diagn_health_diag_bridge_r", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    bridge->total_conversion_time_us = 0;
    bridge->conversion_count = 0;
    nimcp_mutex_unlock(bridge->base.mutex);
}

bool health_diag_bridge_is_ready(const health_diag_bridge_t* bridge) {
    if (!bridge) {
        return false;
    }
    /* Phase 8: Heartbeat at operation start */
    health_diagnostic_bridge_heartbeat("health_diagn_health_diag_bridge_i", 0.0f);


    return bridge->initialized && bridge->magic == HEALTH_DIAG_BRIDGE_MAGIC;
}

/* ============================================================================
 * Utility Functions
 * ============================================================================ */

const char* health_diag_bridge_anomaly_type_name(anomaly_type_t type) {
    switch (type) {
        case ANOMALY_NONE: return "NONE";
        case ANOMALY_MEMORY_LEAK: return "MEMORY_LEAK";
        case ANOMALY_PERFORMANCE_DEGRADATION: return "PERFORMANCE_DEGRADATION";
        case ANOMALY_ERROR_SPIKE: return "ERROR_SPIKE";
        case ANOMALY_THROUGHPUT_DROP: return "THROUGHPUT_DROP";
        case ANOMALY_CACHE_THRASHING: return "CACHE_THRASHING";
        case ANOMALY_RESOURCE_EXHAUSTION: return "RESOURCE_EXHAUSTION";
        case ANOMALY_NUMERICAL_INSTABILITY: return "NUMERICAL_INSTABILITY";
        case ANOMALY_THREAD_CONTENTION: return "THREAD_CONTENTION";
        case ANOMALY_UNKNOWN: return "UNKNOWN";
        default: return "INVALID";
    }
}

const char* health_diag_bridge_agent_msg_type_name(health_agent_msg_type_t type) {
    switch (type) {
        case HEALTH_MSG_ANOMALY_DETECTED: return "ANOMALY_DETECTED";
        case HEALTH_MSG_CYTOKINE_SIGNAL: return "CYTOKINE_SIGNAL";
        case HEALTH_MSG_EMERGENCY: return "EMERGENCY";
        case HEALTH_MSG_RECOVERY_REQUEST: return "RECOVERY_REQUEST";
        case HEALTH_MSG_STATE_CORRUPTION: return "STATE_CORRUPTION";
        case HEALTH_MSG_HEARTBEAT_TIMEOUT: return "HEARTBEAT_TIMEOUT";
        case HEALTH_MSG_DEADLOCK_DETECTED: return "DEADLOCK_DETECTED";
        case HEALTH_MSG_NAN_DETECTED: return "NAN_DETECTED";
        case HEALTH_MSG_MEMORY_CORRUPTION: return "MEMORY_CORRUPTION";
        case HEALTH_MSG_RESOURCE_EXHAUSTION: return "RESOURCE_EXHAUSTION";
        case HEALTH_MSG_STATUS_UPDATE: return "STATUS_UPDATE";
        default: return "UNKNOWN";
    }
}

const char* health_diag_bridge_version(void) {
    return HEALTH_DIAG_BRIDGE_VERSION;
}
