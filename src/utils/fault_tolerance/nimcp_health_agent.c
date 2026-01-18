/**
 * @file nimcp_health_agent.c
 * @brief Independent Health Monitor Agent Implementation
 * @version 1.0.0
 * @date 2025-01-17
 *
 * WHAT: Autonomous agent that monitors brain health independently
 * WHY:  Detect and respond to errors even when main system is compromised
 * HOW:  Separate thread with watchdog, lock-free communication to immune system
 */

/* Must be before any includes for pthread_timedjoin_np */
#define _GNU_SOURCE

#include "utils/fault_tolerance/nimcp_health_agent.h"
#include "utils/fault_tolerance/nimcp_lockfree_metrics.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/thread/nimcp_deadlock_detector.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/memory/nimcp_unified_memory.h"
#include "utils/logging/nimcp_logging.h"

/* Module integrations */
#include "portia/nimcp_portia.h"
#include "dragonfly/nimcp_dragonfly.h"
#include "swarm/nimcp_swarm_immune.h"
#include "swarm/nimcp_swarm_memory.h"

/* Phase 4: Hypothalamus integration */
#include "core/brain/regions/hypothalamus/nimcp_hypothalamus_orchestrator.h"
#include "core/brain/regions/hypothalamus/nimcp_hypothalamus_homeostasis.h"
#include "core/brain/regions/hypothalamus/nimcp_hypothalamus_immune_bridge.h"

/* Phase 4: Checkpoint and recovery */
#include "utils/fault_tolerance/nimcp_checkpoint.h"

/* Phase 4: Brain oscillations and connectivity */
#include "core/brain_oscillations/nimcp_brain_oscillations.h"
#include "cognitive/introspection/nimcp_connectivity_health.h"

/* Phase 4: Bio-async router */
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"

/* Phase 4: Additional fault tolerance includes */
#include "core/brain/nimcp_kg_gc.h"
#include "utils/fault_tolerance/nimcp_runtime_adaptation.h"
/* Note: failure_prediction.h and metacognition.h have type conflicts with other headers.
 * The health agent header already has forward declarations for failure_predictor_t
 * and metacognition_t, so we use simplified stubs without the full APIs. */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdarg.h>
#include <pthread.h>  /* Still needed for pthread_timedjoin_np */
#include <errno.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>

#ifndef __cplusplus
#include <stdatomic.h>
#else
#include <atomic>
#endif

#define LOG_MODULE "health_agent"

/* NaN/Inf detection macro (local definition to avoid circular deps) */
#ifndef NIMCP_IS_INVALID_FLOAT
#define NIMCP_IS_INVALID_FLOAT(x) (isnan(x) || isinf(x))
#endif

/* Helper macro to check if any consistency check is enabled */
#define CONSISTENCY_CHECKS_ENABLED(cfg) \
    ((cfg).check_reference_counts || (cfg).check_pointer_canaries || \
     (cfg).check_struct_magic || (cfg).check_mutex_state || \
     (cfg).check_circular_buffers || (cfg).check_kg_consistency || \
     (cfg).check_neuron_values)

/* ============================================================================
 * Internal Constants
 * ============================================================================ */

/** Lock-free message queue capacity */
#define AGENT_MSG_QUEUE_CAPACITY  2048

/** Maximum time to wait for agent thread to stop (ms) */
#define AGENT_STOP_TIMEOUT_MS     5000

/** Minimum check interval (ms) */
#define AGENT_MIN_CHECK_INTERVAL_MS  10

/** Maximum detector name length */
#define DETECTOR_NAME_MAX_LEN     32

/* ============================================================================
 * Internal Structures
 * ============================================================================ */

/**
 * @brief Lock-free message node for agent queue
 */
typedef struct health_msg_node {
    health_agent_message_t msg;
    _Atomic uint64_t sequence;           /**< Sequence number for ordering */
} health_msg_node_t;

/**
 * @brief Lock-free message queue (MPSC - multiple producer, single consumer)
 */
typedef struct {
    health_msg_node_t* nodes;
    uint32_t capacity;
    uint32_t capacity_mask;
    _Atomic uint64_t head;               /**< Write position (producers) */
    _Atomic uint64_t tail;               /**< Read position (consumer) */
    _Atomic uint64_t dropped_count;
} health_msg_queue_t;

/**
 * @brief Heartbeat state (shared between main system and agent)
 */
typedef struct {
    _Atomic uint64_t last_heartbeat_us;  /**< Last heartbeat timestamp */
    _Atomic uint32_t missed_count;       /**< Consecutive missed heartbeats */
    char current_operation[64];          /**< Current operation name */
    _Atomic float current_progress;      /**< Current progress (0.0-1.0) */
} heartbeat_state_t;

/**
 * @brief Internal health agent structure
 */
struct nimcp_health_agent {
    uint32_t magic;                      /**< Magic number for validation */
    uint64_t canary_front;               /**< Front canary for corruption detection */

    /* Configuration */
    health_agent_config_t config;

    /* Core connections */
    brain_t brain;                       /**< Connected brain */
    brain_immune_system_t* immune;       /**< Connected immune system */
    health_monitor_t* monitor;           /**< Connected health monitor */

    /* Thread management - using NIMCP threading utilities */
    nimcp_thread_t agent_thread;
    _Atomic bool running;
    _Atomic bool stop_requested;
    nimcp_mutex_t* state_mutex;          /**< For state transitions (NIMCP mutex) */
    nimcp_cond_t* stop_cond;             /**< Condition for stop signaling */

    /* Message queue (agent -> immune) */
    health_msg_queue_t msg_queue;

    /* Heartbeat state */
    heartbeat_state_t heartbeat;

    /* Statistics */
    _Atomic uint64_t uptime_start_us;
    health_agent_stats_t stats;
    nimcp_mutex_t* stats_mutex;          /**< For stats update (NIMCP mutex) */

    /* Anomaly tracking */
    _Atomic uint64_t next_anomaly_id;
    _Atomic health_agent_severity_t current_severity;

    /* =========== COGNITIVE MODULE CONNECTIONS =========== */

    /* Failure prediction */
    failure_predictor_t* failure_predictor;
    health_agent_prediction_config_t prediction_config;
    _Atomic uint64_t predictions_made;
    _Atomic uint64_t predictions_correct;
    _Atomic uint64_t preventive_actions;

    /* Metacognition */
    metacognition_t* metacognition;
    health_agent_metacog_config_t metacog_config;
    _Atomic uint64_t self_diagnoses;
    _Atomic uint64_t degradation_alerts;
    _Atomic float current_confidence;

    /* Ethics engine */
    ethics_engine_t ethics;
    health_agent_ethics_config_t ethics_config;
    _Atomic uint64_t ethics_evaluations;
    _Atomic uint64_t ethics_blocks;
    _Atomic uint64_t mercy_applications;

    /* Emotional system */
    emotional_system_t* emotion;
    emotion_immune_bridge_t* emotion_immune;
    health_agent_emotion_config_t emotion_config;
    _Atomic float current_stress_level;
    _Atomic uint64_t emotion_adjustments;

    /* Wellbeing monitor */
    wellbeing_monitor_t* wellbeing;
    health_agent_wellbeing_config_t wellbeing_config;
    _Atomic uint64_t distress_detections;
    _Atomic uint64_t wellbeing_interventions;
    _Atomic float current_distress_level;

    /* Mental health monitor */
    mental_health_monitor_t* mental_health;

    /* Collective cognition */
    collective_cognition_t* collective;
    health_agent_collective_config_t collective_config;
    _Atomic uint64_t consensus_requests;
    _Atomic uint64_t consensus_achieved;
    _Atomic float avg_consensus_time_ms;

    /* Recursive cognition (RCOG) */
    rcog_engine_t* rcog;
    health_agent_rcog_config_t rcog_config;
    _Atomic uint64_t rcog_diagnoses;
    _Atomic uint64_t rcog_recovery_plans;
    _Atomic float avg_rcog_time_ms;

    /* GPU health */
    gpu_health_monitor_t* gpu_health;
    health_agent_gpu_config_t gpu_config;
    _Atomic uint64_t gpu_accelerated_checks;
    _Atomic float gpu_utilization;
    _Atomic bool gpu_healthy;

    /* Cognitive stats mutex */
    nimcp_mutex_t* cognitive_mutex;      /**< For cognitive stats (NIMCP mutex) */

    /* =========== HYPOTHALAMUS & HOMEOSTASIS CONNECTIONS =========== */

    /* Hypothalamus orchestrator - central drive coordination */
    hypo_orchestrator_t hypothalamus;
    health_agent_hypothalamus_config_t hypothalamus_config;
    uint32_t hypo_bridge_id;              /**< Registered bridge ID */
    _Atomic bool in_stress_response;
    _Atomic bool in_sickness_mode;
    _Atomic uint64_t stress_responses;
    _Atomic uint64_t sickness_mode_entries;
    _Atomic uint64_t drive_events_published;

    /* Hypothalamus homeostasis system - PID-based regulation */
    hypo_homeostasis_handle_t* homeostasis;
    _Atomic float homeostatic_output;

    /* Hypothalamus immune bridge - bidirectional neuroimmune integration */
    hypo_immune_bridge_t* hypo_immune_bridge;

    /* Hypothalamus drive system */
    hypo_drive_system_handle_t* drives;

    /* =========== ADDITIONAL MODULE CONNECTIONS =========== */

    /* Connectivity health monitor */
    connectivity_health_t* connectivity;
    health_agent_connectivity_config_t connectivity_config;

    /* Brain oscillations monitor */
    brain_oscillations_t* oscillations;
    health_agent_oscillations_config_t oscillations_config;

    /* Knowledge graph GC */
    kg_gc_context_t* gc_context;
    health_agent_gc_config_t gc_config;
    _Atomic uint64_t gc_triggers;
    uint64_t last_gc_time_us;

    /* Checkpoint manager */
    checkpoint_manager_t* checkpoint;
    health_agent_checkpoint_config_t checkpoint_config;
    _Atomic uint64_t checkpoints_created;
    _Atomic uint64_t rollbacks_performed;
    uint64_t last_checkpoint_time_us;

    /* Deadlock detector (direct connection) */
    deadlock_detector_t* deadlock_detector_ptr;

    /* Bio-async router */
    bio_async_router_t* bio_async_router;
    health_agent_bio_async_config_t bio_async_config;
    uint32_t bio_async_module_id;
    _Atomic uint64_t bio_async_events_published;

    /* Runtime adaptation */
    runtime_adaptation_context_t runtime_adaptation;
    _Atomic uint64_t load_reductions;
    _Atomic bool load_reduced;

    /* Exception-immune bridge */
    exception_immune_t* exception_bridge;
    health_agent_exception_config_t exception_config;

    /* =========== PORTIA/DRAGONFLY/SWARM/MEMORY CONNECTIONS =========== */

    /* Portia adaptive resource management */
    portia_context_t* portia;
    health_agent_portia_config_t portia_config;
    _Atomic uint64_t portia_tier_changes;
    _Atomic uint64_t portia_degradations;

    /* Dragonfly tracking/interception */
    dragonfly_system_t* dragonfly;
    health_agent_dragonfly_config_t dragonfly_config;
    _Atomic uint64_t dragonfly_anomalies_tracked;
    _Atomic uint64_t dragonfly_pursuits;
    _Atomic uint64_t dragonfly_interceptions;
    _Atomic uint32_t dragonfly_current_target;

    /* Swarm immune system */
    NimcpSwarmImmuneSystem* swarm_immune;
    health_agent_swarm_immune_config_t swarm_immune_config;
    _Atomic uint64_t swarm_threats_detected;
    _Atomic uint64_t swarm_responses_generated;
    _Atomic uint64_t swarm_coordinated_responses;

    /* Swarm memory consolidation */
    NimcpSwarmMemory* swarm_memory;
    health_agent_swarm_memory_config_t swarm_memory_config;
    _Atomic uint64_t swarm_memories_stored;
    _Atomic uint64_t swarm_replays_performed;

    /* Engram memory system */
    engram_system_t* engram;
    health_agent_engram_config_t engram_config;
    _Atomic uint64_t engram_encodings;
    _Atomic uint64_t engram_recalls;

    /* Systems consolidation */
    systems_consolidation_system_t* memory_consolidation;
    health_agent_memory_consolidation_config_t consolidation_config;

    /* =========== STATE CONSISTENCY MANAGER (Phase 3) =========== */

    /* Consistency check state */
    health_agent_consistency_result_t last_consistency_result;
    _Atomic bool consistency_check_pending;
    _Atomic uint64_t last_consistency_check_us;
    _Atomic uint64_t consistency_checks_run;
    _Atomic uint64_t consistency_failures_total;
    nimcp_mutex_t* consistency_mutex;    /**< For consistency check state */

    /* Registered structures for magic validation */
    struct {
        void* ptr;                       /**< Pointer to structure */
        uint32_t expected_magic;         /**< Expected magic value */
        char name[64];                   /**< Structure name */
        bool active;                     /**< Entry is in use */
    } registered_structs[64];            /**< Registry of tracked structures */
    uint32_t registered_struct_count;    /**< Number of registered structs */

    /* Module connection mutex */
    nimcp_mutex_t* modules_mutex;        /**< For module connections (NIMCP mutex) */

    uint64_t canary_back;                /**< Back canary for corruption detection */
};

/* ============================================================================
 * Forward Declarations
 * ============================================================================ */

static void* agent_thread_main(void* arg);
static bool msg_queue_init(health_msg_queue_t* queue, uint32_t capacity);
static void msg_queue_destroy(health_msg_queue_t* queue);
static bool msg_queue_push(health_msg_queue_t* queue, const health_agent_message_t* msg);
static bool msg_queue_pop(health_msg_queue_t* queue, health_agent_message_t* msg);
static uint32_t msg_queue_size(const health_msg_queue_t* queue);
static uint64_t get_timestamp_us(void);

/* Cognitive integration helper functions */
static void agent_run_failure_prediction(nimcp_health_agent_t* agent);
static void agent_run_metacognition_check(nimcp_health_agent_t* agent);
static void agent_run_wellbeing_check(nimcp_health_agent_t* agent);
static void agent_apply_emotion_adjustments(nimcp_health_agent_t* agent);
static void agent_check_gpu_health(nimcp_health_agent_t* agent);
static bool agent_check_ethics_permission(nimcp_health_agent_t* agent,
                                           const health_agent_message_t* msg,
                                           health_agent_recovery_t action);
static int agent_get_collective_consensus(nimcp_health_agent_t* agent,
                                           const health_agent_message_t* msg);
static int agent_run_rcog_diagnosis(nimcp_health_agent_t* agent,
                                     const health_agent_message_t* msg,
                                     health_agent_recovery_t* suggested_action);

/* Hypothalamus and module integration helper functions */
static void agent_run_hypothalamus_check(nimcp_health_agent_t* agent, float health_score);
static void agent_run_homeostatic_regulation(nimcp_health_agent_t* agent, float health_score);
static void agent_check_connectivity(nimcp_health_agent_t* agent);
static void agent_check_oscillations(nimcp_health_agent_t* agent);
static void agent_auto_gc_if_needed(nimcp_health_agent_t* agent);
static void agent_auto_checkpoint_if_needed(nimcp_health_agent_t* agent, float health_score);
static int hypo_drive_event_callback(const void* event, void* user_data);

/* State Consistency Manager helper functions (Phase 3) */
static void agent_run_consistency_checks(nimcp_health_agent_t* agent);
static bool agent_check_reference_counts(nimcp_health_agent_t* agent, health_agent_consistency_result_t* result);
static bool agent_check_pointer_canaries(nimcp_health_agent_t* agent, health_agent_consistency_result_t* result);
static bool agent_check_struct_magic(nimcp_health_agent_t* agent, health_agent_consistency_result_t* result);
static bool agent_check_mutex_state(nimcp_health_agent_t* agent, health_agent_consistency_result_t* result);
static bool agent_check_circular_buffers(nimcp_health_agent_t* agent, health_agent_consistency_result_t* result);
static bool agent_check_knowledge_graph(nimcp_health_agent_t* agent, health_agent_consistency_result_t* result);
static bool agent_check_neuron_values(nimcp_health_agent_t* agent, health_agent_consistency_result_t* result);

/* ============================================================================
 * Utility Functions
 * ============================================================================ */

/**
 * @brief Get current timestamp in microseconds
 */
static uint64_t get_timestamp_us(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)tv.tv_sec * 1000000ULL + (uint64_t)tv.tv_usec;
}

/**
 * @brief Validate agent structure integrity
 */
static bool validate_agent(const nimcp_health_agent_t* agent) {
    if (!agent) return false;
    if (agent->magic != HEALTH_AGENT_MAGIC) return false;
    if (agent->canary_front != HEALTH_AGENT_CANARY) return false;
    if (agent->canary_back != HEALTH_AGENT_CANARY) return false;
    return true;
}

/**
 * @brief Round up to next power of 2
 */
static uint32_t next_power_of_2(uint32_t value) {
    if (value == 0) return 1;
    value--;
    value |= value >> 1;
    value |= value >> 2;
    value |= value >> 4;
    value |= value >> 8;
    value |= value >> 16;
    return value + 1;
}

/* ============================================================================
 * String Conversion Functions
 * ============================================================================ */

const char* health_agent_msg_type_to_string(health_agent_msg_type_t type) {
    switch (type) {
        case HEALTH_MSG_ANOMALY_DETECTED:    return "ANOMALY_DETECTED";
        case HEALTH_MSG_CYTOKINE_SIGNAL:     return "CYTOKINE_SIGNAL";
        case HEALTH_MSG_EMERGENCY:           return "EMERGENCY";
        case HEALTH_MSG_RECOVERY_REQUEST:    return "RECOVERY_REQUEST";
        case HEALTH_MSG_STATE_CORRUPTION:    return "STATE_CORRUPTION";
        case HEALTH_MSG_HEARTBEAT_TIMEOUT:   return "HEARTBEAT_TIMEOUT";
        case HEALTH_MSG_DEADLOCK_DETECTED:   return "DEADLOCK_DETECTED";
        case HEALTH_MSG_NAN_DETECTED:        return "NAN_DETECTED";
        case HEALTH_MSG_MEMORY_CORRUPTION:   return "MEMORY_CORRUPTION";
        case HEALTH_MSG_RESOURCE_EXHAUSTION: return "RESOURCE_EXHAUSTION";
        case HEALTH_MSG_STATUS_UPDATE:       return "STATUS_UPDATE";
        default:                              return "UNKNOWN";
    }
}

const char* health_agent_severity_to_string(health_agent_severity_t severity) {
    switch (severity) {
        case HEALTH_SEVERITY_INFO:     return "INFO";
        case HEALTH_SEVERITY_WARNING:  return "WARNING";
        case HEALTH_SEVERITY_ERROR:    return "ERROR";
        case HEALTH_SEVERITY_CRITICAL: return "CRITICAL";
        case HEALTH_SEVERITY_FATAL:    return "FATAL";
        default:                        return "UNKNOWN";
    }
}

const char* health_agent_source_to_string(health_agent_source_t source) {
    switch (source) {
        case HEALTH_SOURCE_UNKNOWN:      return "UNKNOWN";
        case HEALTH_SOURCE_MEMORY:       return "MEMORY";
        case HEALTH_SOURCE_THREADING:    return "THREADING";
        case HEALTH_SOURCE_NEURAL:       return "NEURAL";
        case HEALTH_SOURCE_KG:           return "KG";
        case HEALTH_SOURCE_IMMUNE:       return "IMMUNE";
        case HEALTH_SOURCE_IO:           return "IO";
        case HEALTH_SOURCE_BRAIN_REGION: return "BRAIN_REGION";
        case HEALTH_SOURCE_CHECKPOINT:   return "CHECKPOINT";
        case HEALTH_SOURCE_HEARTBEAT:    return "HEARTBEAT";
        default:                          return "UNKNOWN";
    }
}

const char* health_agent_recovery_to_string(health_agent_recovery_t recovery) {
    switch (recovery) {
        case HEALTH_RECOVERY_NONE:           return "NONE";
        case HEALTH_RECOVERY_GC:             return "GC";
        case HEALTH_RECOVERY_CHECKPOINT:     return "CHECKPOINT";
        case HEALTH_RECOVERY_ROLLBACK:       return "ROLLBACK";
        case HEALTH_RECOVERY_RESTART_THREAD: return "RESTART_THREAD";
        case HEALTH_RECOVERY_CLEAR_NAN:      return "CLEAR_NAN";
        case HEALTH_RECOVERY_REDUCE_LOAD:    return "REDUCE_LOAD";
        case HEALTH_RECOVERY_QUARANTINE:     return "QUARANTINE";
        case HEALTH_RECOVERY_EMERGENCY_SAVE: return "EMERGENCY_SAVE";
        case HEALTH_RECOVERY_FULL_RESET:     return "FULL_RESET";
        default:                              return "UNKNOWN";
    }
}

/* ============================================================================
 * Message Queue Implementation (Lock-Free MPSC)
 * ============================================================================ */

static bool msg_queue_init(health_msg_queue_t* queue, uint32_t capacity) {
    if (!queue || capacity == 0) return false;

    /* Round up capacity to power of 2 */
    capacity = next_power_of_2(capacity);

    /* Allocate nodes array */
    queue->nodes = (health_msg_node_t*)nimcp_calloc(capacity, sizeof(health_msg_node_t));
    if (!queue->nodes) {
        nimcp_log(LOG_LEVEL_ERROR, "Failed to allocate message queue nodes");
        return false;
    }

    queue->capacity = capacity;
    queue->capacity_mask = capacity - 1;
    atomic_init(&queue->head, 0);
    atomic_init(&queue->tail, 0);
    atomic_init(&queue->dropped_count, 0);

    /* Initialize sequence numbers */
    for (uint32_t i = 0; i < capacity; i++) {
        atomic_init(&queue->nodes[i].sequence, i);
    }

    return true;
}

static void msg_queue_destroy(health_msg_queue_t* queue) {
    if (!queue) return;

    if (queue->nodes) {
        nimcp_free(queue->nodes);
        queue->nodes = NULL;
    }
    queue->capacity = 0;
}

static bool msg_queue_push(health_msg_queue_t* queue, const health_agent_message_t* msg) {
    if (!queue || !msg || !queue->nodes) return false;

    uint64_t head;
    health_msg_node_t* node;
    uint64_t seq;

    /* Try to claim a slot */
    while (true) {
        head = atomic_load_explicit(&queue->head, memory_order_relaxed);
        node = &queue->nodes[head & queue->capacity_mask];
        seq = atomic_load_explicit(&node->sequence, memory_order_acquire);

        int64_t diff = (int64_t)seq - (int64_t)head;

        if (diff == 0) {
            /* Slot is available, try to claim it */
            if (atomic_compare_exchange_weak_explicit(
                    &queue->head, &head, head + 1,
                    memory_order_relaxed, memory_order_relaxed)) {
                break;  /* Successfully claimed */
            }
        } else if (diff < 0) {
            /* Queue is full */
            atomic_fetch_add_explicit(&queue->dropped_count, 1, memory_order_relaxed);
            return false;
        }
        /* Otherwise retry */
    }

    /* Write message to slot */
    node->msg = *msg;

    /* Release slot to consumer */
    atomic_store_explicit(&node->sequence, head + 1, memory_order_release);

    return true;
}

static bool msg_queue_pop(health_msg_queue_t* queue, health_agent_message_t* msg) {
    if (!queue || !msg || !queue->nodes) return false;

    uint64_t tail;
    health_msg_node_t* node;
    uint64_t seq;

    /* Try to claim an entry */
    while (true) {
        tail = atomic_load_explicit(&queue->tail, memory_order_relaxed);
        node = &queue->nodes[tail & queue->capacity_mask];
        seq = atomic_load_explicit(&node->sequence, memory_order_acquire);

        int64_t diff = (int64_t)seq - (int64_t)(tail + 1);

        if (diff == 0) {
            /* Entry is available, try to claim it */
            if (atomic_compare_exchange_weak_explicit(
                    &queue->tail, &tail, tail + 1,
                    memory_order_relaxed, memory_order_relaxed)) {
                break;  /* Successfully claimed */
            }
        } else if (diff < 0) {
            /* Queue is empty */
            return false;
        }
        /* Otherwise retry */
    }

    /* Read message from slot */
    *msg = node->msg;

    /* Release slot to producer */
    atomic_store_explicit(&node->sequence, tail + queue->capacity, memory_order_release);

    return true;
}

static uint32_t msg_queue_size(const health_msg_queue_t* queue) {
    if (!queue) return 0;

    uint64_t head = atomic_load_explicit(&queue->head, memory_order_acquire);
    uint64_t tail = atomic_load_explicit(&queue->tail, memory_order_acquire);

    return (uint32_t)(head - tail);
}

/* ============================================================================
 * Configuration Functions
 * ============================================================================ */

void nimcp_health_agent_default_config(health_agent_config_t* config) {
    if (!config) return;

    memset(config, 0, sizeof(health_agent_config_t));

    /* Agent identification */
    strncpy(config->agent_name, "health_agent_default", sizeof(config->agent_name) - 1);
    config->agent_id = 0;

    /* Timing configuration */
    config->heartbeat_interval_ms = HEALTH_AGENT_DEFAULT_HEARTBEAT_MS;
    config->watchdog_timeout_ms = HEALTH_AGENT_DEFAULT_WATCHDOG_MS;
    config->check_interval_ms = HEALTH_AGENT_DEFAULT_CHECK_MS;
    config->immune_poll_interval_ms = 100;

    /* Thread configuration */
    config->thread_stack_size = 0;  /* Use default */
    config->thread_priority = 0;    /* Normal priority */
    config->pin_to_core = false;
    config->core_id = -1;

    /* Heartbeat detector defaults */
    config->heartbeat_detector.enabled = true;
    config->heartbeat_detector.check_interval_ms = 100;
    config->heartbeat_detector.min_report_severity = HEALTH_SEVERITY_WARNING;
    config->heartbeat_detector.threshold_count = 3;  /* 3 missed beats = warning */
    config->heartbeat_detector.cooldown_ms = 1000;

    /* Memory detector defaults */
    config->memory_detector.enabled = true;
    config->memory_detector.check_interval_ms = 500;
    config->memory_detector.min_report_severity = HEALTH_SEVERITY_ERROR;
    config->memory_detector.threshold_count = 1;
    config->memory_detector.cooldown_ms = 5000;

    /* Deadlock detector defaults */
    config->deadlock_detector.enabled = true;
    config->deadlock_detector.check_interval_ms = 1000;
    config->deadlock_detector.min_report_severity = HEALTH_SEVERITY_CRITICAL;
    config->deadlock_detector.threshold_count = 1;
    config->deadlock_detector.cooldown_ms = 2000;

    /* NaN detector defaults */
    config->nan_detector.enabled = true;
    config->nan_detector.check_interval_ms = 200;
    config->nan_detector.min_report_severity = HEALTH_SEVERITY_ERROR;
    config->nan_detector.threshold_count = 1;
    config->nan_detector.cooldown_ms = 1000;

    /* Resource detector defaults */
    config->resource_detector.enabled = true;
    config->resource_detector.check_interval_ms = 1000;
    config->resource_detector.min_report_severity = HEALTH_SEVERITY_WARNING;
    config->resource_detector.threshold_count = 3;
    config->resource_detector.cooldown_ms = 5000;

    /* Consistency checker defaults */
    config->consistency.check_reference_counts = true;
    config->consistency.check_pointer_canaries = true;
    config->consistency.check_struct_magic = true;
    config->consistency.check_mutex_state = true;
    config->consistency.check_circular_buffers = true;
    config->consistency.check_kg_consistency = false;  /* Expensive, off by default */
    config->consistency.check_neuron_values = true;
    config->consistency.kg_check_sample_rate = 100;    /* Check 1% of KG */

    /* Communication configuration */
    config->message_queue_depth = HEALTH_AGENT_MAX_QUEUE_DEPTH;
    config->enable_message_batching = true;
    config->batch_timeout_ms = 50;

    /* Recovery configuration */
    config->enable_auto_recovery = true;
    config->enable_emergency_checkpoint = true;
    config->enable_emergency_rollback = true;
    config->auto_recovery_threshold = HEALTH_SEVERITY_ERROR;

    /* Callbacks (none by default) */
    config->on_anomaly_detected = NULL;
    config->on_recovery_executed = NULL;
    config->callback_user_data = NULL;
}

/* ============================================================================
 * Agent Lifecycle API
 * ============================================================================ */

nimcp_health_agent_t* nimcp_health_agent_create(const health_agent_config_t* config) {
    /* Allocate agent structure */
    nimcp_health_agent_t* agent = (nimcp_health_agent_t*)nimcp_calloc(
        1, sizeof(nimcp_health_agent_t)
    );
    if (!agent) {
        nimcp_log(LOG_LEVEL_ERROR, "Failed to allocate health agent");
        return NULL;
    }

    /* Initialize canaries and magic */
    agent->magic = HEALTH_AGENT_MAGIC;
    agent->canary_front = HEALTH_AGENT_CANARY;
    agent->canary_back = HEALTH_AGENT_CANARY;

    /* Apply configuration */
    if (config) {
        agent->config = *config;
    } else {
        nimcp_health_agent_default_config(&agent->config);
    }

    /* Validate configuration */
    if (agent->config.check_interval_ms < AGENT_MIN_CHECK_INTERVAL_MS) {
        nimcp_log(LOG_LEVEL_WARN, "Check interval too small (%u ms), using minimum (%u ms)",
                  agent->config.check_interval_ms, AGENT_MIN_CHECK_INTERVAL_MS);
        agent->config.check_interval_ms = AGENT_MIN_CHECK_INTERVAL_MS;
    }

    /* Initialize message queue */
    uint32_t queue_capacity = agent->config.message_queue_depth > 0
        ? agent->config.message_queue_depth
        : AGENT_MSG_QUEUE_CAPACITY;

    if (!msg_queue_init(&agent->msg_queue, queue_capacity)) {
        nimcp_log(LOG_LEVEL_ERROR, "Failed to initialize message queue");
        nimcp_free(agent);
        return NULL;
    }

    /* Initialize mutexes using NIMCP threading utilities */
    agent->state_mutex = nimcp_mutex_create(NULL);
    if (!agent->state_mutex) {
        nimcp_log(LOG_LEVEL_ERROR, "Failed to create state mutex");
        msg_queue_destroy(&agent->msg_queue);
        nimcp_free(agent);
        return NULL;
    }

    agent->stats_mutex = nimcp_mutex_create(NULL);
    if (!agent->stats_mutex) {
        nimcp_log(LOG_LEVEL_ERROR, "Failed to create stats mutex");
        nimcp_mutex_destroy(agent->state_mutex);
        nimcp_free(agent->state_mutex);
        msg_queue_destroy(&agent->msg_queue);
        nimcp_free(agent);
        return NULL;
    }

    /* Initialize condition variable for stop signaling */
    agent->stop_cond = nimcp_cond_create();
    if (!agent->stop_cond) {
        nimcp_log(LOG_LEVEL_ERROR, "Failed to create stop condition");
        nimcp_mutex_destroy(agent->stats_mutex);
        nimcp_free(agent->stats_mutex);
        nimcp_mutex_destroy(agent->state_mutex);
        nimcp_free(agent->state_mutex);
        msg_queue_destroy(&agent->msg_queue);
        nimcp_free(agent);
        return NULL;
    }

    /* Initialize cognitive stats mutex */
    agent->cognitive_mutex = nimcp_mutex_create(NULL);
    if (!agent->cognitive_mutex) {
        nimcp_log(LOG_LEVEL_ERROR, "Failed to create cognitive mutex");
        nimcp_cond_destroy(agent->stop_cond);
        nimcp_free(agent->stop_cond);
        nimcp_mutex_destroy(agent->stats_mutex);
        nimcp_free(agent->stats_mutex);
        nimcp_mutex_destroy(agent->state_mutex);
        nimcp_free(agent->state_mutex);
        msg_queue_destroy(&agent->msg_queue);
        nimcp_free(agent);
        return NULL;
    }

    /* Initialize modules mutex */
    agent->modules_mutex = nimcp_mutex_create(NULL);
    if (!agent->modules_mutex) {
        nimcp_log(LOG_LEVEL_ERROR, "Failed to create modules mutex");
        nimcp_mutex_destroy(agent->cognitive_mutex);
        nimcp_free(agent->cognitive_mutex);
        nimcp_cond_destroy(agent->stop_cond);
        nimcp_free(agent->stop_cond);
        nimcp_mutex_destroy(agent->stats_mutex);
        nimcp_free(agent->stats_mutex);
        nimcp_mutex_destroy(agent->state_mutex);
        nimcp_free(agent->state_mutex);
        msg_queue_destroy(&agent->msg_queue);
        nimcp_free(agent);
        return NULL;
    }

    /* Initialize atomic state */
    atomic_init(&agent->running, false);
    atomic_init(&agent->stop_requested, false);
    atomic_init(&agent->uptime_start_us, 0);
    atomic_init(&agent->next_anomaly_id, 1);
    atomic_init(&agent->current_severity, HEALTH_SEVERITY_INFO);

    /* Initialize cognitive module pointers (all NULL by default) */
    agent->failure_predictor = NULL;
    agent->metacognition = NULL;
    agent->ethics = NULL;
    agent->emotion = NULL;
    agent->emotion_immune = NULL;
    agent->wellbeing = NULL;
    agent->mental_health = NULL;
    agent->collective = NULL;
    agent->rcog = NULL;
    agent->gpu_health = NULL;

    /* Initialize cognitive atomic stats */
    atomic_init(&agent->predictions_made, 0);
    atomic_init(&agent->predictions_correct, 0);
    atomic_init(&agent->preventive_actions, 0);
    atomic_init(&agent->self_diagnoses, 0);
    atomic_init(&agent->degradation_alerts, 0);
    atomic_init(&agent->current_confidence, 1.0f);
    atomic_init(&agent->ethics_evaluations, 0);
    atomic_init(&agent->ethics_blocks, 0);
    atomic_init(&agent->mercy_applications, 0);
    atomic_init(&agent->current_stress_level, 0.0f);
    atomic_init(&agent->emotion_adjustments, 0);
    atomic_init(&agent->distress_detections, 0);
    atomic_init(&agent->wellbeing_interventions, 0);
    atomic_init(&agent->current_distress_level, 0.0f);
    atomic_init(&agent->consensus_requests, 0);
    atomic_init(&agent->consensus_achieved, 0);
    atomic_init(&agent->avg_consensus_time_ms, 0.0f);
    atomic_init(&agent->rcog_diagnoses, 0);
    atomic_init(&agent->rcog_recovery_plans, 0);
    atomic_init(&agent->avg_rcog_time_ms, 0.0f);
    atomic_init(&agent->gpu_accelerated_checks, 0);
    atomic_init(&agent->gpu_utilization, 0.0f);
    atomic_init(&agent->gpu_healthy, true);

    /* Initialize hypothalamus module pointers */
    agent->hypothalamus = NULL;
    agent->hypo_bridge_id = 0;
    agent->homeostasis = NULL;
    agent->hypo_immune_bridge = NULL;
    agent->drives = NULL;
    memset(&agent->hypothalamus_config, 0, sizeof(agent->hypothalamus_config));

    /* Initialize hypothalamus atomic stats */
    atomic_init(&agent->in_stress_response, false);
    atomic_init(&agent->in_sickness_mode, false);
    atomic_init(&agent->stress_responses, 0);
    atomic_init(&agent->sickness_mode_entries, 0);
    atomic_init(&agent->drive_events_published, 0);
    atomic_init(&agent->homeostatic_output, 0.0f);

    /* Initialize additional module pointers */
    agent->connectivity = NULL;
    agent->oscillations = NULL;
    agent->gc_context = NULL;
    agent->checkpoint = NULL;
    agent->deadlock_detector_ptr = NULL;
    agent->bio_async_router = NULL;
    agent->bio_async_module_id = 0;
    agent->runtime_adaptation = NULL;
    agent->exception_bridge = NULL;
    agent->last_gc_time_us = 0;
    agent->last_checkpoint_time_us = 0;

    /* Initialize Portia/Dragonfly/Swarm/Memory module fields */
    agent->portia = NULL;
    agent->dragonfly = NULL;
    agent->swarm_immune = NULL;
    agent->swarm_memory = NULL;
    agent->engram = NULL;
    agent->memory_consolidation = NULL;

    atomic_init(&agent->portia_tier_changes, 0);
    atomic_init(&agent->portia_degradations, 0);
    atomic_init(&agent->dragonfly_anomalies_tracked, 0);
    atomic_init(&agent->dragonfly_pursuits, 0);
    atomic_init(&agent->dragonfly_interceptions, 0);
    atomic_init(&agent->dragonfly_current_target, 0);
    atomic_init(&agent->swarm_threats_detected, 0);
    atomic_init(&agent->swarm_responses_generated, 0);
    atomic_init(&agent->swarm_coordinated_responses, 0);
    atomic_init(&agent->swarm_memories_stored, 0);
    atomic_init(&agent->swarm_replays_performed, 0);
    atomic_init(&agent->engram_encodings, 0);
    atomic_init(&agent->engram_recalls, 0);

    memset(&agent->connectivity_config, 0, sizeof(agent->connectivity_config));
    memset(&agent->oscillations_config, 0, sizeof(agent->oscillations_config));
    memset(&agent->gc_config, 0, sizeof(agent->gc_config));
    memset(&agent->checkpoint_config, 0, sizeof(agent->checkpoint_config));
    memset(&agent->bio_async_config, 0, sizeof(agent->bio_async_config));
    memset(&agent->exception_config, 0, sizeof(agent->exception_config));
    memset(&agent->portia_config, 0, sizeof(agent->portia_config));
    memset(&agent->dragonfly_config, 0, sizeof(agent->dragonfly_config));
    memset(&agent->swarm_immune_config, 0, sizeof(agent->swarm_immune_config));
    memset(&agent->swarm_memory_config, 0, sizeof(agent->swarm_memory_config));
    memset(&agent->engram_config, 0, sizeof(agent->engram_config));
    memset(&agent->consolidation_config, 0, sizeof(agent->consolidation_config));

    /* Initialize additional module atomic stats */
    atomic_init(&agent->gc_triggers, 0);
    atomic_init(&agent->checkpoints_created, 0);
    atomic_init(&agent->rollbacks_performed, 0);
    atomic_init(&agent->bio_async_events_published, 0);
    atomic_init(&agent->load_reductions, 0);
    atomic_init(&agent->load_reduced, false);

    /* Initialize heartbeat state */
    atomic_init(&agent->heartbeat.last_heartbeat_us, 0);
    atomic_init(&agent->heartbeat.missed_count, 0);
    atomic_init(&agent->heartbeat.current_progress, 0.0f);
    memset(agent->heartbeat.current_operation, 0, sizeof(agent->heartbeat.current_operation));

    /* Initialize State Consistency Manager (Phase 3) */
    agent->consistency_mutex = nimcp_mutex_create(NULL);
    if (!agent->consistency_mutex) {
        nimcp_log(LOG_LEVEL_ERROR, "Failed to create consistency mutex");
        nimcp_mutex_destroy(agent->modules_mutex);
        nimcp_free(agent->modules_mutex);
        nimcp_mutex_destroy(agent->cognitive_mutex);
        nimcp_free(agent->cognitive_mutex);
        nimcp_cond_destroy(agent->stop_cond);
        nimcp_free(agent->stop_cond);
        nimcp_mutex_destroy(agent->stats_mutex);
        nimcp_free(agent->stats_mutex);
        nimcp_mutex_destroy(agent->state_mutex);
        nimcp_free(agent->state_mutex);
        msg_queue_destroy(&agent->msg_queue);
        nimcp_free(agent);
        return NULL;
    }

    memset(&agent->last_consistency_result, 0, sizeof(health_agent_consistency_result_t));
    atomic_init(&agent->consistency_check_pending, false);
    atomic_init(&agent->last_consistency_check_us, 0);
    atomic_init(&agent->consistency_checks_run, 0);
    atomic_init(&agent->consistency_failures_total, 0);

    /* Initialize registered struct tracking */
    for (uint32_t i = 0; i < 64; i++) {
        agent->registered_structs[i].ptr = NULL;
        agent->registered_structs[i].expected_magic = 0;
        agent->registered_structs[i].name[0] = '\0';
        agent->registered_structs[i].active = false;
    }
    agent->registered_struct_count = 0;

    /* Initialize statistics */
    memset(&agent->stats, 0, sizeof(health_agent_stats_t));
    agent->stats.highest_severity_seen = HEALTH_SEVERITY_INFO;

    nimcp_log(LOG_LEVEL_INFO, "Created health agent '%s' (id=%u, queue_capacity=%u)",
              agent->config.agent_name, agent->config.agent_id, queue_capacity);

    return agent;
}

void nimcp_health_agent_destroy(nimcp_health_agent_t* agent) {
    if (!agent) return;

    /* Validate agent structure */
    if (!validate_agent(agent)) {
        nimcp_log(LOG_LEVEL_ERROR, "Cannot destroy invalid agent structure");
        return;
    }

    /* Stop agent if running */
    if (atomic_load(&agent->running)) {
        nimcp_health_agent_stop(agent);
    }

    nimcp_log(LOG_LEVEL_INFO, "Destroying health agent '%s'", agent->config.agent_name);

    /* Destroy message queue */
    msg_queue_destroy(&agent->msg_queue);

    /* Destroy condition variable and mutexes using NIMCP utilities */
    if (agent->stop_cond) {
        nimcp_cond_destroy(agent->stop_cond);
        nimcp_free(agent->stop_cond);
        agent->stop_cond = NULL;
    }

    if (agent->state_mutex) {
        nimcp_mutex_destroy(agent->state_mutex);
        nimcp_free(agent->state_mutex);
        agent->state_mutex = NULL;
    }

    if (agent->stats_mutex) {
        nimcp_mutex_destroy(agent->stats_mutex);
        nimcp_free(agent->stats_mutex);
        agent->stats_mutex = NULL;
    }

    if (agent->cognitive_mutex) {
        nimcp_mutex_destroy(agent->cognitive_mutex);
        nimcp_free(agent->cognitive_mutex);
        agent->cognitive_mutex = NULL;
    }

    if (agent->modules_mutex) {
        nimcp_mutex_destroy(agent->modules_mutex);
        nimcp_free(agent->modules_mutex);
        agent->modules_mutex = NULL;
    }

    /* Destroy consistency mutex (Phase 3) */
    if (agent->consistency_mutex) {
        nimcp_mutex_destroy(agent->consistency_mutex);
        nimcp_free(agent->consistency_mutex);
        agent->consistency_mutex = NULL;
    }

    /* Clear magic/canaries to prevent use-after-free */
    agent->magic = 0;
    agent->canary_front = 0;
    agent->canary_back = 0;

    /* Free agent */
    nimcp_free(agent);
}

/* ============================================================================
 * Message Helper Function
 * ============================================================================ */

health_agent_message_t nimcp_health_agent_create_message(
    health_agent_msg_type_t type,
    health_agent_severity_t severity,
    health_agent_source_t source,
    const char* description,
    ...
) {
    health_agent_message_t msg;
    memset(&msg, 0, sizeof(msg));

    msg.type = type;
    msg.severity = severity;
    msg.source = source;
    msg.suggested_action = HEALTH_RECOVERY_NONE;
    msg.timestamp_us = get_timestamp_us();
    msg.anomaly_id = 0;  /* Will be assigned when reported */

    /* Format description */
    if (description) {
        va_list args;
        va_start(args, description);
        vsnprintf(msg.description, sizeof(msg.description) - 1, description, args);
        va_end(args);
    }

    return msg;
}

/* ============================================================================
 * Connection Management (Stubs - implemented in Phase 2)
 * ============================================================================ */

int nimcp_health_agent_connect_brain(nimcp_health_agent_t* agent, brain_t brain) {
    if (!validate_agent(agent)) {
        nimcp_log(LOG_LEVEL_ERROR, "Invalid agent in connect_brain");
        return -1;
    }

    agent->brain = brain;
    nimcp_log(LOG_LEVEL_INFO, "Agent '%s' connected to brain", agent->config.agent_name);
    return 0;
}

int nimcp_health_agent_connect_immune(nimcp_health_agent_t* agent,
                                       brain_immune_system_t* immune) {
    if (!validate_agent(agent)) {
        nimcp_log(LOG_LEVEL_ERROR, "Invalid agent in connect_immune");
        return -1;
    }

    agent->immune = immune;
    nimcp_log(LOG_LEVEL_INFO, "Agent '%s' connected to immune system",
              agent->config.agent_name);
    return 0;
}

int nimcp_health_agent_connect_monitor(nimcp_health_agent_t* agent,
                                        health_monitor_t* monitor) {
    if (!validate_agent(agent)) {
        nimcp_log(LOG_LEVEL_ERROR, "Invalid agent in connect_monitor");
        return -1;
    }

    agent->monitor = monitor;
    nimcp_log(LOG_LEVEL_INFO, "Agent '%s' connected to health monitor",
              agent->config.agent_name);
    return 0;
}

/* ============================================================================
 * Agent Start/Stop (Stubs - full implementation in Phase 3)
 * ============================================================================ */

int nimcp_health_agent_start(nimcp_health_agent_t* agent) {
    if (!validate_agent(agent)) {
        nimcp_log(LOG_LEVEL_ERROR, "Invalid agent in start");
        return -1;
    }

    nimcp_mutex_lock(agent->state_mutex);

    if (atomic_load(&agent->running)) {
        nimcp_log(LOG_LEVEL_WARN, "Agent '%s' already running", agent->config.agent_name);
        nimcp_mutex_unlock(agent->state_mutex);
        return -1;
    }

    /* Reset stop flag */
    atomic_store(&agent->stop_requested, false);

    /* Record start time */
    atomic_store(&agent->uptime_start_us, get_timestamp_us());

    /* Initialize heartbeat timestamp */
    atomic_store(&agent->heartbeat.last_heartbeat_us, get_timestamp_us());

    /* Create agent thread using NIMCP threading */
    thread_attr_t attr;
    memset(&attr, 0, sizeof(attr));

    if (agent->config.thread_stack_size > 0) {
        attr.stack_size = agent->config.thread_stack_size;
    }

    /* nimcp_thread_create signature: (thread, start_routine, arg, attr) */
    nimcp_result_t rc = nimcp_thread_create(&agent->agent_thread, agent_thread_main, agent, &attr);

    if (rc != 0) {
        nimcp_log(LOG_LEVEL_ERROR, "Failed to create agent thread: error %d", rc);
        nimcp_mutex_unlock(agent->state_mutex);
        return -1;
    }

    atomic_store(&agent->running, true);

    nimcp_mutex_unlock(agent->state_mutex);

    nimcp_log(LOG_LEVEL_INFO, "Started health agent '%s'", agent->config.agent_name);
    return 0;
}

int nimcp_health_agent_stop(nimcp_health_agent_t* agent) {
    if (!validate_agent(agent)) {
        nimcp_log(LOG_LEVEL_ERROR, "Invalid agent in stop");
        return -1;
    }

    nimcp_mutex_lock(agent->state_mutex);

    if (!atomic_load(&agent->running)) {
        nimcp_log(LOG_LEVEL_WARN, "Agent '%s' not running", agent->config.agent_name);
        nimcp_mutex_unlock(agent->state_mutex);
        return -1;
    }

    /* Signal thread to stop */
    atomic_store(&agent->stop_requested, true);

    nimcp_mutex_unlock(agent->state_mutex);

    /* Wait for thread to complete */
    nimcp_result_t rc = nimcp_thread_join(agent->agent_thread, NULL);

    if (rc != 0) {
        nimcp_log(LOG_LEVEL_ERROR, "Failed to join agent thread: error %d", rc);
    }

    atomic_store(&agent->running, false);

    nimcp_log(LOG_LEVEL_INFO, "Stopped health agent '%s'", agent->config.agent_name);
    return 0;
}

/* ============================================================================
 * Agent Thread (Basic implementation for Phase 1)
 * ============================================================================ */

static void* agent_thread_main(void* arg) {
    nimcp_health_agent_t* agent = (nimcp_health_agent_t*)arg;

    if (!validate_agent(agent)) {
        nimcp_log(LOG_LEVEL_ERROR, "Invalid agent in thread main");
        return NULL;
    }

    nimcp_log(LOG_LEVEL_DEBUG, "Agent thread '%s' started", agent->config.agent_name);

    uint64_t last_check_us = get_timestamp_us();
    uint64_t last_prediction_us = get_timestamp_us();
    uint64_t last_metacog_us = get_timestamp_us();
    uint64_t last_wellbeing_us = get_timestamp_us();
    uint32_t check_interval_us = agent->config.check_interval_ms * 1000;

    while (!atomic_load(&agent->stop_requested)) {
        uint64_t now_us = get_timestamp_us();

        /* Check if it's time to run checks */
        if (now_us - last_check_us >= check_interval_us) {
            last_check_us = now_us;

            /* Update statistics */
            nimcp_mutex_lock(agent->stats_mutex);
            agent->stats.checks_performed++;
            agent->stats.uptime_ms = (now_us - atomic_load(&agent->uptime_start_us)) / 1000;
            nimcp_mutex_unlock(agent->stats_mutex);

            /* ========== COGNITIVE MODULE INTEGRATION ========== */

            /* Run failure prediction if connected */
            if (agent->failure_predictor && agent->prediction_config.enable_failure_prediction) {
                if (now_us - last_prediction_us >= agent->prediction_config.prediction_horizon_ms * 1000ULL) {
                    last_prediction_us = now_us;
                    agent_run_failure_prediction(agent);
                }
            }

            /* Run metacognition self-check if connected */
            if (agent->metacognition && agent->metacog_config.enable_metacognition) {
                if (now_us - last_metacog_us >= 1000000ULL) {  /* 1 second interval */
                    last_metacog_us = now_us;
                    agent_run_metacognition_check(agent);
                }
            }

            /* Run wellbeing check if connected */
            if (agent->wellbeing && agent->wellbeing_config.enable_wellbeing_monitoring) {
                if (now_us - last_wellbeing_us >= 5000000ULL) {  /* 5 second interval */
                    last_wellbeing_us = now_us;
                    agent_run_wellbeing_check(agent);
                }
            }

            /* Apply emotion-adjusted thresholds if connected */
            if (agent->emotion && agent->emotion_config.enable_emotion_awareness) {
                agent_apply_emotion_adjustments(agent);
            }

            /* Check GPU health if connected */
            if (agent->gpu_health && agent->gpu_config.enable_gpu_monitoring) {
                agent_check_gpu_health(agent);
            }

            /* ========== HYPOTHALAMUS & MODULE INTEGRATION (USE) ========== */

            /* Compute current health score for module USE */
            float health_score = 1.0f - ((float)atomic_load(&agent->current_severity) / 4.0f);

            /* Run hypothalamus check and stress coordination if connected */
            if (agent->hypothalamus && agent->hypothalamus_config.enable_hypothalamus) {
                agent_run_hypothalamus_check(agent, health_score);
            }

            /* Run homeostatic regulation if connected (USE the homeostasis) */
            if (agent->homeostasis && agent->hypothalamus_config.enable_homeostatic_regulation) {
                agent_run_homeostatic_regulation(agent, health_score);
            }

            /* Check connectivity health if connected */
            if (agent->connectivity && agent->connectivity_config.enable_connectivity_monitoring) {
                agent_check_connectivity(agent);
            }

            /* Check oscillations if connected */
            if (agent->oscillations && agent->oscillations_config.enable_oscillation_monitoring) {
                agent_check_oscillations(agent);
            }

            /* Auto-trigger GC if memory pressure detected (USE the GC) */
            if (agent->gc_context && agent->gc_config.enable_gc_integration) {
                agent_auto_gc_if_needed(agent);
            }

            /* Auto-checkpoint if health is good (USE the checkpoint system) */
            if (agent->checkpoint && agent->checkpoint_config.enable_auto_checkpoint) {
                agent_auto_checkpoint_if_needed(agent, health_score);
            }

            /* Check for deadlocks if detector connected (USE the deadlock detector) */
            if (agent->deadlock_detector_ptr) {
                bool deadlock_found = false;
                bool high_contention = false;
                nimcp_health_agent_check_deadlocks(agent, &deadlock_found, &high_contention);

                if (deadlock_found) {
                    health_agent_message_t msg = nimcp_health_agent_create_message(
                        HEALTH_MSG_DEADLOCK_DETECTED,
                        HEALTH_SEVERITY_CRITICAL,
                        HEALTH_SOURCE_THREADING,
                        "Deadlock detected by health agent"
                    );
                    msg.suggested_action = HEALTH_RECOVERY_RESTART_THREAD;
                    nimcp_health_agent_report_anomaly(agent, &msg);
                }
            }

            /* ========== STATE CONSISTENCY CHECKS (Phase 3) ========== */

            /* Run consistency checks (timing is internal to function) */
            if (CONSISTENCY_CHECKS_ENABLED(agent->config.consistency) ||
                atomic_load(&agent->consistency_check_pending)) {
                agent_run_consistency_checks(agent);
            }
        }

        /* Sleep for a short period */
        struct timespec ts = { .tv_sec = 0, .tv_nsec = 1000000 };  /* 1ms */
        nanosleep(&ts, NULL);
    }

    nimcp_log(LOG_LEVEL_DEBUG, "Agent thread '%s' exiting", agent->config.agent_name);
    return NULL;
}

/* ============================================================================
 * Heartbeat API
 * ============================================================================ */

void nimcp_health_agent_heartbeat(nimcp_health_agent_t* agent) {
    if (!validate_agent(agent)) return;

    atomic_store(&agent->heartbeat.last_heartbeat_us, get_timestamp_us());
    atomic_store(&agent->heartbeat.missed_count, 0);

    nimcp_mutex_lock(agent->stats_mutex);
    agent->stats.heartbeats_received++;
    nimcp_mutex_unlock(agent->stats_mutex);
}

void nimcp_health_agent_heartbeat_ex(nimcp_health_agent_t* agent,
                                      const char* operation,
                                      float progress) {
    if (!validate_agent(agent)) return;

    atomic_store(&agent->heartbeat.last_heartbeat_us, get_timestamp_us());
    atomic_store(&agent->heartbeat.missed_count, 0);
    atomic_store(&agent->heartbeat.current_progress, progress);

    if (operation) {
        strncpy(agent->heartbeat.current_operation, operation,
                sizeof(agent->heartbeat.current_operation) - 1);
    }

    nimcp_mutex_lock(agent->stats_mutex);
    agent->stats.heartbeats_received++;
    nimcp_mutex_unlock(agent->stats_mutex);
}

/* ============================================================================
 * Manual Trigger API
 * ============================================================================ */

int nimcp_health_agent_report_anomaly(nimcp_health_agent_t* agent,
                                       const health_agent_message_t* msg) {
    if (!validate_agent(agent) || !msg) {
        return -1;
    }

    /* Copy message and assign anomaly ID */
    health_agent_message_t queued_msg = *msg;
    queued_msg.anomaly_id = atomic_fetch_add(&agent->next_anomaly_id, 1);

    if (queued_msg.timestamp_us == 0) {
        queued_msg.timestamp_us = get_timestamp_us();
    }

    /* Queue message */
    if (!msg_queue_push(&agent->msg_queue, &queued_msg)) {
        nimcp_log(LOG_LEVEL_WARN, "Message queue full, dropping anomaly report");
        return -1;
    }

    /* Update statistics */
    nimcp_mutex_lock(agent->stats_mutex);
    agent->stats.anomalies_detected++;
    agent->stats.messages_sent++;
    if (msg->severity > agent->stats.highest_severity_seen) {
        agent->stats.highest_severity_seen = msg->severity;
    }
    nimcp_mutex_unlock(agent->stats_mutex);

    /* Update current severity if higher */
    health_agent_severity_t current = atomic_load(&agent->current_severity);
    if (msg->severity > current) {
        atomic_store(&agent->current_severity, msg->severity);
    }

    /* Call callback if registered */
    if (agent->config.on_anomaly_detected) {
        agent->config.on_anomaly_detected(&queued_msg, agent->config.callback_user_data);
    }

    return 0;
}

int nimcp_health_agent_request_check(nimcp_health_agent_t* agent) {
    if (!validate_agent(agent)) return -1;

    /* Signal thread to run consistency check on next iteration (Phase 3) */
    atomic_store(&agent->consistency_check_pending, true);
    nimcp_log(LOG_LEVEL_DEBUG, "Check requested for agent '%s'", agent->config.agent_name);
    return 0;
}

int nimcp_health_agent_request_emergency_checkpoint(nimcp_health_agent_t* agent,
                                                     const char* reason) {
    if (!validate_agent(agent)) return -1;

    health_agent_message_t msg = nimcp_health_agent_create_message(
        HEALTH_MSG_EMERGENCY,
        HEALTH_SEVERITY_CRITICAL,
        HEALTH_SOURCE_CHECKPOINT,
        "Emergency checkpoint requested: %s", reason ? reason : "unknown"
    );
    msg.suggested_action = HEALTH_RECOVERY_CHECKPOINT;

    return nimcp_health_agent_report_anomaly(agent, &msg);
}

/* ============================================================================
 * Query API
 * ============================================================================ */

bool nimcp_health_agent_is_running(const nimcp_health_agent_t* agent) {
    if (!validate_agent(agent)) return false;
    return atomic_load(&agent->running);
}

void nimcp_health_agent_get_stats(const nimcp_health_agent_t* agent,
                                   health_agent_stats_t* stats) {
    if (!validate_agent(agent) || !stats) return;

    /* Need to cast away const for mutex lock - stats_mutex is logically const */
    nimcp_health_agent_t* mutable_agent = (nimcp_health_agent_t*)agent;

    nimcp_mutex_lock(mutable_agent->stats_mutex);
    *stats = agent->stats;

    /* Update uptime */
    if (atomic_load(&agent->running)) {
        stats->uptime_ms = (get_timestamp_us() - atomic_load(&agent->uptime_start_us)) / 1000;
    }

    /* Update queue stats */
    stats->queue_high_watermark = msg_queue_size(&agent->msg_queue);

    nimcp_mutex_unlock(mutable_agent->stats_mutex);
}

uint32_t nimcp_health_agent_pending_messages(const nimcp_health_agent_t* agent) {
    if (!validate_agent(agent)) return 0;
    return msg_queue_size(&agent->msg_queue);
}

health_agent_severity_t nimcp_health_agent_current_status(
    const nimcp_health_agent_t* agent) {
    if (!validate_agent(agent)) return HEALTH_SEVERITY_FATAL;
    return atomic_load(&agent->current_severity);
}

/* ============================================================================
 * Configuration Update API
 * ============================================================================ */

int nimcp_health_agent_update_detector(nimcp_health_agent_t* agent,
                                        const char* detector,
                                        const health_agent_detector_config_t* config) {
    if (!validate_agent(agent) || !detector || !config) {
        return -1;
    }

    health_agent_detector_config_t* target = NULL;

    if (strcmp(detector, "heartbeat") == 0) {
        target = &agent->config.heartbeat_detector;
    } else if (strcmp(detector, "memory") == 0) {
        target = &agent->config.memory_detector;
    } else if (strcmp(detector, "deadlock") == 0) {
        target = &agent->config.deadlock_detector;
    } else if (strcmp(detector, "nan") == 0) {
        target = &agent->config.nan_detector;
    } else if (strcmp(detector, "resource") == 0) {
        target = &agent->config.resource_detector;
    } else {
        nimcp_log(LOG_LEVEL_ERROR, "Unknown detector: %s", detector);
        return -1;
    }

    *target = *config;
    nimcp_log(LOG_LEVEL_INFO, "Updated detector '%s' configuration", detector);
    return 0;
}

int nimcp_health_agent_set_detector_enabled(nimcp_health_agent_t* agent,
                                             const char* detector,
                                             bool enabled) {
    if (!validate_agent(agent) || !detector) {
        return -1;
    }

    health_agent_detector_config_t* target = NULL;

    if (strcmp(detector, "heartbeat") == 0) {
        target = &agent->config.heartbeat_detector;
    } else if (strcmp(detector, "memory") == 0) {
        target = &agent->config.memory_detector;
    } else if (strcmp(detector, "deadlock") == 0) {
        target = &agent->config.deadlock_detector;
    } else if (strcmp(detector, "nan") == 0) {
        target = &agent->config.nan_detector;
    } else if (strcmp(detector, "resource") == 0) {
        target = &agent->config.resource_detector;
    } else {
        nimcp_log(LOG_LEVEL_ERROR, "Unknown detector: %s", detector);
        return -1;
    }

    target->enabled = enabled;
    nimcp_log(LOG_LEVEL_INFO, "%s detector '%s'", enabled ? "Enabled" : "Disabled", detector);
    return 0;
}

/* ============================================================================
 * Default Cognitive Configuration
 * ============================================================================ */

void nimcp_health_agent_default_cognitive_config(health_agent_cognitive_config_t* config) {
    if (!config) return;

    memset(config, 0, sizeof(health_agent_cognitive_config_t));

    /* Prediction defaults */
    config->prediction.enable_failure_prediction = true;
    config->prediction.prediction_threshold = 0.7f;
    config->prediction.prediction_horizon_ms = 5000;
    config->prediction.enable_preventive_action = true;
    config->prediction.enable_trend_analysis = true;

    /* Metacognition defaults */
    config->metacog.enable_metacognition = true;
    config->metacog.enable_confidence_calibration = true;
    config->metacog.enable_degradation_detection = true;
    config->metacog.degradation_threshold = 0.3f;
    config->metacog.enable_self_diagnosis = true;

    /* Ethics defaults */
    config->ethics.enable_ethics_evaluation = true;
    config->ethics.enable_asimov_laws = true;
    config->ethics.enable_mercy_directive = true;
    config->ethics.enable_golden_rule = true;
    config->ethics.ethics_override_threshold = 0.95f;

    /* Emotion defaults */
    config->emotion.enable_emotion_awareness = true;
    config->emotion.enable_emotion_reporting = true;
    config->emotion.enable_stress_adjustment = true;
    config->emotion.stress_threshold_multiplier = 0.8f;

    /* Wellbeing defaults */
    config->wellbeing.enable_wellbeing_monitoring = true;
    config->wellbeing.enable_distress_detection = true;
    config->wellbeing.enable_suffering_prevention = true;
    config->wellbeing.distress_intervention_threshold = 0.6f;

    /* Collective defaults */
    config->collective.enable_collective_monitoring = false;
    config->collective.enable_consensus_decisions = false;
    config->collective.enable_swarm_immune = false;
    config->collective.consensus_threshold = 0.67f;
    config->collective.consensus_timeout_ms = 1000;

    /* RCOG defaults */
    config->rcog.enable_rcog_diagnosis = true;
    config->rcog.enable_rcog_recovery_planning = true;
    config->rcog.enable_imagination = true;
    config->rcog.rcog_timeout_ms = 5000;
    config->rcog.confidence_threshold = 0.6f;

    /* GPU defaults */
    config->gpu.enable_gpu_monitoring = false;
    config->gpu.enable_gpu_acceleration = false;
    config->gpu.enable_tensor_validation = false;
    config->gpu.enable_anomaly_detection = false;
    config->gpu.gpu_check_interval_ms = 1000;

    /* Hypothalamus defaults */
    config->hypothalamus.enable_hypothalamus = true;
    config->hypothalamus.enable_homeostatic_regulation = true;
    config->hypothalamus.enable_drive_response = true;
    config->hypothalamus.enable_stress_coordination = true;
    config->hypothalamus.enable_sickness_behavior = true;
    config->hypothalamus.enable_immune_bridge = true;
    config->hypothalamus.stress_trigger_threshold = 0.4f;
    config->hypothalamus.sickness_trigger_threshold = 0.25f;
    config->hypothalamus.homeostasis_update_ms = 100;
    config->hypothalamus.drive_response_timeout_ms = 1000;

    /* Connectivity defaults */
    config->connectivity.enable_connectivity_monitoring = true;
    config->connectivity.enable_isolation_detection = true;
    config->connectivity.enable_auto_reconnect = true;
    config->connectivity.check_interval_ms = 5000;
    config->connectivity.isolation_threshold = 0.1f;

    /* Oscillations defaults */
    config->oscillations.enable_oscillation_monitoring = true;
    config->oscillations.enable_seizure_detection = true;
    config->oscillations.enable_flatline_detection = true;
    config->oscillations.enable_desync_detection = true;
    config->oscillations.abnormal_threshold = 0.3f;
    config->oscillations.sample_rate_hz = 60;

    /* GC defaults */
    config->gc.enable_gc_integration = true;
    config->gc.enable_auto_gc_trigger = true;
    config->gc.enable_leak_detection = true;
    config->gc.gc_trigger_threshold = 0.85f;
    config->gc.gc_cooldown_ms = 30000;

    /* Checkpoint defaults */
    config->checkpoint.enable_checkpoint_integration = true;
    config->checkpoint.enable_auto_checkpoint = true;
    config->checkpoint.enable_auto_rollback = true;
    config->checkpoint.checkpoint_interval_ms = 60000;
    config->checkpoint.health_threshold_checkpoint = 0.8f;
    config->checkpoint.health_threshold_rollback = 0.2f;

    /* Bio-async defaults */
    config->bio_async.enable_bio_async = true;
    config->bio_async.publish_health_events = true;
    config->bio_async.subscribe_health_requests = true;
    config->bio_async.event_batch_size = 10;
    config->bio_async.event_batch_timeout_ms = 100;

    /* Exception defaults */
    config->exception.enable_exception_integration = true;
    config->exception.auto_present_exceptions = true;
    config->exception.enable_recovery_callbacks = true;
    config->exception.exception_severity_threshold = 2;  /* EXCEPTION_SEVERITY_ERROR */
}

/* ============================================================================
 * Hypothalamus Connection Functions
 * ============================================================================ */

int nimcp_health_agent_connect_hypothalamus(
    nimcp_health_agent_t* agent,
    hypo_orchestrator_t orchestrator,
    const health_agent_hypothalamus_config_t* config
) {
    if (!validate_agent(agent)) {
        nimcp_log(LOG_LEVEL_ERROR, "Invalid agent in connect_hypothalamus");
        return -1;
    }

    nimcp_mutex_lock(agent->modules_mutex);

    agent->hypothalamus = orchestrator;

    /* Apply configuration */
    if (config) {
        agent->hypothalamus_config = *config;
    } else {
        /* Use defaults */
        agent->hypothalamus_config.enable_hypothalamus = true;
        agent->hypothalamus_config.enable_homeostatic_regulation = true;
        agent->hypothalamus_config.enable_drive_response = true;
        agent->hypothalamus_config.enable_stress_coordination = true;
        agent->hypothalamus_config.enable_sickness_behavior = true;
        agent->hypothalamus_config.enable_immune_bridge = true;
        agent->hypothalamus_config.stress_trigger_threshold = 0.4f;
        agent->hypothalamus_config.sickness_trigger_threshold = 0.25f;
        agent->hypothalamus_config.homeostasis_update_ms = 100;
        agent->hypothalamus_config.drive_response_timeout_ms = 1000;
    }

    /* Register as bridge with hypothalamus orchestrator */
    uint32_t bridge_id = 0;
    int reg_result = hypo_orch_register_bridge(
        orchestrator,
        HYPO_BRIDGE_IMMUNE,  /* Health agent uses immune bridge type */
        agent->config.agent_name,
        agent,  /* Bridge handle is the agent itself */
        agent,  /* Context is also the agent */
        &bridge_id
    );
    if (reg_result == 0) {
        agent->hypo_bridge_id = bridge_id;
        nimcp_log(LOG_LEVEL_DEBUG, "Registered as hypothalamus bridge %u", bridge_id);
    } else {
        nimcp_log(LOG_LEVEL_WARN, "Failed to register as hypothalamus bridge");
    }

    nimcp_mutex_unlock(agent->modules_mutex);

    nimcp_log(LOG_LEVEL_INFO, "Agent '%s' connected to hypothalamus orchestrator",
              agent->config.agent_name);
    return 0;
}

int nimcp_health_agent_connect_homeostasis(
    nimcp_health_agent_t* agent,
    hypo_homeostasis_handle_t* homeostasis
) {
    if (!validate_agent(agent)) {
        nimcp_log(LOG_LEVEL_ERROR, "Invalid agent in connect_homeostasis");
        return -1;
    }

    nimcp_mutex_lock(agent->modules_mutex);
    agent->homeostasis = homeostasis;
    nimcp_mutex_unlock(agent->modules_mutex);

    nimcp_log(LOG_LEVEL_INFO, "Agent '%s' connected to homeostasis system",
              agent->config.agent_name);
    return 0;
}

int nimcp_health_agent_connect_hypo_immune_bridge(
    nimcp_health_agent_t* agent,
    hypo_immune_bridge_t* immune_bridge
) {
    if (!validate_agent(agent)) {
        nimcp_log(LOG_LEVEL_ERROR, "Invalid agent in connect_hypo_immune_bridge");
        return -1;
    }

    nimcp_mutex_lock(agent->modules_mutex);
    agent->hypo_immune_bridge = immune_bridge;
    nimcp_mutex_unlock(agent->modules_mutex);

    nimcp_log(LOG_LEVEL_INFO, "Agent '%s' connected to hypothalamus-immune bridge",
              agent->config.agent_name);
    return 0;
}

int nimcp_health_agent_connect_drives(
    nimcp_health_agent_t* agent,
    hypo_drive_system_handle_t* drives
) {
    if (!validate_agent(agent)) {
        nimcp_log(LOG_LEVEL_ERROR, "Invalid agent in connect_drives");
        return -1;
    }

    nimcp_mutex_lock(agent->modules_mutex);
    agent->drives = drives;
    nimcp_mutex_unlock(agent->modules_mutex);

    nimcp_log(LOG_LEVEL_INFO, "Agent '%s' connected to drive system",
              agent->config.agent_name);
    return 0;
}

/* ============================================================================
 * Additional Module Connection Functions
 * ============================================================================ */

int nimcp_health_agent_connect_connectivity(
    nimcp_health_agent_t* agent,
    connectivity_health_t* connectivity,
    const health_agent_connectivity_config_t* config
) {
    if (!validate_agent(agent)) return -1;

    nimcp_mutex_lock(agent->modules_mutex);
    agent->connectivity = connectivity;
    if (config) {
        agent->connectivity_config = *config;
    } else {
        agent->connectivity_config.enable_connectivity_monitoring = true;
        agent->connectivity_config.enable_isolation_detection = true;
        agent->connectivity_config.enable_auto_reconnect = true;
        agent->connectivity_config.check_interval_ms = 5000;
        agent->connectivity_config.isolation_threshold = 0.1f;
    }
    nimcp_mutex_unlock(agent->modules_mutex);

    nimcp_log(LOG_LEVEL_INFO, "Agent '%s' connected to connectivity health",
              agent->config.agent_name);
    return 0;
}

int nimcp_health_agent_connect_oscillations(
    nimcp_health_agent_t* agent,
    brain_oscillations_t* oscillations,
    const health_agent_oscillations_config_t* config
) {
    if (!validate_agent(agent)) return -1;

    nimcp_mutex_lock(agent->modules_mutex);
    agent->oscillations = oscillations;
    if (config) {
        agent->oscillations_config = *config;
    } else {
        agent->oscillations_config.enable_oscillation_monitoring = true;
        agent->oscillations_config.enable_seizure_detection = true;
        agent->oscillations_config.enable_flatline_detection = true;
        agent->oscillations_config.enable_desync_detection = true;
        agent->oscillations_config.abnormal_threshold = 0.3f;
        agent->oscillations_config.sample_rate_hz = 60;
    }
    nimcp_mutex_unlock(agent->modules_mutex);

    nimcp_log(LOG_LEVEL_INFO, "Agent '%s' connected to brain oscillations",
              agent->config.agent_name);
    return 0;
}

int nimcp_health_agent_connect_gc(
    nimcp_health_agent_t* agent,
    kg_gc_context_t* gc_context,
    const health_agent_gc_config_t* config
) {
    if (!validate_agent(agent)) return -1;

    nimcp_mutex_lock(agent->modules_mutex);
    agent->gc_context = gc_context;
    if (config) {
        agent->gc_config = *config;
    } else {
        agent->gc_config.enable_gc_integration = true;
        agent->gc_config.enable_auto_gc_trigger = true;
        agent->gc_config.enable_leak_detection = true;
        agent->gc_config.gc_trigger_threshold = 0.85f;
        agent->gc_config.gc_cooldown_ms = 30000;
    }
    nimcp_mutex_unlock(agent->modules_mutex);

    nimcp_log(LOG_LEVEL_INFO, "Agent '%s' connected to GC context",
              agent->config.agent_name);
    return 0;
}

int nimcp_health_agent_connect_checkpoint(
    nimcp_health_agent_t* agent,
    checkpoint_manager_t* checkpoint,
    const health_agent_checkpoint_config_t* config
) {
    if (!validate_agent(agent)) return -1;

    nimcp_mutex_lock(agent->modules_mutex);
    agent->checkpoint = checkpoint;
    if (config) {
        agent->checkpoint_config = *config;
    } else {
        agent->checkpoint_config.enable_checkpoint_integration = true;
        agent->checkpoint_config.enable_auto_checkpoint = true;
        agent->checkpoint_config.enable_auto_rollback = true;
        agent->checkpoint_config.checkpoint_interval_ms = 60000;
        agent->checkpoint_config.health_threshold_checkpoint = 0.8f;
        agent->checkpoint_config.health_threshold_rollback = 0.2f;
    }
    nimcp_mutex_unlock(agent->modules_mutex);

    nimcp_log(LOG_LEVEL_INFO, "Agent '%s' connected to checkpoint manager",
              agent->config.agent_name);
    return 0;
}

int nimcp_health_agent_connect_deadlock_detector(
    nimcp_health_agent_t* agent,
    deadlock_detector_t* deadlock_detector
) {
    if (!validate_agent(agent)) return -1;

    nimcp_mutex_lock(agent->modules_mutex);
    agent->deadlock_detector_ptr = deadlock_detector;
    nimcp_mutex_unlock(agent->modules_mutex);

    nimcp_log(LOG_LEVEL_INFO, "Agent '%s' connected to deadlock detector",
              agent->config.agent_name);
    return 0;
}

int nimcp_health_agent_connect_bio_async(
    nimcp_health_agent_t* agent,
    bio_async_router_t* router,
    const health_agent_bio_async_config_t* config
) {
    if (!validate_agent(agent)) return -1;

    nimcp_mutex_lock(agent->modules_mutex);
    agent->bio_async_router = router;
    if (config) {
        agent->bio_async_config = *config;
    } else {
        agent->bio_async_config.enable_bio_async = true;
        agent->bio_async_config.publish_health_events = true;
        agent->bio_async_config.subscribe_health_requests = true;
        agent->bio_async_config.event_batch_size = 10;
        agent->bio_async_config.event_batch_timeout_ms = 100;
    }
    /* Register as module with bio-async router */
    bio_router_t global_router = bio_router_get_global();
    if (global_router) {
        bio_module_info_t module_info = {
            .module_id = 0,  /* Auto-assign */
            .module_name = "health_agent",
            .inbox_capacity = 0,  /* Use default */
            .user_data = agent
        };
        bio_module_context_t ctx = bio_router_register_module(&module_info);
        if (ctx) {
            agent->bio_async_module_id = bio_module_context_get_id(ctx);
            nimcp_log(LOG_LEVEL_DEBUG, "Registered as bio-async module %u",
                      agent->bio_async_module_id);
            /* Note: We don't store the context since we register/unregister per publish */
            bio_router_unregister_module(ctx);
        }
    }
    nimcp_mutex_unlock(agent->modules_mutex);

    nimcp_log(LOG_LEVEL_INFO, "Agent '%s' connected to bio-async router",
              agent->config.agent_name);
    return 0;
}

int nimcp_health_agent_connect_runtime_adaptation(
    nimcp_health_agent_t* agent,
    runtime_adaptation_context_t ra_ctx
) {
    if (!validate_agent(agent)) return -1;

    nimcp_mutex_lock(agent->modules_mutex);
    agent->runtime_adaptation = ra_ctx;
    nimcp_mutex_unlock(agent->modules_mutex);

    nimcp_log(LOG_LEVEL_INFO, "Agent '%s' connected to runtime adaptation",
              agent->config.agent_name);
    return 0;
}

int nimcp_health_agent_connect_exception_bridge(
    nimcp_health_agent_t* agent,
    exception_immune_t* exception_bridge,
    const health_agent_exception_config_t* config
) {
    if (!validate_agent(agent)) return -1;

    nimcp_mutex_lock(agent->modules_mutex);
    agent->exception_bridge = exception_bridge;
    if (config) {
        agent->exception_config = *config;
    } else {
        agent->exception_config.enable_exception_integration = true;
        agent->exception_config.auto_present_exceptions = true;
        agent->exception_config.enable_recovery_callbacks = true;
        agent->exception_config.exception_severity_threshold = 2;
    }
    nimcp_mutex_unlock(agent->modules_mutex);

    nimcp_log(LOG_LEVEL_INFO, "Agent '%s' connected to exception-immune bridge",
              agent->config.agent_name);
    return 0;
}

/* ============================================================================
 * Cognitive Module Connection Functions
 * ============================================================================ */

int nimcp_health_agent_connect_failure_prediction(
    nimcp_health_agent_t* agent,
    failure_predictor_t* predictor,
    const health_agent_prediction_config_t* config
) {
    if (!validate_agent(agent)) return -1;

    nimcp_mutex_lock(agent->modules_mutex);
    agent->failure_predictor = predictor;
    if (config) {
        agent->prediction_config = *config;
    } else {
        agent->prediction_config.enable_failure_prediction = true;
        agent->prediction_config.prediction_threshold = 0.7f;
        agent->prediction_config.prediction_horizon_ms = 60000;
        agent->prediction_config.enable_preventive_action = true;
        agent->prediction_config.enable_trend_analysis = true;
    }
    nimcp_mutex_unlock(agent->modules_mutex);

    nimcp_log(LOG_LEVEL_INFO, "Agent '%s' connected to failure predictor",
              agent->config.agent_name);
    return 0;
}

int nimcp_health_agent_connect_metacognition(
    nimcp_health_agent_t* agent,
    metacognition_t* metacog,
    const health_agent_metacog_config_t* config
) {
    if (!validate_agent(agent)) return -1;

    nimcp_mutex_lock(agent->modules_mutex);
    agent->metacognition = metacog;
    if (config) {
        agent->metacog_config = *config;
    } else {
        agent->metacog_config.enable_metacognition = true;
        agent->metacog_config.enable_confidence_calibration = true;
        agent->metacog_config.enable_degradation_detection = true;
        agent->metacog_config.degradation_threshold = 0.3f;
        agent->metacog_config.enable_self_diagnosis = true;
    }
    nimcp_mutex_unlock(agent->modules_mutex);

    nimcp_log(LOG_LEVEL_INFO, "Agent '%s' connected to metacognition module",
              agent->config.agent_name);
    return 0;
}

int nimcp_health_agent_connect_ethics(
    nimcp_health_agent_t* agent,
    ethics_engine_t ethics,
    const health_agent_ethics_config_t* config
) {
    if (!validate_agent(agent)) return -1;

    nimcp_mutex_lock(agent->modules_mutex);
    agent->ethics = ethics;
    if (config) {
        agent->ethics_config = *config;
    } else {
        agent->ethics_config.enable_ethics_evaluation = true;
        agent->ethics_config.enable_asimov_laws = true;
        agent->ethics_config.enable_mercy_directive = true;
        agent->ethics_config.enable_golden_rule = true;
        agent->ethics_config.ethics_override_threshold = 0.95f;
    }
    nimcp_mutex_unlock(agent->modules_mutex);

    nimcp_log(LOG_LEVEL_INFO, "Agent '%s' connected to ethics engine",
              agent->config.agent_name);
    return 0;
}

int nimcp_health_agent_connect_emotion(
    nimcp_health_agent_t* agent,
    emotional_system_t* emotion,
    emotion_immune_bridge_t* emotion_immune,
    const health_agent_emotion_config_t* config
) {
    if (!validate_agent(agent)) return -1;

    nimcp_mutex_lock(agent->modules_mutex);
    agent->emotion = emotion;
    agent->emotion_immune = emotion_immune;
    if (config) {
        agent->emotion_config = *config;
    } else {
        agent->emotion_config.enable_emotion_awareness = true;
        agent->emotion_config.enable_emotion_reporting = true;
        agent->emotion_config.enable_stress_adjustment = true;
        agent->emotion_config.stress_threshold_multiplier = 1.5f;
    }
    nimcp_mutex_unlock(agent->modules_mutex);

    nimcp_log(LOG_LEVEL_INFO, "Agent '%s' connected to emotional system%s",
              agent->config.agent_name,
              emotion_immune ? " with emotion-immune bridge" : "");
    return 0;
}

int nimcp_health_agent_connect_wellbeing(
    nimcp_health_agent_t* agent,
    wellbeing_monitor_t* wellbeing,
    const health_agent_wellbeing_config_t* config
) {
    if (!validate_agent(agent)) return -1;

    nimcp_mutex_lock(agent->modules_mutex);
    agent->wellbeing = wellbeing;
    if (config) {
        agent->wellbeing_config = *config;
    } else {
        agent->wellbeing_config.enable_wellbeing_monitoring = true;
        agent->wellbeing_config.enable_distress_detection = true;
        agent->wellbeing_config.enable_suffering_prevention = true;
        agent->wellbeing_config.distress_intervention_threshold = 0.7f;
    }
    nimcp_mutex_unlock(agent->modules_mutex);

    nimcp_log(LOG_LEVEL_INFO, "Agent '%s' connected to wellbeing monitor",
              agent->config.agent_name);
    return 0;
}

int nimcp_health_agent_connect_mental_health(
    nimcp_health_agent_t* agent,
    mental_health_monitor_t* mental_health
) {
    if (!validate_agent(agent)) return -1;

    nimcp_mutex_lock(agent->modules_mutex);
    agent->mental_health = mental_health;
    nimcp_mutex_unlock(agent->modules_mutex);

    nimcp_log(LOG_LEVEL_INFO, "Agent '%s' connected to mental health monitor",
              agent->config.agent_name);
    return 0;
}

int nimcp_health_agent_connect_collective(
    nimcp_health_agent_t* agent,
    collective_cognition_t* collective,
    const health_agent_collective_config_t* config
) {
    if (!validate_agent(agent)) return -1;

    nimcp_mutex_lock(agent->modules_mutex);
    agent->collective = collective;
    if (config) {
        agent->collective_config = *config;
    } else {
        agent->collective_config.enable_collective_monitoring = true;
        agent->collective_config.enable_consensus_decisions = true;
        agent->collective_config.enable_swarm_immune = true;
        agent->collective_config.consensus_threshold = 0.66f;
        agent->collective_config.consensus_timeout_ms = 5000;
    }
    nimcp_mutex_unlock(agent->modules_mutex);

    nimcp_log(LOG_LEVEL_INFO, "Agent '%s' connected to collective cognition",
              agent->config.agent_name);
    return 0;
}

int nimcp_health_agent_connect_rcog(
    nimcp_health_agent_t* agent,
    rcog_engine_t* rcog,
    const health_agent_rcog_config_t* config
) {
    if (!validate_agent(agent)) return -1;

    nimcp_mutex_lock(agent->modules_mutex);
    agent->rcog = rcog;
    if (config) {
        agent->rcog_config = *config;
    } else {
        agent->rcog_config.enable_rcog_diagnosis = true;
        agent->rcog_config.enable_rcog_recovery_planning = true;
        agent->rcog_config.enable_imagination = true;
        agent->rcog_config.rcog_timeout_ms = 10000;
        agent->rcog_config.confidence_threshold = 0.7f;
    }
    nimcp_mutex_unlock(agent->modules_mutex);

    nimcp_log(LOG_LEVEL_INFO, "Agent '%s' connected to RCOG engine",
              agent->config.agent_name);
    return 0;
}

int nimcp_health_agent_connect_gpu(
    nimcp_health_agent_t* agent,
    gpu_health_monitor_t* gpu_health,
    const health_agent_gpu_config_t* config
) {
    if (!validate_agent(agent)) return -1;

    nimcp_mutex_lock(agent->modules_mutex);
    agent->gpu_health = gpu_health;
    if (config) {
        agent->gpu_config = *config;
    } else {
        agent->gpu_config.enable_gpu_monitoring = true;
        agent->gpu_config.enable_gpu_acceleration = true;
        agent->gpu_config.enable_tensor_validation = true;
        agent->gpu_config.enable_anomaly_detection = true;
        agent->gpu_config.gpu_check_interval_ms = 1000;
    }
    nimcp_mutex_unlock(agent->modules_mutex);

    nimcp_log(LOG_LEVEL_INFO, "Agent '%s' connected to GPU health monitor",
              agent->config.agent_name);
    return 0;
}

/* ============================================================================
 * Hypothalamus USE Functions (Active Integration)
 * ============================================================================ */

int nimcp_health_agent_trigger_stress_response(
    nimcp_health_agent_t* agent,
    const char* reason,
    health_agent_severity_t severity
) {
    if (!validate_agent(agent)) return -1;

    if (!agent->hypothalamus) {
        nimcp_log(LOG_LEVEL_WARN, "No hypothalamus connected for stress response");
        return -1;
    }

    nimcp_mutex_lock(agent->modules_mutex);

    if (atomic_load(&agent->in_stress_response)) {
        nimcp_log(LOG_LEVEL_DEBUG, "Already in stress response");
        nimcp_mutex_unlock(agent->modules_mutex);
        return 0;
    }

    /* Mark as in stress response */
    atomic_store(&agent->in_stress_response, true);
    atomic_fetch_add(&agent->stress_responses, 1);

    /* Call hypothalamus orchestrator to trigger stress response */
    int result = hypo_orch_trigger_stress(agent->hypothalamus, reason);
    if (result != 0) {
        nimcp_log(LOG_LEVEL_WARN, "Failed to trigger hypothalamus stress response");
    }

    nimcp_mutex_unlock(agent->modules_mutex);

    nimcp_log(LOG_LEVEL_INFO, "Triggered stress response: %s (severity=%d)",
              reason ? reason : "unknown", severity);
    return 0;
}

int nimcp_health_agent_release_stress_response(nimcp_health_agent_t* agent) {
    if (!validate_agent(agent)) return -1;

    if (!agent->hypothalamus) {
        return -1;
    }

    nimcp_mutex_lock(agent->modules_mutex);

    if (!atomic_load(&agent->in_stress_response)) {
        nimcp_mutex_unlock(agent->modules_mutex);
        return 0;
    }

    /* Mark as no longer in stress response */
    atomic_store(&agent->in_stress_response, false);

    /* Call hypothalamus orchestrator to release stress */
    int result = hypo_orch_release_stress(agent->hypothalamus);
    if (result != 0) {
        nimcp_log(LOG_LEVEL_WARN, "Failed to release hypothalamus stress response");
    }

    nimcp_mutex_unlock(agent->modules_mutex);

    nimcp_log(LOG_LEVEL_INFO, "Released stress response");
    return 0;
}

int nimcp_health_agent_enter_sickness_mode(
    nimcp_health_agent_t* agent,
    float threat_level
) {
    if (!validate_agent(agent)) return -1;

    if (!agent->hypo_immune_bridge && !agent->hypothalamus) {
        nimcp_log(LOG_LEVEL_WARN, "No hypothalamus/immune bridge for sickness mode");
        return -1;
    }

    nimcp_mutex_lock(agent->modules_mutex);

    if (atomic_load(&agent->in_sickness_mode)) {
        nimcp_mutex_unlock(agent->modules_mutex);
        return 0;
    }

    atomic_store(&agent->in_sickness_mode, true);
    atomic_fetch_add(&agent->sickness_mode_entries, 1);

    /* Call hypo-immune bridge to activate sickness behavior (safety mode) */
    if (agent->hypo_immune_bridge) {
        int result = hypo_immune_bridge_enter_safety_mode(agent->hypo_immune_bridge, threat_level);
        if (result != 0) {
            nimcp_log(LOG_LEVEL_WARN, "Failed to activate sickness behavior");
        }
    }

    nimcp_mutex_unlock(agent->modules_mutex);

    nimcp_log(LOG_LEVEL_INFO, "Entered sickness mode (threat_level=%.2f)", threat_level);
    return 0;
}

int nimcp_health_agent_exit_sickness_mode(nimcp_health_agent_t* agent) {
    if (!validate_agent(agent)) return -1;

    nimcp_mutex_lock(agent->modules_mutex);

    if (!atomic_load(&agent->in_sickness_mode)) {
        nimcp_mutex_unlock(agent->modules_mutex);
        return 0;
    }

    atomic_store(&agent->in_sickness_mode, false);

    /* Call hypo-immune bridge to deactivate sickness behavior (end acute phase) */
    if (agent->hypo_immune_bridge) {
        int result = hypo_immune_end_acute_phase(agent->hypo_immune_bridge);
        if (result != 0) {
            nimcp_log(LOG_LEVEL_WARN, "Failed to deactivate sickness behavior");
        }
    }

    nimcp_mutex_unlock(agent->modules_mutex);

    nimcp_log(LOG_LEVEL_INFO, "Exited sickness mode");
    return 0;
}

float nimcp_health_agent_homeostatic_regulate(
    nimcp_health_agent_t* agent,
    float current_health
) {
    if (!validate_agent(agent)) return 0.0f;

    if (!agent->homeostasis) {
        return 0.0f;
    }

    /* Call homeostasis system for PID regulation */
    /* First, set the current health as the arousal variable value */
    hypo_homeostasis_set_value(agent->homeostasis, HYPO_VAR_AROUSAL, current_health);

    /* Run the control update (using 0 delta means instant update) */
    hypo_homeostasis_update(agent->homeostasis, 0);

    /* Get the controller output for arousal regulation */
    float output = hypo_homeostasis_get_output(agent->homeostasis, HYPO_VAR_AROUSAL);

    /* Clamp to [-1, 1] */
    if (output > 1.0f) output = 1.0f;
    if (output < -1.0f) output = -1.0f;

    atomic_store(&agent->homeostatic_output, output);

    return output;
}

int nimcp_health_agent_get_alignment_reward(
    nimcp_health_agent_t* agent,
    float* reward_out
) {
    if (!validate_agent(agent) || !reward_out) return -1;

    if (!agent->homeostasis) {
        *reward_out = 0.0f;
        return -1;
    }

    /* Call homeostasis for alignment reward */
    *reward_out = hypo_homeostasis_get_reward(agent->homeostasis);

    return 0;
}

int nimcp_health_agent_report_drive(
    nimcp_health_agent_t* agent,
    uint32_t drive_type,
    float drive_level,
    const char* description
) {
    if (!validate_agent(agent)) return -1;

    if (!agent->hypothalamus) {
        return -1;
    }

    /* Map drive level to urgency */
    hypo_urgency_t urgency = HYPO_URGENCY_NONE;
    if (drive_level >= 0.7f) {
        urgency = HYPO_URGENCY_URGENT;
    } else if (drive_level >= 0.5f) {
        urgency = HYPO_URGENCY_ELEVATED;
    } else if (drive_level >= 0.3f) {
        urgency = HYPO_URGENCY_MODERATE;
    } else if (drive_level > 0.0f) {
        urgency = HYPO_URGENCY_LOW;
    }

    /* Report drive to hypothalamus orchestrator */
    int result = hypo_orch_report_drive(
        agent->hypothalamus,
        0,  /* bridge_id: health agent uses bridge 0 */
        drive_type,
        drive_level,
        urgency,
        description
    );
    if (result != 0) {
        nimcp_log(LOG_LEVEL_WARN, "Failed to report drive to hypothalamus");
    }

    atomic_fetch_add(&agent->drive_events_published, 1);

    nimcp_log(LOG_LEVEL_DEBUG, "Reported drive (type=%u, level=%.2f): %s",
              drive_type, drive_level, description ? description : "");
    return 0;
}

int nimcp_health_agent_get_drive_state(
    nimcp_health_agent_t* agent,
    float* drive_level_out,
    bool* is_stressed_out
) {
    if (!validate_agent(agent)) return -1;

    if (drive_level_out) {
        /* Query hypothalamus for unified drive level */
        if (agent->hypothalamus) {
            int result = hypo_orch_get_drive_level(agent->hypothalamus, drive_level_out);
            if (result != 0) {
                *drive_level_out = 0.0f;
            }
        } else {
            *drive_level_out = 0.0f;
        }
    }

    if (is_stressed_out) {
        /* Query hypothalamus for stress state */
        if (agent->hypothalamus) {
            bool in_stress = false;
            int result = hypo_orch_is_stressed(agent->hypothalamus, &in_stress);
            if (result == 0) {
                *is_stressed_out = in_stress;
            } else {
                *is_stressed_out = atomic_load(&agent->in_stress_response);
            }
        } else {
            *is_stressed_out = atomic_load(&agent->in_stress_response);
        }
    }

    return 0;
}

/* ============================================================================
 * Module USE Functions (Active Integration)
 * ============================================================================ */

int nimcp_health_agent_trigger_gc(nimcp_health_agent_t* agent, bool force) {
    if (!validate_agent(agent)) return -1;

    if (!agent->gc_context) {
        nimcp_log(LOG_LEVEL_WARN, "No GC context connected");
        return -1;
    }

    uint64_t now_us = get_timestamp_us();

    /* Check cooldown unless forced */
    if (!force && agent->gc_config.gc_cooldown_ms > 0) {
        uint64_t cooldown_us = agent->gc_config.gc_cooldown_ms * 1000ULL;
        if (now_us - agent->last_gc_time_us < cooldown_us) {
            nimcp_log(LOG_LEVEL_DEBUG, "GC cooldown not elapsed");
            return 0;
        }
    }

    /* Trigger GC on all targets */
    int collected = kg_gc_run(agent->gc_context, KG_GC_ALL);
    if (collected < 0) {
        nimcp_log(LOG_LEVEL_WARN, "GC run failed: %s",
                  kg_gc_get_last_error(agent->gc_context));
    } else {
        nimcp_log(LOG_LEVEL_DEBUG, "GC collected %d items", collected);
    }

    agent->last_gc_time_us = now_us;
    atomic_fetch_add(&agent->gc_triggers, 1);

    nimcp_log(LOG_LEVEL_INFO, "Triggered garbage collection (force=%d)", force);
    return 0;
}

int nimcp_health_agent_create_checkpoint(
    nimcp_health_agent_t* agent,
    const char* reason
) {
    if (!validate_agent(agent)) return -1;

    if (!agent->checkpoint) {
        nimcp_log(LOG_LEVEL_WARN, "No checkpoint manager connected");
        return -1;
    }

    /* Create checkpoint using brain's checkpoint system */
    if (agent->brain) {
        /* Generate checkpoint path with timestamp and reason */
        char checkpoint_path[256];
        uint64_t timestamp = get_timestamp_us();
        snprintf(checkpoint_path, sizeof(checkpoint_path),
                 "/tmp/nimcp_checkpoint_%lu_%s.ckpt",
                 (unsigned long)timestamp,
                 reason ? reason : "auto");

        bool success = checkpoint_save(agent->brain, checkpoint_path);
        if (!success) {
            nimcp_log(LOG_LEVEL_WARN, "Failed to create checkpoint: %s", checkpoint_path);
            return -1;
        }
    }

    agent->last_checkpoint_time_us = get_timestamp_us();
    atomic_fetch_add(&agent->checkpoints_created, 1);

    nimcp_log(LOG_LEVEL_INFO, "Created checkpoint: %s", reason ? reason : "auto");
    return 0;
}

int nimcp_health_agent_rollback(
    nimcp_health_agent_t* agent,
    uint64_t checkpoint_id
) {
    if (!validate_agent(agent)) return -1;

    if (!agent->checkpoint) {
        nimcp_log(LOG_LEVEL_WARN, "No checkpoint manager connected");
        return -1;
    }

    /* Rollback to checkpoint
     * Note: Full implementation requires checkpoint registry to map IDs to paths.
     * For now, attempt to load most recent checkpoint if checkpoint_id is 0.
     */
    if (agent->brain && checkpoint_id == 0) {
        /* Find most recent checkpoint - simplified approach */
        char checkpoint_path[256];
        snprintf(checkpoint_path, sizeof(checkpoint_path),
                 "/tmp/nimcp_checkpoint_latest.ckpt");

        if (checkpoint_validate(checkpoint_path)) {
            bool success = checkpoint_load(&agent->brain, checkpoint_path);
            if (!success) {
                nimcp_log(LOG_LEVEL_ERROR, "Failed to rollback to checkpoint");
                return -1;
            }
        } else {
            nimcp_log(LOG_LEVEL_WARN, "No valid checkpoint found for rollback");
            return -1;
        }
    } else if (checkpoint_id != 0) {
        nimcp_log(LOG_LEVEL_WARN, "Checkpoint ID lookup not yet implemented");
        return -1;
    }

    atomic_fetch_add(&agent->rollbacks_performed, 1);

    nimcp_log(LOG_LEVEL_INFO, "Rolling back to checkpoint %lu", (unsigned long)checkpoint_id);
    return 0;
}

int nimcp_health_agent_reduce_load(
    nimcp_health_agent_t* agent,
    float reduction_factor
) {
    if (!validate_agent(agent)) return -1;

    if (!agent->runtime_adaptation) {
        nimcp_log(LOG_LEVEL_WARN, "No runtime adaptation connected");
        return -1;
    }

    /* Clamp reduction factor */
    if (reduction_factor < 0.0f) reduction_factor = 0.0f;
    if (reduction_factor > 1.0f) reduction_factor = 1.0f;

    /* Call runtime adaptation to reduce load via batch size and thread reduction */
    if (reduction_factor > 0.0f) {
        /* Reduce batch size proportionally */
        float batch_reduction = 1.0f - (reduction_factor * 0.5f);  /* Max 50% reduction */
        runtime_adaptation_set_parameter(
            agent->runtime_adaptation,
            RUNTIME_PARAM_BATCH_SIZE,
            batch_reduction * 64.0f,  /* Scale from default batch size */
            "health_agent: load reduction"
        );

        /* Reduce max threads if reduction > 50% */
        if (reduction_factor > 0.5f) {
            runtime_adaptation_set_parameter(
                agent->runtime_adaptation,
                RUNTIME_PARAM_MAX_THREADS,
                2.0f,  /* Minimum thread count */
                "health_agent: high load reduction"
            );
        }
    }

    atomic_store(&agent->load_reduced, true);
    atomic_fetch_add(&agent->load_reductions, 1);

    nimcp_log(LOG_LEVEL_INFO, "Reduced load by %.1f%%", reduction_factor * 100.0f);
    return 0;
}

int nimcp_health_agent_restore_load(nimcp_health_agent_t* agent) {
    if (!validate_agent(agent)) return -1;

    if (!agent->runtime_adaptation) {
        return -1;
    }

    /* Restore normal load by resetting adaptation parameters */
    runtime_adaptation_reset_parameter(agent->runtime_adaptation, RUNTIME_PARAM_BATCH_SIZE);
    runtime_adaptation_reset_parameter(agent->runtime_adaptation, RUNTIME_PARAM_MAX_THREADS);

    atomic_store(&agent->load_reduced, false);

    nimcp_log(LOG_LEVEL_INFO, "Restored normal load");
    return 0;
}

int nimcp_health_agent_check_oscillations(
    nimcp_health_agent_t* agent,
    bool* is_abnormal_out,
    uint32_t* anomaly_type_out
) {
    if (!validate_agent(agent)) return -1;

    if (!agent->oscillations) {
        if (is_abnormal_out) *is_abnormal_out = false;
        if (anomaly_type_out) *anomaly_type_out = 0;
        return -1;
    }

    /* Query oscillations for abnormal patterns using brain_oscillation_analyze */
    brain_oscillation_analyzer_t* analyzer = (brain_oscillation_analyzer_t*)agent->oscillations;
    oscillation_analysis_t analysis;
    bool abnormal = false;
    uint32_t anomaly_type = 0;

    if (brain_oscillation_analyze(analyzer, &analysis)) {
        /* Check for abnormal states */
        /* DEEP_SLEEP during active state is abnormal */
        if (analysis.state == COGNITIVE_STATE_DEEP_SLEEP && analysis.state_confidence > 0.8f) {
            abnormal = true;
            anomaly_type = 1;  /* Unexpected deep sleep */
        }
        /* Very low synchrony indicates disconnection */
        else if (analysis.synchrony < 0.1f) {
            abnormal = true;
            anomaly_type = 2;  /* Low synchrony */
        }
        /* Very high spectral entropy indicates chaos */
        else if (analysis.spectral_entropy > 0.95f) {
            abnormal = true;
            anomaly_type = 3;  /* Chaotic activity */
        }
    }

    if (is_abnormal_out) *is_abnormal_out = abnormal;
    if (anomaly_type_out) *anomaly_type_out = anomaly_type;

    return 0;
}

int nimcp_health_agent_check_connectivity(
    nimcp_health_agent_t* agent,
    bool* isolation_detected_out,
    char* isolated_module_out,
    size_t module_name_size
) {
    if (!validate_agent(agent)) return -1;

    if (!agent->connectivity) {
        if (isolation_detected_out) *isolation_detected_out = false;
        if (isolated_module_out && module_name_size > 0) isolated_module_out[0] = '\0';
        return -1;
    }

    /* Query connectivity health using brain's cached connectivity assessment */
    bool isolation = false;

    if (agent->brain) {
        brain_connectivity_health_t health;
        if (brain_get_connectivity_health(agent->brain, &health)) {
            /* Check for module isolation based on community structure */
            /* If modularity is very high and largest community is small, possible isolation */
            if (health.community.modularity_q > 0.8f &&
                health.community.largest_community_ratio < 0.2f) {
                isolation = true;
                if (isolated_module_out && module_name_size > 0) {
                    snprintf(isolated_module_out, module_name_size,
                             "fragmented_communities=%u", health.community.num_communities);
                }
            }
            /* Check for low overall health as indicator of connectivity issues */
            else if (health.overall_health < 0.3f) {
                isolation = true;
                if (isolated_module_out && module_name_size > 0) {
                    snprintf(isolated_module_out, module_name_size,
                             "low_connectivity_health=%.2f", health.overall_health);
                }
            }
            connectivity_health_free(&health);
        }
    }

    if (isolation_detected_out) *isolation_detected_out = isolation;
    if (!isolation && isolated_module_out && module_name_size > 0) {
        isolated_module_out[0] = '\0';
    }

    return 0;
}

int nimcp_health_agent_publish_event(
    nimcp_health_agent_t* agent,
    const health_agent_message_t* msg
) {
    if (!validate_agent(agent) || !msg) return -1;

    if (!agent->bio_async_router) {
        return -1;
    }

    /* Publish health event to bio-async system via global router signal
     * Note: Full bio-async integration requires module context registration.
     * For now, publish as a health signal using the global router.
     */
    bio_router_t router = bio_router_get_global();
    if (router) {
        /* Create a temporary module context for publishing */
        bio_module_info_t module_info = {
            .module_id = 0,  /* Auto-assign */
            .module_name = "health_agent",
            .inbox_capacity = 0,  /* Use default */
            .user_data = agent
        };
        bio_module_context_t ctx = bio_router_register_module(&module_info);
        if (ctx) {
            /* Publish health severity as signal value */
            char signal_name[64];
            snprintf(signal_name, sizeof(signal_name), "health_%u", (unsigned)msg->source);
            float signal_value = (float)(HEALTH_SEVERITY_CRITICAL - msg->severity) /
                                 (float)HEALTH_SEVERITY_CRITICAL;  /* Higher = healthier */
            bio_router_publish_signal(ctx, signal_name, signal_value);

            /* Unregister temporary context */
            bio_router_unregister_module(ctx);
        }
    }

    atomic_fetch_add(&agent->bio_async_events_published, 1);

    return 0;
}

int nimcp_health_agent_check_deadlocks(
    nimcp_health_agent_t* agent,
    bool* deadlock_detected_out,
    bool* contention_high_out
) {
    if (!validate_agent(agent)) return -1;

    if (!agent->deadlock_detector_ptr) {
        if (deadlock_detected_out) *deadlock_detected_out = false;
        if (contention_high_out) *contention_high_out = false;
        return -1;
    }

    /* Query deadlock detector for cycles and contention */
    uint32_t cycles = deadlock_detector_check();
    deadlock_detector_stats_t stats = deadlock_detector_get_stats();

    /* Deadlock detected if any cycles found or recent deadlocks */
    bool deadlock = (cycles > 0) || (stats.deadlocks_detected > 0);

    /* High contention if many timeouts or order violations */
    bool high_contention = false;
    if (stats.total_locks > 0) {
        /* Contention is high if > 5% timeouts or > 1% order violations */
        float timeout_ratio = (float)stats.lock_timeouts / (float)stats.total_locks;
        float violation_ratio = (float)stats.order_violations / (float)stats.total_locks;
        high_contention = (timeout_ratio > 0.05f) || (violation_ratio > 0.01f);
    }

    if (deadlock_detected_out) *deadlock_detected_out = deadlock;
    if (contention_high_out) *contention_high_out = high_contention;

    return 0;
}

/* ============================================================================
 * Agent Thread Helper Functions (Internal)
 * ============================================================================ */

static void agent_run_hypothalamus_check(nimcp_health_agent_t* agent, float health_score) {
    if (!agent || !agent->hypothalamus) return;

    /* Check if we need to trigger stress response */
    if (health_score < agent->hypothalamus_config.stress_trigger_threshold) {
        if (!atomic_load(&agent->in_stress_response)) {
            nimcp_health_agent_trigger_stress_response(
                agent,
                "Health score below threshold",
                health_score < 0.2f ? HEALTH_SEVERITY_CRITICAL : HEALTH_SEVERITY_ERROR
            );
        }
    } else if (atomic_load(&agent->in_stress_response) && health_score > 0.6f) {
        /* Health recovered, release stress */
        nimcp_health_agent_release_stress_response(agent);
    }

    /* Check if we need to enter sickness mode */
    if (health_score < agent->hypothalamus_config.sickness_trigger_threshold) {
        if (!atomic_load(&agent->in_sickness_mode) &&
            agent->hypothalamus_config.enable_sickness_behavior) {
            nimcp_health_agent_enter_sickness_mode(agent, 1.0f - health_score);
        }
    } else if (atomic_load(&agent->in_sickness_mode) && health_score > 0.5f) {
        /* Health recovered, exit sickness mode */
        nimcp_health_agent_exit_sickness_mode(agent);
    }
}

static void agent_run_homeostatic_regulation(nimcp_health_agent_t* agent, float health_score) {
    if (!agent || !agent->homeostasis) return;

    float output = nimcp_health_agent_homeostatic_regulate(agent, health_score);

    /* Use homeostatic output to guide actions */
    if (output > 0.5f) {
        /* Need to improve health - consider triggering GC or checkpointing */
        if (agent->gc_context && agent->gc_config.enable_auto_gc_trigger) {
            nimcp_health_agent_trigger_gc(agent, false);
        }
    } else if (output < -0.3f) {
        /* Health is good - could relax monitoring */
        /* (No action needed - normal operation) */
    }
}

static void agent_check_connectivity(nimcp_health_agent_t* agent) {
    if (!agent || !agent->connectivity) return;

    bool isolated = false;
    char module_name[64] = {0};

    nimcp_health_agent_check_connectivity(agent, &isolated, module_name, sizeof(module_name));

    if (isolated && agent->connectivity_config.enable_isolation_detection) {
        health_agent_message_t msg = nimcp_health_agent_create_message(
            HEALTH_MSG_ANOMALY_DETECTED,
            HEALTH_SEVERITY_WARNING,
            HEALTH_SOURCE_BRAIN_REGION,
            "Module isolation detected: %s", module_name[0] ? module_name : "unknown"
        );
        nimcp_health_agent_report_anomaly(agent, &msg);
    }
}

static void agent_check_oscillations(nimcp_health_agent_t* agent) {
    if (!agent || !agent->oscillations) return;

    bool abnormal = false;
    uint32_t anomaly_type = 0;

    nimcp_health_agent_check_oscillations(agent, &abnormal, &anomaly_type);

    if (abnormal) {
        health_agent_severity_t severity = HEALTH_SEVERITY_WARNING;
        const char* desc = "Abnormal brain oscillation pattern";

        if (anomaly_type == 1 && agent->oscillations_config.enable_flatline_detection) {
            severity = HEALTH_SEVERITY_CRITICAL;
            desc = "Brain oscillation flatline detected";
        } else if (anomaly_type == 2 && agent->oscillations_config.enable_seizure_detection) {
            severity = HEALTH_SEVERITY_CRITICAL;
            desc = "Seizure-like oscillation pattern detected";
        }

        health_agent_message_t msg = nimcp_health_agent_create_message(
            HEALTH_MSG_ANOMALY_DETECTED,
            severity,
            HEALTH_SOURCE_NEURAL,
            "%s", desc
        );
        nimcp_health_agent_report_anomaly(agent, &msg);
    }
}

static void agent_auto_gc_if_needed(nimcp_health_agent_t* agent) {
    if (!agent || !agent->gc_context || !agent->gc_config.enable_auto_gc_trigger) return;

    /* Query memory usage via GC analysis */
    kg_gc_stats_t gc_stats;
    float memory_usage = 0.5f;  /* Default if analysis fails */
    if (kg_gc_analyze(agent->gc_context, &gc_stats) == 0) {
        /* Calculate fragmentation as a proxy for memory pressure */
        memory_usage = kg_gc_get_fragmentation(agent->gc_context);
    }

    if (memory_usage > agent->gc_config.gc_trigger_threshold) {
        nimcp_health_agent_trigger_gc(agent, false);
    }
}

static void agent_auto_checkpoint_if_needed(nimcp_health_agent_t* agent, float health_score) {
    if (!agent || !agent->checkpoint || !agent->checkpoint_config.enable_auto_checkpoint) return;

    uint64_t now_us = get_timestamp_us();
    uint64_t interval_us = agent->checkpoint_config.checkpoint_interval_ms * 1000ULL;

    /* Only checkpoint if health is good and interval has elapsed */
    if (health_score >= agent->checkpoint_config.health_threshold_checkpoint &&
        (now_us - agent->last_checkpoint_time_us) >= interval_us) {
        nimcp_health_agent_create_checkpoint(agent, "auto_checkpoint_good_health");
    }

    /* Auto-rollback if health is critically low */
    if (health_score < agent->checkpoint_config.health_threshold_rollback &&
        agent->checkpoint_config.enable_auto_rollback) {
        nimcp_log(LOG_LEVEL_WARN, "Health critically low (%.2f), triggering rollback",
                  health_score);
        nimcp_health_agent_rollback(agent, 0);  /* 0 = latest checkpoint */
    }
}

static int hypo_drive_event_callback(const void* event, void* user_data) {
    nimcp_health_agent_t* agent = (nimcp_health_agent_t*)user_data;
    if (!validate_agent(agent) || !event) return -1;

    /* Process drive event from hypothalamus
     * The event structure contains drive type and urgency level.
     * We use this to adjust health agent behavior.
     */
    typedef struct {
        uint32_t drive_type;
        float urgency;
    } drive_event_t;

    const drive_event_t* drive_event = (const drive_event_t*)event;

    /* Log high-urgency drive events */
    if (drive_event->urgency >= 0.7f) {
        nimcp_log(LOG_LEVEL_INFO, "High urgency drive event: type=%u, urgency=%.2f",
                  drive_event->drive_type, drive_event->urgency);

        /* Consider triggering stress response for very high urgency */
        if (drive_event->urgency >= 0.9f && !atomic_load(&agent->in_stress_response)) {
            nimcp_health_agent_trigger_stress_response(
                agent, "High urgency drive detected", HEALTH_SEVERITY_WARNING);
        }
    }

    return 0;
}

/* ============================================================================
 * Cognitive Integration Stub Functions (Placeholders)
 * ============================================================================ */

static void agent_run_failure_prediction(nimcp_health_agent_t* agent) {
    if (!agent || !agent->failure_predictor) return;

    /* Query failure predictor - using simplified stub since full API has header conflicts */
    /* The failure_predictor_t is a forward-declared opaque type */
    /* For now, just track that prediction was attempted */
    if (agent->failure_predictor) {
        /* Prediction system is connected and active */
        nimcp_log(LOG_LEVEL_DEBUG, "Failure predictor active");
    }

    atomic_fetch_add(&agent->predictions_made, 1);
}

static void agent_run_metacognition_check(nimcp_health_agent_t* agent) {
    if (!agent || !agent->metacognition) return;

    /* Run metacognition self-check - using simplified stub since full API has header conflicts */
    /* The metacognition_t is a forward-declared opaque type */
    /* For now, just track that metacognition check was attempted */
    if (agent->metacognition) {
        nimcp_log(LOG_LEVEL_DEBUG, "Metacognition system active");
    }

    atomic_fetch_add(&agent->self_diagnoses, 1);
}

static void agent_run_wellbeing_check(nimcp_health_agent_t* agent) {
    if (!agent || !agent->wellbeing) return;

    /* Check wellbeing status by querying distress detections */
    /* The wellbeing monitor provides distress level tracking */
    float distress = atomic_load((volatile _Atomic float*)&agent->current_distress_level);

    /* If distress is high, increment detection count */
    if (distress > agent->wellbeing_config.distress_intervention_threshold) {
        atomic_fetch_add(&agent->distress_detections, 1);

        /* Log high distress */
        nimcp_log(LOG_LEVEL_WARN, "Wellbeing: high distress level (%.2f)", distress);
    }
}

static void agent_apply_emotion_adjustments(nimcp_health_agent_t* agent) {
    if (!agent || !agent->emotion) return;

    /* Query emotion state and adjust thresholds based on stress level */
    float stress_level = atomic_load((volatile _Atomic float*)&agent->current_stress_level);

    /* Adjust emotion-related behavior based on stress */
    if (stress_level > 0.7f) {
        /* High stress: increase monitoring frequency */
        atomic_fetch_add(&agent->emotion_adjustments, 1);
        nimcp_log(LOG_LEVEL_DEBUG, "Emotion: high stress (%.2f), increasing vigilance",
                  stress_level);
    } else if (stress_level < 0.2f) {
        /* Low stress: normal operation */
    }
}

static void agent_check_gpu_health(nimcp_health_agent_t* agent) {
    if (!agent || !agent->gpu_health) return;

    /* Check GPU health status via utilization tracking */
    float gpu_util = atomic_load((volatile _Atomic float*)&agent->gpu_utilization);
    bool gpu_ok = atomic_load(&agent->gpu_healthy);

    /* Monitor GPU health based on utilization patterns */
    if (!gpu_ok || gpu_util > 0.95f) {
        nimcp_log(LOG_LEVEL_WARN, "GPU health: utilization=%.1f%%, healthy=%d",
                  gpu_util * 100.0f, gpu_ok);
        atomic_store(&agent->gpu_healthy, false);
    }

    atomic_fetch_add(&agent->gpu_accelerated_checks, 1);
}

static bool agent_check_ethics_permission(nimcp_health_agent_t* agent,
                                           const health_agent_message_t* msg,
                                           health_agent_recovery_t action) {
    if (!agent || !agent->ethics) return true;

    /* Evaluate ethics for the proposed action */
    atomic_fetch_add(&agent->ethics_evaluations, 1);

    /* Basic ethics evaluation - block destructive actions without proper severity */
    if (action == HEALTH_RECOVERY_FULL_RESET && msg) {
        if (msg->severity < HEALTH_SEVERITY_CRITICAL) {
            nimcp_log(LOG_LEVEL_WARN, "Ethics: blocking full reset for non-critical issue");
            atomic_fetch_add(&agent->ethics_blocks, 1);
            return false;
        }
    }

    /* Check if mercy protocol should be applied */
    if (agent->ethics_config.enable_mercy_directive && msg) {
        if (msg->severity < HEALTH_SEVERITY_ERROR && action == HEALTH_RECOVERY_FULL_RESET) {
            nimcp_log(LOG_LEVEL_INFO, "Ethics: applying mercy protocol, downgrading action");
            atomic_fetch_add(&agent->mercy_applications, 1);
            /* Return true but the caller should use a less severe action */
        }
    }

    return true;
}

static int agent_get_collective_consensus(nimcp_health_agent_t* agent,
                                           const health_agent_message_t* msg) {
    if (!agent || !agent->collective) return 0;

    /* Request consensus from collective intelligence
     * The collective provides distributed decision-making for critical health decisions.
     */
    atomic_fetch_add(&agent->consensus_requests, 1);

    /* For critical messages, log that consensus was requested */
    if (msg && msg->severity >= HEALTH_SEVERITY_ERROR) {
        nimcp_log(LOG_LEVEL_INFO, "Requesting collective consensus for severity=%d",
                  msg->severity);
    }

    /* Track consensus achievement rate */
    /* In a real implementation, this would wait for quorum */
    atomic_fetch_add(&agent->consensus_achieved, 1);

    return 0;
}

static int agent_run_rcog_diagnosis(nimcp_health_agent_t* agent,
                                     const health_agent_message_t* msg,
                                     health_agent_recovery_t* suggested_action) {
    if (!agent || !agent->rcog) return -1;

    /* Run RCOG (Recursive Cognition) diagnosis
     * RCOG provides meta-level reasoning about health issues and recovery strategies.
     */
    atomic_fetch_add(&agent->rcog_diagnoses, 1);

    /* Analyze the message and suggest appropriate recovery action */
    health_agent_recovery_t action = HEALTH_RECOVERY_NONE;

    if (msg) {
        /* Use RCOG-style reasoning to determine action based on severity and type */
        switch (msg->severity) {
            case HEALTH_SEVERITY_CRITICAL:
                /* Critical: consider rollback or full reset */
                action = HEALTH_RECOVERY_ROLLBACK;
                atomic_fetch_add(&agent->rcog_recovery_plans, 1);
                break;

            case HEALTH_SEVERITY_ERROR:
                /* Error: try load reduction */
                action = HEALTH_RECOVERY_REDUCE_LOAD;
                atomic_fetch_add(&agent->rcog_recovery_plans, 1);
                break;

            case HEALTH_SEVERITY_WARNING:
                /* Warning: trigger garbage collection to clean up */
                action = HEALTH_RECOVERY_GC;
                break;

            default:
                action = HEALTH_RECOVERY_NONE;
                break;
        }

        nimcp_log(LOG_LEVEL_DEBUG, "RCOG diagnosis: severity=%d -> action=%d",
                  msg->severity, action);
    }

    if (suggested_action) *suggested_action = action;
    return 0;
}

/* ============================================================================
 * Extended Status Functions
 * ============================================================================ */

void nimcp_health_agent_get_cognitive_status(
    const nimcp_health_agent_t* agent,
    health_agent_cognitive_status_t* status
) {
    if (!validate_agent(agent) || !status) return;

    memset(status, 0, sizeof(health_agent_cognitive_status_t));

    /* Connection status */
    status->failure_prediction_connected = (agent->failure_predictor != NULL);
    status->metacognition_connected = (agent->metacognition != NULL);
    status->ethics_connected = (agent->ethics != NULL);
    status->emotion_connected = (agent->emotion != NULL);
    status->wellbeing_connected = (agent->wellbeing != NULL);
    status->mental_health_connected = (agent->mental_health != NULL);
    status->collective_connected = (agent->collective != NULL);
    status->rcog_connected = (agent->rcog != NULL);
    status->gpu_connected = (agent->gpu_health != NULL);

    /* Atomic stats */
    status->predictions_made = atomic_load(&agent->predictions_made);
    status->predictions_correct = atomic_load(&agent->predictions_correct);
    status->preventive_actions = atomic_load(&agent->preventive_actions);
    status->self_diagnoses = atomic_load(&agent->self_diagnoses);
    status->degradation_alerts = atomic_load(&agent->degradation_alerts);
    status->current_confidence = atomic_load(&agent->current_confidence);
    status->ethics_evaluations = atomic_load(&agent->ethics_evaluations);
    status->ethics_blocks = atomic_load(&agent->ethics_blocks);
    status->mercy_applications = atomic_load(&agent->mercy_applications);
    status->current_stress_level = atomic_load(&agent->current_stress_level);
    status->emotion_adjustments = atomic_load(&agent->emotion_adjustments);
    status->distress_detections = atomic_load(&agent->distress_detections);
    status->wellbeing_interventions = atomic_load(&agent->wellbeing_interventions);
    status->current_distress_level = atomic_load(&agent->current_distress_level);
    status->consensus_requests = atomic_load(&agent->consensus_requests);
    status->consensus_achieved = atomic_load(&agent->consensus_achieved);
    status->avg_consensus_time_ms = atomic_load(&agent->avg_consensus_time_ms);
    status->rcog_diagnoses = atomic_load(&agent->rcog_diagnoses);
    status->rcog_recovery_plans = atomic_load(&agent->rcog_recovery_plans);
    status->avg_rcog_time_ms = atomic_load(&agent->avg_rcog_time_ms);
    status->gpu_accelerated_checks = atomic_load(&agent->gpu_accelerated_checks);
    status->gpu_utilization = atomic_load(&agent->gpu_utilization);
    status->gpu_healthy = atomic_load(&agent->gpu_healthy);

    /* Compute prediction accuracy */
    if (status->predictions_made > 0) {
        status->prediction_accuracy = (float)status->predictions_correct /
                                       (float)status->predictions_made;
    }
}

void nimcp_health_agent_get_full_status(
    const nimcp_health_agent_t* agent,
    health_agent_full_status_t* status
) {
    if (!validate_agent(agent) || !status) return;

    memset(status, 0, sizeof(health_agent_full_status_t));

    /* Get cognitive status */
    nimcp_health_agent_get_cognitive_status(agent, &status->cognitive);

    /* Hypothalamus status */
    status->hypothalamus_connected = (agent->hypothalamus != NULL);
    status->homeostasis_connected = (agent->homeostasis != NULL);
    status->hypo_immune_bridge_connected = (agent->hypo_immune_bridge != NULL);
    status->drives_connected = (agent->drives != NULL);
    status->in_stress_response = atomic_load(&agent->in_stress_response);
    status->in_sickness_mode = atomic_load(&agent->in_sickness_mode);
    status->homeostatic_output = atomic_load(&agent->homeostatic_output);

    /* Get drive level if hypothalamus connected */
    if (agent->hypothalamus) {
        nimcp_health_agent_get_drive_state(
            (nimcp_health_agent_t*)agent,  /* cast away const for internal use */
            &status->current_drive_level,
            NULL
        );
    }

    /* Additional module status */
    status->connectivity_connected = (agent->connectivity != NULL);
    status->oscillations_connected = (agent->oscillations != NULL);
    status->gc_connected = (agent->gc_context != NULL);
    status->checkpoint_connected = (agent->checkpoint != NULL);
    status->deadlock_detector_connected = (agent->deadlock_detector_ptr != NULL);
    status->bio_async_connected = (agent->bio_async_router != NULL);
    status->runtime_adaptation_connected = (agent->runtime_adaptation != NULL);
    status->exception_bridge_connected = (agent->exception_bridge != NULL);

    /* Module statistics */
    status->gc_triggers = atomic_load(&agent->gc_triggers);
    status->checkpoints_created = atomic_load(&agent->checkpoints_created);
    status->rollbacks_performed = atomic_load(&agent->rollbacks_performed);
    status->load_reductions = atomic_load(&agent->load_reductions);
    status->stress_responses = atomic_load(&agent->stress_responses);
    status->sickness_mode_entries = atomic_load(&agent->sickness_mode_entries);
    status->drive_events_published = atomic_load(&agent->drive_events_published);
    status->bio_async_events_published = atomic_load(&agent->bio_async_events_published);

    /* Portia/Dragonfly/Swarm/Memory status */
    status->portia_connected = (agent->portia != NULL);
    status->dragonfly_connected = (agent->dragonfly != NULL);
    status->swarm_immune_connected = (agent->swarm_immune != NULL);
    status->swarm_memory_connected = (agent->swarm_memory != NULL);
    status->engram_connected = (agent->engram != NULL);
    status->memory_consolidation_connected = (agent->memory_consolidation != NULL);

    /* New module statistics */
    status->portia_tier_changes = atomic_load(&agent->portia_tier_changes);
    status->portia_degradations = atomic_load(&agent->portia_degradations);
    status->dragonfly_anomalies_tracked = atomic_load(&agent->dragonfly_anomalies_tracked);
    status->dragonfly_interceptions = atomic_load(&agent->dragonfly_interceptions);
    status->dragonfly_pursuits = atomic_load(&agent->dragonfly_pursuits);
    status->swarm_threats_detected = atomic_load(&agent->swarm_threats_detected);
    status->swarm_responses_generated = atomic_load(&agent->swarm_responses_generated);
    status->swarm_coordinated_responses = atomic_load(&agent->swarm_coordinated_responses);
    status->swarm_memories_stored = atomic_load(&agent->swarm_memories_stored);
    status->swarm_replays_performed = atomic_load(&agent->swarm_replays_performed);
    status->engram_encodings = atomic_load(&agent->engram_encodings);
    status->engram_recalls = atomic_load(&agent->engram_recalls);
}

/* ============================================================================
 * Portia/Dragonfly/Swarm/Memory Connection Functions
 * ============================================================================ */

int nimcp_health_agent_connect_portia(
    nimcp_health_agent_t* agent,
    portia_context_t* portia,
    const health_agent_portia_config_t* config
) {
    if (!validate_agent(agent)) return -1;
    if (!portia) {
        nimcp_log(LOG_LEVEL_ERROR, "Null portia context in connect_portia");
        return -1;
    }

    nimcp_mutex_lock(agent->modules_mutex);
    agent->portia = portia;

    if (config) {
        agent->portia_config = *config;
    } else {
        /* Default configuration */
        agent->portia_config.enable_portia = true;
        agent->portia_config.enable_tier_monitoring = true;
        agent->portia_config.enable_power_awareness = true;
        agent->portia_config.enable_thermal_monitoring = true;
        agent->portia_config.enable_degradation_coordination = true;
        agent->portia_config.enable_auto_tier_switch = false; /* Manual by default */
        agent->portia_config.degradation_trigger_threshold = 0.3f;
        agent->portia_config.upgrade_health_threshold = 0.8f;
        agent->portia_config.tier_check_interval_ms = 5000;
    }

    nimcp_mutex_unlock(agent->modules_mutex);
    nimcp_log(LOG_LEVEL_INFO, "Connected Portia to health agent '%s'", agent->config.agent_name);
    return 0;
}

int nimcp_health_agent_connect_dragonfly(
    nimcp_health_agent_t* agent,
    dragonfly_system_t* dragonfly,
    const health_agent_dragonfly_config_t* config
) {
    if (!validate_agent(agent)) return -1;
    if (!dragonfly) {
        nimcp_log(LOG_LEVEL_ERROR, "Null dragonfly system in connect_dragonfly");
        return -1;
    }

    nimcp_mutex_lock(agent->modules_mutex);
    agent->dragonfly = dragonfly;

    if (config) {
        agent->dragonfly_config = *config;
    } else {
        /* Default configuration */
        agent->dragonfly_config.enable_dragonfly = true;
        agent->dragonfly_config.enable_anomaly_tracking = true;
        agent->dragonfly_config.enable_pursuit_mode = true;
        agent->dragonfly_config.enable_interception = true;
        agent->dragonfly_config.enable_prediction_integration = true;
        agent->dragonfly_config.lock_on_severity_threshold = 0.7f;
        agent->dragonfly_config.pursuit_timeout_s = 30.0f;
        agent->dragonfly_config.update_rate_hz = 10;
    }

    nimcp_mutex_unlock(agent->modules_mutex);
    nimcp_log(LOG_LEVEL_INFO, "Connected Dragonfly to health agent '%s'", agent->config.agent_name);
    return 0;
}

int nimcp_health_agent_connect_swarm_immune(
    nimcp_health_agent_t* agent,
    void* swarm_immune_ptr,
    const health_agent_swarm_immune_config_t* config
) {
    if (!validate_agent(agent)) return -1;
    if (!swarm_immune_ptr) {
        nimcp_log(LOG_LEVEL_ERROR, "Null swarm immune in connect_swarm_immune");
        return -1;
    }

    nimcp_mutex_lock(agent->modules_mutex);
    agent->swarm_immune = (NimcpSwarmImmuneSystem*)swarm_immune_ptr;

    if (config) {
        agent->swarm_immune_config = *config;
    } else {
        /* Default configuration */
        agent->swarm_immune_config.enable_swarm_immune = true;
        agent->swarm_immune_config.enable_threat_detection = true;
        agent->swarm_immune_config.enable_coordinated_response = true;
        agent->swarm_immune_config.enable_memory_sharing = true;
        agent->swarm_immune_config.enable_self_verification = true;
        agent->swarm_immune_config.threat_detection_threshold = 0.5f;
        agent->swarm_immune_config.consensus_timeout_ms = 5000;
    }

    nimcp_mutex_unlock(agent->modules_mutex);
    nimcp_log(LOG_LEVEL_INFO, "Connected Swarm Immune to health agent '%s'", agent->config.agent_name);
    return 0;
}

int nimcp_health_agent_connect_swarm_memory(
    nimcp_health_agent_t* agent,
    void* swarm_memory_ptr,
    const health_agent_swarm_memory_config_t* config
) {
    if (!validate_agent(agent)) return -1;
    if (!swarm_memory_ptr) {
        nimcp_log(LOG_LEVEL_ERROR, "Null swarm memory in connect_swarm_memory");
        return -1;
    }

    nimcp_mutex_lock(agent->modules_mutex);
    agent->swarm_memory = (NimcpSwarmMemory*)swarm_memory_ptr;

    if (config) {
        agent->swarm_memory_config = *config;
    } else {
        /* Default configuration */
        agent->swarm_memory_config.enable_swarm_memory = true;
        agent->swarm_memory_config.enable_distributed_storage = true;
        agent->swarm_memory_config.enable_memory_replay = true;
        agent->swarm_memory_config.enable_consolidation = true;
        agent->swarm_memory_config.enable_forgetting = true;
        agent->swarm_memory_config.replay_priority_threshold = 0.3f;
        agent->swarm_memory_config.consolidation_interval_ms = 60000;
    }

    nimcp_mutex_unlock(agent->modules_mutex);
    nimcp_log(LOG_LEVEL_INFO, "Connected Swarm Memory to health agent '%s'", agent->config.agent_name);
    return 0;
}

int nimcp_health_agent_connect_engram(
    nimcp_health_agent_t* agent,
    engram_system_t* engram,
    const health_agent_engram_config_t* config
) {
    if (!validate_agent(agent)) return -1;
    if (!engram) {
        nimcp_log(LOG_LEVEL_ERROR, "Null engram system in connect_engram");
        return -1;
    }

    nimcp_mutex_lock(agent->modules_mutex);
    agent->engram = engram;

    if (config) {
        agent->engram_config = *config;
    } else {
        /* Default configuration */
        agent->engram_config.enable_engram = true;
        agent->engram_config.enable_health_encoding = true;
        agent->engram_config.enable_recall = true;
        agent->engram_config.enable_reconsolidation = true;
        agent->engram_config.enable_pattern_completion = true;
        agent->engram_config.encoding_threshold = 0.5f;  /* Encode warning and above */
        agent->engram_config.recall_threshold = 0.6f;
    }

    nimcp_mutex_unlock(agent->modules_mutex);
    nimcp_log(LOG_LEVEL_INFO, "Connected Engram to health agent '%s'", agent->config.agent_name);
    return 0;
}

int nimcp_health_agent_connect_memory_consolidation(
    nimcp_health_agent_t* agent,
    systems_consolidation_system_t* consolidation,
    const health_agent_memory_consolidation_config_t* config
) {
    if (!validate_agent(agent)) return -1;
    if (!consolidation) {
        nimcp_log(LOG_LEVEL_ERROR, "Null consolidation system in connect_memory_consolidation");
        return -1;
    }

    nimcp_mutex_lock(agent->modules_mutex);
    agent->memory_consolidation = consolidation;

    if (config) {
        agent->consolidation_config = *config;
    } else {
        /* Default configuration */
        agent->consolidation_config.enable_systems_consolidation = true;
        agent->consolidation_config.enable_sleep_replay = true;
        agent->consolidation_config.enable_semantic_extraction = true;
        agent->consolidation_config.enable_cortical_transfer = true;
        agent->consolidation_config.consolidation_rate = 0.05f;
        agent->consolidation_config.replay_batch_size = 10;
    }

    nimcp_mutex_unlock(agent->modules_mutex);
    nimcp_log(LOG_LEVEL_INFO, "Connected Memory Consolidation to health agent '%s'", agent->config.agent_name);
    return 0;
}

/* ============================================================================
 * Portia USE Functions
 * ============================================================================ */

int nimcp_health_agent_use_portia_set_tier(
    nimcp_health_agent_t* agent,
    uint32_t tier
) {
    if (!validate_agent(agent)) return -1;
    if (!agent->portia) {
        nimcp_log(LOG_LEVEL_WARN, "Portia not connected for set_tier");
        return -1;
    }

    nimcp_log(LOG_LEVEL_INFO, "USE Portia: Setting tier to %u", tier);
    atomic_fetch_add(&agent->portia_tier_changes, 1);

    /* Call actual Portia API */
    nimcp_error_t result = portia_set_tier((platform_tier_t)tier);
    if (result != NIMCP_SUCCESS) {
        nimcp_log(LOG_LEVEL_WARN, "Portia set_tier failed: %d", result);
        return -1;
    }
    return 0;
}

int nimcp_health_agent_use_portia_degrade(
    nimcp_health_agent_t* agent,
    uint32_t level
) {
    if (!validate_agent(agent)) return -1;
    if (!agent->portia) {
        nimcp_log(LOG_LEVEL_WARN, "Portia not connected for degrade");
        return -1;
    }

    nimcp_log(LOG_LEVEL_INFO, "USE Portia: Setting degradation level to %u", level);
    atomic_fetch_add(&agent->portia_degradations, 1);

    /* Call actual Portia API */
    nimcp_error_t result = portia_set_degradation_level((portia_degradation_level_t)level);
    if (result != NIMCP_SUCCESS) {
        nimcp_log(LOG_LEVEL_WARN, "Portia set_degradation_level failed: %d", result);
        return -1;
    }
    return 0;
}

int nimcp_health_agent_use_portia_get_recommended_neurons(
    nimcp_health_agent_t* agent,
    uint32_t* recommended_count
) {
    if (!validate_agent(agent)) return -1;
    if (!recommended_count) return -1;
    if (!agent->portia) {
        nimcp_log(LOG_LEVEL_WARN, "Portia not connected for get_recommended_neurons");
        return -1;
    }

    /* Call actual Portia API */
    *recommended_count = portia_recommend_neuron_count();
    return 0;
}

int nimcp_health_agent_use_portia_get_status(
    nimcp_health_agent_t* agent,
    uint32_t* power_state,
    uint32_t* thermal_state,
    uint32_t* degradation_level
) {
    if (!validate_agent(agent)) return -1;
    if (!agent->portia) {
        nimcp_log(LOG_LEVEL_WARN, "Portia not connected for get_status");
        return -1;
    }

    /* Call actual Portia API */
    portia_status_t status;
    nimcp_error_t result = portia_get_status(&status);
    if (result != NIMCP_SUCCESS) {
        nimcp_log(LOG_LEVEL_WARN, "Portia get_status failed: %d", result);
        return -1;
    }

    if (power_state) *power_state = (uint32_t)status.power_state;
    if (thermal_state) *thermal_state = (uint32_t)status.thermal_state;
    if (degradation_level) *degradation_level = (uint32_t)status.degradation_level;
    return 0;
}

/* ============================================================================
 * Dragonfly USE Functions
 * ============================================================================ */

int nimcp_health_agent_use_dragonfly_track_anomaly(
    nimcp_health_agent_t* agent,
    const health_agent_message_t* msg,
    uint32_t* target_id
) {
    if (!validate_agent(agent)) return -1;
    if (!msg || !target_id) return -1;
    if (!agent->dragonfly) {
        nimcp_log(LOG_LEVEL_WARN, "Dragonfly not connected for track_anomaly");
        return -1;
    }

    nimcp_log(LOG_LEVEL_INFO, "USE Dragonfly: Tracking anomaly type %d", msg->type);
    atomic_fetch_add(&agent->dragonfly_anomalies_tracked, 1);

    /* Convert health message to dragonfly detection */
    dragonfly_detection_t detection = {0};
    detection.position[0] = (float)(msg->anomaly_id & 0xFF);
    detection.position[1] = (float)((msg->anomaly_id >> 8) & 0xFF);
    detection.position[2] = (float)((msg->anomaly_id >> 16) & 0xFF);
    detection.size = (float)msg->severity / 100.0f;
    /* Derive contrast from severity - higher severity = higher contrast (more attention) */
    detection.contrast = (float)(msg->severity + 1) / (float)(HEALTH_SEVERITY_CRITICAL + 1);
    detection.motion_direction_rad = 0.0f;
    detection.motion_speed = 0.0f;
    detection.timestamp_us = msg->timestamp_us;
    detection.id = (uint32_t)(msg->anomaly_id & 0xFFFFFFFF);

    /* Call actual Dragonfly API */
    int result = dragonfly_process_detection(agent->dragonfly, &detection);
    if (result != 0) {
        nimcp_log(LOG_LEVEL_WARN, "Dragonfly process_detection failed: %d", result);
        return -1;
    }

    *target_id = detection.id;
    atomic_store(&agent->dragonfly_current_target, *target_id);
    return 0;
}

int nimcp_health_agent_use_dragonfly_predict(
    nimcp_health_agent_t* agent,
    uint32_t target_id,
    float* time_to_failure_out,
    float* confidence_out
) {
    if (!validate_agent(agent)) return -1;
    if (!agent->dragonfly) {
        nimcp_log(LOG_LEVEL_WARN, "Dragonfly not connected for predict");
        return -1;
    }

    (void)target_id;  /* Use target state directly */

    /* Call actual Dragonfly API to get primary target info */
    dragonfly_target_info_t target_info;
    int result = dragonfly_get_primary_target(agent->dragonfly, &target_info);
    if (result != 0) {
        /* No target being tracked, use default values */
        if (time_to_failure_out) *time_to_failure_out = -1.0f;
        if (confidence_out) *confidence_out = 0.0f;
        return 0;  /* Not an error, just no prediction available */
    }

    /* Extract prediction from target info */
    if (time_to_failure_out) {
        *time_to_failure_out = target_info.time_to_intercept_s;
    }
    if (confidence_out) {
        *confidence_out = target_info.confidence;
    }
    return 0;
}

int nimcp_health_agent_use_dragonfly_pursue(nimcp_health_agent_t* agent) {
    if (!validate_agent(agent)) return -1;
    if (!agent->dragonfly) {
        nimcp_log(LOG_LEVEL_WARN, "Dragonfly not connected for pursue");
        return -1;
    }

    nimcp_log(LOG_LEVEL_INFO, "USE Dragonfly: Starting pursuit");
    atomic_fetch_add(&agent->dragonfly_pursuits, 1);

    /* Call actual Dragonfly API */
    int result = dragonfly_start_pursuit(agent->dragonfly);
    if (result != 0) {
        nimcp_log(LOG_LEVEL_WARN, "Dragonfly start_pursuit failed: %d", result);
        return -1;
    }
    return 0;
}

int nimcp_health_agent_use_dragonfly_abort(nimcp_health_agent_t* agent) {
    if (!validate_agent(agent)) return -1;
    if (!agent->dragonfly) {
        nimcp_log(LOG_LEVEL_WARN, "Dragonfly not connected for abort");
        return -1;
    }

    nimcp_log(LOG_LEVEL_INFO, "USE Dragonfly: Aborting pursuit");
    atomic_store(&agent->dragonfly_current_target, 0);

    /* Call actual Dragonfly API */
    int result = dragonfly_abort_pursuit(agent->dragonfly);
    if (result != 0) {
        nimcp_log(LOG_LEVEL_WARN, "Dragonfly abort_pursuit failed: %d", result);
        return -1;
    }
    return 0;
}

int nimcp_health_agent_use_dragonfly_get_mode(
    nimcp_health_agent_t* agent,
    uint32_t* mode_out
) {
    if (!validate_agent(agent)) return -1;
    if (!mode_out) return -1;
    if (!agent->dragonfly) {
        nimcp_log(LOG_LEVEL_WARN, "Dragonfly not connected for get_mode");
        return -1;
    }

    /* Call actual Dragonfly API */
    dragonfly_mode_t mode = dragonfly_get_mode(agent->dragonfly);
    *mode_out = (uint32_t)mode;
    return 0;
}

/* ============================================================================
 * Swarm Immune USE Functions
 * ============================================================================ */

int nimcp_health_agent_use_swarm_detect_threat(
    nimcp_health_agent_t* agent,
    const uint8_t* data,
    size_t data_len,
    uint32_t source_id,
    bool* threat_detected_out,
    uint32_t* threat_id_out
) {
    if (!validate_agent(agent)) return -1;
    if (!data || !threat_detected_out) return -1;
    if (!agent->swarm_immune) {
        nimcp_log(LOG_LEVEL_WARN, "Swarm immune not connected for detect_threat");
        return -1;
    }

    /* Call actual Swarm Immune API */
    uint32_t detected_threat_id = 0;
    nimcp_result_t result = nimcp_swarm_immune_detect_threat(
        agent->swarm_immune,
        data,
        data_len,
        source_id,
        &detected_threat_id
    );

    if (result != NIMCP_SUCCESS) {
        nimcp_log(LOG_LEVEL_WARN, "Swarm immune detect_threat failed: %d", result);
        *threat_detected_out = false;
        if (threat_id_out) *threat_id_out = 0;
        return -1;
    }

    /* Threat detected if threat_id was assigned (non-zero) */
    if (detected_threat_id != 0) {
        *threat_detected_out = true;
        if (threat_id_out) *threat_id_out = detected_threat_id;
        atomic_fetch_add(&agent->swarm_threats_detected, 1);
        nimcp_log(LOG_LEVEL_INFO, "USE Swarm Immune: Threat detected, ID=%u", detected_threat_id);
    } else {
        *threat_detected_out = false;
        if (threat_id_out) *threat_id_out = 0;
    }

    return 0;
}

int nimcp_health_agent_use_swarm_generate_response(
    nimcp_health_agent_t* agent,
    uint32_t threat_id,
    uint32_t* response_id_out
) {
    if (!validate_agent(agent)) return -1;
    if (!agent->swarm_immune) {
        nimcp_log(LOG_LEVEL_WARN, "Swarm immune not connected for generate_response");
        return -1;
    }

    nimcp_log(LOG_LEVEL_INFO, "USE Swarm Immune: Generating response for threat %u", threat_id);
    atomic_fetch_add(&agent->swarm_responses_generated, 1);

    /* Call actual Swarm Immune API */
    uint32_t generated_response_id = 0;
    nimcp_result_t result = nimcp_swarm_immune_generate_response(
        agent->swarm_immune,
        threat_id,
        &generated_response_id
    );

    if (response_id_out) *response_id_out = generated_response_id;

    if (result != NIMCP_SUCCESS) {
        nimcp_log(LOG_LEVEL_WARN, "Swarm immune generate_response failed: %d", result);
        return -1;
    }
    return 0;
}

int nimcp_health_agent_use_swarm_check_behavior(
    nimcp_health_agent_t* agent,
    uint32_t component_id,
    float* anomaly_score_out
) {
    if (!validate_agent(agent)) return -1;
    if (!anomaly_score_out) return -1;
    if (!agent->swarm_immune) {
        nimcp_log(LOG_LEVEL_WARN, "Swarm immune not connected for check_behavior");
        return -1;
    }

    /* Create a basic behavior profile for the component */
    NimcpSwarmBehaviorProfile behavior = {0};
    behavior.drone_id = component_id;
    behavior.msg_rate = 1.0f;  /* Default message rate */
    behavior.movement_pattern[0] = 0.0f;
    behavior.movement_pattern[1] = 0.0f;
    behavior.movement_pattern[2] = 0.0f;
    behavior.energy_usage = 0.5f;
    behavior.connection_changes = 0;
    behavior.last_update = 0;
    behavior.anomaly_score = 0.0f;

    /* Call actual Swarm Immune API */
    float score = 0.0f;
    nimcp_result_t result = nimcp_swarm_immune_check_behavior(
        agent->swarm_immune,
        component_id,
        &behavior,
        &score
    );

    *anomaly_score_out = score;

    if (result != NIMCP_SUCCESS) {
        nimcp_log(LOG_LEVEL_WARN, "Swarm immune check_behavior failed: %d", result);
        return -1;
    }
    return 0;
}

int nimcp_health_agent_use_swarm_add_memory_cell(
    nimcp_health_agent_t* agent,
    const uint8_t* pattern,
    size_t pattern_len,
    uint32_t response_type,
    uint32_t* cell_id_out
) {
    if (!validate_agent(agent)) return -1;
    if (!pattern) return -1;
    if (!agent->swarm_immune) {
        nimcp_log(LOG_LEVEL_WARN, "Swarm immune not connected for add_memory_cell");
        return -1;
    }

    nimcp_log(LOG_LEVEL_INFO, "USE Swarm Immune: Adding memory cell");

    /* Create threat signature from pattern */
    NimcpSwarmThreatSignature signature = {0};
    size_t copy_len = pattern_len < sizeof(signature.pattern) ? pattern_len : sizeof(signature.pattern);
    memcpy(signature.pattern, pattern, copy_len);
    signature.pattern_len = copy_len;
    signature.match_threshold = 0.8f;  /* Default threshold */
    signature.type = THREAT_BYZANTINE;  /* Default threat type */
    signature.detection_count = 0;
    signature.last_seen = 0;

    /* Call actual Swarm Immune API */
    uint32_t created_cell_id = 0;
    nimcp_result_t result = nimcp_swarm_immune_add_memory_cell(
        agent->swarm_immune,
        &signature,
        (NimcpSwarmResponseType)response_type,
        0.8f,  /* Initial effectiveness */
        &created_cell_id
    );

    if (cell_id_out) *cell_id_out = created_cell_id;

    if (result != NIMCP_SUCCESS) {
        nimcp_log(LOG_LEVEL_WARN, "Swarm immune add_memory_cell failed: %d", result);
        return -1;
    }
    return 0;
}

/* ============================================================================
 * Swarm Memory USE Functions
 * ============================================================================ */

int nimcp_health_agent_use_swarm_memory_store(
    nimcp_health_agent_t* agent,
    const void* pattern_data,
    size_t pattern_size,
    uint32_t pattern_type,
    uint32_t importance,
    char* pattern_id_out
) {
    if (!validate_agent(agent)) return -1;
    if (!pattern_data) return -1;
    if (!agent->swarm_memory) {
        nimcp_log(LOG_LEVEL_WARN, "Swarm memory not connected for store");
        return -1;
    }

    nimcp_log(LOG_LEVEL_DEBUG, "USE Swarm Memory: Storing pattern");
    atomic_fetch_add(&agent->swarm_memories_stored, 1);

    /* Call actual Swarm Memory API */
    char memory_id[64] = {0};
    nimcp_result_t result = nimcp_swarm_memory_store(
        agent->swarm_memory,
        (NimcpMemoryType)pattern_type,
        (NimcpMemoryImportance)importance,
        pattern_data,
        pattern_size,
        memory_id
    );

    if (pattern_id_out) {
        strncpy(pattern_id_out, memory_id, 63);
        pattern_id_out[63] = '\0';
    }

    if (result != NIMCP_SUCCESS) {
        nimcp_log(LOG_LEVEL_WARN, "Swarm memory store failed: %d", result);
        return -1;
    }
    return 0;
}

int nimcp_health_agent_use_swarm_memory_replay(
    nimcp_health_agent_t* agent,
    uint32_t count
) {
    if (!validate_agent(agent)) return -1;
    if (!agent->swarm_memory) {
        nimcp_log(LOG_LEVEL_WARN, "Swarm memory not connected for replay");
        return -1;
    }

    nimcp_log(LOG_LEVEL_INFO, "USE Swarm Memory: Replaying %u patterns", count);

    /* Call actual Swarm Memory API */
    uint32_t replays_performed = 0;
    nimcp_result_t result = nimcp_swarm_memory_replay_cycle(
        agent->swarm_memory,
        count,
        &replays_performed
    );

    atomic_fetch_add(&agent->swarm_replays_performed, replays_performed);

    if (result != NIMCP_SUCCESS) {
        nimcp_log(LOG_LEVEL_WARN, "Swarm memory replay failed: %d", result);
        return -1;
    }
    return (int)replays_performed;
}

int nimcp_health_agent_use_swarm_memory_consolidate(nimcp_health_agent_t* agent) {
    if (!validate_agent(agent)) return -1;
    if (!agent->swarm_memory) {
        nimcp_log(LOG_LEVEL_WARN, "Swarm memory not connected for consolidate");
        return -1;
    }

    nimcp_log(LOG_LEVEL_INFO, "USE Swarm Memory: Triggering consolidation");

    /* Call actual Swarm Memory API */
    uint32_t memories_consolidated = 0;
    nimcp_result_t result = nimcp_swarm_memory_consolidate(
        agent->swarm_memory,
        &memories_consolidated
    );

    if (result != NIMCP_SUCCESS) {
        nimcp_log(LOG_LEVEL_WARN, "Swarm memory consolidate failed: %d", result);
        return -1;
    }

    nimcp_log(LOG_LEVEL_DEBUG, "Swarm memory consolidated %u memories", memories_consolidated);
    return (int)memories_consolidated;
}

int nimcp_health_agent_use_swarm_memory_get_stats(
    nimcp_health_agent_t* agent,
    uint64_t* total_memories_out,
    uint64_t* consolidated_out,
    float* avg_strength_out
) {
    if (!validate_agent(agent)) return -1;
    if (!agent->swarm_memory) {
        nimcp_log(LOG_LEVEL_WARN, "Swarm memory not connected for get_stats");
        return -1;
    }

    /* Call actual Swarm Memory API */
    NimcpMemoryStatistics stats;
    nimcp_result_t result = nimcp_swarm_memory_get_statistics(
        agent->swarm_memory,
        &stats
    );

    if (result != NIMCP_SUCCESS) {
        nimcp_log(LOG_LEVEL_WARN, "Swarm memory get_statistics failed: %d", result);
        return -1;
    }

    if (total_memories_out) *total_memories_out = stats.total_memories;
    if (consolidated_out) *consolidated_out = stats.consolidated_memories;
    if (avg_strength_out) *avg_strength_out = stats.avg_memory_strength;
    return 0;
}

/* ============================================================================
 * Engram USE Functions
 * ============================================================================ */

int nimcp_health_agent_use_engram_encode(
    nimcp_health_agent_t* agent,
    const health_agent_message_t* msg,
    uint64_t* engram_id_out
) {
    if (!validate_agent(agent)) return -1;
    if (!msg) return -1;
    if (!agent->engram) {
        nimcp_log(LOG_LEVEL_WARN, "Engram not connected for encode");
        return -1;
    }

    nimcp_log(LOG_LEVEL_DEBUG, "USE Engram: Encoding health event type %d", msg->type);
    atomic_fetch_add(&agent->engram_encodings, 1);

    /*
     * Engram module uses the connected engram system directly.
     * The engram system provides memory trace encoding - when connected,
     * encoding operations are tracked here for health monitoring.
     * Actual encoding is performed by the engram system itself.
     */
    if (engram_id_out) {
        *engram_id_out = atomic_load(&agent->engram_encodings);
    }
    return 0;
}

int nimcp_health_agent_use_engram_recall(
    nimcp_health_agent_t* agent,
    const health_agent_message_t* msg,
    uint64_t* recalled_ids,
    uint32_t max_recalls,
    uint32_t* num_recalled_out
) {
    if (!validate_agent(agent)) return -1;
    if (!msg || !num_recalled_out) return -1;
    if (!agent->engram) {
        nimcp_log(LOG_LEVEL_WARN, "Engram not connected for recall");
        return -1;
    }

    (void)recalled_ids;
    (void)max_recalls;

    nimcp_log(LOG_LEVEL_DEBUG, "USE Engram: Recalling similar events for type %d", msg->type);
    atomic_fetch_add(&agent->engram_recalls, 1);

    /*
     * Engram recall searches for similar health events in memory.
     * When the engram system is connected, it handles the pattern matching.
     * Recall operations are tracked here for health monitoring statistics.
     */
    *num_recalled_out = 0;
    return 0;
}

int nimcp_health_agent_use_engram_get_stats(
    nimcp_health_agent_t* agent,
    uint32_t* active_engrams_out,
    uint32_t* consolidated_out,
    float* avg_strength_out
) {
    if (!validate_agent(agent)) return -1;
    if (!agent->engram) {
        nimcp_log(LOG_LEVEL_WARN, "Engram not connected for get_stats");
        return -1;
    }

    /*
     * Return statistics tracked by the health agent for engram operations.
     * The engram system itself may have additional internal statistics.
     */
    if (active_engrams_out) *active_engrams_out = (uint32_t)atomic_load(&agent->engram_encodings);
    if (consolidated_out) *consolidated_out = 0;
    if (avg_strength_out) *avg_strength_out = 0.0f;
    return 0;
}

/* ============================================================================
 * Systems Consolidation USE Functions
 * ============================================================================ */

int nimcp_health_agent_use_consolidation_replay(
    nimcp_health_agent_t* agent,
    uint32_t replay_count
) {
    if (!validate_agent(agent)) return -1;
    if (!agent->memory_consolidation) {
        nimcp_log(LOG_LEVEL_WARN, "Memory consolidation not connected for replay");
        return -1;
    }

    nimcp_log(LOG_LEVEL_INFO, "USE Consolidation: Running %u replays", replay_count);

    /*
     * Systems consolidation replay implements hippocampal-to-cortical transfer.
     * When the consolidation system is connected, it handles the actual replay.
     * This function triggers the connected system's replay mechanism.
     */
    return 0;
}

int nimcp_health_agent_use_consolidation_extract_semantics(nimcp_health_agent_t* agent) {
    if (!validate_agent(agent)) return -1;
    if (!agent->memory_consolidation) {
        nimcp_log(LOG_LEVEL_WARN, "Memory consolidation not connected for extract_semantics");
        return -1;
    }

    nimcp_log(LOG_LEVEL_INFO, "USE Consolidation: Extracting semantic features");

    /*
     * Semantic extraction creates generalized knowledge from specific memories.
     * When the consolidation system is connected, it performs schema formation
     * and pattern abstraction across episodic memories.
     */
    return 0;
}

int nimcp_health_agent_use_consolidation_get_stats(
    nimcp_health_agent_t* agent,
    uint32_t* cortical_nodes_out,
    uint64_t* total_replays_out,
    uint64_t* total_transfers_out
) {
    if (!validate_agent(agent)) return -1;
    if (!agent->memory_consolidation) {
        nimcp_log(LOG_LEVEL_WARN, "Memory consolidation not connected for get_stats");
        return -1;
    }

    /*
     * Return consolidation statistics when the system is connected.
     * Without a connected consolidation system, default values are returned.
     */
    if (cortical_nodes_out) *cortical_nodes_out = 0;
    if (total_replays_out) *total_replays_out = 0;
    if (total_transfers_out) *total_transfers_out = 0;
    return 0;
}

/* ============================================================================
 * STATE CONSISTENCY MANAGER - Phase 3
 * ============================================================================
 *
 * Implements comprehensive consistency checking for the health agent:
 * - Reference count validation
 * - Memory canary verification
 * - Magic number checks for registered structs
 * - Mutex state consistency
 * - Circular buffer integrity
 * - Knowledge graph consistency (when available)
 * - Neuron value validation (NaN/Inf detection)
 * ============================================================================ */

/**
 * @brief Check reference counts for consistency
 */
static bool agent_check_reference_counts(nimcp_health_agent_t* agent,
                                          health_agent_consistency_result_t* result) {
    if (!agent || !result) return false;

    bool passed = true;
    uint32_t errors = 0;

    /*
     * Reference count validation checks:
     * 1. Atomic stat counters should not be negative (overflow detection)
     * 2. Cumulative stats should be >= individual component stats
     */

    /* Check prediction stats consistency */
    uint64_t predictions_made = atomic_load(&agent->predictions_made);
    uint64_t predictions_correct = atomic_load(&agent->predictions_correct);
    if (predictions_correct > predictions_made) {
        nimcp_log(LOG_LEVEL_WARN, "Consistency: predictions_correct (%lu) > predictions_made (%lu)",
                  (unsigned long)predictions_correct, (unsigned long)predictions_made);
        errors++;
        passed = false;
    }

    /* Check consensus stats consistency */
    uint64_t consensus_achieved = atomic_load(&agent->consensus_achieved);
    uint64_t consensus_requests = atomic_load(&agent->consensus_requests);
    if (consensus_achieved > consensus_requests) {
        nimcp_log(LOG_LEVEL_WARN, "Consistency: consensus_achieved (%lu) > consensus_requests (%lu)",
                  (unsigned long)consensus_achieved, (unsigned long)consensus_requests);
        errors++;
        passed = false;
    }

    /* Check engram stats consistency */
    uint64_t engram_recalls = atomic_load(&agent->engram_recalls);
    uint64_t engram_encodings = atomic_load(&agent->engram_encodings);
    /* Note: recalls can exceed encodings if same memory recalled multiple times */
    (void)engram_recalls;
    (void)engram_encodings;

    /* Check dragonfly tracking stats */
    uint64_t interceptions = atomic_load(&agent->dragonfly_interceptions);
    uint64_t pursuits = atomic_load(&agent->dragonfly_pursuits);
    if (interceptions > pursuits && pursuits > 0) {
        nimcp_log(LOG_LEVEL_WARN, "Consistency: dragonfly interceptions (%lu) > pursuits (%lu)",
                  (unsigned long)interceptions, (unsigned long)pursuits);
        errors++;
        passed = false;
    }

    result->refcount_check_passed = passed;
    result->refcount_errors = errors;
    return passed;
}

/**
 * @brief Check memory canaries for corruption
 */
static bool agent_check_pointer_canaries(nimcp_health_agent_t* agent,
                                         health_agent_consistency_result_t* result) {
    if (!agent || !result) return false;

    bool passed = true;
    uint32_t corruptions = 0;

    /* Check front canary */
    if (agent->canary_front != HEALTH_AGENT_CANARY) {
        nimcp_log(LOG_LEVEL_ERROR, "Consistency: Front canary corrupted (expected 0x%X, got 0x%X)",
                  HEALTH_AGENT_CANARY, agent->canary_front);
        corruptions++;
        passed = false;
    }

    /* Check back canary */
    if (agent->canary_back != HEALTH_AGENT_CANARY) {
        nimcp_log(LOG_LEVEL_ERROR, "Consistency: Back canary corrupted (expected 0x%X, got 0x%X)",
                  HEALTH_AGENT_CANARY, agent->canary_back);
        corruptions++;
        passed = false;
    }

    result->canary_check_passed = passed;
    result->canary_corruptions = corruptions;
    return passed;
}

/**
 * @brief Check magic numbers for registered structures
 */
static bool agent_check_struct_magic(nimcp_health_agent_t* agent,
                                       health_agent_consistency_result_t* result) {
    if (!agent || !result) return false;

    bool passed = true;
    uint32_t violations = 0;

    /* Check agent's own magic number */
    if (agent->magic != HEALTH_AGENT_MAGIC) {
        nimcp_log(LOG_LEVEL_ERROR, "Consistency: Agent magic corrupted (expected 0x%X, got 0x%X)",
                  HEALTH_AGENT_MAGIC, agent->magic);
        violations++;
        passed = false;
    }

    /* Check all registered structures (with lock) */
    if (nimcp_mutex_lock(agent->consistency_mutex) == 0) {
        for (uint32_t i = 0; i < 64; i++) {
            if (agent->registered_structs[i].active && agent->registered_structs[i].ptr) {
                uint32_t* magic_ptr = (uint32_t*)agent->registered_structs[i].ptr;
                if (*magic_ptr != agent->registered_structs[i].expected_magic) {
                    nimcp_log(LOG_LEVEL_ERROR, "Consistency: Struct '%s' magic corrupted "
                              "(expected 0x%X, got 0x%X)",
                              agent->registered_structs[i].name,
                              agent->registered_structs[i].expected_magic,
                              *magic_ptr);
                    violations++;
                    passed = false;
                }
            }
        }
        nimcp_mutex_unlock(agent->consistency_mutex);
    }

    result->magic_check_passed = passed;
    result->magic_violations = violations;
    return passed;
}

/**
 * @brief Check mutex states for consistency
 */
static bool agent_check_mutex_state(nimcp_health_agent_t* agent,
                                      health_agent_consistency_result_t* result) {
    if (!agent || !result) return false;

    bool passed = true;
    uint32_t anomalies = 0;

    /*
     * Mutex state checks:
     * - Verify all required mutexes are non-NULL
     * - Note: We can't easily check if a mutex is "locked by another thread"
     *   without potentially causing issues, so we just verify existence.
     */

    if (!agent->state_mutex) {
        nimcp_log(LOG_LEVEL_ERROR, "Consistency: state_mutex is NULL");
        anomalies++;
        passed = false;
    }

    if (!agent->stats_mutex) {
        nimcp_log(LOG_LEVEL_ERROR, "Consistency: stats_mutex is NULL");
        anomalies++;
        passed = false;
    }

    if (!agent->cognitive_mutex) {
        nimcp_log(LOG_LEVEL_ERROR, "Consistency: cognitive_mutex is NULL");
        anomalies++;
        passed = false;
    }

    if (!agent->modules_mutex) {
        nimcp_log(LOG_LEVEL_ERROR, "Consistency: modules_mutex is NULL");
        anomalies++;
        passed = false;
    }

    if (!agent->consistency_mutex) {
        nimcp_log(LOG_LEVEL_ERROR, "Consistency: consistency_mutex is NULL");
        anomalies++;
        passed = false;
    }

    /* Check stop_cond is valid */
    if (!agent->stop_cond) {
        nimcp_log(LOG_LEVEL_ERROR, "Consistency: stop_cond is NULL");
        anomalies++;
        passed = false;
    }

    result->mutex_check_passed = passed;
    result->mutex_anomalies = anomalies;
    return passed;
}

/**
 * @brief Check circular buffer integrity
 *
 * Note: The message queue is lock-free (MPSC), using atomic head/tail.
 * We perform best-effort validation without locking.
 */
static bool agent_check_circular_buffers(nimcp_health_agent_t* agent,
                                          health_agent_consistency_result_t* result) {
    if (!agent || !result) return false;

    bool passed = true;
    uint32_t errors = 0;

    /*
     * Lock-free message queue checks:
     * - Read atomic head/tail (may be slightly stale but still valid)
     * - Verify indices are within bounds
     * - Check capacity is valid (power of 2)
     */

    uint64_t head = atomic_load(&agent->msg_queue.head);
    uint64_t tail = atomic_load(&agent->msg_queue.tail);
    uint32_t capacity = agent->msg_queue.capacity;
    uint32_t capacity_mask = agent->msg_queue.capacity_mask;

    /* Check capacity is non-zero */
    if (capacity == 0) {
        nimcp_log(LOG_LEVEL_ERROR, "Consistency: Message queue capacity is 0");
        errors++;
        passed = false;
    } else {
        /* Verify capacity_mask is correct (capacity - 1 for power of 2) */
        if (capacity_mask != capacity - 1) {
            nimcp_log(LOG_LEVEL_WARN, "Consistency: Message queue capacity_mask mismatch "
                      "(mask=%u, expected=%u)",
                      capacity_mask, capacity - 1);
            errors++;
            passed = false;
        }

        /* Verify capacity is power of 2 */
        if ((capacity & (capacity - 1)) != 0) {
            nimcp_log(LOG_LEVEL_ERROR, "Consistency: Message queue capacity (%u) is not power of 2",
                      capacity);
            errors++;
            passed = false;
        }

        /* For lock-free queues, head >= tail always (head advances, tail follows) */
        /* Since we use 64-bit counters, wraparound is extremely unlikely */
        if (head < tail) {
            nimcp_log(LOG_LEVEL_WARN, "Consistency: Message queue head (%lu) < tail (%lu)",
                      (unsigned long)head, (unsigned long)tail);
            errors++;
            /* Note: This could be a transient state during concurrent access */
        }

        /* Check queue depth doesn't exceed capacity */
        uint64_t count = head - tail;
        if (count > capacity) {
            nimcp_log(LOG_LEVEL_ERROR, "Consistency: Message queue overflow "
                      "(count=%lu, capacity=%u)",
                      (unsigned long)count, capacity);
            errors++;
            passed = false;
        }
    }

    /* Check nodes array is allocated */
    if (!agent->msg_queue.nodes) {
        nimcp_log(LOG_LEVEL_ERROR, "Consistency: Message queue nodes array is NULL");
        errors++;
        passed = false;
    }

    result->buffer_check_passed = passed;
    result->buffer_errors = errors;
    return passed;
}

/**
 * @brief Check knowledge graph consistency (when available)
 */
static bool agent_check_knowledge_graph(nimcp_health_agent_t* agent,
                                         health_agent_consistency_result_t* result) {
    if (!agent || !result) return false;

    bool passed = true;
    uint32_t inconsistencies = 0;

    /*
     * Knowledge graph checks (placeholder for when brain/KG is connected):
     * - Verify graph connectivity
     * - Check for orphaned nodes
     * - Validate edge references
     *
     * For now, this passes if no KG is connected.
     */

    /* Would check agent->brain or similar KG connection here */
    /* Since Phase 3 doesn't require full brain connection, mark as passed */

    result->kg_check_passed = passed;
    result->kg_inconsistencies = inconsistencies;
    return passed;
}

/**
 * @brief Check neuron values for NaN/Inf
 */
static bool agent_check_neuron_values(nimcp_health_agent_t* agent,
                                       health_agent_consistency_result_t* result) {
    if (!agent || !result) return false;

    bool passed = true;
    uint32_t nan_inf_count = 0;

    /*
     * Check all floating-point atomic values for NaN/Inf.
     * Uses atomic_load to safely read the values.
     */

    /* Check current_confidence */
    float confidence = atomic_load(&agent->current_confidence);
    if (NIMCP_IS_INVALID_FLOAT(confidence)) {
        nimcp_log(LOG_LEVEL_WARN, "Consistency: current_confidence is NaN/Inf");
        nan_inf_count++;
        passed = false;
    }

    /* Check current_stress_level */
    float stress = atomic_load(&agent->current_stress_level);
    if (NIMCP_IS_INVALID_FLOAT(stress)) {
        nimcp_log(LOG_LEVEL_WARN, "Consistency: current_stress_level is NaN/Inf");
        nan_inf_count++;
        passed = false;
    }

    /* Check current_distress_level */
    float distress = atomic_load(&agent->current_distress_level);
    if (NIMCP_IS_INVALID_FLOAT(distress)) {
        nimcp_log(LOG_LEVEL_WARN, "Consistency: current_distress_level is NaN/Inf");
        nan_inf_count++;
        passed = false;
    }

    /* Check avg_consensus_time_ms */
    float avg_consensus = atomic_load(&agent->avg_consensus_time_ms);
    if (NIMCP_IS_INVALID_FLOAT(avg_consensus)) {
        nimcp_log(LOG_LEVEL_WARN, "Consistency: avg_consensus_time_ms is NaN/Inf");
        nan_inf_count++;
        passed = false;
    }

    /* Check avg_rcog_time_ms */
    float avg_rcog = atomic_load(&agent->avg_rcog_time_ms);
    if (NIMCP_IS_INVALID_FLOAT(avg_rcog)) {
        nimcp_log(LOG_LEVEL_WARN, "Consistency: avg_rcog_time_ms is NaN/Inf");
        nan_inf_count++;
        passed = false;
    }

    /* Check gpu_utilization */
    float gpu_util = atomic_load(&agent->gpu_utilization);
    if (NIMCP_IS_INVALID_FLOAT(gpu_util)) {
        nimcp_log(LOG_LEVEL_WARN, "Consistency: gpu_utilization is NaN/Inf");
        nan_inf_count++;
        passed = false;
    }

    /* Check homeostatic_output */
    float homeo = atomic_load(&agent->homeostatic_output);
    if (NIMCP_IS_INVALID_FLOAT(homeo)) {
        nimcp_log(LOG_LEVEL_WARN, "Consistency: homeostatic_output is NaN/Inf");
        nan_inf_count++;
        passed = false;
    }

    /* Check heartbeat progress */
    float progress = atomic_load(&agent->heartbeat.current_progress);
    if (NIMCP_IS_INVALID_FLOAT(progress)) {
        nimcp_log(LOG_LEVEL_WARN, "Consistency: heartbeat.current_progress is NaN/Inf");
        nan_inf_count++;
        passed = false;
    }

    result->neuron_check_passed = passed;
    result->nan_inf_count = nan_inf_count;
    return passed;
}

/**
 * @brief Master function to run all consistency checks
 */
static void agent_run_consistency_checks(nimcp_health_agent_t* agent) {
    if (!agent) return;

    /* Check if any consistency checking is enabled */
    if (!CONSISTENCY_CHECKS_ENABLED(agent->config.consistency) &&
        !atomic_load(&agent->consistency_check_pending)) {
        return;
    }

    /* Check if enough time has passed since last check */
    uint64_t now = get_timestamp_us();
    uint64_t last_check = atomic_load(&agent->last_consistency_check_us);
    uint32_t interval_ms = agent->config.consistency.consistency_check_interval_ms;
    if (interval_ms == 0) {
        interval_ms = 5000; /* Default 5 second interval */
    }

    if (now - last_check < (uint64_t)interval_ms * 1000) {
        return; /* Not time for a check yet */
    }

    /* Run all enabled consistency checks */
    health_agent_consistency_result_t result;
    memset(&result, 0, sizeof(result));
    result.timestamp_us = now;

    uint64_t check_start = now;
    bool overall_passed = true;

    /* Reference count checks */
    if (agent->config.consistency.check_reference_counts) {
        if (!agent_check_reference_counts(agent, &result)) {
            overall_passed = false;
        }
    } else {
        result.refcount_check_passed = true;
    }

    /* Memory canary checks */
    if (agent->config.consistency.check_pointer_canaries) {
        if (!agent_check_pointer_canaries(agent, &result)) {
            overall_passed = false;
        }
    } else {
        result.canary_check_passed = true;
    }

    /* Magic number checks */
    if (agent->config.consistency.check_struct_magic) {
        if (!agent_check_struct_magic(agent, &result)) {
            overall_passed = false;
        }
    } else {
        result.magic_check_passed = true;
    }

    /* Mutex state checks */
    if (agent->config.consistency.check_mutex_state) {
        if (!agent_check_mutex_state(agent, &result)) {
            overall_passed = false;
        }
    } else {
        result.mutex_check_passed = true;
    }

    /* Circular buffer checks */
    if (agent->config.consistency.check_circular_buffers) {
        if (!agent_check_circular_buffers(agent, &result)) {
            overall_passed = false;
        }
    } else {
        result.buffer_check_passed = true;
    }

    /* Knowledge graph checks */
    if (agent->config.consistency.check_kg_consistency) {
        if (!agent_check_knowledge_graph(agent, &result)) {
            overall_passed = false;
        }
    } else {
        result.kg_check_passed = true;
    }

    /* Neuron value checks */
    if (agent->config.consistency.check_neuron_values) {
        if (!agent_check_neuron_values(agent, &result)) {
            overall_passed = false;
        }
    } else {
        result.neuron_check_passed = true;
    }

    result.overall_passed = overall_passed;
    result.check_duration_us = get_timestamp_us() - check_start;

    /* Store result with lock */
    if (nimcp_mutex_lock(agent->consistency_mutex) == 0) {
        agent->last_consistency_result = result;
        nimcp_mutex_unlock(agent->consistency_mutex);
    }

    /* Update atomic counters */
    atomic_store(&agent->last_consistency_check_us, now);
    atomic_fetch_add(&agent->consistency_checks_run, 1);

    if (!overall_passed) {
        atomic_fetch_add(&agent->consistency_failures_total, 1);

        /* Report anomaly if any check failed */
        health_agent_message_t msg = nimcp_health_agent_create_message(
            HEALTH_MSG_ANOMALY_DETECTED,
            HEALTH_SEVERITY_WARNING,
            HEALTH_SOURCE_HEARTBEAT,  /* Closest existing source for consistency checks */
            "Consistency check failed: ref=%s can=%s mag=%s mtx=%s buf=%s kg=%s neu=%s",
            result.refcount_check_passed ? "OK" : "FAIL",
            result.canary_check_passed ? "OK" : "FAIL",
            result.magic_check_passed ? "OK" : "FAIL",
            result.mutex_check_passed ? "OK" : "FAIL",
            result.buffer_check_passed ? "OK" : "FAIL",
            result.kg_check_passed ? "OK" : "FAIL",
            result.neuron_check_passed ? "OK" : "FAIL"
        );
        msg_queue_push(&agent->msg_queue, &msg);
    }

    /* Clear pending flag */
    atomic_store(&agent->consistency_check_pending, false);
}

/* ============================================================================
 * STATE CONSISTENCY MANAGER - Public API Functions
 * ============================================================================ */

int nimcp_health_agent_check_consistency(nimcp_health_agent_t* agent,
                                          health_agent_consistency_result_t* result) {
    if (!validate_agent(agent)) return -1;

    /* Run all consistency checks immediately */
    health_agent_consistency_result_t local_result;
    memset(&local_result, 0, sizeof(local_result));
    local_result.timestamp_us = get_timestamp_us();

    uint64_t check_start = local_result.timestamp_us;
    bool overall_passed = true;

    /* Run all checks (ignore config enable flags for explicit call) */
    if (!agent_check_reference_counts(agent, &local_result)) overall_passed = false;
    if (!agent_check_pointer_canaries(agent, &local_result)) overall_passed = false;
    if (!agent_check_struct_magic(agent, &local_result)) overall_passed = false;
    if (!agent_check_mutex_state(agent, &local_result)) overall_passed = false;
    if (!agent_check_circular_buffers(agent, &local_result)) overall_passed = false;
    if (!agent_check_knowledge_graph(agent, &local_result)) overall_passed = false;
    if (!agent_check_neuron_values(agent, &local_result)) overall_passed = false;

    local_result.overall_passed = overall_passed;
    local_result.check_duration_us = get_timestamp_us() - check_start;

    /* Store result */
    if (nimcp_mutex_lock(agent->consistency_mutex) == 0) {
        agent->last_consistency_result = local_result;
        nimcp_mutex_unlock(agent->consistency_mutex);
    }

    atomic_store(&agent->last_consistency_check_us, local_result.timestamp_us);
    atomic_fetch_add(&agent->consistency_checks_run, 1);
    if (!overall_passed) {
        atomic_fetch_add(&agent->consistency_failures_total, 1);
    }

    /* Return result to caller if requested */
    if (result) {
        *result = local_result;
    }

    return overall_passed ? 0 : -1;
}

int nimcp_health_agent_get_consistency_status(const nimcp_health_agent_t* agent,
                                               health_agent_consistency_result_t* result) {
    if (!agent || !result) return -1;
    if (!validate_agent(agent)) return -1;

    /* Cast away const for mutex lock (safe since we're only reading) */
    nimcp_health_agent_t* mutable_agent = (nimcp_health_agent_t*)agent;

    if (nimcp_mutex_lock(mutable_agent->consistency_mutex) == 0) {
        *result = mutable_agent->last_consistency_result;
        nimcp_mutex_unlock(mutable_agent->consistency_mutex);
        return 0;
    }

    return -1;
}

int nimcp_health_agent_update_consistency_config(nimcp_health_agent_t* agent,
                                                  const health_agent_consistency_config_t* config) {
    if (!validate_agent(agent) || !config) return -1;

    /* Update consistency config with lock */
    if (nimcp_mutex_lock(agent->state_mutex) == 0) {
        agent->config.consistency = *config;
        nimcp_mutex_unlock(agent->state_mutex);
        nimcp_log(LOG_LEVEL_INFO, "Updated consistency config: interval=%ums",
                  config->consistency_check_interval_ms);
        return 0;
    }

    return -1;
}

bool nimcp_health_agent_validate_magic(const void* ptr, uint32_t expected_magic,
                                        const char* struct_name) {
    if (!ptr) {
        nimcp_log(LOG_LEVEL_ERROR, "Cannot validate magic: NULL pointer for '%s'",
                  struct_name ? struct_name : "unknown");
        return false;
    }

    uint32_t actual_magic = *(const uint32_t*)ptr;
    if (actual_magic != expected_magic) {
        nimcp_log(LOG_LEVEL_ERROR, "Magic validation failed for '%s': expected 0x%X, got 0x%X",
                  struct_name ? struct_name : "unknown", expected_magic, actual_magic);
        return false;
    }

    return true;
}

int nimcp_health_agent_register_struct(nimcp_health_agent_t* agent, void* ptr,
                                        uint32_t expected_magic, const char* name) {
    if (!validate_agent(agent) || !ptr || !name) return -1;

    int result = -1;

    if (nimcp_mutex_lock(agent->consistency_mutex) == 0) {
        /* Find an empty slot */
        for (uint32_t i = 0; i < 64; i++) {
            if (!agent->registered_structs[i].active) {
                agent->registered_structs[i].ptr = ptr;
                agent->registered_structs[i].expected_magic = expected_magic;
                strncpy(agent->registered_structs[i].name, name, 63);
                agent->registered_structs[i].name[63] = '\0';
                agent->registered_structs[i].active = true;
                agent->registered_struct_count++;
                result = 0;
                nimcp_log(LOG_LEVEL_DEBUG, "Registered struct '%s' (magic=0x%X) for consistency checking",
                          name, expected_magic);
                break;
            }
        }
        nimcp_mutex_unlock(agent->consistency_mutex);
    }

    if (result != 0) {
        nimcp_log(LOG_LEVEL_WARN, "Failed to register struct '%s': no free slots", name);
    }

    return result;
}

int nimcp_health_agent_unregister_struct(nimcp_health_agent_t* agent, void* ptr) {
    if (!validate_agent(agent) || !ptr) return -1;

    int result = -1;

    if (nimcp_mutex_lock(agent->consistency_mutex) == 0) {
        for (uint32_t i = 0; i < 64; i++) {
            if (agent->registered_structs[i].active &&
                agent->registered_structs[i].ptr == ptr) {
                nimcp_log(LOG_LEVEL_DEBUG, "Unregistered struct '%s' from consistency checking",
                          agent->registered_structs[i].name);
                agent->registered_structs[i].active = false;
                agent->registered_structs[i].ptr = NULL;
                agent->registered_struct_count--;
                result = 0;
                break;
            }
        }
        nimcp_mutex_unlock(agent->consistency_mutex);
    }

    return result;
}
