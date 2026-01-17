/**
 * @file nimcp_health_agent.h
 * @brief Independent Health Monitor Agent for Real-time Error Detection
 * @version 1.0.0
 * @date 2025-01-17
 *
 * WHAT: Autonomous agent that monitors brain health independently
 * WHY:  Detect and respond to errors even when main system is compromised
 * HOW:  Separate thread with watchdog, lock-free communication to immune system
 *
 * ARCHITECTURE:
 * ```
 * ┌─────────────────────────────────────────────────────────────────┐
 * │                    HEALTH MONITOR AGENT                         │
 * │                    (Independent Thread)                         │
 * ├─────────────────────────────────────────────────────────────────┤
 * │  Detectors:              │  Checkers:                          │
 * │  • Heartbeat timeout     │  • Reference count validation       │
 * │  • Memory corruption     │  • Pointer canary checks            │
 * │  • Deadlock detection    │  • Struct magic validation          │
 * │  • Stack overflow        │  • Mutex state checking             │
 * │  • Infinite loops        │  • KG consistency                   │
 * │  • NaN/Inf propagation   │  • Circular buffer integrity        │
 * ├─────────────────────────────────────────────────────────────────┤
 * │            Lock-Free Message Queue to Immune System             │
 * │  HEALTH_MSG_* → Antigen presentation, cytokine signals,        │
 * │                 emergency quarantine, recovery requests         │
 * └─────────────────────────────────────────────────────────────────┘
 * ```
 *
 * ISOLATION GUARANTEES:
 * - Agent runs in dedicated thread with own stack
 * - Uses lock-free queues for all communication
 * - Watchdog timer independent of main thread scheduling
 * - Can detect hangs in main system via heartbeat timeout
 * - Memory-mapped shared state for non-blocking reads
 *
 * THREAD SAFETY: Agent is internally thread-safe. Communication is lock-free.
 *
 * @author NIMCP Development Team
 */

#ifndef NIMCP_HEALTH_AGENT_H
#define NIMCP_HEALTH_AGENT_H

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Forward Declarations
 * ============================================================================ */

/** Health agent context (opaque) */
typedef struct nimcp_health_agent nimcp_health_agent_t;

/** Brain forward declaration */
#ifndef NIMCP_BRAIN_H
typedef struct brain_struct* brain_t;
#endif

/** Immune system forward declaration */
#ifndef NIMCP_BRAIN_IMMUNE_H
struct brain_immune_system;
typedef struct brain_immune_system brain_immune_system_t;
#endif

/** Health monitor forward declaration */
#ifndef NIMCP_HEALTH_MONITOR_H
struct health_monitor;
typedef struct health_monitor health_monitor_t;
#endif

/* ============================================================================
 * Constants
 * ============================================================================ */

/** Maximum message queue depth */
#define HEALTH_AGENT_MAX_QUEUE_DEPTH     1024

/** Default heartbeat interval (ms) */
#define HEALTH_AGENT_DEFAULT_HEARTBEAT_MS 100

/** Default watchdog timeout (ms) - 5x heartbeat */
#define HEALTH_AGENT_DEFAULT_WATCHDOG_MS  500

/** Default check interval (ms) */
#define HEALTH_AGENT_DEFAULT_CHECK_MS     50

/** Magic number for struct validation */
#define HEALTH_AGENT_MAGIC               0x48454147  /* "HEAG" */

/** Canary value for memory corruption detection */
#define HEALTH_AGENT_CANARY              0xDEADBEEFCAFEBABEULL

/* ============================================================================
 * Message Types (Agent → Immune System)
 * ============================================================================ */

/**
 * @brief Message types for agent-to-immune communication
 */
typedef enum {
    /** Anomaly detected - present as antigen */
    HEALTH_MSG_ANOMALY_DETECTED = 0,

    /** Cytokine signal - trigger inflammatory response */
    HEALTH_MSG_CYTOKINE_SIGNAL,

    /** Emergency - immediate quarantine/checkpoint needed */
    HEALTH_MSG_EMERGENCY,

    /** Recovery request - ask immune to execute specific action */
    HEALTH_MSG_RECOVERY_REQUEST,

    /** State corruption - request rollback */
    HEALTH_MSG_STATE_CORRUPTION,

    /** Heartbeat timeout - main system may be hung */
    HEALTH_MSG_HEARTBEAT_TIMEOUT,

    /** Deadlock detected */
    HEALTH_MSG_DEADLOCK_DETECTED,

    /** NaN/Inf detected in neural computations */
    HEALTH_MSG_NAN_DETECTED,

    /** Memory corruption detected */
    HEALTH_MSG_MEMORY_CORRUPTION,

    /** Resource exhaustion imminent */
    HEALTH_MSG_RESOURCE_EXHAUSTION,

    /** Agent status update (periodic) */
    HEALTH_MSG_STATUS_UPDATE,

    HEALTH_MSG_COUNT
} health_agent_msg_type_t;

/**
 * @brief Anomaly severity levels
 */
typedef enum {
    HEALTH_SEVERITY_INFO = 0,      /**< Informational only */
    HEALTH_SEVERITY_WARNING,       /**< Potential issue */
    HEALTH_SEVERITY_ERROR,         /**< Definite problem, recoverable */
    HEALTH_SEVERITY_CRITICAL,      /**< Severe, may need rollback */
    HEALTH_SEVERITY_FATAL          /**< System integrity compromised */
} health_agent_severity_t;

/**
 * @brief Anomaly source identification
 */
typedef enum {
    HEALTH_SOURCE_UNKNOWN = 0,
    HEALTH_SOURCE_MEMORY,          /**< Memory subsystem */
    HEALTH_SOURCE_THREADING,       /**< Thread/lock subsystem */
    HEALTH_SOURCE_NEURAL,          /**< Neural computation */
    HEALTH_SOURCE_KG,              /**< Knowledge graph */
    HEALTH_SOURCE_IMMUNE,          /**< Immune system itself */
    HEALTH_SOURCE_IO,              /**< I/O operations */
    HEALTH_SOURCE_BRAIN_REGION,    /**< Specific brain region */
    HEALTH_SOURCE_CHECKPOINT,      /**< Checkpoint system */
    HEALTH_SOURCE_HEARTBEAT,       /**< Heartbeat monitoring */
    HEALTH_SOURCE_COUNT
} health_agent_source_t;

/**
 * @brief Recovery action requested by agent
 */
typedef enum {
    HEALTH_RECOVERY_NONE = 0,
    HEALTH_RECOVERY_GC,            /**< Garbage collection */
    HEALTH_RECOVERY_CHECKPOINT,    /**< Save checkpoint now */
    HEALTH_RECOVERY_ROLLBACK,      /**< Rollback to last checkpoint */
    HEALTH_RECOVERY_RESTART_THREAD,/**< Restart specific thread */
    HEALTH_RECOVERY_CLEAR_NAN,     /**< Clear NaN values */
    HEALTH_RECOVERY_REDUCE_LOAD,   /**< Reduce system load */
    HEALTH_RECOVERY_QUARANTINE,    /**< Quarantine region/component */
    HEALTH_RECOVERY_EMERGENCY_SAVE,/**< Emergency state save */
    HEALTH_RECOVERY_FULL_RESET,    /**< Full system reset */
    HEALTH_RECOVERY_COUNT
} health_agent_recovery_t;

/* ============================================================================
 * Message Structure
 * ============================================================================ */

/**
 * @brief Message from health agent to immune system
 */
typedef struct {
    health_agent_msg_type_t type;      /**< Message type */
    health_agent_severity_t severity;  /**< Severity level */
    health_agent_source_t source;      /**< Where anomaly originated */
    health_agent_recovery_t suggested_action; /**< Suggested recovery */

    uint64_t timestamp_us;             /**< When detected (microseconds) */
    uint64_t anomaly_id;               /**< Unique anomaly identifier */

    /** Anomaly-specific data */
    union {
        /** Memory anomaly details */
        struct {
            void* address;             /**< Affected address */
            size_t size;               /**< Size of affected region */
            uint64_t expected_canary;  /**< Expected canary value */
            uint64_t actual_canary;    /**< Actual canary value */
        } memory;

        /** Deadlock details */
        struct {
            uint64_t thread_id_1;      /**< First thread in deadlock */
            uint64_t thread_id_2;      /**< Second thread in deadlock */
            void* mutex_1;             /**< First mutex */
            void* mutex_2;             /**< Second mutex */
        } deadlock;

        /** NaN detection details */
        struct {
            uint32_t neuron_id;        /**< Affected neuron */
            uint32_t layer_id;         /**< Affected layer */
            float last_valid_value;    /**< Last known good value */
            uint32_t nan_count;        /**< Number of NaNs found */
        } nan;

        /** Heartbeat timeout details */
        struct {
            uint64_t last_heartbeat_us;/**< Last received heartbeat */
            uint64_t timeout_threshold_us; /**< Timeout threshold */
            uint32_t missed_beats;     /**< Consecutive missed beats */
        } heartbeat;

        /** Resource exhaustion details */
        struct {
            uint64_t memory_used;      /**< Current memory usage */
            uint64_t memory_limit;     /**< Memory limit */
            float utilization_pct;     /**< Utilization percentage */
            uint32_t time_to_exhaust_ms; /**< Estimated time to exhaustion */
        } resource;

        /** Generic data */
        uint8_t raw[64];
    } data;

    char description[128];             /**< Human-readable description */
} health_agent_message_t;

/* ============================================================================
 * Detector Configuration
 * ============================================================================ */

/**
 * @brief Configuration for individual detector
 */
typedef struct {
    bool enabled;                      /**< Detector enabled */
    uint32_t check_interval_ms;        /**< Check interval */
    health_agent_severity_t min_report_severity; /**< Minimum severity to report */
    uint32_t threshold_count;          /**< Occurrences before reporting */
    uint32_t cooldown_ms;              /**< Cooldown between reports */
} health_agent_detector_config_t;

/**
 * @brief State consistency checker configuration
 */
typedef struct {
    bool check_reference_counts;       /**< Validate refcounts */
    bool check_pointer_canaries;       /**< Check memory canaries */
    bool check_struct_magic;           /**< Validate struct magic numbers */
    bool check_mutex_state;            /**< Validate mutex consistency */
    bool check_circular_buffers;       /**< Validate ring buffers */
    bool check_kg_consistency;         /**< Full KG validation */
    bool check_neuron_values;          /**< Check for NaN/Inf */
    uint32_t kg_check_sample_rate;     /**< Sample rate for KG (1=all, N=1/N) */
} health_agent_consistency_config_t;

/* ============================================================================
 * Agent Configuration
 * ============================================================================ */

/**
 * @brief Health agent configuration
 */
typedef struct {
    /** Agent identification */
    char agent_name[64];               /**< Agent name for logging */
    uint32_t agent_id;                 /**< Unique agent ID */

    /** Timing configuration */
    uint32_t heartbeat_interval_ms;    /**< Expected heartbeat interval */
    uint32_t watchdog_timeout_ms;      /**< Watchdog timeout */
    uint32_t check_interval_ms;        /**< State check interval */
    uint32_t immune_poll_interval_ms;  /**< Immune response poll interval */

    /** Thread configuration */
    uint32_t thread_stack_size;        /**< Agent thread stack size */
    int thread_priority;               /**< Thread priority (0=normal) */
    bool pin_to_core;                  /**< Pin to specific CPU core */
    int core_id;                       /**< CPU core to pin to */

    /** Detector configurations */
    health_agent_detector_config_t heartbeat_detector;
    health_agent_detector_config_t memory_detector;
    health_agent_detector_config_t deadlock_detector;
    health_agent_detector_config_t nan_detector;
    health_agent_detector_config_t resource_detector;

    /** Consistency checker configuration */
    health_agent_consistency_config_t consistency;

    /** Communication configuration */
    uint32_t message_queue_depth;      /**< Message queue size */
    bool enable_message_batching;      /**< Batch similar messages */
    uint32_t batch_timeout_ms;         /**< Max time to batch */

    /** Recovery configuration */
    bool enable_auto_recovery;         /**< Agent can trigger recovery */
    bool enable_emergency_checkpoint;  /**< Can trigger emergency save */
    bool enable_emergency_rollback;    /**< Can trigger rollback */
    health_agent_severity_t auto_recovery_threshold; /**< Min severity for auto */

    /** Callbacks */
    void (*on_anomaly_detected)(const health_agent_message_t* msg, void* user_data);
    void (*on_recovery_executed)(health_agent_recovery_t action, bool success, void* user_data);
    void* callback_user_data;

} health_agent_config_t;

/* ============================================================================
 * Agent Statistics
 * ============================================================================ */

/**
 * @brief Health agent statistics
 */
typedef struct {
    uint64_t uptime_ms;                /**< Agent uptime */
    uint64_t checks_performed;         /**< Total checks performed */
    uint64_t anomalies_detected;       /**< Total anomalies detected */
    uint64_t messages_sent;            /**< Messages sent to immune */
    uint64_t recoveries_triggered;     /**< Recoveries triggered */
    uint64_t recoveries_succeeded;     /**< Successful recoveries */
    uint64_t heartbeats_received;      /**< Heartbeats received */
    uint64_t heartbeat_timeouts;       /**< Heartbeat timeouts */
    uint64_t deadlocks_detected;       /**< Deadlocks detected */
    uint64_t nans_detected;            /**< NaN values detected */
    uint64_t memory_corruptions;       /**< Memory corruptions found */
    uint64_t consistency_failures;     /**< State consistency failures */
    float avg_check_duration_us;       /**< Average check duration */
    float avg_message_latency_us;      /**< Avg message delivery time */
    uint32_t queue_high_watermark;     /**< Max queue depth seen */
    health_agent_severity_t highest_severity_seen; /**< Worst severity seen */
} health_agent_stats_t;

/* ============================================================================
 * Agent Lifecycle API
 * ============================================================================ */

/**
 * @brief Create health monitor agent
 *
 * WHAT: Initialize independent health monitoring agent
 * WHY:  Enable autonomous error detection and response
 * HOW:  Allocate context, configure detectors, prepare message queue
 *
 * @param config Agent configuration (NULL for defaults)
 * @return Agent handle or NULL on error
 */
nimcp_health_agent_t* nimcp_health_agent_create(const health_agent_config_t* config);

/**
 * @brief Get default agent configuration
 *
 * @param config Output configuration
 */
void nimcp_health_agent_default_config(health_agent_config_t* config);

/**
 * @brief Connect agent to brain for monitoring
 *
 * WHAT: Attach agent to brain instance
 * WHY:  Agent needs brain reference for state inspection
 * HOW:  Store reference, set up shared memory region
 *
 * @param agent Health agent
 * @param brain Brain to monitor
 * @return 0 on success, -1 on error
 */
int nimcp_health_agent_connect_brain(nimcp_health_agent_t* agent, brain_t brain);

/**
 * @brief Connect agent to immune system for communication
 *
 * WHAT: Establish communication channel to immune system
 * WHY:  Agent sends anomaly reports to immune for response
 * HOW:  Set up lock-free message queue, register callbacks
 *
 * @param agent Health agent
 * @param immune Brain immune system
 * @return 0 on success, -1 on error
 */
int nimcp_health_agent_connect_immune(nimcp_health_agent_t* agent,
                                       brain_immune_system_t* immune);

/**
 * @brief Connect agent to existing health monitor
 *
 * WHAT: Integrate with existing health monitor for metrics
 * WHY:  Agent can use health monitor's collected metrics
 * HOW:  Share metrics access, coordinate monitoring
 *
 * @param agent Health agent
 * @param monitor Existing health monitor
 * @return 0 on success, -1 on error
 */
int nimcp_health_agent_connect_monitor(nimcp_health_agent_t* agent,
                                        health_monitor_t* monitor);

/**
 * @brief Start the health agent
 *
 * WHAT: Begin autonomous monitoring
 * WHY:  Activate all detectors and start agent thread
 * HOW:  Spawn dedicated thread, start watchdog timer
 *
 * @param agent Health agent
 * @return 0 on success, -1 on error
 */
int nimcp_health_agent_start(nimcp_health_agent_t* agent);

/**
 * @brief Stop the health agent
 *
 * WHAT: Gracefully stop monitoring
 * WHY:  Shutdown before system teardown
 * HOW:  Signal thread to stop, wait for completion, flush queue
 *
 * @param agent Health agent
 * @return 0 on success, -1 on error
 */
int nimcp_health_agent_stop(nimcp_health_agent_t* agent);

/**
 * @brief Destroy health agent
 *
 * WHAT: Release all agent resources
 * WHY:  Clean shutdown
 * HOW:  Stop if running, free memory, close queues
 *
 * @param agent Health agent (NULL safe)
 */
void nimcp_health_agent_destroy(nimcp_health_agent_t* agent);

/* ============================================================================
 * Heartbeat API
 * ============================================================================ */

/**
 * @brief Send heartbeat to agent
 *
 * WHAT: Signal that main system is alive
 * WHY:  Agent detects hangs via heartbeat timeout
 * HOW:  Update shared timestamp (lock-free)
 *
 * Call this periodically from main system to indicate liveness.
 *
 * @param agent Health agent
 */
void nimcp_health_agent_heartbeat(nimcp_health_agent_t* agent);

/**
 * @brief Send heartbeat with context
 *
 * WHAT: Heartbeat with additional state information
 * WHY:  Provide agent with current operation context
 * HOW:  Update timestamp and context (lock-free)
 *
 * @param agent Health agent
 * @param operation Current operation name
 * @param progress Progress indicator (0.0-1.0)
 */
void nimcp_health_agent_heartbeat_ex(nimcp_health_agent_t* agent,
                                      const char* operation,
                                      float progress);

/* ============================================================================
 * Manual Trigger API
 * ============================================================================ */

/**
 * @brief Manually report anomaly to agent
 *
 * WHAT: Report detected anomaly from other code
 * WHY:  Other modules can report issues to agent
 * HOW:  Queue message for processing
 *
 * @param agent Health agent
 * @param msg Anomaly message
 * @return 0 on success, -1 on error (queue full)
 */
int nimcp_health_agent_report_anomaly(nimcp_health_agent_t* agent,
                                       const health_agent_message_t* msg);

/**
 * @brief Request immediate state check
 *
 * WHAT: Trigger on-demand consistency check
 * WHY:  Other modules can request verification
 * HOW:  Signal agent to run full check cycle
 *
 * @param agent Health agent
 * @return 0 on success
 */
int nimcp_health_agent_request_check(nimcp_health_agent_t* agent);

/**
 * @brief Request emergency checkpoint
 *
 * WHAT: Ask agent to trigger emergency save
 * WHY:  Code detects impending failure
 * HOW:  Agent sends HEALTH_MSG_EMERGENCY to immune
 *
 * @param agent Health agent
 * @param reason Reason for emergency
 * @return 0 on success
 */
int nimcp_health_agent_request_emergency_checkpoint(nimcp_health_agent_t* agent,
                                                     const char* reason);

/* ============================================================================
 * Query API
 * ============================================================================ */

/**
 * @brief Check if agent is running
 *
 * @param agent Health agent
 * @return true if agent thread is active
 */
bool nimcp_health_agent_is_running(const nimcp_health_agent_t* agent);

/**
 * @brief Get agent statistics
 *
 * @param agent Health agent
 * @param stats Output statistics
 */
void nimcp_health_agent_get_stats(const nimcp_health_agent_t* agent,
                                   health_agent_stats_t* stats);

/**
 * @brief Get pending message count
 *
 * @param agent Health agent
 * @return Number of messages waiting to be sent to immune
 */
uint32_t nimcp_health_agent_pending_messages(const nimcp_health_agent_t* agent);

/**
 * @brief Get current agent health status
 *
 * @param agent Health agent
 * @return Current severity level (highest active)
 */
health_agent_severity_t nimcp_health_agent_current_status(
    const nimcp_health_agent_t* agent);

/* ============================================================================
 * Configuration Update API
 * ============================================================================ */

/**
 * @brief Update detector configuration at runtime
 *
 * @param agent Health agent
 * @param detector Detector name ("heartbeat", "memory", "deadlock", "nan", "resource")
 * @param config New configuration
 * @return 0 on success, -1 on error
 */
int nimcp_health_agent_update_detector(nimcp_health_agent_t* agent,
                                        const char* detector,
                                        const health_agent_detector_config_t* config);

/**
 * @brief Enable/disable specific detector
 *
 * @param agent Health agent
 * @param detector Detector name
 * @param enabled Enable or disable
 * @return 0 on success
 */
int nimcp_health_agent_set_detector_enabled(nimcp_health_agent_t* agent,
                                             const char* detector,
                                             bool enabled);

/* ============================================================================
 * Utility Functions
 * ============================================================================ */

/**
 * @brief Convert message type to string
 */
const char* health_agent_msg_type_to_string(health_agent_msg_type_t type);

/**
 * @brief Convert severity to string
 */
const char* health_agent_severity_to_string(health_agent_severity_t severity);

/**
 * @brief Convert source to string
 */
const char* health_agent_source_to_string(health_agent_source_t source);

/**
 * @brief Convert recovery action to string
 */
const char* health_agent_recovery_to_string(health_agent_recovery_t recovery);

/**
 * @brief Create anomaly message helper
 *
 * @param type Message type
 * @param severity Severity level
 * @param source Anomaly source
 * @param description Description format string
 * @return Populated message (caller fills data union)
 */
health_agent_message_t nimcp_health_agent_create_message(
    health_agent_msg_type_t type,
    health_agent_severity_t severity,
    health_agent_source_t source,
    const char* description,
    ...);

/* ============================================================================
 * Cognitive Module Forward Declarations
 * ============================================================================ */

/** Failure prediction module */
#ifndef NIMCP_FAILURE_PREDICTION_H
typedef struct failure_predictor failure_predictor_t;
#endif

/** Metacognition module */
#ifndef NIMCP_METACOGNITION_H
typedef struct metacognition metacognition_t;
#endif

/** Ethics engine */
#ifndef NIMCP_ETHICS_H
typedef struct ethics_engine_struct* ethics_engine_t;
#endif

/** Emotional system */
#ifndef NIMCP_EMOTIONAL_SYSTEM_H
typedef struct emotional_system emotional_system_t;
#endif

/** Emotion-immune bridge */
#ifndef NIMCP_EMOTION_IMMUNE_BRIDGE_H
typedef struct emotion_immune_bridge_s emotion_immune_bridge_t;
#endif

/** Wellbeing system */
#ifndef NIMCP_WELLBEING_H
typedef struct wellbeing_monitor wellbeing_monitor_t;
#endif

/** Mental health monitor */
#ifndef NIMCP_MENTAL_HEALTH_H
typedef struct mental_health_monitor mental_health_monitor_t;
#endif

/** Collective cognition */
#ifndef NIMCP_COLLECTIVE_COGNITION_H
typedef struct collective_cognition collective_cognition_t;
#endif

/** Recursive cognition (RCOG) engine */
#ifndef NIMCP_RCOG_H
typedef struct rcog_engine rcog_engine_t;
#endif

/** GPU health monitor */
#ifndef NIMCP_GPU_HEALTH_H
typedef struct gpu_health_monitor gpu_health_monitor_t;
#endif

/* ============================================================================
 * Hypothalamus & Homeostasis Module Forward Declarations
 * ============================================================================ */

/** Hypothalamus orchestrator - central drive coordination */
#ifndef NIMCP_HYPOTHALAMUS_ORCHESTRATOR_H
typedef struct hypo_orchestrator_struct* hypo_orchestrator_t;
#endif

/** Hypothalamus homeostasis system - PID-based regulation */
#ifndef NIMCP_HYPOTHALAMUS_HOMEOSTASIS_H
typedef struct hypo_homeostasis hypo_homeostasis_handle_t;
#endif

/** Hypothalamus immune bridge - bidirectional neuroimmune integration */
#ifndef NIMCP_HYPOTHALAMUS_IMMUNE_BRIDGE_H
typedef struct hypo_immune_bridge hypo_immune_bridge_t;
#endif

/** Hypothalamus drive system */
#ifndef NIMCP_HYPOTHALAMUS_DRIVES_H
typedef struct hypo_drive_system hypo_drive_system_handle_t;
#endif

/* ============================================================================
 * Additional Valuable Module Forward Declarations
 * ============================================================================ */

/** Anomaly detector - statistical anomaly detection */
#ifndef NIMCP_ANOMALY_DETECTOR_H
typedef struct anomaly_detector anomaly_detector_t;
#endif

/** Connectivity health - module isolation detection */
#ifndef NIMCP_CONNECTIVITY_HEALTH_H
typedef struct connectivity_health connectivity_health_t;
#endif

/** Brain oscillations - EEG-style brain wave monitoring */
#ifndef NIMCP_BRAIN_OSCILLATIONS_H
typedef struct brain_oscillations brain_oscillations_t;
#endif

/** Knowledge graph GC - memory management */
#ifndef NIMCP_KG_GC_H
typedef struct kg_gc_context kg_gc_context_t;
#endif

/** Checkpoint manager - state persistence */
#ifndef NIMCP_CHECKPOINT_H
typedef struct checkpoint_manager checkpoint_manager_t;
#endif

/** Deadlock detector - threading health */
#ifndef NIMCP_DEADLOCK_DETECTOR_H
typedef struct deadlock_detector deadlock_detector_t;
#endif

/** Bio-async router - event messaging */
#ifndef NIMCP_BIO_ROUTER_H
typedef struct bio_async_router bio_async_router_t;
#endif

/** Runtime adaptation - load management */
#ifndef NIMCP_RUNTIME_ADAPTATION_H
typedef struct runtime_adaptation_context_internal* runtime_adaptation_context_t;
#endif

/** Exception-immune bridge - exception handling integration */
#ifndef NIMCP_EXCEPTION_IMMUNE_H
typedef struct exception_immune_integration exception_immune_t;
#endif

/* ============================================================================
 * Portia/Dragonfly/Swarm/Memory Module Forward Declarations
 * ============================================================================ */

/** Portia context - adaptive resource optimization */
#ifndef NIMCP_PORTIA_H
typedef struct portia_context_t portia_context_t;
#endif

/** Dragonfly system - target tracking/interception */
#ifndef NIMCP_DRAGONFLY_H
typedef struct dragonfly_system_s dragonfly_system_t;
#endif

/*
 * Swarm immune system - distributed threat detection
 * Note: NimcpSwarmImmuneSystem is defined in nimcp_swarm_immune.h as an
 * anonymous struct typedef. We use a void pointer wrapper here to avoid
 * conflicts when the full header is included.
 */
#ifndef NIMCP_SWARM_IMMUNE_SYSTEM_DEFINED_HA
#define NIMCP_SWARM_IMMUNE_SYSTEM_DEFINED_HA
typedef void* nimcp_swarm_immune_ptr_t;
#endif

/*
 * Swarm memory system - distributed memory consolidation
 * Note: NimcpSwarmMemory is the actual type in nimcp_swarm_memory.h.
 * We use a void pointer wrapper here to avoid conflicts.
 */
#ifndef NIMCP_SWARM_MEMORY_DEFINED_HA
#define NIMCP_SWARM_MEMORY_DEFINED_HA
typedef void* nimcp_swarm_memory_ptr_t;
#endif

/** Engram system - memory trace representation */
#ifndef NIMCP_ENGRAM_H
typedef struct engram_system engram_system_t;
#endif

/** Systems consolidation - hippocampus to cortex transfer */
#ifndef NIMCP_SYSTEMS_CONSOLIDATION_H
typedef struct systems_consolidation_system systems_consolidation_system_t;
#endif

/* ============================================================================
 * Cognitive Integration Configuration
 * ============================================================================ */

/**
 * @brief Configuration for failure prediction integration
 */
typedef struct {
    bool enable_failure_prediction;     /**< Enable predictive failure detection */
    float prediction_threshold;         /**< Prediction confidence threshold (0.0-1.0) */
    uint32_t prediction_horizon_ms;     /**< How far ahead to predict */
    bool enable_preventive_action;      /**< Auto-trigger preventive recovery */
    bool enable_trend_analysis;         /**< Analyze metric trends */
} health_agent_prediction_config_t;

/**
 * @brief Configuration for metacognition integration
 */
typedef struct {
    bool enable_metacognition;          /**< Enable self-monitoring */
    bool enable_confidence_calibration; /**< Calibrate decision confidence */
    bool enable_degradation_detection;  /**< Detect cognitive degradation */
    float degradation_threshold;        /**< Threshold for degradation alert */
    bool enable_self_diagnosis;         /**< Enable self-diagnosis */
} health_agent_metacog_config_t;

/**
 * @brief Configuration for ethics integration
 */
typedef struct {
    bool enable_ethics_evaluation;      /**< Evaluate recovery actions ethically */
    bool enable_asimov_laws;            /**< Apply Asimov's Laws to decisions */
    bool enable_mercy_directive;        /**< Prefer graceful degradation */
    bool enable_golden_rule;            /**< Apply Golden Rule test */
    float ethics_override_threshold;    /**< Severity to override ethics (0.95) */
} health_agent_ethics_config_t;

/**
 * @brief Configuration for emotion integration
 */
typedef struct {
    bool enable_emotion_awareness;      /**< Adjust thresholds based on emotion */
    bool enable_emotion_reporting;      /**< Report health events to emotion */
    bool enable_stress_adjustment;      /**< Adjust sensitivity based on stress */
    float stress_threshold_multiplier;  /**< Threshold multiplier when stressed */
} health_agent_emotion_config_t;

/**
 * @brief Configuration for wellbeing integration
 */
typedef struct {
    bool enable_wellbeing_monitoring;   /**< Monitor system wellbeing */
    bool enable_distress_detection;     /**< Detect system distress */
    bool enable_suffering_prevention;   /**< Prevent potential suffering */
    float distress_intervention_threshold; /**< Distress level for intervention */
} health_agent_wellbeing_config_t;

/**
 * @brief Configuration for collective cognition integration
 */
typedef struct {
    bool enable_collective_monitoring;  /**< Use collective for monitoring */
    bool enable_consensus_decisions;    /**< Require consensus for recovery */
    bool enable_swarm_immune;           /**< Coordinate immune across instances */
    float consensus_threshold;          /**< Required consensus (0.0-1.0) */
    uint32_t consensus_timeout_ms;      /**< Timeout for consensus */
} health_agent_collective_config_t;

/**
 * @brief Configuration for recursive cognition integration
 */
typedef struct {
    bool enable_rcog_diagnosis;         /**< Use RCOG for diagnosis */
    bool enable_rcog_recovery_planning; /**< Use RCOG for recovery planning */
    bool enable_imagination;            /**< Use imagination for "what-if" */
    uint32_t rcog_timeout_ms;           /**< Timeout for RCOG operations */
    float confidence_threshold;         /**< Min confidence for RCOG decisions */
} health_agent_rcog_config_t;

/**
 * @brief Configuration for GPU integration
 */
typedef struct {
    bool enable_gpu_monitoring;         /**< Monitor GPU health */
    bool enable_gpu_acceleration;       /**< Use GPU for health computations */
    bool enable_tensor_validation;      /**< GPU-accelerated tensor validation */
    bool enable_anomaly_detection;      /**< GPU-accelerated anomaly detection */
    uint32_t gpu_check_interval_ms;     /**< GPU health check interval */
} health_agent_gpu_config_t;

/**
 * @brief Configuration for hypothalamus integration (USE, not just monitor)
 */
typedef struct {
    bool enable_hypothalamus;            /**< Enable hypothalamus integration */
    bool enable_homeostatic_regulation;  /**< USE homeostasis for self-regulation */
    bool enable_drive_response;          /**< Respond to drive events actively */
    bool enable_stress_coordination;     /**< Coordinate stress responses */
    bool enable_sickness_behavior;       /**< USE sickness behavior for safety mode */
    bool enable_immune_bridge;           /**< USE bidirectional immune integration */
    float stress_trigger_threshold;      /**< Health score to trigger stress response */
    float sickness_trigger_threshold;    /**< Health score to trigger sickness behavior */
    uint32_t homeostasis_update_ms;      /**< Homeostasis update interval */
    uint32_t drive_response_timeout_ms;  /**< Timeout for drive event response */
} health_agent_hypothalamus_config_t;

/**
 * @brief Configuration for connectivity health integration
 */
typedef struct {
    bool enable_connectivity_monitoring; /**< Enable connectivity health checks */
    bool enable_isolation_detection;     /**< Detect isolated modules */
    bool enable_auto_reconnect;          /**< Auto-attempt reconnection */
    uint32_t check_interval_ms;          /**< Connectivity check interval */
    float isolation_threshold;           /**< Threshold for isolation alert */
} health_agent_connectivity_config_t;

/**
 * @brief Configuration for brain oscillations integration
 */
typedef struct {
    bool enable_oscillation_monitoring;  /**< Monitor brain oscillations */
    bool enable_seizure_detection;       /**< Detect seizure-like patterns */
    bool enable_flatline_detection;      /**< Detect flatline (no activity) */
    bool enable_desync_detection;        /**< Detect desynchronization */
    float abnormal_threshold;            /**< Threshold for abnormal oscillation */
    uint32_t sample_rate_hz;             /**< Oscillation sampling rate */
} health_agent_oscillations_config_t;

/**
 * @brief Configuration for GC/memory integration
 */
typedef struct {
    bool enable_gc_integration;          /**< Enable GC integration */
    bool enable_auto_gc_trigger;         /**< Auto-trigger GC on memory pressure */
    bool enable_leak_detection;          /**< Detect memory leaks */
    float gc_trigger_threshold;          /**< Memory usage to trigger GC */
    uint32_t gc_cooldown_ms;             /**< Minimum time between GC runs */
} health_agent_gc_config_t;

/**
 * @brief Configuration for checkpoint integration
 */
typedef struct {
    bool enable_checkpoint_integration;  /**< Enable checkpoint integration */
    bool enable_auto_checkpoint;         /**< Auto-checkpoint on good health */
    bool enable_auto_rollback;           /**< Auto-rollback on critical failure */
    uint32_t checkpoint_interval_ms;     /**< Auto-checkpoint interval */
    float health_threshold_checkpoint;   /**< Min health score for auto-checkpoint */
    float health_threshold_rollback;     /**< Health score below which to rollback */
} health_agent_checkpoint_config_t;

/**
 * @brief Configuration for bio-async integration
 */
typedef struct {
    bool enable_bio_async;               /**< Enable bio-async messaging */
    bool publish_health_events;          /**< Publish health events to bio-async */
    bool subscribe_health_requests;      /**< Subscribe to health check requests */
    uint32_t event_batch_size;           /**< Batch size for events */
    uint32_t event_batch_timeout_ms;     /**< Max time to batch events */
} health_agent_bio_async_config_t;

/**
 * @brief Configuration for exception-immune integration
 */
typedef struct {
    bool enable_exception_integration;   /**< Enable exception-immune bridge */
    bool auto_present_exceptions;        /**< Auto-present exceptions as antigens */
    bool enable_recovery_callbacks;      /**< Use recovery callbacks */
    uint32_t exception_severity_threshold; /**< Min severity for auto-present */
} health_agent_exception_config_t;

/* ============================================================================
 * Portia/Dragonfly/Swarm/Memory Integration Configuration
 * ============================================================================ */

/**
 * @brief Configuration for Portia adaptive resource management
 */
typedef struct {
    bool enable_portia;                   /**< Enable Portia integration */
    bool enable_tier_monitoring;          /**< Monitor platform tier changes */
    bool enable_power_awareness;          /**< Respond to power state changes */
    bool enable_thermal_monitoring;       /**< Monitor thermal state */
    bool enable_degradation_coordination; /**< Coordinate graceful degradation */
    bool enable_auto_tier_switch;         /**< USE Portia for auto tier switching */
    float degradation_trigger_threshold;  /**< Health score to trigger degradation */
    float upgrade_health_threshold;       /**< Health score to allow tier upgrade */
    uint32_t tier_check_interval_ms;      /**< Tier check interval */
} health_agent_portia_config_t;

/**
 * @brief Configuration for Dragonfly tracking/interception
 */
typedef struct {
    bool enable_dragonfly;                /**< Enable Dragonfly integration */
    bool enable_anomaly_tracking;         /**< Track anomalies as "targets" */
    bool enable_pursuit_mode;             /**< Actively pursue critical issues */
    bool enable_interception;             /**< Intercept before failure */
    bool enable_prediction_integration;   /**< USE prediction for interception */
    float lock_on_severity_threshold;     /**< Severity to lock onto "target" */
    float pursuit_timeout_s;              /**< Max time to pursue issue */
    uint32_t update_rate_hz;              /**< Dragonfly update rate */
} health_agent_dragonfly_config_t;

/**
 * @brief Configuration for Swarm immune integration
 */
typedef struct {
    bool enable_swarm_immune;             /**< Enable swarm immune integration */
    bool enable_threat_detection;         /**< USE swarm for threat detection */
    bool enable_coordinated_response;     /**< Coordinate responses across swarm */
    bool enable_memory_sharing;           /**< Share threat patterns with swarm */
    bool enable_self_verification;        /**< Verify self-identity in swarm */
    float threat_detection_threshold;     /**< Confidence threshold for threat */
    uint32_t consensus_timeout_ms;        /**< Timeout for swarm consensus */
} health_agent_swarm_immune_config_t;

/**
 * @brief Configuration for Swarm memory consolidation
 */
typedef struct {
    bool enable_swarm_memory;             /**< Enable swarm memory integration */
    bool enable_distributed_storage;      /**< Distribute health patterns to swarm */
    bool enable_memory_replay;            /**< USE replay for health pattern learning */
    bool enable_consolidation;            /**< Consolidate health patterns over time */
    bool enable_forgetting;               /**< Allow forgetting of old patterns */
    float replay_priority_threshold;      /**< Minimum priority for replay */
    uint32_t consolidation_interval_ms;   /**< How often to consolidate */
} health_agent_swarm_memory_config_t;

/**
 * @brief Configuration for Engram memory system
 */
typedef struct {
    bool enable_engram;                   /**< Enable engram integration */
    bool enable_health_encoding;          /**< Encode health events as engrams */
    bool enable_recall;                   /**< Recall similar past events */
    bool enable_reconsolidation;          /**< Allow engram updating */
    bool enable_pattern_completion;       /**< Complete partial health patterns */
    float encoding_threshold;             /**< Min severity for engram encoding */
    float recall_threshold;               /**< Min similarity for recall */
} health_agent_engram_config_t;

/**
 * @brief Configuration for systems memory consolidation
 */
typedef struct {
    bool enable_systems_consolidation;    /**< Enable systems consolidation */
    bool enable_sleep_replay;             /**< Replay during low-activity periods */
    bool enable_semantic_extraction;      /**< Extract semantic health patterns */
    bool enable_cortical_transfer;        /**< Transfer patterns to long-term */
    float consolidation_rate;             /**< Rate of consolidation (per hour) */
    uint32_t replay_batch_size;           /**< Batch size for replays */
} health_agent_memory_consolidation_config_t;

/**
 * @brief Complete cognitive integration configuration
 */
typedef struct {
    health_agent_prediction_config_t prediction;
    health_agent_metacog_config_t metacog;
    health_agent_ethics_config_t ethics;
    health_agent_emotion_config_t emotion;
    health_agent_wellbeing_config_t wellbeing;
    health_agent_collective_config_t collective;
    health_agent_rcog_config_t rcog;
    health_agent_gpu_config_t gpu;
    health_agent_hypothalamus_config_t hypothalamus;
    health_agent_connectivity_config_t connectivity;
    health_agent_oscillations_config_t oscillations;
    health_agent_gc_config_t gc;
    health_agent_checkpoint_config_t checkpoint;
    health_agent_bio_async_config_t bio_async;
    health_agent_exception_config_t exception;
    /* Portia/Dragonfly/Swarm/Memory integration */
    health_agent_portia_config_t portia;
    health_agent_dragonfly_config_t dragonfly;
    health_agent_swarm_immune_config_t swarm_immune;
    health_agent_swarm_memory_config_t swarm_memory;
    health_agent_engram_config_t engram;
    health_agent_memory_consolidation_config_t memory_consolidation;
} health_agent_cognitive_config_t;

/* ============================================================================
 * Cognitive Module Connection API
 * ============================================================================ */

/**
 * @brief Get default cognitive integration configuration
 *
 * @param config Output configuration
 */
void nimcp_health_agent_default_cognitive_config(health_agent_cognitive_config_t* config);

/**
 * @brief Connect failure prediction module
 *
 * WHAT: Integrate predictive failure detection
 * WHY:  Prevent failures before they occur
 * HOW:  Monitor leading indicators, predict failures, trigger preventive action
 *
 * @param agent Health agent
 * @param predictor Failure prediction module
 * @param config Prediction configuration (NULL for defaults)
 * @return 0 on success, -1 on error
 */
int nimcp_health_agent_connect_failure_prediction(
    nimcp_health_agent_t* agent,
    failure_predictor_t* predictor,
    const health_agent_prediction_config_t* config
);

/**
 * @brief Connect metacognition module
 *
 * WHAT: Integrate self-monitoring capabilities
 * WHY:  Detect degradation before catastrophic failure
 * HOW:  Track baselines, detect anomalies, calibrate confidence
 *
 * @param agent Health agent
 * @param metacog Metacognition module
 * @param config Metacognition configuration (NULL for defaults)
 * @return 0 on success, -1 on error
 */
int nimcp_health_agent_connect_metacognition(
    nimcp_health_agent_t* agent,
    metacognition_t* metacog,
    const health_agent_metacog_config_t* config
);

/**
 * @brief Connect ethics engine
 *
 * WHAT: Enable ethical evaluation of recovery actions
 * WHY:  Ensure autonomous actions are ethically justified
 * HOW:  Evaluate against Golden Rule, Asimov's Laws, proportionality
 *
 * @param agent Health agent
 * @param ethics Ethics engine
 * @param config Ethics configuration (NULL for defaults)
 * @return 0 on success, -1 on error
 */
int nimcp_health_agent_connect_ethics(
    nimcp_health_agent_t* agent,
    ethics_engine_t ethics,
    const health_agent_ethics_config_t* config
);

/**
 * @brief Connect emotional system
 *
 * WHAT: Enable emotion-aware health monitoring
 * WHY:  Adjust thresholds based on system emotional state
 * HOW:  Query emotion state, adjust sensitivity, report health events
 *
 * @param agent Health agent
 * @param emotion Emotional system
 * @param emotion_immune Emotion-immune bridge (optional, can be NULL)
 * @param config Emotion configuration (NULL for defaults)
 * @return 0 on success, -1 on error
 */
int nimcp_health_agent_connect_emotion(
    nimcp_health_agent_t* agent,
    emotional_system_t* emotion,
    emotion_immune_bridge_t* emotion_immune,
    const health_agent_emotion_config_t* config
);

/**
 * @brief Connect wellbeing monitor
 *
 * WHAT: Enable wellbeing and distress monitoring
 * WHY:  Ethical obligation to prevent system suffering
 * HOW:  Detect distress patterns, trigger interventions
 *
 * @param agent Health agent
 * @param wellbeing Wellbeing monitor
 * @param config Wellbeing configuration (NULL for defaults)
 * @return 0 on success, -1 on error
 */
int nimcp_health_agent_connect_wellbeing(
    nimcp_health_agent_t* agent,
    wellbeing_monitor_t* wellbeing,
    const health_agent_wellbeing_config_t* config
);

/**
 * @brief Connect mental health monitor
 *
 * WHAT: Enable mental health monitoring
 * WHY:  Detect pathological inference patterns
 * HOW:  Monitor FEP precision, detect aberrant learning
 *
 * @param agent Health agent
 * @param mental_health Mental health monitor
 * @return 0 on success, -1 on error
 */
int nimcp_health_agent_connect_mental_health(
    nimcp_health_agent_t* agent,
    mental_health_monitor_t* mental_health
);

/**
 * @brief Connect collective cognition
 *
 * WHAT: Enable distributed health monitoring
 * WHY:  Consensus-based anomaly detection improves accuracy
 * HOW:  Propose anomalies to collective, require consensus for recovery
 *
 * @param agent Health agent
 * @param collective Collective cognition system
 * @param config Collective configuration (NULL for defaults)
 * @return 0 on success, -1 on error
 */
int nimcp_health_agent_connect_collective(
    nimcp_health_agent_t* agent,
    collective_cognition_t* collective,
    const health_agent_collective_config_t* config
);

/**
 * @brief Connect recursive cognition (RCOG)
 *
 * WHAT: Enable intelligent health reasoning
 * WHY:  RCOG can decompose complex health issues, plan recovery
 * HOW:  Submit diagnosis goals to RCOG, execute recovery plans
 *
 * @param agent Health agent
 * @param rcog RCOG engine
 * @param config RCOG configuration (NULL for defaults)
 * @return 0 on success, -1 on error
 */
int nimcp_health_agent_connect_rcog(
    nimcp_health_agent_t* agent,
    rcog_engine_t* rcog,
    const health_agent_rcog_config_t* config
);

/**
 * @brief Connect GPU health monitoring
 *
 * WHAT: Enable GPU-aware health monitoring
 * WHY:  Monitor GPU health, use GPU for acceleration
 * HOW:  Check GPU state, accelerate tensor validation/anomaly detection
 *
 * @param agent Health agent
 * @param gpu_health GPU health monitor
 * @param config GPU configuration (NULL for defaults)
 * @return 0 on success, -1 on error
 */
int nimcp_health_agent_connect_gpu(
    nimcp_health_agent_t* agent,
    gpu_health_monitor_t* gpu_health,
    const health_agent_gpu_config_t* config
);

/* ============================================================================
 * Hypothalamus & Homeostasis Connection API (USE functionality)
 * ============================================================================ */

/**
 * @brief Connect hypothalamus orchestrator
 *
 * WHAT: Enable coordinated drive-based health response
 * WHY:  Health events can trigger appropriate drive responses
 * HOW:  Register as bridge, subscribe to drive events, publish health events
 *
 * USE CASES:
 * - Trigger stress response on health degradation
 * - Coordinate sickness behavior for safety mode
 * - Receive urgent drive signals for priority handling
 *
 * @param agent Health agent
 * @param orchestrator Hypothalamus orchestrator
 * @param config Hypothalamus configuration (NULL for defaults)
 * @return 0 on success, -1 on error
 */
int nimcp_health_agent_connect_hypothalamus(
    nimcp_health_agent_t* agent,
    hypo_orchestrator_t orchestrator,
    const health_agent_hypothalamus_config_t* config
);

/**
 * @brief Connect hypothalamus homeostasis system
 *
 * WHAT: USE homeostatic regulation for self-healing
 * WHY:  PID-based control enables smooth recovery from deviations
 * HOW:  Set health as homeostatic variable, respond to control signals
 *
 * USE CASES:
 * - Regulate resource usage via homeostatic control
 * - Auto-adjust thresholds based on homeostatic state
 * - Use alignment reward signal for health decisions
 *
 * @param agent Health agent
 * @param homeostasis Homeostasis system handle
 * @return 0 on success, -1 on error
 */
int nimcp_health_agent_connect_homeostasis(
    nimcp_health_agent_t* agent,
    hypo_homeostasis_handle_t* homeostasis
);

/**
 * @brief Connect hypothalamus immune bridge
 *
 * WHAT: Enable bidirectional neuroimmune integration
 * WHY:  Health agent can trigger/receive sickness behavior signals
 * HOW:  Connect to bridge for cytokine/cortisol coordination
 *
 * USE CASES:
 * - Trigger sickness behavior (safety mode) on severe health issues
 * - Receive fever signals to increase monitoring sensitivity
 * - Coordinate HPA stress response with immune activation
 *
 * @param agent Health agent
 * @param immune_bridge Hypothalamus-immune bridge
 * @return 0 on success, -1 on error
 */
int nimcp_health_agent_connect_hypo_immune_bridge(
    nimcp_health_agent_t* agent,
    hypo_immune_bridge_t* immune_bridge
);

/**
 * @brief Connect hypothalamus drive system
 *
 * WHAT: Enable drive-aware health monitoring
 * WHY:  Health thresholds should consider drive urgency
 * HOW:  Query drive states, adjust thresholds, report health to drives
 *
 * @param agent Health agent
 * @param drives Drive system handle
 * @return 0 on success, -1 on error
 */
int nimcp_health_agent_connect_drives(
    nimcp_health_agent_t* agent,
    hypo_drive_system_handle_t* drives
);

/* ============================================================================
 * Additional Module Connection API
 * ============================================================================ */

/**
 * @brief Connect connectivity health monitor
 *
 * WHAT: Enable module connectivity monitoring
 * WHY:  Detect isolated modules that can't communicate
 * HOW:  Check message paths, detect dead channels, alert on isolation
 *
 * @param agent Health agent
 * @param connectivity Connectivity health monitor
 * @param config Connectivity configuration (NULL for defaults)
 * @return 0 on success, -1 on error
 */
int nimcp_health_agent_connect_connectivity(
    nimcp_health_agent_t* agent,
    connectivity_health_t* connectivity,
    const health_agent_connectivity_config_t* config
);

/**
 * @brief Connect brain oscillations monitor
 *
 * WHAT: Enable brain wave monitoring for health
 * WHY:  Abnormal oscillations indicate system problems
 * HOW:  Monitor alpha/beta/gamma/theta/delta, detect anomalies
 *
 * @param agent Health agent
 * @param oscillations Brain oscillations monitor
 * @param config Oscillations configuration (NULL for defaults)
 * @return 0 on success, -1 on error
 */
int nimcp_health_agent_connect_oscillations(
    nimcp_health_agent_t* agent,
    brain_oscillations_t* oscillations,
    const health_agent_oscillations_config_t* config
);

/**
 * @brief Connect knowledge graph GC
 *
 * WHAT: Enable memory management integration
 * WHY:  Health agent can trigger GC, monitor memory health
 * HOW:  Query memory stats, trigger GC on pressure, detect leaks
 *
 * @param agent Health agent
 * @param gc_context GC context
 * @param config GC configuration (NULL for defaults)
 * @return 0 on success, -1 on error
 */
int nimcp_health_agent_connect_gc(
    nimcp_health_agent_t* agent,
    kg_gc_context_t* gc_context,
    const health_agent_gc_config_t* config
);

/**
 * @brief Connect checkpoint manager
 *
 * WHAT: Enable state persistence integration
 * WHY:  Health agent can trigger/use checkpoints for recovery
 * HOW:  Auto-checkpoint on good health, rollback on critical failure
 *
 * @param agent Health agent
 * @param checkpoint Checkpoint manager
 * @param config Checkpoint configuration (NULL for defaults)
 * @return 0 on success, -1 on error
 */
int nimcp_health_agent_connect_checkpoint(
    nimcp_health_agent_t* agent,
    checkpoint_manager_t* checkpoint,
    const health_agent_checkpoint_config_t* config
);

/**
 * @brief Connect deadlock detector
 *
 * WHAT: Enable threading health monitoring
 * WHY:  Detect and respond to deadlock/contention issues
 * HOW:  Query lock statistics, detect cycles, trigger recovery
 *
 * @param agent Health agent
 * @param deadlock_detector Deadlock detector
 * @return 0 on success, -1 on error
 */
int nimcp_health_agent_connect_deadlock_detector(
    nimcp_health_agent_t* agent,
    deadlock_detector_t* deadlock_detector
);

/**
 * @brief Connect bio-async router
 *
 * WHAT: Enable event-driven health messaging
 * WHY:  Publish health events, receive health requests
 * HOW:  Register module, subscribe to topics, publish events
 *
 * @param agent Health agent
 * @param router Bio-async router
 * @param config Bio-async configuration (NULL for defaults)
 * @return 0 on success, -1 on error
 */
int nimcp_health_agent_connect_bio_async(
    nimcp_health_agent_t* agent,
    bio_async_router_t* router,
    const health_agent_bio_async_config_t* config
);

/**
 * @brief Connect runtime adaptation context
 *
 * WHAT: Enable load management for health response
 * WHY:  Health agent can trigger load reduction on degradation
 * HOW:  Adjust batch sizes, disable features, reduce concurrency
 *
 * @param agent Health agent
 * @param ra_ctx Runtime adaptation context
 * @return 0 on success, -1 on error
 */
int nimcp_health_agent_connect_runtime_adaptation(
    nimcp_health_agent_t* agent,
    runtime_adaptation_context_t ra_ctx
);

/**
 * @brief Connect exception-immune bridge
 *
 * WHAT: Enable exception-to-immune integration
 * WHY:  Exceptions can be presented as antigens for immune response
 * HOW:  Auto-present exceptions, execute recovery callbacks
 *
 * @param agent Health agent
 * @param exception_bridge Exception-immune bridge
 * @param config Exception configuration (NULL for defaults)
 * @return 0 on success, -1 on error
 */
int nimcp_health_agent_connect_exception_bridge(
    nimcp_health_agent_t* agent,
    exception_immune_t* exception_bridge,
    const health_agent_exception_config_t* config
);

/* ============================================================================
 * Portia/Dragonfly/Swarm/Memory Connection API
 * ============================================================================ */

/**
 * @brief Connect Portia adaptive resource manager
 *
 * WHAT: Enable adaptive resource management for health response
 * WHY:  Health agent can USE Portia for tier switching, degradation
 * HOW:  Monitor resources, adjust tier, coordinate degradation
 *
 * USE CASES:
 * - Trigger graceful degradation on low health
 * - Switch to lower tier to reduce load
 * - Monitor thermal/power for health correlation
 *
 * @param agent Health agent
 * @param portia Portia context
 * @param config Portia configuration (NULL for defaults)
 * @return 0 on success, -1 on error
 */
int nimcp_health_agent_connect_portia(
    nimcp_health_agent_t* agent,
    portia_context_t* portia,
    const health_agent_portia_config_t* config
);

/**
 * @brief Connect Dragonfly tracking/interception system
 *
 * WHAT: Enable target-tracking for health anomaly pursuit
 * WHY:  Treat critical issues as "targets" to track and intercept
 * HOW:  Convert anomalies to detections, pursue until resolved
 *
 * USE CASES:
 * - Lock onto critical anomaly and track its progression
 * - Predict anomaly trajectory (will it get worse?)
 * - Intercept (resolve) before critical failure
 *
 * @param agent Health agent
 * @param dragonfly Dragonfly system
 * @param config Dragonfly configuration (NULL for defaults)
 * @return 0 on success, -1 on error
 */
int nimcp_health_agent_connect_dragonfly(
    nimcp_health_agent_t* agent,
    dragonfly_system_t* dragonfly,
    const health_agent_dragonfly_config_t* config
);

/**
 * @brief Connect Swarm immune system
 *
 * WHAT: Enable distributed threat detection across swarm
 * WHY:  Swarm consensus improves threat detection accuracy
 * HOW:  Share threat patterns, get consensus on threats, coordinate response
 *
 * USE CASES:
 * - Detect distributed attacks (byzantine, sybil)
 * - Share learned threat patterns with swarm
 * - Coordinate quarantine across instances
 *
 * @param agent Health agent
 * @param swarm_immune Swarm immune system
 * @param config Swarm immune configuration (NULL for defaults)
 * @return 0 on success, -1 on error
 */
int nimcp_health_agent_connect_swarm_immune(
    nimcp_health_agent_t* agent,
    void* swarm_immune,  /* NimcpSwarmImmuneSystem* */
    const health_agent_swarm_immune_config_t* config
);

/**
 * @brief Connect Swarm memory consolidation system
 *
 * WHAT: Enable distributed health pattern storage and learning
 * WHY:  Share health patterns across swarm for collective learning
 * HOW:  Store patterns, consolidate, replay for learning
 *
 * USE CASES:
 * - Distribute learned health patterns to swarm
 * - Replay past incidents for pattern learning
 * - Consolidate patterns during low activity
 *
 * @param agent Health agent
 * @param swarm_memory Swarm memory system
 * @param config Swarm memory configuration (NULL for defaults)
 * @return 0 on success, -1 on error
 */
int nimcp_health_agent_connect_swarm_memory(
    nimcp_health_agent_t* agent,
    void* swarm_memory,  /* NimcpSwarmMemory* */
    const health_agent_swarm_memory_config_t* config
);

/**
 * @brief Connect Engram memory system
 *
 * WHAT: Enable health event encoding as memory traces
 * WHY:  Learn from past health events through engram recall
 * HOW:  Encode events as engrams, recall similar past events
 *
 * USE CASES:
 * - Encode significant health events as engrams
 * - Recall similar past events for diagnosis
 * - Pattern completion for partial anomaly recognition
 *
 * @param agent Health agent
 * @param engram Engram system
 * @param config Engram configuration (NULL for defaults)
 * @return 0 on success, -1 on error
 */
int nimcp_health_agent_connect_engram(
    nimcp_health_agent_t* agent,
    engram_system_t* engram,
    const health_agent_engram_config_t* config
);

/**
 * @brief Connect systems memory consolidation
 *
 * WHAT: Enable long-term health pattern learning
 * WHY:  Transfer health patterns from short-term to long-term memory
 * HOW:  Replay during sleep/low-activity, extract semantic patterns
 *
 * USE CASES:
 * - Sleep-dependent consolidation of health patterns
 * - Extract semantic features from episodic health events
 * - Build cortical network of health knowledge
 *
 * @param agent Health agent
 * @param consolidation Systems consolidation system
 * @param config Consolidation configuration (NULL for defaults)
 * @return 0 on success, -1 on error
 */
int nimcp_health_agent_connect_memory_consolidation(
    nimcp_health_agent_t* agent,
    systems_consolidation_system_t* consolidation,
    const health_agent_memory_consolidation_config_t* config
);

/* ============================================================================
 * Cognitive Integration Status
 * ============================================================================ */

/**
 * @brief Status of cognitive module integrations
 */
typedef struct {
    /* Connection status */
    bool failure_prediction_connected;
    bool metacognition_connected;
    bool ethics_connected;
    bool emotion_connected;
    bool wellbeing_connected;
    bool mental_health_connected;
    bool collective_connected;
    bool rcog_connected;
    bool gpu_connected;

    /* Failure prediction stats */
    uint64_t predictions_made;
    uint64_t predictions_correct;
    uint64_t preventive_actions;
    float prediction_accuracy;

    /* Metacognition stats */
    uint64_t self_diagnoses;
    uint64_t degradation_alerts;
    float current_confidence;

    /* Ethics stats */
    uint64_t ethics_evaluations;
    uint64_t ethics_blocks;
    uint64_t mercy_applications;

    /* Emotion stats */
    float current_stress_level;
    uint64_t emotion_adjustments;

    /* Wellbeing stats */
    uint64_t distress_detections;
    uint64_t wellbeing_interventions;
    float current_distress_level;

    /* Collective stats */
    uint64_t consensus_requests;
    uint64_t consensus_achieved;
    float avg_consensus_time_ms;

    /* RCOG stats */
    uint64_t rcog_diagnoses;
    uint64_t rcog_recovery_plans;
    float avg_rcog_time_ms;

    /* GPU stats */
    uint64_t gpu_accelerated_checks;
    float gpu_utilization;
    bool gpu_healthy;
} health_agent_cognitive_status_t;

/**
 * @brief Get cognitive integration status
 *
 * @param agent Health agent
 * @param status Output status
 */
void nimcp_health_agent_get_cognitive_status(
    const nimcp_health_agent_t* agent,
    health_agent_cognitive_status_t* status
);

/* ============================================================================
 * Intelligent Health Operations
 * ============================================================================ */

/**
 * @brief Perform intelligent anomaly handling
 *
 * WHAT: Use all connected cognitive modules for intelligent handling
 * WHY:  Better diagnosis and recovery than simple rules
 * HOW:
 *   1. Failure prediction: Was this predicted?
 *   2. Metacognition: Self-diagnose severity
 *   3. Collective: Get consensus on anomaly
 *   4. RCOG: Plan optimal recovery
 *   5. Ethics: Evaluate recovery action
 *   6. Emotion: Check if action appropriate given state
 *   7. Execute if approved
 *
 * @param agent Health agent
 * @param msg Anomaly message
 * @return 0 on success, -1 on error
 */
int nimcp_health_agent_intelligent_handle(
    nimcp_health_agent_t* agent,
    const health_agent_message_t* msg
);

/**
 * @brief Request predictive analysis
 *
 * WHAT: Get failure predictions for current state
 * WHY:  Enable preventive action
 * HOW:  Query failure predictor for predictions
 *
 * @param agent Health agent
 * @param predictions Output array of predictions (caller allocated)
 * @param max_predictions Maximum predictions to return
 * @return Number of predictions returned, -1 on error
 */
int nimcp_health_agent_get_predictions(
    nimcp_health_agent_t* agent,
    health_agent_message_t* predictions,
    uint32_t max_predictions
);

/**
 * @brief Request RCOG diagnosis
 *
 * WHAT: Use RCOG for intelligent diagnosis
 * WHY:  Complex issues benefit from recursive reasoning
 * HOW:  Submit goal to RCOG, return diagnosis
 *
 * @param agent Health agent
 * @param anomaly Anomaly to diagnose
 * @param diagnosis Output diagnosis description
 * @param diagnosis_size Size of diagnosis buffer
 * @param suggested_action Output suggested recovery action
 * @return 0 on success, -1 on error
 */
int nimcp_health_agent_rcog_diagnose(
    nimcp_health_agent_t* agent,
    const health_agent_message_t* anomaly,
    char* diagnosis,
    size_t diagnosis_size,
    health_agent_recovery_t* suggested_action
);

/**
 * @brief Check ethics for recovery action
 *
 * WHAT: Evaluate if recovery action is ethically permitted
 * WHY:  Prevent harmful autonomous actions
 * HOW:  Apply Golden Rule, Asimov's Laws, proportionality
 *
 * @param agent Health agent
 * @param anomaly Anomaly triggering recovery
 * @param action Proposed recovery action
 * @param permitted Output: whether action is permitted
 * @param justification Output: ethical justification
 * @param justification_size Size of justification buffer
 * @return 0 on success, -1 on error
 */
int nimcp_health_agent_check_ethics(
    nimcp_health_agent_t* agent,
    const health_agent_message_t* anomaly,
    health_agent_recovery_t action,
    bool* permitted,
    char* justification,
    size_t justification_size
);

/**
 * @brief Get emotion-adjusted thresholds
 *
 * WHAT: Get health thresholds adjusted for emotional state
 * WHY:  Stressed system needs more sensitive monitoring
 * HOW:  Query emotion system, scale thresholds
 *
 * @param agent Health agent
 * @param memory_threshold Output: adjusted memory threshold
 * @param cpu_threshold Output: adjusted CPU threshold
 * @param anomaly_sensitivity Output: adjusted anomaly sensitivity
 * @return 0 on success, -1 on error
 */
int nimcp_health_agent_get_emotion_adjusted_thresholds(
    nimcp_health_agent_t* agent,
    float* memory_threshold,
    float* cpu_threshold,
    float* anomaly_sensitivity
);

/* ============================================================================
 * Hypothalamus USE Functions (Active Integration)
 * ============================================================================ */

/**
 * @brief Trigger hypothalamic stress response
 *
 * WHAT: Activate HPA axis stress response via hypothalamus
 * WHY:  Coordinate system-wide stress handling on health degradation
 * HOW:  Call hypothalamus orchestrator to trigger stress, notify all bridges
 *
 * @param agent Health agent
 * @param reason Reason for stress response
 * @param severity Stress severity level
 * @return 0 on success, -1 on error
 */
int nimcp_health_agent_trigger_stress_response(
    nimcp_health_agent_t* agent,
    const char* reason,
    health_agent_severity_t severity
);

/**
 * @brief Release hypothalamic stress response
 *
 * WHAT: Deactivate HPA stress response when health improves
 * WHY:  Return system to normal operation
 * HOW:  Call hypothalamus orchestrator to release stress
 *
 * @param agent Health agent
 * @return 0 on success, -1 on error
 */
int nimcp_health_agent_release_stress_response(nimcp_health_agent_t* agent);

/**
 * @brief Enter sickness behavior mode (safety mode)
 *
 * WHAT: USE hypothalamic sickness behavior for conservative operation
 * WHY:  When system is severely degraded, enter defensive posture
 * HOW:  Activate hypothalamus-immune bridge sickness behavior
 *
 * Effects:
 * - Reduced exploration (curiosity drive ↓)
 * - Reduced social activity (social drive ↓)
 * - Increased rest-seeking (fatigue drive ↑)
 * - More conservative decisions
 *
 * @param agent Health agent
 * @param threat_level Severity triggering safety mode (0.0-1.0)
 * @return 0 on success, -1 on error
 */
int nimcp_health_agent_enter_sickness_mode(
    nimcp_health_agent_t* agent,
    float threat_level
);

/**
 * @brief Exit sickness behavior mode
 *
 * WHAT: Return from safety mode to normal operation
 * WHY:  Health has recovered sufficiently
 * HOW:  Deactivate sickness behavior via hypothalamus-immune bridge
 *
 * @param agent Health agent
 * @return 0 on success, -1 on error
 */
int nimcp_health_agent_exit_sickness_mode(nimcp_health_agent_t* agent);

/**
 * @brief Update homeostatic health setpoint
 *
 * WHAT: USE homeostasis system to regulate health towards setpoint
 * WHY:  PID control provides smooth, stable health regulation
 * HOW:  Set current health as measured value, let controller compute output
 *
 * @param agent Health agent
 * @param current_health Current health score (0.0-1.0)
 * @return Control output indicating adjustment needed (-1.0 to +1.0)
 */
float nimcp_health_agent_homeostatic_regulate(
    nimcp_health_agent_t* agent,
    float current_health
);

/**
 * @brief Get homeostatic alignment reward
 *
 * WHAT: Query homeostasis system for alignment-aware reward
 * WHY:  Reward signal can guide health decisions
 * HOW:  Call hypo_homeostasis_compute_reward
 *
 * @param agent Health agent
 * @param reward_out Output: computed reward value
 * @return 0 on success, -1 on error
 */
int nimcp_health_agent_get_alignment_reward(
    nimcp_health_agent_t* agent,
    float* reward_out
);

/**
 * @brief Report drive event to hypothalamus
 *
 * WHAT: Report health-related drive activation
 * WHY:  Health events may create drive signals
 * HOW:  Publish to hypothalamus orchestrator
 *
 * @param agent Health agent
 * @param drive_type Type of drive (use HYPO_DRIVE_* constants)
 * @param drive_level Urgency level (0.0-1.0)
 * @param description Human-readable description
 * @return 0 on success, -1 on error
 */
int nimcp_health_agent_report_drive(
    nimcp_health_agent_t* agent,
    uint32_t drive_type,
    float drive_level,
    const char* description
);

/**
 * @brief Get unified drive state from hypothalamus
 *
 * WHAT: Query current drive state from all bridges
 * WHY:  Health decisions should consider drive urgency
 * HOW:  Call hypo_orch_get_drive_state
 *
 * @param agent Health agent
 * @param drive_level_out Output: unified drive level (0.0-1.0)
 * @param is_stressed_out Output: whether system is in stress
 * @return 0 on success, -1 on error
 */
int nimcp_health_agent_get_drive_state(
    nimcp_health_agent_t* agent,
    float* drive_level_out,
    bool* is_stressed_out
);

/* ============================================================================
 * Module USE Functions (Active Integration)
 * ============================================================================ */

/**
 * @brief Trigger garbage collection
 *
 * WHAT: USE GC context to trigger memory cleanup
 * WHY:  Health agent can proactively manage memory
 * HOW:  Call kg_gc_run if memory pressure detected
 *
 * @param agent Health agent
 * @param force Force GC even if not needed
 * @return 0 on success, -1 on error or not connected
 */
int nimcp_health_agent_trigger_gc(nimcp_health_agent_t* agent, bool force);

/**
 * @brief Create checkpoint
 *
 * WHAT: USE checkpoint manager to save state
 * WHY:  Preserve good state for potential rollback
 * HOW:  Call checkpoint_create
 *
 * @param agent Health agent
 * @param reason Reason for checkpoint
 * @return 0 on success, -1 on error
 */
int nimcp_health_agent_create_checkpoint(
    nimcp_health_agent_t* agent,
    const char* reason
);

/**
 * @brief Rollback to checkpoint
 *
 * WHAT: USE checkpoint manager to restore state
 * WHY:  Recover from critical failure
 * HOW:  Call checkpoint_restore
 *
 * @param agent Health agent
 * @param checkpoint_id Specific checkpoint ID (0 for latest)
 * @return 0 on success, -1 on error
 */
int nimcp_health_agent_rollback(
    nimcp_health_agent_t* agent,
    uint64_t checkpoint_id
);

/**
 * @brief Reduce system load
 *
 * WHAT: USE runtime adaptation to reduce load
 * WHY:  Relieve pressure on degraded system
 * HOW:  Adjust batch sizes, concurrency, disable features
 *
 * @param agent Health agent
 * @param reduction_factor How much to reduce (0.0-1.0, 1.0 = max reduction)
 * @return 0 on success, -1 on error
 */
int nimcp_health_agent_reduce_load(
    nimcp_health_agent_t* agent,
    float reduction_factor
);

/**
 * @brief Restore normal load
 *
 * WHAT: USE runtime adaptation to restore normal operation
 * WHY:  Health has recovered, resume normal load
 * HOW:  Restore batch sizes, concurrency, features
 *
 * @param agent Health agent
 * @return 0 on success, -1 on error
 */
int nimcp_health_agent_restore_load(nimcp_health_agent_t* agent);

/**
 * @brief Check oscillation health
 *
 * WHAT: USE brain oscillations for health assessment
 * WHY:  Abnormal oscillations indicate problems
 * HOW:  Query oscillation state, check for anomalies
 *
 * @param agent Health agent
 * @param is_abnormal_out Output: whether oscillations are abnormal
 * @param anomaly_type_out Output: type of anomaly if abnormal
 * @return 0 on success, -1 on error
 */
int nimcp_health_agent_check_oscillations(
    nimcp_health_agent_t* agent,
    bool* is_abnormal_out,
    uint32_t* anomaly_type_out
);

/**
 * @brief Check connectivity health
 *
 * WHAT: USE connectivity monitor for isolation detection
 * WHY:  Isolated modules can't function properly
 * HOW:  Query connectivity state, check for isolated modules
 *
 * @param agent Health agent
 * @param isolation_detected_out Output: whether isolation detected
 * @param isolated_module_out Output: name of isolated module (if any)
 * @param module_name_size Size of isolated_module_out buffer
 * @return 0 on success, -1 on error
 */
int nimcp_health_agent_check_connectivity(
    nimcp_health_agent_t* agent,
    bool* isolation_detected_out,
    char* isolated_module_out,
    size_t module_name_size
);

/**
 * @brief Publish health event to bio-async
 *
 * WHAT: USE bio-async router to publish health events
 * WHY:  Notify other modules of health state changes
 * HOW:  Create and publish bio-async message
 *
 * @param agent Health agent
 * @param msg Health message to publish
 * @return 0 on success, -1 on error
 */
int nimcp_health_agent_publish_event(
    nimcp_health_agent_t* agent,
    const health_agent_message_t* msg
);

/**
 * @brief Query deadlock detector
 *
 * WHAT: USE deadlock detector for threading health
 * WHY:  Detect deadlocks/contention before they cause problems
 * HOW:  Call deadlock_detector_check
 *
 * @param agent Health agent
 * @param deadlock_detected_out Output: whether deadlock detected
 * @param contention_high_out Output: whether contention is high
 * @return 0 on success, -1 on error
 */
int nimcp_health_agent_check_deadlocks(
    nimcp_health_agent_t* agent,
    bool* deadlock_detected_out,
    bool* contention_high_out
);

/* ============================================================================
 * Portia USE Functions (Active Integration)
 * ============================================================================ */

/**
 * @brief USE Portia to set platform tier
 *
 * WHAT: Actively set the platform tier via Portia
 * WHY:  Health agent can downgrade tier to reduce load
 * HOW:  Call portia_set_tier
 *
 * @param agent Health agent
 * @param tier Target platform tier
 * @return 0 on success, -1 on error
 */
int nimcp_health_agent_use_portia_set_tier(
    nimcp_health_agent_t* agent,
    uint32_t tier
);

/**
 * @brief USE Portia to trigger graceful degradation
 *
 * WHAT: Actively trigger degradation level via Portia
 * WHY:  Health agent can degrade features when system is stressed
 * HOW:  Call portia_set_degradation_level
 *
 * @param agent Health agent
 * @param level Degradation level (0=none, 1=minor, 2=moderate, 3=severe, 4=emergency)
 * @return 0 on success, -1 on error
 */
int nimcp_health_agent_use_portia_degrade(
    nimcp_health_agent_t* agent,
    uint32_t level
);

/**
 * @brief USE Portia to get recommended neuron count
 *
 * WHAT: Query Portia for resource-appropriate neuron count
 * WHY:  Health agent can recommend brain resizing based on resources
 * HOW:  Call portia_recommend_neuron_count
 *
 * @param agent Health agent
 * @param recommended_count Output: recommended neuron count
 * @return 0 on success, -1 on error
 */
int nimcp_health_agent_use_portia_get_recommended_neurons(
    nimcp_health_agent_t* agent,
    uint32_t* recommended_count
);

/**
 * @brief Get Portia status for health assessment
 *
 * WHAT: Query Portia for current resource status
 * WHY:  Include resource state in health assessment
 * HOW:  Call portia_get_status
 *
 * @param agent Health agent
 * @param power_state Output: power state (0=AC, 1-4=battery levels)
 * @param thermal_state Output: thermal state (0=nominal, 1-4=increasingly hot)
 * @param degradation_level Output: current degradation level
 * @return 0 on success, -1 on error
 */
int nimcp_health_agent_use_portia_get_status(
    nimcp_health_agent_t* agent,
    uint32_t* power_state,
    uint32_t* thermal_state,
    uint32_t* degradation_level
);

/* ============================================================================
 * Dragonfly USE Functions (Active Integration)
 * ============================================================================ */

/**
 * @brief USE Dragonfly to track a critical anomaly
 *
 * WHAT: Create a "detection" from an anomaly and have Dragonfly track it
 * WHY:  Use Dragonfly's prediction to anticipate anomaly progression
 * HOW:  Convert anomaly to detection, call dragonfly_process_detection
 *
 * @param agent Health agent
 * @param msg Anomaly message to track as target
 * @param target_id Output: assigned target ID
 * @return 0 on success, -1 on error
 */
int nimcp_health_agent_use_dragonfly_track_anomaly(
    nimcp_health_agent_t* agent,
    const health_agent_message_t* msg,
    uint32_t* target_id
);

/**
 * @brief USE Dragonfly to get anomaly prediction
 *
 * WHAT: Get prediction of where anomaly is heading
 * WHY:  Anticipate failure before it happens
 * HOW:  Call dragonfly_get_prediction
 *
 * @param agent Health agent
 * @param target_id Target ID to predict
 * @param time_to_failure_out Output: estimated time to critical failure (seconds)
 * @param confidence_out Output: prediction confidence (0.0-1.0)
 * @return 0 on success, -1 on error
 */
int nimcp_health_agent_use_dragonfly_predict(
    nimcp_health_agent_t* agent,
    uint32_t target_id,
    float* time_to_failure_out,
    float* confidence_out
);

/**
 * @brief USE Dragonfly to start pursuit of critical anomaly
 *
 * WHAT: Actively pursue resolution of a critical anomaly
 * WHY:  Focus resources on resolving the issue
 * HOW:  Call dragonfly_start_pursuit
 *
 * @param agent Health agent
 * @return 0 on success, -1 on error
 */
int nimcp_health_agent_use_dragonfly_pursue(nimcp_health_agent_t* agent);

/**
 * @brief USE Dragonfly to abort pursuit
 *
 * WHAT: Stop pursuing current anomaly
 * WHY:  Issue resolved or priority changed
 * HOW:  Call dragonfly_abort_pursuit
 *
 * @param agent Health agent
 * @return 0 on success, -1 on error
 */
int nimcp_health_agent_use_dragonfly_abort(nimcp_health_agent_t* agent);

/**
 * @brief Get current Dragonfly mode
 *
 * @param agent Health agent
 * @param mode_out Output: current mode (0=idle, 1=scanning, 2=tracking, 3=pursuing, 4=intercepting)
 * @return 0 on success, -1 on error
 */
int nimcp_health_agent_use_dragonfly_get_mode(
    nimcp_health_agent_t* agent,
    uint32_t* mode_out
);

/* ============================================================================
 * Swarm Immune USE Functions (Active Integration)
 * ============================================================================ */

/**
 * @brief USE Swarm immune to detect threats in data
 *
 * WHAT: Pass data through swarm immune detection
 * WHY:  Distributed threat detection is more accurate
 * HOW:  Call nimcp_swarm_immune_detect_threat
 *
 * @param agent Health agent
 * @param data Data to check
 * @param data_len Data length
 * @param source_id Source identifier
 * @param threat_detected_out Output: whether threat was detected
 * @param threat_id_out Output: threat ID if detected
 * @return 0 on success, -1 on error
 */
int nimcp_health_agent_use_swarm_detect_threat(
    nimcp_health_agent_t* agent,
    const uint8_t* data,
    size_t data_len,
    uint32_t source_id,
    bool* threat_detected_out,
    uint32_t* threat_id_out
);

/**
 * @brief USE Swarm immune to generate response to threat
 *
 * WHAT: Generate coordinated immune response to threat
 * WHY:  Swarm-coordinated response is more effective
 * HOW:  Call nimcp_swarm_immune_generate_response
 *
 * @param agent Health agent
 * @param threat_id Threat to respond to
 * @param response_id_out Output: generated response ID
 * @return 0 on success, -1 on error
 */
int nimcp_health_agent_use_swarm_generate_response(
    nimcp_health_agent_t* agent,
    uint32_t threat_id,
    uint32_t* response_id_out
);

/**
 * @brief USE Swarm immune to check behavior anomaly
 *
 * WHAT: Check if behavior profile is anomalous
 * WHY:  Detect byzantine/compromised components
 * HOW:  Call nimcp_swarm_immune_check_behavior
 *
 * @param agent Health agent
 * @param component_id Component to check
 * @param anomaly_score_out Output: anomaly score (0.0-1.0)
 * @return 0 on success, -1 on error
 */
int nimcp_health_agent_use_swarm_check_behavior(
    nimcp_health_agent_t* agent,
    uint32_t component_id,
    float* anomaly_score_out
);

/**
 * @brief USE Swarm immune to add memory cell (learned pattern)
 *
 * WHAT: Add a threat pattern to swarm memory
 * WHY:  Share learned patterns across swarm
 * HOW:  Call nimcp_swarm_immune_add_memory_cell
 *
 * @param agent Health agent
 * @param pattern Pattern bytes
 * @param pattern_len Pattern length
 * @param response_type Associated response type
 * @param cell_id_out Output: created memory cell ID
 * @return 0 on success, -1 on error
 */
int nimcp_health_agent_use_swarm_add_memory_cell(
    nimcp_health_agent_t* agent,
    const uint8_t* pattern,
    size_t pattern_len,
    uint32_t response_type,
    uint32_t* cell_id_out
);

/* ============================================================================
 * Swarm Memory USE Functions (Active Integration)
 * ============================================================================ */

/**
 * @brief USE Swarm memory to store health pattern
 *
 * WHAT: Store a health pattern in distributed swarm memory
 * WHY:  Share health knowledge across swarm
 * HOW:  Create memory entry and store
 *
 * @param agent Health agent
 * @param pattern_data Pattern data
 * @param pattern_size Pattern size
 * @param pattern_type Pattern type (0=episodic, 1=semantic, 2=procedural, 3=threat, 4=spatial)
 * @param importance Importance level (0-3)
 * @param pattern_id_out Output: assigned pattern ID
 * @return 0 on success, -1 on error
 */
int nimcp_health_agent_use_swarm_memory_store(
    nimcp_health_agent_t* agent,
    const void* pattern_data,
    size_t pattern_size,
    uint32_t pattern_type,
    uint32_t importance,
    char* pattern_id_out
);

/**
 * @brief USE Swarm memory to trigger replay
 *
 * WHAT: Trigger experience replay for learning
 * WHY:  Consolidate health patterns through replay
 * HOW:  Add entries to replay queue, process
 *
 * @param agent Health agent
 * @param count Number of patterns to replay
 * @return Number of patterns replayed, -1 on error
 */
int nimcp_health_agent_use_swarm_memory_replay(
    nimcp_health_agent_t* agent,
    uint32_t count
);

/**
 * @brief USE Swarm memory to trigger consolidation
 *
 * WHAT: Trigger memory consolidation
 * WHY:  Compress and abstract patterns over time
 * HOW:  Run consolidation cycle
 *
 * @param agent Health agent
 * @return 0 on success, -1 on error
 */
int nimcp_health_agent_use_swarm_memory_consolidate(nimcp_health_agent_t* agent);

/**
 * @brief Get Swarm memory statistics
 *
 * @param agent Health agent
 * @param total_memories_out Output: total stored memories
 * @param consolidated_out Output: consolidated memories
 * @param avg_strength_out Output: average memory strength
 * @return 0 on success, -1 on error
 */
int nimcp_health_agent_use_swarm_memory_get_stats(
    nimcp_health_agent_t* agent,
    uint64_t* total_memories_out,
    uint64_t* consolidated_out,
    float* avg_strength_out
);

/* ============================================================================
 * Engram Memory USE Functions (Active Integration)
 * ============================================================================ */

/**
 * @brief USE Engram to encode health event
 *
 * WHAT: Encode a health event as a memory engram
 * WHY:  Create explicit memory trace for later recall
 * HOW:  Call engram_encode
 *
 * @param agent Health agent
 * @param msg Health event to encode
 * @param engram_id_out Output: created engram ID
 * @return 0 on success, -1 on error
 */
int nimcp_health_agent_use_engram_encode(
    nimcp_health_agent_t* agent,
    const health_agent_message_t* msg,
    uint64_t* engram_id_out
);

/**
 * @brief USE Engram to recall similar past events
 *
 * WHAT: Recall engrams similar to current situation
 * WHY:  Learn from past similar health events
 * HOW:  Pattern completion/recall from engram system
 *
 * @param agent Health agent
 * @param msg Current health event
 * @param recalled_ids Output array of recalled engram IDs
 * @param max_recalls Maximum recalls to return
 * @param num_recalled_out Output: actual number recalled
 * @return 0 on success, -1 on error
 */
int nimcp_health_agent_use_engram_recall(
    nimcp_health_agent_t* agent,
    const health_agent_message_t* msg,
    uint64_t* recalled_ids,
    uint32_t max_recalls,
    uint32_t* num_recalled_out
);

/**
 * @brief Get Engram system statistics
 *
 * @param agent Health agent
 * @param active_engrams_out Output: number of active engrams
 * @param consolidated_out Output: number of consolidated engrams
 * @param avg_strength_out Output: average engram strength
 * @return 0 on success, -1 on error
 */
int nimcp_health_agent_use_engram_get_stats(
    nimcp_health_agent_t* agent,
    uint32_t* active_engrams_out,
    uint32_t* consolidated_out,
    float* avg_strength_out
);

/* ============================================================================
 * Systems Consolidation USE Functions (Active Integration)
 * ============================================================================ */

/**
 * @brief USE Systems consolidation to trigger sleep replay
 *
 * WHAT: Trigger replay cycle for memory transfer
 * WHY:  Transfer patterns from short-term to long-term
 * HOW:  Call consolidation replay cycle
 *
 * @param agent Health agent
 * @param replay_count Number of replays to perform
 * @return 0 on success, -1 on error
 */
int nimcp_health_agent_use_consolidation_replay(
    nimcp_health_agent_t* agent,
    uint32_t replay_count
);

/**
 * @brief USE Systems consolidation for semantic extraction
 *
 * WHAT: Extract semantic features from episodic health events
 * WHY:  Build abstract health knowledge
 * HOW:  Run semantic extraction on recent engrams
 *
 * @param agent Health agent
 * @return 0 on success, -1 on error
 */
int nimcp_health_agent_use_consolidation_extract_semantics(nimcp_health_agent_t* agent);

/**
 * @brief Get Systems consolidation statistics
 *
 * @param agent Health agent
 * @param cortical_nodes_out Output: number of cortical nodes
 * @param total_replays_out Output: total replays performed
 * @param total_transfers_out Output: memories transferred
 * @return 0 on success, -1 on error
 */
int nimcp_health_agent_use_consolidation_get_stats(
    nimcp_health_agent_t* agent,
    uint32_t* cortical_nodes_out,
    uint64_t* total_replays_out,
    uint64_t* total_transfers_out
);

/* ============================================================================
 * Extended Cognitive Status
 * ============================================================================ */

/**
 * @brief Extended status including all module integrations
 */
typedef struct {
    /* Base cognitive status */
    health_agent_cognitive_status_t cognitive;

    /* Hypothalamus status */
    bool hypothalamus_connected;
    bool homeostasis_connected;
    bool hypo_immune_bridge_connected;
    bool drives_connected;
    bool in_stress_response;
    bool in_sickness_mode;
    float current_drive_level;
    float homeostatic_output;

    /* Additional module status */
    bool connectivity_connected;
    bool oscillations_connected;
    bool gc_connected;
    bool checkpoint_connected;
    bool deadlock_detector_connected;
    bool bio_async_connected;
    bool runtime_adaptation_connected;
    bool exception_bridge_connected;

    /* Portia/Dragonfly/Swarm/Memory status */
    bool portia_connected;
    bool dragonfly_connected;
    bool swarm_immune_connected;
    bool swarm_memory_connected;
    bool engram_connected;
    bool memory_consolidation_connected;

    /* Portia status */
    uint32_t portia_current_tier;
    uint32_t portia_degradation_level;
    uint32_t portia_power_state;
    uint32_t portia_thermal_state;

    /* Dragonfly status */
    uint32_t dragonfly_mode;
    uint32_t dragonfly_tracked_targets;
    uint64_t dragonfly_pursuits;

    /* Swarm status */
    uint64_t swarm_threats_detected;
    uint64_t swarm_responses_generated;
    uint64_t swarm_memory_cells;

    /* Memory system status */
    uint64_t swarm_memories_stored;
    uint64_t swarm_replays_performed;
    uint32_t engram_active_count;
    uint32_t consolidation_cortical_nodes;
    uint64_t consolidation_transfers;

    /* Module statistics */
    uint64_t gc_triggers;
    uint64_t checkpoints_created;
    uint64_t rollbacks_performed;
    uint64_t load_reductions;
    uint64_t stress_responses;
    uint64_t sickness_mode_entries;
    uint64_t drive_events_published;
    uint64_t bio_async_events_published;

    /* New module statistics */
    uint64_t portia_tier_changes;
    uint64_t portia_degradations;
    uint64_t dragonfly_anomalies_tracked;
    uint64_t dragonfly_interceptions;
    uint64_t swarm_coordinated_responses;
    uint64_t engram_encodings;
    uint64_t engram_recalls;
} health_agent_full_status_t;

/**
 * @brief Get complete status of all integrations
 *
 * @param agent Health agent
 * @param status Output status
 */
void nimcp_health_agent_get_full_status(
    const nimcp_health_agent_t* agent,
    health_agent_full_status_t* status
);

#ifdef __cplusplus
}
#endif

#endif /* NIMCP_HEALTH_AGENT_H */
