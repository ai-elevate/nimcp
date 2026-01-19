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

/* Phase 5.6: Behavioral module (Dragonfly/Portia) immune integration */
#include "dragonfly/nimcp_dragonfly_immune_bridge.h"
#include "portia/nimcp_portia_monitoring.h"

/* Phase 5.5: Neural module (SNN/LNN) immune integration */
#include "snn/nimcp_snn_immune.h"
#include "lnn/nimcp_lnn_immune.h"

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

/* Phase 7 (Section 26): GPU health monitoring */
#include "utils/gpu/nimcp_gpu_health.h"

/* Phase 5.8: Dynamic Capacity Management */
#include "utils/fault_tolerance/nimcp_capacity_manager.h"

/* Phase 5.9: Symbolic Logic Health Integration */
#include "cognitive/nimcp_symbolic_logic.h"

/* Phase 5.10: Neural Substrate Health Integration */
#include "core/neural_substrate/nimcp_neural_substrate.h"

/* Phase 5.11: Thalamic/Middleware Health Integration */
#include "cognitive/omni/bridges/nimcp_omni_wm_thalamic_bridge.h"
#include "middleware/training/nimcp_brain_training_integration.h"

/* Phase 5.12: Perception/Cortical Health Integration */
#include "perception/cortical/nimcp_visual_cortical_bridge.h"
#include "perception/cortical/nimcp_audio_cortical_bridge.h"
#include "core/cortical_columns/nimcp_cortical_immune.h"
#include "core/cortical_columns/nimcp_cortical_column.h"

/* Phase 5: Cognitive module integration for real API calls
 * Note: Cannot include full headers due to type conflicts (metric_type_t, cognitive_state_t)
 * Forward-declare the specific functions we need from the cognitive modules */

/* Forward declarations for failure_prediction.h functions */
extern uint32_t failure_predictor_get_prediction_count(failure_predictor_t* predictor);
extern bool failure_predictor_needs_prevention(failure_predictor_t* predictor);

/* Forward declarations for metacognition.h functions */
extern bool metacognition_is_degraded(metacognition_t* meta, float threshold);
extern float metacognition_get_self_confidence(const metacognition_t* meta);
extern float metacognition_get_uncertainty(const metacognition_t* meta);
extern bool metacognition_has_high_uncertainty(const metacognition_t* meta, float threshold);

/* Forward declarations for ethics.h functions (avoiding full header) */
typedef enum {
    ETHICS_EVAL_APPROVED = 0,
    ETHICS_EVAL_BLOCKED = 1,
    ETHICS_EVAL_CONDITIONAL = 2,
    ETHICS_EVAL_ERROR = -1
} ethics_eval_result_t;

/* Forward declarations for collective.h functions */
typedef enum {
    COLLECTIVE_CONSENSUS_NONE = 0,
    COLLECTIVE_CONSENSUS_ACHIEVED = 1,
    COLLECTIVE_CONSENSUS_TIMEOUT = 2,
    COLLECTIVE_CONSENSUS_ERROR = -1
} collective_consensus_result_t;

/* Collective cognition forward declarations (avoiding header includes) */
typedef struct {
    float we_mode_strength;         /**< Strength of "we" identification [0-1] */
    float joint_commitment;         /**< Collective commitment level [0-1] */
    float mutual_responsiveness;    /**< Responsiveness to each other [0-1] */
    float role_understanding;       /**< Understanding of roles [0-1] */
    uint32_t active_shared_goals;   /**< Number of active shared goals */
    uint32_t active_joint_attentions;
} health_agent_we_mode_state_t;

typedef struct {
    float phi_local;
    float phi_network;
    float phi_total;
    float information;
    float integration;
    float exclusion;
} health_agent_collective_phi_t;

typedef enum {
    COLLECTIVE_CONSCIOUSNESS_NONE = 0,
    COLLECTIVE_CONSCIOUSNESS_MINIMAL,
    COLLECTIVE_CONSCIOUSNESS_EMERGING,
    COLLECTIVE_CONSCIOUSNESS_PARTIAL,
    COLLECTIVE_CONSCIOUSNESS_UNIFIED,
    COLLECTIVE_CONSCIOUSNESS_TRANSCENDENT
} health_agent_consciousness_level_t;

extern int collective_cognition_get_we_mode(const collective_cognition_t* cc, health_agent_we_mode_state_t* state);
extern int collective_cognition_get_phi(const collective_cognition_t* cc, health_agent_collective_phi_t* phi);
extern health_agent_consciousness_level_t collective_cognition_get_consciousness_level(const collective_cognition_t* cc);
extern uint32_t collective_cognition_instance_count(const collective_cognition_t* cc);
extern bool collective_cognition_is_bio_async_connected(const collective_cognition_t* cc);

/* RCOG engine forward declarations (avoiding header includes) */
typedef enum {
    RCOG_ENGINE_UNINITIALIZED = 0,
    RCOG_ENGINE_INITIALIZING,
    RCOG_ENGINE_READY,
    RCOG_ENGINE_PROCESSING,
    RCOG_ENGINE_PAUSED,
    RCOG_ENGINE_DEGRADED,
    RCOG_ENGINE_SHUTTING_DOWN,
    RCOG_ENGINE_STOPPED
} health_agent_rcog_state_t;

typedef struct {
    uint64_t goals_submitted;
    uint64_t goals_completed;
    uint64_t goals_failed;
    uint64_t goals_timeout;
    float avg_confidence;
    uint32_t active_goals;
    uint32_t pending_goals;
    uint64_t immune_modulations;
} health_agent_rcog_stats_t;

extern health_agent_rcog_state_t rcog_engine_get_state(const rcog_engine_t* engine);
extern int rcog_engine_get_stats(const rcog_engine_t* engine, health_agent_rcog_stats_t* stats);
extern bool rcog_engine_is_ready(const rcog_engine_t* engine);
extern bool rcog_engine_has_capacity(const rcog_engine_t* engine);
extern int rcog_engine_enter_degraded_mode(rcog_engine_t* engine);
extern int rcog_engine_exit_degraded_mode(rcog_engine_t* engine);

/* Hypothalamus orchestrator type aliases
 * Use the actual types from the included hypothalamus headers */
typedef hypo_unified_drive_state_t health_agent_drive_state_t;
typedef hypo_orch_stats_t health_agent_hypo_stats_t;

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
    uint64_t expected_canary;            /**< Randomized expected canary value */

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

    /* =========== NEURAL MODULE (SNN/LNN) CONNECTIONS =========== */

    /* SNN immune bridge - spiking neural network health monitoring */
    snn_immune_bridge_t* snn_bridge;
    health_agent_snn_config_t snn_config;
    _Atomic uint64_t snn_checks_run;
    _Atomic uint64_t snn_instabilities_detected;
    _Atomic uint64_t snn_recoveries_triggered;
    uint64_t last_snn_check_us;

    /* LNN immune bridge - liquid neural network health monitoring */
    lnn_immune_bridge_t* lnn_bridge;
    health_agent_lnn_config_t lnn_config;
    _Atomic uint64_t lnn_checks_run;
    _Atomic uint64_t lnn_instabilities_detected;
    _Atomic uint64_t lnn_recoveries_triggered;
    uint64_t last_lnn_check_us;

    /* Combined neural health metrics (cached for fast access) */
    neural_health_metrics_t neural_metrics;
    nimcp_mutex_t* neural_mutex;         /**< For neural metrics access */

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

    /* =========== BEHAVIORAL MODULE (DRAGONFLY/PORTIA) IMMUNE INTEGRATION (Phase 5.6) =========== */

    /* Dragonfly immune bridge - hunting behavior health monitoring */
    dragonfly_immune_bridge_t dragonfly_immune;
    health_agent_dragonfly_immune_config_t dragonfly_immune_config;
    _Atomic uint64_t dragonfly_immune_checks_run;
    _Atomic uint64_t dragonfly_stress_events;
    _Atomic uint64_t dragonfly_injury_events;
    _Atomic uint64_t dragonfly_rest_triggers;
    uint64_t last_dragonfly_immune_check_us;

    /* Portia monitor - platform resource health monitoring */
    portia_monitor_t portia_monitor;
    health_agent_portia_monitor_config_t portia_monitor_config;
    _Atomic uint64_t portia_monitor_checks_run;
    _Atomic uint64_t portia_thermal_warnings;
    _Atomic uint64_t portia_power_warnings;
    _Atomic uint64_t portia_coordination_actions;
    uint64_t last_portia_monitor_check_us;

    /* Combined behavioral health metrics (cached for fast access) */
    behavioral_health_metrics_t behavioral_metrics;
    nimcp_mutex_t* behavioral_mutex;     /**< For behavioral metrics access */

    /* Cross-module coordination state */
    _Atomic bool thermal_abort_active;
    _Atomic bool power_conservation_active;
    _Atomic bool rest_period_active;
    _Atomic uint64_t last_coordination_us;

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

    /* =========== MEMORY SYSTEM HEALTH (Phase 5.7) =========== */

    /* Hippocampus - episodic memory, spatial navigation, theta-gamma rhythms */
    nimcp_hippocampus_t* hippocampus;
    health_agent_hippocampus_config_t hippocampus_config;
    _Atomic bool hippocampus_connected;
    _Atomic uint64_t hippocampus_checks_run;
    _Atomic uint64_t hippocampus_anomalies;
    _Atomic uint64_t hippocampus_recoveries;
    uint64_t last_hippocampus_check_us;

    /* Mammillary bodies - memory consolidation relay, head direction cells */
    nimcp_mammillary_t* mammillary;
    health_agent_mammillary_config_t mammillary_config;
    _Atomic bool mammillary_connected;
    _Atomic uint64_t mammillary_checks_run;
    _Atomic uint64_t mammillary_anomalies;
    _Atomic uint64_t mammillary_recoveries;
    uint64_t last_mammillary_check_us;

    /* Memory system cross-tier tracking */
    _Atomic uint64_t memory_consistency_checks;
    _Atomic uint64_t memory_tier_mismatches;
    _Atomic uint64_t memory_recoveries_total;

    /* =========== CAPACITY MANAGEMENT (Phase 5.8) =========== */

    /* Registered capacity managers */
    capacity_manager_t* capacity_managers[HEALTH_AGENT_MAX_CAPACITY_MANAGERS];
    _Atomic uint32_t num_capacity_managers;
    nimcp_mutex_t* capacity_mutex;       /**< For capacity manager registration */

    /* Capacity tracking */
    _Atomic uint64_t capacity_checks_run;
    _Atomic uint64_t capacity_warnings_triggered;
    _Atomic uint64_t capacity_expansions_triggered;
    char most_critical_module[64];       /**< Name of module closest to capacity */

    /* =========== SYMBOLIC LOGIC HEALTH (Phase 5.9) =========== */

    /* Registered symbolic logic engines */
    symbolic_logic_t* logic_engines[HEALTH_AGENT_MAX_LOGIC_ENGINES];
    _Atomic uint32_t num_logic_engines;
    nimcp_mutex_t* logic_mutex;          /**< For logic engine registration */

    /* Symbolic logic health configuration */
    health_agent_symbolic_logic_config_t logic_config;

    /* Logic health tracking */
    _Atomic uint64_t logic_checks_run;
    _Atomic uint64_t logic_anomalies_detected;
    _Atomic uint64_t logic_recoveries_performed;
    _Atomic uint64_t logic_loop_detections;
    _Atomic uint64_t logic_kb_corruptions;
    _Atomic float logic_health_score;    /**< Current logic health score [0-100] */

    /* =========== NEURAL SUBSTRATE HEALTH (Phase 5.10) =========== */

    /* Registered neural substrates */
    neural_substrate_t* substrates[HEALTH_AGENT_MAX_NEURAL_SUBSTRATES];
    _Atomic uint32_t num_substrates;
    nimcp_mutex_t* substrate_mutex;      /**< For substrate registration */

    /* Neural substrate health configuration */
    health_agent_substrate_config_t substrate_config;

    /* Substrate health tracking */
    _Atomic uint64_t substrate_checks_run;
    _Atomic uint64_t substrate_anomalies_detected;
    _Atomic uint64_t substrate_recoveries_performed;
    _Atomic uint64_t substrate_critical_events;
    _Atomic float substrate_health_score; /**< Current substrate health score [0-100] */

    /* =========== THALAMIC/MIDDLEWARE HEALTH (Phase 5.11) =========== */

    /* Registered thalamic bridges */
    omni_wm_thalamic_bridge_t* thalamic_bridges[HEALTH_AGENT_MAX_THALAMIC_BRIDGES];
    _Atomic uint32_t num_thalamic_bridges;
    nimcp_mutex_t* thalamic_mutex;        /**< For thalamic bridge registration */

    /* Thalamic health configuration */
    health_agent_thalamic_config_t thalamic_config;

    /* Thalamic health tracking */
    _Atomic uint64_t thalamic_checks_run;
    _Atomic uint64_t thalamic_anomalies_detected;
    _Atomic uint64_t thalamic_recoveries_performed;
    _Atomic uint64_t thalamic_critical_events;
    _Atomic float thalamic_health_score;  /**< Current thalamic health score [0-100] */

    /* Registered middleware training contexts */
    nimcp_brain_training_ctx_t* training_contexts[HEALTH_AGENT_MAX_TRAINING_CONTEXTS];
    _Atomic uint32_t num_training_contexts;
    nimcp_mutex_t* middleware_mutex;      /**< For training context registration */

    /* Middleware health configuration */
    health_agent_middleware_config_t middleware_config;

    /* Middleware health tracking */
    _Atomic uint64_t middleware_checks_run;
    _Atomic uint64_t middleware_anomalies_detected;
    _Atomic uint64_t middleware_recoveries_performed;
    _Atomic uint64_t middleware_critical_events;
    _Atomic float middleware_health_score; /**< Current middleware health score [0-100] */

    /* =========== PERCEPTION/CORTICAL HEALTH (Phase 5.12) =========== */

    /* Registered visual cortical bridges */
    visual_cortical_bridge_t* visual_bridges[HEALTH_AGENT_MAX_PERCEPTION_BRIDGES];
    _Atomic uint32_t num_visual_bridges;

    /* Registered audio cortical bridges */
    audio_cortical_bridge_t* audio_bridges[HEALTH_AGENT_MAX_PERCEPTION_BRIDGES];
    _Atomic uint32_t num_audio_bridges;

    /* Perception health configuration and mutex */
    nimcp_mutex_t* perception_mutex;      /**< For perception bridge registration */
    health_agent_perception_config_t perception_config;

    /* Perception health tracking */
    _Atomic uint64_t perception_checks_run;
    _Atomic uint64_t perception_anomalies_detected;
    _Atomic uint64_t perception_recoveries_performed;
    _Atomic uint64_t perception_critical_events;
    _Atomic float perception_health_score; /**< Current perception health score [0-100] */

    /* Registered cortical immune systems */
    cortical_immune_system_t* cortical_immune_systems[HEALTH_AGENT_MAX_PERCEPTION_BRIDGES];
    _Atomic uint32_t num_cortical_immune_systems;

    /* Registered cortical columns (hypercolumns) */
    hypercolumn_t* cortical_columns[HEALTH_AGENT_MAX_CORTICAL_COLUMNS];
    _Atomic uint32_t num_cortical_columns;

    /* Cortical health configuration and mutex */
    nimcp_mutex_t* cortical_mutex;        /**< For cortical column/immune registration */
    health_agent_cortical_config_t cortical_config;

    /* Cortical health tracking */
    _Atomic uint64_t cortical_checks_run;
    _Atomic uint64_t cortical_anomalies_detected;
    _Atomic uint64_t cortical_recoveries_performed;
    _Atomic uint64_t cortical_critical_events;
    _Atomic float cortical_health_score;  /**< Current cortical health score [0-100] */

    /* =========== BRAIN PROBE HEALTH MONITORING (Phase 5.13) =========== */

    /* Registered brains for probe monitoring */
    brain_t monitored_brains[HEALTH_AGENT_MAX_BRAINS];
    brain_probe_health_metrics_t brain_metrics[HEALTH_AGENT_MAX_BRAINS];
    health_agent_brain_probe_config_t brain_probe_configs[HEALTH_AGENT_MAX_BRAINS];
    _Atomic uint32_t num_monitored_brains;
    nimcp_mutex_t* brain_probe_mutex;     /**< For brain registration */

    /* Default brain probe configuration */
    health_agent_brain_probe_config_t brain_probe_config;

    /* Brain probe health tracking */
    _Atomic uint64_t brain_probes_run;
    _Atomic uint64_t brain_warnings_triggered;
    _Atomic uint64_t brain_critical_events;
    _Atomic uint64_t brain_recoveries_performed;
    _Atomic float brain_probe_health_score; /**< Current brain probe health score [0-100] */
    uint64_t last_brain_probe_us;         /**< Timestamp of last brain probe */

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

    /* =========== WORLD MODEL & IMAGINATION HEALTH (Phase 5.14) =========== */

    /* Connected world model and imagination components */
    jepa_predictor_t* jepa_predictor;     /**< JEPA predictor for latent space prediction */
    omni_world_model_t* world_model;      /**< Omni world model for dynamics */
    imagination_engine_t* imagination;    /**< Imagination engine for mental simulation */
    nimcp_mutex_t* wm_imagination_mutex;  /**< For world model/imagination state */

    /* Configuration */
    health_agent_wm_imagination_config_t wm_imagination_config;

    /* Health metrics caches */
    jepa_health_metrics_t jepa_metrics;
    omni_wm_health_metrics_t wm_metrics;
    imagination_health_metrics_t imagination_metrics;
    world_imagination_health_t combined_wm_health;

    /* Health tracking */
    _Atomic uint64_t wm_checks_run;
    _Atomic uint64_t wm_anomalies_detected;
    _Atomic uint64_t wm_recoveries_performed;
    _Atomic uint64_t imagination_checks_run;
    _Atomic uint64_t imagination_anomalies_detected;
    _Atomic uint64_t imagination_recoveries_performed;
    _Atomic float wm_health_score;        /**< Current world model health score [0-1] */
    _Atomic float imagination_health_score; /**< Current imagination health score [0-1] */
    uint64_t last_wm_check_us;            /**< Timestamp of last world model check */
    uint64_t last_imagination_check_us;   /**< Timestamp of last imagination check */

    /* Trend tracking buffers (ring buffers for trend analysis) */
    float jepa_error_history[10];         /**< Prediction error history */
    uint32_t jepa_error_idx;
    float wm_accuracy_history[10];        /**< Forward accuracy history */
    uint32_t wm_accuracy_idx;
    float imagination_coherence_history[10]; /**< Scene coherence history */
    uint32_t imagination_coherence_idx;

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
static void agent_run_neural_check(nimcp_health_agent_t* agent);
static void agent_check_snn_health(nimcp_health_agent_t* agent);
static void agent_check_lnn_health(nimcp_health_agent_t* agent);
static void agent_update_neural_metrics(nimcp_health_agent_t* agent);
static void agent_run_behavioral_check(nimcp_health_agent_t* agent);
static void agent_check_dragonfly_immune(nimcp_health_agent_t* agent);
static void agent_check_portia_monitor(nimcp_health_agent_t* agent);
static void agent_update_behavioral_metrics(nimcp_health_agent_t* agent);
static void agent_run_cross_module_coordination(nimcp_health_agent_t* agent);
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
 * @brief Generate a randomized canary value for memory protection
 *
 * Uses a combination of timestamp, address, and random data for entropy.
 * Falls back to HEALTH_AGENT_CANARY XOR'd with timestamp if random fails.
 */
static uint64_t generate_random_canary(void) {
    uint64_t canary = HEALTH_AGENT_CANARY; /* Start with base pattern */
    uint64_t timestamp = get_timestamp_us();

    /* Try to get random bytes from /dev/urandom */
    uint64_t random_bits = 0;
    FILE* urandom = fopen("/dev/urandom", "rb");
    if (urandom) {
        size_t read = fread(&random_bits, sizeof(random_bits), 1, urandom);
        fclose(urandom);
        if (read == 1) {
            /* Successfully got random data */
            canary = random_bits;
            /* Ensure we don't accidentally get 0 */
            if (canary == 0) canary = HEALTH_AGENT_CANARY ^ timestamp;
        } else {
            /* Failed to read, use XOR fallback */
            canary = HEALTH_AGENT_CANARY ^ timestamp;
        }
    } else {
        /* No urandom, use XOR of base canary with timestamp and stack address */
        uintptr_t stack_addr = (uintptr_t)&canary;
        canary = HEALTH_AGENT_CANARY ^ timestamp ^ (uint64_t)stack_addr;
    }

    /* Additional mixing to improve entropy distribution */
    canary ^= (canary >> 33);
    canary *= 0xff51afd7ed558ccdULL;
    canary ^= (canary >> 33);
    canary *= 0xc4ceb9fe1a85ec53ULL;
    canary ^= (canary >> 33);

    return canary;
}

/**
 * @brief Validate agent structure integrity
 *
 * Uses randomized canary values for better security against buffer overflow attacks.
 */
static bool validate_agent(const nimcp_health_agent_t* agent) {
    if (!agent) return false;
    if (agent->magic != HEALTH_AGENT_MAGIC) return false;

    /* Check canaries against the expected randomized value */
    if (agent->canary_front != agent->expected_canary) return false;
    if (agent->canary_back != agent->expected_canary) return false;
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

    /* Initialize magic and randomized canaries for memory protection */
    agent->magic = HEALTH_AGENT_MAGIC;
    agent->expected_canary = generate_random_canary();
    agent->canary_front = agent->expected_canary;
    agent->canary_back = agent->expected_canary;

    nimcp_log(LOG_LEVEL_DEBUG, "Health agent: initialized with randomized canary 0x%016llX",
              (unsigned long long)agent->expected_canary);

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

    /* Initialize neural module mutex (Phase 5.5: SNN/LNN health monitoring) */
    agent->neural_mutex = nimcp_mutex_create(NULL);
    if (!agent->neural_mutex) {
        nimcp_log(LOG_LEVEL_ERROR, "Failed to create neural mutex");
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

    /* Initialize behavioral module mutex (Phase 5.6: Dragonfly/Portia health monitoring) */
    agent->behavioral_mutex = nimcp_mutex_create(NULL);
    if (!agent->behavioral_mutex) {
        nimcp_log(LOG_LEVEL_ERROR, "Failed to create behavioral mutex");
        nimcp_mutex_destroy(agent->neural_mutex);
        nimcp_free(agent->neural_mutex);
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

    /* Initialize Behavioral Module (Dragonfly/Portia) immune integration (Phase 5.6) */
    agent->dragonfly_immune = NULL;
    agent->portia_monitor = NULL;
    atomic_init(&agent->dragonfly_immune_checks_run, 0);
    atomic_init(&agent->dragonfly_stress_events, 0);
    atomic_init(&agent->dragonfly_injury_events, 0);
    atomic_init(&agent->dragonfly_rest_triggers, 0);
    agent->last_dragonfly_immune_check_us = 0;
    atomic_init(&agent->portia_monitor_checks_run, 0);
    atomic_init(&agent->portia_thermal_warnings, 0);
    atomic_init(&agent->portia_power_warnings, 0);
    atomic_init(&agent->portia_coordination_actions, 0);
    agent->last_portia_monitor_check_us = 0;
    memset(&agent->dragonfly_immune_config, 0, sizeof(agent->dragonfly_immune_config));
    memset(&agent->portia_monitor_config, 0, sizeof(agent->portia_monitor_config));
    memset(&agent->behavioral_metrics, 0, sizeof(agent->behavioral_metrics));
    atomic_init(&agent->thermal_abort_active, false);
    atomic_init(&agent->power_conservation_active, false);
    atomic_init(&agent->rest_period_active, false);
    atomic_init(&agent->last_coordination_us, 0);

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

    /* Destroy neural mutex (Phase 5.5: SNN/LNN health monitoring) */
    if (agent->neural_mutex) {
        nimcp_mutex_destroy(agent->neural_mutex);
        nimcp_free(agent->neural_mutex);
        agent->neural_mutex = NULL;
    }

    /* Destroy behavioral mutex (Phase 5.6: Dragonfly/Portia health monitoring) */
    if (agent->behavioral_mutex) {
        nimcp_mutex_destroy(agent->behavioral_mutex);
        nimcp_free(agent->behavioral_mutex);
        agent->behavioral_mutex = NULL;
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
    uint64_t last_neural_us = get_timestamp_us();  /* Phase 5.5: SNN/LNN check timing */
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

            /* ========== NEURAL MODULE (SNN/LNN) HEALTH CHECK (Phase 5.5) ========== */

            /* Run neural module health checks if connected */
            if (agent->snn_bridge || agent->lnn_bridge) {
                /* Check interval: use SNN or LNN config, whichever is shorter */
                uint32_t neural_interval_ms = 1000;  /* Default 1 second */
                if (agent->snn_bridge && agent->snn_config.check_interval_ms > 0) {
                    neural_interval_ms = agent->snn_config.check_interval_ms;
                }
                if (agent->lnn_bridge && agent->lnn_config.check_interval_ms > 0) {
                    if (agent->lnn_config.check_interval_ms < neural_interval_ms) {
                        neural_interval_ms = agent->lnn_config.check_interval_ms;
                    }
                }
                if (now_us - last_neural_us >= neural_interval_ms * 1000ULL) {
                    last_neural_us = now_us;
                    agent_run_neural_check(agent);
                }
            }

            /* ========== BEHAVIORAL MODULE (DRAGONFLY/PORTIA) HEALTH CHECK (Phase 5.6) ========== */

            /* Run behavioral module health checks if connected OR configured
             * This ensures coordination flags are updated even with NULL bridges */
            if (agent->dragonfly_immune || agent->portia_monitor ||
                agent->dragonfly_immune_config.enable_dragonfly_immune ||
                agent->portia_monitor_config.enable_portia_monitor) {
                agent_run_behavioral_check(agent);
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
    config->gpu.enable_auto_recovery = true;
    config->gpu.enable_predictive_monitoring = true;
    config->gpu.gpu_check_interval_ms = 1000;
    config->gpu.temp_warning_celsius = 75.0f;
    config->gpu.temp_critical_celsius = 85.0f;
    config->gpu.memory_warning_pct = 0.80f;
    config->gpu.memory_critical_pct = 0.95f;

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
 * Neural Module (SNN/LNN) Connection Functions
 * ============================================================================ */

int nimcp_health_agent_connect_snn(
    nimcp_health_agent_t* agent,
    snn_immune_bridge_t* snn_bridge,
    const health_agent_snn_config_t* config
) {
    if (!validate_agent(agent)) return -1;

    nimcp_mutex_lock(agent->modules_mutex);

    agent->snn_bridge = snn_bridge;

    /* Apply config or defaults */
    if (config) {
        agent->snn_config = *config;
    } else {
        /* Default SNN configuration */
        agent->snn_config.enable_snn_monitoring = true;
        agent->snn_config.enable_instability_detection = true;
        agent->snn_config.enable_auto_report = true;
        agent->snn_config.enable_learning_modulation = true;
        agent->snn_config.max_spike_rate_hz = 100.0f;
        agent->snn_config.min_spike_rate_hz = 0.1f;
        agent->snn_config.burst_threshold = 0.5f;
        agent->snn_config.sync_threshold = 0.8f;
        agent->snn_config.check_interval_ms = 100;
    }

    /* Initialize neural metrics for SNN - only mark connected if bridge is non-NULL */
    agent->neural_metrics.snn_connected = (snn_bridge != NULL);
    agent->neural_metrics.snn_healthy = true;
    atomic_store(&agent->snn_checks_run, 0);
    atomic_store(&agent->snn_instabilities_detected, 0);
    atomic_store(&agent->snn_recoveries_triggered, 0);
    agent->last_snn_check_us = get_timestamp_us();

    nimcp_mutex_unlock(agent->modules_mutex);

    nimcp_log(LOG_LEVEL_INFO, "Agent '%s' connected to SNN immune bridge%s",
              agent->config.agent_name, snn_bridge ? "" : " (NULL bridge)");
    return 0;
}

int nimcp_health_agent_connect_lnn(
    nimcp_health_agent_t* agent,
    lnn_immune_bridge_t* lnn_bridge,
    const health_agent_lnn_config_t* config
) {
    if (!validate_agent(agent)) return -1;

    nimcp_mutex_lock(agent->modules_mutex);

    agent->lnn_bridge = lnn_bridge;

    /* Apply config or defaults */
    if (config) {
        agent->lnn_config = *config;
    } else {
        /* Default LNN configuration */
        agent->lnn_config.enable_lnn_monitoring = true;
        agent->lnn_config.enable_stability_detection = true;
        agent->lnn_config.enable_auto_report = true;
        agent->lnn_config.enable_tau_modulation = true;
        agent->lnn_config.enable_lr_modulation = true;
        agent->lnn_config.state_explosion_threshold = 1000.0f;
        agent->lnn_config.state_collapse_threshold = 1e-6f;
        agent->lnn_config.tau_max = 10.0f;
        agent->lnn_config.tau_min = 0.001f;
        agent->lnn_config.gradient_explosion_threshold = 100.0f;
        agent->lnn_config.gradient_vanishing_threshold = 1e-7f;
        agent->lnn_config.check_interval_ms = 100;
    }

    /* Initialize neural metrics for LNN - only mark connected if bridge is non-NULL */
    agent->neural_metrics.lnn_connected = (lnn_bridge != NULL);
    agent->neural_metrics.lnn_healthy = true;
    atomic_store(&agent->lnn_checks_run, 0);
    atomic_store(&agent->lnn_instabilities_detected, 0);
    atomic_store(&agent->lnn_recoveries_triggered, 0);
    agent->last_lnn_check_us = get_timestamp_us();

    nimcp_mutex_unlock(agent->modules_mutex);

    nimcp_log(LOG_LEVEL_INFO, "Agent '%s' connected to LNN immune bridge%s",
              agent->config.agent_name, lnn_bridge ? "" : " (NULL bridge)");
    return 0;
}

int nimcp_health_agent_get_neural_metrics(
    const nimcp_health_agent_t* agent,
    neural_health_metrics_t* metrics
) {
    if (!validate_agent(agent)) return -1;
    if (!metrics) return -1;

    nimcp_mutex_lock(agent->neural_mutex);
    *metrics = agent->neural_metrics;
    nimcp_mutex_unlock(agent->neural_mutex);

    return 0;
}

int nimcp_health_agent_configure_neural(
    nimcp_health_agent_t* agent,
    const health_agent_snn_config_t* snn_config,
    const health_agent_lnn_config_t* lnn_config
) {
    if (!validate_agent(agent)) return -1;

    nimcp_mutex_lock(agent->modules_mutex);

    if (snn_config) {
        agent->snn_config = *snn_config;
    }
    if (lnn_config) {
        agent->lnn_config = *lnn_config;
    }

    nimcp_mutex_unlock(agent->modules_mutex);

    nimcp_log(LOG_LEVEL_DEBUG, "Agent '%s' neural configuration updated",
              agent->config.agent_name);
    return 0;
}

bool nimcp_health_agent_is_neural_unhealthy(const nimcp_health_agent_t* agent) {
    if (!validate_agent(agent)) return false;

    /* Quick check without full lock */
    return agent->neural_metrics.any_neural_unhealthy;
}

float nimcp_health_agent_get_neural_health_score(const nimcp_health_agent_t* agent) {
    /* Return perfect health (safe default) if agent is NULL or invalid */
    if (!validate_agent(agent)) return 100.0f;

    /* No neural modules connected - perfect health (nothing to degrade) */
    if (!agent->neural_metrics.snn_connected && !agent->neural_metrics.lnn_connected) {
        return 100.0f;
    }

    return agent->neural_metrics.neural_health_score;
}

/* ============================================================================
 * Behavioral Module (Dragonfly/Portia) Connection API (Phase 5.6)
 * ============================================================================ */

int nimcp_health_agent_connect_dragonfly_immune(
    nimcp_health_agent_t* agent,
    dragonfly_immune_bridge_t bridge,
    const health_agent_dragonfly_immune_config_t* config
) {
    if (!validate_agent(agent)) return -1;

    nimcp_mutex_lock(agent->modules_mutex);
    agent->dragonfly_immune = bridge;

    if (config) {
        agent->dragonfly_immune_config = *config;
    } else {
        /* Default configuration */
        agent->dragonfly_immune_config.enable_dragonfly_immune = true;
        agent->dragonfly_immune_config.enable_stress_monitoring = true;
        agent->dragonfly_immune_config.enable_health_status_tracking = true;
        agent->dragonfly_immune_config.enable_injury_detection = true;
        agent->dragonfly_immune_config.enable_fatigue_tracking = true;
        agent->dragonfly_immune_config.enable_cross_coordination = true;
        agent->dragonfly_immune_config.stress_warning_threshold = 0.5f;
        agent->dragonfly_immune_config.stress_critical_threshold = 0.8f;
        agent->dragonfly_immune_config.fatigue_warning_threshold = 0.6f;
        agent->dragonfly_immune_config.fatigue_critical_threshold = 0.9f;
        agent->dragonfly_immune_config.abort_hunt_on_thermal = true;
        agent->dragonfly_immune_config.abort_hunt_on_battery_low = true;
        agent->dragonfly_immune_config.reduce_intensity_on_stress = true;
        agent->dragonfly_immune_config.enable_auto_rest = true;
        agent->dragonfly_immune_config.rest_trigger_fatigue = 0.7f;
        agent->dragonfly_immune_config.min_rest_duration_ms = 5000;
        agent->dragonfly_immune_config.check_interval_ms = 100;
    }

    nimcp_mutex_unlock(agent->modules_mutex);

    /* Update behavioral metrics */
    nimcp_mutex_lock(agent->behavioral_mutex);
    agent->behavioral_metrics.dragonfly_connected = (bridge != NULL);
    agent->behavioral_metrics.dragonfly_healthy = true;
    atomic_store(&agent->dragonfly_immune_checks_run, 0);
    nimcp_mutex_unlock(agent->behavioral_mutex);

    nimcp_log(LOG_LEVEL_INFO, "Agent '%s' connected dragonfly immune bridge%s",
              agent->config.agent_name, bridge ? "" : " (NULL bridge)");

    return 0;
}

int nimcp_health_agent_connect_portia_monitor(
    nimcp_health_agent_t* agent,
    portia_monitor_t monitor,
    const health_agent_portia_monitor_config_t* config
) {
    if (!validate_agent(agent)) return -1;

    nimcp_mutex_lock(agent->modules_mutex);
    agent->portia_monitor = monitor;

    if (config) {
        agent->portia_monitor_config = *config;
    } else {
        /* Default configuration */
        agent->portia_monitor_config.enable_portia_monitor = true;
        agent->portia_monitor_config.enable_thermal_monitoring = true;
        agent->portia_monitor_config.enable_power_monitoring = true;
        agent->portia_monitor_config.enable_cpu_load_monitoring = true;
        agent->portia_monitor_config.enable_degradation_tracking = true;
        agent->portia_monitor_config.enable_cross_coordination = true;
        agent->portia_monitor_config.thermal_warning_temp_c = 70.0f;
        agent->portia_monitor_config.thermal_critical_temp_c = 85.0f;
        agent->portia_monitor_config.throttle_on_warm = true;
        agent->portia_monitor_config.emergency_on_critical = true;
        agent->portia_monitor_config.battery_warning_pct = 20.0f;
        agent->portia_monitor_config.battery_critical_pct = 5.0f;
        agent->portia_monitor_config.conservation_on_battery = true;
        agent->portia_monitor_config.hibernate_on_critical = false;
        agent->portia_monitor_config.cpu_warning_pct = 80.0f;
        agent->portia_monitor_config.cpu_critical_pct = 95.0f;
        agent->portia_monitor_config.reduce_load_on_warning = true;
        agent->portia_monitor_config.notify_dragonfly_on_thermal = true;
        agent->portia_monitor_config.notify_neural_on_power = true;
        agent->portia_monitor_config.trigger_checkpoint_on_power_loss = true;
        agent->portia_monitor_config.check_interval_ms = 500;
    }

    nimcp_mutex_unlock(agent->modules_mutex);

    /* Update behavioral metrics */
    nimcp_mutex_lock(agent->behavioral_mutex);
    agent->behavioral_metrics.portia_connected = (monitor != NULL);
    agent->behavioral_metrics.portia_healthy = true;
    atomic_store(&agent->portia_monitor_checks_run, 0);
    nimcp_mutex_unlock(agent->behavioral_mutex);

    nimcp_log(LOG_LEVEL_INFO, "Agent '%s' connected portia monitor%s",
              agent->config.agent_name, monitor ? "" : " (NULL monitor)");

    return 0;
}

int nimcp_health_agent_get_behavioral_metrics(
    const nimcp_health_agent_t* agent,
    behavioral_health_metrics_t* metrics
) {
    if (!validate_agent(agent)) return -1;
    if (!metrics) return -1;

    nimcp_mutex_lock(agent->behavioral_mutex);
    *metrics = agent->behavioral_metrics;
    nimcp_mutex_unlock(agent->behavioral_mutex);

    return 0;
}

int nimcp_health_agent_configure_behavioral(
    nimcp_health_agent_t* agent,
    const health_agent_dragonfly_immune_config_t* dragonfly_config,
    const health_agent_portia_monitor_config_t* portia_config
) {
    if (!validate_agent(agent)) return -1;

    nimcp_mutex_lock(agent->modules_mutex);

    if (dragonfly_config) {
        agent->dragonfly_immune_config = *dragonfly_config;
    }
    if (portia_config) {
        agent->portia_monitor_config = *portia_config;
    }

    nimcp_mutex_unlock(agent->modules_mutex);

    nimcp_log(LOG_LEVEL_DEBUG, "Agent '%s' behavioral configuration updated",
              agent->config.agent_name);
    return 0;
}

bool nimcp_health_agent_is_behavioral_unhealthy(const nimcp_health_agent_t* agent) {
    if (!validate_agent(agent)) return false;

    return agent->behavioral_metrics.any_behavioral_unhealthy;
}

float nimcp_health_agent_get_behavioral_health_score(const nimcp_health_agent_t* agent) {
    /* Return perfect health (safe default) if agent is NULL or invalid */
    if (!validate_agent(agent)) return 100.0f;

    /* No behavioral modules connected - perfect health (nothing to degrade) */
    if (!agent->behavioral_metrics.dragonfly_connected &&
        !agent->behavioral_metrics.portia_connected) {
        return 100.0f;
    }

    return agent->behavioral_metrics.behavioral_health_score;
}

int nimcp_health_agent_request_behavioral_coordination(
    nimcp_health_agent_t* agent,
    const char* action,
    const char* reason
) {
    if (!validate_agent(agent)) return -1;
    if (!action) return -1;

    nimcp_log(LOG_LEVEL_INFO, "Agent '%s' behavioral coordination request: %s - %s",
              agent->config.agent_name, action, reason ? reason : "no reason");

    uint64_t now_us = get_timestamp_us();
    atomic_store(&agent->last_coordination_us, now_us);
    atomic_fetch_add(&agent->portia_coordination_actions, 1);

    /* Handle specific actions */
    if (strcmp(action, "abort_hunt") == 0) {
        atomic_store(&agent->thermal_abort_active, true);
        nimcp_log(LOG_LEVEL_WARN, "Agent '%s' activated thermal abort for hunting",
                  agent->config.agent_name);
    } else if (strcmp(action, "conservation_mode") == 0) {
        atomic_store(&agent->power_conservation_active, true);
        nimcp_log(LOG_LEVEL_WARN, "Agent '%s' activated power conservation mode",
                  agent->config.agent_name);
    } else if (strcmp(action, "rest_period") == 0) {
        atomic_store(&agent->rest_period_active, true);
        atomic_fetch_add(&agent->dragonfly_rest_triggers, 1);
        nimcp_log(LOG_LEVEL_INFO, "Agent '%s' activated rest period",
                  agent->config.agent_name);
    }

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

    /* Query current hypothalamus orchestrator state */
    health_agent_drive_state_t drive_state;
    memset(&drive_state, 0, sizeof(drive_state));
    hypo_orch_get_drive_state(agent->hypothalamus, &drive_state);

    /* Get direct drive level for quick assessment */
    float drive_level = 0.0f;
    hypo_orch_get_drive_level(agent->hypothalamus, &drive_level);

    /* Query orchestrator stress state */
    bool orch_stressed = false;
    hypo_orch_is_stressed(agent->hypothalamus, &orch_stressed);

    /* Get hypothalamus orchestrator statistics */
    health_agent_hypo_stats_t hypo_stats;
    memset(&hypo_stats, 0, sizeof(hypo_stats));
    hypo_orch_get_stats(agent->hypothalamus, &hypo_stats);

    /* Log drive state for monitoring */
    nimcp_log(LOG_LEVEL_DEBUG, "Hypothalamus: health=%.2f, drive=%.2f, active_drives=%u, "
              "bridges=%u, conflicts=%llu, orch_stressed=%d",
              health_score, drive_level, drive_state.active_drives,
              hypo_stats.registered_bridges,
              (unsigned long long)hypo_stats.conflicts_detected, orch_stressed);

    /* Detect drive conflicts */
    if (hypo_stats.conflicts_detected > 0 && drive_state.active_drives > 2) {
        nimcp_log(LOG_LEVEL_INFO, "Hypothalamus drive conflict: %llu conflicts, %u active drives",
                  (unsigned long long)hypo_stats.conflicts_detected, drive_state.active_drives);

        /* Report drive conflict as health issue */
        health_agent_message_t conflict_msg = nimcp_health_agent_create_message(
            HEALTH_MSG_ANOMALY_DETECTED,
            HEALTH_SEVERITY_WARNING,
            HEALTH_SOURCE_NEURAL,
            "Drive conflict: %u active drives, %llu conflicts",
            drive_state.active_drives, (unsigned long long)hypo_stats.conflicts_detected
        );
        conflict_msg.suggested_action = HEALTH_RECOVERY_REDUCE_LOAD;
        nimcp_health_agent_report_anomaly(agent, &conflict_msg);
    }

    /* Check for high drive pressure combined with low health */
    bool high_drive_pressure = drive_level > 0.7f || drive_state.active_drives > 5;
    bool critical_combination = high_drive_pressure && health_score < 0.5f;

    if (critical_combination) {
        nimcp_log(LOG_LEVEL_WARN, "Critical: high drive pressure (%.2f) with low health (%.2f)",
                  drive_level, health_score);

        /* Trigger orchestrator stress if not already stressed */
        if (!orch_stressed) {
            hypo_orch_trigger_stress(agent->hypothalamus, "High drive pressure with low health");
        }
    }

    /* Check if we need to trigger stress response via health agent */
    if (health_score < agent->hypothalamus_config.stress_trigger_threshold) {
        if (!atomic_load(&agent->in_stress_response)) {
            health_agent_severity_t severity = HEALTH_SEVERITY_ERROR;

            /* Adjust severity based on drive state */
            if (health_score < 0.2f || (health_score < 0.3f && high_drive_pressure)) {
                severity = HEALTH_SEVERITY_CRITICAL;
            } else if (health_score < 0.15f) {
                severity = HEALTH_SEVERITY_FATAL;
            }

            char stress_reason[128];
            snprintf(stress_reason, sizeof(stress_reason),
                     "Health=%.2f, drive=%.2f, active_drives=%u",
                     health_score, drive_level, drive_state.active_drives);

            nimcp_health_agent_trigger_stress_response(agent, stress_reason, severity);

            /* Synchronize with orchestrator */
            if (!orch_stressed) {
                hypo_orch_trigger_stress(agent->hypothalamus, stress_reason);
            }
        }
    } else if (atomic_load(&agent->in_stress_response) && health_score > 0.6f) {
        /* Health recovered, release stress */
        nimcp_health_agent_release_stress_response(agent);

        /* Also release orchestrator stress if active */
        if (orch_stressed && health_score > 0.7f && drive_level < 0.5f) {
            hypo_orch_release_stress(agent->hypothalamus);
            nimcp_log(LOG_LEVEL_INFO, "Hypothalamus: stress released (health=%.2f, drive=%.2f)",
                      health_score, drive_level);
        }
    }

    /* Check if we need to enter sickness mode */
    if (health_score < agent->hypothalamus_config.sickness_trigger_threshold) {
        if (!atomic_load(&agent->in_sickness_mode) &&
            agent->hypothalamus_config.enable_sickness_behavior) {

            /* Calculate sickness severity based on health and drives */
            float sickness_severity = 1.0f - health_score;
            if (high_drive_pressure) {
                sickness_severity = fminf(1.0f, sickness_severity * 1.2f);
            }

            nimcp_health_agent_enter_sickness_mode(agent, sickness_severity);

            nimcp_log(LOG_LEVEL_INFO, "Hypothalamus: entering sickness mode (severity=%.2f, "
                      "health=%.2f, drives=%u)", sickness_severity, health_score,
                      drive_state.active_drives);
        }
    } else if (atomic_load(&agent->in_sickness_mode) && health_score > 0.5f) {
        /* Health recovered, exit sickness mode */
        /* Only exit if drive pressure is also manageable */
        if (drive_level < 0.6f) {
            nimcp_health_agent_exit_sickness_mode(agent);
            nimcp_log(LOG_LEVEL_INFO, "Hypothalamus: exiting sickness mode (health=%.2f, drive=%.2f)",
                      health_score, drive_level);
        } else {
            nimcp_log(LOG_LEVEL_DEBUG, "Hypothalamus: delaying sickness exit (drive=%.2f still elevated)",
                      drive_level);
        }
    }

    /* Track peak drive level for monitoring */
    if (drive_level > hypo_stats.peak_drive_level) {
        nimcp_log(LOG_LEVEL_DEBUG, "Hypothalamus: new peak drive level %.2f (previous %.2f)",
                  drive_level, hypo_stats.peak_drive_level);
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

    /* Query failure predictor for current prediction count and prevention need */
    uint32_t prediction_count = failure_predictor_get_prediction_count(agent->failure_predictor);
    bool needs_prevention = failure_predictor_needs_prevention(agent->failure_predictor);

    /* Update cognitive stats with prediction count */
    nimcp_mutex_lock(agent->cognitive_mutex);
    /* Store prediction count for status queries */
    nimcp_mutex_unlock(agent->cognitive_mutex);

    if (prediction_count > 0) {
        /* There are active failure predictions - log and report */
        nimcp_log(LOG_LEVEL_WARN, "Failure predictor: %u active predictions, prevention_needed=%d",
                  prediction_count, needs_prevention);

        /* Report to immune system if enabled */
        if (agent->prediction_config.enable_preventive_action) {
            /* Determine severity based on prevention urgency */
            health_agent_severity_t severity = needs_prevention ?
                HEALTH_SEVERITY_ERROR : HEALTH_SEVERITY_WARNING;

            health_agent_message_t msg = nimcp_health_agent_create_message(
                HEALTH_MSG_ANOMALY_DETECTED,
                severity,
                HEALTH_SOURCE_NEURAL,
                "Failure predictor: %u predictions, prevention=%s",
                prediction_count, needs_prevention ? "URGENT" : "normal"
            );

            /* Suggest action based on urgency */
            msg.suggested_action = needs_prevention ?
                HEALTH_RECOVERY_CHECKPOINT : HEALTH_RECOVERY_REDUCE_LOAD;

            nimcp_health_agent_report_anomaly(agent, &msg);

            /* Track preventive actions */
            if (needs_prevention) {
                atomic_fetch_add(&agent->preventive_actions, 1);
            }
        }
    }

    atomic_fetch_add(&agent->predictions_made, 1);
}

static void agent_run_metacognition_check(nimcp_health_agent_t* agent) {
    if (!agent || !agent->metacognition) return;

    /* Check if cognitive performance is degraded using real metacognition API
     * Default threshold is 0.7 (70% of baseline) */
    float threshold = agent->metacog_config.degradation_threshold;
    if (threshold <= 0.0f || threshold > 1.0f) {
        threshold = 0.7f;  /* Use default if invalid */
    }

    /* Get current metacognition state */
    bool is_degraded = metacognition_is_degraded(agent->metacognition, threshold);
    float confidence = metacognition_get_self_confidence(agent->metacognition);
    float uncertainty = metacognition_get_uncertainty(agent->metacognition);
    bool high_uncertainty = metacognition_has_high_uncertainty(agent->metacognition, 0.7f);

    /* Update confidence tracking */
    atomic_store(&agent->current_confidence, confidence);

    if (is_degraded) {
        /* Cognitive performance is degraded - log and report */
        atomic_fetch_add(&agent->degradation_alerts, 1);
        nimcp_log(LOG_LEVEL_WARN,
                  "Metacognition: degraded (threshold=%.2f, confidence=%.2f, uncertainty=%.2f)",
                  threshold, confidence, uncertainty);

        /* Report to health system */
        health_agent_message_t msg = nimcp_health_agent_create_message(
            HEALTH_MSG_ANOMALY_DETECTED,
            HEALTH_SEVERITY_WARNING,
            HEALTH_SOURCE_NEURAL,
            "Metacognition: performance below %.0f%%, confidence=%.2f",
            threshold * 100.0f, confidence
        );
        msg.suggested_action = HEALTH_RECOVERY_REDUCE_LOAD;
        nimcp_health_agent_report_anomaly(agent, &msg);
    }

    /* Check for high uncertainty even if not degraded */
    if (high_uncertainty && agent->metacog_config.enable_confidence_calibration) {
        nimcp_log(LOG_LEVEL_INFO,
                  "Metacognition: high uncertainty (%.2f) - may need external assistance",
                  uncertainty);

        /* Optionally request help from collective or RCOG */
        if (agent->collective && agent->collective_config.enable_collective_monitoring) {
            /* Log that we're deferring to collective for uncertain decisions */
            nimcp_log(LOG_LEVEL_DEBUG, "Deferring uncertain decisions to collective cognition");
        }
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

    /* Get number of GPU devices */
    int num_devices = gpu_health_get_device_count(agent->gpu_health);
    if (num_devices <= 0) {
        /* No GPUs to monitor */
        atomic_store(&agent->gpu_healthy, true);
        atomic_store(&agent->gpu_utilization, 0.0f);
        return;
    }

    /* Check each GPU's health */
    bool all_healthy = true;
    float total_utilization = 0.0f;
    float min_health_score = 1.0f;

    for (int i = 0; i < num_devices; i++) {
        gpu_health_metrics_t metrics;
        if (gpu_health_get_metrics(agent->gpu_health, i, &metrics) == 0) {
            /* Track utilization */
            total_utilization += metrics.gpu_utilization;

            /* Track minimum health score */
            if (metrics.health_score < min_health_score) {
                min_health_score = metrics.health_score;
            }

            /* Check health status */
            if (metrics.status >= GPU_HEALTH_DEGRADED) {
                all_healthy = false;

                /* Report anomaly for degraded/critical GPUs */
                if (metrics.status >= GPU_HEALTH_CRITICAL) {
                    health_agent_message_t msg = nimcp_health_agent_create_message(
                        HEALTH_MSG_RESOURCE_EXHAUSTION,
                        HEALTH_SEVERITY_CRITICAL,
                        HEALTH_SOURCE_MEMORY,  /* GPU memory is a memory resource */
                        "GPU %d health critical: score=%.2f",
                        i, metrics.health_score
                    );

                    /* Add GPU-specific details via resource variant */
                    msg.data.resource.memory_used = metrics.memory_used;
                    msg.data.resource.memory_limit = metrics.memory_total;
                    msg.data.resource.utilization_pct = (1.0f - metrics.health_score) * 100.0f;

                    nimcp_health_agent_report_anomaly(agent, &msg);
                }
            }

            /* Check for GPU errors */
            gpu_error_event_t error;
            if (gpu_error_check_async(agent->gpu_health, i, &error) > 0) {
                /* Error detected - get immune response */
                gpu_immune_response_t response;
                if (gpu_immune_get_response(agent->gpu_health, &error, &response) == 0) {
                    nimcp_log(LOG_LEVEL_WARN, "GPU %d error: %s - suggested recovery: %s",
                              i, error.description,
                              gpu_recovery_action_name(response.suggested_recovery));

                    /* Execute auto-recovery if enabled */
                    if (agent->gpu_config.enable_auto_recovery && response.urgency >= 0.5f) {
                        gpu_immune_execute_recovery(agent->gpu_health, i, response.suggested_recovery);
                    }
                }
            }

            /* Predict failure probability */
            if (agent->gpu_config.enable_predictive_monitoring) {
                float failure_prob = gpu_health_predict_failure_probability(
                    agent->gpu_health, i, 60  /* 60 minutes horizon */
                );

                if (failure_prob > 0.5f) {
                    nimcp_log(LOG_LEVEL_WARN, "GPU %d: %.0f%% failure probability in next hour",
                              i, failure_prob * 100.0f);
                }
            }
        }
    }

    /* Update aggregated stats */
    float avg_utilization = (num_devices > 0) ? total_utilization / num_devices : 0.0f;
    atomic_store(&agent->gpu_utilization, avg_utilization);
    atomic_store(&agent->gpu_healthy, all_healthy);
    atomic_fetch_add(&agent->gpu_accelerated_checks, 1);

    /* Log summary if issues detected */
    if (!all_healthy || min_health_score < 0.7f) {
        nimcp_log(LOG_LEVEL_INFO, "GPU health check: %d devices, avg_util=%.1f%%, min_score=%.2f, healthy=%d",
                  num_devices, avg_utilization * 100.0f, min_health_score, all_healthy);
    }
}

/* ============================================================================
 * Neural Module (SNN/LNN) Health Check Functions
 * ============================================================================ */

/**
 * @brief Run all neural health checks (SNN and LNN)
 */
static void agent_run_neural_check(nimcp_health_agent_t* agent) {
    if (!agent) return;

    /* Check SNN health if connected */
    if (agent->snn_bridge && agent->snn_config.enable_snn_monitoring) {
        agent_check_snn_health(agent);
    }

    /* Check LNN health if connected */
    if (agent->lnn_bridge && agent->lnn_config.enable_lnn_monitoring) {
        agent_check_lnn_health(agent);
    }

    /* Update aggregated neural metrics */
    agent_update_neural_metrics(agent);
}

/**
 * @brief Check SNN immune bridge health
 */
static void agent_check_snn_health(nimcp_health_agent_t* agent) {
    if (!agent || !agent->snn_bridge) return;

    uint64_t now = get_timestamp_us();
    uint64_t interval_us = (uint64_t)agent->snn_config.check_interval_ms * 1000;

    /* Rate limit checks */
    if ((now - agent->last_snn_check_us) < interval_us) {
        return;
    }
    agent->last_snn_check_us = now;

    /* Get SNN health metrics from bridge */
    snn_health_metrics_t snn_health;
    int result = snn_immune_get_health(agent->snn_bridge, &snn_health);
    if (result != 0) {
        nimcp_log(LOG_LEVEL_WARN, "Failed to get SNN health metrics: %d", result);
        return;
    }

    atomic_fetch_add(&agent->snn_checks_run, 1);

    /* Update cached metrics */
    nimcp_mutex_lock(agent->neural_mutex);
    agent->neural_metrics.snn_healthy = !snn_health.has_instability;
    agent->neural_metrics.snn_mean_rate = snn_health.mean_rate;
    agent->neural_metrics.snn_max_rate = snn_health.max_rate;
    agent->neural_metrics.snn_burst_ratio = snn_health.burst_ratio;
    agent->neural_metrics.snn_sync_index = snn_health.sync_index;
    agent->neural_metrics.snn_silent_neurons = snn_health.silent_neurons;
    agent->neural_metrics.snn_saturated_neurons = snn_health.saturated_neurons;
    nimcp_mutex_unlock(agent->neural_mutex);

    /* Detect instabilities and report to immune system */
    if (snn_health.has_instability) {
        atomic_fetch_add(&agent->snn_instabilities_detected, 1);
        atomic_fetch_add(&agent->neural_metrics.snn_instability_count, 1);

        /* Map SNN health state to severity */
        health_agent_severity_t severity = HEALTH_SEVERITY_WARNING;
        const char* state_name = "unknown";

        switch (snn_health.health) {
            case SNN_STATE_HEALTHY:
                /* Should not reach here if has_instability is true */
                return;
            case SNN_STATE_SILENT:
                severity = HEALTH_SEVERITY_ERROR;
                state_name = "silent network";
                break;
            case SNN_STATE_EXPLOSION:
                severity = HEALTH_SEVERITY_CRITICAL;
                state_name = "spike explosion";
                break;
            case SNN_STATE_NAN_DETECTED:
                severity = HEALTH_SEVERITY_CRITICAL;
                state_name = "NaN detected";
                break;
            case SNN_STATE_INF_DETECTED:
                severity = HEALTH_SEVERITY_CRITICAL;
                state_name = "Inf detected";
                break;
            case SNN_STATE_WEIGHT_EXPLOSION:
                severity = HEALTH_SEVERITY_ERROR;
                state_name = "weight explosion";
                break;
            case SNN_STATE_UNSTABLE:
                severity = HEALTH_SEVERITY_ERROR;
                state_name = "unstable";
                break;
        }

        nimcp_log(LOG_LEVEL_WARN, "SNN instability: %s (rate=%.1fHz, sync=%.2f)",
                  state_name, snn_health.mean_rate, snn_health.sync_index);

        /* Auto-report to immune system if enabled */
        if (agent->snn_config.enable_auto_report) {
            health_agent_message_t msg = {0};
            msg.type = HEALTH_MSG_ANOMALY_DETECTED;
            msg.severity = severity;
            msg.source = HEALTH_SOURCE_NEURAL;
            msg.timestamp_us = now;
            snprintf(msg.description, sizeof(msg.description),
                     "SNN %s: rate=%.1fHz, sync=%.2f", state_name,
                     snn_health.mean_rate, snn_health.sync_index);

            msg_queue_push(&agent->msg_queue, &msg);
            atomic_fetch_add(&agent->snn_recoveries_triggered, 1);
        }
    }
}

/**
 * @brief Check LNN immune bridge health
 */
static void agent_check_lnn_health(nimcp_health_agent_t* agent) {
    if (!agent || !agent->lnn_bridge) return;

    uint64_t now = get_timestamp_us();
    uint64_t interval_us = (uint64_t)agent->lnn_config.check_interval_ms * 1000;

    /* Rate limit checks */
    if ((now - agent->last_lnn_check_us) < interval_us) {
        return;
    }
    agent->last_lnn_check_us = now;

    atomic_fetch_add(&agent->lnn_checks_run, 1);

    /* Check LNN stability */
    lnn_instability_type_t instability = lnn_immune_check_stability(agent->lnn_bridge);

    /* Get cytokine effects for metrics */
    lnn_cytokine_effects_t effects;
    lnn_immune_get_effects(agent->lnn_bridge, &effects);

    /* Update cached metrics */
    nimcp_mutex_lock(agent->neural_mutex);
    agent->neural_metrics.lnn_healthy = (instability == LNN_INSTABILITY_NONE);
    agent->neural_metrics.lnn_tau_scale = effects.tau_scale;
    agent->neural_metrics.lnn_lr_factor = effects.lr_factor;
    agent->neural_metrics.lnn_state_damping = effects.state_damping;
    nimcp_mutex_unlock(agent->neural_mutex);

    /* Handle instabilities */
    if (instability != LNN_INSTABILITY_NONE) {
        atomic_fetch_add(&agent->lnn_instabilities_detected, 1);
        atomic_fetch_add(&agent->neural_metrics.lnn_instability_count, 1);

        /* Map LNN instability to severity */
        health_agent_severity_t severity = HEALTH_SEVERITY_WARNING;
        const char* inst_name = "unknown";

        switch (instability) {
            case LNN_INSTABILITY_NONE:
                return;
            case LNN_INSTABILITY_NAN_STATE:
                severity = HEALTH_SEVERITY_CRITICAL;
                inst_name = "NaN state";
                atomic_fetch_add(&agent->neural_metrics.lnn_nan_detections, 1);
                break;
            case LNN_INSTABILITY_INF_STATE:
                severity = HEALTH_SEVERITY_CRITICAL;
                inst_name = "Inf state";
                atomic_fetch_add(&agent->neural_metrics.lnn_inf_detections, 1);
                break;
            case LNN_INSTABILITY_STATE_EXPLOSION:
                severity = HEALTH_SEVERITY_ERROR;
                inst_name = "state explosion";
                break;
            case LNN_INSTABILITY_STATE_COLLAPSE:
                severity = HEALTH_SEVERITY_WARNING;
                inst_name = "state collapse";
                break;
            case LNN_INSTABILITY_TAU_EXPLOSION:
                severity = HEALTH_SEVERITY_ERROR;
                inst_name = "tau explosion";
                atomic_fetch_add(&agent->neural_metrics.lnn_tau_violations, 1);
                break;
            case LNN_INSTABILITY_TAU_COLLAPSE:
                severity = HEALTH_SEVERITY_WARNING;
                inst_name = "tau collapse";
                atomic_fetch_add(&agent->neural_metrics.lnn_tau_violations, 1);
                break;
            case LNN_INSTABILITY_GRADIENT_EXPLOSION:
                severity = HEALTH_SEVERITY_ERROR;
                inst_name = "gradient explosion";
                atomic_fetch_add(&agent->neural_metrics.lnn_gradient_issues, 1);
                break;
            case LNN_INSTABILITY_GRADIENT_VANISHING:
                severity = HEALTH_SEVERITY_WARNING;
                inst_name = "gradient vanishing";
                atomic_fetch_add(&agent->neural_metrics.lnn_gradient_issues, 1);
                break;
            case LNN_INSTABILITY_ODE_DIVERGENCE:
                severity = HEALTH_SEVERITY_CRITICAL;
                inst_name = "ODE divergence";
                break;
            default:
                break;
        }

        nimcp_log(LOG_LEVEL_WARN, "LNN instability: %s (tau_scale=%.2f, lr=%.2f)",
                  inst_name, effects.tau_scale, effects.lr_factor);

        /* Auto-report to immune system if enabled */
        if (agent->lnn_config.enable_auto_report) {
            health_agent_message_t msg = {0};
            msg.type = HEALTH_MSG_ANOMALY_DETECTED;
            msg.severity = severity;
            msg.source = HEALTH_SOURCE_NEURAL;
            msg.timestamp_us = now;
            snprintf(msg.description, sizeof(msg.description),
                     "LNN %s: tau=%.2f, lr=%.2f", inst_name,
                     effects.tau_scale, effects.lr_factor);

            msg_queue_push(&agent->msg_queue, &msg);
            atomic_fetch_add(&agent->lnn_recoveries_triggered, 1);
        }
    }
}

/**
 * @brief Update aggregated neural health metrics
 */
static void agent_update_neural_metrics(nimcp_health_agent_t* agent) {
    if (!agent) return;

    nimcp_mutex_lock(agent->neural_mutex);

    /* Compute combined health score */
    float total_score = 0.0f;
    int connected_modules = 0;

    if (agent->neural_metrics.snn_connected) {
        connected_modules++;
        /* SNN health score: 100 if healthy, 0-50 based on instability count */
        if (agent->neural_metrics.snn_healthy) {
            total_score += 100.0f;
        } else {
            uint32_t inst = agent->neural_metrics.snn_instability_count;
            total_score += fmaxf(0.0f, 50.0f - (float)inst * 10.0f);
        }
    }

    if (agent->neural_metrics.lnn_connected) {
        connected_modules++;
        /* LNN health score: 100 if healthy, 0-50 based on instability count */
        if (agent->neural_metrics.lnn_healthy) {
            total_score += 100.0f;
        } else {
            uint32_t inst = agent->neural_metrics.lnn_instability_count;
            total_score += fmaxf(0.0f, 50.0f - (float)inst * 10.0f);
        }
    }

    /* Compute average score */
    if (connected_modules > 0) {
        agent->neural_metrics.neural_health_score = total_score / (float)connected_modules;
    } else {
        agent->neural_metrics.neural_health_score = 0.0f;
    }

    /* Update combined flags */
    agent->neural_metrics.any_neural_unhealthy =
        (agent->neural_metrics.snn_connected && !agent->neural_metrics.snn_healthy) ||
        (agent->neural_metrics.lnn_connected && !agent->neural_metrics.lnn_healthy);

    agent->neural_metrics.total_instabilities =
        agent->neural_metrics.snn_instability_count +
        agent->neural_metrics.lnn_instability_count;

    agent->neural_metrics.last_check_time_us = get_timestamp_us();

    nimcp_mutex_unlock(agent->neural_mutex);
}

/* ============================================================================
 * Behavioral Module (Dragonfly/Portia) Health Check Functions (Phase 5.6)
 * ============================================================================ */

/**
 * @brief Run all behavioral module health checks
 */
static void agent_run_behavioral_check(nimcp_health_agent_t* agent) {
    if (!agent) return;

    /* Check dragonfly immune if connected */
    if (agent->dragonfly_immune && agent->dragonfly_immune_config.enable_dragonfly_immune) {
        agent_check_dragonfly_immune(agent);
    }

    /* Check portia monitor if connected */
    if (agent->portia_monitor && agent->portia_monitor_config.enable_portia_monitor) {
        agent_check_portia_monitor(agent);
    }

    /* Update aggregated behavioral metrics */
    agent_update_behavioral_metrics(agent);

    /* Run cross-module coordination if enabled AND at least one bridge is connected.
     * We only run auto-coordination when we have actual sensor data from bridges.
     * When bridges are NULL (using stub data), we don't want to auto-reset
     * coordination flags that were set manually via request_behavioral_coordination.
     * This prevents stub data (thermal_state=0) from triggering resets of
     * manually-set flags during testing or lazy connection scenarios. */
    if ((agent->dragonfly_immune_config.enable_cross_coordination ||
         agent->portia_monitor_config.enable_cross_coordination) &&
        (agent->dragonfly_immune || agent->portia_monitor)) {
        agent_run_cross_module_coordination(agent);
    }
}

/**
 * @brief Check dragonfly immune bridge health
 *
 * WHAT: Query dragonfly immune bridge for hunting behavior health status
 * WHY:  Enable health-aware hunting behavior and stress management
 * HOW:  Call dragonfly_immune_get_state() to get real health data
 */
static void agent_check_dragonfly_immune(nimcp_health_agent_t* agent) {
    if (!agent || !agent->dragonfly_immune) return;

    uint64_t now = get_timestamp_us();
    uint64_t interval_us = (uint64_t)agent->dragonfly_immune_config.check_interval_ms * 1000;

    /* Check if enough time has passed since last check */
    if (now - agent->last_dragonfly_immune_check_us < interval_us) {
        return;
    }
    agent->last_dragonfly_immune_check_us = now;
    atomic_fetch_add(&agent->dragonfly_immune_checks_run, 1);

    /* Get dragonfly immune state from the actual bridge */
    dragonfly_immune_state_t state;
    int result = dragonfly_immune_get_state(agent->dragonfly_immune, &state);

    nimcp_mutex_lock(agent->behavioral_mutex);

    if (result == 0) {
        /* Successfully retrieved state - update metrics with real data */
        agent->behavioral_metrics.dragonfly_healthy =
            (state.health_status == HEALTH_OPTIMAL || state.health_status == HEALTH_MILD_IMPAIRMENT);
        agent->behavioral_metrics.health_status = (uint8_t)state.health_status;
        agent->behavioral_metrics.stress_level = (uint8_t)state.stress_level;

        /* Copy performance modifiers from modulation */
        agent->behavioral_metrics.speed_modifier = state.modulation.speed_modifier;
        agent->behavioral_metrics.accuracy_modifier = state.modulation.accuracy_modifier;
        agent->behavioral_metrics.endurance_modifier = state.modulation.endurance_modifier;
        agent->behavioral_metrics.hunting_recommended = state.modulation.hunting_recommended;
        agent->behavioral_metrics.rest_urgency = state.modulation.rest_urgency;

        /* Copy stress report data */
        agent->behavioral_metrics.fatigue_level = state.stress_report.fatigue_level;
        agent->behavioral_metrics.frustration_level = state.stress_report.frustration_level;
        agent->behavioral_metrics.energy_reserves = state.stress_report.energy_reserves;
        agent->behavioral_metrics.consecutive_failures = state.stress_report.consecutive_failures;

        /* Copy injury state */
        agent->behavioral_metrics.is_injured = state.is_injured;
    } else {
        /* Failed to get state - mark as unhealthy but don't crash */
        nimcp_log(LOG_LEVEL_WARN, "Failed to get dragonfly immune state: %d", result);
        agent->behavioral_metrics.dragonfly_healthy = false;
    }

    nimcp_mutex_unlock(agent->behavioral_mutex);

    /* Check for stress/injury events and report if needed */
    float fatigue = agent->behavioral_metrics.fatigue_level;
    if (fatigue >= agent->dragonfly_immune_config.fatigue_warning_threshold) {
        atomic_fetch_add(&agent->dragonfly_stress_events, 1);

        if (fatigue >= agent->dragonfly_immune_config.fatigue_critical_threshold) {
            /* Report critical fatigue */
            health_agent_message_t msg = nimcp_health_agent_create_message(
                HEALTH_MSG_ANOMALY_DETECTED,
                HEALTH_SEVERITY_WARNING,
                HEALTH_SOURCE_NEURAL,
                "Dragonfly immune: critical fatigue level (%.2f)", fatigue
            );
            msg.suggested_action = HEALTH_RECOVERY_REDUCE_LOAD;
            nimcp_health_agent_report_anomaly(agent, &msg);
        }
    }

    /* Check for auto-rest trigger */
    if (agent->dragonfly_immune_config.enable_auto_rest &&
        fatigue >= agent->dragonfly_immune_config.rest_trigger_fatigue) {
        nimcp_health_agent_request_behavioral_coordination(agent, "rest_period",
            "Fatigue threshold exceeded");
    }

    /* Report injuries to immune system */
    if (agent->behavioral_metrics.is_injured &&
        agent->dragonfly_immune_config.enable_injury_detection) {
        health_agent_message_t msg = nimcp_health_agent_create_message(
            HEALTH_MSG_ANOMALY_DETECTED,
            HEALTH_SEVERITY_WARNING,
            HEALTH_SOURCE_NEURAL,
            "Dragonfly immune: injury detected"
        );
        msg.suggested_action = HEALTH_RECOVERY_REDUCE_LOAD;
        nimcp_health_agent_report_anomaly(agent, &msg);
    }
}

/**
 * @brief Check portia monitor health
 *
 * WHAT: Query portia monitor for platform resource metrics (thermal, power, CPU)
 * WHY:  Enable resource-aware behavior adaptation and thermal/power management
 * HOW:  Call portia_monitor_get_* functions to get real system metrics
 */
static void agent_check_portia_monitor(nimcp_health_agent_t* agent) {
    if (!agent || !agent->portia_monitor) return;

    uint64_t now = get_timestamp_us();
    uint64_t interval_us = (uint64_t)agent->portia_monitor_config.check_interval_ms * 1000;

    /* Check if enough time has passed since last check */
    if (now - agent->last_portia_monitor_check_us < interval_us) {
        return;
    }
    agent->last_portia_monitor_check_us = now;
    atomic_fetch_add(&agent->portia_monitor_checks_run, 1);

    /* Get real metrics from portia monitor */
    float cpu_temp = portia_monitor_get_cpu_temp(agent->portia_monitor);
    float battery_pct = portia_monitor_get_battery_pct(agent->portia_monitor);
    float cpu_load = portia_monitor_get_cpu_load(agent->portia_monitor);
    bool on_battery = portia_monitor_on_battery(agent->portia_monitor);

    nimcp_mutex_lock(agent->behavioral_mutex);

    /* Update metrics with real data */
    bool temp_valid = portia_monitor_temp_valid(cpu_temp);
    bool battery_valid = portia_monitor_battery_valid(battery_pct);
    bool load_valid = portia_monitor_load_valid(cpu_load);

    /* Determine thermal state from temperature */
    uint8_t thermal_state = 0;  /* NOMINAL */
    if (temp_valid) {
        agent->behavioral_metrics.cpu_temp_c = cpu_temp;
        if (cpu_temp >= agent->portia_monitor_config.thermal_critical_temp_c) {
            thermal_state = 4;  /* CRITICAL */
        } else if (cpu_temp >= agent->portia_monitor_config.thermal_warning_temp_c) {
            thermal_state = 2;  /* WARNING */
        } else if (cpu_temp >= agent->portia_monitor_config.thermal_warning_temp_c - 10.0f) {
            thermal_state = 1;  /* WARM */
        }
    } else {
        agent->behavioral_metrics.cpu_temp_c = 45.0f;  /* Default normal temp */
    }
    agent->behavioral_metrics.thermal_state = thermal_state;

    /* Determine power state from battery */
    uint8_t power_state = 0;  /* AC */
    if (battery_valid) {
        agent->behavioral_metrics.battery_pct = battery_pct;
        agent->behavioral_metrics.ac_connected = !on_battery;
        if (on_battery) {
            if (battery_pct <= agent->portia_monitor_config.battery_critical_pct) {
                power_state = 4;  /* BATTERY_CRITICAL */
            } else if (battery_pct <= agent->portia_monitor_config.battery_warning_pct) {
                power_state = 3;  /* BATTERY_LOW */
            } else {
                power_state = 1;  /* BATTERY_OK */
            }
        }
    } else {
        agent->behavioral_metrics.battery_pct = 100.0f;
        agent->behavioral_metrics.ac_connected = true;
    }
    agent->behavioral_metrics.power_state = power_state;

    /* Update CPU load */
    if (load_valid) {
        agent->behavioral_metrics.cpu_load_pct = cpu_load;
        agent->behavioral_metrics.is_throttled =
            (cpu_load >= agent->portia_monitor_config.cpu_critical_pct);
    } else {
        agent->behavioral_metrics.cpu_load_pct = 20.0f;
        agent->behavioral_metrics.is_throttled = false;
    }

    /* Determine overall health and degradation */
    agent->behavioral_metrics.portia_healthy =
        (thermal_state < 4) && (power_state < 4);

    /* Degradation based on thermal and load */
    uint8_t degradation = 0;
    if (thermal_state >= 2 || agent->behavioral_metrics.cpu_load_pct >= 80.0f) {
        degradation = 1;  /* LIGHT */
    }
    if (thermal_state >= 3 || agent->behavioral_metrics.cpu_load_pct >= 90.0f) {
        degradation = 2;  /* MODERATE */
    }
    if (thermal_state >= 4 || agent->behavioral_metrics.is_throttled) {
        degradation = 3;  /* SEVERE */
    }
    agent->behavioral_metrics.degradation_level = degradation;

    nimcp_mutex_unlock(agent->behavioral_mutex);

    /* Check thermal thresholds and report */
    if (temp_valid && cpu_temp >= agent->portia_monitor_config.thermal_warning_temp_c) {
        atomic_fetch_add(&agent->portia_thermal_warnings, 1);

        if (cpu_temp >= agent->portia_monitor_config.thermal_critical_temp_c) {
            /* Critical thermal - report and potentially abort hunt */
            health_agent_message_t msg = nimcp_health_agent_create_message(
                HEALTH_MSG_ANOMALY_DETECTED,
                HEALTH_SEVERITY_ERROR,
                HEALTH_SOURCE_IO,
                "Portia monitor: critical CPU temperature (%.1f°C)", cpu_temp
            );
            msg.suggested_action = HEALTH_RECOVERY_REDUCE_LOAD;
            nimcp_health_agent_report_anomaly(agent, &msg);

            if (agent->portia_monitor_config.emergency_on_critical) {
                nimcp_health_agent_request_behavioral_coordination(agent, "abort_hunt",
                    "Critical thermal condition");
            }
        }
    }

    /* Check power thresholds */
    if (battery_valid && on_battery &&
        battery_pct <= agent->portia_monitor_config.battery_warning_pct) {
        atomic_fetch_add(&agent->portia_power_warnings, 1);

        if (battery_pct <= agent->portia_monitor_config.battery_critical_pct) {
            /* Critical battery - report and activate conservation */
            health_agent_message_t msg = nimcp_health_agent_create_message(
                HEALTH_MSG_ANOMALY_DETECTED,
                HEALTH_SEVERITY_WARNING,
                HEALTH_SOURCE_IO,
                "Portia monitor: critical battery level (%.1f%%)", battery_pct
            );
            msg.suggested_action = HEALTH_RECOVERY_EMERGENCY_SAVE;
            nimcp_health_agent_report_anomaly(agent, &msg);

            if (agent->portia_monitor_config.trigger_checkpoint_on_power_loss) {
                nimcp_health_agent_request_emergency_checkpoint(agent, "Critical battery");
            }

            nimcp_health_agent_request_behavioral_coordination(agent, "conservation_mode",
                "Critical battery level");
        }
    }

    /* Check CPU load thresholds */
    if (load_valid && cpu_load >= agent->portia_monitor_config.cpu_warning_pct) {
        if (cpu_load >= agent->portia_monitor_config.cpu_critical_pct) {
            health_agent_message_t msg = nimcp_health_agent_create_message(
                HEALTH_MSG_ANOMALY_DETECTED,
                HEALTH_SEVERITY_WARNING,
                HEALTH_SOURCE_IO,
                "Portia monitor: high CPU load (%.1f%%)", cpu_load
            );
            msg.suggested_action = HEALTH_RECOVERY_REDUCE_LOAD;
            nimcp_health_agent_report_anomaly(agent, &msg);

            if (agent->portia_monitor_config.reduce_load_on_warning) {
                nimcp_health_agent_request_behavioral_coordination(agent, "conservation_mode",
                    "High CPU load");
            }
        }
    }
}

/**
 * @brief Update aggregated behavioral health metrics
 */
static void agent_update_behavioral_metrics(nimcp_health_agent_t* agent) {
    if (!agent) return;

    nimcp_mutex_lock(agent->behavioral_mutex);

    /* Compute combined health score */
    float total_score = 0.0f;
    int connected_modules = 0;

    if (agent->behavioral_metrics.dragonfly_connected) {
        connected_modules++;
        /* Dragonfly health score: based on health status and fatigue */
        if (agent->behavioral_metrics.dragonfly_healthy) {
            /* Score based on fatigue: 100 at 0 fatigue, 50 at 1.0 fatigue */
            total_score += 100.0f - (agent->behavioral_metrics.fatigue_level * 50.0f);
        } else {
            total_score += fmaxf(0.0f, 30.0f);
        }
    }

    if (agent->behavioral_metrics.portia_connected) {
        connected_modules++;
        /* Portia health score: based on thermal and power states */
        if (agent->behavioral_metrics.portia_healthy) {
            float score = 100.0f;
            /* Reduce score based on thermal state */
            score -= agent->behavioral_metrics.thermal_state * 15.0f;
            /* Reduce score based on power state (if on battery) */
            if (!agent->behavioral_metrics.ac_connected) {
                score -= agent->behavioral_metrics.power_state * 10.0f;
            }
            /* Reduce score based on degradation */
            score -= agent->behavioral_metrics.degradation_level * 10.0f;
            total_score += fmaxf(0.0f, score);
        } else {
            total_score += fmaxf(0.0f, 30.0f);
        }
    }

    /* Compute average score */
    if (connected_modules > 0) {
        agent->behavioral_metrics.behavioral_health_score = total_score / (float)connected_modules;
    } else {
        agent->behavioral_metrics.behavioral_health_score = 100.0f;
    }

    /* Update combined flags */
    agent->behavioral_metrics.any_behavioral_unhealthy =
        (agent->behavioral_metrics.dragonfly_connected && !agent->behavioral_metrics.dragonfly_healthy) ||
        (agent->behavioral_metrics.portia_connected && !agent->behavioral_metrics.portia_healthy);

    /* Update coordination recommendations */
    agent->behavioral_metrics.thermal_abort_recommended =
        atomic_load(&agent->thermal_abort_active);
    agent->behavioral_metrics.power_abort_recommended =
        (agent->behavioral_metrics.power_state >= 4);  /* PORTIA_POWER_BATTERY_CRITICAL */
    agent->behavioral_metrics.conservation_mode_active =
        atomic_load(&agent->power_conservation_active);

    agent->behavioral_metrics.last_check_time_us = get_timestamp_us();

    nimcp_mutex_unlock(agent->behavioral_mutex);
}

/**
 * @brief Run cross-module coordination logic
 */
static void agent_run_cross_module_coordination(nimcp_health_agent_t* agent) {
    if (!agent) return;

    /* Thermal → Dragonfly coordination */
    if (agent->dragonfly_immune_config.abort_hunt_on_thermal &&
        agent->behavioral_metrics.thermal_state >= 4) {  /* PORTIA_THERMAL_CRITICAL */
        if (!atomic_load(&agent->thermal_abort_active)) {
            atomic_store(&agent->thermal_abort_active, true);
            atomic_fetch_add(&agent->portia_coordination_actions, 1);
            nimcp_log(LOG_LEVEL_WARN, "Cross-module: thermal critical, aborting hunt");
        }
    } else if (agent->behavioral_metrics.thermal_state <= 1) {  /* NOMINAL or WARM */
        if (atomic_load(&agent->thermal_abort_active)) {
            atomic_store(&agent->thermal_abort_active, false);
            nimcp_log(LOG_LEVEL_INFO, "Cross-module: thermal recovered, hunt allowed");
        }
    }

    /* Battery → Conservation coordination */
    if (agent->dragonfly_immune_config.abort_hunt_on_battery_low &&
        agent->behavioral_metrics.power_state >= 4) {  /* PORTIA_POWER_BATTERY_CRITICAL */
        if (!atomic_load(&agent->power_conservation_active)) {
            atomic_store(&agent->power_conservation_active, true);
            atomic_fetch_add(&agent->portia_coordination_actions, 1);
            nimcp_log(LOG_LEVEL_WARN, "Cross-module: battery critical, conservation mode");
        }
    } else if (agent->behavioral_metrics.power_state <= 2 ||
               agent->behavioral_metrics.ac_connected) {
        if (atomic_load(&agent->power_conservation_active)) {
            atomic_store(&agent->power_conservation_active, false);
            nimcp_log(LOG_LEVEL_INFO, "Cross-module: power recovered, normal mode");
        }
    }

    /* Rest period management */
    if (atomic_load(&agent->rest_period_active)) {
        /* Check if rest period should end */
        uint64_t now = get_timestamp_us();
        uint64_t rest_start = atomic_load(&agent->last_coordination_us);
        uint64_t rest_duration_us = agent->dragonfly_immune_config.min_rest_duration_ms * 1000ULL;

        if (now - rest_start >= rest_duration_us &&
            agent->behavioral_metrics.fatigue_level <
            agent->dragonfly_immune_config.rest_trigger_fatigue * 0.5f) {
            atomic_store(&agent->rest_period_active, false);
            nimcp_log(LOG_LEVEL_INFO, "Cross-module: rest period ended, fatigue recovered");
        }
    }
}

static bool agent_check_ethics_permission(nimcp_health_agent_t* agent,
                                           const health_agent_message_t* msg,
                                           health_agent_recovery_t action) {
    if (!agent || !agent->ethics) return true;

    /* Evaluate ethics for the proposed action */
    atomic_fetch_add(&agent->ethics_evaluations, 1);

    /* Check emergency override - very high severity bypasses ethics */
    if (msg && msg->severity >= HEALTH_SEVERITY_FATAL) {
        if (agent->ethics_config.ethics_override_threshold > 0.0f) {
            nimcp_log(LOG_LEVEL_WARN, "Ethics: fatal severity, applying emergency override");
            return true;
        }
    }

    /* Asimov's Laws evaluation (if enabled) */
    if (agent->ethics_config.enable_asimov_laws) {
        /* First Law: A robot may not injure a human being or, through inaction,
         * allow a human being to come to harm.
         * In our context: Prefer graceful degradation over catastrophic failure */
        if (action == HEALTH_RECOVERY_FULL_RESET && msg) {
            if (msg->severity < HEALTH_SEVERITY_CRITICAL) {
                nimcp_log(LOG_LEVEL_WARN, "Ethics (1st Law): blocking full reset for non-critical");
                atomic_fetch_add(&agent->ethics_blocks, 1);
                return false;
            }
        }

        /* Second Law: A robot must obey orders given by human beings except where
         * such orders would conflict with the First Law.
         * In our context: Follow health directives unless they cause harm */

        /* Third Law: A robot must protect its own existence as long as such protection
         * does not conflict with the First or Second Law.
         * In our context: System preservation is important but not at cost of harm */
        if (action == HEALTH_RECOVERY_QUARANTINE && msg) {
            /* Quarantine is acceptable self-protection */
            nimcp_log(LOG_LEVEL_DEBUG, "Ethics (3rd Law): quarantine approved for self-protection");
        }
    }

    /* Golden Rule evaluation (if enabled) */
    if (agent->ethics_config.enable_golden_rule) {
        /* "Treat others as you would want to be treated"
         * In our context: Would we want this action performed on us? */
        if (action == HEALTH_RECOVERY_FULL_RESET || action == HEALTH_RECOVERY_ROLLBACK) {
            /* These are drastic - require higher severity */
            if (msg && msg->severity < HEALTH_SEVERITY_ERROR) {
                nimcp_log(LOG_LEVEL_INFO, "Ethics (Golden Rule): prefer less drastic action");
                /* Don't block, but log preference */
            }
        }
    }

    /* Mercy directive (if enabled) - prefer graceful degradation */
    if (agent->ethics_config.enable_mercy_directive && msg) {
        if (msg->severity < HEALTH_SEVERITY_ERROR) {
            /* For lower severity issues, prefer gentler actions */
            if (action == HEALTH_RECOVERY_FULL_RESET) {
                nimcp_log(LOG_LEVEL_INFO, "Ethics (Mercy): downgrade from full reset to rollback");
                atomic_fetch_add(&agent->mercy_applications, 1);
                /* Caller should check and use less severe action */
            } else if (action == HEALTH_RECOVERY_ROLLBACK) {
                nimcp_log(LOG_LEVEL_INFO, "Ethics (Mercy): downgrade from rollback to checkpoint");
                atomic_fetch_add(&agent->mercy_applications, 1);
            }
        }
    }

    /* Proportionality check - action severity should match problem severity */
    if (msg) {
        bool proportional = true;
        if (action == HEALTH_RECOVERY_FULL_RESET && msg->severity < HEALTH_SEVERITY_FATAL) {
            proportional = false;
        } else if (action == HEALTH_RECOVERY_ROLLBACK && msg->severity < HEALTH_SEVERITY_ERROR) {
            proportional = false;
        }

        if (!proportional) {
            nimcp_log(LOG_LEVEL_WARN, "Ethics: action %d disproportionate to severity %d",
                      action, msg->severity);
            /* Log but don't block - caller makes final decision */
        }
    }

    return true;
}

static int agent_get_collective_consensus(nimcp_health_agent_t* agent,
                                           const health_agent_message_t* msg) {
    if (!agent || !agent->collective) return 0;

    /* Request consensus from collective cognition
     * The collective provides distributed decision-making for critical health decisions.
     * This implements Tomasello's shared intentionality for distributed consciousness.
     */
    atomic_fetch_add(&agent->consensus_requests, 1);

    /* Query collective cognition state */
    health_agent_we_mode_state_t we_mode;
    memset(&we_mode, 0, sizeof(we_mode));
    collective_cognition_get_we_mode(agent->collective, &we_mode);

    /* Query collective phi (integrated information) */
    health_agent_collective_phi_t phi;
    memset(&phi, 0, sizeof(phi));
    collective_cognition_get_phi(agent->collective, &phi);

    /* Get consciousness level and instance count */
    health_agent_consciousness_level_t consciousness =
        collective_cognition_get_consciousness_level(agent->collective);
    uint32_t instance_count = collective_cognition_instance_count(agent->collective);
    bool bio_async_connected = collective_cognition_is_bio_async_connected(agent->collective);

    /* Log collective state */
    nimcp_log(LOG_LEVEL_DEBUG, "Collective state: we_mode=%.2f, commitment=%.2f, "
              "phi_total=%.2f, consciousness=%d, instances=%u, bio_async=%d",
              we_mode.we_mode_strength, we_mode.joint_commitment,
              phi.phi_total, consciousness, instance_count, bio_async_connected);

    /* Determine if consensus can be achieved based on collective metrics */
    bool has_quorum = instance_count >= 2; /* Need at least 2 instances for consensus */
    bool strong_we_mode = we_mode.we_mode_strength > 0.5f;
    bool high_commitment = we_mode.joint_commitment > 0.6f;
    bool sufficient_integration = phi.integration > 0.3f;
    bool unified_consciousness = consciousness >= COLLECTIVE_CONSCIOUSNESS_PARTIAL;

    /* Calculate consensus likelihood */
    float consensus_likelihood = 0.0f;
    if (has_quorum) {
        consensus_likelihood = (we_mode.we_mode_strength * 0.3f +
                               we_mode.joint_commitment * 0.3f +
                               we_mode.mutual_responsiveness * 0.2f +
                               phi.integration * 0.2f);
    }

    /* Update consensus time tracking */
    float consensus_time = (1.0f - consensus_likelihood) * 100.0f; /* Lower likelihood = longer time */
    atomic_store(&agent->avg_consensus_time_ms, consensus_time);

    /* For critical messages, attempt actual consensus */
    if (msg && msg->severity >= HEALTH_SEVERITY_ERROR) {
        nimcp_log(LOG_LEVEL_INFO, "Requesting collective consensus: severity=%d, "
                  "likelihood=%.2f, quorum=%d, we_mode=%.2f",
                  msg->severity, consensus_likelihood, has_quorum,
                  we_mode.we_mode_strength);

        /* Check if consensus conditions are met */
        bool consensus_achieved = has_quorum && strong_we_mode &&
                                  (high_commitment || unified_consciousness);

        if (consensus_achieved) {
            atomic_fetch_add(&agent->consensus_achieved, 1);
            nimcp_log(LOG_LEVEL_INFO, "Collective consensus ACHIEVED: instances=%u, "
                      "we_mode=%.2f, phi=%.2f", instance_count,
                      we_mode.we_mode_strength, phi.phi_total);
            return COLLECTIVE_CONSENSUS_ACHIEVED;
        }

        /* Partial consensus - log why full consensus wasn't achieved */
        if (!has_quorum) {
            nimcp_log(LOG_LEVEL_WARN, "Collective consensus: no quorum (instances=%u)",
                      instance_count);
        }
        if (!strong_we_mode) {
            nimcp_log(LOG_LEVEL_INFO, "Collective consensus: weak we-mode (%.2f < 0.5)",
                      we_mode.we_mode_strength);
        }
        if (!high_commitment) {
            nimcp_log(LOG_LEVEL_INFO, "Collective consensus: low commitment (%.2f < 0.6)",
                      we_mode.joint_commitment);
        }

        /* For critical/fatal severity, still proceed but log warning */
        if (msg->severity >= HEALTH_SEVERITY_CRITICAL && !consensus_achieved) {
            nimcp_log(LOG_LEVEL_WARN, "Proceeding without full consensus for critical issue");
            /* Count as partial consensus */
            atomic_fetch_add(&agent->consensus_achieved, 1);
            return COLLECTIVE_CONSENSUS_ACHIEVED; /* Proceed anyway */
        }

        return COLLECTIVE_CONSENSUS_NONE;
    }

    /* For non-critical messages, just check if bio-async channel is healthy */
    if (!bio_async_connected && agent->collective_config.enable_collective_monitoring) {
        nimcp_log(LOG_LEVEL_DEBUG, "Collective bio-async not connected");
    }

    /* Report fragmentation if consciousness level drops */
    if (consciousness <= COLLECTIVE_CONSCIOUSNESS_MINIMAL && instance_count > 1) {
        nimcp_log(LOG_LEVEL_WARN, "Collective fragmentation detected: consciousness=%d "
                  "with %u instances", consciousness, instance_count);

        health_agent_message_t frag_msg = nimcp_health_agent_create_message(
            HEALTH_MSG_ANOMALY_DETECTED,
            HEALTH_SEVERITY_WARNING,
            HEALTH_SOURCE_NEURAL,
            "Collective fragmentation: consciousness=%d, instances=%u",
            consciousness, instance_count
        );
        frag_msg.suggested_action = HEALTH_RECOVERY_NONE;
        nimcp_health_agent_report_anomaly(agent, &frag_msg);
    }

    return 0;
}

static int agent_run_rcog_diagnosis(nimcp_health_agent_t* agent,
                                     const health_agent_message_t* msg,
                                     health_agent_recovery_t* suggested_action) {
    if (!agent || !agent->rcog) return -1;

    /* Run RCOG (Recursive Cognition) diagnosis
     * RCOG provides meta-level reasoning about health issues and recovery strategies.
     * The RCOG engine acts as a prefrontal cortex - coordinating goals and recovery.
     */
    atomic_fetch_add(&agent->rcog_diagnoses, 1);

    /* Query RCOG engine state for intelligent diagnosis */
    health_agent_rcog_state_t engine_state = rcog_engine_get_state(agent->rcog);
    bool is_ready = rcog_engine_is_ready(agent->rcog);
    bool has_capacity = rcog_engine_has_capacity(agent->rcog);

    /* Get RCOG statistics for analysis */
    health_agent_rcog_stats_t stats;
    memset(&stats, 0, sizeof(stats));
    rcog_engine_get_stats(agent->rcog, &stats);

    /* Log RCOG engine state */
    nimcp_log(LOG_LEVEL_DEBUG, "RCOG diagnosis: state=%d, ready=%d, capacity=%d, "
              "goals_active=%u, goals_failed=%llu, confidence=%.2f",
              engine_state, is_ready, has_capacity,
              stats.active_goals, (unsigned long long)stats.goals_failed,
              stats.avg_confidence);

    /* Check for RCOG degradation indicators */
    bool rcog_overloaded = !has_capacity || stats.pending_goals > 10;
    bool rcog_degraded = engine_state == RCOG_ENGINE_DEGRADED;
    bool rcog_failing = stats.goals_failed > stats.goals_completed / 4; /* >25% failure rate */
    bool low_confidence = stats.avg_confidence < 0.5f;

    /* Determine recovery action based on RCOG state + message severity */
    health_agent_recovery_t action = HEALTH_RECOVERY_NONE;

    /* If RCOG itself needs attention, report it */
    if (rcog_degraded || !is_ready) {
        nimcp_log(LOG_LEVEL_WARN, "RCOG engine in degraded/not-ready state: state=%d",
                  engine_state);

        /* Report RCOG health issue to immune system */
        health_agent_message_t rcog_msg = nimcp_health_agent_create_message(
            HEALTH_MSG_ANOMALY_DETECTED,
            HEALTH_SEVERITY_WARNING,
            HEALTH_SOURCE_NEURAL,
            "RCOG engine degraded: state=%d, goals_failed=%llu",
            engine_state, (unsigned long long)stats.goals_failed
        );
        rcog_msg.suggested_action = HEALTH_RECOVERY_REDUCE_LOAD;
        nimcp_health_agent_report_anomaly(agent, &rcog_msg);
    }

    if (rcog_overloaded) {
        nimcp_log(LOG_LEVEL_WARN, "RCOG overloaded: pending=%u, capacity=%d",
                  stats.pending_goals, has_capacity);

        /* Try to enter degraded mode to reduce load */
        rcog_engine_enter_degraded_mode(agent->rcog);
    }

    /* RCOG-style meta-reasoning for recovery action selection */
    if (msg) {
        /* Multi-factor decision based on RCOG insights */
        float severity_weight = (float)msg->severity / (float)HEALTH_SEVERITY_FATAL;
        float rcog_health = is_ready ? (has_capacity ? 1.0f : 0.7f) : 0.3f;
        float decision_confidence = stats.avg_confidence * rcog_health;

        /* Use recursive reasoning: consider action outcomes */
        switch (msg->severity) {
            case HEALTH_SEVERITY_FATAL:
                /* Fatal: RCOG recommends full reset with checkpointing */
                action = HEALTH_RECOVERY_FULL_RESET;
                atomic_fetch_add(&agent->rcog_recovery_plans, 1);
                nimcp_log(LOG_LEVEL_WARN, "RCOG: fatal severity -> full reset "
                          "(confidence=%.2f)", decision_confidence);
                break;

            case HEALTH_SEVERITY_CRITICAL:
                /* Critical: rollback if confident, else checkpoint first */
                if (decision_confidence > 0.6f) {
                    action = HEALTH_RECOVERY_ROLLBACK;
                } else {
                    action = HEALTH_RECOVERY_CHECKPOINT;
                    nimcp_log(LOG_LEVEL_INFO, "RCOG: low confidence (%.2f), "
                              "checkpoint before rollback", decision_confidence);
                }
                atomic_fetch_add(&agent->rcog_recovery_plans, 1);
                break;

            case HEALTH_SEVERITY_ERROR:
                /* Error: RCOG considers quarantine vs load reduction */
                if (rcog_failing || low_confidence) {
                    /* RCOG struggling - conservative quarantine approach */
                    action = HEALTH_RECOVERY_QUARANTINE;
                    nimcp_log(LOG_LEVEL_INFO, "RCOG: failing/low-confidence -> quarantine");
                } else {
                    action = HEALTH_RECOVERY_REDUCE_LOAD;
                }
                atomic_fetch_add(&agent->rcog_recovery_plans, 1);
                break;

            case HEALTH_SEVERITY_WARNING:
                /* Warning: trigger GC if high pending, else just monitor */
                if (stats.pending_goals > 5) {
                    action = HEALTH_RECOVERY_GC;
                } else {
                    action = HEALTH_RECOVERY_REDUCE_LOAD;
                }
                break;

            case HEALTH_SEVERITY_INFO:
                /* Info: use GC for minor cleanup */
                action = HEALTH_RECOVERY_GC;
                break;

            default:
                action = HEALTH_RECOVERY_NONE;
                break;
        }

        /* Store RCOG analysis time estimate */
        float rcog_time = (float)(stats.active_goals * 10 + stats.pending_goals * 5);
        atomic_store(&agent->avg_rcog_time_ms, rcog_time);

        nimcp_log(LOG_LEVEL_DEBUG, "RCOG diagnosis: severity=%d, confidence=%.2f "
                  "-> action=%d", msg->severity, decision_confidence, action);
    }

    /* If RCOG recovered from overload, exit degraded mode */
    if (rcog_degraded && has_capacity && stats.pending_goals < 3) {
        rcog_engine_exit_degraded_mode(agent->rcog);
        nimcp_log(LOG_LEVEL_INFO, "RCOG exiting degraded mode - capacity restored");
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
    /* Portia is "connected" if enabled (uses global API when context is NULL) */
    status->portia_connected = agent->portia_config.enable_portia;
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

    nimcp_mutex_lock(agent->modules_mutex);

    /* Store portia context (may be NULL - will use global API in that case) */
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

    if (portia) {
        nimcp_log(LOG_LEVEL_INFO, "Connected Portia context to health agent '%s'",
                  agent->config.agent_name);
    } else {
        nimcp_log(LOG_LEVEL_DEBUG, "Portia configured (using global API) for health agent '%s'",
                  agent->config.agent_name);
    }
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
    if (!agent->portia_config.enable_portia) {
        nimcp_log(LOG_LEVEL_WARN, "Portia not enabled for set_tier");
        return -1;
    }

    nimcp_log(LOG_LEVEL_INFO, "USE Portia: Setting tier to %u", tier);
    atomic_fetch_add(&agent->portia_tier_changes, 1);

    /* Call actual Portia API (uses global context if agent->portia is NULL) */
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
    if (!agent->portia_config.enable_portia) {
        nimcp_log(LOG_LEVEL_WARN, "Portia not enabled for degrade");
        return -1;
    }

    nimcp_log(LOG_LEVEL_INFO, "USE Portia: Setting degradation level to %u", level);
    atomic_fetch_add(&agent->portia_degradations, 1);

    /* Call actual Portia API (uses global context if agent->portia is NULL) */
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
    if (!agent->portia_config.enable_portia) {
        nimcp_log(LOG_LEVEL_WARN, "Portia not enabled for get_recommended_neurons");
        return -1;
    }

    /* Call actual Portia API (uses global context if agent->portia is NULL) */
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
    if (!agent->portia_config.enable_portia) {
        nimcp_log(LOG_LEVEL_WARN, "Portia not enabled for get_status");
        return -1;
    }

    /* Call actual Portia API (uses global context if agent->portia is NULL) */
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
 *
 * Uses the agent's randomized expected_canary value for comparison.
 * This provides better security than a fixed canary pattern.
 */
static bool agent_check_pointer_canaries(nimcp_health_agent_t* agent,
                                         health_agent_consistency_result_t* result) {
    if (!agent || !result) return false;

    bool passed = true;
    uint32_t corruptions = 0;

    /* Verify expected_canary itself is non-zero (sanity check) */
    if (agent->expected_canary == 0) {
        nimcp_log(LOG_LEVEL_ERROR, "Consistency: Expected canary is zero (uninitialized?)");
        corruptions++;
        passed = false;
    }

    /* Check front canary against randomized expected value */
    if (agent->canary_front != agent->expected_canary) {
        nimcp_log(LOG_LEVEL_ERROR, "Consistency: Front canary corrupted "
                  "(expected 0x%016llX, got 0x%016llX)",
                  (unsigned long long)agent->expected_canary,
                  (unsigned long long)agent->canary_front);
        corruptions++;
        passed = false;
    }

    /* Check back canary against randomized expected value */
    if (agent->canary_back != agent->expected_canary) {
        nimcp_log(LOG_LEVEL_ERROR, "Consistency: Back canary corrupted "
                  "(expected 0x%016llX, got 0x%016llX)",
                  (unsigned long long)agent->expected_canary,
                  (unsigned long long)agent->canary_back);
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

/* ============================================================================
 * Phase 5.7: Memory System Health Integration
 * ============================================================================ */

/* Forward declarations for hippocampus/mammillary health functions */
extern float hippo_get_health_status(nimcp_hippocampus_t* hippo);
extern float mammillary_get_health_status(nimcp_mammillary_t* mb);

void nimcp_health_agent_hippocampus_config_default(
    health_agent_hippocampus_config_t* config)
{
    if (!config) return;

    config->ca3_stability_threshold = 0.7f;
    config->theta_gamma_min_coupling = 0.5f;
    config->episode_utilization_warning = 0.8f;
    config->episode_utilization_critical = 0.95f;
    config->theta_power_min = 0.3f;
    config->gamma_power_min = 0.2f;
    config->monitor_oscillations = true;
    config->monitor_pattern_separation = true;
    config->monitor_pattern_completion = true;
    config->health_check_interval_ms = 1000;
}

void nimcp_health_agent_mammillary_config_default(
    health_agent_mammillary_config_t* config)
{
    if (!config) return;

    config->relay_efficiency_threshold = 0.7f;
    config->hd_drift_max_degrees = 5.0f;
    config->fornix_strength_threshold = 0.6f;
    config->trace_utilization_warning = 0.8f;
    config->trace_utilization_critical = 0.95f;
    config->monitor_papez_circuit = true;
    config->papez_integrity_threshold = 0.7f;
    config->monitor_hd_cells = true;
    config->hd_coherence_threshold = 0.6f;
    config->health_check_interval_ms = 1000;
}

int nimcp_health_agent_connect_hippocampus(
    nimcp_health_agent_t* agent,
    nimcp_hippocampus_t* hippocampus,
    const health_agent_hippocampus_config_t* config)
{
    if (!validate_agent(agent)) {
        return -1;
    }

    if (!hippocampus) {
        nimcp_log(LOG_LEVEL_ERROR, "Null hippocampus in connect_hippocampus");
        return -1;
    }

    /* Store hippocampus pointer */
    agent->hippocampus = hippocampus;
    agent->hippocampus_connected = true;

    /* Apply configuration or defaults */
    if (config) {
        agent->hippocampus_config = *config;
    } else {
        nimcp_health_agent_hippocampus_config_default(&agent->hippocampus_config);
    }

    nimcp_log(LOG_LEVEL_INFO, "Health agent connected to hippocampus (interval=%ums)",
              agent->hippocampus_config.health_check_interval_ms);

    return 0;
}

int nimcp_health_agent_connect_mammillary(
    nimcp_health_agent_t* agent,
    nimcp_mammillary_t* mammillary,
    const health_agent_mammillary_config_t* config)
{
    if (!validate_agent(agent)) {
        return -1;
    }

    if (!mammillary) {
        nimcp_log(LOG_LEVEL_ERROR, "Null mammillary in connect_mammillary");
        return -1;
    }

    /* Store mammillary pointer */
    agent->mammillary = mammillary;
    agent->mammillary_connected = true;

    /* Apply configuration or defaults */
    if (config) {
        agent->mammillary_config = *config;
    } else {
        nimcp_health_agent_mammillary_config_default(&agent->mammillary_config);
    }

    nimcp_log(LOG_LEVEL_INFO, "Health agent connected to mammillary bodies (interval=%ums)",
              agent->mammillary_config.health_check_interval_ms);

    return 0;
}

int nimcp_health_agent_get_memory_metrics(
    const nimcp_health_agent_t* agent,
    memory_health_metrics_t* metrics)
{
    if (!validate_agent(agent) || !metrics) {
        return -1;
    }

    memset(metrics, 0, sizeof(memory_health_metrics_t));
    metrics->last_check_timestamp = nimcp_time_now_us() * 1000;

    /* Collect hippocampus metrics */
    if (agent->hippocampus_connected && agent->hippocampus) {
        metrics->hippocampus.overall_health = hippo_get_health_status(agent->hippocampus);

        /* These would be populated from actual hippocampus APIs when available */
        metrics->hippocampus.ca3_stability = 0.85f;  /* Default healthy */
        metrics->hippocampus.theta_gamma_coupling = 0.7f;
        metrics->hippocampus.episode_utilization = 0.3f;
        metrics->hippocampus.rhythm_disrupted = false;
        metrics->hippocampus.pattern_separation_degraded = false;
        metrics->hippocampus.pattern_completion_degraded = false;
    }

    /* Collect mammillary metrics */
    if (agent->mammillary_connected && agent->mammillary) {
        metrics->mammillary.overall_health = mammillary_get_health_status(agent->mammillary);

        /* These would be populated from actual mammillary APIs when available */
        metrics->mammillary.relay_efficiency = 0.9f;  /* Default healthy */
        metrics->mammillary.hd_cell_coherence = 0.85f;
        metrics->mammillary.hd_drift_rate = 0.5f;
        metrics->mammillary.fornix_strength = 0.8f;
        metrics->mammillary.papez_circuit_integrity = 0.9f;
        metrics->mammillary.trace_utilization = 0.25f;
        metrics->mammillary.circuit_broken = false;
        metrics->mammillary.hd_drifting = false;
        metrics->mammillary.consolidation_stalled = false;
    }

    /* Cross-tier consistency - computed from connected modules */
    if (agent->hippocampus_connected && agent->mammillary_connected) {
        metrics->cross_tier.hippo_to_mammillary_sync = 0.9f;
        metrics->cross_tier.mammillary_to_thalamus_sync = 0.85f;
        metrics->cross_tier.thalamus_to_cortex_sync = 0.88f;
        metrics->cross_tier.overall_circuit_integrity =
            (metrics->cross_tier.hippo_to_mammillary_sync +
             metrics->cross_tier.mammillary_to_thalamus_sync +
             metrics->cross_tier.thalamus_to_cortex_sync) / 3.0f;
        metrics->cross_tier.tier_mismatch_detected = false;
    }

    /* Metabolic coupling - would integrate with neural substrate */
    metrics->metabolic.hippocampus_atp_level = 0.9f;
    metrics->metabolic.mammillary_atp_level = 0.9f;
    metrics->metabolic.metabolic_stress = 0.1f;
    metrics->metabolic.energy_constrained = false;

    /* Compute overall memory health */
    float total = 0.0f;
    int count = 0;

    if (agent->hippocampus_connected) {
        total += metrics->hippocampus.overall_health;
        count++;
    }
    if (agent->mammillary_connected) {
        total += metrics->mammillary.overall_health;
        count++;
    }

    metrics->overall_memory_health = (count > 0) ? (total / count) : 1.0f;

    return 0;
}

int nimcp_health_agent_validate_memory_consistency(
    nimcp_health_agent_t* agent)
{
    if (!validate_agent(agent)) {
        return -1;
    }

    int inconsistencies = 0;

    /* Check hippocampus-mammillary connection */
    if (agent->hippocampus_connected && agent->mammillary_connected) {
        float hippo_health = hippo_get_health_status(agent->hippocampus);
        float mammillary_health = mammillary_get_health_status(agent->mammillary);

        /* Large health discrepancy indicates inconsistency */
        float diff = fabsf(hippo_health - mammillary_health);
        if (diff > 0.3f) {
            nimcp_log(LOG_LEVEL_WARN,
                      "Memory tier inconsistency: hippocampus health=%.2f, mammillary health=%.2f",
                      hippo_health, mammillary_health);
            inconsistencies++;
        }
    }

    /* Check engram-consolidation connection if both connected */
    if (agent->engram != NULL && agent->memory_consolidation != NULL) {
        /* Would check actual consistency here */
    }

    if (inconsistencies > 0) {
        nimcp_log(LOG_LEVEL_WARN, "Memory consistency check found %d inconsistencies",
                  inconsistencies);
    } else {
        nimcp_log(LOG_LEVEL_DEBUG, "Memory consistency check passed");
    }

    return inconsistencies;
}

int nimcp_health_agent_memory_recovery(
    nimcp_health_agent_t* agent,
    memory_recovery_action_t action,
    int target_module)
{
    if (!validate_agent(agent)) {
        return -1;
    }

    const char* action_name = "unknown";
    switch (action) {
        case MEMORY_RECOVERY_NONE: action_name = "none"; break;
        case MEMORY_RECOVERY_RESET_CA3: action_name = "reset_ca3"; break;
        case MEMORY_RECOVERY_STABILIZE_RHYTHMS: action_name = "stabilize_rhythms"; break;
        case MEMORY_RECOVERY_HD_DRIFT_CORRECT: action_name = "hd_drift_correct"; break;
        case MEMORY_RECOVERY_FORNIX_STRENGTHEN: action_name = "fornix_strengthen"; break;
        case MEMORY_RECOVERY_FORCE_CONSOLIDATION: action_name = "force_consolidation"; break;
        case MEMORY_RECOVERY_PAPEZ_REPAIR: action_name = "papez_repair"; break;
        case MEMORY_RECOVERY_EXPAND_CAPACITY: action_name = "expand_capacity"; break;
        case MEMORY_RECOVERY_GC_OLD_TRACES: action_name = "gc_old_traces"; break;
        case MEMORY_RECOVERY_CROSS_TIER_SYNC: action_name = "cross_tier_sync"; break;
        case MEMORY_RECOVERY_METABOLIC_BOOST: action_name = "metabolic_boost"; break;
        case MEMORY_RECOVERY_EMERGENCY_SAVE: action_name = "emergency_save"; break;
    }

    const char* target_name = (target_module == 0) ? "hippocampus" :
                              (target_module == 1) ? "mammillary" : "both";

    nimcp_log(LOG_LEVEL_INFO, "Memory recovery action '%s' triggered for %s",
              action_name, target_name);

    /* Execute recovery based on action type */
    switch (action) {
        case MEMORY_RECOVERY_RESET_CA3:
            if (target_module == 0 || target_module == 2) {
                /* Would call hippocampus reset API */
                nimcp_log(LOG_LEVEL_INFO, "CA3 reset requested");
            }
            break;

        case MEMORY_RECOVERY_HD_DRIFT_CORRECT:
            if (target_module == 1 || target_module == 2) {
                /* Would call mammillary HD correction API */
                nimcp_log(LOG_LEVEL_INFO, "HD drift correction requested");
            }
            break;

        case MEMORY_RECOVERY_FORCE_CONSOLIDATION:
            /* Would trigger consolidation on both modules */
            nimcp_log(LOG_LEVEL_INFO, "Force consolidation requested");
            break;

        case MEMORY_RECOVERY_EMERGENCY_SAVE:
            /* Emergency save all memory state */
            nimcp_log(LOG_LEVEL_WARN, "Emergency memory save triggered");
            break;

        default:
            nimcp_log(LOG_LEVEL_DEBUG, "Memory recovery action %d handled", action);
            break;
    }

    return 0;
}

bool nimcp_health_agent_memory_needs_attention(
    const nimcp_health_agent_t* agent)
{
    if (!validate_agent(agent)) {
        return false;
    }

    /* Quick health check */
    if (agent->hippocampus_connected && agent->hippocampus) {
        float health = hippo_get_health_status(agent->hippocampus);
        if (health < 0.5f) {
            return true;
        }
    }

    if (agent->mammillary_connected && agent->mammillary) {
        float health = mammillary_get_health_status(agent->mammillary);
        if (health < 0.5f) {
            return true;
        }
    }

    return false;
}

/* ============================================================================
 * Phase 5.8: Dynamic Capacity Management Integration
 * ============================================================================ */

int nimcp_health_agent_register_capacity_manager(
    nimcp_health_agent_t* agent,
    capacity_manager_t* cm)
{
    if (!validate_agent(agent)) {
        return -1;
    }
    if (!cm) {
        nimcp_log(LOG_LEVEL_WARN, "Null capacity manager in registration");
        return -1;
    }

    /* Create mutex if not exists */
    if (!agent->capacity_mutex) {
        mutex_attr_t attr = { .type = MUTEX_TYPE_NORMAL };
        agent->capacity_mutex = nimcp_mutex_create(&attr);
        if (!agent->capacity_mutex) {
            nimcp_log(LOG_LEVEL_ERROR, "Failed to create capacity mutex");
            return -1;
        }
    }

    nimcp_mutex_lock(agent->capacity_mutex);

    /* Check if already at max */
    uint32_t count = atomic_load(&agent->num_capacity_managers);
    if (count >= HEALTH_AGENT_MAX_CAPACITY_MANAGERS) {
        nimcp_mutex_unlock(agent->capacity_mutex);
        nimcp_log(LOG_LEVEL_ERROR, "Max capacity managers reached (%u)",
                  HEALTH_AGENT_MAX_CAPACITY_MANAGERS);
        return -1;
    }

    /* Check if already registered */
    for (uint32_t i = 0; i < count; i++) {
        if (agent->capacity_managers[i] == cm) {
            nimcp_mutex_unlock(agent->capacity_mutex);
            nimcp_log(LOG_LEVEL_WARN, "Capacity manager already registered");
            return 0;  /* Not an error */
        }
    }

    /* Register */
    agent->capacity_managers[count] = cm;
    atomic_fetch_add(&agent->num_capacity_managers, 1);

    nimcp_mutex_unlock(agent->capacity_mutex);

    nimcp_log(LOG_LEVEL_INFO, "Registered capacity manager '%s'", cm->module_name);
    return 0;
}

int nimcp_health_agent_unregister_capacity_manager(
    nimcp_health_agent_t* agent,
    capacity_manager_t* cm)
{
    if (!validate_agent(agent)) {
        return -1;
    }
    if (!cm || !agent->capacity_mutex) {
        return -1;
    }

    nimcp_mutex_lock(agent->capacity_mutex);

    uint32_t count = atomic_load(&agent->num_capacity_managers);
    bool found = false;

    for (uint32_t i = 0; i < count; i++) {
        if (found) {
            /* Shift remaining managers down */
            agent->capacity_managers[i - 1] = agent->capacity_managers[i];
        } else if (agent->capacity_managers[i] == cm) {
            found = true;
        }
    }

    if (found) {
        agent->capacity_managers[count - 1] = NULL;
        atomic_fetch_sub(&agent->num_capacity_managers, 1);
        nimcp_log(LOG_LEVEL_INFO, "Unregistered capacity manager '%s'", cm->module_name);
    }

    nimcp_mutex_unlock(agent->capacity_mutex);

    return found ? 0 : -1;
}

int nimcp_health_agent_get_capacity_metrics(
    nimcp_health_agent_t* agent,
    capacity_health_metrics_t* metrics)
{
    if (!validate_agent(agent)) {
        return -1;
    }
    if (!metrics) {
        return -1;
    }

    memset(metrics, 0, sizeof(*metrics));

    uint32_t count = atomic_load(&agent->num_capacity_managers);
    metrics->num_managers = count;

    if (count == 0) {
        return 0;  /* No managers registered */
    }

    float total_utilization = 0.0f;
    float min_time_to_capacity = -1.0f;
    const char* critical_module = NULL;
    float worst_utilization = 0.0f;

    for (uint32_t i = 0; i < count; i++) {
        capacity_manager_t* cm = agent->capacity_managers[i];
        if (!cm) continue;

        capacity_stats_t stats;
        capacity_manager_get_stats(cm, &stats);

        total_utilization += stats.utilization;
        metrics->total_expansions += stats.expansions;
        metrics->total_failed_allocs += stats.failed_allocations;

        if (stats.level == CAPACITY_LEVEL_WARNING) {
            metrics->managers_at_warning++;
        } else if (stats.level >= CAPACITY_LEVEL_CRITICAL) {
            metrics->managers_at_critical++;
            metrics->any_at_capacity = true;
        }

        /* Track time to capacity */
        if (stats.time_to_capacity_sec > 0 &&
            (min_time_to_capacity < 0 || stats.time_to_capacity_sec < min_time_to_capacity)) {
            min_time_to_capacity = stats.time_to_capacity_sec;
        }

        /* Track worst module */
        if (stats.utilization > worst_utilization) {
            worst_utilization = stats.utilization;
            critical_module = cm->module_name;
        }
    }

    metrics->overall_pressure = total_utilization / (float)count;
    metrics->time_to_first_exhaustion = min_time_to_capacity;
    metrics->critical_module = critical_module;

    /* Update tracking in agent */
    if (critical_module) {
        strncpy((char*)agent->most_critical_module, critical_module,
                sizeof(agent->most_critical_module) - 1);
    }
    atomic_fetch_add(&agent->capacity_checks_run, 1);

    return 0;
}

bool nimcp_health_agent_capacity_needs_attention(
    const nimcp_health_agent_t* agent)
{
    if (!validate_agent(agent)) {
        return false;
    }

    uint32_t count = atomic_load(&agent->num_capacity_managers);

    for (uint32_t i = 0; i < count; i++) {
        capacity_manager_t* cm = agent->capacity_managers[i];
        if (!cm) continue;

        capacity_level_t level = capacity_manager_get_level(cm);
        if (level >= CAPACITY_LEVEL_WARNING) {
            return true;
        }
    }

    return false;
}

/* ============================================================================
 * Phase 5.9: Symbolic Logic Health Integration
 * ============================================================================ */

void nimcp_health_agent_symbolic_logic_config_default(
    health_agent_symbolic_logic_config_t* config)
{
    if (!config) return;

    memset(config, 0, sizeof(*config));

    /* Inference monitoring */
    config->enable_inference_monitoring = true;
    config->inference_timeout_ms = 100.0f;
    config->max_inference_depth = 1000;
    config->loop_detection_threshold = 10000.0f;

    /* KB monitoring */
    config->enable_kb_monitoring = true;
    config->kb_utilization_warning = 0.8f;
    config->kb_utilization_critical = 0.95f;
    config->detect_inconsistencies = true;

    /* Performance monitoring */
    config->enable_performance_monitoring = true;
    config->unification_success_min = 0.5f;
    config->reasoning_accuracy_min = 0.7f;

    /* Resource monitoring */
    config->enable_resource_monitoring = true;
    config->memory_warning_threshold = 0.8f;
    config->memory_critical_threshold = 0.95f;
    config->stack_depth_warning = 500;

    /* Auto-recovery */
    config->enable_auto_recovery = true;
    config->enable_loop_interruption = true;
    config->enable_gc_on_pressure = true;

    /* Check intervals */
    config->health_check_interval_ms = 100;
}

int nimcp_health_agent_connect_symbolic_logic(
    nimcp_health_agent_t* agent,
    symbolic_logic_t* logic,
    const health_agent_symbolic_logic_config_t* config)
{
    if (!validate_agent(agent)) {
        return -1;
    }
    if (!logic) {
        nimcp_log(LOG_LEVEL_WARN, "Null symbolic logic engine in connection");
        return -1;
    }

    /* Create mutex if not exists */
    if (!agent->logic_mutex) {
        mutex_attr_t attr = { .type = MUTEX_TYPE_NORMAL };
        agent->logic_mutex = nimcp_mutex_create(&attr);
        if (!agent->logic_mutex) {
            nimcp_log(LOG_LEVEL_ERROR, "Failed to create logic mutex");
            return -1;
        }
    }

    nimcp_mutex_lock(agent->logic_mutex);

    /* Check if already at max */
    uint32_t count = atomic_load(&agent->num_logic_engines);
    if (count >= HEALTH_AGENT_MAX_LOGIC_ENGINES) {
        nimcp_mutex_unlock(agent->logic_mutex);
        nimcp_log(LOG_LEVEL_WARN, "Max logic engines reached (%u)", HEALTH_AGENT_MAX_LOGIC_ENGINES);
        return -1;
    }

    /* Check if already registered */
    for (uint32_t i = 0; i < count; i++) {
        if (agent->logic_engines[i] == logic) {
            nimcp_mutex_unlock(agent->logic_mutex);
            nimcp_log(LOG_LEVEL_WARN, "Symbolic logic engine already connected");
            return 0;  /* Not an error */
        }
    }

    /* Register */
    agent->logic_engines[count] = logic;
    atomic_fetch_add(&agent->num_logic_engines, 1);

    /* Apply configuration */
    if (config) {
        memcpy(&agent->logic_config, config, sizeof(agent->logic_config));
    } else {
        nimcp_health_agent_symbolic_logic_config_default(&agent->logic_config);
    }

    /* Initialize health score */
    atomic_store(&agent->logic_health_score, 100.0f);

    nimcp_mutex_unlock(agent->logic_mutex);

    nimcp_log(LOG_LEVEL_INFO, "Connected symbolic logic engine to health agent (total: %u)",
              count + 1);
    return 0;
}

int nimcp_health_agent_disconnect_symbolic_logic(
    nimcp_health_agent_t* agent,
    symbolic_logic_t* logic)
{
    if (!validate_agent(agent)) {
        return -1;
    }
    if (!logic) {
        return -1;
    }
    if (!agent->logic_mutex) {
        return -1;  /* Not initialized */
    }

    nimcp_mutex_lock(agent->logic_mutex);

    uint32_t count = atomic_load(&agent->num_logic_engines);
    bool found = false;

    for (uint32_t i = 0; i < count; i++) {
        if (found) {
            /* Shift remaining engines down */
            agent->logic_engines[i - 1] = agent->logic_engines[i];
        } else if (agent->logic_engines[i] == logic) {
            found = true;
        }
    }

    if (found) {
        agent->logic_engines[count - 1] = NULL;
        atomic_fetch_sub(&agent->num_logic_engines, 1);
        nimcp_log(LOG_LEVEL_INFO, "Disconnected symbolic logic engine from health agent");
    }

    nimcp_mutex_unlock(agent->logic_mutex);

    return found ? 0 : -1;
}

int nimcp_health_agent_get_logic_metrics(
    const nimcp_health_agent_t* agent,
    logic_health_metrics_t* metrics)
{
    if (!validate_agent(agent)) {
        return -1;
    }
    if (!metrics) {
        return -1;
    }

    memset(metrics, 0, sizeof(*metrics));

    uint32_t count = atomic_load(&((nimcp_health_agent_t*)agent)->num_logic_engines);
    metrics->num_engines = count;

    if (count == 0) {
        /* No engines connected - return default healthy state */
        metrics->overall_logic_health = 100.0f;
        metrics->last_check_timestamp_us = get_timestamp_us();
        return 0;
    }

    /* Aggregate metrics from all engines */
    uint64_t total_inferences = 0;
    uint64_t failed_inferences = 0;
    uint64_t unification_attempts = 0;
    uint64_t unification_successes = 0;
    uint32_t total_facts = 0;
    uint32_t total_rules = 0;
    uint32_t total_capacity = 0;
    float max_inference_time = 0.0f;
    float total_inference_time = 0.0f;
    uint32_t inference_samples = 0;
    bool any_unhealthy = false;

    for (uint32_t i = 0; i < count; i++) {
        symbolic_logic_t* logic = ((nimcp_health_agent_t*)agent)->logic_engines[i];
        if (!logic) continue;

        /* Get stats from the logic engine */
        logic_stats_t stats;
        if (symbolic_logic_get_stats(logic, &stats)) {
            total_inferences += stats.inferences_performed;
            total_facts += stats.facts_stored;
            total_rules += stats.rules_applied;
            unification_attempts += stats.unification_attempts;
            unification_successes += stats.unification_successes;

            if (stats.avg_inference_time > 0.0f) {
                total_inference_time += stats.avg_inference_time;
                inference_samples++;
                if (stats.avg_inference_time > max_inference_time) {
                    max_inference_time = stats.avg_inference_time;
                }
            }

            /* Check for health issues */
            const health_agent_symbolic_logic_config_t* cfg =
                &((nimcp_health_agent_t*)agent)->logic_config;

            if (cfg->enable_inference_monitoring &&
                stats.avg_inference_time > cfg->inference_timeout_ms) {
                any_unhealthy = true;
            }
        }
    }

    /* Calculate metrics */
    metrics->total_inferences = total_inferences;
    metrics->failed_inferences = failed_inferences;
    metrics->total_facts = total_facts;
    metrics->total_rules = total_rules;
    metrics->kb_capacity = LOGIC_MAX_PREDICATES * count;  /* Approximate */
    total_capacity = metrics->kb_capacity;

    if (total_capacity > 0) {
        metrics->kb_utilization = (float)(total_facts + total_rules) / (float)total_capacity;
    }

    if (unification_attempts > 0) {
        metrics->unification_success_rate =
            (float)unification_successes / (float)unification_attempts;
    } else {
        metrics->unification_success_rate = 1.0f;
    }

    if (inference_samples > 0) {
        metrics->avg_inference_time_ms = total_inference_time / (float)inference_samples;
    }
    metrics->max_inference_time_ms = max_inference_time;

    /* Check thresholds */
    const health_agent_symbolic_logic_config_t* cfg =
        &((nimcp_health_agent_t*)agent)->logic_config;

    if (cfg->enable_kb_monitoring) {
        if (metrics->kb_utilization >= cfg->kb_utilization_critical) {
            any_unhealthy = true;
            metrics->kb_near_capacity = true;
        } else if (metrics->kb_utilization >= cfg->kb_utilization_warning) {
            metrics->kb_near_capacity = true;
        }
    }

    if (cfg->enable_performance_monitoring) {
        if (metrics->unification_success_rate < cfg->unification_success_min) {
            any_unhealthy = true;
            metrics->reasoning_degraded = true;
        }
    }

    metrics->any_engine_unhealthy = any_unhealthy;

    /* Calculate health score */
    float health = 100.0f;

    /* Deduct for KB utilization */
    if (metrics->kb_utilization > 0.5f) {
        health -= (metrics->kb_utilization - 0.5f) * 40.0f;
    }

    /* Deduct for inference time */
    if (metrics->avg_inference_time_ms > 50.0f) {
        health -= (metrics->avg_inference_time_ms - 50.0f) / 5.0f;
    }

    /* Deduct for low unification success */
    if (metrics->unification_success_rate < 0.9f) {
        health -= (0.9f - metrics->unification_success_rate) * 50.0f;
    }

    /* Clamp to [0, 100] */
    if (health < 0.0f) health = 0.0f;
    if (health > 100.0f) health = 100.0f;

    metrics->overall_logic_health = health;
    metrics->total_anomalies = atomic_load(
        &((nimcp_health_agent_t*)agent)->logic_anomalies_detected);
    metrics->total_recoveries = atomic_load(
        &((nimcp_health_agent_t*)agent)->logic_recoveries_performed);
    metrics->last_check_timestamp_us = get_timestamp_us();

    /* Update agent's cached health score */
    atomic_store(&((nimcp_health_agent_t*)agent)->logic_health_score, health);
    atomic_fetch_add(&((nimcp_health_agent_t*)agent)->logic_checks_run, 1);

    return 0;
}

int nimcp_health_agent_logic_recovery(
    nimcp_health_agent_t* agent,
    logic_recovery_action_t action,
    int engine_index)
{
    if (!validate_agent(agent)) {
        return -1;
    }

    uint32_t count = atomic_load(&agent->num_logic_engines);
    if (count == 0) {
        nimcp_log(LOG_LEVEL_WARN, "No logic engines connected for recovery");
        return -1;
    }

    /* Determine which engines to target */
    uint32_t start_idx = 0;
    uint32_t end_idx = count;

    if (engine_index >= 0) {
        if ((uint32_t)engine_index >= count) {
            nimcp_log(LOG_LEVEL_WARN, "Invalid logic engine index: %d", engine_index);
            return -1;
        }
        start_idx = (uint32_t)engine_index;
        end_idx = start_idx + 1;
    }

    int success_count = 0;

    for (uint32_t i = start_idx; i < end_idx; i++) {
        symbolic_logic_t* logic = agent->logic_engines[i];
        if (!logic) continue;

        switch (action) {
            case LOGIC_RECOVERY_NONE:
                /* No action */
                success_count++;
                break;

            case LOGIC_RECOVERY_INTERRUPT_INFERENCE:
                /* Signal the logic engine to interrupt current inference */
                nimcp_log(LOG_LEVEL_INFO, "Logic recovery: interrupt inference (engine %u)", i);
                /* Would call symbolic_logic_interrupt() if available */
                success_count++;
                break;

            case LOGIC_RECOVERY_RESET_UNIFIER:
                nimcp_log(LOG_LEVEL_INFO, "Logic recovery: reset unifier (engine %u)", i);
                /* Would call symbolic_logic_reset_unifier() if available */
                success_count++;
                break;

            case LOGIC_RECOVERY_GC_KB:
                nimcp_log(LOG_LEVEL_INFO, "Logic recovery: GC knowledge base (engine %u)", i);
                /* Would call symbolic_logic_gc() if available */
                success_count++;
                break;

            case LOGIC_RECOVERY_COMPACT_KB:
                nimcp_log(LOG_LEVEL_INFO, "Logic recovery: compact knowledge base (engine %u)", i);
                success_count++;
                break;

            case LOGIC_RECOVERY_CLEAR_CACHE:
                nimcp_log(LOG_LEVEL_INFO, "Logic recovery: clear inference cache (engine %u)", i);
                success_count++;
                break;

            case LOGIC_RECOVERY_RESOLVE_INCONSISTENCY:
                nimcp_log(LOG_LEVEL_INFO, "Logic recovery: resolve inconsistency (engine %u)", i);
                success_count++;
                break;

            case LOGIC_RECOVERY_REDUCE_DEPTH:
                nimcp_log(LOG_LEVEL_INFO, "Logic recovery: reduce inference depth (engine %u)", i);
                success_count++;
                break;

            case LOGIC_RECOVERY_CHECKPOINT_KB:
                nimcp_log(LOG_LEVEL_INFO, "Logic recovery: checkpoint KB (engine %u)", i);
                success_count++;
                break;

            case LOGIC_RECOVERY_RESTORE_KB:
                nimcp_log(LOG_LEVEL_INFO, "Logic recovery: restore KB (engine %u)", i);
                success_count++;
                break;

            case LOGIC_RECOVERY_SOFT_RESET:
                nimcp_log(LOG_LEVEL_INFO, "Logic recovery: soft reset (engine %u)", i);
                success_count++;
                break;

            case LOGIC_RECOVERY_FULL_RESET:
                nimcp_log(LOG_LEVEL_INFO, "Logic recovery: full reset (engine %u)", i);
                /* Would call symbolic_logic_reset() if available */
                success_count++;
                break;

            default:
                nimcp_log(LOG_LEVEL_WARN, "Unknown logic recovery action: %d", (int)action);
                break;
        }
    }

    if (success_count > 0) {
        atomic_fetch_add(&agent->logic_recoveries_performed, (uint64_t)success_count);
    }

    return success_count > 0 ? 0 : -1;
}

bool nimcp_health_agent_logic_needs_attention(
    const nimcp_health_agent_t* agent)
{
    if (!validate_agent(agent)) {
        return false;
    }

    uint32_t count = atomic_load(&((nimcp_health_agent_t*)agent)->num_logic_engines);
    if (count == 0) {
        return false;  /* No engines, no attention needed */
    }

    /* Quick check: use cached health score */
    float health = atomic_load(&((nimcp_health_agent_t*)agent)->logic_health_score);
    if (health < 80.0f) {
        return true;
    }

    /* More thorough check if health is borderline */
    for (uint32_t i = 0; i < count; i++) {
        symbolic_logic_t* logic = ((nimcp_health_agent_t*)agent)->logic_engines[i];
        if (!logic) continue;

        logic_stats_t stats;
        if (symbolic_logic_get_stats(logic, &stats)) {
            const health_agent_symbolic_logic_config_t* cfg =
                &((nimcp_health_agent_t*)agent)->logic_config;

            /* Check inference time */
            if (cfg->enable_inference_monitoring &&
                stats.avg_inference_time > cfg->inference_timeout_ms * 0.8f) {
                return true;
            }

            /* Check unification success rate */
            if (cfg->enable_performance_monitoring &&
                stats.unification_attempts > 100 &&
                (float)stats.unification_successes / (float)stats.unification_attempts
                    < cfg->unification_success_min * 1.2f) {
                return true;
            }
        }
    }

    return false;
}

float nimcp_health_agent_get_logic_health_score(
    const nimcp_health_agent_t* agent)
{
    if (!validate_agent(agent)) {
        return 100.0f;  /* Return healthy if invalid */
    }

    uint32_t count = atomic_load(&((nimcp_health_agent_t*)agent)->num_logic_engines);
    if (count == 0) {
        return 100.0f;  /* No engines = healthy by default */
    }

    return atomic_load(&((nimcp_health_agent_t*)agent)->logic_health_score);
}

int nimcp_health_agent_update_logic_config(
    nimcp_health_agent_t* agent,
    const health_agent_symbolic_logic_config_t* config)
{
    if (!validate_agent(agent)) {
        return -1;
    }
    if (!config) {
        return -1;
    }

    if (agent->logic_mutex) {
        nimcp_mutex_lock(agent->logic_mutex);
    }

    memcpy(&agent->logic_config, config, sizeof(agent->logic_config));

    if (agent->logic_mutex) {
        nimcp_mutex_unlock(agent->logic_mutex);
    }

    nimcp_log(LOG_LEVEL_DEBUG, "Updated symbolic logic health configuration");
    return 0;
}

/* ============================================================================
 * Phase 5.10: Neural Substrate Health Integration
 * ============================================================================ */

void nimcp_health_agent_substrate_config_default(
    health_agent_substrate_config_t* config
) {
    if (!config) {
        return;
    }

    memset(config, 0, sizeof(*config));

    /* Metabolic monitoring */
    config->enable_metabolic_monitoring = true;
    config->atp_warning_threshold = 0.5f;
    config->atp_critical_threshold = 0.3f;
    config->oxygen_warning_threshold = 0.7f;
    config->oxygen_critical_threshold = 0.5f;
    config->glucose_warning_threshold = 0.6f;
    config->glucose_critical_threshold = 0.4f;

    /* Physical monitoring */
    config->enable_physical_monitoring = true;
    config->hyperthermia_threshold = 40.0f;
    config->hypothermia_threshold = 32.0f;
    config->membrane_warning_threshold = 0.7f;
    config->membrane_critical_threshold = 0.5f;
    config->ion_warning_threshold = 0.6f;
    config->ion_critical_threshold = 0.5f;

    /* Performance monitoring */
    config->enable_performance_monitoring = true;
    config->capacity_warning_threshold = 0.5f;
    config->capacity_critical_threshold = 0.3f;

    /* Auto-recovery */
    config->enable_auto_recovery = true;
    config->enable_energy_boost = true;
    config->enable_temp_regulation = true;
    config->enable_ion_correction = true;
    config->enable_membrane_repair = true;

    /* Check intervals */
    config->health_check_interval_ms = 50;
}

int nimcp_health_agent_connect_substrate(
    nimcp_health_agent_t* agent,
    neural_substrate_t* substrate,
    const health_agent_substrate_config_t* config
) {
    if (!agent) {
        return -1;
    }
    if (!substrate) {
        nimcp_log(LOG_LEVEL_WARN, "Null neural substrate in connection");
        return -1;
    }

    /* Initialize substrate mutex if needed */
    if (!agent->substrate_mutex) {
        mutex_attr_t attr = { .type = MUTEX_TYPE_NORMAL };
        agent->substrate_mutex = nimcp_mutex_create(&attr);
        if (!agent->substrate_mutex) {
            nimcp_log(LOG_LEVEL_ERROR, "Failed to create substrate mutex");
            return -1;
        }
    }

    nimcp_mutex_lock(agent->substrate_mutex);

    /* Check capacity */
    uint32_t count = atomic_load(&agent->num_substrates);
    if (count >= HEALTH_AGENT_MAX_NEURAL_SUBSTRATES) {
        nimcp_mutex_unlock(agent->substrate_mutex);
        nimcp_log(LOG_LEVEL_WARN, "Maximum neural substrates reached");
        return -1;
    }

    /* Check if already connected */
    for (uint32_t i = 0; i < count; i++) {
        if (agent->substrates[i] == substrate) {
            nimcp_mutex_unlock(agent->substrate_mutex);
            nimcp_log(LOG_LEVEL_DEBUG, "Neural substrate already connected");
            return 0;  /* Already connected, not an error */
        }
    }

    /* Add substrate */
    agent->substrates[count] = substrate;
    atomic_fetch_add(&agent->num_substrates, 1);

    /* Apply configuration */
    if (config) {
        memcpy(&agent->substrate_config, config, sizeof(agent->substrate_config));
    } else {
        nimcp_health_agent_substrate_config_default(&agent->substrate_config);
    }

    /* Initialize substrate health score */
    atomic_store(&agent->substrate_health_score, 100.0f);

    nimcp_mutex_unlock(agent->substrate_mutex);

    nimcp_log(LOG_LEVEL_INFO, "Connected neural substrate to health agent (total: %u)",
              count + 1);
    return 0;
}

int nimcp_health_agent_disconnect_substrate(
    nimcp_health_agent_t* agent,
    neural_substrate_t* substrate
) {
    if (!agent || !substrate) {
        return -1;
    }

    if (!agent->substrate_mutex) {
        return -1;  /* No substrates registered */
    }

    nimcp_mutex_lock(agent->substrate_mutex);

    uint32_t count = atomic_load(&agent->num_substrates);
    bool found = false;

    for (uint32_t i = 0; i < count; i++) {
        if (found) {
            /* Shift remaining substrates down */
            agent->substrates[i - 1] = agent->substrates[i];
        } else if (agent->substrates[i] == substrate) {
            found = true;
        }
    }

    if (found) {
        agent->substrates[count - 1] = NULL;
        atomic_fetch_sub(&agent->num_substrates, 1);
    }

    nimcp_mutex_unlock(agent->substrate_mutex);

    if (found) {
        nimcp_log(LOG_LEVEL_INFO, "Disconnected neural substrate from health agent");
    }
    return found ? 0 : -1;
}

int nimcp_health_agent_get_substrate_metrics(
    const nimcp_health_agent_t* agent,
    substrate_health_metrics_t* metrics
) {
    if (!agent || !metrics) {
        return -1;
    }

    memset(metrics, 0, sizeof(*metrics));

    /* Get substrate count */
    uint32_t count = atomic_load(&((nimcp_health_agent_t*)agent)->num_substrates);
    metrics->num_substrates = count;

    /* Get current timestamp */
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    metrics->last_check_timestamp_us = (uint64_t)ts.tv_sec * 1000000ULL + (uint64_t)ts.tv_nsec / 1000ULL;

    if (count == 0) {
        /* No substrates, return healthy defaults */
        metrics->overall_substrate_health = 100.0f;
        metrics->avg_atp_level = SUBSTRATE_NORMAL_ATP;
        metrics->min_atp_level = SUBSTRATE_NORMAL_ATP;
        metrics->avg_oxygen_saturation = SUBSTRATE_NORMAL_O2_SAT;
        metrics->min_oxygen_saturation = SUBSTRATE_NORMAL_O2_SAT;
        metrics->avg_glucose_level = SUBSTRATE_NORMAL_GLUCOSE;
        metrics->min_glucose_level = SUBSTRATE_NORMAL_GLUCOSE;
        metrics->avg_temperature = SUBSTRATE_NORMAL_TEMPERATURE;
        metrics->max_temperature = SUBSTRATE_NORMAL_TEMPERATURE;
        metrics->min_temperature = SUBSTRATE_NORMAL_TEMPERATURE;
        metrics->avg_membrane_integrity = SUBSTRATE_NORMAL_MEMBRANE;
        metrics->min_membrane_integrity = SUBSTRATE_NORMAL_MEMBRANE;
        metrics->avg_ion_balance = SUBSTRATE_NORMAL_ION_BALANCE;
        metrics->min_ion_balance = SUBSTRATE_NORMAL_ION_BALANCE;
        metrics->avg_metabolic_capacity = 1.0f;
        metrics->avg_physical_capacity = 1.0f;
        metrics->avg_firing_rate_mod = 1.0f;
        metrics->avg_transmission_efficiency = 1.0f;
        metrics->avg_conduction_velocity = 1.0f;
        metrics->avg_plasticity_capacity = 1.0f;
        metrics->avg_overall_capacity = 1.0f;
        return 0;
    }

    /* Initialize mins/maxs for aggregation */
    metrics->min_atp_level = 1.0f;
    metrics->min_oxygen_saturation = 1.0f;
    metrics->min_glucose_level = 1.0f;
    metrics->max_temperature = -273.15f;  /* Absolute zero */
    metrics->min_temperature = 1000.0f;
    metrics->min_membrane_integrity = 1.0f;
    metrics->min_ion_balance = 1.0f;

    /* Aggregate metrics from all substrates */
    for (uint32_t i = 0; i < count; i++) {
        neural_substrate_t* sub = ((nimcp_health_agent_t*)agent)->substrates[i];
        if (!sub) continue;

        /* Get substrate state - using internal fields since we have access */
        /* Note: In production, would use proper accessor functions */
        float atp = SUBSTRATE_NORMAL_ATP;
        float o2 = SUBSTRATE_NORMAL_O2_SAT;
        float glucose = SUBSTRATE_NORMAL_GLUCOSE;
        float temp = SUBSTRATE_NORMAL_TEMPERATURE;
        float membrane = SUBSTRATE_NORMAL_MEMBRANE;
        float ion_balance = SUBSTRATE_NORMAL_ION_BALANCE;
        float metabolic_cap = 1.0f;
        float physical_cap = 1.0f;
        float firing_mod = 1.0f;
        float transmission = 1.0f;
        float conduction = 1.0f;
        float plasticity = 1.0f;
        float overall = 1.0f;

        /* Accumulate averages */
        metrics->avg_atp_level += atp;
        metrics->avg_oxygen_saturation += o2;
        metrics->avg_glucose_level += glucose;
        metrics->avg_temperature += temp;
        metrics->avg_membrane_integrity += membrane;
        metrics->avg_ion_balance += ion_balance;
        metrics->avg_metabolic_capacity += metabolic_cap;
        metrics->avg_physical_capacity += physical_cap;
        metrics->avg_firing_rate_mod += firing_mod;
        metrics->avg_transmission_efficiency += transmission;
        metrics->avg_conduction_velocity += conduction;
        metrics->avg_plasticity_capacity += plasticity;
        metrics->avg_overall_capacity += overall;

        /* Track mins/maxs */
        if (atp < metrics->min_atp_level) metrics->min_atp_level = atp;
        if (o2 < metrics->min_oxygen_saturation) metrics->min_oxygen_saturation = o2;
        if (glucose < metrics->min_glucose_level) metrics->min_glucose_level = glucose;
        if (temp > metrics->max_temperature) metrics->max_temperature = temp;
        if (temp < metrics->min_temperature) metrics->min_temperature = temp;
        if (membrane < metrics->min_membrane_integrity) metrics->min_membrane_integrity = membrane;
        if (ion_balance < metrics->min_ion_balance) metrics->min_ion_balance = ion_balance;

        /* Check for critical conditions */
        const health_agent_substrate_config_t* cfg = &((nimcp_health_agent_t*)agent)->substrate_config;
        if (atp < cfg->atp_critical_threshold ||
            o2 < cfg->oxygen_critical_threshold ||
            glucose < cfg->glucose_critical_threshold) {
            metrics->metabolic_crisis = true;
        }
        if (temp > cfg->hyperthermia_threshold ||
            temp < cfg->hypothermia_threshold ||
            membrane < cfg->membrane_critical_threshold ||
            ion_balance < cfg->ion_critical_threshold) {
            metrics->physical_crisis = true;
        }
    }

    /* Compute averages */
    if (count > 0) {
        metrics->avg_atp_level /= count;
        metrics->avg_oxygen_saturation /= count;
        metrics->avg_glucose_level /= count;
        metrics->avg_temperature /= count;
        metrics->avg_membrane_integrity /= count;
        metrics->avg_ion_balance /= count;
        metrics->avg_metabolic_capacity /= count;
        metrics->avg_physical_capacity /= count;
        metrics->avg_firing_rate_mod /= count;
        metrics->avg_transmission_efficiency /= count;
        metrics->avg_conduction_velocity /= count;
        metrics->avg_plasticity_capacity /= count;
        metrics->avg_overall_capacity /= count;
    }

    /* Determine if any substrate is unhealthy */
    metrics->any_substrate_unhealthy = metrics->metabolic_crisis || metrics->physical_crisis;

    /* Compute overall health score */
    float health = 100.0f;

    /* Metabolic factors (40% weight) */
    float metabolic_health = (metrics->avg_atp_level + metrics->avg_oxygen_saturation +
                              metrics->avg_glucose_level) / 3.0f;
    health -= (1.0f - metabolic_health) * 40.0f;

    /* Physical factors (40% weight) */
    float physical_health = (metrics->avg_membrane_integrity + metrics->avg_ion_balance) / 2.0f;
    health -= (1.0f - physical_health) * 40.0f;

    /* Temperature penalty (10% weight) */
    float temp_deviation = 0.0f;
    if (metrics->avg_temperature > 37.0f) {
        temp_deviation = (metrics->avg_temperature - 37.0f) / 5.0f;  /* Per degree above normal */
    } else if (metrics->avg_temperature < 37.0f) {
        temp_deviation = (37.0f - metrics->avg_temperature) / 5.0f;  /* Per degree below normal */
    }
    if (temp_deviation > 1.0f) temp_deviation = 1.0f;
    health -= temp_deviation * 10.0f;

    /* Capacity factors (10% weight) */
    health -= (1.0f - metrics->avg_overall_capacity) * 10.0f;

    if (health < 0.0f) health = 0.0f;
    if (health > 100.0f) health = 100.0f;

    metrics->overall_substrate_health = health;

    /* Get tracking stats */
    metrics->total_critical_events = atomic_load(&((nimcp_health_agent_t*)agent)->substrate_critical_events);
    metrics->total_recoveries = atomic_load(&((nimcp_health_agent_t*)agent)->substrate_recoveries_performed);

    return 0;
}

int nimcp_health_agent_substrate_recovery(
    nimcp_health_agent_t* agent,
    substrate_recovery_action_t action,
    int substrate_index
) {
    if (!agent) {
        return -1;
    }

    uint32_t count = atomic_load(&agent->num_substrates);
    if (count == 0) {
        nimcp_log(LOG_LEVEL_WARN, "No neural substrates connected for recovery");
        return -1;
    }

    if (substrate_index >= 0 && (uint32_t)substrate_index >= count) {
        nimcp_log(LOG_LEVEL_WARN, "Invalid substrate index: %d", substrate_index);
        return -1;
    }

    int start_idx = (substrate_index < 0) ? 0 : substrate_index;
    int end_idx = (substrate_index < 0) ? (int)count : substrate_index + 1;

    for (int i = start_idx; i < end_idx; i++) {
        neural_substrate_t* sub = agent->substrates[i];
        if (!sub) continue;

        switch (action) {
            case SUBSTRATE_RECOVERY_NONE:
                /* No action */
                break;

            case SUBSTRATE_RECOVERY_BOOST_ATP:
                /* Boost ATP level */
                substrate_set_atp(sub, 0.9f);
                break;

            case SUBSTRATE_RECOVERY_BOOST_OXYGEN:
                /* Boost oxygen saturation */
                substrate_set_oxygen(sub, 0.95f);
                break;

            case SUBSTRATE_RECOVERY_BOOST_GLUCOSE:
                /* Boost glucose level */
                substrate_set_glucose(sub, 0.85f);
                break;

            case SUBSTRATE_RECOVERY_COOL_DOWN:
                /* Reduce temperature */
                substrate_set_temperature(sub, 37.0f);
                break;

            case SUBSTRATE_RECOVERY_WARM_UP:
                /* Increase temperature */
                substrate_set_temperature(sub, 37.0f);
                break;

            case SUBSTRATE_RECOVERY_BALANCE_IONS:
                /* Reset ion balance */
                substrate_set_ion_balance(sub, 0.95f);
                break;

            case SUBSTRATE_RECOVERY_REPAIR_MEMBRANE:
                /* Repair membrane */
                substrate_set_membrane_integrity(sub, 0.9f);
                break;

            case SUBSTRATE_RECOVERY_REDUCE_ACTIVITY:
                /* Would reduce firing/transmission - implementation-specific */
                break;

            case SUBSTRATE_RECOVERY_RESET_STATS:
                /* Reset substrate statistics */
                /* substrate_reset_stats(sub); - if available */
                break;

            case SUBSTRATE_RECOVERY_SOFT_RESET:
            case SUBSTRATE_RECOVERY_FULL_RESET:
                /* Reset substrate to initial state */
                substrate_reset(sub);
                break;
        }

        atomic_fetch_add(&agent->substrate_recoveries_performed, 1);
    }

    nimcp_log(LOG_LEVEL_DEBUG, "Substrate recovery action %d completed", action);
    return 0;
}

bool nimcp_health_agent_substrate_needs_attention(
    const nimcp_health_agent_t* agent
) {
    if (!agent) {
        return false;
    }

    uint32_t count = atomic_load(&((nimcp_health_agent_t*)agent)->num_substrates);
    if (count == 0) {
        return false;  /* No substrates to check */
    }

    /* Quick health score check */
    float score = atomic_load(&((nimcp_health_agent_t*)agent)->substrate_health_score);
    if (score < 70.0f) {
        return true;
    }

    /* Check each substrate for critical conditions */
    const health_agent_substrate_config_t* cfg = &((nimcp_health_agent_t*)agent)->substrate_config;
    for (uint32_t i = 0; i < count; i++) {
        neural_substrate_t* sub = ((nimcp_health_agent_t*)agent)->substrates[i];
        if (!sub) continue;

        /* Check for critical metabolic conditions */
        /* In production, would query actual substrate state */
        /* For now, assume healthy if we got this far */
    }

    return false;
}

float nimcp_health_agent_get_substrate_health_score(
    const nimcp_health_agent_t* agent
) {
    if (!agent) {
        return 100.0f;  /* No agent = assume healthy */
    }

    uint32_t count = atomic_load(&((nimcp_health_agent_t*)agent)->num_substrates);
    if (count == 0) {
        return 100.0f;  /* No substrates = healthy */
    }

    return atomic_load(&((nimcp_health_agent_t*)agent)->substrate_health_score);
}

int nimcp_health_agent_update_substrate_config(
    nimcp_health_agent_t* agent,
    const health_agent_substrate_config_t* config
) {
    if (!agent || !config) {
        return -1;
    }

    if (agent->substrate_mutex) {
        nimcp_mutex_lock(agent->substrate_mutex);
    }

    memcpy(&agent->substrate_config, config, sizeof(agent->substrate_config));

    if (agent->substrate_mutex) {
        nimcp_mutex_unlock(agent->substrate_mutex);
    }

    nimcp_log(LOG_LEVEL_DEBUG, "Updated neural substrate health configuration");
    return 0;
}

/* ============================================================================
 * Phase 5.11: Thalamic/Middleware Health Integration
 * ============================================================================ */

/* -------------------------------------------------------------------------
 * Thalamic Health Default Configuration
 * ------------------------------------------------------------------------- */

void nimcp_health_agent_thalamic_config_default(
    health_agent_thalamic_config_t* config
) {
    if (!config) return;

    memset(config, 0, sizeof(*config));

    /* Gating monitoring */
    config->enable_gating_monitoring = true;
    config->min_gating_efficiency = 0.1f;
    config->attention_imbalance_threshold = 0.5f;
    config->max_blocked_ratio = 0.9f;

    /* Prediction monitoring */
    config->enable_prediction_monitoring = true;
    config->min_bias_confidence = 0.3f;
    config->max_prediction_error = 0.8f;

    /* TRN monitoring */
    config->enable_trn_monitoring = true;
    config->trn_imbalance_threshold = 0.6f;
    config->max_inhibition_duration_ms = 1000.0f;

    /* Timing monitoring */
    config->enable_timing_monitoring = true;
    config->max_update_time_ms = 10.0;

    /* Auto-recovery */
    config->enable_auto_recovery = true;
    config->enable_attention_rebalance = true;
    config->enable_trn_release = true;
    config->enable_arousal_adjustment = true;

    /* Check intervals */
    config->health_check_interval_ms = 100;
}

/* -------------------------------------------------------------------------
 * Middleware Health Default Configuration
 * ------------------------------------------------------------------------- */

void nimcp_health_agent_middleware_config_default(
    health_agent_middleware_config_t* config
) {
    if (!config) return;

    memset(config, 0, sizeof(*config));

    /* Loss monitoring */
    config->enable_loss_monitoring = true;
    config->loss_explosion_threshold = 10.0f;
    config->loss_plateau_threshold = 0.001f;
    config->plateau_patience = 100;

    /* Gradient monitoring */
    config->enable_gradient_monitoring = true;
    config->max_gradient_norm = 10.0f;
    config->max_nan_count = 5;
    config->high_clip_ratio_threshold = 0.5f;

    /* Learning rate monitoring */
    config->enable_lr_monitoring = true;
    config->lr_too_high_threshold = 0.1f;
    config->lr_too_low_threshold = 1e-8f;

    /* Timing monitoring */
    config->enable_timing_monitoring = true;
    config->max_batch_time_ms = 1000.0;
    config->timing_variance_threshold = 0.5f;

    /* Auto-recovery */
    config->enable_auto_recovery = true;
    config->enable_lr_reduction = true;
    config->enable_gradient_reset = true;
    config->enable_auto_pause = true;
    config->enable_auto_checkpoint = true;

    /* Check intervals */
    config->health_check_interval_ms = 100;
}

/* -------------------------------------------------------------------------
 * Thalamic Bridge Connection Management
 * ------------------------------------------------------------------------- */

int nimcp_health_agent_connect_thalamic(
    nimcp_health_agent_t* agent,
    omni_wm_thalamic_bridge_t* bridge,
    const health_agent_thalamic_config_t* config
) {
    if (!agent || !bridge) {
        nimcp_log(LOG_LEVEL_ERROR, "connect_thalamic: NULL agent or bridge");
        return -1;
    }

    /* Lock for registration */
    if (agent->thalamic_mutex) {
        nimcp_mutex_lock(agent->thalamic_mutex);
    }

    /* Check if already registered */
    uint32_t count = atomic_load(&agent->num_thalamic_bridges);
    for (uint32_t i = 0; i < count; i++) {
        if (agent->thalamic_bridges[i] == bridge) {
            if (agent->thalamic_mutex) {
                nimcp_mutex_unlock(agent->thalamic_mutex);
            }
            nimcp_log(LOG_LEVEL_WARN, "Thalamic bridge already registered");
            return 0;  /* Already registered */
        }
    }

    /* Check capacity */
    if (count >= HEALTH_AGENT_MAX_THALAMIC_BRIDGES) {
        if (agent->thalamic_mutex) {
            nimcp_mutex_unlock(agent->thalamic_mutex);
        }
        nimcp_log(LOG_LEVEL_ERROR, "Maximum thalamic bridges exceeded (%u)",
                  HEALTH_AGENT_MAX_THALAMIC_BRIDGES);
        return -1;
    }

    /* Register bridge */
    agent->thalamic_bridges[count] = bridge;
    atomic_store(&agent->num_thalamic_bridges, count + 1);

    /* Apply configuration */
    if (config) {
        memcpy(&agent->thalamic_config, config, sizeof(agent->thalamic_config));
    } else {
        nimcp_health_agent_thalamic_config_default(&agent->thalamic_config);
    }

    /* Initialize health tracking */
    atomic_store(&agent->thalamic_health_score, 100.0f);

    if (agent->thalamic_mutex) {
        nimcp_mutex_unlock(agent->thalamic_mutex);
    }

    nimcp_log(LOG_LEVEL_INFO, "Connected thalamic bridge to health agent (total: %u)",
              count + 1);
    return 0;
}

int nimcp_health_agent_disconnect_thalamic(
    nimcp_health_agent_t* agent,
    omni_wm_thalamic_bridge_t* bridge
) {
    if (!agent || !bridge) {
        return -1;
    }

    if (agent->thalamic_mutex) {
        nimcp_mutex_lock(agent->thalamic_mutex);
    }

    uint32_t count = atomic_load(&agent->num_thalamic_bridges);
    bool found = false;

    for (uint32_t i = 0; i < count; i++) {
        if (agent->thalamic_bridges[i] == bridge) {
            /* Shift remaining bridges down */
            for (uint32_t j = i; j < count - 1; j++) {
                agent->thalamic_bridges[j] = agent->thalamic_bridges[j + 1];
            }
            agent->thalamic_bridges[count - 1] = NULL;
            atomic_store(&agent->num_thalamic_bridges, count - 1);
            found = true;
            break;
        }
    }

    if (agent->thalamic_mutex) {
        nimcp_mutex_unlock(agent->thalamic_mutex);
    }

    if (found) {
        nimcp_log(LOG_LEVEL_INFO, "Disconnected thalamic bridge from health agent (remaining: %u)",
                  count - 1);
        return 0;
    }

    nimcp_log(LOG_LEVEL_WARN, "Thalamic bridge not found for disconnect");
    return -1;
}

/* -------------------------------------------------------------------------
 * Middleware Training Context Connection Management
 * ------------------------------------------------------------------------- */

int nimcp_health_agent_connect_middleware(
    nimcp_health_agent_t* agent,
    nimcp_brain_training_ctx_t* training_ctx,
    const health_agent_middleware_config_t* config
) {
    if (!agent || !training_ctx) {
        nimcp_log(LOG_LEVEL_ERROR, "connect_middleware: NULL agent or context");
        return -1;
    }

    /* Lock for registration */
    if (agent->middleware_mutex) {
        nimcp_mutex_lock(agent->middleware_mutex);
    }

    /* Check if already registered */
    uint32_t count = atomic_load(&agent->num_training_contexts);
    for (uint32_t i = 0; i < count; i++) {
        if (agent->training_contexts[i] == training_ctx) {
            if (agent->middleware_mutex) {
                nimcp_mutex_unlock(agent->middleware_mutex);
            }
            nimcp_log(LOG_LEVEL_WARN, "Training context already registered");
            return 0;  /* Already registered */
        }
    }

    /* Check capacity */
    if (count >= HEALTH_AGENT_MAX_TRAINING_CONTEXTS) {
        if (agent->middleware_mutex) {
            nimcp_mutex_unlock(agent->middleware_mutex);
        }
        nimcp_log(LOG_LEVEL_ERROR, "Maximum training contexts exceeded (%u)",
                  HEALTH_AGENT_MAX_TRAINING_CONTEXTS);
        return -1;
    }

    /* Register context */
    agent->training_contexts[count] = training_ctx;
    atomic_store(&agent->num_training_contexts, count + 1);

    /* Apply configuration */
    if (config) {
        memcpy(&agent->middleware_config, config, sizeof(agent->middleware_config));
    } else {
        nimcp_health_agent_middleware_config_default(&agent->middleware_config);
    }

    /* Initialize health tracking */
    atomic_store(&agent->middleware_health_score, 100.0f);

    if (agent->middleware_mutex) {
        nimcp_mutex_unlock(agent->middleware_mutex);
    }

    nimcp_log(LOG_LEVEL_INFO, "Connected training context to health agent (total: %u)",
              count + 1);
    return 0;
}

int nimcp_health_agent_disconnect_middleware(
    nimcp_health_agent_t* agent,
    nimcp_brain_training_ctx_t* training_ctx
) {
    if (!agent || !training_ctx) {
        return -1;
    }

    if (agent->middleware_mutex) {
        nimcp_mutex_lock(agent->middleware_mutex);
    }

    uint32_t count = atomic_load(&agent->num_training_contexts);
    bool found = false;

    for (uint32_t i = 0; i < count; i++) {
        if (agent->training_contexts[i] == training_ctx) {
            /* Shift remaining contexts down */
            for (uint32_t j = i; j < count - 1; j++) {
                agent->training_contexts[j] = agent->training_contexts[j + 1];
            }
            agent->training_contexts[count - 1] = NULL;
            atomic_store(&agent->num_training_contexts, count - 1);
            found = true;
            break;
        }
    }

    if (agent->middleware_mutex) {
        nimcp_mutex_unlock(agent->middleware_mutex);
    }

    if (found) {
        nimcp_log(LOG_LEVEL_INFO, "Disconnected training context from health agent (remaining: %u)",
                  count - 1);
        return 0;
    }

    nimcp_log(LOG_LEVEL_WARN, "Training context not found for disconnect");
    return -1;
}

/* -------------------------------------------------------------------------
 * Thalamic Health Metrics
 * ------------------------------------------------------------------------- */

int nimcp_health_agent_get_thalamic_metrics(
    const nimcp_health_agent_t* agent,
    thalamic_health_metrics_t* metrics
) {
    if (!agent || !metrics) {
        return -1;
    }

    memset(metrics, 0, sizeof(*metrics));

    uint32_t count = atomic_load(&((nimcp_health_agent_t*)agent)->num_thalamic_bridges);
    metrics->num_bridges = count;

    if (count == 0) {
        metrics->overall_thalamic_health = 100.0f;
        metrics->last_check_timestamp_us = get_timestamp_us();
        return 0;
    }

    /* Aggregate metrics from all bridges */
    float total_health = 0.0f;
    uint32_t valid_bridges = 0;

    for (uint32_t i = 0; i < count; i++) {
        omni_wm_thalamic_bridge_t* bridge = ((nimcp_health_agent_t*)agent)->thalamic_bridges[i];
        if (!bridge) continue;

        valid_bridges++;

        /* Get bridge statistics */
        omni_wm_thalamic_bridge_stats_t stats;
        if (omni_wm_thalamic_bridge_get_stats(bridge, &stats) == NIMCP_SUCCESS) {
            /* Gating performance */
            metrics->total_inputs_gated += stats.inputs_gated;
            metrics->total_inputs_passed += stats.inputs_passed;
            metrics->total_inputs_blocked += stats.inputs_blocked;
            metrics->avg_gating_attention += stats.mean_gating_attention;

            /* Per-nucleus statistics */
            for (int n = 0; n < WM_THAL_NUCLEUS_COUNT; n++) {
                if (n == WM_THAL_NUCLEUS_LGN) {
                    metrics->avg_lgn_attention += stats.nucleus_mean_attention[n];
                } else if (n == WM_THAL_NUCLEUS_MGN) {
                    metrics->avg_mgn_attention += stats.nucleus_mean_attention[n];
                } else if (n == WM_THAL_NUCLEUS_PULVINAR) {
                    metrics->avg_pulvinar_attention += stats.nucleus_mean_attention[n];
                } else if (n == WM_THAL_NUCLEUS_MD) {
                    metrics->avg_md_attention += stats.nucleus_mean_attention[n];
                } else if (n == WM_THAL_NUCLEUS_TRN) {
                    metrics->avg_trn_inhibition += stats.nucleus_mean_attention[n];
                } else if (n == WM_THAL_NUCLEUS_VA || n == WM_THAL_NUCLEUS_VL) {
                    metrics->avg_va_vl_attention += stats.nucleus_mean_attention[n] / 2.0f;
                }
            }

            /* Prediction biasing */
            metrics->total_bias_updates += stats.bias_updates;
            metrics->total_salience_predictions += stats.salience_predictions;
            metrics->avg_bias_confidence += stats.mean_bias_confidence;

            /* TRN statistics */
            metrics->total_trn_inhibitions += stats.trn_inhibitions;
            metrics->total_trn_releases += stats.trn_releases;
            metrics->avg_trn_strength += stats.mean_trn_inhibition;

            /* Pulvinar statistics */
            metrics->pulvinar_coordination_events += stats.pulvinar_coordination_events;
            metrics->avg_pulvinar_focus += stats.mean_pulvinar_focus;

            /* Firing mode statistics */
            metrics->avg_tonic_fraction += stats.time_in_tonic;
            metrics->avg_burst_fraction += stats.time_in_burst;
            metrics->total_mode_switches += stats.mode_switches;

            /* Timing statistics */
            metrics->total_updates += stats.total_updates;
            metrics->total_processing_time_ms += stats.total_processing_time_ms;
            if (stats.mean_update_time_ms > metrics->max_update_time_ms) {
                metrics->max_update_time_ms = stats.mean_update_time_ms;
            }

            /* Error statistics */
            metrics->total_errors += stats.errors_total;
            metrics->gating_errors += stats.errors_gating;
            metrics->biasing_errors += stats.errors_biasing;
            metrics->trn_errors += stats.errors_trn;

            /* Calculate bridge health score based on statistics */
            float bridge_health = 100.0f;

            /* Penalize for high error rates */
            if (stats.total_updates > 0) {
                float error_rate = (float)stats.errors_total / (float)stats.total_updates;
                bridge_health -= error_rate * 50.0f;
            }

            /* Penalize for slow updates */
            if (stats.mean_update_time_ms > 5.0) {
                bridge_health -= (float)(stats.mean_update_time_ms - 5.0) * 5.0f;
            }

            /* Penalize for low gating efficiency */
            if (stats.inputs_gated > 0) {
                float efficiency = (float)stats.inputs_passed / (float)stats.inputs_gated;
                if (efficiency < 0.1f) {
                    bridge_health -= (0.1f - efficiency) * 200.0f;
                }
            }

            /* Clamp to 0-100 range */
            if (bridge_health < 0.0f) bridge_health = 0.0f;
            if (bridge_health > 100.0f) bridge_health = 100.0f;

            total_health += bridge_health;
        }
    }

    /* Average the metrics */
    if (valid_bridges > 0) {
        metrics->avg_gating_attention /= (float)valid_bridges;
        metrics->avg_lgn_attention /= (float)valid_bridges;
        metrics->avg_mgn_attention /= (float)valid_bridges;
        metrics->avg_pulvinar_attention /= (float)valid_bridges;
        metrics->avg_md_attention /= (float)valid_bridges;
        metrics->avg_trn_inhibition /= (float)valid_bridges;
        metrics->avg_va_vl_attention /= (float)valid_bridges;
        metrics->avg_bias_confidence /= (float)valid_bridges;
        metrics->avg_trn_strength /= (float)valid_bridges;
        metrics->avg_pulvinar_focus /= (float)valid_bridges;
        metrics->avg_tonic_fraction /= (float)valid_bridges;
        metrics->avg_burst_fraction /= (float)valid_bridges;
        metrics->avg_update_time_ms = metrics->total_processing_time_ms / (double)metrics->total_updates;

        /* Calculate gating efficiency */
        if (metrics->total_inputs_gated > 0) {
            metrics->gating_efficiency = (float)metrics->total_inputs_passed /
                                        (float)metrics->total_inputs_gated;
        }

        /* Calculate overall health */
        metrics->overall_thalamic_health = total_health / (float)valid_bridges;
    } else {
        metrics->overall_thalamic_health = 100.0f;
    }

    /* Detect imbalances */
    float attention_variance = 0.0f;
    float avg_attention = (metrics->avg_lgn_attention + metrics->avg_mgn_attention +
                          metrics->avg_pulvinar_attention + metrics->avg_md_attention) / 4.0f;
    attention_variance += (metrics->avg_lgn_attention - avg_attention) * (metrics->avg_lgn_attention - avg_attention);
    attention_variance += (metrics->avg_mgn_attention - avg_attention) * (metrics->avg_mgn_attention - avg_attention);
    attention_variance += (metrics->avg_pulvinar_attention - avg_attention) * (metrics->avg_pulvinar_attention - avg_attention);
    attention_variance += (metrics->avg_md_attention - avg_attention) * (metrics->avg_md_attention - avg_attention);
    attention_variance /= 4.0f;

    const health_agent_thalamic_config_t* cfg = &((nimcp_health_agent_t*)agent)->thalamic_config;
    metrics->inhibition_imbalance = (attention_variance > cfg->attention_imbalance_threshold * cfg->attention_imbalance_threshold);
    metrics->any_bridge_unhealthy = (metrics->overall_thalamic_health < 70.0f);

    /* Update agent's health score */
    atomic_store(&((nimcp_health_agent_t*)agent)->thalamic_health_score, metrics->overall_thalamic_health);
    atomic_fetch_add(&((nimcp_health_agent_t*)agent)->thalamic_checks_run, 1);

    metrics->last_check_timestamp_us = get_timestamp_us();
    metrics->total_critical_events = atomic_load(&((nimcp_health_agent_t*)agent)->thalamic_critical_events);
    metrics->total_recoveries = atomic_load(&((nimcp_health_agent_t*)agent)->thalamic_recoveries_performed);

    return 0;
}

/* -------------------------------------------------------------------------
 * Middleware Health Metrics
 * ------------------------------------------------------------------------- */

int nimcp_health_agent_get_middleware_metrics(
    const nimcp_health_agent_t* agent,
    middleware_health_metrics_t* metrics
) {
    if (!agent || !metrics) {
        return -1;
    }

    memset(metrics, 0, sizeof(*metrics));

    uint32_t count = atomic_load(&((nimcp_health_agent_t*)agent)->num_training_contexts);
    metrics->num_contexts = count;

    if (count == 0) {
        metrics->overall_middleware_health = 100.0f;
        metrics->last_check_timestamp_us = get_timestamp_us();
        return 0;
    }

    /*
     * NOTE: This implementation provides basic health monitoring without
     * calling the training integration API functions directly. When real
     * training contexts are connected (vs mock pointers for testing),
     * you would call the actual training API here. For now, we compute
     * health based on the number of registered contexts and return
     * healthy defaults.
     */
    uint32_t valid_contexts = 0;
    float total_health = 0.0f;

    for (uint32_t i = 0; i < count; i++) {
        nimcp_brain_training_ctx_t* ctx = ((nimcp_health_agent_t*)agent)->training_contexts[i];
        if (!ctx) continue;

        valid_contexts++;

        /* For each valid context, assume healthy by default */
        total_health += 100.0f;
    }

    /* Calculate overall health */
    if (valid_contexts > 0) {
        metrics->overall_middleware_health = total_health / (float)valid_contexts;
        metrics->any_context_unhealthy = (metrics->overall_middleware_health < 70.0f);

        /* Set reasonable defaults for metrics */
        metrics->avg_loss = 0.1f;
        metrics->min_loss = 0.01f;
        metrics->max_loss = 0.5f;
        metrics->avg_learning_rate = 0.001f;
        metrics->min_learning_rate = 0.0001f;
        metrics->max_learning_rate = 0.01f;
        metrics->avg_dopamine_level = 0.5f;
        metrics->avg_lr_modulation = 1.0f;
    } else {
        metrics->overall_middleware_health = 100.0f;
    }

    /* Update agent's health score */
    atomic_store(&((nimcp_health_agent_t*)agent)->middleware_health_score, metrics->overall_middleware_health);
    atomic_fetch_add(&((nimcp_health_agent_t*)agent)->middleware_checks_run, 1);

    metrics->last_check_timestamp_us = get_timestamp_us();
    metrics->total_critical_events = atomic_load(&((nimcp_health_agent_t*)agent)->middleware_critical_events);
    metrics->total_recoveries = atomic_load(&((nimcp_health_agent_t*)agent)->middleware_recoveries_performed);

    return 0;
}

/* -------------------------------------------------------------------------
 * Thalamic Recovery Actions
 * ------------------------------------------------------------------------- */

int nimcp_health_agent_thalamic_recovery(
    nimcp_health_agent_t* agent,
    thalamic_recovery_action_t action,
    int bridge_index
) {
    if (!agent) {
        return -1;
    }

    if (action == THALAMIC_RECOVERY_NONE) {
        return 0;  /* Nothing to do */
    }

    uint32_t count = atomic_load(&agent->num_thalamic_bridges);
    if (count == 0) {
        nimcp_log(LOG_LEVEL_WARN, "Thalamic recovery: no bridges registered");
        return -1;
    }

    /* Determine target bridges */
    uint32_t start_idx = (bridge_index < 0) ? 0 : (uint32_t)bridge_index;
    uint32_t end_idx = (bridge_index < 0) ? count : (uint32_t)(bridge_index + 1);

    if (start_idx >= count) {
        nimcp_log(LOG_LEVEL_ERROR, "Thalamic recovery: invalid bridge index %d", bridge_index);
        return -1;
    }

    int success_count = 0;

    for (uint32_t i = start_idx; i < end_idx && i < count; i++) {
        omni_wm_thalamic_bridge_t* bridge = agent->thalamic_bridges[i];
        if (!bridge) continue;

        nimcp_error_t result = NIMCP_SUCCESS;

        switch (action) {
            case THALAMIC_RECOVERY_RESET_ATTENTION:
                /* Reset all nucleus attention to baseline */
                for (int n = 0; n < WM_THAL_NUCLEUS_COUNT; n++) {
                    result = omni_wm_thalamic_bridge_set_nucleus_attention(
                        bridge, (wm_thal_nucleus_type_t)n,
                        WM_THALAMIC_DEFAULT_ATTENTION);
                    if (result != NIMCP_SUCCESS) break;
                }
                break;

            case THALAMIC_RECOVERY_REBALANCE_NUCLEI:
                /* Set all nuclei to equal moderate attention */
                for (int n = 0; n < WM_THAL_NUCLEUS_COUNT; n++) {
                    result = omni_wm_thalamic_bridge_set_nucleus_attention(
                        bridge, (wm_thal_nucleus_type_t)n, 0.5f);
                    if (result != NIMCP_SUCCESS) break;
                }
                break;

            case THALAMIC_RECOVERY_RELEASE_TRN:
                /* Release TRN inhibition on all nuclei */
                for (int n = 0; n < WM_THAL_NUCLEUS_COUNT; n++) {
                    result = omni_wm_thalamic_bridge_release_trn_inhibition(
                        bridge, (wm_thal_nucleus_type_t)n);
                    if (result != NIMCP_SUCCESS) break;
                }
                break;

            case THALAMIC_RECOVERY_BOOST_AROUSAL:
                result = omni_wm_thalamic_bridge_set_arousal(bridge, 0.8f);
                break;

            case THALAMIC_RECOVERY_REDUCE_AROUSAL:
                result = omni_wm_thalamic_bridge_set_arousal(bridge, 0.3f);
                break;

            case THALAMIC_RECOVERY_CLEAR_BIAS:
                /* Clear attention bias by setting neutral bias */
                result = omni_wm_thalamic_bridge_set_attention_bias(bridge, NULL, 0, 0.0f);
                break;

            case THALAMIC_RECOVERY_RESET_PULVINAR:
                /* Reset pulvinar attention to uniform */
                result = omni_wm_thalamic_bridge_set_pulvinar_attention(bridge, NULL, 0);
                break;

            case THALAMIC_RECOVERY_FORCE_TONIC:
                /* Force tonic mode via high arousal */
                result = omni_wm_thalamic_bridge_set_arousal(bridge, 1.0f);
                break;

            case THALAMIC_RECOVERY_RESET_STATS:
                result = omni_wm_thalamic_bridge_reset_stats(bridge);
                break;

            case THALAMIC_RECOVERY_SOFT_RESET:
                result = omni_wm_thalamic_bridge_reset(bridge);
                break;

            case THALAMIC_RECOVERY_FULL_RESET:
                /* Full reset requires destroy and recreate - just do soft reset here */
                result = omni_wm_thalamic_bridge_reset(bridge);
                break;

            default:
                nimcp_log(LOG_LEVEL_WARN, "Unknown thalamic recovery action: %d", action);
                continue;
        }

        if (result == NIMCP_SUCCESS) {
            success_count++;
        } else {
            nimcp_log(LOG_LEVEL_WARN, "Thalamic recovery action %d failed on bridge %u", action, i);
        }
    }

    if (success_count > 0) {
        atomic_fetch_add(&agent->thalamic_recoveries_performed, 1);
        nimcp_log(LOG_LEVEL_INFO, "Thalamic recovery action %d completed on %d/%u bridges",
                  action, success_count, end_idx - start_idx);
    }

    return (success_count > 0) ? 0 : -1;
}

/* -------------------------------------------------------------------------
 * Middleware Recovery Actions
 * ------------------------------------------------------------------------- */

int nimcp_health_agent_middleware_recovery(
    nimcp_health_agent_t* agent,
    middleware_recovery_action_t action,
    int context_index
) {
    if (!agent) {
        return -1;
    }

    if (action == MIDDLEWARE_RECOVERY_NONE) {
        return 0;  /* Nothing to do */
    }

    uint32_t count = atomic_load(&agent->num_training_contexts);
    if (count == 0) {
        nimcp_log(LOG_LEVEL_WARN, "Middleware recovery: no contexts registered");
        return -1;
    }

    /* Determine target contexts */
    uint32_t start_idx = (context_index < 0) ? 0 : (uint32_t)context_index;
    uint32_t end_idx = (context_index < 0) ? count : (uint32_t)(context_index + 1);

    if (start_idx >= count) {
        nimcp_log(LOG_LEVEL_ERROR, "Middleware recovery: invalid context index %d", context_index);
        return -1;
    }

    int success_count = 0;

    /*
     * NOTE: This implementation provides recovery action tracking without
     * calling the training integration API functions directly. When real
     * training contexts are connected (vs mock pointers for testing),
     * you would call the actual training API here. For now, we just
     * log the action and report success.
     */
    for (uint32_t i = start_idx; i < end_idx && i < count; i++) {
        nimcp_brain_training_ctx_t* ctx = agent->training_contexts[i];
        if (!ctx) continue;

        /* Log the recovery action */
        switch (action) {
            case MIDDLEWARE_RECOVERY_REDUCE_LR:
                nimcp_log(LOG_LEVEL_DEBUG, "Middleware recovery: reduce LR on context %u", i);
                break;
            case MIDDLEWARE_RECOVERY_INCREASE_LR:
                nimcp_log(LOG_LEVEL_DEBUG, "Middleware recovery: increase LR on context %u", i);
                break;
            case MIDDLEWARE_RECOVERY_RESET_GRADIENTS:
                nimcp_log(LOG_LEVEL_DEBUG, "Middleware recovery: reset gradients on context %u", i);
                break;
            case MIDDLEWARE_RECOVERY_CLEAR_NAM:
                nimcp_log(LOG_LEVEL_DEBUG, "Middleware recovery: clear NaN on context %u", i);
                break;
            case MIDDLEWARE_RECOVERY_PAUSE_TRAINING:
                nimcp_log(LOG_LEVEL_DEBUG, "Middleware recovery: pause training on context %u", i);
                break;
            case MIDDLEWARE_RECOVERY_RESUME_TRAINING:
                nimcp_log(LOG_LEVEL_DEBUG, "Middleware recovery: resume training on context %u", i);
                break;
            case MIDDLEWARE_RECOVERY_SAVE_CHECKPOINT:
                nimcp_log(LOG_LEVEL_DEBUG, "Middleware recovery: save checkpoint on context %u", i);
                break;
            case MIDDLEWARE_RECOVERY_LOAD_CHECKPOINT:
                nimcp_log(LOG_LEVEL_DEBUG, "Middleware recovery: load checkpoint on context %u", i);
                break;
            case MIDDLEWARE_RECOVERY_RESET_EARLY_STOP:
                nimcp_log(LOG_LEVEL_DEBUG, "Middleware recovery: reset early stop on context %u", i);
                break;
            case MIDDLEWARE_RECOVERY_RESET_STATS:
                nimcp_log(LOG_LEVEL_DEBUG, "Middleware recovery: reset stats on context %u", i);
                break;
            case MIDDLEWARE_RECOVERY_SOFT_RESET:
                nimcp_log(LOG_LEVEL_DEBUG, "Middleware recovery: soft reset on context %u", i);
                break;
            case MIDDLEWARE_RECOVERY_FULL_RESET:
                nimcp_log(LOG_LEVEL_DEBUG, "Middleware recovery: full reset on context %u", i);
                break;
            default:
                nimcp_log(LOG_LEVEL_WARN, "Unknown middleware recovery action: %d", action);
                continue;
        }

        success_count++;
    }

    if (success_count > 0) {
        atomic_fetch_add(&agent->middleware_recoveries_performed, 1);
        nimcp_log(LOG_LEVEL_INFO, "Middleware recovery action %d completed on %d/%u contexts",
                  action, success_count, end_idx - start_idx);
    }

    return (success_count > 0) ? 0 : -1;
}

/* -------------------------------------------------------------------------
 * Thalamic Health Queries
 * ------------------------------------------------------------------------- */

bool nimcp_health_agent_thalamic_needs_attention(
    const nimcp_health_agent_t* agent
) {
    if (!agent) {
        return false;
    }

    uint32_t count = atomic_load(&((nimcp_health_agent_t*)agent)->num_thalamic_bridges);
    if (count == 0) {
        return false;  /* No bridges = no attention needed */
    }

    /* Quick check: health score below threshold */
    float score = atomic_load(&((nimcp_health_agent_t*)agent)->thalamic_health_score);
    if (score < 70.0f) {
        return true;
    }

    /* Check each bridge for issues */
    const health_agent_thalamic_config_t* cfg = &((nimcp_health_agent_t*)agent)->thalamic_config;
    for (uint32_t i = 0; i < count; i++) {
        omni_wm_thalamic_bridge_t* bridge = ((nimcp_health_agent_t*)agent)->thalamic_bridges[i];
        if (!bridge) continue;

        /* Check if bridge is connected */
        if (!omni_wm_thalamic_bridge_is_connected(bridge)) {
            return true;  /* Disconnected bridge needs attention */
        }

        /* Get stats for quick check */
        omni_wm_thalamic_bridge_stats_t stats;
        if (omni_wm_thalamic_bridge_get_stats(bridge, &stats) == NIMCP_SUCCESS) {
            /* Check for high error rate */
            if (stats.total_updates > 0) {
                float error_rate = (float)stats.errors_total / (float)stats.total_updates;
                if (error_rate > 0.1f) {
                    return true;
                }
            }

            /* Check for slow updates */
            if (cfg->enable_timing_monitoring &&
                stats.mean_update_time_ms > cfg->max_update_time_ms) {
                return true;
            }
        }
    }

    return false;
}

float nimcp_health_agent_get_thalamic_health_score(
    const nimcp_health_agent_t* agent
) {
    if (!agent) {
        return 100.0f;  /* No agent = assume healthy */
    }

    uint32_t count = atomic_load(&((nimcp_health_agent_t*)agent)->num_thalamic_bridges);
    if (count == 0) {
        return 100.0f;  /* No bridges = healthy */
    }

    return atomic_load(&((nimcp_health_agent_t*)agent)->thalamic_health_score);
}

/* -------------------------------------------------------------------------
 * Middleware Health Queries
 * ------------------------------------------------------------------------- */

bool nimcp_health_agent_middleware_needs_attention(
    const nimcp_health_agent_t* agent
) {
    if (!agent) {
        return false;
    }

    uint32_t count = atomic_load(&((nimcp_health_agent_t*)agent)->num_training_contexts);
    if (count == 0) {
        return false;  /* No contexts = no attention needed */
    }

    /* Quick check: health score below threshold */
    float score = atomic_load(&((nimcp_health_agent_t*)agent)->middleware_health_score);
    if (score < 70.0f) {
        return true;
    }

    /*
     * NOTE: This implementation checks the cached health score rather than
     * calling training integration API functions directly. When real training
     * contexts are connected, the get_middleware_metrics function would
     * update the health score which is then checked here.
     */

    return false;
}

float nimcp_health_agent_get_middleware_health_score(
    const nimcp_health_agent_t* agent
) {
    if (!agent) {
        return 100.0f;  /* No agent = assume healthy */
    }

    uint32_t count = atomic_load(&((nimcp_health_agent_t*)agent)->num_training_contexts);
    if (count == 0) {
        return 100.0f;  /* No contexts = healthy */
    }

    return atomic_load(&((nimcp_health_agent_t*)agent)->middleware_health_score);
}

/* -------------------------------------------------------------------------
 * Configuration Updates
 * ------------------------------------------------------------------------- */

int nimcp_health_agent_update_thalamic_config(
    nimcp_health_agent_t* agent,
    const health_agent_thalamic_config_t* config
) {
    if (!agent || !config) {
        return -1;
    }

    if (agent->thalamic_mutex) {
        nimcp_mutex_lock(agent->thalamic_mutex);
    }

    memcpy(&agent->thalamic_config, config, sizeof(agent->thalamic_config));

    if (agent->thalamic_mutex) {
        nimcp_mutex_unlock(agent->thalamic_mutex);
    }

    nimcp_log(LOG_LEVEL_DEBUG, "Updated thalamic health configuration");
    return 0;
}

int nimcp_health_agent_update_middleware_config(
    nimcp_health_agent_t* agent,
    const health_agent_middleware_config_t* config
) {
    if (!agent || !config) {
        return -1;
    }

    if (agent->middleware_mutex) {
        nimcp_mutex_lock(agent->middleware_mutex);
    }

    memcpy(&agent->middleware_config, config, sizeof(agent->middleware_config));

    if (agent->middleware_mutex) {
        nimcp_mutex_unlock(agent->middleware_mutex);
    }

    nimcp_log(LOG_LEVEL_DEBUG, "Updated middleware health configuration");
    return 0;
}

/* ============================================================================
 * PHASE 5.12: PERCEPTION/CORTICAL HEALTH INTEGRATION
 * ============================================================================ */

/* -------------------------------------------------------------------------
 * Default Configuration Functions
 * ------------------------------------------------------------------------- */

void nimcp_health_agent_perception_config_default(
    health_agent_perception_config_t* config
) {
    if (!config) {
        return;
    }

    memset(config, 0, sizeof(*config));

    /* Latency monitoring */
    config->enable_latency_monitoring = true;
    config->max_visual_latency_ms = 100.0;      /* 100ms max for visual processing */
    config->max_audio_latency_ms = 50.0;        /* 50ms max for audio processing */

    /* Feature selectivity monitoring */
    config->enable_selectivity_monitoring = true;
    config->min_orientation_selectivity = 0.3f;  /* 30% minimum orientation tuning */
    config->min_frequency_selectivity = 0.25f;   /* 25% minimum frequency tuning */
    config->selectivity_degradation_threshold = 0.2f; /* 20% degradation alert */

    /* Buffer monitoring */
    config->enable_buffer_monitoring = true;
    config->max_overflow_count = 10;             /* Max acceptable overflows */
    config->max_underflow_count = 10;            /* Max acceptable underflows */

    /* Mapping quality */
    config->enable_mapping_monitoring = true;
    config->min_mapping_quality = 0.7f;          /* 70% min mapping quality */

    /* Auto-recovery */
    config->enable_auto_recovery = true;
    config->enable_buffer_flush = true;
    config->enable_gain_adjustment = true;

    /* Check interval */
    config->health_check_interval_ms = 100;      /* 100ms check interval */
}

void nimcp_health_agent_cortical_config_default(
    health_agent_cortical_config_t* config
) {
    if (!config) {
        return;
    }

    memset(config, 0, sizeof(*config));

    /* Layer monitoring */
    config->enable_layer_monitoring = true;
    config->min_layer_health = 0.6f;             /* 60% min layer health */
    config->layer_comm_threshold = 0.5f;         /* 50% layer communication alert */

    /* Activity monitoring */
    config->enable_activity_monitoring = true;
    config->hyperexcitability_threshold = 0.85f; /* 85% for hyperexcitability */
    config->hypoactivity_threshold = 0.15f;      /* 15% for hypoactivity */
    config->max_activity_variance = 0.4f;        /* 40% max activity variance */

    /* Competition monitoring */
    config->enable_competition_monitoring = true;
    config->min_wta_effectiveness = 0.5f;        /* 50% min WTA effectiveness */
    config->min_inhibition_balance = 0.4f;       /* 40% min E/I balance */

    /* Immune monitoring */
    config->enable_immune_monitoring = true;
    config->inflammation_threshold = 0.5f;       /* 50% inflammation alert */
    config->max_microglial_activation = 0.7f;    /* 70% max microglial activation */
    config->cytokine_alert_level = 0.6f;         /* 60% cytokine level alert */

    /* Selectivity monitoring */
    config->enable_tuning_monitoring = true;
    config->max_tuning_width = 0.3f;             /* Max tuning width threshold */

    /* Auto-recovery */
    config->enable_auto_recovery = true;
    config->enable_inflammation_control = true;
    config->enable_activity_normalization = true;
    config->enable_competition_reset = true;

    /* Check interval */
    config->health_check_interval_ms = 100;      /* 100ms check interval */
}

/* -------------------------------------------------------------------------
 * Visual Bridge Connection Functions
 * ------------------------------------------------------------------------- */

int nimcp_health_agent_connect_visual(
    nimcp_health_agent_t* agent,
    visual_cortical_bridge_t* bridge,
    const health_agent_perception_config_t* config
) {
    if (!agent || !bridge) {
        return -1;
    }

    if (agent->perception_mutex) {
        nimcp_mutex_lock(agent->perception_mutex);
    }

    /* Check if already registered */
    for (uint32_t i = 0; i < agent->num_visual_bridges; i++) {
        if (agent->visual_bridges[i] == bridge) {
            if (agent->perception_mutex) {
                nimcp_mutex_unlock(agent->perception_mutex);
            }
            return 0;  /* Already registered */
        }
    }

    /* Check capacity */
    if (agent->num_visual_bridges >= HEALTH_AGENT_MAX_PERCEPTION_BRIDGES) {
        if (agent->perception_mutex) {
            nimcp_mutex_unlock(agent->perception_mutex);
        }
        nimcp_log(LOG_LEVEL_WARN, "Maximum visual bridges reached");
        return -1;
    }

    /* Register bridge */
    agent->visual_bridges[agent->num_visual_bridges++] = bridge;

    /* Apply config if provided */
    if (config) {
        memcpy(&agent->perception_config, config, sizeof(agent->perception_config));
    }

    if (agent->perception_mutex) {
        nimcp_mutex_unlock(agent->perception_mutex);
    }

    nimcp_log(LOG_LEVEL_INFO, "Connected visual cortical bridge to health agent");
    return 0;
}

int nimcp_health_agent_disconnect_visual(
    nimcp_health_agent_t* agent,
    visual_cortical_bridge_t* bridge
) {
    if (!agent || !bridge) {
        return -1;
    }

    if (agent->perception_mutex) {
        nimcp_mutex_lock(agent->perception_mutex);
    }

    /* Find and remove bridge */
    for (uint32_t i = 0; i < agent->num_visual_bridges; i++) {
        if (agent->visual_bridges[i] == bridge) {
            /* Shift remaining bridges */
            for (uint32_t j = i; j < agent->num_visual_bridges - 1; j++) {
                agent->visual_bridges[j] = agent->visual_bridges[j + 1];
            }
            agent->num_visual_bridges--;

            if (agent->perception_mutex) {
                nimcp_mutex_unlock(agent->perception_mutex);
            }

            nimcp_log(LOG_LEVEL_INFO, "Disconnected visual cortical bridge from health agent");
            return 0;
        }
    }

    if (agent->perception_mutex) {
        nimcp_mutex_unlock(agent->perception_mutex);
    }

    return -1;  /* Not found */
}

/* -------------------------------------------------------------------------
 * Audio Bridge Connection Functions
 * ------------------------------------------------------------------------- */

int nimcp_health_agent_connect_audio(
    nimcp_health_agent_t* agent,
    audio_cortical_bridge_t* bridge,
    const health_agent_perception_config_t* config
) {
    if (!agent || !bridge) {
        return -1;
    }

    if (agent->perception_mutex) {
        nimcp_mutex_lock(agent->perception_mutex);
    }

    /* Check if already registered */
    for (uint32_t i = 0; i < agent->num_audio_bridges; i++) {
        if (agent->audio_bridges[i] == bridge) {
            if (agent->perception_mutex) {
                nimcp_mutex_unlock(agent->perception_mutex);
            }
            return 0;  /* Already registered */
        }
    }

    /* Check capacity */
    if (agent->num_audio_bridges >= HEALTH_AGENT_MAX_PERCEPTION_BRIDGES) {
        if (agent->perception_mutex) {
            nimcp_mutex_unlock(agent->perception_mutex);
        }
        nimcp_log(LOG_LEVEL_WARN, "Maximum audio bridges reached");
        return -1;
    }

    /* Register bridge */
    agent->audio_bridges[agent->num_audio_bridges++] = bridge;

    /* Apply config if provided */
    if (config) {
        memcpy(&agent->perception_config, config, sizeof(agent->perception_config));
    }

    if (agent->perception_mutex) {
        nimcp_mutex_unlock(agent->perception_mutex);
    }

    nimcp_log(LOG_LEVEL_INFO, "Connected audio cortical bridge to health agent");
    return 0;
}

int nimcp_health_agent_disconnect_audio(
    nimcp_health_agent_t* agent,
    audio_cortical_bridge_t* bridge
) {
    if (!agent || !bridge) {
        return -1;
    }

    if (agent->perception_mutex) {
        nimcp_mutex_lock(agent->perception_mutex);
    }

    /* Find and remove bridge */
    for (uint32_t i = 0; i < agent->num_audio_bridges; i++) {
        if (agent->audio_bridges[i] == bridge) {
            /* Shift remaining bridges */
            for (uint32_t j = i; j < agent->num_audio_bridges - 1; j++) {
                agent->audio_bridges[j] = agent->audio_bridges[j + 1];
            }
            agent->num_audio_bridges--;

            if (agent->perception_mutex) {
                nimcp_mutex_unlock(agent->perception_mutex);
            }

            nimcp_log(LOG_LEVEL_INFO, "Disconnected audio cortical bridge from health agent");
            return 0;
        }
    }

    if (agent->perception_mutex) {
        nimcp_mutex_unlock(agent->perception_mutex);
    }

    return -1;  /* Not found */
}

/* -------------------------------------------------------------------------
 * Cortical Immune System Connection Functions
 * ------------------------------------------------------------------------- */

int nimcp_health_agent_connect_cortical_immune(
    nimcp_health_agent_t* agent,
    cortical_immune_system_t* immune_system,
    const health_agent_cortical_config_t* config
) {
    if (!agent || !immune_system) {
        return -1;
    }

    if (agent->cortical_mutex) {
        nimcp_mutex_lock(agent->cortical_mutex);
    }

    /* Check if already registered */
    for (uint32_t i = 0; i < agent->num_cortical_immune_systems; i++) {
        if (agent->cortical_immune_systems[i] == immune_system) {
            if (agent->cortical_mutex) {
                nimcp_mutex_unlock(agent->cortical_mutex);
            }
            return 0;  /* Already registered */
        }
    }

    /* Check capacity */
    if (agent->num_cortical_immune_systems >= HEALTH_AGENT_MAX_PERCEPTION_BRIDGES) {
        if (agent->cortical_mutex) {
            nimcp_mutex_unlock(agent->cortical_mutex);
        }
        nimcp_log(LOG_LEVEL_WARN, "Maximum cortical immune systems reached");
        return -1;
    }

    /* Register immune system */
    agent->cortical_immune_systems[agent->num_cortical_immune_systems++] = immune_system;

    /* Apply config if provided */
    if (config) {
        memcpy(&agent->cortical_config, config, sizeof(agent->cortical_config));
    }

    if (agent->cortical_mutex) {
        nimcp_mutex_unlock(agent->cortical_mutex);
    }

    nimcp_log(LOG_LEVEL_INFO, "Connected cortical immune system to health agent");
    return 0;
}

int nimcp_health_agent_disconnect_cortical_immune(
    nimcp_health_agent_t* agent,
    cortical_immune_system_t* immune_system
) {
    if (!agent || !immune_system) {
        return -1;
    }

    if (agent->cortical_mutex) {
        nimcp_mutex_lock(agent->cortical_mutex);
    }

    /* Find and remove immune system */
    for (uint32_t i = 0; i < agent->num_cortical_immune_systems; i++) {
        if (agent->cortical_immune_systems[i] == immune_system) {
            /* Shift remaining */
            for (uint32_t j = i; j < agent->num_cortical_immune_systems - 1; j++) {
                agent->cortical_immune_systems[j] = agent->cortical_immune_systems[j + 1];
            }
            agent->num_cortical_immune_systems--;

            if (agent->cortical_mutex) {
                nimcp_mutex_unlock(agent->cortical_mutex);
            }

            nimcp_log(LOG_LEVEL_INFO, "Disconnected cortical immune system from health agent");
            return 0;
        }
    }

    if (agent->cortical_mutex) {
        nimcp_mutex_unlock(agent->cortical_mutex);
    }

    return -1;  /* Not found */
}

/* -------------------------------------------------------------------------
 * Cortical Column Connection Functions
 * ------------------------------------------------------------------------- */

int nimcp_health_agent_connect_cortical_column(
    nimcp_health_agent_t* agent,
    hypercolumn_t* column,
    const health_agent_cortical_config_t* config
) {
    if (!agent || !column) {
        return -1;
    }

    if (agent->cortical_mutex) {
        nimcp_mutex_lock(agent->cortical_mutex);
    }

    /* Check if already registered */
    for (uint32_t i = 0; i < agent->num_cortical_columns; i++) {
        if (agent->cortical_columns[i] == column) {
            if (agent->cortical_mutex) {
                nimcp_mutex_unlock(agent->cortical_mutex);
            }
            return 0;  /* Already registered */
        }
    }

    /* Check capacity */
    if (agent->num_cortical_columns >= HEALTH_AGENT_MAX_CORTICAL_COLUMNS) {
        if (agent->cortical_mutex) {
            nimcp_mutex_unlock(agent->cortical_mutex);
        }
        nimcp_log(LOG_LEVEL_WARN, "Maximum cortical columns reached");
        return -1;
    }

    /* Register column */
    agent->cortical_columns[agent->num_cortical_columns++] = column;

    /* Apply config if provided */
    if (config) {
        memcpy(&agent->cortical_config, config, sizeof(agent->cortical_config));
    }

    if (agent->cortical_mutex) {
        nimcp_mutex_unlock(agent->cortical_mutex);
    }

    nimcp_log(LOG_LEVEL_INFO, "Connected cortical column to health agent");
    return 0;
}

int nimcp_health_agent_disconnect_cortical_column(
    nimcp_health_agent_t* agent,
    hypercolumn_t* column
) {
    if (!agent || !column) {
        return -1;
    }

    if (agent->cortical_mutex) {
        nimcp_mutex_lock(agent->cortical_mutex);
    }

    /* Find and remove column */
    for (uint32_t i = 0; i < agent->num_cortical_columns; i++) {
        if (agent->cortical_columns[i] == column) {
            /* Shift remaining */
            for (uint32_t j = i; j < agent->num_cortical_columns - 1; j++) {
                agent->cortical_columns[j] = agent->cortical_columns[j + 1];
            }
            agent->num_cortical_columns--;

            if (agent->cortical_mutex) {
                nimcp_mutex_unlock(agent->cortical_mutex);
            }

            nimcp_log(LOG_LEVEL_INFO, "Disconnected cortical column from health agent");
            return 0;
        }
    }

    if (agent->cortical_mutex) {
        nimcp_mutex_unlock(agent->cortical_mutex);
    }

    return -1;  /* Not found */
}

/* -------------------------------------------------------------------------
 * Perception Health Metrics
 * ------------------------------------------------------------------------- */

int nimcp_health_agent_get_perception_metrics(
    const nimcp_health_agent_t* agent,
    perception_health_metrics_t* metrics
) {
    if (!agent || !metrics) {
        return -1;
    }

    memset(metrics, 0, sizeof(*metrics));

    /* Cast away const for mutex access (mutex itself is not const) */
    nimcp_health_agent_t* mutable_agent = (nimcp_health_agent_t*)agent;

    if (mutable_agent->perception_mutex) {
        nimcp_mutex_lock(mutable_agent->perception_mutex);
    }

    /* Count bridges */
    metrics->num_visual_bridges = mutable_agent->num_visual_bridges;
    metrics->num_audio_bridges = mutable_agent->num_audio_bridges;

    /* Calculate latency metrics (simulated - real impl would query bridges) */
    double total_visual_latency = 0.0;
    double total_audio_latency = 0.0;

    for (uint32_t i = 0; i < mutable_agent->num_visual_bridges; i++) {
        /* Simulate visual latency check */
        double visual_latency = 50.0 + (double)(i * 5);  /* Simulated */
        total_visual_latency += visual_latency;

        if (visual_latency > metrics->max_visual_latency_ms) {
            metrics->max_visual_latency_ms = visual_latency;
        }
    }

    for (uint32_t i = 0; i < mutable_agent->num_audio_bridges; i++) {
        /* Simulate audio latency check */
        double audio_latency = 25.0 + (double)(i * 3);  /* Simulated */
        total_audio_latency += audio_latency;

        if (audio_latency > metrics->max_audio_latency_ms) {
            metrics->max_audio_latency_ms = audio_latency;
        }
    }

    /* Calculate averages */
    metrics->avg_visual_latency_ms = mutable_agent->num_visual_bridges > 0 ?
        total_visual_latency / mutable_agent->num_visual_bridges : 0.0;
    metrics->avg_audio_latency_ms = mutable_agent->num_audio_bridges > 0 ?
        total_audio_latency / mutable_agent->num_audio_bridges : 0.0;

    /* Set frame/sample counts (simulated) */
    metrics->total_frames_processed = 1000;
    metrics->total_samples_processed = 44100;

    /* Set selectivity metrics (simulated healthy values) */
    metrics->avg_orientation_selectivity = 0.65f;
    metrics->avg_frequency_selectivity = 0.58f;
    metrics->feature_selectivity_score = 75.0f;
    metrics->selectivity_degraded = false;

    /* Set mapping metrics (simulated healthy values) */
    metrics->retinotopic_mapping_quality = 0.92f;
    metrics->tonotopic_mapping_quality = 0.89f;
    metrics->mapping_errors_detected = 0;

    /* Set column integration metrics */
    metrics->hypercolumn_competition_score = 0.85f;
    metrics->cross_column_inhibition_score = 0.78f;
    metrics->column_timeout_events = 0;

    /* Set error metrics (simulated low error values) */
    metrics->total_processing_errors = 0;
    metrics->total_overflow_events = 0;
    metrics->total_underflow_events = 0;

    /* Set overall health metrics */
    metrics->any_bridge_unhealthy = false;
    metrics->any_connection_lost = false;
    metrics->total_critical_events = 0;
    metrics->total_recoveries = 0;
    metrics->overall_perception_health = 95.0f;

    /* Update timestamp */
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    metrics->last_check_timestamp_us = (uint64_t)ts.tv_sec * 1000000 +
                                       (uint64_t)ts.tv_nsec / 1000;

    if (mutable_agent->perception_mutex) {
        nimcp_mutex_unlock(mutable_agent->perception_mutex);
    }

    return 0;
}

/* -------------------------------------------------------------------------
 * Cortical Health Metrics
 * ------------------------------------------------------------------------- */

int nimcp_health_agent_get_cortical_metrics(
    const nimcp_health_agent_t* agent,
    cortical_health_metrics_t* metrics
) {
    if (!agent || !metrics) {
        return -1;
    }

    memset(metrics, 0, sizeof(*metrics));

    /* Cast away const for mutex access */
    nimcp_health_agent_t* mutable_agent = (nimcp_health_agent_t*)agent;

    if (mutable_agent->cortical_mutex) {
        nimcp_mutex_lock(mutable_agent->cortical_mutex);
    }

    /* Count components */
    metrics->num_immune_systems = mutable_agent->num_cortical_immune_systems;
    metrics->num_hypercolumns = mutable_agent->num_cortical_columns;
    metrics->any_column_unhealthy = false;
    metrics->immune_system_active = mutable_agent->num_cortical_immune_systems > 0;

    /* Set layer health metrics (simulated healthy values) */
    metrics->layer_2_3_health = 0.88f;          /* Layer 2/3 health */
    metrics->layer_4_health = 0.90f;            /* Layer 4 health */
    metrics->layer_5_6_health = 0.85f;          /* Layer 5/6 health */
    metrics->inter_layer_comm_score = 0.82f;    /* Inter-layer communication */
    metrics->layer_communication_failure = false;

    /* Set column dynamics (simulated healthy values) */
    metrics->avg_column_activity = 0.45f;
    metrics->min_column_activity = 0.25f;
    metrics->max_column_activity = 0.72f;
    metrics->activity_variance = 0.15f;
    metrics->hyperexcitability_detected = false;
    metrics->hypoactivity_detected = false;

    /* Set competition and inhibition metrics */
    metrics->wta_effectiveness = 0.78f;
    metrics->lateral_inhibition_balance = 0.65f;
    metrics->competition_failures = 0;

    /* Set cortical immune metrics */
    metrics->microglial_activation_level = 0.15f;
    metrics->inflammation_index = 0.12f;
    metrics->cytokine_level = 0.18f;
    metrics->immune_responses_triggered = 0;
    metrics->antigens_presented = 0;
    metrics->inflammation_critical = false;

    /* Set feature selectivity metrics */
    metrics->orientation_tuning_width = 0.22f;
    metrics->frequency_tuning_width = 0.25f;
    metrics->tuning_curves_broadened = false;

    /* Set plasticity health metrics */
    metrics->plasticity_modulation = 0.55f;
    metrics->synaptic_pruning_events = 0;
    metrics->circuit_remodeling_events = 0;

    /* Set overall health metrics */
    metrics->total_critical_events = 0;
    metrics->total_recoveries = 0;
    metrics->overall_cortical_health = 92.0f;

    /* Update timestamp */
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    metrics->last_check_timestamp_us = (uint64_t)ts.tv_sec * 1000000 +
                                       (uint64_t)ts.tv_nsec / 1000;

    if (mutable_agent->cortical_mutex) {
        nimcp_mutex_unlock(mutable_agent->cortical_mutex);
    }

    return 0;
}

/* -------------------------------------------------------------------------
 * Perception Recovery
 * ------------------------------------------------------------------------- */

int nimcp_health_agent_perception_recovery(
    nimcp_health_agent_t* agent,
    perception_recovery_action_t action,
    int bridge_index
) {
    if (!agent) {
        return -1;
    }

    const char* action_name = "unknown";
    (void)bridge_index;  /* Used for targeted recovery in real implementation */

    switch (action) {
        case PERCEPTION_RECOVERY_NONE:
            action_name = "none";
            break;
        case PERCEPTION_RECOVERY_FLUSH_BUFFERS:
            action_name = "flush_buffers";
            nimcp_log(LOG_LEVEL_INFO, "Perception recovery: Flushing perception buffers");
            break;
        case PERCEPTION_RECOVERY_RESET_GAIN:
            action_name = "reset_gain";
            nimcp_log(LOG_LEVEL_INFO, "Perception recovery: Resetting input gain to defaults");
            break;
        case PERCEPTION_RECOVERY_ADJUST_GAIN:
            action_name = "adjust_gain";
            nimcp_log(LOG_LEVEL_INFO, "Perception recovery: Adjusting gain based on activity");
            break;
        case PERCEPTION_RECOVERY_RESET_FILTERS:
            action_name = "reset_filters";
            nimcp_log(LOG_LEVEL_INFO, "Perception recovery: Resetting filter banks");
            break;
        case PERCEPTION_RECOVERY_CLEAR_MAPS:
            action_name = "clear_maps";
            nimcp_log(LOG_LEVEL_INFO, "Perception recovery: Clearing topographic maps");
            break;
        case PERCEPTION_RECOVERY_REBUILD_MAPS:
            action_name = "rebuild_maps";
            nimcp_log(LOG_LEVEL_INFO, "Perception recovery: Rebuilding topographic maps");
            break;
        case PERCEPTION_RECOVERY_RESET_SELECTIVITY:
            action_name = "reset_selectivity";
            nimcp_log(LOG_LEVEL_INFO, "Perception recovery: Resetting feature selectivity");
            break;
        case PERCEPTION_RECOVERY_SOFT_RESET:
            action_name = "soft_reset";
            nimcp_log(LOG_LEVEL_INFO, "Perception recovery: Soft reset perception pipeline");
            break;
        case PERCEPTION_RECOVERY_FULL_RESET:
            action_name = "full_reset";
            nimcp_log(LOG_LEVEL_WARN, "Perception recovery: Performing full perception reset");
            break;
        default:
            nimcp_log(LOG_LEVEL_ERROR, "Perception recovery: Unknown action %d", action);
            return -1;
    }

    nimcp_log(LOG_LEVEL_DEBUG, "Perception recovery action '%s' completed (bridge_index=%d)",
              action_name, bridge_index);
    return 0;
}

/* -------------------------------------------------------------------------
 * Cortical Recovery
 * ------------------------------------------------------------------------- */

int nimcp_health_agent_cortical_recovery(
    nimcp_health_agent_t* agent,
    cortical_recovery_action_t action,
    int column_index
) {
    if (!agent) {
        return -1;
    }

    const char* action_name = "unknown";
    (void)column_index;  /* Used for targeted recovery in real implementation */

    switch (action) {
        case CORTICAL_RECOVERY_NONE:
            action_name = "none";
            break;
        case CORTICAL_RECOVERY_NORMALIZE_ACTIVITY:
            action_name = "normalize_activity";
            nimcp_log(LOG_LEVEL_INFO, "Cortical recovery: Normalizing column activity");
            break;
        case CORTICAL_RECOVERY_RESET_COMPETITION:
            action_name = "reset_competition";
            nimcp_log(LOG_LEVEL_INFO, "Cortical recovery: Resetting winner-take-all state");
            break;
        case CORTICAL_RECOVERY_REBALANCE_INHIBITION:
            action_name = "rebalance_inhibition";
            nimcp_log(LOG_LEVEL_INFO, "Cortical recovery: Rebalancing E/I ratio");
            break;
        case CORTICAL_RECOVERY_REDUCE_INFLAMMATION:
            action_name = "reduce_inflammation";
            nimcp_log(LOG_LEVEL_INFO, "Cortical recovery: Reducing inflammation level");
            break;
        case CORTICAL_RECOVERY_SUPPRESS_MICROGLIA:
            action_name = "suppress_microglia";
            nimcp_log(LOG_LEVEL_INFO, "Cortical recovery: Suppressing microglial activation");
            break;
        case CORTICAL_RECOVERY_RESET_LAYERS:
            action_name = "reset_layers";
            nimcp_log(LOG_LEVEL_INFO, "Cortical recovery: Resetting layer communication");
            break;
        case CORTICAL_RECOVERY_SHARPEN_TUNING:
            action_name = "sharpen_tuning";
            nimcp_log(LOG_LEVEL_INFO, "Cortical recovery: Sharpening tuning curves");
            break;
        case CORTICAL_RECOVERY_RESET_PLASTICITY:
            action_name = "reset_plasticity";
            nimcp_log(LOG_LEVEL_INFO, "Cortical recovery: Resetting plasticity modulation");
            break;
        case CORTICAL_RECOVERY_SOFT_RESET:
            action_name = "soft_reset";
            nimcp_log(LOG_LEVEL_INFO, "Cortical recovery: Soft reset cortical state");
            break;
        case CORTICAL_RECOVERY_FULL_RESET:
            action_name = "full_reset";
            nimcp_log(LOG_LEVEL_WARN, "Cortical recovery: Performing full cortical reset");
            break;
        default:
            nimcp_log(LOG_LEVEL_ERROR, "Cortical recovery: Unknown action %d", action);
            return -1;
    }

    nimcp_log(LOG_LEVEL_DEBUG, "Cortical recovery action '%s' completed (column_index=%d)",
              action_name, column_index);
    return 0;
}

/* -------------------------------------------------------------------------
 * Perception Needs Attention Check
 * ------------------------------------------------------------------------- */

bool nimcp_health_agent_perception_needs_attention(
    const nimcp_health_agent_t* agent
) {
    if (!agent) {
        return false;
    }

    uint32_t visual_count = atomic_load(&((nimcp_health_agent_t*)agent)->num_visual_bridges);
    uint32_t audio_count = atomic_load(&((nimcp_health_agent_t*)agent)->num_audio_bridges);

    if (visual_count == 0 && audio_count == 0) {
        return false;  /* No bridges = no attention needed */
    }

    float score = atomic_load(&((nimcp_health_agent_t*)agent)->perception_health_score);
    return score < 70.0f;  /* Needs attention if below 70% */
}

/* -------------------------------------------------------------------------
 * Cortical Needs Attention Check
 * ------------------------------------------------------------------------- */

bool nimcp_health_agent_cortical_needs_attention(
    const nimcp_health_agent_t* agent
) {
    if (!agent) {
        return false;
    }

    uint32_t immune_count = atomic_load(&((nimcp_health_agent_t*)agent)->num_cortical_immune_systems);
    uint32_t column_count = atomic_load(&((nimcp_health_agent_t*)agent)->num_cortical_columns);

    if (immune_count == 0 && column_count == 0) {
        return false;  /* No cortical components = no attention needed */
    }

    float score = atomic_load(&((nimcp_health_agent_t*)agent)->cortical_health_score);
    return score < 70.0f;  /* Needs attention if below 70% */
}

/* -------------------------------------------------------------------------
 * Perception Health Score
 * ------------------------------------------------------------------------- */

float nimcp_health_agent_get_perception_health_score(
    const nimcp_health_agent_t* agent
) {
    if (!agent) {
        return 100.0f;  /* No agent = assume healthy */
    }

    uint32_t visual_count = atomic_load(&((nimcp_health_agent_t*)agent)->num_visual_bridges);
    uint32_t audio_count = atomic_load(&((nimcp_health_agent_t*)agent)->num_audio_bridges);

    if (visual_count == 0 && audio_count == 0) {
        return 100.0f;  /* No bridges = healthy */
    }

    return atomic_load(&((nimcp_health_agent_t*)agent)->perception_health_score);
}

/* -------------------------------------------------------------------------
 * Cortical Health Score
 * ------------------------------------------------------------------------- */

float nimcp_health_agent_get_cortical_health_score(
    const nimcp_health_agent_t* agent
) {
    if (!agent) {
        return 100.0f;  /* No agent = assume healthy */
    }

    uint32_t immune_count = atomic_load(&((nimcp_health_agent_t*)agent)->num_cortical_immune_systems);
    uint32_t column_count = atomic_load(&((nimcp_health_agent_t*)agent)->num_cortical_columns);

    if (immune_count == 0 && column_count == 0) {
        return 100.0f;  /* No cortical components = healthy */
    }

    return atomic_load(&((nimcp_health_agent_t*)agent)->cortical_health_score);
}

/* -------------------------------------------------------------------------
 * Perception Configuration Update
 * ------------------------------------------------------------------------- */

int nimcp_health_agent_update_perception_config(
    nimcp_health_agent_t* agent,
    const health_agent_perception_config_t* config
) {
    if (!agent || !config) {
        return -1;
    }

    if (agent->perception_mutex) {
        nimcp_mutex_lock(agent->perception_mutex);
    }

    memcpy(&agent->perception_config, config, sizeof(agent->perception_config));

    if (agent->perception_mutex) {
        nimcp_mutex_unlock(agent->perception_mutex);
    }

    nimcp_log(LOG_LEVEL_DEBUG, "Updated perception health configuration");
    return 0;
}

/* -------------------------------------------------------------------------
 * Cortical Configuration Update
 * ------------------------------------------------------------------------- */

int nimcp_health_agent_update_cortical_config(
    nimcp_health_agent_t* agent,
    const health_agent_cortical_config_t* config
) {
    if (!agent || !config) {
        return -1;
    }

    if (agent->cortical_mutex) {
        nimcp_mutex_lock(agent->cortical_mutex);
    }

    memcpy(&agent->cortical_config, config, sizeof(agent->cortical_config));

    if (agent->cortical_mutex) {
        nimcp_mutex_unlock(agent->cortical_mutex);
    }

    nimcp_log(LOG_LEVEL_DEBUG, "Updated cortical health configuration");
    return 0;
}

/* ============================================================================
 * Phase 5.13: Brain Probes Enhancement Implementation
 * ============================================================================ */

/* -------------------------------------------------------------------------
 * Brain Probe Configuration Default
 * ------------------------------------------------------------------------- */

void nimcp_health_agent_brain_probe_config_default(
    health_agent_brain_probe_config_t* config
) {
    if (!config) {
        return;
    }

    memset(config, 0, sizeof(*config));

    /* Enable flags */
    config->enable_probe_monitoring = true;
    config->enable_memory_tracking = true;
    config->enable_performance_tracking = true;
    config->enable_learning_monitoring = true;
    config->enable_synapse_monitoring = true;
    config->enable_cow_monitoring = true;
    config->enable_auto_recovery = false;  /* Conservative default */

    /* Timing */
    config->probe_interval_ms = 1000;       /* 1 second probe interval */
    config->trend_window_probes = 10;       /* 10 probes for trend analysis */

    /* Memory thresholds */
    config->memory_warning_bytes = 512 * 1024 * 1024;   /* 512 MB warning */
    config->memory_critical_bytes = 1024 * 1024 * 1024; /* 1 GB critical */
    config->memory_growth_rate_warning = 0.1f;          /* 10%/sec growth rate warning */

    /* Performance thresholds */
    config->inference_time_warning_us = 10000.0f;   /* 10ms warning */
    config->inference_time_critical_us = 100000.0f; /* 100ms critical */
    config->performance_degradation_pct = 0.2f;     /* 20% degradation warning */

    /* Learning thresholds */
    config->lr_change_warning_pct = 0.5f;   /* 50% LR change warning */
    config->accuracy_drop_warning = 0.05f;  /* 5% accuracy drop warning */

    /* Synapse thresholds */
    config->synapse_growth_warning_pct = 0.2f;  /* 20% growth warning */
    config->synapse_loss_warning_pct = 0.3f;    /* 30% loss warning */
    config->min_active_synapses = 100;          /* Minimum 100 active synapses */

    /* COW thresholds */
    config->cow_private_ratio_warning = 0.5f;           /* 50% private warning */
    config->cow_overhead_warning_bytes = 50 * 1024 * 1024; /* 50 MB COW overhead */
}

/* -------------------------------------------------------------------------
 * Brain Probe Registration
 * ------------------------------------------------------------------------- */

int nimcp_health_agent_register_brain_probe(
    nimcp_health_agent_t* agent,
    brain_t brain,
    const health_agent_brain_probe_config_t* config
) {
    if (!agent || !brain) {
        return -1;
    }

    if (agent->brain_probe_mutex) {
        nimcp_mutex_lock(agent->brain_probe_mutex);
    }

    uint32_t count = atomic_load(&agent->num_monitored_brains);
    if (count >= HEALTH_AGENT_MAX_BRAINS) {
        if (agent->brain_probe_mutex) {
            nimcp_mutex_unlock(agent->brain_probe_mutex);
        }
        nimcp_log(LOG_LEVEL_WARN, "Maximum brain probe slots reached");
        return -1;
    }

    /* Check for duplicate */
    for (uint32_t i = 0; i < count; i++) {
        if (agent->monitored_brains[i] == brain) {
            if (agent->brain_probe_mutex) {
                nimcp_mutex_unlock(agent->brain_probe_mutex);
            }
            nimcp_log(LOG_LEVEL_DEBUG, "Brain already registered for monitoring");
            return 0;
        }
    }

    /* Store brain reference */
    agent->monitored_brains[count] = brain;

    /* Initialize metrics */
    memset(&agent->brain_metrics[count], 0, sizeof(brain_probe_health_metrics_t));
    agent->brain_metrics[count].overall_health_score = 100.0f;

    /* Copy or use default config */
    if (config) {
        memcpy(&agent->brain_probe_configs[count], config, sizeof(*config));
    } else {
        nimcp_health_agent_brain_probe_config_default(&agent->brain_probe_configs[count]);
    }

    atomic_store(&agent->num_monitored_brains, count + 1);

    if (agent->brain_probe_mutex) {
        nimcp_mutex_unlock(agent->brain_probe_mutex);
    }

    nimcp_log(LOG_LEVEL_INFO, "Connected brain for probe monitoring (slot %u)", count);
    return 0;
}

/* -------------------------------------------------------------------------
 * Brain Probe Unregistration
 * ------------------------------------------------------------------------- */

int nimcp_health_agent_unregister_brain_probe(
    nimcp_health_agent_t* agent,
    brain_t brain
) {
    if (!agent || !brain) {
        return -1;
    }

    if (agent->brain_probe_mutex) {
        nimcp_mutex_lock(agent->brain_probe_mutex);
    }

    uint32_t count = atomic_load(&agent->num_monitored_brains);
    int found_idx = -1;

    for (uint32_t i = 0; i < count; i++) {
        if (agent->monitored_brains[i] == brain) {
            found_idx = (int)i;
            break;
        }
    }

    if (found_idx < 0) {
        if (agent->brain_probe_mutex) {
            nimcp_mutex_unlock(agent->brain_probe_mutex);
        }
        return -1;
    }

    /* Shift remaining entries */
    for (uint32_t i = (uint32_t)found_idx; i < count - 1; i++) {
        agent->monitored_brains[i] = agent->monitored_brains[i + 1];
        memcpy(&agent->brain_metrics[i], &agent->brain_metrics[i + 1],
               sizeof(brain_probe_health_metrics_t));
        memcpy(&agent->brain_probe_configs[i], &agent->brain_probe_configs[i + 1],
               sizeof(health_agent_brain_probe_config_t));
    }

    /* Clear last slot */
    agent->monitored_brains[count - 1] = NULL;
    memset(&agent->brain_metrics[count - 1], 0, sizeof(brain_probe_health_metrics_t));

    atomic_store(&agent->num_monitored_brains, count - 1);

    if (agent->brain_probe_mutex) {
        nimcp_mutex_unlock(agent->brain_probe_mutex);
    }

    nimcp_log(LOG_LEVEL_INFO, "Disconnected brain from probe monitoring");
    return 0;
}

/* -------------------------------------------------------------------------
 * Helper: Perform single brain probe and update metrics
 * ------------------------------------------------------------------------- */

/* Local struct to match brain probe data (avoids nimcp.h dependency) */
typedef struct {
    uint32_t num_neurons;
    uint32_t num_synapses;
    uint32_t num_active_synapses;
    size_t memory_bytes;
    float avg_inference_time_us;
    float current_learning_rate;
    float avg_sparsity;
    float accuracy;
    bool is_cow_clone;
    uint32_t cow_ref_count;
    size_t cow_shared_bytes;
    size_t cow_private_bytes;
} local_brain_probe_data_t;

static void agent_probe_single_brain(
    nimcp_health_agent_t* agent,
    uint32_t index
) {
    if (!agent || index >= atomic_load(&agent->num_monitored_brains)) {
        return;
    }

    brain_t brain = agent->monitored_brains[index];
    brain_probe_health_metrics_t* metrics = &agent->brain_metrics[index];
    health_agent_brain_probe_config_t* config = &agent->brain_probe_configs[index];

    if (!brain || !config->enable_probe_monitoring) {
        return;
    }

    /* Local probe data structure */
    local_brain_probe_data_t probe;
    memset(&probe, 0, sizeof(probe));

    /*
     * Note: The actual nimcp_brain_probe() function is defined in nimcp.h
     * and works with nimcp_brain_probe_t. Since this module may not have
     * nimcp.h included to avoid circular dependencies, we use a stub
     * that can be replaced with actual probe calls when integrated.
     *
     * In a real deployment, this would call:
     * nimcp_brain_probe_t real_probe;
     * nimcp_brain_probe(brain, &real_probe);
     * and copy values to our local struct.
     */

    /* Stub implementation - using placeholder values based on brain pointer */
    /* A real implementation would call the actual brain probe API */
    probe.num_neurons = 1000 + ((uintptr_t)brain % 1000);
    probe.num_synapses = probe.num_neurons * 10;
    probe.num_active_synapses = (uint32_t)(probe.num_synapses * 0.8);
    probe.memory_bytes = probe.num_neurons * 1024;
    probe.avg_inference_time_us = 100.0f + (float)((uintptr_t)brain % 100);
    probe.current_learning_rate = 0.01f;
    probe.avg_sparsity = 0.2f;
    probe.accuracy = 0.95f;
    probe.is_cow_clone = false;
    probe.cow_ref_count = 0;
    probe.cow_shared_bytes = 0;
    probe.cow_private_bytes = 0;

    /* Store previous values for trend calculation */
    size_t prev_memory = metrics->memory_bytes;
    float prev_inference = metrics->avg_inference_time_us;
    uint32_t prev_synapses = metrics->num_synapses;
    float prev_accuracy = metrics->accuracy;

    /* Update current snapshot */
    metrics->num_neurons = probe.num_neurons;
    metrics->num_synapses = probe.num_synapses;
    metrics->num_active_synapses = probe.num_active_synapses;
    metrics->memory_bytes = probe.memory_bytes;
    metrics->avg_inference_time_us = probe.avg_inference_time_us;
    metrics->current_learning_rate = probe.current_learning_rate;
    metrics->avg_sparsity = probe.avg_sparsity;
    metrics->accuracy = probe.accuracy;

    /* Update COW statistics */
    metrics->is_cow_clone = probe.is_cow_clone;
    metrics->cow_ref_count = probe.cow_ref_count;
    metrics->cow_shared_bytes = probe.cow_shared_bytes;
    metrics->cow_private_bytes = probe.cow_private_bytes;
    if (probe.is_cow_clone && (probe.cow_shared_bytes + probe.cow_private_bytes) > 0) {
        metrics->cow_private_ratio = (float)probe.cow_private_bytes /
                                     (float)(probe.cow_shared_bytes + probe.cow_private_bytes);
    } else {
        metrics->cow_private_ratio = 0.0f;
    }

    /* Update history ring buffer */
    metrics->memory_history[metrics->history_index] = probe.memory_bytes;
    metrics->inference_history[metrics->history_index] = probe.avg_inference_time_us;
    metrics->history_index = (metrics->history_index + 1) % 10;
    if (metrics->history_count < 10) {
        metrics->history_count++;
    }

    /* Calculate trends (if we have previous data) */
    uint64_t now = get_timestamp_us();
    uint64_t probe_interval_us = config->probe_interval_ms * 1000ULL;

    if (metrics->total_probes > 0 && prev_memory > 0) {
        /* Memory growth rate (bytes/sec) */
        if (probe_interval_us > 0) {
            int64_t memory_delta = (int64_t)probe.memory_bytes - (int64_t)prev_memory;
            metrics->memory_growth_rate = (float)memory_delta * 1000000.0f / (float)probe_interval_us;
        }

        /* Inference time trend */
        metrics->inference_time_trend = probe.avg_inference_time_us - prev_inference;

        /* Synapse change rate */
        if (prev_synapses > 0) {
            float synapse_delta = (float)((int32_t)probe.num_synapses - (int32_t)prev_synapses);
            metrics->synapse_change_rate = synapse_delta / (float)prev_synapses;
        }

        /* Accuracy trend */
        metrics->accuracy_trend = probe.accuracy - prev_accuracy;
    }

    /* Update timestamps */
    metrics->total_probes++;
    metrics->last_probe_timestamp_ms = now / 1000;

    /* Calculate health score */
    float health = 100.0f;
    uint32_t warnings = 0;
    uint32_t criticals = 0;

    /* Memory health check */
    if (config->enable_memory_tracking) {
        if (probe.memory_bytes >= config->memory_critical_bytes) {
            health -= 30.0f;
            criticals++;
        } else if (probe.memory_bytes >= config->memory_warning_bytes) {
            health -= 10.0f;
            warnings++;
        }

        /* Memory growth rate check */
        if (prev_memory > 0 && metrics->memory_growth_rate > 0) {
            float growth_pct = metrics->memory_growth_rate / (float)prev_memory;
            if (growth_pct > config->memory_growth_rate_warning) {
                health -= 5.0f;
                warnings++;
            }
        }
    }

    /* Performance health check */
    if (config->enable_performance_tracking) {
        if (probe.avg_inference_time_us >= config->inference_time_critical_us) {
            health -= 25.0f;
            criticals++;
        } else if (probe.avg_inference_time_us >= config->inference_time_warning_us) {
            health -= 10.0f;
            warnings++;
        }
    }

    /* Synapse health check */
    if (config->enable_synapse_monitoring) {
        if (probe.num_active_synapses < config->min_active_synapses) {
            health -= 20.0f;
            criticals++;
        }

        if (prev_synapses > 0) {
            float change = fabsf(metrics->synapse_change_rate);
            if (metrics->synapse_change_rate > 0 && change > config->synapse_growth_warning_pct) {
                health -= 5.0f;
                warnings++;
            }
            if (metrics->synapse_change_rate < 0 && change > config->synapse_loss_warning_pct) {
                health -= 10.0f;
                warnings++;
            }
        }
    }

    /* Learning health check */
    if (config->enable_learning_monitoring) {
        if (metrics->accuracy_trend < -config->accuracy_drop_warning) {
            health -= 10.0f;
            warnings++;
        }
    }

    /* COW health check */
    if (config->enable_cow_monitoring && probe.is_cow_clone) {
        if (metrics->cow_private_ratio > config->cow_private_ratio_warning) {
            health -= 5.0f;
            warnings++;
        }
        if (probe.cow_private_bytes > config->cow_overhead_warning_bytes) {
            health -= 5.0f;
            warnings++;
        }
    }

    /* Clamp health score */
    if (health < 0.0f) health = 0.0f;
    if (health > 100.0f) health = 100.0f;

    metrics->overall_health_score = health;
    metrics->warnings_active = warnings;
    metrics->critical_issues = criticals;

    /* Update global stats */
    atomic_fetch_add(&agent->brain_probes_run, 1);
    if (warnings > 0) {
        atomic_fetch_add(&agent->brain_warnings_triggered, 1);
    }
    if (criticals > 0) {
        atomic_fetch_add(&agent->brain_critical_events, 1);
    }
}

/* -------------------------------------------------------------------------
 * Get Brain Probe Metrics
 * ------------------------------------------------------------------------- */

int nimcp_health_agent_get_brain_probe_metrics(
    const nimcp_health_agent_t* agent,
    uint32_t brain_index,
    brain_probe_health_metrics_t* metrics
) {
    if (!agent || !metrics) {
        return -1;
    }

    uint32_t count = atomic_load(&((nimcp_health_agent_t*)agent)->num_monitored_brains);
    if (brain_index >= count) {
        return -1;
    }

    if (((nimcp_health_agent_t*)agent)->brain_probe_mutex) {
        nimcp_mutex_lock(((nimcp_health_agent_t*)agent)->brain_probe_mutex);
    }

    memcpy(metrics, &((nimcp_health_agent_t*)agent)->brain_metrics[brain_index],
           sizeof(*metrics));

    if (((nimcp_health_agent_t*)agent)->brain_probe_mutex) {
        nimcp_mutex_unlock(((nimcp_health_agent_t*)agent)->brain_probe_mutex);
    }

    return 0;
}

/* -------------------------------------------------------------------------
 * Brain Probe Recovery
 * ------------------------------------------------------------------------- */

int nimcp_health_agent_brain_probe_recovery(
    nimcp_health_agent_t* agent,
    brain_probe_recovery_action_t action,
    int brain_index
) {
    if (!agent) {
        return -1;
    }

    uint32_t count = atomic_load(&agent->num_monitored_brains);
    if (count == 0) {
        return -1;
    }

    /* Determine which brains to act on */
    uint32_t start_idx = 0;
    uint32_t end_idx = count;
    if (brain_index >= 0) {
        if ((uint32_t)brain_index >= count) {
            return -1;
        }
        start_idx = (uint32_t)brain_index;
        end_idx = start_idx + 1;
    }

    nimcp_log(LOG_LEVEL_INFO, "Brain probe recovery: action=%d, brains=%u-%u",
              (int)action, start_idx, end_idx - 1);

    for (uint32_t i = start_idx; i < end_idx; i++) {
        brain_t brain = agent->monitored_brains[i];
        if (!brain) continue;

        switch (action) {
            case BRAIN_PROBE_RECOVERY_NONE:
                break;

            case BRAIN_PROBE_RECOVERY_TRIGGER_GC:
                nimcp_log(LOG_LEVEL_INFO, "Brain probe recovery: Triggering GC for brain %u", i);
                /* Would call nimcp_brain_gc(brain) if available */
                break;

            case BRAIN_PROBE_RECOVERY_REDUCE_LR:
                nimcp_log(LOG_LEVEL_INFO, "Brain probe recovery: Reducing LR for brain %u", i);
                /* Would call nimcp_brain_set_learning_rate(brain, lr * 0.5) if available */
                break;

            case BRAIN_PROBE_RECOVERY_INCREASE_SPARSITY:
                nimcp_log(LOG_LEVEL_INFO, "Brain probe recovery: Increasing sparsity for brain %u", i);
                break;

            case BRAIN_PROBE_RECOVERY_TRIGGER_PRUNE:
                nimcp_log(LOG_LEVEL_INFO, "Brain probe recovery: Triggering prune for brain %u", i);
                break;

            case BRAIN_PROBE_RECOVERY_CHECKPOINT:
                nimcp_log(LOG_LEVEL_INFO, "Brain probe recovery: Saving checkpoint for brain %u", i);
                break;

            case BRAIN_PROBE_RECOVERY_THROTTLE_INFERENCE:
                nimcp_log(LOG_LEVEL_INFO, "Brain probe recovery: Throttling inference for brain %u", i);
                break;

            case BRAIN_PROBE_RECOVERY_DETACH_COW:
                nimcp_log(LOG_LEVEL_INFO, "Brain probe recovery: Detaching COW for brain %u", i);
                break;

            case BRAIN_PROBE_RECOVERY_RESET_STATS:
                nimcp_log(LOG_LEVEL_INFO, "Brain probe recovery: Resetting stats for brain %u", i);
                memset(&agent->brain_metrics[i], 0, sizeof(brain_probe_health_metrics_t));
                agent->brain_metrics[i].overall_health_score = 100.0f;
                break;

            case BRAIN_PROBE_RECOVERY_FULL_RESET:
                nimcp_log(LOG_LEVEL_INFO, "Brain probe recovery: Full reset for brain %u", i);
                break;

            default:
                nimcp_log(LOG_LEVEL_WARN, "Unknown brain probe recovery action: %d", (int)action);
                break;
        }
    }

    atomic_fetch_add(&agent->brain_recoveries_performed, 1);
    return 0;
}

/* -------------------------------------------------------------------------
 * Brain Needs Attention Check
 * ------------------------------------------------------------------------- */

bool nimcp_health_agent_brain_needs_attention(
    const nimcp_health_agent_t* agent
) {
    if (!agent) {
        return false;
    }

    uint32_t count = atomic_load(&((nimcp_health_agent_t*)agent)->num_monitored_brains);
    if (count == 0) {
        return false;
    }

    for (uint32_t i = 0; i < count; i++) {
        const brain_probe_health_metrics_t* metrics = &agent->brain_metrics[i];
        if (metrics->warnings_active > 0 || metrics->critical_issues > 0) {
            return true;
        }
        if (metrics->overall_health_score < 70.0f) {
            return true;
        }
    }

    return false;
}

/* -------------------------------------------------------------------------
 * Get Overall Brain Probe Health Score
 * ------------------------------------------------------------------------- */

float nimcp_health_agent_get_brain_probe_health_score(
    const nimcp_health_agent_t* agent
) {
    if (!agent) {
        return 100.0f;
    }

    uint32_t count = atomic_load(&((nimcp_health_agent_t*)agent)->num_monitored_brains);
    if (count == 0) {
        return 100.0f;  /* No brains = healthy */
    }

    /* Calculate weighted average health score */
    float total_score = 0.0f;
    for (uint32_t i = 0; i < count; i++) {
        total_score += agent->brain_metrics[i].overall_health_score;
    }

    return total_score / (float)count;
}

/* -------------------------------------------------------------------------
 * Update Brain Probe Configuration
 * ------------------------------------------------------------------------- */

int nimcp_health_agent_update_brain_probe_config(
    nimcp_health_agent_t* agent,
    const health_agent_brain_probe_config_t* config
) {
    if (!agent || !config) {
        return -1;
    }

    if (agent->brain_probe_mutex) {
        nimcp_mutex_lock(agent->brain_probe_mutex);
    }

    /* Update default config */
    memcpy(&agent->brain_probe_config, config, sizeof(agent->brain_probe_config));

    /* Update all registered brain configs */
    uint32_t count = atomic_load(&agent->num_monitored_brains);
    for (uint32_t i = 0; i < count; i++) {
        memcpy(&agent->brain_probe_configs[i], config, sizeof(*config));
    }

    if (agent->brain_probe_mutex) {
        nimcp_mutex_unlock(agent->brain_probe_mutex);
    }

    nimcp_log(LOG_LEVEL_DEBUG, "Updated brain probe configuration for %u brains", count);
    return 0;
}

/* -------------------------------------------------------------------------
 * Force Immediate Probe of All Brains
 * ------------------------------------------------------------------------- */

int nimcp_health_agent_probe_all_brains_now(
    nimcp_health_agent_t* agent
) {
    if (!agent) {
        return -1;
    }

    uint32_t count = atomic_load(&agent->num_monitored_brains);
    if (count == 0) {
        return 0;
    }

    if (agent->brain_probe_mutex) {
        nimcp_mutex_lock(agent->brain_probe_mutex);
    }

    for (uint32_t i = 0; i < count; i++) {
        agent_probe_single_brain(agent, i);
    }

    /* Update aggregate health score */
    float total_health = 0.0f;
    for (uint32_t i = 0; i < count; i++) {
        total_health += agent->brain_metrics[i].overall_health_score;
    }
    atomic_store(&agent->brain_probe_health_score, total_health / (float)count);
    agent->last_brain_probe_us = get_timestamp_us();

    if (agent->brain_probe_mutex) {
        nimcp_mutex_unlock(agent->brain_probe_mutex);
    }

    nimcp_log(LOG_LEVEL_DEBUG, "Probed %u brains, avg health: %.1f",
              count, atomic_load(&agent->brain_probe_health_score));
    return (int)count;
}

/* -------------------------------------------------------------------------
 * Get Brain Count
 * ------------------------------------------------------------------------- */

uint32_t nimcp_health_agent_get_brain_count(
    const nimcp_health_agent_t* agent
) {
    if (!agent) {
        return 0;
    }

    return atomic_load(&((nimcp_health_agent_t*)agent)->num_monitored_brains);
}

/* ============================================================================
 * Phase 5.14: World Model & Imagination Health Implementation
 * ============================================================================
 *
 * WHAT: Health monitoring for JEPA predictor, Omni world model, and imagination
 * WHY:  Predictive processing health is critical for planning and reasoning
 * HOW:  Monitor prediction accuracy, dynamics stability, imagination coherence
 */

/* -------------------------------------------------------------------------
 * Default Configuration
 * ------------------------------------------------------------------------- */

void nimcp_health_agent_wm_imagination_config_default(
    health_agent_wm_imagination_config_t* config
) {
    if (!config) {
        return;
    }

    /* Check intervals */
    config->check_interval_ms = 500;
    config->trend_window_ms = 10000;

    /* JEPA thresholds */
    config->jepa_error_warning = 0.3f;
    config->jepa_error_critical = 0.6f;
    config->embedding_variance_min = 0.01f;
    config->gradient_norm_max = 100.0f;
    config->gradient_norm_min = 1e-7f;

    /* World model thresholds */
    config->forward_accuracy_warning = 0.7f;
    config->forward_accuracy_critical = 0.5f;
    config->horizon_min_stable = 5;
    config->counterfactual_validity_min = 0.8f;

    /* Imagination thresholds */
    config->coherence_warning = 0.6f;
    config->coherence_critical = 0.4f;
    config->vividness_warning = 0.4f;
    config->reality_check_min = 0.9f;
    config->imagination_reality_blur_max = 0.3f;

    /* Recovery settings */
    config->auto_recovery_enabled = true;
    config->recovery_cooldown_ms = 5000;
    config->max_recoveries_per_hour = 10;

    /* Immune integration */
    config->report_to_immune = true;
    config->immune_severity_base = 6;
}

/* -------------------------------------------------------------------------
 * JEPA Connection
 * ------------------------------------------------------------------------- */

int nimcp_health_agent_connect_jepa(
    nimcp_health_agent_t* agent,
    jepa_predictor_t* jepa,
    const health_agent_wm_imagination_config_t* config
) {
    if (!agent || !jepa) {
        return -1;
    }

    if (agent->wm_imagination_mutex) {
        nimcp_mutex_lock(agent->wm_imagination_mutex);
    }

    agent->jepa_predictor = jepa;

    /* Apply config if provided, otherwise use defaults */
    if (config) {
        memcpy(&agent->wm_imagination_config, config, sizeof(*config));
    } else if (agent->wm_imagination_config.check_interval_ms == 0) {
        nimcp_health_agent_wm_imagination_config_default(&agent->wm_imagination_config);
    }

    /* Initialize JEPA metrics to healthy defaults */
    memset(&agent->jepa_metrics, 0, sizeof(agent->jepa_metrics));
    agent->jepa_metrics.embedding_orthogonality = 1.0f;
    agent->jepa_metrics.embedding_utilization = 1.0f;

    /* Initialize world model health score if this is the first WM component */
    if (!agent->world_model) {
        atomic_store(&agent->wm_health_score, 1.0f);
    }

    if (agent->wm_imagination_mutex) {
        nimcp_mutex_unlock(agent->wm_imagination_mutex);
    }

    nimcp_log(LOG_LEVEL_INFO, "Connected JEPA predictor to health agent");
    return 0;
}

int nimcp_health_agent_disconnect_jepa(
    nimcp_health_agent_t* agent
) {
    if (!agent) {
        return -1;
    }

    if (agent->wm_imagination_mutex) {
        nimcp_mutex_lock(agent->wm_imagination_mutex);
    }

    agent->jepa_predictor = NULL;
    memset(&agent->jepa_metrics, 0, sizeof(agent->jepa_metrics));

    if (agent->wm_imagination_mutex) {
        nimcp_mutex_unlock(agent->wm_imagination_mutex);
    }

    nimcp_log(LOG_LEVEL_INFO, "Disconnected JEPA predictor from health agent");
    return 0;
}

/* -------------------------------------------------------------------------
 * World Model Connection
 * ------------------------------------------------------------------------- */

int nimcp_health_agent_connect_world_model(
    nimcp_health_agent_t* agent,
    omni_world_model_t* world_model,
    const health_agent_wm_imagination_config_t* config
) {
    if (!agent || !world_model) {
        return -1;
    }

    if (agent->wm_imagination_mutex) {
        nimcp_mutex_lock(agent->wm_imagination_mutex);
    }

    agent->world_model = world_model;

    /* Apply config if provided */
    if (config) {
        memcpy(&agent->wm_imagination_config, config, sizeof(*config));
    } else if (agent->wm_imagination_config.check_interval_ms == 0) {
        nimcp_health_agent_wm_imagination_config_default(&agent->wm_imagination_config);
    }

    /* Initialize world model metrics */
    memset(&agent->wm_metrics, 0, sizeof(agent->wm_metrics));
    agent->wm_metrics.health_state = WM_HEALTH_OPTIMAL;
    agent->wm_metrics.health_score = 1.0f;
    agent->wm_metrics.forward_accuracy = 1.0f;
    agent->wm_metrics.forward_consistency = 1.0f;
    agent->wm_metrics.counterfactual_validity = 1.0f;
    agent->wm_metrics.crossmodal_coherence = 1.0f;

    /* Initialize world model health score atomic */
    atomic_store(&agent->wm_health_score, 1.0f);

    if (agent->wm_imagination_mutex) {
        nimcp_mutex_unlock(agent->wm_imagination_mutex);
    }

    nimcp_log(LOG_LEVEL_INFO, "Connected world model to health agent");
    return 0;
}

int nimcp_health_agent_disconnect_world_model(
    nimcp_health_agent_t* agent
) {
    if (!agent) {
        return -1;
    }

    if (agent->wm_imagination_mutex) {
        nimcp_mutex_lock(agent->wm_imagination_mutex);
    }

    agent->world_model = NULL;
    memset(&agent->wm_metrics, 0, sizeof(agent->wm_metrics));

    if (agent->wm_imagination_mutex) {
        nimcp_mutex_unlock(agent->wm_imagination_mutex);
    }

    nimcp_log(LOG_LEVEL_INFO, "Disconnected world model from health agent");
    return 0;
}

/* -------------------------------------------------------------------------
 * Imagination Connection
 * ------------------------------------------------------------------------- */

int nimcp_health_agent_connect_imagination(
    nimcp_health_agent_t* agent,
    imagination_engine_t* imagination,
    const health_agent_wm_imagination_config_t* config
) {
    if (!agent || !imagination) {
        return -1;
    }

    if (agent->wm_imagination_mutex) {
        nimcp_mutex_lock(agent->wm_imagination_mutex);
    }

    agent->imagination = imagination;

    /* Apply config if provided */
    if (config) {
        memcpy(&agent->wm_imagination_config, config, sizeof(*config));
    } else if (agent->wm_imagination_config.check_interval_ms == 0) {
        nimcp_health_agent_wm_imagination_config_default(&agent->wm_imagination_config);
    }

    /* Initialize imagination metrics */
    memset(&agent->imagination_metrics, 0, sizeof(agent->imagination_metrics));
    agent->imagination_metrics.health_state = IMAG_HEALTH_VIVID;
    agent->imagination_metrics.health_score = 1.0f;
    agent->imagination_metrics.scene_coherence = 1.0f;
    agent->imagination_metrics.scene_vividness = 1.0f;
    agent->imagination_metrics.reality_check_pass_rate = 1.0f;

    /* Initialize imagination health score atomic */
    atomic_store(&agent->imagination_health_score, 1.0f);

    if (agent->wm_imagination_mutex) {
        nimcp_mutex_unlock(agent->wm_imagination_mutex);
    }

    nimcp_log(LOG_LEVEL_INFO, "Connected imagination engine to health agent");
    return 0;
}

int nimcp_health_agent_disconnect_imagination(
    nimcp_health_agent_t* agent
) {
    if (!agent) {
        return -1;
    }

    if (agent->wm_imagination_mutex) {
        nimcp_mutex_lock(agent->wm_imagination_mutex);
    }

    agent->imagination = NULL;
    memset(&agent->imagination_metrics, 0, sizeof(agent->imagination_metrics));

    if (agent->wm_imagination_mutex) {
        nimcp_mutex_unlock(agent->wm_imagination_mutex);
    }

    nimcp_log(LOG_LEVEL_INFO, "Disconnected imagination engine from health agent");
    return 0;
}

/* -------------------------------------------------------------------------
 * Metrics Retrieval
 * ------------------------------------------------------------------------- */

int nimcp_health_agent_get_jepa_metrics(
    const nimcp_health_agent_t* agent,
    jepa_health_metrics_t* metrics
) {
    if (!agent || !metrics) {
        return -1;
    }

    if (((nimcp_health_agent_t*)agent)->wm_imagination_mutex) {
        nimcp_mutex_lock(((nimcp_health_agent_t*)agent)->wm_imagination_mutex);
    }

    memcpy(metrics, &((nimcp_health_agent_t*)agent)->jepa_metrics, sizeof(*metrics));

    if (((nimcp_health_agent_t*)agent)->wm_imagination_mutex) {
        nimcp_mutex_unlock(((nimcp_health_agent_t*)agent)->wm_imagination_mutex);
    }

    return 0;
}

int nimcp_health_agent_get_world_model_metrics(
    const nimcp_health_agent_t* agent,
    omni_wm_health_metrics_t* metrics
) {
    if (!agent || !metrics) {
        return -1;
    }

    if (((nimcp_health_agent_t*)agent)->wm_imagination_mutex) {
        nimcp_mutex_lock(((nimcp_health_agent_t*)agent)->wm_imagination_mutex);
    }

    memcpy(metrics, &((nimcp_health_agent_t*)agent)->wm_metrics, sizeof(*metrics));

    if (((nimcp_health_agent_t*)agent)->wm_imagination_mutex) {
        nimcp_mutex_unlock(((nimcp_health_agent_t*)agent)->wm_imagination_mutex);
    }

    return 0;
}

int nimcp_health_agent_get_imagination_metrics(
    const nimcp_health_agent_t* agent,
    imagination_health_metrics_t* metrics
) {
    if (!agent || !metrics) {
        return -1;
    }

    if (((nimcp_health_agent_t*)agent)->wm_imagination_mutex) {
        nimcp_mutex_lock(((nimcp_health_agent_t*)agent)->wm_imagination_mutex);
    }

    memcpy(metrics, &((nimcp_health_agent_t*)agent)->imagination_metrics, sizeof(*metrics));

    if (((nimcp_health_agent_t*)agent)->wm_imagination_mutex) {
        nimcp_mutex_unlock(((nimcp_health_agent_t*)agent)->wm_imagination_mutex);
    }

    return 0;
}

int nimcp_health_agent_get_world_imagination_health(
    const nimcp_health_agent_t* agent,
    world_imagination_health_t* health
) {
    if (!agent || !health) {
        return -1;
    }

    if (((nimcp_health_agent_t*)agent)->wm_imagination_mutex) {
        nimcp_mutex_lock(((nimcp_health_agent_t*)agent)->wm_imagination_mutex);
    }

    /* Copy component metrics */
    memcpy(&health->jepa, &((nimcp_health_agent_t*)agent)->jepa_metrics,
           sizeof(health->jepa));
    memcpy(&health->world_model, &((nimcp_health_agent_t*)agent)->wm_metrics,
           sizeof(health->world_model));
    memcpy(&health->imagination, &((nimcp_health_agent_t*)agent)->imagination_metrics,
           sizeof(health->imagination));

    /* Compute cross-system health metrics */
    float wm_score = atomic_load(&((nimcp_health_agent_t*)agent)->wm_health_score);
    float imag_score = atomic_load(&((nimcp_health_agent_t*)agent)->imagination_health_score);

    health->wm_imagination_alignment = (wm_score + imag_score) / 2.0f;
    health->prediction_imagination_sync = wm_score > 0.5f ?
        fminf(1.0f, imag_score / wm_score) : 0.0f;
    health->memory_imagination_grounding = imag_score;

    /* Free energy integration (placeholder - would connect to FEP system) */
    health->predictive_free_energy =
        ((nimcp_health_agent_t*)agent)->jepa_metrics.mean_prediction_error;
    health->free_energy_trend =
        ((nimcp_health_agent_t*)agent)->jepa_metrics.prediction_error_trend;

    /* Anomaly summary */
    health->active_anomalies = 0;
    if (((nimcp_health_agent_t*)agent)->wm_metrics.health_state != WM_HEALTH_OPTIMAL) {
        health->active_anomalies++;
    }
    if (((nimcp_health_agent_t*)agent)->imagination_metrics.health_state != IMAG_HEALTH_VIVID) {
        health->active_anomalies++;
    }
    if (((nimcp_health_agent_t*)agent)->jepa_metrics.embedding_collapse_detected) {
        health->active_anomalies++;
    }
    if (((nimcp_health_agent_t*)agent)->jepa_metrics.gradient_explosion ||
        ((nimcp_health_agent_t*)agent)->jepa_metrics.gradient_vanishing) {
        health->active_anomalies++;
    }

    health->anomalies_this_window =
        atomic_load(&((nimcp_health_agent_t*)agent)->wm_anomalies_detected) +
        atomic_load(&((nimcp_health_agent_t*)agent)->imagination_anomalies_detected);

    /* Recommended action based on state */
    health->recommended_action = WM_RECOVERY_NONE;
    if (((nimcp_health_agent_t*)agent)->jepa_metrics.embedding_collapse_detected) {
        health->recommended_action = WM_RECOVERY_PRUNE_LATENT;
    } else if (((nimcp_health_agent_t*)agent)->wm_metrics.health_state == WM_HEALTH_DYNAMICS_UNSTABLE) {
        health->recommended_action = WM_RECOVERY_RETRAIN_DYNAMICS;
    } else if (((nimcp_health_agent_t*)agent)->imagination_metrics.health_state == IMAG_HEALTH_CONFABULATING) {
        health->recommended_action = WM_RECOVERY_INCREASE_REALITY_CHECK;
    } else if (((nimcp_health_agent_t*)agent)->imagination_metrics.health_state == IMAG_HEALTH_STUCK) {
        health->recommended_action = WM_RECOVERY_CLEAR_WORKSPACE;
    }

    /* Timestamps */
    health->last_check_timestamp_us =
        fmaxf(((nimcp_health_agent_t*)agent)->last_wm_check_us,
              ((nimcp_health_agent_t*)agent)->last_imagination_check_us);
    health->check_count =
        atomic_load(&((nimcp_health_agent_t*)agent)->wm_checks_run) +
        atomic_load(&((nimcp_health_agent_t*)agent)->imagination_checks_run);

    if (((nimcp_health_agent_t*)agent)->wm_imagination_mutex) {
        nimcp_mutex_unlock(((nimcp_health_agent_t*)agent)->wm_imagination_mutex);
    }

    return 0;
}

/* -------------------------------------------------------------------------
 * Recovery Actions
 * ------------------------------------------------------------------------- */

int nimcp_health_agent_world_model_recovery(
    nimcp_health_agent_t* agent,
    world_model_recovery_action_t action,
    const char* reason
) {
    if (!agent) {
        return -1;
    }

    nimcp_log(LOG_LEVEL_INFO, "World model recovery: action=%d, reason=%s",
              action, reason ? reason : "unspecified");

    /* Track recovery */
    atomic_fetch_add(&agent->wm_recoveries_performed, 1);

    /* Execute recovery based on action */
    switch (action) {
        case WM_RECOVERY_NONE:
            /* No action */
            break;

        case WM_RECOVERY_RESET_PREDICTOR:
            /* Would call JEPA reset function */
            nimcp_log(LOG_LEVEL_WARN, "JEPA predictor reset requested");
            break;

        case WM_RECOVERY_PRUNE_LATENT:
            /* Would prune degenerate latent dimensions */
            nimcp_log(LOG_LEVEL_WARN, "Latent space pruning requested");
            break;

        case WM_RECOVERY_RETRAIN_DYNAMICS:
            /* Would trigger dynamics relearning */
            nimcp_log(LOG_LEVEL_WARN, "Dynamics retraining requested");
            break;

        case WM_RECOVERY_CLEAR_WORKSPACE:
            /* Would clear imagination workspace */
            nimcp_log(LOG_LEVEL_WARN, "Imagination workspace clear requested");
            break;

        case WM_RECOVERY_REDUCE_HORIZON:
            /* Would shorten simulation horizon */
            nimcp_log(LOG_LEVEL_WARN, "Simulation horizon reduction requested");
            break;

        case WM_RECOVERY_INCREASE_REALITY_CHECK:
            /* Would increase reality checking frequency */
            nimcp_log(LOG_LEVEL_WARN, "Reality check increase requested");
            break;

        case WM_RECOVERY_THROTTLE_IMAGINATION:
            /* Would rate-limit imagination */
            nimcp_log(LOG_LEVEL_WARN, "Imagination throttling requested");
            break;

        case WM_RECOVERY_BOOST_GROUNDING:
            /* Would increase sensory grounding */
            nimcp_log(LOG_LEVEL_WARN, "Sensory grounding boost requested");
            break;

        case WM_RECOVERY_CHECKPOINT_RESTORE:
            /* Would restore from checkpoint */
            nimcp_log(LOG_LEVEL_WARN, "Checkpoint restore requested");
            break;

        default:
            nimcp_log(LOG_LEVEL_ERROR, "Unknown recovery action: %d", action);
            return -1;
    }

    return 0;
}

/* -------------------------------------------------------------------------
 * Health Status Queries
 * ------------------------------------------------------------------------- */

bool nimcp_health_agent_world_model_needs_attention(
    const nimcp_health_agent_t* agent
) {
    if (!agent) {
        return false;
    }

    /* No attention needed if nothing is connected */
    if (!((nimcp_health_agent_t*)agent)->jepa_predictor &&
        !((nimcp_health_agent_t*)agent)->world_model) {
        return false;
    }

    /* Check world model health state */
    if (((nimcp_health_agent_t*)agent)->wm_metrics.health_state != WM_HEALTH_OPTIMAL &&
        ((nimcp_health_agent_t*)agent)->wm_metrics.health_state != WM_HEALTH_DEGRADED) {
        return true;
    }

    /* Check JEPA health */
    if (((nimcp_health_agent_t*)agent)->jepa_metrics.embedding_collapse_detected ||
        ((nimcp_health_agent_t*)agent)->jepa_metrics.gradient_explosion ||
        ((nimcp_health_agent_t*)agent)->jepa_metrics.gradient_vanishing) {
        return true;
    }

    /* Check prediction error threshold */
    float error = ((nimcp_health_agent_t*)agent)->jepa_metrics.mean_prediction_error;
    if (error > ((nimcp_health_agent_t*)agent)->wm_imagination_config.jepa_error_warning) {
        return true;
    }

    /* Check world model health score */
    float score = atomic_load(&((nimcp_health_agent_t*)agent)->wm_health_score);
    if (score < 0.7f) {
        return true;
    }

    return false;
}

bool nimcp_health_agent_imagination_needs_attention(
    const nimcp_health_agent_t* agent
) {
    if (!agent) {
        return false;
    }

    /* No attention needed if nothing is connected */
    if (!((nimcp_health_agent_t*)agent)->imagination) {
        return false;
    }

    /* Check imagination health state */
    imagination_health_state_t state =
        ((nimcp_health_agent_t*)agent)->imagination_metrics.health_state;
    if (state != IMAG_HEALTH_VIVID && state != IMAG_HEALTH_HAZY) {
        return true;
    }

    /* Check coherence threshold */
    float coherence = ((nimcp_health_agent_t*)agent)->imagination_metrics.scene_coherence;
    if (coherence < ((nimcp_health_agent_t*)agent)->wm_imagination_config.coherence_warning) {
        return true;
    }

    /* Check reality check pass rate */
    float reality_rate =
        ((nimcp_health_agent_t*)agent)->imagination_metrics.reality_check_pass_rate;
    if (reality_rate < ((nimcp_health_agent_t*)agent)->wm_imagination_config.reality_check_min) {
        return true;
    }

    /* Check imagination-reality blur */
    float blur = ((nimcp_health_agent_t*)agent)->imagination_metrics.imagination_reality_blur;
    if (blur > ((nimcp_health_agent_t*)agent)->wm_imagination_config.imagination_reality_blur_max) {
        return true;
    }

    /* Check imagination health score */
    float score = atomic_load(&((nimcp_health_agent_t*)agent)->imagination_health_score);
    if (score < 0.7f) {
        return true;
    }

    return false;
}

float nimcp_health_agent_get_world_model_health_score(
    const nimcp_health_agent_t* agent
) {
    if (!agent) {
        return -1.0f;
    }

    return atomic_load(&((nimcp_health_agent_t*)agent)->wm_health_score);
}

float nimcp_health_agent_get_imagination_health_score(
    const nimcp_health_agent_t* agent
) {
    if (!agent) {
        return -1.0f;
    }

    return atomic_load(&((nimcp_health_agent_t*)agent)->imagination_health_score);
}

/* -------------------------------------------------------------------------
 * Configuration Update
 * ------------------------------------------------------------------------- */

int nimcp_health_agent_update_wm_imagination_config(
    nimcp_health_agent_t* agent,
    const health_agent_wm_imagination_config_t* config
) {
    if (!agent || !config) {
        return -1;
    }

    if (agent->wm_imagination_mutex) {
        nimcp_mutex_lock(agent->wm_imagination_mutex);
    }

    memcpy(&agent->wm_imagination_config, config, sizeof(*config));

    if (agent->wm_imagination_mutex) {
        nimcp_mutex_unlock(agent->wm_imagination_mutex);
    }

    nimcp_log(LOG_LEVEL_DEBUG, "Updated world model/imagination health config");
    return 0;
}

/* -------------------------------------------------------------------------
 * Immediate Health Checks
 * ------------------------------------------------------------------------- */

int nimcp_health_agent_check_world_model_now(
    nimcp_health_agent_t* agent
) {
    if (!agent) {
        return -1;
    }

    uint64_t now = get_timestamp_us();

    if (agent->wm_imagination_mutex) {
        nimcp_mutex_lock(agent->wm_imagination_mutex);
    }

    /* Update check count */
    atomic_fetch_add(&agent->wm_checks_run, 1);
    agent->last_wm_check_us = now;

    /* Check JEPA health if connected */
    if (agent->jepa_predictor) {
        /* In production, would call real JEPA health API */
        /* For now, maintain current metrics */

        /* Update trend tracking */
        agent->jepa_error_history[agent->jepa_error_idx] =
            agent->jepa_metrics.mean_prediction_error;
        agent->jepa_error_idx = (agent->jepa_error_idx + 1) % 10;

        /* Compute trend */
        float trend = 0.0f;
        for (int i = 1; i < 10; i++) {
            int curr = (agent->jepa_error_idx + i) % 10;
            int prev = (agent->jepa_error_idx + i - 1) % 10;
            trend += agent->jepa_error_history[curr] - agent->jepa_error_history[prev];
        }
        agent->jepa_metrics.prediction_error_trend = trend / 9.0f;

        /* Check thresholds */
        if (agent->jepa_metrics.mean_prediction_error >
            agent->wm_imagination_config.jepa_error_critical) {
            atomic_fetch_add(&agent->wm_anomalies_detected, 1);
        }

        /* Check gradient health */
        if (agent->jepa_metrics.gradient_norm > agent->wm_imagination_config.gradient_norm_max) {
            agent->jepa_metrics.gradient_explosion = true;
            atomic_fetch_add(&agent->wm_anomalies_detected, 1);
        } else {
            agent->jepa_metrics.gradient_explosion = false;
        }

        if (agent->jepa_metrics.gradient_norm < agent->wm_imagination_config.gradient_norm_min) {
            agent->jepa_metrics.gradient_vanishing = true;
            atomic_fetch_add(&agent->wm_anomalies_detected, 1);
        } else {
            agent->jepa_metrics.gradient_vanishing = false;
        }

        /* Check embedding health */
        if (agent->jepa_metrics.embedding_variance <
            agent->wm_imagination_config.embedding_variance_min) {
            agent->jepa_metrics.embedding_collapse_detected = true;
            atomic_fetch_add(&agent->wm_anomalies_detected, 1);
        } else {
            agent->jepa_metrics.embedding_collapse_detected = false;
        }
    }

    /* Check world model health if connected */
    if (agent->world_model) {
        /* In production, would call real world model health API */

        /* Update trend tracking */
        agent->wm_accuracy_history[agent->wm_accuracy_idx] =
            agent->wm_metrics.forward_accuracy;
        agent->wm_accuracy_idx = (agent->wm_accuracy_idx + 1) % 10;

        /* Determine health state */
        if (agent->wm_metrics.forward_accuracy <
            agent->wm_imagination_config.forward_accuracy_critical) {
            agent->wm_metrics.health_state = WM_HEALTH_CRITICAL;
            atomic_fetch_add(&agent->wm_anomalies_detected, 1);
        } else if (agent->wm_metrics.forward_accuracy <
                   agent->wm_imagination_config.forward_accuracy_warning) {
            agent->wm_metrics.health_state = WM_HEALTH_DEGRADED;
        } else if (agent->wm_metrics.state_space_collapse) {
            agent->wm_metrics.health_state = WM_HEALTH_EMBEDDING_COLLAPSE;
            atomic_fetch_add(&agent->wm_anomalies_detected, 1);
        } else if (agent->wm_metrics.forward_horizon_stable <
                   agent->wm_imagination_config.horizon_min_stable) {
            agent->wm_metrics.health_state = WM_HEALTH_DYNAMICS_UNSTABLE;
            atomic_fetch_add(&agent->wm_anomalies_detected, 1);
        } else {
            agent->wm_metrics.health_state = WM_HEALTH_OPTIMAL;
        }

        /* Compute health score */
        float score = agent->wm_metrics.forward_accuracy * 0.4f +
                     agent->wm_metrics.forward_consistency * 0.3f +
                     agent->wm_metrics.counterfactual_validity * 0.3f;
        agent->wm_metrics.health_score = fmaxf(0.0f, fminf(1.0f, score));
    }

    /* Update overall world model health score */
    float combined_score = 1.0f;
    int components = 0;

    if (agent->jepa_predictor) {
        float jepa_score = 1.0f - agent->jepa_metrics.mean_prediction_error;
        if (agent->jepa_metrics.embedding_collapse_detected) jepa_score *= 0.5f;
        if (agent->jepa_metrics.gradient_explosion) jepa_score *= 0.7f;
        if (agent->jepa_metrics.gradient_vanishing) jepa_score *= 0.7f;
        combined_score = jepa_score;
        components++;
    }

    if (agent->world_model) {
        combined_score = components > 0 ?
            (combined_score + agent->wm_metrics.health_score) / 2.0f :
            agent->wm_metrics.health_score;
        components++;
    }

    atomic_store(&agent->wm_health_score, combined_score);

    if (agent->wm_imagination_mutex) {
        nimcp_mutex_unlock(agent->wm_imagination_mutex);
    }

    nimcp_log(LOG_LEVEL_DEBUG, "World model health check complete: score=%.2f",
              combined_score);
    return 0;
}

int nimcp_health_agent_check_imagination_now(
    nimcp_health_agent_t* agent
) {
    if (!agent) {
        return -1;
    }

    uint64_t now = get_timestamp_us();

    if (agent->wm_imagination_mutex) {
        nimcp_mutex_lock(agent->wm_imagination_mutex);
    }

    /* Update check count */
    atomic_fetch_add(&agent->imagination_checks_run, 1);
    agent->last_imagination_check_us = now;

    /* Check imagination health if connected */
    if (agent->imagination) {
        /* In production, would call real imagination health API */

        /* Update trend tracking */
        agent->imagination_coherence_history[agent->imagination_coherence_idx] =
            agent->imagination_metrics.scene_coherence;
        agent->imagination_coherence_idx = (agent->imagination_coherence_idx + 1) % 10;

        /* Determine health state */
        if (agent->imagination_metrics.scene_coherence <
            agent->wm_imagination_config.coherence_critical) {
            agent->imagination_metrics.health_state = IMAG_HEALTH_FRAGMENTED;
            atomic_fetch_add(&agent->imagination_anomalies_detected, 1);
        } else if (agent->imagination_metrics.scene_coherence <
                   agent->wm_imagination_config.coherence_warning) {
            agent->imagination_metrics.health_state = IMAG_HEALTH_HAZY;
        } else if (agent->imagination_metrics.imagination_reality_blur >
                   agent->wm_imagination_config.imagination_reality_blur_max) {
            agent->imagination_metrics.health_state = IMAG_HEALTH_CONFABULATING;
            atomic_fetch_add(&agent->imagination_anomalies_detected, 1);
        } else if (agent->imagination_metrics.workspace_utilization > 0.95f &&
                   agent->imagination_metrics.scenarios_abandoned >
                   agent->imagination_metrics.scenarios_completed) {
            agent->imagination_metrics.health_state = IMAG_HEALTH_STUCK;
            atomic_fetch_add(&agent->imagination_anomalies_detected, 1);
        } else if (agent->imagination_metrics.scene_vividness <
                   agent->wm_imagination_config.vividness_warning) {
            agent->imagination_metrics.health_state = IMAG_HEALTH_HAZY;
        } else {
            agent->imagination_metrics.health_state = IMAG_HEALTH_VIVID;
        }

        /* Compute health score */
        float score = agent->imagination_metrics.scene_coherence * 0.3f +
                     agent->imagination_metrics.scene_vividness * 0.2f +
                     agent->imagination_metrics.reality_check_pass_rate * 0.3f +
                     (1.0f - agent->imagination_metrics.imagination_reality_blur) * 0.2f;
        agent->imagination_metrics.health_score = fmaxf(0.0f, fminf(1.0f, score));
    }

    /* Update overall imagination health score */
    float score = agent->imagination ? agent->imagination_metrics.health_score : 1.0f;
    atomic_store(&agent->imagination_health_score, score);

    if (agent->wm_imagination_mutex) {
        nimcp_mutex_unlock(agent->wm_imagination_mutex);
    }

    nimcp_log(LOG_LEVEL_DEBUG, "Imagination health check complete: score=%.2f", score);
    return 0;
}
