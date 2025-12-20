//=============================================================================
// nimcp_training_logic_bridge.h - Training-Logic Bridge Integration
//=============================================================================
//
// WHAT: Bidirectional bridge integrating training pipeline with neural logic
//       gates for symbolic reasoning about training decisions.
//
// WHY: Models prefrontal cortex executive control over learning. Training
//      metrics provide evidence for logical decision-making, while logic
//      gates determine when to adjust LR, pause training, or checkpoint.
//
// HOW: Training -> Logic: Metrics converted to boolean conditions
//      Logic -> Training: Gate outputs modulate training parameters
//      Integrates with immune (inflammation), Portia (resources), swarm (consensus)
//
//=============================================================================

#ifndef NIMCP_TRAINING_LOGIC_BRIDGE_H
#define NIMCP_TRAINING_LOGIC_BRIDGE_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// Forward Declarations
//=============================================================================

typedef struct training_logic_bridge training_logic_bridge_t;
typedef struct nimcp_brain_training_ctx nimcp_brain_training_ctx_t;
typedef struct training_immune_system training_immune_system_t;
typedef struct portia_logic_bridge portia_logic_bridge_t;
typedef struct swarm_logic_bridge swarm_logic_bridge_t;
typedef struct portia_swarm_logic_bridge portia_swarm_logic_bridge_t;

//=============================================================================
// Constants
//=============================================================================

/** Module identification */
#define TRAINING_LOGIC_MODULE_NAME      "training_logic_bridge"
#define TRAINING_LOGIC_MODULE_VERSION   "1.0.0"

/** Pre-built gate IDs */
#define TRAINING_LOGIC_GATE_STABILITY_CHECK      1
#define TRAINING_LOGIC_GATE_NEED_INTERVENTION    2
#define TRAINING_LOGIC_GATE_SAFE_TO_INCREASE_LR  3
#define TRAINING_LOGIC_GATE_BATCH_SIZE_DECISION  4
#define TRAINING_LOGIC_GATE_CHECKPOINT_DECISION  5
#define TRAINING_LOGIC_GATE_CUSTOM_START         100

/** Default thresholds */
#define TRAINING_LOGIC_DEFAULT_STABLE_STEPS         10
#define TRAINING_LOGIC_DEFAULT_LR_INCREASE_FACTOR   1.1f
#define TRAINING_LOGIC_DEFAULT_LR_DECREASE_FACTOR   0.5f
#define TRAINING_LOGIC_DEFAULT_BATCH_SCALE_FACTOR   0.75f
#define TRAINING_LOGIC_DEFAULT_CHECKPOINT_INTERVAL  100
#define TRAINING_LOGIC_DEFAULT_CONSENSUS_TIMEOUT_MS 1000
#define TRAINING_LOGIC_DEFAULT_CONFIDENCE_THRESHOLD 0.7f

/** Limits */
#define TRAINING_LOGIC_MAX_CUSTOM_GATES    50
#define TRAINING_LOGIC_MAX_HISTORY_SIZE    100
#define TRAINING_LOGIC_MAX_REASON_LENGTH   256

//=============================================================================
// Enumerations
//=============================================================================

/**
 * @brief Training condition types for logic evaluation
 *
 * WHAT: Boolean conditions derived from training state
 * WHY: Map continuous training metrics to discrete logic inputs
 * HOW: Each condition becomes an input to logic gates
 */
typedef enum {
    TRAINING_COND_LOSS_STABLE = 0,      /**< Loss is not oscillating */
    TRAINING_COND_GRAD_STABLE,           /**< Gradients within normal range */
    TRAINING_COND_LR_REASONABLE,         /**< Learning rate within bounds */
    TRAINING_COND_MEMORY_OK,             /**< Training memory available */
    TRAINING_COND_THROUGHPUT_OK,         /**< Batch throughput acceptable */
    TRAINING_COND_NOT_MID_BATCH,         /**< Not in middle of batch */
    TRAINING_COND_SUFFICIENT_PROGRESS,   /**< Enough progress since checkpoint */
    TRAINING_COND_GRAD_EXPLODING,        /**< Gradient explosion detected */
    TRAINING_COND_LOSS_NAN,              /**< Loss is NaN/Inf */
    TRAINING_COND_DIVERGING,             /**< Training is diverging */
    TRAINING_COND_STABLE_FOR_N_STEPS,    /**< Stable for N consecutive steps */
    TRAINING_COND_IMMUNE_OK,             /**< No severe inflammation */
    TRAINING_COND_RESOURCE_OK,           /**< Portia resources available */
    TRAINING_COND_SWARM_CONSENSUS,       /**< Swarm agrees on action */
    TRAINING_COND_COUNT
} training_logic_condition_t;

/**
 * @brief Training logic decision types
 *
 * WHAT: Possible actions the logic system can recommend/apply
 * WHY: Discrete decision outputs from gate evaluations
 * HOW: Each decision type has associated modulation parameters
 */
typedef enum {
    TRAINING_DECISION_CONTINUE = 0,      /**< Continue training normally */
    TRAINING_DECISION_PAUSE,             /**< Pause training */
    TRAINING_DECISION_RESUME,            /**< Resume training */
    TRAINING_DECISION_INCREASE_LR,       /**< Increase learning rate */
    TRAINING_DECISION_DECREASE_LR,       /**< Decrease learning rate */
    TRAINING_DECISION_INCREASE_BATCH,    /**< Increase batch size */
    TRAINING_DECISION_DECREASE_BATCH,    /**< Decrease batch size */
    TRAINING_DECISION_CHECKPOINT,        /**< Create checkpoint */
    TRAINING_DECISION_ROLLBACK,          /**< Rollback to checkpoint */
    TRAINING_DECISION_TERMINATE,         /**< Terminate training */
    TRAINING_DECISION_COUNT
} training_logic_decision_type_t;

/**
 * @brief Operation mode for training-logic bridge
 *
 * WHAT: How the bridge processes and applies decisions
 * WHY: Different use cases need different automation levels
 * HOW: Modes control whether decisions are advisory or automatic
 */
typedef enum {
    TRAINING_LOGIC_MODE_DISABLED = 0,        /**< Bridge disabled */
    TRAINING_LOGIC_MODE_MONITOR_ONLY,        /**< Monitor but don't modulate */
    TRAINING_LOGIC_MODE_ADVISORY,            /**< Provide recommendations */
    TRAINING_LOGIC_MODE_AUTOMATIC,           /**< Automatically apply decisions */
    TRAINING_LOGIC_MODE_CONSENSUS_REQUIRED   /**< Require swarm consensus */
} training_logic_mode_t;

/**
 * @brief Instability types from training pipeline
 *
 * WHAT: Types of training instabilities that can be signaled
 * WHY: Maps to training-immune instability types
 * HOW: Used to trigger appropriate intervention gates
 */
typedef enum {
    TRAINING_INSTABILITY_NONE = 0,
    TRAINING_INSTABILITY_LOSS_NAN,
    TRAINING_INSTABILITY_LOSS_INF,
    TRAINING_INSTABILITY_LOSS_EXPLOSION,
    TRAINING_INSTABILITY_GRAD_EXPLOSION,
    TRAINING_INSTABILITY_GRAD_VANISHING,
    TRAINING_INSTABILITY_LOSS_PLATEAU,
    TRAINING_INSTABILITY_OSCILLATION,
    TRAINING_INSTABILITY_COUNT
} training_logic_instability_t;

//=============================================================================
// Structures
//=============================================================================

/**
 * @brief Training condition state for logic evaluation
 *
 * WHAT: Current state of all training conditions
 * WHY: Provides inputs to logic gates
 * HOW: Updated via metrics or manually for testing
 */
typedef struct {
    /* Boolean conditions for gate inputs */
    bool loss_stable;            /**< Loss not oscillating */
    bool grad_stable;            /**< Gradients within range */
    bool lr_reasonable;          /**< LR within bounds */
    bool memory_ok;              /**< Memory available */
    bool throughput_ok;          /**< Throughput acceptable */
    bool not_mid_batch;          /**< Not mid-batch */
    bool sufficient_progress;    /**< Enough progress */
    bool grad_exploding;         /**< Gradient explosion */
    bool loss_nan;               /**< Loss NaN */
    bool diverging;              /**< Diverging */
    bool stable_for_n_steps;     /**< Stable N steps */
    bool immune_ok;              /**< Inflammation acceptable */
    bool resource_ok;            /**< Portia resources ok */
    bool swarm_consensus;        /**< Swarm consensus */

    /* Numeric values for fine-grained decisions */
    float loss_current;          /**< Current loss value */
    float loss_trend;            /**< Loss trend (-1 to 1) */
    float grad_norm;             /**< Current gradient norm */
    float learning_rate;         /**< Current LR */
    float memory_usage;          /**< Memory usage (0-1) */
    float throughput;            /**< Batches per second */
    uint32_t steps_since_checkpoint;  /**< Steps since last checkpoint */
    uint32_t stable_step_count;  /**< Consecutive stable steps */
    uint64_t current_step;       /**< Current training step */
} training_logic_conditions_t;

/**
 * @brief Training logic decision result
 *
 * WHAT: Output of logic gate evaluation
 * WHY: Communicates what action should be taken
 * HOW: Includes decision type, confidence, and explanation
 */
typedef struct {
    training_logic_decision_type_t type;  /**< Decision type */
    bool approved;                         /**< Decision approved */
    float confidence;                      /**< Decision confidence [0-1] */
    float modulation_factor;              /**< Factor for LR/batch adjustment */
    char reason[TRAINING_LOGIC_MAX_REASON_LENGTH]; /**< Human-readable reason */
    uint64_t evaluation_time_us;          /**< Evaluation time (microseconds) */

    /* Gate results that contributed to decision */
    bool stability_check_passed;
    bool intervention_needed;
    bool safe_to_increase_lr;
    bool batch_size_ok;
    bool checkpoint_needed;
} training_logic_decision_t;

/**
 * @brief Configuration for training-logic bridge
 *
 * WHAT: All configurable parameters for the bridge
 * WHY: Allow customization for different training scenarios
 * HOW: Passed to create function, copied internally
 */
typedef struct {
    training_logic_mode_t mode;          /**< Operation mode */

    /* Gate thresholds */
    float stability_threshold;           /**< Threshold for stability check */
    float intervention_threshold;        /**< Threshold for intervention */
    float lr_increase_threshold;         /**< Threshold to allow LR increase */
    float confidence_threshold;          /**< Min confidence for decisions */

    /* Modulation parameters */
    float lr_increase_factor;            /**< Factor for LR increase (default 1.1) */
    float lr_decrease_factor;            /**< Factor for LR decrease (default 0.5) */
    float batch_scale_factor;            /**< Factor for batch scaling (default 0.75) */
    uint32_t stable_steps_required;      /**< Steps required before LR increase */
    uint32_t checkpoint_interval;        /**< Steps between checkpoints */

    /* Integration flags */
    bool enable_immune_integration;      /**< Enable immune awareness */
    bool enable_portia_integration;      /**< Enable Portia integration */
    bool enable_swarm_integration;       /**< Enable swarm consensus */
    bool enable_bio_async;               /**< Enable bio-async messaging */

    /* Safety settings */
    float min_learning_rate;             /**< Minimum LR floor */
    float max_learning_rate;             /**< Maximum LR ceiling */
    uint32_t min_batch_size;             /**< Minimum batch size */
    uint32_t max_batch_size;             /**< Maximum batch size */

    /* Consensus settings */
    uint32_t consensus_timeout_ms;       /**< Timeout for swarm consensus */
    float consensus_threshold;           /**< Required consensus level */

    /* History settings */
    uint32_t history_size;               /**< Size of metric history buffer */

    /* Testing support */
    bool disable_auto_update;            /**< Disable auto condition updates */
} training_logic_config_t;

/**
 * @brief Statistics for training-logic bridge
 *
 * WHAT: Tracking of bridge activity and performance
 * WHY: Monitoring and debugging
 * HOW: Accumulated during operation, queryable via API
 */
typedef struct {
    /* Decision counts */
    uint64_t total_decisions;
    uint64_t decisions_by_type[TRAINING_DECISION_COUNT];

    /* Gate evaluation stats */
    uint64_t stability_checks;
    uint64_t stability_passed;
    uint64_t intervention_triggers;
    uint64_t lr_increase_allowed;
    uint64_t lr_decrease_triggered;
    uint64_t batch_adjustments;
    uint64_t checkpoints_triggered;

    /* Modulation stats */
    uint64_t lr_increases;
    uint64_t lr_decreases;
    uint64_t batch_increases;
    uint64_t batch_decreases;

    /* Timing */
    float avg_decision_time_us;
    float max_decision_time_us;
    uint64_t total_decision_time_us;

    /* Consensus stats */
    uint64_t consensus_requests;
    uint64_t consensus_achieved;
    uint64_t consensus_timeouts;

    /* Current state */
    training_logic_mode_t current_mode;
    bool currently_paused;
    uint64_t last_decision_time_ms;

    /* Custom gates */
    uint32_t custom_gate_count;
} training_logic_stats_t;

//=============================================================================
// Lifecycle API
//=============================================================================

/**
 * @brief Initialize configuration with default values
 *
 * WHAT: Populates config struct with sensible defaults
 * WHY: Ensure all fields have valid initial values
 * HOW: Sets mode to ADVISORY, enables bio-async, etc.
 *
 * @param config Configuration to initialize (must not be NULL)
 */
void training_logic_default_config(training_logic_config_t* config);

/**
 * @brief Create a new training-logic bridge
 *
 * WHAT: Allocates and initializes bridge structure
 * WHY: Entry point for using the bridge
 * HOW: Creates logic network, gates, allocates history buffer
 *
 * @param config Configuration (NULL uses defaults)
 * @return Bridge handle or NULL on failure
 */
training_logic_bridge_t* training_logic_create(
    const training_logic_config_t* config
);

/**
 * @brief Destroy a training-logic bridge
 *
 * WHAT: Frees all resources associated with bridge
 * WHY: Proper cleanup
 * HOW: Disconnects integrations, frees memory, destroys mutex
 *
 * @param bridge Bridge to destroy (NULL-safe)
 */
void training_logic_destroy(training_logic_bridge_t* bridge);

/**
 * @brief Start the training-logic bridge
 *
 * WHAT: Activates the bridge for operation
 * WHY: Allows deferred start after configuration
 * HOW: Connects bio-async if enabled, initializes state
 *
 * @param bridge Bridge to start
 * @return 0 on success, negative on error
 */
int training_logic_start(training_logic_bridge_t* bridge);

/**
 * @brief Stop the training-logic bridge
 *
 * WHAT: Deactivates the bridge
 * WHY: Pause operation without destroying
 * HOW: Disconnects bio-async, preserves state
 *
 * @param bridge Bridge to stop
 * @return 0 on success, negative on error
 */
int training_logic_stop(training_logic_bridge_t* bridge);

//=============================================================================
// Integration API
//=============================================================================

/**
 * @brief Connect to brain training context
 *
 * WHAT: Links bridge to training pipeline
 * WHY: Enables automatic metric updates
 * HOW: Stores reference, may register callbacks
 *
 * @param bridge Bridge to connect
 * @param training_ctx Training context (may be NULL)
 * @return 0 on success, negative on error
 */
int training_logic_connect_brain_training(
    training_logic_bridge_t* bridge,
    nimcp_brain_training_ctx_t* training_ctx
);

/**
 * @brief Connect to training-immune system
 *
 * WHAT: Links bridge to immune awareness
 * WHY: Inflammation affects training decisions
 * HOW: Queries inflammation level for immune_ok condition
 *
 * @param bridge Bridge to connect
 * @param immune_system Immune system (may be NULL)
 * @return 0 on success, negative on error
 */
int training_logic_connect_training_immune(
    training_logic_bridge_t* bridge,
    training_immune_system_t* immune_system
);

/**
 * @brief Connect to Portia-logic bridge
 *
 * WHAT: Links bridge to resource awareness
 * WHY: Resource state affects training decisions
 * HOW: Queries Portia conditions for resource_ok
 *
 * @param bridge Bridge to connect
 * @param portia_logic Portia-logic bridge (may be NULL)
 * @return 0 on success, negative on error
 */
int training_logic_connect_portia_logic(
    training_logic_bridge_t* bridge,
    portia_logic_bridge_t* portia_logic
);

/**
 * @brief Connect to swarm-logic bridge
 *
 * WHAT: Links bridge to swarm consensus
 * WHY: Distributed training requires agreement
 * HOW: Requests consensus for decisions when in consensus mode
 *
 * @param bridge Bridge to connect
 * @param swarm_logic Swarm-logic bridge (may be NULL)
 * @return 0 on success, negative on error
 */
int training_logic_connect_swarm_logic(
    training_logic_bridge_t* bridge,
    swarm_logic_bridge_t* swarm_logic
);

/**
 * @brief Connect to unified Portia-Swarm-Logic bridge
 *
 * WHAT: Links bridge to unified coordination
 * WHY: Full coordination across all systems
 * HOW: Uses unified bridge for coordinated decisions
 *
 * @param bridge Bridge to connect
 * @param unified_bridge Unified bridge (may be NULL)
 * @return 0 on success, negative on error
 */
int training_logic_connect_unified(
    training_logic_bridge_t* bridge,
    portia_swarm_logic_bridge_t* unified_bridge
);

//=============================================================================
// Training -> Logic: Metric Updates
//=============================================================================

/**
 * @brief Update training metrics
 *
 * WHAT: Reports current training state to logic system
 * WHY: Metrics are converted to conditions for gates
 * HOW: Updates loss_stable, grad_stable, etc. based on values
 *
 * @param bridge Bridge to update
 * @param loss Current loss value
 * @param grad_norm Current gradient norm
 * @param learning_rate Current learning rate
 * @param step Current training step
 * @return 0 on success, negative on error
 */
int training_logic_update_metrics(
    training_logic_bridge_t* bridge,
    float loss,
    float grad_norm,
    float learning_rate,
    uint64_t step
);

/**
 * @brief Update batch-related metrics
 *
 * WHAT: Reports batch processing state
 * WHY: Affects batch size decisions
 * HOW: Updates memory_ok, throughput_ok conditions
 *
 * @param bridge Bridge to update
 * @param batch_size Current batch size
 * @param throughput Batches per second
 * @param memory_usage Memory usage (0-1)
 * @return 0 on success, negative on error
 */
int training_logic_update_batch_metrics(
    training_logic_bridge_t* bridge,
    uint32_t batch_size,
    float throughput,
    float memory_usage
);

/**
 * @brief Signal a training instability
 *
 * WHAT: Reports specific instability event
 * WHY: Triggers appropriate intervention gates
 * HOW: Sets corresponding condition flags
 *
 * @param bridge Bridge to signal
 * @param instability_type Type of instability
 * @param severity Severity level (0-10)
 * @return 0 on success, negative on error
 */
int training_logic_signal_instability(
    training_logic_bridge_t* bridge,
    training_logic_instability_t instability_type,
    uint32_t severity
);

//=============================================================================
// Logic -> Training: Decision Evaluation
//=============================================================================

/**
 * @brief Check training stability
 *
 * WHAT: Evaluates STABILITY_CHECK gate (AND gate)
 * WHY: Determine if training is healthy
 * HOW: Returns true if loss_stable AND grad_stable AND lr_reasonable
 *
 * @param bridge Bridge to query
 * @return true if stable, false otherwise
 */
bool training_logic_check_stability(training_logic_bridge_t* bridge);

/**
 * @brief Check if intervention is needed
 *
 * WHAT: Evaluates NEED_INTERVENTION gate (OR gate)
 * WHY: Detect training problems requiring action
 * HOW: Returns true if grad_exploding OR loss_nan OR diverging
 *
 * @param bridge Bridge to query
 * @return true if intervention needed, false otherwise
 */
bool training_logic_needs_intervention(training_logic_bridge_t* bridge);

/**
 * @brief Check if learning rate can be increased
 *
 * WHAT: Evaluates SAFE_TO_INCREASE_LR gate (IMPLIES gate)
 * WHY: Only increase LR when safe
 * HOW: stable_for_n_steps IMPLIES (immune_ok AND resource_ok)
 *
 * @param bridge Bridge to query
 * @return true if safe to increase, false otherwise
 */
bool training_logic_can_increase_lr(training_logic_bridge_t* bridge);

/**
 * @brief Check if batch size should be adjusted
 *
 * WHAT: Evaluates BATCH_SIZE_DECISION gate (AND gate)
 * WHY: Determine if batch adjustment is appropriate
 * HOW: Returns true if memory_ok AND throughput_ok
 *
 * @param bridge Bridge to query
 * @param increase_batch Output: true to increase, false to decrease
 * @return true if adjustment recommended, false otherwise
 */
bool training_logic_should_adjust_batch(
    training_logic_bridge_t* bridge,
    bool* increase_batch
);

/**
 * @brief Check if checkpoint should be created
 *
 * WHAT: Evaluates CHECKPOINT_DECISION gate (AND gate)
 * WHY: Trigger checkpoints at appropriate times
 * HOW: memory_ok AND not_mid_batch AND sufficient_progress
 *
 * @param bridge Bridge to query
 * @return true if checkpoint recommended, false otherwise
 */
bool training_logic_should_checkpoint(training_logic_bridge_t* bridge);

/**
 * @brief Get comprehensive decision
 *
 * WHAT: Evaluates all gates and returns best decision
 * WHY: Single call for decision-making
 * HOW: Evaluates gates in priority order, returns highest priority
 *
 * @param bridge Bridge to query
 * @param decision Output decision result
 * @return 0 on success, negative on error
 */
int training_logic_get_decision(
    training_logic_bridge_t* bridge,
    training_logic_decision_t* decision
);

//=============================================================================
// Modulation API
//=============================================================================

/**
 * @brief Get modulated learning rate
 *
 * WHAT: Apply logic-based LR modulation
 * WHY: Automatic LR adjustment based on conditions
 * HOW: Multiplies base_lr by computed factor
 *
 * @param bridge Bridge to query
 * @param base_lr Base learning rate
 * @return Modulated learning rate
 */
float training_logic_get_lr_modulation(
    const training_logic_bridge_t* bridge,
    float base_lr
);

/**
 * @brief Get modulated batch size
 *
 * WHAT: Apply logic-based batch size modulation
 * WHY: Automatic batch size adjustment based on conditions
 * HOW: Scales base_batch_size by computed factor
 *
 * @param bridge Bridge to query
 * @param base_batch_size Base batch size
 * @return Modulated batch size
 */
uint32_t training_logic_get_batch_size_modulation(
    const training_logic_bridge_t* bridge,
    uint32_t base_batch_size
);

/**
 * @brief Apply a decision
 *
 * WHAT: Execute a training logic decision
 * WHY: Automatic mode applies decisions
 * HOW: Updates internal state, signals integrations
 *
 * @param bridge Bridge to apply to
 * @param decision Decision to apply
 * @return 0 on success, negative on error
 */
int training_logic_apply_decision(
    training_logic_bridge_t* bridge,
    const training_logic_decision_t* decision
);

//=============================================================================
// Condition Management API
//=============================================================================

/**
 * @brief Update all conditions from connected systems
 *
 * WHAT: Refresh conditions from integrations
 * WHY: Sync with external state
 * HOW: Queries immune, Portia, swarm for current state
 *
 * @param bridge Bridge to update
 * @return 0 on success, negative on error
 */
int training_logic_update_conditions(training_logic_bridge_t* bridge);

/**
 * @brief Get current conditions
 *
 * WHAT: Retrieve current condition state
 * WHY: Debugging and monitoring
 * HOW: Copies internal conditions to output
 *
 * @param bridge Bridge to query
 * @param conditions Output conditions
 * @return 0 on success, negative on error
 */
int training_logic_get_conditions(
    const training_logic_bridge_t* bridge,
    training_logic_conditions_t* conditions
);

/**
 * @brief Set a specific condition manually
 *
 * WHAT: Override a condition value
 * WHY: Testing and manual control
 * HOW: Sets specified condition to value
 *
 * @param bridge Bridge to modify
 * @param condition Condition to set
 * @param value Value to set
 * @return 0 on success, negative on error
 */
int training_logic_set_condition(
    training_logic_bridge_t* bridge,
    training_logic_condition_t condition,
    bool value
);

/**
 * @brief Set a numeric condition value
 *
 * WHAT: Set numeric metric directly
 * WHY: Testing and manual control
 * HOW: Updates specified numeric field
 *
 * @param bridge Bridge to modify
 * @param name Condition name ("loss", "grad_norm", "learning_rate", etc.)
 * @param value Numeric value
 * @return 0 on success, negative on error
 */
int training_logic_set_numeric_condition(
    training_logic_bridge_t* bridge,
    const char* name,
    float value
);

//=============================================================================
// Custom Gate API
//=============================================================================

/**
 * @brief Add a custom logic gate
 *
 * WHAT: Create user-defined logic gate
 * WHY: Extend default gates for specific needs
 * HOW: Parses expression, creates gate in network
 *
 * @param bridge Bridge to add gate to
 * @param expression Logic expression (e.g., "A AND B", "NOT C")
 * @param gate_id Output gate ID
 * @return 0 on success, negative on error
 */
int training_logic_add_custom_gate(
    training_logic_bridge_t* bridge,
    const char* expression,
    uint32_t* gate_id
);

/**
 * @brief Evaluate a specific gate
 *
 * WHAT: Evaluate gate by ID
 * WHY: Query specific gate result
 * HOW: Evaluates gate with current conditions
 *
 * @param bridge Bridge to query
 * @param gate_id Gate ID to evaluate
 * @return Gate result (true/false)
 */
bool training_logic_evaluate_gate(
    training_logic_bridge_t* bridge,
    uint32_t gate_id
);

/**
 * @brief Get detailed gate decision
 *
 * WHAT: Get full decision info for gate
 * WHY: Debugging and explanation
 * HOW: Evaluates gate, populates decision struct
 *
 * @param bridge Bridge to query
 * @param gate_id Gate ID to evaluate
 * @param decision Output decision
 * @return 0 on success, negative on error
 */
int training_logic_get_gate_decision(
    training_logic_bridge_t* bridge,
    uint32_t gate_id,
    training_logic_decision_t* decision
);

//=============================================================================
// Bio-Async API
//=============================================================================

/**
 * @brief Connect to bio-async router
 *
 * WHAT: Register with bio-async messaging system
 * WHY: Enable inter-module communication
 * HOW: Registers module, creates inbox
 *
 * @param bridge Bridge to connect
 * @return 0 on success, negative on error
 */
int training_logic_connect_bio_async(training_logic_bridge_t* bridge);

/**
 * @brief Disconnect from bio-async router
 *
 * WHAT: Unregister from bio-async system
 * WHY: Clean shutdown
 * HOW: Unregisters module
 *
 * @param bridge Bridge to disconnect
 * @return 0 on success, negative on error
 */
int training_logic_disconnect_bio_async(training_logic_bridge_t* bridge);

/**
 * @brief Check if bio-async is connected
 *
 * WHAT: Query bio-async connection status
 * WHY: Determine if messaging is available
 * HOW: Returns internal flag
 *
 * @param bridge Bridge to query
 * @return true if connected, false otherwise
 */
bool training_logic_is_bio_async_connected(const training_logic_bridge_t* bridge);

/**
 * @brief Process incoming bio-async messages
 *
 * WHAT: Handle messages in inbox
 * WHY: Respond to external events
 * HOW: Dequeues and processes messages
 *
 * @param bridge Bridge to process
 * @return Number of messages processed, negative on error
 */
int training_logic_process_inbox(training_logic_bridge_t* bridge);

/**
 * @brief Broadcast a decision via bio-async
 *
 * WHAT: Send decision to subscribers
 * WHY: Notify other modules of training decisions
 * HOW: Creates message, sends to router
 *
 * @param bridge Bridge to broadcast from
 * @param decision Decision to broadcast
 * @return 0 on success, negative on error
 */
int training_logic_broadcast_decision(
    training_logic_bridge_t* bridge,
    const training_logic_decision_t* decision
);

//=============================================================================
// Statistics API
//=============================================================================

/**
 * @brief Get bridge statistics
 *
 * WHAT: Retrieve accumulated statistics
 * WHY: Monitoring and debugging
 * HOW: Copies internal stats to output
 *
 * @param bridge Bridge to query
 * @param stats Output statistics
 * @return 0 on success, negative on error
 */
int training_logic_get_stats(
    const training_logic_bridge_t* bridge,
    training_logic_stats_t* stats
);

/**
 * @brief Reset bridge statistics
 *
 * WHAT: Clear accumulated statistics
 * WHY: Start fresh measurement period
 * HOW: Zeros all stat counters
 *
 * @param bridge Bridge to reset
 * @return 0 on success, negative on error
 */
int training_logic_reset_stats(training_logic_bridge_t* bridge);

//=============================================================================
// Utility API
//=============================================================================

/**
 * @brief Convert condition enum to string
 *
 * @param condition Condition to convert
 * @return String name of condition
 */
const char* training_logic_condition_to_string(training_logic_condition_t condition);

/**
 * @brief Convert decision type enum to string
 *
 * @param type Decision type to convert
 * @return String name of decision type
 */
const char* training_logic_decision_type_to_string(training_logic_decision_type_t type);

/**
 * @brief Convert mode enum to string
 *
 * @param mode Mode to convert
 * @return String name of mode
 */
const char* training_logic_mode_to_string(training_logic_mode_t mode);

/**
 * @brief Dump bridge state for debugging
 *
 * @param bridge Bridge to dump
 */
void training_logic_dump_state(const training_logic_bridge_t* bridge);

#ifdef __cplusplus
}
#endif

#endif /* NIMCP_TRAINING_LOGIC_BRIDGE_H */
