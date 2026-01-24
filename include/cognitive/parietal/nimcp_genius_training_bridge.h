/**
 * @file nimcp_genius_training_bridge.h
 * @brief Mathematical Genius - Training System Integration Bridge
 * @version 1.0.0
 * @date 2026-01-24
 *
 * WHAT: Bridge between mathematical genius module and the brain training system
 * WHY:  Enable systematic training of mathematical reasoning capabilities through
 *       curriculum learning, mixed precision training, and continual learning
 * HOW:  Integrate genius module with training dispatch, curriculum scheduling,
 *       and learning rate adaptation for progressive mathematical skill development
 *
 * THEORETICAL FOUNDATIONS:
 * - Bengio (2009): Curriculum learning for neural networks
 * - Kirkpatrick (2017): Overcoming catastrophic forgetting in neural networks
 * - Dehaene (2011): Teaching mathematical intuition
 *
 * BIOLOGICAL BASIS:
 * - Progressive skill acquisition in parietal cortex
 * - Consolidation during rest/sleep for mathematical memory
 * - Transfer learning between mathematical domains
 * - Meta-learning for adapting to new problem types
 *
 * TRAINING ASPECTS:
 * - Curriculum: Progress from simple to complex mathematical problems
 * - Continual: Learn new mathematics without forgetting fundamentals
 * - Transfer: Apply learned patterns across domains
 * - Meta-learning: Learn to learn mathematics more efficiently
 *
 * @see nimcp_mathematical_genius.h
 * @see nimcp_training_dispatch.h
 * @see nimcp_curriculum_learning.h
 * @see nimcp_continual_learning.h
 */

#ifndef NIMCP_GENIUS_TRAINING_BRIDGE_H
#define NIMCP_GENIUS_TRAINING_BRIDGE_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "cognitive/parietal/nimcp_genius_modes.h"

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// Constants
//=============================================================================

/** @brief Maximum training tasks */
#define GENIUS_TRAINING_MAX_TASKS         64

/** @brief Maximum curriculum stages */
#define GENIUS_TRAINING_MAX_STAGES        16

/** @brief Default batch size */
#define GENIUS_TRAINING_DEFAULT_BATCH     32

/** @brief Bio-async module ID */
#define BIO_MODULE_GENIUS_TRAINING        0x039A

//=============================================================================
// Enumerations
//=============================================================================

/**
 * @brief Training domain types
 */
typedef enum {
    GENIUS_TRAIN_DOMAIN_NUMBER_THEORY = 0,   /**< Number theory (Gauss) */
    GENIUS_TRAIN_DOMAIN_CALCULUS,             /**< Calculus (Newton) */
    GENIUS_TRAIN_DOMAIN_COMBINATORICS,        /**< Combinatorics (Erdos) */
    GENIUS_TRAIN_DOMAIN_GRAPH_THEORY,         /**< Graph theory */
    GENIUS_TRAIN_DOMAIN_ALGEBRA,              /**< Abstract algebra */
    GENIUS_TRAIN_DOMAIN_GEOMETRY,             /**< Geometry */
    GENIUS_TRAIN_DOMAIN_PROBABILITY,          /**< Probability theory */
    GENIUS_TRAIN_DOMAIN_LOGIC,                /**< Mathematical logic */
    GENIUS_TRAIN_DOMAIN_CROSS_DOMAIN,         /**< Cross-domain problems */
    GENIUS_TRAIN_DOMAIN_COUNT
} genius_train_domain_t;

/**
 * @brief Training task types
 */
typedef enum {
    GENIUS_TASK_PATTERN_RECOGNITION = 0,     /**< Recognize patterns */
    GENIUS_TASK_THEOREM_PROVING,              /**< Prove theorems */
    GENIUS_TASK_CONJECTURE_GENERATION,        /**< Generate conjectures */
    GENIUS_TASK_ANALOGY_FINDING,              /**< Find cross-domain analogies */
    GENIUS_TASK_PROBLEM_SOLVING,              /**< General problem solving */
    GENIUS_TASK_ELEGANCE_OPTIMIZATION,        /**< Optimize for elegant proofs */
    GENIUS_TASK_MODE_SELECTION,               /**< Learn optimal mode selection */
    GENIUS_TASK_COUNT
} genius_task_type_t;

/**
 * @brief Curriculum stage
 */
typedef enum {
    GENIUS_STAGE_NOVICE = 0,                 /**< Basic concepts */
    GENIUS_STAGE_INTERMEDIATE,                /**< Standard problems */
    GENIUS_STAGE_ADVANCED,                    /**< Advanced techniques */
    GENIUS_STAGE_EXPERT,                      /**< Expert-level problems */
    GENIUS_STAGE_MASTERY,                     /**< Mastery challenges */
    GENIUS_STAGE_RESEARCH                     /**< Research-level problems */
} genius_curriculum_stage_t;

/**
 * @brief Training state
 */
typedef enum {
    GENIUS_TRAINING_STATE_IDLE = 0,
    GENIUS_TRAINING_STATE_TRAINING,
    GENIUS_TRAINING_STATE_VALIDATING,
    GENIUS_TRAINING_STATE_CONSOLIDATING,
    GENIUS_TRAINING_STATE_CURRICULUM_ADVANCING,
    GENIUS_TRAINING_STATE_PAUSED,
    GENIUS_TRAINING_STATE_ERROR
} genius_training_state_t;

//=============================================================================
// Configuration
//=============================================================================

/**
 * @brief Genius-Training bridge configuration
 */
typedef struct {
    /* Training parameters */
    float base_learning_rate;            /**< Base learning rate */
    float learning_rate_decay;           /**< Learning rate decay factor */
    float momentum;                      /**< Momentum coefficient */
    float weight_decay;                  /**< Weight decay (L2 regularization) */
    uint32_t batch_size;                 /**< Training batch size */

    /* Curriculum parameters */
    bool enable_curriculum;              /**< Enable curriculum learning */
    float curriculum_advancement_thresh; /**< Score to advance stage */
    float curriculum_regression_thresh;  /**< Score to regress stage */
    uint32_t min_examples_per_stage;     /**< Minimum examples before advancing */

    /* Continual learning */
    bool enable_continual_learning;      /**< Enable continual learning */
    float ewc_lambda;                    /**< EWC regularization strength */
    bool enable_replay;                  /**< Enable experience replay */
    uint32_t replay_buffer_size;         /**< Replay buffer size */

    /* Transfer learning */
    bool enable_transfer;                /**< Enable transfer learning */
    float transfer_weight;               /**< Weight for transfer loss */

    /* Meta-learning */
    bool enable_meta_learning;           /**< Enable meta-learning */
    float meta_learning_rate;            /**< Meta-learning rate */

    /* Mode-specific training */
    bool train_gauss_mode;               /**< Train Gauss mode */
    bool train_newton_mode;              /**< Train Newton mode */
    bool train_erdos_mode;               /**< Train Erdos mode */
    float mode_specific_weight;          /**< Weight for mode-specific loss */

    /* Scheduling */
    uint32_t epochs_per_domain;          /**< Epochs per domain */
    bool enable_domain_rotation;         /**< Rotate training domains */
    float domain_rotation_period;        /**< Domain rotation period */

    /* Validation */
    float validation_split;              /**< Validation data fraction */
    uint32_t validation_frequency;       /**< Validation every N batches */

    /* Checkpointing */
    bool enable_checkpointing;           /**< Enable model checkpoints */
    uint32_t checkpoint_frequency;       /**< Checkpoint every N epochs */

    /* Bio-async integration */
    bool enable_bio_async;               /**< Enable bio-async callbacks */
} genius_training_config_t;

//=============================================================================
// State Structures
//=============================================================================

/**
 * @brief Training task specification
 */
typedef struct {
    uint32_t task_id;                    /**< Task identifier */
    genius_task_type_t type;             /**< Task type */
    genius_train_domain_t domain;        /**< Training domain */
    genius_curriculum_stage_t stage;     /**< Curriculum stage */
    float difficulty;                    /**< Task difficulty [0-1] */
    float weight;                        /**< Task weight in curriculum */
    uint32_t num_examples;               /**< Number of training examples */
    float completion_rate;               /**< Completion rate */
    float avg_score;                     /**< Average score achieved */
} genius_training_task_t;

/**
 * @brief Curriculum state
 */
typedef struct {
    genius_curriculum_stage_t current_stage;  /**< Current stage */
    genius_train_domain_t current_domain;     /**< Current domain */
    uint32_t examples_in_stage;               /**< Examples in current stage */
    float stage_score;                        /**< Current stage score */
    float domain_scores[GENIUS_TRAIN_DOMAIN_COUNT]; /**< Score per domain */
    float stage_progress;                     /**< Progress in stage [0-1] */
    uint32_t stages_completed;                /**< Total stages completed */
    uint32_t advancements;                    /**< Total advancements */
    uint32_t regressions;                     /**< Total regressions */
} genius_curriculum_state_t;

/**
 * @brief Training progress
 */
typedef struct {
    uint64_t total_examples_trained;     /**< Total examples trained */
    uint64_t total_batches;              /**< Total batches processed */
    uint64_t total_epochs;               /**< Total epochs completed */
    float current_loss;                  /**< Current training loss */
    float current_accuracy;              /**< Current accuracy */
    float validation_loss;               /**< Latest validation loss */
    float validation_accuracy;           /**< Latest validation accuracy */
    float learning_rate_current;         /**< Current learning rate */
    float best_validation_score;         /**< Best validation score achieved */
} genius_training_progress_t;

/**
 * @brief Mode-specific training state
 */
typedef struct {
    float gauss_skill;                   /**< Gauss mode skill level */
    float newton_skill;                  /**< Newton mode skill level */
    float erdos_skill;                   /**< Erdos mode skill level */
    float gauss_loss;                    /**< Gauss mode training loss */
    float newton_loss;                   /**< Newton mode training loss */
    float erdos_loss;                    /**< Erdos mode training loss */
    genius_mode_t best_mode;             /**< Current best mode */
    float mode_selection_accuracy;       /**< Mode selection accuracy */
} genius_mode_training_state_t;

/**
 * @brief Bridge state information
 */
typedef struct {
    genius_training_state_t state;       /**< Current operational state */
    uint32_t active_tasks;               /**< Number of active tasks */
    float overall_progress;              /**< Overall training progress [0-1] */
    float overall_skill;                 /**< Overall mathematical skill */
    genius_curriculum_stage_t max_stage; /**< Highest stage reached */
} genius_training_bridge_state_t;

/**
 * @brief Bridge statistics
 */
typedef struct {
    uint64_t total_training_time_us;     /**< Total training time */
    uint64_t total_validation_time_us;   /**< Total validation time */
    uint64_t total_examples;             /**< Total examples processed */
    uint64_t successful_proofs;          /**< Successful proofs during training */
    uint64_t failed_proofs;              /**< Failed proofs during training */
    uint64_t patterns_learned;           /**< Patterns successfully learned */
    uint64_t conjectures_generated;      /**< Conjectures generated in training */
    float mean_batch_time_ms;            /**< Mean batch processing time */
    float peak_accuracy;                 /**< Peak accuracy achieved */
    float final_accuracy;                /**< Final accuracy */
    float curriculum_efficiency;         /**< Curriculum advancement rate */
} genius_training_stats_t;

//=============================================================================
// Opaque Bridge Handle
//=============================================================================

typedef struct genius_training_bridge genius_training_bridge_t;

//=============================================================================
// Forward Declarations
//=============================================================================

struct mathematical_genius;
struct nimcp_brain_training_ctx;

//=============================================================================
// Callbacks
//=============================================================================

/** @brief Epoch completion callback */
typedef void (*genius_training_epoch_callback_t)(
    genius_training_bridge_t* bridge,
    uint64_t epoch,
    float loss,
    float accuracy,
    void* user_data
);

/** @brief Curriculum advancement callback */
typedef void (*genius_training_curriculum_callback_t)(
    genius_training_bridge_t* bridge,
    genius_curriculum_stage_t old_stage,
    genius_curriculum_stage_t new_stage,
    genius_train_domain_t domain,
    void* user_data
);

/** @brief Checkpoint callback */
typedef void (*genius_training_checkpoint_callback_t)(
    genius_training_bridge_t* bridge,
    uint64_t checkpoint_id,
    float validation_score,
    void* user_data
);

//=============================================================================
// Lifecycle Functions
//=============================================================================

/**
 * @brief Create default configuration
 * @return Default configuration structure
 */
genius_training_config_t genius_training_config_default(void);

/**
 * @brief Create genius training bridge
 * @param config Configuration (NULL for defaults)
 * @return Bridge handle or NULL on failure
 */
genius_training_bridge_t* genius_training_create(
    const genius_training_config_t* config
);

/**
 * @brief Destroy genius training bridge
 * @param bridge Bridge to destroy
 */
void genius_training_destroy(genius_training_bridge_t* bridge);

/**
 * @brief Reset bridge state
 * @param bridge Bridge handle
 * @return 0 on success, -1 on failure
 */
int genius_training_reset(genius_training_bridge_t* bridge);

/**
 * @brief Link to mathematical genius module
 * @param bridge Bridge handle
 * @param genius Mathematical genius handle
 * @return 0 on success, -1 on failure
 */
int genius_training_link_genius(
    genius_training_bridge_t* bridge,
    struct mathematical_genius* genius
);

/**
 * @brief Link to brain training context
 * @param bridge Bridge handle
 * @param ctx Brain training context
 * @return 0 on success, -1 on failure
 */
int genius_training_link_brain_training(
    genius_training_bridge_t* bridge,
    struct nimcp_brain_training_ctx* ctx
);

//=============================================================================
// Task Management
//=============================================================================

/**
 * @brief Register a training task
 * @param bridge Bridge handle
 * @param task Task specification
 * @return Task ID on success, -1 on failure
 */
int genius_training_register_task(
    genius_training_bridge_t* bridge,
    const genius_training_task_t* task
);

/**
 * @brief Unregister a training task
 * @param bridge Bridge handle
 * @param task_id Task ID
 * @return 0 on success, -1 on failure
 */
int genius_training_unregister_task(
    genius_training_bridge_t* bridge,
    uint32_t task_id
);

/**
 * @brief Get task state
 * @param bridge Bridge handle
 * @param task_id Task ID
 * @param task Output task structure
 * @return 0 on success, -1 on failure
 */
int genius_training_get_task(
    genius_training_bridge_t* bridge,
    uint32_t task_id,
    genius_training_task_t* task
);

//=============================================================================
// Training Functions
//=============================================================================

/**
 * @brief Start training
 * @param bridge Bridge handle
 * @return 0 on success, -1 on failure
 */
int genius_training_start(genius_training_bridge_t* bridge);

/**
 * @brief Pause training
 * @param bridge Bridge handle
 * @return 0 on success, -1 on failure
 */
int genius_training_pause(genius_training_bridge_t* bridge);

/**
 * @brief Resume training
 * @param bridge Bridge handle
 * @return 0 on success, -1 on failure
 */
int genius_training_resume(genius_training_bridge_t* bridge);

/**
 * @brief Stop training
 * @param bridge Bridge handle
 * @return 0 on success, -1 on failure
 */
int genius_training_stop(genius_training_bridge_t* bridge);

/**
 * @brief Train single batch
 * @param bridge Bridge handle
 * @param inputs Input data
 * @param targets Target data
 * @param batch_size Batch size
 * @return Loss value, -1 on failure
 */
float genius_training_train_batch(
    genius_training_bridge_t* bridge,
    const void* inputs,
    const void* targets,
    uint32_t batch_size
);

/**
 * @brief Train single epoch
 * @param bridge Bridge handle
 * @return Epoch loss, -1 on failure
 */
float genius_training_train_epoch(genius_training_bridge_t* bridge);

/**
 * @brief Run validation
 * @param bridge Bridge handle
 * @return Validation accuracy, -1 on failure
 */
float genius_training_validate(genius_training_bridge_t* bridge);

/**
 * @brief Consolidate learned knowledge
 * @param bridge Bridge handle
 * @return 0 on success, -1 on failure
 */
int genius_training_consolidate(genius_training_bridge_t* bridge);

//=============================================================================
// Curriculum Functions
//=============================================================================

/**
 * @brief Get curriculum state
 * @param bridge Bridge handle
 * @param state Output curriculum state
 * @return 0 on success, -1 on failure
 */
int genius_training_get_curriculum_state(
    genius_training_bridge_t* bridge,
    genius_curriculum_state_t* state
);

/**
 * @brief Manually advance curriculum stage
 * @param bridge Bridge handle
 * @return New stage, -1 on failure
 */
int genius_training_advance_curriculum(genius_training_bridge_t* bridge);

/**
 * @brief Set curriculum stage
 * @param bridge Bridge handle
 * @param stage New stage
 * @return 0 on success, -1 on failure
 */
int genius_training_set_curriculum_stage(
    genius_training_bridge_t* bridge,
    genius_curriculum_stage_t stage
);

/**
 * @brief Set training domain
 * @param bridge Bridge handle
 * @param domain New domain
 * @return 0 on success, -1 on failure
 */
int genius_training_set_domain(
    genius_training_bridge_t* bridge,
    genius_train_domain_t domain
);

//=============================================================================
// Learning Rate Functions
//=============================================================================

/**
 * @brief Get current learning rate
 * @param bridge Bridge handle
 * @return Learning rate, -1 on error
 */
float genius_training_get_learning_rate(genius_training_bridge_t* bridge);

/**
 * @brief Set learning rate
 * @param bridge Bridge handle
 * @param lr New learning rate
 * @return 0 on success, -1 on failure
 */
int genius_training_set_learning_rate(
    genius_training_bridge_t* bridge,
    float lr
);

/**
 * @brief Apply learning rate schedule step
 * @param bridge Bridge handle
 * @return New learning rate, -1 on error
 */
float genius_training_lr_step(genius_training_bridge_t* bridge);

//=============================================================================
// State Query Functions
//=============================================================================

/**
 * @brief Get training progress
 * @param bridge Bridge handle
 * @param progress Output progress structure
 * @return 0 on success, -1 on failure
 */
int genius_training_get_progress(
    genius_training_bridge_t* bridge,
    genius_training_progress_t* progress
);

/**
 * @brief Get mode-specific training state
 * @param bridge Bridge handle
 * @param state Output mode training state
 * @return 0 on success, -1 on failure
 */
int genius_training_get_mode_state(
    genius_training_bridge_t* bridge,
    genius_mode_training_state_t* state
);

/**
 * @brief Get bridge state
 * @param bridge Bridge handle
 * @param state Output state structure
 * @return 0 on success, -1 on failure
 */
int genius_training_get_state(
    genius_training_bridge_t* bridge,
    genius_training_bridge_state_t* state
);

/**
 * @brief Get bridge statistics
 * @param bridge Bridge handle
 * @param stats Output statistics structure
 * @return 0 on success, -1 on failure
 */
int genius_training_get_stats(
    genius_training_bridge_t* bridge,
    genius_training_stats_t* stats
);

/**
 * @brief Reset statistics
 * @param bridge Bridge handle
 * @return 0 on success, -1 on failure
 */
int genius_training_reset_stats(genius_training_bridge_t* bridge);

//=============================================================================
// Checkpoint Functions
//=============================================================================

/**
 * @brief Save checkpoint
 * @param bridge Bridge handle
 * @param path Checkpoint file path
 * @return 0 on success, -1 on failure
 */
int genius_training_save_checkpoint(
    genius_training_bridge_t* bridge,
    const char* path
);

/**
 * @brief Load checkpoint
 * @param bridge Bridge handle
 * @param path Checkpoint file path
 * @return 0 on success, -1 on failure
 */
int genius_training_load_checkpoint(
    genius_training_bridge_t* bridge,
    const char* path
);

//=============================================================================
// Callback Registration
//=============================================================================

/**
 * @brief Register epoch completion callback
 * @param bridge Bridge handle
 * @param callback Callback function
 * @param user_data User data pointer
 * @return 0 on success, -1 on failure
 */
int genius_training_register_epoch_callback(
    genius_training_bridge_t* bridge,
    genius_training_epoch_callback_t callback,
    void* user_data
);

/**
 * @brief Register curriculum advancement callback
 * @param bridge Bridge handle
 * @param callback Callback function
 * @param user_data User data pointer
 * @return 0 on success, -1 on failure
 */
int genius_training_register_curriculum_callback(
    genius_training_bridge_t* bridge,
    genius_training_curriculum_callback_t callback,
    void* user_data
);

/**
 * @brief Register checkpoint callback
 * @param bridge Bridge handle
 * @param callback Callback function
 * @param user_data User data pointer
 * @return 0 on success, -1 on failure
 */
int genius_training_register_checkpoint_callback(
    genius_training_bridge_t* bridge,
    genius_training_checkpoint_callback_t callback,
    void* user_data
);

//=============================================================================
// Bio-Async Integration
//=============================================================================

/**
 * @brief Connect to bio-async router
 * @param bridge Bridge handle
 * @return 0 on success, -1 on failure
 */
int genius_training_bio_async_connect(genius_training_bridge_t* bridge);

/**
 * @brief Disconnect from bio-async router
 * @param bridge Bridge handle
 * @return 0 on success, -1 on failure
 */
int genius_training_bio_async_disconnect(genius_training_bridge_t* bridge);

/**
 * @brief Check bio-async connection status
 * @param bridge Bridge handle
 * @return true if connected, false otherwise
 */
bool genius_training_is_bio_async_connected(genius_training_bridge_t* bridge);

#ifdef __cplusplus
}
#endif

#endif /* NIMCP_GENIUS_TRAINING_BRIDGE_H */
