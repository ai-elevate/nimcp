/**
 * @file nimcp.h
 * @brief Unified NIMCP Public API - Single entry point for all language bindings
 * @version 2.6.3
 * @date 2025-11-04
 *
 * This is the ONLY header file that language bindings should include.
 * It provides a consistent, stable API with proper namespacing.
 *
 * Design Goals:
 * - Single namespace (nimcp_* prefix for ALL symbols)
 * - Opaque handles (no struct exposure)
 * - Version stability (semantic versioning)
 * - Clean C API (C99 compatible)
 * - Language binding friendly (simple types, no complex macros)
 */

#ifndef NIMCP_H
#define NIMCP_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "core/brain/learning/nimcp_brain_experience.h"
#include "core/brain/factory/nimcp_brain_innate.h"

/* Deprecation macro for API evolution */
#if defined(__GNUC__) || defined(__clang__)
#define NIMCP_DEPRECATED(msg) __attribute__((deprecated(msg)))
#elif defined(_MSC_VER)
#define NIMCP_DEPRECATED(msg) __declspec(deprecated(msg))
#else
#define NIMCP_DEPRECATED(msg)
#endif

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// Version Information
//=============================================================================

#ifndef NIMCP_VERSION_MAJOR
#define NIMCP_VERSION_MAJOR 2
#endif
#ifndef NIMCP_VERSION_MINOR
#define NIMCP_VERSION_MINOR 6
#endif
#ifndef NIMCP_VERSION_PATCH
#define NIMCP_VERSION_PATCH 3
#endif
#ifndef NIMCP_VERSION_STRING
#define NIMCP_VERSION_STRING "2.6.3"
#endif

/* P2-7: Define NIMCP_MAX_LABEL_SIZE for predict buffer sizing */
#ifndef NIMCP_MAX_LABEL_SIZE
#define NIMCP_MAX_LABEL_SIZE 64
#endif

/* Maximum size for decision explanation buffers */
#ifndef NIMCP_MAX_EXPLANATION_SIZE
#define NIMCP_MAX_EXPLANATION_SIZE 256
#endif

/**
 * @brief Get NIMCP version as string
 * @return Version string (e.g., "2.6.3")
 */
const char* nimcp_version(void);

/**
 * @brief Get NIMCP version as integer (MAJOR * 10000 + MINOR * 100 + PATCH)
 * @return Version integer (e.g., 20601)
 */
int nimcp_version_int(void);

//=============================================================================
// Opaque Handle Types (consistent naming: nimcp_*_t)
//=============================================================================

/**
 * @brief Opaque handle to a brain instance (high-level learning system)
 */
typedef struct nimcp_brain_handle* nimcp_brain_t;

/**
 * @brief Opaque handle to a neural network instance (low-level control)
 */
typedef struct nimcp_network_handle* nimcp_network_t;

/**
 * @brief Opaque handle to an ethics module
 */
typedef struct nimcp_ethics_handle* nimcp_ethics_t;

/**
 * @brief Opaque handle to a knowledge graph
 */
typedef struct nimcp_knowledge_handle* nimcp_knowledge_t;


//=============================================================================
// Enumerations (consistent naming: NIMCP_*)
//=============================================================================

/**
 * @brief Brain size presets
 */
typedef enum {
    NIMCP_BRAIN_TINY = 0,   /**< 100 neurons, <1MB,  ~0.1ms inference */
    NIMCP_BRAIN_SMALL = 1,  /**< 1K neurons,  ~10MB, ~0.5ms inference */
    NIMCP_BRAIN_MEDIUM = 2, /**< 10K neurons, ~50MB, ~5ms inference */
    NIMCP_BRAIN_LARGE = 3   /**< 100K neurons, ~500MB, ~50ms inference */
} nimcp_brain_size_t;

/**
 * @brief Brain task templates
 */
typedef enum {
    NIMCP_TASK_CLASSIFICATION = 0,   /**< Multi-class classification */
    NIMCP_TASK_REGRESSION = 1,       /**< Continuous value prediction */
    NIMCP_TASK_PATTERN_MATCHING = 2, /**< Pattern recognition */
    NIMCP_TASK_SEQUENCE = 3,         /**< Temporal sequence learning */
    NIMCP_TASK_ASSOCIATION = 4       /**< Association learning */
} nimcp_brain_task_t;

/**
 * @brief Network architecture types for training dispatch
 *
 * WHAT: Specifies which neural network architecture to use for training
 * WHY:  Different architectures require different training algorithms:
 *       - ADAPTIVE: Standard backpropagation (default)
 *       - SNN: Spike-timing dependent plasticity, surrogate gradients, eProp
 *       - LNN: Adjoint method for ODE-based liquid neural networks
 *       - CNN: Convolutional backpropagation with layer-wise gradients
 *       - HYBRID: Multiple network types with unified training
 * HOW:  Set in training config, brain dispatches to appropriate trainer
 */
typedef enum {
    NIMCP_NETWORK_ADAPTIVE = 0,  /**< Standard adaptive network (backprop) - default */
    NIMCP_NETWORK_SNN = 1,       /**< Spiking Neural Network (STDP/eProp/surrogate) */
    NIMCP_NETWORK_LNN = 2,       /**< Liquid Neural Network (adjoint ODE) */
    NIMCP_NETWORK_CNN = 3,       /**< Convolutional Neural Network */
    NIMCP_NETWORK_HYBRID = 4     /**< Mixed architecture (multiple network types) */
} nimcp_network_type_t;

/**
 * @brief SNN training method selection
 */
typedef enum {
    NIMCP_SNN_TRAIN_STDP = 0,        /**< Spike-Timing Dependent Plasticity */
    NIMCP_SNN_TRAIN_R_STDP = 1,      /**< Reward-modulated STDP */
    NIMCP_SNN_TRAIN_EPROP = 2,       /**< Eligibility Propagation (online) */
    NIMCP_SNN_TRAIN_SURROGATE = 3,   /**< Surrogate gradient backprop */
    NIMCP_SNN_TRAIN_HOMEOSTATIC = 4  /**< Homeostatic plasticity */
} nimcp_snn_train_method_t;

/**
 * @brief LNN training method selection
 */
typedef enum {
    NIMCP_LNN_TRAIN_ADJOINT = 0,     /**< Adjoint method (memory efficient) - default */
    NIMCP_LNN_TRAIN_BPTT = 1,        /**< Backprop through time */
    NIMCP_LNN_TRAIN_RTRL = 2,        /**< Real-time recurrent learning */
    NIMCP_LNN_TRAIN_EPROP = 3        /**< Eligibility propagation */
} nimcp_lnn_train_method_t;

/**
 * @brief Return codes for all NIMCP functions
 *
 * NIMCP uses a unified positive integer error code system:
 *   - NIMCP_OK/NIMCP_SUCCESS = 0
 *   - Error codes are positive integers (1000+)
 *
 * Note: These undefines prevent conflicts with internal error code macros
 * when this header is included after internal headers.
 */
#undef NIMCP_OK
#undef NIMCP_SUCCESS
#undef NIMCP_ERROR
#undef NIMCP_ERROR_NULL_ARG
#undef NIMCP_ERROR_INVALID
#undef NIMCP_ERROR_MEMORY
#undef NIMCP_ERROR_IO

typedef enum {
    NIMCP_OK = 0,                 /**< Success */
    NIMCP_ERROR = 1000,           /**< Generic error */
    NIMCP_ERROR_NULL_ARG = 1003,  /**< NULL argument provided */
    NIMCP_ERROR_INVALID = 1004,   /**< Invalid argument value */
    NIMCP_ERROR_MEMORY = 2000,    /**< Memory allocation failed */
    NIMCP_ERROR_IO = 4000         /**< I/O operation failed */
} nimcp_status_t;

/* Alias for compatibility */
#define NIMCP_SUCCESS NIMCP_OK

//=============================================================================
// Brain API - High-Level Learning Interface
//=============================================================================

/**
 * @brief Create a brain with preset configuration
 *
 * @param name Human-readable name (e.g., "ethics", "classifier")
 * @param size Brain size preset (NIMCP_BRAIN_TINY to NIMCP_BRAIN_LARGE)
 * @param task Task template (NIMCP_TASK_CLASSIFICATION, etc.)
 * @param num_inputs Input dimension
 * @param num_outputs Output dimension
 * @return Brain handle or NULL on error
 */
nimcp_brain_t nimcp_brain_create(
    const char* name,
    nimcp_brain_size_t size,
    nimcp_brain_task_t task,
    uint32_t num_inputs,
    uint32_t num_outputs
);

/**
 * @brief Create brain with explicit neuron count (bypasses size preset)
 *
 * @param name Human-readable name
 * @param task Task template (NIMCP_TASK_CLASSIFICATION, etc.)
 * @param num_inputs Input dimension
 * @param num_outputs Output dimension
 * @param neuron_count Exact number of hidden neurons
 * @return Brain handle or NULL on error
 */
nimcp_brain_t nimcp_brain_create_with_neurons(
    const char* name,
    nimcp_brain_task_t task,
    uint32_t num_inputs,
    uint32_t num_outputs,
    uint32_t neuron_count
);

/**
 * @brief Create brain with ALL subsystems enabled (RESEARCH profile + extras)
 *
 * Uses the RESEARCH config profile as a base and additionally enables:
 * world model, creative system, LGSS safety, neuromodulators, immune bridges,
 * collective cognition bridges, spike NLP, health agent, cycle coordinator.
 * All lazy initialization is disabled — everything initializes at creation.
 *
 * @param name Brain name
 * @param task Task type
 * @param num_inputs Number of input features
 * @param num_outputs Number of output classes
 * @param neuron_count Number of neurons
 * @return Brain handle, or NULL on failure
 */
nimcp_brain_t nimcp_brain_create_full(
    const char* name,
    nimcp_brain_task_t task,
    uint32_t num_inputs,
    uint32_t num_outputs,
    uint32_t neuron_count
);

/**
 * @brief Create brain with fast initialization (core + training only)
 *
 * Initializes only the subsystems required for training and inference.
 * All other subsystems are deferred to first use (lazy initialization).
 * 3-5x faster than full init for large brains.
 *
 * @param name Brain name
 * @param task Task type
 * @param num_inputs Input dimension
 * @param num_outputs Output dimension
 * @param neuron_count Number of neurons
 * @return Brain handle or NULL on error
 */
nimcp_brain_t nimcp_brain_create_fast(
    const char* name,
    nimcp_brain_task_t task,
    uint32_t num_inputs,
    uint32_t num_outputs,
    uint32_t neuron_count
);

/**
 * @brief Destroy a brain and free all resources
 *
 * @param brain Brain handle
 */
void nimcp_brain_destroy(nimcp_brain_t brain);

/**
 * @brief Learn from a single example
 *
 * @param brain Brain handle
 * @param features Input features array
 * @param num_features Number of input features
 * @param label Output label/class
 * @param confidence Confidence/importance of this example (0.0-1.0)
 * @return NIMCP_OK on success, error code otherwise
 */
nimcp_status_t nimcp_brain_learn_example(
    nimcp_brain_t brain,
    const float* features,
    uint32_t num_features,
    const char* label,
    float confidence
);

/**
 * @brief Learn from a dense target vector (teacher distillation / generative training)
 *
 * Unlike nimcp_brain_learn_example() which converts a string label to one-hot,
 * this function trains toward an arbitrary dense target vector (e.g. a semantic
 * embedding from a teacher model). Uses LEARN_MODE_DISTILLATION internally.
 *
 * @param brain Brain handle
 * @param features Input features array
 * @param num_features Number of input features
 * @param target Dense target output vector
 * @param target_size Size of target vector (must match brain num_outputs)
 * @param label Optional semantic label for tracking (can be NULL)
 * @param confidence Training weight [0.0-1.0]
 * @return NIMCP_OK on success, error code otherwise
 */
nimcp_status_t nimcp_brain_learn_vector(
    nimcp_brain_t brain,
    const float* features,
    uint32_t num_features,
    const float* target,
    uint32_t target_size,
    const char* label,
    float confidence
);

/**
 * @brief Unified experience: perceive input, predict output, and learn
 * @see brain_experience() for detailed documentation
 */
nimcp_status_t nimcp_brain_experience(
    nimcp_brain_t brain,
    const float* input,
    uint32_t input_size,
    float* output,
    uint32_t output_size,
    float teacher_reward,
    brain_experience_result_t* result
);

/**
 * @brief Configure experience-based learning
 */
nimcp_status_t nimcp_brain_experience_configure(
    nimcp_brain_t brain,
    const brain_experience_config_t* config
);

/**
 * @brief Provide a correction after an experience
 * @return Loss value (0=perfect), -1.0 on error
 */
float nimcp_brain_experience_correct(
    nimcp_brain_t brain,
    const float* expected,
    uint32_t expected_size
);

/**
 * @brief Direct attention to a specific modality
 */
nimcp_status_t nimcp_brain_experience_attend(
    nimcp_brain_t brain,
    const char* modality,
    float strength
);

/**
 * @brief Hardwire innate circuits into a brain
 * @see brain_innate_hardwire() for detailed documentation
 */
nimcp_status_t nimcp_brain_innate_hardwire(
    nimcp_brain_t brain,
    const innate_config_t* config
);

/**
 * @brief Learn from a batch of examples
 *
 * @param brain Brain handle
 * @param features_array Array of feature pointers (one per example)
 * @param num_features_array Array of feature counts (one per example, can be NULL if uniform)
 * @param labels Array of label strings (one per example)
 * @param confidences Array of confidence values (one per example, can be NULL for default 1.0)
 * @param num_examples Number of examples in the batch
 * @param losses_out Caller-allocated float[num_examples] for per-example losses (can be NULL)
 * @return NIMCP_OK on success, error code otherwise
 */
nimcp_status_t nimcp_brain_learn_batch(
    nimcp_brain_t brain,
    const float** features_array,
    const uint32_t* num_features_array,
    const char** labels,
    const float* confidences,
    uint32_t num_examples,
    float* losses_out
);

/**
 * @brief Get the loss value from the most recent learn_example call
 *
 * @param brain Brain handle
 * @return Loss value (>= 0.0 on success, -1.0 if no learning has occurred)
 */
float nimcp_brain_get_last_loss(nimcp_brain_t brain);

/**
 * @brief Get the gradient L2 norm from the most recent learn() call
 *
 * @param brain Brain handle
 * @return Gradient norm (>= 0.0), or -1.0 if brain is NULL
 */
float nimcp_brain_get_last_gradient_norm(nimcp_brain_t brain);

/**
 * @brief Get running label-match accuracy (EMA)
 *
 * @param brain Brain handle
 * @return Running accuracy [0.0, 1.0] or 0.0 if brain is NULL
 */
float nimcp_brain_get_accuracy(nimcp_brain_t brain);

/**
 * @brief Make a decision/prediction
 *
 * @param brain Brain handle
 * @param features Input features array
 * @param num_features Number of input features
 * @param out_label Buffer to store predicted label (must be pre-allocated, min 64 bytes)
 * @param out_confidence Pointer to store prediction confidence (0.0-1.0)
 * @return NIMCP_OK on success, error code otherwise
 */
nimcp_status_t nimcp_brain_predict(
    nimcp_brain_t brain,
    const float* features,
    uint32_t num_features,
    char* out_label,
    float* out_confidence
);

/**
 * @brief Fast prediction — forward pass only, no cognitive stages
 *
 * WHAT: Runs neural network forward pass and returns predicted label
 * WHY:  nimcp_brain_predict() runs 28 cognitive stages (sleep, curiosity, theory of mind,
 *        mirror neurons, ethics, etc.) on every call. This function skips all that and
 *        does a pure forward pass, making it 10-100x faster for training loops.
 * HOW:  Calls adaptive_network_forward() directly, then determine_output_label()
 *
 * @param brain Brain handle
 * @param features Input features array
 * @param num_features Number of input features
 * @param out_label Buffer to store predicted label (must be pre-allocated, min 64 bytes)
 * @param out_confidence Pointer to store prediction confidence (0.0-1.0)
 * @return NIMCP_OK on success, error code otherwise
 */
nimcp_status_t nimcp_brain_predict_fast(
    nimcp_brain_t brain,
    const float* features,
    uint32_t num_features,
    char* out_label,
    float* out_confidence
);

/**
 * @brief Domain-scoped fast prediction (forward pass only)
 *
 * WHAT: Same as predict_fast but restricts argmax to output neurons whose label
 *       starts with the given domain prefix (e.g., "biology:" only considers
 *       neurons mapped to biology labels).
 * WHY:  Multi-domain training causes first-mover dominance — domains that train
 *       first develop stronger output activations, causing predict_fast to always
 *       return labels from the dominant domain for inputs with similar features.
 * HOW:  Forward pass + domain-filtered argmax + domain-scoped confidence
 *
 * @param brain Brain handle
 * @param features Input features array
 * @param num_features Number of input features
 * @param domain_prefix Domain prefix to filter by (e.g., "biology:"). If NULL,
 *        behaves identically to nimcp_brain_predict_fast.
 * @param out_label Buffer to store predicted label (must be pre-allocated, min 64 bytes)
 * @param out_confidence Pointer to store prediction confidence (0.0-1.0)
 * @return NIMCP_OK on success, error code otherwise
 */
nimcp_status_t nimcp_brain_predict_in_domain(
    nimcp_brain_t brain,
    const float* features,
    uint32_t num_features,
    const char* domain_prefix,
    char* out_label,
    float* out_confidence
);

/**
 * @brief Run inference and get raw output vector
 *
 * WHAT: Forward pass through network returning raw outputs
 * WHY:  For numeric predictions, embeddings, or when you don't need label classification
 * HOW:  Processes inputs through network, returns output activations
 *
 * @param brain Brain handle
 * @param features Input features array
 * @param num_features Number of input features
 * @param outputs Output array to fill (must be pre-allocated)
 * @param num_outputs Number of outputs (must match brain's output dimension)
 * @return NIMCP_OK on success, error code otherwise
 *
 * EXAMPLE:
 * ```c
 * float features[20] = {...};
 * float outputs[10];
 * nimcp_status_t result = nimcp_brain_infer(brain, features, 20, outputs, 10);
 * ```
 */
nimcp_status_t nimcp_brain_infer(
    nimcp_brain_t brain,
    const float* features,
    uint32_t num_features,
    float* outputs,
    uint32_t num_outputs
);

/**
 * @brief Run full cognitive pipeline and return complete decision data
 *
 * Unlike nimcp_brain_predict() which only returns label+confidence,
 * this function returns the full brain_decision_t contents including
 * explanation text, output vector, active neuron count, and sparsity.
 *
 * @param brain Brain handle
 * @param features Input feature array
 * @param num_features Number of features
 * @param out_label Buffer for predicted label (min 64 bytes)
 * @param out_confidence Pointer to store confidence
 * @param out_explanation Buffer for explanation text (min 256 bytes)
 * @param out_output_vector Buffer for raw output vector
 * @param out_output_size In: buffer capacity, Out: actual output size
 * @param out_num_active_neurons Pointer to store active neuron count
 * @param out_sparsity Pointer to store sparsity value
 * @param out_inference_time_us Pointer to store inference time in microseconds
 * @return NIMCP_OK on success, error code otherwise
 */
nimcp_status_t nimcp_brain_decide_full(
    nimcp_brain_t brain,
    const float* features, uint32_t num_features,
    char* out_label, float* out_confidence,
    char* out_explanation,
    float* out_output_vector, uint32_t* out_output_size,
    uint32_t* out_num_active_neurons, float* out_sparsity,
    uint64_t* out_inference_time_us
);

/**
 * @brief Generate spoken text from brain's neural state
 *
 * Routes semantic representation through Broca's language production
 * pipeline (lexical selection → syntactic encoding → text output).
 *
 * @param brain Brain handle
 * @param semantic_input Semantic vector (brain output), or NULL to use last decision
 * @param semantic_dim Dimension of semantic_input (ignored if NULL)
 * @param out_text Output text buffer for generated utterance
 * @param text_max_len Maximum length of out_text buffer
 * @param out_confidence Pointer to store production confidence [0,1] (optional)
 * @param out_fluency Pointer to store fluency score [0,1] (optional)
 * @return NIMCP_OK on success, error code otherwise
 */
nimcp_status_t nimcp_brain_speak(
    nimcp_brain_t brain,
    const float* semantic_input,
    uint32_t semantic_dim,
    char* out_text,
    uint32_t text_max_len,
    float* out_confidence,
    float* out_fluency
);

/**
 * @brief Connect two brains through collective cognition
 *
 * Links brain_b into brain_a's collective cognition system so they
 * share consciousness metrics, hyperscanning sync, and load balancing.
 * Both brains must have collective cognition enabled.
 *
 * @param brain_a First brain (host collective)
 * @param brain_b Second brain (joining collective)
 * @param instance_id Unique ID for brain_b in the collective
 * @return NIMCP_OK on success, error code otherwise
 */
nimcp_status_t nimcp_brain_connect_collective(
    nimcp_brain_t brain_a,
    nimcp_brain_t brain_b,
    uint32_t instance_id
);

/**
 * @brief Avatar face state — visemes, FACS action units, emotion, gaze
 *
 * Collected from Broca's speech motor planner, emotional system, and
 * attention system. Drives a parameterized face mesh for synchronized
 * audio-visual communication.
 */
typedef struct {
    /* Viseme / mouth shape (from speech motor articulators) */
    float mouth_open;           /**< Jaw opening [0=closed, 1=wide open] */
    float lip_round;            /**< Lip rounding [0=spread, 1=rounded] */
    float lip_upper;            /**< Upper lip raise [0,1] */
    float lip_lower;            /**< Lower lip drop [0,1] */
    float tongue_position;      /**< Tongue front-back [0=front, 1=back] */
    uint8_t current_viseme;     /**< Active viseme ID (0-21, Preston Blair) */

    /* FACS Action Units (Ekman) — primary face muscles */
    float au1_inner_brow_raise; /**< AU1: Inner brow raiser (frontalis medial) */
    float au2_outer_brow_raise; /**< AU2: Outer brow raiser (frontalis lateral) */
    float au4_brow_lower;       /**< AU4: Brow lowerer (corrugator) */
    float au5_upper_lid_raise;  /**< AU5: Upper lid raiser (levator) */
    float au6_cheek_raise;      /**< AU6: Cheek raiser (orbicularis oculi) */
    float au7_lid_tighten;      /**< AU7: Lid tightener */
    float au9_nose_wrinkle;     /**< AU9: Nose wrinkler (levator labii) */
    float au10_upper_lip_raise; /**< AU10: Upper lip raiser */
    float au12_lip_corner_pull; /**< AU12: Lip corner puller (zygomatic) — SMILE */
    float au15_lip_corner_drop; /**< AU15: Lip corner depressor — FROWN */
    float au17_chin_raise;      /**< AU17: Chin raiser (mentalis) */
    float au20_lip_stretch;     /**< AU20: Lip stretcher (risorius) */
    float au23_lip_tighten;     /**< AU23: Lip tightener (orbicularis oris) */
    float au25_lips_part;       /**< AU25: Lips part */
    float au26_jaw_drop;        /**< AU26: Jaw drop (masseter relax) */
    float au28_lip_suck;        /**< AU28: Lip suck */

    /* Emotional state */
    float valence;              /**< Emotional valence [-1=negative, +1=positive] */
    float arousal;              /**< Emotional arousal [0=calm, 1=excited] */
    float dominance;            /**< Dominance [-1=submissive, +1=dominant] */
    uint32_t emotion_id;        /**< Primary emotion (0-18, emotion_category_t) */
    float emotion_intensity;    /**< Emotion intensity [0,1] */

    /* Gaze and head pose */
    float gaze_x;               /**< Horizontal gaze [-1=left, +1=right] */
    float gaze_y;               /**< Vertical gaze [-1=down, +1=up] */
    float head_pitch;           /**< Head pitch (nod) [-1=down, +1=up] */
    float head_yaw;             /**< Head yaw (turn) [-1=left, +1=right] */
    float head_roll;            /**< Head roll (tilt) [-1=left, +1=right] */
    float blink;                /**< Blink state [0=open, 1=closed] */

    /* Voice parameters (for TTS prosody sync) */
    float pitch_hz;             /**< Voice pitch (Hz) */
    float speaking_rate;        /**< Speaking rate multiplier (1.0=normal) */
    float volume;               /**< Voice volume [0,1] */

    /* Metadata */
    uint64_t timestamp_us;      /**< Timestamp in microseconds */
    bool is_speaking;           /**< Whether currently producing speech */
} nimcp_avatar_state_t;

/**
 * @brief Get current avatar face state for rendering
 *
 * Collects viseme data from speech motor planner, FACS action units from
 * emotional system, and gaze from attention. Call after decide_full() or
 * speak() to get synchronized state.
 *
 * @param brain Brain handle
 * @param state Output avatar state
 * @return NIMCP_OK on success, error code otherwise
 */
nimcp_status_t nimcp_brain_get_avatar_state(
    nimcp_brain_t brain,
    nimcp_avatar_state_t* state
);

/**
 * @brief Save brain to file
 *
 * @param brain Brain handle
 * @param filepath Path to save file
 * @return NIMCP_OK on success, error code otherwise
 */
nimcp_status_t nimcp_brain_save(nimcp_brain_t brain, const char* filepath);

/**
 * @brief Freeze brain for inference-only mode
 *
 * Disables learning, locks weights, and optimizes for fast inference.
 * After freezing: learning rate = 0, GPU weight cache is finalized,
 * eligibility traces freed, weight dirty flag cleared.
 *
 * @param brain Brain handle
 * @return NIMCP_OK on success, error code otherwise
 */
nimcp_status_t nimcp_brain_freeze(nimcp_brain_t brain);

/**
 * @brief Check if brain is frozen
 *
 * @param brain Brain handle
 * @return true if brain is in frozen inference-only mode
 */
bool nimcp_brain_is_frozen(nimcp_brain_t brain);

/**
 * @brief Load brain from file
 *
 * @param filepath Path to saved brain file
 * @return Brain handle or NULL on error
 */
nimcp_brain_t nimcp_brain_load(const char* filepath);

//=============================================================================
// Training Pipeline API - Loss Functions, Optimizers, LR Schedulers
//=============================================================================

/**
 * @brief Loss function types for training (public API)
 * @note Uses distinct names to avoid conflicts with internal training module types
 */
typedef enum {
    NIMCP_API_LOSS_MSE = 0,              /**< Mean Squared Error (regression) */
    NIMCP_API_LOSS_CROSS_ENTROPY = 1,    /**< Cross-Entropy (classification) */
    NIMCP_API_LOSS_BINARY_CE = 2,        /**< Binary Cross-Entropy */
    NIMCP_API_LOSS_HUBER = 3,            /**< Huber Loss (robust regression) */
    NIMCP_API_LOSS_MAE = 4,              /**< Mean Absolute Error */
    NIMCP_API_LOSS_FOCAL = 5,            /**< Focal Loss (imbalanced classes) */
    NIMCP_API_LOSS_KL_DIV = 6            /**< KL Divergence */
} nimcp_api_loss_t;

/**
 * @brief Optimizer types for training (public API)
 */
typedef enum {
    NIMCP_API_OPT_SGD = 0,               /**< Stochastic Gradient Descent */
    NIMCP_API_OPT_MOMENTUM = 1,          /**< SGD with Momentum */
    NIMCP_API_OPT_ADAM = 2,              /**< Adam optimizer */
    NIMCP_API_OPT_ADAMW = 3,             /**< AdamW (weight decay) */
    NIMCP_API_OPT_RMSPROP = 4,           /**< RMSprop */
    NIMCP_API_OPT_ADAGRAD = 5            /**< Adagrad */
} nimcp_api_optimizer_t;

/**
 * @brief Learning rate scheduler types (public API)
 */
typedef enum {
    NIMCP_API_SCHED_CONSTANT = 0,           /**< Constant learning rate */
    NIMCP_API_SCHED_STEP = 1,               /**< Step decay */
    NIMCP_API_SCHED_EXPONENTIAL = 2,        /**< Exponential decay */
    NIMCP_API_SCHED_COSINE = 3,             /**< Cosine annealing */
    NIMCP_API_SCHED_WARMUP_COSINE = 4,      /**< Warmup + Cosine annealing */
    NIMCP_API_SCHED_REDUCE_ON_PLATEAU = 5,  /**< Reduce when metric plateaus */
    NIMCP_API_SCHED_CYCLIC = 6              /**< Cyclic learning rate */
} nimcp_api_scheduler_t;

/**
 * @brief Training configuration for the training pipeline
 */
typedef struct {
    nimcp_api_loss_t loss_type;           /**< Loss function type */
    nimcp_api_optimizer_t optimizer_type; /**< Optimizer type */
    nimcp_api_scheduler_t scheduler_type; /**< LR scheduler type */

    float learning_rate;       /**< Initial learning rate (default: 0.001) */
    float weight_decay;        /**< L2 regularization (default: 0.0) */
    float momentum;            /**< Momentum for SGD/Momentum (default: 0.9) */
    float beta1;               /**< Adam beta1 (default: 0.9) */
    float beta2;               /**< Adam beta2 (default: 0.999) */
    float epsilon;             /**< Adam epsilon (default: 1e-8) */

    uint32_t scheduler_step_size;   /**< Steps between LR updates */
    float scheduler_gamma;          /**< LR decay factor (default: 0.1) */
    uint32_t warmup_steps;          /**< Warmup steps (default: 0) */

    bool enable_gradient_clipping;  /**< Enable gradient clipping */
    float gradient_clip_value;      /**< Max gradient norm (default: 1.0) */

    bool enable_biological_modulation; /**< Enable plasticity bridge modulation */
    float biological_blend;            /**< Biological modulation strength (0-1) */

    // === NETWORK TYPE DISPATCH (NEW) ===
    nimcp_network_type_t network_type;     /**< Network architecture type (default: ADAPTIVE) */

    // SNN-specific training options (used when network_type == NIMCP_NETWORK_SNN)
    nimcp_snn_train_method_t snn_method;   /**< SNN training method (default: STDP) */
    float snn_eligibility_tau;             /**< Eligibility trace decay (ms, default: 20.0) */
    float snn_reward_tau;                  /**< Reward signal decay (ms, default: 100.0) */
    float snn_surrogate_beta;              /**< Surrogate gradient steepness (default: 5.0) */

    // LNN-specific training options (used when network_type == NIMCP_NETWORK_LNN)
    nimcp_lnn_train_method_t lnn_method;   /**< LNN training method (default: ADJOINT) */
    uint32_t lnn_bptt_truncation;          /**< BPTT truncation length (default: 100) */
    bool lnn_use_adjoint_checkpointing;    /**< Memory-efficient checkpointing (default: true) */

    /* Rubric integration (cognitive quality monitoring during training) */
    bool enable_rubric;                /**< Enable periodic rubric evaluation (default: false) */
    uint32_t rubric_interval;          /**< Steps between evaluations (0 = epoch-only, default: 0) */
    float rubric_min_score;            /**< Minimum acceptable score [0-1] (0.0 = no threshold) */
    bool rubric_stop_on_threshold;     /**< Stop training if score < min (default: false) */
} nimcp_training_config_t;

/**
 * @brief Training result/statistics from a training step
 */
typedef struct {
    float loss;                /**< Loss value */
    float learning_rate;       /**< Current learning rate */
    uint32_t step;             /**< Training step number */
    bool early_stopped;        /**< True if early stopping triggered */
    float gradient_norm;       /**< Gradient norm (if clipping enabled) */

    /* Rubric (populated only when evaluated this step) */
    bool rubric_evaluated;             /**< True if rubric was evaluated this step */
    float rubric_score;                /**< Overall rubric score [0-1] */
    char rubric_grade;                 /**< Letter grade A/B/C/D/F */
    char rubric_grade_modifier;        /**< +/-/space */
} nimcp_training_result_t;

/**
 * @brief Get default training configuration
 *
 * Returns a sensible default configuration:
 * - Loss: Cross-Entropy
 * - Optimizer: Adam (lr=0.001, betas=0.9/0.999)
 * - Scheduler: Cosine annealing
 * - Biological modulation: enabled at 50%
 *
 * @return Default training configuration
 */
nimcp_training_config_t nimcp_training_config_default(void);

/**
 * @brief Configure brain's training pipeline
 *
 * Sets up the internal training coordinator with specified loss function,
 * optimizer, and learning rate scheduler. Must be called before using
 * nimcp_brain_train_step() or nimcp_brain_train_batch().
 *
 * @param brain Brain handle
 * @param config Training configuration
 * @return NIMCP_OK on success, error code otherwise
 *
 * EXAMPLE:
 * ```c
 * nimcp_training_config_t config = nimcp_training_config_default();
 * config.loss_type = NIMCP_API_LOSS_CROSS_ENTROPY;
 * config.optimizer_type = NIMCP_API_OPT_ADAM;
 * config.learning_rate = 0.001f;
 * nimcp_brain_configure_training(brain, &config);
 * ```
 */
nimcp_status_t nimcp_brain_configure_training(
    nimcp_brain_t brain,
    const nimcp_training_config_t* config
);

/**
 * @brief Train brain using the training pipeline (single step)
 *
 * Performs a complete training step using the configured training pipeline:
 * 1. Forward pass to get predictions
 * 2. Loss computation (via Loss Functions module)
 * 3. Gradient computation (backpropagation)
 * 4. Gradient health check (via Gradient Manager)
 * 5. Regularization (if enabled)
 * 6. Biological modulation (via Plasticity Bridge, if enabled)
 * 7. Weight update (via Optimizer)
 * 8. Learning rate update (via LR Scheduler)
 *
 * This uses the full training coordinator with all integrated modules.
 *
 * @param brain Brain handle
 * @param features Input features array
 * @param num_features Number of input features
 * @param targets Target output array (one-hot or continuous)
 * @param num_targets Number of target outputs
 * @param result Output: training result (can be NULL)
 * @return NIMCP_OK on success, error code otherwise
 *
 * EXAMPLE:
 * ```c
 * float features[784] = {...};
 * float targets[10] = {0,0,0,1,0,0,0,0,0,0};  // One-hot for class 3
 * nimcp_training_result_t result;
 * nimcp_brain_train_step(brain, features, 784, targets, 10, &result);
 * printf("Loss: %.4f, LR: %.6f\n", result.loss, result.learning_rate);
 * ```
 */
nimcp_status_t nimcp_brain_train_step(
    nimcp_brain_t brain,
    const float* features,
    uint32_t num_features,
    const float* targets,
    uint32_t num_targets,
    nimcp_training_result_t* result
);

/**
 * @brief Train brain on a batch of examples
 *
 * Performs training on multiple examples, accumulating gradients before
 * applying weight updates (mini-batch gradient descent).
 *
 * @param brain Brain handle
 * @param features 2D array of features [batch_size][num_features]
 * @param targets 2D array of targets [batch_size][num_targets]
 * @param batch_size Number of examples in batch
 * @param num_features Features per example
 * @param num_targets Targets per example
 * @param result Output: training result with averaged loss (can be NULL)
 * @return NIMCP_OK on success, error code otherwise
 *
 * EXAMPLE:
 * ```c
 * float features[32][784];  // 32 images, 784 pixels each
 * float targets[32][10];    // 32 one-hot labels
 * nimcp_training_result_t result;
 * nimcp_brain_train_batch(brain, (float*)features, (float*)targets,
 *                          32, 784, 10, &result);
 * ```
 */
nimcp_status_t nimcp_brain_train_batch(
    nimcp_brain_t brain,
    const float* features,
    const float* targets,
    uint32_t batch_size,
    uint32_t num_features,
    uint32_t num_targets,
    nimcp_training_result_t* result
);

/**
 * @brief Get current training statistics
 *
 * @param brain Brain handle
 * @param total_steps Output: total training steps
 * @param total_loss Output: cumulative loss
 * @param current_lr Output: current learning rate
 * @return NIMCP_OK on success, error code otherwise
 */
nimcp_status_t nimcp_brain_get_training_stats(
    nimcp_brain_t brain,
    uint64_t* total_steps,
    float* total_loss,
    float* current_lr
);

/**
 * @brief Step the learning rate scheduler (call at epoch end)
 *
 * Updates the learning rate according to the configured scheduler.
 * For ReduceOnPlateau, provide the validation metric.
 *
 * @param brain Brain handle
 * @param validation_metric Validation metric (for ReduceOnPlateau, ignored otherwise)
 * @return New learning rate
 */
float nimcp_brain_step_scheduler(nimcp_brain_t brain, float validation_metric);

//=============================================================================
// Training Callbacks API - Event-driven training monitoring and control
//=============================================================================

/**
 * @brief Training callback event types (public API)
 */
typedef enum {
    NIMCP_CB_STEP_COMPLETE = 0,    /**< Training step finished */
    NIMCP_CB_EPOCH_COMPLETE,        /**< Epoch finished */
    NIMCP_CB_LOSS_COMPUTED,         /**< Loss calculated */
    NIMCP_CB_WEIGHTS_UPDATED,       /**< Weights modified */
    NIMCP_CB_LR_CHANGED,            /**< Learning rate changed */
    NIMCP_CB_CONVERGENCE,           /**< Early stopping triggered */
    NIMCP_CB_DIVERGENCE,            /**< Training instability */
    NIMCP_CB_CHECKPOINT,            /**< Checkpoint saved */
    NIMCP_CB_EVENT_COUNT
} nimcp_callback_event_t;

/**
 * @brief Callback return actions
 */
typedef enum {
    NIMCP_CB_ACTION_CONTINUE = 0,   /**< Continue training normally */
    NIMCP_CB_ACTION_STOP,           /**< Stop training loop */
    NIMCP_CB_ACTION_SKIP,           /**< Skip current step */
    NIMCP_CB_ACTION_ROLLBACK,       /**< Rollback to checkpoint */
    NIMCP_CB_ACTION_REDUCE_LR,      /**< Reduce learning rate */
    NIMCP_CB_ACTION_INCREASE_LR     /**< Increase learning rate */
} nimcp_callback_action_t;

/**
 * @brief Training metrics passed to callbacks
 */
typedef struct {
    uint64_t step;                  /**< Current training step */
    uint64_t epoch;                 /**< Current epoch */
    float loss;                     /**< Current loss value */
    float loss_ema;                 /**< Exponential moving average of loss */
    float learning_rate;            /**< Current learning rate */
    float gradient_norm;            /**< L2 norm of gradients */
    uint64_t step_time_us;          /**< Time for current step */
    bool is_converging;             /**< Loss trending down */
    bool is_diverging;              /**< Loss trending up rapidly */
} nimcp_callback_metrics_t;

/**
 * @brief Callback function signature
 *
 * @param event Event type that triggered callback
 * @param metrics Current training metrics
 * @param user_data User-provided context
 * @return Action for training loop to take
 */
typedef nimcp_callback_action_t (*nimcp_training_callback_fn)(
    nimcp_callback_event_t event,
    const nimcp_callback_metrics_t* metrics,
    void* user_data
);

/**
 * @brief Callback configuration for registration
 */
typedef struct {
    bool enable_auto_checkpoint;     /**< Enable automatic checkpointing */
    uint32_t checkpoint_interval;    /**< Steps between checkpoints */
    bool enable_early_stopping;      /**< Enable early stopping */
    uint32_t patience;               /**< Steps without improvement before stop */
    float min_delta;                 /**< Minimum improvement to reset patience */
    float divergence_threshold;      /**< Loss increase ratio for divergence */
    uint32_t log_interval;           /**< Steps between log outputs (0=disabled) */
} nimcp_callback_config_t;

/**
 * @brief Get default callback configuration
 *
 * @return Default configuration with sensible values
 */
nimcp_callback_config_t nimcp_callback_config_default(void);

/**
 * @brief Enable training callbacks for a brain
 *
 * Must be called after nimcp_brain_configure_training().
 * Enables the callback system with the specified configuration.
 *
 * @param brain Brain handle
 * @param config Callback configuration (NULL for defaults)
 * @return NIMCP_OK on success, error code otherwise
 */
nimcp_status_t nimcp_brain_enable_callbacks(
    nimcp_brain_t brain,
    const nimcp_callback_config_t* config
);

/**
 * @brief Disable training callbacks for a brain
 *
 * @param brain Brain handle
 * @return NIMCP_OK on success, error code otherwise
 */
nimcp_status_t nimcp_brain_disable_callbacks(nimcp_brain_t brain);

/**
 * @brief Register a callback for a specific event type
 *
 * @param brain Brain handle
 * @param event Event type to subscribe to
 * @param callback Callback function
 * @param user_data User context passed to callback
 * @param name Callback name for logging (can be NULL)
 * @return Callback ID (>0) on success, 0 on error
 *
 * EXAMPLE:
 * ```c
 * nimcp_callback_action_t my_logger(
 *     nimcp_callback_event_t event,
 *     const nimcp_callback_metrics_t* m,
 *     void* user_data)
 * {
 *     printf("Step %lu: loss=%.4f lr=%.6f\n", m->step, m->loss, m->learning_rate);
 *     return NIMCP_CB_ACTION_CONTINUE;
 * }
 *
 * uint32_t cb_id = nimcp_brain_register_callback(
 *     brain, NIMCP_CB_STEP_COMPLETE, my_logger, NULL, "logger");
 * ```
 */
uint32_t nimcp_brain_register_callback(
    nimcp_brain_t brain,
    nimcp_callback_event_t event,
    nimcp_training_callback_fn callback,
    void* user_data,
    const char* name
);

/**
 * @brief Unregister a callback
 *
 * @param brain Brain handle
 * @param callback_id Callback ID from registration
 * @return NIMCP_OK on success, error code otherwise
 */
nimcp_status_t nimcp_brain_unregister_callback(
    nimcp_brain_t brain,
    uint32_t callback_id
);

/**
 * @brief Get callback statistics
 *
 * @param brain Brain handle
 * @param total_fired Output: total callbacks fired
 * @param avg_time_us Output: average callback execution time
 * @param early_stops Output: early stops triggered
 * @return NIMCP_OK on success, error code otherwise
 */
nimcp_status_t nimcp_brain_get_callback_stats(
    nimcp_brain_t brain,
    uint64_t* total_fired,
    float* avg_time_us,
    uint32_t* early_stops
);

//=============================================================================
// Phase 2.8: Dynamic Brain Resizing
//=============================================================================

/**
 * @brief Manually resize brain to a specific neuron count
 *
 * @param brain Brain handle
 * @param new_neuron_count Target number of neurons
 * @return true on success, false on failure
 */
bool nimcp_brain_resize(nimcp_brain_t brain, uint32_t new_neuron_count);

/**
 * @brief Automatically resize brain based on hardware capabilities and utilization
 *
 * @param brain Brain handle
 * @return true if resized, false if no resize needed or on error
 */
bool nimcp_brain_auto_resize(nimcp_brain_t brain);

/**
 * @brief Get current neuron count
 *
 * @param brain Brain handle
 * @return Current number of neurons, or 0 on error
 */
uint32_t nimcp_brain_get_neuron_count(nimcp_brain_t brain);

/**
 * @brief Get brain utilization metrics
 *
 * @param brain Brain handle
 * @param utilization Output: percentage of neurons being utilized (0.0-1.0)
 * @param saturation Output: percentage of neurons at capacity (0.0-1.0)
 * @return true on success, false on error
 */
bool nimcp_brain_get_utilization_metrics(nimcp_brain_t brain, float* utilization, float* saturation);

//=============================================================================
// Brain Snapshots - Named, timestamped backups for versioning & A/B testing
//=============================================================================

/**
 * @brief Snapshot metadata information
 */
typedef struct {
    char name[128];           /**< Snapshot name */
    char description[512];    /**< User description */
    uint64_t timestamp;       /**< Unix timestamp when snapshot was created */
    uint32_t file_size;       /**< Size of snapshot file in bytes */
    bool is_compressed;       /**< Whether snapshot is compressed */
    bool is_encrypted;        /**< Whether snapshot is encrypted */
} nimcp_brain_snapshot_info_t;

/**
 * @brief Save a named snapshot of the brain state
 *
 * Snapshots are different from checkpoints:
 * - Checkpoints: Auto-saved to a single file for resumption
 * - Snapshots: Named, timestamped backups for versioning/backup/A/B testing
 *
 * Example usage:
 * ```c
 * nimcp_brain_snapshot_save(brain, "before_training", "Baseline state");
 * // Train the model...
 * nimcp_brain_snapshot_save(brain, "after_epoch_1", "After 1 epoch");
 * // More training...
 * nimcp_brain_snapshot_save(brain, "final", "Production model");
 * ```
 *
 * Snapshots are saved to: {snapshot_dir}/{name}_{timestamp}.snapshot
 *
 * @param brain Brain to snapshot
 * @param name Snapshot name (no path, just name)
 * @param description Optional description (can be NULL)
 * @return NIMCP_SUCCESS on success, error code otherwise
 */
nimcp_status_t nimcp_brain_snapshot_save(
    nimcp_brain_t brain,
    const char* name,
    const char* description
);

/**
 * @brief Restore brain from a named snapshot
 *
 * @param brain Current brain (can be NULL to create new brain from snapshot)
 * @param name Snapshot name or full path to snapshot file
 * @return New brain instance restored from snapshot, or NULL on error
 */
nimcp_brain_t nimcp_brain_snapshot_restore(
    nimcp_brain_t brain,
    const char* name
);

/**
 * @brief List all available snapshots
 *
 * @param brain Brain instance (to get snapshot directory)
 * @param infos Array to store snapshot information
 * @param max_count Maximum number of snapshots to list
 * @param out_count Pointer to store actual count (can be NULL)
 * @return NIMCP_SUCCESS on success, error code otherwise
 */
nimcp_status_t nimcp_brain_snapshot_list(
    nimcp_brain_t brain,
    nimcp_brain_snapshot_info_t* infos,
    uint32_t max_count,
    uint32_t* out_count
);

/**
 * @brief Delete a named snapshot
 *
 * @param brain Brain instance (to get snapshot directory)
 * @param name Snapshot name to delete
 * @return NIMCP_SUCCESS on success, error code otherwise
 */
nimcp_status_t nimcp_brain_snapshot_delete(
    nimcp_brain_t brain,
    const char* name
);

/**
 * @brief Create brain from YAML or JSON configuration file
 *
 * Supports loading brain configuration from YAML (.yaml, .yml) or JSON (.json) files.
 * Configuration includes architecture, training parameters, plasticity settings, and ethics.
 *
 * Example YAML config:
 * ```yaml
 * brain:
 *   name: "classifier"
 *   size: small           # tiny, small, medium, large
 *   task: classification  # classification, regression, pattern_matching, sequence, association
 *   architecture:
 *     num_inputs: 784
 *     num_outputs: 10
 *     num_hidden: 256
 *     learning_rate: 0.01
 * ```
 *
 * @param config_filepath Path to YAML or JSON configuration file
 * @return Brain handle or NULL on error
 */
nimcp_brain_t nimcp_brain_create_from_config(const char* config_filepath);

/**
 * @brief Brain probe statistics - comprehensive snapshot of brain state
 */
typedef struct {
    char task_name[64];           /**< Brain name */
    nimcp_brain_size_t size;      /**< Size preset */
    nimcp_brain_task_t task;      /**< Task type */
    uint32_t num_neurons;         /**< Total neurons */
    uint32_t num_synapses;        /**< Total synapses */
    uint32_t num_active_synapses; /**< Non-pruned synapses */

    uint64_t total_inferences;     /**< Total inference count */
    uint64_t total_learning_steps; /**< Total learning steps */

    float avg_sparsity;          /**< Average sparsity (0.0-1.0) */
    float avg_inference_time_us; /**< Average inference time (microseconds) */
    float current_learning_rate; /**< Current learning rate */

    float accuracy;      /**< Validation accuracy (0.0-1.0) */
    size_t memory_bytes; /**< Memory usage in bytes */

    uint32_t num_inputs;  /**< Number of inputs */
    uint32_t num_outputs; /**< Number of outputs */

    // Copy-on-Write (COW) cache statistics
    bool is_cow_clone;          /**< True if this brain is a COW clone */
    uint32_t cow_ref_count;     /**< Reference count for shared data (0 if not COW) */
    size_t cow_shared_bytes;    /**< Bytes shared via COW (0 if not COW) */
    size_t cow_private_bytes;   /**< Bytes private to this brain (always > 0) */

    // GPU status
    bool gpu_available;         /**< True if GPU acceleration is active */
} nimcp_brain_probe_t;

/**
 * @brief Probe brain state - get comprehensive statistics
 *
 * This function provides a snapshot of the brain's current state including
 * architecture, performance metrics, and resource usage. Similar to a network
 * probe, it returns all relevant information for monitoring and debugging.
 *
 * @param brain Brain handle
 * @param probe Output structure to fill with brain statistics
 * @return NIMCP_OK on success, error code otherwise
 */
nimcp_status_t nimcp_brain_probe(nimcp_brain_t brain, nimcp_brain_probe_t* probe);

/**
 * @brief Broadcast brain probe data via bio-async message system
 *
 * WHAT: Sends brain probe metrics to all interested subscribers via bio-async
 * WHY:  Enables loose coupling - metrics module receives data without direct dependency
 * HOW:  Fills bio_msg_brain_probe_data_t and broadcasts via BIO_MSG_BRAIN_PROBE_DATA
 *
 * This function allows multiple brains to be monitored concurrently. Each brain
 * broadcasts its metrics using its unique brain_id (pointer address), allowing
 * subscribers to differentiate between different brain instances.
 *
 * @param brain Brain handle to probe and broadcast
 * @return NIMCP_OK on success, error code otherwise
 */
nimcp_status_t nimcp_brain_broadcast_probe(nimcp_brain_t brain);

//=============================================================================
// Brain Health Probes - System-Level Metrics Collection
//=============================================================================

/**
 * @brief Immune system metrics snapshot
 *
 * WHAT: Aggregated immune system health metrics for external monitoring
 * WHY:  Enables health dashboards to track immune activity without
 *       exposing internal immune system structures
 * HOW:  Reads from brain_immune_system_t stats and continuous inflammation
 */
typedef struct {
    uint32_t total_exceptions;       /**< Total exceptions processed by immune system */
    uint32_t recovered_exceptions;   /**< Exceptions successfully recovered */
    float inflammation_level;        /**< Continuous inflammation [0.0-1.0] */
    uint32_t active_antibodies;      /**< Number of active antibody responses */
} nimcp_immune_metrics_t;

/**
 * @brief Synapse statistics snapshot
 *
 * WHAT: Current synapse count and growth delta since last probe call
 * WHY:  Tracks network connectivity growth over time
 * HOW:  Reads total synapses from brain_stats, computes delta from last call
 */
typedef struct {
    uint64_t total_synapses;         /**< Current total synapse count */
    int64_t  growth_since_last;      /**< Change since previous call (can be negative if pruned) */
} nimcp_synapse_stats_t;

/**
 * @brief Get process resident set size (RSS) in bytes
 *
 * WHAT: Returns the current process RSS from /proc/self/status (Linux)
 * WHY:  Monitors memory pressure without requiring external tools
 * HOW:  Parses VmRSS line from /proc/self/status; returns 0 on non-Linux
 *
 * @return Process RSS in bytes, or 0 if unavailable
 */
size_t nimcp_brain_get_memory_rss(void);

/**
 * @brief Get approximate GPU VRAM in use by this brain
 *
 * WHAT: Returns tracked GPU memory allocation from the brain's GPU context
 * WHY:  Enables GPU memory pressure monitoring for health dashboards
 * HOW:  Reads allocated_memory from brain's nimcp_gpu_context_t
 *
 * @param brain Brain handle
 * @return GPU VRAM bytes in use, or 0 if no GPU or brain is NULL
 */
size_t nimcp_brain_get_gpu_vram_used(nimcp_brain_t brain);

/**
 * @brief Get fraction of neurons with at least one outgoing synapse
 *
 * WHAT: Measures neuron utilization as connected_neurons / total_neurons
 * WHY:  Low utilization indicates many dead/orphan neurons (wasted capacity)
 * HOW:  Samples every Nth neuron (N = total/15000) via NEURON_OUT_COUNT
 *
 * @param brain Brain handle
 * @return Fraction [0.0-1.0], or 0.0 on error
 */
float nimcp_brain_get_neuron_utilization(nimcp_brain_t brain);

/**
 * @brief Get immune system metrics for the brain
 *
 * WHAT: Fills output struct with immune activity, inflammation, and recovery stats
 * WHY:  Enables external dashboards to monitor brain immune health
 * HOW:  Calls brain_immune_get_stats() on the brain's immune system
 *
 * @param brain Brain handle
 * @param out   Output structure (zeroed on error)
 * @return NIMCP_OK on success, error code otherwise
 */
nimcp_status_t nimcp_brain_get_immune_metrics(nimcp_brain_t brain, nimcp_immune_metrics_t* out);

/**
 * @brief Get synapse count and growth since last probe call
 *
 * WHAT: Returns current synapse total and delta from previous invocation
 * WHY:  Tracks network growth rate for adaptive training decisions
 * HOW:  Reads num_synapses from brain stats; stores previous value internally
 *
 * @param brain Brain handle
 * @param out   Output structure (zeroed on error)
 * @return NIMCP_OK on success, error code otherwise
 */
nimcp_status_t nimcp_brain_get_synapse_stats(nimcp_brain_t brain, nimcp_synapse_stats_t* out);

//=============================================================================
// Cognitive Output Rubric - Human-Style Quality Evaluation
//=============================================================================

/**
 * @brief Rubric evaluation result — flat public struct
 *
 * Two-tier quality assessment of brain_decide() output:
 *   Tier 1 (Structural): rule-based, automatable checks
 *   Tier 2 (Qualitative): holistic, subsystem-dependent scores
 *
 * Grade scale: A+ (>=0.93) through F (<0.50)
 */
typedef struct {
    /* Tier 1: Structural */
    float internal_consistency;      /**< [0-1] Output vector self-agreement */
    float confidence_calibration;    /**< [0-1] Confidence vs running accuracy */
    float completeness;              /**< [0-1] Non-zero output coverage */
    float reasoning_chain_quality;   /**< [0-1] Explanation depth & structure */
    float epistemic_quality;         /**< [0-1] Evidence quality */
    float ethical_alignment;         /**< [0-1] Golden Rule alignment */
    float tier1_score;               /**< Weighted aggregate of Tier 1 */

    /* Tier 2: Qualitative */
    float originality;               /**< [0-1] Creative novelty */
    float integration_depth;         /**< [0-1] Cross-system breadth */
    float communication_clarity;     /**< [0-1] Label specificity + readability */
    float engagement_quality;        /**< [0-1] Berlyne hedonic + arousal */
    float empathetic_accuracy;       /**< [0-1] Mirror neuron match quality */
    float information_density;       /**< [0-1] Normalized Shannon entropy */
    float tier2_score;               /**< Weighted aggregate of Tier 2 */

    /* Overall */
    float overall_score;             /**< [0-1] Weighted combination of tiers */
    char  grade;                     /**< A/B/C/D/F */
    char  grade_modifier;            /**< +/-/space */
    uint32_t subsystems_available;   /**< Bitmask of available subsystems */
    uint64_t evaluation_time_us;     /**< Wall-clock evaluation time */
} nimcp_rubric_t;

/**
 * @brief Evaluate quality of last brain decision using two-tier rubric
 *
 * WHAT: Grades brain output like a professor grading an essay
 * WHY:  Unified quality metric combining structural + qualitative dimensions
 * HOW:  Calls internal subsystems (epistemic, ethics, aesthetic, mirror neurons)
 *       and aggregates into a single grade. Missing subsystems score 0.5 (neutral).
 *
 * Must be called AFTER nimcp_brain_predict() or nimcp_brain_decide_full().
 *
 * @param brain Brain handle
 * @param rubric Output rubric result
 * @return NIMCP_OK on success, error code otherwise
 */
nimcp_status_t nimcp_brain_rubric(nimcp_brain_t brain, nimcp_rubric_t* rubric);

/**
 * @brief Broadcast rubric data via bio-async message system
 *
 * @param brain Brain handle (must have been rubric'd first)
 * @return NIMCP_OK on success, error code otherwise
 */
nimcp_status_t nimcp_brain_broadcast_rubric(nimcp_brain_t brain);

/**
 * @brief Set dedicated validation features for rubric evaluation during training
 *
 * When set, rubric evaluations during training use these features instead of
 * the current training step's features. Features are copied internally.
 *
 * @param brain Brain handle
 * @param features Validation feature array (copied internally)
 * @param num_features Number of features
 * @return NIMCP_OK on success
 */
nimcp_status_t nimcp_brain_set_rubric_validation(
    nimcp_brain_t brain, const float* features, uint32_t num_features);

/**
 * @brief Get rubric statistics accumulated during training
 *
 * @param brain Brain handle
 * @param eval_count Output: number of rubric evaluations
 * @param min_score Output: minimum observed score
 * @param max_score Output: maximum observed score
 * @param avg_score Output: average score
 * @param last_rubric Output: most recent rubric result (NULL to skip)
 * @return NIMCP_OK on success
 */
nimcp_status_t nimcp_brain_get_rubric_training_stats(
    nimcp_brain_t brain,
    uint64_t* eval_count, float* min_score, float* max_score,
    float* avg_score, nimcp_rubric_t* last_rubric);

//=============================================================================
// Copy-on-Write (COW) Cache API - Efficient Memory Sharing
//=============================================================================

/**
 * @brief Opaque handle to a brain snapshot (for COW save/restore)
 */
typedef struct nimcp_brain_snapshot_handle* nimcp_brain_snapshot_t;

/**
 * @brief Clone a brain using copy-on-write (COW) caching
 *
 * WHAT: Creates a lightweight clone that shares memory with the original
 * WHY:  Enables efficient brain replication (86% memory savings)
 * HOW:  Uses nimcp_cache_reference() to share large structures (weights, connections)
 *
 * The clone initially shares all large data structures (neural network weights,
 * connections, knowledge base) with the original brain. Memory is only copied
 * when either brain modifies shared data (copy-on-write semantics).
 *
 * PERFORMANCE:
 * - Clone time: <10ms (vs ~1000ms for full copy)
 * - Memory overhead: ~1MB metadata (vs ~50MB full copy)
 * - Memory savings: 86% for replicas, 99% for snapshots
 *
 * THREAD SAFETY: Thread-safe
 * MEMORY: O(1) initially, O(n) after modifications
 *
 * @param original Brain to clone
 * @return Cloned brain handle or NULL on error
 *
 * EXAMPLE:
 * ```c
 * nimcp_brain_t original = nimcp_brain_create(...);
 * nimcp_brain_t clone = nimcp_brain_clone_cow(original);
 * // clone shares memory with original (fast, low memory)
 * nimcp_brain_learn_example(clone, ...);  // Triggers copy on first write
 * ```
 */
nimcp_brain_t nimcp_brain_clone_cow(nimcp_brain_t original);

/**
 * @brief Create instant snapshot of brain state using COW
 *
 * WHAT: Creates zero-copy snapshot for rollback/checkpointing
 * WHY:  Enables instant state capture without memory overhead
 * HOW:  Uses nimcp_cache_reference() to share all brain data
 *
 * Snapshots are instant (<1ms) and use minimal memory (~48 bytes overhead).
 * The original brain can continue learning while snapshot preserves the
 * original state. Snapshot automatically copies data if brain modifies it.
 *
 * PERFORMANCE:
 * - Snapshot time: <1ms (zero-copy)
 * - Memory overhead: ~48 bytes metadata
 * - Memory savings: 99% vs traditional snapshot
 *
 * THREAD SAFETY: Thread-safe
 * MEMORY: O(1) overhead
 *
 * @param brain Brain to snapshot
 * @return Snapshot handle or NULL on error
 *
 * EXAMPLE:
 * ```c
 * nimcp_brain_snapshot_t checkpoint = nimcp_brain_snapshot_cow(brain);
 * train_epochs(brain, 100);
 * if (performance < threshold) {
 *     nimcp_brain_restore_cow(brain, checkpoint);  // Instant rollback
 * }
 * nimcp_brain_snapshot_destroy(checkpoint);
 * ```
 */
nimcp_brain_snapshot_t nimcp_brain_snapshot_cow(nimcp_brain_t brain);

/**
 * @brief Restore brain state from COW snapshot
 *
 * WHAT: Restores brain to snapshot state
 * WHY:  Enables instant rollback to previous state
 * HOW:  Replaces current state with snapshot references
 *
 * The brain's current state is replaced with the snapshot state.
 * This is instant because it just swaps cached pointers.
 *
 * PERFORMANCE: <1ms (pointer swapping)
 * THREAD SAFETY: Thread-safe
 * MEMORY: O(1)
 *
 * @param brain Brain to restore
 * @param snapshot Snapshot to restore from
 * @return NIMCP_OK on success, error code otherwise
 */
nimcp_status_t nimcp_brain_restore_cow(nimcp_brain_t brain, nimcp_brain_snapshot_t snapshot);

/**
 * @brief Destroy brain snapshot and release references
 *
 * @param snapshot Snapshot handle
 */
void nimcp_brain_snapshot_destroy(nimcp_brain_snapshot_t snapshot);

//=============================================================================
// Phase 10.2: Working Memory API - Active Representation Buffer
//=============================================================================

/**
 * @brief Add item to working memory for reasoning
 *
 * Adds a feature vector to the brain's working memory buffer (Miller's 7±2).
 * Items are stored with salience-based priority. When capacity is reached,
 * lowest-salience items are evicted. Items decay over time unless refreshed.
 *
 * WHAT: Store active representation in working memory
 * WHY:  Enable reasoning, planning, and cognitive processing over recent inputs
 * HOW:  Priority queue based on salience, automatic decay, capacity-limited
 *
 * @param brain Brain instance
 * @param data Item data (feature vector)
 * @param size Item size (number of floats)
 * @param salience Initial salience (0.0-1.0, higher = more important)
 * @return NIMCP_OK on success, error code otherwise
 *
 * @note Requires enable_working_memory=true in brain config
 * @note Items with salience < 0.01 are automatically removed during decay
 *
 * Example:
 * ```c
 * float features[64] = {...};
 * nimcp_brain_working_memory_add(brain, features, 64, 0.8);  // High salience
 * ```
 */
nimcp_status_t nimcp_brain_working_memory_add(
    nimcp_brain_t brain,
    const float* data,
    uint32_t size,
    float salience
);

/**
 * @brief Get item from working memory by index
 *
 * Retrieves an item from working memory. Items are ordered by salience
 * (index 0 = highest salience). The returned pointer is valid until
 * the next working memory operation.
 *
 * @param brain Brain instance
 * @param index Item index (0 = highest salience)
 * @param size_out Output: item size (number of floats)
 * @return Item data pointer or NULL if index invalid or working memory disabled
 *
 * Example:
 * ```c
 * uint32_t size;
 * const float* item = nimcp_brain_working_memory_get(brain, 0, &size);
 * if (item) {
 *     // Process highest-salience item
 * }
 * ```
 */
const float* nimcp_brain_working_memory_get(
    nimcp_brain_t brain,
    uint32_t index,
    uint32_t* size_out
);

/**
 * @brief Get working memory statistics
 *
 * Returns current size and capacity of working memory buffer.
 *
 * @param brain Brain instance
 * @param current_size_out Output: current number of items
 * @param capacity_out Output: maximum capacity (typically 7)
 * @return NIMCP_OK on success, error code otherwise
 *
 * Example:
 * ```c
 * uint32_t size, capacity;
 * nimcp_brain_working_memory_stats(brain, &size, &capacity);
 * printf("Working memory: %u/%u items\n", size, capacity);
 * ```
 */
nimcp_status_t nimcp_brain_working_memory_stats(
    nimcp_brain_t brain,
    uint32_t* current_size_out,
    uint32_t* capacity_out
);

/**
 * @brief Refresh item in working memory (prevent decay)
 *
 * Resets the timestamp of an item, preventing temporal decay.
 * This simulates attention/rehearsal in biological working memory
 * (frontal-parietal loop).
 *
 * @param brain Brain instance
 * @param index Item index to refresh
 * @return NIMCP_OK on success, error code otherwise
 *
 * Example:
 * ```c
 * // Keep important item in memory by refreshing it
 * nimcp_brain_working_memory_refresh(brain, 0);  // Refresh highest-salience
 * ```
 */
nimcp_status_t nimcp_brain_working_memory_refresh(
    nimcp_brain_t brain,
    uint32_t index
);

//=============================================================================
// Global Workspace API - Conscious Access and Broadcasting
//=============================================================================

/**
 * @brief Cognitive module identifiers for Global Workspace Theory
 *
 * These modules can compete for conscious access via the global workspace.
 * Based on Global Workspace Theory (Baars, 1988; Dehaene, 2011).
 */
typedef enum {
    NIMCP_MODULE_NONE = 0,
    NIMCP_MODULE_PERCEPTION,
    NIMCP_MODULE_WORKING_MEMORY,
    NIMCP_MODULE_EXECUTIVE,
    NIMCP_MODULE_THEORY_OF_MIND,
    NIMCP_MODULE_ETHICS,
    NIMCP_MODULE_ATTENTION,
    NIMCP_MODULE_EMOTION,
    NIMCP_MODULE_SALIENCE,
    NIMCP_MODULE_MOTOR,
    NIMCP_MODULE_LANGUAGE,
    NIMCP_MODULE_METACOGNITION,
    NIMCP_MODULE_CURIOSITY,
    NIMCP_MODULE_INTROSPECTION,
    NIMCP_MODULE_PREDICTIVE,
    NIMCP_MODULE_CONSOLIDATION,
    NIMCP_MODULE_EPISODIC_MEMORY,
    NIMCP_MODULE_SEMANTIC_MEMORY,
    NIMCP_MODULE_WELLBEING,
    NIMCP_MODULE_MENTAL_HEALTH,
    NIMCP_MODULE_GOAL_MOTIVATION,
    NIMCP_MODULE_COGNITIVE_CONTROL,
    NIMCP_MODULE_CUSTOM_START = 100
} nimcp_cognitive_module_t;

/**
 * @brief Compete for global workspace access
 *
 * WHAT: Submit content to global workspace for potential conscious broadcast
 * WHY:  Enable cross-module information integration via conscious access
 * HOW:  Content competes with other modules; winner gets broadcast
 *
 * @param brain Brain instance
 * @param module Source module identifier
 * @param content Content vector (size = workspace capacity, typically 256 floats)
 * @param content_dim Content dimension (must match workspace capacity)
 * @param strength Competition strength (0.0 to 1.0, higher = more likely to win)
 * @return NIMCP_OK if won and broadcast, NIMCP_ERROR otherwise
 *
 * @note Requires enable_global_workspace=true in brain config
 * @note Refractory period (default 50ms) prevents rapid successive broadcasts
 * @note Ignition threshold (default 0.6) gates conscious access
 *
 * Example:
 * @code
 * float features[256] = {...};
 * nimcp_status_t status = nimcp_brain_workspace_compete(
 *     brain, NIMCP_MODULE_PERCEPTION, features, 256, 0.85
 * );
 * if (status == NIMCP_OK) {
 *     printf("Content reached conscious access!\n");
 * }
 * @endcode
 */
nimcp_status_t nimcp_brain_workspace_compete(
    nimcp_brain_t brain,
    nimcp_cognitive_module_t module,
    const float* content,
    uint32_t content_dim,
    float strength
);

/**
 * @brief Read current global workspace broadcast
 *
 * WHAT: Retrieve content from current conscious broadcast
 * WHY:  Allow modules to access globally broadcast information
 * HOW:  Copy broadcast content to provided buffer
 *
 * @param brain Brain instance
 * @param content Output buffer for broadcast content
 * @param max_dim Maximum buffer size
 * @param actual_dim Output: actual content dimension
 * @param source_module Output: source module of broadcast
 * @return NIMCP_OK if broadcast available, NIMCP_ERROR otherwise
 *
 * Example:
 * @code
 * float content[256];
 * uint32_t dim;
 * nimcp_cognitive_module_t source;
 * if (nimcp_brain_workspace_read(brain, content, 256, &dim, &source) == NIMCP_OK) {
 *     printf("Broadcast from module %d, dimension %u\n", source, dim);
 * }
 * @endcode
 */
nimcp_status_t nimcp_brain_workspace_read(
    nimcp_brain_t brain,
    float* content,
    uint32_t max_dim,
    uint32_t* actual_dim,
    nimcp_cognitive_module_t* source_module
);

/**
 * @brief Subscribe module to workspace broadcasts
 *
 * WHAT: Register module to receive all future broadcasts
 * WHY:  Enable module to stay informed of conscious content
 * HOW:  Add module to subscriber list
 *
 * @param brain Brain instance
 * @param module Module to subscribe
 * @return NIMCP_OK on success, NIMCP_ERROR otherwise
 */
nimcp_status_t nimcp_brain_workspace_subscribe(
    nimcp_brain_t brain,
    nimcp_cognitive_module_t module
);

/**
 * @brief Unsubscribe module from workspace broadcasts
 *
 * @param brain Brain instance
 * @param module Module to unsubscribe
 * @return NIMCP_OK on success, NIMCP_ERROR otherwise
 */
nimcp_status_t nimcp_brain_workspace_unsubscribe(
    nimcp_brain_t brain,
    nimcp_cognitive_module_t module
);

/**
 * @brief Check if workspace has active broadcast
 *
 * @param brain Brain instance
 * @param has_broadcast Output: true if broadcast active
 * @return NIMCP_OK on success, NIMCP_ERROR otherwise
 */
nimcp_status_t nimcp_brain_workspace_has_broadcast(
    nimcp_brain_t brain,
    bool* has_broadcast
);

/**
 * @brief Get workspace statistics
 *
 * @param brain Brain instance
 * @param total_broadcasts Output: total broadcasts since creation
 * @param total_competitions Output: total competition attempts
 * @param avg_strength Output: average broadcast strength
 * @return NIMCP_OK on success, NIMCP_ERROR otherwise
 */
nimcp_status_t nimcp_brain_workspace_stats(
    nimcp_brain_t brain,
    uint32_t* total_broadcasts,
    uint32_t* total_competitions,
    float* avg_strength
);

//=============================================================================
// Neural Network API - Low-Level Interface (Advanced Users)
//=============================================================================

/**
 * @brief Create a neural network with custom configuration
 *
 * @param num_inputs Number of input neurons
 * @param num_outputs Number of output neurons
 * @param num_hidden Number of hidden neurons
 * @param learning_rate Learning rate (typically 0.001 - 0.1)
 * @return Network handle or NULL on error
 */
nimcp_network_t nimcp_network_create(
    uint32_t num_inputs,
    uint32_t num_outputs,
    uint32_t num_hidden,
    float learning_rate
);

/**
 * @brief Destroy a neural network
 *
 * @param network Network handle
 */
void nimcp_network_destroy(nimcp_network_t network);

/**
 * @brief Forward pass through network
 *
 * @param network Network handle
 * @param inputs Input array (size = num_inputs)
 * @param num_inputs Number of inputs
 * @param outputs Output array (size = num_outputs, pre-allocated)
 * @param num_outputs Number of outputs
 * @return NIMCP_OK on success, error code otherwise
 */
nimcp_status_t nimcp_network_forward(
    nimcp_network_t network,
    const float* inputs,
    uint32_t num_inputs,
    float* outputs,
    uint32_t num_outputs
);

/**
 * @brief Train network on a single example (supervised learning)
 *
 * @param network Network handle
 * @param inputs Input array
 * @param num_inputs Number of inputs
 * @param targets Target output array
 * @param num_targets Number of target outputs
 * @return NIMCP_OK on success, error code otherwise
 */
nimcp_status_t nimcp_network_train(
    nimcp_network_t network,
    const float* inputs,
    uint32_t num_inputs,
    const float* targets,
    uint32_t num_targets
);

//=============================================================================
// Ethics Module API
//=============================================================================

/**
 * @brief Create an ethics module
 *
 * @return Ethics module handle or NULL on error
 */
nimcp_ethics_t nimcp_ethics_create(void);

/**
 * @brief Destroy an ethics module
 *
 * @param ethics Ethics module handle
 */
void nimcp_ethics_destroy(nimcp_ethics_t ethics);

/**
 * @brief Check if an action is ethically acceptable
 *
 * @param ethics Ethics module handle
 * @param situation Situation features array
 * @param num_features Number of features
 * @param out_score Ethical score (-1.0 = harmful, 0.0 = neutral, 1.0 = beneficial)
 * @return NIMCP_OK on success, error code otherwise
 */
nimcp_status_t nimcp_ethics_check(
    nimcp_ethics_t ethics,
    const float* situation,
    uint32_t num_features,
    float* out_score
);

//=============================================================================
// Knowledge Graph API
//=============================================================================

/**
 * @brief Create a knowledge graph
 *
 * @return Knowledge graph handle or NULL on error
 */
nimcp_knowledge_t nimcp_knowledge_create(void);

/**
 * @brief Destroy a knowledge graph
 *
 * @param knowledge Knowledge graph handle
 */
void nimcp_knowledge_destroy(nimcp_knowledge_t knowledge);

/**
 * @brief Add a fact to the knowledge graph
 *
 * @param knowledge Knowledge graph handle
 * @param subject Subject entity
 * @param predicate Relationship type
 * @param object Object entity
 * @return NIMCP_OK on success, error code otherwise
 */
nimcp_status_t nimcp_knowledge_add_fact(
    nimcp_knowledge_t knowledge,
    const char* subject,
    const char* predicate,
    const char* object
);

/**
 * @brief Query the knowledge graph
 *
 * @param knowledge Knowledge graph handle
 * @param query Query string
 * @param out_result Result buffer (pre-allocated, min 1024 bytes)
 * @param max_result_len Maximum result length
 * @return NIMCP_OK on success, error code otherwise
 */
nimcp_status_t nimcp_knowledge_query(
    nimcp_knowledge_t knowledge,
    const char* query,
    char* out_result,
    uint32_t max_result_len
);

//=============================================================================
// Complex Number & Oscillation API - Phase Coding & Neural Synchrony
//=============================================================================

/**
 * @brief Oscillation phasor (complex number) representation
 *
 * Represents neural oscillations as amplitude + phase:
 * - amplitude: Strength of oscillation (magnitude)
 * - phase: Timing within oscillation cycle (radians, -π to π)
 *
 * Mathematical form: z = amplitude·e^(i·phase)
 */
typedef struct {
    float amplitude;  /**< Oscillation amplitude (≥ 0) */
    float phase;      /**< Phase angle in radians (-π to π) */
} nimcp_oscillation_phasor_t;

/**
 * @brief Enable or disable complex oscillation features
 *
 * WHAT: Activates phase coding and complex number support in brain
 * WHY:  Complex oscillations enable phase-based memory and binding
 * HOW:  Configures neurons/synapses to track amplitude + phase
 *
 * When enabled:
 * - Neurons track oscillatory phase and amplitude
 * - Synapses compute phase-dependent weights
 * - Phase coherence and PAC metrics become available
 * - ~15% memory overhead for phase tracking
 *
 * Performance impact:
 * - Memory: +15% (phase data storage)
 * - Compute: +10% (complex arithmetic)
 * - Benefits: Phase-based memory, theta-gamma coupling
 *
 * @param brain Brain handle
 * @param enable true to enable, false to disable
 * @return true on success, false on error
 *
 * EXAMPLE:
 * ```c
 * nimcp_brain_t brain = nimcp_brain_create(...);
 * nimcp_enable_complex_oscillations(brain, true);
 * // Now can query phase coherence, PAC, etc.
 * ```
 */
bool nimcp_enable_complex_oscillations(nimcp_brain_t brain, bool enable);

/**
 * @brief Check if complex oscillation features are enabled
 *
 * @param brain Brain handle
 * @return true if enabled, false if disabled or error
 */
bool nimcp_is_complex_oscillations_enabled(nimcp_brain_t brain);

/**
 * @brief Get oscillation phasor for a specific neuron
 *
 * WHAT: Retrieves amplitude and phase of neuron's oscillatory activity
 * WHY:  Enables analysis of neural phase coding and synchronization
 * HOW:  Extracts complex phasor from neuron's oscillation state
 *
 * Returns the neuron's current oscillatory state as a phasor.
 * Requires complex oscillations to be enabled.
 *
 * @param brain Brain handle
 * @param neuron_id Neuron identifier
 * @return Phasor with amplitude and phase (returns {0,0} on error or if disabled)
 *
 * EXAMPLE:
 * ```c
 * nimcp_oscillation_phasor_t phasor = nimcp_get_oscillation_phasor(brain, 42);
 * printf("Neuron 42: amplitude=%.2f, phase=%.2f rad\n",
 *        phasor.amplitude, phasor.phase);
 * ```
 */
nimcp_oscillation_phasor_t nimcp_get_oscillation_phasor(
    nimcp_brain_t brain,
    uint32_t neuron_id
);

/**
 * @brief Compute phase coherence across multiple neurons
 *
 * WHAT: Measures phase synchronization between neural oscillations
 * WHY:  Phase coherence indicates coordinated neural activity
 * HOW:  Computes inter-trial phase coherence (ITPC) across neurons
 *
 * Phase coherence range: [0, 1] where:
 * - 0.0 = random phases (no synchronization)
 * - 0.3 = weak synchronization
 * - 0.6 = moderate synchronization
 * - 1.0 = perfect phase locking
 *
 * Neuroscience application:
 * - High coherence → neurons fire at same phase (coordinated)
 * - Low coherence → independent oscillations
 * - Used to detect neural binding and communication
 *
 * @param brain Brain handle
 * @param neuron_ids Array of neuron identifiers to analyze
 * @param count Number of neurons in array
 * @return Phase coherence value [0, 1] (returns 0.0 on error or if disabled)
 *
 * EXAMPLE:
 * ```c
 * uint32_t neurons[] = {10, 20, 30, 40, 50};
 * float coherence = nimcp_get_phase_coherence(brain, neurons, 5);
 * if (coherence > 0.6f) {
 *     printf("High synchronization detected!\n");
 * }
 * ```
 */
float nimcp_get_phase_coherence(
    nimcp_brain_t brain,
    const uint32_t* neuron_ids,
    uint32_t count
);

/**
 * @brief Compute phase-amplitude coupling (PAC) modulation index
 *
 * WHAT: Measures theta-gamma cross-frequency coupling strength
 * WHY:  PAC is a key mechanism for memory and attention
 * HOW:  Computes modulation of gamma amplitude by theta phase
 *
 * PAC modulation index range: [0, 1] where:
 * - 0.0 = no coupling (theta and gamma independent)
 * - 0.2 = weak coupling
 * - 0.4 = moderate coupling
 * - 0.6+ = strong coupling (indicates active encoding/retrieval)
 *
 * Neuroscience foundation:
 * - Hippocampal theta (4-8Hz) phase modulates gamma (30-100Hz) amplitude
 * - Used for sequence encoding (place cells), working memory
 * - Higher PAC correlates with better memory performance
 *
 * @param brain Brain handle
 * @param theta_freq Theta frequency in Hz (typically 4-8 Hz)
 * @param gamma_freq Gamma frequency in Hz (typically 30-100 Hz)
 * @return PAC modulation index [0, 1] (returns 0.0 on error or if disabled)
 *
 * EXAMPLE:
 * ```c
 * // Measure hippocampal theta-gamma coupling
 * float pac = nimcp_get_pac_modulation(brain, 6.0f, 40.0f);
 * if (pac > 0.4f) {
 *     printf("Strong theta-gamma coupling (memory encoding active)\n");
 * }
 * ```
 */
float nimcp_get_pac_modulation(
    nimcp_brain_t brain,
    float theta_freq,
    float gamma_freq
);

//=============================================================================
// Utility Functions
//=============================================================================

/**
 * @brief Get error message for last error
 *
 * @return Error message string (statically allocated)
 */
const char* nimcp_get_error(void);

/**
 * @brief Check if the linked library version is compatible with the header version.
 * @return NIMCP_OK if compatible, NIMCP_ERROR if major version mismatch
 */
nimcp_status_t nimcp_version_check(void);

/**
 * @brief Initialize NIMCP library (call once at startup)
 *
 * @return NIMCP_OK on success, error code otherwise
 */
nimcp_status_t nimcp_init(void);

/**
 * @brief Shutdown NIMCP library (call once at cleanup)
 */
void nimcp_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif // NIMCP_H
