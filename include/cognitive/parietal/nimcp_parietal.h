/**
 * @file nimcp_parietal.h
 * @brief Parietal Lobe Orchestrator with Full System Integration
 *
 * The parietal lobe module orchestrates mathematical and scientific reasoning,
 * spatial processing, and pattern recognition. This module provides:
 *
 * CORE CAPABILITIES:
 * - Number sense (Weber-Fechner, subitizing)
 * - Spatial reasoning (mental rotation, coordinate transforms)
 * - Mathematical intuition (pattern detection, symmetry)
 * - Scientific reasoning (hypothesis testing, causal inference)
 * - Equation manipulation (symbolic math, differentiation)
 *
 * FULL SYSTEM INTEGRATION:
 * - Brain region integration (REGION_PARIETAL)
 * - Neural network processing (Hamiltonian-Lagrangian NN)
 * - Immune system bridge (inflammation affects precision)
 * - Thalamic routing (attention gating)
 * - Substrate layer (hardware acceleration)
 * - Free Energy Principle bridge
 * - Sleep/circadian modulation
 * - Working memory integration
 * - Logic gate module integration
 * - Training module integration
 * - Perception module integration
 * - Bio-async messaging
 *
 * BIOLOGICAL BASIS:
 * The parietal cortex, particularly the intraparietal sulcus (IPS) and
 * posterior parietal cortex (PPC), handles spatial cognition, numerical
 * processing, and attention. Dense connections with prefrontal, motor,
 * and visual cortices enable integrated mathematical reasoning.
 *
 * USAGE:
 * ```c
 * parietal_lobe_t* parietal = parietal_create();
 *
 * // Connect to brain module
 * parietal_attach_to_brain(parietal, brain, REGION_PARIETAL);
 *
 * // Process request
 * parietal_request_t req = {.type = PARIETAL_PATTERN_DETECT, ...};
 * parietal_result_t result = parietal_process(parietal, &req);
 *
 * parietal_destroy(parietal);
 * ```
 */

#ifndef NIMCP_PARIETAL_H
#define NIMCP_PARIETAL_H

#include "utils/validation/nimcp_common.h"
#include "cognitive/parietal/nimcp_number_sense.h"
#include "cognitive/parietal/nimcp_spatial_reasoning.h"
#include "cognitive/parietal/nimcp_mathematical_intuition.h"
#include "cognitive/parietal/nimcp_scientific_reasoning.h"
#include "cognitive/parietal/nimcp_equation_manipulation.h"
#include "cognitive/parietal/nimcp_chemistry.h"
#include "cognitive/parietal/nimcp_biology.h"
#include "cognitive/parietal/nimcp_software_engineering.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * FORWARD DECLARATIONS (Integration Dependencies)
 * ============================================================================ */

/* Brain regions */
typedef struct brain_module brain_module_t;
typedef struct brain_region brain_region_t;

/* Neural network */
typedef struct neural_network neural_network_t;

/* Immune system */
typedef struct code_immune_system code_immune_system_t;

/* Thalamic routing */
typedef struct thalamic_router thalamic_router_t;

/* Substrate layer */
typedef struct substrate_interface substrate_interface_t;

/* FEP bridge */
typedef struct fep_brain fep_brain_t;

/* Working memory */
typedef struct working_memory working_memory_t;

/* Logic gates */
typedef struct logic_gate_network logic_gate_network_t;

/* Training */
typedef struct training_engine training_engine_t;

/* Perception */
typedef struct perception_system perception_system_t;

/* Sleep modulation */
typedef struct sleep_system sleep_system_t;

/* ============================================================================
 * CONSTANTS
 * ============================================================================ */

/** Bio-async module ID for parietal lobe */
#define BIO_MODULE_PARIETAL             0x0380

/** Maximum pending requests */
#define PARIETAL_MAX_PENDING_REQUESTS   64

/** Maximum result history */
#define PARIETAL_MAX_HISTORY            256

/** Parietal brain region type (matches REGION_PARIETAL = 41) */
#define PARIETAL_BRAIN_REGION_TYPE      41

/* ============================================================================
 * TYPES
 * ============================================================================ */

/** Opaque handle for parietal lobe */
typedef struct parietal_lobe parietal_lobe_t;

/**
 * @brief Request types for parietal processing
 */
typedef enum {
    /* Number sense */
    PARIETAL_ESTIMATE_QUANTITY = 0,
    PARIETAL_COMPARE_QUANTITIES,
    PARIETAL_APPROXIMATE_ARITHMETIC,

    /* Spatial reasoning */
    PARIETAL_MENTAL_ROTATION,
    PARIETAL_COORDINATE_TRANSFORM,
    PARIETAL_SPATIAL_QUERY,

    /* Mathematical intuition */
    PARIETAL_PATTERN_DETECT,
    PARIETAL_PATTERN_EXTRAPOLATE,
    PARIETAL_SYMMETRY_DETECT,
    PARIETAL_SOLVE_ANALOGY,

    /* Scientific reasoning */
    PARIETAL_HYPOTHESIS_CREATE,
    PARIETAL_HYPOTHESIS_UPDATE,
    PARIETAL_DIMENSIONAL_ANALYSIS,
    PARIETAL_CAUSAL_INFERENCE,

    /* Equation manipulation */
    PARIETAL_PARSE_EXPRESSION,
    PARIETAL_DIFFERENTIATE,
    PARIETAL_SIMPLIFY,
    PARIETAL_EVALUATE,
    PARIETAL_SOLVE_EQUATION,

    /* Physics NN */
    PARIETAL_PHYSICS_PREDICT,
    PARIETAL_LEARN_DYNAMICS,

    PARIETAL_REQUEST_TYPE_COUNT
} parietal_request_type_t;

/**
 * @brief Parietal processing request
 */
typedef struct {
    parietal_request_type_t type;       /**< Request type */
    uint64_t request_id;                /**< Unique request ID */
    uint64_t timestamp;                 /**< Request timestamp */
    uint8_t priority;                   /**< Processing priority (0-255) */

    /* Input data (interpretation depends on type) */
    union {
        /* Number sense inputs */
        struct {
            float* values;
            uint32_t num_values;
        } quantity_input;

        struct {
            float magnitude_a;
            float magnitude_b;
        } comparison_input;

        struct {
            float a, b;
            char operation;  /* '+', '-', '*', '/' */
        } arithmetic_input;

        /* Spatial inputs */
        struct {
            spatial_object_t* object_a;
            spatial_object_t* object_b;
        } rotation_input;

        struct {
            vec3_t position;
            observer_pose_t* observer;
            bool ego_to_allocentric;
        } transform_input;

        struct {
            vec3_t query_point;
            float radius;
            uint32_t k;
        } spatial_query_input;

        /* Pattern inputs */
        struct {
            float* sequence;
            uint32_t length;
        } pattern_input;

        struct {
            vec3_t* points;
            uint32_t num_points;
        } symmetry_input;

        struct {
            float a, b, c;
        } analogy_input;

        /* Hypothesis inputs */
        struct {
            char description[256];
            float prior;
        } hypothesis_create_input;

        struct {
            hypothesis_t* hypothesis;
            data_sample_t* samples;
            uint32_t num_samples;
        } hypothesis_update_input;

        /* Equation inputs */
        struct {
            char expression[512];
            char variable[32];
        } equation_input;

        struct {
            expr_node_t* expr;
            variable_binding_t* bindings;
            uint32_t num_bindings;
        } eval_input;

        /* Physics NN inputs */
        struct {
            float* state;
            uint32_t state_dim;
            float dt;
        } physics_input;

    } input;

    /* Routing hints */
    bool use_neural_network;            /**< Route through NN */
    bool use_thalamic_gating;           /**< Apply thalamic attention */
    bool use_substrate_accel;           /**< Use hardware acceleration */
    bool async;                         /**< Asynchronous processing */
} parietal_request_t;

/**
 * @brief Parietal processing result
 */
typedef struct {
    parietal_request_type_t type;       /**< Result type */
    uint64_t request_id;                /**< Matching request ID */
    uint64_t processing_time_us;        /**< Processing time */
    float confidence;                   /**< Result confidence [0,1] */
    bool success;                       /**< Processing succeeded */
    char error_message[128];            /**< Error if !success */

    /* Output data */
    union {
        number_estimate_t estimate;
        number_comparison_t comparison;
        approx_arithmetic_t arithmetic;
        rotation_result_t rotation;
        vec3_t transformed_position;
        spatial_query_result_t* spatial_results;
        detected_pattern_t pattern;
        symmetry_result_t symmetry;
        analogy_result_t analogy;
        hypothesis_t hypothesis;
        expr_node_t* expression;
        float evaluated_value;

        struct {
            float* predicted_state;
            float energy;
        } physics_output;

    } output;
} parietal_result_t;

/**
 * @brief Parietal configuration
 */
typedef struct {
    /* Submodule configurations */
    number_sense_config_t number_sense;
    spatial_config_t spatial;
    math_intuition_config_t math_intuition;
    scientific_config_t scientific;
    equation_config_t equation;

    /* Integration settings */
    bool enable_neural_network;         /**< Enable NN processing (default: true) */
    bool enable_immune_bridge;          /**< Enable immune modulation (default: true) */
    bool enable_thalamic_routing;       /**< Enable thalamic gating (default: true) */
    bool enable_substrate_accel;        /**< Enable substrate acceleration (default: true) */
    bool enable_fep_bridge;             /**< Enable FEP integration (default: true) */
    bool enable_sleep_modulation;       /**< Enable sleep effects (default: true) */
    bool enable_working_memory;         /**< Enable working memory (default: true) */
    bool enable_logic_gates;            /**< Enable logic gate integration (default: true) */
    bool enable_training;               /**< Enable training integration (default: true) */
    bool enable_perception;             /**< Enable perception integration (default: true) */
    bool enable_bio_async;              /**< Enable bio-async messaging (default: true) */

    /* Neural network settings */
    uint32_t nn_hidden_size;            /**< NN hidden layer size (default: 256) */
    float nn_learning_rate;             /**< NN learning rate (default: 0.001) */
    bool nn_use_hamiltonian;            /**< Use Hamiltonian structure (default: true) */
    bool nn_use_lagrangian;             /**< Use Lagrangian constraints (default: true) */

    /* Performance settings */
    uint32_t max_parallel_requests;     /**< Max parallel processing (default: 8) */
    uint32_t request_timeout_ms;        /**< Request timeout (default: 5000) */
} parietal_config_t;

/**
 * @brief Parietal lobe statistics
 */
typedef struct {
    /* Request counts by type */
    uint64_t requests_by_type[PARIETAL_REQUEST_TYPE_COUNT];
    uint64_t total_requests;
    uint64_t failed_requests;

    /* Submodule statistics */
    number_sense_stats_t number_sense;
    spatial_stats_t spatial;
    math_intuition_stats_t math_intuition;
    scientific_stats_t scientific;
    equation_stats_t equation;

    /* Integration statistics */
    uint64_t neural_network_activations;
    uint64_t thalamic_gates_applied;
    uint64_t substrate_accelerations;
    uint64_t immune_modulations;
    uint64_t fep_predictions;

    /* Performance */
    float avg_processing_time_us;
    float avg_confidence;
    float current_inflammation;
    float current_fatigue;
} parietal_stats_t;

/**
 * @brief Callback for async processing
 */
typedef void (*parietal_callback_t)(
    parietal_lobe_t* parietal,
    const parietal_result_t* result,
    void* user_data
);

/* ============================================================================
 * LIFECYCLE API
 * ============================================================================ */

/**
 * @brief Create parietal lobe with default configuration
 * @return Handle or NULL on error
 */
parietal_lobe_t* parietal_create(void);

/**
 * @brief Create parietal lobe with custom configuration
 * @param config Configuration (NULL for defaults)
 * @return Handle or NULL on error
 */
parietal_lobe_t* parietal_create_custom(const parietal_config_t* config);

/**
 * @brief Destroy parietal lobe
 * @param parietal Handle (NULL safe)
 */
void parietal_destroy(parietal_lobe_t* parietal);

/**
 * @brief Get default configuration
 * @return Default configuration struct
 */
parietal_config_t parietal_default_config(void);

/**
 * @brief Validate configuration
 * @param config Configuration to validate
 * @return true if valid
 */
bool parietal_validate_config(const parietal_config_t* config);

/* ============================================================================
 * BRAIN INTEGRATION API
 * ============================================================================ */

/**
 * @brief Attach to brain module
 *
 * Creates brain region and establishes connections.
 *
 * @param parietal Parietal lobe handle
 * @param brain Brain module handle
 * @param num_neurons Number of neurons for region
 * @return 0 on success
 */
int parietal_attach_to_brain(
    parietal_lobe_t* parietal,
    brain_module_t* brain,
    uint32_t num_neurons
);

/**
 * @brief Get attached brain region
 *
 * @param parietal Parietal lobe handle
 * @return Brain region or NULL if not attached
 */
brain_region_t* parietal_get_brain_region(parietal_lobe_t* parietal);

/**
 * @brief Connect to other brain region
 *
 * @param parietal Parietal lobe handle
 * @param target_region_id Target region ID
 * @param connection_strength Connection strength [0,1]
 * @return 0 on success
 */
int parietal_connect_to_region(
    parietal_lobe_t* parietal,
    uint32_t target_region_id,
    float connection_strength
);

/* ============================================================================
 * INTEGRATION BRIDGES API
 * ============================================================================ */

/**
 * @brief Attach immune system bridge
 *
 * @param parietal Parietal lobe handle
 * @param immune Immune system handle
 * @return 0 on success
 */
int parietal_attach_immune(
    parietal_lobe_t* parietal,
    code_immune_system_t* immune
);

/**
 * @brief Attach thalamic router
 *
 * @param parietal Parietal lobe handle
 * @param thalamus Thalamic router handle
 * @return 0 on success
 */
int parietal_attach_thalamus(
    parietal_lobe_t* parietal,
    thalamic_router_t* thalamus
);

/**
 * @brief Attach substrate interface
 *
 * @param parietal Parietal lobe handle
 * @param substrate Substrate interface handle
 * @return 0 on success
 */
int parietal_attach_substrate(
    parietal_lobe_t* parietal,
    substrate_interface_t* substrate
);

/**
 * @brief Attach FEP brain bridge
 *
 * @param parietal Parietal lobe handle
 * @param fep FEP brain handle
 * @return 0 on success
 */
int parietal_attach_fep(
    parietal_lobe_t* parietal,
    fep_brain_t* fep
);

/**
 * @brief Attach working memory
 *
 * @param parietal Parietal lobe handle
 * @param wm Working memory handle
 * @return 0 on success
 */
int parietal_attach_working_memory(
    parietal_lobe_t* parietal,
    working_memory_t* wm
);

/**
 * @brief Attach logic gate network
 *
 * @param parietal Parietal lobe handle
 * @param logic Logic gate network handle
 * @return 0 on success
 */
int parietal_attach_logic_gates(
    parietal_lobe_t* parietal,
    logic_gate_network_t* logic
);

/**
 * @brief Attach training engine
 *
 * @param parietal Parietal lobe handle
 * @param training Training engine handle
 * @return 0 on success
 */
int parietal_attach_training(
    parietal_lobe_t* parietal,
    training_engine_t* training
);

/**
 * @brief Attach perception system
 *
 * @param parietal Parietal lobe handle
 * @param perception Perception system handle
 * @return 0 on success
 */
int parietal_attach_perception(
    parietal_lobe_t* parietal,
    perception_system_t* perception
);

/**
 * @brief Attach sleep system
 *
 * @param parietal Parietal lobe handle
 * @param sleep Sleep system handle
 * @return 0 on success
 */
int parietal_attach_sleep(
    parietal_lobe_t* parietal,
    sleep_system_t* sleep
);

/* ============================================================================
 * PROCESSING API
 * ============================================================================ */

/**
 * @brief Process parietal request (synchronous)
 *
 * @param parietal Parietal lobe handle
 * @param request Request to process
 * @return Processing result
 */
parietal_result_t parietal_process(
    parietal_lobe_t* parietal,
    const parietal_request_t* request
);

/**
 * @brief Process parietal request (asynchronous)
 *
 * @param parietal Parietal lobe handle
 * @param request Request to process
 * @param callback Completion callback
 * @param user_data User data for callback
 * @return Request ID or 0 on error
 */
uint64_t parietal_process_async(
    parietal_lobe_t* parietal,
    const parietal_request_t* request,
    parietal_callback_t callback,
    void* user_data
);

/**
 * @brief Poll for async result
 *
 * @param parietal Parietal lobe handle
 * @param request_id Request ID
 * @param result Output result (if ready)
 * @return 1 if result ready, 0 if pending, -1 on error
 */
int parietal_poll_result(
    parietal_lobe_t* parietal,
    uint64_t request_id,
    parietal_result_t* result
);

/**
 * @brief Wait for async result
 *
 * @param parietal Parietal lobe handle
 * @param request_id Request ID
 * @param timeout_ms Timeout in milliseconds
 * @param result Output result
 * @return 0 on success, -1 on timeout/error
 */
int parietal_wait_result(
    parietal_lobe_t* parietal,
    uint64_t request_id,
    uint32_t timeout_ms,
    parietal_result_t* result
);

/* ============================================================================
 * NEURAL NETWORK API (Hamiltonian-Lagrangian NN)
 * ============================================================================ */

/**
 * @brief Train physics neural network
 *
 * @param parietal Parietal lobe handle
 * @param states Training states [num_samples x state_dim]
 * @param derivatives State derivatives [num_samples x state_dim]
 * @param num_samples Number of training samples
 * @param epochs Number of training epochs
 * @return Final training loss
 */
float parietal_train_physics_nn(
    parietal_lobe_t* parietal,
    const float** states,
    const float** derivatives,
    uint32_t num_samples,
    uint32_t epochs
);

/**
 * @brief Predict state evolution using physics NN
 *
 * @param parietal Parietal lobe handle
 * @param initial_state Initial state
 * @param state_dim State dimension
 * @param dt Time step
 * @param steps Number of steps to predict
 * @param predicted_states Output predicted states [steps x state_dim]
 * @return 0 on success
 */
int parietal_predict_dynamics(
    parietal_lobe_t* parietal,
    const float* initial_state,
    uint32_t state_dim,
    float dt,
    uint32_t steps,
    float** predicted_states
);

/**
 * @brief Get Hamiltonian (total energy) for state
 *
 * @param parietal Parietal lobe handle
 * @param state System state
 * @param state_dim State dimension
 * @return Hamiltonian value
 */
float parietal_compute_hamiltonian(
    parietal_lobe_t* parietal,
    const float* state,
    uint32_t state_dim
);

/* ============================================================================
 * CONVENIENCE WRAPPERS API
 * ============================================================================ */

/**
 * @brief Estimate quantity (convenience wrapper)
 */
number_estimate_t parietal_estimate_quantity(
    parietal_lobe_t* parietal,
    const float* values,
    uint32_t num_values
);

/**
 * @brief Detect pattern (convenience wrapper)
 */
detected_pattern_t parietal_detect_pattern(
    parietal_lobe_t* parietal,
    const float* sequence,
    uint32_t length
);

/**
 * @brief Parse and differentiate expression (convenience wrapper)
 */
expr_node_t* parietal_differentiate_expression(
    parietal_lobe_t* parietal,
    const char* expr_string,
    const char* variable
);

/**
 * @brief Mental rotation comparison (convenience wrapper)
 */
rotation_result_t parietal_mental_rotate(
    parietal_lobe_t* parietal,
    const spatial_object_t* object_a,
    const spatial_object_t* object_b
);

/* ============================================================================
 * MODULATION API
 * ============================================================================ */

/**
 * @brief Set global inflammation level
 *
 * Propagates to all submodules.
 *
 * @param parietal Parietal lobe handle
 * @param level Inflammation level [0,1]
 * @return 0 on success
 */
int parietal_set_inflammation(
    parietal_lobe_t* parietal,
    float level
);

/**
 * @brief Set global fatigue level
 *
 * @param parietal Parietal lobe handle
 * @param level Fatigue level [0,1]
 * @return 0 on success
 */
int parietal_set_fatigue(
    parietal_lobe_t* parietal,
    float level
);

/**
 * @brief Update from sleep system
 *
 * Called automatically if sleep system attached.
 *
 * @param parietal Parietal lobe handle
 * @return 0 on success
 */
int parietal_update_from_sleep(parietal_lobe_t* parietal);

/**
 * @brief Update from immune system
 *
 * Called automatically if immune system attached.
 *
 * @param parietal Parietal lobe handle
 * @return 0 on success
 */
int parietal_update_from_immune(parietal_lobe_t* parietal);

/* ============================================================================
 * STEPPING API
 * ============================================================================ */

/**
 * @brief Step parietal lobe forward in time
 *
 * Processes queued requests and updates neural state.
 *
 * @param parietal Parietal lobe handle
 * @param delta_t Time step in microseconds
 * @return 0 on success
 */
int parietal_step(
    parietal_lobe_t* parietal,
    uint64_t delta_t
);

/**
 * @brief Process all pending requests
 *
 * @param parietal Parietal lobe handle
 * @return Number of requests processed
 */
uint32_t parietal_process_pending(parietal_lobe_t* parietal);

/* ============================================================================
 * DIRECT SUBMODULE ACCESS API
 * ============================================================================ */

/**
 * @brief Get number sense submodule
 */
number_sense_t* parietal_get_number_sense(parietal_lobe_t* parietal);

/**
 * @brief Get spatial reasoning submodule
 */
spatial_reasoning_t* parietal_get_spatial(parietal_lobe_t* parietal);

/**
 * @brief Get mathematical intuition submodule
 */
math_intuition_t* parietal_get_math_intuition(parietal_lobe_t* parietal);

/**
 * @brief Get scientific reasoning submodule
 */
scientific_reasoning_t* parietal_get_scientific(parietal_lobe_t* parietal);

/**
 * @brief Get equation engine submodule
 */
equation_engine_t* parietal_get_equation_engine(parietal_lobe_t* parietal);

/* ============================================================================
 * STATISTICS API
 * ============================================================================ */

/**
 * @brief Get aggregated statistics
 *
 * @param parietal Parietal lobe handle
 * @param stats Output statistics
 * @return 0 on success
 */
int parietal_get_stats(
    const parietal_lobe_t* parietal,
    parietal_stats_t* stats
);

/**
 * @brief Reset all statistics
 *
 * @param parietal Parietal lobe handle
 */
void parietal_reset_stats(parietal_lobe_t* parietal);

/**
 * @brief Get last error message
 * @return Thread-local error message
 */
const char* parietal_get_last_error(void);

/* ============================================================================
 * BIO-ASYNC MESSAGING
 * ============================================================================ */

/**
 * @brief Bio-async message types for parietal lobe
 */
typedef enum {
    BIO_MSG_PARIETAL_ESTIMATE = 0x0380,
    BIO_MSG_PARIETAL_COMPARE,
    BIO_MSG_PARIETAL_ROTATE,
    BIO_MSG_PARIETAL_PATTERN,
    BIO_MSG_PARIETAL_HYPOTHESIS,
    BIO_MSG_PARIETAL_EQUATION,
    BIO_MSG_PARIETAL_PHYSICS,
    BIO_MSG_PARIETAL_RESULT,
} bio_msg_parietal_t;

/**
 * @brief Handle bio-async message
 *
 * @param parietal Parietal lobe handle
 * @param msg_type Message type
 * @param payload Message payload
 * @param payload_size Payload size
 * @return 0 on success
 */
int parietal_handle_bio_msg(
    parietal_lobe_t* parietal,
    uint32_t msg_type,
    const void* payload,
    uint32_t payload_size
);

#ifdef __cplusplus
}
#endif

#endif /* NIMCP_PARIETAL_H */
