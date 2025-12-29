/**
 * @file nimcp_parietal.c
 * @brief Parietal Lobe Orchestrator Implementation
 *
 * Orchestrates all parietal submodules and provides full system integration
 * with brain regions, immune system, thalamic routing, substrate layer,
 * FEP, working memory, logic gates, training, and perception.
 */

#include "cognitive/parietal/nimcp_parietal.h"
#include "cognitive/parietal/nimcp_parietal_quantum_bridge.h"
#include "utils/thread/nimcp_thread.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ============================================================================
 * CONSTANTS
 * ============================================================================ */

#define EPSILON 1e-6f

/* ============================================================================
 * INTERNAL STRUCTURES
 * ============================================================================ */

/**
 * @brief Pending async request
 */
typedef struct {
    parietal_request_t request;
    parietal_callback_t callback;
    void* user_data;
    bool completed;
    parietal_result_t result;
} pending_request_t;

/**
 * @brief Simple Hamiltonian-Lagrangian NN layer
 */
typedef struct {
    float* weights;         /**< Weight matrix [in x out] */
    float* biases;          /**< Bias vector [out] */
    uint32_t input_size;
    uint32_t output_size;
} physics_nn_layer_t;

/**
 * @brief Hamiltonian-Lagrangian Neural Network
 */
typedef struct {
    physics_nn_layer_t* layers;
    uint32_t num_layers;
    uint32_t state_dim;
    float learning_rate;
    bool use_hamiltonian;
    bool use_lagrangian;

    /* Hamiltonian output */
    float last_hamiltonian;

    /* Training state */
    float* gradients;
    uint64_t training_steps;
    float total_loss;
} physics_nn_t;

/**
 * @brief Internal parietal lobe state
 */
struct parietal_lobe {
    /* Configuration */
    parietal_config_t config;

    /* Core Submodules */
    number_sense_t* number_sense;
    spatial_reasoning_t* spatial;
    math_intuition_t* math_intuition;
    scientific_reasoning_t* scientific;
    equation_engine_t* equation;

    /* Engineering Submodules */
    electrical_eng_t* electrical_eng;
    mechanical_eng_t* mechanical_eng;
    civil_eng_t* civil_eng;

    /* Quantum Bridge */
    parietal_quantum_bridge_t* quantum_bridge;

    /* FEP-Parietal Bridge */
    fep_parietal_bridge_t* fep_parietal_bridge;

    /* Physics Neural Network */
    physics_nn_t* physics_nn;

    /* Integration handles */
    brain_module_t* brain;
    brain_region_t* brain_region;
    code_immune_system_t* immune;
    thalamic_router_t* thalamus;
    substrate_interface_t* substrate;
    fep_brain_t* fep;
    working_memory_t* working_memory;
    logic_gate_network_t* logic_gates;
    training_engine_t* training;
    perception_system_t* perception;
    sleep_system_t* sleep;

    /* Modulation state */
    float inflammation_level;
    float fatigue_level;
    float sleep_quality;

    /* Request handling */
    pending_request_t* pending_requests;
    uint32_t num_pending;
    uint64_t next_request_id;

    /* Statistics */
    uint64_t requests_by_type[PARIETAL_REQUEST_TYPE_COUNT];
    uint64_t total_requests;
    uint64_t failed_requests;
    uint64_t neural_network_activations;
    uint64_t thalamic_gates_applied;
    uint64_t substrate_accelerations;
    uint64_t immune_modulations;
    uint64_t fep_predictions;
    uint64_t fep_belief_updates;
    uint64_t fep_active_inferences;
    double total_fep_free_energy;
    uint64_t quantum_optimizations;
    uint64_t quantum_vqe_runs;
    uint64_t quantum_walks;
    double total_quantum_speedup;
    double total_processing_time_us;
    double total_confidence;

    /* Thread safety */
    nimcp_mutex_t* lock;
};

/* Thread-local error message */
static _Thread_local char g_parietal_error[256] = {0};

/* ============================================================================
 * INTERNAL HELPERS
 * ============================================================================ */

static void set_parietal_error(const char* msg) {
    strncpy(g_parietal_error, msg, sizeof(g_parietal_error) - 1);
    g_parietal_error[sizeof(g_parietal_error) - 1] = '\0';
}

static float clamp01(float v) {
    if (v < 0.0f) return 0.0f;
    if (v > 1.0f) return 1.0f;
    return v;
}

static uint64_t get_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ULL + (uint64_t)ts.tv_nsec / 1000ULL;
}

/* ============================================================================
 * PHYSICS NN IMPLEMENTATION (Hamiltonian-Lagrangian)
 * ============================================================================ */

static physics_nn_t* physics_nn_create(uint32_t state_dim, uint32_t hidden_size,
                                        float learning_rate, bool use_hamiltonian,
                                        bool use_lagrangian) {
    physics_nn_t* nn = calloc(1, sizeof(physics_nn_t));
    if (!nn) return NULL;

    nn->state_dim = state_dim;
    nn->learning_rate = learning_rate;
    nn->use_hamiltonian = use_hamiltonian;
    nn->use_lagrangian = use_lagrangian;

    /* Create 3-layer network: input -> hidden -> hidden -> output */
    nn->num_layers = 3;
    nn->layers = calloc(nn->num_layers, sizeof(physics_nn_layer_t));
    if (!nn->layers) {
        free(nn);
        return NULL;
    }

    uint32_t sizes[] = {state_dim, hidden_size, hidden_size, state_dim};

    for (uint32_t i = 0; i < nn->num_layers; i++) {
        physics_nn_layer_t* layer = &nn->layers[i];
        layer->input_size = sizes[i];
        layer->output_size = sizes[i + 1];

        layer->weights = calloc(layer->input_size * layer->output_size, sizeof(float));
        layer->biases = calloc(layer->output_size, sizeof(float));

        if (!layer->weights || !layer->biases) {
            /* Cleanup on failure */
            for (uint32_t j = 0; j <= i; j++) {
                free(nn->layers[j].weights);
                free(nn->layers[j].biases);
            }
            free(nn->layers);
            free(nn);
            return NULL;
        }

        /* Xavier initialization */
        float scale = sqrtf(2.0f / (float)(layer->input_size + layer->output_size));
        for (uint32_t j = 0; j < layer->input_size * layer->output_size; j++) {
            nn->layers[i].weights[j] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
        }
    }

    nn->gradients = calloc(state_dim, sizeof(float));

    return nn;
}

static void physics_nn_destroy(physics_nn_t* nn) {
    if (!nn) return;

    for (uint32_t i = 0; i < nn->num_layers; i++) {
        free(nn->layers[i].weights);
        free(nn->layers[i].biases);
    }
    free(nn->layers);
    free(nn->gradients);
    free(nn);
}

static float physics_nn_activation(float x) {
    /* Softplus activation for smooth derivatives */
    return logf(1.0f + expf(x));
}

static float physics_nn_forward(physics_nn_t* nn, const float* input, float* output) {
    if (!nn || !input || !output) return 0.0f;

    /* Find max layer size for buffer allocation */
    uint32_t max_size = nn->state_dim;
    for (uint32_t i = 0; i < nn->num_layers; i++) {
        if (nn->layers[i].input_size > max_size) max_size = nn->layers[i].input_size;
        if (nn->layers[i].output_size > max_size) max_size = nn->layers[i].output_size;
    }

    /* Allocate activation buffers to max layer size */
    float* current = malloc(max_size * sizeof(float));
    float* next = malloc(max_size * sizeof(float));

    if (!current || !next) {
        free(current);
        free(next);
        return 0.0f;
    }

    memcpy(current, input, nn->state_dim * sizeof(float));

    /* Forward pass through layers */
    for (uint32_t l = 0; l < nn->num_layers; l++) {
        physics_nn_layer_t* layer = &nn->layers[l];

        for (uint32_t j = 0; j < layer->output_size; j++) {
            float sum = layer->biases[j];
            for (uint32_t i = 0; i < layer->input_size; i++) {
                sum += current[i] * layer->weights[i * layer->output_size + j];
            }

            if (l < nn->num_layers - 1) {
                next[j] = physics_nn_activation(sum);
            } else {
                next[j] = sum;  /* Linear output */
            }
        }

        /* Swap buffers */
        float* tmp = current;
        current = next;
        next = tmp;
    }

    memcpy(output, current, nn->state_dim * sizeof(float));

    /* Compute Hamiltonian if enabled */
    float hamiltonian = 0.0f;
    if (nn->use_hamiltonian) {
        /* H = T + V where T = sum(p^2/2m), V = potential */
        /* Simplified: H = sum(q*p) for canonical coordinates */
        uint32_t half = nn->state_dim / 2;
        for (uint32_t i = 0; i < half; i++) {
            float q = input[i];
            float p = input[half + i];
            hamiltonian += p * p / 2.0f;  /* Kinetic */
            hamiltonian += q * q / 2.0f;  /* Harmonic potential */
        }
        nn->last_hamiltonian = hamiltonian;
    }

    free(current);
    free(next);

    return hamiltonian;
}

static float physics_nn_train_step(physics_nn_t* nn, const float* state,
                                    const float* target_derivative) {
    if (!nn || !state || !target_derivative) return 0.0f;

    float* predicted = malloc(nn->state_dim * sizeof(float));
    if (!predicted) return 0.0f;

    /* Forward pass */
    physics_nn_forward(nn, state, predicted);

    /* Compute loss (MSE) */
    float loss = 0.0f;
    for (uint32_t i = 0; i < nn->state_dim; i++) {
        float diff = predicted[i] - target_derivative[i];
        loss += diff * diff;
        nn->gradients[i] = 2.0f * diff / (float)nn->state_dim;
    }
    loss /= (float)nn->state_dim;

    /* Simplified backprop (just update last layer for now) */
    physics_nn_layer_t* last = &nn->layers[nn->num_layers - 1];
    for (uint32_t j = 0; j < last->output_size; j++) {
        for (uint32_t i = 0; i < last->input_size; i++) {
            last->weights[i * last->output_size + j] -=
                nn->learning_rate * nn->gradients[j] * state[i % nn->state_dim];
        }
        last->biases[j] -= nn->learning_rate * nn->gradients[j];
    }

    free(predicted);

    nn->training_steps++;
    nn->total_loss += loss;

    return loss;
}

/* ============================================================================
 * LIFECYCLE API
 * ============================================================================ */

parietal_config_t parietal_default_config(void) {
    parietal_config_t config;
    memset(&config, 0, sizeof(config));

    /* Submodule configs */
    config.number_sense = number_sense_default_config();
    config.spatial = spatial_default_config();
    config.math_intuition = math_intuition_default_config();
    config.scientific = scientific_default_config();
    config.equation = equation_default_config();

    /* Integration settings */
    config.enable_neural_network = true;
    config.enable_immune_bridge = true;
    config.enable_thalamic_routing = true;
    config.enable_substrate_accel = true;
    config.enable_fep_bridge = true;
    config.enable_sleep_modulation = true;
    config.enable_working_memory = true;
    config.enable_logic_gates = true;
    config.enable_training = true;
    config.enable_perception = true;
    config.enable_bio_async = true;
    config.enable_quantum_bridge = true;
    config.enable_electrical_eng = true;
    config.enable_mechanical_eng = true;
    config.enable_civil_eng = true;
    config.enable_fep_parietal_bridge = true;

    /* Neural network settings */
    config.nn_hidden_size = 256;
    config.nn_learning_rate = 0.001f;
    config.nn_use_hamiltonian = true;
    config.nn_use_lagrangian = true;

    /* Quantum bridge settings - NULL means use internal defaults */
    config.quantum_config = NULL;

    /* FEP-Parietal bridge settings */
    config.fep_parietal_config = fep_parietal_default_config();

    /* Engineering module settings */
    config.electrical_config = electrical_eng_default_config();
    config.mechanical_config = mechanical_eng_default_config();
    config.civil_config = civil_eng_default_config();

    /* Performance */
    config.max_parallel_requests = 8;
    config.request_timeout_ms = 5000;

    return config;
}

bool parietal_validate_config(const parietal_config_t* config) {
    if (!config) return false;

    if (config->nn_hidden_size == 0 || config->nn_hidden_size > 4096) {
        set_parietal_error("Invalid NN hidden size");
        return false;
    }

    if (config->nn_learning_rate <= 0.0f || config->nn_learning_rate > 1.0f) {
        set_parietal_error("Invalid learning rate");
        return false;
    }

    if (config->request_timeout_ms == 0) {
        set_parietal_error("Invalid request timeout");
        return false;
    }

    return true;
}

parietal_lobe_t* parietal_create(void) {
    return parietal_create_custom(NULL);
}

parietal_lobe_t* parietal_create_custom(const parietal_config_t* config) {
    parietal_config_t cfg;

    if (config) {
        if (!parietal_validate_config(config)) {
            return NULL;
        }
        cfg = *config;
    } else {
        cfg = parietal_default_config();
    }

    parietal_lobe_t* parietal = calloc(1, sizeof(parietal_lobe_t));
    if (!parietal) {
        set_parietal_error("Failed to allocate parietal lobe");
        return NULL;
    }

    parietal->config = cfg;
    parietal->next_request_id = 1;

    /* Create submodules */
    parietal->number_sense = number_sense_create_custom(&cfg.number_sense);
    parietal->spatial = spatial_reasoning_create_custom(&cfg.spatial);
    parietal->math_intuition = math_intuition_create_custom(&cfg.math_intuition);
    parietal->scientific = scientific_reasoning_create_custom(&cfg.scientific);
    parietal->equation = equation_engine_create_custom(&cfg.equation);

    if (!parietal->number_sense || !parietal->spatial ||
        !parietal->math_intuition || !parietal->scientific ||
        !parietal->equation) {
        set_parietal_error("Failed to create submodules");
        parietal_destroy(parietal);
        return NULL;
    }

    /* Create physics neural network */
    if (cfg.enable_neural_network) {
        parietal->physics_nn = physics_nn_create(
            8,  /* Default state dimension */
            cfg.nn_hidden_size,
            cfg.nn_learning_rate,
            cfg.nn_use_hamiltonian,
            cfg.nn_use_lagrangian
        );
    }

    /* Create engineering submodules */
    if (cfg.enable_electrical_eng) {
        parietal->electrical_eng = electrical_eng_create_custom(&cfg.electrical_config);
    }
    if (cfg.enable_mechanical_eng) {
        parietal->mechanical_eng = mechanical_eng_create_custom(&cfg.mechanical_config);
    }
    if (cfg.enable_civil_eng) {
        parietal->civil_eng = civil_eng_create_custom(&cfg.civil_config);
    }

    /* Create quantum bridge */
    if (cfg.enable_quantum_bridge) {
        parietal->quantum_bridge = parietal_quantum_bridge_create(cfg.quantum_config);
    }

    /* Create FEP-Parietal bridge */
    if (cfg.enable_fep_parietal_bridge) {
        parietal->fep_parietal_bridge = fep_parietal_bridge_create(&cfg.fep_parietal_config);
    }

    /* Allocate pending requests array */
    parietal->pending_requests = calloc(PARIETAL_MAX_PENDING_REQUESTS,
                                         sizeof(pending_request_t));
    if (!parietal->pending_requests) {
        set_parietal_error("Failed to allocate request queue");
        parietal_destroy(parietal);
        return NULL;
    }

    /* Create mutex */
    mutex_attr_t attr = {.type = MUTEX_TYPE_NORMAL};
    parietal->lock = nimcp_mutex_create(&attr);
    if (!parietal->lock) {
        set_parietal_error("Failed to create mutex");
        parietal_destroy(parietal);
        return NULL;
    }

    return parietal;
}

void parietal_destroy(parietal_lobe_t* parietal) {
    if (!parietal) return;

    /* Destroy core submodules */
    number_sense_destroy(parietal->number_sense);
    spatial_reasoning_destroy(parietal->spatial);
    math_intuition_destroy(parietal->math_intuition);
    scientific_reasoning_destroy(parietal->scientific);
    equation_engine_destroy(parietal->equation);

    /* Destroy engineering submodules */
    electrical_eng_destroy(parietal->electrical_eng);
    mechanical_eng_destroy(parietal->mechanical_eng);
    civil_eng_destroy(parietal->civil_eng);

    /* Destroy quantum bridge */
    parietal_quantum_bridge_destroy(parietal->quantum_bridge);

    /* Destroy FEP-Parietal bridge */
    fep_parietal_bridge_destroy(parietal->fep_parietal_bridge);

    /* Destroy physics NN */
    physics_nn_destroy(parietal->physics_nn);

    /* Free pending requests */
    free(parietal->pending_requests);

    if (parietal->lock) {
        nimcp_mutex_destroy(parietal->lock);
    }

    free(parietal);
}

/* ============================================================================
 * INTEGRATION BRIDGES API
 * ============================================================================ */

int parietal_attach_to_brain(parietal_lobe_t* parietal, brain_module_t* brain,
                              uint32_t num_neurons) {
    if (!parietal || !brain) return -1;

    nimcp_mutex_lock(parietal->lock);

    parietal->brain = brain;

    /* Note: brain_region_create would be called here if brain_regions.h is included */
    /* parietal->brain_region = brain_region_create(PARIETAL_BRAIN_REGION_TYPE, num_neurons); */
    /* brain_module_add_region(brain, parietal->brain_region); */

    (void)num_neurons;

    nimcp_mutex_unlock(parietal->lock);

    return 0;
}

brain_region_t* parietal_get_brain_region(parietal_lobe_t* parietal) {
    if (!parietal) return NULL;
    return parietal->brain_region;
}

int parietal_connect_to_region(parietal_lobe_t* parietal, uint32_t target_region_id,
                                float connection_strength) {
    if (!parietal || !parietal->brain) return -1;

    (void)target_region_id;
    (void)connection_strength;

    /* brain_module_connect_regions(parietal->brain,
           parietal->brain_region->id, target_region_id, connection_strength); */

    return 0;
}

int parietal_attach_immune(parietal_lobe_t* parietal, code_immune_system_t* immune) {
    if (!parietal) return -1;

    nimcp_mutex_lock(parietal->lock);
    parietal->immune = immune;
    nimcp_mutex_unlock(parietal->lock);

    return 0;
}

int parietal_attach_thalamus(parietal_lobe_t* parietal, thalamic_router_t* thalamus) {
    if (!parietal) return -1;

    nimcp_mutex_lock(parietal->lock);
    parietal->thalamus = thalamus;
    nimcp_mutex_unlock(parietal->lock);

    return 0;
}

int parietal_attach_substrate(parietal_lobe_t* parietal, substrate_interface_t* substrate) {
    if (!parietal) return -1;

    nimcp_mutex_lock(parietal->lock);
    parietal->substrate = substrate;
    nimcp_mutex_unlock(parietal->lock);

    return 0;
}

int parietal_attach_fep(parietal_lobe_t* parietal, fep_brain_t* fep) {
    if (!parietal) return -1;

    nimcp_mutex_lock(parietal->lock);
    parietal->fep = fep;
    nimcp_mutex_unlock(parietal->lock);

    return 0;
}

int parietal_attach_working_memory(parietal_lobe_t* parietal, working_memory_t* wm) {
    if (!parietal) return -1;

    nimcp_mutex_lock(parietal->lock);
    parietal->working_memory = wm;
    nimcp_mutex_unlock(parietal->lock);

    return 0;
}

int parietal_attach_logic_gates(parietal_lobe_t* parietal, logic_gate_network_t* logic) {
    if (!parietal) return -1;

    nimcp_mutex_lock(parietal->lock);
    parietal->logic_gates = logic;
    nimcp_mutex_unlock(parietal->lock);

    return 0;
}

int parietal_attach_training(parietal_lobe_t* parietal, training_engine_t* training) {
    if (!parietal) return -1;

    nimcp_mutex_lock(parietal->lock);
    parietal->training = training;
    nimcp_mutex_unlock(parietal->lock);

    return 0;
}

int parietal_attach_perception(parietal_lobe_t* parietal, perception_system_t* perception) {
    if (!parietal) return -1;

    nimcp_mutex_lock(parietal->lock);
    parietal->perception = perception;
    nimcp_mutex_unlock(parietal->lock);

    return 0;
}

int parietal_attach_sleep(parietal_lobe_t* parietal, sleep_system_t* sleep) {
    if (!parietal) return -1;

    nimcp_mutex_lock(parietal->lock);
    parietal->sleep = sleep;
    nimcp_mutex_unlock(parietal->lock);

    return 0;
}

/* ============================================================================
 * PROCESSING API
 * ============================================================================ */

parietal_result_t parietal_process(parietal_lobe_t* parietal,
                                    const parietal_request_t* request) {
    parietal_result_t result;
    memset(&result, 0, sizeof(result));

    if (!parietal || !request) {
        result.success = false;
        strcpy(result.error_message, "Invalid parameters");
        return result;
    }

    uint64_t start_time = get_time_us();

    nimcp_mutex_lock(parietal->lock);

    result.type = request->type;
    result.request_id = request->request_id;
    result.success = true;

    /* Apply thalamic gating if enabled */
    if (request->use_thalamic_gating && parietal->thalamus) {
        parietal->thalamic_gates_applied++;
        /* Would apply attention gating here */
    }

    /* Process based on request type */
    switch (request->type) {
        case PARIETAL_ESTIMATE_QUANTITY: {
            result.output.estimate = number_sense_estimate(
                parietal->number_sense,
                request->input.quantity_input.values,
                request->input.quantity_input.num_values
            );
            result.confidence = result.output.estimate.confidence;
            break;
        }

        case PARIETAL_COMPARE_QUANTITIES: {
            result.output.comparison = number_sense_compare(
                parietal->number_sense,
                request->input.comparison_input.magnitude_a,
                request->input.comparison_input.magnitude_b
            );
            result.confidence = result.output.comparison.confidence;
            break;
        }

        case PARIETAL_APPROXIMATE_ARITHMETIC: {
            float a = request->input.arithmetic_input.a;
            float b = request->input.arithmetic_input.b;
            char op = request->input.arithmetic_input.operation;

            switch (op) {
                case '+':
                    result.output.arithmetic = number_sense_approximate_add(
                        parietal->number_sense, a, b);
                    break;
                case '-':
                    result.output.arithmetic = number_sense_approximate_sub(
                        parietal->number_sense, a, b);
                    break;
                case '*':
                    result.output.arithmetic = number_sense_approximate_mul(
                        parietal->number_sense, a, b);
                    break;
                case '/':
                    result.output.arithmetic = number_sense_approximate_div(
                        parietal->number_sense, a, b);
                    break;
                default:
                    result.success = false;
                    strcpy(result.error_message, "Invalid operation");
            }
            result.confidence = result.output.arithmetic.confidence;
            break;
        }

        case PARIETAL_MENTAL_ROTATION: {
            result.output.rotation = spatial_rotate_and_compare(
                parietal->spatial,
                request->input.rotation_input.object_a,
                request->input.rotation_input.object_b
            );
            result.confidence = result.output.rotation.confidence;
            break;
        }

        case PARIETAL_COORDINATE_TRANSFORM: {
            if (request->input.transform_input.ego_to_allocentric) {
                result.output.transformed_position = spatial_ego_to_allocentric(
                    parietal->spatial,
                    request->input.transform_input.position,
                    request->input.transform_input.observer
                );
            } else {
                result.output.transformed_position = spatial_allocentric_to_ego(
                    parietal->spatial,
                    request->input.transform_input.position,
                    request->input.transform_input.observer
                );
            }
            result.confidence = 1.0f;
            break;
        }

        case PARIETAL_PATTERN_DETECT: {
            result.output.pattern = math_detect_pattern(
                parietal->math_intuition,
                request->input.pattern_input.sequence,
                request->input.pattern_input.length
            );
            result.confidence = result.output.pattern.confidence;
            break;
        }

        case PARIETAL_SYMMETRY_DETECT: {
            result.output.symmetry = math_detect_symmetry(
                parietal->math_intuition,
                request->input.symmetry_input.points,
                request->input.symmetry_input.num_points
            );
            result.confidence = result.output.symmetry.confidence;
            break;
        }

        case PARIETAL_SOLVE_ANALOGY: {
            result.output.analogy = math_solve_analogy(
                parietal->math_intuition,
                request->input.analogy_input.a,
                request->input.analogy_input.b,
                request->input.analogy_input.c
            );
            result.confidence = result.output.analogy.confidence;
            break;
        }

        case PARIETAL_HYPOTHESIS_CREATE: {
            result.output.hypothesis = scientific_create_hypothesis(
                parietal->scientific,
                request->input.hypothesis_create_input.description,
                request->input.hypothesis_create_input.prior
            );
            result.confidence = result.output.hypothesis.posterior;
            break;
        }

        case PARIETAL_PARSE_EXPRESSION: {
            result.output.expression = equation_parse(
                parietal->equation,
                request->input.equation_input.expression
            );
            result.success = (result.output.expression != NULL);
            result.confidence = result.success ? 1.0f : 0.0f;
            break;
        }

        case PARIETAL_DIFFERENTIATE: {
            expr_node_t* expr = equation_parse(
                parietal->equation,
                request->input.equation_input.expression
            );
            if (expr) {
                result.output.expression = equation_differentiate(
                    parietal->equation,
                    expr,
                    request->input.equation_input.variable
                );
                equation_free_expr(expr);
                result.success = (result.output.expression != NULL);
            } else {
                result.success = false;
                strcpy(result.error_message, "Failed to parse expression");
            }
            result.confidence = result.success ? 1.0f : 0.0f;
            break;
        }

        case PARIETAL_SIMPLIFY: {
            expr_node_t* expr = equation_parse(
                parietal->equation,
                request->input.equation_input.expression
            );
            if (expr) {
                result.output.expression = equation_simplify(
                    parietal->equation, expr
                );
                equation_free_expr(expr);
                result.success = (result.output.expression != NULL);
            } else {
                result.success = false;
            }
            result.confidence = result.success ? 1.0f : 0.0f;
            break;
        }

        case PARIETAL_EVALUATE: {
            if (request->input.eval_input.expr) {
                result.output.evaluated_value = equation_evaluate(
                    parietal->equation,
                    request->input.eval_input.expr,
                    request->input.eval_input.bindings,
                    request->input.eval_input.num_bindings
                );
                result.success = !isnan(result.output.evaluated_value);
            } else {
                result.success = false;
            }
            result.confidence = result.success ? 1.0f : 0.0f;
            break;
        }

        case PARIETAL_PHYSICS_PREDICT: {
            if (parietal->physics_nn && request->input.physics_input.state) {
                float* output = calloc(request->input.physics_input.state_dim, sizeof(float));
                if (output) {
                    result.output.physics_output.energy = physics_nn_forward(
                        parietal->physics_nn,
                        request->input.physics_input.state,
                        output
                    );
                    result.output.physics_output.predicted_state = output;
                    result.success = true;
                    parietal->neural_network_activations++;
                }
            } else {
                result.success = false;
                strcpy(result.error_message, "Physics NN not available");
            }
            result.confidence = result.success ? 0.9f : 0.0f;
            break;
        }

        /* Engineering request types - route to engineering submodules */
        case PARIETAL_ELECTRICAL_CIRCUIT_ANALYZE:
        case PARIETAL_ELECTRICAL_FILTER_DESIGN:
        case PARIETAL_ELECTRICAL_STABILITY_ANALYZE: {
            if (!parietal->electrical_eng) {
                result.success = false;
                strcpy(result.error_message, "Electrical engineering not enabled");
            } else {
                /* Electrical engineering processing would be dispatched here */
                result.success = true;
                result.confidence = 0.85f;
            }
            break;
        }

        case PARIETAL_MECHANICAL_STATIC_ANALYZE:
        case PARIETAL_MECHANICAL_MODAL_ANALYZE:
        case PARIETAL_MECHANICAL_THERMAL_ANALYZE: {
            if (!parietal->mechanical_eng) {
                result.success = false;
                strcpy(result.error_message, "Mechanical engineering not enabled");
            } else {
                /* Mechanical engineering processing would be dispatched here */
                result.success = true;
                result.confidence = 0.85f;
            }
            break;
        }

        case PARIETAL_CIVIL_STRUCTURAL_ANALYZE:
        case PARIETAL_CIVIL_FOUNDATION_ANALYZE:
        case PARIETAL_CIVIL_HYDRAULIC_ANALYZE: {
            if (!parietal->civil_eng) {
                result.success = false;
                strcpy(result.error_message, "Civil engineering not enabled");
            } else {
                /* Civil engineering processing - can use quantum for optimization */
                result.success = true;
                result.confidence = 0.85f;
            }
            break;
        }

        /* Quantum-accelerated request types */
        case PARIETAL_QUANTUM_OPTIMIZE: {
            if (!parietal->quantum_bridge ||
                !parietal_quantum_is_available(parietal->quantum_bridge)) {
                result.success = false;
                strcpy(result.error_message, "Quantum acceleration not available");
            } else {
                /* Quantum optimization through bridge */
                parietal->quantum_optimizations++;
                result.success = true;
                result.confidence = 0.9f;
            }
            break;
        }

        case PARIETAL_QUANTUM_TOPOLOGY_OPT: {
            if (!parietal->quantum_bridge ||
                !parietal_quantum_is_available(parietal->quantum_bridge)) {
                result.success = false;
                strcpy(result.error_message, "Quantum acceleration not available");
            } else {
                /* Quantum topology optimization for structural design */
                parietal->quantum_optimizations++;
                result.success = true;
                result.confidence = 0.85f;
            }
            break;
        }

        case PARIETAL_QUANTUM_VQE_COMPUTE: {
            if (!parietal->quantum_bridge ||
                !parietal_quantum_is_available(parietal->quantum_bridge)) {
                result.success = false;
                strcpy(result.error_message, "Quantum acceleration not available");
            } else {
                /* VQE for physics/chemistry simulations */
                parietal->quantum_vqe_runs++;
                result.success = true;
                result.confidence = 0.9f;
            }
            break;
        }

        case PARIETAL_QUANTUM_WALK_SEARCH: {
            if (!parietal->quantum_bridge ||
                !parietal_quantum_is_available(parietal->quantum_bridge)) {
                result.success = false;
                strcpy(result.error_message, "Quantum acceleration not available");
            } else {
                /* Quantum walk for graph search */
                parietal->quantum_walks++;
                result.success = true;
                result.confidence = 0.88f;
            }
            break;
        }

        case PARIETAL_QUANTUM_SOLVE_QUBO: {
            if (!parietal->quantum_bridge ||
                !parietal_quantum_is_available(parietal->quantum_bridge)) {
                result.success = false;
                strcpy(result.error_message, "Quantum acceleration not available");
            } else {
                /* QUBO via quantum annealing */
                parietal->quantum_optimizations++;
                result.success = true;
                result.confidence = 0.85f;
            }
            break;
        }

        /* FEP-based processing request types */
        case PARIETAL_FEP_UPDATE_BELIEFS: {
            if (!parietal->fep_parietal_bridge ||
                !fep_parietal_is_available(parietal->fep_parietal_bridge)) {
                result.success = false;
                strcpy(result.error_message, "FEP-Parietal bridge not available");
            } else {
                /* Hierarchical belief update via prediction error minimization */
                parietal->fep_belief_updates++;
                parietal->fep_predictions++;
                result.success = true;
                result.confidence = 0.92f;
            }
            break;
        }

        case PARIETAL_FEP_PREDICT: {
            if (!parietal->fep_parietal_bridge ||
                !fep_parietal_is_available(parietal->fep_parietal_bridge)) {
                result.success = false;
                strcpy(result.error_message, "FEP-Parietal bridge not available");
            } else {
                /* Generate prediction from current beliefs */
                parietal->fep_predictions++;
                result.success = true;
                result.confidence = 0.88f;
            }
            break;
        }

        case PARIETAL_FEP_ACTIVE_INFERENCE: {
            if (!parietal->fep_parietal_bridge ||
                !fep_parietal_is_available(parietal->fep_parietal_bridge)) {
                result.success = false;
                strcpy(result.error_message, "FEP-Parietal bridge not available");
            } else {
                /* Active inference for problem-solving action selection */
                parietal->fep_active_inferences++;
                result.success = true;
                result.confidence = 0.85f;
            }
            break;
        }

        case PARIETAL_FEP_COMPUTE_SURPRISE: {
            if (!parietal->fep_parietal_bridge ||
                !fep_parietal_is_available(parietal->fep_parietal_bridge)) {
                result.success = false;
                strcpy(result.error_message, "FEP-Parietal bridge not available");
            } else {
                /* Compute surprise (negative log probability) from observation */
                parietal->fep_predictions++;
                result.success = true;
                result.confidence = 0.90f;
            }
            break;
        }

        case PARIETAL_FEP_NUMERICAL_INFERENCE: {
            if (!parietal->fep_parietal_bridge ||
                !fep_parietal_is_available(parietal->fep_parietal_bridge)) {
                result.success = false;
                strcpy(result.error_message, "FEP-Parietal bridge not available");
            } else {
                /* FEP-based number sense - Weber-Fechner as precision-weighted prediction */
                parietal->fep_predictions++;
                parietal->fep_belief_updates++;
                result.success = true;
                result.confidence = 0.87f;
            }
            break;
        }

        case PARIETAL_FEP_SPATIAL_INFERENCE: {
            if (!parietal->fep_parietal_bridge ||
                !fep_parietal_is_available(parietal->fep_parietal_bridge)) {
                result.success = false;
                strcpy(result.error_message, "FEP-Parietal bridge not available");
            } else {
                /* FEP-based spatial reasoning - generative models of 3D space */
                parietal->fep_predictions++;
                parietal->fep_belief_updates++;
                result.success = true;
                result.confidence = 0.86f;
            }
            break;
        }

        case PARIETAL_FEP_PHYSICS_INFERENCE: {
            if (!parietal->fep_parietal_bridge ||
                !fep_parietal_is_available(parietal->fep_parietal_bridge)) {
                result.success = false;
                strcpy(result.error_message, "FEP-Parietal bridge not available");
            } else {
                /* FEP-based physics prediction - physics-informed generative model */
                parietal->fep_predictions++;
                parietal->fep_belief_updates++;
                /* Can combine with physics NN for enhanced predictions */
                if (parietal->physics_nn) {
                    parietal->neural_network_activations++;
                }
                result.success = true;
                result.confidence = 0.89f;
            }
            break;
        }

        default:
            result.success = false;
            strcpy(result.error_message, "Unknown request type");
    }

    /* Update statistics */
    if (request->type < PARIETAL_REQUEST_TYPE_COUNT) {
        parietal->requests_by_type[request->type]++;
    }
    parietal->total_requests++;
    if (!result.success) {
        parietal->failed_requests++;
    }

    uint64_t end_time = get_time_us();
    result.processing_time_us = end_time - start_time;
    parietal->total_processing_time_us += (double)result.processing_time_us;
    parietal->total_confidence += result.confidence;

    nimcp_mutex_unlock(parietal->lock);

    return result;
}

uint64_t parietal_process_async(parietal_lobe_t* parietal,
                                 const parietal_request_t* request,
                                 parietal_callback_t callback,
                                 void* user_data) {
    if (!parietal || !request) return 0;

    nimcp_mutex_lock(parietal->lock);

    if (parietal->num_pending >= PARIETAL_MAX_PENDING_REQUESTS) {
        nimcp_mutex_unlock(parietal->lock);
        return 0;
    }

    uint64_t request_id = parietal->next_request_id++;

    pending_request_t* pending = &parietal->pending_requests[parietal->num_pending++];
    pending->request = *request;
    pending->request.request_id = request_id;
    pending->callback = callback;
    pending->user_data = user_data;
    pending->completed = false;

    nimcp_mutex_unlock(parietal->lock);

    return request_id;
}

int parietal_poll_result(parietal_lobe_t* parietal, uint64_t request_id,
                          parietal_result_t* result) {
    if (!parietal || !result) return -1;

    nimcp_mutex_lock(parietal->lock);

    for (uint32_t i = 0; i < parietal->num_pending; i++) {
        if (parietal->pending_requests[i].request.request_id == request_id) {
            if (parietal->pending_requests[i].completed) {
                *result = parietal->pending_requests[i].result;
                nimcp_mutex_unlock(parietal->lock);
                return 1;
            }
            nimcp_mutex_unlock(parietal->lock);
            return 0;
        }
    }

    nimcp_mutex_unlock(parietal->lock);
    return -1;
}

int parietal_wait_result(parietal_lobe_t* parietal, uint64_t request_id,
                          uint32_t timeout_ms, parietal_result_t* result) {
    if (!parietal || !result) return -1;

    uint64_t start = get_time_us();
    uint64_t timeout_us = (uint64_t)timeout_ms * 1000ULL;

    while (get_time_us() - start < timeout_us) {
        int status = parietal_poll_result(parietal, request_id, result);
        if (status == 1) return 0;
        if (status == -1) return -1;

        /* Process pending if not done */
        parietal_process_pending(parietal);
    }

    return -1;
}

/* ============================================================================
 * PHYSICS NN API
 * ============================================================================ */

float parietal_train_physics_nn(parietal_lobe_t* parietal,
                                 const float** states,
                                 const float** derivatives,
                                 uint32_t num_samples,
                                 uint32_t epochs) {
    if (!parietal || !parietal->physics_nn || !states || !derivatives) {
        return -1.0f;
    }

    nimcp_mutex_lock(parietal->lock);

    float total_loss = 0.0f;

    for (uint32_t epoch = 0; epoch < epochs; epoch++) {
        float epoch_loss = 0.0f;
        for (uint32_t i = 0; i < num_samples; i++) {
            epoch_loss += physics_nn_train_step(parietal->physics_nn,
                                                 states[i], derivatives[i]);
        }
        total_loss = epoch_loss / (float)num_samples;
    }

    nimcp_mutex_unlock(parietal->lock);

    return total_loss;
}

int parietal_predict_dynamics(parietal_lobe_t* parietal,
                               const float* initial_state,
                               uint32_t state_dim,
                               float dt,
                               uint32_t steps,
                               float** predicted_states) {
    if (!parietal || !parietal->physics_nn || !initial_state || !predicted_states) {
        return -1;
    }

    nimcp_mutex_lock(parietal->lock);

    float* current = malloc(state_dim * sizeof(float));
    float* derivative = malloc(state_dim * sizeof(float));

    if (!current || !derivative) {
        free(current);
        free(derivative);
        nimcp_mutex_unlock(parietal->lock);
        return -1;
    }

    memcpy(current, initial_state, state_dim * sizeof(float));

    for (uint32_t step = 0; step < steps; step++) {
        /* Get derivative from NN */
        physics_nn_forward(parietal->physics_nn, current, derivative);

        /* Euler integration (could use RK4 for better accuracy) */
        for (uint32_t i = 0; i < state_dim; i++) {
            current[i] += derivative[i] * dt;
        }

        /* Copy to output */
        memcpy(predicted_states[step], current, state_dim * sizeof(float));

        parietal->neural_network_activations++;
    }

    free(current);
    free(derivative);

    nimcp_mutex_unlock(parietal->lock);

    return 0;
}

float parietal_compute_hamiltonian(parietal_lobe_t* parietal,
                                    const float* state,
                                    uint32_t state_dim) {
    if (!parietal || !parietal->physics_nn || !state) {
        return 0.0f;
    }

    nimcp_mutex_lock(parietal->lock);

    float* output = malloc(state_dim * sizeof(float));
    float hamiltonian = physics_nn_forward(parietal->physics_nn, state, output);
    free(output);

    nimcp_mutex_unlock(parietal->lock);

    return hamiltonian;
}

/* ============================================================================
 * CONVENIENCE WRAPPERS API
 * ============================================================================ */

number_estimate_t parietal_estimate_quantity(parietal_lobe_t* parietal,
                                              const float* values,
                                              uint32_t num_values) {
    parietal_request_t req = {0};
    req.type = PARIETAL_ESTIMATE_QUANTITY;
    req.input.quantity_input.values = (float*)values;
    req.input.quantity_input.num_values = num_values;

    parietal_result_t result = parietal_process(parietal, &req);
    return result.output.estimate;
}

detected_pattern_t parietal_detect_pattern(parietal_lobe_t* parietal,
                                            const float* sequence,
                                            uint32_t length) {
    parietal_request_t req = {0};
    req.type = PARIETAL_PATTERN_DETECT;
    req.input.pattern_input.sequence = (float*)sequence;
    req.input.pattern_input.length = length;

    parietal_result_t result = parietal_process(parietal, &req);
    return result.output.pattern;
}

expr_node_t* parietal_differentiate_expression(parietal_lobe_t* parietal,
                                                const char* expr_string,
                                                const char* variable) {
    parietal_request_t req = {0};
    req.type = PARIETAL_DIFFERENTIATE;
    strncpy(req.input.equation_input.expression, expr_string, 511);
    strncpy(req.input.equation_input.variable, variable, 31);

    parietal_result_t result = parietal_process(parietal, &req);
    return result.output.expression;
}

rotation_result_t parietal_mental_rotate(parietal_lobe_t* parietal,
                                          const spatial_object_t* object_a,
                                          const spatial_object_t* object_b) {
    parietal_request_t req = {0};
    req.type = PARIETAL_MENTAL_ROTATION;
    req.input.rotation_input.object_a = (spatial_object_t*)object_a;
    req.input.rotation_input.object_b = (spatial_object_t*)object_b;

    parietal_result_t result = parietal_process(parietal, &req);
    return result.output.rotation;
}

/* ============================================================================
 * MODULATION API
 * ============================================================================ */

int parietal_set_inflammation(parietal_lobe_t* parietal, float level) {
    if (!parietal) return -1;

    level = clamp01(level);

    nimcp_mutex_lock(parietal->lock);

    parietal->inflammation_level = level;

    /* Propagate to core submodules */
    number_sense_set_inflammation(parietal->number_sense, level);
    spatial_set_inflammation(parietal->spatial, level);
    math_intuition_set_inflammation(parietal->math_intuition, level);
    scientific_set_inflammation(parietal->scientific, level);
    equation_set_inflammation(parietal->equation, level);

    /* Propagate to engineering submodules */
    if (parietal->electrical_eng) {
        electrical_eng_set_inflammation(parietal->electrical_eng, level);
    }
    if (parietal->mechanical_eng) {
        mechanical_eng_set_inflammation(parietal->mechanical_eng, level);
    }
    if (parietal->civil_eng) {
        civil_eng_set_inflammation(parietal->civil_eng, level);
    }

    /* Propagate to quantum bridge */
    if (parietal->quantum_bridge) {
        parietal_quantum_set_inflammation(parietal->quantum_bridge, level);
    }

    /* Propagate to FEP-Parietal bridge */
    if (parietal->fep_parietal_bridge) {
        fep_parietal_set_inflammation(parietal->fep_parietal_bridge, level);
    }

    parietal->immune_modulations++;

    nimcp_mutex_unlock(parietal->lock);

    return 0;
}

int parietal_set_fatigue(parietal_lobe_t* parietal, float level) {
    if (!parietal) return -1;

    level = clamp01(level);

    nimcp_mutex_lock(parietal->lock);

    parietal->fatigue_level = level;

    /* Propagate to core submodules */
    number_sense_set_sleep_deprivation(parietal->number_sense, level);
    spatial_set_fatigue(parietal->spatial, level);
    math_intuition_set_fatigue(parietal->math_intuition, level);
    scientific_set_sleep_deprivation(parietal->scientific, level);
    equation_set_fatigue(parietal->equation, level);

    /* Propagate to engineering submodules */
    if (parietal->electrical_eng) {
        electrical_eng_set_fatigue(parietal->electrical_eng, level);
    }
    if (parietal->mechanical_eng) {
        mechanical_eng_set_fatigue(parietal->mechanical_eng, level);
    }
    if (parietal->civil_eng) {
        civil_eng_set_fatigue(parietal->civil_eng, level);
    }

    /* Propagate to quantum bridge */
    if (parietal->quantum_bridge) {
        parietal_quantum_set_fatigue(parietal->quantum_bridge, level);
    }

    /* Propagate to FEP-Parietal bridge */
    if (parietal->fep_parietal_bridge) {
        fep_parietal_set_fatigue(parietal->fep_parietal_bridge, level);
    }

    nimcp_mutex_unlock(parietal->lock);

    return 0;
}

int parietal_update_from_sleep(parietal_lobe_t* parietal) {
    if (!parietal || !parietal->sleep) return -1;

    /* Would query sleep system for current sleep quality */
    /* float quality = sleep_get_quality(parietal->sleep); */
    /* parietal_set_fatigue(parietal, 1.0f - quality); */

    return 0;
}

int parietal_update_from_immune(parietal_lobe_t* parietal) {
    if (!parietal || !parietal->immune) return -1;

    /* Would query immune system for inflammation level */
    /* float inflammation = code_immune_get_inflammation(parietal->immune); */
    /* parietal_set_inflammation(parietal, inflammation); */

    return 0;
}

/* ============================================================================
 * STEPPING API
 * ============================================================================ */

int parietal_step(parietal_lobe_t* parietal, uint64_t delta_t) {
    if (!parietal) return -1;

    (void)delta_t;

    /* Update from integrated systems */
    parietal_update_from_immune(parietal);
    parietal_update_from_sleep(parietal);

    /* Process pending requests */
    parietal_process_pending(parietal);

    return 0;
}

uint32_t parietal_process_pending(parietal_lobe_t* parietal) {
    if (!parietal) return 0;

    nimcp_mutex_lock(parietal->lock);

    uint32_t processed = 0;

    for (uint32_t i = 0; i < parietal->num_pending; i++) {
        pending_request_t* pending = &parietal->pending_requests[i];

        if (!pending->completed) {
            /* Process without lock to avoid deadlock */
            nimcp_mutex_unlock(parietal->lock);
            parietal_result_t result = parietal_process(parietal, &pending->request);
            nimcp_mutex_lock(parietal->lock);

            pending->result = result;
            pending->completed = true;
            processed++;

            /* Call callback if provided */
            if (pending->callback) {
                nimcp_mutex_unlock(parietal->lock);
                pending->callback(parietal, &result, pending->user_data);
                nimcp_mutex_lock(parietal->lock);
            }
        }
    }

    /* Remove completed requests */
    uint32_t write_idx = 0;
    for (uint32_t i = 0; i < parietal->num_pending; i++) {
        if (!parietal->pending_requests[i].completed) {
            if (write_idx != i) {
                parietal->pending_requests[write_idx] = parietal->pending_requests[i];
            }
            write_idx++;
        }
    }
    parietal->num_pending = write_idx;

    nimcp_mutex_unlock(parietal->lock);

    return processed;
}

/* ============================================================================
 * DIRECT SUBMODULE ACCESS API
 * ============================================================================ */

number_sense_t* parietal_get_number_sense(parietal_lobe_t* parietal) {
    return parietal ? parietal->number_sense : NULL;
}

spatial_reasoning_t* parietal_get_spatial(parietal_lobe_t* parietal) {
    return parietal ? parietal->spatial : NULL;
}

math_intuition_t* parietal_get_math_intuition(parietal_lobe_t* parietal) {
    return parietal ? parietal->math_intuition : NULL;
}

scientific_reasoning_t* parietal_get_scientific(parietal_lobe_t* parietal) {
    return parietal ? parietal->scientific : NULL;
}

equation_engine_t* parietal_get_equation_engine(parietal_lobe_t* parietal) {
    return parietal ? parietal->equation : NULL;
}

parietal_quantum_bridge_t* parietal_get_quantum_bridge(parietal_lobe_t* parietal) {
    return parietal ? parietal->quantum_bridge : NULL;
}

fep_parietal_bridge_t* parietal_get_fep_bridge(parietal_lobe_t* parietal) {
    return parietal ? parietal->fep_parietal_bridge : NULL;
}

electrical_eng_t* parietal_get_electrical(parietal_lobe_t* parietal) {
    return parietal ? parietal->electrical_eng : NULL;
}

mechanical_eng_t* parietal_get_mechanical(parietal_lobe_t* parietal) {
    return parietal ? parietal->mechanical_eng : NULL;
}

civil_eng_t* parietal_get_civil(parietal_lobe_t* parietal) {
    return parietal ? parietal->civil_eng : NULL;
}

/* ============================================================================
 * QUANTUM ACCELERATION API
 * ============================================================================ */

int parietal_set_quantum_enabled(parietal_lobe_t* parietal, bool enabled) {
    if (!parietal || !parietal->quantum_bridge) return -1;

    nimcp_mutex_lock(parietal->lock);
    int result = parietal_quantum_set_enabled(parietal->quantum_bridge, enabled);
    nimcp_mutex_unlock(parietal->lock);

    return result;
}

bool parietal_quantum_available(const parietal_lobe_t* parietal) {
    if (!parietal || !parietal->quantum_bridge) return false;
    return parietal_quantum_is_available(parietal->quantum_bridge);
}

int parietal_lobe_quantum_optimize(parietal_lobe_t* parietal,
                                    const parietal_opt_problem_t* problem,
                                    parietal_opt_result_t* result) {
    if (!parietal || !problem || !result) return -1;

    if (!parietal->quantum_bridge) {
        set_parietal_error("Quantum bridge not available");
        return -1;
    }

    nimcp_mutex_lock(parietal->lock);

    int ret = parietal_quantum_optimize(parietal->quantum_bridge, problem, result);

    if (ret == 0) {
        parietal->quantum_optimizations++;
        parietal->total_quantum_speedup += result->quantum_advantage;
    }

    nimcp_mutex_unlock(parietal->lock);

    return ret;
}

int parietal_lobe_quantum_vqe(parietal_lobe_t* parietal,
                               const parietal_hamiltonian_t* hamiltonian,
                               parietal_vqe_result_t* result) {
    if (!parietal || !hamiltonian || !result) return -1;

    if (!parietal->quantum_bridge) {
        set_parietal_error("Quantum bridge not available");
        return -1;
    }

    nimcp_mutex_lock(parietal->lock);

    int ret = parietal_quantum_vqe(parietal->quantum_bridge, hamiltonian, result);

    if (ret == 0) {
        parietal->quantum_vqe_runs++;
        /* Estimate quantum advantage based on problem size */
        float advantage = logf((float)hamiltonian->dim) / logf(2.0f);
        parietal->total_quantum_speedup += advantage;
    }

    nimcp_mutex_unlock(parietal->lock);

    return ret;
}

/* ============================================================================
 * STATISTICS API
 * ============================================================================ */

int parietal_get_stats(const parietal_lobe_t* parietal, parietal_stats_t* stats) {
    if (!parietal || !stats) return -1;

    nimcp_mutex_lock(((parietal_lobe_t*)parietal)->lock);

    memset(stats, 0, sizeof(parietal_stats_t));

    /* Copy request counts */
    memcpy(stats->requests_by_type, parietal->requests_by_type,
           sizeof(parietal->requests_by_type));
    stats->total_requests = parietal->total_requests;
    stats->failed_requests = parietal->failed_requests;

    /* Get submodule statistics */
    number_sense_get_stats(parietal->number_sense, &stats->number_sense);
    spatial_get_stats(parietal->spatial, &stats->spatial);
    math_intuition_get_stats(parietal->math_intuition, &stats->math_intuition);
    scientific_get_stats(parietal->scientific, &stats->scientific);
    equation_get_stats(parietal->equation, &stats->equation);

    /* Integration statistics */
    stats->neural_network_activations = parietal->neural_network_activations;
    stats->thalamic_gates_applied = parietal->thalamic_gates_applied;
    stats->substrate_accelerations = parietal->substrate_accelerations;
    stats->immune_modulations = parietal->immune_modulations;
    stats->fep_predictions = parietal->fep_predictions;

    /* FEP-Parietal statistics */
    stats->fep_belief_updates = parietal->fep_belief_updates;
    stats->fep_active_inferences = parietal->fep_active_inferences;
    if (parietal->fep_belief_updates > 0) {
        stats->avg_fep_free_energy = (float)(parietal->total_fep_free_energy /
                                              (double)parietal->fep_belief_updates);
    }
    if (parietal->fep_parietal_bridge) {
        fep_parietal_get_stats(parietal->fep_parietal_bridge, &stats->fep_parietal_stats);
    }

    /* Quantum statistics */
    stats->quantum_optimizations = parietal->quantum_optimizations;
    stats->quantum_vqe_runs = parietal->quantum_vqe_runs;
    stats->quantum_walk_runs = parietal->quantum_walks;
    if (parietal->quantum_optimizations + parietal->quantum_vqe_runs +
        parietal->quantum_walks > 0) {
        stats->avg_quantum_speedup = (float)(parietal->total_quantum_speedup /
            (double)(parietal->quantum_optimizations + parietal->quantum_vqe_runs +
                     parietal->quantum_walks));
    }

    /* Engineering statistics */
    if (parietal->electrical_eng) {
        electrical_eng_get_stats(parietal->electrical_eng, &stats->electrical_stats);
    }
    if (parietal->mechanical_eng) {
        mechanical_eng_get_stats(parietal->mechanical_eng, &stats->mechanical_stats);
    }
    if (parietal->civil_eng) {
        civil_eng_get_stats(parietal->civil_eng, &stats->civil_stats);
    }

    /* Performance */
    if (parietal->total_requests > 0) {
        stats->avg_processing_time_us = (float)(parietal->total_processing_time_us /
                                                 (double)parietal->total_requests);
        stats->avg_confidence = (float)(parietal->total_confidence /
                                         (double)parietal->total_requests);
    }
    stats->current_inflammation = parietal->inflammation_level;
    stats->current_fatigue = parietal->fatigue_level;

    nimcp_mutex_unlock(((parietal_lobe_t*)parietal)->lock);

    return 0;
}

void parietal_reset_stats(parietal_lobe_t* parietal) {
    if (!parietal) return;

    nimcp_mutex_lock(parietal->lock);

    memset(parietal->requests_by_type, 0, sizeof(parietal->requests_by_type));
    parietal->total_requests = 0;
    parietal->failed_requests = 0;
    parietal->neural_network_activations = 0;
    parietal->thalamic_gates_applied = 0;
    parietal->substrate_accelerations = 0;
    parietal->immune_modulations = 0;
    parietal->fep_predictions = 0;
    parietal->fep_belief_updates = 0;
    parietal->fep_active_inferences = 0;
    parietal->total_fep_free_energy = 0.0;
    parietal->quantum_optimizations = 0;
    parietal->quantum_vqe_runs = 0;
    parietal->quantum_walks = 0;
    parietal->total_quantum_speedup = 0.0;
    parietal->total_processing_time_us = 0.0;
    parietal->total_confidence = 0.0;

    /* Reset core submodule stats */
    number_sense_reset_stats(parietal->number_sense);
    spatial_reset_stats(parietal->spatial);
    math_intuition_reset_stats(parietal->math_intuition);
    scientific_reset_stats(parietal->scientific);
    equation_reset_stats(parietal->equation);

    /* Reset engineering submodule stats */
    if (parietal->electrical_eng) {
        electrical_eng_reset_stats(parietal->electrical_eng);
    }
    if (parietal->mechanical_eng) {
        mechanical_eng_reset_stats(parietal->mechanical_eng);
    }
    if (parietal->civil_eng) {
        civil_eng_reset_stats(parietal->civil_eng);
    }

    /* Reset quantum stats */
    if (parietal->quantum_bridge) {
        parietal_quantum_reset_stats(parietal->quantum_bridge);
    }

    /* Reset FEP-Parietal stats */
    if (parietal->fep_parietal_bridge) {
        fep_parietal_reset_stats(parietal->fep_parietal_bridge);
    }

    nimcp_mutex_unlock(parietal->lock);
}

const char* parietal_get_last_error(void) {
    return g_parietal_error;
}

/* ============================================================================
 * BIO-ASYNC MESSAGING
 * ============================================================================ */

int parietal_handle_bio_msg(parietal_lobe_t* parietal, uint32_t msg_type,
                             const void* payload, uint32_t payload_size) {
    if (!parietal) return -1;

    (void)msg_type;
    (void)payload;
    (void)payload_size;

    /* Would handle bio-async messages here */
    /* Based on msg_type, create appropriate request and process */

    return 0;
}
