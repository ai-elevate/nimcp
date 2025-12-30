/**
 * @file nimcp_cerebellum_adapter.c
 * @brief Implementation of Cerebellum brain adapter
 *
 * WHAT: Unified adapter connecting cerebellum sub-modules to the brain system
 * WHY:  Enable motor coordination, timing, and error-based learning
 * HOW:  Orchestrates granule cells, Purkinje cells, and deep nuclei
 *
 * @version Phase B4: Cerebellum Brain Integration
 * @date 2025-12-30
 */

#include "core/brain/regions/cerebellum/nimcp_cerebellum_adapter.h"
#include "async/nimcp_bio_messages.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/memory/nimcp_memory_pool.h"
#include "utils/logging/nimcp_logging.h"
#include <string.h>
#include <math.h>

/*=============================================================================
 * LOGGING MODULE IDENTIFIER
 *===========================================================================*/

#define CEREBELLUM_LOG_MODULE "CEREBELLUM"

/*=============================================================================
 * INTERNAL STRUCTURES
 *===========================================================================*/

/**
 * @brief Granule cell state
 */
typedef struct {
    float activation;                /**< Current activation [0, 1] */
    float* mossy_weights;            /**< Weights from mossy fibers */
    uint32_t num_mossy_inputs;       /**< Number of mossy inputs */
    bool active;                     /**< Currently firing */
} granule_cell_t;

/**
 * @brief Purkinje cell state
 */
typedef struct {
    float simple_spike_rate;         /**< Simple spike firing rate */
    float complex_spike;             /**< Complex spike flag */
    float* parallel_weights;         /**< Weights from parallel fibers */
    uint32_t num_parallel_inputs;    /**< Number of parallel inputs */
    float inhibition_output;         /**< Output to deep nuclei */
    float learning_eligibility;      /**< Eligibility trace for learning */
} purkinje_cell_t;

/**
 * @brief Deep nucleus neuron state
 */
typedef struct {
    float activity;                  /**< Neuron activity level */
    float* purkinje_weights;         /**< Weights from Purkinje cells */
    uint32_t num_purkinje_inputs;    /**< Number of Purkinje inputs */
    float motor_output[8];           /**< Motor command output */
} nucleus_neuron_t;

/**
 * @brief Granule layer (internal implementation)
 */
struct granule_layer {
    granule_cell_t* cells;           /**< Granule cells array */
    uint32_t num_cells;              /**< Number of cells */
    float* mossy_input_buffer;       /**< Input from mossy fibers */
    uint32_t num_mossy_fibers;       /**< Number of mossy fibers */
    float sparse_coding_threshold;   /**< Threshold for sparse activation */
};

/**
 * @brief Purkinje layer (internal implementation)
 */
struct purkinje_layer {
    purkinje_cell_t* cells;          /**< Purkinje cells array */
    uint32_t num_cells;              /**< Number of cells */
    float* parallel_input_buffer;    /**< Input from parallel fibers */
    float ltd_rate;                  /**< LTD learning rate */
    float ltp_rate;                  /**< LTP learning rate */
};

/**
 * @brief Deep nuclei (internal implementation)
 */
struct deep_nuclei {
    nucleus_neuron_t* dentate;       /**< Dentate nucleus neurons */
    uint32_t num_dentate;            /**< Number of dentate neurons */
    nucleus_neuron_t* interposed;    /**< Interposed nucleus neurons */
    uint32_t num_interposed;         /**< Number of interposed neurons */
    nucleus_neuron_t* fastigial;     /**< Fastigial nucleus neurons */
    uint32_t num_fastigial;          /**< Number of fastigial neurons */
    float* output_buffer;            /**< Output command buffer */
};

/**
 * @brief Climbing fiber system (internal implementation)
 */
struct climbing_fiber_system {
    float* error_signals;            /**< Current error signals */
    uint32_t num_fibers;             /**< Number of climbing fibers */
    uint32_t* purkinje_targets;      /**< Target Purkinje cells */
    float error_threshold;           /**< Threshold to trigger learning */
};

/**
 * @brief Forward model internal state
 */
typedef struct {
    float* weights;                  /**< Model weights */
    uint32_t input_dim;              /**< Input dimension */
    uint32_t output_dim;             /**< Output dimension */
    float learning_rate;             /**< Model learning rate */
    float last_prediction[8];        /**< Last prediction */
    float last_error;                /**< Last prediction error */
} forward_model_t;

/**
 * @brief Timing controller internal state
 */
typedef struct {
    float target_timing_ms;          /**< Target timing */
    float predicted_timing_ms;       /**< Predicted timing */
    float actual_timing_ms;          /**< Actual timing */
    float timing_error_history[16];  /**< Rolling error history */
    uint32_t error_history_idx;      /**< Current history index */
    float adaptation_rate;           /**< Timing adaptation rate */
} timing_controller_t;

/**
 * @brief Internal adapter structure
 */
struct cerebellum_adapter {
    /* Configuration */
    cerebellum_config_t config;

    /* Sub-modules */
    granule_layer_t* granule;
    purkinje_layer_t* purkinje;
    deep_nuclei_t* nuclei;
    climbing_fiber_system_t* climbing;

    /* Forward model */
    forward_model_t forward_model;

    /* Timing controller */
    timing_controller_t timing;

    /* Output buffers */
    nuclei_output_t* output_buffer;
    uint32_t output_count;
    uint32_t output_capacity;

    /* Callbacks */
    cerebellum_motor_callback_t motor_callback;
    void* motor_user_data;
    cerebellum_error_callback_t error_callback;
    void* error_user_data;
    cerebellum_event_callback_t event_callback;
    void* event_user_data;

    /* State */
    cerebellum_status_t status;
    cerebellum_error_t last_error;
    double current_time_ms;
    float adaptation_level;

    /* Bio-async communication context */
    bio_module_context_t bio_ctx;

    /* Statistics */
    cerebellum_stats_t stats;
};

/*=============================================================================
 * INTERNAL HELPERS
 *===========================================================================*/

/**
 * @brief Set error state
 */
static void set_error(cerebellum_adapter_t* adapter, cerebellum_error_t error) {
    if (!adapter) return;
    adapter->last_error = error;
    if (error != CEREBELLUM_ERROR_NONE) {
        adapter->status = CEREBELLUM_STATUS_ERROR;
        LOG_ERROR("[%s] Error set: %d", CEREBELLUM_LOG_MODULE, error);
    }
}

/**
 * @brief Emit event to callback
 */
static void emit_event(cerebellum_adapter_t* adapter, uint32_t event_type, const void* data) {
    if (adapter->config.enable_events && adapter->event_callback) {
        adapter->event_callback(event_type, data, adapter->event_user_data);
    }
}

/**
 * @brief Create granule layer
 */
static granule_layer_t* create_granule_layer(uint32_t num_cells, uint32_t num_mossy) {
    granule_layer_t* layer = nimcp_calloc(1, sizeof(granule_layer_t));
    if (!layer) return NULL;

    layer->num_cells = num_cells;
    layer->num_mossy_fibers = num_mossy;
    layer->sparse_coding_threshold = 0.5f;

    layer->cells = nimcp_calloc(num_cells, sizeof(granule_cell_t));
    if (!layer->cells) {
        nimcp_free(layer);
        return NULL;
    }

    layer->mossy_input_buffer = nimcp_calloc(num_mossy, sizeof(float));
    if (!layer->mossy_input_buffer) {
        nimcp_free(layer->cells);
        nimcp_free(layer);
        return NULL;
    }

    /* Initialize cells with random weights */
    for (uint32_t i = 0; i < num_cells; i++) {
        layer->cells[i].num_mossy_inputs = 4;  /* Each granule receives ~4 mossy inputs */
        layer->cells[i].mossy_weights = nimcp_calloc(4, sizeof(float));
        if (layer->cells[i].mossy_weights) {
            for (uint32_t j = 0; j < 4; j++) {
                layer->cells[i].mossy_weights[j] = 0.25f;  /* Uniform initial weights */
            }
        }
        layer->cells[i].activation = 0.0f;
        layer->cells[i].active = false;
    }

    return layer;
}

/**
 * @brief Destroy granule layer
 */
static void destroy_granule_layer(granule_layer_t* layer) {
    if (!layer) return;
    if (layer->cells) {
        for (uint32_t i = 0; i < layer->num_cells; i++) {
            if (layer->cells[i].mossy_weights) {
                nimcp_free(layer->cells[i].mossy_weights);
            }
        }
        nimcp_free(layer->cells);
    }
    if (layer->mossy_input_buffer) {
        nimcp_free(layer->mossy_input_buffer);
    }
    nimcp_free(layer);
}

/**
 * @brief Create Purkinje layer
 */
static purkinje_layer_t* create_purkinje_layer(uint32_t num_cells, uint32_t num_parallel,
                                                float ltd_rate, float ltp_rate) {
    purkinje_layer_t* layer = nimcp_calloc(1, sizeof(purkinje_layer_t));
    if (!layer) return NULL;

    layer->num_cells = num_cells;
    layer->ltd_rate = ltd_rate;
    layer->ltp_rate = ltp_rate;

    layer->cells = nimcp_calloc(num_cells, sizeof(purkinje_cell_t));
    if (!layer->cells) {
        nimcp_free(layer);
        return NULL;
    }

    layer->parallel_input_buffer = nimcp_calloc(num_parallel, sizeof(float));
    if (!layer->parallel_input_buffer) {
        nimcp_free(layer->cells);
        nimcp_free(layer);
        return NULL;
    }

    /* Initialize Purkinje cells */
    for (uint32_t i = 0; i < num_cells; i++) {
        uint32_t inputs_per_cell = (num_cells > 0) ? (num_parallel / num_cells) : num_parallel;
        if (inputs_per_cell < 10) inputs_per_cell = 10;

        layer->cells[i].num_parallel_inputs = inputs_per_cell;
        layer->cells[i].parallel_weights = nimcp_calloc(inputs_per_cell, sizeof(float));
        if (layer->cells[i].parallel_weights) {
            for (uint32_t j = 0; j < inputs_per_cell; j++) {
                layer->cells[i].parallel_weights[j] = 0.5f;  /* Initial weights */
            }
        }
        layer->cells[i].simple_spike_rate = 0.0f;
        layer->cells[i].complex_spike = 0.0f;
        layer->cells[i].inhibition_output = 0.0f;
        layer->cells[i].learning_eligibility = 0.0f;
    }

    return layer;
}

/**
 * @brief Destroy Purkinje layer
 */
static void destroy_purkinje_layer(purkinje_layer_t* layer) {
    if (!layer) return;
    if (layer->cells) {
        for (uint32_t i = 0; i < layer->num_cells; i++) {
            if (layer->cells[i].parallel_weights) {
                nimcp_free(layer->cells[i].parallel_weights);
            }
        }
        nimcp_free(layer->cells);
    }
    if (layer->parallel_input_buffer) {
        nimcp_free(layer->parallel_input_buffer);
    }
    nimcp_free(layer);
}

/**
 * @brief Create deep nuclei
 */
static deep_nuclei_t* create_deep_nuclei(uint32_t num_dentate, uint32_t num_interposed,
                                          uint32_t num_fastigial, uint32_t num_purkinje) {
    deep_nuclei_t* nuclei = nimcp_calloc(1, sizeof(deep_nuclei_t));
    if (!nuclei) return NULL;

    nuclei->num_dentate = num_dentate;
    nuclei->num_interposed = num_interposed;
    nuclei->num_fastigial = num_fastigial;

    /* Allocate dentate nucleus */
    nuclei->dentate = nimcp_calloc(num_dentate, sizeof(nucleus_neuron_t));
    if (!nuclei->dentate) {
        nimcp_free(nuclei);
        return NULL;
    }

    /* Allocate interposed nucleus */
    nuclei->interposed = nimcp_calloc(num_interposed, sizeof(nucleus_neuron_t));
    if (!nuclei->interposed) {
        nimcp_free(nuclei->dentate);
        nimcp_free(nuclei);
        return NULL;
    }

    /* Allocate fastigial nucleus */
    nuclei->fastigial = nimcp_calloc(num_fastigial, sizeof(nucleus_neuron_t));
    if (!nuclei->fastigial) {
        nimcp_free(nuclei->interposed);
        nimcp_free(nuclei->dentate);
        nimcp_free(nuclei);
        return NULL;
    }

    /* Allocate output buffer */
    uint32_t total_neurons = num_dentate + num_interposed + num_fastigial;
    nuclei->output_buffer = nimcp_calloc(total_neurons * 8, sizeof(float));

    /* Initialize neurons with Purkinje weights */
    uint32_t purkinje_per_neuron = num_purkinje / total_neurons;
    if (purkinje_per_neuron < 1) purkinje_per_neuron = 1;

    for (uint32_t i = 0; i < num_dentate; i++) {
        nuclei->dentate[i].num_purkinje_inputs = purkinje_per_neuron;
        nuclei->dentate[i].purkinje_weights = nimcp_calloc(purkinje_per_neuron, sizeof(float));
        if (nuclei->dentate[i].purkinje_weights) {
            for (uint32_t j = 0; j < purkinje_per_neuron; j++) {
                nuclei->dentate[i].purkinje_weights[j] = -1.0f;  /* Inhibitory */
            }
        }
    }

    for (uint32_t i = 0; i < num_interposed; i++) {
        nuclei->interposed[i].num_purkinje_inputs = purkinje_per_neuron;
        nuclei->interposed[i].purkinje_weights = nimcp_calloc(purkinje_per_neuron, sizeof(float));
        if (nuclei->interposed[i].purkinje_weights) {
            for (uint32_t j = 0; j < purkinje_per_neuron; j++) {
                nuclei->interposed[i].purkinje_weights[j] = -1.0f;
            }
        }
    }

    for (uint32_t i = 0; i < num_fastigial; i++) {
        nuclei->fastigial[i].num_purkinje_inputs = purkinje_per_neuron;
        nuclei->fastigial[i].purkinje_weights = nimcp_calloc(purkinje_per_neuron, sizeof(float));
        if (nuclei->fastigial[i].purkinje_weights) {
            for (uint32_t j = 0; j < purkinje_per_neuron; j++) {
                nuclei->fastigial[i].purkinje_weights[j] = -1.0f;
            }
        }
    }

    return nuclei;
}

/**
 * @brief Destroy deep nuclei
 */
static void destroy_deep_nuclei(deep_nuclei_t* nuclei) {
    if (!nuclei) return;

    if (nuclei->dentate) {
        for (uint32_t i = 0; i < nuclei->num_dentate; i++) {
            if (nuclei->dentate[i].purkinje_weights) {
                nimcp_free(nuclei->dentate[i].purkinje_weights);
            }
        }
        nimcp_free(nuclei->dentate);
    }

    if (nuclei->interposed) {
        for (uint32_t i = 0; i < nuclei->num_interposed; i++) {
            if (nuclei->interposed[i].purkinje_weights) {
                nimcp_free(nuclei->interposed[i].purkinje_weights);
            }
        }
        nimcp_free(nuclei->interposed);
    }

    if (nuclei->fastigial) {
        for (uint32_t i = 0; i < nuclei->num_fastigial; i++) {
            if (nuclei->fastigial[i].purkinje_weights) {
                nimcp_free(nuclei->fastigial[i].purkinje_weights);
            }
        }
        nimcp_free(nuclei->fastigial);
    }

    if (nuclei->output_buffer) {
        nimcp_free(nuclei->output_buffer);
    }

    nimcp_free(nuclei);
}

/**
 * @brief Create climbing fiber system
 */
static climbing_fiber_system_t* create_climbing_fibers(uint32_t num_fibers, uint32_t num_purkinje) {
    climbing_fiber_system_t* cf = nimcp_calloc(1, sizeof(climbing_fiber_system_t));
    if (!cf) return NULL;

    cf->num_fibers = num_fibers;
    cf->error_threshold = 0.1f;

    cf->error_signals = nimcp_calloc(num_fibers, sizeof(float));
    cf->purkinje_targets = nimcp_calloc(num_fibers, sizeof(uint32_t));

    if (!cf->error_signals || !cf->purkinje_targets) {
        if (cf->error_signals) nimcp_free(cf->error_signals);
        if (cf->purkinje_targets) nimcp_free(cf->purkinje_targets);
        nimcp_free(cf);
        return NULL;
    }

    /* Assign climbing fibers to Purkinje cells (1:1 in adult cerebellum) */
    for (uint32_t i = 0; i < num_fibers; i++) {
        cf->purkinje_targets[i] = i % num_purkinje;
    }

    return cf;
}

/**
 * @brief Destroy climbing fiber system
 */
static void destroy_climbing_fibers(climbing_fiber_system_t* cf) {
    if (!cf) return;
    if (cf->error_signals) nimcp_free(cf->error_signals);
    if (cf->purkinje_targets) nimcp_free(cf->purkinje_targets);
    nimcp_free(cf);
}

/**
 * @brief Process granule layer
 */
static void process_granule_layer(cerebellum_adapter_t* adapter) {
    if (!adapter || !adapter->granule) return;

    granule_layer_t* g = adapter->granule;
    uint32_t active_count = 0;

    /* Sparse coding: only ~5% of granule cells active */
    float max_activation = 0.0f;
    for (uint32_t i = 0; i < g->num_cells; i++) {
        /* Compute activation from mossy inputs */
        float sum = 0.0f;
        for (uint32_t j = 0; j < g->cells[i].num_mossy_inputs && j < g->num_mossy_fibers; j++) {
            uint32_t idx = (i * 4 + j) % g->num_mossy_fibers;
            sum += g->mossy_input_buffer[idx] * g->cells[i].mossy_weights[j];
        }

        g->cells[i].activation = sum;
        if (sum > max_activation) max_activation = sum;
    }

    /* Apply sparse threshold */
    float threshold = max_activation * g->sparse_coding_threshold;
    for (uint32_t i = 0; i < g->num_cells; i++) {
        g->cells[i].active = (g->cells[i].activation > threshold);
        if (g->cells[i].active) {
            active_count++;
            adapter->stats.granule_activations++;
        }
    }

    LOG_DEBUG("[%s] Granule layer: %u/%u cells active", CEREBELLUM_LOG_MODULE,
              active_count, g->num_cells);
}

/**
 * @brief Process Purkinje layer
 */
static void process_purkinje_layer(cerebellum_adapter_t* adapter) {
    if (!adapter || !adapter->purkinje || !adapter->granule) return;

    purkinje_layer_t* p = adapter->purkinje;
    granule_layer_t* g = adapter->granule;

    /* Transfer granule outputs to parallel fiber buffer */
    for (uint32_t i = 0; i < g->num_cells && i < p->num_cells * 10; i++) {
        p->parallel_input_buffer[i] = g->cells[i].active ? g->cells[i].activation : 0.0f;
    }

    /* Process each Purkinje cell */
    for (uint32_t i = 0; i < p->num_cells; i++) {
        /* Compute simple spike rate from parallel fiber inputs */
        float sum = 0.0f;
        for (uint32_t j = 0; j < p->cells[i].num_parallel_inputs; j++) {
            uint32_t idx = i * p->cells[i].num_parallel_inputs + j;
            if (idx < g->num_cells) {
                sum += p->parallel_input_buffer[idx] * p->cells[i].parallel_weights[j];
            }
        }

        /* Purkinje cells have high baseline firing (~50-100 Hz equivalent) */
        p->cells[i].simple_spike_rate = 0.5f + 0.5f * tanhf(sum);
        p->cells[i].inhibition_output = p->cells[i].simple_spike_rate;

        adapter->stats.purkinje_simple_spikes++;

        /* Update eligibility trace */
        p->cells[i].learning_eligibility *= 0.95f;  /* Decay */
        p->cells[i].learning_eligibility += sum * 0.1f;  /* Accumulate */
    }
}

/**
 * @brief Process deep nuclei
 */
static void process_deep_nuclei(cerebellum_adapter_t* adapter) {
    if (!adapter || !adapter->nuclei || !adapter->purkinje) return;

    deep_nuclei_t* n = adapter->nuclei;
    purkinje_layer_t* p = adapter->purkinje;

    /* Process dentate nucleus (lateral, planning) */
    for (uint32_t i = 0; i < n->num_dentate; i++) {
        float inhibition = 0.0f;
        for (uint32_t j = 0; j < n->dentate[i].num_purkinje_inputs && j < p->num_cells; j++) {
            inhibition += p->cells[j].inhibition_output * n->dentate[i].purkinje_weights[j];
        }
        /* Baseline excitation from mossy fiber collaterals, minus Purkinje inhibition */
        n->dentate[i].activity = fmaxf(0.0f, 0.8f + inhibition);

        /* Generate motor command (simplified) */
        for (uint32_t d = 0; d < 8; d++) {
            n->dentate[i].motor_output[d] = n->dentate[i].activity * 0.1f;
        }
    }

    /* Process interposed nucleus (intermediate, execution) */
    for (uint32_t i = 0; i < n->num_interposed; i++) {
        float inhibition = 0.0f;
        uint32_t offset = n->num_dentate;
        for (uint32_t j = 0; j < n->interposed[i].num_purkinje_inputs; j++) {
            uint32_t idx = (offset + j) % p->num_cells;
            inhibition += p->cells[idx].inhibition_output * n->interposed[i].purkinje_weights[j];
        }
        n->interposed[i].activity = fmaxf(0.0f, 0.8f + inhibition);
    }

    /* Process fastigial nucleus (medial, balance/posture) */
    for (uint32_t i = 0; i < n->num_fastigial; i++) {
        float inhibition = 0.0f;
        uint32_t offset = n->num_dentate + n->num_interposed;
        for (uint32_t j = 0; j < n->fastigial[i].num_purkinje_inputs; j++) {
            uint32_t idx = (offset + j) % p->num_cells;
            inhibition += p->cells[idx].inhibition_output * n->fastigial[i].purkinje_weights[j];
        }
        n->fastigial[i].activity = fmaxf(0.0f, 0.8f + inhibition);
    }
}

/**
 * @brief Apply LTD learning
 */
static void apply_ltd_learning(cerebellum_adapter_t* adapter, uint32_t purkinje_id, float error) {
    if (!adapter || !adapter->purkinje) return;

    purkinje_layer_t* p = adapter->purkinje;
    if (purkinje_id >= p->num_cells) return;

    purkinje_cell_t* cell = &p->cells[purkinje_id];

    /* Apply LTD: depress parallel fiber synapses that were active during error */
    for (uint32_t i = 0; i < cell->num_parallel_inputs; i++) {
        /* Weight change proportional to eligibility and error */
        float dw = -p->ltd_rate * error * cell->learning_eligibility;
        cell->parallel_weights[i] += dw;

        /* Clamp weights */
        if (cell->parallel_weights[i] < 0.0f) cell->parallel_weights[i] = 0.0f;
        if (cell->parallel_weights[i] > 1.0f) cell->parallel_weights[i] = 1.0f;
    }

    /* Reset eligibility after learning */
    cell->learning_eligibility = 0.0f;

    adapter->stats.ltd_events++;
}

/*=============================================================================
 * LIFECYCLE FUNCTIONS
 *===========================================================================*/

cerebellum_config_t cerebellum_default_config(void) {
    cerebellum_config_t config;
    memset(&config, 0, sizeof(config));

    config.num_granule_cells = CEREBELLUM_DEFAULT_GRANULE_CELLS;
    config.num_purkinje_cells = CEREBELLUM_DEFAULT_PURKINJE_CELLS;
    config.num_parallel_fibers = CEREBELLUM_DEFAULT_PARALLEL_FIBERS;
    config.num_climbing_fibers = CEREBELLUM_DEFAULT_CLIMBING_FIBERS;
    config.num_mossy_fibers = 100;

    config.num_dentate_neurons = 20;
    config.num_interposed_neurons = 10;
    config.num_fastigial_neurons = 10;

    config.enable_timing = true;
    config.enable_error_learning = true;
    config.enable_forward_models = true;
    config.enable_motor_adaptation = true;

    config.ltd_rate = CEREBELLUM_DEFAULT_LTD_RATE;
    config.ltp_rate = CEREBELLUM_DEFAULT_LTP_RATE;
    config.learning_rate = 0.01f;

    config.timing_resolution_ms = CEREBELLUM_DEFAULT_TIMING_RESOLUTION_MS;
    config.max_prediction_horizon_ms = 500.0f;

    config.enable_events = true;
    config.enable_bio_async = true;
    config.default_channel = BIO_CHANNEL_ACETYLCHOLINE;

    return config;
}

cerebellum_adapter_t* cerebellum_create(const cerebellum_config_t* config) {
    LOG_INFO("[%s] Creating cerebellum adapter", CEREBELLUM_LOG_MODULE);

    cerebellum_adapter_t* adapter = nimcp_calloc(1, sizeof(cerebellum_adapter_t));
    if (!adapter) {
        LOG_ERROR("[%s] Failed to allocate adapter memory", CEREBELLUM_LOG_MODULE);
        return NULL;
    }

    /* Set configuration */
    if (config) {
        adapter->config = *config;
        LOG_DEBUG("[%s] Using provided configuration", CEREBELLUM_LOG_MODULE);
    } else {
        adapter->config = cerebellum_default_config();
        LOG_DEBUG("[%s] Using default configuration", CEREBELLUM_LOG_MODULE);
    }

    /* Create granule layer */
    LOG_DEBUG("[%s] Creating granule layer (%u cells)", CEREBELLUM_LOG_MODULE,
              adapter->config.num_granule_cells);
    adapter->granule = create_granule_layer(adapter->config.num_granule_cells,
                                             adapter->config.num_mossy_fibers);
    if (!adapter->granule) {
        LOG_ERROR("[%s] Failed to create granule layer", CEREBELLUM_LOG_MODULE);
        cerebellum_destroy(adapter);
        return NULL;
    }

    /* Create Purkinje layer */
    LOG_DEBUG("[%s] Creating Purkinje layer (%u cells)", CEREBELLUM_LOG_MODULE,
              adapter->config.num_purkinje_cells);
    adapter->purkinje = create_purkinje_layer(adapter->config.num_purkinje_cells,
                                               adapter->config.num_parallel_fibers,
                                               adapter->config.ltd_rate,
                                               adapter->config.ltp_rate);
    if (!adapter->purkinje) {
        LOG_ERROR("[%s] Failed to create Purkinje layer", CEREBELLUM_LOG_MODULE);
        cerebellum_destroy(adapter);
        return NULL;
    }

    /* Create deep nuclei */
    LOG_DEBUG("[%s] Creating deep nuclei", CEREBELLUM_LOG_MODULE);
    adapter->nuclei = create_deep_nuclei(adapter->config.num_dentate_neurons,
                                          adapter->config.num_interposed_neurons,
                                          adapter->config.num_fastigial_neurons,
                                          adapter->config.num_purkinje_cells);
    if (!adapter->nuclei) {
        LOG_ERROR("[%s] Failed to create deep nuclei", CEREBELLUM_LOG_MODULE);
        cerebellum_destroy(adapter);
        return NULL;
    }

    /* Create climbing fiber system */
    LOG_DEBUG("[%s] Creating climbing fiber system", CEREBELLUM_LOG_MODULE);
    adapter->climbing = create_climbing_fibers(adapter->config.num_climbing_fibers,
                                                adapter->config.num_purkinje_cells);
    if (!adapter->climbing) {
        LOG_ERROR("[%s] Failed to create climbing fiber system", CEREBELLUM_LOG_MODULE);
        cerebellum_destroy(adapter);
        return NULL;
    }

    /* Initialize forward model */
    adapter->forward_model.input_dim = 8;
    adapter->forward_model.output_dim = 8;
    adapter->forward_model.learning_rate = adapter->config.learning_rate;
    adapter->forward_model.weights = nimcp_calloc(64, sizeof(float));
    if (!adapter->forward_model.weights) {
        LOG_ERROR("[%s] Failed to allocate forward model weights", CEREBELLUM_LOG_MODULE);
        cerebellum_destroy(adapter);
        return NULL;
    }
    /* Initialize as identity matrix */
    for (uint32_t i = 0; i < 8; i++) {
        adapter->forward_model.weights[i * 8 + i] = 1.0f;
    }

    /* Initialize timing controller */
    adapter->timing.target_timing_ms = 0.0f;
    adapter->timing.predicted_timing_ms = 0.0f;
    adapter->timing.adaptation_rate = 0.1f;

    /* Allocate output buffer */
    adapter->output_capacity = 32;
    adapter->output_buffer = nimcp_calloc(adapter->output_capacity, sizeof(nuclei_output_t));
    if (!adapter->output_buffer) {
        LOG_ERROR("[%s] Failed to allocate output buffer", CEREBELLUM_LOG_MODULE);
        cerebellum_destroy(adapter);
        return NULL;
    }

    /* Initialize bio-async communication */
    adapter->bio_ctx = NULL;
    if (adapter->config.enable_bio_async && bio_router_is_initialized()) {
        LOG_DEBUG("[%s] Registering with bio-async router", CEREBELLUM_LOG_MODULE);

        bio_module_info_t bio_info = {
            .module_id = BIO_MODULE_CEREBELLUM,
            .module_name = "cerebellum",
            .inbox_capacity = 64,
            .user_data = adapter
        };

        adapter->bio_ctx = bio_router_register_module(&bio_info);
        if (adapter->bio_ctx) {
            LOG_INFO("[%s] Bio-async registration successful", CEREBELLUM_LOG_MODULE);
        } else {
            LOG_WARNING("[%s] Failed to register with bio-async router", CEREBELLUM_LOG_MODULE);
        }
    }

    /* Initialize state */
    adapter->status = CEREBELLUM_STATUS_IDLE;
    adapter->last_error = CEREBELLUM_ERROR_NONE;
    adapter->current_time_ms = 0.0;
    adapter->adaptation_level = 0.0f;

    LOG_INFO("[%s] Cerebellum adapter created successfully", CEREBELLUM_LOG_MODULE);
    return adapter;
}

void cerebellum_destroy(cerebellum_adapter_t* adapter) {
    if (!adapter) return;

    LOG_INFO("[%s] Destroying cerebellum adapter", CEREBELLUM_LOG_MODULE);

    /* Unregister from bio-async router */
    if (adapter->bio_ctx) {
        LOG_DEBUG("[%s] Unregistering from bio-async router", CEREBELLUM_LOG_MODULE);
        bio_router_unregister_module(adapter->bio_ctx);
        adapter->bio_ctx = NULL;
    }

    /* Destroy sub-modules */
    destroy_granule_layer(adapter->granule);
    destroy_purkinje_layer(adapter->purkinje);
    destroy_deep_nuclei(adapter->nuclei);
    destroy_climbing_fibers(adapter->climbing);

    /* Free forward model */
    if (adapter->forward_model.weights) {
        nimcp_free(adapter->forward_model.weights);
    }

    /* Free output buffer */
    if (adapter->output_buffer) {
        nimcp_free(adapter->output_buffer);
    }

    LOG_DEBUG("[%s] Cerebellum adapter destroyed", CEREBELLUM_LOG_MODULE);
    nimcp_free(adapter);
}

bool cerebellum_reset(cerebellum_adapter_t* adapter) {
    if (!adapter) return false;

    LOG_DEBUG("[%s] Resetting adapter state", CEREBELLUM_LOG_MODULE);

    /* Reset granule layer */
    if (adapter->granule) {
        for (uint32_t i = 0; i < adapter->granule->num_cells; i++) {
            adapter->granule->cells[i].activation = 0.0f;
            adapter->granule->cells[i].active = false;
        }
        memset(adapter->granule->mossy_input_buffer, 0,
               adapter->granule->num_mossy_fibers * sizeof(float));
    }

    /* Reset Purkinje layer */
    if (adapter->purkinje) {
        for (uint32_t i = 0; i < adapter->purkinje->num_cells; i++) {
            adapter->purkinje->cells[i].simple_spike_rate = 0.5f;
            adapter->purkinje->cells[i].complex_spike = 0.0f;
            adapter->purkinje->cells[i].learning_eligibility = 0.0f;
        }
    }

    /* Reset output buffer */
    adapter->output_count = 0;

    /* Reset timing */
    adapter->timing.target_timing_ms = 0.0f;
    adapter->timing.predicted_timing_ms = 0.0f;
    memset(adapter->timing.timing_error_history, 0, sizeof(adapter->timing.timing_error_history));
    adapter->timing.error_history_idx = 0;

    /* Reset state */
    adapter->status = CEREBELLUM_STATUS_IDLE;
    adapter->last_error = CEREBELLUM_ERROR_NONE;

    LOG_DEBUG("[%s] Adapter reset complete", CEREBELLUM_LOG_MODULE);
    return true;
}

/*=============================================================================
 * MOSSY FIBER INPUT PROCESSING
 *===========================================================================*/

bool cerebellum_process_mossy_input(cerebellum_adapter_t* adapter,
                                     const mossy_fiber_input_t* input) {
    if (!adapter || !input) {
        set_error(adapter, CEREBELLUM_ERROR_INVALID_INPUT);
        return false;
    }

    if (!adapter->granule) {
        set_error(adapter, CEREBELLUM_ERROR_GRANULE_FAILURE);
        return false;
    }

    adapter->status = CEREBELLUM_STATUS_RECEIVING;

    /* Store input in mossy buffer */
    uint32_t idx = input->fiber_id % adapter->granule->num_mossy_fibers;
    adapter->granule->mossy_input_buffer[idx] = input->activity;

    adapter->stats.mossy_inputs_processed++;

    return true;
}

bool cerebellum_process_mossy_batch(cerebellum_adapter_t* adapter,
                                     const mossy_fiber_input_t* inputs,
                                     uint32_t count) {
    if (!adapter || !inputs || count == 0) {
        set_error(adapter, CEREBELLUM_ERROR_INVALID_INPUT);
        return false;
    }

    for (uint32_t i = 0; i < count; i++) {
        if (!cerebellum_process_mossy_input(adapter, &inputs[i])) {
            return false;
        }
    }

    return true;
}

/*=============================================================================
 * CLIMBING FIBER ERROR PROCESSING
 *===========================================================================*/

bool cerebellum_process_climbing_signal(cerebellum_adapter_t* adapter,
                                         const climbing_fiber_signal_t* signal) {
    if (!adapter || !signal) {
        set_error(adapter, CEREBELLUM_ERROR_INVALID_INPUT);
        return false;
    }

    if (!adapter->climbing || !adapter->purkinje) {
        set_error(adapter, CEREBELLUM_ERROR_LEARNING_FAILURE);
        return false;
    }

    adapter->status = CEREBELLUM_STATUS_LEARNING;

    /* Store error signal */
    uint32_t cf_idx = signal->fiber_id % adapter->climbing->num_fibers;
    adapter->climbing->error_signals[cf_idx] = signal->error_signal;

    /* Trigger complex spike in target Purkinje cell */
    uint32_t pc_idx = signal->target_purkinje_id % adapter->purkinje->num_cells;
    adapter->purkinje->cells[pc_idx].complex_spike = 1.0f;
    adapter->stats.purkinje_complex_spikes++;

    /* Apply LTD learning if error exceeds threshold */
    if (fabsf(signal->error_signal) > adapter->climbing->error_threshold) {
        if (adapter->config.enable_error_learning) {
            apply_ltd_learning(adapter, pc_idx, signal->error_signal);
        }
    }

    adapter->stats.climbing_signals++;

    return true;
}

bool cerebellum_broadcast_error(cerebellum_adapter_t* adapter,
                                 float error_magnitude,
                                 uint8_t error_type) {
    if (!adapter) {
        set_error(adapter, CEREBELLUM_ERROR_INVALID_INPUT);
        return false;
    }

    /* Create signal for each climbing fiber */
    for (uint32_t i = 0; i < adapter->climbing->num_fibers; i++) {
        climbing_fiber_signal_t signal;
        signal.fiber_id = i;
        signal.error_signal = error_magnitude;
        signal.timestamp_ms = (float)adapter->current_time_ms;
        signal.target_purkinje_id = adapter->climbing->purkinje_targets[i];
        signal.error_type = error_type;

        cerebellum_process_climbing_signal(adapter, &signal);
    }

    return true;
}

/*=============================================================================
 * MOTOR COORDINATION PIPELINE
 *===========================================================================*/

bool cerebellum_begin_coordination(cerebellum_adapter_t* adapter) {
    if (!adapter) return false;

    cerebellum_reset(adapter);
    adapter->status = CEREBELLUM_STATUS_IDLE;

    return true;
}

bool cerebellum_process(cerebellum_adapter_t* adapter,
                         motor_coordination_result_t* result) {
    if (!adapter) return false;

    adapter->status = CEREBELLUM_STATUS_PROCESSING;

    /* Process granule layer */
    process_granule_layer(adapter);

    /* Process Purkinje layer */
    process_purkinje_layer(adapter);

    /* Process deep nuclei */
    process_deep_nuclei(adapter);

    /* Timing computation */
    if (adapter->config.enable_timing) {
        adapter->status = CEREBELLUM_STATUS_TIMING;

        /* Predict timing based on Purkinje activity patterns */
        float timing_estimate = 0.0f;
        for (uint32_t i = 0; i < adapter->purkinje->num_cells; i++) {
            timing_estimate += adapter->purkinje->cells[i].simple_spike_rate;
        }
        timing_estimate = timing_estimate / adapter->purkinje->num_cells * 100.0f;  /* ms */

        adapter->timing.predicted_timing_ms = timing_estimate;
    }

    /* Fill result if provided */
    if (result) {
        memset(result, 0, sizeof(motor_coordination_result_t));

        result->predicted_timing_ms = adapter->timing.predicted_timing_ms;
        result->timing_error_ms = adapter->timing.predicted_timing_ms - adapter->timing.target_timing_ms;
        result->timing_correction = -result->timing_error_ms * 0.1f;

        /* Get motor gains from dentate nucleus */
        for (uint32_t i = 0; i < 8 && i < adapter->nuclei->num_dentate; i++) {
            result->motor_gain[i] = adapter->nuclei->dentate[i].activity;
        }
        result->num_motor_dims = 8;
        result->motor_ready = true;

        result->current_error = adapter->forward_model.last_error;
        result->learning_progress = 1.0f - fabsf(result->current_error);
        result->adaptation_active = adapter->config.enable_motor_adaptation;

        /* Forward model prediction */
        memcpy(result->predicted_outcome, adapter->forward_model.last_prediction,
               sizeof(result->predicted_outcome));
        result->confidence = 0.8f;  /* Base confidence */
    }

    adapter->status = CEREBELLUM_STATUS_OUTPUT_READY;
    adapter->stats.motor_commands_output++;

    emit_event(adapter, 1 /* CEREBELLUM_EVENT_PROCESSING_COMPLETE */, result);

    return true;
}

bool cerebellum_get_nuclei_output(cerebellum_adapter_t* adapter,
                                   nuclei_output_t* output) {
    if (!adapter || !output || !adapter->nuclei) return false;

    /* Aggregate output from all nuclei */
    memset(output, 0, sizeof(nuclei_output_t));

    output->nucleus_id = 0;  /* Combined output */
    output->timestamp_ms = (float)adapter->current_time_ms;

    /* Sum activity from all nuclei */
    float total_activity = 0.0f;

    for (uint32_t i = 0; i < adapter->nuclei->num_dentate; i++) {
        total_activity += adapter->nuclei->dentate[i].activity;
        for (uint32_t d = 0; d < 8; d++) {
            output->motor_command[d] += adapter->nuclei->dentate[i].motor_output[d];
        }
    }

    for (uint32_t i = 0; i < adapter->nuclei->num_interposed; i++) {
        total_activity += adapter->nuclei->interposed[i].activity;
    }

    for (uint32_t i = 0; i < adapter->nuclei->num_fastigial; i++) {
        total_activity += adapter->nuclei->fastigial[i].activity;
    }

    uint32_t total_neurons = adapter->nuclei->num_dentate +
                              adapter->nuclei->num_interposed +
                              adapter->nuclei->num_fastigial;
    output->activity = total_activity / total_neurons;
    output->num_dimensions = 8;

    /* Timing adjustment from Purkinje activity */
    output->timing_adjustment = adapter->timing.predicted_timing_ms - adapter->timing.target_timing_ms;

    /* Invoke callback if set */
    if (adapter->motor_callback) {
        adapter->motor_callback(output, adapter->motor_user_data);
    }

    return true;
}

/*=============================================================================
 * TIMING CONTROL
 *===========================================================================*/

bool cerebellum_set_target_timing(cerebellum_adapter_t* adapter, float target_ms) {
    if (!adapter) return false;

    adapter->timing.target_timing_ms = target_ms;
    return true;
}

bool cerebellum_predict_timing(cerebellum_adapter_t* adapter,
                                float* predicted_ms,
                                float* confidence) {
    if (!adapter || !predicted_ms) return false;

    *predicted_ms = adapter->timing.predicted_timing_ms;

    if (confidence) {
        /* Confidence based on history consistency */
        float variance = 0.0f;
        float mean = 0.0f;
        uint32_t count = 0;

        for (uint32_t i = 0; i < 16; i++) {
            if (adapter->timing.timing_error_history[i] != 0.0f) {
                mean += adapter->timing.timing_error_history[i];
                count++;
            }
        }
        if (count > 0) {
            mean /= count;
            for (uint32_t i = 0; i < 16 && count > 1; i++) {
                float diff = adapter->timing.timing_error_history[i] - mean;
                variance += diff * diff;
            }
            variance /= count;
        }

        /* Lower variance = higher confidence */
        *confidence = 1.0f / (1.0f + sqrtf(variance) / 10.0f);
    }

    return true;
}

bool cerebellum_report_timing(cerebellum_adapter_t* adapter, float actual_ms) {
    if (!adapter) return false;

    adapter->timing.actual_timing_ms = actual_ms;

    /* Compute timing error */
    float error = actual_ms - adapter->timing.target_timing_ms;

    /* Store in history */
    adapter->timing.timing_error_history[adapter->timing.error_history_idx] = error;
    adapter->timing.error_history_idx = (adapter->timing.error_history_idx + 1) % 16;

    /* Update stats */
    adapter->stats.avg_timing_error_ms = (adapter->stats.avg_timing_error_ms * 0.9f) +
                                          (fabsf(error) * 0.1f);
    if (fabsf(error) > adapter->stats.max_timing_error_ms) {
        adapter->stats.max_timing_error_ms = fabsf(error);
    }

    /* Trigger learning if error is significant */
    if (fabsf(error) > adapter->config.timing_resolution_ms) {
        /* Broadcast timing error as climbing fiber signal */
        if (adapter->config.enable_error_learning) {
            cerebellum_broadcast_error(adapter, error / 100.0f, 1 /* TIMING_ERROR */);
        }
    }

    return true;
}

/*=============================================================================
 * FORWARD MODEL OPERATIONS
 *===========================================================================*/

bool cerebellum_update_forward_model(cerebellum_adapter_t* adapter,
                                      const float* motor_command,
                                      const float* outcome,
                                      uint32_t num_dims) {
    if (!adapter || !motor_command || !outcome || num_dims == 0) return false;
    if (!adapter->config.enable_forward_models) return true;

    forward_model_t* fm = &adapter->forward_model;

    /* Predict outcome */
    float prediction[8] = {0};
    uint32_t dims = (num_dims < 8) ? num_dims : 8;

    for (uint32_t i = 0; i < dims; i++) {
        for (uint32_t j = 0; j < dims; j++) {
            prediction[i] += motor_command[j] * fm->weights[i * 8 + j];
        }
    }

    /* Compute error */
    float error = 0.0f;
    for (uint32_t i = 0; i < dims; i++) {
        float diff = outcome[i] - prediction[i];
        error += diff * diff;

        /* Update weights (gradient descent) */
        for (uint32_t j = 0; j < dims; j++) {
            fm->weights[i * 8 + j] += fm->learning_rate * diff * motor_command[j];
        }
    }

    fm->last_error = sqrtf(error / dims);
    memcpy(fm->last_prediction, prediction, sizeof(prediction));

    /* Update stats */
    adapter->stats.avg_error_after_learning = fm->last_error;

    return true;
}

bool cerebellum_predict_outcome(cerebellum_adapter_t* adapter,
                                 const float* motor_command,
                                 uint32_t num_dims,
                                 float* predicted_outcome,
                                 float* confidence) {
    if (!adapter || !motor_command || !predicted_outcome) return false;

    forward_model_t* fm = &adapter->forward_model;
    uint32_t dims = (num_dims < 8) ? num_dims : 8;

    /* Compute prediction */
    memset(predicted_outcome, 0, dims * sizeof(float));
    for (uint32_t i = 0; i < dims; i++) {
        for (uint32_t j = 0; j < dims; j++) {
            predicted_outcome[i] += motor_command[j] * fm->weights[i * 8 + j];
        }
    }

    if (confidence) {
        /* Confidence based on recent error */
        *confidence = 1.0f / (1.0f + fm->last_error);
    }

    memcpy(fm->last_prediction, predicted_outcome, dims * sizeof(float));

    return true;
}

/*=============================================================================
 * MOTOR ADAPTATION
 *===========================================================================*/

bool cerebellum_adapt_gains(cerebellum_adapter_t* adapter,
                             float* gains,
                             uint32_t num_dims,
                             float adaptation_rate) {
    if (!adapter || !gains || num_dims == 0) return false;
    if (!adapter->config.enable_motor_adaptation) return true;

    uint32_t dims = (num_dims < 8) ? num_dims : 8;

    /* Get adaptation signal from deep nuclei */
    for (uint32_t i = 0; i < dims && i < adapter->nuclei->num_dentate; i++) {
        float adaptation = (adapter->nuclei->dentate[i].activity - 0.5f) * adaptation_rate;
        gains[i] += adaptation;

        /* Clamp gains */
        if (gains[i] < 0.1f) gains[i] = 0.1f;
        if (gains[i] > 2.0f) gains[i] = 2.0f;
    }

    /* Update adaptation level */
    adapter->adaptation_level = (adapter->adaptation_level * 0.99f) + (adaptation_rate * 0.01f);

    return true;
}

bool cerebellum_get_adaptation_state(cerebellum_adapter_t* adapter,
                                      float* adaptation_level) {
    if (!adapter || !adaptation_level) return false;

    *adaptation_level = adapter->adaptation_level;
    return true;
}

/*=============================================================================
 * CALLBACK REGISTRATION
 *===========================================================================*/

bool cerebellum_set_motor_callback(cerebellum_adapter_t* adapter,
                                    cerebellum_motor_callback_t callback,
                                    void* user_data) {
    if (!adapter) return false;
    adapter->motor_callback = callback;
    adapter->motor_user_data = user_data;
    return true;
}

bool cerebellum_set_error_callback(cerebellum_adapter_t* adapter,
                                    cerebellum_error_callback_t callback,
                                    void* user_data) {
    if (!adapter) return false;
    adapter->error_callback = callback;
    adapter->error_user_data = user_data;
    return true;
}

bool cerebellum_set_event_callback(cerebellum_adapter_t* adapter,
                                    cerebellum_event_callback_t callback,
                                    void* user_data) {
    if (!adapter) return false;
    adapter->event_callback = callback;
    adapter->event_user_data = user_data;
    return true;
}

/*=============================================================================
 * STATUS AND DIAGNOSTICS
 *===========================================================================*/

cerebellum_status_t cerebellum_get_status(const cerebellum_adapter_t* adapter) {
    if (!adapter) return CEREBELLUM_STATUS_ERROR;
    return adapter->status;
}

cerebellum_error_t cerebellum_get_last_error(const cerebellum_adapter_t* adapter) {
    if (!adapter) return CEREBELLUM_ERROR_INTERNAL;
    return adapter->last_error;
}

const char* cerebellum_error_string(cerebellum_error_t error) {
    switch (error) {
        case CEREBELLUM_ERROR_NONE: return "No error";
        case CEREBELLUM_ERROR_INVALID_INPUT: return "Invalid input";
        case CEREBELLUM_ERROR_GRANULE_FAILURE: return "Granule layer failure";
        case CEREBELLUM_ERROR_PURKINJE_FAILURE: return "Purkinje layer failure";
        case CEREBELLUM_ERROR_TIMING_FAILURE: return "Timing computation failure";
        case CEREBELLUM_ERROR_LEARNING_FAILURE: return "Learning failure";
        case CEREBELLUM_ERROR_NUCLEI_FAILURE: return "Deep nuclei failure";
        case CEREBELLUM_ERROR_BUFFER_OVERFLOW: return "Buffer overflow";
        case CEREBELLUM_ERROR_INTERNAL: return "Internal error";
        default: return "Unknown error";
    }
}

const char* cerebellum_status_string(cerebellum_status_t status) {
    switch (status) {
        case CEREBELLUM_STATUS_IDLE: return "Idle";
        case CEREBELLUM_STATUS_RECEIVING: return "Receiving input";
        case CEREBELLUM_STATUS_PROCESSING: return "Processing";
        case CEREBELLUM_STATUS_TIMING: return "Timing computation";
        case CEREBELLUM_STATUS_LEARNING: return "Learning";
        case CEREBELLUM_STATUS_OUTPUT_READY: return "Output ready";
        case CEREBELLUM_STATUS_ERROR: return "Error";
        default: return "Unknown";
    }
}

bool cerebellum_get_stats(const cerebellum_adapter_t* adapter, cerebellum_stats_t* stats) {
    if (!adapter || !stats) return false;
    *stats = adapter->stats;
    return true;
}

bool cerebellum_get_config(const cerebellum_adapter_t* adapter, cerebellum_config_t* config) {
    if (!adapter || !config) return false;
    *config = adapter->config;
    return true;
}

/*=============================================================================
 * SUB-MODULE ACCESS
 *===========================================================================*/

granule_layer_t* cerebellum_get_granule_layer(cerebellum_adapter_t* adapter) {
    if (!adapter) return NULL;
    return adapter->granule;
}

purkinje_layer_t* cerebellum_get_purkinje_layer(cerebellum_adapter_t* adapter) {
    if (!adapter) return NULL;
    return adapter->purkinje;
}

deep_nuclei_t* cerebellum_get_deep_nuclei(cerebellum_adapter_t* adapter) {
    if (!adapter) return NULL;
    return adapter->nuclei;
}

/*=============================================================================
 * BIO-ASYNC COMMUNICATION
 *===========================================================================*/

bio_module_context_t cerebellum_get_bio_context(cerebellum_adapter_t* adapter) {
    if (!adapter) return NULL;
    return adapter->bio_ctx;
}

uint32_t cerebellum_process_bio_messages(cerebellum_adapter_t* adapter, uint32_t max_messages) {
    if (!adapter || !adapter->bio_ctx) return 0;

    uint32_t processed = bio_router_process_inbox(adapter->bio_ctx, max_messages);
    if (processed > 0) {
        LOG_DEBUG("[%s] Processed %u bio-async messages", CEREBELLUM_LOG_MODULE, processed);
    }
    return processed;
}

nimcp_bio_future_t cerebellum_request_coordination_async(
    cerebellum_adapter_t* adapter,
    const float* motor_command,
    uint32_t num_dims) {

    if (!adapter || !adapter->bio_ctx || !motor_command) {
        LOG_WARNING("[%s] Cannot request coordination: invalid arguments", CEREBELLUM_LOG_MODULE);
        return NULL;
    }

    LOG_DEBUG("[%s] Requesting motor coordination async", CEREBELLUM_LOG_MODULE);

    /* Create motor coordination request message */
    bio_msg_motor_command_request_t msg;
    memset(&msg, 0, sizeof(msg));

    msg.header.type = BIO_MSG_MOTOR_COMMAND_REQUEST;
    msg.header.source_module = BIO_MODULE_CEREBELLUM;
    msg.header.target_module = BIO_MODULE_MOTOR_CORTEX;
    msg.header.payload_size = sizeof(msg);
    msg.header.channel = adapter->config.default_channel;

    /* Store first motor command value */
    if (num_dims > 0) {
        msg.phoneme = (uint8_t)(motor_command[0] * 255.0f);  /* Reuse field */
    }

    nimcp_bio_promise_t promise = bio_router_send_async(
        adapter->bio_ctx, &msg, sizeof(msg), adapter->config.default_channel);

    if (!promise) {
        LOG_ERROR("[%s] Failed to send coordination request", CEREBELLUM_LOG_MODULE);
        return NULL;
    }

    return nimcp_bio_promise_get_future(promise);
}

nimcp_error_t cerebellum_broadcast_adaptation_complete(
    cerebellum_adapter_t* adapter,
    const motor_coordination_result_t* result) {

    if (!adapter || !result) {
        return NIMCP_BIO_ERROR_NOT_INITIALIZED;
    }

    if (!adapter->bio_ctx) {
        LOG_DEBUG("[%s] Cannot broadcast: bio-async not available", CEREBELLUM_LOG_MODULE);
        return NIMCP_SUCCESS;
    }

    LOG_INFO("[%s] Broadcasting adaptation complete (error=%.4f)",
             CEREBELLUM_LOG_MODULE, result->current_error);

    bio_msg_motor_command_result_t msg;
    memset(&msg, 0, sizeof(msg));

    msg.header.type = BIO_MSG_MOTOR_COMMAND_RESULT;
    msg.header.source_module = BIO_MODULE_CEREBELLUM;
    msg.header.target_module = 0;  /* Broadcast */
    msg.header.payload_size = sizeof(msg);
    msg.header.channel = adapter->config.default_channel;
    msg.header.flags = BIO_MSG_FLAG_BROADCAST;

    /* Store results in available fields */
    msg.lip_aperture = result->motor_gain[0];
    msg.tongue_height = result->motor_gain[1];
    msg.timestamp_ms = result->predicted_timing_ms;

    return bio_router_broadcast(adapter->bio_ctx, &msg, sizeof(msg));
}
