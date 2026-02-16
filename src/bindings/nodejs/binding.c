#include <node_api.h>
#include "security/nimcp_security.h"
#include "security/nimcp_blood_brain_barrier.h"

#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_router.h"

#include "nimcp.h"
#include "utils/metrics/nimcp_metrics.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/exception/nimcp_exception.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <stddef.h>  /* for NULL */
#include "utils/memory/nimcp_memory.h"
#include "constants/nimcp_learning_constants.h"
//=============================================================================
// Health Agent Integration (Phase 8: System-Wide Health Integration)
//=============================================================================
struct nimcp_health_agent;
typedef struct nimcp_health_agent nimcp_health_agent_t;
extern void nimcp_health_agent_heartbeat_ex(nimcp_health_agent_t* agent,
                                             const char* operation,
                                             float progress);

/** Global health agent for binding module */
static nimcp_health_agent_t* g_binding_health_agent = NULL;

/**
 * @brief Set health agent for binding heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
static void binding_set_health_agent(nimcp_health_agent_t* agent) {
    g_binding_health_agent = agent;
}

/** @brief Send heartbeat from binding module */
static inline void binding_heartbeat(const char* operation, float progress) {
    if (g_binding_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_binding_health_agent, operation, progress);
    }
}


// Wrapper for brain handle
typedef struct {
    nimcp_brain_t brain;
} BrainWrap;

// Wrapper for neural network handle
typedef struct {
    nimcp_network_t network;
} NetworkWrap;

// Wrapper for metrics collector handle
typedef struct {
    nimcp_metrics_collector_t collector;
} MetricsCollectorWrap;

// Destructor for Brain
static void brain_finalize(napi_env env, void* finalize_data, void* finalize_hint) {
    BrainWrap* wrap = (BrainWrap*)finalize_data;
    if (wrap && wrap->brain) {
        nimcp_brain_destroy(wrap->brain);
    }
    nimcp_free(wrap);
}

// Destructor for Network
static void network_finalize(napi_env env, void* finalize_data, void* finalize_hint) {
    NetworkWrap* wrap = (NetworkWrap*)finalize_data;
    if (wrap && wrap->network) {
        nimcp_network_destroy(wrap->network);
    }
    nimcp_free(wrap);
}

// Destructor for MetricsCollector
static void metrics_collector_finalize(napi_env env, void* finalize_data, void* finalize_hint) {
    MetricsCollectorWrap* wrap = (MetricsCollectorWrap*)finalize_data;
    if (wrap && wrap->collector) {
        nimcp_metrics_destroy(wrap->collector);
    }
    nimcp_free(wrap);
}

// Create neural network: new NeuralNetwork(config)
static napi_value CreateNeuralNetwork(napi_env env, napi_callback_info info) {
    napi_status status;
    size_t argc = 1;
    napi_value args[1];
    napi_value this_arg;

    status = napi_get_cb_info(env, info, &argc, args, &this_arg, NULL);
    if (status != napi_ok) {
        NIMCP_THROW(NIMCP_ERROR_INVALID_PARAM, "CreateNeuralNetwork: Failed to get callback info");
        return NULL;
    }

    // Parse config object
    uint32_t num_inputs = 0;
    uint32_t num_outputs = 0;
    uint32_t num_hidden = 0;
    float learning_rate = NIMCP_LEARNING_RATE_DEFAULT;

    // Get num_inputs
    napi_value num_inputs_val;
    status = napi_get_named_property(env, args[0], "num_inputs", &num_inputs_val);
    if (status == napi_ok) {
        napi_get_value_uint32(env, num_inputs_val, &num_inputs);
    }

    // Get num_outputs
    napi_value num_outputs_val;
    status = napi_get_named_property(env, args[0], "num_outputs", &num_outputs_val);
    if (status == napi_ok) {
        napi_get_value_uint32(env, num_outputs_val, &num_outputs);
    }

    // Get num_hidden
    napi_value num_hidden_val;
    status = napi_get_named_property(env, args[0], "num_hidden", &num_hidden_val);
    if (status == napi_ok) {
        napi_get_value_uint32(env, num_hidden_val, &num_hidden);
    }

    // Get learning_rate
    napi_value learning_rate_val;
    status = napi_get_named_property(env, args[0], "learning_rate", &learning_rate_val);
    if (status == napi_ok) {
        double lr;
        napi_get_value_double(env, learning_rate_val, &lr);
        learning_rate = (float)lr;
    }

    // Create neural network using unified API
    nimcp_network_t network = nimcp_network_create(num_inputs, num_outputs, num_hidden, learning_rate);
    if (!network) {
        NIMCP_THROW_BRAIN(NIMCP_ERROR_NOT_INITIALIZED, 0, "nodejs_binding",
                         "CreateNeuralNetwork: Failed to create network (%u inputs, %u outputs)",
                         num_inputs, num_outputs);
        napi_throw_error(env, NULL, nimcp_get_error());
        return NULL;
    }

    // Wrap the handle
    NetworkWrap* wrap = (NetworkWrap*)nimcp_malloc(sizeof(NetworkWrap));
    if (!wrap) {
        NIMCP_THROW_MEMORY(NIMCP_ERROR_NO_MEMORY, sizeof(NetworkWrap),
                          "CreateNeuralNetwork: Failed to allocate NetworkWrap");
        nimcp_network_destroy(network);
        return NULL;
    }
    wrap->network = network;

    status = napi_wrap(env, this_arg, wrap, network_finalize, NULL, NULL);
    if (status != napi_ok) {
        NIMCP_THROW(NIMCP_ERROR_OPERATION_FAILED, "CreateNeuralNetwork: Failed to wrap network object");
        nimcp_free(wrap);
        nimcp_network_destroy(network);
        return NULL;
    }

    return this_arg;
}

// Forward pass: network.forward(inputs)
static napi_value Forward(napi_env env, napi_callback_info info) {
    napi_status status;
    size_t argc = 1;
    napi_value args[1];
    napi_value this_arg;

    status = napi_get_cb_info(env, info, &argc, args, &this_arg, NULL);
    if (status != napi_ok) {
        NIMCP_THROW(NIMCP_ERROR_INVALID_PARAM, "Forward: Failed to get callback info");
        return NULL;
    }

    // Unwrap the neural network
    NetworkWrap* wrap;
    status = napi_unwrap(env, this_arg, (void**)&wrap);
    if (status != napi_ok || !wrap || !wrap->network) {
        NIMCP_THROW(NIMCP_ERROR_NOT_INITIALIZED, "Forward: Invalid neural network object");
        napi_throw_error(env, NULL, "Invalid neural network object");
        return NULL;
    }

    // Get input array length
    uint32_t num_inputs;
    napi_get_array_length(env, args[0], &num_inputs);

    // Allocate and fill input array
    float* inputs = (float*)nimcp_malloc(num_inputs * sizeof(float));
    if (!inputs) {
        NIMCP_THROW_MEMORY(NIMCP_ERROR_NO_MEMORY, num_inputs * sizeof(float),
                          "Forward: Failed to allocate memory for inputs");
        napi_throw_error(env, NULL, "Failed to allocate memory for inputs");
        return NULL;
    }
    for (uint32_t i = 0; i < num_inputs; i++) {
        napi_value element;
        napi_get_element(env, args[0], i, &element);
        double value;
        napi_get_value_double(env, element, &value);
        inputs[i] = (float)value;
    }

    // Allocate output array (assume same size as input for now)
    uint32_t num_outputs = num_inputs;  // TODO: get actual output size from network
    float* outputs = (float*)nimcp_malloc(num_outputs * sizeof(float));
    if (!outputs) {
        nimcp_free(inputs);
        NIMCP_THROW_MEMORY(NIMCP_ERROR_NO_MEMORY, num_outputs * sizeof(float),
                          "Forward: Failed to allocate memory for outputs");
        napi_throw_error(env, NULL, "Failed to allocate memory for outputs");
        return NULL;
    }

    // Forward pass using unified API
    nimcp_status_t result = nimcp_network_forward(wrap->network, inputs, num_inputs, outputs, num_outputs);

    nimcp_free(inputs);

    if (result != NIMCP_OK) {
        nimcp_free(outputs);
        NIMCP_THROW_BRAIN(NIMCP_ERROR_OPERATION_FAILED, 0, "nodejs_binding",
                         "Forward: Network forward pass failed");
        napi_throw_error(env, NULL, nimcp_get_error());
        return NULL;
    }

    // Create JavaScript array for outputs
    napi_value result_array;
    napi_create_array_with_length(env, num_outputs, &result_array);

    for (uint32_t i = 0; i < num_outputs; i++) {
        napi_value num;
        napi_create_double(env, outputs[i], &num);
        napi_set_element(env, result_array, i, num);
    }

    nimcp_free(outputs);
    return result_array;
}

// Create metrics collector: new MetricsCollector(config)
static napi_value CreateMetricsCollector(napi_env env, napi_callback_info info) {
    napi_status status;
    size_t argc = 1;
    napi_value args[1];
    napi_value this_arg;

    status = napi_get_cb_info(env, info, &argc, args, &this_arg, NULL);
    if (status != napi_ok) {
        NIMCP_THROW(NIMCP_ERROR_INVALID_PARAM, "CreateMetricsCollector: Failed to get callback info");
        return NULL;
    }

    // Create metrics collector
    nimcp_metrics_collector_t collector = nimcp_metrics_create();
    if (!collector) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY,
                             "CreateMetricsCollector: Failed to create metrics collector");
        napi_throw_error(env, NULL, "Failed to create metrics collector");
        return NULL;
    }

    // Parse config if provided
    if (argc > 0) {
        napi_valuetype valuetype;
        napi_typeof(env, args[0], &valuetype);

        if (valuetype == napi_object) {
            // Get directory
            napi_value directory_val;
            status = napi_get_named_property(env, args[0], "directory", &directory_val);
            if (status == napi_ok) {
                size_t str_len;
                napi_get_value_string_utf8(env, directory_val, NULL, 0, &str_len);
                char* directory = (char*)nimcp_malloc(str_len + 1);
                if (!directory) {
                    nimcp_metrics_destroy(collector);
                    NIMCP_THROW_MEMORY(NIMCP_ERROR_NO_MEMORY, str_len + 1,
                                      "CreateMetricsCollector: Failed to allocate memory for directory");
                    napi_throw_error(env, NULL, "Failed to allocate memory for directory");
                    return NULL;
                }
                napi_get_value_string_utf8(env, directory_val, directory, str_len + 1, &str_len);
                nimcp_metrics_set_directory(collector, directory);
                nimcp_free(directory);
            }
        }
    }

    // Wrap the handle
    MetricsCollectorWrap* wrap = (MetricsCollectorWrap*)nimcp_malloc(sizeof(MetricsCollectorWrap));
    if (!wrap) {
        NIMCP_THROW_MEMORY(NIMCP_ERROR_NO_MEMORY, sizeof(MetricsCollectorWrap),
                          "CreateMetricsCollector: Failed to allocate MetricsCollectorWrap");
        nimcp_metrics_destroy(collector);
        return NULL;
    }
    wrap->collector = collector;

    status = napi_wrap(env, this_arg, wrap, metrics_collector_finalize, NULL, NULL);
    if (status != napi_ok) {
        NIMCP_THROW(NIMCP_ERROR_OPERATION_FAILED,
                   "CreateMetricsCollector: Failed to wrap metrics collector object");
        nimcp_free(wrap);
        nimcp_metrics_destroy(collector);
        return NULL;
    }

    return this_arg;
}

// Record counter: collector.recordCounter(name, value, category)
static napi_value RecordCounter(napi_env env, napi_callback_info info) {
    napi_status status;
    size_t argc = 3;
    napi_value args[3];
    napi_value this_arg;

    status = napi_get_cb_info(env, info, &argc, args, &this_arg, NULL);
    if (status != napi_ok || argc < 2) {
        NIMCP_THROW(NIMCP_ERROR_INVALID_PARAM, "RecordCounter: Invalid arguments");
        return NULL;
    }

    // Unwrap the metrics collector
    MetricsCollectorWrap* wrap;
    status = napi_unwrap(env, this_arg, (void**)&wrap);
    if (status != napi_ok || !wrap || !wrap->collector) {
        NIMCP_THROW(NIMCP_ERROR_NOT_INITIALIZED, "RecordCounter: Invalid metrics collector object");
        napi_throw_error(env, NULL, "Invalid metrics collector object");
        return NULL;
    }

    // Get name
    size_t str_len;
    napi_get_value_string_utf8(env, args[0], NULL, 0, &str_len);
    char* name = (char*)nimcp_malloc(str_len + 1);
    if (!name) {
        NIMCP_THROW_MEMORY(NIMCP_ERROR_NO_MEMORY, str_len + 1,
                          "RecordCounter: Failed to allocate memory for name");
        napi_throw_error(env, NULL, "Failed to allocate memory for name");
        return NULL;
    }
    napi_get_value_string_utf8(env, args[0], name, str_len + 1, &str_len);

    // Get value
    int64_t value;
    napi_get_value_int64(env, args[1], &value);

    // Get category (optional, default to CUSTOM)
    int32_t category = NIMCP_METRIC_CATEGORY_CUSTOM;
    if (argc >= 3) {
        napi_get_value_int32(env, args[2], &category);
    }

    // Record counter
    bool success = nimcp_metrics_record_counter(wrap->collector, name, (uint64_t)value,
                                                  (nimcp_metric_category_t)category);
    nimcp_free(name);

    if (!success) {
        NIMCP_THROW(NIMCP_ERROR_OPERATION_FAILED, "RecordCounter: Failed to record counter");
        napi_throw_error(env, NULL, "Failed to record counter");
        return NULL;
    }

    return this_arg;
}

// Record gauge: collector.recordGauge(name, value, category)
static napi_value RecordGauge(napi_env env, napi_callback_info info) {
    napi_status status;
    size_t argc = 3;
    napi_value args[3];
    napi_value this_arg;

    status = napi_get_cb_info(env, info, &argc, args, &this_arg, NULL);
    if (status != napi_ok || argc < 2) {
        NIMCP_THROW(NIMCP_ERROR_INVALID_PARAM, "RecordGauge: Invalid arguments");
        return NULL;
    }

    // Unwrap the metrics collector
    MetricsCollectorWrap* wrap;
    status = napi_unwrap(env, this_arg, (void**)&wrap);
    if (status != napi_ok || !wrap || !wrap->collector) {
        NIMCP_THROW(NIMCP_ERROR_NOT_INITIALIZED, "RecordGauge: Invalid metrics collector object");
        napi_throw_error(env, NULL, "Invalid metrics collector object");
        return NULL;
    }

    // Get name
    size_t str_len;
    napi_get_value_string_utf8(env, args[0], NULL, 0, &str_len);
    char* name = (char*)nimcp_malloc(str_len + 1);
    if (!name) {
        NIMCP_THROW_MEMORY(NIMCP_ERROR_NO_MEMORY, str_len + 1,
                          "RecordGauge: Failed to allocate memory for name");
        napi_throw_error(env, NULL, "Failed to allocate memory for name");
        return NULL;
    }
    napi_get_value_string_utf8(env, args[0], name, str_len + 1, &str_len);

    // Get value
    double value;
    napi_get_value_double(env, args[1], &value);

    // Get category (optional, default to CUSTOM)
    int32_t category = NIMCP_METRIC_CATEGORY_CUSTOM;
    if (argc >= 3) {
        napi_get_value_int32(env, args[2], &category);
    }

    // Record gauge
    bool success = nimcp_metrics_record_gauge(wrap->collector, name, value,
                                               (nimcp_metric_category_t)category);
    nimcp_free(name);

    if (!success) {
        NIMCP_THROW(NIMCP_ERROR_OPERATION_FAILED, "RecordGauge: Failed to record gauge");
        napi_throw_error(env, NULL, "Failed to record gauge");
        return NULL;
    }

    return this_arg;
}

// Record timer: collector.recordTimer(name, duration_ms, category)
static napi_value RecordTimer(napi_env env, napi_callback_info info) {
    napi_status status;
    size_t argc = 3;
    napi_value args[3];
    napi_value this_arg;

    status = napi_get_cb_info(env, info, &argc, args, &this_arg, NULL);
    if (status != napi_ok || argc < 2) {
        NIMCP_THROW(NIMCP_ERROR_INVALID_PARAM, "RecordTimer: Invalid arguments");
        return NULL;
    }

    // Unwrap the metrics collector
    MetricsCollectorWrap* wrap;
    status = napi_unwrap(env, this_arg, (void**)&wrap);
    if (status != napi_ok || !wrap || !wrap->collector) {
        NIMCP_THROW(NIMCP_ERROR_NOT_INITIALIZED, "RecordTimer: Invalid metrics collector object");
        napi_throw_error(env, NULL, "Invalid metrics collector object");
        return NULL;
    }

    // Get name
    size_t str_len;
    napi_get_value_string_utf8(env, args[0], NULL, 0, &str_len);
    char* name = (char*)nimcp_malloc(str_len + 1);
    if (!name) {
        NIMCP_THROW_MEMORY(NIMCP_ERROR_NO_MEMORY, str_len + 1,
                          "RecordTimer: Failed to allocate memory for name");
        napi_throw_error(env, NULL, "Failed to allocate memory for name");
        return NULL;
    }
    napi_get_value_string_utf8(env, args[0], name, str_len + 1, &str_len);

    // Get duration
    double duration_ms;
    napi_get_value_double(env, args[1], &duration_ms);

    // Get category (optional, default to PERFORMANCE)
    int32_t category = NIMCP_METRIC_CATEGORY_PERFORMANCE;
    if (argc >= 3) {
        napi_get_value_int32(env, args[2], &category);
    }

    // Record timer
    bool success = nimcp_metrics_record_timer(wrap->collector, name, duration_ms,
                                               (nimcp_metric_category_t)category);
    nimcp_free(name);

    if (!success) {
        NIMCP_THROW(NIMCP_ERROR_OPERATION_FAILED, "RecordTimer: Failed to record timer");
        napi_throw_error(env, NULL, "Failed to record timer");
        return NULL;
    }

    return this_arg;
}

// Flush: collector.flush()
static napi_value Flush(napi_env env, napi_callback_info info) {
    napi_status status;
    napi_value this_arg;

    status = napi_get_cb_info(env, info, NULL, NULL, &this_arg, NULL);
    if (status != napi_ok) {
        NIMCP_THROW(NIMCP_ERROR_INVALID_PARAM, "Flush: Failed to get callback info");
        return NULL;
    }

    // Unwrap the metrics collector
    MetricsCollectorWrap* wrap;
    status = napi_unwrap(env, this_arg, (void**)&wrap);
    if (status != napi_ok || !wrap || !wrap->collector) {
        NIMCP_THROW(NIMCP_ERROR_NOT_INITIALIZED, "Flush: Invalid metrics collector object");
        napi_throw_error(env, NULL, "Invalid metrics collector object");
        return NULL;
    }

    int32_t count = nimcp_metrics_flush(wrap->collector);
    if (count < 0) {
        NIMCP_THROW_IO(NIMCP_ERROR_IO, NULL, "Flush: Failed to flush metrics");
        napi_throw_error(env, NULL, "Failed to flush metrics");
        return NULL;
    }

    napi_value result;
    napi_create_int32(env, count, &result);
    return result;
}

// Export to Tableau CSV: collector.exportTableauCsv(filename)
static napi_value ExportTableauCsv(napi_env env, napi_callback_info info) {
    napi_status status;
    size_t argc = 1;
    napi_value args[1];
    napi_value this_arg;

    status = napi_get_cb_info(env, info, &argc, args, &this_arg, NULL);
    if (status != napi_ok || argc < 1) {
        NIMCP_THROW(NIMCP_ERROR_INVALID_PARAM, "ExportTableauCsv: Invalid arguments");
        return NULL;
    }

    // Unwrap the metrics collector
    MetricsCollectorWrap* wrap;
    status = napi_unwrap(env, this_arg, (void**)&wrap);
    if (status != napi_ok || !wrap || !wrap->collector) {
        NIMCP_THROW(NIMCP_ERROR_NOT_INITIALIZED, "ExportTableauCsv: Invalid metrics collector object");
        napi_throw_error(env, NULL, "Invalid metrics collector object");
        return NULL;
    }

    // Get filename
    size_t str_len;
    napi_get_value_string_utf8(env, args[0], NULL, 0, &str_len);
    char* filename = (char*)nimcp_malloc(str_len + 1);
    if (!filename) {
        NIMCP_THROW_MEMORY(NIMCP_ERROR_NO_MEMORY, str_len + 1,
                          "ExportTableauCsv: Failed to allocate memory for filename");
        napi_throw_error(env, NULL, "Failed to allocate memory for filename");
        return NULL;
    }
    napi_get_value_string_utf8(env, args[0], filename, str_len + 1, &str_len);

    bool success = nimcp_metrics_export_tableau_csv(wrap->collector, filename);

    if (!success) {
        NIMCP_THROW_IO(NIMCP_ERROR_FILE_WRITE, filename,
                      "ExportTableauCsv: Failed to export to Tableau CSV '%s'", filename);
        nimcp_free(filename);
        napi_throw_error(env, NULL, "Failed to export to Tableau CSV");
        return NULL;
    }
    nimcp_free(filename);

    return this_arg;
}

// Export to PowerBI JSON: collector.exportPowerBiJson(filename)
static napi_value ExportPowerBiJson(napi_env env, napi_callback_info info) {
    napi_status status;
    size_t argc = 1;
    napi_value args[1];
    napi_value this_arg;

    status = napi_get_cb_info(env, info, &argc, args, &this_arg, NULL);
    if (status != napi_ok || argc < 1) {
        NIMCP_THROW(NIMCP_ERROR_INVALID_PARAM, "ExportPowerBiJson: Invalid arguments");
        return NULL;
    }

    // Unwrap the metrics collector
    MetricsCollectorWrap* wrap;
    status = napi_unwrap(env, this_arg, (void**)&wrap);
    if (status != napi_ok || !wrap || !wrap->collector) {
        NIMCP_THROW(NIMCP_ERROR_NOT_INITIALIZED, "ExportPowerBiJson: Invalid metrics collector object");
        napi_throw_error(env, NULL, "Invalid metrics collector object");
        return NULL;
    }

    // Get filename
    size_t str_len;
    napi_get_value_string_utf8(env, args[0], NULL, 0, &str_len);
    char* filename = (char*)nimcp_malloc(str_len + 1);
    if (!filename) {
        NIMCP_THROW_MEMORY(NIMCP_ERROR_NO_MEMORY, str_len + 1,
                          "ExportPowerBiJson: Failed to allocate memory for filename");
        napi_throw_error(env, NULL, "Failed to allocate memory for filename");
        return NULL;
    }
    napi_get_value_string_utf8(env, args[0], filename, str_len + 1, &str_len);

    bool success = nimcp_metrics_export_powerbi_json(wrap->collector, filename);

    if (!success) {
        NIMCP_THROW_IO(NIMCP_ERROR_FILE_WRITE, filename,
                      "ExportPowerBiJson: Failed to export to PowerBI JSON '%s'", filename);
        nimcp_free(filename);
        napi_throw_error(env, NULL, "Failed to export to PowerBI JSON");
        return NULL;
    }
    nimcp_free(filename);

    return this_arg;
}

// Get stats: collector.getStats()
static napi_value GetStats(napi_env env, napi_callback_info info) {
    napi_status status;
    napi_value this_arg;

    status = napi_get_cb_info(env, info, NULL, NULL, &this_arg, NULL);
    if (status != napi_ok) {
        NIMCP_THROW(NIMCP_ERROR_INVALID_PARAM, "GetStats: Failed to get callback info");
        return NULL;
    }

    // Unwrap the metrics collector
    MetricsCollectorWrap* wrap;
    status = napi_unwrap(env, this_arg, (void**)&wrap);
    if (status != napi_ok || !wrap || !wrap->collector) {
        NIMCP_THROW(NIMCP_ERROR_NOT_INITIALIZED, "GetStats: Invalid metrics collector object");
        napi_throw_error(env, NULL, "Invalid metrics collector object");
        return NULL;
    }

    const char* stats_json = nimcp_metrics_get_stats_json(wrap->collector);
    if (!stats_json) {
        NIMCP_THROW(NIMCP_ERROR_OPERATION_FAILED, "GetStats: Failed to get stats");
        napi_throw_error(env, NULL, "Failed to get stats");
        return NULL;
    }

    napi_value result;
    napi_create_string_utf8(env, stats_json, NAPI_AUTO_LENGTH, &result);
    return result;
}

// Module initialization
static napi_value Init(napi_env env, napi_value exports) {
    napi_status status;

    LOG_MODULE_INFO("bindings.nodejs", "Initializing Node.js bindings for NIMCP");

    // Initialize NIMCP library
    if (nimcp_init() != NIMCP_OK) {
        NIMCP_THROW_CRITICAL(NIMCP_ERROR_NOT_INITIALIZED,
                            "Init: Failed to initialize NIMCP core library");
        LOG_MODULE_ERROR("bindings.nodejs", "Failed to initialize NIMCP core library");
        return NULL;
    }

    // Define NeuralNetwork class
    napi_value neural_network_class;
    napi_property_descriptor neural_network_props[] = {
        { "forward", NULL, Forward, NULL, NULL, NULL, napi_default, NULL }
    };

    status = napi_define_class(env, "NeuralNetwork", NAPI_AUTO_LENGTH,
                               CreateNeuralNetwork, NULL,
                               sizeof(neural_network_props) / sizeof(napi_property_descriptor),
                               neural_network_props, &neural_network_class);
    if (status != napi_ok) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "Init: validation failed");
        return NULL;
    }

    // Export NeuralNetwork class
    status = napi_set_named_property(env, exports, "NeuralNetwork", neural_network_class);
    if (status != napi_ok) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "Init: validation failed");
        return NULL;
    }

    // Define MetricsCollector class
    napi_value metrics_collector_class;
    napi_property_descriptor metrics_collector_props[] = {
        { "recordCounter", NULL, RecordCounter, NULL, NULL, NULL, napi_default, NULL },
        { "recordGauge", NULL, RecordGauge, NULL, NULL, NULL, napi_default, NULL },
        { "recordTimer", NULL, RecordTimer, NULL, NULL, NULL, napi_default, NULL },
        { "flush", NULL, Flush, NULL, NULL, NULL, napi_default, NULL },
        { "exportTableauCsv", NULL, ExportTableauCsv, NULL, NULL, NULL, napi_default, NULL },
        { "exportPowerBiJson", NULL, ExportPowerBiJson, NULL, NULL, NULL, napi_default, NULL },
        { "getStats", NULL, GetStats, NULL, NULL, NULL, napi_default, NULL }
    };

    status = napi_define_class(env, "MetricsCollector", NAPI_AUTO_LENGTH,
                               CreateMetricsCollector, NULL,
                               sizeof(metrics_collector_props) / sizeof(napi_property_descriptor),
                               metrics_collector_props, &metrics_collector_class);
    if (status != napi_ok) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "Init: validation failed");
        return NULL;
    }

    // Export MetricsCollector class
    status = napi_set_named_property(env, exports, "MetricsCollector", metrics_collector_class);
    if (status != napi_ok) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "Init: validation failed");
        return NULL;
    }

    // Export metric category constants
    napi_value category_performance, category_memory, category_network;
    napi_value category_learning, category_inference, category_system, category_custom;
    napi_create_int32(env, NIMCP_METRIC_CATEGORY_PERFORMANCE, &category_performance);
    napi_create_int32(env, NIMCP_METRIC_CATEGORY_MEMORY, &category_memory);
    napi_create_int32(env, NIMCP_METRIC_CATEGORY_NETWORK, &category_network);
    napi_create_int32(env, NIMCP_METRIC_CATEGORY_LEARNING, &category_learning);
    napi_create_int32(env, NIMCP_METRIC_CATEGORY_INFERENCE, &category_inference);
    napi_create_int32(env, NIMCP_METRIC_CATEGORY_SYSTEM, &category_system);
    napi_create_int32(env, NIMCP_METRIC_CATEGORY_CUSTOM, &category_custom);

    napi_set_named_property(env, exports, "METRIC_CATEGORY_PERFORMANCE", category_performance);
    napi_set_named_property(env, exports, "METRIC_CATEGORY_MEMORY", category_memory);
    napi_set_named_property(env, exports, "METRIC_CATEGORY_NETWORK", category_network);
    napi_set_named_property(env, exports, "METRIC_CATEGORY_LEARNING", category_learning);
    napi_set_named_property(env, exports, "METRIC_CATEGORY_INFERENCE", category_inference);
    napi_set_named_property(env, exports, "METRIC_CATEGORY_SYSTEM", category_system);
    napi_set_named_property(env, exports, "METRIC_CATEGORY_CUSTOM", category_custom);

    LOG_MODULE_INFO("bindings.nodejs", "Successfully initialized Node.js bindings");
    return exports;
}

NAPI_MODULE(NODE_GYP_MODULE_NAME, Init)
