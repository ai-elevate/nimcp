#include <node_api.h>
#include "api/nimcp.h"
#include "utils/metrics/nimcp_metrics.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    free(wrap);
}

// Destructor for Network
static void network_finalize(napi_env env, void* finalize_data, void* finalize_hint) {
    NetworkWrap* wrap = (NetworkWrap*)finalize_data;
    if (wrap && wrap->network) {
        nimcp_network_destroy(wrap->network);
    }
    free(wrap);
}

// Destructor for MetricsCollector
static void metrics_collector_finalize(napi_env env, void* finalize_data, void* finalize_hint) {
    MetricsCollectorWrap* wrap = (MetricsCollectorWrap*)finalize_data;
    if (wrap && wrap->collector) {
        nimcp_metrics_destroy(wrap->collector);
    }
    free(wrap);
}

// Create neural network: new NeuralNetwork(config)
static napi_value CreateNeuralNetwork(napi_env env, napi_callback_info info) {
    napi_status status;
    size_t argc = 1;
    napi_value args[1];
    napi_value this_arg;

    status = napi_get_cb_info(env, info, &argc, args, &this_arg, NULL);
    if (status != napi_ok) return NULL;

    // Parse config object
    uint32_t num_inputs = 0;
    uint32_t num_outputs = 0;
    uint32_t num_hidden = 0;
    float learning_rate = 0.01f;

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
        napi_throw_error(env, NULL, nimcp_get_error());
        return NULL;
    }

    // Wrap the handle
    NetworkWrap* wrap = (NetworkWrap*)malloc(sizeof(NetworkWrap));
    if (!wrap) {
        nimcp_network_destroy(network);
        return NULL;
    }
    wrap->network = network;

    status = napi_wrap(env, this_arg, wrap, network_finalize, NULL, NULL);
    if (status != napi_ok) {
        free(wrap);
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
    if (status != napi_ok) return NULL;

    // Unwrap the neural network
    NetworkWrap* wrap;
    status = napi_unwrap(env, this_arg, (void**)&wrap);
    if (status != napi_ok || !wrap || !wrap->network) {
        napi_throw_error(env, NULL, "Invalid neural network object");
        return NULL;
    }

    // Get input array length
    uint32_t num_inputs;
    napi_get_array_length(env, args[0], &num_inputs);

    // Allocate and fill input array
    float* inputs = (float*)malloc(num_inputs * sizeof(float));
    for (uint32_t i = 0; i < num_inputs; i++) {
        napi_value element;
        napi_get_element(env, args[0], i, &element);
        double value;
        napi_get_value_double(env, element, &value);
        inputs[i] = (float)value;
    }

    // Allocate output array (assume same size as input for now)
    uint32_t num_outputs = num_inputs;  // TODO: get actual output size from network
    float* outputs = (float*)malloc(num_outputs * sizeof(float));

    // Forward pass using unified API
    nimcp_status_t result = nimcp_network_forward(wrap->network, inputs, num_inputs, outputs, num_outputs);

    free(inputs);

    if (result != NIMCP_OK) {
        free(outputs);
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

    free(outputs);
    return result_array;
}

// Create metrics collector: new MetricsCollector(config)
static napi_value CreateMetricsCollector(napi_env env, napi_callback_info info) {
    napi_status status;
    size_t argc = 1;
    napi_value args[1];
    napi_value this_arg;

    status = napi_get_cb_info(env, info, &argc, args, &this_arg, NULL);
    if (status != napi_ok) return NULL;

    // Create metrics collector
    nimcp_metrics_collector_t collector = nimcp_metrics_create();
    if (!collector) {
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
                char* directory = (char*)malloc(str_len + 1);
                napi_get_value_string_utf8(env, directory_val, directory, str_len + 1, &str_len);
                nimcp_metrics_set_directory(collector, directory);
                free(directory);
            }
        }
    }

    // Wrap the handle
    MetricsCollectorWrap* wrap = (MetricsCollectorWrap*)malloc(sizeof(MetricsCollectorWrap));
    if (!wrap) {
        nimcp_metrics_destroy(collector);
        return NULL;
    }
    wrap->collector = collector;

    status = napi_wrap(env, this_arg, wrap, metrics_collector_finalize, NULL, NULL);
    if (status != napi_ok) {
        free(wrap);
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
    if (status != napi_ok || argc < 2) return NULL;

    // Unwrap the metrics collector
    MetricsCollectorWrap* wrap;
    status = napi_unwrap(env, this_arg, (void**)&wrap);
    if (status != napi_ok || !wrap || !wrap->collector) {
        napi_throw_error(env, NULL, "Invalid metrics collector object");
        return NULL;
    }

    // Get name
    size_t str_len;
    napi_get_value_string_utf8(env, args[0], NULL, 0, &str_len);
    char* name = (char*)malloc(str_len + 1);
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
    free(name);

    if (!success) {
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
    if (status != napi_ok || argc < 2) return NULL;

    // Unwrap the metrics collector
    MetricsCollectorWrap* wrap;
    status = napi_unwrap(env, this_arg, (void**)&wrap);
    if (status != napi_ok || !wrap || !wrap->collector) {
        napi_throw_error(env, NULL, "Invalid metrics collector object");
        return NULL;
    }

    // Get name
    size_t str_len;
    napi_get_value_string_utf8(env, args[0], NULL, 0, &str_len);
    char* name = (char*)malloc(str_len + 1);
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
    free(name);

    if (!success) {
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
    if (status != napi_ok || argc < 2) return NULL;

    // Unwrap the metrics collector
    MetricsCollectorWrap* wrap;
    status = napi_unwrap(env, this_arg, (void**)&wrap);
    if (status != napi_ok || !wrap || !wrap->collector) {
        napi_throw_error(env, NULL, "Invalid metrics collector object");
        return NULL;
    }

    // Get name
    size_t str_len;
    napi_get_value_string_utf8(env, args[0], NULL, 0, &str_len);
    char* name = (char*)malloc(str_len + 1);
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
    free(name);

    if (!success) {
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
    if (status != napi_ok) return NULL;

    // Unwrap the metrics collector
    MetricsCollectorWrap* wrap;
    status = napi_unwrap(env, this_arg, (void**)&wrap);
    if (status != napi_ok || !wrap || !wrap->collector) {
        napi_throw_error(env, NULL, "Invalid metrics collector object");
        return NULL;
    }

    int32_t count = nimcp_metrics_flush(wrap->collector);
    if (count < 0) {
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
    if (status != napi_ok || argc < 1) return NULL;

    // Unwrap the metrics collector
    MetricsCollectorWrap* wrap;
    status = napi_unwrap(env, this_arg, (void**)&wrap);
    if (status != napi_ok || !wrap || !wrap->collector) {
        napi_throw_error(env, NULL, "Invalid metrics collector object");
        return NULL;
    }

    // Get filename
    size_t str_len;
    napi_get_value_string_utf8(env, args[0], NULL, 0, &str_len);
    char* filename = (char*)malloc(str_len + 1);
    napi_get_value_string_utf8(env, args[0], filename, str_len + 1, &str_len);

    bool success = nimcp_metrics_export_tableau_csv(wrap->collector, filename);
    free(filename);

    if (!success) {
        napi_throw_error(env, NULL, "Failed to export to Tableau CSV");
        return NULL;
    }

    return this_arg;
}

// Export to PowerBI JSON: collector.exportPowerBiJson(filename)
static napi_value ExportPowerBiJson(napi_env env, napi_callback_info info) {
    napi_status status;
    size_t argc = 1;
    napi_value args[1];
    napi_value this_arg;

    status = napi_get_cb_info(env, info, &argc, args, &this_arg, NULL);
    if (status != napi_ok || argc < 1) return NULL;

    // Unwrap the metrics collector
    MetricsCollectorWrap* wrap;
    status = napi_unwrap(env, this_arg, (void**)&wrap);
    if (status != napi_ok || !wrap || !wrap->collector) {
        napi_throw_error(env, NULL, "Invalid metrics collector object");
        return NULL;
    }

    // Get filename
    size_t str_len;
    napi_get_value_string_utf8(env, args[0], NULL, 0, &str_len);
    char* filename = (char*)malloc(str_len + 1);
    napi_get_value_string_utf8(env, args[0], filename, str_len + 1, &str_len);

    bool success = nimcp_metrics_export_powerbi_json(wrap->collector, filename);
    free(filename);

    if (!success) {
        napi_throw_error(env, NULL, "Failed to export to PowerBI JSON");
        return NULL;
    }

    return this_arg;
}

// Get stats: collector.getStats()
static napi_value GetStats(napi_env env, napi_callback_info info) {
    napi_status status;
    napi_value this_arg;

    status = napi_get_cb_info(env, info, NULL, NULL, &this_arg, NULL);
    if (status != napi_ok) return NULL;

    // Unwrap the metrics collector
    MetricsCollectorWrap* wrap;
    status = napi_unwrap(env, this_arg, (void**)&wrap);
    if (status != napi_ok || !wrap || !wrap->collector) {
        napi_throw_error(env, NULL, "Invalid metrics collector object");
        return NULL;
    }

    const char* stats_json = nimcp_metrics_get_stats_json(wrap->collector);
    if (!stats_json) {
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

    // Initialize NIMCP library
    nimcp_init();

    // Define NeuralNetwork class
    napi_value neural_network_class;
    napi_property_descriptor neural_network_props[] = {
        { "forward", NULL, Forward, NULL, NULL, NULL, napi_default, NULL }
    };

    status = napi_define_class(env, "NeuralNetwork", NAPI_AUTO_LENGTH,
                               CreateNeuralNetwork, NULL,
                               sizeof(neural_network_props) / sizeof(napi_property_descriptor),
                               neural_network_props, &neural_network_class);
    if (status != napi_ok) return NULL;

    // Export NeuralNetwork class
    status = napi_set_named_property(env, exports, "NeuralNetwork", neural_network_class);
    if (status != napi_ok) return NULL;

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
    if (status != napi_ok) return NULL;

    // Export MetricsCollector class
    status = napi_set_named_property(env, exports, "MetricsCollector", metrics_collector_class);
    if (status != napi_ok) return NULL;

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

    return exports;
}

NAPI_MODULE(NODE_GYP_MODULE_NAME, Init)
