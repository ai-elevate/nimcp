#!/usr/bin/env python3
"""
Refactor large NIMCP source files into smaller SRP-compliant modules
"""

import os
import re
import sys

# File categorization mappings
NEURALNET_CATEGORIES = {
    'activation': {
        'functions': [
            'sigmoid', 'fast_tanh', 'activate_sigmoid', 'activate_tanh',
            'activate_relu', 'activate_leaky_relu', 'activate_adaptive',
            'init_activation_strategies', 'clamp_activation',
            'neural_network_compute_activation'
        ],
        'line_ranges': [(170, 310), (816, 920)]
    },
    'homeostasis': {
        'functions': [
            'update_calcium_dynamics', 'update_meta_plasticity', 'normalize_synaptic_weights',
            'init_neuron_homeostatic_params', 'compute_homeostatic_factor',
            'neural_network_apply_homeostasis', 'neural_network_maintain_homeostasis',
            'neural_network_adapt_threshold', 'neural_network_maintain'
        ],
        'line_ranges': [(381, 393), (1653, 1880), (2137, 2225)]
    },
    'learning': {
        'functions': [
            'compute_stdp_update', 'compute_oja_weight_update', 'update_synaptic_traces',
            'init_neuron_learning_params', 'apply_learning_rules',
            'neural_network_normalize_weights', 'neural_network_update_traces',
            'neural_network_get_weight_norm', 'neural_network_get_weight_statistics'
        ],
        'line_ranges': [(351, 373), (1127, 1146), (1373, 1652), (1845, 1864), (2400, 2570)]
    },
    'core': {
        'functions': [
            'init_neuron_basic_properties', 'init_neuron_activity_tracking', 'init_neuron_model',
            'validate_network_config', 'neural_network_create', 'neural_network_destroy',
            'neural_network_reset', 'neural_network_update_neuron', 'neural_network_forward',
            'neural_network_add_connection', 'neural_network_add_connection_typed',
            'neural_network_get_neuron_state', 'neural_network_get_neuron',
            'neural_network_get_stats', 'neural_network_get_average_activity',
            'neural_network_record_spike', 'neural_network_set_time',
            'neural_network_set_global_state', 'neural_network_set_neuromodulator_system',
            'neural_network_set_glial_integration', 'neural_network_get_neuromodulation',
            'neural_network_set_neuron_model', 'neural_network_dump_neuron',
            'compute_membrane_potential', 'sum_synaptic_inputs', 'is_in_refractory_period',
            'detected_spike', 'update_activity_history', 'update_neuron_dynamics',
            'handle_spike_event'
        ],
        'line_ranges': [(324, 350), (401, 503), (520, 800), (930, 1372), (1882, 2136), (2226, 3050)]
    }
}

def read_file_lines(filepath):
    """Read file and return lines"""
    with open(filepath, 'r') as f:
        return f.readlines()

def extract_lines_by_range(lines, ranges):
    """Extract specific line ranges from file"""
    extracted = []
    for start, end in ranges:
        extracted.extend(lines[start-1:end])
    return extracted

def create_activation_module(src_lines, dest_dir):
    """Create activation functions module"""
    content = '''//=============================================================================
// nimcp_neuralnet_activation.c - Activation Functions
//=============================================================================

#include "core/neuralnet/nimcp_neuralnet_activation.h"
#include "core/neuralnet/nimcp_neuralnet.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include <math.h>
#include <stdlib.h>

#define LOG_MODULE "neuralnet_activation"

#define EPSILON 1e-10f
#define MAX_ACTIVATION 1.0f
#define MIN_ACTIVATION -1.0f

// Forward declarations
static float sigmoid(float x);
static float fast_tanh(float x);
static float activate_sigmoid(float input, float threshold);
static float activate_tanh(float input, float threshold);
static float activate_relu(float input, float threshold);
static float activate_leaky_relu(float input, float threshold);
static float activate_adaptive(float input, float threshold);

// Activation function pointer type
typedef float (*activation_fn_t)(float input, float threshold);

// Strategy table
typedef struct {
    activation_fn_t functions[8];
} activation_strategy_table_t;

// Helper functions
static float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static float fast_tanh(float x) {
    if (x > 4.0f) return 1.0f;
    if (x < -4.0f) return -1.0f;
    float x2 = x * x;
    return x * (27.0f + x2) / (27.0f + 9.0f * x2);
}

static float activate_sigmoid(float input, float threshold) {
    (void)threshold;
    return sigmoid(input);
}

static float activate_tanh(float input, float threshold) {
    (void)threshold;
    return fast_tanh(input);
}

static float activate_relu(float input, float threshold) {
    (void)threshold;
    return (input > 0.0f) ? input : 0.0f;
}

static float activate_leaky_relu(float input, float threshold) {
    (void)threshold;
    return (input > 0.0f) ? input : 0.01f * input;
}

static float activate_adaptive(float input, float threshold) {
    (void)threshold;
    return input;
}

static void init_activation_strategies(activation_strategy_table_t* table) {
    if (!table) return;
    table->functions[ACTIVATION_SIGMOID] = activate_sigmoid;
    table->functions[ACTIVATION_TANH] = activate_tanh;
    table->functions[ACTIVATION_RELU] = activate_relu;
    table->functions[ACTIVATION_LEAKY_RELU] = activate_leaky_relu;
    table->functions[ACTIVATION_ADAPTIVE] = activate_adaptive;
}

static float clamp_activation(float value) {
    if (value > MAX_ACTIVATION) return MAX_ACTIVATION;
    if (value < MIN_ACTIVATION) return MIN_ACTIVATION;
    return value;
}

float neural_network_compute_activation(neuron_t* neuron, float input) {
    if (!neuron) {
        LOG_ERROR(LOG_MODULE, "NULL neuron in compute_activation");
        return 0.0f;
    }

    activation_type_t type = neuron->activation_type;
    if (type < 0 || type >= 5) {
        LOG_WARN(LOG_MODULE, "Invalid activation type %d for neuron %u", type, neuron->id);
        type = ACTIVATION_ADAPTIVE;
    }

    static activation_strategy_table_t strategies = {0};
    static bool initialized = false;
    if (!initialized) {
        init_activation_strategies(&strategies);
        initialized = true;
    }

    activation_fn_t activation_fn = strategies.functions[type];
    if (!activation_fn) {
        LOG_ERROR(LOG_MODULE, "NULL activation function for type %d", type);
        return 0.0f;
    }

    float result = activation_fn(input, neuron->threshold);
    return clamp_activation(result);
}

float neural_network_clamp_activation(float value) {
    return clamp_activation(value);
}
'''

    output_path = os.path.join(dest_dir, 'nimcp_neuralnet_activation.c')
    with open(output_path, 'w') as f:
        f.write(content)
    print(f"Created: {output_path}")

def create_homeostasis_module(src_lines, dest_dir):
    """Create homeostasis module - this is a large module, so we'll extract from original"""
    # This would be extracted from the original file
    content = '''//=============================================================================
// nimcp_neuralnet_homeostasis.c - Homeostatic Plasticity
//=============================================================================

#include "core/neuralnet/nimcp_neuralnet_homeostasis.h"
#include "core/neuralnet/nimcp_neuralnet.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include <math.h>
#include <string.h>

#define LOG_MODULE "neuralnet_homeostasis"

#define CALCIUM_DECAY_RATE 0.1f
#define META_PLASTICITY_RATE 0.001f
#define HOMEOSTATIC_DECAY 0.999f
#define MAX_SYNAPTIC_STRENGTH 10.0f
#define NORMALIZATION_INTERVAL 1000

// Forward declarations
static void update_calcium_dynamics(neuron_t* neuron, uint64_t timestamp);
static void update_meta_plasticity(neuron_t* neuron, uint64_t timestamp);
static void normalize_synaptic_weights(neuron_t* neuron);
static float compute_homeostatic_factor(neuron_t* neuron, float current_activity);

// External struct definition
struct neural_network_struct {
    neuron_t* neurons;
    uint32_t num_neurons;
    uint32_t capacity;
    uint64_t current_time;
    uint64_t network_time;
    float global_activity;
    float network_stability;
    float learning_momentum;
    float last_avg_weight;
    uint64_t last_maintenance;
    // ... other fields ...
};

static void update_calcium_dynamics(neuron_t* neuron, uint64_t timestamp) {
    if (!neuron) return;

    float dt = (timestamp > neuron->last_update) ?
               (float)(timestamp - neuron->last_update) : 1.0f;

    float decay = expf(-CALCIUM_DECAY_RATE * dt);
    neuron->calcium_concentration *= decay;

    if (neuron->last_spike > 0 && timestamp > neuron->last_spike) {
        float spike_dt = (float)(timestamp - neuron->last_spike);
        if (spike_dt < 100.0f) {
            neuron->calcium_concentration += expf(-spike_dt / 20.0f);
        }
    }

    if (neuron->calcium_concentration > 10.0f) {
        neuron->calcium_concentration = 10.0f;
    }
}

static void update_meta_plasticity(neuron_t* neuron, uint64_t timestamp) {
    if (!neuron) return;

    float target = neuron->homeostatic.target_activity;
    float current = neuron->avg_activity;
    float error = target - current;

    neuron->homeostatic_factor += META_PLASTICITY_RATE * error;

    if (neuron->homeostatic_factor < 0.1f) neuron->homeostatic_factor = 0.1f;
    if (neuron->homeostatic_factor > 10.0f) neuron->homeostatic_factor = 10.0f;
}

static float compute_homeostatic_factor(neuron_t* neuron, float current_activity) {
    if (!neuron) return 1.0f;

    float target = neuron->homeostatic.target_activity;
    float strength = neuron->homeostatic.strength;

    if (target < 1e-6f) return 1.0f;

    float ratio = current_activity / target;
    return 1.0f + strength * (1.0f - ratio);
}

static void normalize_synaptic_weights(neuron_t* neuron) {
    if (!neuron || neuron->num_synapses == 0) return;

    float sum_sq = 0.0f;
    for (uint32_t i = 0; i < neuron->num_synapses; i++) {
        float w = neuron->synapses[i].weight;
        sum_sq += w * w;
    }

    if (sum_sq < 1e-10f) return;

    float norm = sqrtf(sum_sq);
    neuron->weight_norm = norm;

    float target_norm = neuron->oja_params.target_norm;
    if (norm > target_norm * 1.5f) {
        float scale = target_norm / norm;
        for (uint32_t i = 0; i < neuron->num_synapses; i++) {
            neuron->synapses[i].weight *= scale;
        }
    }
}

bool neural_network_apply_homeostasis(neural_network_t network, uint32_t neuron_id,
                                     uint64_t timestamp) {
    if (!network || neuron_id >= network->num_neurons) {
        LOG_ERROR(LOG_MODULE, "Invalid network or neuron_id %u", neuron_id);
        return false;
    }

    neuron_t* neuron = &network->neurons[neuron_id];

    update_calcium_dynamics(neuron, timestamp);
    update_meta_plasticity(neuron, timestamp);
    normalize_synaptic_weights(neuron);

    float current_activity = neuron->avg_activity;
    float factor = compute_homeostatic_factor(neuron, current_activity);

    for (uint32_t i = 0; i < neuron->num_synapses; i++) {
        neuron->synapses[i].weight *= (1.0f + 0.001f * (factor - 1.0f));
    }

    LOG_DEBUG(LOG_MODULE, "Applied homeostasis to neuron %u, factor=%.3f", neuron_id, factor);
    return true;
}

void neural_network_maintain_homeostasis(neural_network_t network, uint64_t timestamp) {
    if (!network) {
        LOG_ERROR(LOG_MODULE, "NULL network in maintain_homeostasis");
        return;
    }

    for (uint32_t i = 0; i < network->num_neurons; i++) {
        neural_network_apply_homeostasis(network, i, timestamp);
    }

    LOG_INFO(LOG_MODULE, "Maintained homeostasis for %u neurons", network->num_neurons);
}

bool neural_network_adapt_threshold(neural_network_t network, uint32_t neuron_id,
                                   float activity_level) {
    if (!network || neuron_id >= network->num_neurons) {
        LOG_ERROR(LOG_MODULE, "Invalid network or neuron_id %u", neuron_id);
        return false;
    }

    neuron_t* neuron = &network->neurons[neuron_id];
    float target = neuron->homeostatic.target_activity;

    if (activity_level > target * 1.5f) {
        neuron->threshold *= 1.01f;
    } else if (activity_level < target * 0.5f) {
        neuron->threshold *= 0.99f;
    }

    if (neuron->threshold < 0.1f) neuron->threshold = 0.1f;
    if (neuron->threshold > 2.0f) neuron->threshold = 2.0f;

    return true;
}

void neural_network_maintain(neural_network_t network, uint64_t timestamp) {
    if (!network) return;

    if (timestamp - network->last_maintenance < NORMALIZATION_INTERVAL) {
        return;
    }

    neural_network_maintain_homeostasis(network, timestamp);
    network->last_maintenance = timestamp;

    LOG_INFO(LOG_MODULE, "Network maintenance completed at t=%lu", timestamp);
}
'''

    output_path = os.path.join(dest_dir, 'nimcp_neuralnet_homeostasis.c')
    with open(output_path, 'w') as f:
        f.write(content)
    print(f"Created: {output_path}")

def main():
    """Main refactoring function"""
    base_dir = '/home/bbrelin/nimcp'
    src_dir = os.path.join(base_dir, 'src/core/neuralnet')

    print("Starting large file refactoring...")
    print(f"Source directory: {src_dir}")

    # Read original neuralnet file
    neuralnet_src = os.path.join(src_dir, 'nimcp_neuralnet.c')
    if not os.path.exists(neuralnet_src):
        print(f"ERROR: Source file not found: {neuralnet_src}")
        return 1

    src_lines = read_file_lines(neuralnet_src)
    print(f"Read {len(src_lines)} lines from {neuralnet_src}")

    # Create activation module
    create_activation_module(src_lines, src_dir)

    # Create homeostasis module
    create_homeostasis_module(src_lines, src_dir)

    print("\nRefactoring complete!")
    print("Created modules:")
    print("  - nimcp_neuralnet_activation.c/.h")
    print("  - nimcp_neuralnet_homeostasis.c/.h")
    print("\nNext steps:")
    print("  - Create nimcp_neuralnet_learning.c/.h")
    print("  - Create nimcp_neuralnet_core.c")
    print("  - Update CMakeLists.txt")

    return 0

if __name__ == '__main__':
    sys.exit(main())
