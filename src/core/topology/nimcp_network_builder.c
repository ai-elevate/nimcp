//=============================================================================
// nimcp_network_builder.c - Network Builder Implementation
//=============================================================================

#include "nimcp_network_builder.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

//=============================================================================
// Default Configuration
//=============================================================================

network_builder_config_t network_builder_default(void) {
    network_builder_config_t config;
    memset(&config, 0, sizeof(config));

    // Basic network defaults
    config.num_neurons = 100;
    config.ei_ratio = 0.8f;  // 80% excitatory
    config.enable_stdp = true;
    config.enable_homeostasis = true;

    // No topology by default
    config.use_topology = false;

    // No pink noise weights by default
    config.use_pink_noise_weights = false;
    config.noise_amplitude = 0.5f;
    config.base_weight = 0.0f;

    // Random seed
    config.random_seed = 0;  // 0 means use time-based seed
    config.verbose = false;

    return config;
}

//=============================================================================
// Network Building
//=============================================================================

neural_network_t network_builder_build(const network_builder_config_t* config) {
    // Guard: NULL config
    if (!config) {
        fprintf(stderr, "ERROR: network_builder_build: NULL config\n");
        return NULL;
    }

    // Guard: Invalid neuron count
    if (config->num_neurons < 1) {
        fprintf(stderr, "ERROR: network_builder_build: num_neurons must be >= 1\n");
        return NULL;
    }

    if (config->verbose) {
        printf("Building network with %u neurons...\n", config->num_neurons);
    }

    // Step 1: Create base network
    network_config_t net_config = {
        .num_neurons = config->num_neurons,
        .ei_ratio = config->ei_ratio,
        .enable_stdp = config->enable_stdp,
        .enable_homeostasis = config->enable_homeostasis,
        .learning_rate = 0.01f,
        .hebbian_rate = 0.001f,
        .stdp_window = 20.0f,
        .homeostatic_rate = 0.001f,
        .target_activity = 0.05f,
        .adaptation_rate = 0.01f,
        .refractory_period = 2.0f,
        .min_weight = -1.0f,
        .max_weight = 1.0f,
        .update_interval = 1,
        .input_size = 0,
        .output_size = 0,
        .num_layers = 0,
        .layer_sizes = NULL,
        .enable_hebbian = false,
        .enable_oja = false,
        .neuron_model = NEURON_MODEL_LIF,
        .model_params = NULL
    };

    neural_network_t network = neural_network_create(&net_config);
    if (!network) {
        fprintf(stderr, "ERROR: Failed to create neural network\n");
        return NULL;
    }

    // Step 2: Apply topology if configured
    if (config->use_topology) {
        if (config->verbose) {
            printf("Applying topology...\n");
        }

        topology_stats_t stats;
        bool success = topology_generate(network, &config->topology_config, &stats);

        if (!success) {
            fprintf(stderr, "ERROR: Topology generation failed: %s\n",
                    topology_get_last_error());
            neural_network_destroy(network);
            return NULL;
        }

        if (config->verbose) {
            printf("Topology created: %u neurons, %u synapses, %u hubs\n",
                   stats.num_neurons, stats.num_synapses, stats.num_hubs);
            printf("Average degree: %.2f, std: %.2f\n",
                   stats.avg_degree, stats.degree_std);
        }
    }

    // Step 3: Initialize weights with pink noise if configured
    if (config->use_pink_noise_weights) {
        if (config->verbose) {
            printf("Initializing weights with pink noise (amplitude: %.2f)...\n",
                   config->noise_amplitude);
        }

        bool success = network_init_weights_pink_noise(
            network,
            config->noise_amplitude,
            config->base_weight
        );

        if (!success) {
            fprintf(stderr, "WARNING: Pink noise weight initialization failed\n");
            // Don't fail the whole build, just use default weights
        }
    }

    if (config->verbose) {
        printf("Network build complete!\n");
    }

    return network;
}

//=============================================================================
// Shorthand Functions
//=============================================================================

neural_network_t network_create_scale_free(uint32_t num_neurons, float gamma) {
    network_builder_config_t config = network_builder_default();
    config.num_neurons = num_neurons;
    config.use_topology = true;
    config.topology_config.type = TOPOLOGY_SCALE_FREE;
    config.topology_config.params.scale_free = topology_default_scale_free_config();
    config.topology_config.params.scale_free.power_law_gamma = gamma;

    return network_builder_build(&config);
}

neural_network_t network_create_fractal(uint32_t num_neurons, float fractal_dimension) {
    network_builder_config_t config = network_builder_default();
    config.num_neurons = num_neurons;
    config.use_topology = true;
    config.topology_config.type = TOPOLOGY_FRACTAL;
    config.topology_config.params.fractal = topology_default_fractal_config();
    config.topology_config.params.fractal.fractal_dimension = fractal_dimension;

    return network_builder_build(&config);
}

//=============================================================================
// Pink Noise Weight Initialization
//=============================================================================

// Internal network structure (defined in nimcp_neuralnet.c)
// WHAT: Access internal structure for synapse iteration
// WHY: neural_network_t is opaque, need to cast to access neurons
// HOW: Forward declare the internal structure
struct neural_network_struct {
    neuron_t* neurons;
    uint32_t num_neurons;
    uint32_t capacity;
    // ... other fields not needed here
};

bool network_init_weights_pink_noise(
    neural_network_t network,
    float amplitude,
    float base_weight
) {
    // Guard: NULL network
    if (!network) {
        fprintf(stderr, "ERROR: network_init_weights_pink_noise: NULL network\n");
        return false;
    }

    // Step 1: Count total number of synapses
    struct neural_network_struct* net = (struct neural_network_struct*)network;
    uint32_t total_synapses = 0;

    for (uint32_t i = 0; i < net->num_neurons; i++) {
        total_synapses += net->neurons[i].num_synapses;
    }

    if (total_synapses == 0) {
        fprintf(stderr, "INFO: No synapses in network, skipping weight initialization\n");
        return true;
    }

    // Step 2: Create pink noise generator
    pink_noise_config_t noise_config = {
        .method = PINK_NOISE_VOSS,
        .alpha = 1.0f,              // Pure pink noise (1/f)
        .amplitude = amplitude,
        .min_frequency = 0.1f,
        .max_frequency = 1000.0f,
        .sample_rate = 44100.0f,
        .seed = 0                   // Use time-based seed
    };

    pink_noise_generator_t generator = pink_noise_create(&noise_config);
    if (!generator) {
        fprintf(stderr, "ERROR: Failed to create pink noise generator\n");
        return false;
    }

    // Step 3: Generate all noise samples at once
    float* noise_samples = (float*)malloc(total_synapses * sizeof(float));
    if (!noise_samples) {
        fprintf(stderr, "ERROR: Failed to allocate noise samples array\n");
        pink_noise_destroy(generator);
        return false;
    }

    if (!pink_noise_generate(generator, noise_samples, total_synapses)) {
        fprintf(stderr, "ERROR: Failed to generate pink noise samples\n");
        free(noise_samples);
        pink_noise_destroy(generator);
        return false;
    }

    // Step 4: Apply noise to all synapses
    uint32_t sample_idx = 0;
    for (uint32_t i = 0; i < net->num_neurons; i++) {
        neuron_t* neuron = &net->neurons[i];

        for (uint32_t j = 0; j < neuron->num_synapses; j++) {
            // Set weight = base_weight + pink_noise_sample
            neuron->synapses[j].weight = base_weight + noise_samples[sample_idx];
            sample_idx++;
        }
    }

    // Cleanup
    free(noise_samples);
    pink_noise_destroy(generator);

    return true;
}
