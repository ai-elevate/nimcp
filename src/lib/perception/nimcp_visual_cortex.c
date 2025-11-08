/**
 * @file nimcp_visual_cortex.c
 * @brief Implementation of biologically-inspired visual processing
 *
 * WHAT: CNN-based visual cortex with V1-style edge detection
 * WHY:  Enable visual perception and memory in NIMCP
 * HOW:  Lightweight convolution + Gabor filters + feature extraction
 *
 * @author NIMCP Development Team
 * @date 2025
 * @version 2.6
 */

#include "include/perception/nimcp_visual_cortex.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/validation/nimcp_validate.h"
#include "utils/logging/nimcp_logging.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

//=============================================================================
// Activation Functions
//=============================================================================

/**
 * @brief Apply activation function
 */
static inline float apply_activation(float x, visual_activation_type_t activation)
{
    switch (activation) {
        case VISUAL_ACTIVATION_RELU:
            return (x > 0.0f) ? x : 0.0f;
        case VISUAL_ACTIVATION_SIGMOID:
            return 1.0f / (1.0f + expf(-x));
        case VISUAL_ACTIVATION_TANH:
            return tanhf(x);
        case VISUAL_ACTIVATION_NONE:
        default:
            return x;
    }
}

//=============================================================================
// Convolution Layer Implementation
//=============================================================================

struct conv_layer_struct {
    uint32_t input_width;
    uint32_t input_height;
    uint32_t input_channels;
    uint32_t num_filters;
    uint32_t kernel_size;
    uint32_t stride;
    uint32_t padding;
    visual_activation_type_t activation;

    uint32_t output_width;
    uint32_t output_height;

    float* kernels;  // [num_filters][kernel_size][kernel_size][input_channels]
    float* bias;     // [num_filters]
};

/**
 * WHAT: Create convolution layer
 */
conv_layer_t* conv_layer_create(const conv_layer_config_t* config)
{
    // Guard: Validate inputs
    if (!nimcp_validate_pointer(config, "config")) {
        return NULL;
    }

    if (config->input_width == 0 || config->input_height == 0 ||
        config->input_channels == 0 || config->num_filters == 0 ||
        config->kernel_size == 0 || config->stride == 0) {
        NIMCP_LOGGING_ERROR("Invalid convolution layer configuration parameters");
        return NULL;
    }

    // Allocate layer
    conv_layer_t* layer = (conv_layer_t*)nimcp_calloc(1, sizeof(conv_layer_t));
    if (!layer) {
        NIMCP_LOGGING_ERROR("Failed to allocate convolution layer");
        return NULL;
    }

    layer->input_width = config->input_width;
    layer->input_height = config->input_height;
    layer->input_channels = config->input_channels;
    layer->num_filters = config->num_filters;
    layer->kernel_size = config->kernel_size;
    layer->stride = config->stride;
    layer->padding = config->padding;
    layer->activation = config->activation;

    // Compute output dimensions
    layer->output_width = (config->input_width + 2 * config->padding - config->kernel_size) / config->stride + 1;
    layer->output_height = (config->input_height + 2 * config->padding - config->kernel_size) / config->stride + 1;

    // Allocate kernels
    uint32_t kernel_total_size = config->num_filters * config->kernel_size * config->kernel_size * config->input_channels;
    layer->kernels = (float*)nimcp_calloc(kernel_total_size, sizeof(float));
    if (!nimcp_validate_pointer(layer->kernels, "kernels")) {
        NIMCP_LOGGING_ERROR("Failed to allocate convolution kernels");
        conv_layer_destroy(layer);
        return NULL;
    }

    // Initialize kernels with small random values
    for (uint32_t i = 0; i < kernel_total_size; i++) {
        layer->kernels[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }

    // Allocate bias
    layer->bias = (float*)nimcp_calloc(config->num_filters, sizeof(float));
    if (!nimcp_validate_pointer(layer->bias, "bias")) {
        NIMCP_LOGGING_ERROR("Failed to allocate convolution bias");
        conv_layer_destroy(layer);
        return NULL;
    }

    return layer;
}

/**
 * WHAT: Destroy convolution layer
 */
void conv_layer_destroy(conv_layer_t* layer)
{
    if (!layer) {
        return;
    }

    if (layer->kernels) {
        nimcp_free(layer->kernels);
    }
    if (layer->bias) {
        nimcp_free(layer->bias);
    }

    nimcp_free(layer);
}

/**
 * WHAT: Set kernel weights
 */
bool conv_layer_set_kernel(conv_layer_t* layer, uint32_t filter_idx, const float* kernel)
{
    // Guard: Validate inputs
    if (!nimcp_validate_pointer(layer, "layer") || !nimcp_validate_pointer(kernel, "kernel")) {
        return false;
    }

    if (filter_idx >= layer->num_filters) {
        NIMCP_LOGGING_ERROR("Invalid filter index: %u >= %u", filter_idx, layer->num_filters);
        return false;
    }

    uint32_t kernel_size_per_filter = layer->kernel_size * layer->kernel_size * layer->input_channels;
    uint32_t offset = filter_idx * kernel_size_per_filter;

    memcpy(&layer->kernels[offset], kernel, kernel_size_per_filter * sizeof(float));
    return true;
}

/**
 * WHAT: Forward pass through convolution
 */
bool conv_layer_forward(conv_layer_t* layer, const float* input, float* output)
{
    // Guard: Validate inputs
    if (!nimcp_validate_pointer(layer, "layer") ||
        !nimcp_validate_pointer(input, "input") ||
        !nimcp_validate_pointer(output, "output")) {
        return false;
    }

    uint32_t kernel_size_per_filter = layer->kernel_size * layer->kernel_size * layer->input_channels;

    // For each filter
    for (uint32_t f = 0; f < layer->num_filters; f++) {
        float* kernel = &layer->kernels[f * kernel_size_per_filter];

        // For each output position
        for (uint32_t oy = 0; oy < layer->output_height; oy++) {
            for (uint32_t ox = 0; ox < layer->output_width; ox++) {
                float sum = layer->bias[f];

                // Convolve
                for (uint32_t c = 0; c < layer->input_channels; c++) {
                    for (uint32_t ky = 0; ky < layer->kernel_size; ky++) {
                        for (uint32_t kx = 0; kx < layer->kernel_size; kx++) {
                            int32_t ix = ox * layer->stride + kx - layer->padding;
                            int32_t iy = oy * layer->stride + ky - layer->padding;

                            // Check bounds
                            if (ix >= 0 && ix < (int32_t)layer->input_width &&
                                iy >= 0 && iy < (int32_t)layer->input_height) {

                                uint32_t input_idx = (iy * layer->input_width + ix) * layer->input_channels + c;
                                uint32_t kernel_idx = (ky * layer->kernel_size + kx) * layer->input_channels + c;

                                sum += input[input_idx] * kernel[kernel_idx];
                            }
                        }
                    }
                }

                // Apply activation
                uint32_t output_idx = (oy * layer->output_width + ox) * layer->num_filters + f;
                output[output_idx] = apply_activation(sum, layer->activation);
            }
        }
    }

    return true;
}

/**
 * WHAT: Get output dimensions
 */
uint32_t conv_layer_get_output_width(const conv_layer_t* layer)
{
    return layer ? layer->output_width : 0;
}

uint32_t conv_layer_get_output_height(const conv_layer_t* layer)
{
    return layer ? layer->output_height : 0;
}

uint32_t conv_layer_get_output_channels(const conv_layer_t* layer)
{
    return layer ? layer->num_filters : 0;
}

//=============================================================================
// Pooling Layer Implementation
//=============================================================================

struct pool_layer_struct {
    uint32_t input_width;
    uint32_t input_height;
    uint32_t input_channels;
    uint32_t pool_size;
    uint32_t stride;
    pool_type_t type;

    uint32_t output_width;
    uint32_t output_height;
};

/**
 * WHAT: Create pooling layer
 */
pool_layer_t* pool_layer_create(const pool_layer_config_t* config)
{
    // Guard: Validate inputs
    if (!nimcp_validate_pointer(config, "config")) {
        return NULL;
    }

    if (config->input_width == 0 || config->input_height == 0 ||
        config->input_channels == 0 || config->pool_size == 0 || config->stride == 0) {
        NIMCP_LOGGING_ERROR("Invalid pooling layer configuration parameters");
        return NULL;
    }

    pool_layer_t* layer = (pool_layer_t*)nimcp_calloc(1, sizeof(pool_layer_t));
    if (!nimcp_validate_pointer(layer, "layer")) {
        NIMCP_LOGGING_ERROR("Failed to allocate pooling layer");
        return NULL;
    }

    layer->input_width = config->input_width;
    layer->input_height = config->input_height;
    layer->input_channels = config->input_channels;
    layer->pool_size = config->pool_size;
    layer->stride = config->stride;
    layer->type = config->type;

    layer->output_width = (config->input_width - config->pool_size) / config->stride + 1;
    layer->output_height = (config->input_height - config->pool_size) / config->stride + 1;

    return layer;
}

/**
 * WHAT: Destroy pooling layer
 */
void pool_layer_destroy(pool_layer_t* layer)
{
    if (layer) {
        nimcp_free(layer);
    }
}

/**
 * WHAT: Forward pass through pooling
 */
bool pool_layer_forward(pool_layer_t* layer, const float* input, float* output)
{
    // Guard: Validate inputs
    if (!nimcp_validate_pointer(layer, "layer") ||
        !nimcp_validate_pointer(input, "input") ||
        !nimcp_validate_pointer(output, "output")) {
        return false;
    }

    // For each channel
    for (uint32_t c = 0; c < layer->input_channels; c++) {
        // For each output position
        for (uint32_t oy = 0; oy < layer->output_height; oy++) {
            for (uint32_t ox = 0; ox < layer->output_width; ox++) {
                uint32_t base_y = oy * layer->stride;
                uint32_t base_x = ox * layer->stride;

                float pool_value;
                if (layer->type == POOL_MAX) {
                    pool_value = -INFINITY;
                } else {  // POOL_AVG
                    pool_value = 0.0f;
                }

                // Pool over window
                for (uint32_t py = 0; py < layer->pool_size; py++) {
                    for (uint32_t px = 0; px < layer->pool_size; px++) {
                        uint32_t iy = base_y + py;
                        uint32_t ix = base_x + px;

                        if (iy < layer->input_height && ix < layer->input_width) {
                            uint32_t input_idx = (iy * layer->input_width + ix) * layer->input_channels + c;

                            if (layer->type == POOL_MAX) {
                                if (input[input_idx] > pool_value) {
                                    pool_value = input[input_idx];
                                }
                            } else {  // POOL_AVG
                                pool_value += input[input_idx];
                            }
                        }
                    }
                }

                if (layer->type == POOL_AVG) {
                    pool_value /= (layer->pool_size * layer->pool_size);
                }

                uint32_t output_idx = (oy * layer->output_width + ox) * layer->input_channels + c;
                output[output_idx] = pool_value;
            }
        }
    }

    return true;
}

//=============================================================================
// Gabor Filter Implementation
//=============================================================================

/**
 * WHAT: Create Gabor kernel
 */
float* gabor_create_kernel(int kernel_size, const gabor_params_t* params)
{
    // Guard: Validate inputs
    if (!nimcp_validate_pointer(params, "params")) {
        return NULL;
    }

    if (kernel_size <= 0 || kernel_size % 2 == 0) {
        NIMCP_LOGGING_ERROR("Invalid kernel size: %d (must be positive and odd)", kernel_size);
        return NULL;
    }

    float* kernel = (float*)nimcp_calloc(kernel_size * kernel_size, sizeof(float));
    if (!nimcp_validate_pointer(kernel, "kernel")) {
        NIMCP_LOGGING_ERROR("Failed to allocate Gabor kernel");
        return NULL;
    }

    int center = kernel_size / 2;
    float theta = params->orientation * M_PI / 180.0f;  // Convert to radians
    float sigma = params->wavelength * params->bandwidth;
    float gamma = params->aspect_ratio;
    float lambda = params->wavelength;
    float psi = params->phase * M_PI / 180.0f;

    // Generate Gabor function
    for (int y = 0; y < kernel_size; y++) {
        for (int x = 0; x < kernel_size; x++) {
            float x_offset = x - center;
            float y_offset = y - center;

            // Rotate coordinates
            float x_rot = x_offset * cosf(theta) + y_offset * sinf(theta);
            float y_rot = -x_offset * sinf(theta) + y_offset * cosf(theta);

            // Gabor function
            float gaussian = expf(-(x_rot * x_rot + gamma * gamma * y_rot * y_rot) / (2.0f * sigma * sigma));
            float sinusoid = cosf(2.0f * M_PI * x_rot / lambda + psi);

            kernel[y * kernel_size + x] = gaussian * sinusoid;
        }
    }

    return kernel;
}

//=============================================================================
// Attention Map Implementation
//=============================================================================

struct attention_map_struct {
    uint32_t width;
    uint32_t height;
    float* values;  // [height][width]
};

/**
 * WHAT: Create attention map
 */
attention_map_t* attention_map_create(uint32_t width, uint32_t height)
{
    if (width == 0 || height == 0) {
        NIMCP_LOGGING_ERROR("Invalid attention map dimensions: %u x %u", width, height);
        return NULL;
    }

    attention_map_t* map = (attention_map_t*)nimcp_calloc(1, sizeof(attention_map_t));
    if (!nimcp_validate_pointer(map, "map")) {
        NIMCP_LOGGING_ERROR("Failed to allocate attention map");
        return NULL;
    }

    map->width = width;
    map->height = height;
    map->values = (float*)nimcp_calloc(width * height, sizeof(float));

    if (!nimcp_validate_pointer(map->values, "map->values")) {
        NIMCP_LOGGING_ERROR("Failed to allocate attention map values");
        nimcp_free(map);
        return NULL;
    }

    return map;
}

/**
 * WHAT: Destroy attention map
 */
void attention_map_destroy(attention_map_t* map)
{
    if (!map) {
        return;
    }

    if (map->values) {
        nimcp_free(map->values);
    }
    nimcp_free(map);
}

/**
 * WHAT: Get attention value
 */
float attention_map_get(const attention_map_t* map, uint32_t x, uint32_t y)
{
    if (!nimcp_validate_pointer(map, "map")) {
        return -1.0f;
    }

    if (x >= map->width || y >= map->height) {
        NIMCP_LOGGING_ERROR("Attention map coordinates out of bounds: (%u, %u) >= (%u, %u)", x, y, map->width, map->height);
        return -1.0f;
    }

    return map->values[y * map->width + x];
}

/**
 * WHAT: Set attention value
 */
bool attention_map_set(attention_map_t* map, uint32_t x, uint32_t y, float value)
{
    if (!nimcp_validate_pointer(map, "map")) {
        return false;
    }

    if (x >= map->width || y >= map->height) {
        NIMCP_LOGGING_ERROR("Attention map coordinates out of bounds: (%u, %u) >= (%u, %u)", x, y, map->width, map->height);
        return false;
    }

    map->values[y * map->width + x] = value;
    return true;
}

//=============================================================================
// Visual Cortex Implementation
//=============================================================================

#define MAX_VISUAL_MEMORIES 1000

struct visual_cortex_struct {
    // Configuration
    uint32_t input_width;
    uint32_t input_height;
    uint32_t num_v1_filters;
    uint32_t feature_dim;
    bool enable_attention;
    bool enable_memory;

    // Layers
    conv_layer_t* v1_layer;      // Edge detection
    pool_layer_t* pool_layer;    // Spatial pooling

    // Feature extraction (simplified - just average pooling to feature_dim)
    float* feature_weights;      // For compact feature representation

    // Visual memory
    visual_memory_t* memories[MAX_VISUAL_MEMORIES];
    uint32_t num_memories;

    // Statistics
    uint32_t images_processed;
    double total_processing_time;
};

/**
 * WHAT: Create visual cortex
 */
visual_cortex_t* visual_cortex_create(const visual_cortex_config_t* config)
{
    // Guard: Validate inputs
    if (!nimcp_validate_pointer(config, "config")) {
        return NULL;
    }

    if (config->input_width == 0 || config->input_height == 0 ||
        config->num_v1_filters == 0 || config->feature_dim == 0) {
        NIMCP_LOGGING_ERROR("Invalid visual cortex configuration parameters");
        return NULL;
    }

    visual_cortex_t* cortex = (visual_cortex_t*)nimcp_calloc(1, sizeof(visual_cortex_t));
    if (!cortex) {
        NIMCP_LOGGING_ERROR("Failed to allocate visual cortex");
        return NULL;
    }

    cortex->input_width = config->input_width;
    cortex->input_height = config->input_height;
    cortex->num_v1_filters = config->num_v1_filters;
    cortex->feature_dim = config->feature_dim;
    cortex->enable_attention = config->enable_attention;
    cortex->enable_memory = config->enable_memory;

    // Create V1 layer (edge detection with Gabor-like filters)
    conv_layer_config_t conv_config = {
        .input_width = config->input_width,
        .input_height = config->input_height,
        .input_channels = 1,  // Start with grayscale
        .num_filters = config->num_v1_filters,
        .kernel_size = 7,
        .stride = 2,
        .padding = 3,
        .activation = VISUAL_ACTIVATION_RELU
    };

    cortex->v1_layer = conv_layer_create(&conv_config);
    if (!nimcp_validate_pointer(cortex->v1_layer, "v1_layer")) {
        NIMCP_LOGGING_ERROR("Failed to create V1 convolution layer");
        visual_cortex_destroy(cortex);
        return NULL;
    }

    // Initialize V1 with Gabor filters at different orientations
    int num_orientations = (config->num_v1_filters >= 4) ? 4 : config->num_v1_filters;
    for (uint32_t i = 0; i < config->num_v1_filters; i++) {
        float orientation = (i % num_orientations) * (180.0f / num_orientations);

        gabor_params_t gabor_params = {
            .wavelength = 4.0f,
            .orientation = orientation,
            .phase = 0.0f,
            .aspect_ratio = 0.5f,
            .bandwidth = 1.0f
        };

        float* gabor_kernel = gabor_create_kernel(7, &gabor_params);
        if (gabor_kernel) {
            conv_layer_set_kernel(cortex->v1_layer, i, gabor_kernel);
            nimcp_free(gabor_kernel);
        }
    }

    // Create pooling layer
    uint32_t v1_output_w = conv_layer_get_output_width(cortex->v1_layer);
    uint32_t v1_output_h = conv_layer_get_output_height(cortex->v1_layer);

    pool_layer_config_t pool_config = {
        .input_width = v1_output_w,
        .input_height = v1_output_h,
        .input_channels = config->num_v1_filters,
        .pool_size = 2,
        .stride = 2,
        .type = POOL_MAX
    };

    cortex->pool_layer = pool_layer_create(&pool_config);
    if (!nimcp_validate_pointer(cortex->pool_layer, "pool_layer")) {
        NIMCP_LOGGING_ERROR("Failed to create pooling layer");
        visual_cortex_destroy(cortex);
        return NULL;
    }

    // Allocate feature weights (simplified feature extraction)
    uint32_t pooled_size = (v1_output_w / 2) * (v1_output_h / 2) * config->num_v1_filters;
    cortex->feature_weights = (float*)nimcp_calloc(pooled_size, sizeof(float));
    if (!nimcp_validate_pointer(cortex->feature_weights, "feature_weights")) {
        NIMCP_LOGGING_ERROR("Failed to allocate feature weights");
        visual_cortex_destroy(cortex);
        return NULL;
    }

    // Initialize feature weights
    for (uint32_t i = 0; i < pooled_size; i++) {
        cortex->feature_weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.01f;
    }

    // NIMCP 2.7 Phase 8.5: Fractal Topology Integration (Future Enhancement)
    // TODO: Add internal recurrent network with scale-free topology
    //
    // WHAT: Internal spiking network for recurrent V1 processing
    // WHY:  Enable temporal integration, feedback, and attention modulation
    // HOW:  Create neural_network with config->internal_neurons neurons,
    //       apply topology_generate_scale_free() with configured parameters
    //
    // BIOLOGICAL MOTIVATION:
    // While V1's feedforward structure (CNN) is innate, recurrent connections
    // within V1 (horizontal connections, feedback from V2/V4) are extensive and
    // exhibit scale-free properties. This internal network would model:
    // - Horizontal connections for contour integration
    // - Feedback modulation for attention and expectation
    // - Temporal integration for motion processing
    //
    // IMPLEMENTATION PLAN:
    // if (config->enable_fractal_topology && config->internal_neurons > 0) {
    //     // Create internal recurrent network
    //     cortex->internal_network = neural_network_create(config->internal_neurons);
    //
    //     // Generate scale-free topology
    //     scale_free_config_t topo_config = {
    //         .power_law_gamma = config->power_law_gamma,
    //         .hub_ratio = config->hub_ratio,
    //         .min_degree = 2,
    //         .max_degree = config->internal_neurons / 10,
    //         .spatial_constraint = 0.5f,  // V1 has spatial organization
    //         .bidirectional = false
    //     };
    //     topology_stats_t stats;
    //     topology_generate_scale_free(cortex->internal_network, &topo_config, &stats);
    //
    //     NIMCP_LOGGING_INFO("V1 internal network: %u neurons, %u synapses, %.2f avg degree",
    //                        stats.num_neurons, stats.num_synapses, stats.avg_degree);
    // }
    //
    // This enhancement deferred to Phase 8.7 (Specialized Neuron Types) to avoid
    // scope creep. Configuration is in place and ready for future implementation.

    return cortex;
}

/**
 * WHAT: Destroy visual cortex
 */
void visual_cortex_destroy(visual_cortex_t* cortex)
{
    if (!cortex) {
        return;
    }

    if (cortex->v1_layer) {
        conv_layer_destroy(cortex->v1_layer);
    }

    if (cortex->pool_layer) {
        pool_layer_destroy(cortex->pool_layer);
    }

    if (cortex->feature_weights) {
        nimcp_free(cortex->feature_weights);
    }

    // Free visual memories
    for (uint32_t i = 0; i < cortex->num_memories; i++) {
        if (cortex->memories[i]) {
            if (cortex->memories[i]->features) {
                nimcp_free(cortex->memories[i]->features);
            }
            nimcp_free(cortex->memories[i]);
        }
    }

    nimcp_free(cortex);
}

/**
 * WHAT: Process image through visual cortex
 */
bool visual_cortex_process(
    visual_cortex_t* cortex,
    const uint8_t* image,
    uint32_t width,
    uint32_t height,
    uint32_t channels,
    float* features)
{
    // Guard: Validate inputs
    if (!nimcp_validate_pointer(cortex, "cortex") ||
        !nimcp_validate_pointer(image, "image") ||
        !nimcp_validate_pointer(features, "features")) {
        return false;
    }

    if (width != cortex->input_width || height != cortex->input_height || channels == 0) {
        NIMCP_LOGGING_ERROR("Invalid image dimensions: %ux%ux%u (expected %ux%ux>0)",
                           width, height, channels, cortex->input_width, cortex->input_height);
        return false;
    }

    clock_t start = clock();

    // Convert uint8 image to float (normalize to 0-1)
    uint32_t input_size = width * height;
    float* input_float = (float*)nimcp_calloc(input_size, sizeof(float));
    if (!nimcp_validate_pointer(input_float, "input_float")) {
        NIMCP_LOGGING_ERROR("Failed to allocate input buffer");
        return false;
    }

    // Convert to grayscale if RGB
    for (uint32_t i = 0; i < input_size; i++) {
        float val = 0.0f;
        for (uint32_t c = 0; c < channels; c++) {
            val += image[i * channels + c];
        }
        input_float[i] = (val / channels) / 255.0f;
    }

    // V1: Edge detection
    uint32_t v1_output_w = conv_layer_get_output_width(cortex->v1_layer);
    uint32_t v1_output_h = conv_layer_get_output_height(cortex->v1_layer);
    uint32_t v1_output_size = v1_output_w * v1_output_h * cortex->num_v1_filters;

    float* v1_output = (float*)nimcp_calloc(v1_output_size, sizeof(float));
    if (!nimcp_validate_pointer(v1_output, "v1_output")) {
        NIMCP_LOGGING_ERROR("Failed to allocate V1 output buffer");
        nimcp_free(input_float);
        return false;
    }

    if (!conv_layer_forward(cortex->v1_layer, input_float, v1_output)) {
        nimcp_free(input_float);
        nimcp_free(v1_output);
        return false;
    }

    // Pooling
    uint32_t pooled_w = v1_output_w / 2;
    uint32_t pooled_h = v1_output_h / 2;
    uint32_t pooled_size = pooled_w * pooled_h * cortex->num_v1_filters;

    float* pooled_output = (float*)nimcp_calloc(pooled_size, sizeof(float));
    if (!nimcp_validate_pointer(pooled_output, "pooled_output")) {
        NIMCP_LOGGING_ERROR("Failed to allocate pooled output buffer");
        nimcp_free(input_float);
        nimcp_free(v1_output);
        return false;
    }

    if (!pool_layer_forward(cortex->pool_layer, v1_output, pooled_output)) {
        nimcp_free(input_float);
        nimcp_free(v1_output);
        nimcp_free(pooled_output);
        return false;
    }

    // Feature extraction (simplified: average pooling to feature_dim)
    memset(features, 0, cortex->feature_dim * sizeof(float));

    uint32_t features_per_bin = (pooled_size + cortex->feature_dim - 1) / cortex->feature_dim;
    for (uint32_t i = 0; i < pooled_size; i++) {
        uint32_t bin = i / features_per_bin;
        if (bin < cortex->feature_dim) {
            features[bin] += pooled_output[i];
        }
    }

    // Normalize features
    float norm = 0.0f;
    for (uint32_t i = 0; i < cortex->feature_dim; i++) {
        norm += features[i] * features[i];
    }
    norm = sqrtf(norm);
    if (norm > 1e-6f) {
        for (uint32_t i = 0; i < cortex->feature_dim; i++) {
            features[i] /= norm;
        }
    }

    // Update statistics
    cortex->images_processed++;
    clock_t end = clock();
    cortex->total_processing_time += ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;

    nimcp_free(input_float);
    nimcp_free(v1_output);
    nimcp_free(pooled_output);

    return true;
}

/**
 * WHAT: Compute visual attention
 */
bool visual_cortex_compute_attention(
    visual_cortex_t* cortex,
    const uint8_t* image,
    uint32_t width,
    uint32_t height,
    attention_map_t* attn_map)
{
    // Guard: Validate inputs
    if (!nimcp_validate_pointer(cortex, "cortex") ||
        !nimcp_validate_pointer(image, "image") ||
        !nimcp_validate_pointer(attn_map, "attn_map")) {
        return false;
    }

    if (!cortex->enable_attention) {
        return false;
    }

    // Compute gradient magnitude (simple edge-based attention)
    for (uint32_t y = 1; y < height - 1; y++) {
        for (uint32_t x = 1; x < width - 1; x++) {
            float gx = (float)image[(y * width + x + 1)] - (float)image[(y * width + x - 1)];
            float gy = (float)image[((y + 1) * width + x)] - (float)image[((y - 1) * width + x)];
            float magnitude = sqrtf(gx * gx + gy * gy) / 255.0f;

            attention_map_set(attn_map, x, y, magnitude);
        }
    }

    return true;
}

/**
 * WHAT: Store visual memory
 */
bool visual_cortex_store_memory(
    visual_cortex_t* cortex,
    const float* features,
    float salience)
{
    // Guard: Validate inputs
    if (!nimcp_validate_pointer(cortex, "cortex") ||
        !nimcp_validate_pointer(features, "features")) {
        return false;
    }

    if (!cortex->enable_memory) {
        return false;
    }

    if (cortex->num_memories >= MAX_VISUAL_MEMORIES) {
        return false;  // Memory full
    }

    // Allocate memory entry
    visual_memory_t* memory = (visual_memory_t*)nimcp_calloc(1, sizeof(visual_memory_t));
    if (!nimcp_validate_pointer(memory, "memory")) {
        NIMCP_LOGGING_ERROR("Failed to allocate visual memory entry");
        return false;
    }

    memory->feature_dim = cortex->feature_dim;
    memory->features = (float*)nimcp_calloc(cortex->feature_dim, sizeof(float));
    if (!nimcp_validate_pointer(memory->features, "memory->features")) {
        NIMCP_LOGGING_ERROR("Failed to allocate visual memory features");
        nimcp_free(memory);
        return false;
    }

    memcpy(memory->features, features, cortex->feature_dim * sizeof(float));
    memory->salience = salience;
    memory->timestamp = (uint64_t)time(NULL);

    cortex->memories[cortex->num_memories++] = memory;
    return true;
}

/**
 * WHAT: Recall visual memories
 */
bool visual_cortex_recall_memory(
    visual_cortex_t* cortex,
    const float* query_features,
    int max_results,
    visual_memory_t*** memories,
    int* num_memories)
{
    // Guard: Validate inputs
    if (!nimcp_validate_pointer(cortex, "cortex") ||
        !nimcp_validate_pointer(query_features, "query_features") ||
        !nimcp_validate_pointer(memories, "memories") ||
        !nimcp_validate_pointer(num_memories, "num_memories")) {
        return false;
    }

    if (!cortex->enable_memory || cortex->num_memories == 0) {
        *memories = NULL;
        *num_memories = 0;
        return true;
    }

    // Compute similarity to all memories
    typedef struct {
        visual_memory_t* memory;
        float similarity;
    } memory_similarity_t;

    memory_similarity_t* similarities = (memory_similarity_t*)nimcp_calloc(
        cortex->num_memories, sizeof(memory_similarity_t));
    if (!nimcp_validate_pointer(similarities, "similarities")) {
        NIMCP_LOGGING_ERROR("Failed to allocate similarity buffer");
        return false;
    }

    for (uint32_t i = 0; i < cortex->num_memories; i++) {
        similarities[i].memory = cortex->memories[i];

        // Cosine similarity
        float dot = 0.0f;
        for (uint32_t j = 0; j < cortex->feature_dim; j++) {
            dot += query_features[j] * cortex->memories[i]->features[j];
        }
        similarities[i].similarity = dot;
    }

    // Sort by similarity (simple bubble sort for small arrays)
    for (uint32_t i = 0; i < cortex->num_memories - 1; i++) {
        for (uint32_t j = 0; j < cortex->num_memories - i - 1; j++) {
            if (similarities[j].similarity < similarities[j + 1].similarity) {
                memory_similarity_t temp = similarities[j];
                similarities[j] = similarities[j + 1];
                similarities[j + 1] = temp;
            }
        }
    }

    // Return top results
    int num_results = (max_results < (int)cortex->num_memories) ? max_results : (int)cortex->num_memories;
    *memories = (visual_memory_t**)nimcp_calloc(num_results, sizeof(visual_memory_t*));
    if (!nimcp_validate_pointer(*memories, "result_memories")) {
        NIMCP_LOGGING_ERROR("Failed to allocate memory results array");
        nimcp_free(similarities);
        return false;
    }

    for (int i = 0; i < num_results; i++) {
        (*memories)[i] = similarities[i].memory;
    }
    *num_memories = num_results;

    nimcp_free(similarities);
    return true;
}

/**
 * WHAT: Get statistics
 */
bool visual_cortex_get_stats(const visual_cortex_t* cortex, visual_cortex_stats_t* stats)
{
    if (!nimcp_validate_pointer(cortex, "cortex") ||
        !nimcp_validate_pointer(stats, "stats")) {
        return false;
    }

    stats->images_processed = cortex->images_processed;
    stats->memories_stored = cortex->num_memories;
    stats->avg_processing_time = (cortex->images_processed > 0) ?
        (float)(cortex->total_processing_time / cortex->images_processed) : 0.0f;

    // Estimate memory usage (rough)
    stats->memory_usage_mb = (sizeof(visual_cortex_t) +
                              cortex->num_memories * (sizeof(visual_memory_t) + cortex->feature_dim * sizeof(float))) /
                             (1024.0f * 1024.0f);

    return true;
}

//=============================================================================
// Brain Integration Helper Functions
//=============================================================================

/**
 * WHAT: Compute novelty score for curiosity system
 * WHY:  Drive exploration of novel visual patterns
 * HOW:  Compare features against visual memory, return novelty (0-1)
 *
 * High novelty (close to 1.0) = never seen before → triggers curiosity
 * Low novelty (close to 0.0) = familiar → less interesting
 */
float visual_cortex_compute_novelty(visual_cortex_t* cortex, const float* features)
{
    // Guard: Validate inputs
    if (!nimcp_validate_pointer(cortex, "cortex") ||
        !nimcp_validate_pointer(features, "features")) {
        return 0.0f;
    }

    // If no memories, everything is novel
    if (cortex->num_memories == 0) {
        return 1.0f;
    }

    // Find maximum similarity to existing memories
    float max_similarity = 0.0f;
    for (uint32_t i = 0; i < cortex->num_memories; i++) {
        // Cosine similarity
        float dot = 0.0f;
        for (uint32_t j = 0; j < cortex->feature_dim; j++) {
            dot += features[j] * cortex->memories[i]->features[j];
        }

        if (dot > max_similarity) {
            max_similarity = dot;
        }
    }

    // Novelty = 1 - similarity (range 0-1)
    // High similarity → low novelty
    // Low similarity → high novelty
    float novelty = 1.0f - max_similarity;

    // Clamp to [0, 1]
    if (novelty < 0.0f) novelty = 0.0f;
    if (novelty > 1.0f) novelty = 1.0f;

    return novelty;
}

/**
 * WHAT: Get maximum attention location
 * WHY:  Identify most salient region for attention system
 * HOW:  Find peak in attention map
 *
 * Returns the (x, y) location with highest attention value
 */
bool visual_cortex_get_attention_peak(
    const attention_map_t* attn_map,
    uint32_t* max_x,
    uint32_t* max_y,
    float* max_value)
{
    // Guard: Validate inputs
    if (!nimcp_validate_pointer(attn_map, "attn_map") ||
        !nimcp_validate_pointer(max_x, "max_x") ||
        !nimcp_validate_pointer(max_y, "max_y") ||
        !nimcp_validate_pointer(max_value, "max_value")) {
        return false;
    }

    if (!nimcp_validate_pointer(attn_map->values, "attn_map->values")) {
        NIMCP_LOGGING_ERROR("Attention map has no values");
        return false;
    }

    // Find maximum
    *max_value = -INFINITY;
    *max_x = 0;
    *max_y = 0;

    for (uint32_t y = 0; y < attn_map->height; y++) {
        for (uint32_t x = 0; x < attn_map->width; x++) {
            float value = attn_map->values[y * attn_map->width + x];
            if (value > *max_value) {
                *max_value = value;
                *max_x = x;
                *max_y = y;
            }
        }
    }

    return true;
}

/**
 * WHAT: Prepare visual features for memory consolidation
 * WHY:  Enable hippocampus to store visual experiences
 * HOW:  Package features with metadata for memory system
 *
 * This function stores visual features in the visual cortex's local memory
 * AND prepares them for transfer to the hippocampus for long-term consolidation.
 *
 * Integration flow:
 * Visual Cortex → local storage (fast recall)
 *              → Hippocampus (long-term consolidation)
 *              → Knowledge Graph (semantic integration)
 */
bool visual_cortex_consolidate_memory(
    visual_cortex_t* cortex,
    const float* features,
    float salience,
    const char* context)
{
    // Guard: Validate inputs
    if (!nimcp_validate_pointer(cortex, "cortex") ||
        !nimcp_validate_pointer(features, "features")) {
        return false;
    }

    // Store in local visual memory
    if (!visual_cortex_store_memory(cortex, features, salience)) {
        return false;
    }

    // TODO: Integration with hippocampus
    // This is a placeholder for future hippocampus integration
    // Will call: hippocampus_consolidate_visual_memory(features, salience, context)
    //
    // The hippocampus would:
    // 1. Store visual experience with episodic context
    // 2. Link to semantic knowledge graph
    // 3. Enable sleep-based memory consolidation
    // 4. Support visual imagination/recall
    (void)context;  // Suppress unused warning until hippocampus integration

    return true;
}
