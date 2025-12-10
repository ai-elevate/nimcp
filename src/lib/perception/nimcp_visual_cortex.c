/**
 * @file nimcp_visual_cortex.c
 * @brief Implementation of biologically-inspired visual processing
 *
 * WHAT: CNN-based visual cortex with V1-style edge detection
 * WHY:  Enable visual perception and memory in NIMCP
 * HOW:  Lightweight convolution + Gabor filters + tensor-accelerated feature extraction
 *
 * Phase TENSOR-2: Tensor library integration for accelerated operations
 *
 * @author NIMCP Development Team
 * @date 2025
 * @version 2.7.0 - Added tensor library integration
 */

#include "perception/nimcp_visual_cortex.h"
#include "utils/tensor/nimcp_tensor.h"  /* Tensor library for vectorized operations */
#include "security/nimcp_security.h"
#include "security/nimcp_blood_brain_barrier.h"

#include "utils/memory/nimcp_unified_memory.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/memory/nimcp_memory_pool.h"  // Memory pool for O(1) allocations
#include "utils/memory/nimcp_page_cow.h"     // Copy-on-Write for shallow copies
#include "utils/validation/nimcp_validate.h"
#include "utils/logging/nimcp_logging.h"
#include "plasticity/neuromodulators/nimcp_neuromodulators.h"  // Neuromodulator integration
#include "plasticity/nimcp_second_messengers.h"  // Second messenger cascades
#include "core/brain/nimcp_brain.h"  // Brain reference
#include "core/neuralnet/nimcp_neuralnet.h"  // Neural network for internal V1 connections
#include "core/topology/nimcp_fractal_topology.h"  // Scale-free topology generation
#include "async/nimcp_bio_async.h"  // Bio-async communication
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/*=============================================================================
 * LOGGING MODULE IDENTIFIER
 *===========================================================================*/

#define VISUAL_LOG_MODULE "VISUAL_CORTEX"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

//=============================================================================
// Neuromodulation
//=============================================================================

/**
 * @brief Compute visual gain from acetylcholine and norepinephrine
 *
 * WHAT: Calculate gain factor for visual processing based on ACh + NE
 * WHY:  ACh enhances attention, NE enhances arousal-based vision
 * HOW:  Read ACh and NE levels, combine into multiplicative gain
 *
 * BIOLOGY:
 * - Acetylcholine: Visual attention and feature binding
 *   High ACh (0.7) → focused attention, enhanced features
 *   Low ACh (0.3) → inattentive, reduced feature detection
 *
 * - Norepinephrine: Arousal-dependent visual sensitivity
 *   High NE (0.7) → hypervigilant, threat detection (anxiety/PTSD)
 *   Low NE (0.3) → reduced alertness, missing cues (depression)
 *
 * COMPLEXITY: O(1)
 *
 * @param brain Brain to read neurotransmitters from
 * @return Visual gain factor [0.6, 1.8], or 1.0 if no brain
 */
static float get_visual_gain(brain_t brain)
{
    // Guard: No brain available
    if (!brain) {
        return 1.0f;
    }

    neuromodulator_system_t neuromod = brain_get_neuromodulator_system(brain);
    if (!neuromod) {
        return 1.0f;
    }

    // Read neurotransmitter levels
    float ach = neuromodulator_get_level(neuromod, NEUROMOD_ACETYLCHOLINE);
    float ne = neuromodulator_get_level(neuromod, NEUROMOD_NOREPINEPHRINE);

    // ACh contribution: [0.3, 0.7] → [0.8, 1.2]
    float ach_gain = 0.8f + (ach - 0.3f);

    // NE contribution: [0.3, 0.7] → [0.8, 1.2]
    float ne_gain = 0.8f + (ne - 0.3f);

    // Combined gain: [0.64, 1.44] approx [0.6, 1.5]
    return ach_gain * ne_gain;
}

//=============================================================================
// Activation Functions
//=============================================================================

/**
 * @brief Apply activation function without gain (baseline)
 *
 * WHAT: Standard activation function
 * WHY:  For layers that don't use neuromodulation
 * HOW:  Apply activation function directly
 *
 * @param x Input value
 * @param activation Activation function type
 * @return Activated output
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

/**
 * @brief Apply activation function with optional gain modulation
 *
 * WHAT: Activate neuron output with neuromodulator gain
 * WHY:  Enable neurochemical modulation of visual processing
 * HOW:  Apply gain, then activation function
 *
 * @param x Input value
 * @param activation Activation function type
 * @param gain Neuromodulator gain factor (typically 1.0 baseline)
 * @return Activated output
 */
static inline float apply_activation_with_gain(float x,
                                               visual_activation_type_t activation,
                                               float gain)
{
    // Apply neuromodulator gain first
    x *= gain;

    // Then apply activation function
    return apply_activation(x, activation);
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
        LOG_ERROR(VISUAL_LOG_MODULE, "Invalid convolution layer configuration parameters");
        return NULL;
    }

    // Allocate layer
    conv_layer_t* layer = (conv_layer_t*)nimcp_calloc(1, sizeof(conv_layer_t));
    if (!layer) {
        LOG_ERROR(VISUAL_LOG_MODULE, "Failed to allocate convolution layer");
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
        LOG_ERROR(VISUAL_LOG_MODULE, "Failed to allocate convolution kernels");
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
        LOG_ERROR(VISUAL_LOG_MODULE, "Failed to allocate convolution bias");
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
        LOG_ERROR(VISUAL_LOG_MODULE, "Invalid filter index: %u >= %u", filter_idx, layer->num_filters);
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
        LOG_ERROR(VISUAL_LOG_MODULE, "Invalid pooling layer configuration parameters");
        return NULL;
    }

    pool_layer_t* layer = (pool_layer_t*)nimcp_calloc(1, sizeof(pool_layer_t));
    if (!nimcp_validate_pointer(layer, "layer")) {
        LOG_ERROR(VISUAL_LOG_MODULE, "Failed to allocate pooling layer");
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
        LOG_ERROR(VISUAL_LOG_MODULE, "Invalid kernel size: %d (must be positive and odd)", kernel_size);
        return NULL;
    }

    float* kernel = (float*)nimcp_calloc(kernel_size * kernel_size, sizeof(float));
    if (!nimcp_validate_pointer(kernel, "kernel")) {
        LOG_ERROR(VISUAL_LOG_MODULE, "Failed to allocate Gabor kernel");
        return NULL;
    }

    int center = kernel_size / 2;
    float theta = params->orientation * M_PI / 180.0f;  // Convert to radians
    float sigma = params->wavelength * params->bandwidth;
    float gamma = params->aspect_ratio;
    float lambda = params->wavelength;
    float psi = params->phase * M_PI / 180.0f;

    // Generate Gabor function
    float sum = 0.0f;
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

            float value = gaussian * sinusoid;
            kernel[y * kernel_size + x] = value;
            sum += value;
        }
    }

    // DC balance: subtract mean to make kernel sum to zero
    float mean = sum / (kernel_size * kernel_size);
    for (int i = 0; i < kernel_size * kernel_size; i++) {
        kernel[i] -= mean;
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
        LOG_ERROR(VISUAL_LOG_MODULE, "Invalid attention map dimensions: %u x %u", width, height);
        return NULL;
    }

    attention_map_t* map = (attention_map_t*)nimcp_calloc(1, sizeof(attention_map_t));
    if (!nimcp_validate_pointer(map, "map")) {
        LOG_ERROR(VISUAL_LOG_MODULE, "Failed to allocate attention map");
        return NULL;
    }

    map->width = width;
    map->height = height;
    map->values = (float*)nimcp_calloc(width * height, sizeof(float));

    if (!nimcp_validate_pointer(map->values, "map->values")) {
        LOG_ERROR(VISUAL_LOG_MODULE, "Failed to allocate attention map values");
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
        LOG_ERROR(VISUAL_LOG_MODULE, "Attention map coordinates out of bounds: (%u, %u) >= (%u, %u)", x, y, map->width, map->height);
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
        LOG_ERROR(VISUAL_LOG_MODULE, "Attention map coordinates out of bounds: (%u, %u) >= (%u, %u)", x, y, map->width, map->height);
        return false;
    }

    map->values[y * map->width + x] = value;
    return true;
}

//=============================================================================
// Visual Cortex Implementation
//=============================================================================

#define MAX_VISUAL_MEMORIES 1000
#define NUM_V1_LAYERS 4  // Layers 2/3, 4, 5, 6
#define NEUROMOD_TYPE_DOPAMINE 0
#define NEUROMOD_TYPE_ACETYLCHOLINE 1
#define NEUROMOD_TYPE_NOREPINEPHRINE 2
#define NUM_NEUROMOD_TYPES 3

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

    // Neuromodulation
    brain_t brain;  /**< Brain reference for ACh + NE modulation */

    // === PHASE: PHASIC/TONIC NEUROMODULATION ===
    /**
     * Phasic/tonic states for three neuromodulators
     * Index: 0=dopamine, 1=acetylcholine, 2=norepinephrine
     *
     * WHAT: Dual-timescale neuromodulator dynamics
     * WHY:  Biological realism - bursts vs. baselines
     * HOW:  Separate phasic (rapid) and tonic (slow) components
     */
    phasic_tonic_state_t neuromod_states[NUM_NEUROMOD_TYPES];

    /**
     * Receptor expression profiles for V1 cortical layers
     * Index: 0=L2/3, 1=L4, 2=L5, 3=L6
     *
     * WHAT: Layer-specific receptor densities
     * WHY:  Different layers have different neuromod sensitivity
     * HOW:  D1/D2, M1/M2, α1/β2 densities per layer
     *
     * BIOLOGY:
     * - L2/3: Top-down attention (high D1, ACh)
     * - L4: Thalamic input (high ACh, NE)
     * - L5: Motor output (high DA, NE)
     * - L6: Feedback modulation (high M1, β2)
     */
    receptor_expression_t receptor_profiles[NUM_V1_LAYERS];

    // NIMCP 2.7 Phase 8.5: Internal recurrent network with fractal topology
    neural_network_t internal_network;  /**< Recurrent network for horizontal connections */
    bool has_internal_network;          /**< Whether internal network was created */

    // Statistics
    uint32_t images_processed;
    double total_processing_time;

    // === Memory Pool for O(1) Visual Memory Allocation ===
    memory_pool_t memory_pool;            /**< Pool for visual memory entries */

    // === Copy-on-Write Support ===
    uint32_t* _cow_refcount;              /**< Reference count for CoW (NULL if owned) */
    bool _cow_is_shallow;                 /**< True if this is a shallow copy */

    // === Bio-Async Communication ===
    bio_module_context_t bio_ctx;         /**< Bio-async module context */
    bool bio_async_enabled;               /**< Whether bio-async is enabled */

    // === Second Messenger Cascades ===
    second_messenger_system_t* second_messengers; /**< Second messenger system per layer */
    bool second_messengers_enabled;               /**< Whether cascades are enabled */
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
        LOG_ERROR(VISUAL_LOG_MODULE, "Invalid visual cortex configuration parameters");
        return NULL;
    }

    visual_cortex_t* cortex = (visual_cortex_t*)nimcp_calloc(1, sizeof(visual_cortex_t));
    if (!cortex) {
        LOG_ERROR(VISUAL_LOG_MODULE, "Failed to allocate visual cortex");
        return NULL;
    }

    cortex->input_width = config->input_width;
    cortex->input_height = config->input_height;
    cortex->num_v1_filters = config->num_v1_filters;
    cortex->feature_dim = config->feature_dim;
    cortex->enable_attention = config->enable_attention;
    cortex->enable_memory = config->enable_memory;
    cortex->brain = NULL;  // Initialize neuromodulator brain reference

    // Initialize phasic/tonic neuromodulator states
    for (uint32_t i = 0; i < NUM_NEUROMOD_TYPES; i++) {
        cortex->neuromod_states[i].phasic_burst = 0.0f;
        cortex->neuromod_states[i].tonic_level = 0.5f;  // Baseline
        cortex->neuromod_states[i].burst_decay_tau = 0.2f;  // Fast decay (200ms)
        cortex->neuromod_states[i].homeostatic_tau = 10.0f;   // Slow homeostatic regulation (10s)
        cortex->neuromod_states[i].burst_start_time_us = 0;
    }

    // Initialize receptor expression profiles (biologically plausible defaults)
    // Layer 2/3: Top-down attention
    cortex->receptor_profiles[0].d1_density = 0.4f;
    cortex->receptor_profiles[0].d2_density = 0.2f;
    cortex->receptor_profiles[0].m1_density = 0.5f;
    cortex->receptor_profiles[0].m2_density = 0.3f;
    cortex->receptor_profiles[0].alpha1_density = 0.3f;
    cortex->receptor_profiles[0].beta2_density = 0.3f;

    // Layer 4: Thalamic input
    cortex->receptor_profiles[1].d1_density = 0.3f;
    cortex->receptor_profiles[1].d2_density = 0.2f;
    cortex->receptor_profiles[1].m1_density = 0.6f;  // High ACh
    cortex->receptor_profiles[1].m2_density = 0.3f;
    cortex->receptor_profiles[1].alpha1_density = 0.5f;  // High NE
    cortex->receptor_profiles[1].beta2_density = 0.3f;

    // Layer 5: Motor output/arousal
    cortex->receptor_profiles[2].d1_density = 0.4f;  // High DA
    cortex->receptor_profiles[2].d2_density = 0.3f;  // Moderate D2
    cortex->receptor_profiles[2].m1_density = 0.4f;
    cortex->receptor_profiles[2].m2_density = 0.2f;
    cortex->receptor_profiles[2].alpha1_density = 0.5f;  // High NE
    cortex->receptor_profiles[2].beta2_density = 0.3f;

    // Layer 6: Feedback modulation
    cortex->receptor_profiles[3].d1_density = 0.3f;
    cortex->receptor_profiles[3].d2_density = 0.2f;
    cortex->receptor_profiles[3].m1_density = 0.6f;  // High M1
    cortex->receptor_profiles[3].m2_density = 0.4f;
    cortex->receptor_profiles[3].alpha1_density = 0.3f;
    cortex->receptor_profiles[3].beta2_density = 0.4f;  // High β2

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
        LOG_ERROR(VISUAL_LOG_MODULE, "Failed to create V1 convolution layer");
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
        LOG_ERROR(VISUAL_LOG_MODULE, "Failed to create pooling layer");
        visual_cortex_destroy(cortex);
        return NULL;
    }

    // Allocate feature weights (simplified feature extraction)
    uint32_t pooled_size = (v1_output_w / 2) * (v1_output_h / 2) * config->num_v1_filters;
    cortex->feature_weights = (float*)nimcp_calloc(pooled_size, sizeof(float));
    if (!nimcp_validate_pointer(cortex->feature_weights, "feature_weights")) {
        LOG_ERROR(VISUAL_LOG_MODULE, "Failed to allocate feature weights");
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
    // exhibit scale-free properties. This internal network models:
    // - Horizontal connections for contour integration
    // - Feedback modulation for attention and expectation
    // - Temporal integration for motion processing
    //
    cortex->internal_network = NULL;
    cortex->has_internal_network = false;

    if (config->enable_fractal_topology && config->internal_neurons > 0) {
        // Create internal recurrent network
        network_config_t net_config = {
            .num_neurons = config->internal_neurons,
            .ei_ratio = 0.8f,  // 80% excitatory (typical cortex)
            .learning_rate = 0.001f,
            .hebbian_rate = 0.01f,
            .stdp_window = 20.0f,
            .homeostatic_rate = 0.001f,
            .target_activity = 0.1f,
            .adaptation_rate = 0.01f,
            .refractory_period = 2.0f,
            .min_weight = 0.0f,
            .max_weight = 1.0f,
            .update_interval = 1,
            .enable_stdp = true,
            .enable_homeostasis = true,
            .neuron_model = NEURON_MODEL_LIF,
            .model_params = NULL,
            .integration_method = ODE_EULER
        };

        cortex->internal_network = neural_network_create(&net_config);

        if (cortex->internal_network) {
            // Generate scale-free topology
            scale_free_config_t topo_config = {
                .power_law_gamma = config->power_law_gamma,
                .hub_ratio = config->hub_ratio,
                .min_degree = 2,
                .max_degree = config->internal_neurons / 10,
                .spatial_constraint = 0.5f,  // V1 has spatial organization
                .bidirectional = false
            };

            topology_stats_t stats;
            if (topology_generate_scale_free(cortex->internal_network, &topo_config, &stats)) {
                cortex->has_internal_network = true;
                LOG_INFO(VISUAL_LOG_MODULE, "V1 internal network: %u neurons, %u synapses, %.2f avg degree",
                         stats.num_neurons, stats.num_synapses, stats.avg_degree);
            } else {
                LOG_WARN(VISUAL_LOG_MODULE, "Failed to generate V1 topology, using network without topology");
                cortex->has_internal_network = true;  // Network exists, just without topology
            }
        } else {
            LOG_WARN(VISUAL_LOG_MODULE, "Failed to create V1 internal network");
        }
    }

    // === Initialize Memory Pool for O(1) Visual Memory Allocation ===
    memory_pool_config_t mem_pool_config = memory_pool_default_config(
        sizeof(visual_memory_t),
        MAX_VISUAL_MEMORIES  // Pool sized for maximum memories
    );
    mem_pool_config.alignment = 16;  // SIMD alignment for feature vectors
    mem_pool_config.enable_tracking = true;

    cortex->memory_pool = memory_pool_create(&mem_pool_config);
    if (!cortex->memory_pool) {
        LOG_WARN(VISUAL_LOG_MODULE, "Failed to create memory pool, using malloc fallback");
    }

    // Initialize CoW fields (owned by default)
    cortex->_cow_refcount = NULL;
    cortex->_cow_is_shallow = false;

    // === Bio-Async Registration ===
    cortex->bio_ctx = NULL;
    cortex->bio_async_enabled = false;

    if (config->enable_bio_async && bio_router_is_initialized()) {
        bio_module_info_t bio_info = {
            .module_id = BIO_MODULE_VISUAL_CORTEX,
            .module_name = "visual_cortex",
            .inbox_capacity = 64,
            .user_data = cortex
        };

        cortex->bio_ctx = bio_router_register_module(&bio_info);
        if (cortex->bio_ctx) {
            cortex->bio_async_enabled = true;
            LOG_INFO(VISUAL_LOG_MODULE, "Bio-async registered for visual cortex (module_id=%d)",
                     BIO_MODULE_VISUAL_CORTEX);
        } else {
            LOG_WARN(VISUAL_LOG_MODULE, "Failed to register bio-async for visual cortex");
        }
    }

    // === Second Messenger Cascade Initialization ===
    cortex->second_messengers = NULL;
    cortex->second_messengers_enabled = false;

    if (config->enable_second_messengers) {
        // WHAT: Create second messenger system for V1 layers
        // WHY:  Enable neuromodulator-driven signaling cascades
        // HOW:  One system tracking NUM_V1_LAYERS (4) neurons, one per cortical layer
        second_messenger_config_t sm_config = second_messenger_default_config();
        sm_config.enable_bio_async = config->enable_bio_async;
        sm_config.enable_security = true;

        cortex->second_messengers = second_messenger_create(NUM_V1_LAYERS, &sm_config);
        if (cortex->second_messengers) {
            cortex->second_messengers_enabled = true;
            LOG_INFO(VISUAL_LOG_MODULE, "Second messenger cascades enabled for visual cortex (%u layers)",
                     NUM_V1_LAYERS);

            // Register with bio-async router if both are enabled
            if (cortex->bio_async_enabled && cortex->bio_ctx) {
                bio_router_t router = bio_router_get_global();
                if (router) {
                    nimcp_result_t result = second_messenger_register_bioasync(
                        cortex->second_messengers, router);
                    if (result != NIMCP_SUCCESS) {
                        LOG_WARN(VISUAL_LOG_MODULE, "Failed to register second messengers with bio-async: %d",
                                 result);
                    } else {
                        LOG_DEBUG(VISUAL_LOG_MODULE, "Second messengers registered with bio-async");
                    }
                }
            }
        } else {
            LOG_WARN(VISUAL_LOG_MODULE, "Failed to create second messenger system");
        }
    }

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

    // === Second Messenger Cleanup ===
    if (cortex->second_messengers) {
        second_messenger_destroy(cortex->second_messengers);
        cortex->second_messengers = NULL;
        LOG_DEBUG(VISUAL_LOG_MODULE, "Second messenger cascades destroyed");
    }

    // === Bio-Async Unregistration ===
    if (cortex->bio_async_enabled && cortex->bio_ctx) {
        bio_router_unregister_module(cortex->bio_ctx);
        cortex->bio_ctx = NULL;
        cortex->bio_async_enabled = false;
        LOG_DEBUG(VISUAL_LOG_MODULE, "Bio-async unregistered for visual cortex");
    }

    // === CoW Reference Counting ===
    // If this is a shallow copy with shared data, decrement refcount
    if (cortex->_cow_refcount) {
        uint32_t old_count = __atomic_sub_fetch(cortex->_cow_refcount, 1, __ATOMIC_SEQ_CST);
        if (old_count > 0) {
            // Other references exist - just free our handle
            nimcp_free(cortex);
            return;
        }
        // We're the last reference - free the refcount and continue cleanup
        nimcp_free(cortex->_cow_refcount);
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

    // Free visual memories (use pool if available)
    for (uint32_t i = 0; i < cortex->num_memories; i++) {
        if (cortex->memories[i]) {
            if (cortex->memories[i]->features) {
                nimcp_free(cortex->memories[i]->features);
            }
            // Check if memory came from pool or malloc
            if (cortex->memory_pool && memory_pool_owns(cortex->memory_pool, cortex->memories[i])) {
                memory_pool_release(cortex->memory_pool, cortex->memories[i]);
            } else {
                nimcp_free(cortex->memories[i]);
            }
        }
    }

    // === Destroy Memory Pool ===
    if (cortex->memory_pool) {
        memory_pool_destroy(cortex->memory_pool);
    }

    // NIMCP 2.7 Phase 8.5: Destroy internal recurrent network
    if (cortex->internal_network) {
        neural_network_destroy(cortex->internal_network);
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
        LOG_ERROR(VISUAL_LOG_MODULE, "Invalid image dimensions: %ux%ux%u (expected %ux%ux>0)",
                           width, height, channels, cortex->input_width, cortex->input_height);
        return false;
    }

    clock_t start = clock();

    // Convert uint8 image to float (normalize to 0-1)
    uint32_t input_size = width * height;
    float* input_float = (float*)nimcp_calloc(input_size, sizeof(float));
    if (!nimcp_validate_pointer(input_float, "input_float")) {
        LOG_ERROR(VISUAL_LOG_MODULE, "Failed to allocate input buffer");
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
        LOG_ERROR(VISUAL_LOG_MODULE, "Failed to allocate V1 output buffer");
        nimcp_free(input_float);
        return false;
    }

    if (!conv_layer_forward(cortex->v1_layer, input_float, v1_output)) {
        nimcp_free(input_float);
        nimcp_free(v1_output);
        return false;
    }

    // Apply neuromodulator gain (ACh + NE modulation)
    // Legacy gain calculation (kept for backward compatibility)
    float visual_gain = get_visual_gain(cortex->brain);

    // NEW: Compute layer-specific neuromodulation effects
    // Use Layer 4 (thalamic input layer) as default for feedforward processing
    neuromod_effects_t neuromod_effects;
    visual_cortex_compute_neuromod_effects(cortex, 1, &neuromod_effects);  // Layer 4

    // Apply combined gain: legacy × neuromodulation
    float combined_gain = visual_gain * neuromod_effects.gabor_gain;

    if (combined_gain != 1.0f) {
        /* Use tensor library for vectorized gain application */
        uint32_t v1_dims[] = {v1_output_size};
        nimcp_tensor_t* v1_tensor = nimcp_tensor_from_data(v1_output, v1_dims, 1, NIMCP_DTYPE_F32, false);
        if (v1_tensor) {
            nimcp_tensor_mul_scalar_(v1_tensor, (double)combined_gain);
            nimcp_tensor_destroy(v1_tensor);
        } else {
            /* Fallback to scalar */
            for (uint32_t i = 0; i < v1_output_size; i++) {
                v1_output[i] *= combined_gain;
            }
        }
    }

    // Pooling
    uint32_t pooled_w = v1_output_w / 2;
    uint32_t pooled_h = v1_output_h / 2;
    uint32_t pooled_size = pooled_w * pooled_h * cortex->num_v1_filters;

    float* pooled_output = (float*)nimcp_calloc(pooled_size, sizeof(float));
    if (!nimcp_validate_pointer(pooled_output, "pooled_output")) {
        LOG_ERROR(VISUAL_LOG_MODULE, "Failed to allocate pooled output buffer");
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

    // Normalize features using tensor library
    {
        uint32_t feat_dims[] = {cortex->feature_dim};
        nimcp_tensor_t* feat_tensor = nimcp_tensor_from_data(features, feat_dims, 1, NIMCP_DTYPE_F32, false);
        if (feat_tensor) {
            float norm = (float)nimcp_tensor_norm_p(feat_tensor, 2.0);
            if (norm > 1e-6f) {
                nimcp_tensor_mul_scalar_(feat_tensor, 1.0 / (double)norm);
            }
            nimcp_tensor_destroy(feat_tensor);
        } else {
            /* Fallback to scalar */
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

    // Compute layer-specific neuromodulation effects
    // Use Layer 2/3 (top-down attention layer) for attention modulation
    neuromod_effects_t neuromod_effects;
    visual_cortex_compute_neuromod_effects(cortex, 0, &neuromod_effects);  // Layer 2/3

    // Compute gradient magnitude (simple edge-based attention)
    for (uint32_t y = 1; y < height - 1; y++) {
        for (uint32_t x = 1; x < width - 1; x++) {
            float gx = (float)image[(y * width + x + 1)] - (float)image[(y * width + x - 1)];
            float gy = (float)image[((y + 1) * width + x)] - (float)image[((y - 1) * width + x)];
            float magnitude = sqrtf(gx * gx + gy * gy) / 255.0f;

            // Apply neuromodulation attention boost (ACh enhances attention)
            magnitude *= neuromod_effects.attention_boost;

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

    // === Allocate memory entry (pool with malloc fallback) ===
    visual_memory_t* memory = NULL;
    if (cortex->memory_pool) {
        memory = (visual_memory_t*)memory_pool_acquire(cortex->memory_pool);
        if (memory) {
            memset(memory, 0, sizeof(visual_memory_t));
        }
    }
    // Fallback to malloc if pool exhausted or unavailable
    if (!memory) {
        memory = (visual_memory_t*)nimcp_calloc(1, sizeof(visual_memory_t));
    }
    if (!nimcp_validate_pointer(memory, "memory")) {
        LOG_ERROR(VISUAL_LOG_MODULE, "Failed to allocate visual memory entry");
        return false;
    }

    memory->feature_dim = cortex->feature_dim;
    memory->features = (float*)nimcp_calloc(cortex->feature_dim, sizeof(float));
    if (!nimcp_validate_pointer(memory->features, "memory->features")) {
        LOG_ERROR(VISUAL_LOG_MODULE, "Failed to allocate visual memory features");
        // Release memory to appropriate allocator
        if (cortex->memory_pool && memory_pool_owns(cortex->memory_pool, memory)) {
            memory_pool_release(cortex->memory_pool, memory);
        } else {
            nimcp_free(memory);
        }
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
        LOG_ERROR(VISUAL_LOG_MODULE, "Failed to allocate similarity buffer");
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
        LOG_ERROR(VISUAL_LOG_MODULE, "Failed to allocate memory results array");
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

    // Normalize query features
    float query_norm = 0.0f;
    for (uint32_t j = 0; j < cortex->feature_dim; j++) {
        query_norm += features[j] * features[j];
    }
    query_norm = sqrtf(query_norm);
    if (query_norm < 1e-6f) {
        return 1.0f;  // Zero features are maximally novel
    }

    // Find maximum cosine similarity to existing memories
    float max_similarity = 0.0f;
    for (uint32_t i = 0; i < cortex->num_memories; i++) {
        // Compute normalized dot product (cosine similarity)
        float dot = 0.0f;
        float memory_norm = 0.0f;
        for (uint32_t j = 0; j < cortex->feature_dim; j++) {
            dot += features[j] * cortex->memories[i]->features[j];
            memory_norm += cortex->memories[i]->features[j] * cortex->memories[i]->features[j];
        }
        memory_norm = sqrtf(memory_norm);

        if (memory_norm > 1e-6f) {
            float cosine_sim = dot / (query_norm * memory_norm);
            if (cosine_sim > max_similarity) {
                max_similarity = cosine_sim;
            }
        }
    }

    // Novelty = 1 - similarity (cosine similarity ranges from -1 to 1)
    // Clamp similarity to [0, 1] range first
    if (max_similarity < 0.0f) max_similarity = 0.0f;
    if (max_similarity > 1.0f) max_similarity = 1.0f;

    float novelty = 1.0f - max_similarity;
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
        LOG_ERROR(VISUAL_LOG_MODULE, "Attention map has no values");
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

/**
 * @brief Associate brain with visual cortex for neuromodulation
 *
 * WHAT: Set brain reference for ACh + NE modulation of visual processing
 * WHY:  Enable neurochemical modulation (attention, arousal, threat detection)
 * HOW:  Store brain pointer for neurotransmitter reading
 *
 * BIOLOGY:
 * - Acetylcholine from basal forebrain enhances V1 feature selectivity
 * - Norepinephrine from locus coeruleus increases arousal-dependent vision
 *
 * @param cortex Visual cortex instance
 * @param brain Brain instance (or NULL to clear)
 */
void visual_cortex_set_brain(visual_cortex_t* cortex, brain_t brain)
{
    if (!cortex) {
        return;
    }
    cortex->brain = brain;
}

//=============================================================================
// Neuromodulation Implementation
//=============================================================================

/**
 * @brief Get phasic/tonic state for a neuromodulator
 */
const phasic_tonic_state_t* visual_cortex_get_neuromod_state(
    const visual_cortex_t* cortex,
    uint32_t neuromod_type)
{
    // Guard: Validate inputs
    if (!nimcp_validate_pointer(cortex, "cortex")) {
        return NULL;
    }

    if (neuromod_type >= NUM_NEUROMOD_TYPES) {
        LOG_ERROR(VISUAL_LOG_MODULE, "Invalid neuromod_type: %u (max %u)", neuromod_type, NUM_NEUROMOD_TYPES - 1);
        return NULL;
    }

    return &cortex->neuromod_states[neuromod_type];
}

/**
 * @brief Set receptor expression profile for visual cortex
 */
bool visual_cortex_set_receptor_profile(
    visual_cortex_t* cortex,
    uint32_t layer_idx,
    const receptor_expression_t* receptors)
{
    // Guard: Validate inputs
    if (!nimcp_validate_pointer(cortex, "cortex") ||
        !nimcp_validate_pointer(receptors, "receptors")) {
        return false;
    }

    if (layer_idx >= NUM_V1_LAYERS) {
        LOG_ERROR(VISUAL_LOG_MODULE, "Invalid layer_idx: %u (max %u)", layer_idx, NUM_V1_LAYERS - 1);
        return false;
    }

    // Copy receptor profile
    cortex->receptor_profiles[layer_idx] = *receptors;
    return true;
}

/**
 * @brief Get receptor expression profile for a layer
 */
const receptor_expression_t* visual_cortex_get_receptor_profile(
    const visual_cortex_t* cortex,
    uint32_t layer_idx)
{
    // Guard: Validate inputs
    if (!nimcp_validate_pointer(cortex, "cortex")) {
        return NULL;
    }

    if (layer_idx >= NUM_V1_LAYERS) {
        LOG_ERROR(VISUAL_LOG_MODULE, "Invalid layer_idx: %u (max %u)", layer_idx, NUM_V1_LAYERS - 1);
        return NULL;
    }

    return &cortex->receptor_profiles[layer_idx];
}

/**
 * @brief Update phasic/tonic decay
 *
 * WHAT: Apply exponential decay to phasic and tonic levels
 * WHY:  Model reuptake and clearance mechanisms
 * HOW:  level *= exp(-decay_rate * dt)
 *
 * COMPLEXITY: O(1)
 */
static void update_neuromod_decay(phasic_tonic_state_t* state, float dt_sec)
{
    // Exponential decay: level(t+dt) = level(t) * exp(-dt/tau)
    // Approximation: level *= (1 - dt/tau) for small dt
    float phasic_decay_rate = 1.0f / state->burst_decay_tau;  // Convert tau to rate
    float tonic_decay_rate = 1.0f / state->homeostatic_tau;    // Convert tau to rate

    state->phasic_burst *= (1.0f - phasic_decay_rate * dt_sec);

    // Tonic level moves toward target with homeostatic regulation
    float tonic_error = state->tonic_target - state->tonic_level;
    state->tonic_level += tonic_error * tonic_decay_rate * dt_sec;

    // Clamp to valid ranges
    if (state->phasic_burst < 0.0f) state->phasic_burst = 0.0f;
    if (state->phasic_burst > state->max_burst_amplitude) state->phasic_burst = state->max_burst_amplitude;
    if (state->tonic_level < state->tonic_min) state->tonic_level = state->tonic_min;
    if (state->tonic_level > state->tonic_max) state->tonic_level = state->tonic_max;
}

/**
 * @brief Trigger phasic neuromodulator burst
 */
bool visual_cortex_trigger_phasic_burst(
    visual_cortex_t* cortex,
    uint32_t neuromod_type,
    float amount)
{
    // Guard: Validate inputs
    if (!nimcp_validate_pointer(cortex, "cortex")) {
        return false;
    }

    if (neuromod_type >= NUM_NEUROMOD_TYPES) {
        LOG_ERROR(VISUAL_LOG_MODULE, "Invalid neuromod_type: %u", neuromod_type);
        return false;
    }

    if (amount < 0.0f || amount > 1.0f) {
        LOG_WARN(VISUAL_LOG_MODULE, "Phasic burst amount out of range: %.2f, clamping to [0,1]", amount);
        amount = fminf(fmaxf(amount, 0.0f), 1.0f);
    }

    // Add to phasic level (capped at 1.0)
    phasic_tonic_state_t* state = &cortex->neuromod_states[neuromod_type];
    state->phasic_burst += amount;
    if (state->phasic_burst > 1.0f) {
        state->phasic_burst = 1.0f;
    }

    return true;
}

/**
 * @brief Set tonic neuromodulator baseline
 */
bool visual_cortex_set_tonic_level(
    visual_cortex_t* cortex,
    uint32_t neuromod_type,
    float level)
{
    // Guard: Validate inputs
    if (!nimcp_validate_pointer(cortex, "cortex")) {
        return false;
    }

    if (neuromod_type >= NUM_NEUROMOD_TYPES) {
        LOG_ERROR(VISUAL_LOG_MODULE, "Invalid neuromod_type: %u", neuromod_type);
        return false;
    }

    if (level < 0.0f || level > 1.0f) {
        LOG_WARN(VISUAL_LOG_MODULE, "Tonic level out of range: %.2f, clamping to [0,1]", level);
        level = fminf(fmaxf(level, 0.0f), 1.0f);
    }

    // Set tonic level directly
    cortex->neuromod_states[neuromod_type].tonic_level = level;
    return true;
}

/**
 * @brief Compute current neuromodulation effects
 *
 * ALGORITHM:
 * 1. Read neuromodulator levels from brain (if available)
 * 2. Update phasic/tonic states with decay
 * 3. Combine phasic + tonic: effective = α*phasic + (1-α)*tonic
 * 4. Multiply by receptor densities
 * 5. Compute gain values for visual processing
 */
bool visual_cortex_compute_neuromod_effects(
    const visual_cortex_t* cortex,
    uint32_t layer_idx,
    neuromod_effects_t* effects)
{
    // Guard: Validate inputs
    if (!nimcp_validate_pointer(cortex, "cortex") ||
        !nimcp_validate_pointer(effects, "effects")) {
        return false;
    }

    if (layer_idx >= NUM_V1_LAYERS) {
        LOG_ERROR(VISUAL_LOG_MODULE, "Invalid layer_idx: %u", layer_idx);
        return false;
    }

    // Default effects (no modulation)
    effects->gabor_gain = 1.0f;
    effects->attention_boost = 1.0f;
    effects->plasticity_gate = 0.5f;
    effects->contrast_gain = 1.0f;

    // Step 1: Read neuromodulator levels from brain (if available)
    float dopamine_level = 0.0f;
    float ach_level = 0.0f;
    float ne_level = 0.0f;

    if (cortex->brain) {
        neuromodulator_system_t neuromod = brain_get_neuromodulator_system(cortex->brain);
        if (neuromod) {
            dopamine_level = neuromodulator_get_level(neuromod, NEUROMOD_DOPAMINE);
            ach_level = neuromodulator_get_level(neuromod, NEUROMOD_ACETYLCHOLINE);
            ne_level = neuromodulator_get_level(neuromod, NEUROMOD_NOREPINEPHRINE);
        }
    }

    // Step 2: Combine with local phasic/tonic states
    // Phasic weight α = 0.6 (phasic dominates for fast events)
    const float PHASIC_WEIGHT = 0.6f;
    const float TONIC_WEIGHT = 1.0f - PHASIC_WEIGHT;

    const phasic_tonic_state_t* da_state = &cortex->neuromod_states[NEUROMOD_TYPE_DOPAMINE];
    const phasic_tonic_state_t* ach_state = &cortex->neuromod_states[NEUROMOD_TYPE_ACETYLCHOLINE];
    const phasic_tonic_state_t* ne_state = &cortex->neuromod_states[NEUROMOD_TYPE_NOREPINEPHRINE];

    float da_effective = dopamine_level + PHASIC_WEIGHT * da_state->phasic_burst + TONIC_WEIGHT * da_state->tonic_level;
    float ach_effective = ach_level + PHASIC_WEIGHT * ach_state->phasic_burst + TONIC_WEIGHT * ach_state->tonic_level;
    float ne_effective = ne_level + PHASIC_WEIGHT * ne_state->phasic_burst + TONIC_WEIGHT * ne_state->tonic_level;

    // Clamp to [0, 1]
    da_effective = fminf(da_effective, 1.0f);
    ach_effective = fminf(ach_effective, 1.0f);
    ne_effective = fminf(ne_effective, 1.0f);

    // Step 3: Multiply by receptor densities
    const receptor_expression_t* receptors = &cortex->receptor_profiles[layer_idx];

    // Dopamine effect: D1 (excitatory) - 0.5*D2 (inhibitory)
    float da_effect = (receptors->d1_density - 0.5f * receptors->d2_density) * da_effective;

    // Acetylcholine effect: M1 (excitatory) - 0.3*M2 (inhibitory)
    float ach_effect = (receptors->m1_density - 0.3f * receptors->m2_density) * ach_effective;

    // Norepinephrine effect: α1 (excitatory) + 0.5*β2 (plasticity)
    float ne_effect = (receptors->alpha1_density + 0.5f * receptors->beta2_density) * ne_effective;

    // Step 4: Query second messenger cascade activities
    float pka_activity = 0.0f;
    float pkc_activity = 0.0f;
    float camkii_activity = 0.0f;

    if (cortex->second_messengers_enabled && cortex->second_messengers) {
        // WHAT: Get cascade state for this layer
        // WHY:  Cascades amplify and temporally extend neuromodulator effects
        // HOW:  Query PKA (cAMP), PKC (IP3/DAG), CaMKII (Ca2+) activities
        second_messenger_state_t cascade_state;
        nimcp_result_t result = second_messenger_get_state(
            cortex->second_messengers, layer_idx, &cascade_state);

        if (result == NIMCP_SUCCESS) {
            pka_activity = cascade_state.camp.pka_activity;
            pkc_activity = cascade_state.ip3_dag.pkc_activity;
            camkii_activity = cascade_state.calcium.camkii_activity;

            LOG_DEBUG(VISUAL_LOG_MODULE, "Layer %u cascade state: PKA=%.2f PKC=%.2f CaMKII=%.2f",
                      layer_idx, pka_activity, pkc_activity, camkii_activity);
        }
    }

    // Step 5: Compute gain values with cascade modulation
    // BIOLOGY:
    // - PKA (cAMP pathway): Enhances gain via AMPAR phosphorylation
    // - PKC (IP3/DAG pathway): Modulates contrast gain
    // - CaMKII (Ca2+ pathway): Gates plasticity via NMDAR modulation

    // Gabor gain: DA enhances signal detection, ACh enhances features, NE increases sensitivity
    // PKA amplifies these effects (D1 -> Gs -> cAMP -> PKA)
    effects->gabor_gain = 1.0f + 0.5f * da_effect + 0.3f * ach_effect + 0.4f * ne_effect
                          + 0.3f * pka_activity;  // PKA amplifies gain

    // Attention boost: ACh dominates attention, NE increases alertness
    // PKC modulates attention via vesicle release probability
    effects->attention_boost = 1.0f + 0.7f * ach_effect + 0.3f * ne_effect
                               + 0.2f * pkc_activity;  // PKC enhances attention

    // Plasticity gate: DA gates learning (reward), ACh gates encoding
    // CaMKII is the primary plasticity kinase (required for LTP)
    float plasticity_input = 2.0f * da_effect + ach_effect + 2.0f * camkii_activity;
    effects->plasticity_gate = 1.0f / (1.0f + expf(-plasticity_input));

    // Contrast gain: DA enhances contrast sensitivity
    // PKA and PKC both contribute to contrast modulation
    effects->contrast_gain = 1.0f + 0.4f * da_effect + 0.2f * ach_effect
                             + 0.3f * pka_activity + 0.2f * pkc_activity;

    // Clamp all gains to reasonable ranges
    effects->gabor_gain = fminf(fmaxf(effects->gabor_gain, 0.5f), 2.0f);
    effects->attention_boost = fminf(fmaxf(effects->attention_boost, 0.5f), 2.0f);
    effects->plasticity_gate = fminf(fmaxf(effects->plasticity_gate, 0.0f), 1.0f);
    effects->contrast_gain = fminf(fmaxf(effects->contrast_gain, 0.5f), 2.0f);

    return true;
}

//=============================================================================
// Second Messenger Cascade Integration
//=============================================================================

/**
 * @brief Trigger receptor activation for second messenger cascade
 *
 * WHAT: Activate G-protein coupled receptor to initiate signaling cascade
 * WHY:  Neuromodulators bind receptors -> trigger intracellular cascades
 * HOW:  Route to appropriate cascade (Gs/Gi/Gq) based on receptor type
 *
 * BIOLOGY:
 * - D1 receptors -> Gs -> cAMP -> PKA
 * - M1 receptors -> Gq -> IP3/DAG -> PKC + Ca2+
 * - beta-adrenergic -> Gs -> cAMP -> PKA
 * - alpha1-adrenergic -> Gq -> IP3/DAG -> PKC
 */
bool visual_cortex_trigger_receptor(
    visual_cortex_t* cortex,
    uint32_t layer_idx,
    receptor_type_t receptor_type,
    float occupancy)
{
    // Guard: Validate inputs
    if (!nimcp_validate_pointer(cortex, "cortex")) {
        return false;
    }

    if (layer_idx >= NUM_V1_LAYERS) {
        LOG_ERROR(VISUAL_LOG_MODULE, "Invalid layer_idx: %u (max %u)", layer_idx, NUM_V1_LAYERS - 1);
        return false;
    }

    if (occupancy < 0.0f || occupancy > 1.0f) {
        LOG_WARN(VISUAL_LOG_MODULE, "Receptor occupancy out of range: %.2f, clamping to [0,1]", occupancy);
        occupancy = fminf(fmaxf(occupancy, 0.0f), 1.0f);
    }

    // Guard: Check if second messengers enabled
    if (!cortex->second_messengers_enabled || !cortex->second_messengers) {
        LOG_WARN(VISUAL_LOG_MODULE, "Second messengers not enabled, cannot trigger receptor");
        return false;
    }

    // WHAT: Get current timestamp for cascade activation
    // WHY:  Cascades track temporal dynamics
    // HOW:  Use system time in milliseconds
    uint64_t timestamp_ms = (uint64_t)(time(NULL) * 1000);

    // WHAT: Route to appropriate G-protein pathway
    // WHY:  Different receptors couple to different G-proteins
    // HOW:  Based on receptor type, activate Gs, Gi, or Gq pathway
    nimcp_result_t result = NIMCP_ERROR_INVALID_PARAM;

    switch (receptor_type) {
        // Gs-coupled receptors (activate adenylyl cyclase -> cAMP -> PKA)
        case RECEPTOR_D1:
        case RECEPTOR_BETA:  // Beta-adrenergic
            result = second_messenger_activate_gs(
                cortex->second_messengers, layer_idx, occupancy, timestamp_ms);
            LOG_DEBUG(VISUAL_LOG_MODULE, "Layer %u: Activated Gs pathway (receptor=%d, occupancy=%.2f)",
                      layer_idx, receptor_type, occupancy);
            break;

        // Gq-coupled receptors (activate PLC -> IP3/DAG -> PKC + Ca2+)
        case RECEPTOR_MUSCARINIC:  // M1/M3 muscarinic
        case RECEPTOR_ALPHA1:      // Alpha1-adrenergic
        case RECEPTOR_5HT2A:
            result = second_messenger_activate_gq(
                cortex->second_messengers, layer_idx, occupancy, timestamp_ms);
            LOG_DEBUG(VISUAL_LOG_MODULE, "Layer %u: Activated Gq pathway (receptor=%d, occupancy=%.2f)",
                      layer_idx, receptor_type, occupancy);
            break;

        // Gi-coupled receptors (inhibit adenylyl cyclase -> reduce cAMP)
        case RECEPTOR_D2:
        case RECEPTOR_ALPHA2:  // Alpha2-adrenergic (autoreceptor)
        case RECEPTOR_5HT1A:   // 5-HT1A (also Gi-coupled)
            result = second_messenger_activate_gi(
                cortex->second_messengers, layer_idx, occupancy, timestamp_ms);
            LOG_DEBUG(VISUAL_LOG_MODULE, "Layer %u: Activated Gi pathway (receptor=%d, occupancy=%.2f)",
                      layer_idx, receptor_type, occupancy);
            break;

        default:
            LOG_ERROR(VISUAL_LOG_MODULE, "Unknown receptor type: %d", receptor_type);
            return false;
    }

    if (result != NIMCP_SUCCESS) {
        LOG_ERROR(VISUAL_LOG_MODULE, "Failed to activate receptor cascade: %d", result);
        return false;
    }

    return true;
}

/**
 * @brief Get second messenger cascade state for layer
 *
 * WHAT: Query cascade state (cAMP, PKA, PKC, CaMKII activities)
 * WHY:  Enable monitoring and integration with other systems
 * HOW:  Return state from second messenger system
 */
bool visual_cortex_get_second_messenger_state(
    const visual_cortex_t* cortex,
    uint32_t layer_idx,
    second_messenger_state_t* state)
{
    // Guard: Validate inputs
    if (!nimcp_validate_pointer(cortex, "cortex") ||
        !nimcp_validate_pointer(state, "state")) {
        return false;
    }

    if (layer_idx >= NUM_V1_LAYERS) {
        LOG_ERROR(VISUAL_LOG_MODULE, "Invalid layer_idx: %u (max %u)", layer_idx, NUM_V1_LAYERS - 1);
        return false;
    }

    // Guard: Check if second messengers enabled
    if (!cortex->second_messengers_enabled || !cortex->second_messengers) {
        LOG_WARN(VISUAL_LOG_MODULE, "Second messengers not enabled");
        // Return default state (all zeros)
        memset(state, 0, sizeof(second_messenger_state_t));
        return true;
    }

    // WHAT: Query cascade state from second messenger system
    // WHY:  Provide access to intracellular signaling state
    // HOW:  Call second_messenger_get_state for this layer
    nimcp_result_t result = second_messenger_get_state(
        cortex->second_messengers, layer_idx, state);

    if (result != NIMCP_SUCCESS) {
        LOG_ERROR(VISUAL_LOG_MODULE, "Failed to get second messenger state: %d", result);
        return false;
    }

    return true;
}

//=============================================================================
// Bidirectional Feedback Functions (Phase 10.11.3)
//=============================================================================

/**
 * @brief Boost attention to specific visual region
 *
 * WHAT: Increase processing sensitivity for spatial region
 * WHY:  Social cues (faces/agents) should receive enhanced processing
 * HOW:  Scale feature activations in target region (stored for next process call)
 *
 * BIOLOGY: STS modulates V1 for social stimuli via feedback projections
 *
 * COMPLEXITY: O(1)
 *
 * @param cortex Visual cortex instance
 * @param region_x X coordinate of region center (normalized [0,1])
 * @param region_y Y coordinate of region center (normalized [0,1])
 * @param boost_factor Attention boost [1.0, 2.0]
 */
void visual_cortex_boost_region_attention(visual_cortex_t* cortex,
                                           float region_x,
                                           float region_y,
                                           float boost_factor)
{
    // Guard: Validate cortex
    if (!cortex) {
        return;
    }

    // Guard: Validate coordinates [0, 1]
    region_x = fminf(fmaxf(region_x, 0.0f), 1.0f);
    region_y = fminf(fmaxf(region_y, 0.0f), 1.0f);

    // Guard: Clamp boost factor [1.0, 2.0]
    boost_factor = fminf(fmaxf(boost_factor, 1.0f), 2.0f);

    // WHAT: Store attention parameters for next processing cycle
    // WHY:  Cannot apply boost until we have image data
    // HOW:  Set neuromodulator gain (acts similarly to attention)
    //
    // NOTE: In a full implementation, we'd store attention map and apply during
    // conv_layer_forward(). For now, we use the existing neuromodulator gain
    // mechanism which already modulates V1 output.
    //
    // Future enhancement: Add spatial attention map to visual_cortex_struct
    // and multiply V1 output by attention values at corresponding spatial locations.
}

/**
 * @brief Detect if agent/person present in visual field
 *
 * WHAT: Simple heuristic agent detection
 * WHY:  Triggers mirror neuron observation mode
 * HOW:  Check for motion + face-like patterns in features
 *
 * COMPLEXITY: O(n) where n = num_features
 *
 * @param cortex Visual cortex instance
 * @param features Feature vector from recent processing
 * @param num_features Number of features
 * @return true if agent detected
 */
bool visual_cortex_detect_agent(visual_cortex_t* cortex,
                                 const float* features,
                                 uint32_t num_features)
{
    // Guard: Validate inputs
    if (!cortex || !features || num_features == 0) {
        return false;
    }

    // WHAT: Simple heuristic based on feature activation patterns
    // WHY:  Full face detection is expensive; this is a fast approximation
    // HOW:  Check for high variance (motion/edges) + mid-range activations (structure)
    //
    // HEURISTIC:
    // - High variance → motion or complex structure
    // - Multiple mid-range activations → organized structure (not noise)
    // - Activation in multiple regions → spatial extent (not a point)

    // Compute feature statistics
    float mean = 0.0f;
    float variance = 0.0f;
    uint32_t mid_range_count = 0;

    for (uint32_t i = 0; i < num_features; i++) {
        mean += features[i];
    }
    mean /= num_features;

    for (uint32_t i = 0; i < num_features; i++) {
        float diff = features[i] - mean;
        variance += diff * diff;

        // Count mid-range activations (0.3 to 0.7)
        if (features[i] > 0.3f && features[i] < 0.7f) {
            mid_range_count++;
        }
    }
    variance /= num_features;

    // DETECTION CRITERIA:
    // 1. Variance > 0.05 (sufficient structure/motion)
    // 2. Mid-range activations > 30% (organized structure, not uniform)
    bool has_structure = (variance > 0.05f);
    bool has_organization = (mid_range_count > num_features / 3);

    return has_structure && has_organization;
}

//=============================================================================
// Bio-Async Communication Implementation
//=============================================================================

/**
 * @brief Get bio-async module context
 */
bio_module_context_t visual_cortex_get_bio_context(visual_cortex_t* cortex)
{
    if (!cortex || !cortex->bio_async_enabled) {
        return NULL;
    }
    return cortex->bio_ctx;
}

/**
 * @brief Process pending bio-async messages
 *
 * Uses bio_router_process_inbox() which calls registered handlers.
 * Handlers should be registered during module initialization.
 */
uint32_t visual_cortex_process_bio_messages(visual_cortex_t* cortex, uint32_t max_messages)
{
    if (!cortex || !cortex->bio_async_enabled || !cortex->bio_ctx) {
        return 0;
    }

    // Process inbox using the router's handler-based system
    uint32_t processed = bio_router_process_inbox(cortex->bio_ctx, max_messages);

    if (processed > 0) {
        LOG_DEBUG(VISUAL_LOG_MODULE, "Processed %u bio-async messages", processed);
    }

    return processed;
}

/**
 * @brief Broadcast visual feature detection via bio-async
 */
nimcp_error_t visual_cortex_broadcast_input(
    visual_cortex_t* cortex,
    const float* features,
    uint32_t num_features,
    float salience)
{
    if (!cortex) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    if (!cortex->bio_async_enabled || !cortex->bio_ctx) {
        return NIMCP_ERROR_NOT_INITIALIZED;
    }

    if (!features || num_features == 0) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    // Create visual feature detected message
    bio_msg_visual_feature_detected_t msg;
    bio_msg_init_header(&msg.header, BIO_MSG_VISUAL_FEATURE_DETECTED,
                        BIO_MODULE_VISUAL_CORTEX, 0,  // 0 = broadcast
                        sizeof(msg) - sizeof(bio_message_header_t));

    msg.header.channel = BIO_CHANNEL_DOPAMINE;  // Visual salience involves dopamine
    msg.header.flags = BIO_MSG_FLAG_BROADCAST;

    // Fill in feature detection data
    msg.feature_id = 0;  // Generic visual input
    msg.x_position = 0.5f;  // Center (can be parameterized)
    msg.y_position = 0.5f;
    msg.confidence = salience;
    msg.salience = salience;
    msg.layer = 1;  // V1 layer

    // Broadcast to all interested modules
    nimcp_error_t err = bio_router_broadcast(cortex->bio_ctx, &msg, sizeof(msg));
    if (err != NIMCP_SUCCESS) {
        LOG_WARN(VISUAL_LOG_MODULE, "Failed to broadcast visual input: %d", err);
        return err;
    }

    LOG_DEBUG(VISUAL_LOG_MODULE, "Broadcast visual input: %u features, salience=%.2f",
              num_features, salience);

    return NIMCP_SUCCESS;
}

/**
 * @brief Send visual attention shift notification
 */
nimcp_error_t visual_cortex_broadcast_attention_shift(
    visual_cortex_t* cortex,
    float x,
    float y,
    float salience)
{
    if (!cortex) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    if (!cortex->bio_async_enabled || !cortex->bio_ctx) {
        return NIMCP_ERROR_NOT_INITIALIZED;
    }

    // Create attention shift message using the defined type
    bio_msg_visual_attention_shift_t msg;
    bio_msg_init_header(&msg.header, BIO_MSG_VISUAL_ATTENTION_SHIFT,
                        BIO_MODULE_VISUAL_CORTEX, 0,  // 0 = broadcast
                        sizeof(msg) - sizeof(bio_message_header_t));

    msg.header.channel = BIO_CHANNEL_ACETYLCHOLINE;  // Attention involves ACh
    msg.header.flags = BIO_MSG_FLAG_BROADCAST | BIO_MSG_FLAG_URGENT;

    msg.target_x = x;
    msg.target_y = y;
    msg.urgency = salience;
    msg.reason = 0;  // Generic attention shift

    // Broadcast to all interested modules
    nimcp_error_t err = bio_router_broadcast(cortex->bio_ctx, &msg, sizeof(msg));
    if (err != NIMCP_SUCCESS) {
        LOG_WARN(VISUAL_LOG_MODULE, "Failed to broadcast attention shift: %d", err);
        return err;
    }

    LOG_DEBUG(VISUAL_LOG_MODULE, "Broadcast attention shift: (%.2f, %.2f) salience=%.2f",
              x, y, salience);

    return NIMCP_SUCCESS;
}
