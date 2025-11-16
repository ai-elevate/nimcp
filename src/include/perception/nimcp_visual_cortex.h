/**
 * @file visual_cortex.h
 * @brief Visual Cortex - Specialized brain region for visual processing
 *
 * WHAT: V1-style visual cortex as intrinsic NIMCP brain region
 * WHY:  Provide specialized visual perception capabilities to NIMCP brains
 * HOW:  CNN architecture models V1's orientation-selective columnar structure
 *
 * ARCHITECTURE PHILOSOPHY:
 * The visual cortex is a SPECIALIZED BRAIN REGION (like biological V1), not a
 * general neural network. Just as V1 has specialized columnar architecture for
 * edge detection, this module uses CNN structure as its intrinsic mechanism.
 *
 * BRAIN INTEGRATION:
 * Visual Cortex (V1) → Hippocampus (visual memory consolidation)
 *                    → Prefrontal Cortex (visual reasoning)
 *                    → Curiosity System (novelty detection)
 *                    → Attention System (salience processing)
 *                    → LLM Interface (vision-language grounding)
 *
 * NEUROSCIENCE BACKGROUND:
 * Biological V1 (primary visual cortex):
 * - Specialized columnar architecture (orientation columns, ocular dominance)
 * - Simple cells: Edge detection (modeled by Gabor filters)
 * - Complex cells: Position-invariant features (modeled by pooling)
 * - Hierarchical: V1 → V2 → V4 → IT
 *
 * THIS IMPLEMENTATION (V1 PRIMARY VISUAL CORTEX):
 * - Gabor filters: Model orientation-selective simple cells
 * - Convolution + pooling: Hierarchical feature extraction
 * - Attention integration: Salience-driven visual focus
 * - Memory integration: Persistent visual experiences
 * - Curiosity integration: Novelty-driven visual exploration
 *
 * WHY CNN NOT NEURAL NETWORK:
 * V1's structure is genetically determined (innate columnar organization).
 * It's not a general-purpose network - it's a specialized visual processor.
 * This CNN architecture accurately models that biological specialization.
 *
 * FUTURE EXTENSIONS (V2/V4/IT):
 * Higher visual areas (V2, V4, IT) could use NIMCP neural networks to learn
 * more complex features, creating a hybrid: innate V1 + learned higher areas.
 *
 * USE CASES:
 * - Robot vision: Camera → Visual Cortex → Navigation/Manipulation
 * - LLM grounding: Image → Visual Cortex → Language understanding
 * - Autonomous agents: Visual Cortex + Curiosity → Active exploration
 * - Multi-modal learning: Vision + Language + Memory integration
 *
 * @author NIMCP Development Team
 * @date 2025
 * @version 2.6 (Visual Cortex as Brain Region)
 */

#ifndef NIMCP_VISUAL_CORTEX_H
#define NIMCP_VISUAL_CORTEX_H

#include <stdbool.h>
#include <stdint.h>
#include "plasticity/neuromodulators/nimcp_phasic_tonic.h"

#ifdef __cplusplus
extern "C" {
#endif

// Forward declaration for brain integration
typedef struct brain_struct* brain_t;

//=============================================================================
// Activation Functions
//=============================================================================

/**
 * @brief Activation function types
 */
typedef enum {
    VISUAL_ACTIVATION_NONE,      /**< Linear (no activation) */
    VISUAL_ACTIVATION_RELU,      /**< Rectified Linear Unit */
    VISUAL_ACTIVATION_SIGMOID,   /**< Sigmoid (0-1) */
    VISUAL_ACTIVATION_TANH       /**< Hyperbolic tangent (-1 to 1) */
} visual_activation_type_t;

//=============================================================================
// Convolution Layer
//=============================================================================

/**
 * @brief Opaque convolution layer structure
 */
typedef struct conv_layer_struct conv_layer_t;

/**
 * @brief Convolution layer configuration
 */
typedef struct {
    uint32_t input_width;       /**< Input image width */
    uint32_t input_height;      /**< Input image height */
    uint32_t input_channels;    /**< Number of input channels (1=grayscale, 3=RGB) */
    uint32_t num_filters;       /**< Number of convolution filters */
    uint32_t kernel_size;       /**< Kernel size (typically 3, 5, or 7) */
    uint32_t stride;            /**< Stride (typically 1 or 2) */
    uint32_t padding;           /**< Padding (0 for valid, kernel_size/2 for same) */
    visual_activation_type_t activation; /**< Activation function */
} conv_layer_config_t;

/**
 * WHAT: Create convolution layer
 * WHY:  Initialize learnable filters for feature extraction
 * HOW:  Allocate memory for kernels and output buffers
 *
 * @param config Layer configuration
 * @return Layer handle or NULL on failure
 *
 * COMPLEXITY: O(K²·F·C) where K=kernel_size, F=num_filters, C=input_channels
 * MEMORY: O(K²·F·C + W·H·F) for kernels + output
 */
conv_layer_t* conv_layer_create(const conv_layer_config_t* config);

/**
 * WHAT: Destroy convolution layer
 * WHY:  Free allocated resources
 * HOW:  Free kernels and buffers
 *
 * @param layer Layer to destroy
 */
void conv_layer_destroy(conv_layer_t* layer);

/**
 * WHAT: Set filter kernel weights
 * WHY:  Configure learned or predefined filters
 * HOW:  Copy weights into layer's kernel buffer
 *
 * @param layer Convolution layer
 * @param filter_idx Filter index (0 to num_filters-1)
 * @param kernel Kernel weights [kernel_size × kernel_size × input_channels]
 * @return true on success, false on failure
 */
bool conv_layer_set_kernel(conv_layer_t* layer, uint32_t filter_idx, const float* kernel);

/**
 * WHAT: Forward pass through convolution layer
 * WHY:  Extract features from input
 * HOW:  Convolve each filter with input, apply activation
 *
 * @param layer Convolution layer
 * @param input Input tensor [H × W × C]
 * @param output Output tensor [H' × W' × num_filters]
 * @return true on success, false on failure
 *
 * COMPLEXITY: O(H·W·K²·C·F) where K=kernel, C=channels, F=filters
 *
 * OUTPUT DIMENSIONS:
 * - H' = (H + 2·padding - kernel_size) / stride + 1
 * - W' = (W + 2·padding - kernel_size) / stride + 1
 */
bool conv_layer_forward(conv_layer_t* layer, const float* input, float* output);

/**
 * WHAT: Get output dimensions
 * WHY:  Calculate output buffer size
 * HOW:  Apply conv formula
 */
uint32_t conv_layer_get_output_width(const conv_layer_t* layer);
uint32_t conv_layer_get_output_height(const conv_layer_t* layer);
uint32_t conv_layer_get_output_channels(const conv_layer_t* layer);

//=============================================================================
// Pooling Layer
//=============================================================================

/**
 * @brief Pooling types
 */
typedef enum {
    POOL_MAX,    /**< Max pooling */
    POOL_AVG     /**< Average pooling */
} pool_type_t;

/**
 * @brief Opaque pooling layer structure
 */
typedef struct pool_layer_struct pool_layer_t;

/**
 * @brief Pooling layer configuration
 */
typedef struct {
    uint32_t input_width;
    uint32_t input_height;
    uint32_t input_channels;
    uint32_t pool_size;    /**< Pool window size (typically 2) */
    uint32_t stride;       /**< Stride (typically 2 for non-overlapping) */
    pool_type_t type;      /**< Max or average pooling */
} pool_layer_config_t;

/**
 * WHAT: Create pooling layer
 * WHY:  Reduce spatial dimensions, provide translation invariance
 * HOW:  Allocate buffers for pooling operation
 */
pool_layer_t* pool_layer_create(const pool_layer_config_t* config);

/**
 * WHAT: Destroy pooling layer
 */
void pool_layer_destroy(pool_layer_t* layer);

/**
 * WHAT: Forward pass through pooling layer
 * WHY:  Downsample feature maps
 * HOW:  Apply max or average over pool windows
 *
 * COMPLEXITY: O(H·W·C·P²) where P=pool_size
 */
bool pool_layer_forward(pool_layer_t* layer, const float* input, float* output);

//=============================================================================
// Gabor Filters (V1 Edge Detection)
//=============================================================================

/**
 * @brief Gabor filter parameters
 *
 * WHAT: Parameters for biologically-plausible edge detection
 * WHY:  V1 simple cells respond to oriented edges like Gabor filters
 * HOW:  Sine wave modulated by Gaussian envelope
 */
typedef struct {
    float wavelength;     /**< Wavelength (λ) in pixels (3-7 typical) */
    float orientation;    /**< Orientation in degrees (0-180) */
    float phase;          /**< Phase offset (0 or 90 for even/odd) */
    float aspect_ratio;   /**< Spatial aspect ratio (γ, typically 0.5) */
    float bandwidth;      /**< Bandwidth (σ, typically 1.0) */
} gabor_params_t;

/**
 * WHAT: Create Gabor kernel
 * WHY:  Generate V1-like edge detector
 * HOW:  Apply Gabor function: g(x,y) = exp(-(x'² + γ²y'²)/(2σ²)) · cos(2π·x'/λ + φ)
 *
 * @param kernel_size Kernel dimension (typically 7 or 9, must be odd)
 * @param params Gabor parameters
 * @return Gabor kernel [kernel_size × kernel_size] or NULL on failure
 *
 * NEUROSCIENCE:
 * - V1 simple cells are tuned to specific orientations and frequencies
 * - Gabor filters model this selectivity
 * - Multiple orientations (0°, 45°, 90°, 135°) detect all edge types
 */
float* gabor_create_kernel(int kernel_size, const gabor_params_t* params);

//=============================================================================
// Attention Map
//=============================================================================

/**
 * @brief Opaque attention map structure
 */
typedef struct attention_map_struct attention_map_t;

/**
 * WHAT: Create attention map
 * WHY:  Track where the visual system is attending
 * HOW:  Allocate 2D salience map
 */
attention_map_t* attention_map_create(uint32_t width, uint32_t height);

/**
 * WHAT: Destroy attention map
 */
void attention_map_destroy(attention_map_t* map);

/**
 * WHAT: Get attention value at location
 * WHY:  Query salience at specific pixel
 * HOW:  Return normalized attention (0-1)
 *
 * @param map Attention map
 * @param x X coordinate
 * @param y Y coordinate
 * @return Attention value (0-1) or -1.0 on error
 */
float attention_map_get(const attention_map_t* map, uint32_t x, uint32_t y);

/**
 * WHAT: Set attention value at location
 */
bool attention_map_set(attention_map_t* map, uint32_t x, uint32_t y, float value);

//=============================================================================
// Neuromodulation Structures
//=============================================================================

/**
 * @brief Phasic vs Tonic neuromodulator state
 *
 * WHAT: Tracks both rapid bursts (phasic) and baseline (tonic) levels
 * WHY:  Biological neuromodulation has two timescales
 * HOW:  Phasic = rapid events, Tonic = slow baseline
 *
 * BIOLOGY:
 * - Phasic dopamine: Reward prediction errors (50-200ms bursts)
 * - Tonic dopamine: Background motivation level (seconds to minutes)
 * - Phasic ACh: Attention cue detection (100-300ms)
 * - Tonic ACh: Sustained attention level (seconds)
 * - Phasic NE: Arousal spike from threat (50-150ms)
 * - Tonic NE: Background arousal/stress (minutes)
 */
#ifndef PHASIC_TONIC_STATE_DEFINED
// phasic_tonic_state_t already defined in nimcp_phasic_tonic.h
#define PHASIC_TONIC_STATE_DEFINED
#endif

/**
 * @brief Receptor expression profile for visual cortex neurons
 *
 * WHAT: Density of different receptor subtypes in V1
 * WHY:  Different receptors mediate different effects
 * HOW:  Multiply global neuromodulator by local receptor density
 *
 * BIOLOGY:
 * - D1 receptors: Enhance signal, increase plasticity (excitatory)
 * - D2 receptors: Reduce noise, decrease plasticity (inhibitory)
 * - M1 receptors: Slow excitation, attention modulation
 * - M2 receptors: Presynaptic inhibition, gain control
 * - α1 receptors: Increase arousal, enhance processing
 * - β2 receptors: Facilitate plasticity, learning
 *
 * LAYER SPECIFICITY (V1 cortical layers):
 * - Layer 2/3: High D1, moderate ACh (top-down attention)
 * - Layer 4: High ACh, moderate NE (thalamic input)
 * - Layer 5: High D1, D2, NE (motor output, arousal)
 * - Layer 6: High M1, β2 (feedback modulation)
 */
#ifndef RECEPTOR_EXPRESSION_DEFINED
#define RECEPTOR_EXPRESSION_DEFINED
typedef struct {
    // Dopamine receptors
    float d1_density;    /**< D1 receptor density (0-1), default: 0.3 */
    float d2_density;    /**< D2 receptor density (0-1), default: 0.2 */

    // Acetylcholine receptors
    float m1_density;    /**< M1 muscarinic density (0-1), default: 0.5 */
    float m2_density;    /**< M2 muscarinic density (0-1), default: 0.3 */

    // Norepinephrine receptors
    float alpha1_density; /**< α1-adrenergic density (0-1), default: 0.4 */
    float beta2_density;  /**< β2-adrenergic density (0-1), default: 0.3 */
} receptor_expression_t;
#endif

/**
 * @brief Computed neuromodulation effects on visual processing
 *
 * WHAT: Final gain/modulation values applied to V1 processing
 * WHY:  Separate computation from application for clarity
 * HOW:  Computed from phasic/tonic states × receptor expression
 */
typedef struct {
    float gabor_gain;        /**< Gain for Gabor filter outputs (0.5-2.0) */
    float attention_boost;   /**< Boost for attention map (0.5-2.0) */
    float plasticity_gate;   /**< Gate for learning (0-1) */
    float contrast_gain;     /**< Contrast sensitivity modulation (0.5-2.0) */
} neuromod_effects_t;

//=============================================================================
// Visual Memory
//=============================================================================

/**
 * @brief Visual memory entry
 */
typedef struct {
    float* features;      /**< Feature vector */
    uint32_t feature_dim; /**< Feature dimensionality */
    float salience;       /**< Memory salience (0-1) */
    uint64_t timestamp;   /**< When stored */
} visual_memory_t;

//=============================================================================
// Visual Cortex
//=============================================================================

/**
 * @brief Opaque visual cortex structure
 */
typedef struct visual_cortex_struct visual_cortex_t;

/**
 * @brief Visual cortex configuration
 */
typedef struct {
    uint32_t input_width;       /**< Expected input image width */
    uint32_t input_height;      /**< Expected input image height */
    uint32_t num_v1_filters;    /**< Number of V1 filters (edge detectors) */
    uint32_t feature_dim;       /**< Output feature dimensionality */
    bool enable_attention;      /**< Enable visual attention system */
    bool enable_memory;         /**< Enable visual memory storage */

    // NIMCP 2.7 Phase 8.5: Fractal Topology Integration
    bool enable_fractal_topology;  /**< Enable scale-free topology within V1 */
    float hub_ratio;               /**< Fraction of hub neurons (0.1-0.2), default: 0.15 */
    float power_law_gamma;         /**< Power-law exponent (-2 to -3), default: -2.1 */
    uint32_t internal_neurons;     /**< Number of internal V1 neurons, default: num_v1_filters * 10 */
} visual_cortex_config_t;

/**
 * WHAT: Create V1 visual cortex brain region
 * WHY:  Initialize specialized visual processing for NIMCP brain
 * HOW:  CNN architecture models V1's orientation-selective columns
 *
 * @param config Visual cortex configuration
 * @return Visual cortex handle or NULL on failure
 *
 * V1 COLUMNAR ARCHITECTURE (Biologically accurate):
 * - Orientation columns: Gabor filters at 0°, 45°, 90°, 135°
 * - Simple cells: Detect edges via convolution
 * - Complex cells: Position invariance via pooling
 * - Feature extraction: Compact representation for other brain regions
 *
 * BRAIN INTEGRATION:
 * - Visual features → Hippocampus (memory consolidation)
 * - Visual features → Curiosity (novelty detection)
 * - Attention maps → Salience system (visual focus)
 * - Visual memories → Knowledge graph (visual understanding)
 *
 * ARCHITECTURE:
 * - V1 Layer: Gabor convolution (orientation-selective)
 * - Pooling: Max pooling (translation invariance)
 * - Features: Dimensionality reduction for brain integration
 * - Attention: Optional salience map generation
 * - Memory: Optional visual experience storage
 *
 * COMPLEXITY: O(W·H·F + F·D) for initialization
 * MEMORY: O(W·H·F + F·D + M) where M=memory_capacity
 */
visual_cortex_t* visual_cortex_create(const visual_cortex_config_t* config);

/**
 * WHAT: Destroy visual cortex
 * WHY:  Free all visual processing resources
 * HOW:  Recursively destroy layers and memories
 */
void visual_cortex_destroy(visual_cortex_t* cortex);

/**
 * WHAT: Process image through visual cortex
 * WHY:  Extract visual features for memory/LLM
 * HOW:  Forward through V1 → pooling → feature extraction
 *
 * @param cortex Visual cortex
 * @param image Input image [H × W × channels]
 * @param width Image width
 * @param height Image height
 * @param channels Number of channels (1=grayscale, 3=RGB)
 * @param features Output feature vector [feature_dim]
 * @return true on success, false on failure
 *
 * COMPLEXITY: O(H·W·K²·F) for convolution + O(F·D) for features
 *
 * USAGE:
 * ```c
 * uint8_t image[640 * 480];  // Grayscale camera image
 * float features[128];
 * visual_cortex_process(cortex, image, 640, 480, 1, features);
 * // Features now contain compact visual representation
 * ```
 */
bool visual_cortex_process(
    visual_cortex_t* cortex,
    const uint8_t* image,
    uint32_t width,
    uint32_t height,
    uint32_t channels,
    float* features
);

/**
 * WHAT: Compute visual attention map
 * WHY:  Identify salient regions in image
 * HOW:  Combine V1 responses weighted by novelty/contrast
 *
 * @param cortex Visual cortex
 * @param image Input image
 * @param width Image width
 * @param height Image height
 * @param attn_map Output attention map
 * @return true on success, false on failure
 *
 * NEUROSCIENCE:
 * - Attention is driven by bottom-up salience (contrast, edges)
 * - Top-down attention from goals/expectations
 * - Winner-take-all competition between locations
 *
 * USAGE:
 * ```c
 * attention_map_t* map = attention_map_create(640, 480);
 * visual_cortex_compute_attention(cortex, image, 640, 480, map);
 * // Find most salient location
 * float max_attn = 0;
 * int focus_x = 0, focus_y = 0;
 * for (y = 0; y < 480; y++) {
 *     for (x = 0; x < 640; x++) {
 *         float attn = attention_map_get(map, x, y);
 *         if (attn > max_attn) {
 *             max_attn = attn;
 *             focus_x = x;
 *             focus_y = y;
 *         }
 *     }
 * }
 * ```
 */
bool visual_cortex_compute_attention(
    visual_cortex_t* cortex,
    const uint8_t* image,
    uint32_t width,
    uint32_t height,
    attention_map_t* attn_map
);

/**
 * WHAT: Store visual features in memory
 * WHY:  Build persistent visual memory
 * HOW:  Add features to memory with salience weighting
 *
 * @param cortex Visual cortex (must have enable_memory=true)
 * @param features Feature vector to store
 * @param salience Memory salience (0-1, higher = more important)
 * @return true on success, false on failure
 *
 * INTEGRATION WITH NIMCP:
 * - Links to NIMCP memory consolidation system
 * - Salience drives consolidation probability
 * - Can trigger curiosity if novel visual pattern
 */
bool visual_cortex_store_memory(
    visual_cortex_t* cortex,
    const float* features,
    float salience
);

/**
 * WHAT: Recall similar visual memories
 * WHY:  "Have I seen this before?"
 * HOW:  Nearest-neighbor search in feature space
 *
 * @param cortex Visual cortex
 * @param query_features Query feature vector
 * @param max_results Maximum number of results to return
 * @param memories Output array of memories (caller must free)
 * @param num_memories Number of memories returned
 * @return true on success, false on failure
 *
 * USAGE:
 * ```c
 * float features[128];
 * visual_cortex_process(cortex, image, 640, 480, 1, features);
 *
 * visual_memory_t** memories = NULL;
 * int num_memories = 0;
 * visual_cortex_recall_memory(cortex, features, 5, &memories, &num_memories);
 *
 * for (int i = 0; i < num_memories; i++) {
 *     printf("Similar memory with salience: %.2f\n", memories[i]->salience);
 * }
 * nimcp_free(memories);
 * ```
 */
bool visual_cortex_recall_memory(
    visual_cortex_t* cortex,
    const float* query_features,
    int max_results,
    visual_memory_t*** memories,
    int* num_memories
);

/**
 * WHAT: Get statistics about visual processing
 * WHY:  Monitor performance and resource usage
 * HOW:  Return internal metrics
 */
typedef struct {
    uint32_t images_processed;   /**< Total images processed */
    uint32_t memories_stored;    /**< Visual memories stored */
    float avg_processing_time;   /**< Average processing time (ms) */
    float memory_usage_mb;       /**< Memory usage in MB */
} visual_cortex_stats_t;

bool visual_cortex_get_stats(const visual_cortex_t* cortex, visual_cortex_stats_t* stats);

//=============================================================================
// Brain Integration Helpers
//=============================================================================

/**
 * WHAT: Compute novelty score for curiosity system
 * WHY:  Drive exploration of novel visual patterns
 * HOW:  Compare features against visual memory, return novelty (0-1)
 *
 * @param cortex Visual cortex
 * @param features Feature vector from visual_cortex_process()
 * @return Novelty score (0=familiar, 1=novel), -1 on error
 *
 * BRAIN INTEGRATION:
 * Visual Cortex → Curiosity System → Exploration behavior
 *
 * USAGE:
 * ```c
 * float features[128];
 * visual_cortex_process(cortex, image, w, h, 1, features);
 * float novelty = visual_cortex_compute_novelty(cortex, features);
 * if (novelty > 0.7) {
 *     // Trigger curiosity-driven exploration
 * }
 * ```
 */
float visual_cortex_compute_novelty(visual_cortex_t* cortex, const float* features);

/**
 * WHAT: Get maximum attention location
 * WHY:  Identify most salient region for attention system
 * HOW:  Find peak in attention map
 *
 * @param attn_map Attention map from visual_cortex_compute_attention()
 * @param max_x Output: X coordinate of peak attention
 * @param max_y Output: Y coordinate of peak attention
 * @param max_value Output: Attention value at peak
 * @return true on success, false on error
 *
 * BRAIN INTEGRATION:
 * Visual Cortex → Attention System → Salience-driven focus
 *
 * USAGE:
 * ```c
 * attention_map_t* map = attention_map_create(640, 480);
 * visual_cortex_compute_attention(cortex, image, 640, 480, map);
 * uint32_t focus_x, focus_y;
 * float salience;
 * visual_cortex_get_attention_peak(map, &focus_x, &focus_y, &salience);
 * // Focus camera/processing on (focus_x, focus_y)
 * ```
 */
bool visual_cortex_get_attention_peak(
    const attention_map_t* attn_map,
    uint32_t* max_x,
    uint32_t* max_y,
    float* max_value
);

/**
 * WHAT: Prepare visual features for memory consolidation
 * WHY:  Enable hippocampus to store visual experiences
 * HOW:  Package features with metadata for memory system
 *
 * @param cortex Visual cortex
 * @param features Feature vector
 * @param salience Importance score (0-1) for consolidation priority
 * @param context Optional context string (e.g., "saw red ball")
 * @return true on success, false on error
 *
 * BRAIN INTEGRATION:
 * Visual Cortex → Hippocampus → Long-term visual memory
 *
 * USAGE:
 * ```c
 * float features[128];
 * visual_cortex_process(cortex, image, w, h, 1, features);
 * float salience = visual_cortex_compute_novelty(cortex, features);
 * visual_cortex_consolidate_memory(cortex, features, salience, "first time seeing cat");
 * // Later: Hippocampus consolidates during "sleep"
 * ```
 */
bool visual_cortex_consolidate_memory(
    visual_cortex_t* cortex,
    const float* features,
    float salience,
    const char* context
);

/**
 * WHAT: Associate brain with visual cortex for neuromodulation
 * WHY:  Enable ACh + NE modulation of visual processing
 * HOW:  Store brain reference for neurotransmitter reading
 *
 * @param cortex Visual cortex instance
 * @param brain Brain instance (or NULL to clear)
 *
 * BIOLOGY:
 * - Acetylcholine from basal forebrain enhances V1 attention
 * - Norepinephrine from locus coeruleus increases arousal-dependent vision
 *
 * CLINICAL EXAMPLES:
 * - ADHD (low ACh): Reduced visual attention, misses cues
 * - Anxiety (high NE): Hypervigilant vision, sees threats everywhere
 */
void visual_cortex_set_brain(visual_cortex_t* cortex, brain_t brain);

//=============================================================================
// Neuromodulation API
//=============================================================================

/**
 * @brief Get phasic/tonic state for a neuromodulator
 *
 * WHAT: Query current phasic and tonic levels
 * WHY:  Enable monitoring and debugging of neuromodulation
 * HOW:  Return pointer to internal state
 *
 * @param cortex Visual cortex instance
 * @param type Neuromodulator type (dopamine, acetylcholine, norepinephrine)
 * @return Phasic/tonic state or NULL on error
 *
 * USAGE:
 * ```c
 * phasic_tonic_state_t* ach_state =
 *     visual_cortex_get_neuromod_state(cortex, 0);  // 0=dopamine, 1=ach, 2=ne
 * printf("ACh: phasic=%.2f, tonic=%.2f\n",
 *        ach_state->phasic_level, ach_state->tonic_level);
 * ```
 */
const phasic_tonic_state_t* visual_cortex_get_neuromod_state(
    const visual_cortex_t* cortex,
    uint32_t neuromod_type);

/**
 * @brief Set receptor expression profile for visual cortex
 *
 * WHAT: Configure receptor densities for different V1 layers
 * WHY:  Model layer-specific neuromodulation sensitivity
 * HOW:  Store receptor expression array
 *
 * @param cortex Visual cortex instance
 * @param layer_idx Layer index (0=layer 2/3, 1=layer 4, 2=layer 5, 3=layer 6)
 * @param receptors Receptor expression profile
 * @return true on success
 *
 * BIOLOGY:
 * Layer 2/3: High D1 (0.4), moderate ACh (0.5) - top-down attention
 * Layer 4:   High ACh (0.6), moderate NE (0.5) - thalamic input
 * Layer 5:   High D1 (0.4), D2 (0.3), NE (0.5) - motor output
 * Layer 6:   High M1 (0.6), β2 (0.4) - feedback modulation
 */
bool visual_cortex_set_receptor_profile(
    visual_cortex_t* cortex,
    uint32_t layer_idx,
    const receptor_expression_t* receptors);

/**
 * @brief Get receptor expression profile for a layer
 *
 * @param cortex Visual cortex instance
 * @param layer_idx Layer index (0-3)
 * @return Receptor profile or NULL on error
 */
const receptor_expression_t* visual_cortex_get_receptor_profile(
    const visual_cortex_t* cortex,
    uint32_t layer_idx);

/**
 * @brief Trigger phasic neuromodulator burst
 *
 * WHAT: Inject rapid neuromodulator pulse (e.g., reward, surprise)
 * WHY:  Simulate phasic bursts from midbrain/brainstem
 * HOW:  Add to phasic level, which decays rapidly
 *
 * @param cortex Visual cortex instance
 * @param neuromod_type 0=dopamine, 1=acetylcholine, 2=norepinephrine
 * @param amount Burst amplitude (0-1), typically 0.3-0.8
 * @return true on success
 *
 * USAGE:
 * ```c
 * // Reward → dopamine burst
 * visual_cortex_trigger_phasic_burst(cortex, 0, 0.5);
 *
 * // Surprising stimulus → ACh burst
 * visual_cortex_trigger_phasic_burst(cortex, 1, 0.6);
 *
 * // Threat detected → NE burst
 * visual_cortex_trigger_phasic_burst(cortex, 2, 0.7);
 * ```
 */
bool visual_cortex_trigger_phasic_burst(
    visual_cortex_t* cortex,
    uint32_t neuromod_type,
    float amount);

/**
 * @brief Set tonic neuromodulator baseline
 *
 * WHAT: Set slow baseline level (mood, arousal, motivation)
 * WHY:  Model sustained states (stress, focus, motivation)
 * HOW:  Set tonic level directly
 *
 * @param cortex Visual cortex instance
 * @param neuromod_type 0=dopamine, 1=acetylcholine, 2=norepinephrine
 * @param level Tonic baseline (0-1)
 * @return true on success
 *
 * CLINICAL EXAMPLES:
 * - Depression: Low dopamine tonic (0.2), normal phasic bursts
 * - ADHD: Low ACh tonic (0.3), impaired sustained attention
 * - Anxiety: High NE tonic (0.7), chronic hypervigilance
 */
bool visual_cortex_set_tonic_level(
    visual_cortex_t* cortex,
    uint32_t neuromod_type,
    float level);

/**
 * @brief Compute current neuromodulation effects
 *
 * WHAT: Calculate final gain/modulation values
 * WHY:  Consolidate all neuromodulation into single effect struct
 * HOW:  Combine phasic+tonic levels × receptor expression
 *
 * @param cortex Visual cortex instance
 * @param layer_idx Layer to compute effects for (0-3)
 * @param effects Output: computed neuromodulation effects
 * @return true on success
 *
 * ALGORITHM:
 * 1. Read phasic + tonic levels from brain neuromodulator system
 * 2. Combine: effective_level = α*phasic + (1-α)*tonic, α=0.6
 * 3. Multiply by receptor densities:
 *    - DA effect = (D1 - 0.5*D2) * dopamine_level
 *    - ACh effect = (M1 - 0.3*M2) * ach_level
 *    - NE effect = (α1 + 0.5*β2) * ne_level
 * 4. Compute gains:
 *    - gabor_gain = 1.0 + 0.5*DA + 0.3*ACh + 0.4*NE
 *    - attention_boost = 1.0 + 0.7*ACh + 0.3*NE
 *    - plasticity_gate = sigmoid(2*DA + ACh)
 *    - contrast_gain = 1.0 + 0.4*DA + 0.2*ACh
 */
bool visual_cortex_compute_neuromod_effects(
    const visual_cortex_t* cortex,
    uint32_t layer_idx,
    neuromod_effects_t* effects);

//=============================================================================
// Bidirectional Feedback Functions (Phase 10.11.3)
//=============================================================================

/**
 * @brief Boost attention to specific visual region
 *
 * WHAT: Increase processing sensitivity for spatial region
 * WHY:  Social cues (faces/agents) should receive enhanced processing
 * HOW:  Scale feature activations in target region
 *
 * BIOLOGY: STS modulates V1 for social stimuli via feedback projections
 *
 * @param cortex Visual cortex instance
 * @param region_x X coordinate of region center (normalized [0,1])
 * @param region_y Y coordinate of region center (normalized [0,1])
 * @param boost_factor Attention boost [1.0, 2.0]
 */
void visual_cortex_boost_region_attention(visual_cortex_t* cortex,
                                           float region_x,
                                           float region_y,
                                           float boost_factor);

/**
 * @brief Detect if agent/person present in visual field
 *
 * WHAT: Simple heuristic agent detection
 * WHY:  Triggers mirror neuron observation mode
 * HOW:  Check for motion + face-like patterns
 *
 * @param cortex Visual cortex instance
 * @param features Feature vector from recent processing
 * @param num_features Number of features
 * @return true if agent detected
 */
bool visual_cortex_detect_agent(visual_cortex_t* cortex,
                                 const float* features,
                                 uint32_t num_features);

#ifdef __cplusplus
}
#endif

#endif /* NIMCP_VISUAL_CORTEX_H */
